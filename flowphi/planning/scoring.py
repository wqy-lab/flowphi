"""Shared physical-contact costs for ordinary and phrase planning."""
import cmath
import math
from collections import OrderedDict

from flowphi.charts.model import NoteType
from flowphi.motion import world_xy


def line_distance_cost(planner, record):
    """Soft normal-distance preference; all legal vertical judgement survives.

    Sample the whole sustained path. Flicks are measured at their head only:
    their mandatory swipe must never be priced as unnecessary travel.
    """
    tasks=[planner.original_tasks[n] for n in record['note_ids']]
    a,b=record['start'],record['end']
    count=min(16,max(1,math.ceil((b-a)/.1)))
    times=[a+(b-a)*(i+.5)/count for i in range(count)]
    if record['kind']=='flick':
        times=[a]
        holds=[task for task in tasks if task.note.type==NoteType.HOLD]
        tail=max((task.end/1000 for task in tasks if task.note.type==NoteType.FLICK),default=b)
        if holds and b>tail:
            times.extend(tail+(b-tail)*(i+.5)/count for i in range(count))
    from flowphi.motion import point_at
    result=0.
    for t in times:
        relevant=[task for task in tasks if task.note.seconds<=t+.035] or tasks[:1]
        held=[task for task in relevant if task.note.type==NoteType.HOLD and
              task.note.seconds<=t<=task.note.seconds+task.note.hold]
        task=min(held or relevant,key=lambda task:abs(task.note.seconds-t))
        when=t if task.note.type==NoteType.HOLD else task.note.seconds
        center=task.line.pos(when,task.note.offset)
        x,y=point_at(record['points'],t)
        relative=(complex(x*planner.screen.width,y*planner.screen.height)-center)*cmath.exp(-1j*(task.line.angle @ when))
        distance=abs(relative.imag)*planner.physical[0]/planner.screen.width
        result+=(max(0.,distance-.012)/.035)**2
    return result/len(times)*max(.08,b-a)


def score_contact(planner, record, history, movement, posture, violations):
    from flowphi.planning.planner import (CROSS_RAMP_SECONDS, CROSS_UNIT, CROSS_WEIGHT,
                                          COMFORT, PREFERENCE, load_at)
    settings = planner.settings
    hand, finger, t = record['hand'], record['finger'], record['start']
    previous = next((c for c in reversed(history) if (c['hand'], c['finger']) == (hand, finger)), None)
    source = min((planner.original_tasks[n] for n in record['note_ids']), key=lambda task: task.start)
    kind = source.note.type
    f, b = planner.load_at(history,t,hand,finger)
    hf, hb = planner.load_at(history,t,hand)
    cps = 1 / max(.001, t-previous['start']) if previous else 0.
    comfort = COMFORT[finger] * (1-.3*f-.15*hf)
    record['burst_cost'] = .1 * max(0., cps/comfort-1) if kind == NoteType.TAP else 0.
    record['effort'] = .018 if kind == NoteType.DRAG else .055
    if settings.fingering_objective != 'load':
        record['effort'] += .02*min(3., math.sqrt(movement))
    af, ab = planner.load_at(history,record['end'],hand,finger)
    ownf, ownb = load_at((record,), record['end'], hand, finger, settings)
    af, ab = min(1., af+ownf), max(0., ab-(1-ownb))
    x, y = record['points'][0][1:]
    center = source.line.pos(source.note.seconds, source.note.offset)
    along = ((complex(x*planner.screen.width, y*planner.screen.height)-center) *
             cmath.exp(-1j*(source.line.angle @ source.note.seconds))).real
    side = -1 if hand == 'left' else 1
    habit = .08 if previous and kind == NoteType.TAP and cps < 2 else 0.
    # A real continuous touch already saves work. Do not reward repeated Drag
    # DOWNs merely because a later postprocessor might join them.
    travel = sum(math.dist(world_xy(a[1:], planner.physical), world_xy(b[1:], planner.physical))
                 for a, b in zip(record['points'], record['points'][1:]))
    # Crossed hands are rare in Phigros: price the wrist-order inversion instead
    # of relying on fingertip distance alone, which a sustained crossing avoids.
    excess, crossed_seconds, crossed_at = planner.hand_crossing(record, history)
    crossing = 0. if not excess else CROSS_WEIGHT * (excess / CROSS_UNIT) * (
        1. + crossed_seconds / CROSS_RAMP_SECONDS)
    from flowphi.planning.occlusion import scorer
    occlusion = scorer(planner).contact(record) if settings.occlusion_weight else 0.
    from flowphi.planning.interaction import contact_relation
    interaction = contact_relation(planner, record, history) if settings.hand_interaction_weight else dict(cost=0.)
    costs = dict(preference=PREFERENCE[finger], fatigue=.8*f+1.5*max(0., af-f),
        burst=2*record['burst_cost']+.2*(1-ab), hand=.6*hf+.3*(1-hb),
        movement=.35*movement + (.15*(travel/.1)**2 if any(
            planner.original_tasks[n].note.type == NoteType.DRAG for n in record['note_ids']) else 0.),
        posture=.8*posture, visual=(along/planner.screen.width)**2*80,
        side=max(0., -.03-side*(x-.5))**2*12, habit=-habit, crossing=crossing,
        occlusion=settings.occlusion_weight*occlusion,
        line_distance=settings.judge_line_weight*line_distance_cost(planner,record),
        interaction=settings.hand_interaction_weight*interaction['cost'], degraded=1000.*len(violations))
    if settings.fingering_objective == 'load':
        for name in ('preference', 'movement', 'posture', 'visual', 'side', 'habit', 'crossing', 'degraded'):
            costs[name] = 0.
    record['decision'] = dict(costs=costs, fatigue_before=f, fatigue_after=af,
        burst_before=b, burst_after=ab, hand_fatigue=hf, finger_cps=cps, degraded=list(violations),
        crossing=dict(excess_m=excess, seconds=crossed_seconds, at=crossed_at) if excess else None)
    if getattr(planner,'_speculative_transfers',False):
        pending=set(violations)&{'finger_transfer','wrist_transfer'}
        if getattr(planner,'_physical_plain_search',False):
            pending |= set(violations)&{'burst_capacity'}
        if pending:
            record['decision']['pending_transfer_verification']=sorted(pending)
            costs['degraded']=1000.*len(set(violations)-pending)
    record['decision']['occlusion'] = dict(weighted_note_seconds=occlusion,
        model='projected_contact_proxy', lookahead_seconds=.65)
    record['decision']['interaction'] = interaction
    return sum(costs.values())


def score_history(planner, contacts):
    """Replay every candidate at the same frontier using exactly one scorer."""
    from flowphi.planning.planner import State
    cache = getattr(planner, '_phrase_score_cache', None)
    if cache is None:
        cache = planner._phrase_score_cache = OrderedDict()
        planner._phrase_score_serial = 0
    history, total, token = [], 0., 0
    for original in sorted(contacts, key=lambda c: (c['start'], c['pointer'])):
        record = dict(original)
        # Immutable physical prefixes are shared by thousands of alternatives.
        # Metadata (including strategy provenance) is copied from this branch,
        # never from the cached branch. Cached points retain identity ownership.
        key = (bool(getattr(planner,'_speculative_transfers',False)),
               bool(getattr(planner,'_physical_plain_search',False)),token,
               record['hand'], record['finger'], record['start'], record['end'],
               record['kind'], tuple(record['note_ids']), id(record['points']))
        cached = cache.get(key)
        if cached is not None:
            cache.move_to_end(key)
            token, total, numerical = cached
            for name in ('effort', 'burst_cost', 'decision'):
                record[name] = numerical[name]
        else:
            errors, movement, posture = planner.constraints(record, State(tuple(history)),
                -1 if record['hand'] == 'left' else 1, record['finger'])
            total += score_contact(planner, record, history, movement, posture, errors)
            planner._phrase_score_serial += 1
            token = planner._phrase_score_serial
            cache[key] = (token, total, record)
            if len(cache) > 4096:
                cache.popitem(last=False)
        history.append(record)
    # Physical window costs are already paid by earlier decisions. Replaying
    # contact costs must not silently discard them on the next expansion.
    motion_costs = {d['window']: d['hand_motion']['cost'] for c in history
                    for d in c.get('ribbon_decisions', ()) if d.get('hand_motion')}
    ordinary_costs = {d['window']:d['hand_motion']['cost'] for c in history
                      for d in c.get('ordinary_decisions',())}
    return State(tuple(history), total+sum(motion_costs.values())+sum(ordinary_costs.values()))
