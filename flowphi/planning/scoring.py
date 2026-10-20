"""Shared physical-contact costs for ordinary and phrase planning."""
import cmath
import math
from collections import OrderedDict

from flowphi.charts.model import NoteType
from flowphi.motion import world_xy


def score_contact(planner, record, history, movement, posture, violations):
    from flowphi.planning.planner import COMFORT, PREFERENCE, load_at
    settings = planner.settings
    hand, finger, t = record['hand'], record['finger'], record['start']
    previous = next((c for c in reversed(history) if (c['hand'], c['finger']) == (hand, finger)), None)
    source = min((planner.original_tasks[n] for n in record['note_ids']), key=lambda task: task.start)
    kind = source.note.type
    f, b = load_at(history, t, hand, finger, settings)
    hf, hb = load_at(history, t, hand, settings=settings)
    cps = 1 / max(.001, t-previous['start']) if previous else 0.
    comfort = COMFORT[finger] * (1-.3*f-.15*hf)
    record['burst_cost'] = .1 * max(0., cps/comfort-1) if kind == NoteType.TAP else 0.
    record['effort'] = .018 if kind == NoteType.DRAG else .055
    if settings.fingering_objective != 'load':
        record['effort'] += .02*min(3., math.sqrt(movement))
    af, ab = load_at(history, record['end'], hand, finger, settings)
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
    costs = dict(preference=PREFERENCE[finger], fatigue=.8*f+1.5*max(0., af-f),
        burst=2*record['burst_cost']+.2*(1-ab), hand=.6*hf+.3*(1-hb),
        movement=.35*movement + (.15*(travel/.1)**2 if any(
            planner.original_tasks[n].note.type == NoteType.DRAG for n in record['note_ids']) else 0.),
        posture=.8*posture, visual=(along/planner.screen.width)**2*80,
        side=max(0., -.03-side*(x-.5))**2*12, habit=-habit, degraded=1000.*len(violations))
    if settings.fingering_objective == 'load':
        for name in ('preference', 'movement', 'posture', 'visual', 'side', 'habit', 'degraded'):
            costs[name] = 0.
    record['decision'] = dict(costs=costs, fatigue_before=f, fatigue_after=af,
        burst_before=b, burst_after=ab, hand_fatigue=hf, finger_cps=cps, degraded=list(violations))
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
        key = (token, record['hand'], record['finger'], record['start'], record['end'],
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
    return State(tuple(history), total)
