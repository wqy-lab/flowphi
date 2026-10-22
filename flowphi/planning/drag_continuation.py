"""Keep an existing physical DOWN through a nearby Drag, without rewriting its prefix."""
from __future__ import annotations

from collections import OrderedDict
import math

from flowphi.charts.model import NoteType
from flowphi.motion import point_at, world_xy
from flowphi.planning.contacts import ContactGeometry, refine_contact, validate_contact


# Search horizon, not a game judgement window. Long idle gaps remain separate.
MAX_IDLE_GAP = .35
TAIL_CACHE_LIMIT=512


def _refined_tail(planner,geometry,tail):
    """Exact geometry memo shared by different finger/history assignments."""
    cache=getattr(planner,'_drag_tail_cache',None)
    if cache is None:cache=planner._drag_tail_cache=OrderedDict()
    key=(geometry,tail['start'],tail['end'],tuple(tail['note_ids']),
         tuple(map(tuple,tail['points'])),tuple(sorted(tail.get('judgement_times',{}).items())),
         repr({k:tail.get(k) for k in ('joint_note_coverage','joint_note_windows','collective_hold',
             'refinement_region','block_clearance_m','refinement_start','refinement_end')}))
    if key in cache:
        cache.move_to_end(key)
        return cache[key]
    extra=set();points=None
    for _ in range(3):
        trial,_=refine_contact(geometry,tail,extra_times=extra)
        if trial is None:break
        bad=validate_contact(geometry,tail,trial,limit=20)
        if not bad:points=trial;break
        extra.update(bad)
    cache[key]=points
    if len(cache)>TAIL_CACHE_LIMIT:cache.popitem(last=False)
    return points


def _window_valid(planner, geometry, contact, points):
    """Memoize the same millisecond checks, scoped to relevant note windows.

    Planner trajectories are immutable: branches replace lists when extending
    or refining them. Retain the list with the cached verdict to prevent id reuse.
    Unusual authored geometry options use the general validator unchanged.
    """
    if any(key in contact for key in ('collective_hold', 'refinement_region',
                                      'joint_note_windows', 'block_clearance_m')):
        return not validate_contact(geometry, contact, points, limit=1)
    low, high = contact['start'], contact['end']
    ids = []
    for nid in contact['note_ids']:
        note = geometry.notes[nid][1]
        beat = geometry.judgement_time(contact, nid)
        end = note.seconds + note.hold if note.type == NoteType.HOLD else beat + .011
        # Include edge rounding as well as the validator's fractional Hold samples.
        start = note.seconds if note.type == NoteType.HOLD else beat
        if start - .002 <= high and end + .002 >= low:
            ids.append(nid)
    timing = {str(nid): contact['judgement_times'][str(nid)] for nid in ids
              if str(nid) in contact.get('judgement_times', {})}
    key = (id(points), low, high, tuple(ids), tuple(timing.items()),
           bool(contact.get('joint_note_coverage')))
    cache = getattr(planner, '_drag_window_cache', None)
    if cache is None:
        cache = planner._drag_window_cache = OrderedDict()
    cached = cache.get(key)
    if cached is not None:
        cache.move_to_end(key)
        return cached[1]
    window = dict(contact, note_ids=ids, judgement_times=timing)
    good = not validate_contact(geometry, window, points, limit=1)
    cache[key] = (points, good)
    if len(cache) > 2048:
        cache.popitem(last=False)
    return good


def drag_continuations(planner, task, state):
    from flowphi.planning.planner import State

    if task.note.type != NoteType.DRAG:
        return []
    if getattr(planner, '_drag_geometry', None) is None:
        planner._drag_geometry = ContactGeometry(planner.chart, physical_screen=planner.physical,
                                                 extra_clearance_m=.004, cache_judgement=True)
    geometry = planner._drag_geometry
    beat = task.beat / 1000
    latest = {}
    for index, contact in enumerate(state.contacts):
        if contact['start'] <= beat:
            latest[contact['hand'], contact['finger']] = index
    result = []
    for index in latest.values():
        old = state.contacts[index]
        if task.id in old['note_ids'] or beat - old['end'] > MAX_IDLE_GAP:
            continue
        # Copy on write: failed candidates do not clone a long path/history.
        merged = dict(old)
        merged.update(note_ids=[*old['note_ids'], task.id], joint_note_coverage=True,
                      end=max(old['end'], (task.beat + 13) / 1000),
                      shared_contact='continued_drag')
        if old['kind'] == 'tap':
            merged['kind'] = 'drag'
        merged['judgement_times'] = dict(old.get('judgement_times', {}))
        merged['judgement_times'].update(planner.timing_metadata(task).get('judgement_times', {}))
        changed_from = old['end']
        if merged['end'] > old['end']:
            merged.pop('terminal_hold_release_ms', None)
            merged.pop('terminal_hold_release_reason', None)
            merged['points'] = [*old['points'], [merged['end'] - .001, *old['points'][-1][1:]]]

        # Only the new interval/windows need rechecking: all earlier samples
        # are immutable, preserving Tap/Hold heads and the original Flick stroke.
        low = min(old['end'] - .001, task.note.seconds - .0011)
        low = max(old['start'], low)
        high = merged['end'] if merged['end'] > old['end'] else min(old['end'], task.note.seconds + .0111)
        window = dict(merged, start=low, end=high)
        good = _window_valid(planner, geometry, window, merged['points'])

        if not good:
            # Never pull an active Hold or unfinished Flick away from its
            # existing trajectory. A finished action can slide into the Drag.
            if old['end'] > beat + 1e-9:
                continue
            start = round(old['end'] - .001, 3)
            # A Tap's chosen UP/dwell is not a mandatory stationary Hold.
            # Within an uncommitted physical-search window, compare starting
            # the continuous movement after its judgement guard instead of
            # freezing the entire dwell and rushing into the next Drag.
            originals=[planner.original_tasks[n] for n in old['note_ids']]
            if (getattr(planner,'_physical_spatial_policy',False) and
                    old['start']>=getattr(planner,'_physical_window_start',float('inf')) and
                    all(t.note.type in (NoteType.TAP,NoteType.DRAG) for t in originals)):
                heads=[t for t in originals if t.note.type==NoteType.TAP]
                if len(heads)==1:
                    guard=max(old['start'],geometry.judgement_time(old,heads[0].id)+.012)
                    start=min(start,math.ceil((guard-1e-9)*1000)/1000)
            anchor = point_at(old['points'], start)
            tail = dict(merged, kind='drag', start=start,
                        refinement_start=list(anchor),
                        points=[[start, *anchor], [merged['end'] - .001, *anchor]])
            points = _refined_tail(planner,geometry,tail)
            if points is None:
                continue
            # The endpoint constraint makes the join continuous. Prefix points,
            # including every original Flick sample, stay bit-for-bit identical.
            prefix=[p for p in old['points'] if p[0]<start]
            merged['points'] = [*prefix,*points]
            changed_from = start
            window = dict(merged, start=start)
            if validate_contact(geometry, window, merged['points'], limit=1):
                continue

        if planner.blocks.path_violations(merged['points'], window['start'], window['end']):
            continue
        others = tuple(c for i, c in enumerate(state.contacts) if i != index)
        # A live touch can collect a Drag by metadata alone. Its physical path
        # and lifetime are identical, so inherited posture warnings cannot be
        # a reason to introduce an extra finger. Keep those warnings intact.
        unchanged_motion = merged['end'] == old['end'] and merged['points'] == old['points']
        if not unchanged_motion:
            # Only the newly occupied/moving suffix adds physical constraints.
            # It continues a DOWN, so there is no new lifted-finger transfer.
            suffix = dict(merged, start=changed_from, points=[
                [changed_from, *point_at(merged['points'], changed_from)],
                *[p for p in merged['points'] if p[0] > changed_from]])
            side = -1 if old['hand'] == 'left' else 1
            violations, _, _ = planner.constraints(suffix, State(others), side, old['finger'],
                                                     continuation=True)
            if violations:
                continue

        def travel(points):
            return sum(math.dist(world_xy(a[1:], planner.physical), world_xy(b[1:], planner.physical))
                       for a, b in zip(points, points[1:]))
        added_travel = 0. if unchanged_motion else max(0., travel(merged['points']) - travel(old['points']))
        increment = .15 * (added_travel / .1) ** 2 + .05 * max(0., merged['end'] - old['end'])
        previous_decision = old.get('decision', {})
        merged['decision'] = dict(previous_decision,
            drag_continuations=[*previous_decision.get('drag_continuations', ()), task.id],
            costs=dict(previous_decision.get('costs', {})))
        costs = merged['decision']['costs']
        costs['continuation'] = costs.get('continuation', 0.) + increment
        # A continuation changes the whole physical lifetime; only charge the
        # difference from the previous silhouette, never a fresh DOWN's cost.
        if planner.settings.occlusion_weight:
            from flowphi.planning.occlusion import scorer
            visibility = scorer(planner)
            occlusion = visibility.contact(merged)
            delta = planner.settings.occlusion_weight*(occlusion-visibility.contact(old))
            increment += delta
            costs['occlusion'] = planner.settings.occlusion_weight*occlusion
            merged['decision']['occlusion'] = dict(weighted_note_seconds=occlusion,
                model='projected_contact_proxy', lookahead_seconds=.65)
        if planner.settings.hand_interaction_weight:
            from flowphi.planning.interaction import contact_relation
            others = tuple(c for i, c in enumerate(state.contacts) if i != index)
            interaction = contact_relation(planner, merged, others)
            before = contact_relation(planner, old, others)
            increment += planner.settings.hand_interaction_weight*(interaction['cost']-before['cost'])
            costs['interaction'] = planner.settings.hand_interaction_weight*interaction['cost']
            merged['decision']['interaction'] = interaction
        contacts = list(state.contacts)
        contacts[index] = merged
        if planner.contact_occupancy_valid(contacts):
            result.append(State(tuple(contacts), state.score + increment))
    return result
