"""Bounded whole-window search for dense Drag phrases, including mixed heads.

All strategies finish the same tasks before ranking. No global swept-note flag
is used: a window is consumed only after all retained branches have finished it.
"""
from collections import Counter, OrderedDict
from dataclasses import dataclass, replace
import copy
import math
import time

from shapely.affinity import scale

from flowphi.charts.model import NoteType
from flowphi.motion import FINGER_ORDER, point_at, world_xy, travel_time
from flowphi.planning.contacts import ContactGeometry, project, refine_contact, validate_contact
from flowphi.planning.drag_continuation import _window_valid, MAX_IDLE_GAP
from flowphi.planning.scoring import score_history
from flowphi.planning.timing import PLANNING_WINDOW_MS

LOCAL_WIDTH = 32
FAMILY_LIMIT = 16
MAX_TRACKS = 4
MAX_GROUPS = 8
MAX_CACHE = 512
FAMILIES = ('continuous', 'relay', 'stationary')


@dataclass(frozen=True)
class RibbonWindow:
    start: int
    stop: int
    drag_ids: tuple


def _geometry(planner):
    if getattr(planner, '_ribbon_geometry', None) is None:
        planner._ribbon_geometry = ContactGeometry(planner.chart, physical_screen=planner.physical,
                                                    extra_clearance_m=.004, cache_judgement=True)
    return planner._ribbon_geometry


def detect_windows(planner):
    """Connect legal physical bands, not line IDs or visible-note distances."""
    tasks = planner.tasks
    drags = [t for t in tasks if t.note.type == NoteType.DRAG]
    parents = list(range(len(drags)))
    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    zones = [scale(planner.zone(t, t.beat), xfact=planner.physical[0]/planner.screen.width,
                   yfact=planner.physical[1]/planner.screen.height, origin=(0, 0)) for t in drags]
    for i, task in enumerate(drags):
        for j in range(i-1, -1, -1):
            dt = (task.beat-drags[j].beat)/1000
            if dt > .120001:
                break
            if not zones[i].is_empty and not zones[j].is_empty and zones[i].distance(zones[j]) <= planner.settings.finger_speed*dt+1e-9:
                parents[root(i)] = root(j)
    components = {}
    for i, task in enumerate(drags):
        components.setdefault(root(i), []).append(task)
    intervals = []
    for group in components.values():
        if len(group) >= 4:
            # Include nearby heads and Flick preparation on both boundaries.
            intervals.append((min(t.start for t in group)-120, max(t.start for t in group)+120))
    merged = []
    for low, high in sorted(intervals):
        if merged and low <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(high, merged[-1][1]))
        else:
            merged.append((low, high))
    windows = []
    for low, high in merged:
        indices = [i for i, task in enumerate(tasks) if low <= task.start <= high]
        if not indices:
            continue
        begin, stop = indices[0], indices[-1]+1
        while begin < stop:
            end = begin+1
            while end < stop and tasks[end].start-tasks[begin].start < 2000:
                end += 1
            # Equal-start tasks are never split across windows.
            while end < stop and tasks[end].start == tasks[end-1].start:
                end += 1
            ids = tuple(t.id for t in tasks[begin:end] if t.note.type == NoteType.DRAG)
            if ids:
                windows.append(RibbonWindow(begin, end, ids))
            begin = end
    return windows


def _tracks(planner, tasks):
    tracks = []
    for task in sorted(tasks, key=lambda t: (t.beat, t.id)):
        zone = planner.zone(task, task.beat)
        options = []
        for i, track in enumerate(tracks):
            last = track[-1]
            dt = (task.beat-last.beat)/1000
            previous = planner.zone(last, last.beat)
            distance = previous.distance(zone)*planner.physical[0]/planner.screen.width
            if dt < 0 or dt > .120001 or distance > planner.settings.finger_speed*dt+1e-9:
                continue
            if dt == 0:
                common = zone
                for item in reversed(track):
                    if item.beat != task.beat:
                        break
                    common = common.intersection(planner.zone(item, item.beat))
                if common.is_empty:
                    continue
            center_distance = abs(task.line.pos(task.note.seconds, task.note.offset)-
                                  last.line.pos(last.note.seconds, last.note.offset))
            options.append((distance, center_distance, i))
        if options:
            tracks[min(options)[2]].append(task)
        else:
            tracks.append([task])
    return tracks


def _groups(planner, drags, family):
    tracks = _tracks(planner, drags)
    if family == 'continuous':
        return tracks if len(tracks) <= MAX_TRACKS else []
    if family == 'relay':
        groups = []
        for track in tracks:
            split = len(track)//2
            turns = []
            for i in range(1, len(track)-1):
                positions = [t.line.pos(t.note.seconds, t.note.offset) for t in track[i-1:i+2]]
                a, b = positions[1]-positions[0], positions[2]-positions[1]
                if abs(a) > 1e-6 and abs(b) > 1e-6:
                    turns.append((((a.conjugate()*b).real)/(abs(a)*abs(b)), i+1))
            if turns and min(turns)[0] < 0:
                split = min(turns)[1]
            if len(track) >= 4:
                groups.extend((track[:split], track[split:]))
            else:
                groups.append(track)
        return groups if len(groups) <= MAX_GROUPS else []
    # A stationary finger may cover different lines and time-disjoint branches.
    groups, regions = [], []
    for task in sorted(drags, key=lambda t: (t.beat, t.id)):
        zone = planner.zone(task, task.beat)
        possible = [(regions[i].intersection(zone).area, i) for i in range(len(groups))
                    if not regions[i].intersection(zone).is_empty]
        if possible:
            _, i = max(possible)
            groups[i].append(task)
            regions[i] = regions[i].intersection(zone)
        else:
            groups.append([task])
            regions.append(zone)
    return groups if len(groups) <= min(MAX_GROUPS, len(planner.keys)) else []


def _group_variants(planner, tasks, family):
    drags = [t for t in tasks if t.note.type == NoteType.DRAG]
    primary = _groups(planner, drags, family)
    if primary:
        yield primary
    if family != 'relay':
        return
    tracks = _tracks(planner, drags)
    heads = [t for t in tasks if t.note.type != NoteType.DRAG]
    # An obstacle gap or an interleaved head can be a better handoff boundary
    # than the geometric midpoint. Try each policy as a whole-window variant.
    for policy in ('blocks', 'heads'):
        groups = []
        for track in tracks:
            cuts = [0]
            for i in range(1, len(track)):
                a, b = track[i-1].beat/1000, track[i].beat/1000
                event = (any(a < t < b for t in planner.blocks.key_times) if policy == 'blocks' else
                         any(a < h.beat/1000 <= b for h in heads))
                if event:
                    cuts.append(i)
            cuts.append(len(track))
            groups.extend(track[a:b] for a, b in zip(cuts, cuts[1:]) if a < b)
        if 1 < len(groups) <= MAX_GROUPS and groups != primary:
            yield groups
    groups = []
    for track in tracks:
        current = []
        for task in track:
            if current and (task.beat-current[0].beat > 120 or task.beat-current[-1].beat > 50):
                groups.append(current)
                current = []
            current.append(task)
        if current:
            groups.append(current)
    if 1 < len(groups) <= MAX_GROUPS and groups != primary:
        yield groups
    # Short relays remain useful on straight fast runs: a geometric midpoint
    # alone may leave a whole half incompatible with the currently occupied hand.
    for size in (3, 2):
        groups = [track[i:i+size] for track in tracks for i in range(0, len(track), size)]
        if 1 < len(groups) <= MAX_GROUPS and groups != primary:
            yield groups


def _route(planner, tasks, key, family, seed=None):
    cache = getattr(planner, '_ribbon_route_cache', None)
    if cache is None:
        cache = planner._ribbon_route_cache = OrderedDict()
    cache_key = (tuple(t.id for t in tasks), key, family == 'stationary', tuple(seed) if seed is not None else None)
    if cache_key in cache:
        cache.move_to_end(cache_key)
        return cache[cache_key]
    geometry = _geometry(planner)
    side, finger = key
    hand = 'left' if side == -1 else 'right'
    start, end = tasks[0].beat/1000, (tasks[-1].beat+13)/1000
    offset = planner.offset(side, finger)
    preferred = seed if seed is not None else (.5+side*.24+offset[0]/planner.physical[0], .60)
    contact = dict(hand=hand, finger=finger, pointer=(0 if side == -1 else 5)+FINGER_ORDER.index(finger),
        start=start, end=end, beat=tasks[0].beat/1000, kind='drag', note_ids=[t.id for t in tasks],
        joint_note_coverage=True, planned=True, effort=.018, burst_cost=0., decision=dict(degraded=[]))
    timing = {}
    for task in tasks:
        timing.update(planner.timing_metadata(task).get('judgement_times', {}))
    contact['judgement_times'] = timing
    points = []
    for ms in sorted({t.beat for t in tasks}):
        when = ms/1000
        xy = project(geometry.zone(contact, when), preferred)
        if xy is None:
            cache[cache_key] = None
            if len(cache) > MAX_CACHE:
                cache.popitem(last=False)
            return None
        points.append([when, *xy])
    points.append([end-.001, *points[-1][1:]])
    contact['points'] = points
    extra, result = set(), None
    for _ in range(3):
        trial, info = refine_contact(geometry, contact, iterations=32, extra_times=extra)
        if trial is None:
            break
        if family == 'stationary' and not info.get('stationary'):
            break
        bad = validate_contact(geometry, contact, trial, limit=20)
        if not bad and not planner.blocks.path_violations(trial, start, end):
            result = dict(contact, points=trial)
            break
        extra.update(bad)
    cache[cache_key] = result
    if len(cache) > MAX_CACHE:
        cache.popitem(last=False)
    return result


def _transition_errors(planner, record, previous, hand_previous, occupied_until):
    """Exactly the lifecycle/transfer subset consumed by the neighbour check.

    Inputs are chronological prefix summaries. Collision/posture checks belong
    to the changed suffix, not to every immutable historical contact again.
    """
    t = record['start']
    if t < min(record['end'], occupied_until)-1e-9 or (previous and previous['end'] > t+1e-9):
        return ['occupied']
    if not planner.settings.speed_limits:
        return []
    settings = planner.settings
    errors = []
    xy = world_xy(record['points'][0][1:], planner.physical)
    if previous:
        distance = math.dist(world_xy(previous['points'][-1][1:], planner.physical), xy)
        required = max(.014, travel_time(distance, settings.finger_speed, settings.finger_acceleration))
        if t-previous['end'] < required:
            errors.append('finger_transfer')
    if hand_previous and hand_previous['end'] < t:
        distance = math.dist(planner.anchor(hand_previous, hand_previous['end']), planner.anchor(record, t))
        required = travel_time(max(0., distance-.025), settings.wrist_speed, settings.wrist_acceleration)
        if required > t-hand_previous['end']:
            errors.append('wrist_transfer')
    cache = getattr(planner, '_ribbon_speed_cache', None)
    if cache is None:
        cache = planner._ribbon_speed_cache = OrderedDict()
    key = id(record['points'])
    cached = cache.get(key)
    if cached is not None and cached[0] is record['points']:
        fast = cached[1]
        cache.move_to_end(key)
    else:
        fast = any(math.dist(world_xy(a[1:], planner.physical), world_xy(b[1:], planner.physical)) /
                   (b[0]-a[0]) > settings.finger_speed for a, b in zip(record['points'], record['points'][1:]))
        cache[key] = (record['points'], fast)
        if len(cache) > MAX_CACHE:
            cache.popitem(last=False)
    if fast:
        errors.append('contact_speed')
    return sorted(errors)


def _motion_ok(planner, candidate, old, contacts, route, reject=None):
    from flowphi.planning.planner import State
    geometry = _geometry(planner)
    if old is not None and candidate['end'] == old['end'] and candidate['points'] == old['points']:
        changed = None
    elif old is not None:
        stationary_extension = all(p[1:] == old['points'][-1][1:] for p in candidate['points'][len(old['points']):])
        changed = old['end'] if stationary_extension else old['points'][-1][0]
    else:
        changed = candidate['start']
    others = tuple(c for c in contacts if c is not old)
    if changed is not None:
        suffix = dict(candidate, start=changed, points=[[changed, *point_at(candidate['points'], changed)],
                      *[p for p in candidate['points'] if p[0] > changed]])
        # Continuation mode has no preceding-finger or wrist transfer term.
        # Only contacts overlapping the changed suffix can affect its checks;
        # keep old long Holds even when their DOWN was much earlier.
        overlapping = tuple(c for c in others if c['end'] > changed and c['start'] < candidate['end'])
        checks = getattr(planner, '_ribbon_suffix_cache', None)
        if checks is None:
            checks = planner._ribbon_suffix_cache = OrderedDict()
        check_key = (candidate['hand'], candidate['finger'], changed, candidate['end'],
                     tuple(tuple(p) for p in suffix['points']),
                     tuple((c['hand'], c['finger'], c['start'], c['end'], id(c['points'])) for c in overlapping))
        cached = checks.get(check_key)
        if cached is None:
            errors, _, _ = planner.constraints(suffix, State(overlapping),
                -1 if candidate['hand'] == 'left' else 1, candidate['finger'], continuation=True)
            # Retain point owners used in identity keys; numerical trajectories
            # are immutable during search, while branch metadata may differ.
            checks[check_key] = (tuple(errors), tuple(c['points'] for c in overlapping))
            if len(checks) > 2048:
                checks.popitem(last=False)
        else:
            checks.move_to_end(check_key)
            errors = cached[0]
        if errors:
            if reject is not None:
                reject.update('constraint_'+error for error in errors)
            return False
        # Check both neighbours: an appended tail may leave no time for a later head.
        ordered = sorted([*others, candidate], key=lambda c: (c['start'], c['pointer']))
        last_finger, last_hand, occupied_until = {}, {}, {}
        for record in ordered:
            key = record['hand'], record['finger']
            if key == (candidate['hand'], candidate['finger']):
                errors = _transition_errors(planner, record, last_finger.get(key), last_hand.get(record['hand']),
                                            occupied_until.get(key, -math.inf))
                if record is candidate and old is not None:
                    errors = [e for e in errors if e not in old.get('decision', {}).get('degraded', ())]
                if any(e in ('occupied', 'finger_transfer', 'wrist_transfer', 'contact_speed') for e in errors):
                    if reject is not None:
                        reject.update('neighbour_'+e for e in errors)
                    return False
            last_finger[key], last_hand[record['hand']] = record, record
            occupied_until[key] = max(occupied_until.get(key, -math.inf), record['end'])
    low = max(candidate['start'], min(route['start'], old['end']-.001 if old else route['start']))
    high = min(candidate['end'], route['end'])
    window = dict(candidate, start=low, end=high)
    cache = getattr(planner, '_ribbon_validation_cache', None)
    if cache is None:
        cache = planner._ribbon_validation_cache = OrderedDict()
    cache_key = (low, high, candidate['kind'], tuple(candidate['note_ids']),
                 bool(candidate.get('joint_note_coverage')), tuple(sorted(candidate.get('judgement_times', {}).items())),
                 repr({k: candidate.get(k) for k in ('collective_hold', 'refinement_region', 'joint_note_windows', 'block_clearance_m')}),
                 tuple(tuple(p) for p in candidate['points']))
    valid = cache.get(cache_key)
    if valid is None:
        valid = (_window_valid(planner, geometry, window, candidate['points']) and
                 not planner.blocks.path_violations(candidate['points'], low, high))
        cache[cache_key] = valid
        if len(cache) > MAX_CACHE:
            cache.popitem(last=False)
    else:
        cache.move_to_end(cache_key)
    if not valid and reject is not None:
        reject['continuation_geometry'] += 1
    return valid


def _attach(planner, state, route, reject=None):
    from flowphi.planning.planner import State
    old = next((c for c in reversed(state.contacts) if (c['hand'], c['finger']) ==
                (route['hand'], route['finger']) and c['start'] <= route['start']), None)
    output = []
    if old is None or old['end'] <= route['start']:
        fresh = dict(route)
        if _motion_ok(planner, fresh, None, state.contacts, route, reject):
            output.append(State(tuple(sorted((*state.contacts, fresh), key=lambda c: (c['start'], c['pointer'])))))
    if old is not None and route['start']-old['end'] <= MAX_IDLE_GAP:
        if old['end'] > route['start'] and not any(k in old for k in
                ('collective_hold', 'refinement_region', 'joint_note_windows', 'block_clearance_m')):
            # This interval is immutable. A later fitted suffix cannot repair
            # an already missed Drag here. Run the SAME millisecond validator
            # before spending time fitting that suffix (memoized per prefix).
            fixed = dict(route, end=min(old['end'], route['end']))
            if not _window_valid(planner, _geometry(planner), fixed, old['points']):
                if reject is not None:
                    reject['continuation_fixed_prefix'] += 1
                return output
        merged = dict(old, end=max(old['end'], route['end']),
                      note_ids=list(dict.fromkeys([*old['note_ids'], *route['note_ids']])),
                      joint_note_coverage=True, shared_contact='ribbon_continuation',
                      judgement_times={**old.get('judgement_times', {}), **route.get('judgement_times', {})})
        if old['kind'] == 'tap':
            merged['kind'] = 'drag'
        if old['end'] > route['start']:
            points = old['points']
            if merged['end'] > old['end']:
                # Refit the suffix with a pinned start. Simply splicing a path
                # already moving before UP creates an artificial speed spike.
                suffix = dict(route, start=old['end'], refinement_start=points[-1][1:],
                    points=[[old['end'], *points[-1][1:]],
                            *[p for p in route['points'] if p[0] > old['end']]])
                extensions = getattr(planner, '_ribbon_extension_cache', None)
                if extensions is None:
                    extensions = planner._ribbon_extension_cache = OrderedDict()
                extension_key = (suffix['start'], suffix['end'], tuple(suffix['note_ids']),
                    tuple(sorted(suffix.get('judgement_times', {}).items())), tuple(suffix['refinement_start']),
                    tuple(tuple(p) for p in suffix['points']))
                cached = extensions.get(extension_key)
                if cached is None:
                    trial, _ = refine_contact(_geometry(planner), suffix, iterations=32)
                    extensions[extension_key] = (trial,)
                    if len(extensions) > MAX_CACHE:
                        extensions.popitem(last=False)
                else:
                    extensions.move_to_end(extension_key)
                    trial = cached[0]
                if trial is None:
                    if reject is not None:
                        reject['continuation_pinned_start'] += 1
                    return output
                points = [*points, *trial]
                if points[-1][0] < merged['end']-.001:
                    points = [*points, [merged['end']-.001, *points[-1][1:]]]
        else:
            points = [*old['points'], *[p for p in route['points'] if p[0] > old['points'][-1][0]]]
        # Numerically identical immutable paths should also share ownership.
        # The score-prefix cache deliberately uses point-list identity, so
        # recreating the same concatenation defeats otherwise exact reuse.
        pool = getattr(planner, '_ribbon_point_pool', None)
        if pool is None:
            pool = planner._ribbon_point_pool = OrderedDict()
        point_key = tuple(tuple(p) for p in points)
        shared_points = pool.get(point_key)
        if shared_points is None:
            shared_points = pool[point_key] = points
            if len(pool) > MAX_CACHE:
                pool.popitem(last=False)
        else:
            pool.move_to_end(point_key)
        merged['points'] = shared_points
        if merged['end'] > old['end']:
            merged.pop('terminal_hold_release_ms', None)
            merged.pop('terminal_hold_release_reason', None)
        if _motion_ok(planner, merged, old, state.contacts, route, reject):
            output.append(State(tuple(merged if c is old else c for c in state.contacts)))
    return output


def _route_seeds(planner, state, group, key):
    """At most three placements; vertical judgement freedom must reach search."""
    side, finger = key
    hand = 'left' if side == -1 else 'right'
    previous = next((c for c in reversed(state.contacts) if (c['hand'], c['finger']) == (hand, finger)
                     and c['start'] <= group[0].beat/1000), None)
    seeds = [tuple(point_at(previous['points'], min(previous['end'], group[0].beat/1000)))] if previous else []
    x = .5+side*.24+planner.offset(side, finger)[0]/planner.physical[0]
    seeds.extend(((x, .38), (x, .82)))
    return seeds[:3]


def _heads_ok(planner, contacts, tasks):
    candidates = {}
    for task in tasks:
        if task.note.type in (NoteType.TAP, NoteType.HOLD):
            candidates[task.id] = [i for i, c in enumerate(contacts) if task.id in c['note_ids'] and
                abs(c['start']-c.get('judgement_times', {}).get(str(task.id), task.note.seconds)) <=
                PLANNING_WINDOW_MS[task.note.type]/1000+.0011]
    owners = {}
    def match(nid, seen):
        for i in candidates[nid]:
            if i not in seen:
                seen.add(i)
                if i not in owners or match(owners[i], seen):
                    owners[i] = nid
                    return True
        return False
    return all(match(nid, set()) for nid in candidates)


def _complete(planner, state, tasks):
    wanted = {t.id for t in tasks}
    if not wanted.issubset({n for c in state.contacts for n in c['note_ids']}):
        return False
    if not planner.contact_occupancy_valid(state.contacts) or not _heads_ok(planner, state.contacts, tasks):
        return False
    low = min(t.start for t in tasks)/1000
    high = max(t.end+1 if t.note.type in (NoteType.HOLD, NoteType.FLICK) else t.beat+13 for t in tasks)/1000
    for c in state.contacts:
        if not wanted.intersection(c['note_ids']):
            continue
        window = dict(c, start=max(c['start'], low), end=min(c['end'], high))
        if window['end'] > window['start'] and not _window_valid(planner, _geometry(planner), window, c['points']):
            return False
    return True


def metrics(planner, contacts, tasks):
    wanted = {t.id for t in tasks}
    records = [c for c in contacts if wanted.intersection(c['note_ids'])]
    travel, acceleration = 0., 0.
    totals = Counter()
    for c in records:
        velocity = []
        for a, b in zip(c['points'], c['points'][1:]):
            p, q = world_xy(a[1:], planner.physical), world_xy(b[1:], planner.physical)
            dt = b[0]-a[0]
            travel += math.dist(p, q)
            velocity.append(((a[0]+b[0])/2, ((q[0]-p[0])/dt, (q[1]-p[1])/dt)))
        for a, b in zip(velocity, velocity[1:]):
            acceleration = max(acceleration, math.dist(a[1], b[1])/max(.001, b[0]-a[0]))
        totals.update(c.get('decision', {}).get('costs', {}))
    low = min(t.start for t in tasks)/1000
    high = max(t.beat+13 for t in tasks)/1000
    return dict(contact_path_m=travel, downs=len(records),
        new_downs=sum(low <= c['start'] < high for c in records),
        relays=max(0, len([c for c in records if wanted.intersection(c['note_ids']) and
                          any(planner.original_tasks[n].note.type == NoteType.DRAG for n in c['note_ids'])])-1),
        sustained_seconds=sum(max(0., c['end']-c['start']) for c in records),
        peak_turn_acceleration_m_s2=acceleration, costs=dict(totals))


def _insert_heads(planner, tasks, states, reject, tick):
    """Insert independent heads around reserved future ribbon contacts."""
    from flowphi.planning.planner import State
    heads = [t for t in tasks if t.note.type != NoteType.DRAG]
    beam = list(states)[:8]
    for i, task in enumerate(heads):
        tick('heads', i, len(heads))
        expanded = []
        for state in beam:
            # Ordinary per-note planning expects chronological history. Future
            # reservations are checked as neighbours AFTER proposing the head.
            past = State(tuple(c for c in state.contacts if c['start'] <= task.start/1000), state.score)
            for _, record in planner.choices(task, past):
                if _motion_ok(planner, record, None, state.contacts, record, reject):
                    contacts = tuple(sorted((*state.contacts, record), key=lambda c: (c['start'], c['pointer'])))
                    expanded.append(score_history(planner, contacts))
        if not expanded:
            reject['deferred_heads_unplanned'] += 1
            return []
        beam = sorted(expanded, key=lambda s: s.score)[:LOCAL_WIDTH]
    return beam


def _family_candidates(planner, tasks, scaffolds, family, reject, tick, groups=None, deferred_heads=False):
    from flowphi.planning.planner import State
    if groups is None:
        groups = _groups(planner, [t for t in tasks if t.note.type == NoteType.DRAG], family)
    if not groups:
        reject['track_budget'] += 1
        return []
    groups.sort(key=lambda group: (group[0].beat, group[0].id))
    beam = [(state, None) for state in scaffolds]
    for index, group in enumerate(groups):
        tick(family, index, len(groups))
        expanded = []
        for state, last_key in beam:
            for key in planner.keys:
                if family == 'relay' and len(groups) > 1 and key == last_key:
                    continue
                reject['attempted_assignments'] += 1
                route = _route(planner, group, key, family)
                if route is None:
                    reject['geometry'] += 1
                    attached = []
                else:
                    attached = _attach(planner, state, route, reject)
                if not attached and (route is not None or family != 'stationary'):
                    for seed in _route_seeds(planner, state, group, key):
                        alternative = _route(planner, group, key, family, seed)
                        if alternative is not None:
                            attached.extend(_attach(planner, state, alternative, reject))
                if not attached:
                    reject['occupancy_or_motion'] += 1
                for candidate in attached:
                    scored = score_history(planner, candidate.contacts)
                    expanded.append((scored, key))
        expanded.sort(key=lambda item: item[0].score)
        seen, beam = set(), []
        for state, key in expanded:
            signature = tuple((c['hand'], c['finger'], tuple(c['note_ids']), c['start'], c['end'],
                               tuple(c['points'][0][1:]), tuple(c['points'][-1][1:])) for c in state.contacts)
            if signature not in seen:
                seen.add(signature)
                beam.append((state, key))
            if len(beam) >= LOCAL_WIDTH:
                break
        if not beam:
            break
    if deferred_heads:
        beam = [(s, None) for s in _insert_heads(planner, tasks, [s for s, _ in beam], reject, tick)]
    complete = []
    for state, _ in beam:
        if _complete(planner, state, tasks):
            complete.append(state)
        else:
            reject['coverage_head_or_geometry'] += 1
    return sorted(complete, key=lambda s: s.score)[:FAMILY_LIMIT]


def plan_window(planner, tasks, incoming, tick):
    """Return full-frontier alternatives and a compact, replayable comparison.

    Per-key planning is deliberately absent from the compared strategies: it is
    only a rescue for windows that no whole-window strategy can cover.
    """
    started = time.monotonic()
    from flowphi.planning.hand_evaluation import HandEvaluator, MAX_EVALUATIONS_PER_FAMILY
    from flowphi.planning.planner import State
    if getattr(planner, '_hand_evaluator', None) is None:
        planner._hand_evaluator = HandEvaluator(planner)
    evaluator = planner._hand_evaluator
    pool, uncertain, reports = [], [], {}
    motion_reports = {}
    non_drags = [t for t in tasks if t.note.type != NoteType.DRAG]
    missing = []
    scaffolds = planner._run_plain(tasks=non_drags, initial_states=incoming, return_states=True,
        width=FAMILY_LIMIT, on_unplanned=lambda t, r: missing.append(t),
        progress=lambda done, total, task: tick('heads', done, total)) if non_drags else incoming
    # Preserve several head assignments; geometry templates are shared across them.
    scaffolds = sorted(scaffolds, key=lambda s: s.score)[:FAMILY_LIMIT] if not missing else []
    heads_seconds = time.monotonic()-started
    for family in FAMILIES:
        family_started = time.monotonic()
        reject = Counter()
        candidates = []
        variants = list(_group_variants(planner, tasks, family))
        for groups in variants:
            candidates.extend(_family_candidates(planner, tasks, scaffolds, family, reject, tick, groups))
        if not candidates and non_drags:
            # Heads-first can occupy exactly the fingers a ribbon needs. Try
            # the reverse reservation order, within two fixed partition trials.
            for groups in sorted(variants, key=lambda gs: max(map(len, gs)))[:2]:
                reject['reverse_reservation_attempts'] += 1
                candidates.extend(_family_candidates(planner, tasks, incoming, family, reject, tick, groups,
                                                      deferred_heads=True))
                if candidates:
                    break
        candidates = sorted(candidates, key=lambda s: s.score)[:FAMILY_LIMIT]
        if not candidates and not reject:
            reject['track_budget'] += 1
        if not scaffolds:
            reject['head_scaffold_unplanned'] += max(1, len(missing))
        attempts = reject.pop('attempted_assignments', 0)
        touch_count = len(candidates)
        search_seconds = time.monotonic()-family_started
        hand_started = time.monotonic()
        checked = []
        # Prefer distinct hand/finger histories within the fixed rollout budget.
        signatures, chosen = set(), []
        for state in candidates:
            signature = tuple((c['hand'], c['finger'], tuple(c['note_ids'])) for c in state.contacts
                              if any(t.id in c['note_ids'] for t in tasks))
            if signature not in signatures:
                signatures.add(signature)
                chosen.append(state)
        chosen.extend(s for s in candidates if not any(s is c for c in chosen))
        for state in chosen[:MAX_EVALUATIONS_PER_FAMILY]:
            motion = evaluator.evaluate(state.contacts, tasks, lambda done, total: tick('hand_motion', done, total))
            scored = State(state.contacts, state.score+motion['cost'])
            motion_reports[id(scored)] = motion
            checked.append(scored)
            if motion['valid']:
                pool.append((scored, family))
            else:
                reject.update('hand_'+r for r in motion['reasons'])
                uncertain.append((scored, family))
        checked.sort(key=lambda s: (not motion_reports[id(s)]['valid'], s.score))
        feasible = [s for s in checked if motion_reports[id(s)]['valid']]
        reports[family] = dict(valid=bool(candidates), touch_valid=bool(candidates), count=touch_count,
                               search_seconds=search_seconds, hand_seconds=time.monotonic()-hand_started,
                               hand_valid=bool(feasible), hand_feasible_count=len(feasible),
                               touch_candidate_count=touch_count, hand_evaluated=len(checked),
                               attempted_assignments=attempts, rejections=dict(reject))
        if checked:
            best = checked[0]
            wanted = {t.id for t in tasks}
            reports[family].update(score=best.score, hand_motion=motion_reports[id(best)],
                metrics=metrics(planner, best.contacts, tasks),
                contacts=copy.deepcopy([c for c in best.contacts if wanted.intersection(c['note_ids'])]))
    motion_degraded = not pool and bool(uncertain)
    if motion_degraded:
        # Preserve the best touch-legal whole-window draft, explicitly failed by
        # the hand model. Per-note rescue cannot magically solve that failure.
        pool = uncertain
    if not pool:
        # No whole-window strategy covers this window. Plain per-key planning is
        # only a rescue so the notes survive; it is never compared or preferred.
        fallback_started = time.monotonic()
        pending = []
        from flowphi.planning.physical_plain import run_ordinary
        plain = run_ordinary(planner,tasks=tasks, initial_states=incoming, return_states=True,
            on_unplanned=lambda t, r: pending.append((t, r)),
            progress=lambda done, total, task: tick('baseline', done, total))
        plain = [type(state)(tuple(planner.join_drags(list(state.contacts))), state.score)
                 for state in plain]
        complete = [state for state in plain if _complete(planner, state, tasks)]
        usable = complete or plain
        best = min(usable, key=lambda s: s.score) if usable else None
        reports['baseline'] = dict(valid=bool(complete), fallback=True, count=len(complete),
            search_states=len(plain),
            score=best.score if best else None,
            metrics=metrics(planner, best.contacts, tasks) if best else None,
            rejections=dict(unplanned=len(pending), incomplete_or_invalid=len(plain)-len(complete)))
        if best and complete:
            reports['baseline']['hand_motion'] = evaluator.evaluate(best.contacts, tasks,
                lambda done, total: tick('hand_motion', done, total))
            reports['baseline']['hand_valid'] = reports['baseline']['hand_motion']['valid']
        else:
            reports['baseline']['hand_valid'] = False
        # Invalid search states may still be needed by the draft editor to
        # preserve unresolved notes; they must never be labelled valid results.
        return usable[:planner.settings.beam_width], dict(fallback=True, fallback_reason='no_touch_legal_whole_strategy', candidates=reports,
            timings=dict(heads_seconds=heads_seconds, search_seconds=sum(r['search_seconds'] for f, r in reports.items() if f in FAMILIES),
                         hand_seconds=sum(r['hand_seconds'] for f, r in reports.items() if f in FAMILIES),
                         fallback_seconds=time.monotonic()-fallback_started),
            hand_motion=reports['baseline'].get('hand_motion'),
            hand_motion_degraded=not reports['baseline'].get('hand_valid', False),
            seconds=time.monotonic()-started, selected='baseline'), [] if complete else pending
    pool.sort(key=lambda item: item[0].score)
    # Keep one feasible representative per strategy when the outer beam permits.
    selected, used = [], set()
    for item in pool:
        if item[1] not in used:
            selected.append(item)
            used.add(item[1])
            if len(selected) >= planner.settings.beam_width:
                break
    for item in pool:
        if len(selected) >= planner.settings.beam_width:
            break
        if not any(item[0] is s[0] for s in selected):
            selected.append(item)
    selected.sort(key=lambda item: item[0].score)
    result = []
    window_id = tasks[0].id
    for state, family in selected:
        contacts = list(state.contacts)
        wanted = {t.id for t in tasks}
        marker = next(i for i in range(len(contacts)-1, -1, -1)
                      if any(t.id in contacts[i]['note_ids'] for t in tasks))
        c = contacts[marker]
        contacts[marker] = dict(c, ribbon_decisions=[*c.get('ribbon_decisions', ()),
            dict(window=window_id, strategy=family, score=state.score,
                 hand_motion=motion_reports[id(state)],
                 metrics=metrics(planner, state.contacts, tasks),
                 route_signature=[(r['hand'], r['finger'], sorted(wanted.intersection(r['note_ids'])),
                                   r['start'], r['end'], r['points'][-1][1:])
                                  for r in state.contacts if wanted.intersection(r['note_ids'])])])
        result.append(State(tuple(contacts), state.score))
    best = selected[0]
    return result, dict(fallback=False, hand_motion_degraded=motion_degraded,
                       timings=dict(heads_seconds=heads_seconds, search_seconds=sum(r['search_seconds'] for r in reports.values()),
                                    hand_seconds=sum(r['hand_seconds'] for r in reports.values())),
                       hand_motion=motion_reports[id(best[0])], candidates=reports, seconds=time.monotonic()-started,
                       selected=best[1], selected_score=best[0].score), []


def run_ribbons(planner, console=None, *, on_unplanned=None, progress=None):
    from flowphi.planning.planner import State, PlanningError
    from flowphi.planning.physical_plain import run_ordinary
    started = time.monotonic()
    windows = detect_windows(planner)
    diagnostics = planner.ribbon_diagnostics = dict(enabled=True, recognition_seconds=time.monotonic()-started,
        local_state_limit=LOCAL_WIDTH, family_candidate_limit=FAMILY_LIMIT, windows=[])
    if not windows:
        return run_ordinary(planner,console, on_unplanned=on_unplanned, progress=progress)
    tasks, beam, cursor = planner.tasks, [State()], 0
    def plain(begin, end):
        return run_ordinary(planner,console, tasks=tasks[begin:end], initial_states=beam,
            return_states=True, on_unplanned=on_unplanned,
            progress=(lambda done, total, task: progress(begin+done, len(tasks), task)) if progress else None)
    for window in windows:
        if cursor < window.start:
            beam = plain(cursor, window.start)
        scene = tasks[window.start:window.stop]
        def tick(family, done, total):
            planner.ribbon_progress = dict(start=scene[0].start/1000, end=scene[-1].start/1000,
                strategy=family, done=done, total=total, elapsed_seconds=time.monotonic()-started)
            if progress:
                progress(window.start, len(tasks), replace(scene[0], planning_detail=planner.ribbon_progress))
        beam, report, pending = plan_window(planner, scene, beam, tick)
        if pending:
            for task, reasons in pending:
                if on_unplanned is None:
                    raise PlanningError(f'No feasible fingering for note {task.id} in Drag phrase')
                on_unplanned(task, reasons)
        report.update(id=scene[0].id, start=scene[0].start/1000, end=scene[-1].start/1000,
                      note_ids=[t.id for t in scene], drag_ids=list(window.drag_ids))
        diagnostics['windows'].append(report)
        if console:
            console.print(f'DRAG_PHRASE {report["start"]:.3f}–{report["end"]:.3f}s '
                          f'{report["selected"]} {report["seconds"]:.2f}s')
        cursor = window.stop
    planner.ribbon_progress = None
    if cursor < len(tasks):
        beam = plain(cursor, len(tasks))
    if progress:
        progress(len(tasks), len(tasks), None)
    contacts = list(min(beam, key=lambda s: s.score).contacts)
    winners = {d['window']: d for c in contacts for d in c.get('ribbon_decisions', ())}
    for report in diagnostics['windows']:
        winner = winners.get(report['id'])
        if winner:
            report.update(selected=winner['strategy'], selected_score=winner['score'],
                          selected_metrics=winner['metrics'], hand_motion=winner.get('hand_motion'),
                          hand_motion_degraded=not winner.get('hand_motion', {}).get('valid', True))
    diagnostics['candidate_seconds'] = sum(w['seconds'] for w in diagnostics['windows'])
    diagnostics['attempted_assignments'] = sum(c.get('attempted_assignments', 0)
        for w in diagnostics['windows'] for c in w['candidates'].values())
    return contacts


def audit_hand_motion(planner, contacts, progress=None):
    """Audit final contacts AFTER joining, pending assignment and compaction."""
    if not getattr(planner, '_hand_evaluator', None):
        return
    from flowphi.planning.hand_evaluation import MAX_EVALUATIONS_PER_FAMILY
    started = time.monotonic()
    diagnostics = planner.ribbon_diagnostics
    for report in diagnostics['windows']:
        wanted = set(report['note_ids'])
        scene = [t for t in planner.tasks if t.id in wanted]
        def tick(done, total):
            if progress:
                detail = dict(start=report['start'], end=report['end'], strategy='hand_motion',
                              done=done, total=total, elapsed_seconds=time.monotonic()-started)
                progress(len(planner.tasks), len(planner.tasks), replace(scene[0], planning_detail=detail))
        motion = planner._hand_evaluator.evaluate(contacts, scene, tick)
        report.update(hand_motion=motion, hand_motion_degraded=not motion['valid'])
    diagnostics['hand_audit_seconds'] = time.monotonic()-started
    diagnostics['hand_motion_failed_windows'] = sum(w.get('hand_motion_degraded', False) for w in diagnostics['windows'])
    diagnostics['hand_evaluation_limit_per_family'] = MAX_EVALUATIONS_PER_FAMILY
    if planner.settings.hand_motion_repair and diagnostics['hand_motion_failed_windows']:
        from flowphi.planning.repair import repair_failed_windows
        repair_failed_windows(planner,contacts,progress)
    planner.ribbon_progress = None
    if progress:
        progress(len(planner.tasks), len(planner.tasks), None)
