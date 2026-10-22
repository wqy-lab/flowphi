from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
import copy
import math

from flowphi import hand_limits as limits

from flowphi.motion import (MotionIndex, default_profile, finger_pose, finish_motion, palm_offsets,
                            point_at, smooth, world_xy, wrist_pose)
from flowphi.pose import (FINGERS, PoseSampler, contact_reach, finger_deltas, finger_planes, fit_wrist, frame_values,
                         idle_weight, rotate, rotation_matrix, wrist_constraints)


def add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def scaled(a, value):
    return tuple(x * value for x in a)


def unit(a):
    return scaled(a, 1 / max(1e-12, math.dist(a, (0., 0., 0.))))


def chain_to(points, target, lengths):
    points = list(points)
    base = points[0]
    if math.dist(base, target) >= sum(lengths):
        direction = unit(sub(target, base))
        return [base, *[add(base, scaled(direction, sum(lengths[:i]))) for i in (1, 2, 3)]]
    for _ in range(32):
        points[-1] = target
        for i in (2, 1, 0):
            points[i] = add(points[i + 1], scaled(unit(sub(points[i], points[i + 1])), lengths[i]))
        points[0] = base
        for i in range(3):
            points[i + 1] = add(points[i], scaled(unit(sub(points[i + 1], points[i])), lengths[i]))
        if math.dist(points[-1], target) < 1e-6:
            break
    return points


def idle_chain(base, tip, length, deltas, reference=None):
    sizes = [length * value for value in (.43, .34, .23)]
    if reference is None:
        delta = sub(tip, base)
        points = [base, *[add(add(base, scaled(delta, u)), (0., 0., length * .22 * math.sin(math.pi * u)))
                         for u in (.43, .77, 1.)]]
        points = chain_to(points, tip, sizes)
    else:
        points = reference
    if not any(abs(value) > 1e-9 for delta in deltas for value in delta):
        return points, sizes
    rotations = []
    result = [base]
    for i, angles in enumerate(deltas):
        rotations.append(rotation_matrix(angles[2], angles[0], angles[1]))
        segment = sub(points[i + 1], points[i])
        for matrix in reversed(rotations):
            segment = rotate(matrix, segment)
        result.append(add(result[-1], segment))
    return result, sizes


class HandMotionSolver:
    """Headless constrained skeleton used by planning and the interactive preview."""
    def __init__(self):
        self.ready = False
        self.warning = ''
        self.sampler = PoseSampler()
        self._reset_limits()

    def _reset_limits(self):
        self._states = OrderedDict()
        self._checkpoints = {}
        self._substates = OrderedDict()

    def refresh(self):
        pass

    def set_motion(self, contacts, profile, screen, *, settings=None, pose=None, guides=(), pose_style=None):
        self._reset_limits()
        # Planning diagnostics contain entire rollout reports. They are not
        # motion inputs; copying them into every speculative solver needlessly
        # duplicates the search tree. Keep all physical/contact fields intact.
        self.contacts = copy.deepcopy([{k:v for k,v in c.items() if k not in
            ('decision','ribbon_decisions','ordinary_decisions')} for c in contacts])
        self.guides = guides
        self.sampler = PoseSampler(pose)
        self.contacts.sort(key=lambda c: (c['start'], c['hand'], c['finger']))
        self._event_times = sorted({float(t) for c in self.contacts
            for t in (c['start'],c['end'],c.get('beat',c['start']))})
        self.profile = default_profile(profile['scale'])
        self.profile['fingers'].update(profile.get('fingers', {}))
        self.rests = finish_motion(self.contacts, self.profile, screen, .28 / .82)
        self.index = MotionIndex(self.contacts, self.rests)
        self.offsets = palm_offsets(self.profile)
        self.rest_chains = {(int(key.split(':')[0]), key.split(':')[1]):
            idle_chain(tuple(data['base']), tuple(data['offset']), data['length'], [(0., 0., 0.)] * 3)[0]
            for key, data in self.profile['fingers'].items()}
        self.joint_models = {key: limits.chain_models(rest, key[1]) for key, rest in self.rest_chains.items()}
        self.tracks = {(side, finger): [c for c in self.contacts if c['hand'] == hand and c['finger'] == finger]
                       for side, hand in ((-1, 'left'), (1, 'right')) for finger in FINGERS}
        if any(a['end'] > b['start'] + 1e-9 for events in self.tracks.values() for a, b in zip(events, events[1:])):
            self.warning = '同一手指有重叠动作，请在“问题”页检查指法'
        self.job = dict(screen=screen, hand_scale=self.profile['scale'], contact_height=.0005,
            palm_motion='v4', finger_motion='whole_finger', palm_lift_mode='gentle',
            palm_lift_ratio=.70, palm_strike_speed=1.8, stroke_peak_phase=.62, palm_lift_low=.042,
            palm_lift_high=.035, stroke_gap_limit=1., transfer_sway=.004, lift_height=.034)
        self.job.update((pose_style or {}))
        self.job['wrist_speed'] = (settings or {}).get('wrist_speed', limits.WRIST_SPEED)
        self.ready = True

    def reuse_prefix(self, other):
        """Share only states before any input to the motion solver can differ.

        Derived preparation/release and wrist-rest intervals are part of this
        comparison: equal DOWN events alone do not imply equal earlier poses.
        """
        if self.job != other.job or self.guides != other.guides or self.profile != other.profile:
            return -60
        if self.sampler is not other.sampler and self.sampler.__dict__ != other.sampler.__dict__:
            return -60
        earliest = math.inf
        fields = ('start', 'end', 'beat', 'prepare', 'release_until', 'lift', 'points')
        for key, events in self.tracks.items():
            old = other.tracks[key]
            for i in range(max(len(events), len(old))):
                a = events[i] if i < len(events) else None
                b = old[i] if i < len(old) else None
                if a is None or b is None or any(a.get(f) != b.get(f) for f in fields):
                    earliest = min(earliest, *(c['prepare'] for c in (a, b) if c is not None))
                    break
        for hand in ('left', 'right'):
            current = [r for r in self.rests if r['hand'] == hand]
            old = [r for r in other.rests if r['hand'] == hand]
            for i in range(max(len(current), len(old))):
                a = current[i] if i < len(current) else None
                b = old[i] if i < len(old) else None
                if a != b:
                    earliest = min(earliest, *(r['start'] for r in (a, b) if r is not None))
                    break
        last = math.floor(earliest*60)-1 if math.isfinite(earliest) else max(other._checkpoints, default=-60)
        available = {k: v for k, v in (*other._checkpoints.items(), *other._states.items()) if k <= last}
        if not available:
            return -60
        self._checkpoints = {k: v for k, v in other._checkpoints.items() if k <= last}
        self._substates = OrderedDict((k,v) for k,v in other._substates.items() if k<=last)
        k = max(available)
        self._states[k] = available[k]
        return k

    def sample(self, key, t, rest, job):
        events = self.tracks[key]
        i = bisect_right(events, t, key=lambda c: c['start']) - 1
        if i >= 0 and abs(t - events[i]['end']) < 1e-9:
            return (*world_xy(point_at(events[i]['points'], t), job['screen']), job['contact_height']), 1., True
        return finger_pose(events, t, rest, job, key)

    def _raw_frame(self, t, solve_fingers=True):
        if not self.ready:
            return None
        self.refresh()
        frame = self.sampler.frame(t)
        job = dict(self.job, pose_frame=frame)
        result = dict(screen=job['screen'], hands={}, warning=self.warning)
        for side, hand in ((-1, 'left'), (1, 'right')):
            keys = [(side, finger) for finger in FINGERS]
            rest_root = (side * .09, -.16, .055 * job['hand_scale'] / .29)
            samples = {key: self.sample(key, t, add(rest_root, self.profile['fingers'][f'{side}:{key[1]}']['offset']), job)
                       for key in keys}
            aiming = [(pos, weight) for pos, weight, _ in samples.values() if weight > .001]
            yaw = 0.
            if aiming:
                total = sum(w for _, w in aiming)
                aim = tuple(sum(p[axis] * w for p, w in aiming) / total for axis in range(3))
                yaw = max(-.45, min(.45, -math.atan2(aim[0] - side * .12, max(.08, aim[1] + .24))))
            for guide in self.guides:
                if guide['hand'] == hand and guide['prepare'] <= t <= guide['release']:
                    weight = (smooth((t - guide['prepare']) / max(1e-6, guide['start'] - guide['prepare'])) if t < guide['start']
                              else smooth((guide['release'] - t) / max(1e-6, guide['release'] - guide['end'])) if t > guide['end'] else 1.)
                    yaw += (guide['yaw'] - yaw) * weight
            values = frame_values(job, side)
            rotation = rotation_matrix(yaw + values['yaw'], values['pitch'], values['roll'])
            offsets = {key: rotate(rotation, self.offsets[key]) for key in self.offsets}
            bases = {key: rotate(rotation, self.profile['fingers'][f'{side}:{key[1]}']['base']) for key in keys}
            root = wrist_pose(self.contacts, self.rests, side, t, job, offsets, self.offsets, self.index)
            manual = any(abs(values[name]) > 1e-9 for name in ('x', 'y', 'z', 'yaw', 'pitch', 'roll'))
            constraints = [(sub(pos, bases[key]), self.profile['fingers'][f'{side}:{key[1]}']['length'] * .94)
                           for key, (pos, _, down) in samples.items() if down]
            active = [sub(pos, offsets[key]) for key, (pos, _, down) in samples.items() if down]
            if active:
                anchor = tuple(sum(p[a] for p in active) / len(active) for a in range(3))
                delta = (root[0] - anchor[0], root[1] - anchor[1], 0.)
                radius = .015 * job['hand_scale'] / .29
                if math.dist(delta, (0., 0., 0.)) > radius:
                    xy = add(anchor, scaled(unit(delta), radius))
                    root = (xy[0], xy[1], root[2])
            for _ in range(16):
                for center, radius in constraints:
                    r = math.sqrt(max(.000001, radius ** 2 - (root[2] - center[2]) ** 2))
                    delta = (root[0] - center[0], root[1] - center[1], 0.)
                    if math.dist(delta, (0., 0., 0.)) > r:
                        xy = add(center, scaled(unit(delta), r))
                        root = (xy[0], xy[1], root[2])
            requested, residual = root, 0.
            if manual:
                requested = add(root, (values['x'], values['y'], values['z']))
                constraints = [(sub(pos, bases[key]), contact_reach(self.profile['fingers'][f'{side}:{key[1]}']['length'], key[1]))
                               for key, (pos, _, down) in samples.items() if down]
                planes = [plane for key, (pos, _, down) in samples.items() if down
                          for plane in finger_planes(sub(pos, bases[key]), rotation, key[1], job.get('finger_lateral_limit'))]
                constraints, planes = wrist_constraints(root, values, constraints, planes)
                root, residual = fit_wrist(requested, constraints, planes=planes)
            fingers = {}
            for key in keys:
                finger = key[1]
                data = self.profile['fingers'][f'{side}:{finger}']
                tip_rest = add(root, rotate(rotation, data['offset']))
                tip_rest = (*tip_rest[:2], max(.012, tip_rest[2]))
                pos, weight, down = self.sample(key, t, tip_rest, job)
                idle = idle_weight(self.tracks[key], t)
                deltas = finger_deltas(frame, side, finger, data['length'])
                adjusted = [scaled(delta, idle) for delta in deltas]
                local, sizes = idle_chain(tuple(data['base']), tuple(data['offset']), data['length'], adjusted, self.rest_chains[key])
                joints = [add(root, rotate(rotation, p)) for p in local]
                influence = weight * (1 - idle if any(abs(v) > 1e-9 for row in deltas for v in row) else 1.)
                if solve_fingers and influence > 1e-5:
                    solved = chain_to(joints, pos, sizes)
                    joints = [add(scaled(a, 1 - influence), scaled(b, influence)) for a, b in zip(joints, solved)]
                fingers[finger] = dict(joints=joints, down=down, idle=idle, target=pos,
                    influence=influence,
                    relaxed=influence <= 1e-5 and not any(abs(v) > 1e-9 for row in deltas for v in row),
                    error=math.dist(joints[-1], pos) if down else 0.)
            result['hands'][hand] = dict(root=root, requested=requested, rotation=rotation, fingers=fingers,
                correction=math.dist(root, requested), reach_error=residual,
                palm=[add(root, rotate(rotation, p)) for p in ((-.035, .012, 0.), (.035, .012, 0.),
                    tuple(self.profile['fingers'][f'{side}:little']['base']), tuple(self.profile['fingers'][f'{side}:index']['base']))])
        return result

    def _advance(self, t, previous, dt):
        raw = self._raw_frame(t, solve_fingers=False)
        state = {}
        for side, hand in ((-1, 'left'), (1, 'right')):
            data = raw['hands'][hand]
            old = previous.get(hand) if previous else None
            rotation = limits.matrix_quaternion(data['rotation'])
            root = data['root']
            if old:
                root = limits.bounded_position(old['root'], root, self.job['wrist_speed'], dt)
                rotation = limits.bounded_rotation(old['rotation'], rotation, limits.WRIST_TURN_SPEED, dt)
            # Solve the target relative to the already limited palm, so a delayed
            # wrist can still use the remaining finger reach before reporting a miss.
            inverse = limits.conjugate(limits.matrix_quaternion(data['rotation']))
            limited_inverse = limits.conjugate(rotation)
            fingers = {}
            for finger, value in data['fingers'].items():
                rest = self.rest_chains[side, finger]
                if value['relaxed']:
                    rotations = [limits.IDENTITY] * 3
                else:
                    desired = [limits.apply(inverse, sub(p, data['root'])) for p in value['joints']]
                    if value['influence'] > 1e-5:
                        target = limits.apply(limited_inverse, sub(value['target'], root))
                        lengths = [math.dist(rest[i], rest[i+1]) for i in range(3)]
                        solved = chain_to(desired, target, lengths)
                        desired = [add(scaled(a, 1-value['influence']), scaled(b, value['influence']))
                                   for a, b in zip(desired, solved)]
                    rotations = limits.chain_rotations(rest, desired)
                before = old['fingers'][finger] if old else [limits.IDENTITY] * 3
                models = self.joint_models[side, finger]
                bounded = [limits.joint_rotation(before[i], q, limits.IDENTITY, i, dt, models[i])
                           for i, q in enumerate(rotations)]
                if value['influence'] > 1e-5:
                    # After limiting distal curl, aim the remaining curved finger
                    # with its root joint rather than trying to unfold the tip.
                    shape = limits.forward_chain(rest, [limits.IDENTITY, *bounded[1:]])
                    target = limits.apply(limited_inverse, sub(value['target'], root))
                    aiming = limits.between(sub(shape[-1], rest[0]), sub(target, rest[0]))
                    desired = limits.interpolate(rotations[0], aiming, value['influence'])
                    bounded[0] = limits.joint_rotation(before[0], desired, limits.IDENTITY, 0, dt, models[0])
                fingers[finger] = bounded
            state[hand] = dict(root=root, rotation=rotation, fingers=fingers)
        return state

    def _cell(self, tick, previous):
        """Subdivide a fixed cell at exact contact boundaries, keeping real dt."""
        if tick in self._substates:
            self._substates.move_to_end(tick)
            return self._substates[tick]
        low,high=(tick-1)/60,tick/60
        a,b=bisect_right(self._event_times,low+1e-10),bisect_right(self._event_times,high-1e-10)
        nodes=[(low,previous)]
        for when in [*self._event_times[a:b],high]:
            state=self._advance(when,nodes[-1][1],when-nodes[-1][0])
            nodes.append((when,state))
        self._substates[tick]=nodes
        if len(self._substates)>120:
            self._substates.popitem(last=False)
        return nodes

    def _state(self, tick):
        # Fixed time sampling makes scrubbing and playback produce identical poses.
        # Keep one checkpoint per second and a bounded hot cache for nearby frames.
        if tick in self._states:
            return self._states[tick]
        if not self._checkpoints:
            self._checkpoints[-60] = self._advance(-1., None, 1/60)
        start = max(k for k in (*self._checkpoints, *self._states) if k <= tick)
        state = self._states.get(start, self._checkpoints.get(start))
        for k in range(start + 1, tick + 1):
            state = self._cell(k,state)[-1][1]
            self._states[k] = state
            if k % 60 == 0:
                self._checkpoints[k] = state
            while len(self._states) > 120:
                self._states.popitem(last=False)
        return state

    def frame(self, t):
        if not self.ready:
            return None
        self.refresh()
        t = max(-1., t)
        tick = math.floor(t * 60)
        u = t * 60 - tick
        first = self._state(tick)
        second = self._state(tick + 1) if u > 1e-9 else first
        if u>1e-9:
            nodes=self._cell(tick+1,first)
            index=max(0,bisect_right(nodes,t,key=lambda row:row[0])-1)
            lo,first=nodes[index]
            hi,second=nodes[min(index+1,len(nodes)-1)]
            u=(t-lo)/max(1e-12,hi-lo)
        result = self._raw_frame(t, solve_fingers=False)
        max_error = 0.
        for side, hand in ((-1, 'left'), (1, 'right')):
            data, a, b = result['hands'][hand], first[hand], second[hand]
            root = tuple(x + (y-x)*u for x, y in zip(a['root'], b['root']))
            rotation = limits.interpolate(a['rotation'], b['rotation'], u)
            inverse = limits.conjugate(limits.matrix_quaternion(data['rotation']))
            palm_local = [limits.apply(inverse, sub(p, data['root'])) for p in data['palm']]
            data.update(root=root, rotation=limits.quaternion_matrix(rotation),
                        correction=math.dist(root, data['requested']),
                        palm=[add(root, limits.apply(rotation, p)) for p in palm_local])
            for finger, value in data['fingers'].items():
                joints = [limits.joint_interpolate(x, y, u, limits.IDENTITY, model)
                          for x, y, model in zip(a['fingers'][finger], b['fingers'][finger], self.joint_models[side, finger])]
                local = limits.forward_chain(self.rest_chains[side, finger], joints)
                value['joints'] = [add(root, limits.apply(rotation, p)) for p in local]
                value['error'] = math.dist(value['joints'][-1], value['target']) if value['down'] else 0.
                max_error = max(max_error, value['error'])
            data['reach_error'] = max(v['error'] for v in data['fingers'].values())
        if max_error > .001:
            result['warning'] = (result['warning'] + ' · ' if result['warning'] else '') + f'运动约束后触点偏差最大 {max_error*1000:.1f} mm，需要调整指法'
        return result
