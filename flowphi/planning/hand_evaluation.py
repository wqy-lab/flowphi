"""Bounded rollouts through the same constrained skeleton as the hand preview."""
from collections import OrderedDict
import math
import time

from shapely.geometry import Point

from flowphi.charts.model import NoteType
from flowphi.hand_solver import HandMotionSolver
from flowphi.hand_limits import ANATOMY_MODEL
from flowphi.motion import point_at,world_xy
from flowphi.planning.contacts import ContactGeometry

MAX_EVALUATIONS_PER_FAMILY = 8
MAX_CACHE = 64
CONTACT_TOLERANCE = .004
HEIGHT_TOLERANCE = .002


def _dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def _sub(a, b):
    return tuple(x-y for x, y in zip(a, b))


def segment_distance(a, b, c, d):
    """Distance of closed 3D segments, including parallel and degenerate cases."""
    u, v, w = _sub(b, a), _sub(d, c), _sub(a, c)
    aa, bb, cc, dd, ee = _dot(u, u), _dot(u, v), _dot(v, v), _dot(u, w), _dot(v, w)
    determinant = aa*cc-bb*bb
    s = max(0., min(1., (bb*ee-cc*dd)/determinant)) if determinant > 1e-15 else 0.
    t = max(0., min(1., (bb*s+ee)/max(cc, 1e-15)))
    s = max(0., min(1., (bb*t-dd)/max(aa, 1e-15)))
    t = max(0., min(1., (bb*s+ee)/max(cc, 1e-15)))
    return math.sqrt(sum((w[i]+s*u[i]-t*v[i])**2 for i in range(3)))


def _finger_segments(joints):
    # Exclude only the webbing near MCP, not the proximal phalanx.
    first = tuple(a*.65+b*.35 for a, b in zip(joints[0], joints[1]))
    return ((first, joints[1]), (joints[1], joints[2]), (joints[2], joints[3]))


class HandEvaluator:
    def __init__(self, planner):
        self.planner = planner
        self.cache = OrderedDict()
        self.solvers = []
        if getattr(planner, '_ribbon_geometry', None) is None:
            planner._ribbon_geometry = ContactGeometry(planner.chart, physical_screen=planner.physical,
                                                        extra_clearance_m=.004, cache_judgement=True)

    def evaluate(self, contacts, tasks, tick=None, *, time_range=None):
        started = time.monotonic()
        cpu_started = time.process_time()
        low = min(t.start for t in tasks)/1000
        high = max(t.end+1 if t.note.type == NoteType.FLICK else t.beat+13 for t in tasks)/1000
        if time_range is not None:
            low,high=time_range
        signature = (low, high, tuple(t.id for t in tasks),tuple((c['hand'], c['finger'], c['start'], c['end'],
                     c['kind'], tuple(c['note_ids']), tuple(sorted(c.get('judgement_times', {}).items())),
                     tuple(tuple(p) for p in c['points'])) for c in contacts))
        if signature in self.cache:
            self.cache.move_to_end(signature)
            return self.cache[signature]
        p = self.planner
        solver = HandMotionSolver()
        solver.set_motion(contacts, p.profile, p.physical, settings=vars(p.settings))
        # Prefix reuse is exact and checks derived preparation and rest inputs,
        # not a guessed short warm-up or an independently reset wrist pose.
        best_tick, best = -61, None
        for previous in self.solvers:
            trial = solver.reuse_prefix(previous)
            if trial > best_tick:
                best_tick, best = trial, (dict(solver._checkpoints), OrderedDict(solver._states),OrderedDict(solver._substates))
            solver._reset_limits()
        if best:
            solver._checkpoints, solver._states, solver._substates = best
        times = {low, high-1e-6, *(i/60 for i in range(math.ceil(low*60), math.ceil((high+.18)*60)))}
        for task in tasks:
            if low<=task.beat/1000<high:
                times.add(task.beat/1000)
            if task.note.type in (NoteType.HOLD, NoteType.FLICK):
                tail=min(high-1e-6, (task.end-1)/1000)
                if tail>=low:
                    times.add(tail)
        relevant = [c for c in solver.contacts if c['start'] < high and c['end'] >= low]
        for c in relevant:
            times.update(t for t in (c['start'], c['end']-1e-6) if low <= t < high)
        by_key = {}
        for c in relevant:
            by_key.setdefault((c['hand'], c['finger']), []).append(c)
        squared, samples, misses, max_error, max_height, penetration, crossed = 0., 0, 0, 0., 0., 0., 0.
        reasons = set()
        failed_notes=set()
        worst = []
        travel, previous_roots = 0., None
        flick_positions = {}
        from flowphi.planning.occlusion import scorer
        visibility = scorer(p) if p.settings.occlusion_weight else None
        # Uniform midpoints keep a soft visibility integral independent of the
        # number of note/contact boundary samples in the hard feasibility pass.
        visibility_count = max(1, math.ceil((high-low)/.10))
        visibility_step = (high-low)/visibility_count
        visibility_times = [low+(i+.5)*visibility_step for i in range(visibility_count)] if visibility else []
        hard_times = set(times)
        visibility_set = set(visibility_times)
        times.update(visibility_times)
        hidden_note_seconds = 0.
        max_hidden = 0.
        from flowphi.planning.interaction import frame_relation, density
        interaction_integrals = [0., 0., 0., 0.]
        interaction_previous, interaction_peak = None, [0., 0., 0., 0.]
        setup_seconds = time.monotonic()-started
        warmup_seconds = warmup_cpu_seconds = 0.
        for i, t in enumerate(sorted(times)):
            if tick and i % 12 == 0:
                tick(i, len(times))
            if i == 0:
                warmup_started = time.monotonic(), time.process_time()
            frame = solver.frame(t)
            if i == 0:
                warmup_seconds = time.monotonic()-warmup_started[0]
                warmup_cpu_seconds = time.process_time()-warmup_started[1]
                if warmup_seconds > 5:
                    from flowphi.progress import report_log
                    report_log(f'手部历史回放 {best_tick/60:.3f}–{t:.3f} s: '
                               f'{warmup_seconds:.2f} s; CPU {warmup_cpu_seconds:.2f} s')
            if t in visibility_set:
                hidden = visibility.frame(t, frame)
                hidden_note_seconds += hidden*visibility_step
                max_hidden = max(max_hidden, hidden)
            if t not in hard_times:
                continue
            # Reuse the exact constrained frames already required by the hard
            # audit; projected crossings remain a separate soft preference.
            values = frame_relation(frame, p.profile['scale']/.27)
            if interaction_previous is not None:
                last_t, last_values = interaction_previous
                interaction_integrals = [a+(b+c)*.5*(t-last_t)
                    for a, b, c in zip(interaction_integrals, values, last_values)]
            interaction_peak = [max(a,b) for a,b in zip(interaction_peak, values)]
            interaction_previous = t, values
            roots = [frame['hands'][h]['root'] for h in ('left', 'right')]
            if previous_roots:
                travel += sum(math.dist(a, b) for a, b in zip(roots, previous_roots))
            previous_roots = roots
            crossed = max(crossed, roots[0][0]-roots[1][0]-.012)
            active = []
            all_fingers = [(hand, finger, data) for hand, values in frame['hands'].items()
                           for finger, data in values['fingers'].items()]
            for hand, finger, data in all_fingers:
                records = by_key.get((hand, finger), ())
                record = next((c for c in records if c['start']-1e-9 <= t < min(high, c['end'])), None)
                if record is None:
                    continue
                active.append((hand, finger))
                # A later overlapping DOWN can hide an older expected Hold.
                # Preview's error is zero while UP; that must not mask a miss.
                expected=(*world_xy(point_at(record['points'],t),p.physical),solver.job['contact_height'])
                error = math.dist(data['joints'][-1],expected)
                if not data['down']:
                    reasons.add('contact_state')
                height = abs(data['joints'][-1][2]-solver.job['contact_height'])
                squared += error*error
                samples += 1
                max_error, max_height = max(max_error, error), max(max_height, height)
                tip = data['joints'][-1]
                if record['kind'] == 'flick' and record['start'] >= low:
                    flick_positions.setdefault(id(record), []).append(tip[:2])
                xy = (.5+tip[0]/p.physical[0], .5-tip[1]/p.physical[1])
                # The planned path uses the conservative centre strip; the
                # realised fingertip is checked against the legal outer strip.
                # A sub-mm tracking error must not fail a path merely because
                # its optimum lies on the *planning margin*, inside judgement.
                legal = p._ribbon_geometry.zone(record, t, central=False).buffer(1e-8).covers(Point(*xy))
                failed = error > CONTACT_TOLERANCE or height > HEIGHT_TOLERANCE or not legal or not data['down']
                if failed:
                    failed_notes.update(record['note_ids'])
                    misses += 1
                    if error > CONTACT_TOLERANCE:
                        reasons.add('contact_reach')
                    if height > HEIGHT_TOLERANCE:
                        reasons.add('contact_height')
                    if not legal:
                        reasons.add('actual_tip_outside_judgement')
                    worst.append(dict(time=t, hand=hand, finger=finger, error_mm=error*1000,
                                      height_mm=height*1000, note_ids=record['note_ids']))
            # Conservative skeleton capsules, not a skin/mesh collision claim.
            segments = {(h, f): _finger_segments(d['joints']) for h, f, d in all_fingers} if active else {}
            for a, (hand, finger, data) in enumerate(all_fingers):
                for other_hand, other_finger, other in all_fingers[a+1:]:
                    if (hand, finger) not in active and (other_hand, other_finger) not in active:
                        continue
                    for first, second in segments[hand, finger]:
                        for third, fourth in segments[other_hand, other_finger]:
                            gap = segment_distance(first, second, third, fourth)
                            penetration = max(penetration, .007-gap)
        if penetration > .003:
            reasons.add('skeleton_collision')
        for positions in flick_positions.values():
            if max((math.dist(positions[0], v) for v in positions), default=0.) < p.physical[0]*.10:
                reasons.add('actual_flick_motion')
        rms = math.sqrt(squared/max(1, samples))
        cost = 100*(rms/.01)**2 + 10*misses/max(1, samples) + 2*travel/.1 + 10*(penetration/.005)**2 + 5*crossed/.02
        occlusion_cost = p.settings.occlusion_weight*hidden_note_seconds
        cost += occlusion_cost
        interaction_cost = p.settings.hand_interaction_weight*density(interaction_integrals)
        cost += interaction_cost
        report = dict(valid=not reasons, reasons=sorted(reasons), cost=cost, samples=samples,
            failed_note_ids=sorted(failed_notes),
            missed_samples=misses, max_error_mm=max_error*1000, rms_error_mm=rms*1000,
            max_height_error_mm=max_height*1000, skeleton_penetration_mm=penetration*1000,
            wrist_crossing_m=crossed, wrist_travel_m=travel, reused_through=best_tick/60,
            seconds=time.monotonic()-started,
            cpu_seconds=time.process_time()-cpu_started,
            timings=dict(setup_seconds=setup_seconds, warmup_seconds=warmup_seconds,
                         warmup_cpu_seconds=warmup_cpu_seconds),
            worst=sorted(worst, key=lambda v: v['error_mm'], reverse=True)[:4],
            model='shared_constrained_skeleton', anatomy_model=ANATOMY_MODEL,
            mesh_validated=False)
        report['occlusion'] = dict(cost=occlusion_cost, weighted_note_seconds=hidden_note_seconds,
            max_weighted_hidden_notes=max_hidden, lookahead_seconds=.65,
            model='top_down_skeleton_and_palm', mesh_validated=False)
        report['interaction'] = dict(cost=interaction_cost,
            projected_crossing_seconds=interaction_integrals[0], palm_intrusion_seconds=interaction_integrals[1],
            tip_inversion_seconds=interaction_integrals[2], twist_radian_seconds=interaction_integrals[3],
            peaks=interaction_peak, model='realised_skeleton_projection', hard_failure=False)
        self.solvers.append(solver)
        self.solvers = self.solvers[-8:]
        self.cache[signature] = report
        if len(self.cache) > MAX_CACHE:
            self.cache.popitem(last=False)
        return report
