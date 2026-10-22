import math
import unittest

from flowphi import hand_limits as limits
from flowphi.motion import default_profile
from flowphi.pose import rotation_matrix
from flowphi.ui.model import PlanDocument
from flowphi.ui.pose_preview import PosePreview


def document():
    contacts = [dict(hand='left', finger='middle', kind='hold', start=.5, end=1., note_ids=[0],
                     points=[[.5, .9, .6], [.999, .9, .6]]),
                dict(hand='left', finger='ring', kind='tap', start=1.004, end=1.065, note_ids=[1],
                     points=[[1.004, .1, .6], [1.064, .1, .6]])]
    return PlanDocument(dict(version=1, algorithm=5, timebase='chart_seconds', coordinates='normalized_screen',
        profile=default_profile(), settings=dict(fingers=['middle', 'ring'], wrist_speed=.4),
        physical_screen=[.28, .1575], palm_motion='v4', contacts=contacts))


class HandLimitTests(unittest.TestCase):
    def test_saved_old_hand_proof_is_not_presented_as_new_anatomy_proof(self):
        from test_drag_continuation import make_chart
        from flowphi.ui.validation import validate
        doc = document()
        motion = dict(valid=True, model='shared_constrained_skeleton')
        doc.baseline['diagnostics'] = dict(ordinary_hand_search=dict(windows=[
            dict(start=.5, end=1.1, note_ids=[0, 1], final_motion=motion)]))
        chart = make_chart([(3, .5, 0., .5), (1, 1.004, 0., 0.)])
        report = validate(doc, chart, geometry=False)
        self.assertIn('outdated_hand_anatomy', [p['code'] for p in report['issues']])
        motion['anatomy_model'] = limits.ANATOMY_MODEL
        report = validate(doc, chart, geometry=False)
        self.assertNotIn('outdated_hand_anatomy', [p['code'] for p in report['issues']])

    def test_distal_rejects_sideways_and_longitudinal_twist_on_both_hands(self):
        from flowphi.hand_solver import HandMotionSolver
        solver = HandMotionSolver()
        solver.set_motion([], default_profile(), (.28, .1575))
        for (side, finger), models in solver.joint_models.items():
            for joint in (1, 2):
                model = models[joint]
                for axis in (model.spread, model.forward):
                    desired = limits.axis_rotation(axis, .8)
                    q = limits.IDENTITY
                    for _ in range(120):
                        q = limits.joint_rotation(q, desired, limits.IDENTITY, joint, 1/60, model)
                    self.assertLess(limits.angular_distance(q, limits.IDENTITY), 1e-10, (side, finger, joint))

    def test_root_spread_envelope_no_twist_and_continuous_speed_cap(self):
        for finger in limits.MCP_SPREAD:
            model = limits.joint_model(finger, 0)
            q = limits.IDENTITY
            for sign in (1, -1, 1):
                desired = limits.multiply(limits.axis_rotation(model.spread, sign*1.3),
                                          limits.axis_rotation(model.flex, sign*.8))
                for dt in (1/60, .00001, .003, 1/60)*30:
                    next_q = limits.joint_rotation(q, desired, limits.IDENTITY, 0, dt, model)
                    last = q
                    for n in range(1, 8):
                        value = limits.joint_interpolate(q, next_q, n/7, limits.IDENTITY, model)
                        flex, spread = limits.joint_angles(value, limits.IDENTITY, model)
                        self.assertLessEqual(abs(spread), model.lateral+1e-10)
                        self.assertTrue(model.minimum-1e-10 <= flex <= model.maximum+1e-10)
                        self.assertLess(limits.angular_distance(value, limits.project_joint(value, limits.IDENTITY, model)), 1e-9)
                        self.assertLessEqual(limits.angular_distance(last, value), limits.MCP_SPEED*dt/7+1e-9)
                        last = value
                    q = next_q

    def test_rotated_axes_and_nonidentity_relaxed_reference(self):
        basis = limits.matrix_quaternion(rotation_matrix(.8, .3, -.5))
        model = limits.joint_model('middle', 1,
            flex_axis=limits.apply(basis, (1., 0., 0.)), spread_axis=limits.apply(basis, (0., 0., 1.)))
        reference = limits.matrix_quaternion(rotation_matrix(-.4, .6, .2))
        desired = limits.multiply(reference, limits.axis_rotation(model.flex, -.8))
        q = reference
        for _ in range(60):
            value = limits.joint_rotation(q, desired, reference, 1, 1/60, model)
            self.assertLessEqual(limits.angular_distance(q, value), limits.DISTAL_SPEED/60+1e-10)
            q = value
        self.assertAlmostEqual(limits.DISTAL_EXCURSION, limits.angular_distance(reference, q))
        self.assertLess(limits.angular_distance(q, limits.joint_pose(reference, model, (-.2, 0.))), 1e-10)

    def test_solver_checkpoints_and_subframes_stay_on_anatomical_axes(self):
        from flowphi.hand_solver import HandMotionSolver
        solver = HandMotionSolver()
        contacts = [dict(hand='left', finger=finger, kind='tap', note_ids=[i], start=.6+i*.1,
                         end=.68+i*.1, points=[[.6+i*.1, x, .75]])
                    for i, (finger, x) in enumerate((('index', .9), ('middle', .1), ('index', .3)))]
        solver.set_motion(contacts, default_profile(), (.28, .1575))
        solver.frame(1.01)
        for t in (.50123, .60007, .69999, .80013, .90917):
            frame = solver.frame(t)
            for side, hand in ((-1, 'left'), (1, 'right')):
                palm = frame['hands'][hand]
                inverse = limits.conjugate(limits.matrix_quaternion(palm['rotation']))
                for finger, data in palm['fingers'].items():
                    local = [limits.apply(inverse, tuple(p-r for p, r in zip(point, palm['root'])))
                             for point in data['joints']]
                    rest = solver.rest_chains[side, finger]
                    parent, qs = limits.IDENTITY, []
                    for j, model in enumerate(solver.joint_models[side, finger]):
                        segment = tuple(b-a for a, b in zip(rest[j], rest[j+1]))
                        actual = tuple(b-a for a, b in zip(local[j], local[j+1]))
                        desired = limits.between(segment, limits.apply(limits.conjugate(parent), actual))
                        q = limits.project_joint(desired, limits.IDENTITY, model)
                        qs.append(q)
                        parent = limits.multiply(parent, q)
                    rebuilt = limits.forward_chain(rest, qs)
                    self.assertLess(max(math.dist(a, b) for a, b in zip(local, rebuilt)), 1e-9)
        for states in (solver._states.values(), (s for nodes in solver._substates.values() for _, s in nodes)):
            for state in states:
                for side, hand in ((-1, 'left'), (1, 'right')):
                    for finger, joints in state[hand]['fingers'].items():
                        for joint, q in enumerate(joints):
                            model = solver.joint_models[side, finger][joint]
                            self.assertLess(limits.angular_distance(q, limits.project_joint(q, limits.IDENTITY, model)), 1e-9)

    def test_position_cap_includes_vertical_and_tiny_event_gaps(self):
        for dt in (.00001, .004, 1/60):
            value = limits.bounded_position((0., 0., 0.), (.2, .1, .4), .4, dt)
            self.assertAlmostEqual(math.dist(value, (0., 0., 0.)), .4*dt)

    def test_quaternion_roundtrip_including_half_turns(self):
        for angles in ((.3, -.7, .4), (math.pi, 0., 0.), (0., math.pi, 0.), (0., 0., math.pi)):
            matrix = rotation_matrix(*angles)
            actual = limits.quaternion_matrix(limits.matrix_quaternion(matrix))
            for a, b in zip(matrix, actual):
                for x, y in zip(a, b):
                    self.assertAlmostEqual(x, y)

    def test_distal_excursion_and_angular_speed_cannot_be_bypassed(self):
        desired = limits.matrix_quaternion(rotation_matrix(0., -1.5, 0.))
        previous = limits.IDENTITY
        for _ in range(60):
            q = limits.joint_rotation(previous, desired, limits.IDENTITY, 1, 1/60)
            self.assertLessEqual(limits.angular_distance(previous, q), limits.DISTAL_SPEED/60 + 1e-8)
            self.assertLessEqual(limits.angular_distance(limits.IDENTITY, q), limits.DISTAL_EXCURSION + 1e-8)
            previous = q
        self.assertAlmostEqual(limits.angular_distance(limits.IDENTITY, q), limits.DISTAL_EXCURSION)
        # Quaternion signs represent the same orientation and must not cause a snap.
        self.assertAlmostEqual(0., limits.angular_distance(q, tuple(-v for v in q)))

    def test_chain_reconstruction_preserves_lengths(self):
        rest = [(0., 0., 0.), (0., .04, -.01), (0., .07, -.025), (0., .09, -.045)]
        desired = [limits.apply(limits.matrix_quaternion(rotation_matrix(.6, -.4, .1)), p) for p in rest]
        actual = limits.forward_chain(rest, limits.chain_rotations(rest, desired))
        for a, b in zip(actual, desired):
            self.assertLess(math.dist(a, b), 1e-9)

    def test_preview_caps_wrist_reports_miss_and_is_seek_deterministic(self):
        preview = PosePreview()
        preview.set_document(document())
        first = preview.frame(1.)
        second = preview.frame(1.004)
        self.assertLessEqual(math.dist(first['hands']['left']['root'], second['hands']['left']['root']), .4*.004+1e-9)
        self.assertIn('触点偏差', second['warning'])
        other = PosePreview()
        other.set_document(document())
        direct = other.frame(1.004)
        self.assertEqual(second, direct)
        preview.frame(.8)
        self.assertEqual(second, preview.frame(1.004))
        for hand in second['hands'].values():
            for finger in hand['fingers'].values():
                self.assertTrue(all(math.isfinite(v) for p in finger['joints'] for v in p))

    def test_pose_change_invalidates_motion_history(self):
        preview = PosePreview()
        preview.set_document(document())
        preview.frame(1.)
        self.assertTrue(preview._checkpoints)
        preview.set_pose(None)
        self.assertFalse(preview._checkpoints)


if __name__ == '__main__':
    unittest.main()
