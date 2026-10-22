import copy
from collections import Counter
import unittest
from unittest.mock import patch

from flowphi.hand_solver import HandMotionSolver
from flowphi.motion import default_profile
from flowphi.planning.hand_evaluation import HandEvaluator, segment_distance
from flowphi.planning.planner import Planner, Settings, State
from flowphi.planning import ribbons
from flowphi.planning.scoring import score_history
from flowphi.ui.model import PlanDocument
from flowphi.ui.pose_preview import PosePreview
from test_drag_continuation import make_chart
from test_hand_limits import document


class HandEvaluationTests(unittest.TestCase):
    def planner(self, notes=None, **settings):
        notes = notes or [(2, 1+i*.08, -2+i, 0.) for i in range(5)]
        return Planner(make_chart(notes), Settings(fingers=('index', 'middle'), beam_width=4, **settings), default_profile())

    def test_planner_and_preview_use_identical_solver(self):
        doc = document()
        shared = HandMotionSolver()
        shared.set_motion(doc.baseline['contacts'], doc.baseline['profile'], doc.baseline['physical_screen'], settings=doc.baseline['settings'])
        preview = PosePreview()
        preview.set_document(doc)
        for t in (1., 1.004, .8, 1.02):
            a, b = shared.frame(t), preview.frame(t)
            self.assertEqual(a['hands'], b['hands'])

    def test_reused_prefix_matches_fresh_rollout_with_future_preparation(self):
        doc = document()
        contacts = copy.deepcopy(doc.baseline['contacts'])
        contacts[1].update(start=3., end=3.061, points=[[3., .1, .6]])
        def solver(values):
            result = HandMotionSolver()
            result.set_motion(values, doc.baseline['profile'], doc.baseline['physical_screen'], settings=doc.baseline['settings'])
            return result
        old = solver(contacts)
        old.frame(2.)
        future = dict(contacts[0], start=5., end=5.2, points=[[5., .5, .6]])
        reused = solver([*contacts, future])
        self.assertGreaterEqual(reused.reuse_prefix(old), 120)
        self.assertEqual(solver([*contacts, future]).frame(5.1), reused.frame(5.1))
        # Moving the new event earlier changes an already-sampled rest interval.
        future = dict(future, start=1.2, end=1.4, points=[[1.2, .5, .6]])
        changed = solver([*contacts, future])
        self.assertLess(changed.reuse_prefix(old), 72)
        self.assertEqual(solver([*contacts, future]).frame(1.25), changed.frame(1.25))

    def test_same_legal_route_fails_when_wrist_cannot_reach_it(self):
        normal = self.planner()
        route = ribbons._route(normal, normal.tasks, (-1, 'index'), 'continuous')
        normal_report = HandEvaluator(normal).evaluate((route,), normal.tasks)
        self.assertTrue(normal_report['valid'], normal_report)
        slow = self.planner(wrist_speed=.01)
        report = HandEvaluator(slow).evaluate((route,), slow.tasks)
        self.assertFalse(report['valid'])
        self.assertIn('contact_reach', report['reasons'])
        self.assertGreater(report['cost'], normal_report['cost'])

    def test_segment_collision_distance_handles_crossing_and_parallel(self):
        self.assertAlmostEqual(0., segment_distance((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0)))
        self.assertAlmostEqual(2., segment_distance((0, 0, 0), (1, 0, 0), (0, 2, 0), (1, 2, 0)))

    def test_occlusion_is_soft_and_keeps_identical_hard_feasibility_samples(self):
        enabled, disabled = self.planner(), self.planner(occlusion_weight=0.)
        route = ribbons._route(enabled, enabled.tasks, (-1, 'index'), 'continuous')
        a = HandEvaluator(enabled).evaluate((route,), enabled.tasks)
        b = HandEvaluator(disabled).evaluate((route,), disabled.tasks)
        for field in ('valid', 'reasons', 'samples', 'missed_samples', 'max_error_mm',
                      'max_height_error_mm', 'skeleton_penetration_mm', 'wrist_travel_m'):
            self.assertEqual(a[field], b[field], field)
        self.assertAlmostEqual(a['cost']-b['cost'], a['occlusion']['cost'])

    def test_interaction_ranking_preserves_every_hard_audit_metric(self):
        enabled, disabled = self.planner(), self.planner(hand_interaction_weight=0.)
        route = ribbons._route(enabled, enabled.tasks, (-1, 'index'), 'continuous')
        a = HandEvaluator(enabled).evaluate((route,), enabled.tasks)
        b = HandEvaluator(disabled).evaluate((route,), disabled.tasks)
        for field in ('valid', 'reasons', 'samples', 'missed_samples', 'max_error_mm',
                      'rms_error_mm', 'max_height_error_mm', 'skeleton_penetration_mm', 'wrist_travel_m'):
            self.assertEqual(a[field], b[field], field)
        self.assertAlmostEqual(a['cost']-b['cost'], a['interaction']['cost'])
        self.assertEqual(0.,b['interaction']['cost'])

    def test_feasible_hand_candidate_beats_cheaper_unreachable_candidate(self):
        p = self.planner()
        a = ribbons._route(p, p.tasks, (-1, 'index'), 'continuous')
        b = ribbons._route(p, p.tasks, (1, 'index'), 'continuous')
        def family(*args, **kwargs):
            return [State((a,), 0.)] if args[3] == 'continuous' else [State((b,), 10.)] if args[3] == 'relay' else []
        def evaluate(self, contacts, tasks, tick=None):
            valid = contacts[0]['hand'] == 'right'
            return dict(valid=valid, cost=0., reasons=[] if valid else ['contact_reach'], max_error_mm=0. if valid else 50.)
        with patch.object(ribbons, '_family_candidates', side_effect=family), \
                patch.object(HandEvaluator, 'evaluate', evaluate):
            states, report, _ = ribbons.plan_window(p, p.tasks, [State()], lambda *a: None)
        self.assertEqual('relay', report['selected'])
        self.assertFalse(report['hand_motion_degraded'])
        self.assertTrue(report['candidates']['continuous']['touch_valid'])
        self.assertFalse(report['candidates']['continuous']['hand_valid'])
        self.assertTrue(all(s.contacts[0]['hand'] == 'right' for s in states))

    def test_no_physical_candidate_is_explicit_draft_without_fake_feasibility(self):
        p = self.planner()
        route = ribbons._route(p, p.tasks, (-1, 'index'), 'continuous')
        with patch.object(ribbons, '_family_candidates', return_value=[State((route,))]), \
                patch.object(HandEvaluator, 'evaluate', return_value=dict(valid=False, cost=10., reasons=['contact_reach'], max_error_mm=30.)), \
                patch.object(p, '_run_plain', side_effect=AssertionError('must retain the whole-window draft')):
            states, report, _ = ribbons.plan_window(p, p.tasks, [State()], lambda *a: None)
        self.assertTrue(states)
        self.assertFalse(report['fallback'])
        self.assertTrue(report['hand_motion_degraded'])
        self.assertFalse(report['hand_motion']['valid'])

    def test_overlapping_touch_can_extend_into_moving_ribbon_without_rewriting_prefix(self):
        p = self.planner([(1, 1., 0., 0.), (2, 1.08, 0., 0.), (2, 1.16, 2., 0.)])
        _, old = p.choices(p.tasks[0], State())[0]
        old = dict(old, end=1.12)
        frozen = copy.deepcopy(old)
        side = -1 if old['hand'] == 'left' else 1
        route = ribbons._route(p, p.tasks[1:], (side, old['finger']), 'continuous')
        attached = ribbons._attach(p, State((old,)), route)
        self.assertTrue(attached)
        merged = attached[-1].contacts[0]
        self.assertEqual(frozen, old)
        self.assertEqual(frozen['points'], merged['points'][:len(frozen['points'])])
        self.assertNotEqual(merged['points'][-1][1:], old['points'][-1][1:])
        self.assertTrue(ribbons._complete(p, attached[-1], p.tasks))

    def test_deferred_head_can_use_finger_before_its_reserved_future_drag(self):
        p = self.planner([(1, 1., -2., 0.), (2, 1.4, 2., 0.)])
        route = ribbons._route(p, p.tasks[1:], (-1, 'index'), 'continuous')
        # Only this finger may be used; a future reservation is not occupation now.
        p.keys = [(-1, 'index')]
        result = ribbons._insert_heads(p, p.tasks, [State((route,))], Counter(), lambda *a: None)
        self.assertTrue(result)
        self.assertTrue(ribbons._heads_ok(p, result[0].contacts, p.tasks))

    def test_physical_window_cost_survives_contact_rescoring(self):
        p = self.planner()
        route = ribbons._route(p, p.tasks, (-1, 'index'), 'continuous')
        base = score_history(p, (route,)).score
        tagged = dict(route, ribbon_decisions=[dict(window=0, hand_motion={'cost': 12.})])
        self.assertAlmostEqual(base+12., score_history(p, (tagged,)).score)


if __name__ == '__main__':
    unittest.main()
