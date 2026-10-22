import copy
import unittest
from unittest.mock import patch

from flowphi.motion import default_profile
from flowphi.planning.planner import Planner,Settings,State
from flowphi.planning.physical_plain import run_ordinary,_touch_check,_trials,ROLLOUT_LIMIT
from flowphi.planning.hand_evaluation import HandEvaluator
from flowphi.planning.scoring import score_history
from test_drag_continuation import make_chart


def motion(valid):
    return dict(valid=valid,reasons=[] if valid else ['contact_reach'],
        failed_note_ids=[] if valid else [0],max_error_mm=0. if valid else 50.,
        max_height_error_mm=0.,skeleton_penetration_mm=0.,cost=0.,
        interaction=dict(projected_crossing_seconds=0.,palm_intrusion_seconds=0.,tip_inversion_seconds=0.))


class PhysicalPlainTests(unittest.TestCase):
    def planner(self,**settings):
        return Planner(make_chart([(1,1.,2.,0.)]),Settings(fingers=('index',),**settings),default_profile())

    def test_physical_candidate_beats_lower_cost_wrong_hand_before_commit(self):
        p=self.planner();base=p.choices(p.tasks[0],State())[0][1]
        a=dict(base,hand='left',pointer=0);b=dict(base,hand='right',pointer=5)
        states=[State((a,),0.),State((b,),20.)]
        calls=[]
        def evaluate(self,contacts,tasks,tick=None,**kwargs):
            calls.append(contacts[0]['hand']);return motion(contacts[0]['hand']=='right')
        with patch.object(p,'_run_plain',return_value=states), \
             patch('flowphi.planning.physical_plain._touch_check',return_value=[]), \
             patch.object(HandEvaluator,'evaluate',evaluate):
            result=run_ordinary(p,return_states=True)
        self.assertTrue(result)
        self.assertTrue(all(s.contacts[0]['hand']=='right' for s in result))
        self.assertIn('left',calls)
        self.assertLessEqual(len(calls),ROLLOUT_LIMIT)
        self.assertFalse(p._physical_plain_search)
        self.assertFalse(p._speculative_transfers)
        self.assertEqual(.9,p.settings.wrist_speed)

    def test_no_physical_candidate_is_not_reported_as_valid(self):
        p=self.planner(allow_degraded=False);record=p.choices(p.tasks[0],State())[0][1]
        pending=[]
        with patch.object(p,'_run_plain',return_value=[State((record,),0.)]), \
             patch('flowphi.planning.physical_plain._touch_check',return_value=[]), \
             patch.object(HandEvaluator,'evaluate',return_value=motion(False)):
            result=run_ordinary(p,return_states=True,on_unplanned=lambda t,r:pending.append(t.id))
        self.assertEqual([0],pending)
        self.assertEqual((),result[0].contacts)
        self.assertFalse(p.ordinary_hand_diagnostics['windows'][0]['valid'])

    def test_physical_proposals_do_not_bypass_continuous_contact_speed(self):
        p=self.planner();p._physical_plain_search=p._speculative_transfers=True
        with patch.object(p,'constraints',return_value=(['wrist_transfer','contact_speed'],0.,0.)):
            self.assertEqual([],p.choices(p.tasks[0],State()))

    def test_real_head_check_rejects_two_heads_using_one_down(self):
        p=Planner(make_chart([(1,1.,0.,0.),(1,1.1,0.,0.)]),Settings(fingers=('index',)),default_profile())
        HandEvaluator(p)
        c=dict(hand='left',finger='index',pointer=0,kind='tap',note_ids=[0,1],
            start=1.,end=1.2,points=[[1.,.5,.5]])
        errors=_touch_check(p,State((c,)),p.tasks,(.9,1.3))
        self.assertIn('missing_down',[e['code'] for e in errors])

    def test_replay_keeps_physical_cost_once_and_separates_search_modes(self):
        p=self.planner();c=p.choices(p.tasks[0],State())[0][1]
        normal=score_history(p,(c,))
        marked=dict(c,ordinary_decisions=[dict(window=0,hand_motion=dict(cost=7.))])
        self.assertAlmostEqual(normal.score+7.,score_history(p,(marked,)).score)
        with patch.object(p,'constraints',return_value=(['wrist_transfer'],0.,0.)):
            p._phrase_score_cache.clear()
            a=score_history(p,(c,))
            p._physical_plain_search=p._speculative_transfers=True
            b=score_history(p,(c,))
            p._physical_plain_search=p._speculative_transfers=False
            again=score_history(p,(c,))
        self.assertAlmostEqual(1000.,a.score-b.score)
        self.assertEqual(a.score,again.score)

    def test_trial_budget_is_bounded(self):
        states=[]
        for i in range(100):
            c=dict(hand='left' if i%2 else 'right',finger='index',note_ids=[0],points=[[1.,i/100,.5]])
            states.append(State((c,),float(i)))
        self.assertLessEqual(len(_trials(states,{0})),ROLLOUT_LIMIT)

    def test_solver_ignores_diagnostics_without_changing_motion(self):
        from flowphi.hand_solver import HandMotionSolver
        p=self.planner();c=p.choices(p.tasks[0],State())[0][1]
        heavy=copy.deepcopy(c);heavy['ordinary_decisions']=[dict(window=1,hand_motion=motion(True))]
        a,b=HandMotionSolver(),HandMotionSolver()
        a.set_motion((c,),p.profile,p.physical,settings=vars(p.settings))
        b.set_motion((heavy,),p.profile,p.physical,settings=vars(p.settings))
        for t in (.8,1.,1.03):self.assertEqual(a.frame(t),b.frame(t))
        self.assertNotIn('ordinary_decisions',b.contacts[0])

    def test_load_cache_matches_exact_integral_after_append_and_is_bounded(self):
        from flowphi.planning.planner import load_at
        p=self.planner();history=[]
        for i in range(40):
            history.append(dict(hand='left',finger='index',kind='hold',start=i*.1,end=i*.1+.2,
                                effort=.055,burst_cost=.01))
            for t in (i*.1,i*.1+.25):
                self.assertEqual(load_at(history,t,'left','index',p.settings),p.load_at(history,t,'left','index'))
                self.assertEqual(p.load_at(history,t,'left','index'),p.load_at(history,t,'left','index'))
        for i in range(600):p.load_at((),i*.1,'right')
        self.assertLessEqual(len(p._load_cache),512)

    def test_uncommitted_tap_can_anticipate_drag_but_preserves_down_and_guard(self):
        from flowphi.motion import point_at
        p=Planner(make_chart([(1,1.,-2.,0.),(2,1.076,2.,0.)]),Settings(fingers=('index',)),default_profile())
        cost,old=p.choices(p.tasks[0],State())[0]
        p._physical_spatial_policy=p._physical_plain_search=p._speculative_transfers=True
        p._physical_window_start=1.
        values=p.share_drag(p.tasks[1],State((old,),cost))
        self.assertTrue(values)
        c=values[0].contacts[0]
        self.assertEqual(old['start'],c['start'])
        self.assertEqual(old['points'][0],c['points'][0])
        self.assertEqual(point_at(old['points'],1.01),point_at(c['points'],1.01))
        self.assertNotEqual(point_at(old['points'],1.05),point_at(c['points'],1.05))
        self.assertEqual([0],old['note_ids'])
        p._physical_window_start=1.01
        for state in p.share_drag(p.tasks[1],State((old,),cost)):
            self.assertEqual(old['points'],state.contacts[0]['points'][:len(old['points'])])

    def test_fractional_tap_and_repeated_drag_extension_keep_strict_point_times(self):
        from flowphi.ui.model import check_points
        p=Planner(make_chart([(1,1.0004,-2.,0.),(2,1.0764,0.,0.),(2,1.1524,2.,0.)]),
                  Settings(fingers=('index',)),default_profile())
        cost,c=p.choices(p.tasks[0],State())[0]
        p._physical_spatial_policy=p._physical_plain_search=p._speculative_transfers=True
        p._physical_window_start=1.
        states=[State((c,),cost)]
        for task in p.tasks[1:]:
            states=[child for s in states for child in p.share_drag(task,s)]
            self.assertTrue(states)
            for s in states:
                for c in s.contacts:check_points(c,c['points'])

    def test_malformed_candidate_never_reaches_skeleton_or_output(self):
        p=self.planner();c=p.choices(p.tasks[0],State())[0][1]
        c=dict(c,points=[c['points'][0],c['points'][0]])
        pending=[]
        with patch.object(p,'_run_plain',return_value=[State((c,),0.)]), \
             patch.object(HandEvaluator,'evaluate') as evaluate:
            states=run_ordinary(p,return_states=True,on_unplanned=lambda t,r:pending.append(t.id))
        evaluate.assert_not_called()
        self.assertEqual([0],pending)
        self.assertEqual((),states[0].contacts)
        self.assertEqual('invalid_controls',p.ordinary_hand_diagnostics['windows'][0]['structural_rejections'][0]['code'])

    def test_certified_projection_preserves_notches_holes_and_closed_edges(self):
        import numpy as np
        from shapely.geometry import Point,box
        from shapely.ops import nearest_points
        from flowphi.planning.contacts import ContactGeometry,project
        p=self.planner();screen=ContactGeometry(p.chart).screen.buffer(-2e-5)
        zones=[screen,box(.2,.1,.4,.95),screen.difference(box(.4,.7,.6,.9)),Point(.2,.8)]
        points=[(.01,.05),(.05,.88),(.95,.88),(.5,.88),(.5,.8),(.988-.00002,.5),(.3,.05)]
        for zone in zones:
            for xy in points:
                expected=nearest_points(zone,Point(*xy))[0]
                np.testing.assert_allclose(project(zone,xy),(expected.x,expected.y),atol=1e-14,rtol=0)

    def test_exact_drag_tail_cache_uses_same_proof_and_different_timing_misses(self):
        from flowphi.planning import drag_continuation as d
        from flowphi.planning.contacts import ContactGeometry
        p=self.planner();g=ContactGeometry(p.chart,cache_judgement=True)
        tail=dict(kind='drag',note_ids=[0],start=1.,end=1.061,
                  points=[[1.,.5,.5],[1.06,.5,.5]],refinement_start=[.5,.5])
        with patch.object(d,'refine_contact',wraps=d.refine_contact) as fit:
            a=d._refined_tail(p,g,tail)
            b=d._refined_tail(p,g,dict(tail,hand='right'))
            self.assertEqual(a,b)
            self.assertEqual(1,fit.call_count)
            d._refined_tail(p,g,dict(tail,end=1.062))
            self.assertEqual(2,fit.call_count)

    def test_fractional_future_head_at_exclusive_end_is_not_missing_current_down(self):
        p=Planner(make_chart([(1,1.,0.,0.),(1,1.0756,0.,0.)]),Settings(fingers=('index',)),default_profile())
        HandEvaluator(p)
        cost,c=p.choices(p.tasks[0],State())[0]
        # The future beat is 1.0756, its rounded DOWN is 1.076, outside this window.
        self.assertEqual(1076,p.tasks[1].start)
        errors=_touch_check(p,State((c,),cost),p.tasks[:1],(.9,1.076))
        self.assertEqual([],errors)
        missing=_touch_check(p,State((c,),cost),p.tasks,(.9,1.2))
        self.assertIn('missing_down',[e['code'] for e in missing])
