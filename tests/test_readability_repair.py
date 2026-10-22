import copy
import unittest
from unittest.mock import patch
from flowphi.planning.readability import acceptable,repair_readability,MAX_ROLLOUTS,handoff_regressions
from flowphi.planning.planner import Planner,Settings
from flowphi.motion import default_profile
from test_drag_continuation import make_chart


def report(valid=False,error=8.,hidden=1.):
    return dict(valid=valid,reasons=[] if valid else ['contact_reach'],failed_note_ids=[] if valid else [0],
        max_error_mm=error,max_height_error_mm=0.,skeleton_penetration_mm=0.,
        interaction=dict(projected_crossing_seconds=0.,palm_intrusion_seconds=0.),
        occlusion=dict(weighted_note_seconds=hidden))


class ReadabilityRepairTests(unittest.TestCase):
    def test_free_gap_cannot_hide_new_upward_handoff(self):
        p=Planner(make_chart([(1,1.,0.,0.)]),Settings(fingers=('index',)),default_profile())
        def contact(start,y,n):
            return dict(hand='right',finger='index',start=start,end=start+.06,
                        note_ids=[n],points=[[start,.7,y]])
        old=[contact(.8,.25,0),contact(1.,.25,1),contact(1.2,.25,2)]
        candidate=copy.deepcopy(old);candidate[1]['points'][0][2]=.85
        snapshot=copy.deepcopy(candidate)
        issues=handoff_regressions(p,old,candidate,.9,1.1)
        self.assertEqual({'entry','exit'},{row['edge'] for row in issues})
        self.assertTrue(all(row['hand']=='right' for row in issues))
        self.assertEqual([2],next(row for row in issues if row['edge']=='exit')['after']['to_note_ids'])
        self.assertEqual(snapshot,candidate)

    def test_coherent_lane_replanning_and_small_handoff_remain_available(self):
        p=Planner(make_chart([(1,1.,0.,0.)]),Settings(fingers=('index',)),default_profile())
        def contact(start,y,n):
            return dict(hand='right',finger='index',start=start,end=start+.06,
                        note_ids=[n],points=[[start,.7,y]])
        old=[contact(.8,.85,0),contact(1.,.25,1),contact(1.2,.85,2)]
        candidate=copy.deepcopy(old);candidate[1]['points'][0][2]=.85
        self.assertEqual([],handoff_regressions(p,old,candidate,.9,1.1))
        old[1]['points'][0][2]=.85;candidate[1]['points'][0][2]=.82
        self.assertEqual([],handoff_regressions(p,old,candidate,.9,1.1))

    def test_boundary_compares_wrist_anchor_across_finger_changes(self):
        from types import SimpleNamespace
        p=SimpleNamespace(anchor=lambda c,t:tuple(c['anchor']))
        old=[dict(hand='right',finger='index',start=.8,end=.86,note_ids=[0],anchor=(0.,0.)),
             dict(hand='right',finger='middle',start=1.,end=1.06,note_ids=[1],anchor=(0.,0.)),
             dict(hand='right',finger='index',start=1.2,end=1.26,note_ids=[2],anchor=(0.,0.))]
        candidate=copy.deepcopy(old);candidate[1]['anchor']=(0.,.08)
        self.assertEqual(2,len(handoff_regressions(p,old,candidate,.9,1.1)))

    def test_pending_transfer_never_bypasses_contact_speed_or_clears_itself(self):
        from flowphi.planning.planner import State
        p=Planner(make_chart([(1,1.,0.,0.)]),Settings(fingers=('index',)),default_profile())
        p._speculative_transfers=True
        with patch.object(p,'constraints',return_value=(['wrist_transfer','contact_speed'],0.,0.)):
            self.assertEqual([],p.choices(p.tasks[0],State()))
        with patch.object(p,'constraints',return_value=(['wrist_transfer'],0.,0.)):
            values=p.choices(p.tasks[0],State())
        self.assertTrue(values)
        for _,c in values:
            self.assertIn('wrist_transfer',c['decision']['degraded'])
            self.assertIn('wrist_transfer',c['decision']['pending_transfer_verification'])
            self.assertNotIn('verified_transfer',c['decision'])
        self.assertEqual(.9,p.settings.wrist_speed)

    def test_new_collision_or_miss_cannot_buy_lower_occlusion(self):
        before=report();after=report(error=2.,hidden=.1)
        after['reasons']=['skeleton_collision']
        self.assertFalse(acceptable(before,after))
        after=report(error=2.,hidden=.1);after['failed_note_ids']=[1]
        self.assertFalse(acceptable(before,after))

    def test_complete_rollout_still_cannot_buy_new_crossing(self):
        after=report(valid=True,error=0.,hidden=.1)
        after['interaction']['projected_crossing_seconds']=.1
        self.assertFalse(acceptable(report(),after))
        after['interaction']['projected_crossing_seconds']=0.
        self.assertTrue(acceptable(report(),after))

    def test_failed_counterfactual_leaves_original_contacts_and_caps(self):
        p=Planner(make_chart([(1,1.,0.,0.),(1,1.2,0.,0.),(1,1.4,0.,0.)]),
                  Settings(fingers=('index',)),default_profile())
        contacts=[dict(hand='left',finger='index',pointer=0,kind='tap',note_ids=[i],
            start=t.start/1000,end=t.start/1000+.06,points=[[t.start/1000,.5,.05]],decision={})
            for i,t in enumerate(p.tasks)]
        original=copy.deepcopy(contacts)
        bad=report(error=1.,hidden=.1);bad['reasons']=['skeleton_collision']
        with patch('flowphi.planning.hand_evaluation.HandEvaluator.evaluate',side_effect=[report(),*[bad]*MAX_ROLLOUTS]), \
             patch('flowphi.ui.validation.validate',return_value=dict(valid=True)):
            result=repair_readability(p,contacts)
        self.assertEqual(original,contacts)
        self.assertFalse(result['windows'][0]['accepted'])
        self.assertLessEqual(result['windows'][0]['evaluated'],MAX_ROLLOUTS)
        self.assertEqual(.9,p.settings.wrist_speed)
        self.assertFalse(getattr(p,'_speculative_transfers',False))
