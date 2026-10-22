import copy
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from flowphi import hand_limits as limits
from flowphi.hand_solver import HandMotionSolver
from flowphi.motion import default_profile,stroke_lift
from flowphi.planning.planner import Planner,Settings,State
from flowphi.planning.hand_evaluation import HandEvaluator
from flowphi.planning.repair import _occupancy_conflicts,_placements,repair_failed_windows
from flowphi.ui.pose_preview import PosePreview
from test_drag_continuation import make_chart
from test_hand_limits import document


def report(valid=True):
    return dict(valid=valid,reasons=[] if valid else ['contact_reach'],max_error_mm=0. if valid else 8.,
        max_height_error_mm=0.,skeleton_penetration_mm=0.,cost=0.,failed_note_ids=[])


class HandRepairTests(unittest.TestCase):
    def test_swap_lead_fixes_previous_hands_without_swapping_the_failed_landing(self):
        p=Planner(make_chart([(1,1.,2.,0.),(1,1.2,-2.,0.),(1,1.4,2.,0.)]),
            Settings(fingers=('index',)),default_profile())
        contacts=[]
        for i,t in enumerate(p.tasks):
            x=t.line.pos(t.note.seconds,t.note.offset).real/p.screen.width
            hand='left' if i==0 else 'right'
            contacts.append(dict(hand=hand,finger='index',pointer=0 if hand=='left' else 5,kind='tap',
                note_ids=[t.id],start=t.start/1000,end=t.start/1000+.061,points=[[t.start/1000,x,.8]]))
        m=report(False);m['worst']=[dict(note_ids=[2],hand='right')]
        proposals=list(_placements(p,contacts,p.tasks,m))
        choice=next(values for name,values,changed in proposals if name.startswith('swap_lead:(1.0,'))
        self.assertEqual(['right','left','right'],[c['hand'] for c in choice])
        self.assertEqual(contacts[2],choice[2])
        self.assertEqual(['left','right','right'],[c['hand'] for c in contacts])

    def test_stroke_reserves_lateral_speed_and_compressed_downstroke(self):
        job=dict(palm_lift_mode='gentle',palm_lift_ratio=.7,stroke_peak_phase=.62,
            palm_strike_speed=1.8,wrist_speed=.9,stroke_distance_m=.025)
        gap=.05
        lift=stroke_lift({},gap,job)
        derivative=math.pi/(2*.38)
        xy=1.25*.025/gap
        z=lift*.7*derivative/gap
        self.assertLessEqual(math.hypot(xy,z),.9+1e-10)
        self.assertLess(lift,stroke_lift({},gap,dict(job,stroke_distance_m=0.)))

    def test_exact_down_node_and_every_substep_keep_all_original_caps(self):
        preview=PosePreview()
        preview.set_document(document())
        frame=preview.frame(1.004)
        nodes=preview._cell(61,preview._state(60))
        event=next(state for t,state in nodes if abs(t-1.004)<1e-10)
        for hand in ('left','right'):
            self.assertLess(math.dist(frame['hands'][hand]['root'],event[hand]['root']),1e-12)
        for (ta,a),(tb,b) in zip(nodes,nodes[1:]):
            dt=tb-ta
            for hand in ('left','right'):
                self.assertLessEqual(math.dist(a[hand]['root'],b[hand]['root']),.4*dt+1e-10)
                self.assertLessEqual(limits.angular_distance(a[hand]['rotation'],b[hand]['rotation']),4*dt+1e-10)
                for finger in a[hand]['fingers']:
                    for i,(qa,qb) in enumerate(zip(a[hand]['fingers'][finger],b[hand]['fingers'][finger])):
                        self.assertLessEqual(limits.angular_distance(qa,qb),(8 if i==0 else 1.5)*dt+1e-10)
                        if i:
                            self.assertLessEqual(limits.angular_distance(limits.IDENTITY,qb),.2+1e-10)

    def test_coarse_transfer_rejection_can_be_verified_without_raising_caps(self):
        p=Planner(make_chart([(1,1.,-.5,0.),(1,1.12,1.5,0.)]),Settings(fingers=('index',)),default_profile())
        contacts=[]
        for task in p.tasks:
            x=task.line.pos(task.note.seconds,task.note.offset).real/p.screen.width
            contacts.append(dict(hand='left',finger='index',pointer=0,kind='tap',note_ids=[task.id],
                start=task.start/1000,end=task.start/1000+.061,points=[[task.start/1000,x,.65]]))
        violations=p.constraints(contacts[1],State((contacts[0],)),-1,'index')[0]
        self.assertIn('finger_transfer',violations)
        self.assertIn('wrist_transfer',violations)
        motion=HandEvaluator(p).evaluate(contacts,p.tasks,time_range=(1.,1.181))
        self.assertTrue(motion['valid'],motion)
        self.assertLess(motion['max_error_mm'],4.)
        self.assertEqual(.9,p.settings.wrist_speed)

    def test_existing_remote_draft_conflict_does_not_block_legal_local_placements(self):
        p=Planner(make_chart([(1,.1,0.,0.),(1,.1,0.,0.),(1,1.,0.,0.)]),Settings(fingers=('index',)),default_profile())
        contacts=[dict(hand='left' if i<2 else 'right',finger='index',pointer=0 if i<2 else 5,
            kind='tap',note_ids=[i],start=t.note.seconds,end=t.note.seconds+.061,points=[[t.note.seconds,.5,.6]])
            for i,t in enumerate(p.tasks)]
        frozen=copy.deepcopy(contacts)
        motion=report(False)
        motion['worst']=[]
        proposals=list(_placements(p,contacts,[p.tasks[-1]],motion))
        self.assertTrue(proposals)
        for _,candidate,_ in proposals:
            self.assertTrue(_occupancy_conflicts(candidate)<=_occupancy_conflicts(contacts))
            self.assertEqual([(c['start'],c['end'],c['note_ids']) for c in candidate],
                             [(c['start'],c['end'],c['note_ids']) for c in contacts])
        self.assertEqual(frozen,contacts)

    def test_previously_valid_window_cannot_be_sacrificed(self):
        p=Planner(make_chart([(2,1.,0.,0.),(2,2.,0.,0.)]),Settings(fingers=('index',)),default_profile())
        bad=dict(id=0,start=1.,end=1.1,note_ids=[0],hand_motion=report(False))
        good=dict(id=1,start=2.,end=2.1,note_ids=[1],hand_motion=report())
        p.ribbon_diagnostics=dict(windows=[bad,good])
        contacts=[dict(hand='left',finger='index',pointer=0,kind='drag',start=1.,end=1.1,
            points=[[1.,.5,.5]],note_ids=[0])]
        alternative=copy.deepcopy(contacts)
        alternative[0]['points'][0][1]=.6
        def evaluate(values,tasks,tick=None,**kwargs):
            return report(True) if kwargs or tasks[0].id==0 or values[0]['points'][0][1]==.5 else report(False)
        p._hand_evaluator=SimpleNamespace(evaluate=evaluate)
        with patch('flowphi.planning.repair._placements',return_value=iter([('test',alternative,[0])])):
            summary=repair_failed_windows(p,contacts,windows=[bad])
        self.assertEqual(0,summary['accepted'])
        self.assertEqual(.5,contacts[0]['points'][0][1])
        self.assertTrue(good['hand_motion']['valid'])

    def test_impossible_contact_state_is_reported_not_fitted_away(self):
        p=SimpleNamespace(_hand_evaluator=None)
        bad=dict(id=0,start=1.,end=2.,hand_motion=dict(report(False),reasons=['contact_state']))
        p.ribbon_diagnostics=dict(windows=[bad])
        summary=repair_failed_windows(p,[])
        self.assertEqual(0,summary['windows'][0]['attempted'])
        self.assertEqual(1,summary['remaining'])

    def test_physically_valid_but_newly_crossed_candidate_is_rejected(self):
        p=Planner(make_chart([(2,1.,0.,0.)]),Settings(fingers=('index',)),default_profile())
        bad=dict(id=0,start=1.,end=1.1,note_ids=[0],hand_motion=report(False))
        p.ribbon_diagnostics=dict(windows=[bad])
        contacts=[dict(hand='left',finger='index',pointer=0,kind='drag',start=1.,end=1.1,
            points=[[1.,.5,.5]],note_ids=[0])]
        alternative=copy.deepcopy(contacts)
        alternative[0]['points'][0][1]=.6
        crossed=dict(report(),interaction=dict(projected_crossing_seconds=.2))
        p._hand_evaluator=SimpleNamespace(evaluate=lambda *a,**k:crossed)
        with patch('flowphi.planning.repair._placements',return_value=iter([('test',alternative,[0])])):
            summary=repair_failed_windows(p,contacts)
        self.assertEqual(0,summary['accepted'])
        self.assertEqual(.5,contacts[0]['points'][0][1])

    def test_expected_hold_cannot_be_hidden_by_zero_preview_error_while_up(self):
        p=Planner(make_chart([(3,1.,0.,1.),(1,1.1,0.,0.)]),Settings(fingers=('index',)),default_profile())
        contacts=[dict(hand='left',finger='index',pointer=0,kind=kind,note_ids=[i],start=a,end=b,
            points=[[a,.5,.6]]) for i,kind,a,b in ((0,'hold',1.,2.),(1,'tap',1.1,1.161))]
        solver=HandMotionSolver()
        solver.set_motion(contacts,p.profile,p.physical,settings=vars(p.settings))
        value=solver.frame(1.2)['hands']['left']['fingers']['index']
        self.assertFalse(value['down'])
        self.assertEqual(0.,value['error'])
        result=HandEvaluator(p).evaluate(contacts,p.tasks,time_range=(1.2,1.3))
        self.assertFalse(result['valid'])
        self.assertIn('contact_state',result['reasons'])

    def test_interval_audit_does_not_sample_old_hold_head_outside_requested_range(self):
        p=Planner(make_chart([(3,.5,0.,2.)]),Settings(fingers=('index',)),default_profile())
        c=dict(hand='left',finger='index',pointer=0,kind='hold',start=.5,end=2.5,
            note_ids=[0],points=[[.5,.5,.6],[2.499,.5,.6]])
        times=[]
        original=HandMotionSolver.frame
        def frame(solver,t):
            times.append(t)
            return original(solver,t)
        with patch.object(HandMotionSolver,'frame',frame):
            HandEvaluator(p).evaluate([c],p.tasks,time_range=(1.2,1.3))
        self.assertGreaterEqual(min(times),1.2)
        self.assertLessEqual(max(times),1.3+.18)


if __name__=='__main__':
    unittest.main()
