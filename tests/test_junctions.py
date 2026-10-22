import copy
import math
import unittest
from unittest.mock import patch

from test_drag_continuation import make_chart
from test_physical_plain import motion
from flowphi.motion import default_profile,FINGER_ORDER,point_at,world_xy
from flowphi.planning.planner import Planner,Settings,State
from flowphi.planning.hand_evaluation import HandEvaluator
from flowphi.planning.junctions import proposals,improve,_suffix,ROLLOUT_LIMIT
from flowphi.ui.model import PlanDocument
from flowphi.ui.validation import head_check


class JunctionTests(unittest.TestCase):
    def planner(self,notes):
        p=Planner(make_chart(notes),Settings(fingers=('index',)),default_profile())
        p._hand_evaluator=HandEvaluator(p)
        return p

    def contact(self,p,n,hand):
        c=copy.deepcopy(p.choices(p.tasks[n],State())[0][1])
        c.update(hand=hand,finger='index',pointer=(0 if hand=='left' else 5)+FINGER_ORDER.index('index'))
        return c

    def join(self,p,c,n):
        choices=p.share_drag(p.tasks[n],State((c,),0.))
        self.assertTrue(choices)
        return copy.deepcopy(choices[0].contacts[0])

    def document(self,p,contacts):
        return PlanDocument(dict(version=1,algorithm=5,timebase='chart_seconds',
            coordinates='normalized_screen',contacts=contacts,settings=vars(p.settings),profile=p.profile,
            physical_screen=list(p.physical)))

    def test_simultaneous_yellow_handoff_keeps_prior_blue_downs(self):
        p=self.planner([(1,1.,-4.,0.),(2,1.1,-2.,0.),(1,1.1,3.,0.),
                        (2,1.2,0.,0.),(2,1.2,4.,0.)])
        left=self.join(p,self.contact(p,0,'left'),1)
        right=self.join(p,self.contact(p,2,'right'),3)
        extra=self.contact(p,4,'left');contacts=[left,right,extra];snapshot=copy.deepcopy(contacts)
        values=list(proposals(p,contacts,p.tasks,frozen_before=.9))
        matches=[cs for name,cs in values if name=='simultaneous_drag_handoff' and
                 any(set(c['note_ids'])=={0,1,3} and c['hand']=='left' for c in cs)]
        self.assertTrue(matches)
        for cs in matches:
            self.assertEqual(set(range(5)),{n for c in cs for n in c['note_ids']})
            for original,head in ((left,0),(right,2)):
                c=next(c for c in cs if head in c['note_ids'])
                self.assertEqual(original['start'],c['start'])
                self.assertEqual(original['points'][0],c['points'][0])
            notes=[(t.line,t.note) for _,t in sorted(p.original_tasks.items())]
            self.assertEqual([],head_check(self.document(p,cs),notes))
        self.assertEqual(snapshot,contacts)

    def test_flick_direction_and_yellow_owner_are_joint_candidates(self):
        p=self.planner([(4,1.,2.,0.),(2,1.076,0.,0.)])
        red=self.contact(p,0,'right');yellow=self.contact(p,1,'left');contacts=[red,yellow]
        values=list(proposals(p,contacts,p.tasks,frozen_before=.9))
        matches=[cs for name,cs in values if name=='flick_drag_handoff' and len(cs)==1]
        self.assertTrue(matches)
        for cs in matches:
            c=cs[0]
            self.assertEqual([0,1],c['note_ids'])
            self.assertEqual(red['start'],c['start'])
            end=(p.tasks[0].end)/1000
            distance=math.dist(world_xy(c['points'][0][1:],p.physical),world_xy(point_at(c['points'],end),p.physical))
            self.assertGreaterEqual(distance,p.physical[0]*.10-1e-9)

    def test_committed_flick_stroke_is_never_reoriented(self):
        p=self.planner([(4,1.,2.,0.),(2,1.076,0.,0.)])
        red=self.contact(p,0,'right');yellow=self.contact(p,1,'left')
        self.assertEqual([],list(proposals(p,[red,yellow],p.tasks,frozen_before=red['start']+.001)))

    def test_two_blue_heads_stay_distinct_when_following_yellow_changes_hands(self):
        p=self.planner([(1,1.,-3.,0.),(1,1.,3.,0.),(2,1.125,0.,0.)])
        left=self.join(p,self.contact(p,0,'left'),2);right=self.contact(p,1,'right')
        values=[cs for name,cs in proposals(p,[left,right],p.tasks,frozen_before=.9) if name=='head_drag_handoff']
        self.assertTrue(values)
        notes=[(t.line,t.note) for _,t in sorted(p.original_tasks.items())]
        for cs in values:
            self.assertEqual(2,len(cs))
            self.assertEqual([],head_check(self.document(p,cs),notes))
            self.assertEqual([1.,1.],[c['start'] for c in cs])

    def test_actual_evaluation_budget_is_hard_bounded_even_for_large_generator(self):
        p=self.planner([(1,1.,0.,0.)]);c=self.contact(p,0,'left');before=motion(False)
        with patch('flowphi.planning.junctions.proposals',return_value=iter([('trial',[c])]*100)), \
             patch.object(p._hand_evaluator,'evaluate',return_value=before) as evaluate:
            result,after,report=improve(p,[c],p.tasks,before,frozen_before=.9)
        self.assertLessEqual(evaluate.call_count,ROLLOUT_LIMIT)
        self.assertEqual(report['attempted'],evaluate.call_count)
        self.assertFalse(after['valid'])
        self.assertEqual([c],result)

    def test_removing_yellow_cannot_move_a_committed_up_back_in_time(self):
        p=self.planner([(1,1.,0.,0.),(2,1.125,0.,0.)])
        c=self.join(p,self.contact(p,0,'left'),1)
        self.assertIsNone(_suffix(p,c,[0],1.125,1.1))

    def test_changed_suffix_preserves_all_motion_before_committed_boundary(self):
        p=self.planner([(1,1.,-3.,0.),(2,1.125,0.,0.),(2,1.25,2.,0.)])
        c=self.join(p,self.contact(p,0,'left'),1)
        result=_suffix(p,c,[0,1,2],1.25,1.1)
        self.assertIsNotNone(result)
        for t in (1.,1.01,1.05,1.099):self.assertEqual(point_at(c['points'],t),point_at(result['points'],t))
