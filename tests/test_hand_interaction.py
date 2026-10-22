import copy
import unittest

from flowphi.motion import default_profile
from flowphi.planning.interaction import (contact_relation, frame_relation, relation,
    intersects, density, MAX_CACHE)
from flowphi.planning.planner import Planner, Settings, State
from flowphi.planning.scoring import score_contact
from test_drag_continuation import make_chart
from test_hand_order import record


IDENTITY = ((1.,0.,0.), (0.,1.,0.), (0.,0.,1.))


def hand(joints, palm, x):
    return dict(root=(x, -.1, .05), rotation=IDENTITY, palm=palm,
        fingers={'index': dict(joints=joints, down=True, influence=1.)})


class HandInteractionTests(unittest.TestCase):
    def planner(self, **kwargs):
        return Planner(make_chart([(1,1.,0.,0.), (2,1.15,0.,0.)]),
                       Settings(fingers=('index',), **kwargs), default_profile())

    def test_ordered_wrists_can_have_crossed_fingers_without_3d_collision(self):
        left = [(-.06,.02,.01), (-.02,.06,.01), (.02,.10,.01), (.06,.14,.01)]
        right = [( .06,.02,.08), ( .02,.06,.08), (-.02,.10,.08), (-.06,.14,.08)]
        frame = dict(hands={'left': hand(left, [], -.1), 'right': hand(right, [], .1)})
        self.assertLess(frame['hands']['left']['root'][0], frame['hands']['right']['root'][0])
        self.assertGreater(frame_relation(frame)[0], 0.)
        self.assertGreater(frame_relation(frame)[2], 0.)
        from flowphi.planning.hand_evaluation import segment_distance
        self.assertGreater(segment_distance(left[1], left[2], right[1], right[2]), .05)

    def test_parallel_fingers_are_not_projected_crossings(self):
        self.assertFalse(intersects((-.03,0),(-.03,.1),(.03,0),(.03,.1)))
        self.assertFalse(intersects((0,0),(0,.1),(0,.1),(.1,.1)))

    def test_shaft_through_other_palm_is_found_when_tip_is_outside(self):
        left = dict(fingers={'index': dict(joints=[(-.1,0,0),(.1,0,0)], engagement=1.)}, palm=[])
        right = dict(fingers={'index': dict(joints=[(.2,0,0),(.2,.1,0)], engagement=1.)},
            palm=[(-.04,-.02),(.04,-.02),(.04,.02),(-.04,.02)])
        self.assertGreater(relation(left, right)[1], 0.)

    def test_recent_released_opposite_contact_is_included_during_transfer(self):
        p = self.planner()
        previous = record('left','index',.25,.25,start=.7,end=.8)
        opposite = record('right','index',.45,.45,start=.9,end=1.)
        crossed = record('left','index',.65,.65,start=1.05,end=1.13)
        ordered = dict(crossed, points=[[1.05,.25,.5],[1.13,.25,.5]])
        # Neither contact is DOWN simultaneously. Old wrist-overlap metric is zero.
        self.assertEqual((0.,0.,None), p.hand_crossing(crossed, (previous,opposite)))
        self.assertGreater(contact_relation(p,crossed,(previous,opposite))['cost'],
                           contact_relation(p,ordered,(previous,opposite))['cost'])

    def test_old_live_hold_is_not_lost_behind_finished_contacts(self):
        p = self.planner()
        held = record('right','index',.35,.35,start=-2.,end=4.)
        ended = record('left','index',.25,.25,start=.5,end=.6)
        new = record('left','index',.65,.65)
        self.assertGreater(contact_relation(p,new,(held,ended))['cost'], 0.)

    def test_shared_scorer_and_continuation_keep_interaction_cost(self):
        p = self.planner()
        right = record('right','index',.4,.4,start=.8,end=1.25)
        left = record('left','index',.6,.6,start=1.,end=1.061)
        left.update(kind='tap', pointer=0)
        base = score_contact(p,left,(right,),0.,0.,[])
        self.assertGreater(left['decision']['costs']['interaction'], 0.)
        frozen = copy.deepcopy(left)
        shared = p.share_drag(p.tasks[1],State((right,left),base))
        self.assertTrue(shared)
        merged = next(c for c in shared[0].contacts if 1 in c['note_ids'])
        self.assertIn('interaction',merged['decision']['costs'])
        self.assertEqual(frozen,left)

    def test_bounded_cache_retains_identity_owners(self):
        p = self.planner()
        right = record('right','index',.4,.4,start=.9,end=1.3)
        for i in range(MAX_CACHE+10):
            left = record('left','index',.6,.6,start=1.,end=1.1+i*.00001)
            contact_relation(p,left,(right,))
        self.assertEqual(MAX_CACHE,len(p._interaction_cache))
        for key,(owners,_) in p._interaction_cache.items():
            self.assertEqual(tuple(v[-1] for v in key),tuple(id(p) for p in owners))

    def test_weight_zero_disables_cost_without_changing_constraints(self):
        p, disabled = self.planner(), self.planner(hand_interaction_weight=0.)
        right = record('right','index',.4,.4,start=.9,end=1.3)
        left = record('left','index',.6,.6,start=1.,end=1.1)
        a,b = copy.deepcopy(left),copy.deepcopy(left)
        score_contact(p,a,(right,),0.,0.,[])
        score_contact(disabled,b,(right,),0.,0.,[])
        self.assertEqual(0.,b['decision']['costs']['interaction'])
        self.assertEqual(p.constraints(a,State((right,)),-1,'index'),
                         disabled.constraints(b,State((right,)),-1,'index'))
        for weight in (-1.,float('nan'),float('inf')):
            with self.assertRaises(ValueError):
                Settings(hand_interaction_weight=weight)

    def test_edge_seeds_stay_inside_the_independent_screen_boundary(self):
        from flowphi.planning.contacts import ContactGeometry, validate_contact
        p = self.planner()
        task = p.tasks[0]
        for y in (0.,p.screen.height):
            seed = p.project(p.zone(task,task.beat),(p.screen.width*.5,y))
            points=p.path(task,seed,task.beat+30)
            c=dict(start=task.start/1000,end=(task.beat+31)/1000,kind='tap',note_ids=[task.id])
            g=ContactGeometry(p.chart,physical_screen=p.physical)
            self.assertEqual([],validate_contact(g,c,points))

    def test_current_edited_transfers_are_reported_without_stale_decisions(self):
        from flowphi.ui.validation import hand_interaction_warnings
        from test_hand_order import CrossedHandWarningTests
        left=record('left','index',.75,.75,start=1.05,end=1.20)
        right=record('right','index',.3,.3,start=.9,end=1.)
        doc=CrossedHandWarningTests.Document([right,left])
        warnings=hand_interaction_warnings(doc)
        self.assertTrue(warnings)
        self.assertTrue(all(w['severity']=='warning' for w in warnings))
        left['points']=[[1.05,.2,.5],[1.20,.2,.5]]
        self.assertEqual([],hand_interaction_warnings(doc))


if __name__ == '__main__':
    unittest.main()
