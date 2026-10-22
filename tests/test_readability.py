import math
import unittest
from types import SimpleNamespace
from shapely.geometry import Point
from flowphi.planning.scoring import line_distance_cost
from flowphi.planning.planner import Planner,Settings
from flowphi.planning.contacts import ContactGeometry
from flowphi.motion import default_profile
from test_drag_continuation import make_chart


class ReadabilityTests(unittest.TestCase):
    def test_preference_keeps_far_vertical_judgement_legal(self):
        chart=make_chart([(1,1.,0.,0.)]);p=Planner(chart,Settings(),default_profile())
        c=dict(kind='tap',note_ids=[0],start=1.,end=1.06,points=[[1.,.5,.5]])
        far=dict(c,points=[[1.,.5,.05]])
        g=ContactGeometry(chart)
        self.assertTrue(g.zone(c,1.).covers(Point(.5,.5)))
        self.assertTrue(g.zone(far,1.).covers(Point(.5,.05)))
        self.assertLess(line_distance_cost(p,c),line_distance_cost(p,far))

    def test_vertical_line_uses_normal_x_not_screen_y(self):
        class Angle:
            def __matmul__(self,t):return math.pi/2
        note=SimpleNamespace(seconds=1.,offset=0.,type=1)
        line=SimpleNamespace(angle=Angle(),pos=lambda t,offset:500+300j)
        p=SimpleNamespace(original_tasks={0:SimpleNamespace(note=note,line=line)},
            screen=SimpleNamespace(width=1000,height=600),physical=(.28,.168))
        c=dict(kind='tap',note_ids=[0],start=1.,end=1.06,points=[[1.,.5,.1]])
        self.assertAlmostEqual(0.,line_distance_cost(p,c))
        self.assertGreater(line_distance_cost(p,dict(c,points=[[1.,.8,.5]])),0.)

    def test_mandatory_flick_travel_is_not_a_normal_offset(self):
        p=Planner(make_chart([(4,1.,0.,0.)]),Settings(),default_profile())
        c=dict(kind='flick',note_ids=[0],start=1.,end=1.07,points=[[1.,.5,.5],[1.06,.5,.8]])
        self.assertEqual(line_distance_cost(p,c),line_distance_cost(p,dict(c,points=[[1.,.5,.5],[1.06,.5,.9]])))

    def test_bad_line_preference_is_rejected(self):
        for value in (-1.,float('nan'),float('inf')):
            with self.assertRaises(ValueError):Settings(judge_line_weight=value)
