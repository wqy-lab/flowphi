import random
import unittest

from flowphi.charts.blocks import BlockAreas
from flowphi.charts.model import Note, NoteType
from flowphi.motion import default_profile
from flowphi.planning.contacts import ContactGeometry, validate_contact
from flowphi.planning.drag_continuation import _window_valid
from flowphi.planning.planner import Planner, Settings
from test_normal_judgement import chart_for


class GeometryAccelerationTests(unittest.TestCase):
    def chart(self):
        chart = chart_for(kind=3, hold=128., angle=37., turn=True, move=True)
        chart.lines[0].notes.extend([
            Note(NoteType.TAP, 1.1, 0., 0j),
            Note(NoteType.DRAG, 1.25, 0., 0j),
            Note(NoteType.FLICK, 1.5, 0., 80+0j),
        ])
        chart.block_areas = BlockAreas([dict(
            bottomLeftPercentage=dict(x=.4, y=.4), topRightPercentage=dict(x=.6, y=.6),
            appearTime=1.2, enableTime=1.2, disableTime=1.4, disappearTime=1.4,
        )], chart.width, chart.height)
        return chart

    def test_fast_geometry_matches_polygon_path_with_motion_blocks_and_boundaries(self):
        chart = self.chart()
        reference = ContactGeometry(chart, extra_clearance_m=.004)
        accelerated = ContactGeometry(chart, extra_clearance_m=.004, cache_judgement=True)
        rng = random.Random(731)
        points = [(rng.random(), rng.random()) for _ in range(40)]
        points += [(x, y) for x in (.012-1e-10, .012, .1, .5, .9, .988, .988+1e-10)
                   for y in (.012, .1, .5, .988)]
        for flags in ({}, dict(joint_note_coverage=True),
                      dict(joint_note_coverage=True, refinement_region=[.3, .3, .7, .7]),
                      dict(joint_note_windows=[dict(start=1.2, end=1.4)]),
                      dict(judgement_times={'2': 1.251})):
            contact = dict(note_ids=[0, 1, 2, 3], **flags)
            for t in (.999, 1., 1.1, 1.199999, 1.2, 1.249, 1.25, 1.251, 1.26, 1.4, 1.5, 2.99):
                for xy in points:
                    self.assertEqual(reference.allows_point(contact, t, xy),
                                     accelerated.allows_point(contact, t, xy), (flags, t, xy))

    def test_window_cache_matches_full_history_validation(self):
        chart = self.chart()
        planner = Planner(chart, Settings(), default_profile())
        reference = ContactGeometry(chart, extra_clearance_m=.004)
        accelerated = ContactGeometry(chart, extra_clearance_m=.004, cache_judgement=True)
        for low in (.999, 1.09, 1.24, 1.255, 1.49, 2.9):
            for xy in ((.5, .5), (.6, .4), (.05, .05)):
                points = [[.99, *xy], [3., *xy]]
                contact = dict(start=low, end=low+.014, note_ids=[0, 1, 2, 3],
                               joint_note_coverage=True, judgement_times={'2': 1.251})
                expected = not validate_contact(reference, contact, points, limit=1)
                for _ in range(2):
                    self.assertEqual(expected, _window_valid(planner, accelerated, contact, points))

    def test_strip_edge_falls_back_without_changing_tolerance(self):
        import cmath
        from flowphi.planning.geometry import JUDGE_HALF_TAP
        chart = chart_for(kind=1, angle=37.)
        reference = ContactGeometry(chart)
        accelerated = ContactGeometry(chart, cache_judgement=True)
        line, note = chart.lines[0], chart.lines[0].notes[0]
        center = line.pos(1., note.offset)
        tangent = cmath.exp(1j*(line.angle @ 1.))
        for delta in (-1e-6, -1e-10, 0., 1e-10, 1e-6):
            p = center + tangent * chart.width * (JUDGE_HALF_TAP*.5 + delta)
            xy = p.real/chart.width, p.imag/chart.height
            contact = dict(note_ids=[0])
            self.assertEqual(reference.allows_point(contact, 1., xy),
                             accelerated.allows_point(contact, 1., xy))
