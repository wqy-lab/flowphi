import cmath
import json
import math
import unittest

from shapely.geometry import Point

from flowphi.charts.loader import load_chart
from flowphi.motion import default_profile
from flowphi.planning.planner import Planner, Settings, State
from flowphi.services.candidates import checked_report
from flowphi.ui.generation import generate_draft
from flowphi.ui.model import PlanDocument


def chart_for(kind=1, angle=0., hold=0., move=False, turn=False):
    # 120 BPM: 64 chart ticks = 1 second; a Hold spans seconds 1..3.
    return load_chart(json.dumps(dict(formatVersion=3, offset=0., judgeLineList=[dict(
        bpm=120., notesAbove=[dict(type=kind, time=64, positionX=0., holdTime=hold,
            speed=1., floorPosition=1.)], notesBelow=[],
        speedEvents=[dict(startTime=0, endTime=10000, value=1., floorPosition=0.)],
        judgeLineMoveEvents=[dict(startTime=0, endTime=256, start=.5, end=.5,
                                 start2=.2 if move else .5, end2=.8 if move else .5)],
        judgeLineRotateEvents=[dict(startTime=0, endTime=256, start=angle,
                                   end=angle+50 if turn else angle)],
        judgeLineDisappearEvents=[dict(startTime=0, endTime=10000, start=1., end=1.)],
    )])))


class NormalJudgementTests(unittest.TestCase):
    def planner(self, chart):
        return Planner(chart, Settings(fingers=('index',)), default_profile())

    def test_normal_freedom_for_horizontal_vertical_and_rotated_lines(self):
        for angle in (0., 90., 37.):
            for kind in (1, 2, 3, 4):
                with self.subTest(angle=angle, kind=kind):
                    planner = self.planner(chart_for(kind=kind, angle=angle, hold=128. if kind == 3 else 0.))
                    task = planner.tasks[0]
                    zone = planner.zone(task, task.beat)
                    center = task.line.pos(task.note.seconds, task.note.offset)
                    tangent = cmath.exp(1j * (task.line.angle @ task.note.seconds))
                    for sign in (-1, 1):
                        normal = center + sign * .22 * planner.screen.width * tangent * 1j
                        self.assertTrue(zone.covers(Point(normal.real, normal.imag)))
                    invalid = center + .25 * planner.screen.width * tangent
                    self.assertFalse(zone.covers(Point(invalid.real, invalid.imag)))

    def test_previous_finger_position_remains_a_seed_far_from_line(self):
        planner = self.planner(chart_for())
        old = dict(hand='left', finger='index', start=0., end=.1, points=[[0., .5, .85]])
        seeds = planner.seeds(planner.tasks[0], State((old,)), -1, 'index')
        self.assertTrue(any(math.dist(p, (.5 * planner.screen.width, .85 * planner.screen.height)) < 1e-8
                            for p in seeds))

    def test_moving_line_does_not_drag_stationary_hold_along_normal(self):
        chart = chart_for(kind=3, hold=128., move=True)
        planner = self.planner(chart)
        task = planner.tasks[0]
        seed = (.5 * planner.screen.width, .7 * planner.screen.height)
        points = planner.path(task, seed, task.end)
        self.assertIsNotNone(points)
        self.assertTrue(all(math.dist(p[1:], (.5, .7)) < 1e-8 for p in points))
        draft = generate_draft(chart, planner.settings, planner.profile)
        self.assertTrue(checked_report(PlanDocument(draft), chart)['valid'])
        contact = draft['contacts'][0]
        self.assertTrue(all(math.dist(p[1:], contact['points'][0][1:]) < 1e-8 for p in contact['points']))

    def test_common_region_allows_stationary_hold_when_line_rotates(self):
        planner = self.planner(chart_for(kind=3, hold=128., turn=True))
        task = planner.tasks[0]
        points = planner.path(task, (.5 * planner.screen.width, .7 * planner.screen.height), task.end)
        self.assertIsNotNone(points)
        self.assertTrue(all(math.dist(p[1:], points[0][1:]) < 1e-8 for p in points))
        for t, x, y in points:
            self.assertTrue(planner.zone(task, round(t*1000)).buffer(1e-8).covers(
                Point(x*planner.screen.width, y*planner.screen.height)))

    def test_free_normal_does_not_bypass_screen_or_pause_area(self):
        planner = self.planner(chart_for())
        zone = planner.zone(planner.tasks[0], 1000)
        self.assertFalse(zone.covers(Point(planner.screen.width*.5, -1)))
        self.assertTrue(zone.intersection(planner.pause_poly).area < 1e-8)

    def test_generated_tap_uses_comfortable_off_line_contact(self):
        chart = chart_for()
        planner = self.planner(chart)
        draft = generate_draft(chart, planner.settings, planner.profile)
        contact = draft['contacts'][0]
        self.assertGreater(abs(contact['points'][0][2] - .5), .05)
        self.assertTrue(checked_report(PlanDocument(draft), chart)['valid'])


if __name__ == '__main__':
    unittest.main()
