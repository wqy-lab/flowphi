import math
import random
import unittest
from unittest.mock import patch

import numpy as np

from flowphi.motion import default_profile
from flowphi.planning.contacts import ContactGeometry, project
from flowphi.planning.planner import Planner, Settings, State
from flowphi.planning.ribbons import _transition_errors
from flowphi.planning import ribbons
import test_contact_geometry_cache as geometry_tests
from test_drag_continuation import make_chart


class RibbonSpeedTests(unittest.TestCase):
    def test_neighbour_summary_matches_full_constraints_on_chronological_histories(self):
        rng = random.Random(13751)
        for enabled in (False, True):
            p = Planner(make_chart([(2, 1., 0., 0.)]), Settings(speed_limits=enabled), default_profile())
            history, fingers, hands, occupied = [], {}, {}, {}
            for i in range(160):
                hand = rng.choice(('left', 'right'))
                finger = rng.choice(('index', 'middle', 'ring', 'little', 'thumb'))
                start = i*.031
                duration = rng.choice((.018, .06, .6))
                record = dict(hand=hand, finger=finger, start=start, end=start+duration,
                    points=[[start, rng.random(), rng.random()],
                            [start+duration*.8, rng.random(), rng.random()]])
                key = hand, finger
                expected = p.constraints(record, State(tuple(history)), -1 if hand == 'left' else 1, finger)[0]
                expected = [e for e in expected if e in ('occupied', 'finger_transfer', 'wrist_transfer', 'contact_speed')]
                actual = _transition_errors(p, record, fingers.get(key), hands.get(hand), occupied.get(key, -math.inf))
                self.assertEqual(expected, actual, (enabled, i))
                fingers[key], hands[hand] = record, record
                occupied[key] = max(occupied.get(key, -math.inf), record['end'])
                history.append(record)

    def test_speed_cache_retains_points_and_is_bounded(self):
        p = Planner(make_chart([(2, 1., 0., 0.)]), Settings(), default_profile())
        for i in range(530):
            r = dict(hand='left', finger='index', start=1., end=1.1,
                     points=[[1., .5, .5], [1.09, i/530, .5]])
            _transition_errors(p, r, None, None, -math.inf)
        self.assertEqual(512, len(p._ribbon_speed_cache))
        for identity, (points, _) in p._ribbon_speed_cache.items():
            self.assertEqual(identity, id(points))

    def test_changed_suffix_keeps_all_live_holds_when_skipping_irrelevant_history(self):
        rng = random.Random(5701)
        p = Planner(make_chart([(2, 1., 0., 0.)]), Settings(), default_profile())
        for _ in range(100):
            record = dict(hand='left', finger='index', start=10., end=10.3,
                points=[[10., rng.random(), rng.random()], [10.29, rng.random(), rng.random()]])
            others = [dict(hand=rng.choice(('left', 'right')), finger=rng.choice(('index', 'middle')),
                      start=a, end=b, points=[[a, rng.random(), rng.random()]])
                      for a, b in ((0., 12.), (2., 3.), (10., 10.2), (10.1, 10.5), (12., 13.))]
            filtered = [c for c in others if c['end'] > record['start'] and c['start'] < record['end']]
            full = p.constraints(record, State(tuple(others)), -1, 'index', continuation=True)
            fast = p.constraints(record, State(tuple(filtered)), -1, 'index', continuation=True)
            self.assertEqual(full, fast)

    def test_cached_sample_regions_match_independent_geometry_and_own_their_lists(self):
        chart = geometry_tests.GeometryAccelerationTests().chart()
        reference = ContactGeometry(chart, extra_clearance_m=.004)
        cached = ContactGeometry(chart, extra_clearance_m=.004, cache_judgement=True)
        for flags in ({}, {'joint_note_coverage': True}, {'judgement_times': {'2': 1.251}},
                      {'refinement_region': [.2, .2, .8, .8]},
                      {'joint_note_windows': [dict(start=1.2, end=1.4)]},
                      {'block_clearance_m': .009}):
            record = dict(start=.999, end=1.6, kind='drag', note_ids=[0, 1, 2, 3], **flags)
            times, zones = reference.sample_zones(record, extra_times=(1.2495, 1.40001))
            for _ in range(2):
                actual, regions = cached.sample_zones(record, extra_times=(1.40001, 1.2495))
                np.testing.assert_array_equal(times, actual)
                self.assertEqual([g.wkb for g in zones], [g.wkb for g in regions])
                actual[0] = -10.
                regions[0] = None

    def test_different_finger_seeds_share_regions_but_not_endpoint_mutations(self):
        p = Planner(make_chart([(2, 1+i*.08, -2+i, 0.) for i in range(5)]), Settings(), default_profile())
        geometry = ContactGeometry(p.chart, cache_judgement=True)
        record = dict(start=1., end=1.334, kind='drag', note_ids=list(range(5)), joint_note_coverage=True)
        a = geometry.sample_zones(dict(record, refinement_start=[.4, .6], finger='index', points=[[1., .4, .6]]))
        b = geometry.sample_zones(dict(record, refinement_start=[.7, .8], finger='middle', points=[[1., .7, .8]]))
        self.assertEqual(1, len(geometry._sample_cache))
        self.assertEqual([x.wkb for x in a[1]], [x.wkb for x in b[1]])
        self.assertIsNot(a[1], b[1])

    def test_point_free_projection_matches_original_on_edges_holes_and_fragments(self):
        from shapely.geometry import Point, box
        from shapely.affinity import rotate
        from shapely.ops import nearest_points
        rng = random.Random(2501)
        shapes = (box(.1, .2, .7, .8), rotate(box(.1, .2, .7, .8), 37),
                  box(0, 0, 1, 1).difference(box(.4, .4, .6, .6)),
                  box(.1, .2, .3, .4).union(box(.7, .6, .9, .8)))
        for zone in shapes:
            coordinates = [(rng.random(), rng.random()) for _ in range(100)]
            coordinates.extend((x, y) for x in (0., .1, .4, .6, .7, 1.) for y in (.2, .4, .6, .8))
            for xy in coordinates:
                p = Point(*xy)
                q = p if zone.covers(p) else nearest_points(zone, p)[0]
                np.testing.assert_array_equal([q.x, q.y], project(zone, xy))

    def test_identical_pinned_suffix_is_fitted_once_and_shared_across_branches(self):
        p = Planner(make_chart([(1, 1., 0., 0.), (2, 1.08, 0., 0.), (2, 1.16, 2., 0.),
                               (1, 3., 0., 0.)]), Settings(), default_profile())
        _, old = p.choices(p.tasks[0], State())[0]
        old = dict(old, end=1.12)
        side = -1 if old['hand'] == 'left' else 1
        route = ribbons._route(p, p.tasks[1:3], (side, old['finger']), 'continuous')
        _, future = p.choices(p.tasks[3], State())[0]
        with patch.object(ribbons, 'refine_contact', wraps=ribbons.refine_contact) as fit:
            first = ribbons._attach(p, State((old,)), route)
            second = ribbons._attach(p, State((old, future)), route)
        self.assertTrue(first)
        self.assertTrue(second)
        self.assertEqual(1, fit.call_count)
        self.assertIs(first[-1].contacts[0]['points'], second[-1].contacts[0]['points'])

    def test_missed_immutable_prefix_is_rejected_before_fitting_suffix(self):
        p = Planner(make_chart([(1, 1., -3., 0.), (2, 1.08, 3., 0.), (2, 1.16, 3., 0.)]),
                    Settings(), default_profile())
        _, old = p.choices(p.tasks[0], State())[0]
        old = dict(old, end=1.12)
        side = -1 if old['hand'] == 'left' else 1
        route = ribbons._route(p, p.tasks[1:], (side, old['finger']), 'continuous')
        with patch.object(ribbons, 'refine_contact', wraps=ribbons.refine_contact) as fit:
            self.assertEqual([], ribbons._attach(p, State((old,)), route))
        self.assertEqual(0, fit.call_count)


if __name__ == '__main__':
    unittest.main()
