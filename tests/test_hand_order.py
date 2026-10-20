import unittest

from flowphi.motion import default_profile
from flowphi.planning.planner import Planner, Settings, State
from flowphi.planning.scoring import score_contact
from flowphi.ui.validation import hand_crossing_warnings
from test_drag_continuation import make_chart


def record(hand, finger, x0, x1, start=1., end=1.4):
    return dict(hand=hand, finger=finger, kind='drag', start=start, end=end,
                note_ids=[0], effort=.018, burst_cost=.0, planned=True,
                decision=dict(degraded=[]), points=[[start, x0, .5], [end, x1, .5]])


class HandOrderTests(unittest.TestCase):
    def planner(self):
        return Planner(make_chart([(1, 1., 0., 0.)]),
                       Settings(fingers=('index',), beam_width=2), default_profile())

    def test_ordered_hands_are_not_crossed(self):
        planner = self.planner()
        left, right = record('left', 'index', .35, .35), record('right', 'index', .65, .65)
        self.assertEqual((0., 0., None), planner.hand_crossing(right, (left,)))

    def test_hands_passing_close_do_not_invert_the_wrists(self):
        # Fingertips can be closer than a hand width while the wrists stay ordered.
        planner = self.planner()
        left, right = record('left', 'index', .55, .55), record('right', 'index', .45, .45)
        self.assertEqual(0., planner.hand_crossing(right, (left,))[0])

    def test_deep_crossing_is_measured_over_its_whole_overlap(self):
        planner = self.planner()
        left, right = record('left', 'index', .95, .95), record('right', 'index', .2, .2)
        excess, seconds, at = planner.hand_crossing(right, (left,))
        self.assertGreater(excess, .05)
        self.assertAlmostEqual(.4, seconds, places=6)
        self.assertIsNotNone(at)

    def test_fingertip_proximity_misses_what_the_order_test_sees(self):
        # Crossed wrists are far apart, so pad_collision never fires on this pair.
        planner = self.planner()
        left, right = record('left', 'index', .95, .95), record('right', 'index', .2, .2)
        errors, _, _ = planner.constraints(right, State((left,)), 1, 'index')
        self.assertNotIn('pad_collision', errors)
        self.assertGreater(planner.hand_crossing(right, (left,))[0], .05)

    def test_sustained_crossing_costs_more_than_a_transient(self):
        planner = self.planner()
        right = record('right', 'index', .2, .2)
        long = record('left', 'index', .95, .95)
        short = record('left', 'index', .95, .95, end=1.02)
        score_contact(planner, long, (right,), 0., 0., [])
        score_contact(planner, short, (right,), 0., 0., [])
        long_cost = long['decision']['costs']['crossing']
        short_cost = short['decision']['costs']['crossing']
        self.assertGreater(long_cost, short_cost)
        self.assertGreater(short_cost, 0.)


class CrossedHandWarningTests(unittest.TestCase):
    class Document:
        def __init__(self, contacts):
            from flowphi.motion import default_profile
            self.baseline = dict(profile=default_profile(), physical_screen=[.28, .1575])
            self.contacts = contacts

        def __len__(self):
            return len(self.contacts)

        def contact(self, i):
            return self.contacts[i]

    def test_crossed_contacts_are_reported_as_warnings(self):
        contacts = [record('left', 'index', .95, .95), record('right', 'index', .2, .2)]
        warnings = hand_crossing_warnings(self.Document(contacts))
        self.assertEqual(1, len(warnings))
        self.assertEqual('warning', warnings[0]['severity'])
        self.assertGreater(warnings[0]['message'].count('mm'), 0)

    def test_ordered_contacts_are_silent(self):
        contacts = [record('left', 'index', .35, .35), record('right', 'index', .65, .65)]
        self.assertEqual([], hand_crossing_warnings(self.Document(contacts)))


if __name__ == '__main__':
    unittest.main()
