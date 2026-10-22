import copy
import json
import unittest

from flowphi.charts.blocks import BlockAreas
from flowphi.charts.loader import load_chart
from flowphi.motion import default_profile
from flowphi.planning.planner import Planner, Settings, State, load_at
from flowphi.planning.ribbons import detect_windows, plan_window
from flowphi.services.candidates import checked_report
from flowphi.ui.generation import generate_draft
from flowphi.ui.model import PlanDocument
from test_drag_continuation import make_chart


class RibbonTests(unittest.TestCase):
    def settings(self, **options):
        return Settings(fingers=('index', 'middle'), beam_width=4, candidates_per_finger=2, **options)

    def generate(self, notes):
        chart = make_chart(notes)
        result = generate_draft(chart, self.settings(), default_profile())
        report = checked_report(PlanDocument(result), chart)
        self.assertTrue(report['valid'], report['issues'])
        self.assertEqual([], result['diagnostics']['draft_fallback_notes'])
        self.assertEqual(set(range(len(notes))), {n for c in result['contacts'] for n in c['note_ids']})
        return chart, result

    def test_simple_chain_has_valid_continuous_candidate(self):
        _, result = self.generate([(2, 1.+i*.08, -2.+i, 0.) for i in range(5)])
        report = result['diagnostics']['ribbons']['windows'][0]
        self.assertTrue(report['candidates']['continuous']['valid'])
        self.assertEqual(1, report['candidates']['continuous']['metrics']['downs'])
        self.assertEqual(5, len(report['candidates']['continuous']['contacts'][0]['note_ids']))
        self.assertIn(report['selected'], report['candidates'])
        # Per-key planning is only a rescue; it is never a compared strategy.
        self.assertNotIn('baseline', report['candidates'])

    def test_turning_phrase_generates_relay_and_stationary_alternatives(self):
        _, result = self.generate([(2, 1.+i*.08, x, 0.) for i, x in enumerate((-3., -1., 1., 3., 1., -1., -3.))])
        candidates = result['diagnostics']['ribbons']['windows'][0]['candidates']
        self.assertTrue(candidates['relay']['valid'])
        self.assertTrue(candidates['stationary']['valid'])
        self.assertGreaterEqual(candidates['relay']['metrics']['relays'], 1)
        self.assertAlmostEqual(0., candidates['stationary']['metrics']['contact_path_m'])

    def test_mixed_tap_and_flick_keep_distinct_down_and_swipe(self):
        notes = [(2, 1.+i*.1, -3.+i*.8, 0.) for i in range(7)]
        notes += [(1, 1.25, 4., 0.), (4, 1.45, 4., 0.)]
        _, result = self.generate(notes)
        self.assertTrue(result['diagnostics']['ribbons']['windows'])
        self.assertTrue(any(c['kind'] == 'flick' and 8 in c['note_ids'] for c in result['contacts']))

    def test_cross_line_detection_uses_reachable_bands(self):
        # Alternate line identity while preserving the same legal trajectory.
        chart = make_chart([(2, 1.+i*.08, -2.+i, 0.) for i in range(5)])
        line = chart.lines[0]
        other = copy.copy(line)
        other.notes = line.notes[1::2]
        line.notes = line.notes[::2]
        chart.lines.append(other)
        planner = Planner(chart, self.settings(), default_profile())
        windows = detect_windows(planner)
        self.assertEqual(1, len(windows))
        self.assertEqual(5, len(windows[0].drag_ids))

    def test_two_lanes_are_not_collapsed_into_one_touch(self):
        notes = [(2, 1.+i*.1, x, 0.) for i in range(5) for x in (-4., 4.)]
        _, result = self.generate(notes)
        best = result['diagnostics']['ribbons']['windows'][0]['candidates']['continuous']
        self.assertTrue(best['valid'])
        self.assertGreaterEqual(best['metrics']['downs'], 2)

    def test_active_hold_can_collect_whole_phrase_without_extra_down(self):
        _, result = self.generate([(3, 1., 0., 1.)]+[(2, 1.1+i*.08, 0., 0.) for i in range(5)])
        self.assertEqual(1, len(result['contacts']))
        self.assertEqual('hold', result['contacts'][0]['kind'])

    def test_long_phrase_windows_cover_every_note(self):
        notes = [(2, 1.+i*.1, 0., 0.) for i in range(23)]
        _, result = self.generate(notes)
        windows = result['diagnostics']['ribbons']['windows']
        self.assertEqual(2, len(windows))
        self.assertEqual(set(range(23)), {n for w in windows for n in w['note_ids']})
        self.assertEqual(1, len(result['contacts']))

    def test_disable_preserves_plain_path_and_old_settings_default(self):
        chart = make_chart([(2, 1.+i*.1, 0., 0.) for i in range(5)])
        result = generate_draft(chart, self.settings(drag_ribbon_planning=False), default_profile())
        self.assertFalse(result['diagnostics']['ribbons']['enabled'])
        self.assertTrue(Settings().drag_ribbon_planning)

    def test_all_long_contacts_have_sustained_load(self):
        contact = dict(hand='left', finger='index', start=0., end=2., kind='drag', effort=.018)
        sustained, _ = load_at([contact], 1., 'left', 'index')
        short, _ = load_at([dict(contact, end=.012)], 1., 'left', 'index')
        self.assertGreater(sustained, short)

    def test_obstacle_gap_requires_release_and_relay(self):
        chart = make_chart([(2, 1.+i*.1, 0., 0.) for i in range(6)])
        chart.block_areas = BlockAreas([dict(
            bottomLeftPercentage=dict(x=0., y=0.), topRightPercentage=dict(x=1., y=1.),
            appearTime=1.14, enableTime=1.14, disableTime=1.16, disappearTime=1.16,
        )], chart.width, chart.height)
        result = generate_draft(chart, self.settings(), default_profile())
        self.assertTrue(checked_report(PlanDocument(result), chart)['valid'])
        candidates = result['diagnostics']['ribbons']['windows'][0]['candidates']
        self.assertFalse(candidates['continuous']['valid'])
        self.assertTrue(candidates['relay']['valid'])
        self.assertGreaterEqual(len(result['contacts']), 2)

    def test_baseline_is_only_a_rescue_when_every_strategy_fails(self):
        from unittest import mock
        from flowphi.planning import ribbons
        chart = make_chart([(2, 1.+i*.08, 0., 0.) for i in range(5)])
        with mock.patch.object(ribbons, '_family_candidates', lambda *args, **kwargs: []):
            result = generate_draft(chart, self.settings(), default_profile())
        window = result['diagnostics']['ribbons']['windows'][0]
        self.assertTrue(window['fallback'])
        self.assertEqual('baseline', window['selected'])
        self.assertTrue(window['candidates']['baseline']['fallback'])
        self.assertTrue(window['candidates']['baseline']['valid'])
        self.assertEqual(set(range(5)), {n for c in result['contacts'] for n in c['note_ids']})
        self.assertTrue(checked_report(PlanDocument(result), chart)['valid'])

    def test_incomplete_rescue_is_retained_but_not_reported_valid(self):
        from unittest import mock
        from flowphi.planning import ribbons
        planner = Planner(make_chart([(2, 1.+i*.08, 0., 0.) for i in range(5)]),
                          self.settings(), default_profile())
        def unresolved(*args, **kwargs):
            for task in kwargs['tasks']:
                kwargs['on_unplanned'](task, {'unavailable': 1})
            return [State()]
        with mock.patch.object(ribbons, '_family_candidates', return_value=[]), \
                mock.patch.object(planner, '_run_plain', side_effect=unresolved):
            states, report, pending = plan_window(planner, planner.tasks, [State()], lambda *a: None)
        rescue = report['candidates']['baseline']
        self.assertFalse(rescue['valid'])
        self.assertEqual(0, rescue['count'])
        self.assertEqual(1, rescue['search_states'])
        self.assertEqual(5, len(pending))
        self.assertEqual(1, len(states))
        self.assertEqual((), states[0].contacts)

    def test_fully_covered_but_invalid_rescue_is_not_reported_valid(self):
        from unittest import mock
        from flowphi.planning import ribbons
        planner = Planner(make_chart([(2, 1.+i*.08, 0., 0.) for i in range(5)]),
                          self.settings(), default_profile())
        contact = dict(hand='left', finger='index', pointer=0, kind='drag',
            start=1., end=1.334, note_ids=list(range(5)), joint_note_coverage=True,
            points=[[1., .99, .99]], decision={'degraded': []})
        state = State((contact,))
        with mock.patch.object(ribbons, '_family_candidates', return_value=[]), \
                mock.patch.object(planner, '_run_plain', return_value=[state]):
            states, report, pending = plan_window(planner, planner.tasks, [State()], lambda *a: None)
        self.assertFalse(report['candidates']['baseline']['valid'])
        self.assertEqual(1, report['candidates']['baseline']['rejections']['incomplete_or_invalid'])
        # The physical selector now rejects malformed geometry instead of
        # forwarding it as a fully covered draft. Preserve all unplanned IDs.
        self.assertEqual(set(range(5)),{t.id for t,_ in pending})
        self.assertFalse(report['candidates']['baseline']['hand_valid'])
        self.assertEqual(1, len(states))
        self.assertEqual((),states[0].contacts)

    def test_window_evaluation_does_not_rewrite_input_history(self):
        chart = make_chart([(1, .5, 0., 0.)]+[(2, 1.+i*.1, 0., 0.) for i in range(5)])
        planner = Planner(chart, self.settings(), default_profile())
        cost, old = planner.choices(planner.tasks[0], State())[0]
        original = copy.deepcopy(old)
        incoming = [State((old,), cost)]
        states, report, pending = plan_window(planner, planner.tasks[1:], incoming, lambda *args: None)
        self.assertFalse(pending)
        self.assertEqual(original, old)
        for state in states:
            owner = next(c for c in state.contacts if 0 in c['note_ids'])
            self.assertEqual(original['points'], owner['points'][:len(original['points'])])

    def test_progress_carries_window_and_strategy_and_can_interrupt(self):
        planner = Planner(make_chart([(2, 1.+i*.1, 0., 0.) for i in range(5)]),
                          self.settings(), default_profile())
        class Cancelled(Exception):
            pass
        def progress(done, total, task):
            if task and task.planning_detail:
                self.assertEqual('continuous', task.planning_detail['strategy'])
                raise Cancelled()
        with self.assertRaises(Cancelled):
            planner.run(progress=progress)
