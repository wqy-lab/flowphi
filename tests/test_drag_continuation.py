import copy
import json
import unittest

from flowphi.charts.blocks import BlockAreas
from flowphi.charts.loader import load_chart
from flowphi.motion import default_profile
from flowphi.planning.planner import Planner, Settings, State
from flowphi.services.candidates import checked_report
from flowphi.ui.compiler import compile_plan
from flowphi.ui.generation import generate_draft
from flowphi.ui.model import PlanDocument
from flowphi.rendering.pipeline import contacts_from_psap


def make_chart(notes):
    return load_chart(json.dumps(dict(formatVersion=3, offset=0., judgeLineList=[dict(
        bpm=120., notesAbove=[dict(type=kind, time=t*64, positionX=x, holdTime=hold*64,
            speed=1., floorPosition=t) for kind, t, x, hold in notes], notesBelow=[],
        speedEvents=[dict(startTime=0, endTime=10000, value=1., floorPosition=0.)],
        judgeLineMoveEvents=[dict(startTime=0, endTime=10000, start=.5, end=.5, start2=.5, end2=.5)],
        judgeLineRotateEvents=[dict(startTime=0, endTime=10000, start=0., end=0.)],
        judgeLineDisappearEvents=[dict(startTime=0, endTime=10000, start=1., end=1.)],
    )])))


class DragContinuationTests(unittest.TestCase):
    def draft(self, chart):
        result = generate_draft(chart, Settings(fingers=('index',)), default_profile())
        doc = PlanDocument(result)
        report = checked_report(doc, chart)
        self.assertTrue(report['valid'], report['issues'])
        self.assertEqual([], result['diagnostics']['draft_fallback_notes'])
        return result, doc

    def test_tap_can_stay_down_for_later_drags(self):
        chart = make_chart([(1, 1., 0., 0.), (2, 1.125, 0., 0.), (2, 1.25, 0., 0.)])
        result, doc = self.draft(chart)
        self.assertEqual(1, len(result['contacts']))
        self.assertEqual([0, 1, 2], result['contacts'][0]['note_ids'])
        raw, _ = compile_plan(doc, chart, check=True)
        _, decoded = contacts_from_psap(raw)
        self.assertEqual(1, len(decoded), 'Tap and Drags must use one physical DOWN, not just adjacent editor rows')

    def test_drag_chain_can_slide_without_repressing(self):
        chart = make_chart([(2, 1., -2., 0.), (2, 1.125, 0., 0.), (2, 1.25, 2., 0.)])
        result, _ = self.draft(chart)
        self.assertEqual(1, len(result['contacts']))
        self.assertEqual({0, 1, 2}, set(result['contacts'][0]['note_ids']))

    def test_active_hold_collects_drag_even_when_free_finger_exists(self):
        chart = make_chart([(3, 1., 0., 1.), (2, 1.25, 0., 0.)])
        result, _ = self.draft(chart)
        self.assertEqual(1, len(result['contacts']))
        self.assertEqual('hold', result['contacts'][0]['kind'])
        self.assertEqual([0, 1], result['contacts'][0]['note_ids'])

    def test_completed_flick_keeps_every_original_stroke_sample(self):
        chart = make_chart([(4, 1., 0., 0.), (2, 1.125, 0., 0.)])
        planner = Planner(chart, Settings(fingers=('index',)), default_profile())
        cost, old = planner.choices(planner.tasks[0], State())[0]
        original = copy.deepcopy(old)
        shared = planner.share_drag(planner.tasks[1], State((old,), cost))
        self.assertTrue(shared)
        for candidate in shared:
            merged = next(c for c in candidate.contacts if 1 in c['note_ids'])
            self.assertEqual(original['points'], merged['points'][:len(original['points'])])
            self.assertEqual(original['start'], merged['start'])
        self.assertEqual(original, old)
        result, _ = self.draft(chart)
        self.assertEqual(1, len(result['contacts']))

    def test_unchanged_hold_can_collect_drag_with_inherited_collision_warning(self):
        chart = make_chart([(3, 1., 0., 1.), (3, 1., 0., 1.), (2, 1.25, 0., 0.)])
        planner = Planner(chart, Settings(fingers=('index', 'middle')), default_profile())
        cost, old = planner.choices(planner.tasks[0], State())[0]
        other = copy.deepcopy(old)
        other.update(finger='middle' if old['finger'] == 'index' else 'index', note_ids=[1])
        old['decision']['degraded'] = ['pad_collision']
        violations, _, _ = planner.constraints(old, State((other,)),
                                               -1 if old['hand'] == 'left' else 1, old['finger'])
        self.assertIn('pad_collision', violations)
        shared = planner.share_drag(planner.tasks[2], State((old, other), cost))
        self.assertTrue(shared)
        merged = next(c for c in shared[0].contacts if 2 in c['note_ids'])
        self.assertEqual(old['points'], merged['points'])
        self.assertEqual(old['end'], merged['end'])
        self.assertEqual(['pad_collision'], merged['decision']['degraded'])

    def test_later_tap_still_has_its_own_down(self):
        chart = make_chart([(1, 1., 0., 0.), (2, 1.125, 0., 0.), (1, 1.25, 0., 0.)])
        result, doc = self.draft(chart)
        self.assertEqual(2, len(result['contacts']))
        self.assertTrue(all(len({0, 2} & set(c['note_ids'])) <= 1 for c in result['contacts']))
        raw, _ = compile_plan(doc, chart, check=True)
        _, decoded = contacts_from_psap(raw)
        self.assertEqual(2, len(decoded))

    def test_hold_tail_can_extend_past_inherited_collision(self):
        chart = make_chart([(3, 1., 0., .25), (3, 1., 0., .25), (2, 1.25, 0., 0.)])
        planner = Planner(chart, Settings(fingers=('index', 'middle')), default_profile())
        cost, old = planner.choices(planner.tasks[0], State())[0]
        other = copy.deepcopy(old)
        other.update(finger='middle' if old['finger'] == 'index' else 'index', note_ids=[1])
        old['decision']['degraded'] = ['pad_collision']
        shared = planner.share_drag(planner.tasks[2], State((old, other), cost))
        self.assertTrue(shared)
        merged = next(c for c in shared[0].contacts if 2 in c['note_ids'])
        self.assertGreater(merged['end'], old['end'])
        self.assertEqual(old['points'], merged['points'][:len(old['points'])])
        self.assertEqual(['pad_collision'], merged['decision']['degraded'])

    def test_extension_rejects_new_collision_in_the_gap(self):
        chart = make_chart([(1, 1., 0., 0.), (3, 1.03, 0., 1.), (2, 1.125, 0., 0.)])
        planner = Planner(chart, Settings(fingers=('index', 'middle')), default_profile())
        cost, old = planner.choices(planner.tasks[0], State())[0]
        obstacle = copy.deepcopy(old)
        obstacle.update(finger='middle' if old['finger'] == 'index' else 'index', note_ids=[1],
                        kind='hold', start=1.03, end=2.031,
                        points=[[1.03, *old['points'][-1][1:]]])
        shared = planner.share_drag(planner.tasks[2], State((old, obstacle), cost))
        self.assertFalse(any(0 in c['note_ids'] and 2 in c['note_ids']
                             for s in shared for c in s.contacts))

    def test_long_idle_gap_is_not_turned_into_indefinite_hold(self):
        chart = make_chart([(2, 1., 0., 0.), (2, 2., 0., 0.)])
        result, _ = self.draft(chart)
        self.assertEqual(2, len(result['contacts']))

    def test_cannot_bridge_through_temporary_full_screen_block(self):
        chart = make_chart([(1, 1., 0., 0.), (2, 1.25, 0., 0.)])
        chart.block_areas = BlockAreas([dict(
            bottomLeftPercentage=dict(x=0., y=0.), topRightPercentage=dict(x=1., y=1.),
            appearTime=1.08, enableTime=1.08, disableTime=1.16, disappearTime=1.16,
        )], chart.width, chart.height)
        result, _ = self.draft(chart)
        self.assertEqual(2, len(result['contacts']))

    def test_cannot_modify_two_tap_heads_into_one_held_touch(self):
        chart = make_chart([(1, 1., 0., 0.), (1, 1.125, 0., 0.)])
        planner = Planner(chart, Settings(fingers=('index',)), default_profile())
        cost, contact = planner.choices(planner.tasks[0], State())[0]
        self.assertEqual([], planner.share_drag(planner.tasks[1], State((contact,), cost)))


if __name__ == '__main__':
    unittest.main()
