import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from flowphi.motion import default_profile
from flowphi.planning.planner import Settings
from flowphi.services.candidates import CandidateStore, evaluate_candidate, fingerprint, movement_metrics
from flowphi.services.operations import apply_operations, contact_ids
from flowphi.ui.generation import generate_draft
from flowphi.ui.model import PlanDocument
from flowphi.ui.sources import create_demo, load_source


class CandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temp_root = Path(__file__).resolve().parents[1] / '.local' / 'test-temp'
        temp_root.mkdir(parents=True, exist_ok=True)
        cls.temporary = tempfile.TemporaryDirectory(dir=temp_root)
        assert Path(cls.temporary.name).resolve().is_relative_to(temp_root.resolve())
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.source = load_source(create_demo(cls.root), cls.root)
        draft = generate_draft(cls.source.chart, Settings(fingers=('index', 'middle')), default_profile())
        cls.snapshot = PlanDocument(draft, chart=cls.source.reference).snapshot()

    def setUp(self):
        self.doc = PlanDocument.from_dict(copy.deepcopy(self.snapshot))

    def evaluate(self, operations, **kwargs):
        return evaluate_candidate(self.doc, self.source.chart, operations, **kwargs)

    @staticmethod
    def op(name, ids=None, **values):
        return dict(op=name, scope=dict(ids=ids) if ids else dict(all=True), values=values)

    def collision_ops(self):
        # The demo ends in a simultaneous two-note chord.
        return [self.op('assign', [len(self.doc)-1, len(self.doc)], hand='left', finger='index')]

    def test_noop_compiles_actual_touch_data(self):
        result = self.evaluate([])
        self.assertTrue(result['result']['eligible'])
        self.assertTrue(result['report']['geometry_checked'])
        self.assertEqual(len(result['report']['validated_psap_sha256']), 64)
        self.assertFalse(result['report']['native_ap_validated'])
        self.assertFalse(result['report']['mesh_checked'])
        self.assertGreater(result['result']['candidate']['movement']['total_finger_travel_m'], 0)
        self.assertEqual(0., result['result']['comparison']['total_finger_travel_delta_m'])

    def test_movement_metrics_respect_physical_units_and_contact_edges(self):
        plan = dict(physical_screen=[.2, .1], contacts=[
            dict(hand='left', finger='index', start=0., end=1., points=[[0., 0., 0.], [.5, .5, 0.]]),
            dict(hand='left', finger='index', start=2., end=3., points=[[2., 1., 0.]]),
            dict(hand='right', finger='index', start=1., end=2., points=[[1., .5, 0.]]),
        ])
        metrics = movement_metrics(plan)
        self.assertAlmostEqual(.1, metrics['contact_path_m'])
        self.assertAlmostEqual(.1, metrics['transfer_distance_m'])
        self.assertAlmostEqual(.2, metrics['total_finger_travel_m'])
        self.assertAlmostEqual(.1, metrics['peak_transfer_speed_lower_bound_m_s'])
        self.assertEqual(1, metrics['peak_simultaneous_contacts'])

    def test_pose_candidate_does_not_claim_mesh_or_native_verification(self):
        result = self.evaluate([dict(op='pose_defaults', values=dict(
            hand='left', target='hand', values=dict(yaw=.1)))])
        self.assertTrue(result['result']['pose_changed'])
        self.assertTrue(result['result']['eligible'])
        self.assertFalse(result['result']['candidate']['mesh_checked'])
        self.assertFalse(result['result']['candidate']['native_ap_validated'])

    def test_evaluation_never_mutates_source_or_operations(self):
        before, revision = copy.deepcopy(self.doc.snapshot()), self.doc.revision
        operations = [self.op('swap', [1])]
        original_ops = copy.deepcopy(operations)
        result = self.evaluate(operations)
        self.assertEqual(before, self.doc.snapshot())
        self.assertEqual(revision, self.doc.revision)
        self.assertEqual([], self.doc.undo_stack)
        self.assertEqual(original_ops, operations)
        self.assertEqual([1], result['result']['changed_contacts'])

    def test_collision_rejected_and_cannot_commit(self):
        result = self.evaluate(self.collision_ops())
        self.assertFalse(result['result']['eligible'])
        self.assertGreater(result['result']['candidate']['errors'], 0)
        store = CandidateStore()
        public = store.remember(result, self.doc, revision=self.doc.revision)
        with self.assertRaisesRegex(ValueError, '未通过'):
            store.prepare_commit(public['candidate_id'], self.doc)

    def test_outside_judgement_area_rejected(self):
        result = self.evaluate([self.op('paths', [1], operation='set', x=.99, y=.5)])
        self.assertFalse(result['result']['eligible'])
        self.assertIn('geometry', result['result']['candidate']['issue_counts'])

    def test_shortened_hold_rejected_full_chart(self):
        hold = next(i+1 for i in range(len(self.doc)) if self.doc.contact(i)['kind'] == 'hold')
        result = self.evaluate([self.op('release', [hold], mode='duration', value=.02)])
        self.assertFalse(result['result']['eligible'])
        self.assertIn('hold_gap', result['result']['candidate']['issue_counts'])
        self.assertEqual('full_chart', result['result']['candidate']['scope'])

    def test_music_offset_applied_once_and_issues_are_one_based(self):
        first = self.doc.contact(0)
        offset = 7.25
        ops = [dict(op='contact', scope=dict(time_range=[first['start']+offset, first['end']+offset]),
                    values=dict(points=[[t+offset, .99, y] for t, x, y in first['points']]))]
        result = self.evaluate(ops, offset=offset)
        self.assertFalse(result['result']['eligible'])
        issue = next(i for i in result['result']['candidate']['issues'] if i['code'] == 'geometry')
        self.assertIn(1, issue['contacts'])
        self.assertGreaterEqual(issue['time'], first['start']+offset-.001)

    def test_successful_commit_is_undoable(self):
        result = self.evaluate([self.op('swap', [1])])
        store = CandidateStore()
        public = store.remember(result, self.doc, revision=0)
        candidate, _ = store.prepare_commit(public['candidate_id'], self.doc)
        before = fingerprint(self.doc)
        self.doc.replace_plan(candidate, '采用候选打法')
        self.assertNotEqual(before, fingerprint(self.doc))
        self.doc.undo()
        self.assertEqual(before, fingerprint(self.doc))

    def test_edit_and_undo_still_invalidates_old_candidate(self):
        store = CandidateStore()
        public = store.remember(self.evaluate([]), self.doc, revision=0)
        self.doc.swap_hands([0])
        self.doc.undo()
        with self.assertRaisesRegex(ValueError, 'stale'):
            store.prepare_commit(public['candidate_id'], self.doc)

    def test_document_switch_and_offset_change_rejected(self):
        store = CandidateStore()
        public = store.remember(self.evaluate([]), self.doc, revision=0)
        clone = PlanDocument.from_dict(copy.deepcopy(self.doc.snapshot()))
        for doc, offset in ((clone, 0.), (self.doc, .1)):
            with self.assertRaisesRegex(ValueError, 'stale'):
                store.prepare_commit(public['candidate_id'], doc, offset=offset)

    def test_untracked_content_change_rejected(self):
        store = CandidateStore()
        public = store.remember(self.evaluate([]), self.doc, revision=0)
        self.doc.title += ' changed'
        with self.assertRaisesRegex(ValueError, 'stale'):
            store.prepare_commit(public['candidate_id'], self.doc)

    def test_changed_while_background_job_runs_rejected(self):
        result = self.evaluate([])
        self.doc.swap_hands([0])
        with self.assertRaisesRegex(ValueError, 'stale'):
            CandidateStore().remember(result, self.doc, revision=0)

    def test_store_bounded_and_public_result_isolated(self):
        result = self.evaluate([])
        store = CandidateStore(capacity=1)
        first = store.remember(result, self.doc, revision=0)
        second = store.remember(result, self.doc, revision=0)
        with self.assertRaisesRegex(ValueError, '过期'):
            store.prepare_commit(first['candidate_id'], self.doc)
        second['eligible'] = False
        second['candidate']['issues'].append(dict(code='fake'))
        _, actual = store.prepare_commit(second['candidate_id'], self.doc)
        self.assertTrue(actual['eligible'])
        self.assertNotIn('fake', [p['code'] for p in actual['candidate']['issues']])

    def test_private_snapshot_isolated_from_worker_result(self):
        result = self.evaluate([])
        store = CandidateStore()
        public = store.remember(result, self.doc, revision=0)
        result['snapshot']['title'] = 'tampered'
        candidate, _ = store.prepare_commit(public['candidate_id'], self.doc)
        self.assertEqual(self.doc.title, candidate.title)

    def test_selection_and_sequential_scopes_use_candidate_state(self):
        hand = self.doc.contact(0)['hand']
        swapped = 'right' if hand == 'left' else 'left'
        ops = [self.op('swap', [1]), dict(op='lock', scope=dict(ids=[1], hand=swapped), values=dict(locked=True))]
        result = apply_operations(self.doc, ops, chart=self.source.chart)
        self.assertEqual(swapped, result.contact(0)['hand'])
        self.assertTrue(result.contact(0)['assignment_locked'])
        self.assertEqual([0], contact_ids(self.doc, selected_ids=[0]))

    def test_malformed_operations_are_atomic(self):
        before = fingerprint(self.doc)
        with self.assertRaises((ValueError, KeyError)):
            self.evaluate([self.op('swap', [1]), dict(op='unsupported', values={})])
        self.assertEqual(before, fingerprint(self.doc))
        for operations in ({}, [self.op('swap', [1])]*201):
            with self.assertRaises(ValueError):
                self.evaluate(operations)

    def test_worker_roundtrip_and_store_compatibility(self):
        work = self.root / 'worker-probe'
        work.mkdir(exist_ok=True)
        project = work / 'input.fingering.json'
        project.write_text(json.dumps(self.doc.snapshot()), encoding='utf-8')
        request = work / 'request.json'
        request.write_text(json.dumps(dict(kind='evaluate_candidate', work=str(work), cache=str(self.root),
            project=str(project), operations=[self.op('swap', [1])], offset=0.)), encoding='utf-8')
        completed = subprocess.run([sys.executable, '-m', 'flowphi.ui.worker', str(request)],
                                   capture_output=True, text=True, encoding='utf-8', timeout=30,
                                   env=os.environ | dict(PYTHONUTF8='1', PYTHONIOENCODING='utf-8'))
        self.assertEqual(0, completed.returncode, completed.stdout + completed.stderr)
        envelope = json.loads((work / 'result.json').read_text(encoding='utf-8'))
        self.assertTrue(envelope['ok'])
        result = envelope['result']
        store = CandidateStore()
        public = store.remember(result, self.doc, revision=0)
        candidate, _ = store.prepare_commit(public['candidate_id'], self.doc)
        self.assertNotEqual(candidate.contact(0)['hand'], self.doc.contact(0)['hand'])

    def test_tools_available_in_all_existing_model_protocols(self):
        from flowphi.ui.model_api import build_request
        config = dict(base_url='https://example.invalid', api_key='test-only', model='test-only')
        for protocol in ('openai', 'responses', 'anthropic'):
            with self.subTest(protocol=protocol):
                _, _, raw = build_request(config | dict(protocol=protocol), [])
                payload = json.loads(raw)
                tools = [t.get('function', t) for t in payload['tools']]
                names = {t['name'] for t in tools}
                self.assertTrue({'evaluate_candidate', 'commit_candidate'} <= names)
                evaluation = next(t for t in tools if t['name'] == 'evaluate_candidate')
                schema = evaluation.get('parameters', evaluation.get('input_schema'))
                self.assertEqual(200, schema['properties']['operations']['maxItems'])

    def test_candidate_arguments_capture_message_selection(self):
        from types import SimpleNamespace
        from flowphi.ui.model_edit import ModelEditPanel
        scope = dict(time_range=[24., 29.])
        call = dict(name='evaluate_candidate', arguments=dict(operations=[dict(op='swap', values={})]))
        result = ModelEditPanel.arguments(SimpleNamespace(scope=scope), call)
        self.assertEqual(scope, result['operations'][0]['scope'])
        self.assertNotIn('scope', call['arguments']['operations'][0])


if __name__ == '__main__':
    unittest.main()
