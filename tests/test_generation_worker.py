import copy
from dataclasses import replace
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from flowphi.motion import default_profile
from flowphi.planning.planner import Planner, Settings
from flowphi.ui import worker
from flowphi.ui.generation import generate_draft
from flowphi.ui.model import read_document
from flowphi.ui.validation import validate
from test_drag_continuation import make_chart


class GenerationWorkerTests(unittest.TestCase):
    def temporary(self):
        root = Path(__file__).resolve().parents[1] / '.local' / 'test-temp'
        root.mkdir(parents=True, exist_ok=True)
        return TemporaryDirectory(dir=root)

    def test_missing_optional_progress_time_does_not_abort_audit(self):
        task = Planner(make_chart([(1, 1., 0., 0.)]), Settings(), default_profile()).tasks[0]
        with patch.object(worker, 'event') as event:
            worker.planning_progress(1, 1, replace(task, planning_detail=dict(
                start=.9, end=1.1, strategy='ordinary_audit', done=0, total=3)))
        self.assertIn('0/3', event.call_args.kwargs['detail'])

    def test_real_four_finger_job_reaches_save_through_final_audit(self):
        chart = make_chart([(1, 1., -2., 0.), (1, 1., 2., 0.)])
        with self.temporary() as tmp:
            source = SimpleNamespace(chart=chart, reference=dict(path='probe.json', note_count=2))
            request = dict(kind='generate', work=tmp, cache=tmp, chart='probe.json', options=dict(
                settings=dict(fingers=['index', 'middle'], beam_width=1, candidates_per_finger=1)))
            with patch.object(worker, 'load_source', return_value=source), patch.object(worker, 'event') as event:
                result = worker.execute(request)
            doc = read_document(result['project'])
            self.assertEqual({0, 1}, set(doc.note_contacts))
            self.assertTrue(doc.baseline['diagnostics']['generation']['complete'])
            windows = doc.baseline['diagnostics']['ordinary_hand_search']['windows']
            self.assertTrue(windows)
            self.assertTrue(all('final_motion' in w for w in windows))
            self.assertTrue(any('普通音符手部复核' in str(c) for c in event.call_args_list))
            recovery = read_document(Path(tmp) / 'recovery.fingering.json')
            self.assertFalse(recovery.baseline['diagnostics']['generation']['complete'])

    def test_late_failure_keeps_complete_contacts_in_explicit_recovery_draft(self):
        chart = make_chart([(1, 1., 0., 0.)])
        saved = []
        with patch('flowphi.planning.physical_plain.audit_ordinary', side_effect=RuntimeError('late audit failure')):
            with self.assertRaisesRegex(RuntimeError, 'late audit failure'):
                generate_draft(chart, Settings(fingers=('index',), hand_motion_repair=False), default_profile(),
                               checkpoint=lambda draft: saved.append(copy.deepcopy(draft)))
        self.assertTrue(saved)
        self.assertEqual({0}, {n for c in saved[-1]['contacts'] for n in c['note_ids']})
        self.assertFalse(saved[-1]['diagnostics']['generation']['complete'])
        from flowphi.ui.model import PlanDocument
        report = validate(PlanDocument(saved[-1]), chart, geometry=False)
        self.assertIn('generation_incomplete', [i['code'] for i in report['issues']])

    def test_worker_error_exposes_recovery_location(self):
        with self.temporary() as tmp:
            request = Path(tmp) / 'request.json'
            request.write_text(json.dumps(dict(kind='generate', work=tmp)), encoding='utf-8')
            recovery = Path(tmp) / 'recovery.fingering.json'
            recovery.write_text('{}', encoding='utf-8')
            with patch.object(worker.sys, 'argv', ['worker', str(request)]), \
                 patch.dict(os.environ), \
                 patch.object(worker, 'execute', side_effect=RuntimeError('audit failed')), \
                 patch.object(worker.traceback, 'print_exc'), patch.object(worker, 'event'), patch.object(worker, 'report_log'):
                self.assertEqual(1, worker.main())
            result = json.loads((Path(tmp) / 'result.json').read_text(encoding='utf-8'))
            self.assertFalse(result['ok'])
            self.assertEqual(str(recovery), result['recovery_project'])
            self.assertIn(str(recovery), result['error'])
