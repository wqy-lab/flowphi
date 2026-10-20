"""Projection and rendered integration checks for the combined hand/chart view."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PySide6.QtCore import QPointF, QSettings
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from flowphi.motion import default_profile
from flowphi.planning.planner import Settings
from flowphi.ui.generation import generate_draft
from flowphi.ui.model import PlanDocument
from flowphi.ui.pose import PoseCanvas
from flowphi.ui.sources import create_demo, load_source
from flowphi.ui.widgets import IndexedPreview


class PoseChartTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
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
        self.canvas = PoseCanvas()
        self.canvas.resize(900, 500)
        self.preview = IndexedPreview(self.source.chart, width=640)
        self.canvas.set_chart_preview(self.preview)
        self.canvas.set_document(self.doc, self.source.chart)
        self.canvas.set_time(9.6)

    def tearDown(self):
        self.canvas.close()

    def test_chart_pixels_match_physical_screen_for_all_views(self):
        image = QImage(640, 360, QImage.Format_ARGB32)
        screen = (.28, .1575)
        for view in ('angle', 'top', 'side'):
            self.canvas.set_view(view)
            for zoom in (.7, 1.5, 2.):
                self.canvas.zoom = zoom
                transform = self.canvas.chart_transform(screen, image)
                for u, v in ((0., 0.), (1., 0.), (1., 1.), (0., 1.), (.31, .72)):
                    with self.subTest(view=view, zoom=zoom, u=u, v=v):
                        actual = transform.map(QPointF(u * image.width(), v * image.height()))
                        expected = self.canvas.project(((u-.5)*screen[0], (.5-v)*screen[1], 0.))
                        self.assertAlmostEqual(expected.x(), actual.x(), places=7)
                        self.assertAlmostEqual(expected.y(), actual.y(), places=7)

    def test_time_and_lookahead_shared_with_chart_renderer(self):
        self.canvas.set_chart_preview(self.preview, lookahead=3.)
        first = self.canvas.grab().toImage()
        self.assertEqual((9.6, 3.), self.preview.cached_key)
        self.canvas.set_time(11.5)
        second = self.canvas.grab().toImage()
        self.assertEqual((11.5, 3.), self.preview.cached_key)
        self.assertNotEqual(first, second)

    def test_overlay_toggle_changes_screen_without_changing_hit_targets(self):
        enabled = self.canvas.grab().toImage()
        hits = copy.deepcopy(self.canvas.hits)
        self.canvas.set_show_chart(False)
        disabled = self.canvas.grab().toImage()
        self.assertEqual(hits, self.canvas.hits)
        self.assertNotEqual(enabled, disabled)
        # Compare the board itself, not the changed caption above it.
        screen = self.canvas.engine.frame(self.canvas.time)['screen']
        center = self.canvas.project((0., 0., 0.))
        x, y = round(center.x()), round(center.y())
        self.assertNotEqual(enabled.copy(x-70, y-35, 140, 70), disabled.copy(x-70, y-35, 140, 70))
        self.assertGreater(screen[0], 0.)

    def test_hand_opacity_keeps_contacts_selectable(self):
        self.canvas.grab()
        expected = copy.deepcopy(self.canvas.hits)
        self.canvas.set_hand_opacity(10)
        self.canvas.grab()
        self.assertEqual(expected, self.canvas.hits)
        hand, target, a, b, _ = self.canvas.hits[0]
        self.assertIsNotNone(self.canvas._hit(a))

    def test_detached_chart_falls_back_to_grid(self):
        self.canvas.grab()
        self.canvas.set_chart_preview(None)
        self.canvas.grab()
        self.assertIsNone(self.canvas.chart_preview)
        self.assertGreater(len(self.canvas.hits), 0)

    def test_window_rebinds_preview_after_performance_or_source_change(self):
        from flowphi.ui.gui import EditorWindow
        with patch('flowphi.ui.gui.app_settings', side_effect=lambda: QSettings(
                str(self.root / 'window-settings.ini'), QSettings.IniFormat)):
            window = EditorWindow()
        try:
            window._set_source(self.source)
            window._set_document(self.doc)
            self.assertIs(window.canvas.preview, window.pose_canvas.chart_preview)
            old = window.canvas.preview
            window.performance['preview_width'] = 640 if window.canvas.render_width != 640 else 800
            window.performance['lookahead'] = 4.
            window._apply_performance()
            self.assertIsNot(old, window.canvas.preview)
            self.assertIs(window.canvas.preview, window.pose_canvas.chart_preview)
            self.assertEqual(4., window.pose_canvas.lookahead)
            window._set_source(None)
            self.assertIsNone(window.pose_canvas.chart_preview)
        finally:
            window._set_document(None)
            window.close()


if __name__ == '__main__':
    unittest.main()
