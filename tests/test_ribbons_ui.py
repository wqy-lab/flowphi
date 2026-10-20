import os
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtWidgets import QApplication
from flowphi.ui.dialogs import GenerateDialog
from flowphi.ui.preferences import validate_preferences


class RibbonDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_generation_toggle_roundtrips_and_defaults_on(self):
        dialog = GenerateDialog({})
        self.assertTrue(dialog.values()['settings']['drag_ribbon_planning'])
        dialog.ribbons.setChecked(False)
        self.assertFalse(dialog.values()['settings']['drag_ribbon_planning'])
        validate_preferences({'drag_ribbon_planning': False})
        restored = GenerateDialog({'drag_ribbon_planning': False})
        self.assertFalse(restored.ribbons.isChecked())
        dialog.close()
        restored.close()
