import importlib
import importlib.util
import os
import tempfile
import time
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch


@unittest.skipUnless(importlib.util.find_spec('PySide6') or importlib.util.find_spec('PySide2'),
                     'Qt bindings are not installed')
class PoseUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        cls.ui = importlib.import_module('3dsmax_pose_tool.pose_tool')
        cls.app = cls.ui.QtWidgets.QApplication.instance() or cls.ui.QtWidgets.QApplication([])

    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.body = Mock(spec=self.ui._biped_body.BodyRetargeter)
        self.hands = {side: Mock(spec=self.ui._biped_hand.HandRetargeter)
                      for side in ('left', 'right')}
        for adapter in (self.body, *self.hands.values()):
            adapter.can_restore.return_value = False
        stack.enter_context(patch.object(self.ui._biped_body, 'BodyRetargeter', return_value=self.body))
        stack.enter_context(patch.object(self.ui._biped_hand, 'HandRetargeter', side_effect=self.hands.get))
        self.errors = []
        stack.enter_context(patch.object(self.ui.QtWidgets.QMessageBox, 'critical',
                                        side_effect=lambda parent, title, text: self.errors.append(text)))
        folder = stack.enter_context(tempfile.TemporaryDirectory())
        self.image_path = str(Path(folder) / 'input.png')
        image = self.ui.QtGui.QImage(8, 8, self.ui.QtGui.QImage.Format_ARGB32)
        image.fill(self.ui.QtGui.QColor('white'))
        self.assertTrue(image.save(self.image_path))
        self.window = self.ui.MediaPipePoseWindow()
        self.addCleanup(self.close_window)

    def close_window(self):
        for tab in (self.window.body_tab, self.window.hand_tab):
            if tab._thread is not None:
                tab._thread.quit()
                tab._thread.wait(3000)
                self.app.processEvents()
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    def wait_for_request(self, tab):
        deadline = time.monotonic() + 5
        while tab._thread is not None and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.005)
        self.assertIsNone(tab._thread, 'Inference thread did not finish')
        self.assertEqual(self.errors, [])

    def test_body_generate_uses_frozen_target_frame_and_keeps_preview(self):
        tab = self.window.body_tab
        frame = {'test': 'body'}
        request = {'time': 7, 'target': 'Actor'}
        self.body.capture_request.return_value = request
        self.body.validate_request.return_value = 'Actor'
        self.body.apply.return_value = {'written': 52}
        tab.request_function = Mock(return_value=frame)
        self.assertTrue(tab.image_drop.setImagePath(self.image_path))
        tab.generate_button.click()
        self.wait_for_request(tab)
        self.body.validate_request.assert_called_once_with(request)
        self.body.apply.assert_called_once_with(frame, 'Actor', 7, keep_source_bones=True)
        self.assertIs(tab.cached_frame, frame)
        self.body.can_restore.return_value = True
        tab._update_controls()
        tab.restore_button.click()
        self.body.restore.assert_called_once_with()
        self.assertIsNone(tab.last_result)

    def test_each_hand_generate_and_restore_use_the_selected_adapter(self):
        tab = self.window.hand_tab
        self.window.tabs.setCurrentWidget(tab)
        for side, radio in (('left', tab.left_radio), ('right', tab.right_radio)):
            with self.subTest(side=side):
                radio.setChecked(True)
                frame = {'test': side}
                adapter = self.hands[side]
                adapter.apply.return_value = {'target_side': side}
                tab.request_function = Mock(return_value=frame)
                self.assertTrue(tab.image_drop.setImagePath(self.image_path))
                tab.generate_button.click()
                self.assertFalse(tab.left_radio.isEnabled())
                self.assertFalse(tab.right_radio.isEnabled())
                self.wait_for_request(tab)
                adapter.apply.assert_called_once_with(frame)
                self.assertIs(tab._last_applied, adapter)
                adapter.can_restore.return_value = True
                tab._update_controls()
                tab.restore_button.click()
                adapter.restore.assert_called_once_with()
                self.assertIsNone(tab._last_applied)


if __name__ == '__main__':
    unittest.main()
