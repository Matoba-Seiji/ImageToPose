"""Unified 3ds Max MediaPipe tool with body and hand tabs."""

from __future__ import annotations

import importlib
import tempfile
import uuid
from pathlib import Path

try:
    from PySide6 import QtCore, QtGui, QtWidgets
except ImportError:
    from PySide2 import QtCore, QtGui, QtWidgets


_package_name = __package__ or "3dsmax_pose_tool"
_pose_client = importlib.import_module("{}.pose_client".format(_package_name))
_pose_client = importlib.reload(_pose_client)
_hand_retarget_math = importlib.import_module(
    "{}.hand_retarget_math".format(_package_name)
)
_hand_retarget_math = importlib.reload(_hand_retarget_math)
_thumb_rotation_math = importlib.import_module(
    "{}.thumb_rotation_math".format(_package_name)
)
_thumb_rotation_math = importlib.reload(_thumb_rotation_math)
_thumb_reference = importlib.import_module(
    "{}.thumb_reference".format(_package_name)
)
_thumb_reference = importlib.reload(_thumb_reference)
_biped_hand = importlib.import_module("{}.biped_hand".format(_package_name))
_biped_hand = importlib.reload(_biped_hand)
_body_retarget_math = importlib.import_module("{}.body_retarget_math".format(_package_name))
_body_retarget_math = importlib.reload(_body_retarget_math)
_biped_body = importlib.import_module("{}.biped_body".format(_package_name))
_biped_body = importlib.reload(_biped_body)


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
WINDOW_OBJECT_NAME = "DccMaxMediaPipeUnifiedPoseToolWindow"
WINDOW_UI_VERSION = "3dsmax-mediapipe-unified-pose-qt-v16-window-paste"
WINDOW_SIZE = QtCore.QSize(560, 620)
_WINDOW = None
_APP = None


try:
    QT_DIALOG = QtCore.Qt.Dialog
except AttributeError:
    QT_DIALOG = QtCore.Qt.WindowType.Dialog

try:
    QT_WINDOW_CLOSE_BUTTON_HINT = QtCore.Qt.WindowCloseButtonHint
except AttributeError:
    QT_WINDOW_CLOSE_BUTTON_HINT = QtCore.Qt.WindowType.WindowCloseButtonHint

try:
    QT_ALIGN_CENTER = QtCore.Qt.AlignCenter
except AttributeError:
    QT_ALIGN_CENTER = QtCore.Qt.AlignmentFlag.AlignCenter

try:
    QT_LEFT_BUTTON = QtCore.Qt.LeftButton
except AttributeError:
    QT_LEFT_BUTTON = QtCore.Qt.MouseButton.LeftButton

try:
    QT_STRONG_FOCUS = QtCore.Qt.StrongFocus
except AttributeError:
    QT_STRONG_FOCUS = QtCore.Qt.FocusPolicy.StrongFocus

try:
    QT_POINTING_HAND_CURSOR = QtCore.Qt.PointingHandCursor
except AttributeError:
    QT_POINTING_HAND_CURSOR = QtCore.Qt.CursorShape.PointingHandCursor

try:
    QT_KEEP_ASPECT_RATIO = QtCore.Qt.KeepAspectRatio
except AttributeError:
    QT_KEEP_ASPECT_RATIO = QtCore.Qt.AspectRatioMode.KeepAspectRatio

try:
    QT_SMOOTH_TRANSFORMATION = QtCore.Qt.SmoothTransformation
except AttributeError:
    QT_SMOOTH_TRANSFORMATION = QtCore.Qt.TransformationMode.SmoothTransformation

try:
    QT_ANTIALIASING = QtGui.QPainter.Antialiasing
except AttributeError:
    QT_ANTIALIASING = QtGui.QPainter.RenderHint.Antialiasing


def _error_text(error):
    return str(error).strip() or type(error).__name__


def _image_path(path):
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_file() or candidate.suffix.lower() not in IMAGE_EXTENSIONS:
        return None
    return candidate


def _is_image_path(path):
    return (
        path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and not QtGui.QImage(str(path)).isNull()
    )


def _clipboard_image_path(prefix):
    clipboard = QtWidgets.QApplication.clipboard()
    return _mime_image_path(clipboard.mimeData(), prefix)


def _mime_image_path(mime, prefix):
    if mime is None:
        return None
    if mime.hasImage():
        image = QtGui.QImage(mime.imageData())
        if not image.isNull():
            # A new paste must not overwrite the file used by an earlier request.
            path = Path(tempfile.gettempdir()) / (prefix + "_" + uuid.uuid4().hex + ".png")
            if image.save(str(path), "PNG"):
                return str(path)
    if mime.hasUrls():
        for url in mime.urls():
            if url.isLocalFile() and _is_image_path(Path(url.toLocalFile())):
                return url.toLocalFile()
    return None


def max_main_window():
    """Return 3ds Max's Qt main window when the qtmax helper is available."""

    try:
        from qtmax import GetQMaxMainWindow

        return GetQMaxMainWindow()
    except Exception:
        return None


class _RequestWorker(QtCore.QObject):
    finished = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, request_function, image_path, parent=None):
        super().__init__(parent)
        self.request_function = request_function
        self.image_path = image_path

    def run(self):
        try:
            frame = self.request_function(self.image_path)
        except Exception as error:
            self.failed.emit(_error_text(error))
            return
        self.finished.emit(frame)


class ImageDropWidget(QtWidgets.QFrame):
    """Reusable image picker with proportional preview."""

    imageSelected = QtCore.Signal(str)

    def __init__(self, placeholder, clipboard_prefix, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setFocusPolicy(QT_STRONG_FOCUS)
        self.setMinimumSize(496, 390)
        self.setCursor(QT_POINTING_HAND_CURSOR)
        self._placeholder = placeholder
        self._clipboard_prefix = clipboard_prefix
        self._path = ""
        self._pixmap = QtGui.QPixmap()
        self._drag_active = False
        self.setToolTip("点击选择图片；Ctrl+V 或右键粘贴图片")

    @property
    def path(self):
        return self._path

    def setImagePath(self, path):
        candidate = _image_path(path)
        if candidate is None or not _is_image_path(candidate):
            return False
        pixmap = QtGui.QPixmap(str(candidate))
        if pixmap.isNull():
            return False
        self._path = str(candidate)
        self._pixmap = pixmap
        self._drag_active = False
        self.update()
        self.imageSelected.emit(self._path)
        return True

    def _choose_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "选择图片",
            "",
            "图片 (*.png *.jpg *.jpeg *.bmp *.webp)",
        )
        if path:
            self.setImagePath(path)

    def mousePressEvent(self, event):
        if event.button() == QT_LEFT_BUTTON:
            self._choose_file()
        super().mousePressEvent(event)

    def dragEnterEvent(self, event):
        mime = event.mimeData()
        if mime.hasImage() or mime.hasUrls():
            self._drag_active = True
            self.update()
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self._drag_active = False
        self.update()
        event.accept()

    def dropEvent(self, event):
        mime = event.mimeData()
        pasted_path = _mime_image_path(mime, self._clipboard_prefix) if self.isEnabled() else None
        accepted = bool(pasted_path and self.setImagePath(pasted_path))
        self._drag_active = False
        self.update()
        if accepted:
            event.acceptProposedAction()
        else:
            event.ignore()

    def paste_image(self):
        if not self.isEnabled():
            return False
        pasted_path = _clipboard_image_path(self._clipboard_prefix)
        return bool(pasted_path and self.setImagePath(pasted_path))

    def contextMenuEvent(self, event):
        if not self.isEnabled():
            return
        menu = QtWidgets.QMenu(self)
        action = menu.addAction("粘贴图片")
        action.triggered.connect(self.paste_image)
        execute = getattr(menu, "exec", None) or menu.exec_
        execute(event.globalPos())

    def keyPressEvent(self, event):
        try:
            is_paste = event.matches(QtGui.QKeySequence.Paste)
        except AttributeError:
            is_paste = event.matches(QtGui.QKeySequence.StandardKey.Paste)
        if is_paste:
            self.paste_image()
            event.accept()
            return
        super().keyPressEvent(event)

    def paintEvent(self, event):
        del event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QT_ANTIALIASING)
        painter.fillRect(self.rect(), QtGui.QColor("#464646"))
        border_color = QtGui.QColor("#4da3ff" if self._drag_active else "#5b5b5b")
        painter.setPen(QtGui.QPen(border_color, 2 if self._drag_active else 1))
        painter.drawRect(self.rect().adjusted(1, 1, -1, -1))
        if not self._pixmap.isNull():
            target = self.rect().adjusted(10, 10, -10, -10)
            preview = self._pixmap.scaled(
                target.size(), QT_KEEP_ASPECT_RATIO, QT_SMOOTH_TRANSFORMATION
            )
            point = QtCore.QPoint(
                target.center().x() - preview.width() // 2,
                target.center().y() - preview.height() // 2,
            )
            painter.drawPixmap(point, preview)
        else:
            painter.setPen(QtGui.QColor("#d0d0d0"))
            painter.drawText(self.rect(), QT_ALIGN_CENTER, self._placeholder)


class InferenceTab(QtWidgets.QWidget):
    """One independent Max tab with its own request and scene writer."""

    def __init__(
        self,
        request_function,
        button_text,
        placeholder,
        clipboard_prefix,
        parent=None,
    ):
        super().__init__(parent)
        self.request_function = request_function
        self._thread = None
        self._worker = None
        self.image_drop = ImageDropWidget(placeholder, clipboard_prefix, self)
        self.generate_button = QtWidgets.QPushButton(button_text, self)
        self.generate_button.setMinimumHeight(42)
        self.generate_button.setEnabled(False)
        self.image_drop.imageSelected.connect(self._on_image_selected)
        self.generate_button.clicked.connect(self.generate)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 12, 0, 0)
        layout.setSpacing(12)
        layout.addWidget(self.image_drop, 1)
        layout.addWidget(self.generate_button, 0)

    @property
    def is_running(self):
        return self._thread is not None and self._thread.isRunning()

    def _on_image_selected(self, path):
        self.generate_button.setEnabled(bool(path) and not self.is_running)

    def generate(self):
        if not self.image_drop.path or self.is_running:
            return
        self.generate_button.setEnabled(False)
        self._thread = QtCore.QThread(self)
        self._worker = _RequestWorker(self.request_function, self.image_drop.path)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_ready)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._on_finished)
        self._thread.start()

    def _on_failed(self, message):
        QtWidgets.QMessageBox.critical(self, "生成失败", str(message))

    def _on_finished(self):
        thread = self._thread
        self._thread = None
        self._worker = None
        if thread is not None:
            thread.deleteLater()
        self.generate_button.setEnabled(bool(self.image_drop.path))


class BodyInferenceTab(InferenceTab):
    def __init__(self, parent=None):
        super().__init__(_pose_client.request_pose, "生成Pose", "上传全身人物图片",
                         "3dsmax_mediapipe_body_clipboard", parent)
        self.retargeter = _biped_body.BodyRetargeter()
        self.cached_frame = None
        self.last_result = None
        self._pending_request = None
        self.layout().removeWidget(self.generate_button)
        self.restore_button = QtWidgets.QPushButton("恢复", self)
        self.restore_button.setMinimumHeight(42)
        self.restore_button.setEnabled(False)
        self.restore_button.setToolTip("撤销最近一次全身写入")
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.generate_button, 1)
        row.addWidget(self.restore_button)
        self.layout().addLayout(row)
        self.restore_button.clicked.connect(self.restore_pose)
        self._guard_timer = QtCore.QTimer(self)
        self._guard_timer.setInterval(300)
        self._guard_timer.timeout.connect(self._update_controls)
        self._guard_timer.start()

    def generate(self):
        if self._thread is not None or not self.image_drop.path:
            return
        try:
            self._pending_request = self.retargeter.capture_request()
        except Exception as error:
            self._on_failed(_error_text(error))
            return
        self.last_result = None
        super().generate()
        self._update_controls()

    def _on_image_selected(self, path):
        self.cached_frame = None
        super()._on_image_selected(path)

    def _on_ready(self, frame):
        self.cached_frame = frame
        try:
            if self._pending_request is None:
                raise RuntimeError("No pending body operation")
            target = self.retargeter.validate_request(self._pending_request)
            self.last_result = self.retargeter.apply(
                frame, target, self._pending_request["time"], keep_source_bones=True
            )
        except Exception as error:
            self._on_failed("识别成功，但 Biped 全身重定向失败：{}".format(_error_text(error)))
        self._update_controls()

    def _on_finished(self):
        super()._on_finished()
        self._pending_request = None
        self._update_controls()

    def _update_controls(self):
        busy = self._thread is not None
        self.image_drop.setEnabled(not busy)
        self.generate_button.setEnabled(not busy and bool(self.image_drop.path))
        self.generate_button.setText("处理中…" if busy else "生成Pose")
        self.restore_button.setEnabled(not busy and self.retargeter.can_restore())

    def restore_pose(self):
        try:
            self.retargeter.restore()
            self.last_result = None
        except Exception as error:
            self._on_failed(_error_text(error))
        self._update_controls()


class HandInferenceTab(InferenceTab):
    def __init__(
        self,
        request_function,
        button_text,
        placeholder,
        clipboard_prefix,
        parent=None,
    ):
        super().__init__(
            request_function,
            button_text,
            placeholder,
            clipboard_prefix,
            parent,
        )
        self.retargeters = {side: _biped_hand.HandRetargeter(side) for side in ("left", "right")}
        self.retargeter = self.retargeters["left"]
        self._last_applied = None
        self._pending_apply = None
        self.cached_frame = None
        self.target_group = QtWidgets.QButtonGroup(self)
        self.left_radio = QtWidgets.QRadioButton("左手", self)
        self.right_radio = QtWidgets.QRadioButton("右手", self)
        self.left_radio.setToolTip("Bip001 L Hand")
        self.right_radio.setToolTip("Bip001 R Hand")
        self.target_group.addButton(self.left_radio, 0)
        self.target_group.addButton(self.right_radio, 1)
        self.left_radio.setChecked(True)
        target_row = QtWidgets.QHBoxLayout()
        target_row.addWidget(QtWidgets.QLabel("应用到", self))
        target_row.addWidget(self.left_radio)
        target_row.addWidget(self.right_radio)
        target_row.addStretch(1)
        self.layout().removeWidget(self.generate_button)
        self.layout().addLayout(target_row)
        self.generate_button.setText("生成Pose")
        self.restore_button = QtWidgets.QPushButton("恢复", self)
        self.restore_button.setMinimumHeight(42)
        self.restore_button.setEnabled(False)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.generate_button, 1)
        row.addWidget(self.restore_button)
        self.layout().addLayout(row)
        self.restore_button.clicked.connect(self.restore_pose)
        self.left_radio.toggled.connect(self._target_changed)
        self.right_radio.toggled.connect(lambda checked: self._target_changed(False) if checked else None)
        self._guard_timer = QtCore.QTimer(self)
        self._guard_timer.setInterval(300)
        self._guard_timer.timeout.connect(self._update_controls)
        self._guard_timer.start()

    def _on_image_selected(self, path):
        self.cached_frame = None
        super()._on_image_selected(path)

    def generate(self):
        if not self.image_drop.path or self._thread is not None:
            return
        # Bind the operation to the target selected at click. The writer
        # always enforces isolated finger tracks.
        self._pending_apply = self.retargeter
        super().generate()
        self._update_controls()

    def _on_ready(self, frame):
        self.cached_frame = frame
        try:
            if self._pending_apply is None:
                raise RuntimeError("No pending hand operation")
            target = self._pending_apply
            result = target.apply(frame)
            self._last_applied = target
            label = "左手" if result["target_side"] == "left" else "右手"
            self.restore_button.setToolTip("撤销最近一次{}写入".format(label))
        except Exception as error:
            self._on_failed("识别成功，但 Biped 手指重定向失败：{}".format(_error_text(error)))

    def _on_finished(self):
        super()._on_finished()
        self._pending_apply = None
        self._update_controls()

    def _update_controls(self):
        busy = self._thread is not None
        self.image_drop.setEnabled(not busy)
        self.left_radio.setEnabled(not busy)
        self.right_radio.setEnabled(not busy)
        self.generate_button.setEnabled(not busy and bool(self.image_drop.path))
        self.generate_button.setText("处理中…" if busy else "生成Pose")
        self.restore_button.setEnabled(not busy and self._last_applied is not None and self._last_applied.can_restore())

    def _target_changed(self, left_checked):
        side = "left" if left_checked else "right"
        self.retargeter = self.retargeters[side]
        self._update_controls()

    def restore_pose(self):
        try:
            if self._last_applied is None:
                return
            self._last_applied.restore()
            self._last_applied = None
        except Exception as error:
            self._on_failed(_error_text(error))
        self._update_controls()


class MediaPipePoseWindow(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent or max_main_window())
        self.setObjectName(WINDOW_OBJECT_NAME)
        self.setProperty("pose_ui_version", WINDOW_UI_VERSION)
        self.setWindowTitle("图生Pose")
        self.setFixedSize(WINDOW_SIZE)
        self.setWindowFlags(QT_DIALOG | QT_WINDOW_CLOSE_BUTTON_HINT)
        self.setStyleSheet(
            "QDialog { background: #353535; }"
            "QPushButton { background: #646464; color: #f0f0f0; border: 0;"
            " padding: 8px; font-size: 14px; }"
            "QPushButton:hover { background: #747474; }"
            "QPushButton:disabled { color: #999999; background: #4d4d4d; }"
        )

        self.tabs = QtWidgets.QTabWidget(self)
        self.body_tab = BodyInferenceTab(self)
        self.hand_tab = HandInferenceTab(
            _pose_client.request_hand_pose,
            "生成Pose",
            "上传清晰手部图片",
            "3dsmax_mediapipe_hand_clipboard",
            self,
        )
        self.tabs.addTab(self.body_tab, "全身")
        self.tabs.addTab(self.hand_tab, "手部")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.addWidget(self.tabs, 1)
        shortcut_class = getattr(QtGui, "QShortcut", None) or QtWidgets.QShortcut
        try:
            paste_key = QtGui.QKeySequence.StandardKey.Paste
        except AttributeError:
            paste_key = QtGui.QKeySequence.Paste
        self._paste_shortcut = shortcut_class(QtGui.QKeySequence(paste_key), self)
        self._paste_shortcut.activated.connect(self.paste_image)

    def paste_image(self):
        tab = self.tabs.currentWidget()
        if tab._thread is not None:
            return False
        return tab.image_drop.paste_image()

    def closeEvent(self, event):
        if self.body_tab._thread is not None or self.hand_tab._thread is not None:
            QtWidgets.QMessageBox.information(
                self,
                "正在生成",
                "当前识别还没有完成，请等待完成后再关闭工具。",
            )
            event.ignore()
            return
        global _WINDOW
        self.body_tab._guard_timer.stop()
        self.body_tab.retargeter.close()
        self.hand_tab._guard_timer.stop()
        for retargeter in self.hand_tab.retargeters.values():
            retargeter.close()
        _WINDOW = None
        super().closeEvent(event)


def show_pose_window(tab_index=0):
    """Show the unified Max MediaPipe tool and select a tab."""

    global _APP, _WINDOW
    _APP = QtWidgets.QApplication.instance()
    if _APP is None:
        _APP = QtWidgets.QApplication([])

    if _WINDOW is not None:
        try:
            if _WINDOW.property("pose_ui_version") == WINDOW_UI_VERSION:
                _WINDOW.tabs.setCurrentIndex(int(tab_index))
                _WINDOW.show()
                _WINDOW.raise_()
                _WINDOW.activateWindow()
                return _WINDOW
            _WINDOW.close()
            _WINDOW.deleteLater()
        except RuntimeError:
            pass
        _WINDOW = None

    for widget in _APP.topLevelWidgets():
        try:
            if widget.objectName() == WINDOW_OBJECT_NAME:
                widget.close()
                widget.deleteLater()
        except RuntimeError:
            pass

    _WINDOW = MediaPipePoseWindow(parent=max_main_window())
    _WINDOW.tabs.setCurrentIndex(int(tab_index))
    _WINDOW.show()
    _WINDOW.raise_()
    _WINDOW.activateWindow()
    return _WINDOW


def show_hand_pose_window():
    """Open the unified Max tool on the standalone hand tab."""

    return show_pose_window(tab_index=1)


if __name__ == "__main__":
    show_pose_window()
