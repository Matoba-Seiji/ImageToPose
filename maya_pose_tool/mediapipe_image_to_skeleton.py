"""Unified MediaPipe Maya tool with body and hand tabs.

The body-and-hand recognizer and the hand-only recognizer remain separate
providers. This module only combines their Maya UI into one dialog so the
user can switch between them without opening two windows.
"""

from __future__ import annotations

import importlib
import tempfile
from pathlib import Path

try:
    from PySide6 import QtCore, QtGui, QtWidgets
    from shiboken6 import wrapInstance
except ImportError:
    from PySide2 import QtCore, QtGui, QtWidgets
    from shiboken2 import wrapInstance

from maya import OpenMayaUI
import maya.cmds as cmds


def _load_local_module(module_basename):
    module_name = "{}.{}".format(__package__, module_basename) if __package__ else module_basename
    return importlib.import_module(module_name)


_mediapipe_pose = _load_local_module("mediapipe_pose")
_mediapipe_hand_pose = _load_local_module("mediapipe_hand_pose")
_pose_skeleton = _load_local_module("pose_skeleton")
_hand_skeleton = _load_local_module("hand_skeleton")

create_skeleton_from_pose = _pose_skeleton.create_skeleton_from_pose
create_hand_skeleton_from_pose = _hand_skeleton.create_hand_skeleton_from_pose


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
MAYA_SCALE = 100.0
WINDOW_OBJECT_NAME = "MediaPipeUnifiedPoseImageToSkeletonWindow"
WINDOW_UI_VERSION = "maya-mediapipe-unified-pose-ui-v1-tabs"
BODY_SKELETON_NAME_PREFIX = "MediaPipePoseSkeleton_"
BODY_SKELETON_MARKER_VALUE = "maya_mediapipe_pose_tool"
HAND_SKELETON_NAME_PREFIX = "MediaPipeHandPoseSkeleton_"
_WINDOW = None


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
    QT_STRONG_FOCUS = QtCore.Qt.StrongFocus
except AttributeError:
    QT_STRONG_FOCUS = QtCore.Qt.FocusPolicy.StrongFocus

try:
    QT_LEFT_BUTTON = QtCore.Qt.LeftButton
except AttributeError:
    QT_LEFT_BUTTON = QtCore.Qt.MouseButton.LeftButton

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

try:
    QT_PASTE = QtGui.QKeySequence.Paste
except AttributeError:
    QT_PASTE = QtGui.QKeySequence.StandardKey.Paste


def maya_main_window():
    pointer = OpenMayaUI.MQtUtil.mainWindow()
    if pointer is None:
        return None
    try:
        return wrapInstance(int(pointer), QtWidgets.QWidget)
    except TypeError:
        return wrapInstance(pointer, QtWidgets.QWidget)


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
    mime = clipboard.mimeData()
    if mime.hasImage():
        image = QtGui.QImage(mime.imageData())
        if not image.isNull():
            path = Path(tempfile.gettempdir()) / (prefix + "_clipboard.png")
            if image.save(str(path), "PNG"):
                return str(path)
    if mime.hasUrls():
        for url in mime.urls():
            if url.isLocalFile() and _is_image_path(Path(url.toLocalFile())):
                return url.toLocalFile()
    return None


class _InferenceWorker(QtCore.QObject):
    """Run one provider outside Maya's UI thread."""

    finished = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, infer_function, image_path, parent=None):
        super().__init__(parent)
        self.infer_function = infer_function
        self.image_path = image_path

    def run(self):
        try:
            frame = self.infer_function(self.image_path)
        except Exception as error:
            self.failed.emit(_error_text(error))
            return
        self.finished.emit(frame)


class ImageDropWidget(QtWidgets.QFrame):
    """Reusable image picker shared by both tabs."""

    imageSelected = QtCore.Signal(str)

    def __init__(self, placeholder, clipboard_prefix, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setFocusPolicy(QT_STRONG_FOCUS)
        self.setMinimumSize(496, 390)
        self._placeholder = placeholder
        self._clipboard_prefix = clipboard_prefix
        self._path = ""
        self._pixmap = QtGui.QPixmap()
        self._drag_active = False
        self.setCursor(QtCore.Qt.PointingHandCursor)

    @property
    def path(self):
        return self._path

    def setImagePath(self, path):
        candidate = _image_path(path)
        if candidate is None or not _is_image_path(candidate):
            return False
        self._path = str(candidate)
        self._pixmap = QtGui.QPixmap(self._path)
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
        accepted = False
        if mime.hasUrls():
            for url in mime.urls():
                if url.isLocalFile() and self.setImagePath(url.toLocalFile()):
                    accepted = True
                    break
        elif mime.hasImage():
            pasted_path = _clipboard_image_path(self._clipboard_prefix)
            accepted = bool(pasted_path and self.setImagePath(pasted_path))
        self._drag_active = False
        self.update()
        if accepted:
            event.acceptProposedAction()
        else:
            event.ignore()

    def keyPressEvent(self, event):
        if event.matches(QT_PASTE):
            pasted_path = _clipboard_image_path(self._clipboard_prefix)
            if pasted_path:
                self.setImagePath(pasted_path)
            event.accept()
            return
        super().keyPressEvent(event)

    def paintEvent(self, event):
        del event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QT_ANTIALIASING)
        painter.fillRect(self.rect(), QtGui.QColor("#1f252c"))
        border_color = QtGui.QColor("#4da3ff" if self._drag_active else "#46515e")
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
            painter.setPen(QtGui.QColor("#aab4c0"))
            painter.drawText(self.rect(), QT_ALIGN_CENTER, self._placeholder)


class InferenceTab(QtWidgets.QWidget):
    """One provider tab with its own image, worker, and output callback."""

    def __init__(
        self,
        infer_function,
        create_function,
        button_text,
        placeholder,
        clipboard_prefix,
        parent=None,
    ):
        super().__init__(parent)
        self._infer_function = infer_function
        self._create_function = create_function
        self._thread = None
        self._worker = None

        self.image_drop = ImageDropWidget(
            placeholder,
            clipboard_prefix,
            self,
        )
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
        self._worker = _InferenceWorker(
            self._infer_function,
            self.image_drop.path,
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_ready)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._on_finished)
        self._thread.start()

    def _on_ready(self, frame):
        try:
            self._create_function(frame)
        except Exception as error:
            self._on_failed(
                "识别成功，但 Maya 骨架创建失败：{}".format(_error_text(error))
            )

    def _on_failed(self, message):
        QtWidgets.QMessageBox.critical(self, "生成失败", str(message))

    def _on_finished(self):
        thread = self._thread
        self._thread = None
        self._worker = None
        if thread is not None:
            thread.deleteLater()
        self.generate_button.setEnabled(bool(self.image_drop.path))


def _create_body_skeleton(frame):
    return create_skeleton_from_pose(
        frame,
        cmds,
        scale=MAYA_SCALE,
        name_prefix=BODY_SKELETON_NAME_PREFIX,
        generated_marker_value=BODY_SKELETON_MARKER_VALUE,
    )


def _create_hand_skeleton(frame):
    return create_hand_skeleton_from_pose(
        frame,
        cmds,
        scale=MAYA_SCALE,
        name_prefix=HAND_SKELETON_NAME_PREFIX,
    )


class MediaPipeUnifiedPoseWindow(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent or maya_main_window())
        self.setObjectName(WINDOW_OBJECT_NAME)
        self.setProperty("pose_ui_version", WINDOW_UI_VERSION)
        self.setWindowTitle("MediaPipe Maya Pose 工具")
        self.setFixedSize(520, 520)
        self.setWindowFlags(QT_DIALOG | QT_WINDOW_CLOSE_BUTTON_HINT)

        self.tabs = QtWidgets.QTabWidget(self)
        self.body_tab = InferenceTab(
            _mediapipe_pose.infer_single_image,
            _create_body_skeleton,
            "使用 MediaPipe 识别并创建身体和手指骨架",
            "上传全身人物图片",
            "maya_mediapipe_body_clipboard",
            self,
        )
        self.hand_tab = InferenceTab(
            _mediapipe_hand_pose.infer_single_hand_image,
            _create_hand_skeleton,
            "使用 MediaPipe 识别并创建手部骨架",
            "上传清晰手部图片",
            "maya_mediapipe_hand_clipboard",
            self,
        )
        self.tabs.addTab(self.body_tab, "身体 + 手指")
        self.tabs.addTab(self.hand_tab, "单独手部")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.addWidget(self.tabs, 1)

    def generate_skeleton_from_pose_frame(self, frame):
        return _create_body_skeleton(frame)

    def generate_hand_skeleton_from_pose_frame(self, frame):
        return _create_hand_skeleton(frame)

    def closeEvent(self, event):
        if self.body_tab.is_running or self.hand_tab.is_running:
            QtWidgets.QMessageBox.information(
                self,
                "正在识别",
                "当前识别还没有完成，请等待完成后再关闭工具。",
            )
            event.ignore()
            return
        global _WINDOW
        _WINDOW = None
        super().closeEvent(event)


def _close_legacy_windows(app):
    legacy_names = {
        "MediaPipePoseImageToSkeletonWindow",
        "MediaPipeHandPoseImageToSkeletonWindow",
    }
    for widget in app.topLevelWidgets():
        try:
            if widget.objectName() in legacy_names:
                widget.close()
                widget.deleteLater()
        except RuntimeError:
            pass


def show_mediapipe_pose_window(tab_index=0):
    """Show the unified MediaPipe tool and select a tab."""

    global _WINDOW
    app = QtWidgets.QApplication.instance()
    if app is not None:
        _close_legacy_windows(app)
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

    _WINDOW = MediaPipeUnifiedPoseWindow(parent=maya_main_window())
    _WINDOW.tabs.setCurrentIndex(int(tab_index))
    _WINDOW.show()
    _WINDOW.raise_()
    _WINDOW.activateWindow()
    return _WINDOW


def show_mediapipe_hand_pose_window():
    return show_mediapipe_pose_window(tab_index=1)


def generate_skeleton_from_pose_frame(frame):
    return _create_body_skeleton(frame)


def generate_hand_skeleton_from_pose_frame(frame):
    return _create_hand_skeleton(frame)


show_pose_window = show_mediapipe_pose_window


if __name__ == "__main__":
    show_mediapipe_pose_window()
