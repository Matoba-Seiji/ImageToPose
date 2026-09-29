"""Standalone HTTP service for a single hand image and Biped finger pose."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


MAX_UPLOAD_BYTES = 20 * 1024 * 1024
PROTOCOL_VERSION = "pose-skeleton-v3-tabs"
LANDMARKS = (
    "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
)
FINGERS = {
    "thumb": ("thumb_cmc", "thumb_mcp", "thumb_ip"),
    "index": ("index_mcp", "index_pip", "index_dip"),
    "middle": ("middle_mcp", "middle_pip", "middle_dip"),
    "ring": ("ring_mcp", "ring_pip", "ring_dip"),
    "pinky": ("pinky_mcp", "pinky_pip", "pinky_dip"),
}
JOINTS = ("wrist",) + tuple(
    "{}_{}".format(finger, segment)
    for finger in FINGERS for segment in (1, 2, 3)
)
PARENTS = {"{}_1".format(finger): "wrist" for finger in FINGERS}
for _finger in FINGERS:
    PARENTS["{}_2".format(_finger)] = "{}_1".format(_finger)
    PARENTS["{}_3".format(_finger)] = "{}_2".format(_finger)


class HandServiceError(RuntimeError):
    pass


def _finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise HandServiceError("手部关键点不是数字。") from error
    if not math.isfinite(number):
        raise HandServiceError("手部关键点不是有限数字。")
    return number


def _item(position, confidence, source):
    return {
        "position": [_finite(value) for value in position],
        "visibility": max(0.0, min(1.0, _finite(confidence))),
        "source": source,
    }


def _category(handedness):
    if isinstance(handedness, (list, tuple)):
        handedness = handedness[0] if handedness else None
    label = str(getattr(handedness, "category_name", "") or
                getattr(handedness, "display_name", "")).strip().lower()
    try:
        score = _finite(getattr(handedness, "score", 1.0))
    except HandServiceError:
        score = 1.0
    return label if label in {"left", "right"} else "unknown", max(0.0, min(1.0, score))


def _infer_hand(image_bytes, model_path):
    try:
        import cv2
        import mediapipe as mp
        import numpy as np
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
    except ImportError as error:
        raise HandServiceError("运行时缺少 MediaPipe 图像依赖。") from error

    image_array = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image_array is None or image_array.size == 0:
        raise HandServiceError("图片解码失败，请确认图片没有损坏。")
    rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_buffer=Path(model_path).read_bytes()),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    try:
        with vision.HandLandmarker.create_from_options(options) as landmarker:
            result = landmarker.detect(image)
    except Exception as error:
        raise HandServiceError("MediaPipe 手部姿态识别失败：{}".format(error)) from error

    world_hands = getattr(result, "hand_world_landmarks", None) or []
    image_hands = getattr(result, "hand_landmarks", None) or []
    handedness = getattr(result, "handedness", None) or []
    count = min(len(world_hands), len(image_hands))
    if count == 0:
        raise HandServiceError("图片中没有返回可用的手部三维关键点。请使用清晰的单手图片，并确保手腕完整可见。")
    scores = [_category(handedness[i] if i < len(handedness) else None)[1] for i in range(count)]
    selected = max(range(count), key=lambda i: scores[i])
    label, score = _category(handedness[selected] if selected < len(handedness) else None)
    hand = world_hands[selected]
    if len(hand) < len(LANDMARKS):
        raise HandServiceError("MediaPipe 返回的手部姿态点不足 21 个。")

    points = {}
    for index, name in enumerate(LANDMARKS):
        landmark = hand[index]
        position = tuple(_finite(getattr(landmark, axis)) for axis in ("x", "y", "z"))
        visibility = getattr(landmark, "visibility", 1.0)
        presence = getattr(landmark, "presence", 1.0)
        confidence = min(_finite(visibility if visibility is not None else 1.0),
                         _finite(presence if presence is not None else 1.0))
        points[name] = (position, confidence)

    wrist_position, wrist_confidence = points["wrist"]
    landmarks = {}
    for name, (position, confidence) in points.items():
        relative = tuple(position[i] - wrist_position[i] for i in range(3))
        landmarks[name] = _item(relative, confidence, name)
    joints = {"wrist": _item((0.0, 0.0, 0.0), wrist_confidence, "wrist")}
    for finger, names in FINGERS.items():
        for segment, name in enumerate(names, start=1):
            position, confidence = points[name]
            relative = tuple(position[i] - wrist_position[i] for i in range(3))
            joints["{}_{}".format(finger, segment)] = _item(
                relative, min(wrist_confidence, confidence), name
            )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "skeleton": "maya_hand_skeleton",
        "coordinate_system": "root_relative_mediapipe_hand_world",
        "unit": "meters",
        "handedness": label,
        "handedness_score": score,
        "joints": joints,
        "parents": dict(PARENTS),
        "landmarks": landmarks,
        "source": {
            "provider": "MediaPipe Hand Landmarker",
            "image": None,
            "landmark_count": len(LANDMARKS),
            "world_coordinates": True,
            "hands_detected": count,
            "finger_joint_count": len(JOINTS) - 1,
        },
    }


def _extract_image(content_type, body):
    if content_type.startswith("image/"):
        return body
    if not content_type.startswith("multipart/form-data"):
        raise HandServiceError("请求必须是 multipart/form-data，字段名为 image。")
    header = "Content-Type: {}\r\nMIME-Version: 1.0\r\n\r\n".format(content_type)
    message = BytesParser(policy=policy.default).parsebytes(header.encode("ascii", "replace") + body)
    for part in message.walk():
        if (part.get_content_disposition() == "form-data" and
                part.get_param("name", header="Content-Disposition") == "image"):
            data = part.get_payload(decode=True)
            if data:
                return data
    raise HandServiceError("请求中没有找到 image 图片字段。")


class HandRequestHandler(BaseHTTPRequestHandler):
    server_version = "DCCHandPoseServer/1.0"

    def _json(self, status, value):
        payload = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path.rstrip("/") == "/health":
            self._json(200, {"ok": True, "service": "dcc-hand-pose", "protocol_version": PROTOCOL_VERSION})
        else:
            self._json(404, {"ok": False, "error": "not_found"})

    def do_POST(self):
        if self.path.rstrip("/") != "/v1/hand-pose":
            self._json(404, {"ok": False, "error": "not_found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0:
            self._json(400, {"ok": False, "error": "empty_request"})
            return
        if length > MAX_UPLOAD_BYTES:
            self._json(413, {"ok": False, "error": "image_too_large", "max_bytes": MAX_UPLOAD_BYTES})
            return
        try:
            image = _extract_image(self.headers.get("Content-Type", ""), self.rfile.read(length))
            if len(image) > MAX_UPLOAD_BYTES:
                raise HandServiceError("图片文件超过大小限制。")
            response = _infer_hand(image, self.server.hand_model_path)
        except HandServiceError as error:
            self._json(422, {"ok": False, "error": str(error)})
            return
        except Exception as error:
            self._json(500, {"ok": False, "error": "服务内部错误：{}".format(error)})
            return
        self._json(200, response)

    def log_message(self, format_string, *args):
        print("[DCCHandPoseServer] " + format_string % args)


def main(argv=None):
    parser = argparse.ArgumentParser(description="DCC hand image pose service")
    parser.add_argument("--host", default=os.environ.get("POSE_SERVER_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("POSE_SERVER_PORT", "18765")))
    default_model = Path(sys.executable).resolve().parent / "models" / "hand_landmarker.task" if getattr(sys, "frozen", False) else None
    parser.add_argument("--hand-model", default=os.environ.get("MEDIAPIPE_HAND_MODEL") or default_model)
    args = parser.parse_args(argv)
    model = Path(args.hand_model).expanduser().resolve() if args.hand_model else Path.home() / ".mediapipe" / "models" / "hand_landmarker.task"
    if not model.is_file():
        parser.error("hand_landmarker.task is missing: {}".format(model))
    server = ThreadingHTTPServer((args.host, args.port), HandRequestHandler)
    server.hand_model_path = str(model)
    print("[DCCHandPoseServer] listening on http://{}:{}/v1/hand-pose".format(args.host, args.port))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
