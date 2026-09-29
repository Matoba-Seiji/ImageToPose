"""Small HTTP server for single-image MediaPipe pose inference.

The service returns DCC-neutral joint positions. Maya and 3ds Max clients
convert that protocol into their own local skeleton APIs. The standard library
HTTP server keeps local deployment and cloud deployment identical.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

try:
    from maya_pose_tool.mediapipe_pose import (
        PoseLandmarkerError,
        find_hand_model_path,
        find_model_path,
        infer_single_image,
    )
    from maya_pose_tool.mediapipe_hand_pose import infer_single_hand_image
    from maya_pose_tool.pose_skeleton import (
        BODY_AND_FINGER_JOINTS,
        BODY_PARENTS,
    )
    from maya_pose_tool.hand_skeleton import HAND_JOINTS, HAND_PARENTS
except ImportError:
    from ..maya_pose_tool.mediapipe_pose import (  # pragma: no cover
        PoseLandmarkerError,
        find_hand_model_path,
        find_model_path,
        infer_single_image,
    )
    from ..maya_pose_tool.mediapipe_hand_pose import infer_single_hand_image  # pragma: no cover
    from ..maya_pose_tool.pose_skeleton import (  # pragma: no cover
        BODY_AND_FINGER_JOINTS,
        BODY_PARENTS,
    )
    from ..maya_pose_tool.hand_skeleton import HAND_JOINTS, HAND_PARENTS  # pragma: no cover


MAX_UPLOAD_BYTES = 20 * 1024 * 1024
PROTOCOL_VERSION = "pose-skeleton-v3-tabs"


class PoseServiceError(RuntimeError):
    """Raised for a malformed client request or unavailable model."""


def _json_bytes(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )


def _pose_response(frame):
    joints = frame.get("joints") or {}
    return {
        "protocol_version": PROTOCOL_VERSION,
        "skeleton": "maya_pose_skeleton",
        "coordinate_system": frame.get("coordinate_system"),
        "unit": frame.get("unit", "meters"),
        "include_fingers": bool(frame.get("include_fingers")),
        "joints": {
            name: joints[name]
            for name in BODY_AND_FINGER_JOINTS
            if name in joints
        },
        "face_world": frame.get("face_world", {}),
        "parents": {
            child: parent
            for child, parent in BODY_PARENTS.items()
            if child in joints and parent in joints
        },
        "source": {
            "provider": "MediaPipe Pose Landmarker",
            "landmark_count": frame.get("source", {}).get("landmark_count", 33),
            "world_coordinates": True,
            "hands_detected": frame.get("source", {}).get("hands_detected", 0),
            "finger_joint_count": frame.get("source", {}).get("finger_joint_count", 0),
        },
    }


def _hand_pose_response(frame):
    joints = frame.get("joints") or {}
    return {
        "protocol_version": PROTOCOL_VERSION,
        "skeleton": "maya_hand_skeleton",
        "coordinate_system": frame.get("coordinate_system"),
        "unit": frame.get("unit", "meters"),
        "handedness": frame.get("handedness", "unknown"),
        "handedness_score": frame.get("handedness_score", 0.0),
        "joints": {
            name: joints[name]
            for name in HAND_JOINTS
            if name in joints
        },
        "parents": {
            child: parent
            for child, parent in HAND_PARENTS.items()
            if child in joints and parent in joints
        },
        "landmarks": frame.get("landmarks", {}),
        "source": {
            "provider": "MediaPipe Hand Landmarker",
            "landmark_count": frame.get("source", {}).get("landmark_count", 21),
            "world_coordinates": True,
            "hands_detected": frame.get("source", {}).get("hands_detected", 0),
            "finger_joint_count": frame.get("source", {}).get("finger_joint_count", 15),
        },
    }


def _extract_image(content_type, body):
    if content_type.startswith("image/"):
        return body

    if not content_type.startswith("multipart/form-data"):
        raise PoseServiceError(
            "请求必须是 multipart/form-data，字段名为 image。"
        )

    header = (
        "Content-Type: {}\r\n"
        "MIME-Version: 1.0\r\n"
        "\r\n"
    ).format(content_type).encode("ascii", "replace")
    message = BytesParser(policy=policy.default).parsebytes(header + body)
    for part in message.walk():
        if part.get_content_disposition() != "form-data":
            continue
        if part.get_param("name", header="Content-Disposition") != "image":
            continue
        data = part.get_payload(decode=True)
        if data:
            return data
    raise PoseServiceError("请求中没有找到 image 图片字段。")


def _infer_uploaded_image(image_bytes, model_path, hand_model_path):
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="pose_upload_",
            suffix=".image",
            delete=False,
        ) as temporary:
            temporary.write(image_bytes)
            temporary_path = Path(temporary.name)

        frame = infer_single_image(
            temporary_path,
            model_path=model_path,
            hand_model_path=hand_model_path,
        )
        response = _pose_response(frame)
        response["source"]["image"] = None
        return response
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except OSError:
                pass


def _infer_uploaded_hand_image(image_bytes, hand_model_path):
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="hand_pose_upload_",
            suffix=".image",
            delete=False,
        ) as temporary:
            temporary.write(image_bytes)
            temporary_path = Path(temporary.name)

        frame = infer_single_hand_image(
            temporary_path,
            hand_model_path=hand_model_path,
        )
        response = _hand_pose_response(frame)
        response["source"]["image"] = None
        return response
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except OSError:
                pass


class PoseRequestHandler(BaseHTTPRequestHandler):
    """Handle health checks and one-image pose requests."""

    server_version = "DCCPoseServer/1.0"

    def _send_json(self, status, value):
        payload = _json_bytes(value)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path.rstrip("/") == "/health":
            self._send_json(
                200,
                {
                    "ok": True,
                    "service": "dcc-pose",
                    "protocol_version": PROTOCOL_VERSION,
                },
            )
            return
        self._send_json(404, {"ok": False, "error": "not_found"})

    def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler API
        endpoint = self.path.rstrip("/")
        if endpoint not in {"/v1/pose", "/v1/hand-pose"}:
            self._send_json(404, {"ok": False, "error": "not_found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0:
            self._send_json(400, {"ok": False, "error": "empty_request"})
            return
        if length > MAX_UPLOAD_BYTES:
            self._send_json(
                413,
                {
                    "ok": False,
                    "error": "image_too_large",
                    "max_bytes": MAX_UPLOAD_BYTES,
                },
            )
            return

        try:
            body = self.rfile.read(length)
            content_type = self.headers.get("Content-Type", "")
            image_bytes = _extract_image(content_type, body)
            if len(image_bytes) > MAX_UPLOAD_BYTES:
                raise PoseServiceError("图片文件超过大小限制。")
            if endpoint == "/v1/hand-pose":
                response = _infer_uploaded_hand_image(
                    image_bytes,
                    hand_model_path=self.server.hand_model_path,
                )
            else:
                response = _infer_uploaded_image(
                    image_bytes,
                    model_path=self.server.model_path,
                    hand_model_path=self.server.hand_model_path,
                )
        except (PoseServiceError, PoseLandmarkerError) as error:
            self._send_json(422, {"ok": False, "error": str(error)})
            return
        except Exception as error:  # Keep server alive for the next request.
            self._send_json(500, {"ok": False, "error": "服务内部错误：{}".format(error)})
            return

        self._send_json(200, response)

    def log_message(self, format_string, *args):
        print("[DCCPoseServer] " + format_string % args)


def make_server(
    host="127.0.0.1",
    port=18765,
    model_path=None,
    hand_model_path=None,
):
    resolved_model = find_model_path(model_path)
    if resolved_model is None:
        raise PoseServiceError(
            "没有找到 {}，请设置 MEDIAPIPE_POSE_MODEL 或把模型放入 ~/.mediapipe/models。".format(
                "pose_landmarker_full.task"
            )
        )
    resolved_hand_model = find_hand_model_path(hand_model_path)
    if resolved_hand_model is None:
        raise PoseServiceError(
            "没有找到 hand_landmarker.task，请设置 MEDIAPIPE_HAND_MODEL 或把模型放入 ~/.mediapipe/models。"
        )
    server = ThreadingHTTPServer((host, int(port)), PoseRequestHandler)
    server.model_path = str(resolved_model)
    server.hand_model_path = str(resolved_hand_model)
    return server


def main(argv=None):
    parser = argparse.ArgumentParser(description="DCC single-image pose service")
    parser.add_argument("--host", default=os.environ.get("POSE_SERVER_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("POSE_SERVER_PORT", "18765")))
    parser.add_argument("--model", default=os.environ.get("MEDIAPIPE_POSE_MODEL"))
    parser.add_argument("--hand-model", default=os.environ.get("MEDIAPIPE_HAND_MODEL"))
    args = parser.parse_args(argv)

    server = make_server(
        args.host,
        args.port,
        args.model,
        args.hand_model,
    )
    print("[DCCPoseServer] listening on http://{}:{}/v1/pose".format(args.host, args.port))
    print("[DCCPoseServer] model: {}".format(server.model_path))
    print("[DCCPoseServer] hand model: {}".format(server.hand_model_path))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[DCCPoseServer] stopping")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
