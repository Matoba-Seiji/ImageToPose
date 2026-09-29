"""3ds Max client for the shared single-image pose service.

This module uses only Python's standard library for HTTP and JSON. MediaPipe
stays on the pose server, so 3ds Max 2025 does not need AI dependencies.
The scene writer uses pymxs only when called from inside 3ds Max.
"""

from __future__ import annotations

import json
import math
import mimetypes
import os
import re
import subprocess
import time
import uuid
from pathlib import Path
from urllib import error as urllib_error
from urllib.parse import urlparse
from urllib import request as urllib_request


BODY_JOINTS = (
    "pelvis",
    "spine_1",
    "spine_2",
    "spine_3",
    "neck",
    "head",
    "left_collar",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_collar",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "left_hip",
    "left_knee",
    "left_ankle",
    "left_foot",
    "right_hip",
    "right_knee",
    "right_ankle",
    "right_foot",
)

BODY_PARENTS = {
    "spine_1": "pelvis",
    "spine_2": "spine_1",
    "spine_3": "spine_2",
    "neck": "spine_3",
    "head": "neck",
    "left_collar": "spine_3",
    "left_shoulder": "left_collar",
    "left_elbow": "left_shoulder",
    "left_wrist": "left_elbow",
    "right_collar": "spine_3",
    "right_shoulder": "right_collar",
    "right_elbow": "right_shoulder",
    "right_wrist": "right_elbow",
    "left_hip": "pelvis",
    "left_knee": "left_hip",
    "left_ankle": "left_knee",
    "left_foot": "left_ankle",
    "right_hip": "pelvis",
    "right_knee": "right_hip",
    "right_ankle": "right_knee",
    "right_foot": "right_ankle",
}

FINGER_JOINTS = tuple(
    "{}_{}_{}".format(side, finger, segment)
    for side in ("left", "right")
    for finger in ("thumb", "index", "middle", "ring", "pinky")
    for segment in (1, 2, 3)
)
BODY_AND_FINGER_JOINTS = BODY_JOINTS + FINGER_JOINTS
for _side in ("left", "right"):
    for _finger in ("thumb", "index", "middle", "ring", "pinky"):
        BODY_PARENTS["{}_{}_1".format(_side, _finger)] = "{}_wrist".format(_side)
        BODY_PARENTS["{}_{}_2".format(_side, _finger)] = "{}_{}_1".format(
            _side, _finger
        )
        BODY_PARENTS["{}_{}_3".format(_side, _finger)] = "{}_{}_2".format(
            _side, _finger
        )


HAND_JOINTS = (
    "wrist",
    "thumb_1", "thumb_2", "thumb_3",
    "index_1", "index_2", "index_3",
    "middle_1", "middle_2", "middle_3",
    "ring_1", "ring_2", "ring_3",
    "pinky_1", "pinky_2", "pinky_3",
)
DEFAULT_SERVER_URL = "http://127.0.0.1:18765/v1/pose"
DEFAULT_HAND_SERVER_URL = "http://127.0.0.1:18765/v1/hand-pose"
LOCAL_SERVER_HEALTH_URL = "http://127.0.0.1:18765/health"
LOCAL_SERVER_PORT = 18765
BODY_NAME_PREFIX = "MediaPipePoseSkeleton_"
MARKER_KEY = "poseToolGenerated"
BODY_MARKER_VALUE = "3dsmax_mediapipe_pose_tool"
FACE_METADATA_KEY = "poseToolFaceWorld"
FACE_WORLD_NAMES = ("left_eye", "right_eye", "left_ear", "right_ear",
                    "nose", "mouth_left", "mouth_right")
_SPINE_CURVE_MIN_VISIBILITY = 0.5
_SPINE_CURVE_START_ANGLE = 12.0
_SPINE_CURVE_FULL_ANGLE = 45.0
_SPINE_CURVE_MAX_OFFSET_RATIO = 0.08
_SPINE_CURVE_EPSILON = 1e-8
_SPINE_JOINT_PARAMETERS = (
    ("spine_1", 0.25),
    ("spine_2", 0.55),
    ("spine_3", 0.80),
)


class MaxPoseError(RuntimeError):
    """Raised when the server response or 3ds Max scene cannot be used."""


def _is_local_endpoint(endpoint):
    parsed = urlparse(endpoint)
    return (
        parsed.scheme in {"http", "https"}
        and parsed.hostname in {"127.0.0.1", "localhost"}
        and (parsed.port or 80) == LOCAL_SERVER_PORT
    )


def _connection_refused(error):
    reason = getattr(error, "reason", error)
    winerror = getattr(reason, "winerror", None)
    text = str(reason).lower()
    return (
        isinstance(reason, ConnectionRefusedError)
        or winerror == 10061
        or "10061" in text
        or "refused" in text
    )


def _start_local_pose_server():
    """Start the bundled local server in the background when it is missing."""

    executable = _local_server_executable()
    if executable is None:
        return False

    command = [
        str(executable),
        "--host",
        "127.0.0.1",
        "--port",
        str(LOCAL_SERVER_PORT),
    ]
    try:
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        subprocess.Popen(
            command,
            cwd=str(executable.parent),
            env=os.environ.copy(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creation_flags,
        )
    except (OSError, ValueError):
        return False
    return True


def _local_server_executable():
    """Find the installed runtime or the development build beside this project."""

    configured = os.environ.get("POSE_SERVER_EXE", "").strip()
    if configured:
        path = Path(configured).expanduser().resolve()
        return path if path.is_file() else None

    module_dir = Path(__file__).resolve().parent
    for root in (module_dir, *list(module_dir.parents)[:8]):
        for relative in (
            Path("PoseRuntime") / "jade_pose_server.exe",
            Path("dist") / "jade_pose_server" / "jade_pose_server.exe",
        ):
            path = root / relative
            if path.is_file():
                return path
    return None


def _wait_for_local_pose_server(timeout=30.0):
    deadline = time.time() + float(timeout)
    while time.time() < deadline:
        try:
            with urllib_request.urlopen(LOCAL_SERVER_HEALTH_URL, timeout=1.0) as response:
                return response.status == 200
        except Exception:
            time.sleep(0.25)
    return False


def server_url(value=None, mode="body"):
    """Return the configured body or hand endpoint."""

    if mode not in {"body", "hand"}:
        raise MaxPoseError("不支持的姿态模式：{}".format(mode))
    default = DEFAULT_SERVER_URL if mode == "body" else DEFAULT_HAND_SERVER_URL
    environment_name = "POSE_SERVER_URL" if mode == "body" else "HAND_POSE_SERVER_URL"
    value = value or os.environ.get(environment_name) or default
    value = value.rstrip("/")
    if mode == "body":
        if value.endswith("/v1/hand-pose"):
            value = value[: -len("/v1/hand-pose")] + "/v1/pose"
        elif not value.endswith("/v1/pose"):
            value += "/v1/pose"
    elif value.endswith("/v1/pose"):
        value = value[: -len("/v1/pose")] + "/v1/hand-pose"
    elif not value.endswith("/v1/hand-pose"):
        value += "/v1/hand-pose"
    return value


def _multipart_body(image_path):
    path = Path(image_path).expanduser().resolve()
    if not path.is_file():
        raise MaxPoseError("找不到人物图片：{}".format(path))
    try:
        image_bytes = path.read_bytes()
    except OSError as error:
        raise MaxPoseError("无法读取人物图片：{}".format(path)) from error
    if not image_bytes:
        raise MaxPoseError("图片文件为空：{}".format(path))

    boundary = "----DCCPose{}".format(uuid.uuid4().hex)
    content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    header = (
        "--{boundary}\r\n"
        "Content-Disposition: form-data; name=\"image\"; filename=\"{filename}\"\r\n"
        "Content-Type: {content_type}\r\n"
        "\r\n"
    ).format(
        boundary=boundary,
        filename=path.name,
        content_type=content_type,
    ).encode("utf-8")
    footer = "\r\n--{}--\r\n".format(boundary).encode("ascii")
    return header + image_bytes + footer, "multipart/form-data; boundary={}".format(boundary)


def _request_frame(image_path, endpoint=None, timeout=120, mode="body"):
    """Upload one image to a selected MediaPipe endpoint."""

    body, content_type = _multipart_body(image_path)
    endpoint_url = server_url(endpoint, mode=mode)

    def send_request():
        request = urllib_request.Request(
            endpoint_url,
            data=body,
            headers={
                "Content-Type": content_type,
                "Content-Length": str(len(body)),
                "Accept": "application/json",
                "User-Agent": "3dsmax-mediapipe-tool/2.0",
            },
            method="POST",
        )
        with urllib_request.urlopen(request, timeout=float(timeout)) as response:
            return response.read()

    try:
        payload = send_request()
    except urllib_error.HTTPError as error:
        try:
            detail = error.read().decode("utf-8", "replace")
        except Exception:
            detail = str(error)
        raise MaxPoseError(
            "姿态服务返回错误 {}：{}".format(error.code, detail)
        ) from error
    except urllib_error.URLError as error:
        if _connection_refused(error) and _is_local_endpoint(endpoint_url):
            if _start_local_pose_server() and _wait_for_local_pose_server():
                try:
                    payload = send_request()
                except urllib_error.HTTPError as retry_error:
                    try:
                        detail = retry_error.read().decode("utf-8", "replace")
                    except Exception:
                        detail = "HTTP {}".format(retry_error.code)
                    raise MaxPoseError(
                        "本地姿态服务返回错误 {}：{}".format(
                            retry_error.code,
                            detail,
                        )
                    ) from retry_error
                except Exception as retry_error:
                    raise MaxPoseError(
                        "本地姿态服务已启动，但图片请求失败。"
                    ) from retry_error
            else:
                raise MaxPoseError(
                    "本地姿态服务不可用（127.0.0.1:{}）。"
                    "请确认已安装 PoseRuntime/jade_pose_server.exe。".format(
                        LOCAL_SERVER_PORT
                    )
                ) from error
        else:
            raise MaxPoseError(
                "姿态服务连接失败：{}".format(endpoint_url)
            ) from error
    except Exception as error:
        raise MaxPoseError("请求姿态服务失败：{}".format(error)) from error

    try:
        frame = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MaxPoseError("姿态服务返回的数据不是有效 JSON。") from error
    if not isinstance(frame, dict) or frame.get("ok") is False:
        raise MaxPoseError(str(frame.get("error", "姿态服务返回了无效结果")))
    _validate_frame(frame, mode=mode)
    if mode == "body":
        _apply_spine_curve_if_linear(frame)
    return frame


def request_pose(image_path, endpoint=None, timeout=120):
    """Upload one body image and return the DCC-neutral body frame."""

    return _request_frame(image_path, endpoint=endpoint, timeout=timeout, mode="body")


def request_hand_pose(image_path, endpoint=None, timeout=120):
    """Upload one hand image and return the DCC-neutral hand frame."""

    return _request_frame(image_path, endpoint=endpoint, timeout=timeout, mode="hand")


def _validate_frame(frame, mode="body"):
    if mode not in {"body", "hand"}:
        raise MaxPoseError("不支持的姿态骨架模式：{}".format(mode))
    joints = frame.get("joints")
    if not isinstance(joints, dict):
        raise MaxPoseError("姿态服务没有返回 joints 数据。")
    if mode == "body":
        expected = _body_frame_joint_names(frame)
    else:
        expected = HAND_JOINTS
    expected_skeleton = "maya_pose_skeleton" if mode == "body" else "maya_hand_skeleton"
    if frame.get("skeleton") != expected_skeleton:
        raise MaxPoseError("姿态服务返回的骨架类型不匹配。")
    missing = [name for name in expected if name not in joints]
    if missing:
        raise MaxPoseError("姿态服务缺少关节：{}".format(", ".join(missing)))
    for name in expected:
        item = joints[name]
        position = item.get("position") if isinstance(item, dict) else None
        if not isinstance(position, (list, tuple)) or len(position) != 3:
            raise MaxPoseError("关节 {} 的 position 无效。".format(name))
        try:
            values = [float(value) for value in position]
        except (TypeError, ValueError) as error:
            raise MaxPoseError("关节 {} 的 position 不是数字。".format(name)) from error
        if not all(math.isfinite(value) for value in values):
            raise MaxPoseError("关节 {} 的 position 不是有限数字。".format(name))


def _body_frame_joint_names(frame):
    """Return body joints plus only complete finger chains in the response."""

    joints = frame.get("joints") or {}
    names = list(BODY_JOINTS)
    if frame.get("include_fingers"):
        for side in ("left", "right"):
            for finger in ("thumb", "index", "middle", "ring", "pinky"):
                chain = tuple(
                    "{}_{}_{}".format(side, finger, segment)
                    for segment in (1, 2, 3)
                )
                if all(name in joints for name in chain):
                    names.extend(chain)
    return tuple(names)


def _runtime(runtime=None):
    if runtime is not None:
        return runtime
    try:
        from pymxs import runtime as max_runtime
    except ImportError as error:
        raise MaxPoseError("请在 3ds Max 的 Python 环境中运行骨架创建器。") from error
    return max_runtime


def _valid_node(runtime, node):
    if node is None:
        return False
    try:
        return bool(runtime.isValidNode(node))
    except Exception:
        return str(node) != "undefined"


def _point(runtime, values):
    return runtime.Point3(float(values[0]), float(values[1]), float(values[2]))


def _distance(first, second):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(first, second)))


def _spine_unit(vector):
    if not all(math.isfinite(value) for value in vector):
        return None
    length = _distance(vector, (0.0, 0.0, 0.0))
    if not math.isfinite(length) or length <= _SPINE_CURVE_EPSILON:
        return None
    return tuple(value / length for value in vector)


def _spine_angle(first, second):
    first_direction = _spine_unit(first)
    second_direction = _spine_unit(second)
    if first_direction is None or second_direction is None:
        return None
    cosine = sum(
        first_direction[axis] * second_direction[axis]
        for axis in range(3)
    )
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def _joint_visibility(joints, name):
    try:
        visibility = float(joints[name].get("visibility", 0.0))
    except (AttributeError, TypeError, ValueError, OverflowError):
        return 0.0
    if not math.isfinite(visibility):
        return 0.0
    return max(0.0, min(1.0, visibility))


def _apply_spine_curve_if_linear(frame):
    """Apply Maya's spine curve to linear frames from older pose runtimes.

    The shared service already emits curved spine joints after the server
    update. Keeping this client-side fallback lets existing frozen Max
    runtimes produce the same result without curving new-service frames twice.
    """

    if (
        frame.get("skeleton") != "maya_pose_skeleton"
        or frame.get("coordinate_system") != "root_relative_mediapipe_world"
    ):
        return frame

    joints = frame.get("joints")
    if not isinstance(joints, dict):
        return frame

    required = (
        "pelvis",
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "spine_1",
        "spine_2",
        "spine_3",
    )
    if any(name not in joints for name in required):
        return frame

    try:
        position = {
            name: tuple(float(value) for value in joints[name]["position"])
            for name in required
        }
        if any(len(position[name]) != 3 for name in required):
            return frame
        if any(
            not all(math.isfinite(value) for value in position[name])
            for name in required
        ):
            return frame
    except (KeyError, TypeError, ValueError, OverflowError):
        return frame

    pelvis = position["pelvis"]
    shoulder_center = tuple(
        (position["left_shoulder"][axis] + position["right_shoulder"][axis])
        * 0.5
        for axis in range(3)
    )
    torso_axis = tuple(
        shoulder_center[axis] - pelvis[axis] for axis in range(3)
    )
    torso_length = _distance(torso_axis, (0.0, 0.0, 0.0))
    torso_direction = _spine_unit(torso_axis)
    torso_confidence = min(
        _joint_visibility(joints, name)
        for name in (
            "left_hip",
            "right_hip",
            "left_shoulder",
            "right_shoulder",
        )
    )
    if (
        torso_confidence < _SPINE_CURVE_MIN_VISIBILITY
        or not math.isfinite(torso_length)
        or torso_length <= _SPINE_CURVE_EPSILON
        or torso_direction is None
    ):
        return frame

    # Frames already curved by the current service are left untouched.
    for name, amount in _SPINE_JOINT_PARAMETERS:
        linear_position = tuple(
            pelvis[axis] + torso_axis[axis] * amount
            for axis in range(3)
        )
        if _distance(position[name], linear_position) > _SPINE_CURVE_EPSILON:
            return frame

    leg_candidates = []
    for side in ("left", "right"):
        names = (side + "_hip", side + "_knee", side + "_ankle")
        confidence = min(_joint_visibility(joints, name) for name in names)
        if confidence < _SPINE_CURVE_MIN_VISIBILITY:
            continue

        hip = position[side + "_hip"]
        knee = position[side + "_knee"]
        ankle = position[side + "_ankle"]
        knee_angle = _spine_angle(
            tuple(hip[axis] - knee[axis] for axis in range(3)),
            tuple(ankle[axis] - knee[axis] for axis in range(3)),
        )
        support_direction = _spine_unit(
            tuple(pelvis[axis] - ankle[axis] for axis in range(3))
        )
        if knee_angle is None or support_direction is None:
            continue
        leg_candidates.append((knee_angle, confidence, support_direction))

    if not leg_candidates:
        return frame

    _knee_angle, _confidence, support_direction = max(
        leg_candidates,
        key=lambda candidate: (candidate[0], candidate[1]),
    )
    bend_angle = _spine_angle(torso_direction, support_direction)
    if bend_angle is None or bend_angle <= _SPINE_CURVE_START_ANGLE:
        return frame

    bend_strength = min(
        1.0,
        (bend_angle - _SPINE_CURVE_START_ANGLE)
        / (_SPINE_CURVE_FULL_ANGLE - _SPINE_CURVE_START_ANGLE),
    )
    amplitude = torso_length * _SPINE_CURVE_MAX_OFFSET_RATIO * bend_strength
    projection = sum(
        support_direction[axis] * torso_direction[axis]
        for axis in range(3)
    )
    perpendicular = tuple(
        support_direction[axis] - projection * torso_direction[axis]
        for axis in range(3)
    )
    curve_direction = _spine_unit(perpendicular)
    if curve_direction is None:
        return frame

    for name, amount in _SPINE_JOINT_PARAMETERS:
        offset = amplitude * 4.0 * amount * (1.0 - amount)
        curved_position = [
            pelvis[axis]
            + torso_axis[axis] * amount
            + curve_direction[axis] * offset
            for axis in range(3)
        ]
        joints[name] = dict(joints[name], position=curved_position)
    return frame


def _normalize(vector):
    length = math.sqrt(sum(value * value for value in vector))
    if length <= 1e-8:
        return (0.0, 0.0, 1.0)
    return tuple(value / length for value in vector)


def _position_map(frame, runtime, joint_names=BODY_JOINTS):
    coordinate_system = frame.get("coordinate_system")
    if coordinate_system not in {
        "root_relative_mediapipe_world",
        "root_relative_mediapipe_hand_world",
        "root_relative_y_up",
        "root_relative_camera_y_down",
        "root_relative_camera_xyz",
    }:
        raise MaxPoseError(
            "不支持的姿态坐标系：{}".format(coordinate_system)
        )

    try:
        units_per_meter = float(runtime.units.decodeValue("1m"))
    except Exception:
        # 3ds Max's common default is centimeters. This fallback only applies
        # when a test double or an older runtime lacks units.decodeValue.
        units_per_meter = 100.0

    positions = {}
    for name in joint_names:
        source = frame["joints"][name]["position"]
        x, y, z = (float(value) for value in source)
        if coordinate_system in {
            "root_relative_mediapipe_world",
            "root_relative_mediapipe_hand_world",
        }:
            # 3ds Max is Z-up, and this rig uses negative Y as its forward
            # direction. MediaPipe's depth axis therefore maps directly to
            # Max Y: Max = (MediaPipe X, MediaPipe Z, -MediaPipe Y).
            converted = (x, z, -y)
        elif coordinate_system == "root_relative_y_up":
            converted = (x, -z, y)
        else:
            converted = (x, -z, -y)
        positions[name] = tuple(value * units_per_meter for value in converted)
    return positions, units_per_meter


def _face_position_map(frame, runtime):
    """Convert optional world-space face landmarks; old frames remain valid."""
    if not isinstance(frame, dict):
        return None
    face = frame.get("face_world")
    if not isinstance(face, dict) or any(name not in face for name in FACE_WORLD_NAMES):
        return None
    try:
        for name in FACE_WORLD_NAMES:
            item = face[name]
            visibility = float(item.get("visibility", 0.))
            if not math.isfinite(visibility) or visibility < .5:
                return None
            position = item["position"]
            if len(position) != 3 or not all(math.isfinite(float(v)) for v in position):
                return None
        converted, _ = _position_map(
            {"coordinate_system": frame["coordinate_system"], "joints": face},
            runtime, joint_names=FACE_WORLD_NAMES,
        )
        return converted
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError, MaxPoseError):
        return None


def _children_map(joint_names=BODY_JOINTS, parents=BODY_PARENTS):
    joint_names = set(joint_names)
    children = {name: [] for name in joint_names}
    for child, parent in parents.items():
        if child in joint_names and parent in joint_names:
            children[parent].append(child)
    return children


def _leaf_end(name, positions, parent_name, units_per_meter):
    current = positions[name]
    if parent_name in positions:
        direction = _normalize(
            tuple(current[index] - positions[parent_name][index] for index in range(3))
        )
        length = max(
            _distance(current, positions[parent_name]) * 0.35,
            units_per_meter * 0.05,
        )
    else:
        direction = (0.0, 0.0, 1.0)
        length = units_per_meter * 0.1
    return tuple(current[index] + direction[index] * length for index in range(3))


def _set_marker(runtime, node, marker_value=BODY_MARKER_VALUE):
    try:
        runtime.setUserProp(node, MARKER_KEY, marker_value)
    except Exception:
        pass


def _get_marker(runtime, node):
    try:
        return str(runtime.getUserProp(node, MARKER_KEY) or "")
    except Exception:
        return ""


def _delete_previous(
    runtime,
    prefix=BODY_NAME_PREFIX,
    joint_names=BODY_JOINTS,
    marker_value=BODY_MARKER_VALUE,
    strict=False,
):
    expected = set(prefix + name for name in joint_names)
    pattern = re.compile(
        "^{}(?:{}|[0-9]+_(?:{}))$".format(
            re.escape(prefix),
            "|".join(re.escape(name) for name in joint_names),
            "|".join(re.escape(name) for name in joint_names),
        )
    )
    nodes = []
    try:
        all_nodes = list(runtime.objects)
    except Exception:
        if strict:
            raise
        all_nodes = []
    for node in all_nodes:
        try:
            node_name = str(node.name)
        except Exception:
            if strict:
                raise
            continue

        if node_name in expected or pattern.match(node_name) or _get_marker(runtime, node) == marker_value:
            if _valid_node(runtime, node) and node not in nodes:
                nodes.append(node)

    for node in nodes:
        try:
            if _valid_node(runtime, node):
                runtime.delete(node)
        except Exception:
            if strict:
                raise
            continue

    if strict and any(_valid_node(runtime, node) for node in nodes):
        raise MaxPoseError("Could not remove the previous generated body skeleton")


def create_max_skeleton(
    frame,
    runtime=None,
    prefix=BODY_NAME_PREFIX,
    replace_previous=True,
    joint_names=None,
    parents=BODY_PARENTS,
    marker_value=BODY_MARKER_VALUE,
):
    """Create or replace the full-body preview skeleton in 3ds Max."""

    return _create_bone_skeleton(
        frame,
        runtime=runtime,
        prefix=prefix,
        replace_previous=replace_previous,
        joint_names=joint_names or _body_frame_joint_names(frame),
        parents=parents,
        marker_value=marker_value,
    )


def _create_bone_skeleton(
    frame,
    runtime=None,
    prefix=BODY_NAME_PREFIX,
    replace_previous=True,
    joint_names=BODY_JOINTS,
    parents=BODY_PARENTS,
    marker_value=BODY_MARKER_VALUE,
):
    """Create one DCC-neutral bone hierarchy in 3ds Max."""

    runtime = _runtime(runtime)
    joint_names = tuple(joint_names)
    _validate_frame(frame, mode="body")
    positions, units_per_meter = _position_map(
        frame,
        runtime,
        joint_names=joint_names,
    )
    if replace_previous:
        _delete_previous(
            runtime,
            prefix=prefix,
            joint_names=joint_names,
            marker_value=marker_value,
        )

    children = _children_map(joint_names=joint_names, parents=parents)
    bones = {}
    try:
        for name in joint_names:
            parent_name = parents.get(name)
            child_names = children[name]
            end_name = child_names[0] if child_names else None
            if end_name:
                end = positions[end_name]
            else:
                end = _leaf_end(name, positions, parent_name, units_per_meter)
            start_point = _point(runtime, positions[name])
            end_point = _point(runtime, end)
            bone = runtime.BoneSys.createBone(
                start_point,
                end_point,
                runtime.Point3(0.0, 0.0, 1.0),
            )
            if not _valid_node(runtime, bone):
                raise MaxPoseError("3ds Max 无法创建骨骼：{}".format(name))
            bone.name = prefix + name
            try:
                bone.width = max(units_per_meter * 0.025, 0.1)
                bone.height = max(units_per_meter * 0.025, 0.1)
            except Exception:
                pass
            _set_marker(runtime, bone, marker_value=marker_value)
            bones[name] = bone

        for child_name, parent_name in parents.items():
            if child_name in bones and parent_name in bones:
                bones[child_name].parent = bones[parent_name]
        if isinstance(frame.get("face_world"), dict):
            metadata = {"coordinate_system": frame.get("coordinate_system"),
                        "face_world": frame["face_world"]}
            runtime.setUserProp(
                bones[joint_names[0]], FACE_METADATA_KEY,
                json.dumps(metadata, ensure_ascii=False, separators=(",", ":")),
            )
    except Exception as error:
        for node in reversed(list(bones.values())):
            try:
                if _valid_node(runtime, node):
                    runtime.delete(node)
            except Exception:
                pass
        if isinstance(error, MaxPoseError):
            raise
        raise MaxPoseError("3ds Max 骨骼创建失败：{}".format(error)) from error

    return {
        "success": True,
        "root": bones[joint_names[0]],
        "joint_count": len(bones),
        "joints": bones,
        "name_prefix": prefix,
        "units_per_meter": units_per_meter,
        "coordinate_system": frame.get("coordinate_system"),
        "handedness": frame.get("handedness"),
        "scene_changes": (
            "replaced_previous_generated_skeleton"
            if replace_previous
            else "created_skeleton_only"
        ),
    }


__all__ = [
    "BODY_JOINTS",
    "BODY_PARENTS",
    "BODY_AND_FINGER_JOINTS",
    "FINGER_JOINTS",
    "DEFAULT_SERVER_URL",
    "DEFAULT_HAND_SERVER_URL",
    "HAND_JOINTS",
    "MaxPoseError",
    "create_max_skeleton",
    "request_pose",
    "request_hand_pose",
    "server_url",
]
