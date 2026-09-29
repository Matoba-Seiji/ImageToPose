"""Single-image MediaPipe body and hand landmark adapter.

This module intentionally does not import Maya.  It can run inside Maya's
Python interpreter when ``mediapipe`` is installed there, and it can also be
tested or hosted by another Python process without importing any Maya module.
The adapter turns MediaPipe's 33 body landmarks and up to two sets of 21 hand
landmarks into the 22 body and 30 finger joints expected by the Maya skeleton
writer.
"""

from __future__ import annotations

import math
import os
from pathlib import Path


MODEL_FILENAME = "pose_landmarker_full.task"
HAND_MODEL_FILENAME = "hand_landmarker.task"
MODEL_DOWNLOAD_URL = (
    "https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python"
)
HAND_MODEL_DOWNLOAD_URL = (
    "https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python"
)
MEDIAPIPE_POSE_VERSION = "mediapipe-pose-hand-landmarker-v3-head-orientation"


class PoseLandmarkerError(RuntimeError):
    """Raised when the MediaPipe model cannot produce a pose frame."""


LANDMARK_NAMES = (
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)

LANDMARK_INDEX = {name: index for index, name in enumerate(LANDMARK_NAMES)}

HAND_LANDMARK_NAMES = (
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
)

HAND_LANDMARK_INDEX = {
    name: index for index, name in enumerate(HAND_LANDMARK_NAMES)
}

# The current Maya skeleton exposes three joints per finger. Tips are retained
# in the landmark result but are not created as separate Maya joints.
FINGER_SEGMENT_INDEXES = {
    "thumb": ("thumb_cmc", "thumb_mcp", "thumb_ip"),
    "index": ("index_mcp", "index_pip", "index_dip"),
    "middle": ("middle_mcp", "middle_pip", "middle_dip"),
    "ring": ("ring_mcp", "ring_pip", "ring_dip"),
    "pinky": ("pinky_mcp", "pinky_pip", "pinky_dip"),
}


def _finite(value, label):
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise PoseLandmarkerError("{} must be numeric".format(label)) from error
    if not math.isfinite(number):
        raise PoseLandmarkerError("{} must be finite".format(label))
    return number


def _as_path(path):
    try:
        return Path(path).expanduser().resolve()
    except (TypeError, ValueError, OSError) as error:
        raise PoseLandmarkerError("invalid path: {}".format(path)) from error


def default_model_candidates(filename=MODEL_FILENAME):
    """Return model locations checked when the UI path is left blank."""

    candidates = []
    environment_name = (
        "MEDIAPIPE_HAND_MODEL"
        if filename == HAND_MODEL_FILENAME
        else "MEDIAPIPE_POSE_MODEL"
    )
    environment_path = os.environ.get(environment_name, "").strip()
    if environment_path:
        candidates.append(_as_path(environment_path))

    package_dir = Path(__file__).resolve().parent
    candidates.extend(
        (
            Path.home() / ".mediapipe" / "models" / filename,
            package_dir / "models" / filename,
            package_dir.parent / "models" / filename,
        )
    )
    return tuple(candidates)


def find_model_path(model_path=None, filename=MODEL_FILENAME):
    """Resolve an explicit or conventional local Pose Landmarker model path."""

    if model_path:
        candidate = _as_path(model_path)
        if candidate.is_file():
            return candidate
        raise PoseLandmarkerError(
            "找不到 MediaPipe 模型文件：{}\n请下载 {} 并在工具中选择 .task 文件。".format(
                candidate, filename
            )
        )

    for candidate in default_model_candidates(filename):
        if candidate.is_file():
            return candidate
    return None


def find_hand_model_path(model_path=None):
    """Resolve the local MediaPipe Hand Landmarker task model."""

    return find_model_path(model_path, filename=HAND_MODEL_FILENAME)


def _model_base_options(model, python):
    """Pass model bytes to MediaPipe so non-ASCII Windows paths also work."""

    try:
        model_bytes = model.read_bytes()
    except OSError as error:
        raise PoseLandmarkerError("无法读取 MediaPipe 模型：{}".format(model)) from error
    return python.BaseOptions(model_asset_buffer=model_bytes)


def _landmark_value(landmark, name, default=0.0):
    return _finite(getattr(landmark, name, default), "landmark {}".format(name))


def _landmark_position(landmark):
    return (
        _landmark_value(landmark, "x"),
        _landmark_value(landmark, "y"),
        _landmark_value(landmark, "z"),
    )


def _visibility(landmark):
    visibility = getattr(landmark, "visibility", 1.0)
    presence = getattr(landmark, "presence", 1.0)
    if visibility is None:
        visibility = 1.0
    if presence is None:
        presence = 1.0
    visibility = _finite(visibility, "visibility")
    presence = _finite(presence, "presence")
    return max(0.0, min(1.0, visibility, presence))


def _average(first, second):
    return tuple((a + b) * 0.5 for a, b in zip(first, second))


def _weighted_average(items):
    total_weight = sum(weight for _position, _confidence, weight in items)
    if total_weight <= 0.0:
        raise PoseLandmarkerError("无法计算头部中心：没有有效面部点。")
    return tuple(
        sum(position[axis] * weight for position, _confidence, weight in items)
        / total_weight
        for axis in range(3)
    )


def _head_center(points):
    """Estimate a cranial center from several stable face landmarks."""

    left_eye = points["left_eye"][0]
    right_eye = points["right_eye"][0]
    left_ear = points["left_ear"][0]
    right_ear = points["right_ear"][0]
    mouth_center = _average(
        points["mouth_left"][0], points["mouth_right"][0]
    )

    # Ears anchor the skull width; eyes, nose, and mouth keep the center stable
    # when one ear is hidden or the face is turned in profile.
    groups = (
        (_average(left_ear, right_ear), _min_confidence(points, "left_ear", "right_ear"), 0.45),
        (_average(left_eye, right_eye), _min_confidence(points, "left_eye", "right_eye"), 0.25),
        (points["nose"][0], points["nose"][1], 0.15),
        (mouth_center, _min_confidence(points, "mouth_left", "mouth_right"), 0.15),
    )
    center = _weighted_average(groups)
    confidence = sum(confidence * weight for _position, confidence, weight in groups)
    return center, confidence


def _lerp(first, second, amount):
    return tuple(a + (b - a) * amount for a, b in zip(first, second))


def _subtract(position, origin):
    return tuple(value - root for value, root in zip(position, origin))


def _joint_item(position, confidence, source=None):
    item = {
        "position": [float(value) for value in position],
        "visibility": float(max(0.0, min(1.0, confidence))),
    }
    if source:
        item["source"] = source
    return item


def _raw_landmarks(landmarks):
    if landmarks is None or len(landmarks) < len(LANDMARK_NAMES):
        raise PoseLandmarkerError(
            "MediaPipe 返回的姿态点不足 33 个，无法创建身体骨架。"
        )
    return {
        name: (
            _landmark_position(landmarks[index]),
            _visibility(landmarks[index]),
        )
        for name, index in LANDMARK_INDEX.items()
    }


def _raw_hand_landmarks(landmarks):
    if landmarks is None or len(landmarks) < len(HAND_LANDMARK_NAMES):
        raise PoseLandmarkerError(
            "MediaPipe 返回的手部姿态点不足 21 个，无法创建手指骨架。"
        )
    return {
        name: (
            _landmark_position(landmarks[index]),
            _visibility(landmarks[index]),
        )
        for name, index in HAND_LANDMARK_INDEX.items()
    }


def _landmark_xy(landmark):
    return (
        _landmark_value(landmark, "x"),
        _landmark_value(landmark, "y"),
    )


def _hand_side_from_category(category):
    if isinstance(category, (list, tuple)):
        category = category[0] if category else None
    label = getattr(category, "category_name", "") or getattr(category, "display_name", "")
    label = str(label).strip().lower()
    return label if label in ("left", "right") else None


def _assign_hand_sides(hand_image_landmarks, pose_image_landmarks, handedness):
    """Attach each detected hand to the person's left or right body wrist.

    Matching image-space wrists to the Pose Landmarker wrists avoids the
    mirrored-input ambiguity of the Hand Landmarker handedness category.
    """

    hand_count = min(len(hand_image_landmarks or ()), 2)
    if hand_count == 0:
        return []

    pose_wrists = {}
    if pose_image_landmarks is not None and len(pose_image_landmarks) >= len(LANDMARK_NAMES):
        for side in ("left", "right"):
            landmark = pose_image_landmarks[LANDMARK_INDEX[side + "_wrist"]]
            pose_wrists[side] = _landmark_xy(landmark)

    hand_wrists = [_landmark_xy(hand[HAND_LANDMARK_INDEX["wrist"]]) for hand in hand_image_landmarks[:hand_count]]
    if len(pose_wrists) == 2:
        distances = {
            index: {
                side: math.hypot(
                    hand_xy[0] - pose_xy[0],
                    hand_xy[1] - pose_xy[1],
                )
                for side, pose_xy in pose_wrists.items()
            }
            for index, hand_xy in enumerate(hand_wrists)
        }
        if hand_count == 1:
            return [min(distances[0], key=distances[0].get)]
        normal_cost = distances[0]["left"] + distances[1]["right"]
        swapped_cost = distances[0]["right"] + distances[1]["left"]
        if normal_cost <= swapped_cost:
            return ["left", "right"]
        return ["right", "left"]

    assigned = []
    for index, hand_xy in enumerate(hand_wrists):
        category_side = None
        if handedness is not None and index < len(handedness):
            category_side = _hand_side_from_category(handedness[index])
        fallback_side = "right" if hand_xy[0] < 0.5 else "left"
        side = category_side or fallback_side
        if side in assigned:
            side = "left" if side == "right" else "right"
        assigned.append(side)
    return assigned


def _build_finger_joints(
    points,
    hand_world_landmarks,
    hand_image_landmarks,
    pose_image_landmarks,
    handedness,
):
    """Convert detected hand landmarks into root-relative Maya finger joints."""

    if not hand_world_landmarks:
        return {}

    assignments = _assign_hand_sides(
        hand_image_landmarks,
        pose_image_landmarks,
        handedness,
    )
    finger_joints = {}
    used_sides = set()
    for hand_index, hand_world in enumerate(hand_world_landmarks[:2]):
        if hand_index >= len(assignments):
            break
        side = assignments[hand_index]
        if side in used_sides:
            continue
        try:
            hand_points = _raw_hand_landmarks(hand_world)
        except PoseLandmarkerError:
            continue

        pose_wrist_position, pose_wrist_confidence = points[side + "_wrist"]
        hand_wrist_position, hand_wrist_confidence = hand_points["wrist"]
        used_sides.add(side)
        for finger_name, segment_names in FINGER_SEGMENT_INDEXES.items():
            for segment, landmark_name in enumerate(segment_names, start=1):
                hand_position, hand_confidence = hand_points[landmark_name]
                position = tuple(
                    pose_wrist_position[axis]
                    + hand_position[axis]
                    - hand_wrist_position[axis]
                    for axis in range(3)
                )
                joint_name = "{}_{}_{}".format(side, finger_name, segment)
                finger_joints[joint_name] = _joint_item(
                    position,
                    min(pose_wrist_confidence, hand_wrist_confidence, hand_confidence),
                    source="hand_{}".format(landmark_name),
                )
    return finger_joints


def _min_confidence(points, *names):
    return min(points[name][1] for name in names)


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


def _vector_length(vector):
    if not all(math.isfinite(value) for value in vector):
        return None
    length = math.sqrt(sum(value * value for value in vector))
    return length if math.isfinite(length) else None


def _normalize_vector(vector):
    length = _vector_length(vector)
    if length is None or length <= _SPINE_CURVE_EPSILON:
        return None
    return tuple(value / length for value in vector)


def _angle_degrees(first, second):
    first_direction = _normalize_vector(first)
    second_direction = _normalize_vector(second)
    if first_direction is None or second_direction is None:
        return None
    cosine = sum(
        first_direction[axis] * second_direction[axis]
        for axis in range(3)
    )
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def _spine_positions(points, pelvis, shoulder_center):
    """Estimate gently curved spine joints when the leg axis is reliable."""

    torso_axis = _subtract(shoulder_center, pelvis)
    torso_length = _vector_length(torso_axis)
    torso_direction = _normalize_vector(torso_axis)
    linear_positions = {
        name: _lerp(pelvis, shoulder_center, amount)
        for name, amount in _SPINE_JOINT_PARAMETERS
    }
    torso_confidence = _min_confidence(
        points,
        "left_hip",
        "right_hip",
        "left_shoulder",
        "right_shoulder",
    )
    if (
        torso_confidence < _SPINE_CURVE_MIN_VISIBILITY
        or torso_length is None
        or torso_length <= _SPINE_CURVE_EPSILON
        or torso_direction is None
    ):
        return linear_positions

    leg_candidates = []
    for side in ("left", "right"):
        names = (side + "_hip", side + "_knee", side + "_ankle")
        confidence = _min_confidence(points, *names)
        if confidence < _SPINE_CURVE_MIN_VISIBILITY:
            continue

        hip = points[side + "_hip"][0]
        knee = points[side + "_knee"][0]
        ankle = points[side + "_ankle"][0]
        knee_angle = _angle_degrees(
            _subtract(hip, knee),
            _subtract(ankle, knee),
        )
        support_direction = _normalize_vector(_subtract(pelvis, ankle))
        if knee_angle is None or support_direction is None:
            continue

        # The most extended visible knee is the best available support-axis
        # estimate; use landmark confidence only to break close comparisons.
        leg_candidates.append((knee_angle, confidence, support_direction))

    if not leg_candidates:
        return linear_positions

    _knee_angle, _confidence, support_direction = max(
        leg_candidates, key=lambda candidate: (candidate[0], candidate[1])
    )
    bend_angle = _angle_degrees(torso_direction, support_direction)
    if bend_angle is None or bend_angle <= _SPINE_CURVE_START_ANGLE:
        return linear_positions

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
    curve_direction = _normalize_vector(perpendicular)
    if curve_direction is None:
        return linear_positions

    return {
        name: tuple(
            linear_positions[name][axis]
            + curve_direction[axis]
            * amplitude
            * 4.0
            * amount
            * (1.0 - amount)
            for axis in range(3)
        )
        for name, amount in _SPINE_JOINT_PARAMETERS
    }


def build_pose_frame_from_landmarks(
    world_landmarks,
    image_landmarks=None,
    image_path=None,
    hand_world_landmarks=None,
    hand_image_landmarks=None,
    handedness=None,
):
    """Build the Maya pose protocol from one MediaPipe result.

    MediaPipe world coordinates are measured in meters with the hip midpoint
    as origin.  They use a camera-style Y-down axis, so the Maya writer flips
    Y and Z when it receives ``root_relative_mediapipe_world``.
    """

    points = _raw_landmarks(world_landmarks)
    pelvis = _average(points["left_hip"][0], points["right_hip"][0])
    shoulder_center = _average(
        points["left_shoulder"][0], points["right_shoulder"][0]
    )
    head_center, head_confidence = _head_center(points)
    spine_positions = _spine_positions(points, pelvis, shoulder_center)

    # MediaPipe does not provide the extra torso and collar joints used by the
    # Maya hierarchy, so these joints are placed along the visible body lines.
    derived = {
        "pelvis": (pelvis, _min_confidence(points, "left_hip", "right_hip"), "hips"),
        "spine_1": (
            spine_positions["spine_1"],
            _min_confidence(points, "left_hip", "right_hip", "left_shoulder", "right_shoulder"),
            "interpolated_torso",
        ),
        "spine_2": (
            spine_positions["spine_2"],
            _min_confidence(points, "left_hip", "right_hip", "left_shoulder", "right_shoulder"),
            "interpolated_torso",
        ),
        "spine_3": (
            spine_positions["spine_3"],
            _min_confidence(points, "left_hip", "right_hip", "left_shoulder", "right_shoulder"),
            "interpolated_torso",
        ),
        "neck": (
            _lerp(shoulder_center, head_center, 0.35),
            min(
                _min_confidence(points, "left_shoulder", "right_shoulder"),
                head_confidence,
            ),
            "interpolated_neck_to_face_center",
        ),
        "head": (head_center, head_confidence, "multi_point_face_center"),
        "left_collar": (
            _lerp(shoulder_center, points["left_shoulder"][0], 0.45),
            _min_confidence(points, "left_shoulder", "right_shoulder"),
            "interpolated_collar",
        ),
        "right_collar": (
            _lerp(shoulder_center, points["right_shoulder"][0], 0.45),
            _min_confidence(points, "left_shoulder", "right_shoulder"),
            "interpolated_collar",
        ),
    }

    direct_joints = {
        "left_shoulder": "left_shoulder",
        "left_elbow": "left_elbow",
        "left_wrist": "left_wrist",
        "left_hip": "left_hip",
        "left_knee": "left_knee",
        "left_ankle": "left_ankle",
        "left_foot": "left_foot_index",
        "right_shoulder": "right_shoulder",
        "right_elbow": "right_elbow",
        "right_wrist": "right_wrist",
        "right_hip": "right_hip",
        "right_knee": "right_knee",
        "right_ankle": "right_ankle",
        "right_foot": "right_foot_index",
    }
    for joint_name, source_name in direct_joints.items():
        position, confidence = points[source_name]
        derived[joint_name] = (position, confidence, source_name)

    root_relative = {
        name: _joint_item(
            _subtract(position, pelvis), confidence, source=source
        )
        for name, (position, confidence, source) in derived.items()
    }
    finger_joints = _build_finger_joints(
        points,
        hand_world_landmarks,
        hand_image_landmarks,
        image_landmarks,
        handedness,
    )
    root_relative.update(finger_joints)

    face_names = ("left_eye", "right_eye", "left_ear", "right_ear",
                  "nose", "mouth_left", "mouth_right")
    face_world = {
        name: _joint_item(_subtract(points[name][0], pelvis), points[name][1])
        for name in face_names
    }

    if image_landmarks is not None and len(image_landmarks) >= len(LANDMARK_NAMES):
        image_points = _raw_landmarks(image_landmarks)
        image_data = {
            name: {
                "position": list(position),
                "visibility": confidence,
            }
            for name, (position, confidence) in image_points.items()
        }
    else:
        image_data = {}

    return {
        "skeleton": "maya_pose_skeleton",
        "version": MEDIAPIPE_POSE_VERSION,
        "coordinate_system": "root_relative_mediapipe_world",
        "unit": "meters",
        "include_fingers": bool(finger_joints),
        "joints": root_relative,
        "face_world": face_world,
        "landmarks": image_data,
        "source": {
            "provider": "MediaPipe Pose Landmarker",
            "image": str(image_path) if image_path else None,
            "landmark_count": len(LANDMARK_NAMES),
            "world_coordinates": True,
            "hands_detected": len(hand_world_landmarks or ()),
            "finger_joint_count": len(finger_joints),
        },
    }


def _import_mediapipe():
    try:
        import mediapipe as mp
        import cv2
        import numpy
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
    except ImportError as error:
        raise PoseLandmarkerError(
            "当前 Maya Python 缺少 MediaPipe 图像依赖。请使用 Maya 的 mayapy "
            "安装 mediapipe、opencv-contrib-python 和 numpy 后重启 Maya。"
        ) from error
    return mp, cv2, numpy, python, vision


def _load_image_from_unicode_path(image, mp, cv2, numpy):
    """Decode a Windows path in Python before handing pixels to MediaPipe.

    ``mp.Image.create_from_file`` can fail on non-ASCII Windows paths because
    the native image decoder receives an incorrectly encoded path. Python's
    file API handles the path correctly; ``imdecode`` then avoids passing the
    path through that native boundary a second time.
    """

    try:
        encoded = image.read_bytes()
    except OSError as error:
        raise PoseLandmarkerError(
            "无法读取人物图片：{}".format(image)
        ) from error
    buffer = numpy.frombuffer(encoded, dtype=numpy.uint8)
    bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if bgr is None or bgr.size == 0:
        raise PoseLandmarkerError(
            "图片解码失败，请确认文件没有损坏：{}".format(image)
        )

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    try:
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    except Exception as error:
        raise PoseLandmarkerError(
            "MediaPipe 无法创建图片数据：{}".format(error)
        ) from error


def infer_single_image(
    image_path,
    model_path=None,
    hand_model_path=None,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
):
    """Run body and hand landmarkers in IMAGE mode for one Maya pose frame."""

    image = _as_path(image_path)
    if not image.is_file():
        raise PoseLandmarkerError("找不到人物图片：{}".format(image))

    model = find_model_path(model_path)
    if model is None:
        raise PoseLandmarkerError(
            "没有找到 {}。请下载模型文件并在 Maya 工具的“模型文件”中选择它：\n{}".format(
                MODEL_FILENAME, MODEL_DOWNLOAD_URL
            )
        )
    hand_model = find_hand_model_path(hand_model_path)
    if hand_model is None:
        raise PoseLandmarkerError(
            "没有找到 {}。请将手部模型放到 C:\\Users\\<用户名>\\.mediapipe\\models，"
            "或设置 MEDIAPIPE_HAND_MODEL。下载说明：{}".format(
                HAND_MODEL_FILENAME, HAND_MODEL_DOWNLOAD_URL
            )
        )

    mp, cv2, numpy, python, vision = _import_mediapipe()
    try:
        options = vision.PoseLandmarkerOptions(
            base_options=_model_base_options(model, python),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=float(min_pose_detection_confidence),
            min_pose_presence_confidence=float(min_pose_presence_confidence),
            min_tracking_confidence=float(min_tracking_confidence),
        )
        mp_image = _load_image_from_unicode_path(image, mp, cv2, numpy)
        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            result = landmarker.detect(mp_image)
    except PoseLandmarkerError:
        raise
    except Exception as error:
        raise PoseLandmarkerError(
            "MediaPipe 姿态识别失败：{}".format(error)
        ) from error

    world_poses = getattr(result, "pose_world_landmarks", None) or []
    image_poses = getattr(result, "pose_landmarks", None) or []
    if not world_poses:
        raise PoseLandmarkerError(
            "图片中没有检测到完整的人体。请使用全身图片，并减少遮挡。"
        )

    hand_options = vision.HandLandmarkerOptions(
        base_options=_model_base_options(hand_model, python),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=float(min_hand_detection_confidence),
        min_hand_presence_confidence=float(min_hand_presence_confidence),
        min_tracking_confidence=float(min_tracking_confidence),
    )
    try:
        with vision.HandLandmarker.create_from_options(hand_options) as hand_landmarker:
            hand_result = hand_landmarker.detect(mp_image)
    except PoseLandmarkerError:
        raise
    except Exception as error:
        raise PoseLandmarkerError(
            "MediaPipe 手部姿态识别失败：{}".format(error)
        ) from error

    world_landmarks = world_poses[0]
    image_landmarks = image_poses[0] if image_poses else None
    hand_world_landmarks = getattr(hand_result, "hand_world_landmarks", None) or []
    hand_image_landmarks = getattr(hand_result, "hand_landmarks", None) or []
    handedness = getattr(hand_result, "handedness", None) or []
    return build_pose_frame_from_landmarks(
        world_landmarks,
        image_landmarks=image_landmarks,
        image_path=image,
        hand_world_landmarks=hand_world_landmarks,
        hand_image_landmarks=hand_image_landmarks,
        handedness=handedness,
    )


__all__ = [
    "HAND_MODEL_FILENAME",
    "HAND_MODEL_DOWNLOAD_URL",
    "HAND_LANDMARK_NAMES",
    "LANDMARK_NAMES",
    "MODEL_DOWNLOAD_URL",
    "MODEL_FILENAME",
    "PoseLandmarkerError",
    "build_pose_frame_from_landmarks",
    "default_model_candidates",
    "find_hand_model_path",
    "find_model_path",
    "infer_single_image",
]
