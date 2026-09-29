"""Single-image MediaPipe Hand Landmarker adapter.

Unlike the body adapter, this module does not require Pose Landmarker or a
body image. It selects one hand from a clear hand image, roots all 21 hand
landmarks at the wrist, and returns 15 finger joints plus the fingertip
landmarks consumed by the Max client.
"""

from __future__ import annotations

from .mediapipe_pose import (
    HAND_LANDMARK_INDEX,
    HAND_LANDMARK_NAMES,
    FINGER_SEGMENT_INDEXES,
    PoseLandmarkerError,
    _as_path,
    _finite,
    _import_mediapipe,
    _joint_item,
    _landmark_value,
    _load_image_from_unicode_path,
    _model_base_options,
    _raw_hand_landmarks,
    find_hand_model_path,
    HAND_MODEL_FILENAME,
    HAND_MODEL_DOWNLOAD_URL,
)


HAND_POSE_VERSION = "mediapipe-hand-landmarker-v1"
HAND_JOINT_NAMES = ("wrist",) + tuple(
    "{}_{}".format(finger, segment)
    for finger, segments in FINGER_SEGMENT_INDEXES.items()
    for segment in (1, 2, 3)
)
HAND_PARENTS = {
    "{}_1".format(finger): "wrist"
    for finger in FINGER_SEGMENT_INDEXES
}
for _finger in FINGER_SEGMENT_INDEXES:
    HAND_PARENTS["{}_2".format(_finger)] = "{}_1".format(_finger)
    HAND_PARENTS["{}_3".format(_finger)] = "{}_2".format(_finger)


def _handedness_label(category):
    if isinstance(category, (list, tuple)):
        category = category[0] if category else None
    label = getattr(category, "category_name", "") or getattr(category, "display_name", "")
    label = str(label).strip().lower()
    return label if label in {"left", "right"} else "unknown"


def _handedness_score(category):
    if isinstance(category, (list, tuple)):
        category = category[0] if category else None
    value = getattr(category, "score", 1.0)
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = 1.0
    return max(0.0, min(1.0, value))


def _hand_point_item(position, confidence, source=None):
    return _joint_item(position, confidence, source=source)


def _select_hand(hand_world_landmarks, hand_image_landmarks, handedness):
    count = min(
        len(hand_world_landmarks or ()),
        len(hand_image_landmarks or ()),
    )
    if count <= 0:
        raise PoseLandmarkerError(
            "图片中没有返回可用的手部三维关键点。请使用清晰的单手图片，并确保手腕完整可见。"
        )

    scores = []
    for index in range(count):
        category = handedness[index] if handedness and index < len(handedness) else None
        scores.append(_handedness_score(category))
    selected = max(range(count), key=lambda index: scores[index])
    category = handedness[selected] if handedness and selected < len(handedness) else None
    return (
        hand_world_landmarks[selected],
        hand_image_landmarks[selected],
        _handedness_label(category),
        scores[selected],
        count,
    )


def infer_single_hand_image(
    image_path,
    hand_model_path=None,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
):
    """Run only MediaPipe Hand Landmarker on one image."""

    image = _as_path(image_path)
    if not image.is_file():
        raise PoseLandmarkerError("找不到手部图片：{}".format(image))

    hand_model = find_hand_model_path(hand_model_path)
    if hand_model is None:
        raise PoseLandmarkerError(
            "没有找到 {}。请将手部模型放到 C:\\Users\\<用户名>\\.mediapipe\\models，"
            "或设置 MEDIAPIPE_HAND_MODEL。下载说明：{}".format(
                HAND_MODEL_FILENAME,
                HAND_MODEL_DOWNLOAD_URL,
            )
        )

    mp, cv2, numpy, python, vision = _import_mediapipe()
    mp_image = _load_image_from_unicode_path(image, mp, cv2, numpy)
    options = vision.HandLandmarkerOptions(
        base_options=_model_base_options(hand_model, python),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=float(min_hand_detection_confidence),
        min_hand_presence_confidence=float(min_hand_presence_confidence),
        min_tracking_confidence=float(min_tracking_confidence),
    )
    try:
        with vision.HandLandmarker.create_from_options(options) as landmarker:
            result = landmarker.detect(mp_image)
    except Exception as error:
        raise PoseLandmarkerError(
            "MediaPipe 手部姿态识别失败：{}".format(error)
        ) from error

    hand_world_landmarks = getattr(result, "hand_world_landmarks", None) or []
    hand_image_landmarks = getattr(result, "hand_landmarks", None) or []
    handedness = getattr(result, "handedness", None) or []
    world_hand, _image_hand, hand_side, side_score, detected_count = _select_hand(
        hand_world_landmarks,
        hand_image_landmarks,
        handedness,
    )
    points = _raw_hand_landmarks(world_hand)
    wrist_position, wrist_confidence = points["wrist"]

    root_relative = {
        "wrist": _hand_point_item(
            (0.0, 0.0, 0.0),
            wrist_confidence,
            source="wrist",
        )
    }
    for finger_name, segment_names in FINGER_SEGMENT_INDEXES.items():
        for segment, landmark_name in enumerate(segment_names, start=1):
            position, confidence = points[landmark_name]
            root_relative["{}_{}".format(finger_name, segment)] = _hand_point_item(
                tuple(position[axis] - wrist_position[axis] for axis in range(3)),
                min(wrist_confidence, confidence),
                source=landmark_name,
            )

    all_landmarks = {}
    for landmark_name, (position, confidence) in points.items():
        all_landmarks[landmark_name] = _hand_point_item(
            tuple(position[axis] - wrist_position[axis] for axis in range(3)),
            confidence,
            source=landmark_name,
        )

    return {
        "skeleton": "maya_hand_skeleton",
        "version": HAND_POSE_VERSION,
        "coordinate_system": "root_relative_mediapipe_hand_world",
        "unit": "meters",
        "handedness": hand_side,
        "handedness_score": _finite(side_score, "handedness score"),
        "joints": root_relative,
        "parents": dict(HAND_PARENTS),
        "landmarks": all_landmarks,
        "source": {
            "provider": "MediaPipe Hand Landmarker",
            "image": str(image),
            "landmark_count": len(HAND_LANDMARK_NAMES),
            "hands_detected": detected_count,
            "world_coordinates": True,
            "finger_joint_count": len(HAND_JOINT_NAMES) - 1,
        },
    }


__all__ = [
    "HAND_JOINT_NAMES",
    "HAND_PARENTS",
    "HAND_POSE_VERSION",
    "PoseLandmarkerError",
    "infer_single_hand_image",
]
