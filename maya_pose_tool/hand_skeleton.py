"""Create a standalone Maya hand joint hierarchy from MediaPipe data."""

from __future__ import annotations

import math


HAND_SKELETON_DATA_VERSION = "maya-hand-skeleton-v1"
GENERATED_MARKER_ATTRIBUTE = "mayaPoseToolSkeleton"
GENERATED_MARKER_VALUE = "maya_mediapipe_hand_tool"
HAND_JOINTS = (
    "wrist",
    "thumb_1", "thumb_2", "thumb_3", "thumb_tip",
    "index_1", "index_2", "index_3", "index_tip",
    "middle_1", "middle_2", "middle_3", "middle_tip",
    "ring_1", "ring_2", "ring_3", "ring_tip",
    "pinky_1", "pinky_2", "pinky_3", "pinky_tip",
)
HAND_PARENTS = {
    "{}_1".format(finger): "wrist"
    for finger in ("thumb", "index", "middle", "ring", "pinky")
}
for _finger in ("thumb", "index", "middle", "ring", "pinky"):
    HAND_PARENTS["{}_2".format(_finger)] = "{}_1".format(_finger)
    HAND_PARENTS["{}_3".format(_finger)] = "{}_2".format(_finger)
    HAND_PARENTS["{}_tip".format(_finger)] = "{}_3".format(_finger)


class HandPoseDataError(ValueError):
    """Raised when supplied hand data cannot describe a hand skeleton."""


def _finite_number(value, label):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise HandPoseDataError("{} must contain numbers".format(label)) from error
    if not math.isfinite(number):
        raise HandPoseDataError("{} must contain finite numbers".format(label))
    return number


def source_to_maya(position, coordinate_system):
    if not isinstance(position, (list, tuple)) or len(position) != 3:
        raise HandPoseDataError("hand joint position must be a three-number array")
    x, y, z = (_finite_number(value, "hand joint position") for value in position)
    if coordinate_system in {"root_relative_y_up", "root_relative_hand_y_up"}:
        return x, y, z
    if coordinate_system == "root_relative_mediapipe_hand_world":
        return x, -y, -z
    raise HandPoseDataError(
        "unsupported hand coordinate system: {}".format(coordinate_system)
    )


def hand_joint_positions(frame):
    if not isinstance(frame, dict):
        raise HandPoseDataError("hand pose response must be an object")
    if frame.get("skeleton") != "maya_hand_skeleton":
        raise HandPoseDataError("pose frame is not a standalone Maya hand frame")
    joints = frame.get("joints")
    if not isinstance(joints, dict):
        raise HandPoseDataError("hand pose response has no joints object")
    coordinate_system = frame.get("coordinate_system")
    if not isinstance(coordinate_system, str):
        raise HandPoseDataError("hand pose response has no coordinate system")

    positions = {}
    missing = []
    landmarks = frame.get("landmarks")
    if not isinstance(landmarks, dict):
        raise HandPoseDataError("hand pose response has no fingertip landmarks")
    for name in HAND_JOINTS:
        # Tips live in the full landmark data, not the legacy 16-joint map.
        item = landmarks.get(name) if name.endswith("_tip") else joints.get(name)
        if not isinstance(item, dict) or "position" not in item:
            missing.append(name)
            continue
        positions[name] = source_to_maya(item["position"], coordinate_system)
    if missing:
        raise HandPoseDataError(
            "hand pose response is missing joints or fingertip landmarks: {}".format(", ".join(missing))
        )
    for name in HAND_JOINTS:
        if name.endswith("_tip"):
            parent = positions[HAND_PARENTS[name]]
            if math.dist(positions[name], parent) <= 1e-10:
                raise HandPoseDataError("zero-length fingertip bone: " + name)
    return positions


def _valid_node(cmds, node):
    return bool(node and cmds.objExists(node))


def _existing_nodes(cmds, prefix):
    existing = []

    def add(node):
        if _valid_node(cmds, node) and node not in existing:
            existing.append(node)

    for name in HAND_JOINTS:
        add(prefix + name)
    for node in cmds.ls(type="joint", long=True) or ():
        leaf = node.rsplit("|", 1)[-1]
        if not leaf.startswith(prefix):
            continue
        try:
            marked = cmds.attributeQuery(
                GENERATED_MARKER_ATTRIBUTE,
                node=node,
                exists=True,
            )
            if marked and cmds.getAttr(
                node + "." + GENERATED_MARKER_ATTRIBUTE
            ) == GENERATED_MARKER_VALUE:
                add(node)
        except Exception:
            continue
    return existing


def _delete_previous(cmds, prefix):
    for node in _existing_nodes(cmds, prefix):
        if _valid_node(cmds, node):
            try:
                cmds.delete(node)
            except Exception:
                pass


def _mark_root(cmds, root):
    if not hasattr(cmds, "addAttr") or not hasattr(cmds, "setAttr"):
        return
    try:
        if not cmds.attributeQuery(
            GENERATED_MARKER_ATTRIBUTE,
            node=root,
            exists=True,
        ):
            cmds.addAttr(root, longName=GENERATED_MARKER_ATTRIBUTE, dataType="string")
        cmds.setAttr(
            root + "." + GENERATED_MARKER_ATTRIBUTE,
            GENERATED_MARKER_VALUE,
            type="string",
        )
    except Exception:
        pass


def create_hand_skeleton_from_pose(
    frame,
    cmds,
    scale=100.0,
    name_prefix="MediaPipeHandPoseSkeleton_",
    orient_joints=True,
    replace_previous=True,
):
    """Create or replace only the standalone MediaPipe hand hierarchy."""

    scale = _finite_number(scale, "scale")
    if scale <= 0.0:
        raise HandPoseDataError("scale must be greater than zero")
    if not isinstance(name_prefix, str) or not name_prefix:
        raise HandPoseDataError("name prefix must be a non-empty string")

    positions = hand_joint_positions(frame)
    if replace_previous:
        _delete_previous(cmds, name_prefix)
    created = {}
    try:
        for name in HAND_JOINTS:
            created[name] = cmds.createNode("joint", name=name_prefix + name)
            cmds.xform(
                created[name],
                worldSpace=True,
                translation=tuple(component * scale for component in positions[name]),
            )
        for name in HAND_JOINTS:
            parent_name = HAND_PARENTS.get(name)
            if parent_name is not None:
                cmds.parent(created[name], created[parent_name], absolute=True)
    except Exception:
        for node in reversed(tuple(created.values())):
            try:
                if _valid_node(cmds, node):
                    cmds.delete(node)
            except Exception:
                pass
        raise

    _mark_root(cmds, created["wrist"])
    orientation_applied = False
    if orient_joints and hasattr(cmds, "joint"):
        try:
            cmds.joint(
                created["wrist"],
                edit=True,
                orientJoint="xyz",
                secondaryAxisOrient="yup",
                children=True,
                zeroScaleOrient=True,
            )
            orientation_applied = True
        except Exception:
            orientation_applied = False

    return {
        "success": True,
        "root": created["wrist"],
        "joint_count": len(created),
        "joints": created,
        "handedness": frame.get("handedness", "unknown"),
        "name_prefix": name_prefix,
        "scale": scale,
        "coordinate_system": frame.get("coordinate_system"),
        "orientation_applied": orientation_applied,
        "scene_changes": (
            "replaced_previous_generated_hand_skeleton"
            if replace_previous
            else "created_hand_skeleton_only"
        ),
    }


__all__ = [
    "GENERATED_MARKER_VALUE",
    "HAND_JOINTS",
    "HAND_PARENTS",
    "HAND_SKELETON_DATA_VERSION",
    "HandPoseDataError",
    "create_hand_skeleton_from_pose",
    "hand_joint_positions",
    "source_to_maya",
]
