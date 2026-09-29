import copy
import importlib
import json
import math
import unittest
from unittest.mock import MagicMock, patch

client = importlib.import_module("3dsmax_pose_tool.pose_client")
SPINE_JOINT_PARAMETERS = (
    ("spine_1", 0.25),
    ("spine_2", 0.55),
    ("spine_3", 0.80),
)


def make_frame(shoulder_center, leg_positions=None, confidences=None):
    leg_positions = leg_positions or {}
    confidences = confidences or {}
    joints = {
        name: {"position": [0.0, 0.0, 0.0], "visibility": 1.0}
        for name in client.BODY_JOINTS
    }
    joints["pelvis"]["position"] = [0.0, 0.0, 0.0]
    joints["left_shoulder"]["position"] = [
        shoulder_center[0] - 0.1,
        shoulder_center[1],
        shoulder_center[2],
    ]
    joints["right_shoulder"]["position"] = [
        shoulder_center[0] + 0.1,
        shoulder_center[1],
        shoulder_center[2],
    ]

    for side in ("left", "right"):
        positions = leg_positions.get(
            side,
            {
                "hip": (0.0, 0.0, 0.0),
                "knee": (0.0, 0.5, 0.0),
                "ankle": (0.0, 1.0, 0.0),
            },
        )
        for part in ("hip", "knee", "ankle"):
            name = side + "_" + part
            joints[name]["position"] = list(positions[part])

    for name, visibility in confidences.items():
        joints[name]["visibility"] = visibility

    for name, amount in SPINE_JOINT_PARAMETERS:
        joints[name]["position"] = [
            component * amount for component in shoulder_center
        ]

    return {
        "skeleton": "maya_pose_skeleton",
        "coordinate_system": "root_relative_mediapipe_world",
        "include_fingers": False,
        "joints": joints,
    }


def offset_from_chord(frame, joint_name, shoulder_center, amount):
    point = frame["joints"][joint_name]["position"]
    chord_point = [component * amount for component in shoulder_center]
    return [point[i] - chord_point[i] for i in range(3)]


def distance(vector):
    return math.sqrt(sum(component * component for component in vector))


class MaxPoseSpineCurveTests(unittest.TestCase):
    def assert_linear(self, frame, shoulder_center):
        for name, amount in SPINE_JOINT_PARAMETERS:
            point = frame["joints"][name]["position"]
            for axis in range(3):
                self.assertAlmostEqual(
                    point[axis], shoulder_center[axis] * amount, places=7
                )

    def test_curves_legacy_linear_body_frames_and_is_idempotent(self):
        shoulder_center = (1.0, 0.0, 0.0)
        frame = make_frame(shoulder_center)

        client._apply_spine_curve_if_linear(frame)

        offset = offset_from_chord(frame, "spine_2", shoulder_center, 0.55)
        self.assertAlmostEqual(distance(offset), 0.08 * 4.0 * 0.55 * 0.45, places=6)
        first_result = copy.deepcopy(frame)
        client._apply_spine_curve_if_linear(frame)
        self.assertEqual(frame, first_result)

    def test_body_request_curves_linear_service_response(self):
        shoulder_center = (1.0, 0.0, 0.0)
        service_frame = make_frame(shoulder_center)
        response = MagicMock()
        response.read.return_value = json.dumps(service_frame).encode("utf-8")
        response.__enter__.return_value = response

        with patch.object(
            client,
            "_multipart_body",
            return_value=(b"image", "multipart/form-data; boundary=test"),
        ), patch.object(
            client.urllib_request,
            "urlopen",
            return_value=response,
        ):
            result = client._request_frame(
                "unused.png",
                endpoint="http://127.0.0.1:18765/v1/pose",
            )

        self.assertLess(
            result["joints"]["spine_2"]["position"][1],
            service_frame["joints"]["spine_2"]["position"][1],
        )

    def test_small_angles_stay_straight_and_medium_angles_are_partial(self):
        for angle in (0.0, 12.0):
            radians = math.radians(angle)
            shoulder_center = (math.sin(radians), -math.cos(radians), 0.0)
            frame = make_frame(shoulder_center)
            client._apply_spine_curve_if_linear(frame)
            with self.subTest(angle=angle):
                self.assert_linear(frame, shoulder_center)

        radians = math.radians(30.0)
        shoulder_center = (math.sin(radians), -math.cos(radians), 0.0)
        frame = make_frame(shoulder_center)
        client._apply_spine_curve_if_linear(frame)
        middle_offset = distance(
            offset_from_chord(frame, "spine_2", shoulder_center, 0.55)
        )
        self.assertGreater(middle_offset, 0.0)
        self.assertLess(middle_offset, 0.08 * 4.0 * 0.55 * 0.45)

    def test_most_extended_leg_wins_over_bent_leg(self):
        shoulder_center = (1.0, 0.0, 0.0)
        legs = {
            "left": {
                "hip": (0.0, 0.0, 0.0),
                "knee": (0.0, 0.5, 0.0),
                "ankle": (-1.0, 0.0, 0.0),
            },
            "right": {
                "hip": (0.0, 0.0, 0.0),
                "knee": (0.0, 0.5, 0.0),
                "ankle": (0.0, 1.0, 0.0),
            },
        }
        frame = make_frame(shoulder_center, legs)

        client._apply_spine_curve_if_linear(frame)

        self.assertLess(frame["joints"]["spine_2"]["position"][1], 0.0)

    def test_low_confidence_falls_back_and_curved_server_frame_is_preserved(self):
        shoulder_center = (1.0, 0.0, 0.0)
        frame = make_frame(
            shoulder_center,
            confidences={"left_ankle": 0.49, "right_ankle": 0.09},
        )
        client._apply_spine_curve_if_linear(frame)
        self.assert_linear(frame, shoulder_center)

        already_curved = make_frame(shoulder_center)
        already_curved["joints"]["spine_1"]["position"][1] = -0.01
        already_curved["joints"]["spine_2"]["position"][1] = -0.02
        already_curved["joints"]["spine_3"]["position"][1] = -0.01
        original = copy.deepcopy(already_curved)
        client._apply_spine_curve_if_linear(already_curved)
        self.assertEqual(already_curved, original)

    def test_ignores_hand_frames_and_other_coordinate_systems(self):
        hand_frame = {
            "skeleton": "maya_hand_skeleton",
            "coordinate_system": "root_relative_mediapipe_hand_world",
            "joints": {},
        }
        body_frame = make_frame((1.0, 0.0, 0.0))
        body_frame["coordinate_system"] = "root_relative_y_up"
        original_body = copy.deepcopy(body_frame)

        self.assertIs(client._apply_spine_curve_if_linear(hand_frame), hand_frame)
        self.assertIs(client._apply_spine_curve_if_linear(body_frame), body_frame)
        self.assertEqual(body_frame, original_body)


if __name__ == "__main__":
    unittest.main()
