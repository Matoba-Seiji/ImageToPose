import math
import unittest
from types import SimpleNamespace

from maya_pose_tool import mediapipe_pose as pose


def make_landmarks(shoulder_center, leg_positions=None, confidences=None):
    leg_positions = leg_positions or {}
    confidences = confidences or {}
    points = {
        name: ((0.0, 0.0, 0.0), 1.0)
        for name in pose.LANDMARK_NAMES
    }
    points["left_hip"] = ((0.0, 0.0, 0.0), 1.0)
    points["right_hip"] = ((0.0, 0.0, 0.0), 1.0)
    points["left_shoulder"] = (
        (shoulder_center[0] - 0.1, shoulder_center[1], shoulder_center[2]),
        1.0,
    )
    points["right_shoulder"] = (
        (shoulder_center[0] + 0.1, shoulder_center[1], shoulder_center[2]),
        1.0,
    )

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
            points[name] = (
                positions[part],
                confidences.get(name, 1.0),
            )

    for name, confidence in confidences.items():
        if name in points:
            position, _old_confidence = points[name]
            points[name] = (position, confidence)

    return [
        SimpleNamespace(
            x=position[0],
            y=position[1],
            z=position[2],
            visibility=confidence,
            presence=1.0,
        )
        for name in pose.LANDMARK_NAMES
        for position, confidence in (points[name],)
    ]


def build_frame(shoulder_center, leg_positions=None, confidences=None):
    return pose.build_pose_frame_from_landmarks(
        make_landmarks(shoulder_center, leg_positions, confidences)
    )


def distance(vector):
    return math.sqrt(sum(component * component for component in vector))


def offset_from_chord(frame, name, shoulder_center, amount):
    point = frame["joints"][name]["position"]
    chord_point = [component * amount for component in shoulder_center]
    return [point[i] - chord_point[i] for i in range(3)]


class MayaPoseSpineCurveTests(unittest.TestCase):
    def assert_linear_spine(self, frame, shoulder_center):
        for name, amount in (
            ("spine_1", 0.25),
            ("spine_2", 0.55),
            ("spine_3", 0.80),
        ):
            point = frame["joints"][name]["position"]
            for axis in range(3):
                self.assertAlmostEqual(
                    point[axis], shoulder_center[axis] * amount, places=7
                )

    def test_upright_and_small_angles_remain_linear(self):
        for angle in (0.0, 10.0, 12.0):
            radians = math.radians(angle)
            shoulder_center = (
                math.sin(radians),
                -math.cos(radians),
                0.0,
            )
            with self.subTest(angle=angle):
                result = build_frame(shoulder_center)
                self.assert_linear_spine(result, shoulder_center)

    def test_curve_grows_smoothly_and_saturates_at_maximum(self):
        frames = {}
        for angle in (30.0, 45.0, 60.0):
            radians = math.radians(angle)
            shoulder_center = (
                math.sin(radians),
                -math.cos(radians),
                0.0,
            )
            frames[angle] = (
                build_frame(shoulder_center),
                shoulder_center,
            )

        mid_offsets = {
            angle: distance(offset_from_chord(frame, "spine_2", shoulder, 0.55))
            for angle, (frame, shoulder) in frames.items()
        }
        expected_at_30 = 0.08 * (18.0 / 33.0) * 4.0 * 0.55 * 0.45
        expected_at_max = 0.08 * 4.0 * 0.55 * 0.45
        self.assertAlmostEqual(mid_offsets[30.0], expected_at_30, places=6)
        self.assertAlmostEqual(mid_offsets[45.0], expected_at_max, places=6)
        self.assertAlmostEqual(mid_offsets[60.0], expected_at_max, places=6)
        self.assertLess(mid_offsets[30.0], mid_offsets[45.0])

    def test_prefers_the_most_extended_reliable_leg(self):
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

        result = build_frame(shoulder_center, legs)

        # The extended right leg supplies a vertical support axis. The bent
        # left leg's axis is parallel to the torso and would produce no curve.
        self.assertLess(result["joints"]["spine_2"]["position"][1], 0.0)

    def test_confidence_breaks_equal_knee_extension_tie(self):
        shoulder_center = (0.0, -1.0, 0.0)
        legs = {
            "left": {
                "hip": (0.0, 0.0, 0.0),
                "knee": (0.0, 0.5, 0.0),
                "ankle": (0.0, 1.0, 0.0),
            },
            "right": {
                "hip": (0.0, 0.0, 0.0),
                "knee": (0.5, 0.0, 0.0),
                "ankle": (1.0, 0.0, 0.0),
            },
        }
        confidences = {
            "left_hip": 0.8,
            "left_knee": 0.8,
            "left_ankle": 0.8,
            "right_hip": 0.9,
            "right_knee": 0.9,
            "right_ankle": 0.9,
        }

        result = build_frame(shoulder_center, legs, confidences)

        # Both knees are straight, so the higher-confidence right leg wins.
        self.assertLess(result["joints"]["spine_2"]["position"][0], 0.0)

    def test_unreliable_legs_and_degenerate_projection_fall_back_to_line(self):
        shoulder_center = (1.0, 0.0, 0.0)
        low_confidence = build_frame(
            shoulder_center,
            confidences={"left_ankle": 0.49, "right_ankle": 0.09},
        )
        self.assert_linear_spine(low_confidence, shoulder_center)

        low_torso_confidence = build_frame(
            shoulder_center,
            confidences={"left_shoulder": 0.49},
        )
        self.assert_linear_spine(low_torso_confidence, shoulder_center)

        opposite_axis_legs = {
            side: {
                "hip": (0.0, 0.0, 0.0),
                "knee": (0.5, 0.0, 0.0),
                "ankle": (1.0, 0.0, 0.0),
            }
            for side in ("left", "right")
        }
        degenerate_projection = build_frame(
            shoulder_center,
            opposite_axis_legs,
        )
        self.assert_linear_spine(degenerate_projection, shoulder_center)

    def test_curve_stays_finite_and_keeps_existing_joint_count(self):
        result = build_frame((1.0, 0.0, 0.0))

        self.assertEqual(len(result["joints"]), 22)
        for joint in result["joints"].values():
            self.assertTrue(
                all(math.isfinite(value) for value in joint["position"])
            )
        positions = [
            result["joints"][name]["position"]
            for name in ("spine_1", "spine_2", "spine_3")
        ]
        self.assertNotEqual(positions[0][1], positions[1][1])


if __name__ == "__main__":
    unittest.main()
