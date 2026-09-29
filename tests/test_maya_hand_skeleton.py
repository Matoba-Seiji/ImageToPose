import copy
import unittest

from maya_pose_tool import hand_skeleton as hand


FINGERS = ("thumb", "index", "middle", "ring", "pinky")
PREFIX = "MediaPipeHandPoseSkeleton_"


def frame():
    joints = {"wrist": {"position": [0, 0, 0]}}
    landmarks = {}
    for i, finger in enumerate(FINGERS):
        for segment in (1, 2, 3):
            joints[f"{finger}_{segment}"] = {
                "position": [0.01 * i, 0.02 * segment, 0.005 * i]
            }
        landmarks[finger + "_tip"] = {
            "position": [0.01 * i + 0.008, 0.055, 0.005 * i + 0.012]
        }
    return {"skeleton": "maya_hand_skeleton", "handedness": "right",
            "coordinate_system": "root_relative_mediapipe_hand_world",
            "joints": joints, "landmarks": landmarks}


class Commands:
    def __init__(self):
        self.nodes = {}
        self.oriented = []

    def createNode(self, kind, name):
        self.assert_new(name)
        self.nodes[name] = {"parent": None, "position": None}
        return name

    def assert_new(self, name):
        if name in self.nodes:
            raise AssertionError("Previous node was not removed: " + name)

    def xform(self, name, worldSpace, translation):
        self.nodes[name]["position"] = translation

    def parent(self, name, parent, absolute):
        self.nodes[name]["parent"] = parent

    def objExists(self, name):
        return name in self.nodes

    def ls(self, **kwargs):
        return list(self.nodes)

    def attributeQuery(self, *args, **kwargs):
        return False

    def delete(self, name):
        for child in list(self.nodes):
            if child in self.nodes and self.nodes[child]["parent"] == name:
                self.delete(child)
        self.nodes.pop(name, None)

    def joint(self, name, **kwargs):
        self.oriented.append((name, kwargs))


class MayaHandSkeletonTests(unittest.TestCase):
    def test_creates_21_joints_with_measured_tips_and_preserves_input(self):
        for coordinate_system in ("root_relative_mediapipe_hand_world", "root_relative_y_up", "root_relative_hand_y_up"):
            for scale in (1.0, 100.0):
                with self.subTest(coordinates=coordinate_system, scale=scale):
                    cmds = Commands()
                    source = frame()
                    source["coordinate_system"] = coordinate_system
                    original = copy.deepcopy(source)
                    result = hand.create_hand_skeleton_from_pose(source, cmds, scale=scale)
                    self.assertEqual(result["joint_count"], 21)
                    self.assertTrue(result["orientation_applied"])
                    self.assertEqual(len(cmds.nodes), 21)
                    for child, parent in hand.HAND_PARENTS.items():
                        self.assertEqual(cmds.nodes[PREFIX + child]["parent"], PREFIX + parent)
                    for finger in FINGERS:
                        tip = finger + "_tip"
                        x, y, z = source["landmarks"][tip]["position"]
                        sign = -1 if coordinate_system == "root_relative_mediapipe_hand_world" else 1
                        self.assertEqual(cmds.nodes[PREFIX + tip]["position"], (x*scale, y*scale*sign, z*scale*sign))
                    self.assertEqual(source, original)

    def test_replaces_old_and_new_hand_without_deleting_body(self):
        cmds = Commands()
        unrelated = ("MediaPipePoseSkeleton_left_wrist", "CharacterHand")
        for name in unrelated:
            cmds.createNode("joint", name)
        old_names = [name for name in hand.HAND_JOINTS if not name.endswith("_tip")]
        for name in old_names:
            cmds.createNode("joint", PREFIX + name)
        for name in old_names:
            if name in hand.HAND_PARENTS:
                cmds.parent(PREFIX + name, PREFIX + hand.HAND_PARENTS[name], True)
        for _ in range(2):
            result = hand.create_hand_skeleton_from_pose(frame(), cmds)
            self.assertEqual(result["joint_count"], 21)
            self.assertEqual(set(cmds.nodes), set(unrelated) | {PREFIX + name for name in hand.HAND_JOINTS})

    def test_invalid_tip_data_preserves_existing_scene(self):
        invalid_frames = []
        for invalid in (None, [], [1, 2], [0, 0, float("nan")], [0, 0, float("inf")], ["bad", 0, 0], [0.02, 0.06, 0.01]):
            source = frame()
            source["landmarks"]["middle_tip"]["position"] = invalid
            invalid_frames.append(source)
        source = frame()
        del source["landmarks"]["middle_tip"]
        invalid_frames.append(source)
        source = frame()
        del source["landmarks"]
        invalid_frames.append(source)
        for source in invalid_frames:
            with self.subTest(source=source):
                cmds = Commands()
                hand.create_hand_skeleton_from_pose(frame(), cmds)
                before = copy.deepcopy(cmds.nodes)
                with self.assertRaises(hand.HandPoseDataError):
                    hand.create_hand_skeleton_from_pose(source, cmds)
                self.assertEqual(cmds.nodes, before)


if __name__ == "__main__":
    unittest.main()
