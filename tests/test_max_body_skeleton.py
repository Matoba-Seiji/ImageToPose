import importlib
import unittest


client = importlib.import_module("3dsmax_pose_tool.pose_client")


class _Point:
    def __init__(self, x, y, z):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)


class _Node:
    def __init__(self, start, end):
        self.name = ""
        self.start = start
        self.end = end
        self.parent = None
        self.width = None
        self.height = None
        self.marker = ""
        self.valid = True


class _BoneSystem:
    def __init__(self, runtime):
        self.runtime = runtime

    def createBone(self, start, end, _up):
        node = _Node(start, end)
        self.runtime.objects.append(node)
        return node


class _Units:
    @staticmethod
    def decodeValue(_value):
        return 100.0


class _Runtime:
    def __init__(self):
        self.objects = []
        self.units = _Units()
        self.BoneSys = _BoneSystem(self)

    @staticmethod
    def Point3(x, y, z):
        return _Point(x, y, z)

    @staticmethod
    def isValidNode(node):
        return bool(node and node.valid)

    def delete(self, node):
        node.valid = False
        if node in self.objects:
            self.objects.remove(node)

    @staticmethod
    def setUserProp(node, key, value):
        if key == client.MARKER_KEY:
            node.marker = value

    @staticmethod
    def getUserProp(node, key):
        return node.marker if key == client.MARKER_KEY else ""


def _frame():
    return {
        "skeleton": "maya_pose_skeleton",
        "coordinate_system": "root_relative_mediapipe_world",
        "joints": {name: {"position": [i*0.01, i*0.02, 0.0]}
                   for i, name in enumerate(client.BODY_JOINTS)},
    }


class MaxBodySkeletonTests(unittest.TestCase):
    def test_body_leaf_extrapolation_is_unchanged(self):
        runtime = _Runtime()
        frame = _frame()
        result = client.create_max_skeleton(frame, runtime=runtime)
        positions, scale = client._position_map(frame, runtime)
        for name in ("head", "left_wrist", "right_foot"):
            expected = client._leaf_end(name, positions, client.BODY_PARENTS[name], scale)
            end = result["joints"][name].end
            self.assertEqual((end.x, end.y, end.z), expected)

    def test_creates_body_hierarchy_and_preserves_unrelated_nodes(self):
        runtime = _Runtime()
        frame = _frame()
        first = client.create_max_skeleton(frame, runtime=runtime)
        first_nodes = set(runtime.objects)
        self.assertEqual(first["joint_count"], 22)
        self.assertIs(first["root"], first["joints"]["pelvis"])
        self.assertEqual(first["root"].marker, client.BODY_MARKER_VALUE)
        for child, parent in client.BODY_PARENTS.items():
            if child in first["joints"]:
                self.assertIs(first["joints"][child].parent, first["joints"][parent])
        unrelated = _Node(None, None)
        unrelated.name = "MediaPipeHandPoseSkeleton_wrist"
        runtime.objects.append(unrelated)
        second = client.create_max_skeleton(frame, runtime=runtime)
        self.assertEqual(len(runtime.objects), 23)
        self.assertTrue(all(not node.valid for node in first_nodes))
        self.assertTrue(all(node.valid for node in runtime.objects))
        self.assertIn(unrelated, runtime.objects)
        self.assertEqual(second["name_prefix"], client.BODY_NAME_PREFIX)


if __name__ == "__main__":
    unittest.main()
