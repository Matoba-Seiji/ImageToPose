import copy
import importlib
import math
import unittest

m = importlib.import_module("3dsmax_pose_tool.hand_retarget_math")


def frame():
    points = {"wrist": {"position": [0, 0, 0]}}
    for f, chain in m.CHAINS.items():
        x = {"thumb": 2, "index": 1, "middle": 0, "ring": -0.5, "pinky": -1}[f]
        for i, n in enumerate(chain):
            points[n] = {"position": [x, 2+i, 0]}
    return {"coordinate_system": "root_relative_mediapipe_hand_world", "landmarks": points}


class RetargetMathTests(unittest.TestCase):
    def test_isolated_finger_key_policy(self):
        policy = importlib.import_module(
            "3dsmax_pose_tool.biped_hand"
        ).permitted_isolated_finger_keys
        self.assertTrue(policy("Bip001 L Finger0", (), ("0f",), "0f"))
        self.assertFalse(policy("Bip001 L Forearm", (), ("0f",), "0f"))
        self.assertFalse(policy("Bip001 R Finger0", (), ("0f",), "0f"))
        self.assertFalse(policy("Bip001 L Finger0", ("5f",), ("0f",), "0f"))

    def test_arm_sync_key_policy(self):
        module = importlib.import_module("3dsmax_pose_tool.biped_hand")
        policy = module.permitted_hand_sync_keys
        self.assertTrue(policy("Bip001 L UpperArm", (), ("0f",), "0f"))
        self.assertTrue(policy("Bip001 L ForeTwist1", (), ("0f",), "0f"))
        self.assertTrue(policy("Bip001 R UpperArm", (), ("0f",), "0f", "right"))
        self.assertTrue(policy("Bip001 R Finger0", (), ("0f",), "0f", "right"))
        self.assertFalse(policy("Bip001 L UpperArm", (), ("0f",), "0f", "right"))
        self.assertFalse(policy("Bip001 R UpperArm", (), ("0f",), "0f", "left"))
        self.assertTrue(module.is_hand_sync_node("Bip001 RUpArmTwist", "right"))
        self.assertTrue(module.is_hand_finger_node("Bip001 R Finger42", "right"))
        self.assertFalse(module.is_hand_finger_node("Bip001 L Finger42", "right"))

    def test_palm_normal_left(self):
        f = frame()
        # This fixture's palm faces camera -Z; +Z is the back of the hand.
        f["landmarks"]["index_tip"]["position"][2] = -1
        directions = m.source_directions(f)
        self.assertGreater(directions["index"][2][1], 0)
        f["landmarks"]["index_tip"]["position"][2] = 1
        self.assertLess(m.source_directions(f)["index"][2][1], 0)

    def test_left_palm_radial_axis_is_not_swapped(self):
        f = frame()
        f["landmarks"]["thumb_tip"]["position"][0] += 1
        # Target +X follows the fingers, +Y leaves the palm, -Z is radial.
        directions = m.source_directions(f)
        self.assertLess(directions["thumb"][2][2], 0)

    def test_reference_photo_flexion_maps_to_positive_y(self):
        f = frame()
        observed = {
            "index_mcp": (0.067895118, -0.085588830, 0.022607415),
            "index_pip": (0.097405057, -0.097375967, 0.022812372),
            "index_dip": (0.114095036, -0.105319936, 0.011987777),
            "index_tip": (0.111360850, -0.105260905, -0.016649557),
            "middle_mcp": (0.084594501, -0.066553988, 0.026809972),
            "pinky_mcp": (0.091192178, -0.026503976, 0.010262232),
        }
        for name, position in observed.items():
            f["landmarks"][name]["position"] = position
        directions = m.source_directions(f)
        self.assertTrue(all(v[1] > 0 for v in directions["index"]))
        self.assertGreater(directions["index"][2][1], 0.96)
        identity = ((1,0,0), (0,1,0), (0,0,1))
        reference = {finger: [{"name": finger+str(i), "local_axes": identity, "aim": (1,0,0)} for i in range(3)] for finger in m.FINGERS}
        solved = m.solve_hinges(directions, identity, reference)
        for i in range(3):
            self.assertGreater(solved["index"+str(i)]["axes"][0][1], 0)

    def test_all_21_points_required(self):
        for name in frame()["landmarks"]:
            f = frame()
            del f["landmarks"][name]
            with self.assertRaises(ValueError):
                m.source_directions(f)

    def test_bad_data(self):
        for v in ([0, 0, float("nan")], [0, float("inf"), 0], [1, 2]):
            f = frame()
            f["landmarks"]["index_tip"]["position"] = v
            with self.assertRaises(ValueError):
                m.source_directions(f)
        f = frame()
        f["landmarks"]["index_tip"] = f["landmarks"]["index_dip"]
        with self.assertRaises(ValueError):
            m.source_directions(f)

    def test_confidence_does_not_modify_directions(self):
        a = frame()
        b = copy.deepcopy(a)
        b["handedness"] = "left"
        b["handedness_score"] = 0
        for p in b["landmarks"].values():
            p["confidence"] = 0
        self.assertEqual(m.source_directions(a), m.source_directions(b))

    def test_unknown_label_keeps_left_compatibility(self):
        a = frame()
        a['landmarks']['index_tip']['position'][2] = -1
        expected = m.source_directions(a)
        for label in ('left', 'Left', 'unknown', '', None):
            a['handedness'] = label
            self.assertEqual(expected, m.source_directions(a))

    def test_mirrored_right_matches_left_without_mutating_cache(self):
        left = frame()
        left['handedness'] = 'left'
        for finger, chain in m.CHAINS.items():
            for i, name in enumerate(chain):
                left['landmarks'][name]['position'][2] = -0.25*i*i
        right = copy.deepcopy(left)
        right['handedness'] = ' Right '
        for point in right['landmarks'].values():
            x,y,z = point['position']
            # Mirror, then rotate and translate: conversion is independent
            # of the source hand's camera orientation and root location.
            point['position'] = [-y+3, -x-4, z+2]
        saved = copy.deepcopy(right)
        expected = m.source_directions(left)
        actual = m.source_directions(right)
        for finger in m.FINGERS:
            for a,b in zip(expected[finger],actual[finger]):
                for x,y in zip(a,b):
                    self.assertAlmostEqual(x,y)
        self.assertEqual(saved,right)
        self.assertEqual(actual,m.source_directions(right))

    def test_right_conversion_reaches_both_hybrid_solvers(self):
        thumb = importlib.import_module('3dsmax_pose_tool.thumb_rotation_math')
        left = frame()
        left['handedness'] = 'left'
        for chain in m.CHAINS.values():
            for i,name in enumerate(chain):
                left['landmarks'][name]['position'][2] = -0.2*i*i
        right = copy.deepcopy(left)
        right['handedness'] = 'right'
        for point in right['landmarks'].values():
            point['position'][0] *= -1
        basis = ((0,1,0),(-1,0,0),(0,0,1))
        rig = {f:[dict(name=f+str(i),local_axes=((1,0,0),(0,1,0),(0,0,1)),aim=(1,0,0)) for i in range(3)] for f in m.FINGERS}
        results = []
        for source in (left,right):
            directions = m.source_directions(source)
            solved = m.solve_hinges(directions,basis,rig,fingers=m.FINGERS[1:])
            solved.update(thumb.solve(directions,basis,basis,rig,fingers=('thumb',)))
            results.append(solved)
        self.assertEqual(results[0],results[1])
        self.assertGreater(m.source_directions(right)['index'][2][1],0)

    def test_rigid_invariance(self):
        a = frame()
        b = copy.deepcopy(a)
        for p in b["landmarks"].values():
            x,y,z = p["position"]
            p["position"] = [-y+10, x-4, z+2]
        self.assertEqual(m.source_directions(a), m.source_directions(b))

    def test_solver_preserves_orthonormality_and_is_repeatable(self):
        identity = ((1,0,0), (0,1,0), (0,0,1))
        reference = {f: [{"name": f+str(i), "local_axes": identity, "aim": (1,0,0)} for i in range(3)] for f in m.FINGERS}
        directions = m.source_directions(frame())
        directions["index"] = ((0,1,0), (-1,0,0), (0,-1,0))
        a = m.solve_hinges(directions, identity, reference)
        self.assertEqual(a, m.solve_hinges(directions, identity, reference))
        for item in a.values():
            for x,y in zip(item["axes"][0], item["direction"]):
                self.assertAlmostEqual(x,y)
            self.assertAlmostEqual(m.dot(m.cross(*item["axes"][:2]), item["axes"][2]), 1)

    def test_distal_joints_are_single_axis(self):
        identity = ((1,0,0), (0,1,0), (0,0,1))
        rig = {f: [{"name": f+str(i)} for i in range(3)] for f in m.FINGERS}
        directions = m.source_directions(frame())
        directions["index"] = tuple(m.unit(v) for v in ((1,1,0), (0,1,0.3), (-1,1,-0.2)))
        solved = m.solve_hinges(directions, identity, rig)
        for finger in m.FINGERS[1:]:
            for i in (1,2):
                parent = solved[finger+str(i-1)]["axes"]
                child = solved[finger+str(i)]["axes"]
                self.assertAlmostEqual(m.dot(child[0], parent[2]), 0)
                self.assertAlmostEqual(m.dot(child[1], parent[2]), 0)
                self.assertAlmostEqual(m.dot(child[2], parent[2]), 1)
        self.assertGreater(abs(solved["index1"]["off_plane_degrees"]), 10)
        self.assertAlmostEqual(solved["index1"]["hinge_angle_degrees"], 45)

    def test_root_spread_and_flexion_without_roll(self):
        directions = m.source_directions(frame())
        for f in m.FINGERS:
            directions[f] = ((0.5, 0.5, 0.5), (0,1,0), (-1,0,0))
        for f in m.FINGERS[1:]:
            self.assertAlmostEqual(m.finger_hinge(directions[f])[1], 0)

    def test_hinge_solver_excludes_thumb(self):
        identity = ((1,0,0), (0,1,0), (0,0,1))
        directions = m.source_directions(frame())
        rig = {f: [{"name": f+str(i)} for i in range(3)] for f in m.FINGERS}
        self.assertEqual(len(m.solve_hinges(directions, identity, rig)), 12)
        with self.assertRaises(ValueError):
            m.solve_hinges(directions, identity, rig, fingers=("thumb",))

    def test_pose_independence_and_hand_rotation(self):
        identity = ((1,0,0), (0,1,0), (0,0,1))
        rotated = ((0,1,0), (-1,0,0), (0,0,1))
        rig = {f: [{"name": f+str(i)} for i in range(3)] for f in m.FINGERS}
        directions = m.source_directions(frame())
        a = m.solve_hinges(directions, identity, rig)
        b = m.solve_hinges(directions, rotated, copy.deepcopy(rig))
        for name in a:
            for old,new in zip(a[name]["axes"],b[name]["axes"]):
                for x,y in zip(m.world(old,rotated),new):
                    self.assertAlmostEqual(x,y)

    def test_degenerate_hinge_projection_rejected(self):
        with self.assertRaises(ValueError):
            m.projected((0,0,1), (0,0,1))


if __name__ == "__main__":
    unittest.main()
