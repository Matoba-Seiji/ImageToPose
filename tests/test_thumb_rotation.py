import copy
import importlib
import unittest

m = importlib.import_module('3dsmax_pose_tool.hand_retarget_math')
legacy = importlib.import_module('3dsmax_pose_tool.thumb_rotation_math')
IDENTITY = ((1, 0, 0), (0, 1, 0), (0, 0, 1))


def reference():
    return {f: [{'name': f+str(i), 'local_axes': IDENTITY, 'aim': (1,0,0)} for i in range(3)] for f in m.FINGERS}


class ThumbRotationTests(unittest.TestCase):
    def test_shortest_swing_degenerate_and_general(self):
        for target in ((1,0,0), (-1,0,0), (0,1,0), (1,2,3)):
            result = legacy.swing((1,0,0), (1,0,0), target, (0,1,0))
            for a,b in zip(result,m.unit(target)):
                self.assertAlmostEqual(a,b)

    def test_legacy_matches_full_3d_direction_and_keeps_reference(self):
        directions = {f: tuple(m.unit(v) for v in ((1,1,0), (0,1,0.4), (-1,1,-0.3))) for f in m.FINGERS}
        ref = reference()
        saved = copy.deepcopy(ref)
        solved = legacy.solve(directions,IDENTITY,IDENTITY,ref)
        self.assertEqual(ref,saved)
        self.assertEqual(solved,legacy.solve(directions,IDENTITY,IDENTITY,ref))
        constrained = m.solve_hinges(directions,IDENTITY,ref)
        for f in m.FINGERS:
            for i in range(3):
                item = solved[f+str(i)]
                for a,b in zip(item['axes'][0],directions[f][i]):
                    self.assertAlmostEqual(a,b)
                self.assertAlmostEqual(m.dot(m.cross(*item['axes'][:2]),item['axes'][2]),1)
        self.assertGreater(abs(solved['index1']['axes'][0][2]-constrained['index1']['axes'][0][2]),0.3)

    def test_arbitrary_reference_aim(self):
        ref = reference()
        for chain in ref.values():
            for entry in chain:
                entry['aim'] = (0,1,0)
        directions = {f: ((1,0,0),(0,0,1),(0,-1,0)) for f in m.FINGERS}
        solved = legacy.solve(directions,IDENTITY,IDENTITY,ref)
        for f in m.FINGERS:
            for i in range(3):
                for a,b in zip(solved[f+str(i)]['axes'][1],directions[f][i]):
                    self.assertAlmostEqual(a,b)

    def test_target_hand_rotation_equivariance(self):
        basis = ((0,1,0),(-1,0,0),(0,0,1))
        directions = {f: ((1,0,0),(0,1,0),(0,0,1)) for f in m.FINGERS}
        a = legacy.solve(directions,IDENTITY,IDENTITY,reference())
        b = legacy.solve(directions,basis,basis,reference())
        for name in a:
            for row,other in zip(a[name]['axes'],b[name]['axes']):
                for x,y in zip(m.world(row,basis),other):
                    self.assertAlmostEqual(x,y)


if __name__ == '__main__':
    unittest.main()
