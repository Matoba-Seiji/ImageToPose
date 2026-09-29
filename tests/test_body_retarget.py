import copy
import importlib
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from body_test_data import fixture, add_fingers

b = importlib.import_module('3dsmax_pose_tool.body_retarget_math')
a = importlib.import_module('3dsmax_pose_tool.biped_body')
m = b.m


class BodyMathTests(unittest.TestCase):
    def assertRows(self, actual, expected, tolerance=1e-6):
        self.assertLess(max(abs(x-y) for row, old in zip(actual, expected) for x, y in zip(row, old)), tolerance)

    def test_pair_compensates_non_x_foot_axis(self):
        rows = b.align_pair((9, 13, .4), (0, 0, 1), (-.3, -.8, -.2), (1, 0, 0))
        self.assertLess(b.angle(m.world((9, 13, .4), rows), (-.3, -.8, -.2)), 1e-5)
        self.assertAlmostEqual(m.dot(m.cross(rows[0], rows[1]), rows[2]), 1)

    def test_shortest_swing_handles_antiparallel(self):
        rows = b.aim_swing(b.IDENTITY, (1, 0, 0), (-1, 0, 0))
        self.assertLess(b.angle(rows[0], (-1, 0, 0)), 1e-5)

    def test_body_directions_and_preserved_local_rotations(self):
        points, rig = fixture()
        saved = copy.deepcopy((points, rig))
        result = b.solve(points, rig)
        for name, desired in result['directions'].items():
            self.assertLess(b.angle(m.world(rig[name]['aim'], result['rotations'][name]), desired), 1e-5, name)
        for name in result['preserved']:
            self.assertRows(b.relative(result['rotations'][name], result['rotations'][rig[name]['parent']]), rig[name]['local_axes'])
        self.assertEqual((points, rig), saved)

    def test_hip_width_sets_pelvic_roll(self):
        points, rig = fixture()
        result = b.solve(points, rig)
        local = m.local(m.sub(rig['L Thigh']['position'], rig['R Thigh']['position']), rig['Pelvis']['axes'])
        self.assertLess(b.angle(m.world(local, result['rotations']['Pelvis']), m.sub(points['left_hip'], points['right_hip'])), 1e-5)

    def test_both_elbows_and_knees_use_negative_flexion(self):
        points, rig = fixture()
        rows = b.solve(points, rig)['rotations']
        for letter in ('L', 'R'):
            for first, second in (('UpperArm', 'Forearm'), ('Thigh', 'Calf')):
                x, y = rows[letter+' '+first], rows[letter+' '+second]
                self.assertLess(m.dot(m.cross(x[0], y[0]), x[2]), 0)
                self.assertAlmostEqual(m.dot(x[2], y[2]), 1)

    def test_straight_limbs_stay_finite(self):
        points, rig = fixture()
        for side in ('left', 'right'):
            for first, mid, end in (('shoulder', 'elbow', 'wrist'), ('hip', 'knee', 'ankle')):
                points[side+'_'+end] = tuple(2*y-x for x,y in zip(points[side+'_'+first], points[side+'_'+mid]))
        result = b.solve(points, rig)
        self.assertTrue(all(math.isfinite(x) for rows in result['rotations'].values() for row in rows for x in row))

    def test_near_straight_knee_uses_reference_plane_without_moving_ankle(self):
        points, rig = fixture()
        points['left_knee'] = (11, 2, -45)
        points['left_ankle'] = (8, 0, -85)
        solution = b.solve(points, rig)
        corrected = b.stabilize_knee(points, 'left', 'L', rig, solution['rotations']['Pelvis'])
        self.assertGreater(b.length(m.sub(corrected, points['left_knee'])), 0.1)
        self.assertAlmostEqual(b.length(m.sub(corrected, points['left_hip'])),
                               b.length(m.sub(points['left_knee'], points['left_hip'])), places=6)
        self.assertAlmostEqual(b.length(m.sub(points['left_ankle'], corrected)),
                               b.length(m.sub(points['left_ankle'], points['left_knee'])), places=6)
        reference_hinge = m.world(m.local(rig['L Thigh']['axes'][2], rig['Pelvis']['axes']),
                                  solution['rotations']['Pelvis'])
        def hinge(knee):
            return b.scale(m.unit(m.cross(m.sub(knee, points['left_hip']),
                                           m.sub(points['left_ankle'], knee))), -1)
        self.assertLess(b.angle(hinge(corrected), reference_hinge),
                        b.angle(hinge(points['left_knee']), reference_hinge))
        self.assertIn('L Thigh', solution['projection_degrees'])
        self.assertIn('L Calf', solution['projection_degrees'])
        self.assertNotIn('R Thigh', solution['projection_degrees'])
        for part, start, end in (('Thigh', 'hip', 'knee'), ('Calf', 'knee', 'ankle')):
            expected = dict(points, left_knee=corrected)
            self.assertLess(b.angle(m.world(rig['L '+part]['aim'], solution['rotations']['L '+part]),
                                    m.sub(expected['left_'+end], expected['left_'+start])), 1e-5)

    def test_near_straight_elbow_caps_excess_roll_and_preserves_wrist(self):
        points, rig = fixture()
        axis = (1., 0., 0.)
        reference = m.projected(rig['L UpperArm']['axes'][2], axis)
        radial = m.unit(m.cross(reference, axis))
        turned = b.add(b.scale(radial, math.cos(math.radians(110))),
                       b.scale(m.cross(axis, radial), math.sin(math.radians(110))))
        points['left_shoulder'] = (0., 0., 0.)
        points['left_elbow'] = b.add((10., 0., 0.), b.scale(turned, 1.5))
        points['left_wrist'] = (20., 0., 0.)
        corrected = b.stabilize_elbow(points, 'left', 'L', rig, rig['L Clavicle']['axes'])
        self.assertGreater(b.length(m.sub(corrected, points['left_elbow'])), 0.1)
        self.assertAlmostEqual(b.length(m.sub(corrected, points['left_shoulder'])),
                               b.length(m.sub(points['left_elbow'], points['left_shoulder'])), places=6)
        self.assertAlmostEqual(b.length(m.sub(points['left_wrist'], corrected)),
                               b.length(m.sub(points['left_wrist'], points['left_elbow'])), places=6)
        def hinge(elbow):
            return b.scale(m.unit(m.cross(m.sub(elbow, points['left_shoulder']),
                                           m.sub(points['left_wrist'], elbow))), -1)
        self.assertAlmostEqual(b.angle(hinge(corrected), reference), 60., delta=0.01)

    def test_fingers_single_side_keep_distal_and_other_hand(self):
        points, rig = fixture()
        add_fingers(points)
        result = b.solve(points, rig)
        for f in range(5):
            name = 'L Finger'+str(f)+'2'
            self.assertIn(name, result['preserved'])
            self.assertRows(b.relative(result['rotations'][name], result['rotations'][rig[name]['parent']]), rig[name]['local_axes'])
        self.assertIn('R Hand', result['preserved'])
        self.assertNotIn('L Hand', result['preserved'])
        self.assertEqual(len(result['directions']), len(b.BODY_LINKS)+10)

    def test_two_hands_and_missing_chain(self):
        points, rig = fixture()
        add_fingers(points)
        add_fingers(points, 'right')
        del points['left_thumb_2']
        result = b.solve(points, rig)
        self.assertIn('L Finger0', result['preserved'])
        self.assertIn('R Finger0', result['directions'])

    def test_point_validation_rejects_bad_inputs(self):
        points, _ = fixture()
        b.validate_points(points, a.client.BODY_JOINTS)
        for bad in ((float('nan'), 0, 0), (0, 1), None):
            altered = dict(points, head=bad)
            with self.assertRaises(ValueError):
                b.validate_points(altered, a.client.BODY_JOINTS)
        with self.assertRaises(ValueError):
            b.validate_points(dict(points, head=points['neck']), a.client.BODY_JOINTS)
        del points['head']
        with self.assertRaises(ValueError):
            b.validate_points(points, a.client.BODY_JOINTS)

    def test_source_scale_does_not_change_rotations(self):
        points, rig = fixture()
        first = b.solve(points, rig)
        second = b.solve({k:b.scale(p,100) for k,p in points.items()},rig)
        for name in first['rotations']:
            self.assertRows(first['rotations'][name], second['rotations'][name])


class TargetSelectionTests(unittest.TestCase):
    def test_replace_removes_numbered_body_and_old_fingers_only(self):
        base = a.client.BODY_NAME_PREFIX
        nodes = [SimpleNamespace(name=name, valid=True) for name in (
            base+'pelvis', base+'1_pelvis', base+'3_left_thumb_3',
            'MediaPipeHandPoseSkeleton_wrist', 'Bip001', 'UserBone')]
        rt = SimpleNamespace(objects=nodes, isValidNode=lambda n:n.valid,
                             getUserProp=lambda n,k:None,
                             delete=lambda n:setattr(n,'valid',False))
        a.client._delete_previous(rt, joint_names=a.client.BODY_AND_FINGER_JOINTS, strict=True)
        self.assertEqual([n.valid for n in nodes], [False, False, False, True, True, True])

    def test_replace_rejects_silent_deletion_failure(self):
        n = SimpleNamespace(name=a.client.BODY_NAME_PREFIX+'pelvis')
        rt = SimpleNamespace(objects=[n], isValidNode=lambda n:True,
                             getUserProp=lambda n,k:None, delete=lambda n:None)
        with self.assertRaises(a.client.MaxPoseError):
            a.client._delete_previous(rt, strict=True)

    def runtime(self, count=2):
        roots = [SimpleNamespace(name='Actor'+str(i), kind='Biped_Object', handle=i,
                                 controller=SimpleNamespace(kind='Vertical_Horizontal_Turn')) for i in range(count)]
        child = SimpleNamespace(name='Actor0 L Hand', kind='Biped_Object', handle=10,
                                controller=SimpleNamespace(kind='BipDriven_Control', rootNode=roots[0]))
        rt = SimpleNamespace(objects=roots+[child], selection=[], isValidNode=lambda n:n is not None,
                             classOf=lambda n:n.kind, getHandleByAnim=lambda n:n.handle)
        return rt, roots, child

    def test_selected_child_resolves_character(self):
        rt, roots, child = self.runtime()
        rt.selection = [child]
        self.assertIs(a.resolve_target(rt), roots[0])

    def test_multiple_targets_rejected(self):
        rt, roots, child = self.runtime()
        with self.assertRaises(RuntimeError):
            a.resolve_target(rt)
        rt.selection = roots
        with self.assertRaises(RuntimeError):
            a.resolve_target(rt)

    def test_only_character_auto_selected(self):
        rt, roots, child = self.runtime(1)
        self.assertIs(a.resolve_target(rt), roots[0])

    def test_explicit_invalid_target_does_not_fall_back(self):
        rt, _, _ = self.runtime(1)
        with self.assertRaises(RuntimeError):
            a.resolve_target(rt, SimpleNamespace(kind='Dummy'))

    def test_waiting_request_rejects_scene_and_node_changes(self):
        target = SimpleNamespace(handle=1)
        rt = SimpleNamespace(maxFilePath='scene/', maxFileName='a.max',
                             isValidNode=lambda n:True, getHandleByAnim=lambda n:n.handle)
        retargeter = a.BodyRetargeter()
        request = dict(target=target, handle=1, scene_serial=0, file='scene/a.max', bones={'Pelvis':1})
        with patch.object(a.hand,'runtime',return_value=rt), patch.object(a,'inspect_rig',return_value=({'Pelvis':target},{})):
            self.assertIs(retargeter.validate_request(request),target)
            target.handle = 2
            with self.assertRaises(RuntimeError):
                retargeter.validate_request(request)
            target.handle = 1
            retargeter._scene_changed()
            with self.assertRaises(RuntimeError):
                retargeter.validate_request(request)


if __name__ == '__main__':
    unittest.main()
