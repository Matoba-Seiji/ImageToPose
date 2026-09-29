import importlib
import math
import unittest
from types import SimpleNamespace

from body_test_data import fixture

b = importlib.import_module('3dsmax_pose_tool.body_retarget_math')
client = importlib.import_module('3dsmax_pose_tool.pose_client')
service = importlib.import_module('pose_server.server')


def face_from_basis(basis):
    right, forward, up = basis
    def point(x=0., y=0., z=0.):
        return b.add(b.add(b.scale(right, x), b.scale(forward, y)), b.scale(up, z))
    return {
        'left_eye': point(-.03, 0., .04),
        'right_eye': point(.03, 0., .04),
        'left_ear': point(-.07),
        'right_ear': point(.07),
        'nose': point(0., .045, .005),
        'mouth_left': point(-.02, 0., -.035),
        'mouth_right': point(.02, 0., -.035),
    }


class HeadOrientationTests(unittest.TestCase):
    def test_face_basis_drives_head_and_missing_face_preserves_it(self):
        points, rig = fixture()
        reference_body = b.body_orientation_basis(
            rig['L UpperArm']['position'], rig['R UpperArm']['position'],
            b.m.sub(rig['Neck']['position'], rig['Pelvis']['position']))
        face = face_from_basis(reference_body)
        neutral = b.solve(points, rig, face_points=face)
        self.assertTrue(neutral['head_rotation_estimated'])
        self.assertNotIn('Head', neutral['preserved'])
        self.assertLess(max(b.angle(a, c) for a, c in zip(
            neutral['rotations']['Head'], rig['Head']['axes'])), 1e-4)

        theta = math.radians(30.)
        right, forward, up = reference_body
        turned = (
            b.add(b.scale(right, math.cos(theta)), b.scale(forward, math.sin(theta))),
            b.add(b.scale(forward, math.cos(theta)), b.scale(right, -math.sin(theta))),
            up,
        )
        turned_solution = b.solve(points, rig, face_points=face_from_basis(turned))
        self.assertTrue(turned_solution['head_rotation_estimated'])
        self.assertGreater(max(b.angle(a, c) for a, c in zip(
            turned_solution['rotations']['Head'], neutral['rotations']['Head'])), 20.)

        no_face = b.solve(points, rig)
        self.assertFalse(no_face['head_rotation_estimated'])
        self.assertIn('Head', no_face['preserved'])
        face['nose'] = face['left_ear']
        face['right_eye'] = face['left_eye']
        invalid = b.solve(points, rig, face_points=face)
        self.assertFalse(invalid['head_rotation_estimated'])
        self.assertIn('Head', invalid['preserved'])

    def test_world_face_points_reach_max_solver_with_visibility_guard(self):
        rt = SimpleNamespace(units=SimpleNamespace(decodeValue=lambda value: 100.))
        face = {name: {'position': [1., 2., 3.], 'visibility': .9}
                for name in client.FACE_WORLD_NAMES}
        frame = {'coordinate_system': 'root_relative_mediapipe_world',
                 'face_world': face}
        converted = client._face_position_map(frame, rt)
        self.assertEqual(converted['nose'], (100., 300., -200.))
        response = service._pose_response(dict(frame, joints={}))
        self.assertEqual(response['face_world'], face)
        face['nose']['visibility'] = .2
        self.assertIsNone(client._face_position_map(frame, rt))


if __name__ == '__main__':
    unittest.main()
