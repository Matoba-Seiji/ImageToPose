import importlib
import unittest

from pose_server import protocol, server
from pose_server.mediapipe_hand_pose import HAND_JOINT_NAMES

client = importlib.import_module('3dsmax_pose_tool.pose_client')


class ServerProtocolTests(unittest.TestCase):
    def test_body_layout_matches_max_client(self):
        self.assertEqual(protocol.BODY_AND_FINGER_JOINTS, client.BODY_AND_FINGER_JOINTS)
        self.assertEqual(protocol.BODY_PARENTS, client.BODY_PARENTS)

    def test_hand_layout_matches_max_client(self):
        self.assertEqual(HAND_JOINT_NAMES, client.HAND_JOINTS)

    def test_body_and_hand_responses_validate_in_max(self):
        for mode, names, convert in (
            ('body', protocol.BODY_AND_FINGER_JOINTS, server._pose_response),
            ('hand', HAND_JOINT_NAMES, server._hand_pose_response),
        ):
            frame = {'joints': {name: {'position': [0., 1., 2.]} for name in names},
                     'include_fingers': True,
                     'coordinate_system': 'root_relative_mediapipe_world'}
            client._validate_frame(convert(frame), mode=mode)


if __name__ == '__main__':
    unittest.main()
