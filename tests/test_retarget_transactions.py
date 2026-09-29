import importlib
import unittest
from contextlib import ExitStack, nullcontext
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from max_test_support import inject_rotation_failure


body = importlib.import_module('3dsmax_pose_tool.biped_body')
hand = body.hand
IDENTITY = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))


class _Hold:
    def __init__(self, writes):
        self.writes = writes
        self.cancelled = False
        self.accepted = False

    def Begin(self):
        self.before = list(self.writes)

    def Cancel(self):
        self.writes[:] = self.before
        self.cancelled = True

    def Accept(self, label):
        self.accepted = True


def runtime():
    writes = []
    return SimpleNamespace(
        writes=writes, objects=[], sliderTime=0,
        theHold=_Hold(writes), redrawViews=Mock(),
        biped=SimpleNamespace(setTransform=lambda *args: writes.append(args)),
        Point3=lambda *v: v, Matrix3=lambda *rows: SimpleNamespace(rotation=rows),
        Name=lambda value: value, getHandleByAnim=id, classOf=lambda node: 'Biped_Object',
    )


class RetargetTransactionTests(unittest.TestCase):
    def test_body_apply_forwards_face_and_preview_to_active_solver(self):
        adapter = body.BodyRetargeter()
        frame, points, face = {}, {'pelvis': (0, 0, 0)}, {'nose': (1, 2, 3)}
        rt = runtime()
        with ExitStack() as stack:
            stack.enter_context(patch.object(hand, 'runtime', return_value=rt))
            stack.enter_context(patch.object(body.client, '_validate_frame'))
            stack.enter_context(patch.object(body.client, '_body_frame_joint_names', return_value=('pelvis',)))
            stack.enter_context(patch.object(body.client, '_position_map', return_value=(points, 100)))
            stack.enter_context(patch.object(body.client, '_face_position_map', return_value=face))
            apply_points = stack.enter_context(patch.object(adapter, '_apply_points', return_value='written'))
            self.assertEqual(adapter.apply(frame, 'Actor', 7, keep_source_bones=True), 'written')
            apply_points.assert_called_once_with(points, 'Actor', 7, source_frame=frame, face_points=face)

    def test_body_rotation_failure_cancels_transaction_and_restores_tracks(self):
        rt = runtime()
        root = SimpleNamespace(name='Actor', controller=SimpleNamespace(
            **{flag: False for flag in body.TRACK_FLAGS}))
        nodes = {name: SimpleNamespace(name='Actor ' + name, parent=root)
                 for name in ('Pelvis', 'Head')}
        rig = {name: {'position': (0., 0., 0.), 'size': (1., 1., 1.)} for name in nodes}
        adapter = body.BodyRetargeter()
        request = {'target': root, 'file': 'test.max', 'handle': 1, 'time': 0}
        pymxs = ModuleType('pymxs')
        pymxs.attime = lambda time: nullcontext()
        with ExitStack() as stack:
            stack.enter_context(patch.dict('sys.modules', {'pymxs': pymxs}))
            stack.enter_context(patch.object(body, '_REFERENCES', {}))
            stack.enter_context(patch.object(body, 'EFFECTORS', ()))
            stack.enter_context(patch.object(adapter, 'capture_request', return_value=request))
            stack.enter_context(patch.object(body, 'inspect_rig', return_value=(nodes, rig)))
            stack.enter_context(patch.object(body, '_descendants', return_value=list(nodes.values())))
            stack.enter_context(patch.object(body, '_driven_handles', return_value=set()))
            stack.enter_context(patch.object(body, '_keys', return_value={}))
            stack.enter_context(patch.object(body, '_key_settings', return_value={}))
            stack.enter_context(patch.object(hand, 'matrix', return_value=IDENTITY + ((0., 0., 0.),)))
            stack.enter_context(patch.object(body.maths, 'validate_points', return_value={}))
            stack.enter_context(patch.object(body.maths, 'solve', return_value={
                'rotations': {name: IDENTITY for name in nodes}}))
            with inject_rotation_failure(hand, rt, 2):
                with self.assertRaisesRegex(RuntimeError, 'Injected test rotation failure'):
                    adapter._apply_points({}, root, 0)
        self.assertEqual(rt.writes, [])
        self.assertTrue(rt.theHold.cancelled)
        self.assertFalse(rt.theHold.accepted)
        self.assertIsNone(adapter.restore_token)
        self.assertTrue(all(not getattr(root.controller, flag) for flag in body.TRACK_FLAGS))

    def test_both_hands_write_all_fingers_and_roll_back_partial_writes(self):
        for side in ('left', 'right'):
            with self.subTest(side=side):
                rt = runtime()
                root = SimpleNamespace(name='Bip001', controller=SimpleNamespace(sepArmsTracks=False))
                adapter = hand.HandRetargeter(side)
                nodes = {'Bip001': root}
                adapter.rig = {}
                for index, finger in enumerate(hand.maths.FINGERS):
                    names = [adapter.prefix + 'Finger' + str(index) + suffix
                             for suffix in ('', '1', '2', 'Nub')]
                    for i, name in enumerate(names):
                        nodes[name] = SimpleNamespace(name=name, position=(float(i), 0., 0.))
                    adapter.rig[finger] = [{'name': name, 'child': child}
                                          for name, child in zip(names, names[1:])]
                solution = {ref['name']: {'axes': IDENTITY}
                            for chain in adapter.rig.values() for ref in chain}
                with ExitStack() as stack:
                    stack.enter_context(patch.object(hand, 'check_mode'))
                    stack.enter_context(patch.object(hand, 'node', side_effect=lambda runtime, name: nodes[name]))
                    stack.enter_context(patch.object(hand, 'matrix', side_effect=lambda node: IDENTITY + (node.position,)))
                    stack.enter_context(patch.object(adapter, 'inspect_rig'))
                    stack.enter_context(patch.object(adapter, '_validate'))
                    stack.enter_context(patch.object(adapter, '_commit_current_arm_pose'))
                    stack.enter_context(patch.object(adapter, '_solve_current_pose', return_value=solution))
                    # Reaching write 15 proves neither side silently skips a finger.
                    with inject_rotation_failure(hand, rt, 15, node_contains=' Finger'):
                        with self.assertRaisesRegex(RuntimeError, 'Injected test rotation failure'):
                            adapter.apply({})
                self.assertEqual(rt.writes, [])
                self.assertTrue(rt.theHold.cancelled)
                self.assertFalse(rt.theHold.accepted)
                self.assertFalse(root.controller.sepArmsTracks)
                self.assertIsNone(adapter.restore_token)


if __name__ == '__main__':
    unittest.main()
