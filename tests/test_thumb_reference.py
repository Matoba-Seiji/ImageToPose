import copy
import importlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

cache = importlib.import_module('3dsmax_pose_tool.thumb_reference')
m = importlib.import_module('3dsmax_pose_tool.hand_retarget_math')
old = importlib.import_module('3dsmax_pose_tool.thumb_rotation_math')
IDENTITY = ((1,0,0),(0,1,0),(0,0,1))


class ThumbReferenceTests(unittest.TestCase):
    def setUp(self):
        self.chain = [dict(name='thumb'+str(i), parent='parent'+str(i), child='child'+str(i), length=1.0) for i in range(3)]
        self.reference = [dict(r,local_axes=IDENTITY,aim=(1,0,0)) for r in self.chain]
        self.handles = {'hand':1,'thumb':2}

    def test_roundtrip_and_rig_identity(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(cache,'CACHE_DIR',Path(folder)):
            cache.save('scene',self.handles,self.chain,self.reference,'legacy_reference')
            record = cache.load('scene',self.handles,self.chain)
            self.assertIsNotNone(record)
            self.assertIsNone(cache.load('other_scene',self.handles,self.chain))
            self.assertIsNone(cache.load('scene',{'hand':1,'thumb':3},self.chain))
            modified = copy.deepcopy(self.chain)
            modified[0]['length'] = 2
            self.assertIsNone(cache.load('scene',self.handles,modified))
            modified = copy.deepcopy(self.chain)
            modified[0]['parent'] = 'other'
            self.assertIsNone(cache.load('scene',self.handles,modified))

    def test_corrupt_reference_rejected(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(cache,'CACHE_DIR',Path(folder)):
            cache.cache_path('scene').write_text('{invalid',encoding='utf-8')
            self.assertIsNone(cache.load('scene',self.handles,self.chain))
            bad = copy.deepcopy(self.reference)
            bad[0]['aim'] = (float('nan'),0,0)
            with self.assertRaises(ValueError):
                cache.save('scene',self.handles,self.chain,bad,'test')

    def test_hand_pose_rebind_metadata(self):
        pose = ((1,0,0),(0,1,0),(0,0,1),(4,5,6))
        moved = ((0,-1,0),(1,0,0),(0,0,1),(8,5,6))
        with tempfile.TemporaryDirectory() as folder, patch.object(cache,'CACHE_DIR',Path(folder)):
            cache.save('scene',self.handles,self.chain,self.reference,'test',hand_pose=pose)
            record = cache.load('scene',self.handles,self.chain,hand_pose=pose)
            self.assertIsNotNone(record)
            self.assertTrue(cache.hand_pose_matches(record,pose))
            self.assertIsNone(cache.load('scene',self.handles,self.chain,hand_pose=moved))

    def test_hybrid_does_not_evaluate_thumb_plane(self):
        directions = {f: ((1,0,0),(0,1,0),(0,0,1)) for f in m.FINGERS}
        rig = {f:[dict(name=f+str(i),local_axes=IDENTITY,aim=(1,0,0)) for i in range(3)] for f in m.FINGERS}
        # Thumb third segment is perpendicular to its inferred plane. The
        # hybrid solver must not call the old planar thumb path at all.
        for f in m.FINGERS[1:]:
            directions[f] = ((1,0,0),(0,1,0),(-1,0,0))
        four = m.solve_hinges(directions,IDENTITY,rig,fingers=m.FINGERS[1:])
        thumb = old.solve(directions,IDENTITY,IDENTITY,rig,fingers=('thumb',))
        self.assertEqual(len(four),12)
        self.assertEqual(len(thumb),3)
        for i in range(3):
            for a,b in zip(thumb['thumb'+str(i)]['axes'][0],directions['thumb'][i]):
                self.assertAlmostEqual(a,b)


if __name__ == '__main__':
    unittest.main()
