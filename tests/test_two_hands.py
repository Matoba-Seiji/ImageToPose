import copy
import importlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

m = importlib.import_module('3dsmax_pose_tool.hand_retarget_math')
a = importlib.import_module('3dsmax_pose_tool.biped_hand')
cache = importlib.import_module('3dsmax_pose_tool.thumb_reference')


def sample():
    points = {'wrist': {'position': [0,0,0]}}
    for j, chain in enumerate(m.CHAINS.values()):
        for i,name in enumerate(chain):
            points[name] = {'position':[2-j,2+i,-0.2*i*i]}
    return dict(coordinate_system='root_relative_mediapipe_hand_world',handedness='left',landmarks=points)


class TwoHandTests(unittest.TestCase):
    def test_fixed_target_names(self):
        self.assertEqual(a.HandRetargeter('left').hand_name,'Bip001 L Hand')
        self.assertEqual(a.HandRetargeter('right').hand_name,'Bip001 R Hand')
        with self.assertRaises(ValueError):
            a.HandRetargeter('other')
        with self.assertRaises(ValueError):
            m.source_directions(sample(),'other')

    def test_all_source_target_pairs_and_right_palm_sign(self):
        left = sample()
        right = copy.deepcopy(left)
        right['handedness'] = 'right'
        for p in right['landmarks'].values():
            p['position'][0] *= -1
        saved = copy.deepcopy((left,right))
        for side in ('left','right'):
            d1 = m.source_directions(left,side)
            d2 = m.source_directions(right,side)
            self.assertEqual(d1,d2)
            self.assertGreater(d1['index'][2][1],0)
        ld,rd = m.source_directions(left,'left'),m.source_directions(left,'right')
        for f in m.FINGERS:
            for l,r in zip(ld[f],rd[f]):
                self.assertEqual((l[0],l[1],-l[2]),r)
        self.assertEqual((left,right),saved)

    def test_reference_files_are_isolated_and_left_path_compatible(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(cache,'CACHE_DIR',Path(folder)):
            chain = [dict(name='thumb'+str(i),parent='p'+str(i),child='c'+str(i),length=1.) for i in range(3)]
            ref = [dict(r,local_axes=((1,0,0),(0,1,0),(0,0,1)),aim=(1,0,0)) for r in chain]
            left = cache.save('scene',{'L':1},chain,ref,'test')
            original = cache.cache_path('scene').read_bytes()
            cache.save('scene',{'R':2},chain,ref,'test',side='right')
            self.assertEqual(cache.cache_path('scene'),cache.cache_path('scene','left'))
            self.assertNotEqual(cache.cache_path('scene'),cache.cache_path('scene','right'))
            self.assertEqual(original,cache.cache_path('scene').read_bytes())
            self.assertIsNotNone(cache.load('scene',{'R':2},chain,side='right'))
            self.assertIsNotNone(cache.load('scene',{'L':1},chain))
            self.assertIsNone(cache.load('scene',{'L':1},chain,side='right'))


if __name__ == '__main__':
    unittest.main()
