"""Small independent rig fixture with non-X foot offsets and mirrored limbs."""
import importlib

a = importlib.import_module('3dsmax_pose_tool.biped_body')
b = a.maths
m = b.m


def fixture():
    points = {'pelvis': (0, 0, 0), 'spine_1': (0, 0, 12), 'spine_2': (2, 0, 26),
              'spine_3': (5, -1, 40), 'neck': (6, -3, 55), 'head': (4, -5, 65)}
    values = {'left': [(12, -2, 48), (22, -4, 46), (25, -22, 30), (28, -42, 36),
                        (8, -2, -3), (1, -8, -44), (2, -5, -85), (3, -18, -94)],
              'right': [(-3, 1, 52), (-12, 4, 56), (-28, 10, 68), (-45, 7, 83),
                        (-8, 2, 3), (-15, 9, -25), (-30, 38, -24), (-39, 42, -30)]}
    for side, data in values.items():
        for part, p in zip(('collar', 'shoulder', 'elbow', 'wrist', 'hip', 'knee', 'ankle', 'foot'), data):
            points[side + '_' + part] = p
    positions = {'Pelvis': (0, 0, 95), 'Spine': (0, 0, 102), 'Spine1': (0, 0, 116),
                 'Spine2': (0, 1, 130), 'Neck': (0, 3, 151), 'Head': (0, 1, 158)}
    for letter, sign in (('L', 1), ('R', -1)):
        for part, p in {'Clavicle': (3, 3, 145), 'UpperArm': (19, 4, 142),
                        'Forearm': (40, 1, 126), 'Hand': (58, -15, 113),
                        'Thigh': (10, 0, 95), 'Calf': (12, 1, 51),
                        'Foot': (14, 6, 11), 'Toe0': (14, -7, 2)}.items():
            positions[letter+' '+part] = (sign*p[0], p[1], p[2])
        for f in range(5):
            for i, suffix in enumerate(('', '1', '2', 'Nub')):
                positions[letter+' Finger'+str(f)+suffix] = (sign*(61 + i*3), -19+f*2, 108-i)
    rig, pending = {}, dict(a.CHILDREN)
    for part in list(pending):
        if part == 'Pelvis':
            parent = None
        elif part.startswith('Spine'):
            parent = {'Spine': 'Pelvis', 'Spine1': 'Spine', 'Spine2': 'Spine1'}[part]
        elif part in ('Neck', 'Head'):
            parent = 'Spine2' if part == 'Neck' else 'Neck'
        else:
            letter, bone = part.split(' ')
            if bone == 'Clavicle':
                parent = 'Spine2'
            elif bone == 'Thigh':
                parent = 'Pelvis'
            elif bone.startswith('Finger'):
                parent = letter+' Hand' if len(bone) == 7 else letter+' '+bone[:7]+('' if bone[-1] == '1' else '1')
            else:
                parent = letter+' '+{'UpperArm':'Clavicle', 'Forearm':'UpperArm', 'Hand':'Forearm',
                                    'Calf':'Thigh', 'Foot':'Calf', 'Toe0':'Foot'}[bone]
        child = a.CHILDREN[part]
        offset = m.sub(positions[child], positions[part]) if child else (1, 0, 0)
        if part.endswith('Foot'):
            axes = ((0., 0., -1.), (0., -1., 0.), (-1., 0., 0.))
        elif part.endswith('Hand'):
            axes = b.aim_basis(offset, (0, 0, 1))
        elif child:
            hint = (1, 0, 0) if part in ('Pelvis', 'Spine', 'Spine1', 'Spine2', 'Neck') else (0, 0, 1)
            axes = b.aim_basis(offset, hint)
        else:
            axes = rig[parent]['axes']
        rig[part] = {'parent':parent, 'position':positions[part], 'axes':axes,
                     'local_axes':b.relative(axes, rig[parent]['axes'] if parent else b.IDENTITY),
                     'aim':m.local(offset,axes)}
    return points, rig


def add_fingers(points, side='left'):
    sign = 1 if side == 'left' else -1
    wrist = points[side+'_wrist']
    for f, finger in enumerate(m.FINGERS):
        for i in (1, 2, 3):
            points[side+'_'+finger+'_'+str(i)] = (wrist[0]+sign*(f-2)*2,
                                                 wrist[1]-6-i*3, wrist[2]-i*i*.3)

