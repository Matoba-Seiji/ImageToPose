"""Joint layout shared by the body inference response and Max client contract."""

BODY_JOINTS = (
    'pelvis', 'spine_1', 'spine_2', 'spine_3', 'neck', 'head',
    'left_collar', 'left_shoulder', 'left_elbow', 'left_wrist',
    'right_collar', 'right_shoulder', 'right_elbow', 'right_wrist',
    'left_hip', 'left_knee', 'left_ankle', 'left_foot',
    'right_hip', 'right_knee', 'right_ankle', 'right_foot',
)
FINGER_JOINTS = tuple(
    '{}_{}_{}'.format(side, finger, segment)
    for side in ('left', 'right')
    for finger in ('thumb', 'index', 'middle', 'ring', 'pinky')
    for segment in (1, 2, 3)
)
BODY_AND_FINGER_JOINTS = BODY_JOINTS + FINGER_JOINTS
BODY_PARENTS = {
    'spine_1': 'pelvis', 'spine_2': 'spine_1', 'spine_3': 'spine_2',
    'neck': 'spine_3', 'head': 'neck',
}
for side in ('left', 'right'):
    for child, parent in (('collar', 'spine_3'), ('shoulder', side + '_collar'),
                          ('elbow', side + '_shoulder'), ('wrist', side + '_elbow'),
                          ('hip', 'pelvis'), ('knee', side + '_hip'),
                          ('ankle', side + '_knee'), ('foot', side + '_ankle')):
        BODY_PARENTS[side + '_' + child] = parent
    for finger in ('thumb', 'index', 'middle', 'ring', 'pinky'):
        for segment in (1, 2, 3):
            name = '{}_{}_{}'.format(side, finger, segment)
            BODY_PARENTS[name] = (side + '_wrist' if segment == 1 else
                                  '{}_{}_{}'.format(side, finger, segment - 1))
