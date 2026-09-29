"""Rig-bound thumb reference cache, separate from scene animation and Undo."""

import copy
import hashlib
import json
import math
from pathlib import Path

VERSION = 2
HAND_POSE_TOLERANCE = 1e-3
CACHE_DIR = Path(__file__).resolve().parents[1] / 'outputs' / 'hand_references'


def cache_path(scene, side='left'):
    if side not in {'left', 'right'}:
        raise ValueError('Invalid target hand')
    # Keep the original left-hand filename to preserve the validated cache.
    suffix = '' if side == 'left' else '.right'
    return CACHE_DIR / (hashlib.sha256(scene.encode('utf-8')).hexdigest() + suffix + '.json')


def _hand_pose(value):
    try:
        rows = tuple(tuple(float(v) for v in row) for row in value)
    except (TypeError, ValueError):
        return None
    if len(rows) != 4 or any(len(row) != 3 for row in rows):
        return None
    if not all(math.isfinite(v) for row in rows for v in row):
        return None
    return rows


def hand_pose_matches(record, hand_pose, tolerance=HAND_POSE_TOLERANCE):
    """Return whether a cache was captured from the current hand pose."""
    saved = _hand_pose(record.get('hand_pose')) if isinstance(record, dict) else None
    current = _hand_pose(hand_pose)
    if saved is None or current is None:
        return False
    return max(
        abs(a - b)
        for saved_row, current_row in zip(saved, current)
        for a, b in zip(saved_row, current_row)
    ) <= float(tolerance)


def compatible(record, scene, handles, chain, hand_pose=None):
    if not isinstance(record, dict):
        return False
    if record.get('version') != VERSION or record.get('scene') != scene or record.get('handles') != handles:
        return False
    entries = record.get('reference')
    if not isinstance(entries, list) or len(entries) != len(chain):
        return False
    try:
        for ref, bone in zip(entries, chain):
            if any(ref[k] != bone[k] for k in ('name', 'parent', 'child')):
                return False
            if abs(ref['length'] - bone['length']) > max(1e-4, bone['length'] * 1e-4):
                return False
            rows = ref['local_axes']
            if len(rows) != 3 or any(len(r) != 3 for r in rows) or len(ref['aim']) != 3:
                return False
            values = [v for row in rows for v in row] + list(ref['aim']) + [ref['length']]
            if not all(math.isfinite(float(v)) for v in values):
                return False
            for i, row in enumerate(rows):
                for j, other in enumerate(rows):
                    if abs(sum(x*y for x,y in zip(row,other)) - (1 if i == j else 0)) > 1e-3:
                        return False
            if abs(sum(v*v for v in ref['aim']) - 1) > 1e-3:
                return False
    except (KeyError, TypeError, ValueError):
        return False
    if hand_pose is not None and not hand_pose_matches(record, hand_pose):
        return False
    return True


def load(scene, handles, chain, side='left', hand_pose=None):
    try:
        record = json.loads(cache_path(scene, side).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    return record if compatible(record, scene, handles, chain, hand_pose=hand_pose) else None


def save(scene, handles, chain, reference, source, side='left', hand_pose=None):
    record = dict(version=VERSION, scene=scene, handles=copy.deepcopy(handles),
                  reference=copy.deepcopy(reference), source=source)
    if hand_pose is not None:
        normalized_pose = _hand_pose(hand_pose)
        if normalized_pose is None:
            raise ValueError('Invalid hand pose for thumb reference')
        record['hand_pose'] = normalized_pose
    if not compatible(record, scene, handles, chain):
        raise ValueError('Invalid thumb reference for this rig')
    path = cache_path(scene, side)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, indent=2), encoding='utf-8')
    temporary.replace(path)
    return record
