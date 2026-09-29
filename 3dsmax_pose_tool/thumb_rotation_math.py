"""Reference-based shortest-swing rotations used by the hybrid thumb solver."""

import importlib
import math

m = importlib.import_module(__package__ + ".hand_retarget_math")


def swing(vector, source, target, fallback):
    """Shortest swing, with a stable reference axis at 180 degrees."""
    a, b = m.unit(source), m.unit(target)
    cosine = max(-1.0, min(1.0, m.dot(a, b)))
    axis = m.cross(a, b)
    sine = math.sqrt(m.dot(axis, axis))
    if sine < 1e-8:
        if cosine > 0:
            return tuple(vector)
        axis = m.unit(m.sub(fallback, tuple(m.dot(fallback, a) * k for k in a)))
        return tuple(2 * m.dot(axis, vector) * axis[i] - vector[i] for i in range(3))
    axis = tuple(k / sine for k in axis)
    av = m.cross(axis, vector)
    return tuple(vector[i] * cosine + av[i] * sine + axis[i] * m.dot(axis, vector) * (1-cosine) for i in range(3))


def solve(directions, target_basis, hand_axes, reference, fingers=m.FINGERS):
    """Use the saved parent-local reference, never the last output pose."""
    result = {}
    for finger in fingers:
        parent = hand_axes
        for i, ref in enumerate(reference[finger]):
            inherited = tuple(m.world(row, parent) for row in ref["local_axes"])
            aim = m.unit(m.world(ref["aim"], inherited))
            desired = m.unit(m.world(directions[finger][i], target_basis))
            fallback = min(inherited, key=lambda row: abs(m.dot(row, aim)))
            rows = tuple(m.unit(swing(row, aim, desired, fallback)) for row in inherited)
            result[ref["name"]] = {"axes": rows, "direction": desired, "off_plane_degrees": 0.0}
            parent = rows
    return result
