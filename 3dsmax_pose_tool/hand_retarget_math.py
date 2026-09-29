"""Left-hand palm-relative angles with planar finger hinge constraints."""

import math

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
CHAINS = {f: tuple(f + "_" + s for s in (("cmc", "mcp", "ip", "tip") if f == "thumb" else ("mcp", "pip", "dip", "tip"))) for f in FINGERS}

def sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def unit(v):
    length = math.sqrt(dot(v, v))
    if not math.isfinite(length) or length < 1e-9:
        raise ValueError("Degenerate hand direction or palm plane")
    return tuple(x / length for x in v)


def world(v, basis):
    return tuple(sum(v[j] * basis[j][i] for j in range(3)) for i in range(3))


def local(v, basis):
    return tuple(dot(v, row) for row in basis)


def palm_basis(forward, normal):
    y = unit(normal)
    x = unit(sub(forward, tuple(dot(forward, y) * k for k in y)))
    return (x, y, unit(cross(x, y)))


def source_directions(frame, target_side="left"):
    """Return target-hand palm-relative directions without modifying the frame.

    Reflect right-hand landmarks across camera X before rebuilding the palm
    basis. Only an explicit right label triggers conversion; older responses
    without handedness keep the previous left-hand interpretation.
    """
    if target_side not in {"left", "right"}:
        raise ValueError("Target hand must be left or right")
    if frame.get("coordinate_system") != "root_relative_mediapipe_hand_world":
        raise ValueError("Expected raw MediaPipe hand world coordinates")
    mirror_right = str(frame.get("handedness", "")).strip().lower() == "right"
    points = {}
    for name in ("wrist",) + tuple(n for chain in CHAINS.values() for n in chain):
        try:
            v = tuple(float(x) for x in frame["landmarks"][name]["position"])
            if len(v) != 3 or not all(math.isfinite(x) for x in v):
                raise ValueError()
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Missing or invalid landmark: " + name) from error
        points[name] = (-v[0], v[1], v[2]) if mirror_right else v
    # In MediaPipe camera coordinates, pinky x index points out of the LEFT
    # palm. Reversing this maps flexion to the back of the target hand and
    # also swaps its radial/ulnar axis. Never infer this sign from finger curl.
    normal = cross(sub(points["pinky_mcp"], points["wrist"]), sub(points["index_mcp"], points["wrist"]))
    basis = palm_basis(sub(points["middle_mcp"], points["wrist"]), normal)
    directions = {f: tuple(local(unit(sub(points[b], points[a])), basis) for a, b in zip(chain, chain[1:])) for f, chain in CHAINS.items()}
    # Both target palms use +Y. Their radial/ulnar directions are opposite,
    # so mirror only the palm-local transverse coordinate for a right target.
    if target_side == "right":
        directions = {f: tuple((x, y, -z) for x, y, z in chain) for f, chain in directions.items()}
    return directions


def projected(v, normal):
    return unit(sub(v, tuple(dot(v, normal) * k for k in normal)))


def finger_hinge(directions):
    """A fixed bending plane for a non-thumb finger in palm coordinates."""
    first = unit(directions[0])
    flat = (first[0], 0.0, first[2])
    if dot(flat, flat) < 1e-12:
        flat = (1.0, 0.0, 0.0)
    return unit(cross(unit(flat), (0.0, 1.0, 0.0)))


def solve_hinges(directions, target_basis, rig, fingers=FINGERS[1:]):
    """+X aims down each Biped bone; +Z is a common hinge within its chain.

    Four-finger roots have spread and flexion, but no independent axial roll.
    The two distal joints rotate only about inherited local Z. The thumb is
    handled separately by the reference-based shortest-swing solver.
    No angle clamps, smoothing, or pose-dependent calibration are applied.
    """
    result = {}
    for finger in fingers:
        if finger not in FINGERS[1:]:
            raise ValueError("Hinge solving supports only non-thumb fingers")
        hinge = finger_hinge(directions[finger])
        previous = None
        for i, ref in enumerate(rig[finger]):
            source = unit(directions[finger][i])
            aim = projected(source, hinge)
            axes = (world(aim, target_basis), world(unit(cross(hinge, aim)), target_basis), world(hinge, target_basis))
            angle = None if previous is None else math.atan2(dot(cross(previous, aim), hinge), dot(previous, aim))
            result[ref["name"]] = {
                "axes": axes, "direction": axes[0],
                "hinge_angle_degrees": None if angle is None else math.degrees(angle),
                "off_plane_degrees": math.degrees(math.asin(max(-1.0, min(1.0, dot(source, hinge))))),
            }
            previous = aim
    return result
