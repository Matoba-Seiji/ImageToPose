"""Position-driven Biped pose solving, independent of Max and Qt."""

import importlib
import math

m = importlib.import_module(__package__ + ".hand_retarget_math")
swing = importlib.import_module(__package__ + ".thumb_rotation_math").swing
IDENTITY = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))
ELBOW_REFERENCE_PLANE_LIMIT_DEGREES = 60.
BODY_LINKS = {
    "Spine": ("spine_1", "spine_2"),
    "Spine1": ("spine_2", "spine_3"),
    "Spine2": ("spine_3", "neck"),
    "Neck": ("neck", "head"),
}
for _side, _letter in (("left", "L"), ("right", "R")):
    for _part, _a, _b in (("Clavicle", "collar", "shoulder"),
                         ("UpperArm", "shoulder", "elbow"),
                         ("Forearm", "elbow", "wrist"),
                         ("Thigh", "hip", "knee"),
                         ("Calf", "knee", "ankle"),
                         ("Foot", "ankle", "foot")):
        BODY_LINKS[_letter + " " + _part] = (_side + "_" + _a, _side + "_" + _b)


def length(v):
    return math.sqrt(m.dot(v, v))


def scale(v, k):
    return tuple(x * k for x in v)


def add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def angle(a, b):
    return math.degrees(math.acos(max(-1., min(1., m.dot(m.unit(a), m.unit(b))))))


def compose(local_axes, parent_axes):
    return tuple(m.world(row, parent_axes) for row in local_axes)


def relative(axes, parent):
    return tuple(m.local(row, parent) for row in axes)


def aim_basis(aim, transverse):
    x = m.unit(aim)
    z = m.projected(transverse, x)
    return x, m.unit(m.cross(z, x)), z


def align_pair(local_aim, local_transverse, aim, transverse):
    source = aim_basis(local_aim, local_transverse)
    target = aim_basis(aim, transverse)
    return tuple(m.world(m.local(row, source), target) for row in IDENTITY)


def aim_swing(inherited, local_aim, desired):
    old = m.world(local_aim, inherited)
    fallback = min(inherited, key=lambda row: abs(m.dot(row, m.unit(old))))
    return tuple(m.unit(swing(row, old, desired, fallback)) for row in inherited)


def validate_points(points, required):
    result = {}
    for name, value in points.items():
        try:
            p = tuple(float(x) for x in value)
            if len(p) != 3 or not all(math.isfinite(x) for x in p):
                raise ValueError()
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("Invalid source joint: " + name) from error
        result[name] = p
    missing = set(required) - set(result)
    if missing:
        raise ValueError("Missing source joints: " + ", ".join(sorted(missing)))
    for a, b in BODY_LINKS.values():
        if length(m.sub(result[b], result[a])) < 1e-7:
            raise ValueError("Zero-length source bone: " + a)
    return result


def stabilize_knee(points, side, letter, rig, pelvis_axes):
    """Bias a nearly straight knee toward the rig's anatomical bend plane.

    A small knee offset in one image can spin the inferred knee plane through a
    large angle. Rotate only that offset about the hip-to-ankle line: the hip,
    ankle, knee bend magnitude and limb lengths remain unchanged.
    """
    hip = points[side + "_hip"]
    knee = points[side + "_knee"]
    ankle = points[side + "_ankle"]
    upper = m.unit(m.sub(knee, hip))
    lower = m.unit(m.sub(ankle, knee))
    bend = angle(upper, lower)
    if bend >= 35.:
        return knee
    axis = m.unit(m.sub(ankle, hip))
    offset = m.sub(knee, hip)
    along = m.dot(offset, axis)
    radial = m.sub(offset, scale(axis, along))
    radius = length(radial)
    if radius < 1e-7:
        return knee

    reference_hinge = m.world(
        m.local(rig[letter + " Thigh"]["axes"][2], rig["Pelvis"]["axes"]),
        pelvis_axes,
    )
    if length(m.cross(reference_hinge, axis)) < 1e-5:
        return knee
    target_hinge = m.projected(reference_hinge, axis)
    target_radial = m.unit(m.cross(target_hinge, axis))
    source_radial = scale(radial, 1. / radius)
    correction = math.atan2(
        m.dot(m.cross(source_radial, target_radial), axis),
        m.dot(source_radial, target_radial),
    )
    # Full reference-plane correction below 10 degrees; ease to zero at 35.
    amount = max(0., min(1., (35. - bend) / 25.))
    amount = amount * amount * (3. - 2. * amount)
    turn = correction * amount
    rotated = add(scale(source_radial, math.cos(turn)),
                  scale(m.cross(axis, source_radial), math.sin(turn)))
    return add(hip, add(scale(axis, along), scale(rotated, radius)))


def stabilize_elbow(points, side, letter, rig, clavicle_axes):
    """Limit upper-arm roll inferred from a nearly straight elbow."""
    shoulder = points[side + "_shoulder"]
    elbow = points[side + "_elbow"]
    wrist = points[side + "_wrist"]
    upper = m.unit(m.sub(elbow, shoulder))
    lower = m.unit(m.sub(wrist, elbow))
    bend = angle(upper, lower)
    if bend >= 35.:
        return elbow
    axis = m.unit(m.sub(wrist, shoulder))
    offset = m.sub(elbow, shoulder)
    along = m.dot(offset, axis)
    radial = m.sub(offset, scale(axis, along))
    radius = length(radial)
    if radius < 1e-7:
        return elbow

    reference_hinge = m.world(
        m.local(rig[letter + " UpperArm"]["axes"][2],
                rig[letter + " Clavicle"]["axes"]),
        clavicle_axes,
    )
    if length(m.cross(reference_hinge, axis)) < 1e-5:
        return elbow
    target_hinge = m.projected(reference_hinge, axis)
    target_radial = m.unit(m.cross(target_hinge, axis))
    source_radial = scale(radial, 1. / radius)
    correction = math.atan2(
        m.dot(m.cross(source_radial, target_radial), axis),
        m.dot(source_radial, target_radial),
    )
    excess = max(0., abs(correction) - math.radians(ELBOW_REFERENCE_PLANE_LIMIT_DEGREES))
    if excess < 1e-7:
        return elbow
    # Cap the reference-plane difference at 60 degrees below 20 degrees of
    # elbow flexion; fade the correction out by 35 degrees.
    amount = max(0., min(1., (35. - bend) / 15.))
    amount = amount * amount * (3. - 2. * amount)
    turn = math.copysign(excess * amount, correction)
    rotated = add(scale(source_radial, math.cos(turn)),
                  scale(m.cross(axis, source_radial), math.sin(turn)))
    return add(shoulder, add(scale(axis, along), scale(rotated, radius)))


def body_orientation_basis(left, right, up):
    """Anatomical right, forward and up from shoulders and torso."""
    lateral = m.unit(m.sub(right, left))
    vertical = m.projected(up, lateral)
    return lateral, m.unit(m.cross(vertical, lateral)), vertical


def estimated_head_axes(points, face_points, rig):
    """Map the observed face frame onto the reference head's anatomical axes."""
    if not face_points:
        return None
    try:
        eye_mid = scale(add(face_points["left_eye"], face_points["right_eye"]), .5)
        ear_mid = scale(add(face_points["left_ear"], face_points["right_ear"]), .5)
        mouth_mid = scale(add(face_points["mouth_left"], face_points["mouth_right"]), .5)
        eye_span = m.unit(m.sub(face_points["right_eye"], face_points["left_eye"]))
        ear_span = m.unit(m.sub(face_points["right_ear"], face_points["left_ear"]))
        if m.dot(eye_span, ear_span) < .35:
            return None
        lateral = m.unit(add(scale(eye_span, .4), scale(ear_span, .6)))
        vertical = m.projected(m.sub(eye_mid, mouth_mid), lateral)
        face_basis = lateral, m.unit(m.cross(vertical, lateral)), vertical
        nose_direction = m.sub(face_points["nose"], ear_mid)
        if m.dot(nose_direction, face_basis[1]) <= 0.:
            return None
        source_body = body_orientation_basis(
            points["left_shoulder"], points["right_shoulder"],
            m.sub(points["neck"], points["pelvis"]),
        )
        if angle(face_basis[0], source_body[0]) > 110. or angle(face_basis[2], source_body[2]) > 100.:
            return None
        reference_body = body_orientation_basis(
            rig["L UpperArm"]["position"], rig["R UpperArm"]["position"],
            m.sub(rig["Neck"]["position"], rig["Pelvis"]["position"]),
        )
        return compose(relative(rig["Head"]["axes"], reference_body), face_basis)
    except (KeyError, ValueError, TypeError, OverflowError):
        return None


def solve(points, rig, face_points=None):
    """Return world rotations and measured directions; rig records are parent-first.

    Each rig record contains parent, axes, local_axes, aim (local child offset),
    and position. Unknown terminal rotations inherit their original local pose.
    """
    result, directions, projected_errors, preserved = {}, {}, {}, ["Head", "L Toe0", "R Toe0"]
    head_axes = estimated_head_axes(points, face_points, rig)
    hip_span = m.sub(points["left_hip"], points["right_hip"])
    shoulder_span = m.sub(points["left_shoulder"], points["right_shoulder"])
    torso = m.sub(points["spine_1"], points["pelvis"])
    pelvis = rig["Pelvis"]
    target_span = m.local(m.sub(rig["L Thigh"]["position"], rig["R Thigh"]["position"]), pelvis["axes"])
    # Hip width determines pelvic roll; project the up direction off that axis.
    desired_up = m.projected(torso, m.unit(hip_span))
    local_up = m.projected(pelvis["aim"], m.unit(target_span))
    result["Pelvis"] = align_pair(local_up, target_span, desired_up, hip_span)
    effective_points = dict(points)
    for side, letter in (("left", "L"), ("right", "R")):
        knee_name = side + "_knee"
        corrected = stabilize_knee(points, side, letter, rig, result["Pelvis"])
        if length(m.sub(corrected, points[knee_name])) > 1e-7:
            effective_points[knee_name] = corrected
            for part, start, end in (("Thigh", "hip", "knee"),
                                     ("Calf", "knee", "ankle")):
                a, b = side + "_" + start, side + "_" + end
                projected_errors[letter + " " + part] = angle(
                    m.sub(points[b], points[a]),
                    m.sub(effective_points[b], effective_points[a]),
                )
    hinges = {}
    for side, letter in (("left", "L"), ("right", "R")):
        a, b = letter + " Thigh", letter + " Calf"
        da = m.unit(m.sub(effective_points[BODY_LINKS[a][1]], effective_points[BODY_LINKS[a][0]]))
        db = m.unit(m.sub(effective_points[BODY_LINKS[b][1]], effective_points[BODY_LINKS[b][0]]))
        cross = m.cross(da, db)
        hinges[a] = hinges[b] = scale(m.unit(cross), -1.) if length(cross) > 1e-4 else None

    for name, ref in rig.items():
        if name == "Pelvis":
            continue
        if name in ("L UpperArm", "R UpperArm"):
            letter = name[0]
            side = "left" if letter == "L" else "right"
            elbow_name = side + "_elbow"
            corrected = stabilize_elbow(effective_points, side, letter, rig,
                                        result[letter + " Clavicle"])
            effective_points[elbow_name] = corrected
            if length(m.sub(effective_points[elbow_name], points[elbow_name])) > 1e-7:
                for part, start, end in (("UpperArm", "shoulder", "elbow"),
                                         ("Forearm", "elbow", "wrist")):
                    a, b = side + "_" + start, side + "_" + end
                    projected_errors[letter + " " + part] = angle(
                        m.sub(points[b], points[a]),
                        m.sub(effective_points[b], effective_points[a]),
                    )
            a, b = letter + " UpperArm", letter + " Forearm"
            da = m.unit(m.sub(effective_points[BODY_LINKS[a][1]], effective_points[BODY_LINKS[a][0]]))
            db = m.unit(m.sub(effective_points[BODY_LINKS[b][1]], effective_points[BODY_LINKS[b][0]]))
            cross = m.cross(da, db)
            hinges[a] = hinges[b] = scale(m.unit(cross), -1.) if length(cross) > 1e-4 else None
        parent = result.get(ref["parent"], rig[ref["parent"]]["axes"] if ref["parent"] in rig else IDENTITY)
        inherited = compose(ref["local_axes"], parent)
        rows = inherited
        if name == "Head" and head_axes is not None:
            rows = head_axes
            preserved.remove("Head")
        elif name in BODY_LINKS:
            a, b = BODY_LINKS[name]
            desired = m.unit(m.sub(effective_points[b], effective_points[a]))
            if name in ("Spine", "Spine1", "Spine2"):
                amount = ("Spine", "Spine1", "Spine2").index(name) / 2.
                transverse = tuple((1-amount)*h + amount*s for h, s in zip(hip_span, shoulder_span))
                rows = align_pair(ref["aim"], (0, 0, 1), desired, transverse)
            elif name in hinges:
                transverse = hinges[name]
                if transverse is None:
                    transverse = inherited[2]
                    if length(m.cross(desired, transverse)) < 1e-7:
                        transverse = min(inherited, key=lambda r: abs(m.dot(r, desired)))
                rows = align_pair(ref["aim"], (0, 0, 1), desired, transverse)
            else:
                rows = aim_swing(inherited, ref["aim"], desired)
            directions[name] = desired
        result[name] = rows

    for side, letter in (("left", "L"), ("right", "R")):
        hand = letter + " Hand"
        roots = [side + "_" + f + "_1" for f in ("index", "middle", "pinky")]
        if all(n in points for n in roots) and hand in rig:
            wrist = points[side + "_wrist"]
            forward = m.sub(points[roots[1]], wrist)
            normal = m.cross(m.sub(points[roots[2]], wrist), m.sub(points[roots[0]], wrist))
            if side == "right":
                normal = scale(normal, -1)
            palm = m.palm_basis(forward, normal)
            local_forward = m.local(m.sub(rig[letter + " Finger2"]["position"], rig[hand]["position"]), rig[hand]["axes"])
            # Both palms use local +Y. Use the actual wrist-to-middle offset.
            result[hand] = align_pair(local_forward, (0, 1, 0), palm[0], palm[1])
        else:
            preserved.append(hand)

        for f, finger in enumerate(m.FINGERS):
            chain = [letter + " Finger" + str(f) + suffix for suffix in ("", "1", "2")]
            if not all(n in rig for n in chain):
                preserved.append(side + "_" + finger)
                continue
            source = [side + "_" + finger + "_" + str(i) for i in (1, 2, 3)]
            complete = all(n in points for n in source)
            parent = result[hand]
            hinge = None
            for i, name in enumerate(chain):
                ref = rig[name]
                inherited = compose(ref["local_axes"], parent)
                rows = inherited
                if complete and i < 2:
                    desired = m.unit(m.sub(points[source[i+1]], points[source[i]]))
                    if finger == "thumb":
                        rows = aim_swing(inherited, ref["aim"], desired)
                    else:
                        if hinge is None:
                            local_direction = m.local(desired, result[hand])
                            hinge = m.world(m.finger_hinge([local_direction]), result[hand])
                        projected = m.projected(desired, hinge)
                        rows = align_pair(ref["aim"], (0, 0, 1), projected, hinge)
                        projected_errors[name] = angle(desired, projected)
                        desired = projected
                    directions[name] = desired
                else:
                    preserved.append(name)
                result[name] = parent = rows
    return {"rotations": result, "directions": directions,
            "projection_degrees": projected_errors, "preserved": preserved,
            "head_rotation_estimated": head_axes is not None}
