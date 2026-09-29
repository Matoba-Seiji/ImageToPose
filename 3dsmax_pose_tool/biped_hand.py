"""Biped hand adapters. All scene entry points must run on Max's UI thread."""

import importlib
import uuid

maths = importlib.import_module(__package__ + ".hand_retarget_math")
thumb_maths = importlib.import_module(__package__ + ".thumb_rotation_math")
thumb_cache = importlib.import_module(__package__ + ".thumb_reference")
TARGET_HANDS = {"left": "Bip001 L Hand", "right": "Bip001 R Hand"}
ARM_CHAIN_PARTS = ("Clavicle", "UpperArm", "Forearm", "Hand")
THUMB_LOCAL_POSE_TOLERANCE = 2e-3
HAND_SYNC_POSE_TOLERANCE = 5e-3
HAND_SYNC_WRITE_TOLERANCE = 5e-3


def runtime():
    from pymxs import runtime as rt
    try:
        from PySide6.QtCore import QThread
        from PySide6.QtWidgets import QApplication
    except ImportError:
        from PySide2.QtCore import QThread
        from PySide2.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None or QThread.currentThread() != app.thread():
        raise RuntimeError("Biped scene operations require the Max main thread")
    return rt


def vector(p):
    return (float(p.x), float(p.y), float(p.z))


def matrix(n):
    t = n.transform
    return tuple(vector(row) for row in (t.row1, t.row2, t.row3, t.row4))


def axes(n):
    rows = matrix(n)[:3]
    if any(abs(maths.dot(r, r)-1) > 1e-3 for r in rows) or abs(maths.dot(maths.cross(rows[0], rows[1]), rows[2])-1) > 1e-3:
        raise RuntimeError("Non-rigid Biped transform: " + str(n.name))
    return tuple(maths.unit(r) for r in rows)


def node(rt, name):
    matches = [n for n in rt.objects if str(n.name) == name]
    if len(matches) != 1:
        raise RuntimeError("Expected exactly one scene node: " + name)
    return matches[0]


def check_mode(rt):
    body = node(rt, "Bip001").controller
    for prop in ("figureMode", "footstepMode", "motionMode", "mixerMode", "moveAllMode"):
        if bool(rt.getProperty(body, rt.Name(prop))):
            raise RuntimeError("Disable Biped " + prop + " before retargeting")
    if rt.theHold.Holding():
        raise RuntimeError("Another Max Undo transaction is active")


def key_state(rt, n):
    try:
        count = int(rt.numKeys(n.controller))
    except Exception:
        return None
    if count < 0:
        return None
    return tuple(str(rt.getKeyTime(n.controller, i)) for i in range(1, count+1))


def permitted_isolated_finger_keys(name, before, after, frame, target_side="left"):
    """Allow Biped to key the selected side's other finger tracks at this frame."""
    return (is_hand_finger_node(name, target_side) and before is not None and after is not None
            and set(after) == set(before) | {frame})


def permitted_hand_sync_keys(name, before, after, frame, target_side="left"):
    """Allow current keys used while synchronizing Biped arm state."""
    selected = is_hand_sync_node(name, target_side) or is_hand_finger_node(name, target_side)
    return (selected and before is not None and after is not None
            and set(after) == set(before) | {frame})


def is_hand_sync_node(name, target_side="left"):
    """Return whether a node belongs to the selected arm's Biped solve."""
    if target_side not in TARGET_HANDS:
        raise ValueError("Target hand must be left or right")
    side_prefix = "Bip001 " + ("L" if target_side == "left" else "R")
    chain_prefix = side_prefix + " "
    if name in {chain_prefix + part for part in ARM_CHAIN_PARTS}:
        return True
    return (name.startswith(chain_prefix) and "ForeTwist" in name) or (
        name.startswith(side_prefix + "UpArmTwist")
    )


def is_hand_finger_node(name, target_side="left"):
    """Return whether a node is one of the selected side's finger chain."""
    if target_side not in TARGET_HANDS:
        raise ValueError("Target hand must be left or right")
    side_prefix = "Bip001 " + ("L" if target_side == "left" else "R")
    return name.startswith(side_prefix + " Finger")


class HandRetargeter:
    def __init__(self, target_side="left"):
        if target_side not in TARGET_HANDS:
            raise ValueError("Target hand must be left or right")
        self.target_side = target_side
        self.hand_name = TARGET_HANDS[target_side]
        self.prefix = "Bip001 " + ("L" if target_side == "left" else "R") + " "
        self.rig = None
        self.restore_token = None
        self.callback_id = "HandRetarget_" + uuid.uuid4().hex
        self.history_serial = 0
        self._callback = None
        self.thumb_record = None

    def _history_changed(self):
        self.history_serial += 1
        self.restore_token = None

    def _scene_changed(self):
        self._history_changed()
        self.rig = None
        self.thumb_record = None

    def _install_callbacks(self, rt):
        if self._callback is not None:
            return
        self._callback = self._history_changed
        for event in ("sceneUndo", "sceneRedo"):
            rt.callbacks.addScript(rt.Name(event), self._callback, id=rt.Name(self.callback_id))
        for event in ("systemPreReset", "filePreOpen"):
            rt.callbacks.addScript(rt.Name(event), self._scene_changed, id=rt.Name(self.callback_id))

    def close(self):
        if self._callback is not None:
            runtime().callbacks.removeScripts(id=runtime().Name(self.callback_id))
            self._callback = None

    def inspect_rig(self):
        """Read topology and fixed Biped axes; never capture a pose as a zero."""
        rt = runtime()
        check_mode(rt)
        self._install_callbacks(rt)
        h = node(rt, self.hand_name)
        rig = {}
        handles = {self.hand_name: int(rt.getHandleByAnim(h))}
        for f, finger in enumerate(maths.FINGERS):
            parent = h
            chain = [node(rt, self.prefix + "Finger" + str(f) + s) for s in ("", "1", "2", "Nub")]
            rig[finger] = []
            for n, child in zip(chain, chain[1:]):
                if str(rt.classOf(n)) != "Biped_Object" or n.parent != parent or child.parent != n:
                    raise RuntimeError("Unexpected finger hierarchy: " + str(n.name))
                a = axes(n)
                direction = maths.unit(maths.sub(matrix(child)[3], matrix(n)[3]))
                aim = maths.local(direction, a)
                if max(abs(x-y) for x,y in zip(aim, (1.0, 0.0, 0.0))) > 1e-3:
                    raise RuntimeError("Unsupported Biped bone axis (expected local +X): " + str(n.name))
                rig[finger].append({"name": str(n.name), "child": str(child.name), "parent": str(parent.name), "length": maths.dot(maths.sub(matrix(child)[3], matrix(n)[3]), direction)})
                handles[str(n.name)] = int(rt.getHandleByAnim(n))
                handles[str(child.name)] = int(rt.getHandleByAnim(child))
                parent = n
        self.rig = rig
        self.handles = handles
        self.scene = str(rt.maxFilePath) + str(rt.maxFileName)
        self.restore_token = None
        return {"frame": str(rt.sliderTime), "bones": 15, "hand": self.hand_name}

    def _validate(self, rt):
        if self.rig is None:
            raise RuntimeError("No previous hand operation to restore")
        if self.scene != str(rt.maxFilePath) + str(rt.maxFileName):
            raise RuntimeError("Scene changed since the last hand operation")
        for name, handle in self.handles.items():
            if int(rt.getHandleByAnim(node(rt, name))) != handle:
                raise RuntimeError("Rig changed since the last hand operation")
        for chain in self.rig.values():
            for ref in chain:
                if str(node(rt, ref["name"]).parent.name) != ref["parent"] or str(node(rt, ref["child"]).parent.name) != ref["name"]:
                    raise RuntimeError("Hierarchy changed since the last hand operation")
                v = maths.sub(matrix(node(rt, ref["child"]))[3], matrix(node(rt, ref["name"]))[3])
                if abs(maths.dot(v, maths.unit(v)) - ref["length"]) > max(1e-4, ref["length"] * 1e-4):
                    raise RuntimeError("Bone length changed since the last hand operation")

    def _stamp(self, rt):
        return (int(rt.theHold.GetGlobalPutCount()), self.history_serial)

    def can_restore(self):
        if self.restore_token is None:
            return False
        rt = runtime()
        try:
            self._validate(rt)
            return not rt.theHold.Holding() and self._stamp(rt) == self.restore_token
        except Exception:
            return False

    def restore(self):
        if not self.can_restore():
            raise RuntimeError("Restore unavailable after another scene edit or Undo")
        rt = runtime()
        self.restore_token = None
        rt.execute("max undo")
        rt.redrawViews()

    def _solve_current_pose(self, rt, frame):
        directions = maths.source_directions(frame, target_side=self.target_side)
        h = node(rt, self.hand_name)
        palm = maths.palm_basis(maths.sub(matrix(node(rt, self.prefix + "Finger2"))[3], matrix(h)[3]), axes(h)[1])
        reference = self._thumb_reference(rt)
        solution = maths.solve_hinges(directions, palm, self.rig, fingers=maths.FINGERS[1:])
        solution.update(thumb_maths.solve(directions, palm, axes(h), {"thumb": reference}, fingers=("thumb",)))
        return solution

    def _commit_current_arm_pose(self, rt):
        """Synchronize Biped's arm controller with the visible hand pose.

        Biped can expose a hand transform that looks current while its
        internal arm solve still uses an older parent state. Writing the first
        finger then re-solves that stale chain and rejects the target direction.
        Capture the selected arm chain first and commit it from the clavicle to
        the hand so each child is written against the same parent pose.
        """
        prefix = self.prefix
        values = []
        for part in ARM_CHAIN_PARTS:
            n = node(rt, prefix + part)
            tm = n.transform
            values.append((
                n,
                rt.Point3(float(tm.row4.x), float(tm.row4.y), float(tm.row4.z)),
                rt.Quat(
                    float(tm.rotation.x), float(tm.rotation.y),
                    float(tm.rotation.z), float(tm.rotation.w),
                ),
            ))
        for n, position, rotation in values:
            rt.biped.setTransform(n, rt.Name("pos"), position, True)
            rt.biped.setTransform(n, rt.Name("rotation"), rotation, True)

    def _thumb_reference(self, rt):
        chain = self.rig['thumb']
        names = {self.hand_name} | {r['name'] for r in chain} | {r['child'] for r in chain}
        handles = {name: self.handles[name] for name in names}
        hand_pose = matrix(node(rt, self.hand_name))
        if thumb_cache.compatible(self.thumb_record, self.scene, handles, chain, hand_pose=hand_pose):
            return self.thumb_record['reference']
        self.thumb_record = thumb_cache.load(
            self.scene, handles, chain, side=self.target_side, hand_pose=hand_pose
        )
        if self.thumb_record is not None:
            return self.thumb_record['reference']

        # A rigid hand move changes the world frame but not the thumb's local
        # rig basis. Rebind that basis to the current hand pose. If the thumb
        # chain changed locally as well, capture the new pose instead.
        structural = thumb_cache.load(self.scene, handles, chain, side=self.target_side)
        if structural is not None and self._thumb_chain_matches_reference(
            rt, chain, structural['reference']
        ):
            self.thumb_record = thumb_cache.save(
                self.scene,
                handles,
                chain,
                structural['reference'],
                'hand_pose_rebind',
                side=self.target_side,
                hand_pose=hand_pose,
            )
            return self.thumb_record['reference']

        # Capture when the hand and thumb pose no longer agree with the saved
        # reference, but do not recapture merely because a prior output changed
        # the finger pose while the hand itself stayed put.
        reference = []
        for ref in chain:
            n, child = node(rt, ref['name']), node(rt, ref['child'])
            a = axes(n)
            direction = maths.unit(maths.sub(matrix(child)[3], matrix(n)[3]))
            reference.append(dict(ref, local_axes=tuple(maths.local(row, axes(node(rt, ref['parent']))) for row in a),
                                  aim=maths.local(direction, a)))
        self.thumb_record = thumb_cache.save(
            self.scene,
            handles,
            chain,
            reference,
            'hand_pose_changed',
            side=self.target_side,
            hand_pose=hand_pose,
        )
        return self.thumb_record['reference']

    def _thumb_chain_matches_reference(self, rt, chain, reference):
        if len(chain) != len(reference):
            return False
        try:
            for ref, saved in zip(chain, reference):
                if any(saved[key] != ref[key] for key in ('name', 'parent', 'child')):
                    return False
                n = node(rt, ref['name'])
                parent = node(rt, ref['parent'])
                child = node(rt, ref['child'])
                current_axes = axes(n)
                current_local_axes = tuple(
                    maths.local(row, axes(parent)) for row in current_axes
                )
                direction = maths.unit(
                    maths.sub(matrix(child)[3], matrix(n)[3])
                )
                current_aim = maths.local(direction, current_axes)
                local_error = max(
                    abs(a - b)
                    for current_row, saved_row in zip(
                        current_local_axes, saved['local_axes']
                    )
                    for a, b in zip(current_row, saved_row)
                )
                aim_error = max(
                    abs(a - b) for a, b in zip(current_aim, saved['aim'])
                )
                if max(local_error, aim_error) > THUMB_LOCAL_POSE_TOLERANCE:
                    return False
        except (KeyError, TypeError, ValueError, RuntimeError):
            return False
        return True

    def apply(self, frame):
        """Write the solved fingers through isolated Biped arm tracks.
        """
        rt = runtime()
        check_mode(rt)
        self.inspect_rig()
        solution = self._solve_current_pose(rt, frame)
        refs = [ref for f in maths.FINGERS for ref in self.rig[f]]
        names = {ref["name"] for ref in refs}
        allowed = names | {ref["child"] for ref in refs}
        body_controller = node(rt, "Bip001").controller
        original_sep_tracks = bool(body_controller.sepArmsTracks)
        write_time = str(rt.sliderTime)
        isolated_finger_changes = []
        self.restore_token = None
        protected = [(n, matrix(n), key_state(rt, n)) for n in rt.objects
                     if str(n.name).startswith("Bip001") and str(n.name) not in allowed]
        arm_sync_names = {
            str(n.name)
            for n in rt.objects
            if str(n.name).startswith("Bip001")
            and (
                is_hand_sync_node(str(n.name), self.target_side)
                or is_hand_finger_node(str(n.name), self.target_side)
            )
        }
        rt.theHold.Begin()
        try:
            # Normalize the parent arm while Biped is on its native shared
            # arm solve. Switching to separated tracks first can make the
            # opposite arm fall back to an older shared pose, especially for
            # the right-hand Biped. The separated setting is enabled only for
            # the actual finger writes below and remains enabled afterward.
            body_controller.sepArmsTracks = False
            self._commit_current_arm_pose(rt)
            body_controller.sepArmsTracks = True
            if not bool(body_controller.sepArmsTracks):
                raise RuntimeError("Unable to enable isolated Biped arm tracks")
            self._validate(rt)
            solution = self._solve_current_pose(rt, frame)
            synced_arm = {
                str(n.name): matrix(n)
                for n, _, _ in protected
                if str(n.name) in arm_sync_names
            }
            for n, previous, _ in protected:
                if str(n.name) not in arm_sync_names:
                    continue
                delta = max(
                    abs(a - b)
                    for r, s in zip(previous, synced_arm[str(n.name)])
                    for a, b in zip(r, s)
                )
                if delta > HAND_SYNC_POSE_TOLERANCE:
                    raise RuntimeError(
                        "Biped arm pose changed while synchronizing " + str(n.name)
                    )
            before = {ref["name"]: matrix(node(rt, ref["name"])) for ref in refs}
            lengths = {ref["name"]: maths.dot(maths.sub(matrix(node(rt, ref["child"]))[3], before[ref["name"]][3]), maths.unit(maths.sub(matrix(node(rt, ref["child"]))[3], before[ref["name"]][3]))) for ref in refs}
            for ref in refs:
                rows = solution[ref["name"]]["axes"]
                tm = rt.Matrix3(*(rt.Point3(*r) for r in rows), rt.Point3(0, 0, 0))
                rt.biped.setTransform(node(rt, ref["name"]), rt.Name("rotation"), tm.rotation, True)
            errors = {}
            for ref in refs:
                n, child = node(rt, ref["name"]), node(rt, ref["child"])
                v = maths.sub(matrix(child)[3], matrix(n)[3])
                if abs(maths.dot(v, maths.unit(v))-lengths[ref["name"]]) > max(1e-4, lengths[ref["name"]]*1e-4):
                    raise RuntimeError("Biped bone length changed: " + ref["name"])
                err = max(abs(a-b) for a, b in zip(maths.unit(v), solution[ref["name"]]["direction"]))
                errors[ref["name"]] = err
                if err > 1e-3:
                    raise RuntimeError("Biped did not accept the target direction: " + ref["name"])
                if ref["parent"] == self.hand_name and max(abs(a-b) for a,b in zip(matrix(n)[3], before[ref["name"]][3])) > 1e-4:
                    raise RuntimeError("Finger root moved: " + ref["name"])
            for n, previous, keys in protected:
                current = matrix(n)
                if str(n.name) in arm_sync_names:
                    sync_pose = synced_arm[str(n.name)]
                    pose_delta = max(
                        abs(a - b)
                        for r, s in zip(sync_pose, current)
                        for a, b in zip(r, s)
                    )
                    if pose_delta > HAND_SYNC_WRITE_TOLERANCE:
                        raise RuntimeError(
                            "Biped arm pose changed during isolated finger write: "
                            + str(n.name)
                        )
                elif max(abs(a-b) for r,s in zip(previous, current) for a,b in zip(r,s)) > 1e-4:
                    raise RuntimeError("Non-target Biped transform changed: " + str(n.name))
                after_keys = key_state(rt, n)
                isolated_finger_key = permitted_isolated_finger_keys(
                    str(n.name), keys, after_keys, write_time, self.target_side
                )
                arm_sync_key = permitted_hand_sync_keys(
                    str(n.name), keys, after_keys, write_time,
                    self.target_side
                )
                if after_keys != keys and not isolated_finger_key and not arm_sync_key:
                    raise RuntimeError("Non-target Biped transform or key times changed: " + str(n.name) + ". All writes rolled back. Isolated finger tracks reject non-target writes.")
                if after_keys != keys:
                    if isolated_finger_key:
                        isolated_finger_changes.append(str(n.name))
            rt.theHold.Accept("MediaPipe " + self.target_side.title() + " Hand " + uuid.uuid4().hex[:8])
        except Exception:
            rt.theHold.Cancel()
            body_controller.sepArmsTracks = original_sep_tracks
            rt.redrawViews()
            raise
        self.restore_token = self._stamp(rt)
        rt.redrawViews()
        return {"frame": str(rt.sliderTime), "hand": self.hand_name, "target_side": self.target_side, "written": len(refs), "max_direction_error": max(errors.values()), "separate_arm_tracks": bool(body_controller.sepArmsTracks), "isolated_finger_key_nodes": isolated_finger_changes,
                "max_off_plane_degrees": max(abs(solution[ref["name"]]["off_plane_degrees"]) for ref in refs)}
