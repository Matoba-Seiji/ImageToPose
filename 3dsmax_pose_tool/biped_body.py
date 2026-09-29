"""Transactional whole-body Biped adapter. Scene entry points run on Max UI thread."""

import importlib
import uuid

client = importlib.import_module(__package__ + ".pose_client")
hand = importlib.import_module(__package__ + ".biped_hand")
maths = importlib.import_module(__package__ + ".body_retarget_math")
m = maths.m
TRACK_FLAGS = ("sepArmsTracks", "sepLegsTracks", "sepSpineTracks", "sepNeckTracks")
_REFERENCES = globals().get("_REFERENCES", {})
EFFECTORS = ("L Hand", "R Hand", "L Foot", "R Foot")
CHILDREN = {"Pelvis": "Spine", "Spine": "Spine1", "Spine1": "Spine2",
            "Spine2": "Neck", "Neck": "Head", "Head": None}
for _side in ("L", "R"):
    for _part, _child in (("Clavicle", "UpperArm"), ("UpperArm", "Forearm"),
                          ("Forearm", "Hand"), ("Hand", "Finger2"),
                          ("Thigh", "Calf"), ("Calf", "Foot"), ("Foot", "Toe0"), ("Toe0", None)):
        CHILDREN[_side + " " + _part] = _side + " " + _child if _child else None
    for _finger in range(5):
        _base = _side + " Finger" + str(_finger)
        for _suffix, _next in (("", "1"), ("1", "2"), ("2", "Nub")):
            CHILDREN[_base + _suffix] = _base + _next


def _root(rt, n):
    if n is None or not rt.isValidNode(n) or str(rt.classOf(n)) != "Biped_Object":
        return None
    if str(rt.classOf(n.controller)) == "Vertical_Horizontal_Turn":
        return n
    return n.controller.rootNode


def resolve_target(rt, target=None):
    if isinstance(target, str):
        target = hand.node(rt, target)
    if target is not None:
        root = _root(rt, target)
        if root is None:
            raise RuntimeError("Select a Biped node as the target")
        return root
    selected = {}
    for n in rt.selection:
        root = _root(rt, n)
        if root is not None:
            selected[int(rt.getHandleByAnim(root))] = root
    if len(selected) == 1:
        return next(iter(selected.values()))
    if len(selected) > 1:
        raise RuntimeError("Multiple selected Bipeds; select one character")
    roots = [n for n in rt.objects if str(rt.classOf(n)) == "Biped_Object"
             and str(rt.classOf(n.controller)) == "Vertical_Horizontal_Turn"]
    if len(roots) != 1:
        raise RuntimeError("Expected one Biped; select a node of the target character")
    return roots[0]


def _check_mode(rt, root):
    if rt.theHold.Holding():
        raise RuntimeError("Another Max Undo transaction is active")
    for prop in ("figureMode", "footstepMode", "motionMode", "mixerMode", "moveAllMode",
                 "bufferMode", "rubberBandMode"):
        if bool(getattr(root.controller, prop)):
            raise RuntimeError("Disable Biped " + prop + " before retargeting")
    if int(rt.biped.numLayers(root.controller)) != 0:
        raise RuntimeError("Body retargeting requires the Biped base layer without additional layers")
    for prop, expected in (("spineLinks", 3), ("neckLinks", 1), ("legLinks", 3),
                           ("fingers", 5), ("fingerLinks", 3), ("toes", 1), ("toeLinks", 1)):
        if int(getattr(root.controller, prop)) != expected:
            raise RuntimeError("Unsupported Biped topology: " + prop + " must be " + str(expected))


def _descendants(root):
    result, queue = [], [root]
    while queue:
        n = queue.pop(0)
        result.append(n)
        queue.extend(list(n.children))
    return result


def _driven_handles(rt, owned):
    """Follow actual Max references: this rig also drives a separate export rig."""
    handles, visited = set(), set()
    queue = list(owned)
    while queue:
        source = queue.pop()
        handle = int(rt.getHandleByAnim(source))
        if handle in visited:
            continue
        visited.add(handle)
        for dependent in rt.refs.dependents(source):
            if rt.isValidNode(dependent) and str(rt.classOf(dependent)) != "Biped_Object":
                for n in _descendants(dependent):
                    if str(rt.classOf(n)) != "Biped_Object":
                        handles.add(int(rt.getHandleByAnim(n)))
                        queue.append(n)
    return handles


def inspect_rig(rt, root):
    prefix = str(root.name) + " "
    nodes = {part: hand.node(rt, prefix + part) for part in CHILDREN}
    rig = {}
    pending = dict(nodes)
    while pending:
        progress = False
        for part, n in list(pending.items()):
            if _root(rt, n) != root:
                raise RuntimeError("Bone belongs to another Biped: " + str(n.name))
            parent = str(n.parent.name)[len(prefix):] if n.parent != root else None
            expected_parent = next((p for p, c in CHILDREN.items() if c == part), None)
            if part.endswith("Clavicle"):
                expected_parent = "Spine2"
            elif part.endswith("Thigh"):
                expected_parent = "Pelvis"
            elif " Finger" in part and len(part.split(" ")[1]) == 7:
                expected_parent = part[0] + " Hand"
            if part == "Pelvis":
                expected_parent = None
            if parent != expected_parent:
                raise RuntimeError("Unsupported parent for " + part + ": " + str(parent))
            if parent is not None and parent not in rig:
                continue
            a = hand.axes(n)
            pa = hand.axes(n.parent)
            child = hand.node(rt, prefix + CHILDREN[part]) if CHILDREN[part] else None
            if child is not None and child.parent != n:
                raise RuntimeError("Unexpected Biped child: " + str(child.name))
            offset = m.sub(hand.matrix(child)[3], hand.matrix(n)[3]) if child else m.world((1., 0., 0.), a)
            rig[part] = {"parent": parent, "axes": a, "local_axes": maths.relative(a, pa),
                         "position": hand.matrix(n)[3], "aim": m.local(offset, a),
                         "length": maths.length(offset), "child": child,
                         "size": hand.vector(rt.biped.getTransform(n, rt.Name("scale")))}
            del pending[part]
            progress = True
        if not progress:
            raise RuntimeError("Unsupported Biped parent hierarchy: " + ", ".join(pending))
    return nodes, rig


def _controllers(rt, n):
    controllers = [("node", n.controller)]
    if str(rt.classOf(n.controller)) == "Vertical_Horizontal_Turn":
        controllers.extend((p, getattr(n.controller, p).controller) for p in ("vertical", "horizontal", "turning"))
    return controllers


def _keys(rt, n):
    result = {}
    for name, controller in _controllers(rt, n):
        try:
            count = int(rt.numKeys(controller))
            result[name] = tuple(str(rt.getKeyTime(controller, i)) for i in range(1, count+1)) if count >= 0 else None
        except Exception:
            result[name] = None
    return result


def _key_settings(rt, n, excluded_time):
    result = {}
    for track, controller in _controllers(rt, n):
        try:
            count = int(rt.numKeys(controller))
        except Exception:
            continue
        for i in range(1, count+1):
            key_time = rt.getKeyTime(controller, i)
            if abs(float(key_time) - excluded_time) < 1e-8:
                continue
            key = rt.biped.getKey(controller, i)
            result[(track, str(key_time))] = {
                str(p): str(rt.getProperty(key, p)) for p in rt.getPropNames(key)
                # ikPivots is a derived, pose-evaluated geometry array, not a
                # stored key value; preserve the actual IK pivot index instead.
                if str(p) not in ("selected", "ikPivots")
            }
    return result


def _current_key(rt, node, time):
    controller = node.controller
    index = next(i for i in range(1, int(rt.numKeys(controller))+1)
                 if abs(float(rt.getKeyTime(controller, i))-time) < 1e-8)
    return rt.biped.getKey(controller, index)


def _set_free(key):
    key.ikBlend = 0.
    key.ikSpace = 0
    key.ikJoinedPivot = False


class BodyRetargeter:
    def __init__(self):
        self.callback_id = "BodyRetarget_" + uuid.uuid4().hex
        self.restore_token = None
        self.history_serial = 0
        self.scene_serial = 0
        self._callbacks = []

    def _history_changed(self):
        self.history_serial += 1
        self.restore_token = None

    def _scene_changed(self):
        _REFERENCES.clear()
        self.scene_serial += 1
        self._history_changed()

    def _install_callbacks(self, rt):
        if self._callbacks:
            return
        for event, callback in (("sceneUndo", self._history_changed), ("sceneRedo", self._history_changed),
                                ("systemPreReset", self._scene_changed), ("filePreOpen", self._scene_changed)):
            rt.callbacks.addScript(rt.Name(event), callback, id=rt.Name(self.callback_id))
            self._callbacks.append(callback)

    def close(self):
        if self._callbacks:
            rt = hand.runtime()
            rt.callbacks.removeScripts(id=rt.Name(self.callback_id))
            self._callbacks = []
        self.restore_token = None

    def capture_request(self, target=None, time=None):
        rt = hand.runtime()
        self._install_callbacks(rt)
        root = resolve_target(rt, target)
        _check_mode(rt, root)
        nodes, _ = inspect_rig(rt, root)
        return {"target": root, "handle": int(rt.getHandleByAnim(root)),
                "time": float(rt.sliderTime if time is None else time), "scene_serial": self.scene_serial,
                "bones": {part: int(rt.getHandleByAnim(n)) for part, n in nodes.items()},
                "file": str(rt.maxFilePath) + str(rt.maxFileName)}

    def validate_request(self, request):
        rt = hand.runtime()
        if (request["scene_serial"] != self.scene_serial
                or request["file"] != str(rt.maxFilePath) + str(rt.maxFileName)
                or not rt.isValidNode(request["target"])
                or int(rt.getHandleByAnim(request["target"])) != request["handle"]):
            raise RuntimeError("Scene or target changed while waiting for pose inference")
        nodes, _ = inspect_rig(rt, request["target"])
        if request["bones"] != {part: int(rt.getHandleByAnim(n)) for part, n in nodes.items()}:
            raise RuntimeError("Biped bones changed while waiting for pose inference")
        return request["target"]

    def _stamp(self, rt):
        return int(rt.theHold.GetGlobalPutCount()), self.history_serial

    def can_restore(self):
        if self.restore_token is None:
            return False
        rt = hand.runtime()
        try:
            self.validate_request(self._last_request)
            return not rt.theHold.Holding() and self._stamp(rt) == self.restore_token
        except Exception:
            return False

    def restore(self):
        if not self.can_restore():
            raise RuntimeError("Restore unavailable after another scene edit or Undo")
        rt = hand.runtime()
        self.restore_token = None
        rt.execute("max undo")
        rt.redrawViews()

    def apply(self, frame, target=None, time=None, keep_source_bones=False):
        """Retarget, optionally retaining a new Bone hierarchy in the same Undo."""
        rt = hand.runtime()
        client._validate_frame(frame, mode="body")
        names = client._body_frame_joint_names(frame)
        points, _ = client._position_map(frame, rt, joint_names=names)
        face_points = client._face_position_map(frame, rt)
        return self._apply_points(points, target, time,
                                  source_frame=frame if keep_source_bones else None,
                                  face_points=face_points)

    def _apply_points(self, points, target, time, source_frame=None,
                      face_points=None):
        rt = hand.runtime()
        from pymxs import attime
        request = self.capture_request(target, time)
        root = request["target"]
        points = maths.validate_points(points, client.BODY_JOINTS)
        with attime(request["time"]):
            nodes, rig = inspect_rig(rt, root)
            cache_key = (request["file"], request["handle"], request["time"])
            signature = {part: (int(rt.getHandleByAnim(n)), int(rt.getHandleByAnim(n.parent)),
                                rig[part]["size"]) for part, n in nodes.items()}
            current = {part: hand.matrix(n) for part, n in nodes.items()}
            cached = _REFERENCES.get(cache_key)
            reference = rig
            if cached and cached["signature"] == signature:
                error = max(abs(x-y) for part in nodes for row, old in zip(current[part], cached["output"][part]) for x,y in zip(row,old))
                if error < 1e-4:
                    reference = cached["rig"]
            solution = maths.solve(points, reference, face_points=face_points)
            owned = _descendants(root)
            owned_handles = {int(rt.getHandleByAnim(n)) for n in owned}
            driven_handles = _driven_handles(rt, owned)
            protected = [(n, hand.matrix(n), _keys(rt, n)) for n in rt.objects
                         if int(rt.getHandleByAnim(n)) not in owned_handles]
            before_keys = [(n, _keys(rt, n)) for n in owned]
            before_settings = [(n, _key_settings(rt, n, request["time"])) for n in owned
                               if str(rt.classOf(n)) == "Biped_Object" and not str(n.name).endswith("Footsteps")]
            flags = {prop: bool(getattr(root.controller, prop)) for prop in TRACK_FLAGS}
            pelvis_position = rig["Pelvis"]["position"]
            height = max(r["position"][2] for r in rig.values()) - min(r["position"][2] for r in rig.values())
            tolerance = max(1e-5, height * 1e-6)
            self.restore_token = None
            source_result = None
            rt.theHold.Begin()
            try:
                for prop in TRACK_FLAGS:
                    setattr(root.controller, prop, True)
                for part in EFFECTORS:
                    # setFreeKey follows Max's interactive time/selection state.
                    # Explicit keys remain bound to the inference request time.
                    key = rt.biped.addNewKey(nodes[part].controller, request["time"])
                    _set_free(key)
                for part, rows in solution["rotations"].items():
                    tm = rt.Matrix3(*(rt.Point3(*r) for r in rows), rt.Point3(0, 0, 0))
                    rt.biped.setTransform(nodes[part], rt.Name("rotation"), tm.rotation, True)
                for part in solution["preserved"]:
                    if part not in nodes:
                        continue
                    rows = maths.compose(reference[part]["local_axes"], hand.axes(nodes[part].parent))
                    tm = rt.Matrix3(*(rt.Point3(*r) for r in rows), rt.Point3(0, 0, 0))
                    rt.biped.setTransform(nodes[part], rt.Name("rotation"), tm.rotation, True)
                # setTransform may copy the surrounding IK blend onto its key.
                # Commit the free state after the entire chain has been keyed.
                for part in EFFECTORS:
                    _set_free(_current_key(rt, nodes[part], request["time"]))
                # Pelvis orientation is controlled through COM; compensate only
                # a potential root offset, never change its requested world position.
                delta = m.sub(pelvis_position, hand.matrix(nodes["Pelvis"])[3])
                if maths.length(delta) > tolerance:
                    pos = tuple(a+b for a, b in zip(hand.matrix(root)[3], delta))
                    rt.biped.setTransform(root, rt.Name("pos"), rt.Point3(*pos), True)
                errors = {}
                self.last_diagnostics = {}
                for part, ref in rig.items():
                    n = nodes[part]
                    self.last_diagnostics[part] = {"before": ref["position"], "after": hand.matrix(n)[3],
                                                   "axes": hand.axes(n), "desired": solution["rotations"][part],
                                                   "size_before": ref["size"], "size_after": hand.vector(rt.biped.getTransform(n, rt.Name("scale")))}
                    if ref["child"] is not None:
                        v = m.sub(hand.matrix(ref["child"])[3], hand.matrix(n)[3])
                        self.last_diagnostics[part].update(length_before=ref["length"], length_after=maths.length(v),
                                                          local_before=ref["aim"], local_after=m.local(v, hand.axes(n)))
                        if part in solution["directions"]:
                            self.last_diagnostics[part]["angle"] = maths.angle(v, solution["directions"][part])
                for part, ref in rig.items():
                    size = hand.vector(rt.biped.getTransform(nodes[part], rt.Name("scale")))
                    if any(abs(a-b) > max(1e-5, abs(b)*1e-4) for a, b in zip(size, ref["size"])):
                        raise RuntimeError("Biped figure size changed: " + part)
                    if ref["child"] is None:
                        continue
                    v = m.sub(hand.matrix(ref["child"])[3], hand.matrix(nodes[part])[3])
                    # Biped spine/neck attachment offsets are pose-dependent;
                    # their figure sizes above, not pivot spacing, define length.
                    if part not in ("Pelvis", "Spine", "Spine1", "Spine2", "Neck") and abs(maths.length(v) - ref["length"]) > max(1e-5, ref["length"] * 1e-4):
                        raise RuntimeError("Biped bone length changed: " + part + " " + str((ref["length"], maths.length(v))))
                    if part in solution["directions"]:
                        errors[part] = maths.angle(v, solution["directions"][part])
                if max(errors.values(), default=0) > .5:
                    raise RuntimeError("Biped direction mismatch: " + str({n: round(e, 3) for n, e in errors.items() if e > .5}))
                pelvis_error = maths.length(m.sub(pelvis_position, hand.matrix(nodes["Pelvis"])[3]))
                if pelvis_error > tolerance:
                    raise RuntimeError("Biped pelvis position changed")
                for part in EFFECTORS:
                    key = _current_key(rt, nodes[part], request["time"])
                    if float(key.ikBlend) != 0. or int(key.ikSpace) != 0 or bool(key.ikJoinedPivot):
                        raise RuntimeError("Biped did not retain a free key: " + part)
                preserved_errors = {}
                for part in solution["preserved"]:
                    if part in nodes:
                        local_axes = maths.relative(hand.axes(nodes[part]), hand.axes(nodes[part].parent))
                        preserved_errors[part] = max(maths.angle(row, old) for row, old in zip(local_axes, reference[part]["local_axes"]))
                if max(preserved_errors.values(), default=0) > .05:
                    raise RuntimeError("Biped changed a preserved local rotation: " + str(preserved_errors))
                head_rotation_error = None
                if solution["head_rotation_estimated"]:
                    head_rotation_error = max(maths.angle(row, desired) for row, desired in zip(
                        hand.axes(nodes["Head"]), solution["rotations"]["Head"]))
                    if head_rotation_error > .5:
                        raise RuntimeError("Biped head rotation mismatch")
                for n, previous, keys in protected:
                    pose_changed = max(abs(a-b) for row, old in zip(hand.matrix(n), previous) for a, b in zip(row, old)) > 1e-5
                    if (pose_changed and int(rt.getHandleByAnim(n)) not in driven_handles) or _keys(rt, n) != keys:
                        raise RuntimeError("Non-target scene node changed: " + str(n.name))
                write_time = str(rt.currentTime)
                for n, keys in before_keys:
                    after = _keys(rt, n)
                    for track, times in keys.items():
                        if times is not None and after[track] is not None:
                            if set(after[track]) - set(times) - {write_time} or set(times) - set(after[track]):
                                raise RuntimeError("Unexpected key times changed: " + str(n.name))
                for n, settings in before_settings:
                    if _key_settings(rt, n, request["time"]) != settings:
                        raise RuntimeError("Non-current Biped key settings changed: " + str(n.name))
                if source_frame is not None:
                    # Replace all prior body previews, including numbered copies
                    # and fingers absent from the new frame, within this Undo.
                    client._delete_previous(
                        rt, joint_names=client.BODY_AND_FINGER_JOINTS, strict=True
                    )
                    source = client.create_max_skeleton(
                        source_frame, runtime=rt, prefix=client.BODY_NAME_PREFIX,
                        replace_previous=False,
                    )
                    source_result = {"root": str(source["root"].name),
                                     "joint_count": source["joint_count"],
                                     "name_prefix": source["name_prefix"]}
                rt.theHold.Accept("MediaPipe Body " + uuid.uuid4().hex[:8])
            except Exception:
                rt.theHold.Cancel()
                for prop, value in flags.items():
                    if bool(getattr(root.controller, prop)) != value:
                        setattr(root.controller, prop, value)
                rt.redrawViews()
                raise
            self._last_request = request
            self.restore_token = self._stamp(rt)
            if len(_REFERENCES) >= 64 and cache_key not in _REFERENCES:
                del _REFERENCES[next(iter(_REFERENCES))]
            _REFERENCES[cache_key] = {"signature": signature, "rig": reference,
                                      "output": {part: hand.matrix(n) for part, n in nodes.items()}}
        rt.redrawViews()
        return {"target": str(root.name), "frame": write_time, "written": len(solution["rotations"]),
                "source_skeleton": source_result,
                "max_direction_error_degrees": max(errors.values(), default=0), "direction_errors": errors,
                "pelvis_position_error": pelvis_error, "preserved": solution["preserved"],
                "max_preserved_rotation_error_degrees": max(preserved_errors.values(), default=0),
                "head_rotation_estimated": solution["head_rotation_estimated"],
                "head_rotation_error_degrees": head_rotation_error,
                "max_figure_size_relative_error": max(abs(x-y)/max(abs(y), 1e-8) for part in nodes for x,y in zip(self.last_diagnostics[part]["size_after"], rig[part]["size"])),
                "projection_degrees": solution["projection_degrees"]}
