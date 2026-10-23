"""Matched translation and bearing renderer counterfactuals."""
from __future__ import annotations
import math
import numpy as np
from common import cfg, require

def paired_deltas(target_xyz, eef_xyz):
    relative = np.asarray(target_xyz, dtype=float) - np.asarray(eef_xyz, dtype=float)
    radius = float(np.linalg.norm(relative[:2]))
    require(np.isfinite(relative).all() and radius > 1e-9, "Undefined horizontal target-to-EEF relation")
    cap = float(cfg()["max_displacement_m"])
    angle = min(math.radians(float(cfg()["bearing_degrees"])), 2*math.asin(min(1.0, cap/(2*radius))))
    radial = relative[:2] / radius
    tangent = np.array([-radial[1], radial[0]])
    rotated = radius * (math.cos(angle)*radial + math.sin(angle)*tangent)
    bearing_plus = np.r_[rotated-relative[:2], 0.0]
    displacement = float(np.linalg.norm(bearing_plus))
    require(displacement <= cap+1e-12 and displacement > 0, "Invalid capped bearing displacement")
    rotated_minus = radius * (math.cos(-angle)*radial + math.sin(-angle)*tangent)
    output = {
        "translation_plus": np.array([displacement, 0.0, 0.0]),
        "translation_minus": np.array([-displacement, 0.0, 0.0]),
        "bearing_plus": bearing_plus,
        "bearing_minus": np.r_[rotated_minus-relative[:2], 0.0],
    }
    require(all(abs(v[2]) < 1e-12 and abs(np.linalg.norm(v)-displacement) < 1e-9 for v in output.values()),
            "A/B displacement matching or height invariant failed")
    return output, {"horizontal_range_m": radius, "bearing_degrees": math.degrees(angle),
                    "matched_displacement_m": displacement}

def descendant_bodies(model, root):
    roots = {int(root)}
    changed = True
    while changed:
        changed = False
        for body in range(int(model.nbody)):
            if int(model.body_parentid[body]) in roots and body not in roots:
                roots.add(body); changed = True
    return roots

def resolve_entity(env, name, unary=False):
    inner, model = env.env, env.sim.model
    state = inner.object_states_dict[name]
    parent = getattr(state, "parent_name", name)
    obj = inner.get_object(parent)
    joint_names = []
    is_site = getattr(state, "object_state_type", None) == "site"
    if is_site:
        site = inner.object_sites_dict[name]
        joint_names = list(getattr(site, "joints", ()) or ())
        if not joint_names:
            raise ValueError("Non-renderable static region: " + name)
    elif unary:
        joint_names = list(getattr(obj, "joints", ()) or ())
    if joint_names:
        joint = int(model.joint_name2id(joint_names[0]))
        root = int(model.jnt_bodyid[joint])
    else:
        root_name = getattr(obj, "root_body", None)
        require(root_name is not None, "Object lacks a root body: " + parent)
        root = int(model.body_name2id(root_name))
    bodies = descendant_bodies(model, root)
    geoms = np.flatnonzero(np.isin(np.asarray(model.geom_bodyid), list(bodies))).astype(int).tolist()
    require(geoms, "Entity has no renderable geometry: " + name)
    position = np.asarray(state.get_geom_state()["pos"], dtype=float).reshape(3)
    anchor = "site" if is_site else ("joint_body" if joint_names else "object_body")
    return {"name": name, "parent": parent, "root_body_id": root, "geom_ids": geoms,
            "position": position, "anchor":anchor}

def task_candidates(env):
    goals = env.env.parsed_problem["goal_state"]
    require(goals, "Task has no goal predicates")
    ordered = []
    for goal in goals:
        unary = len(goal) == 2
        for name in goal[1:]:
            if name not in ordered:
                ordered.append(name)
    resolved, failures = [], []
    for name in ordered:
        try: resolved.append(resolve_entity(env, name, unary=any(len(g)==2 and g[1]==name for g in goals)))
        except Exception as error: failures.append({"name": name, "error": repr(error)})
    require(resolved, "No renderable task-relevant entity; failures=" + repr(failures))
    return resolved, failures

class SceneTranslation:
    def __init__(self, scene, geom_ids, delta):
        import mujoco
        self.module, self.scene = mujoco, scene
        self.ids, self.delta, self.calls = set(map(int, geom_ids)), np.asarray(delta, dtype=float), 0
    def __enter__(self):
        self.original = self.module.mjr_render
        def render(*args, **kwargs):
            scene = kwargs.get("scn") if "scn" in kwargs else (args[1] if len(args)>1 else None)
            if scene is not self.scene: return self.original(*args, **kwargs)
            saved = []
            try:
                for index in range(scene.ngeom):
                    geom = scene.geoms[index]
                    if int(geom.objtype) == 5 and int(geom.objid) in self.ids:
                        saved.append((geom, np.array(geom.pos, copy=True))); geom.pos[:] += self.delta
                require(saved, "Selected geoms did not enter the owned render scene")
                self.calls += 1
                return self.original(*args, **kwargs)
            finally:
                for geom, value in saved: geom.pos[:] = value
        self.module.mjr_render = render
        return self
    def __exit__(self, *_):
        self.module.mjr_render = self.original

def render(env, entity, delta, resolution=256):
    from robosuite import macros
    scene = env.sim._render_context_offscreen.scn
    with SceneTranslation(scene, entity["geom_ids"], delta) as hook:
        rgb = env.sim.render(camera_name="robot0_eye_in_hand", width=resolution, height=resolution)
        seg = env.sim.render(camera_name="robot0_eye_in_hand", width=resolution, height=resolution, segmentation=True)
    require(hook.calls == 2, "Expected RGB and segmentation render calls")
    convention = 1 if macros.IMAGE_CONVENTION == "opengl" else -1
    rgb = np.asarray(rgb)[::convention][::-1, ::-1].copy()
    seg = np.asarray(seg)[::convention][::-1, ::-1].copy()
    mask = (seg[:, :, 0] == 5) & np.isin(seg[:, :, 1], entity["geom_ids"])
    return rgb, mask
