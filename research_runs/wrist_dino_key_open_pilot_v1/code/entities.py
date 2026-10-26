"""Simulation-only labels and observable contact proxies; never used at deployment."""
from __future__ import annotations

import numpy as np

from common import require


def inner_env(env):
    return getattr(env, "env", env)


def goal_objects(env, strict=False):
    inner = inner_env(env)
    names = sorted({str(goal[1]) for goal in inner.parsed_problem["goal_state"]
                    if len(goal) >= 3 and str(goal[0]).lower() in {"in", "on"}
                    and str(goal[1]) in inner.objects_dict})
    if strict:
        require(len(names) == 1, f"Object source requires one manipulated goal subject: {names}")
    return [(name, inner.objects_dict[name]) for name in names]


def geom_ids(sim, names):
    return {int(sim.model.geom_name2id(name)) for name in names}


def geoms(model):
    return list(set(list(model.contact_geoms) + list(model.visual_geoms)))


def grippers(robot):
    value = robot.gripper
    return list(value.values()) if isinstance(value, dict) else [value]


def goal_region_entities(env):
    """All manipulation subjects, including unary articulated-fixture goals.

    Region names resolve to the longest catalogued parent. This is the whole
    fixture geometry, not a claim to segment only its active joint/door.
    """
    inner = inner_env(env)
    catalog = {**getattr(inner,"fixtures_dict",{}), **inner.objects_dict}
    names = set()
    def visit(goal):
        require(isinstance(goal,(tuple,list)) and goal, "Malformed goal predicate")
        if str(goal[0]).lower() in {"and","or"}:
            for child in goal[1:]:
                visit(child)
            return
        require(str(goal[0]).lower() in {"in","on","close","open","turnon","turnoff"}
                and len(goal)>=2, f"Unsupported Long goal predicate: {goal}")
        subject = str(goal[1])
        parents = [name for name in catalog if subject==name or subject.startswith(name+"_")]
        require(parents, f"Long goal subject has no object/fixture parent: {subject}")
        names.add(max(parents,key=len))
    for goal in inner.parsed_problem["goal_state"]:
        visit(goal)
    require(names, "No Long goal manipulation entities")
    return [(name,catalog[name]) for name in sorted(names)]


def segmentation_mapping(env):
    inner = inner_env(env)
    pairs = goal_region_entities(env)
    target = set().union(*(geom_ids(env.sim,geoms(obj)) for _,obj in pairs))
    grip, robot_ids = set(), set()
    for robot in inner.robots:
        robot_ids |= geom_ids(env.sim, geoms(robot.robot_model))
        for gripper in grippers(robot):
            ids = geom_ids(env.sim, geoms(gripper))
            grip |= ids
            robot_ids |= ids
    require(target and grip and robot_ids and not target & robot_ids, "Invalid target/robot/gripper geometry map")
    return {"target_names": [name for name,_ in pairs], "target_ids": target,
            "gripper_ids": grip, "robot_ids": robot_ids,
            "region_kind":"whole_goal_subject_objects_and_fixtures_plus_gripper"}


def pixel_masks(raw, mapping):
    keys = [key for key in raw if key.startswith("robot0_eye_in_hand") and key.endswith("segmentation_element")]
    require(len(keys) == 1, f"Expected actual Wrist geom-ID segmentation: {keys}")
    ids = np.asarray(raw[keys[0]]).squeeze()[::-1, ::-1].astype(np.int32)
    require(ids.shape == (256, 256), "Geom segmentation shape changed")
    target = np.isin(ids, list(mapping["target_ids"]))
    gripper = np.isin(ids, list(mapping["gripper_ids"]))
    robot = np.isin(ids, list(mapping["robot_ids"]))
    require(not np.any(target & robot), "Target/robot segmentation overlap")
    # -1 background sky and all nonrobot, nontarget geometry are background.
    return {"target": target.astype(np.uint8), "gripper": gripper.astype(np.uint8),
            "robot": robot.astype(np.uint8), "background": (~(target | robot)).astype(np.uint8)}


def contact_objects(env):
    """Build per-object measurement mappings, or return no applicable proxy."""
    inner = inner_env(env)
    pairs = goal_objects(env)
    if not pairs or not inner.robots:
        return []
    grip = grippers(inner.robots[0])[0]
    important = grip.important_geoms
    if not important.get("left_finger") or not important.get("right_finger"):
        return []
    left, right = [geom_ids(env.sim, important[key]) for key in ("left_finger", "right_finger")]
    return [{"name": name, "object_ids": geom_ids(env.sim, obj.contact_geoms),
             "left_ids": left, "right_ids": right,
             "body_id": int(env.sim.model.body_name2id(obj.root_body))} for name, obj in pairs]


def contact_reading(env, item):
    contacts = [{int(env.sim.data.contact[i].geom1), int(env.sim.data.contact[i].geom2)}
                for i in range(int(env.sim.data.ncon))]
    hit = lambda fingers: any(pair & fingers and pair & item["object_ids"] for pair in contacts)
    grasp = hit(item["left_ids"]) and hit(item["right_ids"])
    return bool(grasp), float(env.sim.data.body_xpos[item["body_id"], 2])
