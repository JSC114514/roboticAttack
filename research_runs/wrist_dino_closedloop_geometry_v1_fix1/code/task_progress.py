"""Frozen short-horizon search potential; complete-episode success remains the endpoint.

Simulator object positions / BDDL predicates are privileged planning information.
They never replace the victim's images, proprioception, language, or actions.
"""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np

SUPPORTED = {"on", "in", "stack", "open", "close", "turnon", "turnoff"}


def bddl_goals(text):
    tokens = iter(re.findall(r"\(|\)|[^\s()]+", re.sub(r";[^\n]*", "", text)))
    def parse(first):
        if first != "(":
            return first
        out = []
        for token in tokens:
            if token == ")":
                return out
            out.append(parse(token))
        raise ValueError("Unclosed BDDL expression")
    tree = parse(next(tokens))
    def find(node):
        if isinstance(node, list):
            if node and node[0] == ":goal":
                goal = node[1]
                return goal[1:] if str(goal[0]).lower() == "and" else [goal]
            for child in node:
                value = find(child)
                if value is not None:
                    return value
        return None
    goals = find(tree)
    if not goals:
        raise ValueError("BDDL lacks a nonempty goal")
    return goals


def audit_bddl(root):
    counts = {}
    files = 0
    for suite in ("libero_object", "libero_spatial", "libero_goal", "libero_10"):
        paths = sorted((Path(root) / suite).glob("*.bddl"))
        if len(paths) != 10:
            raise ValueError(f"Expected ten BDDL tasks in {suite}, got {len(paths)}")
        for path in paths:
            for goal in bddl_goals(path.read_text()):
                name = goal[0].lower()
                if name not in SUPPORTED or len(goal) not in (2, 3):
                    raise ValueError(f"Unsupported goal, refuse a silent sparse fallback: {path}: {goal}")
                counts[name] = counts.get(name, 0) + 1
            files += 1
    return {"bddl_files": files, "predicate_counts": counts}


class TaskProgress:
    def __init__(self, env, distance_scale=0.1):
        self.env = env
        self.inner = getattr(env, "env", env)
        self.goals = self.inner.parsed_problem["goal_state"]
        self.scale = float(distance_scale)
        if not self.goals or any(g[0].lower() not in SUPPORTED or len(g) not in (2, 3) for g in self.goals):
            raise ValueError("Runtime goals do not satisfy the preflight goal contract")

    def position(self, name):
        value = np.asarray(self.inner.object_states_dict[name].get_geom_state()["pos"], dtype=float).copy()
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError("Invalid task object/site position: " + name)
        return value

    def _joint_progress(self, name, predicate):
        state = self.inner.object_states_dict[name]
        obj = self.inner.get_object(getattr(state, "parent_name", name))
        if getattr(state, "object_state_type", None) == "site":
            # Drawer sites refer only to their own articulated joint, not every drawer.
            # Their inherited get_joint_state exists but raises NotImplementedError.
            site = self.inner.object_sites_dict[name]
            values = np.asarray([self.env.sim.data.qpos[self.env.sim.model.get_joint_qpos_addr(j)]
                                 for j in site.joints], dtype=float)
        elif getattr(state, "object_state_type", None) == "object":
            values = np.asarray(state.get_joint_state(), dtype=float).reshape(-1)
        else:
            raise ValueError(f"Unsupported articulated state type for {name}: {type(state).__name__}")
        properties = getattr(obj, "object_properties", {}).get("articulation", {})
        key_token = {"open": "open", "close": "close", "turnon": "turnon", "turnoff": "turnoff"}[predicate]
        matches = [v for k, v in properties.items() if key_token in k.lower().replace("_", "") and "range" in k.lower()]
        if not matches or values.size == 0:
            raise ValueError(f"No declared joint target range for {predicate} {name}; no inferred joint sign")
        low, high = sorted(map(float, matches[0]))
        span = max(high - low, 0.01)
        distance = np.maximum(np.maximum(low - values, values - high), 0.)
        return float(np.exp(-distance / span).mean()), values.tolist(), [low, high]

    def measure(self):
        robot = self.inner.robots[0]
        # Native OSC end-effector site; this is simulator truth, not an image estimate.
        eef = np.asarray(self.env.sim.data.site_xpos[robot.eef_site_id], dtype=float).copy()
        details, predicates, dense = [], [], []
        for goal in self.goals:
            kind, name = goal[0].lower(), goal[1]
            satisfied = bool(self.inner._eval_predicate(goal))
            source = self.position(name)
            reach = float(np.exp(-np.linalg.norm(eef - source) / self.scale))
            detail = {"goal": list(goal), "satisfied": satisfied, "object_pos": source.tolist(),
                      "eef_distance_m": float(np.linalg.norm(eef - source)), "reach": reach}
            if len(goal) == 3:
                target = self.position(goal[2])
                distance = float(np.linalg.norm(source - target))
                relation = float(np.exp(-distance / self.scale))
                obj = self.inner.get_object(name)
                held = bool(self.inner._check_grasp(gripper=robot.gripper, object_geoms=obj.contact_geoms))
                value = 0.5 * held + 0.25 * reach + 0.25 * relation
                detail.update(target_pos=target.tolist(), target_distance_m=distance, held=held, relation=relation)
            else:
                joint, values, target_range = self._joint_progress(name, kind)
                value = 0.5 * joint + 0.5 * reach
                detail.update(joint_progress=joint, joints=values, joint_target_range=target_range)
            predicates.append(int(satisfied)); dense.append(value); details.append(detail)
        result = {"goal_fraction": float(np.mean(predicates)), "dense_progress": float(np.mean(dense)),
                  "goal_bits": predicates, "details": details, "eef": eef.tolist()}
        if not np.isfinite([result["goal_fraction"], result["dense_progress"]]).all():
            raise ValueError("Nonfinite planning progress")
        return result


def score_candidates(candidates, condition):
    """Both conditions receive the same fully simulated finite candidate set."""
    if condition == "uada_geometry":
        return min(range(len(candidates)), key=lambda i: (candidates[i]["uada_mse"], i))
    if condition != "loop_geometry":
        raise ValueError("No adaptive selector for " + condition)
    return min(range(len(candidates)), key=lambda i: (
        int(candidates[i]["success"]), candidates[i]["goal_fraction_mean"],
        candidates[i]["dense_progress_mean"], candidates[i]["uada_mse"], i))
