"""Transactional in-place branch rollouts, including controllers and solver warm starts."""
from __future__ import annotations
import copy
import hashlib
import numpy as np
from loop_core import cfg, load_snapshot_vendor, require

VENDOR = load_snapshot_vendor()
DATA_FIELDS = ("qacc_warmstart", "qfrc_applied", "xfrc_applied", "ctrl", "act", "mocap_pos", "mocap_quat", "userdata")
MODEL_FIELDS = ("site_rgba", "geom_rgba", "light_active")
CONTROLLER_CACHE = ("ee_pos", "ee_ori_mat", "ee_pos_vel", "ee_ori_vel", "joint_pos", "joint_vel",
                    "J_pos", "J_ori", "J_full", "mass_matrix", "base_pos", "base_ori")


def arrays(root, names):
    return {name: np.asarray(getattr(root, name)).copy() for name in names if hasattr(root, name)}


class BranchState:
    def __init__(self, env, raw):
        import torch
        self.base = VENDOR.capture_snapshot(env)
        self.raw = copy.deepcopy(raw)
        inner = getattr(env, "env", env)
        self.cache = copy.deepcopy(getattr(inner, "_obs_cache", {}))
        self.extra_data = arrays(env.sim.data, DATA_FIELDS)
        self.extra_model = arrays(env.sim.model, MODEL_FIELDS)
        self.properties = {}
        for name, obj in {**getattr(inner, "objects_dict", {}), **getattr(inner, "fixtures_dict", {})}.items():
            if hasattr(obj, "object_properties"):
                self.properties[name] = copy.deepcopy(obj.object_properties)
        self.robot = []
        for robot in inner.robots:
            controller = robot.controller
            cache = {k: copy.deepcopy(getattr(controller, k)) for k in CONTROLLER_CACHE if hasattr(controller, k)}
            gripper = copy.deepcopy(getattr(robot.gripper, "current_action", None))
            self.robot.append((cache, gripper))
        self.torch_rng = torch.get_rng_state()
        self.cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        self.identity = hashlib.sha256(np.asarray(self.base["sim_state"]).tobytes()).hexdigest()

    def restore(self, env):
        import torch
        # Restore visuals before forward/render, including articulated stove sites.
        for key, value in self.extra_model.items():
            getattr(env.sim.model, key)[...] = value
        VENDOR.restore_snapshot(env, self.base)
        for key, value in self.extra_data.items():
            getattr(env.sim.data, key)[...] = value
        inner = getattr(env, "env", env)
        for name, properties in self.properties.items():
            inner.get_object(name).object_properties = copy.deepcopy(properties)
        inner._obs_cache = copy.deepcopy(self.cache)
        for robot, (cache, gripper) in zip(inner.robots, self.robot):
            for key, value in cache.items():
                setattr(robot.controller, key, copy.deepcopy(value))
            if gripper is not None:
                robot.gripper.current_action = copy.deepcopy(gripper)
        torch.set_rng_state(self.torch_rng)
        if self.cuda_rng is not None:
            torch.cuda.set_rng_state_all(self.cuda_rng)
        error = VENDOR.state_error(env, self.base)
        require(error <= cfg()["restore_state_atol"], f"Candidate rollback changed the real episode: {error}")
        return copy.deepcopy(self.raw), error


def state_vector(env):
    return np.asarray(env.get_sim_state(), dtype=np.float64).copy()


def state_difference(left, right):
    left, right = np.asarray(left), np.asarray(right)
    require(left.shape == right.shape, "Simulator replay shape changed")
    return float(np.abs(left - right).max(initial=0.))
