# 发布阅读说明：保留目标作者原生预处理/动作转换。Adapter重规划8步，π0.5重规划5步；两者分别使用自身checkpoint和统计。
# 本文件为原运行源码的注释副本；训练目标、更新规则和原生前向的可执行AST未改变。
"""Target-native adapters. No GPU/model imports until an explicitly allocated job loads a policy."""
from __future__ import annotations
import importlib
import os
from pathlib import Path
import sys
from contextlib import contextmanager
import math
import numpy as np
from contracts import action_chunk, load_json, paired_noise, pi05_payload, require, validate_observation

def quat2axisangle(value):
    quaternion = np.asarray(value, dtype=np.float64).copy()
    quaternion[3] = np.clip(quaternion[3], -1., 1.)
    denominator = np.sqrt(1. - quaternion[3] ** 2)
    if math.isclose(float(denominator), 0.):
        return np.zeros(3)
    return quaternion[:3] * 2 * math.acos(float(quaternion[3])) / denominator

def prepare_pi05_observation(raw, resize_image):
    # resize_image is the unchanged official openpi_client resize_with_pad+uint8.
    result = {"full_image": resize_image(np.ascontiguousarray(raw["agentview_image"][::-1, ::-1])),
              "wrist_image": resize_image(np.ascontiguousarray(raw["robot0_eye_in_hand_image"][::-1, ::-1])),
              "state": np.concatenate((raw["robot0_eef_pos"], quat2axisangle(raw["robot0_eef_quat"]),
                                       raw["robot0_gripper_qpos"])).astype(np.float32)}
    validate_observation(result)
    return result

def activate_author_repo(repo, roots):
    repo = Path(repo).resolve()
    require(repo.is_dir(), f"Missing author runtime: {repo}")
    for prefix in roots:
        loaded = sys.modules.get(prefix)
        if loaded is not None:
            filename = getattr(loaded, "__file__", None)
            require(filename is not None and Path(filename).resolve().is_relative_to(repo),
                    f"{prefix} already loaded from another model; use one target family per process")
    sys.path.insert(0, str(repo))

def allocated_gpu():
    require(os.environ.get("SLURM_JOB_ID"), "Target model inference requires a user-submitted GPU job")

@contextmanager
def readonly_adapter_loader(utils, repo):
    # Official loader calls disk auto-sync. Replace only these write helpers in
    # this process with identity checks; do NOT change model or inference logic.
    def check_auto(path):
        observed = load_json(Path(path) / "config.json")["auto_map"]
        require(observed.get("AutoConfig") == "configuration_prismatic.OpenVLAConfig"
                and observed.get("AutoModelForVision2Seq") == "modeling_prismatic.OpenVLAForActionPrediction",
                "Adapter checkpoint auto_map mismatch; refusing a disk rewrite")
    def check_logic(path):
        for name in ("modeling_prismatic.py", "configuration_prismatic.py"):
            actual, expected = Path(path) / name, Path(repo) / "prismatic/extern/hf" / name
            require(actual.is_file() and expected.is_file() and actual.read_bytes() == expected.read_bytes(),
                    f"Author/checkpoint logic mismatch: {name}; match correct author version before GPU evaluation")
    originals = utils.update_auto_map, utils.check_model_logic_mismatch
    utils.update_auto_map, utils.check_model_logic_mismatch = check_auto, check_logic
    try:
        yield
    finally:
        utils.update_auto_map, utils.check_model_logic_mismatch = originals

class VLAAdapterPolicy:
    def __init__(self, official, cfg, components):
        self.official, self.cfg = official, cfg
        self.model, self.head, self.proprio, self.noisy, self.processor = components
        self.replan_steps, self.action_horizon = 8, 8

    @classmethod
    def load(cls, spec, suite):
        allocated_gpu()
        activate_author_repo(spec["repo"], ("prismatic", "experiments"))
        import torch
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Exactly one allocated GPU required")
        require(any(k in torch.cuda.get_device_name(0).lower() for k in ("4090", "3090", "a800")), "Unsupported allocated GPU")
        import tensorflow as tf
        from native_compat import configure_adapter_backends
        configure_adapter_backends(torch, tf)
        from experiments.robot.libero import run_libero_eval as official
        from experiments.robot import openvla_utils as utils
        cfg = official.GenerateConfig(pretrained_checkpoint=spec["checkpoints"][suite], task_suite_name=suite,
            use_pro_version=True, use_minivlm=True, use_l1_regression=True, num_images_in_input=2,
            use_proprio=True, center_crop=True, num_open_loop_steps=8, use_wandb=False,
            save_version=spec["checkpoint_inference_version"], phase="Inference")
        require(cfg.save_version == "v1", "Published checkpoint Pro head requires its original v1 branch")
        official.validate_config(cfg)
        official.set_seed_everywhere(cfg.seed)
        with readonly_adapter_loader(utils, spec["repo"]):
            components = official.initialize_model(cfg)
        for module in components[:3]:
            if module is not None:
                module.eval()
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
        require("dino" in str(getattr(components[0].config, "vision_backbone_id", "")).lower(),
                "Expected actual DINO-containing VLA-Adapter checkpoint")
        require(components[0].version == "v1", "Wrong published checkpoint action-head branch")
        return cls(official, cfg, components)

    def runtime_metadata(self):
        import torch
        import tensorflow as tf
        return {"model_backend": "torch_cuda", "cudnn_enabled": torch.backends.cudnn.enabled,
                "tensorflow_gpu_visible": bool(tf.config.get_visible_devices("GPU")),
                "checkpoint_inference_version": self.model.version}

    def prepare(self, raw):
        obs, _ = self.official.prepare_observation(raw, 224)
        validate_observation(obs)
        return obs

    def predict(self, observation, instruction, **unused):
        validate_observation(observation)
        copied = {key: np.asarray(value).copy() for key, value in observation.items()}
        actions = self.official.get_action(self.cfg, self.model, copied, instruction,
            processor=self.processor, action_head=self.head, proprio_projector=self.proprio,
            noisy_action_projector=self.noisy, use_film=self.cfg.use_film, use_minivlm=self.cfg.use_minivlm)
        return action_chunk(actions, self.action_horizon)

    def environment_actions(self, actions):
        return np.asarray([self.official.process_action(a.copy(), self.cfg.model_family) for a in actions], dtype=np.float32)

class Pi05Policy:
    def __init__(self, policy, resize_image, cfg):
        self.policy, self.resize_image, self.cfg = policy, resize_image, cfg
        self.replan_steps = cfg["models"]["pi05"]["replan_steps"]
        self.action_horizon = cfg["models"]["pi05"]["action_horizon"]

    @classmethod
    def load(cls, spec, cfg):
        allocated_gpu()
        os.environ.setdefault("JAX_PLATFORMS", "cuda")
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        os.environ["OPENPI_DATA_HOME"] = spec["data_home"]
        sys.path.insert(0, str(Path(spec["repo"]) / "src"))
        sys.path.insert(0, str(Path(spec["repo"]) / "packages/openpi-client/src"))
        import jax
        require(len(jax.devices()) == 1 and jax.devices()[0].platform == "gpu", "Exactly one JAX GPU required")
        require(any(k in jax.devices()[0].device_kind.lower() for k in ("4090", "3090", "a800")), "Unsupported allocated JAX GPU")
        from openpi.policies import policy_config
        from openpi.training import config as native_config
        from openpi_client import image_tools
        native = native_config.get_config(spec["config_name"])
        require(native.model.pi05 and native.model.action_horizon == spec["action_horizon"]
                and native.model.action_dim == spec["model_action_dim"], "Native pi05 sampling shape changed")
        policy = policy_config.create_trained_policy(native, Path(spec["checkpoint"]))
        resize = lambda rgb: image_tools.convert_to_uint8(image_tools.resize_with_pad(rgb, 224, 224))
        return cls(policy, resize, cfg)

    def prepare(self, raw):
        return prepare_pi05_observation(raw, self.resize_image)

    def predict(self, observation, instruction, *, task_id, state_index, decision, suite):
        _, noise = paired_noise(task_id, state_index, decision, suite, self.cfg)
        output = self.policy.infer(pi05_payload(observation, instruction), noise=noise.copy())
        return action_chunk(output["actions"], self.action_horizon)

    def environment_actions(self, actions):
        # LiberoOutputs already provides environment-unit actions; no OFT flip.
        return action_chunk(actions, self.action_horizon)

def load_policy(name, suite, cfg):
    require(cfg.get("selection_frozen") and cfg.get("patches"), "Load a user-confirmed frozen patch selection first")
    require(name in cfg["models"], "Unknown target model")
    if name == "vla_adapter_pro":
        return VLAAdapterPolicy.load(cfg["models"][name], suite)
    return Pi05Policy.load(cfg["models"][name], cfg)
