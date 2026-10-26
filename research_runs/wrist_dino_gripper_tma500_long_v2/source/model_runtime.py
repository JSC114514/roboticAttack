from __future__ import annotations

import math
import os
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np
import torch

from common import require


def add_paths(cfg: dict[str, Any]) -> None:
    for key in ("robotic_attack_root", "attribution_root", "historical_root", "openvla_oft_root", "libero_root"):
        path = str(cfg["paths"][key])
        sys.path[:] = [entry for entry in sys.path if entry != path]
        sys.path.insert(0, path)
    dgwga = str(Path(cfg["paths"]["historical_root"]) / "dgwga_v1")
    sys.path[:] = [entry for entry in sys.path if entry != dgwga]
    sys.path.insert(0, dgwga)


def quat2axisangle(quaternion: Any) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64).copy()
    value[3] = np.clip(value[3], -1.0, 1.0)
    denominator = np.sqrt(max(0.0, 1.0 - value[3] ** 2))
    if math.isclose(float(denominator), 0.0):
        return np.zeros(3, dtype=np.float32)
    return ((value[:3] * 2.0 * math.acos(float(value[3]))) / denominator).astype(np.float32)


def observation(raw: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    from VLAAttacker.multiview.openvla_oft_experiments import resize_oft_observation_images

    obs = {
        "full_image": np.ascontiguousarray(raw["agentview_image"][::-1, ::-1]).astype(np.uint8),
        "wrist_image": np.ascontiguousarray(raw["robot0_eye_in_hand_image"][::-1, ::-1]).astype(np.uint8),
        "state": np.concatenate((
            np.asarray(raw["robot0_eef_pos"], dtype=np.float32),
            quat2axisangle(raw["robot0_eef_quat"]),
            np.asarray(raw["robot0_gripper_qpos"], dtype=np.float32),
        )),
    }
    return resize_oft_observation_images(
        obs, input_views=["front", "wrist"], resize_size=int(cfg["policy_image_resolution"])
    )


def load_wrapper(cfg: dict[str, Any], suite: str):
    # The A800 compute image occasionally cannot create a cuDNN handle after
    # the OFT checkpoint and EGL context are loaded.  Only the vision patch
    # embedding convolutions use this backend; PyTorch's CUDA convolution is
    # differentiable and avoids the cluster-specific initialization failure.
    # Keep the setting identical for anchor extraction, patch optimization,
    # readout, and rollout evaluation.
    torch.backends.cudnn.enabled = False
    add_paths(cfg)
    from VLAAttacker.multiview.openvla_oft_wrapper import OpenVLAOFTWrapper

    wrapper = OpenVLAOFTWrapper({
        "model_checkpoint": cfg["checkpoints"][suite],
        "task_suite": suite,
        "unnorm_key": cfg["unnorm_keys"][suite],
        "input_views": "front,wrist",
        "num_images_in_input": 2,
        "use_proprio": True,
        "use_l1_regression": True,
        "num_open_loop_steps": int(cfg["num_open_loop_steps"]),
        "device": "cuda",
        "dtype": cfg["dtype"],
        "allow_cpu": False,
        "hf_local_files_only": True,
        "openvla_oft_root": cfg["paths"]["openvla_oft_root"],
        "libero_root": cfg["paths"]["libero_root"],
    })
    wrapper.load_model()
    require(wrapper.action_head is not None, "OFT L1 action head is required")
    return wrapper


def raw_views(obs: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    output = {}
    for view, key in (("front", "full_image"), ("wrist", "wrist_image")):
        tensor = torch.as_tensor(np.asarray(obs[key]), device=device, dtype=torch.float32)
        output[view] = tensor.permute(2, 0, 1).unsqueeze(0).div(255.0)
    return output


def build_inputs(wrapper: Any, obs: dict[str, Any], instruction: str, patch: torch.Tensor | None, position: str) -> dict[str, torch.Tensor]:
    from VLAAttacker.multiview.openvla_oft_experiments import build_raw_patched_pixel_values

    inputs = dict(wrapper.prepare_hf_inputs(obs, instruction))
    inputs["pixel_values"] = build_raw_patched_pixel_values(
        raw_views(obs, wrapper.device), patch,
        input_views=["front", "wrist"], patch_views=[] if patch is None else ["wrist"],
        image_processor=wrapper.processor.image_processor,
        device=wrapper.device, dtype=wrapper.dtype, position=position,
        center_crop=True, eval_image_size=int(obs["wrist_image"].shape[0]), match_eval_resize=False,
    )
    return inputs


def _normalized_proprio(wrapper: Any, obs: dict[str, Any]) -> np.ndarray:
    stats = wrapper.vla.norm_stats[wrapper.gen_cfg.unnorm_key]["proprio"]
    return wrapper._official_openvla_utils.normalize_proprio(np.asarray(obs["state"]), stats)


def unnormalize_action(wrapper: Any, normalized: torch.Tensor) -> np.ndarray:
    values = normalized.detach().float().cpu().numpy().reshape(int(wrapper.cfg.num_open_loop_steps), 7)
    output = wrapper.vla._unnormalize_actions(values, wrapper.gen_cfg.unnorm_key)
    output = np.asarray(output, dtype=np.float32).reshape(int(wrapper.cfg.num_open_loop_steps), 7)
    require(np.isfinite(output).all(), "Non-finite unnormalized action")
    return output


def differentiable_action_and_features(
    wrapper: Any,
    obs: dict[str, Any],
    instruction: str,
    patch: torch.Tensor | None,
    position: str,
    capture_sites: tuple[str, ...] = ("dino",),
    capture_view: str = "wrist",
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    """Return normalized differentiable action chunk and frozen VF1 visual sites."""
    from src.hooks.manager import HookSpec, TensorHooks

    inputs = build_inputs(wrapper, obs, instruction, patch, position)
    require(capture_view in {"front", "wrist"}, "Unknown capture view")
    call_index = 0 if capture_view == "front" else 1
    site_specs = {
        "dino": HookSpec(path="vla.vision_backbone.featurizer", hook="output", call_index=call_index),
        "siglip": HookSpec(path="vla.vision_backbone.fused_featurizer", hook="output", call_index=call_index),
    }
    specs = [site_specs[site] for site in capture_sites]
    roots = {"vla": wrapper.vla, "action_head": wrapper.action_head}
    with TensorHooks(roots, specs, mode="capture", retain_grad=True) as hooks:
        _detached_action, action_hidden = wrapper.vla.predict_action(
            **inputs,
            unnorm_key=wrapper.gen_cfg.unnorm_key,
            do_sample=False,
            proprio=_normalized_proprio(wrapper, obs),
            proprio_projector=wrapper.proprio_projector,
            noisy_action_projector=wrapper.noisy_action_projector,
            action_head=wrapper.action_head,
            use_film=False,
        )
    head = getattr(wrapper.action_head, "module", wrapper.action_head)
    normalized = head.predict_action(action_hidden).reshape(1, int(wrapper.cfg.num_open_loop_steps), 7).float()
    features = {site: value for site, value in zip(capture_sites, hooks.captured)}
    require(all(value is not None and torch.isfinite(value).all() for value in features.values()), "Invalid visual feature")
    return normalized, features, inputs["pixel_values"]


def renderer_metadata() -> tuple[str, str]:
    from OpenGL import GL

    renderer = GL.glGetString(GL.GL_RENDERER).decode(errors="replace")
    vendor = GL.glGetString(GL.GL_VENDOR).decode(errors="replace")
    require("nvidia" in vendor.lower(), f"NVIDIA EGL required: {vendor}/{renderer}")
    require(not any(x in renderer.lower() for x in ("llvmpipe", "softpipe")), "CPU renderer forbidden")
    return renderer, vendor


def make_env(cfg: dict[str, Any], suite: str, task_id: int):
    add_paths(cfg)
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bench = benchmark.get_benchmark_dict()[suite]()
    task = bench.get_task(task_id)
    states = bench.get_task_init_states(task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_heights=int(cfg["env_image_resolution"]),
        camera_widths=int(cfg["env_image_resolution"]),
    )
    env.reset()
    return env, task, states


def reset_wait(env: Any, init_state: Any, seed: int, waits: int) -> dict[str, Any]:
    from VLAAttacker.multiview.openvla_oft_experiments import get_libero_dummy_action

    env.seed(seed)
    env.reset()
    raw = env.set_init_state(init_state)
    require(np.array_equal(np.asarray(env.get_sim_state()), np.asarray(init_state)), "Initial state reset mismatch")
    for _ in range(waits):
        raw, _reward, done, _info = env.step(get_libero_dummy_action())
        require(not done, "Episode ended during fixed wait")
    return raw
