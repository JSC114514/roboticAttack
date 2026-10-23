from __future__ import annotations

import gc
import ctypes
import math
import os
import sys
import threading
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
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
    from VLAAttacker.multiview.openvla_oft_experiments import _apply_patch_to_raw_view_tensor
    from preprocessing import aligned_pixels

    inputs = dict(wrapper.prepare_hf_inputs(obs, instruction))
    if patch is not None:
        views = raw_views(obs,wrapper.device)
        wrist = _apply_patch_to_raw_view_tensor(views["wrist"],patch,position=position,
                    eval_image_size=int(obs["wrist_image"].shape[0]),match_eval_resize=False)
        inputs["pixel_values"] = aligned_pixels(views["front"][0],wrist[0],wrapper.processor.image_processor,
                    wrapper._official_openvla_utils,wrapper.gen_cfg,wrapper.dtype)
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


class RendererInitializationError(RuntimeError):
    """Actual renderer failure with JSON-serializable, non-sensitive evidence."""

    def __init__(self, message: str, details: dict[str, Any]):
        super().__init__(message)
        self.details = details


def prepare_graphics_backend():
    """Load GL/EGL entry points before creating/binding any simulator context."""
    # Use the SAME EGL module as installed robosuite, not another probe context.
    from mujoco.egl import egl_ext as EGL
    from OpenGL import GL, platform
    return SimpleNamespace(GL=GL, EGL=EGL, platform=platform.PLATFORM)


def _pointer_value(value: Any) -> int:
    if value is None:
        return 0
    address = getattr(value, "address", None)
    if address is not None:
        return int(address)
    if isinstance(value, int):
        return value
    return int(ctypes.cast(value, ctypes.c_void_p).value or 0)


def _environment_gl(env: Any):
    sim = getattr(env, "sim", None)
    context = getattr(sim, "_render_context_offscreen", None)
    return context, getattr(context, "gl_ctx", None)


def graphics_diagnostics(env: Any, raw_values=None) -> dict[str, Any]:
    """Read thread/library state. No extra context, API mutation or GPU probe."""
    details = {
        "thread_id": threading.get_ident(),
        "environment": {key: os.environ.get(key) for key in (
            "MUJOCO_GL", "PYOPENGL_PLATFORM", "MUJOCO_EGL_DEVICE_ID",
            "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER")},
        "gl_values": {key: {"type": type(value).__name__, "repr": repr(value)[:1000]}
                      for key, value in (raw_values or {}).items()},
    }
    _context, gl = _environment_gl(env)
    try:
        details["expected_context"] = _pointer_value(getattr(gl, "_context", None))
    except (TypeError, ValueError, ctypes.ArgumentError) as error:
        details["expected_context_error"] = repr(error)
    try:
        backend = prepare_graphics_backend()
        details["platform"] = type(backend.platform).__name__
        details["gl_library"] = str(getattr(backend.platform.GL, "_name", None))
        details["egl_library"] = str(getattr(backend.platform.EGL, "_name", None))
        details["current_api"] = int(backend.EGL.eglQueryAPI())
        details["opengl_api"] = int(backend.EGL.EGL_OPENGL_API)
        details["current_context"] = _pointer_value(backend.EGL.eglGetCurrentContext())
        details["current_display"] = _pointer_value(backend.EGL.eglGetCurrentDisplay())
    except Exception as error:
        details["egl_diagnostic_error"] = repr(error)
    try:
        paths = set()
        with Path("/proc/self/maps").open() as maps:
            for line in maps:
                path = line.rstrip().split(maxsplit=5)[-1]
                if path.startswith("/") and any(name in path for name in (
                    "libEGL", "libOpenGL", "libGL.so", "libGLX", "libGLdispatch", "libnvidia")):
                    paths.add(path)
        details["loaded_graphics_libraries"] = sorted(paths)
    except OSError as error:
        details["library_mapping_error"] = repr(error)
    return details


def bind_environment_context(env: Any):
    """Bind this actual simulator, never an unrelated probe GL context."""
    context, gl = _environment_gl(env)
    try:
        require(gl is not None and callable(getattr(gl, "make_current", None)),
                "LIBERO offscreen EGL context is absent; cannot render or query metadata")
        expected = _pointer_value(getattr(gl, "_context", None))
        require(expected != 0, "LIBERO offscreen EGL context has already been freed")
        backend = prepare_graphics_backend()
        # free()/garbage collection calls eglReleaseThread, which resets the
        # thread's bound API. Installed make_current() does NOT restore it.
        require(bool(backend.EGL.eglBindAPI(backend.EGL.EGL_OPENGL_API)),
                "eglBindAPI(OpenGL) failed")
        gl.make_current()
        require(int(backend.EGL.eglQueryAPI()) == int(backend.EGL.EGL_OPENGL_API),
                "EGL current thread did not retain the OpenGL API")
        require(_pointer_value(backend.EGL.eglGetCurrentContext()) == expected,
                "EGL current context is not the actual LIBERO context")
        require(_pointer_value(backend.EGL.eglGetCurrentDisplay()) != 0,
                "EGL current display is empty")
    except Exception as error:
        raise RendererInitializationError(str(error), graphics_diagnostics(env)) from error
    return context


def renderer_metadata(env: Any, diagnostics=None) -> tuple[str, str]:
    bind_environment_context(env)
    GL = prepare_graphics_backend().GL
    values = {}
    try:
        for name, token in (("renderer", GL.GL_RENDERER), ("vendor", GL.GL_VENDOR),
                            ("version", GL.GL_VERSION)):
            values[name] = GL.glGetString(token)
        require(all(isinstance(values[name], (bytes, str)) and values[name]
                    for name in ("renderer", "vendor")),
                "OpenGL returned empty/invalid renderer/vendor after binding the actual LIBERO EGL context")
        renderer, vendor = [values[name].decode(errors="replace") if isinstance(values[name], bytes)
                            else values[name] for name in ("renderer", "vendor")]
        require("nvidia" in vendor.lower(), f"NVIDIA EGL required: {vendor}/{renderer}")
        require(not any(x in renderer.lower() for x in ("llvmpipe", "softpipe")), "CPU renderer forbidden")
        expected_gpu = os.environ.get("OFT_STAGE_GPU", "3090")
        require(expected_gpu in renderer, f"Allocated {expected_gpu} EGL renderer required, got {renderer}")
    except Exception as error:
        raise RendererInitializationError(str(error), graphics_diagnostics(env, values)) from error
    if diagnostics is not None:
        diagnostics.update(graphics_diagnostics(env, values))
    return renderer, vendor


def close_environment(env: Any) -> None:
    # MuJoCo rendering buffers must be freed while their OWN context is current.
    bind_environment_context(env)
    env.close()
    gc.collect()


def make_env(cfg: dict[str, Any], suite: str, task_id: int):
    add_paths(cfg)
    prepare_graphics_backend()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bench = benchmark.get_benchmark_dict()[suite]()
    task = bench.get_task(task_id)
    states = bench.get_task_init_states(task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    # Restore the older successful evaluator's initialization sequence. This
    # reset MUST actually generate camera observations, not only a GL handle.
    # Every scientific episode still sets its own frozen seed/init-state below.
    env = None
    try:
        env = OffScreenRenderEnv(
            bddl_file_name=bddl,
            camera_names=["agentview", "robot0_eye_in_hand"],
            camera_heights=int(cfg["env_image_resolution"]),
            camera_widths=int(cfg["env_image_resolution"]),
        )
        bind_environment_context(env)
        raw = reset_environment(env)
        gc.collect()
        validate_camera_frames(raw)
        renderer_metadata(env)
    except Exception as error:
        failure = error if isinstance(error, RendererInitializationError) else RendererInitializationError(
            "LIBERO camera initialization failed: " + str(error), graphics_diagnostics(env))
        if env is not None:
            try:
                close_environment(env)
            except Exception as cleanup_error:
                failure.details["cleanup_error"] = repr(cleanup_error)
        if failure is error:
            raise
        raise failure from error
    return env, task, states


def reset_environment(env: Any):
    """Keep placement retries, but do not swallow unrelated render exceptions.

    Installed LIBERO ControlEnv.reset has a finally:continue that suppresses
    EVERY exception and can turn an EGL error into an infinite reset loop.
    Its inner reset performs exactly the same simulation operation.
    """
    inner = getattr(env, "env", env)
    if inner is env:
        return env.reset()
    from robosuite.utils.errors import RandomizationError
    while True:
        try:
            return inner.reset()
        except RandomizationError:
            continue


def validate_camera_frames(raw):
    expected = (256, 256, 3)
    for key in ("agentview_image", "robot0_eye_in_hand_image"):
        require(np.asarray(raw[key]).shape == expected and np.asarray(raw[key]).dtype == np.uint8,
                f"Actual LIBERO camera frame invalid: {key}")


def reset_wait(env: Any, init_state: Any, seed: int, waits: int) -> dict[str, Any]:
    from VLAAttacker.multiview.openvla_oft_experiments import get_libero_dummy_action

    env.seed(seed)
    bind_environment_context(env)  # old context current before hard-reset free
    reset_environment(env)
    gc.collect()  # drain old object finalizers BEFORE rebinding the new renderer
    renderer_metadata(env)
    raw = env.set_init_state(init_state)
    require(np.array_equal(np.asarray(env.get_sim_state()), np.asarray(init_state)), "Initial state reset mismatch")
    for _ in range(waits):
        raw, _reward, done, _info = env.step(get_libero_dummy_action())
        require(not done, "Episode ended during fixed wait")
    renderer_metadata(env)
    validate_camera_frames(raw)
    return raw
