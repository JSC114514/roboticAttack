"""Read-only native model loading and actual-environment EGL lifecycle helpers.

Adapted from the hashed V2 source listed in COPIED_SOURCES.json. Old patch,
intervention and 256px environment routines are intentionally excluded.
"""
from __future__ import annotations

import gc
import ctypes
import math
import os
import sys
import threading
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from atlas_common import require


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
    from atlas_readonly import install_readonly_loader
    install_readonly_loader(wrapper, cfg["paths"]["openvla_oft_root"])
    wrapper.load_model()
    require(wrapper.action_head is not None, "OFT L1 action head is required")
    return wrapper


class RendererInitializationError(RuntimeError):
    """Actual renderer failure with JSON-serializable, non-sensitive evidence."""

    def __init__(self, message: str, details: dict[str, Any]):
        super().__init__(message)
        self.details = details


def resolve_gl_string_query(egl):
    """Use the entry point from our EGL dispatch, with its exact C ABI.

    Do not let a late PyOpenGL GL/GLX symbol lookup select a different library
    after the model stack loads. No context is created or GPU probed here.
    """
    address = _pointer_value(egl.eglGetProcAddress("glGetString"))
    require(address != 0, "EGL cannot resolve glGetString")
    return ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_uint)(address)


@lru_cache(maxsize=1)
def prepare_graphics_backend():
    """Load GL/EGL entry points before creating/binding any simulator context."""
    # Use the SAME EGL module as installed robosuite, not another probe context.
    from mujoco.egl import egl_ext as EGL
    from OpenGL import GL, platform
    # Force loading while import order is still under our control (GL is lazy).
    platform.PLATFORM.GL
    query = resolve_gl_string_query(EGL)
    return SimpleNamespace(GL=GL, EGL=EGL, platform=platform.PLATFORM, gl_string_query=query)


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
    backend = prepare_graphics_backend()
    values = {}
    try:
        for name, token in (("renderer", 0x1F01), ("vendor", 0x1F00), ("version", 0x1F02)):
            pointer = backend.gl_string_query(token)
            require(_pointer_value(pointer) != 0, f"EGL glGetString returned NULL: {name}")
            values[name] = ctypes.string_at(pointer)
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


