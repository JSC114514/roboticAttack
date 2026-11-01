# 发布阅读说明：保持原生相机采样、物理状态、EGL生命周期和严格重置证据；不通过额外渲染推进控制时钟。
# 本文件为原运行源码的注释副本；训练目标、更新规则和原生前向的可执行AST未改变。
"""EGL/reset lifecycle copied from the completed OFT 4090 evaluator; no model imports."""
from __future__ import annotations
import gc
import ctypes
import os
import sys
import threading
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import numpy as np
from contracts import require

def add_paths(cfg):
    path = str(cfg["paths"]["libero_root"])
    if path not in sys.path:
        sys.path.insert(0, path)


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
        require(expected_gpu.lower() in renderer.lower(), f"Allocated {expected_gpu} EGL renderer required, got {renderer}")
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


def make_env(cfg: dict[str, Any], suite: str, task_id: int, on_phase=None):
    add_paths(cfg)
    prepare_graphics_backend()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bench = benchmark.get_benchmark_dict()[suite]()
    task = bench.get_task(task_id)
    from contracts import load_frozen
    from native_compat import load_frozen_initial_states
    state_path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    frozen_sha = load_frozen()["environment_sources"].get(str(state_path.resolve()))
    if on_phase is not None:
        on_phase("load_frozen_initial_states")
    states = load_frozen_initial_states(state_path, frozen_sha)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    # Restore the older successful evaluator's initialization sequence. This
    # reset MUST actually generate camera observations, not only a GL handle.
    # Every scientific episode still sets its own frozen seed/init-state below.
    env = None
    try:
        if on_phase is not None:
            on_phase("create_offscreen_environment")
        env = OffScreenRenderEnv(
            bddl_file_name=bddl,
            camera_names=["agentview", "robot0_eye_in_hand"],
            camera_heights=int(cfg["env_image_resolution"]),
            camera_widths=int(cfg["env_image_resolution"]),
        )
        if on_phase is not None:
            on_phase("bind_initial_context", graphics=graphics_diagnostics(env))
        bind_environment_context(env)
        if on_phase is not None:
            on_phase("render_initial_camera_frames")
        raw = reset_environment(env)
        gc.collect()
        validate_camera_frames(raw)
        if on_phase is not None:
            on_phase("query_initial_renderer", graphics=graphics_diagnostics(env))
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

    env.seed(seed)
    bind_environment_context(env)  # old context current before hard-reset free
    reset_environment(env)
    gc.collect()  # drain old object finalizers BEFORE rebinding the new renderer
    renderer_metadata(env)
    raw = env.set_init_state(init_state)
    require(np.array_equal(np.asarray(env.get_sim_state()), np.asarray(init_state)), "Initial state reset mismatch")
    for _ in range(waits):
        raw, _reward, done, _info = env.step([0.0] * 6 + [-1.0])
        require(not done, "Episode ended during fixed wait")
    renderer_metadata(env)
    validate_camera_frames(raw)
    return raw

