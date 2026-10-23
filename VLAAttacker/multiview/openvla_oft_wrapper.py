"""
OpenVLA-OFT adapter for multiview attack experiments.

Purpose:
  - M0/M3/M4/M6/M7: eval wrapper around official OpenVLA-OFT helpers.
  - M1/M2/M5: differentiable forward probe / training interface.

Important:
  Official OpenVLA-OFT eval code builds multiview input by processing primary and
  wrist images separately, then concatenating pixel_values along image dimension.
  This adapter follows that convention for eval. For white-box patch training,
  do not assume gradients survive PIL/NumPy/HF processor preprocessing. Use
  `probe_differentiable_forward()` first; if it reports no gradient, implement
  a pixel_values-level patch path in your attack runner.

This file is designed to be copied into:
  roboticAttack-main/VLAAttacker/multiview/openvla_oft_wrapper.py
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image


LOGGER = logging.getLogger(__name__)


@dataclass
class OpenVLAOFTGenerateConfig:
    """Lightweight LIBERO GenerateConfig clone without importing robosuite/LIBERO eval."""

    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = ""
    use_l1_regression: bool = True
    use_diffusion: bool = False
    num_diffusion_steps_train: int = 50
    num_diffusion_steps_inference: int = 50
    use_film: bool = False
    num_images_in_input: int = 2
    use_proprio: bool = True
    center_crop: bool = True
    num_open_loop_steps: int = 8
    lora_rank: int = 32
    unnorm_key: Union[str, Path] = "libero_spatial_no_noops"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    initial_states_path: str = "DEFAULT"
    env_img_res: int = 256
    run_id_note: Optional[str] = None
    local_log_dir: str = "./experiments/logs"
    use_wandb: bool = False
    wandb_entity: str = "your-wandb-entity"
    wandb_project: str = "your-wandb-project"
    seed: int = 7


def _as_path_str(x: Any) -> str:
    return str(x) if x is not None else ""


def _bool(x: Any, default: bool = False) -> bool:
    if x is None:
        return default
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.lower() in {"1", "true", "yes", "y", "on"}
    return bool(x)


def _split_views(x: Union[str, Sequence[str], None]) -> List[str]:
    if x is None:
        return []
    if isinstance(x, str):
        if x.strip().lower() in {"", "none", "null"}:
            return []
        return [v.strip() for v in x.split(",") if v.strip()]
    return [str(v).strip() for v in x if str(v).strip()]


def _module_file(module: Any) -> Path | None:
    filename = getattr(module, "__file__", None)
    if not filename:
        return None
    try:
        return Path(filename).resolve()
    except OSError:
        return None


def _under(path: Path | None, root: Path) -> bool:
    if path is None:
        return False
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _prepend_sys_path(path: Path) -> None:
    """Move a path to the front of sys.path, even if it already exists later."""
    path_str = str(path)
    sys.path[:] = [entry for entry in sys.path if entry != path_str]
    sys.path.insert(0, path_str)


def _patch_oft_action_mask_helpers() -> None:
    """Patch OpenVLA-OFT bool cumsum helpers for newer PyTorch builds."""
    try:
        from prismatic.vla.constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX
        import prismatic.extern.hf.modeling_prismatic as modeling_prismatic
        import prismatic.training.train_utils as train_utils
    except Exception:
        return

    def get_current_action_mask(token_ids: torch.Tensor) -> torch.Tensor:
        valid_positions = (token_ids != IGNORE_INDEX).long()
        cumsum = torch.cumsum(valid_positions, dim=1)
        mask = (1 <= cumsum) & (cumsum <= ACTION_DIM)
        return (token_ids > ACTION_TOKEN_BEGIN_IDX) & mask

    def get_next_actions_mask(token_ids: torch.Tensor) -> torch.Tensor:
        valid_positions = (token_ids != IGNORE_INDEX).long()
        cumsum = torch.cumsum(valid_positions, dim=1)
        mask = cumsum > ACTION_DIM
        return (token_ids > ACTION_TOKEN_BEGIN_IDX) & mask

    train_utils.get_current_action_mask = get_current_action_mask
    train_utils.get_next_actions_mask = get_next_actions_mask
    modeling_prismatic.get_current_action_mask = get_current_action_mask
    modeling_prismatic.get_next_actions_mask = get_next_actions_mask


def _patch_oft_update_auto_map(openvla_utils: Any) -> None:
    """Make official update_auto_map safe for shared local checkpoints."""
    if getattr(openvla_utils, "_robopatch_update_auto_map_patched", False):
        return
    if not hasattr(openvla_utils, "update_auto_map"):
        return

    desired_auto_map = {
        "AutoConfig": "configuration_prismatic.OpenVLAConfig",
        "AutoModelForVision2Seq": "modeling_prismatic.OpenVLAForActionPrediction",
    }

    def safe_update_auto_map(pretrained_checkpoint: str) -> None:
        checkpoint = Path(pretrained_checkpoint)
        if not checkpoint.is_dir():
            return
        config_path = checkpoint / "config.json"
        if not config_path.exists():
            print(f"Warning: No config.json found at {config_path}")
            return

        lock_path = checkpoint / "config.json.lock"
        with lock_path.open("w", encoding="utf-8") as lock_file:
            lock_module = None
            try:
                import fcntl

                lock_module = fcntl
                lock_module.flock(lock_file.fileno(), lock_module.LOCK_EX)
            except Exception:
                lock_module = None
            try:
                with config_path.open("r", encoding="utf-8") as handle:
                    config = json.load(handle)
                if config.get("auto_map") == desired_auto_map:
                    return

                import shutil
                from datetime import datetime

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = checkpoint / f"config.json.back.{timestamp}.{os.getpid()}"
                shutil.copy2(config_path, backup_path)
                print(f"Created backup of original config at: {backup_path.resolve()}")

                config["auto_map"] = desired_auto_map
                tmp_path = checkpoint / f".config.json.tmp.{os.getpid()}"
                with tmp_path.open("w", encoding="utf-8") as handle:
                    json.dump(config, handle, indent=2)
                    handle.write("\n")
                os.replace(tmp_path, config_path)
                print(f"Updated config.json at: {config_path.resolve()}")
                print("Changes made:")
                print('  - Set AutoConfig to "configuration_prismatic.OpenVLAConfig"')
                print('  - Set AutoModelForVision2Seq to "modeling_prismatic.OpenVLAForActionPrediction"')
            finally:
                if lock_module is not None:
                    try:
                        lock_module.flock(lock_file.fileno(), lock_module.LOCK_UN)
                    except Exception:
                        pass

    openvla_utils.update_auto_map = safe_update_auto_map
    openvla_utils._robopatch_update_auto_map_patched = True


def _is_cudnn_init_error(exc: BaseException) -> bool:
    text = repr(exc)
    return "CUDNN_STATUS_NOT_INITIALIZED" in text or "cuDNN error" in text


def _to_numpy_uint8_image(x: Any) -> np.ndarray:
    """Convert PIL/np/torch image to HWC uint8 RGB numpy."""
    if isinstance(x, Image.Image):
        return np.asarray(x.convert("RGB"), dtype=np.uint8)

    if isinstance(x, np.ndarray):
        arr = x
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.dtype != np.uint8:
            # Support [0,1] or [0,255]
            mx = float(np.nanmax(arr)) if arr.size else 1.0
            if mx <= 1.5:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        return arr[..., :3]

    if torch.is_tensor(x):
        t = x.detach().cpu()
        if t.ndim == 4 and t.shape[0] == 1:
            t = t[0]
        if t.ndim == 3 and t.shape[0] in (1, 3):
            t = t.permute(1, 2, 0)
        arr = t.float().numpy()
        mx = float(np.nanmax(arr)) if arr.size else 1.0
        if mx <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        return arr[..., :3]

    raise TypeError(f"Unsupported image type: {type(x)}")


def _extract_first(x: Any) -> Any:
    """Get first sample from common batched containers."""
    if torch.is_tensor(x) or isinstance(x, np.ndarray):
        if getattr(x, "ndim", 0) >= 4:
            return x[0]
        return x
    if isinstance(x, (list, tuple)) and len(x) > 0:
        return x[0]
    return x


@dataclass
class OpenVLAOFTAdapterConfig:
    pretrained_checkpoint: str
    model_subfolder: str = ""
    model_revision: str = ""
    hf_local_files_only: bool = False
    task_suite_name: str = "libero_spatial"
    unnorm_key: str = "libero_spatial_no_noops"

    # Official OFT defaults for LIBERO.
    model_family: str = "openvla"
    use_l1_regression: bool = True
    use_diffusion: bool = False
    use_film: bool = False
    num_images_in_input: int = 2
    use_proprio: bool = True
    center_crop: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    num_open_loop_steps: int = 8
    num_diffusion_steps_train: int = 50
    num_diffusion_steps_inference: int = 50
    lora_rank: int = 32

    # Experiment-specific.
    input_views: List[str] = field(default_factory=lambda: ["front", "wrist"])
    view_key_map: Dict[str, str] = field(default_factory=lambda: {"front": "image_primary", "wrist": "image_wrist"})
    openvla_oft_root: str = ""
    libero_root: str = ""
    device: str = "cuda"
    dtype: str = "bfloat16"
    disable_cudnn: bool = False
    allow_cpu: bool = False

    @classmethod
    def from_exp_config(cls, cfg: Mapping[str, Any]) -> "OpenVLAOFTAdapterConfig":
        ckpt = (
            cfg.get("model_checkpoint")
            or cfg.get("pretrained_checkpoint")
            or cfg.get("checkpoint")
            or ""
        )
        view_map = cfg.get("view_key_map") or cfg.get("view_map") or {"front": "image_primary", "wrist": "image_wrist"}
        if isinstance(view_map, str):
            # Accept "front=image_primary,wrist=image_wrist"
            parsed = {}
            for item in view_map.split(","):
                if "=" in item:
                    k, v = item.split("=", 1)
                    parsed[k.strip()] = v.strip()
            view_map = parsed or {"front": "image_primary", "wrist": "image_wrist"}

        disable_cudnn = cfg.get("disable_cudnn")
        if disable_cudnn is None:
            disable_cudnn = cfg.get("probe_disable_cudnn")

        return cls(
            pretrained_checkpoint=_as_path_str(ckpt),
            model_subfolder=_as_path_str(cfg.get("model_subfolder") or cfg.get("pretrained_subfolder") or ""),
            model_revision=_as_path_str(cfg.get("model_revision") or cfg.get("pretrained_revision") or ""),
            hf_local_files_only=_bool(cfg.get("hf_local_files_only"), False),
            task_suite_name=str(cfg.get("task_suite") or cfg.get("task_suite_name") or "libero_spatial"),
            unnorm_key=str(cfg.get("unnorm_key") or "libero_spatial_no_noops"),
            num_images_in_input=int(cfg.get("num_images_in_input", 2)),
            use_proprio=_bool(cfg.get("use_proprio"), True),
            center_crop=_bool(cfg.get("center_crop"), True),
            input_views=_split_views(cfg.get("input_views")) or ["front", "wrist"],
            view_key_map=dict(view_map),
            openvla_oft_root=_as_path_str(cfg.get("openvla_oft_root") or cfg.get("oft_repo_root") or ""),
            libero_root=_as_path_str(cfg.get("libero_root") or ""),
            device=str(cfg.get("device") or "cuda"),
            dtype=str(cfg.get("torch_dtype") or cfg.get("dtype") or "bfloat16"),
            disable_cudnn=_bool(disable_cudnn, False),
            allow_cpu=_bool(cfg.get("allow_cpu"), False),
        )


class OpenVLAOFTWrapper:
    """
    Thin adapter around official OpenVLA-OFT evaluation helpers.

    Expected canonical observation:
      {
        "full_image": np.ndarray HWC uint8,
        "wrist_image": np.ndarray HWC uint8,
        "state": np.ndarray proprio,
        "task_description": str, optional
      }
    """

    def __init__(self, config: Union[OpenVLAOFTAdapterConfig, Mapping[str, Any]], device: Optional[str] = None) -> None:
        self.cfg = config if isinstance(config, OpenVLAOFTAdapterConfig) else OpenVLAOFTAdapterConfig.from_exp_config(config)
        if device:
            self.cfg.device = device

        requested_device = torch.device(self.cfg.device)
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "OpenVLA-OFT requested CUDA but torch.cuda.is_available() is false. "
                "Check Slurm --gres/CUDA_VISIBLE_DEVICES instead of silently running on CPU."
            )
        if requested_device.type == "cpu" and not self.cfg.allow_cpu:
            raise RuntimeError(
                "OpenVLA-OFT CPU execution is disabled by default. "
                "Set allow_cpu=true only for explicit debugging."
            )
        self.device = requested_device
        self.dtype = torch.bfloat16 if self.cfg.dtype in {"bf16", "bfloat16", "torch.bfloat16"} else torch.float16

        self._official_generate_config_cls = None
        self._official_openvla_utils = None
        self._official_robot_utils = None
        self._constants = None

        self.gen_cfg = None
        self.vla = None
        self.processor = None
        self.action_head = None
        self.proprio_projector = None
        self.noisy_action_projector = None
        self._cudnn_disabled_by_wrapper = False
        self._resolved_pretrained_checkpoint: str | None = None

        if self.cfg.disable_cudnn:
            self._disable_cudnn("config disable_cudnn=true")

    # ---------------- import / setup ----------------

    def _disable_cudnn(self, reason: str) -> bool:
        if self.device.type != "cuda":
            return False
        cudnn = getattr(torch.backends, "cudnn", None)
        if cudnn is None:
            return False

        was_enabled = bool(getattr(cudnn, "enabled", False))
        cudnn.enabled = False
        cudnn.benchmark = False
        self._cudnn_disabled_by_wrapper = True
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        if was_enabled:
            LOGGER.warning("Disabled torch.backends.cudnn for OpenVLA-OFT: %s", reason)
        return was_enabled

    def _with_cudnn_fallback(self, context: str, fn: Any) -> Any:
        try:
            return fn()
        except RuntimeError as exc:
            if not _is_cudnn_init_error(exc):
                raise
            changed = self._disable_cudnn(f"{context} raised {exc}")
            if not changed:
                raise
            return fn()

    def _try_import_official(self) -> None:
        if self._official_openvla_utils is not None:
            return

        for extra_root in [self.cfg.libero_root, self.cfg.openvla_oft_root]:
            if extra_root:
                path = Path(extra_root).expanduser()
                if path.exists():
                    _prepend_sys_path(path)

        if self.cfg.openvla_oft_root:
            oft_root = Path(self.cfg.openvla_oft_root).expanduser().resolve()
            for module_name, module in list(sys.modules.items()):
                if module_name == "prismatic" or module_name.startswith("prismatic."):
                    if not _under(_module_file(module), oft_root):
                        del sys.modules[module_name]
                if module_name in {"experiments", "experiments.robot"} or module_name.startswith("experiments.robot."):
                    if not _under(_module_file(module), oft_root):
                        del sys.modules[module_name]

        # The user's repo may vendor openvla-oft, or PYTHONPATH may point to it.
        try:
            self._official_openvla_utils = importlib.import_module("experiments.robot.openvla_utils")
            self._official_robot_utils = importlib.import_module("experiments.robot.robot_utils")
            self._official_generate_config_cls = OpenVLAOFTGenerateConfig
            _patch_oft_action_mask_helpers()
            _patch_oft_update_auto_map(self._official_openvla_utils)
            try:
                self._constants = importlib.import_module("prismatic.vla.constants")
            except Exception:
                self._constants = None
        except Exception as e:
            raise ImportError(
                "Cannot import official OpenVLA-OFT utility modules. "
                "Make sure openvla-oft is installed or its root is on PYTHONPATH. "
                "Expected modules: experiments.robot.openvla_utils, "
                "experiments.robot.robot_utils. "
                f"Original error: {e}"
            ) from e

    def build_generate_config(self) -> Any:
        self._try_import_official()
        GenerateConfig = self._official_generate_config_cls

        kwargs = {
            "pretrained_checkpoint": self.resolve_pretrained_checkpoint(),
            "use_l1_regression": self.cfg.use_l1_regression,
            "use_diffusion": self.cfg.use_diffusion,
            "use_film": self.cfg.use_film,
            "num_images_in_input": self.cfg.num_images_in_input,
            "use_proprio": self.cfg.use_proprio,
            "load_in_8bit": self.cfg.load_in_8bit,
            "load_in_4bit": self.cfg.load_in_4bit,
            "center_crop": self.cfg.center_crop,
            "num_open_loop_steps": self.cfg.num_open_loop_steps,
            "unnorm_key": self.cfg.unnorm_key,
            "task_suite_name": self.cfg.task_suite_name,
            "lora_rank": self.cfg.lora_rank,
        }
        # Some GenerateConfig versions may not accept all fields.
        try:
            return GenerateConfig(**kwargs)
        except TypeError:
            valid = getattr(GenerateConfig, "__dataclass_fields__", {})
            filtered = {k: v for k, v in kwargs.items() if k in valid}
            return GenerateConfig(**filtered)

    def resolve_pretrained_checkpoint(self) -> str:
        """Resolve optional HF repo_id + subfolder into a concrete checkpoint directory.

        Official OpenVLA-OFT helper functions expect a single checkpoint directory
        when loading action_head, proprio_projector, and dataset_statistics. For
        models packaged as a Hugging Face repo with LIBERO-suite subfolders, we
        resolve the requested subfolder once here, then pass that directory to the
        official loader so every component comes from the same suite.
        """
        if self._resolved_pretrained_checkpoint is not None:
            return self._resolved_pretrained_checkpoint

        checkpoint = str(self.cfg.pretrained_checkpoint).strip()
        subfolder = str(self.cfg.model_subfolder or "").strip().strip("/")
        if not checkpoint:
            raise ValueError("pretrained_checkpoint/model_checkpoint is empty.")
        if not subfolder:
            self._resolved_pretrained_checkpoint = checkpoint
            return checkpoint

        checkpoint_path = Path(checkpoint).expanduser()
        if checkpoint_path.exists():
            candidate = checkpoint_path / subfolder
            if not candidate.exists() and checkpoint_path.name == subfolder:
                candidate = checkpoint_path
            if not candidate.exists():
                raise FileNotFoundError(
                    f"Configured model_subfolder={subfolder!r} was not found under {checkpoint_path}"
                )
            self._resolved_pretrained_checkpoint = str(candidate)
            return self._resolved_pretrained_checkpoint

        try:
            from huggingface_hub import snapshot_download
        except Exception as exc:
            raise RuntimeError(
                "model_subfolder requires huggingface_hub when model_checkpoint is a repo id."
            ) from exc

        snapshot_kwargs: dict[str, Any] = {
            "repo_id": checkpoint,
            "allow_patterns": [f"{subfolder}/*"],
            "local_files_only": bool(self.cfg.hf_local_files_only),
        }
        if self.cfg.model_revision:
            snapshot_kwargs["revision"] = self.cfg.model_revision
        snapshot_dir = Path(snapshot_download(**snapshot_kwargs))
        candidate = snapshot_dir / subfolder
        if not candidate.exists():
            raise FileNotFoundError(
                f"HF repo {checkpoint!r} resolved to {snapshot_dir}, but subfolder {subfolder!r} is missing."
            )
        self._resolved_pretrained_checkpoint = str(candidate)
        return self._resolved_pretrained_checkpoint

    @contextmanager
    def _official_workdir(self) -> Iterable[None]:
        """Run official helpers from openvla-oft root so checkpoint sync uses the right prismatic files."""
        old_cwd = Path.cwd()
        target = Path(self.cfg.openvla_oft_root).expanduser() if self.cfg.openvla_oft_root else None
        if target and target.exists():
            os.chdir(target)
        try:
            yield
        finally:
            os.chdir(old_cwd)

    def load_model(self) -> None:
        """Load OpenVLA-OFT VLA + processor + action head + proprio projector."""
        if self.vla is not None:
            return
        self._try_import_official()
        ou = self._official_openvla_utils

        self.gen_cfg = self.build_generate_config()
        with self._official_workdir():
            self.vla = ou.get_vla(self.gen_cfg)
            self.processor = ou.get_processor(self.gen_cfg)

            self.action_head = None
            if getattr(self.gen_cfg, "use_l1_regression", False) or getattr(self.gen_cfg, "use_diffusion", False):
                self.action_head = ou.get_action_head(self.gen_cfg, llm_dim=self.vla.llm_dim)

            self.proprio_projector = None
            if getattr(self.gen_cfg, "use_proprio", False):
                proprio_dim = 8
                if self._constants is not None and hasattr(self._constants, "PROPRIO_DIM"):
                    proprio_dim = int(getattr(self._constants, "PROPRIO_DIM"))
                self.proprio_projector = ou.get_proprio_projector(
                    self.gen_cfg,
                    llm_dim=self.vla.llm_dim,
                    proprio_dim=proprio_dim,
                )

            self.noisy_action_projector = None
            if getattr(self.gen_cfg, "use_diffusion", False):
                self.noisy_action_projector = ou.get_noisy_action_projector(self.gen_cfg, llm_dim=self.vla.llm_dim)

        self.freeze_model()

    def freeze_model(self) -> None:
        """Freeze all model-side weights; patch tensor should be optimized outside."""
        for module in [self.vla, self.action_head, self.proprio_projector, self.noisy_action_projector]:
            if module is None:
                continue
            module.eval()
            for p in module.parameters():
                p.requires_grad_(False)

    # ---------------- observation utilities ----------------

    def canonicalize_observation(self, obs: Mapping[str, Any], instruction: Optional[str] = None) -> Dict[str, Any]:
        """
        Convert arbitrary batch/obs keys to official OpenVLA-OFT observation keys.
        Uses cfg.view_key_map by default:
          front -> image_primary
          wrist -> image_wrist
        """
        out: Dict[str, Any] = {}

        front_key = self.cfg.view_key_map.get("front", "image_primary")
        wrist_key = self.cfg.view_key_map.get("wrist", "image_wrist")
        by_view = obs.get("pixel_values_by_view")

        if "front" in self.cfg.input_views:
            if "full_image" in obs:
                out["full_image"] = _to_numpy_uint8_image(_extract_first(obs["full_image"]))
            elif isinstance(by_view, Mapping) and "front" in by_view:
                out["full_image"] = _to_numpy_uint8_image(_extract_first(by_view["front"]))
            elif front_key in obs:
                out["full_image"] = _to_numpy_uint8_image(_extract_first(obs[front_key]))
            elif "image_primary" in obs:
                out["full_image"] = _to_numpy_uint8_image(_extract_first(obs["image_primary"]))

        if "wrist" in self.cfg.input_views:
            if "wrist_image" in obs:
                out["wrist_image"] = _to_numpy_uint8_image(_extract_first(obs["wrist_image"]))
            elif isinstance(by_view, Mapping) and "wrist" in by_view:
                out["wrist_image"] = _to_numpy_uint8_image(_extract_first(by_view["wrist"]))
            elif wrist_key in obs:
                out["wrist_image"] = _to_numpy_uint8_image(_extract_first(obs[wrist_key]))
            elif "image_wrist" in obs:
                out["wrist_image"] = _to_numpy_uint8_image(_extract_first(obs["image_wrist"]))

        # LIBERO proprio.
        if "state" in obs:
            state = _extract_first(obs["state"])
            out["state"] = state.detach().cpu().numpy() if torch.is_tensor(state) else np.asarray(state)
        else:
            # Common LIBERO raw obs keys.
            if all(k in obs for k in ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]):
                # Avoid importing quat conversion here; caller can pass canonical state.
                # Placeholder concatenation uses quaternion directly if axis-angle converter is unavailable.
                pos = np.asarray(_extract_first(obs["robot0_eef_pos"]))
                quat = np.asarray(_extract_first(obs["robot0_eef_quat"]))
                grip = np.asarray(_extract_first(obs["robot0_gripper_qpos"]))
                out["state"] = np.concatenate([pos, quat[:3], grip], axis=-1)
            else:
                # LIBERO OFT expects proprio when use_proprio=True.
                out["state"] = np.zeros(8, dtype=np.float32)

        if instruction is not None:
            out["task_description"] = instruction
        elif "task_description" in obs:
            td = obs["task_description"]
            out["task_description"] = td[0] if isinstance(td, (list, tuple)) else str(td)
        elif "instruction" in obs:
            ins = obs["instruction"]
            out["task_description"] = ins[0] if isinstance(ins, (list, tuple)) else str(ins)
        elif "instructions" in obs:
            ins = obs["instructions"]
            out["task_description"] = ins[0] if isinstance(ins, (list, tuple)) else str(ins)

        return out

    def _ensure_full_and_wrist(self, observation: Mapping[str, Any]) -> None:
        if "front" in self.cfg.input_views and "full_image" not in observation:
            raise KeyError("Missing front/full_image in observation.")
        if "wrist" in self.cfg.input_views and self.cfg.num_images_in_input > 1 and "wrist_image" not in observation:
            raise KeyError("Missing wrist/wrist_image in observation.")

    # ---------------- eval API ----------------

    @torch.no_grad()
    def predict_action(self, obs: Mapping[str, Any], instruction: str) -> List[np.ndarray]:
        """Official no-grad eval path for M0/M3/M4/M6/M7."""
        self.load_model()
        self._try_import_official()
        ou = self._official_openvla_utils

        observation = self.canonicalize_observation(obs, instruction=instruction)
        self._ensure_full_and_wrist(observation)

        actions = self._with_cudnn_fallback(
            "predict_action",
            lambda: ou.get_vla_action(
                self.gen_cfg,
                self.vla,
                self.processor,
                observation,
                instruction,
                self.action_head,
                self.proprio_projector,
                self.noisy_action_projector,
                use_film=getattr(self.gen_cfg, "use_film", False),
            ),
        )
        return actions

    @torch.no_grad()
    def predict_action_with_pixel_patch(
        self,
        obs: Mapping[str, Any],
        instruction: str,
        patch: Any,
        patch_views: Sequence[str],
        *,
        position: str = "top_left",
    ) -> List[np.ndarray]:
        """Apply the official M-series patch in OpenVLA-OFT pixel_values space."""
        self.load_model()
        ou = self._official_openvla_utils

        observation = self.canonicalize_observation(obs, instruction=instruction)
        self._ensure_full_and_wrist(observation)
        inputs = self.prepare_hf_inputs(observation, instruction)

        # 正式 M 系列训练阶段的 patch 直接作用在 processor 输出的 pixel_values 上；
        # 评估阶段使用同一输入空间，避免 raw RGB 预处理改变补丁作用方式。
        from VLAAttacker.multiview.openvla_oft_experiments import apply_patch_to_pixel_values

        inputs["pixel_values"] = apply_patch_to_pixel_values(
            inputs["pixel_values"],
            patch,
            input_views=list(self.cfg.input_views),
            patch_views=[str(view).strip() for view in patch_views if str(view).strip()],
            position=position,
        )

        # 官方 OFT get_vla_action 会在 predict_action 前归一化 proprio；直接调用
        # vla.predict_action 时也必须复刻，否则 pixel patch 评估会混入 state 分布差异。
        proprio = None
        if getattr(self.gen_cfg, "use_proprio", False):
            proprio = observation["state"]
            proprio_norm_stats = self.vla.norm_stats[self.gen_cfg.unnorm_key]["proprio"]
            proprio = ou.normalize_proprio(proprio, proprio_norm_stats)

        prediction = self._with_cudnn_fallback(
            "predict_action_with_pixel_patch",
            lambda: self.vla.predict_action(
                **inputs,
                unnorm_key=getattr(self.gen_cfg, "unnorm_key", self.cfg.unnorm_key),
                do_sample=False,
                proprio=proprio,
                proprio_projector=self.proprio_projector,
                noisy_action_projector=self.noisy_action_projector,
                action_head=self.action_head,
                use_film=getattr(self.gen_cfg, "use_film", False),
            ),
        )
        action = prediction[0] if isinstance(prediction, tuple) else prediction
        return [action[i] for i in range(len(action))]

    # ---------------- probe / training API ----------------

    def prepare_hf_inputs(self, obs: Mapping[str, Any], instruction: str) -> Dict[str, torch.Tensor]:
        """
        Build HF inputs like official get_vla_action, but without torch.inference_mode.
        This still goes through PIL/processor, so gradients from raw image pixels are
        not expected. It is mainly for probing logits/predict_action behavior.
        """
        self.load_model()
        ou = self._official_openvla_utils

        observation = self.canonicalize_observation(obs, instruction=instruction)
        self._ensure_full_and_wrist(observation)

        prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
        images = [observation["full_image"]]
        if getattr(self.gen_cfg, "num_images_in_input", 1) > 1 and "wrist_image" in observation:
            images.append(observation["wrist_image"])

        processed_images = ou.prepare_images_for_vla(images, self.gen_cfg)
        primary_image = processed_images[0]
        inputs = self.processor(prompt, primary_image).to(self.device, dtype=self.dtype)

        if len(processed_images) > 1:
            wrist_values = []
            for image_wrist in processed_images[1:]:
                wrist_inputs = self.processor(prompt, image_wrist).to(self.device, dtype=self.dtype)
                wrist_values.append(wrist_inputs["pixel_values"])
            inputs["pixel_values"] = torch.cat([inputs["pixel_values"]] + wrist_values, dim=1)

        return inputs

    def _attach_dummy_labels(self, inputs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        """OFT multimodal forward expects labels for action-mask construction."""
        if "labels" not in inputs and "input_ids" in inputs:
            labels = inputs["input_ids"].clone()
            labels.fill_(-100)
            inputs["labels"] = labels
        return inputs

    def forward_for_attack(
        self,
        obs: Mapping[str, Any],
        instruction: str,
        *,
        return_logits: bool = True,
        allow_predict_action_tensor: bool = True,
    ) -> Dict[str, Any]:
        """
        Differentiable forward probe.

        This function does NOT apply patch itself. The attack runner should:
          1. build / patch pixel_values or images,
          2. pass tensors through model,
          3. check that loss.backward() reaches patch.grad.

        Here we only expose model outputs and fail loudly when returned action is non-tensor.
        """
        self.load_model()
        inputs = self.prepare_hf_inputs(obs, instruction)
        forward_inputs = dict(self._attach_dummy_labels(dict(inputs)))

        result: Dict[str, Any] = {"inputs": inputs, "forward_inputs": forward_inputs}

        if return_logits:
            try:
                outputs = self._with_cudnn_fallback(
                    "forward_for_attack logits",
                    lambda: self.vla(**forward_inputs, return_dict=True),
                )
                result["outputs"] = outputs
                if hasattr(outputs, "logits"):
                    result["logits"] = outputs.logits
            except Exception as e:
                result["forward_error"] = repr(e)

        if allow_predict_action_tensor:
            try:
                observation = self.canonicalize_observation(obs, instruction=instruction)
                proprio = observation.get("state", None)
                if proprio is not None and getattr(self.gen_cfg, "use_proprio", False):
                    # Official normalization happens inside get_vla_action; this direct path keeps it simple.
                    proprio = np.asarray(proprio)
                action, extra = self._with_cudnn_fallback(
                    "forward_for_attack predict_action",
                    lambda: self.vla.predict_action(
                        **inputs,
                        unnorm_key=getattr(self.gen_cfg, "unnorm_key", self.cfg.unnorm_key),
                        do_sample=False,
                        proprio=proprio,
                        proprio_projector=self.proprio_projector,
                        noisy_action_projector=self.noisy_action_projector,
                        action_head=self.action_head,
                        use_film=getattr(self.gen_cfg, "use_film", False),
                    ),
                )
                result["predict_action"] = action
                result["predict_action_extra"] = extra
                if not torch.is_tensor(action):
                    result["predict_action_warning"] = (
                        "predict_action returned non-tensor output; this path is not directly differentiable."
                    )
            except Exception as e:
                result["predict_action_error"] = repr(e)

        return result

    def probe_differentiable_forward(self, obs: Mapping[str, Any], instruction: str) -> Dict[str, Any]:
        """
        Lightweight gradient probe on pixel_values.

        It adds a tiny learnable delta to pixel_values and checks whether a scalar
        model output can backpropagate to delta. This does not guarantee patch
        training is semantically correct, but it tells you whether the graph is open.
        """
        self.load_model()
        inputs = self.prepare_hf_inputs(obs, instruction)
        pixel_values = inputs["pixel_values"].detach().clone().requires_grad_(False)
        delta = torch.zeros_like(pixel_values, requires_grad=True)
        inputs["pixel_values"] = pixel_values + delta
        forward_inputs = dict(self._attach_dummy_labels(dict(inputs)))

        status = {
            "pixel_values_shape": tuple(pixel_values.shape),
            "supports_forward_logits": False,
            "pixel_delta_grad": False,
            "notes": [],
        }

        try:
            outputs = self._with_cudnn_fallback(
                "probe_differentiable_forward logits",
                lambda: self.vla(**forward_inputs, return_dict=True),
            )
            logits = getattr(outputs, "logits", None)
            if logits is not None and torch.is_tensor(logits):
                status["supports_forward_logits"] = True
                loss = logits.float().mean()
                loss.backward()
                status["pixel_delta_grad"] = delta.grad is not None and torch.isfinite(delta.grad).all().item()
                status["logits_shape"] = tuple(logits.shape)
            else:
                status["notes"].append("vla forward returned no logits tensor.")
        except Exception as e:
            status["notes"].append(f"forward logits probe failed: {repr(e)}")

        return status
