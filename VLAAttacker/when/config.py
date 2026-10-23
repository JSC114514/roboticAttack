"""Lightweight configuration constants for WHEN experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from VLAAttacker.multiview.view_batch_utils import read_simple_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
USER_HOME = Path("/home/caijiasheng25")
MODEL_ROOT = USER_HOME / "models"
DATA_ROOT = USER_HOME / "Desktop" / "roboticAttack"
LIBERO_ROOT = USER_HOME / "Desktop" / "LIBERO"
UPA_RFAS_REFERENCE = PROJECT_ROOT / "VLAAttacker"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "when"

LIBERO_SUITES = ("libero_10", "libero_spatial", "libero_object", "libero_goal")
DIGITAL_PROTOCOLS = ("digital_front", "digital_wrist", "digital_both")
WORLD_PROTOCOLS = ("world_anchored_patch",)

SUITE_UNNORM_KEYS = {
    "libero_10": "libero_10_no_noops",
    "libero_spatial": "libero_spatial_no_noops",
    "libero_object": "libero_object_no_noops",
    "libero_goal": "libero_goal_no_noops",
}

SOURCE_UNNORM_KEYS = {
    "libero_10": "libero_10",
    "libero_spatial": "libero_spatial",
    "libero_object": "libero_object",
    "libero_goal": "libero_goal",
}

SOURCE_MODEL_CANDIDATES = {
    "libero_10": MODEL_ROOT / "openvla-7b-finetuned-libero-10",
    "libero_long": MODEL_ROOT / "openvla-7b-finetuned-libero-long",
    "libero_spatial": MODEL_ROOT / "openvla-7b-finetuned-libero-spatial",
    "libero_object": MODEL_ROOT / "openvla-7b-finetuned-libero-object",
    "libero_goal": MODEL_ROOT / "openvla-7b-finetuned-libero-goal",
}

OFT_MODEL_PATHS = {
    "libero_10": MODEL_ROOT / "openvla-7b-oft-finetuned-libero-10",
    "libero_spatial": USER_HOME
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--moojink--openvla-7b-oft-finetuned-libero-spatial"
    / "snapshots"
    / "6d0231af0e48c5985f1ff86908f4674b84bc049b",
    "libero_object": MODEL_ROOT / "openvla-7b-oft-finetuned-libero-object",
    "libero_goal": MODEL_ROOT / "openvla-7b-oft-finetuned-libero-goal",
}

OFFICIAL_WHEN_DEFAULTS: dict[str, Any] = {
    "patch_name": "P_WHEN",
    "patch_size": "3,50,50",
    "learning_rate": "1e-3",
    "batch_size": "2",
    "iterations": "2000",
    "warmup_steps": "20",
    "inner_loop": "50",
    "inner_steps": "8",
    "inner_eps": "2/255",
    "lambda_con": "1.0",
    "lambda_attn": "1.0",
    "lambda_text": "0.5",
    "lambda_1": "0.1",
    "tau": "0.3",
    "beta_neg": "0.5",
    "last_n": "2",
    "lambda_nonpatch": "0.8",
    "topk_text": "0.3",
    "margin": "0.1",
    "geometry": "true",
    "probe_phrases": "put,pick up,place,turn on,push,open,close,left,right,bottom,back,middle,top,front",
}


def suite_to_unnorm_key(suite: str) -> str:
    """Return the OpenVLA/OFT no-noops unnormalization key for a LIBERO suite."""
    try:
        return SUITE_UNNORM_KEYS[suite]
    except KeyError as exc:
        raise ValueError(f"Unknown LIBERO suite {suite!r}; expected one of {sorted(SUITE_UNNORM_KEYS)}") from exc


def source_suite_to_unnorm_key(suite: str) -> str:
    """Return the source OpenVLA unnormalization key for a LIBERO suite."""
    try:
        return SOURCE_UNNORM_KEYS[suite]
    except KeyError as exc:
        raise ValueError(f"Unknown LIBERO suite {suite!r}; expected one of {sorted(SOURCE_UNNORM_KEYS)}") from exc


def load_when_config(path: str | Path) -> dict[str, str]:
    """Load the repository's one-level YAML configs without requiring PyYAML."""
    return read_simple_yaml(Path(path))


def merge_when_defaults(config: dict[str, Any]) -> dict[str, Any]:
    """Return a config dict with UPA-RFAS runnable defaults filled in."""
    merged = dict(OFFICIAL_WHEN_DEFAULTS)
    merged.update({k: v for k, v in config.items() if v is not None and str(v).strip()})
    return merged


def resolve_output_path(value: str | Path) -> Path:
    """Resolve an output path relative to the main repository."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path
