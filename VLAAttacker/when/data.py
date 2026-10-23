"""Path and requirement checks for WHEN experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import DATA_ROOT, LIBERO_ROOT, LIBERO_SUITES, MODEL_ROOT, OFT_MODEL_PATHS, SOURCE_MODEL_CANDIDATES
from .config import SUITE_UNNORM_KEYS, UPA_RFAS_REFERENCE


def _exists(path: Path) -> dict[str, Any]:
    return {"path": str(path), "exists": path.exists(), "is_dir": path.is_dir()}


def check_model_paths() -> dict[str, Any]:
    """Check local model directories without importing torch/transformers."""
    source = {name: _exists(path) for name, path in SOURCE_MODEL_CANDIDATES.items()}
    oft = {suite: _exists(path) for suite, path in OFT_MODEL_PATHS.items()}
    return {
        "model_root": _exists(MODEL_ROOT),
        "source": source,
        "oft": oft,
        "base_openvla": _exists(MODEL_ROOT / "openvla-7b"),
    }


def check_data_paths() -> dict[str, Any]:
    """Check LIBERO RLDS/TFDS data directories expected by the dataloader."""
    roots = [
        DATA_ROOT / "openvla-main" / "dataset",
        DATA_ROOT / "dataset",
    ]
    datasets = {}
    for suite in LIBERO_SUITES:
        key = SUITE_UNNORM_KEYS[suite]
        datasets[suite] = {
            str(root): _exists(root / key / "1.0.0")
            for root in roots
        }
    return {
        "data_root": _exists(DATA_ROOT),
        "candidate_dataset_roots": [_exists(root) for root in roots],
        "datasets": datasets,
        "explicit_libero_long": {
            str(root): _exists(root / "libero_long_no_noops")
            for root in roots
        },
    }


def check_reference_repo() -> dict[str, Any]:
    """Check that the local WHEN feature-space reference has the required files."""
    required = [
        "white_patch/feature_space.py",
        "white_patch/appply_random_transform.py",
    ]
    return {
        "root": _exists(UPA_RFAS_REFERENCE),
        "required_files": {
            rel: _exists(UPA_RFAS_REFERENCE / rel)
            for rel in required
        },
    }


def collect_requirement_status() -> dict[str, Any]:
    """Collect all lightweight path checks used by scripts/when/check_requirements.py."""
    return {
        "models": check_model_paths(),
        "data": check_data_paths(),
        "reference": check_reference_repo(),
        "libero_root": _exists(LIBERO_ROOT),
    }


def missing_requirements(status: dict[str, Any] | None = None) -> list[str]:
    """Return human-readable blockers from a collected requirement status."""
    status = status or collect_requirement_status()
    missing: list[str] = []
    source = status["models"]["source"]
    if not source["libero_10"]["exists"] and not source["libero_long"]["exists"]:
        missing.append("Missing source OpenVLA LIBERO-10/Long checkpoint under /home/caijiasheng25/models.")
    for suite, item in status["models"]["oft"].items():
        if not item["exists"]:
            missing.append(f"Missing OpenVLA-OFT checkpoint for {suite}: {item['path']}")
    datasets = status["data"]["datasets"]
    for suite, roots in datasets.items():
        if not any(item["exists"] for item in roots.values()):
            missing.append(f"Missing RLDS/TFDS data for {suite}.")
    required_files = status["reference"]["required_files"]
    for rel, item in required_files.items():
        if not item["exists"]:
            missing.append(f"UPA-RFAS reference file missing: {rel}")
    return missing
