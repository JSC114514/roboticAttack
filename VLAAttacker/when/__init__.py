"""WHEN / UPA-RFAS experiment helpers.

This package is intentionally separate from ``VLAAttacker.multiview``.  The
legacy M-series keeps its pixel-values/action-head losses, while WHEN follows
the UPA-RFAS raw-RGB feature/attention/semantic patch recipe.
"""

from .config import (
    DIGITAL_PROTOCOLS,
    LIBERO_SUITES,
    PROJECT_ROOT,
    load_when_config,
    suite_to_unnorm_key,
)
from .patch_io import patch_sha256, read_patch_manifest, require_patch_manifest
from .transforms import apply_digital_patch_to_observation, assert_digital_protocol_effect

__all__ = [
    "DIGITAL_PROTOCOLS",
    "LIBERO_SUITES",
    "PROJECT_ROOT",
    "apply_digital_patch_to_observation",
    "assert_digital_protocol_effect",
    "load_when_config",
    "patch_sha256",
    "read_patch_manifest",
    "require_patch_manifest",
    "suite_to_unnorm_key",
]
