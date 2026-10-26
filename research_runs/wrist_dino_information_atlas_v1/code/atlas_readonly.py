"""Prevent native local-checkpoint auto-sync writes; identical loads only."""
from __future__ import annotations
from pathlib import Path
from atlas_common import load_json, require, sha256_file

AUTO_MAP = {"AutoConfig": "configuration_prismatic.OpenVLAConfig",
            "AutoModelForVision2Seq": "modeling_prismatic.OpenVLAForActionPrediction"}


def verify_checkpoint(checkpoint, official_root):
    folder = Path(checkpoint)
    require(load_json(folder / "config.json").get("auto_map") == AUTO_MAP,
            "Checkpoint auto_map mismatch; refusing to rewrite shared model")
    hashes = {}
    for filename in ("modeling_prismatic.py", "configuration_prismatic.py"):
        local = folder / filename
        reference = Path(official_root) / "prismatic/extern/hf" / filename
        require(local.is_file() and sha256_file(local) == sha256_file(reference),
                f"Checkpoint logic mismatch: {filename}; refusing native auto-sync")
        hashes[str(local.resolve())] = sha256_file(local)
    return hashes


def install_readonly_loader(wrapper, official_root):
    # The existing wrapper imports/registers the actual native OFT classes.
    # Replace ONLY disk-sync helpers in this job's memory, not inference logic.
    wrapper._try_import_official()
    utils = wrapper._official_openvla_utils
    def verify_only(checkpoint):
        verify_checkpoint(checkpoint, official_root)
    utils.update_auto_map = verify_only
    utils.check_model_logic_mismatch = verify_only
