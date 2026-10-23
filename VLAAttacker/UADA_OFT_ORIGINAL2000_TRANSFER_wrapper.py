"""Original OFT-UADA unchanged; isolated configuration and provenance guard."""
import fcntl
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "VLAAttacker"), str(ROOT / "scripts")]
import torch
from dino_transfer_common import config, SOURCE_SHA, BANK_SHA
from UADA_OFT_DINO_wrapper import load_wrapper
from white_patch.UADA_OFT_DINO import ExploringOFTOpenVLAAttacker, require, sha256_file, atomic_json


def main():
    config()  # immutable Exploring source and bank validation
    path = ROOT / "configs/exploring_oft_dino/uada_original2000_transfer_v1.json"
    cfg = json.loads(path.read_text())
    require(cfg["outer_iterations"] == 2000 and cfg["inner_loop"] == 50
            and cfg["checkpoint_outer_iterations"] == [0, 500, 1000, 1500, 2000]
            and cfg["methods"] == ["exploring_oft_uada"] and cfg["structure_gates"] == {},
            "Original baseline protocol drift")
    if "--dry-run" in sys.argv:
        print(json.dumps({"status": "PASS", "original_uada": True, "updates": 100000,
                          "model_loaded": False, "sbatch_submitted": False}))
        return
    require(os.environ.get("SLURM_JOB_ID") and torch.cuda.is_available()
            and torch.cuda.device_count() == 1, "User-submitted single GPU required")
    folder = Path(cfg["output_root"])
    folder.mkdir(parents=True, exist_ok=True)
    manifest = {"config": cfg, "config_sha256": sha256_file(path), "source_uada": SOURCE_SHA,
                "bank": BANK_SHA, "trainer": sha256_file(ROOT / "VLAAttacker/white_patch/UADA_OFT_DINO.py")}
    with (folder / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt = folder / "RUN_MANIFEST.json"
        if receipt.exists():
            require(json.loads(receipt.read_text()) == manifest, "Original training provenance drift")
        else:
            atomic_json(manifest, receipt)
        bank = torch.load(cfg["training_bank"], map_location="cpu", weights_only=False)
        wrapper = load_wrapper(cfg)
        ExploringOFTOpenVLAAttacker(wrapper, folder, resize_patch=False).patchattack_unconstrained_oft(
            bank["records"], cfg, "exploring_oft_uada")


if __name__ == "__main__":
    main()
