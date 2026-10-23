#!/usr/bin/env python3
"""One manually submitted A800 per method, with same source-only state schedule."""
from __future__ import annotations

import argparse
import time
import torch

from gripper_common import config, digest, preparation, read_training_pool, require_stage_gpu, runtime_config, source_manifest
from engine import OFTEngine
from model_runtime_copy import load_wrapper
from TMA_OFT import ExploringOFTGripperTMA
from long_source import validate_records


def train(method):
    started = time.monotonic()
    cfg, prep = config(), preparation()
    require_stage_gpu("training")
    pool, receipt = read_training_pool()
    bank = torch.load(cfg["training_bank"],map_location="cpu",weights_only=False)
    validate_records(bank["records"],cfg)
    if bank["training_suite"] != cfg["training_suite"] or bank["preparation_signature"] != digest(prep):
        raise RuntimeError("Long bank training suite/preparation changed")
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    suite = cfg["training_suite"]
    wrapper = load_wrapper(runtime_config(cfg,suite),suite)
    identity = {"preparation_signature":digest(prep),"pool_sha256":receipt["pool_sha256"],
                "source_bank_sha256":receipt["source_bank_sha256"],"training_suite":suite,
                "actual_runtime_source":source_manifest(wrapper)}
    ExploringOFTGripperTMA(OFTEngine(wrapper)).patchattack_unconstrained(bank["records"],pool,cfg,method,identity,
                                                                    session_started=started)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method",choices=config()["methods"],required=True)
    train(parser.parse_args().method)
