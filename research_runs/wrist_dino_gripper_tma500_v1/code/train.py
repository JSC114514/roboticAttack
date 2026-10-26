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


def train(method):
    started = time.monotonic()
    cfg, prep = config(), preparation()
    require_stage_gpu("training")
    pool, receipt = read_training_pool()
    bank = torch.load(cfg["training_bank"],map_location="cpu",weights_only=False)
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    wrapper = load_wrapper(runtime_config(cfg,"libero_object"),"libero_object")
    identity = {"preparation_signature":digest(prep),"pool_sha256":receipt["pool_sha256"],
                "actual_runtime_source":source_manifest(wrapper)}
    ExploringOFTGripperTMA(OFTEngine(wrapper)).patchattack_unconstrained(bank["records"],pool,cfg,method,identity,
                                                                    session_started=started)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method",choices=config()["methods"],required=True)
    train(parser.parse_args().method)
