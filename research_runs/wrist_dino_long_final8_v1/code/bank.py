"""Long source preparation and replay-bank assembly; CPU stages only."""
from __future__ import annotations

import argparse
from pathlib import Path
import torch
from common import (ROOT, SUITES, atomic_json, atomic_torch, clean_records, config, load_json,
                    observation_key, preflight, require, run_signature, sha256_file)


def prepare():
    cfg = config()
    report = preflight()
    actions = torch.cat([r["clean_action"].float().reshape(-1, 7) for r in clean_records()])
    bounds = {"q01": torch.quantile(actions, .01, dim=0).tolist(),
              "q99": torch.quantile(actions, .99, dim=0).tolist(),
              "source": "Long clean300 normalized continuous head, 2400 action slots"}
    require(bounds["q99"][0] > bounds["q01"][0], "Degenerate source X bounds")
    stats = load_json(Path(cfg["evaluation"]["checkpoints"]["libero_10"]) / "dataset_statistics.json")
    action_stats = stats[cfg["evaluation"]["unnorm_keys"]["libero_10"]]["action"]
    raw_zero = [(lo + hi) / 2 for lo, hi in zip(action_stats["q01"][:3], action_stats["q99"][:3])]
    report.update(bounds=bounds, tma_normalized_target=cfg["tma_normalized_target"],
                  tma_environment_target_at_normalized_zero=raw_zero, claim="Fixed normalized XYZ, not physical stopping")
    path = ROOT / "reports/PREPARATION.json"
    if path.exists():
        require(load_json(path) == report, "Preparation changed; preserve old run and use new version")
    else:
        atomic_json(report, path)
    print("PREPARATION_COMPLETE: Long bounds only; no model/render", flush=True)


def assemble():
    signature = run_signature()
    path = ROOT / "data/STATE_BANK.pt"
    receipt = ROOT / "data/STATE_BANK_STATUS.json"
    if receipt.exists():
        old = load_json(receipt)
        require(old["signature"] == signature and old["sha256"] == sha256_file(path), "Existing bank changed")
        return
    observations, clean, clips = {}, [], []
    for record in clean_records():
        uid = "clean:" + observation_key(record["task_id"], record["state_index"], record["decision_index"])
        observations[uid] = record
        clean.append(uid)
    all_episodes = []
    for task in range(10):
        shard_path = ROOT / "data" / f"replay_task_{task:02d}.pt"
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        require(shard["signature"] == signature and shard["complete"] and shard["task_id"] == task,
                "Incomplete/wrong replay shard")
        episodes = shard["episodes"]
        require(len(episodes) == 5, "Replay must cover all five original source starts")
        eligible = [e for e in episodes if e["success"]] or episodes
        all_episodes.extend([{k: v for k, v in e.items() if k not in {"records", "clips"}} for e in episodes])
        for episode in eligible:
            require(episode["state_index"] < 40, "Test state in replay")
            for uid, record in episode["records"].items():
                require(uid.startswith("replay:") and record["source_suite"] == "libero_10", "Wrong replay origin")
                require(uid not in observations, "Duplicate or clean/replay state collision")
                observations[uid] = record
            clips.extend(episode["clips"])
    require(len(clean) == 300 and clips and {r["task_id"] for r in clips} == set(range(10)), "Bank coverage")
    require(all(uid in observations for clip in clips for uid in clip["uids"]), "Clip references missing observations")
    value = {"signature": signature, "clean": clean, "observations": observations, "clips": clips,
             "episodes": all_episodes, "source_suite": "libero_10", "selection": "survivors_else_all_per_task"}
    atomic_torch(value, path)
    atomic_json({"status": "COMPLETE", "signature": signature, "sha256": sha256_file(path),
                 "clean_states": 300, "distinct_teacher_states": len(observations), "clips": len(clips),
                 "teacher_updates_per_family": len(observations) * config()["teacher_updates"]}, receipt)
    print(f"STATE_BANK_COMPLETE states={len(observations)} clips={len(clips)}", flush=True)


def merge_teachers():
    signature = run_signature()
    for family in ("uada", "tma"):
        records = {}
        for shard in range(4):
            value = torch.load(ROOT / "teachers" / f"{family}_{shard}.pt", map_location="cpu", weights_only=False)
            require(value["complete"] and value["signature"] == signature and value["family"] == family,
                    "Incomplete/wrong teacher shard")
            require(not set(records) & set(value["records"]), "Overlapping teacher shards")
            records.update(value["records"])
        bank = torch.load(ROOT / "data/STATE_BANK.pt", map_location="cpu", weights_only=False)
        require(set(records) == set(bank["observations"]), "Teacher bank does not cover the actual states")
        path = ROOT / "teachers" / f"{family}_BANK.pt"
        atomic_torch({"signature": signature, "family": family, "records": records}, path)
        atomic_json({"status": "COMPLETE", "signature": signature, "sha256": sha256_file(path),
                     "states": len(records), "valid_teachers": sum(int(r["valid"]) for r in records.values()),
                     "optimizer_updates": 100 * len(records)}, path.with_suffix(".json"))
    print("TEACHER_BANKS_COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prepare", "assemble", "merge-teachers"), required=True)
    {"prepare": prepare, "assemble": assemble, "merge-teachers": merge_teachers}[parser.parse_args().stage]()
