"""CPU-only successor cache, preserving the parent's actual candidate-zero prefix."""
from __future__ import annotations
import argparse
import gc
import time
from pathlib import Path
import numpy as np
from common import (ROOT, config, lineage, require, load_json, run_signature, sha256_file,
                    atomic_json, atomic_torch, atomic_parquet, preflight, prepared)
from rlds_source import candidate_schedule, normalized_expert, episode_iterator, resize_policy_jpeg, text_value

def pair_schedule(candidates, episodes, count=500, gap=8):
    lookup = {e["episode_id"]: e for e in episodes}
    result = []
    for row in candidates[:count]:
        uid = row[0]; eid, step = uid.rsplit(":", 1); step = int(step)
        require(step >= 0 and step + 8 <= lookup[eid]["length"], "Parent action window invalid")
        query = f"{eid}:{step+gap}" if step + gap + 8 <= lookup[eid]["length"] else None
        result.append({"uid": uid, "query_uid": query, "episode_id": eid, "step_index": step})
    require(len(result) == count, "Candidate prefix incomplete")
    return result

def source_subset():
    import torch
    p = lineage()["parent"]
    require(sha256_file(p["bank_path"]) == p["bank_sha256"], "Parent bank bytes changed")
    index = load_json(Path(p["root"]) / "data/DATASET_INDEX.json")
    b = torch.load(p["bank_path"], map_location="cpu", weights_only=False)
    require(b["signature"] == p["signature"], "Parent bank signature changed")
    expected = candidate_schedule(index["episodes"], 2000, 4, config()["data_seed"])
    require(b["candidates"] == expected, "Actual parent candidate schedule differs from regenerated schedule")
    pairs = pair_schedule(b["candidates"], index["episodes"])
    wanted = {r[k] for r in pairs for k in ("uid", "query_uid") if r[k] is not None}
    records = {uid: b["records"][uid] for uid in wanted if uid in b["records"]}
    statistics = b["statistics"]
    del b; gc.collect()
    # Return unused native allocator pages before TensorFlow import.
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass
    missing = sum(r["query_uid"] is None for r in pairs)
    require(missing == config()["missing_successor_expected"], "Actual missing-successor count differs from approved14")
    return index, pairs, wanted, records, statistics

def validate_records(records, pairs):
    import torch
    for row in pairs:
        current = records[row["uid"]]
        require(current["episode_id"] == row["episode_id"] and current["step_index"] == row["step_index"], "Current window identity changed")
        for key in ("uid", "query_uid"):
            if row[key] is None:
                continue
            r = records[row[key]]; a = r["expert_actions"]
            require(a.shape == (1, 8, 7) and torch.isfinite(a).all(), "Bad expert action labels")
            for image in ("full_image", "wrist_image"):
                x = r["observation"][image]
                require(x.shape == (224, 224, 3) and x.dtype == np.uint8, "Native source image changed")
            require(r["observation"]["state"].shape == (8,) and np.isfinite(r["observation"]["state"]).all(), "Invalid expert proprio")
        if row["query_uid"] is not None:
            q = records[row["query_uid"]]
            require(q["episode_id"] == current["episode_id"] and q["step_index"] - current["step_index"] == 8
                    and q["task_name"] == current["task_name"], "Cross-episode/gap/language mismatch")

def prepare():
    import torch
    from common import history_rows
    if (ROOT / "reports/PREPARATION.json").exists():
        prepared(); print("PAIR_CACHE_ALREADY_PREPARED"); return
    start = time.monotonic(); report = preflight(bank=True)
    index, pairs, wanted, records, statistics = source_subset()
    missing = wanted - records.keys()
    episode_map = {e["episode_id"]: e for e in index["episodes"]}
    language_ids = {v: i for i, v in enumerate(sorted({e["language"] for e in index["episodes"]}))}
    requested = {}
    for uid in missing:
        eid, step = uid.rsplit(":", 1); requested.setdefault(eid, []).append(int(step))
    if requested:
        for number, episode in enumerate(episode_iterator()):
            eid = text_value(episode["tfds_id"])
            if eid not in requested:
                continue
            steps, meta = episode["steps"], episode_map[eid]
            require(int(steps["action"].shape[0]) == meta["length"], "RLDS episode length drift")
            require(text_value(steps["language_instruction"][0]) == meta["language"], "RLDS language drift")
            acts = normalized_expert(steps["action"].numpy(), statistics)
            for step in requested[eid]:
                uid = f"{eid}:{step}"
                records[uid] = {"uid": uid, "episode_id": eid, "step_index": step,
                    "task_id": language_ids[meta["language"]], "task_name": meta["language"],
                    "task_id_definition": "sorted official language, not simulator benchmark order",
                    "source_suite": "libero_10", "source_split": "train", "file_path": meta["file_path"],
                    "expert_actions": torch.from_numpy(acts[step:step+8].copy()).unsqueeze(0),
                    "observation": {"full_image": resize_policy_jpeg(steps["observation"]["image"][step]),
                                    "wrist_image": resize_policy_jpeg(steps["observation"]["wrist_image"][step]),
                                    "state": steps["observation"]["state"][step].numpy().astype(np.float32)}}
            if (number+1) % 25 == 0:
                print(f"PAIR_CACHE episode={number+1}/379 records={len(records)}/{len(wanted)}", flush=True)
    require(set(records) == wanted, "Successor cache incomplete")
    validate_records(records, pairs)
    bank = {"signature": report["signature"], "parent_signature": report["parent_signature"],
            "pairs": pairs, "records": records, "statistics": statistics, "gap": 8,
            "source": "expert successor observations; NOT attacked simulator successors"}
    atomic_torch(bank, ROOT / "data/PAIR_BANK.pt")
    atomic_parquet(history_rows(), ROOT / "data/HISTORICAL_ROLLOUTS.parquet")
    atomic_json({"pairs": pairs, "gap": 8, "complete_pairs": 486, "fallback_rows": 14,
                 "current_source": "actual parent candidate0 first500", "signature": report["signature"]}, ROOT / "data/PAIR_INDEX.json")
    report.update(elapsed_s=time.monotonic()-start, cached_records=len(records),
        reused_cached_records=len(wanted)-len(missing), decoded_successors=len(missing), complete_pairs=486,
        prepared_artifacts={p: sha256_file(ROOT / p) for p in ("data/PAIR_BANK.pt", "data/HISTORICAL_ROLLOUTS.parquet", "data/PAIR_INDEX.json")})
    atomic_json(report, ROOT / "reports/PREPARATION.json")
    print("PAIR_CACHE_PREPARED: 486 paired +14 native fallback; no model/GPU labels", flush=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--audit-source", action="store_true"); a = p.parse_args()
    if a.audit_source:
        from resources import snapshot
        index, pairs, wanted, records, _ = source_subset()
        print({"actual_parent_prefix_checked": True, "pairs": len(pairs), "complete_pairs": sum(p["query_uid"] is not None for p in pairs),
               "unique_required_records": len(wanted), "already_cached": len(records), "models_loaded": False, "files_written": False,
               **snapshot()})
    else:
        prepare()
