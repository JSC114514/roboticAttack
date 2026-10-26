"""Read the Spatial bank once; release its 1024 center references before model loading."""
from __future__ import annotations
from collections import defaultdict, deque
import torch
from common import ROOT, atomic_json, atomic_torch, atomic_parquet, lineage, preflight, run_signature, sha256_file, require

def balanced(records, limit, excluded_episodes=()):
    queues = defaultdict(deque); excluded = set(excluded_episodes)
    for r in records:
        if r["episode_id"] not in excluded: queues[int(r["task_id"])].append(r)
    result = []
    while len(result) < limit and any(queues.values()):
        for task in sorted(queues):
            if queues[task] and len(result) < limit: result.append(queues[task].popleft())
    return result

def split_records(records, train_count=20, validation_count=20):
    train = balanced(records, train_count)
    used = {r["uid"] for r in train}; episodes = {r["episode_id"] for r in train}
    validation = balanced([r for r in records if r["uid"] not in used], validation_count, episodes)
    return train, validation

def prepare():
    report = preflight(bank=True); signature = run_signature()
    final = ROOT / "data/EXPERT_CLOSE_BANK.pt"
    receipt = ROOT / "reports/PREPARATION.json"
    if receipt.exists():
        from common import prepared
        prepared(); print("PREPARATION_ALREADY_COMPLETE"); return
    b = torch.load(lineage()["bank_path"], map_location="cpu", weights_only=False)
    order = [row[0] for row in b["candidates"]]
    require(len(order) == len(set(order)) == 250, "Wrong Spatial training pool")
    records = [b["records"][u] for u in order if (b["records"][u]["expert_actions"][0, :, 6] < .5).any()]
    slots = sum(int((r["expert_actions"][0, :, 6] < .5).sum()) for r in records)
    require(len(records) == 150 and slots == 1077, "Previously verified expert-close counts changed")
    atomic_torch({"signature": signature, "records": records, "original_pool": 250}, final)
    del b
    atomic_parquet(__import__("common").history_rows(), ROOT / "data/HISTORICAL_ROLLOUTS.parquet")
    from resources import snapshot
    report.update(expert_close_observations=150, expert_close_slots=slots,
                  prepared_artifacts={str(p.relative_to(ROOT)): sha256_file(p) for p in
                    [final, ROOT / "data/HISTORICAL_ROLLOUTS.parquet"]}, **snapshot())
    atomic_json(report, receipt)
    print("PREPARED Spatial150 expert-close observations; clean intersection deferred to native GPU", flush=True)
