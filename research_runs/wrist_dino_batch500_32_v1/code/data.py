"""CPU preparation: reuse expert windows/statistics, never relabel or collect."""
from __future__ import annotations
from common import ROOT, atomic_json, load_json, load_bank, preflight, require


def prepare():
    report = preflight()
    path = ROOT / "reports/PREPARATION.json"
    if path.exists():
        require(load_json(path)["signature"] == report["signature"], "Existing preparation changed")
    atomic_json(report, path)
    bank = load_bank()
    atomic_json({"status": "COMPLETE", "signature": report["signature"], "source": "read-only official Long train",
                 "candidate_draws": len(bank["candidates"]), "cached_records": len(bank["records"]),
                 "center_reference_draws": len(bank["center_reference_uids"]),
                 "data_labels_modified": False, "policy_labels_generated": False}, ROOT / "reports/DATA_STATUS.json")
    print("BATCH500_DATA_COMPLETE: same Long250 pool / 1024 train-only mean references; no GPU", flush=True)


if __name__ == "__main__":
    prepare()
