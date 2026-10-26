"""Reuse prepared official expert windows; no extraction, rollout or overwrite."""
from __future__ import annotations
import argparse
from common import ROOT, base, preflight, atomic_json, digest, require


def prepare():
    report = preflight()
    bank = base.load_bank()
    schedule = bank["candidates"]
    require(len(schedule) == 250 and all(len(row) == 1 and row[0] in bank["records"] for row in schedule),
            "Shared expert-window sequence changed")
    report.update(status="COMPLETE", record_schedule_sha256=digest(schedule),
                  source_training_bank_signature=bank["signature"], cached_records=len(bank["records"]),
                  selected_windows=250, source="official Long train; original cache read-only")
    atomic_json(report, ROOT / "reports/PREPARATION.json")
    print("PREPARATION_COMPLETE: reused same250 official expert windows; no data/model writes; no GPU", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true", required=True)
    parser.parse_args()
    prepare()
