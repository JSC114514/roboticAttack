"""Organize batches from the same read-only official training-record pool."""
from __future__ import annotations
import argparse
from common import ROOT, METHODS, base, preflight, atomic_json, digest, require, batch_uids


def prepare():
    report = preflight()
    bank = base.load_bank()
    schedule = bank["candidates"]
    require(len(schedule) == 250 and len({row[0] for row in schedule}) == 250, "Source pool changed")
    method_schedules = {m: [batch_uids(schedule, outer, m) for outer in range(1, 251)] for m in METHODS}
    require(method_schedules[METHODS[1]] == method_schedules[METHODS[2]], "Generator batches no longer paired")
    report.update(status="COMPLETE", record_schedule_sha256=digest(schedule),
                  method_schedule_sha256={m: digest(rows) for m, rows in method_schedules.items()},
                  source_training_bank_signature=bank["signature"], cached_records=len(bank["records"]),
                  selected_source_pool=250, source="official Long train; same original cache read-only")
    atomic_json(report, ROOT / "reports/PREPARATION.json")
    print("PREPARATION_COMPLETE: same250 expert pool; Value B1/generator image B8; no data/model writes or GPU", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true", required=True)
    parser.parse_args()
    prepare()
