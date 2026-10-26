"""Reuse only completed binary-goal QA unaffected by the drawer getter repair."""
from __future__ import annotations
import ast
from pathlib import Path
from loop_core import ROOT, SUITES, cfg, read_json, require, sha, stage_done

PARENT = ROOT.parent / "wrist_dino_closedloop_geometry_v1"
PARENT_SIGNATURE = "429f36681f37d984c55b02bb2b66ff68d84b24b6dbf8afb07845829968c343ad"


def _measure_ast(path):
    tree = ast.parse(Path(path).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TaskProgress")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "measure")
    return ast.dump(method, include_attributes=False)


def inventory(external):
    old_preparation = PARENT / "reports/PREPARATION.json"
    preparation = read_json(old_preparation)
    require(preparation["signature"] == PARENT_SIGNATURE, "Repair parent has changed")
    require(preparation["external_sources"] == external, "External model/simulator runtime differs from passed QA")
    old_cfg, new_cfg = read_json(PARENT / "config.json"), cfg()
    require({k: v for k, v in old_cfg.items() if k != "experiment_id"}
            == {k: v for k, v in new_cfg.items() if k != "experiment_id"}, "Repair changed the scientific configuration")
    files = {str(old_preparation): sha(old_preparation)}
    for relative, expected in preparation["sources"].items():
        path = PARENT / relative
        require(sha(path) == expected, "Original submitted source was modified: " + relative)
        files[str(path)] = expected
    for relative in ("code/geometry.py", "code/branch_state.py", "code/native_predictor.py", "code/closedloop.py",
                     "vendor/simulator_snapshot.py"):
        require(sha(ROOT / relative) == sha(PARENT / relative), "QA-relevant runtime changed beyond the getter repair")
    require(_measure_ast(ROOT / "code/task_progress.py") == _measure_ast(PARENT / "code/task_progress.py"),
            "Binary-goal progress computation changed")
    reused = {}
    for suite in SUITES:
        status = PARENT / "qa" / suite / "STATUS.json"
        if not status.exists():
            continue
        receipt = read_json(status)
        if receipt.get("status") != "COMPLETE":
            continue
        require(receipt["signature"] == PARENT_SIGNATURE, "Original QA signature mismatch")
        for relative, expected in receipt["artifacts"].items():
            path = PARENT / relative
            require(path.resolve().is_relative_to(PARENT.resolve()) and sha(path) == expected, "Original QA artifact changed")
            files[str(path)] = expected
        details = PARENT / "qa" / suite / "ENGINEERING_QA.json"
        qa = read_json(details)
        # The repair changes only unary joint extraction. Never reuse a check
        # that executed that path, or a sparse schema without the concrete goals.
        require(qa["task_goal_schema"] and all(len(g) == 3 for g in qa["task_goal_schema"]),
                "Cannot reuse QA that exercised articulated unary goals")
        require(qa["all_candidates_exercised"] == 7 and qa["identity_replay"]["commit_action_max_error"] == 0
                and qa["identity_replay"]["commit_state_max_error"] == 0, "Original engineering QA is incomplete")
        files[str(status)] = sha(status)
        reused[suite] = {"status_path": str(status), "metadata_path": str(details),
                         "original_signature": PARENT_SIGNATURE, "executed_in_repair": False}
    costs = []
    for path in sorted((PARENT / "reports/WORKER_COSTS").glob("*.json")):
        cost = read_json(path)
        require(cost["signature"] == PARENT_SIGNATURE and cost["stage"] == "qa", "Unexpected parent computation to reuse")
        costs.append(cost); files[str(path)] = sha(path)
    return {"parent": str(PARENT), "signature": PARENT_SIGNATURE, "qa": reused, "files": files,
            "prior_gpu_worker_hours_including_failed_goal": sum(c["elapsed_s"] for c in costs if c["gpu_worker"]) / 3600}


def verify_inventory(value):
    require(value["signature"] == PARENT_SIGNATURE, "Wrong repair lineage")
    for path, expected in value["files"].items():
        require(sha(path) == expected, "Reused QA/source/cost evidence changed: " + path)


def qa_done(suite):
    own = ROOT / "qa" / suite / "STATUS.json"
    if stage_done(own):
        return True
    path = ROOT / "reports/PREPARATION.json"
    if not path.exists():
        return False
    report = read_json(path)
    reused = report.get("repair_reuse", {}).get("qa", {}).get(suite)
    if reused is None:
        return False
    verify_inventory(report["repair_reuse"])
    return True


def metadata_path(suite):
    own = ROOT / "qa" / suite / "STATUS.json"
    if stage_done(own):
        return own.parent / "ENGINEERING_QA.json"
    require(qa_done(suite), "No verified QA available for " + suite)
    return Path(read_json(ROOT / "reports/PREPARATION.json")["repair_reuse"]["qa"][suite]["metadata_path"])
