"""CPU audit allowing immutable completed shards across the replay-boundary fix."""
from __future__ import annotations
import ast,hashlib,json
from pathlib import Path
from common import ROOT,atomic_json,read_json,require,resources,sha,signature

def resource_migration_matches(relative,original,current):
    """Only the declared single-A800 scheduling change may bypass a source hash."""
    if relative=="slurm/environment.sh":
        return current==original.replace("  export MUJOCO_EGL_DEVICE_ID=",
                   "  OFT_STAGE_GPU=${OFT_STAGE_GPU^^}\n  export MUJOCO_EGL_DEVICE_ID=")
    if relative=="config.json":
        expected=json.loads(original)
        for stage in ("cache","train","validate","heldout"):
            require(expected["resources"][stage]["gpu_type"]=="3090","Unexpected source GPU")
            expected["resources"][stage].update(gpu_type="a800",partition="a800",qos="a800")
        return json.loads(current)==expected
    if relative in {"slurm/02_cache.sbatch","slurm/03_train.sbatch",
                    "slurm/04_validate.sbatch","slurm/06_heldout.sbatch"}:
        expected=original.replace("#SBATCH --partition=rtx",
                                  "#SBATCH --partition=a800\n#SBATCH --qos=a800")
        expected=expected.replace("#SBATCH --gres=gpu:3090:1","#SBATCH --gres=gpu:a800:1")
        return current==expected
    return False

def audit_reuse(proof_path):
    import torch
    proof_path=Path(proof_path).resolve(); proof=read_json(proof_path); origin=proof["origin_signature"]
    # The witness was recorded only after checking both original source file hashes.
    for relative,expected in proof["components"].items():
        nodes={n.name:n for n in ast.parse((ROOT/relative).read_text()).body
               if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        for name,fingerprint in expected.items():
            actual=hashlib.sha256(ast.dump(nodes[name],include_attributes=False).encode()).hexdigest()
            require(actual==fingerprint,"Capture/training input logic changed: "+relative+":"+name)
    allowed={str((ROOT/p).resolve()) for p in ("code/cache.py","code/common.py","code/train.py",
             "code/evaluate.py","code/pipeline.py","README.md","PLAN.md")}
    for path,expected in proof["source_hashes"].items():
        if path in allowed or Path(path).parent==(ROOT/"tests").resolve():continue
        relative=Path(path).relative_to(ROOT).as_posix() if Path(path).is_relative_to(ROOT) else None
        original=proof.get("resource_migration_originals",{}).get(relative)
        if original is not None:
            require(hashlib.sha256(original.encode()).hexdigest()==expected,"Original resource witness hash mismatch")
            require(resource_migration_matches(relative,original,Path(path).read_text()),
                    "Change exceeds the declared single-A800 resource migration: "+path)
            continue
        require(Path(path).is_file() and sha(path)==expected,"Cache reuse source changed: "+path)
    previous=ROOT/"reports/REVISIONS"/origin/"reports/PREPARE_COMPLETE.json"
    if not previous.is_file():previous=ROOT/"reports/PREPARE_COMPLETE.json"
    previous=read_json(previous)
    require(previous["signature"]==origin,"Original preparation receipt unavailable")
    require(sha(ROOT/"data/INDEX.json")==previous["artifacts"]["data/INDEX.json"],"Selected observation index changed")
    index=read_json(ROOT/"data/INDEX.json"); entries={}; observations=0
    for receipt in sorted((ROOT/"data/cache").glob("*/*/task_*.json")):
        if receipt.name.endswith(".cost.json"):continue
        old=read_json(receipt)
        if old.get("signature")==signature():continue
        require(old.get("status")=="COMPLETE" and old.get("signature")==origin,"Unexpected incomplete/cache revision")
        for relative,expected in old["artifacts"].items():
            require(sha(ROOT/relative)==expected,"Original cache artifact changed: "+relative)
        shard=torch.load(receipt.with_suffix(".pt"),map_location="cpu",weights_only=False)
        require(shard["signature"]==origin,"Shard/receipt origin mismatch")
        expected=[r["state_id"] for r in index if r["split"]==old["split"] and r["suite"]==old["suite"]
                  and r["task_id"]==old["task_id"]]
        rows=shard["records"]
        require(len(rows)==old["frames"]==len(expected) and sorted(r["state_id"] for r in rows)==sorted(expected),
                "Cache observation membership changed")
        require(all(r["native_camera_alignment_verified"] for r in rows),"Cache lacks native alignment evidence")
        if old["split"]=="train":
            for row in rows:
                require(set(row["target_features"])=={"translation_plus","translation_minus","bearing_plus","bearing_minus"},
                        "Incomplete cached target features")
                require(all(tuple(t.shape)==(256,1024) and torch.isfinite(t).all()
                            for t in row["target_features"].values()),"Invalid cached features")
        relative=receipt.relative_to(ROOT).as_posix()
        entries[relative]={"origin_signature":origin,"sha256":sha(receipt),"frames":len(rows)}
        observations+=len(rows); del shard,rows
    report={"status":"AUDITED","target_signature":signature(),"origin_signature":origin,
            "proof_file":str(proof_path.relative_to(ROOT)),"proof_sha256":sha(proof_path),
            "receipts":entries,"reused_shards":len(entries),"reused_observations":observations,
            "underlying_artifacts_modified":False,"capture_and_feature_logic_unchanged":True,
            "prefix_probe_required_before_model_loading":True,**resources()}
    atomic_json(report,ROOT/"reports/CACHE_REUSE.json")
    return report

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument("--proof",required=True)
    args=parser.parse_args();result=audit_reuse(args.proof)
    print({k:result[k] for k in ("status","target_signature","reused_shards","reused_observations")})
