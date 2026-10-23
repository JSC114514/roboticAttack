"""Report observed chain diagnostics without inventing a causal root cause."""
from __future__ import annotations
import pandas as pd
from common import ROOT,METHODS,SUITES,atomic_json,atomic_parquet,load_json

def report():
    training=[];steps=[];replays=[];failures=[]
    for method in METHODS:
        p=ROOT/"training"/method
        for name,out in [("diagnostics.parquet",training),("inner_steps.parquet",steps)]:
            path=p/name
            if path.exists():out.append(pd.read_parquet(path))
            else:failures.append(str(path))
    for s in SUITES:
        for t in range(10):
            p=ROOT/"evaluation"/s/f"task_{t:02d}_REPLAY_STATUS.json"
            if not p.exists() or load_json(p).get("status")!="COMPLETE":failures.append(str(p));continue
            replays.append(pd.read_parquet(p.with_name(f"task_{t:02d}_REPLAY.parquet")))
    out=ROOT/"results"
    if training:
        df=pd.concat(training,ignore_index=True);atomic_parquet(df,out/"INTERNAL_TRAINING.parquet")
        columns=[c for c in df if c not in ["outer_iteration","inner_iteration","seed"] and pd.api.types.is_numeric_dtype(df[c])]
        df.groupby("method")[columns].mean().to_csv(out/"INTERNAL_TRAINING_MEANS.csv")
    if steps:
        df=pd.concat(steps,ignore_index=True);atomic_parquet(df,out/"INNER_STEPS.parquet")
        df.groupby("method").agg(inner_steps=("main_loss","size"),
            mean_rgb_gradient=("gradient_rgb_l2","mean"),mean_rgb_update_rms=("rgb_update_rms","mean"),
            mean_quantized_changed_fraction=("quantized_changed_fraction","mean"),
            mean_saturation=("saturated_pixel_fraction","mean"),
            non_descent_steps=("gradient_dot_actual_step",lambda x:int((x>0).sum()))).to_csv(out/"INNER_STEP_SUMMARY.csv")
    if replays:
        df=pd.concat(replays,ignore_index=True);atomic_parquet(df,out/"SAME_STATE_REPLAY.parquet")
        cols=[c for c in df if c not in ["task_id","state_index","decision_block"] and pd.api.types.is_numeric_dtype(df[c])]
        df.groupby(["suite","condition"])[cols].mean().to_csv(out/"SAME_STATE_REPLAY_MEANS.csv")
    status={"status":"COMPLETE" if not failures else "INCOMPLETE_OR_FAILED", "missing_or_failed":failures,
            "interpretation":"Observed gradient/feature/action links; do not infer root cause from a mean or a correlation"}
    atomic_json(status,out/"DIAGNOSTIC_STATUS.json")
    lines=["# Internal diagnostics", "", f"Status: {status['status']}", "",
           "Training probes use exact per-step training records and EOT draws. Inner-step logs cover actual optimizer updates.",
           "LLM layer metrics use all sequence tokens; action_hidden metrics use the native action-token hidden states.",
           "Replay compares clean, PNG and DINO restoration on the same observed state; different rollout states are not treated as matched.",
           "All layer distances are descriptive. DINO restoration is an off-manifold intervention, not proof of the sole cause of task failure.",
           "A DINO representation change alone is not labelled successful task-relevant information destruction.", "",
           "Read INTERNAL_TRAINING_MEANS.csv, INNER_STEP_SUMMARY.csv and SAME_STATE_REPLAY_MEANS.csv alongside paired CASR contrasts."]
    if failures:lines += ["", "Missing or failed diagnostic assets:", *["- "+p for p in failures]]
    (out/"INTERNAL_DIAGNOSTICS.md").write_text("\n".join(lines)+"\n")
    result=out/"STATUS.json"
    if failures and result.exists():
        x=load_json(result);x["status"]="RESULTS_COMPLETE_DIAGNOSTICS_INCOMPLETE";atomic_json(x,result)

if __name__=="__main__":report()
