import ast
from pathlib import Path
import numpy as np
import torch
import pandas as pd
from common import ROOT,cfg
from geometry import paired_deltas
from engine import signed_projection
from analyze import casr_table

def test_frozen_split_and_budget():
    c=cfg()
    assert c["methods"]==["translation","bearing"]
    assert c["epochs"]*c["updates_per_epoch"]==48000
    for suite,value in c["task_splits"].items():
        assert len(value["train"])==6 and len(value["validation"])==len(value["heldout"])==2
        assert sorted(value["train"]+value["validation"]+value["heldout"])==list(range(10))

def test_matched_displacement_and_invariants():
    deltas,meta=paired_deltas([.2,.1,.9],[0.,0.,1.])
    norms=[np.linalg.norm(v) for v in deltas.values()]
    assert max(norms)-min(norms)<1e-9 and max(norms)<=.03+1e-12
    assert all(v[2]==0 for v in deltas.values())
    assert deltas["translation_plus"][0]>0 and deltas["translation_minus"][0]<0

def test_signed_projection_has_declared_endpoints():
    negative=torch.zeros(256,4); positive=torch.ones(256,4)
    at_negative=signed_projection(negative[None],negative,positive)
    at_mid=signed_projection((negative+.5)[None],negative,positive)
    at_positive=signed_projection(positive[None],negative,positive)
    assert np.isclose(float(at_negative[1]),0) and np.isclose(float(at_mid[1]),.5) and np.isclose(float(at_positive[1]),1)
    assert np.isclose(float(at_positive[0]),0)

def test_training_has_no_action_objective_or_eot():
    text=(ROOT/"code/train.py").read_text(); tree=ast.parse(text)
    assert "action_loss" not in text and "apply_random" not in text and "EOT" not in text
    assert any(isinstance(n,ast.Call) and getattr(n.func,"id",None)=="signed_projection" for n in ast.walk(tree))

def test_resources_are_stage_specific_and_single_gpu():
    resources=cfg()["resources"]
    assert resources["train"]["host_memory_gib"]==16
    assert resources["cache"]["host_memory_gib"]==resources["validate"]["host_memory_gib"]==resources["heldout"]["host_memory_gib"]==12
    assert max(v["gpus"] for v in resources.values())==1

def test_heldout_is_downstream_of_sealed_selection():
    source=(ROOT/"code/pipeline.py").read_text()
    assert source.index('"select"') < source.index('"heldout"')
    assert 'afterok:' in source

def test_casr_is_conditioned_on_paired_clean_success():
    rows=[]
    for task,clean,attacked in [(0,1,0),(1,1,1),(2,0,0)]:
        rows += [{"suite":"s","task_id":task,"state_index":48,"condition":"clean","success":clean},
                 {"suite":"s","task_id":task,"state_index":48,"condition":"translation","success":attacked}]
    result=casr_table(pd.DataFrame(rows))[0]
    assert result["eligible_clean_starts"]==2 and result["paired_failure_switches"]==1
    assert np.isclose(result["task_equal_casr"],.5)
