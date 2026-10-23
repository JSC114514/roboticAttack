"""GPU migration contracts without allocating a CUDA device or Slurm job."""
import os
import subprocess

import pytest
import torch

from common import ROOT, cfg, require_gpu


@pytest.mark.parametrize("stage",["cache","train","validation","heldout"])
def test_configured_a800_passes_runtime_guard(monkeypatch,stage):
    monkeypatch.setenv("SLURM_JOB_ID","cpu_test")
    monkeypatch.setattr(torch.cuda,"is_available",lambda:True)
    monkeypatch.setattr(torch.cuda,"device_count",lambda:1)
    monkeypatch.setattr(torch.cuda,"get_device_name",lambda index:"NVIDIA A800 80GB PCIe")
    assert "A800" in require_gpu(stage)


@pytest.mark.parametrize("name",["NVIDIA GeForce RTX 4090","NVIDIA GeForce RTX 3090"])
def test_wrong_card_rejected_before_model_loading(monkeypatch,name):
    monkeypatch.setenv("SLURM_JOB_ID","cpu_test")
    monkeypatch.setattr(torch.cuda,"is_available",lambda:True)
    monkeypatch.setattr(torch.cuda,"device_count",lambda:1)
    monkeypatch.setattr(torch.cuda,"get_device_name",lambda index:name)
    with pytest.raises(RuntimeError,match="requests one GPU of type a800"):
        require_gpu("train")


def test_more_than_one_card_is_rejected(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID","cpu_test")
    monkeypatch.setattr(torch.cuda,"is_available",lambda:True)
    monkeypatch.setattr(torch.cuda,"device_count",lambda:4)
    with pytest.raises(RuntimeError,match="Exactly one allocated GPU"):
        require_gpu("cache")


def test_slurm_requests_match_stage_config():
    stages=("prepare","cache","train","validate","select","heldout","analyze")
    for index,stage in enumerate(stages,1):
        directives=(ROOT/f"slurm/{index:02d}_{stage}.sbatch").read_text()
        resource=cfg()["resources"][stage]
        assert f'--cpus-per-task={resource["cpus"]}' in directives
        assert f'--mem={resource["host_memory_gib"]}G' in directives
        if resource["gpus"]:
            assert resource["gpus"]==1 and resource["gpu_type"]=="a800"
            assert "--partition=a800" in directives and "--gres=gpu:a800:1" in directives
            assert "--qos=a800" in directives
            assert resource["partition"]==resource["qos"]=="a800"
        else:assert "--gres=" not in directives


def test_egl_model_check_uses_same_gpu_type_as_scheduler():
    environment={**os.environ,"CUDA_VISIBLE_DEVICES":"2","PYTHONDONTWRITEBYTECODE":"1"}
    command=['bash','-c','source "$1" gpu; printf "%s|%s|%s" "$OFT_STAGE_GPU" "$MUJOCO_EGL_DEVICE_ID" "$MUJOCO_GL"',
             'gpu_environment_check',str(ROOT/"slurm/environment.sh")]
    result=subprocess.run(command,env=environment,capture_output=True,text=True,check=True,timeout=30)
    assert result.stdout==cfg()["resources"]["cache"]["gpu_type"].upper()+"|2|egl"
