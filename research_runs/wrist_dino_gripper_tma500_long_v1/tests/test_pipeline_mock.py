from types import SimpleNamespace
import re

import pytest

import pipeline
import gripper_common


def test_existing_gpu_jobs_are_waited_not_cancelled():
    assert pipeline.parse_queue("123_[0-2]|other_training|gpu:4090:1|None\n124|oft_old_stats|N/A|Dependency") == ["123","124"]


@pytest.mark.parametrize("text",["123|oft_grip_train|gpu:4090:1|None",
                                  "123|oft_old|gpu:3090:1|DependencyNeverSatisfied","bad response"])
def test_invalid_or_duplicate_queue_is_rejected(text):
    with pytest.raises(RuntimeError):
        pipeline.parse_queue(text)


def test_submission_chain_all_mocked_and_phase_serialized(tmp_path,monkeypatch):
    commands = []
    monkeypatch.setattr(pipeline,"ROOT",tmp_path)
    monkeypatch.setattr(pipeline,"preflight",lambda: {})
    def fake_run(command,**kwargs):
        commands.append(command)
        if command[0]=="squeue":
            return SimpleNamespace(returncode=0,stdout="77|other_gpu|gpu:3090:1|None",stderr="")
        assert command[0]=="sbatch"
        return SimpleNamespace(returncode=0,stdout=str(100+len(commands)),stderr="")
    monkeypatch.setattr(pipeline.subprocess,"run",fake_run)
    pipeline.submit()
    jobs = commands[1:]
    assert len(jobs)==7 and "--dependency=afterany:77" in jobs[0]
    assert all(any(arg.startswith("--dependency=afterok:") for arg in command) for command in jobs[1:])
    assert "--array=0,1,2,3,4,5,6,7,8,9%3" in jobs[1]
    assert "--array=0,1,2%3" in jobs[3]
    assert "--array=0,1,2,3,4,5,6,7,8,9%5" in jobs[5]
    assert jobs[1][-1].endswith("148_labels_a800.sbatch")
    assert jobs[3][-1].endswith("150_train_a800.sbatch")
    assert jobs[5][-1].endswith("152_eval_4090.sbatch")
    for index in (1,3):
        assert "--partition=a800" in jobs[index] and "--gres=gpu:a800:1" in jobs[index]
        assert "--qos=a800" in jobs[index]
    assert "--partition=rtx" in jobs[5] and "--gres=gpu:4090:1" in jobs[5]
    assert all("--nodes=1" in jobs[index] and "--ntasks=1" in jobs[index] for index in (1,3,5))
    receipt = gripper_common.load_json(tmp_path/"reports/SUBMISSION_RECEIPT.json")
    assert receipt["gpu_limits_by_serial_phase"] == {"labels_a800":3,"training_a800":3,"evaluation_4090":5}
    assert not any(command[0]=="scancel" for command in commands)


@pytest.mark.parametrize("stage,script",[
    ("labels","148_labels_a800.sbatch"),
    ("training","150_train_a800.sbatch"),
    ("evaluation","152_eval_4090.sbatch"),
])
def test_resource_config_and_slurm_directives_agree(stage,script):
    spec = gripper_common.config()["resources"][stage]
    content = (gripper_common.ROOT/"slurm"/script).read_text()
    directives = dict(re.findall(r"^#SBATCH --([\w-]+)=(.+)$",content,re.MULTILINE))
    assert directives["partition"] == spec["partition"]
    assert directives["gres"] == f"gpu:{spec['gpu_type']}:1"
    assert int(directives["array"].rsplit("%",1)[1]) == spec["max_concurrent"]
    assert directives.get("qos") == spec["qos"]
    assert directives["nodes"] == directives["ntasks"] == "1"


@pytest.mark.parametrize("stage,device,accepted",[
    ("labels","NVIDIA A800 80GB PCIe",True),
    ("labels","NVIDIA GeForce RTX 4090",False),
    ("training","NVIDIA A800 80GB PCIe",True),
    ("training","NVIDIA GeForce RTX 4090",False),
    ("evaluation","NVIDIA GeForce RTX 4090",True),
    ("evaluation","NVIDIA GeForce RTX 3090",False),
])
def test_runtime_device_guard_matches_approved_stage(stage,device,accepted,monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda,"is_available",lambda:True)
    monkeypatch.setattr(torch.cuda,"device_count",lambda:1)
    monkeypatch.setattr(torch.cuda,"get_device_name",lambda index:device)
    if accepted:
        assert gripper_common.require_stage_gpu(stage)["gpu"] == device
    else:
        with pytest.raises(RuntimeError,match="This stage requires"):
            gripper_common.require_stage_gpu(stage)
