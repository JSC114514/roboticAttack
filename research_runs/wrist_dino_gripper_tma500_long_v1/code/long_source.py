"""Long-only source contracts; no rendering, model loading or Slurm on import."""
from __future__ import annotations

import inspect
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from gripper_common import require, sha256_file


def decision_indices(count, selected=6):
    require(count >= selected >= 2, "Successful source episode has too few distinct decisions")
    indices = np.rint(np.linspace(0, count-1, selected)).astype(int).tolist()
    require(len(set(indices)) == selected, "Duplicate source decisions")
    return indices


def phase_bucket(index, decisions):
    return min(2, int(3*index/max(1, decisions-1)))


def records_digest(records):
    h = hashlib.sha256()
    for record in records:
        scalar = {k:v for k,v in record.items() if k not in {"observation","sim_state","clean_action","native_action"}}
        h.update(json.dumps(scalar,sort_keys=True,separators=(",",":")).encode())
        fields = {**record["observation"], **{k:record[k] for k in ("sim_state","clean_action","native_action")}}
        for name in sorted(fields):
            value = fields[name]
            array = value.detach().cpu().numpy() if isinstance(value,torch.Tensor) else np.asarray(value)
            h.update(f"{name}:{array.dtype}:{array.shape}".encode())
            h.update(np.ascontiguousarray(array).tobytes())
    return h.hexdigest()


def environment_inventory(cfg):
    """CPU read-only; the existing LIBERO config must exist before package import."""
    from model_runtime_copy import add_paths
    source = cfg["training_source"]
    config_file = Path(source["libero_config_file"])
    require(config_file.is_file(), "Existing LIBERO config missing; no auto-create or interactive fallback")
    add_paths({"paths":cfg["evaluation"]["paths"]})
    import libero.libero as libero_package
    require(Path(libero_package.config_file).resolve() == config_file.resolve(), "Unprepared LIBERO config selected")
    from libero.libero import benchmark, get_libero_path
    require(Path(inspect.getfile(benchmark)).resolve().is_relative_to(
            Path(cfg["evaluation"]["paths"]["libero_root"]).resolve()), "Wrong LIBERO package loaded")
    require(Path(get_libero_path("bddl_files")).joinpath(source["suite"]).resolve()
            == Path(source["bddl_directory"]).resolve(), "Long BDDL path differs from prepared source")
    require(Path(get_libero_path("init_states")).joinpath(source["suite"]).resolve()
            == Path(source["init_states_directory"]).resolve(), "Long initial-state path differs from prepared source")
    bench = benchmark.get_benchmark_dict()[source["suite"]]()
    require(bench.n_tasks == source["task_count"] == 10, "Wrong Long benchmark")
    paths = {config_file, Path(inspect.getfile(benchmark))}
    paths.update(Path(inspect.getfile(benchmark)).parent.glob("*.py"))
    paths.update(Path(inspect.getfile(benchmark)).parent.parent.joinpath("envs").glob("*.py"))
    tasks = []
    for task_id in range(source["task_count"]):
        task = bench.get_task(task_id)
        bddl = Path(source["bddl_directory"])/task.bddl_file
        states = Path(source["init_states_directory"])/task.init_states_file
        require(bddl.is_file() and states.is_file(), f"Long task assets missing: {task_id}")
        values = torch.load(states, map_location="cpu", weights_only=False)
        require(len(values) > max([*source["state_indices"], *cfg["evaluation"]["state_indices"]]),
                "Long init states do not contain the frozen source/evaluation grids")
        paths.update((bddl,states))
        tasks.append({"task_id":task_id,"task_name":task.language,"bddl":str(bddl.resolve()),
                      "init_states":str(states.resolve()),"initial_state_count":len(values)})
    return {"files":{str(p.resolve()):sha256_file(p) for p in sorted(paths)},"tasks":tasks}


def validate_records(records, cfg):
    source = cfg["training_source"]
    per_task = source["successful_trajectories_per_task"]*source["records_per_trajectory"]
    require(len(records) == source["task_count"]*per_task == 300, "Long source record count changed")
    require([r["bank_index"] for r in records] == list(range(300)), "Long bank indices missing/duplicated")
    keys = [(r["task_id"],r["trajectory_rank"],r["decision_index"]) for r in records]
    require(len(set(keys)) == 300, "Long source decisions duplicated")
    require(set(r["task_id"] for r in records) == set(range(10)), "Long source tasks incomplete")
    for record in records:
        require(record["source_suite"] == source["suite"] == "libero_10" and record["episode_success"] is True,
                "Non-Long or unsuccessful data in the source bank")
        require(record["state_index"] in source["state_indices"]
                and record["state_index"] not in cfg["evaluation"]["state_indices"], "Source/evaluation state leakage")
        require(record["rollout_seed"] == source["seed"]+record["task_id"]*100+record["state_index"],
                "Source seed changed")
        for field in ("full_image","wrist_image"):
            image = np.asarray(record["observation"][field])
            require(image.shape == (224,224,3) and image.dtype == np.uint8, "Long source RGB shape/dtype")
        state = np.asarray(record["observation"]["state"])
        require(state.shape == (8,) and np.isfinite(state).all(), "Long source proprio invalid")
        sim_state = np.asarray(record["sim_state"])
        require(sim_state.ndim == 1 and sim_state.size > 0 and np.isfinite(sim_state).all(), "Missing source simulator state")
        for key, shape in (("clean_action",(1,8,7)),("native_action",(8,7))):
            value = torch.as_tensor(record[key])
            require(tuple(value.shape) == shape and torch.isfinite(value).all(), "Long source action invalid")
        require(record["phase_bucket"] == phase_bucket(record["decision_index"],record["total_decisions"]),
                "Long source phase changed")
    for task_id in range(10):
        states = []
        for rank in range(source["successful_trajectories_per_task"]):
            group = [r for r in records if r["task_id"] == task_id and r["trajectory_rank"] == rank]
            require(len(group) == source["records_per_trajectory"] and len({r["state_index"] for r in group}) == 1
                    and len({r["total_decisions"] for r in group}) == 1, "Long trajectory grouping changed")
            require(sorted(r["decision_index"] for r in group) == decision_indices(
                    group[0]["total_decisions"],source["records_per_trajectory"]), "Source decisions are not full-episode uniform")
            states.append(group[0]["state_index"])
        require(len(set(states)) == source["successful_trajectories_per_task"], "Repeated initial state counted as a new source trajectory")
    return records
