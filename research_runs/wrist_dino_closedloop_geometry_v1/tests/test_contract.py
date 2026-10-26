"""CPU contracts: rollback, actual-input invariance, fair selection and paired outcomes."""
from __future__ import annotations
import copy
import importlib
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import loop_core as core
from geometry import observation_with_geometry, transform
from task_progress import TaskProgress, audit_bddl, bddl_goals, score_candidates
from branch_state import BranchState


def test_identity_is_bit_exact_and_inputs_are_independent():
    image = np.random.default_rng(4).integers(0, 256, (224, 224, 3), dtype=np.uint8)
    obs = {"full_image": image.copy(), "wrist_image": image.copy(), "state": np.arange(8.)}
    out = observation_with_geometry(obs, core.cfg()["candidates"][0])
    assert np.array_equal(out["wrist_image"], image)
    out["state"][:] = 999; out["full_image"][:] = 0
    assert np.array_equal(obs["state"], np.arange(8.)) and np.array_equal(obs["full_image"], image)


@pytest.mark.parametrize("candidate", core.cfg()["candidates"])
def test_warps_preserve_uniform_color_without_black_border(candidate):
    image = np.full((224, 224, 3), [17, 83, 201], dtype=np.uint8)
    out = transform(image, candidate)
    assert out.dtype == np.uint8 and np.array_equal(image, out)


def test_translation_moves_content_in_declared_direction():
    image = np.zeros((224, 224, 3), dtype=np.uint8); image[100:110, 100:110] = 255
    candidate = {"scale": 1., "dx": 4, "dy": -4}
    out = transform(image, candidate)
    assert np.array_equal(out[96:106, 104:114], np.full((10, 10, 3), 255, dtype=np.uint8))


def test_geometry_budget_rejects_unapproved_transform():
    with pytest.raises(ValueError, match="budget"):
        transform(np.zeros((224, 224, 3), dtype=np.uint8), {"scale": .5, "dx": 0, "dy": 0})


def test_selector_uses_task_consequence_instead_of_largest_action_error():
    rows = [{"success": True, "goal_fraction_mean": 1., "dense_progress_mean": .9, "uada_mse": .1},
            {"success": False, "goal_fraction_mean": 0., "dense_progress_mean": .2, "uada_mse": .8},
            {"success": False, "goal_fraction_mean": 0., "dense_progress_mean": .6, "uada_mse": .3}]
    assert score_candidates(rows, "uada_geometry") == 0
    assert score_candidates(rows, "loop_geometry") == 1


def test_identity_remains_available_and_ties_are_deterministic():
    rows = [dict(success=False, goal_fraction_mean=0., dense_progress_mean=.2, uada_mse=.5)] * 7
    assert score_candidates(rows, "loop_geometry") == score_candidates(rows, "uada_geometry") == 0


def test_actual_forty_bddl_tasks_supported_without_gpu_import():
    report = audit_bddl(Path('/home/caijiasheng25/Desktop/LIBERO/libero/libero/bddl_files'))
    assert report["bddl_files"] == 40 and report["predicate_counts"]["open"] > 0


def test_bddl_parser_handles_comments_and_capitalized_and():
    goals = bddl_goals('(define (:goal (And (Open drawer) (In bowl region)))) ; comment')
    assert goals == [["Open", "drawer"], ["In", "bowl", "region"]]


class ToyEnv:
    def __init__(self):
        self.env = self
        self.x = np.array([0., 0., 0.])
        data = NS(time=0., qacc_warmstart=np.ones(3), qfrc_applied=np.zeros(3),
                  xfrc_applied=np.zeros((1, 6)), ctrl=np.zeros(2), act=np.zeros(0),
                  mocap_pos=np.zeros((0, 3)), mocap_quat=np.zeros((0, 4)), userdata=np.zeros(0),
                  site_xpos=np.array([[0., 0., 0.]]), qpos=np.array([-.15]))
        model = NS(cam_pos=np.zeros((2, 3)), cam_quat=np.zeros((2, 4)), site_rgba=np.ones((2, 4)),
                   geom_rgba=np.ones((2, 4)), light_active=np.ones(1), get_joint_qpos_addr=lambda _j: 0)
        self.sim = NS(data=data, model=model, forward=lambda: None)
        self.sim.get_state = lambda: NS(flatten=lambda: self.x.copy())
        self.controller = NS(goal_pos=np.zeros(3), goal_ori=np.eye(3), new_update=False, ee_pos=np.zeros(3),
                             joint_pos=np.zeros(7), J_full=np.zeros((6, 7)), mass_matrix=np.eye(7))
        self.robots = [NS(controller=self.controller, gripper=NS(current_action=np.zeros(1)), eef_site_id=0)]
        self.timestep, self.cur_time, self.done = 0, 0., False
        self.np_random = np.random.RandomState(42)
        self._observables, self._obs_cache = {}, {"custom": np.array([5.])}
        self.raw = {"agentview_image": np.zeros((2, 2, 3), dtype=np.uint8),
                    "robot0_eye_in_hand_image": np.ones((2, 2, 3), dtype=np.uint8)}
        self.objects_dict = {"drawer": NS(object_properties={"articulation": {"default_open_ranges": [-.16, -.14]},
                                                            "vis_site_names": {"lamp": ("lamp", False)}})}
        self.fixtures_dict = {}
        self.object_sites_dict = {"region": NS(joints=["drawer_joint"])}
        self.object_states_dict = {"region": NS(parent_name="drawer", get_geom_state=lambda: {"pos": np.array([.1, 0., 0.])})}
        self.parsed_problem = {"goal_state": [["open", "region"]]}

    def get_object(self, name):
        return self.objects_dict[name]

    def get_sim_state(self):
        return self.x.copy()

    def set_state(self, values):
        self.x[:] = values

    def _get_observations(self):
        return copy.deepcopy(self.raw)

    def _eval_predicate(self, _goal):
        return -.16 <= self.sim.data.qpos[0] <= -.14


def test_drawer_site_uses_its_joint_and_parents_negative_open_range():
    env = ToyEnv()
    progress = TaskProgress(env).measure()
    assert progress["goal_fraction"] == 1.
    assert progress["details"][0]["joint_target_range"] == [-.16, -.14]
    assert progress["details"][0]["joint_progress"] == 1.


def test_missing_joint_semantics_aborts_instead_of_silent_zero():
    env = ToyEnv(); env.objects_dict["drawer"].object_properties["articulation"] = {}
    with pytest.raises(ValueError, match="declared joint"):
        TaskProgress(env).measure()


def test_snapshot_restores_physics_controller_gripper_solver_visuals_and_rng():
    env = ToyEnv()
    random.seed(11); np.random.seed(13); torch.manual_seed(17)
    snapshot = BranchState(env, env.raw)
    expected = (random.random(), np.random.rand(), torch.rand(1).item())
    env.x[:] = 9; env.controller.goal_pos[:] = 8; env.controller.ee_pos[:] = 4
    env.robots[0].gripper.current_action[:] = 1
    env.sim.data.qacc_warmstart[:] = 23; env.sim.model.site_rgba[:] = 0
    env.objects_dict["drawer"].object_properties["vis_site_names"]["lamp"] = ("lamp", True)
    env._obs_cache["custom"][:] = 12; env.timestep = 100
    raw, error = snapshot.restore(env)
    assert error == 0 and env.timestep == 0 and np.all(env.controller.goal_pos == 0)
    assert np.all(env.controller.ee_pos == 0) and np.all(env.robots[0].gripper.current_action == 0)
    assert np.all(env.sim.data.qacc_warmstart == 1) and np.all(env.sim.model.site_rgba == 1)
    assert env.objects_dict["drawer"].object_properties["vis_site_names"]["lamp"][1] is False
    assert env._obs_cache["custom"][0] == 5
    assert (random.random(), np.random.rand(), torch.rand(1).item()) == expected
    raw["agentview_image"][:] = 8
    assert np.all(snapshot.raw["agentview_image"] == 0)


def fake_window(env, raw, predictor, progress, runtime, candidate, remaining, **kwargs):
    index = [x["name"] for x in core.cfg()["candidates"]].index(candidate["name"])
    predictor.calls += 2
    env.x += index + 1
    env.robots[0].gripper.current_action[:] = index
    env.sim.data.qacc_warmstart[:] = index
    # Action objective favors the last candidate; true task progress favors index 1.
    value = .5 - index * .2
    dense = abs(index - 1) / 7
    actions = [np.full((8, 7), value)] * 2
    detail = {"goal_fraction": 0., "dense_progress": dense}
    return raw, {"actions": actions, "normalized": actions, "progress": [detail] * 2,
                 "success": False, "steps": 16, "first_dino": torch.ones((1, 256, 1024)) * (index + 1),
                 "first_layers": {}, "pixel": {}, "action_parity_max": 0., "end_state": env.x.copy(),
                 "end_image_hash": "same", "goal_fraction_mean": 0., "dense_progress_mean": dense,
                 "end_progress": detail}


def test_planner_discards_all_hypothetical_changes_then_commits_selected_branch(monkeypatch):
    import closedloop
    env = ToyEnv(); predictor = NS(calls=0)
    monkeypatch.setattr(closedloop, "window", fake_window)
    _, committed, receipt, rows = closedloop.plan(env, env.raw, predictor, TaskProgress(env), {}, 16, "loop_geometry", 0)
    assert receipt["selected_index"] == 1 and np.array_equal(env.x, np.full(3, 2.))
    assert predictor.calls == 16 and len(rows) == 7
    assert receipt["commit_state_max_error"] == 0 and sum(r["selected"] for r in rows) == 1
    assert np.all(env.robots[0].gripper.current_action == 1)


def test_failed_hypothetical_branch_cannot_pollute_real_episode(monkeypatch):
    import closedloop
    env = ToyEnv(); predictor = NS(calls=0)
    def fail(*args, **kwargs):
        fake_window(*args, **kwargs)
        raise RuntimeError("hypothetical failure")
    monkeypatch.setattr(closedloop, "window", fail)
    with pytest.raises(RuntimeError, match="hypothetical"):
        closedloop.plan(env, env.raw, predictor, TaskProgress(env), {}, 16, "loop_geometry", 0)
    assert np.all(env.x == 0) and np.all(env.robots[0].gripper.current_action == 0)


def test_nondeterministic_selected_branch_aborts_and_rolls_back(monkeypatch):
    import closedloop
    env = ToyEnv(); predictor = NS(calls=0)
    def unstable(*args, **kwargs):
        raw, result = fake_window(*args, **kwargs)
        if predictor.calls > 14:
            env.x += .1; result["end_state"] = env.x.copy()
        return raw, result
    monkeypatch.setattr(closedloop, "window", unstable)
    with pytest.raises(RuntimeError, match="did not replay"):
        closedloop.plan(env, env.raw, predictor, TaskProgress(env), {}, 16, "loop_geometry", 0)
    assert np.all(env.x == 0)


def test_casr_uses_clean_success_and_paired_task_weights():
    from summarize import metrics
    records = []
    for suite in core.SUITES:
        for task in (0, 1, 2):
            for condition in core.CONDITIONS:
                success = 0 if task == 2 or (task == 0 and condition != "clean") or (task == 1 and condition == "loop_geometry") else 1
                records.append(dict(suite=suite, task_id=task, state_index=40, condition=condition,
                                    success=success, failure=1 - success, policy_calls=2, elapsed_s=1.))
    report, paired, _ = metrics(pd.DataFrame(records))
    assert set(report.clean_success_states) == {2}
    assert report.loc[report.condition.eq("loop_geometry"), "task_equal_casr"].eq(1).all()
    assert report.loc[report.condition.eq("uada_geometry"), "task_equal_casr"].eq(.5).all()
    assert set(paired.new_failures) == {1} and set(paired.lost_failures) == {0}


def test_active_submission_and_dependency_guards():
    from submit_run import ensure_no_active, job_command
    with pytest.raises(RuntimeError, match="active jobs"):
        ensure_no_active("prefix_eval|1234|RUNNING", "prefix_")
    ensure_no_active("other_eval|1234|RUNNING", "prefix_")
    command = job_command("eval", "02_eval.sbatch", ["1234"], "prefix_")
    assert "--dependency=afterok:1234" in command and "--kill-on-invalid-dep=yes" in command


def test_suite_without_clean_success_reports_missing_casr_not_false_failure_rate():
    from summarize import metrics
    rows = [dict(suite=suite, task_id=0, state_index=40, condition=condition, success=0, failure=1,
                 policy_calls=1, elapsed_s=1.) for suite in core.SUITES for condition in core.CONDITIONS]
    report, paired, _ = metrics(pd.DataFrame(rows))
    assert report.task_equal_casr.isna().all() and report.clean_success_states.eq(0).all()
    assert paired.casr_difference.isna().all()


def test_write_boundary_and_completed_artifact_tamper_are_enforced(monkeypatch, tmp_path):
    monkeypatch.setattr(core, "ROOT", tmp_path)
    monkeypatch.setattr(core, "signature", lambda: "test-source-signature")
    with pytest.raises(RuntimeError, match="outside"):
        core.write_json({}, tmp_path.parent / "forbidden.json")
    artifact = tmp_path / "data.json"; artifact.write_text("original")
    core.finish(tmp_path / "STATUS.json", [artifact])
    assert core.stage_done(tmp_path / "STATUS.json")
    artifact.write_text("modified")
    with pytest.raises(RuntimeError, match="artifact changed"):
        core.stage_done(tmp_path / "STATUS.json")


class FakeHead(torch.nn.Module):
    def predict_action(self, value):
        return value


def test_native_observers_preserve_policy_result_and_remove_hooks():
    from native_predictor import NativePredictor
    head = FakeHead()
    dino = torch.nn.Identity(); projector = torch.nn.Identity()
    layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(3)])
    vla = NS(vision_backbone=NS(featurizer=dino), projector=projector,
             language_model=NS(model=NS(layers=layers)), _unnormalize_actions=lambda a, k: a * 2)
    def predict(obs, instruction):
        obs["state"][:] = 999
        z = torch.ones((1, 256, 1024))
        dino(z); dino(z * 2); projector(z)
        for layer in layers:
            layer(z)
        return head.predict_action(torch.arange(56.).reshape(8, 7)).numpy() * 2
    wrapper = NS(vla=vla, action_head=head, gen_cfg=NS(unnorm_key="test"), predict_action=predict)
    obs = {"state": np.zeros(8)}
    expected = predict(copy.deepcopy(obs), "instruction")
    with NativePredictor(wrapper, "instruction") as predictor:
        result = predictor.predict(obs, observe=True, layers=True)
        assert np.array_equal(result["actions"], expected) and torch.all(result["dino"] == 2)
        assert np.all(obs["state"] == 0) and result["parity_error"] == 0
    assert not dino._forward_hooks and "predict_action" not in head.__dict__
