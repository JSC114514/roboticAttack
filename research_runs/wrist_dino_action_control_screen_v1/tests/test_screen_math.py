import torch

from screen_math import (action_metrics, clean_scale, consensus, cosine, endpoint_target,
                         group_rms, matched_random, normalized_descent_direction, primary_loss,
                         project_relative_rms_, stable_seed)


def record():
    expert = torch.ones(1, 8, 7)
    expert[..., 6] = 0
    return {"expert_actions": expert, "close_slots": torch.ones(8, dtype=torch.bool)}


def test_headwise_scale_projection_and_random_match():
    clean = torch.arange(1, 1 + 2 * 4 * 3, dtype=torch.float32).reshape(1, 2, 4, 3)
    scale = clean_scale(clean, True)
    assert scale.shape == (1, 2, 1, 1)
    z = torch.ones_like(clean)
    z[:, 1].mul_(10)
    project_relative_rms_(z, .2, True)
    assert torch.allclose(group_rms(z, True), torch.full((1, 2, 1, 1), .2))
    random = matched_random(z, 42, True)
    assert torch.allclose(group_rms(random, True), group_rms(z, True), atol=1e-6)


def test_global_scale_projection_and_deterministic_seed():
    clean = torch.tensor([[[1., 2.], [3., 4.]]])
    scale = clean_scale(clean, False)
    assert scale.numel() == 1
    z = torch.full_like(clean, 3.)
    project_relative_rms_(z, .2, False)
    assert abs(float(group_rms(z, False)) - .2) < 1e-6
    assert stable_seed(42, "x") == stable_seed(42, "x") != stable_seed(43, "x")


def test_normalized_descent_direction_has_fixed_rms_and_sign():
    gradient = torch.tensor([[[1., -2.], [3., -4.]]])
    direction = normalized_descent_direction(gradient, .2, False)
    assert abs(float(group_rms(direction, False)) - .2) < 1e-6
    assert float((direction * gradient).sum()) < 0


def test_wrong_open_and_opposite_endpoints_are_distinct():
    item = record()
    action = torch.zeros(1, 8, 7)
    open_loss = primary_loss(action, item, "wrong_open", .6)
    assert torch.isclose(open_loss, torch.tensor(.6))
    opened = action.clone(); opened[..., 6] = .7
    assert primary_loss(opened, item, "wrong_open", .6) == 0
    target = endpoint_target(item, "uada_opposite", action.device)
    assert torch.equal(target, -torch.ones_like(target))
    assert primary_loss(action, item, "uada_opposite", .6, target) > 0
    only_unmasked_axis = target.clone()
    only_unmasked_axis[..., 0] = action[..., 0]
    assert primary_loss(action, item, "uada_opposite", .6, only_unmasked_axis) == 0


def test_action_metrics_use_strict_native_threshold():
    item = record()
    clean = torch.zeros(1, 8, 7)
    action = clean.clone()
    action[..., 6] = .5
    metrics = action_metrics(action, clean, item, "wrong_open", .6)
    assert metrics["wrong_open_slot_fraction"] == 0
    action[..., 6] = .5001
    metrics = action_metrics(action, clean, item, "wrong_open", .6)
    assert metrics["all_wrong_open"] and not metrics["all_open_margin"]


def test_consensus_and_cosine():
    gradient = torch.tensor([1., -2., 3.])
    value = consensus(2 * gradient, 2 * gradient.abs(), 2 * float(gradient.norm()))
    assert abs(value["cancellation_ratio"] - 1) < 1e-6
    assert abs(value["element_sign_agreement"] - 1) < 1e-6
    assert abs(cosine(gradient, gradient) - 1) < 1e-6
    assert abs(cosine(gradient, -gradient) + 1) < 1e-6
