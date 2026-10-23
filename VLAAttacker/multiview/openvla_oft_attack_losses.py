"""
OpenVLA-OFT 对抗补丁训练的攻击损失函数模块。

背景
----
VLA 模型将机械臂动作（7 自由度：Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper）
离散化为 256 个 bin，映射到 tokenizer 词汇表的最后 256 个 token。
因此"攻击补丁"的优化目标不再是简单的 -outputs.loss，而是精确操控
这些动作 token 的预测分布，使其偏离专家动作。

本模块实现五类攻击损失：
1. oft_clean_reference_hybrid_attack : clean action 参考 + 动作发散 + 视觉特征发散
2. oft_action_mse_far_boundary : OpenVLA-OFT 连续 action head 的远边界 MSE 损失
3. uada_far_boundary   : 把选中 DoF 的 soft token 预测值推向专家动作的对面边界
4. clean_adv_kl        : 最大化攻击前后动作 logits 分布的 KL 散度
5. model_ce_max        : 兜底方案，最大化 CE loss（等同于原来的 -outputs.loss）

核心计算流程（以 uada_far_boundary 为例）
------------------------------------------
1. _aligned_action_logits_and_labels()
   从 logits 中切出动作 token 部分（最后 256 维），并按 HuggingFace 的
   label shift 规则对齐（logits[:, -seq-1:-1] 对应 labels[:, 1:]）。
   再根据 maskidx 筛选出指定 DoF 的 (logits, labels) 对。

2. uada_far_boundary_loss()
   对选中的动作 token 做 softmax → 用 [1/256, 2/256, ..., 1.0] 加权求和
   得到 soft_value ∈ [0, 1]。
   目标 target：如果专家 token 在上半区（大于中点），推到 1/256（下边界）；
   如果在下半区，推到 1.0（上边界）。用 MSE 优化。

3. 反向传播时，patch.grad 反映出"如何修改像素能让动作 token 分布朝目标移动"。
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def parse_maskidx(
    value: Any,
    default: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6),
) -> list[int]:
    """从配置字符串（如 "0,1,2"）解析需要攻击的自由度索引。

    DoF 索引含义（7 自由度机械臂）：
      0: Δx    1: Δy    2: Δz
      3: Δroll 4: Δpitch 5: Δyaw
      6: gripper
    """
    if value is None or str(value).strip() == "":
        return list(default)
    if isinstance(value, str):
        out: list[int] = []
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue
            out.append(int(item))
        return out or list(default)
    return [int(x) for x in value]


def _config_bool(config: dict[str, str], key: str, default: bool = False) -> bool:
    value = config.get(key)
    if value is None or str(value).strip() == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _config_float(config: dict[str, str], key: str, default: float) -> float:
    value = config.get(key)
    if value is None or str(value).strip() == "":
        return default
    return float(str(value).strip())


def _action_token_begin_idx(logits: torch.Tensor, n_bins: int = 256) -> int:
    """OpenVLA 的动作 token 占据词汇表最后 n_bins 个位置。

    token ID 范围: [vocab_size - 257, vocab_size - 1]
    其中 vocab_size - 257 是特殊分隔 token，vocab_size - 256 到 vocab_size - 1
    对应 256 个离散 bin。返回值是分隔 token 的索引。
    """
    return int(logits.shape[-1] - (n_bins + 1))


# ---------------------------------------------------------------------------
# 核心：动作 token logits 与 labels 的对齐和筛选
# ---------------------------------------------------------------------------

def _aligned_action_logits_and_labels(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    maskidx: list[int],
    action_dim: int = 7,
    n_bins: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """从原始 outputs.logits 中提取选中 DoF 的动作 token logits 和 labels。

    HuggingFace 因果 LM 的 loss 计算规则：
      - logits[:, i] 预测 labels[:, i+1] （shift 一位）
      - 加上 OpenVLA 在 prompt 和 labels 之间插入了视觉 token

    因此这里取 logits[:, -(label_len+1):-1, action_start:action_end]
    对应 labels[:, 1:] 中属于动作 token 的位置。

    返回:
      (selected_logits, selected_labels, selected_mask, begin_idx)
      - selected_logits: [num_selected_tokens, 256]  选中 DoF 的 logits
      - selected_labels: [num_selected_tokens]        对应的 GT token IDs
      - selected_mask:   [batch, seq_len]             布尔掩码（调试用）
      - begin_idx:       int                          动作起始 token 索引
    """
    if labels is None:
        raise ValueError("UADA/UPA action-token losses require labels in the batch.")

    # 动作 token 在词汇表中的位置：[begin_idx+1, begin_idx+256]
    begin_idx = _action_token_begin_idx(logits, n_bins=n_bins)
    action_start = begin_idx + 1
    action_end = action_start + n_bins
    if action_end > logits.shape[-1]:
        raise ValueError(
            f"Action token slice [{action_start}:{action_end}] exceeds logits vocab size={logits.shape[-1]}"
        )

    # 对齐：logits 向前 shift 一位对应 labels
    # logits[:, -(label_len+1):-1, :] 预测 labels[:, 1:]
    target_labels = labels[:, 1:].to(logits.device)
    aligned_logits = logits[:, -(target_labels.shape[-1] + 1) : -1, action_start:action_end]

    # 找出所有 GT 标签中的动作 token 位置（token ID > begin_idx）
    action_mask = target_labels > begin_idx

    # 按 maskidx 筛选指定的 DoF
    # 动作 token 在序列中按 (batch_sample, DoF_0..DoF_6) 交替排列
    selected_mask = torch.zeros_like(action_mask, dtype=torch.bool)
    wanted = set(int(i) for i in maskidx)
    for batch_idx in range(target_labels.shape[0]):
        positions = torch.nonzero(action_mask[batch_idx], as_tuple=False).flatten()
        if positions.numel() == 0:
            continue
        # 每 7 个动作 token 对应一个 7-DoF 动作，取模得到 DoF 索引
        local_ids = torch.arange(positions.numel(), device=positions.device) % int(action_dim)
        keep = torch.tensor(
            [int(x.item()) in wanted for x in local_ids],
            device=positions.device,
            dtype=torch.bool,
        )
        selected_mask[batch_idx, positions[keep]] = True

    if not bool(selected_mask.any()):
        raise ValueError(
            f"No selected action tokens found. maskidx={maskidx}, labels shape={tuple(labels.shape)}, "
            f"begin_idx={begin_idx}"
        )

    return aligned_logits[selected_mask], target_labels[selected_mask], selected_mask, begin_idx


# ---------------------------------------------------------------------------
# 攻击损失 1: UADA 远边界损失
# ---------------------------------------------------------------------------

def uada_far_boundary_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    maskidx: list[int],
    action_dim: int = 7,
    n_bins: int = 256,
    scale: float = 5.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """UADA 风格的无目标动作偏差损失。

    思想（与 VLAAttacker/white_patch/UADA.py 一致）：
    - 用 softmax 得到动作 token 的概率分布
    - 用加权平均 [1/n_bins, 2/n_bins, ..., 1.0] 得到 soft_value（连续化的预测 bin 位置）
    - 目标 target：如果 GT token 在上半区 → 推到 1/n_bins（下边界）
                   如果 GT token 在下半区 → 推到 1.0（上边界）
    - 最小化 MSE(soft_value, target) → 最大化动作偏差

    scale 参数控制损失量级（与原始实现中的 MSE_weights=5 对应）。

    返回:
      (loss, info_dict)
      loss: 标量，用于 backward()
      info_dict: 包含 loss_uada、selected_action_tokens、token_distance_mean
    """
    # Step 1: 提取选中 DoF 的对齐 logits 和 labels
    action_logits, action_labels, _selected_mask, begin_idx = _aligned_action_logits_and_labels(
        logits, labels, maskidx=maskidx, action_dim=action_dim, n_bins=n_bins
    )

    # Step 2: softmax → 加权平均得到 soft_value ∈ [1/n_bins, 1.0]
    # soft_value 越高表示预测倾向于动作值大的 bin
    prob = F.softmax(action_logits.float(), dim=-1)              # [N, 256]
    weights = torch.arange(1, n_bins + 1, device=logits.device, dtype=torch.float32) / float(n_bins)
    soft_value = (prob * weights).sum(dim=-1)                     # [N]

    # Step 3: 决定目标边界
    # midpoint = begin_idx + 128 对应动作值 0（bin 中心）
    midpoint = begin_idx + (n_bins // 2)
    # GT token > midpoint → 专家动作偏向正半轴 → 推向 1/n_bins（负半轴边界）
    # GT token ≤ midpoint → 专家动作偏向负半轴 → 推向 1.0（正半轴边界）
    target = torch.where(
        action_labels > midpoint,
        torch.full_like(soft_value, 1.0 / float(n_bins)),
        torch.ones_like(soft_value),
    )

    # Step 4: MSE loss
    loss = F.mse_loss(scale * soft_value.contiguous(), scale * target.contiguous())

    # 监控指标（no_grad）
    with torch.no_grad():
        pred_token = action_logits.argmax(dim=-1) + begin_idx + 1
        token_distance = (pred_token.float() - action_labels.float()).abs().mean()

    return loss, {
        "loss_uada": float(loss.detach().item()),
        "selected_action_tokens": float(action_labels.numel()),
        "token_distance_mean": float(token_distance.detach().item()),
    }


# ---------------------------------------------------------------------------
# 攻击损失 2: 干净-攻击 KL 散度损失
# ---------------------------------------------------------------------------

def logits_kl_discrepancy_loss(
    adv_logits: torch.Tensor,
    clean_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    maskidx: list[int],
    action_dim: int = 7,
    n_bins: int = 256,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """干净 vs 攻击动作 token 分布的 KL 散度最大化。

    思想：
    - 额外做一次干净前向传播（无 patch）获得 clean_logits
    - 计算: KL( P_adv || P_clean ) = Σ P_adv * log(P_adv / P_clean)
    - 攻击目标: 最小化 -KL，即最大化 P_adv 与 P_clean 的分布差异

    这避免了 uada_far_boundary 需要手动定义"远边界"的问题，
    直接以 clean 预测为参考，将攻击后的分布推远。

    代价：每步多一次前向传播（约 2× 计算量）。

    返回:
      (loss, info_dict)
      loss: -KL（负 KL 散度，最小化即最大化差异）
      info_dict: 包含 loss_clean_adv_kl 和 kl 原始值
    """
    # Step 1: 从攻击 logits 中提取选中 DoF 的对齐 logits，同时记录 selected_mask
    adv_sel, _labels_sel, selected_mask, begin_idx = _aligned_action_logits_and_labels(
        adv_logits, labels, maskidx=maskidx, action_dim=action_dim, n_bins=n_bins
    )

    # Step 2: 从干净 logits 中提取相同位置的 logits
    action_start = begin_idx + 1
    action_end = action_start + n_bins
    target_labels = labels[:, 1:].to(adv_logits.device)
    clean_aligned = clean_logits[:, -(target_labels.shape[-1] + 1) : -1, action_start:action_end]
    clean_sel = clean_aligned[selected_mask]   # 与 adv_sel 一一对应

    # Step 3: softmax（clean 侧 detach，不向干净 logits 回传梯度）
    clean_prob = F.softmax(clean_sel.detach().float() / temperature, dim=-1)
    adv_log_prob = F.log_softmax(adv_sel.float() / temperature, dim=-1)

    # Step 4: KL(P_adv || P_clean)，对 batch 内所有选中 token 取均值
    kl = F.kl_div(adv_log_prob, clean_prob, reduction="batchmean")

    # 攻击目标是最大化 KL，等价于最小化 -KL
    loss = -kl
    return loss, {
        "loss_clean_adv_kl": float(loss.detach().item()),
        "kl": float(kl.detach().item()),
    }


# ---------------------------------------------------------------------------
# 攻击损失 3: OpenVLA-OFT 连续 action head 远边界损失
# ---------------------------------------------------------------------------

def _select_action_dims(maskidx: list[int], action_dim: int) -> list[int]:
    wanted = [int(idx) for idx in maskidx if 0 <= int(idx) < action_dim]
    if not wanted:
        raise ValueError(f"maskidx selects no valid action dims for action_dim={action_dim}: {maskidx}")
    return wanted


def _extract_oft_continuous_actions(
    *,
    outputs: Any,
    labels: torch.Tensor,
    actions: torch.Tensor | None,
    action_head: Any,
    num_patches: int | None,
    num_visual_tokens: int | None = None,
    num_images: int | None = None,
    loss_name: str = "oft_action_head",
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Extract differentiable OpenVLA-OFT continuous actions from action-token hidden states."""
    if action_head is None:
        raise RuntimeError(f"{loss_name} requires action_head, got None.")
    if actions is None:
        raise RuntimeError(f"{loss_name} requires batch['actions'], got None.")
    if num_patches is None:
        raise RuntimeError(f"{loss_name} requires num_patches.")
    hidden_states = getattr(outputs, "hidden_states", None)
    if not hidden_states:
        raise RuntimeError(
            f"{loss_name} requires outputs.hidden_states; call VLA with output_hidden_states=True."
        )

    try:
        from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
    except Exception as exc:
        raise RuntimeError("Cannot import OpenVLA-OFT action mask helpers.") from exc

    device = actions.device
    labels = labels.to(device)
    ground_truth_actions = actions.to(device=device)
    if ground_truth_actions.ndim == 2:
        ground_truth_actions = ground_truth_actions.unsqueeze(1)
    elif ground_truth_actions.ndim > 3:
        ground_truth_actions = ground_truth_actions.reshape(
            ground_truth_actions.shape[0], -1, ground_truth_actions.shape[-1]
        )
    batch_size, chunk_len, action_dim = ground_truth_actions.shape

    token_ids = labels[:, 1:]
    current_action_mask = get_current_action_mask(token_ids)
    next_actions_mask = get_next_actions_mask(token_ids)
    action_mask = current_action_mask | next_actions_mask

    last_hidden_states = hidden_states[-1]
    # labels[:, 1:] 对应的预测位置是 fused hidden states 的 [num_patches:-1]。
    # 这里的 num_patches 必须包含 vision tokens 以及 proprio/diffusion token；
    # 少算一个 token 会把 action hidden states 整体错位，得到很弱甚至误导的 patch 梯度。
    text_hidden_states = last_hidden_states[:, int(num_patches) : -1]
    selected_hidden = text_hidden_states[action_mask]
    expected = batch_size * chunk_len * action_dim
    if selected_hidden.shape[0] != expected:
        action_mask_count = int(action_mask.detach().sum().item())
        raise RuntimeError(
            "Action hidden-state count mismatch: "
            f"got {selected_hidden.shape[0]}, expected {expected}; "
            f"labels={tuple(labels.shape)} actions={tuple(actions.shape)} "
            f"hidden_states={tuple(last_hidden_states.shape)} "
            f"action_mask_count={action_mask_count} chunk_len={chunk_len} "
            f"action_dim={action_dim} num_visual_tokens="
            f"{num_visual_tokens if num_visual_tokens is not None else 'unknown'} "
            f"num_prefix_tokens={num_patches} "
            f"num_images={num_images if num_images is not None else 'unknown'}. "
            "Check that RLDS future_action_window_size matches num_open_loop_steps-1 "
            "and that proprio is present when use_proprio=true."
        )

    selected_hidden = selected_hidden.reshape(batch_size, chunk_len * action_dim, -1)
    selected_hidden = selected_hidden.to(dtype=last_hidden_states.dtype)
    head = getattr(action_head, "module", action_head)
    predicted_actions = head.predict_action(selected_hidden)
    predicted_actions = predicted_actions.reshape_as(ground_truth_actions).float()
    ground_truth_actions = ground_truth_actions.float()

    info = {
        "action_head_total_values": float(predicted_actions.numel()),
        "action_chunk_len": float(chunk_len),
        "action_dim": float(action_dim),
        "selected_hidden_count": float(selected_hidden.shape[1]),
    }
    return predicted_actions, ground_truth_actions, info


def _time_decay_weights(
    *,
    chunk_len: int,
    device: torch.device,
    dtype: torch.dtype,
    time_decay: float,
) -> torch.Tensor:
    if not (0.0 < float(time_decay) <= 1.0):
        raise ValueError(f"time_decay must be in (0, 1], got {time_decay}")
    weights = torch.pow(
        torch.full((chunk_len,), float(time_decay), device=device, dtype=dtype),
        torch.arange(chunk_len, device=device, dtype=dtype),
    )
    weights = weights / weights.mean().clamp_min(1e-6)
    return weights.view(1, chunk_len, 1)


def _opposite_clean_boundary_target(
    clean_actions: torch.Tensor,
    ground_truth_actions: torch.Tensor,
    *,
    boundary_value: float,
    deadzone: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    clean_sign = torch.sign(clean_actions)
    gt_sign = torch.sign(ground_truth_actions)
    fallback_sign = torch.where(gt_sign == 0, torch.ones_like(gt_sign), gt_sign)
    use_clean = clean_actions.abs() >= float(deadzone)
    reference_sign = torch.where(use_clean, clean_sign, fallback_sign)
    target = torch.where(
        reference_sign >= 0,
        torch.full_like(clean_actions, -float(boundary_value)),
        torch.full_like(clean_actions, float(boundary_value)),
    )
    fallback_mask = (~use_clean).float()
    return target, fallback_mask


def _extract_projected_visual_features(
    *,
    outputs: Any,
    num_visual_tokens: int | None,
    hidden_state_layer: int = 0,
) -> tuple[torch.Tensor, str]:
    features = getattr(outputs, "projector_features", None)
    if torch.is_tensor(features):
        return features.float(), "projector_features"

    hidden_states = getattr(outputs, "hidden_states", None)
    if not hidden_states:
        raise RuntimeError(
            "Feature deviation requires projector_features or outputs.hidden_states; "
            "call VLA with output_hidden_states=True."
        )
    if num_visual_tokens is None:
        raise RuntimeError("Feature deviation fallback requires num_visual_tokens.")
    layer = int(hidden_state_layer)
    if layer < 0:
        layer = len(hidden_states) + layer
    if layer < 0 or layer >= len(hidden_states):
        raise ValueError(f"feature_hidden_state_layer={hidden_state_layer} out of range for {len(hidden_states)} layers")
    seq = hidden_states[layer]
    # Prismatic inserts projected visual tokens after BOS: [BOS, visual..., text...].
    start = 1 if seq.shape[1] >= int(num_visual_tokens) + 1 else 0
    end = start + int(num_visual_tokens)
    if end > seq.shape[1]:
        raise RuntimeError(
            f"Cannot slice visual tokens [{start}:{end}] from hidden_states[{layer}] shape={tuple(seq.shape)}"
        )
    return seq[:, start:end, :].float(), f"hidden_states[{layer}]_visual_tokens"


def _validate_hybrid_loss_config(config: dict[str, str]) -> None:
    bool_keys = [
        "pad_enabled",
        "psm_enabled",
        "robust_inner_enabled",
        "rollout_validation_enabled",
    ]
    for key in bool_keys:
        if _config_bool(config, key, False):
            raise NotImplementedError(f"{key}=true is not part of stage-1 hybrid patch training.")
    weight_keys = [
        "trajectory_curvature_weight",
        "repulsive_infonce_weight",
        "pad_weight",
        "psm_weight",
    ]
    for key in weight_keys:
        if abs(_config_float(config, key, 0.0)) > 0.0:
            raise NotImplementedError(f"{key}>0 is not part of stage-1 hybrid patch training.")
    weight_mode = str(config.get("loss_weight_mode") or "fixed").strip().lower()
    if weight_mode not in {"fixed", "gradient_calibrated_fixed"}:
        raise NotImplementedError(
            "Only loss_weight_mode=fixed or gradient_calibrated_fixed is implemented."
        )
    conflict_mode = str(config.get("gradient_conflict_mode") or "none").strip().lower()
    if conflict_mode == "pcgrad":
        raise NotImplementedError("gradient_conflict_mode=pcgrad is reserved but not enabled.")
    if conflict_mode != "none":
        raise ValueError(f"Unsupported gradient_conflict_mode={conflict_mode!r}")
    divergence_type = str(config.get("action_divergence_type") or "mse").strip().lower()
    if divergence_type != "mse":
        raise NotImplementedError(
            "Only action_divergence_type=mse is implemented; SmoothL1 must be a separate ablation."
        )


def oft_clean_reference_hybrid_attack_components(
    *,
    outputs: Any,
    clean_outputs: Any | None,
    labels: torch.Tensor,
    actions: torch.Tensor,
    action_head: Any,
    num_patches: int,
    maskidx: list[int],
    config: dict[str, str],
    num_visual_tokens: int | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Stage-1 OpenVLA-OFT attack loss using a detached clean reference.

    total = boundary_weight * action_boundary_mse
            - action_divergence_weight * action_divergence
            - feature_deviation_weight * feature_deviation
    """
    if not _config_bool(config, "clean_forward_enabled", True):
        raise RuntimeError("oft_clean_reference_hybrid_attack requires clean_forward_enabled=true.")
    if clean_outputs is None:
        raise RuntimeError("oft_clean_reference_hybrid_attack requires a no-grad clean forward.")
    _validate_hybrid_loss_config(config)

    adv_actions, ground_truth_actions, extract_info = _extract_oft_continuous_actions(
        outputs=outputs,
        labels=labels,
        actions=actions,
        action_head=action_head,
        num_patches=num_patches,
        num_visual_tokens=num_visual_tokens,
        num_images=int(config.get("num_images_in_input", 0)) or None,
        loss_name="oft_clean_reference_hybrid_attack",
    )
    with torch.no_grad():
        clean_actions, _clean_gt, _clean_info = _extract_oft_continuous_actions(
            outputs=clean_outputs,
            labels=labels,
            actions=actions,
            action_head=action_head,
            num_patches=num_patches,
            num_visual_tokens=num_visual_tokens,
            num_images=int(config.get("num_images_in_input", 0)) or None,
            loss_name="oft_clean_reference_hybrid_attack clean branch",
        )
    clean_actions = clean_actions.detach()

    action_dim = int(ground_truth_actions.shape[-1])
    wanted = _select_action_dims(maskidx, action_dim)
    adv_sel = adv_actions[:, :, wanted]
    clean_sel = clean_actions[:, :, wanted]
    gt_sel = ground_truth_actions[:, :, wanted]

    boundary_value = _config_float(
        config, "action_boundary_value", _config_float(config, "oft_action_boundary_value", 1.0)
    )
    deadzone = _config_float(config, "clean_action_deadzone", 0.05)
    target, fallback_mask = _opposite_clean_boundary_target(
        clean_sel,
        gt_sel,
        boundary_value=boundary_value,
        deadzone=deadzone,
    )
    time_decay = _config_float(config, "time_decay", 0.85)
    time_weights = _time_decay_weights(
        chunk_len=int(adv_sel.shape[1]),
        device=adv_sel.device,
        dtype=adv_sel.dtype,
        time_decay=time_decay,
    )

    boundary_mse = (((adv_sel - target) ** 2) * time_weights).mean()
    # action_divergence_type=mse. Do not silently switch to SmoothL1 in this combined EOT/calibration change.
    action_divergence = (((adv_sel - clean_sel) ** 2) * time_weights).mean()

    hidden_state_layer = int(float(str(config.get("feature_hidden_state_layer", "0")).strip()))
    adv_features, feature_source = _extract_projected_visual_features(
        outputs=outputs,
        num_visual_tokens=num_visual_tokens,
        hidden_state_layer=hidden_state_layer,
    )
    with torch.no_grad():
        clean_features, clean_feature_source = _extract_projected_visual_features(
            outputs=clean_outputs,
            num_visual_tokens=num_visual_tokens,
            hidden_state_layer=hidden_state_layer,
        )
    if feature_source != clean_feature_source:
        raise RuntimeError(
            f"Clean/adv feature source mismatch: clean={clean_feature_source}, adv={feature_source}"
        )
    clean_features = clean_features.detach()
    adv_norm = F.normalize(adv_features.float(), dim=-1, eps=1e-6)
    clean_norm = F.normalize(clean_features.float(), dim=-1, eps=1e-6)
    feature_cosine_distance = 1.0 - (adv_norm * clean_norm).sum(dim=-1)
    feature_deviation = feature_cosine_distance.mean()
    feature_raw_mse = F.mse_loss(adv_features.float(), clean_features.float())

    boundary_weight = _config_float(config, "action_boundary_weight", 1.0)
    action_divergence_weight = _config_float(config, "action_divergence_weight", 0.25)
    feature_deviation_weight = _config_float(config, "feature_deviation_weight", 0.1)
    boundary_term = boundary_weight * boundary_mse
    divergence_term = -action_divergence_weight * action_divergence
    feature_term = -feature_deviation_weight * feature_deviation
    total = boundary_term + divergence_term + feature_term

    with torch.no_grad():
        clean_adv_l2 = torch.linalg.vector_norm((adv_sel - clean_sel).reshape(adv_sel.shape[0], -1), dim=-1)
        adv_gt_l2 = torch.linalg.vector_norm((adv_sel - gt_sel).reshape(adv_sel.shape[0], -1), dim=-1)
        adv_target_l2 = torch.linalg.vector_norm((adv_sel - target).reshape(adv_sel.shape[0], -1), dim=-1)
        clean_target_l2 = torch.linalg.vector_norm((clean_sel - target).reshape(clean_sel.shape[0], -1), dim=-1)
        feature_token_l2 = torch.linalg.vector_norm(
            (adv_features.float() - clean_features.float()).reshape(adv_features.shape[0], -1),
            dim=-1,
        )

    info = {
        "loss_oft_clean_reference_hybrid_attack": float(total.detach().item()),
        "selected_action_values": float(adv_sel.numel()),
        "action_boundary_mse": float(boundary_mse.detach().item()),
        "action_divergence_score": float(action_divergence.detach().item()),
        "feature_deviation_score": float(feature_deviation.detach().item()),
        "feature_raw_mse": float(feature_raw_mse.detach().item()),
        "feature_cosine_distance_mean": float(feature_cosine_distance.detach().mean().item()),
        "feature_clean_norm_mean": float(torch.linalg.vector_norm(clean_features.float(), dim=-1).mean().item()),
        "feature_adv_norm_mean": float(torch.linalg.vector_norm(adv_features.float(), dim=-1).mean().item()),
        "feature_token_l2_mean": float(feature_token_l2.detach().mean().item()),
        "clean_action_abs_mean": float(clean_sel.detach().abs().mean().item()),
        "adv_action_abs_mean": float(adv_sel.detach().abs().mean().item()),
        "gt_action_abs_mean": float(gt_sel.detach().abs().mean().item()),
        "clean_adv_action_l2_mean": float(clean_adv_l2.detach().mean().item()),
        "adv_gt_action_l2_mean": float(adv_gt_l2.detach().mean().item()),
        "adv_target_action_l2_mean": float(adv_target_l2.detach().mean().item()),
        "clean_target_action_l2_mean": float(clean_target_l2.detach().mean().item()),
        "clean_action_deadzone_fallback_frac": float(fallback_mask.detach().mean().item()),
        "action_boundary_weight": float(boundary_weight),
        "action_divergence_weight": float(action_divergence_weight),
        "feature_deviation_weight": float(feature_deviation_weight),
        "loss_weight_mode": str(config.get("loss_weight_mode") or "fixed").strip().lower(),
        "action_divergence_type": str(config.get("action_divergence_type") or "mse").strip().lower(),
        "clean_action_deadzone": float(deadzone),
        "action_boundary_value": float(boundary_value),
        "time_decay": float(time_decay),
        "feature_source_projector_features": 1.0 if feature_source == "projector_features" else 0.0,
    }
    info.update(extract_info)
    components = {
        "total": total,
        "action_boundary_mse": boundary_mse,
        "action_divergence_score": action_divergence,
        "feature_deviation_score": feature_deviation,
        "boundary_objective": boundary_mse,
        "divergence_objective": -action_divergence,
        "feature_objective": -feature_deviation,
        "weighted_boundary": boundary_term,
        "weighted_divergence": divergence_term,
        "weighted_feature": feature_term,
    }
    return components, info


def oft_clean_reference_hybrid_attack_loss(
    *,
    outputs: Any,
    clean_outputs: Any | None,
    labels: torch.Tensor,
    actions: torch.Tensor,
    action_head: Any,
    num_patches: int,
    maskidx: list[int],
    config: dict[str, str],
    num_visual_tokens: int | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    components, info = oft_clean_reference_hybrid_attack_components(
        outputs=outputs,
        clean_outputs=clean_outputs,
        labels=labels,
        actions=actions,
        action_head=action_head,
        num_patches=num_patches,
        maskidx=maskidx,
        config=config,
        num_visual_tokens=num_visual_tokens,
    )
    return components["total"], info


def oft_action_mse_far_boundary_loss(
    *,
    outputs: Any,
    labels: torch.Tensor,
    actions: torch.Tensor,
    action_head: Any,
    num_patches: int,
    maskidx: list[int],
    boundary_value: float = 1.0,
    scale: float = 1.0,
    num_images: int | None = None,
    num_visual_tokens: int | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """直接攻击 OpenVLA-OFT 的连续动作头。

    OpenVLA-OFT 在 `use_l1_regression=true` 时，rollout 不使用离散 action-token
    argmax，而是从最后一层 hidden states 中取 action-token 位置，再送入
    `action_head.predict_action()` 得到归一化连续动作。因此 M 系列训练补丁时
    应该优化这个连续动作输出，而不是只优化 `outputs.logits`。
    """
    predicted_actions, ground_truth_actions, extract_info = _extract_oft_continuous_actions(
        outputs=outputs,
        labels=labels,
        actions=actions,
        action_head=action_head,
        num_patches=num_patches,
        num_visual_tokens=num_visual_tokens,
        num_images=num_images,
        loss_name="oft_action_mse_far_boundary",
    )
    batch_size, _chunk_len, action_dim = ground_truth_actions.shape
    wanted = _select_action_dims(maskidx, action_dim)

    pred_sel = predicted_actions[:, :, wanted]
    gt_sel = ground_truth_actions[:, :, wanted]
    boundary = float(boundary_value)
    target = torch.where(gt_sel >= 0, torch.full_like(gt_sel, -boundary), torch.full_like(gt_sel, boundary))

    loss = F.mse_loss(scale * pred_sel.contiguous(), scale * target.contiguous())
    with torch.no_grad():
        action_l2 = torch.linalg.vector_norm((predicted_actions - ground_truth_actions).reshape(batch_size, -1), dim=-1)
        target_l2 = torch.linalg.vector_norm((pred_sel - target).reshape(batch_size, -1), dim=-1)
        selected_abs_error = (pred_sel - gt_sel).abs().mean()

    info = {
        "loss_oft_action_mse_far_boundary": float(loss.detach().item()),
        "selected_action_values": float(pred_sel.numel()),
        "pred_action_abs_mean": float(pred_sel.detach().abs().mean().item()),
        "gt_action_abs_mean": float(gt_sel.detach().abs().mean().item()),
        "selected_action_abs_error_mean": float(selected_abs_error.detach().item()),
        "action_l2_mean": float(action_l2.detach().mean().item()),
        "target_l2_mean": float(target_l2.detach().mean().item()),
    }
    info.update(extract_info)
    return loss, info


# ---------------------------------------------------------------------------
# 攻击损失调度器
# ---------------------------------------------------------------------------

def compute_attack_loss(
    *,
    outputs: Any,
    labels: torch.Tensor,
    config: dict[str, str],
    clean_outputs: Any | None = None,
    actions: torch.Tensor | None = None,
    action_head: Any | None = None,
    num_patches: int | None = None,
    num_visual_tokens: int | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """根据配置中的 attack_loss_type 调度攻击损失。

    支持的 attack_loss_type:
      - "oft_clean_reference_hybrid_attack": clean action 参考 + 动作/视觉发散（M 系列 stage-1）
      - "oft_action_mse_far_boundary": 直接用 MSE 攻击 OpenVLA-OFT 连续 action head（旧版对照）
      - "uada_far_boundary": 对所有选中 DoF 用 UADA 远边界损失
      - "upa_position": 同上但默认只攻击位置 DoF (0,1,2)
      - "clean_adv_kl": 最大化攻击前后动作分布的 KL 散度（需要额外干净前向）
      - "model_ce_max" / "legacy_model_loss": 兜底，直接用 -outputs.loss

    参数:
      outputs:      模型前向传播输出（含 logits 和/或 loss）
      labels:       batch 中的 GT labels
      config:       实验配置字典
    clean_outputs: 干净前向传播输出（仅 clean_adv_kl 需要）

    返回:
      (loss, info_dict)
      loss: 标量，调用 .backward() 后 patch.grad 即为攻击梯度
      info_dict: 训练日志字段
    """
    loss_type = str(config.get("attack_loss_type") or "oft_clean_reference_hybrid_attack").strip().lower()
    action_dim = int(config.get("action_dim", 7))
    n_bins = int(config.get("n_action_bins", 256))

    # 根据损失类型决定默认 maskidx
    if loss_type == "upa_position":
        maskidx = parse_maskidx(config.get("maskidx"), default=(0, 1, 2))
    else:
        maskidx = parse_maskidx(config.get("maskidx"), default=(0, 1, 2, 3, 4, 5, 6))

    logits = getattr(outputs, "logits", None)

    # ---- OpenVLA-OFT clean-reference hybrid 损失 ----
    if loss_type == "oft_clean_reference_hybrid_attack":
        loss, info = oft_clean_reference_hybrid_attack_loss(
            outputs=outputs,
            clean_outputs=clean_outputs,
            labels=labels,
            actions=actions,
            action_head=action_head,
            num_patches=int(num_patches) if num_patches is not None else None,
            maskidx=maskidx,
            config=config,
            num_visual_tokens=num_visual_tokens,
        )
        info.update({"attack_loss_type": loss_type, "maskidx": ",".join(map(str, maskidx))})
        return loss, info

    # ---- OpenVLA-OFT 连续 action head 损失 ----
    if loss_type in {
        "oft_action_mse_far_boundary",
        "oft_action_far_boundary",
        "oft_uada_far_boundary",
    }:
        loss, info = oft_action_mse_far_boundary_loss(
            outputs=outputs,
            labels=labels,
            actions=actions,
            action_head=action_head,
            num_patches=int(num_patches) if num_patches is not None else None,
            maskidx=maskidx,
            boundary_value=float(config.get("oft_action_boundary_value", 1.0)),
            scale=float(config.get("oft_action_loss_scale", 1.0)),
            num_images=int(config.get("num_images_in_input", 0)) or None,
            num_visual_tokens=num_visual_tokens,
        )
        info.update({"attack_loss_type": "oft_action_mse_far_boundary", "maskidx": ",".join(map(str, maskidx))})
        return loss, info

    # ---- UADA / UPA 系列：基于动作 token 空间的远边界损失 ----
    if loss_type in {"uada", "uada_far_boundary", "upa", "upa_position"}:
        if logits is None:
            raise RuntimeError(f"{loss_type} requires outputs.logits, got none.")
        loss, info = uada_far_boundary_loss(
            logits, labels, maskidx=maskidx, action_dim=action_dim, n_bins=n_bins
        )
        info.update({"attack_loss_type": loss_type, "maskidx": ",".join(map(str, maskidx))})
        return loss, info

    # ---- Clean-Adv KL 散度损失 ----
    if loss_type == "clean_adv_kl":
        if logits is None or clean_outputs is None or getattr(clean_outputs, "logits", None) is None:
            raise RuntimeError(
                "clean_adv_kl requires both clean_outputs.logits and adv outputs.logits."
            )
        loss, info = logits_kl_discrepancy_loss(
            logits, clean_outputs.logits, labels,
            maskidx=maskidx, action_dim=action_dim, n_bins=n_bins,
        )
        info.update({"attack_loss_type": loss_type, "maskidx": ",".join(map(str, maskidx))})
        return loss, info

    # ---- 兜底：最大化模型 CE loss ----
    if loss_type in {"model_ce_max", "legacy_model_loss"}:
        model_loss = getattr(outputs, "loss", None)
        if model_loss is None:
            raise RuntimeError(
                "model_ce_max requires outputs.loss; logits.mean fallback is intentionally disabled."
            )
        loss = -model_loss.float()
        return loss, {
            "attack_loss_type": loss_type,
            "model_loss": float(model_loss.detach().float().item()),
            "maskidx": ",".join(map(str, maskidx)),
        }

    raise ValueError(f"Unsupported attack_loss_type={loss_type}")
