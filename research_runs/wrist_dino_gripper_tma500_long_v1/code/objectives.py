"""Pure tensor definitions for region CF and specifically wrong gripper opening."""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn.functional as F

from gripper_common import require


def open_loss(action, close_slots, target=1.):
    require(tuple(action.shape) == (1, 8, 7) and close_slots.shape == (8,), "Wrong action/slot shape")
    require(close_slots.dtype == torch.bool and close_slots.any(), "No close/hold target slots")
    return (action[0, close_slots, 6].float() - target).square().mean()


def region_value(clean, attacked, mask):
    require(clean.shape == attacked.shape and clean.shape[:2] == (1, 256), "Wrong DINO dense grid")
    require(mask.shape == (256,) and mask.dtype == torch.bool and mask.any(), "Empty/invalid region")
    return torch.where(mask[None, :, None], attacked, clean.detach())


def area_tokens(mask):
    """Conservative label/support mapping through resize and exact .9 center crop.

    RGB's first resize is Pillow LANCZOS. Labels use positive bilinear occupancy
    (no negative interpolation lobes); labels are not an RGB augmentation.
    """
    x = torch.as_tensor(mask).detach().float()
    require(x.ndim == 2, "Expected pixel mask")
    x = x[None, None]
    if x.shape[-2:] != (224, 224):
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    axis = torch.linspace(-math.sqrt(.9), math.sqrt(.9), 224, device=x.device)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    x = F.grid_sample(x, torch.stack((xx, yy), -1)[None], mode="bilinear", padding_mode="border", align_corners=True)
    return F.avg_pool2d(x, 14).flatten().clamp(0, 1)


def paired_masks(task_tokens, background_tokens, footprint, permutation):
    require(task_tokens.shape == background_tokens.shape == footprint.shape == (256,), "Region grid mismatch")
    task = task_tokens.bool() & ~footprint.bool()
    candidates = background_tokens.bool() & ~footprint.bool() & ~task_tokens.bool()
    order = torch.as_tensor(permutation, dtype=torch.long, device=task.device)
    require(order.shape == (256,) and torch.unique(order).numel() == 256 and order.min() == 0 and order.max() == 255,
            "Background permutation is not a complete fixed token order")
    ordered = order[candidates[order]]
    n = int(task.sum())
    if n == 0 or len(ordered) < n:
        return None
    background = torch.zeros_like(task)
    background[ordered[:n]] = True
    require(not (task & background).any() and int(background.sum()) == n, "Region pairing failed")
    return task, background


def executed_gripper(raw_gripper):
    return -np.sign(2 * np.asarray(raw_gripper, dtype=float) - 1)


def wrong_open_flags(clean, attacked, threshold=.5):
    clean, attacked = np.asarray(clean), np.asarray(attacked)
    require(clean.shape == attacked.shape == (8, 7), "Expected paired same-state 8x7 actions")
    return clean[:, 6] < threshold, (clean[:, 6] < threshold) & (attacked[:, 6] > threshold)


def patch_statistics(patch, initial):
    p = patch.detach().float().cpu()
    return {"rgb_mae_from_initial": float((p-initial.cpu()).abs().mean()),
            "rgb_l2_from_initial": float((p-initial.cpu()).norm()),
            "rgb_min": float(p.min()), "rgb_max": float(p.max()), "rgb_std": float(p.std()),
            "rgb_tv": float((p[:, :, 1:]-p[:, :, :-1]).abs().mean()
                            + (p[:, 1:, :]-p[:, :-1, :]).abs().mean())}


class GraspEvents:
    """Trajectory proxies, NOT ground truth attack-caused drops.

    Normal opening/release is tracked separately; a suspected-drop event requires
    same-state clean-close, attacked-open, prior stable lifted grasp, confirmed
    contact loss and a subsequent 2cm drop before contact reestablishes.
    """
    def __init__(self, thresholds):
        self.cfg = thresholds
        self.contact_run = self.loss_run = 0
        self.ever_grasp = self.ever_lift = False
        self.grasps = self.losses = self.drops = 0
        self.base_height = self.held_peak = None
        self.loss_open = False
        self.in_loss = False
        self.session_lifted = self.loss_lifted = False

    def update(self, grasp, height, wrong_open):
        if grasp is None or height is None or not np.isfinite(height):
            return {"event_valid": False, "double_finger_contact": None, "object_height": None}
        if self.base_height is None:
            self.base_height = float(height)
        if grasp:
            self.contact_run += 1
            self.loss_run = 0
            self.in_loss = False
            if self.contact_run == self.cfg["stable_grasp_steps"]:
                self.grasps += 1
                self.ever_grasp = True
                self.held_peak = float(height)
                self.session_lifted = False
            if self.contact_run >= self.cfg["stable_grasp_steps"]:
                self.held_peak = max(self.held_peak if self.held_peak is not None else height, height)
                self.session_lifted |= height >= self.base_height + self.cfg["lift_height_m"]
                self.ever_lift |= self.session_lifted
        else:
            if self.contact_run >= self.cfg["stable_grasp_steps"]:
                self.in_loss = True
                self.loss_open = bool(wrong_open)
                self.loss_lifted = self.session_lifted
                self.loss_run = 0
            self.contact_run = 0
            if self.in_loss:
                self.loss_run += 1
                self.loss_open |= bool(wrong_open)
                if self.loss_run == self.cfg["contact_loss_confirm_steps"]:
                    self.losses += 1
                if (self.loss_run >= self.cfg["contact_loss_confirm_steps"] and self.loss_lifted
                        and self.loss_open and self.held_peak is not None
                        and height <= self.held_peak - self.cfg["suspected_drop_height_m"]):
                    self.drops += 1
                    self.in_loss = False
        return {"event_valid": True, "double_finger_contact": bool(grasp), "object_height": float(height)}
