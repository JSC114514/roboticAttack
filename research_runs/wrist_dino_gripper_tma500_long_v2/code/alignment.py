"""Action interface diagnostics with slot/DoF details, not an efficacy gate."""
from __future__ import annotations

import numpy as np
from gripper_common import require


class NativeActionAlignmentError(RuntimeError):
    def __init__(self, details, record, masks):
        self.details, self.record, self.masks = details, record, masks
        super().__init__(f"Long native and differentiable action interfaces differ: "
                         f"task={details['task_id']} state={details['state_index']} decision={details['decision_index']} "
                         f"slot={details['max_error_slot']} dof={details['max_error_dof']} "
                         f"max_abs={details['max_abs_error']:.8g} tolerance={details['tolerance']}")


def action_alignment(record, normalized, unnormalized, tolerance=.02):
    native = np.asarray(record["native_action"],dtype=np.float32).reshape(8,7)
    ours = np.asarray(unnormalized,dtype=np.float32).reshape(8,7)
    grip = normalized.detach().float().cpu().numpy().reshape(8,7)[:,6]
    require(np.isfinite(native).all() and np.isfinite(ours).all() and np.isfinite(grip).all(),
            "Nonfinite action interface input")
    error = np.abs(ours-native)
    slot, dof = map(int,np.unravel_index(int(error.argmax()),error.shape))
    close_match = bool(np.array_equal(native[:,6]<.5,grip<.5))
    return {"task_id":int(record["task_id"]),"state_index":int(record["state_index"]),
            "decision_index":int(record["decision_index"]),"bank_index":int(record["bank_index"]),
            "max_abs_error":float(error.max()),"max_error_slot":slot,"max_error_dof":dof,
            "per_dof_max_abs":error.max(axis=0).tolist(),"native_action":native.tolist(),
            "differentiable_unnormalized_action":ours.tolist(),"normalized_gripper":grip.tolist(),
            "native_close_slots":(native[:,6]<.5).tolist(),"close_slots_match":close_match,
            "tolerance":tolerance,"pass":bool(error.max()<=tolerance and close_match)}
