"""ILPD engineering revision: calibrated CUDA startup-gradient comparison.

The V1 training loop, RGB/EOT pipeline and all scientific settings are inherited
unchanged. V1 sources/results remain available for auditing the zero-step failure.
"""
from __future__ import annotations

import json

import torch
import torch.nn.functional as F

from white_patch.UADA_OFT_ILPD import ExploringOFTILPDAttacker, source_manifest as v1_source_manifest
from white_patch.UADA_OFT_ILA import seed_transform
from white_patch.UADA_OFT_DINO import require, sha256_file
from white_patch.oft_ilpd_features import METHODS, transform_dense

# Engineering startup safeguards, NOT attack optimization hyperparameters.
GRADIENT_RELATIVE_CAP = 0.05
GRADIENT_COSINE_FLOOR = 0.995
GRADIENT_CALIBRATION_MULTIPLIER = 4.0
GRADIENT_RELATIVE_FLOOR = 1e-6


def gradient_diagnostics(gradient, reference):
    require(gradient.shape == reference.shape, "Startup RGB gradient shape changed")
    a, b = gradient.detach().cpu().double().flatten(), reference.detach().cpu().double().flatten()
    require(bool(torch.isfinite(a).all() and torch.isfinite(b).all()), "Nonfinite startup RGB gradient")
    an, bn = float(a.norm()), float(b.norm())
    require(min(an, bn) > 1e-12, "Zero startup RGB gradient")
    diff = a - b
    return {"relative_l2": float(diff.norm()) / max(an, bn),
            "max_abs_error": float(diff.abs().max()),
            "cosine": max(-1., min(1., float(torch.dot(a, b)) / (an * bn))),
            "gradient_norm": an, "reference_norm": bn,
            "legacy_elementwise_close": bool(torch.allclose(gradient, reference, rtol=1e-5, atol=1e-6))}


def require_gradient_stability(metrics, allowance=None):
    require(metrics["relative_l2"] <= GRADIENT_RELATIVE_CAP
            and metrics["cosine"] >= GRADIENT_COSINE_FLOOR,
            "Startup RGB gradient changes exceed the hard engineering stability bounds")
    if allowance is not None:
        require(metrics["relative_l2"] <= allowance,
                "Identity RGB gradient drift exceeds measured ordinary-forward repeatability")


def calibrated_allowance(normal_comparisons):
    require(bool(normal_comparisons), "Ordinary-gradient calibration missing")
    for metrics in normal_comparisons:
        require_gradient_stability(metrics)
    noise = max(metrics["relative_l2"] for metrics in normal_comparisons)
    return min(GRADIENT_RELATIVE_CAP, max(GRADIENT_RELATIVE_FLOOR, GRADIENT_CALIBRATION_MULTIPLIER * noise))


def source_manifest(cfg, wrapper=None):
    sources = v1_source_manifest(cfg, wrapper)
    from pathlib import Path
    project = Path(__file__).resolve().parents[2]
    for relative in ("VLAAttacker/white_patch/UADA_OFT_ILPD_V2.py",
                     "VLAAttacker/UADA_OFT_ILPD_V2_wrapper.py",
                     "slurm/exploring_oft_dino/61_train_dino_ilpd_refine200_v2_3090.sbatch",
                     "configs/exploring_oft_dino/object_dino_ilpd_refine200_v2.json",
                     "tests/test_exploring_oft_ilpd_v2.py"):
        path = project / relative
        sources[str(path.resolve())] = sha256_file(path)
    return sources


class ExploringOFTILPDV2Attacker(ExploringOFTILPDAttacker):
    def _validate_ilpd_runtime(self, record, reference, cfg, method):
        validation = self._validate_runtime(record, reference, cfg)
        validation.update(status="CHECKING", engineering_revision=2)
        audit = validation["ilpd"] = {"method": method, "phase": "clean_anchor", "checks": [],
                                      "normal_repeat_comparisons": [], "normal_reference_count": 3,
                                      "relative_cap": GRADIENT_RELATIVE_CAP,
                                      "cosine_floor": GRADIENT_COSINE_FLOOR,
                                      "calibration_multiplier": GRADIENT_CALIBRATION_MULTIPLIER,
                                      "relative_floor": GRADIENT_RELATIVE_FLOOR}
        # Keep partial metrics available to inherited failure handling.
        self.runtime_alignment = validation
        inputs, proprio = self._base_inputs(record)
        obs = record["observation"]
        seed = cfg["seed"] * 10_000_000 + (cfg["base_outer_iterations"] + 1) * 10_000
        low = torch.tensor(cfg["action_head_q01"], device=reference.device).reshape(1, 1, 7)
        high = torch.tensor(cfg["action_head_q99"], device=reference.device).reshape(1, 1, 7)
        with torch.no_grad():
            validation["full_forward_calls"] += 1
            action, dino, siglip = self._forward(inputs, proprio, obs, None, False, capture_siglip=True)
        require(siglip is not None, "Clean SigLIP anchor missing")
        anchors = {"dino": dino.detach(), "siglip": siglip.detach()}
        target = torch.where(action >= (low + high) / 2, low, high).detach()

        def sample(forward_method, alpha):
            # Independent graph and leaf: no accumulated gradients/retained state.
            patch = reference.detach().clone().requires_grad_(True)
            seed_transform(seed)
            validation["full_forward_calls"] += 1
            result, capture = self._modified_forward(
                inputs, proprio, obs, patch, cfg["geometry"], forward_method, anchors, alpha)
            selected = "siglip" if forward_method == METHODS[3] else "dino"
            modified = forward_method != METHODS[0] and alpha != 1.
            local_stats = None
            loss = F.mse_loss(result[0][..., cfg["maskidx"]], target[..., cfg["maskidx"]])
            if modified:
                raw, forwarded = capture[selected][1]["raw"], capture[selected][1]["forward"]
                gradient, raw_grad, forwarded_grad = torch.autograd.grad(loss, (patch, raw, forwarded))
                local_stats = gradient_diagnostics(raw_grad, alpha * forwarded_grad)
                local_stats["passed"] = bool(torch.allclose(raw_grad.float(), alpha * forwarded_grad.float(),
                                                            rtol=1e-5, atol=1e-6))
            else:
                gradient = torch.autograd.grad(loss, patch)[0]
            require(bool(torch.isfinite(gradient).all()) and float(gradient.norm()) > 1e-12,
                    "Real-model action-to-RGB gradient is zero/nonfinite")
            features = {}
            for branch, entries in capture.items():
                features[branch] = []
                for index, entry in enumerate(entries):
                    raw, forwarded = entry["raw"], entry["forward"]
                    expect_identity = not (modified and branch == selected and index == 1)
                    if expect_identity:
                        # A numerical clone with a changed derivative must NOT pass.
                        require(forwarded is raw, "Identity hook changed the actual feature Tensor/graph")
                    expected = (transform_dense(raw.detach(), anchors[branch], alpha, forward_method == METHODS[1])
                                if not expect_identity else raw.detach())
                    require(torch.equal(forwarded.detach(), expected), "Actual forwarded feature formula mismatch")
                    features[branch].append({"raw": raw.detach(), "forward": forwarded.detach()})
            return {"action": result[0].detach(), "gradient": gradient.detach(),
                    "features": features, "local_gradient": local_stats}

        def compare_forward(current, base, identity_actions):
            for branch, entries in current["features"].items():
                for index, entry in enumerate(entries):
                    require(torch.equal(entry["raw"], base["features"][branch][index]["raw"]),
                            "EOT pairing or raw Front/encoder output drift")
            if identity_actions:
                require(torch.equal(current["action"], base["action"]),
                        "Identity/backward-only forward changed actions")

        audit["phase"] = "ordinary_gradient_calibration"
        normal = []
        for index in range(3):
            current = sample(METHODS[0], 1.)
            if normal:
                compare_forward(current, normal[0], True)
            for previous_index, previous in enumerate(normal):
                metrics = gradient_diagnostics(current["gradient"], previous["gradient"])
                metrics.update(left_repeat=index, right_repeat=previous_index)
                audit["normal_repeat_comparisons"].append(metrics)
                print("ILPD_V2_ORDINARY_GRADIENT " + json.dumps(metrics), flush=True)
                require_gradient_stability(metrics)
            normal.append(current)
        allowance = audit["identity_relative_allowance"] = calibrated_allowance(audit["normal_repeat_comparisons"])

        for alpha in (1., cfg["feature_alpha"]):
            audit["phase"] = f"candidate_alpha_{alpha}"
            current = sample(method, alpha)
            identity_forward = alpha == 1. or method in (METHODS[0], METHODS[1])
            compare_forward(current, normal[0], identity_forward)
            check = {"alpha": alpha, "formula_exact": True,
                     "front_and_other_encoder_unchanged": True,
                     "identity_tensor_graph_unchanged": alpha == 1. or method == METHODS[0],
                     "patch_grad_norm": float(current["gradient"].norm()),
                     "local_gradient": current["local_gradient"], "identity_gradient_comparisons": []}
            audit["checks"].append(check)
            if alpha == 1. or method == METHODS[0]:
                for index, base in enumerate(normal):
                    metrics = gradient_diagnostics(current["gradient"], base["gradient"])
                    metrics["normal_reference"] = index
                    check["identity_gradient_comparisons"].append(metrics)
                print("ILPD_V2_IDENTITY_GRADIENT " + json.dumps(check), flush=True)
                for metrics in check["identity_gradient_comparisons"]:
                    require_gradient_stability(metrics, allowance)
            if current["local_gradient"] is not None:
                require(current["local_gradient"]["passed"], "Actual encoder local gradient does not match alpha")
            print("ILPD_V2_RUNTIME_CHECK " + json.dumps(check), flush=True)
        audit["phase"] = "complete"
        validation["status"] = "PASS"
        return validation
