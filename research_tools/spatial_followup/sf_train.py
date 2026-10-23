"""Reuse audited nonlinear two-branch VJP, with private output paths."""
import copy
import numpy as np
import torch
from sf_common import ROOT, config, source, native, require, atomic, read, training_module, candidates
from sr_engine import Engine as SourceEngine


class Engine(SourceEngine):
    def gate(self, row, patch, method, key):
        # Do not call the historical gate: its paths belong to a sealed run.
        ref = source.reference_engine_module()
        deployed = native.paste_numpy(row['observation'], patch.detach().clamp(0,1).mul(255).round().byte().permute(1,2,0).cpu().numpy())
        with torch.no_grad():
            pixels, _ = ref.patch_pixels(row['observation'], patch, self.wrapper)
            expected = self.wrapper.prepare_hf_inputs(copy.deepcopy(deployed), row['language'])['pixel_values']
            require(torch.equal(pixels.detach(), expected), 'Actual PNG preprocessing mismatch')
            action, caps = self.action(deployed, row['language'])
            official = np.asarray(self.wrapper.predict_action(copy.deepcopy(deployed), row['language'])).reshape(8,7)
            error = float(np.max(abs(self.physical(action)-official)))
            require(error <= config()['action_tolerance'], 'Native PNG action mismatch')
            zero, _ = self.action(deployed, row['language'], replacements={'dino': caps['dino'], 'siglip': caps['siglip']})
            require(torch.equal(action, zero), 'Zero replacement changed actions')
            clean, _ = self.action(row['observation'], row['language'])
            require(torch.equal(clean.cpu(), row['teacher']['clean']), 'Cached clean actions changed')
            stats = self.wrapper.vla.get_action_stats(self.wrapper.gen_cfg.unnorm_key)
            values = row['teacher']['clean'].numpy()[..., :3]
            lo, hi = np.array(stats['q01'])[:3], np.array(stats['q99'])[:3]
            prepared = np.clip(.5*(values+1)*(hi-lo+1e-8)+lo,-1,1)*.05
            command_error = float(np.max(abs(self.command(clean, clipped=True).cpu().numpy()-prepared)))
            require(command_error <= 1e-7, 'Prepared controller normalization mismatch')
        gradient, metrics = self.staged_gradient(row, patch, method)
        require(torch.isfinite(gradient).all() and gradient.norm() > 0, 'Nonfinite/zero RGB gradient')
        modules = (self.wrapper.vla, self.wrapper.action_head, self.wrapper.proprio_projector, self.wrapper.noisy_action_projector)
        require(all(not m.training and all(not p.requires_grad and p.grad is None for p in m.parameters()) for m in modules if m is not None), 'Victim not frozen')
        atomic({'status': 'PASS', 'png_pixels_exact': True, 'native_action_max_abs_error': error,
                'command_cache_error_m': command_error, 'zero_replacement_identity': True,
                'model_frozen': True, 'dual_branch_RGB_gradient': True, **metrics}, ROOT / f'reports/GATES/{key}.json')


def make_engine(checkpointing=False):
    return Engine(native.load_wrapper('libero_goal'), checkpointing)


def run(index, monitor):
    module = training_module()
    # One seed per worker: its two matched arms run serially, at most two GPUs total.
    seed = config()['seeds'][index]
    for i, (_, _, candidate_seed) in enumerate(candidates()):
        if seed == candidate_seed: module.run(i, monitor)


def select(monitor):
    training_module().select(monitor)
