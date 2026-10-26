"""Frozen DINO relation decoder and geometric controls; independent of runtime."""
import math
import numpy as np
import torch
import torch.nn.functional as F


class Readout:
    def __init__(self, path, device='cpu'):
        with np.load(path, allow_pickle=False) as z:
            self.mean = torch.tensor(z['mean'], device=device, dtype=torch.float64)
            self.scale = torch.tensor(z['scale'], device=device, dtype=torch.float64)
            self.coefficients = torch.tensor(z['coefficients'][:, :3], device=device, dtype=torch.float64)
            self.intercept = torch.tensor(z['intercept'][:3], device=device, dtype=torch.float64)
        self.weight = self.coefficients / self.scale[:, None]
        self.inverse = torch.linalg.pinv(self.weight)
        if self.weight.shape != (16384, 3) or int(torch.linalg.matrix_rank(self.weight)) != 3:
            raise RuntimeError('XYZ relation readout is not full rank')

    @staticmethod
    def pool(tokens, rounded=True):
        x = torch.as_tensor(tokens).reshape(-1, 16, 16, 1024).permute(0, 3, 1, 2)
        # Match atlas: float32 pooling, then float16 cache. Tensor casts retain gradients.
        h = F.adaptive_avg_pool2d(x.float() if rounded else x, (4, 4))
        h = h.permute(0, 2, 3, 1).reshape(-1, 16384)
        return h.to(torch.float16).to(torch.float64) if rounded else h.to(torch.float64)

    def from_grid(self, grid):
        return ((grid - self.mean) / self.scale) @ self.coefficients + self.intercept

    def __call__(self, tokens):
        return self.from_grid(self.pool(tokens))

    @staticmethod
    def expand(grid):
        return grid.reshape(4, 4, 1024).repeat_interleave(4, 0).repeat_interleave(4, 1).reshape(1, 256, 1024)

    def intervention(self, delta):
        delta = torch.as_tensor(delta, device=self.weight.device, dtype=torch.float64)
        grid = delta @ self.inverse
        if not torch.allclose(grid @ self.weight, delta, atol=1e-9, rtol=1e-8):
            raise RuntimeError('Pseudoinverse failed physical XYZ constraint')
        return self.expand(grid)

    def random_null(self, norm, seed):
        gen = torch.Generator(device='cpu').manual_seed(seed)
        u = torch.randn(16384, generator=gen, dtype=torch.float64).to(self.weight.device)
        grid = u - (u @ self.weight) @ self.inverse
        dense = self.expand(grid)
        dense = dense * (norm / dense.norm())
        check = self.pool(dense, rounded=False) @ self.weight
        if not torch.allclose(check, torch.zeros_like(check), atol=1e-9, rtol=0):
            raise RuntimeError('Random control leaks XYZ before model quantization')
        return dense


def relation_loss(predicted, clean, delta, scale=.03):
    return ((predicted - clean - delta).square().sum(-1) / scale**2).mean()


def gate_decision(rows):
    valid = [r for r in rows if r.get('applicable')]
    if not valid:
        return {'passed': False, 'applicable': 0, 'relation_median': None, 'advantage_median': None}
    relation = np.array([r['relation_score'] for r in valid])
    random = np.array([r['random_score'] for r in valid])
    return {'passed': bool(np.median(relation) > 0 and np.median(relation-random) > 0),
            'applicable': len(valid), 'relation_median': float(np.median(relation)),
            'advantage_median': float(np.median(relation-random))}


def learning_rate(step, total=2000, warmup=20, peak=.002):
    if step <= warmup:
        return peak * step / warmup
    return peak * .5 * (1 + math.cos(math.pi * (step-warmup) / (total-warmup)))


def sample_order(size, seed, epoch):
    import random
    indices = list(range(size))
    random.Random(seed + epoch).shuffle(indices)
    return indices
