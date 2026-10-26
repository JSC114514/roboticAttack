"""Numerical contracts for wrist routing and the constrained action objective."""
import copy
from pathlib import Path
import sys
import unittest
import numpy as np
import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from wr_engine import WristLowRank,route,Engine
from wr_math import objectives,objective,update_dual,learning_rate


class AdaptationContracts(unittest.TestCase):
    def test_zero_adapter_and_front_identity_after_update(self):
        torch.manual_seed(42)
        base=nn.Linear(3,4);layer=WristLowRank(base,2,2);x=torch.randn(2,3)
        expected=base(x).detach()
        with route([layer],True):self.assertTrue(torch.equal(layer(x),expected))
        with torch.no_grad():layer.B.add_(.1)
        with route([layer],False):self.assertTrue(torch.equal(layer(x),expected))
        with route([layer],True):self.assertFalse(torch.equal(layer(x),expected))
        self.assertFalse(layer.enabled)
        self.assertTrue(all(not p.requires_grad for p in base.parameters()))

    def test_route_restores_after_exception(self):
        m=WristLowRank(nn.Linear(2,2),1,1)
        try:
            with route([m],True):raise ValueError('test')
        except ValueError:pass
        self.assertFalse(m.enabled)

    def test_staged_vjp_matches_direct_nonlinear_chain(self):
        torch.manual_seed(42)
        layer=WristLowRank(nn.Linear(3,4),2,2)
        with torch.no_grad():layer.B.fill_(.05)
        class Vision(nn.Module):
            def forward(self,x):return torch.tanh(layer(x.mean((2,3))))[:,None,:].repeat(1,256,256)
        engine=Engine.__new__(Engine);engine.dino=Vision();engine.adapters={'l':layer}
        engine.parameters=[layer.A,layer.B]
        pixels=torch.randn(1,12,2,2)
        engine.inputs=lambda obs,lang:dict(pixel_values=pixels)
        def action(obs,lang,pixels=None,replacement=None):
            if replacement is None:
                with route([layer],True):v=engine.dino(pixels[:,6:9])
            else:v=replacement
            # A nonlinear frozen 'controller' after the visual boundary.
            return torch.sin(v.mean()).expand(8,7)+torch.tanh(v[0,:8,:7])
        engine.action=action;g=torch.randn(8,7)
        direct=action(None,None,pixels=pixels)
        expected=torch.autograd.grad(direct,engine.parameters,grad_outputs=g)
        engine.vjp(None,None,g)
        for p,want in zip(engine.parameters,expected):torch.testing.assert_close(p.grad,want,rtol=1e-5,atol=1e-6)

    def test_loss_distinguishes_invariance_and_real_response(self):
        a=torch.zeros(8,7);b=torch.ones(8,7)*.1;y2=torch.ones(8,7)*.2
        v=objectives(a,a,a,b,a,y2)
        self.assertEqual(float(v['C']),0.)
        self.assertGreater(float(v['R']),0.)
        v2=objectives(a,a,a,y2,a,y2)
        self.assertEqual(float(v2['B']),0.);self.assertEqual(float(v2['R']),0.)

    def test_constraint_formula_and_multiplier_direction(self):
        v={k:torch.tensor(x) for k,x in dict(B=2.,R=.5,C=.25).items()}
        baseline=dict(B=1.,R=1.,C=.5);spec=dict(method='constrained',tau=.1)
        actual=objective(v,baseline,spec,dict(B=2.,R=3.))
        want=.25/(.5+1e-6)+2*(2-1.1)/(1+1e-6)+3*(.5-1.1)/(1+1e-6)
        self.assertAlmostEqual(float(actual),want,places=6)
        dual=dict(B=1.,R=1.);ema=dict(B=0.,R=0.)
        update_dual(dual,ema,{k:float(x) for k,x in v.items()},baseline,.1,
                    dict(ema=.95,lr=.01,maximum=50.,epsilon=1e-6))
        self.assertGreater(dual['B'],1.);self.assertLess(dual['R'],1.)

    def test_learning_rate_endpoints(self):
        cfg=dict(warmup=50,lr=1e-4,min_lr=1e-5)
        self.assertAlmostEqual(learning_rate(49,1000,cfg),1e-4)
        self.assertAlmostEqual(learning_rate(999,1000,cfg),1e-5)


if __name__=='__main__':unittest.main()
