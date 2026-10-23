"""Check geometric target definitions without a VLA or simulator."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'VLAAttacker/spatial_relation'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

targets=load('targets')
matched=load('matched_targets')

class TargetTests(unittest.TestCase):
    def test_retreat_constrains_radial_and_preserves_tangent(self):
        clean=np.array([[.02,.01,-.01],[.03,-.02,.015]])
        actual=targets.retreat_target(clean,np.array([1.,0,0]),np.full(3,-.05),np.full(3,.05),.005)
        np.testing.assert_allclose(actual[:,0],-.005,atol=1e-8)
        np.testing.assert_allclose(actual[:,1:],clean[:,1:],atol=1e-8)

    def test_feasible_command_is_identity(self):
        value=np.array([[-.02,.01,0]])
        actual=targets.retreat_target(value,np.array([1.,0,0]),np.full(3,-.05),np.full(3,.05),.005)
        np.testing.assert_allclose(value,actual,atol=1e-8)

    def test_infeasible_target_rejected(self):
        with self.assertRaises(ValueError):
            targets.retreat_target(np.zeros((1,3)),np.array([1.,0,0]),np.full(3,-.01),np.full(3,.01),.02)

    def test_counterpart_rotation_has_equal_length_and_shared_bounds(self):
        clean=np.zeros((1,8,3));clean[...,1]=.049
        original=clean.copy();original[...,0]=-.02
        left,right,alpha=matched.matched_targets(clean,original,np.array([1.,0,0]),np.array([0.,-1,0]))
        self.assertGreaterEqual(alpha,0);self.assertLessEqual(alpha,1)
        self.assertLessEqual(max(abs(left).max(),abs(right).max()),.05000001)
        np.testing.assert_allclose(np.linalg.norm(left-clean,axis=-1),np.linalg.norm(right-clean,axis=-1),atol=1e-8)
        np.testing.assert_allclose(left-clean,alpha*(original-clean),atol=1e-8)

    def test_opposite_rotation_maps_direction(self):
        x=np.array([1.,0,0]);rotation=matched.rotation_between(x,-x)
        np.testing.assert_allclose(rotation@x,-x,atol=1e-8)
        np.testing.assert_allclose(rotation.T@rotation,np.eye(3),atol=1e-8)

    def test_far_target_selects_most_distant_endpoint(self):
        values=np.array([-.04,.04,0])
        np.testing.assert_allclose(targets.far_target(values,np.full(3,-.05),np.full(3,.05)),[.05,-.05,.05])

if __name__=='__main__':unittest.main()
