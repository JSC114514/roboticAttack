"""CPU tests for physical state invariants and selection, without GL or policy."""
from pathlib import Path
from types import SimpleNamespace
import sys,unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from diag_common import choose,conditions
from diag_geometry import set_translation,validity,vector

class GeometryTests(unittest.TestCase):
    def test_selection_uses_physical_eligibility_and_earlier_tie(self):
        def row(distance,decision,visible=1,contact=False,grasp=False):
            return dict(row=dict(decision_block=decision),visible_pixels=visible,
                        truth=dict(distance_m=distance,contact=contact,grasp=grasp))
        a=[row(.01,1,contact=True),row(.02,2,visible=0),row(.03,4,grasp=True),row(.04,6),row(.04,5)]
        self.assertEqual(choose(a)['row']['decision_block'],5)
        self.assertIsNone(choose(a[:3]))
    def test_factorial_signs_are_retained(self):
        c=conditions();self.assertEqual(len(c),9)
        self.assertEqual(len({r['name'] for r in c}),9)
        self.assertEqual({(r['target_shift_m'],r['camera_shift_m']) for r in c},
                         {(a,b) for a in (-.02,0,.02) for b in (-.02,0,.02)})
    def test_real_mujoco_translation_changes_only_target_position(self):
        import mujoco
        m=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
        <body name="robot" pos="1 0 0"><geom type="box" size=".05 .05 .05"/></body>
        <body name="target" pos="0 0 .1"><freejoint/><geom type="box" size=".02 .02 .02" mass="1"/></body>
        <body name="other" pos=".4 0 .1"><freejoint/><geom type="box" size=".02 .02 .02" mass="1"/></body>
        </worldbody></mujoco>''')
        d=mujoco.MjData(m);mujoco.mj_forward(m,d)
        env=SimpleNamespace(sim=SimpleNamespace(data=d,forward=lambda:mujoco.mj_forward(m,d)))
        q=d.qpos.copy();v=d.qvel.copy();t=d.time;robot=d.xpos[1].copy();other=d.xpos[3].copy()
        set_translation(env,dict(qpos_address=0),np.array([.02,0,0]))
        self.assertAlmostEqual(d.qpos[0]-q[0],.02)
        np.testing.assert_array_equal(d.qpos[3:],q[3:]);np.testing.assert_array_equal(d.qvel,v)
        np.testing.assert_array_equal(d.xpos[1],robot);np.testing.assert_array_equal(d.xpos[3],other)
        self.assertEqual(d.time,t)
        after=d.qpos.copy();set_translation(env,dict(qpos_address=0),np.zeros(3))
        np.testing.assert_array_equal(d.qpos,after)
    def test_mujoco_metric_forward_refreshes_pose_without_advancing_time(self):
        import mujoco
        m=mujoco.MjModel.from_xml_string('<mujoco><option gravity="0 0 0" timestep=".002"/><worldbody><body><freejoint/><geom type="sphere" size=".02" mass="1"/></body></worldbody></mujoco>')
        d=mujoco.MjData(m);d.qvel[0]=1;mujoco.mj_forward(m,d);mujoco.mj_step(m,d)
        self.assertNotEqual(d.qpos[0],d.xpos[1,0])
        q=d.qpos.copy();v=d.qvel.copy();now=d.time
        mujoco.mj_forward(m,d)
        self.assertAlmostEqual(d.xpos[1,0],d.qpos[0])
        np.testing.assert_array_equal(d.qpos,q);np.testing.assert_array_equal(d.qvel,v)
        self.assertEqual(d.time,now)
    def test_robot_base_x_can_differ_from_world_x(self):
        r=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        env=SimpleNamespace(env=SimpleNamespace(robots=[SimpleNamespace(robot_model=SimpleNamespace(root_body='base'))]),
            sim=SimpleNamespace(model=SimpleNamespace(body_name2id=lambda n:0),data=SimpleNamespace(body_xmat=r.reshape(1,9))))
        np.testing.assert_allclose(vector(env,.02),[0,.02,0],atol=1e-12)
    def test_support_loss_and_penetration_separate(self):
        base=[dict(pair=(1,2),other=2,dist=-.0001,support=True,hand=False)]
        self.assertEqual(validity(base,base),[])
        self.assertIn('lost_original_support',validity(base,[]))
        obstacle=base+[dict(pair=(1,3),other=3,dist=-.003,support=False,hand=False)]
        self.assertIn('new_or_increased_penetration',validity(base,obstacle))
        self.assertIn('baseline_has_no_identified_support',validity([],[]))

if __name__=='__main__':unittest.main()
