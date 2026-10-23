"""Read-only physics-substep sampling; aggregate once per control step."""
import numpy as np
from rr_common import require
from rr_math import orientation_error_radians

class PhysicsLogger:
    def __init__(self,env,force_function=None):
        self.env=env;self.sim=env.sim;self.robot=env.env.robots[0];self.force_function=force_function;self.armed=False
    def __enter__(self):
        if self.force_function is None:
            import mujoco
            self.force_function=mujoco.mj_contactForce
        m=self.sim.model;root=int(m.body_name2id(self.robot.robot_model.root_body));self.robot_bodies=set()
        for body in range(m.nbody):
            current=body
            while current!=0 and current!=root:current=int(m.body_parentid[current])
            if current==root:self.robot_bodies.add(body)
        self.had_override='step' in vars(self.sim);self.previous_override=vars(self.sim).get('step');self.original=self.sim.step
        def step(*args,**kwargs):
            result=self.original(*args,**kwargs)
            if self.armed:self.sample()
            return result
        self.sim.step=step;return self
    def arm(self):
        self.armed=True;self.count=0;self.contacts={};self.position_peak=0.;self.orientation_peak=0.;self.torque_clip_peak=0.
        self.actuator_bound_hits=0;self.joint_margin=None;self.last=None
    def sample(self):
        sim=self.sim;m=sim.model;d=sim.data;c=self.robot.controller;self.count+=1
        site=int(self.robot.eef_site_id);pos=np.array(d.site_xpos[site],copy=True);ori=np.asarray(d.site_xmat[site]).reshape(3,3)
        goal=np.array(c.goal_pos,copy=True);goal_ori=np.array(c.goal_ori,copy=True)
        require(np.isfinite(pos).all() and np.isfinite(ori).all() and np.isfinite(goal).all() and np.isfinite(goal_ori).all(),'Nonfinite controller pose')
        position=float(np.linalg.norm(goal-pos));angle=orientation_error_radians(goal_ori,ori)
        self.position_peak=max(self.position_peak,position);self.orientation_peak=max(self.orientation_peak,angle)
        raw=np.array(c.torques,copy=True);applied=np.array(self.robot.torques,copy=True)
        require(np.isfinite(raw).all() and np.isfinite(applied).all(),'Nonfinite arm torque')
        self.torque_clip_peak=max(self.torque_clip_peak,float(np.max(np.abs(raw-applied))))
        limited=np.asarray(m.actuator_ctrllimited,dtype=bool);ctrl=np.asarray(d.ctrl);ranges=np.asarray(m.actuator_ctrlrange)
        self.actuator_bound_hits+=int(np.count_nonzero(limited&((ctrl<=ranges[:,0]+1e-8)|(ctrl>=ranges[:,1]-1e-8))))
        joints=np.flatnonzero(np.asarray(m.jnt_limited,dtype=bool));q=np.asarray(d.qpos)[np.asarray(m.jnt_qposadr)[joints]]
        margin=np.minimum(q-m.jnt_range[joints,0],m.jnt_range[joints,1]-q)
        self.joint_margin=margin.copy() if self.joint_margin is None else np.minimum(self.joint_margin,margin)
        for i in range(d.ncon):
            contact=d.contact[i];g1,g2=int(contact.geom1),int(contact.geom2);key=tuple(sorted((g1,g2)))
            force=np.zeros(6,dtype=np.float64);self.force_function(m._model,d._data,i,force)
            require(np.isfinite(force).all(),'Nonfinite MuJoCo contact wrench')
            if key not in self.contacts:
                self.contacts[key]={'geom1':m.geom_id2name(key[0]) or str(key[0]),'geom2':m.geom_id2name(key[1]) or str(key[1]),
                    'geom1_id':key[0],'geom2_id':key[1],
                    'body1':m.body_id2name(int(m.geom_bodyid[key[0]])),'body2':m.body_id2name(int(m.geom_bodyid[key[1]])),
                    'involves_robot':int(m.geom_bodyid[key[0]]) in self.robot_bodies or int(m.geom_bodyid[key[1]]) in self.robot_bodies,
                    'peak_normal_force_n':0.,'peak_force_norm_n':0.,'first_substep_time_s':float(d.time),
                    'last_substep_time_s':float(d.time),'samples':0}
            record=self.contacts[key];record['samples']+=1;record['last_substep_time_s']=float(d.time)
            record['peak_normal_force_n']=max(record['peak_normal_force_n'],float(abs(force[0])))
            record['peak_force_norm_n']=max(record['peak_force_norm_n'],float(np.linalg.norm(force[:3])))
        self.last={'controller_goal_xyz':goal.tolist(),'controller_goal_orientation':goal_ori.reshape(-1).tolist(),
            'actual_site_xyz':pos.tolist(),'actual_site_orientation':ori.reshape(-1).tolist(),
            'tracking_error_final_m':position,'orientation_error_final_rad':angle,'raw_arm_torques':raw.tolist(),
            'applied_arm_torques':applied.tolist(),'limited_joint_names':[m.joint_id2name(int(j)) for j in joints],
            'joint_margin_units':'hinge radians / slide meters, according to MuJoCo joint type',
            'limited_joint_types':np.asarray(m.jnt_type)[joints].tolist()}
    def take(self):
        require(self.armed and self.count>0,'Physics logger missed native substeps');self.armed=False
        return {**self.last,'physics_substeps':self.count,'tracking_error_peak_m':self.position_peak,
            'orientation_error_peak_rad':self.orientation_peak,'torque_clip_peak_nm':self.torque_clip_peak,
            'actuator_bound_samples':self.actuator_bound_hits,'joint_min_margin_by_joint':self.joint_margin.tolist(),
            'contacts':list(self.contacts.values()),'contact_pair_count':len(self.contacts),
            'peak_robot_contact_force_n':max((r['peak_force_norm_n'] for r in self.contacts.values() if r['involves_robot']),default=0.),
            'peak_contact_force_n':max((r['peak_force_norm_n'] for r in self.contacts.values()),default=0.)}
    def __exit__(self,*args):
        self.armed=False
        if self.had_override:self.sim.step=self.previous_override
        else:del self.sim.step
