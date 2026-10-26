# 发布注释：配对干预：真实目标沿机器人基座X移动；相机沿腕部局部X移动。位移单位为米。
# 运行来源与发布哈希分别记录在SOURCE_MANIFEST.json；历史运行目录不改动。
"""Native-time rendering paired with a real free-joint target translation.

Initial RGB uses each camera's own native sensor snapshot. The target's sampled
geometry is translated in the realized-model renderer, and its actual free joint
is translated by exactly the same world vector before executing actions. This
preserves native sensor timing without force-refreshing observable clocks.
"""
import copy
from functools import wraps
import numpy as np
from diag_common import require,config,native_runtime,native_helpers
from wr_camera import CameraSamples,ShadowRenderer,restore_sample_visuals,local_shift,nrmse
from wr_data import geometry_module


def crop_count(seg,ids):
    n=int(np.sqrt(.9)*256);k=(256-n)//2
    return int(((seg[:,:,0]==5)&np.isin(seg[:,:,1],ids))[k:k+n,k:k+n].sum())

class Samples(CameraSamples):
    def __init__(self,env,anchor):
        super().__init__(env,config()['rgb_tolerance'],anchor['render_ids'])
        self.anchor=anchor;self.front=None
    def arm(self,key):
        super().arm(key);self.front=None
    def __enter__(self):
        super().__enter__()
        self.front_obs=self.env.env._observables['agentview_image'];self.front_original=self.front_obs._sensor
        self.wrist_wrapped=self.observable._sensor
        @wraps(self.front_original)
        def front(cache):
            pixels=self.front_original(cache)
            if self.key is not None:
                self.front=dict(pixels=np.array(pixels,copy=True),snapshot=copy.copy(self.env.sim.data._data),
                    visuals=dict(site_rgba=np.array(self.env.sim.model.site_rgba,copy=True)),
                    sample_time_s=float(self.env.sim.data.time))
            return pixels
        @wraps(self.wrist_wrapped)
        def wrist(cache):
            pixels=self.wrist_wrapped(cache)
            if self.key is not None:self.latest['truth']=geometry_module().truth(self.env,self.anchor)
            return pixels
        self.front_obs._sensor=front;self.observable._sensor=wrist
        return self
    def both(self,raw):
        w=self.take(raw)
        require(self.front is not None and np.array_equal(raw['agentview_image'],self.front['pixels']),
                'Front camera snapshot does not match native returned pixels')
        return self.front,w
    def __exit__(self,*args):
        self.front_obs._sensor=self.front_original
        return super().__exit__(*args)

class Renderer(ShadowRenderer):
    # 从原生传感器采样快照渲染副本；初始RGB中的目标位移与真实free-joint位移一致。不能在此重算所有机器人姿态，否则零干预画面会改变。
    def image(self,sample,name,anchor,delta=None,camera_shift=0.,segmentation=False):
        before=native_helpers().fingerprint(self.env)
        try:
            restore_sample_visuals(self.sim.model,sample['visuals'])
            self.mujoco.mj_copyData(self.sim.data._data,self.sim.model._model,sample['snapshot'])
            # Do not mj_forward here: native sampling uses existing derived poses.
            # Recomputing every robot pose would change the zero RGB reference.
            if delta is not None and np.any(delta):
                for field,ids in [('geom_xpos',anchor['all_geom_ids']),('body_xpos',anchor['body_ids']),
                                  ('xipos',anchor['body_ids']),('site_xpos',anchor['site_ids'])]:
                    getattr(self.sim.data,field)[ids]+=np.asarray(delta)
            camera=self.sim.model.camera_name2id(name)
            if camera_shift:
                require(name=='robot0_eye_in_hand','Camera intervention must be wrist-only')
                self.sim.data.cam_xpos[camera]=local_shift(self.sim.data.cam_xpos[camera],
                    self.sim.data.cam_xmat[camera],'x',camera_shift)
            self.context.gl_ctx.make_current()
            a=self.sim.render(camera_name=name,width=256,height=256,segmentation=segmentation)
            from robosuite import macros
            return np.asarray(a)[::(1 if macros.IMAGE_CONVENTION=='opengl' else -1)].copy()
        finally:
            native_runtime().bind_environment_context(self.env)
            require(native_helpers().fingerprint(self.env)==before,'Renderer changed live environment')
    def zero_gate(self,raw,samples,anchor):
        f,w=samples.both(raw);errors={}
        for name,sample in [('agentview',f),('robot0_eye_in_hand',w)]:
            errors[name]=nrmse(self.image(sample,name,anchor),sample['pixels'])
            require(errors[name]<=config()['rgb_tolerance'],'Zero snapshot RGB differs: '+name)
        return errors
    # 前视/腕部分别使用各自原生采样快照，保持传感器时钟；相机干预只作用腕部。
    def paired_raw(self,raw,samples,anchor,delta,camera_shift):
        f,w=samples.both(raw)
        changed=dict(raw)
        changed['agentview_image']=self.image(f,'agentview',anchor,delta) if np.any(delta) else f['pixels'].copy()
        changed['robot0_eye_in_hand_image']=(self.image(w,'robot0_eye_in_hand',anchor,delta,camera_shift)
            if np.any(delta) or camera_shift else w['pixels'].copy())
        seg=self.image(w,'robot0_eye_in_hand',anchor,delta,camera_shift,True)
        return changed,dict(visible_pixels=crop_count(seg,anchor['render_ids']),
                           front_sample_time_s=f['sample_time_s'],wrist_sample_time_s=w['sample_time_s'],
                           clocks_unchanged=True,actual_freejoint_translation_required=True)


def anchor_for(env,suite,task):
    a=geometry_module().anchor(env,suite,task);m=env.sim.model
    require(a['kind']=='object','Protocol only supports free objects')
    obj=env.env.get_object(a['entity']);joints=list(obj.joints)
    free=[m.joint_name2id(j) for j in joints if int(m.jnt_type[m.joint_name2id(j)])==0]
    require(len(free)==1,'Target must have exactly one free joint')
    descendants={a['body']}
    for b in range(m.nbody):
        if int(m.body_parentid[b]) in descendants:descendants.add(b)
    a.update(joint=free[0],qpos_address=int(m.jnt_qposadr[free[0]]),body_ids=sorted(descendants),
             all_geom_ids=[g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in descendants],
             site_ids=[g for g in range(m.nsite) if int(m.site_bodyid[g]) in descendants])
    return a

# 将机器人基座X轴映射到世界坐标；目标位移不是固定世界X，也不是相机局部X。
def vector(env,distance):
    robot=env.env.robots[0].robot_model.root_body
    b=env.sim.model.body_name2id(robot)
    axis=np.asarray(env.sim.data.body_xmat[b]).reshape(3,3)[:,0]
    require(abs(axis[2])<1e-8,'Robot base X is not horizontal')
    return axis*distance

# 只写目标free-joint的XYZ qpos；forward刷新派生几何，不推进时间。机器人qpos、所有qvel与时钟必须保持原值。
def set_translation(env,anchor,delta):
    d=env.sim.data;i=anchor['qpos_address'];q=np.array(d.qpos,copy=True)
    velocity=np.array(d.qvel,copy=True);now=float(d.time)
    d.qpos[i:i+3]+=np.asarray(delta);env.sim.forward()
    expected=q.copy();expected[i:i+3]+=delta
    require(np.array_equal(d.qpos,expected) and np.array_equal(d.qvel,velocity) and float(d.time)==now,
            'Physical translation changed other joint state, velocity or time')
    return dict(delta_world_m=np.asarray(delta).tolist(),qpos_address=i,other_joint_state_exact=True,
                velocity_exact=True,clock_exact=True)

def contacts(env,anchor):
    target=set(anchor['all_geom_ids']);hand=set(anchor['hand_ids']);rows=[]
    center=geometry_module().truth(env,anchor)['target_xyz']
    for i in range(env.sim.data.ncon):
        c=env.sim.data.contact[i];a,b=int(c.geom1),int(c.geom2)
        if (a in target)==(b in target):continue
        other=b if a in target else a
        rows.append(dict(pair=tuple(sorted((a,b))),other=other,dist=float(c.dist),
            support=bool(other not in hand and abs(float(c.frame[2]))>=config()['support_normal_z_min']
                         and float(c.pos[2])<center[2]),hand=other in hand))
    return rows

# 比较原支撑与新增穿插，返回物理变体异常。无效变体保留并对所有模型使用同一配对集合。
def validity(baseline,current):
    old={tuple(x['pair']):x for x in baseline};supports={x['other'] for x in baseline if x['support']}
    reasons=[]
    if not supports:reasons.append('baseline_has_no_identified_support')
    if not any(x['support'] and x['other'] in supports and x['dist']<=config()['support_gap_m'] for x in current):
        reasons.append('lost_original_support')
    for x in current:
        ref=old.get(tuple(x['pair']))
        allowed=min(0.,ref['dist']) if ref is not None else 0.
        if x['dist']<allowed-config()['new_penetration_m']:
            reasons.append('new_or_increased_penetration');break
    return reasons
