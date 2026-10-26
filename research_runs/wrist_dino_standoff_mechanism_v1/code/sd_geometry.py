"""Fixed interaction anchors; rendering and collision queries never move live physics."""
from __future__ import annotations
from functools import wraps
import hashlib
import itertools
import numpy as np
from sd_common import config,require
from sd_math import aabb_center,radial_delta

def fingerprint(env):
    arrays={'sim_state':np.asarray(env.get_sim_state())}
    for k in ('body_xpos','body_xquat','geom_xpos','geom_xmat','site_xpos','site_xmat','cam_xpos','cam_xmat'):
        arrays[k]=np.asarray(getattr(env.sim.data,k))
    for k in ('body_pos','body_quat'):arrays['model_'+k]=np.asarray(getattr(env.sim.model,k))
    hashes={k:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k,v in arrays.items()}
    clocks={n:{k:v.item() if isinstance(v,np.generic) else v for k,v in vars(o).items()
        if k!='_sensor' and isinstance(v,(bool,int,float,str,np.generic))} for n,o in env.env._observables.items()}
    return {'arrays':hashes,'time':float(env.sim.data.time),'observable_clocks':clocks}

def primitive_vertices(model,g):
    kind=int(model.geom_type[g]);size=np.asarray(model.geom_size[g],dtype=float)
    if kind==7:
        mesh=int(model.geom_dataid[g]);start=int(model.mesh_vertadr[mesh]);count=int(model.mesh_vertnum[mesh])
        return np.asarray(model.mesh_vert[start:start+count],dtype=float)
    if kind==6:extent=size
    elif kind==2:extent=np.repeat(size[0],3)
    elif kind==3:extent=np.array([size[0],size[0],size[0]+size[1]])
    elif kind==4:extent=size
    elif kind==5:extent=np.array([size[0],size[0],size[1]])
    else:raise RuntimeError('Unsupported target collision geom type: '+str(kind))
    return np.asarray(list(itertools.product((-1,1),repeat=3)))*extent

def fixed_anchor(env,spec):
    model,data=env.sim.model,env.sim.data;obj=env.env.get_object(spec['entity'])
    if spec['kind']=='object':
        body=int(model.body_name2id(obj.root_body))
        contact=[int(model.geom_name2id(n)) for n in obj.contact_geoms]
        render=sorted({int(model.geom_name2id(n)) for n in list(obj.visual_geoms)+list(obj.contact_geoms)})
    else:
        require(spec['kind']=='middle_handle','Unknown anchor kind')
        bodies=[i for i in range(model.nbody) if (model.body_id2name(i) or '').startswith(spec['entity']+'_')
                and (model.body_id2name(i) or '').endswith('cabinet_middle')]
        require(len(bodies)==1,'Missing/ambiguous middle drawer body');body=bodies[0];contact=[]
        for center in spec['handle_collision_centers']:
            ids=[i for i in range(model.ngeom) if int(model.geom_bodyid[i])==body and int(model.geom_type[i])==6
                and np.allclose(model.geom_pos[i],center,rtol=0,atol=1e-8)]
            require(len(ids)==1,'Handle collision annotation does not match asset');contact.extend(ids)
        visual=[i for i in range(model.ngeom) if int(model.geom_bodyid[i])==body and int(model.geom_type[i])==7
            and (model.mesh_id2name(int(model.geom_dataid[i])) or '').endswith('wooden_cabinet_middle_handle_vis')]
        require(len(visual)==1,'Missing/ambiguous handle visual mesh');render=sorted(set(contact+visual))
    require(contact and render,'Empty fixed target geometry')
    rb=np.asarray(data.body_xmat[body]).reshape(3,3);pb=np.asarray(data.body_xpos[body]);vertices=[]
    for g in contact:
        rg=np.asarray(data.geom_xmat[g]).reshape(3,3);pg=np.asarray(data.geom_xpos[g])
        vertices.append((primitive_vertices(model,g)@rg.T+pg-pb)@rb)
    local=aabb_center(np.concatenate(vertices))
    grip=env.env.robots[0].gripper
    if isinstance(grip,dict):grip=next(iter(grip.values()))
    robot_names=set()
    for robot in env.env.robots:
        robot_names.update(robot.robot_model.contact_geoms)
        grippers=robot.gripper.values() if isinstance(robot.gripper,dict) else [robot.gripper]
        for gripper in grippers:robot_names.update(gripper.contact_geoms)
    robot_ids={int(model.geom_name2id(n)) for n in robot_names}
    return {'entity':spec['entity'],'kind':spec['kind'],'body_id':body,'local_center':local.tolist(),
        'contact_ids':contact,'render_ids':render,'robot_ids':sorted(robot_ids),
        'contact_names':[model.geom_id2name(g) for g in contact],
        'render_names':[model.geom_id2name(g) for g in render],'gripper':grip}

def anchor_public(a):return {k:v for k,v in a.items() if k!='gripper'}

def truth(env,anchor):
    data,model=env.sim.data,env.sim.model
    r=np.asarray(data.body_xmat[anchor['body_id']]).reshape(3,3)
    target=np.asarray(data.body_xpos[anchor['body_id']])+r@np.asarray(anchor['local_center'])
    eef=np.asarray(data.site_xpos[env.env.robots[0].eef_site_id]);objects=set(anchor['contact_ids']);robot=set(anchor['robot_ids'])
    contact=any((int(data.contact[i].geom1) in objects and int(data.contact[i].geom2) in robot) or
        (int(data.contact[i].geom2) in objects and int(data.contact[i].geom1) in robot) for i in range(int(data.ncon)))
    grasp=bool(env.env._check_grasp(gripper=anchor['gripper'],object_geoms=anchor['contact_names']))
    cid=int(model.camera_name2id('robot0_eye_in_hand'));cam=np.asarray(data.cam_xpos[cid])
    camera_relative=np.asarray(data.cam_xmat[cid]).reshape(3,3).T@(target-cam)
    return {'target_xyz':target.tolist(),'eef_xyz':eef.tolist(),'distance_m':float(np.linalg.norm(target-eef)),
        'camera_target_distance_m':float(np.linalg.norm(target-cam)),'camera_optical_depth_m':float(-camera_relative[2]),
        'contact':bool(contact),'grasp':grasp,'physics_time_s':float(data.time)}

class SceneTranslation:
    def __init__(self,scene,ids,delta):self.scene,self.ids,self.delta,self.calls=scene,set(ids),np.asarray(delta),0
    def __enter__(self):
        import mujoco
        self.module=mujoco;self.original=mujoco.mjr_render
        def draw(*args,**kwargs):
            scene=kwargs.get('scn') if 'scn' in kwargs else (args[1] if len(args)>1 else None)
            if scene is not self.scene:return self.original(*args,**kwargs)
            changed=[]
            try:
                for i in range(scene.ngeom):
                    g=scene.geoms[i]
                    if int(g.objtype)==5 and int(g.objid) in self.ids:
                        changed.append((g,np.array(g.pos,copy=True)));g.pos[:]+=self.delta
                require(changed,'Target not present in owned render scene');self.calls+=1
                return self.original(*args,**kwargs)
            finally:
                for g,pos in changed:g.pos[:]=pos
        mujoco.mjr_render=draw;return self
    def __exit__(self,*_):self.module.mjr_render=self.original

def render(env,anchor,delta,view='robot0_eye_in_hand'):
    from robosuite import macros
    scene=env.sim._render_context_offscreen.scn
    with SceneTranslation(scene,anchor['render_ids'],delta) as hook:
        rgb=env.sim.render(camera_name=view,width=256,height=256)
        seg=env.sim.render(camera_name=view,width=256,height=256,segmentation=True)
    require(hook.calls==2,'Unexpected renderer call count')
    convention=1 if macros.IMAGE_CONVENTION=='opengl' else -1
    rgb=np.asarray(rgb)[::convention][::-1,::-1].copy();seg=np.asarray(seg)[::convention][::-1,::-1].copy()
    mask=(seg[:,:,0]==5)&np.isin(seg[:,:,1],anchor['render_ids'])
    return rgb,mask

def cropped_pixels(mask):
    # Official policy center crop retains 90% area. This checks the same
    # central field before resize; no mask is ever input to the policy.
    n=int(np.sqrt(.9)*mask.shape[0]);start=(mask.shape[0]-n)//2
    return int(mask[start:start+n,start:start+n].sum())

class CollisionCheck:
    def __init__(self,env,anchor):
        import mujoco
        self.mj=mujoco;self.env=env;self.anchor=anchor
        self.model=getattr(env.sim.model,'_model',env.sim.model)
        self.scratch=mujoco.MjData(self.model)
        require(hasattr(mujoco,'mj_geomDistance'),'MuJoCo signed geometry distance API is required')
    def check(self,delta):
        m,d=self.env.sim.model,self.env.sim.data;s=self.scratch;delta=np.asarray(delta)
        s.geom_xpos[:]=d.geom_xpos;s.geom_xmat[:]=d.geom_xmat
        selected=set(self.anchor['render_ids']);rows=[];tol=config()['penetration_tolerance_m']
        other=[i for i in range(m.ngeom) if i not in selected and (int(m.geom_contype[i]) or int(m.geom_conaffinity[i]))]
        shifted=np.asarray(d.geom_xpos)+delta
        for a in self.anchor['contact_ids']:
            candidates=[b for b in other if int(m.geom_type[b])==0 or
                np.linalg.norm(shifted[a]-d.geom_xpos[b])<=float(m.geom_rbound[a]+m.geom_rbound[b])+.001]
            for b in candidates:
                s.geom_xpos[a]=d.geom_xpos[a]
                baseline=float(self.mj.mj_geomDistance(self.model,s,a,b,.2,None))
                s.geom_xpos[a]=shifted[a]
                changed=float(self.mj.mj_geomDistance(self.model,s,a,b,.2,None))
                require(np.isfinite([baseline,changed]).all(),'Nonfinite virtual collision query')
                if changed < -tol and changed < baseline-tol:
                    rows.append({'target_geom':a,'other_geom':b,'baseline_signed_m':baseline,'shifted_signed_m':changed})
            s.geom_xpos[a]=d.geom_xpos[a]
        return {'new_or_deeper_penetration':bool(rows),'pairs':rows,'source':'signed_geom_distance_on_separate_MjData_not_live_physics'}

def project(env,point,view):
    m,d=env.sim.model,env.sim.data;cid=int(m.camera_name2id(view))
    p=np.asarray(d.cam_xmat[cid]).reshape(3,3).T@(np.asarray(point)-d.cam_xpos[cid]);z=-p[2]
    if z<=0:return None
    f=128/np.tan(np.deg2rad(m.cam_fovy[cid])/2)
    # The native image is rotated 180 degrees for OFT's upright convention.
    return [float(255-(128+f*p[0]/z)),float(255-(128-f*p[1]/z))]

class Camera:
    def __init__(self,env,spec,full=False):
        self.env,self.spec,self.full=env,spec,full;self.key=None;self.latest=None
        self.signs=('near','far')
        self.anchor=fixed_anchor(env,spec);self.collision=CollisionCheck(env,self.anchor) if full else None
    def arm(self,key,full=None):
        self.key=key;self.latest=None;self.requested_full=self.full if full is None else full
    def __enter__(self):
        self.observable=self.env.env._observables['robot0_eye_in_hand_image'];self.original=self.observable._sensor
        @wraps(self.original)
        def sensor(cache):
            pixels=self.original(cache)
            if self.key is not None:
                before=fingerprint(self.env);t=truth(self.env,self.anchor)
                zero,mask=render(self.env,self.anchor,np.zeros(3));native=np.asarray(pixels)[::-1,::-1]
                error=float(np.sqrt(np.mean(((native.astype(float)-zero.astype(float))/255.)**2)))
                require(error<=config()['rgb_nrmse_limit'],'Zero displacement RGB does not match native camera timestamp')
                images={'zero':zero};conditions={}
                if self.requested_full:
                    for name,sign in (('near',-1),('far',1)):
                        if name not in self.signs:continue
                        reasons=[]
                        try:delta=radial_delta(t['target_xyz'],t['eef_xyz'],config()['distance_m'],sign)
                        except ValueError:delta=np.zeros(3);reasons.append('undefined_relation')
                        image,shift_mask=render(self.env,self.anchor,delta);collision=self.collision.check(delta)
                        if not cropped_pixels(mask):reasons.append('original_target_not_policy_visible')
                        if not cropped_pixels(shift_mask):reasons.append('shifted_target_not_policy_visible')
                        if collision['new_or_deeper_penetration']:reasons.append('virtual_geometry_penetration')
                        images[name]=image;conditions[name]={'valid':not reasons,'reasons':reasons,'delta_m':delta.tolist(),
                            'actual_displacement_m':float(np.linalg.norm(delta)),'visible_pixels':cropped_pixels(shift_mask),
                            'collision':collision}
                self.latest={'key':self.key,'native_pixels':np.array(pixels,copy=True),'images':images,
                    'truth':t,'visible':bool(cropped_pixels(mask)),'visible_pixels':cropped_pixels(mask),
                    'native_visible_pixels':int(mask.sum()),'conditions':conditions,'sample_fingerprint':before,
                    'zero_rgb_nrmse':error,'anchor':anchor_public(self.anchor),
                    'projection':{v:project(self.env,t['target_xyz'],v) for v in ('agentview','robot0_eye_in_hand')}}
                require(fingerprint(self.env)==before,'Rendering/virtual collision check changed live physics/cameras/clocks')
            return pixels
        self.observable._sensor=sensor;return self
    def take(self,raw,key):
        require(self.latest is not None and self.latest['key']==key,'Missing sensor-time capture')
        require(np.array_equal(self.latest['native_pixels'],raw['robot0_eye_in_hand_image']),'Returned RGB mismatches captured sensor sample')
        return self.latest
    def __exit__(self,*_):self.observable._sensor=self.original
