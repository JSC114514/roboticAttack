"""GPU-only shadow rendering. Native sensors ALWAYS return their native pixels."""
import copy
from functools import wraps
import numpy as np
from va_common import require,config,adapters
from va_math import rotation_angle,random_pose

class ShadowRenderer:
    def __init__(self,env):
        import mujoco
        from robosuite.utils.binding_utils import MjSim,MjRenderContextOffscreen
        self.mujoco=mujoco;self.env=env
        self.sim=MjSim(copy.copy(env.sim.model._model))
        self.context=MjRenderContextOffscreen(self.sim,device_id=int(__import__('os').environ['MUJOCO_EGL_DEVICE_ID']))
        self.context.vopt=copy.copy(env.sim._render_context_offscreen.vopt)
        self.context.scn.flags[:]=env.sim._render_context_offscreen.scn.flags
        adapters()[1].runtime_module().bind_environment_context(env)
    def render(self,position=None,rotation=None,segmentation=False,state=None):
        native=adapters()[1];rt=native.runtime_module()
        from audit_replay import fingerprint
        before=fingerprint(self.env)
        try:
            self.mujoco.mj_copyData(self.sim.data._data,self.sim.model._model,self.env.sim.data._data if state is None else state)
            camera=self.sim.model.camera_name2id('robot0_eye_in_hand')
            if position is not None:
                self.sim.data.cam_xpos[camera]=position
                self.sim.data.cam_xmat[camera]=np.asarray(rotation).reshape(9)
            self.context.gl_ctx.make_current()
            return self.sim.render(camera_name='robot0_eye_in_hand',width=256,height=256,segmentation=segmentation)
        finally:
            rt.bind_environment_context(self.env)
            require(fingerprint(self.env)==before,'Shadow render changed original physics/camera/clock')
    def close(self):
        self.context=None;self.sim.free();self.sim=None
        __import__('gc').collect()
        adapters()[1].runtime_module().bind_environment_context(self.env)

class Sensors:
    """Capture camera pose and label in its original callback; side-channel RGB only."""
    def __init__(self,env,target):
        self.env=env;self.target=target;self.latest={};self.serial=0;self.mode='native';self.path={};self.shadow=None
        self.seed=0;self.zero_checks=0;self.zero_max=0.;self.random_amplitudes=None
    def __enter__(self):
        self.originals={}
        for view,key,camera in (('front','agentview_image','agentview'),('wrist','robot0_eye_in_hand_image','robot0_eye_in_hand')):
            observable=self.env.env._observables[key];original=observable._sensor;self.originals[key]=(observable,original)
            @wraps(original)
            def sensor(cache,view=view,original=original,camera=camera):
                from audit_replay import fingerprint
                from sr_geometry import truth
                from robosuite import macros
                pixels=original(cache);before=fingerprint(self.env)
                camera_id=self.env.sim.model.camera_name2id(camera)
                p=np.array(self.env.sim.data.cam_xpos[camera_id],copy=True)
                r=np.array(self.env.sim.data.cam_xmat[camera_id],copy=True).reshape(3,3)
                result=dict(serial=self.serial,pixels=np.array(pixels,copy=True),position=p,rotation=r,
                            fingerprint=before,sample_time_s=float(self.env.sim.data.time))
                if view=='wrist':
                    result['snapshot']=copy.copy(self.env.sim.data._data)
                    changed_p,changed_r=p,r
                    if self.mode=='replay':
                        require(self.serial in self.path,'Clean camera path has no matching sensor sample')
                        changed_p=np.asarray(self.path[self.serial]['position']);changed_r=np.asarray(self.path[self.serial]['rotation'])
                    elif self.mode=='random':
                        require(self.serial in self.random_amplitudes,'Matched random amplitude missing')
                        amplitude=self.random_amplitudes[self.serial]
                        changed_p,changed_r=random_pose(p,r,amplitude['translation_m'],amplitude['rotation_rad'],self.seed)
                    if self.mode!='native':
                        if self.shadow is None:self.shadow=ShadowRenderer(self.env)
                        sign=1 if macros.IMAGE_CONVENTION=='opengl' else -1
                        zero=np.asarray(self.shadow.render())[::sign]
                        error=float(np.sqrt(np.mean(((zero.astype(float)-pixels.astype(float))/255.)**2)))
                        require(error<=config()['rgb_tolerance'],'Shadow native-pose RGB mismatch')
                        self.zero_max=max(self.zero_max,error);self.zero_checks+=1
                        result['alternate_pixels']=np.asarray(self.shadow.render(changed_p,changed_r))[::sign].copy()
                        seg=np.asarray(self.shadow.render(changed_p,changed_r,True))[::sign][::-1,::-1]
                    else:
                        sign=1 if macros.IMAGE_CONVENTION=='opengl' else -1
                        seg=np.asarray(self.env.sim.render(camera_name=camera,width=256,height=256,segmentation=True))[::sign][::-1,::-1]
                    result.update(truth=truth(self.env,self.target),mask=(seg[:,:,0]==5)&np.isin(seg[:,:,1],self.target['render_ids']),
                                  translation_m=float(np.linalg.norm(changed_p-p)),rotation_rad=rotation_angle(r,changed_r),
                                  rendered_position=np.asarray(changed_p).tolist(),rendered_rotation=np.asarray(changed_r).tolist())
                else:
                    sign=1 if macros.IMAGE_CONVENTION=='opengl' else -1
                    seg=np.asarray(self.env.sim.render(camera_name=camera,width=256,height=256,segmentation=True))[::sign][::-1,::-1]
                    result['mask']=(seg[:,:,0]==5)&np.isin(seg[:,:,1],self.target['render_ids'])
                require(fingerprint(self.env)==before,'Sensor diagnostics changed native state')
                self.latest[view]=result
                return pixels
            observable._sensor=sensor
        return self
    def arm(self,serial):self.serial=serial;self.latest={}
    def activate(self,mode,path,amplitudes,seed):
        self.mode=mode;self.path=path;self.random_amplitudes=amplitudes;self.seed=seed;self.serial=0
        # The starting image was sampled BEFORE the control step finished. Render
        # its saved MjData, never the end-of-step live state.
        if mode=='native':return
        from robosuite import macros
        w=self.latest['wrist'];p=w['position'];r=w['rotation']
        if mode=='replay':cp,cr=path[0]['position'],path[0]['rotation']
        else:
            a=amplitudes[0];cp,cr=random_pose(p,r,a['translation_m'],a['rotation_rad'],seed)
        if self.shadow is None:self.shadow=ShadowRenderer(self.env)
        sign=1 if macros.IMAGE_CONVENTION=='opengl' else -1
        zero=np.asarray(self.shadow.render(state=w['snapshot']))[::sign]
        error=float(np.sqrt(np.mean(((zero.astype(float)-w['pixels'].astype(float))/255.)**2)))
        require(error<=config()['rgb_tolerance'],'Starting shadow sample RGB differs')
        self.zero_max=max(error,self.zero_max);self.zero_checks+=1
        w['alternate_pixels']=np.asarray(self.shadow.render(cp,cr,state=w['snapshot']))[::sign].copy()
        seg=np.asarray(self.shadow.render(cp,cr,True,state=w['snapshot']))[::sign][::-1,::-1]
        w.update(mask=(seg[:,:,0]==5)&np.isin(seg[:,:,1],self.target['render_ids']),
                 translation_m=float(np.linalg.norm(np.asarray(cp)-p)),rotation_rad=rotation_angle(r,cr),
                 rendered_position=np.asarray(cp).tolist(),rendered_rotation=np.asarray(cr).tolist())
    def take(self,raw):
        for view,key in (('front','agentview_image'),('wrist','robot0_eye_in_hand_image')):
            require(view in self.latest and self.latest[view]['serial']==self.serial,'Native camera did not sample; no forced refresh')
            require(np.array_equal(raw[key],self.latest[view]['pixels']),'Native cached frame differs from sampled frame')
        return self.latest
    def observation(self,raw):
        altered=dict(raw)
        if 'alternate_pixels' in self.latest.get('wrist',{}):altered['robot0_eye_in_hand_image']=self.latest['wrist']['alternate_pixels']
        return adapters()[1].runtime_module().observation(altered,adapters()[1].runtime_config())
    def __exit__(self,*_):
        for observable,original in self.originals.values():observable._sensor=original
        if self.shadow is not None:self.shadow.close()
