"""Simulator truth aligned to replayed RGB. Masks are labels, never probe inputs."""
from __future__ import annotations
import hashlib
import re
import numpy as np
from atlas_common import config,require
from atlas_math import crop_map,patch_majority,patch_reduce,rgb_error,rgb_to_lab

VIEWS={'front':'agentview','wrist':'robot0_eye_in_hand'}


def array_hash(a):
    x=np.ascontiguousarray(a)
    return hashlib.sha256(x.tobytes()).hexdigest()


def pose_bundle(env):
    m,d=env.sim.model,env.sim.data
    out={k:array_hash(getattr(m,k)) for k in ('body_pos','body_quat')}
    for k in ('qpos','qvel','body_xpos','body_xquat','cam_xpos','cam_xmat'):
        out[k]=array_hash(getattr(d,k))
    return out


def _geoms(sim,obj):
    return {int(sim.model.geom_name2id(n)) for n in set(list(obj.contact_geoms)+list(obj.visual_geoms))}


def entity_map(env):
    inner=env.env; names=config()['class_names']; result=[]
    catalog={**getattr(inner,'fixtures_dict',{}),**inner.objects_dict}
    for name,obj in sorted(catalog.items()):
        klass=re.sub(r'_\d+$','',name)
        klass=klass if klass in names else 'other_object'
        result.append({'name':name,'class_id':names.index(klass),'geom_ids':sorted(_geoms(env.sim,obj)),
                       'body_id':int(env.sim.model.body_name2id(obj.root_body))})
    # LIBERO deliberately omits the workspace table from fixtures_dict.
    table_ids=[i for i in range(env.sim.model.ngeom)
               if (env.sim.model.geom_id2name(i) or '').startswith('table_')]
    require(table_ids,'Spatial workspace table geometry not found')
    result.append({'name':'main_table','class_id':names.index('main_table'),'geom_ids':table_ids,
                   'body_id':int(env.sim.model.body_name2id('table')),'source':'arena_workspace'})
    for robot in inner.robots:
        result.append({'name':'robot_arm','class_id':names.index('robot_arm'),
                       'geom_ids':sorted(_geoms(env.sim,robot.robot_model)),'body_id':None})
        gripper=robot.gripper
        for grip in (gripper.values() if isinstance(gripper,dict) else [gripper]):
            result.append({'name':'gripper','class_id':names.index('gripper'),
                           'geom_ids':sorted(_geoms(env.sim,grip)),'body_id':None})
    require(any(r['name']=='akita_black_bowl_1' for r in result),'Missing Spatial goal subject')
    return result


def physical_labels(env,raw,entities,task):
    inner=env.env; sim=env.sim; d=sim.data
    byname={r['name']:r for r in entities}; subject=task['manipulated_object']; goal=task['goal_object']
    obj=inner.objects_dict[subject]; grip=inner.robots[0].gripper
    if isinstance(grip,dict): grip=next(iter(grip.values()))
    op=d.body_xpos[byname[subject]['body_id']].copy()
    gp=d.body_xpos[byname[goal]['body_id']].copy()
    ee=np.asarray(raw['robot0_eef_pos'],dtype=float)
    from robosuite.utils.transform_utils import quat2mat
    er=np.asarray(quat2mat(np.asarray(raw['robot0_eef_quat'])),dtype=float)
    on=d.body_xmat[byname[subject]['body_id']].reshape(3,3)[:,2].copy()
    # A bowl's normal is invariant to its visually ambiguous axial rotation.
    out={}
    for prefix,value in [('eef_position',ee),('eef_normal',er[:,2]),('object_position',op),
                         ('eef_x_axis',er[:,0]),
                         ('object_normal',on),('object_eef_relative',op-ee),
                         ('relative_normal',er.T@on),('goal_object_relative',gp-op)]:
        out.update({prefix+'_'+axis:float(v) for axis,v in zip('xyz',value)})
    object_ids={int(sim.model.geom_name2id(n)) for n in obj.contact_geoms}
    finger_ids={int(sim.model.geom_name2id(n)) for seq in grip.important_geoms.values()
                for n in ([seq] if isinstance(seq,str) else seq)}
    contact=any((int(d.contact[i].geom1) in object_ids and int(d.contact[i].geom2) in finger_ids) or
                (int(d.contact[i].geom2) in object_ids and int(d.contact[i].geom1) in finger_ids)
                for i in range(int(d.ncon)))
    grasp=bool(inner._check_grasp(gripper=grip,object_geoms=obj.contact_geoms))
    require(not grasp or contact,'Grasp without contact')
    out.update(object_eef_distance=float(np.linalg.norm(op-ee)),
               gripper_opening=float(abs(raw['robot0_gripper_qpos'][0]-raw['robot0_gripper_qpos'][1])),
               contact=float(contact),grasp=float(grasp))
    others=[r for r in entities if r['name'].startswith('akita_black_bowl_') and r['name']!=subject]
    nearest=min(others,key=lambda r:np.linalg.norm(d.body_xpos[r['body_id']]-op)) if others else None
    delta=d.body_xpos[nearest['body_id']]-op if nearest else np.full(3,np.nan)
    out.update({f'distractor_object_relative_{a}':float(v) for a,v in zip('xyz',delta)})
    return out


def render_labels(env,raw,entities,task):
    """Direct render only at checkpoints, preserving the original RGB sensors/model."""
    from robosuite import macros
    convention=1 if macros.IMAGE_CONVENTION=='opengl' else -1
    output={}; class_names=config()['class_names']; before=pose_bundle(env)
    for view,camera in VIEWS.items():
        rgb,depth=env.sim.render(camera_name=camera,width=224,height=224,depth=True)
        seg=env.sim.render(camera_name=camera,width=224,height=224,depth=False,segmentation=True)
        rgb=np.asarray(rgb)[::convention][::-1,::-1].copy()
        depth=np.asarray(depth,dtype=np.float64)[::convention][::-1,::-1].copy()
        seg=np.asarray(seg)[::convention][::-1,::-1].copy()
        error=rgb_error(np.asarray(raw[camera+'_image'])[::-1,::-1],rgb)
        require(error['normalized_rms']<=config()['replay']['rgb_normalized_rms_limit'],
                f'Direct RGB/depth/segmentation coordinate mismatch: {view} {error}')
        # MuJoCo segmentation stores (object type, object id). Only geoms are mapped.
        geom=np.where(seg[:,:,0]==5,seg[:,:,1],-1)
        instance=np.zeros((224,224),dtype=np.int16); classes=np.zeros_like(instance)
        for index,e in enumerate(entities,1):
            mask=np.isin(geom,e['geom_ids']); instance[mask]=index; classes[mask]=e['class_id']
        extent=float(env.sim.model.stat.extent)
        near=float(env.sim.model.vis.map.znear)*extent; far=float(env.sim.model.vis.map.zfar)*extent
        meters=near/(1-depth*(1-near/far))
        require(np.isfinite(meters).all() and (meters>0).all(),'Invalid metric depth')
        inst_crop=crop_map(instance,nearest=True).astype(np.int16)
        class_crop=crop_map(classes,nearest=True).astype(np.int16)
        cls,confidence=patch_majority(class_crop,len(class_names))
        depth_crop=crop_map(meters).astype(np.float32)
        subj_index=next(i for i,e in enumerate(entities,1) if e['name']==task['manipulated_object'])
        goal_index=next(i for i,e in enumerate(entities,1) if e['name']==task['goal_object'])
        summaries={}
        for prefix,index in [('subject',subj_index),('destination',goal_index)]:
            mask=inst_crop==index; y,x=np.where(mask)
            summaries[prefix+'_visible']=bool(len(x))
            summaries[prefix+'_area']=float(mask.mean())
            summaries[prefix+'_u']=float(x.mean()/223) if len(x) else None
            summaries[prefix+'_v']=float(y.mean()/223) if len(y) else None
            summaries[prefix+'_width']=float((x.max()-x.min()+1)/224) if len(x) else None
            summaries[prefix+'_height']=float((y.max()-y.min()+1)/224) if len(y) else None
            summaries[prefix+'_depth']=float(np.median(depth_crop[mask])) if len(x) else None
        mask=inst_crop==subj_index
        if mask.any():
            lab=rgb_to_lab(np.clip(crop_map(rgb)/255.,0,1))
            summaries.update({f'subject_lab_{k}':float(lab[:,:,i][mask].mean()) for i,k in enumerate('Lab')})
            gray=crop_map(rgb).astype(np.float32).mean(-1)/255
            dy,dx=np.gradient(gray)
            summaries['subject_texture_gradient']=float(np.hypot(dx,dy)[mask].mean())
        else:
            summaries.update({f'subject_lab_{k}':None for k in 'Lab'},subject_texture_gradient=None)
        cid=int(env.sim.model.camera_name2id(camera))
        camera_pos=env.sim.data.cam_xpos[cid].copy()
        camera_rotation=env.sim.data.cam_xmat[cid].reshape(3,3).copy()
        tangent=np.tan(np.deg2rad(float(env.sim.model.cam_fovy[cid]))/2)*np.sqrt(.9)
        for prefix,name in [('subject',task['manipulated_object']),('destination',task['goal_object'])]:
            entity=next(e for e in entities if e['name']==name)
            p=camera_rotation.T@(env.sim.data.body_xpos[entity['body_id']]-camera_pos)
            inside=bool(-p[2]>0 and abs(p[0])<=-p[2]*tangent and abs(p[1])<=-p[2]*tangent)
            summaries[prefix+'_center_in_frustum']=inside
            summaries[prefix+'_occlusion_proxy']=inside and not summaries[prefix+'_visible']
        output[view]={'rgb':rgb,'instance_raw':instance,'class_raw':classes,'depth_raw':meters.astype(np.float32),
                      'instance_crop':inst_crop,'class_tokens':cls,'class_confidence':confidence.astype(np.float32),
                      'depth_tokens':patch_reduce(depth_crop).astype(np.float32),'summary':summaries,
                      'camera_position':camera_pos,
                      'camera_rotation':camera_rotation,
                      'camera_fovy':float(env.sim.model.cam_fovy[cid]),'direct_rgb_error':error}
    require(before==pose_bundle(env),'Rendering/label extraction changed physics or camera poses')
    return output


def label_specs():
    specs=[]
    for prefix,kind,unit in [('eef_position','position','m'),('eef_normal','normal','unit'),
            ('eef_x_axis','normal','unit'),
            ('object_position','position','m'),('object_normal','normal','unit'),
            ('object_eef_relative','relation','m'),('relative_normal','normal','unit'),
            ('goal_object_relative','relation','m'),('distractor_object_relative','relation','m')]:
        for axis in 'xyz': specs.append({'name':prefix+'_'+axis,'kind':kind,'unit':unit,'group':prefix})
    specs += [{'name':n,'kind':k,'unit':u,'group':n} for n,k,u in
              [('object_eef_distance','relation','m'),('gripper_opening','interaction','m'),
               ('contact','binary','binary'),('grasp','binary','binary')]]
    for view in config()['views']:
        for n in ('subject_visible','destination_visible','subject_center_in_frustum','subject_occlusion_proxy'):
            specs.append({'name':view+'_'+n,'kind':'binary','unit':'binary','group':view+'_'+n})
        for n in ['subject_u','subject_v','subject_width','subject_height','subject_area','subject_depth',
                  'destination_u','destination_v','destination_width','destination_height','destination_area','destination_depth',
                  'subject_lab_L','subject_lab_a','subject_lab_b','subject_texture_gradient']:
            specs.append({'name':view+'_'+n,'kind':'appearance' if 'lab_' in n or 'texture' in n else 'view_geometry',
                          'unit':'m' if n.endswith('depth') else 'native','group':view+'_'+n})
    return specs
