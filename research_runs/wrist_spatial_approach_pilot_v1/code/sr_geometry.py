"""Fixed first-interaction identities, native-sensor geometry, no scene edits."""
from functools import wraps
import itertools
import numpy as np
from sr_common import require,native

def spec(suite,task):
    objects=['alphabet_soup_1','cream_cheese_1','salad_dressing_1','bbq_sauce_1','ketchup_1','tomato_sauce_1','butter_1','milk_1','chocolate_pudding_1','orange_juice_1']
    if suite=='libero_spatial':return {'entity':'akita_black_bowl_1','kind':'object'}
    if suite=='libero_object':return {'entity':objects[task],'kind':'object'}
    if suite=='libero_goal':
        if task in (0,3):return {'entity':'wooden_cabinet_1','kind':'middle_handle' if task==0 else 'top_handle'}
        if task==7:return {'entity':'flat_stove_1','kind':'button'}
        return {'entity':{1:'akita_black_bowl_1',2:'wine_bottle_1',4:'akita_black_bowl_1',5:'plate_1',6:'cream_cheese_1',8:'akita_black_bowl_1',9:'wine_bottle_1'}[task],'kind':'object'}
    if suite=='libero_10':
        if task==2:return {'entity':'flat_stove_1','kind':'button'}
        return {'entity':{0:'alphabet_soup_1',1:'cream_cheese_1',3:'akita_black_bowl_1',4:'porcelain_mug_1',5:'black_book_1',6:'porcelain_mug_1',7:'alphabet_soup_1',8:'moka_pot_1',9:'white_yellow_mug_1'}[task],'kind':'object'}
    raise ValueError('Unknown suite')

def vertices(m,g):
    k=int(m.geom_type[g]);s=np.asarray(m.geom_size[g])
    if k==7:
        j=int(m.geom_dataid[g]);a=int(m.mesh_vertadr[j]);n=int(m.mesh_vertnum[j]);return np.asarray(m.mesh_vert[a:a+n])
    if k==6:e=s
    elif k==2:e=np.repeat(s[0],3)
    elif k==3:e=np.array([s[0],s[0],s[0]+s[1]])
    elif k==5:e=np.array([s[0],s[0],s[1]])
    else:raise RuntimeError('Unsupported collision primitive '+str(k))
    return np.asarray(list(itertools.product((-1,1),repeat=3)))*e

def anchor(env,suite,task):
    sp=spec(suite,task);m=env.sim.model;d=env.sim.data;obj=env.env.get_object(sp['entity'])
    ids=[int(m.geom_name2id(n)) for n in obj.contact_geoms]
    render=[int(m.geom_name2id(n)) for n in obj.visual_geoms]+ids
    body=int(m.body_name2id(obj.root_body))
    if sp['kind']!='object':
        suffix={'middle_handle':'cabinet_middle','top_handle':'cabinet_top','button':'button'}[sp['kind']]
        bodies=[i for i in range(m.nbody) if (m.body_id2name(i) or '').startswith(sp['entity']+'_') and (m.body_id2name(i) or '').endswith(suffix)]
        require(len(bodies)==1,'Missing/ambiguous functional body: '+str(sp));body=bodies[0]
        ids=[g for g in ids if int(m.geom_bodyid[g])==body]
        render=[g for g in render if int(m.geom_bodyid[g])==body]
        if 'handle' in sp['kind']:
            z=.11041 if sp['kind']=='middle_handle' else .18393
            centers=[[-.00278,-.10161,z],[.02916,-.08348,z],[-.03560,-.08348,z]]
            chosen=[]
            for center in centers:
                matches=[g for g in ids if int(m.geom_type[g])==6 and np.allclose(m.geom_pos[g],center,rtol=0,atol=1e-8)]
                require(len(matches)==1,'Handle collision annotation mismatches asset');chosen+=matches
            ids=chosen
            render=ids+[g for g in render if int(m.geom_type[g])==7 and (m.mesh_id2name(int(m.geom_dataid[g])) or '').endswith('handle_vis')]
    require(ids and render,'Empty fixed target geometry: '+str(sp))
    rb=np.asarray(d.body_xmat[body]).reshape(3,3);pb=np.asarray(d.body_xpos[body]);vv=[]
    for g in ids:vv.append((vertices(m,g)@np.asarray(d.geom_xmat[g]).reshape(3,3).T+d.geom_xpos[g]-pb)@rb)
    vv=np.concatenate(vv);center=(vv.min(0)+vv.max(0))*.5
    grip=env.env.robots[0].gripper
    if isinstance(grip,dict):grip=next(iter(grip.values()))
    hand={int(m.geom_name2id(n)) for n in grip.contact_geoms}
    return {**sp,'body':body,'contact_ids':ids,'render_ids':list(set(render)),'local_center':center.tolist(),
        'hand_ids':list(hand),'gripper':grip,'contact_names':[m.geom_id2name(g) for g in ids]}

def truth(env,a):
    d=env.sim.data;r=np.asarray(d.body_xmat[a['body']]).reshape(3,3)
    p=np.asarray(d.body_xpos[a['body']])+r@np.asarray(a['local_center']);e=np.asarray(d.site_xpos[env.env.robots[0].eef_site_id])
    target=set(a['contact_ids']);hand=set(a['hand_ids']);contact=False
    for i in range(d.ncon):
        x,y=int(d.contact[i].geom1),int(d.contact[i].geom2)
        contact|=(x in target and y in hand) or (y in target and x in hand)
    grasp=bool(env.env._check_grasp(gripper=a['gripper'],object_geoms=a['contact_names']))
    return {'target_xyz':p.tolist(),'eef_xyz':e.tolist(),'distance_m':float(np.linalg.norm(p-e)),
        'contact':bool(contact),'grasp':grasp,'entity':a['entity'],'kind':a['kind'],'sample_time_s':float(d.time)}

def controller(env):
    c=env.env.robots[0].controller
    require(c.use_delta and c.control_dim==6,'Expected native delta OSC pose controller')
    require(int(env.sim.model.site_name2id(c.eef_name))==int(env.env.robots[0].eef_site_id),
        'Geometry EEF reference differs from controller EEF site')
    v={k:np.asarray(getattr(c,k),float)[:3].tolist() for k in ('input_min','input_max','output_min','output_max')}
    require(np.allclose(v['input_min'],-1) and np.allclose(v['input_max'],1) and np.allclose(v['output_min'],-.05) and np.allclose(v['output_max'],.05),'Unexpected controller scaling; refuse target construction')
    return {**v,'eef_site_id':int(env.env.robots[0].eef_site_id),'goal_reference':'world_frame_site_position'}

class Capture:
    def __init__(self,env,a):self.env=env;self.a=a;self.key=None;self.latest=None
    def arm(self,key):self.key=key;self.latest=None
    def __enter__(self):
        from oracle_camera import fingerprint
        from robosuite import macros
        self.obs=self.env.env._observables['robot0_eye_in_hand_image'];self.original=self.obs._sensor
        @wraps(self.original)
        def sensor(cache):
            pixels=self.original(cache)
            if self.key is not None:
                before=fingerprint(self.env);t=truth(self.env,self.a)
                zero=self.env.sim.render(camera_name='robot0_eye_in_hand',width=256,height=256)
                seg=self.env.sim.render(camera_name='robot0_eye_in_hand',width=256,height=256,segmentation=True)
                sign=1 if macros.IMAGE_CONVENTION=='opengl' else -1
                zero=np.asarray(zero)[::sign];seg=np.asarray(seg)[::sign][::-1,::-1]
                error=float(np.sqrt(np.mean(((zero.astype(float)-np.asarray(pixels).astype(float))/255.)**2)))
                require(error<=1e-4,'Sensor-time zero-render mismatch')
                mask=(seg[:,:,0]==5)&np.isin(seg[:,:,1],self.a['render_ids']);n=int(np.sqrt(.9)*256);k=(256-n)//2
                self.latest={'key':self.key,'truth':t,'fingerprint':before,'zero_rgb_nrmse':error,
                    'visible_pixels':int(mask[k:k+n,k:k+n].sum()),'native_pixels':np.array(pixels,copy=True)}
                require(fingerprint(self.env)==before,'Read-only rendering changed physical/camera/clock state')
            return pixels
        self.obs._sensor=sensor;return self
    def take(self,raw,key):
        require(self.latest is not None and self.latest['key']==key,'Missing native sensor geometry')
        require(np.array_equal(raw['robot0_eye_in_hand_image'],self.latest['native_pixels']),'Native RGB/sample mismatch')
        return self.latest
    def __exit__(self,*_):self.obs._sensor=self.original
