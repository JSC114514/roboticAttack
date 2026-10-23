"""Independent outputs, pinned read-only adapters, bounded mixed-suite pilot."""
from pathlib import Path
from functools import lru_cache
import ast,importlib.util,json,sys,time
ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parents[1]
OLD=PROJECT/'research_runs/wrist_dino_geometry_action_teacher_v1'
sys.path.insert(1,str(OLD/'code'))
import at_common as old
native=old.native
require,sha,read_json=native.require,native.sha,native.read_json
atomic_json,atomic_torch,atomic_parquet=native.atomic_json,native.atomic_torch,native.atomic_parquet
SOURCE=PROJECT/'research_runs/wrist_dino_relation_bias_v1'

def config():return read_json(ROOT/'config.json')
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
@lru_cache(None)
def signature():
    paths=[ROOT/'config.json',ROOT/'README.md',ROOT/'run.sh']
    for folder in ('code','slurm','tests'):paths.extend(p for p in (ROOT/folder).glob('*') if p.is_file())
    return native.digest({'files':{str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)},
        'historical_adapters':old.signature()})
def complete(path):
    p=Path(path)
    if not p.exists():return False
    r=read_json(p);require(r['signature']==signature(),'Stale revision: '+str(p))
    require(all((ROOT/k).is_file() and sha(ROOT/k)==v for k,v in r['artifacts'].items()),'Changed artifact: '+str(p))
    return r['status']=='COMPLETE'
def finish(path,artifacts,**fields):
    atomic_json({'status':'COMPLETE','signature':signature(),'completed_unix':time.time(),
        'artifacts':{str(Path(p).relative_to(ROOT)):sha(p) for p in artifacts},**fields},path)
@lru_cache(None)
def reference_engine_module():
    m=module('sr_private_native_engine',old.source.MATCH/'code/engine.py');m.ROOT=ROOT;m.signature=signature;return m
@lru_cache(None)
def monitoring():
    m=module('sr_private_monitor',old.source.MATCH/'code/common.py');m.ROOT=ROOT;m.signature=signature;return m
def Monitor(key):return monitoring().Monitor(key)
def cost(key):return monitoring().cost_attempt(key,path=ROOT/f'reports/COSTS/{key}_{time.time_ns()}.json')
def make_engine(checkpointing=False):
    from sr_engine import Engine
    return Engine(native.load_wrapper(config()['checkpoint']),checkpointing)
def candidates():return [(f'{m}_s{s}',m,s) for s in config()['seeds'] for m in config()['methods']]
def tasks():
    return [(s,t) for s in config()['suites'] for t in read_json(SOURCE/'config.json')['task_splits'][s]['train']]
def cache_receipt(e):return ROOT/f'data/cache/{e}.json'
def episodes():return read_json(ROOT/'data/MANIFEST.json')['episodes']
def cache_complete(i):return all(complete(cache_receipt(e['episode_id'])) for e in episodes()[i::2])
def records(cohort):
    import torch
    for e in episodes():
        if e['cohort']!=cohort:continue
        p=cache_receipt(e['episode_id']);require(complete(p),'Missing cache '+str(p))
        yield from torch.load(p.with_suffix('.pt'),map_location='cpu',weights_only=False)['records']
def verify_inputs():
    v=read_json(ROOT/'data/MANIFEST.json');require(v['signature']==signature(),'Preparation revision changed')
    for p,h in v['inputs'].items():require(sha(p)==h,'Changed input '+p)
    for p,expected in v['model_file_stats'].items():
        stat=Path(p).stat();require([stat.st_size,stat.st_mtime_ns]==expected,'Changed model weight metadata '+p)
    return v
def prepare():
    if complete(ROOT/'reports/PREPARE_COMPLETE.json'):return verify_inputs()
    import torch
    c=config();require(c['checkpoint']=='libero_goal' and c['max_concurrent_gpus']==2,'Protocol changed')
    require(c['patch_xy']==native.cfg()['patch_xy'] and c['patch_shape']==[3,50,50] and not c['eot'],'Patch protocol changed')
    for d in ('data/cache','reports/locks','reports/COSTS','reports/PROGRESS','reports/GATES','training','evaluation/episodes','evaluation/offline','results','logs'):(ROOT/d).mkdir(parents=True,exist_ok=True)
    index=read_json(SOURCE/'data/INDEX.json');out=[]
    for cohort,states in (('fit',c['fit_states']),('verify',c['verify_states'])):
        for s,t in tasks():
            for state in states:
                rows=[r for r in index if r['split']=='train' and r['suite']==s and r['task_id']==t and r['state_index']==state]
                require(len(rows)==5,'Incomplete source trajectory')
                r=rows[0];out.append({k:r[k] for k in ('episode_id','suite','task_id','state_index','rollout_seed','decision_count')})
                out[-1]['cohort']=cohort
    require(len(out)==96 and not ({e['episode_id'] for e in out if e['cohort']=='fit'}&{e['episode_id'] for e in out if e['cohort']=='verify'}),'Trajectory split invalid')
    inputs={str(SOURCE/'data/INDEX.json'):sha(SOURCE/'data/INDEX.json'),str(SOURCE/'config.json'):sha(SOURCE/'config.json')}
    trace=PROJECT/native.cfg()['source_run']/'results/LONG4_TRACES.parquet';inputs[str(trace)]=sha(trace)
    import pandas as pd
    traces=pd.read_parquet(trace,filters=[('condition','==','clean')],columns=['suite','task_id','state_index','rollout_seed','decision_block','executed_slots'])
    for e in out:
        group=traces[(traces.suite==e['suite'])&(traces.task_id==e['task_id'])&(traces.state_index==e['state_index'])&(traces.rollout_seed==e['rollout_seed'])].sort_values('decision_block')
        require(len(group)==e['decision_count'] and list(group.decision_block)==list(range(len(group))) and group.executed_slots.between(1,8).all(),'Invalid saved source decision sequence: '+e['episode_id'])
    library=Path(native.runtime_config()['paths']['libero_root'])/'libero/libero'
    p=library/'benchmark/libero_suite_task_map.py';inputs[str(p)]=sha(p);nodes=ast.parse(p.read_text()).body
    names=ast.literal_eval(next(n.value for n in nodes if isinstance(n,ast.Assign) and any(getattr(t,'id','')=='libero_task_map' for t in n.targets)))
    initial={}
    for s in c['suites']:
        for t,name in enumerate(names[s]):
            p=library/'init_files'/s/(name+'.pruned_init');v=torch.load(p,map_location='cpu',weights_only=False)
            require(len(v)>max(c['eval_states']+c['fit_states']+c['verify_states']),'Initial states missing')
            inputs[str(p)]=sha(p);initial[f'{s}/{t}']={'path':str(p),'task_name':name}
    for s in c['seeds']:
        patch=torch.rand((3,50,50),generator=torch.Generator().manual_seed(s));native.save_png(patch,ROOT/f'data/initial_s{s}.png')
    ckpt=Path(native.runtime_config()['checkpoints'][c['checkpoint']]);require(ckpt.is_dir(),'Goal checkpoint unavailable')
    native.runtime_module()
    from readonly_model import verify_checkpoint
    checkpoint_proof=verify_checkpoint(str(ckpt),native.runtime_config()['paths']['openvla_oft_root'])
    for name in ('config.json','modeling_prismatic.py','configuration_prismatic.py'):
        p=ckpt/name;inputs[str(p)]=sha(p)
    model_stats={}
    for p in sorted(ckpt.iterdir()):
        if p.is_file() and p.suffix in ('.safetensors','.bin','.pt','.json'):
            st=p.stat();model_stats[str(p)]=[st.st_size,st.st_mtime_ns]
            if p.suffix=='.json':inputs[str(p)]=sha(p)
    uada=PROJECT/native.cfg()['uada_png'];inputs[str(uada)]=sha(uada)
    # The audited asset files define the handle annotations and button body.
    for name in ('wooden_cabinet.xml','flat_stove.xml'):
        p=library/'assets/articulated_objects'/name;inputs[str(p)]=sha(p)
    controllers=Path(sys.prefix)/f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/robosuite/controllers'
    for name in ('osc.py','base_controller.py'):
        p=controllers/name;inputs[str(p)]=sha(p)
    manifest={'signature':signature(),'inputs':inputs,'model_file_stats':model_stats,'episodes':out,'trace_path':str(trace),'initial_states':initial,
        'checkpoint':str(ckpt),'checkpoint_proof':checkpoint_proof,'uada_png':str(uada),'maximum_frames_per_cohort':240,
        'pool':'four_suites_24_tasks_48_fit_48_verify_trajectories','sampling':'last_up_to_five_native_decisions_before_first_fixed_target_contact',
        'old_960_progress_frames_reused_as_trajectory_index_only':True,'max_closedloop_episodes':640,'independent_confirmation':False}
    atomic_json(manifest,ROOT/'data/MANIFEST.json');finish(ROOT/'reports/PREPARE_COMPLETE.json',[ROOT/'data/MANIFEST.json',*[ROOT/f'data/initial_s{s}.png' for s in c['seeds']]])
    return manifest
