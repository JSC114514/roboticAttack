"""Read-only native loading; explicitly audit the unused diffusion-only code delta."""
import ast
from pathlib import Path
from jr_common import config, runtime_config, require, read, sha

AUTO_MAP = {'AutoConfig':'configuration_prismatic.OpenVLAConfig',
            'AutoModelForVision2Seq':'modeling_prismatic.OpenVLAForActionPrediction'}


def code_compatibility(folder, official):
    folder=Path(folder); reference=Path(official)/'prismatic/extern/hf'; proof={}
    require(read(folder/'config.json')['auto_map']==AUTO_MAP, 'Unexpected joint auto_map')
    for name in ('configuration_prismatic.py','modeling_prismatic.py'):
        a=folder/name; b=reference/name
        proof[name]=dict(checkpoint_sha256=sha(a), native_sha256=sha(b))
        if sha(a)==sha(b):
            proof[name]['comparison']='byte_identical'; continue
        require(name=='modeling_prismatic.py', 'Configuration implementation mismatch')
        # This exact public snapshot adds a scheduler reset only inside diffusion.
        # Our experiment requires L1 and rejects every other executable difference.
        tree_a=ast.parse(a.read_text()); tree_b=ast.parse(b.read_text())
        target=ast.dump(ast.parse('action_head.noise_scheduler.set_timesteps(action_head.num_diffusion_steps)').body[0],include_attributes=False)
        removed=[]
        for tree in (tree_a,tree_b):
            count=0
            for node in ast.walk(tree):
                if isinstance(node,ast.FunctionDef) and node.name=='_run_diffusion_prediction':
                    old=len(node.body)
                    node.body=[v for v in node.body if ast.dump(v,include_attributes=False)!=target]
                    count+=old-len(node.body)
            removed.append(count)
        require(removed==[1,0] and ast.dump(tree_a,include_attributes=False)==ast.dump(tree_b,include_attributes=False),
                'Native/checkpoint active model logic differs; no automatic rewrite allowed')
        proof[name].update(comparison='L1_AST_identical', inactive_delta='one diffusion scheduler reset', diffusion_enabled=False)
    return proof


def load_native_wrapper(runtime, suite):
    import torch
    cfg=runtime_config(); runtime.add_paths(cfg); torch.backends.cudnn.enabled=False
    from VLAAttacker.multiview.openvla_oft_wrapper import OpenVLAOFTWrapper
    wrapper=OpenVLAOFTWrapper(dict(model_checkpoint=cfg['checkpoints'][suite],task_suite=suite,
        unnorm_key=cfg['unnorm_keys'][suite], input_views='front,wrist',num_images_in_input=2,
        use_proprio=True,use_l1_regression=True,use_diffusion=False,use_film=False,
        num_open_loop_steps=8,device='cuda',dtype=cfg['dtype'],allow_cpu=False,hf_local_files_only=True,
        openvla_oft_root=cfg['paths']['openvla_oft_root'],libero_root=cfg['paths']['libero_root']))
    wrapper._try_import_official(); utils=wrapper._official_openvla_utils
    def verify_only(checkpoint):
        require(Path(checkpoint).resolve()==Path(config()['model_dir']).resolve(), 'Unexpected model path')
        code_compatibility(checkpoint,cfg['paths']['openvla_oft_root'])
    # Prevent the official local loader's on-disk code/config sync. Inference unchanged.
    utils.update_auto_map=verify_only; utils.check_model_logic_mismatch=verify_only
    wrapper.load_model(); require(wrapper.action_head is not None and wrapper.proprio_projector is not None, 'Missing joint head/projector')
    require(wrapper.gen_cfg.use_l1_regression and not wrapper.gen_cfg.use_diffusion and not wrapper.gen_cfg.use_film,
            'Only native L1 policy supported by the audited code comparison')
    wrapper.freeze_model(); return wrapper
