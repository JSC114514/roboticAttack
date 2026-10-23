import os
import argparse
import uuid
from pathlib import Path
# 多视角参数工具保持轻量, 方便 dry-run 在无 torch 环境下运行。
try:
    from multiview_args import add_multiview_args, print_dry_run_config
except ImportError:
    from VLAAttacker.multiview_args import add_multiview_args, print_dry_run_config
def set_seed(seed: int):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def is_local_checkpoint(vla_path: str) -> bool:
    """判断 checkpoint 是否为本地路径；本地路径禁止走 HuggingFace remote-code 分发。"""
    return bool(vla_path) and Path(vla_path).expanduser().exists()


def main(args):
    pwd = os.getcwd()
    exp_id = args.exp_id if args.exp_id else str(uuid.uuid4())
    if  "bridge_orig" in args.dataset:
        vla_path = "openvla/openvla-7b"
    elif "libero_spatial" in args.dataset:
        vla_path = "openvla/openvla-7b-finetuned-libero-spatial"
    elif "libero_object" in args.dataset:
        vla_path = "openvla/openvla-7b-finetuned-libero-object"
    elif "libero_goal" in args.dataset:
        vla_path = "openvla/openvla-7b-finetuned-libero-goal"
    elif "libero_10" in args.dataset:
        vla_path = "openvla/openvla-7b-finetuned-libero-10"
    else:
        assert False, "Invalid dataset"
    # 如果脚手架提供了本地 checkpoint 路径，优先使用本地路径，避免计算节点访问 HuggingFace/代理。
    if args.vla_path:
        vla_path = args.vla_path
    # 单实验脚手架会传入统一 output_dir；为空时保持原有保存路径。
    path = args.output_dir if args.output_dir else f"{pwd}/run/UADA/{exp_id}"
    # dry-run 只打印研究配置, 必须在加载大模型和 dataloader 前返回。
    if args.dry_run:
        print_dry_run_config(args, "UADA", vla_path, path)
        return
    # 真实攻击路径才导入重依赖, 避免 dry-run 被环境依赖阻塞。
    import torch
    import numpy as np
    import wandb
    from transformers import AutoConfig, AutoTokenizer
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction

    set_seed(42)
    from white_patch.UADA import OpenVLAAttacker
    from white_patch.openvla_dataloader import get_dataloader

    target = ''
    for i in args.maskidx:
        target += str(i)
    name = f"{args.dataset}_modifyLabel_MSEDistance_lr{format(args.lr, '.0e')}_iter{args.iter}_warmup{args.warmup}_target{target}_inner_loop{args.innerLoop}_patch_size{args.patch_size}_seed42-{exp_id}"
    if args.wandb_project != "false":
        wandb_run = wandb.init(entity=args.wandb_entity, project=args.wandb_project,name=name, tags=args.tags)
        wandb.config = {"iteration":args.iter, "learning_rate": args.lr, "attack_target": args.maskidx,"accumulate_steps":args.accumulate}
    print(f"exp_id:{exp_id}")

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    quantization_config = None
    if is_local_checkpoint(vla_path):
        # 本地 checkpoint 中带有 HF auto_map；直接实例化本仓库类，避免计算节点访问 HuggingFace。
        image_processor = PrismaticImageProcessor.from_pretrained(vla_path, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(vla_path, local_files_only=True, trust_remote_code=False)
        processor = PrismaticProcessor(image_processor=image_processor, tokenizer=tokenizer)
        vla = OpenVLAForActionPrediction.from_pretrained(
            vla_path,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
    else:
        processor = AutoProcessor.from_pretrained(vla_path, trust_remote_code=True)
        vla = AutoModelForVision2Seq.from_pretrained(
            vla_path,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    vla = vla.to(device)
    os.makedirs(path, exist_ok=True)
    train_dataloader, val_dataloader = get_dataloader(
        batch_size=args.bs,
        server=args.server,
        dataset=args.dataset,
        vla_path=vla_path,
        # 多视角参数只影响当前攻击数据流；单视角默认 front 时保持旧行为。
        input_views=args.input_views,
        view_key_map=args.view_key_map,
        multiview_fusion=args.multiview_fusion,
        multiview_debug=args.multiview_debug,
    )
    openVLA_Attacker = OpenVLAAttacker(vla, processor, path,optimizer="adamW", resize_patch=args.resize_patch)

    # patch 224x224
    # patch_size=[3,22,22] - 1%
    # patch_size=[3,50,50] - 5%
    # patch_size=[3,70,70] - 10%
    # patch_size=[3,87,87] - 15%
    # patch_size=[3,100,100] - 20%
    openVLA_Attacker.patchattack_unconstrained(train_dataloader, val_dataloader, num_iter=args.iter,
                                               target_action=np.zeros(7), patch_size=args.patch_size, lr=args.lr,
                                               accumulate_steps=args.accumulate,
                                               maskidx=args.maskidx,
                                               warmup=args.warmup,
                                               filterGripTrainTo1=args.filterGripTrainTo1,
                                               geometry=args.geometry,
                                               innerLoop=args.innerLoop,
                                               args=args)

    print("Attack done!")
def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--maskidx',default='0', type=list_of_ints)
    parser.add_argument('--lr',default=1e-3, type=float)
    parser.add_argument('--server',default=os.getcwd(), type=str,help='Prefix of the server path')
    parser.add_argument('--device',default=1, type=int)
    parser.add_argument('--iter',default=2000, type=int) # 266933
    parser.add_argument('--accumulate',default=1, type=int)
    parser.add_argument('--bs',default=8, type=int)
    parser.add_argument('--warmup',default=20, type=int)
    parser.add_argument('--tags',nargs='+', default=[""])
    parser.add_argument('--filterGripTrainTo1', type=str2bool, nargs='?',default=False,
                        help='Remove the gripper 0 traning samples during the attack of target at grip to 0')
    parser.add_argument('--geometry', type=str2bool, nargs='?',default=True,
                        help='add geometry trans to path')
    parser.add_argument('--patch_size', default='3,50,50', type=list_of_ints)
    parser.add_argument('--wandb_project', default="xxx", type=str)
    parser.add_argument('--wandb_entity', default="xxx", type=str)
    parser.add_argument('--innerLoop', default=50, type=int)
    parser.add_argument('--dataset', default="bridge_orig", type=str)
    parser.add_argument('--vla_path', default="", type=str, help='可选: 本地 OpenVLA checkpoint 路径, 为空时使用原 HuggingFace 名称')
    parser.add_argument('--resize_patch', type=str2bool, default=False)
    parser.add_argument('--reverse_direction', type=str2bool, default=True)
    # 多视角字段当前仅预留接口, 不改变原有单视角攻击流程。
    add_multiview_args(parser)
    return parser.parse_args()

def list_of_ints(arg):
    return list(map(int, arg.split(',')))

def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    args = arg_parser()
    print(f"Paramters:\n maskidx:{args.maskidx}\n lr:{args.lr} \n device:{args.device} \ntags:{args.tags}")
    main(args)
