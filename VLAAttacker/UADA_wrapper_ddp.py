import os
import argparse
import uuid
# 多视角参数工具保持轻量, 方便 dry-run 在无 torch 环境下运行。
try:
    from multiview_args import add_multiview_args, print_dry_run_config
except ImportError:
    from VLAAttacker.multiview_args import add_multiview_args, print_dry_run_config
# torchrun --nproc_per_node=4 VLAAttacker/UADA_wrapper3_ddp.py
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29501 VLAAttacker/UADA_wrapper3_ddp.py
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


def get_exp_id():
    import torch.distributed as dist

    rank = int(os.environ.get("RANK", 0))

    if rank == 0:
        exp_id = str(uuid.uuid4())
        print(f"Generated exp_id on rank 0: {exp_id}")
    else:
        exp_id = None

    exp_id_list = [exp_id]
    dist.broadcast_object_list(exp_id_list, src=0)

    return exp_id_list[0]

def main(args):
    # dist.init_process_group("nccl")
    pwd = os.getcwd()
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
    dry_run_exp_id = args.exp_id if args.exp_id else "auto-ddp-broadcast"
    # 单实验脚手架会传入统一 output_dir；为空时保持原有保存路径。
    path = args.output_dir if args.output_dir else f"{pwd}/run/UADA/{dry_run_exp_id}"
    # dry-run 不初始化分布式通信, 只打印配置后退出。
    if args.dry_run:
        print_dry_run_config(args, "UADA-DDP", vla_path, path)
        return
    # 真实 DDP 攻击路径才导入分布式和攻击实现。
    import wandb
    import torch.distributed as dist
    from white_patch.UADA_ddp import OpenVLAAttacker

    exp_id = args.exp_id if args.exp_id else str(get_exp_id())
    set_seed(42)
    rank = int(os.environ.get("RANK", 0))
    target = ''
    for i in args.maskidx:
        target += str(i)
    name = f"{args.dataset}_modifyLabel_MSEDistance_lr{format(args.lr, '.0e')}_iter{args.iter}_warmup{args.warmup}_target{target}_inner_loop{args.innerLoop}_patch_size{args.patch_size}_seed42-{exp_id}"
    if args.wandb_project != "false" and rank == 0:
        wandb_run = wandb.init(entity=args.wandb_entity, project=args.wandb_project,name=name, tags=args.tags)
        wandb.config = {"iteration":args.iter, "learning_rate": args.lr, "attack_target": args.maskidx,"accumulate_steps":args.accumulate}
    print(f"exp_id:{exp_id}")
    path = args.output_dir if args.output_dir else f"{pwd}/run/UADA/{exp_id}"

    os.makedirs(path, exist_ok=True)

    # train_dataset, val_dataset = get_dataset(dataset=args.dataset)
    # OpenVLAAttacker.run(vla_path,dataset_name=args.dataset, save_dir=path, resize_patch=args.resize_patch,patch_size=args.patch_size,lr=args.lr,bs=args.bs,warmup=args.warmup,num_iter=args.iter,maskidx=args.maskidx,innerLoop=args.innerLoop,geometry=args.geometry, use_wandb=True,MSE_weights=args.MSE_weights)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    instance_params = {
        "vla_path": vla_path, "dataset_name": args.dataset, "save_dir": path, "resize_patch": args.resize_patch,
        "patch_size": args.patch_size,
        "lr": args.lr, "bs": args.bs, "warmup": args.warmup,
        "num_iter": args.iter, "maskidx": args.maskidx, "innerLoop": args.innerLoop, "geometry": args.geometry,
        "use_wandb": True, "MSE_weights": args.MSE_weights
    }
    OpenVLAAttacker._attack_entry(rank, instance_params, world_size)
    # patch 224x224
    # patch_size=[3,22,22] - 1%
    # patch_size=[3,50,50] - 5%
    # patch_size=[3,70,70] - 10%
    # patch_size=[3,87,87] - 15%
    # patch_size=[3,100,100] - 20%

    print("Attack done!")
def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--maskidx',default='0', type=list_of_ints)
    parser.add_argument('--lr',default=1e-3, type=float)
    parser.add_argument('--iter',default=2000, type=int) # 266933
    parser.add_argument('--MSE_weights',default=5, type=int) # 266933
    parser.add_argument('--accumulate',default=1, type=int)
    parser.add_argument('--bs',default=8, type=int)
    parser.add_argument('--warmup',default=20, type=int)
    parser.add_argument('--tags',nargs='+', default=[""])
    parser.add_argument('--geometry', type=str2bool, nargs='?',default=True,
                        help='add geometry trans to path')
    parser.add_argument('--patch_size', default='3,50,50', type=list_of_ints)
    parser.add_argument('--wandb_project', default="xxx", type=str)
    parser.add_argument('--wandb_entity', default="xxx", type=str)
    parser.add_argument('--innerLoop', default=50, type=int)
    parser.add_argument('--dataset', default="bridge_orig", type=str)
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
    print(f"Paramters:\n maskidx:{args.maskidx}\n lr:{args.lr} \n tags:{args.tags}")
    main(args)
