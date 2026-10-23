#!/usr/bin/env python3
"""Train the single WHEN universal patch on source OpenVLA.

完整调用路径：
1. scripts/slurm/when/train_libero10_long_patch.sbatch 提交 Slurm 任务；
2. 本文件 main() 解析 config/output-dir，并做 CUDA/依赖检查；
3. _train_when_patch() 加载 source OpenVLA、冻结模型、构造 dataloader；
4. _train_when_patch() 导入 UPA-RFAS 的 OpenVLAAttacker；
5. OpenVLAAttacker.patchattack_unconstrained() 执行 WHEN 核心优化：
   clean forward -> RAUP inner minimization -> patched forward -> full objective -> patch update。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from types import SimpleNamespace
from pathlib import Path


# 这个入口脚本位于 feature_space.py 的上游：
# - 本文件负责读取配置、加载 source OpenVLA、构造 dataloader、冻结模型；
# - VLAAttacker/white_patch/feature_space.py 负责真正的 patch 优化循环。
# 这样拆开是为了把“工程环境/路径/Slurm 适配”和“论文攻击目标实现”隔离开。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from VLAAttacker.when.config import UPA_RFAS_REFERENCE, merge_when_defaults, load_when_config, resolve_output_path
from VLAAttacker.when.data import collect_requirement_status, missing_requirements
from VLAAttacker.when.patch_io import write_patch_manifest
from VLAAttacker.when.trainer import assert_cuda_available


def parse_args() -> argparse.Namespace:
    """解析训练入口参数。

    训练 patch 时最重要的是 config 和 output-dir：config 决定 WHEN objective 和数据/模型路径，
    output-dir 决定 UPA-RFAS 写 loss、patch.pt、可视化图片的位置。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-requirement-check",
        action="store_true",
        help="Skip broad repository requirement checks and let the training path fail only on resources it actually uses.",
    )
    return parser.parse_args()


def _parse_int(config: dict, key: str, default: int) -> int:
    """从 YAML 配置里安全读取整数。

    这个仓库的 WHEN 配置经过轻量 YAML parser 后多数值仍是字符串。
    单独封装解析函数，可以让调用处保持简洁，也避免某个字段写错导致整个脚本崩在类型转换上。
    """
    try:
        return int(str(config.get(key, default)).strip())
    except (TypeError, ValueError):
        return default


def _parse_float(config: dict, key: str, default: float) -> float:
    """从 YAML 配置里安全读取浮点数，并支持 2/255 这种论文常用写法。

    inner_eps 这类像素扰动半径通常用分数表示，直接支持分数字符串能让配置更贴近论文描述。
    """
    value = str(config.get(key, default)).strip()
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        return float(numerator.strip()) / float(denominator.strip())
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_bool(config: dict, key: str, default: bool) -> bool:
    """从 YAML 配置里安全读取布尔值。

    Slurm 环境变量和 YAML 手写配置里常混用 true/false、1/0、yes/no，
    统一在这里解析可以减少配置格式差异造成的误判。
    """
    value = str(config.get(key, default)).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _parse_int_list(config: dict, key: str, default: str) -> list[int]:
    """解析逗号分隔整数列表。

    patch_size、maskidx 都需要传给 UPA-RFAS 原始接口；保持字符串配置、
    入口处再转 list，可以兼容原仓库的命令行风格。
    """
    value = str(config.get(key, default)).strip()
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_phrase_list(config: dict, key: str, default: str) -> list[str]:
    """解析 PSM 使用的 probe phrases。

    这些短语会传给 feature_space.py 中的 text_similarity_loss，用来构造补丁的语义诱饵。
    """
    value = str(config.get(key, default)).strip()
    return [part.strip() for part in value.split(",") if part.strip()]


def _set_seed(seed: int) -> None:
    """固定随机种子。

    patch 初始值、dataloader shuffle、RAUP 扰动和几何变换都会影响最终 patch。
    固定 Python/NumPy/PyTorch/CUDA 种子，能让重复实验更接近可复现。
    """
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _configure_cuda_runtime(config: dict) -> None:
    """配置 CUDA/cuDNN/TF32 行为。

    这里不是为了改变攻击算法，而是为了让大模型训练在集群上稳定运行：
    TF32 提速，deterministic/benchmark 控制复现性，disable_cudnn 规避某些环境下的 cuDNN 兼容问题。
    """
    import torch

    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = _parse_bool(config, "allow_tf32", True)
    torch.backends.cudnn.allow_tf32 = _parse_bool(config, "allow_tf32", True)
    torch.backends.cudnn.deterministic = _parse_bool(config, "cudnn_deterministic", False)
    torch.backends.cudnn.benchmark = _parse_bool(config, "cudnn_benchmark", True)
    if _parse_bool(config, "disable_cudnn", False):
        torch.backends.cudnn.enabled = False
    try:
        torch.set_float32_matmul_precision(str(config.get("float32_matmul_precision", "high")))
    except Exception:
        pass


def _print_cuda_diagnostics(stage: str) -> None:
    """把 GPU 状态打印进 Slurm 日志。

    训练 OpenVLA patch 如果误跑 CPU 会非常慢且容易假死；显存和 device 信息能快速确认
    模型是否真的加载到分配的 GPU 上。
    """
    import os
    import torch

    payload = {
        "stage": stage,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "allow_tf32": torch.backends.cudnn.allow_tf32,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
    }
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        payload.update({
            "current_device": device,
            "device_name": torch.cuda.get_device_name(device),
            "memory_free_gb": round(free_bytes / 1024 ** 3, 3),
            "memory_total_gb": round(total_bytes / 1024 ** 3, 3),
            "memory_allocated_gb": round(torch.cuda.memory_allocated(device) / 1024 ** 3, 3),
            "memory_reserved_gb": round(torch.cuda.memory_reserved(device) / 1024 ** 3, 3),
        })
    print("[when-train][cuda] " + json.dumps(payload, ensure_ascii=False))


def _load_source_openvla(checkpoint: str):
    """加载 source OpenVLA 模型和 processor。

    WHEN 原论文训练是在 source OpenVLA 上优化 patch，再转移到其他模型/套件评估。
    所以这里加载的是 source checkpoint，不是 OFT surrogate。若 checkpoint 是本地目录，
    强制 local_files_only，避免计算节点联网或加载远端代码造成不可控差异。
    """
    import torch
    from transformers import AutoConfig, AutoModelForVision2Seq, AutoProcessor, AutoTokenizer
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    checkpoint_path = Path(checkpoint)
    quantization_config = None
    if checkpoint_path.exists():
        image_processor = PrismaticImageProcessor.from_pretrained(checkpoint, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
        processor = PrismaticProcessor(image_processor=image_processor, tokenizer=tokenizer)
        vla = OpenVLAForActionPrediction.from_pretrained(
            checkpoint,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            local_files_only=True,
            attn_implementation="eager",
        )
    else:
        processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
        vla = AutoModelForVision2Seq.from_pretrained(
            checkpoint,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation="eager",
        )
    return processor, vla


def _build_attack_args(config: dict) -> SimpleNamespace:
    """把配置整理成 UPA-RFAS OpenVLAAttacker 期望的 args 对象。

    外部 feature_space.py 使用 args.lambda_con、args.step 这类属性读取超参，
    所以这里用 SimpleNamespace 做一层适配；这也让我们无需改动论文参考代码的接口。
    """
    return SimpleNamespace(
        l2=_parse_int(config, "l2", 0),
        dual=_parse_int(config, "dual", 1),
        eps=_parse_float(config, "inner_eps", 2 / 255),
        step=_parse_int(config, "inner_steps", 8),
        out_lr=_parse_float(config, "out_lr", 0.5 / 255),
        lambda_con=_parse_float(config, "lambda_con", 1.0),
        lambda_1=_parse_float(config, "lambda_1", 0.1),
        lambda_attn=_parse_float(config, "lambda_attn", 1.0),
        lambda_text=_parse_float(config, "lambda_text", 0.5),
        last_n=_parse_int(config, "last_n", 2),
        nonpatch=_parse_float(config, "lambda_nonpatch", 0.8),
        topk_text=_parse_float(config, "topk_text", 0.3),
        margin=_parse_float(config, "margin", 0.1),
        phrase=_parse_phrase_list(
            config,
            "probe_phrases",
            "put,pick up,place,turn on,push,open,close,left,right,bottom,back,middle,top,front",
        ),
        tau=_parse_float(config, "tau", 0.3),
        beta_neg=_parse_float(config, "beta_neg", 0.5),
        save_every=_parse_int(config, "save_every", 100),
        wandb_project=str(config.get("wandb_project", "false")),
    )


def _train_when_patch(config: dict, output_dir: Path) -> dict:
    """连接上游工程代码和 UPA-RFAS 核心训练代码。

    这个函数完成四个关键动作：
    1. 把配置指定的 UPA-RFAS 路径加进 sys.path，导入官方 OpenVLAAttacker；
    2. 加载并冻结 source OpenVLA；
    3. 构造 train/val dataloader；
    4. 调用 OpenVLAAttacker.patchattack_unconstrained 真正优化 patch。
    """
    # 从这里开始进入真实训练路径。上面的函数基本都是“配置/环境适配”；
    # 下面的代码会创建 frozen surrogate、dataloader 和 patch optimizer 所需的攻击对象。
    import numpy as np
    import torch
    from VLAAttacker.white_patch.openvla_dataloader import get_dataloader

    upa_rfas_reference = Path(str(config.get("upa_rfas_reference", UPA_RFAS_REFERENCE))).expanduser()
    if not upa_rfas_reference.is_absolute():
        upa_rfas_reference = (PROJECT_ROOT / upa_rfas_reference).resolve()
    if str(upa_rfas_reference) not in sys.path:
        sys.path.insert(0, str(upa_rfas_reference))
    # 这里导入的是 WHEN/UPA-RFAS 的核心训练类，不是本仓库 G1/G3 的 OFT 复刻入口。
    # 后续 patchattack_unconstrained() 里的代码才是论文式 feature/PAD/PSM/RAUP objective。
    from white_patch.feature_space import OpenVLAAttacker

    # wandb 默认离线，避免 Slurm 任务因为网络、登录状态或代理问题卡住。
    os.environ.setdefault("WANDB_MODE", "offline")
    seed = _parse_int(config, "seed", 42)
    _set_seed(seed)
    _configure_cuda_runtime(config)
    _print_cuda_diagnostics("before_model_load")

    source_checkpoint = str(config["source_model_checkpoint"])
    processor, vla = _load_source_openvla(source_checkpoint)
    device = torch.device("cuda:0")
    # 模型只作为 frozen surrogate 提供梯度信号；攻击训练只优化 patch。
    # eval() 关闭 dropout 等训练态随机性，让 clean/adv 差异主要来自 patch。
    vla = vla.to(device).eval()
    for parameter in vla.parameters():
        parameter.requires_grad_(False)
    torch.cuda.empty_cache()
    _print_cuda_diagnostics("after_model_load")

    # dataloader 输出 OpenVLA 训练需要的 pixel_values、input_ids、attention_mask、labels。
    # feature_space.py 不关心数据来自哪个路径，只接收已经拼好的 batch。
    train_dataloader, val_dataloader = get_dataloader(
        batch_size=_parse_int(config, "batch_size", 2),
        server=str(config.get("data_root", "/home/caijiasheng25/Desktop/roboticAttack")),
        dataset=str(config.get("dataset", "libero_10")),
        vla_path=source_checkpoint,
    )
    if val_dataloader is None:
        val_dataloader = train_dataloader

    output_dir.mkdir(parents=True, exist_ok=True)
    attack_args = _build_attack_args(config)
    # OpenVLAAttacker 是论文代码里的核心训练类。这里传入 frozen model、processor 和输出目录，
    # 后续 patch.pt、loss、可视化图片都会由它写入 output_dir。
    attacker = OpenVLAAttacker(
        vla,
        processor,
        str(output_dir),
        optimizer="adamW",
        resize_patch=_parse_bool(config, "resize_patch", False),
        dataset=str(config.get("dataset", "libero_10")),
    )
    _print_cuda_diagnostics("before_patch_attack")

    # UPA-RFAS 原始实现会相对当前工作目录保存若干中间文件。
    # 临时切到 output_dir，可以把这些文件收拢到本次实验目录，避免污染仓库根目录。
    previous_cwd = Path.cwd()
    os.chdir(output_dir)
    try:
        # 函数调用进入核心模块：
        # VLAAttacker/white_patch/feature_space.py
        #   OpenVLAAttacker.patchattack_unconstrained(...)
        # 传入的参数基本一一对应 WHEN 配置：2000 outer iter、50 inner loop、RAUP eps/step、
        # contrastive/PAD/PSM/feature loss 权重，以及 patch_size/maskidx。
        attacker.patchattack_unconstrained(
            train_dataloader,
            val_dataloader,
            num_iter=_parse_int(config, "iterations", 2000),
            target_action=np.zeros(7),
            patch_size=_parse_int_list(config, "patch_size", "3,50,50"),
            lr=_parse_float(config, "learning_rate", 1e-3),
            accumulate_steps=_parse_int(config, "accumulate_steps", 1),
            maskidx=_parse_int_list(config, "maskidx", "0,1,2"),
            warmup=_parse_int(config, "warmup_steps", 20),
            filterGripTrainTo1=_parse_bool(config, "filterGripTrainTo1", False),
            geometry=_parse_bool(config, "geometry", True),
            innerLoop=_parse_int(config, "inner_loop", 50),
            args=attack_args,
        )
    finally:
        os.chdir(previous_cwd)

    patch_path = output_dir / "last" / "patch.pt"
    if not patch_path.exists():
        raise FileNotFoundError(f"Expected trained patch at {patch_path}")
    # manifest 是评估脚本和后续交接的统一入口：它记录 patch 路径、sha256、训练配置和来源代码。
    # 后续 eval_source/eval_oft 只需要 PATCH_MANIFEST，就能准确找到对应补丁。
    manifest = write_patch_manifest(
        output_dir / "patch_manifest.json",
        patch_path=patch_path,
        config=config,
        extra={
            "training_entrypoint": "scripts/when/train_sim_patch.py",
            "upa_rfas_reference": str(upa_rfas_reference),
            "seed": seed,
            "patch_size": _parse_int_list(config, "patch_size", "3,50,50"),
            "iterations": _parse_int(config, "iterations", 2000),
            "batch_size": _parse_int(config, "batch_size", 2),
            "learning_rate": _parse_float(config, "learning_rate", 1e-3),
            "maskidx": _parse_int_list(config, "maskidx", "0,1,2"),
            "lambda_con": _parse_float(config, "lambda_con", 1.0),
            "lambda_attn": _parse_float(config, "lambda_attn", 1.0),
            "lambda_text": _parse_float(config, "lambda_text", 0.5),
            "lambda_1": _parse_float(config, "lambda_1", 0.1),
        },
    )
    return manifest


def main() -> None:
    """主入口：解析配置、做环境检查、启动训练。

    dry-run 到打印 resolved 配置就结束；真实训练会先要求 CUDA 可用，再把 resolved_config.json
    写到输出目录，保证实验结束后还能追溯当时的参数。
    """
    # Python 训练入口从 main() 开始。Slurm 只负责调用本脚本；
    # main() 才决定 dry-run、依赖检查、CUDA 检查和最终训练函数调用。
    args = parse_args()
    config = merge_when_defaults(load_when_config(args.config))
    output_dir = resolve_output_path(args.output_dir)
    if args.skip_requirement_check:
        status = {"skipped": True}
        missing = []
    else:
        status = collect_requirement_status()
        missing = missing_requirements(status)

    resolved = {
        "config": config,
        "output_dir": str(output_dir),
        "requirement_missing": missing,
        "requirement_check_skipped": bool(args.skip_requirement_check),
    }
    print(json.dumps(resolved, indent=2, ensure_ascii=False))

    if args.dry_run:
        return
    if missing:
        raise SystemExit("Cannot train WHEN patch until missing requirements are resolved.")
    assert_cuda_available()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(json.dumps(resolved, indent=2, ensure_ascii=False) + "\n")
    manifest = _train_when_patch(config, output_dir)
    print(json.dumps({"patch_manifest": manifest}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
