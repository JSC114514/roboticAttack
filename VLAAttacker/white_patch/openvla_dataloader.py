import os

# 这个 dataloader 同时服务两类路径：
# 1. source OpenVLA / WHEN 训练：提供官方 OpenVLA 所需的 pixel_values、input_ids、labels；
# 2. OFT 多视角/RAW patch 实验：额外保留 raw_pixel_values_by_view、actions、proprio。
# 由于 RLDS/TFDS 自己会开线程，默认把 PyTorch、BLAS、TensorFlow 线程数压低，
# 可以避免 Slurm 作业在 CPU 侧过度抢占。
#
# WHEN 训练路径里的位置：
# scripts/when/train_sim_patch.py::_train_when_patch()
#   -> get_dataloader()
#      -> RLDSDataset 读取 LIBERO/Bridge RLDS 数据
#      -> RLDSBatchTransform 把 observation/action 转成 OpenVLA 输入
#      -> PaddedCollatorForActionPrediction 拼 batch
#   -> 返回 train_dataloader / val_dataloader 给 feature_space.py 的优化循环。
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TORCH_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_NUM_PARALLEL_CALLS", "2")
os.environ.setdefault("TF_TRAJ_TRANSFORM_THREADS", "2")
os.environ.setdefault("TF_TRAJ_READ_THREADS", "2")

import inspect
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor
from transformers import AutoConfig, AutoImageProcessor, AutoTokenizer
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction as OfficialPaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform as OfficialRLDSBatchTransform, RLDSDataset
from prismatic.vla.constants import IGNORE_INDEX

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
import random


def _positive_int_env(name, default):
    """从环境变量读取正整数。

    线程数通常由 sbatch 或 shell 环境控制；这里统一做容错，避免空字符串或非法值传给
    torch/tensorflow 的线程接口。
    """
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def configure_runtime_threads():
    """限制 PyTorch 和 TensorFlow 线程数。

    RLDSDataset 底层使用 TensorFlow/TFDS 管线，如果 DataLoader 再开很多 worker/线程，
    会很容易把登录节点或计算节点 CPU 打满。限制线程能让 GPU 训练更稳定。
    """
    torch_threads = _positive_int_env("TORCH_NUM_THREADS", _positive_int_env("OMP_NUM_THREADS", 2))
    try:
        import torch

        torch.set_num_threads(torch_threads)
        torch.set_num_interop_threads(min(2, torch_threads))
    except Exception as exc:
        print(f"[dataloader][warn] failed to set torch thread limits: {exc}")

    try:
        import tensorflow as tf

        tf.config.threading.set_intra_op_parallelism_threads(_positive_int_env("TF_NUM_INTRAOP_THREADS", 2))
        tf.config.threading.set_inter_op_parallelism_threads(_positive_int_env("TF_NUM_INTEROP_THREADS", 1))
    except Exception as exc:
        print(f"[dataloader][warn] failed to set tensorflow thread limits: {exc}")

VIEW_TO_CAMERA_VIEW = {
    "front": "primary",
    "primary": "primary",
    "third_person": "primary",
    "agent": "primary",
    "side": "secondary",
    "secondary": "secondary",
    "wrist": "wrist",
    "hand": "wrist",
}

VIEW_TO_OBS_KEY = {
    "front": "image_primary",
    "primary": "image_primary",
    "third_person": "image_primary",
    "agent": "image_primary",
    "side": "image_secondary",
    "secondary": "image_secondary",
    "wrist": "image_wrist",
    "hand": "image_wrist",
}


def is_local_checkpoint(vla_path):
    """判断 checkpoint 是否为本地路径。

    本地路径优先走 local_files_only，目的是让集群任务不依赖外网，也避免 trust_remote_code
    拉取远端实现造成训练不可复现。
    """
    return bool(vla_path) and Path(vla_path).expanduser().exists()


def load_openvla_processor(vla_path):
    """加载 OpenVLA processor。

    processor 决定图像 normalize/resize、tokenizer 和 prompt 编码方式，必须和 checkpoint 匹配。
    本地 checkpoint 使用 PrismaticProcessor 手动组装，避免自动类访问 HuggingFace。
    """
    if is_local_checkpoint(vla_path):
        image_processor = PrismaticImageProcessor.from_pretrained(vla_path, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(vla_path, local_files_only=True, trust_remote_code=False)
        return PrismaticProcessor(image_processor=image_processor, tokenizer=tokenizer)
    return AutoProcessor.from_pretrained(vla_path, trust_remote_code=True)


def parse_views(value):
    """解析 front,wrist 形式的视角列表。

    多视角 OFT 和 raw patch 路径会传入 front,wrist；source OpenVLA WHEN 训练通常只用 front。
    统一解析能让上游脚本用同一套 dataloader 接口。
    """
    if not value:
        return ["front"]
    if isinstance(value, str):
        return [view.strip() for view in value.split(",") if view.strip() and view.strip() != "none"]
    return [str(view).strip() for view in value if str(view).strip() and str(view).strip() != "none"]


def parse_view_key_map(value):
    """解析 front=image_primary,wrist=image_wrist 形式的显式映射。

    不同 LIBERO/RLDS 数据集 observation key 可能不同。显式映射可以避免把 wrist/front
    错接到错误摄像头上。
    """
    if isinstance(value, dict):
        return value
    mapping = {}
    if not value:
        return mapping
    for item in str(value).split(","):
        if "=" not in item:
            continue
        key, val = item.split("=", 1)
        if key.strip() and val.strip():
            mapping[key.strip()] = val.strip()
    return mapping


def infer_camera_views(input_views, view_key_map=None):
    """把研究视角名转换成 RLDS/OXE 的 camera view 名称。

    RLDSDataset 的 load_camera_views 通常使用 primary/wrist 这样的底层名字，
    而实验配置更常用 front/wrist。这里负责把研究语义映射到底层数据字段。
    """
    view_key_map = view_key_map or {}
    camera_views = []
    for view in parse_views(input_views):
        mapped_key = view_key_map.get(view, "")
        if mapped_key.startswith("image_"):
            camera_view = mapped_key.replace("image_", "", 1)
        else:
            camera_view = VIEW_TO_CAMERA_VIEW.get(view, view)
        if camera_view not in camera_views:
            camera_views.append(camera_view)
    return tuple(camera_views or ["primary"])


def resolve_obs_key(view, view_key_map=None):
    """把实验视角名解析成 RLDS observation key。

    batch transform 阶段直接访问 observation['image_primary'] 这类 key，因此需要在这里
    把 front/wrist/side 等实验名转换成真实 observation 字段名。
    """
    view_key_map = view_key_map or {}
    return view_key_map.get(view, VIEW_TO_OBS_KEY.get(view, view))


def pil_to_chw_float(image):
    """把 PIL/ndarray 图像转成 [3,H,W] float tensor。

    raw RGB patch transform 需要在进入 processor 前保留原始视角图像。这里转换成 torch tensor，
    后续才能用可微 resize/crop/normalize 把 raw patch 梯度传回 patch。
    """
    if not hasattr(image, "convert"):
        from PIL import Image
        image = Image.fromarray(np.asarray(image, dtype=np.uint8))
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


class PatchAwareRLDSBatchTransform:
    """把单条 RLDS 样本转换成 OpenVLA 训练样本。

    官方 RLDSBatchTransform 只返回 processor 后的 pixel_values 和文本/action token。
    这里额外保存 raw per-view RGB、连续 actions、proprio 等字段，是为了让后续 OFT/raw patch
    实验能在不重读数据集的情况下构造不同视角分支。
    """

    def __init__(
            self,
            action_tokenizer,
            base_tokenizer,
            image_transform,
            prompt_builder_fn,
            predict_stop_token=True,
            input_views=("front",),
            view_key_map=None,
            multiview_fusion="stitch",
            use_wrist_image=False,
            use_proprio=False):
        """保存 tokenizer、图像 transform 和视角配置。

        这些对象都来自和 checkpoint 匹配的 processor；如果 tokenizer/processor 不一致，
        labels 和 pixel_values 会和模型预期错位。
        """
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn
        self.predict_stop_token = predict_stop_token
        self.input_views = tuple(parse_views(input_views) or ["front"])
        self.view_key_map = view_key_map or {}
        self.multiview_fusion = multiview_fusion
        self.use_wrist_image = bool(use_wrist_image)
        self.use_proprio = bool(use_proprio)
        self._missing_view_warnings = set()

    def _collect_raw_images(self, observation):
        """从 RLDS observation 中提取原始 RGB 图像。

        保留 raw 图像的原因是 processed pixel_values 已经 normalize 到模型空间；
        若想训练 raw RGB patch，就必须在 processor 前贴 patch，再走同一套 resize/normalize。
        """
        from PIL import Image

        available = sorted([key for key in observation.keys() if key.startswith("image_")])
        raw_by_view = {}
        for view in self.input_views:
            obs_key = resolve_obs_key(view, self.view_key_map)
            if obs_key in observation:
                raw_by_view[view] = Image.fromarray(observation[obs_key][0]).convert("RGB")
            else:
                warning_key = (view, obs_key)
                if warning_key not in self._missing_view_warnings:
                    print(
                        f"[raw_patch][warn] requested view {view}->{obs_key} missing; "
                        f"available image keys={available}"
                    )
                    self._missing_view_warnings.add(warning_key)
        if not raw_by_view:
            fallback_key = "image_primary" if "image_primary" in observation else (available[0] if available else None)
            if fallback_key is None:
                raise KeyError(f"No image_* key found in RLDS observation: {list(observation.keys())}")
            raw_by_view["front"] = Image.fromarray(observation[fallback_key][0]).convert("RGB")
        return raw_by_view, available

    def __call__(self, rlds_batch):
        """把 RLDS transition 编码成模型 batch 单样本。

        OpenVLA 把连续动作离散成 action tokens，并和 instruction 拼成对话式 prompt。
        labels 只在 action token 段保留监督，prompt 部分设为 IGNORE_INDEX，避免模型被训练去复述输入。
        """
        dataset_name = rlds_batch["dataset_name"]
        actions = np.array(rlds_batch["action"], dtype=np.float32, copy=True)
        current_action = actions[0]
        observation = rlds_batch["observation"]
        raw_by_view, available = self._collect_raw_images(observation)
        lang_raw = rlds_batch["task"]["language_instruction"]
        lang = lang_raw.decode().lower() if hasattr(lang_raw, "decode") else str(lang_raw).lower()

        future_actions = actions[1:]
        future_actions_string = "".join(self.action_tokenizer(future_actions)) if len(future_actions) else ""
        action_chunk_string = self.action_tokenizer(current_action) + future_actions_string
        action_chunk_len = len(action_chunk_string)

        prompt_builder = self.prompt_builder_fn("openvla")
        for turn in [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": action_chunk_string},
        ]:
            prompt_builder.add_turn(turn["from"], turn["value"])

        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        input_ids = torch.tensor(input_ids)
        labels = input_ids.clone()
        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        primary_view = self.input_views[0] if self.input_views[0] in raw_by_view else next(iter(raw_by_view))
        output = {
            "pixel_values": self.image_transform(raw_by_view[primary_view]),
            "raw_pixel_values_by_view": raw_by_view,
            "input_views": list(self.input_views),
            "available_view_keys": available,
            "input_ids": input_ids,
            "labels": labels,
            "dataset_name": dataset_name,
            "instructions": lang,
            "actions": actions,
        }
        wrist_tensors = [
            self.image_transform(raw_by_view[view])
            for view in self.input_views
            if view != primary_view and "wrist" in view and view in raw_by_view
        ]
        if self.use_wrist_image and not wrist_tensors:
            wrist_tensors = [
                self.image_transform(image)
                for view, image in raw_by_view.items()
                if view != primary_view and "wrist" in view
            ]
        if wrist_tensors:
            output["pixel_values_wrist"] = torch.cat(wrist_tensors, dim=0)
        if self.use_proprio and "proprio" in observation:
            output["proprio"] = np.asarray(observation["proprio"], dtype=np.float32)
        return output


class PatchAwarePaddedCollatorForActionPrediction:
    """把多个 transform 后的样本拼成 batch。

    官方 collator 只负责 padding input_ids/labels 和 stack pixel_values。
    这里还要保留 raw_pixel_values_by_view、actions、proprio，因为 OFT action-head loss、
    raw RGB patch training 和多视角诊断都需要这些额外字段。
    """

    def __init__(self, model_max_length, pad_token_id, padding_side="right"):
        """记录 padding 策略。

        OpenVLA prompt/action token 是右 padding；如果 padding 方向错了，labels 的 action token
        对齐会被破坏。
        """
        self.model_max_length = model_max_length
        self.pad_token_id = pad_token_id
        self.padding_side = padding_side

    def __call__(self, instances):
        """拼 batch 并保留补丁训练所需的额外字段。"""
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        assert self.padding_side == "right", f"Invalid Tokenizer `{self.padding_side = }`"
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]
        attention_mask = input_ids.ne(self.pad_token_id)

        pixel_values = torch.stack([instance["pixel_values"] for instance in instances])
        if "pixel_values_wrist" in instances[0]:
            pixel_values_wrist = torch.stack([instance["pixel_values_wrist"] for instance in instances])
            pixel_values = torch.cat((pixel_values, pixel_values_wrist), dim=1)

        output = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "instructions": [instance.get("instructions", "") for instance in instances],
            "actions": torch.stack([
                torch.as_tensor(np.array(instance["actions"], dtype=np.float32, copy=True), dtype=torch.float32)
                for instance in instances
            ]),
        }
        if "proprio" in instances[0] and instances[0]["proprio"] is not None:
            proprio = torch.stack([
                torch.as_tensor(np.array(instance["proprio"], dtype=np.float32, copy=True), dtype=torch.float32)
                for instance in instances
            ])
            if proprio.ndim > 2:
                proprio = proprio.reshape(proprio.shape[0], -1)
            output["proprio"] = proprio
        if "raw_pixel_values_by_view" in instances[0]:
            view_names = list(instances[0].get("input_views", []))
            if not view_names:
                view_names = sorted({
                    view
                    for instance in instances
                    for view in instance.get("raw_pixel_values_by_view", {}).keys()
                })
            raw_by_view = {}
            for view in view_names:
                raw_by_view[view] = torch.stack([
                    pil_to_chw_float(instance["raw_pixel_values_by_view"][view])
                    for instance in instances
                ])
            output["raw_pixel_values_by_view"] = raw_by_view
            output["input_views"] = view_names
            output["available_view_keys"] = instances[0].get("available_view_keys", [])
        if "dataset_name" in instances[0]:
            output["dataset_names"] = [instance["dataset_name"] for instance in instances]
        return output

def get_bridge_dataloader(batch_size,server):
    """构造 Bridge 数据集 dataloader。

    这个函数保留原始 OpenVLA/UPA-RFAS 的 Bridge 训练路径；当前 LIBERO-10 WHEN 复现主要走
    get_dataloader，但保留 Bridge 入口方便复现实验或迁移测试。
    """
    configure_runtime_threads()
    vla_path: str = "openvla/openvla-7b"
    data_root_dir = Path(f"{server}/openvla-main/dataset")
    dataset_name = "bridge_orig"
    shuffle_buffer_size = 100_000
    image_aug = False
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Load OpenVLA Processor and Model using HF AutoClasses
    processor = load_openvla_processor(vla_path)

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    bridge_transform_params = inspect.signature(OfficialRLDSBatchTransform).parameters
    bridge_transform_kwargs = {
        "prompt_builder_fn": PurePromptBuilder if "v01" not in vla_path else VicunaV15ChatPromptBuilder,
    }
    if "image_transform" in bridge_transform_params:
        bridge_transform_kwargs["image_transform"] = processor.image_processor.apply_transform
    batch_transform = OfficialRLDSBatchTransform( #
        action_tokenizer,
        processor.tokenizer,
        **bridge_transform_kwargs,
    )
    vla_dataset_train = RLDSDataset(
        data_root_dir,
        dataset_name,
        batch_transform,
        resize_resolution=tuple([224,224]),
        shuffle_buffer_size=shuffle_buffer_size,
        train=True,
        image_aug=image_aug,
    )
    vla_dataset_val = RLDSDataset(
        data_root_dir,
        dataset_name,
        batch_transform,
        resize_resolution=tuple([224,224]),
        shuffle_buffer_size=shuffle_buffer_size,
        train=False,
        image_aug=image_aug,
    )

    # Create Collator and DataLoader
    collator = OfficialPaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    train_dataloader = DataLoader(
        vla_dataset_train,
        batch_size=batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
        # shuffle=True
    )
    val_dataloader = DataLoader(
        vla_dataset_val,
        batch_size=8,#32
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
    )
    return train_dataloader, val_dataloader


LIBERO_LONG_DATASET_ALIASES = {
    "libero_long",
    "libero_long_no_noops",
    "LIBERO_LONG",
    "LIBERO_LONG_no_noops",
    "LIBERO-Long",
    "LIBERO-Long_no_noops",
    "libero-long",
    "libero-long_no_noops",
    "libero_90",
    "libero_90_no_noops",
    "libero-long-rlds",
    "libero-long-rlds_no_noops",
}


def resolve_rlds_data_root_dir(server):
    """兼容不同 TFDS/RLDS 数据根目录布局。

    有的调用传入数据集目录本身，有的传入 openvla-main 或 openvla-main/dataset。
    在这里统一解析，可以避免每个 sbatch 都写一套路径判断。
    """
    base = Path(server).expanduser()
    if (base / "1.0.0").exists() or (base / "dataset_info.json").exists():
        return base.parent
    for candidate in [
        base / "openvla-main" / "dataset",
        base / "dataset",
        base,
    ]:
        if candidate.exists():
            return candidate
    return base / "openvla-main" / "dataset"


def resolve_existing_dataset_name(data_root_dir, requested, candidates):
    """从候选数据集名里选择本地真实存在的目录。

    libero_10 / LIBERO-Long 在不同仓库和导出脚本里命名不完全一致；
    自动选择存在的目录，能减少因为别名差异导致的训练失败。
    """
    names = [str(requested)]
    names.extend(str(item) for item in candidates)
    deduped = []
    for name in names:
        if name and name not in deduped:
            deduped.append(name)
    for name in deduped:
        path = data_root_dir / name
        if path.exists() or (path / "1.0.0").exists():
            return name
    return deduped[0]


def get_dataloader(
        batch_size,
        server,
        dataset,
        vla_path,
        input_views="front",
        view_key_map="",
        multiview_fusion="stitch",
        use_proprio=False,
        num_images_in_input=None,
        action_chunk_len=None,
        return_raw_images=False,
        multiview_debug=False):
    """构造 OpenVLA/LIBERO 训练 dataloader。

    这是 train_sim_patch.py 调用的数据入口。它做三件关键事：
    1. 根据 suite 名选择 checkpoint 默认值和 TFDS dataset 名；
    2. 注册 OpenVLA/Prismatic 的 HuggingFace AutoClass；
    3. 选择官方 transform 或 patch-aware transform，最后返回 PyTorch DataLoader。
    """
    # 这是 WHEN 数据流真正开始的地方。返回的 batch 会直接进入
    # OpenVLAAttacker.patchattack_unconstrained()，所以这里的图像 transform、
    # tokenizer 和 action tokenizer 必须与 source OpenVLA checkpoint 匹配。
    configure_runtime_threads()
    # vla_path: str = "openvla/openvla-7b"
    data_root_dir = resolve_rlds_data_root_dir(server)
    # dataset_name = "bridge_orig"
    # 这里把人类可读的 suite 名转换成 RLDS/TFDS 实际目录名。
    # LIBERO 官方数据常带 _no_noops，动作统计和 OpenVLA checkpoint 也依赖这个约定。
    if dataset == "bridge_orig":
        # 保留外部传入的本地 checkpoint；为空时才回退到原 HuggingFace 名称。
        vla_path = vla_path or "openvla/openvla-7b"
    elif dataset == "libero_spatial":
        vla_path = vla_path or "openvla/openvla-7b-finetuned-libero-spatial"
        dataset += "_no_noops"
    elif dataset == "libero_object":
        vla_path = vla_path or "openvla/openvla-7b-finetuned-libero-object"
        dataset += "_no_noops"
    elif dataset == "libero_goal":
        vla_path = vla_path or "openvla/openvla-7b-finetuned-libero-goal"
        dataset += "_no_noops"
    elif dataset == "libero_10":
        vla_path = vla_path or "openvla/openvla-7b-finetuned-libero-10"
        dataset += "_no_noops"
    elif str(dataset) in LIBERO_LONG_DATASET_ALIASES:
        vla_path = vla_path or "openvla/openvla-7b-finetuned-libero-long"
        dataset = resolve_existing_dataset_name(
            data_root_dir,
            dataset,
            [
                f"{dataset}_no_noops" if not str(dataset).endswith("_no_noops") else str(dataset),
                "libero_long_no_noops",
                "libero_long",
                "LIBERO-Long",
                "libero-long",
                "libero_90_no_noops",
                "libero_90",
                "libero-long-rlds",
            ],
        )
    else:
        assert False, "Invalid dataset"
    shuffle_buffer_size = 100_000
    image_aug = False
    # 多视角实验需要区分“实验视角名”和“RLDS 底层 camera view”。
    # source WHEN 训练一般只用默认 front，但同一个函数也服务 OFT front/wrist 路径。
    parsed_view_key_map = parse_view_key_map(view_key_map)
    load_camera_views = infer_camera_views(input_views, parsed_view_key_map)
    if multiview_debug:
        print(
            "[multiview_debug] dataloader "
            f"input_views={parse_views(input_views)} load_camera_views={load_camera_views} "
            f"view_key_map={parsed_view_key_map if parsed_view_key_map else '默认映射'} "
            f"fusion={multiview_fusion}"
        )
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Load OpenVLA Processor and Model using HF AutoClasses
    # processor 同时决定 image normalization 和 tokenizer；训练 patch 时，
    # clean/adv 两条分支必须共享同一个 processor，否则 feature deviation 没有可比性。
    processor = load_openvla_processor(vla_path)

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    # return_raw_images=True 时使用 patch-aware transform，额外保留 raw RGB；
    # 否则走官方 transform，保持 source OpenVLA WHEN 训练路径尽量接近原实现。
    transform_cls = PatchAwareRLDSBatchTransform if return_raw_images else OfficialRLDSBatchTransform
    transform_params = inspect.signature(transform_cls).parameters
    transform_kwargs = {
        "prompt_builder_fn": PurePromptBuilder if "v01" not in vla_path else VicunaV15ChatPromptBuilder,
    }
    if "image_transform" in transform_params:
        transform_kwargs["image_transform"] = processor.image_processor.apply_transform
    if "input_views" in transform_params:
        transform_kwargs["input_views"] = tuple(parse_views(input_views))
    if "view_key_map" in transform_params:
        transform_kwargs["view_key_map"] = parsed_view_key_map
    if "multiview_fusion" in transform_params:
        transform_kwargs["multiview_fusion"] = multiview_fusion
    if "use_wrist_image" in transform_params:
        requested_views = set(parse_views(input_views))
        requested_num_images = int(num_images_in_input) if num_images_in_input else len(requested_views)
        transform_kwargs["use_wrist_image"] = "wrist" in requested_views or requested_num_images > 1
    if "use_proprio" in transform_params:
        transform_kwargs["use_proprio"] = bool(use_proprio)
    if multiview_debug:
        print(
            "[multiview_debug] RLDSBatchTransform "
            f"module={transform_cls.__module__} supported={sorted(transform_params)} "
            f"kwargs={sorted(transform_kwargs)} return_raw_images={return_raw_images}"
        )
    batch_transform = transform_cls(action_tokenizer, processor.tokenizer, **transform_kwargs)

    dataset_params = inspect.signature(RLDSDataset).parameters
    dataset_kwargs = {
        "resize_resolution": tuple([224,224]),
        "shuffle_buffer_size": shuffle_buffer_size,
        "image_aug": image_aug,
    }
    if "load_camera_views" in dataset_params:
        dataset_kwargs["load_camera_views"] = load_camera_views
    if "load_proprio" in dataset_params:
        dataset_kwargs["load_proprio"] = bool(use_proprio)
    if "future_action_window_size" in dataset_params and action_chunk_len is not None:
        # OpenVLA-OFT rollout predicts an open-loop action chunk. Matching the
        # RLDS future window makes the action-token mask and action_head target
        # cover the same horizon instead of only the first 7 DoF tokens.
        dataset_kwargs["future_action_window_size"] = max(0, int(action_chunk_len) - 1)
    if multiview_debug:
        print(
            "[multiview_debug] RLDSDataset "
            f"module={RLDSDataset.__module__} supported={sorted(dataset_params)} "
            f"kwargs={sorted(dataset_kwargs)}"
        )
    # RLDSDataset 每次产出单条轨迹样本；batch_transform 在内部把原始 observation、
    # language instruction、action 转成模型 forward 需要的字段。
    vla_dataset_train = RLDSDataset(
        data_root_dir,
        dataset,
        batch_transform,
        train=True,
        **dataset_kwargs,
    )
    try:
        vla_dataset_val = RLDSDataset(
            data_root_dir,
            dataset,
            batch_transform,
            train=False,
            **dataset_kwargs,
        )
    except ValueError as exc:
        # 一些本地导出的 RLDS 数据只有 train split。训练 patch 仍然可以继续，
        # 只是验证 dataloader 退化为 None，由上游决定是否复用 train。
        if "Unknown split 'val'" not in str(exc):
            raise
        vla_dataset_val = None
        print("[dataloader][warn] validation split not found; continuing with train dataloader only.")

    # Create Collator and DataLoader
    # num_workers=0 是刻意的：RLDS/TFDS 自身已经有并行管线，PyTorch 再开 worker
    # 容易造成 CPU 线程爆炸或 fork TensorFlow 状态异常。
    collator_cls = PatchAwarePaddedCollatorForActionPrediction if return_raw_images else OfficialPaddedCollatorForActionPrediction
    collator = collator_cls(processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right")
    # DataLoader 输出的 batch 字段会在 feature_space.py 中被拆成：
    # pixel_values -> clean/patch transform；input_ids/attention_mask -> VLA language input；
    # labels -> action token supervision 和验证指标。
    train_dataloader = DataLoader(
        vla_dataset_train,
        batch_size=batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
        # shuffle=True
    )
    val_dataloader = None
    if vla_dataset_val is not None:
        val_dataloader = DataLoader(
            vla_dataset_val,
            batch_size=8,#32
            sampler=None,
            collate_fn=collator,
            num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
        )
    return train_dataloader, val_dataloader
