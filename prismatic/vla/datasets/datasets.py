"""
datasets.py

Lightweight PyTorch Dataset Definition for wrapping RLDS TFDS Pipeline; just defines transform from RLDS default
format to OpenVLA, IterableDataset shim.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple, Type

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import tree_map
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets.rlds import make_interleaved_dataset, make_single_dataset
from prismatic.vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights
from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


def _positive_int_env(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default

DEFAULT_VIEW_KEY_MAP = {
    "front": "image_primary",
    "primary": "image_primary",
    "third_person": "image_primary",
    "agent": "image_primary",
    "side": "image_secondary",
    "secondary": "image_secondary",
    "wrist": "image_wrist",
    "hand": "image_wrist",
}


def parse_views(value: Any) -> Tuple[str, ...]:
    """解析多视角字符串；none/空值会被过滤，避免误贴 patch。"""
    if value is None:
        return tuple()
    if isinstance(value, str):
        return tuple(view.strip() for view in value.split(",") if view.strip() and view.strip() != "none")
    return tuple(str(view).strip() for view in value if str(view).strip() and str(view).strip() != "none")


def parse_view_key_map(value: Any) -> Dict[str, str]:
    """解析 front=image_primary,wrist=image_wrist 形式的字段映射。"""
    if isinstance(value, dict):
        return {str(k).strip(): str(v).strip() for k, v in value.items() if str(k).strip() and str(v).strip()}
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


def resolve_view_key(view: str, view_key_map: Dict[str, str]) -> str:
    """把研究视角名映射到 RLDS 标准 observation key。"""
    return view_key_map.get(view, DEFAULT_VIEW_KEY_MAP.get(view, view))


def fuse_view_images(images, mode: str = "stitch") -> Image.Image:
    """把多个 PIL 视角压回单张 224 输入；当前最小实现使用水平拼接后缩放。"""
    if not images:
        raise ValueError("fuse_view_images requires at least one image")
    if len(images) == 1 or mode != "stitch":
        return images[0].convert("RGB")

    rgb_images = [img.convert("RGB") for img in images]
    width, height = rgb_images[0].size
    canvas = Image.new("RGB", (width * len(rgb_images), height))
    for idx, img in enumerate(rgb_images):
        canvas.paste(img.resize((width, height)), (idx * width, 0))
    return canvas.resize((width, height))


@dataclass
class RLDSBatchTransform:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    # image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True
    input_views: Tuple[str, ...] = ("front",)
    view_key_map: Dict[str, str] = field(default_factory=dict)
    multiview_fusion: str = "stitch"
    use_proprio: bool = False
    _missing_view_warnings: set = field(default_factory=set, init=False, repr=False)

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        dataset_name = rlds_batch["dataset_name"]
        actions = np.asarray(rlds_batch["action"], dtype=np.float32)
        action = actions[0]
        observation = rlds_batch["observation"]
        available_view_keys = sorted([key for key in observation.keys() if key.startswith("image_")])

        pixel_values_by_view = {}
        requested_views = self.input_views or ("front",)
        for view in requested_views:
            obs_key = resolve_view_key(view, self.view_key_map)
            if obs_key in observation:
                pixel_values_by_view[view] = Image.fromarray(observation[obs_key][0]).convert("RGB")
            else:
                warning_key = (view, obs_key)
                if warning_key not in self._missing_view_warnings:
                    print(
                        f"[multiview][warn] 请求视角 {view} -> {obs_key} 不在 observation 中; "
                        f"可用图像字段={available_view_keys}; 将回退到 image_primary/首个可用视角。"
                    )
                    self._missing_view_warnings.add(warning_key)

        # 无法按请求取到视角时，保留原单视角 image_primary 行为，避免破坏既有复现实验。
        if not pixel_values_by_view:
            fallback_key = "image_primary" if "image_primary" in observation else (available_view_keys[0] if available_view_keys else None)
            if fallback_key is None:
                raise KeyError(f"RLDS observation 中没有可用图像字段: {list(observation.keys())}")
            pixel_values_by_view["front"] = Image.fromarray(observation[fallback_key][0]).convert("RGB")

        selected_images = [pixel_values_by_view[view] for view in requested_views if view in pixel_values_by_view]
        if not selected_images:
            # 如果请求 wrist 但数据集中没有 wrist，则使用前面构造的回退视角，日志里已经标注待确认。
            selected_images = list(pixel_values_by_view.values())
        img = fuse_view_images(selected_images, mode=self.multiview_fusion)
        language = rlds_batch["task"]["language_instruction"]
        lang = language.decode().lower() if hasattr(language, "decode") else str(language).lower()

        # OFT 的连续 action head 训练使用当前动作 + future action chunk。
        # 如果这里仍只写入单步动作，action-token mask 只能选到 7 个 hidden states，
        # 后面的 action_head loss 会缺少 rollout 时实际使用的 8-step 开环动作监督。
        future_actions = actions[1:]
        future_actions_string = "".join(self.action_tokenizer(future_actions)) if len(future_actions) else ""
        action_chunk_string = self.action_tokenizer(action) + future_actions_string
        action_chunk_len = len(action_chunk_string)

        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": action_chunk_string},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        # pixel_values = self.image_transform(img)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        output = dict(
            pixel_values=img,
            pixel_values_by_view=pixel_values_by_view,
            input_views=list(requested_views),
            available_view_keys=available_view_keys,
            input_ids=input_ids,
            labels=labels,
            dataset_name=dataset_name,
            instructions=lang,
            # actions 是 OFT action_head loss 的监督目标，不只是日志字段。
            # 保留下来后，梯度才能从 action_head 的连续动作误差回传到视觉 patch。
            actions=actions,
        )
        if self.use_proprio and "proprio" in observation:
            # proprio 会在 OFT forward 中额外插入一个 token；训练时缺失会让
            # num_patches/hidden-state 切片与 rollout 路径不一致，梯度会被错位使用。
            output["proprio"] = np.asarray(observation["proprio"], dtype=np.float32)
        return output
        # return dict(pixel_values=img, input_ids=input_ids, labels=labels, dataset_name=lang)


class RLDSDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSBatchTransform,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        image_aug: bool = False,
        load_camera_views: Tuple[str, ...] = ("primary",),
        load_proprio: bool = False,
        future_action_window_size: int = 0,
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders."""
        self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform
        # 多视角实验按需加载 primary/wrist/secondary；默认仍只加载 primary，保持单视角兼容。
        self.load_camera_views = load_camera_views or ("primary",)

        # Configure RLDS Dataset(s)
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]

        # fmt: off
        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=self.load_camera_views,
            load_depth=False,
            load_proprio=load_proprio,
            load_language=True,
            action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
        )
        frame_parallel_calls = _positive_int_env("TF_NUM_PARALLEL_CALLS", 2)
        traj_transform_threads = _positive_int_env("TF_TRAJ_TRANSFORM_THREADS", len(mixture_spec))
        traj_read_threads = _positive_int_env("TF_TRAJ_READ_THREADS", len(mixture_spec))
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=1,                                      # If we wanted to feed / predict more than one step
                future_action_window_size=future_action_window_size, # For OFT action chunking
                skip_unlabeled=True,                                # Skip trajectories without language labels
                goal_relabeling_strategy="uniform",                 # Goals are currently unused
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                num_parallel_calls=frame_parallel_calls,          # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=traj_transform_threads,
            traj_read_threads=traj_read_threads,
            train=train,
        )

        # If applicable, enable image augmentations
        if image_aug:
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )}),
        # fmt: on

        # Initialize RLDS Dataset
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def make_dataset(self, rlds_config):
        return make_interleaved_dataset(**rlds_config)

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            yield self.batch_transform(rlds_batch)

    def __len__(self) -> int:
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")


class EpisodicRLDSDataset(RLDSDataset):
    """Returns full episodes as list of steps instead of individual transitions (useful for visualizations)."""

    def make_dataset(self, rlds_config):
        per_dataset_kwargs = rlds_config["dataset_kwargs_list"]
        assert len(per_dataset_kwargs) == 1, "Only support single-dataset `mixes` for episodic datasets."

        return make_single_dataset(
            per_dataset_kwargs[0],
            train=rlds_config["train"],
            traj_transform_kwargs=rlds_config["traj_transform_kwargs"],
            frame_transform_kwargs=rlds_config["frame_transform_kwargs"],
        )

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            out = [
                self.batch_transform(tree_map(lambda x: x[i], rlds_batch))  # noqa: B023
                for i in range(rlds_batch["action"].shape[0])
            ]
            yield out


class DummyDataset(Dataset):
    def __init__(
        self,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
    ) -> None:
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn

        # Note =>> We expect the dataset to store statistics for action de-normalization. Specifically, we store the
        # per-dimension 1st and 99th action quantile. The values below correspond to "no normalization" for simplicity.
        self.dataset_statistics = {
            "dummy_dataset": {
                "action": {"q01": np.zeros((7,), dtype=np.float32), "q99": np.ones((7,), dtype=np.float32)}
            }
        }

    def __len__(self):
        # TODO =>> Replace with number of elements in your dataset!
        return 10000

    def __getitem__(self, idx):
        # TODO =>> Load image, action and instruction from disk -- we use dummy values
        image = Image.fromarray(np.asarray(np.random.rand(224, 224, 3) * 255.0, dtype=np.uint8))
        action = np.asarray(np.random.rand(7), dtype=np.float32)
        instruction = "do something spectacular"

        # Add instruction to VLA prompt
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {instruction}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF .forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(image)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(len(action) + 1)] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
