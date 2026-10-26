"""Read one actual official episode, compare source functions; no model/GPU."""
from __future__ import annotations
import ast
import importlib.util
import json
import os
import sys
from enum import IntEnum
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple, Union
import numpy as np
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import common
import data
sys.path.append(str(common.PROJECT))


class NormalizationType(IntEnum):
    NORMAL = 0
    BOUNDS = 1
    BOUNDS_Q99 = 2


def extract_function(path, name, namespace):
    parsed = ast.parse(Path(path).read_text())
    node = next(n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def main():
    import tensorflow as tf
    import dlimp as dl
    tf.config.set_visible_devices([], "GPU")
    settings = common.config()
    official = Path(settings["evaluation"]["paths"]["openvla_oft_root"])
    utils = official / "prismatic/vla/datasets/rlds/utils/data_utils.py"
    transforms = official / "prismatic/vla/datasets/rlds/oxe/transforms.py"
    image_utils = official / "experiments/robot/openvla_utils.py"
    namespace = {"tf": tf, "dl": dl, "np": np, "Dict": Dict, "Any": Any, "List": List,
                 "Union": Union, "Tuple": Tuple, "NormalizationType": NormalizationType,
                 "Image": Image, "OPENVLA_IMAGE_SIZE": 224}
    extract_function(utils, "invert_gripper_actions", namespace)
    transform = extract_function(transforms, "libero_dataset_transform", namespace)
    normalize = extract_function(utils, "normalize_action_and_proprio", namespace)
    resize = extract_function(image_utils, "resize_image_for_policy", namespace)
    first = next(iter(data.episode_iterator()))
    second = next(iter(data.episode_iterator()))
    identity = data.text_value(first["tfds_id"])
    common.require(identity == data.text_value(second["tfds_id"]), "Official TFDS episode ID order unstable")
    steps = first["steps"]
    length = int(steps["action"].shape[0])
    stats = common.load_json(settings["evaluation"]["checkpoints"]["libero_10"] + "/dataset_statistics.json")[settings["evaluation"]["unnorm_keys"]["libero_10"]]
    raw = steps["action"].numpy()
    adapted = data.normalized_expert(raw, stats["action"])
    trajectory = transform({"action": steps["action"], "observation": {"state": steps["observation"]["state"]}})
    trajectory["observation"]["proprio"] = tf.concat([trajectory["observation"]["EEF_state"], trajectory["observation"]["gripper_state"]], -1)
    tensor_stats = {kind: {k: tf.constant(v, dtype=tf.bool if k == "mask" else tf.float32)
                           for k, v in stats[kind].items()} for kind in ("action", "proprio")}
    normalized = normalize(trajectory, tensor_stats, NormalizationType.BOUNDS_Q99)["action"].numpy()
    max_error = float(np.abs(adapted - normalized).max())
    common.require(max_error <= 1e-6, "Expert normalization differs from official OFT source")
    errors = {}
    for view in ("image", "wrist_image"):
        encoded = steps["observation"][view][0]
        decoded = tf.io.decode_jpeg(encoded, channels=3).numpy()
        exact = resize(decoded, 224)
        cached = data.resize_policy_jpeg(encoded)
        errors[view] = int(np.abs(exact.astype(int) - cached.astype(int)).max())
        common.require(errors[view] == 0, "Official policy RGB resize mismatch")
    common.require(length >= 8 and adapted[:8].shape == (8, 7)
                   and tuple(steps["observation"]["state"].shape) == (length, 8), "Expert future-window/proprio alignment failed")
    # Actual checkpoint image processor, actual official crop, actual new EOT:
    # no VLA class or weights are instantiated here.
    import torch
    from engine import OFTEngine
    from source_transform import RandomPatchTransform
    from deployment import paste_png
    for function in ("check_image_format", "crop_and_resize", "center_crop_image", "prepare_images_for_vla"):
        extract_function(image_utils, function, namespace)
    checkpoint = Path(settings["evaluation"]["checkpoints"]["libero_10"])
    processor_source = checkpoint / "processing_prismatic.py"
    spec = importlib.util.spec_from_file_location("official_long_cpu_image_processor", processor_source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    processor = module.PrismaticImageProcessor.from_pretrained(str(checkpoint), local_files_only=True)
    official_images = SimpleNamespace(prepare_images_for_vla=namespace["prepare_images_for_vla"])
    image_cfg = SimpleNamespace(center_crop=True)
    obs = {"full_image": data.resize_policy_jpeg(steps["observation"]["image"][0]),
           "wrist_image": data.resize_policy_jpeg(steps["observation"]["wrist_image"][0])}
    clean_images = official_images.prepare_images_for_vla([obs["full_image"], obs["wrist_image"]], image_cfg)
    clean_pixels = torch.cat([processor(im, return_tensors="pt")["pixel_values"] for im in clean_images], 1).bfloat16()
    engine = object.__new__(OFTEngine)
    engine.wrapper = SimpleNamespace(device=torch.device("cpu"), dtype=torch.bfloat16,
                                     processor=SimpleNamespace(image_processor=processor),
                                     _official_openvla_utils=official_images, gen_cfg=image_cfg)
    engine.transform = RandomPatchTransform(torch.device("cpu"), resize_patch=False)
    patch = torch.rand(3, 50, 50, generator=torch.Generator().manual_seed(42)).requires_grad_()
    pasted = paste_png(obs, patch.detach().mul(255).round().byte().permute(1, 2, 0).numpy())
    expected_images = official_images.prepare_images_for_vla([pasted["full_image"], pasted["wrist_image"]], image_cfg)
    expected_pixels = torch.cat([processor(im, return_tensors="pt")["pixel_values"] for im in expected_images], 1).bfloat16()
    actual_pixels, _ = engine.pixels({"observation": obs}, patch, 20261011, diverse=False)
    gap = float((actual_pixels - expected_pixels).float().abs().max())
    common.require(gap == 0, "Fixed-position native processor differs from real PNG deployment")
    gradient_norms = {}
    for diverse in (False, True):
        values, footprint = engine.pixels({"observation": obs}, patch, 20261011, diverse=diverse)
        common.require(torch.equal(values[:, :6], clean_pixels[:, :6]), "Actual EOT changed Front")
        gradient = torch.autograd.grad(values[:, 6:9].float().sum(), patch)[0]
        common.require(torch.isfinite(gradient).all() and gradient.norm() > 0
                       and 0 < (footprint > 0).sum() < 256, "Actual RGB STE/EOT backward or support invalid")
        gradient_norms[str(diverse)] = float(gradient.norm())
    print(json.dumps({"status": "ACTUAL_OFFICIAL_RECORD_CPU_PASS", "episode_id": identity,
                      "steps": length, "expert_normalization_max_abs_error": max_error,
                      "image_resize_max_uint8_error": errors, "tfds_id_two_pass_identity": True,
                      "metadata_sha256": common.sha256_file(Path(settings["dataset_directory"]) / "dataset_info.json"),
                      "reference_source_sha256": {str(p): common.sha256_file(p) for p in (utils, transforms, image_utils, processor_source)},
                      "native_fixed_png_forward_max_error": gap, "eot_rgb_ste_gradient_l2": gradient_norms,
                      "front_unchanged_in_real_processor": True,
                      "models_loaded": False, "gpu_started": False, "slurm_submitted": False,
                      "scientific_result": "NOT_TESTED"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
