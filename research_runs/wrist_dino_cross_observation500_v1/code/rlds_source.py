"""CPU-only official RLDS indexing/cache. No simulator, policy labels, or replay."""
from __future__ import annotations
import argparse
import os
import time
import numpy as np
from common import (ROOT, atomic_json, atomic_torch, config, dataset_files, digest,
                    load_json, preflight, require, run_signature, sha256_file)

def normalized_expert(actions, statistics):
    """Exactly LIBERO transform + OFT BOUNDS_Q99; inputs never mutated."""
    raw = np.asarray(actions, dtype=np.float32)
    require(raw.ndim == 2 and raw.shape[1] == 7 and np.isfinite(raw).all(), "Invalid expert action array")
    converted = raw.copy()
    converted[:, 6] = 1 - np.clip(raw[:, 6], 0, 1)
    lo, hi = np.asarray(statistics["q01"], np.float32), np.asarray(statistics["q99"], np.float32)
    mask = np.asarray(statistics.get("mask", [True] * 7), dtype=bool)
    require(lo.shape == hi.shape == mask.shape == (7,), "Action normalization shape changed")
    require(np.all(hi[:3] > lo[:3]) and mask[:3].all(), "Position bounds degenerate/unmasked")
    normalized = np.where(mask, np.clip(2 * (converted - lo) / (hi - lo + 1e-8) - 1, -1, 1), converted)
    normalized[:, np.asarray(statistics["min"]) == np.asarray(statistics["max"])] = 0
    return normalized.astype(np.float32)

def candidate_schedule(episodes, outer=2000, count=4, seed=20261011):
    lengths = np.asarray([e["eligible_windows"] for e in episodes], dtype=np.int64)
    require((lengths > 0).all(), "Episode has no complete eight-action window")
    cumulative = np.cumsum(lengths)
    numbers = np.random.default_rng(seed).integers(0, int(cumulative[-1]), size=(outer, count))
    which = np.searchsorted(cumulative, numbers, side="right")
    starts = np.r_[0, cumulative[:-1]]
    return [[f"{episodes[int(e)]['episode_id']}:{int(n - starts[int(e)])}" for e, n in zip(es, ns)]
            for es, ns in zip(which, numbers)]

def episode_iterator():
    # A CPU Slurm stage must never claim an unallocated TensorFlow GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    import tensorflow_datasets as tfds
    tf.config.set_visible_devices([], "GPU")
    options = tf.data.Options()
    options.experimental_deterministic = True
    options.threading.private_threadpool_size = 2
    options.threading.max_intra_op_parallelism = 1
    builder = tfds.builder_from_directory(config()["dataset_directory"])
    read = tfds.ReadConfig(add_tfds_id=True, interleave_cycle_length=1, interleave_block_length=1, options=options)
    return builder.as_dataset(split="train", shuffle_files=False, read_config=read,
                              decoders={"steps": tfds.decode.SkipDecoding()})

def text_value(tensor):
    raw = tensor.numpy()
    return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)

def resize_policy_jpeg(encoded):
    """Same native decode/encode/lanczos3 sequence as resize_image_for_policy."""
    import tensorflow as tf
    rgb = tf.io.decode_jpeg(encoded, channels=3)
    require(tuple(rgb.shape) == (256, 256, 3), "Official JPEG shape changed")
    rgb = tf.io.decode_image(tf.image.encode_jpeg(rgb), expand_animations=False, dtype=tf.uint8)
    value = tf.image.resize(rgb, (224, 224), method="lanczos3", antialias=True)
    return tf.cast(tf.clip_by_value(tf.round(value), 0, 255), tf.uint8).numpy().copy()

