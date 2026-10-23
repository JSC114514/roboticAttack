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

def prepare():
    import torch
    cfg, signature, started = config(), run_signature(), time.monotonic()
    report = preflight()
    preparation = ROOT / "reports/PREPARATION.json"
    if preparation.exists():
        require(load_json(preparation)["signature"] == signature,
                "Existing preparation identity changed; do not overwrite old provenance")
    atomic_json(report, preparation)
    final = ROOT / "data/CANDIDATE_BANK.pt"
    done = ROOT / "data/DATA_STATUS.json"
    if done.exists():
        old = load_json(done)
        require(old["signature"] == signature, "Existing official-data preparation identity changed")
        if old["status"] == "COMPLETE":
            require(sha256_file(final) == old["bank_sha256"], "Completed cache changed")
            print("OFFICIAL_DATA_ALREADY_COMPLETE", flush=True)
            return
    index_path = ROOT / "data/DATASET_INDEX.json"
    if index_path.exists():
        index = load_json(index_path)
        require(index["signature"] == signature, "Official dataset index identity changed")
        require(all(sha256_file(p) == h for p, h in index["shard_sha256"].items()), "Official shard bytes changed")
    else:
        episodes = []
        for number, episode in enumerate(episode_iterator()):
            steps = episode["steps"]
            length = int(steps["action"].shape[0])
            require(length >= 8 and tuple(steps["observation"]["state"].shape) == (length, 8), "RLDS episode/action/state shape changed")
            language = text_value(steps["language_instruction"][0])
            require(language.strip(), "Empty official instruction")
            episodes.append({"episode_id": text_value(episode["tfds_id"]), "length": length,
                             "eligible_windows": length - 7, "language": language,
                             "file_path": text_value(episode["episode_metadata"]["file_path"])})
            if (number + 1) % 50 == 0:
                print(f"OFFICIAL_INDEX {number + 1}/{cfg['dataset_episodes']} elapsed_min={(time.monotonic()-started)/60:.1f}", flush=True)
        require(len(episodes) == cfg["dataset_episodes"] and len({e["episode_id"] for e in episodes}) == cfg["dataset_episodes"], "Official source episode coverage failed")
        require(len({e["language"] for e in episodes}) == 10, "Official source ten-task language coverage failed")
        index = {"signature": signature, "source": "official_train_RLDS", "episodes": episodes,
                 "valid_action_windows": sum(e["eligible_windows"] for e in episodes),
                 "shard_sha256": {str(p.resolve()): sha256_file(p) for p in dataset_files()}}
        atomic_json(index, index_path)
    candidates = candidate_schedule(index["episodes"], cfg["bank_window_count"], cfg["candidates_per_outer"], cfg["data_seed"])
    center_uids = [row[0] for row in candidate_schedule(index["episodes"], cfg["center_reference_count"], 1, cfg["data_seed"] + 1)]
    episode_map = {e["episode_id"]: e for e in index["episodes"]}
    language_ids = {lang: i for i, lang in enumerate(sorted({e["language"] for e in index["episodes"]}))}
    requested = {}
    for uid in ({uid for group in candidates for uid in group} | set(center_uids)):
        episode_id, step = uid.rsplit(":", 1)
        requested.setdefault(episode_id, set()).add(int(step))
    statistics = load_json(cfg["evaluation"]["checkpoints"][cfg["training_suite"]] + "/dataset_statistics.json")[cfg["evaluation"]["unnorm_keys"][cfg["training_suite"]]]["action"]
    partial = ROOT / "data/CANDIDATE_BANK.partial.pt"
    saved = torch.load(partial, map_location="cpu", weights_only=False) if partial.exists() else None
    if saved:
        require(saved["signature"] == signature and saved["candidates"] == candidates
                and saved["center_reference_uids"] == center_uids, "Partial cache identity changed")
    records = {} if saved is None else saved["records"]
    for number, episode in enumerate(episode_iterator()):
        episode_id = text_value(episode["tfds_id"])
        if episode_id not in requested:
            continue
        steps, meta = episode["steps"], episode_map[episode_id]
        require(int(steps["action"].shape[0]) == meta["length"], "Second RLDS pass episode drift")
        actions = normalized_expert(steps["action"].numpy(), statistics)
        for step in sorted(requested[episode_id]):
            uid = f"{episode_id}:{step}"
            if uid in records:
                continue
            require(step + 8 <= meta["length"], "Future action window crosses episode")
            state = steps["observation"]["state"][step].numpy().astype(np.float32)
            require(state.shape == (8,) and np.isfinite(state).all(), "Invalid expert proprio")
            records[uid] = {"uid": uid, "episode_id": episode_id, "step_index": step,
                            "task_id": language_ids[meta["language"]], "task_name": meta["language"],
                            "task_id_definition": "sorted official language, not simulator benchmark order",
                            "source_suite": cfg["training_suite"], "source_split": "train", "file_path": meta["file_path"],
                            "expert_actions": torch.from_numpy(actions[step:step + 8].copy()).unsqueeze(0),
                            "observation": {"full_image": resize_policy_jpeg(steps["observation"]["image"][step]),
                                            "wrist_image": resize_policy_jpeg(steps["observation"]["wrist_image"][step]),
                                            "state": state}}
        if (number + 1) % 50 == 0:
            atomic_torch({"signature": signature, "candidates": candidates, "center_reference_uids": center_uids, "records": records}, partial)
            print(f"OFFICIAL_CACHE episodes={number + 1}/{cfg['dataset_episodes']} windows={len(records)}/{sum(map(len, requested.values()))} elapsed_min={(time.monotonic()-started)/60:.1f}", flush=True)
    expected = {uid for group in candidates for uid in group} | set(center_uids)
    require(set(records) == expected, "Official candidate cache incomplete")
    require({r["task_id"] for r in records.values()} == set(range(10)), "Budget sampled windows missed a source task")
    bank = {"signature": signature, "candidates": candidates, "center_reference_uids": center_uids, "records": records,
            "index_sha256": sha256_file(index_path), "statistics": statistics,
            "source": "official Spatial RLDS expert demonstrations; no policy actions/replay"}
    atomic_torch(bank, final)
    atomic_json({"status": "COMPLETE", "signature": signature, "bank_sha256": sha256_file(final),
                 "index_sha256": bank["index_sha256"], "official_episodes": cfg["dataset_episodes"],
                 "eligible_windows": index["valid_action_windows"], "cached_unique_windows": len(records),
                 "cached_source_episodes": len({r["episode_id"] for r in records.values()}),
                 "candidate_draws": cfg["bank_window_count"] * cfg["candidates_per_outer"], "center_reference_draws": len(center_uids),
                 "effective_batch_size": cfg["batch_size"], "expert_future_actions": True,
                 "elapsed_s": time.monotonic() - started, "models_loaded": False}, done)
    print(f"OFFICIAL_DATA_COMPLETE {len(records)} cached windows; {cfg['dataset_episodes']}-episode sampling pool; no model labels", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true", required=True)
    parser.parse_args()
    prepare()
