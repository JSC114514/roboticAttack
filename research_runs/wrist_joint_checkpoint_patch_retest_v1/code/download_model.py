"""Pinned public model download only; never import a policy or simulator."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def download():
    # Avoid saturating the login node's proxy with 4 x 16 Xet range requests.
    os.environ.setdefault('HF_XET_NUM_CONCURRENT_RANGE_GETS', '2')
    from huggingface_hub import snapshot_download
    c = json.loads((ROOT / 'config.json').read_text())
    p = Path(snapshot_download(repo_id=c['repo_id'], revision=c['revision'],
        local_dir=c['model_dir'], ignore_patterns=['lora_adapter/*'], max_workers=4, token=False))
    required = ['config.json', 'dataset_statistics.json', 'model.safetensors.index.json',
                'action_head--300000_checkpoint.pt', 'proprio_projector--300000_checkpoint.pt']
    index = json.loads((p / 'model.safetensors.index.json').read_text())
    required += sorted(set(index['weight_map'].values()))
    artifacts = {}
    for name in required:
        q = p / name
        if not q.is_file(): raise RuntimeError('Missing downloaded component: ' + name)
        h = hashlib.sha256()
        with q.open('rb') as f:
            for b in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(b)
        metadata = (p / '.cache/huggingface/download' / (name + '.metadata')).read_text().splitlines()
        if metadata[0] != c['revision']: raise RuntimeError('Snapshot differs: ' + name)
        if len(metadata[1]) == 64 and h.hexdigest() != metadata[1]: raise RuntimeError('Official LFS checksum mismatch: ' + name)
        artifacts[name] = {'sha256': h.hexdigest(), 'bytes': q.stat().st_size}
    value = {'repo_id': c['repo_id'], 'revision': c['revision'], 'status': 'COMPLETE', 'files': artifacts}
    temp = p / ('.JOINT_DOWNLOAD.' + str(os.getpid()) + '.json')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(p / 'JOINT_DOWNLOAD.json')
    print('MODEL_DOWNLOAD_COMPLETE', p, flush=True)


if __name__ == '__main__': download()
