# 发布注释：以任务→轨迹为重采样单位；同一轨迹的帧、动作槽位不是独立样本。
# 运行来源与发布哈希分别记录在SOURCE_MANIFEST.json；历史运行目录不改动。
"""CPU-only paired task response report; missing branches remain missing."""
from __future__ import annotations

from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np


def branch_rows(manifest, results, models):
    rows = []
    for anchor in manifest.get('anchors', []):
        for model in models:
            for condition in anchor.get('conditions', []):
                result = results.get((model, anchor['episode_id'], condition['name']))
                valid = bool(condition['valid'])
                row = dict(model=model, episode_id=anchor['episode_id'], suite=anchor['suite'],
                           task_id=anchor['task_id'], state_index=anchor.get('state_index'), condition=condition['name'],
                           target_shift_m=condition['target_shift_m'], camera_shift_m=condition['camera_shift_m'],
                           status='INVALID' if not valid else 'COMPLETE' if result is not None else 'MISSING',
                           valid=valid, invalid_reasons=json.dumps(condition.get('reasons', [])))
                for field in ('initial_distance_m', 'final_distance_m', 'approach_m', 'eef_displacement_m',
                              'executed_steps', 'first_contact_step', 'first_grasp_step', 'success_flag'):
                    value = result.get(field) if result is not None and valid else None
                    row[field] = json.dumps(value) if isinstance(value, (list, dict)) else value
                row['contact_observed'] = None if result is None or not valid else int(result.get('first_contact_step') is not None)
                row['grasp_observed'] = None if result is None or not valid else int(result.get('first_grasp_step') is not None)
                rows.append(row)
    return rows


def write_branches(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ['model', 'episode_id', 'suite', 'task_id', 'condition', 'status', 'approach_m', 'invalid_reasons']
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def absolute_by_task(rows):
    groups = defaultdict(list)
    for row in rows: groups[(row['model'], row['suite'], row['task_id'])].append(row)
    out = {}
    for (model, suite, task), values in groups.items():
        done = [r for r in values if r['status'] == 'COMPLETE']
        approaches = [r['approach_m'] for r in done if r['approach_m'] is not None]
        out[f'{model}/{suite}/t{task}'] = dict(expected=len(values), completed=len(done),
            invalid=sum(r['status'] == 'INVALID' for r in values), missing=sum(r['status'] == 'MISSING' for r in values),
            contact_count=sum(r['contact_observed'] or 0 for r in done), grasp_count=sum(r['grasp_observed'] or 0 for r in done),
            contact_grasp_denominator=len(done), approach_denominator=len(approaches),
            mean_absolute_approach_m=float(np.mean(approaches)) if approaches else None)
    return out


def distance_figures(results, root, models):
    if not results: return dict(files=[], errors=[])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    paths, errors = [], []
    for episode in sorted({key[1] for key in results}):
        fig, axes = plt.subplots(1, len(models), figsize=(5*len(models), 4), sharey=True, squeeze=False)
        for axis, model in zip(axes[0], models):
            for (m, e, condition), result in sorted(results.items()):
                if (m, e) != (model, episode): continue
                branch = root/'evaluation'/m/e/condition
                try:
                    receipt = json.loads((branch/'COMPLETE.json').read_text())
                    step_path = next(root/p for p in receipt['artifacts'] if Path(p).name == 'STEPS.parquet')
                    steps = pd.read_parquet(step_path).sort_values('step')
                    x = [0]+steps['step'].tolist()
                    y = [result['initial_distance_m']]+steps['distance_m'].tolist()
                    axis.plot(x, y, marker='.', linewidth=1, label=condition)
                except (OSError, ValueError, KeyError, StopIteration) as error:
                    errors.append(dict(model=m, episode_id=e, condition=condition, error=str(error)))
            axis.set_title(model); axis.set_xlabel('Executed control step'); axis.grid(alpha=.2)
            axis.set_xticks(range(9)); axis.legend(fontsize=5, loc='best')
        axes[0, 0].set_ylabel('Actual end-effector to target distance (m)')
        fig.suptitle(episode); fig.tight_layout()
        folder = root/'reports/distance_curves'; folder.mkdir(parents=True, exist_ok=True)
        for suffix in ('svg', 'png'):
            path = folder/(episode+'.'+suffix); fig.savefig(path, dpi=160); paths.append(str(path.relative_to(root)))
        plt.close(fig)
    return dict(files=paths, errors=errors)


def _estimate(rows, draws=2000):
    """Equal suite/task means; resample tasks then trajectories within suites."""
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[row['suite']][row['task_id']].append(row['value'])
    if not groups:
        return dict(n_trajectories=0, task_denominators={}, mean=None, ci95=None)
    arrays = {s: {t: np.asarray(v, dtype=float) for t, v in ts.items()} for s, ts in groups.items()}
    mean = np.mean([np.mean([v.mean(axis=0) for v in ts.values()], axis=0) for ts in arrays.values()], axis=0)
    rng = np.random.default_rng(42)
    boot = []
    for _ in range(draws):
        suites = []
        for ts in arrays.values():
            tasks = list(ts.values())
            suites.append(np.mean([tasks[i][rng.integers(len(tasks[i]), size=len(tasks[i]))].mean(axis=0)
                                   for i in rng.integers(len(tasks), size=len(tasks))], axis=0))
        boot.append(np.mean(suites, axis=0))
    return dict(n_trajectories=len(rows), task_denominators={f'{s}/t{t}': len(v) for s, ts in arrays.items() for t, v in ts.items()},
                mean=np.asarray(mean).tolist(), ci95=np.quantile(boot, [.025, .975], axis=0).tolist())


def aggregate(manifest, results, models, draws=2000):
    """results maps (model, episode, condition) to completed RESULT objects."""
    invalid_results = [dict(model=m, episode_id=e, condition=c, reason='executed_steps_must_equal_8',
                            executed_steps=result.get('executed_steps'))
                       for (m, e, c), result in results.items() if result.get('executed_steps') != 8]
    results = {key: result for key, result in results.items() if result.get('executed_steps') == 8}
    metrics = defaultdict(list)
    counts = {m: dict(expected=0, invalid=0, missing=0, completed=0) for m in models}
    for anchor in manifest.get('anchors', []):
        episode = anchor['episode_id']
        conditions = anchor.get('conditions', [])
        by_shift = {(float(c['target_shift_m']), float(c['camera_shift_m'])): c for c in conditions}
        for m in models:
            for c in conditions:
                counts[m]['expected'] += 1
                if not c['valid']:
                    counts[m]['invalid'] += 1
                elif (m, episode, c['name']) not in results:
                    counts[m]['missing'] += 1
                else:
                    counts[m]['completed'] += 1
        for (target, camera), c in by_shift.items():
            if not c['valid']:
                continue
            # Every contrast requires the same pair for all three models.
            for kind, control in [('physical', by_shift.get((0., camera))), ('camera', by_shift.get((target, 0.)))]:
                if (kind == 'physical' and target == 0) or (kind == 'camera' and camera == 0):
                    continue
                if not control or not control['valid']:
                    continue
                if not all((m, episode, n['name']) in results for m in models for n in (c, control)):
                    continue
                delta = {}
                for m in models:
                    a, b = (results[m, episode, n['name']] for n in (c, control))
                    delta[m] = dict(approach_m=float(a['approach_m']) - float(b['approach_m']))
                    for field in ('eef_displacement_m', 'actions', 'normalized_actions'):
                        if field in a and field in b:
                            av, bv = np.asarray(a[field], dtype=float), np.asarray(b[field], dtype=float)
                            delta[m][field] = av - bv
                            if field.endswith('actions'):
                                delta[m][field+'_rms'] = float(np.sqrt(np.mean((av-bv)**2)))
                for m in models:
                    for field, value in delta[m].items():
                        key = f'{kind}/{c["name"]}/{m}/{field}/response'
                        row = dict(suite=anchor['suite'], task_id=anchor['task_id'], episode_id=episode, value=np.asarray(value).tolist())
                        metrics[key].append(row)
                        if m != 'base' and field in delta['base']:
                            key = f'{kind}/{c["name"]}/{m}/{field}/interaction_vs_base'
                            metrics[key].append({**row, 'value': np.asarray(value - delta['base'][field]).tolist()})
    return dict(status='COMPLETE' if manifest.get('status') == 'COMPLETE' and all(not c['missing'] for c in counts.values()) else 'INCOMPLETE',
                counts=counts, invalid_results=invalid_results,
                no_valid_anchor=[a['episode_id'] for a in manifest.get('anchors', []) if a.get('selected') is None],
                estimates={k: _estimate(v, draws) for k, v in sorted(metrics.items())}, paired_values=dict(metrics),
                estimand='approach = initial minus final actual simulator target distance; physical interaction = (model(t,c)-model(0,c))-(base(t,c)-base(0,c))',
                weighting='equal suites, equal tasks within suite, equal trajectories within task',
                bootstrap=dict(seed=42, draws=draws, hierarchy='task then trajectory within fixed suites; conditions stay paired'),
                short_task_flag_is_full_rollout_success=False)


def _video_metadata(path, ffmpeg):
    import imageio_ffmpeg
    # Decode every packet with ffmpeg, then independently count yielded frames.
    check = subprocess.run([ffmpeg, '-v', 'error', '-xerror', '-nostdin', '-threads', '2', '-i', str(path),
                            '-threads', '2', '-f', 'null', '-'], capture_output=True, text=True, timeout=120)
    if check.returncode: raise RuntimeError('Full video decode failed: '+check.stderr[-1000:])
    reader = imageio_ffmpeg.read_frames(str(path), input_params=['-threads', '2'], output_params=['-threads', '2'])
    try:
        info = next(reader)
        info['decoded_frames'] = sum(1 for _ in reader)
    finally: reader.close()
    if info['decoded_frames'] < 1: raise RuntimeError('Video contains no decoded frames')
    return info


def publish_videos(results, root, project):
    from diag_common import sha, atomic
    destination = project/'research_videos/wrist_camera_task_response_diag_v1_singleview'
    exports, errors = [], []
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except (ImportError, RuntimeError) as error:
            return dict(status='FAILED', exports=[], errors=['ffmpeg unavailable: '+str(error)])
    for (model, episode, condition), result in sorted(results.items()):
        branch = root/'evaluation'/model/episode/condition
        for view in ('front', 'wrist'):
            source = branch/(view+'.mp4')
            target = destination/model/condition/view/(episode+f'_short_done_{int(bool(result.get("success_flag")))}.mp4')
            webm = target.parent/'IDE_WebM'/(target.stem+'.webm')
            proof = webm.with_suffix('.json')
            try:
                if not source.exists(): raise RuntimeError('Missing source video')
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() and not os.path.samefile(source, target): target.unlink()
                if not target.exists():
                    try: os.link(source, target)
                    except OSError: shutil.copy2(source, target)
                webm.parent.mkdir(exist_ok=True)
                source_hash = sha(source)
                previous = json.loads(proof.read_text()) if proof.exists() else {}
                fresh = previous.get('source_sha256') == source_hash and webm.exists() and previous.get('webm_sha256') == sha(webm)
                if not fresh:
                    temp = webm.with_suffix('.partial.webm')
                    process = subprocess.run([ffmpeg, '-v', 'error', '-xerror', '-nostdin', '-y', '-threads', '2', '-i', str(target),
                        '-an', '-c:v', 'libvpx', '-deadline', 'realtime', '-cpu-used', '6', '-threads', '2', '-b:v', '500k',
                        '-pix_fmt', 'yuv420p', str(temp)], capture_output=True, text=True, timeout=120)
                    if process.returncode: raise RuntimeError(process.stderr[-1000:])
                    temp.replace(webm)
                a, b = _video_metadata(target, ffmpeg), _video_metadata(webm, ffmpeg)
                if a['size'] != b['size'] or tuple(a['size']) != (224, 224): raise RuntimeError('Video dimensions differ from native 224')
                if abs(a['fps']-b['fps']) > 1e-6 or abs(a['fps']-result['fps']) > 1e-3: raise RuntimeError('Video fps mismatch')
                if abs(a['duration']-b['duration']) > 1/a['fps']+.001: raise RuntimeError('Video duration mismatch')
                expected = int(result['executed_steps'])+1
                if a['decoded_frames'] != expected or b['decoded_frames'] != expected: raise RuntimeError('Video frame count differs from steps+1')
                if abs(a['duration']-expected/result['fps']) > 1/a['fps']+.001: raise RuntimeError('Video duration differs from steps/fps')
                record = dict(mp4=str(target), webm=str(webm), source_sha256=source_hash, mp4_sha256=sha(target), webm_sha256=sha(webm),
                    format_checks='both_full_decode_same_native224_dimensions_fps_duration_frames', mp4_metadata=a, webm_metadata=b)
                atomic(record, proof); exports.append(record)
            except (OSError, ValueError, KeyError, RuntimeError, subprocess.TimeoutExpired) as error:
                errors.append(dict(source=str(source), webm=str(webm), error=str(error)))
    return dict(status='COMPLETE' if not errors else 'INCOMPLETE', exports=exports, errors=errors)


def report(monitor=None):
    from diag_common import ROOT, PROJECT, config, read, atomic, complete, signature, finish
    manifest_path = ROOT/'data/MANIFEST.json'
    manifest = read(manifest_path) if manifest_path.exists() else dict(status='MISSING', anchors=[])
    if manifest.get('signature') != signature():
        manifest = dict(status='STALE_OR_MISSING', anchors=[])
    results, invalid_receipts = {}, []
    models = config()['models']
    for anchor in manifest.get('anchors', []):
        for model in models:
            for c in anchor.get('conditions', []):
                branch = ROOT/'evaluation'/model/anchor['episode_id']/c['name']
                try:
                    if c['valid'] and complete(branch/'COMPLETE.json'):
                        results[model, anchor['episode_id'], c['name']] = read(branch/'RESULT.json')
                except (RuntimeError, ValueError, OSError) as error:
                    invalid_receipts.append(dict(path=str(branch), error=str(error)))
    summary = aggregate(manifest, results, models)
    results = {key: result for key, result in results.items() if result.get('executed_steps') == 8}
    rows = branch_rows(manifest, results, models)
    csv_path = ROOT/'reports/BRANCHES.csv'
    write_branches(csv_path, rows)
    summary.update(signature=signature(), manifest_status=manifest.get('status'), invalid_receipts=invalid_receipts,
                   absolute_by_task=absolute_by_task(rows), distance_figures=distance_figures(results, ROOT, models),
                   video_exports=publish_videos(results, ROOT, PROJECT))
    path = ROOT/'reports/SUMMARY.json'
    atomic(summary, path)
    lines = ['# Wrist camera task response diagnostic', '', f'Status: **{summary["status"]}**', '',
             'Single eight-step chunks. Task flags describe short chunks; no full-rollout success rate or CASR is estimated.', '',
             '| Model | Completed | Invalid | Missing |', '|---|---:|---:|---:|']
    for model, count in summary['counts'].items():
        lines.append(f'| {model} | {count["completed"]} | {count["invalid"]} | {count["missing"]} |')
    lines += ['', 'Physical response is approach to the actual simulator target. Interactions subtract the paired base response.',
              'Absolute branch distances, approach, displacement, contact/grasp flags and invalid reasons: [BRANCHES.csv](BRANCHES.csv).',
              f'Video export status: {summary["video_exports"]["status"]}. Export errors are separate from experiment completeness.',
              'Each estimate uses complete condition pairs in every model. Signs are reported separately. Task denominators and task→trajectory bootstrap intervals are in SUMMARY.json.', '',
              '| Physical approach interaction | Mean (m) | 95% interval (m) | Paired trajectories |', '|---|---:|---|---:|']
    for key, est in summary['estimates'].items():
        if key.startswith('physical/') and key.endswith('/approach_m/interaction_vs_base'):
            lines.append(f'| {key} | {est["mean"]:.6f} | {est["ci95"]} | {est["n_trajectories"]} |')
    markdown = ROOT/'reports/REPORT.md'
    lines += ['', '| Model/task | Completed / expected | Invalid | Missing | Contact / denominator | Grasp / denominator |',
              '|---|---:|---:|---:|---:|---:|']
    for key, value in summary['absolute_by_task'].items():
        lines.append(f'| {key} | {value["completed"]}/{value["expected"]} | {value["invalid"]} | {value["missing"]} | {value["contact_count"]}/{value["contact_grasp_denominator"]} | {value["grasp_count"]}/{value["contact_grasp_denominator"]} |')
    for curve in summary['distance_figures']['files']:
        if curve.endswith('.png'): lines.extend(['', f'![Actual target distance]({Path(curve).relative_to("reports")})'])
    markdown.write_text('\n'.join(lines)+'\n')
    if summary['status'] == 'COMPLETE': finish(ROOT/'reports/REPORT_COMPLETE.json', [path, markdown, csv_path])
    if monitor: monitor.update(completed=len(results), report_status=summary['status'])
    return summary
