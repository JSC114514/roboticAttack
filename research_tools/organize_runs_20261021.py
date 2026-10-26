"""Execute the reviewed, recoverable archive plan. No model imports or Slurm mutation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

PROJECT = Path(__file__).resolve().parents[1]
RUNS = PROJECT / 'research_runs'
AUDIT = PROJECT / 'research_maintenance/2026-10-21'
ARCHIVE = PROJECT / 'research_archive/2026-10-21'
RESULTS = PROJECT / 'research_results'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(path)


def source_files(root):
    records = {}
    for base, dirs, files in os.walk(root, followlinks=False):
        for name in list(dirs):
            path = Path(base) / name
            if path.is_symlink():
                records[path.relative_to(RUNS).as_posix()] = {'link': os.readlink(path)}
                dirs.remove(name)
        for name in files:
            path = Path(base) / name
            if path.suffix.lower() in ('.db', '.sqlite', '.sqlite3'):
                raise RuntimeError('Database excluded from cleanup: ' + str(path))
            st = path.lstat()
            record = {'link': os.readlink(path)} if path.is_symlink() else {
                'sha256': sha(path), 'bytes': st.st_size, 'mtime_ns': st.st_mtime_ns}
            records[path.relative_to(RUNS).as_posix()] = record
    return records


def verify_tar(archive, expected):
    actual = {}
    process = subprocess.Popen(['zstd', '-q', '-d', '-c', str(archive)], stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=process.stdout, mode='r|') as stream:
            for entry in stream:
                name = entry.name.rstrip('/')
                assert not Path(name).is_absolute() and '..' not in Path(name).parts
                if entry.isdir():
                    continue
                if entry.issym():
                    actual[name] = {'link': entry.linkname}
                elif entry.islnk():
                    actual[name] = dict(actual[entry.linkname])
                elif entry.isfile():
                    h = hashlib.sha256()
                    with stream.extractfile(entry) as data:
                        for block in iter(lambda: data.read(4 * 1024 * 1024), b''):
                            h.update(block)
                    actual[name] = {'sha256': h.hexdigest(), 'bytes': entry.size}
                else:
                    raise RuntimeError('Unsupported archive member: ' + name)
    finally:
        process.stdout.close()
    assert process.wait() == 0, 'Archive decompression failed'
    normalized = {p: {k: v for k, v in value.items() if k != 'mtime_ns'} for p, value in expected.items()}
    assert actual == normalized, 'Archive content does not exactly match original files'


def snapshot_results(inventory, active):
    saved = []
    for name in sorted(inventory['runs']):
        if name in active:
            continue  # Running output is linked in the index, never copied as final evidence.
        root = RUNS / name
        candidates = [root / n for n in ('README.md', 'PLAN.md', 'config.json')]
        for selection in (root / 'data/SELECTED.json', root / 'data/SELECTION.json'):
            if selection.is_file():
                candidates.append(selection)
        # Preserve small PNG/JSON checkpoints for convenient inspection; their
        # original names/receipts identify whether they are final or partial.
        candidates += list((root / 'training').glob('*/*.png'))
        candidates += list((root / 'training').glob('*/*.json'))
        if (root / 'results').is_dir():
            candidates += list((root / 'results').rglob('*'))
        copied = 0
        for path in sorted(set(candidates)):
            if not path.is_file() or path.is_symlink() or path.suffix.lower() not in (
                    '.md', '.json', '.csv', '.parquet', '.png', '.pdf', '.txt'):
                continue
            st = path.stat()
            if st.st_size > 20 * 1024**2 or copied + st.st_size > 64 * 1024**2:
                continue  # Full originals remain in their original run or verified archive.
            digest = sha(path)
            destination = RESULTS / name / path.relative_to(root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            assert sha(destination) == digest
            assert path.stat().st_mtime_ns == st.st_mtime_ns
            saved.append({'source': str(path.relative_to(PROJECT)),
                          'copy': str(destination.relative_to(PROJECT)), 'sha256': digest,
                          'bytes': st.st_size, 'source_mtime_ns': st.st_mtime_ns})
            copied += st.st_size
    write(RESULTS / 'SNAPSHOT_MANIFEST.json', {
        'created_unix': time.time(), 'active_runs_excluded': sorted(active), 'files': saved,
        'scope': 'Small result tables/reports/figures plus plan/config; snapshot, not new evaluation. '
                 'Includes small PNG/JSON training checkpoints, without selecting a new best checkpoint. '
                 'Per file <=20MiB and per run <=64MiB. Full originals retained live or in archive.'})
    return saved


def archive_run(name):
    root = RUNS / name
    assert root.parent == RUNS and root.is_dir() and not root.is_symlink()
    destination = ARCHIVE / (name + '.tar.zst')
    receipt = ARCHIVE / (name + '.json')
    assert not destination.exists() and not receipt.exists(), 'Existing archive requires manual inspection'
    expected = source_files(root)
    temporary = destination.with_suffix(destination.suffix + '.partial')
    subprocess.run(['tar', '--format=pax', '--use-compress-program=zstd -q -T1 -1', '-cf', str(temporary),
                    '-C', str(RUNS), name], check=True)
    verify_tar(temporary, expected)
    # Never delete a source which changed while the archive was being written.
    assert source_files(root) == expected, 'Original run changed during archiving'
    temporary.replace(destination)
    value = {'run': name, 'original_path': str(root), 'archive': str(destination),
             'archive_sha256': sha(destination), 'archive_bytes': destination.stat().st_size,
             'source_bytes': sum(v.get('bytes', 0) for v in expected.values()),
             'files': expected, 'all_archive_files_verified': True, 'created_unix': time.time(),
             'original_removed': False}
    write(receipt, value)  # Full recovery evidence is durable before removing the source.
    shutil.rmtree(root)
    value['original_removed'] = True
    write(receipt, value)
    print('ARCHIVED', name, 'files', len(expected), 'bytes', value['archive_bytes'], flush=True)
    return value


def restore(name):
    assert Path(name).name == name
    receipt = ARCHIVE / (name + '.json')
    value = json.loads(receipt.read_text())
    archive = Path(value['archive'])
    assert archive.parent == ARCHIVE and sha(archive) == value['archive_sha256']
    assert not (RUNS / name).exists(), 'Refuse to overwrite an existing run'
    verify_tar(archive, value['files'])
    subprocess.run(['tar', '--use-compress-program=zstd -q -d', '-xf', str(archive),
                    '-C', str(RUNS)], check=True)
    restored = source_files(RUNS / name)
    # GNU tar stores subsecond timestamps separately depending on its format.
    fields = lambda rows: {p: {k: v for k, v in v.items() if k != 'mtime_ns'} for p, v in rows.items()}
    assert fields(restored) == fields(value['files'])
    print('RESTORED', name)


def main():
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--execute', action='store_true')
    group.add_argument('--restore')
    args = parser.parse_args()
    if args.restore:
        restore(args.restore)
        return
    plan = json.loads((AUDIT / 'PLAN.json').read_text())
    inventory = json.loads((AUDIT / 'INVENTORY.json').read_text())
    queue = json.loads((AUDIT / 'QUEUE_BEFORE.json').read_text())
    assert time.time() - queue['checked_unix'] < 600, 'Refresh queue evidence before cleanup'
    assert set(plan['archive_runs']).isdisjoint(plan['protected_active_dependencies'])
    assert set(queue['active_run_roots']).issubset(set(plan['protected_active_dependencies']))
    for name in plan['archive_runs']:
        assert not any(r['to'] == name and r['kind'] == 'runtime_or_metadata'
                       for r in inventory['references']), 'Archive candidate has runtime references'
    saved = snapshot_results(inventory, set(plan['active_runs']))
    write(AUDIT / 'RESULT_SNAPSHOT.json', {'files': len(saved), 'bytes': sum(v['bytes'] for v in saved)})
    print('RESULT SNAPSHOT', len(saved), 'files', flush=True)
    archives = []
    for name in plan['archive_runs']:
        archives.append(archive_run(name))
        write(AUDIT / 'ARCHIVED.json', {'runs': archives, 'complete': False})
    write(AUDIT / 'ARCHIVED.json', {'runs': archives, 'complete': True})


if __name__ == '__main__':
    main()
