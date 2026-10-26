#!/usr/bin/env python3
"""Refresh publication indexes after editing; no staging, commit, push or GPU.

The changed-file inventory excludes itself and the complete checksum index to
avoid self-referential hashes. The full index includes all other release files.
Only files selected by Git (tracked or nonignored untracked) enter either index.
"""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
EXCLUDED = {'docs/UPDATE_FILES.tsv', 'release_checksums.json'}


def sha(path):
    """Hash a small published file without importing experiment dependencies."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_files(*flags):
    """Read NUL-delimited paths so spaces cannot split a filename."""
    output = subprocess.check_output(['git', 'ls-files', *flags, '-z'], cwd=ROOT)
    return sorted({name for name in output.decode().split('\0') if name})


def main():
    """Build current changed-file and complete-file inventories, in that order."""
    changed = [name for name in git_files('-m', '-o', '--exclude-standard')
               if name not in EXCLUDED and (ROOT / name).is_file()]
    with (ROOT / 'docs/UPDATE_FILES.tsv').open('w') as stream:
        stream.write('path\tbytes\tsha256\n')
        for name in changed:
            path = ROOT / name
            stream.write(f'{name}\t{path.stat().st_size}\t{sha(path)}\n')
    files = [name for name in git_files('-c', '-o', '--exclude-standard')
             if name != 'release_checksums.json' and (ROOT / name).is_file()]
    content = {'sha256': {name: sha(ROOT / name) for name in files}}
    (ROOT / 'release_checksums.json').write_text(json.dumps(content, ensure_ascii=False, indent=2) + '\n')
    size = sum((ROOT / name).stat().st_size for name in files) + (ROOT / 'release_checksums.json').stat().st_size
    print(json.dumps({'changed_files_plus_indexes': len(changed) + 2,
                      'total_files': len(files) + 1, 'total_bytes': size,
                      'git_staged': False, 'git_pushed': False}, indent=2))


if __name__ == '__main__':
    main()
