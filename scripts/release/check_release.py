#!/usr/bin/env python3
"""Validate the curated source release on CPU without importing robot models.

The source manifest binds published bytes. For annotation-only copies, a second
hash binds the executable AST to the original source. Historical runtime hashes
are not rewritten or treated as completed execution of the publication copy.
"""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def sha(raw):
    """Return the same byte-level SHA256 used in SOURCE_MANIFEST.json."""
    return hashlib.sha256(raw).hexdigest()


def check_link(doc, target):
    """Resolve local Markdown links; remote links are not fetched by this check."""
    if target.startswith(('http:', 'https:', '#', 'mailto:')):
        return None
    target = target.split('#')[0].strip('<>')
    if not (doc.parent / target).exists():
        return 'Broken navigation: ' + str(doc.relative_to(ROOT)) + ' -> ' + target
    return None


def main():
    """Check files, source equivalence, shell syntax and generated navigation."""
    manifest = json.loads((ROOT / 'SOURCE_MANIFEST.json').read_text())
    errors = []
    annotated = 0
    for name, row in manifest['source_files'].items():
        path = ROOT / name
        if not path.is_file():
            errors.append('Missing source: ' + name)
            continue
        raw = path.read_bytes()
        if sha(raw) != row['sha256']:
            errors.append('Changed published source: ' + name)
        if row.get('annotation_only'):
            original_ast = ast.dump(ast.parse(raw), include_attributes=False).encode()
            if sha(original_ast) != row['source_ast_sha256']:
                errors.append('Executable AST changed: ' + name)
            if not row.get('original_sha256'):
                errors.append('Original source hash missing: ' + name)
            annotated += 1
    forbidden = {'.pt', '.pth', '.safetensors', '.ckpt', '.parquet', '.hdf5', '.h5',
                 '.mp4', '.webm', '.zip', '.zst'}
    paths = [p for p in ROOT.rglob('*') if p.is_file()
             and '.git' not in p.parts and '__pycache__' not in p.parts]
    syntax = shells = 0
    for path in paths:
        if path.suffix in forbidden:
            errors.append('Generated/binary asset: ' + str(path.relative_to(ROOT)))
        if path.stat().st_size > 1_000_000:
            errors.append('File exceeds 1MB: ' + str(path.relative_to(ROOT)))
        if path.suffix == '.py':
            try:
                ast.parse(path.read_bytes(), filename=str(path.relative_to(ROOT)))
                syntax += 1
            except (SyntaxError, ValueError) as exc:
                errors.append(str(exc))
        elif path.suffix == '.sh':
            # bash -n parses only; it does not submit jobs or execute commands.
            result = subprocess.run(['bash', '-n', str(path)], capture_output=True, text=True)
            shells += 1
            if result.returncode:
                errors.append('Shell syntax: ' + str(path.relative_to(ROOT)) + ': ' + result.stderr)
    docs = [ROOT / 'README.md', ROOT / 'THIRD_PARTY_NOTICES.md', ROOT / 'results/README.md',
            ROOT / 'archive/negative_results/README.md', *list((ROOT / 'docs').rglob('*.md')),
            *list((ROOT / 'research_runs').glob('*/EXPERIMENT_DETAILS.md'))]
    for doc in docs:
        for target in re.findall(r'\]\(([^)]+)\)', doc.read_text()):
            issue = check_link(doc, target)
            if issue:
                errors.append(issue)
    catalog = json.loads((ROOT / 'docs/EXPERIMENT_CATALOG.json').read_text())
    for row in catalog:
        for name in row['config_files']:
            if name not in manifest['source_files'] or not (ROOT / name).is_file():
                errors.append('Untracked configuration source: ' + name)
    checksums = ROOT / 'release_checksums.json'
    if checksums.exists():
        for name, expected in json.loads(checksums.read_text())['sha256'].items():
            path = ROOT / name
            if not path.is_file() or sha(path.read_bytes()) != expected:
                errors.append('Release checksum mismatch: ' + name)
    result = {'status': 'PASS' if not errors else 'FAILED',
              'source_files_verified': len(manifest['source_files']),
              'annotation_ast_verified': annotated, 'runs_documented': len(catalog),
              'python_files_parsed': syntax, 'shell_files_parsed': shells,
              'files': len(paths), 'bytes': sum(p.stat().st_size for p in paths),
              'gpu_used': False, 'slurm_submitted': False, 'errors': errors}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
