#!/usr/bin/env python3
"""CPU-only audit of the curated release; never loads VLA or submits jobs."""
import ast
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[2]

def main():
    manifest=json.loads((ROOT/'SOURCE_MANIFEST.json').read_text())
    errors=[]
    syntax=0
    for name,row in manifest['source_files'].items():
        path=ROOT/name
        if not path.is_file():
            errors.append('Missing source: '+name)
        elif hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
            errors.append('Changed source: '+name)
    forbidden={'.pt','.pth','.safetensors','.ckpt','.parquet','.hdf5','.h5','.mp4','.webm','.zip'}
    paths=[p for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts]
    for path in paths:
        if path.suffix in forbidden:errors.append('Generated/binary asset: '+str(path.relative_to(ROOT)))
        if path.stat().st_size>1_000_000:errors.append('File exceeds 1MB: '+str(path.relative_to(ROOT)))
        if path.suffix=='.py':
            try:ast.parse(path.read_text(),filename=str(path.relative_to(ROOT)));syntax+=1
            except SyntaxError as e:errors.append(str(e))
    docs=[ROOT/'README.md',ROOT/'THIRD_PARTY_NOTICES.md',ROOT/'results/README.md',
          ROOT/'archive/negative_results/README.md',*list((ROOT/'docs').rglob('*.md'))]
    for doc in docs:
        for target in re.findall(r'\]\(([^)]+)\)',doc.read_text()):
            if target.startswith(('http:','https:','#')):continue
            target=target.split('#')[0]
            if not (doc.parent/target).exists():errors.append('Broken navigation: '+str(doc.relative_to(ROOT))+' -> '+target)
    result={'status':'PASS' if not errors else 'FAILED','source_files_verified':len(manifest['source_files']),
            'python_files_parsed':syntax,'files':len(paths),'bytes':sum(p.stat().st_size for p in paths),
            'gpu_used':False,'slurm_submitted':False,'errors':errors}
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if errors:raise SystemExit(1)

if __name__=='__main__':main()
