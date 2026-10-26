"""Private in-process bindings to audited native infrastructure; no historical writes."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parents[1]
SCREEN=PROJECT/'research_runs/wrist_observation_reliability_screen_v1'
BASE=PROJECT/'research_runs/wrist_joint_checkpoint_patch_retest_v1'
sys.path.insert(1,str(SCREEN/'code'))
import sc_common as screen

# Obtain the actual historical source signature BEFORE rebinding globals.
SCREEN_SIGNATURE=screen.signature()
common=screen.common
common.signature.cache_clear()
common.ROOT=ROOT
screen.ROOT=ROOT
_dependencies=common.dependencies
def dependencies():
    return _dependencies()+list((SCREEN/'code').glob('*.py'))+[SCREEN/'config.json']
common.dependencies=dependencies
from jr_common import (config,read,sha,digest,atomic,require,signature,complete,finish,verify,
    manifest,episode_receipt,episode_files,resources,lock,Monitor,module,runtime,runtime_config,PYTHON)
native=screen.native
geometry=screen.geometry

def source_files(case):
    receipt=SCREEN/'episodes'/case['start']/'clean/COMPLETE.json'
    if not receipt.is_file():return None
    proof=read(receipt)
    require(proof['status']=='COMPLETE' and proof['signature']==SCREEN_SIGNATURE,'Historical clean source revision differs')
    files={}
    for rel,h in proof['artifacts'].items():
        p=(SCREEN/rel).resolve()
        require(p.is_relative_to(SCREEN.resolve()) and p.is_file() and sha(p)==h,'Historical clean artifact changed: '+str(p))
        # Prefer the attempt-specific start over its identical paired-start copy.
        files.setdefault(p.name,p)
    files['SOURCE_COMPLETE.json']=receipt
    return files
