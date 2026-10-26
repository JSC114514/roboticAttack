"""Reuse native infrastructure in this process; historical directories stay read-only."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
RESCUE = PROJECT / 'research_runs/wrist_joint_feature_rescue_v1'
sys.path.insert(1, str(RESCUE / 'code'))
import fr_common as rescue

SCREEN = rescue.SCREEN
SCREEN_SIGNATURE = rescue.SCREEN_SIGNATURE
common = rescue.common
common.signature.cache_clear()
common.ROOT = ROOT
rescue.ROOT = ROOT
rescue.screen.ROOT = ROOT
_dependencies = common.dependencies

def dependencies():
    return _dependencies() + [RESCUE / 'code' / n for n in ('fr_common.py', 'fr_vision.py', 'fr_math.py')]

common.dependencies = dependencies
from jr_common import (config, read, sha, atomic, require, signature, verify, complete,
                       finish, episode_receipt, episode_files, resources, lock, Monitor, module,
                       runtime, runtime_config, PYTHON)
native = rescue.native
geometry = rescue.geometry
source_files = rescue.source_files


def reference_files(case):
    return {k: Path(v) for k, v in case['reference_files'].items()}


def eligible(case):
    return case['eligibility'] == 'ELIGIBLE'


def expected_receipts(manifest):
    return [episode_receipt(case, row['name']) for case in manifest['cases'] if eligible(case)
            for row in manifest['conditions']]
