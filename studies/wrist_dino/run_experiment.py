#!/usr/bin/env python3
"""Direct scientific entry points, without a cluster scheduler.

The publication tree contains source/configuration snapshots and curated results.
An explicit --run-root selects a separately prepared writable runtime workspace;
model weights, candidate banks and genuine parent-stage receipts live there.
This entry changes only file lookup, never loss, EOT, optimizer or budget.
Use the target-native Python environment: OFT/Adapter use Torch, pi05 uses JAX.
"""
import argparse
import importlib
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOTS = {
    'main': HERE,
    'factorial': HERE / 'supplements/wrist_dino_factorial_v1',
    'sensitivity': HERE / 'supplements/wrist_dino_sensitivity_v1',
    'crosscheck': HERE / 'supplements/wrist_dino_crosscheck_v1',
}


def main():
    """Dispatch a single operation; never submit Slurm or select GPU resources."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('family', choices=ROOTS)
    parser.add_argument('operation', choices=[
        'build-bank', 'preflight', 'train', 'clean-eval', 'eval',
        'collect-replay', 'diagnose', 'score', 'analyze', 'report', 'audit'])
    parser.add_argument('--run-root', type=Path, required=True,
                        help='Prepared writable workspace; not the published results directory')
    parser.add_argument('--id', default='B01')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--scope', default='development',
                        choices=['development', 'full', 'confirmation_extra', 'holdout'])
    parser.add_argument('--stop', type=int, default=500)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--model', choices=['vla_adapter_pro', 'pi05'])
    parser.add_argument('--suite', choices=['libero_spatial', 'libero_object', 'libero_goal', 'libero_10'])
    parser.add_argument('--mode', choices=['transfer', 'mechanism'], default='transfer')
    args = parser.parse_args()
    # Historical scientific run(args) functions use the original CLI field.
    args.command = args.operation
    args.run_root = args.run_root.expanduser().resolve()
    if not args.run_root.is_dir():
        parser.error('--run-root must be a prepared workspace; see REPRODUCIBILITY.md')
    if args.run_root == ROOTS[args.family].resolve():
        parser.error('Choose a separate runtime workspace to preserve publication artifacts')
    if args.family == 'crosscheck':
        if args.operation != 'eval' or not args.model or not args.suite:
            parser.error('crosscheck supports eval with explicit --model and --suite')
        sys.path.insert(0, str(ROOTS[args.family] / 'code'))
        common = importlib.import_module('common')
        common.ROOT = args.run_root
        common.PROJECT = args.run_root.parents[1]
        common.SOURCE = common.PROJECT / 'supplements/wrist_dino_sensitivity_v1'
        common.FACTORIAL = common.PROJECT / 'supplements/wrist_dino_factorial_v1'
        common.PYTHON = common.PI_PYTHON = sys.executable
        importlib.import_module('worker').run(args.model, args.suite, args.mode)
        return
    sys.path.insert(0, str(ROOTS[args.family]))
    common = importlib.import_module('joint_repro.common')
    common.ROOT = args.run_root
    common.PYTHON = sys.executable
    if args.family != 'sensitivity' and args.scope == 'holdout':
        parser.error('holdout is defined only by the sensitivity study')
    if args.family == 'factorial':
        if args.operation not in ['train', 'eval', 'diagnose', 'report']:
            parser.error('factorial only trains/evaluates D07 and diagnoses M04')
        if args.operation in ['train', 'eval'] and (args.id != 'D07' or args.seed not in [42, 43, 44]):
            parser.error('factorial uses only D07 seeds42/43/44')
        importlib.import_module('joint_repro.supplement').validate()
    if args.operation == 'build-bank':
        with common.Monitor('P01_bank') as monitor:
            importlib.import_module('joint_repro.bank').build(monitor)
    elif args.family == 'sensitivity' and args.operation == 'preflight':
        print(importlib.import_module('joint_repro.sweep').preflight())
    elif args.operation == 'report' and args.family in ['sensitivity', 'factorial']:
        module = 'sweep_report' if args.family == 'sensitivity' else 'ablation_report'
        target = importlib.import_module('joint_repro.' + module)
        target.report() if args.family == 'sensitivity' else target.run()
    else:
        modules = {'preflight': 'preflight', 'train': 'train', 'clean-eval': 'evaluate',
                   'eval': 'evaluate', 'collect-replay': 'replay', 'diagnose': 'diagnostics',
                   'score': 'score', 'analyze': 'analyze', 'audit': 'audit'}
        if args.operation not in modules:
            parser.error('Operation not available for this family')
        importlib.import_module('joint_repro.' + modules[args.operation]).run(args)


if __name__ == '__main__':
    main()
