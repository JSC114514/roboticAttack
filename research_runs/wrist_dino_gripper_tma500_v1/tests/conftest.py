"""CPU mocks only; no CUDA context, simulator creation or real scheduler calls."""
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]/"code"
PROJECT = CODE.parents[2]
sys.path[:0] = [str(CODE),str(PROJECT)]
