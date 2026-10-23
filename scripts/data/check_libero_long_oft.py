#!/usr/bin/env python3
"""Compatibility wrapper for the active libero_10 / LIBERO-Long checker."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data.check_libero10_oft import (  # noqa: E402
    check_candidate_fields,
    discover_libero10,
    is_libero10_or_long_name,
    main,
)


discover_libero_long = discover_libero10
is_libero_long_name = is_libero10_or_long_name


if __name__ == "__main__":
    raise SystemExit(main())
