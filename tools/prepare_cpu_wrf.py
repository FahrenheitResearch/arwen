#!/usr/bin/env python3
"""Source-checkout entry point for :mod:`gpuwm.source_cli`."""

from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gpuwm.source_cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
