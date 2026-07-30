#!/usr/bin/env python3
"""Source-checkout entry point for the native ERA5 direct-WRF proof."""

from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gpuwm.era5_direct import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())

