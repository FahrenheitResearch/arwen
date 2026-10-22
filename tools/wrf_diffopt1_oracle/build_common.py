"""Shared WRF routine extraction and build service for diffusion oracles."""
from pathlib import Path
import sys

# Keep direct `python tools/wrf_diffopt1_oracle/build.py` invocations working
# from any directory, without requiring an installed `tools` namespace.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.wrf_diffusion_oracle.build_common import (  # noqa: E402
    CONFIG_SOURCE as CONFIG_SOURCE, FLAGS as FLAGS, SOURCE_PINS as SOURCE_PINS,
    WRF_COMMIT as WRF_COMMIT, build as build,
    extract_routines as extract_routines, sha256 as sha256,
)
