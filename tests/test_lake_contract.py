"""CPU controls for the full WRF lake transcription and restart contract."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path
import sys

import numpy as np

from gpuwm.core.lake_schema import (
    LAKE_DEFAULT_DEPTH, LAKE_DEFAULT_MIN_ELEV, LAKE_DEFAULT_USE_DEPTH,
    LAKE_FORCING_WORDS, LAKE_OUTPUT_WORDS, LAKE_STATE_WORDS,
    LAKE_STATIC_WORDS,
)

ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / "gpuwm/data/lake/oracle"


def test_lake_keeps_wrf_layer_counts_and_defaults():
    assert (LAKE_STATE_WORDS, LAKE_STATIC_WORDS, LAKE_FORCING_WORDS, LAKE_OUTPUT_WORDS) == (131, 71, 13, 9)
    assert (LAKE_DEFAULT_DEPTH, LAKE_DEFAULT_MIN_ELEV, LAKE_DEFAULT_USE_DEPTH) == (50.0, 5.0, 1)


def test_lake_exports_are_visible_to_the_cuda_frame_census():
    # Breakage: a macro in place of __global__ hides both kernels from the
    # production frame inventory and admits domains with zero priced frames.
    source=(ROOT / "gpuwm/core/kernels/lake.cu").read_text()
    assert set(re.findall(r'extern\s+"C"\s+__global__\s+void\s+(\w+)',source)) == {
        "lake_init_columns","lake_step_columns"}


def test_lake_generator_reproduces_every_shipped_wrf_statement():
    path = ROOT / "tools/lake_wrf461_oracle/translate.py"
    spec = importlib.util.spec_from_file_location("lake_transcription_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert module.generate() == (ROOT / "gpuwm/core/kernels/lake_wrf.cuh").read_text(encoding="utf-8")


def test_lake_oracle_has_snow_ice_equatorial_and_long_column_controls():
    with np.load(ORACLE / "columns.npz") as fixture:
        assert fixture["trace_state"].shape == (300, 131, 12)
        assert fixture["trace_output"].shape == (300, 9, 12)
        assert set(fixture["initial_state"][3].astype(int)) == {0, -1, -2, -3, -4, -5}
        assert {0.5, 1.0}.issubset(set(fixture["seed"][4].tolist()))
        assert 0 in fixture["forcing"][12]
        speed=np.hypot(fixture["forcing"][5],fixture["forcing"][6])
        assert np.any(speed==0)
        assert np.any((speed>0)&(speed<0.1))
        assert all(np.isfinite(fixture[name]).all() for name in fixture.files)
        # Breakage: a fixed-SST substitute can look plausible for one call but
        # cannot reproduce the native changing 10-layer lake temperatures.
        assert not np.array_equal(fixture["trace_state"][0, 5:15], fixture["trace_state"][-1, 5:15])


def test_lake_nonpositive_explicit_default_uses_wrf_reference_geometry():
    with np.load(ORACLE / "initialization.npz") as fixture:
        np.testing.assert_array_equal(fixture["defaults"], [50., 0., -1.])
        np.testing.assert_array_equal(fixture["reference_static"][0, 0], np.full(12, 50, np.float32))
        np.testing.assert_array_equal(fixture["reference_static"][1:, 0], np.ones((2, 12), np.float32))


def test_lake_fixture_and_upstream_hashes_are_bound():
    manifest = json.loads((ORACLE / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
