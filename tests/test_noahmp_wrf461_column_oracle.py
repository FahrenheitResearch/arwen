"""WOOF's shipped Noah-MP device path against WRF v4.6.1's own driver, bit for bit.

``tests/data/oracles/noahmp/column`` holds two runs of the unmodified WRF
v4.6.1 ``NOAHMP_INIT`` + ``noahmplsm`` (phys/noahmp at 848f54a, built at -O0
with gfortran 13.3.0 on glibc 2.39 by
``tools/noahmp_wrf461_oracle/column/build_column.sh``) over 128 columns: a
July afternoon at dt = 60 s for 36 steps and a January afternoon at
dt = 300 s for 24 steps.  The columns span sunlit convective land, stable
nights, sub-layer, one-, two- and three-layer snow, melting snow and rain on
snow, the 2000 mm SWE cap, frozen soil, six glacier columns, deserts,
saturated wetlands under heavy rain, high terrain, urban cells, the southern
hemisphere, open water, partial and full sea ice and fourteen edge cases
(``regimes-*.txt``), with forcing that changes every step.

This module runs WOOF's forecast entry (``noahmp_cold_start`` then
``noahmp_lsm_step`` per step, the slab orchestration with the CUDA leaves,
the CUDA glacier column and the device write-back) on the same inputs and
requires every compared value of every snapshot to carry the same bits.
73 fields (94 values per column), 37 + 25 snapshots, about 745,000 values.

The only values not compared are EFLXBXY, SOILENERGY and SNOWENERGY on the
six glacier columns, which WRF fills from locals the glacier arm never
assigns (``compare_column.py`` docstring; WRF publishes the preceding land
column's EFLXB, which the receipt shows).  They are counted, not hidden.

Two defects this oracle found are pinned here: ACSNOM was summed as
``ACSNOM + (QMELT*DT + PONDINGXY)`` instead of WRF's left-to-right order
(1 ULP on 8 winter values), and the cold start did not seed T2MVXY/T2MBXY,
so every water and sea-ice column published 0 K where WRF publishes the
NOAHMP_INIT seed (796 values).
"""

from __future__ import annotations

import gzip
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import cupy  # noqa: F401  (a device module: the gpu marker follows)
import numpy as np
import pytest

from conftest import requires_gpu

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "data" / "oracles" / "noahmp" / "column"
HARNESS = ROOT / "tools" / "noahmp_wrf461_oracle" / "column"
CONFIGS = ("summer60", "winter300")


def _load(name):
    spec = importlib.util.spec_from_file_location(
        f"noahmp_column_{name}", HARNESS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _unzip(tmp_path, name):
    target = tmp_path / name[:-3]
    target.write_bytes(gzip.decompress((FIXTURES / name).read_bytes()))
    return target


def _woof_run(tmp_path, config):
    inp = _unzip(tmp_path, f"in-{config}.bin.gz")
    wrf = _unzip(tmp_path, f"wrf461-O0-{config}.bin.gz")
    out = tmp_path / f"woof-{config}.bin"
    # This oracle predates the default's corrected night snow aging.
    receipt = _load("run_woof_column").run(
        str(inp), str(out), noahmp_snow_age="wrf_461")
    return inp, wrf, out, receipt


def test_the_harness_ignores_another_oracles_generator(monkeypatch):
    """Other oracle files also import a module named gen_columns."""
    monkeypatch.setitem(sys.modules, "gen_columns", ModuleType("gen_columns"))
    from tools.noahmp_wrf461_oracle.column import gen_columns

    runner = _load("run_woof_column")
    compare = _load("compare_column")
    assert runner.CONFIGS is gen_columns.CONFIGS
    assert runner.read_records is compare.read_records is gen_columns.read_records


@requires_gpu
@pytest.mark.parametrize("config", CONFIGS)
def test_woof_matches_the_wrf461_column_oracle_bit_for_bit(tmp_path, config):
    inp, wrf, out, receipt = _woof_run(tmp_path, config)
    assert receipt["census_first"]["land"] == 113
    assert receipt["census_first"]["glacier"] == 6
    assert receipt["census_first"]["water"] == 4
    assert receipt["census_first"]["sea_ice"] == 5
    assert "cuda" in receipt["census_first"]["glacier_path"]

    result = _load("compare_column").compare(str(wrf), str(out), str(inp))
    assert result["columns"] == 128
    assert result["fields"] == 73
    assert result["compared_values"] > 300_000
    misses = {name: rec for name, rec in result["per_field"].items()
              if rec["differing"]}
    assert result["differing_values"] == 0, misses
    # The documented exclusion is real and bounded: six glacier columns,
    # three fields, every snapshot.
    assert result["glacier_columns"] == [43, 44, 45, 46, 47, 48]
    excluded = result["excluded"]["glacier_undefined"]
    assert excluded["values"] == 3 * 6 * result["snapshots"]


@requires_gpu
def test_the_column_comparison_can_fail(tmp_path):
    """One flipped low bit in one value must be reported as one miss."""
    inp, wrf, out, _ = _woof_run(tmp_path, "winter300")
    compare = _load("compare_column")
    data = bytearray(out.read_bytes())
    # The last record of the file is zsnsoxy of the last snapshot; flip the
    # lowest bit of its final value.
    data[-4] ^= 0x01
    out.write_bytes(bytes(data))
    result = compare.compare(str(wrf), str(out), str(inp))
    assert result["differing_values"] == 1
    assert result["per_field"]["zsnsoxy"]["max_ulp"] == 1


@requires_gpu
def test_the_cold_start_seeds_the_two_metre_temperatures_over_water(tmp_path):
    """NOAHMP_INIT seeds T2MVXY/T2MBXY on every column; noahmplsm never
    rewrites them on open water or sea ice, so the seed is what WRF
    publishes there for the whole run."""
    inp, wrf, out, _ = _woof_run(tmp_path, "summer60")
    compare = _load("compare_column")
    reads = _load("gen_columns").read_records(str(inp))
    skipped = np.flatnonzero((reads["xland"][0] >= 1.5)
                             | (reads["xice"][0] >= 0.5))
    assert skipped.size == 9
    woof = compare.snapshots(str(out))
    for snap in (woof[0], woof[-1]):
        for name in ("t2mvxy", "t2mbxy"):
            assert np.all(snap[name][skipped] > 200.0), name
