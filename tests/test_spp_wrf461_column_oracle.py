"""SPP column oracle (WRF v4.6.1): fixture integrity, coverage, CPU reference.

The fixtures under ``tests/data/oracles/spp/columns`` come from
``tools/spp_wrf461_column_oracle`` (build.py, then pack.py) run against the
unmodified, byte-pinned WRF v4.6.1 ``module_bl_mynn.F`` and GF modules with
gfortran 13.3.0 -O0 -ffp-contract=off on glibc 2.39:

* MYNN, ``spp_pbl = 1``: 72 columns (six families -- convective day, stable
  night, marine stratocumulus, shallow cumulus, a sub-freezing cold pool
  with fog, ice and snow, high wind -- each at sea level, about 1.4 km and
  about 3 km on a deep grid), four pattern shapes (zero, WRF's default
  +-0.3, an edge table with +-1, +-0.9, -2/3 and both zeros, alternating
  +-0.9), eight 20 s steps, mixing lengths 1 and 2.
* GF, closure SPP: the 216 GF fixture columns (18 cases x 6 dx x 2 shallow
  arms) with a different four-member closure pattern on every column, values
  beyond, at and just inside +-1, clipped by GFDRV's own statements, in two
  pattern sets.

The GPU grade is tests/test_spp_wrf461_column_oracle_gpu.py.  This module
checks the fixture is the one the receipt names, that it covers what it
claims, and that the NumPy MYNN reference is word for word WRF on it.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "spp_wrf461_column_oracle"
sys.path.insert(0, str(TOOL))

import spp_column_fixtures as columns  # noqa: E402
import spp_column_grade as grade  # noqa: E402

FIXTURES = columns.FIXTURE_DIR


def test_fixture_bytes_are_the_receipted_ones():
    manifest = json.loads((FIXTURES / "fixtures.json").read_text())
    assert sorted(manifest) == ["gf-spp-o0.npz", "gf-spp-o9.npz",
                                "mynn-spp-ml1.npz", "mynn-spp-ml2.npz"]
    for name, digest in manifest.items():
        assert hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest() == digest, name
    receipt = json.loads((FIXTURES / "receipt.json").read_text())
    assert receipt["wrf_version"] == "4.6.1"
    assert receipt["compiler"].startswith("GNU Fortran") and " 13.3.0" in receipt["compiler"]
    assert receipt["libc"] == ["glibc", "2.39"]
    assert "-ffp-contract=off" in receipt["flags"] and "-O0" in receipt["flags"]


def test_generated_harnesses_are_the_receipted_ones():
    """The receipt's harness digests are what sources.py builds today."""
    import spp_column_sources as sources
    receipt = json.loads((FIXTURES / "receipt.json").read_text())
    built = {"run_mynn_spp_columns.F90": sources.mynn_source(),
             "run_gf_spp_columns.F90": sources.gf_source()}
    assert {name: hashlib.sha256(text.encode()).hexdigest()
            for name, text in built.items()} == receipt["generated_sources"]
    # Only the column drivers change: the SPP switch, the pattern and the
    # column set.  The reference modules are compiled from pinned bytes.
    assert "spp_pbl = 1" in built["run_mynn_spp_columns.F90"]
    assert "rand_clos(its, n) = rawp(n)" in built["run_gf_spp_columns.F90"]


@pytest.mark.parametrize("mixlength", (1, 2))
def test_mynn_columns_cover_the_regimes_and_edges(mixlength):
    cases, steps = columns.mynn_steps(FIXTURES / f"mynn-spp-ml{mixlength}.npz")
    assert len(cases) == 72 and len(steps) == 8
    labels = [columns.regime(case) for case in cases]
    for axis, want in (("family", 6), ("terrain", 3), ("pattern", 4)):
        assert len({label[axis] for label in labels}) == want, axis
    first = steps[1]
    pattern = np.asarray([[row["rstoch"] for row in block] for block in first], np.float32)
    words = set(pattern.view(np.uint32).ravel().tolist())
    for value in (-1.0, -0.9, -0.6666667, 0.9, 1.0, 0.0, -0.0):
        assert np.float32(value).view(np.uint32) in words, value
    assert np.abs(pattern).max() == np.float32(1.0)
    # The deep grid of the high-terrain columns reaches the diffusivity
    # taper exp(-(zw - 8000)/2000), which the other grids never do.
    dz = np.asarray([[row["dz"] for row in block] for block in first], np.float32)
    assert dz.sum(axis=1).max() > 20000.0
    ps = np.asarray([block[0]["ps"] for block in first], np.float32)
    assert ps.min() < 70000.0 < ps.max()
    for block in steps.values():
        for column in block:
            for row in column:
                for name in ("rublten", "rthblten", "qke", "exch_h"):
                    assert np.isfinite(row[name]), name


@pytest.mark.parametrize("offset", (0, 9))
def test_gf_columns_cover_clipping_and_convection(offset):
    oracle = columns.load_gf(FIXTURES / f"gf-spp-o{offset}.npz")
    assert oracle["key"].shape == (216, 3)
    spp = oracle["spp"]
    assert np.any(spp > 1.0) and np.any(spp < -1.0)
    assert np.any(spp == 1.0) and np.any(spp == -1.0)
    assert np.any((np.abs(spp) < 1.0) & (spp != 0.0))
    # Every column carries a four-member pattern of its own.
    assert len({row.tobytes() for row in spp}) == 216
    assert int(np.count_nonzero(oracle["ierr"] == 0)) >= 50
    converging = oracle["ierr"] == 0
    assert np.any(converging & np.all(np.abs(spp) < 1.0, axis=1))
    assert np.any(converging & np.any(np.abs(spp) >= 1.0, axis=1))


@pytest.mark.parametrize("mixlength", (1, 2))
@pytest.mark.parametrize("replay", (True, False), ids=("replay", "free"))
def test_cpu_mynn_reference_is_wrf_word_for_word_with_spp(mixlength, replay):
    graded = grade.grade_mynn(FIXTURES / f"mynn-spp-ml{mixlength}.npz", mixlength,
                              replay=replay, backend="cpu")
    worst = {name: field["max_ulp"] for name, field in graded["fields"].items()
             if field["max_ulp"]}
    assert worst == {}, worst
    assert graded["words"] == 366912
