"""The Eta similarity surface layer against WRF v4.6.1, word for word.

The fixture (tests/data/oracles/myjsfc/myjsfc-wrf461.npz) is the output of
tools/myjsfc_wrf461_oracle/build.sh: the byte-unmodified
phys/module_sf_myjsfc.F compiled by gfortran 13.3 on glibc 2.39 (W1, Xeon
Platinum 8559C), driven through MYJSFCINIT and successive MYJSFC calls over
224 columns in three sets (cold start at ITIMESTEP 1-3, warm seeded state
at ITIMESTEP 8-10 across the three sea viscous regimes, and the two-level
minimum column).  Its compiler.txt, o2-equality.txt and coverage-myjsfc.txt
beside it are the build receipts: no FMA, no libmvec, WRF's own -O2 flags
write the same bytes as -O0, and every executable line of MYJSFC and SFCDIF
runs.

What is asserted:

* CPU, everywhere: the PSIM/PSIH tables WOOF builds on the host are
  MYJSFCINIT's, all 4 x 10,001 words and the eight scalars.  They used to
  be built with NumPy's float32 log/atan/exp and differed on up to 3,996
  PSIM words (2,048 ULP), by an amount that depended on the host's CPU.
* CPU, everywhere: the float32 CPU authority (myj_ref.np_myjsfc_column)
  is bit-identical to MYJSFC on every field of every call, so the oracle
  gates on a machine with no GPU too.
* GPU: every INOUT and output field of every call is bit-identical to
  MYJSFC, both replaying WRF's pre-call state and free-running on the
  kernel's own.  The strict build (GPUWM_WRF_EXACT=1) is the standard of
  proof; the same assertion holds under default arithmetic on this
  fixture, so the test does not branch on the mode.
* A negative control: handing the kernel HT = 0 (the old ground-relative
  column) moves PBLH over the terrain columns, so the fixture can see the
  seed it pins.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu
from gpuwm.verify import myjsfc_oracle


@pytest.fixture(scope="module")
def fixture():
    return myjsfc_oracle.load()


def test_the_fixture_covers_the_regimes_it_claims(fixture):
    names = myjsfc_oracle.set_names(fixture)
    assert names == ["cold", "nz2", "warm"]
    cold = fixture["cold/in/xland"]
    assert np.any(cold > 1.5) and np.any(cold < 1.5)
    # Terrain in every set, and above 4 km somewhere (the warm set's
    # tallest column is 4,308 m, the cold set's 3,652 m).
    for name in names:
        assert float(fixture[f"{name}/in/ht"].max()) > 2500.0, name
    assert max(float(fixture[f"{s}/in/ht"].max()) for s in names) > 4000.0
    warm_ust = fixture["warm/in/ustar"]
    assert np.any(warm_ust < 0.225)
    assert np.any((warm_ust >= 0.225) & (warm_ust < 0.7))
    assert np.any(warm_ust >= 0.7)
    columns = sum(int(fixture[f"{s}/meta"][0]) for s in names)
    assert columns >= 64
    fields = [str(f) for f in fixture["fields"]]
    pblh = fixture["cold/out"][:, fields.index("pblh")]
    # Both PBLH branches: above and below the 1000 m BTGH switch.
    assert np.any(pblh > 1000.0) and np.any(pblh < 1000.0)


def test_the_host_tables_are_myjsfcinits_word_for_word(fixture):
    report = myjsfc_oracle.compare_tables(fixture)
    for name, stats in report.items():
        assert stats["mismatch"] == 0, (name, stats)


def test_the_cpu_authority_is_myjsfc_word_for_word(fixture):
    """CPU, everywhere: the float32 twin myj_ref.np_myjsfc_column.

    Its LOG/EXP/POW are WOOF's own float32 routines (gpuwm.core.noahmp_libm);
    with NumPy's float32 log/exp/pow it missed on 32 of 35 fields (up to
    873 ULP on HFX, 505 on RMOL; measured on W1).  Graded exactly as the
    kernel is, so the oracle has a gate that runs without a GPU.
    """
    report = myjsfc_oracle.measure_cpu_authority(fixture)
    bad = {mode: {name: stats for name, stats in report[mode].items()
                  if stats["mismatch"]}
           for mode in ("replay", "free_run")}
    assert bad == {"replay": {}, "free_run": {}}, bad


@pytest.mark.gpu
@requires_gpu
def test_every_field_of_every_call_is_wrfs(fixture):
    import cupy  # noqa: F401  (device test: the conftest marker reads this import)
    report = myjsfc_oracle.measure(fixture)
    bad = {mode: {name: stats for name, stats in report[mode].items()
                  if stats["mismatch"]}
           for mode in ("replay", "free_run")}
    assert bad == {"replay": {}, "free_run": {}}, bad


@pytest.mark.gpu
@requires_gpu
def test_the_terrain_seed_is_visible_to_the_fixture(fixture):
    """Zero HT and PBLH moves on terrain columns: the seed is graded."""
    import cupy  # noqa: F401  (device test: the conftest marker reads this import)
    zeroed = dict(fixture)
    for name in myjsfc_oracle.set_names(fixture):
        zeroed[f"{name}/in/ht"] = np.zeros_like(fixture[f"{name}/in/ht"])
    report = myjsfc_oracle.measure(zeroed)
    assert report["replay"]["pblh"]["mismatch"] > 0
