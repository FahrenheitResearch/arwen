"""Sixth-order diffusion against compiled, unmodified WRF v4.6.1, every word.

The fixture is sealed from the full WRF 4.6.1 column oracle
(tools/wrf_diffusion_oracle/diff6_wrf461_*): the byte-unmodified routine
compiled with gfortran 13.3 under strict IEEE flags and under WRF's own
stock flags (the two agree word for word on every case), fed real WRF 4.6.1
states: a convective Iowa 3 km storm, a CONUS 12 km snow/ice night, a steep
ridge and map-factor extremes, plus subnormal and tiny-tail edge cases.

Every case starts from a nonzero incoming tendency or probes an edge case,
so it grades the production row ``dycore.add_diff6_row`` where WRF's
in-place association ``(T0 + tendency_x) + tendency_y`` matters.

One WRF defect is documented here and deliberately not copied: WRF's 'v'
branch trims the y loop bounds whenever open_xs is set (``IF
(config_flags%open_xs .or. specified)``), so an open-x, periodic-y run skips
v's three edge rows on each side of a periodic axis.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pytest

from conftest import requires_gpu

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests/data/wrf461_diff6"
TOOLS = ROOT / "tools/wrf_diffusion_oracle"


def oracle():
    meta = json.loads((DATA / "diff6-wrf461-columns.json").read_text())
    return meta, np.load(DATA / "diff6-wrf461-columns.npz")


def inputs(fixture, case):
    prefix = f"c{case:04d}__"
    return {k[len(prefix):]: fixture[k] for k in fixture.files if k.startswith(prefix)}


def product(case, values, **kw):
    from gpuwm.verify.diff6_wrf461_oracle import launch_diff6_wrf461_case
    return launch_diff6_wrf461_case(case, values, **kw)


def v_open_x_rows(case):
    """Rows WRF's open_xs typo excludes for v on a periodic y axis."""
    if case["var"] != "V" or case["mode_name"] != "open_x":
        return None
    ny = case["dims"][1]
    return [0, 1, 2, ny - 2, ny - 1, ny]


def test_fixture_is_pinned_compiled_wrf461():
    meta, _ = oracle()
    for line in (DATA / "diff6-wrf461-sha256sums.txt").read_text().splitlines():
        digest, name = line.split()
        assert hashlib.sha256((DATA / name).read_bytes()).hexdigest() == digest, name
    for name, digest in meta["tools_sha256"].items():
        assert hashlib.sha256((TOOLS / name).read_bytes()).hexdigest() == digest, name
    assert meta["wrf_version"] == "4.6.1"
    assert meta["wrf_commit"] == "d66e442fccc04111067e29274c9f9eaccc3cef28"
    assert meta["source_sha256"] == "8f0649b458fceabd5c31c87ec9f266840fb164e2629322904a3f6ee04c72331a"
    # The 4.6.1 routine is byte-identical to the 4.7.1 slice pinned by
    # tests/test_diff6_wrf471_parity.py.
    assert meta["routine_slice_sha256"] == "a4534919fdb15c91789f6d0dc504fe12f8eac6ee711f51053b3900999a055567"
    assert meta["routine_slice_equals_wrf471"] and meta["slice_is_byte_unmodified"]
    assert "13.3.0" in meta["compiler"]
    assert "-ffp-contract=off" in meta["builds"]["strict"]["compile"][1]
    assert "-fcheck=bounds" in meta["builds"]["strict"]["compile"][1]
    assert "-funroll-loops" in meta["builds"]["stock"]["compile"][1]
    assert meta["builds"]["stock"]["fma_instructions"] == 0
    assert meta["strict_equals_stock"].startswith("every case")
    assert meta["full_oracle_cases"] == 542


def test_fixture_discriminates_modes_staggers_and_edges():
    meta, fixture = oracle()
    cases = meta["cases"]
    assert {c["mode_name"] for c in cases} == {"periodic", "specified", "nested", "open", "open_x", "open_y"}
    assert {c["stagger"] for c in cases} == set("uvwm")
    assert {(c["opt"], c["slopeopt"]) for c in cases} == {(1, 0), (2, 0), (1, 1), (2, 1)}
    assert any(c["scalar_row"] for c in cases) and any(c["num"] for c in cases)
    assert any(c["var"] in ("QSNOW", "QICE", "QGRAUP") for c in cases)
    assert sum(c["accumulate"] for c in cases) >= 30
    assert {"steep_ridge", "map_extremes", "subnormal_checkerboard", "tiny_tails"} <= {c["group"] for c in cases}
    values = {c["case"]: inputs(fixture, c["case"]) for c in cases}
    assert all(np.isfinite(v["reference"]).all() for v in values.values())
    storm = [values[c["case"]] for c in cases if c["group"] == "iowa_storm" and c["var"] == "W"]
    assert max(np.abs(v["field"]).max() for v in storm) > 5.0       # a real updraft, lowest 12 levels
    for c in cases:
        rows = v_open_x_rows(c)
        if rows is not None:                                            # WRF left them untouched
            v = values[c["case"]]
            assert np.array_equal(v["reference"][:, rows].view(np.uint32), v["tendency"][:, rows].view(np.uint32))


#: Under DEFAULT arithmetic CuPy compiles with flush-to-zero, so a
#: subnormal intermediate (a subnormal field difference, or a limiter
#: product dflux*gradient that underflows) becomes zero where WRF keeps it.
#: The kernel has no other default-only switch that can act: every multiply
#: is an explicit __fmul_rn (nothing to contract), the one divide is
#: __fdiv_rn and there is no square root.  Measured over the full 542-case
#: oracle on an RTX PRO 6000: 42,708 of 429,524,312 words, all in
#: hydrometeor tails and these two probes, at most 8.9e-15 apart.  The
#: strict build (GPUWM_WRF_EXACT=1, --ftz=false) has none.
FTZ_PROBES = ("subnormal_checkerboard", "tiny_tails")
FTZ_MAX_ABS = 1.0e-14


@pytest.mark.gpu
@requires_gpu
def test_production_row_is_bit_identical_to_compiled_wrf461():
    import cupy  # noqa: F401  (device test: the conftest marker reads this import)
    from gpuwm import wrf_exact
    meta, fixture = oracle()
    words = ftz_words = 0
    for case in meta["cases"]:
        values = inputs(fixture, case["case"])
        got, want = product(case, values), values["reference"]
        changed = got.view(np.uint32) != want.view(np.uint32)
        rows = v_open_x_rows(case)
        if rows is not None:
            # WOOF computes the periodic axis in full; WRF's typo skips it.
            changed[:, rows] = False
        if not wrf_exact.ENABLED and case["group"] in FTZ_PROBES and changed.any():
            gap = np.abs(got[changed].astype(np.float64) - want[changed].astype(np.float64))
            assert gap.max() <= FTZ_MAX_ABS, (case, float(gap.max()))
            ftz_words += int(changed.sum())
            changed[:] = False
        assert not changed.any(), (case, int(changed.sum()), np.argwhere(changed)[:4].tolist())
        words += want.size
    print(f"wrf461 diff6 columns strict={wrf_exact.ENABLED} cases={len(meta['cases'])} "
          f"words={words} mismatches=0 ftz_probe_words={ftz_words}")


@pytest.mark.gpu
@requires_gpu
def test_fixture_rejects_the_separately_added_increment():
    """The pre-fix composition, T0 + (tx + ty), must not pass this fixture."""
    import cupy  # noqa: F401  (device test: the conftest marker reads this import)
    meta, fixture = oracle()
    changed = 0
    for case in meta["cases"]:
        if not case["accumulate"]:
            continue
        values = inputs(fixture, case["case"])
        got = product(case, values, legacy_composition=True)
        changed += int((got.view(np.uint32) != values["reference"].view(np.uint32)).sum())
    assert changed > 1000, changed
