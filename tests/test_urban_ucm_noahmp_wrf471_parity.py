"""The UCM's Noah-MP coupling against WRF v4.7.1, bit for bit.

``tools/urban_wrf471_oracle/ucm_noahmp_oracle.F90`` calls the byte-unmodified
``noahmp_urban`` (``phys/noahmp/drivers/wrf/module_sf_noahmpdrv.F`` at
noahmp e5c0859) with ``sf_urban_physics = 1`` on the grid fields noahmplsm
leaves, then runs WRF's own option-1 surface-driver override block
(``module_surface_driver.F:3383-3405``, extracted verbatim by the build):
eleven urban columns and one grass column, four carried steps, the three
NLCD types, FRC 1.0 and 0.05, weak exchange (the 1.0E-02 floors and
``CHS2 = CQS2``), day, calm night and rain.

**Measured: every blended grid field (TSK HFX QFX LH GRDFLX ALBEDO QSFC UST
CHS CHS2 CQS2), every urban state word, every per-step urban output, and
the override block's Q2 U10 V10 PSIM PSIH GZ1OZ0 AKHS AKMS are bit-identical
to WRF on all 44 urban column steps**, from recorded and from carried state;
the grass column is untouched.

**T2 and TH2 carry a declared divergence** (one of the urban port's two;
the other is YSU's rural drag under BEP, ``tests/test_ysu_bep_rural_drag.py``)
and are bit-identical to WRF
built with it (``build_ucm_noahmp.sh`` with ``UCM_T2_TEMPERATURE_FIX=1``,
fixture ``ucm-noahmp-t2fix.csv.gz``): WRF's ``module_surface_driver.F:3393``
divides the UCM's 2 m value by ``(1.E5/PSFC)**RCP`` as if it were a
potential temperature, but ``module_sf_urban.F:1686`` builds it from TS and
TA, both absolute temperatures (TA_URB = T3D, noahmpdrv.F:3392; WRF's own
note at module_sf_urban.F:1679).  The stock and fixed fixtures differ in
T2/TH2 on the urban rows and in nothing else, which the last test pins.

Made to fire before commit: deleting the kernel's ``CHS2 = CQS2`` (WRF's
noahmpdrv.F:3445) fails both parametrizations on T2, TH2 and TH2_URB2D over
40 rows.  One mutation that did not fire, recorded so nobody reads it as
coverage: moving ``RCP`` by one ULP changes ``(1.E5/PSFC)**RCP`` by about
3e-10 relative at these pressures, below float32 resolution.
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu


@requires_gpu
@pytest.mark.parametrize("carried", [False, True], ids=["recorded", "carried"])
def test_noahmp_blend_and_overrides_are_bitwise_wrf(carried):
    from gpuwm.core.fp32_ulp import fp32_ulp_distance
    from gpuwm.verify.urban_ucm_oracle import (load_variant, noahmp_replay,
                                               port_params)

    _, table, switches = load_variant("ucm-nlcd-default")
    port, ref, untouched = noahmp_replay(port_params(table, switches),
                                         carried=carried)
    assert untouched, "a kernel wrote a non-urban column"
    ref.update(_t2fix_reference())
    lines = []
    for key, want in ref.items():
        got = np.asarray(port[key], np.float32)
        bad = np.nonzero(got.view(np.uint32) != want.view(np.uint32))[0]
        if bad.size:
            lines.append(f"{key}: {bad.size} rows, max "
                         f"{int(np.max(fp32_ulp_distance(got, want)))} ULP")
    assert not lines, "\n".join(lines)
    assert port["tsk"].size == 44


def test_noahmp_fixture_reaches_what_it_claims():
    from gpuwm.verify.urban_ucm_oracle import load_noahmp

    rows = load_noahmp()
    urban = rows["utype"] > 0
    assert int(urban.sum()) == 44 and int((~urban).sum()) == 4
    assert np.any(rows["chs_in"][urban] < 0.01)
    # WRF sets CHS2 = CQS2 after flooring (noahmpdrv.F:3445)
    assert np.all(rows["chs2"][urban] == rows["cqs2"][urban])
    assert np.any(rows["rainbl"][urban] > 0)
    assert np.any(np.sqrt(rows["u1"] ** 2 + rows["v1"] ** 2)[urban] < 1.0)
    # the override block really moved T2 off Noah-MP's own bare-soil value
    assert np.all(rows["t2"][urban] != rows["t2mbxy"][urban])


def _t2fix_reference():
    """T2/TH2 of the fixed-WRF fixture, in noahmp_replay's row order."""
    from gpuwm.verify.urban_ucm_oracle import NOAHMP_T2FIX_FIXTURE, load_noahmp

    rows = load_noahmp(fixture=NOAHMP_T2FIX_FIXTURE)
    order = [np.nonzero((rows["step"] == step) & (rows["utype"] > 0))[0]
             for step in sorted(set(rows["step"].tolist()))]
    return {name: np.concatenate([rows[name][sel] for sel in order])
            for name in ("t2", "th2")}


def test_the_t2_fixture_differs_from_wrf_only_where_the_divergence_is():
    """The fixed build changes T2 and TH2 on urban rows and nothing else, and
    by exactly WRF's second conversion: stock T2's urban part is the fixed
    one's times (PSFC/1.E5)**RCP."""
    from gpuwm.verify.urban_ucm_oracle import NOAHMP_T2FIX_FIXTURE, load_noahmp

    stock, fixed = load_noahmp(), load_noahmp(fixture=NOAHMP_T2FIX_FIXTURE)
    assert stock.keys() == fixed.keys()
    urban = stock["utype"] > 0
    for name in stock:
        same = stock[name].view(np.uint32) == fixed[name].view(np.uint32)
        if name in ("t2", "th2"):
            assert same[~urban].all() and not same[urban].any(), name
        else:
            assert same.all(), name
    f = stock["frc"][urban].astype(np.float64)
    exner = (stock["psfc"][urban].astype(np.float64) / 1.0e5) ** (2.0 / 7.0)  # RCP = R_D/CP, CP = 3.5 R_D
    # T2 = rural*(1-f) + TH2U*f (fixed) versus rural*(1-f) + TH2U*exner*f
    delta = fixed["t2"][urban].astype(np.float64) - stock["t2"][urban]
    th2u = stock["th2_urb"][urban].astype(np.float64)
    assert np.allclose(delta, th2u * f * (1.0 - exner), atol=2e-4)


def test_the_t2_divergence_is_declared_where_a_reader_looks():
    """The kernel names it, the oracle build applies exactly it, and the
    public physics page states it with its reason: a divergence nobody can
    find is a silent one."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    kernel = (root / "gpuwm" / "core" / "kernels" / "urban_ucm.cu").read_text(
        encoding="utf-8")
    assert "NAMED DIVERGENCE from WRF v4.7.1 module_surface_driver.F:3392-3393" in kernel
    build = (root / "tools" / "urban_wrf471_oracle" / "build_ucm_noahmp.sh").read_text(
        encoding="utf-8")
    assert "UCM_T2_TEMPERATURE_FIX" in build
    assert "TH2_URB2D(i,j)*FRC_URB2D(I,J)" in build
    physics = (root / "docs" / "public" / "PHYSICS.md").read_text(encoding="utf-8")
    assert "The UCM's 2 m temperature under Noah-MP" in physics
    assert "as if it were a potential temperature" in physics
