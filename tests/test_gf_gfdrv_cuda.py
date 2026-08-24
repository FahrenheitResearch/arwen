"""The whole GF chain on the device against the pinned WRF boundary.

The oracle is gf-levels.csv / gf-surface.csv -- GFDRV itself, 18 cases x 6
grid spacings x 2 ishallow arms = 216 columns.  gf_gfdrv_stage runs the
driver's mixed-precision preparation, CUP_gf_sh, neg_check, cup_gf,
neg_check and the output algebra per column, one thread per column, with
fzu COMPUTED (gfk_tgamma is glibc's own words, so the pin the CPU suite
needs does not exist here) and the WRF-faithful k22 flag set, since the
reference being graded against is WRF.

The 208/216 mixed-precision boundary is handled exactly as the CPU gate
handles it (tests/test_gf_driver_parity.py): the 8 columns of
gf_oracle.stage_rows_to_distrust are the rows where GFDRV's OWN
decomposition (run_cup_gf.F90) disagrees with GFDRV -- module_gfs_physcons
real(8) constants initialised from real(4) literals -- and on those the
port is graded against the stage path instead, bitwise on all 216, which
attributes the residual to the driver and proves the port adds nothing.

The corrected-k22 half of the shallow ledger lives here too: with the
shipped default (corrected indexing) every word of every GFDRV output is
identical to the faithful run over the whole fixture, because the three
columns the correction touches are rejected under both modes.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TOOLS = os.path.join(_ROOT, "tools", "gf_wrf461_oracle")
for _p in (_ROOT, _TOOLS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gpuwm.core.fp32_ulp import fp32_ulp_distance             # noqa: E402
from gpuwm.core.gf import gf_workspace_floats                 # noqa: E402
from gpuwm.core.kernels import load_module                    # noqa: E402
from gpuwm.verify.gf_oracle import (                          # noqa: E402
    GF_NZ, load_gf_oracle, stage_rows_to_distrust,
)
from gf_field_lists import (                                  # noqa: E402
    DRV_IN_LEV, DRV_IN_SCA, DRV_ISCA_FIELDS, DRV_LEV_FIELDS, DRV_SCA_FIELDS,
)

NZ = GF_NZ

_WRF_LEV = ["rthcuten", "rqvcuten", "rqccuten", "rqicuten", "dudt_phy",
            "dvdt_phy", "gdc", "gdc2"]
_SEAM_LEV = ["outt", "outq", "outqc", "outu", "outv", "outts", "outqs",
             "outqcs"]


@pytest.fixture(scope="module")
def module():
    return load_module("gf")


@pytest.fixture(scope="module")
def fixture():
    return load_gf_oracle()


def _launch(module, fixture, k22_wrf_faithful):
    gl = fixture.levels
    gs = fixture.surface
    n = fixture.ncol
    lvin = np.zeros((n, len(DRV_IN_LEV), NZ), dtype=np.float32)
    scin = np.zeros((n, len(DRV_IN_SCA)), dtype=np.float32)
    iin = np.zeros((n, 3), dtype=np.int32)
    for j, name in enumerate(DRV_IN_LEV):
        lvin[:, j, :] = gl[name]
    for j, name in enumerate(DRV_IN_SCA):
        scin[:, j] = gs[name].astype(np.float32)
    iin[:, 0] = gs["kpbl"].astype(np.int32)
    iin[:, 1] = gs["ishallow"].astype(np.int32)
    iin[:, 2] = gs["ichoice"].astype(np.int32)
    d_lv = cp.asarray(np.ascontiguousarray(lvin))
    d_sc = cp.asarray(np.ascontiguousarray(scin))
    d_ii = cp.asarray(np.ascontiguousarray(iin))
    d_lev = cp.zeros((n, len(DRV_LEV_FIELDS), NZ), dtype=cp.float32)
    d_sca = cp.zeros((n, len(DRV_SCA_FIELDS)), dtype=cp.float32)
    d_isc = cp.zeros((n, len(DRV_ISCA_FIELDS)), dtype=cp.int32)
    fn = module.get_function("gf_gfdrv_stage")
    d_ws = cp.empty(gf_workspace_floats(NZ, n), dtype=cp.float32)
    fn(((n + 63) // 64,), (64,),
       (d_lv, d_sc, d_ii, d_lev, d_sca, d_isc, d_ws,
        np.int32(k22_wrf_faithful), np.int32(n), np.int32(NZ)))
    cp.cuda.Stream.null.synchronize()
    return dict(lev=cp.asnumpy(d_lev), sca=cp.asnumpy(d_sca),
                isc=cp.asnumpy(d_isc))


@pytest.fixture(scope="module")
def drv(module, fixture):
    return _launch(module, fixture, 1)


@pytest.fixture(scope="module")
def exact(fixture):
    return ~stage_rows_to_distrust(fixture)


def _lev(drv, name):
    return drv["lev"][:, DRV_LEV_FIELDS.index(name), :]


def _sca(drv, name):
    return drv["sca"][:, DRV_SCA_FIELDS.index(name)]


def _assert_bit_exact(got, wantw, what):
    gb = np.ascontiguousarray(got, dtype=np.float32).view(np.uint32)
    wb = np.ascontiguousarray(wantw, dtype=np.float32).view(np.uint32)
    bad = np.flatnonzero(gb.ravel() != wb.ravel())
    assert bad.size == 0, (
        f"{what}: {bad.size}/{gb.size} lanes differ; first at flat "
        f"{int(bad[0])}: got 0x{gb.ravel()[int(bad[0])]:08X} "
        f"want 0x{wb.ravel()[int(bad[0])]:08X}")


# ==========================================================================
# the tendencies GFDRV writes, on the 208 columns its own decomposition
# reproduces
# ==========================================================================
@pytest.mark.gpu
@pytest.mark.parametrize("field", _WRF_LEV)
def test_gfdrv_level_output_bitwise(drv, fixture, exact, field):
    _assert_bit_exact(_lev(drv, field)[exact], fixture.levels[field][exact],
                      f"gfdrv.{field}")


@pytest.mark.gpu
@pytest.mark.parametrize("field", ["raincv", "pratec"])
def test_gfdrv_precip_bitwise(drv, fixture, exact, field):
    _assert_bit_exact(_sca(drv, field)[exact],
                      fixture.surface[field][exact], f"gfdrv.{field}")


@pytest.mark.gpu
@pytest.mark.parametrize("field", ["htop", "hbot", "xmb_shallow"])
def test_gfdrv_xmb_free_surface_output_bitwise_everywhere(drv, fixture,
                                                          field):
    """No deep xmb in these three, so all 216 columns are exact."""
    _assert_bit_exact(_sca(drv, field), fixture.surface[field],
                      f"gfdrv.{field}")


@pytest.mark.gpu
@pytest.mark.parametrize("field", ["gdc", "gdc2"])
def test_the_cloud_water_diagnostics_are_bitwise_on_all_216(drv, fixture,
                                                            field):
    """cupclw carries no mass flux, so the 8 mixed-precision columns cannot
    move it -- and do not."""
    _assert_bit_exact(_lev(drv, field), fixture.levels[field],
                      f"gfdrv.{field}")


@pytest.mark.gpu
@pytest.mark.parametrize(
    "field", ["ktop_deep", "k22_shallow", "kbcon_shallow", "ktop_shallow"])
def test_gfdrv_surface_indices_exact(drv, fixture, field):
    got = drv["isc"][:, DRV_ISCA_FIELDS.index(field)].astype(np.int64)
    want = fixture.surface[field].astype(np.int64)
    bad = np.flatnonzero(got != want)
    assert bad.size == 0, f"{field}: columns {bad[:8].tolist()}"


# ==========================================================================
# the stage seam, on ALL 216 columns -- what proves the 8-column residual
# is GFDRV's own and the port adds nothing
# ==========================================================================
@pytest.mark.gpu
@pytest.mark.parametrize("field", _SEAM_LEV)
def test_the_port_reproduces_the_stage_path_on_every_column(drv, fixture,
                                                            field):
    _assert_bit_exact(_lev(drv, field), fixture.stage_levels[field],
                      f"seam.{field}")


@pytest.mark.gpu
@pytest.mark.parametrize("field", ["pret", "prets"])
def test_the_stage_precipitation_seam(drv, fixture, field):
    ss = fixture.stage_surface
    idx = {}
    for i in range(ss["case"].shape[0]):
        idx[(int(ss["case"][i]), int(ss["idx"][i]), int(ss["arm"][i]))] = i
    order = [idx[tuple(int(v) for v in k)] for k in fixture.key]
    _assert_bit_exact(_sca(drv, field), ss[field][order], f"seam.{field}")


# ==========================================================================
# the 8 mixed-precision columns
# ==========================================================================
@pytest.mark.gpu
def test_the_eight_inexact_columns_are_the_fixtures_own(drv, fixture,
                                                        exact):
    """Exactly stage_rows_to_distrust: a 9th inexact column or a healed one
    is a finding either way."""
    got = _lev(drv, "rthcuten")
    want = fixture.levels["rthcuten"]
    inexact = ~np.all(got.view(np.uint32) == want.view(np.uint32), axis=1)
    assert np.array_equal(inexact, ~exact)
    assert int(inexact.sum()) == 8


@pytest.mark.gpu
def test_on_those_eight_the_residual_is_amplitude_not_structure(drv,
                                                                fixture,
                                                                exact):
    """A 1-2 ULP xmb shift scaling the profile: bounded at the CPU gate's
    own envelope and flipping no branch."""
    bad = ~exact
    got = _lev(drv, "rthcuten")[bad]
    want = fixture.levels["rthcuten"][bad]
    d = fp32_ulp_distance(got, want)
    assert int(d[d >= 0].max()) <= 34
    live = np.abs(want) > np.float32(1e-12)
    rel = np.abs(got[live] - want[live]) / np.abs(want[live])
    assert float(rel.max()) < 1e-4
    ktd = drv["isc"][:, DRV_ISCA_FIELDS.index("ktop_deep")].astype(np.int64)
    assert np.array_equal(ktd[bad],
                          fixture.surface["ktop_deep"].astype(np.int64)[bad])
    _assert_bit_exact(_sca(drv, "htop")[bad], fixture.surface["htop"][bad],
                      "htop@8")
    _assert_bit_exact(_sca(drv, "hbot")[bad], fixture.surface["hbot"][bad],
                      "hbot@8")


# ==========================================================================
# the gating
# ==========================================================================
@pytest.mark.gpu
def test_the_cuten_gate_and_the_shallow_switch(drv, fixture):
    pret = _sca(drv, "pret")
    cuten = _sca(drv, "cuten")
    assert np.array_equal(cuten, (pret > 0).astype(np.float32))
    arm = np.array([int(k[2]) for k in fixture.key])
    cutens = _sca(drv, "cutens")
    assert np.all(cutens[arm == 0] == 0.0)
    on = arm == 1
    xmbs = _sca(drv, "xmb_shallow")[on]
    assert np.array_equal(cutens[on], (xmbs > 0).astype(np.float32))


@pytest.mark.gpu
def test_every_column_is_covered(drv):
    assert drv["isc"].shape[0] == 216


# ==========================================================================
# the corrected-k22 ledger, driver half
# ==========================================================================
@pytest.mark.gpu
def test_the_shipped_default_is_bitwise_at_the_wrf_boundary(module, fixture,
                                                            drv):
    """The SHIPPED kernel (corrected k22, flag 0) against the faithful run:
    every output word identical over the whole fixture, because the three
    columns the corrected indexing touches are rejected under both modes
    (tests/test_gf_shallow_cuda.py::test_the_corrected_k22_ledger_entry
    holds the scheme-level half with the per-case numbers)."""
    shipped = _launch(module, fixture, 0)
    for j, name in enumerate(DRV_LEV_FIELDS):
        a = drv["lev"][:, j, :].view(np.uint32)
        b = shipped["lev"][:, j, :].view(np.uint32)
        assert np.array_equal(a, b), name
    for j, name in enumerate(DRV_SCA_FIELDS):
        a = drv["sca"][:, j].view(np.uint32)
        b = shipped["sca"][:, j].view(np.uint32)
        assert np.array_equal(a, b), name
    for j, name in enumerate(DRV_ISCA_FIELDS):
        assert np.array_equal(drv["isc"][:, j], shipped["isc"][:, j]), name
