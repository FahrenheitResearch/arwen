"""New Tiedtke takes one grid spacing PER COLUMN where the driver carries it.

MPAS-A v8.4.1 hands cu_ntiedtke ``dx = len_disp / meshDensity**0.25`` for
each cell (mpas_atmphys_driver_convection.F:840) and the scheme builds its
scale-dependency factors from it column by column (cu_ntiedtke.F90:230-238).
The adapter used to fill the kernels' ``dx`` vector with ``cfg.dx``, so a
column seam on a variable-resolution mesh ran every cell at one nominal
spacing.  These tests hold the per-column path to the scalar path: a column
handed spacing d must produce the words a whole batch at ``cfg.dx = d``
produces.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.core import ntiedtke


def test_the_host_mirror_is_the_source_formula():
    sf, sf2 = ntiedtke.nt_scale_factors_host(
        np.array([60000.0, 15000.0, 14999.0, 3000.0], np.float32))
    # cu_ntiedtke.F90:236-237, the coarse arm, and `<` puts 15 km on it.
    np.testing.assert_allclose(sf[0], 1.0 + 1.33e-5 * 60000.0, rtol=1e-6)
    np.testing.assert_allclose(sf[1], 1.0 + 1.33e-5 * 15000.0, rtol=1e-6)
    assert sf2[0] == 1.0 and sf2[1] == 1.0
    # :233-234, the fine arm.
    np.testing.assert_allclose(
        sf[3], (1.06133 + np.log(5.0)) ** 3, rtol=1e-5)
    np.testing.assert_allclose(sf2[3], np.sqrt(sf[3]), rtol=1e-6)
    # Discontinuous at the join, as the source is.
    assert sf[2] < sf[1]
    assert ntiedtke.NT_DXREF_M == 15000.0


def _columns(cp, ncol, nz=30):
    """A moist, conditionally unstable sounding repeated over ``ncol``."""
    z = np.linspace(0.0, 18000.0, nz + 1).astype(np.float32)
    zc = 0.5 * (z[1:] + z[:-1])
    p = (100000.0 * np.exp(-zc / 8000.0)).astype(np.float32)
    pi = (100000.0 * np.exp(-z / 8000.0)).astype(np.float32)
    theta = (300.0 + 3.5e-3 * zc).astype(np.float32)
    exner = ((p / 1.0e5) ** 0.2857).astype(np.float32)
    tk = theta * exner
    qv = (0.018 * np.exp(-zc / 2800.0)).astype(np.float32)
    rho = (p / (287.0 * tk)).astype(np.float32)
    dz = np.diff(z).astype(np.float32)

    def lev(profile):
        return cp.asarray(np.ascontiguousarray(
            np.repeat(profile[:, None], ncol, axis=1).reshape(
                profile.shape[0], 1, ncol).astype(np.float32)))

    zero = cp.zeros((nz, 1, ncol), dtype=cp.float32)
    atmosphere = {
        "exner": lev(exner), "p_interface": lev(pi),
        "temperature": lev(tk), "qv": lev(qv), "qc": zero.copy(),
        "qi": zero.copy(),
        "u": lev(np.linspace(2.0, 20.0, nz).astype(np.float32)),
        "v": lev(np.linspace(-1.0, 6.0, nz).astype(np.float32)),
        "pressure": lev(p), "dz": lev(dz), "rho": lev(rho),
    }
    fields = {
        "xland": cp.ones((1, ncol), dtype=cp.float32),
        "hfx": cp.full((1, ncol), 150.0, dtype=cp.float32),
        "qfx": cp.full((1, ncol), 1.5e-4, dtype=cp.float32),
    }
    # elapsed_seconds: the exact chain's NTiedtke takes WRF's ITIMESTEP from
    # the state clock (ntiedtke._nt_itimestep).  The second step (dt = 60 s):
    # on the first, WRF zeroes the forcing pair (module_cu_ntiedtke.F:
    # 451-457) and this sounding's closure has nothing to convect on.
    state = SimpleNamespace(
        p=lev(p), w=cp.zeros((nz + 1, 1, ncol), dtype=cp.float32),
        elapsed_seconds=60.0)
    return atmosphere, fields, state


def _cfg(dx):
    return SimpleNamespace(dx=dx, dt=60.0, clock_dt=0.0,
                           ntiedtke_tiedtke_closure=False)


def _driver(cp, ncol, nz, dx_column=None):
    driver = SimpleNamespace(
        gf_rthdynten=cp.full((nz, 1, ncol), 3.0e-5, dtype=cp.float32),
        gf_rqvdynten=cp.full((nz, 1, ncol), 4.0e-8, dtype=cp.float32),
        gf_rthblten=None, gf_rqvblten=None,
        rthratenlw=None, rthratensw=None)
    if dx_column is not None:
        driver.gf_dx_column = cp.asarray(
            np.asarray(dx_column, np.float32).reshape(1, ncol))
    return driver


def _run(cp, *, ncol, cfg_dx, dx_column=None, scale_aware=True, nz=30):
    atmosphere, fields, state = _columns(cp, ncol, nz)
    scheme = ntiedtke.NewTiedtke()
    scheme.scale_aware = scale_aware
    scheme.bind_driver(_driver(cp, ncol, nz, dx_column))
    result = scheme(atmosphere=atmosphere, fields=fields, state=state,
                    cfg=_cfg(cfg_dx))
    workspace = scheme._pipeline[1].w
    out = {
        "rthcuten": cp.asnumpy(result.rthcuten).reshape(nz, ncol),
        "rqvcuten": cp.asnumpy(result.rqvcuten).reshape(nz, ncol),
        "rucuten": cp.asnumpy(result.rucuten).reshape(nz, ncol),
        "rainc": cp.asnumpy(result.rainc).reshape(ncol),
        "scale_fac": cp.asnumpy(workspace.bind("scale_fac", 1))[:ncol].copy(),
        "scale_fac2": cp.asnumpy(workspace.bind("scale_fac2", 1))[:ncol].copy(),
        "dx": cp.asnumpy(workspace.bind("dx", 1))[:ncol].copy(),
    }
    scheme.release()
    return out


@pytest.fixture(scope="module")
def cp():
    cupy = pytest.importorskip("cupy")
    if cupy.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("no CUDA device")
    return cupy


@pytest.mark.gpu
def test_each_column_runs_at_its_own_spacing(cp):
    ncol = 96
    spacings = (60000.0, 15000.0, 9000.0, 3000.0)
    dx_column = np.repeat(np.asarray(spacings, np.float32), ncol // 4)
    mixed = _run(cp, ncol=ncol, cfg_dx=60000.0, dx_column=dx_column)

    np.testing.assert_array_equal(mixed["dx"], dx_column)
    sf, sf2 = ntiedtke.nt_scale_factors_host(dx_column)
    np.testing.assert_allclose(mixed["scale_fac"], sf, rtol=2e-6)
    np.testing.assert_allclose(mixed["scale_fac2"], sf2, rtol=2e-6)

    rain = {}
    for index, spacing in enumerate(spacings):
        uniform = _run(cp, ncol=ncol, cfg_dx=spacing)
        part = slice(index * (ncol // 4), (index + 1) * (ncol // 4))
        for name in ("rthcuten", "rqvcuten", "rucuten"):
            np.testing.assert_array_equal(
                mixed[name][:, part], uniform[name][:, part],
                err_msg=f"{name} at dx={spacing}")
        np.testing.assert_array_equal(mixed["rainc"][part],
                                      uniform["rainc"][part])
        rain[spacing] = float(mixed["rainc"][part].mean())

    # The sounding convects, or the comparison above compared zeros.
    assert rain[60000.0] > 0.0
    # The deep closure's adjustment time is ztauc * scale_fac (:676):
    # 1.80 at 60 km, 1.20 at 15 km, 3.89 at 9 km, 19.1 at 3 km.  A column
    # that differs only in spacing must not rain the same at all four.
    assert len({rain[s] for s in spacings}) > 1
    assert rain[3000.0] < rain[15000.0]


@pytest.mark.gpu
def test_scale_aware_off_holds_fine_columns_at_dxref(cp):
    ncol = 64
    dx_column = np.repeat(np.asarray((30000.0, 3000.0), np.float32), ncol // 2)
    off = _run(cp, ncol=ncol, cfg_dx=30000.0, dx_column=dx_column,
               scale_aware=False)
    np.testing.assert_array_equal(off["dx"][: ncol // 2], 30000.0)
    np.testing.assert_array_equal(off["dx"][ncol // 2:], 15000.0)
    at_ref = _run(cp, ncol=ncol, cfg_dx=15000.0)
    coarse = _run(cp, ncol=ncol, cfg_dx=30000.0)
    half = ncol // 2
    for name in ("rthcuten", "rqvcuten", "rucuten"):
        np.testing.assert_array_equal(off[name][:, :half], coarse[name][:, :half])
        np.testing.assert_array_equal(off[name][:, half:], at_ref[name][:, half:])
    np.testing.assert_array_equal(off["rainc"][half:], at_ref["rainc"][half:])
    np.testing.assert_array_equal(off["scale_fac2"], 1.0)


@pytest.mark.gpu
def test_no_column_carrier_is_the_scalar_path_unchanged(cp):
    """A lat-lon driver carries no gf_dx_column: every column sees cfg.dx."""
    out = _run(cp, ncol=64, cfg_dx=4500.0)
    np.testing.assert_array_equal(out["dx"], np.float32(4500.0))
    sf, _ = ntiedtke.nt_scale_factors_host(np.float32(4500.0))
    np.testing.assert_allclose(out["scale_fac"], sf, rtol=2e-6)


@pytest.mark.gpu
def test_a_misshapen_column_carrier_is_refused(cp):
    atmosphere, fields, state = _columns(cp, 32)
    scheme = ntiedtke.NewTiedtke()
    driver = _driver(cp, 32, 30)
    driver.gf_dx_column = cp.full((1, 31), 3000.0, dtype=cp.float32)
    scheme.bind_driver(driver)
    with pytest.raises(ValueError, match="gf_dx_column must be"):
        scheme(atmosphere=atmosphere, fields=fields, state=state,
               cfg=_cfg(3000.0))
