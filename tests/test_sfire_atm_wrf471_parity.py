"""Atmospheric SFIRE exchange against compiled WRF v4.7.1.

Exponential heat and moisture insertion, fire-grid summation and bilinear
interpolation compare the original routine words. Smoke also retains the
original single-column output as a negative control for the corrected loop.
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu
from tools.sfire_wrf471_oracle.fixture import load


def _plane(a):
    return np.ascontiguousarray(np.asarray(a))


def _column(a):
    return np.ascontiguousarray(np.asarray(a))


def _bits(got, expected, label):
    import cupy as cp
    from gpuwm.core.fp32_ulp import assert_bit_exact
    assert_bit_exact(cp.asnumpy(got), expected, label)


@requires_gpu
@pytest.mark.parametrize("case", [1, 2, 3, 4])
def test_exponential_fire_tendency_matches_original_wrf(case):
    from gpuwm.core.sfire_atm import fire_tendency
    fixture = load(f"atm/exponential_{case}")
    nz = fixture["rho"].shape[0] - 1
    th, qv = fire_tendency(
        *(_plane(fixture[name]) for name in ("grnhfx", "grnqfx", "canhfx", "canqfx")),
        terrain=_plane(fixture["terrain"]), z_at_w=_column(fixture["z_at_w"]),
        dz8w=_column(fixture["dz8w"])[:nz], mu=_plane(fixture["mu"]),
        c1h=fixture["c1h"][:nz], c2h=fixture["c2h"][:nz],
        rho=_column(fixture["rho"])[:nz],
        fire_ext_grnd=float(fixture["ext_grnd"]),
        fire_ext_crwn=float(fixture["ext_crwn"]),
        crown_height=float(fixture["crown_height"]))
    _bits(th, _column(fixture["rthfrten"])[:nz], "RTHFRTEN")
    _bits(qv, _column(fixture["rqvfrten"])[:nz], "RQVFRTEN")


@requires_gpu
def test_fire_cell_sum_matches_original_wrf_order():
    from gpuwm.core.sfire_atm import sum_fire_cells
    fixture = load("atm/sum_cells")
    got = sum_fire_cells(_plane(fixture["fine"]), int(fixture["sr_x"]), int(fixture["sr_y"]))
    _bits(got, _plane(fixture["sum"]), "sum_2d_cells")


@requires_gpu
def test_fire_interpolation_matches_original_wrf():
    from gpuwm.core.sfire_atm import interpolate_2d
    fixture = load("atm/interpolate_2d")
    want = _plane(fixture["fine"])
    got = interpolate_2d(_plane(fixture["coarse"]), want.shape,
                         int(fixture["sr_x"]), int(fixture["sr_y"]))
    _bits(got, want, "interpolate_2d")


@requires_gpu
def test_smoke_corrects_original_single_column_loop():
    import cupy as cp
    from gpuwm.core.sfire_atm import add_fire_tracer_emissions
    fixture = load("atm/original_smoke_column_defect")
    original = _column(fixture["tracer"][0])[:-1]
    initial = _column(fixture["tracer0"][0])[:-1]
    ny, nx = fixture["rho"].shape[1:]
    rx, ry = int(fixture["sr_x"]), int(fixture["sr_y"])
    tracer = cp.asarray(initial)
    add_fire_tracer_emissions(tracer, _plane(fixture["burnt"]), _plane(fixture["fuel"]),
        rho=_column(fixture["rho"])[:-1], dz8w=_column(fixture["dz8w"])[:-1],
        sr_x=rx, sr_y=ry, smoke_yield=0.015)
    got = cp.asnumpy(tracer)
    from gpuwm.core.fp32_ulp import assert_bit_exact
    assert_bit_exact(got[:, :, 1], original[:, :, 1], "original written smoke column")
    assert np.count_nonzero(original[0, 1:-1, 2:-1]) == 0
    assert np.all(got[0, 1:-1, 1:-1] > 0)
    assert np.count_nonzero(got[1:]) == 0
    assert np.count_nonzero(got[0, [0, -1]]) == 0
    assert np.count_nonzero(got[0, :, [0, -1]]) == 0
    rho, dz = _column(fixture["rho"])[:-1], _column(fixture["dz8w"])[:-1]
    burnt, fuel = _plane(fixture["burnt"]), _plane(fixture["fuel"])
    for j in range(1, ny-1):
        for i in range(1, nx-1):
            dry_mass = np.sum(burnt[j*ry:(j+1)*ry, i*rx:(i+1)*rx].astype(np.float64) *
                              fuel[j*ry:(j+1)*ry, i*rx:(i+1)*rx]) / (rx*ry)
            emitted = float(got[0,j,i]) * float(rho[0,j,i]) * float(dz[0,j,i]) / 1000
            assert emitted == pytest.approx(float(np.float32(0.015))*dry_mass, rel=4e-7)


@requires_gpu
def test_log_profile_branches_have_expected_physical_bounds():
    import cupy as cp
    from gpuwm.core.sfire_atm import log_wind_at_height
    heights = cp.asarray([1., 10., 100.], dtype=cp.float32)[:, None, None]
    wind = cp.asarray([2., 5., 9.], dtype=cp.float32)[:, None, None]
    rough = cp.asarray([[0.1]], dtype=cp.float32)
    assert float(log_wind_at_height(wind, heights, rough, 0.05)[0,0]) == 0
    assert float(log_wind_at_height(wind, heights, rough, 10)[0,0]) == 5
    assert float(log_wind_at_height(wind, heights, rough, 150)[0,0]) == 9
    assert 0 < float(log_wind_at_height(wind, heights, rough, 0.5)[0,0]) < 2


@requires_gpu
def test_tg_height_is_initialized_and_crown_sensible_flux_is_independent():
    import cupy as cp
    from gpuwm.core.sfire_atm import fire_tendency, truncated_gaussian
    shape = (12, 4, 5)
    dz = cp.full(shape, 10, cp.float32)
    zs = cp.full(shape[1:], 100, cp.float32)
    zw = zs[None]+cp.arange(13, dtype=cp.float32)[:,None,None]*10
    prop = truncated_gaussian(dz, zw, zs, peak=40, upper=200, extinction=60)
    assert bool(cp.all(cp.isfinite(prop)))
    assert float(prop[4,2,2]) > float(prop[0,2,2])
    zero = cp.zeros(shape[1:], cp.float32)
    args = dict(terrain=zs, z_at_w=zw, dz8w=dz, mu=cp.full_like(zs,90000),
                c1h=cp.ones(12,cp.float32), c2h=cp.zeros(12,cp.float32),
                rho=cp.ones(shape,cp.float32),fire_ext_grnd=60,fire_ext_crwn=60,
                crown_height=20,fire_sfc_flx=1,fire_heat_peak=40,fire_tg_ub=200)
    th, _ = fire_tendency(zero,zero,zero,cp.full_like(zs,4000),**args)
    assert int(cp.count_nonzero(th)) == 0
    th, _ = fire_tendency(zero,zero,cp.full_like(zs,4000),zero,**args)
    assert int(cp.count_nonzero(th)) > 0


@requires_gpu
@pytest.mark.parametrize("peak,upper,extinction",[(0.,200.,10.),(47.,83.,35.),(110.,200.,20.)])
def test_corrected_tg_conserves_column_heat_water_and_smoke(peak,upper,extinction):
    import cupy as cp
    from gpuwm.core.sfire_atm import fire_tendency, add_fire_tracer_emissions
    z=cp.asarray([0.,3.,10.,24.,40.,67.,88.,120.,160.],dtype=cp.float32)[:,None,None]
    dz=cp.broadcast_to(z[1:]-z[:-1],(8,2,3)).copy()
    z=cp.broadcast_to(z,(9,2,3)).copy()
    rho=cp.broadcast_to(cp.asarray([1.2,1.18,1.15,1.1,1.05,1.,0.98,0.93],dtype=cp.float32)[:,None,None],dz.shape).copy()
    terrain=cp.zeros((2,3),cp.float32)
    domain=(0,2,0,1)
    mu=cp.full((2,3),90000,cp.float32)
    c1=cp.linspace(0.9,0.3,8,dtype=cp.float32)
    c2=cp.linspace(3,11,8,dtype=cp.float32)
    hg=cp.full_like(mu,25000); qg=cp.full_like(mu,7000)
    hc=cp.full_like(mu,9000); qc=cp.full_like(mu,3000)
    th,qv=fire_tendency(hg,qg,hc,qc,terrain=terrain,z_at_w=z,dz8w=dz,mu=mu,
        c1h=c1,c2h=c2,rho=rho,fire_ext_grnd=extinction,fire_ext_crwn=extinction+5,
        crown_height=15,fire_sfc_flx=1,fire_heat_peak=peak,fire_tg_ub=upper,domain=domain)
    assert bool(cp.all(th>=0)) and bool(cp.all(qv>=0))
    assert int(cp.count_nonzero(th[-1]))==0
    mass=c1[:,None,None]*mu[None]+c2[:,None,None]
    sensible=cp.sum(th.astype(cp.float64)*rho*dz/mass,axis=0)*1004.5
    latent=cp.sum(qv.astype(cp.float64)*rho*dz/mass,axis=0)*2500000.
    np.testing.assert_allclose(cp.asnumpy(sensible),cp.asnumpy(hg+hc),rtol=5e-7)
    np.testing.assert_allclose(cp.asnumpy(latent),cp.asnumpy(qg+qc),rtol=5e-7)
    tracer=cp.zeros_like(dz)
    burnt=cp.full((4,6),0.125,cp.float32)
    fuel=cp.full_like(burnt,0.8)
    add_fire_tracer_emissions(tracer,burnt,fuel,rho=rho,dz8w=dz,sr_x=2,sr_y=2,
        smoke_yield=0.015,scheme=1,z_at_w=z,terrain=terrain,peak=peak,upper=upper,
        extinction=extinction,domain=domain)
    emitted=cp.sum(tracer.astype(cp.float64)*rho*dz,axis=0)/1000.
    np.testing.assert_allclose(cp.asnumpy(emitted),0.125*float(np.float32(0.8))*float(np.float32(0.015)),rtol=5e-7)
