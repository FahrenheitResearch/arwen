"""WRF real default vertical-interpolation parity (audit FIX-A).

Oracle: WRF v4.6.1 dyn_em/module_initialize_real.F vert_interp /
lagrange_setup / lagrange_interp at the reference run's Registry defaults
(interp_theta=F, lagrange_order=2, use_surface=T, use_levels_below_ground=T,
force_sfc_in_vinterp=1, zap_close_levels=500, t_extrap_type=2,
extrap_type=2).  Closed-form pins are hand-computed from those formulas;
the met_em pin reproduces the ingest audit's measurement method
(.superpowers/sdd/codex/audit-findings-ingest.json, findings 1-2).
"""

import os
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu
from gpuwm.core import constants as c
from gpuwm.verify.npref import np_wrf_real_vert_interp


BUNDLE = Path(os.environ.get("GPUWM_TEST_WRF74_BUNDLE",
                    "gpuwm-fixture-unset/wrf74-bundle"))
MET_EM = BUNDLE / "met_em" / "met_em.d01.1974-04-03_12_00_00.nc"
requires_bundle = pytest.mark.skipif(
    not MET_EM.is_file(), reason="WRF_1974_MP55 reference bundle not present"
)

#: Audit-measured pre-fix theta rms error (K) at eta levels k=0..3 on 775
#: bundle met_em columns (audit-findings-ingest.json finding 1).
AUDIT_LOW_LEVEL_THETA_RMS_K = (1.93, 1.63, 1.31, 0.98)


def _column(values):
    return np.asarray(values, dtype=np.float64)[:, None, None]


def _scalar(value):
    return np.full((1, 1), float(value))


def _linear_logp(p1, y1, p2, y2, pt):
    """Two-point Lagrange (WRF lagrange_interp n=1) in LOG(p)."""
    weight = (math.log(pt) - math.log(p1)) / (math.log(p2) - math.log(p1))
    return y1 + weight * (y2 - y1)


def _quadratic(xs, ys, xt):
    """Closed-form three-point Lagrange polynomial (WRF lagrange_interp n=2)."""
    x0, x1, x2 = xs
    y0, y1, y2 = ys
    return (y0 * (xt - x1) * (xt - x2) / ((x0 - x1) * (x0 - x2))
            + y1 * (xt - x0) * (xt - x2) / ((x1 - x0) * (x1 - x2))
            + y2 * (xt - x0) * (xt - x1) / ((x2 - x0) * (x2 - x1)))


class TestColumnAssembly:
    """vert_interp column build: surface insertion, force_sfc, zaps."""

    PD = [100000.0, 96800.0, 95000.0, 90000.0, 85000.0, 80000.0,
          70000.0, 60000.0, 50000.0, 40000.0, 30000.0, 20000.0]
    VAL = [280.0, 288.8, 287.0, 284.0, 281.0, 278.0,
           271.0, 264.0, 255.0, 244.0, 231.0, 218.0]
    PSFC = 97000.0
    SFC_VAL = 290.0

    def _run(self, targets, **kwargs):
        out = np_wrf_real_vert_interp(
            _column(self.VAL), _scalar(self.SFC_VAL), _column(self.PD),
            _scalar(self.PSFC), _column(targets), **kwargs)
        return out[:, 0, 0]

    def test_force_sfc_makes_surface_bound_the_lowest_layers(self):
        # force_sfc_in_vinterp=1 discards the 96800 Pa input level (it lies
        # between the surface and the eta-1 target 96500), so targets in
        # [95000, 97000] interpolate straight from the surface value
        # (module_initialize_real.F:5978-6002).  vboundb=4 makes the first
        # four targets linear (:6418, :6565-6568).
        got = self._run([96500.0, 96000.0, 94000.0, 88000.0])
        assert got[0] == pytest.approx(_linear_logp(
            97000.0, 290.0, 95000.0, 287.0, 96500.0), abs=1e-12)
        assert got[1] == pytest.approx(_linear_logp(
            97000.0, 290.0, 95000.0, 287.0, 96000.0), abs=1e-12)
        # The zapped 96800 level would have trapped 96500/96000 instead.
        wrong = _linear_logp(96800.0, 288.8, 95000.0, 287.0, 96500.0)
        assert abs(got[0] - wrong) > 0.1
        assert got[2] == pytest.approx(_linear_logp(
            95000.0, 287.0, 90000.0, 284.0, 94000.0), abs=1e-12)
        assert got[3] == pytest.approx(_linear_logp(
            90000.0, 284.0, 85000.0, 281.0, 88000.0), abs=1e-12)

    def test_fifth_eta_level_uses_averaged_quadratic_pairs(self):
        # Assembled column: [100000, 97000(sfc), 95000, 90000, 85000,
        # 80000, 70000, ...].  Target 82000 at 0-based index 4 is the first
        # outside the vboundb linear band: average of the two overlapping
        # order-2 Lagrange polynomials (:6537-6549).
        got = self._run([96500.0, 96000.0, 94000.0, 88000.0,
                         82000.0, 75000.0])
        lp = [math.log(p) for p in
              (100000.0, 97000.0, 95000.0, 90000.0, 85000.0, 80000.0,
               70000.0)]
        oy = [280.0, 290.0, 287.0, 284.0, 281.0, 278.0, 271.0]
        upper = _quadratic(lp[4:7], oy[4:7], math.log(82000.0))
        lower = _quadratic(lp[3:6], oy[3:6], math.log(82000.0))
        assert got[4] == pytest.approx(0.5 * (upper + lower), abs=1e-12)
        upper = _quadratic(lp[5:7] + [math.log(60000.0)],
                           oy[5:7] + [264.0], math.log(75000.0))
        lower = _quadratic(lp[4:7], oy[4:7], math.log(75000.0))
        assert got[5] == pytest.approx(0.5 * (upper + lower), abs=1e-12)

    def test_below_ground_level_within_500pa_of_surface_is_zapped(self):
        # 97300 Pa is 300 Pa below the 97000 Pa surface: dropped
        # (:5957-5961), so a 98000 Pa target interpolates between the
        # retained 100000 Pa below-ground level and the surface.
        pd = [100000.0, 97300.0] + self.PD[2:]
        val = [280.0, 289.5] + self.VAL[2:]
        out = np_wrf_real_vert_interp(
            _column(val), _scalar(self.SFC_VAL), _column(pd),
            _scalar(self.PSFC), _column([98000.0]))
        assert out[0, 0, 0] == pytest.approx(_linear_logp(
            100000.0, 280.0, 97000.0, 290.0, 98000.0), abs=1e-12)

    def test_below_ground_extrapolation_starts_at_deepest_column_entry(self):
        # A target below the whole column extrapolates from all_y(1) — the
        # retained below-ground isobaric level, NOT the surface
        # (:6439-6493).  extrap='constant' is extrap_type=2 (:6486-6488);
        # extrap='temperature' is the t_extrap_type=2 CRC branch
        # (:6459-6467).
        constant = self._run([101000.0], extrap="constant")
        assert constant[0] == 280.0
        crc = self._run([101000.0], extrap="temperature")
        t1 = 280.0 * (100000.0 / c.P0) ** c.RCP
        pavg = 0.5 * (101000.0 + 100000.0)
        dhdp = 11880.516 * 0.1902632 * (pavg / 100.0) ** (0.1902632 - 1.0)
        dt = dhdp * ((101000.0 - 100000.0) / 100.0) * 0.0065
        expected = (t1 + dt) * (c.P0 / 101000.0) ** c.RCP
        assert crc[0] == pytest.approx(expected, abs=1e-12)

    def test_surface_lowest_branch_iteratively_zaps_close_levels(self):
        # With the surface as the lowest level (no below-ground input), the
        # fill loop re-checks every candidate against the last accepted
        # level and never removes the top (:6062-6077).  99000-98800=200
        # < 500 drops the 98800 level.
        pd = [98800.0, 90000.0, 80000.0, 60000.0, 40000.0]
        val = [289.0, 284.0, 278.0, 264.0, 244.0]
        out = np_wrf_real_vert_interp(
            _column(val), _scalar(290.0), _column(pd), _scalar(99000.0),
            _column([95000.0]), force_sfc_in_vinterp=0)
        assert out[0, 0, 0] == pytest.approx(_linear_logp(
            99000.0, 290.0, 90000.0, 284.0, 95000.0), abs=1e-12)

    def test_target_above_source_top_is_fatal(self):
        with pytest.raises(ValueError, match="above source top"):
            self._run([15000.0])


class TestLagrangeSemantics:
    """lagrange_setup order/space semantics on analytic columns."""

    def _case_b(self, values_of, targets, **kwargs):
        pd = _column([100000.0, 85000.0, 70000.0, 55000.0,
                      40000.0, 25000.0, 10000.0])
        return np_wrf_real_vert_interp(
            values_of(pd), values_of(np.full((1, 1), 105000.0)), pd,
            _scalar(105000.0), _column(targets), **kwargs)[:, 0, 0]

    def test_quadratic_logp_data_is_exact_above_the_linear_band(self):
        # Averaged overlapping order-2 polynomials reproduce any quadratic
        # in LOG(p) exactly; the vboundb linear band does not.  Targets 3
        # and 4 share one pressure, pinning the vboundb=4 boundary sharply.
        def f(p):
            lp = np.log(p)
            return 2.0 + 3.0 * lp + 0.5 * lp * lp

        targets = [103000.0, 101000.0, 95000.0, 78000.0, 78000.0,
                   45000.0, 30000.0, 12000.0]
        got = self._case_b(f, targets)
        np.testing.assert_allclose(
            got[4:], f(np.asarray(targets[4:])), rtol=0.0, atol=1e-9)
        linear = _linear_logp(85000.0, float(f(np.array(85000.0))),
                              70000.0, float(f(np.array(70000.0))), 78000.0)
        assert got[3] == pytest.approx(linear, abs=1e-12)
        assert abs(got[3] - got[4]) > 1e-4

    def test_pressure_field_space_is_linear_in_p_not_logp(self):
        # WRF forces interp_type=1 for the full pressure field
        # (module_initialize_real.F:1805-1820): data linear in p is then
        # reproduced exactly at every order, while the LOG(p) space is not.
        def g(p):
            return 1000.0 + 0.004 * p

        targets = [103000.0, 101000.0, 95000.0, 78000.0,
                   60000.0, 45000.0, 30000.0, 12000.0]
        got = self._case_b(g, targets, interp_in_logp=False)
        np.testing.assert_allclose(
            got, g(np.asarray(targets)), rtol=0.0, atol=1e-9)
        got_logp = self._case_b(g, targets, interp_in_logp=True)
        assert np.max(np.abs(got_logp - g(np.asarray(targets)))) > 0.1


@requires_gpu
@pytest.mark.gpu
def test_wrf_vert_interp_gpu_matches_float64_authority():
    import cupy as cp
    from gpuwm.ingest.vert import (
        _prepare_wrf_vert_interp_geometry,
        _wrf_vert_interp_gpu_prepared,
        wrf_vert_interp_gpu,
    )

    rng = np.random.default_rng(41)
    ns, ny, nx = 37, 6, 9
    base = np.geomspace(100000.0, 1000.0, ns)
    source = base[:, None, None] * (1.0 + rng.normal(0.0, 0.001, (ns, ny, nx)))
    source = np.sort(source, axis=0)[::-1]
    values = (300.0 - 60.0 * np.log(100000.0 / source)
              + rng.normal(0.0, 1.5, source.shape))
    psfc = rng.uniform(95000.0, 103500.0, (ny, nx))
    psfc[0, 0] = source[0, 0, 0] + 300.0   # below-ground zap-close case
    psfc[1, 1] = source[1, 1, 1] + 300.0   # deeper zap-close case
    sfc_values = 288.0 + rng.normal(0.0, 2.0, (ny, nx))
    weight = np.geomspace(1.0, 0.02, 49)[:, None, None]
    target = (psfc[None] - 50.0) * weight + 10050.0 * (1.0 - weight)
    target[0, 2, 2] = psfc[2, 2] + 500.0   # below-ground extrapolation
    source_dev = cp.asarray(source, cp.float32)
    psfc_dev = cp.asarray(psfc, cp.float32)
    target_dev = cp.asarray(target, cp.float32)
    plan = _prepare_wrf_vert_interp_geometry(
        source_dev, psfc_dev, target_dev)

    for extrap in ("constant", "temperature"):
        for logp in (True, False):
            reference = np_wrf_real_vert_interp(
                values, sfc_values, source, psfc, target,
                interp_in_logp=logp, extrap=extrap)
            values_dev = cp.asarray(values, cp.float32)
            sfc_values_dev = cp.asarray(sfc_values, cp.float32)
            legacy = wrf_vert_interp_gpu(
                values_dev, sfc_values_dev, source_dev, psfc_dev,
                target_dev, interp_in_logp=logp, extrap=extrap)
            prepared = _wrf_vert_interp_gpu_prepared(
                values_dev, sfc_values_dev, plan,
                interp_in_logp=logp, extrap=extrap)
            cp.testing.assert_array_equal(prepared, legacy)
            got = cp.asnumpy(prepared)
            np.testing.assert_allclose(got, reference, rtol=3.0e-5,
                                       atol=5.0e-3)

    ascending_plan = _prepare_wrf_vert_interp_geometry(
        source_dev[::-1], psfc_dev, target_dev)
    prepared_ascending = _wrf_vert_interp_gpu_prepared(
        cp.asarray(values[::-1].copy(), cp.float32),
        cp.asarray(sfc_values, cp.float32), ascending_plan,
        interp_in_logp=True, extrap="temperature")
    legacy_ascending = wrf_vert_interp_gpu(
        cp.asarray(values[::-1].copy(), cp.float32),
        cp.asarray(sfc_values, cp.float32), source_dev[::-1], psfc_dev,
        target_dev, interp_in_logp=True, extrap="temperature")
    cp.testing.assert_array_equal(prepared_ascending, legacy_ascending)

    with pytest.raises(ValueError, match="above source top"):
        wrf_vert_interp_gpu(
            cp.asarray(values, cp.float32), cp.asarray(sfc_values, cp.float32),
            cp.asarray(source, cp.float32), cp.asarray(psfc, cp.float32),
            cp.full((1, ny, nx), 500.0, cp.float32))
    with pytest.raises(ValueError, match="above the surface"):
        wrf_vert_interp_gpu(
            cp.asarray(values, cp.float32), cp.asarray(sfc_values, cp.float32),
            cp.asarray(source, cp.float32),
            cp.full((ny, nx), 200.0, cp.float32),
            cp.asarray(target, cp.float32))


def _met_em_subsample():
    import netCDF4

    j_index = np.arange(4, 200, 8)          # 25 rows
    i_index = np.arange(4, 250, 8)          # 31 columns -> 775 columns
    with netCDF4.Dataset(MET_EM) as ds:
        def grab(name):
            return np.asarray(ds.variables[name][0], dtype=np.float64)

        tt = grab("TT")[:, j_index][:, :, i_index]
        rh = grab("RH")[:, j_index][:, :, i_index]
        ght = grab("GHT")[:, j_index][:, :, i_index]
        pres = grab("PRES")[:, j_index][:, :, i_index]
        uu = grab("UU")[:, j_index][:, :, np.append(i_index, i_index[-1] + 1)]
        vv = grab("VV")[:, np.append(j_index, j_index[-1] + 1)][:, :, i_index]
    import netCDF4 as nc

    with nc.Dataset(BUNDLE / "geo_em" / "geo_em.d01.nc") as ds:
        terrain = np.asarray(ds.variables["HGT_M"][0], dtype=np.float64)
    terrain = terrain[j_index][:, i_index]
    return tt, rh, ght, pres, uu, vv, terrain


def _dewpoint_from_rh2(rh2, t2):
    """Invert the module's Bolton es so RH2m(D2, T2) reproduces met_em RH."""
    es_t2 = 10.0 * c.SVP1 * np.exp(c.SVP2 * (t2 - c.SVPT0) / (t2 - c.SVP3))
    vapor = np.maximum(rh2, 1.0) / 100.0 * es_t2
    log_ratio = np.log(vapor / (10.0 * c.SVP1))
    return (c.SVP2 * c.SVPT0 - c.SVP3 * log_ratio) / (c.SVP2 - log_ratio)


@requires_bundle
@requires_gpu
@pytest.mark.gpu
def test_real74_low_level_theta_error_vs_met_em_oracle_halved():
    """Audit finding 1-2 regression pin, reproducing its measurement method.

    775 bundle met_em columns (rows 4::8 x cols 4::8) are pushed through
    the production ``initialize_real`` and compared against a float64
    WRF-default emulation built from the same columns (surface pseudo-level,
    force_sfc_in_vinterp=1, zap_close_levels=500, temperature in LOG p with
    lagrange_order=2, pressure linear in p, theta formed after).  The audit
    measured pre-fix theta rms of 1.93/1.63/1.31/0.98 K at eta levels
    k=0..3; the fixed path must at least halve every level.
    """
    from gpuwm.config import RunConfig
    from gpuwm.core.grid import make_vertical_coord
    from gpuwm.ingest.horiz import HorizontalSnapshot
    from gpuwm.ingest.real import (_host, _integrate_moisture,
                                   _saturation_mixing_ratio, initialize_real,
                                   surface_pressure_from_surface)
    from gpuwm.verify.cases.real74_d01 import ETA_LEVELS

    tt, rh, ght, pres, uu, vv, terrain = _met_em_subsample()
    ny, nx = terrain.shape
    p_top = 10000.0
    psfc, t2, rh2, soilhgt = pres[0], tt[0], rh[0], ght[0]
    d2 = _dewpoint_from_rh2(rh2, t2)
    levels_hpa = pres[1:, 0, 0] / 100.0
    assert np.ptp(pres[1:], axis=(1, 2)).max() == 0.0

    cfg = RunConfig(nx=nx, ny=ny, nz=49, dx=12000.0, dy=12000.0,
                    ztop=20000.0, dt=30.0, run_seconds=1800.0,
                    hybrid_opt=2, etac=0.2, moist=True, terrain_opt=1,
                    base_temp=290.0)
    coord = make_vertical_coord(49, hybrid_opt=2, etac=0.2,
                                eta_levels=ETA_LEVELS)
    snapshot = HorizontalSnapshot(
        valid_time=datetime(1974, 4, 3, 12), levels_hpa=levels_hpa,
        fields={
            "TT": tt[1:], "RH": rh[1:], "GHT": ght[1:],
            "UU": uu[1:], "VV": vv[1:], "PSFC": psfc, "T2": t2, "D2": d2,
            "U10": uu[0], "V10": vv[0],
        })
    result = initialize_real(snapshot, cfg, coord, terrain,
                             source_orography=soilhgt, p_top=p_top,
                             sfcp_to_sfcp=True)
    theta = _host(result.state.thb + result.state.thp)

    # Float64 WRF-default emulation (the audit's oracle) on the same columns.
    qsfc = _saturation_mixing_ratio(d2, psfc, 100.0)
    qv_col = _saturation_mixing_ratio(tt[1:], pres[1:], rh[1:])
    qv_col = np.where((pres[1:] < 10000.0) & (qv_col > 1.0e-5), 3.0e-6,
                      qv_col)
    pd, intq, order = _integrate_moisture(
        qv_col, pres[1:], tt[1:], ght[1:], psfc, t2, qsfc, soilhgt)
    psfc_adj = surface_pressure_from_surface(psfc, soilhgt, terrain, t2, qsfc)
    dry_mass = psfc_adj - intq - p_top
    target_pd = (coord.c3h[:, None, None] * dry_mass[None]
                 + coord.c4h[:, None, None] + p_top)
    surface_pd = psfc - intq
    t_interp = np_wrf_real_vert_interp(
        tt[1:][order], t2, pd, surface_pd, target_pd,
        interp_in_logp=True, extrap="temperature")
    p_interp = np_wrf_real_vert_interp(
        pres[1:][order], psfc, pd, surface_pd, target_pd,
        interp_in_logp=False, extrap="temperature")
    theta_oracle = t_interp * (c.P0 / np.maximum(p_interp, target_pd)) ** c.RCP

    rms = np.sqrt(np.mean((theta - theta_oracle) ** 2, axis=(1, 2)))
    for k, audit_rms in enumerate(AUDIT_LOW_LEVEL_THETA_RMS_K):
        assert rms[k] <= 0.5 * audit_rms, (
            f"eta level {k}: theta rms {rms[k]:.4f} K vs audit pre-fix "
            f"{audit_rms} K (bar {0.5 * audit_rms:.3f} K)")
    # Mid-levels must stay WRF-equivalent too (audit measured 0.05-0.37 K
    # rms there pre-fix; the fixed path is FP32-noise from the oracle).
    assert float(rms[4:].max()) <= 0.05
