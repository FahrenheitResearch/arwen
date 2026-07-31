import os
import ast
from datetime import datetime, timedelta
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import netCDF4
import numpy as np
import pytest

from conftest import assert_gates, requires_gpu

from gpuwm.io.wrfout import WrfoutWriter


BUNDLE = Path(os.environ.get("GPUWM_TEST_WRF74_BUNDLE",
                    "gpuwm-fixture-unset/wrf74-bundle"))
requires_bundle = pytest.mark.skipif(
    not (BUNDLE / "wrfout_reference" /
         "wrfout_d01_1974-04-03_13_00_00").is_file(),
    reason="WRF_1974_MP55 reference bundle not present",
)


def test_legacy_physics_off_real74_and_phase3_are_supported():
    from gpuwm.config import validate_run_config
    from gpuwm.verify.cases.real74_d01 import config, phase3_config

    les = validate_run_config(config())
    assert les.km_opt == 4 and les.bl_pbl_physics == 0
    supported = validate_run_config(phase3_config())
    assert supported.km_opt == 4 and supported.bl_pbl_physics == 1


def _synthetic_real_frame(nz, ny, nx):
    """Small hydrostatic WRF-like column for compatibility testing."""
    pressure = np.linspace(92500.0, 30000.0, nz, dtype=np.float32)
    height = np.linspace(0.0, 10000.0, nz + 1, dtype=np.float32)
    pb = np.broadcast_to(pressure[:, None, None], (nz, ny, nx)).copy()
    phb = np.broadcast_to(
        (9.81 * height)[:, None, None], (nz + 1, ny, nx)).copy()
    y = np.linspace(30.0, 34.0, ny, dtype=np.float32)[:, None]
    x = np.linspace(-100.0, -94.0, nx, dtype=np.float32)[None, :]
    lat = np.broadcast_to(y, (ny, nx)).copy()
    lon = np.broadcast_to(x, (ny, nx)).copy()
    return {
        "U": np.full((nz, ny, nx + 1), 8.0, np.float32),
        "V": np.full((nz, ny + 1, nx), 3.0, np.float32),
        "W": np.zeros((nz + 1, ny, nx), np.float32),
        "T": np.zeros((nz, ny, nx), np.float32),
        "P": np.zeros((nz, ny, nx), np.float32),
        "PB": pb,
        "PH": np.zeros((nz + 1, ny, nx), np.float32),
        "PHB": phb,
        "QVAPOR": np.full((nz, ny, nx), 0.004, np.float32),
        "QCLOUD": np.zeros((nz, ny, nx), np.float32),
        "QRAIN": np.zeros((nz, ny, nx), np.float32),
        "MU": np.zeros((ny, nx), np.float32),
        "MUB": np.full((ny, nx), 82500.0, np.float32),
        "HGT": np.zeros((ny, nx), np.float32),
        "PSFC": np.full((ny, nx), 100000.0, np.float32),
        "TSK": np.full((ny, nx), 291.0, np.float32),
        "T2": np.full((ny, nx), 290.0, np.float32),
        "TH2": np.full((ny, nx), 290.0, np.float32),
        "Q2": np.full((ny, nx), 0.004, np.float32),
        "U10": np.full((ny, nx), 7.0, np.float32),
        "V10": np.full((ny, nx), 2.0, np.float32),
        "RAINNC": np.zeros((ny, nx), np.float32),
        "XLAT": lat,
        "XLONG": lon,
        "MAPFAC_M": np.ones((ny, nx), np.float32),
        "MAPFAC_U": np.ones((ny, nx + 1), np.float32),
        "MAPFAC_V": np.ones((ny + 1, nx), np.float32),
        "F": np.full((ny, nx), 8.0e-5, np.float32),
        "SINALPHA": np.zeros((ny, nx), np.float32),
        "COSALPHA": np.ones((ny, nx), np.float32),
        "LU_INDEX": np.full((ny, nx), 10.0, np.float32),
    }


def test_monthly_interp_to_date_pins_wrf_midmonth_weights():
    """WRF monthly_interp_to_date (module_initialize_real.F:8023-8089).

    Mid-month anchors on the 15th, whole-Julian-day weighting, and the
    Dec/Jan wrap through fictitious anchors 31 days outside the year.
    Hand-computed April 3 1974: Mar 15 = day 74, Apr 15 = day 105,
    target = day 93, so out = (12*March + 19*April) / 31.
    """
    from gpuwm.static.build import monthly_interp_to_date

    monthly = np.arange(12, dtype=np.float64)[:, None, None] * np.ones((1, 2, 3))

    april3 = monthly_interp_to_date(monthly, datetime(1974, 4, 3, 12))
    np.testing.assert_allclose(
        april3, (12.0 * monthly[2] + 19.0 * monthly[3]) / 31.0,
        rtol=0.0, atol=1e-15)
    # The clock time is discarded (get_julgmt whole days, :8065-8066).
    np.testing.assert_array_equal(
        april3, monthly_interp_to_date(monthly, datetime(1974, 4, 3, 23)))
    # A mid-month anchor returns that month exactly (interval is
    # middle(l) < target <= middle(l+1), :8068).
    np.testing.assert_array_equal(
        monthly_interp_to_date(monthly, datetime(1974, 4, 15)), monthly[3])
    # January wrap: middle(0) = Jan15 - 31 (:8060), so Jan 10 blends
    # (target-middle(0)) = 26 parts January with (15-10) = 5 parts December.
    np.testing.assert_allclose(
        monthly_interp_to_date(monthly, datetime(1974, 1, 10)),
        (26.0 * monthly[0] + 5.0 * monthly[11]) / 31.0, rtol=0.0, atol=1e-15)
    # December wrap: middle(13) = Dec15 + 31 (:8063); Dec 20 (day 354 vs
    # Dec 15 = day 349) blends 5 parts January with 26 parts December.
    np.testing.assert_allclose(
        monthly_interp_to_date(monthly, datetime(1974, 12, 20)),
        (5.0 * monthly[0] + 26.0 * monthly[11]) / 31.0, rtol=0.0, atol=1e-15)
    # Leap year: 1976 Mar 1 = day 61, Feb 15 = day 46, Mar 15 = day 75.
    np.testing.assert_allclose(
        monthly_interp_to_date(monthly, datetime(1976, 3, 1)),
        (14.0 * monthly[1] + 15.0 * monthly[2]) / 29.0, rtol=0.0, atol=1e-15)


@requires_bundle
@requires_gpu
@pytest.mark.gpu
def test_prepared_case_seeds_wrf_surface_climatology_and_landuse_table():
    """GREENFRAC/LAI interpolate while ALBBCK comes from LANDUSE.TBL.

    WRF interpolates GREENFRAC/ALBEDO12M/LAI12M to the run date
    (module_initialize_real.F:1322-1335) before the percent/fraction
    scalings (:1360-1377), but landuse_init overwrites ALBBCK when
    usemonalb=false (module_physics_init.F:1941-1995).  shdmin/shdmax stay
    the monthly min/max (:1348-1351).  April 3 1974 sits 19/31 of the way
    from the March to the April mid-month anchors.
    """
    import cupy as cp
    from gpuwm.core.landuse import load_landuse_table
    from gpuwm.verify.cases.real74_d01 import prepare_phase3_case

    prepared = prepare_phase3_case()
    static = prepared.static_fields
    fields = prepared.initial_result.state.physics.fields

    def blend(name):
        monthly = np.asarray(static[name], dtype=np.float64)
        return (12.0 * monthly[2] + 19.0 * monthly[3]) / 31.0

    np.testing.assert_allclose(
        cp.asnumpy(fields["vegfra"]), 100.0 * blend("GREENFRAC"),
        rtol=0.0, atol=5.0e-3)
    np.testing.assert_allclose(
        cp.asnumpy(fields["albbck"]),
        load_landuse_table().values[
            1, cp.asnumpy(fields["ivgtyp"]).astype(np.int32) - 1, 0] / 100.0,
        rtol=0.0, atol=5.0e-8)
    np.testing.assert_allclose(
        cp.asnumpy(fields["lai"]), blend("LAI12M"), rtol=0.0, atol=5.0e-5)
    np.testing.assert_allclose(
        cp.asnumpy(fields["shdmin"]),
        100.0 * np.asarray(static["GREENFRAC"]).min(axis=0),
        rtol=0.0, atol=5.0e-3)
    np.testing.assert_allclose(
        cp.asnumpy(fields["shdmax"]),
        100.0 * np.asarray(static["GREENFRAC"]).max(axis=0),
        rtol=0.0, atol=5.0e-3)


def test_phase3_real74_configuration_is_exact_plan_domain():
    from gpuwm.verify.cases.real74_d01 import (
        DYNAMICS_SUBSTEPS,
        ETA_LEVELS,
        phase3_config,
        phase3_integration_config,
    )

    cfg = phase3_config()
    integration_cfg = phase3_integration_config(cfg)
    assert (cfg.nx, cfg.ny, cfg.nz) == (250, 200, 49)
    assert (cfg.dx, cfg.dy, cfg.dt, cfg.run_seconds) == (
        12000.0, 12000.0, 60.0, 43200.0)
    # Ratified 2026-07-16: NATIVE dt=60 (the reference run's timestep);
    # the 8-substep compatibility mode is retired (h_diabatic + p_hyd
    # physics wiring + reference emdiv=0.01 restored native stability).
    assert DYNAMICS_SUBSTEPS == 1
    assert integration_cfg.dt == 60.0
    assert integration_cfg.clock_dt == 60.0
    assert cfg.dt / integration_cfg.dt == DYNAMICS_SUBSTEPS
    assert cfg.emdiv == 0.01
    assert ETA_LEVELS.shape == (50,)
    assert cfg.mp_physics == 10
    assert cfg.sf_sfclay_physics in (1, 91)
    assert cfg.sf_surface_physics == 2
    assert cfg.bl_pbl_physics == 1
    assert cfg.ra_physics == 4
    assert cfg.radt == 12.0
    # Task 6b: bundle-namelist KF on d01 (cu_physics=1, cudt=5 minutes).
    assert cfg.cu_physics == 1
    assert cfg.cudt_minutes == 5.0
    assert cfg.specified and cfg.spec_bdy_width == 5
    assert cfg.w_damping == 1


def test_dynamics_substeps_is_frozen_and_absent_from_experiment_path(
        monkeypatch):
    """D1: only the frozen profile owns the retired constant.

    Generic package code may retain loop bookkeeping, but no package module
    outside the frozen profile may import or read DYNAMICS_SUBSTEPS.
    """
    from gpuwm.verify.cases import real74_d01

    assert real74_d01.DYNAMICS_SUBSTEPS == 1
    repo = Path(__file__).resolve().parents[1]
    owner = repo / "gpuwm" / "verify" / "cases" / "real74_d01.py"
    package_paths = [
        path for path in (repo / "gpuwm").rglob("*.py")
        if path != owner
    ]
    for path in package_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        reads = [node for node in ast.walk(tree)
                 if ((isinstance(node, ast.Name)
                      and node.id == "DYNAMICS_SUBSTEPS")
                     or (isinstance(node, ast.Attribute)
                         and node.attr == "DYNAMICS_SUBSTEPS"))]
        assert not reads, f"package code reads the frozen pin: {path}"

    monkeypatch.setattr(real74_d01, "DYNAMICS_SUBSTEPS", 2)
    with pytest.raises(ValueError, match="unsupported configuration"):
        real74_d01.phase3_integration_config(real74_d01.phase3_config())


def test_reflectivity_due_only_on_final_step_before_each_output():
    """The pure history predicate launches one refl kernel per frame."""
    from gpuwm.verify.cases.real74_d01 import _refl_10cm_due

    due = [
        (outer, substep)
        for outer in range(8)
        for substep in range(3)
        if _refl_10cm_due(
            outer, substep, output_outer_steps=4, dynamics_substeps=3)
    ]
    assert due == [(3, 2), (7, 2)]


def test_run_config_validation_accepts_non_12h_duration(monkeypatch,
                                                         tmp_path):
    """The normal run surface accepts a whole-step forecast below 12 h."""
    from gpuwm.verify.cases import real74_d01

    cfg = replace(
        real74_d01.phase3_config(), dt=30.0, time_step_sound=6,
        run_seconds=3600.0, output_interval_s=900.0)
    expected = SimpleNamespace(wrfout_paths=())
    calls = []
    monkeypatch.setattr(real74_d01, "_case_grid", lambda loaded: object())
    monkeypatch.setattr(
        real74_d01, "_integrate_configured_case",
        lambda output_dir, loaded, restart_path=None: calls.append(
            (output_dir, loaded, restart_path))
        or (object(), expected))

    assert real74_d01.run_config(cfg, tmp_path) is expected
    # Task 8: the restart pass-through defaults to a cold start.
    assert calls == [(tmp_path, cfg, None)]
    integration_cfg = real74_d01.phase3_integration_config(cfg)
    # Native ratification: the integration transform is the identity on
    # dt (DYNAMICS_SUBSTEPS=1); the clock is pinned to the outer dt.
    assert integration_cfg.dt == cfg.dt
    assert integration_cfg.clock_dt == cfg.dt
    assert integration_cfg.time_step_sound == 6


def test_clear_sky_proxy_has_diurnal_peak_night_zero_and_brunt_glw():
    """The retained ra=90 diagnostic proxy remains independently selectable."""
    from gpuwm.core.analytic_radiation import analytic_clear_sky_forcing

    lat = np.full((2, 3), 40.0, dtype=np.float64)
    lon = np.full((2, 3), -100.0, dtype=np.float64)
    temperature = np.full((2, 3), 290.0, dtype=np.float64)
    qv = np.full((2, 3), 0.010, dtype=np.float64)
    pressure = np.full((2, 3), 90000.0, dtype=np.float64)

    start = datetime(1974, 4, 3, 12)
    times = [start + timedelta(hours=hour) for hour in range(13)]
    shortwave = [
        analytic_clear_sky_forcing(
            valid, lat, lon, temperature, qv, pressure)[0]
        for valid in times
    ]
    means = np.asarray([field.mean() for field in shortwave])
    peak_time = times[int(np.argmax(means))]
    assert datetime(1974, 4, 3, 18) <= peak_time <= datetime(1974, 4, 3, 20)
    assert 800.0 < float(means.max()) < 1100.0

    night_swdown, glw = analytic_clear_sky_forcing(
        datetime(1974, 4, 3, 6), lat, lon, temperature, qv, pressure)
    np.testing.assert_array_equal(night_swdown, np.zeros_like(lat))
    vapor_pressure_hpa = 0.010 * 90000.0 / (0.622 + 0.010) / 100.0
    emissivity = 0.70 + 0.09 * np.log10(vapor_pressure_hpa)
    expected_glw = emissivity * 5.670374419e-8 * 290.0 ** 4
    np.testing.assert_allclose(glw, expected_glw, rtol=1.0e-12, atol=0.0)


@requires_bundle
@requires_gpu
@pytest.mark.gpu
def test_prepared_case_runs_production_rrtmgp_surface_forcing():
    import cupy as cp

    from gpuwm.core.rrtmgp import RRTMGPRadiation
    from gpuwm.verify.cases.real74_d01 import prepare_phase3_case

    prepared = prepare_phase3_case()
    state = prepared.initial_result.state
    fields = state.physics.fields
    assert isinstance(state.physics.radiation_callable,
                      RRTMGPRadiation)
    assert bool(cp.isfinite(state.p).all())
    assert float(state.p.min()) > 0.0
    state.physics.compute(state, prepared.cfg)
    assert state.physics.call_counts["radiation"] == 1
    assert bool(cp.isfinite(fields["swdown"]).all())
    assert bool(cp.isfinite(fields["glw"]).all())
    assert 100.0 < float(fields["glw"].min())


def test_real_wrfout_metadata_and_wrf_tooling_getvar(tmp_path):
    wrf = pytest.importorskip("wrf")
    from gpuwm.verify.cases.real74_d01 import (
        check_wrf_tooling_compatibility,
    )

    path = tmp_path / "wrfout_d01_1974-04-03_13_00_00"
    nz, ny, nx = 8, 5, 6
    attrs = {
        "MAP_PROJ": 1,
        "TRUELAT1": 30.0,
        "TRUELAT2": 60.0,
        "STAND_LON": -83.9297,
        "MOAD_CEN_LAT": 38.0,
        "CEN_LAT": 38.0,
        "CEN_LON": -97.0,
        "POLE_LAT": 90.0,
        "POLE_LON": 0.0,
    }
    with WrfoutWriter(
            path, nx=nx, ny=ny, nz=nz, dx=12000.0, dy=12000.0,
            global_attrs=attrs) as writer:
        writer.write_frame(
            "1974-04-03_13:00:00", _synthetic_real_frame(nz, ny, nx))

    with netCDF4.Dataset(path) as ds:
        assert int(ds.MAP_PROJ) == 1
        assert float(ds.TRUELAT1) == 30.0
        for name in ("XLAT", "XLONG", "MAPFAC_M", "MAPFAC_U",
                     "MAPFAC_V", "F", "HGT", "LU_INDEX", "TSK", "T2",
                     "U10", "V10", "RAINNC", "P", "PB", "PSFC"):
            assert name in ds.variables

    diagnostics, backend_identity = check_wrf_tooling_compatibility(path)
    assert set(diagnostics) == {"slp", "tk", "uvmet10", "td2", "cape_2d"}
    assert backend_identity == (
        f"module={wrf.__name__}; file={Path(wrf.__file__).resolve()}; "
        f"version={wrf.__version__}"
    )


def test_comparison_metrics_apply_task13_thresholds():
    from gpuwm.verify.cases.real74_d01 import (
        HeadToHeadMetrics,
        task13_gate_failures,
    )

    good = HeadToHeadMetrics(
        temperature_rmse_k={500: 0.7, 700: 0.8, 850: 0.9},
        u_rmse_ms={500: 1.0, 700: 1.1, 850: 1.2},
        v_rmse_ms={500: 1.1, 700: 1.2, 850: 1.3},
        mslp_pattern_correlation=0.985,
    )
    assert task13_gate_failures(good) == []
    bad = HeadToHeadMetrics(
        temperature_rmse_k={500: 1.01, 700: 0.8, 850: 0.9},
        u_rmse_ms={500: 1.0, 700: 2.01, 850: 1.2},
        v_rmse_ms={500: 1.1, 700: 1.2, 850: 1.3},
        mslp_pattern_correlation=0.979,
    )
    failures = task13_gate_failures(bad)
    assert len(failures) == 3
    assert any("T500" in item for item in failures)
    assert any("U700" in item for item in failures)
    assert any("MSLP" in item for item in failures)


def test_era5_metrics_scope_t2_bias_to_initial_snow_free_cells(monkeypatch):
    from gpuwm.verify.cases import real74_d01

    model_t2 = np.array([[210.0, 280.0], [285.0, 290.0]])
    analysis_t2 = np.array([[270.0, 290.0], [290.0, 295.0]])
    initial_swe = np.array([[2.0, 0.0], [1.0, 5.0]])
    monkeypatch.setattr(
        real74_d01,
        "_wrf_diagnostics",
        lambda _path: {
            "levels": {500: {"temperature": np.array([[250.0, 251.0]])}},
            "t2": model_t2,
        },
    )
    analysis = SimpleNamespace(
        levels_hpa=np.array([500.0]),
        fields={
            "TT": np.array([[[250.0, 251.0]]]),
            "T2": analysis_t2,
        },
    )

    metrics = real74_d01._era5_00z_metrics(
        Path("unused"), analysis, initial_swe)

    assert metrics.initial_snow_cell_count == 2
    assert metrics.initial_snow_free_cell_count == 2
    assert metrics.t2_min_k == 210.0
    assert metrics.t2_mean_bias_k == -20.0
    assert metrics.t2_snow_free_mean_bias_k == -7.5
    assert metrics.t2_snow_cell_mean_bias_k == -32.5


def test_era5_metrics_do_not_mask_candidate_t2_nan(monkeypatch):
    from gpuwm.verify.cases import real74_d01

    model_t2 = np.array([[280.0, np.nan], [285.0, 290.0]])
    analysis_t2 = np.array([[280.0, 281.0], [285.0, np.nan]])
    monkeypatch.setattr(
        real74_d01,
        "_wrf_diagnostics",
        lambda _path: {
            "levels": {500: {"temperature": np.array([[250.0, 251.0]])}},
            "t2": model_t2,
        },
    )
    analysis = SimpleNamespace(
        levels_hpa=np.array([500.0]),
        fields={
            "TT": np.array([[[250.0, 251.0]]]),
            "T2": analysis_t2,
        },
    )

    result = real74_d01._era5_00z_metrics(
        Path("unused"), analysis, np.zeros((2, 2)))

    assert np.isnan(result.t2_domain_mean_k)
    assert np.isnan(result.t2_snow_free_mean_bias_k)


def test_identical_wrfout_comparison_and_required_maps(tmp_path):
    from gpuwm.verify.cases.real74_d01 import (
        compare_head_to_head,
        make_task13_maps,
    )

    nz, ny, nx = 8, 5, 6
    attrs = {
        "MAP_PROJ": 1, "TRUELAT1": 30.0, "TRUELAT2": 60.0,
        "STAND_LON": -83.9297, "MOAD_CEN_LAT": 38.0,
        "CEN_LAT": 38.0, "CEN_LON": -97.0,
        "POLE_LAT": 90.0, "POLE_LON": 0.0,
    }
    paths = []
    for hour, rain in ((6, 1.0), (12, 4.5)):
        path = tmp_path / f"wrfout_d01_1974-04-04_{hour:02d}_00_00"
        frame = _synthetic_real_frame(nz, ny, nx)
        frame["RAINNC"] = np.full((ny, nx), rain, np.float32)
        with WrfoutWriter(
                path, nx=nx, ny=ny, nz=nz, dx=12000.0, dy=12000.0,
                global_attrs=attrs) as writer:
            writer.write_frame(
                f"1974-04-04_{hour:02d}:00:00", frame)
        paths.append(path)

    metrics = compare_head_to_head(paths[-1], paths[-1])
    assert metrics.temperature_rmse_k == {500: 0.0, 700: 0.0, 850: 0.0}
    assert metrics.u_rmse_ms == {500: 0.0, 700: 0.0, 850: 0.0}
    assert metrics.v_rmse_ms == {500: 0.0, 700: 0.0, 850: 0.0}
    assert metrics.mslp_pattern_correlation == pytest.approx(1.0)

    maps = make_task13_maps(paths[-1], paths[0], tmp_path / "maps")
    assert {path.name for path in maps} == {
        "real74_mslp_t2.png",
        "real74_500hpa.png",
        "real74_precip_6h.png",
    }
    assert all(path.is_file() and path.stat().st_size > 1000 for path in maps)


@requires_bundle
@requires_gpu
@pytest.mark.gpu
@pytest.mark.slow_acceptance
def test_real74_d01_full_task13_acceptance(tmp_path):
    from gpuwm.verify.cases.real74_d01 import run_phase3_case, summary_metrics

    # The complete gate suite belongs only to ``gpuwm verify real74_d01``;
    # the normal config-driven run surface performs no oracle comparisons.
    summary = run_phase3_case(tmp_path)
    assert_gates("real74_d01", summary_metrics(summary))
    # Native dt=60 ratified 2026-07-16 (compat 8-substep mode retired).
    assert summary.dynamics_substeps == 1
    assert np.isfinite(summary.w_max_ms) and summary.w_max_ms > 0.0
    assert 0.0 < summary.interior_w_max_ms <= summary.w_max_ms
    # Task 6c eliminated the relaxation-row w spike; the domain maximum now
    # lives in the free interior (the old pin documented the defect).
    assert summary.w_max_boundary_row is None
    assert summary.boundary_w_max_ms < summary.interior_w_max_ms
    # radt=12 binds at Task 6a (plan-ratified): 12 h / 12 min = 60 updates.
    # The retired 720 pin encoded the interim radt=1 proxy cadence.
    assert summary.surface_forcing_updates == 60
    assert datetime(1974, 4, 3, 18) <= summary.swdown_peak_time <= datetime(1974, 4, 3, 21)
    assert len(summary.wrfout_paths) == 13
    assert all(path.is_file() for path in summary.wrfout_paths)
    assert np.isfinite(summary.era5_00z.t500_rmse_k)
    assert np.isfinite(summary.era5_00z.t500_pattern_correlation)
    assert np.isfinite(summary.era5_00z.t2_snow_cell_mean_bias_k)
    # Phase 4 BINDING flagship requirement beyond the GATES thresholds
    # (plan :87-89): the forecast must BEAT the interior-convention
    # persistence baselines, not merely clear the absolute bounds.
    from gpuwm.verify.cases.real74_d01 import (
        ERA5_PERSISTENCE_T500_CORRELATION, ERA5_PERSISTENCE_T500_RMSE_K)
    assert summary.era5_00z.t500_rmse_k < ERA5_PERSISTENCE_T500_RMSE_K
    assert (summary.era5_00z.t500_pattern_correlation
            > ERA5_PERSISTENCE_T500_CORRELATION)
    assert set(summary.wrf_tooling_diagnostics) == {
        "slp", "tk", "uvmet10", "td2", "cape_2d"}
    import wrf
    assert summary.wrf_tooling_backend_identity == (
        f"module={wrf.__name__}; file={Path(wrf.__file__).resolve()}; "
        f"version={wrf.__version__}"
    )
    assert len(summary.map_paths) == 3
    assert all(path.is_file() for path in summary.map_paths)


@requires_bundle
@requires_gpu
@pytest.mark.gpu
@pytest.mark.slow_acceptance
def test_real74_d01_kf_one_hour_smoke_rains_in_the_warm_sector(tmp_path):
    """Task 6b gate: 1 h KF-active smoke is NaN-free with warm-sector RAINC.

    Extends the Task-1 wrfout RAINC regression to the real case: the +1 h
    wrfout must carry a RAINC field that is nonzero inside the April 3
    1974 warm sector (the pre-squall southeastern quadrant, lat 28-38N,
    lon 95-80W).
    """
    from gpuwm.config import load_config
    from gpuwm.verify.cases.real74_d01 import run_config

    base = Path(__file__).parents[1] / "configs" / "real74_d01.toml"
    config_path = tmp_path / "real74-kf-smoke.toml"
    config_path.write_text(
        base.read_text(encoding="utf-8").replace(
            "run_seconds = 43200.0", "run_seconds = 3600.0"),
        encoding="utf-8")
    cfg = load_config(config_path)
    assert cfg.cu_physics == 1 and cfg.cudt_minutes == 5.0

    summary = run_config(cfg, tmp_path / "run")

    assert summary.nan_free
    assert summary.ysu_nan_guard_fires == 0
    assert summary.completed_seconds == pytest.approx(3600.0)
    assert summary.rainc_max_mm > 0.0
    assert summary.rainc_max_ji is not None
    with netCDF4.Dataset(summary.wrfout_paths[-1]) as ds:
        rainc = np.asarray(ds.variables["RAINC"][0], dtype=np.float64)
        lat = np.asarray(ds.variables["XLAT"][0], dtype=np.float64)
        lon = np.asarray(ds.variables["XLONG"][0], dtype=np.float64)
    assert np.isfinite(rainc).all()
    # WRF's PRATEC = PPTFLX*(1-FBFRC)/DXSQ is UNCLAMPED (kfeta.F:2504)
    # and PPTFLX = TRPPT - TDER (:1821) goes epsilon-negative in FP32
    # when evaporation nearly balances the precip flux, so RAINC can
    # carry nano-negative cells exactly as WRF's would.  Physical
    # non-negativity holds to FP noise; real negative rain would trip
    # the mm-scale bound.
    assert rainc.min() >= -1.0e-6
    warm_sector = ((lat >= 28.0) & (lat <= 38.0)
                   & (lon >= -95.0) & (lon <= -80.0))
    assert warm_sector.any()
    assert float(rainc[warm_sector].max()) > 0.0


@requires_bundle
@requires_gpu
@pytest.mark.gpu
def test_run_config_short_forecast_honors_duration_and_output_cadence(
        tmp_path):
    from gpuwm.config import load_config
    from gpuwm.verify.cases.real74_d01 import run_config

    config_path = tmp_path / "real74-short.toml"
    base = Path(__file__).parents[1] / "configs" / "real74_d01.toml"
    config_text = base.read_text(encoding="utf-8")
    config_path.write_text(
        config_text.replace("run_seconds = 43200.0", "run_seconds = 120.0")
        .replace("output_interval_s = 3600.0",
                 "output_interval_s = 120.0"),
        encoding="utf-8")
    cfg = load_config(config_path)
    output_dir = tmp_path / "run"

    summary = run_config(cfg, output_dir)

    assert summary.nan_free
    assert summary.completed_seconds == pytest.approx(120.0)
    # radt=12 min: the mandatory ITIMESTEP=1 call is the sole radiation
    # update in this two-minute smoke forecast.
    assert summary.surface_forcing_updates == 1
    # Native dt=60 ratified 2026-07-16 (compat 8-substep mode retired).
    assert summary.dynamics_substeps == 1
    assert [path.name for path in summary.wrfout_paths] == [
        "wrfout_d01_1974-04-03_12_00_00",
        "wrfout_d01_1974-04-03_12_02_00",
    ]
    assert all(path.is_file() and path.stat().st_size > 0
               for path in summary.wrfout_paths)
    assert all(path.parent == output_dir for path in summary.wrfout_paths)
    for path in summary.wrfout_paths:
        with netCDF4.Dataset(path) as dataset:
            for name in ("U", "V", "W", "T", "P", "QVAPOR", "T2"):
                assert np.isfinite(dataset.variables[name][:]).all(), name


def test_t500_scoring_excludes_the_davies_frame():
    """The gate metric scores the free interior only: frame-poisoned
    fields must not move the score (reverting _score_t500 to full-domain
    scoring fails this), and the derivation tool shares the same region
    helper so the persistence baseline cannot fork conventions."""
    from gpuwm.verify.cases.real74_d01 import (
        _score_t500, interior_region)

    truth = np.linspace(250.0, 260.0, 20 * 24).reshape(20, 24)
    model = truth.copy()
    model[:5, :] += 50.0
    model[:, :5] -= 50.0
    model[-5:, :] += 50.0
    model[:, -5:] -= 50.0
    rmse, corr = _score_t500(model, truth)
    assert rmse == 0.0
    assert corr == 1.0
    assert interior_region(model.shape) == (slice(5, -5), slice(5, -5))
    # Tiny synthetic domains fall back to full-domain scoring.
    assert interior_region((8, 8)) == (slice(None), slice(None))
    # The generalized persistence tool derives every case through the SAME
    # reusable scorer, so metric and baseline conventions cannot fork.
    tool = Path(__file__).resolve().parents[1] / "tools" \
        / "derive_persistence_baseline.py"
    source = tool.read_text(encoding="utf-8")
    assert "score_pair" in source
    assert "derive_case_persistence" in source


def test_dcomputeseaprs_hand_pins_and_locality():
    """Hand-derived pins for the DCOMPUTESEAPRS transcription (R16).

    Expected values computed by hand from the Shuell 1995 algorithm as
    the wrf-rust oracle writes it (pressure.rs:60-147): klo search,
    log-p interpolation, USSALR extrapolation, TC capping, reduction.
    """
    from gpuwm.verify.cases.real74_d01 import _dcomputeseaprs

    def cols(a):
        return np.asarray(a, dtype=np.float64)[:, None, None]

    # Sea-level surface: the reduction exponent vanishes, so
    # slp == 0.01 * p_sfc EXACTLY, independent of temperatures.
    slp = _dcomputeseaprs(
        cols([100000.0, 92000.0, 85000.0, 78000.0, 65000.0, 50000.0]),
        cols([288.0, 284.0, 280.0, 276.0, 268.0, 255.0]),
        cols([0.005] * 6),
        cols([0.0, 700.0, 1400.0, 2100.0, 3400.0, 5500.0]))
    assert slp.shape == (1, 1)
    assert slp[0, 0] == pytest.approx(1000.0, abs=1.0e-9)

    # Elevated dry column, hand-worked end to end: klo=2/khi=1,
    # frac=0.6571857, t_at_pconst=278.37126 K, z_at_pconst=2027.1707 m,
    # t_surf=285.53040 K, t_sea_level 291.54787 -> capped at 290.66,
    # slp = 800*exp(19620/(287*576.19040)) = 900.7765 hPa.
    p = [80000.0, 74000.0, 68000.0, 62000.0, 56000.0, 50000.0]
    t = [285.0, 281.0, 277.0, 273.0, 269.0, 265.0]
    z = [1000.0, 1600.0, 2250.0, 2900.0, 3600.0, 4300.0]
    q = [0.0] * 6
    slp = _dcomputeseaprs(cols(p), cols(t), cols(q), cols(z))
    assert slp[0, 0] == pytest.approx(900.7765, abs=0.02)

    # Locality: only the surface and the two bracketing levels enter;
    # perturbing levels above the reference must not move the answer.
    t2 = list(t)
    t2[4] += 15.0
    t2[5] -= 15.0
    slp2 = _dcomputeseaprs(cols(p), cols(t2), cols(q), cols(z))
    np.testing.assert_array_equal(slp, slp2)


@pytest.mark.skipif(
    not (BUNDLE / "wrfout_reference"
         / "wrfout_d01_1974-04-03_13_00_00").is_file(),
    reason="WRF_1974_MP55 reference bundle not present")
def test_reference_wrfout_mslp_is_synoptically_sane():
    """DCOMPUTESEAPRS over the real reference wrfout: hPa range and a
    deep-cyclone signature (the Super Outbreak parent low)."""
    from gpuwm.verify.cases.real74_d01 import REFERENCE_13Z, _wrf_diagnostics

    mslp = _wrf_diagnostics(REFERENCE_13Z)["mslp"]
    assert np.isfinite(mslp).all()
    assert 940.0 < float(mslp.min()) < 1000.0
    assert 990.0 < float(mslp.max()) < 1060.0
