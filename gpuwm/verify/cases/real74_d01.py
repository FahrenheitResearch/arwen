"""April 3 1974 d01 FROZEN data/validation profile (Task 8 acceptance gate).

Phase 5, Task 2: the runtime machinery (prepare/integrate/output) moved
to :mod:`gpuwm.runtime`; this module is the frozen real74 profile -- the
pinned inputs (bundle paths, start time, forcing times, eta levels,
policies), the pinned GATES/persistence baselines, and the Task-13
oracle comparisons.  Every pinned value and gate is byte-untouched; the
profile feeds its pins into the generic runtime as declared data, and
frozen verify cases never migrate to the experiment path (they construct
``RunConfig`` directly).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from dataclasses import dataclass, replace
import os
from pathlib import Path

import numpy as np

from gpuwm import runtime
from gpuwm.case_data import SourceOrography
from gpuwm.config import RunConfig
from gpuwm.core.grid import make_vertical_coord
from gpuwm.core.rrtmgp import RRTMGPRadiation
from gpuwm.experiment import VerticalConfig
from gpuwm.physics_compat import RRTMG_VARIANT_LEGACY, rrtmg_variant
from gpuwm.ingest.grib import cached_era5_snapshots
from gpuwm.ingest.horiz import interpolate_era5_to_lambert
from gpuwm.ingest.lateral_bc import (attach_lateral_boundaries,
                                     build_state_lateral_boundaries)
from gpuwm.ingest.real import initialize_real
from gpuwm.runtime import RealCaseRunSummary
from gpuwm.static.build import build_static
from gpuwm.static.lambert import grids_from_wps_namelist
from gpuwm.verify.metrics import (
    ReflectivityMapSpec, SynopticMapSpec, _dcomputeseaprs,  # noqa: F401
    _interpolate_to_pressure, _pattern_correlation, _rmse,  # noqa: F401
    _wrf_diagnostics, interior_region, make_composite_reflectivity_map,
    make_synoptic_maps, score_pair)
from gpuwm.verify.profiles import (
    AnalysisProfile, AnalysisRecipe, HealthProfile, OracleProfile,
    Threshold, VerificationProfile)


BUNDLE = Path(os.environ.get(
    "GPUWM_REAL74_REFERENCE_BUNDLE",
    Path.home() / "Downloads" / "WRF_1974_MP55_reference_bundle",
))
START_TIME = datetime(1974, 4, 3, 12)
SOURCE_OROGRAPHY = SourceOrography(
    path=BUNDLE / "met_em" / "met_em.d01.1974-04-03_12_00_00.nc",
    variable="SOILHGT")
REFERENCE_13Z = (BUNDLE / "wrfout_reference" /
                 "wrfout_d01_1974-04-03_13_00_00")
#: Ratified 2026-07-16: NATIVE dt=60 (the reference run's own timestep,
#: DYNAMICS_SUBSTEPS=1).  The 8-substep compatibility mode was the
#: Phase-3 stability workaround; the native blowups traced to (a) the
#: missing h_diabatic mechanism (e72db7d), (b) physics consuming the EOS
#: pressure where WRF feeds phy_prep's hydrostatic p_hyd (fda6fc0), and
#: (c) emdiv=0.0 where the reference ran the Registry default 0.01.
#: With all three fixed, native 12 h completes NaN-free and beats the
#: 8-substep mode on every gate metric (native-dt60-final-12h.log:
#: T500 0.6382 K/0.9899 interior, +1 h MSLP corr 0.9979, wall 114.9 s).
DYNAMICS_SUBSTEPS = 1
#: 12Z-vs-00Z ERA5 TT persistence at the 500 hPa source level, scored with
#: the SAME convention as the +12 h gate metric (interior slice(5, -5),
#: centered).  Derived in-repo by tools/derive_persistence_baseline.py;
#: rerun it whenever the scoring convention changes.
ERA5_PERSISTENCE_T500_RMSE_K = 2.9918
ERA5_PERSISTENCE_T500_CORRELATION = 0.8621
#: Frozen real74 data-driven profiles.  The +12 h T500 skill thresholds
#: are the PHASE 4 BINDING flagship gates (plan :87-89), scored on the
#: dynamically free interior, and the acceptance test additionally requires
#: beating the ERA5_PERSISTENCE_* baselines.  External WRF comparisons are
#: isolated in the optional oracle layer; ordinary experiment runs consume
#: neither it nor these case-specific constants.
REAL74_HEALTH_PROFILE = HealthProfile(
    name="real74-d01-health",
    thresholds=(
        Threshold("boundary_zone_blowup", upper=0.5),
        Threshold("ysu_nan_guard_fires", upper=0.5),
    ),
)
REAL74_ANALYSIS_PROFILE = AnalysisProfile(
    name="real74-d01-era5-analysis",
    recipes=(
        AnalysisRecipe(
            START_TIME + timedelta(hours=12), "TT", 500.0, "interior",
            "rmse", Threshold("era5_t500_rmse_k", upper=2.5)),
        AnalysisRecipe(
            START_TIME + timedelta(hours=12), "TT", 500.0, "interior",
            "pattern_correlation",
            Threshold("era5_t500_pattern_correlation", lower=0.95)),
        AnalysisRecipe(
            START_TIME + timedelta(hours=12), "T2", None,
            "initial_snow_free", "absolute_mean_bias",
            Threshold("t2_snow_free_abs_bias_k", upper=10.0)),
        AnalysisRecipe(
            START_TIME + timedelta(hours=12), "T2", None, "full", "min",
            Threshold("t2_min_k", lower=200.0)),
    ),
)
REAL74_ORACLE_PROFILE = OracleProfile(
    name="real74-d01-wrf-oracle",
    reference_paths=((START_TIME + timedelta(hours=1), REFERENCE_13Z),),
    thresholds=(
        Threshold("temperature_rmse_max_k", upper=1.0),
        Threshold("wind_rmse_max_ms", upper=2.0),
        Threshold("mslp_pattern_correlation", lower=0.98),
        Threshold("wrf_tooling_diagnostic_count", lower=4.5),
    ),
    masks=("interior",),
)
REAL74_PROFILE = VerificationProfile(
    name="real74-d01-frozen",
    health=REAL74_HEALTH_PROFILE,
    analysis=REAL74_ANALYSIS_PROFILE,
    oracle=REAL74_ORACLE_PROFILE,
    gate_order=(
        "temperature_rmse_max_k", "wind_rmse_max_ms",
        "mslp_pattern_correlation", "boundary_zone_blowup",
        "t2_snow_free_abs_bias_k", "t2_min_k", "ysu_nan_guard_fires",
        "wrf_tooling_diagnostic_count", "era5_t500_rmse_k",
        "era5_t500_pattern_correlation",
    ),
)
GATES = REAL74_PROFILE.gates()
PHYSICS_AND_DYNAMICS_CAVEATS = (
    "The reference uses ISHMAEL microphysics and Shin-Hong PBL; gpuwm "
    "uses Morrison and YSU, so the +1 h comparison is interpreted "
    "as an early, dynamics/init-dominated gate.",
    "Kain-Fritsch cumulus (cu_physics=1, cudt=5 min) is active on d01 "
    "with WRF's per-column NCA hold; RAINC accumulates the held "
    "convective rain rate every model step.",
    "WRF moist cq pressure-gradient/acoustic factors (cqu/cqv/cqw) are "
    "omitted, a less-than-about-2-percent momentum effect at qtot~0.02.",
    "Constant-K and Smagorinsky horizontal diffusion are map-factor naive "
    "in Phase 3; sixth-order diffusion is WRF-exact with respect to map "
    "factors.",
    "The case integrates NATIVELY at the reference run's 60 s timestep "
    "(ratified 2026-07-16). The Phase-3 8-substep compatibility mode is "
    "retired: the native instabilities traced to the then-missing "
    "h_diabatic mechanism, physics consuming the EOS pressure where WRF "
    "feeds hydrostatic p_hyd, and emdiv=0.0 where the reference ran "
    "0.01; with all three fixed, native 12 h completes NaN-free and "
    "improves every gate metric.",
    "RTE+RRTMGP supplies both surface fluxes and atmospheric radiative "
    "heating on the WRF STEPRA clock.",
)
ETA_LEVELS = np.array([
    1.00000, 0.99780, 0.99519, 0.99212, 0.98849,
    0.98422, 0.97918, 0.97325, 0.96627, 0.95808,
    0.94846, 0.93719, 0.92402, 0.90866, 0.89079,
    0.87006, 0.84612, 0.81857, 0.78706, 0.75124,
    0.71080, 0.66556, 0.61547, 0.56067, 0.50519,
    0.45474, 0.40886, 0.36713, 0.32918, 0.29466,
    0.26328, 0.23473, 0.20877, 0.18516, 0.16369,
    0.14417, 0.12641, 0.11026, 0.09557, 0.08222,
    0.07007, 0.05902, 0.04898, 0.03984, 0.03153,
    0.02398, 0.01710, 0.01085, 0.00517, 0.00000,
], dtype=np.float64)


def config(run_seconds=1800.0) -> RunConfig:
    """Bundle d01 dynamics/BC configuration (physics remains off)."""
    return RunConfig(
        nx=250, ny=200, nz=49, dx=12000.0, dy=12000.0,
        # A conservative 30 s dynamics step is used for the Task 8
        # NaN/w acceptance run on the bundle's exceptionally thin lowest
        # eta layers; the boundary input cadence remains six hours.
        ztop=20000.0, dt=30.0, run_seconds=float(run_seconds),
        time_step_sound=4, epssm=0.5, hybrid_opt=2, etac=0.2,
        moist=True, moist_adv_opt=1, terrain_opt=1, map_proj=1,
        base_temp=290.0, specified=True, spec_bdy_width=5,
        spec_zone=1, relax_zone=4, km_opt=4, c_s=0.25,
        # Reference diff_6th_slopeopt=1 (namelist.input:94) with the
        # Registry-default 0.10 threshold: terrain-slope taper of the
        # 6th-order filter.
        diff_6th_opt=2, diff_6th_factor=0.12, diff_6th_slopeopt=1,
        w_damping=1, damp_opt=3, zdamp=5000.0, dampcoef=0.2,
        # Reference h_sca_adv_order=5 (WRF Registry.EM_COMMON:2872
        # default, unset in the reference namelist) for the rhs_ph
        # geopotential advection.
        h_sca_adv_order=5,
    )


def phase3_config(run_seconds=43200.0) -> RunConfig:
    """Task 13's exact d01 integration and Phase-3 physics selection.

    WRF ``e_we/e_sn/e_vert`` count staggered points, hence their
    251 x 201 x 50 namelist domain is represented by gpuwm's
    250 x 200 x 49 mass/full-level dimensions.
    """
    return RunConfig(
        nx=250, ny=200, nz=49, dx=12000.0, dy=12000.0,
        ztop=20000.0, dt=60.0, run_seconds=float(run_seconds),
        time_step_sound=4, epssm=0.5, hybrid_opt=2, etac=0.2,
        moist=True, mp_physics=10, moist_adv_opt=1, terrain_opt=1,
        map_proj=1, base_temp=290.0, specified=True, spec_bdy_width=5,
        spec_zone=1, relax_zone=4, km_opt=4, c_s=0.25,
        # Reference diff_6th_slopeopt=1 (namelist.input:94) with the
        # Registry-default 0.10 threshold.
        diff_6th_opt=2, diff_6th_factor=0.12, diff_6th_slopeopt=1,
        w_damping=1, damp_opt=3, zdamp=5000.0, dampcoef=0.2,
        # Reference h_sca_adv_order=5 (WRF Registry.EM_COMMON:2872
        # default, unset in the reference namelist) for the rhs_ph
        # geopotential advection.
        h_sca_adv_order=5,
        # Reference emdiv: the group ran the Registry default 0.01
        # (external-mode divergence damping; namelist does not set it).
        # Required for native-dt stability and ratified 2026-07-16 with
        # the native-dt=60 configuration.
        emdiv=0.01,
        # Reference hypsometric_opt=2 (Registry default; wrfout global
        # attr HYPSOMETRIC_OPT=2).  A/B-adjudicated 2026-07-16 on the
        # native config: stable, marginally better +12 h skill
        # (0.6340/0.9900 vs opt-1 0.6382/0.9899; hypso-opt2-12h.log).
        hypsometric_opt=2,
        sf_sfclay_physics=91, sf_surface_physics=2, bl_pbl_physics=1,
        ra_physics=4, cu_physics=1, radt=12.0, cudt_minutes=5.0,
        bldt=0.0, output_interval_s=3600.0,
        case="real74_d01",
    )


def phase3_integration_config(cfg: RunConfig) -> RunConfig:
    """Return the amended Task 13 uniform internal-step configuration."""
    if DYNAMICS_SUBSTEPS != 1:
        raise ValueError(
            "unsupported configuration: DYNAMICS_SUBSTEPS must equal 1; "
            "the historical substep mode was conservation-consistent but "
            "not WRF-equivalent")
    return replace(cfg, dt=cfg.dt / DYNAMICS_SUBSTEPS, clock_dt=cfg.dt)


@dataclass(frozen=True)
class HeadToHeadMetrics:
    """Task 13 +1 h residuals against the group's d01 WRF output."""

    temperature_rmse_k: dict[int, float]
    u_rmse_ms: dict[int, float]
    v_rmse_ms: dict[int, float]
    mslp_pattern_correlation: float


def task13_gate_failures(metrics: HeadToHeadMetrics) -> list[str]:
    """Return human-readable +1 h threshold violations."""
    failures = []
    temperature_limit = GATES["temperature_rmse_max_k"][1]
    wind_limit = GATES["wind_rmse_max_ms"][1]
    for level in (500, 700, 850):
        for label, values, limit in (
                ("T", metrics.temperature_rmse_k, temperature_limit),
                ("U", metrics.u_rmse_ms, wind_limit),
                ("V", metrics.v_rmse_ms, wind_limit)):
            value = float(values[level])
            if not np.isfinite(value) or value > limit:
                failures.append(
                    f"{label}{level} RMSE {value:.6g} exceeds {limit:g}")
    corr = float(metrics.mslp_pattern_correlation)
    correlation_limit = GATES["mslp_pattern_correlation"][0]
    if not np.isfinite(corr) or corr < correlation_limit:
        failures.append(
            f"MSLP pattern correlation {corr:.6g} is below "
            f"{correlation_limit:g}")
    return failures


def _score_t500(model: np.ndarray, source: np.ndarray) -> tuple[float, float]:
    """Interior-convention T500 skill pair (RMSE K, pattern correlation)."""
    return score_pair(model, source, mask="interior")


def compare_head_to_head(output_path, reference_path) -> HeadToHeadMetrics:
    """Compute +1 h residuals on the dynamically free domain interior.

    The outer five cells are WRF's specified-plus-relaxation lateral zone,
    not a freely forecast solution.  Limited-area head-to-head scores omit
    that zone; its health is graded independently by the boundary blowup
    gate in :func:`run_phase3_case`.
    """
    actual = _wrf_diagnostics(Path(output_path))
    reference = _wrf_diagnostics(Path(reference_path))
    interior = interior_region(actual["mslp"].shape)
    t_rmse = {}
    u_rmse = {}
    v_rmse = {}
    for level in (500, 700, 850):
        lhs = actual["levels"][level]
        rhs = reference["levels"][level]
        t_rmse[level] = _rmse(
            lhs["temperature"][interior], rhs["temperature"][interior])
        u_rmse[level] = _rmse(lhs["u"][interior], rhs["u"][interior])
        v_rmse[level] = _rmse(lhs["v"][interior], rhs["v"][interior])
    return HeadToHeadMetrics(
        temperature_rmse_k=t_rmse, u_rmse_ms=u_rmse, v_rmse_ms=v_rmse,
        mslp_pattern_correlation=_pattern_correlation(
            actual["mslp"][interior], reference["mslp"][interior]),
    )


_SYNOPTIC_MAP_SPEC = SynopticMapSpec(
    mslp_t2_filename="real74_mslp_t2.png",
    mslp_t2_title="gpuwm +12 h MSLP and 2 m temperature",
    height_wind_filename="real74_500hpa.png",
    height_wind_title="gpuwm +12 h 500 hPa height and grid-relative wind",
    precip_filename="real74_precip_6h.png",
    precip_title=("gpuwm forecast hours 6-12 total precipitation "
                  "(grid-scale + convective)"),
)


def make_task13_maps(final_path, six_hour_path, output_dir) -> tuple[Path, ...]:
    """Frozen real74 policy wrapper over the generic synoptic renderer."""
    return make_synoptic_maps(
        final_path, six_hour_path, output_dir, _SYNOPTIC_MAP_SPEC)


#: NWS RIDGE-style radar palette: 5 dBZ bins from 5 to 75 dBZ, the
#: standard operational composite-reflectivity ramp (cyan/blue light
#: precip, green/yellow moderate, orange/red convective cores,
#: magenta/purple extreme).
_NWS_REFL_BOUNDS = tuple(float(b) for b in range(5, 80, 5))
_NWS_REFL_COLORS = (
    "#04e9e7", "#019ff4", "#0300f4", "#02fd02", "#01c501",
    "#008e00", "#fdf802", "#e5bc00", "#fd9500", "#fd0000",
    "#d40000", "#bc0000", "#f800fd", "#9854c6",
)

_REFLECTIVITY_MAP_SPEC = ReflectivityMapSpec(
    filename="real74_composite_refl.png",
    title=("gpuwm simulated composite reflectivity "
           "(REFL_10CM column maximum)"),
    bounds=_NWS_REFL_BOUNDS,
    colors=_NWS_REFL_COLORS,
)


def make_reflectivity_map(wrfout_path, output_dir, *, time_index=0) -> Path:
    """Simulated composite (column-maximum) reflectivity PNG.

    Task 9 product hook, alongside :func:`make_task13_maps` but OPTIONAL:
    nothing in the current product path calls it, so Task 7 opts in once
    its wrfouts carry the REFL_10CM variable
    (the output-due microphysics-time stash merged into the output frame).
    The composite is the column maximum of REFL_10CM -- the standard
    NWS-style product -- shaded on the operational 5-75 dBZ palette with
    echoes below 5 dBZ left unshaded.
    """
    return make_composite_reflectivity_map(
        wrfout_path, output_dir, _REFLECTIVITY_MAP_SPEC,
        time_index=time_index)


def _rust_snapshots():
    """Decode all three forcing times through the production Rust bridge.

    Delegates to the resolved-input-keyed decode cache (Phase 5, Task 2;
    the argument-less ``lru_cache(1)`` pattern is retired): the decode
    itself runs once per process for this file/Vtable pair.
    """
    era5 = BUNDLE / "era5_grib"
    snapshots = cached_era5_snapshots(
        era5 / "era5_19740403.grb", era5 / "Vtable.ERA5_CDO")
    return {snapshot.valid_time: snapshot for snapshot in snapshots}


def _rust_snapshot(valid_time: datetime):
    try:
        return _rust_snapshots()[valid_time]
    except KeyError as exc:
        raise ValueError(f"ERA5 GRIB has no snapshot at {valid_time!s}") from exc


def build_forcing(run_seconds=1800.0):
    """Build 12/18/00Z real states and attach their LBCs to the 12Z state."""
    import netCDF4
    from gpuwm.config import validate_run_config

    # ``config`` is the retired dynamics-only Task-8 path.  With PBL off,
    # WRF's km_opt=4 branch also needs vertical_diffusion_2, which gpuwm has
    # not implemented.  The production phase3_config and every shipped
    # real74 TOML explicitly enable PBL and remain supported.
    cfg = validate_run_config(config(run_seconds))
    grid = grids_from_wps_namelist(BUNDLE / "namelists" / "namelist.wps")[0]
    with netCDF4.Dataset(BUNDLE / "geo_em" / "geo_em.d01.nc") as ds:
        terrain = np.asarray(ds.variables["HGT_M"][0], dtype=np.float64)
        map_fields = {
            name: np.asarray(ds.variables[name][0], dtype=np.float64)
            for name in ("MAPFAC_M", "MAPFAC_U", "MAPFAC_V", "F", "E",
                         "SINALPHA", "COSALPHA")
        }
    with netCDF4.Dataset(
            BUNDLE / "met_em" / "met_em.d01.1974-04-03_12_00_00.nc") as ds:
        source_orography = np.asarray(ds.variables["SOILHGT"][0],
                                      dtype=np.float64)
    times = (datetime(1974, 4, 3, 12), datetime(1974, 4, 3, 18),
             datetime(1974, 4, 4, 0))
    states = []
    results = []
    for valid_time in times:
        source = _rust_snapshot(valid_time)
        horizontal = interpolate_era5_to_lambert(source, grid)
        coord = make_vertical_coord(
            cfg.nz, hybrid_opt=cfg.hybrid_opt, etac=cfg.etac,
            eta_levels=ETA_LEVELS)
        result = initialize_real(
            horizontal, cfg, coord, terrain,
            source_orography=source_orography, p_top=10000.0,
            sfcp_to_sfcp=True)
        result.state.set_map_coriolis(
            map_fields["MAPFAC_M"], map_fields["MAPFAC_U"],
            map_fields["MAPFAC_V"], map_fields["F"], map_fields["E"],
            sina=map_fields["SINALPHA"], cosa=map_fields["COSALPHA"])
        results.append(result)
        states.append(result.state)
    boundaries = build_state_lateral_boundaries(states, times)
    attach_lateral_boundaries(states[0], boundaries)
    return cfg, results, boundaries


@dataclass(frozen=True)
class PreparedPhase3Case:
    """Setup-time inputs and live initial state for the 12 h integration."""

    cfg: RunConfig
    grid: object
    static_fields: dict[str, np.ndarray]
    initial_result: object
    analysis_00z: object
    initial_snow_water_kgm2: np.ndarray


@dataclass(frozen=True)
class Era5SanityMetrics:
    t500_rmse_k: float
    t500_pattern_correlation: float
    t2_domain_mean_k: float
    t2_analysis_mean_k: float
    t2_min_k: float
    t2_snow_free_mean_bias_k: float
    t2_snow_cell_mean_bias_k: float
    initial_snow_free_cell_count: int
    initial_snow_cell_count: int

    @property
    def t2_mean_bias_k(self) -> float:
        return self.t2_domain_mean_k - self.t2_analysis_mean_k


@dataclass(frozen=True)
class Phase3CaseSummary:
    wrfout_paths: tuple[Path, ...]
    map_paths: tuple[Path, ...]
    nan_free: bool
    w_max_ms: float
    boundary_w_max_ms: float
    interior_w_max_ms: float
    w_max_boundary_row: int | None
    boundary_zone_blowup: bool
    head_to_head: HeadToHeadMetrics
    head_to_head_failures: tuple[str, ...]
    era5_00z: Era5SanityMetrics
    wrf_tooling_diagnostics: tuple[str, ...]
    wrf_tooling_backend_identity: str
    caveats: tuple[str, ...]
    dynamics_substeps: int
    ysu_nan_guard_fires: int
    surface_forcing_updates: int
    swdown_peak_wm2: float
    swdown_peak_time: datetime


# RealCaseRunSummary moved to gpuwm/runtime.py with the integration loop
# (Phase 5, Task 2); re-exported above so the profile surface is unchanged.


#: Frozen output identity: the wrfout TITLE the profile has always
#: written (output identity left the runtime path with Task 2).
_OUTPUT_TITLE = "gpuwm Phase 3 April 3 1974 d01"


def prepare_phase3_case(cfg: RunConfig | None = None) -> PreparedPhase3Case:
    """Run Tasks 5-12 setup for the registered real74 d01 domain.

    Frozen-profile wrapper (Phase 5, Task 2): the setup pipeline lives in
    :func:`gpuwm.runtime.prepare_real_case`; this profile supplies its
    pinned inputs -- the BUNDLE paths, START_TIME plus the 6-hourly
    three-time forcing tuple, ETA_LEVELS/p_top as the declared vertical
    coordinate, ``sfcp_to_sfcp=True``, and the met_em SOILHGT source
    orography -- as a declared provenance artifact.  The profile also
    declares its historical 330 ppm CO2 composition explicitly.
    """
    cfg = phase3_config() if cfg is None else cfg
    if (cfg.nx, cfg.ny, cfg.nz) != (250, 200, 49):
        raise ValueError("real74_d01 requires the d01 250x200x49 grid")
    grid = grids_from_wps_namelist(BUNDLE / "namelists" / "namelist.wps")[0]
    times = (START_TIME, START_TIME + timedelta(hours=6),
             START_TIME + timedelta(hours=12))
    prepared = runtime.prepare_real_case(
        cfg, grid=grid, geog_root=BUNDLE / "static" / "WPS_GEOG",
        source_orography_path=SOURCE_OROGRAPHY.path,
        source_orography_variable=SOURCE_OROGRAPHY.variable,
        vertical=VerticalConfig(
            eta_levels=tuple(float(value) for value in ETA_LEVELS),
            p_top=10000.0, hybrid_opt=cfg.hybrid_opt, etac=cfg.etac),
        sfcp_to_sfcp=True, snapshot_for=_rust_snapshot,
        forcing_times=times, start_time=START_TIME,
        # The declared 330 ppm CO2 is the RTE+RRTMGP path's historical
        # composition choice.  The legacy port evaluates WRF's
        # ghg_input=0 analytic year formula instead (333.47 ppm at this
        # start date) -- exactly what the CPU reference ran -- and
        # fails closed on explicit overrides, so none is forwarded.
        trace_gas_overrides=(
            None if rrtmg_variant(cfg) == RRTMG_VARIANT_LEGACY
            else {"co2": 330.0e-6}))
    return PreparedPhase3Case(
        cfg=prepared.cfg, grid=prepared.grid,
        static_fields=prepared.static_fields,
        initial_result=prepared.initial_result,
        analysis_00z=prepared.final_analysis,
        initial_snow_water_kgm2=prepared.initial_snow_water_kgm2)


def check_wrf_tooling_compatibility(path) -> tuple[tuple[str, ...], str]:
    """Run the Task 13 ``getvar`` gate and identify the imported backend.

    wrf-rust accepts its pathname-backed ``WrfFile`` without reopening a live
    netCDF4 handle.  Upstream wrf-python exposes no ``WrfFile``, so the same
    check uses its normal netCDF4 Dataset input when it is installable.
    """
    import netCDF4
    import wrf

    names = ("slp", "tk", "uvmet10", "td2", "cape_2d")
    results = {}
    pathname_backend = hasattr(wrf, "WrfFile")
    source = (wrf.WrfFile(str(path)) if pathname_backend
              else netCDF4.Dataset(path))
    try:
        for name in names:
            value = np.asarray(wrf.getvar(source, name, meta=False))
            if value.size == 0:
                raise RuntimeError(
                    f"WRF tooling backend getvar({name!r}) was empty")
            results[name] = value
    finally:
        if not pathname_backend:
            source.close()
    for name in ("slp", "tk"):
        if not np.isfinite(results[name]).all():
            raise RuntimeError(
                f"WRF tooling backend getvar({name!r}) was not finite")
    if not (800.0 < float(results["slp"].mean()) < 1100.0):
        raise RuntimeError("WRF tooling backend SLP is outside 800..1100 hPa")
    if not (150.0 < float(results["tk"].min())
            and float(results["tk"].max()) < 350.0):
        raise RuntimeError(
            "WRF tooling backend temperature is outside 150..350 K")
    module_file = getattr(wrf, "__file__", None)
    if module_file is not None:
        module_file = str(Path(module_file).resolve())
    identity = (
        f"module={getattr(wrf, '__name__', 'unknown')}; "
        f"file={module_file or 'unknown'}; "
        f"version={getattr(wrf, '__version__', 'unknown')}"
    )
    return names, identity


def _era5_00z_metrics(output_path: Path, analysis_00z,
                       initial_snow_water_kgm2) -> Era5SanityMetrics:
    diagnostics = _wrf_diagnostics(output_path)
    model = diagnostics["levels"][500]["temperature"]
    levels = np.asarray(analysis_00z.levels_hpa, dtype=np.float64)
    index = int(np.argmin(np.abs(levels - 500.0)))
    if abs(levels[index] - 500.0) > 1.0e-9:
        raise ValueError("ERA5 snapshot has no 500 hPa analysis level")
    source = analysis_00z.fields["TT"][index]
    if hasattr(source, "get"):
        source = source.get()
    source = np.asarray(source, dtype=np.float64)
    t500_rmse, t500_corr = _score_t500(model, source)
    analysis_t2 = analysis_00z.fields["T2"]
    if hasattr(analysis_t2, "get"):
        analysis_t2 = analysis_t2.get()
    analysis_t2 = np.asarray(analysis_t2, dtype=np.float64)
    model_t2 = np.asarray(diagnostics["t2"], dtype=np.float64)
    initial_swe = np.asarray(initial_snow_water_kgm2, dtype=np.float64)
    if model_t2.shape != analysis_t2.shape or model_t2.shape != initial_swe.shape:
        raise ValueError("model T2, ERA5 T2, and initial SWE shapes must match")
    snow = initial_swe > 1.0
    snow_free = ~snow

    def mean_bias(mask):
        reference_support = mask & np.isfinite(analysis_t2)
        if (not reference_support.any()
                or not np.isfinite(model_t2[reference_support]).all()):
            return float("nan")
        return float(np.mean(
            model_t2[reference_support] - analysis_t2[reference_support],
            dtype=np.float64))

    return Era5SanityMetrics(
        t500_rmse_k=t500_rmse,
        t500_pattern_correlation=t500_corr,
        # Model output has no missing-data convention: ordinary mean/min
        # reductions propagate a candidate NaN into the health metrics.
        t2_domain_mean_k=float(np.mean(model_t2, dtype=np.float64)),
        # ERA5 source gaps remain explicitly missing reference support.
        t2_analysis_mean_k=float(np.nanmean(analysis_t2)),
        t2_min_k=float(np.min(model_t2)),
        t2_snow_free_mean_bias_k=mean_bias(snow_free),
        t2_snow_cell_mean_bias_k=mean_bias(snow),
        initial_snow_free_cell_count=int(np.count_nonzero(snow_free)),
        initial_snow_cell_count=int(np.count_nonzero(snow)))


#: Legacy internal name for the whole-step schedule helper (the
#: implementation moved to gpuwm/runtime.py with Task 2).
_whole_step_count = runtime.whole_step_count

#: Output-due reflectivity predicate (moved to gpuwm/runtime.py; the
#: profile keeps its historical name for the pinned calendar test).
_refl_10cm_due = runtime.refl_10cm_due


def _configured_run_schedule(cfg: RunConfig) -> tuple[int, int]:
    """Validate and return ``(outer_steps, output_outer_steps)``.

    The frozen profile pins its run ceiling to the bundle's declared
    lateral-boundary coverage; the generic runtime derives the same
    ceiling from validated forcing coverage instead of a constant.
    """
    if cfg.run_seconds > 43200.0:
        raise ValueError(
            "real74_d01 has lateral-boundary forcing through 43200 s; "
            f"got run_seconds={cfg.run_seconds}")
    return runtime.configured_run_schedule(cfg)


def _restart_outer_steps(cfg: RunConfig) -> int | None:
    """Restart-write cadence in outer steps; ``None`` when disabled."""
    return runtime.restart_outer_steps(cfg)


def _integrate_configured_case(
        output_dir, cfg: RunConfig, *,
        restart_path=None) -> tuple[PreparedPhase3Case,
                                    RealCaseRunSummary]:
    """Prepare and integrate a configured real74 d01 run.

    Frozen-profile wrapper (Phase 5, Task 2) over
    :func:`gpuwm.runtime.integrate_prepared_case`: applies the profile's
    pinned schedule ceiling, prepares with the pinned bundle inputs,
    asserts the production radiation binding (an acceptance-only check
    that stays out of the generic runtime path), and delegates the loop
    with the retired compatibility integration transform
    (``phase3_integration_config``; at ``DYNAMICS_SUBSTEPS = 1`` it only
    sets ``clock_dt = dt``, which every consumer resolves identically).

    ``restart_path`` resumes a run exactly as before: the deterministic
    preparation runs unchanged, the restart restores the full cross-step
    state and clock, and the loop continues through ``run_seconds`` (the
    TOTAL forecast length from 12Z).  The initial 12Z wrfout is not
    rewritten on resume; ``cfg.restart_interval_s > 0`` writes
    ``gpuwmrst_d01_*`` files into ``output_dir`` on that cadence.
    """
    _configured_run_schedule(cfg)
    _restart_outer_steps(cfg)
    prepared = prepare_phase3_case(cfg)
    cfg = prepared.cfg
    radiation = prepared.initial_result.state.physics.radiation_callable
    if rrtmg_variant(cfg) == RRTMG_VARIANT_LEGACY:
        from gpuwm.core.rrtmg_legacy import RRTMGLegacyRadiation
        expected_radiation_cls = RRTMGLegacyRadiation
    else:
        expected_radiation_cls = RRTMGPRadiation
    if not isinstance(radiation, expected_radiation_cls):
        raise RuntimeError(
            "real74_d01 requires the production ID-4 binding for the "
            f"selected variant ({expected_radiation_cls.__name__}); got "
            f"{type(radiation).__name__}")
    run_summary = runtime.integrate_prepared_case(
        output_dir, prepared, start_time=START_TIME,
        output_title=_OUTPUT_TITLE, domain_id=1,
        integration_cfg=phase3_integration_config(cfg),
        restart_path=restart_path)
    return prepared, run_summary


def run_phase3_case(output_dir, cfg: RunConfig | None = None
                    ) -> Phase3CaseSummary:
    """Run the strict 12 h Task 13 verification and all of its gates."""
    output_dir = Path(output_dir)
    cfg = phase3_config() if cfg is None else cfg
    if cfg.run_seconds != 43200.0:
        raise ValueError(
            "real74_d01 verification requires the complete 12 h integration "
            f"(run_seconds=43200), got {cfg.run_seconds}")
    # Task 13 is defined at the exact 60 s outer clock and hourly cadence.
    # The config-driven ``run_config`` surface is where those values may vary.
    if cfg.dt != 60.0 or cfg.output_interval_s != 3600.0:
        raise ValueError(
            "real74_d01 verification requires dt=60 and "
            "output_interval_s=3600")
    prepared, run_summary = _integrate_configured_case(output_dir, cfg)
    outputs = run_summary.wrfout_paths

    head = compare_head_to_head(outputs[1], REFERENCE_13Z)
    head_failures = tuple(task13_gate_failures(head))
    era5 = _era5_00z_metrics(
        outputs[-1], prepared.analysis_00z,
        prepared.initial_snow_water_kgm2)
    diagnostics, backend = check_wrf_tooling_compatibility(outputs[-1])
    maps = make_task13_maps(outputs[-1], outputs[6], output_dir / "maps")
    return Phase3CaseSummary(
        wrfout_paths=outputs, map_paths=maps,
        nan_free=run_summary.nan_free,
        w_max_ms=run_summary.w_max_ms,
        boundary_w_max_ms=run_summary.boundary_w_max_ms,
        interior_w_max_ms=run_summary.interior_w_max_ms,
        w_max_boundary_row=run_summary.w_max_boundary_row,
        boundary_zone_blowup=run_summary.boundary_zone_blowup,
        head_to_head=head, head_to_head_failures=head_failures,
        era5_00z=era5, wrf_tooling_diagnostics=diagnostics,
        wrf_tooling_backend_identity=backend,
        caveats=PHYSICS_AND_DYNAMICS_CAVEATS,
        dynamics_substeps=run_summary.dynamics_substeps,
        ysu_nan_guard_fires=run_summary.ysu_nan_guard_fires,
        surface_forcing_updates=run_summary.surface_forcing_updates,
        swdown_peak_wm2=run_summary.swdown_peak_wm2,
        swdown_peak_time=run_summary.swdown_peak_time)


def summary_metrics(summary: Phase3CaseSummary) -> dict[str, object]:
    """Flatten a :class:`Phase3CaseSummary` for the shared CLI gate checker."""
    head = summary.head_to_head
    temperature_rmse = [float(head.temperature_rmse_k[level])
                        for level in (500, 700, 850)]
    wind_rmse = [float(values[level])
                 for values in (head.u_rmse_ms, head.v_rmse_ms)
                 for level in (500, 700, 850)]
    era5 = summary.era5_00z
    finite_values = (temperature_rmse + wind_rmse + [
        float(head.mslp_pattern_correlation), float(era5.t500_rmse_k),
        float(era5.t500_pattern_correlation),
        float(era5.t2_snow_free_mean_bias_k), float(era5.t2_min_k),
        float(summary.w_max_ms), float(summary.interior_w_max_ms),
    ])
    return {
        "nan": bool(not summary.nan_free
                    or not np.isfinite(finite_values).all()),
        "temperature_rmse_max_k": max(temperature_rmse),
        "wind_rmse_max_ms": max(wind_rmse),
        "mslp_pattern_correlation": float(head.mslp_pattern_correlation),
        "boundary_zone_blowup": bool(summary.boundary_zone_blowup),
        "t2_snow_free_abs_bias_k": abs(
            float(era5.t2_snow_free_mean_bias_k)),
        "t2_min_k": float(era5.t2_min_k),
        "ysu_nan_guard_fires": int(summary.ysu_nan_guard_fires),
        "wrf_tooling_diagnostic_count": len(
            summary.wrf_tooling_diagnostics),
        # Phase 4 BINDING flagship skill metrics (gated in GATES above).
        "era5_t500_rmse_k": float(era5.t500_rmse_k),
        "era5_t500_pattern_correlation": float(
            era5.t500_pattern_correlation),
        "w_max_ms": float(summary.w_max_ms),
        "interior_w_max_ms": float(summary.interior_w_max_ms),
        "w_max_boundary_row": summary.w_max_boundary_row,
    }


def run(outdir=None) -> dict[str, object]:
    """Shared verification-case entry point used by ``gpuwm verify``."""
    if outdir is not None:
        return summary_metrics(run_phase3_case(outdir))
    import tempfile
    with tempfile.TemporaryDirectory(prefix="gpuwm-real74-") as directory:
        return summary_metrics(run_phase3_case(directory))


def _case_grid(cfg: RunConfig):
    """Validate the config against the registered d01 grid and return it."""
    if cfg.case != "real74_d01":
        raise ValueError(
            f"real74_d01 pipeline requires case='real74_d01', got {cfg.case!r}")
    grid = grids_from_wps_namelist(
        BUNDLE / "namelists" / "namelist.wps")[0]
    if (cfg.nx, cfg.ny, cfg.dx, cfg.dy) != (
            grid.e_we - 1, grid.e_sn - 1, grid.dx, grid.dy):
        raise ValueError(
            "real74_d01 config grid does not match bundle d01: "
            f"got {(cfg.nx, cfg.ny, cfg.dx, cfg.dy)}, expected "
            f"{(grid.e_we - 1, grid.e_sn - 1, grid.dx, grid.dy)}")
    return grid


def _write_npz(path, fields: dict[str, object]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.savez_compressed(stream, **fields)
    return path


def write_static(cfg: RunConfig, output) -> Path:
    """Build the registered domain's WPS_GEOG fields into a portable NPZ."""
    grid = _case_grid(cfg)
    fields = build_static(grid, BUNDLE / "static" / "WPS_GEOG")
    return _write_npz(output, fields)


def write_ingest(cfg: RunConfig, output) -> Path:
    """Run real-data initialization and write its live FP32 state to NPZ.

    This is a stage artifact for inspection/reproducibility, not a restart
    file; ``gpuwm run`` rebuilds the deterministic setup from the same config.
    """
    _case_grid(cfg)
    prepared = prepare_phase3_case(cfg)
    result = prepared.initial_result
    state = result.state
    import cupy as cp
    names = ("u", "v", "w", "thp", "php", "mup", "qv", "qc", "qr",
             "mub2d", "msft", "msfu", "msfv", "f", "e")
    fields = {name: cp.asnumpy(getattr(state, name)) for name in names}
    fields.update({
        "surface_pressure": result.surface_pressure,
        "surface_qv": result.surface_qv,
        "dry_mass": result.dry_mass,
        "dry_pressure": result.dry_pressure,
        "total_pressure": result.total_pressure,
        "total_geopotential": result.total_geopotential,
        "integrated_moisture_pressure": result.integrated_moisture_pressure,
        "case": np.asarray(cfg.case),
    })
    return _write_npz(output, fields)


def run_config(cfg: RunConfig, output_dir,
               restart=None) -> RealCaseRunSummary:
    """Integrate the registered case without running verification gates.

    ``restart`` names a ``gpuwmrst`` file written by an earlier run of the
    SAME configuration (only forecast length / output and restart cadence
    may differ — the restart header enforces this); the run then continues
    bit-identically from the restored clock through ``run_seconds``.
    """
    _case_grid(cfg)
    _configured_run_schedule(cfg)
    _restart_outer_steps(cfg)
    return _integrate_configured_case(
        output_dir, cfg, restart_path=restart)[1]


__all__ = [
    "BUNDLE", "DYNAMICS_SUBSTEPS", "ETA_LEVELS",
    "ERA5_PERSISTENCE_T500_CORRELATION", "ERA5_PERSISTENCE_T500_RMSE_K",
    "Era5SanityMetrics", "GATES", "HeadToHeadMetrics",
    "PHYSICS_AND_DYNAMICS_CAVEATS", "Phase3CaseSummary", "SOURCE_OROGRAPHY",
    "PreparedPhase3Case", "REAL74_ANALYSIS_PROFILE", "REAL74_HEALTH_PROFILE",
    "REAL74_ORACLE_PROFILE", "REAL74_PROFILE", "RealCaseRunSummary",
    "build_forcing",
    "check_wrf_tooling_compatibility",
    "compare_head_to_head", "config",
    "make_reflectivity_map", "make_task13_maps",
    "phase3_config", "phase3_integration_config", "prepare_phase3_case",
    "run", "run_config", "run_phase3_case", "summary_metrics",
    "task13_gate_failures", "write_ingest", "write_static",
]
