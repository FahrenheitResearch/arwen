"""EXPERIMENTAL: the v1.2 end-to-end composition gate, on a synthetic domain.

One command, six lanes, no stubs between them:

    base state
      -> gpuwm.da.perturb              R members, real spectral perturbations
      -> gpuwm.ensemble.cycle          forecast leg (real driver, real manifest)
      -> gpuwm.da.obsop                H(x) on a designated truth member
      -> gpuwm.obs.radar_grid          packed as a real gpuwm-obs.radar-grid.v1
      -> gpuwm.da.obs_radar            adapted to the filter's obs structures
      -> gpuwm.da.letkf                LETKF analysis, real increments
      -> gpuwm.da.positivity           caller-side hydrometeor policy
      -> gpuwm.ensemble.increments     applied, receipted, analysis.npz
      -> gpuwm.ensemble.cycle          second leg, restarted FROM the analysis
      -> gpuwm.da.enprod               products off the analysed ensemble

**What is real and what is not.**  Every seam above is shipped code.  The
one substitution is the *dycore*: ``forecast_leg`` below is a constant
advection plus a light diffusion, applied identically to the truth and to
every member.  That is a deliberate perfect-model twin experiment, the same
setting :mod:`gpuwm.da.osse` gates the filter in, and it is chosen because
this gate is about whether the system composes -- whether the manifest the
engine writes is the manifest the products read, whether the increments the
filter returns are the increments the checkpoint receives -- and a real
integration would make that a GPU-and-staged-data question instead of a
thirty-second one.

**Scientific tuning is explicitly not the goal.**  The amplitudes, the
localisation radius and the observation error here are chosen so the
composition can be exercised at R=10 on a 24x24x6 grid in seconds.  No
number this module prints supports any claim about forecast skill, and the
perfect model means no number here supports an inflation tuning either --
see the caveat in the enkf lane's handoff.
"""

from __future__ import annotations

import json
import types
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: Provenance schema for the gate report.
GATE_SCHEMA = "gpuwm-da.synthetic-cycle-gate.v1"

#: Fields carried through the whole chain.  All four are in the restart
#: prognostic contract, so the engine's increment applier accepts them;
#: ``qr`` is the one with a positivity constraint, which is the point of
#: including it.
STATE_FIELDS = ("u", "v", "w", "qr")

#: Fields the filter analyses.  ``w`` is carried and *observed through* --
#: it is in the beam projection -- but it is not analysed, and that is a
#: cross-lane fact rather than a preference: ``gpuwm.da.perturb``'s
#: documented non-goal 4 is that it perturbs no ``w``, so a ``w`` prior
#: built from that module has exactly zero ensemble spread, and
#: ``gpuwm.da.letkf`` refuses a field with no usable spread rather than
#: returning a zero increment for it.  Both are right.  Naming ``w`` in
#: ``analysis_fields`` against a perturbation library that does not perturb
#: it is the configuration error, and the filter's refusal names the fix.
#: Hydrometeors stay in: withholding them is legitimate and would leave the
#: positivity seam untested.
ANALYSIS_FIELDS = ("u", "v", "qr")


@dataclass
class GateConfig:
    members: int = 10
    nx: int = 24
    ny: int = 24
    nz: int = 6
    dx_m: float = 3000.0
    top_m: float = 12000.0
    base_seed: int = 20260730
    #: Ensemble 1-sigma, in each field's own units.
    wind_amplitude_ms: float = 2.0
    qr_amplitude: float = 2.0e-4
    #: Below span/(2*pi) on the smallest grid this gate is run at (18x18
    #: cells of 3 km = 54 km span, limit 8.59 km), which is where
    #: ``gpuwm.da.perturb`` now admits a prescribed length scale: above it
    #: the documented spectral peak k=1/L falls below the domain's lowest
    #: nonzero wavenumber and the draw is a domain-wide offset.
    length_scale_km: float = 8.0
    #: Radial-velocity observation error, m/s.
    obs_error_ms: float = 1.0
    horizontal_localization_m: float = 18000.0
    vertical_localization_m: float = 4000.0
    leg_seconds: float = 300.0
    #: Rows/columns of unobserved margin around the observed patch.  Wide
    #: enough that the corners sit outside every observation's localisation
    #: lens, so "increments are exactly zero outside localisation" is a
    #: claim with somewhere to be false.  It also keeps observations off
    #: the rim, where the perturbation taper leaves no spread to correct.
    observed_margin_cells: int = 6
    #: Relaxation to prior spread.  Stated, not defaulted: the filter
    #: refuses to guess it, and a two-leg cycle is exactly the setting in
    #: which an unrelaxed ensemble starts losing the spread it needs to keep
    #: responding.  It does not touch the exactly-zero-outside-localisation
    #: claim, which holds for every alpha at prior_inflation = 1.
    rtps_alpha: float = 0.9
    #: Constant advection for the stand-in forecast, grid cells per leg.
    advect_cells: int = 1
    diffusion: float = 0.15
    products: bool = True
    fields_analysed: tuple[str, ...] = field(
        default_factory=lambda: ANALYSIS_FIELDS)


# ---------------------------------------------------------------------------
# The synthetic world
# ---------------------------------------------------------------------------


def target_grid(cfg: GateConfig):
    """The one grid every stage agrees on, hashed."""

    from gpuwm.obs.target_grid import TargetGrid
    from gpuwm.static.lambert import LambertGrid

    projection = LambertGrid(
        ref_lat=35.0, ref_lon=-97.0, truelat1=33.0, truelat2=37.0,
        stand_lon=-97.0, dx=cfg.dx_m, dy=cfg.dx_m,
        e_we=cfg.nx + 1, e_sn=cfg.ny + 1)
    return TargetGrid.from_projection(
        projection, z_w=np.linspace(0.0, cfg.top_m, cfg.nz + 1),
        name="synthetic-cycle")


def _staggered_state(cfg: GateConfig, grid):
    """A DomainState-shaped namespace ``gpuwm.da.perturb`` will accept.

    Real ARW staggering, because the perturbation module checks it and
    would reject a mass-point-everywhere shortcut -- which is exactly the
    check that keeps a member's u from being broadcast into the wrong
    place.
    """

    f32 = np.float32
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    z_w = np.asarray(grid.z_w, dtype=np.float64)
    return types.SimpleNamespace(
        thp=np.zeros((nz, ny, nx), f32),
        qv=np.full((nz, ny, nx), 8.0e-3, f32),
        u=np.zeros((nz, ny, nx + 1), f32),
        v=np.zeros((nz, ny + 1, nx), f32),
        w=np.zeros((nz + 1, ny, nx), f32),
        p=np.full((nz, ny, nx), 8.0e4, f32),
        sina=np.zeros((ny, nx), f32),
        cosa=np.ones((ny, nx), f32),
        heights=0.5 * (z_w[:-1] + z_w[1:]))


def base_fields(cfg: GateConfig, grid) -> dict:
    """The state every member and the truth start from, on mass points."""

    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float64)
    blob = np.exp(-(((yy - ny / 2.0) ** 2 + (xx - nx / 2.0) ** 2)
                    / (2.0 * (min(ny, nx) / 6.0) ** 2)))
    column = np.linspace(1.0, 0.3, nz)[:, None, None]
    return {
        "u": np.full((nz, ny, nx), 10.0),
        "v": np.full((nz, ny, nx), 2.0),
        "w": np.zeros((nz, ny, nx)),
        "qr": 1.0e-3 * blob[None, :, :] * column,
    }


def perturbed_member(cfg: GateConfig, grid, seed: int) -> dict:
    """One member's mass-point state, perturbed by the real perturb lane.

    ``gpuwm.da.perturb`` perturbs ``u``/``v`` on their staggered grids and
    ``thp``/``qv`` on mass points; this gate wants mass-point winds and a
    hydrometeor, so the wind perturbation goes through the module and is
    destaggered with :mod:`gpuwm.da.obsop`'s own destaggering, and the
    ``qr`` perturbation reuses the module's ``gaussian_random_field`` --
    the same spectral construction, on a field the module does not itself
    perturb (documented non-goal 4: no hydrometeer perturbation).
    """

    from gpuwm.da import obsop, perturb

    state = _staggered_state(cfg, grid)
    config = perturb.PerturbationConfig.from_mapping({
        "dx_km": cfg.dx_m / 1000.0, "dy_km": cfg.dx_m / 1000.0,
        "rim_width": 2,
        "fields": [
            {"name": "u", "amplitude": cfg.wind_amplitude_ms,
             "length_scale_km": cfg.length_scale_km},
            {"name": "v", "amplitude": cfg.wind_amplitude_ms,
             "length_scale_km": cfg.length_scale_km},
        ],
    })
    provenance = perturb.apply_perturbations(state, seed, config)

    fields = base_fields(cfg, grid)
    fields["u"] = fields["u"] + np.asarray(obsop.destagger_u(state.u),
                                           dtype=np.float64)
    fields["v"] = fields["v"] + np.asarray(obsop.destagger_v(state.v),
                                           dtype=np.float64)
    noise, _ = perturb.gaussian_random_field(
        (cfg.nz, cfg.ny, cfg.nx), seed=seed, name="qr",
        dx_km=cfg.dx_m / 1000.0, dy_km=cfg.dx_m / 1000.0,
        length_scale_km=cfg.length_scale_km, xp=np)
    fields["qr"] = np.maximum(
        fields["qr"] + cfg.qr_amplitude * np.asarray(noise), 0.0)
    return fields, provenance


def forecast_leg(fields: dict, cfg: GateConfig) -> dict:
    """The dycore stand-in: constant advection plus light diffusion.

    Identical for the truth and every member -- a perfect model, which is
    what makes this a twin experiment and what makes it useless for
    tuning inflation.  It moves mass, so "the analysis helped" is a
    statement about a state that actually evolved.
    """

    shift = int(cfg.advect_cells)
    out = {}
    for name, values in fields.items():
        moved = np.roll(values, shift, axis=2)
        smoothed = (1.0 - cfg.diffusion) * moved + cfg.diffusion * 0.25 * (
            np.roll(moved, 1, axis=1) + np.roll(moved, -1, axis=1)
            + np.roll(moved, 1, axis=2) + np.roll(moved, -1, axis=2))
        out[name] = np.maximum(smoothed, 0.0) if name == "qr" else smoothed
    return out


# ---------------------------------------------------------------------------
# Observations: H(truth) + noise, through the radar lane's own writer
# ---------------------------------------------------------------------------


def _beam(cfg: GateConfig, grid, site):
    """Beam unit vectors at every mass point, from the real obsop geometry."""

    from gpuwm.da import obsop

    geometry = obsop.GridGeometry.from_target_grid(grid)
    beam = obsop.beam_geometry(geometry, site)
    east, north, up = beam.unit_vector_enu()
    shape = (cfg.nz, cfg.ny, cfg.nx)
    return tuple(np.broadcast_to(np.asarray(component, dtype=np.float64),
                                 shape).copy()
                 for component in (east, north, up))


def observed_region(cfg: GateConfig) -> np.ndarray:
    """``(nz, ny, nx)`` boolean: where the synthetic radars see anything.

    An inset patch, for two reasons.  Observations on the rim would be
    assimilated into columns the perturbation taper deliberately left
    unperturbed -- a spread-free background, and a bad test.  And the
    corners have to lie outside every observation's localisation lens, or
    "increments are exactly zero outside localisation" is a claim with
    nowhere to be false.
    """

    margin = int(cfg.observed_margin_cells)
    region = np.zeros((cfg.nz, cfg.ny, cfg.nx), bool)
    region[:, margin:cfg.ny - margin, margin:cfg.nx - margin] = True
    if not region.any():
        raise ValueError(
            f"observed_margin_cells={margin} leaves no observed points on a "
            f"{cfg.ny}x{cfg.nx} grid")
    return region


def unlocalized_region(cfg: GateConfig, grid) -> np.ndarray:
    """Gridpoints beyond the localisation cutoff from every observation.

    Computed by brute force from the observed set and the same *physical*
    metric the filter localises in -- a geodesic between the two mass
    points and a height difference taken in each point's own column --
    rather than by reproducing the filter's index stencil, so agreement
    means something.  The haversine below is written here rather than
    imported for the same reason: a reference that calls the code it is
    checking is not a reference.
    """

    from gpuwm.da import obs_radar

    geometry = obs_radar.letkf_grid_geometry(grid)
    heights = np.asarray(geometry.height_field(cfg.ny, cfg.nx),
                         dtype=np.float64)
    lat = np.radians(np.asarray(geometry.lat_deg, dtype=np.float64))
    lon = np.radians(np.asarray(geometry.lon_deg, dtype=np.float64))
    radius = float(geometry.earth_radius_m)
    observed = observed_region(cfg)
    kz, jj, ii = np.nonzero(observed)
    z_obs = heights[kz, jj, ii]
    lat_obs, lon_obs = lat[jj, ii], lon[jj, ii]

    outside = np.ones((cfg.nz, cfg.ny, cfg.nx), bool)
    for j in range(cfg.ny):
        for i in range(cfg.nx):
            hav = (np.sin((lat_obs - lat[j, i]) / 2.0) ** 2
                   + np.cos(lat[j, i]) * np.cos(lat_obs)
                   * np.sin((lon_obs - lon[j, i]) / 2.0) ** 2)
            near_h = (2.0 * radius * np.arcsin(np.sqrt(hav))
                      < cfg.horizontal_localization_m)
            if not near_h.any():
                continue
            for k in range(cfg.nz):
                dz = np.abs(z_obs - heights[k, j, i])
                if np.any(near_h & (dz < cfg.vertical_localization_m)):
                    outside[k, j, i] = False
    return outside


def write_synthetic_observations(path, cfg: GateConfig, grid, truth: dict, *,
                                 seed: int):
    """Pack H(truth) + noise as a real ``gpuwm-obs.radar-grid.v1`` file.

    The radar lane's own writer, its own schema, its own atomic publish --
    not a dict pretending to be a file.  Two radars, so the per-radar axis
    the EnKF adapter has to honour is actually exercised end to end.
    """

    from gpuwm.da import obsop
    from gpuwm.obs.radar_grid import write_radar_grid
    from gpuwm.obs.superob import GriddedObservations, SuperobParams

    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    shape = (nz, ny, nx)
    rng = np.random.default_rng(seed)

    sites = [
        obsop.RadarSite(latitude_deg=float(grid.lat[ny // 2, 1]),
                        longitude_deg=float(grid.lon[ny // 2, 1]),
                        altitude_m=350.0, name="AAAA"),
        obsop.RadarSite(latitude_deg=float(grid.lat[1, nx // 2]),
                        longitude_deg=float(grid.lon[1, nx // 2]),
                        altitude_m=380.0, name="BBBB"),
    ]

    interior = observed_region(cfg)

    east = np.zeros((2,) + shape)
    north = np.zeros((2,) + shape)
    up = np.zeros((2,) + shape)
    vr_obs = np.zeros((2,) + shape)
    for index, site in enumerate(sites):
        e, n, u = _beam(cfg, grid, site)
        east[index], north[index], up[index] = e, n, u
        clean = truth["u"] * e + truth["v"] * n + truth["w"] * u
        vr_obs[index] = clean + rng.normal(0.0, cfg.obs_error_ms, shape)

    vr_mask = np.broadcast_to(interior, (2,) + shape).astype(np.int8)
    vr_mask = np.where(np.all(np.isfinite(vr_obs), axis=0)[None, ...],
                       vr_mask, np.int8(0))
    vr_obs = np.where(vr_mask.astype(bool), vr_obs, 0.0)

    zeros_plane = np.zeros(shape, np.float32)
    observations = GriddedObservations(
        z_obs=zeros_plane, z_mask=np.zeros(shape, np.int8),
        z_err=zeros_plane, z_max=zeros_plane, z_mean=zeros_plane,
        z_count=np.zeros(shape, np.int32),
        vr_obs=vr_obs.astype(np.float32), vr_mask=vr_mask,
        vr_err=np.where(vr_mask.astype(bool), cfg.obs_error_ms,
                        0.0).astype(np.float32),
        vr_count=vr_mask.astype(np.int32),
        vr_rejected=np.zeros((2,) + shape, np.int32),
        vr_beam_east=east.astype(np.float32),
        vr_beam_north=north.astype(np.float32),
        vr_beam_up=up.astype(np.float32),
        radars=[{"id": site.name, "lat_deg": site.latitude_deg,
                 "lon_deg": site.longitude_deg, "alt_m": site.altitude_m,
                 "valid_time": "1970-01-01T18:00:00Z"} for site in sites],
        counts=[], provenance=[])
    receipt = write_radar_grid(
        path, observations, grid, valid_time="1970-01-01T18:00:00Z",
        params=SuperobParams(), overwrite=True)
    return receipt, sites


# ---------------------------------------------------------------------------
# The analysis
# ---------------------------------------------------------------------------


def analyse(cfg: GateConfig, grid, obs_path, prior_by_member: dict):
    """Adapter -> LETKF.  Returns per-member increments and provenance."""

    from gpuwm.da import letkf, obs_radar

    indices = sorted(prior_by_member)
    prior = {name: np.stack([prior_by_member[i][name] for i in indices])
             for name in cfg.fields_analysed}
    shape = (cfg.nz, cfg.ny, cfg.nx)
    # The full grid, not its identity string: the string is a claim about
    # arrays the observation file only partly stores, and z_w is reachable
    # no other way.  See gpuwm.da.obs_radar.read_document.
    document = obs_radar.read_document(
        obs_path, expected_grid=grid,
        expected_grid_identity=grid.identity_sha256())

    def simulate(radar_index, radar):
        beam = obs_radar.beam_unit_vectors(document, radar_index)
        return np.stack([
            obs_radar.simulated_radial_velocity(
                prior_by_member[i]["u"], prior_by_member[i]["v"],
                prior_by_member[i]["w"], beam)
            for i in indices])

    batches, adapter_provenance = obs_radar.radar_grid_to_gridded_obs(
        document, velocity_simulated=simulate, expected_grid=grid,
        expected_grid_identity=grid.identity_sha256())

    config = letkf.LetkfConfig(
        localization=letkf.Localization(
            horizontal_m=cfg.horizontal_localization_m,
            vertical_m=cfg.vertical_localization_m),
        analysis_fields=tuple(cfg.fields_analysed),
        rtps_alpha=cfg.rtps_alpha)
    diagnostics = letkf.LetkfDiagnostics()
    increments = letkf.analyze(prior, batches,
                               obs_radar.letkf_grid_geometry(grid), config,
                               diagnostics)
    assert all(increments[name].shape == (len(indices),) + shape
               for name in cfg.fields_analysed)
    per_member = {index: {name: increments[name][slot]
                          for name in cfg.fields_analysed}
                  for slot, index in enumerate(indices)}
    return per_member, {
        "method": "LETKF (Hunt, Kostelich & Szunyogh 2007 sec 2.3)",
        "stability": "experimental",
        "analysis_fields": list(cfg.fields_analysed),
        "localization_horizontal_m": cfg.horizontal_localization_m,
        "localization_vertical_m": cfg.vertical_localization_m,
        "observations": adapter_provenance,
        "observed_gridpoints": int(
            getattr(diagnostics, "analysed_points", 0) or 0),
    }


def rmse_against_truth(fields_by_member: dict, truth: dict, names) -> dict:
    """Ensemble-mean RMSE per field, and the total over all named fields."""

    indices = sorted(fields_by_member)
    out = {}
    squares = []
    for name in names:
        mean = np.mean([fields_by_member[i][name] for i in indices], axis=0)
        error = mean - truth[name]
        out[name] = float(np.sqrt(np.mean(error ** 2)))
        squares.append(error.ravel())
    out["total"] = float(np.sqrt(np.mean(np.concatenate(squares) ** 2)))
    return out


def ensemble_spread(fields_by_member: dict, names) -> dict:
    indices = sorted(fields_by_member)
    out = {}
    for name in names:
        stack = np.stack([fields_by_member[i][name] for i in indices])
        out[name] = float(np.sqrt(np.mean(np.var(stack, axis=0, ddof=1))))
    return out


# ---------------------------------------------------------------------------
# Member I/O -- checkpoints the engine's increment applier accepts
# ---------------------------------------------------------------------------

CHECKPOINT_NAME = "gpuwmrst_leg.npz"


def write_checkpoint(path, fields: dict, *, elapsed_seconds: float) -> None:
    """A checkpoint that also states its clock.

    The clock is not decoration.  ``gpuwm.runtime.integrate_prepared_case``
    restores it and measures ``run_seconds`` from the experiment's start
    time, so a cycling driver that hands a leg the wrong horizon is caught
    by the checkpoint's own elapsed time and by nothing else.  Writing it
    here is what lets this gate exercise that contract instead of assuming
    it.
    """
    payload = {f"state/{name}": np.asarray(values, np.float32)
               for name, values in fields.items()}
    payload["meta/elapsed_seconds"] = np.asarray(float(elapsed_seconds))
    np.savez(path, **payload)


def read_checkpoint(path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {key[len("state/"):]: np.asarray(data[key], np.float64)
                for key in data.files if key.startswith("state/")}


def checkpoint_elapsed_seconds(path) -> float:
    with np.load(path, allow_pickle=False) as data:
        if "meta/elapsed_seconds" not in data.files:
            raise ValueError(f"{path} states no elapsed time")
        return float(data["meta/elapsed_seconds"])


def _reflectivity(cfg: GateConfig, fields: dict):
    """REFL_10CM for the products, from the scheme-true authority.

    ``gpuwm.da.obsop.simulated_reflectivity`` on ``mp_physics=1`` -- the
    Smith (1975) fixed-intercept rain-only Rayleigh form ArWen's own
    reflectivity module derives.  A hand-rolled dBZ-from-qr proxy would
    have been three lines and would have made the products a picture of
    this module's arithmetic instead of the model's.
    """

    from gpuwm.config import RunConfig
    from gpuwm.da import obsop

    zeros = np.zeros((cfg.nz, cfg.ny, cfg.nx), np.float32)
    state = types.SimpleNamespace(
        qv=np.full((cfg.nz, cfg.ny, cfg.nx), 8.0e-3, np.float32),
        qc=zeros, qr=np.asarray(fields["qr"], np.float32),
        qi=None, qs=None, qg=None, qh=None,
        p=np.full((cfg.nz, cfg.ny, cfg.nx), 8.0e4, np.float32),
        thp=zeros, thb=np.full((cfg.nz,), 300.0, np.float32))
    run = RunConfig(nx=cfg.nx, ny=cfg.ny, nz=cfg.nz, dx=cfg.dx_m,
                    dy=cfg.dx_m, dt=10.0, run_seconds=cfg.leg_seconds,
                    ztop=cfg.top_m, mp_physics=1)
    return np.asarray(obsop.simulated_reflectivity(state, run), np.float32)


def write_member_wrfout(path, cfg: GateConfig, grid, fields: dict, stamp):
    from gpuwm.io.wrfout import WrfoutWriter

    dbz = _reflectivity(cfg, fields)
    speed = np.hypot(fields["u"][0], fields["v"][0]).astype(np.float32)
    with WrfoutWriter(path, nx=cfg.nx, ny=cfg.ny, nz=cfg.nz, dx=cfg.dx_m,
                      dy=cfg.dx_m, global_attrs={"GRID_ID": 2}) as writer:
        writer.write_frame(stamp, {
            "T": np.zeros((cfg.nz, cfg.ny, cfg.nx), np.float32),
            "MU": np.zeros((cfg.ny, cfg.nx), np.float32),
            "REFL_10CM": dbz,
            "UP_HELI_MAX": (10.0 * speed).astype(np.float32),
            "T2": np.full((cfg.ny, cfg.nx), 292.0, np.float32),
            "U10": fields["u"][0].astype(np.float32),
            "V10": fields["v"][0].astype(np.float32),
            "RAINC": np.zeros((cfg.ny, cfg.nx), np.float32),
            "RAINNC": (1.0e4 * fields["qr"][0]).astype(np.float32),
            "XLAT": np.asarray(grid.lat, np.float32),
            "XLONG": np.asarray(grid.lon, np.float32),
            "HGT": np.asarray(grid.terrain_m, np.float32),
            "SINALPHA": np.zeros((cfg.ny, cfg.nx), np.float32),
            "COSALPHA": np.ones((cfg.ny, cfg.nx), np.float32),
        })


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def run_gate(outdir, cfg: GateConfig | None = None) -> dict:
    """The whole chain, once.  Returns a JSON-serialisable report.

    Raises rather than returning a failing report: this is a gate, and a
    gate that returns "failed" is a gate somebody forgets to check.
    """

    from gpuwm.ensemble.config import load_ensemble_config
    from gpuwm.ensemble.cycle import cycle_root, run_cycles
    from gpuwm.ensemble.manifest import CYCLE_MANIFEST_NAME
    from gpuwm.ensemble.member import MemberOutcome
    from gpuwm.ensemble.state_sha import hash_state_arrays

    cfg = cfg or GateConfig()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    grid = target_grid(cfg)

    # -- truth: its own draw, not one of the R members ---------------------
    truth_leg0, _ = perturbed_member(cfg, grid,
                                     seed=cfg.base_seed + 10_000_019)
    truth_leg1 = forecast_leg(truth_leg0, cfg)
    truth_leg2 = forecast_leg(truth_leg1, cfg)

    obs_path = outdir / "obs-radar-grid.nc"
    obs_receipt, sites = write_synthetic_observations(
        obs_path, cfg, grid, truth_leg1, seed=cfg.base_seed + 991)

    # -- the ensemble, through the real engine -----------------------------
    (outdir / "base.toml").write_text(_BASE_TOML.format(
        nx=cfg.nx, ny=cfg.ny, nz=cfg.nz, dx=cfg.dx_m, top=cfg.top_m,
        seconds=cfg.leg_seconds), encoding="utf-8")
    (outdir / "ensemble.toml").write_text(
        "[ensemble]\n"
        'base_config = "base.toml"\n'
        f"n_members = {cfg.members}\n"
        f"base_seed = {cfg.base_seed}\n"
        'perturbation = "none"\n', encoding="utf-8")
    ens_cfg = load_ensemble_config(outdir / "ensemble.toml")
    ens_root = outdir / "cycles"

    stamps = ("1970-01-01_18:00:00", "1970-01-01_19:00:00")

    def runner(*, base_config, member_dir, index, seed, perturbation,
               perturbation_options, run_seconds=None, restart=None, **_):
        """The stand-in dycore, under the REAL integrator's clock contract.

        ``gpuwm.runtime.integrate_prepared_case`` treats ``run_seconds`` as
        the TOTAL forecast length from the experiment's start time, restores
        the restart's elapsed time, and refuses outright when the restored
        clock has already reached that total.  This runner reproduces that
        contract verbatim -- same arithmetic, same refusal -- so a driver
        that hands leg N+1 a leg-length horizon instead of a cumulative one
        fails here exactly as it failed on the GPU, and the gate is a proof
        of the fix rather than a proof that this runner ignores the
        argument.
        """
        member_dir = Path(member_dir)
        member_dir.mkdir(parents=True, exist_ok=True)
        total_seconds = float(run_seconds if run_seconds is not None
                              else cfg.leg_seconds)
        if restart is None:
            fields, _ = perturbed_member(cfg, grid, seed=int(seed))
            start_seconds = 0.0
        else:
            fields = read_checkpoint(restart)
            start_seconds = checkpoint_elapsed_seconds(restart)
        if start_seconds >= total_seconds:
            raise ValueError(
                f"restart file is already at {start_seconds} s; nothing to "
                f"integrate before run_seconds={total_seconds}")
        leg = int(round(start_seconds / float(cfg.leg_seconds)))
        before = hash_state_arrays(sorted(
            (k, np.asarray(v, np.float32)) for k, v in fields.items()))
        fields = forecast_leg(fields, cfg)
        after = hash_state_arrays(sorted(
            (k, np.asarray(v, np.float32)) for k, v in fields.items()))
        write_checkpoint(member_dir / CHECKPOINT_NAME, fields,
                         elapsed_seconds=total_seconds)
        write_member_wrfout(
            member_dir / f"wrfout_d02_{stamps[leg].replace(':', '-')}.nc",
            cfg, grid, fields, stamps[leg])
        return MemberOutcome(
            index=index, seed=seed, member_dir=member_dir,
            initial_state_sha256=before, final_state_sha256=after,
            wall_seconds=0.0, sim_seconds=total_seconds - start_seconds,
            wrfout_count=1, last_checkpoint=str(member_dir / CHECKPOINT_NAME),
            perturbation={"restart_from": None if restart is None
                          else str(restart)})

    captured: dict = {}

    def assimilate(cycle_index, member_states):
        prior = {index: read_checkpoint(
            Path(info["member_dir"]) / CHECKPOINT_NAME)
            for index, info in member_states.items()}
        increments, provenance = analyse(cfg, grid, obs_path, prior)
        captured["prior"] = prior
        captured["increments"] = increments
        captured["method"] = provenance
        # The driver's method channel: increments AND the provenance that
        # says what produced them.  Returning the mapping alone left the
        # receipt saying "method: null", which cannot distinguish this
        # LETKF analysis from any other analysis of any other observations.
        return increments, provenance

    result = run_cycles(ens_cfg, ens_root, n_cycles=2,
                        cycle_seconds=cfg.leg_seconds, runner=runner,
                        assimilate=assimilate, positivity="clip")
    if result.status != "COMPLETE":
        raise RuntimeError(f"cycling did not complete: {result.status}")

    # -- the numbers -------------------------------------------------------
    prior = captured["prior"]
    increments = captured["increments"]
    analysis = {index: {name: prior[index][name] + increments[index][name]
                        for name in cfg.fields_analysed}
                for index in prior}
    # Positivity is the driver's, and it already ran against the written
    # analysis; mirror it here so the reported numbers are the written ones.
    for index in analysis:
        analysis[index]["qr"] = np.maximum(analysis[index]["qr"], 0.0)

    names = tuple(cfg.fields_analysed)
    prior_rmse = rmse_against_truth(prior, truth_leg1, names)
    analysis_rmse = rmse_against_truth(analysis, truth_leg1, names)

    leg1_root = cycle_root(ens_root, 1)
    posterior = {index: read_checkpoint(
        leg1_root / f"member_{index:03d}" / CHECKPOINT_NAME)
        for index in prior}
    posterior_rmse = rmse_against_truth(posterior, truth_leg2, names)
    free_run = {index: forecast_leg(prior[index], cfg) for index in prior}
    free_run_rmse = rmse_against_truth(free_run, truth_leg2, names)

    manifest = json.loads(
        (ens_root / CYCLE_MANIFEST_NAME).read_text(encoding="utf-8"))
    assimilation = manifest["cycles"][0]["assimilation"]

    # The second leg, checked as a leg and not merely as "two entries".
    # Leg 1 must have been given the CUMULATIVE horizon (2 * leg_seconds
    # from the experiment start), because that is what the integrator
    # measures against the clock it restored; a leg-length horizon is the
    # defect this asserts against, and the runner above refuses it exactly
    # as the real integrator does.
    if len(manifest["cycles"]) != 2:
        raise RuntimeError(
            f"the cycle manifest records {len(manifest['cycles'])} cycles "
            "for a 2-cycle run; one entry per cycle is the contract")
    leg_records = {int(entry["cycle"]): entry for entry in manifest["cycles"]}
    second = leg_records[1]
    expected_total = 2.0 * float(cfg.leg_seconds)
    if float(second["run_seconds_total"]) != expected_total:
        raise RuntimeError(
            f"leg 1 was given run_seconds={second['run_seconds_total']}, "
            f"but the integrator measures the total forecast length from "
            f"the experiment start and leg 1 ends at {expected_total} s")
    if not second["restart_clocks"]["restarted"]:
        raise RuntimeError("leg 1 did not restart from an analysis")
    stated = second["restart_clocks"]["stated_start_seconds"]
    if sorted(map(float, stated.values())) != \
            [float(cfg.leg_seconds)] * cfg.members:
        raise RuntimeError(
            f"leg 1's restarts state clocks {sorted(stated.values())}, not "
            f"the {cfg.leg_seconds} s the timeline says the leg starts at")
    second_leg_seconds = json.loads(
        (leg1_root / "ensemble-manifest.json").read_text("utf-8"))
    advanced = [record["sim_seconds"]
                for record in second_leg_seconds["members"]]
    if any(value != float(cfg.leg_seconds) for value in advanced):
        raise RuntimeError(
            f"leg 1 advanced {advanced} s per member; each member must "
            f"integrate exactly one {cfg.leg_seconds} s leg")

    # Exactly zero, not nearly zero, outside every observation's lens.
    outside = unlocalized_region(cfg, grid)
    outside_count = int(np.count_nonzero(outside))
    if outside_count == 0:
        raise RuntimeError(
            "every gridpoint is inside some observation's localisation "
            "lens, so the exact-zero check would pass vacuously; widen "
            "observed_margin_cells or narrow the localisation radius")
    worst_outside = 0.0
    for index in increments:
        for name in names:
            worst_outside = max(
                worst_outside,
                float(np.abs(increments[index][name][outside]).max()))
    if worst_outside != 0.0:
        raise RuntimeError(
            f"increments are not exactly zero outside localisation "
            f"(largest {worst_outside:g} over {outside_count} gridpoints); "
            "the filter's structural-zero guarantee did not survive the "
            "adapter")
    inside_signal = max(
        float(np.abs(increments[index][name][~outside]).max())
        for index in increments for name in names)
    if inside_signal == 0.0:
        raise RuntimeError(
            "every increment is zero everywhere, so the exact-zero check "
            "above proved nothing")

    # -- products off the analysed ensemble --------------------------------
    products = None
    if cfg.products:
        products = _render_products(leg1_root, outdir / "products")

    report = {
        "schema": GATE_SCHEMA,
        "stability": "experimental",
        "members": cfg.members,
        "grid": {"nz": cfg.nz, "ny": cfg.ny, "nx": cfg.nx,
                 "dx_m": cfg.dx_m,
                 "identity_sha256": grid.identity_sha256()},
        "radars": [site.name for site in sites],
        "observations": {
            "path": str(obs_path),
            "sha256": obs_receipt.get("sha256"),
            "schema": obs_receipt.get("schema"),
            "status": obs_receipt.get("status"),
            "error_ms": cfg.obs_error_ms,
        },
        "method": captured["method"],
        "rmse": {
            "prior": prior_rmse,
            "analysis": analysis_rmse,
            "improvement_pct": {
                name: (100.0 * (prior_rmse[name] - analysis_rmse[name])
                       / prior_rmse[name]) if prior_rmse[name] else 0.0
                for name in (*names, "total")},
        },
        "second_leg_rmse": {
            "from_analysis": posterior_rmse,
            "free_run": free_run_rmse,
            "improvement_pct": {
                name: (100.0 * (free_run_rmse[name] - posterior_rmse[name])
                       / free_run_rmse[name]) if free_run_rmse[name] else 0.0
                for name in (*names, "total")},
        },
        "spread": {"prior": ensemble_spread(prior, names),
                   "analysis": ensemble_spread(analysis, names)},
        "localization": {
            "gridpoints_outside_every_lens": outside_count,
            "gridpoints_total": int(cfg.nz * cfg.ny * cfg.nx),
            "largest_increment_outside": worst_outside,
            "largest_increment_inside": inside_signal,
        },
        "cycles": [
            {
                "cycle": int(entry["cycle"]),
                "attempt": int(entry.get("attempt", 1)),
                "status": entry["status"],
                "forecast_seconds": entry["forecast_seconds"],
                #: The cumulative horizon the integrator was given.
                "run_seconds_total": entry["run_seconds_total"],
                "restarted": bool(entry["restart_clocks"]["restarted"]),
                "start_seconds": entry["restart_clocks"][
                    "expected_start_seconds"],
            }
            for entry in sorted(manifest["cycles"],
                                key=lambda item: item["cycle"])
        ],
        "assimilation": {
            "status": assimilation["status"],
            "attempt": assimilation["attempt"],
            #: Not "null": the receipt names the callable the engine
            #: invoked and carries the provenance that callable declared.
            "method_receipt": assimilation["method"],
            "positivity_policy": assimilation["positivity_policy"],
            "negative_points_total":
                assimilation["negative_points_total"],
            "mass_added_by_clip_total":
                assimilation["mass_added_by_clip_total"],
            "increment_contract": sorted(
                {receipt["contract"] for receipt in
                 assimilation["receipts"]}),
            "members_receipted": assimilation["member_count"],
        },
        "restart_from_analysis": [
            record.get("restart_from") is not None
            for record in json.loads(
                (leg1_root / "ensemble-manifest.json").read_text("utf-8")
            )["members"]],
        "products": products,
        "caveats": [
            "perfect model: truth and members share one forecast operator, "
            "so no inflation tuning may be read off these numbers",
            "the dycore is stood in for; this gate measures composition, "
            "not forecast skill",
            "amplitudes, localisation and obs error are sized for a "
            "seconds-long CPU run and are not calibrated against anything",
        ],
    }
    return report


def _render_products(leg_root, out):
    """enprod over the analysed ensemble.  Returns the filenames written."""

    try:
        import wrf  # noqa: F401
    except Exception as error:                       # pragma: no cover
        return {"rendered": False, "reason": f"wrf unavailable: {error}"}
    try:
        import matplotlib  # noqa: F401
    except Exception as error:                       # pragma: no cover
        return {"rendered": False, "reason": f"matplotlib unavailable: "
                                             f"{error}"}
    from gpuwm import cli

    out = Path(out)
    code = cli.main(["enprod", str(leg_root), "--field", "refl",
                     "--products", "mean,spread,prob,paintball,pmm",
                     "--threshold", "40", "--out", str(out), "--dpi", "72"])
    return {
        "rendered": code == 0,
        "exit_code": code,
        "files": sorted(path.name for path in out.glob("*.png")),
    }


_BASE_TOML = """
[experiment]
name = "synthetic_da_cycle"
start_time = 1970-01-01T18:00:00
run_seconds = {seconds}
restart_interval_s = {seconds}

[projection]
map_proj = "lambert"
ref_lat = 35.0
ref_lon = -97.0
truelat1 = 33.0
truelat2 = 37.0
stand_lon = -97.0

[shared]
nz = {nz}
ztop = {top}
p_top = 10000.0

[[domain]]
grid_id = 1
parent_id = 0
i_parent_start = 1
j_parent_start = 1
parent_grid_ratio = 1
parent_time_step_ratio = 1
nx = {nx}
ny = {ny}
time_step = 10
dx = {dx}
history_interval_s = {seconds}
"""
