"""Moist Kessler bubble integration case (Phase 2 Task 10).

Pre-supercell shakeout: the full moist physics chain -- WK82 analytic
sounding (gpuwm.verify.cases.wk82), positive-definite scalar transport,
theta_m/loading coupling, Kessler warm rain, and the km-scale dissipation
package (km_opt=4 Smagorinsky + monotonic 6th-order filter, the Task 11
production combination) -- on a cheap 2-D x-z channel before the 3-D WK82
benchmark runs it in anger.

Setup (plan Task 10): dx = 1 km, dz ~ 500 m (nz = 40, ztop = 20 km with a
damp_opt=3 implicit-w Rayleigh top over the top 5 km), 128 km periodic
channel, dt = 6 s, 1 h.  A 3 K WK82 thermal transcribed from the local
quarter_ss initializer (dyn_em/module_initialize_ideal.F quarter_ss CASE
block: delt*cos(pi*RAD/2)^2 inside the unit ellipse RAD, 10 km horizontal
radius, 1.5 km center/half-depth) kicks off deep convection in the
unsheared WK82 sounding; the updraft condenses a cloud (qc > 1 g/kg),
autoconverts, and rains out to the surface within the hour.

``run`` integrates on the GPU and returns the gate metrics:

- ``nan``: any non-finite u/w/theta'/moisture at the end;
- ``w_max``: peak updraft (max of +w) over the run, sampled every step;
- ``qc_max``: peak cloud-water mixing ratio (kg/kg) over the run;
- ``qr_max``: peak rain-water mixing ratio (kg/kg) over the run;
- ``rain_mm``: largest accumulated surface rain (mm) in any column
  (Kessler RAINNC) -- "rain reaches the surface";
- ``mass_drift``: relative drift of the domain-total DRY-air mass
  sum(mub + mu') (the conserved quantity of the dry-mass coordinate;
  periodic channel, FP64 reduction);
- ``water_drift``: relative drift of the domain-total water --
  eta-measure column water sum(q_tot*(c1h*mu + c2h)*(-dnw))/g plus the
  Kessler surface rain (1 mm = 1 kg/m^2).  Reported for observability at
  a much looser bound than dry mass, because no single measure conserves
  the whole chain: transport conserves water in the eta measure (measured
  2e-7 over this hour with mp off), while WRF's Kessler sedimentation
  conserves it in its own rho*dzk column measure with rdzk the
  CENTER-TO-CENTER level spacing (module_mp_kessler.F), which differs
  from the eta-measure layer depth by ~(d ln alpha/dk)/2 ~ 3% per cell on
  this 500 m grid.  Measured on a mature storm state: one Kessler call
  closes its own measure to ~1e-6 absolute while leaving an eta-measure
  residual ~27% of that call's surface rain; over the hour the budget
  residual accumulates to ~2e-3 of total water (~3% of total rain) --
  a WRF-inherited discretization artifact, not a transport leak.

With an ``outdir``, writes ``moist_bubble.png`` (theta', qc, and qr
cross-sections at t = 1 h -- the plan's acceptance PNG) and
``wrfout_moist_bubble.nc``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gpuwm.config import RunConfig
from gpuwm.core import constants as c
from gpuwm.core.grid import (BaseState, VerticalCoord, make_base_state,
                             make_vertical_coord)
from gpuwm.verify.cases.wk82 import wk82_sounding

#: WK82 quarter_ss thermal (module_initialize_ideal.F): amplitude (K),
#: horizontal radius (m), center height / half-depth (m).
BUBBLE_DELT = 3.0
BUBBLE_RADIUS = 10000.0
BUBBLE_ZC = 1500.0

#: Pass gates, metric -> (lo, hi) with strict bounds, None = unbounded on
#: that side.  Single-sourced (Task 12): gpuwm.cli._GATES and
#: tests/test_moist.py both consume this export.  Plan Task 10 acceptance:
#: cloud forms (qc_max > 1 g/kg), rain reaches the surface, updraft peaks
#: in 10-35 m/s, dry mass conserved to 1e-6 relative (periodic channel),
#: and the water budget closes to the documented WRF-inherited
#: cross-measure level (module docstring).
GATES = {
    "qc_max": (1.0e-3, None),
    "rain_mm": (0.05, None),
    "w_max": (10.0, 35.0),
    "mass_drift": (None, 1.0e-6),
    "water_drift": (None, 5.0e-3),
}


def default_config() -> RunConfig:
    """Plan setup: dx = 1 km / dz = 500 m, 1 h, Kessler + PD moisture, and
    the Task 11 production dissipation (km_opt=4, diff_6th_opt=2) with the
    implicit-w Rayleigh top over 15-20 km."""
    return RunConfig(nx=128, ny=1, nz=40, dx=1000.0, dy=1000.0,
                     ztop=20000.0, dt=6.0, run_seconds=3600.0,
                     moist=True, mp_physics=1, moist_adv_opt=1,
                     km_opt=4, c_s=0.25, diff_6th_opt=2,
                     diff_6th_factor=0.12,
                     damp_opt=3, zdamp=5000.0, dampcoef=0.2,
                     case="moist_bubble")


def build(cfg: RunConfig, coord: VerticalCoord, base: BaseState):
    """Balanced moist WK82 state with the 3 K quarter_ss thermal.

    The base state carries the WK82 theta profile; ``init_moist_balanced``
    lays the WK82 qv on it and rebalances p'/phi' exactly as the WRF
    quarter_ss initializer does (moist hydrostatic pressure from qv, alpha
    recomputed with the bubble's theta, geopotential re-integrated).
    """
    from gpuwm.core.moist import init_moist_balanced

    def thp_func(x, z):
        xrad = x[None, None, :] / BUBBLE_RADIUS
        zrad = (z[:, None, None] - BUBBLE_ZC) / BUBBLE_ZC
        rad = np.sqrt(xrad ** 2 + zrad ** 2)
        thp = np.where(rad <= 1.0,
                       BUBBLE_DELT * np.cos(0.5 * np.pi * rad) ** 2, 0.0)
        return np.broadcast_to(thp, (cfg.nz, cfg.ny, cfg.nx))

    return init_moist_balanced(cfg, coord, base,
                               lambda z: wk82_sounding(z)[1], thp_func)


def _dry_mass(state) -> float:
    """Domain-total dry-air column mass sum(mub + mu'), FP64 on device."""
    import cupy as cp
    return float(cp.sum((state.mub2d + state.mup).astype(cp.float64)))


def _water_mass(state, rain_kgm2: float = 0.0) -> float:
    """Domain-total water in the eta measure (kg/m^2 summed over columns):
    sum_k q_tot*(c1h*mu + c2h)*(-dnw)/g + accumulated surface rain."""
    import cupy as cp
    qtot = (state.qv + state.qc + state.qr).astype(cp.float64)
    chm = (state.c1h[:, None, None].astype(cp.float64)
           * (state.mub2d + state.mup)[None].astype(cp.float64)
           + state.c2h[:, None, None].astype(cp.float64))
    dnw = state.dnw[:, None, None].astype(cp.float64)
    return float(cp.sum(qtot * chm * (-dnw)) / c.G) + rain_kgm2


def _plot(thp, qc, qr, x, z, path: Path) -> None:
    """theta' / qc / qr x-z cross-sections at t = 1 h, one PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xk, zk = x / 1000.0, z / 1000.0
    tlim = float(np.abs(thp).max()) or 1.0
    fig, axes = plt.subplots(3, 1, figsize=(9.0, 9.5), sharex=True)
    panels = (
        (thp, "theta' (K)", "RdBu_r", np.linspace(-tlim, tlim, 17)),
        (1000.0 * qc, "qc (g/kg)", "Blues",
         np.linspace(0.0, max(1000.0 * qc.max(), 0.1), 11)),
        (1000.0 * qr, "qr (g/kg)", "Greens",
         np.linspace(0.0, max(1000.0 * qr.max(), 0.1), 11)),
    )
    for ax, (f, label, cmap, levels) in zip(axes, panels):
        cs = ax.contourf(xk, zk, f, levels=levels, cmap=cmap, extend="both")
        ax.set_ylabel("z (km)")
        ax.set_ylim(0.0, 16.0)
        fig.colorbar(cs, ax=ax, label=label)
    axes[0].set_title("moist_bubble: WK82 sounding + 3 K bubble + Kessler "
                      "at t = 1 h")
    axes[-1].set_xlabel("x (km)")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run(outdir: Path | None = None) -> dict:
    """Build the case, integrate 1 h, return the gate metrics; with an
    ``outdir``, also write the theta'/qc/qr PNG and
    ``wrfout_moist_bubble.nc`` (a frame every ``cfg.output_interval_s``,
    moisture scalars and accumulated RAINNC included)."""
    import cupy as cp

    from gpuwm.config import validate_run_config
    from gpuwm.core.dycore import run_steps, stability_report

    # The legacy idealized setup disables PBL.  Native WRF consequently
    # adds vertical_diffusion_2 for km_opt=4, which gpuwm does not yet carry.
    # Refuse before device allocation; real74's PBL-on production path stays
    # supported.
    cfg = validate_run_config(default_config())
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, lambda z: wk82_sounding(z)[0],
                           p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = build(cfg, coord, base)

    mass0 = _dry_mass(state)
    water0 = _water_mass(state)
    rainnc = state.scratch((cfg.ny, cfg.nx), "mp_rainnc")

    writer = None
    if outdir is not None:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        from gpuwm.io.wrfout import WrfoutWriter, state_frame, wrf_time_str
        writer = WrfoutWriter(outdir / "wrfout_moist_bubble.nc", nx=cfg.nx,
                              ny=cfg.ny, nz=cfg.nz, dx=cfg.dx, dy=cfg.dy,
                              title="gpuwm moist Kessler bubble")

    def write_frame(t_s: float) -> None:
        writer.write_frame(wrf_time_str(t_s), state_frame(state)
                           | {"RAINNC": cp.asnumpy(rainnc)})

    # Peak metrics are sampled every step: the 10-35 m/s updraft gate
    # should see the true maximum, and the three reductions on this small
    # 2-D channel are negligible next to the step itself.
    n_total = int(round(cfg.run_seconds / cfg.dt))
    n_frame = max(int(round(cfg.output_interval_s / cfg.dt)), 1)
    w_max = qc_max = qr_max = 0.0
    try:
        if writer is not None:
            write_frame(0.0)
        for n in range(1, n_total + 1):
            run_steps(state, cfg, n=1)
            w_max = max(w_max, float(state.w.max()))
            qc_max = max(qc_max, float(state.qc.max()))
            qr_max = max(qr_max, float(state.qr.max()))
            if writer is not None and n % n_frame == 0:
                write_frame(n * cfg.dt)
    finally:
        if writer is not None:
            writer.close()

    report = stability_report(state, cfg)
    # rainnc is mm; 1 mm of rain is RHOWATER/1000 = 1 kg/m^2.  The water
    # budget wants the domain sum, the "rain reached the surface" gate the
    # largest column.
    rain_sum = float(cp.sum(rainnc.astype(cp.float64))) * c.RHOWATER / 1000.0
    finite = all(bool(cp.isfinite(q).all()) for q in
                 (state.qv, state.qc, state.qr))

    metrics = {
        "nan": bool(report["nan"] or not finite),
        "w_max": w_max,
        "qc_max": qc_max,
        "qr_max": qr_max,
        "rain_mm": float(rainnc.max()),
        "mass_drift": abs(_dry_mass(state) - mass0) / mass0,
        "water_drift": abs(_water_mass(state, rain_sum) - water0) / water0,
    }

    if outdir is not None:
        x = (np.arange(cfg.nx) + 0.5) * cfg.dx - 0.5 * cfg.nx * cfg.dx
        z = state.height_half()
        _plot(cp.asnumpy(state.thp).astype(np.float64)[:, 0, :],
              cp.asnumpy(state.qc).astype(np.float64)[:, 0, :],
              cp.asnumpy(state.qr).astype(np.float64)[:, 0, :],
              x, z, outdir / "moist_bubble.png")
    return metrics
