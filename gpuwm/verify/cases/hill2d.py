"""Linear hydrostatic mountain-wave benchmark (Queney bell ridge).

A uniform flow U = 10 m/s in a constant-N (0.01 1/s) atmosphere crosses a
10 m bell ridge (halfwidth a = 10 km): Na/U = 10 (hydrostatic regime) and
Nh0/U = 0.01 (deeply linear), so the steady response is the classic Queney
(1948) vertically propagating hydrostatic mountain wave with upstream phase
tilt.  For h(x) = h0 a^2 / (x^2 + a^2) the analytic vertical velocity is

    w(x, z) = sqrt(rho_s/rho(z)) * U h0 a *
              [ (x^2 - a^2) sin(l z) - 2 a x cos(l z) ] / (x^2 + a^2)^2

with l = N/U (the sqrt-density factor is the WKB anelastic amplitude
growth), and the wave momentum flux M(z) = integral(rho u'w' dx) is
constant with height and equal to the linear value -pi/4 rho_s N U h0^2
(Eliassen-Palm; Durran 1990, eq. for the Witch-of-Agnesi drag).  This is
the standard terrain-dynamics verification: it
gates the metric/PGF wiring (phase structure), the lower kinematic BC
(forcing amplitude), and the damp_opt=3 sponge (no reflection, momentum
flux constant below the layer).

``run`` integrates 10 h on the GPU, time-averages the last hour (the
steady-state estimate; the transient gravity-wave bath oscillates around
the steady wave), and returns the gate metrics (``w_corr``: correlation
with the Queney solution below the sponge; ``mflux_dev``: max relative
deviation of M(z) from its mean in 1-9 km; ``mflux_mean``/
``mflux_linear`` and their ratio ``mflux_ratio``; ``w_max``; ``nan``);
with an ``outdir`` it also writes ``hill2d_w.png`` (model w cross-section
with the analytic solution overlaid) and ``wrfout_hill2d.nc``.

Hill height note: the plan specifies h0 = 10 m (ratified at the plan
level, commit f213484; an earlier plan draft said 1 m).  At FP32 the
model carries a round-off-excited gravity-wave noise bath whose w has
RMS ~3e-4 m/s in this channel (measured; it decorrelates hour to hour
and is absorbed only by the top sponge under periodic lateral BCs).  A
1 m ridge's steady wave (w RMS ~1.5e-4 m/s) would sit a factor ~2 BELOW
that floor, capping the correlation near 0.4 regardless of the dynamics.
h0 = 10 m keeps the flow deeply linear (Nh0/U = 0.01; linear-theory
error O(1%)) while lifting the signal a decade above the noise — the
linear Queney comparison the plan intends.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gpuwm.config import RunConfig
from gpuwm.core import constants as c
from gpuwm.core.grid import (BaseState, VerticalCoord, make_base_state,
                             make_vertical_coord)

#: Brunt-Vaisala frequency of the base state (1/s).
N_BV = 0.01
#: Uniform mean flow (m/s).
U_BAR = 10.0
#: Sponge base height (m): ztop - zdamp; gates apply below this.
Z_SPONGE = 10000.0
#: Momentum-flux gate window (m): above the surface layer, below the sponge.
Z_FLUX_MIN, Z_FLUX_MAX = 1000.0, 9000.0
#: Steady-state estimate: average the last hour, sampling every 2 min.
AVG_WINDOW_S, AVG_SAMPLE_S = 3600.0, 120.0

#: Pass gates, metric -> (lo, hi) with strict bounds, None = unbounded on
#: that side.  Single-sourced (Task 12): gpuwm.cli._GATES and
#: tests/test_case_hill2d.py both consume this export.  The steady-state w
#: must correlate with the analytic Queney solution below the sponge; the
#: wave momentum flux must be constant with height (within 15% in 1-9 km),
#: downward (drag, mflux_mean < 0), and of the linear-theory magnitude
#: (mflux_ratio = mflux_mean / mflux_linear within +/- 50%).
GATES = {
    "w_corr": (0.95, None),
    "mflux_dev": (None, 0.15),
    "mflux_mean": (None, 0.0),
    "mflux_ratio": (0.5, 1.5),
}


def default_config() -> RunConfig:
    """Plan setup: linear bell hill (h0 = 10 m; see the module-docstring
    FP32 note), a = 10 km, dx = 2 km, nz = 80, ztop = 16 km with a 6 km
    damp_opt=3 Rayleigh top, 10 h run.  512 km periodic channel keeps the
    wrapped far field ~1e-3 of the peak.  wrfout frames are hourly (the
    interesting output is the steady state, not the transient bath).
    """
    return RunConfig(nx=256, ny=1, nz=80, dx=2000.0, dy=2000.0,
                     ztop=16000.0, dt=8.0, run_seconds=36000.0,
                     damp_opt=3, zdamp=6000.0, dampcoef=0.2,
                     terrain_opt=1, hill_height=10.0,
                     hill_halfwidth=10000.0, output_interval_s=3600.0,
                     case="hill2d")


def sounding(z) -> np.ndarray:
    """Constant-N base state: theta(z) = 300 * exp(N^2 z / g)."""
    return 300.0 * np.exp(N_BV ** 2 * np.asarray(z, dtype=np.float64) / c.G)


def build(cfg: RunConfig, coord: VerticalCoord, base: BaseState):
    """Balanced terrain-following state with uniform flow U.

    The surface w gets the kinematic BC and the column is filled with the
    WRF ``start_em`` decay ``w(k) = w(0) * znw(k)^2`` (fill_w_flag), which
    shortens the spin-up transient.
    """
    from gpuwm.core.dycore import set_w_surface
    from gpuwm.core.state import init_at_rest

    state = init_at_rest(cfg, coord, base, terrain_z=base.terrain_z)
    state.u[...] = U_BAR
    set_w_surface(state, cfg)
    state.w[1:] = state.w[0][None] * (state.znw[1:, None, None] ** 2)
    return state


def queney_w(x, z, h0: float, a: float, rho_amp=1.0) -> np.ndarray:
    """Queney hydrostatic mountain-wave w (m/s) for the bell ridge.

    ``x``/``z`` broadcast together; ``rho_amp`` is the WKB anelastic
    amplitude factor sqrt(rho_s/rho(z)) (1 for the Boussinesq form).
    At z = 0 this reduces to the kinematic BC w = U dh/dx.
    """
    l = N_BV / U_BAR
    r2 = x ** 2 + a ** 2
    return (rho_amp * U_BAR * h0 * a
            * ((x ** 2 - a ** 2) * np.sin(l * z) - 2.0 * a * x * np.cos(l * z))
            / r2 ** 2)


def _metrics(w, u, nan: bool, cfg: RunConfig, base: BaseState) -> dict:
    """Gate metrics + the fields needed for the plot, all float64 host.

    ``w (nz+1, nx)`` / ``u (nz, nx+1)`` are the time-averaged steady-state
    fields.
    """
    x = (np.arange(cfg.nx) + 0.5) * cfg.dx - 0.5 * cfg.nx * cfg.dx
    zf = np.asarray(base.phb, dtype=np.float64)[:, 0, :] / c.G  # (nz+1, nx)

    # Model w at half levels vs the analytic solution with WKB amplitude
    # growth sqrt(rho_s/rho(z)) = sqrt(alb(z)/alb(0)).
    wh = 0.5 * (w[:-1] + w[1:])                            # (nz, nx)
    zh = 0.5 * (zf[:-1] + zf[1:])
    alb = np.asarray(base.alb, dtype=np.float64)[:, 0, :]  # (nz, nx)
    rho_amp = np.sqrt(alb / alb[0][None])
    w_ana = queney_w(x[None, :], zh, cfg.hill_height, cfg.hill_halfwidth,
                     rho_amp=rho_amp)
    mask = zh < Z_SPONGE
    wm, wa = wh[mask], w_ana[mask]
    w_corr = float(np.corrcoef(wm, wa)[0, 1])

    # Momentum flux M(z) = integral(rho u'w' dx) on half levels; u' is the
    # deviation from the level's x-mean (removes any mean-flow drift).
    uh = 0.5 * (u[:, :-1] + u[:, 1:])                      # (nz, nx)
    up = uh - uh.mean(axis=1, keepdims=True)
    mflux = cfg.dx * np.sum((1.0 / alb) * up * wh, axis=1)  # (nz,)
    zcol = zh.mean(axis=1)
    win = (zcol >= Z_FLUX_MIN) & (zcol <= Z_FLUX_MAX)
    m_mean = float(mflux[win].mean())
    mflux_dev = float(np.abs(mflux[win] - m_mean).max() / abs(m_mean))
    rho_s = float((1.0 / alb[0]).mean())
    m_lin = -np.pi / 4.0 * rho_s * N_BV * U_BAR * cfg.hill_height ** 2

    return {
        "nan": bool(nan or not np.all(np.isfinite(w))),
        "w_corr": w_corr,
        "mflux_dev": mflux_dev,
        "mflux_mean": m_mean,
        "mflux_linear": float(m_lin),
        "mflux_ratio": float(m_mean / m_lin),
        "w_max": float(np.abs(wm).max()),
        "_plot": (x, zh, wh, w_ana, mflux, zcol),
    }


def _plot(x, zh, wh, w_ana, path: Path) -> None:
    """Model w cross-section (filled) with the Queney solution overlaid
    (black contours), mm/s, below the sponge."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xk = x / 1000.0
    zk = zh.mean(axis=1) / 1000.0                          # ~flat (10 m hill)
    keep = zk < Z_SPONGE / 1000.0
    wmm, amm = 1000.0 * wh[keep], 1000.0 * w_ana[keep]
    lim = float(np.abs(amm).max())
    levels = np.linspace(-lim, lim, 17)
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    cs = ax.contourf(xk, zk[keep], wmm, levels=levels, cmap="RdBu_r",
                     extend="both")
    over = levels[::2]
    ax.contour(xk, zk[keep], amm, levels=over[np.abs(over) > 1e-12],
               colors="k", linewidths=0.6, linestyles="solid")
    ax.set_xlim(-100.0, 100.0)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("z (km)")
    ax.set_title("hill2d: w (mm/s) at t = 10 h — model (fill) vs "
                 "Queney (contours)")
    fig.colorbar(cs, ax=ax, label="w (mm/s)")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run(outdir: Path | None = None) -> dict:
    """Build the case, integrate 10 h, time-average the last hour as the
    steady state, return gate metrics; with an ``outdir``, also write the
    w cross-section PNG and ``wrfout_hill2d.nc`` (a frame every
    ``cfg.output_interval_s``)."""
    import cupy as cp

    from gpuwm.core.dycore import run_steps, stability_report
    from gpuwm.core.terrain import bell_hill

    cfg = default_config()
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, sounding, p_surf=cfg.p_surf, ztop=cfg.ztop,
                           terrain_z=bell_hill(cfg))
    state = build(cfg, coord, base)

    writer = None
    if outdir is not None:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        from gpuwm.io.wrfout import WrfoutWriter, state_frame, wrf_time_str
        writer = WrfoutWriter(outdir / "wrfout_hill2d.nc", nx=cfg.nx,
                              ny=cfg.ny, nz=cfg.nz, dx=cfg.dx, dy=cfg.dy,
                              title="gpuwm Queney mountain wave")

    n_total = int(round(cfg.run_seconds / cfg.dt))
    n_avg = int(round(AVG_WINDOW_S / cfg.dt))
    every = max(int(round(AVG_SAMPLE_S / cfg.dt)), 1)
    n_frame = max(int(round(cfg.output_interval_s / cfg.dt)), 1)
    w_sum = np.zeros((cfg.nz + 1, cfg.nx))
    u_sum = np.zeros((cfg.nz, cfg.nx + 1))
    n_samp = 0
    done = 0
    try:
        if writer is not None:
            writer.write_frame(wrf_time_str(0.0), state_frame(state))
        # Spin-up to the averaging window.  run_steps is a plain per-step
        # loop, so chunking it for frame output is bitwise identical to
        # the single call the metric-only path makes.
        while done < n_total - n_avg:
            n = (min(n_frame, n_total - n_avg - done) if writer is not None
                 else n_total - n_avg - done)
            run_steps(state, cfg, n=n)
            done += n
            if writer is not None and done % n_frame == 0:
                writer.write_frame(wrf_time_str(done * cfg.dt),
                                   state_frame(state))
        # Last hour: accumulate the steady-state average every 2 min (the
        # hourly frame boundaries coincide with sample boundaries).
        while done < n_total:
            n = min(every, n_total - done)
            run_steps(state, cfg, n=n)
            done += n
            w_sum += cp.asnumpy(state.w).astype(np.float64)[:, 0, :]
            u_sum += cp.asnumpy(state.u).astype(np.float64)[:, 0, :]
            n_samp += 1
            if writer is not None and done % n_frame == 0:
                writer.write_frame(wrf_time_str(done * cfg.dt),
                                   state_frame(state))
    finally:
        if writer is not None:
            writer.close()

    metrics = _metrics(w_sum / n_samp, u_sum / n_samp,
                       stability_report(state)["nan"], cfg, base)
    x, zh, wh, w_ana, _, _ = metrics.pop("_plot")
    if outdir is not None:
        _plot(x, zh, wh, w_ana, outdir / "hill2d_w.png")
    return metrics
