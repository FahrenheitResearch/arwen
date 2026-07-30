"""Skamarock & Klemp (1994) nonhydrostatic inertia-gravity-wave benchmark.

A 0.01 K potential-temperature packet (half-width a = 5 km, lowest vertical
mode sin(pi z / H)) is released in a constant-N (0.01 1/s) channel of depth
H = 10 km with a uniform 20 m/s mean flow and rigid lids.  The packet is
advected downstream while dispersing into an alternating train of theta'
lobes, symmetric about mid-height.  The linear amplitude makes this the most
sensitive detector of acoustic-solve coefficient errors (wrong dispersion)
and base-state imbalance (spurious theta' >> 1e-3 K everywhere).

``run`` executes the benchmark on the GPU and returns the gate metrics
(``theta_p_max``, ``theta_p_min``, ``centroid_offset``, ``w_max``, ``nan``)
at t = 3000 s, gated against the Skamarock & Klemp (1994) nonhydrostatic
solution as reproduced in the ARW tech note (extrema ~ +2.8e-3 / -1.5e-3 K,
packet centroid ~ u*t = 60 km downstream of its release point); with an
``outdir`` it also writes ``igw_t3000.png`` (theta' contours, full domain)
and ``wrfout_igw.nc`` with a frame every ``cfg.output_interval_s``.
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
U_BAR = 20.0
#: Packet half-width (m) and release-point x (m, domain-centered).
A_HALF = 5.0e3
XC = -50.0e3
#: Channel depth H (m) of the sin(pi z / H) vertical mode (= model top).
H_DEPTH = 10.0e3
#: Perturbation amplitude (K).
THP_AMP = 0.01


#: Pass gates, metric -> (lo, hi) with strict bounds, None = unbounded on
#: that side.  Single-sourced (Task 12): gpuwm.cli._GATES and
#: tests/test_case_igw.py both consume this export.  Values bracket the
#: SK94/ARW-tech-note nonhydrostatic solution (theta' extrema ~ +2.8e-3 /
#: -1.5e-3 K, packet centroid ~ u*t = 60 km) plus a bounded-w sanity gate.
GATES = {
    "theta_p_max": (1.5e-3, 3.5e-3),
    "theta_p_min": (-2.5e-3, -1.0e-3),
    "centroid_offset": (45.0e3, 75.0e3),
    "w_max": (None, 0.1),
}


def default_config() -> RunConfig:
    """WRF standard nonhydrostatic IGW setup: 300 km x 10 km channel,
    dx = 1 km, nz = 100 (dz ~ 100 m), dt = 6 s (ARW dt = 6 s per km of dx
    rule), no diffusion, periodic in x."""
    return RunConfig(nx=300, ny=1, nz=100, dx=1000.0, dy=1000.0,
                     ztop=10000.0, dt=6.0, run_seconds=3000.0, case="igw")


def sounding(z) -> np.ndarray:
    """Constant-N base state: theta(z) = 300 * exp(N^2 z / g)."""
    return 300.0 * np.exp(N_BV ** 2 * np.asarray(z, dtype=np.float64) / c.G)


def build(cfg: RunConfig, coord: VerticalCoord, base: BaseState):
    """Balanced constant-N state carrying the SK94 packet and mean flow.

    theta' = 0.01 K * sin(pi z / H) / (1 + ((x - xc)/a)^2) with a = 5 km,
    xc = -50 km (so the packet crosses the domain center during the run),
    H = 10 km; uniform u = 20 m/s on top of the rebalanced at-rest state.
    """
    from gpuwm.core.state import init_theta_perturbation

    def thp_func(x, z):
        thp = (THP_AMP * np.sin(np.pi * z[:, None, None] / H_DEPTH)
               / (1.0 + ((x[None, None, :] - XC) / A_HALF) ** 2))
        return np.broadcast_to(thp, (cfg.nz, cfg.ny, cfg.nx))

    state = init_theta_perturbation(cfg, coord, base, thp_func)
    state.u[...] = U_BAR
    return state


def _centroid_offset(thp_xz: np.ndarray, x: np.ndarray) -> float:
    """x of the |theta'|-weighted centroid (m) relative to the release
    point xc; ~ u*t for the advected packet."""
    w = np.abs(thp_xz)
    return float((w * x[None, :]).sum() / w.sum()) - XC


def _plot(thp_xz: np.ndarray, x: np.ndarray, z: np.ndarray,
          path: Path) -> None:
    """theta' contours over the full domain, contour interval 5e-4 K from
    -3e-3 to 3e-3 K excluding zero (SK94 Fig. 1 / ARW tech note style)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xk, zk = x / 1000.0, z / 1000.0
    levels = np.concatenate([np.arange(-6, 0), np.arange(1, 7)]) * 5.0e-4
    fig, ax = plt.subplots(figsize=(11.0, 3.2))
    cs = ax.contourf(xk, zk, thp_xz, levels=levels, cmap="RdBu_r",
                     extend="both")
    ax.contour(xk, zk, thp_xz, levels=levels, colors="k", linewidths=0.4)
    ax.set_xlim(float(xk[0]), float(xk[-1]))
    ax.set_ylim(0.0, float(zk[-1]))
    ax.set_xlabel("x (km)")
    ax.set_ylabel("z (km)")
    ax.set_title("Skamarock-Klemp inertia-gravity wave: "
                 "theta' (K) at t = 3000 s")
    fig.colorbar(cs, ax=ax, label="theta' (K)")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run(outdir: Path | None = None) -> dict:
    """Build the case, integrate to t = 3000 s, return the gate metrics;
    with an ``outdir``, write a wrfout frame every ``cfg.output_interval_s``."""
    import cupy as cp

    from gpuwm.core.dycore import run_steps, stability_report

    cfg = default_config()
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, sounding, p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = build(cfg, coord, base)

    n_total = int(round(cfg.run_seconds / cfg.dt))
    if outdir is None:
        run_steps(state, cfg, n=n_total)
    else:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        from gpuwm.io.wrfout import WrfoutWriter, state_frame, wrf_time_str
        with WrfoutWriter(outdir / "wrfout_igw.nc", nx=cfg.nx, ny=cfg.ny,
                          nz=cfg.nz, dx=cfg.dx, dy=cfg.dy,
                          title="gpuwm SK94 inertia-gravity wave") as writer:
            writer.write_frame(wrf_time_str(0.0), state_frame(state))
            n_frame = max(int(round(cfg.output_interval_s / cfg.dt)), 1)
            done = 0
            while done < n_total:
                n = min(n_frame, n_total - done)
                run_steps(state, cfg, n=n)
                done += n
                writer.write_frame(wrf_time_str(done * cfg.dt),
                                   state_frame(state))

    report = stability_report(state, cfg)
    thp = cp.asnumpy(state.thp).astype(np.float64)     # (nz, ny, nx)
    x = (np.arange(cfg.nx) + 0.5) * cfg.dx - 0.5 * cfg.nx * cfg.dx
    z = state.height_half()

    metrics = {
        "nan": bool(report["nan"] or not np.all(np.isfinite(thp))),
        "theta_p_max": float(thp.max()),
        "theta_p_min": float(thp.min()),
        "centroid_offset": _centroid_offset(thp[:, 0, :], x),
        "w_max": float(report["w_max"]),
    }

    if outdir is not None:
        _plot(thp[:, 0, :], x, z, outdir / "igw_t3000.png")
    return metrics
