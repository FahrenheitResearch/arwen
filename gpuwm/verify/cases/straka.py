"""Straka et al. (1993) density-current benchmark.

A -15 K cold blob (4 km x 2 km half-axes, centered at z = 3 km) collapses in
a 300 K isentropic, at-rest atmosphere and spreads along the ground as a
density current whose head sheds Kelvin-Helmholtz rotors.  Physical mixing is
the constant-K diffusion (K = 75 m^2/s) of the published setup, so the
solution converges with resolution and can be gated against the reference:
at t = 900 s the converged (25 m) solution has theta'_min ~ -9.8 K and a
front (surface -1 K crossing) near 15 km; 100 m solutions land near these.

``run`` executes the benchmark on the GPU and returns the gate metrics
(``theta_min``, ``front_km``, ``w_max``, ``symmetry_err``, ``nan``); with an
``outdir`` it also writes ``straka_t900.png`` (theta' contours, x-z) and
``wrfout_straka.nc`` with a frame every ``cfg.output_interval_s``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gpuwm.config import RunConfig
from gpuwm.core import constants as c
from gpuwm.core.grid import (BaseState, VerticalCoord, make_base_state,
                             make_vertical_coord)


#: Pass gates, metric -> (lo, hi) with strict bounds, None = unbounded on
#: that side.  Single-sourced (Task 12): gpuwm.cli._GATES and
#: tests/test_case_straka.py both consume this export.  Values bracket the
#: converged 25 m reference (module docstring): theta'_min ~ -9.8 K, front
#: ~ 15 km, plus bounded w and the IC's preserved x-symmetry.
GATES = {
    "theta_min": (-11.0, -8.0),
    "front_km": (14.0, 17.0),
    "symmetry_err": (None, 0.05),
    "w_max": (None, 40.0),
}


def default_config() -> RunConfig:
    """Published Straka setup: 51.2 km x 6.4 km at dx = dz ~ 100 m,
    x in [-25.6, 25.6] km via domain-centered cell coordinates."""
    return RunConfig(nx=512, ny=1, nz=64, dx=100.0, dy=100.0, ztop=6400.0,
                     dt=0.5, run_seconds=900.0, khdif=75.0, kvdif=75.0,
                     case="straka")


def sounding(z) -> np.ndarray:
    """Isentropic base state: theta = 300 K at every height."""
    return np.full_like(np.asarray(z, dtype=np.float64), 300.0)


def build(cfg: RunConfig, coord: VerticalCoord, base: BaseState):
    """At-rest state carrying the Straka cold blob, hydrostatically balanced.

    Temperature perturbation T' = -15 K * (cos(pi*L) + 1)/2 inside the
    ellipse L = sqrt((x/4000)^2 + ((z - 3000)/2000)^2) <= 1, applied as a
    potential-temperature perturbation theta' = T'/pib(z) with pib the base
    Exner function.
    """
    from gpuwm.core.state import init_theta_perturbation

    exner_b = (np.asarray(base.pb, dtype=np.float64) / c.P0) ** c.RCP  # (nz,)

    def thp_func(x, z):
        L = np.sqrt((x[None, None, :] / 4000.0) ** 2
                    + ((z[:, None, None] - 3000.0) / 2000.0) ** 2)
        t_pert = np.where(L <= 1.0,
                          -15.0 * (np.cos(np.pi * L) + 1.0) / 2.0, 0.0)
        thp = t_pert / exner_b[:, None, None]
        return np.broadcast_to(thp, (cfg.nz, cfg.ny, cfg.nx))

    return init_theta_perturbation(cfg, coord, base, thp_func)


def _front_km(th_sfc: np.ndarray, x: np.ndarray) -> float:
    """x (km from center) of the rightmost surface-level -1 K theta'
    crossing, linearly interpolated between cell centers; NaN if the surface
    theta' never reaches -1 K."""
    below = th_sfc <= -1.0
    idx = np.nonzero(below[:-1] & ~below[1:])[0]
    if idx.size == 0:
        return float("nan")
    i = int(idx[-1])
    t0, t1 = float(th_sfc[i]), float(th_sfc[i + 1])
    frac = (-1.0 - t0) / (t1 - t0)
    return float(x[i] + frac * (x[i + 1] - x[i])) / 1000.0


def _plot(thp_xz: np.ndarray, x: np.ndarray, z: np.ndarray,
          path: Path) -> None:
    """Filled theta' contours -9..0 K step 1 K on the x >= 0 half-domain,
    aspect-correct (Straka et al. 1993 Fig. 1 style)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xk, zk = x / 1000.0, z / 1000.0
    levels = np.arange(-9.0, 1.0, 1.0)                # -9..0 K step 1 K
    fig, ax = plt.subplots(figsize=(11.0, 4.5))
    cs = ax.contourf(xk, zk, thp_xz, levels=levels, cmap="Blues_r",
                     extend="min")
    # Line overlay stops at -1 K: the 0 K contour only traces FP32-scale
    # noise aloft and would obscure the rotor structure.
    ax.contour(xk, zk, thp_xz, levels=levels[:-1], colors="k",
               linewidths=0.4)
    ax.set_xlim(0.0, 19.2)
    ax.set_ylim(0.0, float(zk[-1]))
    ax.set_aspect("equal")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("z (km)")
    ax.set_title("Straka density current: theta' (K) at t = 900 s")
    fig.colorbar(cs, ax=ax, label="theta' (K)", shrink=0.85)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run(outdir: Path | None = None) -> dict:
    """Build the case, integrate to t = 900 s, return the gate metrics;
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
        with WrfoutWriter(outdir / "wrfout_straka.nc", nx=cfg.nx, ny=cfg.ny,
                          nz=cfg.nz, dx=cfg.dx, dy=cfg.dy,
                          title="gpuwm Straka density current") as writer:
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
        "theta_min": float(thp.min()),
        "front_km": _front_km(thp[0, 0], x),
        "w_max": float(report["w_max"]),
        # IC is x-symmetric about the domain center and cell centers mirror
        # as x[nx-1-i] = -x[i], so theta'(x) vs theta'(-x) is a reversal.
        "symmetry_err": float(np.abs(thp - thp[:, :, ::-1]).max()),
    }

    if outdir is not None:
        _plot(thp[:, 0, :], x, z, outdir / "straka_t900.png")
    return metrics
