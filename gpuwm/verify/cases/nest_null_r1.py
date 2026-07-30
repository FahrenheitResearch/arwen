"""N2a dry ratio-1 null nest, executable through the public tree runner.

The parent follows WK82's construction chain but replaces
``init_moist_balanced`` with :func:`gpuwm.core.state.init_theta_perturbation`.
That is the Phase-1/2 dry convention used by Straka and IGW: load the same
discretely balanced dry base, apply the analytic theta perturbation and
re-integrate its geopotential, then add winds.  It is the dry counterpart of
WK82's moist rebalance because no qv loading or moist-pressure correction
exists when ``moist=False``.  The environmental theta sounding, quarter-
circle hodograph, and 3 K thermal therefore still drive nontrivial dry
dynamics while every physics scheme remains off.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from gpuwm.core.grid import make_base_state, make_vertical_coord
from gpuwm.verify.cases import wk82
from gpuwm.verify.cases.nest_ideal_common import (
    identity_halo_run, run_identity_case, synchronized_identity_config)
from gpuwm.verify.cases.nest_ideal_r1_moist import load_scaffold


GATE_METRIC = "null_nest_r1_dry_restriction"

# Full mutable dry prognostic state.  RK/acoustic arrays and p/al/alt are
# step-local or diagnostic, not peer-state prognostics.
DRY_PROGNOSTIC_FIELDS = ("u", "v", "w", "thp", "php", "mup")


def build_dry_wk82(cfg, coord, base):
    """Dry WK82 sounding + hodograph + thermal under Phase-1/2 init."""
    import cupy as cp

    from gpuwm.core.diagnostics import update_diagnostics
    from gpuwm.core.state import DTYPE, init_theta_perturbation

    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    y = (np.arange(ny) + 0.5) * cfg.dy - 0.5 * ny * cfg.dy

    def thp_func(x, z):
        xrad = x[None, None, :] / wk82.BUBBLE_RADIUS
        yrad = y[None, :, None] / wk82.BUBBLE_RADIUS
        zrad = ((z[:, None, None] - wk82.BUBBLE_ZC)
                / wk82.BUBBLE_ZC)
        radius = np.sqrt(xrad ** 2 + yrad ** 2 + zrad ** 2)
        thp = np.where(
            radius <= 1.0,
            wk82.BUBBLE_DELT * np.cos(0.5 * np.pi * radius) ** 2,
            0.0)
        return np.broadcast_to(thp, (nz, ny, nx))

    state = init_theta_perturbation(cfg, coord, base, thp_func)
    u, v = wk82.hodograph_uv(coord, cfg.ztop)
    state.u[...] = cp.asarray(u, dtype=DTYPE)[:, None, None]
    state.v[...] = cp.asarray(v, dtype=DTYPE)[:, None, None]
    # Match init_moist_balanced's time-zero EOS fill.  The shared assembly
    # deliberately re-derives this once more, as production prepare does,
    # before the executor constructs its initialized-state health validators.
    update_diagnostics(state)
    return state


def build_case():
    exp = synchronized_identity_config(load_scaffold(variant="n2a"))

    def state(cfg):
        coord = make_vertical_coord(cfg.nz)
        base = make_base_state(
            coord, lambda z: wk82.wk82_sounding(z)[0],
            p_surf=cfg.p_surf, ztop=cfg.ztop)
        return build_dry_wk82(cfg, coord, base)

    return exp, state(exp.root.run), state(identity_halo_run(exp))


def run(outdir: str | Path) -> dict[str, object]:
    exp, root_state, halo_state = build_case()
    return run_identity_case(
        exp=exp, root_state=root_state, halo_state=halo_state,
        metric=GATE_METRIC,
        field_names=DRY_PROGNOSTIC_FIELDS, outdir=outdir)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m gpuwm.verify.cases.nest_null_r1")
    parser.add_argument("--outdir", required=True, type=Path)
    args = parser.parse_args(argv)
    return 0 if run(args.outdir)["pass"] else 1


__all__ = [
    "DRY_PROGNOSTIC_FIELDS", "GATE_METRIC", "build_case",
    "build_dry_wk82", "main", "run",
]


if __name__ == "__main__":  # pragma: no cover - controller entry point
    raise SystemExit(main())
