"""Dry convective boundary layer: one closure against another.

A flat, horizontally homogeneous, dry column set under a surface warmer
than the air above it.  The surface layer converts that contrast into a
positive heat flux, the boundary-layer closure mixes it upward, and a
well-mixed layer grows into the overlying stable stratification.  This
is the oldest and least ambiguous boundary-layer test there is: the
answer has a known SHAPE (a near-adiabatic mixed layer capped by an
inversion that rises through the morning) even where no reference run
exists to compare against number by number.

Why it is in the tree
---------------------
It is the smallest configuration in which two boundary-layer schemes can
be asked the same question and their answers laid side by side.  The
gates below therefore check VALIDITY -- did the closure produce a
physically admissible boundary layer at all -- and deliberately do NOT
encode a preferred profile.  A scheme that mixes more deeply than
another is not thereby wrong, and this case is not the instrument that
would decide it.

Both the WRF-transcribed schemes and the experimental ArWen-only closure
can drive it, selected exactly as a user selects them: through
``bl_pbl_physics``.  ``run`` executes one closure and returns its gate
metrics; ``compare`` runs two and returns both metric sets plus the
profile pair, which is what makes the side-by-side readable.

Setup
-----
Doubly periodic 16 x 16 columns at dx = 500 m, 40 levels to 2 km
(dz ~ 50 m), initial theta 300 K with a 3 K/km stable lapse rate, at
rest apart from a light 2 m/s geostrophic-scale wind so the surface
layer has a shear scale to work with, land surface at a fixed 305 K --
5 K warmer than the air it touches.  Two hours of model time.  No
moisture, no radiation, no cumulus: the only thing moving heat upward is
the boundary-layer closure under test, which is the point.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gpuwm.config import RunConfig, validate_run_config
from gpuwm.core.grid import (BaseState, VerticalCoord, make_base_state,
                             make_vertical_coord)

#: Surface temperature (K) and the air temperature it sits under.  The
#: 5 K contrast is what drives the whole case.
TSK = 305.0
THETA_SURFACE = 300.0
#: Stable lapse rate the mixed layer grows into (K/m).
LAPSE = 0.003
#: Initial wind (m/s).  Small but nonzero: a surface layer with no shear
#: scale produces no friction velocity, and the closures under test read
#: one.
U_INITIAL = 2.0
#: Model seconds.
RUN_SECONDS = 7200.0

#: Pass gates, metric -> (lo, hi), None = unbounded on that side.
#:
#: VALIDITY, not skill.  Each bound below is a statement about whether a
#: convective boundary layer formed and stayed physical, and every one of
#: them is wide enough to admit closures that disagree with each other:
#:
#: ``nan`` is deliberately NOT a gate row: the driver checks it first
#: and by itself, and the returned metrics carry it beside the gate keys
#: (the case-registry contract).  Gates here are OPEN intervals with
#: strict bounds, ``lo < v < hi``, which is why a zero-valued metric can
#: never be gated with a zero bound.
#:
#: * ``theta_surface_warming`` -- the surface heat flux reached the air.
#:   A closure that mixes nothing leaves this at zero.
#: * ``mixed_layer_depth`` -- an adiabatic layer formed and is bounded by
#:   the domain.  The lower bound rejects "no boundary layer at all"; the
#:   upper rejects a layer that has run away through the model top.
#: * ``theta_gradient_aloft`` -- the stable stratification above the
#:   mixed layer SURVIVED.  This is the gate that catches a closure
#:   mixing the whole column indiscriminately, which is the classic
#:   failure and is invisible in a depth check alone.
#: * ``wind_speed_max`` -- momentum stayed bounded; surface drag did not
#:   run away or reverse.
#: * ``surface_heat_consistency`` -- the column's integrated heat GAIN,
#:   divided by an independent bulk-aerodynamic estimate of what the
#:   surface could have delivered over the run (C_H |V| dTheta t, with
#:   C_H |V| = 0.02 m/s at this case's 2 m/s wind).  Order unity is the
#:   correct answer and the band says so: near zero means the closure
#:   moved no heat off the surface, and far above one means it created
#:   heat the surface never supplied.  The band is deliberately a factor
#:   of a few wide -- the denominator is an estimate, not a budget, and
#:   this gate is here to catch a broken energy path, not to score one
#:   closure against another.
GATES = {
    "theta_surface_warming": (0.05, 20.0),
    "mixed_layer_depth": (100.0, 1800.0),
    "theta_gradient_aloft": (1.0e-3, None),
    "wind_speed_max": (None, 30.0),
    "surface_heat_consistency": (0.2, 3.0),
}


def default_config(bl_pbl_physics: int = 1) -> RunConfig:
    """The case configuration for one boundary-layer closure.

    ``km_opt`` follows the closure: the experimental closure supplies the
    mixing the km_opt operator would otherwise apply, and stacking the
    two would double-count it, so it is refused at config load.  Reading
    the requirement off the registry rather than hard-coding a number
    keeps this case honest if another such scheme is ever registered.
    """
    from gpuwm.config import SASE_PBL_SCHEME

    supplies_own_mixing = bl_pbl_physics == SASE_PBL_SCHEME
    return validate_run_config(RunConfig(
        nx=16, ny=16, nz=40, dx=500.0, dy=500.0, ztop=2000.0,
        dt=3.0, run_seconds=RUN_SECONDS, time_step_sound=4,
        # The state carries moisture ARRAYS (every boundary-layer
        # scheme in this engine mixes water vapour and is refused a dry
        # state) but the sounding is dry and no microphysics runs, so
        # the case stays thermodynamically a dry CBL.
        moist=True, mp_physics=0,
        bl_pbl_physics=bl_pbl_physics,
        sf_sfclay_physics=91,          # classic MM5 surface layer
        sf_surface_physics=0,          # prescribed surface temperature
        km_opt=0 if supplies_own_mixing else 4,
        c_s=0.25 if not supplies_own_mixing else 0.25,
        bldt=0.0, radt=0.0, cu_physics=0,
        case="cbl_dry"))


def sounding(z) -> np.ndarray:
    """Stably stratified dry column the mixed layer must grow into."""
    return THETA_SURFACE + LAPSE * np.asarray(z, dtype=np.float64)


def build(cfg: RunConfig, coord: VerticalCoord, base: BaseState):
    """At rest apart from a light uniform wind; no theta perturbation.

    The boundary layer is driven entirely by the surface contrast, so
    there is nothing to seed: any structure that appears was made by the
    closure under test.  A small deterministic wind perturbation breaks
    exact horizontal homogeneity so a three-dimensional closure has
    something to sense, without seeding the thermodynamics.
    """
    import cupy as cp

    from gpuwm.core.state import init_at_rest

    state = init_at_rest(cfg, coord, base)
    rng = np.random.default_rng(20260801)
    field = np.full(tuple(state.u.shape), U_INITIAL, dtype=np.float64)
    field += 0.01 * rng.standard_normal(field.shape)
    field[:, :, -1] = field[:, :, 0]                    # periodic seam
    state.u[...] = cp.asarray(field, dtype=state.u.dtype)
    return state


def _profiles(state, cfg) -> dict[str, np.ndarray]:
    """Horizontally averaged column profiles, host-side FP64."""
    import cupy as cp

    def mean(field):
        return np.asarray(cp.asnumpy(field), dtype=np.float64).mean(
            axis=(1, 2))

    theta = mean(state.thp) + np.asarray(
        cp.asnumpy(state.thb), dtype=np.float64).reshape(-1)[:cfg.nz]
    phb = np.asarray(cp.asnumpy(state.phb), dtype=np.float64)
    php = np.asarray(cp.asnumpy(state.php), dtype=np.float64)
    phi = phb.reshape(-1, 1, 1) + php if phb.ndim == 1 else phb + php
    z_face = phi.mean(axis=(1, 2)) / 9.81
    z = 0.5 * (z_face[:-1] + z_face[1:])
    u = mean(state.u[:, :, :cfg.nx])
    v = mean(state.v[:, :cfg.ny, :])
    out = {"z": z, "theta": theta, "u": u, "v": v,
           "speed": np.hypot(u, v)}
    e_sgs = getattr(state, "e_sgs", None)
    if e_sgs is not None:
        out["tke"] = mean(e_sgs)
    return out


def _metrics(initial, final, cfg) -> dict[str, float]:
    """Gate metrics from the initial/final horizontally averaged state."""
    z, theta = final["z"], final["theta"]
    warming = float(theta[0] - initial["theta"][0])
    # Mixed-layer depth: the highest level whose potential temperature is
    # still within 0.5 K of the surface value.  A blunt definition on
    # purpose -- it makes no assumption about which closure produced it.
    within = np.nonzero(np.abs(theta - theta[0]) <= 0.5)[0]
    depth = float(z[within[-1]]) if within.size else 0.0
    # Stratification retained above the mixed layer, over the top
    # quarter of the column.
    top = slice(int(0.75 * cfg.nz), cfg.nz)
    gradient = float(np.gradient(theta[top], z[top]).mean())
    speed_max = float(np.max(final["speed"]))
    # Column heat gain against an INDEPENDENT bulk-aerodynamic estimate
    # of the surface supply.  The estimate is deliberately not taken
    # from the model's own flux: a check that reads back the number it
    # is checking proves nothing.
    d_theta = float(np.trapezoid(theta - initial["theta"], z))
    supply = abs(TSK - THETA_SURFACE) * 0.02 * RUN_SECONDS
    consistency = d_theta / supply if supply > 0.0 else 0.0
    nan = float(not np.all(np.isfinite(np.concatenate(
        [theta, final["u"], final["v"]]))))
    return {"nan": nan,
            "theta_surface_warming": warming,
            "mixed_layer_depth": depth,
            "theta_gradient_aloft": gradient,
            "wind_speed_max": speed_max,
            "surface_heat_consistency": consistency}


def run(outdir=None, *, bl_pbl_physics: int = 1) -> dict[str, float]:
    """Integrate one closure and return its gate metrics.

    With ``outdir`` the horizontally averaged initial and final profiles
    are written beside the metrics as ``cbl_dry_<selector>.npz`` so a
    caller can plot the pair without re-running.
    """
    from gpuwm.core.dycore import run_steps
    from gpuwm.core.physics import initialize_physics

    cfg = default_config(bl_pbl_physics)
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, sounding, p_surf=cfg.p_surf,
                           ztop=cfg.ztop)
    state = build(cfg, coord, base)
    initialize_physics(state, cfg, landmask=1.0, tsk=TSK)
    initial = _profiles(state, cfg)
    run_steps(state, cfg, int(round(RUN_SECONDS / cfg.dt)))
    final = _profiles(state, cfg)
    metrics = _metrics(initial, final, cfg)
    if outdir is not None:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        np.savez(outdir / f"cbl_dry_{int(bl_pbl_physics)}.npz",
                 **{f"initial_{k}": v for k, v in initial.items()},
                 **{f"final_{k}": v for k, v in final.items()})
    return metrics


def compare(outdir=None, *, schemes=(1,)) -> dict[str, object]:
    """Run several closures on the identical case and return both sets.

    Differences are RETURNED, never reduced to a verdict: this case
    states whether each closure produced a physically admissible
    convective boundary layer, and leaves which one is better to an
    instrument that can actually answer it.
    """
    results: dict[str, object] = {"gates": dict(GATES), "schemes": {}}
    for selector in schemes:
        metrics = run(outdir, bl_pbl_physics=int(selector))
        passed = all(
            (lo is None or metrics[name] >= lo)
            and (hi is None or metrics[name] <= hi)
            for name, (lo, hi) in GATES.items())
        results["schemes"][str(int(selector))] = {
            "metrics": metrics, "pass": bool(passed)}
    return results


def main(argv=None) -> int:
    """``python -m gpuwm.verify.cases.cbl_dry [--out DIR] [SELECTOR...]``"""
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("selectors", nargs="*", type=int, default=[1],
                        help="bl_pbl_physics values to compare")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    report = compare(args.out, schemes=tuple(args.selectors))
    print(json.dumps(report, indent=2, sort_keys=True, default=float))
    return 0 if all(entry["pass"]
                    for entry in report["schemes"].values()) else 1


if __name__ == "__main__":                       # pragma: no cover
    raise SystemExit(main())
