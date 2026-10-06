"""What forcing the ensemble members does to the ensemble: spread against
innovation, with radar latent heating on and off.  One draw, not a skill
claim.

Why this exists.  HRRR forces only its deterministic pre-forecast with
radar latent heating (``parm/conus/hrrr_wrfpre.nl:109``); its ensemble
members run ``mp_tend_radar = 0`` (``parm/hrrrdas/hrrrdas_wrf.nl:101``).
The research cycle (``tools/da_cycle_prepared.py --radar-tten``) forces the
members instead, with one slot per leg built from the leg's observation
file (valid at the end of the leg) against each member's own state at the
start of the leg, and then the filter analyses those same members against
that same file.  Two effects are expected and this measures both on the
real model: the members get nearly the same heating where the radar
covers, so their spread shrinks there; and the observations the filter is
about to use have already pulled the members toward themselves, so the
innovation shrinks too.  A filter that sees both small spread and small
innovation weights the observation as if it were new information.

Set-up (the case of ``tools/radar_tten_proof/twin.py``): the Weisman-Klemp
supercell sounding and hodograph on a small grid.  Truth starts from the
case's 3 K, 10 km thermal.  Each member starts from a thermal whose
amplitude and radius are drawn around the truth's (``--delt-range``,
``--radius-km-range``) plus a small random low-level theta perturbation,
so the members disagree about the storm's strength and timing.  Every
member runs ``--spinup-min`` free, then one leg of ``--leg-min`` twice from
the same leg-start state: unforced (flag off), and forced with one slot
built from the truth's reflectivity at the end of the leg against that
member's state (flag on, HRRR's 0.07 K/s clamp).  The observation is the
truth's simulated reflectivity at the end of the leg with full coverage:
echo at or above ``--echo-dbz``, observed clear air below it.

Reported at the end of the leg, for each arm:

* theta spread (root mean ensemble variance, ``ddof = 1``) over the echo
  points, over the points the forcing heated in any member, and over the
  observed clear air;
* in observation space at the echo points: the innovation (observation
  minus the ensemble-mean simulated reflectivity) as mean and RMS, the
  ensemble spread of the simulated reflectivity, and their ratio;
* at the clear-air points: the fraction where the ensemble mean holds a
  spurious echo.

All statistics are reduced on the device (CuPy); the host only formats
the receipt.

    python -m tools.radar_tten_proof.spread --out <receipt.json>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def _member_start(cfg, *, delt, radius_m, noise_k, noise_top_m, seed):
    """A WK82 start with its own thermal and a low-level theta draw."""
    import cupy as cp
    from gpuwm.core.diagnostics import update_diagnostics
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.state import DTYPE
    from gpuwm.verify.cases import wk82

    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, lambda z: wk82.wk82_sounding(z)[0],
                           p_surf=cfg.p_surf, ztop=cfg.ztop)
    saved = (wk82.BUBBLE_DELT, wk82.BUBBLE_RADIUS)
    try:
        wk82.BUBBLE_DELT, wk82.BUBBLE_RADIUS = float(delt), float(radius_m)
        state = wk82.build(cfg, coord, base)
    finally:
        wk82.BUBBLE_DELT, wk82.BUBBLE_RADIUS = saved
    if noise_k > 0.0:
        rng = cp.random.default_rng(seed)
        low = cp.asarray(state.height_half() < noise_top_m)[:, None, None]
        noise = rng.standard_normal(state.thp.shape, dtype=cp.float32)
        state.thp += (DTYPE(noise_k) * noise * low).astype(state.thp.dtype)
        update_diagnostics(state)
    return state


def _integrate(state, cfg, minutes, *, forcing=None):
    from gpuwm.core.dycore import step
    from gpuwm.da import radar_tten

    steps = int(round(minutes * 60.0 / cfg.dt))
    if forcing is not None:
        radar_tten.attach(state, forcing, cfg)
    try:
        for _ in range(steps):
            step(state, cfg)
    finally:
        if forcing is not None:
            radar_tten.detach(state)
    return steps


def _document(refl_host, echo_dbz):
    z_mask = (refl_host >= echo_dbz).astype(np.int8)
    return {"variables": {"z_obs": refl_host.astype(np.float32),
                          "z_mask": z_mask,
                          "z0_mask": (1 - z_mask).astype(np.int8)},
            "clear_air_source": "finite_below_floor"}


def _spread(stack, mask):
    """Root of the mean ensemble variance over ``mask``."""
    import cupy as cp

    if not bool(mask.any()):
        return None
    var = cp.var(stack.astype(cp.float64), axis=0, ddof=1)
    return float(cp.sqrt(var[mask].mean()))


def _arm_metrics(thp, refl, y, echo, clear, heated, echo_dbz):
    import cupy as cp

    mean_h = refl.astype(cp.float64).mean(axis=0)
    d = (y.astype(cp.float64) - mean_h)[echo]
    spread_h = cp.sqrt(cp.var(refl.astype(cp.float64), axis=0,
                              ddof=1)[echo].mean())
    rms = cp.sqrt((d * d).mean())
    return {
        "theta_spread_k": {
            "echo_points": _spread(thp, echo),
            "points_heated_by_the_forcing": _spread(thp, heated),
            "clear_air_points": _spread(thp, clear),
        },
        "reflectivity_at_echo_points": {
            "innovation_mean_dbz": float(d.mean()),
            "innovation_rms_dbz": float(rms),
            "ensemble_spread_dbz": float(spread_h),
            "spread_over_innovation_rms": float(spread_h / rms),
        },
        "clear_air_points_with_ensemble_mean_echo_fraction": float(
            ((mean_h >= echo_dbz) & clear).sum() / max(int(clear.sum()), 1)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--nx", type=int, default=80)
    parser.add_argument("--nz", type=int, default=50)
    parser.add_argument("--mp", type=int, default=8)
    parser.add_argument("--members", type=int, default=8)
    parser.add_argument("--spinup-min", type=float, default=30.0)
    parser.add_argument("--leg-min", type=float, default=15.0)
    parser.add_argument("--echo-dbz", type=float, default=5.0)
    parser.add_argument("--delt-range", type=float, nargs=2,
                        default=(2.0, 4.0))
    parser.add_argument("--radius-km-range", type=float, nargs=2,
                        default=(7.0, 13.0))
    parser.add_argument("--noise-k", type=float, default=0.2)
    parser.add_argument("--noise-top-m", type=float, default=1500.0)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    import cupy as cp
    from gpuwm.da import obsop, radar_tten

    from tools.radar_tten_proof.twin import _config

    cfg = _config(args.nx, args.nz, args.mp)
    started = time.time()
    total = args.spinup_min + args.leg_min

    truth = _member_start(cfg, delt=3.0, radius_m=10000.0, noise_k=0.0,
                          noise_top_m=0.0, seed=0)
    _integrate(truth, cfg, total)
    y = obsop.simulated_reflectivity(truth, cfg).copy()
    document = _document(cp.asnumpy(y), args.echo_dbz)
    truth_storm = {"max_w_ms": float(truth.w.max()),
                   "area_35dbz_km2": float(
                       (y.max(axis=0) >= 35.0).sum() * cfg.dx * cfg.dy
                       / 1.0e6)}
    del truth

    draws = np.random.default_rng(args.seed)
    members = [{"delt_k": float(draws.uniform(*args.delt_range)),
                "radius_m": 1000.0 * float(draws.uniform(
                    *args.radius_km_range)),
                "seed": int(args.seed * 1000 + m)}
               for m in range(args.members)]
    fields = {arm: {"thp": [], "refl": [], "w_max": [], "area_35": [],
                    "h_max": []}
              for arm in ("off", "on")}
    heated = cp.zeros(y.shape, dtype=bool)
    receipts = []
    identical_starts = True
    for member in members:
        start_thp = {}
        for arm in ("off", "on"):
            state = _member_start(cfg, delt=member["delt_k"],
                                  radius_m=member["radius_m"],
                                  noise_k=args.noise_k,
                                  noise_top_m=args.noise_top_m,
                                  seed=member["seed"])
            _integrate(state, cfg, args.spinup_min)
            start_thp[arm] = state.thp.copy()
            forcing = None
            if arm == "on":
                forcing = radar_tten.build_forcing_from_documents(
                    state, [document], [args.leg_min])
                slot = forcing.slots[0]
                heated |= (slot > 0.0) & (slot <= 1.0)
            _integrate(state, cfg, args.leg_min, forcing=forcing)
            if forcing is not None:
                record = forcing.receipt()
                receipts.append({
                    "calls_by_slot": record["calls_by_slot"],
                    "mp_tend_lim_k_per_s": record["mp_tend_lim_k_per_s"],
                    "points_heated": record["slots"][0]["points_heated"],
                    "points_zero_tendency": record["slots"][0][
                        "points_zero_tendency"]})
                del forcing, slot
            refl = obsop.simulated_reflectivity(state, cfg).copy()
            fields[arm]["thp"].append(state.thp.copy())
            fields[arm]["refl"].append(refl)
            fields[arm]["w_max"].append(float(state.w.max()))
            # The scheme's own heating rate (h_diabatic keeps it in both
            # arms, :6014), beside the radar tendency's 0.01 K/s cap
            # (radar_ref2tten.f90:210).
            fields[arm]["h_max"].append(float(state.h_diabatic.max()))
            fields[arm]["area_35"].append(float(
                (refl.max(axis=0) >= 35.0).sum() * cfg.dx * cfg.dy / 1.0e6))
            del state
        identical_starts &= bool(cp.array_equal(start_thp["off"],
                                                start_thp["on"]))
        print(f"member delt {member['delt_k']:.2f} K radius "
              f"{member['radius_m'] / 1000:.1f} km done "
              f"({time.time() - started:.0f} s)", flush=True)

    heated[-1] = False
    echo = cp.asarray(document["variables"]["z_mask"] != 0)
    clear = ~echo
    clear[-1] = False
    arms = {}
    for arm in ("off", "on"):
        thp = cp.stack(fields[arm]["thp"])
        refl = cp.stack(fields[arm]["refl"])
        arms[arm] = _arm_metrics(thp, refl, y, echo, clear, heated,
                                 args.echo_dbz)
        arms[arm]["member_max_w_ms"] = fields[arm]["w_max"]
        arms[arm]["member_area_35dbz_km2"] = fields[arm]["area_35"]
        arms[arm]["member_max_microphysics_heating_k_per_s"] =             fields[arm]["h_max"]
    off, on = arms["off"], arms["on"]
    ratios = {
        region: (on["theta_spread_k"][region]
                 / off["theta_spread_k"][region]
                 if off["theta_spread_k"][region] else None)
        for region in off["theta_spread_k"]}
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    receipt = {
        "schema": "gpuwm-da.radar-tten-spread.v1",
        "device": props["name"].decode(),
        "grid": {"nx": cfg.nx, "ny": cfg.ny, "nz": cfg.nz, "dx_m": cfg.dx,
                 "dt_s": cfg.dt, "mp_physics": cfg.mp_physics},
        "arrangement": "the research cycle's: one slot per leg from the "
                       "observation valid at the end of the leg, built "
                       "against each member's state at the start of the "
                       "leg; the same observation is then the one the "
                       "statistics below are taken against",
        "members": members,
        "spinup_minutes": args.spinup_min, "leg_minutes": args.leg_min,
        "perturbation": {"noise_k": args.noise_k,
                         "noise_top_m": args.noise_top_m},
        "truth_at_leg_end": truth_storm,
        "observation": {"echo_threshold_dbz": args.echo_dbz,
                        "echo_points": int(echo.sum()),
                        "clear_air_points": int(clear.sum()),
                        "coverage": "full"},
        "leg_start_identical_between_arms": identical_starts,
        "points_heated_by_the_forcing_in_any_member": int(heated.sum()),
        "forcing": receipts,
        "arms": arms,
        "theta_spread_on_over_off": ratios,
        "wall_seconds": round(time.time() - started, 1),
    }
    args.out.write_text(json.dumps(receipt, indent=1))
    print(json.dumps({"arms": arms, "theta_spread_on_over_off": ratios},
                     indent=1))


if __name__ == "__main__":
    main()
