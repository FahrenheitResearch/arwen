"""Price a fine nested free-forecast leg before running one.

VRAM comes from the repo's own estimator
(:func:`gpuwm.core.preflight.estimate_experiment` plus
:func:`gpuwm.core.preflight.machine_peak_envelope_bytes`), so the number
here is the same one ``gpuwm check --alloc`` would print -- COMPUTED, not
measured.  The DA driver hand-builds its ``ExperimentState`` and
therefore runs no VRAM gate of its own; this is the surface that fills
that gap for the nested route.

Wall time is priced as a RATIO against the parent, because the per-point
-step constant is a property of the card and the parent's own measured
seconds-per-leg is the only honest anchor.  Supply it with
``--parent-leg-seconds`` (the summed per-trajectory wall of one leg from
an existing cycle report) and the projection is arithmetic on top of a
measurement rather than a guess.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime

import numpy as np

try:                                     # python -m tools.da_nest_cost
    from tools.da_nowcast import NOWCAST_DEFAULT_PHYSICS_PROFILE
except ImportError:                      # python tools/da_nest_cost.py
    from da_nowcast import NOWCAST_DEFAULT_PHYSICS_PROFILE

MIB = 1024.0 * 1024.0


def build_experiment(*, nx, ny, nz, dx, dt, leg_seconds, profile):
    from gpuwm.config import RunConfig
    from gpuwm.experiment import (DomainConfig, ExperimentConfig,
                                  ProjectionConfig, VerticalConfig)
    from gpuwm.physics_compat import single_domain_runtime_switches

    switches = dict(single_domain_runtime_switches(profile))
    switches.pop("acknowledgements", None)
    base = dict(
        nx=nx, ny=ny, nz=nz, dx=float(dx), dy=float(dx), ztop=20000.0,
        dt=float(dt), run_seconds=float(leg_seconds),
        output_interval_s=float(leg_seconds),
        specified=True, nested=False, grid_id=1,
        spec_bdy_width=5, spec_zone=1, relax_zone=4,
        moist=True, hypsometric_opt=2, map_proj=1,
    )
    base.update({key: value for key, value in switches.items()
                 if key in RunConfig.__dataclass_fields__})
    run = RunConfig(**base)
    root = DomainConfig(
        grid_id=1, parent_id=0, i_parent_start=1, j_parent_start=1,
        parent_grid_ratio=1, parent_time_step_ratio=1,
        history_interval_s=float(leg_seconds), run=run, time_step=int(dt))
    return ExperimentConfig(
        name="nested-forecast-pricing",
        start_time=datetime(2024, 5, 21, 21),
        run_seconds=float(leg_seconds),
        vertical=VerticalConfig(tuple(np.linspace(1.0, 0.0, nz + 1)),
                                5000.0, 2, 0.2),
        projection=ProjectionConfig("lambert", 35.0, -97.0, 30.0, 60.0,
                                    -97.0),
        restart_interval_s=0.0, domains=(root,))


def price(exp, *, family="windows"):
    from gpuwm.core.preflight import (estimate_experiment,
                                      machine_peak_envelope_bytes,
                                      non_pool_device_bytes)

    estimate = estimate_experiment(exp)
    alloc = int(estimate.alloc_estimate_bytes)
    non_pool = int(non_pool_device_bytes(exp))
    envelope = machine_peak_envelope_bytes(
        alloc_estimate_bytes=alloc, non_pool_bytes=non_pool,
        domains=len(exp.domains), family=family)
    return {
        "domains": len(exp.domains),
        "alloc_estimate_mib": alloc / MIB,
        "non_pool_mib": non_pool / MIB,
        "machine_peak_envelope_mib": envelope / MIB,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--nx", type=int, default=132)
    parser.add_argument("--ny", type=int, default=132)
    parser.add_argument("--nz", type=int, default=49)
    parser.add_argument("--dx-m", type=float, default=3000.0)
    parser.add_argument("--dt-s", type=float, default=15.0)
    parser.add_argument("--leg-seconds", type=float, default=900.0)
    # The DEFAULT must be the profile the nowcast actually binds, not a
    # profile this tool remembers.  A VRAM gate that prices a suite the
    # route no longer runs is worse than no gate: it answers, and the
    # answer fits.  Imported from the same constant the four nowcast
    # entry points read so the six surfaces cannot drift apart again.
    parser.add_argument("--physics-profile",
                        default=NOWCAST_DEFAULT_PHYSICS_PROFILE,
                        help="shipped physics profile to price "
                             f"(default {NOWCAST_DEFAULT_PHYSICS_PROFILE}, "
                             "the nowcast's own default)")
    parser.add_argument("--ratio", type=int, default=3)
    parser.add_argument("--half-width-km", type=float, action="append",
                        default=None,
                        help="repeatable; default sweeps 30/45/60/90 km")
    parser.add_argument("--trajectories", type=int, default=11,
                        help="parent trajectories per leg (control + N)")
    parser.add_argument("--nest-members", type=int, default=0,
                        help="members carrying a nest, beside the control")
    parser.add_argument("--free-legs", type=int, default=6)
    parser.add_argument(
        "--parent-leg-seconds", type=float, default=None,
        help=("MEASURED summed per-trajectory wall seconds for one "
              "parent leg, from an existing cycle report. Without it the "
              "wall-time projection is omitted rather than invented"))
    parser.add_argument("--family", default="windows",
                        choices=("windows", "linux"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    from gpuwm.da import nested_forecast as nf

    parent = build_experiment(
        nx=args.nx, ny=args.ny, nz=args.nz, dx=args.dx_m, dt=args.dt_s,
        leg_seconds=args.leg_seconds, profile=args.physics_profile)
    parent_price = price(parent, family=args.family)

    widths = args.half_width_km or [30.0, 45.0, 60.0, 90.0]
    nest_trajectories = 1 + int(args.nest_members)
    rows = []
    for half_width in widths:
        geometry = nf.NestGeometry(ratio=args.ratio,
                                   half_width_km=float(half_width),
                                   members=int(args.nest_members))
        try:
            child = nf.nest_domain_config(parent, geometry)
        except nf.NestedForecastRefusal as refusal:
            rows.append({"half_width_km": half_width,
                         "refused": str(refusal)})
            continue
        nested = nf.nested_experiment(parent, child)
        nested_price = price(nested, family=args.family)
        cost = nf.nest_cost_model(nested, child)
        row = {
            "half_width_km": float(half_width),
            "child_nx": child.run.nx, "child_ny": child.run.ny,
            "child_dx_m": child.run.dx, "child_dt_s": child.run.dt,
            "covered_parent_fraction": cost["parent_fraction_covered"],
            "dycore_cost_vs_parent": cost["dycore_cost_vs_parent"],
            "vram": nested_price,
            "vram_delta_mib": (nested_price["machine_peak_envelope_mib"]
                               - parent_price["machine_peak_envelope_mib"]),
        }
        if args.parent_leg_seconds is not None:
            per_trajectory = (float(args.parent_leg_seconds)
                              / max(int(args.trajectories), 1))
            nest_leg = (per_trajectory * cost["dycore_cost_vs_parent"]
                        * nest_trajectories)
            row["projected"] = {
                "parent_seconds_per_trajectory_leg": per_trajectory,
                "nest_seconds_added_per_leg": nest_leg,
                "nested_leg_seconds": (float(args.parent_leg_seconds)
                                       + nest_leg),
                "free_forecast_seconds_added": nest_leg * args.free_legs,
            }
        rows.append(row)

    result = {
        "schema": "gpuwm-da.nested-forecast-cost.v1",
        "basis": "computed (gpuwm.core.preflight), not measured",
        "parent": {
            "nx": args.nx, "ny": args.ny, "nz": args.nz,
            "dx_m": args.dx_m, "dt_s": args.dt_s,
            "trajectories": args.trajectories,
            "vram": parent_price,
        },
        "nest_trajectories": nest_trajectories,
        "rows": rows,
    }
    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    print(f"parent {args.nx}x{args.ny}x{args.nz} dx={args.dx_m:g} m: "
          f"alloc {parent_price['alloc_estimate_mib']:.0f} MiB, "
          f"machine peak envelope "
          f"{parent_price['machine_peak_envelope_mib']:.0f} MiB "
          f"({args.family})")
    print(f"nest trajectories: {nest_trajectories} "
          f"(control + {args.nest_members} member(s)); parent carries "
          f"{args.trajectories}")
    print()
    header = (f"{'half-w':>7} {'child':>11} {'dx':>7} {'dt':>6} "
              f"{'covered':>8} {'cost/par':>9} {'peak MiB':>9} "
              f"{'+MiB':>7}")
    print(header)
    print("-" * len(header))
    for row in rows:
        if "refused" in row:
            print(f"{row['half_width_km']:>7.0f} REFUSED: {row['refused']}")
            continue
        print(f"{row['half_width_km']:>7.0f} "
              f"{row['child_nx']:>5}x{row['child_ny']:<5} "
              f"{row['child_dx_m']:>7.0f} {row['child_dt_s']:>6.2f} "
              f"{row['covered_parent_fraction']:>8.1%} "
              f"{row['dycore_cost_vs_parent']:>9.2f} "
              f"{row['vram']['machine_peak_envelope_mib']:>9.0f} "
              f"{row['vram_delta_mib']:>7.0f}")
    if args.parent_leg_seconds is not None:
        print()
        print(f"projected from a MEASURED parent leg of "
              f"{args.parent_leg_seconds:g} s "
              f"({args.trajectories} trajectories):")
        for row in rows:
            if "projected" not in row:
                continue
            p = row["projected"]
            print(f"  half-width {row['half_width_km']:>3.0f} km: "
                  f"+{p['nest_seconds_added_per_leg']:.1f} s/leg "
                  f"-> {p['nested_leg_seconds']:.1f} s/leg, "
                  f"+{p['free_forecast_seconds_added'] / 60.0:.1f} min "
                  f"over {args.free_legs} free legs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
