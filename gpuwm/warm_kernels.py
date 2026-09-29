"""`gpuwm warm-kernels`: pay the first run's kernel compile up front.

A fresh runtime's first forecast compiles its GPU kernels for the local
card before model step 1 can finish -- measured 2026-09-24 at about two
minutes and 157 kernel cache entries on an RTX 5070 Ti -- and the
screen says little while it happens.  CuPy keeps what it compiles in its
on-disk kernel cache (:func:`gpuwm.kernel_compile_notice.
cupy_kernel_cache_dir`), so the cost is paid once per card.  This door
pays it on purpose, at a time the user chooses, and reports what it did:
how many kernels were compiled into the cache and how long it took.

HOW THE SET IS CHOSEN.  The kernels a forecast compiles depend on its
physics, so the door does not keep a list of them.  It runs the forecast
code itself: for each physics profile it builds a small synthetic domain
(a flat, moist, resting column over land), attaches that profile's
physics through the same :func:`gpuwm.core.physics.initialize_physics`
a forecast calls, and advances it two model steps with the production
:func:`gpuwm.core.dycore.step`.  Whatever that compiles is what a
forecast with that physics compiles.  By default the profiles are the
ones the sources default to (:func:`gpuwm.physics_menu.
default_profile_for`), so a plain first forecast finds its kernels
cached; ``--profile`` names others and ``--all-profiles`` takes every
shipped suite.

What it does not reach: kernels only a real case uses (terrain, lateral
boundaries, output diagnostics) still compile on the first real run, and
a run at a vertical level count above a kernel's compiled ceiling adds
its own tier.  Those are seconds, not minutes.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import fields
from datetime import datetime

import numpy as np

WARM_KERNELS_SCHEMA = "gpuwm.warm-kernels.v1"

#: Vertical levels of the synthetic domain.  Below every kernel's
#: compiled level ceiling, so the modules compiled are the ones a
#: forecast at an ordinary level count loads.
DEFAULT_LEVELS = 40

#: Horizontal size and spacing of the synthetic domain.  Small enough to
#: cost nothing but compile time.
_NX = _NY = 16
_DX_M = 3000.0
_ZTOP_M = 16000.0
_DT_S = 12.0

#: Two steps: the first runs radiation and the surface schemes on their
#: first call, the second the path every later step takes.
_STEPS = 2

#: The synthetic column's valid time and location only have to put the
#: sun above the horizon so the shortwave kernels run.
_VALID_TIME = datetime(2020, 6, 21, 12)
_LATITUDE = 30.0
_LONGITUDE = 0.0


def default_profiles() -> tuple[str, ...]:
    """Every profile some source defaults to, in first-seen order."""

    from gpuwm import physics_menu

    seen: list[str] = []
    for source in physics_menu.registered_sources():
        profile = physics_menu.default_profile_for(source)
        if profile is not None and profile not in seen:
            seen.append(profile)
    return tuple(seen)


def _theta(z):
    """Potential temperature of a stably stratified troposphere (N = 0.01/s)."""

    return 300.0 * np.exp(1.0e-4 * np.asarray(z, dtype=np.float64) / 9.81)


def _vapour(z):
    """A moist boundary layer thinning with height, kg/kg."""

    return 0.012 * np.exp(-np.asarray(z, dtype=np.float64) / 2500.0)


def _run_profile(profile: str, levels: int) -> None:
    """Build the synthetic domain with ``profile``'s physics; step it."""

    import cupy as cp

    from gpuwm.config import RunConfig, validate_run_config
    from gpuwm.core.dycore import step
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.moist import init_moist_balanced
    from gpuwm.core.physics import initialize_physics
    from gpuwm.physics_compat import single_domain_runtime_switches

    names = {field.name for field in fields(RunConfig)}
    settings = {key: value for key, value
                in single_domain_runtime_switches(profile).items()
                if key in names}
    # The synthetic column is flat: the moist balanced builder takes flat
    # base states only.
    settings["terrain_opt"] = 0
    cfg = validate_run_config(RunConfig(
        nx=_NX, ny=_NY, nz=int(levels), dx=_DX_M, dy=_DX_M,
        ztop=_ZTOP_M, dt=_DT_S, run_seconds=_DT_S * _STEPS,
        clock_dt=_DT_S, output_interval_s=_DT_S * _STEPS,
        open_x=False, open_y=False, **settings))
    coord = make_vertical_coord(cfg.nz, hybrid_opt=cfg.hybrid_opt,
                                etac=cfg.etac)
    base = make_base_state(coord, _theta, p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_moist_balanced(cfg, coord, base, _vapour)
    grid = (cfg.ny, cfg.nx)
    initialize_physics(
        state, cfg, radiation_start_time=_VALID_TIME,
        radiation_latitude=np.full(grid, _LATITUDE),
        radiation_longitude=np.full(grid, _LONGITUDE),
        noahmp_start_time=_VALID_TIME,
        noahmp_latitude=np.full(grid, _LATITUDE),
        noahmp_longitude=np.full(grid, _LONGITUDE))
    for _ in range(_STEPS):
        step(state, cfg)
    cp.cuda.runtime.deviceSynchronize()


def warm_kernels(profiles=None, *, levels: int = DEFAULT_LEVELS,
                 say=None) -> dict:
    """Compile the forecast kernels of ``profiles`` into the kernel cache.

    Returns the report the door prints: per profile, the cache entries it
    wrote and its seconds, and the totals.  ``say`` receives one line per
    profile as it finishes.
    """

    from gpuwm import kernel_compile_notice as notice

    profiles = tuple(profiles) if profiles else default_profiles()
    capability = notice.current_compute_capability()
    if capability is None:
        raise RuntimeError(
            "no CUDA device answered: warm-kernels compiles for the local "
            "card, so it needs one (gpuwm doctor says why the card is "
            "unreachable)")
    import cupy as cp

    device = cp.cuda.runtime.getDeviceProperties(
        cp.cuda.Device().id)["name"]
    if isinstance(device, bytes):
        device = device.decode(errors="replace")
    cache_dir = notice.cupy_kernel_cache_dir()
    entries_before = notice.scan_kernel_cache(cache_dir)[0]
    rows = []
    started = time.perf_counter()
    for profile in profiles:
        before = notice.scan_kernel_cache(cache_dir)[0]
        began = time.perf_counter()
        _run_profile(profile, levels)
        row = {"profile": profile,
               "kernels_compiled": max(
                   0, notice.scan_kernel_cache(cache_dir)[0] - before),
               "seconds": round(time.perf_counter() - began, 1)}
        rows.append(row)
        if say is not None:
            say(f"warm-kernels: {profile}: {row['kernels_compiled']} "
                f"kernel(s) compiled in {row['seconds']:.1f} s")
    finished = notice.kernel_cache_state(
        cache_dir, compute_capability=capability)
    return {
        "schema": WARM_KERNELS_SCHEMA,
        "device": str(device),
        "compute_capability": str(capability),
        "cache_dir": str(cache_dir),
        "levels": int(levels),
        "profiles": rows,
        "kernels_compiled": sum(row["kernels_compiled"] for row in rows),
        "seconds": round(time.perf_counter() - started, 1),
        "cache_entries_before": int(entries_before),
        "cache_entries_for_this_card": int(finished.entries_for_capability),
    }


def warm_kernels_main(args) -> int:
    from gpuwm import physics_menu

    if args.all_profiles:
        profiles = physics_menu.shipped_profiles()
    else:
        profiles = tuple(args.profile or ()) or None
    say = None if args.json else (lambda text: print(text, flush=True))
    try:
        report = warm_kernels(profiles, levels=args.levels, say=say)
    except (RuntimeError, ValueError) as error:
        if args.json:
            print(json.dumps({"schema": WARM_KERNELS_SCHEMA,
                              "error": str(error)}))
        else:
            print(f"warm-kernels: {error}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report, allow_nan=False))
    else:
        print(f"warm-kernels: {report['kernels_compiled']} kernel(s) "
              f"compiled in {report['seconds']:.1f} s for "
              f"{report['device']} (sm_{report['compute_capability']}); "
              f"the cache at {report['cache_dir']} holds "
              f"{report['cache_entries_for_this_card']} for this card",
              flush=True)
    return 0


def register_cli(subparsers) -> None:
    from gpuwm.cli_numbers import int_at_least

    parser = subparsers.add_parser(
        "warm-kernels",
        help="compile the forecast GPU kernels for this card now, so the "
             "first forecast does not pay the compile")
    parser.add_argument(
        "--profile", action="append", default=None, metavar="PROFILE",
        help="a physics profile to compile for (repeatable); default: "
             "every profile a source defaults to")
    parser.add_argument(
        "--all-profiles", action="store_true",
        help="compile for every shipped physics profile")
    # Four is the vertical stencil width RunConfig refuses below; the
    # parser says so before a CUDA context is made for a domain that
    # cannot be built.
    parser.add_argument(
        "--levels", type=int_at_least(4), default=DEFAULT_LEVELS,
        help=f"vertical levels of the compile domain (default "
             f"{DEFAULT_LEVELS})")
    parser.add_argument(
        "--json", action="store_true",
        help="print the report as one JSON document")
    parser.set_defaults(func=warm_kernels_main)


__all__ = ["DEFAULT_LEVELS", "WARM_KERNELS_SCHEMA", "default_profiles",
           "register_cli", "warm_kernels", "warm_kernels_main"]
