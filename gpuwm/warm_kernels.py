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
import os
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
    """Every profile some source defaults to, in first-seen order.

    At every spacing: a source's own default, then each grid-spacing row
    it admits (:data:`gpuwm.physics_menu.SPACING_DEFAULTS`), so a plain
    sub-km first forecast finds its kernels cached as a coarser one does.
    """

    from gpuwm import physics_menu

    seen: list[str] = []
    for source in physics_menu.registered_sources():
        profiles = [physics_menu.default_profile_for(source)]
        profiles += [row["profile_id"] for row in
                     physics_menu.spacing_default_rows(source) if row["admitted"]]
        for profile in profiles:
            if profile is not None and profile not in seen:
                seen.append(profile)
    return tuple(seen)


def _theta(z):
    """Potential temperature of a stably stratified troposphere (N = 0.01/s)."""

    return 300.0 * np.exp(1.0e-4 * np.asarray(z, dtype=np.float64) / 9.81)


def _vapour(z):
    """A moist boundary layer thinning with height, kg/kg."""

    return 0.012 * np.exp(-np.asarray(z, dtype=np.float64) / 2500.0)


def synthetic_downward_longwave(cfg) -> tuple[float | None, str]:
    """The downward longwave the synthetic column is handed, and why.

    A profile with a longwave scheme writes GLW itself and is handed
    nothing.  A profile with none, whose land surface reads GLW or whose
    shortwave publishes it, is the configuration
    :func:`gpuwm.core.physics.initialize_physics` refuses to invent a
    number for.  A real run of such a profile declares the constant with
    :data:`~gpuwm.physics_compat.CONSTANT_DOWNWARD_LONGWAVE_ACK`, and the
    runtime then types :data:`~gpuwm.physics_compat.DECLARED_CONSTANT_GLW_WM2`
    into that call (:func:`gpuwm.runtime.declared_constant_glw`).  The
    column declares that same constant, so it compiles the kernels such a
    run compiles.  The refusal a real run meets without the
    acknowledgement is untouched: it lives at config load and in
    ``initialize_physics``, and this door hands that call a declaration.

    Returns ``(glw, sentence)``: ``glw`` is ``None`` where nothing is
    declared, and the sentence is what the report says about it.
    """

    from gpuwm.config import radiation_scheme_ids
    from gpuwm.physics_compat import (CONSTANT_DOWNWARD_LONGWAVE_ACK,
                                      DECLARED_CONSTANT_GLW_WM2,
                                      downward_longwave_disposition)

    longwave, shortwave = radiation_scheme_ids(cfg)
    kind, _consumer = downward_longwave_disposition(
        ra_lw_physics=longwave, ra_sw_physics=shortwave,
        sf_surface_physics=int(cfg.sf_surface_physics))
    if kind in ("consumed", "published"):
        return (float(DECLARED_CONSTANT_GLW_WM2),
                f"declared constant {DECLARED_CONSTANT_GLW_WM2:g} W m-2, as a "
                f"run of this suite declares it ({CONSTANT_DOWNWARD_LONGWAVE_ACK})")
    if kind == "scheme":
        return None, f"computed by ra_lw_physics={longwave}"
    return None, "not read by this suite"


def _run_profile(profile: str, levels: int) -> str:
    """Build the synthetic domain with ``profile``'s physics; step it.

    Returns the sentence :func:`synthetic_downward_longwave` gave for the
    downward longwave the column ran on.
    """

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
    glw, longwave = synthetic_downward_longwave(cfg)
    initialize_physics(
        state, cfg, radiation_start_time=_VALID_TIME,
        radiation_latitude=np.full(grid, _LATITUDE),
        radiation_longitude=np.full(grid, _LONGITUDE),
        noahmp_start_time=_VALID_TIME,
        noahmp_latitude=np.full(grid, _LATITUDE),
        noahmp_longitude=np.full(grid, _LONGITUDE),
        **({} if glw is None else {"glw": glw}))
    for _ in range(_STEPS):
        step(state, cfg)
    cp.cuda.runtime.deviceSynchronize()
    return longwave


#: What a profile that reads an unstaged asset is told.  The physics tables
#: published outside the package are staged by ``gpuwm fetch-tables``; any
#: other absent asset is named by ``gpuwm doctor`` with its own command.
MISSING_ASSET_REMEDY = ("gpuwm fetch-tables stages the physics tables this "
                        "suite reads; gpuwm doctor names any other missing "
                        "asset and the command that installs it")


def warm_kernels(profiles=None, *, levels: int = DEFAULT_LEVELS,
                 say=None) -> dict:
    """Compile the forecast kernels of ``profiles`` into the kernel cache.

    Returns the report the door prints: per profile, the cache entries it
    wrote, its seconds and where its downward longwave came from, and the
    totals.  ``say`` receives one line per profile as it finishes.

    A profile whose configuration is refused (a ``ValueError`` from the
    forecast code it runs) is recorded under ``failed`` with the refusal,
    and the others still run: one refused suite used to stop the whole
    ``--all-profiles`` pass at the first such profile, leaving every
    later one cold.  A profile that reads a table asset this install has
    not staged (a ``FileNotFoundError``, such as a Thompson table before
    ``gpuwm fetch-tables``) is recorded the same way, with
    :data:`MISSING_ASSET_REMEDY`, because the suites that do not read it
    can still warm.  Anything else, such as no card or a CUDA error,
    still stops the pass, because a device fault is not a property of
    one profile and the ones after it would meet it too.
    """

    from gpuwm import kernel_compile_notice as notice
    from gpuwm import nvrtc_ptx_cache as direct_cache

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
    direct_before = _direct_cache_inventory(cache_dir)
    direct_event_start = len(direct_cache.cache_events())
    rows = []
    failed = []
    started = time.perf_counter()
    for profile in profiles:
        before = notice.scan_kernel_cache(cache_dir)[0]
        direct_profile_start = len(direct_cache.cache_events())
        began = time.perf_counter()
        try:
            longwave = _run_profile(profile, levels)
        except ValueError as refusal:
            failed.append({"profile": profile, "error": str(refusal)})
            if say is not None:
                say(f"warm-kernels: {profile}: FAILED: {refusal}")
            continue
        except FileNotFoundError as missing:
            failed.append({"profile": profile, "error": str(missing),
                           "remedy": MISSING_ASSET_REMEDY})
            if say is not None:
                say(f"warm-kernels: {profile}: FAILED: {missing}. Next: "
                    f"{MISSING_ASSET_REMEDY}")
            continue
        row = {"profile": profile,
               "kernels_compiled": max(
                   0, notice.scan_kernel_cache(cache_dir)[0] - before),
               "seconds": round(time.perf_counter() - began, 1),
               "downward_longwave": longwave}
        direct_events = direct_cache.cache_events()[direct_profile_start:]
        row.update(_direct_cache_counts(direct_events))
        rows.append(row)
        if say is not None:
            say(f"warm-kernels: {profile}: {row['kernels_compiled']} "
                f"kernel(s) compiled in {row['seconds']:.1f} s; "
                f"downward longwave {longwave}")
    finished = notice.kernel_cache_state(
        cache_dir, compute_capability=capability)
    direct_after = _direct_cache_inventory(cache_dir)
    direct_events = direct_cache.cache_events()[direct_event_start:]
    return {
        "schema": WARM_KERNELS_SCHEMA,
        "device": str(device),
        "compute_capability": str(capability),
        "cache_dir": str(cache_dir),
        "levels": int(levels),
        "profiles": rows,
        "failed": failed,
        "kernels_compiled": sum(row["kernels_compiled"] for row in rows),
        "seconds": round(time.perf_counter() - started, 1),
        "cache_entries_before": int(entries_before),
        "cache_entries_for_this_card": int(finished.entries_for_capability),
        "direct_nvrtc_cache": {
            "directory": str(cache_dir / "gpuwm-direct-nvrtc"),
            "before": direct_before,
            "after": direct_after,
            "new_image_files": max(0, direct_after["image_files"]
                                    - direct_before["image_files"]),
            **_direct_cache_counts(direct_events),
            "events": direct_events,
        },
    }


def _direct_cache_inventory(cache_dir) -> dict:
    """Count the direct compiler's image files separately from CuPy entries."""
    count = size = 0
    for root, _directories, names in os.walk(cache_dir / "gpuwm-direct-nvrtc"):
        for name in names:
            if not name.endswith(".image"):
                continue
            try:
                size += os.stat(os.path.join(root, name)).st_size
                count += 1
            except OSError:
                pass
    return {"image_files": count, "bytes": size}


def _direct_cache_counts(events) -> dict:
    return {
        "direct_nvrtc_images_compiled": sum(
            row["status"] != "hit" for row in events),
        "direct_nvrtc_cache_hits": sum(row["status"] == "hit" for row in events),
        "direct_nvrtc_compile_seconds": sum(
            row["compile_seconds"] for row in events),
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
        direct = report["direct_nvrtc_cache"]
        if direct["events"]:
            print("warm-kernels: direct NVRTC cache: "
                  f"{direct['direct_nvrtc_images_compiled']} image(s) compiled, "
                  f"{direct['direct_nvrtc_cache_hits']} cache hit(s); "
                  f"{direct['after']['image_files']} image file(s) in "
                  f"{direct['directory']}", flush=True)
        if report["failed"]:
            print(f"warm-kernels: {len(report['failed'])} profile(s) "
                  "failed: " + ", ".join(row["profile"]
                                         for row in report["failed"]),
                  file=sys.stderr)
    return 1 if report["failed"] else 0


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


__all__ = ["DEFAULT_LEVELS", "MISSING_ASSET_REMEDY", "WARM_KERNELS_SCHEMA",
           "default_profiles",
           "register_cli", "synthetic_downward_longwave", "warm_kernels",
           "warm_kernels_main"]
