"""``gpuwm downscale``: offline parent-history -> standalone CUDA child.

Front door for the native ``ndown`` replacement (:mod:`gpuwm.offline_child`
contracts, :mod:`gpuwm.offline_child_run` driver).  Two child modes:

* ``--child-config TOML`` plus explicit placement (``--ratio``,
  ``--i-parent-start``, ``--j-parent-start``): the config is a legacy
  ``[grid]``/``[dynamics]``/``[run]`` RunConfig TOML with
  ``specified=true``, ``nested=false``.
* ``--point LAT,LON``: the child is derived -- geometry from the parent
  projection (nearest parent mass point, centered footprint, dx and dt
  divided by ``--ratio``), physics inherited verbatim from the parent's
  gpuwm restart evidence, and the extent either given (``--child-size``)
  or fitted to a VRAM budget with the itemized preflight estimator
  (``--card``/``--vram-gib``, the domain wizard's budget convention).
  The derived config is written beside ``--out`` as a reusable TOML.

Boundary cadence defaults to the parent archive's own history cadence
(announced with a one-line warning; ``--max-boundary-interval-seconds``
bounds it explicitly, ``--accept-parent-cadence`` silences the warning).
Parents intended for downscaling should write history at 15-minute (or
denser) cadence.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import netCDF4
import numpy as np

from gpuwm.explain import warn
from gpuwm.offline_child import (
    OfflineChildContractError,
    OfflineChildPlacement,
    bind_parent_physics_from_gpuwm_restart,
    bind_parent_physics_from_wrf_namelist,
    read_child_surface_state,
    validate_parent_history,
)

_FRAME_RE = re.compile(
    r"wrfout_d(?P<dom>\d{2})_\d{4}-\d{2}-\d{2}[_:]\d{2}[_:]\d{2}[_:]\d{2}$")

#: Parent-cadence guidance threshold (seconds).  Coarser boundary forcing
#: is accepted only through the explicit cadence flags, and always with
#: the printed caveat: hourly boundaries cannot reproduce the sub-hourly
#: forcing a live nest receives every parent step.
CADENCE_GUIDANCE_SECONDS = 900.0

#: Keys the derived child config overrides on top of the parent's
#: restart-evidence RunConfig.  Everything else (physics selections,
#: diffusion/damping, acoustic settings) is inherited verbatim, which is
#: receipted -- deriving new physics silently would be fake.
_GEOMETRY_KEYS = (
    "nx", "ny", "dx", "dy", "dt", "grid_id", "specified", "nested",
    "run_seconds", "output_interval_s", "clock_dt", "case",
)


def _card_vram_gib() -> dict:
    """The wizard's card-tier table, imported where it is used.

    One table, two front doors: ``gpuwm domain`` and ``gpuwm downscale``
    have to accept the same tier names or the product documents two
    different answers to "what cards does this support".
    """

    from gpuwm.domain_wizard import CARD_VRAM_GIB

    return CARD_VRAM_GIB


def _discover_parent_series(
        raw_paths: list[Path], parent_domain: int | None) -> list[Path]:
    """Resolve a directory-or-files argument into one ordered frame list."""
    if len(raw_paths) == 1 and raw_paths[0].is_dir():
        candidates = [path for path in sorted(raw_paths[0].iterdir())
                      if _FRAME_RE.match(path.name)]
        if not candidates:
            raise OfflineChildContractError(
                f"{raw_paths[0]} contains no wrfout history frames")
        domains = sorted({_FRAME_RE.match(p.name).group("dom")
                          for p in candidates})
        if parent_domain is not None:
            token = f"{int(parent_domain):02d}"
            if token not in domains:
                raise OfflineChildContractError(
                    f"{raw_paths[0]} has no domain-{token} frames "
                    f"(present: {domains})")
            selected = token
        elif len(domains) == 1:
            selected = domains[0]
        else:
            raise OfflineChildContractError(
                f"{raw_paths[0]} carries multiple domains {domains}; "
                "pass --parent-domain to choose the parent")
        return [p for p in candidates
                if _FRAME_RE.match(p.name).group("dom") == selected]
    files = []
    for path in raw_paths:
        if not path.is_file():
            raise OfflineChildContractError(
                f"parent history argument is not a file: {path}")
        files.append(path)
    # Frame ordering comes from the WRF filename timestamp when present;
    # validate_parent_history re-proves strict time ordering either way.
    return sorted(files, key=lambda p: p.name)


def _parse_point(raw: str) -> tuple[float, float]:
    parts = raw.split(",")
    if len(parts) != 2:
        raise ValueError(f"--point must be LAT,LON, got {raw!r}")
    try:
        lat, lon = float(parts[0]), float(parts[1])
    except ValueError:
        raise ValueError(
            f"--point must be LAT,LON in decimal degrees, got {raw!r}"
        ) from None
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 360.0):
        raise ValueError(f"--point {raw!r} is outside geographic bounds")
    return lat, lon


def parent_initial_condition(path: Path) -> dict[str, object]:
    """The lineage attributes a parent archive publishes, if any.

    ``gpuwm downscale`` is one of the two consumers that read a wrfout
    back, and until now it could not tell a parent initialized from a
    cycle's analysis from one initialized from that cycle's 240 h
    forecast.  The child inherits its parent's initial condition, so the
    child inherits the disclosure with it.  An empty mapping means the
    parent said nothing, which is UNKNOWN and never "analysis".
    """

    from gpuwm.io.wrfout import INITIAL_CONDITION_GLOBAL_ATTRS

    with netCDF4.Dataset(path) as dataset:
        present = set(dataset.ncattrs())
        return {
            name: _jsonable_attr(dataset.getncattr(name))
            for name in INITIAL_CONDITION_GLOBAL_ATTRS if name in present
        }


def _jsonable_attr(value):
    return value.item() if isinstance(value, np.generic) else value


def _parent_geometry(path: Path) -> dict[str, object]:
    """Read the parent grid shape, spacing, latitude/longitude arrays."""
    with netCDF4.Dataset(path) as dataset:
        result = {
            "ny": len(dataset.dimensions["south_north"]),
            "nx": len(dataset.dimensions["west_east"]),
            "nz": len(dataset.dimensions["bottom_top"]),
            "dx": float(dataset.getncattr("DX")),
            "dy": float(dataset.getncattr("DY")),
        }
        for name in ("XLAT", "XLONG"):
            if name not in dataset.variables:
                raise OfflineChildContractError(
                    f"{path} lacks {name}; --point placement needs the "
                    "parent latitude/longitude fields")
            value = np.asarray(dataset.variables[name][:], dtype=np.float64)
            if value.ndim == 3:
                value = value[0]
            result[name.lower()] = value
    return result


def _nearest_parent_index(lat_field, lon_field, lat: float,
                          lon: float) -> tuple[int, int]:
    """Nearest parent mass point, projection-agnostic (0-based j, i)."""
    scale = np.cos(np.deg2rad(lat))
    cost = ((lat_field - lat) ** 2
            + (scale * (lon_field - lon)) ** 2)
    j, i = np.unravel_index(int(np.argmin(cost)), cost.shape)
    return int(j), int(i)


def _centered_placement(parent, *, j0: int, i0: int, ratio: int,
                        child_nx: int, child_ny: int) -> OfflineChildPlacement:
    """Center a ``child_nx x child_ny`` footprint on parent point (j0,i0)."""
    if child_nx % ratio or child_ny % ratio:
        raise OfflineChildContractError(
            f"child extent {child_nx}x{child_ny} must be a multiple of the "
            f"refinement ratio {ratio}")
    span_i = child_nx // ratio
    span_j = child_ny // ratio
    # Round half up (not banker's): a half-cell-ambiguous center resolves
    # deterministically toward the higher parent index.
    i_start = int(math.floor(i0 + 1 - (span_i - 1) / 2.0 + 0.5))
    j_start = int(math.floor(j0 + 1 - (span_j - 1) / 2.0 + 0.5))
    return OfflineChildPlacement(
        parent_nx=int(parent["nx"]), parent_ny=int(parent["ny"]),
        child_nx=int(child_nx), child_ny=int(child_ny),
        parent_grid_ratio=int(ratio),
        i_parent_start=i_start, j_parent_start=j_start)


def _derive_child_run_config(parent_config: dict, *, parent, ratio: int,
                             child_nx: int, child_ny: int,
                             run_seconds: float,
                             output_interval_s: float) -> dict:
    """Child RunConfig dict: parent physics verbatim, geometry rescaled."""
    from dataclasses import fields as dataclass_fields

    from gpuwm.config import RunConfig, validate_run_config

    known = {field.name for field in dataclass_fields(RunConfig)}
    merged = {key: value for key, value in parent_config.items()
              if key in known}
    merged.update({
        "nx": int(child_nx), "ny": int(child_ny),
        "dx": float(parent["dx"]) / ratio,
        "dy": float(parent["dy"]) / ratio,
        "dt": float(parent_config["dt"]) / ratio,
        "grid_id": int(parent_config.get("grid_id", 1)) + 1,
        "specified": True, "nested": False,
        "run_seconds": float(run_seconds),
        "output_interval_s": float(output_interval_s),
        "clock_dt": 0.0, "case": "",
    })
    validate_run_config(RunConfig(**merged))
    return merged


def _render_child_toml(config: dict) -> str:
    """Render one derived RunConfig as a legacy [grid]/[run] TOML."""
    def value(item):
        if isinstance(item, bool):
            return "true" if item else "false"
        if isinstance(item, str):
            return json.dumps(item)
        if isinstance(item, float) and math.isfinite(item):
            return repr(item)
        if isinstance(item, int):
            return str(item)
        raise ValueError(f"cannot render config value {item!r}")

    grid_keys = ("nx", "ny", "nz", "dx", "dy", "ztop")
    lines = ["# derived by `gpuwm downscale --point`: parent physics",
             "# inherited verbatim from restart evidence, geometry",
             "# rescaled by the refinement ratio.", "", "[grid]"]
    for key in grid_keys:
        if key in config:
            lines.append(f"{key} = {value(config[key])}")
    lines += ["", "[run]"]
    for key in sorted(config):
        if key in grid_keys:
            continue
        lines.append(f"{key} = {value(config[key])}")
    lines.append("")
    return "\n".join(lines)


def _fit_child_size(parent, parent_config, *, j0: int, i0: int, ratio: int,
                    run_seconds: float, output_interval_s: float,
                    vram_gib: float) -> int:
    """Largest centered square child whose itemized estimate fits VRAM.

    Budget convention follows the domain wizard: card capacity minus the
    flat near-capacity reserve.  The fit criterion applies the allocator
    headroom and the observed peak envelope to the child's itemized
    resident+transient subtotal -- a conservative standalone-domain
    reading of the experiment path's ``footprint x envelope <= budget``
    gate, with the platform-conditional envelope factor
    (:func:`gpuwm.core.preflight.peak_envelope_factor`).
    """
    from gpuwm.core.preflight import (
        ALLOCATOR_HEADROOM, GIB, estimate_domain,
        observed_peak_envelope_bytes)
    from gpuwm.domain_wizard import vram_reserve_gib
    from gpuwm.experiment import DomainConfig

    budget = (float(vram_gib) - vram_reserve_gib(float(vram_gib))) * GIB

    def fits(size: int) -> bool:
        try:
            _centered_placement(parent, j0=j0, i0=i0, ratio=ratio,
                                child_nx=size, child_ny=size)
            merged = _derive_child_run_config(
                parent_config, parent=parent, ratio=ratio,
                child_nx=size, child_ny=size, run_seconds=run_seconds,
                output_interval_s=output_interval_s)
        except (OfflineChildContractError, ValueError):
            return False
        from gpuwm.config import RunConfig
        run = RunConfig(**merged)
        dc = DomainConfig(
            grid_id=run.grid_id, parent_id=0, i_parent_start=1,
            j_parent_start=1, parent_grid_ratio=1,
            parent_time_step_ratio=1,
            history_interval_s=float(output_interval_s), run=run)
        estimate = estimate_domain(dc, n_lbc_intervals=1)
        subtotal = estimate.resident_bytes + estimate.transient_bytes
        peak = observed_peak_envelope_bytes(
            math.ceil(ALLOCATOR_HEADROOM * subtotal))
        return peak <= budget

    unit = 2 * ratio
    low, high = 0, 1
    while fits(unit * (high + 1)) and unit * high < 4096:
        high *= 2
    if high == 1 and not fits(unit * 2):
        raise OfflineChildContractError(
            f"no child fits the {vram_gib:g} GiB budget inside this parent")
    low = high // 2
    while low + 1 < high:
        mid = (low + high + 1) // 2
        if fits(unit * mid):
            low = mid
        else:
            high = mid
    size = unit * (high if fits(unit * high) else low)
    if size < 2 * unit:
        raise OfflineChildContractError(
            f"no child fits the {vram_gib:g} GiB budget inside this parent")
    return size


def _parent_cadence_seconds(frames: list[Path]) -> float:
    from gpuwm.offline_child import inspect_parent_history_frame

    if len(frames) < 2:
        raise OfflineChildContractError(
            "offline downscaling requires at least two parent frames")
    first = inspect_parent_history_frame(frames[0])
    second = inspect_parent_history_frame(frames[1])
    seconds = (second.valid_time - first.valid_time).total_seconds()
    if seconds <= 0.0:
        raise OfflineChildContractError(
            "parent history times must be strictly increasing")
    return float(seconds)


def downscale_main(args) -> int:
    frames = _discover_parent_series(
        [Path(p) for p in args.parent], args.parent_domain)
    cadence = _parent_cadence_seconds(frames)
    # Provenance: True whenever the ceiling is the archive's own
    # cadence (by flag or by default), False for an explicit bound.
    cadence_is_parents = args.max_boundary_interval_seconds is None
    if args.max_boundary_interval_seconds is not None:
        max_interval = float(args.max_boundary_interval_seconds)
    else:
        # The archive's own cadence is the default: it is already known,
        # it is the only cadence these frames can serve, and the coarse-
        # cadence caveat below still prints.  --accept-parent-cadence is
        # kept as an accepted no-op for existing scripts.
        max_interval = cadence
        if not args.accept_parent_cadence:
            warn(f"using the parent archive's own {cadence:g} s history "
                 "cadence as the boundary cadence; pass "
                 "--max-boundary-interval-seconds SECONDS to bound it",
                 why="Boundary cadence is a scientific choice tied to "
                     "child resolution; write parent history at 15-min "
                     "or denser cadence when planning to downscale.")
    if cadence > CADENCE_GUIDANCE_SECONDS:
        warn(f"parent cadence {cadence:g} s is coarser than the "
             f"{CADENCE_GUIDANCE_SECONDS:g} s guidance for downscaling",
             why="The child sees interval-linear boundary forcing where "
                 "a live nest is forced every parent step -- expect "
                 "boundary-swept differences to grow with the cadence "
                 "gap.  Regenerate the parent archive at 15-min (or "
                 "denser) history cadence for production downscaling.")

    if args.parent_restart is not None:
        binding = bind_parent_physics_from_gpuwm_restart(args.parent_restart)
    elif args.parent_namelist is not None:
        binding = bind_parent_physics_from_wrf_namelist(
            args.parent_namelist, domain_id=args.parent_namelist_domain)
    else:
        raise OfflineChildContractError(
            "parent physics must be bound from companion evidence: pass "
            "--parent-restart (gpuwm) or --parent-namelist (stock WRF)")

    contract = validate_parent_history(
        frames, max_boundary_interval_seconds=max_interval,
        physics_binding=binding)
    window_seconds = (
        contract.end_time - contract.start_time).total_seconds()

    if args.child_config is not None:
        if args.point is not None:
            raise OfflineChildContractError(
                "pass --child-config or --point, not both")
        for name in ("ratio", "i_parent_start", "j_parent_start"):
            if getattr(args, name) is None:
                raise OfflineChildContractError(
                    f"--child-config placement requires --{name.replace('_', '-')}")
        if args.hours is not None or args.output_interval_seconds is not None:
            warn("--hours/--output-interval-seconds are ignored with "
                 "--child-config; the TOML's run_seconds and "
                 "output_interval_s are used")
        child_config = Path(args.child_config)
        ratio = int(args.ratio)
        i_start, j_start = int(args.i_parent_start), int(args.j_parent_start)
    elif args.point is not None:
        if args.parent_restart is None:
            raise OfflineChildContractError(
                "--point derivation inherits the child physics from the "
                "parent's gpuwm restart evidence; stock-WRF parents need "
                "an explicit --child-config")
        from gpuwm.io.restart import read_restart_header
        parent_config = dict(read_restart_header(
            Path(args.parent_restart))["config"])
        parent = _parent_geometry(frames[0])
        lat, lon = _parse_point(args.point)
        j0, i0 = _nearest_parent_index(
            parent["xlat"], parent["xlong"], lat, lon)
        ratio = int(args.ratio if args.ratio is not None else 3)
        run_seconds = (float(args.hours) * 3600.0
                       if args.hours is not None else window_seconds)
        output_interval_s = (float(args.output_interval_seconds)
                             if args.output_interval_seconds is not None
                             else contract.interval_seconds)
        if args.child_size is not None:
            parts = [int(p) for p in str(args.child_size).split(",")]
            child_nx = parts[0]
            child_ny = parts[1] if len(parts) > 1 else parts[0]
        else:
            from gpuwm.domain_wizard import CARD_VRAM_GIB
            if args.vram_gib is not None:
                vram_gib = float(args.vram_gib)
            else:
                vram_gib = CARD_VRAM_GIB[args.card or "24gb"]
            child_nx = child_ny = _fit_child_size(
                parent, parent_config, j0=j0, i0=i0, ratio=ratio,
                run_seconds=run_seconds,
                output_interval_s=output_interval_s, vram_gib=vram_gib)
        placement = _centered_placement(
            parent, j0=j0, i0=i0, ratio=ratio,
            child_nx=child_nx, child_ny=child_ny)
        merged = _derive_child_run_config(
            parent_config, parent=parent, ratio=ratio,
            child_nx=child_nx, child_ny=child_ny,
            run_seconds=run_seconds, output_interval_s=output_interval_s)
        outdir = Path(args.out)
        child_config = outdir.parent / (outdir.name + ".child.toml")
        child_config.parent.mkdir(parents=True, exist_ok=True)
        child_config.write_text(_render_child_toml(merged),
                                encoding="utf-8", newline="\n")
        i_start, j_start = placement.i_parent_start, placement.j_parent_start
        print(f"gpuwm downscale: derived child {child_nx}x{child_ny} at "
              f"dx={merged['dx']:g} m (ratio {ratio}) centered on "
              f"({lat:g}, {lon:g}); parent start ({i_start}, {j_start}); "
              f"wrote {child_config}")
    else:
        raise OfflineChildContractError(
            "pass --child-config TOML or --point LAT,LON")

    if args.child_surface_from is not None:
        from gpuwm.config import load_config, soil_layer_count
        cfg = load_config(child_config)
        surface = read_child_surface_state(
            args.child_surface_from, child_ny=cfg.ny, child_nx=cfg.nx,
            num_soil_layers=soil_layer_count(cfg))
        print(f"gpuwm downscale: child surface source "
              f"{surface.path} ({len(surface.fields)} fields, "
              f"{surface.identity['MMINLU']})")

    lineage = parent_initial_condition(frames[0])
    if int(lineage.get("GPUWM_INITIAL_FORECAST_LEAD_HOURS", 0)):
        # One sentence, before the run, on the fact a published child
        # chart would otherwise lose: the parent's own initial state was
        # a forecast, so every child frame inherits that lead.
        warn(lineage["GPUWM_INITIAL_CONDITION_STATEMENT"],
             why="The child's initial and boundary conditions are the "
                 "parent's history, so the child is no closer to an "
                 "analysis than its parent was.")

    plan = {
        "parent_frames": [str(path) for path in frames],
        "initial_condition": lineage,
        "cadence_seconds": contract.interval_seconds,
        "max_boundary_interval_seconds": max_interval,
        "accepted_parent_cadence": bool(cadence_is_parents),
        "physics_binding": dict(binding.receipt()),
        "child_config": str(child_config),
        "placement": {"ratio": ratio, "i_parent_start": i_start,
                      "j_parent_start": j_start},
        "child_surface_from": (None if args.child_surface_from is None
                               else str(args.child_surface_from)),
        "outdir": str(args.out),
    }
    if args.dry_run:
        print(json.dumps({"event": "downscale_plan", **plan}, indent=2,
                         sort_keys=True))
        return 0

    from gpuwm import offline_child_run
    namespace = argparse.Namespace(
        parent_history=[Path(p) for p in frames],
        parent_restart=(None if args.parent_restart is None
                        else Path(args.parent_restart)),
        parent_namelist=(None if args.parent_namelist is None
                         else Path(args.parent_namelist)),
        parent_domain_id=int(args.parent_namelist_domain),
        child_config=Path(child_config),
        parent_grid_ratio=int(ratio),
        i_parent_start=int(i_start), j_parent_start=int(j_start),
        max_boundary_interval_seconds=float(max_interval),
        accepted_parent_cadence=bool(cadence_is_parents),
        child_surface_from=(None if args.child_surface_from is None
                            else Path(args.child_surface_from)),
        preprocess_backend=args.preprocess_backend,
        health_interval_seconds=float(args.health_interval_seconds),
        outdir=Path(args.out))
    report = offline_child_run.run(namespace)
    return 0 if report["result"] == "PASS" else 1


def register_cli(subparsers) -> None:
    parser = subparsers.add_parser(
        "downscale",
        help="run a standalone CUDA child from archived parent history "
             "(native ndown replacement; gpuwm and stock-WRF parents)")
    parser.add_argument(
        "parent", nargs="+",
        help="parent wrfout directory or explicit history files")
    parser.add_argument("--parent-domain", type=int, default=None,
                        help="parent domain id when the directory carries "
                             "several (e.g. 3 for the innermost archived "
                             "parent)")
    evidence = parser.add_mutually_exclusive_group()
    evidence.add_argument("--parent-restart", type=Path, default=None,
                          help="gpuwm restart of the parent run "
                               "(authoritative physics evidence)")
    evidence.add_argument("--parent-namelist", type=Path, default=None,
                          help="stock-WRF namelist.input of the parent run")
    parser.add_argument("--parent-namelist-domain", type=int, default=1,
                        help="domain column of --parent-namelist (default 1)")
    parser.add_argument("--child-config", type=Path, default=None,
                        help="legacy RunConfig TOML for the child "
                             "(specified=true, nested=false)")
    parser.add_argument("--point", default=None, metavar="LAT,LON",
                        help="derive the child around this point instead "
                             "of --child-config (gpuwm parents only)")
    parser.add_argument("--ratio", type=int, default=None,
                        help="refinement ratio (child-config placement: "
                             "required; --point default 3)")
    parser.add_argument("--i-parent-start", type=int, default=None)
    parser.add_argument("--j-parent-start", type=int, default=None)
    parser.add_argument("--child-size", default=None, metavar="NX[,NY]",
                        help="explicit child extent for --point")
    # THE tier list, not a copy of it.  This tuple used to be written
    # out by hand as ("16gb", "24gb", "32gb") while `gpuwm domain` took
    # its choices from CARD_VRAM_GIB -- so `--card 12gb`, a tier the
    # product advertises and this module already maps three lines down
    # in --point sizing, was an argparse rejection here and nowhere
    # else.  Two commands quoting different tier lists for the same
    # concept is a documentation bug you cannot fix in the docs.
    parser.add_argument("--card", choices=sorted(_card_vram_gib()),
                        default=None,
                        help="VRAM tier for --point sizing (default 24gb; "
                             "the same tiers `gpuwm domain` accepts)")
    parser.add_argument("--vram-gib", type=float, default=None,
                        help="explicit VRAM capacity for --point sizing")
    parser.add_argument("--hours", type=float, default=None,
                        help="--point run window in hours (default: the "
                             "full parent archive window)")
    parser.add_argument("--output-interval-seconds", type=float,
                        default=None,
                        help="--point child history cadence (default: the "
                             "parent cadence)")
    cadence = parser.add_mutually_exclusive_group()
    cadence.add_argument("--max-boundary-interval-seconds", type=float,
                         default=None,
                         help="explicit ceiling on acceptable parent "
                              "cadence (the scientific cadence contract); "
                              "mutually exclusive with "
                              "--accept-parent-cadence")
    cadence.add_argument("--accept-parent-cadence", action="store_true",
                         help="accept the archive's own cadence as the "
                              "ceiling (prints the 15-min guidance when "
                              "coarser); mutually exclusive with "
                              "--max-boundary-interval-seconds")
    parser.add_argument("--child-surface-from", type=Path, default=None,
                        help="child-grid wrfinput/history file with land "
                             "identity + soil warm start (required for "
                             "surface-physics children)")
    parser.add_argument("--preprocess-backend", choices=("cuda", "cpu"),
                        default="cuda")
    parser.add_argument("--health-interval-seconds", type=float,
                        default=60.0)
    parser.add_argument("--out", type=Path, required=True,
                        help="create-only output directory for the child "
                             "run (report.json, wrfout frames, restart)")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate contracts, derive/print the plan, "
                             "write the derived TOML, run nothing")
    parser.set_defaults(func=downscale_main)


__all__ = ["downscale_main", "register_cli"]
