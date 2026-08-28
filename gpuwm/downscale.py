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

from gpuwm import netcdf_bridge

from gpuwm.explain import warn
from gpuwm.offline_child import (
    DERIVED_CHILD_SURFACE_CAVEAT,
    OfflineChildContractError,
    OfflineChildPlacement,
    bind_parent_physics_from_gpuwm_restart,
    bind_parent_physics_from_wrf_namelist,
    child_surface_requirement,
    derive_child_surface_from_parent,
    read_child_surface_state,
    reserve_output_root,
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
    """Read the parent grid shape, spacing, latitude/longitude arrays.

    Through the Rust bridge, unlike :func:`_parent_attrs` above it: this
    one pulls the XLAT/XLONG FIELDS out of the tape, and decoding a
    meteorological field is decode work whoever wrote the file.  The
    attribute reader stays on netCDF4 because reading a global attribute
    off gpuwm's own output is identity plumbing, not decoding.
    """
    with netcdf_bridge.open_dataset(path) as dataset:
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


def _parent_mass_dims(path: Path) -> tuple[int, int]:
    """``(ny, nx)`` of the parent mass grid, dimensions only.

    Separate from :func:`_parent_geometry` on purpose: that one pulls
    XLAT/XLONG FIELDS out of the tape for ``--point`` placement, and the
    surface derivation below needs only the shape the placement is
    validated against.
    """
    with netCDF4.Dataset(path) as dataset:
        return (len(dataset.dimensions["south_north"]),
                len(dataset.dimensions["west_east"]))


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


#: What a derived child config is called inside the run directory it
#: describes.
DERIVED_CHILD_CONFIG_NAME = "child.toml"


def derived_child_config_path(outdir: Path, *, dry_run: bool) -> Path:
    """Where ``--point`` derivation writes the child config it just built.

    A real run gets it INSIDE ``--out``, beside the frames and the
    report, which is where a reader goes looking for the config a run
    used.  A dry run cannot: the run claims ``--out`` for itself so that
    no run ever adopts another's output, and a plan that filled it would
    leave the run that follows facing a directory holding a config it
    did not write.  The dry run therefore writes beside ``--out`` and
    says so.
    """

    if dry_run:
        return outdir.parent / (outdir.name + ".child.toml")
    return outdir / DERIVED_CHILD_CONFIG_NAME


def _render_child_toml(config: dict, *, tiles_mode: str | None = None) -> str:
    """Render one derived RunConfig as a legacy [grid]/[run] TOML.

    ``tiles_mode`` appends the ``[tiles]`` block ``--tiles`` asked for.  Only
    the mode is written: the tiling itself is :mod:`tilestream.autoplan`'s
    answer for the card in front of the run, and a derived config that pinned
    ``tile_nx``/``nbuffers`` would carry this machine's plan to the next one.
    """
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
    if tiles_mode is not None:
        lines += ["", "[tiles]", f"mode = {json.dumps(str(tiles_mode))}"]
    lines.append("")
    return "\n".join(lines)


def _fit_child_size(parent, parent_config, *, j0: int, i0: int, ratio: int,
                    run_seconds: float, output_interval_s: float,
                    vram_gib: float) -> int:
    """Largest centered square child whose peak envelope fits the card.

    Budget and criterion are the live sizing path's, the same arithmetic
    the domain wizard and ``gpuwm check`` price with: the free VRAM a
    card of this capacity really presents
    (:func:`gpuwm.domain_wizard.card_assumed_free_gib`) minus the
    external margin, against the AFFINE machine-peak envelope
    (:func:`gpuwm.core.preflight.estimate_experiment` ->
    ``peak_envelope_bytes``), stopping short of the budget by
    :func:`gpuwm.domain_wizard.fit_headroom_bytes`.

    This used to bind two RETIRED constants -- the flat
    ``vram_reserve_gib`` and the multiplicative
    ``observed_peak_envelope_bytes`` (the 1.75x WDDM floor that
    predicted 3.8x the measured peak on the calibration card) -- so it
    refused children the card holds (stale-guard audit 2026-08-25,
    finding 4).  MEASURED 2026-08-26 on the RTX 3080 10 GiB, real
    386x308 12 km GFS parent, ratio 3: the retired pair admitted
    282x282; this criterion admits 342x342, and the 342x342 child RAN
    WHOLE through this door -- 360 steps, 7,200 s simulated, PASS,
    machine-wide peak 9.24 of 10.24 GB.  Evidence:
    evidence/2026-08-25-stale-guards-engine/.
    """
    from datetime import datetime, timezone

    from gpuwm.core.preflight import (
        EXTERNAL_MARGIN_BYTES, GIB, estimate_experiment)
    from gpuwm.domain_wizard import card_assumed_free_gib, fit_headroom_bytes
    from gpuwm.experiment import experiment_from_run_config

    free_bytes = int(card_assumed_free_gib(float(vram_gib)) * GIB)
    budget = free_bytes - EXTERNAL_MARGIN_BYTES
    limit = budget - fit_headroom_bytes(budget)
    # The memory model is start-time independent; the wrapper needs A
    # datetime, and the child's real clock comes from the parent frames
    # at run time.
    estimate_epoch = datetime(2000, 1, 1, tzinfo=timezone.utc)
    # A size-INDEPENDENT invalidity in the parent's restart-evidence
    # config must surface as itself: every probe used to swallow it and
    # the search then reported "no child fits the budget" -- a VRAM
    # verdict for a validation refusal (walked live 2026-08-17, when a
    # 2.4.1 restart's recorded key was refused by a newer validation).
    last_config_error: ValueError | None = None
    any_size_fit = False

    def fits(size: int) -> bool:
        nonlocal last_config_error, any_size_fit
        try:
            _centered_placement(parent, j0=j0, i0=i0, ratio=ratio,
                                child_nx=size, child_ny=size)
            merged = _derive_child_run_config(
                parent_config, parent=parent, ratio=ratio,
                child_nx=size, child_ny=size, run_seconds=run_seconds,
                output_interval_s=output_interval_s)
        except OfflineChildContractError:
            return False
        except ValueError as error:
            last_config_error = error
            return False
        from gpuwm.config import RunConfig
        run = RunConfig(**merged)
        exp = experiment_from_run_config(run, estimate_epoch)
        estimate = estimate_experiment(exp, vram_gib=float(vram_gib))
        if estimate.peak_envelope_bytes <= limit:
            any_size_fit = True
            return True
        return False

    def cannot_plan() -> OfflineChildContractError:
        # The validation attribution is claimed only when NO probed size
        # ever fit: a search that saw a genuine fit and still failed is
        # a budget/geometry story, not a validation one.
        if last_config_error is not None and not any_size_fit:
            return OfflineChildContractError(
                "the derived child config does not validate at any size "
                "(this is not a VRAM question) -- the parent's restart-"
                "evidence physics is refused as a child config: "
                f"{last_config_error}")
        return OfflineChildContractError(
            f"no child fits the {vram_gib:g} GiB budget inside this parent")

    unit = 2 * ratio
    low, high = 0, 1
    while fits(unit * (high + 1)) and unit * high < 4096:
        high *= 2
    if high == 1 and not fits(unit * 2):
        raise cannot_plan()
    low = high // 2
    while low + 1 < high:
        mid = (low + high + 1) // 2
        if fits(unit * mid):
            low = mid
        else:
            high = mid
    size = unit * (high if fits(unit * high) else low)
    if size < 2 * unit:
        raise cannot_plan()
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


def _release_output_reservation(outdir: Path) -> None:
    """Give back the ``--out`` this command reserved, when it then refuses.

    THE POISONED RETRY: ``--out`` is created create-only before the
    contracts downstream of it are checked -- ``--point`` needs it to
    exist so the config it derives can live inside the run it describes.
    Every refusal raised after that (the parent that cannot seed the
    child's surface, a child-grid surface file that does not match, a
    child config the loader rejects) used to leave the directory behind
    holding ``child.toml``: a run directory describing a run that never
    happened, and the thing the corrected command then collided with.
    A refusal must leave the tree exactly as it found it.

    Only this command's own reservation is released, and only while it
    holds nothing but the config this command wrote into it.  Anything
    else means the directory is not ours to remove, and it is named and
    left alone rather than deleted.
    """

    import shutil

    try:
        held = sorted(child.name for child in outdir.iterdir())
    except OSError:
        return
    unexpected = [name for name in held if name != DERIVED_CHILD_CONFIG_NAME]
    if unexpected:
        warn(f"leaving {outdir} in place: it holds {', '.join(unexpected)}, "
             "which this refused command did not write",
             why="A refused downscale releases only the empty output "
                 "directory it reserved; anything else in there belongs "
                 "to something this command cannot account for.")
        return
    shutil.rmtree(outdir, ignore_errors=True)


class _OutputReservation:
    """The ``--out`` this command created, until something else owns it.

    Held open across the whole plan so that any refusal downstream of the
    reservation hands the directory back.  A directory that ALREADY
    existed (and was empty, so adopting it merges nothing) is never
    recorded here: releasing it would delete something this command did
    not create.
    """

    def __init__(self) -> None:
        self.path: Path | None = None

    def claim(self, outdir: Path) -> Path:
        existed = outdir.exists()
        resolved = reserve_output_root(outdir, flag="--out")
        if not existed:
            self.path = outdir
        return resolved

    def hand_off(self) -> None:
        """The run owns the directory now; its partial output is evidence."""
        self.path = None

    def release(self) -> None:
        if self.path is not None:
            _release_output_reservation(self.path)
            self.path = None


def downscale_main(args) -> int:
    """``gpuwm downscale``, with its output reservation held transactionally."""

    reservation = _OutputReservation()
    try:
        return _downscale_main(args, reservation)
    except BaseException:
        # Every exit that is not this command's own success releases the
        # directory it reserved -- refusals, Ctrl-C, and the failures the
        # CLI boundary turns into tracebacks alike.  A retry must meet the
        # tree the first attempt found.
        reservation.release()
        raise


def _downscale_main(args, reservation: _OutputReservation) -> int:
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

    outdir_reserved = False
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
        if args.tiles is not None:
            raise OfflineChildContractError(
                "--tiles writes a [tiles] block into a config this command "
                "DERIVES, and --child-config supplies its own.  Put "
                "[tiles] in that file instead; the child route reads it "
                "there and honors it.")
        child_config = Path(args.child_config)
        ratio = int(args.ratio)
        i_start, j_start = int(args.i_parent_start), int(args.j_parent_start)
        if not args.dry_run:
            # RESERVED AT THE FRONT DOOR, not inside the runner: this
            # route used to discover an --out collision after the CUDA
            # import and after the whole parent archive had been read,
            # which is as late as the discovery could possibly be made.
            reservation.claim(Path(args.out))
            outdir_reserved = True
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
        child_config = derived_child_config_path(
            outdir, dry_run=bool(args.dry_run))
        if child_config.parent == outdir:
            # The run's own output root, reserved HERE and by the same
            # never-adopt rule the runner applies, so the config that
            # describes the run lands INSIDE it instead of beside it.
            reservation.claim(outdir)
            outdir_reserved = True
        else:
            child_config.parent.mkdir(parents=True, exist_ok=True)
        child_config.write_text(
            _render_child_toml(merged, tiles_mode=args.tiles),
            encoding="utf-8", newline="\n")
        i_start, j_start = placement.i_parent_start, placement.j_parent_start
        print(f"gpuwm downscale: derived child {child_nx}x{child_ny} at "
              f"dx={merged['dx']:g} m (ratio {ratio}) centered on "
              f"({lat:g}, {lon:g}); parent start ({i_start}, {j_start}); "
              f"wrote {child_config}")
        if child_config.parent != outdir:
            print("gpuwm downscale: --dry-run does not reserve "
                  f"{outdir} (a run refuses if it already exists), so the "
                  "derived config is beside it; the real run writes it "
                  f"to {outdir / DERIVED_CHILD_CONFIG_NAME}")
    else:
        raise OfflineChildContractError(
            "pass --child-config TOML or --point LAT,LON")

    from gpuwm.config import load_config, soil_layer_count
    cfg = load_config(child_config)
    surface_requirement = child_surface_requirement(cfg)
    surface_source = None
    if args.child_surface_from is not None:
        surface = read_child_surface_state(
            args.child_surface_from, child_ny=cfg.ny, child_nx=cfg.nx,
            num_soil_layers=soil_layer_count(cfg))
        surface_source = "child-grid-file"
        print(f"gpuwm downscale: child surface source "
              f"{surface.path} ({len(surface.fields)} fields, "
              f"{surface.identity['MMINLU']})")
    elif surface_requirement is not None:
        # THE CLOSED LOOP, OPENED (defect #275).  This used to refuse
        # outright -- and the file it demanded could not be produced for
        # a config-driven (ERA5) parent by any command in the product,
        # so the whole route dead-ended after the parent forecast had
        # been paid for.  The parent's own history carries the nine
        # surface fields and the landuse identity; the child grid is an
        # exact refinement of a parent window, so WRF's own nest-birth
        # operators put them where the child needs them.  Derived HERE,
        # at the front door, before the run: a parent that cannot seed a
        # child still refuses at plan time, naming the missing fields.
        parent_ny, parent_nx = _parent_mass_dims(frames[0])
        try:
            surface = derive_child_surface_from_parent(
                frames[0], placement=OfflineChildPlacement(
                    parent_nx=parent_nx, parent_ny=parent_ny,
                    child_nx=int(cfg.nx), child_ny=int(cfg.ny),
                    parent_grid_ratio=int(ratio),
                    i_parent_start=int(i_start),
                    j_parent_start=int(j_start)),
                num_soil_layers=soil_layer_count(cfg))
        except OfflineChildContractError as error:
            # A parent that cannot seed a child is refused HERE, before
            # any preprocessing, carrying BOTH sentences: the config's
            # requirement (with its remedy) and the reason this parent
            # archive cannot meet it.  --dry-run still prints the plan --
            # deriving the geometry is how a reader learns which child
            # grid to build a surface file FOR -- and warns instead.
            unmet = (f"{surface_requirement}\n"
                     f"  and this parent archive cannot supply one "
                     f"either: {error}")
            if not args.dry_run:
                raise OfflineChildContractError(unmet) from error
            warn(unmet,
                 why="--dry-run continues so the derived plan below can "
                     "be read; the run itself will refuse until the "
                     "child surface state can be resolved.")
        else:
            surface_source = "parent-history-interpolated"
            print(f"gpuwm downscale: child surface derived from "
                  f"{surface.path} ({len(surface.fields)} fields, "
                  f"{surface.identity['MMINLU']})")
            # The caveat is the ACTION half, not the mechanism half: a
            # reader who never types --explain still has to be told that
            # this child's coastline is its parent's.
            warn("child surface state interpolated from the parent's own "
                 "history rather than built on the child grid -- "
                 + DERIVED_CHILD_SURFACE_CAVEAT,
                 why="This is WRF's input_from_file = .false. route for a "
                     "nest with no wrfinput of its own "
                     "(med_nest_initial's med_interp_domain), run through "
                     "the Registry's masked land interpolator.")

    lineage = parent_initial_condition(frames[0])
    if int(lineage.get("GPUWM_INITIAL_FORECAST_LEAD_HOURS", 0)):
        # One sentence, before the run, on the fact a published child
        # chart would otherwise lose: the parent's own initial state was
        # a forecast, so every child frame inherits that lead.
        warn(lineage["GPUWM_INITIAL_CONDITION_STATEMENT"],
             why="The child's initial and boundary conditions are the "
                 "parent's history, so the child is no closer to an "
                 "analysis than its parent was.")

    from gpuwm.config import load_streaming_options

    plan = {
        "parent_frames": [str(path) for path in frames],
        # Read off the config that will actually be run, whether this
        # command derived it or the caller supplied it, so --dry-run reports
        # the mode the child will use rather than the flag that was typed.
        "tiles": load_streaming_options(child_config).to_json(),
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
        "child_surface_required": surface_requirement is not None,
        # WHICH surface the child will actually start from, resolved
        # before the run rather than discovered inside it: the walked
        # 2.4.1 plan said only "child_surface_from": null and left the
        # reader to find out at integration time what that meant.
        "child_surface_source": surface_source,
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
        outdir=Path(args.out),
        # This process created --out moments ago to hold the config it
        # derived; the never-adopt reservation already happened there.
        outdir_reserved=outdir_reserved)
    # From here the directory belongs to the run: a forecast that dies
    # mid-integration leaves frames a reader needs, and no report.json to
    # claim it finished.  Deleting that would be destroying evidence.
    reservation.hand_off()
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
    parser.add_argument("--i-parent-start", type=int, default=None,
                        help="1-based west-east parent index of the "
                             "child's southwest corner (required with "
                             "--child-config; --point derives it)")
    parser.add_argument("--j-parent-start", type=int, default=None,
                        help="1-based south-north parent index of the "
                             "child's southwest corner (required with "
                             "--child-config; --point derives it)")
    parser.add_argument("--child-size", default=None, metavar="NX[,NY]",
                        help="explicit child extent for --point")
    # THE DERIVED CONFIG'S [tiles] BLOCK, and only for --point: with
    # --child-config the block belongs in the caller's own file, which the
    # child route reads and honors.  A refined child is the domain most
    # likely to outgrow the card it is run on -- --card sizes it to fit
    # RESIDENT, and this is how a caller asks for the larger child instead.
    parser.add_argument("--tiles", choices=("on", "auto"), default=None,
                        help="write [tiles] mode into the config --point "
                             "derives, so the child integrates out of a "
                             "pinned host store instead of resident "
                             "('on' always, 'auto' when tilestream.autoplan "
                             "says it does not fit)")
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
                        default="cuda",
                        help="where the parent-to-child interpolation "
                             "runs (default cuda; cpu reproduces it "
                             "off-GPU for verification)")
    parser.add_argument("--health-interval-seconds", type=float,
                        default=60.0,
                        help="model seconds between child health lines "
                             "(CFL, w_max, NaN check; default 60)")
    parser.add_argument("--out", type=Path, required=True,
                        help="create-only output directory for the child "
                             "run (report.json, wrfout frames, restart)")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate contracts, derive/print the plan, "
                             "write the derived TOML, run nothing")
    parser.set_defaults(func=downscale_main)


__all__ = ["downscale_main", "register_cli"]
