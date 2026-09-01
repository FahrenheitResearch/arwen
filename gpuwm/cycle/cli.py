"""``gpuwm cycle`` -- drive the coupled DA / parent / child cycle.

The front door is deliberately thin.  Everything it can refuse, it
refuses in ``--dry-run`` BEFORE a byte of state is written: the clock
invariants, the child tick ratios, the parent kind, the placement
provider's presence.  A cycling run that dies on its own arithmetic at
hour three has already cost the night.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from gpuwm.cycle.clock import CycleClock
from gpuwm.cycle.contracts import (CycleRefusal, PARENT_KINDS,
                                   seconds_to_ticks, ticks_to_seconds)

PLACEMENT_PROVIDERS = ("tracker", "schedule", "obs", "none")

#: Placement providers live in gpuwm.cycle.placement, which is built by a
#: separate producer.  The spine imports it LAZILY so a tree without it
#: still plans, dry-runs and runs forecast-only cycles.
_PLACEMENT_MODULE = "gpuwm.cycle.placement"

#: The engine adapter for a model-backed parent kind.  Named here so the
#: refusal for an unwired kind can say WHICH symbol it imported and from
#: WHERE, instead of "no adapter" -- which tells the lane that owns the
#: adapter nothing about what to land.
_ENGINE_MODULE = "gpuwm.cycle.engine"
_PARENT_BACKEND_SYMBOL = "build_model_parent_engine"

#: ``gpuwm-obs.radar-grid.v1`` ships ``z_obs``, ``z_max`` and ``z_mean``
#: side by side and states in its own contract that the choice between
#: them is the CONSUMER's and that neither is recoverable from the other.
#: So the field is CONFIGURED, not fixed and not sniffed: every canonical
#: file carries all three, which makes "discover it from the file" a
#: hardcode with extra steps.  The default is the file's own observation
#: variable -- what every DA lane reads and what the real three-volume
#: KTLX series was placed against.
DEFAULT_OBS_FIELD = "z_obs"

#: The parent-model plane the tracker provider looks at.  Configured for
#: the same reason: it is a name in somebody else's file.
DEFAULT_TRACKER_FIELD = "composite_reflectivity"


def register_cli(subparsers) -> None:
    parser = subparsers.add_parser(
        "cycle",
        help="run the coupled DA/forecast cycle: one clock, an anchor per "
             "boundary, an append-only ledger")
    parser.add_argument("--root", type=Path, required=True,
                        help="cycle root; the ledger, anchors and per-cycle "
                             "receipts all live here")
    parser.add_argument("--epoch-anchor", required=True, metavar="ISO8601",
                        help="the parent init's config_start_time (UTC); "
                             "the ONLY datetime in the cycling spine, every "
                             "other time is an integer tick from it")
    parser.add_argument("--parent-dt-seconds", type=float, default=120.0,
                        help="parent model step; must be a whole number of "
                             "milliseconds (default 120.0)")
    parser.add_argument("--cycle-seconds", type=float, required=True,
                        help="model seconds between cycle boundaries; must "
                             "be a whole number of parent steps")
    parser.add_argument("--cycles", type=int, required=True,
                        help="how many cycles to run")
    parser.add_argument("--parent-kind", required=True, choices=PARENT_KINDS,
                        help="which engine advances the parent")
    parser.add_argument("--child-slots", type=int, default=0,
                        help="identically shaped dormant nests reserved at "
                             "t=0 (default 0). The RESERVATION is fixed and "
                             "the PLACEMENT is arbitrary: that is what keeps "
                             "VRAM deterministic while a child can be "
                             "anywhere")
    parser.add_argument("--child-dt-seconds", type=float, default=None,
                        help="child model step; must divide the parent step "
                             "exactly (default: the parent step)")
    parser.add_argument("--placement-provider", default="none",
                        choices=PLACEMENT_PROVIDERS,
                        help="where each cycle's child placements come from "
                             "(default none: parent-only cycling)")
    parser.add_argument("--max-forecast-only-cycles", type=int, default=3,
                        help="consecutive cycles allowed with no analysis "
                             "before the run halts STALE_ANALYSIS_BUDGET_"
                             "EXHAUSTED (default 3): forecast-only is "
                             "legitimate, forecast-only forever is a run "
                             "that stopped being a DA cycle")
    parser.add_argument("--allow-placement-clamp", action="store_true",
                        help="accept a placement clamped into the parent "
                             "instead of refusing it (default off; a clamp "
                             "is always receipted either way)")
    parser.add_argument("--accept-snap-offset-seconds", type=float,
                        default=0.0,
                        help="largest analysis-time offset from the parent-"
                             "step lattice this run will accept by name "
                             "(default 0.0: the time must land on a step)")
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume", dest="resume", action="store_true",
                        default=True,
                        help="continue after the last completed cycle in the "
                             "ledger (default)")
    resume.add_argument("--no-resume", dest="resume", action="store_false",
                        help="start again at cycle 1")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the boundary lattice and the resolved "
                             "child ratios, refuse invalid combinations, and "
                             "write nothing")

    state = parser.add_argument_group(
        "parent state",
        "what the parent actually cycles.  Without these the door writes a "
        "receipt for a leg it never ran")
    state.add_argument("--parent-state", default=None, metavar="GLOB",
                       help="the parent's state series: a glob of npz "
                            "frames in the SAME on-disk shape "
                            "tools/cycle_mpas_leg.py --history reads (flat "
                            "npz: prognostic fields, time_seconds, and the "
                            "derived diagnostics beside them). Required for "
                            "a real run; --dry-run does not need it")
    state.add_argument("--parent-mesh-id", default=None, metavar="ID",
                       help="the mesh identity written into every anchor. "
                            "No default: an identity the spine guessed is "
                            "an identity no downstream reader can trust")
    state.add_argument("--analysis-increment", action="append", default=[],
                       metavar="CYCLE=PATH",
                       help="the analysis increment applied at CYCLE, as an "
                            "npz keyed by PROGNOSTIC FIELD NAME. Repeatable. "
                            "Use CYCLE=null for an explicit NULL ARM (a zero "
                            "increment that must be bit-stable through the "
                            "anchor). The three-hash ingestion gate needs "
                            "both arms to mean anything")

    port = parser.add_argument_group(
        "model parent (--parent-kind mpas-cuda / mpas-cuda-frames)",
        "how the door reaches the port's device stack.  Every leg runs in "
        "a FRESH PROCESS and talks to this one only through anchors and "
        "segments on disk: the port pins its own gpuwm checkout and "
        "refuses an in-process spine import, so out-of-process is the "
        "only route that works on a real card")
    port.add_argument("--port-root", default=None, metavar="PATH",
                      help="the MPAS port checkout the forecast worker "
                           "runs from. Required for a model parent kind")
    port.add_argument("--port-config", default=None, metavar="PATH",
                      help="the port's case configuration JSON. Required "
                           "for a model parent kind")
    port.add_argument("--port-steps", type=int, default=None,
                      help="dycore steps per cycle boundary. Required for "
                           "a model parent kind; the step RECEIPTS the "
                           "worker returns are counted against this "
                           "number, and a leg that ran fewer steps than "
                           "asked cannot earn the mpas-cuda stamp")
    port.add_argument("--port-timeout", type=float, default=None,
                      help="seconds to wait for one forecast segment "
                           "(default: no timeout)")

    place = parser.add_argument_group(
        "placement",
        "where each cycle's children go.  A geographic request resolved "
        "against the parent handed in, never a parent index")
    place.add_argument("--parent-geo-file", default=None, metavar="PATH",
                       help="XLAT/XLONG for the parent's mass grid: a "
                            "radar-grid NetCDF or an npz carrying both. "
                            "Defaults to the first --placement-obs-file, "
                            "whose grid IS the target model grid by "
                            "contract")
    place.add_argument("--parent-dx-m", type=float, default=None,
                       help="the parent's grid spacing in metres. Required "
                            "with a placement provider: the child/parent "
                            "refinement ratio is derived from it and a "
                            "guessed spacing silently changes every "
                            "placement")
    place.add_argument("--placement-obs-file", action="append", default=[],
                       metavar="PATH",
                       help="radar-grid observation file the obs provider "
                            "places on. Repeatable, ONE PER CYCLE in order; "
                            "a single file is reused for every cycle. A "
                            "storm that never moves between cycles is the "
                            "defect that hid child retirement for a week")
    place.add_argument("--placement-obs-field", default=DEFAULT_OBS_FIELD,
                       metavar="NAME",
                       help=f"which observation plane the obs provider "
                            f"places on (default {DEFAULT_OBS_FIELD}). The "
                            f"radar-grid contract ships z_obs, z_max and "
                            f"z_mean side by side and calls the choice the "
                            f"consumer's; an absent name is refused naming "
                            f"what the file does carry")
    place.add_argument("--placement-tracker-field",
                       default=DEFAULT_TRACKER_FIELD, metavar="NAME",
                       help=f"which PARENT plane the tracker provider "
                            f"places on (default {DEFAULT_TRACKER_FIELD})")
    place.add_argument("--placement-threshold", type=float, default=40.0,
                       help="trigger value a peak must reach to earn a "
                            "child, in the placement field's own units "
                            "(default 40.0)")
    place.add_argument("--retire-below-strength", type=float, default=None,
                       help="a child with less than this much signal under "
                            "it is retired and its reservation returns to "
                            "the pool. Required with a placement provider "
                            "and deliberately has NO default: its units are "
                            "the trigger field's, so a default would be a "
                            "hardcoded threshold for somebody else's field")
    place.add_argument("--min-separation-km", type=float, default=40.0,
                       help="two children are never planted on one storm "
                            "(default 40.0). A request inside this radius "
                            "of an assigned child is REFUSED by name, never "
                            "silently dropped")
    place.add_argument("--child-nx", type=int, default=199,
                       help="child grid points west-east (default 199)")
    place.add_argument("--child-ny", type=int, default=199,
                       help="child grid points south-north (default 199)")
    place.add_argument("--child-dx-m", type=float, default=1000.0,
                       help="child grid spacing in metres (default 1000.0); "
                            "must divide --parent-dx-m exactly")
    parser.set_defaults(func=cycle_main)


def cycle_main(args) -> int:
    clock = _build_clock(args)
    ratio = None
    if args.child_dt_seconds is not None:
        ratio = clock.child_ratio(args.child_dt_seconds)
    elif args.child_slots:
        ratio = clock.child_ratio(args.parent_dt_seconds)
    if args.child_slots < 0:
        raise CycleRefusal("child slots cannot be negative",
                           child_slots=int(args.child_slots))
    if args.max_forecast_only_cycles < 0:
        raise CycleRefusal("the forecast-only budget cannot be negative",
                           max_forecast_only_cycles=int(
                               args.max_forecast_only_cycles))
    provider = _resolve_placement_provider(args)

    if args.dry_run:
        _print_plan(args, clock, ratio, provider)
        return 0

    # The engine is resolved BEFORE the ledger is opened, so an unwired
    # parent kind or a missing state series leaves the root clean.
    advance_parent = _engine_for(args, clock)

    _print_plan(args, clock, ratio, provider)
    from gpuwm.cycle.ledger import CycleLedger
    from gpuwm.cycle.supervisor import CycleSupervisor

    root = Path(args.root)
    supervisor = CycleSupervisor(
        clock=clock, ledger=CycleLedger(root), root=root,
        advance_parent=advance_parent,
        analyse=_analysis_for(args),
        plan_children=provider,
        advance_children=_child_advance_for(clock, args),
        max_forecast_only_cycles=args.max_forecast_only_cycles,
        allow_placement_clamp=args.allow_placement_clamp)
    result = supervisor.run(resume=args.resume)
    print(f"cycle: completed {result['cycles_completed']} "
          f"(resumed from {result['resumed_from_cycle']})")
    print(f"cycle: ledger {root / 'cycle_ledger.jsonl'}")
    return 0


def _build_clock(args) -> CycleClock:
    return CycleClock.build(epoch_anchor=_parse_epoch(args.epoch_anchor),
                            parent_dt_seconds=args.parent_dt_seconds,
                            cycle_seconds=args.cycle_seconds,
                            n_cycles=args.cycles)


def _parse_epoch(text: str) -> datetime:
    raw = str(text).strip().replace("Z", "+00:00")
    try:
        when = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise CycleRefusal(
            "the epoch anchor is not an ISO 8601 time; write it like "
            "2026-08-14T18:00:00Z",
            epoch_anchor=str(text), error=str(exc)) from exc
    return when if when.tzinfo else when.replace(tzinfo=timezone.utc)


def _latlon_from(path: Path) -> tuple:
    """XLAT/XLONG off a radar-grid NetCDF or an npz, refusing by name.

    The radar-grid file's grid IS the target model grid -- the contract
    says "exactly" -- so a run that places on observations already holds
    its parent's geography and should not be made to restate it.
    """
    import numpy as np

    path = Path(path)
    if path.suffix.lower() in (".nc", ".nc4", ".netcdf"):
        from gpuwm.obs.radar_grid import read_radar_grid

        variables = read_radar_grid(path)["variables"]
    elif path.suffix.lower() == ".npz":
        with np.load(path) as handle:
            variables = {name: handle[name] for name in handle.files}
    else:
        raise CycleRefusal(
            "parent geometry must come from a radar-grid NetCDF or an npz",
            parent_geo_file=str(path), suffix=path.suffix)
    missing = [name for name in ("XLAT", "XLONG") if name not in variables]
    if missing:
        raise CycleRefusal(
            "parent geometry file does not carry the parent's lat/lon",
            parent_geo_file=str(path), missing=missing,
            available=sorted(str(name) for name in variables),
            remedy="point --parent-geo-file at a file carrying XLAT and "
                   "XLONG on the parent's mass grid")
    return variables["XLAT"], variables["XLONG"]


def _parent_geometry(args):
    """The parent a geographic request is resolved against.

    This is the value ``_resolve_placement_provider`` never passed, so
    every placement through the CLI died inside the provider on "placement
    needs the parent's geometry" -- a capability that existed, was tested,
    and could not be reached.
    """
    from gpuwm.cycle.placement import parent_geometry_from_fields

    source = args.parent_geo_file or (args.placement_obs_file[0]
                                      if args.placement_obs_file else None)
    if source is None:
        raise CycleRefusal(
            "placement needs the parent's geography and this run named no "
            "source for it",
            placement_provider=args.placement_provider,
            remedy="pass --parent-geo-file PATH (XLAT/XLONG on the parent's "
                   "mass grid), or --placement-obs-file PATH, whose "
                   "radar-grid IS the target model grid by contract")
    if args.parent_dx_m is None:
        raise CycleRefusal(
            "placement needs the parent's grid spacing",
            placement_provider=args.placement_provider,
            parent_geo_file=str(source),
            remedy="pass --parent-dx-m METRES; the child/parent refinement "
                   "ratio is derived from it and no file on this path "
                   "declares it")
    lat, lon = _latlon_from(Path(source))
    return parent_geometry_from_fields(lat_field=lat, lon_field=lon,
                                       dx_m=float(args.parent_dx_m))


def _tracker_signal_for(args):
    """The parent plane the tracker places on, read off the live anchor.

    Named rather than hardwired: ``composite_reflectivity`` is a key in
    somebody else's file, and a run whose parent calls it something else
    should get a refusal that lists what the anchor does carry, not an
    empty placement list that reads as "no storms".
    """
    import numpy as np

    field = str(args.placement_tracker_field)

    def signal_for(cycle_index, parent_record):
        arrays = parent_record.get("derived_arrays") or {}
        if field not in arrays:
            raise CycleRefusal(
                "the parent's anchor does not carry the tracker's placement "
                "field",
                cycle_index=int(cycle_index),
                placement_tracker_field=field,
                available=sorted(str(name) for name in arrays),
                remedy="pass --placement-tracker-field NAME naming a plane "
                       "this parent actually publishes")
        return np.asarray(arrays[field], dtype=np.float64)

    return signal_for


def _resolve_placement_provider(args):
    """Import the placement provider lazily, or say who owns it."""
    if args.placement_provider == "none":
        return None
    import importlib
    try:
        module = importlib.import_module(_PLACEMENT_MODULE)
    except ImportError as exc:
        raise CycleRefusal(
            "placement providers are supplied by gpuwm.cycle.placement "
            "(build order 3); not present in this tree. Run with "
            "--placement-provider none for parent-only cycling",
            placement_provider=args.placement_provider,
            module=_PLACEMENT_MODULE, error=str(exc)) from exc
    build = getattr(module, "build_placement_provider", None)
    if build is None:
        raise CycleRefusal(
            "gpuwm.cycle.placement is present but does not offer "
            "build_placement_provider(kind=..., ...)",
            placement_provider=args.placement_provider,
            module=_PLACEMENT_MODULE,
            available=sorted(name for name in dir(module)
                             if not name.startswith("_")))
    if args.retire_below_strength is None:
        raise CycleRefusal(
            "a placement provider needs a retirement threshold",
            placement_provider=args.placement_provider,
            remedy="pass --retire-below-strength VALUE in the placement "
                   "field's own units; there is no safe default because a "
                   "default would be a threshold for somebody else's field")
    if args.placement_provider == "obs" and not args.placement_obs_file:
        raise CycleRefusal(
            "the obs placement provider needs a radar-grid file",
            placement_provider="obs",
            remedy="pass --placement-obs-file PATH, once per cycle in order "
                   "(a single file is reused for every cycle)")
    geometry = _parent_geometry(args)
    return build(kind=args.placement_provider,
                 child_slots=args.child_slots,
                 allow_clamp=args.allow_placement_clamp,
                 parent_geometry=geometry,
                 signal_for=(_tracker_signal_for(args)
                             if args.placement_provider == "tracker"
                             else None),
                 radar_grid_paths=[Path(item)
                                   for item in args.placement_obs_file],
                 field=(args.placement_obs_field
                        if args.placement_provider == "obs"
                        else args.placement_tracker_field),
                 threshold=float(args.placement_threshold),
                 min_separation_km=float(args.min_separation_km),
                 retire_below_strength=float(args.retire_below_strength),
                 child_nx=int(args.child_nx), child_ny=int(args.child_ny),
                 child_dx_m=float(args.child_dx_m),
                 child_dt_seconds=float(args.child_dt_seconds
                                        if args.child_dt_seconds is not None
                                        else args.parent_dt_seconds),
                 parent_grid_ratio=max(1, int(round(
                     float(args.parent_dx_m) / float(args.child_dx_m)))))


def _load_frames(pattern: str) -> list[dict]:
    """The parent's state series, split into prognostic and derived.

    The on-disk shape is the one ``tools/cycle_mpas_leg.py --history``
    already reads and writes -- a FLAT npz with the prognostic fields,
    ``time_seconds`` and the derived diagnostics all beside each other.
    The split is by :data:`REQUIRED_PROGNOSTIC_FIELDS`, which is the
    anchor writer's own contract, so a frame this loads is a frame
    ``write_anchor`` will accept.  Reading the other producer's real
    format rather than inventing a third one is the whole point.
    """
    import glob as globlib

    import numpy as np

    from gpuwm.cycle.anchor import REQUIRED_PROGNOSTIC_FIELDS

    paths = sorted(Path(item) for item in globlib.glob(str(pattern)))
    if not paths:
        raise CycleRefusal(
            "the parent state series matched no frames",
            parent_state=str(pattern),
            remedy="point --parent-state at the recorded npz frames, e.g. "
                   "'state/*.npz'")
    frames = []
    for path in paths:
        with np.load(path) as handle:
            fields = {name: np.asarray(handle[name], dtype=np.float64)
                      for name in handle.files}
        missing = [name for name in REQUIRED_PROGNOSTIC_FIELDS
                   if name not in fields]
        if missing:
            raise CycleRefusal(
                "parent state frame does not carry every prognostic field",
                frame=str(path), missing=missing,
                required=list(REQUIRED_PROGNOSTIC_FIELDS),
                available=sorted(fields))
        prognostic = {name: fields[name]
                      for name in (*REQUIRED_PROGNOSTIC_FIELDS,
                                   "time_seconds") if name in fields}
        derived = {name: value for name, value in fields.items()
                   if name not in prognostic}
        frames.append({"prognostic": prognostic, "derived": derived})
    return frames


def _increment_source(args):
    """``increment_for``: the analysis this run applies at each cycle.

    ``CYCLE=null`` is the NULL ARM and is deliberately spellable.  A gate
    with only the applied arm proves nothing -- a hash that moved could
    have been moved by rehydration -- so the door has to be able to ask
    for a zero increment that must stay bit-stable through the anchor.
    """
    import numpy as np

    if not args.analysis_increment:
        return None

    plan: dict[int, str | None] = {}
    for entry in args.analysis_increment:
        text = str(entry)
        if "=" not in text:
            raise CycleRefusal(
                "an analysis increment is written CYCLE=PATH",
                analysis_increment=text,
                remedy="pass --analysis-increment 2=increment.npz, or "
                       "--analysis-increment 1=null for the null arm")
        index, _, where = text.partition("=")
        try:
            cycle_index = int(index)
        except ValueError as exc:
            raise CycleRefusal(
                "an analysis increment's cycle must be an integer",
                analysis_increment=text, cycle=index) from exc
        plan[cycle_index] = None if where.strip().lower() == "null" else where

    def increment_for(cycle_index: int, prognostic):
        if int(cycle_index) not in plan:
            return None
        where = plan[int(cycle_index)]
        if where is None:
            # NULL ARM: shaped like the state, exactly zero.
            return {"rho_u": np.zeros_like(
                np.asarray(prognostic["rho_u"], dtype=np.float64))}
        with np.load(where) as handle:
            arrays = {name: np.asarray(handle[name], dtype=np.float64)
                      for name in handle.files}
        usable = {name: value for name, value in arrays.items()
                  if name in prognostic}
        if not usable:
            # The exact class of defect this lane was sent to kill: a
            # reader whose assumed key names are only ever satisfied by a
            # fixture.  Say what was in the file AND what would have been
            # accepted, rather than applying nothing and reporting green.
            raise CycleRefusal(
                "the analysis increment names no prognostic field",
                cycle_index=int(cycle_index), increment_file=str(where),
                observed_keys=sorted(arrays),
                prognostic_fields=sorted(prognostic),
                remedy="the increment npz is keyed by PROGNOSTIC FIELD "
                       "NAME on the parent's own grid; a wind-space DA "
                       "product must be mapped into prognostic space "
                       "before it reaches the cycle")
        for name, value in usable.items():
            want = np.shape(prognostic[name])
            if np.shape(value) != want:
                raise CycleRefusal(
                    "the analysis increment is not shaped like the state",
                    cycle_index=int(cycle_index), field=name,
                    increment_shape=tuple(np.shape(value)),
                    prognostic_shape=tuple(want),
                    increment_file=str(where))
        return usable

    return increment_for


def _engine_for(args, clock):
    """The parent advance callable for ``--parent-kind``.

    This used to hand the supervisor a closure that returned two integers.
    The consequence was that ``gpuwm cycle`` cycled a clock over an EMPTY
    root: no anchor was written, so the three-hash ingestion gate had
    nothing to compare, the hydrostatic consistency instrument never fired,
    and every receipt carried ``"ingestion": null`` -- while
    :func:`gpuwm.cycle.engine.build_replay_parent_engine`, which does all
    three, sat in the tree reachable only from ``tools/cycle_demo_*.py``.
    The door now runs the same engine the demos did.

    The model-backed kinds are wired by the engine adapter.  Until that
    adapter is present this refuses BEFORE the run starts rather than
    pretending to integrate an atmosphere, and it names the module and
    symbol it looked for so the lane that owns the adapter knows exactly
    what to land.
    """
    from gpuwm.cycle.engine import build_replay_parent_engine

    # The adapter check comes FIRST: an unwired parent kind is refused at
    # plan time whatever else the command line says, so a user who asked
    # for a model this tree cannot run learns that before being sent to
    # assemble inputs for it.
    build_model = None
    if args.parent_kind != "replay":
        import importlib

        module = importlib.import_module(_ENGINE_MODULE)
        build_model = getattr(module, _PARENT_BACKEND_SYMBOL, None)
        if build_model is None:
            raise CycleRefusal(
                f"this tree has no engine adapter for --parent-kind "
                f"{args.parent_kind}; the spine will not pretend to "
                f"integrate an atmosphere. Use --parent-kind replay to "
                f"rehearse the lattice against a recorded series, or land "
                f"{_ENGINE_MODULE}.{_PARENT_BACKEND_SYMBOL}",
                parent_kind=args.parent_kind, known_kinds=list(PARENT_KINDS),
                served_kinds=["replay"],
                adapter_module=_ENGINE_MODULE,
                adapter_symbol=_PARENT_BACKEND_SYMBOL,
                available=sorted(name for name in dir(module)
                                 if not name.startswith("_")))

    # A model parent's state comes OFF THE DEVICE, so it needs the port
    # rather than a recorded series.  Requiring --parent-state of it
    # would be asking for frames that the run then ignores.
    if build_model is not None:
        missing = [flag for flag, value in (("--port-root", args.port_root),
                                            ("--port-config",
                                             args.port_config),
                                            ("--port-steps", args.port_steps))
                   if value in (None, "")]
        if missing:
            raise CycleRefusal(
                "a model parent runs the port's dycore out of process and "
                "this run named no port to run",
                parent_kind=args.parent_kind, missing=missing,
                remedy="pass " + ", ".join(missing) + "; or use "
                       "--parent-kind replay to rehearse the lattice "
                       "against a recorded series")
    elif not args.parent_state:
        raise CycleRefusal(
            "a cycle with no parent state is not a cycle; the door will not "
            "write a receipt for a leg it never ran",
            parent_kind=args.parent_kind,
            remedy="pass --parent-state GLOB naming the parent's recorded "
                   "npz frames, or --dry-run to rehearse the lattice alone")
    if not args.parent_mesh_id:
        raise CycleRefusal(
            "every anchor carries a mesh identity and this run named none",
            parent_kind=args.parent_kind,
            remedy="pass --parent-mesh-id ID; an identity the spine guessed "
                   "is an identity no downstream reader can trust")

    if build_model is not None:
        return build_model(root=Path(args.root), clock=clock,
                           parent_kind=args.parent_kind,
                           mesh_id=str(args.parent_mesh_id),
                           port_root=args.port_root,
                           port_config=args.port_config,
                           port_steps=int(args.port_steps),
                           timeout=args.port_timeout,
                           history_frames=(_load_frames(args.parent_state)
                                           if args.parent_state else None),
                           increment_for=_increment_source(args))

    return build_replay_parent_engine(
        root=Path(args.root), clock=clock,
        history_frames=_load_frames(args.parent_state),
        mesh_id=str(args.parent_mesh_id), parent_kind="replay",
        increment_for=_increment_source(args))


def _analysis_for(args):
    """Hand the supervisor the gate evidence the parent engine produced.

    The supervisor only runs the three-hash ingestion gate when an
    ``analyse`` callable is injected, and the CLI injected NONE.  So even
    with the flat/nested shape defect fixed, the gate could not run
    through the front door: the engine computed the evidence, put it on
    the parent record, and nothing read it.  The receipt then said
    ``"ingestion": null`` about the most important instrument in the
    system for every run a user could actually start.

    What comes back is the FLAT ``verify_ingestion`` block -- the shape
    every shipped producer emits -- not a report nested under
    ``"ingestion"``, which is the shape only the hand-built tests use.

    With no ``--analysis-increment`` this returns ``None`` on purpose:
    that run is forecast-only BY CONSTRUCTION, which the supervisor
    distinguishes from a DA cycle that stopped receiving observations.
    """
    if not args.analysis_increment:
        return None

    def analyse(cycle_index, parent_record):
        block = parent_record.get("ingestion")
        if block is None:
            # Not a missing gate.  The gate compares the state BEFORE an
            # increment against the state AFTER it, so cycle N evidences
            # the increment applied at cycle N-1 and cycle 1 has genuinely
            # nothing before it.
            return {"state": "SKIPPED_NO_OBS",
                    "cycle_index": int(cycle_index),
                    "detail": "no increment preceded this boundary; the "
                              "gate's second hash does not exist until the "
                              "next parent leg has stepped"}
        return dict(block)

    return analyse


def _child_advance_for(clock, args):
    """Step every live child to the boundary, so the triple has children.

    Without this the supervisor's ``advance_children`` is ``None``, the
    arming test sees no children, and the child branch of the clock triple
    -- the branch that catches a nest drifting off the parent's step
    lattice -- is never exercised through the door.  LOUD STUB under
    ``--parent-kind replay``: it advances the child CLOCK, not a child
    dycore, exactly as the demo leg did.
    """
    if not args.child_slots:
        return None
    child_seconds = float(args.child_dt_seconds
                          if args.child_dt_seconds is not None
                          else args.parent_dt_seconds)
    child_step = seconds_to_ticks(child_seconds, label="--child-dt-seconds")

    def advance_children(cycle_index, parent_record, live):
        boundary = clock.boundary_ticks(int(cycle_index))
        return [{"grid_id": int(item["grid_id"]), "state": "LIVE",
                 "child_ticks": boundary // child_step,
                 "child_step_ticks": child_step}
                for item in live if item.get("grid_id") is not None
                and int(item.get("grid_id", -1)) >= 0]

    return advance_children


def _print_plan(args, clock, ratio, provider) -> None:
    print(f"cycle: root {Path(args.root)}")
    print(f"cycle: parent-kind {args.parent_kind}  "
          f"epoch-anchor {clock.epoch_anchor.isoformat()}")
    print(f"cycle: parent step {clock.parent_step_ticks} ticks "
          f"({ticks_to_seconds(clock.parent_step_ticks):g} s)  "
          f"cycle {clock.cycle_ticks} ticks "
          f"({ticks_to_seconds(clock.cycle_ticks):g} s) = "
          f"{clock.steps_per_cycle()} parent steps")
    print("cycle: boundary lattice")
    print(f"  {'index':>5}  {'ticks':>12}  valid time")
    for index, ticks, valid in clock.lattice():
        print(f"  {index:>5}  {ticks:>12}  {valid.isoformat()}")
    if ratio is None:
        print("cycle: no child slots reserved")
    else:
        print(f"cycle: {args.child_slots} child slot(s), step "
              f"{ratio.child_ticks} ticks "
              f"({ticks_to_seconds(ratio.child_ticks):g} s), "
              f"{ratio.ratio} child steps per parent step")
    print(f"cycle: placement-provider {args.placement_provider}"
          + ("" if provider is None else " (loaded)"))
    print(f"cycle: forecast-only budget {args.max_forecast_only_cycles} "
          f"cycles; placement clamp "
          f"{'allowed' if args.allow_placement_clamp else 'refused'}; "
          f"snap offset accepted up to "
          f"{args.accept_snap_offset_seconds:g} s")


def _standalone() -> int:                      # pragma: no cover - manual
    parser = argparse.ArgumentParser(prog="gpuwm cycle")
    sub = parser.add_subparsers(dest="command")
    register_cli(sub)
    return cycle_main(parser.parse_args())
