"""gpuwm command line for real-data workflows and verification cases.

``gpuwm fetch`` acquires initialization/boundary data (GFS/HRRR
download with manifests, ERA5 cdsapi template + validation); the
behavior lives in :mod:`gpuwm.fetch`.

``gpuwm static|ingest|run CONFIG`` sniffs the TOML shape: an
``[experiment]``/``[[domain]]`` file routes to the Phase-5 experiment
path (schema + validation in :mod:`gpuwm.experiment`, declared inputs in
:mod:`gpuwm.case_data`, pipeline in :mod:`gpuwm.runtime`), anything else
is a legacy :class:`RunConfig` dispatched to its registered real case.
``gpuwm resume CONFIG`` is sugar over ``run --restart``: it locates the
newest valid ``gpuwmrst`` checkpoint set in ``--outdir``
(:mod:`gpuwm.resume`) and continues through run's own dispatch, so the
restart machinery's identity refusals apply unchanged.  ``gpuwm render
WRFOUT...`` renders product PNGs (:mod:`gpuwm.render`).
``gpuwm import-namelist WPS INPUT`` translates WRF namelists into a
resolved experiment TOML and prints the substitution report.  ``gpuwm
verify CASE`` runs either an idealized benchmark or a real case, prints
its metrics dict, and exits 0 when every gate passes or 1 naming the
failing metric.  Gate intervals are single-sourced: each case module
exports ``GATES`` and both this driver and the benchmark tests consume
that one export.

No case is named here.  The case tables below are views over
:mod:`gpuwm.verify.cases`, which discovers case modules and classifies
them by the entry points they declare, so a case joins ``gpuwm verify``
and ``gpuwm static|ingest|run`` by existing -- this driver needs no edit.
``gpuwm cases`` prints that registry (``--json`` for a front end).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from gpuwm.config import load_config
from gpuwm.core.preflight import check_main
from gpuwm.core.preflight import register_cli as preflight_register_cli
from gpuwm.doctor import register_cli as doctor_register_cli
from gpuwm.domain_wizard import register_cli as domain_register_cli
from gpuwm.downscale import register_cli as downscale_register_cli
from gpuwm.experiment import is_experiment_toml
from gpuwm.fetch import register_cli as fetch_register_cli
from gpuwm.geog_assets import register_cli as geog_register_cli
from gpuwm.ingest.preflight import register_cli as ingest_register_cli
from gpuwm.render import register_cli as render_register_cli
from gpuwm.table_assets import register_cli as table_assets_register_cli
from gpuwm.verify import cases

#: Discovered verification cases: name -> case module exposing
#: ``run(outdir) -> dict`` and ``GATES``.  Membership comes from the
#: registry, and a module is imported only when it is indexed, so building
#: ``--help`` no longer drags every real-data case into the process.
_CASES = cases.CaseMapping(cases.VERIFY)

#: Config-driven real cases.  Modules own their static, ingest, and run
#: pipeline entry points; the CLI only validates/dispatches the case name.
_REAL_CASES = cases.CaseMapping(cases.CONFIG)

#: Pass gates per case, single-sourced from the case modules' GATES
#: exports: metric -> (lo, hi) with strict bounds lo < value < hi;
#: None = unbounded on that side.  Resolved through the registry rather
#: than through ``_CASES`` so a stubbed runner cannot relax a real gate.
_GATES = cases.GateMapping()


#: Flags whose value is a signed coordinate or coordinate list.
#:
#: argparse treats any token starting with ``-`` as an option string
#: unless it matches its own negative-number matcher,
#: ``^-\d+$|^-\d*\.\d+$``.  ``-33.87,151.21`` has a comma in it and so
#: does not match, which made every southern- or western-hemisphere
#: value fail with ``expected one argument`` -- exactly the worldwide
#: story the product leads with, and every documented example was CONUS
#: with a positive latitude, so the failure was invisible until someone
#: pointed the wizard at Sydney.
_COORDINATE_FLAGS = frozenset({"--point", "--area"})


def _looks_like_coordinates(value: str) -> bool:
    """A leading-minus token that is entirely comma-separated numbers."""

    if not value.startswith("-"):
        return False
    fields = value.split(",")
    if not 1 <= len(fields) <= 4:
        return False
    try:
        for field in fields:
            float(field)
    except ValueError:
        return False
    return True


def _join_negative_coordinates(argv: list[str]) -> list[str]:
    """Rewrite ``--point -33.87,151.21`` as ``--point=-33.87,151.21``.

    Only a coordinate flag immediately followed by an all-numeric
    leading-minus token is joined, so nothing else about option parsing
    changes and ``--point --help`` still errors the way it should.
    """

    joined: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        following = argv[index + 1] if index + 1 < len(argv) else None
        if (token in _COORDINATE_FLAGS and following is not None
                and _looks_like_coordinates(following)):
            joined.append(f"{token}={following}")
            index += 2
            continue
        joined.append(token)
        index += 1
    return joined


def _failing_gate(case: str, metrics: dict) -> str | None:
    """Name of the first failing gate metric, or None if all pass.

    Gates are open intervals with strict bounds, ``lo < v < hi``; ``None``
    leaves that side unbounded.  NaN metric values compare False and
    therefore fail their gate.
    """
    if metrics["nan"]:
        return "nan"
    for name, (lo, hi) in _GATES[case].items():
        v = metrics[name]
        if not ((lo is None or lo < v) and (hi is None or v < hi)):
            return name
    return None


#: The one experiment/config positional description, shared by
#: static/ingest/run/resume (the old "[run].case" text predated the
#: experiment front door and was stale).
_CONFIG_HELP = (
    "experiment TOML ([experiment]/[[domain]] tables, as emitted by "
    "`gpuwm domain` or `gpuwm import-namelist`; config-driven runs "
    "declare their inputs in [case_data]).  A legacy [run]-table "
    "RunConfig naming a registered case is also accepted.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gpuwm", description="GPU-native WRF-ARW-like weather model.")
    sub = parser.add_subparsers(dest="command", required=True)
    preflight_register_cli(sub)
    ingest_register_cli(sub)
    # Combined check policy (T3 handoff): the cheap CPU input/static/table
    # preflight runs first; only a zero return advances to the memory
    # estimator, so --alloc is gated on valid inputs.
    sub.choices["check"].set_defaults(
        func=lambda args: args.ingest_preflight_handler(args)
        or check_main(args))
    fetch_register_cli(sub)
    geog_register_cli(sub)
    domain_register_cli(sub)
    render_register_cli(sub)
    downscale_register_cli(sub)
    doctor_register_cli(sub)
    table_assets_register_cli(sub)
    lst = sub.add_parser(
        "cases", help="list the discovered verification cases and the "
                      "entry points each one declares")
    lst.add_argument("--json", action="store_true",
                     help="emit the registry as JSON for a front end")
    ver = sub.add_parser(
        "verify", help="run a benchmark or real case and check its gates")
    ver.add_argument("case", choices=sorted(_CASES),
                     help="verification case to run")
    ver.add_argument("--outdir", type=Path, default=None, metavar="OUT",
                     help="directory for the PNG and wrfout NetCDF output "
                          "(omit to compute metrics only)")
    static = sub.add_parser(
        "static", help="build static fields for a config-driven real case")
    static.add_argument("config", type=Path, metavar="CONFIG",
                        help=_CONFIG_HELP)
    static.add_argument("--output", type=Path,
                        default=Path("out/static_fields.npz"), metavar="NPZ",
                        help="static-field NPZ output")
    ingest = sub.add_parser(
        "ingest", help="build initialized fields for a config-driven case")
    ingest.add_argument("config", type=Path, metavar="CONFIG",
                        help=_CONFIG_HELP)
    ingest.add_argument("--output", type=Path,
                        default=Path("out/initial_state.npz"), metavar="NPZ",
                        help="initialized-state NPZ output")
    run = sub.add_parser("run", help="integrate a config-driven real case")
    run.add_argument("config", type=Path, metavar="CONFIG",
                     help=_CONFIG_HELP)
    run.add_argument("--outdir", type=Path, default=Path("out/run"),
                     metavar="OUT", help="wrfout output directory")
    run.add_argument("--restart", type=Path, default=None, metavar="RST",
                     help="resume from a gpuwmrst restart file written by "
                          "an earlier run of the SAME config (only the "
                          "forecast length / output and restart cadence "
                          "may differ); restart writing itself is the "
                          "restart_interval_s config key")
    resume = sub.add_parser(
        "resume",
        help="continue a run from its newest valid gpuwmrst checkpoint "
             "(sugar over run --restart; same config, same outdir)")
    resume.add_argument("config", type=Path, metavar="CONFIG",
                        help="the SAME config the interrupted run used; "
                             "the restart identity check refuses any "
                             "other")
    resume.add_argument("--from", dest="from_checkpoint", default="latest",
                        metavar="CKPT|latest",
                        help="explicit gpuwmrst_*.npz checkpoint, or "
                             "'latest' (default) to take the newest set "
                             "in --outdir whose members validate")
    resume.add_argument("--outdir", type=Path, default=Path("out/run"),
                        metavar="OUT",
                        help="the interrupted run's wrfout/checkpoint "
                             "directory (default out/run)")
    # Compose with the existing memory-preflight registrar above; Task 15
    # owns only the run parser's supervision flags and dispatch helper.
    from gpuwm.supervisor import register_cli as supervisor_register_cli
    supervisor_register_cli(sub)
    # resume continues a supervised run, so it carries run's exact
    # supervision surface; the checkpoint is resolved below and dispatch
    # is then run's own path with args.restart set.
    supervisor_register_cli(sub, "resume")
    imp = sub.add_parser(
        "import-namelist",
        help="translate WRF namelist.wps + namelist.input into a "
             "resolved gpuwm experiment TOML plus a substitution report")
    imp.add_argument("wps", type=Path, metavar="WPS",
                     help="WPS namelist.wps (projection + nest layout)")
    imp.add_argument("input", type=Path, metavar="INPUT",
                     help="WRF namelist.input (domains/physics/dynamics/"
                          "bdy_control)")
    imp.add_argument("--output", type=Path, default=None, metavar="TOML",
                     help="write the resolved experiment TOML here "
                          "(omit to print the report only)")
    imp.add_argument("--name", default=None, metavar="NAME",
                     help="[experiment].name for the resolved TOML "
                          "(default derived from start time and domain "
                          "count)")
    imp.add_argument("--rrtmg-variant", default="rte-rrtmgp",
                     choices=("rte-rrtmgp", "rrtmg_legacy"),
                     dest="rrtmg_variant",
                     help="implementation for a WRF RRTMG 4/4 request: "
                          "the established RTE+RRTMGP substitution "
                          "(default, unchanged output) or the exact "
                          "legacy-RRTMG port (fails closed at physics "
                          "setup until its compute kernels land)")
    args = parser.parse_args(_join_negative_coordinates(
        list(sys.argv[1:] if argv is None else argv)))

    try:
        return _dispatch(args)
    except ModuleNotFoundError as error:
        # The base install ships no GPU runtime.  A missing cupy is a
        # documented install gap with one remedy, not a traceback.
        if (error.name or "").split(".")[0] in ("cupy", "cupy_backends"):
            print(
                f"gpuwm {args.command}: this command needs the GPU "
                "runtime (CuPy), which the base install does not "
                "include.\n"
                "  remedy: pip install 'gpuwm[gpu]'   "
                "(or: pip install cupy-cuda12x)\n"
                "  `gpuwm doctor` checks the whole runtime estate.",
                file=sys.stderr)
            return 2
        raise
    except ValueError as error:
        # Documented refusals (the wizard's latitude/antimeridian
        # windows, fetch's source contracts, downscale's cadence and
        # evidence contracts, the importer's namelist contract, the
        # config loaders' schema errors, ...) are raised as ValueError
        # (incl. DomainFitError).  They are user-facing messages, not
        # programming errors: print the message, exit 2 (argparse's
        # usage-error convention), no traceback -- uniformly, for every
        # subcommand.
        print(f"gpuwm {args.command}: {error}", file=sys.stderr)
        return 2
    except RuntimeError as error:
        if args.command in ("fetch", "fetch-geog"):
            # fetch/fetch-geog RuntimeErrors are operational outcomes
            # with a stated remedy (no complete latest cycle on the
            # mirrors; a --wait-for window that timed out with its
            # resume story; a download that must be re-run to resume),
            # not programming errors.  Scoped to the fetch commands:
            # elsewhere a RuntimeError (including CUDA runtime
            # failures) must keep its traceback.
            print(f"gpuwm {args.command}: {error}", file=sys.stderr)
            return 2
        raise


def _dispatch(args) -> int:
    if args.command in ("check", "fetch", "fetch-geog", "domain", "render",
                        "downscale", "doctor", "fetch-tables"):
        return args.func(args)

    if args.command == "cases":
        records = cases.manifest()
        if args.json:
            import json
            print(json.dumps(records, indent=2, sort_keys=True))
        else:
            for record in records:
                print(f"{record['name']:<24} "
                      f"{','.join(record['capabilities'])}")
        return 0

    if args.command == "verify":
        metrics = _CASES[args.case].run(outdir=args.outdir)
        # List-valued metrics are time series (wk82 w_max_hist) that would
        # swamp the terminal; gates only ever bound scalars.
        print({k: v for k, v in metrics.items() if not isinstance(v, list)})
        bad = _failing_gate(args.case, metrics)
        if bad is not None:
            print(f"FAIL: gate {bad!r} = {metrics[bad]}", file=sys.stderr)
            return 1
        return 0

    if args.command == "import-namelist":
        from gpuwm.namelist_import import import_namelists
        if args.output is not None and args.output.resolve() in (
                args.wps.resolve(), args.input.resolve()):
            raise ValueError(
                f"--output {args.output} aliases an input namelist; "
                "refusing to overwrite the source file.")
        toml_text, report = import_namelists(
            args.wps, args.input, name=args.name,
            rrtmg_variant=args.rrtmg_variant)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            # Atomic publication: a mid-write failure must never leave a
            # truncated TOML behind.
            tmp = args.output.with_suffix(args.output.suffix + ".tmp")
            with open(tmp, "w", newline="\n") as fh:
                fh.write(toml_text)
            os.replace(tmp, args.output)
            print(f"import-namelist: wrote {args.output}")
        print(report.format())
        return 0

    if args.command == "resume":
        # Locate only; every safety property of the resume (manifest
        # validation, config/setup/physics identity, complete tree set)
        # is the run machinery's own and runs on the path set here.
        from gpuwm.resume import resolve_resume_checkpoint
        resolution = resolve_resume_checkpoint(
            args.outdir, args.from_checkpoint)
        for note in resolution.skipped:
            print(f"resume: skipped newer checkpoint {note}",
                  file=sys.stderr)
        print(f"resume: continuing from {resolution.checkpoint}")
        args.restart = resolution.checkpoint
        # Fall through to the run dispatch below.

    # [[domain]]/[experiment] tables route to the experiment path; the
    # legacy [grid]/[dynamics]/[run] shape stays on the frozen case path.
    # (A missing file falls through to the legacy loader's own error.)
    if args.config.exists() and is_experiment_toml(args.config):
        from gpuwm import runtime
        from gpuwm.case_data import load_experiment_case
        exp, data = load_experiment_case(args.config)
        if args.command == "static":
            output = runtime.write_static(exp, data, args.output)
            print(f"static {exp.name}: {output}")
        elif args.command == "ingest":
            output = runtime.write_ingest(exp, data, args.output)
            print(f"ingest {exp.name}: {output}")
        else:
            if not args.no_supervise:
                from gpuwm.supervisor import supervise_from_cli
                return supervise_from_cli(args)
            summary = runtime.run_experiment(exp, data, args.outdir,
                                             restart=args.restart,
                                             health_debug=args.health_debug)
            print({"experiment": exp.name, "outdir": str(args.outdir),
                   "wrfout_count": len(summary.wrfout_paths),
                   "completed_seconds": summary.completed_seconds,
                   "nan_free": summary.nan_free,
                   "microphysics_transition_receipt": (
                       None if getattr(
                           summary, "microphysics_transition_receipt", None
                       ) is None else str(
                           summary.microphysics_transition_receipt)),
                   "microphysics_transition_receipt_sha256":
                       getattr(
                           summary,
                           "microphysics_transition_receipt_sha256", None),
                   "restarted": args.restart is not None})
        return 0

    cfg = load_config(args.config)
    if not cfg.case:
        raise ValueError(
            f"config file {args.config} must set case in the [run] table")
    if cfg.case not in _REAL_CASES:
        raise ValueError(
            f"config case {cfg.case!r} is not a registered real case; "
            f"choose one of {sorted(_REAL_CASES)}")
    case = _REAL_CASES[cfg.case]
    if args.command == "static":
        output = case.write_static(cfg, args.output)
        print(f"static {cfg.case}: {output}")
    elif args.command == "ingest":
        output = case.write_ingest(cfg, args.output)
        print(f"ingest {cfg.case}: {output}")
    else:
        summary = case.run_config(cfg, args.outdir, restart=args.restart)
        print({"case": cfg.case, "outdir": str(args.outdir),
               "wrfout_count": len(summary.wrfout_paths),
               "completed_seconds": summary.completed_seconds,
               "nan_free": summary.nan_free,
               "restarted": args.restart is not None})
    return 0


if __name__ == "__main__":
    sys.exit(main())
