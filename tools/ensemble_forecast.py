"""Entry point for the sequential single-GPU ensemble engine (EXPERIMENTAL).

Not wired into ``gpuwm``'s default routes: this tool is the only way in,
so nothing that exists today changes behaviour because this landed.

    python -m tools.ensemble_forecast run   --ensemble-config ENS.toml \
                                            --ens-root DIR
    python -m tools.ensemble_forecast cycle --ensemble-config ENS.toml \
                                            --ens-root DIR \
                                            --cycles 2 --cycle-seconds 600
    python -m tools.ensemble_forecast bench --ens-root DIR
    python -m tools.ensemble_forecast status --ens-root DIR

``run`` resumes by default: members already recorded DONE are left
alone.  ``--member N`` targets one member and refuses it if it is DONE.

SCIENTIFIC LIMITS OF WHAT THIS TOOL PRODUCES
--------------------------------------------
Printed by every ``run`` and ``cycle`` invocation as well, because the
place to learn them is before configuring an experiment, not after
reading a finished manifest.  With ``perturbation = "gpuwm.da.perturb"``:

* no mass balance -- ``mu'`` is untouched and the column is not
  re-balanced hydrostatically;
* no wind balance -- the u/v increments are neither non-divergent nor in
  geostrophic/gradient balance with the temperature increment, so the
  first minutes radiate gravity waves;
* one shared, unperturbed lateral boundary file for every member.  The
  rim taper is what keeps that legal, and it means ensemble spread decays
  toward the rim BY CONSTRUCTION;
* lateral taper only -- no vertical taper, so perturbations reach the
  model top and the surface at full amplitude;
* no surface, soil, or physics-parameter perturbation, and no ``w``,
  ``mu'`` or hydrometeor perturbation;
* the draw is FFT-periodic, so opposite edges of the raw field are
  correlated.  Horizontally the rim taper hides it; vertically nothing
  does, and a nonzero ``vertical_scale_levels`` leaves the top and bottom
  levels strongly correlated (the module records the exact figure);
* determinism is "byte-identical given the seed ON ONE SOFTWARE AND
  DEVICE STACK".  A different FFT backend gives different bytes from the
  same seed, and the provenance records which backend ran.

Nothing this tool produces is on a certified forecast path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gpuwm.ensemble.bench import bench_from_manifest  # noqa: E402
from gpuwm.ensemble.config import load_ensemble_config  # noqa: E402
from gpuwm.ensemble.cycle import run_cycles  # noqa: E402
from gpuwm.ensemble.engine import run_ensemble  # noqa: E402
from gpuwm.ensemble.manifest import (  # noqa: E402
    ENSEMBLE_MANIFEST_NAME, ENSEMBLE_MANIFEST_SCHEMA, read_manifest,
)


#: The interpretation limits, as one block, printed before a run starts.
#: Lifted from this module's own docstring so the two cannot drift.
SCIENTIFIC_LIMITS = __doc__.split(
    "SCIENTIFIC LIMITS OF WHAT THIS TOOL PRODUCES")[1].split("--\n", 1)[1]


def _print_limits() -> None:
    print("ensemble: EXPERIMENTAL (v1.2). Interpretation limits:",
          file=sys.stderr)
    print(SCIENTIFIC_LIMITS.strip(), file=sys.stderr)


def _resolve_root(args, cfg) -> Path:
    if args.ens_root is not None:
        return Path(args.ens_root)
    if cfg is not None and cfg.ens_root is not None:
        return cfg.ens_root
    raise ValueError(
        "no ensemble root: pass --ens-root, or set ens_root in the "
        "[ensemble] table")


def _print_event(event) -> None:
    print(json.dumps(event, sort_keys=True), flush=True)


def resolve_assimilate(reference: str):
    """Resolve ``--assimilate module.path:callable`` (or dotted).  Fails closed.

    The callable must have the cycle driver's signature,
    ``assimilate(cycle_index, member_states) -> {member: {field: array}}``.
    It is a dotted path rather than a fixed name because no assimilation
    method belongs to the engine, and hard-coding one would make the seam
    a preference instead of an interface.

    Every failure names what was tried.  A cycling run that quietly fell
    back to forecast-only would produce a manifest full of true statements
    describing a run nobody asked for.
    """
    import importlib

    text = (reference or "").strip()
    if not text:
        raise ValueError("--assimilate needs a dotted path, got an empty "
                         "string")
    if ":" in text:
        module_name, _, attribute = text.partition(":")
    else:
        module_name, _, attribute = text.rpartition(".")
    if not module_name or not attribute:
        raise ValueError(
            f"--assimilate {text!r} is not a dotted path to a callable; "
            "write it as 'package.module:callable' or "
            "'package.module.callable'")
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise ValueError(
            f"--assimilate {text!r}: cannot import {module_name!r} "
            f"({error})") from error
    target = getattr(module, attribute, None)
    if target is None:
        raise ValueError(
            f"--assimilate {text!r}: {module_name} has no {attribute!r}")
    if not callable(target):
        raise ValueError(
            f"--assimilate {text!r}: {module_name}.{attribute} is a "
            f"{type(target).__name__}, not a callable")
    return target


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m tools.ensemble_forecast", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the members sequentially")
    run.add_argument("--ensemble-config", type=Path, required=True)
    run.add_argument("--ens-root", type=Path, default=None)
    run.add_argument("--member", type=int, action="append", default=None,
                     dest="members",
                     help="run only this member (repeatable); refused if "
                          "the manifest already records it DONE")
    run.add_argument("--run-seconds", type=float, default=None,
                     help="override the base config's forecast length "
                          "(bench and smoke use this; the manifest "
                          "records what actually ran)")
    run.add_argument("--no-resume", action="store_true",
                     help="refuse instead of skipping completed members")

    cycle = sub.add_parser(
        "cycle", help="run forecast legs with an assimilation seam between")
    cycle.add_argument("--ensemble-config", type=Path, required=True)
    cycle.add_argument("--ens-root", type=Path, default=None)
    cycle.add_argument("--cycles", type=int, required=True)
    cycle.add_argument("--cycle-seconds", type=float, required=True)
    cycle.add_argument(
        "--assimilate", default=None, metavar="module:callable",
        help="dotted path to assimilate(cycle_index, member_states); "
             "without it the legs are forecast-only and the assimilation "
             "slot stays null")
    cycle.add_argument(
        "--positivity", default="clip", choices=("clip", "reject", "none"),
        help="policy bounding analysed hydrometeors below zero "
             "(default: clip, which ADDS mass and reports how much)")
    cycle.add_argument(
        "--no-restart-from-analysis", action="store_true",
        help="re-prepare every leg from the base config instead of "
             "starting it at the previous leg's analysis")

    bench = sub.add_parser("bench", help="print the per-member timing table")
    bench.add_argument("--ens-root", type=Path, required=True)

    status = sub.add_parser("status", help="print the manifest rollup")
    status.add_argument("--ens-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    try:
        if args.command == "run":
            cfg = load_ensemble_config(args.ensemble_config)
            root = _resolve_root(args, cfg)
            _print_limits()
            result = run_ensemble(
                cfg, root, members=args.members,
                run_seconds=args.run_seconds,
                resume=not args.no_resume, on_event=_print_event)
            print(json.dumps({
                "schema": ENSEMBLE_MANIFEST_SCHEMA,
                "status": result.status,
                "ens_root": str(result.ens_root),
                "manifest": str(result.manifest_path),
                "ran": list(result.ran),
                "skipped": list(result.skipped),
            }, sort_keys=True))
            return 0 if result.status == "COMPLETE" else 1

        if args.command == "cycle":
            cfg = load_ensemble_config(args.ensemble_config)
            root = _resolve_root(args, cfg)
            assimilate = (None if args.assimilate is None
                          else resolve_assimilate(args.assimilate))
            _print_limits()
            # The CLI knows the dotted path the operator named; the engine
            # cannot infer it, and a receipt that could not say which
            # method produced an analysis was the gap this closes.
            method = None if args.assimilate is None else {
                "resolved_from": args.assimilate,
                "declared_by": "tools.ensemble_forecast --assimilate",
                "note": "the engine applied these increments; it makes no "
                        "claim about the method that produced them beyond "
                        "naming what it invoked",
            }
            result = run_cycles(
                cfg, root, n_cycles=args.cycles,
                cycle_seconds=args.cycle_seconds,
                assimilate=assimilate, positivity=args.positivity,
                restart_from_analysis=not args.no_restart_from_analysis,
                assimilation_method=method,
                on_event=_print_event)
            print(json.dumps({
                "status": result.status,
                "ens_root": str(result.ens_root),
                "manifest": str(result.manifest_path),
                "cycles_run": list(result.cycles_run),
                "assimilate": args.assimilate,
                "positivity": args.positivity,
            }, sort_keys=True))
            return 0

        if args.command == "bench":
            _, table = bench_from_manifest(args.ens_root)
            print(table)
            return 0

        if args.command == "status":
            manifest = read_manifest(
                Path(args.ens_root) / ENSEMBLE_MANIFEST_NAME,
                schema=ENSEMBLE_MANIFEST_SCHEMA)
            print(json.dumps({
                "status": manifest.get("status"),
                "n_members": manifest.get("n_members"),
                "members": [
                    {"index": record["index"], "status": record["status"],
                     "final_state_sha256": record.get("final_state_sha256")}
                    for record in manifest.get("members", ())
                ],
            }, indent=2, sort_keys=True))
            return 0
    except ValueError as error:
        print(f"ensemble {args.command}: {error}", file=sys.stderr)
        return 2
    raise AssertionError(f"unreachable command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
