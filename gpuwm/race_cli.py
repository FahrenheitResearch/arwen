"""The table-driven ``gpuwm race`` front door."""
from __future__ import annotations

from pathlib import Path
import sys


def register_cli(subparsers):
    parser = subparsers.add_parser("race", help="race a recipe's locally published forecast against its source objects")
    parser.add_argument("recipe", help="recipe ID or registry alias")
    parser.add_argument("--cycle", required=True, help="HHZ (today UTC) or YYYY-MM-DDTHH")
    parser.add_argument("--devices", type=int)
    parser.add_argument("--dry", action="store_true", help="predict from observed posting times; no forecast")
    parser.add_argument("--hours", type=int, default=48, help="forecast horizon (default 48; use 1 for the live door test)")
    parser.add_argument("--outdir", type=Path, default=Path("race-runs"), help="parent for a fresh receipt/run directory")
    parser.add_argument("--tiles", choices=("auto", "off", "on"), default="auto")
    parser.add_argument("--hydro", default="auto", metavar="auto|on|off|SETTINGS.toml",
                        help="hydro is opt-in: auto keeps the forecast default (off), on enables it, or select settings")
    from gpuwm.smoke_execution import SMOKE_FLAG_HELP
    parser.add_argument("--smoke", action="store_true", default=False, help=SMOKE_FLAG_HELP)
    parser.add_argument("--products", default="all", help="products defining online, passed to forecast")
    parser.add_argument("--publish-root", type=Path, help="local web root; default RUN/www")
    parser.add_argument("--publish-receipt", type=Path, help="explicit hourly publish receipt")
    parser.add_argument("--initial-inputs", type=Path, help="separate analysis manifest for the go fallback")
    parser.add_argument("--data-dir", type=Path, help="existing downloaded input cache")
    parser.add_argument("--posting-times", type=Path, help="timing study JSON; default fetch cycle's live S3 listing")
    parser.add_argument("--receipt", type=Path, help="JSON with prediction_inputs (or top-level timing keys)")
    parser.add_argument("--seconds-per-hour", type=float, help="measured steady wall seconds per forecast hour, including output workload")
    parser.add_argument("--launch-to-first-step", type=float, help="seconds from launch to the first completed step")
    parser.add_argument("--publish-lag", type=float, help="seconds from hour close to products online")
    parser.add_argument("--launch-delay", type=float, help="seconds from analysis LastModified to launch (default 0)")
    parser.add_argument("--measurement-note", action="append", default=[], help="measurement workload or prediction assumption, preserved in receipt")
    parser.add_argument("--box-label", help="label for the measured box in a dry prediction")
    parser.add_argument("--object-endpoint", help="S3-compatible listing URL (supports a local HTTP test store)")
    parser.add_argument("--poll-seconds", type=float, default=30, help="source listing interval, at least 20 seconds (two sources; default 30)")
    parser.add_argument("--watch-timeout", type=float, default=14400, help="observation budget in seconds; never kills forecast")
    parser.set_defaults(func=main)


def main(args):
    from gpuwm import race
    import math
    try:
        row = race.recipe_row(args.recipe)
        cycle = race.cycle_time(args.cycle)
        if not 1 <= args.hours <= row["race"]["max_hours"]:
            raise ValueError(f"--hours must be between 1 and {row['race']['max_hours']} for this race row.")
        if args.hours > row["race"].get("short_cycle_hours", row["race"]["max_hours"]) and cycle.hour not in row["race"]["extended_cycle_hours"]:
            raise ValueError("This cycle does not publish the requested horizon; select a registry-declared extended cycle hour.")
        if args.devices is not None and args.devices < 1:
            raise ValueError("--devices must be positive.")
        if not math.isfinite(args.poll_seconds) or args.poll_seconds < 20:
            raise ValueError("--poll-seconds must be at least 20 to keep the hourly object watch to a few requests per minute.")
        if not math.isfinite(args.watch_timeout) or args.watch_timeout <= 0:
            raise ValueError("--watch-timeout must be finite and positive.")
        if args.dry:
            race.dry_inputs(args)  # Validate before allocating a run directory.
        from gpuwm.run_stamp import allocate
        root = allocate(args.outdir.resolve(), init=cycle)
        receipt = {"schema": "gpuwm.race.v1", "mode": "dry" if args.dry else "live", "status": "starting",
                   "recipe": row["id"], "cycle": race.stamp(cycle), "hours": args.hours,
                   "started_at": race.stamp(), "notes": list(args.measurement_note), "box": race.machine(),
                   "first_step_definition": "first completed model step",
                   "hydro": {"requested": args.hydro, "decision": "prediction assumption" if args.dry else "awaiting forecast receipt"},
                   "products": args.products.split(","), "box_label": args.box_label}
        print(f"race: receipts {root}", flush=True)
        try:
            if args.dry:
                race.dry_race(args, row, cycle, root, receipt)
                print(f"race: predicted f{args.hours:02d} margin {receipt['margin_seconds']:+.1f} s", flush=True)
                return 0
            return race.live_race(args, row, cycle, root, receipt)
        except (OSError, ValueError) as error:
            receipt.update(status="failed", error=str(error))
            receipt["notes"].append(str(error))
            race.write_receipt(root, receipt)
            raise
    except (OSError, ValueError) as error:
        print(f"race: {error}", file=sys.stderr)
        return 2
