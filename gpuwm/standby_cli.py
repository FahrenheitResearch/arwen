"""The recipe-driven, in-process warm-start forecast door."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import sys
import tomllib

from gpuwm.standby import run_standby


def positive_seconds(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError("seconds must be positive and finite")
    return number


def recipe_document(recipe, *, table=None):
    from gpuwm.region_recipe import RECIPE_TABLE
    table = Path(table or RECIPE_TABLE)
    if Path(recipe).is_file():
        path = Path(recipe).resolve()
        row = {"id": path.stem, "config": str(path)}
    else:
        rows = json.loads(table.read_text())["recipes"]
        row = next((r for r in rows if recipe == r["id"] or recipe in r.get("aliases", ())), None)
        if row is None:
            raise ValueError(f"recipe table has no recipe or alias {recipe!r}")
        path = table.parent / row["config"]
    return row, path, tomllib.loads(path.read_text(encoding="utf-8-sig"))


def resolve_cycle(value, *, source, now=None):
    from gpuwm.fetch_routes import route_for
    route = route_for(source)
    if value != "next":
        moment = datetime.strptime(value, "%Y-%m-%dT%H")
        if moment.hour not in route.cycle_hours:
            raise ValueError(f"{source} has no {moment.hour:02d}Z cycle")
        return moment
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is not None:
        now = now.astimezone(timezone.utc).replace(tzinfo=None)
    candidate = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    for _ in range(24):
        if candidate.hour in route.cycle_hours:
            return candidate
        candidate += timedelta(hours=1)
    raise ValueError(f"{source} declares no cycle hours")


def analysis_plan(document, cycle):
    from gpuwm.fetch_routes import resolve_request
    source = document.get("fetch", {}).get("initial_source")
    if not source:
        raise ValueError("standby recipe must declare [fetch].initial_source; "
                         "otherwise its analysis object and decoder are undefined")
    plan = resolve_request(source, cycle=cycle, hours=0)
    primary = {str(p) for p in plan.primary_files}
    objects = [obj for obj in plan.objects if obj.relpath in primary]
    if not objects:
        # A composed primary waits for its first declared component. The
        # ordinary fetch route will assemble the complete inventory afterward.
        objects = list(plan.objects)
    if not objects:
        raise ValueError("analysis source declares no objects to wait for")
    return plan, objects[0]


def standby_main(args):
    from gpuwm.toml_document import emit_experiment_toml
    from gpuwm.standby_runtime import PreparedStandby
    from gpuwm.hydro.runtime import go_scope
    try:
        row, original, document = recipe_document(args.recipe)
        source = document.get("fetch", {}).get("initial_source")
        if not source:
            raise ValueError("recipe has no [fetch].initial_source for standby")
        cycle = resolve_cycle(args.cycle, source=source)
        hours = args.hours if args.hours is not None else document["experiment"]["run_seconds"] / 3600
        if not math.isfinite(hours) or hours <= 0:
            raise ValueError("forecast hours must be positive and finite")
        document["experiment"]["start_time"] = cycle
        document["experiment"]["run_seconds"] = hours * 3600.0
        document["fetch"].update(cycle=cycle.strftime("%Y-%m-%dT%H"),
                                  hours=math.ceil(hours), as_posted=False)
        if args.tiles is not None:
            document.setdefault("tiles", {})["mode"] = args.tiles
        if args.devices is not None:
            if args.devices < 1:
                raise ValueError("devices must be positive")
            document.setdefault("devices", {})["count"] = args.devices
            # Several cards hold the domain on the multi-card store road,
            # which refuses enabled [tiles] ("one domain has one road"):
            # "auto" there means the cards hold it, so it resolves to off.
            tiles_table = document.get("tiles")
            if (args.devices > 1 and isinstance(tiles_table, dict)
                    and tiles_table.get("mode") == "auto" and set(tiles_table) == {"mode"}):
                tiles_table["mode"] = "off"
        if getattr(args, "smoke", False):
            from gpuwm.smoke_execution import apply_smoke_settings
            apply_smoke_settings(document)
        plan, watched = analysis_plan(document, cycle)
        root = Path(args.outdir or f"standby-{row['id']}-{cycle:%Y%m%dT%H}").resolve()
        root.mkdir(parents=True, exist_ok=True)
        # Existing evidence is immutable. A repeat gets a fresh owned attempt.
        if (root / "standby-receipt.json").exists():
            root = root / ("attempt-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f"))
            root.mkdir()
        args.outdir = root
        args.config = root / "experiment.toml"
        args.config.write_text(emit_experiment_toml(document), encoding="utf-8")
        args.hydro = None if args.hydro == "auto" else args.hydro
        backend = PreparedStandby(args, plan, watched)
        # Resolve and retain the normal hydro decision throughout prestage,
        # preparation, and forecast, including its inherited worker scope.
        @go_scope
        def run(scoped):
            from gpuwm.hydro.door import inherited_decision
            return run_standby(backend, urls=watched.urls(plan.ladder), outdir=root,
                metadata={"recipe": row["id"], "recipe_path": str(original),
                          "cycle": cycle.strftime("%Y-%m-%dT%H"), "hours": hours,
                          "hydro": inherited_decision()},
                store=backend.store, poll_s=args.poll_s, timeout_s=args.wait_timeout_s)
        receipt = run(args)
        print(f"standby: first step {receipt['object_seen_to_first_step_seconds']:.3f} s "
              f"after object seen; receipt {root / 'standby-receipt.json'}", flush=True)
        return 0
    except KeyboardInterrupt:
        print("standby: interrupted; any warm resources were closed", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"standby FAILED: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1


def register_cli(subparsers):
    parser = subparsers.add_parser("standby", help="warm a recipe and launch when its analysis appears")
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--cycle", default="next")
    parser.add_argument("--devices", type=int)
    parser.add_argument("--poll-s", type=positive_seconds, default=3.0)
    parser.add_argument("--wait-timeout-s", type=positive_seconds, default=7200.0)
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--hours", type=positive_seconds)
    parser.add_argument("--tiles", choices=("auto", "on", "off"))
    parser.add_argument("--geog-root", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--hydro", default="auto", metavar="auto|off|SETTINGS")
    from gpuwm.smoke_execution import SMOKE_FLAG_HELP
    parser.add_argument("--smoke", action="store_true", default=False, help=SMOKE_FLAG_HELP)
    parser.add_argument("--publish-root", type=Path)
    parser.add_argument("--products", "--render-products", dest="products", default=None)
    parser.add_argument("--fetch-workers", type=int, default=8)
    parser.set_defaults(func=standby_main)
