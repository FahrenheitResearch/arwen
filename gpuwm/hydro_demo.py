"""Pinned NextGen examples: metadata orchestration around Rust asset I/O."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path, PurePosixPath
import sys
from time import perf_counter
from uuid import uuid4


TABLE = Path(__file__).parent / "data/hydro/demos.v1.json"


def examples():
    table = json.loads(TABLE.read_text(encoding="utf-8"))
    if table["schema"] != "gpuwm-nextgen-demos-v1":
        raise ValueError("unsupported NextGen demo table")
    rows = {row["id"]: row for row in table["examples"]}
    if len(rows) != len(table["examples"]):
        raise ValueError("duplicate NextGen example id")
    for row in rows.values():
        paths = set()
        for asset in row["assets"]:
            path = PurePosixPath(asset["path"])
            if (path.is_absolute() or ".." in path.parts or "\\" in asset["path"]
                    or ":" in asset["path"] or not path.parts or path in paths):
                raise ValueError("example assets need distinct safe relative paths")
            paths.add(path)
    return rows


def positive_seconds(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("duration must be finite and positive")
    return value


def settings_toml(settings):
    """Serialize JSON-compatible setup metadata, including nested stage tables."""
    def key(value):
        return json.dumps(value, ensure_ascii=False)

    def scalar(value):
        if isinstance(value, dict):
            return "{ " + ", ".join(f"{key(k)} = {scalar(v)}" for k, v in value.items()) + " }"
        if isinstance(value, list):
            return "[" + ", ".join(scalar(v) for v in value) + "]"
        if value is None:
            raise ValueError("hydro TOML settings cannot contain null")
        return json.dumps(value, ensure_ascii=False, allow_nan=False)

    lines = []

    def table(values, path, *, array=False):
        dotted = ".".join(key(part) for part in path)
        lines.append(f"[[{dotted}]]" if array else f"[{dotted}]")
        nested = []
        for name, value in values.items():
            if isinstance(value, dict) or (isinstance(value, list) and value
                                         and all(isinstance(v, dict) for v in value)):
                nested.append((name, value))
            else:
                lines.append(f"{key(name)} = {scalar(value)}")
        for name, value in nested:
            children = [value] if isinstance(value, dict) else value
            for child in children:
                lines.append("")
                table(child, (*path, name), array=isinstance(value, list))

    table(settings, ("hydro",))
    return "\n".join(lines) + "\n"


def prepare(name, *, outdir, data_dir, strict_numerics=False):
    """Materialize immutable pinned inputs, never load numerical fields here."""
    row = examples()[name]
    admission = row.get("admission", {})
    if admission.get("status") != "permitted":
        reason = admission.get("reason", "dataset licence verification is incomplete")
        raise RuntimeError(f"example {name!r} is unavailable: {reason}")
    from gpuwm.geog_assets import _rw_asset

    parent = Path(outdir).expanduser().resolve()
    cache = Path(data_dir).expanduser().resolve()
    parent.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    run = parent / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                    + "-" + uuid4().hex[:12])
    run.mkdir()
    workers = min(8, os.cpu_count() or 1, len(row["assets"]))
    receipt = {"schema": "gpuwm-nextgen-demo-v1", "example": name,
               "source": row["source"], "status": "preparing", "stages": [],
               "run_directory": str(run), "strict_numerics": bool(strict_numerics)}
    begin = perf_counter()

    def materialize(asset):
        pin = {key: asset[key] for key in ("bytes", "sha256")}
        stored = cache / asset["sha256"]
        started = perf_counter()
        reused = False
        if stored.exists():
            try:
                _rw_asset("hash", input=stored, **pin)
                reused = True
            except RuntimeError as error:
                if "pin refused:" not in str(error):
                    raise
                stored.rename(stored.with_name(stored.name + ".unverified-" + uuid4().hex))
        if not reused:
            _rw_asset("asset", url=asset["url"], output=stored, **pin)
        # Rust streams a separate copy, so mutable runtime inputs cannot change
        # the verified cache or another run's state.
        destination = run / "inputs" / asset["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        _rw_asset("asset", url=stored.as_uri(), output=destination, **pin)
        return {"path": asset["path"], **pin, "cache_reused": reused,
                "seconds": perf_counter() - started}

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            receipt["assets"] = list(pool.map(materialize, row["assets"]))
        receipt["stages"].append({"stage": "verified input acquisition and materialization",
                                  "seconds": perf_counter() - begin,
                                  "parallelism": f"{workers} concurrent Rust asset workers; Rust HTTP range workers per large object"})
        settings = {**row["settings"], "numerics": "strict" if strict_numerics else "fast"}
        for key, value in row["paths"].items():
            settings[key] = str(run / "inputs" / value)
        config = run / "hydro.toml"
        config.write_text(settings_toml(settings), encoding="utf-8")
        receipt.update(status="prepared", settings=str(config))
        return config, run, receipt
    except Exception as error:
        receipt.update(status="preparation failed", error=str(error))
        raise
    finally:
        (run / "demo.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


def demo_main(args):
    try:
        row = examples()[args.example]
        seconds = positive_seconds(args.seconds if args.seconds is not None else row["seconds"])
        if args.device < 0:
            raise ValueError("CUDA device ordinal must be nonnegative")
        runner = getattr(args, "nextgen_demo_runner", None)
        if not args.prepare_only and runner is None:
            raise RuntimeError("the NextGen standalone runner has not been bound to this command; "
                               "--prepare-only acquires inputs without simulating water")
        if not args.prepare_only:
            interval = positive_seconds(row["settings"]["interval_seconds"])
            if seconds % interval != 0:
                raise ValueError("--seconds must be a positive whole number of hydro intervals "
                                 f"({interval:g} seconds for this example)")
        config, run, receipt = prepare(args.example, outdir=args.outdir,
                                      data_dir=args.data_dir, strict_numerics=args.strict_numerics)
        receipt["requested_seconds"] = seconds
        if not args.prepare_only:
            begin = perf_counter()
            try:
                receipt["runtime"] = runner(settings=config, seconds=seconds,
                                             outdir=run / "output", device=args.device)
                receipt["status"] = "completed"
            except Exception as error:
                receipt.update(status="simulation failed", error=str(error))
                raise
            finally:
                receipt["runtime_wall_seconds"] = perf_counter() - begin
                (run / "demo.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        else:
            (run / "demo.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(receipt, sort_keys=True))
        return 0
    except (OSError, RuntimeError, ValueError, argparse.ArgumentTypeError) as error:
        print(f"gpuwm hydro demo: {error}", file=sys.stderr)
        return 2


def add_arguments(parser, *, runner=None):
    parser.add_argument("example", choices=tuple(examples()), help="pinned NextGen example id")
    parser.add_argument("--prepare-only", action="store_true", help="verify and materialize inputs without simulating water")
    parser.add_argument("--outdir", type=Path, default=Path("out/hydro-demo"), help="parent folder for separate example runs")
    parser.add_argument("--data-dir", type=Path, default=Path.home() / ".gpuwm/hydro/demos", help="verified immutable input cache")
    parser.add_argument("--seconds", type=positive_seconds, help="simulation duration in seconds (default: example table)")
    parser.add_argument("--device", type=int, default=0, help="CUDA device ordinal (default: 0)")
    parser.add_argument("--strict-numerics", action="store_true", help="explicit strict proof mode; fast numerics is the default")
    parser.set_defaults(func=demo_main, nextgen_demo_runner=runner)


def register_demo_cli(subparsers, *, runner=None):
    parser = subparsers.add_parser("demo", help="prepare and run a pinned NextGen example")
    add_arguments(parser, runner=runner)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare a pinned NextGen example through Rust asset I/O.")
    add_arguments(parser)
    return demo_main(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
