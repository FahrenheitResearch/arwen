# SPDX-License-Identifier: Apache-2.0
"""CLI configuration and launch for Rust reach-archive analysis."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import tempfile


def register_cli(commands):
    parser = commands.add_parser("retrospective", help="compare reach flow with a published archive")
    _options(parser)
    parser.set_defaults(func=lambda args: _launch(args, parser))


def _options(parser):
    parser.add_argument("--request", type=Path, help="Complete archive request JSON")
    parser.add_argument("--source", default="nwm-v3-conus")
    parser.add_argument("--source-table", type=Path, default=Path(__file__).with_name("data") / "reach-archives.v1.json")
    parser.add_argument("--start", help="First sample UTC, inclusive; omit with --end to inspect coverage")
    parser.add_argument("--end", help="Last sample UTC, inclusive")
    parser.add_argument("--mapping", type=Path, help="Explicit archive_feature_id/candidate_feature_id mapping rows JSON")
    parser.add_argument("--candidate", type=Path, help="arwen.point-flow.v1 candidate JSON")
    parser.add_argument("--store", help="Local captured archive for offline replay")
    parser.add_argument("--cache", type=Path, help="Capture immutable compressed archive objects")
    parser.add_argument("--workers", type=int, help="Bound Rust task concurrency, default available CPU cores")
    parser.add_argument("--output", type=Path, help="Output directory; repeated runs retain results in a new run subfolder")


def _launch(args, parser):
    from gpuwm.zarr_bridge import resolve_zarr_bin
    if bool(args.start) != bool(args.end):
        parser.error("--start and --end must be supplied together")
    if args.start and not (args.mapping or args.request):
        parser.error("event comparisons need --mapping with explicit archive/candidate identifiers")
    output = args.output or Path("hydro-archive-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output = output.expanduser().resolve()
    try:
        binary = resolve_zarr_bin()
        output.parent.mkdir(parents=True, exist_ok=True)
        if args.request:
            request = args.request.expanduser().resolve()
        else:
            table = json.loads(args.source_table.read_text(encoding="utf-8"))
            if table.get("schema") != "arwen.reach-archives.v1":
                raise ValueError("incompatible source table schema")
            source = dict(table["sources"][args.source])
            if args.store:
                source["store"] = args.store
            document = {"source": source}
            if args.start:
                document.update(start=args.start, end=args.end,
                                reaches=json.loads(args.mapping.read_text(encoding="utf-8")))
            if args.candidate:
                document["candidate"] = os.fspath(args.candidate.expanduser().resolve())
            if args.cache:
                document["cache"] = os.fspath(args.cache.expanduser().resolve())
            if args.workers is not None:
                document["workers"] = args.workers
            # This is request configuration, never decoded flow data.
            requests = output.parent / ".reach-archive-requests"
            requests.mkdir(exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=requests,
                                             prefix="request-", suffix=".json", delete=False) as stream:
                json.dump(document, stream, indent=2)
                stream.write("\n")
                request = Path(stream.name)
        return subprocess.call([os.fspath(binary), "reach-archive", os.fspath(request), os.fspath(output)])
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"hydro archive: {exc}\n")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="gpuwm hydro retrospective")
    _options(parser)
    return _launch(parser.parse_args(argv), parser)


if __name__ == "__main__":
    raise SystemExit(main())
