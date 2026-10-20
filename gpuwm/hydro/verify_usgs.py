"""NextGen gauge verification CLI (worldwide gauge rows). All decode, matching and numerical work runs in Rust."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import shutil
import subprocess


def _binary() -> str:
    from gpuwm import bridges
    configured = os.environ.get("GPUWM_NEXTGEN_VERIFY")
    if configured:
        if not Path(configured).is_file():
            raise FileNotFoundError(f"GPUWM_NEXTGEN_VERIFY names a missing executable: {configured}")
        return str(Path(configured).resolve())
    name = bridges.executable_name("rw_nextgen_verify")
    checkout = Path(__file__).resolve().parents[2] / "tools" / "nextgen_verify"
    for candidate in (checkout / "target" / "release" / name, checkout / "target" / "debug" / name):
        if candidate.is_file():
            return str(bridges.accept_resolved(candidate.resolve()))
    installed = bridges.find_artifact("GPUWM_NEXTGEN_VERIFY", name)
    if installed is None and bridges.fetch_on_demand("rw_nextgen_verify"):
        # On demand since 2.9.3 (not in the platform wheel): staged from this release's bundle.
        installed = bridges.find_artifact("GPUWM_NEXTGEN_VERIFY", name)
    if installed:
        return str(installed)
    found = shutil.which("rw_nextgen_verify")
    if found:
        return found
    raise RuntimeError("The NextGen gauge verification executable is unavailable. Build tools/nextgen_verify and set GPUWM_NEXTGEN_VERIFY to its executable.")


def _execute(args: argparse.Namespace) -> int:
    command = args.verify_command
    table = str(getattr(args, "table", ""))
    if command == "fetch":
        tail = [table, args.source, args.start, args.end, args.out]
    elif command == "import":
        tail = [table, args.layout, args.files, args.out]
    elif command == "score":
        tail = [args.model, args.reference, args.observations, args.out]
    elif command == "compare":
        tail = [args.prediction, args.label, args.observations, args.out]
    elif command == "replay":
        tail = [table, args.source, args.station, args.raw, args.expected, args.out]
    elif command == "stations":
        tail = [args.attributes, args.records, args.station_field, args.feature_field, args.out]
    elif command == "match":
        tail = [table, args.fabric, args.stations, args.out]
    elif command == "match-check":
        tail = [args.fabric, args.matches, args.out]
    elif command == "discover":
        tail = [table, args.source, args.bbox, args.out]
    elif command == "describe":
        tail = [table, args.source, args.out]
    else:
        tail = [args.observations, args.out]
    return subprocess.run([_binary(), command, *map(str, tail)], check=False).returncode


def _arguments(parser: argparse.ArgumentParser) -> None:
    children = parser.add_subparsers(dest="verify_command", required=True)
    for name in ("fetch", "import", "compare", "score", "replay", "observations", "stations", "match", "match-check", "discover", "describe"):
        p = children.add_parser(name)
        p.add_argument("--out", required=True)
        if name in ("fetch", "import", "replay", "match", "discover", "describe"):
            p.add_argument("--table", type=Path, default=Path(__file__).with_name("usgs_table.json"))
        if name in ("fetch", "replay", "discover", "describe"):
            p.add_argument("--source", default="usgs-continuous")
        fields = {"fetch": ("start", "end"), "import": ("layout", "files"), "compare": ("prediction", "label", "observations"), "score": ("model", "reference", "observations"), "replay": ("station", "raw", "expected"), "observations": ("observations",), "stations": ("attributes", "records", "station-field", "feature-field"), "match": ("fabric", "stations"), "match-check": ("fabric", "matches"), "discover": ("bbox",), "describe": ()}[name]
        for field in fields:
            p.add_argument("--" + field, required=True)
        p.set_defaults(func=_execute)


def register_cli(subparsers) -> None:
    _arguments(subparsers.add_parser("verify-usgs", help="Fetch gauges worldwide (USGS, PEGELONLINE, ECCC/HYDAT, BOM), match them to fabric reaches and verify streamflow"))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _arguments(parser)
    return _execute(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
