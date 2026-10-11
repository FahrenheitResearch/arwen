"""Source-pinned model configurations, read from packaged metadata.

The registry distinguishes a configuration specification from a qualified
runtime. Export never substitutes schemes or disables requested switches.
Unimplemented requirements refuse a launch before allocating a device.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
from urllib.parse import urlparse
from urllib.request import urlopen

REGISTRY = Path(__file__).with_name("data") / "model_presets" / "registry.v1.json"
SCHEMA = "gpuwm-model-presets-v1"
#: A declared divergence is physics that runs its own way rather than the
#: pinned source (the 2026-08-20 ruling: physics parity is retired, the
#: verification of record is observation skill), or the hex dycore that
#: ruling keeps pinned to MPAS-A v8.4.1 instead of a fork's own MPAS.  It is reported by every
#: verb that reports blockers, and never blocks a launch: what keeps it
#: in view is that it is named, not that it refuses.
DIVERGENCE_KEYS = ("id", "component", "pinned", "runs", "reason", "follow_up")


def load_registry() -> dict:
    document = json.loads(REGISTRY.read_text(encoding="utf-8"))
    if document.get("schema") != SCHEMA:
        raise ValueError("model preset registry schema differs")
    names = set()
    for row in document["presets"]:
        if row["id"] in names:
            raise ValueError("duplicate model preset identifier")
        names.add(row["id"])
        heights = row["vertical"]["interfaces_m"]
        if (len(heights) != row["vertical"]["layers"] + 1
                or heights[0] != 0
                or any(a >= b for a, b in zip(heights, heights[1:]))
                or heights[-1] != row["vertical"]["top_m"]):
            raise ValueError("model preset vertical interfaces disagree with layer count or top")
        for requirement in row["requirements"]:
            if type(requirement["implemented"]) is not bool or not requirement["breakage"]:
                raise ValueError("every model preset requirement needs an implementation flag and breakage")
        if not row["requirements"]:
            raise ValueError("a model preset must declare its runtime requirements")
        identifiers = [requirement["id"] for requirement in row["requirements"]]
        for divergence in row.get("declared_divergences", []):
            if set(divergence) != set(DIVERGENCE_KEYS) or not all(
                    isinstance(divergence[key], str) and divergence[key]
                    for key in DIVERGENCE_KEYS):
                raise ValueError(
                    "every declared divergence names its component, the pinned source, "
                    "what runs instead, why and the follow-up")
            identifiers.append(divergence["id"])
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("model preset requirement and divergence identifiers collide")
    return document


def preset(name: str) -> dict:
    for row in load_registry()["presets"]:
        if row["id"] == name:
            return row
    raise ValueError(f"unknown model preset {name!r}")


def launch_blockers(row: dict) -> list[dict]:
    return [requirement for requirement in row["requirements"]
            if not requirement["implemented"]]


def declared_divergences(row: dict) -> list[dict]:
    return list(row.get("declared_divergences", []))


def verify_sources(row: dict) -> list[dict]:
    """Verify pinned public source bytes, not forecast or implementation parity."""
    import hashlib
    results = []
    for source in row["sources"]:
        parsed = urlparse(source["url"])
        parts = parsed.path.strip("/").split("/")
        if (parsed.scheme != "https" or parsed.netloc != "github.com"
                or len(parts) < 5 or parts[2] != "blob"
                or re.fullmatch(r"[0-9a-f]{40}", parts[3]) is None
                or parsed.query or parsed.fragment):
            raise ValueError("model preset source must name a pinned public GitHub blob")
        url = "https://raw.githubusercontent.com/" + "/".join(parts[:2] + parts[3:])
        try:
            with urlopen(url, timeout=30) as stream:
                payload = stream.read(2_000_001)
            if len(payload) > 2_000_000:
                raise ValueError("model preset source exceeds the metadata source size limit")
            actual = hashlib.sha256(payload).hexdigest()
            results.append({"url": source["url"], "sha256": actual,
                            "matches": actual == source["sha256"]})
        except OSError as error:
            results.append({"url": source["url"], "matches": False,
                            "error": str(error)})
    return results


def _fortran(value) -> str:
    if type(value) is bool:
        return ".true." if value else ".false."
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        import math
        if not math.isfinite(value):
            raise ValueError("non-finite model preset namelist value")
        return repr(value)
    if isinstance(value, str) and not any(c in value for c in "\r\n\x00"):
        return "'" + value.replace("'", "''") + "'"
    raise ValueError("unsupported model preset namelist value")


def namelist(row: dict, *, start: str, seconds: int) -> str:
    """Emit the requested native configuration without changing its physics."""
    try:
        parsed = datetime.strptime(start, "%Y-%m-%d_%H:%M:%S")
    except ValueError as error:
        raise ValueError("start must be UTC YYYY-MM-DD_HH:MM:SS") from error
    if parsed.strftime("%Y-%m-%d_%H:%M:%S") != start:
        raise ValueError("start must use zero-padded UTC fields")
    dt = row["namelist"]["nhyd_model"]["config_dt"]
    if type(seconds) is not int or seconds <= 0 or seconds % dt:
        raise ValueError(f"seconds must be a positive integer multiple of timestep {dt}")
    days, rest = divmod(seconds, 86400)
    hours, rest = divmod(rest, 3600)
    minutes, rest = divmod(rest, 60)
    duration = f"{days:03d}_{hours:02d}:{minutes:02d}:{rest:02d}"
    identifier = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")
    lines = []
    for group, settings in row["namelist"].items():
        if not identifier.fullmatch(group):
            raise ValueError("invalid model preset namelist group")
        settings = dict(settings)
        if group == "nhyd_model":
            settings.update(config_start_time=start, config_run_duration=duration)
        lines.append("&" + group)
        for key, value in settings.items():
            if not identifier.fullmatch(key):
                raise ValueError("invalid model preset namelist key")
            lines.append(f"    {key} = {_fortran(value)}")
        lines.append("/")
    return "\n".join(lines) + "\n"


def main(args: argparse.Namespace) -> int:
    if args.preset_action == "list":
        print(json.dumps({"schema": SCHEMA, "presets": [
            {"id": row["id"], "status": row["status"],
             "launchable": not launch_blockers(row)}
            for row in load_registry()["presets"]]}, indent=2))
        return 0
    row = preset(args.name)
    if args.preset_action == "run":
        blockers = launch_blockers(row)
        if blockers and not args.experimental:
            print(json.dumps({"preset": row["id"], "launchable": False,
                              "blockers": blockers,
                              "declared_divergences": declared_divergences(row)},
                             indent=2))
            return 3
        command = launch_command(row, args)
        import subprocess
        return subprocess.call(command)
    if args.preset_action == "show":
        print(json.dumps(row, indent=2))
        return 0
    if args.preset_action == "verify-sources":
        results = verify_sources(row)
        print(json.dumps({"preset": row["id"], "sources": results}, indent=2))
        return 0 if all(item["matches"] for item in results) else 3
    if args.preset_action == "check":
        blockers = launch_blockers(row)
        print(json.dumps({"preset": row["id"], "status": row["status"],
                          "launchable": not blockers, "blockers": blockers,
                          "declared_divergences": declared_divergences(row)}, indent=2))
        return 3 if blockers else 0
    text = namelist(row, start=args.start, seconds=args.seconds)
    if args.out is None:
        print(text, end="")
    else:
        # Never truncate an existing configuration through an export command.
        with args.out.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
    return 0


def register_cli(subparsers) -> None:
    command = subparsers.add_parser(
        "preset", help="inspect and export source-pinned model configurations")
    verbs = command.add_subparsers(dest="preset_action", required=True)
    listing = verbs.add_parser("list", help="list specifications and launch status")
    listing.set_defaults(func=main)
    for action in ("show", "check", "verify-sources"):
        verb = verbs.add_parser(action, help={
            "show": "show settings, sources and requirements",
            "check": "report unmet runtime requirements without allocating a GPU",
            "verify-sources": "verify the pinned public metadata source hashes"}[action])
        verb.add_argument("name")
        verb.set_defaults(func=main)
    export = verbs.add_parser("export", help="emit native namelist without substitutions")
    export.add_argument("name")
    export.add_argument("--start", required=True, help="UTC YYYY-MM-DD_HH:MM:SS")
    export.add_argument("--seconds", required=True, type=int)
    export.add_argument("--out", type=Path)
    export.set_defaults(func=main)
    run = verbs.add_parser("run", help="run a table-defined configuration through the hex forecast door")
    run.add_argument("name")
    run.add_argument("--experimental", action="store_true",
                     help="run despite unmet requirements; a preset with none launches without it")
    run.add_argument("--mesh", required=True)
    for name in ("grid", "static", "init", "lbc-dir", "invariant", "out"):
        run.add_argument("--" + name, type=Path, required=True)
    run.add_argument("--start-time", required=True)
    run.add_argument("--init-source", required=True)
    run.add_argument("--hours", type=float, required=True)
    run.add_argument("--history-every-minutes", type=int, default=60)
    run.add_argument("--card", required=True)
    run.add_argument("--fingerprint-every", type=int,
                     help="full-field hash cadence in steps; defaults to each history endpoint")
    run.set_defaults(func=main)


def launch_command(row: dict, args: argparse.Namespace) -> list[str]:
    """Translate one metadata arm into the existing measured forecast door.

    The experimental arm carries all source refusals in its measurement receipt.
    It never changes the native specification or registers an earned anchor.
    """
    import math
    import sys
    runtime = row.get("runtime")
    if runtime is None or args.mesh not in runtime["meshes"]:
        raise ValueError("model preset has no runtime row for the requested mesh")
    settings = runtime["meshes"][args.mesh]
    dt = settings["dt_seconds"]
    seconds = args.hours * 3600
    if (not math.isfinite(seconds) or seconds <= 0
            or abs(seconds / dt - round(seconds / dt)) > 1e-9):
        raise ValueError("hours must contain a positive whole number of model steps")
    datetime.strptime(args.start_time, "%Y-%m-%d_%H:%M:%S")
    root = Path(__file__).resolve().parents[1]
    tool = root / "tree" / "tools" / "check_samehost_forecast_v841.py"
    if not tool.is_file():
        raise ValueError("model preset hex forecast runner is absent from this checkout")
    fingerprint = getattr(args, "fingerprint_every", None)
    if fingerprint is None:
        fingerprint = int(round(args.history_every_minutes * 60 / dt))
    if fingerprint < 1:
        raise ValueError("fingerprint cadence must be positive to retain matched endpoint hashes")
    command = [sys.executable, str(tool), "--measurement-mint", "--dt", str(dt),
               "--card", args.card, "--fingerprint-every", str(fingerprint), "--report",
               str(args.out.with_name(args.out.name + ".measurement.json")), "--", "forecast",
               "--mesh", args.mesh, "--physics-suite", runtime["physics_suite"],
               "--gpuwm-checkout", str(root), "--scratch",
               str(args.out.with_name(args.out.name + "-scratch")), "--stop-on-refusal"]
    for flag in ("grid", "static", "init", "lbc_dir", "invariant", "out",
                 "start_time", "init_source", "hours", "history_every_minutes"):
        command.extend(["--" + flag.replace("_", "-"), str(getattr(args, flag))])
    for flag, value in {**runtime["forecast_options"], **settings["forecast_options"]}.items():
        if isinstance(value, bool):
            if value:
                command.append("--" + flag)
        else:
            command.extend(["--" + flag, str(value)])
    return command
