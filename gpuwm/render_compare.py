"""``gpuwm render --compare REFERENCE``: a run beside a reference model.

One sheet per product and valid time: the run's own product on the left,
the reference model's own field for the same cycle and lead on the right,
and -- for the continuous fields -- the run minus the reference in a third
panel.  Same grid, same projection and extent, same colour scale.

This module orchestrates and nothing else.  Every number and every pixel
is ``rw_compare``'s (``tools/rustwx/crates/rw-wrfbatch/src/bin/compare.rs``):
the wrfout import, the GRIB2 decode of the reference's messages, the
fetch from the public bucket, the grid match, the difference, the panels
and their composition into a sheet are all the Rust engine's.  What is
decided here is which files are one run's frames, which earlier frame an
hourly accumulation needs beside it, and where the sheets go.

The vocabulary -- which references exist, which products each sheet
covers -- is the engine's too, asked with ``--list-products`` rather than
transcribed, so a reference or product added to the engine's tables is
reachable from this door without a line changing here.
"""

from __future__ import annotations

import argparse
import datetime
import re
import subprocess
import sys
from pathlib import Path

#: The brand a comparison sheet puts over the run's panel.
RUN_LABEL = "WOOF"

#: ``wrfout_d01_2026-10-03_06:00:00`` and ``wrfout_d01_2026-10-03_06_00_00``.
_WRFOUT_NAME = re.compile(
    r"^wrfout_(?P<domain>d\d{2})_"
    r"(?P<stamp>\d{4}-\d{2}-\d{2}_\d{2}[:_]\d{2}[:_]\d{2})")

#: Where a run folder keeps its history frames, tried in order.
_RUN_FOLDER_FRAMES = ("", "wrfout", "out/wrfout", "forecast/wrfout")


def frame_identity(path: Path):
    """``(domain, valid time)`` from a history frame's name, or ``None``."""

    match = _WRFOUT_NAME.match(path.name)
    if match is None:
        return None
    digits = re.sub(r"\D", "", match.group("stamp"))
    try:
        valid = datetime.datetime.strptime(digits, "%Y%m%d%H%M%S")
    except ValueError:
        return None
    return match.group("domain"), valid


def expand_inputs(inputs) -> list[Path]:
    """Files stay files; a run folder becomes its history frames.

    A folder is searched where a run keeps its frames (itself, then
    ``wrfout/``, ``out/wrfout/``, ``forecast/wrfout/``) and the first of
    those that holds any is taken whole, so a run folder and its
    ``wrfout`` directory name the same frames.
    """

    frames: list[Path] = []
    for item in inputs:
        path = Path(item)
        if not path.is_dir():
            frames.append(path)
            continue
        found: list[Path] = []
        for relative in _RUN_FOLDER_FRAMES:
            folder = path / relative if relative else path
            if not folder.is_dir():
                continue
            found = sorted(
                entry for entry in folder.iterdir()
                if entry.is_file() and frame_identity(entry) is not None)
            if found:
                break
        if not found:
            raise ValueError(
                f"{path} holds no wrfout_dNN_YYYY-MM-DD_HH:MM:SS history "
                "frames (looked in it and in its wrfout, out/wrfout and "
                "forecast/wrfout folders)")
        frames.extend(found)
    return frames


def group_frames(frames) -> list[tuple[list[Path], list[Path]]]:
    """``[(frames to compare, context frames)]``, one entry per run nest.

    Frames of one folder and one domain are one run's time series and go
    to the engine together.  Each compared frame's hourly accumulation
    needs the same run's frame one hour earlier; when that frame exists
    beside it and was not itself asked for, it rides along as context
    (imported, never drawn).
    """

    groups: dict[tuple[str, str], list[Path]] = {}
    for frame in frames:
        identity = frame_identity(frame)
        domain = identity[0] if identity else ""
        groups.setdefault((str(frame.parent.resolve()), domain), []).append(frame)
    out = []
    for (_, _), members in sorted(groups.items()):
        members = sorted(dict.fromkeys(members))
        named = {member.name for member in members}
        context: list[Path] = []
        for member in members:
            identity = frame_identity(member)
            if identity is None:
                continue
            domain, valid = identity
            earlier = valid - datetime.timedelta(hours=1)
            for separator in (":", "_"):
                name = (f"wrfout_{domain}_{earlier:%Y-%m-%d_%H}"
                        f"{separator}{earlier:%M}{separator}{earlier:%S}")
                candidate = member.with_name(name)
                if (name not in named and candidate.is_file()
                        and candidate not in context):
                    context.append(candidate)
                    break
        out.append((members, context))
    return out


def default_source_label() -> str:
    """``WOOF <executing version>``: the stamp under the run's panel."""

    from gpuwm.provenance import UNKNOWN_VERSION
    from gpuwm.provenance_gate import executing_version

    try:
        version = executing_version()
    except Exception:                                   # noqa: BLE001
        return RUN_LABEL
    if not version or version == UNKNOWN_VERSION:
        return RUN_LABEL
    return f"{RUN_LABEL} {version}"


def default_reference_cache() -> Path:
    """Where fetched reference subsets are kept between renders."""

    return Path.home() / ".gpuwm" / "cache" / "compare-reference"


def reference_catalog(text: str) -> dict[str, dict]:
    """Reference names, labels, and product capabilities from the native table."""

    references: dict[str, dict] = {}
    for line in text.splitlines():
        fields = line.split("\t")
        if fields[0] == "REFERENCE" and len(fields) >= 3:
            references[fields[1]] = {"label": fields[2], "products": {}}
        elif fields[0] == "CAPABILITY" and len(fields) >= 4:
            reference = references.get(fields[1])
            if reference is not None:
                reference["products"][fields[2]] = fields[3]
    return references


def selected_references(text: str, catalog: dict[str, dict]) -> list[str]:
    """Validate and preserve the caller's comma-separated reference order."""

    selected: list[str] = []
    for name in text.split(","):
        name = name.strip()
        if name not in catalog:
            raise ValueError(f"unknown comparison reference {name!r}; this "
                             f"native build knows: {', '.join(catalog)}")
        if name not in selected:
            selected.append(name)
    if not selected:
        raise ValueError("--compare named no references")
    return selected


def engine_command(engine: Path, args: argparse.Namespace, *, store: Path,
                   frames, context, size: tuple[int, int]) -> list[str]:
    """The exact ``rw_compare`` invocation for one run nest."""

    command = [
        str(engine), "--store-root", str(store), "--out-dir", str(args.out),
        "--reference", str(args.compare),
        "--products", str(args.products or "all"),
        "--difference", str(args.compare_difference),
        "--layout", str(args.layout),
        "--width", str(size[0]), "--height", str(size[1]),
        "--source-label", args.source_label or default_source_label(),
        "--run-label", args.compare_label or RUN_LABEL,
        "--reference-cache",
        str(args.compare_cache or default_reference_cache()),
    ]
    if args.compare_reference_dir is not None:
        command += ["--reference-dir", str(args.compare_reference_dir)]
    if args.compare_cycle:
        command += ["--cycle", str(args.compare_cycle)]
    if args.compare_offline:
        command.append("--offline")
    if args.compare_gallery is not None:
        command += ["--flat-dir", str(args.compare_gallery)]
    if getattr(args, "theme", None):
        command += ["--theme", str(args.theme)]
    for frame in context:
        command += ["--context", str(frame)]
    command += [str(frame) for frame in frames]
    return command


def _relay(line: str, tally: dict) -> None:
    """One engine line, said the way this door says things."""

    fields = line.rstrip("\n").split("\t")
    word = fields[0]
    if word == "RENDERED" and len(fields) >= 4:
        tally["rendered"].append(fields[3])
        print(f"render: {fields[3]}")
    elif word == "SKIPPED" and len(fields) >= 4:
        tally["skipped"] += 1
        what = "every product" if fields[1] == "*" else fields[1]
        print(f"render: skipped {what} for {Path(fields[2]).name}: "
              f"{fields[3]}", file=sys.stderr)
    elif word == "FAILED" and len(fields) >= 4:
        tally["failed"] += 1
        what = "every product" if fields[1] == "*" else fields[1]
        print(f"render: FAILED {what} for {Path(fields[2]).name}: "
              f"{fields[3]}", file=sys.stderr)
    elif word == "SOURCE" and len(fields) >= 6:
        if fields[2] == "observed":
            print(f"render: reference {fields[1]} observed {fields[3]} "
                  f"product {fields[4]} from {fields[5]}")
        else:
            print(f"render: reference {fields[1]} cycle {fields[2]} "
                  f"{fields[3]}Z {fields[4]} from {fields[5]}")
    elif word == "MATCH" and len(fields) >= 2:
        print("render: grid match " + " ".join(fields[1:]))
    elif word == "STATS" and len(fields) >= 3:
        print("render: difference " + " ".join(fields[1:]))
    elif word.startswith("FINISHED"):
        return
    elif word == "IMPORT_NOTE":
        return
    elif line.strip() and "Download" not in line:
        print(f"render: {line.rstrip()}", file=sys.stderr)


def compare_main(args: argparse.Namespace) -> int:
    """The ``--compare`` route of ``gpuwm render``."""

    from gpuwm import explain, render, rustwx, rustwx_lanes

    try:
        engine = rustwx_lanes.require_compare_bin()
    except RuntimeError as error:
        print("render: " + explain.render(
            str(error), explain=explain.explain_enabled(args),
            command="gpuwm render"), file=sys.stderr)
        return 2
    if args.list_products:
        listing = subprocess.run(
            [str(engine), "--list-products"], capture_output=True, text=True,
            errors="replace")
        sys.stdout.write(listing.stdout)
        return listing.returncode
    if not args.wrfout:
        print("render: --compare needs at least one WRFOUT frame or run "
              "folder", file=sys.stderr)
        return 2
    try:
        listing = subprocess.run(
            [str(engine), "--list-products"], capture_output=True, text=True,
            errors="replace", env=rustwx.renderer_env())
        if listing.returncode != 0:
            raise ValueError("native comparison catalogue failed: "
                             + (listing.stderr.strip() or str(listing.returncode)))
        catalog = reference_catalog(listing.stdout)
        args.compare = ",".join(selected_references(args.compare, catalog))
        # The reference-model renderer has a fixed-canvas contract.
        # The single-run renderer's automatic canvas keeps that existing
        # comparison default at the reference renderer's native size.
        size = render.parse_size(args.size) or (1200, 900)
        frames = expand_inputs(args.wrfout)
    except ValueError as error:
        print(f"render: {error}", file=sys.stderr)
        return 2
    missing = [frame for frame in frames if not frame.is_file()]
    if missing:
        print("render: no such history frame: "
              + ", ".join(str(frame) for frame in missing), file=sys.stderr)
        return 2

    render._claim_run_dir(args, frames)
    tally = {"rendered": [], "skipped": 0, "failed": 0}
    status = 0
    try:
        for members, context in group_frames(frames):
            with render.scratch_store(args.out) as store:
                command = engine_command(
                    engine, args, store=store, frames=members,
                    context=context, size=size)
                process = subprocess.Popen(
                    command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, errors="replace", env=rustwx.renderer_env())
                assert process.stdout is not None
                for line in process.stdout:
                    _relay(line, tally)
                if process.wait() != 0:
                    status = 1
    finally:
        render._publish_run_dir(args)
    print(f"render: {len(tally['rendered'])} comparison sheet(s) -> "
          f"{args.out}"
          + (f" ({tally['skipped']} skipped)" if tally["skipped"] else "")
          + (f" ({tally['failed']} FAILED)" if tally["failed"] else ""))
    if args.compare_gallery is not None and tally["rendered"]:
        print(f"render: flat copies -> {args.compare_gallery}")
    return status


def register_arguments(parser: argparse.ArgumentParser) -> None:
    """The ``--compare`` family of ``gpuwm render``."""

    parser.add_argument(
        "--compare", metavar="REFERENCE[,REFERENCE...]", default=None,
        help="draw each frame beside ordered native references (for example "
             "hrrr,mrms or hrrr,rrfs,mrms), on one grid and colour scale. "
             "Forecast references share the valid time; observation panels "
             "label their actual observation time. The native --list-products "
             "catalogue lists references and supported products. WRFOUT may "
             "be frames or a run folder")
    parser.add_argument(
        "--compare-difference", choices=("auto", "on", "off"),
        default="auto",
        help="run-minus-reference panels: 'auto' (default) draws continuous "
             "differences with one reference and only field panels with a "
             "reference list; 'on' adds a difference for each reference "
             "whose units have a difference ladder; 'off' never")
    parser.add_argument(
        "--compare-reference-dir", type=Path, default=None, metavar="DIR",
        help="read the reference's GRIB2 files from DIR (by their "
             "published names, flat or under the bucket's own folders) "
             "before fetching anything")
    parser.add_argument(
        "--compare-cache", type=Path, default=None, metavar="DIR",
        help="where fetched reference subsets are kept between renders "
             "(default ~/.gpuwm/cache/compare-reference)")
    parser.add_argument(
        "--compare-cycle", default=None, metavar="YYYYMMDDHH",
        help="compare against THIS reference cycle instead of the run's "
             "own start time")
    parser.add_argument(
        "--compare-offline", action="store_true",
        help="never fetch: use only --compare-reference-dir and the cache")
    parser.add_argument(
        "--compare-gallery", type=Path, default=None, metavar="DIR",
        help="also copy every sheet, flat, into DIR; DIR may equal the output "
             "directory. Multi-reference sheets keep the requested panel order")
    parser.add_argument(
        "--compare-label", default=None, metavar="TEXT",
        help=f"the title over the run's panel (default {RUN_LABEL})")


__all__ = ["RUN_LABEL", "compare_main", "default_reference_cache",
           "default_source_label", "engine_command", "expand_inputs",
           "frame_identity", "group_frames", "register_arguments"]
