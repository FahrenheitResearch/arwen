#!/usr/bin/env python3
"""Read the Noah-MP runtime units' per-thread local frames on THIS card.

  python tools/measure_noahmp_frames.py measure --output receipt.json \\
      [--box "weather-node-1"] [--platform-family linux]
  python tools/measure_noahmp_frames.py verify
  python tools/measure_noahmp_frames.py resolve [--extra gpu-cu13]

``measure`` compiles all fifteen Noah-MP runtime translation units through
the production factory (``gpuwm.core.noahmp_kernel_sources.compile_runtime_unit``)
in a FRESH process with an empty CuPy cache, reads every exported kernel's
function attributes (no launch), writes the reading as JSON and prints the
``ComposedUnitFrameRecording`` row it becomes -- paste that row into
``gpuwm/core/kernel_frame_recordings.py`` (``NOAHMP_COMPOSED_FRAME_RECORDINGS``)
to make this card's compile platform a measured one instead of one priced
from the ceiling over the recorded platforms.

``verify`` takes the same reading and compares it with the row the tree
already holds for this platform: exit 0 when every frame and every unit
identity agree, 1 with the differences otherwise, 2 when this platform has no
row.  It is the driver gate for the composed units, the counterpart of
``tools/vram_reserve_probe.py frames`` for the standalone ones.

``resolve`` is the packaging half of the same gate.  The NVRTC build a
fresh install compiles on is set by the cuda-toolkit release the package's
``[ctk]`` extra resolves to on the day pip runs, so it asks pip (``pip
install --dry-run --report``, a clean resolution against the live index,
nothing installed) what ``gpuwm[gpu-cu13]`` / ``gpuwm[gpu-cu12]`` resolve
to right now and compares the ``nvidia-cuda-nvrtc`` version with the
declaration in ``RESOLVED_TOOLCHAIN_PINS``: exit 0 when the declared build
is what the index resolves, 1 when it moved -- which is the day the table
needs a new row on every architecture the pin lists, or every fresh install
drops from a measured Noah-MP price to the ceiling.  It needs
the network and is run before a cut; the CPU gate in
``tests/test_kernel_frame_recordings.py`` holds the declaration and the rows
in step with each other and with pyproject without it.

What a reading is and is not: a frame is a compile attribute of one
(target architecture, NVRTC build) pair and prices that pair exactly; a
card on any other pair is priced from the element-wise ceiling over the
recorded rows, and plan review says so beside the number.  A reading here is
not a forecast, not numerical parity, and not a statement about any other
card.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import platform as _platform
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpuwm.core.kernel_frame_recordings import (  # noqa: E402
    RESOLVED_TOOLCHAIN_PINS, ResolvedToolchainPin)
from gpuwm.core.noahmp_frame_provenance import (  # noqa: E402
    MEASURE_COMMAND, RECORDINGS_MODULE, compare_with_tree, measure_live,
    render_row)


def _fresh_subprocess(argv: list[str]) -> int:
    """Run ``argv`` (this script) in a child with an empty CuPy cache.

    The parent imports no CuPy and touches no device: the fresh default
    stack limit is part of the reading, and a cache hit would be a compile
    this process never observed.
    """
    with tempfile.TemporaryDirectory(prefix="noahmp-frames-cupy-") as cache:
        env = dict(os.environ, CUPY_CACHE_DIR=cache)
        return subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), *argv],
            env=env, check=False).returncode


def _take_reading() -> dict:
    log = io.StringIO()
    measurement = measure_live(log=log)
    measurement["nvrtc_log"] = log.getvalue()
    measurement["created_utc"] = datetime.now(timezone.utc).isoformat()
    measurement["host"] = _platform.node()
    measurement["os"] = _platform.platform()
    return measurement


def measure_worker(out: Path, *, box: str, platform_family: str) -> int:
    measurement = _take_reading()
    row = render_row(measurement, box=box, platform_family=platform_family,
                     measured=measurement["created_utc"][:10])
    measurement["row"] = row
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=out.parent,
                                     prefix=out.name, suffix=".tmp",
                                     delete=False) as handle:
        temp = Path(handle.name)
        json.dump(measurement, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temp, out)
    differences = compare_with_tree(measurement)
    print(f"# reading written to {out}", file=sys.stderr)
    if differences:
        print(f"# {RECORDINGS_MODULE} does not hold this reading:", file=sys.stderr)
        for line in differences:
            print(f"#   {line}", file=sys.stderr)
        print("# the row this reading becomes:", file=sys.stderr)
    else:
        print(f"# {RECORDINGS_MODULE} already holds exactly this reading",
              file=sys.stderr)
    print(row)
    return 0


def verify_worker() -> int:
    measurement = _take_reading()
    differences = compare_with_tree(measurement)
    platform = measurement["platform"]
    where = (f"sm_{platform['device_compute_capability']} / NVRTC "
             f"{platform['nvrtc_build']} ({measurement['device']['name']})")
    if not differences:
        print(f"{RECORDINGS_MODULE}: the row for {where} is exactly what this "
              "card compiles to")
        return 0
    if differences[0].startswith("no row"):
        print(f"{RECORDINGS_MODULE}: {differences[0]}; take one with "
              f"`{sys.argv[0]} measure --output <receipt.json>`", file=sys.stderr)
        return 2
    print(f"{RECORDINGS_MODULE}: the row for {where} has gone stale:",
          file=sys.stderr)
    for line in differences:
        print(f"  {line}", file=sys.stderr)
    return 1


def resolve_requirement(requirement: str, *,
                        python: str = sys.executable) -> dict[str, str]:
    """Distribution name (lower-case) -> version pip would install today.

    A clean resolution of ``requirement`` alone against the configured
    index (``--ignore-installed``: what a FRESH environment gets, not what
    this one already holds), with nothing downloaded or installed.
    """
    with tempfile.TemporaryDirectory(prefix="noahmp-pin-") as tmp:
        report = Path(tmp) / "report.json"
        proc = subprocess.run(
            [python, "-m", "pip", "install", "--dry-run", "--ignore-installed",
             "--quiet", "--disable-pip-version-check", "--report", str(report),
             requirement],
            capture_output=True, text=True, check=False)
        if proc.returncode != 0 or not report.exists():
            raise RuntimeError(
                f"pip could not resolve {requirement!r}: "
                f"{(proc.stderr or proc.stdout).strip()}")
        data = json.loads(report.read_text(encoding="utf-8"))
    return {item["metadata"]["name"].lower(): item["metadata"]["version"]
            for item in data.get("install", ())}


def compare_resolution(pin: ResolvedToolchainPin,
                       resolved: dict[str, str]) -> list[str]:
    """What moved between the declared pin and a live resolution.

    The NVRTC distribution's version IS the compile platform's build
    string (the wheel ``nvidia-cuda-nvrtc 13.4.59`` is the library whose
    banner reads 13.4.59), so it is compared exactly; a fourth component
    would be a different build and is reported as one.
    """
    differences: list[str] = []
    nvrtc = resolved.get(pin.nvrtc_distribution.lower())
    if nvrtc is None:
        differences.append(
            f"{pin.requirement} no longer installs {pin.nvrtc_distribution}")
    elif nvrtc != pin.nvrtc_build:
        differences.append(
            f"{pin.nvrtc_distribution}: declared {pin.nvrtc_build}, the index "
            f"resolves {nvrtc}")
    toolkit = resolved.get("cuda-toolkit")
    if toolkit != pin.cuda_toolkit:
        differences.append(
            f"cuda-toolkit: declared {pin.cuda_toolkit}, the index resolves "
            f"{toolkit}")
    return differences


def resolve_main(extras: list[str] | None) -> int:
    pins = [pin for pin in RESOLVED_TOOLCHAIN_PINS
            if pin.current and (not extras or pin.extra in extras)]
    if not pins:
        raise ValueError(
            f"no current pin declared for {extras}; the declared extras are "
            f"{sorted({pin.extra for pin in RESOLVED_TOOLCHAIN_PINS if pin.current})}")
    moved = False
    for pin in pins:
        resolved = resolve_requirement(pin.requirement)
        nvrtc = resolved.get(pin.nvrtc_distribution.lower())
        print(f"{pin.extra}: {pin.requirement} -> cuda-toolkit "
              f"{resolved.get('cuda-toolkit')}, {pin.nvrtc_distribution} {nvrtc} "
              f"(declared {pin.cuda_toolkit} / NVRTC {pin.nvrtc_build}, "
              f"resolved {pin.resolved})")
        differences = compare_resolution(pin, resolved)
        if not differences:
            continue
        moved = True
        for line in differences:
            print(f"  {line}", file=sys.stderr)
        if pin.noahmp_architectures:
            archs = ", ".join(f"sm_{arch}" for arch in pin.noahmp_architectures)
            print(f"  a fresh `pip install gpuwm[{pin.extra}]` now compiles on a "
                  f"platform with no Noah-MP row for {archs}: every "
                  "sf_surface_physics = 4 run on such an install is priced from "
                  "the ceiling over the recorded platforms instead of from a "
                  "reading of its own, and the docs that name these cards as "
                  f"measured are wrong until it has one.  Take the row with "
                  f"`{MEASURE_COMMAND}` inside an environment resolved today on "
                  "each of those cards, add it to NOAHMP_COMPOSED_FRAME_RECORDINGS, "
                  "and re-declare the pin in RESOLVED_TOOLCHAIN_PINS (keep the "
                  "old one, current=False).", file=sys.stderr)
        else:
            print("  re-declare the pin in RESOLVED_TOOLCHAIN_PINS; this extra "
                  "lists no measured architecture, so no row is owed and every "
                  "card on it is priced from the ceiling.", file=sys.stderr)
    return 1 if moved else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    measure = sub.add_parser("measure", help="take this card's reading and print its row")
    measure.add_argument("--output", required=True, type=Path)
    measure.add_argument("--box", default=_platform.node() or "this machine",
                         help="how the row names the machine it was read on")
    measure.add_argument("--platform-family", default=(
        "windows" if sys.platform == "win32" else "linux"),
        choices=("windows", "linux"))
    measure.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    verify = sub.add_parser("verify", help="compare the tree's row with this card")
    verify.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    resolve = sub.add_parser(
        "resolve", help="compare the declared NVRTC pins with what the index "
                        "resolves for the GPU extras today (network)")
    resolve.add_argument("--extra", action="append", default=None,
                         help="limit to this extra (repeatable); default: every "
                              "current pin")
    args = parser.parse_args(argv)
    try:
        if args.command == "resolve":
            return resolve_main(args.extra)
        if args.command == "verify":
            if not args.worker:
                return _fresh_subprocess(["verify", "--worker"])
            return verify_worker()
        out = args.output.resolve()
        if out.exists():
            raise FileExistsError(
                f"{out} exists; a reading is never overwritten -- choose a new "
                "path so two readings cannot be confused")
        if not args.worker:
            return _fresh_subprocess([
                "measure", "--worker", "--output", str(out), "--box", args.box,
                "--platform-family", args.platform_family])
        return measure_worker(out, box=args.box,
                              platform_family=args.platform_family)
    except Exception as error:  # noqa: BLE001 -- the reason IS the output
        print(f"Noah-MP frame reading refused: {type(error).__name__}: {error}",
              file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
