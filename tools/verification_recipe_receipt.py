"""Execute the published reproduce commands, and commit what happened.

The reproduce recipe in the verification document is a claim that a reader can
run those commands and get a comparison.  As shipped it was not: the
comparator invocation omitted ``--start-time`` and ``--done-file``, so the
poll loop had no terminating condition and every ``forecast_hour`` cell it
wrote was empty -- which is exactly the metrics CSV certification refuses,
because the acceptance band is keyed by lead.

This tool amends nothing by hand.  It builds a two-frame fixture pair, runs
the *amended* comparator command through the production entry point as a
subprocess, and records the exact argv, the exit code, and the digests of what
came out.  The document is rewritten afterwards, citing this receipt -- never
the other way round.

    python tools/verification_recipe_receipt.py

The ``gpuwm run`` half of the recipe is not executed here: the run of record is
a four-domain six-hour integration, and no fixture stands in for it.  What this
tool does check, through the production parser rather than by reading the help
text, is that every flag the amended line carries is accepted.  The receipt
says which half was executed and which was parsed, in those words.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: Where the receipt lands: certification data, not development scratch.
RECEIPT_DIR = REPO_ROOT / "gpuwm" / "data" / "certification" / "recipe-receipt"

#: Fixture geometry.  Small on purpose -- this receipt is about the commands
#: terminating and writing a lead, not about any metric value.
_NY, _NX, _NZ = 16, 16, 4
_START = datetime(1970, 1, 1, 12, 0, 0)
_FRAME_COUNT = 2
_CADENCE = timedelta(hours=1)

_2D = ("T2", "PSFC", "U10", "V10", "RAINNC")
_3D = ("QVAPOR", "W", "REFL_10CM")


def _write_frame(path: Path, stamp: str, seed: int) -> None:
    import netCDF4
    import numpy as np

    rng = np.random.default_rng(seed)
    with netCDF4.Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("Time", None)
        ds.createDimension("DateStrLen", 19)
        ds.createDimension("south_north", _NY)
        ds.createDimension("west_east", _NX)
        ds.createDimension("bottom_top", _NZ)
        times = ds.createVariable("Times", "S1", ("Time", "DateStrLen"))
        times[0] = np.array(list(stamp.encode("ascii")), dtype="S1")
        for name in _2D:
            variable = ds.createVariable(
                name, "f4", ("Time", "south_north", "west_east"))
            variable[0] = rng.normal(size=(_NY, _NX)).astype("f4")
        for name in _3D:
            variable = ds.createVariable(
                name, "f4", ("Time", "bottom_top", "south_north", "west_east"))
            variable[0] = rng.normal(
                size=(_NZ, _NY, _NX)).astype("f4")


def build_fixture(root: Path) -> tuple[Path, Path, str]:
    """A two-frame matched pair, aged past the comparator's settle window."""
    gpu_dir = root / "gpu"
    cpu_dir = root / "cpu"
    gpu_dir.mkdir(parents=True, exist_ok=True)
    cpu_dir.mkdir(parents=True, exist_ok=True)
    aged = time.time() - 3600.0
    for index in range(_FRAME_COUNT):
        valid = _START + index * _CADENCE
        stamp = valid.strftime("%Y-%m-%d_%H:%M:%S")
        name = f"wrfout_d01_{valid.strftime('%Y-%m-%d_%H_%M_%S')}"
        for directory, seed in ((gpu_dir, 100 + index), (cpu_dir, 200 + index)):
            path = directory / name
            _write_frame(path, stamp, seed)
            os.utime(path, (aged, aged))
    return gpu_dir, cpu_dir, _START.strftime("%Y-%m-%d_%H:%M:%S")


def comparator_argv(gpu_dir: Path, cpu_dir: Path, out_csv: Path,
                    start_time: str, done_file: Path) -> list[str]:
    """The amended comparator invocation, exactly as the document will read."""
    return [
        sys.executable, "tools/matched_wrfout_stream_compare.py",
        "--gpu-dir", str(gpu_dir), "--cpu-dir", str(cpu_dir),
        "--out-csv", str(out_csv), "--exclude-rows", "5",
        "--start-time", start_time, "--done-file", str(done_file),
    ]


def run_flags_accepted(argv: list[str]) -> dict[str, object]:
    """Whether the production parser accepts the amended ``gpuwm run`` line."""
    from gpuwm.cli import build_parser

    parser = build_parser()
    try:
        parsed = parser.parse_args(argv)
    except SystemExit as stop:
        return {"accepted": False, "detail": f"parser exited {stop.code}"}
    return {"accepted": True,
            "directory_input_hash": getattr(parsed, "directory_input_hash",
                                            None)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=None,
                        help="where the fixture is built (a temporary "
                             "directory by default)")
    parser.add_argument("--receipt-dir", type=Path, default=RECEIPT_DIR)
    args = parser.parse_args(argv)

    import tempfile

    with tempfile.TemporaryDirectory() as scratch:
        work = args.work_dir or Path(scratch)
        work.mkdir(parents=True, exist_ok=True)
        gpu_dir, cpu_dir, start_time = build_fixture(work)
        done_file = work / "run.done"
        done_file.write_text("complete\n", encoding="utf-8")
        out_csv = work / "metrics.csv"
        command = comparator_argv(gpu_dir, cpu_dir, out_csv, start_time,
                                  done_file)
        started = time.time()
        completed = subprocess.run(command, cwd=REPO_ROOT,
                                   capture_output=True, text=True,
                                   check=False, timeout=300)
        elapsed = time.time() - started
        csv_text = out_csv.read_text(encoding="utf-8") if out_csv.exists() else ""

    import csv as csv_module
    import io

    rows = list(csv_module.DictReader(io.StringIO(csv_text)))
    leads = [row["forecast_hour"] for row in rows]

    run_line = [
        "run", "configs/<the run-of-record config>.toml",
        "--outdir", "out/rematch", "--directory-input-hash", "content",
    ]
    receipt = {
        "schema": "gpuwm.verification-recipe-receipt/v1",
        "what_this_is": (
            "the amended section-7 reproduce commands, executed through the "
            "production entry point against a synthetic two-frame fixture, "
            "so the document can cite an execution rather than an intention"),
        "comparator": {
            "executed": True,
            "argv": ["python"] + command[1:],
            "exit_code": completed.returncode,
            "elapsed_seconds": round(elapsed, 3),
            "terminated_without_a_kill": completed.returncode == 0,
            "stdout_tail": completed.stdout.strip().splitlines()[-3:],
            "row_count": len(rows),
            "forecast_hour_values": leads,
            "forecast_hour_all_non_empty": bool(leads) and all(
                value.strip() for value in leads),
            "metrics_csv_sha256": hashlib.sha256(
                csv_text.encode("utf-8")).hexdigest(),
        },
        "run": {
            "executed": False,
            "why_not": (
                "the run of record is a four-domain six-hour integration; no "
                "fixture stands in for it, and a receipt that claimed "
                "otherwise would be a fabricated one"),
            "argv": ["gpuwm"] + run_line,
            "flags_accepted_by_the_production_parser": run_flags_accepted(
                run_line),
        },
        "fixture": {
            "frames_per_side": _FRAME_COUNT,
            "grid": {"south_north": _NY, "west_east": _NX,
                     "bottom_top": _NZ},
            "start_time": start_time,
            "cadence_seconds": int(_CADENCE.total_seconds()),
            "generator": "tools/verification_recipe_receipt.py",
        },
    }

    from gpuwm.certify.band import write_certification_json

    args.receipt_dir.mkdir(parents=True, exist_ok=True)
    (args.receipt_dir / "metrics.csv").write_bytes(
        csv_text.replace("\r\n", "\n").encode("utf-8"))
    write_certification_json(args.receipt_dir / "receipt.json", receipt)
    print(f"comparator exit {completed.returncode}, {len(rows)} rows, "
          f"leads {leads}")
    return 0 if completed.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
