#!/usr/bin/env python3
"""Seal an atomically published native-HRRR bridge tree and external receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone
import re
import time

from gpuwm.hrrr_forecast import validate_hrrr_source_forecast_hours


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _window_shape(value: str) -> str:
    if re.fullmatch(r"[1-9][0-9]*x[1-9][0-9]*", value) is None:
        raise argparse.ArgumentTypeError(
            "expected window shape must be positive NYxNX")
    return value


def _series_hours(path: Path) -> tuple[int, ...]:
    """Return and validate the exact contiguous horizon named by SERIES."""
    hours: list[int] = []
    for number, raw in enumerate(path.read_text().splitlines(), start=1):
        if not raw.strip():
            continue
        columns = raw.split("\t")
        if len(columns) != 3:
            raise ValueError(
                f"{path}:{number}: expected hour, atmosphere, and soil columns")
        try:
            hour = int(columns[0])
        except ValueError as error:
            raise ValueError(
                f"{path}:{number}: invalid forecast hour {columns[0]!r}") from error
        hours.append(hour)
    try:
        return validate_hrrr_source_forecast_hours(hours)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{path}: expected contiguous source forecast hours: {error}") from error


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--series", type=Path, required=True)
    parser.add_argument("--time-evidence", type=Path, required=True)
    parser.add_argument(
        "--expected-window-shape", type=_window_shape, default="207x207")
    args = parser.parse_args()
    if not args.root.is_dir():
        raise FileNotFoundError(args.root)
    hours = _series_hours(args.series)
    gate = dict(
        line.split("\t", 1)
        for line in (args.root / "gate.txt").read_text().splitlines())
    required = {
        "status": "PASS",
        "forecast_hours": ",".join(map(str, hours)),
        "series_count": str(len(hours)),
        "atmosphere_selected_per_time": "561",
        "soil_selected_per_time": "18",
        "window_shape": args.expected_window_shape,
    }
    for key, expected in required.items():
        if gate.get(key) != expected:
            raise ValueError(f"gate {key}={gate.get(key)!r}, expected {expected!r}")
    for hour in hours:
        for role, expected_files in (("atmosphere", 22), ("soil", 2)):
            directory = args.root / f"{role}-f{hour:02d}"
            files = sorted(directory.glob("*.f32le"))
            if len(files) != expected_files:
                raise ValueError(
                    f"{directory} has {len(files)} payloads, expected {expected_files}")
    manifest = args.root / "SHA256SUMS"
    if manifest.exists():
        raise FileExistsError(f"refusing to overwrite {manifest}")
    payloads = sorted(
        path for path in args.root.rglob("*")
        if path.is_file() and path != manifest)
    lines = [
        f"{_sha256(path)}  ./{path.relative_to(args.root).as_posix()}"
        for path in payloads]
    temporary = manifest.with_suffix(".tmp")
    temporary.write_text("\n".join(lines) + "\n")
    os.replace(temporary, manifest)
    manifest_sha = _sha256(manifest)
    receipt = {
        "schema": "gpuwm-native-hrrr-bridge-seal-v1",
        "status": "PASS",
        "sealed_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(args.root.resolve()),
        "manifest": str(manifest.resolve()),
        "manifest_sha256": manifest_sha,
        "payload_file_count": len(payloads),
        "payload_bytes": sum(path.stat().st_size for path in payloads),
        "source_manifest_sha256": args.source_manifest_sha256,
        "source_forecast_hours": list(hours),
        "model_forcing_hours": list(range(len(hours))),
        "decoder_sha256": _sha256(args.decoder),
        "series_sha256": _sha256(args.series),
        "time_evidence_sha256": _sha256(args.time_evidence),
        "gate": gate,
        "wall_seconds": time.perf_counter() - started,
    }
    if args.receipt.exists():
        raise FileExistsError(f"refusing to overwrite {args.receipt}")
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt_tmp = args.receipt.with_suffix(args.receipt.suffix + ".tmp")
    receipt_tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.replace(receipt_tmp, args.receipt)
    print(json.dumps({
        "status": "PASS",
        "manifest_sha256": manifest_sha,
        "payload_file_count": len(payloads),
        "payload_bytes": receipt["payload_bytes"],
        "receipt": str(args.receipt),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
