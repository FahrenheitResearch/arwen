"""Run the real native bridge and retain small mechanical-check receipts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stock", type=Path, required=True)
    parser.add_argument("--patched", type=Path, required=True)
    parser.add_argument("--checks", default="legacy,clock,n02")
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = root / "MANIFEST.txt"

    def own(path):
        assert path.is_relative_to(root)
        with manifest.open("a") as out:
            out.write(str(path) + "\n")
        return path

    def run(label, binary, data, cycle, hours, crop, aerosol=False, signals=False):
        series = own(root / f"{label}-series.tsv")
        hour_of_day = cycle[11:13]
        series.write_text("# forecast_hour\tatmosphere\tsoil\n" + "".join(
            f"{hour}\t{root / 'data' / data / f'hrrr.t{hour_of_day}z.wrfnatf{hour:02}.grib2'}\t"
            f"{root / 'data' / data / f'hrrr.t{hour_of_day}z.soilf{hour:02}.grib2'}\n"
            for hour in hours))
        output = own(root / f"{label}-bridge")
        command = [str(binary)] + (["--analyzed-aerosol"] if aerosol else [])
        command += ["--series-workers-ready" if signals else "--series-workers", "4", str(series), str(output)]
        if signals:
            command.append(str(own(root / f"{label}-signals")))
        command += [cycle, *map(str, crop)]
        started = time.monotonic()
        result = subprocess.run(command, text=True, capture_output=True)
        log = own(root / f"{label}.log")
        log.write_text(result.stdout + result.stderr)
        receipt = {"command": command, "exit_code": result.returncode,
                   "wall_seconds": time.monotonic() - started,
                   "log": str(log), "stderr": result.stderr,
                   "crop": crop, "cycle": cycle}
        if output.exists():
            gate = dict(line.split("\t", 1) for line in (output / "gate.txt").read_text().splitlines())
            receipt["gate"] = gate
            receipt["fields"] = [{"field": str(p.relative_to(output)), "bytes": p.stat().st_size,
                                  "sha256": digest(p)} for p in sorted(output.rglob("*.f32le"))]
        own(root / f"{label}.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(label, result.returncode, flush=True)
        return receipt

    checks = set(args.checks.split(","))
    if not checks <= {"legacy", "clock", "n02"}:
        parser.error("unknown check")
    if "legacy" in checks:
        stock = run("legacy-stock", args.stock, "legacy", "2017-01-19 00:00:00", [0, 1], [800, 900, 300, 400], aerosol=True)
        patched = run("legacy-patched", args.patched, "legacy", "2017-01-19 00:00:00", [0, 1], [800, 900, 300, 400], aerosol=True)
        left = {v["field"]: v["sha256"] for v in stock.get("fields", [])}
        right = {v["field"]: v["sha256"] for v in patched.get("fields", [])}
        reference = json.loads((Path(__file__).resolve().parents[1] /
            "tests/fixtures/native_bridge_stock_hashes.json").read_text())
        historical = {v["field"]: v["stock_sha256"] for v in reference["fields"]}
        identity = {"volume_count": len(left), "all_equal": len(left) == 52 and left == right,
                    "historical_stock_equal": left == historical,
                    "fields": [{"field": f, "stock_sha256": h, "patched_sha256": right.get(f),
                                "equal": h == right.get(f)} for f, h in left.items()]}
        own(root / "legacy-identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    else:
        identity = json.loads((root / "legacy-identity.json").read_text())
    if "clock" in checks:
        clock = run("clock-broad", args.patched, "clock", "2026-10-06 12:00:00",
                    [0, 1, 2, 3], [360, 1438, 0, 840], aerosol=True)
    else:
        clock = json.loads((root / "clock-broad.json").read_text())
    if "n02" in checks:
        n02 = run("n02", args.patched, "n02", "2019-06-09 16:00:00",
                  [0, 1, 2, 3, 4], [838, 1004, 247, 389], signals=True)
    else:
        n02 = json.loads((root / "n02.json").read_text())
    success = identity["all_equal"] and identity.get("historical_stock_equal", False) and clock["exit_code"] == 0 and (
        n02["exit_code"] == 0 or "masked_inside_crop=" in n02["stderr"])
    own(root / "mechanical-summary.json").write_text(json.dumps({
        "status": "PASS" if success else "FAIL", "legacy_identity": identity["all_equal"],
        "clock_bridge_exit": clock["exit_code"], "n02_exit": n02["exit_code"]}, indent=2) + "\n")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
