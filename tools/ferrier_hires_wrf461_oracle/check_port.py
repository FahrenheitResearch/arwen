#!/usr/bin/env python3
"""Mechanical Ferrier release check. Missing evidence fails the check."""
from __future__ import annotations
import argparse
import hashlib
import contextlib
import io
import json
import shutil
import tempfile
from pathlib import Path
from compare_ferrier import compare

SETS = {"lane": 72, "refute": 65, "coverage": 22}
def check(reference: Path, candidate: Path, forecast: Path | None) -> dict:
    result = {"strict": {}, "default": {}, "forecast": False}
    for mode in ("strict", "default"):
        for name, count in SETS.items():
            ref = reference if name == "lane" else reference / name
            inputs = ref / "columns_inputs.csv"
            out = candidate / mode / name
            if not all(p.is_file() for p in (inputs, ref / "levels.csv",
                    ref / "surface.csv", out / "levels.csv", out / "surface.csv")):
                result[mode][name] = {"pass": False, "reason": "missing outputs"}
                continue
            columns = sum(x.startswith("C,") for x in inputs.read_text().splitlines())
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                equal = compare(ref, out, inputs)
            result[mode][name] = {"pass": equal and columns == count,
                                 "columns": columns, "comparison": captured.getvalue()}
    try:
        from gpuwm.config import MP_PHYSICS_ACCEPTED
        from gpuwm.namelist_import import _MP_MAP
        result["admission"] = 5 in MP_PHYSICS_ACCEPTED and 5 in _MP_MAP
    except Exception as e:
        result["admission"] = False
        result["admission_error"] = str(e)
    if forecast is not None and forecast.is_file():
        try:
            f = json.loads(forecast.read_text())
            report_path, sheet_path, hashes_path = (Path(f[k]) for k in
                ("report_path", "sheet_path", "field_hashes_path"))
            digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            report = json.loads(report_path.read_text())
            hashes = json.loads(hashes_path.read_text())
            domain = report["physics"]["execution_plan"]["executed_d01"]["domain"]
            result["forecast"] = (
                report["status"] == "PASS" and report["executor"]["steps"] > 0
                and report["run_seconds"] > 0 and report["domain"]["dx_m"] == 12000
                and domain["mp_physics"] == 5 and domain["bl_pbl_physics"] == 2
                and domain["sf_sfclay_physics"] == 2
                and report["physics"]["microphysics_update_count"] > 0
                and report["health"]["final"]["ok"] is True
                and digest(report_path) == f["report_sha256"]
                and digest(sheet_path) == f["rust_sheet_sha256"]
                and digest(hashes_path) == f["field_hashes_sha256"]
                and hashes["default_repeat_identical"] is True
                and hashes["runs"]["default-a"]["canonical_state"] == report["final_state_digest"])
        except (OSError, KeyError, ValueError, TypeError) as exc:
            result["forecast_error"] = str(exc)
    result["planted_bit_caught"] = False
    source = candidate / "strict/lane"
    if (source / "levels.csv").is_file() and (source / "surface.csv").is_file():
        with tempfile.TemporaryDirectory(prefix="planted-", dir=candidate) as tmp:
            bad = Path(tmp)
            for filename in ("levels.csv", "surface.csv"):
                shutil.copyfile(source / filename, bad / filename)
            rows = (bad / "levels.csv").read_text().splitlines()
            fields = rows[1].split(",")
            fields[3] = f"{int(fields[3],16)^1:08X}"
            rows[1] = ",".join(fields)
            (bad / "levels.csv").write_text("\n".join(rows)+"\n")
            with contextlib.redirect_stdout(io.StringIO()):
                result["planted_bit_caught"] = not compare(source, bad)
    result["pass"] = (all(r["pass"] for r in result["strict"].values())
                      and result["admission"] and result["forecast"] and result["planted_bit_caught"])
    return result
if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reference", type=Path)
    p.add_argument("candidate", type=Path)
    p.add_argument("--forecast", type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    r = check(a.reference, a.candidate, a.forecast)
    a.out.write_text(json.dumps(r, indent=2) + "\n")
    print("PASS" if r["pass"] else "FAIL", a.out)
    raise SystemExit(0 if r["pass"] else 1)
