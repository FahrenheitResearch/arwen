"""Replay round-3 KF failures and their exact no-cumulus controls.

Run on a sprint box, under the GPU queue unless --preflight-only is set.
Missing runs, controls, raw step-1 U, or oracle words fail the check.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from uuid import uuid4


def control_signature(combo):
    """Compare every recorded setting, not just the six scheme selectors."""
    settings = copy.deepcopy({key: combo.get(key, {}) for key in
                              ("factors", "namelist", "woof_settings", "case_overrides")})
    settings["factors"].pop("cu", None)
    settings["namelist"].get("physics", {}).pop("cu_physics", None)
    settings["woof_settings"].pop("cu_physics", None)
    return json.dumps(settings, sort_keys=True, separators=(",", ":"))


def plan_pairs(rows, combos, available_runs=None):
    controls = {}
    for combo in combos.values():
        if combo.get("factors", {}).get("cu") == 0:
            controls.setdefault(control_signature(combo), []).append(combo["id"])
    available = (set(available_runs) if available_runs is not None else
                 {(row["case"], row["combo"]) for row in rows})
    pairs, errors = [], []
    targets = [row for row in rows if row["cu"] == "1" and row["outcome"] == "FAIL"]
    if not targets:
        errors.append("round-3 selection contains no cu=1 FAIL runs")
    for row in targets:
        combo = combos.get(row["combo"])
        if combo is None:
            errors.append(f"missing combo metadata: {row['combo']}")
            continue
        matches = [cid for cid in controls.get(control_signature(combo), [])
                   if (row["case"], cid) in available]
        if not matches:
            errors.append(f"no recorded identical no-cumulus control: {row['combo']}/{row['case']}")
        else:
            pairs.append({"case": row["case"], "kf": row["combo"], "control": sorted(matches)[0]})
    return pairs, errors


def assert_equal_errors(kf, control):
    import math
    if not math.isfinite(kf) or not math.isfinite(control):
        raise AssertionError("step-1 U error is non-finite")
    if kf != control:
        raise AssertionError(f"KF step-1 U max difference {kf!r} != control {control!r}")


def u_error(recording, case, combo, out):
    import numpy as np
    from gpuwm.verify_exact.recording import load_run_record
    run = recording.runs[(case, combo)]
    ref = run.referee("scalar")
    if ref is None or not ref.designated or not run.referee_sound(ref):
        raise AssertionError(f"no sound designated scalar referee: {combo}/{case}")
    from gpuwm.verify_exact.compare import field_digest
    directory = out / "woof" / case / combo
    document = json.loads((directory / "hashes.json").read_text())
    if document.get("woof", {}).get("status") != "OK":
        raise AssertionError(f"replay did not finish: {combo}/{case}")
    reference = recording.reference_record(ref)
    candidate = load_run_record(directory)
    wrf = reference.raw(1)
    if not wrf or "U" not in wrf or "U" not in candidate.digests.get(1, {}):
        raise AssertionError(f"missing raw step-1 U: {combo}/{case}")
    meta = reference.fields["U"]
    if field_digest(wrf["U"], meta.dtype, meta.shape) != reference.digests.get(1, {}).get("U"):
        raise AssertionError(f"raw referee U does not match its recording: {combo}/{case}")
    if candidate.fields.get("U") != meta:
        raise AssertionError(f"step-1 U metadata mismatch: {combo}/{case}")
    # The replay retains only differing fields. Equal digests prove zero
    # error without inventing an absent candidate array.
    if candidate.digests[1]["U"] == reference.digests[1]["U"]:
        if not np.isfinite(wrf["U"]).all():
            raise AssertionError(f"non-finite step-1 U: {combo}/{case}")
        return 0.0
    woof = candidate.raw(1)
    if not woof or "U" not in woof:
        raise AssertionError(f"missing raw candidate step-1 U: {combo}/{case}")
    if field_digest(woof["U"], meta.dtype, meta.shape) != candidate.digests[1]["U"]:
        raise AssertionError(f"raw candidate U does not match this replay: {combo}/{case}")
    a, b = wrf["U"], woof["U"]
    if a.shape != b.shape or a.dtype != b.dtype:
        raise AssertionError(f"step-1 U shape/dtype mismatch: {combo}/{case}")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise AssertionError(f"non-finite step-1 U: {combo}/{case}")
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def validate_sweep(previous, current, text, causes):
    from tools.combo_round4_summary import measurement, report
    import statistics
    expected = {(r["combo"], r["case"]) for r in previous}
    actual = [(r["combo"], r["case"]) for r in current]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise AssertionError("SWEEP4 must contain every round-3 run exactly once")
    if report(previous, current) not in text:
        raise AssertionError("SWEEP4 tables do not match the supplied replay verdicts")
    for family in ("mp", "pbl", "sfclay", "lsm", "cu", "rad"):
        values = {str(r.get("factors", {}).get(family)) for r in current}
        for value in values:
            sizes = [size for r in current if str(r.get("factors", {}).get(family)) == value
                     and (size := measurement(r)) is not None]
            if sizes and statistics.median(sizes) > 0.005:
                entry = causes.get(f"{family}/{value}", {})
                if not all(entry.get(key) for key in ("cause", "repro_command", "code_site", "evidence")):
                    raise AssertionError(f"above-floor family lacks a named evidenced cause: {family}/{value}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--round3", type=Path, required=True)
    ap.add_argument("--combos-file", type=Path, required=True)
    ap.add_argument("--recordings", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--raw-overlay", type=Path, action="append", default=[])
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--sweep-results", type=Path, nargs="+", default=[])
    ap.add_argument("--sweep-report", type=Path)
    ap.add_argument("--causes", type=Path)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(args.round3.open(encoding="utf-8-sig", newline="")))
    combos = {c["id"]: c for c in json.loads(args.combos_file.read_text(encoding="utf-8-sig"))["combos"]}
    from gpuwm.verify_exact.recording import Recording
    recording = Recording(args.recordings, combos_file=args.combos_file, overlays=args.raw_overlay)
    # Extended combo metadata and recordings may contain freshly recorded
    # controls. The frozen round-3 CSV still defines the 41 KF targets.
    pairs, errors = plan_pairs(rows, combos, available_runs=recording.runs)
    targets = [r for r in rows if r["cu"] == "1" and r["outcome"] == "FAIL"]
    missing = [f"{r['combo']}/{r['case']}" for r in targets
               if (r["case"], r["combo"]) not in recording.runs
               or recording.runs[(r["case"], r["combo"])].inputs is None]
    if missing:
        errors.append(f"recording lacks inputs for {len(missing)} of {len(targets)} KF failures")
    if len(rows) != 647 or len(targets) != 41:
        errors.append("acceptance requires the frozen 647-run table and its 41 KF failures")
    if not args.preflight_only:
        try:
            if not args.sweep_results or args.sweep_report is None or args.causes is None:
                raise AssertionError("acceptance requires SWEEP4, full replay verdicts and named causes")
            current = [json.loads(line) for source in args.sweep_results
                       for line in source.read_text().splitlines() if line.strip()]
            validate_sweep(rows, current, args.sweep_report.read_text(), json.loads(args.causes.read_text()))
        except (AssertionError, OSError, ValueError) as exc:
            errors.append(str(exc))
    receipt = {"schema": "combo-round4-check-v1", "passed": False,
               "round3_sha256": hashlib.sha256(args.round3.read_bytes()).hexdigest(),
               "round3_runs": len(rows), "kf_fail_runs": len(targets), "pairs": pairs,
               "missing_inputs": missing, "errors": errors, "measurements": [], "oracle": None}
    path = args.out / "check.json"
    if not args.preflight_only and not errors:
        replay_out = args.out / ("replays-" + uuid4().hex)
        for pair in pairs:
            cmd = [sys.executable, "-m", "gpuwm.cli", "verify-exact", "--recordings", str(args.recordings),
                   "--combos-file", str(args.combos_file), "--combos", pair["kf"] + "," + pair["control"],
                   "--cases", pair["case"], "--out", str(replay_out)]
            for overlay in args.raw_overlay:
                cmd += ["--raw-overlay", str(overlay)]
            with (args.out / f"{pair['kf']}-{pair['case']}.log").open("w") as log:
                result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
            receipt.setdefault("commands", []).append({"argv": cmd, "returncode": result.returncode})
            try:
                if result.returncode not in (0, 1):
                    raise AssertionError(f"replay CLI failed with exit {result.returncode}")
                kf = u_error(recording, pair["case"], pair["kf"], replay_out)
                control = u_error(recording, pair["case"], pair["control"], replay_out)
                receipt["measurements"].append(dict(pair, kf_max_abs=kf, control_max_abs=control))
                assert_equal_errors(kf, control)
            except (AssertionError, OSError, ValueError) as exc:
                errors.append(str(exc))
        from tools.kf_wrf461_oracle.compare_woof import run
        root = Path(__file__).resolve().parents[1]
        fixture = root / "tests/data/oracles/kf"
        oracle = run(fixture / "columns-structured.bin", fixture / "oracle-structured-O0.out")
        receipt["oracle"] = oracle
        if not oracle["words"] or oracle["mismatched_words"] or oracle["max_ulp"] or oracle["wrf_fatal"]:
            errors.append("KF column oracle did not compare every output at 0 ULP")
        receipt["passed"] = not errors and len(receipt["measurements"]) == len(targets)
    if args.preflight_only:
        receipt["preflight_only"] = True
    path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: receipt[key] for key in ("passed", "kf_fail_runs", "errors")}))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
