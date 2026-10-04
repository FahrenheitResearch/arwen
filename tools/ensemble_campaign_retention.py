"""Retire completed campaign field payloads after native scoring, with a ledger.

The original forecast cadence and writer remain unchanged. Only checkpoint
and field files explicitly recorded by a complete scored run can be removed.
Input authorities, manifests, scores and reports remain in place.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import re

# Inside the wheel these are tools.<name>; the bare form is what a
# run by path resolves (`python tools/<this file>`), where tools/ is
# sys.path[0].
try:
    from tools.ensemble_campaign_score import QUANTITIES, instant, product_contract
except ImportError:
    from ensemble_campaign_score import QUANTITIES, instant, product_contract

CHECKPOINT = re.compile(r"gpuwmrst_d[0-9]+_[0-9_-]+(?:__[0-9a-f]+)?\.npz")
HISTORY = re.compile(r"wrfout_d[0-9]+_[0-9_T:-]+(?:__[0-9a-f]+)?(?:\.nc)?")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def pin(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}


def verify(reference):
    path = Path(reference["path"])
    if digest(path) != reference["sha256"]:
        raise ValueError(f"retained campaign authority changed: {path}")
    if "bytes" in reference and path.stat().st_size != reference["bytes"]:
        raise ValueError(f"retained campaign size changed: {path}")
    return path


def read(reference):
    return json.loads(verify(reference).read_bytes())


def contained(path, root):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("checkpoint retirement refuses a symbolic link")
    resolved = path.resolve(strict=True)
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError("checkpoint retirement would leave its owned run directory")
    return resolved


def plan_retention(run_receipt, score_summary, *, owned_root, case_id, recipe,
                   keep_scored_fields=False):
    owner = Path(owned_root).resolve(strict=True)
    run_path = contained(run_receipt, owner)
    run_root = run_path.parent
    if run_root == owner:
        raise ValueError("campaign retention needs a dedicated run subdirectory")
    retained = [pin(run_path), pin(score_summary)]
    run, summary = (read(item) for item in retained)
    window = run.get("execution_window", {})
    if (run.get("schema") != "gpuwm-ensemble-campaign-run.v1" or run.get("status") != "PASS"
            or window.get("purpose") != "calibration" or window.get("calibration_complete") is not True
            or window.get("execution_run_seconds") != window.get("source_window_seconds")):
        raise ValueError("only a complete full-window calibration run can retire checkpoints")
    if (summary.get("schema") != "gpuwm-ensemble-campaign-scores.v1"
            or summary.get("status") != "complete"):
        raise ValueError("checkpoint retention requires completed native observation scores")
    ensemble = read(run["ensemble_manifest"])
    retained.append(dict(run["ensemble_manifest"]))
    forecast_root = contained(run["ensemble_manifest"]["path"], run_root).parent
    order = ensemble.get("member_order")
    if (ensemble.get("status") != "PASS" or not order or len(set(order)) != len(order)
            or ensemble.get("members_completed") != sorted(order)
            or ensemble.get("completed_seconds") != window["execution_run_seconds"]):
        raise ValueError("checkpoint retention needs every original member's completed forecast")

    score_request = read(summary["request"])
    retained.append(dict(summary["request"]))
    cases = [row for row in score_request["cases"] if row["case_id"] == case_id]
    if len(cases) != 1:
        raise ValueError("score request does not identify one completed campaign case")
    case = cases[0]
    arms = [row for row in case["runs"] if row["recipe"] == recipe]
    if len(arms) != 1:
        raise ValueError("score request does not identify one completed recipe arm")
    arm = arms[0]
    archive_ref, product_ref = arm["archive"]["manifest"], arm["production_manifest"]
    archive_path = contained(archive_ref["path"], forecast_root)
    if archive_path != forecast_root / "member-diagnostics" / "manifest.json":
        raise ValueError("native scores used another run's diagnostic archive")
    products = ensemble["products"]
    product_paths = {contained(forecast_root / row["manifest"], forecast_root)
                     for row in products["domain_manifests"]}
    if contained(product_ref["path"], forecast_root) not in product_paths:
        raise ValueError("native scores used another run's production manifest")
    if not keep_scored_fields and len(product_paths) != 1:
        raise ValueError("field retirement needs scoring authority for every forecast domain")
    archive, product = read(archive_ref), read(product_ref)
    retained.extend((dict(archive_ref), dict(product_ref)))
    if (archive.get("coverage", {}).get("status") != "complete"
            or archive.get("unavailable") or products.get("pending_rosters")):
        raise ValueError("native diagnostics must cover the full completed member roster")
    if [row["member_id"] for row in arm["members"]] != order:
        raise ValueError("native scores changed the completed member order")
    clocks = [instant(case["start_time"]) + timedelta(hours=lead) for lead in summary["lead_hours"]]
    product_contract(product, archive, arm["members"], clocks)
    diagnostic_payloads = []
    for row in archive["files"]:
        path = contained(forecast_root / row["path"], forecast_root)
        if not path.is_relative_to(forecast_root / "member-diagnostics") or path.suffix != ".nc":
            raise ValueError("native diagnostic payload is outside its recorded archive")
        reference = {"path": str(path), "sha256": row["sha256"], "bytes": row["bytes"]}
        verify(reference)
        if keep_scored_fields:
            retained.append(reference)
        else:
            diagnostic_payloads.append(reference)

    rows = [row for row in summary["products"] if row["case_id"] == case_id and row["recipe"] == recipe]
    if len(rows) != len(QUANTITIES) or {row["quantity"] for row in rows} != set(QUANTITIES):
        raise ValueError("retention needs every requested scored product")
    for row in rows:
        scored = read(row["receipt"])
        retained.append(dict(row["receipt"]))
        provenance = scored.get("provenance", {})
        if (scored.get("schema") != "gpuwm-ensemble-calibration.receipt.v1"
                or scored.get("scores") != row["scores"]
                or row["scores"]["missing_members"] != 0
                or row["members"] != len(order) or row["scores"]["members"] != len(order)
                or row["scores"]["samples"] <= 0
                or provenance.get("case_id") != case_id or provenance.get("recipe") != recipe
                or provenance.get("quantity") != row["quantity"] or provenance.get("units") != row["units"]
                or provenance.get("diagnostic_manifest") != archive_ref
                or provenance.get("production_manifest") != product_ref
                or provenance.get("members") != arm["members"]
                or provenance.get("lead_hours") != summary["lead_hours"]):
            raise ValueError("scored product lost its exact completed run/member authority")
        for reference in provenance.get("match_receipt", []):
            read(reference)
            retained.append(dict(reference))
        if not provenance.get("match_receipt"):
            raise ValueError("scored product has no native matching receipts")

    candidates, seen = [], set()
    retained_paths = {Path(row["path"]).resolve() for row in retained}
    def candidate(filename, *, kind, under, expected=None):
        path = contained(filename, under)
        if path in seen:
            raise ValueError("ordinary report or product manifest repeats a retirement payload")
        if path in retained_paths:
            raise ValueError("a field payload is still named as retained scoring evidence")
        if expected is not None:
            verify(expected)
        seen.add(path)
        item = pin(path)
        info = path.stat()
        item.update(kind=kind, under=str(under), device=info.st_dev, inode=info.st_ino,
                    mtime_ns=info.st_mtime_ns,
                    last_link_allocated_bytes=(getattr(info, "st_blocks", 0) * 512
                                               if info.st_nlink == 1 else 0))
        candidates.append(item)
    for member in ensemble["member_results"]:
        result = member.get("result", {})
        if result.get("status") != "PASS":
            raise ValueError("checkpoint retirement needs successful ordinary member reports")
        for filename in result.get("restart_contract", {}).get("checkpoints_written", []):
            path = contained(filename, forecast_root / "members")
            if not CHECKPOINT.fullmatch(path.name):
                raise ValueError("ordinary report names another checkpoint file type")
            candidate(path, kind="checkpoint", under=forecast_root / "members")
        if not keep_scored_fields:
            for row in result.get("output", {}).get("files", []):
                path = contained(row["path"], forecast_root / "members")
                if not HISTORY.fullmatch(path.name):
                    raise ValueError("ordinary history report names another file type")
                candidate(path, kind="member-history", under=forecast_root / "members", expected=row)
    if not keep_scored_fields:
        for reference in diagnostic_payloads:
            candidate(reference["path"], kind="member-diagnostic",
                      under=forecast_root / "member-diagnostics", expected=reference)
        product_directory = Path(product_ref["path"]).resolve().parent
        for frame in product["frames"]:
            if frame.get("status") != "complete":
                raise ValueError("an uncompleted product frame cannot be retired")
            for key, suffix, kind in (("products", ".nc", "aggregate-field"),
                                      ("maps", ".png", "aggregate-map")):
                directory = (forecast_root / "maps" / product_directory.name
                             if key == "maps" else product_directory)
                for filename in frame.get(key, []):
                    path = contained(forecast_root / filename, directory)
                    if path.suffix != suffix:
                        raise ValueError("product manifest names an unsupported field payload type")
                    candidate(path, kind=kind, under=directory)
    if not candidates:
        raise ValueError("completed run has no recorded payloads to retire")
    return {"schema": "gpuwm-ensemble-campaign-retention.v1", "status": "planned",
            "owned_root": str(owner), "run_root": str(run_root), "case_id": case_id, "recipe": recipe,
            "retained_authorities": retained, "files": candidates,
            "forecast_root": str(forecast_root), "keep_scored_fields": keep_scored_fields,
            "file_count": len(candidates), "logical_bytes": sum(row["bytes"] for row in candidates),
            "scope": ("completed, natively scored checkpoint payloads only" if keep_scored_fields else
                      "completed, natively scored checkpoint, member diagnostic, history and aggregate payloads")}


def retire(plan, *, audit_directory, apply=False):
    audit = Path(audit_directory).resolve()
    owner = Path(plan["owned_root"])
    if audit == owner or not audit.is_relative_to(owner):
        raise ValueError("retention ledger must stay inside the explicitly owned workspace")
    audit.mkdir(parents=True, exist_ok=False)
    (audit / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    if not apply:
        return {"status": "planned", "files": plan["file_count"], "logical_bytes": plan["logical_bytes"]}
    # Recheck every retained byte and every deletion target before the first
    # mutation. A stale score or rewritten checkpoint causes no deletion.
    for reference in plan["retained_authorities"]:
        verify(reference)
    for reference in plan["files"]:
        forecast = Path(plan["forecast_root"])
        under = Path(reference["under"])
        if not under.is_relative_to(forecast) or under == forecast:
            raise ValueError("retirement payload scope is outside its forecast outputs")
        path = contained(reference["path"], under)
        verify(reference)
        info = path.stat()
        if (info.st_dev, info.st_ino, info.st_mtime_ns) != (
                reference["device"], reference["inode"], reference["mtime_ns"]):
            raise ValueError("checkpoint identity changed before retirement")
    removed = []
    with (audit / "deleted-files.jsonl").open("x", encoding="utf-8") as stream:
        for reference in plan["files"]:
            Path(reference["path"]).unlink()
            removed.append(reference)
            stream.write(json.dumps(reference) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    result = {"schema": plan["schema"], "status": "complete", "deleted_files": len(removed),
              "deleted_logical_bytes": sum(row["bytes"] for row in removed),
              "last_link_allocated_bytes_removed": sum(row["last_link_allocated_bytes"] for row in removed),
              "plan_sha256": digest(audit / "plan.json"), "retained_authorities_rechecked": len(plan["retained_authorities"])}
    (audit / "receipt.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-receipt", required=True, type=Path)
    parser.add_argument("--score-summary", required=True, type=Path)
    parser.add_argument("--owned-root", required=True, type=Path)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--audit-directory", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--keep-scored-fields", action="store_true",
                        help="Retire only checkpoints when field payloads are still needed by an explicit review")
    args = parser.parse_args(argv)
    plan = plan_retention(args.run_receipt, args.score_summary, owned_root=args.owned_root,
                          case_id=args.case_id, recipe=args.recipe, keep_scored_fields=args.keep_scored_fields)
    print(json.dumps(retire(plan, audit_directory=args.audit_directory, apply=args.apply)))


if __name__ == "__main__":
    main()
