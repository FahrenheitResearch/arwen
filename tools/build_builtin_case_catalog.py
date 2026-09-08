"""Merge supplied case catalogs by stable ID into a reproducible data-only ZIP.

The newer document supplies the schema and shared definitions. Its complete
record wins for every duplicate ID; scientific fields are never blended.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import zipfile


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _read(path: Path) -> tuple[dict, dict]:
    raw = path.read_bytes()
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        entries = [row for row in archive.infolist()
                   if not row.is_dir() and Path(row.filename).name == "catalog.json"]
        if len(entries) != 1:
            raise ValueError(f"{path.name} must contain exactly one catalog.json")
        member = entries[0]
        if member.file_size > 128 * 1024 * 1024:
            raise ValueError("Expanded catalog exceeds 128 MiB")
        payload = archive.read(member)
    document = json.loads(payload.decode("utf-8-sig"))
    cases = document.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Each catalog must contain case records")
    ids = [row.get("id") for row in cases]
    if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("Each input must have unique, nonempty case IDs")
    return document, {"filename": path.name, "bytes": len(raw),
                      "sha256": hashlib.sha256(raw).hexdigest(),
                      "catalog_member": member.filename, "catalog_bytes": len(payload),
                      "catalog_sha256": hashlib.sha256(payload).hexdigest(),
                      "schema": document.get("schema"), "title": document.get("title"),
                      "version": document.get("catalog_version"), "case_count": len(cases)}


def build_catalog(older: Path, newer: Path) -> tuple[bytes, dict]:
    previous, previous_source = _read(older)
    current, current_source = _read(newer)
    old = {row["id"]: row for row in previous["cases"]}
    new = {row["id"]: row for row in current["cases"]}
    # Old-only records cannot be copied from a different schema without an
    # explicit conversion. Refuse instead of guessing at field compatibility.
    missing = sorted(old.keys() - new.keys())
    if missing:
        raise ValueError(f"The newer catalog omits earlier case IDs: {missing}")
    if current.get("schema") != "arwen.case-catalog/v2":
        raise ValueError("The bundled historical catalog requires the newer v2 schema")
    merged = deepcopy(current)
    policy = "Deduplicate by case ID; the newer complete record wins. No scientific fields are blended."
    provenance = {
        "schema": "arwen.case-catalog-merge.v1", "policy": policy,
        "bundle_scope": "Catalog records and merge provenance only; authoring, browser, README and schema sidecars remain in the original input archives.",
        "inputs": [previous_source, current_source],
        "older_case_ids_retained": sorted(old), "older_case_ids_missing": missing,
        "new_case_ids_added": sorted(new.keys() - old.keys()),
        "duplicate_case_ids": sorted(old.keys() & new.keys()),
        "case_count": len(new),
        "category_counts": dict(sorted(Counter(row["category"] for row in new.values()).items())),
        "selected_records": [{"id": row["id"], "source": current_source["filename"],
                              "sha256": hashlib.sha256(_json_bytes(row)).hexdigest()}
                             for row in current["cases"]],
    }
    merged.setdefault("metadata", {})["bundled_merge"] = {
        "policy": policy, "inputs": [previous_source, current_source],
        "older_cases_retained": len(old), "new_cases_added": len(new.keys() - old.keys()),
        "total_cases": len(new), "provenance_member": "merge-provenance.json",
    }
    payload = _json_bytes(merged)
    provenance["merged_catalog_sha256"] = hashlib.sha256(payload).hexdigest()
    result = io.BytesIO()
    with zipfile.ZipFile(result, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, content in (("catalog.json", payload),
                              ("merge-provenance.json", _json_bytes(provenance))):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, content, compresslevel=9)
    return result.getvalue(), provenance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--older", type=Path, required=True)
    parser.add_argument("--newer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="compare regenerated bytes without writing")
    args = parser.parse_args()
    payload, provenance = build_catalog(args.older, args.newer)
    if args.check:
        if args.output.read_bytes() != payload:
            raise ValueError("Bundled catalog differs from its reproducible input merge")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("xb") as stream:
            stream.write(payload)
    print(json.dumps({"output": str(args.output), "bytes": len(payload),
                      "sha256": hashlib.sha256(payload).hexdigest(),
                      "case_count": provenance["case_count"],
                      "older_cases_retained": len(provenance["older_case_ids_retained"]),
                      "new_cases_added": len(provenance["new_case_ids_added"]),
                      "checked": args.check}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
