"""Durable renderer result metadata; no weather values are read or calculated."""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import uuid

SUMMARY_SCHEMA = "gpuwm.render-summary.v1"
INVOCATION_SCHEMA = "gpuwm.render-invocation.v1"
SUMMARY_FILENAME = "render-summary.json"
_MAX_RECEIPT_BYTES = 64 * 1024 * 1024
_MAX_STATUS_BYTES = 60 * 1024  # headroom within the selected-job 64 KiB envelope
_CATALOG_REQUESTS = {"all", "direct", "derived", "generic", "heavy", "windowed"}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _family(path: Path, root: Path, layout: str) -> str:
    from gpuwm import render_layout
    relative = path.relative_to(root)
    if layout == render_layout.NESTED and len(relative.parts) >= 4:
        return relative.parts[-3]
    parsed = render_layout.parse_engine_output(path.name)
    return parsed[1] if parsed is not None else render_layout.UNCLASSIFIED


def _output_path(root: Path, name: str) -> Path:
    from gpuwm.render_layout import fs_path
    path = Path(fs_path(name, descend=True)).resolve()
    if not path.is_relative_to(root) or path.suffix.lower() != ".png" or not path.is_file():
        raise ValueError("Render receipt names no regular PNG inside its output directory")
    return path


def publish_invocation(*, root: Path, engine: str, requested_spec: str,
                       written, failures, skipped, layout: str, inputs=(), context_inputs=()) -> dict:
    """Record exact output/skip facts and publish their bounded aggregate."""
    from gpuwm.render_layout import fs_path
    from gpuwm.supervisor import atomic_write_json
    root = Path(fs_path(root, descend=True)).resolve()
    directory = root / ".render-receipts"
    directory.mkdir(parents=True, exist_ok=True)
    paths = list(dict.fromkeys(str(Path(fs_path(path, descend=True)).resolve()) for path in written))
    rendered = []
    for name in paths:
        path = _output_path(root, name)
        rendered.append({"path": str(path), "family": _family(path, root, layout),
                         "size_bytes": path.stat().st_size, "sha256": _hash(path)})
    invocation = {"schema": INVOCATION_SCHEMA, "id": uuid.uuid4().hex,
        "created_utc": datetime.now(timezone.utc).isoformat(), "output_root": str(root),
        "engine": engine, "requested_spec": str(requested_spec), "layout": layout,
        "inputs": [str(Path(path).resolve()) for path in inputs],
        "context_inputs": [str(Path(path).resolve()) for path in context_inputs],
        "rendered": rendered, "skipped": [{"family": str(family), "reason": str(reason)} for family, reason in skipped],
        "failures": [str(reason) for reason in failures]}
    atomic_write_json(directory / (invocation["id"] + ".json"), invocation)
    summary = summarize(root)
    atomic_write_json(root / SUMMARY_FILENAME, summary)
    return summary


def summarize(root: Path) -> dict:
    """Combine early/final invocations, verifying currently published PNGs.

    A PNG path counts once even if a later render repeats it. Skip/failure
    counts are invocation outcomes, explicitly retained as attempts. Exact
    full details remain in each immutable invocation receipt.
    """
    from gpuwm.render_layout import fs_path
    root = Path(fs_path(root, descend=True)).resolve()
    return _summarize_documents(root, _documents(root), verify_images=True)


def _documents(root: Path, paths=None):
    """Read bounded invocation metadata, without visiting scientific/image files."""
    documents = []
    directory = root / ".render-receipts"
    if directory.is_symlink():
        raise ValueError("Render receipt directory must not be a symlink")
    paths = sorted(directory.glob("*.json")) if paths is None else paths
    total = 0
    for path in paths:
        path = Path(path)
        if (path.parent != directory or path.suffix != ".json" or path.is_symlink()
                or len(documents) >= 4096):
            raise ValueError("Render invocation receipt is not a bounded regular file")
        with path.open("rb") as stream:
            payload = stream.read(_MAX_RECEIPT_BYTES - total + 1)
        total += len(payload)
        if total > _MAX_RECEIPT_BYTES:
            raise ValueError("Render invocation metadata exceeds its aggregate byte bound")
        raw = json.loads(payload)
        if not isinstance(raw, dict) or raw.get("schema") != INVOCATION_SCHEMA or Path(raw.get("output_root", "")).resolve() != root:
            raise ValueError("Render invocation belongs to another schema or output directory")
        documents.append((raw, path, payload))
    documents.sort(key=lambda pair: (pair[0]["created_utc"], pair[0]["id"]))
    return documents


def _recorded_output_path(root: Path, name: str) -> Path:
    """Validate a receipt's lexical path without statting or opening its PNG."""
    path = Path(name)
    if not path.is_absolute() or ".." in path.parts or not path.is_relative_to(root) or path.suffix.lower() != ".png":
        raise ValueError("Render receipt names a PNG outside its output directory")
    return path


def _summarize_documents(root: Path, documents, *, verify_images: bool) -> dict:
    current = {}
    specs = []
    skipped = Counter()
    reasons = defaultdict(set)
    failures = []
    for document, _path, _payload in documents:
        if document["requested_spec"] not in specs:
            specs.append(document["requested_spec"])
        for row in document["rendered"]:
            path = _recorded_output_path(root, row["path"])
            if not isinstance(row.get("sha256"), str) or not _SHA256.fullmatch(row["sha256"]):
                raise ValueError("Render receipt has no valid image digest")
            current[os.path.normcase(str(path))] = row
        for row in document["skipped"]:
            skipped[row["family"]] += 1
            reasons[row["family"]].add(row["reason"])
        failures.extend(document["failures"])
    rendered = Counter()
    for name, row in current.items():
        if verify_images:
            path = _output_path(root, row["path"])
            if path.stat().st_size != row["size_bytes"] or _hash(path) != row["sha256"]:
                raise ValueError("A published PNG changed after its render receipt")
        rendered[row["family"]] += 1
    requested = list(dict.fromkeys(token.strip() for spec in specs for token in spec.split(",") if token.strip()))
    explicit = not any(token in _CATALOG_REQUESTS for token in requested)
    skipped_rows = []
    for name, count in sorted(skipped.items()):
        exact = sorted(reasons[name])
        shown = [reason for reason in exact if len(reason.encode("utf-8")) <= 2048][:3]
        skipped_rows.append({"name": name, "count": count, "reasons": shown,
                             "additional_reasons": len(exact) - len(shown)})
    shown_failures = [reason for reason in failures if len(reason.encode("utf-8")) <= 2048][:8]
    summary = {"schema": SUMMARY_SCHEMA, "summary_path": str(root / SUMMARY_FILENAME),
        "requested_specs": specs[:8], "additional_requested_specs": max(0, len(specs)-8),
        "requested_families": requested[:64] if explicit else None,
        "requested_family_count": len(requested) if explicit else None,
        "additional_requested_families": max(0, len(requested)-64) if explicit else 0,
        "rendered_png_count": len(current), "rendered_family_count": len(rendered),
        "rendered_families": [{"name": name, "count": count} for name, count in sorted(rendered.items())][:64],
        "additional_rendered_families": max(0, len(rendered)-64),
        "skipped_count": sum(skipped.values()), "skipped_family_count": len(skipped),
        "skipped_families": skipped_rows[:64], "additional_skipped_families": max(0, len(skipped_rows)-64),
        "failure_count": len(failures), "failures": shown_failures,
        "additional_failures": len(failures)-len(shown_failures),
        "invocation_count": len(documents), "receipt_paths": [str(path) for _doc, path, _payload in documents[-8:]],
        "additional_receipts": max(0, len(documents)-8),
        "first_products_included": any(doc.get("publication", {}).get("kind") == "first-products"
                                        for doc, _path, _payload in documents),
        "count_basis": ("unique published PNG paths; skipped/failed render attempts across these invocations"
                        if verify_images else
                        "unique PNG paths recorded in stored render receipts; no image or weather files read")}
    return _bounded_summary(summary)


def _bounded_summary(summary):
    # Keep terminal/status consumers bounded without changing any quoted
    # reason. Omitted exact details remain in the invocation receipts.
    while len((json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")) > _MAX_STATUS_BYTES:
        rows = [row for row in summary["skipped_families"] if row["reasons"]]
        if rows:
            row = max(rows, key=lambda row: sum(len(reason) for reason in row["reasons"]))
            row["reasons"].pop(); row["additional_reasons"] += 1
        elif summary["failures"]:
            summary["failures"].pop(); summary["additional_failures"] += 1
        elif len(summary["receipt_paths"]) > 1:
            summary["receipt_paths"].pop(0); summary["additional_receipts"] += 1
        else:
            raise ValueError("Render summary metadata exceeds its status bound; exact invocation receipts were retained")
    return summary


def _preserve_bytes(path: Path, payload: bytes):
    """Create an original receipt once; never replace an earlier record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise ValueError("Preserved renderer receipt already exists with different bytes") from None


def relocate_invocations(source_root: Path, root: Path, published) -> dict | None:
    """Preserve early renderer receipts when their PNGs leave scratch storage.

    ``published`` carries the hashes/sizes measured by the PNG publisher just
    before this call. The original native receipt bytes remain intact beside
    the rebased publication, and the native skip/failure details survive too.
    """
    from gpuwm.render_layout import fs_path
    from gpuwm.supervisor import atomic_write_json
    source_root = Path(fs_path(source_root, descend=True)).resolve()
    root = Path(fs_path(root, descend=True)).resolve()
    documents = _documents(source_root)
    if not documents:
        return None
    if (root / ".render-receipts").is_symlink():
        raise ValueError("Render receipt directory must not be a symlink")
    facts = {row["name"]: row for row in published}
    rebased = []
    for document, path, payload in documents:
        value = deepcopy(document)
        for row in value["rendered"]:
            old_path = _recorded_output_path(source_root, row["path"])
            relative = old_path.relative_to(source_root)
            fact = facts.get(relative.as_posix())
            if (fact is None or fact.get("sha256") != row["sha256"]
                    or fact.get("size_bytes") != row["size_bytes"]):
                raise ValueError("Early renderer receipt differs from the published PNG bindings")
            row["path"] = str(root / relative)
        original = root / ".render-receipts" / "originals" / path.name
        value["output_root"] = str(root)
        value["publication"] = {"kind": "first-products", "source_receipt_path": str(path),
            "source_receipt_sha256": hashlib.sha256(payload).hexdigest(),
            "preserved_original_path": str(original), "original_output_root": str(source_root)}
        rebased.append((value, root / ".render-receipts" / path.name, original, payload))
    for document, path, original, payload in rebased:
        _preserve_bytes(original, payload)
        _preserve_bytes(path, (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    summary = _summarize_documents(root, _documents(root), verify_images=False)
    atomic_write_json(root / SUMMARY_FILENAME, summary)
    return summary


def merge_recorded_summary(root: Path, summary: dict) -> dict:
    """Include a legacy early publication using only explicitly stored receipts.

    The final invocation must name that exact early frame as an accumulation
    context. This is the final renderer's existing digest-verified claim that
    the early publication belongs to this render. No PNG or WRF is opened.
    New runs already carry the relocated native invocation and need no merge.
    """
    from gpuwm.first_products import FIRST_PRODUCTS_RECEIPT, FIRST_PRODUCTS_SCHEMA
    from gpuwm.render_layout import NESTED, fs_path
    if summary.get("schema") != SUMMARY_SCHEMA or summary.get("first_products_included") is True:
        return summary
    root = Path(fs_path(root, descend=True)).resolve()
    first_path = root / FIRST_PRODUCTS_RECEIPT
    if not first_path.is_file():
        return summary
    if first_path.is_symlink():
        raise ValueError("First-products receipt must not be a symlink")
    with first_path.open("rb") as stream:
        payload = stream.read(_MAX_RECEIPT_BYTES + 1)
    if len(payload) > _MAX_RECEIPT_BYTES:
        raise ValueError("First-products receipt exceeds its metadata byte bound")
    first = json.loads(payload)
    if not isinstance(first, dict) or first.get("schema") != FIRST_PRODUCTS_SCHEMA:
        raise ValueError("Unsupported first-products receipt schema")
    paths = None if summary.get("additional_receipts", 0) else summary.get("receipt_paths", [])
    documents = _documents(root, paths)
    contexts = {str(Path(name).resolve()) for doc, _path, _payload in documents for name in doc.get("context_inputs", ())}
    frame = Path(first.get("frame", ""))
    if not frame.is_absolute() or str(frame.resolve()) not in contexts:
        return summary
    if not isinstance(first.get("frame_sha256"), str) or not _SHA256.fullmatch(first["frame_sha256"]):
        raise ValueError("First-products receipt has no valid frame binding")
    rows = first.get("written")
    if not isinstance(rows, list) or not rows:
        raise ValueError("First-products receipt names no published images")
    written = []
    for row in rows:
        relative = Path(row["name"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("First-products image path must stay inside its output directory")
        path = _recorded_output_path(root, str(root / relative))
        written.append({"path": str(path), "family": _family(path, root, NESTED),
                        "sha256": row["sha256"], "size_bytes": row.get("size_bytes")})
    digest = hashlib.sha256(payload).hexdigest()
    legacy = {"schema": INVOCATION_SCHEMA, "id": "first-products-" + digest,
        "created_utc": datetime.fromtimestamp(first["published_unix_ms"] / 1000, timezone.utc).isoformat(),
        "output_root": str(root), "requested_spec": str(first.get("render_products") or "all"),
        "rendered": written, "skipped": [], "failures": [],
        "publication": {"kind": "first-products", "legacy_receipt_sha256": digest}}
    documents.append((legacy, first_path, payload))
    documents.sort(key=lambda pair: (pair[0]["created_utc"], pair[0]["id"]))
    merged = _summarize_documents(root, documents, verify_images=False)
    merged["first_products_receipt"] = {"path": str(first_path), "sha256": digest,
        "frame_sha256": first["frame_sha256"], "skips_available": False}
    return _bounded_summary(merged)


def read_summary(root: Path) -> dict | None:
    """Read this renderer's bounded published summary, never infer counts."""
    from gpuwm.render_layout import fs_path
    path = Path(fs_path(Path(root) / SUMMARY_FILENAME))
    if not path.is_file():
        return None
    if path.stat().st_size > _MAX_STATUS_BYTES:
        raise ValueError("Published render summary exceeds its status bound")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != SUMMARY_SCHEMA:
        raise ValueError("Unsupported render summary schema")
    return value
