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


def _family(path: Path, root: Path, layout: str, families=None) -> str:
    from gpuwm import render_layout
    relative = path.relative_to(root)
    if families:
        # A lane whose filenames are not the wrfout engine's grammar
        # (an ensemble panel, a node-side gallery plate) knows its own
        # product slug and says so, rather than every one of its rows
        # landing as ``unclassified`` in the published summary.
        named = families.get(str(path)) or families.get(relative.as_posix())
        if named:
            return str(named)
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
                       written, failures, skipped, layout: str, inputs=(), context_inputs=(),
                       degraded=(), families=None, section_top_km=None,
                       section_fills=()) -> dict:
    """Record exact output/skip facts and publish their bounded aggregate.

    ``section_top_km`` is how tall the vertical cuts in this invocation
    were drawn -- the ceiling the door forwarded, or the engine's own
    14 km when a section was drawn and no ceiling was named, or ``None``
    when the invocation drew no section at all.  Without it a reader
    holding two cuts of the same line had nothing in the receipt that
    said why one of them was five times taller than the other.

    ``section_fills`` carries one row per vertical cut drawn: the
    family, the lowest and highest value its colour bar spans, whether
    the bottom band is the field's absence, and the rule that set the
    two numbers.  A section's bar is fitted to its own frame at both
    ends, so two cuts of one line an hour apart can be drawn on two
    different bars; without this the receipt recorded how TALL each cut
    was drawn and nothing about what its colours meant.

    ``degraded`` carries ``(path, reason)`` for every frame that reached
    the reader by a route the layout does not promise -- filed by copy
    after the move was refused, or left where it was drawn.  Those
    degradations used to exist only as a line on stderr, which meant a
    run that inverted its own layout said so nowhere a later reader
    could find it.  ``families`` names a product slug per written path
    for a lane whose filenames are not the engine's grammar.
    """
    from gpuwm.render_layout import fs_path
    from gpuwm.supervisor import atomic_write_json
    root = Path(fs_path(root, descend=True)).resolve()
    directory = root / ".render-receipts"
    directory.mkdir(parents=True, exist_ok=True)
    paths = list(dict.fromkeys(str(Path(fs_path(path, descend=True)).resolve()) for path in written))
    rendered = []
    for name in paths:
        path = _output_path(root, name)
        rendered.append({"path": str(path), "family": _family(path, root, layout, families),
                         "size_bytes": path.stat().st_size, "sha256": _hash(path)})
    invocation = {"schema": INVOCATION_SCHEMA, "id": uuid.uuid4().hex,
        "created_utc": datetime.now(timezone.utc).isoformat(), "output_root": str(root),
        "engine": engine, "requested_spec": str(requested_spec), "layout": layout,
        "inputs": [str(Path(path).resolve()) for path in inputs],
        "context_inputs": [str(Path(path).resolve()) for path in context_inputs],
        "rendered": rendered, "skipped": [{"family": str(family), "reason": str(reason)} for family, reason in skipped],
        "failures": [str(reason) for reason in failures],
        "degraded": [{"path": str(path), "reason": str(reason)} for path, reason in degraded],
        "section_top_km": None if section_top_km is None else float(section_top_km),
        "section_fills": [_section_fill_row(row) for row in section_fills]}
    atomic_write_json(directory / (invocation["id"] + ".json"), invocation)
    summary = summarize(root)
    atomic_write_json(root / SUMMARY_FILENAME, summary)
    return summary


def deliver(*, root: Path, engine: str, requested_spec: str, written,
            failures=(), skipped=(), layout: str, inputs=(), context_inputs=(),
            families=None, degraded=(), section_top_km=None,
            section_fills=()) -> dict:
    """Record one lane's delivery and publish its receipt; the summary.

    THE delivery seam for every lane that draws pictures.
    :func:`publish_invocation` had exactly one caller, so an ensemble
    suite and a node-side gallery delivered their PNGs with no
    ``render-summary.json`` beside them at all, and the desktop and
    remote surfaces that read that file saw nothing for those runs.  A
    receipt is a property of a DELIVERY, not of one door.

    It also enforces the half of the layout contract a receipt is the
    only place to state: under ``layout=nested`` a delivered picture
    lives at ``<domain>/[<episode>/]<product>/<day>/<file>``, so a path
    with fewer segments than that did not reach the layout.  Those rows
    are recorded as FAILURES naming the file, kept out of the rendered
    count, and the caller gets a summary that says the delivery was
    partial -- rather than a clean receipt for a run whose pictures are
    lying at the root.  ``layout=flat`` has no folders to reach and is
    left exactly alone.

    ``families`` names the product slug per written path for a lane
    whose filenames are not the engine's own grammar, and ``degraded``
    carries the ``(path, reason)`` pairs :func:`gpuwm.render_layout.deliver`
    returned for frames that reached the reader by a lesser route.
    """

    from gpuwm.render_layout import NESTED, fs_path
    resolved_root = Path(fs_path(root, descend=True)).resolve()
    kept, broken = [], [str(reason) for reason in failures]
    for name in written:
        path = Path(fs_path(name, descend=True)).resolve()
        if layout == NESTED and (not path.is_relative_to(resolved_root)
                                 or len(path.relative_to(resolved_root).parts) < 4):
            broken.append(
                f"delivered outside the nested layout: {path} is not "
                f"<domain>/[<episode>/]<product>/<day>/<file> under {resolved_root}")
            continue
        kept.append(path)
    return publish_invocation(
        root=root, engine=engine, requested_spec=requested_spec, written=kept,
        failures=broken, skipped=skipped, layout=layout, inputs=inputs,
        context_inputs=context_inputs, degraded=degraded, families=families,
        section_top_km=section_top_km, section_fills=section_fills)


def _section_fill_row(row) -> dict:
    """One drawn-range row, in the receipt's own shape.

    A row whose numbers are not numbers is a receipt that cannot be
    read back, so the values are coerced here, once, where they enter
    the record rather than where they are printed.
    """

    return {"family": str(row["family"]), "lo": float(row["lo"]),
            "hi": float(row["hi"]), "absence": bool(row.get("absence")),
            "rule": str(row.get("rule", ""))}


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
    degraded = []
    section_tops = []
    section_fills = []
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
        # ``.get``: receipts written before the layout carried its own
        # degradations are still valid receipts and still summarize.
        degraded.extend(document.get("degraded", ()))
        top = document.get("section_top_km")
        if top is not None and float(top) not in section_tops:
            section_tops.append(float(top))
        # ``.get``: a receipt written before the drawn range was
        # recorded is still a valid receipt and still summarizes.
        for row in document.get("section_fills", ()):
            row = _section_fill_row(row)
            if row not in section_fills:
                section_fills.append(row)
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
        "section_tops_km": section_tops[:8],
        "additional_section_tops_km": max(0, len(section_tops)-8),
        "section_fills": section_fills[:8],
        "additional_section_fills": max(0, len(section_fills)-8),
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
        "degraded_count": len(degraded),
        "degraded": [{"path": row["path"], "reason": row["reason"]} for row in degraded[:8]],
        "additional_degraded": max(0, len(degraded)-8),
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
        elif summary.get("degraded"):
            summary["degraded"].pop(); summary["additional_degraded"] += 1
        elif summary.get("section_fills"):
            summary["section_fills"].pop()
            summary["additional_section_fills"] += 1
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


def stamp_status(root: Path, *, status: str, pictures_on_disk: int | None,
                 banner_path=None, pictures_error: str | None = None
                 ) -> dict | None:
    """Record, in the published summary, that this run did not finish.

    The summary is the document every surface reads to learn what a run
    drew (the desktop's native-plots door opens on its presence alone),
    so a run that stopped has to say so HERE rather than only in a file
    beside it.  Three fields, added to the schema rather than replacing
    anything: ``status``, ``pictures_on_disk`` counted from the tree,
    and the path of the banner that states the stop in words.

    An existing summary is AMENDED.  It is the early render's own record
    of what it drew and which families it skipped, and a reader that
    keyed on those rows must not find them gone because the run stopped.

    A render directory with pictures but no summary gets one written.
    That happens when a delivery published PNGs without an invocation
    receipt beside them, and without this the pictures would be on disk
    with no document naming them, which is the state every reader treats
    as "this run drew nothing".  Its ``count_basis`` says the count was
    taken from the tree, so no reader mistakes it for a verified
    per-family aggregate.

    A count that could not be TAKEN is not a count of zero.  A picture
    tree whose listing failed arrives as ``pictures_on_disk=None`` with
    the error in ``pictures_error``; the summary then carries a null
    count beside that error rather than a zero every reader would show
    as "this run drew nothing".

    Best effort: a directory that cannot be read or written is not worth
    failing an already-failed run over, and ``None`` comes back.
    """

    from gpuwm.render_layout import fs_path

    try:
        root = Path(fs_path(Path(root), descend=True))
        summary = read_summary(root)
    except (OSError, ValueError):
        summary = None
    if summary is None:
        summary = {
            "schema": SUMMARY_SCHEMA,
            "summary_path": str(root / SUMMARY_FILENAME),
            "requested_specs": [], "additional_requested_specs": 0,
            "requested_families": None, "requested_family_count": None,
            "additional_requested_families": 0,
            "rendered_png_count": (0 if pictures_on_disk is None
                                   else int(pictures_on_disk)),
            "rendered_family_count": 0, "rendered_families": [],
            "additional_rendered_families": 0,
            "skipped_count": 0, "skipped_family_count": 0,
            "skipped_families": [], "additional_skipped_families": 0,
            "failure_count": 0, "failures": [], "additional_failures": 0,
            "degraded_count": 0, "degraded": [], "additional_degraded": 0,
            "invocation_count": 0, "receipt_paths": [],
            "additional_receipts": 0,
            "first_products_included": False,
            "count_basis": ("PNG files counted in this directory after the "
                            "run stopped; no render invocation receipt was "
                            "found beside them"
                            if pictures_on_disk is not None else
                            "this directory could not be listed after the "
                            "run stopped, so nothing here was counted; "
                            "pictures_on_disk_error says why"),
        }
    summary["status"] = str(status)
    summary["pictures_on_disk"] = (None if pictures_on_disk is None
                                   else int(pictures_on_disk))
    # The SAME pair of key names the failed-render capsule writes into
    # ``report.json``: a count and, beside it, why there is none.  One
    # vocabulary across the two documents, so a reader keys on one.
    summary["pictures_on_disk_error"] = (None if pictures_error is None
                                         else str(pictures_error))
    summary["banner_path"] = None if banner_path is None else str(banner_path)
    try:
        from gpuwm.supervisor import atomic_write_json

        summary = _bounded_summary(summary)
        atomic_write_json(root / SUMMARY_FILENAME, summary)
    except (OSError, ValueError):
        return None
    return summary


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
