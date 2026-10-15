"""Prune only explicitly registered suite artifacts. Never follow links."""
from __future__ import annotations

import re
import os
import time
from pathlib import Path

from gpuwm.cycle.ledger import write_receipt_atomically
from gpuwm.mcp.gpulock import _os_lock


RECEIPT_SUFFIXES = {".json", ".jsonl", ".log", ".toml", ".csv"}


def receipt_bytes(root, items=()):
    """Count registered receipts plus active/control files, without following links.

    Explicit nonreceipt categories are excluded. A registered receipt counts
    regardless of its extension. Unknown files are accounted for, never deleted.
    """
    categories = {str(Path(item["path"]).absolute()): item["kind"] for item in items}
    total = 0
    for base, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [name for name in dirs if not (Path(base) / name).is_symlink()]
        for name in files:
            path = Path(base) / name
            kind = categories.get(str(path.absolute()))
            if (kind == "receipts" or kind is None and path.suffix in RECEIPT_SUFFIXES) and not path.is_symlink():
                try:
                    total += path.lstat().st_size
                except FileNotFoundError:
                    pass
    return total


def duration(value):
    if isinstance(value, (int, float)):
        return float(value)
    match = re.fullmatch(r"(\d+(?:\.\d+)?)(s|m|h|d)", str(value))
    if not match:
        raise ValueError(f"invalid duration {value!r}")
    return float(match[1]) * {"s": 1, "m": 60, "h": 3600, "d": 86400}[match[2]]


def bytes_limit(value):
    match = re.fullmatch(r"(\d+(?:\.\d+)?)(B|KB|MB|GB)", str(value))
    if not match:
        raise ValueError(f"invalid byte cap {value!r}")
    return int(float(match[1]) * {"B": 1, "KB": 1000, "MB": 1000000, "GB": 1000000000}[match[2]])


def register(root, path, kind, *, row, cycle, protected=False):
    root, path = Path(root).resolve(), Path(path).absolute()
    if not path.is_relative_to(root) or path.is_symlink():
        raise ValueError("retention_outside_owner: artifact must be a file inside this suite root")
    if not path.is_file():
        raise ValueError(f"retention_missing: {path}")
    with _os_lock(root / "artifacts.mutex"):
        manifest = root / "artifacts.json"
        import json
        items = json.loads(manifest.read_text()) if manifest.exists() else []
        items = [x for x in items if x["path"] != str(path)]
        items.append(dict(path=str(path), kind=kind, row=row, cycle=cycle,
                          created=time.time(), bytes=path.stat().st_size, protected=protected))
        write_receipt_atomically(manifest, items)


def prune(root, policy, *, now=None, ledger=None):
    import json
    root = Path(root).resolve()
    now = time.time() if now is None else now
    deleted = []
    with _os_lock(root / "artifacts.mutex"):
        manifest = root / "artifacts.json"
        items = json.loads(manifest.read_text()) if manifest.exists() else []
        latest = {}
        for item in items:
            if item["kind"] == "checkpoints":
                latest[item["row"]] = max(latest.get(item["row"], ""), item["cycle"])
        receipt_total = receipt_bytes(root, items)
        cap = bytes_limit(policy.get("receipt_cap", "500MB"))
        retained = []
        for item in sorted(items, key=lambda x: x["created"]):
            path = Path(item["path"])
            # A replaced ancestor link cannot turn the owned manifest into a deletion oracle.
            if not path.absolute().is_relative_to(root) or not path.resolve().is_relative_to(root):
                raise ValueError(f"retention_path_changed: {path}")
            if not path.exists() and not path.is_symlink():
                continue
            keep = policy.get(item["kind"])
            expired = (item["cycle"] != latest.get(item["row"]) if keep == "latest"
                       else keep is not None and now - item["created"] > duration(keep))
            over = item["kind"] == "receipts" and receipt_total > cap
            if (expired or over) and not item.get("protected"):
                size = path.lstat().st_size
                path.unlink()
                deleted.append(dict(path=str(path), bytes=size, reason="cap" if over else "age"))
                if item["kind"] == "receipts":
                    receipt_total -= size
            else:
                retained.append(item)
        write_receipt_atomically(manifest, retained)
    if ledger and deleted:
        ledger.append("retained", deleted=deleted)
    # The inventory and deletion record themselves occupy receipt space.
    receipt_total = receipt_bytes(root, retained)
    return {"deleted": deleted, "receipt_bytes": receipt_total, "receipt_cap_bytes": cap,
            "cap_unmet": receipt_total > cap}
