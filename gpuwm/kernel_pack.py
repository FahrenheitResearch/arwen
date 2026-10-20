"""Compiler-bound pack codec shared with hex (gpuwm.kernel-pack/v1)."""
from __future__ import annotations

import hashlib
import json
import lzma
import os
from pathlib import Path
import tempfile

SCHEMA = "gpuwm.kernel-pack/v1"
MAX_ENTRIES = 16000


def digest(arguments) -> str:
    return hashlib.sha256(json.dumps(arguments, sort_keys=True).encode("utf-8")).hexdigest()


def _load(path: Path):
    data = Path(path).read_bytes()
    if str(path).endswith(".xz"):
        data = lzma.decompress(data)
    return json.loads(data.decode("utf-8"))


def compiler_of(path) -> dict | None:
    """The identity a pack was built under, or None when the file is unreadable."""
    try:
        document = _load(Path(path))
    except (OSError, ValueError, lzma.LZMAError):
        return None
    compiler = document.get("compiler") if isinstance(document, dict) else None
    return compiler if isinstance(compiler, dict) else None


def read(path, compiler) -> list[dict]:
    document = _load(Path(path))
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        raise ValueError(f"{path}: not a {SCHEMA} document")
    if document.get("compiler") != compiler:
        raise ValueError(f"{path}: built for compiler {document.get('compiler')!r}, "
                         f"this process has {compiler!r}")
    records = document.get("entries")
    if not isinstance(records, list) or len(records) > MAX_ENTRIES:
        raise ValueError(f"{path}: entries missing or over {MAX_ENTRIES}")
    rows = []
    for record in records:
        arguments = record.get("arguments") if isinstance(record, dict) else None
        if not isinstance(arguments, dict) or record.get("sha256") != digest(arguments):
            raise ValueError(f"{path}: an entry does not match its recorded digest")
        rows.append(arguments)
    return rows


def write(path, compiler, entries) -> None:
    path = Path(path)
    entries = list(entries)
    if len(entries) > MAX_ENTRIES:
        raise ValueError(f"{len(entries)} entries exceed the {MAX_ENTRIES} cap")
    document = {"schema": SCHEMA, "compiler": compiler,
                "entries": [{"sha256": digest(row), "arguments": row} for row in entries]}
    data = (json.dumps(document, sort_keys=True) + "\n").encode("utf-8")
    if str(path).endswith(".xz"):
        data = lzma.compress(data, preset=9 | lzma.PRESET_EXTREME)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=".kernel-pack.", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
        # mkstemp creates 0600; a pack is package data every user must read.
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
