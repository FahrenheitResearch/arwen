"""Shared, CPU-only primitives for the N5S shadow toolchain.

This module deliberately has no CuPy imports.  The controller, the WSL-side
WRF scripts, the Windows-side gpuwm importer, and the evidence emitters all
use the same restored-input digest implementation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
from typing import Mapping


RESTORED_INPUT_FILENAMES = (
    "wrfbdy_d01",
    "wrfinput_d01",
    "wrfinput_d02",
    "wrfinput_d03",
    "wrfinput_d04",
)


def sha256_file(path: str | Path, *, block_bytes: int = 8 * 1024 * 1024
                ) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def restored_input_sha256(directory: str | Path) -> str:
    """Hash the canonical five-file restored-input byte stream.

    Each sorted record is encoded as ``u64be(name length), UTF-8 name,
    u64be(file length), raw file bytes``.  Length framing makes the
    filename/byte concatenation unambiguous and is identical on Windows and
    WSL.  Missing files fail before a digest is returned.
    """
    root = Path(directory)
    digest = hashlib.sha256()
    for name in sorted(RESTORED_INPUT_FILENAMES):
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"restored input is missing {path}")
        encoded = name.encode("utf-8")
        size = path.stat().st_size
        digest.update(struct.pack(">Q", len(encoded)))
        digest.update(encoded)
        digest.update(struct.pack(">Q", size))
        with path.open("rb") as stream:
            while block := stream.read(8 * 1024 * 1024):
                digest.update(block)
    return digest.hexdigest()


def stable_hash(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: str | Path, payload: Mapping[str, object]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    return target


__all__ = [
    "RESTORED_INPUT_FILENAMES", "restored_input_sha256", "sha256_file",
    "stable_hash", "write_json",
]
