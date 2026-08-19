"""Compressed-object staging for acquisition inputs.

The engine's closed acquisition codec set.  Some agencies publish every
GRIB object wrapped in a byte-level compression layer (DWD open data
wraps each field file in bzip2); the object a caller stages and hashes
is the compressed one, and no table entry can decompress bytes, so the
codec lives here, once, in the generic engine -- the acquisition twin of
a packing template.  Membership is deliberately closed: a codec with no
staged consumer is a capability without a door, so the set grows only
beside a source that exercises it.

Identification is by magic bytes, never by filename, because a filename
is caller-controlled data and the refusal below must describe the bytes
that were actually supplied.
"""

from __future__ import annotations

import bz2
from pathlib import Path


#: Magic prefix -> codec name.  The engine's entire acquisition codec set.
_MAGIC_CODECS = {
    b"BZh": "bz2",
}

#: How much of the head is needed to classify every registered magic.
_MAGIC_PROBE_BYTES = max(len(magic) for magic in _MAGIC_CODECS)

_DECOMPRESS = {
    "bz2": bz2.decompress,
}


def object_codec(path: str | Path) -> str | None:
    """The registered codec wrapping ``path``'s bytes, or ``None``."""

    with open(path, "rb") as stream:
        head = stream.read(_MAGIC_PROBE_BYTES)
    for magic, name in _MAGIC_CODECS.items():
        if head.startswith(magic):
            return name
    return None


def staged_decoded_object(
    path: str | Path, staging_directory: str | Path,
) -> Path:
    """Return a plainly decoded twin of one acquisition object.

    An uncompressed object is returned unchanged.  A compressed one is
    decompressed exactly once into ``staging_directory`` under a
    collision-free name, and the STAGED path is returned; provenance
    (hashes, manifests, references) stays bound to the supplied path,
    because the supplied bytes are what the acquisition actually
    delivered.  The staging directory's lifetime is the caller's --
    decode-scoped, deleted with the decode.
    """

    path = Path(path)
    codec = object_codec(path)
    if codec is None:
        return path
    staging_directory = Path(staging_directory)
    staged = staging_directory / f"decoded-{len(list(staging_directory.iterdir())):06d}-{path.stem}"
    if staged.exists():
        raise ValueError(f"codec staging collision at {staged}")
    data = path.read_bytes()
    try:
        plain = _DECOMPRESS[codec](data)
    except (OSError, EOFError, ValueError) as error:
        raise ValueError(
            f"acquisition object {path} carries the {codec} magic but does "
            f"not decompress: {error}"
        ) from error
    staged.write_bytes(plain)
    return staged


__all__ = ["object_codec", "staged_decoded_object"]
