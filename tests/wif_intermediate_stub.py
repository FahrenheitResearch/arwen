"""The smallest file the WIF resolver counts as WRF's aerosol climatology.

Every door test that needs "the dataset is staged" used to write a
ZERO-BYTE file, because the resolver asked ``is_file()`` and nothing more.
That made the fixture prove the wrong thing: the precondition admitted an
externally forced mp=28 run whose ingest then died decoding an empty file,
which is a refusal after step 0.  The resolver now decodes the first WPS
intermediate (IFV=5) field record, so the fixture writes one.

It is deliberately the SMALLEST such file (a one-cell field) rather than a
copy of WRF's 225 MB dataset: the property under test is "a candidate that
is structurally the dataset resolves", and the bytes of the real climatology
are the ingest lane's business, not the door's.
"""

from __future__ import annotations

import struct


def _record(payload: bytes) -> bytes:
    """One Fortran sequential record: big-endian length, payload, length."""

    marker = struct.pack(">i", len(payload))
    return marker + payload + marker


def wif_intermediate_bytes(field: str = "QNWFA", nx: int = 1,
                           ny: int = 1) -> bytes:
    """One complete IFV=5 field record group, cylindrical-equidistant."""

    header = bytearray(192)
    header[60:69] = field.ljust(9).encode("ascii")
    # xlvl, nx, ny, iproj -- the slice gpuwm/ingest/wif_climatology.py
    # reads at header[140:156]; iproj=0 is the projection the climatology
    # declares and the only one the reader admits.
    header[140:156] = struct.pack(">fiii", 1.0, nx, ny, 0)
    projection = bytearray(28)
    # startlat, startlon, deltalat, deltalon, earth radius at proj[8:28].
    projection[8:28] = struct.pack(">fffff", -90.0, 0.0, 1.0, 1.0, 6370.0)
    return b"".join((
        _record(struct.pack(">i", 5)),
        _record(bytes(header)),
        _record(bytes(projection)),
        _record(struct.pack(">i", 0)),
        _record(struct.pack(">f", 0.0) * (nx * ny)),
    ))


def write_minimal_wif_intermediate(path, field: str = "QNWFA") -> object:
    """Write :func:`wif_intermediate_bytes` at ``path`` and return it."""

    path.write_bytes(wif_intermediate_bytes(field))
    return path
