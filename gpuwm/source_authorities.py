"""Immutable source-family authorities shipped with the RW-WPS wheel."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


_AUTHORITY_ROOT = Path(__file__).with_name("authorities")
_TWENTYCRV3_NAMES = MappingProxyType({
    "mapping": "rw-wps-20crv3-member-grib2.mapping.json",
    "composition": "rw-wps-20crv3-member-grib2.composition.json",
    "provenance": "rw-wps-20crv3-member-grib2.provenance.json",
})
_TWENTYCRV3_SHA256 = MappingProxyType({
    "mapping": "2e9877d51d9c993e83311c87236467b99ce9022638a343985da10ccd195efe09",
    "composition": "aa4f3fac03c09e8461c5e6c5e04a6bed48b5ad477babc4c75e8dd10fd92fe7b2",
    "provenance": "d1248e1b091f59841757a98a024cbe2868cebc25308f4eb4f9608e2c1755f3b1",
})


def twentycrv3_authorities() -> Mapping[str, Path]:
    """Resolve and byte-verify the exact packaged 20CRv3 authorities."""

    resolved: dict[str, Path] = {}
    for role, name in _TWENTYCRV3_NAMES.items():
        path = (_AUTHORITY_ROOT / name).resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"packaged 20CRv3 {role} authority is missing: {path}"
            )
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = _TWENTYCRV3_SHA256[role]
        if observed != expected:
            raise RuntimeError(
                f"packaged 20CRv3 {role} authority hash differs: "
                f"expected {expected}, got {observed}"
            )
        resolved[role] = path
    return MappingProxyType(resolved)


#: The GFS WPS Vtable `gpuwm adapt` documents as its worked example.
#:
#: It lived in `configs/`, which is not a package, so the wheel did not
#: carry it and a pip user following the documented adapt flow was told
#: to pass a file their install did not have.  It ships beside the
#: 20CRv3 authorities now, under the same recursive package-data glob
#: and the same byte contract, because it is the same kind of thing: an
#: immutable input a front door reads, not a config anyone edits.
_GFS_VTABLE_NAME = "Vtable.GFS.rw-wps"
#: Re-pinned 2026-07-30 to the committed bytes: 9e391880... -> ec8e615b...
#:
#: The old value was this file's CRLF form.  It is not a JSON file, so
#: the per-path ``gpuwm/authorities/*.json text eol=lf`` rule did not
#: cover it, and on a Windows clone (git-for-Windows defaults
#: core.autocrlf=true) it materialized with CRLF -- which is the
#: checkout the constant was taken from, and the wheel that was built
#: from it.  The same file on Linux, and in the object database, is LF,
#: so this gate could only ever have been true on one platform: the
#: byte contract it exists to enforce was itself platform-dependent.
#:
#: The repository now declares ``* -text``, so every checkout gets the
#: committed bytes and this is the one hash for all of them.  Nothing
#: was widened: the file is unchanged, the check is unchanged, and the
#: constant now names what the file actually is.
_GFS_VTABLE_SHA256 = (
    "ec8e615ba724b3ddf114c4c199a81083b3a17b4e1705055ec016f1769144090e")


def packaged_gfs_vtable() -> Path:
    """Resolve and byte-verify the packaged GFS WPS Vtable."""

    path = (_AUTHORITY_ROOT / _GFS_VTABLE_NAME).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"packaged GFS Vtable is missing: {path}")
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != _GFS_VTABLE_SHA256:
        raise RuntimeError(
            f"packaged GFS Vtable hash differs: expected "
            f"{_GFS_VTABLE_SHA256}, got {observed}")
    return path


def packaged_gfs_vtable_sha256() -> str:
    """The immutable SHA-256 contract, without touching the filesystem."""

    return _GFS_VTABLE_SHA256


def twentycrv3_authority_sha256() -> Mapping[str, str]:
    """Return the immutable SHA-256 contract without touching the filesystem."""

    return _TWENTYCRV3_SHA256


__all__ = [
    "packaged_gfs_vtable", "packaged_gfs_vtable_sha256",
    "twentycrv3_authorities", "twentycrv3_authority_sha256",
]
