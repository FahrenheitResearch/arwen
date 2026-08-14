"""The WRF reference manifest, and the precondition that it be complete.

A matched-run comparison is a claim about two binaries on the same case.  The
capsule witnesses one of them.  This module is the other half: which WRF
executable produced the reference stream, from which build recipe, under which
namelists, and with which output bytes.  Certification refuses to proceed while
any of those four hashes is absent -- not because the missing hash is likely to
be wrong, but because a verdict that did not know what it was compared against
is not a verdict.

The manifests themselves live under ``docs/public/wrf-reference/`` (the WRF
side of the comparison is documentation the reader can act on; the reference
wrfouts and the ERA5 inputs are not redistributable and only their hashes
appear).  The refusal below is unconditional code and does not depend on any
manifest having been committed yet.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

MANIFEST_SCHEMA_ID = "gpuwm.wrf-reference-manifest/v1"

#: Repository-relative home of the committed manifests.
MANIFEST_DIR_NAME = "docs/public/wrf-reference"

#: The four hash groups a certification-grade reference manifest must carry.
#: Each is required by name, so a manifest that simply omits one is refused by
#: the same code path as one that carries it empty.
REQUIRED_HASH_KEYS: tuple[str, ...] = (
    "wrf_exe_sha256",
    "build_recipe_sha256",
    "namelist_sha256",
    "reference_wrfout_sha256",
)

#: Which of those groups is a single digest, and which is a set of them.
SCALAR_HASH_KEYS: tuple[str, ...] = ("wrf_exe_sha256", "build_recipe_sha256")
MAPPING_HASH_KEYS: tuple[str, ...] = ("namelist_sha256",
                                      "reference_wrfout_sha256")

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class WrfReferenceError(ValueError):
    """A WRF reference manifest does not satisfy its contract."""


def _is_digest(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.match(value) is not None


def absent_reference_hashes(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    """Hash groups the manifest does not actually carry, in declared order.

    Absent means any of: the key is missing; its value is null; a scalar group
    is not a SHA-256 digest; a mapping group is empty or carries an entry whose
    value is not a SHA-256 digest.  An ``unavailable`` marker is absent too --
    an honest admission is still not a hash.
    """
    absent: list[str] = []
    for key in REQUIRED_HASH_KEYS:
        value = manifest.get(key)
        if key in SCALAR_HASH_KEYS:
            if not _is_digest(value):
                absent.append(key)
            continue
        if not isinstance(value, Mapping) or not value:
            absent.append(key)
            continue
        if not all(_is_digest(entry) for entry in value.values()):
            absent.append(key)
    return tuple(absent)


def validate_wrf_reference_manifest(manifest: Mapping[str, Any]
                                    ) -> dict[str, Any]:
    """Check the manifest's shape.  Completeness is certify's refusal, not
    this function's: a manifest may legitimately be committed incomplete while
    the reference bank is being measured, and it must still parse."""
    if manifest.get("schema") != MANIFEST_SCHEMA_ID:
        raise WrfReferenceError(
            f"not a {MANIFEST_SCHEMA_ID} document: "
            f"schema is {manifest.get('schema')!r}")
    for key in ("wrf_version", "config_sha256"):
        if not isinstance(manifest.get(key), str) or not manifest[key]:
            raise WrfReferenceError(
                f"WRF reference manifest carries no {key}")
    if not _SHA256.match(manifest["config_sha256"]):
        raise WrfReferenceError(
            "WRF reference manifest config_sha256 is not a SHA-256 digest")
    return dict(manifest)


def reference_binding(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """The part of the manifest a verdict binds itself to."""
    return {
        "schema": manifest.get("schema"),
        "wrf_version": manifest.get("wrf_version"),
        "wrf_commit": manifest.get("wrf_commit"),
        "config_sha256": manifest.get("config_sha256"),
        **{key: manifest.get(key) for key in REQUIRED_HASH_KEYS},
    }


__all__ = [
    "MANIFEST_DIR_NAME",
    "MANIFEST_SCHEMA_ID",
    "MAPPING_HASH_KEYS",
    "REQUIRED_HASH_KEYS",
    "SCALAR_HASH_KEYS",
    "WrfReferenceError",
    "absent_reference_hashes",
    "reference_binding",
    "validate_wrf_reference_manifest",
]
