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
from pathlib import Path
from typing import Any, Mapping

from gpuwm.certify.band import sha256_file

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


#: Artifacts a manifest may both NAME and commit beside itself, paired with
#: the field that publishes the digest of each.  ``build_recipe`` is one
#: filename pinned by the scalar ``build_recipe_sha256``; ``namelists`` is a
#: ``name -> filename`` mapping whose digests live under ``namelist_sha256``,
#: keyed the same way.  The executable and the reference wrfouts are named
#: in neither: they are not redistributable and only their hashes ship.
COMMITTED_ARTIFACT_KEYS: tuple[tuple[str, str], ...] = (
    ("build_recipe", "build_recipe_sha256"),
    ("namelists", "namelist_sha256"),
)


def mismatched_reference_artifacts(manifest: Mapping[str, Any],
                                   directory: str | Path
                                   ) -> tuple[dict[str, Any], ...]:
    """Committed artifacts whose published digest is not their digest.

    :func:`absent_reference_hashes` asks only whether a value is 64 hex
    characters, and that is the whole of what the certification chain ever
    checked.  A digest that pins nothing satisfies it exactly as well as one
    that pins the committed bytes -- which is how this repository shipped a
    ``build_recipe_sha256`` that had never, in any commit, hashed the recipe
    the same manifest names.  The recipe says of itself that "the digest is
    the SHA-256 of this file's bytes"; nothing recomputed it, so nothing
    said otherwise.

    Only artifacts the manifest *both names and commits beside itself* are
    recomputed.  An artifact that is named but absent is not a mismatch: the
    reference wrfouts and the WRF executable are deliberately outside the
    release and only their digests appear, so reporting them would turn a
    disclosed limit into a false alarm.  This reports what a reader holding
    the repository can check, and stays silent about what they cannot.

    Each entry carries the manifest field, the file name, the digest the
    manifest publishes and the digest the bytes have.
    """
    root = Path(directory)
    mismatches: list[dict[str, Any]] = []
    for name_key, digest_key in COMMITTED_ARTIFACT_KEYS:
        named = manifest.get(name_key)
        published = manifest.get(digest_key)
        if isinstance(named, str):
            pairs: list[tuple[str, Any]] = [(named, published)]
        elif isinstance(named, Mapping) and isinstance(published, Mapping):
            pairs = [(value, published.get(key))
                     for key, value in sorted(named.items())
                     if isinstance(value, str)]
        else:
            continue
        for filename, declared in pairs:
            artifact = root / Path(filename).name
            if not artifact.is_file():
                continue
            measured = sha256_file(artifact)
            if declared != measured:
                mismatches.append({
                    "key": digest_key,
                    "artifact": artifact.name,
                    "declared": declared,
                    "measured": measured,
                })
    return tuple(mismatches)


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
    "COMMITTED_ARTIFACT_KEYS",
    "MANIFEST_DIR_NAME",
    "MANIFEST_SCHEMA_ID",
    "MAPPING_HASH_KEYS",
    "REQUIRED_HASH_KEYS",
    "SCALAR_HASH_KEYS",
    "WrfReferenceError",
    "absent_reference_hashes",
    "mismatched_reference_artifacts",
    "reference_binding",
    "validate_wrf_reference_manifest",
]
