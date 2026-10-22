"""Build identity from the metadata of the distribution serving this import.

Git is a build tool, never a prerequisite for writing a job receipt. Older
wheels and unpackaged trees have no stamp; say unknown and keep the job alive.
"""
from __future__ import annotations

import json
import re
import sys

from gpuwm import owning_distribution

BUILD_METADATA = "build-provenance.json"
BUILD_SCHEMA = "gpuwm-build-provenance-v1"


def package_build_identity() -> dict:
    """Return version/hash and a copyable receipt line without launching tools."""
    version = commit = "unknown"
    try:
        owner = owning_distribution()
        if owner is not None:
            version = str(owner.version or "unknown")
            stamp = json.loads(owner.read_text(BUILD_METADATA) or "{}")
            candidate = stamp.get("commit")
            if (stamp.get("schema") == BUILD_SCHEMA
                    and stamp.get("version") == version
                    and isinstance(candidate, str)
                    and re.fullmatch(r"[0-9a-f]{40}", candidate)):
                commit = candidate
    except Exception:  # Unreadable optional metadata cannot stop a forecast.
        pass
    line = f"gpuwm build provenance: version={version}, build_hash={commit}"
    if "unknown" in (version, commit):
        line += "; installed package metadata unavailable or incomplete; continuing"
        print(line, file=sys.stderr)
    return dict(version=version, commit=commit,
                identity_source="installed-package-metadata", receipt_line=line)
