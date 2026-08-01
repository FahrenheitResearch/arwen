"""Certification spine: what a run compiled, what it ran on, what it wrote.

The package carries one schema (``gpuwm.certification-capsule/v1``), one
builder, and one loader, so every route that finishes a forecast records the
same evidence in the same shape.  Nothing here imports a case module: the
certification path must stay generic or the receipt describes the case that
happened to exercise it rather than the model.
"""

from __future__ import annotations

from gpuwm.certify.capsule import (CAPSULE_FILENAME, CAPSULE_SCHEMA_ID,
                                   build_capsule, emit_capsule,
                                   load_certification_capsule,
                                   validate_certification_capsule)
from gpuwm.certify.kernel_manifest import (kernel_manifest, record_module,
                                           reset_kernel_manifest,
                                           source_sha256)
from gpuwm.certify.pins import PIN_KEYS, PINS, resolve_pins

__all__ = [
    "CAPSULE_FILENAME",
    "CAPSULE_SCHEMA_ID",
    "PINS",
    "PIN_KEYS",
    "build_capsule",
    "emit_capsule",
    "kernel_manifest",
    "load_certification_capsule",
    "record_module",
    "reset_kernel_manifest",
    "resolve_pins",
    "source_sha256",
    "validate_certification_capsule",
]
