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

#: Every function in this package that can make certification REFUSE.
#:
#: Declared, rather than inferred, because the defect this list exists to
#: prevent is invisible to every other kind of test: a validator that is
#: written, exported and unit-tested, and that the command never calls.
#: ``compile_platform_fingerprint``, ``describe_drift`` and
#: ``manifest_is_empty`` were exactly that through 2.3.3, so ``gpuwm
#: certify`` printed PASS over NVRTC drift and over a capsule whose kernel
#: manifest recorded nothing.
#:
#: ``tests/test_certify_reachability.py`` holds every name here to being
#: reachable from ``gpuwm certify``, ``gpuwm dual-run`` or the capsule
#: emission path by walking this package's call graph, and separately
#: requires that any validator-shaped function be listed -- so neither
#: half can be forgotten without a red test.
CERTIFICATION_CHECKS: tuple[str, ...] = (
    "band:validate_band",
    "capsule:validate_certification_capsule",
    "compile_platform:compile_platform_agreement",
    "compile_platform:describe_drift",
    "compile_platform:recorded_compile_platform",
    "compile_platform:unresolved_fingerprint_items",
    "kernel_manifest:manifest_is_empty",
    "pins:unresolved_pins",
    "verdict:failing_conditions",
    "wrf_reference:absent_reference_hashes",
    "wrf_reference:validate_wrf_reference_manifest",
)

__all__ = [
    "CERTIFICATION_CHECKS",
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
