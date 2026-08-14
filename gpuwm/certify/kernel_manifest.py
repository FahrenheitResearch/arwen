"""Per-module record of exactly what the process compiled.

Every ``cp.RawModule`` construction in the tree reports here with the same
locals it handed the compiler, so the capsule's ``kernel_manifest`` describes
the translation units and option tuples of the run rather than a single global
flags string.  The option tuples differ between sites on purpose -- the nest
interpolation module alone carries ``-fmad=false``, the shortwave module alone
carries ``--ftz=false`` -- and a manifest that lost that distinction would
claim two different compiled images were one.

The source half of the record is CPU-checkable: ``source_sha256`` hashes the
translation-unit text, which any reader can rebuild from the ``.cu`` files, the
common header and the preprocessor defines without a device.  The compiled-image
half is only obtainable where a device compiled something, and is recorded with
an explicit status so a CPU run never reads as if it had measured one.
"""

from __future__ import annotations

import hashlib
import threading
from typing import Any, Iterable, Mapping

#: Status of an image hash taken from a module the runtime actually compiled.
IMAGE_STATUS_RESOLVED = "resolved"

#: Status of an image hash no device produced, or the toolchain does not expose.
IMAGE_STATUS_UNAVAILABLE = "unavailable"

_LOCK = threading.Lock()
_MANIFEST: dict[str, dict[str, Any]] = {}


class KernelManifestConflict(RuntimeError):
    """Two different compiled images claimed one module key, indistinguishably."""


def source_sha256(source: str) -> str:
    """SHA-256 of a translation unit, over its UTF-8 bytes."""
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _normalized_options(options: Iterable[str]) -> tuple[str, ...]:
    normalized = tuple(str(option) for option in options)
    if normalized != tuple(options):
        raise TypeError("kernel compile options must already be strings")
    return normalized


def _image_identity(module: Any) -> dict[str, Any]:
    """Hash the compiled image if the toolchain hands one back.

    CuPy does not promise a public accessor for the PTX or cubin behind a
    ``RawModule``; where no attribute yields bytes, the entry says so with the
    reason rather than reporting an image nothing read.
    """
    if module is None:
        return {"status": IMAGE_STATUS_UNAVAILABLE,
                "reason": "no compiled module was offered to the recorder"}
    for owner in (module, getattr(module, "module", None)):
        if owner is None:
            continue
        for kind in ("cubin", "ptx"):
            try:
                image = getattr(owner, kind, None)
            except Exception:  # pragma: no cover - defensive on foreign objects
                image = None
            if isinstance(image, (bytes, bytearray)):
                return {"status": IMAGE_STATUS_RESOLVED, "kind": kind,
                        "sha256": hashlib.sha256(bytes(image)).hexdigest()}
            if isinstance(image, str) and image:
                return {"status": IMAGE_STATUS_RESOLVED, "kind": kind,
                        "sha256": hashlib.sha256(
                            image.encode("utf-8")).hexdigest()}
    return {"status": IMAGE_STATUS_UNAVAILABLE,
            "reason": "the installed CuPy exposes no PTX or cubin on RawModule"}


def _distinct_key(module_key: str, entry: Mapping[str, Any]) -> str:
    """A key that no other compiled image already holds.

    One translation unit compiled twice with different options, or one option
    tuple applied to two different sources, is two compiled images and the
    manifest owes an entry to each.  A negative control that recompiles a
    perturbed source, or the same unit with FMA contraction disabled, is
    exactly that case, and it must not overwrite the production entry or stop
    the process it runs in.  The suffixes are deterministic, so two runs that
    compile the same set produce the same keys.
    """
    previous = _MANIFEST.get(module_key)
    if previous is None or _same_image(previous, entry):
        return module_key
    by_options = f"{module_key} {' '.join(entry['options'])}".rstrip()
    previous = _MANIFEST.get(by_options)
    if previous is None or _same_image(previous, entry):
        return by_options
    by_source = f"{by_options}#{entry['source_sha256'][:12]}"
    previous = _MANIFEST.get(by_source)
    if previous is None or _same_image(previous, entry):
        return by_source
    raise KernelManifestConflict(
        f"module key {by_source!r} already records a different image")


def _same_image(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return all(left[field] == right[field]
               for field in ("source_sha256", "options"))


def record_module(module_key: str, *, source: str,
                  options: Iterable[str], module: Any = None) -> None:
    """Record one compiled module under ``module_key``.

    ``source`` and ``options`` are the very values the caller passed to
    ``cp.RawModule``; passing anything else makes the manifest describe a
    compile that did not happen.  A second, different image offered under a
    key already taken is recorded beside the first under a deterministic
    suffix rather than replacing it: the manifest is a record of everything
    the process compiled, and losing one of two images would under-report it.
    """
    entry = {
        "source_sha256": source_sha256(source),
        "options": list(_normalized_options(options)),
        "compiled_image": _image_identity(module),
    }
    with _LOCK:
        key = _distinct_key(module_key, entry)
        previous = _MANIFEST.get(key)
        if (previous is not None
                and previous["compiled_image"]["status"]
                == IMAGE_STATUS_RESOLVED):
            entry["compiled_image"] = previous["compiled_image"]
        _MANIFEST[key] = entry


def kernel_manifest() -> dict[str, dict[str, Any]]:
    """Snapshot of every module recorded in this process, key-sorted."""
    with _LOCK:
        return {
            key: {
                "source_sha256": entry["source_sha256"],
                "options": list(entry["options"]),
                "compiled_image": dict(entry["compiled_image"]),
            }
            for key, entry in sorted(_MANIFEST.items())
        }


def reset_kernel_manifest() -> None:
    """Drop every record.  For tests that need a known-empty manifest."""
    with _LOCK:
        _MANIFEST.clear()


def manifest_is_empty(manifest: Mapping[str, Any]) -> bool:
    """Whether a manifest recorded nothing -- true of any CPU-only route.

    The manifest is a REQUIRED argument, and deliberately so.  This function
    used to default to the *live* manifest of whatever process asked, which is
    the wrong document for every one of its callers: a certifier compiles no
    kernels, so the default would have judged the certifying process instead
    of the capsule in front of it, and a certifier that HAD compiled something
    would have reported every capsule full.  The guard judges the record it is
    handed or it does not run.
    """
    return not manifest
