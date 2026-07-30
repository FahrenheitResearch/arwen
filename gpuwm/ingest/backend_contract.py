"""Backend-neutral identity and parity contract for native preprocessing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


WORKLOAD_SCHEMA = "gpuwm-preprocess-workload-v1"
RECEIPT_SCHEMA = "gpuwm-preprocess-backend-receipt-v1"
PARITY_SCHEMA = "gpuwm-preprocess-backend-parity-v1"


@dataclass(frozen=True)
class ArrayParityRule:
    """Declared cross-backend contract for one semantic array."""

    mode: str = "numeric"
    rtol: float = 3.0e-5
    atol: float = 5.0e-3

    def __post_init__(self) -> None:
        if self.mode not in ("byte_exact", "numeric"):
            raise ValueError("parity mode must be byte_exact or numeric")
        if not np.isfinite(self.rtol) or not np.isfinite(self.atol):
            raise ValueError("parity tolerances must be finite")
        if self.rtol < 0.0 or self.atol < 0.0:
            raise ValueError("parity tolerances must be non-negative")
        if self.mode == "byte_exact" and (self.rtol != 0.0 or self.atol != 0.0):
            raise ValueError("byte_exact parity requires zero tolerances")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")


def build_workload_contract(
        input_manifest: Path | str, *, target: Mapping[str, object],
        vertical: Mapping[str, object], state_inventory: Sequence[str],
        forcing_times: Sequence[str],
) -> dict[str, object]:
    """Bind all backend-independent inputs to one canonical workload hash.

    Backend name and worker count are intentionally absent.  CUDA and CPU
    receipts can therefore prove that they consumed exactly the same source,
    target geometry, arbitrary eta coordinate, state inventory, and times.
    """

    manifest_path = Path(input_manifest)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    inventory = tuple(state_inventory)
    if not inventory or any(not isinstance(name, str) or not name
                            for name in inventory):
        raise ValueError("state_inventory must contain non-empty names")
    if len(set(inventory)) != len(inventory):
        raise ValueError("state_inventory contains duplicates")
    times = tuple(forcing_times)
    if not times or any(not isinstance(value, str) or not value
                        for value in times):
        raise ValueError("forcing_times must contain non-empty timestamps")
    payload: dict[str, object] = {
        "schema": WORKLOAD_SCHEMA,
        "source_manifest": {
            "bytes": manifest_path.stat().st_size,
            "sha256": _sha256(manifest_path),
        },
        "target": dict(target),
        "vertical": dict(vertical),
        "state_inventory": list(inventory),
        "forcing_times": list(times),
    }
    encoded = _canonical_json(payload)
    payload["workload_sha256"] = hashlib.sha256(encoded).hexdigest()
    # Re-encode after identity insertion solely to reject any nested NaN/Inf.
    _canonical_json(payload)
    return payload


def array_receipt(value) -> dict[str, object]:
    """Return a deterministic identity for one C-contiguous array."""

    if hasattr(value, "get"):
        value = value.get()
    array = np.ascontiguousarray(value)
    if array.dtype.hasobject:
        raise TypeError("object arrays cannot enter backend receipts")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("backend receipts require numeric arrays")
    if not np.isfinite(array).all():
        raise ValueError("backend output contains non-finite values")
    raw = memoryview(array).cast("B")
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "bytes": raw.nbytes,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def build_backend_receipt(
        workload: Mapping[str, object], *, backend: str,
        implementation_sha256: str, workers: int,
        outputs: Mapping[str, object],
) -> dict[str, object]:
    if workload.get("schema") != WORKLOAD_SCHEMA:
        raise ValueError("unrecognized preprocessing workload schema")
    if not isinstance(backend, str) or not backend:
        raise ValueError("backend name is empty")
    if not isinstance(implementation_sha256, str) \
            or len(implementation_sha256) != 64:
        raise ValueError("implementation_sha256 must be a SHA-256 hex digest")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if not outputs:
        raise ValueError("backend receipt has no outputs")
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "PASS",
        "workload_sha256": workload.get("workload_sha256"),
        "backend": backend,
        "implementation_sha256": implementation_sha256.lower(),
        "workers": workers,
        "outputs": {
            name: array_receipt(value)
            for name, value in sorted(outputs.items())
        },
    }
    _canonical_json(receipt)
    return receipt


def compare_backend_outputs(
        reference: Mapping[str, object], candidate: Mapping[str, object], *,
        rules: Mapping[str, ArrayParityRule],
) -> dict[str, object]:
    """Recompute cross-backend parity from arrays, never editable summaries."""

    names = set(reference)
    if names != set(candidate) or names != set(rules):
        raise ValueError(
            "reference, candidate, and parity-rule inventories differ")
    fields: dict[str, object] = {}
    overall = True
    for name in sorted(names):
        left = np.ascontiguousarray(
            reference[name].get() if hasattr(reference[name], "get")
            else reference[name])
        right = np.ascontiguousarray(
            candidate[name].get() if hasattr(candidate[name], "get")
            else candidate[name])
        rule = rules[name]
        if left.shape != right.shape or left.dtype != right.dtype:
            passed = False
            maximum_absolute = None
            maximum_relative = None
        elif not np.isfinite(left).all() or not np.isfinite(right).all():
            passed = False
            maximum_absolute = None
            maximum_relative = None
        elif rule.mode == "byte_exact":
            passed = memoryview(left).cast("B") == memoryview(right).cast("B")
            maximum_absolute = 0.0 if passed else float(
                np.max(np.abs(left.astype(np.float64)
                              - right.astype(np.float64))))
            maximum_relative = 0.0 if passed else None
        else:
            delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
            scale = np.maximum(np.abs(left.astype(np.float64)),
                               np.abs(right.astype(np.float64)))
            bound = rule.atol + rule.rtol * scale
            passed = bool(np.all(delta <= bound))
            maximum_absolute = float(np.max(delta))
            maximum_relative = float(np.max(
                np.divide(delta, scale, out=np.zeros_like(delta),
                          where=scale != 0.0)))
        fields[name] = {
            "status": "PASS" if passed else "FAIL",
            "rule": asdict(rule),
            "shape": list(left.shape),
            "dtype": left.dtype.str,
            "max_abs": maximum_absolute,
            "max_rel": maximum_relative,
        }
        overall &= passed
    return {
        "schema": PARITY_SCHEMA,
        "status": "PASS" if overall else "FAIL",
        "fields": fields,
    }


__all__ = [
    "ArrayParityRule",
    "PARITY_SCHEMA",
    "RECEIPT_SCHEMA",
    "WORKLOAD_SCHEMA",
    "array_receipt",
    "build_backend_receipt",
    "build_workload_contract",
    "compare_backend_outputs",
]
