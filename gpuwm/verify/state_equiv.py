"""Executable peer-FP32 state comparator for the pre-registered N2 gates.

The predicate is owned by ``verify.nest_gates.COMPARATORS``.  This module
contains no case-specific tolerances: all three numeric guards are imported
from that ledger and applied both per field and over the concatenated state.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from gpuwm.verify.nest_gates import (
    FP32_STATE_EQUIV_MAX_ULPS,
    FP32_STATE_EQUIV_MEAN_SIGNED_ULP_MAX_ABS,
    FP32_STATE_EQUIV_P99_MAX_ULPS,
)


def _field(state, name: str):
    if isinstance(state, Mapping):
        return state[name]
    return getattr(state, name)


def _fp32_host(value) -> np.ndarray:
    """Return the peer state exactly as stored/rounded to FP32 on the host."""
    if hasattr(value, "get"):
        value = value.get()
    return np.asarray(value, dtype=np.float32)


def _monotone_fp32_key(value: np.ndarray) -> np.ndarray:
    """Map FP32 bit patterns to increasing unsigned integer keys.

    Negative patterns are complemented and non-negative patterns have their
    sign bit flipped.  Thus every adjacent finite FP32 pattern has adjacent
    keys, including the two signed-zero encodings, and integer subtraction
    has the registered candidate-minus-reference sign.
    """
    bits = np.ascontiguousarray(value, dtype=np.float32).view(np.uint32)
    negative = (bits & np.uint32(0x80000000)) != 0
    return np.where(negative, ~bits, bits ^ np.uint32(0x80000000))


def fp32_signed_ulp(candidate, reference) -> np.ndarray:
    """Signed candidate-minus-reference ULP distance under N2 semantics.

    This is the ONE certified ULP definition: every receipt that reports a
    ULP against the ``fp32_state_equiv`` ceilings computes it here, so a
    second implementation cannot drift away from the comparator that owns
    the verdict.  Both operands are taken exactly as stored on the host in
    FP32, mapped to monotone unsigned keys, and subtracted in int64 -- so
    adjacent finite patterns differ by one, the two signed-zero encodings
    are adjacent, and the sign is candidate minus reference.

    The result has the shared shape of the inputs; a shape disagreement is
    an error rather than a broadcast, because a silently broadcast
    comparison would score a different array than the one named.
    """
    candidate_fp32 = _fp32_host(candidate)
    reference_fp32 = _fp32_host(reference)
    if candidate_fp32.shape != reference_fp32.shape:
        raise ValueError(
            f"shape mismatch: candidate {candidate_fp32.shape}, "
            f"reference {reference_fp32.shape}")
    return (_monotone_fp32_key(candidate_fp32).astype(np.int64)
            - _monotone_fp32_key(reference_fp32).astype(np.int64))


def _failed(*, count: int = 0, reason: str) -> dict[str, object]:
    return {
        "count": int(count),
        "max_ulp": None,
        "p99_ulp": None,
        "mean_signed_ulp": None,
        "pass": False,
        "reason": reason,
    }


def _metrics(signed_ulp: np.ndarray) -> dict[str, object]:
    signed = np.asarray(signed_ulp, dtype=np.int64).reshape(-1)
    count = int(signed.size)
    if count == 0:
        return _failed(reason="empty comparison")
    ulp = np.abs(signed)
    rank = math.ceil(0.99 * count) - 1
    max_ulp = int(np.max(ulp))
    p99_ulp = int(np.sort(ulp)[rank])
    mean_signed_ulp = float(np.mean(signed, dtype=np.float64))
    passed = (
        max_ulp <= FP32_STATE_EQUIV_MAX_ULPS
        and p99_ulp <= FP32_STATE_EQUIV_P99_MAX_ULPS
        and abs(mean_signed_ulp)
        <= FP32_STATE_EQUIV_MEAN_SIGNED_ULP_MAX_ABS
    )
    return {
        "count": count,
        "max_ulp": max_ulp,
        "p99_ulp": p99_ulp,
        "mean_signed_ulp": mean_signed_ulp,
        "pass": bool(passed),
        "reason": None,
    }


def fp32_state_equiv_report(candidate_state, reference_state,
                             field_names: Sequence[str]
                             ) -> dict[str, object]:
    """Compare named peer-state fields under registered N2 semantics.

    The result is JSON-ready and contains per-field metrics plus the same
    metrics over every element concatenated.  Missing fields, unequal shapes,
    empty fields, and non-finite values are represented as blocking failures;
    they are never silently omitted from the aggregate verdict.
    """
    names = tuple(field_names)
    fields: dict[str, dict[str, object]] = {}
    signed_parts: list[np.ndarray] = []
    invalid_reason = None
    if not names:
        invalid_reason = "empty field inventory"

    for name in names:
        try:
            candidate = _fp32_host(_field(candidate_state, name))
            reference = _fp32_host(_field(reference_state, name))
        except (AttributeError, KeyError) as exc:
            fields[name] = _failed(reason=f"missing field: {exc}")
            invalid_reason = invalid_reason or f"field {name!r} is missing"
            continue
        if candidate.shape != reference.shape:
            fields[name] = _failed(
                count=int(candidate.size),
                reason=(f"shape mismatch: candidate {candidate.shape}, "
                        f"reference {reference.shape}"))
            invalid_reason = invalid_reason or f"field {name!r} shape mismatch"
            continue
        if candidate.size == 0:
            fields[name] = _failed(reason="empty comparison")
            invalid_reason = invalid_reason or f"field {name!r} is empty"
            continue
        if not (np.isfinite(candidate).all() and np.isfinite(reference).all()):
            fields[name] = _failed(
                count=int(candidate.size), reason="non-finite value")
            invalid_reason = invalid_reason or f"field {name!r} is non-finite"
            continue

        signed = fp32_signed_ulp(candidate, reference)
        fields[name] = _metrics(signed)
        signed_parts.append(signed.reshape(-1))

    if invalid_reason is not None or not signed_parts:
        aggregate = _failed(
            count=sum(int(part.size) for part in signed_parts),
            reason=invalid_reason or "empty comparison")
    else:
        aggregate = _metrics(np.concatenate(signed_parts))
    return {
        "fields": fields,
        "aggregate": aggregate,
        # COMPARATORS defines the verdict over all compared elements.  The
        # per-field passes are diagnostics, not extra (stricter) gates.
        "pass": bool(aggregate["pass"]),
    }


__all__ = ["fp32_signed_ulp", "fp32_state_equiv_report"]
