"""The DA interface: applying analysis increments (EXPERIMENTAL).

The assimilation step -- whoever implements it -- hands the engine a
plain ``dict[field_name -> ndarray]`` whose arrays match the state's
shapes exactly.  This module is the only thing that touches state on the
DA side, and it is deliberately dumb: add, check, receipt.  It does no
balancing, no localisation, no covariance work of any kind.

Increments are checked before anything is written: an unknown field, a
shape mismatch, a non-finite value, or a dtype that is not a float kind
is a refusal, not a warning.  Half-applied increments are the one
outcome that would silently corrupt a cycling ensemble, so the checks
run over the whole dict first and only then does the apply loop start.

**Finiteness is checked where the value lands, not only where it came
from.**  A ``float64`` increment of ``1e300`` is finite; cast to the
``float32`` a state field actually stores it is ``inf``.  So is a
``float32``-representable increment added to a large enough background.
Validating only the source dtype let both through and wrote ``inf`` into
the analysis, which is exactly the outcome the "a non-finite value is a
refusal" contract exists to prevent.  Every write path therefore checks
the cast addend AND the sum, in the target's own dtype, before anything
is stored -- and computes the sum into a temporary first, so a refusal
leaves the target byte-identical rather than half-poisoned.

**Moment consistency is checked here for the same reason finiteness is.**
This module is the only thing that writes a DA analysis, so it is the
only place a guard cannot be skipped by using a different driver.  A
multi-moment state's mass and number are a pair; an update that moves one
and not the other produces cells holding ``q > 0`` with ``N = 0``, which
the scheme's slope closure evaluates to NaN and the reflectivity operator
correctly refuses to smooth over.  Every write therefore validates the
increment's field set against the pairs the background actually carries
and, after the sum, repairs or refuses the pairs the analysis broke --
see :mod:`gpuwm.da.moments` for the authority each scheme's repair comes
from and for the real-radar cycle that made this necessary.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path
from typing import Mapping

import numpy as np

#: Versioned contract label recorded in every increment receipt.
INCREMENT_CONTRACT = "gpuwm-da-increments.v1"


def _moment_guard(resulting: Mapping[str, object],
                  updated_fields, available_fields, *,
                  moment_policy: str, moment_repair: bool,
                  mp_physics: int | None, morr_rimed_ice: int,
                  where: str) -> tuple[dict, dict]:
    """``(number fields to overwrite, receipt)`` for one analysis.

    Three checks, in the order the failure happens.  First the field set:
    an update that moves a mass field and leaves a paired moment the
    background carries is refused under ``full-moment`` before anything
    is written.  Then the RESULT, in two parts.

    A moment that is not FINITE where the scheme reads it is refused
    whatever the policy and whatever ``moment_repair`` says, because no
    limiter repairs a NaN and IEEE comparison hides one from every
    ordering test the depleted-number check makes: ``qr = 1e-3`` beside
    ``nr = NaN`` is not above-threshold-with-a-non-positive-number, and
    was attested consistent.  ``_require_finite_result`` does not cover
    it either -- it checks the fields the INCREMENT names, and the
    pair's other half is exactly the field the increment did not name.

    Then the depleted pairs: a cell holding mass above the scheme's
    activity threshold with a non-positive number is repaired through the
    scheme's own limiter, or -- with the repair switched off, or for a
    scheme whose limiter is not ported -- is a refusal naming the counts.
    """
    from gpuwm.da.moments import (moment_consistency_report,
                                  nonfinite_moment_refusal, pairs_present,
                                  repair_moments, validate_analysis_fields)

    policy_receipt = validate_analysis_fields(
        tuple(updated_fields), available=tuple(available_fields),
        mp_physics=mp_physics, policy=moment_policy)
    pairs = pairs_present(tuple(available_fields), mp_physics=mp_physics)
    if not pairs:
        return {}, {**policy_receipt, "pairs_checked": [],
                    "offending_cells_total": 0, "nonfinite_cells_total": 0,
                    "consistent": True, "repaired": False}
    finite_report = moment_consistency_report(
        resulting, mp_physics=mp_physics, pairs=pairs)
    if finite_report["nonfinite_cells_total"]:
        raise ValueError(nonfinite_moment_refusal(finite_report, where=where))
    if not moment_repair:
        report = finite_report
        if not report["consistent"]:
            raise ValueError(
                f"the analysis for {where} leaves "
                f"{report['offending_cells_total']} cell(s) holding mass "
                f"above {report['q_threshold_kg_kg']:g} kg/kg with a number "
                "moment at or below zero ("
                + ", ".join(
                    f"{entry['mass_field']}: {entry['offending_cells']}"
                    for entry in report["species"]
                    if entry["offending_cells"])
                + "), and moment_repair is off. That state is one the "
                "scheme's own slope closure evaluates to NaN, so the "
                "reflectivity operator will refuse it rather than invent a "
                "clear-air floor. Enable the repair or analyse the number "
                "moments with the mass.")
        return {}, {**policy_receipt, **report, "repaired": False}
    repaired, report = repair_moments(resulting, mp_physics=mp_physics,
                                      pairs=pairs,
                                      morr_rimed_ice=morr_rimed_ice)
    return repaired, {**policy_receipt, **report}


def _host(value) -> np.ndarray:
    get = getattr(value, "get", None)
    if callable(get) and hasattr(value, "__cuda_array_interface__"):
        return np.ascontiguousarray(get())
    return np.ascontiguousarray(np.asarray(value))


def _array_sha256(value) -> str:
    host = _host(value)
    digest = hashlib.sha256()
    digest.update(str(host.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(repr(host.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(host.tobytes())
    return digest.hexdigest()


def _validate(name: str, target, increment, *, where: str) -> np.ndarray:
    if target is None:
        raise ValueError(
            f"increment names field {name!r}, which {where} does not "
            "carry; the DA interface is keyed by prognostic field name "
            "and every key must resolve")
    checked = np.asarray(increment)
    target_shape = tuple(np.shape(target))
    if checked.shape != target_shape:
        raise ValueError(
            f"increment for {name!r} has shape {checked.shape}, but "
            f"{where} field has shape {target_shape}; increments must "
            "match the state's shapes exactly")
    if checked.dtype.kind != "f":
        raise ValueError(
            f"increment for {name!r} has dtype {checked.dtype}; "
            "increments must be a float kind")
    if not np.all(np.isfinite(checked)):
        raise ValueError(
            f"increment for {name!r} contains non-finite values; a NaN "
            "or Inf increment would poison the whole member")
    return checked


def _nonfinite_count(value) -> int:
    """How many entries of ``value`` are not finite.  Backend-agnostic."""
    module = _array_module(value)
    return int(module.count_nonzero(~module.isfinite(value)))


def _array_module(value):
    if hasattr(value, "__cuda_array_interface__"):
        import cupy as cp

        return cp
    return np


def _require_finite_result(name: str, target, addend, updated, *,
                           where: str) -> None:
    """Refuse an addend or a sum that is not finite in the target's dtype.

    ``_validate`` proved the increment finite in the dtype the caller
    handed over.  This proves it is still finite after the cast the write
    performs, and that the sum it produces is finite too -- an increment
    that overflows on cast, or that overflows against the background it
    lands on, is a poisoned analysis and not a rounding detail.
    """
    dtype = getattr(target, "dtype", None)
    cast_bad = _nonfinite_count(addend)
    if cast_bad:
        raise ValueError(
            f"increment for {name!r} is finite as given but has "
            f"{cast_bad} non-finite value(s) once cast to the {dtype} of "
            f"{where} field; a cast that overflows is a refusal, not a "
            "silent inf")
    sum_bad = _nonfinite_count(updated)
    if sum_bad:
        raise ValueError(
            f"increment for {name!r} is finite and casts finitely, but "
            f"adding it to {where} field produces {sum_bad} non-finite "
            "value(s); the analysis would carry inf/nan into the next "
            "leg, so this is a refusal")


def _sorted_items(increments: Mapping[str, object]):
    if not isinstance(increments, Mapping):
        raise TypeError(
            "increments must be a mapping of field name to ndarray, got "
            f"{type(increments).__name__}")
    if not increments:
        raise ValueError(
            "increments mapping is empty; an assimilation step that "
            "changes nothing must be recorded as such by its caller, not "
            "passed here as an empty dict")
    # Sorted so the receipt is order-independent: two callers handing the
    # same increments in different dict orders get the same receipt.
    return sorted(increments.items())


def apply_increments(state, increments: Mapping[str, object], *,
                     moment_policy: str = "full-moment",
                     moment_repair: bool = True,
                     mp_physics: int | None = None,
                     morr_rimed_ice: int = 1) -> dict:
    """Add ``increments`` to the matching attributes of a live ``state``.

    Returns a receipt: per-field increment sha256 and the resulting
    field sha256, plus the pre/post whole-state hashes and the
    moment-consistency block.
    """
    from gpuwm.ensemble.state_sha import (live_state_sha256,
                                          serialized_state_attrs)

    items = _sorted_items(increments)
    checked = {}
    for name, increment in items:
        target = getattr(state, name, None)
        checked[name] = _validate(name, target, increment, where="the state")

    # Two passes: prove every field's cast addend and resulting sum are
    # finite before ANY field is written.  A refusal on the last field
    # must not leave the first three applied.
    staged = {}
    # The overflow is the thing being detected, so numpy's warning about
    # it is noise: the refusal below is the report.
    with np.errstate(over="ignore", invalid="ignore"):
        for name, increment in checked.items():
            target = getattr(state, name)
            addend = _as_target_array(target, increment)
            updated = target + addend
            _require_finite_result(name, target, addend, updated,
                                   where="the state")
            staged[name] = updated

    # The moment guard sees the RESULT, not the increment: a pair breaks
    # where prior + increment lands, and the whole point is the cell the
    # background left clear.
    available = tuple(name for name in serialized_state_attrs()
                      if getattr(state, name, None) is not None)
    resulting = {name: staged.get(name, getattr(state, name))
                 for name in available}
    repaired, moments = _moment_guard(
        resulting, tuple(checked), available,
        moment_policy=moment_policy, moment_repair=moment_repair,
        mp_physics=mp_physics, morr_rimed_ice=morr_rimed_ice,
        where="this state")

    before = live_state_sha256(state)
    fields = []
    for name, increment in checked.items():
        target = getattr(state, name)
        target[...] = staged[name]
        fields.append({
            "field": name,
            "shape": list(np.shape(target)),
            "increment_sha256": _array_sha256(increment),
            "field_sha256": _array_sha256(target),
        })
    for name, values in repaired.items():
        target = getattr(state, name)
        target[...] = _as_target_array(target, values)
    return {
        "contract": INCREMENT_CONTRACT,
        "stability": "experimental",
        "field_count": len(fields),
        "fields": fields,
        "moments": moments,
        "state_sha256_before": before,
        "state_sha256_after": live_state_sha256(state),
    }


def _as_target_array(target, increment: np.ndarray):
    """``increment`` cast to the target's dtype and memory space."""
    dtype = getattr(target, "dtype", None)
    if hasattr(target, "__cuda_array_interface__"):
        import cupy as cp

        return cp.asarray(increment, dtype=dtype)
    return np.asarray(increment, dtype=dtype)


#: Suffix of the file an unpublished analysis is staged at.  Named, not
#: incidental: :func:`publish_staged_analysis` is the only thing that
#: turns one into an ``analysis.npz``, and a leftover ``.staged`` file is
#: the visible trace of a crash between staging and publication.
STAGED_SUFFIX = ".staged"


def publish_staged_analysis(staged: str | Path,
                            destination: str | Path) -> Path:
    """Rename a staged analysis into place.  The commit half of the seam."""
    src = Path(staged)
    dst = Path(destination)
    if not src.is_file():
        raise ValueError(
            f"no staged analysis at {src}; nothing to publish")
    src.replace(dst)
    return dst


def apply_increments_to_checkpoint(
        source: str | Path, increments: Mapping[str, object],
        destination: str | Path, *, publish: bool = True,
        moment_policy: str = "full-moment",
        moment_repair: bool = True,
        mp_physics: int | None = None,
        morr_rimed_ice: int = 1) -> dict:
    """Write a copy of a checkpoint with ``increments`` added.

    The source checkpoint is never modified: the analysis is a new file,
    so a failed assimilation always leaves the background recoverable.
    Every key of the source is carried through unchanged except the
    ``state/<field>`` arrays that the increments name.

    ``publish=False`` writes the analysis to ``destination`` +
    :data:`STAGED_SUFFIX` and stops there, so a caller with more than one
    member to analyse can prove every member's write succeeded before any
    member's ``analysis.npz`` becomes visible.  The receipt then carries
    ``published: false`` and ``staged``; :func:`publish_staged_analysis`
    is the commit.  A one-member caller has no use for it and the default
    is the whole operation.
    """
    from gpuwm.ensemble.state_sha import (checkpoint_state_sha256,
                                          serialized_state_attrs)

    src = Path(source)
    dst = Path(destination)
    if not src.is_file():
        raise ValueError(f"no checkpoint at {src}")
    if dst.resolve() == src.resolve():
        raise ValueError(
            f"refusing to write the analysis over its own background "
            f"{src}; pass a distinct destination")
    items = _sorted_items(increments)

    with np.load(src, allow_pickle=False) as data:
        payload = {key: data[key] for key in data.files}

    contract = set(serialized_state_attrs())
    checked = {}
    for name, increment in items:
        if name not in contract:
            raise ValueError(
                f"increment names field {name!r}, which is not in the "
                "restart prognostic contract; the DA interface may only "
                "rewrite serialised prognostic state")
        key = f"state/{name}"
        if key not in payload:
            raise ValueError(
                f"increment names field {name!r}, which checkpoint {src} "
                "does not carry")
        checked[name] = _validate(name, payload[key], increment,
                                  where=f"checkpoint {src}")

    before = checkpoint_state_sha256(src)
    staged = {}
    with np.errstate(over="ignore", invalid="ignore"):
        for name, increment in checked.items():
            key = f"state/{name}"
            original = payload[key]
            addend = increment.astype(original.dtype, copy=False)
            updated = np.asarray(original + addend, dtype=original.dtype)
            _require_finite_result(name, original, addend, updated,
                                   where=f"checkpoint {src}")
            staged[name] = updated

    # The moment guard reads the RESULT the checkpoint is about to carry,
    # over every state array the file holds -- not only the ones the
    # increment names, because the pair's other half is exactly the one
    # the increment did not name.
    available = tuple(key[len("state/"):] for key in payload
                      if key.startswith("state/"))
    resulting = {name: staged.get(name, payload[f"state/{name}"])
                 for name in available}
    repaired, moments = _moment_guard(
        resulting, tuple(checked), available,
        moment_policy=moment_policy, moment_repair=moment_repair,
        mp_physics=mp_physics, morr_rimed_ice=morr_rimed_ice,
        where=f"checkpoint {src.name}")

    fields = []
    for name, increment in checked.items():
        key = f"state/{name}"
        updated = staged[name]
        payload[key] = updated
        fields.append({
            "field": name,
            "shape": list(updated.shape),
            "increment_sha256": _array_sha256(increment),
            "field_sha256": _array_sha256(updated),
        })
    for name, values in repaired.items():
        key = f"state/{name}"
        original = payload[key]
        payload[key] = np.asarray(values, dtype=original.dtype)

    dst.parent.mkdir(parents=True, exist_ok=True)
    landed = dst if publish else dst.with_name(dst.name + STAGED_SUFFIX)
    # Unique tmp name: a fixed one is a collision between two writers
    # aimed at the same destination, and the loser silently wins.
    tmp = landed.with_name(f"{landed.name}.{os.getpid()}."
                           f"{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp, "wb") as stream:
            np.savez(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        tmp.replace(landed)
    finally:
        if tmp.exists():
            tmp.unlink()
    receipt = {
        "contract": INCREMENT_CONTRACT,
        "stability": "experimental",
        "source": str(src),
        "destination": str(dst),
        "published": bool(publish),
        "field_count": len(fields),
        "fields": fields,
        "moments": moments,
        "state_sha256_before": before,
        "state_sha256_after": checkpoint_state_sha256(landed),
    }
    if not publish:
        receipt["staged"] = str(landed)
    return receipt
