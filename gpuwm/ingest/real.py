"""WRF-real-like moist initialization on an explicit hybrid eta grid.

Setup and hydrostatic recurrences are float64.  The pressure-column vertical
interpolations follow WRF real's ratified defaults (interp_theta=F,
lagrange_order=2, use_surface=T with force_sfc_in_vinterp=1 and
zap_close_levels=500) through the common CUDA/parallel-CPU preprocessing
contract, and the completed prognostic state is FP32 on device.
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Mapping, MutableMapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import hashlib
import json
import sys
from time import perf_counter
import zlib

import numpy as np

from gpuwm.config import RunConfig, validate_aerosol_source_options
from gpuwm.core import constants as c
from gpuwm.core.grid import (BaseState, VerticalCoord,
                             finalize_vertical_coord,
                             hybrid_column_ordering_refusal)
from gpuwm.core.state import DomainState
from gpuwm.ingest.horiz import (
    HorizontalSnapshot,
    source_orography_from_catalog as _source_orography_from_catalog,
)
from gpuwm.ingest.preprocess_backend import resolve_preprocess_backend


def _column_worker_count(value) -> int:
    """Validate an explicit setup-only CPU column-worker count."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError("column_workers must be an integer")
    value = int(value)
    if value < 1:
        raise ValueError("column_workers must be positive")
    return value


def _axis0_chunks(length: int, workers: int):
    """Return non-empty, deterministic contiguous chunks of axis zero."""
    workers = min(int(workers), int(length))
    q, r = divmod(int(length), workers)
    start = 0
    chunks = []
    for index in range(workers):
        stop = start + q + (index < r)
        chunks.append((start, stop))
        start = stop
    return tuple(chunks)


def _ordered_levels(array, order):
    """Apply a vertical order without copying monotonic source inventories."""
    array = np.asarray(array)
    order = np.asarray(order)
    identity = np.arange(array.shape[0], dtype=order.dtype)
    if np.array_equal(order, identity):
        return array
    if np.array_equal(order, identity[::-1]):
        return array[::-1]
    return array[order]


def _fill_axis0_chunk(output, start, stop, operation, args, kwargs=None):
    """Evaluate and immediately store one disjoint threaded output slab."""
    output[start:stop] = operation(*args, **({} if kwargs is None else kwargs))


def _host(value) -> np.ndarray:
    if hasattr(value, "get"):
        value = value.get()
    return np.asarray(value, dtype=np.float64)


HRRR_ANALYZED_HYDROMETEORS = ("QC", "QR", "QI", "QS", "QG")

HRRR_HYDROMETEOR_CORRESPONDENCE_SCHEMA_V1 = (
    "gpuwm-real-hydrometeor-correspondence-v1")
HRRR_HYDROMETEOR_CORRESPONDENCE_SCHEMA_V2 = (
    "gpuwm-real-hydrometeor-correspondence-v2")
HRRR_HYDROMETEOR_VERTICAL_DISPOSITION_SCHEMA = (
    "gpuwm-wrf-real-hydrometeor-vertical-disposition-v1")

_HRRR_DISPOSITION_CLASSES = {
    "TARGET_INFLUENCING": 1,
    "WRF_FORCE_SURFACE_EXCLUDED": 2,
    "WRF_ZAP_CLOSE_EXCLUDED": 3,
    "WRF_BELOW_GROUND_OUTSIDE_TARGET_SUPPORT": 4,
    "WRF_ABOVE_TARGET_TOP_OUTSIDE_TARGET_SUPPORT": 5,
    "WRF_NO_TARGET_STENCIL": 6,
}
_HRRR_DISPOSITION_CLASS_BY_CODE = {
    value: name for name, value in _HRRR_DISPOSITION_CLASSES.items()}
_HRRR_DISPOSITION_ENCODING = "base64-zlib-uint8-c-order-v1"
_HRRR_PRODUCTION_SUPPORT_SCHEMA = (
    "gpuwm-wrf-q-source-level-production-support-v1")
_HRRR_PRODUCTION_SUPPORT_ENCODING = (
    "base64-zlib-packbits-little-c-order-v1")
_HRRR_DISPOSITION_EXAMPLE_LIMIT = 16
_HRRR_DISPOSITION_OPERATOR = {
    "schema": "wrf-v4.6.1-real-linear-q-vertical-operator-v1",
    "wrf_version": "v4.6.1",
    "wrf_commit": "d66e442fccc04111067e29274c9f9eaccc3cef28",
    "source": "dyn_em/module_initialize_real.F:5664-6574",
    "use_surface": True,
    "surface_value": "exact-fp32-zero",
    "use_levels_below_ground": True,
    "force_sfc_in_vinterp": 1,
    "zap_close_levels_pa": 500.0,
    "zap_predicate": "pressure_separation < 500 Pa",
    "interp_in_logp": True,
    "lagrange_order": 2,
    "target_operator": "linear-at-every-target-vboundb=ntarget+1",
    "below_ground_extrapolation": "constant-deepest-assembled-point",
    "above_source_top": "fatal",
}

#: The microphysics ids whose WRF Registry ``moist:`` package can receive the
#: decoded HRRR analyzed hydrometeor inventory, so the decoder must supply it.
#:
#: This is ONE tuple on purpose.  It was three separate literals, and mp=28
#: was added to none of them: an mp=28 real-data run reached
#: :func:`initialize_real` with QC/QR/QI/QS/QG present in the snapshot, never
#: declared them required, never shape-checked them, never interpolated them,
#: and produced a condensate-free initial state from a cloudy analysis with
#: no error anywhere.  A single name makes that class of omission impossible.
#:
#: Membership is decided by the Registry package, per id, WRF v4.6.1
#: (commit d66e442fccc04111067e29274c9f9eaccc3cef28):
#:   1  kesslerscheme   Registry.EM_COMMON:3015  moist:qv,qc,qr
#:   6  wsm6scheme      Registry.EM_COMMON:3021  moist:qv,qc,qr,qi,qs,qg
#:   8  thompson        Registry.EM_COMMON:3024  moist:qv,qc,qr,qi,qs,qg
#:  10  morr_two_moment Registry.EM_COMMON:3026  moist:qv,qc,qr,qi,qs,qg
#:  18  nssl_2mom       Registry.EM_COMMON:3033  moist:qv,qc,qr,qi,qs,qg
#:  28  thompsonaero    Registry.EM_COMMON:3036  moist:qv,qc,qr,qi,qs,qg
#: Kessler is a member because real.exe still requires the decoded inventory
#: and then discards what its package lacks (WRF_REAL_KESSLER_FROZEN_POLICY);
#: mp_physics=0 is refused outright above.  mp=28's moist list is character
#: for character mp=8's -- the aerosol-aware scheme adds only ``scalar:``
#: members -- so the mass-species handling is the same at all three sites,
#: and that identity was verified against :3024 and :3036 directly rather
#: than assumed from the mp=8 arm.
HRRR_ANALYZED_HYDROMETEOR_MP_PHYSICS = (1, 6, 8, 10, 16, 18, 28)

# WRF v4.6.1, commit d66e442fccc04111067e29274c9f9eaccc3cef28:
# Registry/Registry.EM_COMMON:3015 gives Kessler only moist:qv,qc,qr.
# dyn_em/module_initialize_real.F:1859-1977 separately interpolates QR/QC/
# QI/QS/QG only after finding each P_Q* in the active num_moist package.
# Thus real.exe retains HRRR QC/QR for Kessler and deliberately does not
# carry analyzed QI/QS/QG into a scheme whose Registry package lacks them.
WRF_REAL_KESSLER_FROZEN_POLICY = {
    "policy": "discard-source-species-absent-from-active-moist-package",
    "wrf_version": "v4.6.1",
    "wrf_commit": "d66e442fccc04111067e29274c9f9eaccc3cef28",
    "registry_citation": "Registry/Registry.EM_COMMON:3015",
    "real_citation": "dyn_em/module_initialize_real.F:1859-1977",
}

#: What a real-data mp_physics=28 initial state does and does NOT contain,
#: as a receipt rather than as a comment: an mp=28 aerosol field that is
#: silently zero and an mp=28 aerosol field that is deliberately zero look
#: identical in the state, and only one of them is correct.
#:
#: ``deferred_to`` is the load-bearing entry.  Nothing in this module fills
#: nwfa/nifa/nwfa2d; the fill is a one-time per-domain step that belongs to
#: the physics init path, exactly as WRF calls ``thompson_init`` from
#: ``phys/module_physics_init.F`` and not from ``module_initialize_real.F``.
WRF_REAL_MP28_AEROSOL_SOURCE_POLICY = {
    "policy": "aer-init-opt-0-zero-then-thompson-init-synthetic-profile",
    "wrf_version": "v4.6.1",
    "wrf_commit": "d66e442fccc04111067e29274c9f9eaccc3cef28",
    "registry_citation": "Registry/Registry.EM_COMMON:3036",
    "real_citation": "dyn_em/module_initialize_real.F:2332-2345",
    "microphysics_citation": (
        "phys/module_mp_thompson.F:493 (CCN MAXVAL test), :498-515 (CCN "
        "fill), :509-510 (nwfa2d), :531 (IN MAXVAL test), :536-551 (IN fill)"
    ),
    "wrf_real_refuses_this_configuration": (
        "dyn_em/module_initialize_real.F:2735-2736"),
    "deferred_to": (
        "gpuwm.core.physics.initialize_physics -> "
        "gpuwm.core.microphysics.microphysics_init"),
    "not_initialized_here": ("nwfa", "nifa", "nwfa2d", "nifa2d"),
    "zeroed_here": ("nc", "nr", "ni"),
}


def _warn_synthetic_aerosol_fallback(source_choice, resolution) -> None:
    """Announce the synthetic aerosol fallback on the way past it.

    The receipt is the durable record, but a receipt is read afterwards and
    this is a fact about what the run IS.  A user who expected a data-
    initialized forecast and got an analytic profile should learn it while
    the run is starting, not while reconciling numbers a week later, so the
    same named sentence also goes to the logger at WARNING.

    A DELIBERATE ``mp28_aerosol_source='synthetic'`` is not warned about at
    that level: nothing went wrong, the user asked for it.  It is still
    recorded in the receipt, because the receipt has to say what the run
    was regardless of who chose it.
    """
    from gpuwm.config import MP28_AEROSOL_SYNTHETIC_FALLBACK

    if source_choice == "synthetic":
        print("mp_physics=28 aerosol: synthetic profile selected by name "
              "(mp28_aerosol_source='synthetic'); no aerosol dataset was "
              "searched for.", file=sys.stderr)
        return
    detail = ""
    if resolution is not None and resolution.fallback_reason:
        detail = " " + str(resolution.fallback_reason)
    print(MP28_AEROSOL_SYNTHETIC_FALLBACK + detail, file=sys.stderr)


def _mp28_aerosol_source_policy(cfg: RunConfig, state) -> dict[str, object]:
    """Receipt for the mp=28 half of the source-absent initialization.

    Emitted as ``RealInitResult.aerosol_initialization`` so that a
    consumer can PROVE, from the returned object alone, three things a
    later reader would otherwise have to take on trust: which aerosol
    selectors this ingest ran under, that the two aerosol fields left this
    function at exact FP32 zero, and which named function is expected to
    fill them next.  The fingerprints are the same
    :func:`array_correspondence_fingerprint` identity used for the mass
    species, so an "already filled" state and an "awaiting fill" state are
    distinguishable after the fact and not only in the moment.
    """
    receipt = {
        **WRF_REAL_MP28_AEROSOL_SOURCE_POLICY,
        "aer_init_opt": int(cfg.aer_init_opt),
        "wif_input_opt": int(cfg.wif_input_opt),
        "awaiting_profile_fill": True,
    }
    receipt["source_absent_state_fields"] = {
        name: array_correspondence_fingerprint(getattr(state, name))
        for name in ("nc", "nr", "ni", "nwfa", "nifa", "nwfa2d", "nifa2d")
    }
    return receipt


def array_correspondence_fingerprint(value) -> dict[str, object]:
    """Return the byte/mask/extrema identity used by init receipts.

    The checksum covers exactly the contiguous array bytes.  Shape and dtype
    are separate fields, and the nonzero mask is independently packed and
    hashed so an all-zero replacement cannot masquerade as analyzed mass.
    """

    if hasattr(value, "get"):
        value = value.get()
    array = np.ascontiguousarray(np.asarray(value))
    if array.size == 0:
        raise ValueError("cannot fingerprint an empty correspondence array")
    mask = np.packbits(
        np.ravel(array != 0.0), bitorder="little")
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        "nonzero_mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
        "nonzero_count": int(np.count_nonzero(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _canonical_receipt_sha256(value: Mapping[str, object]) -> str:
    def plain(item):
        if isinstance(item, Mapping):
            return {str(key): plain(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [plain(child) for child in item]
        if isinstance(item, np.generic):
            return item.item()
        return item

    encoded = json.dumps(
        plain(value), sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _packed_mask_sha256(mask) -> str:
    packed = np.packbits(
        np.ravel(np.asarray(mask, dtype=np.bool_)), bitorder="little")
    return hashlib.sha256(packed.tobytes()).hexdigest()


def _zero_byte_sha256(length: int) -> str:
    digest = hashlib.sha256()
    block = b"\0" * min(length, 1024 * 1024)
    complete, remainder = divmod(length, len(block)) if block else (0, 0)
    for _ in range(complete):
        digest.update(block)
    if remainder:
        digest.update(block[:remainder])
    return digest.hexdigest()


def _host_float32(value) -> np.ndarray:
    if hasattr(value, "get"):
        value = value.get()
    return np.ascontiguousarray(value, dtype=np.float32)


def _hrrr_operator_geometry(
        source_pressure, surface_pressure, target_pressure, active_mask):
    """Classify WRF's assembled Q-column support in exact FP32 geometry.

    ``active_mask`` is the union of positive samples across retained source
    species.  Assembly is field-independent; restricting target-stencil
    searches to that union avoids turning sparse HRRR cloud evidence into a
    dense ``nsource * ntarget * ny * nx`` temporary.
    """

    source = _host_float32(source_pressure)
    surface = _host_float32(surface_pressure)
    target = _host_float32(target_pressure)
    active = np.ascontiguousarray(active_mask, dtype=np.bool_)
    if source.ndim != 3 or target.ndim != 3:
        raise ValueError(
            "hydrometeor pressure geometry must be (level, y, x)")
    if (surface.shape != source.shape[1:]
            or target.shape[1:] != source.shape[1:]
            or active.shape != source.shape):
        raise ValueError("hydrometeor disposition geometry shapes differ")
    if (not np.isfinite(source).all()
            or not np.isfinite(surface).all()
            or not np.isfinite(target).all()
            or np.any(source <= 0.0)
            or np.any(surface <= 0.0)
            or np.any(target <= 0.0)):
        raise ValueError("hydrometeor disposition pressure is invalid")
    if np.any(np.diff(source, axis=0) >= 0.0):
        raise ValueError(
            "hydrometeor disposition source pressure is not descending")
    if np.any(np.diff(target, axis=0) >= 0.0):
        raise ValueError(
            "hydrometeor disposition target pressure is not descending")

    nsource, ny, nx = source.shape
    ncolumn = ny * nx
    pd = source.reshape(nsource, ncolumn)
    psfc = surface.reshape(ncolumn)
    target_column = target.reshape(target.shape[0], ncolumn)
    active_column = active.reshape(nsource, ncolumn)
    above = pd < psfc[None, :]
    if np.any(~np.any(above, axis=0)):
        raise ValueError(
            "hydrometeor disposition has no source level above surface")
    first_above = np.argmax(above, axis=0).astype(np.int32)
    level = np.arange(nsource, dtype=np.int32)[:, None]
    force_candidate = (
        (level >= first_above[None, :])
        & (pd <= target_column[0][None, :]))
    has_force_candidate = np.any(force_candidate, axis=0)
    first_force_candidate = np.argmax(force_candidate, axis=0)
    knext = np.where(
        has_force_candidate, first_force_candidate, first_above).astype(
            np.int32)

    force_code = _HRRR_DISPOSITION_CLASSES[
        "WRF_FORCE_SURFACE_EXCLUDED"]
    zap_code = _HRRR_DISPOSITION_CLASSES["WRF_ZAP_CLOSE_EXCLUDED"]
    # 255 is deliberately invalid.  Assembly must replace every cell with
    # included/no-target-yet (0), force exclusion (2), or zap exclusion (3).
    operator_class = np.full((nsource, ncolumn), 255, dtype=np.uint8)
    has_below_ground = first_above > 0
    surface_lowest = ~has_below_ground
    last_surface_branch_pressure = psfc.copy()
    for k in range(nsource):
        pressure = pd[k]

        below = has_below_ground & (k < first_above)
        close_below = (
            below & (k == first_above - 1)
            & ((pressure - psfc) < np.float32(500.0)))
        operator_class[k, below & ~close_below] = 0
        operator_class[k, close_below] = zap_code

        above_surface = has_below_ground & (k >= first_above)
        forced = above_surface & (k < knext)
        close_above = (
            above_surface & (k == knext)
            & ((psfc - pressure) < np.float32(500.0)))
        operator_class[k, above_surface & ~forced & ~close_above] = 0
        operator_class[k, forced] = force_code
        operator_class[k, close_above] = zap_code

        forced_lowest = surface_lowest & (k < knext)
        candidate_lowest = surface_lowest & (k >= knext)
        close_lowest = (
            candidate_lowest
            & ((last_surface_branch_pressure - pressure)
               < np.float32(500.0))
            & (k < nsource - 1))
        accepted_lowest = candidate_lowest & ~close_lowest
        operator_class[k, accepted_lowest] = 0
        operator_class[k, forced_lowest] = force_code
        operator_class[k, close_lowest] = zap_code
        last_surface_branch_pressure = np.where(
            accepted_lowest, pressure, last_surface_branch_pressure)

    if np.any(operator_class == 255):
        raise AssertionError(
            "WRF hydrometeor column assembly left an unclassified level")

    # All hydrometeor targets are in WRF's linear branch (vboundb=ntarget+1).
    # Use FP32 log-pressure because both production implementations do.  A
    # source sample influences a target only when its computed linear weight
    # is nonzero; the strict endpoint inequalities below encode that fact.
    previous_x = np.full(ncolumn, np.nan, dtype=np.float32)
    has_previous = np.zeros(ncolumn, dtype=np.bool_)
    surface_x = np.log(surface).reshape(ncolumn)
    for k in range(nsource):
        insert_surface = first_above == k
        previous_x = np.where(insert_surface, surface_x, previous_x)
        has_previous |= insert_surface
        relevant = active_column[k] & (operator_class[k] == 0)
        columns = np.flatnonzero(relevant)
        if columns.size:
            if columns.size == ncolumn:
                current_x = np.log(pd[k])
                target_x_active = np.log(target_column)
            else:
                current_x = np.log(pd[k, columns])
                target_x_active = np.log(target_column[:, columns])
            prior = previous_x[columns]
            prior_exists = has_previous[columns]
            bracketed = np.any(
                (target_x_active < prior[None, :])
                & (target_x_active >= current_x[None, :]), axis=0)
            deepest = np.any(
                target_column[:, columns] >= pd[k, columns][None, :],
                axis=0)
            influences = np.where(prior_exists, bracketed, deepest)
            operator_class[k, columns[influences]] = 1
        accepted = operator_class[k] <= 1
        current_x_all = np.log(pd[k])
        previous_x = np.where(
            accepted, current_x_all, previous_x)
        has_previous |= accepted

    next_x = np.full(ncolumn, np.nan, dtype=np.float32)
    has_next = np.zeros(ncolumn, dtype=np.bool_)
    for k in range(nsource - 1, -1, -1):
        relevant = active_column[k] & (operator_class[k] <= 1)
        columns = np.flatnonzero(relevant)
        if columns.size:
            if columns.size == ncolumn:
                current_x = np.log(pd[k])
                target_x_active = np.log(target_column)
            else:
                current_x = np.log(pd[k, columns])
                target_x_active = np.log(target_column[:, columns])
            following = next_x[columns]
            following_exists = has_next[columns]
            bracketed = np.any(
                (target_x_active <= current_x[None, :])
                & (target_x_active > following[None, :]), axis=0)
            influences = following_exists & bracketed
            operator_class[k, columns[influences]] = 1
        accepted = operator_class[k] <= 1
        current_x_all = np.log(pd[k])
        next_x = np.where(accepted, current_x_all, next_x)
        has_next |= accepted
        insert_surface = first_above == k
        next_x = np.where(insert_surface, surface_x, next_x)
        has_next |= insert_surface

    return {
        "source_pressure": source,
        "surface_pressure": surface,
        "target_pressure": target,
        "first_above": first_above.reshape(ny, nx),
        "knext": knext.reshape(ny, nx),
        "operator_class": operator_class.reshape(source.shape),
    }


def _hrrr_disposition_example(
        *, raw_field, raw_labels, order, ordered_level, y, x,
        geometry) -> dict[str, object]:
    """Return inspectable pressure/stencil evidence for one source sample."""

    raw_level = int(order[ordered_level])
    source = geometry["source_pressure"][:, y, x]
    surface = float(geometry["surface_pressure"][y, x])
    target = geometry["target_pressure"][:, y, x]
    included = geometry["operator_class"][:, y, x] <= 1
    first_above = int(geometry["first_above"][y, x])
    assembled: list[tuple[str, int | None, float]] = []
    for level in range(source.size):
        if level == first_above:
            assembled.append(("surface", None, surface))
        if included[level]:
            assembled.append(("source", level, float(source[level])))
    position = next(
        (index for index, item in enumerate(assembled)
         if item[0] == "source" and item[1] == ordered_level), None)

    def point(index):
        if index is None or index < 0 or index >= len(assembled):
            return None
        kind, level, pressure = assembled[index]
        return {
            "kind": kind,
            "wrf_ordered_source_level": level,
            "pressure_pa": pressure,
        }

    influencing_targets = []
    if position is not None:
        pressure_x = np.log(np.asarray(
            [item[2] for item in assembled], dtype=np.float32))
        target_x = np.log(np.asarray(target, dtype=np.float32))
        for target_index, (pt, xt) in enumerate(zip(target, target_x)):
            participants: set[int] = set()
            found = None
            for lower in range(len(assembled) - 1):
                if ((xt - pressure_x[lower])
                        * (xt - pressure_x[lower + 1]) <= 0.0):
                    found = lower
                    break
            if found is None:
                if pt > assembled[0][2] and assembled[0][0] == "source":
                    participants.add(0)
            else:
                if xt != pressure_x[found + 1]:
                    participants.add(found)
                if xt != pressure_x[found]:
                    participants.add(found + 1)
            if position in participants:
                influencing_targets.append(target_index)

    code = int(raw_labels[raw_level, y, x])
    return {
        "class": _HRRR_DISPOSITION_CLASS_BY_CODE[code],
        "raw_source_index": [raw_level, int(y), int(x)],
        "raw_flat_index": int(np.ravel_multi_index(
            (raw_level, y, x), raw_labels.shape)),
        "wrf_ordered_source_level": int(ordered_level),
        "source_value": float(raw_field[raw_level, y, x]),
        "source_pressure_pa": float(source[ordered_level]),
        "surface_pressure_pa": surface,
        "force_target_pressure_pa": float(target[0]),
        "target_pressure_minimum_pa": float(np.min(target)),
        "target_pressure_maximum_pa": float(np.max(target)),
        "previous_assembled_point": (
            point(position - 1) if position is not None else None),
        "next_assembled_point": (
            point(position + 1) if position is not None else None),
        "influencing_target_indices": influencing_targets,
    }


def build_hrrr_hydrometeor_vertical_disposition(
        source_fields: Mapping[str, object], source_level_order,
        source_pressure, surface_pressure, target_pressure,
        initialized_fields: Mapping[str, object], *,
        operator_replay) -> dict[str, object]:
    """Build exhaustive, source-ordered WRF Q-support evidence.

    Every exact nonzero decoded FP32 source sample receives exactly one class:
    target-influencing, one of WRF's two column-assembly exclusions, or one
    of three explicit reasons an assembled sample has no target stencil.
    Labels are stored in decoded-source order and compressed losslessly; their
    nonzero mask must therefore reproduce the decoded field's independently
    published mask checksum.
    """

    if (not source_fields or set(source_fields) != set(initialized_fields)
            or not callable(operator_replay)):
        raise ValueError(
            "hydrometeor disposition source/state/replay inventory differs")
    order = np.ascontiguousarray(source_level_order, dtype=np.int32)
    first_shape = tuple(next(iter(source_fields.values())).shape)
    if (len(first_shape) != 3 or order.shape != (first_shape[0],)
            or not np.array_equal(np.sort(order), np.arange(first_shape[0]))
            or any(tuple(value.shape) != first_shape
                   for value in source_fields.values())):
        raise ValueError("hydrometeor disposition source inventory is invalid")
    active = np.zeros(first_shape, dtype=np.bool_)
    for name in sorted(source_fields):
        raw = _host_float32(source_fields[name])
        if not np.isfinite(raw).all() or np.any(raw < 0.0):
            raise ValueError(
                f"hydrometeor disposition source {name} is invalid")
        active |= _ordered_levels(raw, order) != 0.0
    geometry = _hrrr_operator_geometry(
        source_pressure, surface_pressure, target_pressure, active)
    if geometry["source_pressure"].shape != first_shape:
        raise ValueError(
            "hydrometeor disposition pressure/source shapes differ")
    if np.any(~np.isin(
            geometry["operator_class"], np.asarray([0, 1, 2, 3],
                                                   dtype=np.uint8))):
        raise AssertionError(
            "WRF hydrometeor operator geometry is unclassified")

    # Establish the exact production support of each source sample without
    # trusting the symbolic pressure classifier.  Vertical interpolation is
    # column-local, linear, and nonnegative for this Q path, so replaying one
    # ordered source level of binary ones across every column identifies that
    # level's support exactly: a source cell participates iff any target in
    # its column is exact nonzero.  The nsource replays are shared by every
    # retained hydrometeor species.
    production_support = np.zeros(first_shape, dtype=np.bool_)
    level_replays = []
    level_input = np.zeros(first_shape, dtype=np.bool_)
    for ordered_level in range(first_shape[0]):
        level_input[ordered_level] = True
        level_output = _host_float32(operator_replay(level_input))
        level_input[ordered_level] = False
        if (level_output.shape != geometry["target_pressure"].shape
                or not np.isfinite(level_output).all()
                or np.any(level_output < 0.0)):
            raise ValueError(
                "hydrometeor disposition source-level production replay "
                f"failed at ordered level {ordered_level}")
        level_support = np.any(level_output != 0.0, axis=0)
        production_support[ordered_level] = level_support
        level_replays.append({
            "ordered_source_level": ordered_level,
            "raw_source_level": int(order[ordered_level]),
            "supported_column_count": int(np.count_nonzero(level_support)),
            "support_mask_sha256": _packed_mask_sha256(level_support),
            "output": array_correspondence_fingerprint(level_output),
        })
        del level_output, level_support
    del level_input
    if not np.array_equal(
            active & (geometry["operator_class"] == 1),
            active & production_support):
        raise ValueError(
            "hydrometeor disposition falsely excluded or included WRF "
            "target support in its symbolic pressure partition")
    del active

    raw_production_support = np.zeros_like(production_support)
    raw_production_support[order] = production_support
    packed_production_support = np.packbits(
        raw_production_support.ravel(), bitorder="little")
    production_support_receipt = {
        "schema": _HRRR_PRODUCTION_SUPPORT_SCHEMA,
        "shape": list(first_shape),
        "encoding": _HRRR_PRODUCTION_SUPPORT_ENCODING,
        "mask_base64": base64.b64encode(zlib.compress(
            packed_production_support.tobytes(), level=9)).decode("ascii"),
        "mask_sha256": hashlib.sha256(
            packed_production_support.tobytes()).hexdigest(),
        "support_count": int(np.count_nonzero(raw_production_support)),
        "level_replays": level_replays,
    }
    del raw_production_support, packed_production_support

    geometry_receipt = {
        "source_pressure": array_correspondence_fingerprint(
            geometry["source_pressure"]),
        "surface_pressure": array_correspondence_fingerprint(
            geometry["surface_pressure"]),
        "target_pressure": array_correspondence_fingerprint(
            geometry["target_pressure"]),
        "source_level_order": array_correspondence_fingerprint(order),
        "source_level_order_values": order.tolist(),
        "production_target_support": production_support_receipt,
    }
    species_receipts = {}
    for name in sorted(source_fields):
        raw = _host_float32(source_fields[name])
        ordered = _ordered_levels(raw, order)
        positive = ordered != 0.0
        labels = np.zeros(first_shape, dtype=np.uint8)
        operator_class = geometry["operator_class"]
        excluded = positive & (operator_class >= 2)
        labels[excluded] = operator_class[excluded]
        included_positive = positive & (operator_class <= 1)
        target_influencing = (
            included_positive & (operator_class == 1))
        production_target_influencing = positive & production_support
        if not np.array_equal(
                target_influencing, production_target_influencing):
            raise ValueError(
                "hydrometeor disposition falsely excluded or included WRF "
                f"target support for {name}")
        labels[target_influencing] = _HRRR_DISPOSITION_CLASSES[
            "TARGET_INFLUENCING"]
        unsupported = included_positive & ~target_influencing
        labels[
            unsupported
            & (geometry["source_pressure"]
               >= geometry["surface_pressure"][None, :, :])
        ] = _HRRR_DISPOSITION_CLASSES[
            "WRF_BELOW_GROUND_OUTSIDE_TARGET_SUPPORT"]
        labels[
            unsupported
            & (geometry["source_pressure"]
               < np.min(geometry["target_pressure"], axis=0)[None, :, :])
        ] = _HRRR_DISPOSITION_CLASSES[
            "WRF_ABOVE_TARGET_TOP_OUTSIDE_TARGET_SUPPORT"]
        labels[unsupported & (labels == 0)] = _HRRR_DISPOSITION_CLASSES[
            "WRF_NO_TARGET_STENCIL"]
        if np.any(positive & (labels == 0)) or np.any(~positive & (labels != 0)):
            raise AssertionError(
                f"hydrometeor disposition partition failed for {name}")

        raw_labels = np.zeros_like(labels)
        raw_labels[order] = labels
        encoded = zlib.compress(raw_labels.tobytes(order="C"), level=9)
        counts = {}
        mask_hashes = {}
        examples = {}
        for class_name, code in _HRRR_DISPOSITION_CLASSES.items():
            class_mask = raw_labels == code
            count = int(np.count_nonzero(class_mask))
            counts[class_name] = count
            mask_hashes[class_name] = _packed_mask_sha256(class_mask)
            records = []
            flat_indices = np.flatnonzero(class_mask.ravel())[
                :_HRRR_DISPOSITION_EXAMPLE_LIMIT]
            for flat_index in flat_indices:
                raw_level, y, x = np.unravel_index(
                    int(flat_index), first_shape)
                ordered_level = int(np.flatnonzero(order == raw_level)[0])
                records.append(_hrrr_disposition_example(
                    raw_field=raw, raw_labels=raw_labels, order=order,
                    ordered_level=ordered_level, y=int(y), x=int(x),
                    geometry=geometry))
            examples[class_name] = {
                "complete": count <= _HRRR_DISPOSITION_EXAMPLE_LIMIT,
                "total_count": count,
                "records": records,
            }
        source_fingerprint = array_correspondence_fingerprint(raw)
        live_fingerprint = array_correspondence_fingerprint(
            initialized_fields[name])
        # This is the independent production-authority check on the symbolic
        # pressure partition above.  Q interpolation is linear with
        # nonnegative weights and an exact-zero surface pseudo-level.  Thus a
        # binary replay of every excluded sample must be identically zero,
        # while replaying only TARGET_INFLUENCING samples must be byte-equal
        # to replaying the complete decoded nonzero mask.  The callback uses
        # the very same prepared backend plan/options as the analyzed field.
        source_replay = _host_float32(operator_replay(labels != 0))
        source_replay_fingerprint = array_correspondence_fingerprint(
            source_replay)
        if (list(source_replay.shape) != live_fingerprint["shape"]
                or not np.isfinite(source_replay).all()
                or np.any(source_replay < 0.0)):
            raise ValueError(
                f"hydrometeor disposition operator replay failed for {name}")
        del source_replay
        influencing_replay = _host_float32(
            operator_replay(
                labels == _HRRR_DISPOSITION_CLASSES[
                    "TARGET_INFLUENCING"]))
        if (list(influencing_replay.shape) != live_fingerprint["shape"]
                or not np.isfinite(influencing_replay).all()
                or np.any(influencing_replay < 0.0)):
            raise ValueError(
                f"hydrometeor disposition operator replay failed for {name}")
        influencing_replay_fingerprint = array_correspondence_fingerprint(
            influencing_replay)
        if source_replay_fingerprint != influencing_replay_fingerprint:
            raise ValueError(
                "hydrometeor disposition falsely excluded or included WRF "
                f"target support for {name}")
        del influencing_replay
        excluded_replay = _host_float32(operator_replay(
            (labels != 0)
            & (labels != _HRRR_DISPOSITION_CLASSES[
                "TARGET_INFLUENCING"])))
        if (list(excluded_replay.shape) != live_fingerprint["shape"]
                or not np.isfinite(excluded_replay).all()
                or np.any(excluded_replay < 0.0)):
            raise ValueError(
                f"hydrometeor disposition operator replay failed for {name}")
        if np.any(excluded_replay != 0.0):
            raise ValueError(
                "hydrometeor disposition classified target-influencing "
                f"source mass as excluded for {name}")
        operator_replay_receipt = {
            "schema": "gpuwm-wrf-q-binary-mask-production-replay-v1",
            "source_mask_sha256": _packed_mask_sha256(raw != 0.0),
            "target_influencing_mask_sha256": _packed_mask_sha256(
                raw_labels == _HRRR_DISPOSITION_CLASSES[
                    "TARGET_INFLUENCING"]),
            "excluded_mask_sha256": _packed_mask_sha256(
                (raw_labels != 0)
                & (raw_labels != _HRRR_DISPOSITION_CLASSES[
                    "TARGET_INFLUENCING"])),
            "source_output": source_replay_fingerprint,
            "target_influencing_output": influencing_replay_fingerprint,
            "excluded_output": array_correspondence_fingerprint(
                excluded_replay),
            "source_target_outputs_byte_equal": True,
            "excluded_output_all_exact_zero": True,
        }
        influencing_count = counts["TARGET_INFLUENCING"]
        if influencing_count > 0 and int(live_fingerprint["nonzero_count"]) == 0:
            raise ValueError(
                "WRF target-influencing analyzed hydrometeor source mass "
                f"was lost for {name}")
        if influencing_count == 0 and int(live_fingerprint["nonzero_count"]) != 0:
            raise ValueError(
                "WRF target-independent hydrometeor source produced "
                f"unexpected initialized mass for {name}")
        species_receipts[name] = {
            "shape": list(first_shape),
            "encoding": _HRRR_DISPOSITION_ENCODING,
            "labels_base64": base64.b64encode(encoded).decode("ascii"),
            "labels_sha256": hashlib.sha256(
                raw_labels.tobytes(order="C")).hexdigest(),
            "decoded_source_fingerprint_sha256": (
                _canonical_receipt_sha256(source_fingerprint)),
            "initialized_state_fingerprint_sha256": (
                _canonical_receipt_sha256(live_fingerprint)),
            "partition_complete": True,
            "source_nonzero_count": int(source_fingerprint["nonzero_count"]),
            "target_influencing_source_count": influencing_count,
            "wrf_excluded_source_count": (
                int(source_fingerprint["nonzero_count"])
                - influencing_count),
            "class_counts": counts,
            "class_mask_sha256": mask_hashes,
            "examples": examples,
            "operator_replay": operator_replay_receipt,
        }

    receipt = {
        "schema": HRRR_HYDROMETEOR_VERTICAL_DISPOSITION_SCHEMA,
        "operator": dict(_HRRR_DISPOSITION_OPERATOR),
        "geometry": geometry_receipt,
        "species": species_receipts,
    }
    receipt["evidence_sha256"] = _canonical_receipt_sha256(receipt)
    return receipt


def _decode_hrrr_disposition_labels(
        encoded: object, *, expected_bytes: int) -> np.ndarray:
    if not isinstance(encoded, str) or expected_bytes < 1:
        raise ValueError("hydrometeor disposition label encoding is invalid")
    try:
        compressed = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError(
            "hydrometeor disposition label encoding is invalid") from exc
    decompressor = zlib.decompressobj()
    try:
        decoded = decompressor.decompress(compressed, expected_bytes + 1)
        decoded += decompressor.flush(max(1, expected_bytes + 1 - len(decoded)))
    except zlib.error as exc:
        raise ValueError(
            "hydrometeor disposition labels are not valid zlib data") from exc
    if (len(decoded) != expected_bytes or not decompressor.eof
            or decompressor.unused_data or decompressor.unconsumed_tail):
        raise ValueError(
            "hydrometeor disposition label payload has the wrong length")
    return np.frombuffer(decoded, dtype=np.uint8).copy()


def _decode_hrrr_production_support(
        encoded: object, *, expected_bits: int) -> np.ndarray:
    """Decode one canonical packed production-support certificate."""

    expected_bytes = (expected_bits + 7) // 8
    try:
        packed = _decode_hrrr_disposition_labels(
            encoded, expected_bytes=expected_bytes)
    except ValueError as exc:
        raise ValueError(
            "hydrometeor production support encoding is invalid") from exc
    support = np.unpackbits(
        packed, bitorder="little", count=expected_bits).astype(
            np.bool_, copy=False)
    if not np.array_equal(
            np.packbits(support, bitorder="little"), packed):
        raise ValueError(
            "hydrometeor production support padding is noncanonical")
    return support


def validate_hrrr_hydrometeor_vertical_disposition(
        source_fingerprints: Mapping[str, object],
        initialized_fingerprints: Mapping[str, object],
        disposition: object) -> dict[str, object]:
    """Fail closed over a producer's exhaustive WRF-support partition."""

    if not isinstance(disposition, Mapping):
        raise ValueError("hydrometeor vertical disposition evidence is missing")
    if set(disposition) != {
            "schema", "operator", "geometry", "species",
            "evidence_sha256"}:
        raise ValueError("hydrometeor vertical disposition fields are invalid")
    if (disposition.get("schema")
            != HRRR_HYDROMETEOR_VERTICAL_DISPOSITION_SCHEMA
            or disposition.get("operator") != _HRRR_DISPOSITION_OPERATOR
            or not isinstance(disposition.get("geometry"), Mapping)
            or not isinstance(disposition.get("species"), Mapping)
            or not isinstance(disposition.get("evidence_sha256"), str)):
        raise ValueError("hydrometeor vertical disposition schema is invalid")
    unsigned = dict(disposition)
    observed_hash = unsigned.pop("evidence_sha256")
    if observed_hash != _canonical_receipt_sha256(unsigned):
        raise ValueError("hydrometeor vertical disposition evidence was changed")
    geometry = disposition["geometry"]
    if set(geometry) != {
            "source_pressure", "surface_pressure", "target_pressure",
            "source_level_order", "source_level_order_values",
            "production_target_support"}:
        raise ValueError("hydrometeor disposition geometry is incomplete")
    for name in (
            "source_pressure", "surface_pressure", "target_pressure",
            "source_level_order"):
        if (not isinstance(geometry[name], Mapping)
                or not {"shape", "dtype", "sha256",
                        "nonzero_mask_sha256", "nonzero_count",
                        "minimum", "maximum"} <= set(geometry[name])):
            raise ValueError(
                "hydrometeor disposition geometry fingerprint is invalid")
    if (set(source_fingerprints) != set(initialized_fingerprints)
            or set(disposition["species"]) != set(source_fingerprints)
            or not source_fingerprints):
        raise ValueError(
            "hydrometeor disposition species inventory is incomplete")
    first_source = next(iter(source_fingerprints.values()))
    first_initialized = next(iter(initialized_fingerprints.values()))
    source_shape = first_source.get("shape") \
        if isinstance(first_source, Mapping) else None
    state_shape = first_initialized.get("shape") \
        if isinstance(first_initialized, Mapping) else None
    order_values = geometry["source_level_order_values"]
    if (not isinstance(source_shape, list) or len(source_shape) != 3
            or not isinstance(state_shape, list) or len(state_shape) != 3
            or any(isinstance(value, bool) or not isinstance(value, int)
                   or value < 1 for value in source_shape + state_shape)
            or source_shape[0] * source_shape[1] * source_shape[2]
            > 250_000_000
            or state_shape[0] * state_shape[1] * state_shape[2]
            > 250_000_000
            or not isinstance(order_values, list)
            or len(order_values) != source_shape[0]
            or any(isinstance(value, bool) or not isinstance(value, int)
                   for value in order_values)
            or sorted(order_values) != list(range(source_shape[0]))
            or geometry["source_pressure"].get("shape") != source_shape
            or geometry["source_pressure"].get("dtype") != "float32"
            or geometry["surface_pressure"].get("shape") != source_shape[1:]
            or geometry["surface_pressure"].get("dtype") != "float32"
            or geometry["target_pressure"].get("shape") != state_shape
            or geometry["target_pressure"].get("dtype") != "float32"
            or any(float(geometry[key].get("minimum", np.nan)) <= 0.0
                   for key in ("source_pressure", "surface_pressure",
                               "target_pressure"))
            or geometry["source_level_order"]
            != array_correspondence_fingerprint(
                np.asarray(order_values, dtype=np.int32))):
        raise ValueError("hydrometeor disposition geometry is inconsistent")

    required_fingerprint = {
        "shape", "dtype", "sha256", "nonzero_mask_sha256",
        "nonzero_count", "minimum", "maximum"}
    support_receipt = geometry["production_target_support"]
    if (not isinstance(support_receipt, Mapping)
            or set(support_receipt) != {
                "schema", "shape", "encoding", "mask_base64",
                "mask_sha256", "support_count", "level_replays"}
            or support_receipt.get("schema")
            != _HRRR_PRODUCTION_SUPPORT_SCHEMA
            or support_receipt.get("shape") != source_shape
            or support_receipt.get("encoding")
            != _HRRR_PRODUCTION_SUPPORT_ENCODING
            or not isinstance(support_receipt.get("mask_sha256"), str)
            or isinstance(support_receipt.get("support_count"), bool)
            or not isinstance(support_receipt.get("support_count"), int)
            or not 0 <= support_receipt["support_count"] <= int(
                np.prod(source_shape, dtype=np.int64))
            or not isinstance(support_receipt.get("level_replays"), list)
            or len(support_receipt["level_replays"]) != source_shape[0]):
        raise ValueError(
            "hydrometeor production target support evidence is invalid")
    production_support = _decode_hrrr_production_support(
        support_receipt.get("mask_base64"),
        expected_bits=int(np.prod(source_shape, dtype=np.int64))).reshape(
            source_shape)
    if (support_receipt["mask_sha256"]
            != _packed_mask_sha256(production_support)
            or support_receipt["support_count"]
            != int(np.count_nonzero(production_support))):
        raise ValueError(
            "hydrometeor production target support mask is invalid")
    ncolumn = source_shape[1] * source_shape[2]
    ntarget = state_shape[0]
    for ordered_level, replay in enumerate(
            support_receipt["level_replays"]):
        raw_level = order_values[ordered_level]
        level_support = production_support[raw_level]
        support_count = int(np.count_nonzero(level_support))
        if (not isinstance(replay, Mapping)
                or set(replay) != {
                    "ordered_source_level", "raw_source_level",
                    "supported_column_count", "support_mask_sha256",
                    "output"}
                or replay.get("ordered_source_level") != ordered_level
                or replay.get("raw_source_level") != raw_level
                or isinstance(replay.get("supported_column_count"), bool)
                or replay.get("supported_column_count") != support_count
                or support_count > ncolumn
                or replay.get("support_mask_sha256")
                != _packed_mask_sha256(level_support)
                or not isinstance(replay.get("output"), Mapping)
                or set(replay["output"]) != required_fingerprint
                or replay["output"].get("shape") != state_shape
                or replay["output"].get("dtype") != "float32"
                or isinstance(
                    replay["output"].get("nonzero_count"), bool)
                or not isinstance(
                    replay["output"].get("nonzero_count"), int)
                or not support_count
                <= replay["output"]["nonzero_count"]
                <= support_count * ntarget
                or not 0.0 <= float(
                    replay["output"].get("minimum", np.nan))
                <= float(replay["output"].get("maximum", np.nan))
                <= 1.0):
            raise ValueError(
                "hydrometeor source-level production replay is invalid")

    validated = {}
    for name in sorted(source_fingerprints):
        source = source_fingerprints[name]
        initialized = initialized_fingerprints[name]
        item = disposition["species"][name]
        if (not isinstance(source, Mapping)
                or not isinstance(initialized, Mapping)
                or not isinstance(item, Mapping)):
            raise ValueError(
                f"hydrometeor disposition {name} fingerprints are invalid")
        if (not required_fingerprint <= set(source)
                or not required_fingerprint <= set(initialized)):
            raise ValueError(
                f"hydrometeor disposition {name} fingerprint is incomplete")
        if set(item) != {
                "shape", "encoding", "labels_base64", "labels_sha256",
                "decoded_source_fingerprint_sha256",
                "initialized_state_fingerprint_sha256",
                "partition_complete", "source_nonzero_count",
                "target_influencing_source_count",
                "wrf_excluded_source_count", "class_counts",
                "class_mask_sha256", "examples", "operator_replay"}:
            raise ValueError(
                f"hydrometeor disposition {name} fields are incomplete")
        count_fields = (
            source.get("nonzero_count"), initialized.get("nonzero_count"),
            item.get("source_nonzero_count"),
            item.get("target_influencing_source_count"),
            item.get("wrf_excluded_source_count"),
        )
        if any(isinstance(value, bool) or not isinstance(value, int)
               or value < 0 for value in count_fields):
            raise ValueError(
                f"hydrometeor disposition {name} count is invalid")
        shape = item.get("shape")
        if (not isinstance(shape, list) or len(shape) != 3
                or any(isinstance(value, bool) or not isinstance(value, int)
                       or value < 1 for value in shape)
                or shape != source.get("shape")
                or shape != source_shape
                or initialized.get("shape") != state_shape
                or np.prod(shape, dtype=np.int64) > 250_000_000):
            raise ValueError(
                f"hydrometeor disposition {name} shape is invalid")
        expected_bytes = int(np.prod(shape, dtype=np.int64))
        if item.get("encoding") != _HRRR_DISPOSITION_ENCODING:
            raise ValueError(
                f"hydrometeor disposition {name} encoding is invalid")
        labels = _decode_hrrr_disposition_labels(
            item.get("labels_base64"), expected_bytes=expected_bytes)
        labels = labels.reshape(shape)
        if (item.get("labels_sha256") != hashlib.sha256(
                labels.tobytes(order="C")).hexdigest()
                or item.get("decoded_source_fingerprint_sha256")
                != _canonical_receipt_sha256(source)
                or item.get("initialized_state_fingerprint_sha256")
                != _canonical_receipt_sha256(initialized)
                or item.get("partition_complete") is not True):
            raise ValueError(
                f"hydrometeor disposition {name} identity is invalid")
        allowed_codes = np.asarray(
            [0, *_HRRR_DISPOSITION_CLASS_BY_CODE], dtype=np.uint8)
        if np.any(~np.isin(labels, allowed_codes)):
            raise ValueError(
                f"hydrometeor disposition {name} has an unknown class")
        source_count = int(source.get("nonzero_count", -1))
        if (source_count < 0
                or item.get("source_nonzero_count") != source_count
                or int(np.count_nonzero(labels)) != source_count
                or _packed_mask_sha256(labels != 0)
                != source.get("nonzero_mask_sha256")):
            raise ValueError(
                f"hydrometeor disposition {name} does not partition the "
                "decoded source mask")
        counts = item.get("class_counts")
        mask_hashes = item.get("class_mask_sha256")
        examples = item.get("examples")
        if (not isinstance(counts, Mapping)
                or set(counts) != set(_HRRR_DISPOSITION_CLASSES)
                or not isinstance(mask_hashes, Mapping)
                or set(mask_hashes) != set(_HRRR_DISPOSITION_CLASSES)
                or not isinstance(examples, Mapping)
                or set(examples) != set(_HRRR_DISPOSITION_CLASSES)):
            raise ValueError(
                f"hydrometeor disposition {name} class evidence is invalid")
        observed_counts = {}
        for class_name, code in _HRRR_DISPOSITION_CLASSES.items():
            class_mask = labels == code
            count = int(np.count_nonzero(class_mask))
            observed_counts[class_name] = count
            example = examples[class_name]
            if (isinstance(counts[class_name], bool)
                    or not isinstance(counts[class_name], int)
                    or counts[class_name] != count
                    or mask_hashes[class_name]
                    != _packed_mask_sha256(class_mask)
                    or not isinstance(example, Mapping)
                    or set(example) != {"complete", "total_count", "records"}
                    or isinstance(example.get("total_count"), bool)
                    or not isinstance(example.get("total_count"), int)
                    or example.get("total_count") != count
                    or example.get("complete")
                    is not (count <= _HRRR_DISPOSITION_EXAMPLE_LIMIT)
                    or not isinstance(example.get("records"), list)
                    or len(example["records"])
                    != min(count, _HRRR_DISPOSITION_EXAMPLE_LIMIT)):
                raise ValueError(
                    f"hydrometeor disposition {name}/{class_name} "
                    "evidence is inconsistent")
            seen = set()
            for record in example["records"]:
                if (not isinstance(record, Mapping)
                        or record.get("class") != class_name
                        or not isinstance(record.get("raw_flat_index"), int)
                        or record["raw_flat_index"] in seen
                        or not 0 <= record["raw_flat_index"] < expected_bytes
                        or int(labels.ravel()[record["raw_flat_index"]])
                        != code
                        or not isinstance(
                            record.get("influencing_target_indices"), list)
                        or (class_name == "TARGET_INFLUENCING")
                        != bool(record["influencing_target_indices"])):
                    raise ValueError(
                        f"hydrometeor disposition {name}/{class_name} "
                        "example is invalid")
                seen.add(record["raw_flat_index"])
        influencing_count = observed_counts["TARGET_INFLUENCING"]
        excluded_count = source_count - influencing_count
        expected_target = (labels != 0) & production_support
        if not np.array_equal(
                labels == _HRRR_DISPOSITION_CLASSES["TARGET_INFLUENCING"],
                expected_target):
            raise ValueError(
                f"hydrometeor disposition {name} contradicts exact "
                "production target support")
        if (item.get("target_influencing_source_count") != influencing_count
                or item.get("wrf_excluded_source_count") != excluded_count):
            raise ValueError(
                f"hydrometeor disposition {name} partition totals differ")
        replay = item.get("operator_replay")
        if (not isinstance(replay, Mapping)
                or set(replay) != {
                    "schema", "source_mask_sha256",
                    "target_influencing_mask_sha256",
                    "excluded_mask_sha256", "source_output",
                    "target_influencing_output", "excluded_output",
                    "source_target_outputs_byte_equal",
                    "excluded_output_all_exact_zero"}
                or replay.get("schema")
                != "gpuwm-wrf-q-binary-mask-production-replay-v1"
                or replay.get("source_mask_sha256")
                != _packed_mask_sha256(labels != 0)
                or replay.get("target_influencing_mask_sha256")
                != _packed_mask_sha256(
                    labels == _HRRR_DISPOSITION_CLASSES[
                        "TARGET_INFLUENCING"])
                or replay.get("excluded_mask_sha256")
                != _packed_mask_sha256(
                    (labels != 0)
                    & (labels != _HRRR_DISPOSITION_CLASSES[
                        "TARGET_INFLUENCING"]))
                or replay.get("source_output")
                != replay.get("target_influencing_output")
                or replay.get("source_target_outputs_byte_equal") is not True
                or replay.get("excluded_output_all_exact_zero") is not True):
            raise ValueError(
                f"hydrometeor disposition {name} production replay is invalid")
        excluded_output = replay.get("excluded_output")
        replay_output = replay.get("source_output")
        state_shape = initialized.get("shape")
        if (not isinstance(state_shape, list) or len(state_shape) != 3
                or any(isinstance(value, bool) or not isinstance(value, int)
                       or value < 1 for value in state_shape)
                or np.prod(state_shape, dtype=np.int64) > 250_000_000):
            raise ValueError(
                f"hydrometeor disposition {name} state shape is invalid")
        state_size = int(np.prod(state_shape, dtype=np.int64))
        exact_zero_output = {
            "shape": state_shape,
            "dtype": "float32",
            "sha256": _zero_byte_sha256(4 * state_size),
            "nonzero_mask_sha256": _zero_byte_sha256((state_size + 7) // 8),
            "nonzero_count": 0,
            "minimum": 0.0,
            "maximum": 0.0,
        }
        if (not isinstance(replay_output, Mapping)
                or not required_fingerprint <= set(replay_output)
                or replay_output.get("shape") != initialized.get("shape")
                or replay_output.get("dtype") != "float32"
                or isinstance(replay_output.get("nonzero_count"), bool)
                or not isinstance(replay_output.get("nonzero_count"), int)
                or int(replay_output.get("nonzero_count", -1)) < 0
                or not np.isfinite(float(
                    replay_output.get("minimum", np.nan)))
                or not np.isfinite(float(
                    replay_output.get("maximum", np.nan)))
                or not isinstance(excluded_output, Mapping)
                or any(isinstance(excluded_output.get(key), bool)
                       for key in ("nonzero_count", "minimum", "maximum"))
                or dict(excluded_output) != exact_zero_output):
            raise ValueError(
                f"hydrometeor disposition {name} excluded replay is nonzero")
        live_count = int(initialized.get("nonzero_count", -1))
        if live_count < 0:
            raise ValueError(
                f"hydrometeor disposition {name} state count is invalid")
        if influencing_count > 0 and live_count == 0:
            raise ValueError(
                "WRF target-influencing analyzed hydrometeor source mass "
                f"was lost for {name}")
        if influencing_count == 0 and live_count != 0:
            raise ValueError(
                "WRF target-independent hydrometeor source produced "
                f"unexpected initialized mass for {name}")
        if source_count == 0:
            strength = "VACUOUS"
        elif influencing_count == 0:
            strength = "WRF_EXCLUDED"
        elif excluded_count:
            strength = "PARTIALLY_WRF_EXCLUDED"
        else:
            strength = "PROVEN"
        validated[name] = {
            "strength": strength,
            "source_nonzero_count": source_count,
            "target_influencing_source_count": influencing_count,
            "wrf_excluded_source_count": excluded_count,
            "class_counts": observed_counts,
            "labels_sha256": item["labels_sha256"],
        }
    return {
        "schema": HRRR_HYDROMETEOR_VERTICAL_DISPOSITION_SCHEMA,
        "evidence_sha256": disposition["evidence_sha256"],
        "species": validated,
    }


def _saturation_mixing_ratio_serial(temperature, pressure,
                                    relative_humidity):
    """Evaluate one contiguous WRF/Bolton humidity chunk."""
    temperature, pressure, rh = np.broadcast_arrays(
        np.asarray(temperature, dtype=np.float64),
        np.asarray(pressure, dtype=np.float64),
        np.clip(np.asarray(relative_humidity, dtype=np.float64), 0.0, 100.0),
    )
    # SVP1 is kPa in module_model_constants; convert to hPa to pair with p/100.
    # rh_to_mxrat1 uses its own local EPS = 0.622, NOT module ep_2 = 0.62175
    # (module_initialize_real.F:7379, q = MAX(eps*es/(p/100.-es), 1.E-6)).
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        es_hpa = (rh * 0.01) * (10.0 * c.SVP1) * np.exp(
            c.SVP2 * (temperature - c.SVPT0) / (temperature - c.SVP3))
        candidate = 0.622 * es_hpa / (pressure / 100.0 - es_hpa)
    valid = ((temperature != 0.0) & np.isfinite(es_hpa)
             & (es_hpa < pressure / 100.0))
    return np.where(valid, np.maximum(candidate, 1.0e-6), 1.0e-6)


def _saturation_mixing_ratio(temperature, pressure, relative_humidity=100.0,
                             *, column_workers=1):
    """WRF/Bolton liquid-water saturation mixing ratio (kg kg-1).

    Large setup arrays are divided into deterministic contiguous chunks.
    NumPy performs the unchanged float64 expression in independent threads,
    preserving every element's arithmetic while using otherwise-idle host
    cores during native initial-condition preparation.
    """
    workers = _column_worker_count(column_workers)
    temperature, pressure, rh = np.broadcast_arrays(
        np.asarray(temperature, dtype=np.float64),
        np.asarray(pressure, dtype=np.float64),
        np.clip(np.asarray(relative_humidity, dtype=np.float64), 0.0, 100.0),
    )
    if workers == 1 or temperature.ndim == 0 or temperature.shape[0] < 2:
        return _saturation_mixing_ratio_serial(temperature, pressure, rh)

    chunks = _axis0_chunks(temperature.shape[0], workers)
    output = np.empty(temperature.shape, dtype=np.float64)
    with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
        futures = [
            executor.submit(
                _fill_axis0_chunk, output, start, stop,
                _saturation_mixing_ratio_serial,
                (temperature[start:stop], pressure[start:stop],
                 rh[start:stop]))
            for start, stop in chunks
        ]
        for future in futures:
            future.result()
    return output


def _cap_stratospheric_qv_serial(qv, pressure):
    """Evaluate one contiguous stratospheric-cap chunk."""
    qv = np.asarray(qv, dtype=np.float64)
    pressure = np.asarray(pressure, dtype=np.float64)
    return np.where((pressure < 10000.0) & (qv > 1.0e-5), 3.0e-6, qv)


def _cap_stratospheric_qv(qv, pressure, *, column_workers=1):
    """WRF ``rh_to_mxrat1`` stratospheric qv sanity cap.

    module_initialize_real.F:7490-7498 with the Registry defaults
    (Registry.EM_COMMON:2306-2308): where ``p < qv_max_p_safe`` (10000 Pa,
    strict) and ``qv > qv_max_flag`` (1e-5, strict), force ``qv_max_value``
    (3e-6).  The companion qv_min cap (:7499-7503, p < 110000 Pa, qv < 1e-6
    -> 1e-6) is already realized by the unconditional 1e-6 floor in
    :func:`_saturation_mixing_ratio` for this domain's pressure range.
    """
    workers = _column_worker_count(column_workers)
    qv, pressure = np.broadcast_arrays(
        np.asarray(qv, dtype=np.float64),
        np.asarray(pressure, dtype=np.float64),
    )
    if workers == 1 or qv.ndim == 0 or qv.shape[0] < 2:
        return _cap_stratospheric_qv_serial(qv, pressure)

    chunks = _axis0_chunks(qv.shape[0], workers)
    output = np.empty(qv.shape, dtype=np.float64)
    with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
        futures = [
            executor.submit(
                _fill_axis0_chunk, output, start, stop,
                _cap_stratospheric_qv_serial,
                (qv[start:stop], pressure[start:stop]))
            for start, stop in chunks
        ]
        for future in futures:
            future.result()
    return output


_WPS_SPFH_UNDERSHOOT_LOWER_BOUND = -0.028126


def _specific_humidity_to_mixing_ratio_serial(
        specific_humidity, *, allow_wps_undershoot=False):
    """Convert and validate one contiguous specific-humidity chunk."""
    specific = np.asarray(specific_humidity, dtype=np.float64)
    lower = (_WPS_SPFH_UNDERSHOOT_LOWER_BOUND
             if allow_wps_undershoot else 0.0)
    if (not np.isfinite(specific).all() or np.any(specific < lower)
            or np.any(specific >= 1.0)):
        interval = (f"[{_WPS_SPFH_UNDERSHOOT_LOWER_BOUND}, 1)"
                    if allow_wps_undershoot else "[0, 1)")
        raise ValueError(f"specific humidity must be finite in {interval}")
    return specific / (1.0 - specific)


def _specific_humidity_to_mixing_ratio(
        specific_humidity, *, allow_wps_undershoot=False,
        column_workers=1):
    """Convert HRRR/WPS specific humidity to dry-air mixing ratio.

    This is WRF ``module_initialize_real.F``'s ``flag_sh`` branch:
    ``qv_gc = sh_gc / (1 - sh_gc)``.  Unlike the RH path, WRF does not
    apply the ``rh_to_mxrat1`` floor or stratospheric cap when
    ``use_sh_qv`` is active.  WPS's overlapping-parabolic horizontal
    interpolation can create negative undershoots from an everywhere
    non-negative source field.  ``real.exe`` retains those values in
    ``qv_gc`` for ``integ_moist`` even when ``use_sh_qv = .false.``.  HRRR
    SPFH is gated to 0..0.1 upstream; the 2-D sixteen-point operator's most
    negative coefficient sum is -9/32, so -0.028125 is its exact lower
    envelope.  One extra micro-unit covers FP32 evaluation rounding without
    weakening the explicit direct-qv lane's physical-range check.
    """
    if not isinstance(allow_wps_undershoot, (bool, np.bool_)):
        raise TypeError("allow_wps_undershoot must be boolean")
    workers = _column_worker_count(column_workers)
    specific = np.asarray(specific_humidity, dtype=np.float64)
    if (workers == 1 or specific.ndim == 0 or specific.shape[0] < 2):
        return _specific_humidity_to_mixing_ratio_serial(
            specific, allow_wps_undershoot=allow_wps_undershoot)

    chunks = _axis0_chunks(specific.shape[0], workers)
    output = np.empty(specific.shape, dtype=np.float64)
    with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
        futures = [
            executor.submit(
                _fill_axis0_chunk, output, start, stop,
                _specific_humidity_to_mixing_ratio_serial,
                (specific[start:stop],),
                {"allow_wps_undershoot": allow_wps_undershoot})
            for start, stop in chunks
        ]
        for future in futures:
            future.result()
    return output


#: WRF's ``qv_min`` sanity floor and the pressure below which it applies
#: (Registry.EM_COMMON:2306-2308 defaults ``qv_min_p_safe = 110000`` Pa,
#: ``qv_min_flag = qv_min_value = 1e-6``; applied at
#: ``module_initialize_real.F:7499-7503``).  This is the SAME 1e-6 that
#: :func:`_saturation_mixing_ratio` already applies unconditionally on the
#: RH lane (:7379, ``q = MAX(eps*es/(p/100.-es), 1.E-6)``), named here so
#: the FLAG_SH surface lane can share one floor with it instead of
#: carrying none.
_WRF_QV_MIN_P_SAFE = 110_000.0
_WRF_QV_MIN_VALUE = 1.0e-6

_SURFACE_QV_FLOOR_WRF_REFERENCE = {
    "wrf_version": "v4.6.1",
    "wrf_citation": (
        "dyn_em/module_initialize_real.F:1138-1167 (FLAG_SH branch: "
        "qv_gc = sh_gc/(1-sh_gc) at :1157, no floor), :1253-1259 "
        "(grid%q2 = qv_gc(i,1,j) verbatim); the qv_min sanity floor "
        "real.exe applies to every other moisture value lives in "
        "rh_to_mxrat1 (:7379 unconditional 1e-6; :7499-7503 "
        "qv_min_p_safe/qv_min_flag/qv_min_value) and is called on the "
        "prognostic field only (:1837-1860), never on the surface "
        "pseudo-level of this branch"),
    "wrf_behavior": (
        "real.exe publishes an unfloored Q2: WPS routes SPECHUMD through "
        "the overshooting sixteen_pt operator (METGRID.TBL "
        "interp_option=sixteen_pt+four_pt+average_4pt), so a 2 m specific "
        "humidity that is exactly zero over high dry terrain interpolates "
        "NEGATIVE, and both qv_gc(:,1,:) and grid%q2 keep the sign"),
    "gpuwm_behavior": (
        "the PUBLISHED surface mixing ratio (RealInitResult.surface_qv, "
        "which becomes Q2) is floored at WRF's own qv_min_value where the "
        "WPS undershoot took it below that floor, which is exactly what "
        "the RH lane's _saturation_mixing_ratio already does to the same "
        "quantity; sfcprs2, integ_moist and the vertical-interpolation "
        "pseudo-level keep the raw WRF value, so no prognostic field and "
        "no existing fingerprint moves"),
    "gpuwm_divergence_reason": (
        "a mixing ratio cannot be negative and every consumer of Q2 "
        "(relative humidity, dewpoint, vapour-pressure deficit) needs the "
        "log of a positive number -- the same reason the runtime's own "
        "SFCDIAGS transcription refuses to publish WRF's negative Q2 "
        "(gpuwm/core/physics.py _refresh_surface_diagnostics), applied at "
        "initialization so the value the forecast STARTS from is physical "
        "too"),
}


def _floor_flag_sh_surface_mixing_ratio(surface_qv, surface_pressure):
    """Apply WRF's ``qv_min`` floor to the PUBLISHED 2 m mixing ratio.

    WPS's ``sixteen_pt`` overlapping parabolic is an overshooting
    operator and METGRID.TBL routes ``SPECHUMD`` through it, so a source
    2 m specific humidity that decodes to EXACTLY zero -- which HRRR's
    GRIB2 packing does over high, dry terrain -- interpolates to a small
    negative.  ``real.exe`` keeps that sign all the way into ``grid%q2``
    (see :data:`_SURFACE_QV_FLOOR_WRF_REFERENCE`), and reproducing it
    bit-for-bit would publish a negative 2 m mixing ratio.

    The floor is WRF's own ``qv_min_value``, not an invented number, and
    it is the identical constant the RH lane already applies to the
    identical quantity inside :func:`_saturation_mixing_ratio` -- so this
    makes the two surface lanes agree rather than inventing a third
    behaviour.

    It is applied at the PUBLICATION point and nowhere else.  ``sfcprs2``,
    ``integ_moist`` and the vertical-interpolation surface pseudo-level
    are uses real.exe is DEFINED for, and they keep the raw value, so the
    prognostic state stays bit-for-bit what it was on every lane --
    including the explicit ``use_sh_qv=True`` lane, whose prognostic qv IS
    the interpolated surface value and which WRF likewise never floors.
    Only cells the undershoot pushed below the floor move, and only in the
    published array, which is why no artifact whose surface moisture stays
    above 1e-6 changes at all.

    Returns the floored array and a receipt; the receipt is empty when
    nothing was floored, and that is the overwhelmingly common case.
    """
    surface_qv = np.asarray(surface_qv, dtype=np.float64)
    pressure = np.asarray(surface_pressure, dtype=np.float64)
    if pressure.shape != surface_qv.shape:
        raise ValueError(
            "surface pressure and surface mixing ratio shapes differ")
    below = ((pressure < _WRF_QV_MIN_P_SAFE)
             & (surface_qv < _WRF_QV_MIN_VALUE))
    if not below.any():
        return surface_qv, {}
    floored = np.where(below, _WRF_QV_MIN_VALUE, surface_qv)
    count = int(np.count_nonzero(below))
    negative = int(np.count_nonzero(below & (surface_qv < 0.0)))
    minimum = float(np.min(surface_qv[below]))
    receipt = {
        "policy": "flag-sh-surface-qv-floored-to-wrf-qv-min-value",
        "wrf_reference": dict(_SURFACE_QV_FLOOR_WRF_REFERENCE),
        "qv_min_value": _WRF_QV_MIN_VALUE,
        "qv_min_p_safe": _WRF_QV_MIN_P_SAFE,
        "floored_cells": count,
        "negative_cells": negative,
        "min_pre_floor": minimum,
    }
    print(
        f"surface moisture floor: {count} FLAG_SH 2 m mixing-ratio "
        f"value(s) below WRF's qv_min_value {_WRF_QV_MIN_VALUE:g} floored "
        f"to it ({negative} of them negative; min pre-floor "
        f"{minimum:.6g}); WPS's overshooting sixteen_pt operator "
        "undershoots SPECHUMD that is exactly zero over high dry terrain, "
        "and real.exe carries that sign into grid%q2 unfloored "
        "(module_initialize_real.F:1157,1257)",
        file=sys.stderr)
    return floored, receipt


def _mixing_ratio_to_relative_humidity_serial(
        temperature, pressure, mixing_ratio, *, allow_wps_undershoot=False):
    """Diagnose and validate one contiguous relative-humidity chunk."""
    temperature, pressure, qv = np.broadcast_arrays(
        np.asarray(temperature, dtype=np.float64),
        np.asarray(pressure, dtype=np.float64),
        np.asarray(mixing_ratio, dtype=np.float64),
    )
    minimum_qv = (_WPS_SPFH_UNDERSHOOT_LOWER_BOUND
                  / (1.0 - _WPS_SPFH_UNDERSHOOT_LOWER_BOUND)
                  if allow_wps_undershoot else 0.0)
    if (not np.isfinite(temperature).all()
            or not np.isfinite(pressure).all()
            or not np.isfinite(qv).all()
            or np.any(pressure <= 0.0) or np.any(qv < minimum_qv)):
        qv_requirement = ("inside the bounded WPS undershoot envelope"
                          if allow_wps_undershoot else "non-negative")
        raise ValueError(
            "temperature, pressure, and mixing ratio must be finite; "
            f"pressure must be positive and mixing ratio {qv_requirement}")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        es_hpa = (10.0 * c.SVP1) * np.exp(
            c.SVP2 * (temperature - c.SVPT0) / (temperature - c.SVP3))
        vapor_hpa = qv * (pressure / 100.0) / (qv + 0.622)
        rh = 100.0 * vapor_hpa / es_hpa
    if not np.isfinite(rh).all():
        raise ValueError("diagnosed relative humidity is non-finite")
    return rh


def _mixing_ratio_to_relative_humidity(
        temperature, pressure, mixing_ratio, *, allow_wps_undershoot=False,
        column_workers=1):
    """Diagnose WPS/WRF relative humidity (%) from dry-air mixing ratio.

    HRRR supplies specific humidity.  WPS horizontally maps SPECHUMD into its
    intermediate output; real.exe's FLAG_SH branch converts it to qv_gc
    and overwrites rh_gc before vertical interpolation.  This is the algebraic
    inverse of its later ``rh_to_mxrat1`` Bolton saturation relation, before
    that routine's 0--100 percent RH clipping.
    """
    if not isinstance(allow_wps_undershoot, (bool, np.bool_)):
        raise TypeError("allow_wps_undershoot must be boolean")
    workers = _column_worker_count(column_workers)
    temperature, pressure, qv = np.broadcast_arrays(
        np.asarray(temperature, dtype=np.float64),
        np.asarray(pressure, dtype=np.float64),
        np.asarray(mixing_ratio, dtype=np.float64),
    )
    if (workers == 1 or temperature.ndim == 0
            or temperature.shape[0] < 2):
        return _mixing_ratio_to_relative_humidity_serial(
            temperature, pressure, qv,
            allow_wps_undershoot=allow_wps_undershoot)

    chunks = _axis0_chunks(temperature.shape[0], workers)
    output = np.empty(temperature.shape, dtype=np.float64)
    with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
        futures = [
            executor.submit(
                _fill_axis0_chunk, output, start, stop,
                _mixing_ratio_to_relative_humidity_serial,
                (temperature[start:stop], pressure[start:stop],
                 qv[start:stop]),
                {"allow_wps_undershoot": allow_wps_undershoot})
            for start, stop in chunks
        ]
        for future in futures:
            future.result()
    return output


def _potential_temperature_from_temperature_serial(temperature, pressure):
    """Evaluate WRF's T-to-theta relation over one contiguous chunk."""
    return temperature * (c.P0 / pressure) ** c.RCP


def _potential_temperature_from_temperature(
        temperature, pressure, *, column_workers=1):
    """Column-parallel, byte-stable WRF T-to-theta conversion."""
    workers = _column_worker_count(column_workers)
    temperature, pressure = np.broadcast_arrays(
        np.asarray(temperature, dtype=np.float64),
        np.asarray(pressure, dtype=np.float64),
    )
    if (workers == 1 or temperature.ndim == 0
            or temperature.shape[0] < 2):
        return _potential_temperature_from_temperature_serial(
            temperature, pressure)
    chunks = _axis0_chunks(temperature.shape[0], workers)
    output = np.empty(temperature.shape, dtype=np.float64)
    with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
        futures = [
            executor.submit(
                _fill_axis0_chunk, output, start, stop,
                _potential_temperature_from_temperature_serial,
                (temperature[start:stop], pressure[start:stop]))
            for start, stop in chunks
        ]
        for future in futures:
            future.result()
    return output


def _temperature_from_potential_temperature_serial(theta, pressure):
    """Evaluate WRF's theta-to-T relation over one contiguous chunk."""
    return theta * (pressure / c.P0) ** c.RCP


def _temperature_from_potential_temperature(
        theta, pressure, *, column_workers=1):
    """Column-parallel, byte-stable WRF theta-to-T conversion."""
    workers = _column_worker_count(column_workers)
    theta, pressure = np.broadcast_arrays(
        np.asarray(theta, dtype=np.float64),
        np.asarray(pressure, dtype=np.float64),
    )
    if workers == 1 or theta.ndim == 0 or theta.shape[0] < 2:
        return _temperature_from_potential_temperature_serial(theta, pressure)
    chunks = _axis0_chunks(theta.shape[0], workers)
    output = np.empty(theta.shape, dtype=np.float64)
    with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
        futures = [
            executor.submit(
                _fill_axis0_chunk, output, start, stop,
                _temperature_from_potential_temperature_serial,
                (theta[start:stop], pressure[start:stop]))
            for start, stop in chunks
        ]
        for future in futures:
            future.result()
    return output


def _moist_specific_volume_serial(theta, qv, pressure):
    """Evaluate one contiguous chunk of WRF moist specific volume."""
    theta_m = theta * (1.0 + c.RVOVRD * qv)
    return c.RD * theta_m * (pressure / c.P0) ** c.RCP / pressure


def _moist_specific_volume(theta, qv, pressure, *, column_workers=1):
    """Column-parallel, byte-stable moist specific-volume diagnostic."""
    workers = _column_worker_count(column_workers)
    theta, qv, pressure = np.broadcast_arrays(
        np.asarray(theta, dtype=np.float64),
        np.asarray(qv, dtype=np.float64),
        np.asarray(pressure, dtype=np.float64),
    )
    if workers == 1 or theta.ndim == 0 or theta.shape[0] < 2:
        return _moist_specific_volume_serial(theta, qv, pressure)
    chunks = _axis0_chunks(theta.shape[0], workers)
    output = np.empty(theta.shape, dtype=np.float64)
    with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
        futures = [
            executor.submit(
                _fill_axis0_chunk, output, start, stop,
                _moist_specific_volume_serial,
                (theta[start:stop], qv[start:stop], pressure[start:stop]))
            for start, stop in chunks
        ]
        for future in futures:
            future.result()
    return output


def _wrf_flag_sh_surface_specific_humidity(
        q2, spfh, pressure, *, force_fallback=None):
    """Apply real.exe's whole-domain FLAG_SH surface fallback.

    WPS's surface SPECHUMD occupies metgrid level one.  WRF checks its first
    valid horizontal point; when that value is below ``1e-6``, it replaces the
    complete surface field with the nearest atmospheric SPECHUMD level before
    converting specific humidity to mixing ratio.  Native HRRR stores that
    surface value separately as Q2, so reproduce the same decision explicitly.
    """
    q2 = np.asarray(q2, dtype=np.float64)
    spfh = np.asarray(spfh, dtype=np.float64)
    pressure = np.asarray(pressure, dtype=np.float64)
    if spfh.ndim != 3 or pressure.shape != spfh.shape:
        raise ValueError("SPFH and pressure must share shape (level, y, x)")
    if q2.shape != spfh.shape[1:]:
        raise ValueError("Q2 must match the SPFH horizontal grid")
    if force_fallback is not None and not isinstance(
            force_fallback, (bool, np.bool_)):
        raise TypeError("force_fallback must be boolean or None")
    fallback = (q2[0, 0] < 1.0e-6
                if force_fallback is None else bool(force_fallback))
    if not fallback:
        return q2
    nearest = 0 if pressure[-1, 0, 0] < pressure[0, 0, 0] else -1
    return spfh[nearest].copy()


def _surface_relative_humidity(dewpoint, temperature):
    """ungrib's 2 m relative humidity from D2/T2 (surface RH level 1).

    WPS v4.6 ``ungrib/src/rrpr.F:compute_rh_dewpt`` (:1168-1185):
    ``RH2m = 100 * exp((Xlv/Rv) * (1/T2 - 1/D2))`` with ``Xlv = 2.5e6`` and
    ``Rv = 461.5`` -- a constant-latent-heat Clausius-Clapeyron ratio, not
    the Bolton/Magnus curve used elsewhere in this module.
    The value is deliberately NOT clipped to 100: WRF interpolates ``rh_gc``
    as delivered and clips only inside ``rh_to_mxrat1``
    (module_initialize_real.F:7392-7399), which
    :func:`_saturation_mixing_ratio` mirrors.
    """
    dewpoint = _host(dewpoint)
    temperature = _host(temperature)
    xlv_over_rv = 2.5e6 / 461.5
    return 100.0 * np.exp(
        xlv_over_rv * (1.0 / temperature - 1.0 / dewpoint))


def surface_pressure_from_surface(psfc_in, source_orography, terrain,
                                  surface_temperature, surface_qv):
    """WRF ``sfcprs2`` surface-to-surface pressure adjustment.

    ``source_orography`` is the invariant geopotential-height surface that
    belongs to ``psfc_in`` (the retained profile's declared ``SOILHGT``
    artifact); ``terrain`` is the target WRF terrain.
    """
    psfc = _host(psfc_in)
    source_z = _host(source_orography)
    target_z = _host(terrain)
    temperature = _host(surface_temperature)
    qv = _host(surface_qv)
    if len({a.shape for a in (psfc, source_z, target_z, temperature, qv)}) != 1:
        raise ValueError("surface-pressure input shapes differ")
    # Equal source and target terrain is WRF's DEFINED behavior: sfcprs2
    # returns psfc_in unchanged (exp(0) = 1, module_initialize_real.F:8496).
    # Generic code performs that no-op; protecting a specific case against
    # a silently regressed source-orography artifact (e.g. a SOILHGT
    # regenerated byte-equal to HGT_M) is the job of that case's pinned
    # wrfinput gates, which fail by orders of magnitude on such data.
    virtual_temperature = temperature * (1.0 + 0.608 * qv)
    if (not np.isfinite(virtual_temperature).all()
            or np.any(virtual_temperature <= 0.0)):
        raise ValueError("surface virtual temperature must be finite and positive")
    result = psfc * np.exp(
        c.G * (source_z - target_z) / (c.RD * virtual_temperature))
    if not np.isfinite(result).all() or np.any(result <= 0.0):
        raise ValueError("adjusted surface pressure is invalid")
    return result


def _integrate_moisture_scalar_reference(
        qv, pressure, temperature, height, psfc, tsfc, qsfc,
        surface_height):
    """Pre-vectorization row/column oracle retained for byte-parity tests."""
    """Float64 transcription of ``module_initialize_real.F:integ_moist``.

    Inputs are pressure levels only, normalized here to bottom-to-top order;
    the separate surface values play the WPS surface pseudo-level role.
    Returns bottom-to-top dry pressure and the column-integrated vapor
    pressure removed from surface pressure.
    """
    order = np.argsort(-pressure[:, 0, 0])
    p = pressure[order].copy()
    q = qv[order].copy()
    t = temperature[order].copy()
    z = height[order].copy()
    nlev, ny, nx = p.shape
    pd = np.empty_like(p)
    intq = np.zeros((ny, nx), dtype=np.float64)
    for j in range(ny):
        for i in range(nx):
            above = np.flatnonzero(p[:, j, i] < psfc[j, i])
            if above.size == 0:
                raise ValueError(
                    f"no pressure level above the surface at column ({j}, {i})")
            ka = int(above[0])
            cumulative = np.zeros(nlev, dtype=np.float64)
            pd[-1, j, i] = p[-1, j, i]
            for k in range(nlev - 2, ka - 1, -1):
                rhobar = 0.5 * (p[k, j, i] / (c.RD * t[k, j, i])
                                + p[k + 1, j, i] / (c.RD * t[k + 1, j, i]))
                qbar = 0.5 * (q[k, j, i] + q[k + 1, j, i])
                dz = z[k + 1, j, i] - z[k, j, i]
                if dz > 0.0:
                    cumulative[k] = (cumulative[k + 1]
                                     + c.G * qbar * rhobar / (1.0 + qbar) * dz)
                else:
                    cumulative[k] = cumulative[k + 1]
                pd[k, j, i] = p[k, j, i] - cumulative[k]
            rhobar = 0.5 * (psfc[j, i] / (c.RD * tsfc[j, i])
                            + p[ka, j, i] / (c.RD * t[ka, j, i]))
            qbar = 0.5 * (qsfc[j, i] + q[ka, j, i])
            dz = z[ka, j, i] - surface_height[j, i]
            surface_intq = cumulative[ka]
            if dz > 0.1:
                surface_intq += c.G * qbar * rhobar / (1.0 + qbar) * dz
            intq[j, i] = surface_intq
            pd[:ka, j, i] = p[:ka, j, i] - surface_intq
            # ka was assigned in the loop unless it is the topmost level.
            pd[ka:, j, i] = p[ka:, j, i] - cumulative[ka:]
    return pd, intq, order


def _integrate_moisture_vectorized_slab(
        qv, pressure, temperature, height, psfc, tsfc, qsfc,
        surface_height, *, order, out_pd, out_intq, row_offset=0):
    """Evaluate one contiguous row slab of WRF ``integ_moist``.

    Vertical recurrence order remains top-down within every column, while
    NumPy evaluates all independent horizontal columns in one native loop.
    ``order`` is resolved once from the complete domain so worker partitioning
    cannot change the source-level ordering contract.
    """
    nlev, ny, nx = pressure.shape
    ka = np.full((ny, nx), nlev, dtype=np.intp)
    for k in range(nlev):
        source_k = int(order[k])
        take = (ka == nlev) & (pressure[source_k] < psfc)
        ka[take] = k
    missing = ka == nlev
    if bool(np.any(missing)):
        j, i = np.argwhere(missing)[0]
        raise ValueError(
            "no pressure level above the surface at column "
            f"({int(j) + int(row_offset)}, {i})")

    running = np.zeros_like(psfc, dtype=pressure.dtype)
    surface_intq = np.zeros_like(psfc, dtype=pressure.dtype)
    out_pd[-1] = pressure[int(order[-1])]
    for k in range(nlev - 2, -1, -1):
        source_k = int(order[k])
        source_kp1 = int(order[k + 1])
        pk = pressure[source_k]
        pkp1 = pressure[source_kp1]
        qk = qv[source_k]
        qkp1 = qv[source_kp1]
        tk = temperature[source_k]
        tkp1 = temperature[source_kp1]
        zk = height[source_k]
        zkp1 = height[source_kp1]
        active = ka <= k
        rhobar = 0.5 * (
            pk / (c.RD * tk) + pkp1 / (c.RD * tkp1))
        qbar = 0.5 * (qk + qkp1)
        dz = zkp1 - zk
        increment = c.G * qbar * rhobar / (1.0 + qbar) * dz
        running = np.where(
            active,
            running + np.where(dz > 0.0, increment, 0.0),
            0.0)
        out_pd[k] = pk - running
        surface_intq = np.where(ka == k, running, surface_intq)

    p_ka = np.empty_like(psfc, dtype=pressure.dtype)
    q_ka = np.empty_like(psfc, dtype=qv.dtype)
    t_ka = np.empty_like(psfc, dtype=temperature.dtype)
    z_ka = np.empty_like(psfc, dtype=height.dtype)
    for k in range(nlev):
        selected = ka == k
        source_k = int(order[k])
        p_ka[selected] = pressure[source_k][selected]
        q_ka[selected] = qv[source_k][selected]
        t_ka[selected] = temperature[source_k][selected]
        z_ka[selected] = height[source_k][selected]
    rhobar = 0.5 * (
        psfc / (c.RD * tsfc) + p_ka / (c.RD * t_ka))
    qbar = 0.5 * (qsfc + q_ka)
    dz = z_ka - surface_height
    surface_increment = c.G * qbar * rhobar / (1.0 + qbar) * dz
    surface_intq = np.where(
        dz > 0.1, surface_intq + surface_increment, surface_intq)
    for k in range(nlev):
        below = k < ka
        if bool(np.any(below)):
            source_k = int(order[k])
            out_pd[k, below] = (
                pressure[source_k, below] - surface_intq[below])
    out_intq[...] = surface_intq


def _integrate_moisture(qv, pressure, temperature, height, psfc, tsfc, qsfc,
                        surface_height, *, column_workers=1):
    """Column-parallel transcription of WRF ``integ_moist``.

    The original Python row/column loop is vectorized within each row slab.
    Large domains may additionally divide those independent slabs among an
    explicit number of setup-only host threads.  Every column keeps the same
    vertical operation order, and source-level ordering is resolved once from
    the complete domain, so the threaded result is byte-identical to the
    single-thread vectorized implementation.
    """
    workers = _column_worker_count(column_workers)
    pressure = np.asarray(pressure)
    if pressure.ndim != 3:
        raise ValueError("pressure must have shape (level, y, x)")
    order = np.argsort(-pressure[:, 0, 0])
    pd = np.empty(pressure.shape, dtype=pressure.dtype)
    intq = np.empty(pressure.shape[1:], dtype=pressure.dtype)
    if workers == 1 or pressure.shape[1] < 2:
        _integrate_moisture_vectorized_slab(
            qv, pressure, temperature, height, psfc, tsfc, qsfc,
            surface_height, order=order, out_pd=pd, out_intq=intq)
        return pd, intq, order

    chunks = _axis0_chunks(pressure.shape[1], workers)
    with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
        futures = [
            executor.submit(
                _integrate_moisture_vectorized_slab,
                qv[:, start:stop], pressure[:, start:stop],
                temperature[:, start:stop], height[:, start:stop],
                psfc[start:stop], tsfc[start:stop], qsfc[start:stop],
                surface_height[start:stop], order=order,
                out_pd=pd[:, start:stop], out_intq=intq[start:stop],
                row_offset=start)
            for start, stop in chunks
        ]
        for future in futures:
            future.result()
    return pd, intq, order


def _make_real_base_serial(coord: VerticalCoord, terrain: np.ndarray,
                           p_top: float, base_temp: float,
                           hypsometric_opt: int = 1, *,
                           row_offset: int = 0) -> BaseState:
    """WRF ``module_initialize_real.F`` analytic hydrostatic base state.

    The base geopotential integration is keyed on ``hypsometric_opt``
    exactly as WRF (module_initialize_real.F:3811-3825): opt 1 is the
    discrete ``phb(k+1) = phb(k) - dnw(k)*(c1h*mub+c2h)*alb(k)``
    recurrence; opt 2 the log-pressure form ``phb(k+1) = phb(k) +
    alb(k)*phm*LOG(pfd/pfu)`` on the base-state reference dry pressures
    ``pf/ph = c3*MUB + c4 + p_top`` (F:3816-3822).
    """
    finalize_vertical_coord(coord, p_top)
    terrain = np.array(terrain, dtype=np.float64, copy=True)
    lapse = 50.0
    iso_temperature = 200.0
    root = (base_temp / lapse) ** 2 - (
        2.0 * c.G * terrain / (lapse * c.RD))
    if np.any(root <= 0.0):
        raise ValueError("terrain is outside the analytic base-state range")
    ps_base = c.P0 * np.exp(-base_temp / lapse + np.sqrt(root))
    mub = ps_base - p_top
    pb = (coord.c3h[:, None, None] * mub[None]
          + coord.c4h[:, None, None] + p_top)
    # WRF checks the FULL-level reference column (nest_init_utils.F:1166-1167)
    # and this base state divides that same column at :732 -- an inverted
    # full-level pair silently produces a negative-thickness base layer, so
    # the half-level column alone is not the constraint.
    pb_full = (coord.c3f[:, None, None] * mub[None]
               + coord.c4f[:, None, None] + p_top)
    if (np.any(pb <= 0.0) or not np.all(np.diff(pb, axis=0) < 0.0)
            or not np.all(np.diff(pb_full, axis=0) < 0.0)):
        raise ValueError(hybrid_column_ordering_refusal(
            coord, p_top, ps_base, terrain=terrain,
            quantity="hybrid base pressure", row_offset=row_offset)
            or "hybrid base pressure is not monotonic")
    temperature = np.maximum(
        iso_temperature, base_temp + lapse * np.log(pb / c.P0))
    thb = temperature * (c.P0 / pb) ** c.RCP
    alb = c.RD * thb * (pb / c.P0) ** c.RCP / pb
    phb = np.empty((coord.znw.size,) + terrain.shape, dtype=np.float64)
    phb[0] = c.G * terrain
    if hypsometric_opt == 1:
        for k in range(coord.dnw.size):
            phb[k + 1] = (phb[k] - coord.dnw[k]
                          * (coord.c1h[k] * mub + coord.c2h[k]) * alb[k])
    elif hypsometric_opt == 2:
        # module_initialize_real.F:3816-3822 (indices shifted to 0-based:
        # WRF's k/k-1 full levels and k-1 half level become k+1/k and k).
        for k in range(coord.dnw.size):
            pfu = coord.c3f[k + 1] * mub + coord.c4f[k + 1] + p_top
            pfd = coord.c3f[k] * mub + coord.c4f[k] + p_top
            phm = coord.c3h[k] * mub + coord.c4h[k] + p_top
            phb[k + 1] = phb[k] + alb[k] * phm * np.log(pfd / pfu)
    else:
        raise ValueError(
            f"hypsometric_opt must be 1 or 2, got {hypsometric_opt}")
    return BaseState(mub=mub, p_top=float(p_top), pb=pb, alb=alb, thb=thb,
                     phb=phb, terrain_z=terrain)


def _fill_real_base_rows(output, coord, terrain, start, stop, p_top,
                         base_temp, hypsometric_opt):
    """Build then immediately copy one independent analytic-base row tile."""
    part = _make_real_base_serial(
        coord, terrain[start:stop], p_top, base_temp, hypsometric_opt,
        row_offset=start)
    output.mub[start:stop] = part.mub
    output.pb[:, start:stop] = part.pb
    output.alb[:, start:stop] = part.alb
    output.thb[:, start:stop] = part.thb
    output.phb[:, start:stop] = part.phb
    output.terrain_z[start:stop] = part.terrain_z


def _make_real_base(coord: VerticalCoord, terrain: np.ndarray, p_top: float,
                    base_temp: float, hypsometric_opt: int = 1, *,
                    column_workers=1) -> BaseState:
    """Build the exact analytic base over independent horizontal chunks."""
    workers = _column_worker_count(column_workers)
    finalize_vertical_coord(coord, p_top)
    terrain = np.asarray(terrain, dtype=np.float64)
    if workers == 1 or terrain.shape[0] < 2:
        return _make_real_base_serial(
            coord, terrain, p_top, base_temp, hypsometric_opt)

    ny, nx = terrain.shape
    nz = coord.dnw.size
    output = BaseState(
        mub=np.empty((ny, nx), dtype=np.float64),
        p_top=float(p_top),
        pb=np.empty((nz, ny, nx), dtype=np.float64),
        alb=np.empty((nz, ny, nx), dtype=np.float64),
        thb=np.empty((nz, ny, nx), dtype=np.float64),
        phb=np.empty((nz + 1, ny, nx), dtype=np.float64),
        terrain_z=np.empty((ny, nx), dtype=np.float64),
    )
    # More tiles than workers bounds live temporary BaseStates and avoids a
    # serial multi-gigabyte concatenate on large domains.
    chunks = _axis0_chunks(ny, min(ny, workers * 4))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _fill_real_base_rows, output, coord, terrain, start, stop,
                p_top, base_temp, hypsometric_opt)
            for start, stop in chunks
        ]
        for future in futures:
            future.result()
    return output


def _pressure_at_u(pressure):
    out = np.empty(pressure.shape[:2] + (pressure.shape[2] + 1,),
                   dtype=np.float64)
    out[..., 0] = pressure[..., 0]
    out[..., -1] = pressure[..., -1]
    out[..., 1:-1] = 0.5 * (pressure[..., :-1] + pressure[..., 1:])
    return out


def _pressure_at_v(pressure):
    out = np.empty((pressure.shape[0], pressure.shape[1] + 1,
                    pressure.shape[2]), dtype=np.float64)
    out[:, 0, :] = pressure[:, 0, :]
    out[:, -1, :] = pressure[:, -1, :]
    out[:, 1:-1, :] = 0.5 * (pressure[:, :-1, :] + pressure[:, 1:, :])
    return out


def _slice_base_rows(base: BaseState, start: int, stop: int) -> BaseState:
    """View one contiguous mass-grid row slab of an analytic base state."""
    return BaseState(
        mub=base.mub[start:stop], p_top=base.p_top,
        pb=base.pb[:, start:stop], alb=base.alb[:, start:stop],
        thb=base.thb[:, start:stop], phb=base.phb[:, start:stop],
        terrain_z=base.terrain_z[start:stop])


def _rebalance_moist_pressure_serial(pressure_guess, qv, dry_mass, base,
                                     coord, *, out=None):
    """Integrate the discrete WRF moist w-balance pressure recurrence.

    This is the initialization counterpart of ``pg_buoy_w``: it chooses
    perturbation-pressure differences so the large-step vertical pressure
    gradient, dry-mass perturbation, and vapor loading cancel row by row.
    """
    if out is None:
        out = np.empty_like(pressure_guess)
    elif out.shape != pressure_guess.shape or out.dtype != pressure_guess.dtype:
        raise ValueError("rebalance output shape/dtype differs from pressure")
    perturbation = out
    mup = dry_mass - base.mub
    nz = pressure_guess.shape[0]
    # WRF initializes the top half-level pressure from the rigid-lid row,
    # then integrates downward (module_initialize_real/ideal qvf1/qvf2
    # recurrence).  Anchoring at the top avoids importing horizontally
    # varying surface interpolation error into every pressure level.
    cq = 1.0 / (1.0 + qv[-1])
    load = qv[-1] * cq
    perturbation[-1] = (
        -0.5 * (coord.c1f[nz] * mup
                + load * (coord.c1f[nz] * base.mub + coord.c2f[nz]))
        / coord.rdnw[nz - 1] / cq)
    for k in range(nz - 2, -1, -1):
        kw = k + 1
        qbar = 0.5 * (qv[k] + qv[k + 1])
        cq = 1.0 / (1.0 + qbar)
        load = qbar * cq
        perturbation[k] = (
            perturbation[k + 1]
            - (coord.c1f[kw] * mup
               + load * (coord.c1f[kw] * base.mub + coord.c2f[kw]))
            / cq / coord.rdn[kw])
    np.add(base.pb, perturbation, out=out)
    if not np.isfinite(out).all() or np.any(out <= 0.0):
        raise ValueError("moist hydrostatic pressure recurrence failed")
    return out


def _rebalance_moist_pressure(pressure_guess, qv, dry_mass, base, coord, *,
                              column_workers=1):
    """Run the unchanged vertical recurrence over parallel row slabs."""
    workers = _column_worker_count(column_workers)
    pressure = np.empty_like(pressure_guess)
    if workers == 1 or pressure_guess.shape[1] < 2:
        return _rebalance_moist_pressure_serial(
            pressure_guess, qv, dry_mass, base, coord, out=pressure)

    chunks = _axis0_chunks(pressure_guess.shape[1], workers)
    with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
        futures = [
            executor.submit(
                _rebalance_moist_pressure_serial,
                pressure_guess[:, start:stop], qv[:, start:stop],
                dry_mass[start:stop], _slice_base_rows(base, start, stop),
                coord, out=pressure[:, start:stop])
            for start, stop in chunks
        ]
        for future in futures:
            future.result()
    return pressure


def _half_level_height_agl(base, coord, dry_mass, alpha, *,
                           hypsometric_opt: int = 1) -> np.ndarray:
    """FP64 half-level (mass point) heights AGL of the hydrostatic column.

    The same discrete operator :func:`_fp32_geopotential_split` quantizes
    -- opt 1 ``d(phi) = -dnw*(c1h*mu+c2h)*alpha``, opt 2 ``d(phi) =
    alpha*phm*log(pfd/pfu)`` on the total dry mass -- evaluated in plain
    float64 and referenced to the surface, so the caller gets geometric
    metres above local terrain without any FP32 quantization.  Used to
    place configured initial-state perturbations on the UNPERTURBED
    column before the perturbed geopotential is formed.
    """
    if hypsometric_opt not in (1, 2):
        raise ValueError(
            f"hypsometric_opt must be 1 or 2, got {hypsometric_opt}")
    mu = np.asarray(dry_mass, dtype=np.float64)
    target = np.asarray(alpha, dtype=np.float64)
    nz = coord.dnw.size
    dphi = np.empty_like(target)
    if hypsometric_opt == 2:
        c3f = np.asarray(coord.c3f, dtype=np.float64)[:, None, None]
        c4f = np.asarray(coord.c4f, dtype=np.float64)[:, None, None]
        c3h = np.asarray(coord.c3h, dtype=np.float64)[:, None, None]
        c4h = np.asarray(coord.c4h, dtype=np.float64)[:, None, None]
        p_top = float(base.p_top)
        pfu = c3f[1:] * mu[None] + c4f[1:] + p_top
        pfd = c3f[:-1] * mu[None] + c4f[:-1] + p_top
        phm = c3h * mu[None] + c4h + p_top
        dphi[...] = target * phm * np.log(pfd / pfu)
    else:
        dnw = np.asarray(coord.dnw, dtype=np.float64)[:, None, None]
        increment = (np.asarray(coord.c1h, dtype=np.float64)[:, None, None]
                     * mu[None]
                     + np.asarray(coord.c2h, dtype=np.float64)[:, None, None])
        dphi[...] = -dnw * increment * target
    phi_agl = np.zeros((nz + 1,) + mu.shape, dtype=np.float64)
    np.cumsum(dphi, axis=0, out=phi_agl[1:])
    return (0.5 * (phi_agl[:-1] + phi_agl[1:])) / c.G


def _fp32_geopotential_split_serial(base, coord, dry_mass, alpha,
                                    hypsometric_opt: int = 1):
    """Choose phi' ulps that best preserve the float64 hydrostatic layers.

    Over high terrain the lowest explicit eta layers are only a few metres
    thick.  Forming ``phi=phb+phi'`` in FP32 can otherwise lose enough of a
    layer geopotential difference to create an artificial vertical impulse.
    This setup-only quantizer tests the nearest phi' ulp and its two
    neighbors against the exact diagnostic-alpha equation at every level.

    The target layer thickness and the diagnostic operator are keyed on
    ``hypsometric_opt`` to match the runtime EOS (calc_p_alpha) and WRF's
    real-init geopotential integration: opt 1 inverts ``alpha =
    -d(phi)*rdnw/(c1h*mu+c2h)``; opt 2 integrates ``d(phi) =
    alt*phm*LOG(pfd/pfu)`` on the TOTAL dry-mass reference pressures
    (module_initialize_real.F:3970-3981) and diagnoses ``alt =
    d(phi)/phm/LOG(pfd/pfu)`` (F:4002-4010), all in the FP32 arithmetic
    the device kernel uses.
    """
    if hypsometric_opt not in (1, 2):
        raise ValueError(
            f"hypsometric_opt must be 1 or 2, got {hypsometric_opt}")
    phb = np.asarray(base.phb, dtype=np.float32)
    dnw = np.asarray(coord.dnw, dtype=np.float32)
    rdnw = np.asarray(coord.rdnw, dtype=np.float32)
    increment = np.asarray(
        coord.c1h[:, None, None] * dry_mass[None]
        + coord.c2h[:, None, None], dtype=np.float32)
    target = np.asarray(alpha, dtype=np.float32)
    if hypsometric_opt == 2:
        # Per-layer reference dry pressures on the TOTAL dry mass (WRF
        # MU0 = mub + mu'), in the kernel's FP32 arithmetic.
        c3f = np.asarray(coord.c3f, dtype=np.float32)
        c4f = np.asarray(coord.c4f, dtype=np.float32)
        c3h = np.asarray(coord.c3h, dtype=np.float32)
        c4h = np.asarray(coord.c4h, dtype=np.float32)
        mu32 = np.asarray(dry_mass, dtype=np.float32)
        pt32 = np.float32(base.p_top)
    php = np.zeros_like(phb, dtype=np.float32)
    total_low = np.asarray(phb[0] + php[0], dtype=np.float32)
    for k in range(coord.dnw.size):
        if hypsometric_opt == 2:
            pfu = c3f[k + 1] * mu32 + c4f[k + 1] + pt32
            pfd = c3f[k] * mu32 + c4f[k] + pt32
            phm = c3h[k] * mu32 + c4h[k] + pt32
            log_ratio = np.log(pfd / pfu)              # float32
            desired_dphi = np.asarray(target[k] * phm * log_ratio,
                                      dtype=np.float32)
        else:
            desired_dphi = np.asarray(-dnw[k] * increment[k] * target[k],
                                      dtype=np.float32)
        desired_total = np.asarray(total_low + desired_dphi, dtype=np.float32)
        centre = np.asarray(desired_total - phb[k + 1], dtype=np.float32)
        candidates = (
            np.nextafter(centre, np.float32(-np.inf)), centre,
            np.nextafter(centre, np.float32(np.inf)),
        )
        best = None
        best_error = None
        best_total = None
        for candidate in candidates:
            total = np.asarray(phb[k + 1] + candidate, dtype=np.float32)
            dphi = np.asarray(total - total_low, dtype=np.float32)
            if hypsometric_opt == 2:
                diagnosed = np.asarray(
                    np.asarray(dphi / phm, dtype=np.float32) / log_ratio,
                    dtype=np.float32)
            else:
                diagnosed = np.asarray(
                    np.asarray(-dphi * rdnw[k], dtype=np.float32)
                    / increment[k], dtype=np.float32)
            error = np.abs(diagnosed.astype(np.float64)
                           - target[k].astype(np.float64))
            if best is None:
                best, best_error, best_total = candidate, error, total
            else:
                choose = error < best_error
                best = np.where(choose, candidate, best)
                best_error = np.where(choose, error, best_error)
                best_total = np.where(choose, total, best_total)
        php[k + 1] = best
        total_low = best_total
    return php


def _fp32_geopotential_split(base, coord, dry_mass, alpha,
                             hypsometric_opt: int = 1, *,
                             column_workers=1):
    """Quantize geopotential independently over parallel mass-grid slabs."""
    workers = _column_worker_count(column_workers)
    if workers == 1 or dry_mass.shape[0] < 2:
        return _fp32_geopotential_split_serial(
            base, coord, dry_mass, alpha, hypsometric_opt)

    chunks = _axis0_chunks(dry_mass.shape[0], workers)
    with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
        futures = [
            executor.submit(
                _fp32_geopotential_split_serial,
                _slice_base_rows(base, start, stop), coord,
                dry_mass[start:stop], alpha[:, start:stop],
                hypsometric_opt)
            for start, stop in chunks
        ]
        parts = [future.result() for future in futures]
    return np.concatenate(parts, axis=1)


@dataclass(frozen=True)
class RealInitResult:
    state: DomainState
    coord: VerticalCoord
    base: BaseState
    surface_pressure: np.ndarray
    surface_qv: np.ndarray
    dry_mass: np.ndarray
    dry_pressure: np.ndarray
    total_pressure: np.ndarray
    total_geopotential: np.ndarray
    total_specific_volume: np.ndarray
    integrated_moisture_pressure: np.ndarray
    #: The hypsometric_opt the geopotential/base construction was keyed on;
    #: hydrostatic_residual grades the state against the same operator.
    hypsometric_opt: int = 1
    #: Immutable source/interpolated/state correspondence for native HRRR
    #: analyzed hydrometeors.  Other real-source routes carry an empty map.
    hydrometeor_initialization: dict[str, object] = field(default_factory=dict)
    #: mp_physics=28 only: what this initialization did and did not put in
    #: the aerosol fields, and which named function is expected to fill them
    #: next.  Empty for every other scheme.  Deliberately NOT folded into
    #: ``hydrometeor_initialization``: the aerosol policy applies to both the
    #: native-HRRR lane and the pressure-level/RH lane, and that map only
    #: exists on the former.
    aerosol_initialization: dict[str, object] = field(default_factory=dict)
    #: How many cells of ``surface_qv`` ABOVE were floored at WRF's
    #: ``qv_min_value`` on the way out, and how far below it they were
    #: (:func:`_floor_flag_sh_surface_mixing_ratio`).  Only the FLAG_SH
    #: lane can populate this: it is the one lane whose surface value WPS's
    #: overshooting sixteen_pt operator can push below zero and real.exe
    #: never floors.  Empty when nothing was floored -- the usual case --
    #: and always empty on the RH lane, whose
    #: :func:`_saturation_mixing_ratio` applies the same floor inline and
    #: so has no separate divergence to report.
    surface_moisture_floor: dict[str, object] = field(default_factory=dict)
    #: What the configured [perturbation] bubbles actually wrote into THIS
    #: domain's initial theta/qv: per-bubble cells touched and max deltas
    #: (:mod:`gpuwm.ingest.init_perturbation`).  Empty when no perturbation
    #: was requested -- the OFF path stores nothing and runs nothing.
    initial_perturbation: dict[str, object] = field(default_factory=dict)


def _wif_grid_latlon_from(grid, state):
    """``(lat2d, lon2d)`` for the WIF ingest from whatever the caller holds.

    FOUR CARRIERS, one meaning.  The tree does not have a single geodesy
    object every real front door passes around: ``gpuwm run`` and the
    tilestream cases hold a projection ``Grid``; the direct GFS/ERA5/
    mapped runners hold a geogrid ``static`` mapping; the prepared road
    restores a state that already carries the mass-point mirror
    ``initialize_prepared_physics`` stamps on it.  Accepting all of them
    is what makes this derivation reach every door instead of the one
    that happened to be wired first, and every one of them is the SAME
    mass-point lat/lon -- there is no per-source variant here, only a
    per-caller container.

    ``None`` when the caller holds none of them, which the caller above
    turns into a named refusal rather than a guess.
    """

    for candidate in (grid, state):
        if candidate is None:
            continue
        latlon_mass = getattr(candidate, "latlon_mass", None)
        if callable(latlon_mass):
            lat, lon = latlon_mass()
            return (lat, lon)
        latitude = getattr(candidate, "latitude_deg", None)
        longitude = getattr(candidate, "longitude_deg", None)
        if latitude is not None and longitude is not None:
            return (latitude, longitude)
        try:
            names = ("XLAT_M", "XLONG_M") if "XLAT_M" in candidate                 else ("XLAT", "XLONG")
            return (candidate[names[0]], candidate[names[1]])
        except (TypeError, KeyError, IndexError):
            pass
        if isinstance(candidate, (tuple, list)) and len(candidate) == 2:
            return (candidate[0], candidate[1])
    return None


def initialize_real(snapshot: HorizontalSnapshot, cfg: RunConfig,
                    coord: VerticalCoord, terrain, *, source_orography=None,
                    p_top=5000.0, sfcp_to_sfcp=True,
                    use_sh_qv=False,
                    column_workers=1,
                    preprocess_backend="cuda",
                    preprocess_workers=None,
                    cpu_bridge=None,
                    state_backend="cuda",
                    flag_sh_surface_fallback=None,
                    timing_report=None,
                    scratch_arena=None,
                    dycore_state_workspace=None,
                    initial_perturbation=None,
                    grid=None,
                    wif_grid_latlon=None,
                    wif_valid_date=None) -> RealInitResult:
    """Construct a moist, discretely hydrostatic :class:`DomainState`.

    The pressure-level/RH lane requires TT, RH, GHT, UU, VV, PSFC, T2,
    exactly one of D2 or RH2, U10, and V10.  Native HRRR requires
    per-column PRES, SPFH, RH, and Q2, and for
    WSM6/Thompson/Morrison/NSSL-2 requires analyzed QC/QR/QI/QS/QG.
    Kessler requires the same decoded inventory, retains QC/QR, and records
    WRF-real's explicit active-moist-package discard of QI/QS/QG.  MP off is
    refused because it cannot faithfully retain the five analyzed species.
    Classic
    Thompson's source-absent QNICE/QNRAIN moments are initialized to exact
    zero, matching real.exe, while the five analyzed HRRR mass categories are
    retained.  Aerosol-aware Thompson (mp_physics=28) retains the same five
    categories -- its Registry moist package is character for character
    mp=8's (Registry.EM_COMMON:3024 vs :3036) -- adds nc to the exact-zero
    source-absent moments, and deliberately leaves nwfa/nifa/nwfa2d/nifa2d
    at exact zero for ``microphysics_init`` to fill from thompson_init's
    synthetic profile; ``RealInitResult.aerosol_initialization`` is the
    receipt for that hand-off.  WRF's default
    ``use_sh_qv = .false.`` vertically interpolates HRRR RH and then diagnoses
    qv; direct SPFH/qv interpolation is available only when this function's
    explicit ``use_sh_qv=True`` option is selected.  With WRF's default
    ``use_sh_qv=False``, RH is diagnosed from the already horizontally mapped
    SPFH, temperature, and pressure exactly where real.exe handles FLAG_SH.
    Surface fields build WRF's ``use_surface`` pseudo-level that anchors
    every vertical-interpolation column.
    ``source_orography`` is mandatory for ``sfcp_to_sfcp`` unless the
    horizontal snapshot carries catalog-resolved ``SOURCE_OROGRAPHY``.
    ``preprocess_backend`` selects the interpolation engine independently of
    ``state_backend``.  The latter defaults to ``'cuda'`` so a GPU forecast
    may use CPU transforms and still receive the historical device state.
    Native export callers select ``state_backend='preprocess'``: CPU
    preprocessing then retains the completed setup/export state in NumPy
    host memory, while CUDA preprocessing keeps it on device.
    ``initial_perturbation`` is an optional
    :class:`gpuwm.ingest.init_perturbation.InitialStatePerturbation`:
    when present its theta bubbles are applied once to the final
    theta/qv columns before alpha/geopotential are formed (constant
    analyzed pressure and column mass, geopotential rebalanced -- the
    em_quarter_ss convention) and the per-bubble application stats are
    returned as ``RealInitResult.initial_perturbation``.  ``None`` (the
    default) runs not one instruction of that path.
    """
    if timing_report is not None:
        if not isinstance(timing_report, MutableMapping):
            raise TypeError("timing_report must be a mutable mapping")
        if len(timing_report):
            raise ValueError("timing_report must be empty")
    timing_start = timing_last = perf_counter()

    def mark_timing(name):
        nonlocal timing_last
        now = perf_counter()
        if timing_report is not None:
            timing_report[name] = now - timing_last
        timing_last = now

    if not sfcp_to_sfcp:
        raise ValueError(
            "sfcp_to_sfcp=false branch is not implemented: WRF requires "
            "the PMSL and pressure/GHT profile sfcprs3 reconstruction; "
            "copying the input PSFC is not supported")
    if not cfg.moist:
        raise ValueError("real initialization requires cfg.moist=True")
    if cfg.mp_physics == 28:
        # The aerosol-source selectors decide what this function is even
        # allowed to leave in nwfa/nifa, so they are checked HERE and not
        # only in validate_run_config: initialize_real is reachable with a
        # RunConfig that never went through it, and an mp=28 ingest that
        # silently accepted wif_input_opt=1 would promise a WIF metgrid
        # stream nothing in this module can read and then hand the
        # microphysics an all-zero aerosol field as if that were the
        # requested climatology.  Refusing by name is the same posture the
        # namelist importer already prints, which since lane/wif-default
        # prints a RESOLUTION (MP28_AEROSOL_SOURCE_DEFAULT) or a named
        # fallback (MP28_AEROSOL_SYNTHETIC_FALLBACK) rather than a
        # deviation notice.
        validate_aerosol_source_options(cfg)
    if not isinstance(use_sh_qv, (bool, np.bool_)):
        raise TypeError("use_sh_qv must be boolean")
    use_sh_qv = bool(use_sh_qv)
    column_workers = _column_worker_count(column_workers)
    if cfg.nz != coord.dnw.size:
        raise ValueError("RunConfig.nz and vertical coordinate differ")
    finalize_vertical_coord(coord, float(p_top))
    terrain = _host(terrain)
    if terrain.shape != (cfg.ny, cfg.nx):
        raise ValueError("terrain must have shape (ny, nx)")
    specific_markers = ("PRES", "SPFH", "Q2")
    marker_count = sum(name in snapshot.fields for name in specific_markers)
    if marker_count not in (0, len(specific_markers)):
        present = [name for name in specific_markers if name in snapshot.fields]
        missing_specific = [name for name in specific_markers
                            if name not in snapshot.fields]
        raise KeyError(
            "partial specific-humidity forcing inventory: "
            f"present={present}, missing={missing_specific}")
    has_specific_humidity = marker_count == len(specific_markers)
    if use_sh_qv and not has_specific_humidity:
        raise ValueError(
            "use_sh_qv=True requires PRES, SPFH, and Q2 forcing")
    if has_specific_humidity:
        if cfg.mp_physics == 0:
            raise ValueError(
                "native HRRR preparation with mp_physics=0 is refused: "
                "the MP-off state cannot faithfully retain analyzed "
                "QC/QR/QI/QS/QG, and a radiation-only analyzed-cloud "
                "carrier is out of scope")
        required = ("TT", "PRES", "SPFH", "GHT", "UU", "VV", "PSFC",
                    "T2", "Q2", "U10", "V10")
        if cfg.mp_physics in HRRR_ANALYZED_HYDROMETEOR_MP_PHYSICS:
            required += HRRR_ANALYZED_HYDROMETEORS
    else:
        surface_rh_markers = tuple(
            name for name in ("D2", "RH2") if name in snapshot.fields)
        if len(surface_rh_markers) != 1:
            raise KeyError(
                "pressure-level RH forcing requires exactly one of D2 or RH2")
        surface_rh_name = surface_rh_markers[0]
        required = ("TT", "RH", "GHT", "UU", "VV", "PSFC", "T2",
                    surface_rh_name, "U10", "V10")
    missing = [name for name in required if name not in snapshot.fields]
    if missing:
        raise KeyError(f"missing real-data field(s): {missing}")
    # Only fields consumed by float64 WRF-real setup are materialized on the
    # host.  Winds and analyzed hydrometeors remain in their mapped FP32
    # device representation until the FP32 vertical interpolation.  The old
    # FP32 -> host-FP64 -> device-FP32 round trip changed no bits but cost
    # gigabytes of transfer and host residency on large domains.
    host_required = {"TT", "GHT", "PSFC", "T2"}
    if has_specific_humidity:
        host_required.update({"PRES", "SPFH", "Q2"})
    else:
        host_required.update({"RH", surface_rh_name})
    fields = {
        name: (_host(snapshot.fields[name])
               if name in host_required else snapshot.fields[name])
        for name in required
    }
    # Source precedence is explicit: a case may declare an artifact OR use
    # the forcing catalog's validated era5_z_invariant provider.  Silently
    # replacing a declaration would make provenance depend on GRIB inventory.
    if source_orography is not None and "SOURCE_OROGRAPHY" in snapshot.fields:
        raise ValueError(
            "source-orography conflict: both declared source_orography "
            "argument and forcing catalog SOURCE_OROGRAPHY "
            "(era5_z_invariant/SOILGEO) are present; declare exactly one")
    if "SOURCE_OROGRAPHY" in snapshot.fields:
        source_orography = _host(snapshot.fields["SOURCE_OROGRAPHY"])
    mark_timing("validate_and_materialize_host_fields")
    nsource = snapshot.levels_hpa.size
    mass_shape = (nsource, cfg.ny, cfg.nx)
    mass_names = ["TT", "GHT"]
    mass_names += (["PRES", "SPFH"] if has_specific_humidity else ["RH"])
    if (cfg.mp_physics in HRRR_ANALYZED_HYDROMETEOR_MP_PHYSICS
            and has_specific_humidity):
        mass_names += list(HRRR_ANALYZED_HYDROMETEORS)
    if any(fields[name].shape != mass_shape for name in mass_names):
        raise ValueError(
            f"mass-field shapes do not match levels and mass grid: {mass_names}")
    if fields["UU"].shape != (nsource, cfg.ny, cfg.nx + 1):
        raise ValueError("UU does not have WRF u staggering")
    if fields["VV"].shape != (nsource, cfg.ny + 1, cfg.nx):
        raise ValueError("VV does not have WRF v staggering")
    if fields["U10"].shape != (cfg.ny, cfg.nx + 1):
        raise ValueError("U10 does not have WRF u staggering")
    if fields["V10"].shape != (cfg.ny + 1, cfg.nx):
        raise ValueError("V10 does not have WRF v staggering")

    if has_specific_humidity:
        pressure = fields["PRES"]
        if (not np.isfinite(pressure).all() or np.any(pressure <= 0.0)):
            raise ValueError("PRES must be finite and positive")
        surface_specific = _wrf_flag_sh_surface_specific_humidity(
            fields["Q2"], fields["SPFH"], pressure,
            force_fallback=flag_sh_surface_fallback)
        # The WPS undershoot envelope is ADMITTED here, exactly as WRF
        # admits it, and it stays admitted through sfcprs2, integ_moist
        # and the vertical-interpolation pseudo-level, because those are
        # the uses real.exe is DEFINED for and it hands them the raw
        # value.  Only the value this function PUBLISHES as Q2 is floored,
        # and that happens once at the result, below.
        surface_qv = _specific_humidity_to_mixing_ratio(
            surface_specific, allow_wps_undershoot=True,
            column_workers=column_workers)
        source_qv = _specific_humidity_to_mixing_ratio(
            fields["SPFH"], allow_wps_undershoot=not use_sh_qv,
            column_workers=column_workers)
    else:
        pressure = np.broadcast_to(
            snapshot.levels_hpa[:, None, None] * 100.0, mass_shape).copy()
        if surface_rh_name == "D2":
            surface_qv = _saturation_mixing_ratio(
                fields["D2"], fields["PSFC"], 100.0,
                column_workers=column_workers)
        else:
            surface_qv = _saturation_mixing_ratio(
                fields["T2"], fields["PSFC"], fields["RH2"],
                column_workers=column_workers)
        source_qv = _cap_stratospheric_qv(
            _saturation_mixing_ratio(
                fields["TT"], pressure, fields["RH"],
                column_workers=column_workers),
            pressure, column_workers=column_workers)
    if source_orography is None:
        raise ValueError(
            "source_orography is required when sfcp_to_sfcp=True")
    surface_pressure = surface_pressure_from_surface(
        fields["PSFC"], source_orography, terrain, fields["T2"], surface_qv)
    # WRF integrates moisture on the ORIGINAL met surface (integ_moist is
    # called with p_gc whose level 1 is the met PSFC on SOILHGT,
    # module_initialize_real.F:1457/7022); only p_dts (:1482) pairs the
    # sfcprs2-adjusted psfc with the resulting intq for the dry mass.
    source_pd, intq, order = _integrate_moisture(
        source_qv, pressure, fields["TT"], fields["GHT"], fields["PSFC"],
        fields["T2"], surface_qv,
        _host(source_orography) if source_orography is not None else terrain,
        column_workers=column_workers)
    pressure = _ordered_levels(pressure, order)
    source_temperature = _ordered_levels(fields["TT"], order)
    source_qv = _ordered_levels(source_qv, order)
    if use_sh_qv:
        source_rh = None
    elif has_specific_humidity:
        # module_initialize_real.F:1138-1167: FLAG_SH first converts the
        # horizontally mapped SPECHUMD to qv_gc, then overwrites rh_gc at the
        # same target points.  Deriving RH on the HRRR source grid and mapping
        # it separately is not equivalent because both transforms are
        # nonlinear.  Negative WPS SPFH undershoots produce negative RH here;
        # rh_to_mxrat1 clips that RH to zero only after vertical interpolation.
        source_rh = _mixing_ratio_to_relative_humidity(
            source_temperature, pressure, source_qv,
            allow_wps_undershoot=True,
            column_workers=column_workers)
    else:
        source_rh = _ordered_levels(fields["RH"], order)
    # WRF's vert-interp source column carries the surface pseudo-level at
    # the met-source dry surface pressure pd_gc(:,1,:) = p_gc(:,1,:) - intq
    # (integ_moist:7130), i.e. the ORIGINAL met PSFC minus the vapor column.
    surface_pd = fields["PSFC"] - intq
    if use_sh_qv:
        surface_rh = None
    elif has_specific_humidity:
        surface_rh = _mixing_ratio_to_relative_humidity(
            fields["T2"], fields["PSFC"], surface_qv,
            allow_wps_undershoot=True,
            column_workers=column_workers)
    else:
        surface_rh = (
            _surface_relative_humidity(fields["D2"], fields["T2"])
            if surface_rh_name == "D2" else fields["RH2"].copy())
    dry_mass = surface_pressure - intq - float(p_top)
    if np.any(dry_mass <= 0.0):
        raise ValueError("non-positive dry column mass")
    dry_pressure = (coord.c3h[:, None, None] * dry_mass[None]
                    + coord.c4h[:, None, None] + float(p_top))
    dry_pressure_full = (coord.c3f[:, None, None] * dry_mass[None]
                         + coord.c4f[:, None, None] + float(p_top))
    if (not np.all(np.diff(dry_pressure, axis=0) < 0.0)
            or not np.all(np.diff(dry_pressure_full, axis=0) < 0.0)):
        raise ValueError(hybrid_column_ordering_refusal(
            coord, float(p_top), dry_mass + float(p_top),
            quantity="target dry pressure")
            or "target dry pressure is not monotonic")
    mark_timing("source_moisture_and_dry_mass")

    # Backend-selected FP32 vertical interpolation, after the float64
    # pressure/mass setup.
    # WRF (interp_theta=F defaults) interpolates TEMPERATURE in LOG(p) with
    # the t_extrap_type=2 below-ground branch (module_initialize_real.F:
    # 1784-1802), full pressure linearly in p through the same machinery
    # (:1805-1820, var type 'T' so it shares the temperature extrapolation
    # branch), RH in the default LOG(p) with constant extrapolation
    # (:1736-1748), and converts T -> theta only afterwards with the
    # interpolated pressure (t_to_theta, :1862-1867).
    preprocess = resolve_preprocess_backend(
        preprocess_backend, workers=preprocess_workers,
        cpu_bridge=cpu_bridge)
    backend_xp = preprocess.array_module
    if not isinstance(state_backend, str):
        raise TypeError("state_backend must be 'cuda', 'cpu', or 'preprocess'")
    normalized_state_backend = state_backend.strip().lower()
    if normalized_state_backend == "preprocess":
        state_xp = backend_xp
    elif normalized_state_backend == "cpu":
        state_xp = np
    elif normalized_state_backend == "cuda":
        import cupy as cp
        state_xp = cp
    else:
        raise ValueError(
            "state_backend must be 'cuda', 'cpu', or 'preprocess'")
    order_backend = backend_xp.asarray(order, dtype=backend_xp.int32)

    def backend_ordered_levels(value):
        return backend_xp.take(
            preprocess.float32(value), order_backend, axis=0)

    mass_source_pd_f32 = preprocess.float32(source_pd)
    mass_surface_pd_f32 = preprocess.float32(surface_pd)
    mass_target_pd_f32 = preprocess.float32(dry_pressure)
    mass_vertical_plan = preprocess.prepare_wrf_vertical(
        mass_source_pd_f32, mass_surface_pd_f32, mass_target_pd_f32)
    temperature = mass_vertical_plan.apply(
        preprocess.float32(source_temperature),
        preprocess.float32(fields["T2"]),
        interp_in_logp=True, extrap="temperature")
    if use_sh_qv:
        qv = mass_vertical_plan.apply(
            preprocess.float32(source_qv),
            preprocess.float32(surface_qv),
            interp_in_logp=True, extrap="constant")
        rh = None
    else:
        rh = mass_vertical_plan.apply(
            preprocess.float32(source_rh),
            preprocess.float32(surface_rh),
            interp_in_logp=True, extrap="constant")
        qv = None
    total_pressure = mass_vertical_plan.apply(
        preprocess.float32(pressure),
        preprocess.float32(fields["PSFC"]),
        interp_in_logp=False, extrap="temperature")
    temperature_h = _host(temperature).astype(np.float64)
    rh_h = (None if rh is None else
            _host(rh).astype(np.float64))
    total_pressure_h = np.maximum(
        _host(total_pressure).astype(np.float64), dry_pressure)
    theta_h = _potential_temperature_from_temperature(
        temperature_h, total_pressure_h,
        column_workers=column_workers)
    if use_sh_qv:
        qv_h = _host(qv).astype(np.float64)
        if not np.isfinite(qv_h).all() or np.any(qv_h < 0.0):
            raise ValueError("interpolated specific-humidity qv is invalid")
    else:
        qv_h = _cap_stratospheric_qv(
            _saturation_mixing_ratio(
                temperature_h, total_pressure_h, rh_h,
                column_workers=column_workers),
            total_pressure_h, column_workers=column_workers)
    mark_timing("thermodynamic_vertical_interpolation")

    base = _make_real_base(coord, terrain, float(p_top), cfg.base_temp,
                           hypsometric_opt=cfg.hypsometric_opt,
                           column_workers=column_workers)
    if use_sh_qv:
        # WRF use_sh_qv retains the directly interpolated dry-air mixing
        # ratio while diagnosing the final moist-hydrostatic pressure.
        total_pressure_h = _rebalance_moist_pressure(
            total_pressure_h, qv_h, dry_mass, base, coord,
            column_workers=column_workers)
    else:
        # WRF diagnoses qv from interpolated RH, then recomputes a
        # hydrostatic pressure and diagnoses qv once more from that pressure.
        # Two passes make the q/pressure coupling converge below FP32 setup
        # precision.
        for _ in range(2):
            total_pressure_h = _rebalance_moist_pressure(
                total_pressure_h, qv_h, dry_mass, base, coord,
                column_workers=column_workers)
            temperature_h = _temperature_from_potential_temperature(
                theta_h, total_pressure_h,
                column_workers=column_workers)
            qv_h = _cap_stratospheric_qv(
                _saturation_mixing_ratio(
                    temperature_h, total_pressure_h, rh_h,
                    column_workers=column_workers),
                total_pressure_h, column_workers=column_workers)
        total_pressure_h = _rebalance_moist_pressure(
            total_pressure_h, qv_h, dry_mass, base, coord,
            column_workers=column_workers)
        # WRF invokes rh_to_mxrat1 again against its final hydrostatic
        # pressure; apply the strict pressure-side cap once more in case a
        # target level crossed 10 kPa during the last rebalance.
        qv_h = _cap_stratospheric_qv(
            qv_h, total_pressure_h, column_workers=column_workers)
    mark_timing("base_state_and_moist_rebalance")

    # U/V columns include the 10 m surface pseudo-level with pd averaged to
    # the staggered points exactly like the interior levels
    # (module_initialize_real.F:2785-2811 with vert_interp's 'U'/'V'
    # pressure averaging at :5664-5713; extrap_type=2 constant).
    source_pd_u = _pressure_at_u(source_pd)
    source_pd_v = _pressure_at_v(source_pd)
    surface_pd_u = _pressure_at_u(surface_pd[None])[0]
    surface_pd_v = _pressure_at_v(surface_pd[None])[0]
    target_pd_u = _pressure_at_u(dry_pressure)
    target_pd_v = _pressure_at_v(dry_pressure)
    u_plan = preprocess.prepare_wrf_vertical(
        preprocess.float32(source_pd_u),
        preprocess.float32(surface_pd_u),
        preprocess.float32(target_pd_u))
    v_plan = preprocess.prepare_wrf_vertical(
        preprocess.float32(source_pd_v),
        preprocess.float32(surface_pd_v),
        preprocess.float32(target_pd_v))
    u = u_plan.apply(
        backend_ordered_levels(fields["UU"]),
        preprocess.float32(fields["U10"]),
        interp_in_logp=True, extrap="constant")
    v = v_plan.apply(
        backend_ordered_levels(fields["VV"]),
        preprocess.float32(fields["V10"]),
        interp_in_logp=True, extrap="constant")

    hydrometeors = {}
    hydrometeor_initialization: dict[str, object] = {}
    if (has_specific_humidity
            and cfg.mp_physics in HRRR_ANALYZED_HYDROMETEOR_MP_PHYSICS):
        # WRF's hydrometeor vert_interp calls use var_type='Q' with
        # linear_interp and no dedicated surface analysis.  The metgrid
        # surface pseudo-level is therefore zero; setting vboundb above the
        # target column keeps the shared kernel linear at every eta level.
        zero_surface = backend_xp.zeros(
            (cfg.ny, cfg.nx), dtype=backend_xp.float32)

        def replay_hydrometeor_support(ordered_mask):
            return mass_vertical_plan.apply(
                backend_xp.asarray(ordered_mask, dtype=backend_xp.float32),
                zero_surface,
                interp_in_logp=True, extrap="constant",
                vboundb=cfg.nz + 1, values_are_finite=True)

        invalid_source = []
        for name in HRRR_ANALYZED_HYDROMETEORS:
            source_value = preprocess.float32(fields[name])
            if (not bool(backend_xp.isfinite(source_value).all())
                    or bool((source_value < 0.0).any())):
                invalid_source.append(name)
        if invalid_source:
            raise ValueError(
                "mapped HRRR hydrometeor forcing is non-finite or negative: "
                f"{invalid_source}")
        source_fingerprints = {
            name: array_correspondence_fingerprint(
                preprocess.float32(fields[name]))
            for name in HRRR_ANALYZED_HYDROMETEORS
        }
        retained_names = (
            ("QC", "QR") if cfg.mp_physics == 1
            else HRRR_ANALYZED_HYDROMETEORS
        )
        for name in retained_names:
            value = mass_vertical_plan.apply(
                backend_ordered_levels(fields[name]),
                zero_surface,
                interp_in_logp=True, extrap="constant",
                vboundb=cfg.nz + 1, values_are_finite=True)
            if (not bool(backend_xp.isfinite(value).all())
                    or bool((value < 0.0).any())):
                raise ValueError(
                    f"interpolated HRRR hydrometeor {name} is invalid")
            hydrometeors[name] = value
        discarded = {}
        if cfg.mp_physics == 1:
            discarded = {
                name: {
                    "source": source_fingerprints[name],
                    **WRF_REAL_KESSLER_FROZEN_POLICY,
                }
                for name in ("QI", "QS", "QG")
            }
        vertical_disposition = build_hrrr_hydrometeor_vertical_disposition(
            {name: fields[name] for name in retained_names},
            order,
            mass_source_pd_f32,
            mass_surface_pd_f32,
            mass_target_pd_f32,
            hydrometeors,
            operator_replay=replay_hydrometeor_support,
        )
        hydrometeor_initialization = {
            "schema": HRRR_HYDROMETEOR_CORRESPONDENCE_SCHEMA_V2,
            "source": "native-hrrr-horizontal-decoder-output",
            "mp_physics": int(cfg.mp_physics),
            "decoded_source_species": source_fingerprints,
            "retained_correspondence": {
                name: name.lower() for name in retained_names
            },
            "discarded_source_species": discarded,
            "vertical_disposition": vertical_disposition,
        }

    # -- Configured initial-state perturbation (theta bubbles) ----------
    # Applied ONCE, here, after the base real-data state is final (the
    # WRF two-pass moist rebalance above) and BEFORE the specific volume
    # and geopotential are formed, so alpha/thp/php below are computed
    # FROM the perturbed theta/qv and the state stays discretely
    # hydrostatic at the analyzed pressure -- the em_quarter_ss
    # convention (bubble at constant column mass, geopotential
    # rebalanced), which hydrostatic_residual then grades unchanged.
    # ``initial_perturbation is None`` is the whole OFF contract: not one
    # instruction of this block runs and the state is byte-identical to a
    # build without the feature.
    perturbation_receipt: dict[str, object] = {}
    if initial_perturbation is not None:
        alpha_unperturbed = _moist_specific_volume(
            theta_h, qv_h, total_pressure_h,
            column_workers=column_workers)
        z_half_agl = _half_level_height_agl(
            base, coord, dry_mass, alpha_unperturbed,
            hypsometric_opt=cfg.hypsometric_opt)
        perturbation_receipt = initial_perturbation.apply(
            theta=theta_h, qv=qv_h, pressure=total_pressure_h,
            z_half_agl=z_half_agl)
        del alpha_unperturbed, z_half_agl
        mark_timing("initial_perturbation")

    alpha = _moist_specific_volume(
        theta_h, qv_h, total_pressure_h,
        column_workers=column_workers)
    mark_timing("wind_hydrometeor_interpolation_and_alpha")

    state_kwargs = {}
    if scratch_arena is not None:
        state_kwargs["scratch_arena"] = scratch_arena
    if dycore_state_workspace is not None:
        state_kwargs["dycore_state_workspace"] = dycore_state_workspace
    state = DomainState(cfg, array_module=state_xp, **state_kwargs)
    mark_timing("domain_state_allocation")
    state.load_base(coord, base)
    mark_timing("base_state_upload")
    state.mup[...] = state_xp.asarray(
        dry_mass - base.mub, dtype=state_xp.float32)
    state.thp[...] = state_xp.asarray(
        theta_h - base.thb, dtype=state_xp.float32)
    mark_timing("mass_theta_upload")
    state.php[...] = state_xp.asarray(
        _fp32_geopotential_split(base, coord, dry_mass, alpha,
                                 hypsometric_opt=cfg.hypsometric_opt,
                                 column_workers=column_workers),
        dtype=state_xp.float32)
    mark_timing("fp32_geopotential_split_and_upload")
    state.qv[...] = state_xp.asarray(qv_h, dtype=state_xp.float32)
    if hydrometeors:
        for source_name, value in hydrometeors.items():
            getattr(state, source_name.lower())[...] = state_xp.asarray(
                value, dtype=state_xp.float32)
        hydrometeor_initialization["initialized_state_species"] = {
            state_name: array_correspondence_fingerprint(
                getattr(state, state_name))
            for state_name in hydrometeor_initialization[
                "retained_correspondence"].values()
        }
        if cfg.mp_physics == 8:
            # HRRR provides the shared five WRF mass species but not classic
            # Thompson's Registry scalar QNICE/QNRAIN fields.  Pin real.exe's
            # source-absent policy explicitly rather than diagnosing a
            # distribution here: both transported number moments begin at
            # exact FP32 zero and Thompson owns their first physical update.
            state.ni[...] = state_xp.float32(0.0)
            state.nr[...] = state_xp.float32(0.0)
    else:
        state.qc[...] = 0.0
        state.qr[...] = 0.0
    aerosol_initialization: dict[str, object] = {}
    if cfg.mp_physics == 28:
        # Aerosol-aware Thompson.  Its Registry package
        # (Registry.EM_COMMON:3036) carries the same six moist mass members
        # as mp=8 -- so every mass decision above is already correct for it
        # -- plus SIX scalar members: qni, qnr, qnc, qnwfa, qnifa, qnbca.
        # No real-data source this module reads supplies any of the six, so
        # this block is the mp=28 statement of real.exe's ``i0``
        # source-absent policy.  It runs on BOTH lanes (native HRRR and
        # pressure-level/RH) because the policy is about the scheme's
        # scalars, not about analyzed condensate.  It splits into two
        # halves that must not be confused.
        #
        # (a) THE NUMBER MOMENTS -- exact FP32 zero, mp=8's reasoning with
        # nc added.  mp=28 promotes cloud droplet number from the constant
        # Nt_c to a prognostic scalar, so nc joins ni and nr as a
        # transported moment the analysis does not carry and the scheme
        # owns from its first step.  These are written explicitly rather
        # than left to the allocator: the value is a policy, and a policy
        # that is only ever an allocation default is one nobody can find.
        # qnbca is NOT a gpuwm state field at all -- the port scopes
        # wif_input_opt=0, which is the only value the refusal above
        # admits, and Registry/registry.new3d_wif:82 allocates qnbca only
        # under wif_input_opt==2.
        state.ni[...] = state_xp.float32(0.0)
        state.nr[...] = state_xp.float32(0.0)
        state.nc[...] = state_xp.float32(0.0)
        # (b) THE AEROSOLS -- deliberately NOT written here, and the
        # deliberateness is the whole point.  WRF's own initializer, with
        # aer_init_opt=0, sets QNWFA and QNIFA to 0.0 and nothing else
        # (dyn_em/module_initialize_real.F:2332-2345).  ``thompson_init``
        # then tests MAXVAL(nwfa) < eps (phys/module_mp_thompson.F:493) and
        # MAXVAL(nifa) < eps (:531) and, finding them empty, installs the
        # synthetic boundary-layer-following CCN profile at :498-515 and
        # the IN profile at :536-551, deriving the surface emission nwfa2d
        # at :509-510.  Exact zero is therefore not "unset": it is the
        # SIGNAL that selects that branch.  Assigning any nonzero
        # placeholder here -- a floor, a background, a climatology guess --
        # would flip WRF's own has_CCN/has_IN test and permanently suppress
        # the profile fill, which is the failure mode that costs 5.6x in
        # droplet number and +74% in domain-total RAINNC while raising no
        # error anywhere.
        #
        # WHICH CODE ACTUALLY FILLS IT: gpuwm.core.physics.
        # initialize_physics, via gpuwm.core.microphysics.
        # microphysics_init -- WRF's own split, where thompson_init is
        # called from phys/module_physics_init.F and not from
        # module_initialize_real.F.  Both production real-data front doors
        # reach it (gpuwm/ingest/hrrr_physics.py::initialize_prepared_physics
        # and ::initialize_hrrr_physics both call initialize_physics on the
        # state this function returns), so a fill here would be the FIRST of
        # two calls, never the only one.
        #
        # MEASURED, so the reason given is the real one: a duplicate call is
        # IDEMPOTENT, not destructive.  thompson_aerosol_init_fill re-runs
        # WRF's own MAXVAL presence tests every time, so a second
        # initialize_physics on the same state reports
        # {'ccn': False, 'in': False} and changes not one bit
        # (tests/test_real_init.py::
        # test_mp28_real_ingest_through_initialize_physics_fills_exactly_once
        # asserts precisely that, on device).  The reason this module still
        # must not do the fill is structural, not a race: WRF puts
        # thompson_init in module_physics_init.F and not in
        # module_initialize_real.F, the domain construction the fill belongs
        # to has not happened yet here, and filling would make the
        # "awaiting_profile_fill" receipt this function publishes false.
        # THE DEFAULT (lane/wif-default).  This used to read
        # ``(aer_init_opt, wif_input_opt) == (1, 1)`` -- an opt-in.  An
        # opt-in flag on a correctness remedy is a workaround, so it is
        # gone: a real-data mp=28 run now RESOLVES WRF's monthly WIF
        # climatology and uses it, and reaches thompson_init's synthetic
        # profile only when there is genuinely no dataset to read.  The two
        # namelist selectors keep WRF's Registry defaults (the
        # prepared-forecast runner compares those rows for exact equality);
        # the decision lives in cfg.mp28_aerosol_source, whose default is
        # "auto".
        #
        # ``climatology`` refuses instead of falling back, because a
        # request that silently degrades is worse than one that fails.
        # ``synthetic`` skips the resolver entirely -- it does not "fail to
        # find" anything, so it is not a fallback and must not be reported
        # as one.
        from gpuwm.ingest.wif_climatology import (
            describe_wif_source, resolve_wif_climatology)
        _source_choice = str(
            getattr(cfg, "mp28_aerosol_source", "auto") or "auto")
        _namelist_demand = (
            (int(cfg.aer_init_opt), int(cfg.wif_input_opt)) == (1, 1))
        if _namelist_demand and _source_choice == "auto":
            # aer_init_opt=1/wif_input_opt=1 is real.exe's own spelling of
            # "use the climatology"; honour it as the strict form.
            _source_choice = "climatology"
        if _source_choice == "synthetic":
            wif_resolution = None
        else:
            wif_resolution = resolve_wif_climatology(
                cfg.wif_climatology_path or None,
                explicit_required=(_source_choice == "climatology"))
        wif_climatology_selected = (
            wif_resolution is not None and wif_resolution.resolved)
        wif_receipt = None
        if wif_climatology_selected:
            # (b') THE PORTED CLIMATOLOGY -- real.exe's aer_init_opt=1
            # path (dyn_em/module_initialize_real.F:2357-2536 3-D,
            # :4530-4547 2-D), fed from the SAME global monthly dataset
            # WRF routes through metgrid's constants_name.  This is the
            # one aerosol source that fills the fields HERE rather than
            # leaving thompson_init's synthetic profile to do it: the
            # nonzero nwfa/nifa this writes is exactly the signal WRF's
            # own MAXVAL presence tests (module_mp_thompson.F:493/:531)
            # read to SKIP the synthetic fill, so microphysics_init needs
            # no new teaching -- WRF's mechanism already carries it.  The
            # vertical target is ``dry_pressure``: the ``grid%pb`` the
            # WIF vert_interp call receives is, at that point in
            # init_domain_rk, the p_dry(mu0, znw, p_top) scratch
            # (:1625-1629, :1701), not the final base state.
            from gpuwm.ingest.wif_climatology import (
                load_wif_climatology, wif_fields_for_grid)
            # BOTH INPUTS ARE DERIVED, and neither is guessed.  Making a
            # caller hand these in was the whole reason the capability had
            # no front door: every runner already holds both, so asking
            # for them again meant a configuration that selects the
            # ingest still could not run it.
            #
            # The valid date is ``snapshot.valid_time`` -- the case's own
            # valid time, the same one every runner would have passed and
            # the one metgrid stamps on this forcing frame.  There is no
            # second candidate; a date that disagreed with the snapshot
            # would be interpolating the climatology to a month this
            # state is not.
            #
            # The grid latitudes/longitudes are the model's mass-point
            # geodesy, taken from whatever the caller already carries: a
            # projection Grid (``latlon_mass()``), the state's own
            # ``latitude_deg``/``longitude_deg`` mirror, a geogrid mapping
            # (``XLAT_M``/``XLONG_M``), or a plain ``(lat2d, lon2d)``
            # pair.  This is METADATA, not source data -- which is the
            # arbitrary-acceptance argument the module docstring already
            # makes: the derivation is identical for HRRR, GFS, ERA5,
            # 20CRv3, RUC and any mapped source, because none of them is
            # consulted.
            if wif_valid_date is None:
                wif_valid_date = getattr(snapshot, "valid_time", None)
            if wif_grid_latlon is None:
                wif_grid_latlon = _wif_grid_latlon_from(grid, state)
            if wif_grid_latlon is None or wif_valid_date is None:
                # The dataset resolved but this CALLER still cannot use
                # it.  The derivation above (lane/static-dataset-door)
                # takes the valid date from snapshot.valid_time and the
                # mass-point lat/lon from whichever carrier the caller
                # holds, so reaching here means the caller supplied NONE
                # of them -- not that a front door forgot to pass grid=.
                #
                # WHICH BEHAVIOUR is decided by who asked, not by what is
                # missing (lane/wif-default).  An EXPLICIT climatology
                # request refuses: honouring it is impossible, and
                # silently substituting a different initial condition for
                # a named one is the failure that lane exists to remove.
                # The DEFAULT falls back, because a call site that carries
                # no geodesy at all is a property of the call site, not a
                # configuration error the user made, and a default that
                # hard-fails there is a default nobody can ship.  Either
                # way it is NAMED, and it names WHICH input could not be
                # derived rather than only that one could not.
                missing = []
                if wif_grid_latlon is None:
                    missing.append(
                        "the model mass-point latitudes/longitudes (pass "
                        "grid=<projection Grid>, grid={'XLAT_M':..., "
                        "'XLONG_M':...} or wif_grid_latlon=(lat2d, lon2d))")
                if wif_valid_date is None:
                    missing.append(
                        "the case valid time (the snapshot carries it as "
                        ".valid_time; pass wif_valid_date='YYYY-MM-DD...' "
                        "when it does not)")
                unusable = (
                    "the WIF aerosol climatology resolved at "
                    f"{wif_resolution.path} but this initialization could "
                    "not derive " + " and ".join(missing)
                    + ", which a GLOBAL monthly dataset needs and which "
                    "cannot be guessed: a wrong grid or a wrong month is a "
                    "silently different aerosol field, not an error")
                if _source_choice == "climatology":
                    raise ValueError(
                        "mp28_aerosol_source='climatology' cannot be "
                        "honoured: " + unusable)
                from gpuwm.ingest.wif_climatology import WifSourceResolution
                wif_resolution = WifSourceResolution(
                    None, "resolved-but-caller-supplied-no-grid",
                    wif_resolution.candidates, unusable)
                wif_climatology_selected = False
        if wif_climatology_selected:
            wif_lat2d, wif_lon2d = wif_grid_latlon
            # ONE RESOLVER, not two.  lane/static-dataset-door reached the
            # dataset a second time here through
            # wif_dataset.resolve_wif_climatology_path.  It cannot stay:
            # resolve_wif_climatology (lane/wif-default) has ALREADY chosen
            # the file above -- that choice is what made
            # wif_climatology_selected true -- and a second resolver over
            # the same config field with a different precedence is how a
            # run reads one dataset and reports another.  The staged root
            # that call existed to reach (`gpuwm fetch-tables --wif` ->
            # ~/.gpuwm/wif) is now a rung INSIDE resolve_wif_climatology,
            # so nothing was lost by deleting it.
            wif_fields, wif_receipt = wif_fields_for_grid(
                load_wif_climatology(wif_resolution.path),
                _host(wif_lat2d), _host(wif_lon2d), str(wif_valid_date),
                _host_float32(dry_pressure), _host_float32(base.phb))
            for name in ("nwfa", "nifa", "nwfa2d", "nifa2d"):
                target_field = getattr(state, name)
                target_field[...] = state_xp.asarray(
                    wif_fields[name], dtype=state_xp.float32)
        else:
            for name in ("nwfa", "nifa", "nwfa2d", "nifa2d"):
                value = getattr(state, name)
                if bool((value != 0).any()):
                    raise ValueError(
                        f"mp_physics=28 real initialization left "
                        f"state.{name} "
                        "nonzero; exact zero is WRF's aer_init_opt=0 value "
                        "(dyn_em/module_initialize_real.F:2332-2345) AND "
                        "the signal thompson_init's MAXVAL test reads "
                        "(phys/module_mp_thompson.F:493/:531) to install "
                        "the synthetic profile. The one ingest that MAY "
                        "populate them is the wif-climatology branch "
                        "above, which this run did not select.")
        aerosol_initialization = _mp28_aerosol_source_policy(cfg, state)
        aerosol_initialization["mp28_aerosol_source"] = _source_choice
        if wif_receipt is not None:
            # The fields are ALREADY populated: thompson_init's MAXVAL
            # tests will find them nonzero and skip the synthetic fill,
            # so nothing is awaited.  The stage receipt binds which
            # dataset, which weights and which operators produced them.
            from gpuwm.config import MP28_AEROSOL_SOURCE_DEFAULT
            aerosol_initialization["awaiting_profile_fill"] = False
            aerosol_initialization["wif_climatology"] = wif_receipt
            aerosol_initialization["aerosol_source"] = "wif-climatology"
            aerosol_initialization["aerosol_source_statement"] = (
                MP28_AEROSOL_SOURCE_DEFAULT)
            aerosol_initialization["dataset"] = describe_wif_source(
                wif_resolution)
            aerosol_initialization["policy"] = (
                "wif-climatology-monthly-interp-then-vert-interp")
        else:
            # THE FALLBACK, SAID OUT LOUD.  This is the branch that changes
            # what the forecast IS, so it is the branch that must be
            # impossible to miss in a receipt: a named source, the WRF
            # sentence explaining what a synthetic aerosol profile is, and
            # -- when the fallback was reached rather than requested -- the
            # concrete search that came up empty.
            from gpuwm.config import MP28_AEROSOL_SYNTHETIC_FALLBACK
            aerosol_initialization["aerosol_source"] = (
                "thompson_init-synthetic-profile")
            aerosol_initialization["aerosol_source_statement"] = (
                MP28_AEROSOL_SYNTHETIC_FALLBACK)
            aerosol_initialization["synthetic_fallback_in_use"] = True
            aerosol_initialization["synthetic_fallback_requested"] = (
                _source_choice == "synthetic")
            if wif_resolution is not None:
                aerosol_initialization["dataset"] = describe_wif_source(
                    wif_resolution)
            _warn_synthetic_aerosol_fallback(
                _source_choice, wif_resolution)
    state.u[...] = state_xp.asarray(u, dtype=state_xp.float32)
    state.v[...] = state_xp.asarray(v, dtype=state_xp.float32)
    state.w[...] = 0.0
    total_phi = _host(state.phb + state.php)
    mark_timing("remaining_state_upload_and_geopotential_readback")
    # The ONE place a surface mixing ratio is published rather than
    # consumed: everything above has had WRF's raw value, and this is what
    # becomes Q2 in the physics driver, in wrfinput, and in the prepared
    # cache.  A no-op on the RH lane, whose _saturation_mixing_ratio
    # already returned nothing below this floor.
    published_surface_qv, surface_qv_floor = (
        _floor_flag_sh_surface_mixing_ratio(surface_qv, fields["PSFC"]))
    if timing_report is not None:
        timing_report["total_seconds"] = perf_counter() - timing_start
    return RealInitResult(
        state=state, coord=coord, base=base,
        surface_pressure=surface_pressure, surface_qv=published_surface_qv,
        dry_mass=dry_mass, dry_pressure=dry_pressure,
        total_pressure=total_pressure_h, total_geopotential=total_phi,
        total_specific_volume=alpha,
        integrated_moisture_pressure=intq,
        hypsometric_opt=cfg.hypsometric_opt,
        hydrometeor_initialization=hydrometeor_initialization,
        aerosol_initialization=aerosol_initialization,
        surface_moisture_floor=surface_qv_floor,
        initial_perturbation=perturbation_receipt)


def source_orography_from_catalog(catalog, grid, *,
                                  provider="era5_z_invariant",
                                  valid_time=None) -> np.ndarray:
    """Public real-ingest resolver for catalog-declared source terrain."""
    return _source_orography_from_catalog(
        catalog, grid, provider=provider, valid_time=valid_time)


def hydrostatic_residual(result: RealInitResult) -> np.ndarray:
    """Maximum discrete moist-hydrostatic residual of the live FP32 state.

    Pressure remains the setup-time thermodynamic target, while geopotential,
    dry mass, potential temperature, vapor, coefficients, and arithmetic inputs
    are read back from the initialized :class:`DomainState`.  This makes the
    gate sensitive to the actual ``state.php`` quantization loaded on device.
    """
    state = result.state
    total_phi = _host(state.phb + state.php)
    dry_mass = _host(state.mub2d + state.mup)
    theta = _host(state.thb + state.thp)
    qv = _host(state.qv)
    pressure = np.asarray(result.total_pressure, dtype=np.float64)
    alpha = (c.RD * theta * (1.0 + c.RVOVRD * qv)
             * (pressure / c.P0) ** c.RCP / pressure)
    if result.hypsometric_opt == 2:
        # Log-pressure hydrostatic operator on the total dry mass, the
        # opt-2 counterpart of the discrete d(phi)/d(eta) relation
        # (module_initialize_real.F:3970-3981 / :4002-4010).
        c3h = _host(state.c3h)[:, None, None]
        c4h = _host(state.c4h)[:, None, None]
        c3f = _host(state.c3f)[:, None, None]
        c4f = _host(state.c4f)[:, None, None]
        p_top = float(state.p_top)
        pfu = c3f[1:] * dry_mass[None] + c4f[1:] + p_top
        pfd = c3f[:-1] * dry_mass[None] + c4f[:-1] + p_top
        phm = c3h * dry_mass[None] + c4h + p_top
        residual = (np.diff(total_phi, axis=0)
                    - alpha * phm * np.log(pfd / pfu))
    else:
        c1h = _host(state.c1h)[:, None, None]
        c2h = _host(state.c2h)[:, None, None]
        dnw = _host(state.dnw)[:, None, None]
        increment = c1h * dry_mass[None] + c2h
        residual = np.diff(total_phi, axis=0) + dnw * increment * alpha
    return np.max(np.abs(residual), axis=0)


__all__ = ["HRRR_ANALYZED_HYDROMETEORS",
           "HRRR_ANALYZED_HYDROMETEOR_MP_PHYSICS", "RealInitResult",
           "WRF_REAL_MP28_AEROSOL_SOURCE_POLICY", "hydrostatic_residual",
           "initialize_real", "source_orography_from_catalog",
           "surface_pressure_from_surface"]
