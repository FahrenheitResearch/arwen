"""Shared prepared-state/restart serialization identity primitives.

This deliberately small module is part of the standalone RW-WPS package.
Prepared-cache export and the full forecast restart reader must serialize the
same state inventory and compute the same setup fingerprint without making the
preprocessor depend on the full forecast I/O implementation.
"""

from __future__ import annotations

import hashlib

import numpy as np


STATE_SERIALIZED_ATTRS = (
    "u", "v", "w", "thp", "php", "mup",
    "p", "al", "alt",
    "qv", "qc", "qr", "h_diabatic",
    # WRF's prognostic SGS TKE (Registry.EM_COMMON:312 ``state real tke ikj
    # dyn_em 2 - r``): the trailing ``r`` puts it in the restart stream, and
    # nothing reconstructs it -- a resumed km_opt=2 run that re-zeroed the
    # carrier would cold-start the closure on a fully developed field.
    # Absent (None) under every other km_opt, so non-LES inventories are
    # unchanged.
    "tke",
    "qi", "qs", "qg", "nc", "nr", "ni", "ns", "ng",
    "qh", "qndrop", "qnr", "qni", "qns", "qng", "qnh", "qnn",
    "qvolg", "qvolh",
    "effc", "effr", "effi", "effs",
    # mp_physics=28 (Thompson aerosol-aware).  ``nc`` is already listed
    # above (Morrison allocates it).  ``nwfa``/``nifa`` are prognostic
    # 3-D aerosol number tracers; ``nwfa2d``/``nifa2d`` are the (ny, nx)
    # surface emission tendencies, which are cross-step CONSTANTS -- they
    # are INTENT(IN) to WRF's mp_gt_driver (module_mp_thompson.F:1098) and
    # nothing in the forecast writes them, but they are derived once from
    # thompson_init's synthetic profile (:510) and a restart that dropped
    # them would silently resume with zero surface aerosol emission.
    # WRF agrees: Registry.EM_COMMON:492-493 declares QNWFA2D/QNIFA2D with
    # the IO string ``i01{17}rhdu``, whose ``r`` puts them in the restart
    # stream.  Serializing them is transcription, not a gpuwm invention.
    "nwfa", "nifa", "nwfa2d", "nifa2d",
    # mp_physics=50 (P3 one-category).  ``qir``/``qib`` are the rime MASS
    # and rime VOLUME that WRF declares in the same ``scalar`` package as
    # qni/qnr (Registry.EM_COMMON:3038) and carries in the restart stream;
    # they are transported prognostics in gpuwm too
    # (gpuwm/core/moist.py::P3_SPECIES), and a resume that dropped them
    # would restore rime-free ice -- rho_rime = qirim/birim picks the
    # lookup table's rime-density index, so the resumed run would use a
    # different ice fall speed and a different collection rate than the
    # run it claims to continue.
    #
    # ``th_old``/``qv_old`` are P3's cross-step supersaturation carriers
    # (Registry.EM_COMMON:1598-1599, both with the restart ``r`` in their
    # IO string).  p3_main writes them at the end of every call
    # (module_mp_p3.F:5018-5021) and reads them at the top of the next
    # (:2320-2337).  Re-zeroing them on resume would replay the first-step
    # transient -- WRF's own max(t_old,1.) guard, and the 0/0 sup/supi it
    # produces -- once more in the middle of a trajectory.  Serializing
    # them is transcription of WRF's restart stream, not a gpuwm choice.
    #
    # All four are absent (None) on every other scheme's state, and both
    # the writer and the reader skip on ``is None``, so no existing
    # restart inventory moves.
    "qir", "qib", "th_old", "qv_old",
    # SASE prognostic subgrid turbulence energy.  Like the optional
    # microphysics moments above, the attribute is ABSENT on a state that
    # did not select the closure, and the manifest walk skips what is not
    # there -- so adding it moves no existing restart.
    "e_sgs",
)

# STATE_SETUP_ARRAYS is deliberately NOT extended with nwfa2d/nifa2d, even
# though they are per-domain constants and read like setup: ``setup_fingerprint``
# below does an UNCONDITIONAL ``getattr(state, name)`` over this tuple, so a
# name only some configurations allocate would raise AttributeError on every
# non-mp28 run.  They are covered as serialized state above instead, where
# both the writer and the reader skip on ``is None``.
STATE_SETUP_ARRAYS = (
    "thb", "pb", "alb", "phb", "mub2d", "ht",
    "c1h", "c2h", "c1f", "c2f", "c3h", "c4h", "c3f", "c4f",
    "msft", "msfu", "msfv", "f", "e", "sina", "cosa",
    "dnw", "rdnw", "dn", "rdn", "fnp", "fnm", "znu", "znw",
)

STATE_SETUP_SCALARS = (
    "mub", "p_top", "cf1", "cf2", "cf3", "cfn", "cfn1",
    "has_msf", "rotational",
)


def _host(value) -> np.ndarray:
    if hasattr(value, "get"):
        value = value.get()
    return np.asarray(value)


def _digest_array(digest, name: str, value) -> None:
    host = _host(value)
    digest.update(name.encode())
    digest.update(str(host.shape).encode())
    digest.update(str(host.dtype).encode())
    digest.update(host.tobytes())


def _update_setup_core(digest, state, *, error_type: type[Exception]) -> bool:
    """Hash setup that cannot grow; return whether this is a nest.

    The byte stream is deliberately the prefix of :func:`setup_fingerprint`'s
    long-standing stream.  Keeping that stream byte-for-byte stable preserves
    every existing exact-restart identity while also exposing the immutable
    half for the explicit sealed-forcing extension contract.
    """

    for name in STATE_SETUP_ARRAYS:
        _digest_array(digest, name, getattr(state, name))
    for name in STATE_SETUP_SCALARS:
        value = getattr(state, name)
        if value is not None and not isinstance(value, bool):
            value = float(value)
        digest.update(f"{name}={value!r};".encode())
    nest_class = getattr(state, "_nest_restart_classification", None)
    if nest_class is not None:
        if nest_class != "REBUILT":
            raise error_type(
                f"unknown nest restart classification {nest_class!r}")
        digest.update(b"nest_tables=REBUILT;")
        return True
    return False


def _update_lateral_fingerprint(digest, state) -> None:
    """Append the exact historical LBC portion of the setup digest."""

    boundaries = getattr(state, "lateral_boundaries", None)
    if boundaries is None:
        digest.update(b"lateral_boundaries=None;")
        return
    digest.update(
        f"lbc:width={boundaries.spec_bdy_width};"
        f"spec={boundaries.spec_zone};relax={boundaries.relax_zone};"
        f"intervals={len(boundaries.intervals)};".encode())
    for interval in boundaries.intervals:
        digest.update(
            f"[{interval.start_seconds!r},"
            f"{interval.end_seconds!r}]".encode())
        for name in sorted(interval.fields):
            boundary = interval.fields[name]
            for side_name in ("west", "east", "south", "north"):
                side = getattr(boundary, side_name)
                _digest_array(
                    digest, f"{name}/{side_name}/value", side.value)
                _digest_array(
                    digest, f"{name}/{side_name}/tendency", side.tendency)


def setup_core_fingerprint(
        state, *, error_type: type[Exception] = ValueError) -> str:
    """Hash immutable setup state without a root's growable LBC inventory."""

    digest = hashlib.sha256()
    _update_setup_core(digest, state, error_type=error_type)
    return digest.hexdigest()


LATERAL_BOUNDARY_PREFIX_SCHEMA = "gpuwm-lateral-boundary-prefix-v2"


def lateral_boundary_prefix_identity(
        state, *, error_type: type[Exception] = ValueError):
    """Return interval-level hashes for an append-only forcing proof.

    ``None`` means that this state has no external forcing inventory (a
    prepared child, or a non-specified root).  Each interval digest includes
    its exact bounds, field names, shapes, dtypes, and every side's value and
    tendency bytes.  The compact list is safe to put in a checkpoint header;
    it proves a later preparation retained the old inventory byte-for-byte
    without serializing those forcing tables into the checkpoint itself.
    """

    nest_class = getattr(state, "_nest_restart_classification", None)
    if nest_class is not None:
        if nest_class != "REBUILT":
            raise error_type(
                f"unknown nest restart classification {nest_class!r}")
        return None
    boundaries = getattr(state, "lateral_boundaries", None)
    if boundaries is None:
        return None
    intervals = []
    for interval in boundaries.intervals:
        digest = hashlib.sha256()
        start_frame = hashlib.sha256()
        end_frame = hashlib.sha256()
        digest.update(
            f"[{interval.start_seconds!r},"
            f"{interval.end_seconds!r}]".encode())
        duration = float(interval.end_seconds - interval.start_seconds)
        fields = []
        for name in sorted(interval.fields):
            fields.append(name)
            boundary = interval.fields[name]
            for side_name in ("west", "east", "south", "north"):
                side = getattr(boundary, side_name)
                _digest_array(
                    digest, f"{name}/{side_name}/value", side.value)
                _digest_array(
                    digest, f"{name}/{side_name}/tendency", side.tendency)
                # The forcing consumer rounds host tables to FP32 before
                # use.  Seal both endpoint frames in that exact numerical
                # representation so an appended interval cannot replace the
                # shared restart-boundary frame while preserving the older
                # interval row.
                start = np.asarray(_host(side.value), dtype=np.float32)
                end = np.asarray(
                    _host(side.value) + _host(side.tendency) * duration,
                    dtype=np.float32)
                _digest_array(
                    start_frame, f"{name}/{side_name}/value", start)
                _digest_array(
                    end_frame, f"{name}/{side_name}/value", end)
        intervals.append({
            "start_seconds": interval.start_seconds,
            "end_seconds": interval.end_seconds,
            "fields": fields,
            "sha256": digest.hexdigest(),
            "start_frame_sha256": start_frame.hexdigest(),
            "end_frame_sha256": end_frame.hexdigest(),
        })
    return {
        "schema": LATERAL_BOUNDARY_PREFIX_SCHEMA,
        "spec_bdy_width": boundaries.spec_bdy_width,
        "spec_zone": boundaries.spec_zone,
        "relax_zone": boundaries.relax_zone,
        "intervals": intervals,
    }


def setup_fingerprint(state, *, error_type: type[Exception] = ValueError) -> str:
    """Hash deterministic setup state and attached LBC forcing tables."""

    digest = hashlib.sha256()
    nested = _update_setup_core(digest, state, error_type=error_type)
    if not nested:
        _update_lateral_fingerprint(digest, state)
    return digest.hexdigest()


__all__ = [
    "LATERAL_BOUNDARY_PREFIX_SCHEMA",
    "STATE_SERIALIZED_ATTRS",
    "STATE_SETUP_ARRAYS",
    "STATE_SETUP_SCALARS",
    "lateral_boundary_prefix_identity",
    "setup_core_fingerprint",
    "setup_fingerprint",
]
