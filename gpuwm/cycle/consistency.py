"""Is this state in agreement with its own carried diagnostics?

Every gate that already stands on the restart path is an IDENTITY gate:
it asks whether the state that came back equals the state that went in.
A DA increment must break identity by construction -- that is what an
analysis IS -- so not one of those gates can see the failure this module
exists for: prognostics rewritten by an increment, diagnostics carried
across the boundary unchanged, and a model that resumes integrating a
state whose ``exner`` describes the atmosphere it had BEFORE the radar
spoke.  It will not crash.  It will be wrong.

So the instrument here is POSITIVE rather than an identity check.  It
recomputes exner from ``rho_theta`` through the equation of state and
asks how far the carried exner has drifted from the recomputation.

**The equation of state is the PORT's, not a textbook's.**  This is not
a stylistic preference; the first version of this module got it wrong
and halted a correct closed loop.  ``mpas_port.cuda_backend.recovery``
carries the only authority::

    theta = rtheta / rho;
    const T argument = zz * (rgas / p0) * rtheta;
    exner = mp_pow<T>(argument, rgas / (cp - rgas));
    pressure = zz * rgas * rtheta * exner;

Two things follow, and both were originally missed:

``zz`` is IN the argument.
    The port's prognostic mass variable is ``rho_zz`` (``state.rho``)
    and its mass-weighted potential temperature is ``rho_zz * theta_m``
    (``state.rho_theta``); ``zz = dzw / dz`` is the vertical metric
    (``mpas_port.vertical``).  Recomputing exner from ``rho_theta``
    alone leaves out ``zz`` entirely, and since exner is a power law the
    omission shows up as a departure of exactly ``|zz**(rd/cv) - 1|``.
    On the x1.40962 mesh ``zz`` runs 0.83 to 2.35, which puts that
    departure at 0.407 -- an instrument reading five orders of magnitude
    above the gate, produced by a state that was never wrong.

``rho_theta`` is already MOIST and takes no further correction.
    The port's own recovery names its ``rtheta / rho`` output
    ``theta_m``.  The moisture is inside the prognostic, and the EOS
    consumes ``rtheta`` directly.  Multiplying in another
    ``(1 + Rv/Rd q)`` here would be a second bug wearing the first one's
    clothes.

**It is validated in both directions and it reports its own resolution
floor, always.**  A residual instrument with no floor is how a threshold
gets tuned below the noise and a gate becomes decoration.  The floor is
MEASURED, and it is measured against the precision of the state actually
being graded rather than against the precision of this module's own
arithmetic.  That distinction is the module's second original defect: a
float64 recomputation differenced against itself always reports ~1e-16,
which is true about numpy and false about the instrument.  A port
boundary carries float32 ``exner``, and a float32 exner simply cannot be
pinned closer than its own representation granularity no matter how
exactly the physics agrees.  So the floor combines the granularity of
the carried field with the measured sensitivity of the recomputation to
one ulp of its inputs, and on a real port boundary it lands near 2e-7 --
not 2e-16.

The threshold below is unchanged, and deliberately so.  It sits above
the float32 floor and three orders below the smallest departure anybody
would call an analysis; the honest statement of its margin is in the
constant's own note.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from gpuwm.cycle.contracts import CONSISTENCY_SCHEMA, CycleRefusal

#: The default gate.  Justified against the measured floor rather than a
#: guess, and stated honestly for BOTH precisions the spine sees:
#:
#: * a float64 self-consistent state (the synthetic replay path) floors
#:   at ~1e-16, so 1e-6 clears it by ten orders of magnitude;
#: * a real float32 port boundary floors at ~2.1e-7, so 1e-6 clears it
#:   by ~4.7x -- narrow, and the number to quote rather than the
#:   flattering one.
#:
#: Above the gate the picture is unambiguous either way: a 1 % increment
#: to one ``rho_theta`` cell drives the residual to ~3e-3, three orders
#: above the threshold.  So the gate cannot fire on arithmetic at either
#: precision and cannot miss a real increment.  It is NOT to be widened
#: to admit a state it reads as inconsistent; if this instrument fires on
#: a state believed good, the instrument's physics is the suspect.
CONSISTENCY_THRESHOLD = 1.0e-6

#: The explicit stand-in for "this state lives on a unit vertical
#: metric".  ``vertical_metric`` is a REQUIRED keyword precisely so that
#: no caller can omit ``zz`` by accident a second time: a caller with a
#: metric passes the array, a caller on a synthetic unit-metric state
#: passes this constant, and there is no third way to call the function.
UNIT_VERTICAL_METRIC = "unit"


def _exner_from_rho_theta(rho_theta: np.ndarray, zz: np.ndarray, *,
                          p0: float, rd: float, cp: float) -> np.ndarray:
    """The port's ``pressure_point`` argument, in the port's order."""
    return (zz * (rd / p0) * rho_theta) ** (rd / (cp - rd))


def _vertical_metric(vertical_metric: Any,
                     shape: tuple[int, ...]) -> tuple[np.ndarray, str, Any]:
    """Resolve the metric keyword into an array, and say which it was."""
    if isinstance(vertical_metric, str):
        if vertical_metric != UNIT_VERTICAL_METRIC:
            raise CycleRefusal(
                "vertical_metric must be an array of zz or the literal "
                f"{UNIT_VERTICAL_METRIC!r}",
                vertical_metric=vertical_metric)
        return np.ones(shape, dtype=np.float64), UNIT_VERTICAL_METRIC, \
            np.dtype(np.float64)
    if vertical_metric is None:
        raise CycleRefusal(
            "the consistency instrument will not assume a vertical metric; "
            "MPAS forms exner from zz*(rd/p0)*rho_theta and omitting zz "
            "reads as a 0.4 residual on a correct state",
            remedy=f"pass vertical_metric=<zz array> or "
                   f"vertical_metric={UNIT_VERTICAL_METRIC!r}")
    array = np.asarray(vertical_metric)
    dtype = array.dtype
    array = array.astype(np.float64)
    if array.shape != shape:
        raise CycleRefusal("vertical metric zz does not have the shape of "
                           "rho_theta",
                           rho_theta_shape=tuple(shape),
                           zz_shape=tuple(array.shape))
    if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise CycleRefusal("vertical metric zz must be finite and positive",
                           zz_min=float(np.nanmin(array)),
                           zz_max=float(np.nanmax(array)))
    return array, "supplied", dtype


def hydrostatic_residual(prognostic: Mapping[str, np.ndarray],
                         derived: Mapping[str, np.ndarray], *,
                         vertical_metric: Any,
                         p0: float = 1.0e5, rd: float = 287.0,
                         cp: float = 1004.5) -> dict[str, Any]:
    """Max/mean relative departure of the carried exner from the recomputed.

    ``vertical_metric`` is the port's ``zz``, shaped like ``rho_theta``,
    or :data:`UNIT_VERTICAL_METRIC`.  It has no default on purpose.

    Returns the :data:`CONSISTENCY_SCHEMA` block, including
    ``resolution_floor`` -- the level below which this instrument cannot
    see anything at all for THIS state at THIS precision.
    """
    if "rho_theta" not in prognostic:
        raise CycleRefusal("consistency needs rho_theta to recompute exner",
                           present=sorted(prognostic))
    if "exner" not in derived:
        raise CycleRefusal("consistency needs a carried exner to compare "
                           "against", present=sorted(derived))

    stored = np.asarray(prognostic["rho_theta"])
    rho_theta = stored.astype(np.float64)
    carried_stored = np.asarray(derived["exner"])
    carried = carried_stored.astype(np.float64)
    if carried.shape != rho_theta.shape:
        raise CycleRefusal("carried exner does not have the shape of "
                           "rho_theta",
                           rho_theta_shape=tuple(rho_theta.shape),
                           exner_shape=tuple(carried.shape))

    zz, metric_kind, zz_dtype = _vertical_metric(vertical_metric,
                                                 tuple(rho_theta.shape))

    recomputed = _exner_from_rho_theta(rho_theta, zz, p0=p0, rd=rd, cp=cp)
    scale = np.maximum(np.abs(recomputed), np.finfo(np.float64).tiny)

    # The floor, part one: the same arithmetic, run again, differenced
    # against itself.  Recomputing from a trivially re-materialised copy
    # keeps the compiler from folding the two together.
    twice = _exner_from_rho_theta(
        np.asarray(rho_theta.tolist(), dtype=np.float64), zz,
        p0=p0, rd=rd, cp=cp)
    floor = float(np.max(np.abs(twice - recomputed) / scale))

    # The floor, part two, and the part that matters on a real boundary:
    # how far the recomputation moves when its inputs are nudged by one
    # ulp of the precision they were STORED in, plus the granularity of
    # the carried field itself.  A float32 exner cannot be pinned closer
    # than its own eps however exactly the physics agrees, so an
    # instrument that reports 1e-16 there is lying about what it can see.
    input_eps = max(_relative_eps(stored.dtype), _relative_eps(zz_dtype))
    nudged = _exner_from_rho_theta(rho_theta * (1.0 + input_eps),
                                   zz * (1.0 + input_eps),
                                   p0=p0, rd=rd, cp=cp)
    sensitivity = float(np.max(np.abs(nudged - recomputed) / scale))
    carried_granularity = _relative_eps(carried_stored.dtype)
    floor = max(floor, carried_granularity + sensitivity,
                float(np.finfo(np.float64).eps))

    relative = np.abs(carried - recomputed) / scale
    flat = int(np.argmax(relative))
    argmax = [int(index) for index in np.unravel_index(flat, relative.shape)]
    return {
        "schema": CONSISTENCY_SCHEMA,
        "max_relative_residual": float(relative.reshape(-1)[flat]),
        "mean_relative_residual": float(np.mean(relative)),
        "argmax_index": argmax,
        "resolution_floor": floor,
        "n_points": int(relative.size),
        "vertical_metric": metric_kind,
        "carried_dtype": str(carried_stored.dtype),
        "prognostic_dtype": str(stored.dtype),
    }


def _relative_eps(dtype: Any) -> float:
    """One ulp, relative, of a stored floating dtype (0.0 for integers)."""
    resolved = np.dtype(dtype)
    if resolved.kind != "f":
        return 0.0
    return float(np.finfo(resolved).eps)


def derived_is_stale(anchor_manifest: Mapping[str, Any]) -> bool:
    """Do the carried diagnostics still belong to these prognostics?

    Cheap, structural, and the first thing a resume should ask.  The
    residual instrument is the second: this one catches a state that was
    rewritten at all, that one catches a state that was rewritten in a
    way that matters thermodynamically.
    """
    parent = anchor_manifest.get("parent") or {}
    derived = anchor_manifest.get("derived") or {}
    return derived.get("derived_from_sha256") != parent.get(
        "prognostic_sha256")


def require_consistent(prognostic: Mapping[str, np.ndarray],
                       derived: Mapping[str, np.ndarray], *,
                       vertical_metric: Any,
                       threshold: float = CONSISTENCY_THRESHOLD,
                       label: str, p0: float = 1.0e5, rd: float = 287.0,
                       cp: float = 1004.5) -> dict[str, Any]:
    """Return the residual block, or refuse naming everything observed."""
    result = hydrostatic_residual(prognostic, derived,
                                  vertical_metric=vertical_metric,
                                  p0=p0, rd=rd, cp=cp)
    if result["max_relative_residual"] > float(threshold):
        raise CycleRefusal(
            "state and its carried diagnostics do not agree; rebuild the "
            "diagnostics from the prognostics before resuming",
            label=label,
            max_relative_residual=result["max_relative_residual"],
            mean_relative_residual=result["mean_relative_residual"],
            threshold=float(threshold),
            argmax_index=result["argmax_index"],
            resolution_floor=result["resolution_floor"],
            vertical_metric=result["vertical_metric"],
            n_points=result["n_points"])
    return result


def rebuild_exner(rho_theta: np.ndarray, vertical_metric: Any, *,
                  p0: float = 1.0e5, rd: float = 287.0,
                  cp: float = 1004.5) -> np.ndarray:
    """Rebuild exner the way the port's recovery kernel does.

    Exported so that the one place in the spine that REBUILDS exner and
    the one place that GRADES it can never drift apart again -- they were
    two independent transcriptions of the EOS and both omitted ``zz``.
    """
    zz, _, _ = _vertical_metric(vertical_metric,
                                tuple(np.asarray(rho_theta).shape))
    return _exner_from_rho_theta(np.asarray(rho_theta, dtype=np.float64), zz,
                                 p0=p0, rd=rd, cp=cp)
