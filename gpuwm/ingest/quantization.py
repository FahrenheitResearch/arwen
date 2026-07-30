"""Quantization-aware admission of bounded physical source fields.

The Python half of the argument the GRIB bridges make in Rust (see
``tools/grib1_bridge/src/quantization.rs``).  A quantity that is
physically AT a limit -- saturated soil, a solid-land fraction, an
ice-free ocean cell -- is stored at that limit and comes back a hair
outside it, because every stage between the encoder and here rounds:
GRIB2's ``(R + X * 2^E) * 10^-D`` reconstruction onto a fixed grid,
float32 storage of an f64 decode, unit conversions in a declarative
mapping.  A field user's preparation died on
``GFS_SM010040 value 1.0000000019073487 outside [0,1]`` for exactly that
reason, and the identical refusal was reproduced on the HRRR and mapped
routes.

The bridges can derive their tolerance from the record's own packing
parameters.  Nothing here can: by this point the packing is gone and
only an array remains.  So the tolerance is the round-off the pipeline
demonstrably carries -- a few float32 ulps of the bound's own magnitude
-- which is orders of magnitude below any physically meaningful
difference in a fraction, and orders of magnitude above the excursions
that were being refused.

This never widens a gate.  It moves values that are already at a bound
back ONTO it, counts them, and leaves everything else for the caller's
existing refusal, which is unchanged.  ERA5's soil preflight
(:mod:`gpuwm.ingest.preflight`) reached the same conclusion first, from
observed data, and states its own explicit window; this is the general
form of that judgement.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

#: Float32 ulps of head-room granted at a physical bound.  The same
#: constant the HRRR soil report already uses for its convex-hull
#: comparison, for the same reason: it is the width of the round-off
#: this pipeline carries, not a judgement about the data.
BOUND_ULPS = 4.0


class ClampReport(NamedTuple):
    """What a clamp did, for a receipt or a log line."""

    clamps: int
    max_excursion: float

    @property
    def clamped(self) -> bool:
        return self.clamps > 0


def bound_tolerance(bound: float, span: float) -> float:
    """Head-room at one bound: a few ulps of the larger of the two scales.

    ``span`` carries the case ``bound == 0.0``, where a relative
    tolerance would be zero and no dry cell could ever be admitted.
    """

    eps = float(np.finfo(np.float32).eps)
    return BOUND_ULPS * eps * max(abs(float(bound)), abs(float(span)))


def clamp_bound_kissing(values, *, minimum=None, maximum=None):
    """Move only the cells that are AT a bound, and say how many.

    Returns ``(values, ClampReport)``.  Cells inside the range are
    untouched; cells outside it by more than the tolerance are ALSO
    untouched, so the caller's own refusal still sees them exactly as
    before and still refuses them.  The input array is never modified in
    place -- a copy is made only when there is something to clamp.
    """

    array = np.asarray(values)
    if minimum is None and maximum is None:
        return array, ClampReport(0, 0.0)
    low = -np.inf if minimum is None else float(minimum)
    high = np.inf if maximum is None else float(maximum)
    span = high - low
    if not np.isfinite(span):
        span = max(abs(low) if np.isfinite(low) else 0.0,
                   abs(high) if np.isfinite(high) else 0.0)

    finite = np.isfinite(array)
    clamps = 0
    worst = 0.0
    result = array
    for bound, outside in (
        (low, finite & (array < low)),
        (high, finite & (array > high)),
    ):
        if not np.isfinite(bound) or not outside.any():
            continue
        tolerance = bound_tolerance(bound, span)
        excursion = np.abs(array - bound)
        kissing = outside & (excursion <= tolerance)
        count = int(np.count_nonzero(kissing))
        if count == 0:
            continue
        if result is array:
            result = array.astype(np.float64, copy=True)
        result[kissing] = bound
        clamps += count
        worst = max(worst, float(np.max(excursion[kissing])))
    return result, ClampReport(clamps, worst)


def admit_bounded(values, *, name: str, minimum=None, maximum=None,
                  subject: str = "source"):
    """Clamp what is at a bound, refuse what is past it, in one sentence.

    The refusal names the field, the range, the worst offender and the
    head-room it exceeded, so a reader can tell a packing artefact from
    a broken field without owning the decoder.
    """

    clamped, report = clamp_bound_kissing(
        values, minimum=minimum, maximum=maximum)
    finite = np.isfinite(clamped)
    if not finite.all():
        raise ValueError(f"{subject} {name} is non-finite")
    low = -np.inf if minimum is None else float(minimum)
    high = np.inf if maximum is None else float(maximum)
    bad = (clamped < low) | (clamped > high)
    if bad.any():
        offenders = clamped[bad]
        worst = float(offenders[np.argmax(
            np.maximum(low - offenders, offenders - high))])
        excursion = max(low - worst, worst - high)
        span = high - low if np.isfinite(high - low) else 0.0
        head_room = max(bound_tolerance(low, span) if np.isfinite(low) else 0.0,
                        bound_tolerance(high, span) if np.isfinite(high) else 0.0)
        raise ValueError(
            f"{subject} {name} value {worst!r} is outside "
            f"[{minimum},{maximum}] by {excursion!r}; values within "
            f"{head_room!r} of a bound are quantization and are clamped, "
            f"this one is not")
    return clamped, report


__all__ = [
    "BOUND_ULPS", "ClampReport", "admit_bounded", "bound_tolerance",
    "clamp_bound_kissing",
]
