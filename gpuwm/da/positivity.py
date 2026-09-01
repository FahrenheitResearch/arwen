"""EXPERIMENTAL: the hydrometeor positivity policy the filter refuses to own.

:func:`gpuwm.da.letkf.analyze` returns per-member **increments** and does not
clip them.  That is deliberate and it is right: a Gaussian filter applied to a
bounded, heavily zero-inflated variable will routinely propose a negative
mixing ratio, and the three sane responses -- clip, transform, reject -- are
not equivalent, differ in what they do to mass and to the ensemble's own
spread, and are properly the caller's choice.  A filter that clipped would be
making that choice silently for every caller forever.

So the caller must choose, and this module makes the choice explicit,
bounded, and *counted*.

**What ``clip`` actually does, stated honestly.**  Where ``prior + increment``
would be negative, the increment is replaced by ``-prior`` so the analysis is
exactly ``0``.  That is a change of ``-analysis > 0``: **clipping at zero adds
mass.**  It is not conservative and it is biased in one direction, always
wetward for a mixing ratio.  The bias is small when the filter is well tuned
and large when it is not, which makes the count of clipped points and the
mass added the single most useful diagnostic of whether the analysis is
healthy.  Both go into the receipt, per field, per member.  A cycle whose
clipped mass grows leg over leg is a cycle whose covariances are wrong, and
the manifest should be able to show that without a rerun.

**The alternatives, and why neither is the v1.2 default.**

``anamorphosis``
    Assimilate a transformed variable -- ``log(q + q0)`` or a Gaussian
    anamorphosis against the ensemble's own empirical CDF -- so a negative
    analysis is unrepresentable rather than repaired.  It is the principled
    answer and it is not a policy that can live here: it changes what H(x)
    means, what the observation error means, and what the ensemble spread
    means, so it belongs in the forward operator and the filter's input, not
    in a post-hoc fixup.  The zero-inflation also has to be handled
    explicitly (``log 0`` is not a number and ``q0`` is a tuning knob nobody
    has tuned).
``reject``
    Discard the whole increment at any gridpoint where any constrained field
    would go negative, leaving that column at its background value.  It
    conserves the background exactly and adds no mass, but it introduces
    discontinuities between a rejected gridpoint and its analysed neighbour
    -- the model then has to absorb a gradient the filter invented -- and it
    throws away the correction to *unconstrained* fields at that point, which
    were not the problem.  Implemented here, because it is four lines and the
    comparison is worth being able to run; not the default.

Nothing here is wired into a default route.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

#: Provenance schema for the receipt this module produces.
POSITIVITY_SCHEMA = "gpuwm-da.positivity.v1"

#: Policies, by name.  ``none`` exists so a caller can state that it
#: considered the question and declined, which is different from a caller
#: that never thought about it -- the receipt records which.
POLICIES = ("clip", "reject", "none")

#: State attributes that are physically non-negative and must never be
#: analysed below zero.  Taken from the restart prognostic contract
#: (``gpuwm.io.restart.STATE_SERIALIZED_ATTRS``): mixing ratios, number
#: concentrations, and the two-moment volume variables.  Generic names
#: only -- nothing here knows about any case or any source.
NON_NEGATIVE_FIELDS = (
    "qv", "qc", "qr", "qi", "qs", "qg", "qh", "qndrop",
    "nc", "nr", "ni", "ns", "ng",
    "qnr", "qni", "qns", "qng", "qnh", "qnn",
    "qvolg", "qvolh",
)


class PositivityError(ValueError):
    """A refusal.  Never a silent repair of the repair."""


def constrained_fields(fields: Sequence[str]) -> tuple[str, ...]:
    """Which of ``fields`` this module has a positivity opinion about."""

    return tuple(name for name in fields if name in NON_NEGATIVE_FIELDS)


def _host(array):
    """A host view of a device or host array, without importing cupy."""

    get = getattr(array, "get", None)
    if callable(get) and hasattr(array, "__cuda_array_interface__"):
        return np.asarray(get())
    return np.asarray(array)


def apply_positivity(prior: Mapping[str, object],
                     increments: Mapping[str, object], *,
                     policy: str = "clip",
                     fields: Sequence[str] | None = None) -> tuple[dict, dict]:
    """Enforce non-negativity of ``prior + increment``.  Returns a receipt.

    ``prior`` and ``increments`` are ``{field: array}`` of matching shape --
    either ``(nz, ny, nx)`` for one member or ``(R, nz, ny, nx)`` for a whole
    ensemble; this module is shape-agnostic because the constraint is
    pointwise.

    Fields outside :data:`NON_NEGATIVE_FIELDS` pass through **untouched and
    unexamined**: a theta or wind increment has no positivity constraint and
    inventing one would be a bug that looks like caution.

    The returned increments are a new mapping; the inputs are not mutated.
    """

    if policy not in POLICIES:
        raise PositivityError(
            f"unknown positivity policy {policy!r}; known policies are "
            f"{POLICIES}. There is no default that is right for every "
            "caller, which is why the filter does not pick one")
    names = tuple(increments) if fields is None else tuple(fields)
    missing = [name for name in names if name not in increments]
    if missing:
        raise PositivityError(
            f"positivity was asked about {missing}, which the increment "
            "mapping does not carry")

    constrained = constrained_fields(names)
    adjusted = {name: increments[name] for name in increments}
    per_field: list[dict] = []
    total_points = 0
    total_mass = 0.0

    negative_masks = []
    for name in constrained:
        if name not in prior:
            raise PositivityError(
                f"cannot enforce positivity on {name!r} without its "
                "background: the constraint is on prior + increment, and "
                "an increment alone does not say whether the analysis is "
                "negative")
        background = _host(prior[name]).astype(np.float64, copy=False)
        increment = _host(increments[name]).astype(np.float64, copy=False)
        if background.shape != increment.shape:
            raise PositivityError(
                f"{name}: prior {background.shape} and increment "
                f"{increment.shape} disagree")
        analysis = background + increment
        negative = analysis < 0.0
        negative_masks.append((name, background, increment, analysis,
                               negative))

    if policy == "reject":
        # One shared mask: rejecting per field would leave a column whose
        # species disagree about which analysis they came from.
        shared = None
        for _, _, _, _, negative in negative_masks:
            shared = negative if shared is None else (shared | negative)
        for name, background, increment, analysis, negative in negative_masks:
            count = int(np.count_nonzero(negative))
            mass = float(-analysis[negative].sum()) if count else 0.0
            reverted = np.where(shared, 0.0, increment)
            adjusted[name] = reverted.astype(
                _host(increments[name]).dtype, copy=False)
            total_points += count
            total_mass += mass
            per_field.append({
                "field": name, "policy": "reject",
                "negative_points": count,
                "mass_that_would_have_been_added": mass,
                "points_reverted": int(np.count_nonzero(shared)),
                "total_points": int(negative.size),
            })
    else:
        for name, background, increment, analysis, negative in negative_masks:
            count = int(np.count_nonzero(negative))
            # The clip raises the analysis from `analysis` to 0, so it ADDS
            # this much.  Recorded with that sign and that name, because
            # "clipped mass" reads as mass removed and it is not.
            mass = float(-analysis[negative].sum()) if count else 0.0
            if policy == "clip" and count:
                clipped = np.where(negative, -background, increment)
                adjusted[name] = clipped.astype(
                    _host(increments[name]).dtype, copy=False)
            total_points += count
            total_mass += mass
            per_field.append({
                "field": name, "policy": policy,
                "negative_points": count,
                "mass_added_by_clip": mass if policy == "clip" else 0.0,
                "mass_left_negative": 0.0 if policy == "clip" else mass,
                "total_points": int(negative.size),
                "worst_negative": (float(analysis[negative].min())
                                   if count else 0.0),
            })

    receipt = {
        "schema": POSITIVITY_SCHEMA,
        "stability": "experimental",
        "policy": policy,
        "constrained_fields": list(constrained),
        "unconstrained_fields": [name for name in names
                                 if name not in constrained],
        "negative_points": total_points,
        "mass_added_by_clip": total_mass if policy == "clip" else 0.0,
        # Only 'none' leaves negative mass standing.  'reject' reverts the
        # increment at every offending point, so the analysis there IS the
        # background -- nonnegative by construction -- and reporting the
        # would-have-been-added mass under this name said the opposite of
        # what happened: a receipt claiming the run shipped negative
        # hydrometeor mass it had in fact refused to ship.  The quantity
        # itself is not lost; it is the per-field
        # ``mass_that_would_have_been_added``, which is what it always was.
        "mass_left_negative": total_mass if policy == "none" else 0.0,
        # What 'reject' actually does, named rather than implied.  It reverts
        # the increment for the CONSTRAINED fields at the shared mask and
        # leaves the same covariance increment standing everywhere else --
        # wind and theta at those points keep their analysed values.  That is
        # a constrained-field reject, not a whole-state one, and the two give
        # different analyses; which one this product should do is a ruling to
        # be taken, not a default to be changed quietly here.
        "positivity_semantics": ("constrained-field-reject"
                                 if policy == "reject" else policy),
        "per_field": per_field,
        "note": (
            "clip-at-zero ADDS mass and is biased wetward; the counts above "
            "are the diagnostic, not a formality. Alternatives are "
            "anamorphosis (belongs in the forward operator) and reject "
            "(conserves the background, invents gradients)."),
    }
    if policy == "none":
        receipt["note"] = (
            "policy 'none': negative analyses were counted and left in "
            "place. This is a stated choice, not an oversight, and the "
            "microphysics will meet them.")
    return adjusted, receipt


def verify_non_negative(prior: Mapping[str, object],
                        increments: Mapping[str, object], *,
                        fields: Sequence[str] | None = None) -> None:
    """Post-condition: assert no constrained analysis is below zero.

    Cheap, and it is the check that catches a policy that ran on the wrong
    mapping -- the failure mode where a receipt says 4 117 points were
    clipped and the increments that got written were the unclipped ones.
    """

    names = tuple(increments) if fields is None else tuple(fields)
    for name in constrained_fields(names):
        analysis = (_host(prior[name]).astype(np.float64, copy=False)
                    + _host(increments[name]).astype(np.float64, copy=False))
        worst = float(analysis.min())
        if worst < 0.0:
            raise PositivityError(
                f"{name}: the analysis is still negative after the "
                f"positivity policy ran (minimum {worst:g}); the policy was "
                "applied to a different mapping than the one about to be "
                "written")
