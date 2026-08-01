"""EXPERIMENTAL (ArWen v1.2): what a multi-moment scheme's analysis must be.

A two-moment microphysics scheme's state is PAIRS.  ``qr`` without ``nr``
is not a rain field with a missing diagnostic; it is not a rain field at
all -- the scheme's own slope closure reads both, and the pair is what
carries the drop size distribution.  An analysis that updates one member
of the pair and not the other produces a state the scheme cannot
evaluate.

This is not hypothetical.  A rung-2 real-radar LETKF cycle ran with

    ANALYSIS_FIELDS = ('thp','qv','u','v','qc','qr','qi','qs','qg')

on a Morrison (``mp_physics=10``) state.  Nine fields, no number
concentrations.  Reflectivity assimilation created hydrometeor mass in
cells the background had left clear, so the analysis held ``q`` up to
6.4 g/kg with ``N`` still at the background's exact zero -- for one
member, **100%** of the in-mask offenders were background-clear cells:
21 332 rain, 37 592 snow, 27 548 graupel.  Morrison's slope math then
does what it honestly must:

    lam = (six_c * N / q)**(1/3)  ->  0
    ilam = 1/lam                  ->  +inf
    n0  = N * ... * lam**e        ->  0
    Ze  = n0 * ... * ilam**e      ->  0 * inf = nan

and the reflectivity operator retains the NaN rather than letting a
corrupt state masquerade as its -35 dBZ clear-air floor
(``gpuwm.verify.npref`` :9231-9239).  The operator was right.  The
ANALYSIS STATE was unphysical, and reflectivity OmA was unrecoverable
for that cycle.

That field list lived in a work script, but a product that lets a work
script do this has the defect.  So:

**Part 1 -- the state vector is derived from the scheme, never typed
out.**  :func:`analysis_fields` builds the update list from the selected
scheme's own prognostic moment set (``gpuwm.core.nssl2_contract`` for
NSSL, the Registry-derived pairs below for the rest), and
:func:`validate_analysis_fields` refuses a list that moves a mass field
while leaving a paired number moment the state actually carries
untouched.  A caller that genuinely wants a single-moment update must say
so -- ``policy="single-moment-with-repair"`` -- and that policy REQUIRES
the guard below, because it is the policy that creates the offenders.

**Part 2 -- the guard, after any update.**  :func:`moment_consistency_report`
counts cells where a mass field is above the scheme's own activity
threshold and its paired number moment is at or below zero.
:func:`repair_moments` fixes them **using the scheme's own authority**,
not an invented intercept: for Morrison that is the PSD limiter at
``module_mp_morr_two_moment.F:1525-1638``, invoked here through its
float64 mirror ``gpuwm.verify.npref._np_morrison_slopes``.  Given
``q > 0`` and ``N = 0`` that limiter computes ``lam = 0``, clamps it to
the species' own lower bound ``lam_min``, and back-computes
``N = q * lam_min**3 / six_c`` -- the largest-particle limit, which is
exactly what the scheme would impose at the first step of the next leg.
The repair is therefore not a new closure; it is the state the scheme
was going to bound the analysis to anyway, applied before the
reflectivity operator is asked about it.

**Where a scheme's repair authority is not ported, there is no repair.**
NSSL (18) sets number from mass through its own fixed intercepts in
``module_mp_nssl_2mom.F``; that routine is not in this tree, and
inventing an intercept would be tuning science.  NSSL therefore DETECTS
and REFUSES.  A refusal naming 27 548 graupel cells is a usable result;
a number produced by a made-up intercept is not.

Nothing here is wired into a default route, and nothing here changes a
filter: the LETKF still returns increments and still owns no policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

#: Provenance schema for every receipt this module produces.
MOMENT_SCHEMA = "gpuwm-da.moment-policy.v1"

#: What an analysis may do with a multi-moment scheme's state.
#:
#: ``"full-moment"`` (the default) -- every mass field in the update
#:   brings its paired number moment (and, for NSSL, its volume moment)
#:   with it.  Standard EnKF practice for a two-moment scheme, and the
#:   only policy under which the analysis is a state of the scheme that
#:   produced the background.
#:
#: ``"single-moment-with-repair"`` -- the caller updates mass only, on
#:   purpose, and accepts that the numbers are then the background's.
#:   Legitimate (the moment covariances are noisy and some centres do
#:   exactly this) and it REQUIRES the consistency guard, because it is
#:   the policy that manufactures ``q > 0, N = 0``.  Declaring it without
#:   running the repair is refused, not warned about.
MOMENT_POLICIES = ("full-moment", "single-moment-with-repair")
DEFAULT_MOMENT_POLICY = "full-moment"

#: Where each scheme's repair comes from.  ``None`` means the authority
#: is not ported into this tree and the guard refuses instead.
MORRISON_REPAIR_AUTHORITY = (
    "WRF v4.6.1 module_mp_morr_two_moment.F:1525-1638 PSD limiter, via "
    "its float64 mirror gpuwm.verify.npref._np_morrison_slopes")


class MomentPolicyError(ValueError):
    """A refusal about the moment structure of an analysis."""


@dataclass(frozen=True)
class MomentPair:
    """One species' prognostic moments.

    ``mass`` and ``number`` are state attribute names; ``volume`` is
    NSSL's predicted-density moment where the species has one.  Every
    field a scheme advances for this species and nothing else.
    """

    species: str
    mass: str
    number: str
    volume: str | None = None

    @property
    def fields(self) -> tuple[str, ...]:
        names = [self.mass, self.number]
        if self.volume is not None:
            names.append(self.volume)
        return tuple(names)


@dataclass(frozen=True)
class SchemeMoments:
    """A microphysics scheme's prognostic moment structure.

    Derived from the scheme's own contract, never from a list typed at a
    call site: `mp_physics` picks the scheme and the scheme states what
    it advances.  A field list that disagrees with this is a field list
    that has silently truncated somebody's state.
    """

    mp_physics: int
    name: str
    #: Mass fields the scheme advances with NO prognostic number moment.
    mass_only: tuple[str, ...]
    #: Species carrying a prognostic number (and possibly volume) moment.
    pairs: tuple[MomentPair, ...]
    #: Prognostic fields that are neither a mass nor a paired moment --
    #: NSSL's predicted CCN, for one.  Analysed with the rest under the
    #: full-moment policy; they have no consistency partner.
    unpaired: tuple[str, ...] = ()
    #: How this scheme's q>0/N=0 cells are repaired, or ``None``.
    repair_authority: str | None = None
    #: The scheme's own mass activity threshold (kg/kg).  Below it the
    #: scheme treats the species as absent and demands no number.
    q_threshold: float = 1.0e-14

    @property
    def two_moment(self) -> bool:
        return bool(self.pairs)

    @property
    def mass_fields(self) -> tuple[str, ...]:
        return self.mass_only + tuple(pair.mass for pair in self.pairs)

    @property
    def number_fields(self) -> tuple[str, ...]:
        return tuple(pair.number for pair in self.pairs)

    @property
    def volume_fields(self) -> tuple[str, ...]:
        return tuple(pair.volume for pair in self.pairs
                     if pair.volume is not None)

    @property
    def hydrometeor_fields(self) -> tuple[str, ...]:
        """Every prognostic hydrometeor field, mass and moment alike."""
        names: list[str] = list(self.mass_only)
        for pair in self.pairs:
            names.extend(pair.fields)
        names.extend(self.unpaired)
        return tuple(names)

    def pair_for_mass(self, mass: str) -> MomentPair | None:
        for pair in self.pairs:
            if pair.mass == mass:
                return pair
        return None


#: Morrison's five prognostic number moments (Registry: nc/nr/ni/ns/ng),
#: the scheme whose limiter supplies the repair.
_MORRISON = SchemeMoments(
    mp_physics=10, name="Morrison two-moment",
    mass_only=("qv",),
    pairs=(
        MomentPair("cloud", "qc", "nc"),
        MomentPair("rain", "qr", "nr"),
        MomentPair("ice", "qi", "ni"),
        MomentPair("snow", "qs", "ns"),
        MomentPair("graupel", "qg", "ng"),
    ),
    repair_authority=MORRISON_REPAIR_AUTHORITY,
    #: MQSMALL, module_mp_morr_two_moment.F / gpuwm morrison.cu:26.  The
    #: scheme's own activity gate, and therefore the exact threshold above
    #: which it will read the number moment.
    q_threshold=1.0e-14,
)

#: Thompson (8) predicts rain number and ice number only; its cloud,
#: snow and graupel are single-moment.
_THOMPSON = SchemeMoments(
    mp_physics=8, name="Thompson",
    mass_only=("qv", "qc", "qs", "qg"),
    pairs=(
        MomentPair("rain", "qr", "nr"),
        MomentPair("ice", "qi", "ni"),
    ),
    repair_authority=None,
    q_threshold=1.0e-14,
)

_WSM6 = SchemeMoments(
    mp_physics=6, name="WSM6",
    mass_only=("qv", "qc", "qr", "qi", "qs", "qg"),
    pairs=(),
)

_KESSLER = SchemeMoments(
    mp_physics=1, name="Kessler",
    mass_only=("qv", "qc", "qr"),
    pairs=(),
)


def _nssl_scheme(**selectors) -> SchemeMoments:
    """NSSL's moment set, from the scheme's OWN selector resolution.

    ``gpuwm.core.nssl2_contract.resolve_nssl2_mode`` applies WRF's
    option-18 consistency pass, so hail, predicted CCN and the predicted
    volume moments are on or off exactly as the run has them.  Enumerating
    NSSL's fields here instead would be a second contract, and a second
    contract is how a state gets truncated.
    """
    from gpuwm.core import nssl2_contract as contract

    mode = contract.resolve_nssl2_mode(**selectors)
    transported = set(mode.transported_fields)
    pairs = [
        MomentPair("cloud", "qc", contract.TWO_MOMENT_NUMBER_FIELDS[0]),
        MomentPair("rain", "qr", contract.TWO_MOMENT_NUMBER_FIELDS[1]),
        MomentPair("ice", "qi", contract.TWO_MOMENT_NUMBER_FIELDS[2]),
        MomentPair("snow", "qs", contract.TWO_MOMENT_NUMBER_FIELDS[3]),
        MomentPair("graupel", "qg", contract.TWO_MOMENT_NUMBER_FIELDS[4],
                   contract.GRAUPEL_VOLUME_FIELD
                   if contract.GRAUPEL_VOLUME_FIELD in transported else None),
    ]
    if mode.hail:
        pairs.append(MomentPair(
            "hail", contract.HAIL_MASS_FIELD, contract.HAIL_NUMBER_FIELD,
            contract.HAIL_VOLUME_FIELD
            if contract.HAIL_VOLUME_FIELD in transported else None))
    if not mode.two_moment:
        return SchemeMoments(
            mp_physics=contract.MP_PHYSICS, name="NSSL (single-moment mode)",
            mass_only=tuple(name for name in mode.transported_fields
                            if name.startswith("q")
                            and not name.startswith("qn")),
            pairs=())
    kept = tuple(pair for pair in pairs if pair.number in transported)
    unpaired = ((contract.PREDICTED_CCN_FIELD,)
                if mode.predicted_ccn else ())
    return SchemeMoments(
        mp_physics=contract.MP_PHYSICS, name="NSSL two-moment",
        mass_only=("qv",),
        pairs=kept,
        unpaired=unpaired,
        # module_mp_nssl_2mom.F sets number from mass through the scheme's
        # own fixed intercepts (cnor/cnos/cnoh/cnohl).  That routine is not
        # ported into this tree; inventing an intercept here would be
        # tuning science, so NSSL detects and refuses instead of repairing.
        repair_authority=None,
    )


#: Every scheme this module knows the moment structure of.  Keyed on
#: ``mp_physics``, which is the only identity a run has.
_STATIC_SCHEMES = {scheme.mp_physics: scheme
                   for scheme in (_KESSLER, _WSM6, _THOMPSON, _MORRISON)}


def scheme_moments(mp_physics: int, **selectors) -> SchemeMoments:
    """The prognostic moment structure of ``mp_physics``.

    ``selectors`` are the scheme's own namelist switches where it has
    them (NSSL's ``nssl_hail_on`` and friends); passing them to a scheme
    that has none is a refusal rather than a silently ignored argument.
    """
    key = int(mp_physics)
    if key == 18:
        return _nssl_scheme(**selectors)
    if selectors:
        raise MomentPolicyError(
            f"mp_physics={key} takes no scheme selectors, got "
            f"{sorted(selectors)}; only mp_physics=18 resolves its moment "
            "set from namelist switches")
    if key not in _STATIC_SCHEMES:
        raise MomentPolicyError(
            f"no moment structure is registered for mp_physics={key}; "
            f"known schemes are {sorted(_STATIC_SCHEMES)} and 18. An "
            "analysis cannot decide what a scheme's state vector is by "
            "guessing at it")
    return _STATIC_SCHEMES[key]


#: The non-hydrometeor fields an analysis normally updates.  Offered as a
#: default so a caller states the hydrometeor policy and not the whole
#: list; it is a default, not a contract, and any of it can be replaced.
DEFAULT_BASE_FIELDS = ("thp", "qv", "u", "v")


def analysis_fields(mp_physics: int, *,
                    base: Sequence[str] = DEFAULT_BASE_FIELDS,
                    hydrometeors: bool = True,
                    policy: str = DEFAULT_MOMENT_POLICY,
                    **selectors) -> tuple[str, ...]:
    """The LETKF ``analysis_fields`` for a scheme, derived from the scheme.

    Under ``full-moment`` the hydrometeor half is the scheme's whole
    prognostic hydrometeor set -- masses, numbers, and NSSL's volume
    moments together.  Under ``single-moment-with-repair`` it is the mass
    fields only, which is the caller stating the truncation rather than
    performing it by omission.

    ``hydrometeors=False`` analyses none of them, which is a complete and
    consistent choice: the background's pairs stay exactly as the
    background had them and nothing can become inconsistent.
    """
    _check_policy(policy)
    scheme = scheme_moments(mp_physics, **selectors)
    names = list(dict.fromkeys(base))
    if hydrometeors:
        wanted = (scheme.hydrometeor_fields
                  if policy == "full-moment" else scheme.mass_fields)
        for name in wanted:
            if name not in names:
                names.append(name)
    return tuple(names)


def _check_policy(policy: str) -> None:
    if policy not in MOMENT_POLICIES:
        raise MomentPolicyError(
            f"unknown moment policy {policy!r}; choose from "
            f"{', '.join(MOMENT_POLICIES)}. There is no default that is "
            "right for every caller, and the wrong one produces a state "
            "the scheme cannot evaluate")


def pairs_present(available: Sequence[str], *,
                  mp_physics: int | None = None,
                  **selectors) -> tuple[MomentPair, ...]:
    """The moment pairs a state actually carries.

    With ``mp_physics`` this is the named scheme's pairs restricted to
    what the state has.  WITHOUT it, the pairs are detected from the field
    spellings themselves -- Morrison's ``nr`` and NSSL's ``qnr`` are
    different names for different schemes' moments, so a state answers the
    question "which pairs am I" without being told.  Detection exists
    because the code that writes an analysis
    (:mod:`gpuwm.ensemble.increments`) is handed a checkpoint and no
    namelist, and a guard that could be skipped by not passing a config
    is a guard that will be.
    """
    have = set(available)
    if mp_physics is not None:
        scheme = scheme_moments(mp_physics, **selectors)
        candidates = scheme.pairs
    else:
        if selectors:
            raise MomentPolicyError(
                "scheme selectors need an explicit mp_physics; they cannot "
                "be applied to a detected moment structure")
        candidates = tuple(
            pair for scheme in (_MORRISON, _THOMPSON, _nssl_scheme())
            for pair in scheme.pairs)
    found: dict[str, MomentPair] = {}
    for pair in candidates:
        if pair.mass in have and pair.number in have:
            volume = pair.volume if (pair.volume in have) else None
            found.setdefault(pair.number,
                             MomentPair(pair.species, pair.mass,
                                        pair.number, volume))
    return tuple(found.values())


def validate_analysis_fields(fields: Sequence[str], *,
                             available: Sequence[str] | None = None,
                             mp_physics: int | None = None,
                             policy: str = DEFAULT_MOMENT_POLICY,
                             **selectors) -> dict:
    """Refuse an update that truncates a multi-moment state.

    ``fields`` is what the analysis proposes to update; ``available`` is
    what the background actually carries (the state's own capability).  A
    mass field in ``fields`` whose paired number moment is in
    ``available`` but NOT in ``fields`` is the node-8 defect exactly, and
    under ``full-moment`` it is a refusal naming every such species.

    Under ``single-moment-with-repair`` the same situation is allowed and
    the returned receipt sets ``repair_required``; a caller that ignores
    it and writes the analysis anyway is refused by the guard in
    :mod:`gpuwm.ensemble.increments`, which is where the state is.
    """
    _check_policy(policy)
    updated = tuple(dict.fromkeys(fields))
    carried = tuple(updated) if available is None else tuple(available)
    pairs = pairs_present(carried, mp_physics=mp_physics, **selectors)
    truncated = [pair for pair in pairs
                 if pair.mass in updated and pair.number not in updated]
    volume_truncated = [
        pair for pair in pairs
        if pair.mass in updated and pair.volume is not None
        and pair.volume not in updated]
    receipt = {
        "schema": MOMENT_SCHEMA,
        "policy": policy,
        "mp_physics": None if mp_physics is None else int(mp_physics),
        "scheme_source": "declared" if mp_physics is not None else "detected",
        "pairs_carried": [pair.mass for pair in pairs],
        "pairs_updated": [pair.mass for pair in pairs
                          if pair.mass in updated
                          and pair.number in updated],
        "mass_only_species": [pair.mass for pair in truncated],
        "volume_omitted_species": [pair.mass for pair in volume_truncated],
        "repair_required": bool(truncated or volume_truncated),
    }
    if not truncated and not volume_truncated:
        return receipt
    if policy == "single-moment-with-repair":
        return receipt
    detail = ", ".join(
        f"{pair.mass} without {pair.number}" for pair in truncated)
    volumes = ", ".join(
        f"{pair.mass} without {pair.volume}" for pair in volume_truncated)
    raise MomentPolicyError(
        "this analysis updates the mass of a multi-moment species and "
        "leaves its prognostic moment at the background's value: "
        + "; ".join(part for part in (detail, volumes) if part)
        + ". The background carries those moments, so the result is not a "
        "state of the scheme that produced it: where the analysis creates "
        "mass in a cell the background left clear, the pair becomes "
        "q > 0 with N = 0 and the scheme's slope closure evaluates to "
        "NaN. Analyse both moments (moment policy 'full-moment', which "
        "gpuwm.da.moments.analysis_fields builds for you), or declare "
        "'single-moment-with-repair' and let the moment-consistency guard "
        "repair the pairs it breaks.")


def _host(array):
    """A host float64 view of a device or host array."""
    get = getattr(array, "get", None)
    if callable(get) and hasattr(array, "__cuda_array_interface__"):
        return np.asarray(get(), dtype=np.float64)
    return np.asarray(array, dtype=np.float64)


def _depleted_offenders(mass: np.ndarray, number: np.ndarray,
                        threshold: float) -> np.ndarray:
    """Cells the scheme calls active whose number moment is spent.

    The ONE place this comparison is spelled, because
    :func:`moment_consistency_report` decides who is repaired and
    :func:`repair_moments` writes them, and two spellings of that pair
    would let the repair miss a cell the guard counted (or write one it
    did not).  ``mass > threshold`` is False for a NaN mass, so this
    predicate deliberately says nothing about non-finite cells --
    :func:`_nonfinite_offenders` is where they are found.
    """
    return np.isfinite(mass) & (mass > threshold) & (number <= 0.0)


def _nonfinite_offenders(mass: np.ndarray, number: np.ndarray,
                         volume: np.ndarray | None,
                         threshold: float) -> tuple:
    """``(mass, number, volume)`` masks of non-finite active moments.

    IEEE comparison is why this function exists.  ``NaN <= 0.0`` is
    **False** and ``NaN > threshold`` is **False**, so a cell holding
    ``qr = 1e-3`` with ``nr = NaN`` is neither an active cell nor an
    offender by the depleted-number predicate: the guard called it
    consistent, the writer wrote it, and the receipt attested to it.
    That is precisely the state the module's own docstring says the
    scheme evaluates to NaN -- ``lam = (six_c * N / q)**(1/3)`` is NaN
    for a NaN ``N`` just as surely as it is zero for ``N = 0``.

    A non-finite MASS is counted wherever it appears.  ``-inf`` is not
    above the threshold and ``NaN`` compares False against it, so neither
    can be shown to be an inactive cell the scheme would ignore; a
    hydrometeor mass that is not a finite number of kg/kg is not a state
    to decide pair-consistency for at all.  Number and volume moments are
    counted where the mass is active OR is itself non-finite -- below the
    scheme's own activity gate the scheme reads neither.
    """
    finite_mass = np.isfinite(mass)
    # "active, or not provably inactive": the cells the scheme will read
    # the pair's other moments at.
    reads_moments = (~finite_mass) | (mass > threshold)
    bad_mass = ~finite_mass
    bad_number = reads_moments & ~np.isfinite(number)
    bad_volume = (reads_moments & ~np.isfinite(volume)
                  if volume is not None else None)
    return bad_mass, bad_number, bad_volume


def moment_consistency_report(state: Mapping[str, object], *,
                              mp_physics: int | None = None,
                              pairs: Sequence[MomentPair] | None = None,
                              q_threshold: float | None = None,
                              **selectors) -> dict:
    """Count cells whose moment pair the scheme cannot evaluate.

    Two kinds of cell, counted separately because only one of them can be
    repaired:

    ``offending_cells``
        mass above the scheme's activity threshold with a number moment
        at or below zero.  The scheme's own limiter knows what number
        belongs there (:func:`repair_moments`).

    ``nonfinite_cells``
        a mass, number or volume moment that is not a finite number where
        the scheme will read it.  No limiter repairs this: Morrison's
        bound is ``N = q * lam_min**3 / six_c``, which is NaN for a NaN
        ``q`` and cannot be evaluated for a NaN ``N`` in the first place.
        It is a refusal, and it is counted here rather than left to the
        arithmetic because IEEE comparison hides it from every ``<=``
        and ``>`` in the depleted-number check.

    ``consistent`` is true only when BOTH are zero.

    ``state`` is any ``{field: array}`` mapping -- a live state's
    ``__dict__``, a checkpoint's ``state/*`` arrays, or one member's
    analysis.  The threshold defaults to the scheme's own activity gate,
    which is the exact value above which the scheme will read the number
    moment: below it the scheme treats the species as absent and asks
    nothing of the pair.
    """
    if pairs is None:
        pairs = pairs_present(tuple(state), mp_physics=mp_physics,
                              **selectors)
    if q_threshold is None:
        q_threshold = (scheme_moments(mp_physics, **selectors).q_threshold
                       if mp_physics is not None else _MORRISON.q_threshold)
    threshold = float(q_threshold)
    species: list[dict] = []
    total = 0
    nonfinite_total = 0
    for pair in pairs:
        mass = _host(state[pair.mass])
        number = _host(state[pair.number])
        volume = (_host(state[pair.volume])
                  if pair.volume is not None and pair.volume in state
                  else None)
        offenders = _depleted_offenders(mass, number, threshold)
        count = int(np.count_nonzero(offenders))
        total += count
        bad_mass, bad_number, bad_volume = _nonfinite_offenders(
            mass, number, volume, threshold)
        bad_any = bad_mass | bad_number
        if bad_volume is not None:
            bad_any = bad_any | bad_volume
        bad_count = int(np.count_nonzero(bad_any))
        nonfinite_total += bad_count
        entry = {
            "species": pair.species,
            "mass_field": pair.mass,
            "number_field": pair.number,
            "volume_field": pair.volume,
            "offending_cells": count,
            "max_offending_mass_kg_kg": (
                float(mass[offenders].max()) if count else 0.0),
            "nonfinite_cells": bad_count,
            "nonfinite_mass_cells": int(np.count_nonzero(bad_mass)),
            "nonfinite_number_cells": int(np.count_nonzero(bad_number)),
            "nonfinite_volume_cells": (
                0 if bad_volume is None
                else int(np.count_nonzero(bad_volume))),
        }
        species.append(entry)
    return {
        "schema": MOMENT_SCHEMA,
        "q_threshold_kg_kg": threshold,
        "mp_physics": None if mp_physics is None else int(mp_physics),
        "pairs_checked": [pair.mass for pair in pairs],
        "offending_cells_total": total,
        "nonfinite_cells_total": nonfinite_total,
        "consistent": total == 0 and nonfinite_total == 0,
        "species": species,
    }


def nonfinite_moment_refusal(report: Mapping[str, object], *,
                             where: str) -> str:
    """The message for a state whose moments are not finite.

    One spelling, used by :func:`repair_moments` and by both writers in
    :mod:`gpuwm.ensemble.increments`, so a caller cannot tell which path
    caught it -- and so neither path can be fixed without the other.
    """
    parts = []
    for entry in report["species"]:
        if not entry.get("nonfinite_cells"):
            continue
        detail = [f"{entry['mass_field']}: {entry['nonfinite_mass_cells']}",
                  f"{entry['number_field']}: "
                  f"{entry['nonfinite_number_cells']}"]
        if entry.get("volume_field") and entry.get("nonfinite_volume_cells"):
            detail.append(f"{entry['volume_field']}: "
                          f"{entry['nonfinite_volume_cells']}")
        parts.append(f"{entry['species']} ({', '.join(detail)})")
    return (
        f"the analysis for {where} carries "
        f"{report['nonfinite_cells_total']} cell(s) whose moment pair is "
        "not a finite number where the scheme reads it: "
        + "; ".join(parts)
        + ". IEEE comparison is why this is a separate refusal: NaN is "
          "neither above the activity threshold nor at-or-below zero, so a "
          "NaN number moment beside active mass passes every ordering test "
          "the depleted-number guard makes and would otherwise be attested "
          "as consistent. No scheme limiter repairs it either -- Morrison's "
          "bound N = q * lam_min**3 / six_c is NaN for a NaN q and has "
          "nothing to bound for a NaN N. Fix the background or the "
          "increment; this state cannot be analysed.")


#: Chunk of cells the Morrison limiter is called over at a time.  The
#: mirror allocates roughly twenty float64 arrays of the length it is
#: given, so a whole nest at once is gigabytes for no benefit.
_REPAIR_CHUNK_CELLS = 1 << 20


def _morrison_bounded_numbers(state: Mapping[str, object],
                              pairs: Sequence[MomentPair], *,
                              morr_rimed_ice: int = 1):
    """Morrison's own bounded numbers for the state's five species.

    Calls the scheme mirror -- ``gpuwm.verify.npref._np_morrison_slopes``,
    the float64 transcription of the limiter at
    ``module_mp_morr_two_moment.F:1525-1638`` -- rather than restating its
    constants.  A second copy of ``lam_min`` here is a second copy that
    can drift from the scheme by one edit.

    ``reset_cloud_number=False``: the limiter's ``INUM=1`` branch would
    overwrite every cloud number with 250 cm-3, which is the scheme's
    entry condition and not a repair.

    Density and temperature reach the limiter ONLY through the cloud
    branch's ``pgam``, multiplied by the cloud number -- which at an
    offending cell is zero.  The bounded numbers this returns AT OFFENDING
    CELLS are therefore independent of both, and
    ``tests/test_da_moments.py`` pins that rather than asserting it.
    """
    from gpuwm.verify.npref import _np_morrison_slopes

    by_species = {pair.species: pair for pair in pairs}
    shape = _host(state[pairs[0].mass]).shape
    size = int(np.prod(shape)) if shape else 1
    out = {pair.number: np.empty(size, np.float64) for pair in pairs}
    flat_q, flat_n = {}, {}
    for key, species in (("c", "cloud"), ("r", "rain"), ("i", "ice"),
                         ("s", "snow"), ("g", "graupel")):
        pair = by_species.get(species)
        if pair is None:
            flat_q[key] = np.zeros(size, np.float64)
            flat_n[key] = np.zeros(size, np.float64)
        else:
            flat_q[key] = _host(state[pair.mass]).reshape(-1)
            flat_n[key] = _host(state[pair.number]).reshape(-1)

    for start in range(0, size, _REPAIR_CHUNK_CELLS):
        stop = min(start + _REPAIR_CHUNK_CELLS, size)
        window = slice(start, stop)
        chunk_q = {key: value[window] for key, value in flat_q.items()}
        chunk_n = {key: value[window] for key, value in flat_n.items()}
        # Any positive density and any temperature: see the docstring.
        # The limiter reads them only where it is multiplied by a cloud
        # number that is zero at every cell this repair touches.
        ones = np.ones(stop - start, np.float64)
        _, _, bounded = _np_morrison_slopes(
            chunk_q, chunk_n, ones, 273.15 * ones,
            reset_cloud_number=False, morr_rimed_ice=morr_rimed_ice)
        for key, species in (("c", "cloud"), ("r", "rain"), ("i", "ice"),
                             ("s", "snow"), ("g", "graupel")):
            pair = by_species.get(species)
            if pair is not None:
                out[pair.number][window] = bounded[key]
    return {name: values.reshape(shape) for name, values in out.items()}


def repair_moments(state: Mapping[str, object], *,
                   mp_physics: int | None = None,
                   pairs: Sequence[MomentPair] | None = None,
                   q_threshold: float | None = None,
                   morr_rimed_ice: int = 1,
                   **selectors) -> tuple[dict, dict]:
    """``(repaired fields, receipt)`` for the pairs an analysis broke.

    Only OFFENDING cells are written.  The limiter would rebound every
    active cell's number -- that is what the scheme does at its first
    step -- but a data-assimilation repair that quietly rewrote healthy
    moments would be a second analysis nobody asked for, and it would be
    invisible in the receipt.  So the scheme's bounded number is taken,
    and applied where and only where the pair was broken.

    Refuses when the scheme has no ported repair authority: NSSL's
    intercept-based number initialisation is not in this tree, and a
    number invented here would be a tuning decision wearing a repair's
    clothes.

    Refuses BEFORE that, and for every scheme, when any moment is not
    finite where the scheme reads it.  A repair is a limiter evaluated on
    the state; a limiter evaluated on NaN returns NaN, so "repairing" a
    non-finite pair would replace one unevaluable state with another and
    stamp the receipt ``repaired``.
    """
    scheme = (scheme_moments(mp_physics, **selectors)
              if mp_physics is not None else None)
    if pairs is None:
        pairs = pairs_present(tuple(state), mp_physics=mp_physics,
                              **selectors)
    report = moment_consistency_report(
        state, mp_physics=mp_physics, pairs=pairs, q_threshold=q_threshold)
    if report["nonfinite_cells_total"]:
        raise MomentPolicyError(
            nonfinite_moment_refusal(report, where="this state"))
    if report["consistent"] or not pairs:
        return {}, {**report, "repaired": False, "repaired_cells_total": 0,
                    "authority": None if scheme is None
                    else scheme.repair_authority}

    authority = scheme.repair_authority if scheme is not None else None
    if scheme is None:
        # Detected structure: Morrison's spellings are the only ones this
        # tree can repair, and they identify the scheme unambiguously.
        detected = {pair.number for pair in pairs}
        if detected <= set(_MORRISON.number_fields):
            authority = MORRISON_REPAIR_AUTHORITY
    if authority is None:
        raise MomentPolicyError(
            "this analysis left "
            f"{report['offending_cells_total']} cell(s) holding mass above "
            f"{report['q_threshold_kg_kg']:g} kg/kg with a number moment at "
            "or below zero, and no repair authority for this scheme is "
            "ported into this tree ("
            + ", ".join(f"{entry['mass_field']}: {entry['offending_cells']}"
                        for entry in report["species"]
                        if entry["offending_cells"]) + "). "
            "The scheme's own number initialisation is what would have to "
            "supply the missing moments, and inventing an intercept here "
            "would be a science decision, not a repair. Analyse the number "
            "moments with the mass instead.")

    bounded = _morrison_bounded_numbers(state, pairs,
                                        morr_rimed_ice=morr_rimed_ice)
    threshold = report["q_threshold_kg_kg"]
    repaired: dict[str, np.ndarray] = {}
    total = 0
    for pair, entry in zip(pairs, report["species"]):
        if not entry["offending_cells"]:
            continue
        mass = _host(state[pair.mass])
        number = _host(state[pair.number])
        offenders = _depleted_offenders(mass, number, threshold)
        fixed = np.array(number, copy=True)
        fixed[offenders] = bounded[pair.number][offenders]
        repaired[pair.number] = fixed
        entry["repaired_cells"] = int(entry["offending_cells"])
        entry["repaired_number_min"] = float(fixed[offenders].min())
        entry["repaired_number_max"] = float(fixed[offenders].max())
        total += int(entry["offending_cells"])
    return repaired, {
        **report,
        "repaired": True,
        "repaired_cells_total": total,
        "authority": authority,
        "note": ("only cells with mass above the scheme's activity "
                 "threshold and a non-positive number moment were "
                 "written; healthy moments are untouched"),
    }


__all__ = [
    "DEFAULT_BASE_FIELDS",
    "DEFAULT_MOMENT_POLICY",
    "MOMENT_POLICIES",
    "MOMENT_SCHEMA",
    "MORRISON_REPAIR_AUTHORITY",
    "MomentPair",
    "MomentPolicyError",
    "SchemeMoments",
    "analysis_fields",
    "moment_consistency_report",
    "nonfinite_moment_refusal",
    "pairs_present",
    "repair_moments",
    "scheme_moments",
    "validate_analysis_fields",
]
