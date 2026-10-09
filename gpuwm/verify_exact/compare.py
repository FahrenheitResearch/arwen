"""Bitwise scoring of one run against another.  CPU only, numpy only.

A run is held as a :class:`RunRecord`: the SHA-256 of every field at every
recorded step, plus an optional loader for the raw arrays at the steps that
kept them.  The stock WRF recordings carry digests for every step and raw
arrays at a few steps; a WOOF replay is recorded in the same form, so one
function scores either against the other.

The digest of a field is SHA-256 over its words exactly as stored:
little-endian, C order, the recorded dtype and shape, Time removed.  Two
digests agree only if every word agrees, so a digest match is a 0 ULP match
and needs no array.  Arrays are only needed to say HOW FAR apart two
differing fields are (max ULP, how many words, where), which is why they are
loaded only for fields whose digests already differ.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Callable, Iterable, Mapping

import numpy as np

from gpuwm.core.fp32_ulp import MISMATCH, fp32_ulp_distance

#: Order in which a step's differing fields are listed, and so which field
#: leads a report.  CASE.md's comparison list: prognostic state first, then
#: physics state, then diagnostics, then fields that never change in time.
CLASS_ORDER = ("state", "physics", "diagnostic", "static")

#: Prognostic state that is not a 3-D "Q" species (those are matched by rule).
_STATE = frozenset({"U", "V", "W", "PH", "T", "MU", "P", "QKE", "TKE_PBL", "TKE"})

#: CASE.md's diagnostics, reported separately so a diagnostic-only miss is
#: not read as a state miss.
_DIAGNOSTIC = frozenset({"T2", "Q2", "TH2", "U10", "V10", "PSFC", "REFL_10CM",
                         "REFD_MAX"})

#: Grid, coordinate, clock and time-invariant input fields.  A difference
#: here is an output-door difference, not an integration difference.
_STATIC = frozenset({
    "XLAT", "XLONG", "XLAT_U", "XLONG_U", "XLAT_V", "XLONG_V", "CLAT",
    "MAPFAC_M", "MAPFAC_U", "MAPFAC_V", "MAPFAC_MX", "MAPFAC_MY", "MAPFAC_UX",
    "MAPFAC_UY", "MAPFAC_VX", "MAPFAC_VY", "MF_VX_INV", "F", "E", "SINALPHA",
    "COSALPHA", "HGT", "LANDMASK", "LAKEMASK", "XLAND", "LU_INDEX", "IVGTYP",
    "ISLTYP", "ZNU", "ZNW", "ZS", "DZS", "FNM", "FNP", "RDNW", "RDN", "DNW",
    "DN", "CFN", "CFN1", "CF1", "CF2", "CF3", "RDX", "RDY", "AREA2D", "DX2D",
    "RESM", "ZETATOP", "P_TOP", "T00", "P00", "TLP", "TISO", "TLP_STRAT",
    "P_STRAT", "MAX_MSFTX", "MAX_MSFTY", "PB", "PHB", "MUB", "C1H", "C2H",
    "C1F", "C2F", "C3H", "C4H", "C3F", "C4F", "PCB", "XTIME", "ITIMESTEP",
    "THIS_IS_AN_IDEAL_RUN", "SAVE_TOPO_FROM_REAL", "GOT_VAR_SSO", "NEST_POS",
    "VAR_SSO", "VAR", "BATHYMETRY_FLAG", "SHDMAX", "SHDMIN", "SHDAVG",
    "SNOALB", "TMN", "ALBBCK", "SST_INPUT", "ISEEDARR_SPPT", "ISEEDARR_SKEBS",
    "ISEEDARR_RAND_PERTURB", "ISEEDARRAY_SPP_CONV", "ISEEDARRAY_SPP_PBL",
    "ISEEDARRAY_SPP_LSM", "WATER_DEPTH", "LAKE_DEPTH",
})


_GRID_CONSTANT = ("a time-invariant grid, map-factor or vertical-coordinate constant WRF "
                  "echoes into every frame; it holds no model state, and every use of it "
                  "reaches the compared prognostic state")
_STATIC_INPUT = ("a time-invariant static field WRF copies from wrfinput_d01, which both "
                 "sides read byte for byte; it holds no model state, and every use of it "
                 "reaches the compared state")
_RUN_FLAG = "a run flag, nest position or random seed WRF echoes into every frame; no model value"

#: WRF history fields WOOF may leave out of its own history without failing a
#: run, each with the reason.  Every other reference field WOOF does not write,
#: or writes in another shape, fails the run: a field that is not compared
#: cannot be called exact.  An entry is excused only where the reference's own
#: digests show the field never changed over the steps compared; a listed
#: field that changes in time fails like any other.  Nothing prognostic,
#: physical, accumulated or diagnostic belongs here (the tendencies, TKE_PBL,
#: surface, snow, lake, urban and radiation fields CASE.md names must be
#: written by WOOF and compared).  Fields WOOF writes are always compared,
#: including its own grid fields (XLAT, MAPFAC_M...): WOOF's dynamics use its
#: own map factors, so a difference there is not an output-only difference.
ALLOWED_ABSENT: dict[str, str] = {
    **{name: _GRID_CONSTANT for name in (
        "FNM", "FNP", "RDNW", "RDN", "DNW", "DN", "CFN", "CFN1", "CF1", "CF2", "CF3",
        "RDX", "RDY", "AREA2D", "DX2D", "ZETATOP", "T00", "P00", "TLP", "TISO",
        "TLP_STRAT", "P_STRAT", "MAX_MSFTX", "MAX_MSFTY", "C1H", "C2H", "C1F", "C2F",
        "C3H", "C4H", "C3F", "C4F", "PCB", "MAPFAC_MX", "MAPFAC_MY", "MAPFAC_UX",
        "MAPFAC_UY", "MAPFAC_VX", "MAPFAC_VY", "MF_VX_INV", "CLAT", "ZS", "DZS")},
    **{name: _STATIC_INPUT for name in (
        "VAR_SSO", "VAR", "CON", "OA1", "OA2", "OA3", "OA4", "OL1", "OL2", "OL3", "OL4",
        "VARLS", "CONLS", "OA1LS", "OA2LS", "OA3LS", "OA4LS", "OL1LS", "OL2LS", "OL3LS",
        "OL4LS", "VARSS", "CONSS", "OA1SS", "OA2SS", "OA3SS", "OA4SS", "OL1SS", "OL2SS",
        "OL3SS", "OL4SS", "SHDMAX", "SHDMIN", "SHDAVG", "SNOALB", "ALBBCK", "LAKEMASK",
        "WATER_DEPTH", "BATHYMETRY_FLAG", "SST_INPUT")},
    **{name: _RUN_FLAG for name in (
        "THIS_IS_AN_IDEAL_RUN", "SAVE_TOPO_FROM_REAL", "GOT_VAR_SSO", "NEST_POS",
        "ISEEDARR_SPPT", "ISEEDARR_SKEBS", "ISEEDARR_RAND_PERTURB", "ISEEDARRAY_SPP_CONV",
        "ISEEDARRAY_SPP_PBL", "ISEEDARRAY_SPP_LSM")},
}


def field_class(name: str, ndim: int) -> str:
    """``state``, ``physics``, ``diagnostic`` or ``static`` for one field.

    Every 3-D field whose name starts with ``Q`` is a moist species, number
    concentration or aerosol number (prognostic state), except the PBL
    scheme's own subgrid clouds (``QC_BL``, ``QI_BL``).  Everything not
    named as state, diagnostic or static is physics state.
    """

    if name in _STATE:
        return "state"
    if name in _DIAGNOSTIC:
        return "diagnostic"
    if name in _STATIC:
        return "static"
    if ndim == 3 and name.startswith("Q") and not name.endswith("_BL"):
        return "state"
    return "physics"


def field_digest(values, dtype: str, shape) -> str | None:
    """SHA-256 of ``values`` stored as ``dtype`` (little-endian, C order).

    ``None`` when the array cannot be stored as ``dtype`` and ``shape``
    without changing a word: a different shape, or a value the recorded
    dtype cannot hold exactly (a reader that widened float32 to float64
    converts back exactly; a genuinely different value does not).  The
    caller counts ``None`` as a difference, never as a match.
    """

    want = np.dtype(dtype).newbyteorder("<")
    array = np.asarray(values)
    if tuple(array.shape) != tuple(int(s) for s in shape):
        return None
    if array.dtype != want:
        with np.errstate(invalid="ignore", over="ignore"):
            cast = array.astype(want)
        if not _round_trips(array, cast):
            return None
        array = cast
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _round_trips(original: np.ndarray, cast: np.ndarray) -> bool:
    back = cast.astype(original.dtype)
    if original.dtype.kind == "f":
        return bool(np.array_equal(back, original, equal_nan=True))
    return bool(np.array_equal(back, original))


@dataclass(frozen=True)
class FieldMeta:
    name: str
    dtype: str
    shape: tuple[int, ...]

    @property
    def field_class(self) -> str:
        return field_class(self.name, len(self.shape))


@dataclass
class RunRecord:
    """Digests of one run, by step, and how to get its raw arrays.

    ``digests[step][name]`` is the SHA-256 of field ``name`` at model step
    ``step`` (``None`` where the field could not be stored in its recorded
    form).  ``raw(step)`` returns ``{name: array}`` for a step that kept
    raw arrays, or ``None``; it may return a subset of the fields.
    """

    fields: dict[str, FieldMeta]
    digests: dict[int, dict[str, str | None]]
    raw: Callable[[int], Mapping[str, np.ndarray] | None] = lambda step: None
    raw_steps: tuple[int, ...] = ()
    label: str = ""

    @property
    def steps(self) -> tuple[int, ...]:
        return tuple(sorted(self.digests))

    @property
    def last_step(self) -> int:
        return max(self.digests) if self.digests else -1


@dataclass(frozen=True)
class UlpDetail:
    """How far apart one field is at one step."""

    field: str
    step: int
    max_ulp: int
    differing_words: int
    words: int
    #: (index tuple, candidate value, reference value) of the first and the
    #: worst differing word, in the recorded array's own axis order
    #: (bottom_top, south_north, west_east for a 3-D field).
    first: tuple
    worst: tuple
    nonfinite: bool = False
    #: Largest absolute difference over the finite differing words.  A gap of
    #: a few ULP across zero is billions of ULP; this says how big it is.
    max_abs: float = 0.0

    def describe(self) -> str:
        ulp = "NaN mismatch" if self.max_ulp >= MISMATCH else f"{self.max_ulp} ulp"
        return (f"{ulp} (|diff| <= {self.max_abs:.3g}) at step {self.step} "
                f"({self.differing_words}/{self.words} words, first at {_fmt_index(self.first[0])})")


def _fmt_index(index) -> str:
    return "(" + ",".join(str(int(i)) for i in index) + ")"


def ulp_detail(name: str, step: int, candidate, reference) -> UlpDetail:
    """Max ULP, differing-word count and the first and worst positions.

    Float fields use the repository's one FP32 ULP metric; integer fields
    report the absolute integer difference in the same slot.
    """

    cand = np.asarray(candidate)
    ref = np.asarray(reference)
    if cand.shape != ref.shape:
        raise ValueError(f"{name}: shape {cand.shape} against {ref.shape}")
    if ref.dtype.kind == "f":
        distance = fp32_ulp_distance(cand.astype(np.float32), ref.astype(np.float32))
        differs = (np.ascontiguousarray(cand, dtype=np.float32).view(np.uint32)
                   != np.ascontiguousarray(ref, dtype=np.float32).view(np.uint32))
    else:
        distance = np.abs(cand.astype(np.int64) - ref.astype(np.int64))
        differs = distance != 0
    count = int(np.count_nonzero(differs))
    if count == 0:
        zero = (tuple(0 for _ in ref.shape), None, None)
        return UlpDetail(name, step, 0, 0, int(ref.size), zero, zero)
    flat_first = int(np.flatnonzero(differs.ravel())[0])
    flat_worst = int(np.argmax(np.where(differs, distance, -1).ravel()))
    first = np.unravel_index(flat_first, ref.shape)
    worst = np.unravel_index(flat_worst, ref.shape)
    nonfinite = bool(cand.dtype.kind == "f" and not np.isfinite(cand).all())
    with np.errstate(invalid="ignore", over="ignore"):
        gap = np.abs(cand.astype(np.float64) - ref.astype(np.float64))[differs]
    gap = gap[np.isfinite(gap)]
    max_abs = float(gap.max()) if gap.size else float("nan")
    return UlpDetail(
        name, step, int(distance.ravel()[flat_worst]), count, int(ref.size),
        (tuple(int(i) for i in first), cand.ravel()[flat_first].item(),
         ref.ravel()[flat_first].item()),
        (tuple(int(i) for i in worst), cand.ravel()[flat_worst].item(),
         ref.ravel()[flat_worst].item()),
        nonfinite, max_abs)


@dataclass
class Comparison:
    """One candidate run scored against one reference run."""

    reference: str
    #: Steps both runs recorded, in order.
    steps_compared: tuple[int, ...]
    #: Fields both runs carry (after exclusions), in reference order.
    fields_compared: tuple[str, ...]
    #: Reference fields the candidate never wrote, and the ones excluded.
    absent_in_candidate: tuple[str, ...]
    excluded: tuple[str, ...]
    #: Reference steps the candidate did not reach.
    missing_steps: tuple[int, ...]
    #: Fields both write but in different shapes: not comparable word for
    #: word, so listed here (candidate shape, reference shape).  Any entry
    #: means the run is not a match.
    reshaped: dict[str, tuple] = field(default_factory=dict)
    #: The absent fields ALLOWED_ABSENT excuses (name -> reason), and the
    #: absent fields nothing excuses.  Any unexcused absence means the run
    #: is not a match.
    absent_allowed: dict[str, str] = field(default_factory=dict)
    absent_unexcused: tuple[str, ...] = ()
    #: First step with any differing field, and that step's differing
    #: fields in report order (CLASS_ORDER, then reference order).
    first_step: int | None = None
    first_fields: tuple[str, ...] = ()
    #: Every field that differs at any compared step, and at how many steps.
    differing: dict[str, int] = field(default_factory=dict)
    #: For each field class, the first step at which a field of that class
    #: differs and that step's differing fields of the class.  A run whose
    #: initial frame differs only in time-invariant fields still says where
    #: its prognostic state first left the reference.
    first_by_class: dict[str, tuple[int, tuple[str, ...]]] = field(default_factory=dict)
    #: ULP detail for the lead field (and any other field asked for).
    details: list[UlpDetail] = field(default_factory=list)

    @property
    def matched(self) -> bool:
        """0 ULP on every reference field at every reference step.

        A reference field the candidate did not write (unless ALLOWED_ABSENT
        excuses it) or wrote in another shape was never compared, so a run
        with one is not a match even when every compared field agrees.
        """
        return (self.first_step is None and not self.missing_steps
                and bool(self.steps_compared) and not self.uncompared)

    @property
    def uncompared(self) -> tuple[str, ...]:
        """Reference fields that block a match: unexcused absent, then reshaped."""
        return (*self.absent_unexcused, *self.reshaped)

    @property
    def lead_field(self) -> str | None:
        return self.first_fields[0] if self.first_fields else None

    @property
    def classes(self) -> tuple[str, ...]:
        """The classes of every differing field, in CLASS_ORDER."""
        return tuple(c for c in CLASS_ORDER if c in self._class_set())

    def _class_set(self) -> set[str]:
        return {self._meta_class.get(n, "physics") for n in self.differing}

    _meta_class: dict[str, str] = field(default_factory=dict, repr=False)

    @property
    def diagnostics_only(self) -> bool:
        """Every difference is a diagnostic or static field."""
        return bool(self.differing) and self._class_set() <= {"diagnostic", "static"}

    def lead_detail(self) -> UlpDetail | None:
        return self.detail_for(self.lead_field)

    def detail_for(self, name: str | None) -> UlpDetail | None:
        for detail in self.details:
            if detail.field == name:
                return detail
        return None

    @property
    def state_first(self) -> tuple[int, tuple[str, ...]] | None:
        """First step and fields at which prognostic state differs."""
        return self.first_by_class.get("state")

    def summary(self) -> str:
        """One short clause: ``match`` or the first step, field and ULP."""

        if not self.steps_compared:
            return "no common step" + self._uncompared_clause()
        if self.first_step is None:
            if self.missing_steps:
                return (f"match through step {self.steps_compared[-1]}, candidate stopped "
                        f"before step {self.missing_steps[0]}" + self._uncompared_clause())
            if self.uncompared:
                return (f"{len(self.fields_compared)} compared fields match"
                        + self._uncompared_clause())
            return "match"
        return self._difference_clause() + self._uncompared_clause()

    def _uncompared_clause(self) -> str:
        parts = []
        if self.absent_unexcused:
            parts.append(f"{len(self.absent_unexcused)} WRF fields WOOF does not write "
                         f"({_names(self.absent_unexcused)})")
        if self.reshaped:
            parts.append(f"{len(self.reshaped)} written in another shape "
                         f"({_names(tuple(self.reshaped))})")
        return ("; not compared: " + ", ".join(parts)) if parts else ""

    def _difference_clause(self) -> str:
        more = len(self.first_fields) - 1
        text = f"step {self.first_step} {self.lead_field}"
        if more:
            text += f" (+{more} more)"
        text += _ulp_clause(self.lead_detail())
        state = self.state_first
        if state is not None and state[1][0] != self.lead_field:
            step, names = state
            text += f"; state from step {step} {names[0]}"
            if len(names) > 1:
                text += f" (+{len(names) - 1} more)"
            text += _ulp_clause(self.detail_for(names[0]))
        elif state is None and self.lead_field is not None:
            text += "; prognostic state matches"
        if self.diagnostics_only:
            text += " [diagnostics/static only]"
        return text

    def as_json(self) -> dict:
        return {
            "reference": self.reference,
            "matched": self.matched,
            "steps_compared": list(self.steps_compared),
            "fields_compared": len(self.fields_compared),
            "absent_in_candidate": list(self.absent_in_candidate),
            "excluded": list(self.excluded),
            "absent_allowed": dict(self.absent_allowed),
            "absent_unexcused": list(self.absent_unexcused),
            "reshaped": {n: list(v) for n, v in self.reshaped.items()},
            "missing_steps": list(self.missing_steps),
            "first_step": self.first_step,
            "first_fields": list(self.first_fields),
            "lead_field": self.lead_field,
            "lead_class": (self._meta_class.get(self.lead_field)
                           if self.lead_field else None),
            "differing": dict(self.differing),
            "first_by_class": {c: {"step": v[0], "fields": list(v[1])}
                               for c, v in self.first_by_class.items()},
            "diagnostics_only": self.diagnostics_only,
            "details": [{
                "field": d.field, "step": d.step, "max_ulp": d.max_ulp,
                "differing_words": d.differing_words, "words": d.words,
                "first": {"index": list(d.first[0]), "candidate": d.first[1],
                          "reference": d.first[2]},
                "worst": {"index": list(d.worst[0]), "candidate": d.worst[1],
                          "reference": d.worst[2]},
                "candidate_nonfinite": d.nonfinite, "max_abs_diff": d.max_abs,
            } for d in self.details],
            "summary": self.summary(),
        }


def _names(names: tuple[str, ...], shown: int = 6) -> str:
    text = ", ".join(names[:shown])
    return text + (f" +{len(names) - shown} more" if len(names) > shown else "")


def _ulp_clause(detail: UlpDetail | None) -> str:
    if detail is None:
        return " max ULP needs the reference's raw arrays"
    return " max " + detail.describe()


def compare_runs(candidate: RunRecord, reference: RunRecord, *,
                 exclude: Iterable[str] = (), through_step: int | None = None,
                 detail_fields: Iterable[str] = ()) -> Comparison:
    """Score ``candidate`` against ``reference`` at every common step.

    ``exclude`` names fields never compared (a field the reference's own
    repeat test showed to hold uninitialised memory).  A reference field the
    candidate does not write, or writes in another shape, is listed and
    blocks a match unless ALLOWED_ABSENT excuses its absence and the
    reference never changed it over the compared steps.  ``through_step``
    stops the comparison at a step (a reference that crashed is compared
    only over the steps it wrote).  ULP detail is computed for the lead
    field and every name in ``detail_fields`` wherever both runs carry raw
    arrays: at the first differing step if the reference kept that step,
    otherwise at the next step that kept arrays and still differs.
    """

    excluded = tuple(n for n in reference.fields if n in set(exclude))
    skip = set(excluded)
    reshaped = tuple(n for n in reference.fields
                     if n not in skip and n in candidate.fields
                     and candidate.fields[n].shape != reference.fields[n].shape)
    skip.update(reshaped)
    compared = tuple(n for n in reference.fields
                     if n not in skip and n in candidate.fields)
    absent = tuple(n for n in reference.fields
                   if n not in skip and n not in candidate.fields)
    ref_steps = [s for s in reference.steps
                 if through_step is None or s <= through_step]
    common = tuple(s for s in ref_steps if s in candidate.digests)
    missing = tuple(s for s in ref_steps if s not in candidate.digests)
    meta_class = {n: reference.fields[n].field_class for n in compared}
    order = {n: (CLASS_ORDER.index(meta_class[n]), i) for i, n in enumerate(compared)}

    result = Comparison(reference.label, common, compared, absent, excluded, missing,
                        _meta_class=meta_class)
    result.reshaped = {n: (list(candidate.fields[n].shape), list(reference.fields[n].shape))
                       for n in reshaped}
    for name in absent:
        written = {reference.digests[s].get(name) for s in ref_steps} - {None}
        if name in ALLOWED_ABSENT and len(written) <= 1:
            result.absent_allowed[name] = ALLOWED_ABSENT[name]
    result.absent_unexcused = tuple(n for n in absent if n not in result.absent_allowed)
    for step in common:
        want = reference.digests[step]
        got = candidate.digests[step]
        # A field the candidate did not write at a step the reference wrote
        # it is a difference; one neither wrote there is not compared.
        differing = [n for n in compared
                     if n in want and got.get(n) != want[n]]
        for name in differing:
            result.differing[name] = result.differing.get(name, 0) + 1
        if differing and result.first_step is None:
            result.first_step = step
            result.first_fields = tuple(sorted(differing, key=order.__getitem__))
        for name in sorted(differing, key=order.__getitem__):
            klass = meta_class[name]
            if klass not in result.first_by_class:
                result.first_by_class[klass] = (step, tuple(
                    n for n in sorted(differing, key=order.__getitem__) if meta_class[n] == klass))
    if result.first_step is not None:
        state = result.state_first
        wanted = [result.lead_field, *(() if state is None else (state[1][0],)),
                  *[n for n in detail_fields if n in result.differing]]
        for name in dict.fromkeys(wanted):
            detail = _first_detail(candidate, reference, name, result.first_step, common)
            if detail is not None:
                result.details.append(detail)
    return result


def _first_detail(candidate: RunRecord, reference: RunRecord, name: str,
                  first_step: int, common: tuple[int, ...]) -> UlpDetail | None:
    for step in common:
        if step < first_step:
            continue
        want = reference.digests[step].get(name)
        got = candidate.digests[step].get(name)
        if want is not None and got == want:
            continue
        ref_raw = reference.raw(step)
        if not ref_raw or name not in ref_raw:
            continue
        cand_raw = candidate.raw(step)
        if not cand_raw or name not in cand_raw:
            continue
        return ulp_detail(name, step, cand_raw[name], ref_raw[name])
    return None
