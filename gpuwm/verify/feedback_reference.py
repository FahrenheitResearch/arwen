"""Two-way-nesting (feedback) certification against a stock-WRF A/B pair.

Stock WRF emits no parent->child or child->parent coupling field: nest
coupling is an in-memory operation, no ``auxhist`` stream carries it, and
a ``feedback = 1`` run's history file carries the identical variable set
to a ``feedback = 0`` one.  The tendency-field form of a vs-WRF nest
comparison therefore cannot be archived, and this module does not pretend
otherwise -- :func:`build_feedback_ab_tier` carries the tendency form as
an explicit ``not_available`` entry with its evidence.

What stock WRF does permit is an A/B *state* difference: two runs whose
only difference is ``domains.feedback``, so every non-zero difference in
the parent's state is attributable to the feedback operator and nothing
else.  This module scores that difference:

* the pair's layout is read from its own ``provenance.json`` rather than
  from a path convention, and the single-variable guarantee
  (identical inputs, one namelist difference, pristine build) is
  CHECKED -- :func:`load_reference_pair_layout` refuses a pair that does
  not carry it, because a difference measured across two variables is
  not evidence about either one;
* each field is routed to its feedback operator class by the shipped
  operator map's ``op``/``xstag``/``ystag`` triple, never by field name.
  Name routing is the failure this map exists to prevent: the 10 m wind
  diagnostics are carried at mass points and take the cell-average
  operator, while the prognostic winds take the face average, so a
  name-based router assigns two of the most-inspected fields to the
  wrong operator and does so silently;
* the parent region each class is written over is DERIVED from the
  nest geometry and the class's staggering, then checked against the
  zones the pair itself shipped.

Scope.  This module scores the nest-coupling (feedback) tier only.  The
acoustic and gravity-wave conservation receipts are deferred, not
computed here and not computed anywhere in this release; the deferral is
recorded in the receipt itself through
:func:`gpuwm.verify.conservation_closure.deferred_tier` so a reader of a
receipt sees the absence rather than inferring a silent pass.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

SCHEMA_ID = "gpuwm.feedback-reference-ab/v1"

#: The operator classes WRF's feedback path actually has.  ``mass``,
#: ``masked`` and ``integer`` are unstaggered; ``u_face``/``v_face`` are
#: the two-field special case that carries the momentum feedback;
#: ``mark_domain`` marks the nest position and is not a physical field.
OPERATOR_CLASSES: frozenset[str] = frozenset({
    "mass", "masked", "integer", "u_face", "v_face", "mark_domain",
})

#: Classes whose feedback zone is the unstaggered (cell-centre) zone.
_UNSTAGGERED_CLASSES: frozenset[str] = frozenset({
    "mass", "masked", "integer", "mark_domain",
})

#: WRF restricts every feedback operator to the parent cells that did
#: NOT supply the nest's own lateral boundary condition
#: (``share/interp_fcn.F``, ``spec_zone = 1``).
FEEDBACK_SPEC_ZONE = 1

#: The four scalar metrics the shipped delta analysis reports, in the
#: order it reports them.  Scoring a pair with a different metric set
#: would make the two analyses uncomparable.
DIFFERENCE_METRICS: tuple[str, ...] = (
    "max_abs", "mean_abs", "rms", "frac_nonzero",
)


class ReferencePairError(ValueError):
    """A reference pair that cannot support the attribution it is used for."""


@dataclass(frozen=True)
class Zone:
    """0-based inclusive index box on the parent grid."""

    i0: int
    i1: int
    j0: int
    j1: int

    def __post_init__(self) -> None:
        if self.i1 < self.i0 or self.j1 < self.j0:
            raise ReferencePairError(f"empty zone {self!r}")

    def as_dict(self) -> dict[str, list[int]]:
        return {"i_0based": [self.i0, self.i1], "j_0based": [self.j0, self.j1]}

    def mask(self, ny: int, nx: int) -> np.ndarray:
        """Boolean ``(ny, nx)`` mask; an out-of-range zone raises."""
        if self.i1 >= nx or self.j1 >= ny:
            raise ReferencePairError(
                f"zone {self.as_dict()} does not fit a {ny}x{nx} grid")
        out = np.zeros((ny, nx), dtype=bool)
        out[self.j0:self.j1 + 1, self.i0:self.i1 + 1] = True
        return out


@dataclass(frozen=True)
class ReferencePairLayout:
    """Where the pair lives and what its single-variable guarantee is."""

    #: ``run_a_dir``/``run_b_dir`` stay strings: they are the paths the
    #: pair was PRODUCED at, on the machine that produced it, and turning
    #: a foreign absolute path into a local one would silently rewrite
    #: the record.  A caller scoring a copied subset supplies its own
    #: local roots.
    root: Path
    run_a_dir: str
    run_b_dir: str
    feedback_a: int
    feedback_b: int
    parent_grid_ratio: int
    i_parent_start: int
    j_parent_start: int
    nest_e_we: int
    nest_e_sn: int
    parent_e_we: int
    parent_e_sn: int
    only_difference: str
    inputs_identical: bool
    build_pristine: bool
    build_commit: str
    build_exe_sha256: str
    provenance_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_a_dir": self.run_a_dir,
            "run_b_dir": self.run_b_dir,
            "feedback_a": self.feedback_a,
            "feedback_b": self.feedback_b,
            "parent_grid_ratio": self.parent_grid_ratio,
            "i_parent_start": self.i_parent_start,
            "j_parent_start": self.j_parent_start,
            "nest_e_we": self.nest_e_we,
            "nest_e_sn": self.nest_e_sn,
            "parent_e_we": self.parent_e_we,
            "parent_e_sn": self.parent_e_sn,
            "only_difference": self.only_difference,
            "inputs_identical": self.inputs_identical,
            "build_pristine": self.build_pristine,
            "build_commit": self.build_commit,
            "build_exe_sha256": self.build_exe_sha256,
            "provenance_sha256": self.provenance_sha256,
        }


def _require(mapping: Mapping[str, Any], key: str, what: str) -> Any:
    if key not in mapping:
        raise ReferencePairError(f"{what} is missing {key!r}")
    return mapping[key]


def load_reference_pair_layout(provenance_path: Path) -> ReferencePairLayout:
    """Read the pair's layout from its own provenance record.

    The guarantee this pair exists to provide is that ONE namelist entry
    differs between the two runs.  Three things are checked rather than
    assumed, and each failure raises instead of degrading to a warning:
    the two runs' inputs are byte-identical, their feedback settings are
    the two distinct values 0 and 1, and the binary was pristine.  A pair
    that fails any of them can still be diffed -- it just cannot
    attribute the diff to feedback, which is the whole claim.
    """
    provenance_path = Path(provenance_path)
    raw = provenance_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    try:
        record = json.loads(raw)
    except ValueError as exc:
        raise ReferencePairError(
            f"{provenance_path} is not valid JSON: {exc}") from exc

    experiment = _require(record, "experiment", "provenance")
    build = _require(record, "build", "provenance")
    inputs = _require(record, "inputs", "provenance")
    execution = _require(record, "execution", "provenance")
    runs = _require(execution, "runs", "provenance.execution")
    if len(runs) != 2:
        raise ReferencePairError(
            f"a feedback A/B pair has exactly two runs, got {len(runs)}")

    by_feedback: dict[int, Mapping[str, Any]] = {}
    for name, entry in runs.items():
        value = int(_require(entry, "feedback", f"run {name!r}"))
        if value in by_feedback:
            raise ReferencePairError(
                f"both runs carry feedback = {value}: the pair has no "
                "difference to attribute")
        by_feedback[value] = entry
    if set(by_feedback) != {0, 1}:
        raise ReferencePairError(
            f"a feedback A/B pair is feedback 0 vs 1, got "
            f"{sorted(by_feedback)}")

    if not bool(_require(inputs, "identical", "provenance.inputs")):
        raise ReferencePairError(
            "the pair's inputs are not identical: a state difference "
            "measured across two variables is not evidence about either")
    if not bool(build.get("pristine", False)):
        raise ReferencePairError(
            "the reference binary is not recorded pristine, so a state "
            "difference cannot be attributed to stock WRF's operator")

    domains = _require(experiment, "domains", "provenance.experiment")
    if len(domains) != 2:
        raise ReferencePairError(
            f"the feedback tier scores one parent/nest pair, got "
            f"{len(domains)} domains")
    parent_key, nest_key = sorted(domains)
    parent, nest = domains[parent_key], domains[nest_key]
    ratio = int(_require(nest, "parent_grid_ratio", f"domain {nest_key!r}"))
    if ratio < 2:
        raise ReferencePairError(f"parent_grid_ratio must be >= 2, got {ratio}")

    return ReferencePairLayout(
        root=provenance_path.parent,
        run_a_dir=str(_require(by_feedback[0], "dir", "run A")),
        run_b_dir=str(_require(by_feedback[1], "dir", "run B")),
        feedback_a=0,
        feedback_b=1,
        parent_grid_ratio=ratio,
        i_parent_start=int(_require(nest, "i_parent_start", "nest domain")),
        j_parent_start=int(_require(nest, "j_parent_start", "nest domain")),
        nest_e_we=int(_require(nest, "e_we", "nest domain")),
        nest_e_sn=int(_require(nest, "e_sn", "nest domain")),
        parent_e_we=int(_require(parent, "e_we", "parent domain")),
        parent_e_sn=int(_require(parent, "e_sn", "parent domain")),
        only_difference=str(experiment.get("only_difference", "")),
        inputs_identical=True,
        build_pristine=True,
        build_commit=str(build.get("git_commit", "")),
        build_exe_sha256=str(build.get("wrf_exe_sha256", "")),
        provenance_sha256=digest,
    )


def operator_class(entry: Mapping[str, Any]) -> str:
    """Route one operator-map entry to its class by operator + staggering.

    Routing is on ``(op, xstag, ystag)`` and never on the field's name.
    Two of the most-inspected fields in any nesting comparison -- the
    10 m wind diagnostics -- live at mass points and therefore take the
    cell-average operator despite being winds, while the prognostic
    winds take the face average; a name-based router gets exactly those
    wrong, and gets them wrong quietly.
    """
    op = str(_require(entry, "op", "operator-map entry"))
    xstag = bool(entry.get("xstag", False))
    ystag = bool(entry.get("ystag", False))
    if op == "copy_fcnm":
        return "masked"
    if op == "copy_fcni":
        return "integer"
    if op == "mark_domain":
        return "mark_domain"
    if op == "copy_fcn":
        if xstag and ystag:
            raise ReferencePairError(
                "no WRF feedback operator handles a doubly-staggered field")
        if xstag:
            return "u_face"
        if ystag:
            return "v_face"
        return "mass"
    raise ReferencePairError(f"unknown feedback operator {op!r}")


def load_operator_map(path: Path) -> tuple[dict[str, str], str]:
    """``(field -> operator class, sha256 of the map's exact bytes)``."""
    path = Path(path)
    raw = path.read_bytes()
    record = json.loads(raw)
    if not record:
        raise ReferencePairError(f"{path} carries no fields")
    return ({name: operator_class(entry) for name, entry in record.items()},
            hashlib.sha256(raw).hexdigest())


def operator_class_census(classes: Mapping[str, str]) -> dict[str, int]:
    """Field count per operator class, every known class present."""
    census = {name: 0 for name in sorted(OPERATOR_CLASSES)}
    for value in classes.values():
        if value not in census:
            raise ReferencePairError(f"unknown operator class {value!r}")
        census[value] += 1
    return census


def nest_footprint(layout: ReferencePairLayout) -> Zone:
    """The parent cells the nest covers, 0-based inclusive."""
    i0 = layout.i_parent_start - 1
    j0 = layout.j_parent_start - 1
    ni = (layout.nest_e_we - 1) // layout.parent_grid_ratio
    nj = (layout.nest_e_sn - 1) // layout.parent_grid_ratio
    return Zone(i0=i0, i1=i0 + ni - 1, j0=j0, j1=j0 + nj - 1)


def feedback_zone(layout: ReferencePairLayout, class_name: str) -> Zone:
    """The parent cells WRF's feedback pass writes, for one class.

    The unstaggered zone is the nest footprint shrunk by ``spec_zone``
    on every side: those outermost parent cells supplied the nest's own
    lateral boundary condition, so feeding them back would close a loop
    on the boundary WRF just imposed.  A staggered class covers the
    faces bounding those cells, which is one more point along its own
    stagger and identical to the unstaggered zone across it.
    """
    if class_name not in OPERATOR_CLASSES:
        raise ReferencePairError(f"unknown operator class {class_name!r}")
    spec = FEEDBACK_SPEC_ZONE
    i0 = layout.i_parent_start - 1
    j0 = layout.j_parent_start - 1
    ni = (layout.nest_e_we - 2) // layout.parent_grid_ratio
    nj = (layout.nest_e_sn - 2) // layout.parent_grid_ratio
    zone = Zone(i0=i0 + spec, i1=i0 + ni - spec,
                j0=j0 + spec, j1=j0 + nj - spec)
    if class_name in _UNSTAGGERED_CLASSES:
        return zone
    if class_name == "u_face":
        return Zone(i0=zone.i0, i1=zone.i1 + 1, j0=zone.j0, j1=zone.j1)
    return Zone(i0=zone.i0, i1=zone.i1, j0=zone.j0, j1=zone.j1 + 1)


def full_zone(field: np.ndarray) -> Zone:
    ny, nx = field.shape[-2], field.shape[-1]
    return Zone(i0=0, i1=nx - 1, j0=0, j1=ny - 1)


def score_difference(field_a: np.ndarray, field_b: np.ndarray,
                     zone: Zone) -> dict[str, float]:
    """B-minus-A difference metrics over ``zone``.

    Arrays carry their horizontal dimensions last, so the same call
    scores a 2-D surface field and a 3-D column field without the caller
    reshaping either.  ``frac_nonzero`` is over the zone's elements: on
    a bit-reproducible pair the noise floor is exactly zero, so a
    non-zero fraction is signal and not roundoff, and the pair's own
    null control (B scored against A's own file) reads 0.0 everywhere.
    """
    a = np.asarray(field_a, dtype=np.float64)
    b = np.asarray(field_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ReferencePairError(
            f"pair fields disagree in shape: {a.shape} vs {b.shape}")
    if a.ndim < 2:
        raise ReferencePairError(
            f"a scored field carries (.., y, x); got shape {a.shape}")
    mask = zone.mask(a.shape[-2], a.shape[-1])
    delta = np.abs(b - a)[..., mask]
    if delta.size == 0:
        raise ReferencePairError("zone selects no cells")
    return {
        "max_abs": float(np.max(delta)),
        "mean_abs": float(np.mean(delta)),
        "rms": float(np.sqrt(np.mean(delta * delta))),
        "frac_nonzero": float(np.count_nonzero(delta) / delta.size),
    }


def score_field_regions(field_a: np.ndarray, field_b: np.ndarray, *,
                        layout: ReferencePairLayout,
                        class_name: str) -> dict[str, Any]:
    """One field scored over the feedback zone, footprint and full parent.

    The three regions answer three different questions.  ``fb_zone`` is
    where WRF writes, so it carries the direct signal.  ``footprint``
    adds the one-cell rim the feedback pass deliberately skips, which
    should show only advected difference.  ``full`` measures how far the
    perturbation has spread across the parent, which is the part an
    implementation is most likely to get subtly wrong.
    """
    regions = {
        "fb_zone": feedback_zone(layout, class_name),
        "footprint": nest_footprint(layout),
        "full": full_zone(np.asarray(field_a)),
    }
    scored: dict[str, Any] = {"class": class_name}
    for name, zone in regions.items():
        scored[name] = score_difference(field_a, field_b, zone)
    return scored


def score_frame(fields_a: Mapping[str, np.ndarray],
                fields_b: Mapping[str, np.ndarray], *,
                layout: ReferencePairLayout,
                classes: Mapping[str, str]) -> dict[str, Any]:
    """Score every field of one frame pair, routing each by its class.

    A field with no entry in the operator map raises: an unrouted field
    would otherwise be scored under a default class and reported against
    the wrong parent region.
    """
    if set(fields_a) != set(fields_b):
        raise ReferencePairError(
            "the two runs' frames carry different field sets: "
            f"{sorted(set(fields_a) ^ set(fields_b))}")
    out: dict[str, Any] = {}
    for name in sorted(fields_a):
        if name not in classes:
            raise ReferencePairError(
                f"field {name!r} has no operator-map entry, so the parent "
                "region it is written over is unknown")
        out[name] = score_field_regions(
            fields_a[name], fields_b[name],
            layout=layout, class_name=classes[name])
    return out


#: The nest-coupling evidence stock WRF cannot produce, recorded with
#: what was looked for rather than left as an unexplained gap.
TENDENCY_FORM_UNAVAILABLE: dict[str, str] = {
    "item": "parent<->child coupling tendency fields",
    "reason": (
        "stock WRF emits no nest-coupling field: the coupling is an "
        "in-memory operation, no auxhist stream carries it, and a "
        "feedback=1 run's history carries the identical variable set to a "
        "feedback=0 run"),
    "evidence": (
        "reference-bank inventory of the banked run directories: no "
        "auxhist stream in any of them, and the feedback=1 run's history "
        "variable set equals the feedback=0 run's"),
    "consequence": (
        "the vs-WRF nest tier is scored as an A/B state difference, which "
        "is the strongest evidence form stock WRF permits"),
}


def build_feedback_ab_tier(*, layout: ReferencePairLayout,
                           operator_map_sha256: str,
                           class_census: Mapping[str, int],
                           frames: Sequence[Mapping[str, Any]],
                           null_control: Mapping[str, Any],
                           ) -> dict[str, Any]:
    """The measured vs-WRF nest tier for a conservation receipt.

    ``null_control`` is the pair's own zero check -- run B scored
    against run A's own file -- and is REQUIRED: a scorer that silently
    returned zeros would otherwise be indistinguishable from a run with
    no feedback signal, and the whole tier would read as a pass.
    """
    if not frames:
        raise ReferencePairError(
            "a measured feedback tier carries at least one scored frame")
    if not operator_map_sha256:
        raise ReferencePairError(
            "the tier binds the operator map it routed fields by")
    if not null_control:
        raise ReferencePairError(
            "a measured feedback tier carries its null control: a scorer "
            "returning zeros must be distinguishable from a null result")
    census = dict(class_census)
    missing = sorted(OPERATOR_CLASSES - set(census))
    if missing:
        raise ReferencePairError(
            f"the operator census is missing class(es) {missing}")
    return {
        "schema": SCHEMA_ID,
        "status": "measured",
        "decision": "D-8",
        "method": (
            "A/B state difference between two stock-WRF runs whose only "
            "difference is domains.feedback (0 vs 1), scored per field "
            "against the run's own feedback operator map"),
        "reference": layout.as_dict(),
        "operator_map_sha256": operator_map_sha256,
        "operator_class_census": census,
        "feedback_spec_zone": FEEDBACK_SPEC_ZONE,
        "zones": {name: feedback_zone(layout, name).as_dict()
                  for name in sorted(OPERATOR_CLASSES)},
        "footprint": nest_footprint(layout).as_dict(),
        "metrics": list(DIFFERENCE_METRICS),
        "frames": [dict(frame) for frame in frames],
        "null_control": dict(null_control),
        "not_available": [dict(TENDENCY_FORM_UNAVAILABLE)],
    }
