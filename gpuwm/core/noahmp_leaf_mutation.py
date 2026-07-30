"""Reusable argument-mutation harness for the Noah-MP leaf oracles.

Bitwise parity against a fixture proves the port reproduces the arithmetic of
the regimes the fixture happens to contain.  It does **not** prove the port
reads the arguments it claims to read.  A sibling lane in this project reached
``max_ulp 0`` on 29 columns and then found that 13 of 14 mutants -- each
dropping the use of one argument -- still reproduced the pinned CSV bit for
bit.  The same defect had already been found once, in this project's
condensation fixture, where ``qsq`` turned out entirely undetectable.

So this module makes the mutation study a first-class, reusable gate rather
than a per-leaf afterthought.  For one leaf it generates, for every input slot
(FP32 and integer topology alike):

* the ``zero`` mutant -- the slot is forced to 0.0 in every case, modelling a
  dropped term;
* one ``freeze:<case>`` mutant per case -- the slot is forced to the constant
  that case gives it, in every case, modelling a port that ignores the
  per-column argument and substitutes a constant.

The acceptance rule is:

* a slot the pinned option identity **consumes** must have *every* mutant
  killed.  Equivalently: no constant drawn from the fixture, and not zero,
  can be substituted for that argument and still reproduce the fixture.
* a slot declared **inert** must have *every* mutant survive.  Inertness is a
  claim about the pinned WRF source (a dead branch, or an argument the body
  never references) and each entry carries the WRF line numbers that justify
  it.
* a slot declared **partial** must have exactly the enumerated mutants survive
  and no others.  This is for arguments whose effect is not fully observable in
  the routine's *defined* outputs -- a loop bound whose out-of-range slots WRF
  leaves INTENT(OUT)-undefined, for instance.  The declaration names the exact
  surviving operators, so a new blind spot cannot hide inside the category.

Both declaration tables live beside the fixture in
``tools/noahmp_wrf461_oracle/validate_leaf_oracle.py``.

A surviving mutant on a consumed slot is a *fixture* defect, not a port defect:
it means the fixture cannot detect a port that ignores that argument.  Fix the
fixture (add a case whose value for that slot differs, or whose topology makes
the slot readable, or which sits on the other side of a threshold test), do not
weaken the rule.

Run it directly for one leaf::

    python -m gpuwm.core.noahmp_leaf_mutation thermoprop
    python -m gpuwm.core.noahmp_leaf_mutation            # every ported leaf
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
import csv
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import struct
import sys

import numpy as np

from gpuwm.core.noahmp_leaves import LEAF_EVALUATORS


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURE = (
    REPO_ROOT / "gpuwm" / "data" / "noahmp" / "oracle" / "noahmp-leaves.csv"
)
_VALIDATOR = (
    REPO_ROOT / "tools" / "noahmp_wrf461_oracle" / "validate_leaf_oracle.py"
)


def _load_declarations():
    """Read the inert-argument declarations that live beside the fixture.

    The declarations are kept in the build-time validator so the Fortran
    harness, the validator and this module share one source of truth; the
    validator deliberately depends on nothing but the standard library so it
    can run inside ``build_leaves.sh`` under a bare WSL python3.
    """
    spec = importlib.util.spec_from_file_location(
        "_noahmp_leaf_declarations", _VALIDATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    inert = {leaf: dict(entry["inert"])
             for leaf, entry in module.LEAVES.items()}
    partial = {leaf: dict(entry.get("partial", {}))
               for leaf, entry in module.LEAVES.items()}
    return inert, partial


INERT_ARGUMENTS, PARTIAL_ARGUMENTS = _load_declarations()


@dataclass(frozen=True)
class LeafFixture:
    """One leaf's pinned cases, in the flat slot layout of run_leaves.F90."""

    leaf: str
    cases: tuple[str, ...]
    int_names: tuple[str, ...]
    ints: np.ndarray            # (ncase, n_int) int64
    in_names: tuple[str, ...]
    in_index: tuple[int, ...]
    x: np.ndarray               # (ncase, n_in) float32
    out_names: tuple[str, ...]
    out_index: tuple[int, ...]
    out_bits: np.ndarray        # (ncase, n_out) uint32
    live: np.ndarray            # (ncase, n_out) bool

    @property
    def ncase(self) -> int:
        return len(self.cases)


@dataclass(frozen=True)
class Mutant:
    """One argument mutation and whether the fixture killed it."""

    leaf: str
    role: str          # "in" or "int"
    slot: int          # 1-based, as in the fixture
    name: str
    index: int
    operator: str      # "zero" or "freeze:<case>"
    killed: bool
    killed_by: tuple[str, ...]

    @property
    def argument(self) -> tuple[str, int]:
        return (self.name, self.index)


def _bits_of(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).view(np.uint32)


def load_leaf_fixture(leaf: str, path: Path | None = None) -> LeafFixture:
    """Load one leaf's pinned cases from the oracle CSV."""
    path = DEFAULT_FIXTURE if path is None else Path(path)
    with path.open(newline="", encoding="ascii") as stream:
        rows = [row for row in csv.DictReader(stream) if row["leaf"] == leaf]
    if not rows:
        raise KeyError(f"{path} has no rows for leaf {leaf!r}")

    order: list[str] = []
    packed: dict[str, dict[str, dict[int, dict]]] = {}
    for row in rows:
        case = row["case"]
        if case not in packed:
            packed[case] = {"int": {}, "in": {}, "out": {}}
            order.append(case)
        packed[case][row["role"]][int(row["slot"])] = row

    first = packed[order[0]]
    int_slots = sorted(first["int"])
    in_slots = sorted(first["in"])
    out_slots = sorted(first["out"])

    def bits(case: str, role: str, slot: int) -> int:
        return int(packed[case][role][slot]["bits"], 16)

    ints = np.array(
        [[int(float(packed[c]["int"][s]["value"])) for s in int_slots]
         for c in order], dtype=np.int64).reshape(len(order), len(int_slots))
    x = np.array(
        [[struct.unpack("<f", struct.pack("<I", bits(c, "in", s)))[0]
          for s in in_slots] for c in order], dtype=np.float32)
    out_bits = np.array(
        [[bits(c, "out", s) for s in out_slots] for c in order],
        dtype=np.uint32)
    live = np.array(
        [[packed[c]["out"][s]["live"] == "1" for s in out_slots]
         for c in order], dtype=bool)

    return LeafFixture(
        leaf=leaf,
        cases=tuple(order),
        int_names=tuple(first["int"][s]["name"] for s in int_slots),
        ints=ints,
        in_names=tuple(first["in"][s]["name"] for s in in_slots),
        in_index=tuple(int(first["in"][s]["index"]) for s in in_slots),
        x=x,
        out_names=tuple(first["out"][s]["name"] for s in out_slots),
        out_index=tuple(int(first["out"][s]["index"]) for s in out_slots),
        out_bits=out_bits,
        live=live,
    )


def _mismatched_cases(fixture: LeafFixture,
                      evaluator: Callable[[np.ndarray, np.ndarray], np.ndarray],
                      x: np.ndarray, ints: np.ndarray) -> tuple[str, ...]:
    """Case names where the evaluator disagrees with the pinned live outputs."""
    bad: list[str] = []
    for case_index, case in enumerate(fixture.cases):
        try:
            with np.errstate(all="ignore"):
                got = evaluator(x[case_index], ints[case_index])
        except Exception:
            # A mutant that cannot even run is killed.
            bad.append(case)
            continue
        got_bits = _bits_of(np.asarray(got, dtype=np.float32))
        if got_bits.shape != fixture.out_bits[case_index].shape:
            bad.append(case)
            continue
        selected = fixture.live[case_index]
        want = np.where(selected, fixture.out_bits[case_index],
                        np.uint32(0))
        have = np.where(selected, got_bits, np.uint32(0))
        if not np.array_equal(want, have):
            bad.append(case)
    return tuple(bad)


def baseline_mismatches(
        fixture: LeafFixture,
        evaluator: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
) -> tuple[str, ...]:
    """Cases where the unmutated port already disagrees with the fixture."""
    evaluator = evaluator or LEAF_EVALUATORS[fixture.leaf]
    return _mismatched_cases(fixture, evaluator, fixture.x, fixture.ints)


def run_mutation_study(
        leaf: str,
        fixture: LeafFixture | None = None,
        evaluator: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
) -> tuple[Mutant, ...]:
    """Generate and score every argument mutant of one leaf."""
    fixture = fixture or load_leaf_fixture(leaf)
    evaluator = evaluator or LEAF_EVALUATORS[leaf]

    baseline = baseline_mismatches(fixture, evaluator)
    if baseline:
        raise AssertionError(
            f"{leaf}: the unmutated port already fails the fixture on "
            f"{list(baseline)}; fix parity before reading a mutation study")

    mutants: list[Mutant] = []
    groups = (
        ("in", fixture.in_names, fixture.in_index, fixture.x),
        ("int", fixture.int_names,
         tuple(0 for _ in fixture.int_names), fixture.ints),
    )
    for role, names, indices, table in groups:
        for column, (name, index) in enumerate(zip(names, indices)):
            operators: list[tuple[str, object]] = [("zero", 0)]
            operators += [(f"freeze:{case}", table[case_index, column])
                          for case_index, case in enumerate(fixture.cases)]
            for operator, value in operators:
                x = fixture.x.copy()
                ints = fixture.ints.copy()
                if role == "in":
                    x[:, column] = np.float32(value)
                else:
                    ints[:, column] = np.int64(value)
                bad = _mismatched_cases(fixture, evaluator, x, ints)
                mutants.append(Mutant(
                    leaf=leaf, role=role, slot=column + 1, name=name,
                    index=index, operator=operator, killed=bool(bad),
                    killed_by=bad))
    return tuple(mutants)


@dataclass(frozen=True)
class MutationVerdict:
    """Per-argument outcome of a mutation study."""

    leaf: str
    role: str
    slot: int
    name: str
    index: int
    declared_inert: bool
    justification: str
    n_mutants: int
    survivors: tuple[str, ...]
    expected_survivors: tuple[str, ...] | None = None

    @property
    def declared_partial(self) -> bool:
        return self.expected_survivors is not None

    @property
    def ok(self) -> bool:
        if self.declared_inert:
            return len(self.survivors) == self.n_mutants
        if self.expected_survivors is not None:
            return (set(self.survivors) == set(self.expected_survivors)
                    and len(self.survivors) < self.n_mutants)
        return not self.survivors


def summarise(mutants: Sequence[Mutant],
              inert: Mapping[tuple[str, int], str] | None = None,
              partial: Mapping[tuple[str, int], tuple] | None = None,
              ) -> tuple[MutationVerdict, ...]:
    """Collapse a mutation study to one verdict per argument."""
    if not mutants:
        return ()
    leaf = mutants[0].leaf
    inert = INERT_ARGUMENTS.get(leaf, {}) if inert is None else inert
    partial = PARTIAL_ARGUMENTS.get(leaf, {}) if partial is None else partial
    grouped: dict[tuple[str, int], list[Mutant]] = defaultdict(list)
    for mutant in mutants:
        grouped[(mutant.role, mutant.slot)].append(mutant)
    verdicts = []
    for (role, slot), group in sorted(grouped.items()):
        head = group[0]
        expected = None
        justification = inert.get(head.argument, "")
        if head.argument in partial:
            justification, expected = partial[head.argument]
        verdicts.append(MutationVerdict(
            leaf=leaf, role=role, slot=slot, name=head.name,
            index=head.index, declared_inert=head.argument in inert,
            justification=justification, n_mutants=len(group),
            survivors=tuple(m.operator for m in group if not m.killed),
            expected_survivors=expected))
    return tuple(verdicts)


def failures(verdicts: Sequence[MutationVerdict]) -> tuple[str, ...]:
    """Human-readable failures, empty when a leaf is mutation-clean."""
    messages: list[str] = []
    for verdict in verdicts:
        if verdict.ok:
            continue
        label = f"{verdict.leaf}: {verdict.name}[{verdict.index}] " \
                f"({verdict.role} slot {verdict.slot})"
        if verdict.declared_inert:
            messages.append(
                f"{label} is declared inert ({verdict.justification}) but "
                f"{verdict.n_mutants - len(verdict.survivors)} of "
                f"{verdict.n_mutants} mutants were killed -- the argument is "
                f"consumed after all, so the declaration is wrong")
        elif verdict.declared_partial:
            messages.append(
                f"{label} is declared partially observable "
                f"({verdict.justification}) with expected survivors "
                f"{list(verdict.expected_survivors)}, but the study found "
                f"{list(verdict.survivors)}")
        else:
            messages.append(
                f"{label}: {len(verdict.survivors)} of {verdict.n_mutants} "
                f"mutants SURVIVED ({', '.join(verdict.survivors[:6])}"
                f"{', ...' if len(verdict.survivors) > 6 else ''}). The "
                f"fixture cannot detect a port that ignores this argument. "
                f"Add a case that gives it a different value, a topology that "
                f"reads it, or a value on the other side of its threshold.")
    return tuple(messages)


def main(argv: Sequence[str]) -> int:
    leaves = list(argv[1:]) or sorted(LEAF_EVALUATORS)
    status = 0
    for leaf in leaves:
        fixture = load_leaf_fixture(leaf)
        verdicts = summarise(run_mutation_study(leaf, fixture))
        bad = failures(verdicts)
        total = sum(v.n_mutants for v in verdicts)
        inert = [v for v in verdicts if v.declared_inert]
        part = [v for v in verdicts if v.declared_partial]
        print(f"{leaf:12s} {fixture.ncase:3d} cases  "
              f"{len(verdicts):3d} arguments  {total:5d} mutants  "
              f"{len(inert):2d} inert  {len(part):d} partial  "
              f"{'CLEAN' if not bad else 'FAIL'}")
        for message in bad:
            print(f"  {message}", file=sys.stderr)
            status = 1
    return status


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
