"""``NT_CALL_ORDER`` is checked against the source, not only against itself.

WHY THIS EXISTS (review). ``check_order`` compares OBSERVED launches
against the DECLARED sequence, which catches a launcher that runs the
stages wrongly. It cannot catch a *declaration* that is wrong, because the
declaration is the reference. That is the same structure as every
hand-maintained list this port has been bitten by, and three members were
missing from work counts before anyone noticed: the assembler,
``cu_ntiedtke_post_run``, and ``cu_ntiedtke_run``'s post-conversion. All
three surfaced by walking the sequence, none by reading the list.

TWO INDEPENDENT CROSS-CHECKS, from artifacts that already exist:

1. **Source order.** ``cumastrn`` and ``cu_ntiedtke_run`` are straight-line
   code, so source order IS call order. The ownership manifest maps every
   live line to the stage that performs it, so a stage owning lines
   strictly before another's must precede it.

2. **Dataflow.** No stage may read an array before some stage has written
   it, unless that array is a declared input or seed. A missing stage shows
   up here as its consumer's inputs having no producer -- which is exactly
   how the post-conversion was found, by hand, before this test existed.

The two fail independently: (1) catches a reordering, (2) catches an
omission.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tests.test_ntiedtke_cumastrn_ownership import OWNERSHIP, RUN_OWNERSHIP
from gpuwm.core.ntiedtke import NT_SEEDS
from tests.test_ntiedtke_launch_geometry import NT_CALL_ORDER

_CU = Path(__file__).resolve().parents[1] / "gpuwm" / "core" / "kernels" \
    / "ntiedtke.cu"

#: ``cumastrn`` is called from ``cu_ntiedtke_run`` at :283-291, so the run
#: body splits into a before and an after. Four phases, in execution order.
_BEFORE, _CUMASTRN, _AFTER, _POST_RUN = 0, 1, 2, 3

#: ``cu_ntiedtke_post_run`` is in a different file and runs last.
_POST_RUN_STAGE = "ntiedtke_post_run"

#: Stages whose position the manifest cannot discriminate, with the reason
#: and the evidence that settles it. NOT "these are fine" -- each names
#: what orders it instead.
_TIED = {
    ("ntiedtke_prep", "ntiedtke_convert"):
        "both own cu_ntiedtke_run:230-278; ordered by dataflow instead -- "
        "convert reads tf/qvf/uf/vf/omg/ghtl/ghti/prsl/qvftenz/thftenz, "
        "all of which prep writes",
}


def _stages_of(owner):
    return re.findall(r"ntiedtke_\w+", owner or "")


def _positions():
    """stage -> (phase, first line it owns)."""
    pos = {}
    for a, _, who in OWNERSHIP:
        for st in _stages_of(who):
            pos.setdefault(st, (_CUMASTRN, a))
            if pos[st][1] > a:
                pos[st] = (_CUMASTRN, a)
    for a, _, who in RUN_OWNERSHIP:
        phase = _BEFORE if a < 283 else _AFTER
        for st in _stages_of(who):
            cur = pos.get(st)
            if cur is None or (phase, a) < cur:
                pos[st] = (phase, a)
    pos[_POST_RUN_STAGE] = (_POST_RUN, 502)
    return pos


def test_the_manifests_name_stages_at_all():
    """A gate that scans nothing passes vacuously."""
    pos = _positions()
    assert len(pos) >= 18, f"only {len(pos)} stages found in the manifests"
    for expected in ("ntiedtke_prep", "ntiedtke_cuinin", "ntiedtke_cuascn",
                     "ntiedtke_post_conversion", "ntiedtke_post_run"):
        assert expected in pos, f"{expected} owns no range in either manifest"


def test_every_ordered_stage_owns_a_range_and_every_owner_is_ordered():
    """A stage with a position but no range, or a range but no position.

    Either is the shape all three missing members had.
    """
    pos = set(_positions())
    ordered = set(NT_CALL_ORDER)
    # ntiedtke_midlevel is deliberately outside the order: it exists to
    # grade cubasmcn and cuentrn standalone, and in the real sequence they
    # run INSIDE cuascn. Asserted in test_ntiedtke_launch_geometry.
    ordered_only = ordered - pos
    owned_only = pos - ordered - {"ntiedtke_midlevel"}
    assert not ordered_only, (
        f"stages in NT_CALL_ORDER owning no line of either routine: "
        f"{sorted(ordered_only)}. Give them a range or explain the absence.")
    assert not owned_only, (
        f"stages owning lines but absent from NT_CALL_ORDER: "
        f"{sorted(owned_only)}. That is the shape the post-conversion had "
        f"for the whole port.")


def test_the_declared_order_agrees_with_source_order():
    """Straight-line code: source order is call order."""
    pos = _positions()
    seq = [(st, pos[st]) for st in NT_CALL_ORDER]
    tied = {frozenset(k) for k in _TIED}
    bad = []
    for (a, pa), (b, pb) in zip(seq, seq[1:]):
        if pa < pb:
            continue
        if pa == pb and frozenset((a, b)) in tied:
            continue
        bad.append(f"{a}{pa} is declared before {b}{pb}")
    assert not bad, (
        "NT_CALL_ORDER disagrees with the ownership manifests:\n  "
        + "\n  ".join(bad))


def test_every_tie_names_what_orders_it_instead():
    """A tie with no reason is a pair nobody has thought about."""
    pos = _positions()
    for pair, why in _TIED.items():
        assert pos[pair[0]] == pos[pair[1]], (
            f"{pair} is no longer tied in the manifest -- source order now "
            f"settles it and this entry is stale")
        assert len(why) > 40 and "dataflow" in why or "measured" in why, why


# ---------------------------------------------------------------------------
# 2. dataflow
# ---------------------------------------------------------------------------

#: The seed table lives in gpuwm.core.ntiedtke, because the
#: ASSEMBLER is what acts on it and this file only checks it. A
#: second copy here would be the failure this port has paid for
#: four times over -- and this table has the widest blast radius
#: of anything in the port right now (review).
SEEDS = NT_SEEDS


def _kernel_args():
    src = _CU.read_text(encoding="utf-8")
    macros = "|".join(sorted(set(re.findall(
        r"#define\s+(\w+)\(a,\s*k\)\s*\(a\)\[", src))))
    assert macros, "no level-index macro found in ntiedtke.cu"
    out = {}
    for m in re.finditer(r'extern\s+"C"\s+__global__\s+void\s+(\w+)\(', src):
        name = m.group(1)
        i = m.end() - 1
        depth, j = 0, i
        while True:
            depth += 1 if src[j] == "(" else -1 if src[j] == ")" else 0
            if depth == 0 and src[j] == ")":
                break
            j += 1
        params = re.sub(r"/\*.*?\*/", "", src[i + 1:j], flags=re.S)
        params = re.sub(r"//[^\n]*", "", params)
        args = []
        for p in (x.strip() for x in params.split(",")):
            if not p or "*" not in p:
                continue
            nm = p.split()[-1].lstrip("*")
            args.append((nm, p.startswith("const")))
        out[name] = args
    return out


_TAIL = {"geom_report", "order_report", "ticket"}
_SCRATCH = {"scr", "scr_i"}


def test_the_argument_scan_sees_the_kernels():
    """Vacuity guard for the parser this file's dataflow rests on."""
    args = _kernel_args()
    assert len(args) >= 21, f"only {len(args)} kernels parsed"
    assert len(args["ntiedtke_closure"]) >= 40
    names = {n for a in args.values() for n, _ in a}
    for expected in ("ptenu", "zprecc", "ztenu", "pqsen", "rthcuten"):
        assert expected in names, f"the parser no longer sees {expected}"


def test_no_stage_reads_an_array_before_one_writes_it():
    """The omission check, and the one that found the post-conversion.

    A stage whose inputs have no producer means either a missing stage or
    an undeclared seed. Both are the same failure from the assembler's
    point of view: it will hand that stage an array of zeros and every
    number downstream will be finite, plausible and wrong.
    """
    args = _kernel_args()
    written, unexplained = set(), []
    for stage in NT_CALL_ORDER:
        for name, is_const in args[stage]:
            if name in _TAIL or name in _SCRATCH:
                continue
            if is_const and name not in written and name not in SEEDS:
                unexplained.append(f"{stage} reads {name}")
        for name, is_const in args[stage]:
            if not is_const and name not in _TAIL:
                written.add(name)
    assert not unexplained, (
        "arrays read before any stage writes them and not declared as "
        f"seeds:\n  " + "\n  ".join(unexplained) +
        "\n\nEither a stage is missing from NT_CALL_ORDER, or the array is "
        "a driver input, an alias or an assembler copy -- say which in "
        "SEEDS.")


def test_every_declared_seed_is_actually_read_before_written():
    """The converse. A stale seed hides the day it stops being one."""
    args = _kernel_args()
    written, needed = set(), set()
    for stage in NT_CALL_ORDER:
        for name, is_const in args[stage]:
            if is_const and name not in written:
                needed.add(name)
        for name, is_const in args[stage]:
            if not is_const and name not in _TAIL:
                written.add(name)
    stale = sorted(set(SEEDS) - needed)
    assert not stale, (
        f"declared seeds that no stage reads before a stage writes them: "
        f"{stale}. A stage now produces them, so the assembler no longer "
        f"has to.")


def test_every_alias_target_is_produced_by_a_stage():
    """The hole the first version of this file had, found by falsifying it.

    Dropping ``ntiedtke_post_conversion`` from NT_CALL_ORDER did NOT fail
    the read-before-written test, because post_run's inputs are all
    declared seeds -- including ``rn``, declared "alias of zprecc". The
    declaration said where the value comes from and then stopped anyone
    checking that anything still puts it there. **A seed that names a
    producer must be checked against that producer**, or it is a licence
    to delete the producer.

    Driver seeds are exempt by definition: nothing inside the walk writes
    them, which is what makes them driver seeds.
    """
    args = _kernel_args()
    written = {n for st in NT_CALL_ORDER for n, c in args[st]
               if not c and n not in _TAIL}
    orphaned = []
    for name, why in SEEDS.items():
        m = re.match(r"(?:alias|copy) of (\w+)", why)
        if not m:
            continue                       # driver, or assembler-zeroed
        target = m.group(1)
        # Resolve the chain: an alias may name another alias, and only the
        # end of the chain has to be produced.
        seen = {name}
        while target in SEEDS and target not in seen:
            seen.add(target)
            nxt = re.match(r"(?:alias|copy) of (\w+)", SEEDS[target])
            if not nxt:
                target = None              # ends at a driver or a zeroing
                break
            target = nxt.group(1)
        if target is not None and target not in written:
            orphaned.append(f"{name} is {why}, but nothing writes "
                            f"{target}")
    assert not orphaned, (
        "seeds whose named source nothing produces:\n  "
        + "\n  ".join(orphaned) +
        "\n\nEither the producing stage is missing from NT_CALL_ORDER, or "
        "the seed is really a driver input and should say so.")


def test_every_alias_and_copy_names_its_source():
    """The half that makes the check above possible.

    "alias of something" with no name is a sentence, not a reference.
    """
    vague = [f"{n}: {why}" for n, why in SEEDS.items()
             if why.startswith(("alias", "copy"))
             and not re.match(r"(?:alias|copy) of \w+", why)]
    assert not vague, vague


# ---------------------------------------------------------------------------
# 3. the gates, measured against the failure they exist for
# ---------------------------------------------------------------------------

#: Gates that must between them notice a stage going missing.
_OMISSION_GATES = (
    "test_no_stage_reads_an_array_before_one_writes_it",
    "test_every_alias_target_is_produced_by_a_stage",
    "test_every_ordered_stage_owns_a_range_and_every_owner_is_ordered",
)


@pytest.mark.parametrize("dropped", NT_CALL_ORDER)
def test_removing_any_stage_is_caught_by_at_least_one_gate(dropped,
                                                           monkeypatch):
    """A MUTATION TEST, because "the gate works" is a measurable claim.

    Three members of this port were missing from a hand-declared list and
    none was found by reading it. Writing gates against that is worth
    exactly what the gates actually catch, so this removes each stage in
    turn and requires some gate to notice.

    MEASURED, and the first draft of this paragraph guessed the numbers
    and got two of three wrong -- twelve for ten, and "six stages" beside
    a list of seven. Counted: the manifest gate catches all 20, the
    dataflow gate 10, the alias gate 5.

    SEVEN stages are caught by the manifest ALONE -- cloud_depth,
    cuddrafn, cuflxn, ke_dissipation, momentum_profile, momentum_rescale
    and post_run -- because their outputs are also written by an earlier
    stage, so their absence leaves no unwritten read. That is a real blind
    spot in the dataflow gate, stated rather than papered over: for those
    seven the manifest is the only thing standing between a silent
    omission and a wrong forecast.
    """
    import tests.test_ntiedtke_call_order_vs_source as mod

    monkeypatch.setattr(
        mod, "NT_CALL_ORDER",
        tuple(s for s in NT_CALL_ORDER if s != dropped))
    caught = []
    for gate in _OMISSION_GATES:
        try:
            getattr(mod, gate)()
        except AssertionError:
            caught.append(gate)
    assert caught, (
        f"removing {dropped} from NT_CALL_ORDER is caught by NONE of "
        f"{_OMISSION_GATES}. The declaration could lose it silently, which "
        f"is how the post-conversion went missing for the whole port.")


def test_the_gate_suite_is_not_carried_by_one_gate_alone():
    """If only the manifest ever fires, the other two are decoration.

    Not a claim that all three are needed for every stage -- measured, they
    are not -- but that each earns its place on some stage.
    """
    import tests.test_ntiedtke_call_order_vs_source as mod
    fired = {g: 0 for g in _OMISSION_GATES}
    original = mod.NT_CALL_ORDER
    try:
        for dropped in original:
            mod.NT_CALL_ORDER = tuple(s for s in original if s != dropped)
            for gate in _OMISSION_GATES:
                try:
                    getattr(mod, gate)()
                except AssertionError:
                    fired[gate] += 1
    finally:
        mod.NT_CALL_ORDER = original
    idle = [g for g, n in fired.items() if n == 0]
    assert not idle, f"gates that catch nothing at all: {idle}"
    assert fired[_OMISSION_GATES[2]] == len(original), (
        f"the manifest gate used to catch all {len(original)} and now "
        f"catches {fired[_OMISSION_GATES[2]]}")


def test_the_zeroed_seeds_really_are_zero_in_the_oracle():
    """A LIVE ASSERTION, not a note (review).

    ``ztenu``/``ztenv`` are copied from ``pvom``/``pvol`` at :1019-1024,
    which is BEFORE cududvn writes them, so what is copied is the array
    zeroed at :258-259. Measured: identically zero on all 5,292 fixture
    rows, which is what confirms nothing writes pvom/pvol between :259 and
    :1019.

    The assembler still COPIES rather than zero-filling -- the shortcut is
    right today and it is the shape that has been wrong eight times. This
    exists so that the day pvom is non-zero at :1019 someone is TOLD,
    rather than the copy quietly starting to matter and the shortcut, had
    it been taken, quietly starting to be wrong.
    """
    import numpy as np

    from gpuwm.verify.ntiedtke_oracle import load_csv, word

    rows = list(load_csv("nt-kedis-in-levels.csv"))
    assert len(rows) >= 5000, f"only {len(rows)} rows; the capture shrank"
    for field in ("ztenu", "ztenv"):
        a = np.array([word(r[field]) for r in rows], dtype=np.float32)
        nz = int(np.count_nonzero(a.view(np.uint32)))
        assert nz == 0, (
            f"{field} is non-zero on {nz} of {a.size} rows. Something now "
            f"writes pvom/pvol between :259 and :1019, so the :1019-1024 "
            f"copy carries a real value -- check that the assembler still "
            f"copies rather than zero-fills, and that SEEDS still describes "
            f"it correctly.")


@pytest.mark.parametrize("kind", ["driver", "alias", "copy", "zeroed"])
def test_each_seed_class_is_populated(kind):
    """Three classes, and each must have members.

    A class that empties silently means its members were reclassified
    without anyone noticing which -- and 'driver input' and 'alias' have
    very different consequences for the assembler.
    """
    got = [n for n, why in SEEDS.items() if why.startswith(kind)]
    assert got, f"no seed is classified as {kind!r}"
