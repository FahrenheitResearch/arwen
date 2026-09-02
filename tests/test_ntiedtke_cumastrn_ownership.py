"""Which kernel owns which line of ``cumastrn``, and which lines nobody owns.

WHY THIS EXISTS. The fifth failure in this port was `cumastrn:566-568`
flipping `ktype` between `cuascn` and the closure — two points that look
adjacent, with a promotion rule sitting between them, invisible until a
capture file came back empty. That was fixed at the FIXTURE level by
capturing where each routine reads.

The identical blindness exists at the KERNEL level: a line of `cumastrn`
that falls between two stages is owned by nobody, and nothing looks for it.
There is no equivalent accident waiting to surface it. So the unowned set is
computed here rather than remembered (finding: review).

It also carries a forward obligation. Several class-2 excuses in
``test_ntiedtke_aliasing_audit.py`` are justified by a property of the
reference's CALL SITE — "cumastrn zeroes these four immediately before the
call". That property is only inherited by the port if the port reproduces
that call site. :func:`unowned_ranges` is what makes such an excuse carry a
visible debt until a kernel claims the range it depends on.
"""
from __future__ import annotations

import pytest

#: The executable body of cumastrn. :464 and above are declarations.
CUMASTRN_BODY = (465, 1085)

#: (first, last, owner). ``None`` means NO KERNEL OWNS THIS YET.
#:
#: A range is "owned" when a kernel performs that work, not when a kernel
#: merely consumes its result: the closure takes the POST-FLIP ktype as an
#: input, so :566-568 is unowned even though the closure depends on it.
OWNERSHIP: tuple[tuple[int, int, str | None], ...] = (
    (465, 471, "every kernel (constants; nt_init and the per-stage literals)"),
    (472, 484, "ntiedtke_cuinin"),
    (485, 497, "ntiedtke_cutypen"),
    (498, 542, "ntiedtke_mfub"),
    (543, 558, "ntiedtke_cuascn"),
    # ---- the gap that produced the fifth failure, NOW OWNED ------------
    # The ktype flip (:566-568) and the downdraft zeroing (:580-588) are
    # both in here, so claiming this range clears all four call-site debts.
    (559, 590, "ntiedtke_cloud_depth"),
    (591, 605, "ntiedtke_cudlfsn"),
    (606, 617, "ntiedtke_cuddrafn"),
    (618, 745, "ntiedtke_closure"),
    # 6.5/6.6/6.7.  The dead block at :786-802 is inside this range and is
    # skipped by the kernel; both its guards are .true. parameters.
    (746, 819, "ntiedtke_updraft_scale"),
    (820, 832, "ntiedtke_cuflxn"),
    # CORRECTED 2026-08-29.  This range does NOT produce zmfuus/zmfdus --
    # that is :996-1016, below.  :833-919 is the DOWNDRAFT stability
    # rescale (zmfs applied to pmfd/pmfds/pmfdq/pmfdde_rate/zdmfdp), the
    # entrainment-rate floors, the negative-humidity guards at the
    # downdraft top and near cloud top, and prsfc/pssfc.
    (833, 919, "ntiedtke_adjust"),
    (920, 926, "ntiedtke_cudtdqn"),
    # :927-995 is the momentum bookkeeping; :996-1016 is the MOMENTUM mass-
    # flux rescale that produces zmfuus/zmfdus -- and it is THOSE, not
    # pmfu/pmfd, that reach cududvn.  A cududvn fed the unscaled pair is
    # wrong on exactly the columns the rescale touched, silently.
    (927, 995, "ntiedtke_momentum_profile"),
    (996, 1016, "ntiedtke_momentum_rescale"),
    # :1019-1024 copies pvom/pvol into ztenu/ztenv before cududvn.  One
    # array copy: the ASSEMBLER's work, not a kernel's.  Named rather than
    # left unowned so it is not mistaken for missing transcription.
    (1017, 1025, "the assembler (a copy)"),
    (1026, 1039, "ntiedtke_cududvn"),
    (1040, 1060, "ntiedtke_ke_dissipation"),
    (1061, 1085, None),    # section 10, the deep/shallow switch-off
)


#: cu_ntiedtke_run's executable body.  The manifest originally covered
#: cumastrn only -- but the CONTRACT-level work lives here, one level up,
#: and was therefore outside it entirely.
CU_NTIEDTKE_RUN_BODY = (230, 331)

#: The same map for cu_ntiedtke_run.  Every line, owned or not.
RUN_OWNERSHIP: tuple[tuple[int, int, str | None], ...] = (
    (230, 254, "ntiedtke_prep / ntiedtke_convert"),
    # ONE do-loop, :255-278, and its lines have THREE owners.  The
    # manifest's ranges must be contiguous, so the owner string names all
    # three rather than pretending the loop is homogeneous:
    #   :257-259  pcte/pvom/pvol zeroed        -- the assembler
    #   :260-272  the conversion proper        -- ntiedtke_convert
    #   :273, 275 ptte/pqte seeded with forcing -- ntiedtke_convert
    #   :274, 276 zqq/ztt saved                -- the assembler (a copy)
    # The two assembler lines are the same shape as :1019-1024's
    # ztenu/ztenv copy: one array copy, the caller's work.
    (255, 278, "ntiedtke_convert + the assembler (:257-259 zeroing, "
               ":274/:276 the zqq/ztt copies)"),
    (279, 295, "the cumastrn stages (the call is :283-291)"),
    # CLAIMED 2026-08-29.  This was None and it was the hole between the
    # last cumastrn stage and post_run: the condensate detrainment, the
    # pt/pqv update, zprecc, and the lmfdudv momentum application.
    (296, 326, "ntiedtke_post_conversion"),
    (327, 331, "the assembler (errmsg/errflg/return)"),
)

#: The accumulate-vs-replace contract rests on these ranges, and on nothing
#: else.  Same shape as CALL_SITE_DEBTS: a claim about the reference's
#: pipeline that the port only inherits if the port reproduces it.
ACCUMULATE_DEBTS = {
    "pvom/pvol": (258, 259),
    "ptte/pqte seeding": (273, 276),
    "ptte/pqte differencing": (309, 310),
}

#: Debts DISCHARGED, kept rather than deleted so the record shows what
#: closed them.  A debt that vanishes from the table without a note is
#: indistinguishable from a debt someone quietly dropped.
DISCHARGED_DEBTS = {
    "ptte/pqte differencing": (
        "ntiedtke_post_conversion owns :296-326, which contains :309-310. "
        "Graded at max_ulp == 0 on 108 columns"),
}


def run_owns(line: int) -> str | None:
    for a, b, who in RUN_OWNERSHIP:
        if a <= line <= b:
            return who
    raise AssertionError(f"line {line} is outside {CU_NTIEDTKE_RUN_BODY}")


def unowned_ranges():
    """The complement: every range no kernel performs."""
    return tuple((a, b) for a, b, who in OWNERSHIP if who is None)


def owns(line: int) -> str | None:
    for a, b, who in OWNERSHIP:
        if a <= line <= b:
            return who
    raise AssertionError(f"line {line} is outside {CUMASTRN_BODY}")


def test_the_manifest_is_disjoint_and_complete():
    """Every line of the body is claimed exactly once, owned or not.

    A gap in the manifest is worse than an unowned range: an unowned range
    is visible and a gap is not.
    """
    covered = []
    for a, b, _ in OWNERSHIP:
        assert a <= b, f"inverted range {a}-{b}"
        covered.append((a, b))
    covered.sort()
    assert covered[0][0] == CUMASTRN_BODY[0], covered[0]
    assert covered[-1][1] == CUMASTRN_BODY[1], covered[-1]
    for (a1, b1), (a2, _) in zip(covered, covered[1:]):
        assert b1 + 1 == a2, (
            f"the manifest {'overlaps' if a2 <= b1 else 'has a hole'} "
            f"between {a1}-{b1} and {a2}-")


def test_the_unowned_set_is_reported_not_forgotten():
    """The orchestration's remaining work, computed rather than remembered.

    This test does not require the set to be empty -- it will not be until
    Phase 1 ends. It requires it to be STATED, and it fails if the total
    grows, so that a new unowned range is a deliberate act.
    """
    unowned = unowned_ranges()
    lines = sum(b - a + 1 for a, b in unowned)
    assert unowned, "if the set is empty, delete this test and say so"
    assert lines <= 25, (
        f"unowned cumastrn lines grew to {lines}. A stage that leaves MORE "
        "of the orchestration unclaimed than it found is going the wrong "
        f"way. Ranges: {unowned}")


def test_the_ktype_flip_is_owned():
    """:566-568 is the fifth failure's line, and it is now OWNED.

    It was unowned for eleven slices: the closure kernel takes the
    CLOSURE-TIME ktype as an input, so the flip was supplied by the
    fixture and by nothing in the pipeline. ntiedtke_cloud_depth performs
    it. Inverted from an is-unowned assertion, which is what a
    direction-of-the-gap test is for.
    """
    for line in (566, 567, 568):
        assert owns(line) == "ntiedtke_cloud_depth", (
            f"cumastrn:{line} lost its owner. The ktype flip selects "
            "scale_fac vs scale_fac2 and is the reason this port exists; "
            "it was unowned for eleven slices and must not become so "
            "again.")


def test_the_downdraft_zeroing_is_owned():
    """:580-588 is what four class-2 excuses rest on, and it is now OWNED.

    ``_CALLER_ALWAYS_ZEROES`` excuses cudlfsn's pmfd/pmfds/pmfdq/pdmfdp
    because cumastrn zeroes them immediately before the call. That was a
    property of the REFERENCE's call site, inherited by the port only if
    the port reproduced it. ntiedtke_cloud_depth reproduces it, so those
    four excuses are no longer conditional.
    """
    for line in (580, 588):
        assert owns(line) == "ntiedtke_cloud_depth"


def test_every_call_site_excuse_names_a_range_and_carries_its_debt():
    """An excuse justified by a call-site property must name the range.

    The excuse is not wrong -- it is CONDITIONAL, on work that has not been
    done. This links the two so the condition cannot be lost: while the
    range is unowned the excuse carries a visible debt, and when a kernel
    claims it the debt clears.
    """
    from tests.test_ntiedtke_aliasing_audit import CALL_SITE_DEBTS
    assert CALL_SITE_DEBTS, "no excuse declares a call-site dependency"
    outstanding = []
    for (routine, dummy), (a, b) in CALL_SITE_DEBTS.items():
        who = {owns(x) for x in (a, b)}
        assert len(who) == 1, (
            f"{routine}.{dummy} depends on cumastrn:{a}-{b}, which spans "
            "two owners; split the manifest range")
        if who.pop() is None:
            outstanding.append(f"{routine}.{dummy} -> :{a}-{b}")
    # Not a tautology and not a blocker: the COUNT is declared, so clearing
    # a debt -- or adding one -- forces this number to be updated, which is
    # the act of noticing.  All four are outstanding today.
    assert len(outstanding) == 0, (
        f"call-site debts outstanding changed to {len(outstanding)}: "
        f"{outstanding}.  If a kernel now performs the range, the excuse "
        "in _CALLER_ALWAYS_ZEROES is no longer conditional -- say so there "
        "and update this count.")


def test_the_debt_is_visible_in_the_port_doc():
    """A debt nobody can see is a receipt.

    The port doc must carry the unowned set, because that is the artifact a
    future session reads after a compact -- and the §6 table already
    failing to be a gate is exactly why this is checked rather than
    trusted.
    """
    from pathlib import Path
    doc = Path(__file__).resolve().parents[1] / "docs/ntiedtke/PORT-RECORD.md"
    text = doc.read_text(encoding="utf-8")
    assert "566-568" in text
    assert "580-588" in text


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])


def test_the_run_level_manifest_is_disjoint_and_complete():
    """cu_ntiedtke_run gets the same treatment as cumastrn."""
    cov = sorted((a, b) for a, b, _ in RUN_OWNERSHIP)
    assert cov[0][0] == CU_NTIEDTKE_RUN_BODY[0]
    assert cov[-1][1] == CU_NTIEDTKE_RUN_BODY[1]
    for (_, b1), (a2, _) in zip(cov, cov[1:]):
        assert b1 + 1 == a2, f"hole or overlap at {b1}/{a2}"


def test_the_accumulate_contract_rests_on_unowned_ranges():
    """Replace semantics is CORRECT, and conditional. Both, checked.

    MEASURED at cudtdqn's entry: ptent/ptenq are non-zero on 4,428 of
    5,292 rows -- the scheme accumulates into a NON-ZERO array. Replace
    semantics at the CumulusResult boundary is nevertheless right, but
    only because cu_ntiedtke_run does two things the port does not yet do:

      :258-259  zeroes pvom/pvol, so for MOMENTUM accumulate == replace
      :273-276  seeds ptte/pqte with the FORCING (ptf/pqvf) and saves
                ztt/zqq
      :309-310  differences against those copies, so only the convective
                increment escapes

    None of those three ranges has an owner. So the contract in the port
    doc is conditional in exactly the way cudlfsn's four class-2 excuses
    were, and carries the same visible debt.

    UPDATED 2026-08-29, and the update is the point of the test.
    ntiedtke_post_conversion owns :296-326, which contains :309-310, so
    the differencing debt is DISCHARGED -- by a kernel graded at
    max_ulp == 0 on 108 columns.

    The other two are not, and the distinction matters: their ranges now
    have an OWNER but the owner is "the assembler", which does not exist.
    A debt owned by an unwritten component is still a debt. Counting
    "has a non-None owner" as discharged is exactly the kind of true
    statement that measures a smaller thing than it appears to, so the
    test asks whether the owner is a GRADED KERNEL.
    """
    def discharged(name):
        a, b = ACCUMULATE_DEBTS[name]
        who = run_owns(a)
        return (who == run_owns(b) and who is not None
                and "assembler" not in who and "ntiedtke_" in who)

    outstanding = sorted(n for n in ACCUMULATE_DEBTS if not discharged(n))
    assert outstanding == ["ptte/pqte seeding", "pvom/pvol"], outstanding
    assert set(DISCHARGED_DEBTS) == set(ACCUMULATE_DEBTS) - set(outstanding), (
        f"DISCHARGED_DEBTS says {sorted(DISCHARGED_DEBTS)} but the "
        f"manifest says {sorted(set(ACCUMULATE_DEBTS) - set(outstanding))}")
    for name in outstanding:
        a, _ = ACCUMULATE_DEBTS[name]
        assert "assembler" in (run_owns(a) or ""), (
            f"{name} is outstanding but its range is not owned by the "
            f"assembler either -- it is owned by nothing, which is worse")


def test_the_accumulate_finding_is_in_the_port_doc():
    from pathlib import Path
    doc = (Path(__file__).resolve().parents[1]
           / "docs/ntiedtke/PORT-RECORD.md").read_text(encoding="utf-8")
    assert "4,428" in doc, "the measurement that settles accumulate-vs-replace"
    assert "309-310" in doc


# ---------------------------------------------------------------------------
# The two DEAD blocks, asserted rather than assumed
# ---------------------------------------------------------------------------
# Both are guarded by `.not. lmfscv .or. .not. lmfpen`, and both are
# `logical,parameter:: ... = .true.` in the pinned v4.6.1 source, so neither
# body can run.  They are listed as unowned in the manifest above, and a
# reader could reasonably conclude they are outstanding work.  They are not.
#
# Given the same standing this port gives `zentr` and `zodetr`: a later WRF
# that flips either parameter must BREAK a test rather than be inherited
# silently (asked for by review, review).
#
# The claim is checked against nt_cumastrn_body.inc because that is the
# artifact in this repo; the WRF tree is not.
#
# A SECOND REASON WAS GIVEN AND IS WITHDRAWN.  It read: "if the .inc's
# reading of the guards were wrong, the replication would not converge."
# That cannot discriminate (review, review).  A block that never
# executes cannot cause divergence whether the guard is read correctly or
# not -- and if the guard were live and the .inc reproduced it faithfully,
# both would run it and still converge.  CONVERGENCE IS SILENT ABOUT DEAD
# CODE IN BOTH DIRECTIONS.
#
# The conclusion stands on the first limb alone, which is a proof rather
# than an argument: `logical,parameter:: lmfscv = .true.` is settled at
# compile time.  Dropping the second limb makes the claim stronger, and it
# is dropped here rather than left to be read later as evidence for
# something it does not support -- which is the shape this port has been
# bitten by before.

DEAD_BLOCKS = {
    ":786-802": "the deep/shallow switch-off inside 6.6",
    ":1060-1082": "section 10, the same switch-off a posteriori",
    # A THIRD, found while reading :927-995 rather than by a failure.
    ":943-955": "the momtrans == 1 momentum arm; momtrans is a parameter "
                "= 2, so the pressure-gradient else arm is the live one",
}


def test_the_dead_blocks_are_recorded_as_dead_in_the_replication():
    """Both guards are .true. parameters; both bodies are unreachable."""
    from pathlib import Path
    inc = (Path(__file__).resolve().parents[1] / "tools"
           / "ntiedtke_wrf461_oracle" / "nt_cumastrn_body.inc").read_text(
        encoding="utf-8")
    flat = " ".join(inc.split())
    # The whole proof: a compile-time parameter, not an observed behaviour.
    assert "momtrans is 2" in flat, (
        "the replication no longer records momtrans = 2. It is a parameter, "
        "so :943-955 -- the `if (momtrans == 1)` momentum arm -- cannot "
        "run and the pressure-gradient else arm is the live one. If a "
        "later WRF set it to 1, the port would be transcribing the wrong "
        "arm and nothing else would show it.")
    assert "lmfscv and lmfpen are both .true. parameters" in flat, (
        "the replication no longer records that both guards are .true. "
        "parameters. If a later WRF made either .false., these blocks stop "
        "being dead and become real unowned work -- and NOTHING ELSE WOULD "
        "SHOW IT, because dead code cannot make the replication diverge.")
    for where in DEAD_BLOCKS:
        assert where.lstrip(":") in flat, (
            f"the replication no longer names the dead block at {where}")


def test_the_dead_blocks_are_handled_the_way_their_range_requires():
    """Two dead blocks, now in two different situations.

    :1060-1082 is still inside an UNOWNED range, so it inflates the
    unowned count by ~25 lines that cannot run. Stated rather than
    discounted, because discounting would hide the day it stops being
    dead.

    :786-802 is now inside a range ntiedtke_updraft_scale OWNS. So the
    kernel had a choice and made one: it skips the block. That is only
    safe while the guards are parameters, so the kernel must SAY it
    skipped rather than leave the omission to look like a transcription
    slip -- which is exactly how a missing block reads to a reviewer.
    """
    assert owns(1061) is None and owns(1085) is None, (
        "section 10 gained an owner; if it is now implemented, this test "
        "should require that instead")
    assert owns(786) == "ntiedtke_updraft_scale"
    from pathlib import Path
    cu = (Path(__file__).resolve().parents[1] / "gpuwm" / "core" / "kernels"
          / "ntiedtke.cu").read_text(encoding="utf-8")
    i = cu.index("Stage 16: the updraft rescale")
    header = cu[i:i + 2500]
    assert ":786-802" in header and "dead" in header.lower(), (
        "ntiedtke_updraft_scale owns :746-819 but does not record that it "
        "deliberately skips the dead block at :786-802. An unexplained "
        "omission inside an owned range is indistinguishable from a "
        "transcription slip.")
