"""The aliasing audit is a GATE, not a receipt.

`gpuwm/data/ntiedtke/oracle/nt-aliasing-audit.txt` lists every dummy in
`cu_ntiedtke.F90` whose INCOMING value is load-bearing -- either read
before it is written, or an ``intent(out)`` written only inside a branch so
that a column missing the branch keeps the caller's value.

Two things depend on that list being exact:

* every HARNESS must pass the caller's live arrays for those names.  A
  harness that passes fresh arrays produces a capture that is silently
  wrong, and no self-consistency check inside the harness catches it --
  the harness agrees with itself.  That is how the cutypen fixture shipped
  wrong, and only the NumPy mirror disagreeing with it surfaced the bug.

* every KERNEL must LEAVE THOSE SLOTS ALONE rather than initialising them.
  The natural CUDA idiom of zeroing outputs at entry diverges from WRF on
  every column that misses the branch, and the fixture is 66 of 108
  non-triggering -- so that idiom would be wrong on most of it.

So a change to the list is a change to both contracts.  This test re-runs
the audit and fails if the answer moved: a new routine, an edited
signature, or someone "simplifying" a conditional write into an
unconditional one would all silently invalidate work already graded.

It SKIPS when the WRF source is not reachable, which is the ordinary case
on a machine without the tree.  It never passes vacuously: when the source
is present its digest is checked against the pinned v4.6.1 value first, so
auditing some other file cannot be mistaken for agreement.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from gpuwm.verify.ntiedtke_oracle import ORACLE_DIR

#: The pinned v4.6.1 digest.  v4.8.0 ships this file byte-identical, so a
#: 4.8.0 tree audits to the same answer and is accepted.
_SRC_SHA = "e762101f04d4acd2d19047a92d0b7cd4e244df930f9f9ef7aabae54bfe9a9fd1"

_REL = Path("phys/physics_mmm/cu_ntiedtke.F90")
_CANDIDATES = (
    os.environ.get("NTIEDTKE_WRF_SRC"),
    r"\\wsl$\Ubuntu-22.04\home\user\WRF\MOVING" + "\\" + str(_REL),
    "/tmp/wrf461src/" + _REL.as_posix(),
)
_AUDIT = (Path(__file__).resolve().parents[1] / "tools"
          / "ntiedtke_wrf461_oracle" / "audit_intent_aliasing.py")
_RECEIPT = ORACLE_DIR / "nt-aliasing-audit.txt"


def _source():
    for cand in _CANDIDATES:
        if not cand:
            continue
        p = Path(cand)
        if p.is_file():
            got = hashlib.sha256(p.read_bytes()).hexdigest()
            if got != _SRC_SHA:
                pytest.skip(f"{p} is {got[:12]}, not the pinned v4.6.1 file")
            return p
    pytest.skip("cu_ntiedtke.F90 not reachable; set NTIEDTKE_WRF_SRC")


def _body(text):
    """The receipt without its hand-written header comment."""
    return [ln.rstrip() for ln in text.splitlines()
            if not ln.startswith("#")]


def test_the_aliasing_list_has_not_moved():
    src = _source()
    if not _AUDIT.is_file():
        pytest.skip("audit script missing")
    out = subprocess.run([sys.executable, str(_AUDIT), str(src)],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    got = _body(out.stdout)
    want = _body(_RECEIPT.read_text(encoding="utf-8"))
    assert got == want, (
        "the aliasing audit no longer matches nt-aliasing-audit.txt.\n"
        "That changes BOTH contracts -- which arrays a harness must pass "
        "live, and which slots a kernel must leave alone.  Re-read the "
        "diff before regenerating the receipt.\n\n"
        + "\n".join(
            f"  {'-' if line in want else '+'} {line}"
            for line in (set(want) ^ set(got)) if line.strip())[:2000])


def test_the_receipt_still_names_the_routines_that_matter():
    """A guard against the receipt being regenerated empty.

    If the audit script broke -- a regex that stopped matching, a parse
    that returned nothing -- regenerating would produce a clean-looking
    file with no findings, and this suite would go green on a list that
    says nothing.
    """
    text = _RECEIPT.read_text(encoding="utf-8")
    for routine in ("cutypen", "cuascn", "cubasmcn", "cudlfsn",
                    "cuddrafn", "cuflxn"):
        assert routine in text, f"{routine} absent from the audit receipt"
    assert "cubasmcn   ktype" in text, (
        "cubasmcn's conditionally-written ktype is the single most "
        "load-bearing row in this file and it is gone")


# ===========================================================================
# The audit as a gate on the MIRRORS, not just as a document
# ===========================================================================
# THE SIXTH INSTANCE IS WHY THIS EXISTS.  cudlfsn's ptd/pqd were zeroed in
# the mirror and came out wrong on levels 1-4 of every column -- and the
# audit file ALREADY listed all six of cudlfsn's outputs as class 2.  The
# audit was a gate on the transcription and nothing gated the MIRRORS
# against it, so the list sat in a text file while the fixture was built
# without consulting it.
#
# A load-bearing dummy that a mirror does not ACCEPT AS A PARAMETER cannot
# be honouring the caller's value: there is nowhere for that value to come
# from.  So the check is direct, and it fails at the shape rather than
# waiting for an oracle comparison that may not be able to see it.

import inspect                                             # noqa: E402

import gpuwm.verify.ntiedtke_ref as _ref                   # noqa: E402

#: Reference routine -> the mirror that carries it.  Only PORTED routines
#: are listed; the audit covers the whole scheme and the rest arrive later.
_PORTED = {
    "cuascn": _ref.np_ntiedtke_cuascn,
    "cudlfsn": _ref.np_ntiedtke_cudlfsn,
    "cubasmcn": _ref.np_ntiedtke_cubasmcn,
    "cutypen": _ref.np_ntiedtke_cutypen,
}

#: Load-bearing dummies a mirror legitimately does not take, each with the
#: reason it is safe.  AN ENTRY HERE IS A CLAIM ABOUT THE FORTRAN, and it
#: has to be one someone checked by reading the body -- not by analogy.
_EXCUSED = {
    ("cuascn", "puu"): (
        "class 1 only because its first use is the cubasmcn call argument "
        "at :1973.  Grepping cuascn's executable body (:1890-2258) finds "
        "puu nowhere else, and cubasmcn does not write it either, so no "
        "value flows in or out."),
    ("cuascn", "pvu"): "as puu.",
    ("cutypen", "cubot"): (
        "cutypen's cubot/cutop are the SAME STORAGE as cumastrn's "
        "kcbot/kctop, which the mirror takes under the caller's spelling."),
    ("cutypen", "cutop"): "as cubot.",
}


def _load_bearing():
    """``{routine: {dummy}}`` from the audit's two class sections."""
    out: dict[str, set[str]] = {}
    for line in (ORACLE_DIR / "nt-aliasing-audit.txt").read_text(
            encoding="utf-8").splitlines():
        if not line.startswith("  "):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        routine, dummy = parts[0], parts[1]
        if not routine.startswith("cu"):
            continue
        out.setdefault(routine, set()).add(dummy)
    return out


def test_every_load_bearing_dummy_reaches_its_mirror():
    """A class-1 or class-2 dummy must be an INPUT to the mirror.

    This is the check that would have caught cudlfsn's ptd/pqd before the
    first run instead of after it.
    """
    audit = _load_bearing()
    missing = []
    for routine, mirror in _PORTED.items():
        params = set(inspect.signature(mirror).parameters)
        for dummy in sorted(audit.get(routine, ())):
            if dummy in params or (routine, dummy) in _EXCUSED:
                continue
            missing.append(f"{routine}.{dummy}")
    assert not missing, (
        "load-bearing dummies absent from their mirror's signature, so the "
        f"caller's value has nowhere to come from: {missing}")


def test_the_audit_actually_covers_the_ported_routines():
    """Green must not be reachable by the audit having no rows.

    If audit_intent_aliasing.py ever stops emitting a routine, the test
    above passes vacuously for it.  This is the guard on that.
    """
    audit = _load_bearing()
    for routine in _PORTED:
        assert audit.get(routine), f"{routine} has no rows in the audit"


def test_no_excuse_is_stale():
    """An excuse for a dummy the audit no longer lists is a stale claim.

    Excuses are claims about the Fortran.  One that no longer corresponds
    to an audit row has stopped being checked by anything, and the next
    reader would take it as still true.
    """
    audit = _load_bearing()
    stale = [f"{r}.{d}" for (r, d) in _EXCUSED
             if d not in audit.get(r, set())]
    assert not stale, f"excuses with no matching audit row: {stale}"


# ---------------------------------------------------------------------------
# Limb 2: the fixture must be able to TELL "left alone" from "zeroed"
# ---------------------------------------------------------------------------
# Limb 1 above catches a mirror that cannot receive the caller's value.  It
# does NOT catch the worse case, which is a fixture that cannot see the
# difference: if every column the routine skips happens to carry ZERO on
# entry, then "left alone" and "zeroed" are the same bytes, mirror and
# reference agree, and max_ulp == 0 is structurally blind.
#
# That is the cuentrn degeneracy generalised.  The cudlfsn bug was visible
# only because the entry values were non-zero on levels 1-4 -- luck, and the
# kind that does not repeat, because a quieter pre-state hides it forever.
#
# So for each class-2 dummy the fixture must carry at least one (column,
# level) where the routine did NOT write the slot and the entry value was
# NON-ZERO.  A dummy that cannot satisfy this is a named coverage gap, not
# something to force with a synthetic entry value -- a synthesised pre-state
# is a synthesised fixture, which is what this whole architecture retires.

#: class-2 dummy -> (in csv, in field, out csv, out field).  The AUDIT
#: drives which dummies must appear; this only says where each is recorded.
#: A class-2 dummy with no row here FAILS -- silence is the thing being
#: removed.
_WHERE = {
    ("cudlfsn", d): (f"nt-cudlfsn-in-levels.csv", f"{d}_in",
                     "nt-cudlfsn-out-levels.csv", d)
    for d in ("ptd", "pqd", "pmfd", "pmfds", "pmfdq", "pdmfdp")
}
_WHERE.update({
    ("cutypen", "cutu"): ("nt-cuinin-levels.csv", "ptu",
                          "nt-cutypen-levels.csv", "cutu"),
    ("cutypen", "cuqu"): ("nt-cuinin-levels.csv", "pqu",
                          "nt-cutypen-levels.csv", "cuqu"),
    ("cutypen", "culu"): ("nt-cuinin-levels.csv", "plu",
                          "nt-cutypen-levels.csv", "culu"),
    ("cutypen", "culab"): ("nt-cuinin-levels.csv", "klab",
                           "nt-cutypen-levels.csv", "culab"),
})

#: Class-2 dummies whose CALLER ALWAYS HANDS ZERO at this call site.
#:
#: This is a stronger statement than "the fixture is too narrow", and the
#: distinction matters: it says the class-2 hazard is NOT REACHABLE from
#: this call site, so "left alone" and "zeroed" are the same by
#: construction rather than by luck.  Each claim is CHECKED below against
#: every fixture row, so a later WRF that stops zeroing breaks the excuse
#: instead of inheriting it.
_CALLER_ALWAYS_ZEROES = {
    ("cudlfsn", "pmfd"): "cumastrn:580-588 zeroes pmfd/zmfds/zmfdq/"
                         "zdmfdp/zdpmel immediately before the call.",
    ("cudlfsn", "pmfds"): "as pmfd.",
    ("cudlfsn", "pmfdq"): "as pmfd.",
    ("cudlfsn", "pdmfdp"): "as pmfd.",
    ("cutypen", "culu"): "cuinin leaves plu identically zero -- measured, "
                         "5,292 of 5,292 rows.",
    ("cutypen", "culab"): "cuinin leaves klab identically zero -- measured, "
                          "5,292 of 5,292 rows.",
}

#: Class-2 dummies with no entry record to compare against at all.  A
#: genuine coverage gap, named rather than closed with a synthetic value:
#: a synthesised pre-state is a synthesised fixture.
_NO_ENTRY_RECORD = {
    ("cuascn", "ktype"): (
        "ktype is a per-column scalar and :1910 writes it for exactly the "
        "columns whose entry ktype is already 0, so no column can carry a "
        "non-zero unwritten value.  Covered instead by the closure slice, "
        "which grades the POST-FLIP ktype from its own capture."),
    ("cutypen", "cubot"): (
        "cubot/cutop are cumastrn's kcbot/kctop under another spelling, "
        "recorded only at cutypen's exit; there is no separate entry "
        "record to compare against."),
    ("cutypen", "cutop"): "as cubot.",
}

#: cubasmcn is the audit's extreme case -- ALL THIRTEEN outputs class 2 --
#: and it is covered by a different mechanism, not by the entry/exit
#: comparison this file performs.
#:
#: It is called at EVERY LEVEL of cuascn's loop (:1968) with the live
#: mid-loop arrays, so there is no single entry snapshot for an exit to be
#: compared against: the "caller's value" is a different thing at each of
#: 46 levels.  What grades its class-2 discipline is slice 4a in
#: test_ntiedtke_prep_parity.py, which drives the mirror from cuascn's
#: reconstructed prologue and grades the UNTOUCHED case directly -- the
#: reconstruction being validated by the outputs agreeing at max_ulp == 0
#: on the columns where it does fire.
_COVERED_ELSEWHERE = {
    ("cubasmcn", d): "graded by slice 4a; called per level, no entry snapshot"
    for d in ("ktype", "kcbot", "klab", "pmfub", "plrain", "ptu", "pqu",
              "plu", "pmfu", "pmfus", "pmfuq", "pmful", "pdmfup")
}

#: Excuses justified by a property of the reference's CALL SITE, mapped to
#: the cumastrn range that property lives in.
#:
#: THE EXCUSE IS NOT WRONG, IT IS CONDITIONAL.  "cumastrn zeroes these four
#: immediately before the call" is a fact about the REFERENCE's pipeline,
#: and the port only inherits it if the port reproduces that pipeline.  No
#: kernel performs :580-588 today.  Linking the two here means the
#: condition cannot be lost: test_ntiedtke_cumastrn_ownership.py checks the
#: link, and the range shows as unowned until a kernel claims it.
#:
#: Found by review (review): written as "the hazard is not reachable
#: from that call site", the excuse reads as a closed question.  It is not
#: closed.
CALL_SITE_DEBTS = {
    ("cudlfsn", "pmfd"): (580, 588),
    ("cudlfsn", "pmfds"): (580, 588),
    ("cudlfsn", "pmfdq"): (580, 588),
    ("cudlfsn", "pdmfdp"): (580, 588),
}

#: Fields recorded as decimal integers rather than IEEE-754 hex words.
_INTEGER_FIELDS = {"klab", "culab"}


def _class2_rows():
    """``{(routine, dummy)}`` from the audit's conditional-write section."""
    text = (ORACLE_DIR / "nt-aliasing-audit.txt").read_text(encoding="utf-8")
    tail = text[text.index("intent(out) WITH ONLY CONDITIONAL WRITES"):]
    out = set()
    for line in tail.splitlines():
        parts = line.split()
        if line.startswith("  ") and len(parts) > 2 and parts[2] == "writes":
            out.add((parts[0], parts[1]))
    return out


def _cells(csv_name, field):
    from gpuwm.verify.ntiedtke_oracle import load_csv
    return {(int(r["case"]), r["dx"], int(r["k"])): r[field]
            for r in load_csv(csv_name)}


def _is_zero(raw, field):
    if field in _INTEGER_FIELDS:
        return int(raw) == 0
    return int(raw, 16) in (0x00000000, 0x80000000)


def test_every_class2_dummy_is_accounted_for():
    """A class-2 dummy of a ported routine must be mapped or named.

    Silence is the thing being removed: the sixth instance happened with
    the answer sitting in two places nobody read.
    """
    unmapped = [f"{r}.{d}" for (r, d) in sorted(_class2_rows())
                if r in _PORTED and (r, d) not in _WHERE
                and (r, d) not in _CALLER_ALWAYS_ZEROES
                and (r, d) not in _NO_ENTRY_RECORD
                and (r, d) not in _COVERED_ELSEWHERE]
    assert not unmapped, (
        f"class-2 dummies with nowhere recorded to check them: {unmapped}")


def test_the_fixture_can_tell_left_alone_from_zeroed():
    """The DISCRIMINATING limb, and the one that makes limb 1 a gate.

    Limb 1 catches a mirror that cannot receive the caller's value.  This
    catches the worse case: a fixture where every skipped column carries
    ZERO on entry, so "left alone" and "zeroed" are the same bytes,
    mirror and reference agree, and ``max_ulp == 0`` is structurally
    blind.  That is the cuentrn degeneracy generalised.

    The cudlfsn bug was visible only because the entry values happened to
    be non-zero on levels 1-4.  That was luck, and a quieter pre-state
    would have hidden it permanently.
    """
    blind = []
    for (routine, dummy), (icsv, ifield, ocsv, ofield) in _WHERE.items():
        if routine not in _PORTED:
            continue
        if (routine, dummy) in _CALLER_ALWAYS_ZEROES:
            continue
        entry, exit_ = _cells(icsv, ifield), _cells(ocsv, ofield)
        witnesses = sum(1 for key, ev in entry.items()
                        if key in exit_ and exit_[key] == ev
                        and not _is_zero(ev, ifield))
        if witnesses == 0:
            blind.append(f"{routine}.{dummy}")
    assert not blind, (
        "these class-2 dummies have NO column the routine left unwritten "
        f"with a non-zero entry value: {blind}.  The fixture cannot tell "
        "'correctly left alone' from 'zeroed at entry', so max_ulp == 0 is "
        "structurally blind on them and grading them proves nothing.  Widen "
        "the case table -- do NOT synthesise an entry value.")


def test_the_caller_really_does_always_zero_them():
    """The excuses in _CALLER_ALWAYS_ZEROES are CHECKED, not asserted.

    Each claims a property of the reference: that the caller hands this
    slot zero at every fixture row.  An excuse nothing verifies is the
    receipt this whole gate exists to replace -- and if a later WRF stops
    zeroing, the class-2 hazard becomes reachable and this must break.
    """
    for (routine, dummy), why in _CALLER_ALWAYS_ZEROES.items():
        if routine not in _PORTED:
            continue
        icsv, ifield = _WHERE[(routine, dummy)][:2]
        nonzero = [k for k, v in _cells(icsv, ifield).items()
                   if not _is_zero(v, ifield)]
        assert not nonzero, (
            f"{routine}.{dummy} is excused as always-zero on entry ({why}) "
            f"but {len(nonzero)} fixture rows carry a non-zero value, e.g. "
            f"{nonzero[:3]}.  The excuse is now false and the dummy needs "
            "real discrimination.")


def test_every_call_site_excuse_declares_its_debt():
    """An always-zero excuse resting on the caller must say which lines.

    Without this, "the caller zeroes it" is a claim with no address, and
    nothing can check whether the port reproduces the caller.
    """
    for (routine, dummy), why in _CALLER_ALWAYS_ZEROES.items():
        if "cumastrn" not in why:
            continue
        assert (routine, dummy) in CALL_SITE_DEBTS, (
            f"{routine}.{dummy} is excused by a cumastrn call-site property "
            "but declares no line range, so nothing can check that the "
            "port reproduces it.  Add it to CALL_SITE_DEBTS.")


def test_no_class2_excuse_is_stale():
    """An excuse must correspond to a live class-2 audit row."""
    rows = _class2_rows()
    stale = [f"{r}.{d}"
             for (r, d) in {**_CALLER_ALWAYS_ZEROES, **_NO_ENTRY_RECORD,
                            **_COVERED_ELSEWHERE}
             if (r, d) not in rows]
    assert not stale, f"excuses with no matching class-2 audit row: {stale}"
