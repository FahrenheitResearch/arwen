"""The ``zcons`` family: gate the constants that look alike.

WHY THIS EXISTS. Two failures in this port are the same failure —
**resolution by apparent identity instead of by provenance** (framing:
review):

* ``pmfude_rate`` at cuascn's output *looked like* ``pmfude_rate`` at the
  adjustments block's entry. Same name, same array, different value,
  because :746-819 rescaled it in between. Resolved by name, and the
  mirror came out 1.26x low on 42 of 108 columns (§23).
* ``zcons`` in one scope *looks like* ``zcons`` in another. Same name,
  different literal factor. Resolved by name, and you get a value whose
  provenance is a different scope.

The second has already produced one misreading — the cuascn mirror and
kernel both carried a comment claiming cuascn's cap is "three times looser
than the closure's", which is **false**: ``zcons2`` is ``3/(g*dt)`` in all
three scopes that declare it and they are identical. What differs by three
is ``zcons`` against ``zcons2``, one character apart, both live in
``cumastrn``, and ``zcons`` has exactly ONE consumer.

A wrong ``1/(g*dt)`` factor produces mass fluxes that are finite, plausible
and off by a fixed ratio — the least visible arithmetic error available, and
one that would read as a physics disappointment at f012 rather than a bug.

Built BEFORE porting :996-1016, which is ``zcons``'s only consumer.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

_INC = (Path(__file__).resolve().parents[1] / "tools"
        / "ntiedtke_wrf461_oracle" / "nt_cumastrn_body.inc")

#: (scope, name) -> (what it is, the lines that CONSUME it).
#:
#: Enumerated from cu_ntiedtke.F90 v4.6.1.  Three scopes declare a `zcons2`
#: and all three mean the same thing; only `cumastrn` declares a `zcons`.
CONSTANTS = {
    ("cumastrn", "zcons"):   ("1/(g*dt)", (1000,)),
    ("cumastrn", "zcons2"):  ("3/(g*dt)", (522, 686, 705, 755)),
    ("cuascn", "zcons2"):    ("3/(g*dt)", (1991, 2027)),
    ("cuflxn", "zcons2"):    ("3/(g*dt)", (3043,)),
    ("cuflxn", "zcons1a"):   ("cpd/(alf*g*ztaumel)", (2979,)),
}

#: The one site in the whole scheme that uses the 1x factor.  Everything
#: else that caps a mass flux uses 3x.  Getting this one wrong makes the
#: MOMENTUM rescale three times too permissive.
ZCONS_ONLY_CONSUMER = 1000


def test_no_two_entries_share_a_name_within_a_scope():
    """The table's shape: a name resolves uniquely GIVEN a scope.

    Trivially true as written, and that is the point -- if it ever stops
    being true, `zcons2` in some scope means two things and every use site
    becomes ambiguous.
    """
    seen = set()
    for scope, name in CONSTANTS:
        assert (scope, name) not in seen
        seen.add((scope, name))
    by_name: dict[str, set[str]] = {}
    for scope, name in CONSTANTS:
        by_name.setdefault(name, set()).add(scope)
    assert by_name["zcons2"] == {"cumastrn", "cuascn", "cuflxn"}, by_name
    assert by_name["zcons"] == {"cumastrn"}, (
        "zcons is declared outside cumastrn. It is the 1x factor and the "
        "only thing that separates it from zcons2 is one character.")


def test_zcons_and_zcons2_really_do_differ_by_three():
    """The hazard, measured rather than asserted.

    If these were equal the family would be harmless and this whole file
    would be ceremony.
    """
    g, dt = np.float32(9.81), np.float32(60.0)
    zcons = np.float32(np.float32(1.0) / np.float32(g * dt))
    zcons2 = np.float32(np.float32(3.0) / np.float32(g * dt))
    assert zcons != zcons2
    ratio = float(zcons2) / float(zcons)
    assert abs(ratio - 3.0) < 1e-5, ratio


def test_the_replication_uses_zcons_at_its_only_consumer():
    """Driven off the .inc so it cannot drift.

    ``nt_cumastrn_body.inc`` replicates cumastrn and is proved against
    byte-unmodified WRF at 0 differing words. If it used ``zcons2`` at the
    momentum rescale, the replication would diverge -- so unlike the
    dead-block case, convergence IS evidence here: this line executes.
    """
    lines = _INC.read_text(encoding="utf-8").split("\n")
    uses = [(n + 1, l.strip()) for n, l in enumerate(lines)
            if "zcons" in l and "=" in l and not l.strip().startswith("!")]
    decls = [u for u in uses if u[1].startswith(("zcons ", "zcons ="))]
    assert len(decls) == 1, f"expected one zcons declaration, got {decls}"
    consumers = [u for u in uses
                 if "*zcons" in u[1].replace("*zcons2", "")]
    assert len(consumers) == 1, (
        "the replication uses zcons at a number of sites other than one; "
        f"the reference uses it at exactly one (:{ZCONS_ONLY_CONSUMER}). "
        f"Sites: {consumers}")


def test_every_ported_constant_carries_its_declared_value():
    """The mirrors resolve by SCOPE, not by name.

    Each ported routine's own zcons2 must be 3/(g*dt). This is the check
    that would have caught a mirror that copied cumastrn's zcons into
    cuascn, or vice versa.
    """
    from gpuwm.verify.ntiedtke_ref import NtConstants
    c = NtConstants()
    dt = np.float32(60.0)
    want3 = np.float32(np.float32(3.0) / np.float32(c.g * dt))
    want1 = np.float32(np.float32(1.0) / np.float32(c.g * dt))

    import inspect
    import gpuwm.verify.ntiedtke_ref as ref
    for fn_name, expect in (("np_ntiedtke_cuascn", want3),
                            ("np_ntiedtke_closure", want3),
                            ("np_ntiedtke_cuflxn", want3)):
        src = inspect.getsource(getattr(ref, fn_name))
        assert "_F(3.0) / _F(c.g" in src.replace("\n", " ").replace(
            "  ", " "), (
            f"{fn_name} no longer forms zcons2 as 3/(g*dt); if it now uses "
            "the 1x factor its mass-flux cap is three times too tight")
        assert float(expect) > float(want1)


def test_the_corrected_claim_did_not_come_back():
    """The misreading this family already produced, pinned out.

    Both the cuascn mirror and its kernel carried "three times looser than
    the closure's". The closure uses the SAME number. A comment that is
    wrong about which constants differ is how the next person resolves one
    by name.
    """
    import inspect
    import gpuwm.verify.ntiedtke_ref as ref
    src = inspect.getsource(ref.np_ntiedtke_cuascn)
    assert "looser than the closure" not in src or "CORRECTED" in src, (
        "the cuascn mirror claims its cap is looser than the closure's. "
        "It is not -- both are 3/(g*dt).")
    cu = (Path(__file__).resolve().parents[1] / "gpuwm" / "core" / "kernels"
          / "ntiedtke.cu").read_text(encoding="utf-8")
    i = cu.index("Stage 7: cuascn")
    block = cu[i:i + 12000]
    if "three times looser" in block:
        assert "CORRECTED" in block, (
            "the cuascn kernel still claims its cap is three times looser "
            "than the closure's, without the correction beside it")


def test_the_table_names_every_declaration_in_the_replication():
    """A constant in the .inc with no table row is unaccounted for."""
    lines = _INC.read_text(encoding="utf-8").split("\n")
    declared = {l.strip().split("=")[0].strip() for l in lines
                if l.strip().startswith(("zcons", "zcons2"))
                and "=" in l and "*" not in l.split("=")[0]}
    # VACUITY GUARD (retrofitted; review). `declared <= named` is true
    # of the empty set, so this test passed on any change to the .inc that
    # moved those declarations out of the shape the comprehension expects
    # -- a rename, a continuation, a different indent. The subset check is
    # worth nothing without it.
    # Named rather than counted: the comprehension collects NAMES, and
    # there are two of them however many scopes declare them.  The first
    # draft floored it at 3 by confusing the two, and the guard caught its
    # own author -- which is the argument for naming over counting.
    assert {"zcons", "zcons2"} <= declared, (
        f"the .inc scan found {sorted(declared)}. The subset assertion "
        f"below is vacuously true on an empty set, so this is the real "
        f"gate: if the declarations move out of the shape the "
        f"comprehension expects, it must fail here rather than pass there.")
    named = {name for _, name in CONSTANTS}
    assert declared <= named, (
        f"the replication declares constants with no table row: "
        f"{sorted(declared - named)}")


def test_the_capture_first_default_is_in_the_contract():
    """The rule that generalises this gate, pinned where it is followed.

    The constant family and the six capture instances are the same
    failure: resolution by apparent identity instead of by provenance. The
    gate above covers the constants; the contract covers the data, and a
    rule that lives only in a review thread is the receipt this port keeps
    retiring.
    """
    rules = (Path(__file__).resolve().parents[1]
             / "docs/ntiedtke/STANDING-RULES.md").read_text(encoding="utf-8")
    assert "CAPTURE FIRST" in rules
    assert "provenance" in rules
