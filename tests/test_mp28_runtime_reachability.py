"""Every place ``mp_physics=28`` used to fall off the production runtime.

WHY THIS FILE EXISTS
--------------------
An auditor traced a real mp=28 forecast and found four places where the
scheme silently stopped being a first-class selector: the restart writer
refused it outright, three REFL_10CM admission gates and a fourth in the
prepared single-domain runner omitted it, the legacy-RRTMG radii table had
no row for it, and the offline downscale child could not read an mp=28
parent.  None of those is a numerics defect and none of them NaNs -- which
is precisely why none of the nineteen column fixtures, the 600-step
forecast smoke, or the three adversarial audits saw them.  They are
REACHABILITY defects, and the only thing that finds a reachability defect
is running the lane.

WHAT IS GATED HERE, AND WHY EACH GATE IS THE SHAPE IT IS
--------------------------------------------------------
1. THE CENSUS (``test_no_stale_pre_28_scheme_admission_tuple_survives``).
   The original pinned-gap test named three paths in ``gpuwm/runtime.py``
   and therefore could not see the FOURTH copy of the same tuple in
   ``gpuwm/prepared_single_domain_forecast.py``.  A test that enumerates
   known sites can only ever be as complete as the last audit.  This one
   walks the AST of every ``.py`` file under ``gpuwm/`` instead and fails
   on the stale literal wherever it appears, so a fifth copy added next
   month cannot hide.

2. THE RESTART ROUND TRIP.  ``tests/test_preflight.py::
   test_mp28_survives_a_restart_round_trip`` gates the manifest layer on
   host arrays.  This gates the FILE: a real device state, ``write_restart``
   to a real ``.npz``, ``restore_restart`` into a second state whose aerosol
   fields were zeroed first, and a BITWISE comparison of all five aerosol
   arrays.  Adding the identity string is what made an mp=28 checkpoint
   possible at all, and an identity string that lets a restart proceed while
   dropping the aerosol state would be worse than the old refusal -- so the
   two new inventory guards are gated in both directions too.

3. THE PACKAGING MEASUREMENT.  ``tests/test_package_data_coverage.py``
   already asserts the ``exclude-package-data`` DECLARATION out of
   ``tomllib`` and MEASURES the wheel contents with setuptools -- but the
   project virtualenv has no setuptools, so the measurement half skips
   exactly where it matters.  The gate here finds an interpreter that does
   have setuptools (this repo's CI image and the system python both do) and
   runs the measurement in a subprocess, so the strongest control on a file
   gpuwm has no right to redistribute is not inert in the environment the
   suite actually runs in.

TWO THINGS THE MEASUREMENT CHANGED ABOUT THE PUBLISHED STORY
-------------------------------------------------------------
The evidence document described the REFL gap as "wrfout frames whose
REFL_10CM was never produced".  Reproducing it showed something stronger:
the tree model gates ``refl_10cm_due`` on the history handler and the clock
only (gpuwm/core/model.py:669-675), so the mp=28 adapter DID stage the field
and the unconsumed stash then raised ``RuntimeError: REFL_10CM stash was not
consumed before reuse`` at the SECOND output frame.  See
:func:`test_the_missing_28_refl_admission_was_a_crash_not_a_missing_field`.

And the restart identity was only half the restart gap.  ``mp_physics=8``
has always bound its four table digests into the physics-setup fingerprint;
mp=28 bound none, so an mp=28 checkpoint could resume against a different
``freezeH2O.dat`` or a different ``CCN_ACTIVATE.BIN`` with every identity
check passing.  For this scheme that is a trajectory substitution: the
activation fraction that sets droplet number is READ out of ``tnccn_act``
(module_mp_thompson.F:5229-5230), not computed.  See
:func:`test_the_mp28_restart_identity_binds_every_table_it_reads`.

ONE DEFECT THIS FILE MEASURES AND DOES NOT FIX
-----------------------------------------------
``gpuwm/core/rrtmgp.py``'s scheme map resolves mp=28 to ``"kessler"``.  The
file belongs to another owner, so it is reported rather than edited; the
measurement, the three consequences and the reason the census tolerates it
are in :data:`KNOWN_SCHEME_KEYED_DICTS_WITHOUT_28`.

TWO SKIP SITES, BOTH NAMED
---------------------------
``test_the_use_mp_re_citation_is_checkable_against_stock_wrf`` skips when the
stock WRF v4.6.1 reference tree is absent, and
``test_the_ccn_activation_blob_is_absent_from_the_built_package_data`` skips
when no interpreter on the machine has setuptools.  Neither swallows a
failure: each names the one missing thing.  If this module is adopted into
``tests/test_thompson_aerosol_gpu.py::_SUITE_MODULES`` they must be added to
that file's ``_SKIP_SITES`` census and to the evidence document's skip table
in the same change.
"""

from __future__ import annotations

import ast
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pytest

from conftest import requires_gpu

REPO = pathlib.Path(__file__).resolve().parent.parent
PACKAGE = REPO / "gpuwm"

#: The reflectivity-admission tuple as it was spelled BEFORE mp=28 existed.
#: Four production gates carried it verbatim.
STALE_REFL_ADMISSION = frozenset({1, 6, 8, 10, 18})

#: What every REFL_10CM admission gate must be now.  28 belongs on WRF's own
#: structure, not on a gpuwm convenience: ``mp_gt_driver`` reaches
#: ``calc_refl10cm`` from ONE call site (module_mp_thompson.F:1458) gated on
#: ``diagflag .and. do_radar_ref == 1`` (:1449) and never on
#: ``is_aerosol_aware``, and the routine takes no droplet-number argument
#: (:5710-5711), so the aerosol-aware package publishes the same field on the
#: same cadence as the classic one.
#:
#: 16 (WDM6) joined 2026-08-09 with the mp=16 port, and it is RE-DERIVED on
#: the same WRF structure rather than added to make a red test green:
#: ``wdm6`` reaches ``refl10cm_wdm6`` from ONE call site
#: (module_mp_wdm6.F:291) inside ``IF (diagflag .and. do_radar_ref == 1)``
#: (:279-280), the identical gate and the identical
#: ``refl_10cm(i,k,j) = max(-35., dBZ(k))`` store (:294).  So mp=16 publishes
#: REFL_10CM on exactly the cadence the other admitted schemes do, and every
#: gate that admits 6/8/10 must admit it.  The lane already moved all four
#: PRODUCTION constants; this pin was the test-side copy and was stale, not
#: wrong-headed.  Unlike 28, WDM6's routine DOES take a number argument
#: (nr1d, :2957) -- that is a producer-side difference (the port gives it its
#: own kernel, gpuwm/core/kernels/wdm6_refl.cu) and changes nothing about
#: admission.
REFL_ADMISSION = frozenset({1, 6, 8, 9, 10, 16, 18, 28, 50})

#: The deliberate exception.  ``PORTED_MP_PHYSICS`` names the selectors with
#: a ported MIXED nest edge, and mp=28 has none: every one of its eleven
#: mixed pairs is refused by name through ``UNVALIDATED_MIXED_EDGE_SELECTORS``
#: because no cross-scheme entry closure for nc/nwfa/nifa has been measured.
#: Listing 28 there and then refusing all its pairs would be a
#: self-contradiction.  Pinned by
#: ``tests/test_preflight.py`` (``assert 28 not in mt.PORTED_MP_PHYSICS``);
#: named here so the census below cannot be "fixed" by widening it.
#: The second entry arrived with the 1.4.1 merge, not with the port: the
#: public HRRR hierarchy route is new since the port's base (9c9c20cf) and
#: its gate was written when 28 did not exist.  28 stays out of it, and the
#: reason is the port's OWN published blocker rather than staleness: mp=28
#: has no aerosol lateral boundary condition, which is the one deviation
#: this package records as growing without bound with run length.  The HRRR
#: route is precisely a nested, laterally-forced, multi-hour route, so it is
#: the worst place in the tree to admit that deviation -- and it is the one
#: route a stranger runs from a public config.  Revisit when a QNWFA/QNIFA
#: ingest lane exists, not before.
DELIBERATE_STALE_SITES = {
    ("gpuwm/core/microphysics_transition.py", "PORTED_MP_PHYSICS"),
    ("gpuwm/hrrr_route_inputs.py", "SUPPORTED_MICROPHYSICS"),
}

#: Every ``mp_physics`` value ``gpuwm/config.py`` accepts (:1952).  Used to
#: tell a microphysics scheme table apart from an arbitrary integer-keyed
#: map -- RRTMG's band and g-point tables are keyed 1..16 and would
#: otherwise look exactly like one.  Re-derived 2026-08-09 against the same
#: validator line after mp=16 (WDM6) joined it; the census below is a
#: shape-detector, so a selector missing here would make the detector blind
#: to a dict keyed on it rather than fail.
ACCEPTED_MP_PHYSICS = frozenset({0, 1, 6, 8, 10, 16, 18, 28})

#: Scheme-KEYED DICTS that omit 28, and the judgement for each.  A dict is a
#: third shape the same defect comes in and the AST membership scan cannot
#: see it, so it is scanned separately.
#:
#: * ``microphysics_transition._MASS_FIELDS`` / ``_MOMENT_FIELDS`` are the
#:   MIXED nest-edge inventories.  28 is absent on purpose and the refusal
#:   is named (``UNVALIDATED_MIXED_EDGE_MOMENTS`` records exactly what an
#:   mp=28 mixed edge would have to move).
#:
#: * ``gpuwm/core/rrtmgp.py``'s scheme map is an OPEN DEFECT, measured and
#:   reported by this package rather than fixed here (the file belongs to
#:   another owner).  ``{6: "wsm6", 8: "thompson", 10: "morrison",
#:   18: "nssl"}.get(mp_physics, "kessler")`` resolves mp=28 to "kessler",
#:   with three consequences on the DEFAULT radiation path for an mp=28 run
#:   (rte-rrtmgp): ``ice_active`` becomes False so ``cal_cldfra1`` is called
#:   with ``f_qi = f_qs = False``; ``effective_fields`` stays empty so the
#:   scheme's effc/effi/effs never reach cloud optics; and
#:   ``hydrometeor_paths`` takes its kessler branch, which returns CONSTANT
#:   10 um liquid and 50 um ice radii.  MEASURED on this tree: an
#:   ice-and-snow-bearing column at 258 K/60 kPa and 240 K/40 kPa gets
#:   CLDFRA 1.0 with ice active and 0.0 without -- an overcast ice cloud
#:   radiating as clear sky.  ``gpuwm/core/preflight.py:2713`` already
#:   PRICES mp=28's effective-radius columns on WRF's use_mp_re table, so
#:   the tree currently budgets for radii the radiation never consumes.
#:
#: The assertion below is "no NEW scheme-keyed dict omits 28".  It
#: deliberately TOLERATES the rrtmgp row disappearing (a sibling package may
#: close it in this same wave) while refusing a fresh one.
KNOWN_SCHEME_KEYED_DICTS_WITHOUT_28 = {
    "gpuwm/core/microphysics_transition.py",
    "gpuwm/core/rrtmgp.py",
}


# ---------------------------------------------------------------------------
# 1. The census.
# ---------------------------------------------------------------------------

def _literal_int_container(node: ast.AST) -> frozenset[int] | None:
    """``(1, 6, 8)`` / ``{1, 6, 8}`` / ``frozenset((...))`` -> the int set."""
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        values = []
        for element in node.elts:
            if (isinstance(element, ast.Constant)
                    and isinstance(element.value, int)
                    and not isinstance(element.value, bool)):
                values.append(element.value)
            else:
                return None
        return frozenset(values)
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in ("frozenset", "set") and len(node.args) == 1):
        return _literal_int_container(node.args[0])
    return None


def _mentions_mp_physics(node: ast.AST) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Attribute) and "mp_physics" in sub.attr:
            return True
        if isinstance(sub, ast.Name) and "mp_physics" in sub.id:
            return True
    return False


def scheme_selector_census() -> list[tuple[str, int, str, frozenset[int]]]:
    """Every mp_physics scheme-set literal under ``gpuwm/``.

    Two shapes are collected, because the four known defects came in both:
    a MEMBERSHIP test whose left side names ``mp_physics``, and a module
    CONSTANT holding a scheme set (which is what the fix turns the
    memberships into, and which therefore has to be scanned too or the fix
    would create a new blind spot).
    """
    rows: list[tuple[str, int, str, frozenset[int]]] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        relative = path.relative_to(REPO).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                for op, comparator in zip(node.ops, node.comparators):
                    if not isinstance(op, (ast.In, ast.NotIn)):
                        continue
                    values = _literal_int_container(comparator)
                    if values is None or not _mentions_mp_physics(node.left):
                        continue
                    rows.append((relative, node.lineno, "membership", values))
            elif (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)):
                values = _literal_int_container(node.value)
                if values is None:
                    continue
                name = node.targets[0].id
                if not any(token in name.upper() for token in
                           ("MP_PHYSICS", "MICROPHYSICS", "REFL", "SCHEME")):
                    continue
                rows.append((relative, node.lineno, name, values))
    return rows


def scheme_keyed_dict_census() -> list[tuple[str, int, tuple[int, ...]]]:
    """Dict literals keyed by mp_physics values that omit 28.

    The third shape of the same defect.  ``gpuwm/core/rrtmgp.py``'s
    ``{6: "wsm6", 8: "thompson", ...}.get(mp_physics, "kessler")`` is
    invisible to a membership scan and to a constant scan, and it is the
    most consequential surviving one -- see
    :data:`KNOWN_SCHEME_KEYED_DICTS_WITHOUT_28`.
    """
    rows: list[tuple[str, int, tuple[int, ...]]] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        relative = path.relative_to(REPO).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict) or len(node.keys) < 2:
                continue
            keys = []
            for key in node.keys:
                if (isinstance(key, ast.Constant)
                        and isinstance(key.value, int)
                        and not isinstance(key.value, bool)):
                    keys.append(key.value)
                else:
                    keys = None
                    break
            if not keys:
                continue
            values = set(keys)
            # A microphysics scheme table, not an arbitrary integer map:
            # every key must be an mp_physics selector gpuwm accepts
            # (gpuwm/config.py:1148's set), and it must key 8 plus at least
            # one other ported selector.  Without the subset test this
            # catches RRTMG's band/g-point tables, which are keyed 1..16.
            if (values <= ACCEPTED_MP_PHYSICS and 8 in values
                    and (values & {6, 10, 18}) and 28 not in values):
                rows.append((relative, node.lineno, tuple(sorted(values))))
    return rows


def test_no_new_scheme_keyed_dict_omits_28():
    """The dict shape, with both known rows judged in the constant above."""
    census = scheme_keyed_dict_census()
    unexpected = sorted(
        (path, line, keys) for path, line, keys in census
        if path not in KNOWN_SCHEME_KEYED_DICTS_WITHOUT_28)
    assert unexpected == [], (
        "a NEW mp_physics-keyed dict omits 28:\n  "
        + "\n  ".join(f"{p}:{n} keys={k}" for p, n, k in unexpected)
        + "\nJudge it: either add 28, or record the reason in "
        "KNOWN_SCHEME_KEYED_DICTS_WITHOUT_28 with the measurement.")

    # The deliberate one must still be deliberate.
    from gpuwm.core import microphysics_transition as mt

    # One row per refused selector, and EVERY refused selector has one:
    # 16 joined the selector tuple with the WDM6 port and its moments row
    # was missing, which turned the named refusal into a bare KeyError(16).
    assert (set(mt.UNVALIDATED_MIXED_EDGE_MOMENTS)
            == set(mt.UNVALIDATED_MIXED_EDGE_SELECTORS) == {16, 28})
    assert set(mt.UNVALIDATED_MIXED_EDGE_MOMENTS[28]) == {
        "nr", "ni", "nc", "nwfa", "nifa"}
    # WDM6 is double-moment in cloud AND rain and carries a CCN reservoir:
    # ncr(:,:,1)=nn, ncr(:,:,2)=nc, ncr(:,:,3)=nr (module_mp_wdm6.F:238-240),
    # all three prognostic scalars in the WRF Registry.  It has no ni/ns/ng.
    assert set(mt.UNVALIDATED_MIXED_EDGE_MOMENTS[16]) == {"nr", "nc", "nn"}


def test_no_stale_pre_28_scheme_admission_tuple_survives():
    """SCAN, not a list of known paths.

    The pinned-gap test this replaces asserted
    ``runtime_source.count("mp_physics in (1, 6, 8, 10, 18)") == 3`` and was
    therefore blind to the identical fourth gate in
    ``gpuwm/prepared_single_domain_forecast.py``.  Enumerating sites cannot
    find the site nobody enumerated; scanning can.
    """
    census = scheme_selector_census()
    assert census, "the census walk found nothing -- it is broken"

    stale = [
        (path, line, label) for path, line, label, values in census
        if values == STALE_REFL_ADMISSION
        and (path, label) not in DELIBERATE_STALE_SITES
    ]
    assert stale == [], (
        "these sites still carry the pre-mp=28 admission tuple "
        f"{sorted(STALE_REFL_ADMISSION)}:\n  "
        + "\n  ".join(f"{p}:{n} ({label})" for p, n, label in stale)
        + "\nEvery one of them is a gate an mp=28 forecast reaches.  If a "
        "site genuinely must exclude 28, add it to DELIBERATE_STALE_SITES "
        "with the reason -- do not delete this assertion.")

    # And the deliberate exception must still BE the exception it claims.
    from gpuwm.core import microphysics_transition as mt

    assert set(mt.PORTED_MP_PHYSICS) == STALE_REFL_ADMISSION
    assert 28 in mt.UNVALIDATED_MIXED_EDGE_SELECTORS, (
        "mp=28 left PORTED_MP_PHYSICS without joining the NAMED mixed-edge "
        "refusal, so its nest edges now fall through to the generic "
        "'not a ported selector' message")


def test_every_refl_10cm_admission_constant_admits_28():
    """The four production gates, by value rather than by source text.

    ``tests/test_refl.py::test_refl_stash_has_no_trajectory_or_restart_reader``
    pins the exact set of files that consume the one-frame REFL handoff.
    This asserts that each of them admits 28, so the pinned consumer census
    and the admission census cannot drift apart.
    """
    from gpuwm import prepared_single_domain_forecast as psdf
    from gpuwm import runtime
    from gpuwm.verify.cases import nest_ideal_common, real74_n5s

    for module, attribute in (
            (runtime, "REFL_10CM_MICROPHYSICS"),
            (psdf, "REFL_10CM_MICROPHYSICS"),
            (nest_ideal_common, "REFL_10CM_MICROPHYSICS"),
            (real74_n5s, "REFLECTIVITY_MICROPHYSICS")):
        values = frozenset(getattr(module, attribute))
        assert values == REFL_ADMISSION, (
            f"{module.__name__}.{attribute} is {sorted(values)}, expected "
            f"{sorted(REFL_ADMISSION)}")

    # gpuwm/core/refl.py is the producer side and is deliberately a
    # DIFFERENT set: 18 routes through NSSL's own native reflectivity and
    # never through compute_refl_10cm.  Assert the asymmetry on purpose so
    # a future "consistency" edit has to argue with it.
    refl_source = (PACKAGE / "core" / "refl.py").read_text(encoding="utf-8")
    # Re-derived 2026-08-09 for the WDM6 port: the producer gate gained 16
    # because compute_refl_10cm now HAS an mp=16 branch (:615, launching
    # gpuwm/core/kernels/wdm6_refl.cu).  18 is still absent for the original
    # reason.  Re-derived again 2026-08-11 (1.9.1): mp=9 and mp=50 joined
    # the CONSUMER set because both were shipped staging a field nothing
    # consumed (the exact mp=28 crash, at the second output frame), and
    # both belong on 18's side of the asymmetry -- Milbrandt-Yau stashes
    # the scheme's own Zet (gpuwm/core/milbrandt2.py, the INOUT dummy WRF
    # binds straight to refl_10cm) and P3 stashes its own diagnostic
    # reflectivity (gpuwm/core/p3.py), so neither routes through
    # compute_refl_10cm.  The asymmetry the assertion protects -- consumer
    # set minus producer set == the native-reflectivity schemes -- is
    # asserted as arithmetic below rather than left implicit in two
    # hand-spelled tuples.
    assert "cfg.mp_physics not in (1, 6, 8, 10, 16, 28)" in refl_source
    assert REFL_ADMISSION - frozenset({1, 6, 8, 10, 16, 28}) == {9, 18, 50}
    assert "elif cfg.mp_physics in (8, 28):" in refl_source, (
        "mp=28 no longer shares mp=8's calc_refl10cm branch; WRF has ONE "
        "such routine with no aerosol-aware arm (module_mp_thompson.F:"
        "5710-6028), so a separate branch could only diverge from it")
    # WDM6 does NOT share it: refl10cm_wdm6 is its own Fortran routine
    # (module_mp_wdm6.F:2957-3131) taking a prognostic rain number, so it
    # gets its own branch and its own kernel.
    assert "elif cfg.mp_physics == 16:" in refl_source


def _code_without_comments_or_strings(path: pathlib.Path) -> str:
    """Executable text only.

    A substring search over raw source cannot distinguish a live gate from
    a comment that QUOTES the retired gate, and these files document what
    they replaced -- deliberately, because that is what makes the fix
    legible.  Tokenizing drops comments and string literals so the search
    means what it says.
    """
    import io
    import tokenize

    pieces = []
    with path.open("rb") as handle:
        for token in tokenize.tokenize(io.BytesIO(handle.read()).readline):
            if token.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            pieces.append(token.string)
    return " ".join(pieces)


def test_the_runtime_refl_gates_are_reached_through_the_named_constant():
    """No inlined copy may survive in any of the four owning files.

    The defect was never "the tuple is wrong"; it was "there are four
    copies of the tuple".  Value equality above cannot see a fifth inlined
    copy inside the same file, so the code text is checked too.
    """
    for relative in ("gpuwm/runtime.py",
                     "gpuwm/prepared_single_domain_forecast.py",
                     "gpuwm/verify/cases/nest_ideal_common.py",
                     "gpuwm/verify/cases/real74_n5s.py"):
        code = _code_without_comments_or_strings(REPO / relative)
        normalized = code.replace(" ", "")
        assert "mp_physicsin(1,6,8,10,18)" not in normalized, relative
        assert "mp_physicsnotin(1,6,8,10,18)" not in normalized, relative

    runtime_source = _code_without_comments_or_strings(
        REPO / "gpuwm/runtime.py")
    # Definition plus the three gates (case output, the per-substep
    # refl_due schedule, the nested-tree history handoff).
    assert runtime_source.count("REFL_10CM_MICROPHYSICS") == 4, (
        "gpuwm/runtime.py should reference REFL_10CM_MICROPHYSICS exactly "
        "four times (one definition + three gates); a different count means "
        "a gate was added or one stopped using the constant")


def test_the_prepared_runner_consumes_the_mp28_refl_handoff():
    """Behavioural, on the FOURTH site -- the one the old gap test missed.

    ``_consume_due_native_refl_10cm`` is a pure function of (ticks, state,
    consumer), so it is exercised directly rather than by standing up a
    prepared forecast.  A recording consumer proves the handoff is taken
    for 28 and, equally, that tick 0 still takes nothing.
    """
    from types import SimpleNamespace

    from gpuwm.prepared_single_domain_forecast import (
        _consume_due_native_refl_10cm)

    sentinel = object()
    calls = []

    def consumer(state):
        calls.append(state)
        return sentinel

    for mp_physics in sorted(REFL_ADMISSION):
        state = SimpleNamespace(
            qv=np.zeros(1),
            physics=SimpleNamespace(mp_physics=mp_physics))
        assert _consume_due_native_refl_10cm(state, 1, consumer) is sentinel
        assert _consume_due_native_refl_10cm(state, 0, consumer) is None
    assert len(calls) == len(REFL_ADMISSION)

    # A scheme with no native field must still take nothing.
    other = SimpleNamespace(qv=np.zeros(1),
                            physics=SimpleNamespace(mp_physics=0))
    assert _consume_due_native_refl_10cm(other, 1, consumer) is None


def test_the_runtime_tree_history_frame_consumes_the_mp28_refl_handoff():
    """Behavioural, on the THIRD runtime site (the nested-tree lane).

    ``_submit_tree_history_frame`` is the production handoff for every
    multi-domain run, and mp=28 is reachable exactly there -- the registry
    makes it a per-domain component override, never a base template
    (gpuwm/physics_compat.py's readiness note).  So this is not a corner:
    it is the ONE lane an mp=28 user's history frames go through.
    """
    from types import SimpleNamespace

    from gpuwm import runtime
    from gpuwm.core import refl

    field = np.zeros((2, 3, 4), dtype=np.float32)
    submitted = []

    class _Writers:
        @staticmethod
        def submit(node, ticks, *, refl_field):
            submitted.append((ticks, refl_field))

    class _Refl:
        # The stub MIRRORS gpuwm.core.refl: only the consumer is faked
        # (that is what this pin measures).  The due predicate and the
        # domain-start reader are the real ones, so a stub cannot make
        # the site look reachable by answering a question differently
        # from the module it stands in for.
        consume_refl_10cm = staticmethod(lambda state: field)
        refl_10cm_stash_is_due = staticmethod(refl.refl_10cm_stash_is_due)
        domain_start_ticks_of = staticmethod(refl.domain_start_ticks_of)

    class _UhDiag:
        @staticmethod
        def reset_up_heli_max(state):
            return None

    node = SimpleNamespace(state=SimpleNamespace(
        qv=np.zeros(1), physics=SimpleNamespace(mp_physics=28)))

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setitem(sys.modules, "gpuwm.core.refl", _Refl)
        monkey.setitem(sys.modules, "gpuwm.core.uh_diag", _UhDiag)
        runtime._submit_tree_history_frame(_Writers, node, ticks=6)
        # tick 0 is not a period boundary: nothing was stashed, so nothing
        # may be consumed.
        runtime._submit_tree_history_frame(_Writers, node, ticks=0)
    finally:
        monkey.undo()

    assert submitted == [(6, field), (0, None)], submitted


def test_the_missing_28_refl_admission_was_a_crash_not_a_missing_field():
    """MEASURED consequence, reproduced against the real refl contract.

    The published description of this gap said mp=28 wrfout frames would be
    written "with REFL_10CM never produced".  That understates it.  The tree
    model gates ``refl_10cm_due`` on the history handler and the clock ONLY
    (gpuwm/core/model.py:669-675), never on ``mp_physics``, so the mp=28
    adapter DID stage the field on every history-due step
    (gpuwm/core/microphysics_aerosol.py's ``compute_and_stash_refl_10cm``
    call).  With 28 missing from the consumer's admission set the stash was
    never taken, and ``refl.stash_refl_10cm``'s consume-once contract then
    raises on the NEXT history-due step.

    So the pre-fix behaviour on the ONLY lane mp=28 is reachable through
    (``tools.prepared_domain_tree_forecast``, the sole runner route with
    ``microphysics`` in ``allowed_component_overrides``) was a hard
    ``RuntimeError`` at the second output frame, not a quietly absent
    diagnostic.  This reproduces both sides.
    """
    from types import SimpleNamespace

    from gpuwm.core.refl import consume_refl_10cm, stash_refl_10cm

    def three_history_steps(admission, mp_physics):
        state = SimpleNamespace(
            qv=np.zeros(1),
            physics=SimpleNamespace(mp_physics=mp_physics, refl_10cm=None))
        for _ in range(3):
            stash_refl_10cm(state,
                            np.zeros((2, 3, 4), dtype=np.float32))
            if state.physics.mp_physics in admission:
                consume_refl_10cm(state)

    # The retired admission set: mp=8 survives, mp=28 raises on step 2.
    three_history_steps(STALE_REFL_ADMISSION, 8)
    with pytest.raises(RuntimeError, match="not consumed before reuse"):
        three_history_steps(STALE_REFL_ADMISSION, 28)

    # The current one: both survive.
    three_history_steps(REFL_ADMISSION, 8)
    three_history_steps(REFL_ADMISSION, 28)

    # And the set the production consumer actually uses is the current one.
    from gpuwm import runtime

    three_history_steps(runtime.REFL_10CM_MICROPHYSICS, 28)


def test_the_nest_history_refl_handoff_admits_28():
    """``gpuwm/verify/cases/nest_ideal_common.py`` -- the fifth site.

    It is not one of the four the auditor traced; it was found by the
    census.  It matters because the ideal-nest cases exist to run
    production's seams, and a case that silently skipped the handoff for
    mp=28 would leave the stash unconsumed until ``refl.py`` raised at the
    NEXT history frame -- far from the omission.
    """
    from types import SimpleNamespace

    from gpuwm.core import refl
    from gpuwm.verify.cases import nest_ideal_common

    consumed = []

    class _Refl:
        # Mirrors gpuwm.core.refl; only the consumer is faked.  See the
        # tree-history pin above for why the predicate stays real.
        consume_refl_10cm = staticmethod(consumed.append)
        refl_10cm_stash_is_due = staticmethod(refl.refl_10cm_stash_is_due)
        domain_start_ticks_of = staticmethod(refl.domain_start_ticks_of)

    node = SimpleNamespace(
        cfg=SimpleNamespace(run=SimpleNamespace(mp_physics=28), grid_id=1),
        state=SimpleNamespace(physics=object()))

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setitem(sys.modules, "gpuwm.core.refl", _Refl)
        nest_ideal_common.consume_history_reflectivity(node, ticks=3)
        assert consumed == [node.state]
        # tick 0 is not a period boundary; nothing is staged, nothing taken.
        nest_ideal_common.consume_history_reflectivity(node, ticks=0)
        assert consumed == [node.state]
    finally:
        monkey.undo()


# ---------------------------------------------------------------------------
# 2. Restart.
# ---------------------------------------------------------------------------

MP28_AEROSOL_FIELDS = ("nc", "nwfa", "nifa", "nwfa2d", "nifa2d")


def test_the_restart_identity_names_the_scheme_and_its_provenance():
    """A scheme with no identity row cannot be checkpointed at all.

    The string is not decoration: ``physics_setup_identity`` folds it into a
    SHA-256 that a resume must reproduce exactly, so it is the mechanism by
    which a trajectory-changing implementation change refuses an
    incompatible restart BEFORE any state is restored.  It therefore has to
    name the things that would change the trajectory, in the style of the
    mp=8 row, not just the scheme.
    """
    from gpuwm.io import restart

    identity = restart.MICROPHYSICS_ALGORITHM_IDENTITIES[28]
    assert identity.startswith("thompson-aerosol-aware-wrf-v4.6.1-")
    for token in ("prognostic-nc", "nwfa", "nifa", "ccn-activate-table",
                  "demott", "koop", "scavenging", "surface-emission",
                  "synthetic-aerosol-init"):
        assert token in identity, token
    # It must be a NEW identity, not a re-use of the classic Thompson one.
    assert identity != restart.MICROPHYSICS_ALGORITHM_IDENTITIES[8]
    # And nothing else moved.  Re-derived 2026-08-09: the WDM6 port added
    # key 16 and NOTHING ELSE, which is the claim this pin exists to hold.
    # 16 is the only new key, its identity is its own string rather than a
    # re-use, and every pre-existing row is byte-unchanged (mp=8's exact
    # string is re-asserted below, and mp=28's above).
    #
    # RE-DERIVED AGAIN at the 1.9 gate: 50 is here because the P3
    # one-category port (mp_physics=50, gpuwm/core/p3.py) registered its own
    # restart identity, which is exactly what a new scheme is required to
    # do -- gpuwm/io/restart.py refuses to checkpoint a scheme with no
    # identity row at all.  It is a DELIBERATE pin update, not a widened
    # assertion: 50 is the only new key, and the two properties this case
    # exists for are re-asserted for it below on the same terms as 16's.
    #
    # mp=9 (Milbrandt-Yau) is deliberately ABSENT and that is not an
    # oversight: the mp=9 port registered no restart identity, so a
    # Milbrandt-Yau run cannot be checkpointed.  Stated here rather than
    # left to be rediscovered, because a silent absence in this table is
    # indistinguishable from a forgotten one.
    assert set(restart.MICROPHYSICS_ALGORITHM_IDENTITIES) == {
        0, 1, 6, 8, 10, 16, 18, 28, 50}
    p3_identity = restart.MICROPHYSICS_ALGORITHM_IDENTITIES[50]
    assert p3_identity.startswith("p3-one-category-wrf-v4.6.1-")
    assert p3_identity not in {
        value for key, value in restart.MICROPHYSICS_ALGORITHM_IDENTITIES.items()
        if key != 50}
    wdm6_identity = restart.MICROPHYSICS_ALGORITHM_IDENTITIES[16]
    assert wdm6_identity.startswith("wdm6-double-moment-warm-rain-wrf-v4.6.1-")
    assert wdm6_identity not in {
        value for key, value in restart.MICROPHYSICS_ALGORITHM_IDENTITIES.items()
        if key != 16}
    assert restart.MICROPHYSICS_ALGORITHM_IDENTITIES[8] == (
        "classic-thompson-wrf-v4.6.1-experimental-v3-cloud-fallout-"
        "refl10cm-ng-shadow-snow-rime-mass-number-velocity")


@requires_gpu
def test_the_mp28_restart_identity_binds_every_table_it_reads():
    """mp=8 pinned its table bytes; mp=28 pinned none of them.

    ``physics_setup_identity`` gave mp=28 only ``{"scheme_id": 28}``, so a
    checkpoint could resume against a DIFFERENT ``freezeH2O.dat`` or a
    different ``CCN_ACTIVATE.BIN`` with every identity check passing.  That
    matters more here than for any other scheme: the droplet number that
    defines mp=28 is READ out of ``tnccn_act`` at a fixed radius/kappa index
    (module_mp_thompson.F:5229-5230), not computed, so a table substitution
    IS a trajectory substitution and nothing else in the header would show
    it.

    Both inventories are required.  mp=28 really does load the four classic
    tables (its adapter reuses the frozen mp=8 sedimentation and
    classic-graupel launchers), and it alone loads the activation blob.
    """
    from gpuwm.core.thompson_aerosol_contract import (
        AEROSOL_TABLE_ASSETS, AEROSOL_TABLE_SET_ID)
    from gpuwm.core.thompson_contract import CLASSIC_TABLE_ASSETS, TABLE_SET_ID
    from gpuwm.io import restart

    import cupy as cp

    state, _forcing, cfg = _mp28_forecast_state(cp)
    identity = restart.physics_setup_identity(state, cfg)
    record = identity["microphysics"]["thompson_aerosol"]

    classic = record["classic_tables"]
    assert classic["table_set"] == TABLE_SET_ID
    assert [item["sha256"] for item in classic["assets"]] == [
        asset.sha256 for asset in CLASSIC_TABLE_ASSETS]

    aerosol = record["aerosol_tables"]
    assert aerosol["table_set"] == AEROSOL_TABLE_SET_ID
    assert aerosol["table_set"] != TABLE_SET_ID, (
        "the two coefficient inventories must stay tellable apart")
    assert [item["sha256"] for item in aerosol["assets"]] == [
        asset.sha256 for asset in AEROSOL_TABLE_ASSETS]

    # A PACKAGING fact must never enter the trajectory identity.  This dict
    # is hashed into physics_setup_fingerprint, so a "redistributed" key
    # would make flipping AEROSOL_ASSET_REDISTRIBUTED refuse every mp=28
    # checkpoint written before the flip -- while the bytes bound above,
    # which are the only thing that can move a float, stayed identical.
    # It was written here for part of 2026-08-01 and removed the same day.
    # Asserted as an absence so re-adding it is a test failure, not a
    # silently shipped incompatibility.
    assert "redistributed" not in aerosol, (
        "how the activation blob was DELIVERED is not part of the "
        "trajectory identity; its sha256 already binds what is")
    from gpuwm.core.thompson_aerosol_contract import (
        AEROSOL_ASSET_REDISTRIBUTED)

    assert AEROSOL_ASSET_REDISTRIBUTED is True, (
        "the constant itself stays -- it is published on the registry row "
        "and drives the packaging gates; only the fingerprint is free of it")
    assert "redistributed" not in json.dumps(record), (
        "no nested record may smuggle the packaging flag back into the "
        "hashed physics setup")

    # The aerosol INITIAL CONDITION is part of the identity too: this build
    # runs thompson_init's synthetic profile, and a future WIF metgrid
    # ingest is a different initial condition, not a compatible resume.
    assert "thompson-init-synthetic" in record["aerosol_source"]
    assert "aer-init-opt-0" in record["aerosol_source"]

    # An mp=8 checkpoint must be unaffected -- it carries "thompson", never
    # "thompson_aerosol".
    from dataclasses import replace as _replace

    mp8_cfg = _replace(cfg, mp_physics=8)
    mp8_identity = restart.physics_setup_identity(state, mp8_cfg)
    assert "thompson" in mp8_identity["microphysics"]
    assert "thompson_aerosol" not in mp8_identity["microphysics"]


def _mp28_forecast_state(cp):
    """The G4 forecast domain, with the production aerosol profile."""
    sys.path.insert(0, str(REPO / "tests"))
    from test_mp28_forecast_smoke import (  # noqa: E402
        FORECAST_U, _attach_specified_boundaries, _build_states,
        _forecast_config, _tables_or_skip)

    _tables_or_skip()
    cfg = _forecast_config()
    state, forcing = _build_states(cp, cfg, bubble=False, wind=FORECAST_U)
    _attach_specified_boundaries(state, forcing, cfg)
    from gpuwm.core import microphysics

    receipt = microphysics.microphysics_init(state, cfg)
    assert receipt, "microphysics_init returned an empty mp=28 receipt"
    cp.cuda.Stream.null.synchronize()
    return state, forcing, cfg


@requires_gpu
def test_an_mp28_restart_round_trips_every_aerosol_field_bitwise():
    """WRITE a real file, READ it back, compare BITS.

    The manifest-level gate in tests/test_preflight.py proves the fields are
    classified and copied.  This proves the whole file cycle -- header,
    physics-setup fingerprint, array manifest, npz, restore -- carries them,
    which is the claim a user depends on and the one that was refused
    outright before the identity row existed.

    ``nifa2d`` is deliberately included even though thompson_init leaves it
    at exactly zero (it is not even a thompson_init dummy argument,
    module_mp_thompson.F:424-444): a field that is legitimately zero is the
    easiest one for a restart to "round-trip" by accident, so ``nc`` is
    perturbed to a random field first and every array is compared as raw
    bytes rather than with a tolerance.
    """
    import cupy as cp

    from gpuwm.io import restart

    state, _forcing, cfg = _mp28_forecast_state(cp)
    rng = np.random.default_rng(2028)
    state.nc[...] = cp.asarray(
        rng.random(state.nc.shape, dtype=np.float32) * np.float32(1.5e8))
    # nifa2d is exactly zero out of thompson_init (it is not even one of its
    # dummy arguments, module_mp_thompson.F:424-444) and an all-zero field is
    # the easiest thing in the world to "round-trip" by accident -- a build
    # that dropped it entirely would still pass.  A nonzero nifa2d is legal
    # state (WRF's Registry declares QNIFA2D in the restart stream precisely
    # because a WIF-ingest run has one), so seed one and require it back.
    state.nifa2d[...] = cp.asarray(
        rng.random(state.nifa2d.shape, dtype=np.float32) * np.float32(9.0))
    cp.cuda.Stream.null.synchronize()

    # The profile really is loaded, so this is not a zeros-round-trip.
    assert float(cp.max(state.nwfa)) > 1.0e7
    assert float(cp.max(state.nifa)) > 1.0e5
    assert float(cp.max(state.nwfa2d)) > 0.0

    saved = {name: cp.asnumpy(getattr(state, name)).copy()
             for name in MP28_AEROSOL_FIELDS}

    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "mp28-restart.npz"
        restart.write_restart(path, state, cfg)

        header = restart.read_restart_header(path)
        assert header["physics_setup"]["algorithms"]["microphysics"] == \
            restart.MICROPHYSICS_ALGORITHM_IDENTITIES[28]

        with np.load(path, allow_pickle=False) as stored:
            present = set(stored.files)
        for name in MP28_AEROSOL_FIELDS:
            assert f"state/{name}" in present, name

        restored, _forcing2, _cfg2 = _mp28_forecast_state(cp)
        for name in MP28_AEROSOL_FIELDS:
            getattr(restored, name)[...] = 0
        cp.cuda.Stream.null.synchronize()
        for name in MP28_AEROSOL_FIELDS:
            assert not np.array_equal(
                cp.asnumpy(getattr(restored, name)), saved[name]), name

        restart.restore_restart(path, restored, cfg)
        cp.cuda.Stream.null.synchronize()

    for name in MP28_AEROSOL_FIELDS:
        got = cp.asnumpy(getattr(restored, name))
        assert got.dtype == saved[name].dtype == np.float32, name
        assert got.tobytes() == saved[name].tobytes(), (
            f"state/{name} did not round-trip bitwise through the restart "
            f"file (max |delta| = {np.abs(got - saved[name]).max()!r})")


@requires_gpu
def test_an_mp28_restart_refuses_to_write_without_the_aerosol_state():
    """The guard that makes the identity row safe to add.

    An identity string alone would let a checkpoint proceed while dropping
    the aerosol state, and the resumed run would stay finite and bounded --
    WRF's terminal clamps (module_mp_thompson.F:3976-3982) hold nwfa/nifa at
    their floors and nc at 2/rho -- so nothing downstream would notice.  That
    is strictly worse than the refusal it replaced, which is why the write
    side fails closed on each of the five fields.
    """
    import cupy as cp

    from gpuwm.io import restart

    state, _forcing, cfg = _mp28_forecast_state(cp)
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "ok.npz"
        restart.write_restart(path, state, cfg)   # baseline: it works
        for name in MP28_AEROSOL_FIELDS:
            held = getattr(state, name)
            setattr(state, name, None)
            try:
                with pytest.raises(restart.RestartManifestError) as excinfo:
                    restart.write_restart(
                        pathlib.Path(tmp) / f"missing-{name}.npz", state, cfg)
                assert f"state/{name}" in str(excinfo.value), name
            finally:
                setattr(state, name, held)


@requires_gpu
def test_an_mp28_restart_refuses_a_file_that_omits_the_aerosol_state():
    """The read side of the same guard.

    The generic inventory check compares the FILE against the RESUMING
    state, so two equally aerosol-less endpoints agree with each other.  The
    mp=28 check compares the file against the canonical inventory instead,
    which is what makes a truncated checkpoint a refusal rather than a
    silently aerosol-inert resume.
    """
    import cupy as cp

    from gpuwm.io import restart

    state, _forcing, cfg = _mp28_forecast_state(cp)
    header_key = restart._HEADER_KEY
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "full.npz"
        restart.write_restart(path, state, cfg)
        with np.load(path, allow_pickle=False) as stored:
            payload = {key: stored[key] for key in stored.files}
        header = json.loads(
            bytes(bytearray(payload[header_key])).decode("utf-8"))

        for dropped in MP28_AEROSOL_FIELDS:
            # Drop the ARRAY and its manifest entry, i.e. exactly what a
            # build whose DomainState never allocated the field would
            # write.  Dropping only the array would trip the generic
            # array-manifest consistency check first and prove nothing
            # about the mp=28 inventory guard.
            wounded = json.loads(json.dumps(header))
            wounded["array_manifest"].pop(f"state/{dropped}")
            truncated = pathlib.Path(tmp) / f"drop-{dropped}.npz"
            reduced = {k: v for k, v in payload.items()
                       if k != f"state/{dropped}"}
            reduced[header_key] = np.frombuffer(
                json.dumps(wounded).encode("utf-8"), dtype=np.uint8)
            np.savez(truncated, **reduced)

            with pytest.raises(restart.RestartMismatchError) as excinfo:
                restart.restore_restart(truncated, state, cfg)
            message = str(excinfo.value)
            assert f"state/{dropped}" in message, dropped
            assert "aerosol" in message, (dropped, message)


@requires_gpu
def test_the_mp28_restart_identity_fails_closed_on_the_ccn_table(tmp_path):
    """It must refuse to WRITE an identity it cannot substantiate.

    ``CCN_ACTIVATE.BIN`` ships now, but ``GPUWM_THOMPSON_CCN_ACTIVATE``
    still lets a run bind to another copy, so "the operator pointed it
    somewhere else" is a normal state of the world, not a corrupted install.
    A header that quietly omitted the aerosol table binding in that case
    would be the worst outcome: the checkpoint would look complete and would
    resume against anything.  Both failure modes -- absent and
    byte-different -- are therefore RestartManifestError at write time.
    """
    import cupy as cp

    from gpuwm.io import restart

    state, _forcing, cfg = _mp28_forecast_state(cp)
    from gpuwm.core.thompson_aerosol_contract import (
        AEROSOL_TABLE_PATH_ENV, resolve_aerosol_table_root,
        resolve_ccn_activation_path)

    good = resolve_ccn_activation_path(None, resolve_aerosol_table_root(None))

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setenv(AEROSOL_TABLE_PATH_ENV, str(tmp_path / "absent.bin"))
        with pytest.raises(restart.RestartManifestError, match="CCN"):
            restart.physics_setup_identity(state, cfg)
    finally:
        monkey.undo()

    tampered = tmp_path / "CCN_ACTIVATE.BIN"
    payload = bytearray(good.read_bytes())
    payload[100] ^= 0xFF          # one flipped bit, correct byte count
    tampered.write_bytes(bytes(payload))
    assert tampered.stat().st_size == good.stat().st_size

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setenv(AEROSOL_TABLE_PATH_ENV, str(tampered))
        with pytest.raises(restart.RestartManifestError, match="SHA-256"):
            restart.physics_setup_identity(state, cfg)
    finally:
        monkey.undo()

    # ...and the unmodified environment still succeeds, so the two refusals
    # above are about the asset and not about the code path.
    restart.physics_setup_identity(state, cfg)


@requires_gpu
def test_an_mp28_restart_refuses_a_different_ccn_activation_table():
    """The table identity is LOAD-BEARING, not decorative.

    Writing the digests into the header only helps if the reader compares
    them.  This mutates the stored CCN activation SHA-256, recomputes the
    header's own self-consistency fingerprint so that check still passes,
    and requires the restore to refuse on the LIVE-versus-STORED comparison.

    That is the failure this record exists for: the droplet number mp=28
    predicts is READ out of ``tnccn_act`` at a fixed radius/kappa index
    (module_mp_thompson.F:5229-5230), so resuming a checkpoint against a
    different activation table is resuming a different model.
    """
    import cupy as cp

    from gpuwm.io import restart

    state, _forcing, cfg = _mp28_forecast_state(cp)
    header_key = restart._HEADER_KEY
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "full.npz"
        restart.write_restart(path, state, cfg)
        with np.load(path, allow_pickle=False) as stored:
            payload = {key: stored[key] for key in stored.files}
        header = json.loads(
            bytes(bytearray(payload[header_key])).decode("utf-8"))

        setup = header["physics_setup"]
        record = setup["microphysics"]["thompson_aerosol"]
        original = record["aerosol_tables"]["assets"][0]["sha256"]
        record["aerosol_tables"]["assets"][0]["sha256"] = "0" * 64
        assert original != "0" * 64
        header["physics_setup_fingerprint"] = restart._json_sha256(setup)

        forged = pathlib.Path(tmp) / "forged.npz"
        payload[header_key] = np.frombuffer(
            json.dumps(header).encode("utf-8"), dtype=np.uint8)
        np.savez(forged, **payload)

        with pytest.raises(restart.RestartMismatchError) as excinfo:
            restart.restore_restart(forged, state, cfg)
        assert "different physics setup" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 3. Legacy RRTMG.
# ---------------------------------------------------------------------------

def test_legacy_rrtmg_declares_mp28_radii_on_wrfs_own_scheme_table():
    """``legacy_scheme_declares_radii(28, *)`` used to raise.

    That made ``mp_physics=28 + ra_rrtmg_variant='rrtmg_legacy'`` fail
    closed with a NotImplementedError telling the operator to extend a
    table -- a refusal WRF itself does not make.  WRF v4.6.1 lists
    THOMPSONAERO explicitly and separately from THOMPSON inside the same
    ``use_mp_re`` disjunction (phys/module_physics_init.F:1005 and :1006),
    and the P3/Jensen-Ishmael ``has_reqs = 0`` override at :1027-1033 does
    not touch it, so all three of has_reqc/has_reqi/has_reqs are 1.
    """
    from gpuwm.core.rrtmg_legacy import (_LEGACY_ICE_ACTIVE_MICROPHYSICS,
                                         _MP_DECLARES_RADII,
                                         legacy_ice_active,
                                         legacy_scheme_declares_radii)

    assert _MP_DECLARES_RADII[28] is True
    assert legacy_scheme_declares_radii(28, 1) is True
    # use_mp_re = 0 still switches the whole table off, exactly as for mp=8.
    assert legacy_scheme_declares_radii(28, 0) is False
    assert legacy_scheme_declares_radii(8, 1) is True

    # cal_cldfra1's F_QI/F_QS: Registry.EM_COMMON:3036 gives thompsonaero
    # the identical ``moist:qv,qc,qr,qi,qs,qg`` inventory :3024 gives
    # thompson, so ice is active for 28 exactly as it is for 8.
    assert legacy_ice_active(28) is True
    assert 28 in _LEGACY_ICE_ACTIVE_MICROPHYSICS
    # Morrison stays out of BOTH tables for its own recorded reasons.
    assert legacy_scheme_declares_radii(10, 1) is False
    assert legacy_ice_active(10) is True


@pytest.mark.parametrize("stock_tree", [
    # Machine-local staging paths are deliberately not part of the
    # contract (PROVENANCE.md): name the tree, let the machine say where.
    pathlib.Path(os.environ.get("GPUWM_WRF461_STOCK_TREE",
                                "wrf-stock-v461-gate-20260721")),
])
def test_the_use_mp_re_citation_is_checkable_against_stock_wrf(stock_tree):
    """The citation, verified rather than asserted -- when the tree is here.

    A line-numbered claim about a reference tree that nobody re-reads is a
    claim about the past.  This one is cheap to re-check, so it is checked.
    """
    source = stock_tree / "phys" / "module_physics_init.F"
    if not source.is_file():                                # pragma: no cover
        pytest.skip(f"stock WRF v4.6.1 tree not present at {stock_tree}")
    lines = source.read_text(encoding="utf-8",
                             errors="replace").splitlines()
    assert "THOMPSON" in lines[1004] and "THOMPSONAERO" not in lines[1004]
    assert "THOMPSONAERO" in lines[1005]
    # Both are inside the use_mp_re block that sets has_reqc/i/s.
    block = "\n".join(lines[986:1024])
    assert "use_mp_re" in "\n".join(lines[986:989])
    assert "has_reqc = 1" in block and "has_reqs = 1" in block


# ---------------------------------------------------------------------------
# 4. The N5S wrfinput inventory.
# ---------------------------------------------------------------------------

def _n5s_cfg(mp_physics):
    from types import SimpleNamespace

    return SimpleNamespace(moist=True, mp_physics=mp_physics)


def test_the_n5s_wrfinput_inventory_has_an_mp28_arm():
    """A 28 arm requiring the Thompson moments plus QNCLOUD/QNWFA/QNIFA.

    All three are declared scalars in WRF's Registry with the wrfinput
    stream in their IO strings (Registry.EM_COMMON:542 for QNCLOUD,
    registry.new3d_wif:87-90 for QNWFA/QNIFA), so real.exe writes them.
    QNCLOUD is REQUIRED here rather than optional: it is optional only for
    Morrison, which diagnoses cloud number, whereas for mp=28 it is the
    prognostic droplet number the entire scheme turns on.
    """
    from gpuwm.verify.cases import real74_n5s as n5s

    required, allowed = n5s._active_moisture_inventory(_n5s_cfg(28))
    assert required == frozenset({
        "QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP",
        "QNRAIN", "QNICE", "QNCLOUD", "QNWFA", "QNIFA"})
    assert allowed == required

    mapping = n5s._active_moisture_map(_n5s_cfg(28))
    assert mapping["QNCLOUD"] == "nc"
    assert mapping["QNWFA"] == "nwfa"
    assert mapping["QNIFA"] == "nifa"

    # mp=8 must be untouched: no aerosol names may leak into its inventory.
    mp8_required, mp8_allowed = n5s._active_moisture_inventory(_n5s_cfg(8))
    assert mp8_required == frozenset({
        "QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP",
        "QNRAIN", "QNICE"})
    assert not ({"QNWFA", "QNIFA", "QNCLOUD"} & mp8_allowed)
    assert "QNWFA" not in n5s._active_moisture_map(_n5s_cfg(8))

    # Morrison keeps its documented OPTIONAL QNCLOUD.
    mp10_required, mp10_allowed = n5s._active_moisture_inventory(
        _n5s_cfg(10))
    assert "QNCLOUD" not in mp10_required
    assert "QNCLOUD" in mp10_allowed


def test_an_mp28_wrfinput_without_qncloud_is_refused():
    """The Morrison optional-QNCLOUD exemption must not extend to mp=28.

    ``_restore_active_moisture`` tolerates a missing wrfinput field only if
    it is in ``MORRISON_OPTIONAL_MOISTURE_WRFINPUT``.  QNCLOUD is in that
    tuple, so before the mp=28 arm existed an aerosol-aware wrfinput with no
    QNCLOUD would have restored quietly and started every column at nc = 0 --
    which WRF's terminal clamp (module_mp_thompson.F:3976) then holds at
    2/rho for the whole run.
    """
    from types import SimpleNamespace

    from gpuwm.verify.cases import real74_n5s as n5s

    shape = (2, 3, 4)
    raw = {name: np.zeros(shape, dtype=np.float32) for name in
           ("QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP",
            "QNRAIN", "QNICE", "QNWFA", "QNIFA")}
    state = SimpleNamespace(**{
        name: np.zeros(shape, dtype=np.float32) for name in
        ("qv", "qc", "qr", "qi", "qs", "qg", "nr", "ni", "nc",
         "nwfa", "nifa")})

    with pytest.raises(ValueError, match="QNCLOUD"):
        n5s._restore_active_moisture(state, raw, _n5s_cfg(28), np)

    # With it present the same restore succeeds and lands on ``nc``.
    raw["QNCLOUD"] = np.full(shape, 1.25e8, dtype=np.float32)
    n5s._restore_active_moisture(state, raw, _n5s_cfg(28), np)
    np.testing.assert_array_equal(state.nc, raw["QNCLOUD"])
    np.testing.assert_array_equal(state.nwfa, raw["QNWFA"])


# ---------------------------------------------------------------------------
# 5. Packaging.
# ---------------------------------------------------------------------------

# argv: <project root> <distribution name> <top-level package directory>.
# Parameterised because the Thompson tables this file measures moved into
# the `gpuwm-data` companion distribution in 2.5.0 -- same repository, a
# second pyproject.toml -- and a probe hard-wired to the root project
# would have reported "not shipped" for files that ship perfectly well,
# one distribution over.
_WHEEL_PROBE = r"""
import json, os, sys, tomllib
from pathlib import Path
repo = Path(sys.argv[1])
name = sys.argv[2]
top = sys.argv[3]
os.chdir(repo)
import setuptools
from setuptools.command.build_py import build_py
with (repo / "pyproject.toml").open("rb") as stream:
    tools = tomllib.load(stream)["tool"]["setuptools"]
skip = {"__pycache__", ".pytest_cache"}
packages = []
for dirpath, dirnames, filenames in os.walk(repo / top):
    dirnames[:] = sorted(d for d in dirnames if d not in skip)
    if "__init__.py" in filenames:
        packages.append(".".join(Path(dirpath).relative_to(repo).parts))
distribution = setuptools.dist.Distribution({
    "name": name, "packages": packages,
    "package_data": tools["package-data"],
    "exclude_package_data": tools.get("exclude-package-data", {}),
})
command = build_py(distribution)
command.finalize_options()
shipped = set()
for _package, src_dir, _build_dir, names in (
        command.get_data_files_without_manifest()):
    for name in names:
        shipped.add(Path(src_dir, name).as_posix())
print(json.dumps(sorted(shipped)))
"""


def _interpreter_with_setuptools() -> str | None:
    candidates = [sys.executable]
    for name in ("python3", "python3.12", "python3.11", "python3.13"):
        found = shutil.which(name)
        if found:
            candidates.append(found)
    candidates.extend(("/usr/bin/python3", "/usr/bin/python3.12"))
    for candidate in dict.fromkeys(candidates):
        if not candidate or not os.path.exists(candidate):
            continue
        probe = subprocess.run(
            [candidate, "-c", "import setuptools, tomllib"],
            capture_output=True)
        if probe.returncode == 0:
            return candidate
    return None


def test_the_ccn_activation_blob_is_present_in_the_built_package_data():
    """DRY-RUN BUILD, not a declaration read.

    ``CCN_ACTIVATE.BIN`` is third-party parcel-model output (WRF's own
    comment at phys/module_mp_thompson.F:5102-5108) that this repository now
    redistributes -- ``AEROSOL_ASSET_REDISTRIBUTED = True``, reversed on
    2026-08-01 once WRF's public-domain dedication was checked against the
    committed blob.  Until then an ``[tool.setuptools.exclude-package-data]``
    entry kept it out of every wheel, so the measurement here is inverted
    rather than deleted: a leftover exclusion, or a reinstated .gitignore
    line, would publish an mp=28 that fails closed on a missing activation
    table for every user.

    The sibling declaration gate reads pyproject with ``tomllib`` and cannot
    tell a typo from an entry; the sibling measurement gate uses setuptools
    and SKIPS in this virtualenv, which has none.  This one finds an
    interpreter that does and measures in a subprocess, so the control is
    live in the environment the suite actually runs in.
    """
    # The COMPANION distribution's wheel since 2.5.0: gpuwm/data/thompson/
    # tables moved to gpuwm-data/gpuwm_data/data/thompson/tables when the
    # gpuwm wheel measured 103.62 MiB against PyPI's 100 MiB cap.  The
    # claim under test is unchanged -- gpuwm redistributes WRF's
    # CCN_ACTIVATE.BIN and a user's install must contain it -- so the probe
    # follows the file to the distribution that now carries it.
    project = REPO / "gpuwm-data"
    relative = "gpuwm_data/data/thompson/tables/CCN_ACTIVATE.BIN"
    interpreter = _interpreter_with_setuptools()
    if interpreter is None:                                 # pragma: no cover
        pytest.skip("no interpreter with setuptools + tomllib is available")

    # encoding named: text=True alone decodes the child's stdout with the
    # host locale (cp1252 on Windows), and this stdout is JSON, which is
    # UTF-8 by specification whatever the console codepage is.
    result = subprocess.run(
        [interpreter, "-c", _WHEEL_PROBE, str(project),
         "gpuwm-data", "gpuwm_data"],
        capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stderr
    shipped = set(json.loads(result.stdout))

    assert relative in shipped, (
        "a wheel built from this tree would NOT contain "
        f"{relative}, which gpuwm declares it redistributes; a stale "
        "[tool.setuptools.exclude-package-data] entry is the usual cause")
    # ...and nothing else moved.  The other Thompson tables that DO ship
    # must still ship, or this change silently broke mp=8.
    for still_shipped in (
            "gpuwm_data/data/thompson/tables/qr_acr_qsV2.dat",
            "gpuwm_data/data/thompson/tables/thompson_aux_tables.dat",
            "gpuwm_data/data/thompson/tables/MANIFEST.sha256"):
        assert still_shipped in shipped, still_shipped
    # The two size-externalized tables stay out, as before.
    for externalized in ("gpuwm_data/data/thompson/tables/freezeH2O.dat",
                         "gpuwm_data/data/thompson/tables/qr_acr_qg_V4.dat"):
        assert externalized not in shipped, externalized

    # The wheel probe globs the FILESYSTEM, so a shipped-but-absent file
    # would pass vacuously.  Say so rather than assume.
    assert (project / relative).is_file(), (
        f"{relative} is missing from this tree, so this run measured an "
        "inclusion against an absent file")


def test_the_offline_child_lane_decision_is_recorded_not_undecided():
    """mp=28 is IN the lane for same-scheme, OUT for every mixed edge.

    Recorded here as one assertion rather than left implicit, because the
    failure mode this whole package exists to remove is a scheme that is
    half-supported by accident.
    """
    from gpuwm.offline_child import (OFFLINE_CHILD_MP_PHYSICS,
                                     _CROSS_SCHEME_REFUSED_MP_PHYSICS)
    from gpuwm.offline_child_run import _CAPABILITIES
    from gpuwm.core import microphysics_transition as mt

    assert 28 in OFFLINE_CHILD_MP_PHYSICS
    assert _CAPABILITIES["same_scheme_mp_physics"] == [6, 8, 10, 18, 28]
    assert _CAPABILITIES["cross_scheme_transitions"] == []
    # The offline refusal set must MIRROR the online one, or a downscale
    # could perform a closure the nest lane refuses.  It is now DERIVED from
    # the online tuple rather than re-spelled: the WDM6 port added 16 online
    # and not here, and the earlier OFFLINE_CHILD_MP_PHYSICS gate that
    # happened to cover the hole is a different guarantee ("unreadable"), so
    # it would have stopped covering it the day the QNCCN row landed.
    assert (set(_CROSS_SCHEME_REFUSED_MP_PHYSICS)
            == set(mt.UNVALIDATED_MIXED_EDGE_SELECTORS) == {16, 28})
    assert 16 not in OFFLINE_CHILD_MP_PHYSICS


# ---------------------------------------------------------------------------
# 6. The offline-child lane, end to end.
# ---------------------------------------------------------------------------

@requires_gpu
def test_an_offline_mp28_child_keeps_its_inherited_surface_emission(tmp_path):
    """The interaction the two halves of this change had to agree on.

    ``thompson_init`` fills ``nwfa2d`` at module_mp_thompson.F:510 ONLY
    inside the "no initial CCN" branch (:493).  An offline child that
    inherits a parent's nonzero ``nwfa`` takes the ``has_CCN = .TRUE.``
    branch at :516-522 instead, which fills NOTHING -- so if the child's
    surface aerosol emission did not arrive by interpolation from the
    parent, the child would run its whole forecast at zero emission and
    ``microphysics_init`` would report success.

    This is the check that the two halves line up: the offline lane SINTs
    QNWFA2D from the parent, and the profile fill then declines to touch it.
    """
    import cupy as cp

    sys.path.insert(0, str(REPO / "tests"))
    from test_offline_child import (  # noqa: E402
        _history, _mp28_child_config, _mp28_placement, _physics_binding)

    from datetime import datetime

    from gpuwm.core import microphysics
    from gpuwm.offline_child import (build_offline_child_domain_state,
                                     interpolate_parent_initial_state)

    parent = tmp_path / "parent-mp28.nc"
    _history(parent, datetime(1974, 4, 3, 12), mp=28, ny=18, nx=20,
             signal=0.5)
    initial = interpolate_parent_initial_state(
        parent, _mp28_placement(),
        physics_binding=_physics_binding(tmp_path, mp=28), backend="cpu")
    child = build_offline_child_domain_state(
        initial, _mp28_child_config(), array_module=cp)
    cp.cuda.Stream.null.synchronize()

    assert float(cp.max(child.nwfa2d)) == pytest.approx(4321.0)
    inherited_nwfa2d = cp.asnumpy(child.nwfa2d).copy()
    inherited_nwfa = cp.asnumpy(child.nwfa).copy()

    receipt = microphysics.microphysics_init(child, _mp28_child_config())
    cp.cuda.Stream.null.synchronize()

    profile = receipt["thompson_aerosol_profile"]
    assert profile["ccn"] is False, (
        "the child inherited a nonzero CCN field, so WRF's has_CCN branch "
        "applies and thompson_init must NOT refill the profile")
    assert profile["in"] is False
    np.testing.assert_array_equal(cp.asnumpy(child.nwfa2d),
                                  inherited_nwfa2d)
    np.testing.assert_array_equal(cp.asnumpy(child.nwfa), inherited_nwfa)


@requires_gpu
def test_an_offline_mp28_child_can_write_its_final_checkpoint(tmp_path):
    """``gpuwm downscale`` ends by writing ``gpuwmrst_dNN_final.npz``.

    Before the identity row existed that raised ``RestartManifestError``
    AFTER the whole child forecast had run, i.e. the run's own evidence was
    destroyed at the last step.  It also ties the two halves of this
    package together: the write now REQUIRES nwfa2d/nifa2d to be present,
    which is exactly what the offline lane had to start interpolating.
    """
    import cupy as cp

    sys.path.insert(0, str(REPO / "tests"))
    from test_offline_child import (  # noqa: E402
        _history, _mp28_child_config, _mp28_placement, _physics_binding)

    from datetime import datetime

    from gpuwm.io import restart
    from gpuwm.offline_child import (build_offline_child_domain_state,
                                     interpolate_parent_initial_state)

    parent = tmp_path / "parent-mp28.nc"
    _history(parent, datetime(1974, 4, 3, 12), mp=28, ny=18, nx=20)
    initial = interpolate_parent_initial_state(
        parent, _mp28_placement(),
        physics_binding=_physics_binding(tmp_path, mp=28), backend="cpu")
    cfg = _mp28_child_config()
    child = build_offline_child_domain_state(initial, cfg, array_module=cp)
    cp.cuda.Stream.null.synchronize()

    path = tmp_path / "gpuwmrst_d01_final.npz"
    restart.write_restart(path, child, cfg)
    assert path.is_file()
    header = restart.read_restart_header(path)
    assert header["physics_setup"]["algorithms"]["microphysics"] == \
        restart.MICROPHYSICS_ALGORITHM_IDENTITIES[28]
    with np.load(path, allow_pickle=False) as stored:
        for name in MP28_AEROSOL_FIELDS:
            assert f"state/{name}" in stored.files, name
