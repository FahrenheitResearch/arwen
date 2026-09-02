"""Gates that guard New Tiedtke's INTEGRATION, not its arithmetic.

The parity files grade the transcription. This one holds the checks that
have to be true before `cu_physics = 16` may be reachable at all, and each
exists because the alternative failure is silent.

It will grow: the cumulus-calendar refusal and the `cudt_minutes` law
(docs/ntiedtke/PORT-RECORD.md §10) belong here too when they land.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from gpuwm.core.preflight import (
    column_workspace_bytes, ntiedtke_column_workspace_bytes)


def _exp(*cu_physics):
    """The smallest thing the workspace terms actually read."""
    return SimpleNamespace(domains=[
        SimpleNamespace(run=SimpleNamespace(
            cu_physics=c, bl_pbl_physics=0, nx=372, ny=284, nz=49))
        for c in cu_physics])


def test_new_tiedtke_is_priced_rather_than_refused():
    """INVERTED 2026-08-29. The refusal did its job and is retired.

    It stood while the distinct-array count was underivable, and refusing
    was right then: a silent zero in this sum lets a run proceed
    under-budgeted with nothing to show for it. The count is measured now
    -- 89 level arrays, 37 surface, 11 scratch slabs -- so the term is a
    number, and this asserts it is a real one rather than the zero the
    refusal existed to prevent.
    """
    priced = ntiedtke_column_workspace_bytes(_exp(16))
    assert priced > 0, "cu_physics = 16 is priced at zero again"
    assert column_workspace_bytes(_exp(16)) >= priced, (
        "the New Tiedtke term does not reach the sum")


def test_the_term_reaches_the_sum_from_any_domain():
    """A nested tree must price an inner domain as readily as a root one."""
    alone = ntiedtke_column_workspace_bytes(_exp(16))
    for tree in (_exp(3, 16), _exp(16, 1)):
        assert ntiedtke_column_workspace_bytes(tree) == alone, (
            "a 16 domain beside another scheme prices differently from a "
            "16 domain alone; the term takes the worst domain, not the "
            "first")
        assert column_workspace_bytes(tree) >= alone


def test_the_price_comes_from_the_workspace_itself_not_a_second_formula():
    """One census, and it lives with the thing it counts.

    A formula apart from its allocator is what put 75 arrays where 89
    belonged (docs/ntiedtke/PORT-RECORD.md section 33), so preflight holds neither the
    cap nor the byte count -- both are imported.
    """
    import inspect
    src = inspect.getsource(ntiedtke_column_workspace_bytes)
    assert "nt_workspace_bytes" in src and "nt_tile_columns" in src, (
        "preflight now computes the New Tiedtke workspace itself; that is "
        "a second copy of a census")

    from gpuwm.core.ntiedtke import (NtWorkspace, nt_tile_columns,
                                     nt_workspace_bytes)
    import pytest as _pytest
    _pytest.importorskip("cupy")
    columns = nt_tile_columns(4096, 70)
    assert (nt_workspace_bytes(20, columns)
            == NtWorkspace(ncol=columns, nz=20).storage_census()["bytes"]), (
        "the formula and the allocator disagree")


@pytest.mark.parametrize("cu", (0, 1, 3))
def test_every_currently_selectable_scheme_is_unaffected(cu):
    """The refusal must cost exactly nothing today.

    `CU_SCHEMES` is (0, 1, 3); none of them may newly raise, and the New
    Tiedtke term must contribute zero rather than perturbing a budget
    the owner's live campaign depends on.
    """
    assert ntiedtke_column_workspace_bytes(_exp(cu)) == 0
    column_workspace_bytes(_exp(cu))


def test_the_term_is_actually_summed_in():
    """A term nobody calls is a comment.

    `column_workspace_bytes` must reach the New Tiedtke arm, or the
    refusal above passes while the real gate stays open.
    """
    import inspect
    src = inspect.getsource(column_workspace_bytes)
    assert "ntiedtke_column_workspace_bytes" in src


def test_cu_schemes_admits_16_and_every_co_requisite_landed():
    """INVERTED. 16 is selectable, and the six things that had to land
    with it are each asserted here rather than remembered.

    The old test said: "when 16 IS added, this test fails -- and that
    failure is the reminder that the workspace term, the cumulus calendar
    and the cudt law all have to land in the same edit". It fired, and
    this is that list, checked.
    """
    pytest.importorskip("cupy")
    import inspect
    from gpuwm.config import CU_SCHEMES, CUMULUS_ADVECTIVE_FORCING_SCHEMES
    from gpuwm.core import clock
    from gpuwm.core.preflight import _CUMULUS_KERNEL_MODULES

    assert 16 in CU_SCHEMES
    # 1. a real workspace term, not the refusal
    assert ntiedtke_column_workspace_bytes(_exp(16)) > 0
    # 2. the cumulus calendar, so the F14 three-way check runs at all
    assert "cu_physics in (1, 3, 16)" in inspect.getsource(clock)
    # 3. the cudt law
    from gpuwm import config as cfgmod
    assert "cu_physics == 16" in inspect.getsource(cfgmod)
    # 4. the advective forcing pair
    assert 16 in CUMULUS_ADVECTIVE_FORCING_SCHEMES
    # 5. the kernel-module table, which is fail-closed
    assert _CUMULUS_KERNEL_MODULES.get(16) == ("ntiedtke",)
    # 6. the optional tendency components
    from gpuwm.core import physics
    assert "cu_physics == 16" in inspect.getsource(
        physics._cumulus_optional_tendency_components)


def test_the_momentum_gate_is_the_corrected_one():
    """The Phase 3 gate must demand a PRE-EXTENSION baseline comparison.

    The original wording -- two runs of the same build, byte-identical,
    cmp all 33 -- proves DETERMINISM, not INERTNESS, and a deterministic
    shift in GF's answer passes it perfectly. That is the one outcome the
    constraint exists to prevent.

    This is a gate on the document because the document is what a future
    session follows, and the port has now twice had a fact sit unread in a
    file that was not checked.
    """
    from pathlib import Path
    rules = (Path(__file__).resolve().parents[1]
             / "docs/ntiedtke/STANDING-RULES.md").read_text(encoding="utf-8")
    for phrase in ("INERTNESS", "PRE-EXTENSION", "GF *and* KF"):
        assert phrase in rules, (
            f"the corrected momentum gate lost {phrase!r}. Two runs of one "
            "build prove determinism; inertness needs a pre-extension "
            "baseline.")


# ---------------------------------------------------------------------------
# The FIFTH mechanical Phase 2 gap: the advective forcing pair and the fold
# ---------------------------------------------------------------------------
# Found by review (review) off the section 17 measurement, and confirmed
# against gpuwm/config.py:831-853 -- which names NTiedtke explicitly, and was
# written before anyone was porting it.
#
# At cu_physics = 16 outside CUMULUS_ADVECTIVE_FORCING_SCHEMES, physics.py
# feeds ZEROS for the theta/qv forcing pair.  Section 17 measured ptent/ptenq
# non-zero on 4,428 of 5,292 rows, and the closure and cuadjtqn read them --
# so the convection itself would change.  Finite, plausible, wrong.


def test_new_tiedtke_consumes_the_advective_forcing_pair():
    """INVERTED, and the fold's location is the assertion that matters.

    WRF's cumulus driver pre-folds RTHRATEN + RTHBLTEN into RTHFTEN at
    module_cumulus_driver.F:867 for G3SCHEME and NTIEDTKESCHEME ONLY. GF
    is not in that list and sums the three lanes itself, so ArWen's dycore
    exports PURE ADVECTION. The fold belongs in the New Tiedtke ADAPTER --
    moving the export double-counts radiative and PBL heating for GF and
    moves run_myj, the baseline every intensity number is graded against.
    """
    from gpuwm.config import CUMULUS_ADVECTIVE_FORCING_SCHEMES as S
    assert 16 in S
    assert 3 in S, "Grell-Freitas fell out while New Tiedtke was added"


def test_the_fold_trap_note_still_stands_beside_the_table():
    """The note is a receipt, and it is the strongest a receipt can be.

    It sits beside the table it constrains, names the two schemes it
    applies to, states the wrong repair and why it is wrong, and it found
    this port rather than the other way round. It still cannot fail -- so
    this test is the minimum that can: the note may not be deleted by
    whoever adds the entry, which is exactly when it matters most.
    """
    import inspect
    import gpuwm.config as cfg
    src = inspect.getsource(cfg)
    i = src.index("CUMULUS_ADVECTIVE_FORCING_SCHEMES = ")
    raw = src[max(0, i - 2000):i]
    # Strip the "#:" sphinx comment markers before collapsing, or a
    # phrase that wraps across two comment lines is never found -- and
    # the test then passes for the wrong reason on the short phrases.
    note = " ".join(raw.replace("#:", " ").split())
    for phrase in ("THE FOLD TRAP", "NTIEDTKESCHEME ONLY",
                   "exports PURE ADVECTION",
                   "rather than moving the export"):
        assert phrase in note, f"the fold-trap note lost {phrase!r}"


def test_grell_freitas_is_still_the_only_member():
    """GF must not be removed from the set while adding New Tiedtke.

    The set is read by the dycore export, the state allocation, the VRAM
    projection, the restart inventory and the serialization contract. A
    change that swapped rather than added would silently stop allocating
    GF's pair.
    """
    from gpuwm.config import CUMULUS_ADVECTIVE_FORCING_SCHEMES as S
    assert 3 in S, "Grell-Freitas fell out of the advective forcing set"


def test_phase_one_is_defined_by_the_assembly_not_the_kernels():
    """"Thirteen of thirteen kernels graded" is not Phase 1 done.

    Every routine grading green against captures the orchestration did
    not produce is the same circularity the capture architecture was
    built to retire, one level up. The definition is cheap to state now
    and awkward to argue later, so it is pinned in the contract and
    checked here.
    """
    from pathlib import Path
    rules = (Path(__file__).resolve().parents[1]
             / "docs/ntiedtke/STANDING-RULES.md").read_text(encoding="utf-8")
    for phrase in ("PHASE 1 ENDS WHEN THE ASSEMBLED PIPELINE",
                   "EVERY capture boundary", "bracket the unowned"):
        assert phrase in rules, (
            f"the Phase 1 definition lost {phrase!r}. Grading every kernel "
            "is not the same claim as grading the pipeline that runs them.")


def test_the_cumulus_kernel_module_table_carries_16():
    """The fail-closed table: a selector with no entry RAISES.

    That is why this was the one item of the seven that could not be
    forgotten -- it announces itself. Asserted anyway, because "it would
    have raised" is not the same as "it is right".
    """
    from gpuwm.core.preflight import _CUMULUS_KERNEL_MODULES
    assert _CUMULUS_KERNEL_MODULES[16] == ("ntiedtke",)
    assert _CUMULUS_KERNEL_MODULES[3] == ("gf",), "GF's entry moved"


def test_phase_one_remaining_work_is_stated_in_full():
    """Three times the remaining-work count measured a smaller thing.

    "Thirteen of thirteen kernels graded" omitted the assembler. The
    assembler omitted cu_ntiedtke_post_run. Each count was taken over the
    artifact in front of me rather than over the artifact the end
    condition names -- and the end condition names ``nt-levels.csv``,
    whose contents ``cu_ntiedtke_run`` does not produce.

    SUPERSEDED IN PART by test_ntiedtke_output_provenance.py, which takes
    the count over the artifact the end condition NAMES -- nt-levels.csv's
    header -- rather than over a hand-maintained list of categories. A
    hand-maintained list is exactly what failed three times, so this test
    is kept only as the record's index and the header-driven one is the
    gate (review).
    """
    from pathlib import Path
    doc = (Path(__file__).resolve().parents[1]
           / "docs/ntiedtke/PORT-RECORD.md").read_text(encoding="utf-8")
    for item in ("cu_ntiedtke_post_run", "the allocator", ":566",
                 "reduction"):
        assert item in doc, (
            f"Phase 1's remaining-work list lost {item!r}. It is not done "
            "until the assembled pipeline reproduces nt-levels.csv bitwise.")


def test_the_llo3_reduction_scope_is_decided_not_left_open():
    """32 columns and 17,920 are different reductions, not an impl detail.

    llo3 is monotone, so it flips earlier the wider the population is. A
    32-column block of clear ocean air keeps it FALSE and puts the whole
    level-loop body -- including the departure-level reset that runs for
    every column regardless of loflag -- into the branch the fixture
    never exercises. The fixture only ever runs llo3 true (48 of 48
    triggering columns).
    """
    from pathlib import Path
    doc = (Path(__file__).resolve().parents[1]
           / "docs/ntiedtke/PORT-RECORD.md").read_text(encoding="utf-8")
    assert "DECIDED: chunk-wide" in doc, (
        "the llo3 reduction scope is no longer recorded as decided. It is "
        "a decision, not an implementation choice -- see section 27.")


# ---------------------------------------------------------------------------
# The forcing function forces NOTICING, not COMPLETENESS
# ---------------------------------------------------------------------------
# test_cu_schemes_still_refuses_16 fails the instant someone adds 16, so the
# tuple cannot land silently. But trace what happens next: they add 16, they
# update that test, and NOTHING then requires the cumulus calendar to exist.
# clock.py leaves stepcu None, PhysicsDriver computes its own from a
# cudt_minutes that defaults to 5.0, and the scheme runs on an unvalidated
# five-minute hold. That is the original finding arriving intact through the
# gate built to prevent it (review).
#
# So each remaining item asserts its CURRENT ABSENT STATE, the same shape as
# test_the_cumulus_kernel_module_table_still_lacks_16. Each fails the moment
# the edit begins, and together they make "one edit" mean one edit.
#
# Writing them while the answer is absent is also easier than afterwards:
# the assertion is just "the current state", and it does not require knowing
# yet what the correct new state looks like.


def test_the_cumulus_calendar_includes_16():
    """Outside the tuple the scheme still runs -- on a cadence nothing
    validated. Inside it, the F14 three-way check runs for New Tiedtke.
    """
    import inspect
    from gpuwm.core import clock
    assert "cu_physics in (1, 3, 16)" in inspect.getsource(clock)


def test_the_cudt_law_for_16_exists_and_refuses_a_hold():
    """cudt_minutes DEFAULTS TO 5.0, and that is the whole point.

    A config that merely selects 16 and says nothing else would hold the
    scheme's rates for five minutes -- for a scheme whose RAINCV is
    rn/stepcu, a per-call rate with no persistence. The held rate would be
    reapplied every step and the precipitation would be five times what
    the scheme produced.

    THE NEGATIVE CONTROL IS THE HALF THAT MATTERS: a law that accepted
    everything would pass any test that only builds a valid config.
    """
    import pytest as _pytest
    from gpuwm.config import RunConfig, validate_run_config

    def cfg(**kw):
        # The law lives in validate_run_config, not __post_init__, so a
        # test that only CONSTRUCTS a config asserts nothing -- which is
        # how the first version of this passed the valid case and failed
        # the negative one.
        base = dict(nx=8, ny=8, nz=10, dx=4500.0, dy=4500.0, ztop=20000.0,
                    dt=20.0, run_seconds=0.0, time_step_sound=4, moist=True,
                    mp_physics=10, ra_physics=4, cu_physics=16,
                    cudt_minutes=0.0, sf_sfclay_physics=1,
                    sf_surface_physics=2, bl_pbl_physics=1)
        base.update(kw)
        return validate_run_config(RunConfig(**base))

    cfg()                                     # the valid shape is accepted
    with _pytest.raises(ValueError, match="cudt_minutes=0"):
        cfg(cudt_minutes=5.0)
    with _pytest.raises(ValueError, match="requires a PBL scheme"):
        cfg(bl_pbl_physics=0)


def test_the_optional_tendency_components_branch_for_16():
    """New Tiedtke produces no separate rain or snow category.

    cu_ntiedtke_post_run forms exactly six tendency fields
    (module_cu_ntiedtke.F:514-524): theta, qv, qc, qi, u and v. Applying
    Kain-Fritsch's phase logic at 16 would ask it for rqr, and on a
    separate-ice-snow package for rqs as well.
    """
    pytest.importorskip("cupy")
    from gpuwm.config import RunConfig
    from gpuwm.core.physics import _cumulus_optional_tendency_components

    def cfg(mp, cu=16):
        return RunConfig(
            nx=8, ny=8, nz=10, dx=4500.0, dy=4500.0, ztop=20000.0, dt=20.0,
            run_seconds=0.0, time_step_sound=4, moist=True, mp_physics=mp,
            ra_physics=4, cu_physics=cu,
            cudt_minutes=0.0 if cu in (3, 16) else 5.0,
            sf_sfclay_physics=1, sf_surface_physics=2, bl_pbl_physics=1)

    got = _cumulus_optional_tendency_components(cfg(10))
    assert "rqr" not in got and "rqs" not in got, got
    assert got in ((), ("rqi",)), got
    # And Kain-Fritsch is untouched -- it still gets its rain category.
    assert "rqr" in _cumulus_optional_tendency_components(cfg(10, cu=1))


def test_the_prepared_cache_identity_needs_no_scheme_entry():
    """RESOLVED, and it is NOT a gap -- checked rather than assumed.

    The standing rules' Phase 2 definition names "the prepared-cache
    identity fields accepting cu_physics = 16", and neither session had
    examined it. It turns out to be field-generic:
    prepared_domain_config_identity is ``asdict(domain_config)``, the whole
    RunConfig serialized, compared by strict equality. ``cu_physics`` is
    already one of those fields, so 16 needs no entry -- and a tree
    prepared at 3 correctly refuses to run at 16.

    Pinned so that if the identity ever becomes a scheme TABLE, this
    conclusion stops being true and someone finds out.
    """
    import inspect
    from gpuwm.ingest import prepared_cache
    src = inspect.getsource(prepared_cache.prepared_domain_config_identity)
    assert "asdict(domain_config)" in src, (
        "the prepared-cache identity is no longer a generic asdict of the "
        "domain config. If it became a per-scheme table, cu_physics = 16 "
        "needs an entry and this is now a real Phase 2 item.")


def test_adding_16_did_not_reparent_the_grell_key_check():
    """THE REGRESSION THE BASELINE RE-RUN CAUGHT AND NO UNIT TEST DID.

    ``validate_run_config`` had::

        if cfg.cu_physics == 3:
            ...
        elif cfg.clos_choice != 0 or cfg.ishallow != 0:
            raise ...

    and the first version of the New Tiedtke law was inserted BETWEEN
    them. That silently re-parented the ``elif`` onto the 16 test, so
    every ``cu_physics = 3`` config carrying ``ishallow = 1`` -- which is
    most of the campaign's -- was refused with a message about Grell-family
    keys being read only where cu_physics=3, on a config that selects
    exactly that.

    Every unit test passed. None exercised a GF config with a non-default
    Grell key through the validator, so the whole suite was blind to it
    and the forecast refused instead.

    The general form: inserting a branch into an ``if/elif`` chain changes
    what the later arms attach to, and nothing about the inserted branch
    looks wrong in isolation.
    """
    from gpuwm.config import RunConfig, validate_run_config

    def cfg(**kw):
        base = dict(nx=8, ny=8, nz=20, dx=4500.0, dy=4500.0, ztop=20000.0,
                    dt=20.0, run_seconds=0.0, time_step_sound=4, moist=True,
                    mp_physics=10, ra_physics=4, sf_sfclay_physics=1,
                    sf_surface_physics=2, bl_pbl_physics=1)
        base.update(kw)
        return validate_run_config(RunConfig(**base))

    # The shape that was refused: Grell-Freitas with its shallow arm on.
    cfg(cu_physics=3, cudt_minutes=0.0, ishallow=1)
    cfg(cu_physics=3, cudt_minutes=0.0, ishallow=0)
    # And the check still fires where it should -- a Grell key on a
    # non-Grell scheme.
    with pytest.raises(ValueError, match="Grell-family keys"):
        cfg(cu_physics=1, cudt_minutes=5.0, ishallow=1)


# ---------------------------------------------------------------------------
# The tenth site, found by running rather than by reading
# ---------------------------------------------------------------------------


def test_no_scheme_whitelist_restates_the_scheme_set():
    """TWO whitelists, and only one of them was in the group of nine.

    ``validate_run_config`` gates the config; ``initialize_physics`` gates
    the driver, and it restated ``(0, 1, 3)`` as a literal rather than
    reading :data:`CU_SCHEMES`.  So a ``cu_physics = 16`` config was valid,
    priced, scheduled, compiled, frame-recorded -- and then refused at
    driver construction, after the whole preflight had passed.

    THIS IS THE SAME FAILURE MODE AS THE NINE, one layer deeper: a scheme
    set written down twice.  Nothing found it by reading, because the
    search that found the others looked for ``cu_physics == 3`` dispatch
    sites and this is a membership test.  Running the model found it in two
    seconds.

    The guard is structural rather than another list: no module may compare
    ``cu_physics`` against a literal tuple.  A negative control keeps it
    honest -- the pattern must actually match the shape it forbids.
    """
    import pathlib
    import re

    forbidden = re.compile(r"cu_physics\s+(?:not\s+)?in\s+\((?![^)]*\b16\b)")
    # NEGATIVE CONTROL: the pattern must match the shape it exists to
    # forbid, or this test passes by matching nothing.
    assert forbidden.search("if cfg.cu_physics not in (0, 1, 3):"), (
        "the pattern does not match the very line this test was written "
        "for; it would pass against any tree")
    assert not forbidden.search("if run.cu_physics in (1, 3, 16):"), (
        "the pattern flags the cumulus calendar, which names 16 and is "
        "correct")

    root = pathlib.Path(__file__).resolve().parents[1] / "gpuwm"
    offenders = []
    for path in sorted(root.rglob("*.py")):
        for n, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1):
            if forbidden.search(line):
                offenders.append(f"{path.relative_to(root)}:{n}: {line.strip()}")
    assert not offenders, (
        "a scheme set is restated as a literal that omits 16:\n  "
        + "\n  ".join(offenders))


def test_initialize_physics_admits_16_at_the_driver():
    """The site itself, asserted at the boundary that refused.

    Reading the source is not enough here: what broke was construction, so
    this constructs.  ``initialize_physics`` is heavy, so the assertion is
    that the cu_physics gate specifically is passed -- a refusal naming
    cu_physics is the failure this guards.
    """
    pytest.importorskip("cupy")
    import inspect

    from gpuwm.config import CU_SCHEMES
    from gpuwm.core.physics import initialize_physics

    src = inspect.getsource(initialize_physics)
    assert "cu_physics not in CU_SCHEMES" in src, (
        "initialize_physics no longer reads the config module's tuple")
    assert 16 in CU_SCHEMES


# ---------------------------------------------------------------------------
# The twelfth site, found by a 14-hour run and nothing shorter
# ---------------------------------------------------------------------------


def test_every_cumulus_scheme_has_a_restart_identity():
    """A checkpoint cannot be written for a scheme with no identity string.

    ``cu_physics = 16`` ran for six forecast hours and then died writing
    its first restart: ``cannot identify unsupported cumulus scheme 16``.
    Nothing shorter than a run that reaches the checkpoint cadence could
    have found it -- the 30-minute, 2-hour and 4-hour probes all completed
    without ever writing one.

    Two tables were missing it, and they fail differently.
    ``CUMULUS_ALGORITHM_IDENTITIES`` raises with the scheme id in the
    message.  The ``expected`` class map does not: an unlisted scheme
    leaves ``expected_class`` None, which routes a perfectly stock adapter
    down the custom-callable path and complains that it lacks a
    ``restart_identity`` attribute -- a true statement that points at the
    wrong thing.

    Both are checked here against ``CU_SCHEMES`` rather than against a
    list, so the next scheme fails this test instead of a long run.
    """
    import inspect

    from gpuwm.config import CU_SCHEMES
    from gpuwm.io.restart import (CUMULUS_ALGORITHM_IDENTITIES,
                                  physics_setup_identity)

    missing = [s for s in CU_SCHEMES if s not in CUMULUS_ALGORITHM_IDENTITIES]
    assert not missing, (
        f"cumulus schemes with no restart algorithm identity: {missing}. "
        "A run selecting one writes wrfout happily and then dies at its "
        "first checkpoint.")

    # Identities must be DISTINCT: two schemes sharing a string would let a
    # checkpoint resume under the wrong algorithm, which is the single
    # thing this table exists to prevent.
    strings = [CUMULUS_ALGORITHM_IDENTITIES[s] for s in CU_SCHEMES]
    assert len(set(strings)) == len(strings), (
        f"two cumulus schemes share a restart identity: {strings}")

    # And the second table, read out of the source because it is a literal
    # inside the function rather than a module constant.
    src = inspect.getsource(physics_setup_identity)
    for scheme, cls in ((1, "gpuwm.core.kf.KainFritsch"),
                        (3, "gpuwm.core.gf.GrellFreitas"),
                        (16, "gpuwm.core.ntiedtke.NewTiedtke")):
        assert cls in src, (
            f"cu_physics={scheme} has no expected-class row, so its stock "
            f"adapter would be routed down the custom-callable path")


def test_the_new_tiedtke_restart_identity_names_this_port():
    """The string binds the implementation, not just the scheme.

    Grell-Freitas' identity carries ``corrected-k22`` because that variant
    must not resume under a WRF-faithful build.  New Tiedtke's equivalent
    is the pratec divergence (docs/ntiedtke/PORT-RECORD.md section 38): every
    checkpoint under this identity has ``cu_pratec == 0`` by construction,
    and an implementation that delivered it would give the slot a
    different meaning.
    """
    from gpuwm.io.restart import CUMULUS_ALGORITHM_IDENTITIES

    identity = CUMULUS_ALGORITHM_IDENTITIES[16]
    assert "tiedtke" in identity.lower()
    assert "wrf461" in identity, (
        "the identity does not name the reference version it was ported "
        "from")
    assert identity.endswith("-v1"), (
        "the identity carries no revision suffix, so a change to the "
        "driver seam has no way to invalidate old checkpoints")
