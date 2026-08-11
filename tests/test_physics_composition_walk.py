"""Is gpuwm's physics arbitrary?  Measured, not argued.

The claim under test is that a user composes microphysics, PBL, surface
layer, land surface, radiation, cumulus and turbulence freely -- that there
is no engine lock quietly pinning a run to one of the shipped named suites.
It has been asserted in review threads more than once and believed by
nobody, including the people who wrote it, because the evidence for it was
always "read this file".

So it is measured.  ``tools/report_physics_composition_walk.py`` writes
9781 physics combinations into real experiment TOMLs and pushes every one
through :func:`gpuwm.experiment.build_experiment` -- the single front door
``gpuwm run``, ``gpuwm go``, ``gpuwm check``, both prepared runners and the
DA drivers reach a per-domain ``RunConfig`` through -- and records what
that call did.  This file pins the properties of the result:

* every admitted value of every axis appears in an ACCEPTED run, so no axis
  is decorative;
* the shipped presets are a tiny corner of what is admitted, so the presets
  are not the space;
* every ACCEPTED run kept every switch the file set, so an admission is
  never a silent rewrite;
* every refusal names why, and DOING WHAT THE REFUSAL SAYS produces an
  accepted run -- the property that caught the one defect this walk found
  (the ``km_opt=2`` message used to recommend ``km_opt=3``, which is
  refused for the same reason three lines above it);
* the axis tables are what the loader actually admits, measured by
  scanning integers at it rather than transcribed from reading
  ``gpuwm/config.py``;
* and MYNN -- the suite family that started this -- composes with every
  land surface and every radiation pairing the loader admits at all.

Two controls keep the whole thing from being vacuous: the walk's window is
checked to contain no local night (or the 1.7.1 nocturnal guard would be
answering a different question), and a mutation control shows the harness
can still produce a refusal when one is warranted.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import report_physics_composition_walk as walk

MODEL = Path(__file__).resolve().parents[1]
RECEIPT = (
    MODEL / "docs" / "public" / "receipts" / "physics-composition-walk.json"
)


@pytest.fixture(scope="module")
def receipt() -> dict:
    return json.loads(RECEIPT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def regenerated() -> dict:
    """One walk, shared by every test in the file (9781 loader calls)."""

    return walk.evaluate()


# ---------------------------------------------------------------------------
# The receipt is the measurement, and it is current.
# ---------------------------------------------------------------------------

def test_the_receipt_regenerates_byte_for_byte(regenerated) -> None:
    assert walk.render(regenerated) == RECEIPT.read_bytes(), (
        "docs/public/receipts/physics-composition-walk.json no longer "
        "matches what the loader does; regenerate it with "
        "tools/report_physics_composition_walk.py and land both together")


# ---------------------------------------------------------------------------
# Controls first: an instrument that cannot fail measures nothing.
# ---------------------------------------------------------------------------

def test_the_walk_window_contains_no_local_night(receipt) -> None:
    """The controlled variable, checked with the shipped scanner.

    ``build_experiment`` refuses an undeclared longwave-off/shortwave-on
    suite for any window containing local night.  That guard is correct and
    is pinned elsewhere; here it must stay OUT of the measurement, or a
    time-of-day refusal would be counted as a composition refusal.
    """

    assert receipt["harness"]["window_contains_local_night"] is False


def test_the_walk_grid_is_inside_every_shipped_vertical_bound() -> None:
    """The other harness control, stated as the bounds and not the number.

    An earlier draft ran nz=8 and reported cu_physics=3 as never admitted.
    That was the harness: Grell-Freitas declares a 12-level floor.  Pinning
    the INTERSECTION rather than the literal 40 means a bound that moves
    fails here instead of quietly re-introducing the artefact.
    """

    from gpuwm import physics_vertical_contract as bounds
    from gpuwm.config import SASE_MAX_NZ

    nz = walk.GRID["nz"]
    named = {
        name: getattr(bounds, name) for name in dir(bounds)
        if name.endswith("_VERTICAL_LEVEL_BOUNDS")
    }
    assert named, "no vertical bounds found to check the walk grid against"
    for name, (low, high) in named.items():
        assert low is None or nz >= low, f"{name} floor {low} > nz {nz}"
        assert high is None or nz <= high, f"{name} ceiling {high} < nz {nz}"
    assert nz <= SASE_MAX_NZ


def test_a_mutation_control_still_produces_a_refusal() -> None:
    """The harness can still refuse; it is not accepting everything.

    A value no schema admits must come back REFUSED with the axis named.
    Without this, "746 accepted" could mean the walk stopped checking.
    """

    combination = dict(walk.ANCHORS["ysu-mm5-noah-rrtmgp"], **walk.TIER_A_HELD)
    combination["mp_physics"] = 55
    outcome = walk.attempt(combination)
    assert outcome.verdict == "REFUSED"
    assert "mp_physics must be" in outcome.message


def test_every_anchor_suite_is_itself_accepted(receipt) -> None:
    """Tiers B, C and D are only coverage if their backgrounds load."""

    for name, row in receipt["anchors"].items():
        assert row["verdict"] == "ACCEPTED", (
            f"anchor {name} is refused, so every tier held against it "
            f"measured nothing: {row['message']}")


# ---------------------------------------------------------------------------
# The headline: the space is large, and it is not the preset list.
# ---------------------------------------------------------------------------

def test_the_walk_covers_what_it_says_it_covers(receipt) -> None:
    totals = receipt["totals"]
    assert totals["tried"] == totals["accepted"] + totals["refused"]
    assert totals["tried"] >= 6000, (
        "the walk shrank; a composition claim this size needs the space "
        "actually walked")
    tiers = receipt["tiers"]
    # Tier A is the full cartesian over the six coupled axes.  Spelled out
    # from the axis tables so a widened axis cannot silently shrink it.
    expected = 1
    for name in ("bl_pbl_physics", "sf_sfclay_physics", "sf_surface_physics",
                 "ra_lw_physics", "ra_sw_physics", "km_opt"):
        expected *= len(walk.AXIS_VALUES[name])
    assert tiers["A-nexus-cartesian"]["tried"] == expected


def test_every_admitted_value_of_every_axis_reaches_an_accepted_run(
        receipt) -> None:
    """No axis value is decorative, and now there is no exception.

    ``ra_lw_physics=1`` used to be the one schema-legal, deliberately
    unexecutable value in the whole space: the WRF RRTM longwave kernels
    were not ported, so the schema advertised a choice the engine did not
    offer.  The 1.9 port makes it executable, so the exception set is
    EMPTY and every admitted value of every axis reaches an accepted run.

    The assertion is kept as an equality against the empty set rather than
    deleted.  An empty exception set is the strong claim, and a future
    scheme that lands schema-legal but unrunnable has to come back here
    and say so out loud.
    """

    unreachable = {
        f"{axis}={value}"
        for axis, values in receipt["per_axis"].items()
        for value, counts in values.items()
        if counts["accepted"] == 0
    }
    assert unreachable == set(), (
        "the set of admitted-but-unreachable selector values changed: "
        f"{sorted(unreachable)}")
    # Non-vacuous: the value that used to be the exception now runs, and
    # the pairing the old refusal named as unreachable is accepted.
    combination = dict(walk.ANCHORS["ysu-mm5-noah-rrtmgp"], **walk.TIER_A_HELD)
    combination.update(ra_lw_physics=1, ra_sw_physics=1)
    outcome = walk.attempt(combination)
    assert outcome.verdict == "ACCEPTED", outcome.message


def test_the_shipped_presets_are_a_corner_of_the_admitted_space(
        receipt) -> None:
    """The point of the whole exercise, as a number.

    If the accepted space were the preset list, "compose your own physics"
    would be marketing.  It is not: the walk accepts hundreds of distinct
    suites, and it walked a deliberately narrow slice of the full cartesian.
    """

    from gpuwm.physics_compat import SINGLE_DOMAIN_PHYSICS_PROFILES

    accepted = receipt["totals"]["accepted"]
    presets = len(SINGLE_DOMAIN_PHYSICS_PROFILES)
    assert accepted > 10 * presets, (
        f"only {accepted} accepted suites against {presets} shipped "
        "presets; the composition claim no longer has room to be true")
    assert receipt["totals"]["full_eight_axis_cartesian"] > accepted


def test_an_accepted_run_never_rewrites_a_switch(receipt) -> None:
    """ACCEPTED has to mean the file got what it asked for.

    A loader that admits a configuration and then substitutes a switch
    would pass every count above while still being an engine lock.  The
    walk compares each accepted run's resolved per-domain RunConfig against
    the eight selectors the file set.
    """

    assert receipt["switch_rewrites"] == []
    assert receipt["totals"]["accepted_with_a_rewritten_switch"] == 0


# ---------------------------------------------------------------------------
# Refusals: every one names why, and its own advice works.
# ---------------------------------------------------------------------------

def test_every_refusal_names_the_selector_it_is_about(receipt) -> None:
    """A refusal that does not name a switch cannot be acted on."""

    vocabulary = set(walk.AXIS_NAMES) | {
        "num_soil_layers", "km_opt_zero_acknowledgement", "moist",
        "cudt_minutes", "nz"}
    for row in receipt["refusal_rules"]:
        for message in row["messages"]:
            named = {word for word in vocabulary if word in message["message"]}
            assert named, (
                "this refusal names no selector, so a user cannot tell what "
                f"to change: {message['message']!r}")


def test_every_refusal_rule_has_a_remedy_and_the_remedy_works(
        receipt) -> None:
    """The property that found the defect, kept as a standing gate.

    A refusal is only as good as the next thing it tells you to do.  Every
    distinct rule the walk produces carries an explicit before/after pair
    in the tool, and the after is built by following the message.  A new
    refusal cannot land without someone demonstrating that its advice
    reaches an accepted run.
    """

    remedies = {row["id"]: row for row in receipt["remedy_follow_through"]}
    for row in remedies.values():
        assert row["before"]["verdict"] == "REFUSED", row["id"]
        assert row["after"]["verdict"] == "ACCEPTED", (
            f"{row['id']}: doing what the refusal says still fails -- "
            f"{row['after']['message']}")
        assert row["remedy_works"], row["id"]

    covered = {row["before"]["rule"] for row in remedies.values()}
    observed = {row["rule"] for row in receipt["refusal_rules"]}
    assert observed - covered == set(), (
        "these refusal rules have no remedy pair, so nobody has shown that "
        f"their advice works: {sorted(observed - covered)}")
    assert covered - observed == set(), (
        "these remedy pairs no longer correspond to a rule the walk "
        f"produces and are dead weight: {sorted(covered - observed)}")


def test_the_coupled_adapter_refusal_names_the_keys_and_the_values() -> None:
    """The other defect the walk found, pinned.

    This is the most frequent refusal in the entire space -- 1704 of the
    5784 refusals, roughly one in three -- and it used to read, in full,
    "RTE+RRTMGP (4) and analytic radiation (90) are coupled LW/SW adapters
    and must be selected on both components".  No config key, no offending
    value, no remedy: a user staring at ra_lw_physics=4, ra_sw_physics=0
    was told about two schemes and left to work out which switch to move.
    Every other refusal in this tree names its selector; this one now does
    too.
    """

    base = dict(walk.ANCHORS["ysu-mm5-noah-rrtmgp"], **walk.TIER_A_HELD)
    outcome = walk.attempt(dict(base, ra_lw_physics=4, ra_sw_physics=0))
    assert outcome.verdict == "REFUSED"
    assert "ra_lw_physics=4" in outcome.message
    assert "ra_sw_physics=0" in outcome.message
    # And the remedy it gives reaches an accepted run.
    assert walk.attempt(
        dict(base, ra_lw_physics=4, ra_sw_physics=4)).verdict == "ACCEPTED"


def test_the_prognostic_tke_refusal_no_longer_recommends_a_refused_value(
) -> None:
    """The defect this walk found, pinned so it cannot come back.

    ``km_opt=2`` used to be refused with "select the LES topology or
    km_opt 3/4".  With a PBL scheme on, ``km_opt=3`` is refused by the
    branch immediately above for exactly the same PBL-off gate, so the
    advice sent a user from one refusal to another.  Only ``km_opt=4``
    works, and now that is what the message says.
    """

    base = dict(walk.ANCHORS["ysu-mm5-noah-rrtmgp"], **walk.TIER_A_HELD)
    refused = walk.attempt(dict(base, km_opt=2))
    assert refused.verdict == "REFUSED"
    assert "km_opt=4" in refused.message
    # The measurement behind the wording: 3 does not work here, 4 does.
    assert walk.attempt(dict(base, km_opt=3)).verdict == "REFUSED"
    assert walk.attempt(dict(base, km_opt=4)).verdict == "ACCEPTED"


# ---------------------------------------------------------------------------
# The axis tables are measured, not transcribed.
# ---------------------------------------------------------------------------

def test_each_axis_admits_exactly_the_values_the_walk_declares(
        receipt) -> None:
    """Validate the instrument.

    The axis tables in the tool are a transcription of ``gpuwm/config.py``,
    and a transcription can be wrong or stale.  Tier E offers the loader
    every integer in [-2, 99] plus 900 and 901 on each axis and classifies
    each by whether the refusal is that axis's own schema message.  What
    the loader admits and what the tool claims must be the same set.
    """

    for axis in receipt["axes"]:
        assert axis["measured_admitted_values"] == axis["admitted_values"], (
            f"{axis['name']}: the loader admits "
            f"{axis['measured_admitted_values']} but the walk declares "
            f"{axis['admitted_values']} ({axis['authority']})")


def test_the_declared_axes_match_the_exported_scheme_tables() -> None:
    """The half of the transcription that config.py does export."""

    from gpuwm.config import (CU_SCHEMES, LAND_SURFACE_SCHEMES, PBL_SCHEMES,
                              SURFACE_LAYER_SCHEMES)

    assert walk.AXIS_VALUES["bl_pbl_physics"] == tuple(PBL_SCHEMES)
    assert walk.AXIS_VALUES["sf_sfclay_physics"] == tuple(
        SURFACE_LAYER_SCHEMES)
    assert walk.AXIS_VALUES["sf_surface_physics"] == tuple(
        LAND_SURFACE_SCHEMES)
    assert walk.AXIS_VALUES["cu_physics"] == tuple(CU_SCHEMES)


def test_nothing_is_dropped_without_being_written_down(receipt) -> None:
    """Skips are logged, with a reason, or they are not skips."""

    assert receipt["skipped"], "a tiered walk that logs no skip is lying"
    for entry in receipt["skipped"]:
        assert entry["what"] and entry["why"]
    assert receipt["coverage_rule"]


# ---------------------------------------------------------------------------
# MYNN, the family this whole wave is about.
# ---------------------------------------------------------------------------

def test_mynn_composes_with_every_land_surface(receipt) -> None:
    from gpuwm.config import LAND_SURFACE_SCHEMES

    assert (receipt["mynn_slice"]["land_surface_options_accepted"]
            == sorted(LAND_SURFACE_SCHEMES))


def test_mynn_composes_with_every_radiation_pairing_the_loader_admits(
        receipt) -> None:
    """"Every admitted radiation option" stated exactly.

    Radiation is selected as a PAIR, and the loader admits five pairings
    in total: both off, Dudhia shortwave with longwave off, WRF's classic
    RRTM/Dudhia pair, RTE+RRTMGP on both, and the analytic proxy on both.
    (The 4/90 and 90/4 crossings are refused because the two are coupled
    adapters, and LW=1 pairs only with SW=1, the combination WRF itself
    ships it as.)  MYNN reaches all five -- including the three that carry
    longwave, which is the whole point of the radiation-bearing MYNN
    presets that landed beside this walk.

    1/1 is new at 1.9.  Before the RRTM longwave port, ra_lw_physics=1
    was schema-legal and unrunnable, so this set had four members and the
    longwave-bearing subset had two.
    """

    accepted_anywhere = {
        f"{lw}/{sw}"
        for lw in walk.AXIS_VALUES["ra_lw_physics"]
        for sw in walk.AXIS_VALUES["ra_sw_physics"]
        if any(
            key.endswith(f".lw{lw}.sw{sw}") and verdict == "ACCEPTED"
            for key, verdict in receipt["mynn_matrix"].items())
    }
    assert set(receipt["mynn_slice"]["radiation_options_accepted"]) == \
        accepted_anywhere
    assert accepted_anywhere == {"0/0", "0/1", "1/1", "4/4", "90/90"}
    longwave_on = {pair for pair in accepted_anywhere
                   if pair.split("/")[0] not in ("0",)}
    assert longwave_on == {"1/1", "4/4", "90/90"}


def test_the_mynn_matrix_is_the_same_in_every_land_surface_column(
        receipt) -> None:
    """Independence, which is what "arbitrary composition" actually means.

    The radiation verdict must not depend on which land-surface model is
    selected.  If the two axes were coupled anywhere, these four columns
    would differ.
    """

    from gpuwm.config import LAND_SURFACE_SCHEMES

    columns = {}
    for lsm in LAND_SURFACE_SCHEMES:
        columns[lsm] = tuple(
            receipt["mynn_matrix"][f"lsm{lsm}.lw{lw}.sw{sw}"]
            for lw in walk.AXIS_VALUES["ra_lw_physics"]
            for sw in walk.AXIS_VALUES["ra_sw_physics"])
    assert len(set(columns.values())) == 1, (
        "the radiation verdict depends on the land-surface model: "
        f"{columns}")
    assert "ACCEPTED" in columns[LAND_SURFACE_SCHEMES[0]]


#: The microphysics MYNN reaches in the walk's MYNN slice, measured.
#:
#: The slice's anchor runs RTE+RRTMGP on both radiation streams, and TWO
#: admitted microphysics values are refused against it, both for the same
#: missing thing -- a cloud-optics row in
#: gpuwm.core.rrtmgp._MP_CLOUD_OPTICS_SCHEME -- and for different WRF
#: reasons:
#:
#: * mp_physics=9 (Milbrandt-Yau) publishes NO effective radii at all.  WRF
#:   leaves has_reqc/has_reqi/has_reqs at 0 for MILBRANDT2MOM
#:   (phys/module_physics_init.F:1004-1023) and the scheme's own
#:   effective-radius block is commented out.
#: * mp_physics=50 (P3) publishes cloud and ice radii but no SNOW radius:
#:   WRF's P3 override sets has_reqs=0 (:1027-1033) and P3's single ice
#:   category has no separate snow species.  Every existing row assumes a
#:   snow radius, so there is none to reuse.
#:
#: Neither is a MYNN limitation, and neither is a place to invent physics:
#: handing RRTMGP radii nobody computed is what put mp=28 on Kessler's row
#: until 2026-08-01.  Both refusals name two remedies and both work:
#: ra_rrtmg_variant = 'rrtmg_legacy', or the Dudhia pair.
#:
#: The 50 entry moved OUT of this list at the 1.9 gate, when
#: gpuwm.config.validate_p3_radiation started refusing the pairing the
#: loader used to admit and the run used to die on.
MYNN_MICROPHYSICS_ACCEPTED = [0, 1, 6, 8, 10, 16, 18, 28]

#: The admitted microphysics values the MYNN slice does not reach, and
#: why.  Stated as the pair they are refused against, because each is
#: refused against THAT radiation variant and not in general.  50 joined 9
#: at the 1.9 gate; see MYNN_MICROPHYSICS_ACCEPTED for the two distinct
#: WRF reasons.
MYNN_MICROPHYSICS_EXCLUDED = (9, 50)


def test_mynn_composes_with_every_microphysics_and_cumulus_scheme(
        receipt) -> None:
    """The claim, stated as the measurement instead of as an aspiration.

    This used to assert that MYNN reaches every admitted microphysics
    value.  With the 1.9 microphysics ports landed that is a false
    product statement: the slice's RTE+RRTMGP anchor refuses
    Milbrandt-Yau for absent cloud-optics coupling.  The assertion is now
    the exact measured set plus the named exclusion, so the excluded
    value cannot grow silently and the reason travels with the number.
    """

    from gpuwm.config import CU_SCHEMES

    accepted = receipt["mynn_slice"]["microphysics_options_accepted"]
    assert accepted == MYNN_MICROPHYSICS_ACCEPTED
    assert (receipt["mynn_slice"]["cumulus_options_accepted"]
            == sorted(CU_SCHEMES))

    # The exclusions are exactly these values, and every one is admitted by
    # the schema -- so these are composition refusals, not a shrunken axis.
    excluded = sorted(set(walk.AXIS_VALUES["mp_physics"]) - set(accepted))
    assert excluded == sorted(MYNN_MICROPHYSICS_EXCLUDED)

    # Non-vacuous, and the reason is each microphysics lane's own rule
    # rather than anything about MYNN: the same scheme runs under MYNN on
    # the Dudhia pair, and the refusal names the coupling it cannot make.
    anchor = dict(walk.ANCHORS["mynn-rrtmgp-noah"], **walk.TIER_A_HELD)
    for mp_physics in MYNN_MICROPHYSICS_EXCLUDED:
        refused = walk.attempt(dict(anchor, mp_physics=mp_physics))
        assert refused.verdict == "REFUSED", mp_physics
        assert "has no cloud-optics coupling" in refused.message
        assert "ra_rrtmg_variant='rrtmg_legacy'" in refused.message
        accepted_on_dudhia = walk.attempt(dict(
            anchor, mp_physics=mp_physics,
            ra_lw_physics=0, ra_sw_physics=1))
        assert accepted_on_dudhia.verdict == "ACCEPTED", (
            f"the mp={mp_physics} refusal is about RRTMGP cloud optics, so "
            f"its remedy has to run under MYNN: {accepted_on_dudhia.message}")


def _radiation_bearing(switches) -> bool:
    """Both streams on: the property the 1.8.8 MYNN family exists for."""

    return (int(switches["ra_lw_physics"]) != 0
            and int(switches["ra_sw_physics"]) != 0)


def test_the_shipped_mynn_presets_are_inside_the_measured_space() -> None:
    """The bridge from the walk to the product.

    Every shipped MYNN profile's own switch tuple must be one of the
    combinations the walk accepts -- otherwise the walk is measuring a
    space the presets do not live in.

    AND the radiation-bearing family is asserted SEPARATELY, per land
    surface.  Filtering on ``bl_pbl_physics == 5`` alone is satisfied by
    the three no-radiation MYNN rows that shipped long before this work,
    so a version of this test that stopped at the loop above would pass
    unchanged with the whole radiation-bearing family deleted: it would
    measure nothing the family added.  The property that discriminates
    is the one the family was built to establish -- every land surface
    that offers a no-radiation MYNN preset must also offer a
    radiation-bearing one, so a user who wants MYNN never has to leave
    the nocturnally valid block to get it.  Dropping any single row
    turns this red, and the pairing is derived from the shipped
    switches rather than a transcribed list of three names, so renaming
    a profile cannot quietly satisfy it either.
    """

    from gpuwm.physics_compat import (SINGLE_DOMAIN_PHYSICS_PROFILES,
                                      single_domain_runtime_switches)

    mynn = [
        profile for profile in SINGLE_DOMAIN_PHYSICS_PROFILES
        if int(single_domain_runtime_switches(profile)["bl_pbl_physics"]) == 5
    ]
    assert mynn, "no MYNN profile ships at all"
    for profile in mynn:
        switches = single_domain_runtime_switches(profile)
        combination = {
            name: int(switches[name]) for name in walk.AXIS_NAMES
            if name in switches}
        combination.setdefault("km_opt", 4)
        outcome = walk.attempt(combination)
        assert outcome.verdict == "ACCEPTED", (
            f"{profile} resolves to a combination the walk refuses: "
            f"{outcome.message}")

    by_land_surface: dict[int, dict[bool, list[str]]] = {}
    for profile in mynn:
        switches = single_domain_runtime_switches(profile)
        land_surface = int(switches["sf_surface_physics"])
        bucket = by_land_surface.setdefault(land_surface, {True: [], False: []})
        bucket[_radiation_bearing(switches)].append(profile)

    # The control.  Without a surviving no-radiation row the pairing
    # requirement below would be vacuously satisfied by an empty set.
    dark = {land_surface for land_surface, rows in by_land_surface.items()
            if rows[False]}
    assert dark, (
        "no shortwave-only MYNN preset ships any more; this test's "
        "pairing requirement has gone vacuous and needs rewriting")

    unpaired = sorted(
        land_surface for land_surface in dark
        if not by_land_surface[land_surface][True])
    assert unpaired == [], (
        "every land surface with a shortwave-only MYNN preset must also "
        "ship a radiation-bearing MYNN preset, or a user choosing MYNN "
        "from a menu has no nocturnally valid row to move to; "
        f"sf_surface_physics={unpaired} has none. Shipped MYNN rows: "
        + "; ".join(
            f"lsm{land_surface}: lw+sw={rows[True]} sw-only={rows[False]}"
            for land_surface, rows in sorted(by_land_surface.items())))

    # And the radiation-bearing rows are in the measured space WITH
    # radiation on, which is the claim the walk is cited for.
    radiating = [profile for rows in by_land_surface.values()
                 for profile in rows[True]]
    assert len(radiating) == len(dark), (
        f"expected one radiation-bearing MYNN row per land surface that "
        f"has a shortwave-only row; got {radiating} against {sorted(dark)}")
    for profile in radiating:
        switches = single_domain_runtime_switches(profile)
        combination = {
            name: int(switches[name]) for name in walk.AXIS_NAMES
            if name in switches}
        combination.setdefault("km_opt", 4)
        assert combination["ra_lw_physics"] and combination["ra_sw_physics"]
        outcome = walk.attempt(combination)
        assert outcome.verdict == "ACCEPTED", (
            f"{profile} runs both radiation streams and the walk refuses "
            f"it: {outcome.message}")
