"""The treatment proof: does an enabled stream actually assimilate?

Five A/B screens were voided because a flag was never set and nothing in
the run's own output could tell a live arm from a dead one.  These tests
pin the two halves of the fix: counts come from the batches the filter
solved, and an enabled stream that contributes nothing stops the run.
"""

import pytest

from gpuwm.da.treatment import (OBS_KINDS, PROVABLE_KINDS, TreatmentNotApplied,
                                assimilated_counts, batch_kinds, cycle_record,
                                dealias_recovered, verify_treatment)


# --------------------------------------------------------------------------
# the shapes the driver really hands this module
# --------------------------------------------------------------------------

def radar_provenance(*, velocity=True, reflectivity=True, clear_air=False):
    batches = []
    if velocity:
        batches += [{"name": "vr:KGRR", "kind": "radial_velocity"},
                    {"name": "vr:KIWX", "kind": "radial_velocity"}]
    if reflectivity:
        batches.append({"name": "z", "kind": "reflectivity"})
    if clear_air:
        batches.append({"name": "z0", "kind": "clear_air_reflectivity"})
    return {"batches": batches}


SURFACE_PROVENANCE = {"batches": [
    {"name": "temperature_2m:asos", "kind": "surface"},
    {"name": "wind_speed_10m:asos", "kind": "surface"}]}

CWP_PROVENANCE = {"batches": [{"name": "cwp", "kind": "cloud_water_path"}]}


def innovations(**by_name):
    """innovation_summary output: one entry per batch, with its count."""
    return [{"name": name, "observations": n} for name, n in by_name.items()]


def live_cycle():
    return cycle_record(
        innovations(**{"vr:KGRR": 8100, "vr:KIWX": 6400, "z": 14203,
                       "temperature_2m:asos": 41, "wind_speed_10m:asos": 39,
                       "cwp": 2211}),
        adapter_provenance=radar_provenance(),
        extra_obs_provenance=SURFACE_PROVENANCE,
        cwp_provenance=CWP_PROVENANCE)


# --------------------------------------------------------------------------
# counting
# --------------------------------------------------------------------------

def test_counts_are_keyed_by_the_adapters_own_kinds():
    kinds = batch_kinds(radar_provenance(), SURFACE_PROVENANCE,
                        CWP_PROVENANCE)
    assert kinds["vr:KGRR"] == "radial_velocity"
    assert kinds["z"] == "reflectivity"
    assert kinds["temperature_2m:asos"] == "surface"
    assert kinds["cwp"] == "cloud_water_path"


def test_per_radar_velocity_batches_sum_into_one_type():
    """Two radars are two batches and one stream.

    The count that matters is "did radial velocity contribute", not how
    many files it arrived in.
    """
    record = live_cycle()
    assert record["counts"]["radial_velocity"] == 8100 + 6400


def test_every_known_kind_is_present_even_at_zero():
    """A key that appears only when non-zero cannot be tested for zero."""
    record = cycle_record(innovations(**{"vr:KGRR": 10}),
                          adapter_provenance=radar_provenance(
                              reflectivity=False))
    for kind in OBS_KINDS:
        assert kind in record["counts"]
    assert record["counts"]["cloud_water_path"] == 0
    assert record["counts"]["surface"] == 0


def test_a_batch_with_no_published_kind_is_flagged_not_guessed():
    counts = assimilated_counts(innovations(**{"mystery": 7}), {})
    assert counts["unattributed"] == 7


def test_the_count_is_what_the_filter_solved_not_what_was_enabled():
    """A stream enabled but contributing nothing counts zero.

    This is the exact shape of the bug: the flag was on, the batch was
    built, and the mask was empty.
    """
    record = cycle_record(
        innovations(**{"vr:KGRR": 8100, "cwp": 0}),
        adapter_provenance=radar_provenance(reflectivity=False),
        cwp_provenance=CWP_PROVENANCE)
    assert record["counts"]["cloud_water_path"] == 0
    assert record["counts"]["radial_velocity"] == 8100


# --------------------------------------------------------------------------
# dealiasing: None and 0 are different findings
# --------------------------------------------------------------------------

def test_dealias_never_run_reads_as_none_not_zero():
    assert dealias_recovered(radar_provenance()) is None
    assert dealias_recovered({"batches": [], "dealias": []}) is None
    assert dealias_recovered({"batches": [], "dealias": {}}) is None


def test_dealias_that_ran_and_recovered_nothing_reads_as_zero():
    prov = dict(radar_provenance(),
                dealias=[{"totals": {"gates_unfolded": 0}}])
    assert dealias_recovered(prov) == 0


def test_dealias_recovery_sums_across_radars():
    """The account is a LIST, one entry per contributing radar.

    This is the shape a real multi-radar run produces --
    merge_contributions keeps each radar's unfolding separate, because a
    fold is a property of one radar's Nyquist and one radar's sweep. The
    first live full-stack run crashed here on a dict-shaped assumption,
    which is why the real shape is pinned.
    """
    prov = dict(radar_provenance(),
                dealias=[{"totals": {"gates_unfolded": 10676}},
                         {"totals": {"gates_unfolded": 24}}])
    assert dealias_recovered(prov) == 10700
    assert cycle_record(innovations(**{"vr:KGRR": 1}),
                        adapter_provenance=prov)[
        "dealias_recovered_gates"] == 10700


def test_dealias_recovery_is_reported_as_a_number():
    prov = dict(radar_provenance(),
                dealias=[{"totals": {"gates_unfolded": 10676}}])
    assert cycle_record(innovations(**{"vr:KGRR": 1}),
                        adapter_provenance=prov)[
        "dealias_recovered_gates"] == 10676


# --------------------------------------------------------------------------
# the refusal
# --------------------------------------------------------------------------

ENABLED = ("radial_velocity", "reflectivity", "surface", "cloud_water_path")


def test_it_stays_quiet_on_a_live_run():
    verdict = verify_treatment(ENABLED, [live_cycle(), live_cycle()])
    assert verdict["verdict"] == "proved"
    assert verdict["silent_streams"] == []
    assert verdict["assimilated_over_window"]["cloud_water_path"] == 4422


def test_it_fires_on_a_stream_that_assimilated_nothing():
    """The failure that voided five screens, caught before the verdict."""
    dead = cycle_record(
        innovations(**{"vr:KGRR": 8100, "z": 14203,
                       "temperature_2m:asos": 41, "cwp": 0}),
        adapter_provenance=radar_provenance(),
        extra_obs_provenance=SURFACE_PROVENANCE,
        cwp_provenance=CWP_PROVENANCE)
    with pytest.raises(TreatmentNotApplied) as caught:
        verify_treatment(ENABLED, [dead, dead])
    message = str(caught.value)
    assert "cloud_water_path=0" in message
    # It names what DID work, so the reader can tell a broken stream from
    # a broken run.
    assert "radial_velocity=16200" in message
    assert "full-stack" in message


def test_it_names_every_silent_stream_not_just_the_first():
    dead = cycle_record(
        innovations(**{"vr:KGRR": 8100, "z": 14203}),
        adapter_provenance=radar_provenance())
    with pytest.raises(TreatmentNotApplied) as caught:
        verify_treatment(ENABLED, [dead, dead])
    message = str(caught.value)
    assert "cloud_water_path=0" in message
    assert "surface=0" in message


def test_one_empty_cycle_out_of_two_is_not_a_failure():
    """A satellite granule can be missing for one cycle and present the
    next.  The window is what is judged, not each cycle."""
    dead = cycle_record(
        innovations(**{"vr:KGRR": 8100, "z": 14203,
                       "temperature_2m:asos": 41, "cwp": 0}),
        adapter_provenance=radar_provenance(),
        extra_obs_provenance=SURFACE_PROVENANCE,
        cwp_provenance=CWP_PROVENANCE)
    verdict = verify_treatment(ENABLED, [dead, live_cycle()])
    assert verdict["verdict"] == "proved"


def test_a_run_that_has_not_had_its_chance_yet_is_pending_not_failed():
    verdict = verify_treatment(ENABLED, [live_cycle()])
    assert verdict["verdict"] == "pending"
    assert verdict["cycles_seen"] == 1


def test_a_stream_that_was_never_enabled_is_never_enforced():
    """Off asserts nothing.  Only ON is a claim this module holds to."""
    record = cycle_record(innovations(**{"vr:KGRR": 8100}),
                          adapter_provenance=radar_provenance(
                              reflectivity=False))
    verdict = verify_treatment(("radial_velocity",), [record, record])
    assert verdict["verdict"] == "proved"


def test_clear_air_zeroes_are_counted_but_not_enforced():
    """A domain with no measured clear air legitimately supplies none.

    Counted so the record is complete; not enforced, because zero here is
    a fact about the weather rather than about the configuration.
    """
    assert "clear_air_reflectivity" in OBS_KINDS
    assert "clear_air_reflectivity" not in PROVABLE_KINDS
    record = cycle_record(
        innovations(**{"vr:KGRR": 8100, "z0": 0}),
        adapter_provenance=radar_provenance(reflectivity=False,
                                            clear_air=True))
    verdict = verify_treatment(("radial_velocity", "clear_air_reflectivity"),
                               [record, record])
    assert verdict["verdict"] == "proved"
    assert verdict["not_enforced"] == ["clear_air_reflectivity"]


def test_an_unknown_kind_is_refused_rather_than_silently_unproved():
    """The one way to defeat this check would be to ask it about a stream
    it does not know, and get a pass by default."""
    with pytest.raises(TreatmentNotApplied, match="unknown observation kind"):
        verify_treatment(("radial_velocity", "lidar"), [live_cycle()] * 2)


# --------------------------------------------------------------------------
# the proof has to be WIRED, which is the bug it exists to prevent
# --------------------------------------------------------------------------

def test_the_cycle_driver_actually_calls_the_proof():
    """A check nothing calls is the bug, not the fix.

    Five screens were void because a capability existed and no code path
    reached it. Shipping an unreached treatment proof would be the same
    mistake wearing the fix's clothes, so this asserts the driver reaches
    both halves: the per-leg count and the refusal.
    """

    from pathlib import Path

    driver = (Path(__file__).resolve().parent.parent
              / "tools" / "da_cycle_prepared.py")
    source = driver.read_text(encoding="utf-8")
    assert "treatment.cycle_record(" in source, (
        "the driver records no per-leg assimilated counts")
    assert "treatment.verify_treatment(" in source, (
        "the driver never asks whether its enabled streams contributed")
    assert "TREATMENT_NOT_APPLIED" in source, (
        "the driver does not stop on a silent stream")


def test_the_enabled_set_covers_every_stream_the_driver_can_turn_on():
    """Every --flag that adds observations must map to a provable kind.

    A stream the driver can enable but the proof never hears about is a
    stream that can go silent unnoticed, which is the whole failure.
    """

    from pathlib import Path

    driver = (Path(__file__).resolve().parent.parent
              / "tools" / "da_cycle_prepared.py")
    source = driver.read_text(encoding="utf-8")
    block = source.split("enabled_obs_kinds = [")[1].split(
        "enabled_obs_kinds = tuple(")[0]
    for flag, kind in (("reflectivity_analysis", "reflectivity"),
                       ("clear_air_analysis", "clear_air_reflectivity"),
                       ("surface_obs", "surface"),
                       ("goes_cwp", "cloud_water_path")):
        assert f"args.{flag}" in block, f"{flag} never reaches the proof"
        assert kind in block, f"{flag} maps to no observation kind"
    assert "radial_velocity" in block
