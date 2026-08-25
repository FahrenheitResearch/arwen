"""``[relocation.track]`` through the real config loader.

The track writer is a diagnostic and must never fail a forecast -- so
everything that CAN be caught before the run starts is caught here, at
load, with the key named.  Three ways to configure a file that cannot
produce honest rows, each refused rather than discovered as an empty
file after twelve hours:

* an ``interval_seconds`` that is not a whole number of root steps (the
  writer emits at complete cycle boundaries, so it could never fire);
* an interval finer than the consultation cadence under a STASH-BACKED
  field the row actually READS, where the plane between consultations is
  stale or partial and the row would report a centre the nest was never
  steered by;
* either of those when the grid supplying the 10-m wind runs
  ``sf_sfclay_physics = 0``, where ``u10``/``v10`` are allocated and
  never filled, so the wind would be written as 0.00 m/s -- not a
  missing value, a wrong one.

Both of the last two are refusals about a COLUMN, so both are scoped to
the rows that have it.  A rotation or echo tracker writes POSITION ONLY
-- the mover's own centre off the mover's own grid -- so it reads no
plane and carries no wind, and neither refusal applies to it.  The
TRACKER-side guard (``_refuse_unservable_follow_cadence``) is untouched
by that and still applies in full: the nest is steered by the stashed
plane whatever the file says.

And the property that matters most: **no track table means no track file
and no behaviour change at all.**
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gpuwm.experiment import _build_relocation


def _domains(sfclay=91, history_interval_s=900.0):
    """Stand-ins carrying exactly what the track validation reads."""
    def dom(grid_id, parent_id, **over):
        base = dict(grid_id=grid_id, parent_id=parent_id, time_step=None,
                    time_step_fract_num=0, time_step_fract_den=1,
                    history_interval_s=history_interval_s,
                    run=SimpleNamespace(sf_sfclay_physics=sfclay))
        base.update(over)
        return SimpleNamespace(**base)
    return [dom(1, None, time_step=60), dom(2, 1), dom(3, 2)]


def _build(raw, domains=None, run_seconds=36000.0):
    return _build_relocation(raw, "case.toml", domains or _domains(),
                             run_seconds)


_FOLLOW = dict(field="pressure", threshold=30.0, level_hpa=850.0,
               search_margin_cells=12, min_shift_cells=1,
               max_shift_cells=2, cooldown_seconds=360.0)


def _raw(track=None, follow=None, cadence=360.0, **over):
    table = {"enabled": True, "grid_id": 3, "max_move_parent_cells": 2,
             "cadence_seconds": cadence,
             "follow": dict(follow or _FOLLOW)}
    if track is not None:
        table["track"] = track
    table.update(over)
    return {"relocation": table}


# ---------------------------------------------------------------------------
# Absent means absent
# ---------------------------------------------------------------------------

def test_no_track_table_is_no_track_at_all():
    cfg = _build(_raw())
    assert cfg.track is None
    assert cfg.receipt()["track"] is None


def test_a_track_table_echoes_every_accepted_value():
    cfg = _build(_raw(track={"path": "melissa-track.csv",
                             "interval_seconds": 720.0}))
    assert cfg.track.path == "melissa-track.csv"
    echo = cfg.receipt()["track"]
    assert echo["path"] == "melissa-track.csv"
    assert echo["interval_seconds"] == 720.0


def test_a_minimal_table_is_just_a_path():
    cfg = _build(_raw(track={"path": "t.csv"}))
    assert cfg.track.interval_seconds is None
    assert "interval_seconds" not in cfg.receipt()["track"]


def test_the_melissa_shaped_config_loads():
    """The flagship: field = 'pressure' WITH level_hpa, which is exactly
    the case where the tracker's extremum is metres of geopotential
    height and the writer has to reduce sea-level pressure separately."""
    cfg = _build(_raw(track={"path": "t.csv"}))
    # normalised to a tuple: one shape for every consumer, whether the
    # config named one surface or several
    assert cfg.follow.level_hpa == (850.0,)
    assert cfg.track.path == "t.csv"


# ---------------------------------------------------------------------------
# It is the TRACKER's answer, so it needs a tracker
# ---------------------------------------------------------------------------

def test_a_track_on_a_disabled_relocation_refuses_by_key_name():
    """The generic off-must-be-empty gate catches it first and names the
    key, which is the better error of the two."""
    with pytest.raises(ValueError, match=r"carries \['track'\]"):
        _build({"relocation": {"enabled": False,
                               "track": {"path": "t.csv"}}})


def test_a_hand_built_disabled_config_is_caught_by_the_dataclass_too():
    """The loader's gate is the first net; RelocationConfig itself is the
    second, for an object assembled in code rather than parsed."""
    from gpuwm.core.storm_track_writer import TrackConfig
    from gpuwm.experiment import RelocationConfig
    with pytest.raises(ValueError, match="disabled"):
        RelocationConfig(enabled=False, track=TrackConfig("t.csv"))


def test_a_track_with_only_a_scripted_itinerary_refuses():
    """[[relocation.move]] knows where to put the nest and nothing about
    where the storm is."""
    raw = _raw(track={"path": "t.csv"})
    raw["relocation"].pop("follow")
    raw["relocation"]["move"] = [{"at_seconds": 360.0, "di_parent_cells": 1}]
    with pytest.raises(ValueError, match=r"no \[relocation.follow\] block"):
        _build(raw)


# ---------------------------------------------------------------------------
# The interval
# ---------------------------------------------------------------------------

def test_an_interval_that_is_not_whole_root_steps_refuses():
    with pytest.raises(ValueError, match="whole number of root steps"):
        _build(_raw(track={"path": "t.csv", "interval_seconds": 90.0}))


@pytest.mark.parametrize("interval", [60.0, 120.0, 360.0, 720.0])
def test_an_interval_on_a_root_step_is_served(interval):
    cfg = _build(_raw(track={"path": "t.csv",
                             "interval_seconds": interval}))
    assert cfg.track.interval_seconds == interval


def test_any_cadence_serves_the_track():
    """No format needs an integer forecast hour any more; the file
    carries a valid time, so nothing has to divide 3600."""
    for cadence in (420.0, 480.0, 660.0, 1500.0):
        assert _build(_raw(track={"path": "t.csv"},
                           cadence=cadence)).track.path == "t.csv"


def test_a_finer_interval_is_allowed_on_a_pressure_tracker():
    """A pressure tracker reduces from the LIVE prognostic column, valid
    at every cycle boundary, so there is no stash to outrun."""
    cfg = _build(_raw(track={"path": "t.csv", "interval_seconds": 60.0}))
    assert cfg.track.interval_seconds == 60.0


@pytest.mark.parametrize("field,extra", [
    ("uh", {"fallback_threshold": 40.0}),
    ("reflectivity", {}),
])
def test_a_finer_interval_is_servable_on_a_position_only_field(field, extra):
    """The stash-cadence refusal existed because a row BETWEEN
    consultations would be derived from a stale plane or a partial
    accumulation.  A rotation or echo tracker no longer derives its row
    from the plane at all -- it writes the mover's own centre, off the
    mover's own grid, which is exact at every cycle boundary -- so the
    argument for refusing is gone and the configuration is admitted.

    This is the ADMISSION half; that such a run really does skip
    locating on those boundaries is pinned in
    ``test_relocation_track_runner.py``.
    """
    follow = dict(field=field, threshold=25.0, search_margin_cells=12,
                  min_shift_cells=1, max_shift_cells=2,
                  cooldown_seconds=360.0, **extra)
    cfg = _build(_raw(track={"path": "t.csv", "interval_seconds": 60.0},
                      follow=follow, cadence=900.0),
                 domains=_domains(history_interval_s=900.0))
    assert cfg.track.interval_seconds == 60.0


def test_the_stash_cadence_guard_still_exists_for_a_field_that_reads_one():
    """The guard is kept and expressed as the fields that are BOTH
    stash-backed and not position-only, so it comes back on its own if
    either set changes.  Today that difference is empty, and the test
    asserts the relationship rather than the emptiness -- an assertion
    that the two lists are the same list would pass just as happily if
    both were wrong.
    """
    from gpuwm.core.storm_track_writer import POSITION_ONLY_FIELDS
    from gpuwm.core.storm_tracking import (STASH_BACKED_FIELDS,
                                           TRACKED_FIELDS)

    # Every position-only field is stash-backed: a field that is reduced
    # on demand has a real signal position to report and should report it.
    assert set(POSITION_ONLY_FIELDS) <= set(STASH_BACKED_FIELDS)
    # ...and the one field that is neither still carries a full row.
    reduced = set(TRACKED_FIELDS) - set(STASH_BACKED_FIELDS)
    assert reduced == {"pressure"}


@pytest.mark.parametrize("field,extra", [
    ("uh", {"fallback_threshold": 40.0}),
    ("reflectivity", {}),
])
def test_an_equal_or_coarser_interval_is_fine_on_a_stash(field, extra):
    follow = dict(field=field, threshold=25.0, search_margin_cells=12,
                  min_shift_cells=1, max_shift_cells=2,
                  cooldown_seconds=360.0, **extra)
    cfg = _build(_raw(track={"path": "t.csv", "interval_seconds": 1800.0},
                      follow=follow, cadence=900.0),
                 domains=_domains(history_interval_s=900.0))
    assert cfg.track.interval_seconds == 1800.0


# ---------------------------------------------------------------------------
# The wind needs a surface layer
# ---------------------------------------------------------------------------

def test_no_surface_layer_refuses_the_track_by_name():
    """u10/v10 are allocated unconditionally and filled only by a scheme,
    so sf_sfclay_physics = 0 would write 0.00 m/s: a plausible-looking
    number of the wrong quantity, which a reader cannot detect."""
    with pytest.raises(ValueError, match="sf_sfclay_physics = 0"):
        _build(_raw(track={"path": "t.csv"}), domains=_domains(sfclay=0))


@pytest.mark.parametrize("field,extra", [
    ("uh", {"fallback_threshold": 40.0}),
    ("reflectivity", {}),
])
def test_a_position_only_track_needs_no_surface_layer(field, extra):
    """The refusal above is about a COLUMN, so it is scoped to the rows
    that have it: a rotation or echo track file carries no peak wind,
    and an idealised convective run has no surface-layer scheme to
    offer.  The control is the test above, on the same domains."""
    follow = dict(field=field, threshold=25.0, search_margin_cells=12,
                  min_shift_cells=1, max_shift_cells=2,
                  cooldown_seconds=360.0, **extra)
    # cadence == history_interval_s, because the TRACKER-side guard
    # (_refuse_unservable_follow_cadence) is untouched by any of this:
    # the nest is steered by the stashed plane whatever the file says.
    cfg = _build(_raw(track={"path": "t.csv"}, follow=follow, cadence=900.0),
                 domains=_domains(sfclay=0, history_interval_s=900.0))
    assert cfg.track.path == "t.csv"


@pytest.mark.parametrize("scheme", [1, 2, 5, 91])
def test_every_routed_surface_layer_serves_the_wind(scheme):
    cfg = _build(_raw(track={"path": "t.csv"}),
                 domains=_domains(sfclay=scheme))
    assert cfg.track.path == "t.csv"


def test_the_refine_grid_is_checked_too():
    """The refine grid is where a row's quantities come from when stage
    two applies, so its surface layer is the one that matters."""
    domains = _domains()
    domains[2].run = SimpleNamespace(sf_sfclay_physics=0)   # d03
    with pytest.raises(ValueError, match="d03"):
        _build(_raw(track={"path": "t.csv"},
                    follow=dict(_FOLLOW, refine_grid_id=3)),
               domains=domains)


def test_a_domain_with_no_resolved_selector_is_not_reported_as_zero():
    """'this object carries no resolved selector' and 'this domain runs
    no surface layer' are different statements."""
    domains = _domains()
    for dom in domains:
        dom.run = SimpleNamespace()
    assert _build(_raw(track={"path": "t.csv"}),
                  domains=domains).track.path == "t.csv"


# ---------------------------------------------------------------------------
# Unknown keys
# ---------------------------------------------------------------------------

def test_track_is_an_accepted_relocation_key():
    from gpuwm.experiment import _RELOCATION_KEYS
    assert "track" in _RELOCATION_KEYS


def test_a_misspelt_track_key_refuses_at_the_table():
    with pytest.raises(ValueError, match="does not have key"):
        _build(_raw(track={"path": "t.csv", "every_step": True}))


def test_the_old_array_of_tables_shape_refuses_and_says_why():
    """A config written against the earlier [[relocation.track]] shape
    must be told, not silently mis-parsed."""
    with pytest.raises(ValueError, match="single table"):
        _build(_raw(track=[{"path": "t.csv"}]))


# ---------------------------------------------------------------------------
# output_level: which blocks the file carries
# ---------------------------------------------------------------------------

def test_absent_output_level_is_every_block():
    """The default has to be the file as it was before the key existed,
    or adding the key would change every run that does not use it."""
    cfg = _build(_raw(track={"path": "t.csv"}))
    assert cfg.track.output_level is None
    assert "output_level" not in cfg.receipt()["track"]


@pytest.mark.parametrize("given,want", [
    (0, (0.0,)),
    (850.0, (850.0,)),
    ([0, 850.0], (0.0, 850.0)),
    ([850.0], (850.0,)),
])
def test_output_level_normalises_to_a_tuple(given, want):
    """One shape for every consumer, whether the config wrote a number or
    a list -- the same rule level_hpa follows."""
    cfg = _build(_raw(track={"path": "t.csv", "output_level": given}))
    assert cfg.track.output_level == want
    assert cfg.receipt()["track"]["output_level"] == [float(v) for v in want]


def test_output_level_may_name_the_surface_and_any_tracked_level():
    cfg = _build(_raw(track={"path": "t.csv", "output_level": [0, 850.0]},
                      follow=dict(_FOLLOW, level_hpa=[850.0, 700.0])))
    assert cfg.track.output_level == (0.0, 850.0)


def test_output_level_may_name_a_surface_that_does_not_steer():
    """The two keys answer DIFFERENT questions -- level_hpa is what
    steers the nest, output_level is what the file carries -- so naming a
    surface the tracker does not follow adds it as REPORT-ONLY rather
    than refusing.  It gets a plane, a centre search and its own columns,
    and stays out of the steering mean."""
    cfg = _build(_raw(track={"path": "t.csv", "output_level": 700.0},
                      follow=dict(_FOLLOW, level_hpa=850.0)))
    assert cfg.follow.level_hpa == (850.0,)          # steers, unchanged
    assert cfg.follow.report_level_hpa == (700.0,)   # reported only


def test_a_reported_surface_does_not_join_the_steering_mean():
    """The property that makes the split worth having: a nest tracking
    850 alone is steered by 850 alone, however many surfaces the file
    reports."""
    from gpuwm.core.storm_tracking import (all_levels_of, levels_of,
                                           report_levels_of)

    cfg = _build(_raw(
        track={"path": "t.csv",
               "output_level": [0, 925.0, 850.0, 700.0, 500.0]},
        follow=dict(_FOLLOW, level_hpa=850.0)))
    assert levels_of(cfg.follow) == (850.0,)
    assert report_levels_of(cfg.follow) == (925.0, 700.0, 500.0)
    # Every surface a consultation computes, steering first then the
    # extras in the order output_level asked for them.
    assert all_levels_of(cfg.follow) == (850.0, 925.0, 700.0, 500.0)


def test_the_classic_steering_set_with_a_twenty_level_profile():
    """The configuration this was built for: steer on 850/700/500, print
    twenty surfaces."""
    from gpuwm.core.storm_tracking import all_levels_of

    profile = [925.0, 900.0, 875.0, 850.0, 825.0, 800.0, 775.0, 750.0,
               725.0, 700.0, 675.0, 650.0, 625.0, 600.0, 550.0, 500.0,
               450.0, 400.0, 350.0, 300.0]
    cfg = _build(_raw(track={"path": "t.csv", "output_level": profile},
                      follow=dict(_FOLLOW, level_hpa=[850.0, 700.0, 500.0])))
    assert cfg.follow.level_hpa == (850.0, 700.0, 500.0)
    assert len(all_levels_of(cfg.follow)) == 20
    assert set(all_levels_of(cfg.follow)) == set(profile)


def test_there_is_no_cap_on_how_many_surfaces_are_named():
    """The cap was 8, justified as "a deep-layer mean does not get better
    past a handful" -- true of STEERING and never true of REPORTING.
    Measured, a surface is 2.3 ms and the scaling is flat, so twenty is
    0.15% of a three-day run at the relocation cadence."""
    from gpuwm.core.storm_tracking import MAX_TRACKED_LEVELS, all_levels_of

    assert MAX_TRACKED_LEVELS is None
    # Every one inside LEVEL_HPA_MIN..MAX -- report-only relaxes HOW
    # MANY surfaces may be named, never whether each is a real pressure.
    many = [float(v) for v in range(950, 200, -25)]     # 30 surfaces
    cfg = _build(_raw(track={"path": "t.csv", "output_level": many},
                      follow=dict(_FOLLOW, level_hpa=[850.0, 500.0])))
    assert len(all_levels_of(cfg.follow)) == len(set(many) | {850.0, 500.0})
    # ...and steering is still the two that were asked to steer.
    assert cfg.follow.level_hpa == (850.0, 500.0)


def test_a_surface_cannot_both_steer_and_be_report_only():
    """Naming a tracked surface in output_level selects it for the file;
    it must not ALSO be added as a report-only copy, which would compute
    it twice and put two columns in the header."""
    cfg = _build(_raw(track={"path": "t.csv", "output_level": [850.0, 700.0]},
                      follow=dict(_FOLLOW, level_hpa=[850.0, 500.0])))
    assert cfg.follow.level_hpa == (850.0, 500.0)
    assert cfg.follow.report_level_hpa == (700.0,)


def test_a_reported_surface_outside_the_atmosphere_still_refuses():
    """Report-only relaxes WHICH surfaces may be named, not whether a
    number is a pressure."""
    with pytest.raises(ValueError, match="outside"):
        _build(_raw(track={"path": "t.csv", "output_level": 85000.0},
                    follow=dict(_FOLLOW, level_hpa=850.0)))


def test_output_level_zero_is_available_even_with_no_isobaric_surface():
    """The sea-level tracker (level_hpa = 0) has no level blocks at all,
    and the surface block is exactly what it writes."""
    follow = dict(_FOLLOW, level_hpa=0, threshold=1004.0)
    cfg = _build(_raw(track={"path": "t.csv", "output_level": 0},
                      follow=follow))
    assert cfg.track.output_level == (0.0,)
    # An isobaric surface beside a SEA-LEVEL tracker is the one case
    # report-only cannot serve: the two are different reductions in
    # different units and this run computes only the first.
    with pytest.raises(ValueError, match="SEA LEVEL"):
        _build(_raw(track={"path": "t.csv", "output_level": 850.0},
                    follow=follow))


@pytest.mark.parametrize("field,extra", [
    ("uh", {"fallback_threshold": 40.0}),
    ("reflectivity", {}),
])
def test_output_level_refuses_under_a_position_only_tracker(field, extra):
    """That file is a clock and one position; there is nothing to choose
    between, so a key that chooses is a misunderstanding worth naming."""
    follow = dict(field=field, threshold=25.0, search_margin_cells=12,
                  min_shift_cells=1, max_shift_cells=2,
                  cooldown_seconds=900.0, **extra)
    with pytest.raises(ValueError, match="refuses output_level"):
        _build(_raw(track={"path": "t.csv", "output_level": 0},
                    follow=follow, cadence=900.0),
               domains=_domains(history_interval_s=900.0))


@pytest.mark.parametrize("bad,match", [
    ([], "at least one block"),
    ([0, 0], "repeats a block"),
    ([850.0, 850.0], "repeats a block"),
    (-1, "not a block"),
    ("850", "must be a block"),
    (True, "must be a block"),
])
def test_a_bad_output_level_refuses_by_name(bad, match):
    with pytest.raises(ValueError, match=match):
        _build(_raw(track={"path": "t.csv", "output_level": bad}))


def test_output_level_is_an_accepted_track_key():
    from gpuwm.core.storm_track_writer import TRACK_KEYS
    assert "output_level" in TRACK_KEYS


def test_a_misspelt_output_level_is_offered_the_suggestion():
    with pytest.raises(ValueError, match="output_level"):
        _build(_raw(track={"path": "t.csv", "output_levels": 0}))
