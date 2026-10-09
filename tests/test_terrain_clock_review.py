"""Landing review regressions against the terrain clock's real artifacts."""
from dataclasses import replace
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm import terrain_clock as tc, terrain_clock_local as tl
from test_terrain_clock_local import _fields, _ncar_run, _load, _crest


@pytest.mark.parametrize("arms,dx,crest", [
    (("generated",), 2000., 3400.), (("generated",), 3000., 3400.),
    (("generated",), 12000., 3400.), (("ncar",), 12000., 3450.),
    (("ncar",), 2000., 3400.), (("ncar",), 3000., 3400.),
    (("generated", "ncar"), 12000., 3400.),
])
@pytest.mark.parametrize("wind", [67.5, 75., 95.])
def test_unrun_candidate_wind_is_unknown(arms, dx, crest, wind):
    slope = .35 if arms == ("ncar",) and dx < 12000. else .04
    reading = tc.read_map(dx, crest, slope, wind, 4, tl.candidate_map(arms))
    assert reading.per_km is None
    assert "wind" in reading.beyond
    assert reading.most_stable_per_km is not None


def test_default_missing_wind_takes_shipped_decision():
    _, meta = _load("ncar-conus12km-faces-d01.npz")
    run = _ncar_run(meta)
    run = SimpleNamespace(**{**vars(run), "dt": 60.,
                            "time_step_sound": 4})
    terrain = np.full((12, 14), 3450.)
    terrain[:, :2] = [570., 2010.]
    faces = tl.DomainFaces(_fields(terrain, np.full(terrain.shape, 50.),
                                  radius=5, dx=12000.),
                          tl.candidate_map(("ncar",)), ("ncar",), 60000.,
                          55., 12000.)
    crest = replace(_crest(meta), crest_height_m=3450., wind_m_s=55.)
    got = tc.derive_clock(1, run, Fraction(60), .12, crest, local=faces)
    assert got.never_worse.applied
    assert got.never_worse.reason == tl.NEVER_WORSE_BEYOND
    assert got.never_worse.local.status == "BEYOND_MEASURED"
    assert got.dt == got.never_worse.shipped.dt


@pytest.mark.parametrize("stored_mode", [None, "measured"])
def test_tree_restart_names_default_clock_flip(stored_mode):
    from gpuwm.io.restart import (tree_fingerprint_mismatch_reason,
                                  TERRAIN_CLOCK_RESTART_BREAK_NOTICE)
    stored_run = {} if stored_mode is None else {"terrain_clock": stored_mode}
    stored = {"experiment_identity": {"domains": [
        {"grid_id": 2, "run": stored_run},
        {"grid_id": 1, "run": {"terrain_clock": "pinned"}}]}}
    live = {"experiment_identity": {"domains": [
        {"grid_id": 1, "run": {"terrain_clock": "pinned"}},
        {"grid_id": 2, "run": {"terrain_clock": "local_face"}}]}}
    reason = tree_fingerprint_mismatch_reason(
        1, {"experiment_fingerprint_components": stored},
        SimpleNamespace(_experiment_fingerprint_components=live))
    assert TERRAIN_CLOCK_RESTART_BREAK_NOTICE in reason


@pytest.mark.parametrize("stored,live", [
    (None, None), ({"experiment_identity": None}, {}),
    ({"experiment_identity": {"domains": None}}, {}),
    ({"experiment_identity": {"domains": [{"grid_id": 1, "run": {}}]}},
     {"experiment_identity": {"domains": [{"grid_id": 2, "run":
                                           {"terrain_clock": "local_face"}}]}}),
])
def test_malformed_or_unmatched_tree_has_no_default_flip(stored, live):
    from gpuwm.io.restart import _terrain_clock_default_flip
    assert not _terrain_clock_default_flip(stored, live)


@pytest.mark.parametrize("cause,note", [
    ("corridor", "following nest"), ("terrain", "no terrain"),
    ("mapfac", "other map factors"), ("wind", "no face carried a wind"),
])
def test_default_fallback_is_reported_with_reason(cause, note):
    from test_terrain_clock import _wizard_experiment, _column_state
    from gpuwm.acoustic_adaptation import steepest_slope

    exp = _wizard_experiment([(12, 14)], (), 3000.)
    terrain = np.broadcast_to(np.arange(14) * 40., (12, 14)).copy()
    statics = {1: {"HGT_M": terrain}}
    slope = steepest_slope(terrain, 3000., 3000.).slope
    state = _column_state()
    start = tc.StartWinds.from_fields("d01", state.u, state.v,
                                      state.php, state.phb)
    crest = tc.CrestWind("d01", terrain.max(), 20., "start", "d01", 750.)
    corridors = {1: terrain} if cause == "corridor" else {}
    sources = {1: [start]}
    if cause == "terrain":
        statics = {1: {}}
    if cause == "mapfac":
        statics[1]["MAPFAC_U"] = np.full((12, 15), 1.2)
    if cause == "wind":
        sources = {1: []}
    local = tl.local_readings(exp, slopes={1: slope}, winds={1: crest},
                              statics=statics, sources=sources,
                              corridors=corridors)
    lines = []
    _, adaptations = tc.adapt_experiment_clock(
        exp, {1: slope}, {1: crest}, local=local,
        announce=lines.append, caution=lines.append)
    got = adaptations[0]
    assert note in got.local_note
    assert any(note in line and "domain-wide" in line for line in lines)
    shipped = tc.derive_clock(1, replace(exp.root.run, terrain_clock="measured"),
                              exp.dt_exact(1), slope, crest)
    assert got.dt == shipped.dt and got.time_step_sound == shipped.time_step_sound


@pytest.mark.parametrize("mode,want,flipped", [
    ("local_face", "measured", True), ("pinned", "local_face", False),
    ("local_face", "local_face", False), ("measured", "measured", False),
])
def test_tree_restart_notice_is_only_for_the_default_flip(mode, want, flipped):
    from gpuwm.io.restart import _terrain_clock_default_flip
    def components(value):
        return {"experiment_identity": {"domains": [
            {"grid_id": 1, "run": {"terrain_clock": value}}]}}
    assert _terrain_clock_default_flip(components(want), components(mode)) is flipped


def _stream_fixture(ny=12, nx=14):
    from test_terrain_clock import (_wizard_experiment, _column_state,
                                    _cache_reader, _coupled, _acoustic)
    from gpuwm.acoustic_adaptation import steepest_slope
    from gpuwm.ingest.lateral_bc import build_lateral_boundaries

    exp = _wizard_experiment([(ny, nx)], (), 3000., hours=3)
    terrain = np.broadcast_to(np.minimum(np.arange(nx) * 40., 3400.),
                               (ny, nx)).copy()
    msfu, msfv = np.ones((ny, nx + 1)), np.ones((ny + 1, nx))
    static = {"HGT_M": terrain, "MAPFAC_U": msfu, "MAPFAC_V": msfv}
    state = _column_state(ny=ny, nx=nx)
    frames = []
    for speed in (10., 10., 10., 80.):
        state.u.fill(speed)
        frames.append(_coupled(state, msfu, msfv))
    state.u.fill(10.)
    bounds = build_lateral_boundaries(frames, [0., 3600., 7200., 10800.],
                                      spec_bdy_width=5)
    reader = _cache_reader(state)
    basis = SimpleNamespace(experiment=exp,
        acoustic=_acoustic(exp, {1: steepest_slope(terrain, 3000., 3000.).slope}),
        statics={1: static}, readers={1: reader}, reach={})
    return basis, bounds


def test_streamed_guard_reuses_faces_and_matches_fresh_reading(monkeypatch):
    from gpuwm.prepared_domain_tree_forecast import StreamedClockGuard
    from gpuwm.ingest.lateral_bc import LateralBoundaries

    basis, bounds = _stream_fixture()
    starts = {1: tc.start_winds_from_cache(basis.readers[1], "d01")}
    geometry = tc.boundary_geometry_from_cache(basis.readers[1], basis.statics[1])
    calls = []
    original = tl.read_faces
    def counted(*a, **k):
        calls.append(k["sound_steps"])
        return original(*a, **k)
    monkeypatch.setattr(tl, "read_faces", counted)
    initial = tc.clock_for_domains(basis.experiment, basis.acoustic,
        statics=basis.statics, starts=starts, announce=False)[1]
    guard = StreamedClockGuard(basis, run_clock=tc.clock_receipt(initial),
                               root_grid_id=1, run_seconds=10800.)
    for index, interval in enumerate(bounds.intervals):
        guard(interval)
        count = len(calls)
        boundary = tc.BoundaryWinds("d01", LateralBoundaries(
            tuple(bounds.intervals[:index + 1]), 1, 1, 1), geometry, 10800.)
        fresh = tc.clock_for_domains(basis.experiment, basis.acoustic,
            statics=basis.statics, starts=starts, boundary=boundary,
            announce=False)[1]
        fresh_count = len(calls)
        cached = tc.clock_for_domains(basis.experiment, basis.acoustic,
            statics=basis.statics, starts=starts, boundary=boundary,
            announce=False, local_cache=guard.local_cache)[1]
        assert tc.clock_receipt(cached) == tc.clock_receipt(fresh)
        assert len(calls) == fresh_count  # cached comparison did not read
        if index == 1:
            assert count == previous_count  # unchanged second interval
        previous_count = len(calls)
    assert guard.checked == 3


def test_streamed_guard_still_refuses_a_stronger_boundary():
    from gpuwm.prepared_domain_tree_forecast import StreamedClockGuard
    from gpuwm.ingest.boundary_stream import StreamedClockChanged
    from test_terrain_clock import _acoustic
    from gpuwm.acoustic_adaptation import steepest_slope

    basis, bounds = _stream_fixture()
    h = basis.statics[1]["HGT_M"]
    h[:, 7:] = np.minimum(4500., np.arange(h.shape[1] - 7) * 1050.)
    basis.acoustic = _acoustic(basis.experiment,
        {1: steepest_slope(h, 3000., 3000.).slope})
    starts = {1: tc.start_winds_from_cache(basis.readers[1], "d01")}
    initial = tc.clock_for_domains(basis.experiment, basis.acoustic,
        statics=basis.statics, starts=starts, announce=False)[1]
    guard = StreamedClockGuard(basis, run_clock=tc.clock_receipt(initial),
                               root_grid_id=1, run_seconds=10800.)
    guard(bounds.intervals[0])
    guard(bounds.intervals[1])
    with pytest.raises(StreamedClockChanged, match="moves the terrain-derived clock"):
        guard(bounds.intervals[2])


def test_streamed_cache_keeps_static_nest_readings():
    from test_terrain_clock import _wizard_experiment, _acoustic
    from gpuwm.acoustic_adaptation import steepest_slope
    from gpuwm.ingest.lateral_bc import LateralBoundaries

    basis, bounds = _stream_fixture(60, 60)
    exp = _wizard_experiment([(60, 60), (60, 60)], (3,), 3000., hours=3)
    static = basis.statics[1]
    statics = {1: static, 2: {**static, "HGT_M": static["HGT_M"] / 3.}}
    acoustic = _acoustic(exp, {gid: steepest_slope(s["HGT_M"],
        exp.domain(gid).run.dx, exp.domain(gid).run.dy).slope
        for gid, s in statics.items()})
    starts = {gid: tc.start_winds_from_cache(basis.readers[1], f"d{gid:02d}")
              for gid in (1, 2)}
    geometry = tc.boundary_geometry_from_cache(basis.readers[1], static)
    cache = tl.StreamedFaceCache()
    previous = None
    for index in (0, 1):
        boundary = tc.BoundaryWinds("d01", LateralBoundaries(
            tuple(bounds.intervals[:index + 1]), 1, 1, 1), geometry, 10800.)
        cached = tc.clock_for_domains(exp, acoustic, statics=statics,
            starts=starts, boundary=boundary, announce=False, local_cache=cache)[1]
        fresh = tc.clock_for_domains(exp, acoustic, statics=statics,
            starts=starts, boundary=boundary, announce=False)[1]
        assert tc.clock_receipt(cached) == tc.clock_receipt(fresh)
        if previous is not None:
            assert cache.domains[2] is previous
            assert cached[1].faces_read is previous.reading(
                cached[1].faces_read.sound_steps)
        previous = cache.domains[2]


def test_replaced_stream_interval_rebuilds_its_wind_maximum():
    from gpuwm.ingest.lateral_bc import LateralBoundaries

    basis, bounds = _stream_fixture()
    starts = {1: tc.start_winds_from_cache(basis.readers[1], "d01")}
    geometry = tc.boundary_geometry_from_cache(basis.readers[1], basis.statics[1])
    cache = tl.StreamedFaceCache()
    replaced = replace(bounds.intervals[2], fields=bounds.intervals[0].fields)
    for intervals in (bounds.intervals, (*bounds.intervals[:2], replaced)):
        boundary = tc.BoundaryWinds("d01", LateralBoundaries(
            tuple(intervals), 1, 1, 1), geometry, 10800.)
        cached = tc.clock_for_domains(basis.experiment, basis.acoustic,
            statics=basis.statics, starts=starts, boundary=boundary,
            announce=False, local_cache=cache)[1]
        fresh = tc.clock_for_domains(basis.experiment, basis.acoustic,
            statics=basis.statics, starts=starts, boundary=boundary,
            announce=False)[1]
        assert tc.clock_receipt(cached) == tc.clock_receipt(fresh)
