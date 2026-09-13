"""Cyclone exit is diagnostic termination, never a forecast or mover exception."""
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.core.track_boundary import boundary_reason
from gpuwm.core.storm_tracking import FollowConfig, NestFootprint, StormTracker
from gpuwm.core.storm_track_writer import TrackConfig, TrackWriter
from gpuwm.cyclone_sources import source_ids


def low(ci, cj=20, n=41):
    y, x = np.mgrid[:n, :n]
    return 1020 - 70 * np.exp(-((x-ci)**2 + (y-cj)**2)/40)


def box(n=41):
    return slice(0, n), slice(0, n)


@pytest.mark.parametrize("axis,direction", [(0,-1),(0,1),(1,-1),(1,1)])
def test_all_parent_edges_end_with_reason(axis,direction):
    plane = low(-3 if direction < 0 else 44)
    if axis: plane = plane.T
    center = (1 if direction < 0 else 39,20)
    if axis: center = center[::-1]
    assert "parent-domain boundary" in boundary_reason(plane,box(),center,
                                                       extremum="minimum",radius_cells=15)


def test_search_box_edge_is_not_parent_edge():
    assert boundary_reason(low(8), (slice(5,35),slice(12,30)), (14,20),
                           extremum="minimum") is None


def test_nan_uniform_and_interior_tied_minima_do_not_claim_exit():
    assert boundary_reason(np.full((41,41),np.nan),box(),(0,20),extremum="minimum") is None
    assert boundary_reason(np.ones((41,41)),box(),(0,20),extremum="minimum") is None
    plane = low(0)
    plane[20,10] = plane[20,0]
    assert boundary_reason(plane,box(),(0,20),extremum="minimum") is None


@pytest.mark.parametrize("source", source_ids())
@pytest.mark.parametrize("dx", [1000.,12000.,25000.])
def test_tracking_uses_model_state_not_forcing_name(monkeypatch, source, dx):
    # The forcing name never reaches the tracker. The same forecast low must
    # move and terminate identically at every initialization source.
    import gpuwm.core.storm_tracking as st
    cfg = FollowConfig(field="pressure",threshold=1010.,level_hpa=0,search_margin_cells=50,
                       radius_km=15*dx/1000., cooldown_seconds=0., min_shift_cells=1, max_shift_cells=6)
    tracker = StormTracker(cfg)
    initial_receipts = list(tracker.receipts)
    fp = NestFootprint(grid_id=2, i_parent_start=12,j_parent_start=12,
                       child_nx=25,child_ny=25,parent_grid_ratio=3,parent_dx_m=dx)
    monkeypatch.setattr(st,"planes_for",lambda state,*a,**kw:[(None,state.pressure)])
    interior = tracker.locate(SimpleNamespace(pressure=low(18)),fp,0.)
    assert interior.center_parent_ij[0] == pytest.approx(18.,abs=.2)
    moved = tracker.locate(SimpleNamespace(pressure=low(25)),fp,60.)
    assert moved.center_parent_ij[0] > interior.center_parent_ij[0]
    assert "track_end_reason" not in moved.evidence
    departed = tracker.locate(SimpleNamespace(pressure=low(44)),fp,120.)
    assert "boundary" in departed.evidence["track_end_reason"]
    assert tracker.receipts == initial_receipts  # locate does not alter movement decisions


def test_writer_ends_between_output_times_and_never_appends_after_exit(tmp_path):
    writer = TrackWriter(TrackConfig("track.csv",interval_seconds=60.),
                         initial_time=datetime(2026,9,9),outdir=tmp_path)
    missing = SimpleNamespace(found=None,evidence={})
    writer.emit(missing,t=0,parent_state=None)
    terminal = SimpleNamespace(found=None,evidence={"track_end_reason":"parent-domain boundary"})
    result = writer.emit(terminal,t=1,parent_state=None)
    assert result["status"] == "ended" and result["emitted"] is False
    assert writer.emit(missing,t=120,parent_state=None) is None
    assert writer.close()["termination"]["reason"] == "parent-domain boundary"
    assert len((tmp_path/"track.csv").read_text().splitlines()) == 2


def test_missing_signal_alone_is_a_gap_not_domain_exit(tmp_path):
    writer = TrackWriter(TrackConfig("track.csv"),initial_time=datetime(2026,9,9),outdir=tmp_path)
    for t in (0,60,120):
        assert writer.emit(SimpleNamespace(found=None,evidence={}),t=t,parent_state=None)["no_signal"]
    assert "termination" not in writer.close()
    assert len((tmp_path/"track.csv").read_text().splitlines()) == 4


def test_unavailable_steering_level_does_not_hide_boundary_exit(monkeypatch):
    import gpuwm.core.storm_tracking as st
    cfg = FollowConfig(field="pressure",threshold=10.,level_hpa=(850.,700.),
                       search_margin_cells=50, radius_km=150., cooldown_seconds=0.,
                       min_shift_cells=1,max_shift_cells=6)
    tracker = StormTracker(cfg)
    fp = NestFootprint(grid_id=2,i_parent_start=12,j_parent_start=12,
                       child_nx=25,child_ny=25,parent_grid_ratio=3,parent_dx_m=10000.)
    monkeypatch.setattr(st,"planes_for",lambda *a,**kw:
                        [(850.,low(44)),(700.,np.full((41,41),np.nan))])
    fix = tracker.locate(None,fp,0.)
    assert "boundary" in fix.evidence["track_end_reason"]
    assert len(fix.levels) == 1


@pytest.mark.parametrize("prepared",[False,True])
def test_independent_runtime_follower_receives_its_track_config(tmp_path,monkeypatch,prepared):
    from gpuwm import runtime,cyclone_setup
    _, exp=cyclone_setup.configuration_text(cycle="2026090900",point=(18.,-65.))
    root=SimpleNamespace(cfg=exp.root)
    child=SimpleNamespace(cfg=exp.domains[1],parent=root)
    model=SimpleNamespace(nodes_by_grid_id={1:root,2:child},node=lambda gid: {1:root,2:child}[gid])
    monkeypatch.setattr("gpuwm.core.uh_diag.allocate_declared_follower_windows",lambda *a:None)
    def factory(view,*a,**kw):
        if not view.relocation.enabled:
            return None
        writer=runtime.build_track_writer(view,tmp_path)
        assert writer is not None
        writer.close()
        return SimpleNamespace(config=view.relocation)
    if prepared:
        monkeypatch.setattr(runtime,"build_prepared_tree_relocation_runner",factory)
        result=runtime.build_prepared_tree_relocation_runners(exp,model=model,statics_corridor={},outdir=tmp_path)
    else:
        monkeypatch.setattr(runtime,"build_real_relocation_runner",factory)
        result=runtime.build_real_relocation_runners(exp,None,model,tmp_path)
    assert result.runners[2].config.track is exp.domains[1].follow.track
    assert (tmp_path/"storm-track.d02.csv").exists()


def test_stash_backed_follower_cadence_still_refuses_at_configuration_load():
    from dataclasses import replace
    from gpuwm import cyclone_setup
    from gpuwm.core.nest_lifecycle import validate_follow_tracks
    _, exp = cyclone_setup.configuration_text(cycle="2026090900",point=(18.,-65.))
    child = exp.domains[1]
    tracker = FollowConfig(field="uh", threshold=100., fallback_threshold=35.,
        search_margin_cells=5, min_shift_cells=1, max_shift_cells=6, cooldown_seconds=0.)
    follow = replace(child.follow, tracker=tracker, track=None)
    domains = [exp.root, replace(child, follow=follow)]
    with pytest.raises(ValueError, match="whole multiple"):
        validate_follow_tracks(domains, exp.relocation, "fixture.toml")
