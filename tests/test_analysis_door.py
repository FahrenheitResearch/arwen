"""``gpuwm run --wrfinput DIR --analysis BUNDLE``: the door's plan, its refusals and the
worker's acts, on real files and a host state.  The forecast itself is
tests/test_analysis_door_gpu.py."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import netCDF4
import numpy as np
import pytest

from analysis_bundle_fixtures import bundle_identity, raw_fields, write_wrf_state, write_wrfbdy
from gpuwm.da import analysis_door as door
from gpuwm.da import iau
from gpuwm.io import analysis_bundle as ab

VALID = datetime(2022, 6, 13, 23)
NAMELIST = """&time_control
 run_hours = 1,
 start_year = 2022, start_month = 06, start_day = 13, start_hour = 23,
 end_year = 2022, end_month = 06, end_day = 14, end_hour = 00,
 interval_seconds = 3600,
/
&domains
 time_step = 60, max_dom = {max_dom},
 e_we = 8, e_sn = 7, e_vert = 5,
/
&dynamics
 hypsometric_opt = 2,
/
"""


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _run_directory(root: Path, *, max_dom: int = 1) -> Path:
    root.mkdir(parents=True)
    (root / "namelist.input").write_text(NAMELIST.format(max_dom=max_dom), encoding="ascii")
    start = write_wrf_state(root / "wrfinput_d01")
    write_wrfbdy(root / "wrfbdy_d01", start, width=2)
    return root


def _analysis(path: Path, *, theta: float = 1.0, mass: float = 0.0, rain: float = 0.0) -> Path:
    analysis = write_wrf_state(path)
    with netCDF4.Dataset(analysis, "r+") as dataset:
        t = np.asarray(dataset["T"][0])
        t[1, 2, 3] += theta
        t[0, 0, 0] += theta          # on the rim: the boundary must follow it
        dataset["T"][0] = t
        if mass:
            mu = np.asarray(dataset["MU"][0])
            mu[3, 4] += mass
            dataset["MU"][0] = mu
        if rain:
            q = np.asarray(dataset["QRAIN"][0])
            q[2, 3, 3] += rain
            dataset["QRAIN"][0] = q
    return analysis


def _bundle(tmp_path: Path, run: Path, *, name="bundle", increment=True, **changes) -> ab.AnalysisBundle:
    analysis = _analysis(tmp_path / f"{name}-analysis.nc", **changes)
    bundle, _record = ab.make_bundle(tmp_path / name, background=run / "wrfinput_d01", analysis=analysis,
                                     iau_window_s=600.0 if increment else None, **bundle_identity())
    return bundle


def _request(bundle, **flags):
    request = {"analysis": None if bundle is None else bundle.root, "iau": None, "iau_mode": None,
               "iau_fields": None, "iau_window_seconds": None, "analysis_work": None, "plan": None}
    request.update(flags)
    return request


@pytest.fixture
def run(tmp_path):
    return _run_directory(tmp_path / "run")


# ---------------------------------------------------------------------------
# the plan
# ---------------------------------------------------------------------------
def test_door_flags_without_a_bundle_are_refused(tmp_path, run):
    with pytest.raises(door.AnalysisDoorRefusal, match="no --analysis"):
        door.resolve(_request(None, iau="on"), directory=run, outdir=tmp_path / "out")
    resolved = door.resolve(_request(None), directory=run, outdir=tmp_path / "out")
    assert resolved["plan"] is None and resolved["directory"] == run


def test_direct_insertion_runs_the_analysis_with_its_boundary_rebased(tmp_path, run):
    from gpuwm.ingest.wrfinput import _check_initial_boundary_pair
    from test_analysis_boundary import _restored, _tables

    bundle = _bundle(tmp_path, run, increment=False)
    resolved = door.resolve(_request(bundle), directory=run, outdir=tmp_path / "out")
    plan = door.load_plan(resolved["plan"])
    assert plan["start"] == "direct-insertion" and plan["iau"]["on"] is False
    work_run = resolved["directory"]
    assert work_run == Path(plan["run_directory"]) and work_run.parent.parent.name == "out.analysis-work"
    assert _sha(work_run / "wrfinput_d01") == _sha(bundle.analysis)
    assert (work_run / "namelist.input").read_bytes() == (run / "namelist.input").read_bytes()
    assert plan["record"]["boundary"]["rebased"] is True
    _check_initial_boundary_pair(_restored(work_run / "wrfinput_d01"), _tables(work_run / "wrfbdy_d01"), 2)
    assert resolved["soil_source"] == run.resolve()
    assert plan["forcing"] is None


def test_iau_starts_at_the_background_with_the_boundary_it_was_made_with(tmp_path, run):
    bundle = _bundle(tmp_path, run)
    resolved = door.resolve(_request(bundle), directory=run, outdir=tmp_path / "out")
    plan = door.load_plan(resolved["plan"])
    assert plan["start"] == "iau"
    assert plan["iau"] == {**plan["iau"], "on": True, "mode": "3d", "fields": "all", "window_seconds": 600.0,
                           "window_source": "manifest iau.window_s", "selected": "bundle default",
                           "boundary": "background"}
    work_run = resolved["directory"]
    # the background is rebuilt bit for bit, so it is the wrfinput the boundary was made from
    assert _sha(work_run / "wrfinput_d01") == _sha(run / "wrfinput_d01")
    assert _sha(work_run / "wrfbdy_d01") == _sha(run / "wrfbdy_d01")
    assert plan["record"]["boundary"] == {**plan["record"]["boundary"], "rebased": False, "onto": "background"}
    assert plan["boundary_pair_file"] is None
    at_once, spread = door.read_split(Path(plan["forcing"]["path"]), plan["forcing"]["sha256"])
    assert at_once == {}
    increment = ab.read_increment(bundle.increment)
    # a dry mass that does not move has nothing to force
    assert sorted(spread) == sorted(set(increment) - {"MU"})
    assert not np.any(increment["MU"]) and "zero everywhere" in plan["record"]["split"]["mass"]
    for name in spread:
        assert np.array_equal(spread[name], increment[name])


def test_the_analysis_boundary_knob_rebases_onto_the_analysis_and_pairs_with_it(tmp_path, run):
    from gpuwm.ingest.wrfinput import _check_initial_boundary_pair
    from test_analysis_boundary import _restored, _tables

    bundle = _bundle(tmp_path, run)
    resolved = door.resolve(_request(bundle, iau_boundary="analysis"), directory=run, outdir=tmp_path / "out")
    plan = door.load_plan(resolved["plan"])
    work_run = resolved["directory"]
    assert plan["iau"]["boundary"] == "analysis"
    assert plan["record"]["boundary"]["rebased"] is True and plan["record"]["boundary"]["onto"] == "analysis"
    assert plan["boundary_pair_file"] == str(bundle.analysis)
    _check_initial_boundary_pair(_restored(bundle.analysis), _tables(work_run / "wrfbdy_d01"), 2)
    with pytest.raises(ValueError, match="does not match initial"):
        _check_initial_boundary_pair(_restored(work_run / "wrfinput_d01"), _tables(work_run / "wrfbdy_d01"), 2)
    with pytest.raises(door.AnalysisDoorRefusal, match="G8b"):
        checkpoint = tmp_path / "ckpt"
        checkpoint.write_bytes(b"x")
        door.resolve(_request(bundle, iau_boundary="analysis"), directory=run, outdir=tmp_path / "o2",
                     restart=checkpoint)


def test_a_zero_increment_leaves_every_run_input_byte_for_byte(tmp_path, run):
    bundle, _ = ab.make_bundle(tmp_path / "zero", background=run / "wrfinput_d01",
                               analysis=run / "wrfinput_d01", iau_window_s=600.0, fields="all",
                               **bundle_identity())
    resolved = door.resolve(_request(bundle), directory=run, outdir=tmp_path / "out")
    for name in ("namelist.input", "wrfinput_d01", "wrfbdy_d01"):
        assert _sha(resolved["directory"] / name) == _sha(run / name), name
    plan = door.load_plan(resolved["plan"])
    _at_once, spread = door.read_split(Path(plan["forcing"]["path"]), plan["forcing"]["sha256"])
    assert spread and all(not np.any(value) for value in spread.values())


def test_a_surface_pressure_increment_goes_in_at_the_window_start_balanced(tmp_path, run):
    bundle = _bundle(tmp_path, run, mass=120.0)
    resolved = door.resolve(_request(bundle), directory=run, outdir=tmp_path / "out")
    plan = door.load_plan(resolved["plan"])
    at_once, spread = door.read_split(Path(plan["forcing"]["path"]), plan["forcing"]["sha256"])
    assert sorted(at_once) == ["MU", "PH"]
    assert "MU" not in spread and "T" in spread
    increment = ab.read_increment(bundle.increment)
    np.testing.assert_array_equal(at_once["PH"] + spread["PH"], increment["PH"])
    assert np.count_nonzero(at_once["PH"]) > 0
    assert "G9b" in plan["record"]["split"]["mass_at_window_start"]


def test_the_dynamics_field_set_puts_the_hydrometeors_in_at_once(tmp_path, run):
    bundle = _bundle(tmp_path, run, rain=3.0e-4)
    resolved = door.resolve(_request(bundle, iau_fields="dynamics"), directory=run, outdir=tmp_path / "out")
    plan = door.load_plan(resolved["plan"])
    at_once, spread = door.read_split(Path(plan["forcing"]["path"]), plan["forcing"]["sha256"])
    assert sorted(at_once) == ["PH", "QRAIN"]
    assert "QRAIN" not in spread and "QVAPOR" in spread


@pytest.mark.parametrize("flags, message", [
    (dict(iau_mode="split"), "G12"),
    (dict(iau_mode="4d"), "G11"),
    (dict(iau="on", iau_mode="oneshot"), "which is --iau off"),
    (dict(iau="off", iau_mode="3d"), "which is --iau on"),
    (dict(iau="off", iau_window_seconds=300.0), "shapes an IAU window"),
])
def test_contradictory_or_unbuilt_modes_are_refused_by_name(tmp_path, run, flags, message):
    bundle = _bundle(tmp_path, run)
    with pytest.raises(door.AnalysisDoorRefusal, match=message):
        door.resolve(_request(bundle, **flags), directory=run, outdir=tmp_path / "out")
    assert not (tmp_path / "out.analysis-work" / bundle.manifest_sha256[:16]).exists()


def test_iau_without_an_increment_and_a_window_past_the_run_are_refused(tmp_path, run):
    bare = _bundle(tmp_path, run, name="bare", increment=False)
    with pytest.raises(door.AnalysisDoorRefusal, match="no increment.nc"):
        door.resolve(_request(bare, iau="on"), directory=run, outdir=tmp_path / "out")
    bundle = _bundle(tmp_path, run)
    with pytest.raises(door.AnalysisDoorRefusal, match="G10"):
        door.resolve(_request(bundle), directory=run, outdir=tmp_path / "out", run_seconds=300.0)


def test_nests_are_refused_by_name(tmp_path):
    nested = _run_directory(tmp_path / "nested", max_dom=2)
    bundle = _bundle(tmp_path, nested)
    with pytest.raises(door.AnalysisDoorRefusal, match="G7"):
        door.resolve(_request(bundle), directory=nested, outdir=tmp_path / "out")


def test_a_bundle_on_another_grid_is_refused(tmp_path, run):
    other = tmp_path / "other"
    other.mkdir()
    background = write_wrf_state(other / "wrfinput_d01", hgt_offset=40.0)
    analysis = write_wrf_state(other / "analysis.nc", hgt_offset=40.0, theta_offset=0.5)
    bundle, _ = ab.make_bundle(tmp_path / "bundle", background=background, analysis=analysis,
                               iau_window_s=600.0, **bundle_identity())
    with pytest.raises(door.AnalysisDoorRefusal, match="static fields"):
        door.resolve(_request(bundle), directory=run, outdir=tmp_path / "out")


def test_a_resume_reuses_the_plan_and_a_fresh_run_refuses_a_stale_one(tmp_path, run):
    bundle = _bundle(tmp_path, run)
    first = door.resolve(_request(bundle), directory=run, outdir=tmp_path / "out")
    checkpoint = tmp_path / "out" / "restart_d01.gpuwmrst"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"a checkpoint")
    again = door.resolve(_request(bundle), directory=run, outdir=tmp_path / "out", restart=checkpoint)
    assert again["plan"] == first["plan"] and again["directory"] == first["directory"]
    with pytest.raises(door.AnalysisDoorRefusal, match="already holds a plan"):
        door.resolve(_request(bundle), directory=run, outdir=tmp_path / "out")
    with pytest.raises(door.AnalysisDoorRefusal, match="integrates the leg it continues"):
        door.resolve(_request(bundle, iau_window_seconds=300.0), directory=run, outdir=tmp_path / "out",
                     restart=checkpoint)
    background = Path(first["directory"]) / "wrfinput_d01"
    background.write_bytes(background.read_bytes()[:-4] + b"\0\0\0\1")
    with pytest.raises(door.AnalysisDoorRefusal, match="changed after it was written"):
        door.resolve(_request(bundle), directory=run, outdir=tmp_path / "out", restart=checkpoint)


def test_a_wrfout_form_analysis_starts_only_a_restart(tmp_path, run):
    analysis = _analysis(tmp_path / "a.nc")
    bundle, _ = ab.make_bundle(tmp_path / "wrfout-bundle", background=run / "wrfinput_d01", analysis=analysis,
                               iau_window_s=600.0, form="wrfout", **bundle_identity())
    with pytest.raises(door.AnalysisDoorRefusal, match="cycling restart"):
        door.resolve(_request(bundle), directory=run, outdir=tmp_path / "out")
    checkpoint = tmp_path / "ckpt"
    checkpoint.write_bytes(b"x")
    resolved = door.resolve(_request(bundle), directory=run, outdir=tmp_path / "out2", restart=checkpoint)
    plan = door.load_plan(resolved["plan"])
    assert plan["start"] == "cycling-restart" and resolved["directory"] == run.resolve()


# ---------------------------------------------------------------------------
# the worker
# ---------------------------------------------------------------------------
def _host_state(path: Path):
    raw = raw_fields(path)
    nz, ny, nx = raw["T"].shape
    zeros = lambda *shape: np.zeros(shape, np.float32)  # noqa: E731
    state = SimpleNamespace(
        thp=raw["T"].copy(), u=raw["U"].copy(), v=raw["V"].copy(), w=raw["W"].copy(), php=raw["PH"].copy(),
        mup=raw["MU"].copy(), qv=raw["QVAPOR"].copy(), qr=raw["QRAIN"].copy(),
        mub2d=raw["MUB"].copy(), c1h=raw["C1H"], c2h=raw["C2H"], c1f=raw["C1F"], c2f=raw["C2F"],
        p=(raw["P"] + raw["PB"]).astype(np.float32), thb=np.full(nz, 300.0, np.float32),
        has_msf=False, elapsed_seconds=0.0,
        rth_t=zeros(nz, ny, nx), ru_t=zeros(nz, ny, nx + 1), rv_t=zeros(nz, ny + 1, nx),
        rph_t=zeros(nz + 1, ny, nx), rw_t=zeros(nz + 1, ny, nx))
    return state


def _cfg():
    return SimpleNamespace(moist=True, mp_physics=8, hypsometric_opt=2, dt=60.0, specified=True,
                           nested=False, open_x=False, open_y=False, spec_bdy_width=2)


def _tree(state):
    node = SimpleNamespace(state=state, cfg=SimpleNamespace(run=_cfg()))
    return SimpleNamespace(root=node), {1: node}


@pytest.fixture
def no_diagnostics(monkeypatch):
    calls = []
    monkeypatch.setattr("gpuwm.core.diagnostics.update_diagnostics",
                        lambda state, hopt, **kwargs: calls.append(hopt))
    monkeypatch.setattr("gpuwm.core.moist.moist_species", lambda state: ("qv", "qr"))
    return calls


def _initialization(plan_path: Path, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    return door.AnalysisInitialization(SimpleNamespace(lateral_boundaries=None), door.load_plan(plan_path),
                                       outdir=outdir)


def test_the_worker_forces_the_increment_and_it_lands_whole(tmp_path, run, no_diagnostics):
    bundle = _bundle(tmp_path, run)
    resolved = door.resolve(_request(bundle), directory=run, outdir=tmp_path / "out")
    state = _host_state(Path(resolved["directory"]) / "wrfinput_d01")
    model, nodes = _tree(state)
    worker = _initialization(resolved["plan"], tmp_path / "out")
    worker.after_restore(model=model, nodes=nodes, start_time=VALID, run_seconds=3600.0)
    forcing = getattr(state, iau.IAU_STATE_ATTRIBUTE)
    assert forcing.fields == {"thp", "u", "v", "w", "php", "qv"}
    start = json.loads((tmp_path / "out" / "forecast-start.json").read_text())
    assert start["start"] == "iau" and start["analysis_bundle"] == bundle.manifest_sha256
    increment = ab.read_increment(bundle.increment)
    cfg = nodes[1].cfg.run
    mass = (state.mub2d + state.mup).astype(np.float64)
    added = np.zeros_like(state.thp, dtype=np.float64)
    t = 0.0
    while t < 900.0:
        state.elapsed_seconds = t
        state.rth_t[...] = 0.0
        tendency = iau.fold_step_tendencies(state, cfg, None)
        if tendency is not None:
            tendency.add_to_slow(state)
            added += cfg.dt * state.rth_t / mass[None]
        t += cfg.dt
    # the interior lands whole; the specified ring is excluded like the physics couplers
    np.testing.assert_allclose(added[:, 1:-1, 1:-1], increment["T"][:, 1:-1, 1:-1], rtol=2e-6, atol=1e-7)
    assert forcing.boundary_rows == 1
    assert forcing.receipt()["steps_forced"] == 10 and abs(forcing.applied_fraction - 1.0) < 1e-12
    for minute in range(1, 4):
        worker.step_observer(grid_id=1, model_seconds=60.0 * minute, step_count=minute)
    block = worker.run_report_blocks(model)["analysis_bundle"]
    assert block["iau_forcing"]["applied_fraction"] == pytest.approx(1.0, abs=1e-12)
    assert block["applied_now"]["applied"] is True
    assert block["balance"]["series"]["minutes"] == [1.0, 2.0, 3.0]
    assert block["balance"]["interior_margin_cells"] == 2


def test_with_the_analysis_boundary_the_forcing_leaves_the_boundary_zone(tmp_path, run, no_diagnostics):
    bundle = _bundle(tmp_path, run)
    resolved = door.resolve(_request(bundle, iau_boundary="analysis"), directory=run, outdir=tmp_path / "out")
    state = _host_state(Path(resolved["directory"]) / "wrfinput_d01")
    model, nodes = _tree(state)
    _initialization(resolved["plan"], tmp_path / "out").after_restore(
        model=model, nodes=nodes, start_time=VALID, run_seconds=3600.0)
    forcing = getattr(state, iau.IAU_STATE_ATTRIBUTE)
    assert forcing.boundary_rows == 2               # the stub's spec_bdy_width
    tendency = forcing.fold(state, nodes[1].cfg.run, None)
    rth = tendency.slow["rth_t"]
    assert not rth[:, :2, :].any() and not rth[:, :, -2:].any()


def test_a_window_the_run_left_unfinished_raises(tmp_path, run, no_diagnostics):
    bundle = _bundle(tmp_path, run)
    resolved = door.resolve(_request(bundle), directory=run, outdir=tmp_path / "out")
    state = _host_state(Path(resolved["directory"]) / "wrfinput_d01")
    model, nodes = _tree(state)
    worker = _initialization(resolved["plan"], tmp_path / "out")
    worker.after_restore(model=model, nodes=nodes, start_time=VALID, run_seconds=3600.0)
    state.elapsed_seconds = 900.0
    with pytest.raises(iau.IauError, match="G10"):
        worker.run_report_blocks(model)
    with pytest.raises(door.AnalysisDoorRefusal, match="G10"):
        _initialization(resolved["plan"], tmp_path / "out3").after_restore(
            model=model, nodes=nodes, start_time=VALID, run_seconds=300.0)


def test_a_resume_inside_the_window_rebuilds_the_forcing_and_inserts_nothing(tmp_path, run, no_diagnostics):
    bundle = _bundle(tmp_path, run, mass=80.0)
    resolved = door.resolve(_request(bundle), directory=run, outdir=tmp_path / "out")
    checkpoint = tmp_path / "ckpt.gpuwmrst"
    checkpoint.write_bytes(b"mid-window")
    state = _host_state(Path(resolved["directory"]) / "wrfinput_d01")
    before = state.mup.copy()
    state.elapsed_seconds = 300.0
    model, nodes = _tree(state)
    worker = _initialization(resolved["plan"], tmp_path / "out2")
    worker.after_restore(model=model, nodes=nodes, start_time=VALID, run_seconds=3600.0, restart=checkpoint)
    forcing = getattr(state, iau.IAU_STATE_ATTRIBUTE)
    assert forcing.applied_fraction == pytest.approx(0.5, abs=1e-12)
    assert np.array_equal(state.mup, before)          # the mass went in at t0, before the checkpoint
    assert worker.run_report_blocks(model)["analysis_bundle"]["applied_now"]["applied"] is False
    past = _host_state(Path(resolved["directory"]) / "wrfinput_d01")
    past.elapsed_seconds = 900.0
    model, nodes = _tree(past)
    _initialization(resolved["plan"], tmp_path / "out3").after_restore(
        model=model, nodes=nodes, start_time=VALID, run_seconds=3600.0, restart=checkpoint)
    assert getattr(past, iau.IAU_STATE_ATTRIBUTE, None) is None


def test_a_cycling_restart_applies_at_its_instant_once(tmp_path, run, no_diagnostics):
    bundle = _bundle(tmp_path, run, mass=60.0)
    background = tmp_path / "member-ckpt.gpuwmrst"
    background.write_bytes(b"the member's background at the valid time")
    resolved = door.resolve(_request(bundle, iau="off"), directory=run, outdir=tmp_path / "out",
                            restart=background)
    plan = door.load_plan(resolved["plan"])
    assert plan["start"] == "cycling-restart" and plan["iau"]["mode"] == "oneshot"
    increment = ab.read_increment(bundle.increment)

    def leg(restart):
        state = _host_state(run / "wrfinput_d01")
        state.elapsed_seconds = 3600.0
        model, nodes = _tree(state)
        worker = _initialization(resolved["plan"], tmp_path / "legs")
        worker.after_restore(model=model, nodes=nodes, start_time=datetime(2022, 6, 13, 22),
                             run_seconds=7200.0, restart=restart)
        return state, worker

    state, worker = leg(background)
    expected = (raw_fields(run / "wrfinput_d01")["MU"].astype(np.float64) + increment["MU"]).astype(np.float32)
    assert np.array_equal(state.mup, expected)
    block = worker.run_report_blocks(None)["analysis_bundle"]
    assert block["applied_now"]["applied"] is True
    # the restored state is the background the increment was computed against, field for field
    match = block["background_match"]
    assert match.pop("measured") is True and set(match) == {"U", "V", "W", "PH", "MU", "QVAPOR", "T"}
    assert all(value == 0.0 for value in match.values())
    assert block["balance"]["insertion_jump"]["mu_mean_abs_hpa"] > 0.0
    # the same background checkpoint again (an attempt that died): applied again
    state, _ = leg(background)
    assert np.array_equal(state.mup, expected)
    # the leg's own checkpoint at that instant: the analysis is already in it
    own = tmp_path / "own.gpuwmrst"
    own.write_bytes(b"this leg after the analysis went in")
    state, _ = leg(own)
    assert np.array_equal(state.mup, raw_fields(run / "wrfinput_d01")["MU"])


def test_a_restart_before_the_analysis_time_is_refused(tmp_path, run, no_diagnostics):
    bundle = _bundle(tmp_path, run)
    background = tmp_path / "early.gpuwmrst"
    background.write_bytes(b"x")
    resolved = door.resolve(_request(bundle), directory=run, outdir=tmp_path / "out", restart=background)
    state = _host_state(run / "wrfinput_d01")
    state.elapsed_seconds = 1800.0
    model, nodes = _tree(state)
    with pytest.raises(door.AnalysisDoorRefusal, match="not before it"):
        _initialization(resolved["plan"], tmp_path / "w").after_restore(
            model=model, nodes=nodes, start_time=datetime(2022, 6, 13, 22), run_seconds=7200.0,
            restart=background)


def test_an_increment_the_runs_physics_does_not_carry_is_refused():
    state = SimpleNamespace(thp=np.zeros((2, 2, 2)), qv=np.zeros((2, 2, 2)))
    with pytest.raises(door.AnalysisDoorRefusal, match="QHAIL"):
        door.state_increments({"QHAIL": np.zeros((2, 2, 2))}, state, _cfg())
    assert sorted(door.state_increments({"T": np.zeros((2, 2, 2)), "QVAPOR": np.zeros((2, 2, 2))},
                                        state, _cfg())) == ["qv", "thp"]
