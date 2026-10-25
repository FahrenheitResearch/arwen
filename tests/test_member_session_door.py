"""The door's act on a live model (``AnalysisInitialization.apply_door_plan``) for a resident
member session: the same function the door's restore runs, with the in-memory continuation
and the window retirement the session needs.  Host states; the forecast itself is the C15
proof on a card."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from gpuwm.da import analysis_door as door
from gpuwm.da import iau
from gpuwm.io import analysis_bundle as ab
from test_analysis_door import (_bundle, _host_state, _initialization, _request, _run_directory, _tree,
                                no_diagnostics, run)  # noqa: F401 - pytest fixtures

START = datetime(2022, 6, 13, 22)


def _cycling_plan(tmp_path, run, **changes):
    bundle = _bundle(tmp_path, run, **changes)
    background = tmp_path / "member-ckpt.gpuwmrst"
    background.write_bytes(b"the member's background at the valid time")
    resolved = door.resolve(_request(bundle), directory=run, outdir=tmp_path / "out", restart=background)
    return bundle, background, resolved


def _state_words(state):
    names = ("u", "v", "w", "thp", "php", "mup", "qv", "qr")
    return {name: np.asarray(getattr(state, name)).copy() for name in names if hasattr(state, name)}


@pytest.mark.parametrize("mass_mode", ["spread", "at-once"])
def test_the_in_memory_continuation_is_the_cycling_restart_word_for_word(
        tmp_path, run, no_diagnostics, monkeypatch, mass_mode):
    monkeypatch.setenv("GPUWM_IAU_MASS", mass_mode)
    _bundle_, background, resolved = _cycling_plan(tmp_path, run, mass=60.0)
    plan = door.load_plan(resolved["plan"])
    assert plan["start"] == "cycling-restart" and plan["iau"]["mode"] == "3d"

    def act(**how):
        state = _host_state(run / "wrfinput_d01")
        state.elapsed_seconds = 3600.0
        model, nodes = _tree(state)
        worker = _initialization(resolved["plan"], tmp_path / ("legs" if "restart" in how else "session"))
        if "restart" in how:
            worker.after_restore(model=model, nodes=nodes, start_time=START, run_seconds=7200.0, **how)
        else:
            worker.apply_door_plan(model=model, nodes=nodes, start_time=START, run_seconds=7200.0, **how)
        return state, worker

    leg_state, leg = act(restart=background)
    ledger = Path(plan["work"]) / door.LEDGER
    assert ledger.is_file() is (mass_mode == "at-once")
    if ledger.exists():
        ledger.unlink()
    live_state, live = act(continuation="in-memory")
    assert not ledger.exists()                      # no restart file, nothing to record
    for name, words in _state_words(leg_state).items():
        assert np.array_equal(words.view(np.uint8), _state_words(live_state)[name].view(np.uint8)), name
    leg_forcing = getattr(leg_state, iau.IAU_STATE_ATTRIBUTE)
    live_forcing = getattr(live_state, iau.IAU_STATE_ATTRIBUTE)
    for a, b in zip(leg_forcing.nodes, live_forcing.nodes):
        assert a.knots == b.knots and sorted(a.increments) == sorted(b.increments)
        for name in a.increments:
            assert np.array_equal(np.asarray(a.increments[name]).view(np.uint8),
                                  np.asarray(b.increments[name]).view(np.uint8)), name
    leg_block = leg.run_report_blocks(None)["analysis_bundle"]
    live_block = live.run_report_blocks(None)["analysis_bundle"]
    assert leg_block["applied_now"]["applied"] is live_block["applied_now"]["applied"] is True
    assert live_block["continuation"] == "in-memory"
    assert leg_block["background_match"] == live_block["background_match"]
    if mass_mode == "spread":
        assert leg._mass.applied_fraction == live._mass.applied_fraction == 0.0
        assert sorted(leg._mass.increments) == sorted(live._mass.increments)
        for name, values in leg._mass.increments.items():
            assert np.array_equal(np.asarray(values).view(np.uint8),
                                  np.asarray(live._mass.increments[name]).view(np.uint8)), name


def test_an_in_memory_continuation_off_its_instant_or_with_a_restart_is_refused(tmp_path, run, no_diagnostics):
    _bundle_, background, resolved = _cycling_plan(tmp_path, run)
    state = _host_state(run / "wrfinput_d01")
    state.elapsed_seconds = 3900.0
    model, nodes = _tree(state)
    with pytest.raises(door.AnalysisDoorRefusal, match="its own instant only"):
        _initialization(resolved["plan"], tmp_path / "a").apply_door_plan(
            model=model, nodes=nodes, start_time=START, run_seconds=7200.0, continuation="in-memory")
    state.elapsed_seconds = 3600.0
    with pytest.raises(door.AnalysisDoorRefusal, match="names no restart file"):
        _initialization(resolved["plan"], tmp_path / "b").apply_door_plan(
            model=model, nodes=nodes, start_time=START, run_seconds=7200.0, continuation="in-memory", restart=background)
    with pytest.raises(door.AnalysisDoorRefusal, match="continuation 'disk'"):
        _initialization(resolved["plan"], tmp_path / "c").apply_door_plan(
            model=model, nodes=nodes, start_time=START, run_seconds=7200.0, continuation="disk")


def test_a_window_is_retired_at_its_end_whole_and_the_next_attaches(tmp_path, run, no_diagnostics):
    _bundle_, _background, resolved = _cycling_plan(tmp_path, run)
    state = _host_state(run / "wrfinput_d01")
    state.elapsed_seconds = 3600.0
    model, nodes = _tree(state)
    worker = _initialization(resolved["plan"], tmp_path / "s1")
    worker.apply_door_plan(model=model, nodes=nodes, start_time=START, run_seconds=7200.0, continuation="in-memory")
    cfg = nodes[1].cfg.run
    with pytest.raises(door.AnalysisDoorRefusal, match="retired at its end"):
        worker.retire_window(model)
    t = 3600.0
    while t < 3600.0 + 600.0:                       # the bundle's 600 s window
        state.elapsed_seconds = t
        iau.fold_step_tendencies(state, cfg, None)
        t += cfg.dt
    state.elapsed_seconds = t
    block = worker.retire_window(model)
    assert block["iau_forcing"]["applied_fraction"] == pytest.approx(1.0, abs=1e-12)
    assert block["retired_at_seconds"] == t
    assert getattr(state, iau.IAU_STATE_ATTRIBUTE, None) is None


def test_a_window_retired_unfinished_fails_the_door_g10_check(tmp_path, run, no_diagnostics):
    _bundle_, _background, resolved = _cycling_plan(tmp_path, run)
    state = _host_state(run / "wrfinput_d01")
    state.elapsed_seconds = 3600.0
    model, nodes = _tree(state)
    worker = _initialization(resolved["plan"], tmp_path / "s1")
    worker.apply_door_plan(model=model, nodes=nodes, start_time=START, run_seconds=7200.0, continuation="in-memory")
    state.elapsed_seconds = 3600.0 + 900.0          # past the end, with nothing folded
    with pytest.raises(iau.IauError, match="G10"):
        worker.retire_window(model)


def test_an_in_memory_plan_applies_the_words_the_split_file_holds(tmp_path, run, no_diagnostics):
    """RC2's plan_from_exchange holds the split and the background in the plan; the door reads them where it
    reads iau-split.npz and background.nc, and the state is word for word the files plan's."""
    from gpuwm.io.analysis_bundle import _first_record, _open
    _bundle_, background, resolved = _cycling_plan(tmp_path, run, mass=60.0)
    files_plan = door.load_plan(resolved["plan"])
    at_once, spread = door.read_split(Path(files_plan["forcing"]["path"]), files_plan["forcing"]["sha256"])
    dataset = _open(files_plan["derived"]["background.nc"]["path"])
    held = {name: _first_record(dataset.variables[name]) for name in ("U", "V", "W", "PH", "MU", "QVAPOR", "T")
            if name in dataset.variables}
    memory_plan = dict(files_plan, form="in-memory", derived={}, background=held,
                       forcing={"at_once": at_once, "spread": spread, "names": sorted(set(at_once) | set(spread))})

    def act(plan, where):
        state = _host_state(run / "wrfinput_d01")
        state.elapsed_seconds = 3600.0
        model, nodes = _tree(state)
        (tmp_path / where).mkdir()
        worker = door.AnalysisInitialization(None, plan, outdir=tmp_path / where)
        worker.apply_door_plan(model=model, nodes=nodes, start_time=START, run_seconds=7200.0,
                               continuation="in-memory")
        return state, worker

    a, wa = act(files_plan, "files")
    b, wb = act(memory_plan, "memory")
    for name, words in _state_words(a).items():
        assert np.array_equal(words.view(np.uint8), _state_words(b)[name].view(np.uint8)), name
    for x, y in zip(getattr(a, iau.IAU_STATE_ATTRIBUTE).nodes, getattr(b, iau.IAU_STATE_ATTRIBUTE).nodes):
        for name in x.increments:
            assert np.array_equal(np.asarray(x.increments[name]).view(np.uint8),
                                  np.asarray(y.increments[name]).view(np.uint8)), name
    ma = wa.run_report_blocks(None)["analysis_bundle"]["background_match"]
    mb = wb.run_report_blocks(None)["analysis_bundle"]["background_match"]
    assert ma == mb and ma["measured"] is True
