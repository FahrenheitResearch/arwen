"""The cycling driver's leg boundary, on the CPU.

The driver itself needs a GPU and a prepared case; the card half of the
boundary (a continuous run against two legs joined through the restart
owner, array for array) is ``tests/test_da_cycle_join_gpu.py``.  What is
pinned here is everything the boundary decides without a device: which
owner joins the legs, how a nest born on a later leg is given its own
activation epoch, and what the analysis increment is allowed to hand the
next leg.
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# the leg join is the restart owner's, structurally
#
# The join used to be a host copy of STATE_SERIALIZED_ATTRS and a clock
# placed by hand, which carried the atmosphere and nothing else: soil,
# surface, accumulators, held tendencies and the radiation carriers
# restarted from the prepared background at every analysis.  These pin
# the PLACEMENT of the fix, because a driver that quietly grew a second
# join beside the restart owner would pass any cell that only called the
# helpers.
# ---------------------------------------------------------------------------

def _driver_source():
    import inspect

    from tools import da_cycle_prepared

    return inspect.getsource(da_cycle_prepared)


def _main_calls(name: str) -> list:
    import ast

    main = _main_body()
    return [node.lineno for node in ast.walk(main)
            if isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == name]


def test_every_leg_after_the_first_is_restored_through_the_restart_owner():
    """``restore_leg_restart`` inside the leg loop, and nothing else joins."""
    import ast

    main = _main_body()
    loops = [stmt for stmt in main.body
             if isinstance(stmt, ast.For)
             and getattr(stmt.target, "id", None) == "leg"]
    assert len(loops) == 1
    inside = [node.lineno for node in ast.walk(loops[0])
              if isinstance(node, ast.Call)
              and getattr(node.func, "id", None) in (
                  "restore_leg_restart", "write_leg_restart")]
    assert len(inside) == 2, (
        "the leg loop restores each trajectory through restore_leg_restart "
        "and writes it back through write_leg_restart, once each; found "
        f"{inside}")
    # The retired join: a clock placed by arithmetic instead of by the
    # checkpoint header.  Its name is gone from the driver altogether.
    assert "jump_clock" not in _driver_source()


def test_the_helpers_are_the_restart_owners_own_calls():
    """No private write path: the owner writes, the owner restores."""
    import inspect

    from tools.da_cycle_prepared import restore_leg_restart, write_leg_restart

    assert "write_tree_restart(" in inspect.getsource(write_leg_restart)
    assert "restore_tree_restart(" in inspect.getsource(restore_leg_restart)


def test_a_restored_set_has_to_stand_at_the_leg_it_continues(monkeypatch,
                                                              tmp_path):
    """A generation resumed at the wrong elapsed time is refused by name."""
    from types import SimpleNamespace

    from gpuwm.io import restart as restart_module
    from tools import da_cycle_prepared

    monkeypatch.setattr(
        restart_module, "restore_tree_restart",
        lambda path, model: SimpleNamespace(elapsed_ticks=900, tick_den=1))
    with pytest.raises(RuntimeError) as refusal:
        da_cycle_prepared.restore_leg_restart(
            object(), tmp_path / "gpuwmrst_d01_x.npz", expected_seconds=1800.0)
    assert "900" in str(refusal.value) and "1800" in str(refusal.value)
    info = da_cycle_prepared.restore_leg_restart(
        object(), tmp_path / "gpuwmrst_d01_x.npz", expected_seconds=900.0)
    assert info.elapsed_ticks == 900


def test_the_fingerprint_binds_the_trajectory_not_only_the_ensemble():
    """Member 3's checkpoint restored into member 5's model is refused.

    The restart owner compares fingerprints before it reads an array, so
    the fingerprint has to say WHICH member as well as which ensemble.
    """
    from tools.da_cycle_prepared import (trajectory_fingerprint,
                                         trajectory_identity)
    from tools.da_ensemble_state import EnsembleIdentity

    identity = EnsembleIdentity(
        members=4, nx=8, ny=6, nz=4, dt_s=15.0, mp_physics=8,
        physics_profile="profile-under-test-v1",
        prepared_content_sha256="c" * 64)
    assert trajectory_fingerprint(identity, 3) != trajectory_fingerprint(
        identity, 5)
    assert trajectory_fingerprint(identity, 3) == trajectory_fingerprint(
        identity, 3)
    components = trajectory_identity(identity, "control")
    assert components["trajectory"] == "control"
    assert components["ensemble"] == identity.to_payload()


# ---------------------------------------------------------------------------
# the stage holds nothing when the run ends
#
# A staged set joins one leg to the next and is consumed by the leg that
# restores it; the last leg's sets have no next leg.  Left where they
# were written, every run would leave one set per trajectory behind under
# --out (or on the tmpfs --stage-dir recommends), and a nowcast that
# cycles all day would keep one per trajectory per cycle.  The stage is
# scratch; the generation is the record.
# ---------------------------------------------------------------------------

def _fake_set(directory, seconds: int, *, domains=(1,)):
    """A set's members on disk, named as the owner names them; root is d01."""
    from pathlib import Path

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    root = None
    for gid in domains:
        member = directory / f"gpuwmrst_d{gid:02d}_{seconds:06d}.npz"
        member.write_bytes(b"x" * 1024)
        if gid == 1:
            root = member
    return root


def _leg_loop(stage, *, legs: int, trajectories, save_at, generation):
    """The driver's loop shape on the stage: consume, write, copy out."""
    import shutil

    restarts = {name: None for name in trajectories}
    for leg in range(legs):
        seconds = 60 * (leg + 1)
        for name in trajectories:
            if leg > 0:
                assert stage.holds(restarts[name])
                assert stage.consume(restarts[name]) is True
            restarts[name] = _fake_set(stage.directory(leg, name), seconds)
        analysis = stage.analysis_directory(leg)
        analysis.mkdir(parents=True)
        (analysis / "member_000.npz").write_bytes(b"m")
        if leg == save_at:
            for name in trajectories:
                shutil.copytree(restarts[name].parent,
                                generation / f"restart_{name}")
        shutil.rmtree(analysis, ignore_errors=True)
    return restarts


def test_the_stage_holds_nothing_when_the_run_ends(tmp_path):
    """Three legs, three trajectories, a generation at leg 1: at exit the
    stage is gone, the generation is whole, and the receipt counts the
    last leg's sets."""
    from tools.da_cycle_prepared import StagedRestarts

    out = tmp_path / "out"
    stage = StagedRestarts(out / "stage", default=True)
    generation = tmp_path / "generation"
    trajectories = ["control", 0, 1]
    _leg_loop(stage, legs=3, trajectories=trajectories, save_at=1,
              generation=generation)
    assert stage.inventory() == {"restart_sets": 3,
                                 "restart_bytes": 3 * 1024}
    receipt = stage.clear()
    assert receipt["restart_sets_removed"] == 3
    assert receipt["restart_bytes_removed"] == 3 * 1024
    assert receipt["analysis_directories_removed"] == 0
    assert receipt["restart_directory_left"] is False
    assert receipt["stage_root_removed"] is True
    assert not (out / "stage").exists()
    assert sorted(p.name for p in generation.iterdir()) == [
        "restart_0", "restart_1", "restart_control"]
    assert all(any(p.iterdir()) for p in generation.iterdir())
    # A second clearing is the first one's receipt, not a second removal.
    assert stage.clear() is receipt


def test_a_named_stage_directory_is_emptied_and_kept(tmp_path):
    """``--stage-dir`` is the caller's: what this run staged goes, the
    directory and what else it holds stay."""
    from tools.da_cycle_prepared import StagedRestarts

    given = tmp_path / "tmpfs"
    given.mkdir()
    (given / "not-ours").write_text("kept", encoding="utf-8")
    stage = StagedRestarts(given, default=False)
    _leg_loop(stage, legs=2, trajectories=["control"], save_at=None,
              generation=tmp_path / "unused")
    receipt = stage.clear()
    assert receipt["restart_sets_removed"] == 1
    assert receipt["stage_root_removed"] is False
    assert given.is_dir()
    assert sorted(p.name for p in given.iterdir()) == ["not-ours"]


def test_an_analysis_left_on_the_stage_goes_with_it(tmp_path):
    """A run that stopped inside an analysis leaves its member checkpoints
    staged; the clearing takes those too, and only the ones this run made."""
    from tools.da_cycle_prepared import StagedRestarts

    given = tmp_path / "shared"
    stage = StagedRestarts(given, default=False)
    mine = stage.analysis_directory(4)
    mine.mkdir(parents=True)
    (mine / "member_000.npz").write_bytes(b"m")
    other = given / "cycle_009"
    other.mkdir()
    receipt = stage.clear()
    assert receipt["analysis_directories_removed"] == 1
    assert not mine.exists()
    assert other.exists()


def test_a_generation_set_is_not_consumed_as_staged(tmp_path):
    """A set resumed from a generation lies outside the stage and stays."""
    from tools.da_cycle_prepared import StagedRestarts

    stage = StagedRestarts(tmp_path / "stage", default=True)
    elsewhere = _fake_set(tmp_path / "generation" / "restart_control", 60)
    assert stage.holds(elsewhere) is False
    assert stage.consume(elsewhere) is False
    assert elsewhere.exists()
    staged = _fake_set(stage.directory(0, "control"), 60)
    assert stage.holds(staged) is True
    assert stage.consume(staged) is True
    assert not staged.exists()
    assert not staged.parent.parent.exists()


def test_the_door_clears_the_stage_however_the_cycle_ends(monkeypatch,
                                                           tmp_path):
    """A run that stops early is cleared by the door, not left on disk."""
    from tools import da_cycle_prepared

    stage = da_cycle_prepared.StagedRestarts(tmp_path / "out" / "stage",
                                             default=True)
    staged = _fake_set(stage.directory(2, "control"), 180)

    def stops_early(stages):
        stages.append(stage)
        raise SystemExit("TREATMENT_NOT_APPLIED: nothing was assimilated")

    monkeypatch.setattr(da_cycle_prepared, "cycle", stops_early)
    with pytest.raises(SystemExit, match="TREATMENT_NOT_APPLIED"):
        da_cycle_prepared.main()
    assert not staged.exists()
    assert not (tmp_path / "out" / "stage").exists()


def test_every_staged_path_in_the_cycle_comes_from_the_stage():
    """No ``stage_root / ...`` join inside the cycle: the stage owns the
    layout, so the clearing cannot miss a path the loop composed by hand."""
    import ast

    body = _main_body()
    joins = [node.lineno for node in ast.walk(body)
             if isinstance(node, ast.BinOp)
             and isinstance(node.op, ast.Div)
             and isinstance(node.left, ast.Name)
             and node.left.id == "stage_root"]
    assert joins == []
    assert len(_main_calls("StagedRestarts")) == 1
    assert len(_main_calls("write_leg_restart")) == 1


# ---------------------------------------------------------------------------
# a nest born on a later leg activates at that leg
#
# The driver used to place the child's clock ticks by hand and leave its
# state's model time at the zero a fresh state is allocated with, so the
# child's physics counted its ITIMESTEP from the RUN's start: whether its
# first step ran radiation depended on where the leg boundary fell
# against the radiation cadence.  A child is now declared to start at its
# birth (DomainConfig.start_time), which is the one calendar every
# consumer reads.
# ---------------------------------------------------------------------------

def _nested_pair(birth_seconds: float):
    from test_da_nested_forecast import _nowcast_experiment

    from gpuwm.da import nested_forecast as nf

    exp = _nowcast_experiment()
    base = nf.nest_domain_config(exp, nf.NestGeometry(ratio=3, nx=126, ny=126))
    child = nf.child_born_at(base, exp, birth_seconds)
    return exp, child, nf.nested_experiment(exp, child)


def test_a_child_born_off_the_parent_step_is_refused_by_name():
    from test_da_nested_forecast import _nowcast_experiment

    from gpuwm.da import nested_forecast as nf

    exp = _nowcast_experiment()
    base = nf.nest_domain_config(exp, nf.NestGeometry(ratio=3, nx=126, ny=126))
    with pytest.raises(nf.NestedForecastRefusal, match="parent step"):
        nf.child_born_at(base, exp, 7.0)
    with pytest.raises(nf.NestedForecastRefusal, match="parent step"):
        nf.child_born_at(base, exp, -15.0)


def test_a_birth_off_the_forcing_cadence_is_admitted_only_when_declared():
    """The child reads no forcing snapshot at its birth, so no seam binds it.

    The forcing-seam rule exists for a DECLARED late start that is
    initialized from the forcing at that instant; a child SINT from its
    live parent is the spawn's case, and the clock resolver admits it
    only when the caller says so.
    """
    from gpuwm.core.clock import resolve_clock

    exp, child, nested = _nested_pair(270.0)
    assert child.start_time is not None
    assert (child.start_time - exp.start_time).total_seconds() == 270.0
    with pytest.raises(ValueError, match="boundary-forcing cadence"):
        resolve_clock(nested, lbc_interval_s=3600.0)
    tick = resolve_clock(nested, lbc_interval_s=3600.0,
                         live_born_children=(child.grid_id,))
    assert tick.spec(child.grid_id).start_ticks == 270 * tick.tick_den
    assert tick.spec(exp.root.grid_id).start_ticks == 0


def test_a_newborn_child_counts_its_own_first_step_as_step_one():
    """The physics driver's ITIMESTEP reads the activation epoch.

    ``refresh_model_time`` publishes ``domain_start_offset`` from the
    clock spec, and the driver forms ``itimestep = floor((now - epoch) /
    dt + 0.5) + 1``, so at birth the count is 1 and the mandatory first
    radiation call runs.  With the retired placement (epoch 0, the run's)
    the same child at 270 s and dt = 5 s stood at step 55, and the
    radiation cadence decided whether its land surface had any shortwave.
    """
    from types import SimpleNamespace

    from gpuwm.core.clock import resolve_clock
    from gpuwm.core.physics import _radiation_step_due
    from gpuwm.core.state import refresh_model_time
    from gpuwm.da import nested_forecast as nf

    exp, child, nested = _nested_pair(270.0)
    tick = resolve_clock(nested, lbc_interval_s=3600.0,
                         live_born_children=(child.grid_id,))
    clocks = tick.clocks()
    clock = clocks[child.grid_id]
    nf.place_newborn_clock(clock)
    assert clock.ticks == clock.spec.start_ticks
    assert clock.step_count == 0
    assert float(clock.dtbc_fp32) == 0.0
    assert clock.elapsed_seconds == 270.0
    # The parent and the child stand at the same instant at the boundary.
    parent = clocks[exp.root.grid_id]
    parent.ticks = clock.ticks
    assert parent.elapsed_seconds == clock.elapsed_seconds

    state = SimpleNamespace(elapsed_seconds=0.0, domain_start_offset=0.0)
    refresh_model_time(state, clock)
    assert state.elapsed_seconds == 270.0
    assert state.domain_start_offset == 270.0
    dt = float(child.run.dt)
    stepra = int(round(300.0 / dt))          # a 5-minute radiation cadence
    born = int(np.floor((state.elapsed_seconds - state.domain_start_offset)
                        / dt + 0.5)) + 1
    assert born == 1
    assert _radiation_step_due(born, stepra, 5.0)
    retired = int(np.floor((state.elapsed_seconds - 0.0) / dt + 0.5)) + 1
    assert retired == 55
    assert not _radiation_step_due(retired, stepra, 5.0)


def test_an_advanced_clock_is_not_a_newborn():
    from gpuwm.core.clock import resolve_clock
    from gpuwm.da import nested_forecast as nf

    exp, child, nested = _nested_pair(270.0)
    clock = resolve_clock(nested, lbc_interval_s=3600.0,
                          live_born_children=(child.grid_id,)).clocks()[
                              child.grid_id]
    clock.ticks = clock.spec.start_ticks + clock.spec.step_ticks
    with pytest.raises(nf.NestedForecastRefusal, match="newborn"):
        nf.place_newborn_clock(clock)



# ---------------------------------------------------------------------------
# the fine nest's command-line surface
#
# These refusals fire on the parsed arguments alone, before the driver
# touches CuPy or the prepared authority, so they are reachable from a
# CPU test and a mis-stated nest costs a second rather than a leg.
# ---------------------------------------------------------------------------

_REQUIRED_ARGS = [
    "--prepared-root", "prepared",
    "--proof-sha256", "0" * 64,
    "--source-manifest-sha256", "1" * 64,
    "--prepared-content-sha256", "2" * 64,
    "--physics-profile", "wsm6-ysu-mm5-noah-no-radiation-v1",
    "--run-seconds", "21600",
    "--history-interval-seconds", "900",
    "--out", "out",
]


def _driver_error(monkeypatch, capsys, extra):
    """Run the driver's argument parsing only and return its refusal."""
    import sys

    from tools.da_cycle_prepared import main

    monkeypatch.setattr(sys, "argv", ["da_cycle_prepared"]
                        + _REQUIRED_ARGS + extra)
    with pytest.raises(SystemExit) as exit_info:
        main()
    assert exit_info.value.code == 2
    return capsys.readouterr().err


def test_a_nest_with_no_free_legs_is_not_an_argument_refusal(monkeypatch,
                                                              capsys):
    """The child runs on the analysis legs too, so it needs no free leg.

    Retires the refusal that read "--nest-* needs --free-legs > 0".  That
    rule named a design (the parent assimilates, the child covers the
    forecast past the observations) rather than a breakage, and the
    design is what changed: a child born at the fork has no history, so
    any window comparison across the fork measures the birth as much as
    the weather.  Whatever this command fails on later -- a prepared root
    that is not there, a device that is not present -- it is not the
    argument parser refusing the combination.
    """
    import sys

    from tools.da_cycle_prepared import main

    monkeypatch.setattr(sys, "argv", ["da_cycle_prepared"] + _REQUIRED_ARGS
                        + ["--nest-half-width-km", "60",
                           "--obs", "leg0.npz", "--grid-wrfout", "leg0.nc"])
    with pytest.raises(BaseException) as info:
        main()
    refusal = capsys.readouterr().err
    assert "--nest-* needs --free-legs" not in refusal
    assert not (isinstance(info.value, SystemExit)
                and info.value.code == 2), refusal


def test_nest_members_cannot_exceed_the_parent_ensemble(monkeypatch, capsys):
    message = _driver_error(monkeypatch, capsys, [
        "--nest-half-width-km", "60", "--free-legs", "6",
        "--members", "10", "--nest-members", "11"])
    assert "exceeds --members" in message


def test_nest_members_without_a_nest_is_refused(monkeypatch, capsys):
    message = _driver_error(monkeypatch, capsys, [
        "--free-legs", "6", "--nest-members", "2"])
    assert "without a nest extent" in message


# ---------------------------------------------------------------------
# the background-source surface, at the driver's own front door
# ---------------------------------------------------------------------

def test_the_driver_offers_the_background_roster_and_defaults_to_gfs(
        monkeypatch, capsys):
    """Selecting nothing is selecting GFS, and the roster is the registry.

    The prepared root already IS one source's case; this flag is the
    caller's statement of which, and the front door refuses a
    disagreement.  What matters here is that the default is unchanged,
    so an existing invocation keeps its existing meaning.
    """

    import sys

    from gpuwm.da import background
    from tools import da_cycle_prepared

    monkeypatch.setattr(sys, "argv", ["da_cycle_prepared", "--help"])
    with pytest.raises(SystemExit) as exit_info:
        da_cycle_prepared.main()
    assert exit_info.value.code == 0
    text = " ".join(capsys.readouterr().out.split())
    for name in background.BACKGROUND_SOURCES:
        assert name in text
    assert background.DEFAULT_BACKGROUND_SOURCE == "gfs"


def test_the_driver_refuses_a_source_it_has_no_background_registry_for(
        monkeypatch, capsys):
    """THE DRIVER'S OWN SENTENCE, not the interpreter's.

    This cell used to assert argparse's "invalid choice", which bound it
    to two moving things at once.  ``--source`` carried
    ``choices=sorted(BACKGROUND_SOURCES)``, and that registry is a LIVE
    projection of the runnable source table -- so when 20CRv3 became
    runnable the name this cell passed turned into a VALID choice, the
    parser fell through to "the following arguments are required", and
    the file failed for a reason that had nothing to do with what it was
    written to pin.  argparse's wording is not ours to pin either: 3.12
    dropped the quotes from the choice list and 3.13 put them back.  The
    driver now states the refusal itself, and this asserts that sentence.
    """

    import sys

    from gpuwm.da import background
    from tools import da_cycle_prepared

    # A source gpuwm's adapter table HAS and the background registry does
    # not: the table marks it not runnable.  Read from the registry rather
    # than typed, so a source that becomes runnable cannot quietly turn
    # this cell into the pass-by-accident it was.
    assert "nam" not in background.BACKGROUND_SOURCES
    monkeypatch.setattr(
        sys, "argv", ["da_cycle_prepared", "--source", "nam"])
    with pytest.raises(SystemExit) as exit_info:
        da_cycle_prepared.main()
    assert exit_info.value.code == 2
    stderr = capsys.readouterr().err
    assert "nam has no background registry entry" in stderr
    assert "not runnable" in stderr
    for known in sorted(background.BACKGROUND_SOURCES):
        assert known in stderr


def test_the_driver_still_accepts_every_source_the_registry_carries(
        monkeypatch, capsys):
    """The other half: a registry source reaches the required-argument
    check instead of being refused for its name."""

    import sys

    from gpuwm.da import background
    from tools import da_cycle_prepared

    monkeypatch.setattr(
        sys, "argv", ["da_cycle_prepared", "--source", "20crv3"])
    with pytest.raises(SystemExit) as exit_info:
        da_cycle_prepared.main()
    assert exit_info.value.code == 2
    stderr = capsys.readouterr().err
    assert "20crv3" in background.BACKGROUND_SOURCES
    assert "has no background registry entry" not in stderr
    assert "the following arguments are required" in stderr


# ---------------------------------------------------------------------
# plan review for the DA door: the refusal has to land before the legs
# ---------------------------------------------------------------------
#
# Audit R-051 moved the radar operator's scheme refusals into
# ``RadarAssimilationConfig.__post_init__``, which means they fire
# wherever that configuration is first BUILT.  In this driver that used
# to be inside ``for leg in range(legs)``, at the first analysis seam --
# so a cycle whose scheme has no H(x) burned leg 0's whole ensemble
# integration before being told.  These cells pin the placement, not the
# wording: the driver builds the configuration once above the leg loop,
# and the leg's own configuration comes from that same function.


def _plan_args(**updates):
    """The knobs ``plan_radar_assimilation`` reads, at driver defaults."""
    from types import SimpleNamespace

    values = dict(
        horizontal_loc_m=12000.0, vertical_loc_m=3000.0, rtps_alpha=0.9,
        relaxation="rtps", thin_cells=1, err_inflation=1.0, z_thin_cells=1,
        z_err_inflation=1.0, z0_thin_cells=4, z0_err_inflation=1.0,
        cwp_thin_cells=1, cwp_err_inflation=1.0,
        cwp_horizontal_loc_m=None, cwp_vertical_loc_m=None,
        positivity_policy="clip", solve_device="host",
        memory_budget_mib=512.0, hydrometeors=True,
        reflectivity_analysis=False, clear_air_analysis=False,
        goes_cwp=[])
    values.update(updates)
    return SimpleNamespace(**values)


def _routed_and_fused_schemes():
    from gpuwm.physics_registry import consumer_rows_by_selector

    rows = consumer_rows_by_selector("microphysics", "radar_da")
    routed = sorted(mp for mp, row in rows.items()
                    if row.get("reflectivity_route")
                    in ("operator", "scheme-diagnostic"))
    fused = sorted(mp for mp, row in rows.items()
                   if row.get("reflectivity_route") == "native-not-separable")
    return routed, fused


def test_the_plan_probe_refuses_a_scheme_the_radar_operator_cannot_simulate():
    """The refusal a cycle used to get after leg 0, before leg 0.

    Same function the leg builds its configuration with, so what is
    checked here is what will run.
    """
    from gpuwm.da.radar_assimilation import RadarAssimilationError
    from tools.da_cycle_prepared import (plan_radar_assimilation,
                                         planned_analysis_fields)

    routed, fused = _routed_and_fused_schemes()
    assert routed and fused, (routed, fused)

    args = _plan_args(reflectivity_analysis=True)
    with pytest.raises(RadarAssimilationError) as refusal:
        plan_radar_assimilation(
            args, fused[0], cwp=False,
            analysis_fields=planned_analysis_fields(args, fused[0]))
    message = str(refusal.value)
    # A refusal names its way out, and the way out has to work.
    assert "reflectivity=False" in message, message
    kept = plan_radar_assimilation(
        _plan_args(reflectivity_analysis=False), fused[0], cwp=False,
        analysis_fields=planned_analysis_fields(_plan_args(), fused[0]))
    assert kept.reflectivity is False and kept.velocity

    ok = plan_radar_assimilation(
        args, routed[0], cwp=False,
        analysis_fields=planned_analysis_fields(args, routed[0]))
    assert ok.mp_physics == routed[0] and ok.reflectivity


def test_the_plan_probe_asks_for_the_fields_the_legs_will_analyse():
    """Plan time is the leg's field set before spread narrows it.

    The leg drops whole species the ensemble is constant in, so its set
    is a subset of this one.  That direction is what makes the probe
    safe: it cannot refuse a cycle the legs would have run.
    """
    from gpuwm.da import moments
    from tools.da_cycle_prepared import planned_analysis_fields

    routed, _ = _routed_and_fused_schemes()
    mp = routed[0]
    assert planned_analysis_fields(_plan_args(hydrometeors=False), mp) == (
        "u", "v")
    assert planned_analysis_fields(_plan_args(), mp) == tuple(
        moments.analysis_fields(int(mp)))


def _main_body():
    import ast
    import inspect

    from tools import da_cycle_prepared

    tree = ast.parse(inspect.getsource(da_cycle_prepared))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "cycle":
            return node
    raise AssertionError("tools/da_cycle_prepared.py has no cycle()")


def test_the_driver_plans_the_analysis_before_it_integrates_a_leg():
    """Deleting the probe, or sinking it into the loop, fails here.

    Structural on purpose: the defect is a PLACEMENT, and a cell that
    only called the function would still pass with the call sitting
    where it was -- inside ``for leg in range(legs)``, one ensemble
    integration too late.
    """
    import ast

    main = _main_body()
    probes = [stmt.lineno for stmt in main.body
              if isinstance(stmt, ast.Expr)
              and isinstance(stmt.value, ast.Call)
              and getattr(stmt.value.func, "id", None)
              == "plan_radar_assimilation"]
    assert len(probes) == 1, (
        "the DA plan review must be called exactly once at the top level "
        f"of main(), found {len(probes)}")
    loops = [stmt.lineno for stmt in main.body
             if isinstance(stmt, ast.For)
             and getattr(stmt.target, "id", None) == "leg"]
    assert len(loops) == 1, loops
    assert probes[0] < loops[0], (
        "the analysis configuration is built inside the leg loop, so its "
        "refusals arrive with an ensemble integration already spent")


def test_every_analysis_configuration_in_the_driver_comes_from_one_function():
    """No second construction, so the reviewed plan is the one that runs."""
    import ast

    main = _main_body()
    direct = [node.lineno for node in ast.walk(main)
              if isinstance(node, ast.Call)
              and getattr(node.func, "id", None) == "RadarAssimilationConfig"]
    assert not direct, (
        "RadarAssimilationConfig is constructed directly in main() at "
        f"{direct}; every construction goes through plan_radar_assimilation")
    through = [node.lineno for node in ast.walk(main)
               if isinstance(node, ast.Call)
               and getattr(node.func, "id", None) == "plan_radar_assimilation"]
    assert len(through) == 2, (
        "main() should build the plan-time configuration and the leg's own "
        f"through the same function, found {len(through)} call(s)")


# -- the hot start's insertion summed onto the filter's increment ----------
#
# Each half is bounded on its own and the sum was not, which is how a run
# with the clear-air branch on put water vapour at -1.02e-4 kg/kg and the
# next leg refused the state.  These pin the SUM.


def _merge_fixture():
    """A background and two halves that are each admissible alone."""
    from tools.da_cycle_prepared import merge_hotstart_increments

    rng = np.random.default_rng(11)
    prior = {"qv": (rng.random((4, 5, 6)) * 1e-2 + 1e-3).astype(np.float32),
             "thp": np.zeros((4, 5, 6), np.float32)}
    # The filter takes 60% of the vapour away; so does the insertion.
    # prior + either one is non-negative; prior + both is not.
    half = (-0.6 * prior["qv"]).astype(np.float32)
    filter_increments = {"qv": half.copy(),
                         "thp": np.full((4, 5, 6), 0.5, np.float32)}
    hot_increments = {"qv": half.copy(),
                      "thp": np.full((4, 5, 6), -0.2, np.float32)}
    return merge_hotstart_increments, prior, filter_increments, hot_increments


def test_the_summed_increment_is_bounded_not_just_each_half():
    merge, prior, filt, hot = _merge_fixture()
    for name in ("qv",):
        assert (prior[name] + filt[name]).min() >= 0.0
        assert (prior[name] + hot[name]).min() >= 0.0
    # The naive sum -- what the driver used to hand the next leg.
    naive = prior["qv"] + filt["qv"] + hot["qv"]
    assert naive.min() < 0.0, "the fixture must reproduce the defect"

    merged, overlap, receipt = merge(
        filt, hot, prior=prior, positivity_policy="clip")
    analysis = prior["qv"] + merged["qv"]
    assert analysis.min() >= 0.0
    # Clip lands exactly on zero where the sum went under, and leaves the
    # rest of the increment alone.
    np.testing.assert_allclose(analysis, 0.0, atol=1e-9)
    assert receipt is not None
    assert receipt["policy"] == "clip"
    assert receipt["negative_points"] == prior["qv"].size
    assert receipt["constrained_fields"] == ["qv"]
    # An unconstrained field is the plain sum, untouched by the policy.
    np.testing.assert_allclose(merged["thp"], 0.3, atol=1e-6)
    assert set(overlap) == {"qv", "thp"}
    assert overlap["qv"]["hotstart_rms"] > 0.0


def test_reject_reverts_the_whole_increment_where_the_sum_goes_under():
    merge, prior, filt, hot = _merge_fixture()
    merged, _, receipt = merge(
        filt, hot, prior=prior, positivity_policy="reject")
    assert (prior["qv"] + merged["qv"]).min() >= 0.0
    np.testing.assert_allclose(merged["qv"], 0.0, atol=1e-12)
    assert receipt["policy"] == "reject"


def test_a_run_that_stated_no_policy_gets_the_plain_sum_and_says_so():
    merge, prior, filt, hot = _merge_fixture()
    merged, _, receipt = merge(
        filt, hot, prior=prior, positivity_policy=None)
    assert receipt is None
    np.testing.assert_allclose(merged["qv"], filt["qv"] + hot["qv"],
                               rtol=1e-6)


def test_without_an_insertion_the_filter_increment_is_handed_on_unchanged():
    merge, prior, filt, _ = _merge_fixture()
    merged, overlap, receipt = merge(
        filt, {}, prior=prior, positivity_policy="clip")
    # The policy runs on every leg, because this mapping is what the
    # generation saves; re-binding an increment the filter already bound
    # against this same background leaves it exactly as it was.
    assert receipt is not None and overlap == {}
    for name, values in filt.items():
        np.testing.assert_array_equal(merged[name], values)


def test_a_field_only_the_insertion_carries_is_not_dropped():
    merge, prior, filt, hot = _merge_fixture()
    filt = {name: values for name, values in filt.items() if name != "thp"}
    merged, overlap, _ = merge(
        filt, hot, prior=prior, positivity_policy="clip")
    np.testing.assert_array_equal(merged["thp"], hot["thp"])
    assert set(overlap) == {"qv"}


# --- the analysis increment cannot hand the next leg a broken moment pair ---
#
# An ensemble filter writes every field its own additive increment, so a cell can leave the
# solve with mass above the policy's threshold and a number moment at or below zero.  The
# moment policy then refuses the leg that applies it, which is correct and which stops a
# cycle dead.  These pin the conditioning that keeps the pair whole without inventing an
# intercept: the background's own ratio where the background holds the species, and a
# declined mass increment where it holds none.

class _FakeState:
    def __init__(self, **fields):
        for name, value in fields.items():
            setattr(self, name, np.asarray(value, dtype=np.float64))


def _pairs_fix(state, increment):
    from tools.da_cycle_prepared import keep_moment_pairs
    return keep_moment_pairs(state, increment, mp_physics=28)


def test_a_number_moment_driven_to_zero_takes_the_background_ratio():
    state = _FakeState(qc=np.array([[1.0e-4]]), nc=np.array([[5.0e7]]))
    fixed, report = _pairs_fix(state, {"qc": np.array([[1.0e-4]]),
                                       "nc": np.array([[-6.0e7]])})
    # Mass doubled, so the number doubles: the drop size distribution
    # is the background's.
    np.testing.assert_allclose(state.nc + fixed["nc"], 1.0e8, rtol=1e-12)
    np.testing.assert_allclose(fixed["qc"], 1.0e-4, rtol=1e-12)
    assert report["cells_rescaled"] == 1
    assert report["cells_declined"] == 0


def test_mass_created_where_the_background_had_none_is_declined():
    state = _FakeState(qr=np.array([[0.0]]), nr=np.array([[0.0]]))
    fixed, report = _pairs_fix(state, {"qr": np.array([[2.0e-4]]),
                                       "nr": np.array([[-1.0]])})
    assert fixed["qr"].item() == 0.0
    assert fixed["nr"].item() == 0.0
    assert report["cells_declined"] == 1
    assert report["cells_rescaled"] == 0


def test_a_healthy_increment_is_returned_unchanged():
    state = _FakeState(qc=np.array([[1.0e-4]]), nc=np.array([[5.0e7]]))
    inc = {"qc": np.array([[1.0e-5]]), "nc": np.array([[1.0e6]])}
    fixed, report = _pairs_fix(state, inc)
    for name in inc:
        np.testing.assert_array_equal(fixed[name], inc[name])
    assert report["cells_rescaled"] == 0 and report["cells_declined"] == 0
    assert report["conditioned"] is False


def test_a_pair_whose_number_the_analysis_never_named_still_gets_one():
    # The filter named the mass alone; the conditioning has to be able
    # to write the number.
    state = _FakeState(qc=np.array([[1.0e-4]]), nc=np.array([[0.0]]))
    fixed, report = _pairs_fix(state, {"qc": np.array([[1.0e-4]])})
    # No background distribution to carry forward.
    assert fixed["qc"].item() == 0.0
    assert report["cells_declined"] == 1


def test_the_condensate_threshold_is_the_running_scheme_s_own():
    """1e-12 is Thompson's activity gate, not everybody's.

    Morrison reads a number moment two orders of magnitude lower
    (MQSMALL 1e-14), so a cell carrying 1e-13 of rain with no number is
    broken under Morrison and unremarkable under Thompson.  Holding one
    threshold for both left a Morrison run's smallest broken cells
    unconditioned here and refused by the moment policy at the next leg,
    which is the refusal this conditioning exists to prevent.
    """
    from tools.da_cycle_prepared import (keep_moment_pairs,
                                         moment_mass_threshold)
    assert moment_mass_threshold(8) == 1e-12
    assert moment_mass_threshold(10) == 1e-14

    state = _FakeState(qr=np.array([[0.0]]), nr=np.array([[0.0]]))
    increment = {"qr": np.array([[1.0e-13]]), "nr": np.array([[-1.0]])}

    fixed, report = keep_moment_pairs(state, dict(increment), mp_physics=10)
    assert report["cells_declined"] == 1, (
        "Morrison demands a number above 1e-14, so this cell is broken")
    assert fixed["qr"].item() == 0.0

    fixed, report = keep_moment_pairs(state, dict(increment), mp_physics=8)
    assert report["cells_declined"] == 0, (
        "Thompson does not read a number below 1e-12, so the same cell "
        "is not broken and nothing should touch it")
    np.testing.assert_array_equal(fixed["qr"], increment["qr"])


def test_an_unknown_scheme_falls_back_where_the_moment_policy_falls_back():
    """One answer for an unresolvable scheme, not two."""
    from tools.da_cycle_prepared import (
        MOMENT_MASS_THRESHOLD_FALLBACK_KG_KG, moment_mass_threshold)
    from gpuwm.da.moments import _MORRISON
    assert (MOMENT_MASS_THRESHOLD_FALLBACK_KG_KG
            == _MORRISON.q_threshold)
    assert moment_mass_threshold(-1) == MOMENT_MASS_THRESHOLD_FALLBACK_KG_KG


def test_an_unresolvable_scheme_cannot_quietly_condition_a_leg(monkeypatch):
    """The fallback answers a caller that asks; it never stands in here.

    A bare ``except Exception`` stood in front of the scheme lookup and
    returned Morrison's 1e-14 for anything at all: an import error, a
    renamed attribute, a typo in a caller's ``mp_physics``.  Thompson
    would then have been conditioned two orders of magnitude below its
    own activity gate with nothing saying so, which is a quieter
    spelling of the single-threshold defect the per-scheme lookup
    closed.

    Three properties, each measured rather than assumed: the catch is
    narrow, so a failure that is not a resolution failure propagates;
    the conditioning cannot reach the fallback at all, because the
    moment policy refuses an unregistered scheme by name first; and the
    record says which mass the leg was conditioned against.
    """
    from gpuwm.da import moments as moments_module
    from tools.da_cycle_prepared import (
        MOMENT_MASS_THRESHOLD_FALLBACK_KG_KG, keep_moment_pairs,
        moment_mass_threshold, resolved_moment_mass_threshold)

    assert resolved_moment_mass_threshold(8) == (1e-12, True)
    assert resolved_moment_mass_threshold(-1) == (
        MOMENT_MASS_THRESHOLD_FALLBACK_KG_KG, False)

    state = _FakeState(qr=np.array([[0.0]]), nr=np.array([[0.0]]))
    increment = {"qr": np.array([[1.0e-13]]), "nr": np.array([[-1.0]])}

    with pytest.raises(moments_module.MomentPolicyError) as refusal:
        keep_moment_pairs(state, dict(increment), mp_physics=-1)
    assert "mp_physics=-1" in str(refusal.value), str(refusal.value)

    _, report = keep_moment_pairs(state, dict(increment), mp_physics=10)
    assert report["mass_threshold_kg_kg"] == 1e-14
    _, report = keep_moment_pairs(state, dict(increment), mp_physics=8)
    assert report["mass_threshold_kg_kg"] == 1e-12

    class _SchemeWithoutTheAttribute:
        pass

    monkeypatch.setattr(moments_module, "scheme_moments",
                        lambda *a, **k: _SchemeWithoutTheAttribute())
    with pytest.raises(AttributeError):
        moment_mass_threshold(8)


def test_a_wind_only_increment_is_not_touched():
    state = _FakeState(u=np.array([[1.0]]), v=np.array([[1.0]]))
    inc = {"u": np.array([[0.5]]), "v": np.array([[-0.5]])}
    fixed, report = _pairs_fix(state, inc)
    assert fixed is inc
    assert report["conditioned"] is False


# --- the conditioning rescales only from an ACTIVE background ---------------
#
# Any positive background mass used to qualify for the rescale, so a cloud
# the scheme had already evaporated to 5e-15 kg/kg with a droplet number of
# 5e4 left standing was scaled by an increment of 1e-3 kg/kg: a factor of
# 2e11, a droplet number of 1e16 per kilogram, above the 1e15 the pre-leg
# health gate admits, and the next leg refused the state.  These pin the
# two rules that close it: below the scheme's threshold there is no
# distribution to carry, and a rescaled number the gate would refuse
# declines the whole pair rather than being capped.


def test_a_nearly_evaporated_background_is_not_a_distribution_to_carry():
    from gpuwm.core.health import rule_for_field

    state = _FakeState(qc=np.array([[5.0e-15]]), nc=np.array([[5.0e4]]))
    fixed, report = _pairs_fix(state, {"qc": np.array([[1.0e-3]]),
                                       "nc": np.array([[-1.0e5]])})
    analysis_nc = float((state.nc + fixed["nc"]).item())
    assert analysis_nc <= rule_for_field("nc").upper
    # The whole pair declines to the prior: no mass, no number.
    assert fixed["qc"].item() == 0.0
    assert fixed["nc"].item() == 0.0
    assert report["cells_rescaled"] == 0
    assert report["cells_declined"] == 1
    assert report["species"]["cloud"]["declined"] == 1


def test_an_active_background_whose_rescale_would_pass_the_ceiling_declines():
    """Rescale, check against the gate's own ceiling, decline; never cap."""
    from gpuwm.core.health import rule_for_field

    ceiling = rule_for_field("nc").upper
    # Active (above Thompson's 1e-12) with a number, and an increment that
    # multiplies the mass by 1e11: the background's distribution carried
    # forward would put the number at 5e15, past the ceiling.
    state = _FakeState(qc=np.array([[1.0e-11]]), nc=np.array([[5.0e4]]))
    fixed, report = _pairs_fix(state, {"qc": np.array([[1.0e0]]),
                                       "nc": np.array([[-1.0e5]])})
    assert float((state.nc + fixed["nc"]).item()) <= ceiling
    assert fixed["qc"].item() == 0.0 and fixed["nc"].item() == 0.0
    assert report["cells_rescaled"] == 0
    assert report["cells_declined"] == 1
    assert report["cells_declined_for_ceiling"] == 1
    assert report["species"]["cloud"]["declined_for_ceiling"] == 1
    assert report["species"]["cloud"]["number_ceiling"] == ceiling
    # ...and a capped number is not what came back.
    assert float((state.nc + fixed["nc"]).item()) != ceiling


def test_an_active_background_under_the_ceiling_still_carries_its_ratio():
    """The rule closes the edge; the healthy rescale is untouched."""
    state = _FakeState(qc=np.array([[1.0e-4]]), nc=np.array([[5.0e7]]))
    fixed, report = _pairs_fix(state, {"qc": np.array([[1.0e-4]]),
                                       "nc": np.array([[-6.0e7]])})
    np.testing.assert_allclose(state.nc + fixed["nc"], 1.0e8, rtol=1e-12)
    assert report["cells_rescaled"] == 1
    assert report["cells_declined_for_ceiling"] == 0


def test_the_volume_moment_follows_the_pair_it_belongs_to():
    """NSSL's predicted volume declines with the mass and number it rides."""
    from gpuwm.da import moments

    scheme = moments.scheme_moments(18)
    with_volume = [pair for pair in scheme.pairs if pair.volume]
    if not with_volume:
        pytest.skip("this NSSL mode predicts no volume moment")
    pair = with_volume[0]
    fields = {pair.mass: np.array([[5.0e-15]]),
              pair.number: np.array([[5.0e4]]),
              pair.volume: np.array([[1.0e-9]])}
    state = _FakeState(**fields)
    from tools.da_cycle_prepared import keep_moment_pairs
    fixed, report = keep_moment_pairs(
        state, {pair.mass: np.array([[1.0e-3]]),
                pair.number: np.array([[-1.0e5]]),
                pair.volume: np.array([[2.0e-9]])}, mp_physics=18)
    assert fixed[pair.mass].item() == 0.0
    assert fixed[pair.number].item() == 0.0
    assert fixed[pair.volume].item() == 0.0
    assert report["cells_declined"] == 1


def test_the_repair_path_cannot_produce_a_number_above_the_gate():
    """The scheme bounders stay under the health ceiling across their range.

    The repair in ``gpuwm.ensemble.increments`` writes the scheme's own
    entry or limiter number where an increment left mass with no number.
    Measured here over every mass a state can carry and every density a
    column can have, so the conditioning above is the only place a number
    past the ceiling could have come from.
    """
    from gpuwm.core.health import rule_for_field
    from gpuwm.da import moments

    masses = np.logspace(-12, -1, 45)
    densities = np.array([0.05, 0.3, 1.0, 1.4])
    grid_q, grid_rho = np.meshgrid(masses, densities)
    for mp in (8, 28):
        scheme = moments.scheme_moments(mp)
        state = {"alt": 1.0 / grid_rho}
        for pair in scheme.pairs:
            state[pair.mass] = grid_q
            state[pair.number] = np.zeros_like(grid_q)
        bounded = moments._thompson_bounded_numbers(
            state, scheme.pairs, q_threshold=scheme.q_threshold)
        for pair in scheme.pairs:
            ceiling = rule_for_field(pair.number).upper
            assert np.isfinite(bounded[pair.number]).all()
            assert bounded[pair.number].max() <= ceiling, (mp, pair.number)
    scheme = moments.scheme_moments(10)
    state = {}
    for pair in scheme.pairs:
        state[pair.mass] = grid_q
        state[pair.number] = np.zeros_like(grid_q)
    bounded = moments._morrison_bounded_numbers(state, scheme.pairs)
    for pair in scheme.pairs:
        ceiling = rule_for_field(pair.number).upper
        assert np.isfinite(bounded[pair.number]).all()
        assert bounded[pair.number].max() <= ceiling, pair.number


# ---------------------------------------------------------------------------
# the nest's own picture
# ---------------------------------------------------------------------------
#
# The nest became reachable from the nowcast front door while its product
# was an ``.npz`` and nothing else, so the child forecast had no route to
# the renderer the render law names: rw_wrfbatch reads wrfout files, an
# ``.npz`` states no geolocation, and no gallery panel had adopted the
# render tool's nest entry points.  The driver now writes the child's
# composite as a wrfout beside its ``.npz``, exactly as it does the
# parent's.  These pin the file that makes that possible; the nested run
# under proof/ is the artifact.


def _composite_grid(ny: int, nx: int):
    """A grid object with the two things the writer reads off one."""
    from types import SimpleNamespace

    lat = np.tile(np.linspace(35.0, 36.0, ny, dtype=np.float32)[:, None],
                  (1, nx))
    lon = np.tile(np.linspace(-98.0, -97.0, nx, dtype=np.float32)[None, :],
                  (ny, 1))
    return SimpleNamespace(
        truelat1=33.0, truelat2=37.0, stand_lon=-97.5,
        ref_lat=35.5, ref_lon=-97.5,
        latlon_mass=lambda: (lat, lon))


def _child_domain():
    from types import SimpleNamespace

    return SimpleNamespace(grid_id=2, parent_id=1, i_parent_start=54,
                           j_parent_start=53, parent_grid_ratio=3)


def _write(tmp_path, *, domain, ny=12, nx=15, dx=1000.0, dt=5.0):
    import datetime
    from types import SimpleNamespace

    from tools.da_cycle_prepared import _write_composite_wrfout

    suffix = "" if domain is None else f"_d{domain.grid_id:02d}"
    npz = tmp_path / f"leg02_control{suffix}.npz"
    colmax = np.linspace(-20.0, 58.0, ny * nx,
                         dtype=np.float32).reshape(ny, nx)
    return _write_composite_wrfout(
        npz, colmax, _composite_grid(ny, nx),
        SimpleNamespace(dt=dt, dx=dx, dy=dx), 900.0,
        SimpleNamespace(start_time=datetime.datetime(2026, 9, 17, 22, 0)),
        label="leg 02 member control", domain=domain)


def test_the_child_frame_states_which_domain_it_is(tmp_path):
    """A reader learns the domain from the file, not from the name."""
    netCDF4 = pytest.importorskip("netCDF4")

    child = _child_domain()
    path = _write(tmp_path, domain=child)
    assert path is not None and path.name == "wrfout_leg02_control_d02.nc"
    with netCDF4.Dataset(path) as dataset:
        assert int(dataset.GRID_ID) == 2
        assert int(dataset.PARENT_ID) == 1
        assert int(dataset.I_PARENT_START) == child.i_parent_start
        assert int(dataset.J_PARENT_START) == child.j_parent_start
        assert int(dataset.PARENT_GRID_RATIO) == 3
        # the child's own spacing and step, not the parent's
        assert float(dataset.DX) == 1000.0
        assert float(dataset.DT) == 5.0
        assert dataset.variables["REFL_10CM"].shape[0] == 1


def test_the_parent_frame_is_unchanged_by_the_nest_argument(tmp_path):
    """The root domain keeps writing exactly what it wrote before."""
    netCDF4 = pytest.importorskip("netCDF4")

    path = _write(tmp_path, domain=None, dx=3000.0, dt=15.0)
    assert path is not None and path.name == "wrfout_leg02_control.nc"
    with netCDF4.Dataset(path) as dataset:
        assert not hasattr(dataset, "GRID_ID")
        assert not hasattr(dataset, "PARENT_GRID_RATIO")


def test_both_frames_measure_their_lead_from_the_same_origin(tmp_path):
    """The child cannot label the same instant with a different lead.

    ``rw_wrfbatch`` reads the run origin first and measures every
    product's lead from it.  A child that stamped its OWN start there
    would put the d02 frame of an instant at a different lead than the
    d01 frame beside it -- the failure ``wrf_global_attrs`` documents.
    """
    netCDF4 = pytest.importorskip("netCDF4")

    parent = _write(tmp_path / "p", domain=None, dx=3000.0, dt=15.0)
    child = _write(tmp_path / "c", domain=_child_domain())
    with netCDF4.Dataset(parent) as one, netCDF4.Dataset(child) as two:
        assert one.SIMULATION_START_DATE == two.SIMULATION_START_DATE
        assert one.START_DATE == two.START_DATE
