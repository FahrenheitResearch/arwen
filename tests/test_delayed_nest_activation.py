"""A nest that activates later than the experiment runs to completion.

Defect #205, the root fix.  Every history frame after the EXPERIMENT's
t = 0 consumed a microphysics-time REFL_10CM stash, so a nest whose first
frame is due at its own later activation epoch consumed a stash no step of
that domain had produced and the run died there:

    RuntimeError: REFL_10CM output is due but no microphysics-time field
    is stashed

deterministically, hours of integration in (32,350 s burned across three
identical supervisor restarts, 2026-08-19).  2.5.0 shipped an upfront
refusal at ``build_experiment`` naming that breakage; this suite is what
retires it.

The contract the consume sites now hold: a domain's stash exists only
after a microphysics step OF THAT DOMAIN has run with ``refl_10cm_due``.
The frame due at the domain's OWN start tick precedes every one of its
steps -- for a domain that starts with the experiment that tick is 0, for
an activating nest it is the activation epoch -- so that one frame carries
no REFL_10CM, exactly as the root's analysis frame always has.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.core.clock import build_schedule, resolve_clock
from gpuwm.core.model import (DomainNode, ExperimentState,
                              ModelRuntimeStatus, execute_experiment)
from gpuwm.core.refl import stash_refl_10cm
from gpuwm.static.lambert import grids_from_projection_config
from gpuwm.verify.cases.nest_ideal_r1_moist import load_scaffold

from test_model import _Coupler, _HistoryPhysics, _HistoryState

#: The child activates two root steps (120 s) after the experiment, on a
#: forcing seam and on a parent step boundary, so no structural refusal
#: can answer instead and the activation epoch is a history boundary.
DELAY_SECONDS = 120
RUN_SECONDS = 300.0
HISTORY_SECONDS = 60.0


class _Writers:
    """Records what the production history handoff hands the writer."""

    def __init__(self):
        self.frames = []
        self.drains = 0

    @property
    def pending(self):
        return 0

    def submit(self, node, ticks, *, refl_field=None):
        self.frames.append((node.cfg.grid_id, ticks, refl_field))

    def drain(self):
        self.drains += 1


def _stashing_step(state, cfg, *, refl_10cm_due=False, **_kwargs):
    """The production stash contract, without a device.

    Identical to the fake step in tests/test_model.py's restart-split
    gate: a step stashes exactly when the executor says the history alarm
    rings at its end, which is the only way a stash is ever produced.
    """
    if refl_10cm_due:
        endpoint = int(round(state.elapsed_seconds + cfg.dt))
        stash_refl_10cm(
            state, np.asarray([cfg.grid_id, endpoint], dtype=np.int64))


def _tree(*, delay_s: int):
    """A two-domain CPU tree whose child starts ``delay_s`` late (or not).

    Hand-assembled for the same reason tests/test_model.py's fixtures are:
    ``build_experiment`` needs real forcing and a device, and the defect
    lives entirely in the executor/history seam above both.
    """
    base = load_scaffold(variant="n2b")
    domains = tuple(
        replace(
            dc,
            history_interval_s=HISTORY_SECONDS,
            start_time=(base.start_time + timedelta(seconds=delay_s)
                        if dc.grid_id == 2 and delay_s else dc.start_time),
            run=replace(dc.run, run_seconds=RUN_SECONDS,
                        output_interval_s=HISTORY_SECONDS))
        for dc in base.domains)
    exp = replace(base, run_seconds=RUN_SECONDS, domains=domains)
    tick_clock = resolve_clock(exp, lbc_interval_s=60)
    clocks = tick_clock.clocks()
    grids = grids_from_projection_config(exp)
    root = DomainNode(domains[0], grids[0], _HistoryState(), clocks[1],
                      None, [], None)
    child = DomainNode(domains[1], grids[1], _HistoryState(), clocks[2],
                       root, [], None)
    child.coupler = _Coupler(child)
    child._started = clocks[2].spec.start_ticks == 0
    root.children.append(child)
    model = ExperimentState(
        root, {1: root, 2: child}, build_schedule(exp, tick_clock),
        None, "delayed-activation-fixture")
    model._runtime_status = ModelRuntimeStatus()
    model._resumed = False
    model._resume_committed_history_grid_ids = frozenset()
    model._scratch_arena = None
    model._dycore_state_workspace = None
    model._io_manager = None
    model._last_checkpoint = None
    model._input_catalog = None
    model._prepared_by_grid_id = {}
    model._activation_context = {
        "experiment": exp,
        "case_data": SimpleNamespace(source_orography=None,
                                     sfcp_to_sfcp=None),
        "forcing_times": (),
        "radiation_workspace": None,
    }
    return exp, model


def _run(model, monkeypatch):
    """Drive the tree through the PRODUCTION history handoff."""
    from gpuwm.runtime import _submit_tree_history_frame

    writers = _Writers()
    model._io_manager = writers
    monkeypatch.setattr("gpuwm.core.dycore.step", _stashing_step)
    execute_experiment(
        model, validate_state=False,
        history_handler=lambda _tree, node, ticks: _submit_tree_history_frame(
            writers, node, ticks))
    return writers


def _activate_in_place(monkeypatch):
    """Activate the delayed child without ingest, catalog or device.

    ``on_domain_start`` re-initializes the child from the analysis at its
    activation time; every one of those collaborators needs real forcing
    and a GPU.  Replacing them keeps the executor, the clocks, the
    schedule and the history handoff -- where the defect lives -- exactly
    as production runs them.
    """
    _HistoryPhysics.radiation_callable = None
    monkeypatch.setattr(
        "gpuwm.ingest.nest_init.initialize_child",
        lambda cfg, parent, *args, **kwargs: SimpleNamespace(
            grid=parent.grid, state=_HistoryState()))
    monkeypatch.setattr("gpuwm.runtime.prepare_child_case",
                        lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr("gpuwm.core.nest.NestCoupler",
                        lambda node, feedback=0: _Coupler(node))


# --- the predicate every consume site now shares -------------------------

def test_the_stash_is_never_due_at_the_domains_own_start_tick():
    from gpuwm.core.refl import refl_10cm_stash_is_due

    # A domain that starts with the experiment: its analysis frame is
    # tick 0, every later frame follows one of its steps.
    assert refl_10cm_stash_is_due(0) is False
    assert refl_10cm_stash_is_due(60) is True
    # An activating nest: its analysis frame is the activation epoch,
    # and reading the absolute 0 there is exactly defect #205.
    assert refl_10cm_stash_is_due(120, domain_start_ticks=120) is False
    assert refl_10cm_stash_is_due(180, domain_start_ticks=120) is True


# --- the production tree seam -------------------------------------------

def test_a_delayed_child_survives_its_activation_epoch_frame(monkeypatch):
    """The reproduction, and the fix: the child's first frame is due AT
    its activation epoch, before any of its steps, so it carries no
    REFL_10CM and the run completes instead of raising."""
    _activate_in_place(monkeypatch)
    _exp, model = _tree(delay_s=DELAY_SECONDS)

    writers = _run(model, monkeypatch)

    child_frames = [(ticks, refl) for gid, ticks, refl in writers.frames
                    if gid == 2]
    assert [ticks for ticks, _refl in child_frames] == [120, 180, 240, 300]
    # The activation-epoch frame: no stash, the same shape the root's own
    # analysis frame has always had.
    assert child_frames[0][1] is None
    # Every frame after it follows a step of THAT domain, so it carries
    # the field that step stashed.
    assert all(refl is not None for _ticks, refl in child_frames[1:])


def test_the_root_keeps_its_own_frames_while_the_child_waits(monkeypatch):
    """No gate widening on the domain that starts with the experiment:
    the root's tick-0 frame still carries no stash and every later root
    frame still does."""
    _activate_in_place(monkeypatch)
    _exp, model = _tree(delay_s=DELAY_SECONDS)

    writers = _run(model, monkeypatch)

    root_frames = [(ticks, refl) for gid, ticks, refl in writers.frames
                   if gid == 1]
    assert [ticks for ticks, _refl in root_frames] == [0, 60, 120, 180,
                                                       240, 300]
    assert root_frames[0][1] is None
    assert all(refl is not None for _ticks, refl in root_frames[1:])


def test_all_domains_at_the_experiment_start_are_untouched(monkeypatch):
    """The non-delayed path is the one that must not move: both domains
    publish a stashless tick-0 frame and a stashed frame at every later
    boundary, exactly as before this fix."""
    _exp, model = _tree(delay_s=0)

    writers = _run(model, monkeypatch)

    for grid_id in (1, 2):
        frames = [(ticks, refl) for gid, ticks, refl in writers.frames
                  if gid == grid_id]
        assert [ticks for ticks, _refl in frames] == [0, 60, 120, 180,
                                                      240, 300]
        assert frames[0][1] is None
        assert all(refl is not None for _ticks, refl in frames[1:])


# --- the other consume sites --------------------------------------------

def test_prepared_forecast_due_helper_follows_the_domains_own_start():
    from gpuwm import prepared_single_domain_forecast as runner

    sentinel = object()
    state = SimpleNamespace(
        qv=np.ones((1,), dtype=np.float32),
        physics=SimpleNamespace(mp_physics=10))

    def consumer(_state):
        return sentinel

    assert runner._consume_due_native_refl_10cm(state, 0, consumer) is None
    assert runner._consume_due_native_refl_10cm(
        state, 1, consumer) is sentinel
    assert runner._consume_due_native_refl_10cm(
        state, 120, consumer, domain_start_ticks=120) is None
    assert runner._consume_due_native_refl_10cm(
        state, 180, consumer, domain_start_ticks=120) is sentinel


def test_ideal_nest_history_consume_follows_the_domains_own_start(
        monkeypatch):
    from gpuwm.verify.cases import nest_ideal_common

    consumed = []
    monkeypatch.setattr("gpuwm.core.refl.consume_refl_10cm",
                        lambda state: consumed.append(state))
    state = SimpleNamespace(physics=object())
    node = SimpleNamespace(
        cfg=SimpleNamespace(run=SimpleNamespace(mp_physics=10), grid_id=2),
        state=state,
        clock=SimpleNamespace(spec=SimpleNamespace(start_ticks=120)))

    nest_ideal_common.consume_history_reflectivity(node, 120)
    assert consumed == []
    nest_ideal_common.consume_history_reflectivity(node, 180)
    assert consumed == [state]


# --- the upfront refusal, retired ---------------------------------------

def test_a_delayed_child_config_loads_instead_of_refusing(tmp_path):
    """``build_experiment``'s categorical refusal is gone: the config the
    2.5.0 release refused by name now loads and resolves the child's own
    start."""
    from datetime import datetime

    from gpuwm.experiment import load_experiment
    from test_experiment import _write

    exp = load_experiment(
        _write(tmp_path, d02="start_time = 1974-04-03T12:30:00"))

    assert exp.domain_start_time(2) == datetime(1974, 4, 3, 12, 30)
    assert exp.domain_start_offset_exact(2) == 1800


def test_the_shipped_run_door_accepts_a_delayed_child(tmp_path, capsys):
    """The exit code, not just the exception.  This config exited 2 with
    the categorical refusal in 2.5.0; it now passes the load gate and is
    refused only by what it genuinely lacks here -- declared inputs."""
    import gpuwm.cli as cli
    from test_experiment import _write

    path = _write(tmp_path, d02="start_time = 1974-04-03T12:30:00")
    assert cli.main(["run", str(path)]) == 2
    message = capsys.readouterr().err
    assert "delayed nest activation" not in message
    assert "declared inputs" in message


# --- the shapes that remain unsupported, refused by route ---------------

def test_a_route_without_activation_machinery_refuses_by_name():
    from gpuwm.experiment import (delayed_domain_ids,
                                  refuse_delayed_activation)

    exp = load_scaffold(variant="n2b")
    assert delayed_domain_ids(exp) == ()
    refuse_delayed_activation(exp, "prepared domain-tree")  # no-op

    delayed = replace(exp, domains=tuple(
        replace(dc, start_time=(exp.start_time + timedelta(seconds=120)
                                if dc.grid_id == 2 else dc.start_time))
        for dc in exp.domains))
    assert delayed_domain_ids(delayed) == (2,)
    with pytest.raises(ValueError) as caught:
        refuse_delayed_activation(delayed, "prepared domain-tree")
    message = str(caught.value)
    assert "prepared domain-tree route does not implement delayed nest " \
           "activation" in message
    assert "`gpuwm run`" in message
    assert "tick-exact sync violated" in message


def test_the_prepared_tree_runner_calls_that_refusal():
    """Reachability, not prose: the runner's preflight names the refusal
    beside the spawn one, so a delayed child cannot reach the executor
    through a route that has no activation callback."""
    import inspect

    from gpuwm import prepared_domain_tree_forecast as runner

    source = inspect.getsource(runner.preflight_prepared_tree)
    assert 'refuse_delayed_activation(exp, "prepared domain-tree")' in source


# --- the structural refusals, untouched ---------------------------------

def test_a_misaligned_delayed_start_keeps_its_structural_refusal(tmp_path):
    """Retiring the categorical refusal does not widen the structural
    ones: a delayed start off the parent step boundary still refuses with
    the precise message it always had."""
    from gpuwm.experiment import load_experiment
    from test_experiment import _write

    with pytest.raises(ValueError, match="parent step boundary"):
        load_experiment(_write(tmp_path, d02="start_time = "
                                             "1974-04-03T12:00:50"))
