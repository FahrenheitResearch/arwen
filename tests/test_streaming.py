"""The streaming mode's contract, on the CPU.

The expensive half of this feature -- that a streamed domain reproduces a
resident one bit for bit -- is proven on a GPU by ``tilestream/test_gate.py``
and ``tilestream/test_join.py``.  What is proven here is everything that must
hold BEFORE a card is involved, and above all the OFF contract: a tree that
configures no ``[tiles]`` must execute the code it executed before this
feature existed, through the same call, with the same identity, and must
fingerprint the same.

The OFF contract is stated as an object identity rather than as a behaviour,
because that is the only form of it that cannot rot.  ``make_stepper`` with
streaming off returns ``gpuwm.core.dycore.step`` ITSELF; a wrapper that
merely forwarded would satisfy every behavioural test and would still be a
second code path to keep correct.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime
import json

import pytest

from gpuwm.core import streaming
from gpuwm.core.streaming import (OFF, StreamingOptions, StreamedDomain,
                                  StreamingRefused, decide, is_streaming,
                                  make_stepper)


# --------------------------------------------------------------------------
# the OFF contract
# --------------------------------------------------------------------------

def test_off_returns_the_dycore_step_itself():
    """Not a wrapper.  The same function object, so there is no branch."""
    from gpuwm.core.dycore import step

    assert make_stepper(object(), _cfg(), OFF) is step
    assert make_stepper(object(), _cfg(), None) is step
    assert not is_streaming(step)


def test_auto_that_fits_resident_also_returns_the_dycore_step(monkeypatch):
    """``auto`` on a domain that fits is OFF, not "on with one tile"."""
    from gpuwm.core.dycore import step

    monkeypatch.setattr(streaming, "decide",
                        lambda *_a, **_k: streaming.StreamingDecision(
                            False, "fits"))
    assert make_stepper(object(), _cfg(), StreamingOptions(mode="auto")) \
        is step


def test_absent_streaming_block_is_the_shared_off_object():
    assert StreamingOptions.from_mapping(None) is OFF
    assert OFF.enabled is False
    assert StreamingOptions(mode="on").enabled is True


def test_streaming_contributes_nothing_to_the_restart_identity():
    """A checkpoint written resident must resume streamed, and vice versa.

    Held as a payload comparison rather than as a promise: the identity of an
    experiment that streams and the identity of the same experiment that does
    not must be the SAME STRING, so a fingerprint computed before this
    feature existed still matches.
    """
    from gpuwm.core.model import restart_identity_payload

    exp = _experiment()
    resident = restart_identity_payload(exp)
    streamed = restart_identity_payload(dataclasses.replace(
        exp, tiles=StreamingOptions(mode="on", tile_nx=512, tile_ny=512)))
    assert json.dumps(resident, sort_keys=True) == \
        json.dumps(streamed, sort_keys=True)
    assert "tiles" not in resident
    assert streaming.identity_payload_entry(
        StreamingOptions(mode="on", tile_nx=8, tile_ny=8)) == {}


def test_the_executor_binds_the_dycore_step_when_nothing_is_configured(
        monkeypatch):
    """The seam is the call site: an unconfigured tree still calls ``step``.

    The existing executor tests monkeypatch ``gpuwm.core.dycore.step`` and
    count calls; this asserts that the indirection introduced for streaming
    did not quietly stop that from being the function that runs.
    """
    from test_model import _model                      # noqa: PLC0415
    from gpuwm.core.model import execute_experiment

    _exp, model = _model()
    calls = []
    monkeypatch.setattr(
        "gpuwm.core.dycore.step",
        lambda state, cfg, **kw: calls.append(int(cfg.grid_id)))
    report = execute_experiment(model, validate_state=False,
                                pool_trim_per_period=False)
    assert len(calls) == report.steps == 20


def test_a_bound_stepper_replaces_the_dycore_step_for_that_grid_only(
        monkeypatch):
    """One domain streams; its sibling does not.  The seam is per grid."""
    from test_model import _model                      # noqa: PLC0415
    from gpuwm.core.model import execute_experiment

    _exp, model = _model()
    dycore_calls, streamed_calls = [], []
    monkeypatch.setattr(
        "gpuwm.core.dycore.step",
        lambda state, cfg, **kw: dycore_calls.append(int(cfg.grid_id)))

    def fake_streamed(state, cfg, **kw):
        streamed_calls.append((int(cfg.grid_id), bool(kw["refl_10cm_due"])))

    execute_experiment(model, validate_state=False,
                       pool_trim_per_period=False,
                       history_handler=lambda *_a: None,
                       steppers={1: fake_streamed})
    assert set(dycore_calls) == {2}
    assert {gid for gid, _ in streamed_calls} == {1}
    assert len(dycore_calls) == len(streamed_calls) == 10
    # The REFL_10CM handshake has to reach the streamed domain too: it
    # decides whether the step stages a field the frame about to be written
    # will consume, and a streamed sweep must stage it on every tile.
    assert any(due for _gid, due in streamed_calls)


def test_the_single_domain_loop_defaults_to_the_dycore_step():
    """``integrate_prepared_case`` binds ``step`` when handed no stepper."""
    import inspect

    from gpuwm import runtime

    src = inspect.getsource(runtime.integrate_prepared_case)
    assert "stepper = step if stepper is None else stepper" in src
    assert "stepper(state, integration_cfg, **step_kwargs)" in src
    assert "        step(state, integration_cfg" not in src


# --------------------------------------------------------------------------
# the configuration surface
# --------------------------------------------------------------------------

def test_unknown_keys_are_refused_not_ignored():
    with pytest.raises(ValueError, match="unknown key"):
        StreamingOptions.from_mapping({"mode": "on", "tiles": 4})


def test_an_off_surface_must_be_empty():
    """The [relocation] discipline: a mode that is off carries no settings."""
    with pytest.raises(ValueError, match="carries a tiling while mode"):
        StreamingOptions(tile_nx=64, tile_ny=64)
    with pytest.raises(ValueError, match="together"):
        StreamingOptions(mode="on", tile_nx=64)


@pytest.mark.parametrize("table,message", [
    ({"mode": "sometimes"}, "not one of"),
    ({"mode": "on", "store": "disk"}, "must be"),
    ({"mode": "on", "write_mode": "inplace"}, "read-at-time-t"),
    ({"mode": "on", "nbuffers": 0}, "must be positive"),
])
def test_refusals(table, message):
    with pytest.raises(ValueError, match=message):
        StreamingOptions.from_mapping(table)


def test_experiment_toml_carries_the_block(tmp_path):
    from gpuwm.experiment import build_experiment

    raw = _raw_experiment()
    exp = build_experiment(raw, source="<test>")
    assert exp.tiles is OFF

    raw["tiles"] = {"mode": "auto"}
    exp = build_experiment(raw, source="<test>")
    assert exp.tiles.mode == "auto"

    raw["tiles"] = {"mode": "auto", "tile_nixe": 64}
    with pytest.raises(ValueError, match="unknown key"):
        build_experiment(raw, source="<test>")


def test_a_stray_table_is_still_refused():
    """Adding [tiles] to the schema must not open the schema."""
    from gpuwm.experiment import build_experiment

    raw = _raw_experiment()
    raw["streamimg"] = {"mode": "auto"}
    with pytest.raises(ValueError, match="does not have a table"):
        build_experiment(raw, source="<test>")


# --------------------------------------------------------------------------
# the decision
# --------------------------------------------------------------------------

def test_off_decides_without_importing_the_planner(monkeypatch):
    """``mode = "off"`` must not need a card, a planner or cupy."""
    import sys

    monkeypatch.setitem(sys.modules, "tilestream.autoplan", None)
    got = decide(_cfg(), OFF)
    assert got.stream is False
    assert "off" in got.explain()


def test_a_pinned_tiling_consults_no_planner(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "tilestream.autoplan", None)
    got = decide(_cfg(), StreamingOptions(mode="on", tile_nx=512,
                                          tile_ny=256, nbuffers=3))
    assert (got.stream, got.tile_nx, got.tile_ny, got.nbuffers) == \
        (True, 512, 256, 3)
    # The halo is taken from harness.halo_radius and NEVER configured: it is
    # 10 + 3*time_step_sound//2 and a smaller one is silent and faster.
    assert got.halo == 10 + 3 * _cfg().time_step_sound // 2


def test_a_short_halo_warns_loudly():
    with pytest.warns(RuntimeWarning, match="how this defect hides"):
        decide(_cfg(), StreamingOptions(mode="on", tile_nx=64, tile_ny=64,
                                        halo=4))


# --------------------------------------------------------------------------
# what the run RECORDS about which way it decided
#
# The refusal above is the loud failure.  These cover the QUIET one, which
# is the same defect pointing the other way: `auto` on a domain that fits
# returns an empty stepper dict, and an empty stepper dict is also what
# `off` returns, and what a config with no [tiles] table returns.  So a
# run that was asked to stream and declined looked exactly like a run that
# was never asked -- to an operator reading a log, and to a test.
# --------------------------------------------------------------------------

def _fitting_machine():
    """A machine with room to spare, so ``auto`` decides RESIDENT.

    Explicit rather than ``Machine.detect()``: detect reads free VRAM off a
    real card, so on CI or a busy box this test would measure the neighbour
    instead of the logic -- and would flip verdict depending on who else was
    running.
    """
    from tilestream.autoplan import Machine

    return Machine(vram_bytes=80 * 2 ** 30, host_bytes=512 * 2 ** 30,
                   name="test", host_source="explicit")


def _one_grid_model(cfg=None):
    import types

    cfg = _cfg() if cfg is None else cfg
    node = types.SimpleNamespace(
        cfg=types.SimpleNamespace(grid_id=1, run=cfg), state=object())
    return types.SimpleNamespace(walk_parent_first=lambda: [node])


def test_auto_that_declines_is_recorded_rather_than_merely_absent():
    """The stepper dict says nothing; the decisions mapping says everything."""
    model = _one_grid_model()
    decisions: dict = {}
    steppers = streaming.steppers_for_tree(
        model, StreamingOptions(mode="auto"), machine=_fitting_machine(),
        decisions=decisions)
    assert steppers == {}                      # nothing streamed ...
    assert set(decisions) == {1}               # ... but it was DECIDED
    assert decisions[1].stream is False
    assert "fits resident" in decisions[1].reason


def test_the_receipt_tells_a_declined_auto_apart_from_an_unconfigured_run():
    """THE CONTROL, and it is the whole point of the receipt.

    Both runs execute resident and both hand the executor ``{}``.  If the
    receipt cannot separate them, nothing downstream can, and an ``auto``
    run that never streamed is indistinguishable from a streaming success.
    """
    model, machine = _one_grid_model(), _fitting_machine()

    off_decisions: dict = {}
    streaming.steppers_for_tree(model, OFF, machine=machine,
                                decisions=off_decisions)
    off = streaming.streaming_receipt(OFF, off_decisions)

    auto_decisions: dict = {}
    streaming.steppers_for_tree(model, StreamingOptions(mode="auto"),
                                machine=machine, decisions=auto_decisions)
    auto = streaming.streaming_receipt(StreamingOptions(mode="auto"),
                                       auto_decisions)

    assert off == {}                           # unconfigured contributes nothing
    assert auto != {}                          # configured always says something
    assert auto["streamed_any"] is False
    assert auto["configured_mode"] == "auto"
    assert auto["domains"]["1"]["streamed"] is False
    assert "NO domain streamed" in auto["summary"]
    # The two receipts must not merely differ -- the difference has to be
    # the ANSWER.  A reader must not have to know that an absent key means
    # "never asked" while a present one means "asked and declined".
    assert off.get("streamed_any") is None and auto["streamed_any"] is False


def test_the_control_fires_when_the_recording_is_removed():
    """Break it deliberately: without ``decisions``, the two runs are equal.

    This is the state of the code BEFORE this fix, reproduced exactly by
    declining to pass the out-parameter.  A control that cannot fail is
    worth nothing, so this asserts that the failure is reachable.
    """
    model, machine = _one_grid_model(), _fitting_machine()
    off = streaming.steppers_for_tree(model, OFF, machine=machine)
    auto = streaming.steppers_for_tree(model, StreamingOptions(mode="auto"),
                                       machine=machine)
    assert off == auto == {}
    # ... and with nothing recorded the receipt collapses to the same value
    # for both, which is the ambiguity the fix removes.
    assert streaming.streaming_receipt(OFF, {}) == \
        streaming.streaming_receipt(StreamingOptions(mode="auto"), {}) == {}


def test_a_receipt_for_a_domain_that_did_stream_says_so_and_names_the_tiling():
    model = _one_grid_model()
    options = StreamingOptions(mode="on", tile_nx=64, tile_ny=64)
    decisions: dict = {}
    with pytest.raises(StreamingRefused):
        # No builder here; the decision is still recorded BEFORE the refusal,
        # which is what lets a failed run say what it was trying to do.
        streaming.steppers_for_tree(model, options, decisions=decisions)
    receipt = streaming.streaming_receipt(options, decisions)
    assert receipt["streamed_any"] is True
    assert receipt["domains"]["1"] == {
        "streamed": True, "reason": "[tiles] pins the tiling",
        "tile_nx": 64, "tile_ny": 64, "nbuffers": 2,
        "halo": 10 + 3 * _cfg().time_step_sound // 2, "store": "host"}


@pytest.mark.parametrize("module", [
    "gpuwm.prepared_single_domain_forecast",
    "gpuwm.prepared_domain_tree_forecast"])
def test_both_production_routes_wire_the_builder_AND_record_the_decision(
        module):
    """The regression test for the defect itself, at the ROUTE.

    ``tests`` above prove the seam behaves; this proves the two functions a
    user can actually launch a forecast through USE it.  Both routes once
    called ``steppers_for_tree(model, exp.tiles)`` bare, so every
    ``mode = "on"`` configuration raised ``StreamingRefused`` and no
    forecast the CLI could launch was capable of streaming.  Asserted on the
    source because the alternative is a full prepared-cache forecast, and a
    regression test that needs a GPU and a data bundle is one nobody runs.
    """
    import importlib
    import inspect

    src = inspect.getsource(importlib.import_module(module))
    assert "builders=streaming.builders_for_tree(" in src, \
        "route calls steppers_for_tree with no builder -- it will REFUSE"
    assert "decisions=streaming_decisions" in src, \
        "route does not record which way the decision went"
    assert "streaming.streaming_receipt(" in src
    assert 'report["tiles"] = streaming_report' in src


def test_streaming_without_a_builder_refuses_rather_than_running_resident():
    """The dangerous direction is silently NOT streaming.

    A route that configured streaming and got a resident run would die at
    the allocation the mode existed to avoid, having reported nothing.
    """
    with pytest.raises(StreamingRefused, match="wired no streamed-domain"):
        make_stepper(object(), _cfg(),
                     StreamingOptions(mode="on", tile_nx=64, tile_ny=64))


def test_a_streamed_domain_refuses_a_state_it_was_not_attached_to():
    state, other = object(), object()
    run = _FakeRun(_cfg())
    streamed = StreamedDomain(run, streaming.StreamingDecision(True, "test"),
                              state=state)
    streamed(state, run.cfg, refl_10cm_due=False)
    assert run.sweeps == [(1, {"refl_10cm_due": False})]
    with pytest.raises(StreamingRefused, match="not the one it was attached"):
        streamed(other, run.cfg)


def test_the_step_kwargs_reach_the_sweep():
    """``refl_10cm_due`` is not optional decoration; it reaches every tile."""
    run = _FakeRun(_cfg())
    run.store[streaming.REFL_STORE_KEY] = object()
    streamed = StreamedDomain(run, streaming.StreamingDecision(True, "test"))
    streamed(None, run.cfg, refl_10cm_due=False, health_debug=True)
    assert run.sweeps == [(1, {"refl_10cm_due": False,
                               "health_debug": True})]
    assert streamed.steps == 1


def test_the_kwargs_reach_the_TILE_STEP_and_not_merely_the_sweep():
    """The gap this test used to stop one hop short of.

    ``TiledRun._sweep`` copied the caller's ``step_kwargs`` into a local
    dict and then called ``step(tile, tile_cfg)`` with none of them, so a
    keyword ArWen's loop threaded into a STEP was accepted by ``sweep``,
    documented as forwarded, and dropped at the last hop.  A fake run
    cannot see that -- it never reaches a tile -- so this reads the
    transport's own source.
    """
    import inspect

    from tilestream import driver

    src = inspect.getsource(driver.TiledRun)
    assert "step(tiles[b], tile_cfg, **step_kwargs)" in src, (
        "the tile step no longer forwards step_kwargs; every keyword the "
        "model threads into a STEP is silently dropped under tiling")


def test_a_due_frame_refuses_a_store_with_nowhere_to_put_the_reflectivity():
    """The tiles would each compute their own window and lose all of them.

    ``refl_10cm`` is a REBUILT scratch slot rather than a carrier, so a
    streamed run that omits it integrates the identical forecast -- which is
    exactly why the omission has to be caught at the frame and not at the
    trajectory.  Publishing anyway would put the LAST tile's window over the
    whole domain, right over one tile of N and plausible everywhere.
    """
    run = _FakeRun(_cfg())
    streamed = StreamedDomain(run, streaming.StreamingDecision(True, "test"))
    with pytest.raises(StreamingRefused, match="scratch/refl_10cm"):
        streamed(None, run.cfg, refl_10cm_due=True)
    assert run.sweeps == []


# --------------------------------------------------------------------------
# per-tile lateral boundary windowing
# --------------------------------------------------------------------------

def test_owned_edges_follows_the_window_not_the_interior():
    """Under periodic=False plan_tiles CLAMPS, so the WINDOW decides.

    Both facts matter and they are different tiles' business: the interior
    decides what is scattered back, the window decides which cells the
    boundary kernel writes.
    """
    from tilestream import spec as tspec

    specs = tspec.plan_tiles(128, 96, 32, 32, 16, False)
    census = {0: 0, 1: 0, 2: 0}
    for s in specs:
        owned = streaming.owned_edges(s)
        census[sum(owned.values())] += 1
        assert owned["west"] == (s.ci0 == 0)
        assert owned["east"] == (s.ci0 + s.cnx == s.nx)
    # A 4x3 tiling has four corner tiles (two true edges), six edge tiles
    # (one) and two in the middle with NO true edge at all -- so the plan
    # contains tiles of every kind, which is what makes the seam question
    # meaningful rather than hypothetical.
    assert census == {0: 2, 1: 6, 2: 4}


def test_windowed_side_tables_carry_the_fields_own_staggering():
    """u's south/north tables are cnx+1 wide; v's west/east are cny+1 tall."""
    from tilestream import spec as tspec

    bnd = _boundaries(nz=3, ny=32, nx=48, width=5)
    specs = tspec.plan_tiles(48, 32, 16, 16, 4, False)
    tables = streaming.tile_boundary_tables(bnd, specs)
    for spec_, tbl in zip(specs, tables):
        iv = tbl.intervals[0]
        assert iv.fields["u"].south.value.shape == (3, 5, spec_.cnx + 1)
        assert iv.fields["u"].west.value.shape == (3, spec_.cny, 5)
        assert iv.fields["v"].west.value.shape == (3, spec_.cny + 1, 5)
        assert iv.fields["v"].south.value.shape == (3, 5, spec_.cnx)
        assert iv.fields["mu"].north.value.shape == (3, 5, spec_.cnx)


def test_true_edges_keep_the_domains_values_and_seams_are_inert():
    """The whole mode rests on this split, so it is asserted element-wise."""
    import numpy as np
    from tilestream import spec as tspec

    bnd = _boundaries(nz=2, ny=32, nx=48, width=5)
    specs = tspec.plan_tiles(48, 32, 16, 16, 4, False)
    tables = streaming.tile_boundary_tables(bnd, specs, seam="zeros")
    saw_edge = saw_seam = False
    for spec_, tbl in zip(specs, tables):
        owned = streaming.owned_edges(spec_)
        side = tbl.intervals[0].fields["mu"].west
        if owned["west"]:
            want = np.asarray(
                bnd.intervals[0].fields["mu"].west.value)[
                    :, spec_.cj0:spec_.cj0 + spec_.cny, :]
            assert np.array_equal(side.value, want)
            assert np.any(side.value)
            saw_edge = True
        else:
            assert not np.any(side.value)
            assert not np.any(side.tendency)
            saw_seam = True
    assert saw_edge and saw_seam


def test_the_three_seam_fillings_are_genuinely_different_numbers():
    """Zeros, self-consistent and garbage must not accidentally coincide.

    The claim proven on the GPU is that three COMPLETELY DIFFERENT interior
    seam fillings give the identical answer.  That claim is worthless if the
    three fillings are the same numbers, so the difference is asserted here
    where it is cheap.
    """
    import numpy as np
    from tilestream import spec as tspec

    bnd = _boundaries(nz=2, ny=96, nx=128, width=5)
    specs = tspec.plan_tiles(128, 96, 32, 32, 4, False)
    snapshot = {name: np.full((2, 96 + extra_y, 128 + extra_x), 3.5)
                for name, (extra_y, extra_x) in
                (("mu", (0, 0)), ("u", (0, 1)), ("v", (1, 0)))}
    interior = next(i for i, s in enumerate(specs)
                    if not any(streaming.owned_edges(s).values()))
    got = {}
    for seam in ("zeros", "self", "poison"):
        tables = streaming.tile_boundary_tables(bnd, specs, seam=seam,
                                                snapshot=snapshot)
        got[seam] = np.asarray(
            tables[interior].intervals[0].fields["mu"].west.value)
    assert not np.any(got["zeros"])
    assert np.allclose(got["self"], 3.5)
    assert np.abs(got["poison"]).max() > 1.0e5
    for a, b in (("zeros", "self"), ("self", "poison"),
                 ("zeros", "poison")):
        assert not np.array_equal(got[a], got[b])


def test_seam_self_without_a_snapshot_refuses():
    from tilestream import spec as tspec

    bnd = _boundaries(nz=2, ny=32, nx=48, width=5)
    specs = tspec.plan_tiles(48, 32, 16, 16, 4, False)
    with pytest.raises(ValueError, match="needs the domain snapshot"):
        streaming.tile_boundary_tables(bnd, specs, seam="self")


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

class _FakeRun:
    """Enough of :class:`tilestream.driver.TiledRun` to test the seam."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.store = {}
        self.sweeps = []

    def sweep(self, nsteps=1, *, step_kwargs=None, report=None,
              progress=None):
        self.sweeps.append((nsteps, dict(step_kwargs or {})))
        if report is not None:
            report.update(steps=nsteps)


def _cfg():
    from gpuwm.config import RunConfig

    return RunConfig(nx=256, ny=192, nz=49, dx=12000.0, dy=12000.0,
                     ztop=20000.0, dt=60.0, run_seconds=600.0)


def _boundaries(*, nz, ny, nx, width):
    """A domain ``LateralBoundaries`` with distinguishable, nonzero tables."""
    import numpy as np
    from gpuwm.ingest.lateral_bc import (BoundaryInterval, FieldBoundary,
                                         LateralBoundaries, SideBoundary)

    def side(shape, base):
        value = base + np.arange(int(np.prod(shape)),
                                 dtype=np.float64).reshape(shape)
        return SideBoundary(value, 0.001 * value)

    fields = {}
    for name, (ey, ex) in (("mu", (0, 0)), ("u", (0, 1)), ("v", (1, 0))):
        fields[name] = FieldBoundary(
            west=side((nz, ny + ey, width), 100.0),
            east=side((nz, ny + ey, width), 200.0),
            south=side((nz, width, nx + ex), 300.0),
            north=side((nz, width, nx + ex), 400.0))
    return LateralBoundaries((BoundaryInterval(0.0, 21600.0, fields),),
                             width, 1, 4)


def _raw_experiment() -> dict:
    """A minimal but VALID experiment table, modelled on configs/.

    Deliberately built here rather than loaded from ``configs/``: the shipped
    configs name real forcing files, and a schema test that cannot run
    without a 2 GB GRIB is a schema test nobody runs.
    """
    return {
        "experiment": {"name": "streaming-test",
                       "start_time": datetime(2026, 8, 9, 0, 0, 0),
                       "run_seconds": 600.0, "feedback": 0,
                       "smooth_option": 0, "blend_width": 5,
                       "spec_bdy_width": 5, "restart_interval_s": 0.0},
        "projection": {"map_proj": "lambert", "ref_lat": 35.0,
                       "ref_lon": -97.0, "truelat1": 30.0,
                       "truelat2": 60.0, "stand_lon": -97.0},
        "shared": {
            "nz": 4, "ztop": 20000.0, "p_top": 10000.0,
            "eta_levels": [1.0, 0.8, 0.55, 0.28, 0.0],
            "hybrid_opt": 2, "etac": 0.2, "base_temp": 290.0,
            "time_step_sound": 4, "terrain_opt": 1, "map_proj": 1,
        },
        "domain": [{
            "grid_id": 1, "parent_id": 0, "i_parent_start": 1,
            "j_parent_start": 1, "parent_grid_ratio": 1,
            "parent_time_step_ratio": 1, "nx": 64, "ny": 64,
            "time_step": 60, "dx": 12000.0, "specified": True,
            "nested": False, "history_interval_s": 300.0,
        }],
    }


def _experiment():
    from gpuwm.experiment import build_experiment

    return build_experiment(_raw_experiment(), source="<test>")


# --------------------------------------------------------------------------
# the ROUTE builders -- the half of the seam that did not exist
# --------------------------------------------------------------------------
#
# Until these, both production routes called steppers_for_tree with no
# builders and every [tiles] mode = "on" forecast raised
# StreamingRefused.  The expensive half of the proof -- that the shipped
# builder is bit-exact through execute_experiment -- is on a GPU in
# tilestream/test_route.py.  What is proven here is that the routes REACH
# it, which is the part that rotted silently.

def test_both_production_routes_pass_builders():
    """The regression that produced this whole item, held as source.

    A source assertion rather than a behavioural one because the defect was
    an ABSENT argument: nothing about the routes' behaviour differed, they
    simply refused every streaming configuration they were given, and a
    route that stops passing builders would go back to refusing without any
    other test noticing.
    """
    import inspect

    import gpuwm.prepared_domain_tree_forecast as tree
    import gpuwm.prepared_single_domain_forecast as single

    for module in (single, tree):
        src = inspect.getsource(module)
        calls = [line for line in src.splitlines()
                 if "steppers_for_tree(" in line and "def " not in line]
        assert calls, f"{module.__name__} no longer calls steppers_for_tree"
        assert "builders=streaming.builders_for_tree(" in src, (
            f"{module.__name__} calls steppers_for_tree without wiring "
            "builders; every [tiles] mode='on' forecast on that route "
            "will raise StreamingRefused")


def test_builders_for_tree_is_empty_and_inert_when_streaming_is_off():
    """The OFF contract again: absent [tiles] must import nothing."""
    import sys

    from test_model import _model                      # noqa: PLC0415

    _exp, model = _model()
    saved = {name: sys.modules.get(name)
             for name in ("tilestream.driver", "tilestream.autoplan")}
    try:
        for name in saved:
            sys.modules[name] = None
        assert streaming.builders_for_tree(model, OFF) == {}
        assert streaming.builders_for_tree(model, None) == {}
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_builders_for_tree_covers_every_grid():
    from test_model import _model                      # noqa: PLC0415

    _exp, model = _model()
    builders = streaming.builders_for_tree(
        model, StreamingOptions(mode="on", tile_nx=32, tile_ny=32))
    assert set(builders) == {1, 2}
    assert all(callable(b) for b in builders.values())


def test_a_nest_that_fires_is_refused_rather_than_windowed_wrong():
    """A nest holds no LateralBoundaries, so there is nothing to window.

    Refused at BUILD time, not at builders_for_tree time: a tree with
    mode = "auto" whose nests fit resident must not be refused for having
    nests at all.
    """
    from test_model import _model                      # noqa: PLC0415

    _exp, model = _model()
    builders = streaming.builders_for_tree(
        model, StreamingOptions(mode="on", tile_nx=32, tile_ny=32))
    child = model.node(2)
    with pytest.raises(StreamingRefused, match="is a NEST"):
        builders[2](child.state, child.cfg.run,
                    streaming.StreamingDecision(True, "test", 32, 32, 2, 16))


def test_a_scheme_that_cannot_be_twinned_is_refused_not_shared():
    """Sharing the domain's adapter shares its CARRIERS with every buffer.

    ``cumulus/w0avg`` is a carrier, so one Kain-Fritsch instance across the
    buffers means one w0avg array that the transport gathers and scatters as
    if each buffer owned its own.  This is the defect the geography check
    caught during development, and it is held here where it is cheap.
    """
    import numpy as np

    class _Configured:
        def __init__(self, chunk=1024):
            self.chunk = chunk

    lat = lon = np.zeros((4, 4))
    # A constructor that asks for nothing and configures nothing may be
    # rebuilt; the result must be a DIFFERENT object.
    plain = _Configured()
    twin = streaming._tile_scheme(plain, lat, lon)
    assert twin is not plain and twin.chunk == plain.chunk
    # One that carries policy a default reconstruction would drop may not.
    with pytest.raises(StreamingRefused, match="carries chunk="):
        streaming._tile_scheme(_Configured(chunk=50_000), lat, lon)


def test_a_dataclass_scheme_is_twinned_at_the_tiles_extents():
    import dataclasses

    import numpy as np

    @dataclasses.dataclass
    class _Adapter:
        latitude_deg: object
        longitude_deg: object
        column_chunk: int = 4096

    domain = _Adapter(np.zeros((8, 8)), np.zeros((8, 8)), column_chunk=12_500)
    lat, lon = np.ones((4, 4)), np.ones((4, 4))
    twin = streaming._tile_scheme(domain, lat, lon)
    assert twin is not domain
    assert twin.latitude_deg.shape == (4, 4)
    # The POLICY survives; only the geography moved.
    assert twin.column_chunk == 12_500


# --------------------------------------------------------------------------
# the twin RECIPE -- legacy RRTMG, the default suite's radiation
# --------------------------------------------------------------------------
#
# RRTMGLegacyRadiation is a plain class whose constructor requires
# (start_time, latitude_deg, longitude_deg), so _tile_scheme refused it and
# every [tiles] run of the SHIPPED DEFAULT physics suite died in
# TiledRun.__init__ -- streaming was reachable only by selecting rte-rrtmgp.
# These hold the recipe that rebuilds it and the two audits that keep the
# recipe honest, without constructing one (its constructor wants CUDA).


def _recipe_key(cls):
    """The key _tile_scheme derives for a class."""
    return f"{cls.__module__}:{cls.__qualname__}"


def test_the_default_suites_legacy_rrtmg_has_a_twin_recipe():
    """The registration itself, against the REAL class.

    A source-shaped assertion for the same reason the route-builder test
    above is one: the defect was an ABSENT recipe, and nothing else in this
    suite would notice its removal.
    """
    from gpuwm.core.rrtmg_legacy import RRTMGLegacyRadiation

    key = _recipe_key(RRTMGLegacyRadiation)
    assert key in streaming._TWIN_RECIPES, (
        "legacy RRTMG has no per-buffer twin recipe; every [tiles] run of "
        "the default physics suite will raise StreamingRefused")


def test_the_recipe_reproduces_every_constructor_parameter():
    """The staleness audit, run against the live signature.

    This is the test that fails when someone adds a policy argument to the
    adapter: a recipe that does not carry it would build buffers whose
    physics differs from the domain's at that argument's DEFAULT, which no
    inventory or geography check downstream can see.
    """
    import inspect

    from gpuwm.core.rrtmg_legacy import RRTMGLegacyRadiation

    recipe = streaming._TWIN_RECIPES[_recipe_key(RRTMGLegacyRadiation)]
    declared = {p.name for p in
                inspect.signature(RRTMGLegacyRadiation).parameters.values()
                if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
    assert declared <= recipe.reproduces, (
        f"the recipe does not reproduce {sorted(declared - recipe.reproduces)}")


class _RequiresArgs:
    """A stand-in with legacy RRTMG's SHAPE: required args plus policy."""

    def __init__(self, start_time, latitude_deg, longitude_deg, *,
                 column_chunk=4096):
        self.start_time = start_time
        self.latitude_deg = latitude_deg
        self.longitude_deg = longitude_deg
        self.column_chunk = column_chunk
        self.update_count = 0


def _register(monkeypatch, cls, **kw):
    kw.setdefault("build", lambda scheme, cls_, lat, lon: cls_(
        scheme.start_time, lat, lon, column_chunk=scheme.column_chunk))
    kw.setdefault("reproduces", frozenset({
        "start_time", "latitude_deg", "longitude_deg", "column_chunk"}))
    kw.setdefault("volatile", frozenset({"update_count"}))
    table = dict(streaming._TWIN_RECIPES)
    table[_recipe_key(cls)] = streaming._TwinRecipe(**kw)
    monkeypatch.setattr(streaming, "_TWIN_RECIPES", table)


def test_a_recipe_twin_takes_the_tiles_geography_and_the_domains_policy(
        monkeypatch):
    import numpy as np

    _register(monkeypatch, _RequiresArgs)
    domain = _RequiresArgs("T0", np.zeros((8, 8)), np.zeros((8, 8)),
                           column_chunk=12_500)
    lat, lon = np.ones((4, 4)), np.ones((4, 4))
    twin = streaming._tile_scheme(domain, lat, lon)

    assert twin is not domain                       # a FRESH object, not shared
    assert twin.latitude_deg.shape == (4, 4)        # the TILE's geography
    assert twin.start_time == "T0"                  # the DOMAIN's start time
    assert twin.column_chunk == 12_500              # the DOMAIN's policy


def test_a_stale_recipe_is_refused_rather_than_defaulting_the_policy(
        monkeypatch):
    """A recipe that does not name every constructor parameter is stale."""
    import numpy as np

    _register(monkeypatch, _RequiresArgs,
              reproduces=frozenset({"start_time", "latitude_deg",
                                    "longitude_deg"}))
    domain = _RequiresArgs("T0", np.zeros((8, 8)), np.zeros((8, 8)))
    with pytest.raises(StreamingRefused, match="stale against the adapter"):
        streaming._tile_scheme(domain, np.ones((4, 4)), np.ones((4, 4)))


def test_a_recipe_that_drops_policy_is_refused(monkeypatch):
    """The scalar audit, applied to the recipe's own output."""
    import numpy as np

    _register(monkeypatch, _RequiresArgs,
              build=lambda scheme, cls_, lat, lon: cls_(
                  scheme.start_time, lat, lon))       # drops column_chunk
    domain = _RequiresArgs("T0", np.zeros((8, 8)), np.zeros((8, 8)),
                           column_chunk=12_500)
    with pytest.raises(StreamingRefused, match="drops policy"):
        streaming._tile_scheme(domain, np.ones((4, 4)), np.ones((4, 4)))


def test_a_running_counter_is_not_mistaken_for_dropped_policy(monkeypatch):
    """The domain's adapter has stepped; a fresh twin has not."""
    import numpy as np

    _register(monkeypatch, _RequiresArgs)
    domain = _RequiresArgs("T0", np.zeros((8, 8)), np.zeros((8, 8)))
    domain.update_count = 17
    twin = streaming._tile_scheme(domain, np.ones((4, 4)), np.ones((4, 4)))
    assert twin.update_count == 0


def test_an_adapter_with_no_recipe_is_still_refused():
    """The guard is EXTENDED by the recipe table, not weakened by it."""
    import numpy as np

    domain = _RequiresArgs("T0", np.zeros((8, 8)), np.zeros((8, 8)))
    with pytest.raises(StreamingRefused, match="_TWIN_RECIPES"):
        streaming._tile_scheme(domain, np.ones((4, 4)), np.ones((4, 4)))


# --------------------------------------------------------------------------
# the buffer's VERTICAL COORDINATE -- the rte-rrtmgp warm-up crash
# --------------------------------------------------------------------------
#
# A tile buffer was built on make_vertical_coord's DEFAULT stretch while
# _impose_domain_setup gave it the domain's eta table, so its 3-D base state
# (thb/pb/alb/phb, not imposed because they are gathered) described a
# different atmosphere from its pressure.  The warm-up step then handed
# rte-rrtmgp 120.3-407.3 K and the gas-table validator refused it.  The
# validator is right and is untouched; the buffer was wrong.


class _EtaState:
    def __init__(self, znw):
        self.znw = znw


class _EtaCfg:
    hybrid_opt, etac = 2, 0.2

    def __init__(self, nz):
        self.nz = nz


def _real_eta(nz=8):
    """A table that is NOT the default stretch, as a real case's is not."""
    import numpy as np
    x = np.linspace(0.0, 1.0, nz + 1)
    znw = (1.0 - x) ** 1.7            # monotone 1 -> 0, clustered low
    znw[0], znw[-1] = 1.0, 0.0
    return znw


def test_the_buffer_is_built_on_the_domains_eta_table_not_the_default():
    import numpy as np

    eta = _real_eta(8)
    coord = streaming.domain_vertical_coord(_EtaState(eta), _EtaCfg(8))
    assert np.allclose(np.asarray(coord.znw, dtype=np.float64), eta)

    from gpuwm.core.grid import make_vertical_coord
    default = make_vertical_coord(8, hybrid_opt=2, etac=0.2)
    # The whole point: the two genuinely differ, so building on the default
    # is a real error rather than a harmless rebuild.
    assert not np.allclose(np.asarray(default.znw, dtype=np.float64), eta)


def test_a_domain_without_an_eta_table_is_refused_not_defaulted():
    with pytest.raises(StreamingRefused, match="carries no znw"):
        streaming.domain_vertical_coord(_EtaState(None), _EtaCfg(8))


def test_an_unrebuildable_eta_table_is_refused_not_defaulted():
    """Falling back to the default stretch IS the defect, so never fall back."""
    import numpy as np

    bad = np.array([1.0, 0.5, 0.6, 0.0])       # not monotone
    with pytest.raises(StreamingRefused, match="cannot be rebuilt"):
        streaming.domain_vertical_coord(_EtaState(bad), _EtaCfg(3))


def test_the_prepared_factory_builds_buffers_on_the_domains_coordinate():
    """Held as source: constructing a buffer needs a card and a real domain."""
    import inspect

    src = inspect.getsource(streaming.prepared_tile_state_factory)
    assert "coord = domain_vertical_coord(state, cfg)" in src
    assert "coord=coord" in src, (
        "the factory computes the domain's coordinate and then does not "
        "hand it to make_physics_state, which rebuilds the DEFAULT stretch")


def test_the_routes_that_stream_do_not_refuse_the_mode_that_asks_for_it():
    """mode = 'on' was refused at admission BY THE ROUTES THAT SUPPORT IT.

    refuse_unrouted_streaming was written when the prepared routes passed no
    builders.  They pass them now -- the test above is the proof -- but the
    admission refusal outlived the wiring, so the explicit form was rejected
    with a message asserting the route wired no builder while mode = 'auto'
    streamed through that very builder.  `gpuwm go` mirrored the refusal
    before the download, making the front door most users type the last
    place that rejected streaming.
    """
    import inspect

    import gpuwm.go_cli as go
    import gpuwm.prepared_domain_tree_forecast as tree
    import gpuwm.prepared_single_domain_forecast as single

    for module in (single, tree, go):
        src = inspect.getsource(module)
        calls = [line for line in src.splitlines()
                 if "refuse_unrouted_streaming(" in line
                 and not line.strip().startswith("#")]
        assert not calls, (
            f"{module.__name__} refuses [tiles] mode='on' at admission but "
            f"wires a streamed-domain builder: {calls}")


def test_the_route_that_never_reads_tiles_still_refuses():
    """The guard is REMOVED where it lied, not deleted where it is true."""
    import inspect

    import gpuwm.runtime as runtime

    src = inspect.getsource(runtime)
    assert "refuse_unrouted_streaming(exp, \"gpuwm run\", " \
        "consults_the_seam=False)" in src, (
            "gpuwm run reads [tiles] at no point; dropping its refusal would "
            "make it silently integrate resident, which is the silence the "
            "module forbids")


# --------------------------------------------------------------------------
# REFL_10CM: a REBUILT slot the store still has to hold
# --------------------------------------------------------------------------


class _ScratchState:
    """A state with just the scratch API prime/inventory use."""

    def __init__(self, **slots):
        self._scratch = dict(slots)
        self.requested = []

    def existing_scratch(self, slot):
        return self._scratch.get(slot)

    def scratch(self, shape, slot, dtype=None):
        self.requested.append((slot, tuple(shape)))
        self._scratch.setdefault(slot, ("array", tuple(shape)))
        return self._scratch[slot]


class _Cfg:
    nz, ny, nx = 49, 550, 550


def test_the_reflectivity_slot_is_primed_at_the_domains_extents():
    state = _ScratchState()
    assert streaming.prime_refl_10cm(state, _Cfg()) is True
    assert state.requested == [("refl_10cm", (49, 550, 550))]


def test_priming_an_already_primed_slot_does_nothing():
    """A slot keeps the shape it was first requested with, so never re-ask."""
    state = _ScratchState(refl_10cm=("array", (49, 550, 550)))
    assert streaming.prime_refl_10cm(state, _Cfg()) is False
    assert state.requested == []


def test_the_inventory_carries_the_primed_slot_for_state_and_buffer():
    """The transport applies one inventory_fn to state, store and buffers."""
    state = _ScratchState(refl_10cm="REFL")
    inventory = streaming.refl_inventory(lambda obj, names=None: {"state/thp": 1})
    assert inventory(state) == {"state/thp": 1,
                                streaming.REFL_STORE_KEY: "REFL"}
    # The store already carries it by key and must not be double-added.
    store = {"state/thp": 1, streaming.REFL_STORE_KEY: "REFL"}
    assert streaming.refl_inventory(
        lambda obj, names=None: dict(obj))(store) == store


def test_an_unprimed_slot_is_not_invented_by_the_inventory():
    """prime_refl_10cm creates the slot; the inventory only reports it.

    If the inventory allocated, the refusal for a domain that genuinely
    cannot publish reflectivity could never fire.
    """
    inventory = streaming.refl_inventory(lambda obj, names=None: {"state/thp": 1})
    assert inventory(_ScratchState()) == {"state/thp": 1}


def test_the_prepared_builder_primes_and_carries_the_reflectivity_slot():
    """Held as source: the alternative needs a card and an hour of forecast.

    The defect was an absent call -- the sweep refused the first DUE frame,
    which on an hourly cadence is an hour into a healthy forecast.

    The OUTER wrapper is asserted too, and for a harsher reason: leaving it
    off is not refused anywhere, it publishes a frame short of OLR and
    reports validity PASS.  That is what shipped until 2.2.0, so the
    composition -- diagnostic_inventory OVER refl_inventory, both of them
    over the streaming manifest -- is the thing under test, not either half.
    """
    import inspect

    src = inspect.getsource(streaming.prepared_domain_builder)
    assert "prime_lazy_carriers(state, cfg)" in src
    assert "inventory_fn=diagnostic_inventory(refl_inventory(" in src
    factory_src = inspect.getsource(streaming.prepared_tile_state_factory)
    assert "prime_lazy_carriers(tile, tile_cfg)" in factory_src


def test_the_diagnostic_wrapper_adds_the_output_only_rows_over_any_base():
    """Composable over the STREAMING manifest, not only the carrier one.

    ``tilestream.output.diagnostic_inventory`` composes over
    ``carrier_manifest``; this route's base is ``streaming_inventory``, the
    wider restart manifest.  Wrapping rather than substituting is what keeps
    one naming table, so this pins that the wrapper adds exactly the
    ``diag/`` members and leaves the base's own keys untouched.
    """
    import numpy as np

    class _Driver:
        def __init__(self):
            self.olr = np.zeros((4, 5), dtype=np.float32)

    class _State:
        def __init__(self):
            self.physics = _Driver()

        def existing_scratch(self, name):
            return None

    state = _State()
    base = lambda obj, names=None: {"state/thp": 1}          # noqa: E731
    live = streaming.diagnostic_inventory(base)(state)
    assert live["state/thp"] == 1
    assert live["diag/olr"] is state.physics.olr

    # A store already holds its members by key, so the wrapper adds nothing.
    store = {"state/thp": 1, "diag/olr": "OLR"}
    assert streaming.diagnostic_inventory(
        lambda obj, names=None: dict(obj))(store) == store


def test_the_eddy_viscosity_producers_are_primed_only_when_asked():
    """8 B/cell is 100x OLR's price; a run that does not ask must not pay it.

    And the slot names are LITERALS here for the same reason
    prime_refl_10cm's is: tests/test_preflight.py's scratch-completeness
    gate reads them with an AST scan and classifies a variable expression
    as unpinned.
    """
    import inspect

    src = inspect.getsource(streaming.prime_hmix_k_diag)
    assert '"smag_km"' in src and '"smag_kh"' in src

    class _Cfg:
        nx, ny, nz = 4, 5, 6
        hmix_k_diag = False

    state = _ScratchState()
    assert streaming.prime_hmix_k_diag(state, _Cfg()) == ()

    _Cfg.hmix_k_diag = True
    assert streaming.prime_hmix_k_diag(state, _Cfg()) == (
        "diag/XKMH", "diag/XKHH")
    # Idempotent: a slot that exists is left alone rather than reallocated.
    assert streaming.prime_hmix_k_diag(state, _Cfg()) == ()


# --------------------------------------------------------------------------
# priming REPLACES the throwaway warm-up step
# --------------------------------------------------------------------------


class _KFStandIn:
    def __init__(self):
        self.calls = 0

    def ensure_trigger_history(self, state):
        self.calls += 1
        return "w0avg"


class _PrimeState(_ScratchState):
    def __init__(self, cumulus=None, **slots):
        super().__init__(**slots)
        self.physics = type("_D", (), {"cumulus_callable": cumulus})()


def test_priming_allocates_both_lazy_carriers():
    kf = _KFStandIn()
    state = _PrimeState(cumulus=kf)
    assert streaming.prime_lazy_carriers(state, _Cfg()) == (
        streaming.REFL_STORE_KEY, "cumulus/w0avg")
    assert kf.calls == 1
    assert state.requested == [("refl_10cm", (49, 550, 550))]


def test_priming_a_domain_without_cumulus_primes_only_reflectivity():
    state = _PrimeState(cumulus=None)
    assert streaming.prime_lazy_carriers(state, _Cfg()) == (
        streaming.REFL_STORE_KEY,)


def test_the_prepared_factory_no_longer_steps_a_fabricated_buffer():
    """warmup defaults to 0, because the step was integrating fabricated data.

    A buffer at construction holds an analytic sounding, neutral geography
    and -- on a specified-boundary domain -- the DOMAIN's real lateral
    forcing.  The throwaway step blended real boundary values into an
    analytic interior; MEASURED, that handed rte-rrtmgp 211.0-456.7 K.
    """
    import inspect

    for fn in (streaming.prepared_tile_state_factory,
               streaming.prepared_domain_builder):
        default = inspect.signature(fn).parameters["warmup"].default
        assert default == 0, (
            f"{fn.__name__} still warms buffers by stepping them; the step "
            "integrates an analytic sounding against real lateral forcing")


# --------------------------------------------------------------------------
# the FINAL reads, which were answering for the initial condition
# --------------------------------------------------------------------------


def test_a_resident_stepper_refreshes_nothing_and_says_zero():
    """The OFF contract: a route calls this unconditionally."""
    assert streaming.refresh_streamed_state(object(), "STATE") == 0
    assert streaming.refresh_streamed_state(None, "STATE") == 0


def test_a_streamed_domain_is_copied_back_before_the_final_reads():
    """StateHealthValidator and canonical_state_digest cannot be folded.

    The validator is one block per whole field with no tile-interior form
    and the digest is a whole-trajectory hash, so under a host store both
    answered for the t=0 analysis: a receipt reporting nan_free on a run
    that had integrated for an hour.

    The stepper is a real StreamedDomain built field-wise, because
    ``is_streaming`` is an isinstance check -- a duck-typed stand-in is
    reported RESIDENT and would make this test pass against a no-op.
    """
    seen = {}

    def _refresh(state):
        seen["state"] = state
        return 137

    streamed = object.__new__(streaming.StreamedDomain)
    streamed.refresh_state = _refresh
    assert streaming.refresh_streamed_state(streamed, "STATE") == 137
    assert seen == {"state": "STATE"}


def test_a_resident_store_refresh_reports_zero_rather_than_copying():
    """``store = "device"`` makes the store the state's OWN arrays.

    refresh_state returns 0 there because a refresh would be a self-copy of
    the whole manifest, and the route must record that honestly rather than
    claim a refresh it did not need.
    """
    streamed = object.__new__(streaming.StreamedDomain)
    streamed.refresh_state = lambda state: 0
    assert streaming.refresh_streamed_state(streamed, "STATE") == 0


def test_the_prepared_route_refreshes_before_the_final_health_and_digest():
    """Held as source, and ORDER is the whole point.

    A refresh after the gate reads is a refresh that changed nothing.
    """
    import pathlib

    # Read rather than imported: the runner needs a CUDA build, and this is
    # a question about the ORDER of three statements in its source.
    src = (pathlib.Path(streaming.__file__).parents[1]
           / "prepared_single_domain_forecast.py").read_text(encoding="utf-8")
    refresh = src.index("streaming.refresh_streamed_state(")
    # The FINAL gate specifically.  The runner validates the state at entry
    # too, and that earlier call is not the one this is about: at entry the
    # state IS the domain, because nothing has streamed off it yet.
    health = src.index('validate(phase="final.d01")')
    digest = src.index("final_digest = canonical_state_digest(")
    assert refresh < health < digest, (
        "the streamed domain must be copied back BEFORE the final health "
        "gate and the canonical digest read node.state")
    assert '"carriers_refreshed_before_final_reads": carriers_refreshed' \
        in src, "the receipt must say whether the refresh happened"


# --------------------------------------------------------------------------
# the ANALYSIS frame's health, before any sweep has run
# --------------------------------------------------------------------------


def _bare_fold(**kw):
    """A StreamedStability with no device behind it.

    ``__init__`` allocates cupy buffers; every property these tests exercise
    is decided before a kernel is reached, so the object is built field-wise.
    """
    fold = object.__new__(streaming.StreamedStability)
    fold.cfg = kw.pop("cfg", "CFG")
    fold.boundary_width = kw.pop("boundary_width", 5)
    fold.ntiles = kw.pop("ntiles", 4)
    fold.sweeps_begun = kw.pop("sweeps_begun", 0)
    fold._seen = set(kw.pop("seen", ()))
    fold._report = None
    for key, value in kw.items():
        setattr(fold, key, value)
    return fold


def test_the_analysis_frame_reads_the_state_that_filled_the_store(monkeypatch):
    """A t=0 history frame is asked for its health before anything is stepped.

    There is no fold yet because there has been no sweep -- not a short one,
    none -- and the resident state is still the initial condition the frame
    contains, because attach copied it into the store and neither has moved.
    Refusing here stopped every [tiles] forecast on the product route at its
    ANALYSIS frame, after the sweep had already been proven to build.
    """
    import sys
    import types

    seen = {}

    def _resident(state, cfg, *, boundary_width=None):
        seen.update(state=state, cfg=cfg, boundary_width=boundary_width)
        return {"cfl_max": 0.25}

    # Stubbed as a module: the real gpuwm.core.dycore needs a CUDA build, and
    # what is under test is which reporter the fold reaches for.
    stub = types.ModuleType("gpuwm.core.dycore")
    stub.stability_report = _resident
    monkeypatch.setitem(sys.modules, "gpuwm.core.dycore", stub)
    fold = _bare_fold(sweeps_begun=0)
    assert fold("STATE", "RUNCFG", boundary_width=5) == {"cfl_max": 0.25}
    assert seen == {"state": "STATE", "cfg": "RUNCFG", "boundary_width": 5}


def test_the_analysis_frame_without_a_state_refuses_rather_than_inventing_one():
    fold = _bare_fold(sweeps_begun=0)
    with pytest.raises(StreamingRefused, match="before the first sweep"):
        fold(None, "RUNCFG", boundary_width=5)


def test_a_short_sweep_still_refuses_once_a_sweep_has_run():
    """The guard is NOT weakened: the exemption is keyed on sweeps_begun.

    Once a sweep has run, the resident state is the corpse the fold exists to
    stop reading, so a sweep that skipped tiles must refuse even though a
    state is right there and would answer.
    """
    fold = _bare_fold(sweeps_begun=1, seen=(0, 1))
    with pytest.raises(StreamingRefused, match="contributed a stability"):
        fold("STATE", "RUNCFG", boundary_width=5)


def test_begin_sweep_is_what_ends_the_analysis_frame_exemption():
    fold = _bare_fold(sweeps_begun=0)
    assert fold.sweeps_begun == 0
    streaming.StreamedStability.begin_sweep(fold)
    assert fold.sweeps_begun == 1
    # and the record is cleared, which is begin_sweep's original job
    assert fold._seen == set()


def test_the_ozone_grid_is_carried_by_a_checkpoint_and_not_by_a_sweep():
    """The fourth classification, and why it is not the third.

    Legacy RRTMG's radiation/o33d_grid is host memory the adapter recomputes
    on every radiation call and reads only from a CHILD domain's ozone
    provider.  A checkpoint must carry it; a sweep must not.  With it in the
    sweep's set, a tile buffer that warmed up had it and the prepared domain
    the store was sized from had not run a step, so the inventories differed
    by exactly this key and TiledRun refused every [tiles] forecast of the
    default physics suite.
    """
    from gpuwm.io import restart

    assert "radiation/o33d_grid" in restart.RESTART_ONLY_DRIVER_SLOTS
    # The two directions are genuinely different sets, not one list twice.
    assert not (restart.RESTART_ONLY_DRIVER_SLOTS
                & restart.CARRIED_SCRATCH_SLOTS)
    # The checkpoint half is UNTOUCHED: restart still names the key.
    import inspect
    assert 'manifest["radiation/o33d_grid"] = o33d_grid' in \
        inspect.getsource(restart._driver_manifest)


def test_the_sweep_manifest_subtracts_the_restart_only_slots(monkeypatch):
    """The subtraction itself, on both sides at once.

    Stubbed rather than driven from a real state: carrier_manifest's job here
    is set arithmetic, and the arithmetic is what regressed.
    """
    from gpuwm.io import restart
    from tilestream import physics_inventory as physinv

    monkeypatch.setattr(restart, "state_manifest", lambda s: {"state/thp": 1})
    monkeypatch.setattr(restart, "_scratch_manifest", lambda s: {})
    monkeypatch.setattr(restart, "carried_scratch_manifest", lambda s: {})
    monkeypatch.setattr(restart, "_driver_manifest", lambda d: {
        "cumulus/w0avg": 2, "radiation/o33d_grid": 3})

    class _State:
        physics = object()

    manifest = physinv.carrier_manifest(_State())
    assert "radiation/o33d_grid" not in manifest
    # Only that one: the lazy carrier that IS transported still is.
    assert manifest == {"state/thp": 1, "cumulus/w0avg": 2}


def test_the_legacy_sw_engine_is_shared_rather_than_compiled_per_buffer(
        monkeypatch):
    """One NVRTC compile and one set of device tables per process.

    Streaming builds an adapter per TILE BUFFER.  CudaSW compiles
    rrtmg_sw.cu and uploads the packed tables on construction, so an
    uncached engine would multiply both -- spending the device memory
    streaming exists to save.  Held here because the receipt on the card is
    a VRAM number nothing else asserts on.
    """
    from gpuwm.core import rrtmg_legacy

    built = []

    class _Tables:
        pass

    monkeypatch.setattr(rrtmg_legacy, "_CUDA_SW_CACHE", None)
    monkeypatch.setattr(rrtmg_legacy._sw, "CudaSW",
                        lambda tab: built.append(tab) or object())
    tables = _Tables()
    first = rrtmg_legacy._cuda_sw(tables)
    assert rrtmg_legacy._cuda_sw(tables) is first
    assert len(built) == 1, "the SW engine was compiled more than once"
    # Different coefficients get their own engine rather than borrowing.
    assert rrtmg_legacy._cuda_sw(_Tables()) is not first
    assert len(built) == 2


def test_the_budget_overrides_land_on_the_budget():
    """``vram_budget_bytes`` used to compute a budget of ZERO.

    ``Machine`` reaches its budgets through multipliers -- ``vram_bytes *
    (1 - vram_headroom)`` and ``host_bytes * pinned_fraction`` -- and
    ``decide`` replaced the raw capacity while leaving the multiplier on
    top.  For VRAM the multiplier it installed was ``1 - 1.0``, so every
    configuration that set the key was refused with "no tile fits in
    0.00 GiB of VRAM" however large a number it asked for.

    Held as an arithmetic identity rather than as a planner outcome,
    because the planner's answer depends on the domain and this does not:
    the number in the TOML must BE the budget.
    """
    from tilestream import autoplan

    class _Stop(Exception):
        """Stop inside the planner with the budgets it was actually given."""

    cfg = _cfg()
    seen = {}

    def spy(cfg_, machine, **kw):
        seen["vram"] = machine.vram_budget_bytes
        seen["host"] = machine.host_budget_bytes
        raise _Stop()

    real = autoplan.plan
    autoplan.plan = spy
    try:
        with pytest.raises(_Stop):
            decide(cfg, StreamingOptions(
                mode="auto",
                vram_budget_bytes=20 * 2 ** 30,
                host_budget_bytes=60 * 2 ** 30),
                machine=autoplan.Machine(vram_bytes=32 * 2 ** 30,
                                         host_bytes=128 * 2 ** 30))
    finally:
        autoplan.plan = real
    assert seen["vram"] == 20 * 2 ** 30, (
        f"[tiles] vram_budget_bytes = 20 GiB reached the planner as "
        f"{seen['vram']} bytes")
    assert seen["host"] == 60 * 2 ** 30


def test_write_mode_reaches_the_transport_instead_of_being_ignored():
    """``attach`` used to hardcode ``write_mode="ring"``.

    ``decide`` hands ``autoplan.plan`` the configured write mode and the
    planner sizes the HOST budget with it -- shadow keeps a whole second
    store, ring keeps one plus a small arena -- so a run planned for shadow
    and attached as ring was planned for one footprint and executed with
    another.  The value now rides on the decision, which is the object
    ``attach`` is given.
    """
    got = decide(_cfg(), StreamingOptions(mode="on", tile_nx=64, tile_ny=64,
                                          write_mode="shadow"))
    assert got.write_mode == "shadow"
    assert decide(_cfg(), StreamingOptions(mode="on", tile_nx=64,
                                           tile_ny=64)).write_mode == "ring"


def test_the_case_data_route_refuses_streaming_instead_of_ignoring_it():
    """``gpuwm.runtime.run_experiment`` cannot stream, so it must say so.

    It never reads ``exp.tiles``: its single-domain branch calls
    ``integrate_prepared_case`` without the ``stepper`` argument no caller
    in this checkout supplies, and its tree branch calls
    ``execute_experiment`` without ``steppers``.  Both therefore bind
    ``dycore.step`` unconditionally.  Silently running RESIDENT is the
    dangerous direction -- the user finds out at the allocation the mode
    existed to avoid -- so this refuses at the front door.
    """
    import inspect
    import io
    import tokenize

    from gpuwm import runtime

    src = inspect.getsource(runtime.run_experiment)
    assert "exp.tiles.enabled" in src, (
        "run_experiment no longer refuses a [tiles] configuration it "
        "cannot honour; it will run resident and say nothing")
    # And the reason the refusal is still needed: nobody hands the loop a
    # stepper.  If that ever changes, this test should be replaced by a
    # real streamed run, not deleted.
    #
    # COMMENTS ARE STRIPPED FIRST.  The question is whether a CALLER supplies
    # the argument, which is a property of the code; prose that names the
    # keyword while explaining that nothing passes it -- runtime.py has such a
    # comment, beside the front-door refusal itself -- is the opposite of the
    # thing being looked for, and matching it made this assertion fire on a
    # change that strengthened what it guards.
    source = inspect.getsource(runtime)
    rows = source.splitlines(keepends=True)
    for kind, _text, (r0, c0), (_r1, c1), _l in tokenize.generate_tokens(
            io.StringIO(source).readline):
        if kind == tokenize.COMMENT:
            # Blanked IN PLACE rather than dropped, so every other byte --
            # spacing included -- is still the source's own and the literal
            # `.replace` below keeps matching.
            line = rows[r0 - 1]
            rows[r0 - 1] = line[:c0] + " " * (c1 - c0) + line[c1:]
    whole = "".join(rows)
    assert "stepper=" not in whole.replace(
        "feedback=None, stepper=None", ""), (
        "a caller now supplies integrate_prepared_case's stepper; the "
        "front-door refusal above may be replaceable")
