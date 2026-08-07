"""CPU contracts for spawn materialization (gpuwm.ingest.nest_spawn_init).

Everything runs the REAL initialization machinery on NumPy states: the
real ``parent_only_init`` SINT fill, the real ``blend_terrain`` /
``adjust_tempqv`` / base re-derivation / ``press_adj`` sequence, and the
real executor topology for the end-to-end manual time spawn.  Only the
dycore solve is mocked (NumPy states are not valid CUDA integrator
inputs), exactly as the repo's own tree-topology tests do.

The instrument rule: the terrain adoption has a calibration point (flat
fine terrain must reproduce the plain parent-SINT child) and a
treatment proof in both directions (a hill must LOWER the column dry
mass at -g/alpha per metre, a valley must RAISE it).
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime

import numpy as np
import pytest

from gpuwm.config import RunConfig
from gpuwm.core.diagnostics import update_diagnostics
from gpuwm.core.grid import make_vertical_coord
from gpuwm.core.state import DomainState
from gpuwm.experiment import (DomainConfig, ExperimentConfig,
                              ProjectionConfig, VerticalConfig)
from gpuwm.ingest.nest_init import ParentInitView, parent_only_init
from gpuwm.ingest.nest_spawn_init import (SpawnInitRefusal,
                                          spawn_child_from_parent)
from gpuwm.ingest.real import _make_real_base
from gpuwm.static.lambert import grids_from_projection_config

NZ = 8
P_TOP = 5000.0
G = 9.81


def _run(nx, ny, *, nested, grid_id, dx, dt, moist=True):
    return RunConfig(nx=nx, ny=ny, nz=NZ, dx=dx, dy=dx, ztop=12000.0,
                     dt=dt, run_seconds=360.0, output_interval_s=360.0,
                     nested=nested, specified=not nested, grid_id=grid_id,
                     spec_bdy_width=5, spec_zone=1, relax_zone=4,
                     terrain_opt=1, moist=moist,
                     mp_physics=(1 if moist else 0), map_proj=1)


def _experiment(moist=True):
    prun = _run(60, 50, nested=False, grid_id=1, dx=12000.0, dt=60.0,
                moist=moist)
    crun = _run(45, 45, nested=True, grid_id=2, dx=4000.0, dt=20.0,
                moist=moist)
    root = DomainConfig(
        grid_id=1, parent_id=0, i_parent_start=1, j_parent_start=1,
        parent_grid_ratio=1, parent_time_step_ratio=1,
        history_interval_s=360.0, run=prun, time_step=60)
    child = DomainConfig(
        grid_id=2, parent_id=1, i_parent_start=20, j_parent_start=18,
        parent_grid_ratio=3, parent_time_step_ratio=3,
        history_interval_s=360.0, run=crun)
    return ExperimentConfig(
        name="spawn_micro", start_time=datetime(1974, 4, 3, 12),
        run_seconds=360.0, vertical=VerticalConfig((), 0.0, 1, 0.2),
        projection=ProjectionConfig(
            "lambert", 35.0, -97.0, 30.0, 60.0, -97.0),
        restart_interval_s=0.0, domains=(root, child))


def _live_parent(exp, *, parent_terrain=None):
    """A REAL NumPy parent state: analytic real base + weather-ish fields."""
    grids = tuple(grids_from_projection_config(exp))
    prun = exp.root.run
    coord = make_vertical_coord(NZ)
    terrain = (np.zeros((prun.ny, prun.nx)) if parent_terrain is None
               else np.asarray(parent_terrain, dtype=np.float64))
    base = _make_real_base(coord, terrain, P_TOP, prun.base_temp,
                           prun.hypsometric_opt)
    state = DomainState(prun, array_module=np)
    state.load_base(coord, base)
    f, e = grids[0].coriolis_m()
    sina, cosa = grids[0].rotation_m()
    state.set_map_coriolis(grids[0].mapfac_m(), grids[0].mapfac_u(),
                           grids[0].mapfac_v(), f, e, sina=sina, cosa=cosa)
    rng = np.random.default_rng(7)
    state.u[...] = 5.0 + rng.normal(0, 0.5, state.u.shape)
    state.v[...] = rng.normal(0, 0.5, state.v.shape)
    state.thp[...] = rng.normal(0, 0.5, state.thp.shape)
    if prun.moist:
        state.qv[...] = 0.004
    update_diagnostics(state, prun.hypsometric_opt)
    return ParentInitView(cfg=exp.root, grid=grids[0], state=state), grids


def _host(value):
    return np.asarray(value)


@pytest.fixture(scope="module")
def micro():
    exp = _experiment()
    parent, grids = _live_parent(exp)
    return {"exp": exp, "parent": parent, "grids": grids}


# ---------------------------------------------------------------------------
# The statics-free branch: exactly parent_only_init
# ---------------------------------------------------------------------------

def test_no_statics_spawn_is_bitwise_parent_only_init(micro):
    from gpuwm.ensemble.state_sha import live_state_sha256

    child_dc = micro["exp"].domains[1]
    out = spawn_child_from_parent(child_dc, micro["parent"],
                                  array_module=np)
    direct = parent_only_init(child_dc, micro["parent"], array_module=np)
    assert out["child_state_sha256"] == live_state_sha256(direct.state)
    assert out["terrain"]["static_source"] == "parent-sint"
    assert out["parent_bitwise_unchanged"] is True
    assert out["child_result"].static_fields is None


def test_receipt_carries_the_birth_certificate(micro):
    child_dc = micro["exp"].domains[1]
    trigger = {"decision": "fired", "trigger": "uh",
               "placement": [20, 18]}
    out = spawn_child_from_parent(child_dc, micro["parent"],
                                  array_module=np,
                                  trigger_receipt=trigger)
    assert out["contract"] == "gpuwm-nest-spawn-init.v1"
    assert out["trigger"] is trigger
    assert out["placement"] == [20, 18]
    assert out["atmosphere_source"]["kind"] == "parent-sint"
    assert out["atmosphere_source"]["parent_state_sha256"] == \
        out["parent_state_sha256_before"]
    assert out["rk_seeds_refreshed"]


def test_on_child_built_fires_before_the_receipts(micro):
    child_dc = micro["exp"].domains[1]
    seen = []

    def prepare(initialized, dc, parent_node):
        seen.append((initialized.state is not None, dc.grid_id,
                     parent_node is micro["parent"]))

    spawn_child_from_parent(child_dc, micro["parent"], array_module=np,
                            on_child_built=prepare)
    assert seen == [(True, 2, True)]


def test_off_grid_fired_placement_refuses_at_the_stencil(micro):
    child_dc = replace(micro["exp"].domains[1], i_parent_start=55)
    with pytest.raises(ValueError, match="outside the parent extent"):
        spawn_child_from_parent(child_dc, micro["parent"],
                                array_module=np)


# ---------------------------------------------------------------------------
# Terrain adoption: calibration and treatment, both directions
# ---------------------------------------------------------------------------

def test_flat_terrain_adoption_is_the_identity(micro):
    """The null spawn: fine terrain identical to the parent-SINT terrain
    must reproduce the plain parent-only child.  Bitwise on every field
    except thp, whose one-ULP wobble is the real path's own theta
    -300K/+300K FP32 roundtrip (adjust_tempqv's frame), not ours."""
    child_dc = micro["exp"].domains[1]
    crun = child_dc.run
    plain = spawn_child_from_parent(child_dc, micro["parent"],
                                    array_module=np)
    flat = spawn_child_from_parent(
        child_dc, micro["parent"], array_module=np,
        static_fields={"HGT_M": np.zeros((crun.ny, crun.nx))})
    a, b = plain["child_result"].state, flat["child_result"].state
    for name in ("ht", "mub2d", "phb", "pb", "alb", "thb",
                 "mup", "php", "u", "v", "w", "qv", "p", "al", "alt"):
        left, right = getattr(a, name, None), getattr(b, name, None)
        assert left is not None and right is not None, name
        assert np.array_equal(_host(left), _host(right)), name
    theta_wobble = np.max(np.abs(_host(a.thp).astype(np.float64)
                                 - _host(b.thp).astype(np.float64)))
    assert theta_wobble <= 3.1e-5  # one FP32 ULP at ~300 K
    assert flat["terrain"]["static_source"] == "own-grid"
    assert flat["terrain"]["terrain_max_abs_shift_m"] == 0.0


def test_hill_and_valley_shift_column_mass_hydrostatically(micro):
    """Treatment proof, both signs: dry column mass moves by ~ -g/alpha
    per metre of adopted terrain (the hydrostatic answer), interior
    only, zero where the fine ground equals the parent's."""
    child_dc = micro["exp"].domains[1]
    crun = child_dc.run
    plain = spawn_child_from_parent(child_dc, micro["parent"],
                                    array_module=np)
    base_mub = _host(plain["child_result"].state.mub2d).astype(np.float64)
    alpha0 = float(_host(plain["child_result"].state.alb)[0, 22, 22])
    expected_per_m = G / alpha0

    results = {}
    for label, height in (("hill", 300.0), ("valley", -300.0)):
        terrain = np.zeros((crun.ny, crun.nx))
        terrain[20:25, 20:25] = height
        out = spawn_child_from_parent(
            child_dc, micro["parent"], array_module=np,
            static_fields={"HGT_M": terrain})
        state = out["child_result"].state
        dmub = _host(state.mub2d).astype(np.float64) - base_mub
        results[label] = dmub
        for name in ("thp", "mup", "p", "al", "alt", "qv", "phb"):
            assert np.isfinite(_host(getattr(state, name))).all(), \
                (label, name)
    per_metre_hill = -results["hill"][22, 22] / 300.0
    per_metre_valley = results["valley"][22, 22] / 300.0
    assert results["hill"][22, 22] < 0.0 < results["valley"][22, 22]
    for measured in (per_metre_hill, per_metre_valley):
        assert expected_per_m * 0.9 <= measured <= expected_per_m * 1.1
    # Ground the parent already described is untouched.
    assert results["hill"][5, 5] == 0.0
    assert results["valley"][5, 5] == 0.0
    # And the two arms are mirror images to first order.
    assert abs(per_metre_hill - per_metre_valley) < 0.1 * expected_per_m


def test_adoption_refusals_are_loud(micro):
    child_dc = micro["exp"].domains[1]
    crun = child_dc.run
    wrong_shape = {"HGT_M": np.zeros((5, 5))}
    with pytest.raises(SpawnInitRefusal, match="different footprint"):
        spawn_child_from_parent(child_dc, micro["parent"],
                                array_module=np,
                                static_fields=wrong_shape)
    with pytest.raises(SpawnInitRefusal, match="no HGT_M"):
        spawn_child_from_parent(child_dc, micro["parent"],
                                array_module=np, static_fields={})
    # A dry tree admits the statics-free spawn but refuses the adoption,
    # exactly as the real-data child path refuses non-moist init.
    dry_exp = _experiment(moist=False)
    dry_parent, _grids = _live_parent(dry_exp)
    dry_child = dry_exp.domains[1]
    assert spawn_child_from_parent(
        dry_child, dry_parent, array_module=np)["parent_bitwise_unchanged"]
    with pytest.raises(SpawnInitRefusal, match="moist"):
        spawn_child_from_parent(
            dry_child, dry_parent, array_module=np,
            static_fields={"HGT_M": np.zeros((crun.ny, crun.nx))})


# ---------------------------------------------------------------------------
# End to end: manual time spawn across two executor legs, on CPU
# ---------------------------------------------------------------------------

def test_manual_time_spawn_end_to_end_across_executor_legs(
        micro, monkeypatch):
    """The whole story on a tiny CPU case: a dormant nest is declared;
    leg A integrates the pre-spawn tree (REAL clocks/schedule, mocked
    solve); the time trigger fires at the leg boundary; the nest
    materializes from the LIVE parent through the real init machinery;
    leg B integrates the activated tree with the spawned child in it.
    This is the sequence-of-static-trees pattern the relocation demo
    established, driven by the spawn seams the leg-2 runner will consume.
    """
    from gpuwm.core.model import execute_experiment
    from gpuwm.core.nest_spawn import SpawnConfig, SpawnController
    from gpuwm.experiment import active_experiment, pre_spawn_experiment
    from gpuwm.verify.cases.nest_ideal_common import assemble_idealized_tree

    exp = micro["exp"]
    dormant = replace(exp.domains[1],
                      spawn=SpawnConfig(trigger="time", at_s=120.0))
    dexp = replace(exp, domains=(exp.domains[0], dormant))

    stepped: list[int] = []
    monkeypatch.setattr(
        "gpuwm.core.dycore.step",
        lambda state, cfg, **_kwargs: stepped.append(cfg.grid_id))

    # ---- leg A: the pre-spawn tree (root only; the nest costs zero
    # compute while dormant) -------------------------------------------
    pre = pre_spawn_experiment(dexp)
    assert [dc.grid_id for dc in pre.domains] == [1]
    leg_a = replace(pre, run_seconds=120.0, domains=(
        replace(pre.domains[0], run=replace(
            pre.domains[0].run, run_seconds=120.0)),))
    parent_state = micro["parent"].state
    model_a = assemble_idealized_tree(
        leg_a, parent_state,
        grids=(micro["grids"][0],),
        domain_preparer=lambda *_args, **_kwargs: None)
    report_a = execute_experiment(model_a, validate_state=False)
    assert report_a.steps == 2 and stepped == [1, 1]

    # ---- the trigger fires at the leg boundary -----------------------
    controller = SpawnController.from_experiment(dexp)
    assert controller.pending == (2,)
    events = controller.evaluate_all(
        {1: model_a.root.state},
        float(model_a.root.clock.elapsed_seconds))
    assert len(events) == 1
    event = events[0]
    assert event.grid_id == 2 and event.position == (20, 18)
    assert event.receipt["decision"] == "fired"

    # ---- materialize from the LIVE parent ----------------------------
    from gpuwm.ingest.nest_spawn_init import spawn_child_from_parent
    spawned = spawn_child_from_parent(
        dexp.domain(2), model_a.root, array_module=np,
        trigger_receipt=event.receipt)
    assert spawned["parent_bitwise_unchanged"]

    # ---- leg B: the activated tree -----------------------------------
    act = active_experiment(dexp, {2: event.position})
    assert [dc.grid_id for dc in act.domains] == [1, 2]
    leg_b = replace(act, run_seconds=120.0, domains=tuple(
        replace(dc, history_interval_s=120.0, run=replace(
            dc.run, run_seconds=120.0, output_interval_s=120.0))
        for dc in act.domains))

    child_result = spawned["child_result"]

    def adopt(child_dc, _parent_node, **_kwargs):
        from gpuwm.ingest.nest_init import ChildInitResult

        return ChildInitResult(
            state=child_result.state, grid=child_result.grid,
            coord=child_result.coord, real=None, static_fields=None,
            horizontal=None, soil=None, domain=child_dc)

    class _CountingCoupler:
        def __init__(self, child):
            self.child = child
            self.valid = False
            self.forces = 0

        def force(self, node):
            assert node is self.child
            self.valid = True
            self.forces += 1

        def feedback_prepare(self, node, out):
            out.payload = None

        def feedback_commit(self, node):
            pass

        def feedback_finalize(self, node):
            pass

    stepped.clear()
    model_b = assemble_idealized_tree(
        leg_b, model_a.root.state,
        grids=grids_from_projection_config(leg_b),
        child_initializer=adopt,
        coupler_factory=_CountingCoupler,
        domain_preparer=lambda *_args, **_kwargs: None)
    child_node = model_b.node(2)
    assert child_node.state is child_result.state
    assert child_node.parent is model_b.root
    report_b = execute_experiment(model_b, validate_state=False)
    assert report_b.steps == 2 + 6           # 2 parent + 2*3 child substeps
    assert stepped.count(1) == 2 and stepped.count(2) == 6
    assert child_node.coupler.forces == 2    # one FORCE per parent step
    # The spawned child's state stayed finite through the walk.
    for name in ("thp", "mup", "u", "v", "w"):
        assert np.isfinite(_host(getattr(child_node.state, name))).all()
