"""DA nests use the prepared tree's acoustic rule on their own SINT ground."""
import ast
from datetime import datetime, timedelta
import inspect
import pickle
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from gpuwm.acoustic_adaptation import acoustic_receipt, steepest_slope
from gpuwm.config import RunConfig
from gpuwm.da import nested_forecast as nf
from gpuwm.experiment import (
    DomainConfig, ExperimentConfig, ProjectionConfig, VerticalConfig)
from gpuwm.physics_compat import WSM6_PROFILE_ID, single_domain_runtime_switches
from gpuwm.static.lambert import LambertGrid


def _experiment(*, epssm=0.1, auto=True, latitude=35.0, terrain_opt=1):
    switches = dict(single_domain_runtime_switches(WSM6_PROFILE_ID))
    switches.pop("acknowledgements", None)
    values = {
        key: value for key, value in switches.items()
        if key in RunConfig.__dataclass_fields__}
    values.update(
        nx=33, ny=33, nz=49, dx=3000.0, dy=3000.0, dt=15.0,
        ztop=20000.0, run_seconds=900.0, output_interval_s=900.0,
        specified=True, nested=False, grid_id=1, spec_bdy_width=5,
        spec_zone=1, relax_zone=4, moist=True, hypsometric_opt=2,
        map_proj=1, epssm=epssm, time_step_sound=4, terrain_opt=terrain_opt)
    run = RunConfig(**values)
    parent = DomainConfig(
        grid_id=1, parent_id=0, i_parent_start=1, j_parent_start=1,
        parent_grid_ratio=1, parent_time_step_ratio=1,
        history_interval_s=900.0, run=run, time_step=15)
    exp = ExperimentConfig(
        name="acoustic-nest", start_time=datetime(2024, 1, 1),
        run_seconds=900.0,
        vertical=VerticalConfig(tuple(np.linspace(1.0, 0.0, 50)),
                                5000.0, 2, 0.2),
        projection=ProjectionConfig("lambert", latitude, 0.0,
                                    30.0, 60.0, 0.0),
        restart_interval_s=0.0, domains=(parent,),
        auto_epssm=(1,) if auto else ())
    grid = LambertGrid(
        ref_lat=latitude, ref_lon=0.0, truelat1=30.0, truelat2=60.0,
        stand_lon=0.0, dx=run.dx, dy=run.dy,
        e_we=run.nx + 1, e_sn=run.ny + 1)
    child = nf.nest_domain_config(
        exp, nf.NestGeometry(ratio=3, nx=27, ny=27))
    return exp, child, grid


def _plane(exp, slope):
    shape = (exp.root.run.ny, exp.root.run.nx)
    return np.broadcast_to(
        200.0 + slope * exp.root.run.dx * np.arange(shape[1]),
        shape).astype(np.float32).copy()


def _child_grid(parent_grid, child):
    return parent_grid.nest(
        child.i_parent_start, child.j_parent_start, child.parent_grid_ratio,
        child.run.nx + 1, child.run.ny + 1,
        resolved_dx=child.run.dx, resolved_dy=child.run.dy)


def _prepared_door(exp, child, terrain, parent_grid):
    """Run the real prepared tree entry's terrain adaptation."""
    from gpuwm.prepared_domain_tree_forecast import _with_terrain_acoustics
    from gpuwm.wrfinput_forecast import WrfDomainBundle, WrfTreeInputs

    nested = nf.nested_experiment(exp, child)
    ground = nf.nest_terrain(exp, child, terrain)
    bundles = tuple(
        WrfDomainBundle(
            grid_id=gid, restored=None,
            static_fields=MappingProxyType({"HGT_M": heights}),
            authority_sha256={}, landuse=None, geog_selection=None)
        for gid, heights in (
            (1, np.full_like(terrain, 200.0)), (2, ground)))
    inputs = WrfTreeInputs(
        prepared_root=None, experiment_config=None, experiment=nested,
        grids=(parent_grid, _child_grid(parent_grid, child)),
        domains=bundles, forcing_hours=(0.0, 1.0),
        boundary_interval_seconds=3600, source_identity={},
        execution_plan={}, authority_sha256={}, artifact_paths={},
        boundaries=None)
    return _with_terrain_acoustics(inputs)


def test_da_child_terrain_is_the_initializers_sint_ground():
    from gpuwm.ingest.nest_init import _parent_only_base

    exp, child, _ = _experiment()
    terrain = _plane(exp, 0.82)
    shape = terrain.shape
    parent = SimpleNamespace(
        mub2d=np.ones(shape, dtype=np.float32), p_top=5000.0,
        pb=np.ones((2, *shape), dtype=np.float32),
        alb=np.ones((2, *shape), dtype=np.float32),
        thb=np.ones((2, *shape), dtype=np.float32),
        phb=np.ones((3, *shape), dtype=np.float32), ht=terrain)
    initial = _parent_only_base(
        parent, nf.donor_registration(child, exp.root.run), True)
    ground = nf.nest_terrain(exp, child, terrain)
    assert ground.dtype == np.float32
    assert ground.shape == (child.run.ny, child.run.nx)
    # BaseState carries host setup values as float64. DomainState.load_base
    # installs terrain as float32, which is the ground the child runs on.
    assert initial.terrain_z.dtype == np.float64
    np.testing.assert_array_equal(initial.terrain_z, ground.astype(np.float64))
    assert ground.tobytes() == np.asarray(
        initial.terrain_z, dtype=np.float32).tobytes()
    # Static land fields use nearest-donor, which has a different slope.
    donor_ground = nf.donor_nest_down(
        terrain, nf.donor_registration(child, exp.root.run))
    assert ground.tobytes() != donor_ground.tobytes()


@pytest.mark.parametrize("auto", [False, True])
def test_da_flat_terrain_option_ignores_nonzero_static_ground(auto):
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.state import DomainState
    from gpuwm.ingest.nest_init import _parent_only_base

    exp, child, grid = _experiment(epssm=0.1, auto=auto, terrain_opt=0)
    assert child.run.terrain_opt == 0
    # Nonzero static fields can be present, but this dynamics choice uses
    # the flat initializer branch and ignores them for the child's ground.
    terrain = _plane(exp, 1.7)
    coord = make_vertical_coord(child.run.nz, hybrid_opt=child.run.hybrid_opt,
                                etac=child.run.etac)
    flat_base = make_base_state(
        coord, lambda z: np.full_like(z, 300.0),
        p_surf=child.run.p_surf, ztop=child.run.ztop)
    parent = SimpleNamespace(
        **{name: getattr(flat_base, name)
           for name in ("mub", "p_top", "pb", "alb", "thb", "phb")},
        ht=terrain)
    initial = _parent_only_base(
        parent, nf.donor_registration(child, exp.root.run), False)
    assert initial.terrain_z is None
    state = DomainState(child.run, array_module=np)
    state.load_base(coord, initial)
    ground = nf.nest_terrain(exp, child, terrain)
    assert ground.tobytes() == state.ht.tobytes()
    assert np.count_nonzero(ground) == 0
    before = pickle.dumps(child.run)
    adapted, acoustic = nf.nest_acoustics(
        exp, child, parent_terrain=terrain, parent_grid=grid)
    assert adapted is child
    assert pickle.dumps(adapted.run) == before
    assert adapted.run.epssm == 0.1
    assert adapted.run.time_step_sound == 4
    assert acoustic[0].reading.slope == 0.0
    prepared = _prepared_door(exp, child, terrain, grid)
    assert pickle.dumps(prepared.experiment.domains[-1].run) == before


@pytest.mark.parametrize("epssm,auto", [(0.1, True), (0.5, True), (0.5, False)])
def test_da_steep_child_takes_the_prepared_tree_floor_and_substeps(epssm, auto):
    exp, child, grid = _experiment(epssm=epssm, auto=auto)
    terrain = _plane(exp, 0.82)
    adapted, acoustic = nf.nest_acoustics(
        exp, child, parent_terrain=terrain, parent_grid=grid)
    prepared = _prepared_door(exp, child, terrain, grid)
    expected = prepared.experiment.domains[-1]
    assert adapted.run.epssm == expected.run.epssm == 0.5
    assert adapted.run.time_step_sound == expected.run.time_step_sound == 6
    assert pickle.dumps(adapted.run) == pickle.dumps(expected.run)
    assert acoustic_receipt(acoustic)["domains"] == [
        prepared.acoustic_substeps["domains"][-1]]
    assert exp.root.run.epssm == epssm
    assert exp.root.run.time_step_sound == 4


def test_da_child_reads_its_own_map_factors():
    exp, child, grid = _experiment(latitude=80.0)
    terrain = _plane(exp, 0.48)
    ground = nf.nest_terrain(exp, child, terrain)
    unscaled = steepest_slope(ground, child.run.dx, child.run.dy)
    assert unscaled.slope < 0.5
    adapted, acoustic = nf.nest_acoustics(
        exp, child, parent_terrain=terrain, parent_grid=grid)
    prepared = _prepared_door(exp, child, terrain, grid)
    assert acoustic[0].reading.slope > 0.5
    assert adapted.run.epssm > 0.1
    assert adapted.run == prepared.experiment.domains[-1].run
    assert acoustic_receipt(acoustic)["domains"] == [
        prepared.acoustic_substeps["domains"][-1]]


@pytest.mark.parametrize("epssm", [0.1, 0.3])
def test_da_explicit_epssm_below_the_floor_is_refused_with_remedy(epssm):
    exp, child, grid = _experiment(epssm=epssm, auto=False)
    terrain = _plane(exp, 0.82)
    with pytest.raises(ValueError) as caught:
        nf.nest_acoustics(
            exp, child, parent_terrain=terrain, parent_grid=grid)
    refusal = str(caught.value)
    assert f"d02's epssm {epssm:g} is set explicitly" in refusal
    assert "set d02's epssm to at least 0.5" in refusal
    assert 'or "auto", or leave it unset' in refusal
    assert "stopped within minutes" in refusal
    with pytest.raises(ValueError) as prepared:
        _prepared_door(exp, child, terrain, grid)
    assert str(prepared.value) == refusal
    assert child.run.epssm == epssm


@pytest.mark.parametrize("epssm,auto", [
    (0.1, True), (0.5, True), (0.1, False), (0.5, False)])
def test_da_flat_child_keeps_the_heads_inherited_run_bytes(epssm, auto):
    exp, child, grid = _experiment(epssm=epssm, auto=auto)
    before = pickle.dumps(child.run)
    adapted, acoustic = nf.nest_acoustics(
        exp, child, parent_terrain=np.full((33, 33), 200.0),
        parent_grid=grid)
    assert adapted is child
    assert pickle.dumps(adapted.run) == before
    assert len(acoustic) == 1
    assert acoustic[0].reading.slope == 0.0
    assert not acoustic[0].adapted
    assert not acoustic[0].offcentering_raised


@pytest.mark.parametrize("auto,expected", [(True, (1, 2)), (False, ())])
def test_da_child_inherits_the_parents_auto_label(auto, expected):
    exp, child, _ = _experiment(auto=auto)
    nested = nf.nested_experiment(exp, child)
    assert nested.auto_epssm == expected
    assert nested.domains[0] is exp.root
    assert nested.domains[-1] is child
    assert exp.auto_epssm == ((1,) if auto else ())


def test_da_each_leg_keeps_the_adapted_child_configuration():
    exp, child, grid = _experiment()
    adapted, _ = nf.nest_acoustics(
        exp, child, parent_terrain=_plane(exp, 0.82), parent_grid=grid)
    for birth in (0.0, 270.0, 540.0):
        leg_child = nf.child_born_at(adapted, exp, birth)
        leg_exp = nf.nested_experiment(exp, leg_child)
        assert leg_child.run is adapted.run
        assert leg_child.run.epssm == 0.5
        assert leg_child.run.time_step_sound == 6
        assert leg_exp.auto_epssm == (1, 2)


@pytest.mark.parametrize("auto", [False, True])
@pytest.mark.parametrize("nested_leg", [False, True])
def test_da_leg_writer_carries_the_callers_labels_into_actual_checkpoints(
        monkeypatch, tmp_path, auto, nested_leg):
    from gpuwm.io import restart
    from test_restart import _sealed_tree_fixture
    from tools import da_cycle_prepared as driver

    exp, child, _ = _experiment(auto=auto)
    model, start = _sealed_tree_fixture(
        monkeypatch, forcing_count=2, run_seconds=3600.0, payload_seed=31)
    if not nested_leg:
        model.root.children.clear()
        model.nodes_by_grid_id.pop(2)
        model.walk_parent_first = lambda: iter((model.root,))
    # A specified root has no diabatic heating in its fixed ring.
    field = model.root.state.h_diabatic
    width = model.root.cfg.run.spec_zone
    field[:, :width, :] = 0
    field[:, -width:, :] = 0
    field[:, :, :width] = 0
    field[:, :, -width:] = 0
    assert not hasattr(model, "_declared_experiment")

    # Execute the actual label expression at the cycle's checkpoint call.
    # Calling only the helper would miss the original route's absent label.
    calls = [node for node in ast.walk(ast.parse(inspect.getsource(driver.cycle)))
             if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name)
             and node.func.id == "write_leg_restart"]
    assert len(calls) == 1
    labels = next(keyword.value for keyword in calls[0].keywords
                  if keyword.arg == "auto_epssm")
    expression = ast.fix_missing_locations(ast.Expression(body=labels))
    chosen = eval(compile(expression, "cycle-labels", "eval"), {
        "nested_forecast": nf, "exp": exp, "nest_child_dc": child,
        "model": model})
    expected = (tuple(model.nodes_by_grid_id) if auto else ())
    assert chosen == expected

    root = driver.write_leg_restart(
        model, tmp_path / "checkpoints", auto_epssm=chosen,
        valid_time=start + timedelta(seconds=3600))
    members = restart.tree_restart_members(root)
    assert set(members) == set(model.nodes_by_grid_id)
    for gid, member in members.items():
        header = restart.read_restart_header(member)
        if auto:
            assert header[restart.AUTO_EPSSM_HEADER_KEY] is True
        else:
            assert restart.AUTO_EPSSM_HEADER_KEY not in header
