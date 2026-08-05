"""The parent-inertness ratchet for the radar-DA fine nested forecast leg.

This is the same claim the N3/N4/N5 nest gates hold for the campaign
domains -- attaching a child to a one-way parent must not alter the
parent's trajectory -- carried to the surface this work adds: a nest
built from the parent's LIVE state, assembled through
:mod:`gpuwm.da.nested_forecast` and stepped by the same hand-built
``ExperimentState`` the DA driver uses.

It is a ratchet, not a smoke test.  The comparison is bitwise over the
whole serialised parent state, and it is paired with a liveness assertion
so it cannot pass by the nest quietly doing nothing.
"""

from __future__ import annotations

from datetime import datetime
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest
from conftest import requires_gpu

from gpuwm.config import RunConfig
from gpuwm.da import nested_forecast as nf
from gpuwm.experiment import (DomainConfig, ExperimentConfig,
                              ProjectionConfig, VerticalConfig)
from gpuwm.physics_compat import (WSM6_PROFILE_ID,
                                  single_domain_runtime_switches)

#: Small enough to be a correctness burst, large enough that the child
#: has a real interior outside its own five-point specified frame plus
#: four-cell relaxation zone.
PARENT_NX = PARENT_NY = 33
PARENT_NZ = 49
RATIO = 3
CHILD_NX = CHILD_NY = 27
PARENT_DT = 15.0
#: Four parent steps, hence twelve child steps and four FORCE edges.
RUN_SECONDS = 60.0


# ---------------------------------------------------------------------------
# a cheap parent that nevertheless carries terrain
# ---------------------------------------------------------------------------

def _parent_run() -> RunConfig:
    switches = dict(single_domain_runtime_switches(WSM6_PROFILE_ID))
    switches.pop("acknowledgements", None)
    base = dict(
        nx=PARENT_NX, ny=PARENT_NY, nz=PARENT_NZ,
        dx=3000.0, dy=3000.0, ztop=20000.0, dt=PARENT_DT,
        run_seconds=RUN_SECONDS, output_interval_s=RUN_SECONDS,
        # The parent is PERIODIC here, not specified. The nowcast's parent
        # is specified and reads external LBC tables; this fixture has
        # none and does not need them, because the property under test is
        # about the CHILD's presence, and the parent's own lateral
        # treatment is identical in both arms of the comparison. Making it
        # periodic is what lets the ratchet run without a prepared cache.
        specified=False, nested=False, grid_id=1,
        spec_bdy_width=5, spec_zone=1, relax_zone=4,
        moist=True, hypsometric_opt=2, map_proj=1,
        case="nested_forecast_ratchet",
    )
    base.update({key: value for key, value in switches.items()
                 if key in RunConfig.__dataclass_fields__})
    return RunConfig(**base)


def _experiment(run: RunConfig) -> ExperimentConfig:
    root = DomainConfig(
        grid_id=1, parent_id=0, i_parent_start=1, j_parent_start=1,
        parent_grid_ratio=1, parent_time_step_ratio=1,
        history_interval_s=RUN_SECONDS, run=run, time_step=int(PARENT_DT))
    return ExperimentConfig(
        name="nested_forecast_ratchet",
        start_time=datetime(2024, 5, 21, 21),
        run_seconds=RUN_SECONDS,
        vertical=VerticalConfig((), 0.0, 1, 0.2),
        projection=ProjectionConfig("lambert", 35.0, -97.0, 30.0, 60.0,
                                    -97.0),
        restart_interval_s=0.0, domains=(root,))


def _terrain(run: RunConfig) -> np.ndarray:
    """A smooth hill: enough relief that the terrain SINT branch matters."""
    j = np.arange(run.ny)[:, None]
    i = np.arange(run.nx)[None, :]
    cj, ci = (run.ny - 1) / 2.0, (run.nx - 1) / 2.0
    radius = np.sqrt(((i - ci) / (run.nx / 3.0)) ** 2
                     + ((j - cj) / (run.ny / 3.0)) ** 2)
    return (400.0 * np.exp(-radius ** 2)).astype(np.float64)


def _build_parent_state(run: RunConfig):
    """A hydrostatically balanced parent over that hill, with a thermal.

    ``terrain_opt=1`` is not decoration: it is what the shipped nowcast
    profile carries, and it selects the three-dimensional base-state
    allocation and the terrain branch of ``parent_only_init``'s base
    capture.  A flat fixture would exercise a different branch than the
    one production takes.  ``init_theta_perturbation`` is the terrain-
    capable balanced initialiser (``init_moist_balanced``, which WK82
    uses, refuses a terrain base state by construction).
    """
    import cupy as cp

    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.state import DTYPE, init_theta_perturbation
    from gpuwm.verify.cases import wk82

    coord = make_vertical_coord(run.nz)
    terrain = _terrain(run)
    base = make_base_state(
        coord, lambda z: wk82.wk82_sounding(z)[0],
        p_surf=run.p_surf, ztop=run.ztop, terrain_z=terrain)

    y = (np.arange(run.ny) + 0.5) * run.dy - 0.5 * run.ny * run.dy

    def thp_func(x, z):
        """A 3 K ellipsoidal thermal, so the child has something to do."""
        xrad = x[None, None, :] / 12_000.0
        yrad = y[None, :, None] / 12_000.0
        zrad = (z - 1_500.0) / 1_500.0
        rad = np.sqrt(xrad ** 2 + yrad ** 2 + zrad ** 2)
        return np.where(rad <= 1.0,
                        3.0 * np.cos(0.5 * np.pi * rad) ** 2, 0.0)

    state = init_theta_perturbation(run, coord, base, thp_func,
                                    terrain_z=terrain)
    # A sheared hodograph so the coupler's boundary tables carry real
    # gradients rather than a uniform field that hides an error.
    u_profile, v_profile = wk82.hodograph_uv(coord, run.ztop)
    state.u[...] = cp.asarray(u_profile, dtype=DTYPE)[:, None, None]
    state.v[...] = cp.asarray(v_profile, dtype=DTYPE)[:, None, None]
    # The EOS diagnostics are what the health gate reads first; the
    # production initialisers refresh them at the end of setup and this
    # fixture has to do the same.
    from gpuwm.core.diagnostics import update_diagnostics
    update_diagnostics(state, run.hypsometric_opt)
    return state, coord


def _synthetic_land(run: RunConfig, terrain: np.ndarray):
    """Parent-grid static and Noah inventories with valid categories.

    Values are synthetic; their JOB in this test is to be a real,
    self-consistent land inventory the child's nest-down and physics
    initialisation must survive, not to be a real place.
    """
    ny, nx = run.ny, run.nx
    rng = np.random.default_rng(20240521)
    landmask = (rng.random((ny, nx)) > 0.25).astype(np.float32)
    lu_index = np.where(landmask >= 0.5,
                        rng.integers(1, 15, size=(ny, nx)),
                        17).astype(np.float32)      # 17 = ISWATER
    sct_dom = np.where(landmask >= 0.5,
                       rng.integers(1, 13, size=(ny, nx)),
                       14).astype(np.float32)       # 14 = water soil
    months = np.linspace(0.1, 0.8, 12)[:, None, None]
    static = {
        "HGT_M": terrain.astype(np.float32),
        "LANDMASK": landmask,
        "LU_INDEX": lu_index,
        "SCT_DOM": sct_dom,
        "SCB_DOM": sct_dom.copy(),
        "SNOALB": np.full((ny, nx), 0.55, np.float32),
        "SOILTEMP": np.full((ny, nx), 287.0, np.float32),
        "TMN": np.full((ny, nx), 287.0, np.float32),
        "LANDUSEF": np.zeros((21, ny, nx), np.float32),
        "SOILCTOP": np.zeros((16, ny, nx), np.float32),
        "SOILCBOT": np.zeros((16, ny, nx), np.float32),
        "GREENFRAC": np.broadcast_to(
            months, (12, ny, nx)).astype(np.float32).copy(),
        "LAI12M": np.broadcast_to(
            2.0 * months, (12, ny, nx)).astype(np.float32).copy(),
        "ALBEDO12M": np.full((12, ny, nx), 0.18, np.float32),
    }
    for k in range(21):
        static["LANDUSEF"][k] = (lu_index == (k + 1)).astype(np.float32)
    for k in range(16):
        static["SOILCTOP"][k] = (sct_dom == (k + 1)).astype(np.float32)
        static["SOILCBOT"][k] = static["SOILCTOP"][k]
    surface = {
        "TSK": np.full((ny, nx), 293.0, np.float32),
        "TSLB": np.full((4, ny, nx), 289.0, np.float32),
        "SMOIS": np.full((4, ny, nx), 0.28, np.float32),
        "SH2O": np.full((4, ny, nx), 0.26, np.float32),
        "TMN": np.full((ny, nx), 287.0, np.float32),
        "SEAICE": np.zeros((ny, nx), np.float32),
        "SNOW": np.zeros((ny, nx), np.float32),
        "SNOWH": np.zeros((ny, nx), np.float32),
        "LANDMASK": landmask,
        "XLAND": np.where(landmask >= 0.5, 1.0, 2.0).astype(np.float32),
    }
    identity = {"MMINLU": "MODIFIED_IGBP_MODIS_NOAH", "ISWATER": 17,
                "ISLAKE": 21, "ISICE": 15}
    return static, SimpleNamespace(fields=MappingProxyType(surface)), identity


# ---------------------------------------------------------------------------
# model assembly, exactly as the DA driver does it
# ---------------------------------------------------------------------------

def _wire(exp, *, attach_nest, geometry=None):
    """Hand-build the ExperimentState the DA driver builds, +/- the nest."""
    import cupy as cp

    from gpuwm.core.clock import build_schedule, resolve_clock
    from gpuwm.core.model import (DomainNode, ExperimentState,
                                  ModelRuntimeStatus)
    from gpuwm.core.physics import initialize_physics
    from gpuwm.static.lambert import grids_from_projection_config

    run = exp.root.run
    state, _coord = _build_parent_state(run)
    static, surface, identity = _synthetic_land(run, _terrain(run))
    grid = grids_from_projection_config(exp)[0]

    from gpuwm.core.landuse import initialize_landuse
    lat, lon = grid.latlon_mass()
    landuse = initialize_landuse(
        static["LU_INDEX"], soil_type=static["SCT_DOM"],
        landmask=static["LANDMASK"], snow=surface.fields["SNOW"],
        xice=surface.fields["SEAICE"], valid_time=exp.start_time,
        cen_lat=float(getattr(grid, "cen_lat", grid.ref_lat)),
        mminlu=identity["MMINLU"], iswater=identity["ISWATER"],
        islake=identity["ISLAKE"], isice=identity["ISICE"],
        fractional_seaice=True,
        soil_temperature=surface.fields["TSLB"])
    driver = initialize_physics(
        state, run, landuse=landuse, tsk=surface.fields["TSK"],
        soil_temperature=surface.fields["TSLB"],
        soil_moisture=surface.fields["SMOIS"],
        liquid_moisture=surface.fields["SH2O"],
        ivgtyp=static["LU_INDEX"], isltyp=static["SCT_DOM"],
        vegfra=100.0 * static["GREENFRAC"][4],
        tmn=surface.fields["TMN"], xice=surface.fields["SEAICE"],
        snow=surface.fields["SNOW"], snow_depth=surface.fields["SNOWH"],
        sst=surface.fields["TSK"],
        radiation_start_time=exp.start_time,
        radiation_latitude=lat, radiation_longitude=lon)

    nested_exp = exp
    child_dc = None
    if attach_nest:
        child_dc = nf.nest_domain_config(exp, geometry)
        nested_exp = nf.nested_experiment(exp, child_dc)

    tick = resolve_clock(nested_exp)
    schedule = build_schedule(nested_exp, tick)
    clocks = tick.clocks()
    root = DomainNode(exp.root, grid, state, clocks[1], None, [], None)
    nodes = {1: root}
    child_driver = None
    if attach_nest:
        child_node, child_driver, _receipt = nf.build_nested_child(
            root, child_dc, static=static, surface=surface,
            landuse_identity=identity, valid_time=exp.start_time,
            clock=clocks[child_dc.grid_id], parent_driver=driver)
        nodes[child_dc.grid_id] = child_node

    model = ExperimentState(root, MappingProxyType(nodes), schedule, None,
                            "nested-forecast-ratchet")
    model._runtime_status = ModelRuntimeStatus()
    model._resumed = False
    model._resume_committed_history_grid_ids = frozenset()
    # The shared arena and the shared dycore workspace are what would let
    # two domains write the same bytes; the DA driver disables both and so
    # does this path.  Asserted, not assumed.
    model._scratch_arena = None
    model._dycore_state_workspace = None
    model._io_manager = None
    model._last_checkpoint = None
    model._prepared_by_grid_id = MappingProxyType({})
    return model, root, nodes, driver, child_driver


def _run(model):
    import cupy as cp

    from gpuwm.core.model import execute_experiment

    execute_experiment(model, history_handler=None, progress_callback=None,
                       validate_state=True, skip_feedback_path=True,
                       pool_trim_per_period=True)
    cp.cuda.Stream.null.synchronize()


def _geometry():
    return nf.NestGeometry(ratio=RATIO, nx=CHILD_NX, ny=CHILD_NY)


# ---------------------------------------------------------------------------
# the ratchet
# ---------------------------------------------------------------------------

@requires_gpu
@pytest.mark.gpu
def test_parent_forecast_is_bitwise_unchanged_by_the_presence_of_the_nest():
    """RATCHET: attaching the fine nest must not move the parent by a bit.

    This is the DA-route sibling of ``nest_gates`` N3/N4/N5
    ``ancestor_inertness``: the parent is the trajectory the ensemble was
    scored on, and a nest that perturbs it would invalidate every
    verification number the nowcast reports.
    """
    from gpuwm.ensemble.state_sha import live_state_sha256

    exp = _experiment(_parent_run())

    control, control_root, _n, _d, _cd = _wire(exp, attach_nest=False)
    _run(control)
    alone = live_state_sha256(control_root.state)
    del control, control_root

    nested, nested_root, nodes, _d2, _cd2 = _wire(
        exp, attach_nest=True, geometry=_geometry())
    assert set(nodes) == {1, 2}
    _run(nested)
    with_child = live_state_sha256(nested_root.state)

    assert with_child == alone, (
        "the parent's forecast changed when the nest was attached; "
        "one-way nesting is broken on the DA nested-leg path")


@requires_gpu
@pytest.mark.gpu
def test_the_nest_actually_integrated_so_the_ratchet_is_not_vacuous():
    """The inertness proof is worthless if the child never stepped."""
    exp = _experiment(_parent_run())
    model, root, nodes, _driver, _child_driver = _wire(
        exp, attach_nest=True, geometry=_geometry())
    child = nodes[2]
    before = child.state.thp.copy()

    assert child.coupler is not None
    assert child.coupler.feedback == 0
    _run(model)

    # Ratio 3 in time: the child takes three steps per parent step.
    assert child.clock.step_count == root.clock.step_count * RATIO
    assert root.clock.step_count == int(RUN_SECONDS / PARENT_DT)
    assert not bool((child.state.thp == before).all()), \
        "the child domain did not evolve; the ratchet above proves nothing"


@requires_gpu
@pytest.mark.gpu
def test_building_the_child_does_not_touch_the_parent_state():
    """``parent_only_init`` reads the parent; it must not write it."""
    from gpuwm.ensemble.state_sha import live_state_sha256

    exp = _experiment(_parent_run())
    child_dc = nf.nest_domain_config(exp, _geometry())
    nested_exp = nf.nested_experiment(exp, child_dc)

    from gpuwm.core.clock import resolve_clock
    from gpuwm.core.model import DomainNode
    from gpuwm.static.lambert import grids_from_projection_config

    run = exp.root.run
    state, _coord = _build_parent_state(run)
    static, surface, identity = _synthetic_land(run, _terrain(run))
    grid = grids_from_projection_config(exp)[0]
    clocks = resolve_clock(nested_exp).clocks()
    root = DomainNode(exp.root, grid, state, clocks[1], None, [], None)

    from gpuwm.core.landuse import initialize_landuse
    from gpuwm.core.physics import initialize_physics
    lat, lon = grid.latlon_mass()
    driver = initialize_physics(
        state, run,
        landuse=initialize_landuse(
            static["LU_INDEX"], soil_type=static["SCT_DOM"],
            landmask=static["LANDMASK"], snow=surface.fields["SNOW"],
            xice=surface.fields["SEAICE"], valid_time=exp.start_time,
            cen_lat=float(getattr(grid, "cen_lat", grid.ref_lat)),
            mminlu=identity["MMINLU"], iswater=identity["ISWATER"],
            islake=identity["ISLAKE"], isice=identity["ISICE"],
            fractional_seaice=True,
            soil_temperature=surface.fields["TSLB"]),
        tsk=surface.fields["TSK"],
        soil_temperature=surface.fields["TSLB"],
        soil_moisture=surface.fields["SMOIS"],
        liquid_moisture=surface.fields["SH2O"],
        ivgtyp=static["LU_INDEX"], isltyp=static["SCT_DOM"],
        vegfra=100.0 * static["GREENFRAC"][4], tmn=surface.fields["TMN"],
        xice=surface.fields["SEAICE"], snow=surface.fields["SNOW"],
        snow_depth=surface.fields["SNOWH"], sst=surface.fields["TSK"],
        radiation_start_time=exp.start_time,
        radiation_latitude=lat, radiation_longitude=lon)

    before = live_state_sha256(state)
    nf.build_nested_child(
        root, child_dc, static=static, surface=surface,
        landuse_identity=identity, valid_time=exp.start_time,
        clock=clocks[child_dc.grid_id], parent_driver=driver)
    assert live_state_sha256(state) == before


@requires_gpu
@pytest.mark.gpu
def test_the_child_inherits_the_parents_analysed_state_not_an_analysis_file():
    """The whole point: the nest starts from the parent, interpolated.

    A child cell whose donor column is interior (away from the SINT
    stencil's edge behaviour) must carry the parent's value at that
    column, because SINT reproduces its donor exactly at the sub-cell
    position the donor sits at.  The assertion here is weaker and safer:
    the child's field statistics must match the parent's over the covered
    footprint, which a cold start from an unrelated analysis could not do.
    """
    exp = _experiment(_parent_run())
    model, root, nodes, _driver, _cd = _wire(
        exp, attach_nest=True, geometry=_geometry())
    child_dc = nodes[2].cfg
    parent_state, child_state = root.state, nodes[2].state

    i0 = child_dc.i_parent_start - 1
    j0 = child_dc.j_parent_start - 1
    span = CHILD_NX // RATIO
    covered = parent_state.thp[:, j0:j0 + span, i0:i0 + span].get()
    child = child_state.thp.get()

    assert np.isfinite(child).all()
    # Same air mass, same thermal structure: means agree closely and the
    # ranges overlap.  A GFS cold start over the same box would not.
    assert child.mean() == pytest.approx(covered.mean(), abs=0.05)
    assert child.min() >= covered.min() - 0.5
    assert child.max() <= covered.max() + 0.5


@requires_gpu
@pytest.mark.gpu
def test_the_nested_model_never_shares_a_scratch_arena_between_domains():
    """The shared arena is what would silently corrupt across domains."""
    exp = _experiment(_parent_run())
    model, root, nodes, _driver, _cd = _wire(
        exp, attach_nest=True, geometry=_geometry())
    assert model._scratch_arena is None
    assert model._dycore_state_workspace is None
    parent_scratch = root.state._scratch
    child_scratch = nodes[2].state._scratch
    assert parent_scratch is not child_scratch
    _run(model)
    for name, buffer in parent_scratch.items():
        other = child_scratch.get(name)
        if other is None:
            continue
        assert buffer.data.ptr != other.data.ptr, (
            f"parent and child share the scratch slot {name!r}")


@requires_gpu
@pytest.mark.gpu
def test_the_child_survives_a_leg_boundary_bit_for_bit():
    """The nest's fine structure must cross a leg boundary intact.

    The driver runs consecutive free legs by rebuilding every model from
    the prepared cache and restoring host snapshots.  For the parent that
    is the shipped handoff; the child gets the same one, and it has to be
    lossless -- rebuilding the nest from the parent at every leg boundary
    would flatten away exactly the fine-scale structure the nest exists
    to produce.
    """
    from gpuwm.ensemble.state_sha import live_state_sha256
    from gpuwm.state_serialization_contract import STATE_SERIALIZED_ATTRS

    import cupy as cp

    exp = _experiment(_parent_run())
    first, _root, nodes, _driver, _cd = _wire(
        exp, attach_nest=True, geometry=_geometry())
    _run(first)
    evolved = nodes[2].state
    handoff = {}
    for field in STATE_SERIALIZED_ATTRS:
        value = getattr(evolved, field, None)
        if value is not None:
            handoff[field] = value.get()
    assert handoff, "the child carried no serialised state"
    expected = live_state_sha256(evolved)

    # A second leg: a brand-new model, a brand-new child SINT'd from the
    # parent, then the host snapshot restored over it.
    second, _root2, nodes2, _d2, _cd2 = _wire(
        exp, attach_nest=True, geometry=_geometry())
    fresh = nodes2[2].state
    assert live_state_sha256(fresh) != expected
    for field, host in handoff.items():
        getattr(fresh, field)[...] = cp.asarray(
            host, dtype=getattr(fresh, field).dtype)
    assert live_state_sha256(fresh) == expected
