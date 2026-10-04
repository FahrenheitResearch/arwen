"""Device gathers preserve terrain-drag columns and slab-edge coefficients."""
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.core.terrain_drag import build_terrain_drag, topo_wind_coefficients
from conftest import requires_gpu
from test_terrain_drag_receipts import _load
from tilestream import gather
from tilestream.driver import geography_inventory, assert_geography_gathered
from tilestream.spec import plan_tiles

pytestmark = [pytest.mark.gpu, requires_gpu]


def _host_bits(actual, expected):
    import cupy as cp

    got = cp.asnumpy(actual) if isinstance(actual, cp.ndarray) else actual
    want = cp.asnumpy(expected) if isinstance(expected, cp.ndarray) else expected
    np.testing.assert_array_equal(np.asarray(got).view(np.uint32),
                                  np.asarray(want).view(np.uint32))


@pytest.mark.parametrize("tiles", [2, 4, 8])
@pytest.mark.parametrize("dx", [3000.0, 9000.0])
def test_gathered_gsl_columns_are_resident_bits(tiles, dx):
    import cupy as cp

    inputs = _load("gwdo/inputs")
    shape = (8, 8)
    atmosphere = {
        name: cp.asarray(np.ascontiguousarray(inputs[key].T.reshape((-1, *shape))))
        for name, key in (("u", "u3d"), ("v", "v3d"), ("temperature", "t3d"),
                          ("qv", "qv3d"), ("pressure", "p3d"), ("exner", "pi3d"),
                          ("p_interface", "p3di"), ("z", "z"), ("dz", "dz"))}
    static = {}
    for suffix, source_suffix in (("LS", ""), ("SS", "ss")):
        static[f"VAR{suffix}"] = inputs[f"var2d{source_suffix}"].reshape(shape)
        static[f"CON{suffix}"] = inputs[f"oc12d{source_suffix}"].reshape(shape)
        for family in ("OA", "OL"):
            values = inputs[f"{family.lower()}{source_suffix}"]
            for m in range(4):
                static[f"{family}{m + 1}{suffix}"] = values[:, m].reshape(shape)
    fields = {name: cp.asarray(inputs[name].reshape(shape)) for name in
              ("sina", "cosa", "xland", "br", "pblh", "kpbl")}
    resident = build_terrain_drag(
        topo_wind=0, gwd_opt=3, static=static, ht=cp.zeros(shape, cp.float32),
        xland=fields["xland"], znu=inputs["znu"])
    parent = SimpleNamespace(physics=SimpleNamespace(terrain_drag=resident))
    home = geography_inventory(parent)
    original = {name: cp.asarray(np.ascontiguousarray(
                    inputs[key].T.reshape((-1, *shape))))
                for name, key in (("du", "rublten0"), ("dv", "rvblten0"))}
    expected = {name: value.copy() for name, value in original.items()}
    dt = float(_load("gwdo_gsl/dx3000")["dt"])
    resident.apply_gwd(atmosphere, expected["du"], expected["dv"],
                       **fields, dx=dx, dt=dt)
    joined = {name: np.empty_like(cp.asnumpy(value))
              for name, value in expected.items()}
    nz = atmosphere["u"].shape[0]
    for spec in plan_tiles(8, 8, 8, 8 // tiles, halo=0, periodic=False):
        tile_shape = (spec.cny, spec.cnx)
        neutral = np.zeros(tile_shape, np.float32)
        drag = build_terrain_drag(
            topo_wind=0, gwd_opt=3,
            static={name: neutral for name in static}, ht=cp.asarray(neutral),
            xland=cp.ones(tile_shape, cp.float32), znu=inputs["znu"])
        tile = SimpleNamespace(p=cp.empty((nz, *tile_shape), cp.float32),
                               physics=SimpleNamespace(terrain_drag=drag))
        assert drag.kpblmax == resident.kpblmax
        assert_geography_gathered(tile, keys=home)
        gather.gather_tile(home, tile, spec, inventory_fn=geography_inventory,
                           nz=nz)
        for name, array in drag.geography().items():
            _host_bits(array, home[name][spec.cj0:spec.cj0 + spec.cny, :])
        window = (..., slice(spec.cj0, spec.cj0 + spec.cny), slice(None))
        tile_atm = {name: cp.ascontiguousarray(value[window])
                    for name, value in atmosphere.items()}
        tile_fields = {name: cp.ascontiguousarray(value[window])
                       for name, value in fields.items()}
        rates = {name: value[window].copy() for name, value in original.items()}
        drag.apply_gwd(tile_atm, rates["du"], rates["dv"],
                       **tile_fields, dx=dx, dt=dt)
        for name, value in rates.items():
            joined[name][window] = cp.asnumpy(value)
    for name, value in expected.items():
        _host_bits(joined[name], value)


def test_prepared_topo_slabs_keep_resident_coefficients_at_the_seams():
    import cupy as cp

    from gpuwm.ingest.prepared_store import _complete_slab_topo_wind

    ny, nx = 9, 7
    rng = np.random.default_rng(17)
    height = rng.uniform(50.0, 150.0, (ny, nx))
    variance = np.full((ny, nx), 2500.0, np.float32)
    land = np.ones((ny, nx), np.float32)
    land[:, 1::3] = 2.0
    expected = topo_wind_coefficients(
        cp.asarray(height, dtype=cp.float32), cp.asarray(land),
        topo_wind=1, var_sso=variance)
    negative_control = 0
    for start in (0, 3, 6):
        window = slice(start, start + 3)
        drag = build_terrain_drag(
            topo_wind=1, gwd_opt=0, static={"VAR_SSO": variance[window]},
            ht=cp.asarray(height[window], dtype=cp.float32),
            xland=cp.asarray(land[window]), znu=np.array([0.9, 0.5, 0.1]))
        negative_control += np.count_nonzero(
            cp.asnumpy(drag.ctopo) != cp.asnumpy(expected[0][window]))
        state = SimpleNamespace(physics=SimpleNamespace(
            terrain_drag=drag, fields={"xland": cp.asarray(land[window])}))
        _complete_slab_topo_wind(
            state, SimpleNamespace(topo_wind=1, nx=nx), {"VAR_SSO": variance},
            SimpleNamespace(terrain_z=height), start, 3, ny)
        _host_bits(drag.ctopo, expected[0][window])
        _host_bits(drag.ctopo2, expected[1][window])
    assert negative_control > 0, "local slab edges must expose the old clamp"


@pytest.mark.parametrize("topo,gwd", [(0, 3), (1, 0)])
def test_prepared_drag_buffers_step_with_resident_bits(topo, gwd):
    import cupy as cp

    from gpuwm.core.streaming import prepared_tile_state_factory
    from gpuwm.static.orographic import required_static_fields
    from tilestream import driver, harness, physics_inventory

    cfg = harness.make_config(
        64, 64, 12, dt=1.0, dx=3000.0, dy=3000.0, time_step_sound=4,
        terrain_opt=1, map_proj=1, moist=True, mp_physics=6,
        ra_physics=0, cu_physics=0, bl_pbl_physics=1,
        sf_sfclay_physics=1, sf_surface_physics=0, topo_wind=topo, gwd_opt=gwd)
    geo = harness.make_geography(cfg, terrain=True)
    static = {"HGT_M": geo.terrain}
    for name in required_static_fields(topo, gwd):
        value = (2500.0 if name == "VAR_SSO" else 100.0 if name.startswith("VAR")
                 else 0.5 if name.startswith("CON") else 0.4 if name.startswith("OA")
                 else 0.1)
        static[name] = np.full((cfg.ny, cfg.nx), value, np.float32)
    state, _ = harness.make_physics_state(
        cfg, 4242, geography=geo, terrain_drag_static=static)
    factory = prepared_tile_state_factory(state, cfg)
    inventory = physics_inventory.carrier_inventory
    store = {name: gather.pinned_copy(cp.asnumpy(value))
             for name, value in inventory(state).items()}
    kwargs = driver.geography_run_kwargs(cfg, state, host=True, warmup=0)
    kwargs["tile_state_factory"] = factory
    run = driver.TiledRun(store, cfg, 32, 32, harness.halo_radius(cfg), 2,
                          periodic=True, **kwargs)
    for _ in range(2):
        harness.run_steps(state, cfg, 1)
        run.sweep(1)
        cp.cuda.runtime.deviceSynchronize()
        expected = inventory(state)
        assert set(store) == set(expected)
        for name, values in expected.items():
            assert np.isfinite(store[name]).all(), name
            assert store[name].tobytes() == cp.asnumpy(values).tobytes(), name
