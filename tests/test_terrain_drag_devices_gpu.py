"""Exact prepared geography and ranked forecast checks with active GSL drag."""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from tests.conftest import requires_gpu

pytestmark = pytest.mark.gpu


def _static_fields(cfg, terrain):
    from gpuwm.static.orographic import required_static_fields
    from tests.test_terrain_drag_receipts import _load

    oracle = _load("gwdo/inputs")
    columns = np.arange(cfg.ny * cfg.nx).reshape(cfg.ny, cfg.nx) % len(oracle["var2d"])
    source = {"VAR": oracle["var2d"], "CON": oracle["oc12d"],
              "VARLS": oracle["var2d"], "CONLS": oracle["oc12d"],
              "VARSS": oracle["var2dss"], "CONSS": oracle["oc12dss"],
              "VAR_SSO": oracle["var2d"]}
    for prefix, key in (("OA", "oa"), ("OL", "ol")):
        for direction in range(4):
            source[f"{prefix}{direction + 1}"] = oracle[key][:, direction]
            source[f"{prefix}{direction + 1}LS"] = oracle[key][:, direction]
            source[f"{prefix}{direction + 1}SS"] = oracle[key + "ss"][:, direction]
    return {"HGT_M": terrain, **{
        name: np.ascontiguousarray(source[name][columns], dtype=np.float32)
        for name in required_static_fields(cfg.topo_wind, cfg.gwd_opt)}}


def _install_static_fixture(monkeypatch):
    """Supply physical static inputs to the gate's full-domain fixture only."""
    from tilestream import harness

    original = harness.make_physics_state

    def build(cfg, *args, geography=None, **kwargs):
        if "terrain_drag_static" not in kwargs:
            kwargs["terrain_drag_static"] = _static_fields(cfg, geography.terrain)
        return original(cfg, *args, geography=geography, **kwargs)

    monkeypatch.setattr(harness, "make_physics_state", build)


@requires_gpu
@pytest.mark.parametrize("topo", [0, 1, 2])
def test_prepared_drag_geography_gathers_exact_rank_halos(topo):
    import cupy as cp

    from gpuwm.core import streaming
    from gpuwm.core.devices import DeviceOptions
    from tilestream import driver, gather, harness
    from tilestream.ranks_gate import config

    cfg = replace(config(96, 80, 20), dx=3000.0, dy=3000.0,
                  topo_wind=topo, gwd_opt=3, ra_physics=0, mp_physics=0)
    geo = harness.make_geography(cfg, terrain=True, periodic_faces=False)
    state, _ = harness.make_physics_state(
        cfg, geography=geo, terrain_drag_static=_static_fields(cfg, geo.terrain))
    geography = driver.geography_store(state, host=True)
    names = {name for name in geography if name.startswith("terrain_drag/")}
    assert len(names) == 20 + 2 * bool(topo)
    options = DeviceOptions(count=4, grid=(2, 2), ids=(0, 0, 0, 0))
    windows = streaming.ranked_specs(cfg, options, halo=9)
    factory = streaming.prepared_tile_state_factory(state, cfg)
    for window in windows:
        tile = factory(harness.tile_config(cfg, window.cnx, window.cny))
        driver._pin_scheme_geography(tile)
        gather.gather_tile(geography, tile, window,
                           inventory_fn=driver.geography_inventory, nz=cfg.nz)
        cp.cuda.runtime.deviceSynchronize()
        actual = driver.geography_inventory(tile)
        for name in names:
            expected = geography[name][window.cj0:window.cj0 + window.cny,
                                       window.ci0:window.ci0 + window.cnx]
            assert cp.asnumpy(actual[name]).tobytes() == expected.tobytes(), name
        assert tile.physics.terrain_drag.kpblmax == state.physics.terrain_drag.kpblmax
        driver.assert_geography_gathered(tile, keys=geography)
        del tile


@requires_gpu
@pytest.mark.parametrize("slab_rows", [1, 3, 5])
def test_slab_topo_coefficients_keep_true_domain_neighbours(slab_rows):
    import cupy as cp

    from gpuwm.core.terrain_drag import topo_wind_coefficients
    from gpuwm.ingest.prepared_store import _complete_slab_topo_wind
    from tests.test_terrain_drag_receipts import _load

    inputs = _load("topo_static/inputs")
    ht = np.ascontiguousarray(inputs["ht"].T)
    xland = np.ascontiguousarray(inputs["xland"].T)
    var = np.ascontiguousarray(inputs["var_sso"].T)
    want = _load("topo_static/topo_wind_1")
    ny, nx = ht.shape
    naive_differs = False
    for start in range(0, ny, slab_rows):
        end = min(start + slab_rows, ny)
        raw = topo_wind_coefficients(
            ht[start:end], xland[start:end], topo_wind=1, var_sso=var[start:end])
        naive_differs |= cp.asnumpy(raw[0]).tobytes() != want["ctopo"].T[start:end].tobytes()
        drag = SimpleNamespace(topo_wind=1,
                               ctopo=cp.full((end - start, nx), np.nan, cp.float32),
                               ctopo2=cp.full((end - start, nx), np.nan, cp.float32))
        state = SimpleNamespace(ht=cp.asarray(ht[start:end]),
                                physics=SimpleNamespace(terrain_drag=drag,
                                    fields={"xland": cp.asarray(xland[start:end])}))
        _complete_slab_topo_wind(state, SimpleNamespace(topo_wind=1, nx=nx),
                                 {"VAR_SSO": var}, SimpleNamespace(terrain_z=ht),
                                 start, end - start, ny)
        for name in ("ctopo", "ctopo2"):
            actual = cp.asnumpy(getattr(drag, name))
            assert actual.tobytes() == want[name].T[start:end].tobytes(), (start, name)
    assert naive_differs, "the test must expose false terrain clamping at a slab edge"


@requires_gpu
@pytest.mark.parametrize("topo,pbl", [(0, 5), (1, 1), (2, 1)])
def test_active_gsl_ranked_histories_are_byte_identical(monkeypatch, topo, pbl):
    import cupy as cp

    from gpuwm.core.terrain_drag import TerrainDrag
    from tilestream.ranks_gate import config, integrate

    _install_static_fixture(monkeypatch)
    cfg = replace(config(128, 104, 20, rung="mynn" if pbl == 5 else "lean"),
                  dx=3000.0, dy=3000.0, topo_wind=topo, gwd_opt=3)
    # Require actual momentum changes, so identity cannot pass on inactive drag.
    active_cards = set()
    original = TerrainDrag.apply_gwd

    def apply(drag, atmosphere, du, dv, **kwargs):
        before_u, before_v = du.copy(), dv.copy()
        result = original(drag, atmosphere, du, dv, **kwargs)
        if bool(cp.any(du != before_u)) or bool(cp.any(dv != before_v)):
            active_cards.add(cp.cuda.Device().id)
        return result

    monkeypatch.setattr(TerrainDrag, "apply_gwd", apply)
    count = min(4, cp.cuda.runtime.getDeviceCount())
    cards = list(range(count))
    result = integrate(cfg, grid=(2, 2), devices=cards, mode="threads",
                       nsteps=3, wind=20.0, change_live=False)
    assert result["differing"] == []
    assert result["resident_digest"] == result["ranked_digest"]
    assert active_cards == set(cards), "GSL drag must change momentum on every card"


@requires_gpu
def test_missing_gsl_static_halo_changes_owned_forecast_cells(monkeypatch):
    import cupy as cp

    from tilestream import ranks_gate

    _install_static_fixture(monkeypatch)
    original = ranks_gate.make_ranked
    poisoned = []

    def make(*args, **kwargs):
        streamed = original(*args, **kwargs)
        run = streamed.tiled_run
        window = run.specs[0]
        assert window.halo_right > 0
        seam = window.i1 - window.ci0
        with cp.cuda.Device(run.devices[0]):
            values = run.tiles[0].physics.terrain_drag.gwd["VARSS"]
            before = values.copy()
            values[:, seam:seam + 3] = 0
            assert bool(cp.any(before[:, seam:seam + 3] != 0))
            assert cp.asnumpy(values[:, :seam]).tobytes() == cp.asnumpy(before[:, :seam]).tobytes()
            cp.cuda.runtime.deviceSynchronize()
        poisoned.append(True)
        return streamed

    monkeypatch.setattr(ranks_gate, "make_ranked", make)
    cfg = replace(ranks_gate.config(128, 104, 20, rung="mynn"),
                  dx=3000.0, dy=3000.0, gwd_opt=3)
    # Only rank 0's halo changed. A different owned result proves the halo
    # statics matter to the integrated forecast, not only to its inventory.
    with pytest.raises(AssertionError, match="ranked carriers, scalars or canonical digest differ"):
        ranks_gate.integrate(cfg, grid=(1, 2), devices=[0], mode="threads",
                             nsteps=3, wind=20.0, change_live=False, timing=True)
    assert poisoned == [True]
