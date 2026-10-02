"""A saved-run child's own geography reaches its terrain-drag initializer."""
from datetime import datetime
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from gpuwm import offline_child_geography as geography
from gpuwm import offline_child_run
from gpuwm.config import RunConfig
from gpuwm.offline_child import OfflineChildContractError
from gpuwm.static.build import GeogSelection
from gpuwm.static.orographic import required_static_fields
from gpuwm.static.rust_bridge import OROGRAPHIC_MARKER


START = datetime(2020, 1, 1)
OPTIONS = [(0, 0), (1, 0), (2, 0), (0, 1), (0, 3), (1, 3)]


def _cfg(topo=0, gwd=0):
    return RunConfig(nx=2, ny=2, nz=8, dx=3000.0, dy=3000.0,
                     ztop=15000.0, dt=12.0, run_seconds=3600.0,
                     bl_pbl_physics=1, topo_wind=topo, gwd_opt=gwd)


@pytest.mark.parametrize("topo,gwd", OPTIONS)
def test_child_geography_requests_and_retains_its_drag_fields(
        tmp_path, monkeypatch, topo, gwd):
    from gpuwm.static import build

    cfg = _cfg(topo, gwd)
    grid = SimpleNamespace(e_we=3, e_sn=3, dx=3000.0, dy=3000.0,
                           ref_lat=35.0, ref_lon=-90.0, map_proj="lambert")
    monkeypatch.setattr(geography, "child_projected_grid",
                        lambda *args: (grid, {"verified": True}))
    monkeypatch.setattr(geography, "require_geog_tree", lambda *args, **kw: None)
    monkeypatch.setattr(GeogSelection, "landuse_global_attrs", lambda self: {})
    required = required_static_fields(topo, gwd)
    seen = {}

    def build_static(actual_grid, root, *, selection, timing_report):
        seen["selection"] = selection
        assert set(selection.orographic) == set(required)
        fields = {name: np.full((2, 2), 10.0) for name in
                  ("HGT_M", "LU_INDEX", "LANDMASK", "SCT_DOM", "TMN")}
        fields["GREENFRAC"] = np.ones((12, 2, 2))
        fields.update({name: np.full((2, 2), index + 31.0)
                       for index, name in enumerate(required)})
        return fields

    monkeypatch.setattr(build, "build_static", build_static)
    result = geography.build_child_geography(
        tmp_path / "parent.nc", None, cfg,
        geography.ChildStaticPolicy(geog_root=tmp_path), valid_time=START)
    for index, name in enumerate(required):
        assert np.array_equal(result.fields[name], np.full((2, 2), index + 31.0))
        assert not result.fields[name].flags.writeable
    if required:
        assert result.receipt["orographic_sampling_contract"] == OROGRAPHIC_MARKER
        assert set(result.receipt["orographic_fields"]) == set(required)
    else:
        assert "orographic_sampling_contract" not in result.receipt


@pytest.mark.parametrize("topo,gwd", OPTIONS)
def test_child_physics_receives_the_child_statics(monkeypatch, topo, gwd):
    from gpuwm.core import physics, landuse

    cfg = _cfg(topo, gwd)
    statics = {name: np.full((2, 2), index + 31.0) for index, name in
               enumerate(required_static_fields(topo, gwd))}
    seen = {}

    def initialize(child, actual_cfg, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(fields={})

    monkeypatch.setattr(physics, "initialize_physics", initialize)
    monkeypatch.setattr(landuse, "initialize_landuse", lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace())
    fields = {name: np.ones((2, 2)) for name in
              ("XLAT", "XLONG", "LU_INDEX", "ISLTYP", "LANDMASK", "SNOW",
               "TSK", "VEGFRA", "TMN")}
    fields.update(TSLB=np.ones((4, 2, 2)), SMOIS=np.ones((4, 2, 2)))
    surface = SimpleNamespace(fields=fields, identity={
        "MMINLU": "MODIFIED_IGBP_MODIS_NOAH", "ISWATER": 17, "ISLAKE": 21,
        "ISICE": 15, "ISOILWATER": 14})
    initial = SimpleNamespace(fields=fields, receipt={"p_top": 5000.0})
    offline_child_run._initialize_child_physics(
        None, cfg, initial, surface, START, terrain_drag_static=statics)
    if statics:
        assert seen["terrain_drag_static"] is statics
    else:
        assert "terrain_drag_static" not in seen


def test_missing_child_statistics_refuse_before_physics_allocation(monkeypatch):
    from gpuwm.core import physics

    monkeypatch.setattr(physics, "initialize_physics", lambda *args, **kw:
                        pytest.fail("physics was allocated before static admission"))
    with pytest.raises(OfflineChildContractError, match="VAR_SSO"):
        offline_child_run._initialize_child_physics(
            None, _cfg(1), None, None, START)


def test_parent_terrain_cannot_supply_new_drag_statistics(tmp_path):
    config = tmp_path / "child.toml"
    config.write_text('[static]\nterrain = "parent"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="parent-terrain interpolation does not supply"):
        geography.load_child_static_policy(config, _cfg(1))
    assert not geography.load_child_static_policy(config, _cfg()).own


def test_missing_drag_dataset_is_named_at_child_admission(tmp_path, monkeypatch):
    selection = GeogSelection.fallback(tmp_path)
    for field in ("terrain", "landuse", "soil_top", "soil_bottom", "greenfrac",
                  "lai", "albedo", "snow_albedo", "soil_temperature"):
        path = selection.path(field)
        path.mkdir(parents=True, exist_ok=True)
        (path / "index").write_text("", encoding="utf-8")
    with pytest.raises(geography.ChildGeographyError, match="VAR_SSO.*varsso_10m"):
        geography.require_geog_tree(
            geography.ChildStaticPolicy(geog_root=tmp_path), cfg=_cfg(1))
    geography.require_geog_tree(geography.ChildStaticPolicy(geog_root=tmp_path),
                                cfg=_cfg())
