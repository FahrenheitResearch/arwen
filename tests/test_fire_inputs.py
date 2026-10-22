"""Admission defects: canopy scale, partial structures and regional licence gates."""
import json
from pathlib import Path
import pytest
from gpuwm.fire_inputs import canopy_fields, prepare_sources, region_for_bounds
from gpuwm.static.sfire import source_routes, _wcs_query

@pytest.mark.parametrize("bounds,region", [((-105.3,39.8,-105.1,40.1),"conus"),((-156.8,20.8,-156.5,21.),"hi"),((-151.,62.,-150.,63.),"ak"),((20.,38.,21.,39.),"global")])
def test_region_selection_is_geographic(bounds,region):
    assert region_for_bounds(bounds)["id"] == region

@pytest.mark.parametrize("region,suffix", [("conus",""),("hi","-hi"),("ak","-ak")])
def test_each_region_has_both_fuel_models_and_encoded_canopy_units(region,suffix):
    routes=source_routes()
    for model in (13,40):
        assert routes[f"landfire-fbfm{model}{suffix}"]["role"] == "fuel"
    for name,row in canopy_fields().items():
        route=routes[f"landfire-{row['product'].lower()}{suffix}"]
        assert route["role"]==row["role"]
        assert route["scale_factor"]==row["scale_factor"]
        assert route["nodata_override"]==-9999
        query=_wcs_query(route,2024,(0.,0.,30.,30.),30.,route["crs"])
        assert row["product"] in query

def test_public_acquisition_admits_complete_canopy_by_default(tmp_path):
    seen=[]
    def acquire(source,year,bounds,cache):
        seen.append(source)
        return dict(path="opaque.tif",sha256="a"*64),dict(source=source)
    inputs,receipt,stages=prepare_sources({},(-105.3,39.8,-105.1,40.1),tmp_path,tmp_path,acquire,source_routes())
    assert set(inputs)=={"terrain","fuel",*canopy_fields()}
    assert len(seen)==6 and stages["acquisition_workers"]==6

def test_legacy_bound_rasters_need_no_added_canopy_or_network(tmp_path):
    bound=dict(path="captured.tif",sha256="a"*64)
    def refused(*args):raise AssertionError("unexpected network")
    inputs,_,_=prepare_sources(dict(terrain=bound,fuel=bound),(-105.3,39.8,-105.1,40.1),tmp_path,tmp_path,refused,source_routes())
    assert set(inputs)=={"terrain","fuel"}
    assert Path(inputs["fuel"]["path"]).is_absolute()

def test_partial_bound_canopy_is_refused_before_fetch(tmp_path):
    config=dict(fuel=dict(path="fuel.tif",sha256="a"*64),canopy={"CANOPY_HEIGHT":dict(path="height.tif",sha256="b"*64)})
    with pytest.raises(ValueError,match="together"):
        prepare_sources(config,(-105.3,39.8,-105.1,40.1),tmp_path,tmp_path,None,source_routes())

def test_global_dem_named_licence_gate_precedes_network(tmp_path):
    config=dict(fuel=dict(path="fuel.tif",sha256="a"*64))
    with pytest.raises(ValueError,match="accept_licenses"):
        prepare_sources(config,(20.,38.,21.,39.),tmp_path,tmp_path,None,source_routes())
    licence=source_routes()["copernicus-dem-glo30"]["license"]
    config["sources"]=dict(accept_licenses=[licence])
    calls=[]
    def acquire(source,*args):
        calls.append(source);return dict(path="dem.tif",sha256="c"*64),{}
    inputs,_,_=prepare_sources(config,(20.,38.,21.,39.),tmp_path,tmp_path,acquire,source_routes())
    assert calls==["copernicus-dem-glo30"] and set(inputs)=={"fuel","terrain"}

def test_unmapped_global_fuel_is_named_gap(tmp_path):
    with pytest.raises(ValueError,match="no admitted fuel route"):
        prepare_sources({},(20.,38.,21.,39.),tmp_path,tmp_path,None,source_routes())


def test_sealed_provenance_keeps_notices_without_cache_or_timing_identity():
    from gpuwm.fire_inputs import source_provenance
    receipt = {"terrain": {"license": "named-licence", "attribution": "source notice",
        "wall_seconds": 12, "objects": [{"sha256": "a"*64, "cache_hit": False, "fetched_utc": "now"}]}}
    expected = {"terrain": {"license": "named-licence", "attribution": "source notice",
        "objects": [{"sha256": "a"*64}]}}
    assert source_provenance(receipt) == expected
    assert receipt["terrain"]["wall_seconds"] == 12
