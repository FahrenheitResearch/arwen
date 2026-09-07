"""Explicit footprints use operation bounds, without borrowing point presets."""
from dataclasses import replace
from datetime import datetime
import json
import math

import numpy as np
import pytest

from gpuwm import domain_wizard as dw
from gpuwm.core.nest_interp import register_nest
from gpuwm.ingest.lateral_bc import _field_boundary, _validate_frame_domain
from gpuwm.grid_requirements import boundary_axis, NATIVE_TARGET_INTERIOR_AXIS
from gpuwm.source_adapters import get_source_adapter
from tilestream.spec import plan_tiles


def footprint(tmp_path, lat=35.3, lon=-97.5, width=180, height=150):
    dy=height/(2*111.32); dx=width/(2*111.32*math.cos(math.radians(lat)))
    wrap=lambda value: (value+180)%360-180
    ring=[[wrap(lon-dx),lat-dy],[wrap(lon+dx),lat-dy],
          [wrap(lon+dx),lat+dy],[wrap(lon-dx),lat+dy],[wrap(lon-dx),lat-dy]]
    path=tmp_path/"region.geojson"
    path.write_text(json.dumps({"type":"Polygon","coordinates":[ring]}))
    fp=dw.load_polygon_footprint(path)
    return fp,dw._projection_entries(fp.center_lat,fp.center_lon)


def experiment(fp,projection,dims,ratios=()):
    text=dw.render_config(name="operation_bounds",start_time=datetime(2026,9,5),
        hours=3,projection=projection,dims=dims,ratios=ratios,fetch_hints={},
        case_data=None,root_dx_m=12000.,tiles="off",
        history_interval_s=720.,nest_history_interval_s=180.)
    exp=dw.experiment_from_text(text,source="<operation bounds>")
    dw.verify_polygon_containment(exp,fp,(0.,)*(1+len(ratios)))
    return exp


@pytest.mark.parametrize("lat,lon",[(35.3,-97.5),(35.6895,139.69171),
    (0.,30.),(70.,20.),(10.,179.9),(50.3,18.)])
def test_requested_180_by_150_region_keeps_small_grid(tmp_path,lat,lon):
    fp,proj=footprint(tmp_path,lat,lon)
    dims=dw.polygon_ladder_dims(footprint=fp,projection=proj,ratios=(),buffers_km=(0.,))
    assert dims==[(18,16)]
    exp=experiment(fp,proj,dims)
    root=exp.domains[0].run
    assert (root.dx,root.dy,root.nz)==(12000.,12000.,49)
    assert (exp.projection.ref_lat,exp.projection.ref_lon)==(fp.center_lat,fp.center_lon)
    _field_boundary(np.zeros((2,root.ny,root.nx)),np.ones((2,root.ny,root.nx)),3600.,5)
    _validate_frame_domain(root.ny,root.nx,5,"health")


@pytest.mark.parametrize("ratios",[(4,),(4,3,2),(2,2,2,2)])
def test_small_nested_ladder_uses_actual_sint_donor_clearance(tmp_path,ratios):
    fp,proj=footprint(tmp_path)
    dims=dw.polygon_ladder_dims(footprint=fp,projection=proj,ratios=ratios,
                               buffers_km=(0.,)*(1+len(ratios)))
    exp=experiment(fp,proj,dims,ratios)
    if ratios==(4,):assert dims==[(36,34),(64,56)]
    for child in exp.domains[1:]:
        parent=exp.domain(child.parent_id)
        for stagger in ("","x","y"):
            register_nest(nri=child.parent_grid_ratio,nrj=child.parent_grid_ratio,
                i_parent_start=child.i_parent_start,j_parent_start=child.j_parent_start,
                child_nx=child.run.nx,child_ny=child.run.ny,
                parent_nx=parent.run.nx,parent_ny=parent.run.ny,stagger=stagger,wrapper="bdy")


def test_constructor_requirement_is_root_only_and_not_public_identity(tmp_path,monkeypatch):
    fp,proj=footprint(tmp_path,width=1,height=1)
    generic=dw.polygon_ladder_dims(footprint=fp,projection=proj,ratios=(),buffers_km=(0.,))
    adapter=get_source_adapter("hrrr")
    assert adapter.root_target_interior_axis==NATIVE_TARGET_INTERIOR_AXIS
    native=dw.polygon_ladder_dims(footprint=fp,projection=proj,ratios=(),buffers_km=(0.,),
        root_minimum_axis=boundary_axis(5,interior_points=adapter.root_target_interior_axis))
    assert generic==[(12,12)] and native==[(14,14)]
    from gpuwm.hrrr_route_inputs import target_domain
    target_domain(experiment(fp,proj,native))
    with pytest.raises(ValueError,match="at least 13"):
        target_domain(experiment(fp,proj,generic))
    changed=replace(adapter,root_target_interior_axis=21)
    assert changed.to_dict()==adapter.to_dict()
    # A different/new row's same operation declaration drives the fitter;
    # there is no model-name condition in the minimum selection.
    monkeypatch.setattr(dw,"get_source_adapter",lambda source:changed)
    dims,_=dw.fit_polygon_ladder(footprint=fp,projection=proj,ratios=(2,),
        buffers_km=(0.,0.),free_bytes=32*dw.GIB,vram_gib=32,hours=3,
        source="gfs",name="declared_operation",start_time=datetime(2026,9,5))
    assert dims==[(32,32),(12,12)]


def test_thin_region_is_not_expanded_to_preset_aspect(tmp_path):
    fp,proj=footprint(tmp_path,width=900,height=100)
    assert dw.polygon_ladder_dims(footprint=fp,projection=proj,ratios=(),buffers_km=(0.,))==[(78,14)]
    assert dw._MIN_SCALE==0.55
    assert dw._dims_for_scale(dw._MIN_SCALE,())==[(60,48)]


def test_forced_small_streaming_keeps_real_window_and_halo_guard():
    with pytest.raises(ValueError,match="33"):
        plan_tiles(18,16,1,1,16,periodic=False)


@pytest.mark.parametrize("width",[1,4,5,8])
def test_boundary_consumers_preserve_their_distinct_exact_edges(width):
    size=boundary_axis(width)
    _field_boundary(np.zeros((1,size,size)),np.zeros((1,size,size)),1.,width)
    with pytest.raises(ValueError,match="too small"):
        _field_boundary(np.zeros((1,size-1,size)),np.zeros((1,size-1,size)),1.,width)
    with pytest.raises(ValueError,match="no unique interior"):
        _validate_frame_domain(size,size,width,"test")
    _validate_frame_domain(size+1,size+1,width,"test")


def test_internal_capability_does_not_move_existing_positional_notes():
    from dataclasses import fields
    from gpuwm.source_adapters import SourceAdapter
    adapter=replace(get_source_adapter("gfs"),notes="existing positional notes")
    positional=[getattr(adapter,f.name) for f in fields(SourceAdapter)
                if f.init and f.name != "root_target_interior_axis"]
    rebuilt=SourceAdapter(*positional)
    assert rebuilt.notes=="existing positional notes"
    assert rebuilt.root_target_interior_axis is None
    assert rebuilt.to_dict()==adapter.to_dict()


@pytest.mark.parametrize("mode",["auto","on"])
def test_public_small_grid_keeps_actual_streaming_contract(tmp_path,mode):
    from gpuwm.launchpad_api import create_draft
    request=dict(directory=str(tmp_path),name="tiny_stream_contract",lat=35.3,lon=-97.5,
        width=180,height=150,dx=12,nz=49,hours=1,history=720,chain="",tiles=mode,
        vram=16,profile="",source="gfs",cycle="2026-09-05T00")
    if mode=="on":
        with pytest.raises(dw.DomainFitError,match=r"tile \+ 2\*halo <= n"):
            create_draft(request)
        assert not (tmp_path/"forecast.toml").exists()
    else:
        result=create_draft(request)
        assert result["inspection"]["grid_geometry"]==[
            {"grid_id":1,"nx":18,"ny":16,"dx":12000.,"dy":12000.}]
