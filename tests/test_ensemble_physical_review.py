"""Independent negative controls for physical-source authority and storage."""
from copy import deepcopy
from datetime import timedelta
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.ensemble.physical_store import (
    NativePhysicalStore, digest_file, physical_input_binding, physical_static_identity,
    validate_physical_input_binding,
)


def test_physical_input_floor_is_separate_from_the_perturbation_bound():
    from gpuwm.ensemble.recentered import FieldBounds
    assert FieldBounds("kg kg-1",.005,0.,.1).effective_input_lower == 0
    assert FieldBounds("kg kg-1",.005,0.,.1,input_lower=-.005).effective_input_lower == -.005
    for value in (.001,float("nan"),float("inf")):
        with pytest.raises(ValueError,match="input_lower"):
            FieldBounds("kg kg-1",.005,0.,.1,input_lower=value)


def test_static_identity_binds_values_shapes_and_land_categories():
    first = {"HGT_M": np.arange(6,dtype="f4").reshape(2,3), "LANDMASK": np.ones((2,3),bool)}
    attrs = {"MMINLU": b"MODIS\x00 ", "ISWATER": np.int32(17), "NUM_LAND_CAT": 21.0}
    expected = physical_static_identity(first,attrs)
    equivalent = {key: np.asfortranarray(value,dtype=">f8") for key,value in first.items()}
    assert physical_static_identity(equivalent,{"MMINLU":"MODIS", "ISWATER":17.0,"NUM_LAND_CAT":21}) == expected
    terrain = {**first,"HGT_M":first["HGT_M"].copy()}
    terrain["HGT_M"][0,0] = 1
    assert physical_static_identity(terrain,attrs) != expected
    assert physical_static_identity({key:value.reshape(3,2) for key,value in first.items()},attrs) != expected
    assert physical_static_identity(first,{**attrs,"ISWATER":16}) != expected


@pytest.fixture
def captured(tmp_path):
    from gpuwm.io.nc_writer_bridge import unavailable_reason
    from gpuwm.netcdf_bridge import find_netcdf_bin
    reason = unavailable_reason()
    if reason or find_netcdf_bin() is None:
        pytest.skip("native physical-store writer/reader artifact required: " + str(reason))
    from gpuwm.native_wrf_contract import native_geometry_contract
    from gpuwm.static.lambert import LambertGrid
    from test_ensemble_physical_preparation import snapshot
    cfg = SimpleNamespace(nx=3,ny=2,nz=5,dx=3000.,dy=3000.)
    grid = LambertGrid(ref_lat=40,ref_lon=-100,truelat1=30,truelat2=60,
                       stand_lon=-100,dx=3000,dy=3000,e_we=4,e_sn=3)
    statics = physical_static_identity({"HGT_M":np.zeros((2,3)),"LANDMASK":np.ones((2,3))},
                                       {"MMINLU":"MODIS","ISWATER":17})
    source = {"adapter":"rw-wps-mapped-composition-v2", "input_manifest_sha256":"a"*64,
              "mapping_sha256":"b"*64,"composition_sha256":"c"*64,
              "composition_receipt_sha256":"d"*64,
              "preparation_case_policy":{"id":"ordinary"},
              "water_temperature_overlay":None,
              "preprocessing":{"bridge_sha256":"e"*64}, "static_identity":statics}
    from physical_field_fixtures import analytic_field_contract
    geometry = native_geometry_contract(grid,cfg)
    writer = NativePhysicalStore(tmp_path/"physical",grid_identity=geometry,
                                  source_identity=source, field_contract=analytic_field_contract(geometry))
    writer.write(snapshot())
    writer.seal()
    return NativePhysicalStore(writer.root),grid,cfg,source,statics


@pytest.mark.parametrize("key,value", [
    ("adapter","another-source"), ("mapping_sha256","1"*64),
    ("composition_sha256","2"*64), ("composition_receipt_sha256","3"*64),
    ("preparation_case_policy",{"id":"changed"}),
    ("water_temperature_overlay",{"provider":"changed"}),
    ("static_highres",{"enabled":True}),
])
def test_same_raw_manifest_cannot_hide_changed_source_authority(captured,key,value):
    store,grid,cfg,source,statics = captured
    changed = {**source,key:value}
    with pytest.raises(ValueError,match="source authority"):
        physical_input_binding(store,grid,cfg,changed,input_manifest_sha256="a"*64,
                               static_identity=statics)


def test_binding_permits_distinct_implementations_but_refuses_static_drift(captured):
    store,grid,cfg,source,statics = captured
    changed = {**source,"preprocessing":{"bridge_sha256":"f"*64},
               "git_commit":"f"*40,"source_sha256":{"gpuwm/ingest/real.py":"f"*64}}
    binding = physical_input_binding(store,grid,cfg,changed,input_manifest_sha256="a"*64,
                                      static_identity=statics)
    validate_physical_input_binding(binding,input_manifest_sha256="a"*64,source_identity=changed)
    changed_static = deepcopy(statics)
    changed_static["attributes"]["ISWATER"] = 16
    with pytest.raises(ValueError,match="statics differ"):
        physical_input_binding(store,grid,cfg,source,input_manifest_sha256="a"*64,
                               static_identity=changed_static)
    with pytest.raises(ValueError,match="static identity"):
        physical_input_binding(store,grid,cfg,source,input_manifest_sha256="a"*64)


def test_portable_reader_rechecks_the_captured_base_authority(captured):
    store,grid,cfg,source,statics = captured
    binding = physical_input_binding(store,grid,cfg,source,input_manifest_sha256="a"*64,
                                      static_identity=statics)
    with pytest.raises(ValueError,match="source authority"):
        validate_physical_input_binding(binding,input_manifest_sha256="a"*64,
                                         source_identity={**source,"mapping_sha256":"9"*64})
    old = deepcopy(binding)
    del old["manifest"]["source"]["static_identity"]
    import hashlib
    old["manifest_sha256"] = hashlib.sha256((json.dumps(old["manifest"],sort_keys=True,separators=(",",":"))+"\n").encode()).hexdigest()
    with pytest.raises(ValueError,match="captured static"):
        validate_physical_input_binding(old,input_manifest_sha256="a"*64,source_identity=source)


@pytest.mark.parametrize("mutation", ["time","grid","dtype"])
def test_native_frame_headers_and_dtype_refuse_relabelled_manifest(captured,mutation):
    store,*_ = captured
    document = deepcopy(store.document)
    if mutation == "time":
        document["frames"][0]["valid_time"] = (store.times[0]+timedelta(hours=1)).isoformat()
    elif mutation == "grid":
        document["grid"]["known_x"] += 1
    else:
        document["frames"][0]["arrays"]["field__TT"]["dtype"] = "<f8"
    # Only metadata changes. The untouched native payload still has its
    # original SHA and therefore passes the file-integrity check itself.
    store.manifest_path.write_text(json.dumps(document))
    with pytest.raises(ValueError,match="header|dtype|grid authority"):
        NativePhysicalStore(store.root).read(0)


def test_native_pressure_alignment_uses_each_c_grid_coordinate_and_source_order():
    bridge = os.environ.get("GPUWM_ENSEMBLE_PREPARATION_BRIDGE")
    if not bridge:
        pytest.skip("requires the native ensemble preparation bridge")
    from dataclasses import replace
    from gpuwm.ensemble.native_preparation import NativeEnsemblePreparation
    from gpuwm.ensemble.physical_recenter import RecenteredPhysicalPreparation
    from gpuwm.ingest.preprocess_backend import ParallelCpuPreprocessBackend
    from test_ensemble_physical_preparation import snapshot
    preparation = RecenteredPhysicalPreparation.__new__(RecenteredPhysicalPreparation)
    from gpuwm.ensemble.physical_recenter import PHYSICAL_BOUNDS, _SURFACE
    preparation.specific = True
    preparation.bounds = dict(PHYSICAL_BOUNDS)
    preparation.surface = dict(_SURFACE)
    preparation.native = NativeEnsemblePreparation(bridge)
    preparation.backend = ParallelCpuPreprocessBackend(bridge=bridge,workers=1)
    frame = snapshot()
    xscale = np.asarray([[1.,1.01,1.02],[1.005,1.015,1.025]],dtype="f4")
    source_pressure = frame.fields["PRES"]*xscale
    target_pressure = np.asarray([85000.,75000.,65000.,55000.],dtype="f4")[:,None,None]*xscale
    def face(p,axis):
        shape = list(p.shape)
        shape[axis] += 1
        out = np.empty(shape,dtype="f4")
        sl0 = [slice(None)]*3
        sl1 = [slice(None)]*3
        sl0[axis],sl1[axis] = 0,0
        out[tuple(sl0)] = p[tuple(sl1)]
        sl0[axis],sl1[axis] = -1,-1
        out[tuple(sl0)] = p[tuple(sl1)]
        sl0[axis],sl1[axis] = slice(1,-1),slice(0,-1)
        right = [slice(None)]*3
        right[axis] = slice(1,None)
        out[tuple(sl0)] = ((p[tuple(sl1)].astype("f8")+p[tuple(right)].astype("f8"))*.5).astype("f4")
        return out
    fields = dict(frame.fields,PRES=source_pressure)
    expected = {}
    profiles = {"TT":(280.,15.,None),"SPFH":(.002,.001,None),
                "GHT":(5000.,-7000.,None),"UU":(3.,4.,2),"VV":(-2.,3.,1)}
    surfaces = {"TT":"T2","SPFH":"Q2","GHT":"SOURCE_OROGRAPHY","UU":"U10","VV":"V10"}
    for name,(offset,slope,axis) in profiles.items():
        src = source_pressure if axis is None else face(source_pressure,axis)
        dst = target_pressure if axis is None else face(target_pressure,axis)
        surface_pressure = fields["PSFC"] if axis is None else face(fields["PSFC"][None],axis)[0]
        fields[name] = (offset+slope*np.log(src.astype("f8")/50000.)).astype("f4")
        fields[surfaces[name]] = (offset+slope*np.log(surface_pressure.astype("f8")/50000.)).astype("f4")
        expected[name] = (offset+slope*np.log(dst.astype("f8")/50000.)).astype("f4")
    frame = replace(frame,fields=fields,specific_humidity_authority=False)
    aligned = preparation._align(frame,target_pressure)
    indexed = preparation._align(replace(frame, levels_hpa=np.arange(1.,6.)),target_pressure)
    for name in aligned:
        assert aligned[name].tobytes() == indexed[name].tobytes()
    reversed_fields = {key:(value[::-1].copy() if value.ndim == 3 else value)
                       for key,value in fields.items()}
    reversed_frame = replace(frame,levels_hpa=frame.levels_hpa[::-1].copy(),fields=reversed_fields)
    opposite = preparation._align(reversed_frame,target_pressure)
    for name,value in expected.items():
        np.testing.assert_array_equal(aligned[name].view("u4"),opposite[name].view("u4"))
        # The analytic profile is rounded into float32 source samples before
        # native polynomial interpolation. Bound source quantization and its
        # arithmetic rounds in source units; output ULPs near zero do not
        # measure that error. A shifted mass/U/V pressure coordinate fails
        # this bound by orders of magnitude.
        log_ulp = float(np.spacing(np.float32(np.log(np.max(source_pressure)))))
        tolerance = (8*float(np.spacing(np.float32(np.max(np.abs(fields[name])))))
                     + 4*abs(profiles[name][1])*log_ulp)
        np.testing.assert_allclose(aligned[name],value,rtol=0,atol=tolerance,err_msg=name)
