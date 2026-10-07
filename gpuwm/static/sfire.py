"""SFIRE static preparation: network orchestration around the Rust builder.

The Rust static-fields bridge owns projection, raster decoding, nearest fuel,
four-point terrain, fine-grid derivatives and deterministic NPZ output.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request
import urllib.error
import time

from gpuwm.static import rust_bridge

SCHEMA="gpuwm-sfire-static-v1"
ROUTES=Path(__file__).resolve().parents[1]/"authorities"/"sfire-static-routes.v1.json"


def source_routes():
    document=json.loads(ROUTES.read_text(encoding="utf-8"))
    if document.get("schema")!="gpuwm-sfire-static-routes-v1":
        raise ValueError("SFIRE static route schema differs from the acquisition contract")
    return document["sources"]


def _library():
    if rust_bridge.python_fallback_requested():
        raise ValueError("SFIRE requires Rust raster preparation; the Python fallback would bypass fine-grid static semantics")
    library=rust_bridge.load()
    if not hasattr(library,"gpuwm_static_sfire_v1"):
        raise RuntimeError("Rust static-fields bridge lacks the SFIRE static builder; rebuild static-fields before preparing fire data")
    library.gpuwm_static_sfire_v1.argtypes=[]
    library.gpuwm_static_sfire_v1.restype=ctypes.c_uint32
    if library.gpuwm_static_sfire_v1()!=1:
        raise RuntimeError("Rust SFIRE static builder contract differs from version 1; rebuild static-fields before preparing fire data")
    library.gpuwm_static_sfire_bounds.argtypes=[ctypes.POINTER(ctypes.c_uint8),ctypes.c_size_t,ctypes.POINTER(ctypes.c_double)]
    library.gpuwm_static_sfire_bounds.restype=ctypes.c_int32
    library.gpuwm_static_sfire_coverage_bounds.argtypes=library.gpuwm_static_sfire_bounds.argtypes
    library.gpuwm_static_sfire_coverage_bounds.restype=ctypes.c_int32
    library.gpuwm_static_sfire_build.argtypes=[ctypes.POINTER(ctypes.c_uint8),ctypes.c_size_t,ctypes.POINTER(ctypes.c_uint64)]
    library.gpuwm_static_sfire_build.restype=ctypes.c_int32
    return library


def _json_buffer(document):
    raw=json.dumps(document).encode("utf-8")
    return (ctypes.c_uint8*len(raw)).from_buffer_copy(raw),len(raw)


def read_fire_ideal_inputs(directory, fields=(), *, sounding=None, landuse=None):
    """Read native formatted initializer files into unchanged host f32 buffers.

    Rust owns list-directed records, terminal geometry, table sections and
    every value conversion. Python owns request metadata and buffer custody.
    """
    import numpy as np
    library=_library()
    for symbol in ("gpuwm_static_sfire_ideal_inputs","gpuwm_static_sfire_ideal_field_f32"):
        if not hasattr(library,symbol):
            raise RuntimeError("Rust static-fields lacks native fire initializer text input; rebuild static-fields before fire-ideal")
    library.gpuwm_static_sfire_ideal_inputs.argtypes=[ctypes.POINTER(ctypes.c_uint8),ctypes.c_size_t,ctypes.POINTER(ctypes.c_uint64)]
    library.gpuwm_static_sfire_ideal_inputs.restype=ctypes.c_int32
    library.gpuwm_static_sfire_ideal_field_f32.argtypes=[ctypes.c_uint64,ctypes.POINTER(ctypes.c_uint8),ctypes.c_size_t,ctypes.POINTER(ctypes.c_float),ctypes.c_size_t]
    library.gpuwm_static_sfire_ideal_field_f32.restype=ctypes.c_int64
    library.gpuwm_static_sfire_metadata_json.argtypes=[ctypes.c_uint64,ctypes.POINTER(ctypes.c_uint8),ctypes.c_size_t]
    library.gpuwm_static_sfire_metadata_json.restype=ctypes.c_int64
    library.gpuwm_static_sfire_metadata_drop.argtypes=[ctypes.c_uint64]
    library.gpuwm_static_sfire_metadata_drop.restype=None
    request=dict(directory=str(Path(directory).resolve()),fields=list(fields))
    if sounding is not None:request["sounding"]=str(Path(sounding).resolve())
    if landuse is not None:request["landuse"]=dict(landuse,path=str(Path(landuse["path"]).resolve()))
    buffer,length=_json_buffer(request);handle=ctypes.c_uint64()
    if library.gpuwm_static_sfire_ideal_inputs(buffer,length,ctypes.byref(handle)):
        raise ValueError(rust_bridge.last_error(library))
    try:
        size=library.gpuwm_static_sfire_metadata_json(handle.value,None,0)
        if size<0:raise ValueError(rust_bridge.last_error(library))
        data=(ctypes.c_uint8*size)()
        if library.gpuwm_static_sfire_metadata_json(handle.value,data,size)!=size:
            raise ValueError(rust_bridge.last_error(library))
        metadata=json.loads(bytes(data).decode("utf-8"))
        arrays={}
        for name,shape in metadata["shapes"].items():
            key=name.encode("utf-8");key_buffer=(ctypes.c_uint8*len(key)).from_buffer_copy(key)
            array=np.empty(tuple(shape),dtype=np.float32)
            count=library.gpuwm_static_sfire_ideal_field_f32(handle.value,key_buffer,len(key),array.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),array.size)
            if count!=array.size:raise ValueError(rust_bridge.last_error(library))
            arrays[name]=array
        output={"fields":{name:arrays[name] for name in metadata["field_names"]},
                "sounding":{name:arrays[key] for name,key in metadata["sounding_names"].items()},
                "_metadata":metadata}
        if metadata["landuse"] is not None:output["fields"]["_LANDUSE_TABLE"]=arrays["_LANDUSE_TABLE"]
        return output
    finally:
        library.gpuwm_static_sfire_metadata_drop(handle.value)
        rust_bridge.fieldset_free(handle.value)


def fire_ideal_input_requests(cfg):
    """The native initializer's nine fixed REAL files and producing extents."""
    atmosphere=(int(cfg.nx)+1,int(cfg.ny)+1)
    fine=(atmosphere[0]*int(cfg.sr_x),atmosphere[1]*int(cfg.sr_y))
    entries=[]
    def add(name,filename,shape):
        entries.append(dict(name=name,filename=filename,ni=shape[0],nj=shape[1]))
    if bool(cfg.sfc_full_init):
        for flag,name,filename in (("fire_read_lu","INPUT_LU","input_lu"),
                                   ("fire_read_tsk","INPUT_TSK","input_tsk"),
                                   ("fire_read_tmn","INPUT_TMN","input_tmn")):
            if bool(getattr(cfg,flag)):add(name,filename,atmosphere)
    if int(cfg.fire_mountain_type)==0:
        if bool(cfg.fire_read_atm_ht):add("INPUT_HT","input_ht",atmosphere)
        if bool(cfg.fire_read_fire_ht):add("INPUT_ZSF","input_zsf",fine)
        if bool(cfg.fire_read_fire_grad):
            add("INPUT_DZDXF","input_dzdxf",fine);add("INPUT_DZDYF","input_dzdyf",fine)
    if int(cfg.fire_fuel_read)==2:add("INPUT_FC","input_fc",fine)
    if int(cfg.fire_fmc_read)==2:add("INPUT_FMC_G","input_fmc_g",fine)
    return entries


def read_native_landuse_table(path, section="USGS"):
    """Raw seasonal native columns; the GPU initializer owns landuse math."""
    result=read_fire_ideal_inputs(Path(path).parent,landuse=dict(path=path,section=section))
    metadata=dict(result["_metadata"]["landuse"],file_sha256=next(iter(result["_metadata"]["file_sha256"].values())),
                  source_path=str(Path(path).resolve()))
    return result["fields"]["_LANDUSE_TABLE"],metadata


read_landuse_table=read_native_landuse_table


def fire_bounds(grid_spec,sr_x,sr_y):
    """Obtain the acquisition rectangle from Rust fine-grid projection."""
    library=_library()
    buffer,length=_json_buffer(dict(grid_spec=grid_spec,sr_x=sr_x,sr_y=sr_y))
    out=(ctypes.c_double*4)()
    if library.gpuwm_static_sfire_bounds(buffer,length,out):
        raise ValueError(rust_bridge.last_error(library))
    return tuple(out)


def coverage_bounds(bounds,resolution,*,source_crs="EPSG:4326",pixel_edge_offset=0.):
    """Project and snap a coverage request on the route's native pixel grid."""
    library=_library()
    buffer,length=_json_buffer(dict(bounds=bounds,resolution=resolution,
        source_crs=source_crs,pixel_edge_offset=pixel_edge_offset))
    out=(ctypes.c_double*4)()
    if library.gpuwm_static_sfire_coverage_bounds(buffer,length,out):
        raise ValueError(rust_bridge.last_error(library))
    return tuple(out)


def build_fire_static(grid_spec,sr_x,sr_y,terrain,fuel,*,output=None,observed_perimeter=None):
    """Build all six named fire fields through the native static ABI."""
    library=_library()
    request=dict(grid_spec=grid_spec,sr_x=sr_x,sr_y=sr_y,terrain=terrain,fuel=fuel)
    if observed_perimeter is not None:
        request["observed_perimeter"]=dict(observed_perimeter,grid_spec=grid_spec,sr_x=sr_x,sr_y=sr_y)
    if output is not None:
        request["output"]=str(Path(output).resolve())
    buffer,length=_json_buffer(request)
    handle=ctypes.c_uint64()
    if library.gpuwm_static_sfire_build(buffer,length,ctypes.byref(handle)):
        raise ValueError(rust_bridge.last_error(library))
    try:
        return rust_bridge.fieldset_to_dict(handle.value)
    finally:
        rust_bridge.fieldset_free(handle.value)


def build_observed_perimeter(grid_spec,sr_x,sr_y,geojson,source_contract,*,
                             observed_utc_epoch_ms,model_elapsed_seconds,
                             assumed_burn_age_seconds,output=None):
    """Rasterize captured geometry in Rust, with an explicit burn-age assumption.

    Returns LFN_HIST and HISTORICAL_TIGN for the native perimeter driver.
    The source contract binds geometry bytes, geographic CRS and UTC time.
    """
    library=_library()
    if not hasattr(library,"gpuwm_static_sfire_observed_perimeter"):
        raise RuntimeError("Rust SFIRE library lacks observed-perimeter preparation; rebuild static-fields")
    library.gpuwm_static_sfire_observed_perimeter.argtypes=[ctypes.POINTER(ctypes.c_uint8),ctypes.c_size_t,ctypes.POINTER(ctypes.c_uint64)]
    library.gpuwm_static_sfire_observed_perimeter.restype=ctypes.c_int32
    request=dict(grid_spec=grid_spec,sr_x=sr_x,sr_y=sr_y,geojson=str(Path(geojson).resolve()),
        source_contract=str(Path(source_contract).resolve()),observed_utc_epoch_ms=observed_utc_epoch_ms,
        model_elapsed_seconds=model_elapsed_seconds,assumed_burn_age_seconds=assumed_burn_age_seconds)
    if output is not None:request["output"]=str(Path(output).resolve())
    buffer,length=_json_buffer(request);handle=ctypes.c_uint64()
    if library.gpuwm_static_sfire_observed_perimeter(buffer,length,ctypes.byref(handle)):
        raise ValueError(rust_bridge.last_error(library))
    try:return rust_bridge.fieldset_to_dict(handle.value)
    finally:rust_bridge.fieldset_free(handle.value)


def load_fire_static(path,*,expected_sha256=None,expected_grid_spec=None):
    """Restore native static fields and ``_metadata`` from a sealed Rust bundle.

    Rust owns ZIP/NPY decode, CRC and field hashes, and validates mesh shape,
    WRF REAL spacing, categories and coordinates against embedded metadata.
    Observed bundles also restore the declared LFN_HIST/HISTORICAL_TIGN pair.
    """
    library=_library()
    if not hasattr(library,"gpuwm_static_sfire_load"):
        raise RuntimeError("Rust SFIRE static library predates bundle loading; rebuild static-fields before loading fire data")
    library.gpuwm_static_sfire_load.argtypes=[ctypes.POINTER(ctypes.c_uint8),ctypes.c_size_t,ctypes.POINTER(ctypes.c_uint64)]
    library.gpuwm_static_sfire_load.restype=ctypes.c_int32
    library.gpuwm_static_sfire_metadata_json.argtypes=[ctypes.c_uint64,ctypes.POINTER(ctypes.c_uint8),ctypes.c_size_t]
    library.gpuwm_static_sfire_metadata_json.restype=ctypes.c_int64
    library.gpuwm_static_sfire_metadata_drop.argtypes=[ctypes.c_uint64]
    library.gpuwm_static_sfire_metadata_drop.restype=None
    request=dict(path=str(Path(path).resolve()))
    if expected_sha256 is not None:request["expected_sha256"]=expected_sha256
    if expected_grid_spec is not None:request["expected_grid_spec"]=expected_grid_spec
    buffer,length=_json_buffer(request);handle=ctypes.c_uint64()
    if library.gpuwm_static_sfire_load(buffer,length,ctypes.byref(handle)):
        raise ValueError(rust_bridge.last_error(library))
    try:
        size=library.gpuwm_static_sfire_metadata_json(handle.value,None,0)
        if size<0:raise ValueError(rust_bridge.last_error(library))
        metadata_buffer=(ctypes.c_uint8*size)()
        written=library.gpuwm_static_sfire_metadata_json(handle.value,metadata_buffer,size)
        if written!=size:raise ValueError(rust_bridge.last_error(library))
        fields=rust_bridge.fieldset_to_dict(handle.value)
        fields["_metadata"]=json.loads(bytes(metadata_buffer).decode("utf-8"))
        return fields
    finally:
        library.gpuwm_static_sfire_metadata_drop(handle.value)
        rust_bridge.fieldset_free(handle.value)


def _bound(fetched):
    return dict(path=str(fetched.path.resolve()),sha256=fetched.sha256,
                expected_bytes=fetched.bytes)


def _wcs_open(url,offset):
    """Retry an observed transient 404 from the published coverage service."""
    request=urllib.request.Request(url)
    if offset:
        request.add_header("Range",f"bytes={offset}-")
    for attempt in range(3):
        try:
            return urllib.request.urlopen(request,timeout=120)
        except urllib.error.HTTPError as error:
            if error.code != 404 or attempt == 2:
                raise
            time.sleep(attempt+1)


def acquire(source_id,year,bounds,cache_root):
    """Acquire an opaque public source via its packaged transport row."""
    from gpuwm.static import highres_fetch as fetch
    row=source_routes()[source_id]
    bbox=fetch.FootprintBBox(lat_min=bounds[1],lat_max=bounds[3],lon_min=bounds[0],lon_max=bounds[2])
    if row["kind"]=="terrain-tiles":
        tiles,absent=getattr(fetch,row["fetch_function"])(bbox,Path(cache_root))
        if absent:
            raise ValueError(f"SFIRE terrain source does not cover required tiles: {', '.join(absent)}")
        result=getattr(fetch,row["derive_function"])(tiles,bbox,Path(cache_root))
        return _bound(result),dict(source=source_id,license=row["license"],source_url=row["source_url"],
                                   objects=[item.receipt() for item in tiles],derived=result.receipt())
    if row["kind"]!="wcs-geotiff":
        raise ValueError(f"SFIRE source transport {row['kind']} is unknown")
    years=row.get("years",[])
    eligible=[value for value in years if value<=year]
    if years and not eligible:
        raise ValueError(f"SFIRE source {source_id} has no fuel release at or before requested year {year}; available years: {years}")
    selected=max(eligible) if years else None
    resolution=row.get("resolution",row.get("resolution_degrees"))
    crs=row.get("crs","EPSG:4326")
    bounds=coverage_bounds(bounds,resolution,source_crs=crs,pixel_edge_offset=row.get("pixel_edge_offset",0.))
    query=dict(service="WCS",version="1.0.0",request="GetCoverage",
        coverage=row["coverage"].format(year=selected),crs=crs,response_crs=crs,
        bbox=",".join(format(value,".10f") for value in bounds),
        resx=resolution,resy=resolution,format="GeoTIFF",
        interpolation="nearest neighbor")
    url=row["url"].format(year=selected)+"?"+urllib.parse.urlencode(query)
    key=hashlib.sha256(url.encode()).hexdigest()[:24]
    label=selected if selected is not None else "current"
    path=Path(cache_root)/source_id/f"{label}_{key}.tif"
    result=fetch.fetch_file(url,path,urlopen=_wcs_open)
    with path.open("rb") as stream:
        signature=stream.read(4)
    if signature not in (b"II*\x00",b"MM\x00*",b"II+\x00",b"MM\x00+"):
        deleted=[]
        if not result.cache_hit:
            for own in (path,fetch._sidecar(path)):
                if own.is_file():
                    deleted.append(dict(path=str(own),bytes=own.stat().st_size))
                    own.unlink()
        raise ValueError(f"SFIRE WCS returned an error payload instead of GeoTIFF; removed new files {deleted}")
    bound=_bound(result)
    if crs=="EPSG:4326":
        bound["lattice_resolution_degrees"]=resolution
    return bound,dict(source=source_id,requested_year=year,source_year=selected,
        license=row["license"],license_url=row["license_url"],source_url=row["source_url"],
        object=result.receipt())


def prepare(config,output,*,base=None):
    """Resolve bound inputs or fetch routes and seal a Rust-produced artifact."""
    _library()
    if config.get("schema")!=SCHEMA:
        raise ValueError(f"fire static config must declare {SCHEMA}")
    base=Path.cwd() if base is None else Path(base)
    config=dict(config)
    if "experiment" in config:
        if "grid_spec" in config:
            raise ValueError("fire static request declares both an experiment and a grid_spec; use one atmospheric grid authority")
        experiment=Path(config.pop("experiment"))
        if not experiment.is_absolute():experiment=base/experiment
        config=resolve_experiment_request(experiment,config,domain_id=config.pop("grid_id",None))
    grid=config["grid_spec"];rx=config["sr_x"];ry=config["sr_y"]
    bounds=fire_bounds(grid,rx,ry)
    cache=Path(config.get("cache_root",base/"sfire-static-cache"))
    if not cache.is_absolute():cache=base/cache
    metadata={};inputs={}
    options=config.get("sources",{})
    for role,default in (("terrain","usgs-3dep-wcs"),("fuel","landfire-fbfm40")):
        if role in config:
            bound=dict(config[role]);path=Path(bound["path"])
            if not path.is_absolute():path=base/path
            bound["path"]=str(path.resolve())
            inputs[role]=bound
            metadata[role]=dict(sha256=bound["sha256"],path=bound["path"])
        else:
            source=options.get(role,default)
            if source_routes()[source]["role"]!=role:
                raise ValueError(f"SFIRE source {source} cannot supply required {role} fields")
            inputs[role],metadata[role]=acquire(source,int(options.get("fuel_year",2024)),bounds,cache)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    observed=config.get("observed_perimeter")
    if observed is not None:
        observed=dict(observed)
        for key in ("geojson","source_contract"):
            path=Path(observed[key])
            observed[key]=str((path if path.is_absolute() else base/path).resolve())
    fields=build_fire_static(grid,rx,ry,inputs["terrain"],inputs["fuel"],output=output,observed_perimeter=observed)
    return dict(schema=SCHEMA,compute="Rust static-fields",sr_x=rx,sr_y=ry,
        grid_spec=grid,bounds=bounds,source_inputs=metadata,output=str(output.resolve()),
        experiment=config.get("experiment_identity"),
        output_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        fields={name:dict(shape=list(value.shape),dtype=str(value.dtype)) for name,value in fields.items()})


def experiment_grid_specs(experiment):
    """Resolve the projection and parent/start/ratio chain entirely in Rust."""
    from dataclasses import asdict
    if experiment.projection is None:
        raise ValueError("Fire static preparation requires the experiment's geographic projection")
    library=_library()
    if not hasattr(library,"gpuwm_static_sfire_experiment_grids"):
        raise RuntimeError("Rust SFIRE static library predates experiment grid preparation; rebuild static-fields")
    library.gpuwm_static_sfire_experiment_grids.argtypes=[ctypes.POINTER(ctypes.c_uint8),ctypes.c_size_t,ctypes.POINTER(ctypes.c_uint8),ctypes.c_size_t]
    library.gpuwm_static_sfire_experiment_grids.restype=ctypes.c_int64
    domains=[]
    for domain in experiment.domains:
        row={key:getattr(domain,key) for key in ("grid_id","parent_id","i_parent_start","j_parent_start","parent_grid_ratio")}
        row.update({key:getattr(domain.run,key) for key in ("nx","ny","dx","dy")})
        domains.append(row)
    buffer,length=_json_buffer(dict(projection=asdict(experiment.projection),domains=domains))
    size=library.gpuwm_static_sfire_experiment_grids(buffer,length,None,0)
    if size<0:raise ValueError(rust_bridge.last_error(library))
    out=(ctypes.c_uint8*size)()
    if library.gpuwm_static_sfire_experiment_grids(buffer,length,out,size)!=size:
        raise ValueError(rust_bridge.last_error(library))
    return {int(key):value for key,value in json.loads(bytes(out).decode("utf-8")).items()}


def resolve_experiment_request(path,request=None,*,domain_id=None):
    """Bind source/perimeter options to one resolved experiment domain."""
    from gpuwm.experiment import load_experiment
    experiment=load_experiment(path)
    if domain_id is None:
        candidates=[domain for domain in experiment.domains if domain.run.ifire==2]
        if len(candidates)!=1:
            raise ValueError("Fire static preparation needs --domain when the experiment does not declare exactly one SFIRE domain")
        domain=candidates[0]
    else:
        domain=next((item for item in experiment.domains if item.grid_id==int(domain_id)),None)
        if domain is None:raise ValueError(f"Fire static domain {domain_id} does not exist in the experiment")
    config={} if request is None else dict(request)
    if config.get("schema",SCHEMA)!=SCHEMA:
        raise ValueError(f"Fire source request must use {SCHEMA}")
    for key in ("grid_spec","sr_x","sr_y"):
        if key in config:
            raise ValueError(f"Fire source request cannot override {key}; the resolved experiment owns its fire grid")
    config.update(schema=SCHEMA,grid_spec=experiment_grid_specs(experiment)[domain.grid_id],
                  sr_x=domain.run.sr_x,sr_y=domain.run.sr_y)
    config["experiment_identity"]=dict(sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),grid_id=domain.grid_id)
    sources=dict(config.get("sources",{}));sources.setdefault("fuel_year",experiment.start_time.year)
    config["sources"]=sources
    return config


def _arguments(parser):
    parser.add_argument("config",type=Path,help="static REQUEST.json or resolved EXPERIMENT.toml")
    parser.add_argument("--domain",type=int,help="experiment domain grid_id; defaults to its only SFIRE domain")
    parser.add_argument("--request",type=Path,help="source routes, bound rasters and optional captured observed_perimeter JSON")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--receipt",type=Path)


def _run(args):
    if args.config.suffix.lower()==".toml":
        request={} if args.request is None else json.loads(args.request.read_text(encoding="utf-8"))
        config=resolve_experiment_request(args.config,request,domain_id=args.domain)
        base=args.config.parent if args.request is None else args.request.parent
    else:
        if args.request is not None or args.domain is not None:
            raise ValueError("--request and --domain accompany an experiment TOML; a JSON request declares its experiment and grid_id itself")
        config=json.loads(args.config.read_text(encoding="utf-8"));base=args.config.parent
    receipt=prepare(config,args.output,base=base)
    text=json.dumps(receipt,indent=2,sort_keys=True)+"\n"
    if args.receipt:
        args.receipt.parent.mkdir(parents=True,exist_ok=True);args.receipt.write_text(text,encoding="utf-8")
    print(text,end="")
    return 0


def register_cli(subparsers):
    parser=subparsers.add_parser("fire-static",help="prepare refined fire fuel, terrain and captured perimeter with Rust")
    _arguments(parser);parser.set_defaults(func=_run)


def build_parser():
    """The parser ``gpuwm-sfire-static`` runs, which the CLI reference reads."""
    parser=argparse.ArgumentParser(prog="gpuwm-sfire-static",
                                   description="Prepare refined SFIRE fuel and terrain with Rust")
    _arguments(parser)
    return parser


def main(argv=None):
    return _run(build_parser().parse_args(argv))


if __name__=="__main__":
    raise SystemExit(main())
