"""S3 orchestration and small receipts for a public layer-source preparation.

No solver, decode, regrid or render implementation lives here. The public
CLI owns preparation and the existing RUC initializer owns cold admission.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import tomllib
import numpy as np


def configure(root):
    from gpuwm.toml_document import emit_experiment_toml
    from gpuwm.experiment import load_experiment
    from gpuwm.companion_domains import candidate_wps_text
    source = Path(__file__).resolve().parents[1] / "configs/ecmwf_oper_12km_quickstart.toml"
    raw = tomllib.loads(source.read_text())
    raw["experiment"].update(name="layer-source-prepare-check",
        start_time=datetime(2026,10,7,12), run_seconds=3600.0)
    raw["projection"].update(ref_lat=45.,ref_lon=-109.,stand_lon=-109.,truelat1=30.,truelat2=60.)
    raw["shared"].update(sf_surface_physics=3,num_soil_layers=9,mp_physics=6)
    raw["domain"][0].update(nx=256,ny=256,dx=3000.,time_step=18,cu_physics=0)
    raw["fetch"].update(cycle="2026-10-07T12",hours=3,out=str(root/"raw"))
    config = root/"prepare.toml"
    config.write_text(emit_experiment_toml(raw))
    exp = load_experiment(config)
    wps = root/"prepare.namelist.wps"
    wps.write_text(candidate_wps_text(raw,exp,exp,config))
    print(config)
    print(wps)


def inspect(root,out,expected,floor_input=None):
    from gpuwm.core.ruc import ruc_initialize_cold_start
    runs=sorted(root.glob("run-*"),reverse=True)
    if runs:
        root=next(run for run in runs if any(run.rglob("header.json")))
    records = {}
    selected = None
    for path in sorted(root.rglob("header.json")):
        header = json.loads(path.read_text())
        if header.get("status") != "READY" or not isinstance(header.get("arrays"),dict):
            continue
        for key,spec in sorted(header["arrays"].items()):
            values = np.load(path.parent/spec["file"],allow_pickle=False)
            name = f"{path.relative_to(root).as_posix()}:{key}"
            records[name] = {"dtype": str(values.dtype), "shape": list(values.shape),
                "sha256": hashlib.sha256(values.tobytes(order="C")).hexdigest()}
        if "surface/SMOIS" in header["arrays"]:
            selected=(path,header)
    if selected is None:
        raise ValueError("the real preparation has no sealed surface/SMOIS authority")
    path,header=selected
    read=lambda key: np.load(path.parent/header["arrays"][key]["file"],allow_pickle=False)
    moisture=read("surface/SMOIS")
    if floor_input is not None:
        assert moisture.dtype==np.float32
        floor_input.write_bytes(moisture.tobytes(order="C"))
    temperature=read("surface/TSLB")
    static_path=next(root.rglob("native-static.npz"))
    with np.load(static_path,allow_pickle=False) as static:
        soil=static["SCT_DOM"]
        vegetation=static["LU_INDEX"]
        assert np.equal(soil,np.rint(soil)).all() and np.equal(vegetation,np.rint(vegetation)).all()
        soil=soil.astype(np.int32)
        vegetation=vegetation.astype(np.int32)
    receipt={"prepared_root":str(root),"header":str(path),"array_records":records,
        "smois_min":float(np.min(moisture)),"negative_values":int(np.count_nonzero(moisture<0)),
        "below_floor_values":int(np.count_nonzero(moisture<np.float32(.005))),
        "negative_columns":int(np.count_nonzero((moisture<0).any(axis=0))),
        "smois_sha256":hashlib.sha256(moisture.tobytes()).hexdigest()}
    receipt["smois_values"]=int(moisture.size)
    try:
        initialized=ruc_initialize_cold_start(temperature,moisture,soil,vegetation,read("surface/SEAICE"))
        receipt["cold_start"]="passed"
        receipt["cold_start_output_fields"] = sorted(vars(initialized))
    except ValueError as error:
        receipt["cold_start"]="refused"
        receipt["cold_start_error"]=str(error)
    out.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:v for k,v in receipt.items() if k!="array_records"},sort_keys=True))
    assert receipt["cold_start"]==expected,receipt


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=["configure","inspect"])
    parser.add_argument("root",type=Path)
    parser.add_argument("--out",type=Path)
    parser.add_argument("--expected",choices=["passed","refused"])
    parser.add_argument("--floor-input",type=Path)
    args=parser.parse_args()
    if args.mode=="configure": configure(args.root)
    else: inspect(args.root,args.out,args.expected,args.floor_input)
