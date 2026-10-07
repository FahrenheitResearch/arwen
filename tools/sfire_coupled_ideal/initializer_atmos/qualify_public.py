"""Actual public ideal CLI, native history and fresh-process checkpoint controls."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configuration(extra="", *, noah=False):
    return f"""[grid]
nx = 12
ny = 10
nz = 6
dx = 73.5
dy = 90.0
ztop = 4000.0
[dynamics]
dt = 0.5
moist = true
open_x = true
open_y = true
h_sca_adv_order = 5
sf_surface_physics = {2 if noah else 0}
sf_sfclay_physics = {1 if noah else 0}
num_soil_layers = {4 if noah else 5}
[run]
run_seconds = 4.0
output_interval_s = 1.0
restart_interval_s = 2.0
[fire]
ifire = 2
sr_x = 3
sr_y = 4
fire_fuel_read = 0
fire_fmc_read = 0
fire_fuel_cat = 3
fire_boundary_guard = 2
fire_smoke = true
fmoist_run = true
fmoist_interp = true
fmoist_freq = 1
fire_num_ignitions = 1
fire_ignition_start_x1 = 367.5
fire_ignition_end_x1 = 367.5
fire_ignition_start_y1 = 450.0
fire_ignition_end_y1 = 450.0
fire_ignition_radius1 = 45.0
fire_ignition_start_time1 = 0.0
fire_ignition_end_time1 = 0.0
fire_ignition_ros1 = 100.0
{extra}"""


def command(root, config, sounding, name, extra=()):
    destination=root/name
    argv=[sys.executable,"-m","gpuwm","fire-ideal",str(config),"--sounding",str(sounding),
          "--input-directory",str(config.parent),"--outdir",str(destination),*extra]
    with (root/(name+".log")).open("w") as log:
        subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,check=True)
    receipt=json.loads((destination/"run-receipt.json").read_text())
    assert receipt["completed"] and receipt["health"]["nan"] is False,receipt
    return dict(argv=argv,receipt=receipt,receipt_sha256=sha(destination/"run-receipt.json"),
                log_sha256=sha(root/(name+".log"))),destination


def history(directory):
    from gpuwm import netcdf_bridge
    result={}
    latest=sorted(directory.glob("wrfout*"))[-1]
    with netcdf_bridge.open_dataset(latest) as ds:
        for name,var in ds.variables.items():
            var.set_auto_maskandscale(False)
            result[name]=np.asarray(var[:],dtype=var.dtype)
    return result,latest


def compare_arrays(first, second):
    assert set(first)==set(second),sorted(set(first)^set(second))
    checks={}
    for name,a in first.items():
        b=second[name]
        assert a.shape==b.shape and a.dtype==b.dtype,(name,a.shape,b.shape,a.dtype,b.dtype)
        assert a.tobytes()==b.tobytes(),name
        checks[name]=dict(shape=list(a.shape),dtype=str(a.dtype),bytes=a.nbytes,sha256=hashlib.sha256(a.tobytes()).hexdigest())
    return checks


def supplied_configuration(directory, *, modeled_moisture=False):
    """Nine constant test providers in the native fixed REAL record format."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    for filename,nx,ny,value in (("input_lu",13,11,"5"),("input_tsk",13,11,"296.5"),
        ("input_tmn",13,11,"289.75"),("input_ht",13,11,"35.5"),
        ("input_zsf",39,44,"35.5"),("input_dzdxf",39,44,"0.0125"),
        ("input_dzdyf",39,44,"-0.025"),("input_fc",39,44,"3"),
        ("input_fmc_g",39,44,"0.125")):
        (directory/filename).write_text(f"{nx} {ny}\n"+(" ".join([value]*ny)+"\n")*nx)
    text=configuration("""sfc_full_init = true
sfc_lu_index = 5
sfc_vegfra = 0.75
sfc_canwat = 0.0625
sfc_ivgtyp = 18
sfc_isltyp = 7
fire_read_lu = true
fire_read_tsk = true
fire_read_tmn = true
fire_read_atm_ht = true
fire_read_fire_ht = true
fire_read_fire_grad = true
""",noah=True).replace("fire_fuel_read = 0","fire_fuel_read = 2")
    if not modeled_moisture:
        text=text.replace("fire_fmc_read = 0","fire_fmc_read = 2").replace("fmoist_run = true","fmoist_run = false").replace("fmoist_interp = true","fmoist_interp = false").replace("fmoist_freq = 1","fmoist_freq = 0")
    path=directory/"provided-fields.toml";path.write_text(text);return path


def initial_canopy_control(config,sounding):
    """Grade the actual native input consumer even when CANWAT is not history."""
    import cupy as cp
    from gpuwm.config import load_config
    from dataclasses import replace
    from gpuwm.static.sfire import read_fire_ideal_inputs,fire_ideal_input_requests
    from gpuwm.core.sfire_ideal import build_state
    from gpuwm.core.physics import initialize_physics
    cfg=replace(load_config(config,native_fire_ideal=True),terrain_opt=1,hypsometric_opt=1)
    parsed=read_fire_ideal_inputs(config.parent,fire_ideal_input_requests(cfg),sounding=sounding)
    state,fire,surface,_,_=build_state(cfg,parsed['sounding'],parsed['fields'])
    driver=initialize_physics(state,cfg,fire_static_data=fire,**surface)
    values=cp.asnumpy(driver.fields['canwat'])
    assert values.tobytes()==np.full(values.shape,np.float32(cfg.sfc_canwat),np.float32).tobytes()
    return dict(words=values.size,different_words=0,expected=float(np.float32(cfg.sfc_canwat)),
                dtype=str(values.dtype),sha256=hashlib.sha256(values.tobytes()).hexdigest())


def main():
    parser=argparse.ArgumentParser();parser.add_argument("root",type=Path)
    parser.add_argument("--sounding",type=Path,required=True)
    parser.add_argument("--full-surface",action="store_true")
    parser.add_argument("--demo",type=Path)
    args=parser.parse_args();root=args.root.resolve();root.mkdir(parents=True,exist_ok=True)
    inputs=root/"inputs";inputs.mkdir(exist_ok=True)
    sounding=inputs/"input_sounding";shutil.copyfile(args.sounding,sounding)
    configs={"normal":configuration(),"mountain-bubble":configuration("""fire_mountain_type = 1
fire_mountain_height = 80.0
fire_mountain_start_x = 50.0
fire_mountain_end_x = 750.0
fire_mountain_start_y = -100.0
fire_mountain_end_y = 750.0
delt_perturbation = 1.5
xrad_perturbation = 230.0
yrad_perturbation = 260.0
zrad_perturbation = 900.0
hght_perturbation = 700.0
stretch_hyp = true
""").replace("[dynamics]\n", "[dynamics]\nhybrid_opt = 3\n")}
    results={};outputs={}
    if args.demo is not None:
        demo=args.demo.resolve()
        results["permanent-demo"],outputs["permanent-demo"]=command(root,demo/"fire.toml",demo/"input_sounding","permanent-demo")
    for name,text in configs.items():
        path=inputs/(name+".toml");path.write_text(text)
        results[name],outputs[name]=command(root,path,sounding,name)
    native=inputs/'namelist.input'
    shutil.copyfile(Path(__file__).parent/'reference/native_public_namelist.input',native)
    results['native-namelist'],_=command(root,native,sounding,'native-namelist',('--namelist',))
    assert results['native-namelist']['receipt']['step_count']==3
    if args.full_surface:
        path=supplied_configuration(inputs)
        initial_canopy=initial_canopy_control(path,sounding)
        results["provided-fields"],provided=command(root,path,sounding,"provided-fields")
        results["provided-fields"]["initial_canopy_control"]=initial_canopy
        from gpuwm import netcdf_bridge
        first=sorted(provided.glob("wrfout*"))[0]
        with netcdf_bridge.open_dataset(first) as ds:
            for name,value in (("TSK",296.5),("TMN",289.75),
                               ("LU_INDEX",5),("IVGTYP",5),("HGT",35.5)):
                assert np.all(np.asarray(ds.variables[name][:])==np.float32(value)),name
        assert len(results["provided-fields"]["receipt"]["inputs"]["native_fields"])==9
        modeled=inputs/"modeled";modeled.mkdir()
        path=supplied_configuration(modeled,modeled_moisture=True)
        modeled_sounding=modeled/"input_sounding";shutil.copyfile(sounding,modeled_sounding)
        results["provided-modeled-moisture"],_=command(root,path,modeled_sounding,"provided-modeled-moisture")
        assert len(results["provided-modeled-moisture"]["receipt"]["inputs"]["native_fields"])==8
    cfg=inputs/"mountain-bubble.toml"
    results["prefix"],prefix=command(root,cfg,sounding,"prefix",("--run-seconds","2"))
    checkpoint=sorted(prefix.glob("gpuwmrst*"))[-1]
    results["resumed"],resumed=command(root,cfg,sounding,"resumed",("--restart",str(checkpoint)))
    full_fields,full_history=history(outputs["mountain-bubble"]);continued_fields,resumed_history=history(resumed)
    history_checks=compare_arrays(full_fields,continued_fields)
    key="__gpuwm_restart_header__"
    def payload(path):
        with np.load(path,allow_pickle=False) as data:arrays={name:data[name] for name in data.files}
        header=json.loads(arrays.pop(key).tobytes());return arrays,header
    full_checkpoint=sorted(outputs["mountain-bubble"].glob("gpuwmrst*"))[-1]
    resumed_checkpoint=sorted(resumed.glob("gpuwmrst*"))[-1]
    full_arrays,full_header=payload(full_checkpoint);resumed_arrays,resumed_header=payload(resumed_checkpoint)
    checkpoint_checks=compare_arrays(full_arrays,resumed_arrays)
    for field in ("elapsed_seconds","setup_fingerprint","physics_setup_fingerprint","physics_setup","driver","array_manifest","run_trackers"):
        assert full_header[field]==resumed_header[field],field
    assert np.any(full_fields["FIRE_AREA"]>0) and np.any(full_fields["FGRNHFX"]>0)
    assert np.any(full_fields["fire_smoke"]>0)
    receipt=dict(schema="gpuwm-public-sfire-ideal-qualification-v1",runs=results,
        history=dict(reference=str(full_history),continued=str(resumed_history),fields=history_checks),
        checkpoint=dict(reference=str(full_checkpoint),continued=str(resumed_checkpoint),arrays=checkpoint_checks),
        compared_history_fields=len(history_checks),compared_checkpoint_arrays=len(checkpoint_checks),
        compared_bytes=sum(x["bytes"] for x in history_checks.values())+sum(x["bytes"] for x in checkpoint_checks.values()),
        different_bytes=0,active_fire=True,active_heat=True,active_native_bulk_smoke=True)
    (root/"qualification.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps({k:receipt[k] for k in ("compared_history_fields","compared_checkpoint_arrays","compared_bytes","different_bytes")}))


if __name__=="__main__":main()
