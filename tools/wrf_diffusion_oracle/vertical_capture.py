"""Capture direct WRF vertical/TKE words on real state crops and edges."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from cases import raw_state, state_cases, core, pad2, pad3
from deformation_reference import reference_for_case
from vertical_reference import vertical_reference, tke_reference,vertical_driver_reference


def _portable_build(path):
    receipt = json.loads(Path(path).read_text())
    receipt["commands"] = [[Path(arg).name if Path(arg).is_absolute() else arg
                            for arg in command] for command in receipt["commands"]]
    return receipt


def capture(source,reader,deformation_library,vertical_library,output,evolved=None):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    raw,raw_metadata = raw_state(source,reader,output/"decoded")
    index = []
    state_sets = [(raw,raw_metadata,"")]
    if evolved is not None:
        from vertical_evolved import evolved_flow
        winds,flow_metadata = evolved_flow(evolved,reader,output/"decoded-flow")
        evolved_raw = {**raw,**winds}
        state_sets.append((evolved_raw,{**raw_metadata,**flow_metadata},"evolved_"))
    iterator = ((prefix+name,arrays,meta) for fields,metadata,prefix in state_sets
                for name,arrays,meta in state_cases(fields,metadata))
    for name,arrays,meta in iterator:
        nz,ny,nx = arrays["alt"].shape
        level = np.arange(nz,dtype=np.float32)[:,None,None]
        arrays["tke"] = np.broadcast_to(.5*np.exp(-level/15.),(nz,ny,nx)).astype(np.float32).copy()
        arrays["tke"][:,0,:2] = 0.
        if name.endswith("zero_flow") and not name.endswith("near_zero_flow"): arrays["tke"].fill(0.)
        if name.endswith("near_zero_flow"): arrays["tke"] *= np.float32(1.e-20)
        arrays["ust"] = np.full((ny,nx),.35,dtype=np.float32)
        arrays["hfx"] = np.full((ny,nx),120.,dtype=np.float32)
        arrays["qfx"] = np.full((ny,nx),2.e-5,dtype=np.float32)
        meta.update(dt=12.,c_k=.15,seed_on=0)
        reference = reference_for_case(deformation_library,arrays,meta,km_opt=2,isotropic=0)
        for key in ("ust","hfx","qfx"):
            reference[key] = pad2(arrays[key],nx,ny,meta["bx"],meta["by"])
        reference["qv"] = np.asfortranarray(reference["moist"][:,:,:,1])
        # These leaves use K as a caller input. Keeping the compiled WRF K
        # words isolates diffusion from differences in the closure itself.
        vertical = vertical_reference(vertical_library,reference,reference,meta,
            kmv=reference["kmv"],khv=reference["khv"],var=reference["qv"])
        extra = {}
        for scalar,var,khv,is_tke in (
            ("theta",reference["thp"],reference["khv"],False),
            ("tke",reference["tke"],reference["kmv"],True)):
            result = vertical_reference(vertical_library,reference,reference,meta,
                kmv=reference["kmv"],khv=khv,var=var,tke=reference["tke"],doing_tke=is_tke)
            extra["vertical_s_"+scalar] = result["vertical_s"]
        payload = {"in_"+k:v for k,v in arrays.items()}
        payload.update({"ref_"+k:v for k,v in vertical.items()})
        payload.update({"ref_"+k:v for k,v in extra.items()})
        payload.update({"coefficient_"+k:core(reference[k],nx,ny,nz)
                        for k in ("kmh","kmv","khv")})
        payload.update({"deformation_"+k:core(reference[k],nx,ny,nz)
                        for k in ("d11","d22","d12")})
        payload.update({"metric_"+k:core(reference[k],nx,ny,nz,"z" if k in ("rdz","zx","zy") else "")
                        for k in ("rdz","rdzw","rho","zx","zy")})
        payload["bn2"] = core(reference["bn2"],nx,ny,nz)
        # Surface switches change the RHS, but not the leaf coefficient
        # arguments. Call each WRF term with those identical K inputs.
        for isfflx in (0,1,2):
            terms = tke_reference(vertical_library,reference,reference,reference,meta,isfflx=isfflx,
                                  c_k=meta["c_k"],dt=meta["dt"])
            payload.update({f"ref_{k}_flux{isfflx}":v for k,v in terms.items()})
        # Composite surface forcing is measured on periodic domains. Open
        # rows are independently measured by all four leaves and TKE terms;
        # the whole engine package applies an additional open-row mask.
        if not meta["bx"] and not meta["by"]:
            for km_opt in (2,4):
                driver = reference if km_opt==2 else reference_for_case(deformation_library,arrays,meta,km_opt=4,isotropic=0)
                for key in ("ust","hfx","qfx"):
                    driver[key] = pad2(arrays[key],nx,ny,meta["bx"],meta["by"])
                for key in ("kmh","kmv","khv"):
                    payload[f"driver{km_opt}_coefficient_{key}"] = core(driver[key],nx,ny,nz)
                for isfflx in (0,1,2):
                    outputs = vertical_driver_reference(vertical_library,driver,meta,km_opt=km_opt,isfflx=isfflx)
                    payload.update({f"ref_driver{km_opt}_{k}_flux{isfflx}":v for k,v in outputs.items()})
        path = output/f"vertical-{name}.npz"
        np.savez_compressed(path,**payload)
        index.append({"file":path.name,"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                      "case":name,"metadata":meta,"array_names":sorted(payload),
                      "tke_input":"coherent 0.5 J/kg surface profile tapering with level; zero patch",
                      "surface_input":"UST 0.35 m/s; HFX 120 W/m2; QFX 2e-5 kg/m2/s"})
    receipt = {"wrf_release":"4.7.1","cases":index,
               "fortran_build":_portable_build(Path(vertical_library).parent/"build-receipt.json"),
               "preparation_build":_portable_build(Path(deformation_library).parent/"build-receipt.json"),
               "deformation_library_sha256":hashlib.sha256(Path(deformation_library).read_bytes()).hexdigest(),
               "vertical_library_sha256":hashlib.sha256(Path(vertical_library).read_bytes()).hexdigest(),
               "source_sha256":raw_metadata["source_sha256"],"reader_sha256":raw_metadata["reader_sha256"]}
    (output/"vertical-fixtures.json").write_text(json.dumps(receipt,indent=2)+"\n", encoding="utf-8", newline="\n")
    return receipt


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("source","reader","deformation_library","vertical_library","output"):
        parser.add_argument(key,type=Path)
    parser.add_argument("--evolved",type=Path)
    args = parser.parse_args()
    receipt = capture(**vars(args))
    print(json.dumps({"cases":len(receipt["cases"]),"source_sha256":receipt["source_sha256"]}))
