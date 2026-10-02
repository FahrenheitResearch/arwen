"""Measure every packaged coupling output word through engine launchers."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from gpuwm.verify.bigstep_coupling_oracle import (coupling_cases,
    load_coupling_oracle, coupling_port_outputs, word_measurement,
    coupling_wrf_flux_trace)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("directory",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    reference=load_coupling_oracle(a.directory)
    measurements={}
    actuals={}
    for case in coupling_cases(a.directory):
        name=case["name"]
        actual=coupling_port_outputs(case)
        measurements[name]={}
        for key,value in actual.items():
            expected=reference[name+"/"+key]
            if value.shape != expected.shape:
                raise ValueError((name,key,value.shape,expected.shape))
            measurements[name][key]=word_measurement(value,expected)
            actuals[name+"/"+key]=value
        trace=coupling_wrf_flux_trace(case)
        measurements[name]["ww_wrf_flux_trace"]=word_measurement(trace,reference[name+"/ww"])
    a.output.write_text(json.dumps(measurements,indent=2)+"\n")
    np.savez_compressed(a.output.with_suffix(".npz"),**actuals)
    print(json.dumps(measurements,indent=2))


if __name__=="__main__":
    main()
