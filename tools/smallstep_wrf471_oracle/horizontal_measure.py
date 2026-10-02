"""Run the engine explicit driver and write every-output WRF word metrics."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

from gpuwm.verify.smallstep_oracle import load_cases
from gpuwm.verify.smallstep_horizontal_oracle import write_horizontal_receipt


def horizontal_cases():
    for name,raw,meta in load_cases():
        yield name+"_map",raw,meta,dict(boundary="specified",map_factors=True)
        yield name+"_flat",raw,meta,dict(boundary="specified",map_factors=False)
        if name=="zero_near_zero":
            yield name+"_no_fma_probe",raw,meta,dict(boundary="specified",no_fma_probe=True)
        if name=="real_initial":
            yield name+"_moist_map",raw,meta,dict(boundary="specified",map_factors=True,moist_loading=True)
            yield name+"_moist_flat",raw,meta,dict(boundary="specified",map_factors=False,moist_loading=True)
        if name=="map_extremes":
            yield name+"_moist_map",raw,meta,dict(boundary="specified",map_factors=True,moist_loading=True)
    name,raw,meta=next(iter(load_cases()))
    for suffix,opts in (
        ("periodic",dict(boundary="periodic")),
        ("open",dict(boundary="open")),
        ("top_lid",dict(boundary="specified",top_lid=True)),
        ("zero_reference",dict(boundary="specified",mode="zero-reference")),
        ("later_substep",dict(boundary="specified",first=False)),
        ("full_theta_probe",dict(boundary="specified",mode="zero-reference",full_theta_probe=True)),
        ("no_fma_probe",dict(boundary="specified",no_fma_probe=True)),
        ("full_theta_no_fma_probe",dict(boundary="specified",mode="zero-reference",full_theta_probe=True,no_fma_probe=True)),
    ):
        yield name+"_"+suffix,raw,meta,opts


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--library",default=os.environ.get("GPUWM_SMALLSTEP_ORACLE_LIB"))
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--fixture",type=Path)
    a=p.parse_args()
    result=write_horizontal_receipt(a.output,horizontal_cases(),a.library,a.fixture)
    print(json.dumps({name:{key:r["max_ulp"] for key,r in row.items()} for name,row in result.items()},indent=2))
