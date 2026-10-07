"""Grade the same native controls as the focused suite and record every word."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys

import numpy as np


def grade(destination):
    import cupy as cp
    import gpuwm.core.fp32_ulp as ulp
    from gpuwm.core import sfire_spotting
    from tools.sfire_wrf471_oracle.fixture import words
    engine=Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(engine/"tests"))
    spec=importlib.util.spec_from_file_location("native_spotting_controls",engine/"tests/test_sfire_spotting_wrf471_parity.py")
    controls=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(controls)
    original_float=ulp.assert_bit_exact
    original_equal=np.array_equal
    records=[]
    current=""
    def capture_float(actual,expected,name="",**kwargs):
        result=original_float(actual,expected,name,**kwargs)
        records.append(dict(case=current,output=name,dtype="float32",**words(actual,expected)))
        return result
    def capture_integer(actual,expected,*args,**kwargs):
        result=original_equal(actual,expected,*args,**kwargs)
        a,b=np.asarray(actual),np.asarray(expected)
        if a.dtype==b.dtype==np.dtype(np.int32) and a.shape==b.shape:
            records.append(dict(case=current,output="integer identifiers or clocks",dtype="int32",words=a.size,
                different_words=int(np.count_nonzero(a.view(np.uint32)!=b.view(np.uint32))),max_ulp=0,nonfinite_differences=0))
        return result
    ulp.assert_bit_exact=capture_float
    np.array_equal=capture_integer
    jobs=[(controls.test_initial_firebrand_properties_match_native_wrf,()),
          (controls.test_initial_firebrand_mass_matches_cubic_volume_control,()),
          (controls.test_particle_box_interpolation_matches_native_wrf,()),
          (controls.test_particle_meteorology_sampling_matches_native_wrf,()),
          (controls.test_approximate_release_rank_matches_native_wrf,()),
          (controls.test_eight_neighbor_packets_match_actual_compiled_wrf_mpi_exchange,())]
    jobs.extend((controls.test_firebrand_physics_matches_native_wrf,(case,mode)) for mode in ("burnout","termvel") for case in range(1,5))
    jobs.extend((controls.test_particle_advection_matches_native_wrf,(case,)) for case in range(1,4))
    jobs.extend((controls.test_release_height_distribution_matches_native_wrf,(case,)) for case in range(1,7))
    jobs.extend((controls.test_packed_particle_generation_matches_native_wrf,(case,)) for case in range(1,7))
    jobs.extend((controls.test_particle_column_preparation_matches_corrected_native_driver,(case,)) for case in range(1,5))
    jobs.extend((controls.test_complete_firebrand_driver_matches_corrected_native_wrf,(case,mapped))
                for mapped in (False,True) for case in range(1,5))
    jobs.extend((controls.test_compact_atmosphere_preparation_matches_native_physical_columns,(case,profiles)) for profiles in (False,True) for case in range(1,4))
    try:
        for method,args in jobs:
            current=method.__name__+str(args)
            method(*args)
    finally:
        ulp.assert_bit_exact=original_float
        np.array_equal=original_equal
    summary=dict(checks=len(jobs),coupled_native_frames=96,graded_words=sum(v["words"] for v in records),
        float32_words=sum(v["words"] for v in records if v["dtype"]=="float32"),
        int32_words=sum(v["words"] for v in records if v["dtype"]=="int32"),
        different_words=sum(v["different_words"] for v in records),max_ulp=max(v["max_ulp"] for v in records))
    root=Path(__file__).parent
    sources=("gpuwm/core/sfire_spotting.py","gpuwm/core/kernels/sfire_spotting.cu",
             "gpuwm/core/kernels/glibc_flt32.cuh","gpuwm/core/kernels/glibc_flt64.cuh",
             "tests/test_sfire_spotting_wrf471_parity.py")
    receipt=dict(reference="original byte-extracted firebrand helpers and explicitly corrected complete original WRF driver",
        device=cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"].decode(),
        summary=summary,module_options=sfire_spotting.MODULE_OPTIONS,
        composition_sha256=hashlib.sha256(sfire_spotting.module_source().encode()).hexdigest(),
        source_sha256={name:hashlib.sha256((engine/name).read_bytes()).hexdigest() for name in sources},
        corpus_sha256={name:hashlib.sha256((root/"fixtures"/name/"receipt.json").read_bytes()).hexdigest()
                       for name in ("spotting","spotting_driver","spotting_mpi")},records=records)
    Path(destination).write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(summary))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("destination")
    grade(p.parse_args().destination)
