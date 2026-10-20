"""Exact native hourly forcing -> PET sequence, with Rust packet/capture I/O.

Additional CSR/bounds checks exercise the rain-baseline and window boundaries;
they are engineering tests, not native physical acceptance evidence.
"""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import time


def main():
    parser=argparse.ArgumentParser()
    for name in ('manifest','rust-io','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ('native-forcing','native-pet'):
        parser.add_argument('--'+name,type=Path,nargs='+',required=True)
    parser.add_argument('--pet-options',type=Path)
    parser.add_argument('--native-cfe-entry',type=Path,nargs='+')
    args=parser.parse_args()
    import cupy as cp
    import numpy as np
    from gpuwm import numerics
    from gpuwm.hydro.forcing import FIELDS, PreparedCatchmentForcing
    from gpuwm.hydro.forcing_inputs import load_series
    from gpuwm.hydro.pet import PreparedPET
    start=time.perf_counter()
    lib=ctypes.CDLL(str(args.rust_io))
    fn=lib.arwen_forcing_read_f64
    fn.argtypes=[ctypes.c_char_p,ctypes.POINTER(ctypes.c_double),ctypes.c_size_t]
    fn.restype=ctypes.c_int
    def read(path,width):
        count=path.stat().st_size//8
        storage=(ctypes.c_double*count)()
        if fn(bytes(path),storage,count):raise RuntimeError('Rust native capture read failed')
        return cp.asarray(np.ctypeslib.as_array(storage)).reshape(-1,width)
    native=cp.stack([read(path,8) for path in args.native_forcing],axis=2)
    pet_native=cp.stack([read(path,39) for path in args.native_pet],axis=1)
    nt,_,nc=native.shape
    if pet_native.shape[:2]!=(nt,nc):raise ValueError('native capture dimensions differ')
    forcing=load_series(args.manifest,library=args.rust_io,native_precision=True)
    if args.pet_options:
        from gpuwm.hydro.pet import prepare
        pet=prepare(settings=None,options=json.loads(args.pet_options.read_text()),state=None,
                    physics=None,device_id=0,mode='standalone',components={'forcing':forcing})
    else:
        pet=PreparedPET(forcing.fields,pet_native[0,:,:12].copy(),identity={'case':'native hourly sequence'})
    initial_parameter_different=int(cp.count_nonzero(pet.arrays['parameters'].view(cp.uint64)!=pet_native[0,:,:12].copy().view(cp.uint64)).item())
    history=cp.empty((nt,8,nc),dtype=cp.float64)
    pet_history=cp.empty((nt,nc),dtype=cp.float64)
    # Bind output views before stepping; output-boundary recording is explicit.
    forcing_rows=tuple(history[k] for k in range(nt))
    pet_rows=tuple(pet_history[k] for k in range(nt))
    begin,end=cp.cuda.Event(),cp.cuda.Event()
    begin.record()
    for frow,prow in zip(forcing_rows,pet_rows):
        forcing.advance(3600.)
        pet.advance(3600.)
        cp.copyto(frow,forcing.arrays['fields'])
        cp.copyto(prow,pet.fields['potential_et_m_s'])
    end.record();end.synchronize()
    forcing_different=int(cp.count_nonzero(history.view(cp.uint64)!=native.view(cp.uint64)).item())
    pet_different=int(cp.count_nonzero(pet_history.view(cp.uint64)!=pet_native[:,:,32].copy().view(cp.uint64)).item())
    pet_ulp=int(cp.abs(pet_history.view(cp.int64)-pet_native[:,:,32].copy().view(cp.int64)).max().item())
    # Keep the physical short-case scope explicit instead of inferring it
    # from a full-run mismatch count. All comparisons execute on CUDA.
    pet_delta=pet_history.view(cp.int64)-pet_native[:,:,32].copy().view(cp.int64)
    first48_different=int(cp.count_nonzero(pet_delta[:48]).item())
    mismatches=[]
    for flat in cp.flatnonzero(pet_delta)[:8].get().tolist():
        step,catchment=divmod(flat,nc)
        mismatches.append({'step':step,'catchment':forcing.restart_identity['identity']['catchment_ids'][catchment],
                           'ulps':int(abs(pet_delta[step,catchment]).item()),
                           'gpu_hex':float(pet_history[step,catchment].item()).hex(),
                           'native_hex':float(pet_native[step,catchment,32].item()).hex()})
    final_parameter_different=int(cp.count_nonzero(pet.arrays['parameters'].view(cp.uint64)!=pet_native[-1,:,20:32].copy().view(cp.uint64)).item())
    row1={'gpu_words':[hex(v) for v in pet_history[1].view(cp.uint64).get().tolist()],
          'native_words':[hex(v) for v in pet_native[1,:,32].copy().view(cp.uint64).get().tolist()]}
    cfe_boundary=[]
    if args.native_cfe_entry:
        if len(args.native_cfe_entry)!=nc:raise ValueError('one CFE entry per catchment required')
        read_bytes=lib.arwen_forcing_read_packet
        read_bytes.argtypes=[ctypes.c_char_p,ctypes.c_char_p,ctypes.POINTER(ctypes.c_ubyte),ctypes.c_size_t]
        read_bytes.restype=ctypes.c_int
        for c,path in enumerate(args.native_cfe_entry):
            schema={row['name']:row for row in map(json.loads,path.with_suffix('.jsonl').read_text().splitlines())}
            rain=schema['input.atmosphere_water__liquid_equivalent_precipitation_rate']
            clock=schema['state.current_time_step']
            if rain['units']!='mm h-1' or rain['itemsize']!=8 or clock['itemsize']!=4:
                raise ValueError('CFE native witness units/layout differ')
            size=path.stat().st_size;storage=(ctypes.c_ubyte*size)()
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            if read_bytes(bytes(path),digest.encode(),storage,size):raise RuntimeError('Rust CFE entry read failed')
            raw=cp.asarray(np.ndarray((size,),dtype='uint8',buffer=storage))
            native_rain=raw[rain['offset']:rain['offset']+8].view(cp.float64)
            native_step=raw[clock['offset']:clock['offset']+4].view(cp.int32)
            converted=history[native_step,7,c]*3600.0
            # REFEREE-ONLY witness, measured from pinned ngen UnitsHelper:
            # UDUNITS represents the mm/s -> mm/h multiplier one ULP above
            # exact 3600. The canonical product forcing remains mm/s.
            native_adapter=history[native_step,7,c]*float.fromhex('0x1.c200000000001p+11')
            cfe_boundary.append({'catchment':forcing.restart_identity['identity']['catchment_ids'][c],
                'capture':str(path),'sha256':digest,'native_step':int(native_step.item()),
                'native_mm_h':float(native_rain.item()),'canonical_mm_s':float(history[native_step,7,c].item()),
                'exact_3600_different_words':int(cp.count_nonzero(converted.view(cp.uint64)!=native_rain.view(cp.uint64)).item()),
                'native_udunits_multiplier_hex':'0x1.c200000000001p+11',
                'different_words':int(cp.count_nonzero(native_adapter.view(cp.uint64)!=native_rain.view(cp.uint64)).item())})
    sequence_ms=cp.cuda.get_elapsed_time(begin,end)
    forcing.advance(1.)
    cp.cuda.get_current_stream().synchronize()
    bounds_ok=bool((forcing.arrays['status']==1).all().item()) and bool(cp.isnan(forcing.arrays['fields']).all().item())
    # Fresh first-interval state; separate profiling launches, outside history
    # measurement. These event intervals include Python enqueue gaps.
    forcing.arrays['clock_seconds'].fill(0)
    cp.copyto(pet.arrays['parameters'],pet_native[0,:,:12])
    profile=tuple(cp.cuda.Event() for _ in range(3))
    profile[0].record();forcing.advance(3600.);profile[1].record()
    pet.advance(3600.);profile[2].record();profile[2].synchronize()
    stage_ms={'forcing':cp.cuda.get_elapsed_time(profile[0],profile[1]),
              'pet':cp.cuda.get_elapsed_time(profile[1],profile[2]),
              'scope':'fresh first interval, one separate event profile; shared pool, includes enqueue gaps'}
    # Shared parent cells appear in two catchments: previous rain is per edge.
    fields={key:cp.full(3,10.+i,dtype=cp.float32) for i,key in enumerate(FIELDS[:7])}
    offsets=cp.asarray([0,2,4],dtype=cp.uint64)
    indices=cp.asarray([0,1,1,2],dtype=cp.uint32)
    weights=cp.asarray([.25,.75,.5,.5],dtype=cp.float64)
    rain=cp.asarray([12.,24.,40.],dtype=cp.float32);zero=cp.zeros(3,dtype=cp.float32)
    baseline=cp.asarray([10.,20.,30.],dtype=cp.float32)
    area=PreparedCatchmentForcing(fields,offsets,indices,weights,identity={'test':'overlapping-cells'},time_origin='2000-01-01T00:00:00Z',rain_accumulators=(rain,zero),rain_baseline=(baseline,zero))
    area.advance(2.)
    expected=cp.asarray([1.75,3.5],dtype=cp.float32)
    first=bool((area.fields['RAINRATE']==expected).all().item())
    saved=area.arrays['previous_rain_mm'].copy()
    rain+=cp.asarray([2.,4.,10.],dtype=cp.float32)
    area.advance(2.)
    second=bool((area.fields['RAINRATE']==expected).all().item())
    restored=PreparedCatchmentForcing(fields,offsets,indices,weights,identity={'test':'overlapping-cells'},time_origin='2000-01-01T00:00:00Z',rain_accumulators=(rain,zero),rain_baseline=(baseline,zero))
    cp.copyto(restored.arrays['previous_rain_mm'],saved)
    restored.advance(2.)
    restart=bool((restored.fields['RAINRATE']==area.fields['RAINRATE']).all().item())
    rain.fill(0.)
    area.advance(2.)
    reset_rejected=bool((area.arrays['status']==1).all().item()) and bool(cp.isnan(area.fields['RAINRATE']).all().item())
    receipt={'case':'pinned ngen hourly native provider -> native PET BMI inputs','catchment_ids':forcing.restart_identity['identity']['catchment_ids'],'steps':nt,'catchments':nc,'pet_configuration':pet.restart_identity, 'card':cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),'numerics':numerics.mode(),'forcing_words':nt*8*nc,'forcing_different_words':forcing_different,'sequence_pet_different_words':pet_different,'sequence_pet_max_ulp':pet_ulp,'initial_parameter_different_words':initial_parameter_different,'final_parameter_different_words':final_parameter_different,'row1':row1,'sequence_gpu_ms_with_two_output_copies':sequence_ms,'parallelism':f'{nc} catchments,128 threads/block,2 module kernels/step,2 explicit history copies/step,shared pool,no graphs','bounds_rejected':bounds_ok,'overlap_rain_baseline':first and second,'rain_restart':restart,'rain_reset_rejected':reset_rejected,'total_seconds':time.perf_counter()-start,'physical_runoff_routing_acceptance':False}
    receipt.update(native_cfe_rain_boundary=cfe_boundary,separate_stage_profile_ms=stage_ms,
                   first48_pet_different_words=first48_different,pet_mismatch_samples=mismatches)
    args.out.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))
    if forcing_different or initial_parameter_different or final_parameter_different or any(row['different_words'] for row in cfe_boundary) or (numerics.is_strict() and pet_different) or not all((bounds_ok,first,second,restart,reset_rejected)):raise SystemExit(1)


if __name__=='__main__':main()
