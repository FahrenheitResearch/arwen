"""Capacity timing on actual case(b) CSR, with explicitly imposed native weather.

This is not an event simulation: a captured native meteorological frame is
uniformly imposed on the actual parent-grid geometry. Native equivalence is
established separately by the recorded module replays.
"""
import argparse
import ctypes
import json
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser()
    for key in ('fabric','native-pet','rust-io','out'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    import cupy as cp
    import numpy as np
    from gpuwm import numerics
    from gpuwm.hydro.forcing import PreparedCatchmentForcing,FIELDS
    from gpuwm.hydro.pet import PreparedPET
    start=time.perf_counter()
    lib=ctypes.CDLL(str(a.rust_io))
    read=lib.arwen_forcing_read_packet
    read.argtypes=[ctypes.c_char_p,ctypes.c_char_p,ctypes.POINTER(ctypes.c_ubyte),ctypes.c_size_t]
    read.restype=ctypes.c_int
    meta=json.loads(a.fabric.read_text())
    def array(name):
        f=meta['files'][name];storage=(ctypes.c_ubyte*f['bytes'])()
        if read(bytes(a.fabric.parent/f['path']),f['sha256'].encode(),storage,f['bytes']):raise RuntimeError('Rust fabric hash/read failed')
        return cp.asarray(np.ndarray(tuple(f['shape']),dtype=f['dtype'],buffer=storage))
    offsets,indices,weights=(array(key) for key in ('weights_offsets','weights_indices','weights'))
    fn=lib.arwen_forcing_read_f64
    fn.argtypes=[ctypes.c_char_p,ctypes.POINTER(ctypes.c_double),ctypes.c_size_t]
    count=a.native_pet.stat().st_size//8
    storage=(ctypes.c_double*count)()
    if fn(bytes(a.native_pet),storage,count):raise RuntimeError('Rust native capture read failed')
    record=cp.asarray(np.ctypeslib.as_array(storage)).reshape(-1,39)
    shape=(582,737)
    parent={key:cp.broadcast_to(record[0,12+f].astype(cp.float32),shape).copy() for f,key in enumerate(FIELDS)}
    forcing=PreparedCatchmentForcing(parent,offsets,indices,weights,identity={'fabric':meta['request_sha256'],'weather':'uniform native record0; capacity only'},time_origin='2015-12-01T00:00:00Z')
    parameters=cp.broadcast_to(record[0,:12],(6971,12)).copy()
    pet=PreparedPET(forcing.fields,parameters,identity={'case':'capacity only'})
    def measure(launch):
        for _ in range(10):launch()
        begin,end=cp.cuda.Event(),cp.cuda.Event();begin.record()
        for _ in range(200):launch()
        end.record();end.synchronize()
        return cp.cuda.get_elapsed_time(begin,end)/200
    def step():forcing.advance(300.);pet.advance(300.)
    step()
    cp.cuda.get_current_stream().synchronize()
    aggregate_exact=bool((forcing.arrays['fields']==record[0,12:20].astype(cp.float32)[:,None]).all().item())
    receipt={'card':cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),'numerics':numerics.mode(),'case':'actual LowerColorado_TX_v4 CSR with uniformly imposed captured meteorology; CAPACITY ONLY','catchments':6971,'parent_shape':shape,'csr_edges':int(indices.size),'fabric_request_sha256':meta['request_sha256'],'parallelism':'55 blocks x128 threads,one catchment/thread,float64 ordered weighted sums,float32 fields,float64 PET;pool,no graphs','forcing_ms':measure(lambda:forcing.advance(300.)),'pet_ms':measure(lambda:pet.advance(300.)),'forcing_pet_sequence_ms':measure(step),'constant_grid_aggregation_exact_f32':aggregate_exact,'forcing_status':int(cp.count_nonzero(forcing.arrays['status']).item()),'pet_status':int(cp.count_nonzero(pet.arrays['status']).item()),'total_wall_seconds':time.perf_counter()-start,'whole_hydro_step_ms':None,'event_acceptance':False}
    a.out.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
