"""Check the optional parent binding on CUDA using literal analytic fixtures.

Not native physics replay: deliberately dyadic dimensionless humidity fixtures
distinguish cell conversion from conversion after averaging. All arithmetic and
array comparisons run on CUDA. --baseline-module compares the previous actual
kernel's canonical two-accumulator result, without tolerances or a CPU oracle.
"""
import argparse
import importlib.util
import hashlib
import json
from pathlib import Path
import time
import sys
from types import SimpleNamespace


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--baseline-module',type=Path,required=True)
    args=parser.parse_args()
    start=time.perf_counter()
    import cupy as cp
    from gpuwm import numerics
    from gpuwm.hydro.forcing import PreparedCatchmentForcing,FIELDS,_parent_field
    spec=importlib.util.spec_from_file_location('baseline_forcing',args.baseline_module)
    baseline=importlib.util.module_from_spec(spec);spec.loader.exec_module(baseline)
    checks={}
    def check(name,condition):
        checks[name]=bool(condition)
        if not checks[name]:raise AssertionError(name)
    def equal(a,b):return bool(cp.array_equal(a,b).item())
    offsets=cp.asarray([0,2,5,7],dtype=cp.uint64)
    indices=cp.asarray([0,1,1,2,3,0,3],dtype=cp.uint32)
    weights=cp.asarray([.5,.5,.25,.25,.5,.5,.5],dtype=cp.float64)
    parent={name:cp.ones(4,dtype=cp.float32) for name in FIELDS}
    parent['Q2D']=cp.asarray([0,1,3,7],dtype=cp.float32)
    rain=cp.zeros(4,dtype=cp.float32)
    physics=SimpleNamespace(fields={'q2':parent['Q2D']},rainc=None,
                            microphysics=SimpleNamespace(rainnc=rain))
    state=SimpleNamespace(legacy=parent['T2D'])
    components={'weather':SimpleNamespace(fields=parent)}
    def resolve(path):return _parent_field(path,state=state,physics=physics,components=components)
    check('physics_dictionary_q2',resolve('physics.fields.q2') is parent['Q2D'])
    check('physics_microphysics_rain',resolve('physics.microphysics.rainnc') is rain)
    check('absent_cumulus_not_bound',resolve('physics.rainc') is None)
    check('legacy_state_path',resolve('legacy') is parent['T2D'])
    check('explicit_state_path',resolve('state.legacy') is parent['T2D'])
    check('component_dictionary_path',resolve('components.weather.fields.T2D') is parent['T2D'])
    kwargs=dict(identity={'fixture':'analytic binding only'},time_origin='2023-01-05T00:00:00Z')
    prepared=PreparedCatchmentForcing(parent,offsets,indices,weights,
        rain_accumulators=(rain,),rain_baseline=(rain.copy(),),humidity='mixing_ratio',**kwargs)
    pointers={key:a.data.ptr for key,a in prepared.arrays.items()}
    rain[:]=cp.asarray([0,4,8,12],dtype=cp.float32)
    stream=cp.cuda.Stream(non_blocking=True)
    cp.cuda.get_current_stream().synchronize()  # setup handoff to the test stream
    with stream:prepared.advance(4,stream=stream)
    stream.synchronize()
    check('cellwise_humidity_before_weighting',equal(prepared.fields['Q2D'],cp.asarray([.25,.75,.4375],dtype=cp.float32)))
    check('single_accumulator_mm_per_second',equal(prepared.fields['RAINRATE'],cp.asarray([.5,2.25,1.5],dtype=cp.float32)))
    check('valid_status',not bool(cp.any(prepared.arrays['status']).item()))
    check('restart_humidity_semantics',prepared.restart_identity['humidity']=='mixing_ratio')
    check('restart_single_accumulator',prepared.restart_identity['rain_accumulator_count']==1)
    saved={key:a.copy() for key,a in prepared.arrays.items()}
    rain+=4
    prepared.advance(4)
    check('next_step_rain_difference',equal(prepared.fields['RAINRATE'],cp.ones(3,dtype=cp.float32)))
    expected=prepared.arrays['fields'].copy()
    for key,a in saved.items():cp.copyto(prepared.arrays[key],a)
    rain+=4
    prepared.advance(4)
    check('restart_replay_words',equal(prepared.arrays['fields'].view(cp.uint32),expected.view(cp.uint32)))
    check('stable_restart_pointers',all(prepared.arrays[key].data.ptr==ptr for key,ptr in pointers.items()))
    rain[0]=0
    prepared.advance(4)
    check('decreasing_accumulator_status',equal(prepared.arrays['status'],cp.asarray([1,0,1],dtype=cp.int32)))
    check('decreasing_accumulator_nan',bool(cp.isnan(prepared.fields['RAINRATE'][0]).item()))
    parent['Q2D'][0]=-1
    prepared.advance(4)
    check('invalid_mixing_ratio_nan',bool(cp.isnan(prepared.fields['Q2D'][0]).item()))
    parent['Q2D'][0]=0
    two=(cp.ones(4,dtype=cp.float32),cp.ones(4,dtype=cp.float32))
    init=(cp.zeros(4,dtype=cp.float32),cp.zeros(4,dtype=cp.float32))
    old=baseline.PreparedCatchmentForcing(parent,offsets,indices,weights,rain_accumulators=two,rain_baseline=init,**kwargs)
    new=PreparedCatchmentForcing(parent,offsets,indices,weights,rain_accumulators=two,rain_baseline=init,**kwargs)
    old.advance(4);new.advance(4)
    words=int(cp.count_nonzero(old.arrays['fields'].view(cp.uint32)!=new.arrays['fields'].view(cp.uint32)).item())
    check('canonical_two_accumulator_words',words==0)
    check('canonical_restart_rain_words',equal(old.arrays['previous_rain_mm'].view(cp.uint64),new.arrays['previous_rain_mm'].view(cp.uint64)))
    old_rate=baseline.PreparedCatchmentForcing(parent,offsets,indices,weights,**kwargs)
    new_rate=PreparedCatchmentForcing(parent,offsets,indices,weights,**kwargs)
    old_rate.advance(4);new_rate.advance(4)
    direct_words=int(cp.count_nonzero(old_rate.arrays['fields'].view(cp.uint32)!=new_rate.arrays['fields'].view(cp.uint32)).item())
    check('canonical_direct_rate_words',direct_words==0)
    def measure(component):
        for _ in range(10):component.advance(4)
        begin,end=cp.cuda.Event(),cp.cuda.Event();begin.record()
        for _ in range(200):component.advance(4)
        end.record();end.synchronize()
        return cp.cuda.get_elapsed_time(begin,end)/200
    receipt={'scope':'analytic parent binding regression; not native or physical acceptance',
        'argv':sys.argv,'source_sha256':{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
            (Path(__file__),Path(sys.modules[PreparedCatchmentForcing.__module__].__file__),
             Path(sys.modules[PreparedCatchmentForcing.__module__].__file__).with_suffix('.cu'),
             args.baseline_module,args.baseline_module.with_suffix('.cu'))},
        'checks':checks,'canonical_compared_words':48,'canonical_differing_words':words+direct_words,
        'card':cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),'numerics':numerics.mode(),
        'parallelism':'1 block x128 threads;3 catchments;7 CSR edges;one CUDA launch per advance;no graphs',
        'mixed_humidity_single_rain_ms':measure(prepared),'canonical_two_rain_ms':measure(new),
        'timing_scope':'200 sequential launches,10 warmups,event boundaries only;shared pool not isolated',
        'wall_seconds':time.perf_counter()-start}
    args.out.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
