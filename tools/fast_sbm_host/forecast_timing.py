"""Measure the production microphysics call while driving the ordinary CLI.

CUDA events time the actual call. No state or arithmetic is substituted.
The scalar observations are verification receipts and stay on the box.
"""
import argparse,atexit,json,time
from pathlib import Path
import numpy as np


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    options,argv=parser.parse_known_args()
    if argv[:1]==['--']: argv=argv[1:]
    import cupy as cp
    import gpuwm.core.dycore as dycore
    from gpuwm.cli import main as cli
    original=dycore.apply_microphysics
    rows=[]
    def measure(state,cfg,*args,**kwargs):
        begin=cp.cuda.Event(); end=cp.cuda.Event()
        started=time.perf_counter(); begin.record()
        result=original(state,cfg,*args,**kwargs)
        end.record(); end.synchronize()
        row={'mp_physics':int(cfg.mp_physics),'start_time_s':float(state.elapsed_seconds),
             'dt_s':float(cfg.dt),'gpu_ms':float(cp.cuda.get_elapsed_time(begin,end)),
             'wall_ms':1000*(time.perf_counter()-started),'mass_max':{}}
        for name in ('qv','qc','qr','qi','qs','qg'):
            a=getattr(state,name,None)
            if a is not None: row['mass_max'][name]=float(cp.max(a))
        bins=getattr(state,'fast_sbm_bins',None)
        if bins is not None:
            row['nonzero_bins']=int(cp.count_nonzero(bins))
            row['all_bins_finite']=bool(cp.all(cp.isfinite(bins)))
        rows.append(row)
        return result
    dycore.apply_microphysics=measure
    def save():
        options.out.write_text(json.dumps({'scope':'Production microphysics call inside the ordinary CLI; CUDA-event and wall timings include its packing and status checks. Scalar observations follow the timed interval.','steps':rows},indent=2)+'\n')
    atexit.register(save)
    return cli(argv)


if __name__=='__main__': raise SystemExit(main())
