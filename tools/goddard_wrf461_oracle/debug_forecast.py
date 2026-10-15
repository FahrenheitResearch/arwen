"""Capture one failing real column without changing any kernel arithmetic."""
from pathlib import Path
import json,sys
import numpy as np
import gpuwm.core.kernels as kernels
original=kernels.get_kernel_int_defines
saved={}
root=Path(sys.argv[1])
cli_args=sys.argv[2:]

def get_kernel(name,func,defines):
    kernel=original(name,func,defines)
    if name!='goddard':return kernel
    def call(grid,block,args,**kwargs):
        result=kernel(grid,block,args,**kwargs)
        if func=='goddard_bands_columns':
            saved['flags']=args[1];saved['optics']=args[2]
        if func=='goddard_publish_columns' and not (root/'bad-column.json').exists():
            n,c=int(args[4]),int(args[5]);out=args[3].get().reshape(2*n+15,c)
            bad=np.argwhere(~np.isfinite(out))
            if len(bad):
                col=int(bad[0,1])
                values={k:np.asarray(v.get()).reshape(-1,c)[:,col].tolist() for k,v in
                        [('raw',args[2]),('band',args[0]),('flux',args[1]),('output',args[3]),*saved.items()]}
                (root/'bad-column.json').write_text(json.dumps({'layers':n,'columns':c,'column':col,'values':values},indent=2)+'\n')
        return result
    return call
kernels.get_kernel_int_defines=get_kernel
from gpuwm.cli import main
raise SystemExit(main(cli_args))
