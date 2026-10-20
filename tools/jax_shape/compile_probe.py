"""Load generic and specialized AC handles through production loaders."""
import concurrent.futures,json,os,time
from pathlib import Path
import cupy as cp
from gpuwm.core.kernels import get_kernel
from gpuwm.core.kernel_shape import FUNCTIONS
start=time.perf_counter()
def probe(job):
 name,func,shape=job
 if name == "mynn_pbl": shape=(shape[0],1,shape[1]*shape[2])
 if name == "mynn_surface": shape=(1,1,shape[1]*shape[2])
 with cp.cuda.Device(0):
  before=get_kernel(name,func)
  after=get_kernel(name,func,shape)
  return dict(module=name,function=func,shape=shape,before=before.attributes,after=after.attributes)
if __name__ == '__main__':
 import multiprocessing
 jobs=[(name,func,shape) for name,funcs in FUNCTIONS.items() if name!='mynn_dmp_sibling' for func in funcs for shape in ((44,70,120),(44,117,267))]
 with concurrent.futures.ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn')) as pool:
  rows=list(pool.map(probe,jobs))
 result=dict(mode=os.environ.get('GPUWM_NUMERICS'),seconds=time.perf_counter()-start,rows=rows)
 Path(os.environ['SHAPE_RECEIPT']).write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(dict(seconds=result['seconds'],rows=len(rows))))
