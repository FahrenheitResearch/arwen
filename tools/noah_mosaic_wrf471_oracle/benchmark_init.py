"""Measure the public host init (lsm_mosaic_init) on a one-million-column domain.

Linux only (``resource``).  receipts/init-benchmark.json is one reading.
"""
import json
import resource
from time import perf_counter
import numpy as np
from gpuwm.core.noah_mosaic import lsm_mosaic_init
ny=nx=1000
f=np.zeros((21,ny,nx),np.float32)
f[0]=.6;f[6]=.25;f[11]=.15
state=dict(ivgtyp=np.full((ny,nx),7,np.int32),xland=np.ones((ny,nx),np.float32),
           xice=np.zeros((ny,nx),np.float32))
for n in "tsk snow snowc snowh canwat albedo albbck emiss embck znt".split():
 state[n]=np.full((ny,nx),.2,np.float32)
for n in ("tslb","smois","sh2o"):
 state[n]=np.full((4,ny,nx),.3,np.float32)
t=perf_counter()
out=lsm_mosaic_init(f,**state,mosaic_cat=3,iswater=17,isice=15,fractional_seaice=False)
print(json.dumps(dict(columns=nx*ny,seconds=perf_counter()-t,
                     peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)))
