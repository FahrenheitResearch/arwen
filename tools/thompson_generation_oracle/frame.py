"""Read actual CUDA driver frame sizes for each column capacity."""
import json
from pathlib import Path
import sys

from gpuwm.core.kernels import get_kernel, get_kernel_int_defines

rows = []
for capacity in (64,128,256):
    f = (get_kernel("thompson_column","thompson_column_step") if capacity == 64
         else get_kernel_int_defines("thompson_column","thompson_column_step",(("THOMPSON_COLUMN_KMAX",capacity),)))
    import cupy as cp
    rows.append({"capacity":capacity,"local_size_bytes":cp.cuda.driver.funcGetAttribute(3,f.ptr),
                 "num_regs":cp.cuda.driver.funcGetAttribute(4,f.ptr),
                 "max_threads":cp.cuda.driver.funcGetAttribute(0,f.ptr)})
Path(sys.argv[1]).write_text(json.dumps(rows,indent=2)+"\n")
print(rows)
