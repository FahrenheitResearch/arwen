# CPU-only NVRTC compile of the NSSL production units (no device needed).
# Compiles through gpuwm.nvrtc_cache_key.compile_program (A160: a fixed-name
# direct NVRTC compile can be answered from the cache under another flush mode).
import sys

from gpuwm.core.kernels import module_options, module_source
from gpuwm.nvrtc_cache_key import compile_program

names = sys.argv[1:] or ["nssl2", "nssl2_fused_gs", "nssl2_driver_support", "nssl2_diagnostics", "nssl2_nucond"]
ok = True
for name in names:
    opts = [o if isinstance(o, str) else o.decode() for o in module_options(name)]
    opts += ["-arch=compute_120", "--fmad=false", "-DGPUWM_WRF_EXACT=1"]
    try:
        compile_program(module_source(name), opts, name=name + ".cu")
        print(name, "OK")
    except Exception as e:  # the NVRTC log is in the exception text
        ok = False
        print(name, "FAIL")
        print("\n".join(line for line in str(e).splitlines() if "error" in line)[:4000])
sys.exit(0 if ok else 1)
