# glibc_flt32_extra.cuh against the host's libm, bit for bit

`gpuwm/core/kernels/glibc_flt32_extra.cuh` holds WOOF's own float32
`log10f` (`gfk_log10`, over WOOF's `logf`, `gfk_log` in `glibc_flt32.cuh`)
and `cbrtf` (`gfk_cbrt`).  Morrison (mp=10) needs `log10f` where WRF's
`ALOG10` is evaluated; CUDA's own `log10f` and `cbrtf` round differently.

The proof grades both routines against the C library the WRF v4.6.1 column
oracles link (Ubuntu 24.04, libm.so.6 of glibc 2.39), over every float32:

* `exhaustive-final.log`: the header compiled for the host, all 2^32
  inputs per function;
* `gpu-final.log`: the header compiled by NVRTC (`--fmad=false` and
  `--fmad=true`, CuPy's `--ftz=true`) on an RTX PRO 6000 Blackwell, all
  2^32 inputs per function.  The card flushes subnormal results, which
  `cbrtf` returns only for subnormal arguments; those are counted apart.

```sh
g++ -O2 -std=c++17 -ffp-contract=off -fno-builtin -fopenmp exhaustive.cpp -ldl -lm -o exhaustive
./exhaustive > exhaustive-final.log
g++ -O2 -std=c++17 -ffp-contract=off -fno-builtin -fopenmp -shared -fPIC gpu_oracle.cpp -lm -o gpu_oracle.so
python gpu_check.py > gpu-final.log      # one card, through the gpu mutex
```
