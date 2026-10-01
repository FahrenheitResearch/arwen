# glibc_trig_flt32.cuh against the host's libm, bit for bit

`gpuwm/core/kernels/glibc_trig_flt32.cuh` transcribes the six float32 trig
functions the BEP column calls (`module_sf_bep.F`: sinf, cosf, tanf, asinf,
acosf, atanf) from Ubuntu glibc 2.43 (`2.43-2ubuntu2.4`, x86-64 AVX2+FMA),
the libm the WRF v4.7.1 oracle links: sinf/cosf are the Arm algorithm glibc
selects as its FMA ifunc (`__sinf_fma`/`__cosf_fma`), tanf/asinf/acosf/atanf
are glibc 2.43's CORE-MATH correctly rounded versions (no ifunc).  CUDA's own
float trig differs from these by up to 2 ULP; with it the BEP gate
(tests/test_urban_bep_wrf471_parity.py) fails on every case.

Proof, run on node-1 (evidence in this directory):

* `exhaustive-final.log`: the header's bodies compiled for the host
  (`host_shim.hpp` maps the pinned device intrinsics to plain operations
  under `-ffp-contract=off` and `__fma_rn` to `std::fma`) against the
  system `libm.so.6` over all 2^32 inputs per function: 0 mismatches.
* `gpu-final.log`: the header compiled by NVRTC 13.4 (`--fmad=false` and
  `--fmad=true`, CuPy's `--ftz=true`) on the RTX 4090, 67,109,088 inputs per
  function (bit-pattern sweep, dense [-10,10] and [-1,1], every signed
  subnormal, 224 special inputs): 0 mismatches.
* `final-hashes.txt`: the header digest those runs used (LF text
  `23455db61e22e3c57e23bbb878cf33af97609887f11c645c56dae2790c2160e1`); later
  edits to the file's leading comment block do not touch its code.

```sh
g++ -O2 -std=c++17 -ffp-contract=off -fno-builtin -fopenmp exhaustive.cpp -ldl -lm -o exhaustive
OMP_NUM_THREADS=24 ./exhaustive > exhaustive-final.log
g++ -O2 -std=c++17 -ffp-contract=off -fno-builtin -fopenmp -shared -fPIC gpu_oracle.cpp -lm -o gpu_oracle.so
cp ../../gpuwm/core/kernels/glibc_flt32.cuh ../../gpuwm/core/kernels/glibc_trig_flt32.cuh .
python gpu_check.py > gpu-final.log      # take a gpu-mutex OWNER line first
```

`generate_trig.py` + `trig_prefix.txt` regenerate the CORE-MATH bodies from
the retained upstream glibc 2.43 sources.  The result only holds against a
libm that IS glibc 2.43's; a different glibc is a different reference.
