# Kessler refute column sets (refute-setA/B/C/D/F)

Written on box W1 (Ubuntu 24.04, Intel Xeon Platinum 8559C, gfortran 13.3.0,
glibc 2.39), 2026-10-07, by the adversarial check of lane/verify-kessler-mp1.

- Generator: `tools/kessler_wrf461_oracle/refute/make_refute_columns.py`
  (seed 424242 for sets A to E, 777 for set F).
- Reference: `run_kessler_O0` exactly as `tools/kessler_wrf461_oracle/build.sh`
  builds it (byte-unmodified WRF v4.6.1 commit
  `d66e442fccc04111067e29274c9f9eaccc3cef28`, `-O0 -ffp-contract=off`, scalar
  libm `expf`/`powf`, no FMA instruction in the Kessler object).
- Packing: `tools/kessler_wrf461_oracle/refute/make_fixture.py`. Set A keeps
  the dt 7.2 s x 10 and 900 s x 2 cases and set B the 20 s x 5 case of the
  full runs; C, D and F are whole.

## The C-library variant

On an FMA-capable x86 CPU, glibc's ifunc selects an FMA build of `powf` and
`expf`. Its words differ from the generic build on a handful of inputs:
`powf(x, 0.875)` at x = 0x30740e00, 0x34740e00, 0x48740e00, 0x4c740e00 and
`expf` at x = 0x4202422f, 0xc27c65d9 (all 1 ULP; found by `libm_sweep.py`
over all 2**32 inputs). `refute-setF/kessler-wrf461.npz` is the reference
under the CPU's own dispatch (FMA build);
`refute-setF/kessler-wrf461-generic-libm.npz` is the same binary run with
`GLIBC_TUNABLES=glibc.cpu.hwcaps=-AVX2,-FMA` (generic build). The two differ
in 224 qc and 224 qr words, all in the witness columns. For sets A to E and
the lane's own fixture the two dispatches give byte-identical output.

## Files

| file | sha256 |
|---|---|
| refute-setA/kessler-inputs.npz | 620b3c9c64637b5f8cd56d7fae8df5e989c1b7bcd0514f2a155d6ab2c7a0e691 |
| refute-setA/kessler-wrf461.npz | 5d6c16184dc2e08638cc2213b5f7cf3cf97b974729c55c6e66c0531d041cefe4 |
| refute-setB/kessler-inputs.npz | 882d81d8bb0d9981b1a365a86b02721f028270c9d663053ebfe9230d0075253c |
| refute-setB/kessler-wrf461.npz | 50eeafda1e17a847e087dc6b8780fdfaf304ae531579875864583528abb7f149 |
| refute-setC/kessler-inputs.npz | dc0e1e773b6f7bcdda915881bada8db5c9fe9782b51ed1cfc12fbd7c025a87fe |
| refute-setC/kessler-wrf461.npz | 81c740aae6a5e55dfbf675c59531f04593fb8faf8622516233a929737bcd16c9 |
| refute-setD/kessler-inputs.npz | 1fe0b5c83a57740f0c108f5c03457c560c0aae172580ca4b471e17aa72fee9c6 |
| refute-setD/kessler-wrf461.npz | 211bef9d1dca66649d799ea28691da3e1613b4da7be7b0950f30354c5bcab968 |
| refute-setF/kessler-inputs.npz | be9f46bb252486395e63499827a377a03e83cf01a68a85f2d3e49854a805b10e |
| refute-setF/kessler-wrf461.npz | fed70d407ca9a9d53e53334d52c423769b445fe840bd1bc62b44a323b879577a |
| refute-setF/kessler-wrf461-generic-libm.npz | e5d67bdbe62d63df8a7ddda59819810080d5a82a601a4510c4ab31af4f24f123 |
