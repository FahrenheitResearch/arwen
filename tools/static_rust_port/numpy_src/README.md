# Vendored numpy kernel sources (porting reference only)

Reference copies of the numpy 2.2.6 SIMD kernels that
`tools/rustwx/crates/static-fields/src/projection/npmath.rs` ports for
bit parity (numpy routes float32 sin/cos/exp/log through these kernels
rather than libm; measured by `../probe_libm.py`).  These files are NOT
compiled; they document exactly what was ported.

| file | origin |
|---|---|
| `loops_trigonometric.dispatch.cpp` | <https://raw.githubusercontent.com/numpy/numpy/v2.2.6/numpy/_core/src/umath/loops_trigonometric.dispatch.cpp> |
| `loops_exponent_log.dispatch.c.src` | <https://raw.githubusercontent.com/numpy/numpy/v2.2.6/numpy/_core/src/umath/loops_exponent_log.dispatch.c.src> |
| `npy_simd_data.h` | <https://raw.githubusercontent.com/numpy/numpy/v2.2.6/numpy/_core/src/umath/npy_simd_data.h> |

One measured divergence from the source text: the MSVC-built wheel
CONTRACTS each kernel's rint sequence (`Mul(x, c)` then `Add(…, magic)`)
into a fused multiply-add, which flips the chosen quadrant on
half-integer knife edges.  The Rust port mirrors the fused spelling;
evidence in `../hunt_variant3.py` (sin/cos, 42 values over a 400k sweep)
and `../check_exp_case.py` (exp, 1 value).

License: numpy is BSD-3-Clause, Copyright (c) 2005-2025, NumPy
Developers.  Full text: <https://github.com/numpy/numpy/blob/v2.2.6/LICENSE.txt>.
Redistribution of these source files retains that license; the Rust port
in `npmath.rs` is a derived work and carries the same attribution in its
module docs.
