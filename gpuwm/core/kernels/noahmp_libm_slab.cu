// Elementwise glibc-2.39 FP32 transcendentals over a whole slab of land
// columns, for the vectorised Noah-MP composition.
//
// WHY THIS FILE EXISTS
// --------------------
// The Noah-MP column composition -- ENERGY's own arithmetic and NOAHMP_SFLX's
// prefix and postfix -- is being evaluated for every land column at once with
// CuPy array operations instead of one CPython frame per column.  FP32 add,
// subtract, multiply, divide and sqrt are IEEE-754 correctly rounded, so a
// CuPy float32 ufunc reproduces gpuwm.core.noahmp_libm.f32(a op b) exactly and
// needs nothing from this file.  The transcendentals do not: numpy's float32
// powf/expf and CUDA's device powf/expf are each a *different* non-glibc
// function, and gfortran on x86-64 calls glibc's.  That divergence class has
// already cost this project a session (see docs/wrf_ruc_runtime_admission.md,
// "What is deliberately still on the host").
//
// So the composition's transcendentals go through the ONE audited device
// transcription this tree has: r_pow / r_exp / r_log in noahmp_leaves.cu, and
// nmpe_tanhf in noahmp_energy.cu.  This file adds no arithmetic whatsoever --
// every kernel below is a bare elementwise wrapper around a function that is
// already held at max_ulp 0 against glibc-libm-fp32.csv.  If a kernel here
// ever grows an expression, it has stopped being a wrapper and its gate no
// longer covers it.
//
// COMPOSITION: compiled after noahmp_leaves.cu and noahmp_energy.cu; see
// gpuwm/core/noahmp_kernel_sources.py.  It does not compile alone, and that
// is the point -- there is exactly one copy of each of these tables in the
// translation unit, borrowed rather than re-transcribed.

extern "C" __global__
void nmp_slab_powf(const float *x, const float *y, float *o, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    o[tid] = r_pow(x[tid], y[tid]);
}

extern "C" __global__
void nmp_slab_expf(const float *x, float *o, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    o[tid] = r_exp(x[tid]);
}

extern "C" __global__
void nmp_slab_logf(const float *x, float *o, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    o[tid] = r_log(x[tid]);
}

extern "C" __global__
void nmp_slab_tanhf(const float *x, float *o, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    o[tid] = nmpe_tanhf(x[tid]);
}
