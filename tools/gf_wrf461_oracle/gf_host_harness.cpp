// Host harness: compile gpuwm/core/kernels/gf_deep.cu as plain C++ and run
// it column by column, so the EXACT source that will run on the GPU can be
// graded against the committed oracle fixtures without a GPU.
//
// Fidelity notes.  x86-64 SSE float/double arithmetic is IEEE-754
// round-to-nearest per operation (FLT_EVAL_METHOD == 0); built with
// -ffp-contract=off and without -ffast-math there is no fusing and no
// reassociation, so every __fadd_rn/__fmul_rn/... below maps to the same
// correctly rounded operation the device intrinsic pins.  The one
// difference class left is subnormal handling: the host preserves
// subnormals (MXCSR clear), sm_120 flushes them in FP32 arithmetic -- which
// is exactly what the on-node gate exists to measure.
//
// The CUDA builtins used by the probe kernels (tgammaf/powf negative
// controls) become glibc's own functions here, so those slots are equal by
// construction on the host and are only meaningful on the device.

#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>

// ---- CUDA language shims -------------------------------------------------
#define __device__
#define __global__
#define __constant__ static const
#define __forceinline__ inline
#define __restrict__

struct GfS3 { unsigned x, y, z; };
static GfS3 blockIdx = {0, 0, 0};
static GfS3 blockDim = {1, 1, 1};
static GfS3 threadIdx = {0, 0, 0};

static inline float __fadd_rn(float a, float b) { return a + b; }
static inline float __fsub_rn(float a, float b) { return a - b; }
static inline float __fmul_rn(float a, float b) { return a * b; }
static inline float __fdiv_rn(float a, float b) { return a / b; }
static inline float __fsqrt_rn(float a) { return sqrtf(a); }
static inline double __dadd_rn(double a, double b) { return a + b; }
static inline double __dsub_rn(double a, double b) { return a - b; }
static inline double __dmul_rn(double a, double b) { return a * b; }
static inline double __ddiv_rn(double a, double b) { return a / b; }

static inline float __uint_as_float(unsigned int u)
{ float f; std::memcpy(&f, &u, 4); return f; }
static inline unsigned int __float_as_uint(float f)
{ unsigned int u; std::memcpy(&u, &f, 4); return u; }
static inline float __int_as_float(int i)
{ float f; std::memcpy(&f, &i, 4); return f; }
static inline long long __double_as_longlong(double d)
{ long long l; std::memcpy(&l, &d, 8); return l; }
static inline double __longlong_as_double(long long l)
{ double d; std::memcpy(&d, &l, 8); return d; }
static inline float __double2float_rn(double d) { return (float)d; }

// ---- the kernel source, verbatim -----------------------------------------
// The include ORDER mirrors what gpuwm/core/kernels/__init__.py assembles for
// nvrtc -- preamble (common.cuh), then the module's _EXTRA_HEADERS entry, then
// the .cu -- because a harness that assembles a DIFFERENT string is grading a
// different kernel.  glibc_flt32.cuh joined that list when gf.cu's glibc 2.39
// transcendentals were lifted out for New Tiedtke to share; without this line
// the gfk_* calls left behind in gf.cu do not resolve and the whole no-GPU
// grader stops building, which is the break test_gf_workspace.py names.
#include "common.cuh"
#include "glibc_flt32.cuh"
#include "gf.cu"

// ---- the column workspace, on the host -----------------------------------
// The three stage kernels take their GF_KP column arrays from a caller-
// provided global workspace (gf.cu's "Per-thread column workspace" block),
// laid out one contiguous region per LAUNCH BLOCK and interleaved across
// GFWS_LANES lanes: element k of slot s for lane t lives at
//
//     blockIdx.x * GFWS_BLOCK_FLOATS + (s * GF_KP + k) * GFWS_LANES + t
//
// The kernels read their column as blockIdx.x * blockDim.x + threadIdx.x, so
// the harness has a choice about which of the two shims carries the column
// number.  It carries it in threadIdx.x, with blockDim.x == 1 and blockIdx.x
// pinned at 0: the DATA index is the same `col` it always was (lvin + col *
// ..., lev + col * ..., every fixture graded the same word), while the
// workspace stays inside ONE block's region instead of demanding
// n * GFWS_BLOCK_FLOATS floats -- 1.5 MiB per column, 341 MiB over the
// oracle's 216 columns, three times over.
//
// Bound.  The highest word any lane touches is
//
//     (GFWS_SLOTS * GF_KP - 1) * GFWS_LANES + threadIdx.x
//   = GFWS_BLOCK_FLOATS - GFWS_LANES + (n - 1)
//
// so GFWS_BLOCK_FLOATS + n floats covers every column for ANY n, including
// the n > GFWS_LANES case where a column's lane wraps past lane 63 into the
// words of a lower slot.  That wrap is sound here and only here: the host
// runs the columns strictly one at a time, so no two columns are ever live
// at once, and the device -- where they ARE concurrent -- allocates a full
// GFWS_BLOCK_FLOATS per block and never wraps.  Reuse across columns is the
// same reuse the device performs across tiles, which
// tests/test_gf_workspace.py::test_the_workspace_is_free_of_residue proves
// no column array depends on.
static std::vector<float> gf_ws_storage;

static float *gf_host_workspace(int n)
{
    size_t need = (size_t)GFWS_BLOCK_FLOATS + (size_t)(n > 0 ? n : 1);
    if (gf_ws_storage.size() < need) gf_ws_storage.assign(need, 0.0f);
    return gf_ws_storage.data();
}

// ---- launchers -----------------------------------------------------------
extern "C" void host_gf_deep_stage(const float *lvin, const float *scin,
                                   const int *iin, float *lev, float *sca,
                                   int *isca, int n, int nz)
{
    float *ws = gf_host_workspace(n);
    blockDim.x = 1; blockIdx.x = 0;
    for (int col = 0; col < n; col++) {
        threadIdx.x = (unsigned)col;
        gf_deep_stage(lvin, scin, iin, lev, sca, isca, ws, n, nz);
    }
    threadIdx.x = 0;
}

extern "C" void host_gf_libm_unary(const float *x, float *out, int n)
{
    blockDim.x = 1; threadIdx.x = 0;
    for (int i = 0; i < n; i++) {
        blockIdx.x = (unsigned)i;
        gf_libm_unary_probe(x, out, n);
    }
}

extern "C" void host_gf_libm_pow(const float *x, const float *y, float *out,
                                 int n)
{
    blockDim.x = 1; threadIdx.x = 0;
    for (int i = 0; i < n; i++) {
        blockIdx.x = (unsigned)i;
        gf_libm_pow_probe(x, y, out, n);
    }
}

extern "C" void host_gf_fzu(const float *a, const float *b, float *out, int n)
{
    blockDim.x = 1; threadIdx.x = 0;
    for (int i = 0; i < n; i++) {
        blockIdx.x = (unsigned)i;
        gf_fzu_probe(a, b, out, n);
    }
}

extern "C" void host_gf_const_dump(unsigned int *out)
{
    blockDim.x = 1; threadIdx.x = 0;
    for (int i = 0; i < GF_NCONST; i++) {
        blockIdx.x = (unsigned)i;
        gf_deep_const_dump(out);
    }
}

extern "C" void host_gf_shallow_stage(const float *lvin, const float *scin,
                                      const int *iin, float *lev, float *sca,
                                      int *isca, int k22_wrf_faithful,
                                      int n, int nz)
{
    float *ws = gf_host_workspace(n);
    blockDim.x = 1; blockIdx.x = 0;
    for (int col = 0; col < n; col++) {
        threadIdx.x = (unsigned)col;
        gf_shallow_stage(lvin, scin, iin, lev, sca, isca, ws,
                         k22_wrf_faithful, n, nz);
    }
    threadIdx.x = 0;
}

extern "C" void host_gf_gfdrv_stage(const float *lvin, const float *scin,
                                    const int *iin, float *lev, float *sca,
                                    int *isca, int k22_wrf_faithful,
                                    int n, int nz)
{
    float *ws = gf_host_workspace(n);
    blockDim.x = 1; blockIdx.x = 0;
    for (int col = 0; col < n; col++) {
        threadIdx.x = (unsigned)col;
        gf_gfdrv_stage(lvin, scin, iin, lev, sca, isca, ws,
                       k22_wrf_faithful, n, nz);
    }
    threadIdx.x = 0;
}
