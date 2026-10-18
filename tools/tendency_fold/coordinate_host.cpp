// Isolated host execution of the production coordinate kernel. Build with
// -O2 -ffp-contract=off -fno-fast-math; device qualification remains separate.
#include <algorithm>
#include <cmath>
using std::max;
using std::min;
#define __device__
#define __global__
#define __forceinline__ inline
struct Index3 { int x, y, z; };
static Index3 blockIdx, threadIdx, blockDim = {1, 1, 1};
static inline float __fadd_rn(float a, float b) { return a+b; }
static inline float __fsub_rn(float a, float b) { return a-b; }
static inline float __fmul_rn(float a, float b) { return a*b; }
static inline float __fdiv_rn(float a, float b) { return a/b; }
#include "../../gpuwm/core/kernels/diff_opt1.cu"

extern "C" void coordinate_host(const float* field, const float* kh,
    const float* mu, const float* c1, const float* c2, const float* base,
    const float* mt, const float* mfu, const float* mfv, float rdx, float rdy,
    float* out, int nz, int ny, int nx) {
    for (int k=0; k<nz; ++k)
        for (int j=0; j<ny; ++j)
            for (int i=0; i<nx; ++i) {
                blockIdx = {i, j, k};
                wrf_diff_opt1_horizontal(field, kh, mu, c1, c2, base,
                    mt, mfu, mfv, rdx, rdy, out, nz, ny, nx, 0, 1, 1, 1);
            }
}
