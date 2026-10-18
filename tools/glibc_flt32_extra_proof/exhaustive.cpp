// glibc_flt32_extra.cuh (gfk_log10, gfk_cbrt) compiled for the host against
// the host's libm.so.6 over every float32 input.
#include "../glibc_trig_flt32_proof/host_shim.hpp"
#define FMUL(a, b) __fmul_rn((a), (b))
#define FADD(a, b) __fadd_rn((a), (b))
#define FSUB(a, b) __fsub_rn((a), (b))
#define FDIV(a, b) __fdiv_rn((a), (b))
#define DADD(a, b) __dadd_rn((a), (b))
#define DSUB(a, b) __dsub_rn((a), (b))
#define DMUL(a, b) __dmul_rn((a), (b))
#define DDIV(a, b) __ddiv_rn((a), (b))
inline int __float_as_int(float x) { int u; std::memcpy(&u, &x, 4); return u; }
inline float __int_as_float(int u) { float x; std::memcpy(&x, &u, 4); return x; }
inline int __clz(int x) { return x == 0 ? 32 : __builtin_clz((unsigned)x); }
#define __forceinline__ inline
// gfk_log is WOOF's own logf (glibc_flt32.cuh, proven word for word by the
// oracle sweeps); the host build calls the host's logf, which it equals.
inline float gfk_log(float x) { return ::logf(x); }
#include "../../gpuwm/core/kernels/glibc_flt32_extra.cuh"
#include <omp.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cfenv>
using Fn = float (*)(float);
const char* names[] = {"log10f", "cbrtf"};
Fn oracle[] = {::log10f, ::cbrtf};
Fn trans[] = {gfk_log10, gfk_cbrt};
int main() {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  std::fesetround(FE_TONEAREST);
  for (int f = 0; f < 2; f++) {
    Dl_info d{};
    dladdr((void*)oracle[f], &d);
    printf("%s resolved=%s threads=%d\n", names[f], d.dli_fname,
           omp_get_max_threads());
    auto t = std::chrono::steady_clock::now();
    uint64_t bad = 0, first = UINT64_MAX;
#pragma omp parallel for reduction(+ : bad) reduction(min : first) schedule(static)
    for (uint64_t k = 0; k < 0x100000000ULL; k++) {
      float x = __uint_as_float((uint32_t)k);
      float ra = oracle[f](x), rb = trans[f](x);
      uint32_t a = __float_as_uint(ra), b = __float_as_uint(rb);
      // NaN results compare as NaN (payloads are not part of the claim).
      if (a != b && !(std::isnan(ra) && std::isnan(rb))) {
        bad++;
        if (k < first) first = k;
      }
    }
    printf("%s inputs=4294967296 mismatches=%llu seconds=%.3f", names[f],
           (unsigned long long)bad,
           std::chrono::duration<double>(std::chrono::steady_clock::now() - t)
               .count());
    if (bad) printf(" first=%08x", (uint32_t)first);
    puts("");
  }
}
