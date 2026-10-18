#include <cmath>
#include <cstdint>
#include <cstring>
#include <omp.h>
using Fn = float (*)(float);
static Fn fns[] = {::log10f, ::cbrtf};
extern "C" void reference(int f, uint64_t start, uint32_t* out, uint64_t n) {
#pragma omp parallel for schedule(static)
  for (uint64_t k = 0; k < n; k++) {
    uint32_t u = (uint32_t)(start + k);
    float x;
    std::memcpy(&x, &u, 4);
    float r = fns[f](x);
    std::memcpy(out + k, &r, 4);
  }
}
