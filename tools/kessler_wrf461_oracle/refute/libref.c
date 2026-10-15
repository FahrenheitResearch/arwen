/* Scalar C-library powf/expf over a contiguous range of float32 bit
 * patterns.  Built -O1 -fno-builtin -fno-tree-vectorize so every element is
 * one scalar libm call (checked with nm: no _ZGV vector symbol). */
#include <math.h>
#include <stdint.h>
#include <string.h>

static float from_bits(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }

void ref_pow(float y, uint32_t start, uint32_t n, float *out)
{
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)n; ++i)
        out[i] = powf(from_bits(start + (uint32_t)i), y);
}

void ref_exp(uint32_t start, uint32_t n, float *out)
{
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)n; ++i)
        out[i] = expf(from_bits(start + (uint32_t)i));
}
