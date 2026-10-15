/* The six inputs where the device's gfk_pow/gfk_exp and this box's libm
 * powf/expf disagreed in the full sweep.  Prints libm's words; run once
 * normally and once with GLIBC_TUNABLES masking FMA/AVX2 to see whether the
 * host's answer depends on which libm variant the CPU selects. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static float fb(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static uint32_t bf(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

int main(void)
{
    volatile float e875 = 0.875f;
    uint32_t px[4] = {0x30740e00u, 0x34740e00u, 0x48740e00u, 0x4c740e00u};
    uint32_t ex[2] = {0x4202422fu, 0xc27c65d9u};
    for (int i = 0; i < 4; ++i)
        printf("powf(%08x, 0.875) = %08x\n", px[i], bf(powf(fb(px[i]), e875)));
    for (int i = 0; i < 2; ++i)
        printf("expf(%08x) = %08x\n", ex[i], bf(expf(fb(ex[i]))));
    return 0;
}
