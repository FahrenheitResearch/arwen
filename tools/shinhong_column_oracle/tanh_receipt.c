/* How often the host C library's tanhf is not the once-rounded double tanh,
 * over every float32 x in [-10, 14] (the reach of Shin-Hong's
 * uwstx = -80*|ust/wstar - 0.5| + 14, module_bl_shinhong.F:846), and on how
 * many of those csfac = 0.5*(tanh(uwstx)+3) (:847) is a different word.
 *
 *   gcc -O1 -fopenmp -fno-builtin -o tanh_receipt tanh_receipt.c -lm
 *   OMP_NUM_THREADS=16 ./tanh_receipt
 *
 * On W1 (glibc 2.39): 118674314 and 4098664
 * (tests/data/oracles/shinhong_columns_tanh/tanh-receipt.txt).
 */
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

int main(void)
{
    long d1 = 0, d2 = 0;
#pragma omp parallel for reduction(+:d1,d2) schedule(static)
    for (long b = 0; b < 0x100000000L; ++b) {
        uint32_t u = (uint32_t)b;
        float x;
        memcpy(&x, &u, 4);
        if (!(x >= -10.f && x <= 14.f)) continue;
        volatile float a = tanhf(x);
        float c = (float)tanh((double)x);
        if (memcmp((void *)&a, &c, 4)) {
            d1++;
            volatile float s1 = 0.5f * (a + 3.f), s2 = 0.5f * (c + 3.f);
            if (memcmp((void *)&s1, (void *)&s2, 4)) d2++;
        }
    }
    printf("tanhf!=rn(tanh) on %ld floats in [-10,14]; csfac differs on %ld\n",
           d1, d2);
    return 0;
}
