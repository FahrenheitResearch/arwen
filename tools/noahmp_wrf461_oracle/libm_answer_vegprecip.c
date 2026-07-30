/* Answer 'E <hex>' / 'P <hexbase> <hexexp>' questions with the live glibc
   expf/powf, echoing the question plus the hex of the answer.
   Built and run on the glibc host; the answers are checked against
   gpuwm.core.noahmp_libm by validate_vegprecip_oracle.py --libm-sweep-check.

   No FP arithmetic happens here beyond the library call itself, so nothing
   the compiler does to this file can change the answer being measured. */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

static float asf(uint32_t u) { float x; memcpy(&x, &u, 4); return x; }
static uint32_t asu(float x) { uint32_t u; memcpy(&u, &x, 4); return u; }

int main(void)
{
  char kind;
  uint32_t a, b;
  char line[256];
  while (fgets(line, sizeof line, stdin)) {
    if (sscanf(line, " %c %8X %8X", &kind, &a, &b) < 2) continue;
    if (kind == 'E')
      printf("E %08X %08X\n", a, asu(expf(asf(a))));
    else if (kind == 'P')
      printf("P %08X %08X %08X\n", a, b, asu(powf(asf(a), asf(b))));
  }
  return 0;
}
