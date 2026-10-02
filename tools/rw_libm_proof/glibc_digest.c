/* Reference digests for tools/rustwx/crates/rw-libm/tests/glibc_digest.rs.
 *
 * Draws exactly the inputs that test draws and hashes the REFERENCE
 * library's outputs: glibc's own sinf, cosf, tanf, atanf, expf, logf,
 * log10f, powf and atan2f, and CORE-MATH's cr_asin / cr_acos for the
 * binary64 pair.  Run on a glibc 2.43 x86-64 machine with FMA (the WPS
 * oracle's platform), node-2, Ubuntu GLIBC 2.43-2ubuntu2.4:
 *
 *   gcc -O2 -ffp-contract=off -fno-builtin glibc_digest.c
 *       $B/cref/nofma_asin.o $B/cref/nofma_acos.o -lm -o glibc_digest
 *
 * (the two objects from build_cref.sh).  Output, as the test records it:
 *
 *   sinf   db472efce84b7aba      powf   3f5a23de5c9662d3
 *   cosf   99cf1e32c97b9d1a      atan2f c439aa0d9262afed
 *   tanf   f460750529b80cdb      asin   73e1f4ba5814f8db  (CORE-MATH)
 *   atanf  8764fb960f7d4494      acos   c6346a3a1a9c53f8  (CORE-MATH)
 *   expf   b5ec97f39f697846      asin   a783d84fce25e06d  (glibc, IBM)
 *   logf   1ace987212ae760a      acos   e1912a05377b2b19  (glibc, IBM)
 *   log10f 3d3da09cd283d0da
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

double cr_asin(double);
double cr_acos(double);

static const uint64_t H0 = 0xcbf29ce484222325ull;

static uint64_t mix(uint64_t h, uint64_t v) {
  for (int i = 0; i < 8; i++) {
    h ^= (v >> (8 * i)) & 0xff;
    h *= 0x100000001b3ull;
  }
  return h;
}
static uint64_t canon32(float x) {
  uint32_t u;
  memcpy(&u, &x, 4);
  return isnan(x) ? 0x7fc00000u : u;
}
static uint64_t canon64(double x) {
  uint64_t u;
  memcpy(&u, &x, 8);
  return isnan(x) ? 0x7ff8000000000000ull : u;
}
static float f32of(uint32_t u) {
  float f;
  memcpy(&f, &u, 4);
  return f;
}
static double f64of(uint64_t u) {
  double f;
  memcpy(&f, &u, 8);
  return f;
}
static uint64_t word(uint64_t i, uint64_t salt) {
  uint64_t z = (i + salt) * 0x9e3779b97f4a7c15ull;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return z ^ (z >> 31);
}

static uint64_t digest1(float (*f)(float), uint64_t salt) {
  uint64_t h = H0;
  for (uint64_t i = 0; i < (1ull << 20); i++)
    h = mix(h, canon32(f(f32of((uint32_t)(i * 4097)))));
  for (uint64_t i = 0; i < (1ull << 18); i++)
    h = mix(h, canon32(f(f32of((uint32_t)word(i, salt)))));
  return h;
}

static uint64_t digest_powf(float (*f)(float, float)) {
  uint64_t h = H0;
  for (uint64_t i = 0; i < (1ull << 18); i++) {
    uint64_t w = word(i, 21);
    float x, y;
    if (i % 2 == 0) {
      x = f32of((uint32_t)w);
      y = f32of((uint32_t)(w >> 32));
    } else {
      x = f32of((uint32_t)w & 0x7f7fffffu);
      y = (float)((double)(w >> 32) / 4294967296.0 * 64.0 - 32.0);
    }
    h = mix(h, canon32(f(x, y)));
  }
  return h;
}

static uint64_t digest_atan2f(float (*f)(float, float)) {
  uint64_t h = H0;
  for (uint64_t i = 0; i < (1ull << 18); i++) {
    uint64_t w = word(i, 22);
    float y, x;
    if (i % 2 == 0) {
      y = f32of((uint32_t)w);
      x = f32of((uint32_t)(w >> 32));
    } else {
      y = (float)(((double)(uint32_t)w / 4294967296.0 * 2.0 - 1.0) * 3000.0);
      x = (float)(((double)(w >> 32) / 4294967296.0 * 2.0 - 1.0) * 3000.0);
    }
    h = mix(h, canon32(f(y, x)));
  }
  return h;
}

static uint64_t digest_f64(double (*f)(double), uint64_t salt) {
  uint64_t h = H0;
  for (uint64_t i = 0; i < (1ull << 18); i++) {
    uint64_t w = word(i, salt);
    double x;
    if (i % 2 == 0)
      x = f64of((w % 0x3ff0000000000001ull) | (w & (1ull << 63)));
    else
      x = (double)(w >> 11) / (double)(1ull << 53) * 2.0 - 1.0;
    h = mix(h, canon64(f(x)));
  }
  return h;
}

int main(void) {
  printf("sinf   %016llx\n", (unsigned long long)digest1(sinf, 1));
  printf("cosf   %016llx\n", (unsigned long long)digest1(cosf, 2));
  printf("tanf   %016llx\n", (unsigned long long)digest1(tanf, 3));
  printf("atanf  %016llx\n", (unsigned long long)digest1(atanf, 4));
  printf("expf   %016llx\n", (unsigned long long)digest1(expf, 5));
  printf("logf   %016llx\n", (unsigned long long)digest1(logf, 6));
  printf("log10f %016llx\n", (unsigned long long)digest1(log10f, 7));
  printf("powf   %016llx\n", (unsigned long long)digest_powf(powf));
  printf("atan2f %016llx\n", (unsigned long long)digest_atan2f(atan2f));
  printf("asin   %016llx  (CORE-MATH cr_asin)\n", (unsigned long long)digest_f64(cr_asin, 23));
  printf("acos   %016llx  (CORE-MATH cr_acos)\n", (unsigned long long)digest_f64(cr_acos, 24));
  printf("asin   %016llx  (glibc, for comparison only)\n", (unsigned long long)digest_f64(asin, 23));
  printf("acos   %016llx  (glibc, for comparison only)\n", (unsigned long long)digest_f64(acos, 24));
  return 0;
}
