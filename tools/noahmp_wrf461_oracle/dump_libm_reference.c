#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

static inline uint32_t asu(float f){ uint32_t u; memcpy(&u,&f,4); return u; }
static inline float asf(uint32_t u){ float f; memcpy(&f,&u,4); return f; }
static double urnd(void){ return (double)rand() / ((double)RAND_MAX + 1.0); }

int main(void){
  srand(31337);
  FILE* out = fopen("libm-reference.csv", "w");
  fprintf(out, "call,a,b,result\n");
  /* logf / log10f over a broad random spread of positive floats. */
  for (int i = 0; i < 4000; ++i){
    double m = pow(10.0, -6.0 + 10.0*urnd());
    float x = (float)m;
    if (!(x > 0.0f) || !isfinite(x)) continue;
    fprintf(out, "logf,%08X,,%08X\n", asu(x), asu(logf(x)));
    fprintf(out, "log10f,%08X,,%08X\n", asu(x), asu(log10f(x)));
  }
  /* Dense sweep of the TDFCND LOG10 domain, where the SunPro path is used. */
  for (int i = 0; i < 4000; ++i){
    float x = (float)(0.1 + 0.9*urnd());
    fprintf(out, "log10f,%08X,,%08X\n", asu(x), asu(log10f(x)));
  }
  /* expf over the Noah-MP BDFALL span and wider. */
  for (int i = 0; i < 6000; ++i){
    float x = (float)(-80.0 + 160.0*urnd());
    fprintf(out, "expf,%08X,,%08X\n", asu(x), asu(expf(x)));
  }
  /* powf over the exact bases Noah-MP uses plus broad random pairs. */
  const float bases[] = {7.7f, 2.0f, 2.2f, 0.57f};
  for (int i = 0; i < 4000; ++i){
    float a = bases[i % 4], b = (float)urnd();
    fprintf(out, "powf,%08X,%08X,%08X\n", asu(a), asu(b), asu(powf(a,b)));
  }
  for (int i = 0; i < 4000; ++i){
    float a = (float)(0.01 + 0.99*urnd()), b = (float)(2.0 + 13.0*urnd());
    fprintf(out, "powf,%08X,%08X,%08X\n", asu(a), asu(b), asu(powf(a,b)));
  }
  for (int i = 0; i < 4000; ++i){
    float a = (float)(0.05 + 20.0*urnd()), b = (float)(-4.0 + 8.0*urnd());
    fprintf(out, "powf,%08X,%08X,%08X\n", asu(a), asu(b), asu(powf(a,b)));
  }
  fclose(out);
  printf("done\n");
  return 0;
}
