/* Every float32 x in [-0.39875, 0.39875] (the declination argument range,
   sin(23.5 deg) = 0.39875): the C library asinf against asinl rounded once
   to float (x86 80-bit long double, 64-bit significand), and the long double
   result against the double one, to flag any argument near a rounding
   midpoint.  Prints counts and the first few disagreements. */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
int main(void){
  uint32_t top; float lim=0.39875f; memcpy(&top,&lim,4);
  unsigned long long n=0, bad=0, near=0;
  for (int sgn=0; sgn<2; sgn++) for (uint32_t u=0; u<=top; u++){
    uint32_t w = u | (sgn ? 0x80000000u : 0u); float x; memcpy(&x,&w,4);
    float lib = asinf(x);
    long double rl = asinl((long double)x);
    float cr = (float)rl;
    float crd = (float)asin((double)x);
    n++;
    if (cr != crd) near++;
    if (memcmp(&lib,&cr,4)) { bad++; if (bad<=5) printf("x=%a lib=%a cr=%a\n",x,lib,cr); }
  }
  printf("args %llu  libm-asinf != correctly-rounded %llu  long-double vs double disagreements %llu\n", n, bad, near);
  return 0;
}
