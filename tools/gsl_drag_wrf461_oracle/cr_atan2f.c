/* Diagnostic only, never the oracle of record.  A correctly rounded
 * single-precision atan2 (the double result rounded once to float), linked
 * into a second copy of the -O0 oracle so a sector-choice difference can be
 * attributed to the host C library's atan2f.  glibc 2.39's atan2f is within
 * one ULP but not correctly rounded; glibc 2.43's is correctly rounded. */
#include <math.h>
float atan2f(float y, float x) { return (float)atan2((double)y, (double)x); }
