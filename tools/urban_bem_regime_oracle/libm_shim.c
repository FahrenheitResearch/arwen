/* Attribution control for the BEP+BEM regime oracle (build.sh, BEM_LIBM_SHIM=1).
 *
 * Linked ahead of the C library, these replace the float atanf, asinf, acosf,
 * tanf and log10f that the Fortran ATAN, ASIN, ACOS, TAN and LOG10 intrinsics
 * call at -O0 with the double routine rounded once to float.  On a C library
 * whose float versions of these five are correctly rounded the shim changes
 * nothing; on one whose are not, the words it changes are exactly the words
 * those calls decide.  No other libm entry point is touched. */
#include <math.h>

float atanf(float x) { return (float)atan((double)x); }
float asinf(float x) { return (float)asin((double)x); }
float acosf(float x) { return (float)acos((double)x); }
float tanf(float x) { return (float)tan((double)x); }
float log10f(float x) { return (float)log10((double)x); }
