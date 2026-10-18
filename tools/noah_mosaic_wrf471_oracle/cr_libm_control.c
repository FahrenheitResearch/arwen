/* Oracle control for build.sh with NOAH_MOSAIC_CR_LIBM=1.
 *
 * Correctly rounded float32 atanf and log10f: the double routine, rounded
 * once to float.  Linked ahead of the C library, these replace its float
 * routines for every WRF object in the oracle, so the oracle's words no
 * longer depend on the host C library's float rounding (glibc 2.39's atanf
 * and log10f are not correctly rounded; 2.43's are).  This is the same
 * rounding the GPU port uses for ATAN and LOG10.  No physics lives here.
 */
#include <math.h>

float atanf(float x) { return (float)atan((double)x); }
float log10f(float x) { return (float)log10((double)x); }
