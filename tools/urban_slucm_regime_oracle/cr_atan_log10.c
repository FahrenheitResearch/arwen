/* Attribution control for the SLUCM regime oracle (build.sh, UCM_LIBM_SHIM=1).
 *
 * Linked ahead of the C library, these replace the float atanf and log10f
 * the Fortran ATAN and LOG10 intrinsics call at -O0 with the double routine
 * rounded once to float.  On a C library whose float atanf/log10f are
 * correctly rounded the shim changes nothing; on one whose are not, the rows
 * it changes are exactly the rows those two calls decide.  No other libm
 * entry point is touched. */
#include <math.h>

float atanf(float x) { return (float)atan((double)x); }
float log10f(float x) { return (float)log10((double)x); }
