/* Correctly rounded float libm for the oracle's second build.
 *
 * Each function returns the float nearest the double-precision result.
 * The double functions are accurate to well under an ulp of double, so the
 * rounding to float is the correctly rounded float except for an argument
 * whose exact result lies within ~2^-29 relative of a float tie; the port
 * (gpuwm/core/kernels/topo_radiation.cu) evaluates every transcendental
 * the same way, so the two agree bit for bit wherever the arithmetic does.
 * Linked ahead of -lm, these definitions take precedence over glibc's.
 */
#include <math.h>

float sinf(float x) { return (float)sin((double)x); }
float cosf(float x) { return (float)cos((double)x); }
float tanf(float x) { return (float)tan((double)x); }
float asinf(float x) { return (float)asin((double)x); }
float acosf(float x) { return (float)acos((double)x); }
float atanf(float x) { return (float)atan((double)x); }
float atan2f(float y, float x) { return (float)atan2((double)y, (double)x); }
float expf(float x) { return (float)exp((double)x); }
float logf(float x) { return (float)log((double)x); }
float powf(float x, float y) { return (float)pow((double)x, (double)y); }
/* gcc joins sin(x) and cos(x) of one argument into one sincosf call. */
void sincosf(float x, float *s, float *c)
{
    *s = (float)sin((double)x);
    *c = (float)cos((double)x);
}
