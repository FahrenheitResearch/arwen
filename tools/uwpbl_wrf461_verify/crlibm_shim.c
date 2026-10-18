/* Correctly rounded cos and acos for the "crlibm" oracle variant.
 *
 * Linked into the pristine -O0 objects ahead of libm, these definitions are
 * what the WRF objects' calls to cos and acos resolve to (the executable's
 * own symbols take precedence over libm.so's).  Every other libm call
 * (exp, log, pow, sqrt) still goes to the host libm.  The variant isolates
 * one question: do the port and WRF differ ONLY where the host libm's
 * binary64 cos/acos round differently from the correctly rounded result
 * the port's kernel computes?  cr_cos and cr_acos are the CORE-MATH
 * sources in tools/uwpbl_wrf471_oracle/libm64/upstream (MIT licence).
 */
double cr_cos(double);
double cr_acos(double);

double cos(double x) { return cr_cos(x); }
double acos(double x) { return cr_acos(x); }
