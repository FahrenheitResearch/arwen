/* MPFR references for the rw-crmath comparison harness. */
#include <stdio.h>
#include <gmp.h>
#include <mpfr.h>

enum { SINF, COSF, TANF, ATANF, EXPF, LOGF, LOG10F };

float ref1f(int op, float x) {
  mpfr_set_emin(-148);
  mpfr_set_emax(128);
  mpfr_t y;
  mpfr_init2(y, 24);
  mpfr_set_flt(y, x, MPFR_RNDN);
  int inex = 0;
  switch (op) {
  case SINF: inex = mpfr_sin(y, y, MPFR_RNDN); break;
  case COSF: inex = mpfr_cos(y, y, MPFR_RNDN); break;
  case TANF: inex = mpfr_tan(y, y, MPFR_RNDN); break;
  case ATANF: inex = mpfr_atan(y, y, MPFR_RNDN); break;
  case EXPF: inex = mpfr_exp(y, y, MPFR_RNDN); break;
  case LOGF: inex = mpfr_log(y, y, MPFR_RNDN); break;
  case LOG10F: inex = mpfr_log10(y, y, MPFR_RNDN); break;
  }
  inex = mpfr_subnormalize(y, inex, MPFR_RNDN);
  float r = mpfr_get_flt(y, MPFR_RNDN);
  mpfr_clear(y);
  return r;
}

/* op 0: pow(x, y); op 1: atan2(x, y) i.e. atan2 of the first argument over the second */
float ref2f(int op, float a, float b) {
  mpfr_set_emin(-148);
  mpfr_set_emax(128);
  mpfr_t ma, mb, r;
  mpfr_init2(ma, 24);
  mpfr_init2(mb, 24);
  mpfr_init2(r, 24);
  mpfr_set_flt(ma, a, MPFR_RNDN);
  mpfr_set_flt(mb, b, MPFR_RNDN);
  int inex = op == 0 ? mpfr_pow(r, ma, mb, MPFR_RNDN) : mpfr_atan2(r, ma, mb, MPFR_RNDN);
  inex = mpfr_subnormalize(r, inex, MPFR_RNDN);
  float out = mpfr_get_flt(r, MPFR_RNDN);
  mpfr_clear(ma);
  mpfr_clear(mb);
  mpfr_clear(r);
  return out;
}

/* op 0: asin, op 1: acos */
double ref1d(int op, double x) {
  mpfr_set_emin(-1073);
  mpfr_set_emax(1024);
  mpfr_t y;
  mpfr_init2(y, 53);
  mpfr_set_d(y, x, MPFR_RNDN);
  int inex = op == 0 ? mpfr_asin(y, y, MPFR_RNDN) : mpfr_acos(y, y, MPFR_RNDN);
  inex = mpfr_subnormalize(y, inex, MPFR_RNDN);
  double r = mpfr_get_d(y, MPFR_RNDN);
  mpfr_clear(y);
  return r;
}

void ref_free_cache(void) { mpfr_free_cache(); }
