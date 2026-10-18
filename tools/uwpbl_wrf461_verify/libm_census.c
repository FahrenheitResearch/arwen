/* cos/acos census for the oracle runs: how often the host libm and the
 * correctly rounded functions disagree INSIDE the scheme.
 *
 * Linked into the pristine -O0 objects like crlibm_shim.c, but these
 * definitions return the HOST libm's result (fetched with dlsym from
 * libm.so.6), so the run's outputs are the reference's; each call also
 * evaluates cr_cos / cr_acos and counts calls and disagreements.  At exit
 * one line goes to stderr:
 *   libm-census acos_calls=N acos_differ=D cos_calls=N cos_differ=D
 * The crlibm variant answers whether a disagreement reached a float32
 * output; this answers whether the cases exercised one at all.
 */
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

double cr_cos(double);
double cr_acos(double);

static double (*host_cos)(double), (*host_acos)(double);
static unsigned long long n_cos, d_cos, n_acos, d_acos;

static void bind(void)
{
    void *h = dlopen("libm.so.6", RTLD_NOW | RTLD_LOCAL);
    host_cos = (double (*)(double))dlsym(h, "cos");
    host_acos = (double (*)(double))dlsym(h, "acos");
}

static int same(double a, double b)
{
    uint64_t x, y;
    memcpy(&x, &a, 8); memcpy(&y, &b, 8);
    return x == y;
}

double cos(double x)
{
    if (!host_cos) bind();
    double r = host_cos(x);
    n_cos++;
    if (!same(r, cr_cos(x))) d_cos++;
    return r;
}

double acos(double x)
{
    if (!host_acos) bind();
    double r = host_acos(x);
    n_acos++;
    if (!same(r, cr_acos(x))) d_acos++;
    return r;
}

__attribute__((destructor)) static void report(void)
{
    fprintf(stderr, "libm-census acos_calls=%llu acos_differ=%llu cos_calls=%llu cos_differ=%llu\n",
            n_acos, d_acos, n_cos, d_cos);
}
