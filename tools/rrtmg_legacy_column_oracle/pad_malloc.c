/* Heap padding for the column oracle (no arithmetic).
 *
 * WRF v4.6.1 RRTMG_SWRAD writes one element past its automatic array o31d
 * when o3input /= 2: phys/module_ra_rrtmg_sw.F:10946-10950 loops
 * k = kts, kte+1 and stores o31d(k), declared kts:kte.  The longwave twin
 * guards the same store (module_ra_rrtmg_lw.F:12416, "if (k.le.kte)").
 * o31d is never read after that loop, so the stray store has no numerical
 * meaning; at -O0 it lands on glibc's next heap chunk header and the
 * process aborts in free().  Padding every allocation by 64 bytes gives the
 * stray store slack to land in.  -fcheck=bounds reports no other
 * out-of-bounds access on any oracle regime.  WOOF does not reproduce the
 * store (it is a WRF bug, documented, not copied).
 */
#include <stddef.h>
extern void *__libc_malloc(size_t);
extern void *__libc_calloc(size_t, size_t);
extern void *__libc_realloc(void *, size_t);
void *malloc(size_t n) { return __libc_malloc(n + 64); }
void *calloc(size_t a, size_t b) { return __libc_calloc(1, a * b + 64); }
void *realloc(void *p, size_t n) { return __libc_realloc(p, n + 64); }
