/* Turn on the x86-64 SSE flush-to-zero and denormals-are-zero modes for the
 * rest of the process.  Called once by run_wsm6_columns_ftz (run_refute.sh)
 * before WRF's WSM6 runs, so WRF's own float words show what flushing does. */
#include <xmmintrin.h>
#include <pmmintrin.h>

void set_ftz_daz_(void)
{
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
}
