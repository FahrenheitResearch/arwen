// The chem mass ledger's reduction (gpuwm/core/chem_driver.py).
//
// One block per (field, model row k*ny + j): the block sums
// q * (c1h(k) * (mub + mup) + c2h(k)) / msft^2 over the row's nx columns in
// float64, in a fixed tree order, and writes the row's partial times
// -dnw(k).  The caller sums the (nfields, nz*ny) partials along the row axis
// and scales by dx*dy/g and the row's units, so each species' mass is one
// read of its field with no full-size temporary, and the same inputs give
// the same bytes on every call.  A diagnostic: it moves no model word.
#define LEDGER_THREADS 256

extern "C" __global__ void chem_row_mass(
    const unsigned long long* fields, const float* mub2d, const float* mup,
    const float* c1h, const float* c2h, const float* dnw, const float* msft,
    int has_msf, int nz, int ny, int nx, double* partial)
{
    __shared__ double sh[LEDGER_THREADS];
    const int row = blockIdx.x;
    const int f = blockIdx.y;
    const int k = row / ny;
    const int j = row - k * ny;
    const float* q = reinterpret_cast<const float*>(fields[f]);
    const double c1 = (double)c1h[k];
    const double c2 = (double)c2h[k];
    const size_t plane = (size_t)j * nx;
    const size_t base = ((size_t)k * ny + j) * nx;
    double acc = 0.0;
    for (int i = threadIdx.x; i < nx; i += LEDGER_THREADS) {
        double w = c1 * ((double)mub2d[plane + i] + (double)mup[plane + i])
                   + c2;
        if (has_msf) {
            const double m = (double)msft[plane + i];
            w = w / (m * m);
        }
        acc += (double)q[base + i] * w;
    }
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int s = LEDGER_THREADS / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sh[threadIdx.x] += sh[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        partial[(size_t)f * nz * ny + row] = sh[0] * (-(double)dnw[k]);
    }
}
