// Per-step device-side reduction of the km_opt=2 prognostic-TKE budget.
//
// One launch per model step.  Every term is a COUPLED tendency in the same
// units as the carrying buffer the RK update consumes ((c1*mu+c2) * m2 s-3),
// so the closure identity
//
//     storage == shear + buoyancy + dissipation + limiter
//                + diffusion_h + diffusion_v + diffusion_6th
//                + transport + clip
//
// holds cell by cell up to FP32 rounding, and its violation is the residual
// the receipt commits.  Accumulation is FP64 per (term, level): the terms
// are FP32 fields whose horizontal sums span many orders of magnitude, and a
// budget whose residual is dominated by its own summation is not a budget.
extern "C" __global__
void tke_budget_accumulate(const real* terms,     // (nfield, nz, ny, nx)
                           const real* tke0, const real* tke,
                           const real* raw,
                           const real* mu0, const real* mu,
                           const real* c1, const real* c2,
                           real rdt,
                           double* acc,           // (nfield + 2, nz)
                           int nfield,
                           int nz, int ny, int nx)
{
    extern __shared__ double sdata[];
    const int k = blockIdx.x;
    const int t = blockIdx.y;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const size_t plane = (size_t)ny * (size_t)nx;

    double sum = 0.0;
    for (size_t c = tid; c < plane; c += (size_t)nthreads) {
        const size_t idx = (size_t)k * plane + c;
        double v;
        if (t < nfield) {
            v = (double)terms[(size_t)t * (size_t)nz * plane + idx];
        } else {
            const double chm0 = (double)c1[k] * (double)mu0[c] + (double)c2[k];
            const double chm = (double)c1[k] * (double)mu[c] + (double)c2[k];
            if (t == nfield) {
                // storage: d(coupled e)/dt over the completed step.
                v = (chm * (double)tke[idx] - chm0 * (double)tke0[idx])
                    * (double)rdt;
            } else {
                // clip: what bound_tke (and the PD zero clamp) added or
                // removed at the final update.  A non-conservative term by
                // construction; it is exported, never hidden in a residual.
                v = chm * ((double)tke[idx] - (double)raw[idx]) * (double)rdt;
            }
        }
        sum += v;
    }

    sdata[tid] = sum;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) acc[(size_t)t * (size_t)nz + k] += sdata[0];
}
