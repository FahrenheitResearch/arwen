// Test-only entries. Production callers include portable_libm64.cuh.
extern "C" __global__ void plm_grade64(const double* x, const double* y,
                                      double* out, unsigned long long n, int op) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double a = x[i];
    out[i] = op == 0 ? plm_exp(a) : op == 1 ? plm_log(a) :
             op == 2 ? plm_log1p(a) : plm_pow(a, y[i]);
}
extern "C" __global__ void plm_grade32(const float* x, float* out, unsigned long long n) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = plm_log1pf(x[i]);
}
