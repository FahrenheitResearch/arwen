// gpuwm/core/kernels/saxpy.cu
extern "C" __global__
void saxpy(real a, const real* x, const real* y, real* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a * x[i] + y[i];
}

extern "C" __global__
void emit_g(real* out) { out[0] = G; }
