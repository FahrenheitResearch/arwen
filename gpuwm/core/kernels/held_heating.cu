extern "C" __global__ void add_held_heating(
    float* rth, const float* hd, const float* mub, const float* mup,
    const float* c1, const float* c2, const float* msft,
    unsigned long long n, int ncol, int mapped) {
    unsigned long long at = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (at >= n) return;
    int col = at % ncol;
    int k = at / ncol;
    float mu = __fadd_rn(mub[col], mup[col]);
    float coupling = __fadd_rn(__fmul_rn(c1[k], mu), c2[k]);
    float heating = __fmul_rn(coupling, hd[at]);
    if (mapped) heating = __fdiv_rn(heating, msft[col]);
    rth[at] = __fadd_rn(rth[at], heating);
}
