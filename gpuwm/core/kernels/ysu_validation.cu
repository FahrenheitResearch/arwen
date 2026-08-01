// Read-only validation of the native YSU output bundle.
//
// Kept in a separate NVRTC module so adding or changing this diagnostic
// cannot affect compilation of the numerical ysu_column kernel.

extern "C" __global__
void ysu_validate_outputs(
        const real *du, const real *dv, const real *dtheta,
        const real *dqv, const real *dqc, const real *dqi,
        const real *exch_h, const real *exch_m,
        const real *hpbl, const real *wstar, const real *delta,
        const real *topdown_radsum, const real *wstar3_2,
        unsigned int *status, long long count_3d, long long count_2d) {
    long long index = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int invalid = 0;
    if (index < count_3d) {
        if (!isfinite(du[index])) invalid |= 1u << 0;
        if (!isfinite(dv[index])) invalid |= 1u << 1;
        if (!isfinite(dtheta[index])) invalid |= 1u << 2;
        if (!isfinite(dqv[index])) invalid |= 1u << 3;
        if (!isfinite(dqc[index])) invalid |= 1u << 4;
        if (!isfinite(dqi[index])) invalid |= 1u << 5;
        if (!isfinite(exch_h[index])) invalid |= 1u << 6;
        if (!isfinite(exch_m[index])) invalid |= 1u << 7;
    }
    if (index < count_2d) {
        if (!isfinite(hpbl[index])) invalid |= 1u << 8;
        if (!isfinite(wstar[index])) invalid |= 1u << 10;
        if (!isfinite(delta[index])) invalid |= 1u << 11;
        if (!isfinite(topdown_radsum[index])) invalid |= 1u << 12;
        if (!isfinite(wstar3_2[index])) invalid |= 1u << 13;
    }
    if (invalid != 0) atomicOr(status, invalid);
}
