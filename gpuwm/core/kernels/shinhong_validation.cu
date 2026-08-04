// Read-only validation of the native Shin-Hong output bundle.
//
// Kept in a separate NVRTC module so adding or changing this diagnostic
// cannot affect compilation of the numerical shinhong_column kernel.
//
// Bit numbering follows the launcher's output order (gpuwm/core/shinhong.py
// _SHINHONG_OUTPUTS); kpbl (bit 10) is int32 and necessarily finite, so the
// kernel never sets it.

extern "C" __global__
void shinhong_validate_outputs(
        const real *du, const real *dv, const real *dtheta,
        const real *dqv, const real *dqc, const real *dqi,
        const real *exch_h, const real *tke, const real *el,
        const real *hpbl, const real *wstar, const real *delta,
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
        if (!isfinite(tke[index])) invalid |= 1u << 7;
        if (!isfinite(el[index])) invalid |= 1u << 8;
    }
    if (index < count_2d) {
        if (!isfinite(hpbl[index])) invalid |= 1u << 9;
        if (!isfinite(wstar[index])) invalid |= 1u << 11;
        if (!isfinite(delta[index])) invalid |= 1u << 12;
    }
    if (invalid != 0) atomicOr(status, invalid);
}
