// Read-only validation of PhysicsDriver's canonical surface diagnostics.
//
// The scheme kernels remain in their own translation units.  This compact
// status reduction changes only how their completed outputs are checked.

extern "C" __global__
void microphysics_validate_outputs(
        const real *rainnc, const real *rainncv, const real *sr,
        const real *snownc, const real *snowncv,
        const real *graupelnc, const real *graupelncv,
        const real *hailnc, const real *hailncv,
        unsigned int active, real sr_upper,
        unsigned int *status, long long count) {
    long long index = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= count) return;

    unsigned int invalid = 0;
    if ((active & (1u << 0)) && !isfinite(rainnc[index]))
        invalid |= 1u << 0;
    if ((active & (1u << 1)) && !isfinite(rainncv[index]))
        invalid |= 1u << 1;
    if ((active & (1u << 2)) && !isfinite(sr[index]))
        invalid |= 1u << 2;
    if ((active & (1u << 3)) && !isfinite(snownc[index]))
        invalid |= 1u << 3;
    if ((active & (1u << 4)) && !isfinite(snowncv[index]))
        invalid |= 1u << 4;
    if ((active & (1u << 5)) && !isfinite(graupelnc[index]))
        invalid |= 1u << 5;
    if ((active & (1u << 6)) && !isfinite(graupelncv[index]))
        invalid |= 1u << 6;
    if ((active & (1u << 7)) && !isfinite(hailnc[index]))
        invalid |= 1u << 7;
    if ((active & (1u << 8)) && !isfinite(hailncv[index]))
        invalid |= 1u << 8;
    if (sr[index] < 0.0f) invalid |= 1u << 16;
    if (sr[index] > sr_upper) invalid |= 1u << 17;
    if (invalid != 0) atomicOr(status, invalid);
}
