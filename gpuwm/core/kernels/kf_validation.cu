// Read-only validation of the native Kain--Fritsch feedback bundle.
//
// Kept in a separate NVRTC module so this diagnostic cannot affect
// compilation of the numerical kf_column kernel.  Bit order matches the
// PhysicsDriver's historical first-invalid checks.

extern "C" __global__
void kf_validate_outputs(
        const real *rthcuten, const real *rqvcuten,
        const real *rqccuten, const real *rqicuten,
        const real *rqrcuten, const real *rqscuten,
        const real *nca_seconds, const real *pratec,
        unsigned int active, unsigned int *status,
        long long count_3d, long long count_2d) {
    long long index = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int invalid = 0;

    if (index < count_3d) {
        if ((active & (1u << 0)) && !isfinite(rthcuten[index]))
            invalid |= 1u << 0;
        if ((active & (1u << 1)) && !isfinite(rqvcuten[index]))
            invalid |= 1u << 1;
        if ((active & (1u << 2)) && !isfinite(rqccuten[index]))
            invalid |= 1u << 2;
        if ((active & (1u << 3)) && !isfinite(rqicuten[index]))
            invalid |= 1u << 3;
        if ((active & (1u << 4)) && !isfinite(rqrcuten[index]))
            invalid |= 1u << 4;
        if ((active & (1u << 5)) && !isfinite(rqscuten[index]))
            invalid |= 1u << 5;
    }
    if (index < count_2d) {
        if ((active & (1u << 6)) && !isfinite(nca_seconds[index]))
            invalid |= 1u << 6;
        if ((active & (1u << 7)) && !isfinite(pratec[index]))
            invalid |= 1u << 7;
    }
    if (invalid != 0) atomicOr(status, invalid);
}
