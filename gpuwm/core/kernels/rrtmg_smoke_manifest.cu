// Declared linear interpolation of immutable prescribed profiles in time.
extern "C" __global__ void rrtmg_smoke_time_blend(
    long long count, float fraction, const float* __restrict__ left,
    const float* __restrict__ right, float* __restrict__ output)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float old_weight = __fsub_rn(1.0f, fraction);
    output[i] = __fadd_rn(__fmul_rn(old_weight, left[i]),
                         __fmul_rn(fraction, right[i]));
}
