// phy_prep's pressure-to-Exner statement, with WOOF's own float32 power.
// The caller supplies contiguous float32 arrays of any shape. One thread
// per word; the destination may alias the pressure array.
extern "C" __global__ void physics_exner(
    unsigned long long n, const float* pressure, float* exner)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float ratio;
    asm("div.rn.f32 %0, %1, %2;" : "=f"(ratio) : "f"(pressure[i]), "f"(P0));
    exner[i] = gfk_pow(ratio, RCP);
}
