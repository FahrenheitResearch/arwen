// SPDX-License-Identifier: Apache-2.0
// Original diagnostic of terrain-owned SI surface depth. No routing equations.
extern "C" __global__ void inundation_depth(
    int n, const float *surface_depth, const int *active,
    const unsigned int *source_health, float *depth, float *maximum_depth,
    unsigned int *health) {
    int p=blockIdx.x*blockDim.x+threadIdx.x;
    if (p>=n) return;
    float missing=__int_as_float(0x7fc00000);
    if (*source_health) {
        depth[p]=missing; maximum_depth[p]=missing;
        atomicOr(health,1u); return;
    }
    if (active[p]==0) {
        depth[p]=missing; maximum_depth[p]=missing; return;
    }
    if (active[p]!=1) {
        depth[p]=missing; maximum_depth[p]=missing;
        atomicOr(health,4u); return;
    }
    float h=surface_depth[p];
    if (!isfinite(h) || h<0.f) {
        depth[p]=missing; maximum_depth[p]=missing;
        atomicOr(health,2u); return;
    }
    depth[p]=h;
    maximum_depth[p]=fmaxf(maximum_depth[p],h);
}
