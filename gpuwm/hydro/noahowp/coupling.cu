// SPDX-License-Identifier: Apache-2.0
// Catchment area-weighted exchange with the resident ArWen Noah-MP fields.
// Each CSR row is an ordered, complete catchment average. Weights are overlap
// area / catchment area, supplied by the hydrofabric reader at initialization.
__device__ __forceinline__ void catchment_exchange_cell(
    const int c, const int cells, const long long overlaps, const float dt, const long long *offsets,
    const int *indices, const float *weights,
    const float *runsrf, const float *runsub, const float *soldrain,
    const float *ecan, const float *etran, const float *edir,
    float *surface, float *subsurface, float *recharge, float *evapotranspiration,
    int *status, int *sticky_status, float *runoff_depth = nullptr) {
  float sr = 0.f, sb = 0.f, drain = 0.f, et = 0.f, sum = 0.f;
  int bad = offsets[c] < 0 || offsets[c + 1] <= offsets[c] || offsets[c + 1] > overlaps;
  if (bad) {
    surface[c] = subsurface[c] = recharge[c] = evapotranspiration[c] = __int_as_float(0x7fc00000);
    status[c] = 1;
    if (runoff_depth) runoff_depth[c] = __int_as_float(0x7fc00000);
    if (sticky_status) sticky_status[c] |= 1;
    return;
  }
  for (long long j = offsets[c]; j < offsets[c + 1]; ++j) {
    const int i = indices[j];
    const float w = weights[j];
    if (i < 0 || i >= cells || !(w >= 0.f && w <= 1.f)) {
      bad = 1; continue;
    }
    sum += w;
    sr += w * runsrf[i];
    sb += w * runsub[i];
    drain += w * soldrain[i];
    // Match the parent driver's QFX addition order.
    et += w * ((ecan[i] + edir[i]) + etran[i]);
  }
  // Runoff/drainage are step depths; ET remains an instantaneous flux.
  surface[c] = (sr * .001f) / dt;
  subsurface[c] = (sb * .001f) / dt;
  recharge[c] = (drain * .001f) / dt;
  evapotranspiration[c] = et * .001f;
  // Parent channel runoff includes surface and subsurface publication. Bottom
  // drainage is a separate soil diagnostic and must not be counted again.
  if (runoff_depth) runoff_depth[c] = (sr + sb) * .001f;
  status[c] = bad || !(sum > .9999f && sum < 1.0001f) ? 1 : 0;
  if (sticky_status) sticky_status[c] |= status[c];
}

extern "C" __global__ void catchment_exchange_routed(
    const int n, const int cells, const long long overlaps, const float dt, const long long *offsets,
    const int *indices, const float *weights,
    const float *runsrf, const float *runsub, const float *soldrain,
    const float *ecan, const float *etran, const float *edir,
    float *surface, float *subsurface, float *recharge, float *evapotranspiration,
    int *status, int *sticky_status, float *runoff_depth) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n) return;
  catchment_exchange_cell(c, cells, overlaps, dt, offsets, indices, weights,
      runsrf, runsub, soldrain, ecan, etran, edir,
      surface, subsurface, recharge, evapotranspiration, status, sticky_status, runoff_depth);
}

// Keep the original raw replay ABI. Production additionally retains every
// diagnostic until an explicit new preparation or checkpoint restore.
extern "C" __global__ void catchment_exchange(
    const int n, const int cells, const long long overlaps, const float dt, const long long *offsets,
    const int *indices, const float *weights,
    const float *runsrf, const float *runsub, const float *soldrain,
    const float *ecan, const float *etran, const float *edir,
    float *surface, float *subsurface, float *recharge, float *evapotranspiration,
    int *status) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n) return;
  catchment_exchange_cell(c, cells, overlaps, dt, offsets, indices, weights,
      runsrf, runsub, soldrain, ecan, etran, edir,
      surface, subsurface, recharge, evapotranspiration, status, nullptr);
}

extern "C" __global__ void catchment_exchange_monitored(
    const int n, const int cells, const long long overlaps, const float dt, const long long *offsets,
    const int *indices, const float *weights,
    const float *runsrf, const float *runsub, const float *soldrain,
    const float *ecan, const float *etran, const float *edir,
    float *surface, float *subsurface, float *recharge, float *evapotranspiration,
    int *status, int *sticky_status) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n) return;
  catchment_exchange_cell(c, cells, overlaps, dt, offsets, indices, weights,
      runsrf, runsub, soldrain, ecan, etran, edir,
      surface, subsurface, recharge, evapotranspiration, status, sticky_status);
}
