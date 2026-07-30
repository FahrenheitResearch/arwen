// Production RRTMGP input validation, fused into one device scan and one
// host-visible bitset.  The predicates mirror the full host validators in
// gpuwm/core/rrtmgp.py; shape and dtype normalization remain host-side.

#define BAD_PLAY       (1u << 0)
#define BAD_PLEV       (1u << 1)
#define BAD_TLAY       (1u << 2)
#define BAD_TLEV       (1u << 3)
#define BAD_TSFC       (1u << 4)
#define BAD_QV         (1u << 5)
#define BAD_QC         (1u << 6)
#define BAD_QR         (1u << 7)
#define BAD_QI         (1u << 8)
#define BAD_QS         (1u << 9)
#define BAD_CLDFRA     (1u << 10)
#define BAD_NC         (1u << 11)
#define BAD_NR         (1u << 12)
#define BAD_NI         (1u << 13)
#define BAD_NS         (1u << 14)
#define BAD_EFFC       (1u << 15)
#define BAD_EFFR       (1u << 16)
#define BAD_EFFI       (1u << 17)
#define BAD_EFFS       (1u << 18)
#define BAD_EMISS      (1u << 19)
#define BAD_MCICA      (1u << 20)
#define BAD_THERMO     (1u << 21)
// Physical-plausibility bands for the micron effective-radius contract
// (EFFECTIVE_RADIUS_PLAUSIBLE_UM in gpuwm/core/rrtmgp.py).  A radii writer
// that emits metres (or any other metric prefix) lands outside its band on
// background-filled cells alone.  effr is interface parity only: not gated.
#define BAD_EFFC_UNIT  (1u << 22)
#define BAD_EFFI_UNIT  (1u << 23)
#define BAD_EFFS_UNIT  (1u << 24)

__device__ __forceinline__ bool outside(
    float value, float lower, float upper) {
  return !isfinite(value) || value < lower || value > upper;
}

__device__ __forceinline__ bool outside_pressure(
    float value, double lower, double upper) {
  // The table endpoints are host FP64 while profiles are FP32.  Promoting the
  // input preserves _validate_host_range's exact boundary predicate.
  return !isfinite(value) || (double)value < lower || (double)value > upper;
}

__device__ __forceinline__ bool invalid_nonnegative(float value) {
  return !isfinite(value) || value < 0.0f;
}

extern "C" __global__ void rrtmgp_validate_call(
    const float* play, const float* plev, const float* tlay,
    const float* tlev, const float* tsfc, const float* exner,
    const float* qv, const float* qc, const float* qr,
    const float* qi, const float* qs, const float* cldfra,
    const float* nc, const float* nr, const float* ni, const float* ns,
    const float* effc, const float* effr,
    const float* effi, const float* effs,
    const float* emiss, unsigned int* flags,
    int ncol, int nlay, int nemiss, int morrison, int have_effective,
    double play_lower, double play_upper,
    float temp_lower, float temp_upper,
    float effc_lower, float effc_upper,
    float effi_lower, float effi_upper,
    float effs_lower, float effs_upper) {
  const int idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int ncell = ncol * nlay;
  const int nlev_cell = ncol * (nlay + 1);
  unsigned int found = 0u;

  if (idx < ncell) {
    if (outside_pressure(play[idx], play_lower, play_upper)) found |= BAD_PLAY;
    if (outside(tlay[idx], temp_lower, temp_upper)) found |= BAD_TLAY;
    if (invalid_nonnegative(qv[idx])) found |= BAD_QV;
    if (invalid_nonnegative(qc[idx])) found |= BAD_QC;
    if (invalid_nonnegative(qr[idx])) found |= BAD_QR;
    if (invalid_nonnegative(qi[idx])) found |= BAD_QI;
    if (invalid_nonnegative(qs[idx])) found |= BAD_QS;
    if (outside(cldfra[idx], 0.0f, 1.0f)) found |= BAD_CLDFRA;
    if (morrison) {
      if (invalid_nonnegative(nc[idx])) found |= BAD_NC;
      if (invalid_nonnegative(nr[idx])) found |= BAD_NR;
      if (invalid_nonnegative(ni[idx])) found |= BAD_NI;
      if (invalid_nonnegative(ns[idx])) found |= BAD_NS;
    }
    if (have_effective) {
      if (invalid_nonnegative(effc[idx])) found |= BAD_EFFC;
      if (invalid_nonnegative(effr[idx])) found |= BAD_EFFR;
      if (invalid_nonnegative(effi[idx])) found |= BAD_EFFI;
      if (invalid_nonnegative(effs[idx])) found |= BAD_EFFS;
      if (effc[idx] < effc_lower || effc[idx] > effc_upper)
        found |= BAD_EFFC_UNIT;
      if (effi[idx] < effi_lower || effi[idx] > effi_upper)
        found |= BAD_EFFI_UNIT;
      if (effs[idx] < effs_lower || effs[idx] > effs_upper)
        found |= BAD_EFFS_UNIT;
    }
    const int col = idx / nlay;
    const int lay = idx - col * nlay;
    const float dp = fabsf(plev[col * (nlay + 1) + lay + 1]
                            - plev[col * (nlay + 1) + lay]);
    // Preserve _fluxes_to_radiation's predicates exactly: plev finiteness is
    // checked separately, while Exner historically rejects only <= 0.
    if (dp <= 0.0f || exner[idx] <= 0.0f) found |= BAD_THERMO;
  }
  if (idx < nlev_cell) {
    const float p = plev[idx];
    if (!isfinite(p) || p < 0.0f) found |= BAD_PLEV;
    if (outside(tlev[idx], temp_lower, temp_upper)) found |= BAD_TLEV;
  }
  if (idx < ncol) {
    if (outside(tsfc[idx], temp_lower, temp_upper)) found |= BAD_TSFC;
    if (play[idx * nlay] < play[idx * nlay + 1]) found |= BAD_MCICA;
  }
  if (idx < nemiss && outside(emiss[idx], 0.0f, 1.0f)) {
    found |= BAD_EMISS;
  }
  if (found != 0u) atomicOr(flags, found);
}
