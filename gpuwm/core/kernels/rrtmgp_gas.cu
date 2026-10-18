// RRTMGP gas optical depths and Planck sources.
//
// Transcribed from earth-system-radiation/rte-rrtmgp fa107a1:
// rrtmgp/kernels/mo_gas_optics_rrtmgp_kernels.F90
//   interpolation                    lines 37-170
//   compute_tau_absorption           lines 175-336 (major + both minors)
//   compute_tau_rayleigh             lines 506-565
//   compute_Planck_source            lines 567-700
// rrtmgp/frontend/mo_gas_optics_rrtmgp.F90
//   compute_gas_taus (col_gas = vmr * col_dry), combine_abs_and_rayleigh,
//   init_abs_coeffs (press_ref_log, *_delta, press_ref_trop_log)
// and rte/kernels/mo_gas_optics_utils.F90:get_layer_number.
//
// The unit compiles with --fmad=false and calls WOOF's own float32
// logf/expf (rrtmgp_libm.cuh), so each statement below is the upstream
// single-precision build's arithmetic: one binary32 rounding per operator,
// in the Fortran's association.  tools/rrtmgp_rte_upstream_oracle grades
// every output of this unit against that build at 0 ULP.  The shared
// words (col_dry, eta weights, Planck interpolation, pfrac) live in
// rrtmgp_planck_common.cuh, which rrtmgp_rte.cu includes as well.

__device__ __forceinline__ int km_index(
    int jt, int je, int jp, int g, int neta, int npressk, int ngpt) {
  return (((jt * neta + je) * npressk + jp) * ngpt + g);
}

__device__ __forceinline__ int k2_index(
    int jt, int je, int k, int neta, int nk) {
  return ((jt * neta + je) * nk + k);
}

// interpolation(), the per-cell part: temperature and pressure indices and
// fractions and the lower/upper atmosphere flag.  The reference constants
// are init_abs_coeffs' (temp_ref_delta, press_ref_log, press_ref_log_delta,
// press_ref_trop_log) recomputed per thread from the float32 tables, then
// interpolation()'s own reciprocals and its exp(press_ref_trop_log).
//
// One deliberate divergence from fa107a1: the Fortran evaluates
//   ftemp = (tlay - temp_ref(jtemp_)) * temp_ref_delta_inv
// with the UNCLAMPED index jtemp_, while the tables are read at the clamped
// jtemp.  At the table's top temperature (tlay within a float32 step of
// temp_ref_max = 355 K) jtemp_ = ntemp and jtemp = ntemp-1, so the Fortran
// reads the absorption one reference temperature (15 K) too cold.  Here
// ftemp is measured from the node actually read.  Below the top the
// indices are equal and so are the operations.
#define RRTMGP_PREPASS_BODY                                                 \
  const float p = play[cell];                                               \
  const float t = tlay[cell];                                               \
  const float temp_ref_min = temp_ref[0];                                   \
  const float temp_ref_delta = (temp_ref[ntemp - 1] - temp_ref[0])          \
                               / (float)(ntemp - 1);                        \
  const float temp_ref_delta_inv = 1.0f / temp_ref_delta;                   \
  const float press_ref_log_1 = rrtmgp_logf(press_ref[0]);                  \
  const float press_ref_log_delta =                                         \
      (rrtmgp_logf(press_ref[npres - 1]) - rrtmgp_logf(press_ref[0]))       \
      / (float)(npres - 1);                                                 \
  const float press_ref_log_delta_inv = 1.0f / press_ref_log_delta;         \
  const float trop = rrtmgp_expf(rrtmgp_logf(press_ref_trop));              \
  const int jtemp_ = (int)((t - (temp_ref_min - temp_ref_delta))            \
                           * temp_ref_delta_inv);                           \
  const int jtemp = max(1, min(ntemp - 1, jtemp_));                         \
  const float ftemp = (t - temp_ref[jtemp - 1]) * temp_ref_delta_inv;       \
  const float locpress = 1.0f + (rrtmgp_logf(p) - press_ref_log_1)          \
                                * press_ref_log_delta_inv;                  \
  const float jpress_aint =                                                 \
      fminf((float)(npres - 1), fmaxf(1.0f, truncf(locpress)));             \
  const int jpress = (int)jpress_aint;                                      \
  const int iatm = p > trop ? 0 : 1;                                        \
  iatm_out[cell] = iatm;                                                    \
  jt_out[cell] = jtemp - 1;                                                 \
  jp_out[cell] = (jpress - 1) + iatm;                                       \
  ftemp_out[cell] = ftemp;                                                  \
  fpress_out[cell] = locpress - jpress_aint;

extern "C" __global__ void rrtmgp_interpolation_prepass(
    const float* play, const float* tlay,
    const float* press_ref, const float* temp_ref, float press_ref_trop,
    int* iatm_out, int* jt_out, int* jp_out,
    float* ftemp_out, float* fpress_out,
    int ncell, int ntemp, int npres) {
  const int cell = blockDim.x * blockIdx.x + threadIdx.x;
  if (cell >= ncell) return;
  RRTMGP_PREPASS_BODY
}

// Test oracle: the same statements under a second entry point, so an
// exact-array test notices a change to the launch indexing or to either
// copy of the stored FP32 values.
extern "C" __global__ void rrtmgp_interpolation_inline_reference(
    const float* play, const float* tlay,
    const float* press_ref, const float* temp_ref, float press_ref_trop,
    int* iatm_out, int* jt_out, int* jp_out,
    float* ftemp_out, float* fpress_out,
    int ncell, int ntemp, int npres) {
  const int cell = blockDim.x * blockIdx.x + threadIdx.x;
  if (cell >= ncell) return;
  RRTMGP_PREPASS_BODY
}

// Minor-gas scaling for one (cell, minor contributor): the head of
// gas_optical_depths_minor's inner loop.  col_gas(:, 0) is col_dry.
__device__ __forceinline__ float rrtmgp_minor_scaling(
    const float* v, float col_dry, float p, float t, int idx_h2o,
    int idx_minor, int scales_with_density, int idx_scaling,
    int by_complement) {
  const float pa_to_hpa = 0.01f;
  float scaling = idx_minor == 0 ? col_dry : v[idx_minor] * col_dry;
  if (scales_with_density) {
    scaling = scaling * (pa_to_hpa * p / t);
    if (idx_scaling > 0) {
      const float vmr_fact = 1.0f / col_dry;
      const float col_h2o = v[idx_h2o] * col_dry;
      const float dry_fact = 1.0f / (1.0f + col_h2o * vmr_fact);
      const float col_s = v[idx_scaling] * col_dry;
      scaling = by_complement
          ? scaling * (1.0f - col_s * vmr_fact * dry_fact)
          : scaling * (col_s * vmr_fact * dry_fact);
    }
  }
  return scaling;
}

// The four weights of one (flavor, reference temperature) slice, loaded
// together from shared memory.
struct __align__(16) RRTMGPFlavorWeights {
  float cmix;
  int je;
  float fm0;
  float fm1;
};

// tau for one (cell, g-point): major (interpolate3D_byflav with scaling =
// col_mix), then every minor contributor whose g-point range covers gpt in
// ascending order (the order gas_optical_depths_minor adds them), then for
// the SW Rayleigh and combine_abs_and_rayleigh.
#define RRTMGP_GAS_GPOINT_BODY(WEIGHTS)                                     \
  float tau_abs;                                                            \
  float fm[2][2];                                                           \
  int je[2];                                                                \
  {                                                                         \
    float part[2], cmix[2];                                                 \
    for (int itemp = 0; itemp < 2; ++itemp) {                               \
      const int jtr = jt + itemp;                                           \
      WEIGHTS                                                               \
      const float f11 = (1.0f - fpress) * fm[0][itemp];                     \
      const float f21 = (1.0f - fpress) * fm[1][itemp];                     \
      const float f12 = fpress * fm[0][itemp];                              \
      const float f22 = fpress * fm[1][itemp];                              \
      part[itemp] =                                                         \
          f11 * kmajor[km_index(jtr, je[itemp], jp, gpt,                    \
                                neta, npressk, ngpt)]                       \
        + f21 * kmajor[km_index(jtr, je[itemp] + 1, jp, gpt,                \
                                neta, npressk, ngpt)]                       \
        + f12 * kmajor[km_index(jtr, je[itemp], jp + 1, gpt,                \
                                neta, npressk, ngpt)]                       \
        + f22 * kmajor[km_index(jtr, je[itemp] + 1, jp + 1, gpt,            \
                                neta, npressk, ngpt)];                      \
    }                                                                       \
    tau_abs = 0.0f + (cmix[0] * part[0] + cmix[1] * part[1]);               \
  }                                                                         \
  const int q_end = mstart[gpt + 1];                                        \
  for (int q = mstart[gpt]; q < q_end; ++q) {                               \
    const int m = mlist[q];                                                 \
    const int gs = limits[m * 2];                                           \
    const int kk = starts[m] + (gpt - gs);                                  \
    const float tau_minor =                                                 \
        fm[0][0] * kminor[k2_index(jt, je[0], kk, neta, nk)]                \
      + fm[1][0] * kminor[k2_index(jt, je[0] + 1, kk, neta, nk)]            \
      + fm[0][1] * kminor[k2_index(jt + 1, je[1], kk, neta, nk)]            \
      + fm[1][1] * kminor[k2_index(jt + 1, je[1] + 1, kk, neta, nk)];       \
    tau_abs = tau_abs + s_scaling[m] * tau_minor;                           \
  }                                                                         \
  const int out = cell * ngpt + gpt;                                        \
  if (is_sw) {                                                              \
    const float* kr = rayleigh + iatm * ntemp * neta * ngpt;                \
    const float k =                                                         \
        fm[0][0] * kr[(jt * neta + je[0]) * ngpt + gpt]                     \
      + fm[1][0] * kr[(jt * neta + je[0] + 1) * ngpt + gpt]                 \
      + fm[0][1] * kr[((jt + 1) * neta + je[1]) * ngpt + gpt]               \
      + fm[1][1] * kr[((jt + 1) * neta + je[1] + 1) * ngpt + gpt];          \
    const float tau_ray = k * (col_h2o + col_dry);                          \
    const float total = tau_abs + tau_ray;                                  \
    tau[out] = total;                                                       \
    ssa[out] = total > 2.0f * RRTMGP_TINY ? tau_ray / total : 0.0f;         \
  } else {                                                                  \
    tau[out] = tau_abs;                                                     \
  }

// Per-cell preamble shared by the production and the frozen reference
// kernel: col_dry, col_gas lookups and the interpolation metadata.
#define RRTMGP_GAS_CELL_PREAMBLE                                            \
  const int npressk = npres + 1;                                            \
  const int cell = col * nlay + lay;                                        \
  const float p = play[cell];                                               \
  const float t = tlay[cell];                                               \
  const float* v = vmr + cell * (ngas + 1);                                 \
  const float col_dry = rrtmgp_col_dry(                                     \
      plev[col * (nlay + 1) + lay], plev[col * (nlay + 1) + lay + 1],       \
      v[idx_h2o]);                                                          \
  const float col_h2o = v[idx_h2o] * col_dry;                               \
  const int iatm = iatm_meta[cell];                                         \
  const int jt = jt_meta[cell];                                             \
  const int jp = jp_meta[cell];                                             \
  const float ftemp = ftemp_meta[cell];                                     \
  const float fpress = fpress_meta[cell];                                   \
  const int nk = iatm == 0 ? nk_lower : nk_upper;                           \
  const float* kminor = iatm == 0 ? kminor_lower : kminor_upper;            \
  const int* limits = iatm == 0 ? minor_limits_lower : minor_limits_upper;  \
  const unsigned char* density = iatm == 0 ? density_lower : density_upper; \
  const unsigned char* complement = iatm == 0                               \
      ? complement_lower : complement_upper;                                \
  const int* idx_minor = iatm == 0 ? idx_minor_lower : idx_minor_upper;     \
  const int* idx_scaling = iatm == 0                                        \
      ? idx_scaling_lower : idx_scaling_upper;                              \
  const int* starts = iatm == 0 ? kminor_start_lower : kminor_start_upper;  \
  const int* mstart = iatm == 0                                             \
      ? minor_gpt_start_lower : minor_gpt_start_upper;                      \
  const int* mlist = iatm == 0                                              \
      ? minor_gpt_list_lower : minor_gpt_list_upper;                        \
  const int nminor = iatm == 0 ? nminor_lower : nminor_upper;

#define RRTMGP_GAS_ARGS                                                     \
    const float* play, const float* plev, const float* tlay,                \
    const float* vmr, const int* iatm_meta, const int* jt_meta,             \
    const int* jp_meta, const float* ftemp_meta, const float* fpress_meta,  \
    const float* vmr_ref, const int* flavor, const int* gpoint_flavor,      \
    const float* kmajor,                                                    \
    const float* kminor_lower, const float* kminor_upper,                   \
    const int* minor_limits_lower, const int* minor_limits_upper,           \
    const unsigned char* density_lower, const unsigned char* density_upper, \
    const unsigned char* complement_lower,                                  \
    const unsigned char* complement_upper,                                  \
    const int* idx_minor_lower, const int* idx_minor_upper,                 \
    const int* idx_scaling_lower, const int* idx_scaling_upper,             \
    const int* kminor_start_lower, const int* kminor_start_upper,           \
    const int* minor_gpt_start_lower, const int* minor_gpt_list_lower,      \
    const int* minor_gpt_start_upper, const int* minor_gpt_list_upper,      \
    const float* rayleigh,                                                  \
    float* tau, float* ssa,                                                 \
    int ncol, int nlay, int ngas, int nflav, int ngpt,                      \
    int ntemp, int npres, int neta,                                         \
    int nk_lower, int nk_upper, int idx_h2o, int is_sw,                     \
    int nminor_lower, int nminor_upper

// Production: four cells per 128-thread block, one warp per cell.  The
// warp computes each minor scaling and each (flavor, temperature) weight
// slice once into shared memory, then strides the g-points.  Every output
// element is a pure function of its (cell, g-point) and is evaluated by
// the same statements as the frozen reference below.
extern "C" __global__ void rrtmgp_gas_optics(RRTMGP_GAS_ARGS) {
  const int cell_id = blockIdx.x * 4 + threadIdx.x / 32;
  if (cell_id >= ncol * nlay) return;
  const int col = cell_id / nlay;
  const int lay = cell_id - col * nlay;
  const int minor_slots = (max(nminor_lower, nminor_upper) + 3) & ~3;
  extern __shared__ float s_storage[];
  float* s_scaling = s_storage + (threadIdx.x / 32)
      * (minor_slots + 8 * nflav);
  RRTMGP_GAS_CELL_PREAMBLE
  for (int mm = threadIdx.x % 32; mm < nminor; mm += 32) {
    s_scaling[mm] = rrtmgp_minor_scaling(
        v, col_dry, p, t, idx_h2o, idx_minor[mm], density[mm],
        idx_scaling[mm], complement[mm]);
  }
  RRTMGPFlavorWeights* s_flavor = reinterpret_cast<RRTMGPFlavorWeights*>(
      s_scaling + minor_slots);
  for (int f = threadIdx.x % 32; f < nflav * 2; f += 32) {
    const int iflav = f / 2;
    const int itemp = f % 2;
    const int gas1 = flavor[iflav * 2];
    const int gas2 = flavor[iflav * 2 + 1];
    const int jtr = jt + itemp;
    const float ratio = vmr_ref[(iatm * (ngas + 1) + gas1) * ntemp + jtr]
                        / vmr_ref[(iatm * (ngas + 1) + gas2) * ntemp + jtr];
    const float col1 = gas1 == 0 ? col_dry : v[gas1] * col_dry;
    const float col2 = gas2 == 0 ? col_dry : v[gas2] * col_dry;
    RRTMGPFlavorWeights w;
    rrtmgp_eta_weights(col1, col2, ratio, ftemp, itemp, neta,
                       &w.cmix, &w.je, &w.fm0, &w.fm1);
    s_flavor[f] = w;
  }
  __syncwarp();
  for (int gpt = threadIdx.x % 32; gpt < ngpt; gpt += 32) {
    const int iflav = gpoint_flavor[iatm * ngpt + gpt];
    RRTMGP_GAS_GPOINT_BODY(
      const RRTMGPFlavorWeights w = s_flavor[iflav * 2 + itemp];
      cmix[itemp] = w.cmix; je[itemp] = w.je;
      fm[0][itemp] = w.fm0; fm[1][itemp] = w.fm1;)
  }
}

// Frozen reference for exact-output tests: one block per cell, threads
// striding the g-points, every weight evaluated inline per g-point.  Same
// statements as the production kernel; tests/test_rrtmgp_optics.py holds
// the two equal bit for bit and pins this text.
extern "C" __global__ void rrtmgp_gas_optics_reference(RRTMGP_GAS_ARGS) {
  const int cell_id = blockIdx.x;
  if (cell_id >= ncol * nlay) return;
  const int col = cell_id / nlay;
  const int lay = cell_id - col * nlay;
  extern __shared__ float s_scaling[];
  RRTMGP_GAS_CELL_PREAMBLE
  for (int mm = threadIdx.x; mm < nminor; mm += blockDim.x) {
    s_scaling[mm] = rrtmgp_minor_scaling(
        v, col_dry, p, t, idx_h2o, idx_minor[mm], density[mm],
        idx_scaling[mm], complement[mm]);
  }
  __syncthreads();
  for (int gpt = threadIdx.x; gpt < ngpt; gpt += blockDim.x) {
    const int iflav = gpoint_flavor[iatm * ngpt + gpt];
    const int gas1 = flavor[iflav * 2];
    const int gas2 = flavor[iflav * 2 + 1];
    const float col1 = gas1 == 0 ? col_dry : v[gas1] * col_dry;
    const float col2 = gas2 == 0 ? col_dry : v[gas2] * col_dry;
    RRTMGP_GAS_GPOINT_BODY(
      const float ratio =
          vmr_ref[(iatm * (ngas + 1) + gas1) * ntemp + jtr]
          / vmr_ref[(iatm * (ngas + 1) + gas2) * ntemp + jtr];
      rrtmgp_eta_weights(col1, col2, ratio, ftemp, itemp, neta,
                         &cmix[itemp], &je[itemp], &fm[0][itemp],
                         &fm[1][itemp]);)
  }
}
// END rrtmgp_gas_optics_reference

// compute_Planck_source: one thread per (column, g-point), walking the
// layers.  pfrac is kept in the thread (the level source reads the two
// adjacent layers) up to the 128-layer ceiling the host enforces.
extern "C" __global__ void rrtmgp_planck_sources(
    const float* play, const float* plev, const float* tlay,
    const float* tlev, const float* tsfc, const float* vmr,
    const int* iatm_meta, const int* jt_meta, const int* jp_meta,
    const float* ftemp_meta, const float* fpress_meta,
    const float* temp_ref,
    const float* vmr_ref, const int* flavor, const int* gpoint_flavor,
    const int* gpoint_bands, const float* planck_fraction,
    const float* totplnk, float* lay_source, float* lev_source,
    float* sfc_source, int ncol, int nlay, int ngas, int ngpt,
    int ntemp, int npres, int neta, int nband, int nplanck, int idx_h2o) {
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;
  // Device-side belt for the pfrac[128] ceiling below; the LOUD refusal
  // lives on the host (_planck_sources raises before any launch).
  if (nlay > 128) return;
  const int col = tid / ngpt;
  const int gpt = tid - col * ngpt;
  if (col >= ncol) return;
  const int nlev = nlay + 1;
  const float temp_ref_min = temp_ref[0];
  // load_int: totplnk_delta = (temp_ref_max - temp_ref_min) / (nPlanckTemp-1)
  const float totplnk_delta = (temp_ref[ntemp - 1] - temp_ref[0])
                              / (float)(nplanck - 1);
  const float delta_r = 1.0f / totplnk_delta;
  const int sfc_lay = play[col * nlay] < play[col * nlay + nlay - 1]
      ? nlay - 1 : 0;
  const int band = gpoint_bands[gpt];
  float pfrac[128];
  for (int lay = 0; lay < nlay; ++lay) {
    pfrac[lay] = rrtmgp_pfrac_at(
        col, lay, gpt, nlay, ngas, ngpt, ntemp, npres, neta, idx_h2o,
        plev, vmr, iatm_meta, jt_meta, jp_meta, ftemp_meta, fpress_meta,
        vmr_ref, flavor, gpoint_flavor, planck_fraction);
    lay_source[(col * nlay + lay) * ngpt + gpt] = pfrac[lay]
        * rrtmgp_planck_function(tlay[col * nlay + lay], band, totplnk,
                                 nplanck, nband, temp_ref_min, delta_r);
  }
  sfc_source[col * ngpt + gpt] = pfrac[sfc_lay]
      * rrtmgp_planck_function(tsfc[col], band, totplnk, nplanck, nband,
                               temp_ref_min, delta_r);
  lev_source[(col * nlev) * ngpt + gpt] = pfrac[0]
      * rrtmgp_planck_function(tlev[col * nlev], band, totplnk, nplanck,
                               nband, temp_ref_min, delta_r);
  for (int lev = 1; lev < nlay; ++lev) {
    lev_source[(col * nlev + lev) * ngpt + gpt] =
        sqrtf(pfrac[lev - 1] * pfrac[lev])
        * rrtmgp_planck_function(tlev[col * nlev + lev], band, totplnk,
                                 nplanck, nband, temp_ref_min, delta_r);
  }
  lev_source[(col * nlev + nlay) * ngpt + gpt] = pfrac[nlay - 1]
      * rrtmgp_planck_function(tlev[col * nlev + nlay], band, totplnk,
                               nplanck, nband, temp_ref_min, delta_r);
}


// One thread per (column, layer) fills that cell's whole gas-VMR row.
//
// The row is otherwise built by five separate CuPy expressions -- a full
// zero fill, a scaled h2o write, an ozone write, and one write per
// well-mixed trace gas -- each a strided store into the last axis of a
// (ncol, nlay, ngas+1) cube, and each its own launch.  The stores are
// cheap; the LAUNCHES are not.  `_gas_vmr` runs once per chunk per band
// (2364 times an hour) and radiation is submission bound (HOWTO 13.5b),
// so collapsing five launches into one is the point of this kernel, not
// the arithmetic.
//
// BIT-EXACTNESS.  Every value written here is the value the CuPy path
// wrote: `h2o_scale` is the caller's float32 cast of 0.028964/0.018016 and
// the multiply is single precision, `o3` is `cupy.interp`'s own output
// passed straight through (deliberately NOT recomputed -- matching its
// last bit is not worth owning), and each trace value is the caller's
// float32 cast.  Slots no gas claims are zero, as the fill left them.
extern "C" __global__ void rrtmgp_gas_vmr_fill(
    const float* __restrict__ qv,        // (ncol, nlay)
    const double* __restrict__ o3,       // (ncol * nlay,) interp output
    const int* __restrict__ trace_index, // (ntrace,) slots, may be empty
    const float* __restrict__ trace_value,
    float* __restrict__ vmr,             // (ncol, nlay, ngas + 1)
    float h2o_scale, int h2o_index, int o3_index,
    int ntrace, int ncells, int nslot) {
  // Adjacent threads own adjacent slots, so each store is coalesced.
  const long long index = (long long)blockDim.x * blockIdx.x + threadIdx.x;
  if (index >= (long long)ncells * nslot) return;
  const int cell = index / nslot;
  const int slot = index % nslot;
  float value = 0.0f;
  if (slot == h2o_index) value = qv[cell] * h2o_scale;
  // Preserve the float64 interpolation result's narrowing on store.
  if (slot == o3_index) value = (float)o3[cell];
  // Preserve assignment precedence, including repeated trace slots.
  for (int t = 0; t < ntrace; ++t)
    if (slot == trace_index[t]) value = trace_value[t];
  vmr[index] = value;
}
