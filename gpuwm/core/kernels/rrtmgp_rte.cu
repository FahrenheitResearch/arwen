// RTE longwave no-scattering and shortwave PIFM two-stream solvers.
// One CUDA thread integrates ONE g-point of one column, over a tile of
// `gpt_tile` g-points at a time; the fold (in-block, or the companion
// reduce kernel) adds the tile into the running broadband flux in strict
// ascending g-point order, so the FP32 accumulation sequence is the one the
// Fortran's g-point loop produces.
//
// Transcribed from RTE+RRTMGP fa107a1
//   rte/kernels/mo_rte_solver_kernels.F90
//     lw_solver_noscat_oneangle 51-240, lw_source_noscat 620-675,
//     lw_transport_noscat_dn/up 681-745, sw_solver_2stream 503-609,
//     sw_dif_and_source 985-1127, adding 1135-1245
//   rte/kernels/mo_optical_props_kernels.F90
//     delta_scale_2str_k 76-98, increment_1scalar_by_2stream,
//     increment_2stream_by_2stream
//   rte/frontend/mo_rte_lw.F90 (gauss_Ds/gauss_wts for one angle)
//
// The unit compiles with --fmad=false and calls WOOF's own float32 expf
// (rrtmgp_libm.cuh): every statement is the upstream single-precision
// build's arithmetic, operator for operator, graded at 0 ULP by
// tools/rrtmgp_rte_upstream_oracle.  The broadband sums follow the Fortran
// exactly: LW sums RADIANCES over g-points and multiplies the sum by
// pi * weight once at the end; SW adds flux_up, then (flux_dn + flux_dir)
// as two separate additions, then flux_dir.
//
// The tile exists because these solvers are local-memory heavy (a few KiB
// of per-thread column arrays).  Beyond roughly L2/(ncol*frame) g-points in
// flight the live local working set stops fitting L2 and the kernel slows
// down again, so more parallelism is NOT monotonically better here.

// Layer bound for the per-thread column arrays.  The driver compiles this
// module with the run's actual layer count (gpuwm.core.kernels
// load_module_int_defines), because the frame is what bounds the g-point
// tile: a 128-layer frame on a 74-layer column wastes 40% of the L2 budget
// the tile is sized against, and so costs parallelism, not just memory.
#ifndef RRTMGP_MAX_LAYERS
#define RRTMGP_MAX_LAYERS 128
#endif
#define RRTMGP_REAL float
#define RRTMGP_REAL_EPSILON 0x1p-23f
#define RRTMGP_REAL_SQRT sqrtf

// delta_scale_2str_k with f = g**2 (mo_optical_props_kernels.F90:76-98).
extern "C" __global__ void rrtmgp_delta_scale(
    const float* tau_in, const float* ssa_in, const float* g_in,
    float* tau, float* ssa, float* asym, int n) {
  const int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= n) return;
  const float eps = 3.0f * 1.17549435e-38f;
  const float gg = g_in[idx];
  const float f = gg * gg;
  const float wf = ssa_in[idx] * f;
  tau[idx] = (1.0f - wf) * tau_in[idx];
  ssa[idx] = (ssa_in[idx] - wf) / fmaxf(eps, 1.0f - wf);
  asym[idx] = (gg - f) / fmaxf(eps, 1.0f - f);
}

// In-block fold.  Each thread of a column group buffers the per-g-point
// value of one flux row; when RTE_FOLD_ROWS rows (or the kernel's own
// smaller flush count) are buffered, thread r of the group adds row r's
// tile into its target in ascending g-point order.  A row whose target
// carries tag bit 0 ("paired") also adds the NEXT row's value at each
// g-point, after its own: that is the SW broadband_dn update
//   broadband_dn = broadband_dn + gpt_flux_dn + gpt_flux_dir
// whose two additions round separately.  On the last g-point tile the
// folded sum is multiplied by `final_scale` (LW: pi * weight).
#define RTE_FOLD_ROWS 16
__device__ __forceinline__ void rte_fold_sync(int gpt_tile) {
  if (gpt_tile == 16) {
    const unsigned mask = 0xffffu << (threadIdx.x & 16);
    __syncwarp(mask);
  } else if (gpt_tile == 32) __syncwarp();
  else __syncthreads();
}
__device__ __forceinline__ void rte_fold_flush(
    int gl, int gpt_tile, int gpt0, float* s_v, int& count,
    float final_scale, int last_tile) {
  const int stride = gpt_tile + 1;
  unsigned long long* targets = reinterpret_cast<unsigned long long*>(
      s_v + RTE_FOLD_ROWS * stride);
  rte_fold_sync(gpt_tile);
  if (gl < count) {
    const unsigned long long tag = targets[gl];
    float* target = reinterpret_cast<float*>(tag & ~1ULL);
    const bool paired = (tag & 1ULL) != 0ULL;
    float su = gpt0 == 0 ? 0.0f : *target;
    for (int g = 0; g < gpt_tile; ++g) {
      su = su + s_v[gl * stride + g];
      if (paired) su = su + s_v[(gl + 1) * stride + g];
    }
    if (last_tile) su = final_scale * su;
    *target = su;
  }
  rte_fold_sync(gpt_tile);
  count = 0;
}
__device__ __forceinline__ void rte_fold_emit(
    float* flux, int col, int nlev, int lev, float v, int gl, int gpt_tile,
    int gpt0, float* s_v, int& count, int flush_rows, float final_scale,
    int last_tile, bool last = false, bool paired = false) {
  const int stride = gpt_tile + 1;
  unsigned long long* targets = reinterpret_cast<unsigned long long*>(
      s_v + RTE_FOLD_ROWS * stride);
  s_v[count * stride + gl] = v;
  if (gl == 0) targets[count] =
      reinterpret_cast<unsigned long long>(flux + col * nlev + lev)
      | (paired ? 1ULL : 0ULL);
  ++count;
  if (count == flush_rows || last)
    rte_fold_flush(gl, gpt_tile, gpt0, s_v, count, final_scale, last_tile);
}

// lw_solver_noscat_oneangle with one Gaussian angle (mo_rte_lw.F90:
// gauss_Ds = 1/0.6096748751, gauss_wts = 1), broadband output.  The two
// column arrays left are trans (both recurrences read it) and src_up
// (produced by the downward pass, read by the upward one); src_dn and the
// fluxes are consumed as scalars in the pass that produces them.
extern "C" __global__ void rrtmgp_lw_noscat(
    const float* tau, const float* lay_source, const float* lev_source,
    const float* sfc_source, const float* sfc_emis,
    const float* incident_flux, float* part_up, float* part_dn,
    int ncol, int nlay, int ngpt, int gpt0, int gpt_tile, int top_at_1,
    int warp_fold,
    const float* fz_cld_tau, const float* fz_cld_ssa,
    const float* fz_cld_asy, const int* fz_gpt_bands,
    const unsigned char* fz_mask, int fz_nband,
    int fz_have_mask, int fz_on,
    const float* pk_play, const float* pk_plev, const float* pk_tlay,
    const float* pk_tlev, const float* pk_tsfc, const float* pk_vmr,
    const int* pk_iatm, const int* pk_jt, const int* pk_jp,
    const float* pk_ftemp, const float* pk_fpress,
    const float* pk_temp_ref, const float* pk_vmr_ref,
    const int* pk_flavor, const int* pk_gpoint_flavor,
    const int* pk_gpoint_bands, const float* pk_planck_fraction,
    const float* pk_totplnk, int pk_ngas, int pk_ntemp, int pk_npres,
    int pk_neta, int pk_nband, int pk_nplanck, int pk_idx_h2o, int pk_on) {
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (nlay > RRTMGP_MAX_LAYERS) return;
  const int col = tid / gpt_tile;
  const int gl = tid - col * gpt_tile;
  if (col >= ncol) return;
  const int gpt = gpt0 + gl;
  if (gpt >= ngpt) return;
  const int nlev = nlay + 1;
  const int top = top_at_1 ? 0 : nlay;
  const int sfc = top_at_1 ? nlay : 0;
  const float d = 1.0f / 0.6096748751f;
  const float weight = 1.0f;
  const float pi = 3.14159265358979323846f;
  const float pi_weight = pi * weight;
  const int last_tile = gpt0 + gpt_tile >= ngpt;
  const RRTMGP_REAL tau_thresh =
      RRTMGP_REAL_SQRT(RRTMGP_REAL_SQRT(RRTMGP_REAL_EPSILON));
  float trans[RRTMGP_MAX_LAYERS];
  float src_up[RRTMGP_MAX_LAYERS];
  const int base = gl * nlev;
  // Each packed column owns a separate buffer on the folding path.
  extern __shared__ float s_fold[];
  int fold_count = 0;
  float* fold_group = s_fold + (threadIdx.x / gpt_tile)
      * (RTE_FOLD_ROWS * (gpt_tile + 1) + RTE_FOLD_ROWS * 2);

  // PLANCK, DERIVED HERE: compute_Planck_source evaluated layer by layer
  // through the same device words as the standalone rrtmgp_planck_sources
  // kernel (rrtmgp_planck_common.cuh), so the values are its values.  pfrac
  // does not chain: lev_source[lev] reads the two adjacent layers only, so
  // a three-value window walked in this pass's direction suffices.
  const int pk_band = pk_on ? pk_gpoint_bands[gpt] : 0;
  const float pk_t0 = pk_on ? pk_temp_ref[0] : 0.0f;
  const float pk_delta_r = pk_on
      ? 1.0f / ((pk_temp_ref[pk_ntemp - 1] - pk_temp_ref[0])
                / (float)(pk_nplanck - 1))
      : 1.0f;
#define PK_AT(L) rrtmgp_pfrac_at(col, (L), gpt, nlay, pk_ngas, ngpt, \
    pk_ntemp, pk_npres, pk_neta, pk_idx_h2o, pk_plev, pk_vmr, pk_iatm, \
    pk_jt, pk_jp, pk_ftemp, pk_fpress, pk_vmr_ref, pk_flavor, \
    pk_gpoint_flavor, pk_planck_fraction)
#define PK_B(T) rrtmgp_planck_function((T), pk_band, pk_totplnk, \
    pk_nplanck, pk_nband, pk_t0, pk_delta_r)
  const int pk_first = top_at_1 ? 0 : nlay - 1;
  float pk_prev = (pk_on && pk_first > 0) ? PK_AT(pk_first - 1) : 0.0f;
  float pk_cur = pk_on ? PK_AT(pk_first) : 0.0f;
  float pk_next = (pk_on && pk_first + 1 < nlay)
      ? PK_AT(pk_first + 1) : 0.0f;

  // Incident flux converted to intensity: incident_flux / (pi * weight),
  // an IEEE quotient (a constant divisor is spelled __fdiv_rn, A146).
  float dnv = __fdiv_rn(incident_flux[col * ngpt + gpt], pi_weight);
  if (warp_fold) {
    rte_fold_emit(part_dn, col, nlev, top, dnv, gl, gpt_tile, gpt0,
                  fold_group, fold_count, RTE_FOLD_ROWS, pi_weight,
                  last_tile);
  } else {
    part_dn[(base + top) * ncol + col] = dnv;
  }
  for (int j = 0; j < nlay; ++j) {
    const int lay = top_at_1 ? j : nlay - 1 - j;
    const int cell_lay = col * nlay + lay;
    const int cellg = cell_lay * ngpt + gpt;
    float tau_v = tau[cellg];
    if (fz_on) {
      // increment_1scalar_by_2stream of the McICA-masked band clouds:
      // tau = tau + tau_cld * (1 - ssa_cld); on this path `tau` IS the
      // gas optical depth.
      const int fz_band = (col * nlay + lay) * fz_nband + fz_gpt_bands[gpt];
      float fz_tc = fz_cld_tau[fz_band];
      if (fz_have_mask) fz_tc = fz_tc * (float)fz_mask[cellg];
      tau_v = tau_v + fz_tc * (1.0f - fz_cld_ssa[fz_band]);
    }
    const float tau_loc = tau_v * d;
    const float tr = rrtmgp_expf(-tau_loc);
    trans[lay] = tr;
    const float fact = tau_loc > tau_thresh
        ? (1.0f - tr) / tau_loc - tr
        : tau_loc * (0.5f + tau_loc * (-1.0f / 3.0f
                                       + tau_loc * 1.0f / 8.0f));
    float lays, lo, hi;
    if (pk_on) {
      lays = pk_cur * PK_B(pk_tlay[cell_lay]);
      const float lo_pf = lay == 0 ? pk_cur : sqrtf(pk_prev * pk_cur);
      const float hi_pf = lay + 1 == nlay ? pk_cur : sqrtf(pk_cur * pk_next);
      lo = lo_pf * PK_B(pk_tlev[col * nlev + lay]);
      hi = hi_pf * PK_B(pk_tlev[col * nlev + lay + 1]);
    } else {
      lays = lay_source[cellg];
      lo = lev_source[(col * nlev + lay) * ngpt + gpt];
      hi = lev_source[(col * nlev + lay + 1) * ngpt + gpt];
    }
    // lw_source_noscat: source_inc uses the level above the layer index
    // (ilay+1), source_dec the one at it (ilay).
    const float inc = (1.0f - tr) * hi + 2.0f * fact * (lays - hi);
    const float dec = (1.0f - tr) * lo + 2.0f * fact * (lays - lo);
    src_up[lay] = top_at_1 ? dec : inc;
    if (pk_on) {
      // Advance the pfrac window one layer in this pass's direction.
      if (top_at_1) {
        pk_prev = pk_cur; pk_cur = pk_next;
        pk_next = lay + 2 < nlay ? PK_AT(lay + 2) : 0.0f;
      } else {
        pk_next = pk_cur; pk_cur = pk_prev;
        pk_prev = lay - 2 >= 0 ? PK_AT(lay - 2) : 0.0f;
      }
    }
    dnv = tr * dnv + (top_at_1 ? inc : dec);
    const int dlev = top_at_1 ? lay + 1 : lay;
    if (warp_fold) {
      rte_fold_emit(part_dn, col, nlev, dlev, dnv, gl, gpt_tile, gpt0,
                    fold_group, fold_count, RTE_FOLD_ROWS, pi_weight,
                    last_tile, dlev == sfc);
    } else {
      part_dn[(base + dlev) * ncol + col] = dnv;
    }
  }

  const int sg = col * ngpt + gpt;
  float sfc_src;
  if (pk_on) {
    // pfrac at the highest-pressure layer, which the window has walked
    // away from: one extra evaluation, once per thread.
    const int pk_sfc_lay =
        pk_play[col * nlay] < pk_play[col * nlay + nlay - 1] ? nlay - 1 : 0;
    sfc_src = PK_AT(pk_sfc_lay) * PK_B(pk_tsfc[col]);
  } else {
    sfc_src = sfc_source[sg];
  }
  const float sfc_albedo = 1.0f - sfc_emis[sg];
  float upv = dnv * sfc_albedo + sfc_emis[sg] * sfc_src;
  if (warp_fold) {
    rte_fold_emit(part_up, col, nlev, sfc, upv, gl, gpt_tile, gpt0,
                  fold_group, fold_count, RTE_FOLD_ROWS, pi_weight,
                  last_tile);
  } else {
    part_up[(base + sfc) * ncol + col] = upv;
  }
  for (int j = 0; j < nlay; ++j) {
    const int lay = top_at_1 ? nlay - 1 - j : j;
    upv = trans[lay] * upv + src_up[lay];
    const int ulev = top_at_1 ? lay : lay + 1;
    if (warp_fold) {
      rte_fold_emit(part_up, col, nlev, ulev, upv, gl, gpt_tile, gpt0,
                    fold_group, fold_count, RTE_FOLD_ROWS, pi_weight,
                    last_tile, ulev == top);
    } else {
      part_up[(base + ulev) * ncol + col] = upv;
    }
  }
}
#undef PK_AT
#undef PK_B

// Fold one g-point tile of intensities into the running LW sums, ascending
// g-point order; the last tile converts the sums to flux (times pi*weight).
extern "C" __global__ void rrtmgp_lw_flux_reduce(
    const float* part_up, const float* part_dn,
    float* flux_up, float* flux_dn,
    int ncol, int nlev, int gpt0, int ntile, int last_tile) {
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= ncol * nlev) return;
  const int lev = tid / ncol;
  const int col = tid - lev * ncol;
  const float pi_weight = 3.14159265358979323846f * 1.0f;
  const int o = col * nlev + lev;
  float su = gpt0 == 0 ? 0.0f : flux_up[o];
  float sd = gpt0 == 0 ? 0.0f : flux_dn[o];
  for (int g = 0; g < ntile; ++g) {
    const int q = (g * nlev + lev) * ncol + col;
    su = su + part_up[q];
    sd = sd + part_dn[q];
  }
  if (last_tile) {
    su = pi_weight * su;
    sd = pi_weight * sd;
  }
  flux_up[o] = su;
  flux_dn[o] = sd;
}

// sw_solver_2stream: sw_dif_and_source then adding, broadband output.
// Three column arrays: the optics quartet (Rdif, Tdif, source_dn,
// source_up), the direct beam and the adding pair (albedo, source).  The
// diffuse fluxes are produced as scalars; denom is recomputed in the
// downward sweep from the same two operands (same operation, same bits).
extern "C" __global__ void rrtmgp_sw_2stream(
    const float* tau, const float* ssa, const float* asym,
    const float* mu0, const float* sfc_alb_dir, const float* sfc_alb_dif,
    const float* inc_flux, float* part_up, float* part_dn, float* part_dir,
    int ncol, int nlay, int ngpt, int gpt0, int gpt_tile, int top_at_1,
    int zero_g, int warp_fold,
    const float* fz_cld_tau, const float* fz_cld_ssa,
    const float* fz_cld_asy, const int* fz_gpt_bands,
    const unsigned char* fz_mask, int fz_nband,
    int fz_have_mask, int fz_on) {
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (nlay > RRTMGP_MAX_LAYERS) return;
  const int col = tid / gpt_tile;
  const int gl = tid - col * gpt_tile;
  if (col >= ncol) return;
  const int gpt = gpt0 + gl;
  if (gpt >= ngpt) return;
  const int nlev = nlay + 1;
  const int top = top_at_1 ? 0 : nlay;
  const int top_lay = top_at_1 ? 0 : nlay - 1;
  const int sfc = top_at_1 ? nlay : 0;
  const int sfc_lay = top_at_1 ? nlay - 1 : 0;
  const int last_tile = gpt0 + gpt_tile >= ngpt;
  const float eps = RRTMGP_REAL_EPSILON;
  const float min_k = 1.0e4f * eps;
  const float min_mu0 = sqrtf(eps);
  // Up, dn and dir rows per level: flush every five levels, so a level's
  // dn row and its paired dir row are always folded together.
  const int flush_rows = 15;
  float4 optics[RRTMGP_MAX_LAYERS];
  float direct[RRTMGP_MAX_LAYERS + 1];
  float2 adding[RRTMGP_MAX_LAYERS + 1];  // x: albedo, y: source

  const int spec = col * ngpt + gpt;
  const int base = gl * nlev;
  extern __shared__ float s_fold[];
  int fold_count = 0;
  float* fold_group = s_fold + (threadIdx.x / gpt_tile)
      * (RTE_FOLD_ROWS * (gpt_tile + 1) + RTE_FOLD_ROWS * 2);
  direct[top] = inc_flux[spec] * mu0[col * nlay + top_lay];
  for (int j = 0; j < nlay; ++j) {
    const int lay = top_at_1 ? j : nlay - j - 1;
    const int ilev_in = top_at_1 ? lay : lay + 1;
    const int ilev_out = top_at_1 ? lay + 1 : lay;
    const int idx = (col * nlay + lay) * ngpt + gpt;
    float ts, ws, gs;
    if (fz_on) {
      // increment_2stream_by_2stream of the McICA-masked band clouds into
      // the gas optics (gas g = 0), then delta_scale_2str_k.  `tau`/`ssa`
      // here ARE gas_tau/gas_ssa and `asym` is never read.
      const int fz_band = (col * nlay + lay) * fz_nband + fz_gpt_bands[gpt];
      float fz_tc = fz_cld_tau[fz_band];
      if (fz_have_mask) fz_tc = fz_tc * (float)fz_mask[idx];
      const float fz_wc = fz_cld_ssa[fz_band];
      const float fz_gc = fz_cld_asy[fz_band];
      const float fz_eps = 3.0f * 1.17549435e-38f;
      const float tau1 = tau[idx], ssa1 = ssa[idx];
      const float tau12 = tau1 + fz_tc;
      const float tauscat12 = tau1 * ssa1 + fz_tc * fz_wc;
      const float g12 = (tau1 * ssa1 * 0.0f + fz_tc * fz_wc * fz_gc)
                        / fmaxf(fz_eps, tauscat12);
      const float ssa12 = tauscat12 / fmaxf(fz_eps, tau12);
      const float f = g12 * g12;
      const float wf = ssa12 * f;
      ts = (1.0f - wf) * tau12;
      ws = (ssa12 - wf) / fmaxf(fz_eps, 1.0f - wf);
      gs = (g12 - f) / fmaxf(fz_eps, 1.0f - f);
    } else {
      ts = tau[idx];
      ws = ssa[idx];
      gs = zero_g ? 0.0f : asym[idx];
    }
    // Zdunkowski PIFM.
    const float gamma1 = (8.0f - ws * (5.0f + 3.0f * gs)) * 0.25f;
    const float gamma2 = 3.0f * (ws * (1.0f - gs)) * 0.25f;
    const float kval = sqrtf(fmaxf((gamma1 - gamma2) * (gamma1 + gamma2),
                                   min_k));
    const float ex = rrtmgp_expf(-ts * kval);
    const float ex2 = ex * ex;
    float rt = 1.0f / (kval * (1.0f + ex2) + gamma1 * (1.0f - ex2));
    const float rdif_value = rt * gamma2 * (1.0f - ex2);
    const float tdif_value = rt * 2.0f * kval * ex;
    const float mu = fmaxf(min_mu0, mu0[col * nlay + lay]);
    const float kmu = kval * mu;
    const float dterm = fabsf(1.0f - kmu * kmu) >= eps
        ? 1.0f - kmu * kmu : eps;
    rt = ws * rt / dterm;
    const float gamma3 = (2.0f - 3.0f * mu * gs) * 0.25f;
    const float gamma4 = 1.0f - gamma3;
    const float alpha1 = gamma1 * gamma4 + gamma2 * gamma3;
    const float alpha2 = gamma1 * gamma3 + gamma2 * gamma4;
    const float kg3 = kval * gamma3, kg4 = kval * gamma4;
    const float tnoscat = rrtmgp_expf(-ts / mu);
    float rdir = rt * (
        (1.0f - kmu) * (alpha2 + kg3)
      - (1.0f + kmu) * (alpha2 - kg3) * ex2
      - 2.0f * (kg3 - alpha2 * kmu) * ex * tnoscat);
    float tdir = -rt * (
        (1.0f + kmu) * (alpha1 + kg4) * tnoscat
      - (1.0f - kmu) * (alpha1 - kg4) * ex2 * tnoscat
      - 2.0f * (kg4 + alpha1 * kmu) * ex);
    rdir = fmaxf(0.0f, fminf(rdir, 1.0f - tnoscat));
    tdir = fmaxf(0.0f, fminf(tdir, 1.0f - tnoscat - rdir));
    float src_up_value = rdir * direct[ilev_in];
    float src_dn_value = tdir * direct[ilev_in];
    direct[ilev_out] = tnoscat * direct[ilev_in];
    if (mu0[col * nlay + lay] <= 0.0f)
      src_up_value = src_dn_value = 0.0f;
    optics[lay] = make_float4(rdif_value, tdif_value,
                              src_dn_value, src_up_value);
  }
  const float source_sfc = mu0[col * nlay + sfc_lay] > 0.0f
      ? direct[sfc] * sfc_alb_dir[spec] : 0.0f;

#define SW_EMIT(LEV, LAST)                                                  \
  if (warp_fold) {                                                          \
    rte_fold_emit(part_up, col, nlev, (LEV), upv, gl, gpt_tile, gpt0,       \
                  fold_group, fold_count, flush_rows, 1.0f, last_tile);     \
    rte_fold_emit(part_dn, col, nlev, (LEV), dnv, gl, gpt_tile, gpt0,       \
                  fold_group, fold_count, flush_rows, 1.0f, last_tile,      \
                  false, true);                                             \
    rte_fold_emit(part_dir, col, nlev, (LEV), direct[(LEV)], gl, gpt_tile,  \
                  gpt0, fold_group, fold_count, flush_rows, 1.0f,           \
                  last_tile, (LAST));                                       \
  } else {                                                                  \
    part_up[(base + (LEV)) * ncol + col] = upv;                             \
    part_dn[(base + (LEV)) * ncol + col] = dnv;                             \
    part_dir[(base + (LEV)) * ncol + col] = direct[(LEV)];                  \
  }

  float dnv, upv;
  if (top_at_1) {
    adding[nlay].x = sfc_alb_dif[spec];
    adding[nlay].y = source_sfc;
    for (int lev = nlay - 1; lev >= 0; --lev) {
      const float4 coeff = optics[lev];
      const float dn_ = 1.0f / (1.0f - coeff.x * adding[lev + 1].x);
      adding[lev].x = coeff.x + coeff.y * coeff.y
          * adding[lev + 1].x * dn_;
      adding[lev].y = coeff.w + coeff.y * dn_
          * (adding[lev + 1].y + adding[lev + 1].x * coeff.z);
    }
    dnv = 0.0f;
    upv = dnv * adding[0].x + adding[0].y;
    SW_EMIT(0, false)
    for (int lev = 1; lev < nlev; ++lev) {
      const int lay = lev - 1;
      const float4 coeff = optics[lay];
      const float dn_ = 1.0f / (1.0f - coeff.x * adding[lev].x);
      dnv = (coeff.y * dnv + coeff.x * adding[lev].y + coeff.z) * dn_;
      upv = dnv * adding[lev].x + adding[lev].y;
      SW_EMIT(lev, lev == sfc)
    }
  } else {
    adding[0].x = sfc_alb_dif[spec];
    adding[0].y = source_sfc;
    for (int lev = 0; lev < nlay; ++lev) {
      const float4 coeff = optics[lev];
      const float dn_ = 1.0f / (1.0f - coeff.x * adding[lev].x);
      adding[lev + 1].x = coeff.x + coeff.y * coeff.y * adding[lev].x * dn_;
      adding[lev + 1].y = coeff.w + coeff.y * dn_
          * (adding[lev].y + adding[lev].x * coeff.z);
    }
    dnv = 0.0f;
    upv = dnv * adding[nlay].x + adding[nlay].y;
    SW_EMIT(nlay, false)
    for (int lev = nlay - 1; lev >= 0; --lev) {
      const float4 coeff = optics[lev];
      const float dn_ = 1.0f / (1.0f - coeff.x * adding[lev].x);
      dnv = (coeff.y * dnv + coeff.x * adding[lev].y + coeff.z) * dn_;
      upv = dnv * adding[lev].x + adding[lev].y;
      SW_EMIT(lev, lev == sfc)
    }
  }
#undef SW_EMIT
}

// Fold one g-point tile into the running SW fluxes, ascending g-point order:
//   up  = up + flux_up;  dn = (dn + flux_dn) + flux_dir;  dir = dir + flux_dir
extern "C" __global__ void rrtmgp_sw_flux_reduce(
    const float* part_up, const float* part_dn, const float* part_dir,
    float* flux_up, float* flux_dn, float* flux_dir,
    int ncol, int nlev, int gpt0, int ntile) {
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= ncol * nlev) return;
  const int lev = tid / ncol;
  const int col = tid - lev * ncol;
  const int o = col * nlev + lev;
  float su = gpt0 == 0 ? 0.0f : flux_up[o];
  float sd = gpt0 == 0 ? 0.0f : flux_dn[o];
  float sr = gpt0 == 0 ? 0.0f : flux_dir[o];
  for (int g = 0; g < ntile; ++g) {
    const int q = (g * nlev + lev) * ncol + col;
    su = su + part_up[q];
    sd = sd + part_dn[q] + part_dir[q];
    sr = sr + part_dir[q];
  }
  flux_up[o] = su;
  flux_dn[o] = sd;
  flux_dir[o] = sr;
}
