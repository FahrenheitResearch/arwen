// RTE longwave no-scattering and shortwave PIFM two-stream solvers.
// One CUDA thread integrates ONE g-point of one column, over a tile of
// `gpt_tile` g-points at a time; the companion reduce kernel folds the tile
// into the running flux in strict ascending g-point order, so the FP32
// accumulation sequence is exactly the one a single-thread-per-column loop
// produced.  The accumulate expression is written the same way in both
// kernels (`+= pi * x`) so that FMA contraction, if the compiler applies it,
// applies identically on both sides.
//
// The tile exists because these solvers are local-memory heavy (a few KiB of
// per-thread column arrays).  Beyond roughly L2/(ncol*frame) g-points in
// flight the live local working set stops fitting L2 and the kernel slows
// down again, so more parallelism is NOT monotonically better here.
// Transcribed from RTE+RRTMGP fa107a1
// rte/kernels/mo_rte_solver_kernels.F90:51-240,503-745,985-1245 and
// rte/kernels/mo_optical_props_kernels.F90:76-98.

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

extern "C" __global__ void rrtmgp_delta_scale(
    const float* tau_in, const float* ssa_in, const float* g_in,
    float* tau, float* ssa, float* asym, int n) {
  const int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= n) return;
  const float gg = g_in[idx];
  const float f = gg * gg;
  const float wf = ssa_in[idx] * f;
  tau[idx] = (1.0f - wf) * tau_in[idx];
  ssa[idx] = (ssa_in[idx] - wf) / fmaxf(3.0f * 1.17549435e-38f,
                                         1.0f - wf);
  asym[idx] = (gg - f) / fmaxf(3.0f * 1.17549435e-38f, 1.0f - f);
}

// Slim LW no-scattering solver: same arithmetic, three fewer column arrays.
//
// The original buffers trans, src_dn, src_up, up and dn -- five arrays of
// RRTMGP_MAX_LAYERS -- which is 1192 B of LOCAL memory per thread and about
// two thirds of this kernel's total bandwidth.  Three of them exist only
// because the passes run in different directions:
//
//   * src_dn is produced ascending and consumed by a descending recurrence,
//     so walking the source pass in the RECURRENCE's direction lets it be
//     consumed as a scalar;
//   * dn and up are buffered only to be stored in a final loop, so storing
//     each level as the recurrence produces it removes both.
//
// What is left is trans (both recurrences read it) and src_up (produced by
// the first pass, read by the second, opposite directions).  592 B.
// Fold one tile's contribution to one (column, level) WITHOUT a round trip.
//
// The launcher gives the folding path ONE BLOCK PER COLUMN with
// `blockDim.x == gpt_tile`, so this block IS the fold group: `gl` is
// `threadIdx.x`, the group base is 0, and every thread of the block
// reaches every barrier below (the only two early returns are on `nlay`
// and `col`, both block-uniform).  Lane 0 then adds the tile left to
// right, which is the statement `rrtmgp_*_flux_reduce` makes -- same
// values, same order, bit-identical.
//
// What it removes is the round trip those kernels exist to make, and the
// cost is the SCATTER rather than the bytes: a warp storing consecutive
// `gl` touched 32 separate sectors to write 128 bytes, because
// consecutive `gl` sat `nlev*ncol` floats apart.
//
// Shared memory rather than `__shfl_sync` because the fold group is a
// BLOCK, not a warp: shuffles cap the tile at 32, and the tile the solver
// actually wants once it stops writing partials is 64 (measured, and it
// scales with SM count -- see `_rte_gpt_tile`).
__device__ __forceinline__ void rte_fold_emit(
    float* flux, int col, int nlev, int lev, float v, float scale,
    int gl, int gpt_tile, int gpt0, float* s_v) {
  __syncthreads();          // the previous fold has finished reading s_v
  s_v[gl] = v;
  __syncthreads();
  if (gl == 0) {
    const int o = col * nlev + lev;
    float su = gpt0 == 0 ? 0.0f : flux[o];
    for (int g = 0; g < gpt_tile; ++g) {
      // PINNED (13.6j): ptxas contracts this into an FFMA in the LW
      // compilation and, with scale == 1.0f, folds the multiply away in
      // the SW one -- and __fmaf_rn(1.0f, x, su) is bitwise __fadd_rn.
      // One stated form covers both and stops depending on pressure.
      su = __fmaf_rn(scale, s_v[g], su);
    }
    flux[o] = su;
  }
}

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
    const float* pk_play, const float* pk_tlay, const float* pk_tlev,
    const float* pk_tsfc, const float* pk_vmr,
    const int* pk_iatm, const int* pk_jt, const int* pk_jp,
    const float* pk_ftemp, const float* pk_fpress,
    const float* pk_temp_ref, const float* pk_vmr_ref,
    const int* pk_flavor, const int* pk_gpoint_flavor,
    const int* pk_gpoint_bands, const float* pk_planck_fraction,
    const float* pk_totplnk, int pk_ngas, int pk_ntemp, int pk_npres,
    int pk_neta, int pk_nband, int pk_nplanck, int pk_on) {
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
  const float pi = 3.14159265358979323846f;
  const RRTMGP_REAL tau_thresh =
      RRTMGP_REAL_SQRT(RRTMGP_REAL_SQRT(RRTMGP_REAL_EPSILON));
  float trans[RRTMGP_MAX_LAYERS];
  float src_up[RRTMGP_MAX_LAYERS];
  const int base = gl * nlev;
  // Only read on the folding path, where the block IS the fold group.
  extern __shared__ float s_fold[];

  // PLANCK, DERIVED HERE (all FP ops pinned -- see the header and
  // 13.6j).  `rrtmgp_planck_sources` writes lay_source and lev_source
  // for this solver to read straight back: 455 MiB of the binding
  // workspace phase at the default chunk, and a round trip an ablation
  // priced at 38% of this kernel.  Deriving them costs one
  // `rrtmgp_pk_fraction_at` per layer, because `pfrac` DOES NOT CHAIN:
  // lev_source[lev] is sqrt(pfrac[lev-1] * pfrac[lev]), adjacent layers
  // only, so three registers and a one-layer lookahead compute each
  // exactly once, with no array.
  //
  // Rejected once (f6d776d3) on a premise the block fold expired ("the
  // solver runs at ~64 blocks BY DESIGN" -- it is one block per column
  // now), and failed once more when register pressure flipped an
  // UNPINNED contraction elsewhere in this kernel (13.6j).  Both causes
  // are now dead: the premise by 8aa0b793, the contraction by the pins.
  const int pk_band = pk_on ? pk_gpoint_bands[gpt] : 0;
  const float pk_dpl = pk_on
      ? __fdiv_rn(__fsub_rn(pk_temp_ref[pk_ntemp - 1], pk_temp_ref[0]),
                  (float)(pk_nplanck - 1))
      : 1.0f;
  const float pk_t0 = pk_on ? pk_temp_ref[0] : 0.0f;
#define PK_AT(L) rrtmgp_pk_fraction_at(col, (L), gpt, nlay, pk_ngas, \
    ngpt, pk_ntemp, pk_npres, pk_neta, pk_vmr, pk_iatm, pk_jt, pk_jp, \
    pk_ftemp, pk_fpress, pk_vmr_ref, pk_flavor, pk_gpoint_flavor, \
    pk_planck_fraction)
  const int pk_first = top_at_1 ? 0 : nlay - 1;
  float pk_prev = (pk_on && pk_first > 0) ? PK_AT(pk_first - 1) : 0.0f;
  float pk_cur = pk_on ? PK_AT(pk_first) : 0.0f;
  float pk_next = (pk_on && pk_first + 1 < nlay)
      ? PK_AT(pk_first + 1) : 0.0f;

  float dnv = incident_flux[col * ngpt + gpt] / pi;
  if (warp_fold) {
    rte_fold_emit(part_dn, col, nlev, top, dnv, pi, gl, gpt_tile,
                  gpt0, s_fold);
  } else {
    part_dn[(base + top) * ncol + col] = dnv;
  }
  for (int j = 0; j < nlay; ++j) {
    const int lay = top_at_1 ? j : nlay - 1 - j;
    const int cell_lay = col * nlay + lay;
    const int cellg = cell_lay * ngpt + gpt;
    float tau_v = tau[cellg];
    if (fz_on) {
      // rrtmgp_finalize_cloud_lw, replicated with the same per-op
      // rounding: on this path `tau` IS gas_tau, and the value below is
      // bit-identical to what that kernel would have stored and this one
      // would have loaded back.  What stops existing is the round trip.
      const int fz_band = (col * nlay + lay) * fz_nband + fz_gpt_bands[gpt];
      float fz_tc = fz_cld_tau[fz_band];
      if (fz_have_mask) fz_tc = __fmul_rn(fz_tc, (float)fz_mask[cellg]);
      const float fz_oms = __fsub_rn(1.0f, fz_cld_ssa[fz_band]);
      tau_v = __fadd_rn(tau_v, __fmul_rn(fz_tc, fz_oms));
    }
    const float path = __fmul_rn(tau_v, d);
    const float tr = expf(-path);
    trans[lay] = tr;
    // PINNED (13.6j).  Branch A is sub/div/sub -- nothing contractable.
    // Branch B: HEAD's SASS fuses the Taylor tail into two FFMAs and
    // lowers path/8 to a multiply by 0.125f.
    const float fact = path > tau_thresh
        ? __fsub_rn(__fdiv_rn(__fsub_rn(1.0f, tr), path), tr)
        : __fmul_rn(path, __fmaf_rn(
              path, __fmaf_rn(path, 0.125f, -1.0f / 3.0f), 0.5f));
    float lays, lo, hi;
    if (pk_on) {
      // The standalone kernel's expressions, operand for operand: the
      // pf*interp products are plain FMULs and the interior lev_source
      // takes sqrtf of the adjacent-pfrac FMUL.
      lays = __fmul_rn(pk_cur, rrtmgp_pk_interp(
          pk_tlay[cell_lay], pk_band, pk_totplnk, pk_nplanck, pk_nband,
          pk_t0, pk_dpl));
      const float lo_pf = lay == 0
          ? pk_cur : sqrtf(__fmul_rn(pk_prev, pk_cur));
      const float hi_pf = lay + 1 == nlay
          ? pk_cur : sqrtf(__fmul_rn(pk_cur, pk_next));
      lo = __fmul_rn(lo_pf, rrtmgp_pk_interp(
          pk_tlev[col * nlev + lay], pk_band, pk_totplnk, pk_nplanck,
          pk_nband, pk_t0, pk_dpl));
      hi = __fmul_rn(hi_pf, rrtmgp_pk_interp(
          pk_tlev[col * nlev + lay + 1], pk_band, pk_totplnk, pk_nplanck,
          pk_nband, pk_t0, pk_dpl));
    } else {
      lays = lay_source[cellg];
      lo = lev_source[(col * nlev + lay) * ngpt + gpt];
      hi = lev_source[(col * nlev + lay + 1) * ngpt + gpt];
    }
    // PINNED (13.6j).  HEAD's SASS, operand for operand: FADD for
    // (1 - tr) and for fact + fact (its lowering of 2*fact -- bitwise
    // equal) and for (lays - x); then FMUL for the doubled-fact product
    // and an FFMA folding the (1-tr) product into the add.
    const float omtr = __fsub_rn(1.0f, tr);
    const float fact2 = __fadd_rn(fact, fact);
    const float inc = __fmaf_rn(hi, omtr,
                                __fmul_rn(fact2, __fsub_rn(lays, hi)));
    const float dec = __fmaf_rn(lo, omtr,
                                __fmul_rn(fact2, __fsub_rn(lays, lo)));
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
    // PINNED (13.6j): FFMA in HEAD.
    dnv = __fmaf_rn(tr, dnv, top_at_1 ? inc : dec);
    const int dlev = top_at_1 ? lay + 1 : lay;
    if (warp_fold) {
      rte_fold_emit(part_dn, col, nlev, dlev, dnv, pi, gl, gpt_tile,
                    gpt0, s_fold);
    } else {
      part_dn[(base + dlev) * ncol + col] = dnv;
    }
  }

  const int sg = col * ngpt + gpt;
  // PINNED (13.6j): HEAD's SASS keeps emis*sfc_source as an FMUL and
  // fuses the (1-emis)*dnv product into the add.
  float pk_sfc;
  if (pk_on) {
    // The standalone kernel's sfc_source: pfrac at the highest-pressure
    // layer, which the window has walked away from -- one extra
    // evaluation, once per thread, not per layer.
    const int pk_sfc_lay =
        pk_play[col * nlay] < pk_play[col * nlay + nlay - 1] ? nlay - 1 : 0;
    pk_sfc = __fmul_rn(PK_AT(pk_sfc_lay), rrtmgp_pk_interp(
        pk_tsfc[col], pk_band, pk_totplnk, pk_nplanck, pk_nband, pk_t0,
        pk_dpl));
  } else {
    pk_sfc = sfc_source[sg];
  }
  float upv = __fmaf_rn(__fsub_rn(1.0f, sfc_emis[sg]), dnv,
                        __fmul_rn(sfc_emis[sg], pk_sfc));
  if (warp_fold) {
    rte_fold_emit(part_up, col, nlev, sfc, upv, pi, gl, gpt_tile,
                  gpt0, s_fold);
  } else {
    part_up[(base + sfc) * ncol + col] = upv;
  }
  for (int j = 0; j < nlay; ++j) {
    const int lay = top_at_1 ? nlay - 1 - j : j;
    // PINNED (13.6j): ptxas contracts this into an FMA at 34 registers
    // and does NOT at 48, which is how a branch nothing executed moved
    // 106 flux words.  The intrinsic states the reference behaviour.
    upv = __fmaf_rn(trans[lay], upv, src_up[lay]);
    const int ulev = top_at_1 ? lay : lay + 1;
    if (warp_fold) {
      rte_fold_emit(part_up, col, nlev, ulev, upv, pi, gl, gpt_tile,
                    gpt0, s_fold);
    } else {
      part_up[(base + ulev) * ncol + col] = upv;
    }
  }
}
#undef PK_AT

// Fold one g-point tile into the running LW flux, ascending g-point order.
extern "C" __global__ void rrtmgp_lw_flux_reduce(
    const float* part_up, const float* part_dn,
    float* flux_up, float* flux_dn,
    int ncol, int nlev, int gpt0, int ntile) {
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= ncol * nlev) return;
  const int lev = tid / ncol;
  const int col = tid - lev * ncol;
  const float pi = 3.14159265358979323846f;
  const int o = col * nlev + lev;
  float su = gpt0 == 0 ? 0.0f : flux_up[o];
  float sd = gpt0 == 0 ? 0.0f : flux_dn[o];
  for (int g = 0; g < ntile; ++g) {
    const int q = (g * nlev + lev) * ncol + col;
    su += pi * part_up[q];
    sd += pi * part_dn[q];
  }
  flux_up[o] = su;
  flux_dn[o] = sd;
}

// Slim SW two-stream solver: same arithmetic, three fewer column arrays.
//
// The original buffers ten arrays -- 2688 B of LOCAL memory per thread.
// Three of them are avoidable:
//
//   * dif_up is written by the back-substitution and read only by the final
//     store loop, so it stores as it is produced;
//   * dif_dn is a scalar recurrence read one level back, plus that same
//     store;
//   * denom is 1/(1 - rdif*albedo), and BOTH operands are still live in the
//     back-substitution, so recomputing it there is one FMA and one
//     reciprocal against a local round trip.  Same inputs, same operation,
//     same bits.
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
  const float eps = 1.1920928955078125e-7f;
  const float min_k = 1.0e4f * eps;
  const float min_mu = 3.4526698300124393e-4f;
  float rdif[RRTMGP_MAX_LAYERS], tdif[RRTMGP_MAX_LAYERS];
  float src_dn[RRTMGP_MAX_LAYERS], src_up[RRTMGP_MAX_LAYERS];
  float direct[RRTMGP_MAX_LAYERS + 1];
  float albedo[RRTMGP_MAX_LAYERS + 1], source[RRTMGP_MAX_LAYERS + 1];

  const int spec = col * ngpt + gpt;
  const int base = gl * nlev;
  // Only read on the folding path, where the block IS the fold group.
  extern __shared__ float s_fold[];
  direct[top] = inc_flux[spec] * mu0[col * nlay + top_lay];
  for (int j = 0; j < nlay; ++j) {
    const int lay = top_at_1 ? j : nlay - j - 1;
    const int ilev_in = top_at_1 ? lay : lay + 1;
    const int ilev_out = top_at_1 ? lay + 1 : lay;
    const int idx = (col * nlay + lay) * ngpt + gpt;
    float ts, ws, gs;
    if (fz_on) {
      // rrtmgp_finalize_cloud_sw, replicated: `tau`/`ssa` here ARE
      // gas_tau/gas_ssa and `asym` is never read.  Each volatile line in
      // that kernel is one binary FP op, so an explicit __f*_rn intrinsic
      // reproduces its value exactly; the volatiles existed to stop ptxas
      // contracting ACROSS lines, which the intrinsics also forbid.  The
      // bare tail is copied verbatim.
      const int fz_band = (col * nlay + lay) * fz_nband + fz_gpt_bands[gpt];
      float fz_tc = fz_cld_tau[fz_band];
      if (fz_have_mask) fz_tc = fz_tc * (float)fz_mask[idx];
      const float fz_wc = fz_cld_ssa[fz_band];
      const float fz_gc = fz_cld_asy[fz_band];
      const float fz_floor = 3.0f * 1.17549435e-38f;
      const float fz_total_tau = __fadd_rn(tau[idx], fz_tc);
      const float fz_gas_scatter = __fmul_rn(tau[idx], ssa[idx]);
      const float fz_cloud_scatter = __fmul_rn(fz_tc, fz_wc);
      const float fz_scatter = __fadd_rn(fz_gas_scatter, fz_cloud_scatter);
      const float fz_total_ssa = __fdiv_rn(
          fz_scatter, fmaxf(fz_floor, fz_total_tau));
      const float fz_gas_g = __fmul_rn(fz_gas_scatter, 0.0f);
      const float fz_cloud_g = __fmul_rn(fz_cloud_scatter, fz_gc);
      const float fz_gscatter = __fadd_rn(fz_gas_g, fz_cloud_g);
      const float fz_total_g = __fdiv_rn(
          fz_gscatter, fmaxf(fz_floor, fz_scatter));
      const float fz_f = fz_total_g * fz_total_g;
      const float fz_wf = fz_total_ssa * fz_f;
      ts = (1.0f - fz_wf) * fz_total_tau;
      ws = (fz_total_ssa - fz_wf) / fmaxf(fz_floor, 1.0f - fz_wf);
      gs = (fz_total_g - fz_f) / fmaxf(fz_floor, 1.0f - fz_f);
    } else {
      ts = tau[idx];
      ws = ssa[idx];
      gs = zero_g ? 0.0f : asym[idx];
    }
    const float gamma1 = (8.0f - ws * (5.0f + 3.0f * gs)) * 0.25f;
    const float gamma2 = 3.0f * ws * (1.0f - gs) * 0.25f;
    const float kval = sqrtf(fmaxf((gamma1 - gamma2)
                                   * (gamma1 + gamma2), min_k));
    const float ex = expf(-ts * kval);
    const float ex2 = ex * ex;
    float rt = 1.0f / (kval * (1.0f + ex2) + gamma1 * (1.0f - ex2));
    rdif[lay] = rt * gamma2 * (1.0f - ex2);
    tdif[lay] = rt * 2.0f * kval * ex;
    const float mu = fmaxf(min_mu, mu0[col * nlay + lay]);
    const float kmu = kval * mu;
    float dterm = 1.0f - kmu * kmu;
    if (fabsf(dterm) < eps) dterm = eps;
    rt = ws * rt / dterm;
    const float gamma3 = (2.0f - 3.0f * mu * gs) * 0.25f;
    const float gamma4 = 1.0f - gamma3;
    const float alpha1 = gamma1 * gamma4 + gamma2 * gamma3;
    const float alpha2 = gamma1 * gamma3 + gamma2 * gamma4;
    const float kg3 = kval * gamma3, kg4 = kval * gamma4;
    const float tnoscat = expf(-ts / mu);
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
    src_up[lay] = rdir * direct[ilev_in];
    src_dn[lay] = tdir * direct[ilev_in];
    direct[ilev_out] = tnoscat * direct[ilev_in];
    if (mu0[col * nlay + lay] <= 0.0f)
      src_up[lay] = src_dn[lay] = 0.0f;
  }
  const float source_sfc = mu0[col * nlay + sfc_lay] > 0.0f
      ? direct[sfc] * sfc_alb_dir[spec] : 0.0f;

  float dnv, upv;
  if (top_at_1) {
    albedo[nlay] = sfc_alb_dif[spec];
    source[nlay] = source_sfc;
    for (int lev = nlay - 1; lev >= 0; --lev) {
      const float dn_ = 1.0f / (1.0f - rdif[lev] * albedo[lev + 1]);
      albedo[lev] = rdif[lev] + tdif[lev] * tdif[lev]
          * albedo[lev + 1] * dn_;
      source[lev] = src_up[lev] + tdif[lev] * dn_
          * (source[lev + 1] + albedo[lev + 1] * src_dn[lev]);
    }
    dnv = 0.0f;
    upv = dnv * albedo[0] + source[0];
    if (warp_fold) {
      rte_fold_emit(part_up, col, nlev, 0, upv, 1.0f,
                    gl, gpt_tile, gpt0, s_fold);
      rte_fold_emit(part_dn, col, nlev, 0, dnv + direct[0], 1.0f,
                    gl, gpt_tile, gpt0, s_fold);
      rte_fold_emit(part_dir, col, nlev, 0, direct[0], 1.0f,
                    gl, gpt_tile, gpt0, s_fold);
    } else {
      part_up[base * ncol + col] = upv;
      part_dn[base * ncol + col] = dnv + direct[0];
      part_dir[base * ncol + col] = direct[0];
    }
    for (int lev = 1; lev < nlev; ++lev) {
      const int lay = lev - 1;
      const float dn_ = 1.0f / (1.0f - rdif[lay] * albedo[lev]);
      dnv = (tdif[lay] * dnv + rdif[lay] * source[lev] + src_dn[lay]) * dn_;
      upv = dnv * albedo[lev] + source[lev];
      if (warp_fold) {
        rte_fold_emit(part_up, col, nlev, lev, upv, 1.0f,
                      gl, gpt_tile, gpt0, s_fold);
        rte_fold_emit(part_dn, col, nlev, lev, dnv + direct[lev], 1.0f,
                      gl, gpt_tile, gpt0, s_fold);
        rte_fold_emit(part_dir, col, nlev, lev, direct[lev], 1.0f,
                      gl, gpt_tile, gpt0, s_fold);
      } else {
        part_up[(base + lev) * ncol + col] = upv;
        part_dn[(base + lev) * ncol + col] = dnv + direct[lev];
        part_dir[(base + lev) * ncol + col] = direct[lev];
      }
    }
  } else {
    albedo[0] = sfc_alb_dif[spec];
    source[0] = source_sfc;
    for (int lev = 0; lev < nlay; ++lev) {
      const float dn_ = 1.0f / (1.0f - rdif[lev] * albedo[lev]);
      albedo[lev + 1] = rdif[lev] + tdif[lev] * tdif[lev] * albedo[lev] * dn_;
      source[lev + 1] = src_up[lev] + tdif[lev] * dn_
          * (source[lev] + albedo[lev] * src_dn[lev]);
    }
    dnv = 0.0f;
    upv = dnv * albedo[nlay] + source[nlay];
    if (warp_fold) {
      rte_fold_emit(part_up, col, nlev, nlay, upv, 1.0f,
                    gl, gpt_tile, gpt0, s_fold);
      rte_fold_emit(part_dn, col, nlev, nlay, dnv + direct[nlay], 1.0f,
                    gl, gpt_tile, gpt0, s_fold);
      rte_fold_emit(part_dir, col, nlev, nlay, direct[nlay], 1.0f,
                    gl, gpt_tile, gpt0, s_fold);
    } else {
      part_up[(base + nlay) * ncol + col] = upv;
      part_dn[(base + nlay) * ncol + col] = dnv + direct[nlay];
      part_dir[(base + nlay) * ncol + col] = direct[nlay];
    }
    for (int lev = nlay - 1; lev >= 0; --lev) {
      const float dn_ = 1.0f / (1.0f - rdif[lev] * albedo[lev]);
      dnv = (tdif[lev] * dnv + rdif[lev] * source[lev] + src_dn[lev]) * dn_;
      upv = dnv * albedo[lev] + source[lev];
      if (warp_fold) {
        rte_fold_emit(part_up, col, nlev, lev, upv, 1.0f,
                      gl, gpt_tile, gpt0, s_fold);
        rte_fold_emit(part_dn, col, nlev, lev, dnv + direct[lev], 1.0f,
                      gl, gpt_tile, gpt0, s_fold);
        rte_fold_emit(part_dir, col, nlev, lev, direct[lev], 1.0f,
                      gl, gpt_tile, gpt0, s_fold);
      } else {
        part_up[(base + lev) * ncol + col] = upv;
        part_dn[(base + lev) * ncol + col] = dnv + direct[lev];
        part_dir[(base + lev) * ncol + col] = direct[lev];
      }
    }
  }
}

// Fold one g-point tile into the running SW fluxes, ascending g-point order.
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
    su += part_up[q];
    sd += part_dn[q];
    sr += part_dir[q];
  }
  flux_up[o] = su;
  flux_dn[o] = sd;
  flux_dir[o] = sr;
}
