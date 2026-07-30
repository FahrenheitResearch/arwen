// RTE longwave no-scattering and shortwave PIFM two-stream solvers.
// One CUDA thread integrates one complete spectral atmospheric column.
// Transcribed from RTE+RRTMGP fa107a1
// rte/kernels/mo_rte_solver_kernels.F90:51-240,503-745,985-1245 and
// rte/kernels/mo_optical_props_kernels.F90:76-98.

#define RRTMGP_MAX_LAYERS 128
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

extern "C" __global__ void rrtmgp_lw_noscat(
    const float* tau, const float* lay_source, const float* lev_source,
    const float* sfc_source, const float* sfc_emis,
    const float* incident_flux, float* flux_up, float* flux_dn,
    int ncol, int nlay, int ngpt, int top_at_1) {
  const int col = blockDim.x * blockIdx.x + threadIdx.x;
  if (col >= ncol || nlay > RRTMGP_MAX_LAYERS) return;
  const int nlev = nlay + 1;
  const int top = top_at_1 ? 0 : nlay;
  const int sfc = top_at_1 ? nlay : 0;
  const float d = 1.0f / 0.6096748751f;
  const float pi = 3.14159265358979323846f;
  // Upstream uses epsilon(1._wp)^(1/4).  Derive that branch point from this
  // kernel's float working real so cancellation-prone arithmetic is avoided.
  const RRTMGP_REAL tau_thresh =
      RRTMGP_REAL_SQRT(RRTMGP_REAL_SQRT(RRTMGP_REAL_EPSILON));
  float trans[RRTMGP_MAX_LAYERS];
  float src_dn[RRTMGP_MAX_LAYERS];
  float src_up[RRTMGP_MAX_LAYERS];
  float up[RRTMGP_MAX_LAYERS + 1];
  float dn[RRTMGP_MAX_LAYERS + 1];
  for (int lev = 0; lev < nlev; ++lev) {
    flux_up[col * nlev + lev] = 0.0f;
    flux_dn[col * nlev + lev] = 0.0f;
  }
  for (int gpt = 0; gpt < ngpt; ++gpt) {
    for (int lay = 0; lay < nlay; ++lay) {
      const int cellg = (col * nlay + lay) * ngpt + gpt;
      const float path = tau[cellg] * d;
      const float tr = expf(-path);
      trans[lay] = tr;
      const float fact = path > tau_thresh
          ? (1.0f - tr) / path - tr
          : path * (0.5f + path * (-1.0f / 3.0f + path / 8.0f));
      const float lays = lay_source[cellg];
      const float lo = lev_source[(col * nlev + lay) * ngpt + gpt];
      const float hi = lev_source[(col * nlev + lay + 1) * ngpt + gpt];
      const float inc = (1.0f - tr) * hi + 2.0f * fact * (lays - hi);
      const float dec = (1.0f - tr) * lo + 2.0f * fact * (lays - lo);
      src_dn[lay] = top_at_1 ? inc : dec;
      src_up[lay] = top_at_1 ? dec : inc;
    }
    dn[top] = incident_flux[col * ngpt + gpt] / pi;
    if (top_at_1) {
      for (int lev = 1; lev < nlev; ++lev)
        dn[lev] = trans[lev - 1] * dn[lev - 1] + src_dn[lev - 1];
    } else {
      for (int lev = nlay - 1; lev >= 0; --lev)
        dn[lev] = trans[lev] * dn[lev + 1] + src_dn[lev];
    }
    const int sg = col * ngpt + gpt;
    up[sfc] = dn[sfc] * (1.0f - sfc_emis[sg])
              + sfc_emis[sg] * sfc_source[sg];
    if (top_at_1) {
      for (int lev = nlay - 1; lev >= 0; --lev)
        up[lev] = trans[lev] * up[lev + 1] + src_up[lev];
    } else {
      for (int lev = 1; lev < nlev; ++lev)
        up[lev] = trans[lev - 1] * up[lev - 1] + src_up[lev - 1];
    }
    for (int lev = 0; lev < nlev; ++lev) {
      flux_up[col * nlev + lev] += pi * up[lev];
      flux_dn[col * nlev + lev] += pi * dn[lev];
    }
  }
}

extern "C" __global__ void rrtmgp_sw_2stream(
    const float* tau, const float* ssa, const float* asym,
    const float* mu0, const float* sfc_alb_dir, const float* sfc_alb_dif,
    const float* inc_flux, float* flux_up, float* flux_dn, float* flux_dir,
    int ncol, int nlay, int ngpt, int top_at_1, int zero_g) {
  const int col = blockDim.x * blockIdx.x + threadIdx.x;
  if (col >= ncol || nlay > RRTMGP_MAX_LAYERS) return;
  const int nlev = nlay + 1;
  const int top = top_at_1 ? 0 : nlay;
  const int top_lay = top_at_1 ? 0 : nlay - 1;
  const int sfc = top_at_1 ? nlay : 0;
  const int sfc_lay = top_at_1 ? nlay - 1 : 0;
  // Use the working-precision floor from epsilon(1._wp).  The device
  // implementation is FP32, so retaining the upstream FP64 floor here
  // destabilizes the near-conservative Rayleigh limit through cancellation.
  const float eps = 1.1920928955078125e-7f;
  const float min_k = 1.0e4f * eps;
  const float min_mu = 3.4526698300124393e-4f;
  float rdif[RRTMGP_MAX_LAYERS], tdif[RRTMGP_MAX_LAYERS];
  float src_dn[RRTMGP_MAX_LAYERS], src_up[RRTMGP_MAX_LAYERS];
  float direct[RRTMGP_MAX_LAYERS + 1];
  float albedo[RRTMGP_MAX_LAYERS + 1], source[RRTMGP_MAX_LAYERS + 1];
  float denom[RRTMGP_MAX_LAYERS];
  float dif_dn[RRTMGP_MAX_LAYERS + 1], dif_up[RRTMGP_MAX_LAYERS + 1];
  for (int lev = 0; lev < nlev; ++lev) {
    flux_up[col * nlev + lev] = 0.0f;
    flux_dn[col * nlev + lev] = 0.0f;
    flux_dir[col * nlev + lev] = 0.0f;
  }

  for (int gpt = 0; gpt < ngpt; ++gpt) {
    const int spec = col * ngpt + gpt;
    direct[top] = inc_flux[spec] * mu0[col * nlay + top_lay];
    dif_dn[top] = 0.0f;
    for (int j = 0; j < nlay; ++j) {
      const int lay = top_at_1 ? j : nlay - j - 1;
      const int ilev_in = top_at_1 ? lay : lay + 1;
      const int ilev_out = top_at_1 ? lay + 1 : lay;
      const int idx = (col * nlay + lay) * ngpt + gpt;
      const float ts = tau[idx], ws = ssa[idx];
      const float gs = zero_g ? 0.0f : asym[idx];
      const float gamma1 = (8.0f - ws * (5.0f + 3.0f * gs)) * 0.25f;
      const float gamma2 = 3.0f * ws * (1.0f - gs) * 0.25f;
      const float kval = sqrtf(fmaxf((gamma1 - gamma2)
                                     * (gamma1 + gamma2), min_k));
      const float ex = expf(-ts * kval);
      const float ex2 = ex * ex;
      float rt = 1.0f / (kval * (1.0f + ex2)
                         + gamma1 * (1.0f - ex2));
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
    if (top_at_1) {
      albedo[nlay] = sfc_alb_dif[spec];
      source[nlay] = source_sfc;
      for (int lev = nlay - 1; lev >= 0; --lev) {
        denom[lev] = 1.0f / (1.0f - rdif[lev] * albedo[lev + 1]);
        albedo[lev] = rdif[lev] + tdif[lev] * tdif[lev]
            * albedo[lev + 1] * denom[lev];
        source[lev] = src_up[lev] + tdif[lev] * denom[lev]
            * (source[lev + 1] + albedo[lev + 1] * src_dn[lev]);
      }
      dif_up[0] = dif_dn[0] * albedo[0] + source[0];
      for (int lev = 1; lev < nlev; ++lev) {
        const int lay = lev - 1;
        dif_dn[lev] = (tdif[lay] * dif_dn[lev - 1]
                       + rdif[lay] * source[lev] + src_dn[lay]) * denom[lay];
        dif_up[lev] = dif_dn[lev] * albedo[lev] + source[lev];
      }
    } else {
      albedo[0] = sfc_alb_dif[spec];
      source[0] = source_sfc;
      for (int lev = 0; lev < nlay; ++lev) {
        denom[lev] = 1.0f / (1.0f - rdif[lev] * albedo[lev]);
        albedo[lev + 1] = rdif[lev] + tdif[lev] * tdif[lev]
            * albedo[lev] * denom[lev];
        source[lev + 1] = src_up[lev] + tdif[lev] * denom[lev]
            * (source[lev] + albedo[lev] * src_dn[lev]);
      }
      dif_up[nlay] = dif_dn[nlay] * albedo[nlay] + source[nlay];
      for (int lev = nlay - 1; lev >= 0; --lev) {
        dif_dn[lev] = (tdif[lev] * dif_dn[lev + 1]
                       + rdif[lev] * source[lev] + src_dn[lev]) * denom[lev];
        dif_up[lev] = dif_dn[lev] * albedo[lev] + source[lev];
      }
    }
    for (int lev = 0; lev < nlev; ++lev) {
      flux_up[col * nlev + lev] += dif_up[lev];
      flux_dn[col * nlev + lev] += dif_dn[lev] + direct[lev];
      flux_dir[col * nlev + lev] += direct[lev];
    }
  }
}
