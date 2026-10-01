// RRTMGP band cloud optics, one thread per cell and spectral band.
//
// Transcribed from earth-system-radiation/rte-rrtmgp fa107a1:
// rrtmgp/kernels/mo_cloud_optics_rrtmgp_kernels.F90:24-65 and
// rrtmgp/frontend/mo_cloud_optics_rrtmgp.F90:256-431.

__device__ __forceinline__ float interp_table(
    const float* table, int size, int band, int nband, float fraction) {
  const float lo = table[size * nband + band];
  return lo + fraction * (table[(size + 1) * nband + band] - lo);
}

__device__ __forceinline__ float interp_ice(
    const float* table, int size, int band, int nband, int nrgh,
    float fraction) {
  // The reference all-sky example selects medium ice roughness (category 2)
  // at examples/all-sky/rrtmgp_allsky.F90:211-214.
  const int lo_idx = (size * nband + band) * nrgh + 1;
  const int hi_idx = ((size + 1) * nband + band) * nrgh + 1;
  return table[lo_idx] + fraction * (table[hi_idx] - table[lo_idx]);
}

extern "C" __global__ void rrtmgp_cloud_optics(
    const float* clwp, const float* ciwp, const float* reliq,
    const float* dgice, const float* extliq, const float* ssaliq,
    const float* asyliq, const float* extice, const float* ssaice,
    const float* asyice, float* tau, float* ssa, float* asym,
    int ncell, int nband, int nliq, int nice, int nrgh,
    float liq_offset, float liq_step, float ice_offset, float ice_step) {
  const int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= ncell * nband) return;
  const int cell = idx / nband;
  const int band = idx - cell * nband;
  float ltau = 0.0f, lscatter = 0.0f, lgscatter = 0.0f;
  float itau = 0.0f, iscatter = 0.0f, igscatter = 0.0f;
  if (clwp[cell] > 0.0f) {
    const float pos = (reliq[cell] - liq_offset) / liq_step;
    const int isize = max(0, min((int)floorf(pos), nliq - 2));
    const float fraction = pos - isize;
    ltau = clwp[cell] * interp_table(
        extliq, isize, band, nband, fraction);
    lscatter = ltau * interp_table(
        ssaliq, isize, band, nband, fraction);
    lgscatter = lscatter * interp_table(
        asyliq, isize, band, nband, fraction);
  }
  if (ciwp[cell] > 0.0f) {
    const float pos = (dgice[cell] - ice_offset) / ice_step;
    const int isize = max(0, min((int)floorf(pos), nice - 2));
    const float fraction = pos - isize;
    itau = ciwp[cell] * interp_ice(
        extice, isize, band, nband, nrgh, fraction);
    iscatter = itau * interp_ice(
        ssaice, isize, band, nband, nrgh, fraction);
    igscatter = iscatter * interp_ice(
        asyice, isize, band, nband, nrgh, fraction);
  }
  const float total = ltau + itau;
  const float scatter = lscatter + iscatter;
  tau[idx] = total;
  ssa[idx] = scatter / fmaxf(1.1920928955078125e-7f, total);
  asym[idx] = (lgscatter + igscatter)
              / fmaxf(1.1920928955078125e-7f, scatter);
}

// Expand band-resolved cloud optics at the point of use and emit only the
// final LW absorption optical depth.  Operation order matches
// add_cloud_optics: tau_gas + tau_cloud * (1 - ssa_cloud).  Each CuPy
// operator in that legacy expression was a separate kernel launch, hence a
// separate round-to-nearest FP32 operation.  Explicit intrinsics preserve
// those boundaries inside this kernel; volatile locals alone do not forbid
// ptxas from contracting a multiply followed by an add.
extern "C" __global__ void rrtmgp_finalize_cloud_lw(
    const float* tau_gas, const float* tau_cloud, const float* ssa_cloud,
    const int* gpoint_bands, const unsigned char* cloud_mask,
    float* tau, int n, int ngpt, int nband, int have_mask) {
  const int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= n) return;
  const int gpt = idx % ngpt;
  const int cell = idx / ngpt;
  const int band_idx = cell * nband + gpoint_bands[gpt];
  float tc = tau_cloud[band_idx];
  if (have_mask) tc = __fmul_rn(tc, (float)cloud_mask[idx]);
  const float one_minus_ssa = __fsub_rn(1.0f, ssa_cloud[band_idx]);
  const float cloud_absorption = __fmul_rn(tc, one_minus_ssa);
  tau[idx] = __fadd_rn(tau_gas[idx], cloud_absorption);
}

// Expand/add cloud optics and immediately apply RRTMGP's default SW delta
// scaling.  Volatile intermediates retain the same FP32 materialization seams
// as the former add kernel followed by rrtmgp_delta_scale.
extern "C" __global__ void rrtmgp_finalize_cloud_sw(
    const float* tau_gas, const float* ssa_gas,
    const float* tau_cloud, const float* ssa_cloud, const float* g_cloud,
    const int* gpoint_bands, const unsigned char* cloud_mask,
    float* tau, float* ssa, float* asym,
    int n, int ngpt, int nband, int have_mask) {
  const int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= n) return;
  const int gpt = idx % ngpt;
  const int cell = idx / ngpt;
  const int band_idx = cell * nband + gpoint_bands[gpt];
  float tc = tau_cloud[band_idx];
  if (have_mask) tc = tc * (float)cloud_mask[idx];
  const float wc = ssa_cloud[band_idx];
  const float gc = g_cloud[band_idx];
  const float floor = 3.0f * 1.17549435e-38f;

  volatile float total_tau_round = tau_gas[idx] + tc;
  volatile float gas_scatter_round = tau_gas[idx] * ssa_gas[idx];
  volatile float cloud_scatter_round = tc * wc;
  volatile float scatter_round = gas_scatter_round + cloud_scatter_round;
  const float total_tau = total_tau_round;
  const float scatter = scatter_round;
  volatile float total_ssa_round = scatter / fmaxf(floor, total_tau);
  // Gas Rayleigh asymmetry is identically zero.  Use the literal-zero
  // specialization while retaining the old multiply/add seams (including
  // IEEE propagation for any unexpected non-finite gas optical property).
  volatile float gas_gscatter_round = gas_scatter_round * 0.0f;
  volatile float cloud_gscatter_round = cloud_scatter_round * gc;
  volatile float gscatter_round = gas_gscatter_round + cloud_gscatter_round;
  volatile float total_g_round = gscatter_round / fmaxf(floor, scatter);
  const float total_ssa = total_ssa_round;
  const float total_g = total_g_round;

  const float f = total_g * total_g;
  const float wf = total_ssa * f;
  tau[idx] = (1.0f - wf) * total_tau;
  ssa[idx] = (total_ssa - wf) / fmaxf(floor, 1.0f - wf);
  asym[idx] = (total_g - f) / fmaxf(floor, 1.0f - f);
}

// Pure copy and positive-zero fill for the five clear upper-layer fields.
extern "C" __global__ void rrtmgp_clear_upper_layers(
    const float* a, const float* b, const float* c, const float* d,
    const float* e, float* oa, float* ob, float* oc, float* od, float* oe,
    int ncol, int model_nlay, int nlay) {
  const int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index >= ncol * nlay) return;
  const int col = index / nlay;
  const int lay = index % nlay;
  if (lay < model_nlay) {
    const int src = col * model_nlay + lay;
    oa[index] = a[src]; ob[index] = b[src]; oc[index] = c[src];
    od[index] = d[src]; oe[index] = e[src];
  } else {
    oa[index] = 0.0f; ob[index] = 0.0f; oc[index] = 0.0f;
    od[index] = 0.0f; oe[index] = 0.0f;
  }
}

// Copy the two model-interface flux fields, optionally applying daylight.
extern "C" __global__ void rrtmgp_store_model_flux_pair(
    const float* up, const float* dn, const unsigned char* daylight,
    float* out_up, float* out_dn, int ncol, int source_nlev,
    int model_nlev, int apply_daylight) {
  const int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index >= ncol * model_nlev) return;
  const int col = index / model_nlev;
  const int lev = index % model_nlev;
  if (!apply_daylight || daylight[col]) {
    out_up[index] = up[col * source_nlev + lev];
    out_dn[index] = dn[col * source_nlev + lev];
  } else {
    out_up[index] = 0.0f;
    out_dn[index] = 0.0f;
  }
}

// Three independent broadcasts, with exactly one store for each element.
extern "C" __global__ void rrtmgp_prepare_sw_inputs(
    const float* albedo, const float* solar, const float* mu,
    float* albedo_gpt, float* inc_gpt, float* mu0,
    int ncol, int ngpt, int nlay) {
  const int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index < ncol * ngpt) {
    albedo_gpt[index] = albedo[index / ngpt];
    inc_gpt[index] = solar[index % ngpt];
  }
  if (index < ncol * nlay) mu0[index] = mu[index / nlay];
}

extern "C" __global__ void rrtmgp_prepare_sw_inputs_fp64(
    const float* albedo, const float* solar, const double* mu,
    float* albedo_gpt, float* inc_gpt, float* mu0,
    int ncol, int ngpt, int nlay) {
  const int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index < ncol * ngpt) {
    albedo_gpt[index] = albedo[index / ngpt];
    inc_gpt[index] = solar[index % ngpt];
  }
  if (index < ncol * nlay) mu0[index] = __double2float_rn(mu[index / nlay]);
}


// Preserve concatenate dtype promotion, including float64 cap temperatures.
template <typename T, bool DERIVE_TEMPERATURE = false>
__device__ __forceinline__ void rrtmgp_append_profile_impl(
    const float* play, const float* plev, const float* tlay,
    const float* tlev, const float* qv,
    const float* up_play, const float* up_plev, const T* up_tlay,
    const T* up_tlev, const float* up_qv,
    float* out_play, float* out_plev, T* out_tlay, T* out_tlev, float* out_qv,
    int ncol, int model_nlay, int upper_nlay,
    int spc, int spl, int sec, int sel, int stc, int stl,
    int sic, int sil, int sqc, int sql) {
  const int nlay = model_nlay + upper_nlay;
  const int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index >= ncol * (nlay + 1)) return;
  const int col = index / (nlay + 1);
  const int lev = index % (nlay + 1);
  if (lev <= model_nlay) {
    const int src = col * (model_nlay + 1) + lev;
    out_plev[index] = plev[src]; out_tlev[index] = (T)tlev[src];
  } else {
    const int j = lev - model_nlay - 1;
    out_plev[index] = up_plev[col * sec + j * sel];
    if constexpr (DERIVE_TEMPERATURE) {
      const double top = (double)tlev[col * (model_nlay + 1) + model_nlay];
      const double offset = __dsub_rn(top, up_tlay[col * stc]);
      out_tlev[index] = __dadd_rn(up_tlev[col * sic + j * sil], offset);
    } else {
      out_tlev[index] = up_tlev[col * sic + j * sil];
    }
  }
  if (lev < nlay) {
    const int target = col * nlay + lev;
    if (lev < model_nlay) {
      const int src = col * model_nlay + lev;
      out_play[target] = play[src]; out_tlay[target] = (T)tlay[src];
      out_qv[target] = qv[src];
    } else {
      const int j = lev - model_nlay;
      out_play[target] = up_play[col * spc + j * spl];
      if constexpr (DERIVE_TEMPERATURE) {
        const double top = (double)tlev[col * (model_nlay + 1) + model_nlay];
        const double offset = __dsub_rn(top, up_tlay[col * stc]);
        const double next = __dadd_rn(up_tlev[col * sic + j * sil], offset);
        const double previous = j == 0 ? top :
            __dadd_rn(up_tlev[col * sic + (j - 1) * sil], offset);
        out_tlay[target] = __dmul_rn(0.5, __dadd_rn(previous, next));
      } else {
        out_tlay[target] = up_tlay[col * stc + j * stl];
      }
      out_qv[target] = up_qv[col * sqc + j * sql];
    }
  }
}

extern "C" __global__ void rrtmgp_append_profile_fp32(
    const float* play, const float* plev, const float* tlay,
    const float* tlev, const float* qv,
    const float* up_play, const float* up_plev, const float* up_tlay,
    const float* up_tlev, const float* up_qv,
    float* out_play, float* out_plev, float* out_tlay, float* out_tlev, float* out_qv,
    int ncol, int model_nlay, int upper_nlay,
    int spc, int spl, int sec, int sel, int stc, int stl,
    int sic, int sil, int sqc, int sql) {
  rrtmgp_append_profile_impl<float>(
      play, plev, tlay, tlev, qv, up_play, up_plev, up_tlay, up_tlev, up_qv,
      out_play, out_plev, out_tlay, out_tlev, out_qv, ncol, model_nlay, upper_nlay,
      spc, spl, sec, sel, stc, stl, sic, sil, sqc, sql);
}

extern "C" __global__ void rrtmgp_append_profile_fp64(
    const float* play, const float* plev, const float* tlay,
    const float* tlev, const float* qv,
    const float* up_play, const float* up_plev, const double* up_tlay,
    const double* up_tlev, const float* up_qv,
    float* out_play, float* out_plev, double* out_tlay, double* out_tlev, float* out_qv,
    int ncol, int model_nlay, int upper_nlay,
    int spc, int spl, int sec, int sel, int stc, int stl,
    int sic, int sil, int sqc, int sql) {
  rrtmgp_append_profile_impl<double>(
      play, plev, tlay, tlev, qv, up_play, up_plev, up_tlay, up_tlev, up_qv,
      out_play, out_plev, out_tlay, out_tlev, out_qv, ncol, model_nlay, upper_nlay,
      spc, spl, sec, sel, stc, stl, sic, sil, sqc, sql);
}

extern "C" __global__ void rrtmgp_append_lw_profile(
    const float* play, const float* plev, const float* tlay,
    const float* tlev, const float* qv,
    const float* up_play, const float* up_plev, const double* up_tlay,
    const double* up_tlev, const float* up_qv,
    float* out_play, float* out_plev, double* out_tlay, double* out_tlev, float* out_qv,
    int ncol, int model_nlay, int upper_nlay,
    int spc, int spl, int sec, int sel, int stc, int stl,
    int sic, int sil, int sqc, int sql) {
  rrtmgp_append_profile_impl<double, true>(
      play, plev, tlay, tlev, qv, up_play, up_plev, up_tlay, up_tlev, up_qv,
      out_play, out_plev, out_tlay, out_tlev, out_qv, ncol, model_nlay, upper_nlay,
      spc, spl, sec, sel, stc, stl, sic, sil, sqc, sql);
}

// Narrow both retained float64 temperature fields once for all consumers.
extern "C" __global__ void rrtmgp_solver_temperatures(
    const double* tlay, const double* tlev, float* out_tlay, float* out_tlev,
    int ncol, int nlay) {
  const int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index < ncol * nlay) out_tlay[index] = __double2float_rn(tlay[index]);
  if (index < ncol * (nlay + 1)) out_tlev[index] = __double2float_rn(tlev[index]);
}
