// RRTMGP gas optical depths, one CUDA thread per (column, layer, g-point).
//
// Transcribed from earth-system-radiation/rte-rrtmgp fa107a1:
// rrtmgp/kernels/mo_gas_optics_rrtmgp_kernels.F90
//   interpolation                    lines 37-170
//   gas_optical_depths_major        lines 345-396
//   gas_optical_depths_minor        lines 402-501
//   compute_tau_rayleigh            lines 506-565
// and rte/kernels/mo_gas_optics_utils.F90:get_layer_number lines 127-152.

__device__ __forceinline__ int km_index(
    int jt, int je, int jp, int g, int neta, int npressk, int ngpt) {
  return (((jt * neta + je) * npressk + jp) * ngpt + g);
}

__device__ __forceinline__ int k2_index(
    int jt, int je, int k, int neta, int nk) {
  return ((jt * neta + je) * nk + k);
}

extern "C" __global__ void rrtmgp_interpolation_prepass(
    const float* play, const float* tlay,
    const float* press_ref, const float* temp_ref, float press_ref_trop,
    int* iatm_out, int* jt_out, int* jp_out,
    float* ftemp_out, float* fpress_out,
    int ncell, int ntemp, int npres) {
  const int cell = blockDim.x * blockIdx.x + threadIdx.x;
  if (cell >= ncell) return;
  const float p = play[cell];
  const float t = tlay[cell];
  const float dtemp = (temp_ref[ntemp - 1] - temp_ref[0]) / (ntemp - 1);
  const float dpress = (logf(press_ref[npres - 1]) - logf(press_ref[0]))
                       / (npres - 1);
  const int iatm = p > press_ref_trop ? 0 : 1;
  int jt = (int)((t - (temp_ref[0] - dtemp)) / dtemp);
  jt = max(1, min(ntemp - 1, jt)) - 1;
  const float ftemp = (t - temp_ref[jt]) / dtemp;
  const float locpress = 1.0f + (logf(p) - logf(press_ref[0])) / dpress;
  int jp1 = (int)truncf(locpress);
  jp1 = max(1, min(npres - 1, jp1));
  iatm_out[cell] = iatm;
  jt_out[cell] = jt;
  jp_out[cell] = (jp1 - 1) + iatm;
  ftemp_out[cell] = ftemp;
  fpress_out[cell] = locpress - (float)jp1;
}

// Test oracle retaining the exact interpolation statements that were inline
// in rrtmgp_gas_optics before the production prepass.  Keep this independent
// from rrtmgp_interpolation_prepass so exact-array tests can detect changes to
// either transcription, its launch indexing, or its stored FP32 values.
extern "C" __global__ void rrtmgp_interpolation_inline_reference(
    const float* play, const float* tlay,
    const float* press_ref, const float* temp_ref, float press_ref_trop,
    int* iatm_out, int* jt_out, int* jp_out,
    float* ftemp_out, float* fpress_out,
    int ncell, int ntemp, int npres) {
  const int cell = blockDim.x * blockIdx.x + threadIdx.x;
  if (cell >= ncell) return;
  const float p = play[cell];
  const float t = tlay[cell];
  const float dtemp = (temp_ref[ntemp - 1] - temp_ref[0]) / (ntemp - 1);
  const float dpress = (logf(press_ref[npres - 1]) - logf(press_ref[0]))
                       / (npres - 1);
  const int iatm = p > press_ref_trop ? 0 : 1;
  int jt = (int)((t - (temp_ref[0] - dtemp)) / dtemp);
  jt = max(1, min(ntemp - 1, jt)) - 1;
  const float ftemp = (t - temp_ref[jt]) / dtemp;
  const float locpress = 1.0f + (logf(p) - logf(press_ref[0])) / dpress;
  int jp1 = (int)truncf(locpress);
  jp1 = max(1, min(npres - 1, jp1));
  iatm_out[cell] = iatm;
  jt_out[cell] = jt;
  jp_out[cell] = (jp1 - 1) + iatm;
  ftemp_out[cell] = ftemp;
  fpress_out[cell] = locpress - (float)jp1;
}

extern "C" __global__ void rrtmgp_gas_optics(
    const float* play, const float* plev, const float* tlay,
    const float* vmr, const int* iatm_meta, const int* jt_meta,
    const int* jp_meta, const float* ftemp_meta, const float* fpress_meta,
    const float* vmr_ref, const int* flavor, const int* gpoint_flavor,
    const float* kmajor,
    const float* kminor_lower, const float* kminor_upper,
    const int* minor_limits_lower, const int* minor_limits_upper,
    const unsigned char* density_lower, const unsigned char* density_upper,
    const unsigned char* complement_lower,
    const unsigned char* complement_upper,
    const int* idx_minor_lower, const int* idx_minor_upper,
    const int* idx_scaling_lower, const int* idx_scaling_upper,
    const int* kminor_start_lower, const int* kminor_start_upper,
    const int* minor_gpt_start_lower, const int* minor_gpt_list_lower,
    const int* minor_gpt_start_upper, const int* minor_gpt_list_upper,
    const float* rayleigh,
    float* tau, float* ssa,
    int ncol, int nlay, int ngas, int nflav, int ngpt,
    int ntemp, int npres, int neta,
    int nk_lower, int nk_upper, int idx_h2o, int is_sw,
    int nminor_lower, int nminor_upper) {
  // One thread per (column, layer, g-point).  Neither loop this replaces
  // carried anything across iterations -- every output element is a pure
  // function of its own (cell, gpt) -- so the transcribed body is unchanged
  // and the answers are bit-identical to the one-thread-per-column launch.
  // The bare scopes keep that body byte-for-byte auditable against the
  // Fortran ranges cited above.
  //
  // ONE BLOCK PER CELL, threads striding the g-point axis.  The flat
  // one-thread-per-(cell, g-point) launch it replaces made every one of a
  // cell's ngpt threads rebuild the per-cell preamble; at 64 threads a
  // thread now covers four g-points and pays for it once, and the whole
  // per-cell working set stays in registers across them.  Ablated by
  // removal: the geometry alone is worth 41% of the kernel and the shared
  // scaling below another 8%, and registers fall 56 -> 40.  Bit-identical
  // either way -- every output element is still a pure function of its own
  // (cell, g-point), evaluated by the same expressions in the same order.
  const int cell_id = blockIdx.x;
  // The WHOLE block leaves together, so the __syncthreads below is reached
  // by every thread that reaches the block at all.
  if (cell_id >= ncol * nlay) return;
  const int col = cell_id / nlay;
  const int lay = cell_id - col * nlay;
  extern __shared__ float s_scaling[];

  const int npressk = npres + 1;
  const float avogad = 6.02214076e23f;
  const float m_dry = 0.028964f;
  const float m_h2o = 0.018016f;
  const float grav = 9.80665f;

  {
    const int cell = col * nlay + lay;
    const float p = play[cell];
    const float t = tlay[cell];
    const float h2o_vmr = vmr[cell * (ngas + 1) + idx_h2o];
    const float fact_dry = 1.0f / (1.0f + h2o_vmr);
    const float m_air = (m_dry + m_h2o * h2o_vmr) * fact_dry;
    const float dp = fabsf(plev[col * (nlay + 1) + lay + 1]
                           - plev[col * (nlay + 1) + lay]);
    const float col_dry = dp * avogad * fact_dry
                          / (10000.0f * m_air * grav);
    const int iatm = iatm_meta[cell];
    const int jt = jt_meta[cell];
    const int jp = jp_meta[cell];
    const float ftemp = ftemp_meta[cell];
    const float fpress = fpress_meta[cell];

    const int nk = iatm == 0 ? nk_lower : nk_upper;
    const float* kminor = iatm == 0 ? kminor_lower : kminor_upper;
    const int* limits = iatm == 0 ? minor_limits_lower : minor_limits_upper;
    const unsigned char* density = iatm == 0 ? density_lower : density_upper;
    const unsigned char* complement = iatm == 0
        ? complement_lower : complement_upper;
    const int* idx_minor = iatm == 0 ? idx_minor_lower : idx_minor_upper;
    const int* idx_scaling = iatm == 0
        ? idx_scaling_lower : idx_scaling_upper;
    const int* starts = iatm == 0 ? kminor_start_lower : kminor_start_upper;
    // Only the minor entries whose g-point range covers a given thread, in
    // ascending `m`.  Scanning all of them and skipping was 94% waste: a
    // g-point matches 3.8 of the 60 LW lower entries on average.  The CSR
    // list preserves the visit order exactly, so `tau_abs` takes the same
    // additions in the same sequence -- bit-identical, not close.
    const int* mstart = iatm == 0
        ? minor_gpt_start_lower : minor_gpt_start_upper;
    const int* mlist = iatm == 0
        ? minor_gpt_list_lower : minor_gpt_list_upper;
    // `scaling` is a function of the CELL and the minor entry, never of the
    // g-point -- but each entry covers a band, so every entry's scaling was
    // being evaluated once per g-point in that band: ~16 evaluations of one
    // expression, two divisions each.  The block computes each exactly once.
    // Same expression, same operand order, same bits.
    const int nminor = iatm == 0 ? nminor_lower : nminor_upper;
    for (int mm = threadIdx.x; mm < nminor; mm += blockDim.x) {
      const int im = idx_minor[mm];
      float scaling = (im == 0 ? col_dry
          : vmr[cell * (ngas + 1) + im] * col_dry);
      if (density[mm]) {
        scaling *= 0.01f * p / t;
        const int iscale = idx_scaling[mm];
        if (iscale > 0) {
          const float special = vmr[cell * (ngas + 1) + iscale]
                                / (1.0f + h2o_vmr);
          scaling *= complement[mm] ? 1.0f - special : special;
        }
      }
      s_scaling[mm] = scaling;
    }
    __syncthreads();

    for (int gpt = threadIdx.x; gpt < ngpt; gpt += blockDim.x) {
      const int iflav = gpoint_flavor[iatm * ngpt + gpt];
      const int gas1 = flavor[iflav * 2];
      const int gas2 = flavor[iflav * 2 + 1];
      int je[2];
      float fm[2][2];
      float cmix[2];
      float tau_abs = 0.0f;

#pragma unroll
      for (int itemp = 0; itemp < 2; ++itemp) {
        const int jtr = jt + itemp;
        const float ref1 = vmr_ref[(iatm * (ngas + 1) + gas1) * ntemp + jtr];
        const float ref2 = vmr_ref[(iatm * (ngas + 1) + gas2) * ntemp + jtr];
        const float ratio = ref1 / ref2;
        const float col1 = (gas1 == 0) ? col_dry
            : vmr[cell * (ngas + 1) + gas1] * col_dry;
        const float col2 = (gas2 == 0) ? col_dry
            : vmr[cell * (ngas + 1) + gas2] * col_dry;
        cmix[itemp] = col1 + ratio * col2;
        const float eta = cmix[itemp] > 2.0f * 1.17549435e-38f
            ? col1 / cmix[itemp] : 0.5f;
        const float loceta = eta * (neta - 1);
        je[itemp] = min((int)loceta, neta - 2);
        const float feta = loceta - truncf(loceta);
        const float wt = itemp == 0 ? 1.0f - ftemp : ftemp;
        fm[0][itemp] = (1.0f - feta) * wt;
        fm[1][itemp] = feta * wt;

        const float w00 = (1.0f - fpress) * fm[0][itemp];
        const float w10 = (1.0f - fpress) * fm[1][itemp];
        const float w01 = fpress * fm[0][itemp];
        const float w11 = fpress * fm[1][itemp];
        tau_abs += cmix[itemp] * (
            w00 * kmajor[km_index(jtr, je[itemp],     jp,     gpt,
                                  neta, npressk, ngpt)]
          + w10 * kmajor[km_index(jtr, je[itemp] + 1, jp,     gpt,
                                  neta, npressk, ngpt)]
          + w01 * kmajor[km_index(jtr, je[itemp],     jp + 1, gpt,
                                  neta, npressk, ngpt)]
          + w11 * kmajor[km_index(jtr, je[itemp] + 1, jp + 1, gpt,
                                  neta, npressk, ngpt)]);
      }

      const int q_end = mstart[gpt + 1];
      for (int q = mstart[gpt]; q < q_end; ++q) {
        const int m = mlist[q];
        const int gs = limits[m * 2];
        const float scaling = s_scaling[m];
        const int kk = starts[m] + (gpt - gs);
        const float kval =
            fm[0][0] * kminor[k2_index(jt,     je[0],     kk, neta, nk)]
          + fm[1][0] * kminor[k2_index(jt,     je[0] + 1, kk, neta, nk)]
          + fm[0][1] * kminor[k2_index(jt + 1, je[1],     kk, neta, nk)]
          + fm[1][1] * kminor[k2_index(jt + 1, je[1] + 1, kk, neta, nk)];
        tau_abs += scaling * kval;
      }

      const int out = cell * ngpt + gpt;
      if (is_sw) {
        const float* kr = rayleigh + iatm * ntemp * neta * ngpt;
        const float ray_k =
            fm[0][0] * kr[(jt     * neta + je[0])     * ngpt + gpt]
          + fm[1][0] * kr[(jt     * neta + je[0] + 1) * ngpt + gpt]
          + fm[0][1] * kr[((jt+1) * neta + je[1])     * ngpt + gpt]
          + fm[1][1] * kr[((jt+1) * neta + je[1] + 1) * ngpt + gpt];
        const float tau_ray = ray_k * col_dry * (1.0f + h2o_vmr);
        const float total = tau_abs + tau_ray;
        tau[out] = total;
        ssa[out] = tau_ray / fmaxf(3.0f * 1.17549435e-38f, total);
      } else {
        tau[out] = tau_abs;
      }
    }
  }
}

__device__ __forceinline__ float planck_interp(
    float t, int band, const float* table, int nplanck, int nband,
    float temp_min, float delta) {
  const float val = (t - temp_min) / delta;
  int idx = (int)truncf(val);
  idx = max(0, min(nplanck - 2, idx));
  const float frac = val - truncf(val);
  const float lo = table[idx * nband + band];
  return lo + frac * (table[(idx + 1) * nband + band] - lo);
}

extern "C" __global__ void rrtmgp_planck_sources(
    const float* play, const float* tlay, const float* tlev,
    const float* tsfc, const float* vmr,
    const int* iatm_meta, const int* jt_meta, const int* jp_meta,
    const float* ftemp_meta, const float* fpress_meta,
    const float* temp_ref,
    const float* vmr_ref, const int* flavor, const int* gpoint_flavor,
    const int* gpoint_bands, const float* planck_fraction,
    const float* totplnk, float* lay_source, float* lev_source,
    float* sfc_source, int ncol, int nlay, int ngas, int ngpt,
    int ntemp, int npres, int neta, int nband, int nplanck) {
  // One thread per (column, g-point).  The g-point loop this replaces was
  // fully independent: `pfrac` is rebuilt from scratch inside every iteration
  // and no output slot is written by two g-points.  Keeping `pfrac` in the
  // thread (rather than staging it globally) is what makes this free -- the
  // level sources still read their adjacent layers out of local memory, and
  // consecutive threads now write consecutive g-points of every output.
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (nlay > 128) return;
  const int col = tid / ngpt;
  const int gpt = tid - col * ngpt;
  if (col >= ncol) return;
  const int nlev = nlay + 1;
  const int npressk = npres + 1;
  const float dplanck = (temp_ref[ntemp - 1] - temp_ref[0])
                        / (nplanck - 1);
  const int sfc_lay = play[col * nlay] < play[col * nlay + nlay - 1]
      ? nlay - 1 : 0;
  float pfrac[128];

  {
    const int band = gpoint_bands[gpt];
    for (int lay = 0; lay < nlay; ++lay) {
      const int cell = col * nlay + lay;
      const float t = tlay[cell];
      const int iatm = iatm_meta[cell];
      const int iflav = gpoint_flavor[iatm * ngpt + gpt];
      const int gas1 = flavor[iflav * 2], gas2 = flavor[iflav * 2 + 1];
      const int jt = jt_meta[cell];
      const int jp = jp_meta[cell];
      const float ft = ftemp_meta[cell];
      const float fp = fpress_meta[cell];
      float pf = 0.0f;
      for (int itemp = 0; itemp < 2; ++itemp) {
        const int jtr = jt + itemp;
        const float ratio =
            vmr_ref[(iatm * (ngas + 1) + gas1) * ntemp + jtr]
            / vmr_ref[(iatm * (ngas + 1) + gas2) * ntemp + jtr];
        const float amount1 = gas1 == 0 ? 1.0f
            : vmr[cell * (ngas + 1) + gas1];
        const float amount2 = gas2 == 0 ? 1.0f
            : vmr[cell * (ngas + 1) + gas2];
        const float mix = amount1 + ratio * amount2;
        const float eta = mix > 2.0f * 1.17549435e-38f
            ? amount1 / mix : 0.5f;
        const float loce = eta * (neta - 1);
        const int je = min((int)loce, neta - 2);
        const float fe = loce - truncf(loce);
        const float wt = itemp == 0 ? 1.0f - ft : ft;
        const float w00 = (1.0f - fp) * (1.0f - fe) * wt;
        const float w10 = (1.0f - fp) * fe * wt;
        const float w01 = fp * (1.0f - fe) * wt;
        const float w11 = fp * fe * wt;
        pf += w00 * planck_fraction[km_index(jtr, je,     jp,     gpt,
                                              neta, npressk, ngpt)]
            + w10 * planck_fraction[km_index(jtr, je + 1, jp,     gpt,
                                              neta, npressk, ngpt)]
            + w01 * planck_fraction[km_index(jtr, je,     jp + 1, gpt,
                                              neta, npressk, ngpt)]
            + w11 * planck_fraction[km_index(jtr, je + 1, jp + 1, gpt,
                                              neta, npressk, ngpt)];
      }
      pfrac[lay] = pf;
      lay_source[(cell * ngpt) + gpt] = pf * planck_interp(
          t, band, totplnk, nplanck, nband, temp_ref[0], dplanck);
    }
    sfc_source[col * ngpt + gpt] = pfrac[sfc_lay] * planck_interp(
        tsfc[col], band, totplnk, nplanck, nband, temp_ref[0], dplanck);
    lev_source[(col * nlev) * ngpt + gpt] = pfrac[0] * planck_interp(
        tlev[col * nlev], band, totplnk, nplanck, nband,
        temp_ref[0], dplanck);
    for (int lev = 1; lev < nlay; ++lev) {
      lev_source[(col * nlev + lev) * ngpt + gpt] =
          sqrtf(pfrac[lev - 1] * pfrac[lev]) * planck_interp(
              tlev[col * nlev + lev], band, totplnk, nplanck, nband,
              temp_ref[0], dplanck);
    }
    lev_source[(col * nlev + nlay) * ngpt + gpt] = pfrac[nlay - 1]
        * planck_interp(tlev[col * nlev + nlay], band, totplnk,
                        nplanck, nband, temp_ref[0], dplanck);
  }
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
  const int cell = blockDim.x * blockIdx.x + threadIdx.x;
  if (cell >= ncells) return;

  float* row = vmr + (long long)cell * nslot;
  for (int s = 0; s < nslot; ++s) row[s] = 0.0f;
  if (h2o_index >= 0) row[h2o_index] = qv[cell] * h2o_scale;
  // DOUBLE in, float out: cupy.interp promotes to float64 even for float32
  // tables, and the CuPy path narrowed it on the store into the float32
  // cube.  Narrowing here is that same store, not an extra rounding.
  if (o3_index >= 0) row[o3_index] = (float)o3[cell];
  for (int t = 0; t < ntrace; ++t) row[trace_index[t]] = trace_value[t];
}
