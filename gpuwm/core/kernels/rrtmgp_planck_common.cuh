// RTE+RRTMGP device words shared by rrtmgp_gas.cu and rrtmgp_rte.cu.
//
// Transcribed from earth-system-radiation/rte-rrtmgp fa107a1:
//   rte/kernels/mo_gas_optics_utils.F90           get_layer_number (col_dry)
//   rrtmgp/kernels/mo_gas_optics_rrtmgp_kernels.F90
//       interpolation (eta, fminor, fmajor), compute_Planck_source,
//       interpolate1D, interpolate3D_byflav
//
// Both units compile with --fmad=false (gpuwm.core.kernels._NO_FMAD_MODULES):
// every expression below is one IEEE binary32 rounding per operator, in the
// Fortran's own association (Fortran evaluates a*b*c as (a*b)*c and a+b+c as
// (a+b)+c, as C does), which is what the upstream single-precision build
// computes (tools/rrtmgp_rte_upstream_oracle, graded at 0 ULP).  The
// standalone Planck kernel (rrtmgp_gas.cu) and the LW solver's in-kernel
// Planck derivation (rrtmgp_rte.cu) call the SAME functions, so the two are
// bitwise one computation by construction, not by pinning each other's SASS.

#define RRTMGP_TINY 1.17549435e-38f

__device__ __forceinline__ int rrtmgp_km_index(
    int jt, int je, int jp, int g, int neta, int npressk, int ngpt) {
  return (((jt * neta + je) * npressk + jp) * ngpt + g);
}

// Dry-air molecules per cm^2 of one layer: get_layer_number, operator for
// operator.  plev_a/plev_b are the layer's two interfaces (either order).
__device__ __forceinline__ float rrtmgp_col_dry(
    float plev_a, float plev_b, float vmr_h2o) {
  const float avogad = 6.02214076e23f;
  const float m_dry = 0.028964f;
  const float m_h2o = 0.018016f;
  const float grav = 9.80665f;
  const float delta_plev = fabsf(plev_a - plev_b);
  const float fact = 1.0f / (1.0f + vmr_h2o);
  const float m_air = (m_dry + m_h2o * vmr_h2o) * fact;
  return 10.0f * delta_plev * avogad * fact
         / (1000.0f * m_air * 100.0f * grav);
}

// One (flavor, reference temperature) slice of `interpolation`: the column
// mix, the eta index and the two fminor weights.  itemp is 0 or 1.
//
// One deliberate divergence from fa107a1, at eta == 1 exactly (the second
// key gas absent, e.g. a layer with zero water vapour): the Fortran clamps
// the index (jeta = min(int(loceta)+1, neta-1)) but keeps the UNCLAMPED
// fraction (feta = loceta - aint(loceta) = 0), so it returns the table at
// eta = (neta-2)/(neta-1) instead of eta = 1.  Here the fraction is measured
// from the node actually read, which gives the eta = 1 entry.  Everywhere
// else (loceta < neta-1) the two are the same operations on the same
// values.  tests/test_rrtmgp_upstream_oracle.py pins both behaviours.
__device__ __forceinline__ void rrtmgp_eta_weights(
    float col1, float col2, float ratio, float ftemp, int itemp, int neta,
    float* col_mix, int* jeta, float* fminor0, float* fminor1) {
  const float mix = col1 + ratio * col2;
  const float eta = mix > 2.0f * RRTMGP_TINY ? col1 / mix : 0.5f;
  const float loceta = eta * (float)(neta - 1);
  const int je = min((int)loceta, neta - 2);
  const float feta = loceta - (float)je;
  const float ftemp_term = itemp == 0 ? 1.0f - ftemp : ftemp;
  *col_mix = mix;
  *jeta = je;
  *fminor0 = (1.0f - feta) * ftemp_term;
  *fminor1 = feta * ftemp_term;
}

// interpolate1D on the band's Planck table: the Planck function of band
// `band` at temperature t.
//
// Divergence from fa107a1 at the table's top temperature exactly (t equal to
// temp_ref_max, 355 K for v1.9): the Fortran clamps the index but keeps
// frac = val0 - aint(val0) = 0, which returns the Planck function one table
// step (1 K) below t.  The fraction here is measured from the node read, so
// t = 355 K gets the 355 K entry.  Below the top the two are identical.
__device__ __forceinline__ float rrtmgp_planck_function(
    float t, int band, const float* table, int nplanck, int nband,
    float temp_ref_min, float delta_r) {
  const float val0 = (t - temp_ref_min) * delta_r;
  const int index = max(0, min(nplanck - 2, (int)val0));
  const float frac = val0 - (float)index;
  const float lo = table[index * nband + band];
  return lo + frac * (table[(index + 1) * nband + band] - lo);
}

// The fraction of band `band`'s Planck function carried by g-point gpt in
// layer (col, lay): compute_Planck_source's interpolate3D_byflav with
// scaling = one, on the fmajor weights `interpolation` builds from the gas
// COLUMNS (vmr * col_dry), exactly as the gas optics do.
__device__ __forceinline__ float rrtmgp_pfrac_at(
    int col, int lay, int gpt, int nlay, int ngas, int ngpt, int ntemp,
    int npres, int neta, int idx_h2o,
    const float* plev, const float* vmr, const int* iatm_meta,
    const int* jt_meta, const int* jp_meta, const float* ftemp_meta,
    const float* fpress_meta, const float* vmr_ref, const int* flavor,
    const int* gpoint_flavor, const float* planck_fraction) {
  const int npressk = npres + 1;
  const int cell = col * nlay + lay;
  const float* v = vmr + cell * (ngas + 1);
  const float col_dry = rrtmgp_col_dry(
      plev[col * (nlay + 1) + lay], plev[col * (nlay + 1) + lay + 1],
      v[idx_h2o]);
  const int iatm = iatm_meta[cell];
  const int iflav = gpoint_flavor[iatm * ngpt + gpt];
  const int gas1 = flavor[iflav * 2], gas2 = flavor[iflav * 2 + 1];
  const float col1 = gas1 == 0 ? col_dry : v[gas1] * col_dry;
  const float col2 = gas2 == 0 ? col_dry : v[gas2] * col_dry;
  const int jt = jt_meta[cell];
  const int jp = jp_meta[cell];
  const float ft = ftemp_meta[cell];
  const float fp = fpress_meta[cell];
  float part[2];
  for (int itemp = 0; itemp < 2; ++itemp) {
    const int jtr = jt + itemp;
    const float ratio = vmr_ref[(iatm * (ngas + 1) + gas1) * ntemp + jtr]
                        / vmr_ref[(iatm * (ngas + 1) + gas2) * ntemp + jtr];
    float mix, f0, f1;
    int je;
    rrtmgp_eta_weights(col1, col2, ratio, ft, itemp, neta, &mix, &je,
                       &f0, &f1);
    const float f11 = (1.0f - fp) * f0;
    const float f21 = (1.0f - fp) * f1;
    const float f12 = fp * f0;
    const float f22 = fp * f1;
    part[itemp] =
        f11 * planck_fraction[rrtmgp_km_index(jtr, je, jp, gpt,
                                              neta, npressk, ngpt)]
      + f21 * planck_fraction[rrtmgp_km_index(jtr, je + 1, jp, gpt,
                                              neta, npressk, ngpt)]
      + f12 * planck_fraction[rrtmgp_km_index(jtr, je, jp + 1, gpt,
                                              neta, npressk, ngpt)]
      + f22 * planck_fraction[rrtmgp_km_index(jtr, je + 1, jp + 1, gpt,
                                              neta, npressk, ngpt)];
  }
  // scaling(1) * part1 + scaling(2) * part2 with scaling = (1, 1).
  return part[0] + part[1];
}
