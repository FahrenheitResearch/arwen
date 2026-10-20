// Planck-source evaluation for the paired LW solver, rounded the way a
// DEFAULT process's standalone `rrtmgp_planck_sources` rounds
// (rrtmgp_gas_default.cu, compiled WITH contraction).
//
// rrtmgp_planck_common_default.cuh pins the strict unit's arithmetic; in a
// default process the standalone kernel contracts its pfrac accumulation
// differently, so in-solver Planck through that header does not give the
// staged words there (test_in_solver_planck_matches_the_kernel runs only in
// a strict process for that reason).  These functions state the default
// unit's operations, intrinsic for intrinsic.  The one difference from that
// header: NVVM leaves the mix term as a separate multiply and add (mul.f32,
// add.f32 in the PTX) and ptxas does not contract them, so mix is
// amount1 + rn(ratio * amount2), not an FMA.  Matched against the standalone
// kernel bitwise on 61,440,000 lay_source words (default process, sm_120).
// rrtmgp_lw_noscat_pair uses them when pk_on == 2; the workspace route's
// pk_on == 1 keeps the header above.

__device__ __forceinline__ float rrtmgp_pks_interp(
    float t, int band, const float* table, int nplanck, int nband,
    float temp_min, float delta) {
  const float val = __fdiv_rn(__fsub_rn(t, temp_min), delta);
  int idx = (int)truncf(val);
  idx = max(0, min(nplanck - 2, idx));
  const float frac = __fsub_rn(val, truncf(val));
  const float lo = table[idx * nband + band];
  return __fmaf_rn(frac, __fsub_rn(table[(idx + 1) * nband + band], lo),
                   lo);
}

__device__ __forceinline__ float rrtmgp_pks_fraction_at(
    int col, int lay, int gpt, int nlay, int ngas, int ngpt, int ntemp,
    int npres, int neta,
    const float* vmr, const int* iatm_meta, const int* jt_meta,
    const int* jp_meta, const float* ftemp_meta, const float* fpress_meta,
    const float* vmr_ref, const int* flavor, const int* gpoint_flavor,
    const float* planck_fraction) {
  const int npressk = npres + 1;
  const int cell = col * nlay + lay;
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
    const float ratio = __fdiv_rn(
        vmr_ref[(iatm * (ngas + 1) + gas1) * ntemp + jtr],
        vmr_ref[(iatm * (ngas + 1) + gas2) * ntemp + jtr]);
    const float amount1 = gas1 == 0 ? 1.0f
        : vmr[cell * (ngas + 1) + gas1];
    const float amount2 = gas2 == 0 ? 1.0f
        : vmr[cell * (ngas + 1) + gas2];
    const float mix = __fadd_rn(amount1, __fmul_rn(ratio, amount2));
    const float eta = mix > 2.0f * 1.17549435e-38f
        ? __fdiv_rn(amount1, mix) : 0.5f;
    const float loce = __fmul_rn(eta, (float)(neta - 1));
    const int je = min((int)loce, neta - 2);
    const float fe = __fsub_rn(loce, truncf(loce));
    const float wt = itemp == 0 ? __fsub_rn(1.0f, ft) : ft;
    const float omfp = __fsub_rn(1.0f, fp);
    const float omfe = __fsub_rn(1.0f, fe);
    const float w00 = __fmul_rn(__fmul_rn(omfp, omfe), wt);
    const float w10 = __fmul_rn(__fmul_rn(omfp, fe), wt);
    const float w01 = __fmul_rn(__fmul_rn(fp, omfe), wt);
    const float w11 = __fmul_rn(__fmul_rn(fp, fe), wt);
    float t = __fmul_rn(
        w00, planck_fraction[rrtmgp_pk_km_index(jtr, je,     jp,     gpt,
                                                 neta, npressk, ngpt)]);
    t = __fmaf_rn(
        w10, planck_fraction[rrtmgp_pk_km_index(jtr, je + 1, jp,     gpt,
                                                 neta, npressk, ngpt)], t);
    t = __fmaf_rn(
        w01, planck_fraction[rrtmgp_pk_km_index(jtr, je,     jp + 1, gpt,
                                                 neta, npressk, ngpt)], t);
    t = __fmaf_rn(
        w11, planck_fraction[rrtmgp_pk_km_index(jtr, je + 1, jp + 1, gpt,
                                                 neta, npressk, ngpt)], t);
    pf = __fadd_rn(pf, t);
  }
  return pf;
}
