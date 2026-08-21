// Planck-source evaluation for the LW solver's in-kernel derivation.
//
// `rrtmgp_lw_noscat` derives `lay_source`/`lev_source`/`sfc_source`
// itself rather than loading what `rrtmgp_planck_sources` wrote, which
// deletes all three from the workspace (455 MiB of the binding phase at
// the default chunk) and the round trip an ablation priced at 38% of the
// solver.
//
// EVERY FP OPERATION HERE IS PINNED (HOWTO 13.6j).  The values must be
// bitwise those of `rrtmgp_planck_sources` as compiled in rrtmgp_gas.cu
// at ITS register count -- and contraction is ptxas's choice per
// compilation unless stated.  Each intrinsic below states what that
// kernel's SASS does, operand for operand (nvdisasm -gi on the reference
// build): the mix term and the interp lerp are FFMAs; each weight is the
// pair product then the wt product; the accumulation is one FMUL and
// three chained FFMAs per temperature, added to pf with a plain FADD;
// divisions are IEEE rn.  This header is deliberately NOT shared with
// rrtmgp_gas.cu -- that file keeps its own helpers verbatim so its
// assembled source, PTX and register allocation stay byte-identical.
// tests/test_rrtmgp.py::test_in_solver_planck_matches_the_kernel holds
// the two implementations together bitwise.

__device__ __forceinline__ int rrtmgp_pk_km_index(
    int jt, int je, int jp, int g, int neta, int npressk, int ngpt) {
  return (((jt * neta + je) * npressk + jp) * ngpt + g);
}

__device__ __forceinline__ float rrtmgp_pk_interp(
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

__device__ __forceinline__ float rrtmgp_pk_fraction_at(
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
    const float mix = __fmaf_rn(ratio, amount2, amount1);
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
