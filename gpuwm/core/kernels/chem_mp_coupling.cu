// Aerosol-aware Thompson coupling: nwfa / nifa diagnosed from the GOCART
// species rows, the public NOAA GSL formula.
//
// SOURCE.  ufs-community/ccpp-physics 3e6660c6df54e95a0871e990c2294dd397ae3860,
// physics/MP/Thompson/mp_thompson.F90:1025-1068, `get_niwfa` (Apache-2.0):
//
//   nifa = (a1/4.0737762 + a2/30.459203 + a3/153.45048 + a4/1011.5142
//           + a5/5683.3501)*1.e15
//   nwfa = ((a6/0.0045435214 + a7/0.2907854 + a8/12.91224 + a9/206.2216
//            + a10/4326.23)*9. + a11/0.3053104*5 + a15/0.3232698*8)*1.e15
//
// with aerfld in kg/kg and REAL(kind_phys) = REAL(8).  The denominators and
// 1.e15 are single-precision literals, so each meets the float64 operand as
// the float32 value widened; the group factors 9., 5 and 8 are exact.
//
// TABLE FORM.  No row is named here.  The host sorts the rows that take part
// by (target, group, order) and hands each row its kg/kg conversion, its
// widened denominator and its group's factor.  Per cell and per target the
// kernel forms, in that order,
//
//     acc = 0;  for each group g:  gs = 0; gs = gs + a_r/d_r (rows of g);
//                                  acc = acc + gs*f_g
//     out = acc*scale
//
// which is the Fortran expression word for word: 0 + x is x for the
// non-negative terms the transport's positive-definite update leaves, and a
// single-group target (nifa) takes f = 1, x*1 being x.  Every operation is an
// explicit round-to-nearest intrinsic so NVRTC cannot contract a*b + c.  The
// float64 result is rounded once to the float32 the engine stores.
//
// One thread per cell (the formula has no vertical dependence).

extern "C" __global__ void chem_mp_niwfa(
    const unsigned long long* __restrict__ row_ptr,  // float32 species fields
    const double* __restrict__ row_conv,   // native units -> kg/kg
    const double* __restrict__ row_den,    // widened float32 denominator
    const int* __restrict__ row_target,    // 0 = nifa, 1 = nwfa
    const int* __restrict__ row_group,     // group id within the target
    const double* __restrict__ row_gfac,   // the row's group factor
    const int nrows,
    const double scale,                    // widened float32 1.e15
    float* __restrict__ nifa,
    float* __restrict__ nwfa,
    double* __restrict__ nifa_f64,         // optional (nullptr): pre-rounding value
    double* __restrict__ nwfa_f64,
    const long long ncell)
{
    const long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncell) return;
    for (int t = 0; t < 2; ++t) {
        float* out = (t == 0) ? nifa : nwfa;
        double* out64 = (t == 0) ? nifa_f64 : nwfa_f64;
        if (out == nullptr) continue;
        double acc = 0.0, gs = 0.0, gf = 1.0;
        int g = -1;
        bool any = false;
        for (int r = 0; r < nrows; ++r) {
            if (row_target[r] != t) continue;
            if (row_group[r] != g) {
                if (g >= 0) acc = __dadd_rn(acc, __dmul_rn(gs, gf));
                g = row_group[r];
                gs = 0.0;
                gf = row_gfac[r];
            }
            const float* q = reinterpret_cast<const float*>(row_ptr[r]);
            const double a = __dmul_rn((double)q[c], row_conv[r]);
            gs = __dadd_rn(gs, __ddiv_rn(a, row_den[r]));
            any = true;
        }
        if (!any) continue;  // no row targets this field: leave it alone
        acc = __dadd_rn(acc, __dmul_rn(gs, gf));
        const double v = __dmul_rn(acc, scale);
        out[c] = __double2float_rn(v);
        if (out64 != nullptr) out64[c] = v;
    }
}
