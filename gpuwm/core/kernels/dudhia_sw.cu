// gpuwm/core/kernels/dudhia_sw.cu
//
// Dudhia shortwave radiation (WRF ra_sw_physics = 1), WRF v4.6.1
// phys/module_ra_sw.F SWRAD/SWPARA, evaluated as the Fortran writes it.
// One thread per column; the K loop runs top-down inside the thread, which
// is SWPARA's own recurrence, so no per-layer work array is needed.
//
// Graded bitwise against the byte-unmodified WRF v4.6.1 routine by the
// column oracle tools/dudhia_sw_wrf461_oracle (tests/
// test_dudhia_sw_wrf461_parity.py).  What that takes, each found by the
// oracle (lane/verify-dudhia-sw):
//
//   * ALOG10 and REAL**REAL are the C library's log10f/powf in the WRF
//     build, not CUDA's: the base-10 logarithm below and WOOF's own
//     gfk_pow (glibc_flt32.cuh, prepended by the loader).
//   * The mu/water-path interpolation weights are XI-IIL, IU-XI, YJ-JJL and
//     JU-YJ with XI = frac + IIL and YJ = ALW + 1 rounded first; the four
//     products associate (TAB*wx)*wy and sum left to right, then divide by
//     the integer (IU-IIL)*(JU-JJL) = 1.  A 0..1 fraction used directly
//     rounds differently.
//   * Every multiply-add is rounded separately (the unit compiles
//     --fmad=false; gfortran on baseline x86-64 has no FMA).
//   * SWDOWN is the radiation driver's GSW/(1.-ALBEDO)
//     (module_radiation_driver.F:2877), not SDOWN(kte+1).
//   * RTHRATEN enters SWRAD zeroed; the stored tendency is 0. + TTEN/pi3D.
//
// Supported path: no WRF-Chem aerosol (pm2_5 absent, so XSCA's aerosol
// terms and DABSA/XABSA are exact zeros and are not evaluated), no eclipse
// (obscur = 0, SOLTOP = SOLCON), no slope radiation, every species flag set.
//
// Layout: element (column c, layer k counted from the TOP, k = 0..nlay-1)
// is at base + c*col_stride + k*lay_stride.  gpuwm's bottom-up
// (nz, ny, nx) state has col_stride = 1, lay_stride = -ny*nx and
// base = (nz-1)*ny*nx; top-down (ncol, nlay) columns have col_stride =
// nlay, lay_stride = 1, base = 0.

// WOOF's float32 base-10 logarithm on positive normal arguments (SWPARA's
// argument is WGM + 1 >= 1).  x = 2^k * m: k*log10(2) in two parts plus
// log(m)/ln(10) from WOOF's own gfk_log, combined in float32 with every
// operation separately rounded.  Graded exhaustively against the C
// library's log10f over every float32 in [1, +inf) by
// tools/dudhia_sw_wrf461_oracle/sweep_libm.py.
__device__ float dsw_log10f(float x)
{
    unsigned int hx = __float_as_uint(x);
    if ((hx & 0x7fffffffu) == 0u) return __int_as_float(0xff800000);  // -inf
    if (hx & 0x80000000u) return __int_as_float(0x7fc00000);          // nan
    if (hx >= 0x7f800000u) return FADD(x, x);                       // inf, nan
    int k = 0;
    if (hx < 0x00800000u) {                                         // x < 2^-126
        k = -25;
        hx = __float_as_uint(FMUL(x, 33554432.0f));                 // 2^25
    }
    k += (int)(hx >> 23) - 127;
    // Mantissa in [1,2) for k >= 0 and in [0.5,1) for k < 0.
    const int i = (int)(((unsigned int)k & 0x80000000u) >> 31);
    const float y = (float)(k + i);
    const float m = __uint_as_float((hx & 0x007fffffu)
                                    | ((unsigned int)(0x7f - i) << 23));
    const float z = FADD(FMUL(y, __uint_as_float(0x355427dbu)),  // log10(2) lo
                         FMUL(__uint_as_float(0x3ede5bd9u),      // 1/ln(10)
                              gfk_log(m)));
    return FADD(z, FMUL(y, __uint_as_float(0x3e9a2080u)));       // log10(2) hi
}

// module_ra_sw.F:298-308, ALBTAB(4,5) / ABSTAB(4,5) in Fortran storage
// order: element (IMU, JW) at (JW-1)*4 + (IMU-1).
__device__ const float DSW_ALBTAB[20] = {
    0.f, 0.f, 0.f, 0.f,   69.f, 58.f, 40.f, 15.f,   90.f, 80.f, 70.f, 60.f,
    94.f, 90.f, 82.f, 78.f,   96.f, 92.f, 85.f, 80.f};
__device__ const float DSW_ABSTAB[20] = {
    0.f, 0.f, 0.f, 0.f,   0.f, 2.5f, 4.f, 5.f,   0.f, 2.6f, 7.f, 10.f,
    0.f, 3.3f, 10.f, 14.f,   0.f, 3.7f, 10.f, 15.f};
__device__ const float DSW_XMUVAL[4] = {0.f, 0.2f, 0.5f, 1.0f};

// Bilinear table read exactly as SWPARA writes ALBA / ABSC (Fortran
// one-based indices iil, iu, jjl, ju).
__device__ __forceinline__ float dsw_interp(const float* tab, int iil, int iu,
                                            int jjl, int ju, float xi,
                                            float yj)
{
    const float a = FSUB(xi, (float)iil);   // XI-IIL
    const float b = FSUB((float)iu, xi);    // IU-XI
    const float c = FSUB(yj, (float)jjl);   // YJ-JJL
    const float d = FSUB((float)ju, yj);    // JU-YJ
    float s = FMUL(FMUL(tab[(ju - 1) * 4 + (iu - 1)], a), c);
    s = FADD(s, FMUL(FMUL(tab[(ju - 1) * 4 + (iil - 1)], b), c));
    s = FADD(s, FMUL(FMUL(tab[(jjl - 1) * 4 + (iu - 1)], a), d));
    s = FADD(s, FMUL(FMUL(tab[(jjl - 1) * 4 + (iil - 1)], b), d));
    return FDIV(s, (float)((iu - iil) * (ju - jjl)));
}

extern "C" __global__
void dudhia_sw_column(const float* __restrict__ t,      // temperature (K)
                      const float* __restrict__ p,      // pressure (Pa)
                      const float* __restrict__ qv,
                      const float* __restrict__ qc,
                      const float* __restrict__ qr,
                      const float* __restrict__ qi,
                      const float* __restrict__ qs,
                      const float* __restrict__ qg,
                      const float* __restrict__ dz,     // layer depth (m)
                      const float* __restrict__ exner,  // pi3D (or unused)
                      const float* __restrict__ coszen, // (ncol)
                      const float* __restrict__ albedo, // (ncol)
                      float* __restrict__ heating,      // layer layout
                      float* __restrict__ gsw,          // (ncol)
                      float* __restrict__ swdown,       // (ncol)
                      float soltop, float cssca, int icloud, int use_exner,
                      int ncol, int nlay, long long col_stride,
                      long long lay_stride, long long base)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncol) return;
    const float r = 287.0f;      // module_model_constants r_d
    const float cp = 1004.5f;    // module_model_constants cp = 7*r_d/2
    const float xmu = coszen[col];
    const float alb = albedo[col];
    const long long c0 = base + (long long)col * col_stride;

    // SWPARA: RETURN IF NIGHT (and the eclipse test, SOLTOP = SOLCON).
    // TTEN stays as SWRAD zeroed it and GSW = 0.
    if (xmu <= 1.0e-9f || soltop <= 1.0e-9f) {
        for (int k = 0; k < nlay; ++k) {
            const long long o = c0 + (long long)k * lay_stride;
            const float tten = 0.0f;
            heating[o] = FADD(0.0f, use_exner ? FDIV(tten, exner[o]) : tten);
        }
        const float g = 0.0f;
        gsw[col] = g;
        swdown[col] = FDIV(g, FSUB(1.0f, alb));
        return;
    }

    const float sdown1 = FMUL(soltop, xmu);   // SDOWN(1)
    float sdown = sdown1;                     // SDOWN(K)
    float ww = 0.0f, uv = 0.0f, oldalb = 0.0f, oldabc = 0.0f;
    float totabs = 0.0f, dsca = 0.0f, dabs = 0.0f, dscld = 0.0f;
    for (int k = 0; k < nlay; ++k) {
        const long long o = c0 + (long long)k * lay_stride;
        const float tk = t[o];
        const float dzk = dz[o];
        // SWRAD's copies: QX1D = max(0., QX3D).
        const float qvk = fmaxf(0.0f, qv[o]);
        // RO(K)=P(K)/(R*T(K)); XWVP; XATP; XLWP.
        const float ro = FDIV(p[o], FMUL(r, tk));
        const float xwvp = FMUL(FMUL(FMUL(ro, qvk), dzk), 1000.0f);
        const float xatp = FMUL(ro, dzk);
        float xlwp = 0.0f;
        if (icloud != 0) {
            float q = FADD(fmaxf(0.0f, qc[o]),
                           FMUL(0.1f, fmaxf(0.0f, qi[o])));
            q = FADD(q, FMUL(0.05f, fmaxf(0.0f, qr[o])));
            q = FADD(q, FMUL(0.02f, fmaxf(0.0f, qs[o])));
            q = FADD(q, FMUL(0.05f, fmaxf(0.0f, qg[o])));
            xlwp = FMUL(FMUL(FMUL(ro, 1000.0f), dzk), q);
        }

        ww = FADD(ww, xlwp);
        uv = FADD(uv, xwvp);
        const float wgm = FDIV(ww, xmu);
        const float ugcm = FDIV(FMUL(uv, 0.0001f), xmu);
        const float oldabs = totabs;
        // Lacis and Hansen (1974) water vapour absorption.
        totabs = FDIV(FMUL(2.9f, ugcm),
                      FADD(gfk_pow(FADD(1.0f, FMUL(141.5f, ugcm)), 0.635f),
                           FMUL(5.925f, ugcm)));
        // XSCA = (cssca*XATP + beta*0*bext*DZ + beta*0*bext*DZ)/XMU: the
        // aerosol products are exact +0 (beta > 0), adding them is exact.
        float xsca = FDIV(FMUL(cssca, xatp), xmu);
        // XABS = (TOTABS-OLDABS)*(SDOWN(1)-DSCLD-DSCA-DABSA)/SDOWN(K),
        // DABSA = 0.
        float xabs = FDIV(FMUL(FSUB(totabs, oldabs),
                               FSUB(FSUB(sdown1, dscld), dsca)), sdown);
        if (xabs < 0.0f) xabs = 0.0f;

        float alw = dsw_log10f(FADD(wgm, 1.0f));
        if (alw > 3.999f) alw = 3.999f;

        int iil = 1, iu = 2;
        float xi = 0.0f;
        for (int ii = 1; ii <= 3; ++ii) {
            if (xmu > DSW_XMUVAL[ii - 1]) {
                iil = ii;
                iu = ii + 1;
                xi = FADD(FDIV(FSUB(xmu, DSW_XMUVAL[ii - 1]),
                               FSUB(DSW_XMUVAL[ii], DSW_XMUVAL[ii - 1])),
                          (float)iil);
            }
        }
        const int jjl = (int)alw + 1;   // IFIX(ALW)+1, ALW >= 0
        const int ju = jjl + 1;
        const float yj = FADD(alw, 1.0f);
        const float alba = dsw_interp(DSW_ALBTAB, iil, iu, jjl, ju, xi, yj);
        const float absc = dsw_interp(DSW_ABSTAB, iil, iu, jjl, ju, xi, yj);

        const float avail = FSUB(FSUB(sdown1, dsca), dabs);
        float xalb = FDIV(FMUL(FSUB(alba, oldalb), avail), sdown);
        float xabsc = FDIV(FMUL(FSUB(absc, oldabc), avail), sdown);
        if (xalb < 0.0f) xalb = 0.0f;
        if (xabsc < 0.0f) xabsc = 0.0f;
        dscld = FADD(dscld, FMUL(FMUL(FADD(xalb, xabsc), sdown), 0.01f));
        dsca = FADD(dsca, FMUL(xsca, sdown));
        dabs = FADD(dabs, FMUL(xabs, sdown));
        oldalb = alba;
        oldabc = absc;

        float trans0 = FSUB(FSUB(FSUB(FSUB(100.0f, xalb), xabsc),
                                 FMUL(xabs, 100.0f)), FMUL(xsca, 100.0f));
        if (trans0 < 1.0f) {
            const float ff = FDIV(99.0f,
                                  FADD(FADD(FADD(xalb, xabsc),
                                            FMUL(xabs, 100.0f)),
                                       FMUL(xsca, 100.0f)));
            xalb = FMUL(xalb, ff);
            xabsc = FMUL(xabsc, ff);
            xabs = FMUL(xabs, ff);
            xsca = FMUL(xsca, ff);
            trans0 = 1.0f;
        }
        // TTEN(K)=SDOWN(K)*(XABSC+XABS*100.+XABSA*100.)*0.01/(RO*CP*DZ),
        // XABSA = 0: the + 0. is kept (it turns a -0 sum into +0).
        const float absorbed = FADD(FADD(xabsc, FMUL(xabs, 100.0f)),
                                    FMUL(0.0f, 100.0f));
        const float tten = FDIV(FMUL(FMUL(sdown, absorbed), 0.01f),
                                FMUL(FMUL(ro, cp), dzk));
        heating[o] = FADD(0.0f, use_exner ? FDIV(tten, exner[o]) : tten);
        // SDOWN(K+1)=AMAX1(1.E-9,SDOWN(K)*TRANS0*0.01)
        sdown = fmaxf(1.0e-9f, FMUL(FMUL(sdown, trans0), 0.01f));
    }
    // GSW=(1.-ALBEDO)*SDOWN(kte+1); the driver's SWDOWN=GSW/(1.-ALBEDO).
    const float g = FMUL(FSUB(1.0f, alb), sdown);
    gsw[col] = g;
    swdown[col] = FDIV(g, FSUB(1.0f, alb));
}
