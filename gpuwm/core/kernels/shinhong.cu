// Shin-Hong scale-aware PBL column scheme (bl_pbl_physics=11), WRF v4.6.1.
//
// Transcribed statement for statement from the float32 CPU authority
// gpuwm/verify/shinhong_ref.py::np_shinhong_column, which is itself bitwise
// (max_ulp 0, both arms) against the byte-frozen WRF v4.6.1
// phys/module_bl_shinhong.F (sha256 99f44dbe..., pinned in
// gpuwm/data/shinhong/oracle/oracle-sha256sums.txt).  The :NNNN line numbers
// below are that Fortran file's; they are carried over from the authority so
// every statement here can be walked back to the module it mirrors.  One CUDA
// thread owns one complete surface-to-top column; element k (0-based) of
// column col is [k*st + col] with st = ny*nx; the working arrays are local,
// indexed 1-based like the Fortran, and bounded by SHINHONG_KMAX.
//
// ARM A ONLY.  The oracle records two arms; this kernel implements arm A
// (ctopo = ctopo2 = 1), the only arm WRF's Registry default (topo_wind=0) and
// this engine can reach.  With ctopo == 1 the momentum surface-drag diagonal
// (:1387) multiplies by exactly 1.0f, and with ctopo2 == 1 the u10/v10 blend
// (:1472-1473) returns its input word for word -- both are bitwise identities,
// proved on the CPU gate by
// tests/test_shinhong_wrf461_parity.py::test_the_topo_wind_arm_is_real_and_this_far_from_arm_a
// (which also pins how far arm B actually sits from arm A: ~1.9e9 ULP on the
// momentum tendencies, because they cross zero between the arms).  The ctopo
// multiply and the u10/v10 blend are therefore omitted here, and the kernel
// has no u10/v10 outputs at all.
//
// Two deliberate divergences carried over from the authority, both
// defined-behaviour guards against reads WRF itself never defines:
//   * efxpbl (:1557) reads q2xk(kpbl+1), one past the end when kpbl == kte;
//     guarded here (kpbl < kte), same as the authority.
//   * zfacent is only assigned below kpbl (:1006) but the TKE staging loop
//     (:1550-1552) reads it at every level; WRF reads stack garbage there and
//     never uses the result.  This port keeps those lanes at 0.0f.
//
// libm and pow spellings (pow-probe.txt, measured on the oracle toolchain):
//   * x**2. and x**pfac(2.0) fold to x*x -- spelled as multiplies;
//   * every other real-exponent power stays powf.  Device powf is NOT glibc's
//     correctly rounded powf, so those sites carry ULPs into the pinned
//     tables in tests/test_shinhong_wrf461_parity.py rather than being hidden
//     by a tolerance.  Same for expf and tanhf, and for the FMA contraction
//     nvrtc's default --fmad=true performs (measured in isolation on the
//     partition probe: pq has no transcendental at all and still sits 2 ULP
//     from glibc through pure contraction).
//   * the four x**(1/3) sites were measured BOTH ways against the oracle
//     (powf(x, h1) vs cbrtf(x), the -Ofast spelling; ysu.cu precedent), one
//     site at a time on the full fixture:
//       wstar  :750  cbrtf moved wstar differing lanes 24 -> 60   -> powf
//       wscale :757  cbrtf within noise (du 414 -> 416 lanes)     -> powf
//       wscalek:945  cbrtf closed dv's worst column 1491308 -> 46603 ULP and
//                    exch_h 9 -> 8, at the cost of tke 2005 -> 2022 -> cbrtf
//       ckp   :1980  cbrtf same el max (14), more lanes 1840 -> 1903 -> powf
//     Each site keeps whichever spelling measured closer; the one cbrtf site
//     is marked at its line below.
//   * powf(x, 0.0f) == 1.0f for every x on this device, including +/-0 and
//     subnormal x (device probe, 2026-08-03); no special case needed, and
//     the zfac**(pfac_q-pfac) multiply at :968 is kept (zfac is clamped to
//     [zfmin, 1] anyway, so the base is never 0 there).
//   * sqrtf is correctly rounded on both sides and carries nothing.
//
// FTZ/DAZ: CuPy appends -ftz=true unconditionally, and sm_120 additionally
// flushes FP32 subnormals in ALL arithmetic (DAZ) with --ftz=false
// ineffective.  The fixture's designed probes and what they measured:
//   * br = +1.4e-45 (case 7) on the `br .gt. 0` sfcflg compare (:706): the
//     one branch DAZ actually flips.  Countermeasure and measurements at the
//     compare itself, below (sh_f2d).
//   * subnormal ust (case 13): ust**3. underflows to +0 in float32 on BOTH
//     sides, so WRF's own prfac2 0/0 (:1010) fires here too and the NaN heat
//     column reproduces -- with a different NaN payload than the CSV can
//     carry, so the gate pins NaN GEOMETRY, never NaN bits.
//   * subnormal qfx (case 13), subnormal/signed-zero moisture (case 24) and
//     subnormal u10 (case 11): no branch moves; the columns sit inside the
//     ordinary arithmetic table (measured per-case worst: case 11 at 2 ULP,
//     case 13 at 1 ULP outside its NaN lanes, case 24 in the dqv family).
//     What -ftz DOES reach is the OUTPUT side: every lane where WRF wrote a
//     subnormal tendency, the kernel writes exactly zero (dqv 30 / dqc 24 /
//     dqi 18 lanes, pinned both ways by the gate's subnormal test).
//
// min/max are NumPy-semantics (NaN propagates), NOT fmaxf/fminf: CUDA's
// fmaxf(NaN, x) returns x, which would erase the case-13 NaN geometry at
// the amax1/amin1 sites (:2118 q2 floor, :1032-1034 xkz clamps).  gfortran
// -O0 max/min propagate the NaN on the lanes the fixture reaches, and the
// authority transcribes them as np.maximum/np.minimum for exactly that
// reason.

#define SHINHONG_KMAX 128
#define SHINHONG_K2 (SHINHONG_KMAX + 2)

// 1-based access to a (nz, ny, nx) global field: SH_G(a, 1) is the surface.
#define SH_G(a, k) a[(size_t)((k) - 1) * st + col]
// The scheme's own per-level names, as the authority spells them.
#define SH_UX(k)    SH_G(u, k)
#define SH_VX(k)    SH_G(v, k)
#define SH_THX(k)   SH_G(theta, k)
// Temperature: the engine hands the kernel theta and exner, so tx is formed
// as theta*exner where the scheme reads temperature (same convention as
// ysu.cu; the parity loader reconstructs theta = t/pi2d so thx is bitwise
// WRF's, and the (t/pi)*pi round-trip cost at the tx sites lands in the
// pinned tables).
#define SH_TX(k)    (SH_G(theta, k) * SH_G(exner, k))
#define SH_QVX(k)   SH_G(qv, k)
#define SH_QCX(k)   SH_G(qc, k)
#define SH_QIX(k)   SH_G(qi, k)
#define SH_P2DX(k)  SH_G(p, k)
#define SH_PI2DX(k) SH_G(exner, k)
#define SH_DZ8W(k)  SH_G(dz, k)
#define SH_TKEX(k)  SH_G(tke, k)

// NumPy-semantics min/max: NaN in, NaN out (see header).  For non-NaN input
// these are the plain ternaries.
__device__ __forceinline__ real sh_max(real a, real b) {
    if (a != a) return a;
    if (b != b) return b;
    return a > b ? a : b;
}
__device__ __forceinline__ real sh_min(real a, real b) {
    if (a != a) return a;
    if (b != b) return b;
    return a < b ? a : b;
}

// Subnormal-preserving float -> double, from rrtmg_sw.cu's rsw_f2d: on this
// compile route the cvt.f64.f32 the compiler emits DAZes a subnormal input
// (a plain `(double)brv > 0.0` measured identical to the float32 compare on
// case 7), so the subnormal is decoded from its bits instead.
__device__ double sh_f2d(real x) {
    unsigned int ix = __float_as_uint(x);
    if (((ix >> 23) & 0xffu) == 0u) {     // zero or subnormal
        double v = (double)(ix & 0x7fffffu) * 0x1p-149;
        return (ix >> 31) ? -v : v;
    }
    return (double)x;                     // normal / inf / nan
}

// ------------------------------------------------------------------------
// The five partition functions (:2297-2427).  (doh)**b1 with b1 = 2.0 folds
// to a multiply (pow-probe.txt); the fractional b2 powers stay powf.  The
// h != 0 early-out is exact -- both signed zeros of h return 1.0 -- and IS
// reachable from the scheme (cslen of a non-convective column).
// ------------------------------------------------------------------------

__device__ real sh_pu(real d, real h) {
    // pu (:2297-2319): nonlocal/local momentum-transport partition.
    const real a1 = 1.0f, a2 = 0.070f, a3 = 1.0f, a4 = 0.142f, a5 = 0.071f;
    const real b2 = 0.6666667f;
    real pu;
    if (h != 0.0f) {
        real doh = d / h;
        real t2 = doh * doh;
        real tb = powf(doh, b2);
        real num = a1 * t2 + a2 * tb;
        real den = a3 * t2 + a4 * tb + a5;
        pu = num / den;
    } else {
        pu = 1.0f;
    }
    pu = sh_max(pu, 0.0f);
    pu = sh_min(pu, 1.0f);
    return pu;
}

__device__ real sh_pq(real d, real h) {
    // pq (:2323-2345): moisture-transport partition (b1 = 2.0 only).
    const real a1 = 1.0f, a2 = -0.098f, a3 = 1.0f, a4 = 0.106f, a5 = 0.5f;
    real pq;
    if (h != 0.0f) {
        real doh = d / h;
        real t2 = doh * doh;
        real num = a1 * t2 + a2;
        real den = a3 * t2 + a4;
        pq = a5 * num / den + (1.0f - a5);
    } else {
        pq = 1.0f;
    }
    pq = sh_max(pq, 0.0f);
    pq = sh_min(pq, 1.0f);
    return pq;
}

__device__ real sh_pthnl(real d, real h) {
    // pthnl (:2349-2372): nonlocal heat-transport partition (b2 = 0.875).
    const real a1 = 1.000f, a2 = 0.936f, a3 = -1.110f;
    const real a4 = 1.000f, a5 = 0.312f, a6 = 0.329f, a7 = 0.243f;
    const real b2 = 0.875f;
    real pthnl;
    if (h != 0.0f) {
        real doh = d / h;
        real t2 = doh * doh;
        real tb = powf(doh, b2);
        real num = a1 * t2 + a2 * tb + a3;
        real den = a4 * t2 + a5 * tb + a6;
        pthnl = a7 * num / den + (1.0f - a7);
    } else {
        pthnl = 1.0f;
    }
    pthnl = sh_max(pthnl, 0.0f);
    pthnl = sh_min(pthnl, 1.0f);
    return pthnl;
}

__device__ real sh_pthl(real d, real h) {
    // pthl (:2376-2399): local heat-transport partition (b2 = 0.5).
    // (doh)**b2 stays a powf here too: the authority spells what WRF calls,
    // and this port spells what the authority spells.
    const real a1 = 1.000f, a2 = 0.870f, a3 = -0.913f;
    const real a4 = 1.000f, a5 = 0.153f, a6 = 0.278f, a7 = 0.280f;
    const real b2 = 0.5f;
    real pthl;
    if (h != 0.0f) {
        real doh = d / h;
        real t2 = doh * doh;
        real tb = powf(doh, b2);
        real num = a1 * t2 + a2 * tb + a3;
        real den = a4 * t2 + a5 * tb + a6;
        pthl = a7 * num / den + (1.0f - a7);
    } else {
        pthl = 1.0f;
    }
    pthl = sh_max(pthl, 0.0f);
    pthl = sh_min(pthl, 1.0f);
    return pthl;
}

__device__ real sh_ptke(real d, real h) {
    // ptke (:2403-2426): TKE-transport partition (same curve as pu).
    const real a1 = 1.000f, a2 = 0.070f, a3 = 1.000f, a4 = 0.142f,
               a5 = 0.071f;
    const real b2 = 0.6666667f;
    real ptke;
    if (h != 0.0f) {
        real doh = d / h;
        real t2 = doh * doh;
        real tb = powf(doh, b2);
        real num = a1 * t2 + a2 * tb;
        real den = a3 * t2 + a4 * tb + a5;
        ptke = num / den;
    } else {
        ptke = 1.0f;
    }
    ptke = sh_max(ptke, 0.0f);
    ptke = sh_min(ptke, 1.0f);
    return ptke;
}

// ------------------------------------------------------------------------
// The tridiagonal solver (tridi1n :1619 / tridin_ysu :1710) with the
// sequence-association shift applied: both solvers receive the actual
// argument al, dimensioned (its:ite, kts:kte), into a dummy cl dimensioned
// (its:ite, kts+1:kte+1), so cl(i,k) inside the solver is the caller's
// al(i,k-1) -- transcribed as al[k-1] wherever the Fortran says cl(i,k).
// One factorization serves every right-hand side: per RHS the two solvers
// emit the identical operation sequence, and tridin_ysu's per-sweep
// au recomputation is idempotent (see shinhong_ref._tridi_shared).
// Arrays are 1-based; lau is the LU upper band the Fortran calls au.
// ------------------------------------------------------------------------

__device__ void sh_tridi_factor(const real *al, const real *ad, const real *cu,
                                real *fkk, real *lau, int n) {
    fkk[1] = 1.0f / ad[1];
    lau[1] = fkk[1] * cu[1];
    for (int k = 2; k <= n - 1; ++k) {
        fkk[k] = 1.0f / (ad[k] - al[k - 1] * lau[k - 1]);   // cl(i,k) == al[k-1]
        lau[k] = fkk[k] * cu[k];
    }
    fkk[n] = 1.0f / (ad[n] - al[n - 1] * lau[n - 1]);
}

__device__ void sh_tridi_solve(const real *al, const real *fkk,
                               const real *lau, real *f, int n) {
    f[1] = fkk[1] * f[1];
    for (int k = 2; k <= n - 1; ++k)
        f[k] = fkk[k] * (f[k] - al[k - 1] * f[k - 1]);
    f[n] = fkk[n] * (f[n] - al[n - 1] * f[n - 1]);
    for (int k = n - 1; k >= 1; --k)
        f[k] = f[k] - lau[k] * f[k + 1];
}

// ------------------------------------------------------------------------
// mixlen (:1776-2012), one column, lmh = lmxl = 1.  Writes s2, ri, el
// (1-based, live entries 2..kte); gh is internal.  WRF also returns gh; the
// caller never reads it, so it stays local here.
// ------------------------------------------------------------------------

__device__ void sh_mixlen(const real *u, const real *v, const real *theta,
                          const real *exner, const real *qv, const real *qc,
                          int st, int col,
                          const real *q2, const real *z, real ustar, real corf,
                          real epshol, real hpbl, int lpbl, bool pblflg,
                          real hgamu, real hgamv, real hgamq,
                          const real *mf, const real *ufxpbl,
                          const real *vfxpbl, const real *qfxpbl, int kte,
                          real *s2, real *ri, real *el) {
    // Parameters (:1789-1826).  The a1/a2x/b1/b2/c1 decimals are
    // double-precision literals assigned to default REAL: one rounding of
    // the decimal to float32, which is what a C float literal is.  Every
    // derived parameter is folded per operation in float32, in the
    // association order of the expression tree -- the same chains the
    // authority spells; nvrtc folds float chains per-op (ysu.cu's
    // RV / RD - 1.0f measurement is the precedent).
    const real blckdr = 0.0063f, cn = 0.75f, eps1 = 1.0e-12f, epsl = 0.32f;
    const real epsru = 1.0e-7f, epsrs = 1.0e-7f;
    const real el0max = 1000.0f, el0min = 1.0f;
    const real elfc = 0.23f * 0.5f;              // elfc = 0.23*0.5 (:1791)
    const real alph = 0.30f;
    const real beta = 1.0f / 273.0f;             // beta = 1./273. (:1792)
    const real g_qnse = 9.81f;                   // mixlen local g, not the dummy
    const real btg = beta * g_qnse;              // btg = beta*g (:1792)
    const real a1 = 0.659888514560862645f;
    const real a2x = 0.6574209922667784586f;
    const real b1 = 11.87799326209552761f;
    const real b2 = 7.226971804046074028f;
    const real c1 = 0.000830955950095854396f;
    // adnh = 9.*a1*a2x*a2x*(12.*a1+3.*b2)*btg*btg (:1796)
    const real adnh = 9.0f * a1 * a2x * a2x * (12.0f * a1 + 3.0f * b2)
                    * btg * btg;
    // adnm = 18.*a1*a1*a2x*(b2-3.*a2x)*btg (:1797)
    const real adnm = 18.0f * a1 * a1 * a2x * (b2 - 3.0f * a2x) * btg;
    // bdnh = 3.*a2x*(7.*a1+b2)*btg, bdnm = 6.*a1*a1 (:1798)
    const real bdnh = 3.0f * a2x * (7.0f * a1 + b2) * btg;
    const real bdnm = 6.0f * a1 * a1;
    // aeqh, aeqm (:1802-1805)
    const real aeqh = 9.0f * a1 * a2x * a2x * b1 * btg * btg
                    + 9.0f * a1 * a2x * a2x * (12.0f * a1 + 3.0f * b2)
                    * btg * btg;
    const real aeqm = 3.0f * a1 * a2x * b1
                    * (3.0f * a2x + 3.0f * b2 * c1 + 18.0f * a1 * c1 - b2)
                    * btg
                    + 18.0f * a1 * a1 * a2x * (b2 - 3.0f * a2x) * btg;
    // requ = -aeqh/aeqm; epsgm = requ*epsgh (:1809-1810)
    const real requ = -(aeqh / aeqm);
    const real epsgh = 1.0e-9f;
    const real epsgm = requ * epsgh;
    // ubryl/ubry/ubry3 (:1814-1817)
    const real ubryl = (18.0f * requ * a1 * a1 * a2x * b2 * c1 * btg
                        + 9.0f * a1 * a2x * a2x * b2 * btg * btg)
                     / (requ * adnm + adnh);
    const real ubry = (1.0f + epsrs) * ubryl;
    const real ubry3 = 3.0f * ubry;
    // aubh/aubm/bubh/bubm/cubr/rcubr (:1818-1822)
    const real aubh = 27.0f * a1 * a2x * a2x * b2 * btg * btg - adnh * ubry3;
    const real aubm = 54.0f * a1 * a1 * a2x * b2 * c1 * btg - adnm * ubry3;
    const real bubh = (9.0f * a1 * a2x + 3.0f * a2x * b2) * btg
                    - bdnh * ubry3;
    const real bubm = 18.0f * a1 * a1 * c1 - bdnm * ubry3;
    const real cubr = 1.0f - ubry3;
    const real rcubr = 1.0f / cubr;
    const real elcbl = 0.77f;
    const real karman = 0.4f;
    // ep1: WRF's EP_1 forming, see the kernel body's comment.
    const real ep1 = RV / RD - 1.0f;

    real q1[SHINHONG_K2], dth[SHINHONG_K2], gh[SHINHONG_K2];
    real en2[SHINHONG_K2], elm[SHINHONG_K2], rel[SHINHONG_K2];
    for (int k = 0; k < SHINHONG_K2; ++k)
        q1[k] = dth[k] = gh[k] = en2[k] = elm[k] = rel[k] = 0.0f;

    const real elocp = 2.72e6f / CP;             // :1880
    const real ct = 0.0f;                        // :1881 (inout, zeroed here)
    for (int k = 2; k <= kte; ++k)               // :1887-1889
        dth[k] = SH_THX(k) - SH_THX(k - 1);
    for (int k = 3; k <= kte; ++k) {             // :1891-1896
        if (dth[k] > 0.0f && dth[k - 1] <= 0.0f) {
            dth[k] = dth[k] + ct;                // ct == 0.; dth[k] > 0 so exact
            break;
        }
    }
    for (int k = kte; k >= 2; --k) {             // :1900-1926
        real rdz = 2.0f / (z[k + 1] - z[k - 1]);
        real du = SH_UX(k) - SH_UX(k - 1);
        real dv = SH_VX(k) - SH_VX(k - 1);
        real s2l = (du * du + dv * dv) * rdz * rdz;   // integer **2 -> multiply
        if (pblflg && k <= lpbl) {
            real suk = (SH_UX(k) - SH_UX(k - 1)) * rdz;
            real svk = (SH_VX(k) - SH_VX(k - 1)) * rdz;
            s2l = (suk - hgamu / hpbl - ufxpbl[k]) * suk
                + (svk - hgamv / hpbl - vfxpbl[k]) * svk;
        }
        s2l = sh_max(s2l, epsgm);
        s2[k] = s2l;
        real tem = (SH_TX(k) + SH_TX(k - 1)) * 0.5f;
        real thm = (SH_THX(k) + SH_THX(k - 1)) * 0.5f;
        real a = thm * ep1;                      // p608 == ep1
        real b = (elocp / tem - 1.0f - ep1) * thm;
        real ghl = (dth[k] * ((SH_QVX(k) + SH_QVX(k - 1) + SH_QCX(k)
                               + SH_QCX(k - 1)) * (0.5f * ep1) + 1.0f)
                    + (SH_QVX(k) - SH_QVX(k - 1) + SH_QCX(k) - SH_QCX(k - 1))
                    * a
                    + (SH_QCX(k) - SH_QCX(k - 1)) * b) * rdz;
        if (pblflg && k <= lpbl)                 // :1918-1920
            ghl = ghl - mf[k] - (hgamq / hpbl + qfxpbl[k]) * a;
        if (fabsf(ghl) <= epsgh)
            ghl = epsgh;
        en2[k] = ghl * g_qnse / thm;
        gh[k] = ghl;
        ri[k] = en2[k] / s2l;
    }
    for (int k = kte; k >= 2; --k) {             // :1930-1950
        real s2l = s2[k];
        real ghl = gh[k];
        if (ghl >= epsgh) {
            if (s2l / ghl <= requ) {
                elm[k] = epsl;
            } else {
                real aubr = (aubm * s2l + aubh * ghl) * ghl;
                real bubr = bubm * s2l + bubh * ghl;
                real qol2st = (-0.5f * bubr
                               + sqrtf(bubr * bubr * 0.25f - aubr * cubr))
                            * rcubr;
                real eloq2x = 1.0f / qol2st;
                elm[k] = sh_max(sqrtf(eloq2x * q2[k]), epsl);
            }
        } else {
            real aden = (adnm * s2l + adnh * ghl) * ghl;
            real bden = bdnm * s2l + bdnh * ghl;
            real qol2un = -0.5f * bden + sqrtf(bden * bden * 0.25f - aden);
            real eloq2x = 1.0f / (qol2un + epsru);
            elm[k] = sh_max(sqrtf(eloq2x * q2[k]), epsl);
        }
    }
    for (int k = lpbl; k >= 1; --k)              // :1952-1954  (lmh == 1)
        q1[k] = sqrtf(q2[k]);
    real szq = 0.0f;
    real sq = 0.0f;
    for (int k = kte; k >= 2; --k) {             // :1956-1962
        real qdzl = (q1[k] + q1[k - 1]) * (z[k] - z[k - 1]);
        szq = (z[k] + z[k - 1] - z[1] - z[1]) * qdzl + szq;
        sq = qdzl + sq;
    }
    real el0 = sh_min(alph * szq * 0.5f / sq, el0max);   // :1966
    el0 = sh_max(el0, el0min);
    int lpblm = min(lpbl + 1, kte);              // :1971
    for (int k = kte; k >= lpblm; --k) {         // :1972-1975
        el[k] = (z[k + 1] - z[k - 1]) * elfc;
        rel[k] = el[k] / elm[k];
    }
    epshol = sh_min(epshol, 0.0f);               // :1979
    real ckp = elcbl * powf(1.0f - 8.0f * epshol, 1.0f / 3.0f);   // :1980
    if (lpbl > 1) {                              // :1981-1992
        for (int k = lpbl; k >= 2; --k) {
            real vkrmz = (z[k] - z[1]) * karman;
            if (pblflg) {
                vkrmz = ckp * (z[k] - z[1]) * karman;
                el[k] = vkrmz / (vkrmz / el0 + 1.0f);
            } else {
                el[k] = vkrmz / (vkrmz / el0 + 1.0f);
            }
            rel[k] = el[k] / elm[k];
        }
    }
    for (int k = lpbl - 1; k >= 3; --k) {        // :1994-1997  (lmh+2 == 3)
        real srel = sh_min(((rel[k - 1] + rel[k + 1]) * 0.5f + rel[k]) * 0.5f,
                           rel[k]);
        el[k] = sh_max(srel * elm[k], epsl);
    }
    real fcor = sh_max(corf, eps1);              // :2001 f = max(corf,eps1)
    real rlambda = fcor / (blckdr * ustar);      // ustar == 0 -> inf -> el == 0
    for (int k = kte; k >= 2; --k) {             // :2003-2010
        if (en2[k] >= 0.0f) {
            real vkrmz = (z[k] - z[1]) * karman;
            real rlb = rlambda + 1.0f / vkrmz;
            real rln = sqrtf(2.0f * en2[k] / q2[k]) / cn;
            el[k] = 1.0f / (rlb + rln);
        }
    }
}

// ------------------------------------------------------------------------
// prodq2 (:2016-2125), one column.  Mutates q2 in place.  Note dtturbl is
// dt, NOT dt2 (:1578 passes the wrapper's dt).  WRF also passes s2 and ri
// here; s2 feeds only a dead local and ri is never read, so both stay out.
// ------------------------------------------------------------------------

__device__ void sh_prodq2(real dtturbl, real ustar,
                          const real *u, const real *v, const real *theta,
                          int st, int col, const real *thvx,
                          real *q2, const real *el, const real *z,
                          const real *akm, const real *akh,
                          real hgamu, real hgamv, real hgamq, real delxy,
                          real hpbl, bool pblflg, int kpbl,
                          const real *mf, const real *ufxpbl,
                          const real *vfxpbl, const real *qfxpbl, int kte) {
    const real g_qnse = 9.81f;                   // prodq2 local g (:2029 block)
    const real c0 = 0.55f, ceps = 16.6f;
    const real epsq2l = 0.01f;
    const real ep1 = RV / RD - 1.0f;
    const real rc02 = 2.0f / (c0 * c0);          // :2073
    for (int k = 2; k <= kte; ++k) {             // :2077-2119
        real deltaz = 0.5f * (z[k + 1] - z[k - 1]);
        real q2l = q2[k];
        real suk = (SH_UX(k) - SH_UX(k - 1)) / deltaz;
        real svk = (SH_VX(k) - SH_VX(k - 1)) / deltaz;
        real gthvk = (thvx[k] - thvx[k - 1]) / deltaz;
        real govrthvk = g_qnse / (0.5f * (thvx[k] + thvx[k - 1]));
        real akml = akm[k];
        real akhl = akh[k];
        real thm = (SH_THX(k) + SH_THX(k - 1)) * 0.5f;
        real pru, prv;
        if (pblflg && k <= kpbl) {               // :2092-2098
            pru = (akml * (suk - hgamu / hpbl - ufxpbl[k])) * suk;
            prv = (akml * (svk - hgamv / hpbl - vfxpbl[k])) * svk;
        } else {
            pru = akml * suk * suk;
            prv = akml * svk * svk;
        }
        real pr = pru + prv;
        real bpr;
        if (pblflg && k <= kpbl) {               // :2103-2107
            bpr = (akhl * (gthvk - mf[k]
                           - (hgamq / hpbl + qfxpbl[k]) * ep1 * thm))
                * govrthvk;
        } else {
            bpr = akhl * gthvk * govrthvk;
        }
        real disel = sh_min(delxy, ceps * el[k]);   // :2111
        real dis = powf(q2l, 1.5f) / disel;         // (q2l)**1.5 stays powf
        q2l = q2l + 2.0f * (pr - bpr - dis) * dtturbl;
        q2[k] = sh_max(q2l, epsq2l);                // amax1: NaN propagates
    }
    q2[1] = sh_max(rc02 * ustar * ustar, epsq2l);   // :2123
}

// ------------------------------------------------------------------------
// vdifq (:2129-2231), one column, lmh = 1.  Mutates q2.  The akq/cm/cr/dtoz/
// rsq2 arrays run kts+2..kte in the Fortran (:2172-2176); the loops below
// are transcribed index-exactly against that (trap: the lower boundary
// handling at :2212-2225 reaches akq(3)/cm(3)/cr(3) only).  WRF also passes
// el here and never reads it.
// ------------------------------------------------------------------------

__device__ void sh_vdifq(real dtdif, real *q2, const real *z, const real *akhk,
                         const real *ptke1, const real *hgame, real hpbl,
                         bool pblflg, int kpbl, real efxpbl, int kte) {
    const real c_k = 1.0f;
    const int lmh = 1;
    real zfacentk[SHINHONG_K2], dtoz[SHINHONG_K2], akq[SHINHONG_K2];
    real cr[SHINHONG_K2], cm[SHINHONG_K2], rsq2[SHINHONG_K2];
    for (int k = 0; k < SHINHONG_K2; ++k)
        zfacentk[k] = dtoz[k] = akq[k] = cr[k] = cm[k] = rsq2[k] = 0.0f;

    for (int k = 2; k <= kte; ++k) {             // :2183-2186
        real zak = 0.5f * (z[k] + z[k - 1]);     // za(k-1) of shinhong2d
        zfacentk[k] = powf(zak / hpbl, 3.0f);    // ((zak/hpbl))**3.0 -- a powf
    }
    for (int k = kte; k >= 3; --k) {             // :2188-2193 (kte .. kts+2)
        dtoz[k] = (dtdif + dtdif) / (z[k + 1] - z[k - 1]);
        akq[k] = c_k * (akhk[k] / (z[k + 1] - z[k - 1])
                        + akhk[k - 1] / (z[k] - z[k - 2]));
        akq[k] = akq[k] * ptke1[k];
        cr[k] = -(dtoz[k] * akq[k]);
    }
    real akqs = c_k * akhk[2] / (z[3] - z[1]);   // :2195-2196
    akqs = akqs * ptke1[2];
    cm[kte] = dtoz[kte] * akq[kte] + 1.0f;       // :2197-2198
    rsq2[kte] = q2[kte];
    for (int k = kte - 1; k >= 3; --k) {         // :2200-2210
        real cf = -(dtoz[k] * akq[k + 1] / cm[k + 1]);
        cm[k] = -(cr[k + 1] * cf) + (akq[k + 1] + akq[k]) * dtoz[k] + 1.0f;
        rsq2[k] = -(rsq2[k + 1] * cf) + q2[k];
        if (pblflg && k < kpbl) {
            rsq2[k] = rsq2[k]
                    - dtoz[k] * (2.0f * hgame[k] / hpbl) * akq[k + 1]
                    * (z[k + 1] - z[k])
                    + dtoz[k] * (2.0f * hgame[k - 1] / hpbl) * akq[k]
                    * (z[k] - z[k - 1]);
            rsq2[k] = rsq2[k]
                    - dtoz[k] * 2.0f * efxpbl * zfacentk[k + 1]
                    + dtoz[k] * 2.0f * efxpbl * zfacentk[k];
        }
    }
    real dtozs = (dtdif + dtdif) / (z[3] - z[1]);   // :2212
    real cf = -(dtozs * akq[lmh + 2] / cm[lmh + 2]);   // :2213
    if (pblflg && (lmh + 1) < kpbl) {            // :2215-2225
        q2[2] = dtozs * akqs * q2[1] - rsq2[3] * cf + q2[2]
              - dtozs * (2.0f * hgame[2] / hpbl) * akq[3] * (z[3] - z[2])
              + dtozs * (2.0f * hgame[1] / hpbl) * akqs * (z[2] - z[1]);
        q2[2] = q2[2] - dtozs * 2.0f * efxpbl * zfacentk[3]
              + dtozs * 2.0f * efxpbl * zfacentk[2];
        q2[2] = q2[2] / ((akq[3] + akqs) * dtozs - cr[3] * cf + 1.0f);
    } else {
        q2[2] = (dtozs * akqs * q2[1] - rsq2[3] * cf + q2[2])
              / ((akq[3] + akqs) * dtozs - cr[3] * cf + 1.0f);
    }
    for (int k = 3; k <= kte; ++k)               // :2227-2229 (lmh+2 .. kte)
        q2[k] = (-(cr[k] * q2[k - 1]) + rsq2[k]) / cm[k];
}

// ------------------------------------------------------------------------
// The column: shinhong (:9-262) around shinhong2d (:266-1615), arm A.
// ------------------------------------------------------------------------

extern "C" __global__
void shinhong_column(const real *u, const real *v, const real *theta,
                     const real *qv, const real *qc, const real *qi,
                     const real *p, const real *p_interface,
                     const real *exner, const real *dz, const real *tke,
                     const real *psfc, const real *znt, const real *ust,
                     const real *hfx, const real *qfx, const real *wspd,
                     const real *br, const real *psim, const real *psih,
                     const real *xland, const real *u10, const real *v10,
                     const real *corf,
                     real *du, real *dv, real *dtheta, real *dqv, real *dqc,
                     real *dqi, real *exch_h, real *tke_out, real *el,
                     real *hpbl_out, int *kpbl_out, real *wstar_out,
                     real *delta_out,
                     real dt, real dx, real dy, int tke_diag,
                     int nz, int ny, int nx) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int st = ny * nx;
    if (col >= st || nz > SHINHONG_KMAX) return;
    const int kte = nz;
    const int klpbl = kte;                       // klpbl = kte (:531 region)

    // shinhong2d parameters (:321-346), WRF literals.
    const real xkzminm = 0.1f, xkzminh = 0.01f, xkzmax = 1000.0f;
    const real rimin = -100.0f, rlam = 30.0f, prmin = 0.25f, prmax = 4.0f;
    const real brcr_ub = 0.0f, brcr_sb = 0.25f, cori = 1.0e-4f;
    const real afac = 6.8f, bfac = 6.8f, pfac = 2.0f, pfac_q = 2.0f;
    const real phifac = 8.0f, sfcfrac = 0.1f;
    const real d1 = 0.02f, d2 = 0.05f, d3 = 0.001f;
    const real h1 = 0.33333333f;   // rounds to float32 1/3 (pow-probe.txt)
    const real h2 = 0.6666667f;
    const real zfmin = 1.0e-8f, aphi5 = 5.0f, aphi16 = 16.0f;
    const real tmin = 1.0e-2f, gamcrt = 3.0f, gamcrq = 2.0e-3f;
    const real epsq2l = 0.01f, c_1 = 1.0f, gamcre = 0.224f;
    const real mltop = 1.0f, sfcfracn1 = 0.075f, nlfrac = 0.7f;
    const real enlfrac = -0.4f;
    const real a11 = 1.0f, a12 = -1.15f, ezfac = 1.5f;
    const real cpent = -0.4f, rigsmax = 100.0f, entfmin = 1.0f,
               entfmax = 5.0f;
    // ep1 is WRF's EP_1, and WRF forms it in float32: module_model_constants
    // declares `REAL, PARAMETER :: EP_1 = R_v/R_d - 1.` so the quotient is a
    // float32 divide of two float32 values.  CUDA_DEFINES["RVOVRD"] is RV/RD
    // computed in Python doubles and rounded once to float32, which lands 1
    // ULP below WRF's quotient at 1.608 and therefore 2 ULP below it here at
    // 0.608.  Spelling the divide keeps the whole thv/Richardson/PBL-diagnosis
    // chain on WRF's constant (ysu.cu precedent, measured there).
    const real karman = 0.4f, ep1 = RV / RD - 1.0f;

    real thvx[SHINHONG_K2], zq[SHINHONG_K2], za[SHINHONG_K2];
    real delp[SHINHONG_K2], dza[SHINHONG_K2];
    real q2x[SHINHONG_K2], hgame2d[SHINHONG_K2];
    real tflux_e[SHINHONG_K2], qflux_e[SHINHONG_K2], tvflux_e[SHINHONG_K2];
    real mf[SHINHONG_K2], entfacmf[SHINHONG_K2], entfac[SHINHONG_K2];
    real xkzm[SHINHONG_K2], xkzh[SHINHONG_K2], xkzq[SHINHONG_K2];
    real xkzml[SHINHONG_K2], xkzhl[SHINHONG_K2], zfacent[SHINHONG_K2];
    real al[SHINHONG_K2], ad[SHINHONG_K2], au[SHINHONG_K2];
    real fkk[SHINHONG_K2], lau[SHINHONG_K2];
    // r1/r2/r3 are the solver right-hand sides: heat uses r1 (f1 :1141);
    // moisture uses r1/r2/r3 (f3 components 1..3, :1230-1245); momentum
    // reuses r1/r2 (f1/f2, :1389-1390).
    real r1[SHINHONG_K2], r2[SHINHONG_K2], r3[SHINHONG_K2];
    for (int k = 0; k < SHINHONG_K2; ++k) {
        thvx[k] = zq[k] = za[k] = delp[k] = dza[k] = 0.0f;
        q2x[k] = hgame2d[k] = 0.0f;
        tflux_e[k] = qflux_e[k] = tvflux_e[k] = 0.0f;
        mf[k] = entfacmf[k] = entfac[k] = 0.0f;
        xkzm[k] = xkzh[k] = xkzq[k] = xkzml[k] = xkzhl[k] = zfacent[k] = 0.0f;
        al[k] = ad[k] = au[k] = fkk[k] = lau[k] = 0.0f;
        r1[k] = r2[k] = r3[k] = 0.0f;
    }

    const real psfcv = psfc[col], zntv = znt[col], ustv = ust[col];
    const real hfxv = hfx[col], qfxv = qfx[col], wspdv = wspd[col];
    const real brv = br[col], psimv = psim[col], psihv = psih[col];
    const real xlandv = xland[col], corfv = corf[col];
    const real u10v = u10[col], v10v = v10[col];

    const real cont = CP / G;                    // :544
    const real conpr = bfac * karman * sfcfrac;  // :548
    // conq/conw/conwrc (:545-547) feed only the dusfc/dvsfc/dtsfc/dqsfc
    // accumulators, which the wrapper never reads; not transcribed.

    // thx is the model's theta itself (see SH_THX above); thvx (:563-568).
    for (int k = 1; k <= kte; ++k) {
        real tvcon = 1.0f + ep1 * SH_QVX(k);
        thvx[k] = SH_THX(k) * tvcon;
    }
    real tvcon = 1.0f + ep1 * SH_QVX(1);         // :570-574
    real rhox = psfcv / (RD * SH_TX(1) * tvcon);
    real govrth = G / SH_THX(1);

    zq[1] = 0.0f;                                // :579-587 (zq(:,1) = 0)
    for (int k = 1; k <= kte; ++k)
        zq[k + 1] = SH_DZ8W(k) + zq[k];
    for (int k = 1; k <= kte; ++k) {             // :589-595 (dzq never read)
        za[k] = 0.5f * (zq[k] + zq[k + 1]);
        delp[k] = p_interface[(size_t)(k - 1) * st + col]
                - p_interface[(size_t)k * st + col];
    }
    dza[1] = za[1];                              // :597-605
    for (int k = 2; k <= kte; ++k)
        dza[k] = za[k] - za[k - 1];

    real wspd1 = sqrtf(SH_UX(1) * SH_UX(1) + SH_VX(1) * SH_VX(1))
               + 1.0e-9f;                        // :616

    real dt2 = 2.0f * dt;                        // :624-626
    real rdt = 1.0f / dt2;

    real hgamu = 0.0f, hgamv = 0.0f, delta = 0.0f, efxpbl = 0.0f;
    real epshol = 0.0f, deltaoh = 0.0f, rigs = 0.0f, enlfrac2 = 0.0f;
    real cslen = 0.0f;
    for (int k = 1; k <= kte; ++k)               // :665-669
        q2x[k] = 2.0f * SH_TKEX(k);
    // :671-679 -- el_pbl is zeroed UNCONDITIONALLY, tke_diag on or off, and
    // tke passes through untouched when the diagnostic is off (:1478).
    for (int k = 1; k <= kte; ++k) {
        SH_G(el, k) = 0.0f;
        SH_G(tke_out, k) = SH_TKEX(k);
    }
    // xkzom/xkzoh (:688-693) hold xkzminm/xkzminh on kts..klpbl-1 and zero at
    // klpbl == kte.  Every read below sits at k <= kte-1 (the k < kpbl loop,
    // the free atmosphere, and the three entrainment overwrites all stop at
    // kte-1), so the constants stand in for the arrays.

    real hgamt = 0.0f, hgamq = 0.0f;             // :702-715
    int kpbl = 1;
    real hpbl = zq[1];
    real zl1 = za[1];
    real thermal = thvx[1];
    bool pblflg = true;
    bool sfcflg = true;
    real sflux = hfxv / rhox / CP + qfxv / rhox * ep1 * SH_THX(1);
    // :706 `if(br(i).gt.0.0) sfcflg = .false.` -- done in DOUBLE via
    // sh_f2d, not float32.  sm_120 DAZ reads a subnormal br as zero in
    // every FP32 operation including compares, so the plain float32 compare
    // turned case 7 (br = +1.4e-45) into a regime change, measured on this
    // fixture: wstar 0.2362 where WRF writes 0.0, delta 18.49 where WRF
    // writes 0.0, hpbl 1525 ULP off, momentum tendencies ~1.8e9 ULP
    // (sign-flipped near-cancellations).  A plain `(double)brv > 0.0`
    // measured IDENTICAL to the float32 compare -- the emitted cvt.f64.f32
    // DAZes its input too -- hence the bit-decoding sh_f2d (rrtmg_sw.cu's
    // rsw_f2d), which takes WRF's branch again.  Case 8 (br = -1.4e-45)
    // takes the same arm either way (0 > 0 and -denormal > 0 are both
    // false) and never needed it.
    if (sh_f2d(brv) > 0.0)
        sfcflg = false;

    // ---- first guess of the PBL height (:719-749) ----
    bool stable = false;
    real brup = brv;
    real brcr = brcr_ub;
    real brdn = 0.0f;
    for (int k = 2; k <= klpbl; ++k) {
        if (!stable) {
            brdn = brup;
            real spdk2 = sh_max(SH_UX(k) * SH_UX(k) + SH_VX(k) * SH_VX(k),
                                1.0f);
            brup = (thvx[k] - thermal) * (G * za[k] / thvx[1]) / spdk2;
            kpbl = k;
            stable = brup > brcr;
        }
    }
    {
        int k = kpbl;
        real brint;
        if (brdn >= brcr) brint = 0.0f;
        else if (brup <= brcr) brint = 1.0f;
        else brint = (brcr - brdn) / (brup - brdn);
        hpbl = za[k - 1] + brint * (za[k] - za[k - 1]);
        if (hpbl < zq[2]) kpbl = 1;
        if (kpbl <= 1) pblflg = false;
    }

    // ---- surface-layer scales (:751-780) ----
    real fm = psimv;
    real fh = psihv;
    real zol1 = sh_max(brv * fm * fm / fh, rimin);
    if (sfcflg) zol1 = sh_min(zol1, -zfmin);
    else zol1 = sh_max(zol1, zfmin);
    real hol1 = zol1 * hpbl / zl1 * sfcfrac;
    epshol = hol1;
    real phim, phih, wstar, wstar3;
    if (sfcflg) {
        phim = powf(1.0f - aphi16 * hol1, -(1.0f / 4.0f));
        phih = powf(1.0f - aphi16 * hol1, -(1.0f / 2.0f));
        real bfx0 = sh_max(sflux, 0.0f);
        // hfx0/qfx0 (:766-767) are computed by WRF and never read; omitted.
        wstar3 = govrth * bfx0 * hpbl;
        wstar = powf(wstar3, h1);
    } else {
        phim = 1.0f + aphi5 * hol1;
        phih = phim;
        wstar = 0.0f;
        wstar3 = 0.0f;
    }
    real ust3 = powf(ustv, 3.0f);                // ust**3. stays powf
    real wscale = powf(ust3 + phifac * karman * wstar3 * 0.5f, h1);
    wscale = sh_min(wscale, ustv * aphi16);
    wscale = sh_max(wscale, ustv / aphi5);

    // ---- countergradient terms (:785-800) ----
    if (sfcflg && sflux > 0.0f) {
        real gamfac = bfac / rhox / wscale;
        hgamt = sh_min(gamfac * hfxv / CP, gamcrt);
        hgamq = sh_min(gamfac * qfxv, gamcrq);
        real vpert = (hgamt + ep1 * SH_THX(1) * hgamq) / bfac * afac;
        thermal = thermal + (sh_max(vpert, 0.0f)
                             * sh_min(za[1] / (sfcfrac * hpbl), 1.0f));
        hgamt = sh_max(hgamt, 0.0f);
        hgamq = sh_max(hgamq, 0.0f);
        real brint = -15.9f * ustv * ustv / wspdv * wstar3
                   / powf(wscale, 4.0f);         // wscale**4. stays powf
        hgamu = brint * SH_UX(1);
        hgamv = brint * SH_VX(1);
    } else {
        pblflg = false;
    }

    // ---- enhance the PBL height by the thermal excess (:804-853) ----
    if (pblflg) {
        kpbl = 1;
        hpbl = zq[1];
        stable = false;
        brup = brv;
        brcr = brcr_ub;
    }
    for (int k = 2; k <= klpbl; ++k) {
        if (!stable && pblflg) {
            brdn = brup;
            real spdk2 = sh_max(SH_UX(k) * SH_UX(k) + SH_VX(k) * SH_VX(k),
                                1.0f);
            brup = (thvx[k] - thermal) * (G * za[k] / thvx[1]) / spdk2;
            kpbl = k;
            stable = brup > brcr;
        }
    }
    if (pblflg) {
        int k = kpbl;
        real brint;
        if (brdn >= brcr) brint = 0.0f;
        else if (brup <= brcr) brint = 1.0f;
        else brint = (brcr - brdn) / (brup - brdn);
        hpbl = za[k - 1] + brint * (za[k] - za[k - 1]);
        if (hpbl < zq[2]) kpbl = 1;
        if (kpbl <= 1) pblflg = false;
        // :844-851 -- still inside the outer if(pblflg) block entered above,
        // so csfac/cslen are computed even when pblflg was just falsified.
        real csfac;
        if (wstar != 0.0f) {
            real uwst = fabsf(ustv / wstar - 0.5f);
            real uwstx = -80.0f * uwst + 14.0f;
            csfac = 0.5f * (tanhf(uwstx) + 3.0f);
        } else {
            csfac = 1.0f;   // dead in WRF: pblflg requires sflux>0 => wstar>0
        }
        cslen = csfac * hpbl;
    }

    // ---- stable boundary layer (:857-912) ----
    if (!sfcflg && hpbl < zq[2]) {
        brup = brv;
        stable = false;
    } else {
        stable = true;
    }
    real brcr_sbro = 0.0f;
    if (!stable && (xlandv - 1.5f) >= 0.0f) {    // :867-874
        real wspd10 = u10v * u10v + v10v * v10v;
        wspd10 = sqrtf(wspd10);
        real ross = wspd10 / (cori * zntv);
        brcr_sbro = sh_min(0.16f * powf(1.0e-7f * ross, -0.18f), 0.3f);
    }
    if (!stable) {                               // :876-884
        if ((xlandv - 1.5f) >= 0.0f) brcr = brcr_sbro;
        else brcr = brcr_sb;
    }
    for (int k = 2; k <= klpbl; ++k) {           // :886-896
        if (!stable) {
            brdn = brup;
            real spdk2 = sh_max(SH_UX(k) * SH_UX(k) + SH_VX(k) * SH_VX(k),
                                1.0f);
            brup = (thvx[k] - thermal) * (G * za[k] / thvx[1]) / spdk2;
            kpbl = k;
            stable = brup > brcr;
        }
    }
    if (!sfcflg && hpbl < zq[2]) {               // :898-912
        int k = kpbl;
        real brint;
        if (brdn >= brcr) brint = 0.0f;
        else if (brup <= brcr) brint = 1.0f;
        else brint = (brcr - brdn) / (brup - brdn);
        hpbl = za[k - 1] + brint * (za[k] - za[k - 1]);
        if (hpbl < zq[2]) kpbl = 1;
        if (kpbl <= 1) pblflg = false;
    }

    // ---- scale dependency, nonlocal momentum and moisture (:916-926) ----
    real delxy = sqrtf(dx * dy);
    real pu1 = sh_pu(delxy, cslen);              // cslen == 0 columns take the
    real pq1 = sh_pq(delxy, cslen);              // h == 0 early-out and get 1.
    if (pblflg) {
        hgamu = hgamu * pu1;
        hgamv = hgamv * pu1;
        hgamq = hgamq * pq1;
    }

    // ---- entrainment parameters (:930-985) ----
    real prpbl = 0.0f;
    real wm2 = 0.0f;
    real we = 0.0f;
    real bfxpbl = 0.0f;                          // :629-637
    real hfxpbl = 0.0f;
    real qfxpbl = 0.0f;
    real ufxpbl = 0.0f;
    real vfxpbl = 0.0f;
    real dthvx = 0.0f;
    if (pblflg) {
        int kt = kpbl - 1;                       // k = kpbl - 1 (:931)
        prpbl = 1.0f;
        real wm3 = wstar3 + 5.0f * ust3;
        wm2 = powf(wm3, h2);
        bfxpbl = -0.15f * thvx[1] / G * wm3 / hpbl;
        dthvx = sh_max(thvx[kt + 1] - thvx[kt], tmin);
        real dthx = sh_max(SH_THX(kt + 1) - SH_THX(kt), tmin);
        real dqx = sh_min(SH_QVX(kt + 1) - SH_QVX(kt), 0.0f);
        we = sh_max(bfxpbl / dthvx, -sqrtf(wm2));
        hfxpbl = we * dthx;                      // :943 -- overwritten at
        pq1 = sh_pq(delxy, cslen);               // :1114; ported as written
        qfxpbl = we * dqx * pq1;
        pu1 = sh_pu(delxy, cslen);
        real dux = SH_UX(kt + 1) - SH_UX(kt);
        real dvx = SH_VX(kt + 1) - SH_VX(kt);
        if (dux > tmin)
            ufxpbl = sh_max(prpbl * we * dux * pu1, -(ustv * ustv));
        else if (dux < -tmin)
            ufxpbl = sh_min(prpbl * we * dux * pu1, ustv * ustv);
        else
            ufxpbl = 0.0f;
        if (dvx > tmin)
            vfxpbl = sh_max(prpbl * we * dvx * pu1, -(ustv * ustv));
        else if (dvx < -tmin)
            vfxpbl = sh_min(prpbl * we * dvx * pu1, ustv * ustv);
        else
            vfxpbl = 0.0f;
        real delb = govrth * d3 * hpbl;
        delta = sh_min(d1 * hpbl + d2 * wm2 / delb, 100.0f);
        delb = govrth * dthvx;
        deltaoh = d1 * hpbl + d2 * wm2 / delb;
        deltaoh = sh_max(ezfac * deltaoh, hpbl - za[kpbl - 1] - 1.0f);
        deltaoh = sh_min(deltaoh, hpbl);
        if (dux != 0.0f || dvx != 0.0f) {
            // dux**2. and dvx**2. fold to multiplies (pow-probe, negative
            // bases included).
            rigs = govrth * dthvx * deltaoh / (dux * dux + dvx * dvx);
        } else {
            rigs = rigsmax;
        }
        rigs = sh_max(sh_min(rigs, rigsmax), rimin);
        real cenlfrac;
        if (rigs > 0.0f && fabsf(rigs + cpent) <= 1.0e-6f)
            cenlfrac = entfmax;
        else
            cenlfrac = rigs / (rigs + cpent);
        cenlfrac = sh_min(cenlfrac, entfmax);
        enlfrac2 = sh_max(wm3 / wstar3 * cenlfrac, entfmin);
        enlfrac2 = enlfrac2 * enlfrac;
    }

    for (int k = 1; k <= klpbl; ++k) {           // :987-998
        if (pblflg) {
            real tmf = (zq[k + 1] - hpbl) / deltaoh;
            entfacmf[k] = sqrtf(tmf * tmf);      // sqrt((...)**2.), **2. folds
        }
        if (pblflg && k >= kpbl) {
            real te = (zq[k + 1] - hpbl) / deltaoh;
            entfac[k] = te * te;
        } else {
            entfac[k] = 1.0e30f;
        }
    }

    // ---- diffusion coefficients below the PBL top (:1002-1036) ----
    for (int k = 1; k <= klpbl; ++k) {
        if (k < kpbl) {
            real zfac = sh_min(sh_max(
                1.0f - (zq[k + 1] - zl1) / (hpbl - zl1), zfmin), 1.0f);
            zfacent[k] = powf(1.0f - zfac, 3.0f);   // (1.-zfac)**3. -- powf
            // The ONE cbrtf site (see the header's per-site table): powf
            // left dv's worst column at 1491308 ULP; cbrtf lands closer to
            // the oracle here (dv 46603, exch_h 8) for tke 2005 -> 2022.
            real wscalek = cbrtf(
                ust3 + phifac * karman * wstar3 * (1.0f - zfac));
            real prfac, prfac2, prnumfac;
            if (sfcflg) {
                prfac = conpr;
                // :1010 -- WRF's own 0/0 when both cubes are exactly zero
                // (fixture case 13); the NaN column downstream is the point.
                prfac2 = 15.9f * wstar3 / ust3
                       / (1.0f + 4.0f * karman * wstar3 / ust3);
                real m = sh_max(zq[k + 1] - sfcfrac * hpbl, 0.0f);
                prnumfac = -3.0f * (m * m) / (hpbl * hpbl);   // **2. folds
            } else {
                prfac = 0.0f;
                prfac2 = 0.0f;
                prnumfac = 0.0f;
                real phim8z = 1.0f + aphi5 * zol1 * zq[k + 1] / zl1;
                wscalek = ustv / phim8z;
                wscalek = sh_max(wscalek, 0.001f);
            }
            real prnum0 = phih / phim + prfac;
            prnum0 = sh_max(sh_min(prnum0, prmax), prmin);
            xkzm[k] = wscalek * karman * zq[k + 1] * (zfac * zfac);   // **pfac folds
            real prnum = 1.0f + (prnum0 - 1.0f) * expf(prnumfac);
            // zfac**(pfac_q-pfac) == zfac**0. == 1.0 for every zfac
            // (pow-probe.txt, including 0; device powf(x, 0) is 1 for all x);
            // the multiply is kept.
            xkzq[k] = xkzm[k] / prnum * powf(zfac, pfac_q - pfac);
            prnum0 = prnum0 / (1.0f + prfac2 * karman * sfcfrac);
            prnum = 1.0f + (prnum0 - 1.0f) * expf(prnumfac);
            xkzh[k] = xkzm[k] / prnum;
            xkzm[k] = xkzm[k] + xkzminm;         // + xkzom(k), k <= kte-1
            xkzh[k] = xkzh[k] + xkzminh;         // + xkzoh(k)
            xkzq[k] = xkzq[k] + xkzminh;
            xkzm[k] = sh_min(xkzm[k], xkzmax);
            xkzh[k] = sh_min(xkzh[k], xkzmax);
            xkzq[k] = sh_min(xkzq[k], xkzmax);
        }
    }

    // ---- free atmosphere (:1040-1086) ----
    for (int k = 1; k <= kte - 1; ++k) {
        if (k >= kpbl) {
            real duk = SH_UX(k + 1) - SH_UX(k);
            real dvk = SH_VX(k + 1) - SH_VX(k);
            real ss = (duk * duk + dvk * dvk) / (dza[k + 1] * dza[k + 1])
                    + 1.0e-9f;
            real govrthv = G / (0.5f * (thvx[k + 1] + thvx[k]));
            real ri = govrthv * (thvx[k + 1] - thvx[k]) / (ss * dza[k + 1]);
            // :1049-1058 in cloud (imvdif == 1, nwmass == 3 always):
            // qx(i,kqc+k-1) is qc(k) and qx(i,kqi+k-1) is qi(k) in the
            // ndiff-packed array (kqc = 1+kte, kqi = 1+2*kte).
            if ((SH_QCX(k) + SH_QIX(k)) > 0.01e-3f
                    && (SH_QCX(k + 1) + SH_QIX(k + 1)) > 0.01e-3f) {
                real qmean = 0.5f * (SH_QVX(k) + SH_QVX(k + 1));
                real tmean = 0.5f * (SH_TX(k) + SH_TX(k + 1));
                real alpha = XLV * qmean / RD / tmean;
                real chi = XLV * XLV * qmean / CP / RV / tmean / tmean;
                ri = (1.0f + alpha) * (
                    ri - G * G / ss / tmean / CP
                    * ((chi - alpha) / (1.0f + chi)));
            }
            real zk = karman * zq[k + 1];
            real rlamdz = sh_min(sh_max(0.1f * dza[k + 1], rlam), 300.0f);
            rlamdz = sh_min(dza[k + 1], rlamdz);
            real tt = zk * rlamdz / (rlamdz + zk);
            real rl2 = tt * tt;                  // integer **2 -> multiply
            real dk = rl2 * sqrtf(ss);
            if (ri < 0.0f) {
                ri = sh_max(ri, rimin);
                real sri = sqrtf(-ri);
                xkzm[k] = dk * (1.0f + 8.0f * (-ri)
                                / (1.0f + 1.746f * sri));
                xkzh[k] = dk * (1.0f + 8.0f * (-ri)
                                / (1.0f + 1.286f * sri));
            } else {
                tt = 1.0f + 5.0f * ri;
                xkzh[k] = dk / (tt * tt);        // integer **2 -> multiply
                real prnum = 1.0f + 2.1f * ri;
                prnum = sh_min(prnum, prmax);
                xkzm[k] = xkzh[k] * prnum;
            }
            xkzm[k] = xkzm[k] + xkzminm;         // + xkzom(k), k <= kte-1
            xkzh[k] = xkzh[k] + xkzminh;
            xkzm[k] = sh_min(xkzm[k], xkzmax);
            xkzh[k] = sh_min(xkzh[k], xkzmax);
            xkzml[k] = xkzm[k];
            xkzhl[k] = xkzh[k];
        }
    }

    // ---- prescribed nonlocal heat transport (:1090-1130) ----
    deltaoh = deltaoh / hpbl;                    // :1091, every column
    delxy = sqrtf(dx * dy);
    real mlfrac = mltop - deltaoh;
    // ezfrac (:1097) is computed by WRF and never read; omitted.
    // :1098 -- zfacmf(i,1) is first computed CLAMPED and feeds sfcfracn; the
    // k-loop below then RE-computes it UNCLAMPED (:1119), overwriting index 1,
    // and the mf profile uses the unclamped value.
    real zfacmf1 = sh_min(sh_max(zq[2] / hpbl, zfmin), 1.0f);
    real sfcfracn = sh_max(sfcfracn1, zfacmf1);
    real sflux0 = (a11 + a12 * sfcfracn) * sflux;
    real snlflux0 = nlfrac * sflux0;
    real amf1 = snlflux0 / sfcfracn;
    real amf2 = 0.0f;
    real bmf2 = 0.0f;
    if (pblflg) {
        amf2 = -(snlflux0 / (mlfrac - sfcfracn));
        bmf2 = -(mlfrac * amf2);
    }
    real amf3;
    if (deltaoh == 0.0f && enlfrac2 == 0.0f)     // :1108, exact compares
        amf3 = 0.0f;
    else
        amf3 = snlflux0 * enlfrac2 / deltaoh;
    real bmf3 = -(amf3 * mlfrac);
    // :1114-1116 -- hfxpbl is OVERWRITTEN here (the :943 we*dthx value is
    // discarded) and scaled by pthnl once ...
    hfxpbl = amf3 + bmf3;
    real pth1 = sh_pthnl(delxy, cslen);
    hfxpbl = hfxpbl * pth1;
    for (int k = 1; k <= klpbl; ++k) {
        real zfacmf = sh_max(zq[k + 1] / hpbl, zfmin);   // :1119, unclamped
        if (pblflg && k < kpbl) {
            if (zfacmf <= sfcfracn)
                mf[k] = amf1 * zfacmf;
            else if (zfacmf <= mlfrac)
                mf[k] = amf2 * zfacmf + bmf2;
            // ... and the WHOLE mf -- entrainment term included -- is scaled
            // by pth1 AGAIN at :1127, so the entrainment piece is
            // double-scaled.  WRF's arithmetic, ported as written.
            mf[k] = mf[k] + hfxpbl * expf(-entfacmf[k]);
            mf[k] = mf[k] * pth1;
        }
    }

    // ---- heat: matrix assembly (:1134-1187), solve, recover ----
    ad[1] = 1.0f;
    r1[1] = SH_THX(1) - 300.0f + hfxv / cont / delp[1] * dt2;
    SH_G(exch_h, 1) = 0.0f;   // exch_h(:,1) is never written by the scheme
    for (int k = 1; k <= kte - 1; ++k) {
        real dtodsd = dt2 / delp[k];
        real dtodsu = dt2 / delp[k + 1];
        real dsig = SH_P2DX(k) - SH_P2DX(k + 1);
        real rdz = 1.0f / dza[k + 1];
        real tem1 = dsig * xkzh[k] * rdz;
        if (pblflg && k < kpbl) {
            real dsdzt = tem1 * (-(mf[k] / xkzh[k]));
            r1[k] = r1[k] + dtodsd * dsdzt;
            r1[k + 1] = SH_THX(k + 1) - 300.0f - dtodsu * dsdzt;
        } else if (pblflg && k >= kpbl && entfac[k] < 4.6f) {
            // :1161-1164 -- the entrainment-zone overwrite of xkzh ...
            xkzh[k] = -(we * dza[kpbl] * expf(-entfac[k]));
            xkzh[k] = sqrtf(xkzh[k] * xkzhl[k]);
            xkzh[k] = sh_max(xkzh[k], xkzminh);
            xkzh[k] = sh_min(xkzh[k], xkzmax);
            r1[k + 1] = SH_THX(k + 1) - 300.0f;
        } else {
            r1[k + 1] = SH_THX(k + 1) - 300.0f;
        }
        // :1169 -- ... after which tem1 is RECOMPUTED from the updated
        // coefficient before au/al.  Sequence kept exactly.
        tem1 = dsig * xkzh[k] * rdz;
        real dsdz2 = tem1 * rdz;
        au[k] = -(dtodsd * dsdz2);
        al[k] = -(dtodsu * dsdz2);
        // :1176-1178 -- LOCAL heat partition: h = hpbl (not cslen) and delxy
        // scaled by zfacdx; the NONLOCAL calls above use h = cslen, plain
        // delxy.
        real zfacdx = 0.2f * hpbl / zq[k + 1];
        real delxy_l = sqrtf(dx * dy) * sh_max(zfacdx, 1.0f);
        real pthl1 = sh_pthl(delxy_l, hpbl);
        if (pblflg && k < kpbl) {
            au[k] = au[k] * pthl1;
            al[k] = al[k] * pthl1;
        }
        ad[k] = ad[k] - au[k];
        ad[k + 1] = 1.0f - al[k];
        SH_G(exch_h, k + 1) = xkzh[k];           // :1185, after the overwrite
    }
    sh_tridi_factor(al, ad, au, fkk, lau, kte);  // tridi1n :1198
    sh_tridi_solve(al, fkk, lau, r1, kte);
    for (int k = kte; k >= 1; --k) {             // :1202-1213
        real ttend = (r1[k] - SH_THX(k) + 300.0f) * rdt * SH_PI2DX(k);
        real ttnpk = 0.0f + ttend;               // ttnp zeroed at :612
        // The wrapper's own epilogue (:253): rthblten = ttnp/pi3d.  The
        // divide does NOT round-trip -- (x*pi)/pi is not the float32
        // identity -- and the oracle's rthblten is the divided word, so the
        // kernel performs the same multiply-then-divide.
        SH_G(dtheta, k) = ttnpk / SH_PI2DX(k);
        if (k == kte)
            tflux_e[k] = ttend * SH_DZ8W(k);
        else
            tflux_e[k] = tflux_e[k + 1] + ttend * SH_DZ8W(k);
    }

    // ---- moisture, clouds, ice: assembly (:1217-1320), solve, recover ----
    for (int k = 0; k < SHINHONG_K2; ++k) {
        al[k] = ad[k] = au[k] = 0.0f;
        r1[k] = r2[k] = r3[k] = 0.0f;
    }
    ad[1] = 1.0f;
    r1[1] = SH_QVX(1) + qfxv * G / delp[1] * dt2;
    r2[1] = SH_QCX(1);                           // :1238-1245, ic = 2, 3
    r3[1] = SH_QIX(1);
    for (int k = 1; k <= kte - 1; ++k) {         // :1247-1253
        if (k >= kpbl)
            xkzq[k] = xkzh[k];
    }
    for (int k = 1; k <= kte - 1; ++k) {         // :1255-1293
        real dtodsd = dt2 / delp[k];
        real dtodsu = dt2 / delp[k + 1];
        real dsig = SH_P2DX(k) - SH_P2DX(k + 1);
        real rdz = 1.0f / dza[k + 1];
        real tem1 = dsig * xkzq[k] * rdz;
        if (pblflg && k < kpbl) {
            real dsdzq = tem1 * (-(qfxpbl * zfacent[k] / xkzq[k]));
            r1[k] = r1[k] + dtodsd * dsdzq;
            r1[k + 1] = SH_QVX(k + 1) - dtodsu * dsdzq;
        } else if (pblflg && k >= kpbl && entfac[k] < 4.6f) {
            xkzq[k] = -(we * dza[kpbl] * expf(-entfac[k]));
            xkzq[k] = sqrtf(xkzq[k] * xkzhl[k]);
            xkzq[k] = sh_max(xkzq[k], xkzminh);
            xkzq[k] = sh_min(xkzq[k], xkzmax);
            r1[k + 1] = SH_QVX(k + 1);
        } else {
            r1[k + 1] = SH_QVX(k + 1);
        }
        tem1 = dsig * xkzq[k] * rdz;             // recompute, as at :1275
        real dsdz2 = tem1 * rdz;
        au[k] = -(dtodsd * dsdz2);
        al[k] = -(dtodsu * dsdz2);
        real zfacdx = 0.2f * hpbl / zq[k + 1];   // :1282-1284, h = hpbl
        real delxy_l = sqrtf(dx * dy) * sh_max(zfacdx, 1.0f);
        real pq1_l = sh_pq(delxy_l, hpbl);
        if (pblflg && k < kpbl) {
            au[k] = au[k] * pq1_l;
            al[k] = al[k] * pq1_l;
        }
        ad[k] = ad[k] - au[k];
        ad[k + 1] = 1.0f - al[k];
    }
    for (int k = 1; k <= kte - 1; ++k) {         // :1295-1304
        r2[k + 1] = SH_QCX(k + 1);
        r3[k + 1] = SH_QIX(k + 1);
    }
    // :1324 -- one tridin_ysu call, ndiff = 3: all three components share
    // the same au/al/ad built from xkzq (one LU factorization).
    sh_tridi_factor(al, ad, au, fkk, lau, kte);
    sh_tridi_solve(al, fkk, lau, r1, kte);
    sh_tridi_solve(al, fkk, lau, r2, kte);
    sh_tridi_solve(al, fkk, lau, r3, kte);
    for (int k = kte; k >= 1; --k) {             // :1328-1340
        real qtend = (r1[k] - SH_QVX(k)) * rdt;
        SH_G(dqv, k) = 0.0f + qtend;             // qtnp zeroed at :613
        if (k == kte)
            qflux_e[k] = qtend * SH_DZ8W(k);
        else
            qflux_e[k] = qflux_e[k + 1] + qtend * SH_DZ8W(k);
        tvflux_e[k] = tflux_e[k] + qflux_e[k] * ep1 * SH_THX(k);
    }
    for (int k = 1; k <= kte; ++k) {             // :1342-1356
        if (pblflg && k < kpbl) {
            real hgame_c = c_1 * 0.2f * 2.5f * (G / thvx[k]) * wstar
                         / (0.25f * (q2x[k + 1] + q2x[k]));
            hgame_c = sh_min(hgame_c, gamcre);
            // the k == kte arm (:1347) is unreachable: k < kpbl <= kte
            hgame2d[k] = hgame_c * 0.5f * (tvflux_e[k] + tvflux_e[k + 1])
                       * hpbl;
            hgame2d[k] = sh_max(hgame2d[k], 0.0f);
        }
    }
    for (int k = kte; k >= 1; --k)               // :1358-1370, ic = 2
        SH_G(dqc, k) = 0.0f + (r2[k] - SH_QCX(k)) * rdt;
    for (int k = kte; k >= 1; --k)               // :1358-1370, ic = 3
        SH_G(dqi, k) = 0.0f + (r3[k] - SH_QIX(k)) * rdt;

    // ---- momentum: assembly (:1374-1436), solve, recover ----
    for (int k = 0; k < SHINHONG_K2; ++k) {
        al[k] = ad[k] = au[k] = 0.0f;
        r1[k] = r2[k] = 0.0f;
    }
    real s = wspd1 / wspdv;                      // no floor: WRF has none, and
                                                 // the fixture's 1e-10 wspd
                                                 // probe would measure one
    // :1387 -- both arms run this; the ctopo multiply is the arm-A identity
    // omitted per the header comment.
    ad[1] = 1.0f + ustv * ustv / wspd1 * rhox * G / delp[1] * dt2 * (s * s);
    r1[1] = SH_UX(1);
    r2[1] = SH_VX(1);
    for (int k = 1; k <= kte - 1; ++k) {
        real dtodsd = dt2 / delp[k];
        real dtodsu = dt2 / delp[k + 1];
        real dsig = SH_P2DX(k) - SH_P2DX(k + 1);
        real rdz = 1.0f / dza[k + 1];
        real tem1 = dsig * xkzm[k] * rdz;
        if (pblflg && k < kpbl) {
            real dsdzu = tem1 * (-(hgamu / hpbl)
                                 - ufxpbl * zfacent[k] / xkzm[k]);
            real dsdzv = tem1 * (-(hgamv / hpbl)
                                 - vfxpbl * zfacent[k] / xkzm[k]);
            r1[k] = r1[k] + dtodsd * dsdzu;
            r1[k + 1] = SH_UX(k + 1) - dtodsu * dsdzu;
            r2[k] = r2[k] + dtodsd * dsdzv;
            r2[k + 1] = SH_VX(k + 1) - dtodsu * dsdzv;
        } else if (pblflg && k >= kpbl && entfac[k] < 4.6f) {
            xkzm[k] = prpbl * xkzh[k];
            xkzm[k] = sqrtf(xkzm[k] * xkzml[k]);
            xkzm[k] = sh_max(xkzm[k], xkzminm);
            xkzm[k] = sh_min(xkzm[k], xkzmax);
            r1[k + 1] = SH_UX(k + 1);
            r2[k + 1] = SH_VX(k + 1);
        } else {
            r1[k + 1] = SH_UX(k + 1);
            r2[k + 1] = SH_VX(k + 1);
        }
        tem1 = dsig * xkzm[k] * rdz;             // recompute, as at :1419
        real dsdz2 = tem1 * rdz;
        au[k] = -(dtodsd * dsdz2);
        al[k] = -(dtodsu * dsdz2);
        real zfacdx = 0.2f * hpbl / zq[k + 1];   // :1426-1428, h = hpbl
        real delxy_l = sqrtf(dx * dy) * sh_max(zfacdx, 1.0f);
        real pu1_l = sh_pu(delxy_l, hpbl);
        if (pblflg && k < kpbl) {
            au[k] = au[k] * pu1_l;
            al[k] = al[k] * pu1_l;
        }
        ad[k] = ad[k] - au[k];
        ad[k + 1] = 1.0f - al[k];
    }
    sh_tridi_factor(al, ad, au, fkk, lau, kte);  // tridi1n :1450
    sh_tridi_solve(al, fkk, lau, r1, kte);
    sh_tridi_solve(al, fkk, lau, r2, kte);
    for (int k = kte; k >= 1; --k) {             // :1454-1463
        SH_G(du, k) = 0.0f + (r1[k] - SH_UX(k)) * rdt;
        SH_G(dv, k) = 0.0f + (r2[k] - SH_VX(k)) * rdt;
    }

    // ---- 10 m wind blend (:1471-1474): with ctopo2 = 1 the blend returns
    // its input bitwise (the arm-A identity, see the header); no outputs.

    // ---- SGS TKE diagnostics (:1478-1608) ----
    if (tke_diag == 1) {
        real q2xk[SHINHONG_K2], ptke1[SHINHONG_K2];
        real akmk[SHINHONG_K2], akhk[SHINHONG_K2], mfk[SHINHONG_K2];
        real ufxpblk[SHINHONG_K2], vfxpblk[SHINHONG_K2], qfxpblk[SHINHONG_K2];
        real s2k[SHINHONG_K2], rigk[SHINHONG_K2], elk[SHINHONG_K2];
        for (int k = 0; k < SHINHONG_K2; ++k) {
            q2xk[k] = akmk[k] = akhk[k] = mfk[k] = 0.0f;
            ufxpblk[k] = vfxpblk[k] = qfxpblk[k] = 0.0f;
            s2k[k] = rigk[k] = elk[k] = 0.0f;
            ptke1[k] = 1.0f;                     // :1509
        }
        for (int k = 1; k <= kte - 1; ++k) {     // :1530-1536
            if (pblflg && k <= kpbl) {           // k .le. kpbl, and za(k),
                real zfacdx = 0.2f * hpbl / za[k];   // not zq -- unlike the
                real delxy_l = sqrtf(dx * dy)        // tri loops
                             * sh_max(zfacdx, 1.0f);
                ptke1[k + 1] = sh_ptke(delxy_l, hpbl);
            }
        }
        for (int k = 2; k <= kte; ++k) {         // :1546-1553
            akmk[k] = xkzm[k - 1];
            akhk[k] = xkzh[k - 1];
            mfk[k] = mf[k - 1] / xkzh[k - 1];
            // WRF reads zfacent at k-1 >= kpbl here, where it never assigned
            // it (stack garbage); every consumer is guarded k <= kpbl so the
            // garbage is dead.  This port's zfacent is 0 there instead.
            ufxpblk[k] = ufxpbl * zfacent[k - 1] / xkzm[k - 1];
            vfxpblk[k] = vfxpbl * zfacent[k - 1] / xkzm[k - 1];
            qfxpblk[k] = qfxpbl * zfacent[k - 1] / xkzq[k - 1];
        }
        for (int k = 1; k <= kte; ++k)
            q2xk[k] = q2x[k];
        if (pblflg && kpbl < kte) {
            // :1555-1559.  DOCUMENTED DIVERGENCE: WRF reads q2xk(kpbl+1)
            // with no guard, which is one past the end when kpbl == kte
            // (:1557).  The port computes efxpbl only for kpbl < kte -- the
            // defined behaviour -- and the registry warning covers it
            // (Phase D).  The fixture never reaches the undefined arm (case
            // 22 runs diag off).
            int kk = kpbl - 1;
            real dex = 0.25f * (q2xk[kk + 2] - q2xk[kk]);
            efxpbl = we * dex;
        }
        delxy = sqrtf(dx * dy);                  // :1561
        sh_mixlen(u, v, theta, exner, qv, qc, st, col,
                  q2xk, zq, ustv, corfv, epshol, hpbl, kpbl, pblflg,
                  hgamu, hgamv, hgamq, mfk, ufxpblk, vfxpblk, qfxpblk, kte,
                  s2k, rigk, elk);
        sh_prodq2(dt, ustv, u, v, theta, st, col, thvx,
                  q2xk, elk, zq, akmk, akhk,
                  hgamu, hgamv, hgamq, delxy, hpbl, pblflg, kpbl,
                  mfk, ufxpblk, vfxpblk, qfxpblk, kte);
        sh_vdifq(dt, q2xk, zq, akhk, ptke1, hgame2d, hpbl, pblflg, kpbl,
                 efxpbl, kte);
        for (int k = 1; k <= kte; ++k) {         // :1601-1605
            q2x[k] = sh_max(q2xk[k], epsq2l);    // amax1: NaN propagates
            SH_G(tke_out, k) = 0.5f * q2x[k];
            if (k != 1)
                SH_G(el, k) = elk[k];            // el(kts) stays the :673 zero
        }
    }

    hpbl_out[col] = hpbl;
    kpbl_out[col] = kpbl;                        // the 1-based Fortran index
    wstar_out[col] = wstar;
    delta_out[col] = delta;
}

// ------------------------------------------------------------------------
// Direct probe of the five partition functions, mirroring the oracle's
// shinhong-partition.csv d x h grid.  This is the guaranteed-firing negative
// control for the GPU gate: device powf is not glibc's correctly rounded
// powf, so the probe pins exactly how far the device curves sit from the
// words WRF wrote, without the column dynamics in between.
// ------------------------------------------------------------------------

extern "C" __global__
void shinhong_partition_probe(const real *d, const real *h,
                              real *pu, real *pq, real *pthnl,
                              real *pthl, real *ptke, long long n) {
    long long i = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    pu[i] = sh_pu(d[i], h[i]);
    pq[i] = sh_pq(d[i], h[i]);
    pthnl[i] = sh_pthnl(d[i], h[i]);
    pthl[i] = sh_pthl(d[i], h[i]);
    ptke[i] = sh_ptke(d[i], h[i]);
}
