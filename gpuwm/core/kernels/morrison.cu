// WRF v4.6.1 Morrison two-moment microphysics (mp_physics=10): a literal
// transcription of phys/module_mp_morr_two_moment.F (tag v4.6.1, d66e442f,
// SHA-256 2dcd1c70...): the MP_MORR_TWO_MOMENT driver :563-925 (INUM=1,
// no WRF-Chem), MORR_TWO_MOMENT_MICRO :929-4062, POLYSVP :4066-4149, the
// module GAMMA :4153-4384 and refl10cm_hm :4502-4675, with the words WRF's
// own MORR_TWO_MOMENT_INIT and radar_init compute (morrison_constants.cuh,
// dumped from the WRF build, both morr_rimed_ice modes).
//
// Word-for-word discipline (graded against WRF's own Fortran by
// tools/morrison_wrf461_oracle/validate_morrison_oracle.py):
// * every expression in gfortran's evaluation order and precision (the
//   oracle's -O0 GIMPLE is the reference): REAL operations in float, a
//   DOUBLE PRECISION operand promotes the rest of the expression,
//   assignment rounds;
// * x**r with a real exponent is the libm powf/pow call gfortran makes
//   (WOOF's own float32/binary64 libm: gfk_pow, glibc_pow); x**2 is x*x;
//   x**3 and x**4 are the __powisf2 multiply chains gfortran emits;
//   EXP/LOG/ALOG10/SIN are WOOF's own libm routines;
// * Fortran MAX(a,b) is gfortran's MAX_EXPR<b,a> (a literal operand is
//   always the second); measured on the oracle's own build (gfortran 13.3
//   -O0, x86-64 maxss/minss), MAX(a, b) of two variables returns
//   a > b ? a : b and MAX with a literal k returns k > x ? k : x, so a NaN
//   or the second zero wins as WRF's does (FMAX/FMAXK below);
// * the unit compiles with --fmad=false (kernels/__init__.py
//   _NO_FMAD_MODULES) and every division is __fdiv_rn/__ddiv_rn (FDIV/
//   DDIV): NVRTC compiles x / constant as a reciprocal multiply even under
//   --prec-div=true;
// * WRF's structure: the per-level process stage accumulates float
//   tendencies, sedimentation runs on split steps, and the state is
//   written by the final loop, X3D + X3DTEN*DT.
//
// The one ruled divergence (owner ruling, PARITY-GAPS row 37) is compiled
// in by MORR_ROW37 (defined below, production): the deposition-nucleation
// vapour cap and the wide re-evaluation of a non-finite cloud or rain
// freezing rate.  Each is a named block; with MORR_ROW37 undefined the unit
// is WRF's arithmetic only, which is what the oracle gate grades.

#ifndef MORR_ROW37_OFF
#define MORR_ROW37 1
#endif

#define MORR_KMAX_GENERIC 256
#define MORR_KMAX_SHALLOW 64

// Fortran MAX/MIN as WRF's gfortran build evaluates them (measured,
// see the header): two variables, MAX(a, b) = a > b ? a : b; one literal
// k, either position, MAX = k > x ? k : x.  MIN likewise with <.
#define GMAX(x, y) ((x) > (y) ? (x) : (y))
#define GMIN(x, y) ((x) < (y) ? (x) : (y))
#define FMAX(a, b) GMAX((a), (b))
#define FMIN(a, b) GMIN((a), (b))
#define FMAXK(x, k) GMAX((k), (x))
#define FMINK(x, k) GMIN((k), (x))

// IEEE float add, subtract, multiply and divide WITH subnormals.  CuPy compiles every
// unit with -ftz=true, so a plain a + b flushes a subnormal operand or
// result to zero; WRF's gfortran build keeps it.  The state carries
// subnormal mixing ratios (a column subsaturated to its last bit of
// vapour) and sedimentation moves subnormal fluxes, so the operations
// that write the state, the fallout and the surface sums are spelled as
// PTX without .ftz: these are the IEEE operations gfortran performs.
__device__ __forceinline__ float mo_add(float a, float b)
{
    float r;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}
__device__ __forceinline__ float mo_sub(float a, float b)
{
    float r;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}
__device__ __forceinline__ float mo_mul(float a, float b)
{
    float r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}
__device__ __forceinline__ float mo_div(float a, float b)
{
    float r;
    asm("div.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

// libgcc __powisf2 for the constant exponents gfortran passes it.
__device__ __forceinline__ float mo_powi3(float x) { return x * (x * x); }
__device__ __forceinline__ float mo_powi4(float x)
{
    const float x2 = x * x;
    return x2 * x2;
}

// ---------------------------------------------------------------------
// POLYSVP :4066-4149 (Flatau et al. 1992 polynomials; Goff-Gratch below
// 195.8 K ice / 202 K liquid).
// ---------------------------------------------------------------------
__device__ float mo_polysvp(float t, int type)
{
    float r = 0.0f;
    if (type == 1) {
        if (t >= 195.8f) {
            const float dt = t - 273.15f;
            r = 6.11147274f + dt * (0.503160820f + dt * (0.188439774e-1f
                + dt * (0.420895665e-3f + dt * (0.615021634e-5f
                + dt * (0.602588177e-7f + dt * (0.385852041e-9f
                + dt * (0.146898966e-11f + 0.252751365e-14f * dt)))))));
            r = r * 100.0f;
        } else {
            const float a = -((FDIV(273.16f, t) - 1.0f) * 9.09718f);
            const float b = gfk_log10(FDIV(273.16f, t)) * 3.56654f;
            const float c = (1.0f - FDIV(t, 273.16f)) * 0.876793f;
            // alog10(6.1071) folded by gfortran.
            r = gfk_pow(10.0f, ((a - b) + c) + 7.858350276947021484375e-1f)
                * 100.0f;
        }
    }
    if (type == 0) {
        if (t >= 202.0f) {
            const float dt = t - 273.15f;
            r = 6.11239921f + dt * (0.443987641f + dt * (0.142986287e-1f
                + dt * (0.264847430e-3f + dt * (0.302950461e-5f
                + dt * (0.206739458e-7f + dt * (0.640689451e-10f
                + dt * (-0.952447341e-13f + -0.976195544e-15f * dt)))))));
            r = r * 100.0f;
        } else {
            const float a = gfk_log10(FDIV(373.16f, t)) * 5.02808f
                - (FDIV(373.16f, t) - 1.0f) * 7.90298f;
            const float b = (gfk_pow(10.0f, (1.0f - FDIV(t, 373.16f))
                                     * 11.344f) - 1.0f) * 1.3816e-7f;
            const float c = (gfk_pow(10.0f, -((FDIV(373.16f, t) - 1.0f)
                                              * 3.49149f)) - 1.0f)
                * 8.1328e-3f;
            // alog10(1013.246) folded by gfortran.
            r = gfk_pow(10.0f, ((a - b) + c) + 3.005714893341064453125f)
                * 100.0f;
        }
    }
    return r;
}

// ---------------------------------------------------------------------
// GAMMA :4153-4384 (W. J. Cody's REAL gamma, the module's own).
// ---------------------------------------------------------------------
__device__ __constant__ float mo_gam_p[8] = {
    -1.716185092926025390625e+0f, 2.4765651702880859375e+1f,
    -3.7980426025390625e+2f, 6.293311767578125e+2f,
    8.669661865234375e+2f, -3.14512734375e+4f,
    -3.61444140625e+4f, 6.6456140625e+4f};
__device__ __constant__ float mo_gam_q[8] = {
    -3.08402309417724609375e+1f, 3.15350616455078125e+2f,
    -1.0151563720703125e+3f, -3.107771728515625e+3f,
    2.2538119140625e+4f, 4.75584619140625e+3f,
    -1.34659953125e+5f, -1.151322578125e+5f};
__device__ __constant__ float mo_gam_c[7] = {
    -1.91044411621987819671630859375e-3f,
    8.41713859699666500091552734375e-4f,
    -5.952380015514791011810302734375e-4f,
    7.936508045531809329986572265625e-4f,
    -2.77777784503996372222900390625e-3f,
    8.3333335816860198974609375e-2f,
    5.70838339626789093017578125e-3f};

__device__ float mo_gamma(float x)
{
    const float XBIG = 3.504000091552734375e+1f;
    const float XMININ = 1.17999994577463109229687760495139774515557098299380308267e-38f;
    const float EPS = 1.1899999918796311249025166034698486328125e-7f;
    const float XINF = 3.39999995214436424907732413799364296704e+38f;
    bool parity = false;
    float fact = 1.0f;
    int n = 0;
    float y = x, res;
    if (y <= 0.0f) {
        y = -x;
        const float y1 = truncf(y);
        res = y - y1;
        if (res != 0.0f) {
            if (y1 != truncf(y1 * 0.5f) * 2.0f) parity = true;
            fact = -FDIV(MO_PI, glibc_sinf(res * MO_PI));
            y = y + 1.0f;
        } else {
            return XINF;
        }
    }
    if (y < EPS) {
        if (y >= XMININ) {
            res = FDIV(1.0f, y);
        } else {
            return XINF;
        }
    } else if (y < 12.0f) {
        const float y1 = y;
        float z;
        if (y < 1.0f) {
            z = y;
            y = y + 1.0f;
        } else {
            n = (int)y - 1;
            y = y - (float)n;
            z = y - 1.0f;
        }
        float xnum = 0.0f, xden = 1.0f;
        for (int i = 0; i < 8; ++i) {
            xnum = (xnum + mo_gam_p[i]) * z;
            xden = xden * z + mo_gam_q[i];
        }
        res = FDIV(xnum, xden) + 1.0f;
        if (y1 < y) {
            res = FDIV(res, y1);
        } else if (y1 > y) {
            for (int i = 1; i <= n; ++i) {
                res = res * y;
                y = y + 1.0f;
            }
        }
    } else {
        if (y <= XBIG) {
            const float ysq = y * y;
            float sum = mo_gam_c[6];
            for (int i = 0; i < 6; ++i) sum = FDIV(sum, ysq) + mo_gam_c[i];
            sum = (FDIV(sum, y) - y) + MO_XXX;
            sum = sum + (y - 0.5f) * gfk_log(y);
            res = gfk_exp(sum);
        } else {
            return XINF;
        }
    }
    if (parity) res = -res;
    if (fact != 1.0f) res = FDIV(fact, res);
    return res;
}

// ---------------------------------------------------------------------
// Scratch layout.  A call covers the columns [col0, col0 + ccols) of the
// (nz, ny, nx) state (ncol = ny * nx columns); WORK holds MO_NWORK planes
// of (nz, ccols) for those columns only, plane p of chunk cell
// li = k * ccols + (col - col0) at work[p * nlocal + li], nlocal =
// nz * ccols.  Every WRF column is independent, so a chunk is WRF's own
// column call: the arithmetic of a column does not depend on which chunk
// carries it (col0 = 0, ccols = ncol is the whole domain in one call).
// ---------------------------------------------------------------------
#define MO_T3DTEN 0
#define MO_QV3DTEN 1
#define MO_QC3DTEN 2
#define MO_QI3DTEN 3
#define MO_QNI3DTEN 4
#define MO_QR3DTEN 5
#define MO_NI3DTEN 6
#define MO_NS3DTEN 7
#define MO_NR3DTEN 8
#define MO_QG3DTEN 9
#define MO_NG3DTEN 10
#define MO_NC3DTEN 11
#define MO_T3D 12       // T3D after the process stage
#define MO_CPM 13
#define MO_LAMI 14      // LAMI(K) as the process stage left it
#define MO_ACN 15
#define MO_QRSTEN 16
#define MO_QISTEN 17
#define MO_QNISTEN 18
#define MO_QCSTEN 19
#define MO_QGSTEN 20
#define MO_NWORK 21

// ---------------------------------------------------------------------
// The per-level body of MORR_TWO_MOMENT_MICRO's first K loop
// (:1275-3334).  One thread per cell; LTRUE is OR-ed per column.
// ---------------------------------------------------------------------
extern "C" __global__ void mo_process(
    const float* __restrict__ theta, const float* __restrict__ pii,
    const float* __restrict__ pressure,
    float* __restrict__ qv, float* __restrict__ qc, float* __restrict__ qr,
    float* __restrict__ qi, float* __restrict__ qs, float* __restrict__ qg,
    float* __restrict__ nc, float* __restrict__ nr, float* __restrict__ ni,
    float* __restrict__ ns, float* __restrict__ ng,
    const float* __restrict__ qrcuten, const float* __restrict__ qscuten,
    const float* __restrict__ qicuten, int has_cu,
    float* __restrict__ rho_out, float* __restrict__ work,
    int* __restrict__ ltrue, float dt, int rimed, long long nlocal,
    int ncol, int col0, int ccols)
{
    const long long li = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (li >= nlocal) return;
    const int col = col0 + (int)(li % ccols);
    const long long idx = (li / ccols) * ncol + col;
#define W(p) work[(long long)(p) * nlocal + li]

    // Driver :744, :778-802 (nc1d = 0 placeholder, iinum = 1).
    float t3d = theta[idx] * pii[idx];
    float qv3d = qv[idx], qc3d = qc[idx], qr3d = qr[idx], qi3d = qi[idx];
    float qni3d = qs[idx], qg3d = qg[idx];
    float nc3d = 0.0f, nr3d = nr[idx], ni3d = ni[idx], ns3d = ns[idx];
    float ng3d = ng[idx];
    const float pres = pressure[idx];
    float t3dten = 0.0f, qv3dten = 0.0f, qc3dten = 0.0f, qi3dten = 0.0f;
    float qni3dten = 0.0f, qr3dten = 0.0f, ni3dten = 0.0f, ns3dten = 0.0f;
    float nr3dten = 0.0f, qg3dten = 0.0f, ng3dten = 0.0f, nc3dten = 0.0f;
    float lami = 0.0f;
    bool active = false;

    // :1298-1325
    const float xxlv = 3.1484E6f - t3d * 2370.0f;
    const float xxls = (3.15E6f - t3d * 2370.0f) + 0.3337E6f;
    const float cpm = (qv3d * 0.887f + 1.0f) * MO_CP;
    float evs = FMIN(pres * 0.99f, mo_polysvp(t3d, 0));
    float eis = FMIN(pres * 0.99f, mo_polysvp(t3d, 1));
    if (eis > evs) eis = evs;
    float qvs = FDIV(evs * MO_EP_2, pres - evs);
    float qvi = FDIV(eis * MO_EP_2, pres - eis);
    float qvqvs = FDIV(qv3d, qvs);
    float qvqvsi = FDIV(qv3d, qvi);
    const float rho = FDIV(pres, t3d * MO_R);

    // :1332-1343 cumulus-detrained number.
    if (has_cu) {
        const float qrcu1d = qrcuten[idx], qscu1d = qscuten[idx];
        const float qicu1d = qicuten[idx];
        if (qrcu1d >= 1.E-10f) {
            const float dum = gfk_pow(FDIV(qrcu1d * dt,
                                           (MO_RHOW * MO_PI) * mo_powi3(rho)),
                                      0.25f) * 1.8e5f;
            nr3d = nr3d + dum;
        }
        if (qscu1d >= 1.E-10f) {
            const float dum = gfk_pow(FDIV(qscu1d * dt,
                                           mo_powi3(rho) * MO_CONS1),
                                      FDIV(1.0f, MO_DS + 1.0f)) * 3.e5f;
            ns3d = ns3d + dum;
        }
        if (qicu1d >= 1.E-10f) {
            const float dum = FDIV(qicu1d * dt,
                                   gfk_pow(80.E-6f, MO_DI) * MO_CI);
            ni3d = ni3d + dum;
        }
    }

    // :1348-1377 evaporate/sublimate small amounts when subsaturated.
    if (qvqvs < 0.9f) {
        if (qr3d < 1.E-8f) {
            qv3d = mo_add(qv3d, qr3d);
            t3d = t3d - FDIV(qr3d * xxlv, cpm);
            qr3d = 0.0f;
        }
        if (qc3d < 1.E-8f) {
            qv3d = mo_add(qv3d, qc3d);
            t3d = t3d - FDIV(qc3d * xxlv, cpm);
            qc3d = 0.0f;
        }
    }
    if (qvqvsi < 0.9f) {
        if (qi3d < 1.E-8f) {
            qv3d = mo_add(qv3d, qi3d);
            t3d = t3d - FDIV(qi3d * xxls, cpm);
            qi3d = 0.0f;
        }
        if (qni3d < 1.E-8f) {
            qv3d = mo_add(qv3d, qni3d);
            t3d = t3d - FDIV(qni3d * xxls, cpm);
            qni3d = 0.0f;
        }
        if (qg3d < 1.E-8f) {
            qv3d = mo_add(qv3d, qg3d);
            t3d = t3d - FDIV(qg3d * xxls, cpm);
            qg3d = 0.0f;
        }
    }
    const float xlf = xxls - xxlv;

    // :1386-1410 (effective radii are the final loop's).
    if (qc3d < MO_QSMALL) { qc3d = 0.0f; nc3d = 0.0f; }
    if (qr3d < MO_QSMALL) { qr3d = 0.0f; nr3d = 0.0f; }
    if (qi3d < MO_QSMALL) { qi3d = 0.0f; ni3d = 0.0f; }
    if (qni3d < MO_QSMALL) { qni3d = 0.0f; ns3d = 0.0f; }
    if (qg3d < MO_QSMALL) { qg3d = 0.0f; ng3d = 0.0f; }

    // :1424-1443
    const float mu = FDIV(gfk_pow(t3d, 1.5f) * 1.496E-6f, t3d + 120.0f);
    float dum = gfk_pow(FDIV(MO_RHOSU, rho), 0.54f);
    const float ain = gfk_pow(FDIV(MO_RHOSU, rho), 0.35f) * MO_AI;
    const float arn = dum * MO_AR;
    const float asn = dum * MO_AS;
    const float acn = FDIV(MO_RHOW * MO_G, mu * 18.0f);
    const float agn = dum * MO_AG;
    (void)ain;

    // :1449-1453
    bool skip = false;
    if (qc3d < MO_QSMALL && qi3d < MO_QSMALL && qni3d < MO_QSMALL
            && qr3d < MO_QSMALL && qg3d < MO_QSMALL) {
        if (t3d < 273.15f && qvqvsi < 0.999f) skip = true;
        if (!skip && t3d >= 273.15f && qvqvs < 0.999f) skip = true;
    }
    if (!skip) {
        active = true;
        // :1458-1479
        const float kap = mu * 1.414E3f;
        const float dv = FDIV(gfk_pow(t3d, 1.81f) * 8.794E-5f, pres);
        const float sc = FDIV(mu, rho * dv);
        dum = (t3d * t3d) * MO_RV;
        const float dqsdt = FDIV(xxlv * qvs, dum);
        const float dqsidt = FDIV(xxls * qvi, dum);
        const float abi = FDIV(dqsidt * xxls, cpm) + 1.0f;
        const float ab = FDIV(dqsdt * xxlv, cpm) + 1.0f;

        float lamr = 0.0f, n0rr = 0.0f, lamc = 0.0f, pgam = 0.0f;
        float lams = 0.0f, n0s = 0.0f, lamg = 0.0f, n0g = 0.0f, n0i = 0.0f;
        float cdist1 = 0.0f;
        float lammin, lammax;
        float epsr, epss, epsg, epsi, ratio;

        if (t3d >= 273.15f) {
            // ----------------------------------------- :1493-2076 warm
            nc3d = FDIV(MO_NDCNST * 1.E6f, rho);
            // :1501-1514 melt very small snow and graupel.
            if (qni3d < 1.E-6f) {
                qr3d = qr3d + qni3d;
                nr3d = nr3d + ns3d;
                t3d = t3d - FDIV(qni3d * xlf, cpm);
                qni3d = 0.0f;
                ns3d = 0.0f;
            }
            if (qg3d < 1.E-6f) {
                qr3d = qr3d + qg3d;
                nr3d = nr3d + ng3d;
                t3d = t3d - FDIV(qg3d * xlf, cpm);
                qg3d = 0.0f;
                ng3d = 0.0f;
            }
            float pcc;
            if (qc3d < MO_QSMALL && qni3d < 1.E-8f && qr3d < MO_QSMALL
                    && qg3d < 1.E-8f) {
                // GOTO 300
            } else {
                ns3d = FMAXK(ns3d, 0.0f);
                nc3d = FMAXK(nc3d, 0.0f);
                nr3d = FMAXK(nr3d, 0.0f);
                ng3d = FMAXK(ng3d, 0.0f);
                // :1528-1549 rain
                if (qr3d >= MO_QSMALL) {
                    lamr = gfk_pow(FDIV((MO_RHOW * MO_PI) * nr3d, qr3d),
                                   0.333333343f);
                    n0rr = nr3d * lamr;
                    if (lamr < MO_LAMMINR) {
                        lamr = MO_LAMMINR;
                        n0rr = FDIV(mo_powi4(lamr) * qr3d,
                                    MO_RHOW * MO_PI);
                        nr3d = FDIV(n0rr, lamr);
                    } else if (lamr > MO_LAMMAXR) {
                        lamr = MO_LAMMAXR;
                        n0rr = FDIV(mo_powi4(lamr) * qr3d,
                                    MO_RHOW * MO_PI);
                        nr3d = FDIV(n0rr, lamr);
                    }
                }
                // :1556-1588 cloud droplets
                if (qc3d >= MO_QSMALL) {
                    dum = FDIV(pres, t3d * 287.15f);
                    pgam = (dum * FDIV(nc3d, 1.E6f)) * 0.0005714f + 0.2714f;
                    pgam = FDIV(1.0f, pgam * pgam) - 1.0f;
                    pgam = FMAXK(pgam, 2.0f);
                    pgam = FMINK(pgam, 10.0f);
                    lamc = gfk_pow(FDIV((nc3d * MO_CONS26)
                                        * mo_gamma(pgam + 4.0f),
                                        qc3d * mo_gamma(pgam + 1.0f)),
                                   0.333333343f);
                    lammin = FDIV(pgam + 1.0f, 60.E-6f);
                    lammax = FDIV(pgam + 1.0f, 1.E-6f);
                    if (lamc < lammin) {
                        lamc = lammin;
                        nc3d = FDIV(gfk_exp(((gfk_log(lamc) * 3.0f
                                              + gfk_log(qc3d))
                                             + gfk_log(mo_gamma(pgam + 1.0f)))
                                            - gfk_log(mo_gamma(pgam + 4.0f))),
                                    MO_CONS26);
                    } else if (lamc > lammax) {
                        lamc = lammax;
                        nc3d = FDIV(gfk_exp(((gfk_log(lamc) * 3.0f
                                              + gfk_log(qc3d))
                                             + gfk_log(mo_gamma(pgam + 1.0f)))
                                            - gfk_log(mo_gamma(pgam + 4.0f))),
                                    MO_CONS26);
                    }
                }
                // :1593-1614 snow
                if (qni3d >= MO_QSMALL) {
                    lams = gfk_pow(FDIV(ns3d * MO_CONS1, qni3d),
                                   FDIV(1.0f, MO_DS));
                    n0s = ns3d * lams;
                    if (lams < MO_LAMMINS) {
                        lams = MO_LAMMINS;
                        n0s = FDIV(mo_powi4(lams) * qni3d, MO_CONS1);
                        ns3d = FDIV(n0s, lams);
                    } else if (lams > MO_LAMMAXS) {
                        lams = MO_LAMMAXS;
                        n0s = FDIV(mo_powi4(lams) * qni3d, MO_CONS1);
                        ns3d = FDIV(n0s, lams);
                    }
                }
                // :1619-1638 graupel
                if (qg3d >= MO_QSMALL) {
                    lamg = gfk_pow(FDIV(ng3d * MO_CONS2, qg3d),
                                   FDIV(1.0f, MO_DG));
                    n0g = ng3d * lamg;
                    if (lamg < MO_LAMMING) {
                        lamg = MO_LAMMING;
                        n0g = FDIV(mo_powi4(lamg) * qg3d, MO_CONS2);
                        ng3d = FDIV(n0g, lamg);
                    } else if (lamg > MO_LAMMAXG) {
                        lamg = MO_LAMMAXG;
                        n0g = FDIV(mo_powi4(lamg) * qg3d, MO_CONS2);
                        ng3d = FDIV(n0g, lamg);
                    }
                }
                // :1643-1664
                float prc = 0.0f, nprc = 0.0f, nprc1 = 0.0f, pra = 0.0f;
                float npra = 0.0f, nragg = 0.0f, nsmlts = 0.0f;
                float nsmltr = 0.0f, evpms = 0.0f, pre = 0.0f;
                float nsubr = 0.0f, pracg = 0.0f, npracg = 0.0f;
                float psmlt = 0.0f, pgmlt = 0.0f, evpmg = 0.0f, pracs = 0.0f;
                float npracs = 0.0f, ngmltg = 0.0f, ngmltr = 0.0f;
                float ums, umr, uns, unr, umg, ung;
                // :1679-1697 autoconversion
                if (qc3d >= 1.E-6f) {
                    prc = (gfk_pow(qc3d, 2.47f) * 1350.0f)
                        * gfk_pow(FDIV(nc3d, 1.e6f) * rho, -1.79f);
                    nprc1 = FDIV(prc, MO_CONS29);
                    nprc = FDIV(prc, FDIV(qc3d, nc3d));
                    nprc = FMIN(nprc, FDIV(nc3d, dt));
                    nprc1 = FMIN(nprc1, nprc);
                }
                // :1703-1743 rain-snow collection above freezing
                if (qr3d >= 1.E-8f && qni3d >= 1.E-8f) {
                    ums = FDIV(asn * MO_CONS3, gfk_pow(lams, MO_BS));
                    umr = FDIV(arn * MO_CONS4, gfk_pow(lamr, MO_BR));
                    uns = FDIV(asn * MO_CONS5, gfk_pow(lams, MO_BS));
                    unr = FDIV(arn * MO_CONS6, gfk_pow(lamr, MO_BR));
                    dum = gfk_pow(FDIV(MO_RHOSU, rho), 0.54f);
                    ums = FMIN(ums, dum * 1.2f);
                    uns = FMIN(uns, dum * 1.2f);
                    umr = FMIN(umr, dum * 9.1f);
                    unr = FMIN(unr, dum * 9.1f);
                    const float d = umr * 1.2f - ums * 0.95f;
                    pracs = (FDIV(((gfk_pow(d * d + (ums * 0.08f) * umr, 0.5f)
                                    * rho) * n0rr) * n0s, mo_powi3(lamr))
                             * ((FDIV(5.0f, mo_powi3(lamr) * lams)
                                 + FDIV(2.0f, (lamr * lamr) * (lams * lams)))
                                + FDIV(0.5f, lamr * mo_powi3(lams))))
                        * MO_CONS41;
                }
                // :1749-1787 rain-graupel collection above freezing
                if (qr3d >= 1.E-8f && qg3d >= 1.E-8f) {
                    umg = FDIV(agn * MO_CONS7, gfk_pow(lamg, MO_BG));
                    umr = FDIV(arn * MO_CONS4, gfk_pow(lamr, MO_BR));
                    ung = FDIV(agn * MO_CONS8, gfk_pow(lamg, MO_BG));
                    unr = FDIV(arn * MO_CONS6, gfk_pow(lamr, MO_BR));
                    dum = gfk_pow(FDIV(MO_RHOSU, rho), 0.54f);
                    umg = FMIN(umg, dum * 20.0f);
                    ung = FMIN(ung, dum * 20.0f);
                    umr = FMIN(umr, dum * 9.1f);
                    unr = FMIN(unr, dum * 9.1f);
                    const float d = umr * 1.2f - umg * 0.95f;
                    pracg = (FDIV(((gfk_pow(d * d + (umg * 0.08f) * umr, 0.5f)
                                    * rho) * n0rr) * n0g, mo_powi3(lamr))
                             * ((FDIV(5.0f, mo_powi3(lamr) * lamg)
                                 + FDIV(2.0f, (lamr * lamr) * (lamg * lamg)))
                                + FDIV(0.5f, lamr * mo_powi3(lamg))))
                        * MO_CONS41;
                    dum = FDIV(pracg, 5.2E-7f);
                    const float e = unr - ung;
                    npracg = ((((gfk_pow((e * e) * 1.7f + (unr * 0.3f) * ung,
                                         0.5f) * (rho * MO_CONS32)) * n0rr)
                               * n0g)
                              * ((FDIV(1.0f, mo_powi3(lamr) * lamg)
                                  + FDIV(1.0f, (lamr * lamr) * (lamg * lamg)))
                                 + FDIV(1.0f, lamr * mo_powi3(lamg))));
                    npracg = npracg - dum;
                }
                // :1794-1803 accretion of cloud by rain
                if (qr3d >= 1.E-8f && qc3d >= 1.E-8f) {
                    dum = qc3d * qr3d;
                    pra = gfk_pow(dum, 1.15f) * 67.0f;
                    npra = FDIV(pra, FDIV(qc3d, nc3d));
                }
                // :1810-1820 rain self-collection
                if (qr3d >= 1.E-8f) {
                    const float dum1 = 300.e-6f;
                    if (FDIV(1.0f, lamr) < dum1) {
                        dum = 1.0f;
                    } else if (FDIV(1.0f, lamr) >= dum1) {
                        dum = 2.0f - gfk_exp((FDIV(1.0f, lamr) - dum1)
                                             * 2300.0f);
                    }
                    nragg = -((((dum * 5.78f) * nr3d) * qr3d) * rho);
                }
                // :1825-1842 rain evaporation
                if (qr3d >= MO_QSMALL) {
                    epsr = ((n0rr * 6.283185482025146484375f) * rho * dv)
                        * (FDIV(MO_F1R, lamr * lamr)
                           + FDIV(((gfk_pow(FDIV(arn * rho, mu), 0.5f)
                                    * MO_F2R) * gfk_pow(sc, 0.333333343f))
                                  * MO_CONS9, gfk_pow(lamr, MO_CONS34)));
                } else {
                    epsr = 0.0f;
                }
                if (qv3d < qvs) {
                    pre = FDIV(epsr * (qv3d - qvs), ab);
                    pre = FMINK(pre, 0.0f);
                } else {
                    pre = 0.0f;
                }
                // :1850-1882 melting of snow
                if (qni3d >= 1.E-8f) {
                    dum = -((FDIV(MO_CPW, xlf) * (t3d - 273.15f)) * pracs);
                    psmlt = FDIV((((n0s * 6.283185482025146484375f) * kap)
                                  * (273.15f - t3d)), xlf)
                        * (FDIV(MO_F1S, lams * lams)
                           + FDIV(((gfk_pow(FDIV(asn * rho, mu), 0.5f)
                                    * MO_F2S) * gfk_pow(sc, 0.333333343f))
                                  * MO_CONS10, gfk_pow(lams, MO_CONS35)))
                        + dum;
                    if (qvqvs < 1.0f) {
                        epss = ((n0s * 6.283185482025146484375f) * rho * dv)
                            * (FDIV(MO_F1S, lams * lams)
                               + FDIV(((gfk_pow(FDIV(asn * rho, mu), 0.5f)
                                        * MO_F2S) * gfk_pow(sc, 0.333333343f))
                                      * MO_CONS10, gfk_pow(lams, MO_CONS35)));
                        evpms = FDIV((qv3d - qvs) * epss, ab);
                        evpms = FMAX(evpms, psmlt);
                        psmlt = psmlt - evpms;
                    }
                }
                // :1890-1922 melting of graupel
                if (qg3d >= 1.E-8f) {
                    dum = -((FDIV(MO_CPW, xlf) * (t3d - 273.15f)) * pracg);
                    pgmlt = FDIV((((n0g * 6.283185482025146484375f) * kap)
                                  * (273.15f - t3d)), xlf)
                        * (FDIV(MO_F1S, lamg * lamg)
                           + FDIV(((gfk_pow(FDIV(agn * rho, mu), 0.5f)
                                    * MO_F2S) * gfk_pow(sc, 0.333333343f))
                                  * MO_CONS11, gfk_pow(lamg, MO_CONS36)))
                        + dum;
                    if (qvqvs < 1.0f) {
                        epsg = ((n0g * 6.283185482025146484375f) * rho * dv)
                            * (FDIV(MO_F1S, lamg * lamg)
                               + FDIV(((gfk_pow(FDIV(agn * rho, mu), 0.5f)
                                        * MO_F2S) * gfk_pow(sc, 0.333333343f))
                                      * MO_CONS11, gfk_pow(lamg, MO_CONS36)));
                        evpmg = FDIV((qv3d - qvs) * epsg, ab);
                        evpmg = FMAX(evpmg, pgmlt);
                        pgmlt = pgmlt - evpmg;
                    }
                }
                pracg = 0.0f;
                pracs = 0.0f;
                // :1940-1992 conservation
                dum = (prc + pra) * dt;
                if (dum > qc3d && qc3d >= MO_QSMALL) {
                    ratio = FDIV(qc3d, dum);
                    prc = prc * ratio;
                    pra = pra * ratio;
                }
                dum = ((-psmlt - evpms) + pracs) * dt;
                if (dum > qni3d && qni3d >= MO_QSMALL) {
                    ratio = FDIV(qni3d, dum);
                    psmlt = psmlt * ratio;
                    evpms = evpms * ratio;
                    pracs = pracs * ratio;
                }
                dum = ((-pgmlt - evpmg) + pracg) * dt;
                if (dum > qg3d && qg3d >= MO_QSMALL) {
                    ratio = FDIV(qg3d, dum);
                    pgmlt = pgmlt * ratio;
                    evpmg = evpmg * ratio;
                    pracg = pracg * ratio;
                }
                dum = ((((((-pracs - pracg) - pre) - pra) - prc) + psmlt)
                       + pgmlt) * dt;
                if (dum > qr3d && qr3d >= MO_QSMALL) {
                    ratio = FDIV(((((((FDIV(qr3d, dt) + pracs) + pracg)
                                     + pra) + prc) - psmlt) - pgmlt), -pre);
                    pre = pre * ratio;
                }
                // :1996-2010
                qv3dten = qv3dten + ((-pre - evpms) - evpmg);
                t3dten = t3dten + FDIV((pre * xxlv + (evpms + evpmg) * xxls)
                                       + (((psmlt + pgmlt) - pracs) - pracg)
                                       * xlf, cpm);
                qc3dten = qc3dten + (-pra - prc);
                qr3dten = qr3dten + ((((((pre + pra) + prc) - psmlt) - pgmlt)
                                      + pracs) + pracg);
                qni3dten = qni3dten + ((psmlt + evpms) - pracs);
                qg3dten = qg3dten + ((pgmlt + evpmg) - pracg);
                nc3dten = nc3dten + (-npra - nprc);
                nr3dten = nr3dten + ((nprc1 + nragg) - npracg);
                // :2015-2044
                if (pre < 0.0f) {
                    dum = FDIV(pre * dt, qr3d);
                    dum = FMAXK(dum, -1.0f);
                    nsubr = FDIV(dum * nr3d, dt);
                }
                if (evpms + psmlt < 0.0f) {
                    dum = FDIV((evpms + psmlt) * dt, qni3d);
                    dum = FMAXK(dum, -1.0f);
                    nsmlts = FDIV(dum * ns3d, dt);
                }
                if (psmlt < 0.0f) {
                    dum = FDIV(psmlt * dt, qni3d);
                    dum = FMAXK(dum, -1.0f);
                    nsmltr = FDIV(dum * ns3d, dt);
                }
                if (evpmg + pgmlt < 0.0f) {
                    dum = FDIV((evpmg + pgmlt) * dt, qg3d);
                    dum = FMAXK(dum, -1.0f);
                    ngmltg = FDIV(dum * ng3d, dt);
                }
                if (pgmlt < 0.0f) {
                    dum = FDIV(pgmlt * dt, qg3d);
                    dum = FMAXK(dum, -1.0f);
                    ngmltr = FDIV(dum * ng3d, dt);
                }
                ns3dten = ns3dten + nsmlts;
                ng3dten = ng3dten + ngmltg;
                nr3dten = nr3dten + ((nsubr - nsmltr) - ngmltr);
                (void)nsubr;
            }
            // 300 CONTINUE: :2052-2070 saturation adjustment.
            {
                const float dumt = t3d + dt * t3dten;
                const float dumqv = qv3d + dt * qv3dten;
                dum = FMIN(pres * 0.99f, mo_polysvp(dumt, 0));
                const float dumqss = FDIV(dum * MO_EP_2, pres - dum);
                float dumqc = qc3d + dt * qc3dten;
                dumqc = FMAXK(dumqc, 0.0f);
                const float dums = dumqv - dumqss;
                pcc = FDIV(FDIV(dums, FDIV((xxlv * xxlv) * dumqss,
                                           (dumt * dumt) * (cpm * MO_RV))
                                + 1.0f), dt);
                if (pcc * dt + dumqc < 0.0f) pcc = FDIV(-dumqc, dt);
                qv3dten = qv3dten - pcc;
                t3dten = t3dten + FDIV(pcc * xxlv, cpm);
                qc3dten = qc3dten + pcc;
            }
        } else {
            // ----------------------------------------- :2108-3327 cold
            nc3d = FDIV(MO_NDCNST * 1.E6f, rho);
            ni3d = FMAXK(ni3d, 0.0f);
            ns3d = FMAXK(ns3d, 0.0f);
            nc3d = FMAXK(nc3d, 0.0f);
            nr3d = FMAXK(nr3d, 0.0f);
            ng3d = FMAXK(ng3d, 0.0f);
            // :2125-2147 cloud ice
            if (qi3d >= MO_QSMALL) {
                lami = gfk_pow(FDIV(ni3d * MO_CONS12, qi3d),
                               FDIV(1.0f, MO_DI));
                n0i = ni3d * lami;
                if (lami < MO_LAMMINI) {
                    lami = MO_LAMMINI;
                    n0i = FDIV(mo_powi4(lami) * qi3d, MO_CONS12);
                    ni3d = FDIV(n0i, lami);
                } else if (lami > MO_LAMMAXI) {
                    lami = MO_LAMMAXI;
                    n0i = FDIV(mo_powi4(lami) * qi3d, MO_CONS12);
                    ni3d = FDIV(n0i, lami);
                }
            }
            // :2152-2173 rain
            if (qr3d >= MO_QSMALL) {
                lamr = gfk_pow(FDIV((MO_RHOW * MO_PI) * nr3d, qr3d),
                               0.333333343f);
                n0rr = nr3d * lamr;
                if (lamr < MO_LAMMINR) {
                    lamr = MO_LAMMINR;
                    n0rr = FDIV(mo_powi4(lamr) * qr3d, MO_RHOW * MO_PI);
                    nr3d = FDIV(n0rr, lamr);
                } else if (lamr > MO_LAMMAXR) {
                    lamr = MO_LAMMAXR;
                    n0rr = FDIV(mo_powi4(lamr) * qr3d, MO_RHOW * MO_PI);
                    nr3d = FDIV(n0rr, lamr);
                }
            }
            // :2180-2215 cloud droplets
            if (qc3d >= MO_QSMALL) {
                dum = FDIV(pres, t3d * 287.15f);
                pgam = (dum * FDIV(nc3d, 1.E6f)) * 0.0005714f + 0.2714f;
                pgam = FDIV(1.0f, pgam * pgam) - 1.0f;
                pgam = FMAXK(pgam, 2.0f);
                pgam = FMINK(pgam, 10.0f);
                lamc = gfk_pow(FDIV((nc3d * MO_CONS26) * mo_gamma(pgam + 4.0f),
                                    qc3d * mo_gamma(pgam + 1.0f)),
                               0.333333343f);
                lammin = FDIV(pgam + 1.0f, 60.E-6f);
                lammax = FDIV(pgam + 1.0f, 1.E-6f);
                if (lamc < lammin) {
                    lamc = lammin;
                    nc3d = FDIV(gfk_exp(((gfk_log(lamc) * 3.0f
                                          + gfk_log(qc3d))
                                         + gfk_log(mo_gamma(pgam + 1.0f)))
                                        - gfk_log(mo_gamma(pgam + 4.0f))),
                                MO_CONS26);
                } else if (lamc > lammax) {
                    lamc = lammax;
                    nc3d = FDIV(gfk_exp(((gfk_log(lamc) * 3.0f
                                          + gfk_log(qc3d))
                                         + gfk_log(mo_gamma(pgam + 1.0f)))
                                        - gfk_log(mo_gamma(pgam + 4.0f))),
                                MO_CONS26);
                }
                cdist1 = FDIV(nc3d, mo_gamma(pgam + 1.0f));
            }
            // :2220-2241 snow
            if (qni3d >= MO_QSMALL) {
                lams = gfk_pow(FDIV(ns3d * MO_CONS1, qni3d),
                               FDIV(1.0f, MO_DS));
                n0s = ns3d * lams;
                if (lams < MO_LAMMINS) {
                    lams = MO_LAMMINS;
                    n0s = FDIV(mo_powi4(lams) * qni3d, MO_CONS1);
                    ns3d = FDIV(n0s, lams);
                } else if (lams > MO_LAMMAXS) {
                    lams = MO_LAMMAXS;
                    n0s = FDIV(mo_powi4(lams) * qni3d, MO_CONS1);
                    ns3d = FDIV(n0s, lams);
                }
            }
            // :2246-2267 graupel
            if (qg3d >= MO_QSMALL) {
                lamg = gfk_pow(FDIV(ng3d * MO_CONS2, qg3d),
                               FDIV(1.0f, MO_DG));
                n0g = ng3d * lamg;
                if (lamg < MO_LAMMING) {
                    lamg = MO_LAMMING;
                    n0g = FDIV(mo_powi4(lamg) * qg3d, MO_CONS2);
                    ng3d = FDIV(n0g, lamg);
                } else if (lamg > MO_LAMMAXG) {
                    lamg = MO_LAMMAXG;
                    n0g = FDIV(mo_powi4(lamg) * qg3d, MO_CONS2);
                    ng3d = FDIV(n0g, lamg);
                }
            }
            // :2272-2331
            float mnuccc = 0.0f, nnuccc = 0.0f, prc = 0.0f, nprc = 0.0f;
            float nprc1 = 0.0f, nsagg = 0.0f, psacws = 0.0f, npsacws = 0.0f;
            float psacwi = 0.0f, npsacwi = 0.0f, pracs = 0.0f, npracs = 0.0f;
            float nmults = 0.0f, qmults = 0.0f, nmultr = 0.0f, qmultr = 0.0f;
            float nmultg = 0.0f, qmultg = 0.0f, nmultrg = 0.0f;
            float qmultrg = 0.0f, mnuccr = 0.0f, nnuccr = 0.0f, pra = 0.0f;
            float npra = 0.0f, nragg = 0.0f, prci = 0.0f, nprci = 0.0f;
            float prai = 0.0f, nprai = 0.0f, nnuccd = 0.0f, mnuccd = 0.0f;
            float pcc = 0.0f, pre = 0.0f, prd = 0.0f, prds = 0.0f;
            float eprd = 0.0f, eprds = 0.0f, nsubi = 0.0f;
            float nsubs = 0.0f, nsubr = 0.0f, piacr = 0.0f, niacr = 0.0f;
            float praci = 0.0f, piacrs = 0.0f, niacrs = 0.0f, pracis = 0.0f;
            float pracg = 0.0f, psacr = 0.0f, psacwg = 0.0f, pgsacw = 0.0f;
            float pgracs = 0.0f, prdg = 0.0f, eprdg = 0.0f, npracg = 0.0f;
            float npsacwg = 0.0f, nscng = 0.0f, ngracs = 0.0f, nsubg = 0.0f;
            float ums, umr, uns, unr, umg, ung, fmult = 0.0f;
#ifdef MORR_ROW37
            // Row 37 (owner ruling): the wide freezing rates when WRF's
            // float rate is not finite (see the conservation blocks).
            double mnuccc_wide = 0.0, mnuccr_wide = 0.0, nnuccc_wide = 0.0;
            bool mnuccc_overflow = false, mnuccr_overflow = false;
#endif
            // :2339-2395 contact and immersion freezing of droplets
            if (qc3d >= MO_QSMALL && t3d < 269.15f) {
                const float nacnt = gfk_exp((273.15f - t3d) * 0.262f - 2.80f)
                    * 1000.0f;
                dum = FDIV(FDIV(t3d * 7.37f, pres * 2880.0f), 100.0f);
                const float dap = FDIV((t3d * MO_CONS37)
                                       * (FDIV(dum, MO_RIN) + 1.0f), mu);
                mnuccc = (((dap * MO_CONS38) * nacnt)
                          * gfk_exp((gfk_log(cdist1)
                                     + gfk_log(mo_gamma(pgam + 5.0f)))
                                    - gfk_log(lamc) * 4.0f));
                nnuccc = FDIV(((((dap * 6.283185482025146484375f) * nacnt)
                                * cdist1) * mo_gamma(pgam + 2.0f)), lamc);
                mnuccc = mnuccc
                    + (gfk_exp((gfk_log(cdist1)
                                + gfk_log(mo_gamma(pgam + 7.0f)))
                               - gfk_log(lamc) * 6.0f) * MO_CONS39)
                    * (gfk_exp((273.15f - t3d) * MO_AIMM) - 1.0f);
                nnuccc = nnuccc
                    + (gfk_exp((gfk_log(cdist1)
                                + gfk_log(mo_gamma(pgam + 4.0f)))
                               - gfk_log(lamc) * 3.0f) * MO_CONS40)
                    * (gfk_exp((273.15f - t3d) * MO_AIMM) - 1.0f);
#ifdef MORR_ROW37
                if (!isfinite(mnuccc) || !isfinite(nnuccc)) {
                    // The separate float exponentials overflow in cold
                    // finite air and erase the freezing (WRF :2364-2391).
                    // Re-evaluate the positive factors in double, keep the
                    // mass rate wide until the cloud donor budget below.
                    mnuccc_overflow = true;
                    const double dist = (double)cdist1;
                    const double lam = (double)lamc;
                    const double nuclei = glibc_exp((double)((273.15f - t3d)
                                                       * 0.262f - 2.80f))
                        * 1000.0;
                    const double freezing = glibc_exp((double)((273.15f - t3d)
                                                         * MO_AIMM)) - 1.0;
                    mnuccc_wide = (double)MO_CONS38 * dap * nuclei * dist
                        * mo_gamma(pgam + 5.0f) / glibc_pow(lam, 4.0)
                        + (double)MO_CONS39 * dist * mo_gamma(pgam + 7.0f)
                        / glibc_pow(lam, 6.0) * freezing;
                    nnuccc_wide = 6.283185482025146484375 * dap * nuclei
                        * dist * mo_gamma(pgam + 2.0f) / lam
                        + (double)MO_CONS40 * dist * mo_gamma(pgam + 4.0f)
                        / glibc_pow(lam, 3.0) * freezing;
                    nnuccc = (float)fmin(nnuccc_wide, (double)nc3d / dt);
                    mnuccc = 0.0f;
                } else
#endif
                nnuccc = FMIN(nnuccc, FDIV(nc3d, dt));
            }
            // :2407-2425 autoconversion
            if (qc3d >= 1.E-6f) {
                prc = (gfk_pow(qc3d, 2.47f) * 1350.0f)
                    * gfk_pow(FDIV(nc3d, 1.e6f) * rho, -1.79f);
                nprc1 = FDIV(prc, MO_CONS29);
                nprc = FDIV(prc, FDIV(qc3d, nc3d));
                nprc = FMIN(nprc, FDIV(nc3d, dt));
                nprc1 = FMIN(nprc1, nprc);
            }
            // :2433-2438 snow self-aggregation
            if (qni3d >= 1.E-8f) {
                nsagg = FDIV(((asn * MO_CONS15)
                              * gfk_pow(rho, FDIV(MO_BS + 2.0f, 3.0f)))
                             * gfk_pow(qni3d, FDIV(MO_BS + 2.0f, 3.0f))
                             * gfk_pow(ns3d * rho, FDIV(4.0f - MO_BS, 3.0f)),
                             rho);
            }
            // :2447-2456 accretion of droplets by snow
            if (qni3d >= 1.E-8f && qc3d >= MO_QSMALL) {
                psacws = FDIV((((asn * MO_CONS13) * qc3d) * rho) * n0s,
                              gfk_pow(lams, MO_BS + 3.0f));
                npsacws = FDIV((((asn * MO_CONS13) * nc3d) * rho) * n0s,
                               gfk_pow(lams, MO_BS + 3.0f));
            }
            // :2461-2469 accretion of droplets by graupel
            if (qg3d >= 1.E-8f && qc3d >= MO_QSMALL) {
                psacwg = FDIV((((agn * MO_CONS14) * qc3d) * rho) * n0g,
                              gfk_pow(lamg, MO_BG + 3.0f));
                npsacwg = FDIV((((agn * MO_CONS14) * nc3d) * rho) * n0g,
                               gfk_pow(lamg, MO_BG + 3.0f));
            }
            // :2478-2492 accretion of droplets by cloud ice
            if (qi3d >= 1.E-8f && qc3d >= MO_QSMALL) {
                if (FDIV(1.0f, lami) >= 100.E-6f) {
                    psacwi = FDIV((((ain * MO_CONS16) * qc3d) * rho) * n0i,
                                  gfk_pow(lami, MO_BI + 3.0f));
                    npsacwi = FDIV((((ain * MO_CONS16) * nc3d) * rho) * n0i,
                                   gfk_pow(lami, MO_BI + 3.0f));
                }
            }
            // :2498-2548 rain-snow collection
            if (qr3d >= 1.E-8f && qni3d >= 1.E-8f) {
                ums = FDIV(asn * MO_CONS3, gfk_pow(lams, MO_BS));
                umr = FDIV(arn * MO_CONS4, gfk_pow(lamr, MO_BR));
                uns = FDIV(asn * MO_CONS5, gfk_pow(lams, MO_BS));
                unr = FDIV(arn * MO_CONS6, gfk_pow(lamr, MO_BR));
                dum = gfk_pow(FDIV(MO_RHOSU, rho), 0.54f);
                ums = FMIN(ums, dum * 1.2f);
                uns = FMIN(uns, dum * 1.2f);
                umr = FMIN(umr, dum * 9.1f);
                unr = FMIN(unr, dum * 9.1f);
                const float d = umr * 1.2f - ums * 0.95f;
                pracs = (FDIV(((gfk_pow(d * d + (ums * 0.08f) * umr, 0.5f)
                                * rho) * n0rr) * n0s, mo_powi3(lamr))
                         * ((FDIV(5.0f, mo_powi3(lamr) * lams)
                             + FDIV(2.0f, (lamr * lamr) * (lams * lams)))
                            + FDIV(0.5f, lamr * mo_powi3(lams))))
                    * MO_CONS41;
                const float e = unr - uns;
                npracs = ((((gfk_pow((e * e) * 1.7f + (unr * 0.3f) * uns,
                                     0.5f) * (rho * MO_CONS32)) * n0rr) * n0s)
                          * ((FDIV(1.0f, mo_powi3(lamr) * lams)
                              + FDIV(1.0f, (lamr * lamr) * (lams * lams)))
                             + FDIV(1.0f, lamr * mo_powi3(lams))));
                pracs = FMIN(pracs, FDIV(qr3d, dt));
                if (qni3d >= 0.1E-3f && qr3d >= 0.1E-3f) {
                    psacr = (FDIV(((gfk_pow(d * d + (ums * 0.08f) * umr, 0.5f)
                                    * rho) * n0rr) * n0s, mo_powi3(lams))
                             * ((FDIV(5.0f, mo_powi3(lams) * lamr)
                                 + FDIV(2.0f, (lams * lams) * (lamr * lamr)))
                                + FDIV(0.5f, lams * mo_powi3(lamr))))
                        * MO_CONS31;
                }
            }
            // :2554-2588 rain-graupel collection
            if (qr3d >= 1.E-8f && qg3d >= 1.E-8f) {
                umg = FDIV(agn * MO_CONS7, gfk_pow(lamg, MO_BG));
                umr = FDIV(arn * MO_CONS4, gfk_pow(lamr, MO_BR));
                ung = FDIV(agn * MO_CONS8, gfk_pow(lamg, MO_BG));
                unr = FDIV(arn * MO_CONS6, gfk_pow(lamr, MO_BR));
                dum = gfk_pow(FDIV(MO_RHOSU, rho), 0.54f);
                umg = FMIN(umg, dum * 20.0f);
                ung = FMIN(ung, dum * 20.0f);
                umr = FMIN(umr, dum * 9.1f);
                unr = FMIN(unr, dum * 9.1f);
                const float d = umr * 1.2f - umg * 0.95f;
                pracg = (FDIV(((gfk_pow(d * d + (umg * 0.08f) * umr, 0.5f)
                                * rho) * n0rr) * n0g, mo_powi3(lamr))
                         * ((FDIV(5.0f, mo_powi3(lamr) * lamg)
                             + FDIV(2.0f, (lamr * lamr) * (lamg * lamg)))
                            + FDIV(0.5f, lamr * mo_powi3(lamg))))
                    * MO_CONS41;
                const float e = unr - ung;
                npracg = ((((gfk_pow((e * e) * 1.7f + (unr * 0.3f) * ung,
                                     0.5f) * (rho * MO_CONS32)) * n0rr) * n0g)
                          * ((FDIV(1.0f, mo_powi3(lamr) * lamg)
                              + FDIV(1.0f, (lamr * lamr) * (lamg * lamg)))
                             + FDIV(1.0f, lamr * mo_powi3(lamg))));
                pracg = FMIN(pracg, FDIV(qr3d, dt));
            }
            // :2602-2651 rime splintering, snow
            if (qni3d >= 0.1E-3f) {
                if (qc3d >= 0.5E-3f || qr3d >= 0.1E-3f) {
                    if (psacws > 0.0f || pracs > 0.0f) {
                        if (t3d < 270.16f && t3d > 265.16f) {
                            if (t3d > 270.16f) {
                                fmult = 0.0f;
                            } else if (t3d <= 270.16f && t3d > 268.16f) {
                                fmult = FDIV(270.16f - t3d, 2.0f);
                            } else if (t3d >= 265.16f && t3d <= 268.16f) {
                                fmult = FDIV(t3d - 265.16f, 3.0f);
                            } else if (t3d < 265.16f) {
                                fmult = 0.0f;
                            }
                            if (psacws > 0.0f) {
                                nmults = ((psacws * 35.E4f) * fmult) * 1000.0f;
                                qmults = nmults * MO_MMULT;
                                qmults = FMIN(qmults, psacws);
                                psacws = psacws - qmults;
                            }
                            if (pracs > 0.0f) {
                                nmultr = ((pracs * 35.E4f) * fmult) * 1000.0f;
                                qmultr = nmultr * MO_MMULT;
                                qmultr = FMIN(qmultr, pracs);
                                pracs = pracs - qmultr;
                            }
                        }
                    }
                }
            }
            // :2665-2712 rime splintering, graupel
            if (qg3d >= 0.1E-3f) {
                if (qc3d >= 0.5E-3f || qr3d >= 0.1E-3f) {
                    if (psacwg > 0.0f || pracg > 0.0f) {
                        if (t3d < 270.16f && t3d > 265.16f) {
                            if (t3d > 270.16f) {
                                fmult = 0.0f;
                            } else if (t3d <= 270.16f && t3d > 268.16f) {
                                fmult = FDIV(270.16f - t3d, 2.0f);
                            } else if (t3d >= 265.16f && t3d <= 268.16f) {
                                fmult = FDIV(t3d - 265.16f, 3.0f);
                            } else if (t3d < 265.16f) {
                                fmult = 0.0f;
                            }
                            if (psacwg > 0.0f) {
                                nmultg = ((psacwg * 35.E4f) * fmult) * 1000.0f;
                                qmultg = nmultg * MO_MMULT;
                                qmultg = FMIN(qmultg, psacwg);
                                psacwg = psacwg - qmultg;
                            }
                            if (pracg > 0.0f) {
                                nmultrg = ((pracg * 35.E4f) * fmult) * 1000.0f;
                                qmultrg = nmultrg * MO_MMULT;
                                qmultrg = FMIN(qmultrg, pracg);
                                pracg = pracg - qmultrg;
                            }
                        }
                    }
                }
            }
            // :2719-2739 conversion of rimed snow to graupel
            if (psacws > 0.0f) {
                if (qni3d >= 0.1E-3f && qc3d >= 0.5E-3f) {
                    pgsacw = FMIN(psacws,
                                  FDIV(((((dt * MO_CONS17) * n0s) * qc3d)
                                        * qc3d) * asn * asn,
                                       rho * gfk_pow(lams,
                                                     MO_BS * 2.0f + 2.0f)));
                    dum = FMAXK(FDIV(MO_RHOSN, MO_RHOG - MO_RHOSN) * pgsacw,
                                0.0f);
                    nscng = FDIV(dum, MO_MG0) * rho;
                    nscng = FMIN(nscng, FDIV(ns3d, dt));
                    psacws = psacws - pgsacw;
                }
            }
            // :2743-2764 conversion of rain-snow collection to graupel
            if (pracs > 0.0f) {
                if (qni3d >= 0.1E-3f && qr3d >= 0.1E-3f) {
                    const float s4 = FDIV(4.0f, lams), r4 = FDIV(4.0f, lamr);
                    dum = FDIV((mo_powi3(s4) * MO_CONS18) * mo_powi3(s4),
                               (mo_powi3(s4) * MO_CONS18) * mo_powi3(s4)
                               + (mo_powi3(r4) * MO_CONS19) * mo_powi3(r4));
                    dum = FMINK(dum, 1.0f);
                    dum = FMAXK(dum, 0.0f);
                    pgracs = (1.0f - dum) * pracs;
                    ngracs = (1.0f - dum) * npracs;
                    ngracs = FMIN(ngracs, FDIV(nr3d, dt));
                    ngracs = FMIN(ngracs, FDIV(ns3d, dt));
                    pracs = pracs - pgracs;
                    npracs = npracs - ngracs;
                    psacr = psacr * (1.0f - dum);
                }
            }
            // :2771-2788 heterogeneous freezing of rain
            if (t3d < 269.15f && qr3d >= MO_QSMALL) {
                mnuccr = FDIV(FDIV((nr3d * MO_CONS20)
                                   * (gfk_exp((273.15f - t3d) * MO_AIMM)
                                      - 1.0f), mo_powi3(lamr)),
                              mo_powi3(lamr));
                nnuccr = FDIV(((nr3d * MO_PI) * MO_BIMM)
                              * (gfk_exp((273.15f - t3d) * MO_AIMM) - 1.0f),
                              mo_powi3(lamr));
                nnuccr = FMIN(nnuccr, FDIV(nr3d, dt));
#ifdef MORR_ROW37
                if (!isfinite(mnuccr)) {
                    // A finite Bigg exponential overflows the float
                    // numerator before the two slope-cubed divisions
                    // (WRF :2780-2781).  Keep the rate wide until the rain
                    // donor budget has limited every competing sink.
                    mnuccr_overflow = true;
                    const double lambda3 = (double)lamr * lamr * lamr;
                    mnuccr_wide = (double)MO_CONS20 * nr3d
                        * (glibc_exp((double)((273.15f - t3d) * MO_AIMM)) - 1.0)
                        / lambda3 / lambda3;
                    mnuccr = 0.0f;
                }
#endif
            }
            // :2795-2804 accretion of cloud by rain
            if (qr3d >= 1.E-8f && qc3d >= 1.E-8f) {
                dum = qc3d * qr3d;
                pra = gfk_pow(dum, 1.15f) * 67.0f;
                npra = FDIV(pra, FDIV(qc3d, nc3d));
            }
            // :2811-2821 rain self-collection
            if (qr3d >= 1.E-8f) {
                const float dum1 = 300.e-6f;
                if (FDIV(1.0f, lamr) < dum1) {
                    dum = 1.0f;
                } else if (FDIV(1.0f, lamr) >= dum1) {
                    dum = 2.0f - gfk_exp((FDIV(1.0f, lamr) - dum1) * 2300.0f);
                }
                nragg = -((((dum * 5.78f) * nr3d) * qr3d) * rho);
            }
            // :2829-2839 autoconversion of cloud ice to snow
            if (qi3d >= 1.E-8f && qvqvsi >= 1.0f) {
                nprci = FDIV((((((qv3d - qvi) * MO_CONS21) * rho) * n0i)
                              * gfk_exp(-(lami * MO_DCS))) * dv, abi);
                prci = nprci * MO_CONS22;
                nprci = FMIN(nprci, FDIV(ni3d, dt));
            }
            // :2846-2853 accretion of cloud ice by snow
            if (qni3d >= 1.E-8f && qi3d >= MO_QSMALL) {
                prai = FDIV((((asn * MO_CONS23) * qi3d) * rho) * n0s,
                            gfk_pow(lams, MO_BS + 3.0f));
                nprai = FDIV((((asn * MO_CONS23) * ni3d) * rho) * n0s,
                             gfk_pow(lams, MO_BS + 3.0f));
                nprai = FMIN(nprai, FDIV(ni3d, dt));
            }
            // :2860-2884 rain-cloud ice collection
            if (qr3d >= 1.E-8f && qi3d >= 1.E-8f && t3d <= 273.15f) {
                if (qr3d >= 0.1E-3f) {
                    niacr = FDIV(((ni3d * MO_CONS24) * n0rr) * arn,
                                 gfk_pow(lamr, MO_BR + 3.0f)) * rho;
                    piacr = FDIV(FDIV(((ni3d * MO_CONS25) * n0rr) * arn,
                                      gfk_pow(lamr, MO_BR + 3.0f)),
                                 mo_powi3(lamr)) * rho;
                    praci = FDIV(((qi3d * MO_CONS24) * n0rr) * arn,
                                 gfk_pow(lamr, MO_BR + 3.0f)) * rho;
                    niacr = FMIN(niacr, FDIV(nr3d, dt));
                    niacr = FMIN(niacr, FDIV(ni3d, dt));
                } else {
                    niacrs = FDIV(((ni3d * MO_CONS24) * n0rr) * arn,
                                  gfk_pow(lamr, MO_BR + 3.0f)) * rho;
                    piacrs = FDIV(FDIV(((ni3d * MO_CONS25) * n0rr) * arn,
                                       gfk_pow(lamr, MO_BR + 3.0f)),
                                  mo_powi3(lamr)) * rho;
                    pracis = FDIV(((qi3d * MO_CONS24) * n0rr) * arn,
                                  gfk_pow(lamr, MO_BR + 3.0f)) * rho;
                    niacrs = FMIN(niacrs, FDIV(nr3d, dt));
                    niacrs = FMIN(niacrs, FDIV(ni3d, dt));
                }
            }
            // :2889-2907 deposition/condensation-freezing nucleation
            // (INUC = 0, Cooper 1986).
            if ((qvqvs >= 0.999f && t3d <= 265.15f) || qvqvsi >= 1.08f) {
                float kc2 = (gfk_exp((273.15f - t3d) * 0.304f) * 0.005f)
                    * 1000.0f;
                kc2 = FMINK(kc2, 500.e3f);
                kc2 = FMAXK(FDIV(kc2, rho), 0.0f);
                if (kc2 > (ni3d + ns3d) + ng3d) {
                    nnuccd = FDIV(((kc2 - ni3d) - ns3d) - ng3d, dt);
                    mnuccd = nnuccd * MO_MI0;
#ifdef MORR_ROW37
                    // Row 37 (owner ruling): nucleation may not consume
                    // vapour that is not there.  Below ~159 K POLYSVP's
                    // extrapolated liquid curve drops under its ice curve,
                    // EIS = MIN(EW, ...) makes QVI == QVS, and the
                    // 0.999*QVS trigger fires at or below ice saturation;
                    // the FUDGEF rescale skips that sign pair and the
                    // unbounded MNUCCD drove a model-top level to negative
                    // vapour.  Bound the nucleated mass by the excess over
                    // ice saturation and scale the number with it.
                    const float avail = FDIV(fmaxf(qv3d - qvi, 0.0f), dt);
                    if (mnuccd > avail) {
                        nnuccd = nnuccd * FDIV(avail, mnuccd);
                        mnuccd = avail;
                    }
#endif
                }
            }
            // :2931-2969
            if (qi3d >= MO_QSMALL) {
                epsi = FDIV(((n0i * 6.283185482025146484375f) * rho) * dv,
                            lami * lami);
            } else {
                epsi = 0.0f;
            }
            if (qni3d >= MO_QSMALL) {
                epss = ((n0s * 6.283185482025146484375f) * rho * dv)
                    * (FDIV(MO_F1S, lams * lams)
                       + FDIV(((gfk_pow(FDIV(asn * rho, mu), 0.5f) * MO_F2S)
                               * gfk_pow(sc, 0.333333343f)) * MO_CONS10,
                              gfk_pow(lams, MO_CONS35)));
            } else {
                epss = 0.0f;
            }
            if (qg3d >= MO_QSMALL) {
                epsg = ((n0g * 6.283185482025146484375f) * rho * dv)
                    * (FDIV(MO_F1S, lamg * lamg)
                       + FDIV(((gfk_pow(FDIV(agn * rho, mu), 0.5f) * MO_F2S)
                               * gfk_pow(sc, 0.333333343f)) * MO_CONS11,
                              gfk_pow(lamg, MO_CONS36)));
            } else {
                epsg = 0.0f;
            }
            if (qr3d >= MO_QSMALL) {
                epsr = ((n0rr * 6.283185482025146484375f) * rho * dv)
                    * (FDIV(MO_F1R, lamr * lamr)
                       + FDIV(((gfk_pow(FDIV(arn * rho, mu), 0.5f) * MO_F2R)
                               * gfk_pow(sc, 0.333333343f)) * MO_CONS9,
                              gfk_pow(lamr, MO_CONS34)));
            } else {
                epsr = 0.0f;
            }
            // :2975-2999
            if (qi3d >= MO_QSMALL) {
                dum = 1.0f - gfk_exp(-(lami * MO_DCS)) * (lami * MO_DCS + 1.0f);
                prd = FDIV(epsi * (qv3d - qvi), abi) * dum;
            } else {
                dum = 0.0f;
            }
            if (qni3d >= MO_QSMALL) {
                prds = FDIV(epss * (qv3d - qvi), abi)
                    + FDIV(epsi * (qv3d - qvi), abi) * (1.0f - dum);
            } else {
                prd = prd + FDIV(epsi * (qv3d - qvi), abi) * (1.0f - dum);
            }
            prdg = FDIV(epsg * (qv3d - qvi), abi);
            if (qv3d < qvs) {
                pre = FDIV(epsr * (qv3d - qvs), ab);
                pre = FMINK(pre, 0.0f);
            } else {
                pre = 0.0f;
            }
            // :3004-3015
            dum = FDIV(qv3d - qvi, dt);
            const float fudgef = 0.9999f;
            const float sum_dep = ((prd + prds) + mnuccd) + prdg;
            if ((dum > 0.0f && sum_dep > dum * fudgef)
                    || (dum < 0.0f && sum_dep < dum * fudgef)) {
                mnuccd = FDIV(dum * (fudgef * mnuccd), sum_dep);
                prd = FDIV(dum * (fudgef * prd), sum_dep);
                prds = FDIV(dum * (fudgef * prds), sum_dep);
                prdg = FDIV(dum * (fudgef * prdg), sum_dep);
            }
            // :3019-3030
            if (prd < 0.0f) { eprd = prd; prd = 0.0f; }
            if (prds < 0.0f) { eprds = prds; prds = 0.0f; }
            if (prdg < 0.0f) { eprdg = prdg; prdg = 0.0f; }
            // :3089-3103 conservation of QC
            dum = ((((((((prc + pra) + mnuccc) + psacws) + psacwi) + qmults)
                     + psacwg) + pgsacw) + qmultg) * dt;
#ifdef MORR_ROW37
            if (mnuccc_overflow) {
                // Row 37: WRF's DUM is not finite here.  Ration the cloud
                // donor across every sink with the wide freezing rate.
                const double other = (double)prc + pra + psacws + psacwi
                    + qmults + psacwg + pgsacw + qmultg;
                const double rat = fmin(1.0, ((double)qc3d / dt)
                                        / (other + mnuccc_wide));
                prc = (float)(prc * rat);
                pra = (float)(pra * rat);
                mnuccc = (float)(mnuccc_wide * rat);
                psacws = (float)(psacws * rat);
                psacwi = (float)(psacwi * rat);
                qmults = (float)(qmults * rat);
                qmultg = (float)(qmultg * rat);
                psacwg = (float)(psacwg * rat);
                pgsacw = (float)(pgsacw * rat);
            } else
#endif
            if (dum > qc3d && qc3d >= MO_QSMALL) {
                ratio = FDIV(qc3d, dum);
                prc = prc * ratio;
                pra = pra * ratio;
                mnuccc = mnuccc * ratio;
                psacws = psacws * ratio;
                psacwi = psacwi * ratio;
                qmults = qmults * ratio;
                qmultg = qmultg * ratio;
                psacwg = psacwg * ratio;
                pgsacw = pgsacw * ratio;
            }
            // :3107-3122 conservation of QI
            dum = ((((((((((((-prd - mnuccc) + prci) + prai) - qmults)
                         - qmultg) - qmultr) - qmultrg) - mnuccd) + praci)
                     + pracis) - eprd) - psacwi) * dt;
            if (dum > qi3d && qi3d >= MO_QSMALL) {
                ratio = FDIV((((((((FDIV(qi3d, dt) + prd) + mnuccc) + qmults)
                                  + qmultg) + qmultr) + qmultrg) + mnuccd)
                             + psacwi,
                             (((prci + prai) + praci) + pracis) - eprd);
                prci = prci * ratio;
                prai = prai * ratio;
                praci = praci * ratio;
                pracis = pracis * ratio;
                eprd = eprd * ratio;
            }
            // :3126-3144 conservation of QR
#ifdef MORR_ROW37
            if (mnuccr_overflow) {
                // Row 37: ration the rain donor with the wide freezing
                // rate (WRF's DUM is not finite).
                const double other = -(double)pre + qmultr + qmultrg + pracs
                    + piacr + piacrs + pgracs + pracg;
                const double supply = (double)qr3d / dt + prc + pra;
                const double rat = fmin(1.0, supply / (other + mnuccr_wide));
                pre = (float)(pre * rat);
                pracs = (float)(pracs * rat);
                qmultr = (float)(qmultr * rat);
                qmultrg = (float)(qmultrg * rat);
                mnuccr = (float)(mnuccr_wide * rat);
                piacr = (float)(piacr * rat);
                piacrs = (float)(piacrs * rat);
                pgracs = (float)(pgracs * rat);
                pracg = (float)(pracg * rat);
            } else
#endif
            {
                dum = ((((((pracs - pre) + ((qmultr + qmultrg) - prc))
                          + (mnuccr - pra)) + piacr) + piacrs) + pgracs
                       + pracg) * dt;
                if (dum > qr3d && qr3d >= MO_QSMALL) {
                    ratio = FDIV((FDIV(qr3d, dt) + prc) + pra,
                                 (((((((-pre + qmultr) + qmultrg) + pracs)
                                     + mnuccr) + piacr) + piacrs) + pgracs)
                                 + pracg);
                    pre = pre * ratio;
                    pracs = pracs * ratio;
                    qmultr = qmultr * ratio;
                    qmultrg = qmultrg * ratio;
                    mnuccr = mnuccr * ratio;
                    piacr = piacr * ratio;
                    piacrs = piacrs * ratio;
                    pgracs = pgracs * ratio;
                    pracg = pracg * ratio;
                }
            }
            // :3149-3160 conservation of snow (IGRAUP = 0)
            dum = ((((((((-prds - psacws) - prai) - prci) - pracs) - eprds)
                     + psacr) - piacrs) - pracis) * dt;
            if (dum > qni3d && qni3d >= MO_QSMALL) {
                ratio = FDIV(((((((FDIV(qni3d, dt) + prds) + psacws) + prai)
                                 + prci) + pracs) + piacrs) + pracis,
                             -eprds + psacr);
                eprds = eprds * ratio;
                psacr = psacr * ratio;
            }
            // :3180-3189 conservation of graupel
            dum = (((((((((-psacwg - pracg) - pgsacw) - pgracs) - prdg)
                        - mnuccr) - eprdg) - piacr) - praci) - psacr) * dt;
            if (dum > qg3d && qg3d >= MO_QSMALL) {
                ratio = FDIV((((((((((FDIV(qg3d, dt) + psacwg) + pracg)
                                    + pgsacw) + pgracs) + prdg) + mnuccr)
                                 + psacr) + piacr) + praci), -eprdg);
                eprdg = eprdg * ratio;
            }
            // :3193-3239 tendencies
            qv3dten = qv3dten + (((((((-pre - prd) - prds) - mnuccd) - eprd)
                                   - eprds) - prdg) - eprdg);
            t3dten = t3dten
                + FDIV((pre * xxlv
                        + ((((((prd + prds) + mnuccd) + eprd) + eprds) + prdg)
                           + eprdg) * xxls)
                       + ((((((((((((((psacws + psacwi) + mnuccc) + mnuccr)
                                     + qmults) + qmultg) + qmultr) + qmultrg)
                                 + pracs) + psacwg) + pracg) + pgsacw)
                             + pgracs) + piacr) + piacrs) * xlf, cpm);
            qc3dten = qc3dten + (((((((((-pra - prc) - mnuccc) + pcc)
                                       - psacws) - psacwi) - qmults) - qmultg)
                                  - psacwg) - pgsacw);
            qi3dten = qi3dten + (((((((((((((prd + eprd) + psacwi) + mnuccc)
                                           - prci) - prai) + qmults) + qmultg)
                                       + qmultr) + qmultrg) + mnuccd) - praci)
                                  - pracis));
            qr3dten = qr3dten + ((((((((((pre + pra) + prc) - pracs)
                                        - mnuccr) - qmultr) - qmultrg)
                                     - piacr) - piacrs) - pracg) - pgracs);
            qni3dten = qni3dten + ((((((((prai + psacws) + prds) + pracs)
                                        + prci) + eprds) - psacr) + piacrs)
                                   + pracis);
            ns3dten = ns3dten + ((((nsagg + nprci) - nscng) - ngracs)
                                 + niacrs);
            qg3dten = qg3dten + (((((((((pracg + psacwg) + pgsacw) + pgracs)
                                       + prdg) + eprdg) + mnuccr) + piacr)
                                  + praci) + psacr);
            ng3dten = ng3dten + (((nscng + ngracs) + nnuccr) + niacr);
            nc3dten = nc3dten + (((((-nnuccc - npsacws) - npra) - nprc)
                                  - npsacwi) - npsacwg);
            ni3dten = ni3dten + (((((((((nnuccc - nprci) - nprai) + nmults)
                                       + nmultg) + nmultr) + nmultrg)
                                    + nnuccd) - niacr) - niacrs);
            nr3dten = nr3dten + (((((((nprc1 - npracs) - nnuccr) + nragg)
                                     - niacr) - niacrs) - npracg) - ngracs);
            // :3249-3267 saturation adjustment
            {
                const float dumt = t3d + dt * t3dten;
                const float dumqv = qv3d + dt * qv3dten;
                dum = FMIN(pres * 0.99f, mo_polysvp(dumt, 0));
                const float dumqss = FDIV(dum * MO_EP_2, pres - dum);
                float dumqc = qc3d + dt * qc3dten;
                dumqc = FMAXK(dumqc, 0.0f);
                const float dums = dumqv - dumqss;
                pcc = FDIV(FDIV(dums, FDIV((xxlv * xxlv) * dumqss,
                                           (dumt * dumt) * (cpm * MO_RV))
                                + 1.0f), dt);
                if (pcc * dt + dumqc < 0.0f) pcc = FDIV(-dumqc, dt);
                qv3dten = qv3dten - pcc;
                t3dten = t3dten + FDIV(pcc * xxlv, cpm);
                qc3dten = qc3dten + pcc;
            }
            // :3285-3316
            if (eprd < 0.0f) {
                dum = FDIV(eprd * dt, qi3d);
                dum = FMAXK(dum, -1.0f);
                nsubi = FDIV(dum * ni3d, dt);
            }
            if (eprds < 0.0f) {
                dum = FDIV(eprds * dt, qni3d);
                dum = FMAXK(dum, -1.0f);
                nsubs = FDIV(dum * ns3d, dt);
            }
            if (pre < 0.0f) {
                dum = FDIV(pre * dt, qr3d);
                dum = FMAXK(dum, -1.0f);
                nsubr = FDIV(dum * nr3d, dt);
            }
            if (eprdg < 0.0f) {
                dum = FDIV(eprdg * dt, qg3d);
                dum = FMAXK(dum, -1.0f);
                nsubg = FDIV(dum * ng3d, dt);
            }
            ni3dten = ni3dten + nsubi;
            ns3dten = ns3dten + nsubs;
            ng3dten = ng3dten + nsubg;
            nr3dten = nr3dten + nsubr;
        }
    }

    // Write the level back (the driver's 1-D arrays live in place).
    qv[idx] = qv3d; qc[idx] = qc3d; qr[idx] = qr3d; qi[idx] = qi3d;
    qs[idx] = qni3d; qg[idx] = qg3d; nc[idx] = nc3d; nr[idx] = nr3d;
    ni[idx] = ni3d; ns[idx] = ns3d; ng[idx] = ng3d;
    rho_out[idx] = rho;
    W(MO_T3DTEN) = t3dten; W(MO_QV3DTEN) = qv3dten; W(MO_QC3DTEN) = qc3dten;
    W(MO_QI3DTEN) = qi3dten; W(MO_QNI3DTEN) = qni3dten;
    W(MO_QR3DTEN) = qr3dten; W(MO_NI3DTEN) = ni3dten;
    W(MO_NS3DTEN) = ns3dten; W(MO_NR3DTEN) = nr3dten;
    W(MO_QG3DTEN) = qg3dten; W(MO_NG3DTEN) = ng3dten;
    W(MO_NC3DTEN) = nc3dten;
    W(MO_T3D) = t3d; W(MO_CPM) = cpm; W(MO_LAMI) = lami; W(MO_ACN) = acn;
    W(MO_QRSTEN) = 0.0f; W(MO_QISTEN) = 0.0f; W(MO_QNISTEN) = 0.0f;
    W(MO_QCSTEN) = 0.0f; W(MO_QGSTEN) = 0.0f;
    if (active) atomicOr(&ltrue[col], 1);
#undef W
}

// ---------------------------------------------------------------------
// Sedimentation (:3337-3667).  One block covers 32 columns with five
// warps, one per category (0 cloud, 1 rain, 2 ice, 3 snow, 4 graupel);
// lane l of every warp is column blockIdx.x*32 + l.  Categories are
// independent in WRF except for the shared NSTEP and the per-step
// surface sums, which go through shared memory in WRF's order.  NSTEP is
// MAX over levels of INT(RGVM*DT/DZQ+1.) with RGVM the largest of the ten
// speeds: every operation in that chain is monotone, so the maximum over
// categories of the per-category maxima is WRF's NSTEP exactly.
// ---------------------------------------------------------------------
template <int KMAX>
__device__ __forceinline__ void mo_sediment_impl(
    const float* __restrict__ pressure, const float* __restrict__ dz,
    const float* __restrict__ rho_in,
    const float* __restrict__ qc, const float* __restrict__ qr,
    const float* __restrict__ qi, const float* __restrict__ qs,
    const float* __restrict__ qg, const float* __restrict__ nc,
    const float* __restrict__ nr, const float* __restrict__ ni,
    const float* __restrict__ ns, const float* __restrict__ ng,
    float* __restrict__ work, const int* __restrict__ ltrue,
    float* __restrict__ rainnc, float* __restrict__ rainncv,
    float* __restrict__ snownc, float* __restrict__ snowncv,
    float* __restrict__ graupelnc, float* __restrict__ graupelncv,
    float* __restrict__ sr, float dt, int rimed, int nz, int ncol,
    int col0, int ccols)
{
    __shared__ int s_nstep[5][32];
    __shared__ float s_fall[5][32];
    const int lane = threadIdx.x & 31;
    const int cat = threadIdx.x >> 5;
    const int lc = blockIdx.x * 32 + lane;      // column within the chunk
    const bool valid = lc < ccols;
    const int col = col0 + lc;
    const bool sediments = valid && ltrue[col] != 0;
    const long long nlocal = (long long)nz * ccols;
#define AT(k) ((long long)(k) * ncol + col)
#define WP(p, k) work[(long long)(p) * nlocal + (long long)(k) * ccols + lc]

    const float* qmass = cat == 0 ? qc : cat == 1 ? qr : cat == 2 ? qi
                       : cat == 3 ? qs : qg;
    const float* qnum = cat == 0 ? nc : cat == 1 ? nr : cat == 2 ? ni
                      : cat == 3 ? ns : ng;
    const int mten = cat == 0 ? MO_QC3DTEN : cat == 1 ? MO_QR3DTEN
                   : cat == 2 ? MO_QI3DTEN : cat == 3 ? MO_QNI3DTEN
                   : MO_QG3DTEN;
    const int nten = cat == 0 ? MO_NC3DTEN : cat == 1 ? MO_NR3DTEN
                   : cat == 2 ? MO_NI3DTEN : cat == 3 ? MO_NS3DTEN
                   : MO_NG3DTEN;
    const int msten = cat == 0 ? MO_QCSTEN : cat == 1 ? MO_QRSTEN
                    : cat == 2 ? MO_QISTEN : cat == 3 ? MO_QNISTEN
                    : MO_QGSTEN;

    float dm[KMAX], dn[KMAX], fm[KMAX], fn[KMAX];
    int nstep = 1;
    if (sediments) {
        for (int k = nz - 1; k >= 0; --k) {
            const float rho = rho_in[AT(k)];
            // :3360-3383
            float dmass = mo_add(qmass[AT(k)], mo_mul(WP(mten, k), dt));
#ifndef MORR_ROW37_OFF
            // A source tendency can cancel its category to a negative FP32
            // residue. Inherited fall speed then exported negative water
            // (measured -1.54e-14 mm in a regional forecast). Sedimentation
            // transports nonnegative mass; retain the raw WRF oracle build.
            dmass = FMAXK(dmass, 0.0f);
#endif
            float dnum = mo_add(qnum[AT(k)], mo_mul(WP(nten, k), dt));
            if (cat == 0) dnum = qnum[AT(k)];   // iinum = 1 (:3372-3374)
            dnum = FMAXK(dnum, 0.0f);
            float um = 0.0f, un = 0.0f;
            const float dum = gfk_pow(FDIV(MO_RHOSU, rho), 0.54f);
            if (dmass >= MO_QSMALL) {
                if (cat == 0) {
                    // :3404-3416, :3439-3441
                    const float t3d = WP(MO_T3D, k);
                    const float nc3d = qnum[AT(k)];
                    const float d = FDIV(pressure[AT(k)], t3d * 287.15f);
                    float pgam = (d * FDIV(nc3d, 1.E6f)) * 0.0005714f
                        + 0.2714f;
                    pgam = FDIV(1.0f, pgam * pgam) - 1.0f;
                    pgam = FMAXK(pgam, 2.0f);
                    pgam = FMINK(pgam, 10.0f);
                    float dlamc = gfk_pow(
                        FDIV((dnum * MO_CONS26) * mo_gamma(pgam + 4.0f),
                             dmass * mo_gamma(pgam + 1.0f)), 0.333333343f);
                    const float lammin = FDIV(pgam + 1.0f, 60.E-6f);
                    const float lammax = FDIV(pgam + 1.0f, 1.E-6f);
                    dlamc = FMAX(dlamc, lammin);
                    dlamc = FMIN(dlamc, lammax);
                    const float acn = WP(MO_ACN, k);
                    un = FDIV(acn * mo_gamma((1.0f + MO_BC) + pgam),
                              gfk_pow(dlamc, MO_BC) * mo_gamma(pgam + 1.0f));
                    um = FDIV(acn * mo_gamma((4.0f + MO_BC) + pgam),
                              gfk_pow(dlamc, MO_BC) * mo_gamma(pgam + 4.0f));
                } else if (cat == 1) {
                    float dlamr = gfk_pow(FDIV((MO_RHOW * MO_PI) * dnum,
                                               dmass), 0.333333343f);
                    dlamr = FMAX(dlamr, MO_LAMMINR);
                    dlamr = FMIN(dlamr, MO_LAMMAXR);
                    const float arn = dum * MO_AR;
                    un = FDIV(arn * MO_CONS6, gfk_pow(dlamr, MO_BR));
                    um = FDIV(arn * MO_CONS4, gfk_pow(dlamr, MO_BR));
                } else if (cat == 2) {
                    float dlami = gfk_pow(FDIV(dnum * MO_CONS12, dmass),
                                          FDIV(1.0f, MO_DI));
                    dlami = FMAX(dlami, MO_LAMMINI);
                    dlami = FMIN(dlami, MO_LAMMAXI);
                    const float ain = gfk_pow(FDIV(MO_RHOSU, rho), 0.35f)
                        * MO_AI;
                    un = FDIV(ain * MO_CONS27, gfk_pow(dlami, MO_BI));
                    um = FDIV(ain * MO_CONS28, gfk_pow(dlami, MO_BI));
                } else if (cat == 3) {
                    float dlams = gfk_pow(FDIV(dnum * MO_CONS1, dmass),
                                          FDIV(1.0f, MO_DS));
                    dlams = FMAX(dlams, MO_LAMMINS);
                    dlams = FMIN(dlams, MO_LAMMAXS);
                    const float asn = dum * MO_AS;
                    um = FDIV(asn * MO_CONS3, gfk_pow(dlams, MO_BS));
                    un = FDIV(asn * MO_CONS5, gfk_pow(dlams, MO_BS));
                } else {
                    float dlamg = gfk_pow(FDIV(dnum * MO_CONS2, dmass),
                                          FDIV(1.0f, MO_DG));
                    dlamg = FMAX(dlamg, MO_LAMMING);
                    dlamg = FMIN(dlamg, MO_LAMMAXG);
                    const float agn = dum * MO_AG;
                    um = FDIV(agn * MO_CONS7, gfk_pow(dlamg, MO_BG));
                    un = FDIV(agn * MO_CONS8, gfk_pow(dlamg, MO_BG));
                }
            }
            // :3482-3492 fall-speed limits (cloud has none).
            if (cat == 1) {
                um = FMIN(um, dum * 9.1f);
                un = FMIN(un, dum * 9.1f);
            } else if (cat == 2) {
                um = FMIN(um, gfk_pow(FDIV(MO_RHOSU, rho), 0.35f) * 1.2f);
                un = FMIN(un, gfk_pow(FDIV(MO_RHOSU, rho), 0.35f) * 1.2f);
            } else if (cat == 3) {
                um = FMIN(um, dum * 1.2f);
                un = FMIN(un, dum * 1.2f);
            } else if (cat == 4) {
                um = FMIN(um, dum * 20.0f);
                un = FMIN(un, dum * 20.0f);
            }
            // :3507-3538
            if (k <= nz - 2) {
                if (um < 1.E-10f) um = fm[k + 1];
                if (un < 1.E-10f) un = fn[k + 1];
            }
            fm[k] = um;
            fn[k] = un;
            // :3542-3544 (this category's share of RGVM).
            const float rgvm = GMAX(un, um);
            const int n = (int)(FDIV(rgvm * dt, dz[AT(k)]) + 1.0f);
            nstep = max(n, nstep);
            // :3547-3556
            dm[k] = mo_mul(dmass, rho);
            dn[k] = mo_mul(dnum, rho);
        }
    }
    s_nstep[cat][lane] = nstep;
    __syncthreads();
    int nsteps = 1;
    for (int c = 0; c < 5; ++c) nsteps = max(nsteps, s_nstep[c][lane]);
    // The block steps to its widest column; a column stops at its own.
    int block_steps = 0;
    for (int l = 0; l < 32; ++l) {
        int m = 1;
        for (int c = 0; c < 5; ++c) m = max(m, s_nstep[c][l]);
        block_steps = max(block_steps, m);
    }
    const float fnstep = (float)nsteps;
    float precrt = 0.0f, snowrt = 0.0f, snowprt = 0.0f, grplprt = 0.0f;
    for (int step = 1; step <= block_steps; ++step) {
        const bool on = sediments && step <= nsteps;
        float fall_kts = 0.0f;
        if (on) {
            // :3577-3610 top level.
            int k = nz - 1;
            float fom = mo_mul(fm[k], dm[k]);
            float fon = mo_mul(fn[k], dn[k]);
            float faltndm = mo_div(fom, dz[AT(k)]);
            float faltndn = mo_div(fon, dz[AT(k)]);
            const float rhot = rho_in[AT(k)];
            WP(msten, k) = mo_sub(WP(msten, k),
                                  mo_div(mo_div(faltndm, fnstep), rhot));
            WP(nten, k) = mo_sub(WP(nten, k),
                                 mo_div(mo_div(faltndn, fnstep), rhot));
            dm[k] = mo_sub(dm[k], mo_div(mo_mul(faltndm, dt), fnstep));
            dn[k] = mo_sub(dn[k], mo_div(mo_mul(faltndn, dt), fnstep));
            fall_kts = fom;
            // :3612-3646
            for (k = nz - 2; k >= 0; --k) {
                const float fm_k = mo_mul(fm[k], dm[k]);
                const float fn_k = mo_mul(fn[k], dn[k]);
                faltndm = mo_div(mo_sub(fom, fm_k), dz[AT(k)]);
                faltndn = mo_div(mo_sub(fon, fn_k), dz[AT(k)]);
                const float rho = rho_in[AT(k)];
                WP(msten, k) = mo_add(WP(msten, k),
                                      mo_div(mo_div(faltndm, fnstep), rho));
                WP(nten, k) = mo_add(WP(nten, k),
                                     mo_div(mo_div(faltndn, fnstep), rho));
                dm[k] = mo_add(dm[k], mo_div(mo_mul(faltndm, dt), fnstep));
                dn[k] = mo_add(dn[k], mo_div(mo_mul(faltndn, dt), fnstep));
                fom = fm_k;
                fon = fn_k;
                fall_kts = fm_k;
            }
        }
        s_fall[cat][lane] = fall_kts;
        __syncthreads();
        if (on && cat == 0) {
            // :3660-3665 in WRF's order: R, C, S, I, G.
            const float r = s_fall[1][lane], c = s_fall[0][lane];
            const float i = s_fall[2][lane], s = s_fall[3][lane];
            const float g = s_fall[4][lane];
            precrt = mo_add(precrt, mo_div(mo_mul(mo_add(mo_add(mo_add(
                mo_add(r, c), s), i), g), dt), fnstep));
            snowrt = mo_add(snowrt, mo_div(mo_mul(mo_add(mo_add(s, i), g),
                                                  dt), fnstep));
            snowprt = mo_add(snowprt, mo_div(mo_mul(mo_add(i, s), dt),
                                             fnstep));
            grplprt = mo_add(grplprt, mo_div(mo_mul(g, dt), fnstep));
        }
        __syncthreads();
    }
    // Driver :901-908.
    if (valid && cat == 0) {
        const int s = col;
        rainnc[s] = mo_add(rainnc[s], precrt);
        rainncv[s] = precrt;
        snownc[s] = mo_add(snownc[s], snowprt);
        snowncv[s] = snowprt;
        graupelnc[s] = mo_add(graupelnc[s], grplprt);
        graupelncv[s] = grplprt;
        sr[s] = mo_div(snowrt, mo_add(precrt, 1.E-12f));
    }
#undef WP
#undef AT
}

#define MO_SEDIMENT_PARAMETERS                                               \
    const float* __restrict__ pressure, const float* __restrict__ dz,        \
    const float* __restrict__ rho_in,                                        \
    const float* __restrict__ qc, const float* __restrict__ qr,              \
    const float* __restrict__ qi, const float* __restrict__ qs,              \
    const float* __restrict__ qg, const float* __restrict__ nc,              \
    const float* __restrict__ nr, const float* __restrict__ ni,              \
    const float* __restrict__ ns, const float* __restrict__ ng,              \
    float* __restrict__ work, const int* __restrict__ ltrue,                 \
    float* __restrict__ rainnc, float* __restrict__ rainncv,                 \
    float* __restrict__ snownc, float* __restrict__ snowncv,                 \
    float* __restrict__ graupelnc, float* __restrict__ graupelncv,           \
    float* __restrict__ sr, float dt, int rimed, int nz, int ncol,              int col0, int ccols
#define MO_SEDIMENT_ARGUMENTS                                                \
    pressure, dz, rho_in, qc, qr, qi, qs, qg, nc, nr, ni, ns, ng, work,      \
    ltrue, rainnc, rainncv, snownc, snowncv, graupelnc, graupelncv, sr, dt,  \
    rimed, nz, ncol, col0, ccols

extern "C" __global__ void mo_sediment_64(MO_SEDIMENT_PARAMETERS)
{
    mo_sediment_impl<MORR_KMAX_SHALLOW>(MO_SEDIMENT_ARGUMENTS);
}

extern "C" __global__ void mo_sediment_256(MO_SEDIMENT_PARAMETERS)
{
    mo_sediment_impl<MORR_KMAX_GENERIC>(MO_SEDIMENT_ARGUMENTS);
}

#undef MO_SEDIMENT_ARGUMENTS
#undef MO_SEDIMENT_PARAMETERS

// ---------------------------------------------------------------------
// The final K loop (:3669-4054) and the driver's write-back (:842-866).
// One thread per cell.  A column whose LTRUE stayed 0 skipped to 400 in
// WRF: its state is the process stage's and only TH = T/PII is written
// (its effective radii, which WRF does not return, are the 25 micron
// no-hydrometeor values).
// ---------------------------------------------------------------------
extern "C" __global__ void mo_finalize(
    float* __restrict__ theta, const float* __restrict__ pii,
    const float* __restrict__ pressure, const float* __restrict__ rho_in,
    float* __restrict__ qv, float* __restrict__ qc, float* __restrict__ qr,
    float* __restrict__ qi, float* __restrict__ qs, float* __restrict__ qg,
    float* __restrict__ nc, float* __restrict__ nr, float* __restrict__ ni,
    float* __restrict__ ns, float* __restrict__ ng,
    float* __restrict__ work, const int* __restrict__ ltrue,
    float* __restrict__ effc, float* __restrict__ effi,
    float* __restrict__ effs, float* __restrict__ effr, float dt,
    int rimed, long long nlocal, int ncol, int col0, int ccols)
{
    const long long li = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (li >= nlocal) return;
    const int col = col0 + (int)(li % ccols);
    const long long idx = (li / ccols) * ncol + col;
#define W(p) work[(long long)(p) * nlocal + li]
    float t3d = W(MO_T3D);
    if (ltrue[col] == 0) {
        theta[idx] = FDIV(t3d, pii[idx]);
        effc[idx] = 25.0f; effi[idx] = 25.0f;
        effs[idx] = 25.0f; effr[idx] = 25.0f;
        return;
    }
    const float pres = pressure[idx], rho = rho_in[idx];
    // XXLV/XXLS are the entry temperature's (:1298-1302), recomputed.
    const float t_entry = theta[idx] * pii[idx];
    const float xxlv = 3.1484E6f - t_entry * 2370.0f;
    const float xxls = (3.15E6f - t_entry * 2370.0f) + 0.3337E6f;
    const float xlf = xxls - xxlv;
    const float cpm = W(MO_CPM);
    const float lami_p = W(MO_LAMI);
    float qv3d = qv[idx], qc3d = qc[idx], qr3d = qr[idx], qi3d = qi[idx];
    float qni3d = qs[idx], qg3d = qg[idx], nc3d = nc[idx], nr3d = nr[idx];
    float ni3d = ni[idx], ns3d = ns[idx], ng3d = ng[idx];
    const float t3dten = W(MO_T3DTEN), qv3dten = W(MO_QV3DTEN);
    float qc3dten = W(MO_QC3DTEN), qi3dten = W(MO_QI3DTEN);
    float qni3dten = W(MO_QNI3DTEN), qr3dten = W(MO_QR3DTEN);
    float ni3dten = W(MO_NI3DTEN), ns3dten = W(MO_NS3DTEN);
    const float nr3dten = W(MO_NR3DTEN);
    float qg3dten = W(MO_QG3DTEN);
    const float ng3dten = W(MO_NG3DTEN), nc3dten = W(MO_NC3DTEN);

    // :3673-3677
    qr3dten = mo_add(qr3dten, W(MO_QRSTEN));
    qi3dten = mo_add(qi3dten, W(MO_QISTEN));
    qc3dten = mo_add(qc3dten, W(MO_QCSTEN));
    qg3dten = mo_add(qg3dten, W(MO_QGSTEN));
    qni3dten = mo_add(qni3dten, W(MO_QNISTEN));
    // :3683-3690 large cloud ice to snow.
    if (qi3d >= MO_QSMALL && t3d < 273.15f && lami_p >= 1.E-10f) {
        if (FDIV(1.0f, lami_p) >= MO_DCS * 2.0f) {
            qni3dten = (qni3dten + FDIV(qi3d, dt)) + qi3dten;
            ns3dten = (ns3dten + FDIV(ni3d, dt)) + ni3dten;
            qi3dten = FDIV(-qi3d, dt);
            ni3dten = FDIV(-ni3d, dt);
        }
    }
    // :3695-3711
    qc3d = mo_add(qc3d, mo_mul(qc3dten, dt));
    qi3d = mo_add(qi3d, mo_mul(qi3dten, dt));
    qni3d = mo_add(qni3d, mo_mul(qni3dten, dt));
    qr3d = mo_add(qr3d, mo_mul(qr3dten, dt));
    nc3d = mo_add(nc3d, mo_mul(nc3dten, dt));
    ni3d = mo_add(ni3d, mo_mul(ni3dten, dt));
    ns3d = mo_add(ns3d, mo_mul(ns3dten, dt));
    nr3d = mo_add(nr3d, mo_mul(nr3dten, dt));
    qg3d = mo_add(qg3d, mo_mul(qg3dten, dt));
    ng3d = mo_add(ng3d, mo_mul(ng3dten, dt));
    t3d = mo_add(t3d, mo_mul(t3dten, dt));
    qv3d = mo_add(qv3d, mo_mul(qv3dten, dt));
    // :3716-3727
    const float evs = FMIN(pres * 0.99f, mo_polysvp(t3d, 0));
    float eis = FMIN(pres * 0.99f, mo_polysvp(t3d, 1));
    if (eis > evs) eis = evs;
    const float qvs = FDIV(evs * MO_EP_2, pres - evs);
    const float qvi = FDIV(eis * MO_EP_2, pres - eis);
    const float qvqvs = FDIV(qv3d, qvs);
    const float qvqvsi = FDIV(qv3d, qvi);
    // :3732-3761
    if (qvqvs < 0.9f) {
        if (qr3d < 1.E-8f) {
            qv3d = mo_add(qv3d, qr3d);
            t3d = t3d - FDIV(qr3d * xxlv, cpm);
            qr3d = 0.0f;
        }
        if (qc3d < 1.E-8f) {
            qv3d = mo_add(qv3d, qc3d);
            t3d = t3d - FDIV(qc3d * xxlv, cpm);
            qc3d = 0.0f;
        }
    }
    if (qvqvsi < 0.9f) {
        if (qi3d < 1.E-8f) {
            qv3d = mo_add(qv3d, qi3d);
            t3d = t3d - FDIV(qi3d * xxls, cpm);
            qi3d = 0.0f;
        }
        if (qni3d < 1.E-8f) {
            qv3d = mo_add(qv3d, qni3d);
            t3d = t3d - FDIV(qni3d * xxls, cpm);
            qni3d = 0.0f;
        }
        if (qg3d < 1.E-8f) {
            qv3d = mo_add(qv3d, qg3d);
            t3d = t3d - FDIV(qg3d * xxls, cpm);
            qg3d = 0.0f;
        }
    }
    // :3766-3790
    if (qc3d < MO_QSMALL) { qc3d = 0.0f; nc3d = 0.0f; }
    if (qr3d < MO_QSMALL) { qr3d = 0.0f; nr3d = 0.0f; }
    if (qi3d < MO_QSMALL) { qi3d = 0.0f; ni3d = 0.0f; }
    if (qni3d < MO_QSMALL) { qni3d = 0.0f; ns3d = 0.0f; }
    if (qg3d < MO_QSMALL) { qg3d = 0.0f; ng3d = 0.0f; }

    float lami = lami_p, lamr = 0.0f, lamc = 0.0f, lams = 0.0f;
    float pgam = 0.0f;
    if (!(qc3d < MO_QSMALL && qi3d < MO_QSMALL && qni3d < MO_QSMALL
          && qr3d < MO_QSMALL && qg3d < MO_QSMALL)) {
        // :3803-3812 melt cloud ice above freezing.
        if (qi3d >= MO_QSMALL && t3d >= 273.15f) {
            qr3d = qr3d + qi3d;
            t3d = t3d - FDIV(qi3d * xlf, cpm);
            qi3d = 0.0f;
            nr3d = nr3d + ni3d;
            ni3d = 0.0f;
        }
        // :3819-3825 homogeneous freezing of droplets (ILIQ = 0).
        if (t3d <= 233.15f && qc3d >= MO_QSMALL) {
            qi3d = qi3d + qc3d;
            t3d = t3d + FDIV(qc3d * xlf, cpm);
            qc3d = 0.0f;
            ni3d = ni3d + nc3d;
            nc3d = 0.0f;
        }
        // :3831-3837 freezing of rain to graupel (IGRAUP = 0).
        if (t3d <= 233.15f && qr3d >= MO_QSMALL) {
            qg3d = qg3d + qr3d;
            t3d = t3d + FDIV(qr3d * xlf, cpm);
            qr3d = 0.0f;
            ng3d = ng3d + nr3d;
            nr3d = 0.0f;
        }
        // :3855-3859
        ni3d = FMAXK(ni3d, 0.0f);
        ns3d = FMAXK(ns3d, 0.0f);
        nc3d = FMAXK(nc3d, 0.0f);
        nr3d = FMAXK(nr3d, 0.0f);
        ng3d = FMAXK(ng3d, 0.0f);
        // :3864-3885
        if (qi3d >= MO_QSMALL) {
            lami = gfk_pow(FDIV(ni3d * MO_CONS12, qi3d), FDIV(1.0f, MO_DI));
            if (lami < MO_LAMMINI) {
                lami = MO_LAMMINI;
                const float n0i = FDIV(mo_powi4(lami) * qi3d, MO_CONS12);
                ni3d = FDIV(n0i, lami);
            } else if (lami > MO_LAMMAXI) {
                lami = MO_LAMMAXI;
                const float n0i = FDIV(mo_powi4(lami) * qi3d, MO_CONS12);
                ni3d = FDIV(n0i, lami);
            }
        }
        // :3890-3911
        if (qr3d >= MO_QSMALL) {
            lamr = gfk_pow(FDIV((MO_RHOW * MO_PI) * nr3d, qr3d),
                           0.333333343f);
            if (lamr < MO_LAMMINR) {
                lamr = MO_LAMMINR;
                const float n0rr = FDIV(mo_powi4(lamr) * qr3d,
                                        MO_RHOW * MO_PI);
                nr3d = FDIV(n0rr, lamr);
            } else if (lamr > MO_LAMMAXR) {
                lamr = MO_LAMMAXR;
                const float n0rr = FDIV(mo_powi4(lamr) * qr3d,
                                        MO_RHOW * MO_PI);
                nr3d = FDIV(n0rr, lamr);
            }
        }
        // :3918-3949
        if (qc3d >= MO_QSMALL) {
            const float dum = FDIV(pres, t3d * 287.15f);
            pgam = (dum * FDIV(nc3d, 1.E6f)) * 0.0005714f + 0.2714f;
            pgam = FDIV(1.0f, pgam * pgam) - 1.0f;
            pgam = FMAXK(pgam, 2.0f);
            pgam = FMINK(pgam, 10.0f);
            lamc = gfk_pow(FDIV((nc3d * MO_CONS26) * mo_gamma(pgam + 4.0f),
                                qc3d * mo_gamma(pgam + 1.0f)), 0.333333343f);
            const float lammin = FDIV(pgam + 1.0f, 60.E-6f);
            const float lammax = FDIV(pgam + 1.0f, 1.E-6f);
            if (lamc < lammin) {
                lamc = lammin;
                nc3d = FDIV(gfk_exp(((gfk_log(lamc) * 3.0f + gfk_log(qc3d))
                                     + gfk_log(mo_gamma(pgam + 1.0f)))
                                    - gfk_log(mo_gamma(pgam + 4.0f))),
                            MO_CONS26);
            } else if (lamc > lammax) {
                lamc = lammax;
                nc3d = FDIV(gfk_exp(((gfk_log(lamc) * 3.0f + gfk_log(qc3d))
                                     + gfk_log(mo_gamma(pgam + 1.0f)))
                                    - gfk_log(mo_gamma(pgam + 4.0f))),
                            MO_CONS26);
            }
        }
        // :3954-3974
        if (qni3d >= MO_QSMALL) {
            lams = gfk_pow(FDIV(ns3d * MO_CONS1, qni3d), FDIV(1.0f, MO_DS));
            if (lams < MO_LAMMINS) {
                lams = MO_LAMMINS;
                const float n0s = FDIV(mo_powi4(lams) * qni3d, MO_CONS1);
                ns3d = FDIV(n0s, lams);
            } else if (lams > MO_LAMMAXS) {
                lams = MO_LAMMAXS;
                const float n0s = FDIV(mo_powi4(lams) * qni3d, MO_CONS1);
                ns3d = FDIV(n0s, lams);
            }
        }
        // :3979-4000
        if (qg3d >= MO_QSMALL) {
            float lamg = gfk_pow(FDIV(ng3d * MO_CONS2, qg3d),
                                 FDIV(1.0f, MO_DG));
            if (lamg < MO_LAMMING) {
                lamg = MO_LAMMING;
                const float n0g = FDIV(mo_powi4(lamg) * qg3d, MO_CONS2);
                ng3d = FDIV(n0g, lamg);
            } else if (lamg > MO_LAMMAXG) {
                lamg = MO_LAMMAXG;
                const float n0g = FDIV(mo_powi4(lamg) * qg3d, MO_CONS2);
                ng3d = FDIV(n0g, lamg);
            }
        }
    }
    // 500 CONTINUE: :4006-4035 effective radii.
    effi[idx] = qi3d >= MO_QSMALL
        ? FDIV(FDIV(3.0f, lami), 2.0f) * 1.E6f : 25.0f;
    effs[idx] = qni3d >= MO_QSMALL
        ? FDIV(FDIV(3.0f, lams), 2.0f) * 1.E6f : 25.0f;
    effr[idx] = qr3d >= MO_QSMALL
        ? FDIV(FDIV(3.0f, lamr), 2.0f) * 1.E6f : 25.0f;
    effc[idx] = qc3d >= MO_QSMALL
        ? FDIV(FDIV(FDIV(mo_gamma(pgam + 4.0f), mo_gamma(pgam + 3.0f)),
                    lamc), 2.0f) * 1.E6f
        : 25.0f;
    // :4044-4054
    ni3d = FMIN(ni3d, FDIV(0.3E6f, rho));
    nc3d = FDIV(MO_NDCNST * 1.E6f, rho);
    // Driver write-back.
    qv[idx] = qv3d; qc[idx] = qc3d; qr[idx] = qr3d; qi[idx] = qi3d;
    qs[idx] = qni3d; qg[idx] = qg3d; nc[idx] = nc3d; nr[idx] = nr3d;
    ni[idx] = ni3d; ns[idx] = ns3d; ng[idx] = ng3d;
    theta[idx] = FDIV(t3d, pii[idx]);
    W(MO_T3D) = t3d;   // T1D, the reflectivity's temperature
#undef W
}

// WRF phy_prep's Exner function, (p/p1000mb)**rcp with WOOF's own powf.
extern "C" __global__ void mo_exner(const float* __restrict__ p,
                                    float* __restrict__ pii, float p0,
                                    float rcp, long long n)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    pii[i] = gfk_pow(FDIV(p[i], p0), rcp);
}

// ---------------------------------------------------------------------
// refl10cm_hm (:4502-4675) and module_mp_radar's melting-snow scattering
// (rayleigh_soak_wetgraupel, get_m_mix_nested, m_complex_maxwellgarnett):
// complex arithmetic as gfortran expands it (-fcx-fortran-rules),
// products in full, quotients by Smith's range-reduced division.
// ---------------------------------------------------------------------
struct moz { double re, im; };

__device__ __forceinline__ moz moz_make(double re, double im)
{ moz z; z.re = re; z.im = im; return z; }

__device__ __forceinline__ moz moz_mul(moz a, moz b)
{
    return moz_make(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}

__device__ __forceinline__ moz moz_rmul(double s, moz b)
{
    return moz_make(s * b.re - 0.0 * b.im, s * b.im + 0.0 * b.re);
}

__device__ __forceinline__ moz moz_sq(moz a)
{
    return moz_make(a.re * a.re - a.im * a.im, (a.re * a.im) * 2.0);
}

__device__ __forceinline__ moz moz_div(moz a, moz b)
{
    if (fabs(b.re) < fabs(b.im)) {
        const double ratio = DDIV(b.re, b.im);
        const double div = b.im + b.re * ratio;
        return moz_make(DDIV(a.im + a.re * ratio, div),
                        DDIV(a.im * ratio - a.re, div));
    }
    const double ratio = DDIV(b.im, b.re);
    const double div = b.re + b.im * ratio;
    return moz_make(DDIV(a.re + a.im * ratio, div),
                    DDIV(a.im - a.re * ratio, div));
}

__device__ __forceinline__ double moz_abs(moz a) { return hypot(a.re, a.im); }

__device__ __forceinline__ moz moz_log(moz a)
{
    return moz_make(glibc_log(hypot(a.re, a.im)), atan2(a.im, a.re));
}

__device__ __forceinline__ moz moz_sqrt(moz a)
{
    if (a.re == 0.0 && a.im == 0.0) return moz_make(0.0, a.im);
    const double m = hypot(a.re, a.im);
    if (a.re > 0.0) {
        const double r = sqrt(0.5 * (m + a.re));
        return moz_make(r, DDIV(a.im, 2.0 * r));
    }
    const double s = sqrt(0.5 * (m - a.re));
    return moz_make(DDIV(fabs(a.im), 2.0 * s), copysign(s, a.im));
}

__device__ bool mo_maxwellgarnett(double vol1, double vol2, double vol3,
                                  moz m1, moz m2, moz m3, moz* out)
{
    if (fabs(vol1 + vol2 + vol3 - 1.0) > 1e-6) return false;
    const moz m1t = moz_sq(m1), m2t = moz_sq(m2), m3t = moz_sq(m3);
    const moz d2 = moz_make(m2t.re - m1t.re, m2t.im - m1t.im);
    const moz l2 = moz_log(moz_div(m2t, m1t));
    moz t2 = moz_mul(moz_div(m2t, d2), l2);
    t2 = moz_make(t2.re - 1.0, t2.im);
    const moz beta2 = moz_mul(moz_div(moz_rmul(2.0, m1t), d2), t2);
    const moz d3 = moz_make(m3t.re - m1t.re, m3t.im - m1t.im);
    const moz l3 = moz_log(moz_div(m3t, m1t));
    moz t3 = moz_mul(moz_div(m3t, d3), l3);
    t3 = moz_make(t3.re - 1.0, t3.im);
    const moz beta3 = moz_mul(moz_div(moz_rmul(2.0, m1t), d3), t3);
    const double w1 = 1.0 - vol2 - vol3;
    const moz a = moz_rmul(w1, m1t);
    const moz b = moz_mul(moz_rmul(vol2, beta2), m2t);
    const moz c = moz_mul(moz_rmul(vol3, beta3), m3t);
    const moz num = moz_make((a.re + b.re) + c.re, (a.im + b.im) + c.im);
    const moz e = moz_rmul(vol2, beta2);
    const moz f = moz_rmul(vol3, beta3);
    const moz den = moz_make((w1 + e.re) + f.re, (0.0 + e.im) + f.im);
    *out = moz_sqrt(moz_div(num, den));
    return true;
}

// rayleigh_soak_wetgraupel with Morrison's strings for both snow and
// graupel (host 'air', matrix 'water', hostmatrix 'icewater', spheroidal
// inclusions, Maxwell-Garnett).
__device__ double mo_rayleigh_soak_wetgraupel(double x_g, double a_geo,
    double b_geo, double fmelt, double meltratio_outside)
{
    const moz m_w = moz_make(MO_M_W_0(1), MO_M_W_0(2));
    const moz m_i = moz_make(MO_M_I_0(1), MO_M_I_0(2));
    const moz m_air = moz_make(1.0, 0.0);
    const double PIx = 3.1415926535897932384626434;
    const double fm = fmax(fmin(fmelt, 1.0), 0.0);
    double mra = fmax(fmin(meltratio_outside, 1.0), 0.0);
    mra = mra + (1.0 - mra) * fm;
    const double x_w = x_g * fm;
    const double D_g = a_geo * glibc_pow(x_g, b_geo);
    if (!(D_g >= 1e-12)) return 0.0;
    double vg = DDIV(PIx, 6.) * (D_g * (D_g * D_g));
    const double rhog = fmax(fmin(DDIV(x_g, vg), 900.0), 10.0);
    vg = DDIV(x_g, rhog);
    const double meltratio_outside_grenz = 1.0 - DDIV(rhog, 1000.);
    double volg;
    if (mra <= meltratio_outside_grenz) {
        volg = vg * (1.0 - mra * fm);
    } else {
        const double fmgrenz = DDIV(900.0 - rhog,
            mra * 900.0 - rhog + DDIV(900.0 * rhog, 1000.));
        if (fm <= fmgrenz) {
            volg = (1.0 - mra * fm) * vg;
        } else {
            volg = DDIV(x_g - x_w, 900.0) + DDIV(x_w, 1000.);
        }
    }
    const double D_large = glibc_pow(DDIV(6.0, PIx) * volg,
                                     (double)FDIV(1.f, 3.f));
    const double volice = DDIV(x_g - x_w, volg * 900.0);
    const double volwater = DDIV(x_w, 1000. * volg);
    const double volair = 1.0 - volice - volwater;
    const double vol1 = DDIV(volice, fmax(volice + volwater, 1e-10));
    const double vol2 = 1.0 - vol1;
    moz mtmp, m_core;
    if (!mo_maxwellgarnett(vol2, 0.0, vol1, m_w, m_air, m_i, &mtmp))
        return 0.0;
    if (!mo_maxwellgarnett(1.0 - volair, volair, 0.0, mtmp, m_air,
                           moz_rmul(2.0, m_air), &m_core))
        return 0.0;
    const moz m2 = moz_sq(m_core);
    const moz q = moz_div(moz_make(m2.re - 1.0, m2.im),
                          moz_make(m2.re + 2.0, m2.im + 0.0));
    const double a = moz_abs(q);
    const double d3 = D_large * D_large * D_large;
    return DDIV(a * a * MO_PI5 * (d3 * d3), MO_LAMDA4);
}

// WRF's DOUBLE log10, WOOF's own binary64 routine: y*log10_2lo +
// ivln10*log(x), then + y*log10_2hi, with the zero, negative, subnormal
// and inf/NaN cases of the oracle host's log10 (a NaN state reaches dBZ in
// a NaN column).
__device__ __forceinline__ double mo_log10d(double x)
{
    const double ivln10 = __longlong_as_double(0x3FDBCB7B1526E50ELL);
    const double log10_2hi = __longlong_as_double(0x3FD34413509F6000LL);
    const double log10_2lo = __longlong_as_double(0x3D59FEF311F12B36LL);
    const double two54 = 1.80143985094819840000e+16;
    long long hx = __double_as_longlong(x);
    int kk = 0;
    if (hx < 0x0010000000000000LL) {
        if ((hx & 0x7fffffffffffffffLL) == 0) return DDIV(-two54, fabs(x));
        if (hx < 0) return DDIV(x - x, x - x);
        kk -= 54;
        x *= two54;
        hx = __double_as_longlong(x);
    }
    if (hx >= 0x7ff0000000000000LL) return x + x;
    const int k = kk + (int)(hx >> 52) - 1023;
    const long long i = (long long)(((unsigned long long)(long long)k
                                     & 0x8000000000000000ULL) >> 63);
    hx = (hx & 0x000fffffffffffffLL) | ((0x3ffLL - i) << 52);
    const double y = (double)(k + i);
    const double xm = __longlong_as_double(hx);
    const double z = y * log10_2lo + ivln10 * glibc_log(xm);
    return z + y * log10_2hi;
}

// One level's density and density-scaled moments, :4538-4581.
struct mo_rl { float rr, nr, rs, ns, rg, ng; bool lqr, lqs, lqg; };

__device__ __forceinline__ mo_rl mo_refl_level(
    const float* qv1d, const float* qr1d, const float* nr1d,
    const float* qs1d, const float* ns1d, const float* qg1d,
    const float* ng1d, const float* t1d, const float* p1d, long long at)
{
    mo_rl v;
    const float temp = t1d[at];
    const float qv = FMAXK(qv1d[at], 1.E-10f);
    const float pres = p1d[at];
    const float rho = FDIV(pres * 0.622f, (temp * MO_R) * (qv + 0.622f));
    v.lqr = qr1d[at] > 1.E-9f;
    v.lqs = qs1d[at] > 1.E-9f;
    v.lqg = qg1d[at] > 1.E-9f;
    v.rr = v.lqr ? qr1d[at] * rho : 1.E-12f;
    v.nr = v.lqr ? nr1d[at] * rho : 1.E-12f;
    v.rs = v.lqs ? qs1d[at] * rho : 1.E-12f;
    v.ns = v.lqs ? ns1d[at] * rho : 1.E-12f;
    v.rg = v.lqg ? qg1d[at] * rho : 1.E-12f;
    v.ng = v.lqg ? ng1d[at] * rho : 1.E-12f;
    return v;
}

extern "C" __global__ void mo_refl10cm(
    const float* __restrict__ qv1d, const float* __restrict__ qr1d,
    const float* __restrict__ nr1d, const float* __restrict__ qs1d,
    const float* __restrict__ ns1d, const float* __restrict__ qg1d,
    const float* __restrict__ ng1d, const float* __restrict__ t1d,
    const float* __restrict__ p1d, float* __restrict__ refl, int rimed,
    int nz, int columns)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= columns) return;
#define AT(k) ((long long)(k) * (long long)columns + (long long)col)
#define LEVEL(k) mo_refl_level(qv1d, qr1d, nr1d, qs1d, ns1d, qg1d, ng1d, \
                               t1d, p1d, AT(k))
    const int kts = 0, kte = nz - 1;
    // :4587-4597 melting level.
    bool melti = false;
    int k_0 = kts;
    for (int k = kte - 1; k >= kts; --k) {
        const bool lqr = qr1d[AT(k)] > 1.E-9f;
        const bool lqs1 = qs1d[AT(k + 1)] > 1.E-9f;
        const bool lqg1 = qg1d[AT(k + 1)] > 1.E-9f;
        if ((t1d[AT(k)] > 273.15f) && lqr && (lqs1 || lqg1)) {
            k_0 = max(k + 1, k_0);
            melti = true;
            break;
        }
    }
    const mo_rl top = LEVEL(k_0);
    for (int k = kte; k >= kts; --k) {
        const mo_rl v = LEVEL(k);
        float ze_rain = 1.e-22f, ze_snow = 1.e-22f, ze_graupel = 1.e-22f;
        if (v.lqr) {
            const double lamr = (double)gfk_pow(FDIV(((MO_XCRG(3) * MO_XAM_R)
                                                      * MO_XORG2) * v.nr,
                                                     v.rr), MO_XOBMR);
            const double ilamr = DDIV(1.0, lamr);
            const double N0_r = (double)(v.nr * MO_XORG2)
                * glibc_pow(lamr, (double)MO_XCRE(2));
            ze_rain = (float)((N0_r * (double)MO_XCRG(4))
                              * glibc_pow(ilamr, (double)MO_XCRE(4)));
        }
        double ilams = 0.0, N0_s = 0.0, ilamg = 0.0, N0_g = 0.0;
        if (v.lqs) {
            const double lams = (double)gfk_pow(FDIV(((MO_XCSG(3) * MO_XAM_S)
                                                      * MO_XOSG2) * v.ns,
                                                     v.rs), MO_XOBMS);
            ilams = DDIV(1.0, lams);
            N0_s = (double)(v.ns * MO_XOSG2)
                * glibc_pow(lams, (double)MO_XCSE(2));
            const float c = (FDIV(MO_XAM_S, 900.0f)
                             * 6.9029140472412109375e-1f)
                * FDIV(MO_XAM_S, 900.0f);
            ze_snow = (float)((((double)c * N0_s) * (double)MO_XCSG(4))
                              * glibc_pow(ilams, (double)MO_XCSE(4)));
        }
        if (v.lqg) {
            const double lamg = (double)gfk_pow(FDIV(((MO_XCGG(3) * MO_XAM_G)
                                                      * MO_XOGG2) * v.ng,
                                                     v.rg), MO_XOBMG);
            ilamg = DDIV(1.0, lamg);
            N0_g = (double)(v.ng * MO_XOGG2)
                * glibc_pow(lamg, (double)MO_XCGE(2));
            const float c = (FDIV(MO_XAM_G, 900.0f)
                             * 6.9029140472412109375e-1f)
                * FDIV(MO_XAM_G, 900.0f);
            ze_graupel = (float)((((double)c * N0_g) * (double)MO_XCGG(4))
                                 * glibc_pow(ilamg, (double)MO_XCGE(4)));
        }
        // :4626-4668 melting snow and graupel below the melting level.
        if (melti && k_0 >= kts + 1 && k <= k_0 - 1) {
            if (v.lqs && top.lqs) {
                const double r = 1.0 - (double)FDIV(v.rs, top.rs);
                const double fmelt_s = FMAXK(FMINK(r, 0.99), 0.005);
                double eta = 0.0;
                const double lams = DDIV(1.0, ilams);
                for (int n = 1; n <= 50; ++n) {
                    const double x = (double)MO_XAM_S
                        * glibc_pow(MO_XXDS(n), (double)MO_XBM_S);
                    const double CBACK = mo_rayleigh_soak_wetgraupel(
                        x, (double)MO_XOCMS, (double)MO_XOBMS, fmelt_s, 0.9);
                    const double f_d = N0_s
                        * glibc_pow(MO_XXDS(n), (double)MO_XMU_S)
                        * glibc_exp(-(lams * MO_XXDS(n)));
                    eta = eta + f_d * CBACK * MO_SIMPSON(n) * MO_XDTS(n);
                }
                ze_snow = (float)(DDIV(MO_LAMDA4, MO_PI5 * MO_K_W) * eta);
            }
            if (v.lqg && top.lqg) {
                const double r = 1.0 - (double)FDIV(v.rg, top.rg);
                const double fmelt_g = FMAXK(FMINK(r, 0.99), 0.005);
                double eta = 0.0;
                const double lamg = DDIV(1.0, ilamg);
                for (int n = 1; n <= 50; ++n) {
                    const double x = (double)MO_XAM_G
                        * glibc_pow(MO_XXDG(n), (double)MO_XBM_G);
                    const double CBACK = mo_rayleigh_soak_wetgraupel(
                        x, (double)MO_XOCMG, (double)MO_XOBMG, fmelt_g, 0.9);
                    const double f_d = N0_g
                        * glibc_pow(MO_XXDG(n), (double)MO_XMU_G)
                        * glibc_exp(-(lamg * MO_XXDG(n)));
                    eta = eta + f_d * CBACK * MO_SIMPSON(n) * MO_XDTG(n);
                }
                ze_graupel = (float)(DDIV(MO_LAMDA4, MO_PI5 * MO_K_W) * eta);
            }
        }
        // :4670-4672, driver :916.
        const float dbz = (float)(mo_log10d((double)((ze_rain + ze_snow)
                                                     + ze_graupel) * 1.e18)
                                  * 10.);
        refl[AT(k)] = FMAXK(dbz, -35.0f);
    }
#undef LEVEL
#undef AT
}
