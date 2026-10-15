// gpuwm/core/kernels/nsas.cu
//
// New SAS (YSU) cumulus, cu_physics = 96, with its built-in NSCV shallow
// convection: WRF v4.6.1 phys/module_cu_nsas.F (sha256 092cfc66...b911a),
// cu_nsas + nsas2d + nscv2d + fpvs, one thread per column, in WRF's operation
// order.  Graded word for word against the scalar WRF build's own
// module_cu_nsas.o by tools/nsas_wrf461_oracle and tests/test_nsas_*.
//
// Every routine keeps WRF's names and line references so a reviewer can hold
// the two side by side.  Column arrays are 1-based in k, like the Fortran.
//
// ARITHMETIC.  Every expression is written in Fortran's left-to-right parse
// order with the parentheses it implies; the module compiles with
// --fmad=false (gpuwm/core/kernels/__init__.py), so no multiply-add is
// contracted.  x**2 and x**3 are the products gfortran expands them to
// ((x*x) and (x*x)*x); REAL**REAL and EXP are WOOF's own float32 routines
// gfk_pow / gfk_exp (glibc_flt32.cuh); SQRT is IEEE sqrtf.
//
// The deep search limits are reduced over each complete i-row before the
// column launch, matching module_cu_nsas.F:619-631 across workspace tiles.



// module_model_constants.F values WRF's driver hands the scheme
// (module_cumulus_driver.F:1504-1507).  G, RD, RV, CP and XLV come from the
// engine's constants table (gpuwm/core/constants.py, the kernel preamble).




// nsas2d tunables, module_cu_nsas.F:411-425




















// nscv2d tunables, module_cu_nsas.F:2332-2336 and :2397-2401








// ---------------------------------------------------------------------------
// The per-thread column workspace.  Slot s, level k (0 .. nz+1) of the column
// on tile thread t lives at ws[(s*(nz+2) + k)*nthr + t]: consecutive threads
// touch consecutive words.  nscv2d reuses nsas2d's slots after nsas2d has
// returned; every nscv2d read is of a level nscv2d itself wrote first
// (levels 1..kmax), so the reuse cannot leak a deep value into the shallow
// routine.
// ---------------------------------------------------------------------------
enum NsasSlot {
    // cu_nsas column state (module_cu_nsas.F:119-130)
    S_DEL = 0, S_PRSL, S_PRSI, S_ZL, S_ZII, S_DOT, S_T1, S_Q1, S_U1, S_V1,
    S_QC2, S_QI2, S_PRSLK,
    // nsas2d locals (module_cu_nsas.F:458-519)
    S_P, S_ZI, S_UO, S_VO, S_TO, S_QO, S_HCKO, S_QCKO, S_ETA, S_ETAD,
    S_QRCDO, S_PWO, S_PWDO, S_QESO, S_TVO, S_DBYO, S_HEO, S_HESO, S_QRCD,
    S_DELLAH, S_DELLAQ, S_HCDO, S_QCDO, S_XHCD, S_XQCD, S_XLAMB, S_QCIRS,
    S_DELLAL, S_FENT1, S_FENT2, S_FRH, S_UCKO, S_VCKO, S_UCDO, S_VCDO,
    S_DELLAU, S_DELLAV,
    NSAS_SLOTS
};
// gpuwm/core/column_cumulus.py sizes the workspace from its table row
// (workspace_slots = 50); the two cannot drift apart silently.
#define NSAS_WS_SLOTS 50
static_assert(NSAS_SLOTS == NSAS_WS_SLOTS, "nsas workspace slot count moved");
// nscv2d's own names for the slots it borrows.
#define S_PO     S_QRCDO
#define S_XLAMUE S_XLAMB
#define S_THX    S_FRH

struct NsasCol {
    float* ws;
    int nz;
    int nthr;
    int t;
    __device__ __forceinline__ float& a(int s, int k) const {
        return ws[((size_t)s * (size_t)(nz + 2) + (size_t)k) * (size_t)nthr
                  + (size_t)t];
    }
};

// module_cu_nsas.F:2082-2122, with ice = 1 at every call site.
__device__ float nsas_fpvs(float t, float rd, float rv, float cvap,
                           float cliq, float cice, float hvap, float hsub,
                           float psat, float t0c)
{
    (void)rd;
    const float ttp = t0c + 0.01f;
    const float dldt = cvap - cliq;
    const float xa = -(dldt / rv);
    const float xb = xa + hvap / (rv * ttp);
    const float dldti = cvap - cice;
    const float xai = -(dldti / rv);
    const float xbi = xai + hsub / (rv * ttp);
    float tr = ttp / t;
    float fpvs;
    if (t < ttp) {
        fpvs = psat * gfk_pow(tr, xai) * gfk_exp(xbi * (1.0f - tr));
    } else {
        fpvs = psat * gfk_pow(tr, xa) * gfk_exp(xb * (1.0f - tr));
    }
    if (t < 180.0f) {
        tr = ttp / 180.0f;
        if (t < ttp) {
            fpvs = psat * gfk_pow(tr, xai) * gfk_exp(xbi * (1.0f - tr));
        } else {
            fpvs = psat * gfk_pow(tr, xa) * gfk_exp(xb * (1.0f - tr));
        }
    }
    if (t >= 330.0f) {
        tr = ttp / 330.0f;
        if (t < ttp) {
            fpvs = psat * gfk_pow(tr, xai) * gfk_exp(xbi * (1.0f - tr));
        } else {
            fpvs = psat * gfk_pow(tr, xa) * gfk_exp(xb * (1.0f - tr));
        }
    }
    return fpvs;
}

struct NsasPhys {
    float cp, cliq, cvap, g, hvap, rd, rv, fv, ep2, cice, xls, psat;
};

__device__ __forceinline__ NsasPhys nsas_phys()
{
    NsasPhys c;
    c.cp = CP;
    c.cliq = NSAS_CLIQ;
    c.cvap = 4.0f * RV;
    c.g = G;
    c.hvap = XLV;
    c.rd = RD;
    c.rv = RV;
    c.fv = RV / RD - 1.0f;
    c.ep2 = RD / RV;
    c.cice = NSAS_CICE;
    c.xls = NSAS_XLS;
    c.psat = NSAS_PSAT;
    return c;
}

#define FPVS(tt) nsas_fpvs((tt), c.rd, c.rv, c.cvap, c.cliq, c.cice, xlv0, \
                           c.xls, c.psat, t0c_)

// ===========================================================================
// nsas2d: deep convection, module_cu_nsas.F:313-2080, for one column.
// ===========================================================================
__device__ void nsas2d(const NsasCol& w, const NsasPhys& c, float delt,
                       float delx, int ncloud, float slimsk, float pgcon,
                       int dx_factor_nsas, float* rain, int* kbot, int* ktop,
                       int* icps, const int* row_limits = nullptr)
{
    const int kts = 1, kte = w.nz;
#define A(s, k) w.a((s), (k))
    const float* pcrit = NSAS_PCRIT_TABLE;
    const float* acritt = NSAS_ACRITT_TABLE;
    float acrit[15];

    const float qmin_ = NSAS_QMIN;
    const float t0c_ = NSAS_T0C;
    const float xlv0 = c.hvap;
    const float rcs = 1.0f;
    const float el2orc = c.hvap * c.hvap / (c.rv * c.cp);
    const float eps = c.rd / c.rv;
    const float fact1 = (c.cvap - c.cliq) / c.rv;
    const float fact2 = c.hvap / c.rv - fact1 * t0c_;
    const int kts1 = kts + 1;
    const int kte1 = kte - 1;
    const float dt2 = delt;
    const float dtmin = fmaxf(dt2, 1200.0f);
    const float dtmax = fmaxf(dt2, 3600.0f);
    const float tcrf = 1.0f / (NSAS_TCR - NSAS_TF);

    float w1l, w2l, w3l, w4l, w1s, w2s, w3s, w4s;
    if (dx_factor_nsas == 1) {
        const float dx_factor = 250.0f / delx;
        w1l = dx_factor * 0.1f * (-1.0f);
        w2l = dx_factor * (-1.0f);
        w3l = dx_factor * (-1.0f);
        w4l = dx_factor * 0.1f * (-1.0f);
        w1s = w1l;
        w2s = w2l;
        w3s = w3l;
        w4s = w4l;
    } else {
        w1l = -8.E-3f;
        w2l = -4.E-2f;
        w3l = -5.E-3f;
        w4l = -5.E-4f;
        w1s = -2.E-4f;
        w2s = -2.E-3f;
        w3s = -1.E-3f;
        w4s = -2.E-5f;
    }

    // :590-610
    *rain = 0.0f;
    *kbot = kte + 1;
    *ktop = 0;
    *icps = 0;
    bool cnvflg = true;
    float dtconv = 3600.0f;
    float pdot = 0.0f;
    float edto = 0.0f;
    float edtx = 0.0f;
    float xmbmax = 0.3f;
    float aa2 = 0.0f;
    float qlko_ktcon = 0.0f;
    float pbcdif = 0.0f;
    int lmin = 1;
    int jmin = 1;
    float edt = 0.0f;
    (void)xmbmax; (void)pbcdif;

    for (int k = 0; k < 15; ++k) {
        acrit[k] = acritt[k] * (975.0f - pcrit[k]);
    }

    // :619-631. Stage-only calls use a one-column row.
    int kbmax = kte;
    int kbm = kte;
    int kmax = kte;
    for (int k = kts; k <= kte; ++k) {
        if (A(S_PRSL, k) > A(S_PRSI, 1) * 0.45f) kbmax = k + 1;
        if (A(S_PRSL, k) > A(S_PRSI, 1) * 0.70f) kbm = k + 1;
        if (A(S_PRSL, k) > A(S_PRSI, 1) * 0.04f) kmax = k + 1;
    }
    if (row_limits != nullptr) {
        kbmax = row_limits[0]; kbm = row_limits[1]; kmax = row_limits[2];
    }
    kmax = min(kmax, kte);
    const int kmax1 = kmax - 1;
    kbm = min(kbm, kte);

    // :635-645
    for (int k = kts; k <= kte; ++k) {
        A(S_PWO, k) = 0.0f;
        A(S_PWDO, k) = 0.0f;
        A(S_DELLAL, k) = 0.0f;
        A(S_HCKO, k) = 0.0f;
        A(S_QCKO, k) = 0.0f;
        A(S_HCDO, k) = 0.0f;
        A(S_QCDO, k) = 0.0f;
    }
    // The +1 level is never written by WRF; give it a defined value.
    A(S_HEO, kte + 1) = 0.0f;
    A(S_HESO, kte + 1) = 0.0f;
    A(S_ZI, 1) = 0.0f;
    A(S_ZI, kte + 1) = 0.0f;

    // :647-665
    for (int k = kts; k <= kmax; ++k) {
        A(S_P, k) = A(S_PRSL, k) * 10.0f;
        A(S_PWO, k) = 0.0f;
        A(S_PWDO, k) = 0.0f;
        A(S_TO, k) = A(S_T1, k);
        A(S_QO, k) = A(S_Q1, k);
        A(S_DBYO, k) = 0.0f;
        A(S_FENT1, k) = 1.0f;
        A(S_FENT2, k) = 1.0f;
        A(S_FRH, k) = 0.0f;
        A(S_UCKO, k) = 0.0f;
        A(S_VCKO, k) = 0.0f;
        A(S_UCDO, k) = 0.0f;
        A(S_VCDO, k) = 0.0f;
        A(S_UO, k) = A(S_U1, k) * rcs;
        A(S_VO, k) = A(S_V1, k) * rcs;
    }

    // :675-683
    for (int k = kts; k <= kmax; ++k) {
        float q = 0.01f * FPVS(A(S_TO, k));
        q = eps * q / (A(S_P, k) + (eps - 1.0f) * q);
        A(S_QESO, k) = fmaxf(q, qmin_);
        A(S_QO, k) = fmaxf(A(S_QO, k), 1.e-10f);
        A(S_TVO, k) = A(S_TO, k) + c.fv * A(S_TO, k) * fmaxf(A(S_QO, k), qmin_);
    }

    // :687-692
    for (int k = kts; k <= kmax; ++k) {
        A(S_HEO, k) = c.g * A(S_ZL, k) + c.cp * A(S_TO, k) + c.hvap * A(S_QO, k);
        A(S_HESO, k) = c.g * A(S_ZL, k) + c.cp * A(S_TO, k) + c.hvap * A(S_QESO, k);
    }

    // :697-709
    float hmax = A(S_HEO, 1);
    int kb = 1;
    for (int k = kts1; k <= kbm; ++k) {
        if (A(S_HEO, k) > hmax) {
            kb = k;
            hmax = A(S_HEO, k);
        }
    }

    // :711-741
    for (int k = kts; k <= kmax1; ++k) {
        if (cnvflg) {
            const float dz = .5f * (A(S_ZL, k + 1) - A(S_ZL, k));
            const float dp = .5f * (A(S_P, k + 1) - A(S_P, k));
            const float tk1 = A(S_TO, k + 1);
            const float es = 0.01f * FPVS(tk1);
            const float pprime = A(S_P, k + 1) + (eps - 1.0f) * es;
            const float qs = eps * es / pprime;
            const float dqsdp = -(qs / pprime);
            const float desdt = es * (fact1 / tk1 + fact2 / (tk1 * tk1));
            const float dqsdt = qs * A(S_P, k + 1) * desdt / (es * pprime);
            const float gamma = el2orc * A(S_QESO, k + 1) / (tk1 * tk1);
            const float dt = (c.g * dz + c.hvap * dqsdp * dp) / (c.cp * (1.0f + gamma));
            const float dq = dqsdt * dt + dqsdp * dp;
            A(S_TO, k) = A(S_TO, k + 1) + dt;
            A(S_QO, k) = A(S_QO, k + 1) + dq;
            const float po = .5f * (A(S_P, k) + A(S_P, k + 1));
            float q = 0.01f * FPVS(A(S_TO, k));
            q = eps * q / (po + (eps - 1.0f) * q);
            A(S_QESO, k) = fmaxf(q, qmin_);
            A(S_QO, k) = fmaxf(A(S_QO, k), 1.e-10f);
            A(S_FRH, k) = 1.0f - fminf(A(S_QO, k) / A(S_QESO, k), 1.0f);
            A(S_HEO, k) = .5f * c.g * (A(S_ZL, k) + A(S_ZL, k + 1))
                          + c.cp * A(S_TO, k) + c.hvap * A(S_QO, k);
            A(S_HESO, k) = .5f * c.g * (A(S_ZL, k) + A(S_ZL, k + 1))
                           + c.cp * A(S_TO, k) + c.hvap * A(S_QESO, k);
            A(S_UO, k) = .5f * (A(S_UO, k) + A(S_UO, k + 1));
            A(S_VO, k) = .5f * (A(S_VO, k) + A(S_VO, k + 1));
        }
    }

    // :745-751
    const float hkbo = A(S_HEO, kb);
    const float qkbo = A(S_QO, kb);

    // :753-771
    bool flg = cnvflg;
    int kbcon = kmax;
    for (int k = kts; k <= kbmax; ++k) {
        if (flg && k > kb) {
            const float hsbar = A(S_HESO, k);
            if (hkbo > hsbar) {
                flg = false;
                kbcon = k;
            }
        }
    }
    if (kbcon == kmax) cnvflg = false;
    if (!cnvflg) return;

    // :779-812
    {
        pdot = 10.0f * A(S_DOT, kbcon);
        float w3, w4;
        if (slimsk == 1.0f) {
            w3 = w3l;
            w4 = w4l;
        } else {
            w3 = w3s;
            w4 = w4s;
        }
        float tem;
        if (pdot <= w4) {
            tem = (pdot - w4) / (w3 - w4);
        } else if (pdot >= -w4) {
            tem = -((pdot + w4) / (w4 - w3));
        } else {
            tem = 0.0f;
        }
        tem = fmaxf(tem, -1.0f);
        tem = fminf(tem, 1.0f);
        tem = 1.0f - tem;
        const float tem1 = .5f * (NSAS_CINCRMAX - NSAS_CINCRMIN);
        const float cincr = NSAS_CINCRMAX - tem * tem1;
        pbcdif = -A(S_P, kbcon) + A(S_P, kb);
        if (pbcdif > cincr) cnvflg = false;
    }
    if (!cnvflg) return;

    // :821-831
    for (int k = kts1; k <= kte; ++k) {
        A(S_ZI, k) = 0.5f * (A(S_ZL, k - 1) + A(S_ZL, k));
    }
    for (int k = kts; k <= kte1; ++k) {
        A(S_XLAMB, k) = NSAS_CLAM / A(S_ZI, k + 1);
    }
    // :836-842
    for (int k = kts1; k <= kmax1; ++k) {
        if (k > kbcon) A(S_XLAMB, k) = A(S_XLAMB, kbcon);
    }
    // :847-851
    const float xlamud = A(S_XLAMB, kbcon);
    // :856-864
    for (int k = kts1; k <= kmax1; ++k) {
        if (k > kbcon) {
            const float tem = A(S_QESO, k) / A(S_QESO, kbcon);
            A(S_FENT1, k) = tem * tem;
            A(S_FENT2, k) = tem * tem * tem;
        }
    }
    // :870-877
    for (int k = kts1; k <= kmax1; ++k) {
        if (k >= kbcon) {
            const float tem = NSAS_CXLAMU * A(S_FRH, k) * A(S_FENT2, k);
            A(S_XLAMB, k) = A(S_XLAMB, k) * A(S_FENT1, k) + tem;
        }
    }

    // :881-913
    for (int k = kts; k <= kte; ++k) A(S_ETA, k) = 1.0f;
    for (int k = kbmax; k >= kts1; --k) {
        if (k < kbcon && k >= kb) {
            const float dz = A(S_ZI, k + 2) - A(S_ZI, k + 1);
            const float ptem = 0.5f * (A(S_XLAMB, k) + A(S_XLAMB, k + 1)) - xlamud;
            A(S_ETA, k) = A(S_ETA, k + 1) / (1.0f + ptem * dz);
        }
    }
    for (int k = kts1; k <= kmax1; ++k) {
        if (k > kbcon) {
            const float dz = A(S_ZI, k + 1) - A(S_ZI, k);
            const float ptem = 0.5f * (A(S_XLAMB, k) + A(S_XLAMB, k - 1)) - xlamud;
            A(S_ETA, k) = A(S_ETA, k - 1) * (1.0f + ptem * dz);
        }
    }
    {
        const float dz = A(S_ZI, 3) - A(S_ZI, 2);
        const float ptem = 0.5f * (A(S_XLAMB, 1) + A(S_XLAMB, 2)) - xlamud;
        A(S_ETA, 1) = A(S_ETA, 2) / (1.0f + ptem * dz);
    }

    // :917-926
    A(S_HCKO, kb) = hkbo;
    A(S_QCKO, kb) = qkbo;
    A(S_UCKO, kb) = A(S_UO, kb);
    A(S_VCKO, kb) = A(S_VO, kb);
    float pwavo = 0.0f;

    // :930-948
    for (int k = kts1; k <= kmax1; ++k) {
        if (k > kb) {
            const float dz = A(S_ZI, k + 1) - A(S_ZI, k);
            const float tem = 0.5f * (A(S_XLAMB, k) + A(S_XLAMB, k - 1)) * dz;
            const float tem1 = 0.5f * xlamud * dz;
            const float factor = 1.0f + tem - tem1;
            const float ptem = 0.5f * tem + pgcon;
            const float ptem1 = 0.5f * tem - pgcon;
            A(S_HCKO, k) = ((1.0f - tem1) * A(S_HCKO, k - 1) + tem * 0.5f
                            * (A(S_HEO, k) + A(S_HEO, k - 1))) / factor;
            A(S_UCKO, k) = ((1.0f - tem1) * A(S_UCKO, k - 1) + ptem * A(S_UO, k)
                            + ptem1 * A(S_UO, k - 1)) / factor;
            A(S_VCKO, k) = ((1.0f - tem1) * A(S_VCKO, k - 1) + ptem * A(S_VO, k)
                            + ptem1 * A(S_VO, k - 1)) / factor;
            A(S_DBYO, k) = A(S_HCKO, k) - A(S_HESO, k);
        }
    }

    // :953-986
    flg = cnvflg;
    int kbcon1 = kmax;
    for (int k = kts1; k <= kmax; ++k) {
        if (flg && k >= kbcon && A(S_DBYO, k) > 0.0f) {
            kbcon1 = k;
            flg = false;
        }
    }
    if (kbcon1 == kmax) cnvflg = false;
    if (cnvflg) {
        const float tem = A(S_P, kbcon) - A(S_P, kbcon1);
        if (tem > NSAS_DTHK) cnvflg = false;
    }
    if (!cnvflg) return;

    // :992-1006
    flg = cnvflg;
    int ktcon = 1;
    for (int k = kts1; k <= kmax1; ++k) {
        if (A(S_DBYO, k) < 0.0f && flg && k > kbcon1) {
            ktcon = k;
            flg = false;
        }
    }
    // :1011-1020
    if ((A(S_P, kbcon) - A(S_P, ktcon)) < 150.0f) cnvflg = false;
    if (!cnvflg) return;

    // :1025-1050
    float hmin = A(S_HEO, kbcon1);
    lmin = kbmax;
    jmin = kbmax;
    for (int k = kts1; k <= kbmax; ++k) {
        if (k > kbcon1 && A(S_HEO, k) < hmin) {
            lmin = k + 1;
            hmin = A(S_HEO, k);
        }
    }
    jmin = min(lmin, ktcon - 1);
    jmin = max(jmin, kbcon1 + 1);
    if (jmin >= ktcon) cnvflg = false;

    // :1054-1060
    if (cnvflg) {
        const float dp = 1000.0f * A(S_DEL, kbcon);
        xmbmax = dp / (c.g * dt2);
    }

    // :1065-1103
    float aa1 = 0.0f;
    for (int k = kts1; k <= kmax; ++k) {
        if (cnvflg && k > kb && k < ktcon) {
            const float dz1 = A(S_ZI, k + 1) - A(S_ZI, k);
            const float tk = A(S_TO, k);
            const float gamma = el2orc * A(S_QESO, k) / (tk * tk);
            const float qrch = A(S_QESO, k)
                + gamma * A(S_DBYO, k) / (c.hvap * (1.0f + gamma));
            const float tem = 0.5f * (A(S_XLAMB, k) + A(S_XLAMB, k - 1)) * dz1;
            const float tem1 = 0.5f * xlamud * dz1;
            const float factor = 1.0f + tem - tem1;
            A(S_QCKO, k) = ((1.0f - tem1) * A(S_QCKO, k - 1) + tem * 0.5f
                            * (A(S_QO, k) + A(S_QO, k - 1))) / factor;
            A(S_QCIRS, k) = A(S_ETA, k) * A(S_QCKO, k) - A(S_ETA, k) * qrch;
            if (A(S_QCIRS, k) > 0.0f && k >= kbcon) {
                const float etah = .5f * (A(S_ETA, k) + A(S_ETA, k - 1));
                float qlk;
                if (ncloud > 0 && k > jmin) {
                    const float dp = 1000.0f * A(S_DEL, k);
                    qlk = A(S_QCIRS, k) / (A(S_ETA, k) + etah * (NSAS_C0 + NSAS_C1) * dz1);
                    A(S_DELLAL, k) = etah * NSAS_C1 * dz1 * qlk * c.g / dp;
                } else {
                    qlk = A(S_QCIRS, k) / (A(S_ETA, k) + etah * NSAS_C0 * dz1);
                }
                aa1 = aa1 - dz1 * c.g * qlk;
                const float qc = qlk + qrch;
                A(S_PWO, k) = etah * NSAS_C0 * dz1 * qlk;
                A(S_QCKO, k) = qc;
                pwavo = pwavo + A(S_PWO, k);
            }
        }
    }

    // :1107-1119
    for (int k = kts1; k <= kmax; ++k) {
        if (cnvflg && k >= kbcon && k < ktcon) {
            const float dz1 = A(S_ZL, k + 1) - A(S_ZL, k);
            const float tk = A(S_TO, k);
            const float gamma = el2orc * A(S_QESO, k) / (tk * tk);
            const float rfact = 1.0f + c.fv * c.cp * gamma * tk / c.hvap;
            aa1 = aa1 + dz1 * (c.g / (c.cp * tk)) * A(S_DBYO, k) / (1.0f + gamma) * rfact;
            aa1 = aa1 + dz1 * c.g * c.fv * fmaxf(0.0f, (A(S_QESO, k) - A(S_QO, k)));
        }
    }
    if (cnvflg && aa1 <= 0.0f) cnvflg = false;
    if (!cnvflg) return;

    // :1135-1162
    aa2 = NSAS_AAFAC * aa1;
    flg = cnvflg;
    int ktcon1 = kmax1;
    for (int k = kts1; k <= kmax; ++k) {
        if (flg) {
            if (k >= ktcon && k < kmax) {
                const float dz1 = A(S_ZL, k + 1) - A(S_ZL, k);
                const float tk = A(S_TO, k);
                const float gamma = el2orc * A(S_QESO, k) / (tk * tk);
                const float rfact = 1.0f + c.fv * c.cp * gamma * tk / c.hvap;
                aa2 = aa2 + dz1 * (c.g / (c.cp * tk)) * A(S_DBYO, k) / (1.0f + gamma) * rfact;
                if (aa2 < 0.0f) {
                    ktcon1 = k;
                    flg = false;
                }
            }
        }
    }

    // :1167-1200
    for (int k = kts1; k <= kmax; ++k) {
        if (k >= ktcon && k < ktcon1) {
            const float dz = (A(S_ZI, k + 1) - A(S_ZI, k));
            const float tk = A(S_TO, k);
            const float gamma = el2orc * A(S_QESO, k) / (tk * tk);
            const float qrch = A(S_QESO, k) + gamma * A(S_DBYO, k) / (c.hvap * (1.0f + gamma));
            const float tem = 0.5f * (A(S_XLAMB, k) + A(S_XLAMB, k - 1)) * dz;
            const float tem1 = 0.5f * xlamud * dz;
            const float factor = 1.0f + tem - tem1;
            A(S_QCKO, k) = ((1.0f - tem1) * A(S_QCKO, k - 1) + tem * 0.5f
                            * (A(S_QO, k) + A(S_QO, k - 1))) / factor;
            A(S_QCIRS, k) = A(S_ETA, k) * A(S_QCKO, k) - A(S_ETA, k) * qrch;
            if (A(S_QCIRS, k) > 0.0f) {
                const float etah = .5f * (A(S_ETA, k) + A(S_ETA, k - 1));
                float qlk;
                if (ncloud > 0) {
                    const float dp = 1000.0f * A(S_DEL, k);
                    qlk = A(S_QCIRS, k) / (A(S_ETA, k) + etah * (NSAS_C0 + NSAS_C1) * dz);
                    A(S_DELLAL, k) = etah * NSAS_C1 * dz * qlk * c.g / dp;
                } else {
                    qlk = A(S_QCIRS, k) / (A(S_ETA, k) + etah * NSAS_C0 * dz);
                }
                const float qc = qlk + qrch;
                A(S_PWO, k) = etah * NSAS_C0 * dz * qlk;
                A(S_QCKO, k) = qc;
                pwavo = pwavo + A(S_PWO, k);
            }
        }
    }

    // :1204-1210 exchange ktcon with ktcon1
    {
        const int kk = ktcon;
        ktcon = ktcon1;
        ktcon1 = kk;
    }

    // :1214-1234
    if (ncloud > 0) {
        const int k = ktcon - 1;
        const float tk = A(S_TO, k);
        const float gamma = el2orc * A(S_QESO, k) / (tk * tk);
        const float qrch = A(S_QESO, k) + gamma * A(S_DBYO, k) / (c.hvap * (1.0f + gamma));
        const float dq = A(S_QCKO, k) - qrch;
        if (dq > 0.0f) {
            qlko_ktcon = dq;
            A(S_QCKO, k) = qrch;
        }
    }

    // :1240-1268
    float vshear = 0.0f;
    for (int k = kts1; k <= kmax; ++k) {
        if (k > kb && k <= ktcon) {
            const float du = A(S_UO, k) - A(S_UO, k - 1);
            const float dv = A(S_VO, k) - A(S_VO, k - 1);
            const float shear = sqrtf(du * du + dv * dv);
            vshear = vshear + shear;
        }
    }
    {
        vshear = 1.e3f * vshear / (A(S_ZI, ktcon + 1) - A(S_ZI, kb + 1));
        const float e1 = 1.591f - .639f * vshear
            + .0953f * (vshear * vshear) - .00496f * (vshear * vshear * vshear);
        edt = 1.0f - e1;
        edt = fminf(edt, .9f);
        edt = fmaxf(edt, .0f);
        edto = edt;
        edtx = edt;
    }

    // :1272-1298
    float sumx = 0.0f;
    for (int k = kts; k <= kmax1; ++k) {
        if (k >= 1 && k < kbcon) {
            const float dz = A(S_ZI, k + 2) - A(S_ZI, k + 1);
            sumx = sumx + dz;
        }
    }
    float xlamd;
    {
        float beta = NSAS_BETAS;
        if (slimsk == 1.0f) beta = NSAS_BETAL;
        const float dz = (sumx + A(S_ZI, 2)) / (float)kbcon;
        const float tem = 1.0f / (float)kbcon;
        xlamd = (1.0f - gfk_pow(beta, tem)) / dz;
    }

    // :1302-1326
    for (int k = kts; k <= kmax; ++k) {
        A(S_ETAD, k) = 1.0f;
        A(S_QRCDO, k) = 0.0f;
        A(S_QRCD, k) = 0.0f;
    }
    for (int k = kmax1; k >= kts; --k) {
        if (k < jmin && k >= kbcon) {
            const float dz = (A(S_ZI, k + 2) - A(S_ZI, k + 1));
            const float ptem = NSAS_XLAMDD - NSAS_XLAMDE;
            A(S_ETAD, k) = A(S_ETAD, k + 1) * (1.0f - ptem * dz);
        } else if (k < kbcon) {
            const float dz = (A(S_ZI, k + 2) - A(S_ZI, k + 1));
            const float ptem = xlamd + NSAS_XLAMDD - NSAS_XLAMDE;
            A(S_ETAD, k) = A(S_ETAD, k + 1) * (1.0f - ptem * dz);
        }
    }

    // :1331-1346
    float pwevo = 0.0f;
    {
        const int jmn = jmin;
        A(S_HCDO, jmn) = A(S_HEO, jmn);
        A(S_QCDO, jmn) = A(S_QO, jmn);
        A(S_QRCDO, jmn) = A(S_QESO, jmn);
        A(S_UCDO, jmn) = A(S_UO, jmn);
        A(S_VCDO, jmn) = A(S_VO, jmn);
    }

    // :1348-1371
    for (int k = kmax1; k >= kts; --k) {
        if (k < jmin) {
            const float dz = A(S_ZI, k + 2) - A(S_ZI, k + 1);
            float tem, tem1;
            if (k >= kbcon) {
                tem = NSAS_XLAMDE * dz;
                tem1 = 0.5f * NSAS_XLAMDD * dz;
            } else {
                tem = NSAS_XLAMDE * dz;
                tem1 = 0.5f * (xlamd + NSAS_XLAMDD) * dz;
            }
            const float factor = 1.0f + tem - tem1;
            const float ptem = 0.5f * tem - pgcon;
            const float ptem1 = 0.5f * tem + pgcon;
            A(S_HCDO, k) = ((1.0f - tem1) * A(S_HCDO, k + 1) + tem * 0.5f
                            * (A(S_HEO, k) + A(S_HEO, k + 1))) / factor;
            A(S_UCDO, k) = ((1.0f - tem1) * A(S_UCDO, k + 1) + ptem * A(S_UO, k + 1)
                            + ptem1 * A(S_UO, k)) / factor;
            A(S_VCDO, k) = ((1.0f - tem1) * A(S_VCDO, k + 1) + ptem * A(S_VO, k + 1)
                            + ptem1 * A(S_VO, k)) / factor;
            A(S_DBYO, k) = A(S_HCDO, k) - A(S_HESO, k);
        }
    }

    // :1373-1397
    for (int k = kmax1; k >= kts; --k) {
        if (k < jmin) {
            const float dq = A(S_QESO, k);
            const float dt = A(S_TO, k);
            const float gamma = el2orc * dq / (dt * dt);
            A(S_QRCDO, k) = dq + (1.0f / c.hvap) * (gamma / (1.0f + gamma)) * A(S_DBYO, k);
            const float dz = A(S_ZI, k + 2) - A(S_ZI, k + 1);
            float tem, tem1;
            if (k >= kbcon) {
                tem = NSAS_XLAMDE * dz;
                tem1 = 0.5f * NSAS_XLAMDD * dz;
            } else {
                tem = NSAS_XLAMDE * dz;
                tem1 = 0.5f * (xlamd + NSAS_XLAMDD) * dz;
            }
            const float factor = 1.0f + tem - tem1;
            A(S_QCDO, k) = ((1.0f - tem1) * A(S_QCDO, k + 1) + tem * 0.5f
                            * (A(S_QO, k) + A(S_QO, k + 1))) / factor;
            A(S_PWDO, k) = A(S_ETAD, k + 1) * A(S_QCDO, k) - A(S_ETAD, k + 1) * A(S_QRCDO, k);
            A(S_QCDO, k) = A(S_QRCDO, k);
            pwevo = pwevo + A(S_PWDO, k);
        }
    }

    // :1403-1414
    {
        float edtmax = NSAS_EDTMAXL;
        if (slimsk == 2.0f) edtmax = NSAS_EDTMAXS;
        if (pwevo < 0.0f) {
            edto = -edto * pwavo / pwevo;
            edto = fminf(edto, edtmax);
        } else {
            edto = 0.0f;
        }
    }

    // :1418-1432
    for (int k = kmax1; k >= kts; --k) {
        if (k < jmin) {
            const float tk = A(S_TO, k);
            const float gamma = el2orc * A(S_QESO, k) / (tk * tk);
            const float dhh = A(S_HCDO, k);
            const float dt = tk;
            const float dg = gamma;
            const float dh = A(S_HESO, k);
            const float dz = -1.0f * (A(S_ZL, k + 1) - A(S_ZL, k));
            aa1 = aa1 + edto * dz * (c.g / (c.cp * dt)) * ((dhh - dh) / (1.0f + dg))
                  * (1.0f + c.fv * c.cp * dg * dt / c.hvap);
            aa1 = aa1 + edto * dz * c.g * c.fv * fmaxf(0.0f, (A(S_QESO, k) - A(S_QO, k)));
        }
    }
    if (aa1 <= 0.0f) cnvflg = false;
    if (!cnvflg) return;

    // :1447-1456
    for (int k = kts; k <= kmax; ++k) {
        A(S_DELLAH, k) = 0.0f;
        A(S_DELLAQ, k) = 0.0f;
        A(S_DELLAU, k) = 0.0f;
        A(S_DELLAV, k) = 0.0f;
    }

    // :1458-1470
    {
        const float dp = 1000.0f * A(S_DEL, 1);
        A(S_DELLAH, 1) = edto * A(S_ETAD, 1) * (A(S_HCDO, 1) - A(S_HEO, 1)) * c.g / dp;
        A(S_DELLAQ, 1) = edto * A(S_ETAD, 1) * (A(S_QCDO, 1) - A(S_QO, 1)) * c.g / dp;
        A(S_DELLAU, 1) = edto * A(S_ETAD, 1) * (A(S_UCDO, 1) - A(S_UO, 1)) * c.g / dp;
        A(S_DELLAV, 1) = edto * A(S_ETAD, 1) * (A(S_VCDO, 1) - A(S_VO, 1)) * c.g / dp;
    }

    // :1474-1535
    for (int k = kts1; k <= kmax1; ++k) {
        if (k < ktcon) {
            float aup = 1.0f;
            if (k <= kb) aup = 0.0f;
            float adw = 1.0f;
            if (k > jmin) adw = 0.0f;
            const float dv1 = A(S_HEO, k);
            const float dv2 = .5f * (A(S_HEO, k) + A(S_HEO, k - 1));
            const float dv3 = A(S_HEO, k - 1);
            const float dv1q = A(S_QO, k);
            const float dv2q = .5f * (A(S_QO, k) + A(S_QO, k - 1));
            const float dv3q = A(S_QO, k - 1);
            const float dv1u = A(S_UO, k);
            const float dv2u = .5f * (A(S_UO, k) + A(S_UO, k - 1));
            const float dv3u = A(S_UO, k - 1);
            const float dv1v = A(S_VO, k);
            const float dv2v = .5f * (A(S_VO, k) + A(S_VO, k - 1));
            const float dv3v = A(S_VO, k - 1);
            const float dp = 1000.0f * A(S_DEL, k);
            const float dz = A(S_ZI, k + 1) - A(S_ZI, k);
            const float tem = 0.5f * (A(S_XLAMB, k) + A(S_XLAMB, k - 1));
            const float tem1 = xlamud;
            float ptem, ptem1;
            if (k <= kbcon) {
                ptem = NSAS_XLAMDE;
                ptem1 = xlamd + NSAS_XLAMDD;
            } else {
                ptem = NSAS_XLAMDE;
                ptem1 = NSAS_XLAMDD;
            }
            const float eta = A(S_ETA, k), etam = A(S_ETA, k - 1);
            const float etad = A(S_ETAD, k), etadm = A(S_ETAD, k - 1);
            (void)etadm;
            const float m1 = aup * eta - adw * edto * etad;
            const float m3 = aup * etam - adw * edto * A(S_ETAD, k - 1);
            const float m2 = aup * tem * etam + adw * edto * ptem * etad;
            A(S_DELLAH, k) = A(S_DELLAH, k) +
                ((m1 * dv1
                  - m3 * dv3
                  - m2 * dv2 * dz
                  + aup * tem1 * etam * .5f * (A(S_HCKO, k) + A(S_HCKO, k - 1)) * dz
                  + adw * edto * ptem1 * etad * .5f * (A(S_HCDO, k) + A(S_HCDO, k - 1)) * dz)
                 * c.g / dp);
            A(S_DELLAQ, k) = A(S_DELLAQ, k) +
                ((m1 * dv1q
                  - m3 * dv3q
                  - m2 * dv2q * dz
                  + aup * tem1 * etam * .5f * (A(S_QCKO, k) + A(S_QCKO, k - 1)) * dz
                  + adw * edto * ptem1 * etad * .5f * (A(S_QRCDO, k) + A(S_QRCDO, k - 1)) * dz)
                 * c.g / dp);
            A(S_DELLAU, k) = A(S_DELLAU, k) +
                ((m1 * dv1u
                  - m3 * dv3u
                  - m2 * dv2u * dz
                  + aup * tem1 * etam * .5f * (A(S_UCKO, k) + A(S_UCKO, k - 1)) * dz
                  + adw * edto * ptem1 * etad * .5f * (A(S_UCDO, k) + A(S_UCDO, k - 1)) * dz
                  - pgcon * (aup * etam - adw * edto * etad) * (dv1u - dv3u))
                 * c.g / dp);
            A(S_DELLAV, k) = A(S_DELLAV, k) +
                ((m1 * dv1v
                  - m3 * dv3v
                  - m2 * dv2v * dz
                  + aup * tem1 * etam * .5f * (A(S_VCKO, k) + A(S_VCKO, k - 1)) * dz
                  + adw * edto * ptem1 * etad * .5f * (A(S_VCDO, k) + A(S_VCDO, k - 1)) * dz
                  - pgcon * (aup * etam - adw * edto * etad) * (dv1v - dv3v))
                 * c.g / dp);
        }
    }

    // :1539-1560 cloud top
    {
        const int indx = ktcon;
        const float dp = 1000.0f * A(S_DEL, indx);
        const float dv1 = A(S_HEO, indx - 1);
        A(S_DELLAH, indx) = A(S_ETA, indx - 1) * (A(S_HCKO, indx - 1) - dv1) * c.g / dp;
        const float dvq1 = A(S_QO, indx - 1);
        A(S_DELLAQ, indx) = A(S_ETA, indx - 1) * (A(S_QCKO, indx - 1) - dvq1) * c.g / dp;
        const float dv1u = A(S_UO, indx - 1);
        A(S_DELLAU, indx) = A(S_ETA, indx - 1) * (A(S_UCKO, indx - 1) - dv1u) * c.g / dp;
        const float dv1v = A(S_VO, indx - 1);
        A(S_DELLAV, indx) = A(S_ETA, indx - 1) * (A(S_VCKO, indx - 1) - dv1v) * c.g / dp;
        A(S_DELLAL, indx) = A(S_ETA, indx - 1) * qlko_ktcon * c.g / dp;
    }

    // :1564-1577
    for (int k = kts; k <= kmax; ++k) {
        if (k > ktcon) {
            A(S_QO, k) = A(S_Q1, k);
            A(S_TO, k) = A(S_T1, k);
        }
        if (k <= ktcon) {
            A(S_QO, k) = A(S_DELLAQ, k) * NSAS_MBDT + A(S_Q1, k);
            const float dellat = (A(S_DELLAH, k) - c.hvap * A(S_DELLAQ, k)) / c.cp;
            A(S_TO, k) = dellat * NSAS_MBDT + A(S_T1, k);
            A(S_QO, k) = fmaxf(A(S_QO, k), 1.0e-10f);
        }
    }

    // :1591-1600
    for (int k = kts; k <= kmax; ++k) {
        float q = 0.01f * FPVS(A(S_TO, k));
        q = eps * q / (A(S_P, k) + (eps - 1.0f) * q);
        A(S_QESO, k) = fmaxf(q, qmin_);
        A(S_TVO, k) = A(S_TO, k) + c.fv * A(S_TO, k) * fmaxf(A(S_QO, k), qmin_);
    }

    // :1611-1646
    for (int k = kts; k <= kmax1; ++k) {
        const float dz = .5f * (A(S_ZL, k + 1) - A(S_ZL, k));
        const float dp = .5f * (A(S_P, k + 1) - A(S_P, k));
        const float tk1 = A(S_TO, k + 1);
        const float es = 0.01f * FPVS(tk1);
        const float pprime = A(S_P, k + 1) + (eps - 1.0f) * es;
        const float qs = eps * es / pprime;
        const float dqsdp = -(qs / pprime);
        const float desdt = es * (fact1 / tk1 + fact2 / (tk1 * tk1));
        const float dqsdt = qs * A(S_P, k + 1) * desdt / (es * pprime);
        const float gamma = el2orc * A(S_QESO, k + 1) / (tk1 * tk1);
        const float dt = (c.g * dz + c.hvap * dqsdp * dp) / (c.cp * (1.0f + gamma));
        const float dq = dqsdt * dt + dqsdp * dp;
        A(S_TO, k) = A(S_TO, k + 1) + dt;
        A(S_QO, k) = A(S_QO, k + 1) + dq;
        const float po = .5f * (A(S_P, k) + A(S_P, k + 1));
        float q = 0.01f * FPVS(A(S_TO, k));
        q = eps * q / (po + (eps - 1.0f) * q);
        A(S_QESO, k) = fmaxf(q, qmin_);
        A(S_QO, k) = fmaxf(A(S_QO, k), 1.0e-10f);
        A(S_HEO, k) = .5f * c.g * (A(S_ZL, k) + A(S_ZL, k + 1))
                      + c.cp * A(S_TO, k) + c.hvap * A(S_QO, k);
        A(S_HESO, k) = .5f * c.g * (A(S_ZL, k) + A(S_ZL, k + 1))
                       + c.cp * A(S_TO, k) + c.hvap * A(S_QESO, k);
    }
    {
        const int k = kmax;
        A(S_HEO, k) = c.g * A(S_ZL, k) + c.cp * A(S_TO, k) + c.hvap * A(S_QO, k);
        A(S_HESO, k) = c.g * A(S_ZL, k) + c.cp * A(S_TO, k) + c.hvap * A(S_QESO, k);
    }

    // :1648-1658
    float xaa0 = 0.0f;
    float xpwav = 0.0f;
    {
        const int indx = kb;
        const float xhkb = A(S_HEO, indx);
        const float xqkb = A(S_QO, indx);
        A(S_HCKO, indx) = xhkb;
        A(S_QCKO, indx) = xqkb;
    }

    // :1664-1675
    for (int k = kts1; k <= kmax1; ++k) {
        if (k > kb && k <= ktcon) {
            const float dz = A(S_ZI, k + 1) - A(S_ZI, k);
            const float tem = 0.5f * (A(S_XLAMB, k) + A(S_XLAMB, k - 1)) * dz;
            const float tem1 = 0.5f * xlamud * dz;
            const float factor = 1.0f + tem - tem1;
            A(S_HCKO, k) = ((1.0f - tem1) * A(S_HCKO, k - 1) + tem * 0.5f
                            * (A(S_HEO, k) + A(S_HEO, k - 1))) / factor;
        }
    }

    // :1677-1720
    for (int k = kts1; k <= kmax1; ++k) {
        if (k > kb && k < ktcon) {
            const float dz = A(S_ZI, k + 1) - A(S_ZI, k);
            const float tk = A(S_TO, k);
            const float gamma = el2orc * A(S_QESO, k) / (tk * tk);
            const float xdby = A(S_HCKO, k) - A(S_HESO, k);
            const float xqrch = A(S_QESO, k) + gamma * xdby / (c.hvap * (1.0f + gamma));
            const float tem = 0.5f * (A(S_XLAMB, k) + A(S_XLAMB, k - 1)) * dz;
            const float tem1 = 0.5f * xlamud * dz;
            const float factor = 1.0f + tem - tem1;
            A(S_QCKO, k) = ((1.0f - tem1) * A(S_QCKO, k - 1) + tem * 0.5f
                            * (A(S_QO, k) + A(S_QO, k - 1))) / factor;
            const float dq = A(S_ETA, k) * A(S_QCKO, k) - A(S_ETA, k) * xqrch;
            if (k >= kbcon && dq > 0.0f) {
                const float etah = .5f * (A(S_ETA, k) + A(S_ETA, k - 1));
                float qlk;
                if (ncloud > 0 && k > jmin) {
                    qlk = dq / (A(S_ETA, k) + etah * (NSAS_C0 + NSAS_C1) * dz);
                } else {
                    qlk = dq / (A(S_ETA, k) + etah * NSAS_C0 * dz);
                }
                if (k < ktcon1) {
                    xaa0 = xaa0 - dz * c.g * qlk;
                }
                A(S_QCKO, k) = qlk + xqrch;
                const float xpw = etah * NSAS_C0 * dz * qlk;
                xpwav = xpwav + xpw;
            }
        }
        if (k >= kbcon && k < ktcon1) {
            const float dz1 = A(S_ZL, k + 1) - A(S_ZL, k);
            const float tk = A(S_TO, k);
            const float gamma = el2orc * A(S_QESO, k) / (tk * tk);
            const float rfact = 1.0f + c.fv * c.cp * gamma * tk / c.hvap;
            const float xdby = A(S_HCKO, k) - A(S_HESO, k);
            xaa0 = xaa0 + dz1 * (c.g / (c.cp * tk)) * xdby / (1.0f + gamma) * rfact;
            xaa0 = xaa0 + dz1 * c.g * c.fv * fmaxf(0.0f, (A(S_QESO, k) - A(S_QO, k)));
        }
    }

    // :1727-1738
    float xpwev = 0.0f;
    {
        const int jmn = jmin;
        A(S_XHCD, jmn) = A(S_HEO, jmn);
        A(S_XQCD, jmn) = A(S_QO, jmn);
        A(S_QRCD, jmn) = A(S_QESO, jmn);
    }
    // :1740-1756
    for (int k = kmax1; k >= kts; --k) {
        if (k < jmin) {
            const float dz = A(S_ZI, k + 2) - A(S_ZI, k + 1);
            float tem, tem1;
            if (k >= kbcon) {
                tem = NSAS_XLAMDE * dz;
                tem1 = 0.5f * NSAS_XLAMDD * dz;
            } else {
                tem = NSAS_XLAMDE * dz;
                tem1 = 0.5f * (xlamd + NSAS_XLAMDD) * dz;
            }
            const float factor = 1.0f + tem - tem1;
            A(S_XHCD, k) = ((1.0f - tem1) * A(S_XHCD, k + 1) + tem * 0.5f
                            * (A(S_HEO, k) + A(S_HEO, k + 1))) / factor;
        }
    }
    // :1758-1782
    for (int k = kmax1; k >= kts; --k) {
        if (k < jmin) {
            const float dq = A(S_QESO, k);
            const float dt = A(S_TO, k);
            const float gamma = el2orc * dq / (dt * dt);
            const float dh = A(S_XHCD, k) - A(S_HESO, k);
            A(S_QRCD, k) = dq + (1.0f / c.hvap) * (gamma / (1.0f + gamma)) * dh;
            const float dz = A(S_ZI, k + 2) - A(S_ZI, k + 1);
            float tem, tem1;
            if (k >= kbcon) {
                tem = NSAS_XLAMDE * dz;
                tem1 = 0.5f * NSAS_XLAMDD * dz;
            } else {
                tem = NSAS_XLAMDE * dz;
                tem1 = 0.5f * (xlamd + NSAS_XLAMDD) * dz;
            }
            const float factor = 1.0f + tem - tem1;
            A(S_XQCD, k) = ((1.0f - tem1) * A(S_XQCD, k + 1) + tem * 0.5f
                            * (A(S_QO, k) + A(S_QO, k + 1))) / factor;
            const float xpwd = A(S_ETAD, k + 1) * (A(S_XQCD, k) - A(S_QRCD, k));
            A(S_XQCD, k) = A(S_QRCD, k);
            xpwev = xpwev + xpwd;
        }
    }

    // :1784-1795
    {
        float edtmax = NSAS_EDTMAXL;
        if (slimsk == 2.0f) edtmax = NSAS_EDTMAXS;
        if (xpwev >= 0.0f) {
            edtx = 0.0f;
        } else {
            edtx = -edtx * xpwav / xpwev;
            edtx = fminf(edtx, edtmax);
        }
    }

    // :1799-1814
    for (int k = kmax1; k >= kts; --k) {
        if (k < jmin) {
            const float tk = A(S_TO, k);
            const float gamma = el2orc * A(S_QESO, k) / (tk * tk);
            const float dhh = A(S_XHCD, k);
            const float dt = tk;
            const float dg = gamma;
            const float dh = A(S_HESO, k);
            const float dz = -1.0f * (A(S_ZL, k + 1) - A(S_ZL, k));
            xaa0 = xaa0 + edtx * dz * (c.g / (c.cp * dt)) * ((dhh - dh) / (1.0f + dg))
                   * (1.0f + c.fv * c.cp * dg * dt / c.hvap);
            xaa0 = xaa0 + edtx * dz * c.g * c.fv * fmaxf(0.0f, (A(S_QESO, k) - A(S_QO, k)));
        }
    }

    // :1818-1833 critical cloud work function
    float acrt;
    {
        const float pk = A(S_P, ktcon);
        if (pk < pcrit[14]) {
            acrt = acrit[14] * (975.0f - pk) / (975.0f - pcrit[14]);
        } else if (pk > pcrit[0]) {
            acrt = acrit[0];
        } else {
            int k = (int)((850.0f - pk) / 50.0f) + 2;
            k = min(k, 15);
            k = max(k, 2);
            acrt = acrit[k - 1] + (acrit[k - 2] - acrit[k - 1])
                   * (pk - pcrit[k - 1]) / (pcrit[k - 2] - pcrit[k - 1]);
        }
    }

    // :1835-1863
    float acrtfct = 1.0f;
    {
        float w1 = w1s, w2 = w2s, w3 = w3s, w4 = w4s;
        if (slimsk == 1.0f) {
            w1 = w1l;
            w2 = w2l;
            w3 = w3l;
            w4 = w4l;
        }
        if (pdot <= w4) {
            acrtfct = (pdot - w4) / (w3 - w4);
        } else if (pdot >= -w4) {
            acrtfct = -((pdot + w4) / (w4 - w3));
        } else {
            acrtfct = 0.0f;
        }
        acrtfct = fmaxf(acrtfct, -1.0f);
        acrtfct = fminf(acrtfct, 1.0f);
        acrtfct = 1.0f - acrtfct;
        dtconv = dt2 + fmaxf((1800.0f - dt2), 0.0f) * (pdot - w2) / (w1 - w2);
        dtconv = fmaxf(dtconv, dtmin);
        dtconv = fminf(dtconv, dtmax);
    }

    // :1867-1887 closure
    float xmb = 0.0f;
    {
        const float f = (aa1 - acrt * acrtfct) / dtconv;
        if (f <= 0.0f) cnvflg = false;
        float xk = 0.0f;
        if (cnvflg) {
            xk = (xaa0 - aa1) / NSAS_MBDT;
            if (xk >= 0.0f) cnvflg = false;
        }
        if (cnvflg) {
            xmb = -f / xk;
            xmb = fminf(xmb, xmbmax);
        }
    }
    if (!cnvflg) return;

    // :1896-1908
    for (int k = kts; k <= kmax; ++k) {
        A(S_TO, k) = A(S_T1, k);
        A(S_QO, k) = A(S_Q1, k);
        A(S_UO, k) = A(S_U1, k);
        A(S_VO, k) = A(S_V1, k);
        float q = 0.01f * FPVS(A(S_T1, k));
        q = eps * q / (A(S_P, k) + (eps - 1.0f) * q);
        A(S_QESO, k) = fmaxf(q, qmin_);
    }

    // :1914-1945 feedback
    float delhbar = 0.0f, delqbar = 0.0f, deltbar = 0.0f;
    float delubar = 0.0f, delvbar = 0.0f;
    float qcond = 0.0f;
    for (int k = kts; k <= kmax; ++k) {
        if (k <= ktcon) {
            const float dellat = (A(S_DELLAH, k) - c.hvap * A(S_DELLAQ, k)) / c.cp;
            A(S_T1, k) = A(S_T1, k) + dellat * xmb * dt2;
            A(S_Q1, k) = A(S_Q1, k) + A(S_DELLAQ, k) * xmb * dt2;
            const float tem = 1.0f / rcs;
            A(S_U1, k) = A(S_U1, k) + A(S_DELLAU, k) * xmb * dt2 * tem;
            A(S_V1, k) = A(S_V1, k) + A(S_DELLAV, k) * xmb * dt2 * tem;
            const float dp = 1000.0f * A(S_DEL, k);
            delhbar = delhbar + A(S_DELLAH, k) * xmb * dp / c.g;
            delqbar = delqbar + A(S_DELLAQ, k) * xmb * dp / c.g;
            deltbar = deltbar + dellat * xmb * dp / c.g;
            delubar = delubar + A(S_DELLAU, k) * xmb * dp / c.g;
            delvbar = delvbar + A(S_DELLAV, k) * xmb * dp / c.g;
        }
    }
    (void)delhbar; (void)delubar; (void)delvbar;

    // :1952-1960
    for (int k = kts; k <= kmax; ++k) {
        if (k <= ktcon) {
            float q = 0.01f * FPVS(A(S_T1, k));
            q = eps * q / (A(S_P, k) + (eps - 1.0f) * q);
            A(S_QESO, k) = fmaxf(q, qmin_);
        }
    }

    // :1962-1983
    float rntot = 0.0f, delqev = 0.0f, delq2 = 0.0f;
    flg = cnvflg;
    for (int k = kmax; k >= kts; --k) {
        if (k < ktcon) {
            float aup = 1.0f;
            if (k <= kb) aup = 0.0f;
            float adw = 1.0f;
            if (k >= jmin) adw = 0.0f;
            rntot = rntot + (aup * A(S_PWO, k) + adw * edto * A(S_PWDO, k))
                    * xmb * .001f * dt2;
        }
    }

    // :1987-2030 rainfall and its evaporation
    for (int k = kmax; k >= kts; --k) {
        float delq = 0.0f;
        float deltv = 0.0f;
        float qevap = 0.0f;
        if (k < ktcon) {
            float aup = 1.0f;
            if (k <= kb) aup = 0.0f;
            float adw = 1.0f;
            if (k >= jmin) adw = 0.0f;
            *rain = *rain + (aup * A(S_PWO, k) + adw * edto * A(S_PWDO, k))
                    * xmb * .001f * dt2;
        }
        if (flg && k < ktcon) {
            float evef = edt * NSAS_EVFACTS;
            if (slimsk == 1.0f) evef = edt * NSAS_EVFACTL;
            const float t1k = A(S_T1, k);
            qcond = evef * (A(S_Q1, k) - A(S_QESO, k))
                    / (1.0f + el2orc * A(S_QESO, k) / (t1k * t1k));
            const float dp = 1000.0f * A(S_DEL, k);
            if (*rain > 0.0f && qcond < 0.0f) {
                qevap = -qcond * (1.0f - gfk_exp(-(.32f * sqrtf(dt2 * *rain))));
                qevap = fminf(qevap, *rain * 1000.0f * c.g / dp);
                delq2 = delqev + .001f * qevap * dp / c.g;
                if (delq2 > rntot) {
                    qevap = 1000.0f * c.g * (rntot - delqev) / dp;
                    flg = false;
                }
            }
            if (*rain > 0.0f && qevap > 0.0f) {
                A(S_Q1, k) = A(S_Q1, k) + qevap;
                A(S_T1, k) = A(S_T1, k) - (c.hvap / c.cp) * qevap;
                *rain = *rain - .001f * qevap * dp / c.g;
                delqev = delqev + .001f * dp * qevap / c.g;
                deltv = -((c.hvap / c.cp) * qevap / dt2);
                delq = +qevap / dt2;
            }
            A(S_DELLAQ, k) = A(S_DELLAQ, k) + delq / xmb;
            delqbar = delqbar + delq * dp / c.g;
            deltbar = deltbar + deltv * dp / c.g;
        }
    }

    // :2035-2046
    if (*rain < 0.0f && !flg) *rain = 0.0f;
    if (*rain <= 0.0f) {
        *rain = 0.0f;
    } else {
        *ktop = ktcon;
        *kbot = kbcon;
        *icps = 1;
    }

    // :2048-2057
    if (*rain <= 0.0f) {
        for (int k = kts; k <= kmax; ++k) {
            A(S_T1, k) = A(S_TO, k);
            A(S_Q1, k) = A(S_QO, k);
            A(S_U1, k) = A(S_UO, k);
            A(S_V1, k) = A(S_VO, k);
        }
    }

    // :2061-2078 detrainment of cloud water and ice
    if (ncloud > 0 && *rain > 0.0f) {
        for (int k = kts; k <= kmax; ++k) {
            if (k >= kbcon && k <= ktcon) {
                const float tem = A(S_DELLAL, k) * xmb * dt2;
                const float tem1 = fmaxf(0.0f, fminf(1.0f, (NSAS_TCR - A(S_T1, k)) * tcrf));
                if (ncloud >= 2) {
                    A(S_QI2, k) = A(S_QI2, k) + tem * tem1;
                    A(S_QC2, k) = A(S_QC2, k) + tem * (1.0f - tem1);
                } else {
                    A(S_QC2, k) = A(S_QC2, k) + tem;
                }
            }
        }
    }
#undef A
}

// ===========================================================================
// nscv2d: shallow convection, module_cu_nsas.F:2192-3287, for one column.
// ===========================================================================
__device__ void nscv2d(const NsasCol& w, const NsasPhys& c, float delt,
                       int ncloud, float slimsk, float hpbl, float hfx,
                       float qfx, float pgcon, float* rain, int* kbot,
                       int* ktop, int* icps)
{
    const int kts = 1, kte = w.nz;
#define A(s, k) w.a((s), (k))
    const float t0c_ = NSAS_T0C;
    const float xlv0 = c.hvap;
    const int km1 = kte - 1;
    const float tcrf = 1.0f / (NSAS_TCR - NSAS_TF);

    // :2346-2360 surface buoyancy flux
    for (int k = kts; k <= kte; ++k) {
        A(S_THX, k) = A(S_T1, k) / A(S_PRSLK, k);
    }
    const float tvcon = (1.0f + c.fv * A(S_Q1, 1));
    const float rhox = A(S_PRSL, 1) * 1.e3f / (c.rd * A(S_T1, 1) * tvcon);
    const float sflx = hfx / rhox / c.cp + qfx / rhox * c.fv * A(S_THX, 1);

    // :2364-2387
    bool cnvflg = true;
    if (*icps == 1) cnvflg = false;
    if (sflx <= 0.0f) cnvflg = false;
    if (cnvflg) {
        *kbot = kte + 1;
        *ktop = 0;
    }
    *rain = 0.0f;
    int kbcon = kte;
    int ktcon = 1;
    int kb = kte;
    float pdot = 0.0f;
    float qlko_ktcon = 0.0f;
    float edt = 0.0f;
    float aa1 = 0.0f;
    float vshear = 0.0f;
    if (!cnvflg) return;

    // :2389-2418
    const float dt2 = delt;
    const float el2orc = c.hvap * c.hvap / (c.rv * c.cp);
    const float eps = c.rd / c.rv;
    const float fact1 = (c.cvap - c.cliq) / c.rv;
    const float fact2 = c.hvap / c.rv - fact1 * t0c_;
    const float w1l = -8.e-3f, w2l = -4.e-2f, w3l = -5.e-3f, w4l = -5.e-4f;
    const float w1s = -2.e-4f, w2s = -2.e-3f, w3s = -1.e-3f, w4s = -2.e-5f;
    (void)w1l; (void)w2l; (void)w1s; (void)w2s;

    // :2423-2437
    int kbm = kte;
    int kmax = kte;
    for (int k = kts; k <= kte; ++k) {
        if (A(S_PRSL, k) > A(S_PRSI, 1) * 0.70f) kbm = k + 1;
        if (A(S_PRSL, k) > A(S_PRSI, 1) * 0.60f) kmax = k + 1;
    }
    kbm = min(kbm, kmax);

    // :2442-2456
    A(S_ZI, 1) = 0.0f;
    A(S_ZI, kte + 1) = 0.0f;
    for (int k = kts + 1; k <= kte; ++k) {
        A(S_ZI, k) = 0.5f * (A(S_ZL, k - 1) + A(S_ZL, k));
    }
    for (int k = kts; k <= km1; ++k) {
        A(S_XLAMUE, k) = NSCV_CLAM / A(S_ZI, k + 1);
    }
    A(S_XLAMUE, kte) = A(S_XLAMUE, km1);

    // :2460-2477 pbl height
    bool flg = cnvflg;
    int kpbl = 1;
    for (int k = kts + 1; k <= km1; ++k) {
        if (flg && A(S_ZL, k) <= hpbl) {
            kpbl = k;
        } else {
            flg = false;
        }
    }
    kpbl = min(kpbl, kbm);

    // :2481-2500
    const float rcs = 1.0f;
    for (int k = kts; k <= kte; ++k) {
        if (k <= kmax) {
            A(S_P, k) = A(S_PRSL, k) * 10.0f;
            A(S_ETA, k) = 1.0f;
            A(S_HCKO, k) = 0.0f;
            A(S_QCKO, k) = 0.0f;
            A(S_UCKO, k) = 0.0f;
            A(S_VCKO, k) = 0.0f;
            A(S_DBYO, k) = 0.0f;
            A(S_PWO, k) = 0.0f;
            A(S_DELLAL, k) = 0.0f;
            A(S_TO, k) = A(S_T1, k);
            A(S_QO, k) = A(S_Q1, k);
            A(S_UO, k) = A(S_U1, k) * rcs;
            A(S_VO, k) = A(S_V1, k) * rcs;
        }
    }

    // :2510-2521
    for (int k = kts; k <= kte; ++k) {
        if (k <= kmax) {
            float q = 0.01f * FPVS(A(S_TO, k));
            q = eps * q / (A(S_P, k) + (eps - 1.0f) * q);
            A(S_QESO, k) = fmaxf(q, 1.e-8f);
            A(S_QO, k) = fmaxf(A(S_QO, k), 1.e-10f);
        }
    }

    // :2525-2533
    for (int k = kts; k <= kte; ++k) {
        if (k <= kmax) {
            const float tem = c.g * A(S_ZL, k) + c.cp * A(S_TO, k);
            A(S_HEO, k) = tem + c.hvap * A(S_QO, k);
            A(S_HESO, k) = tem + c.hvap * A(S_QESO, k);
        }
    }

    // :2538-2554
    float hmax = A(S_HEO, 1);
    kb = 1;
    for (int k = kts + 1; k <= kte; ++k) {
        if (k <= kpbl) {
            if (A(S_HEO, k) > hmax) {
                kb = k;
                hmax = A(S_HEO, k);
            }
        }
    }

    // :2556-2575
    for (int k = kts; k <= km1; ++k) {
        if (k <= kmax - 1) {
            const float dz = .5f * (A(S_ZL, k + 1) - A(S_ZL, k));
            const float dp = .5f * (A(S_P, k + 1) - A(S_P, k));
            const float tk1 = A(S_TO, k + 1);
            const float es = 0.01f * FPVS(tk1);
            const float pprime = A(S_P, k + 1) + (eps - 1.0f) * es;
            const float qs = eps * es / pprime;
            const float dqsdp = -(qs / pprime);
            const float desdt = es * (fact1 / tk1 + fact2 / (tk1 * tk1));
            const float dqsdt = qs * A(S_P, k + 1) * desdt / (es * pprime);
            const float gamma = el2orc * A(S_QESO, k + 1) / (tk1 * tk1);
            const float dt = (c.g * dz + c.hvap * dqsdp * dp) / (c.cp * (1.0f + gamma));
            const float dq = dqsdt * dt + dqsdp * dp;
            A(S_TO, k) = A(S_TO, k + 1) + dt;
            A(S_QO, k) = A(S_QO, k + 1) + dq;
            A(S_PO, k) = .5f * (A(S_P, k) + A(S_P, k + 1));
        }
    }

    // :2577-2594
    for (int k = kts; k <= km1; ++k) {
        if (k <= kmax - 1) {
            float q = 0.01f * FPVS(A(S_TO, k));
            q = eps * q / (A(S_PO, k) + (eps - 1.0f) * q);
            A(S_QESO, k) = fmaxf(q, 1.e-8f);
            A(S_QO, k) = fmaxf(A(S_QO, k), 1.e-10f);
            A(S_HEO, k) = .5f * c.g * (A(S_ZL, k) + A(S_ZL, k + 1))
                          + c.cp * A(S_TO, k) + c.hvap * A(S_QO, k);
            A(S_HESO, k) = .5f * c.g * (A(S_ZL, k) + A(S_ZL, k + 1))
                           + c.cp * A(S_TO, k) + c.hvap * A(S_QESO, k);
            A(S_UO, k) = .5f * (A(S_UO, k) + A(S_UO, k + 1));
            A(S_VO, k) = .5f * (A(S_VO, k) + A(S_VO, k + 1));
        }
    }

    // :2598-2624
    flg = cnvflg;
    kbcon = kmax;
    for (int k = kts + 1; k <= km1; ++k) {
        if (flg && k < kbm) {
            if (k > kb && A(S_HEO, kb) > A(S_HESO, k)) {
                kbcon = k;
                flg = false;
            }
        }
    }
    if (kbcon == kmax) cnvflg = false;
    if (!cnvflg) return;

    // :2629-2673
    pdot = 10.0f * A(S_DOT, kbcon);
    {
        float w3, w4;
        if (slimsk == 1.0f) {
            w3 = w3l;
            w4 = w4l;
        } else {
            w3 = w3s;
            w4 = w4s;
        }
        float ptem;
        if (pdot <= w4) {
            ptem = (pdot - w4) / (w3 - w4);
        } else if (pdot >= -w4) {
            ptem = -((pdot + w4) / (w4 - w3));
        } else {
            ptem = 0.0f;
        }
        ptem = fmaxf(ptem, -1.0f);
        ptem = fminf(ptem, 1.0f);
        ptem = 1.0f - ptem;
        const float ptem1 = .5f * (NSAS_CINCRMAX - NSAS_CINCRMIN);
        const float cincr = NSAS_CINCRMAX - ptem * ptem1;
        const float tem1 = A(S_P, kb) - A(S_P, kbcon);
        if (tem1 > cincr) cnvflg = false;
    }
    if (!cnvflg) return;

    // :2678-2682
    const float xlamud = A(S_XLAMUE, kbcon);

    // :2686-2696
    for (int k = km1; k >= kts; --k) {
        if (k < kbcon && k >= kb) {
            const float dz = A(S_ZI, k + 2) - A(S_ZI, k + 1);
            const float ptem = 0.5f * (A(S_XLAMUE, k) + A(S_XLAMUE, k + 1)) - xlamud;
            A(S_ETA, k) = A(S_ETA, k + 1) / (1.0f + ptem * dz);
        }
    }
    // :2700-2710
    for (int k = kts + 1; k <= km1; ++k) {
        if (k > kbcon && k < kmax) {
            const float dz = A(S_ZI, k + 1) - A(S_ZI, k);
            const float ptem = 0.5f * (A(S_XLAMUE, k) + A(S_XLAMUE, k - 1)) - xlamud;
            A(S_ETA, k) = A(S_ETA, k - 1) * (1.0f + ptem * dz);
        }
    }

    // :2714-2721
    A(S_HCKO, kb) = A(S_HEO, kb);
    A(S_UCKO, kb) = A(S_UO, kb);
    A(S_VCKO, kb) = A(S_VO, kb);

    // :2723-2743
    for (int k = kts + 1; k <= km1; ++k) {
        if (k > kb && k < kmax) {
            const float dz = A(S_ZI, k + 1) - A(S_ZI, k);
            const float tem = 0.5f * (A(S_XLAMUE, k) + A(S_XLAMUE, k - 1)) * dz;
            const float tem1 = 0.5f * xlamud * dz;
            const float factor = 1.0f + tem - tem1;
            const float ptem = 0.5f * tem + pgcon;
            const float ptem1 = 0.5f * tem - pgcon;
            A(S_HCKO, k) = ((1.0f - tem1) * A(S_HCKO, k - 1) + tem * 0.5f
                            * (A(S_HEO, k) + A(S_HEO, k - 1))) / factor;
            A(S_UCKO, k) = ((1.0f - tem1) * A(S_UCKO, k - 1) + ptem * A(S_UO, k)
                            + ptem1 * A(S_UO, k - 1)) / factor;
            A(S_VCKO, k) = ((1.0f - tem1) * A(S_VCKO, k - 1) + ptem * A(S_VO, k)
                            + ptem1 * A(S_VO, k - 1)) / factor;
            A(S_DBYO, k) = A(S_HCKO, k) - A(S_HESO, k);
        }
    }

    // :2748-2783
    flg = cnvflg;
    int kbcon1 = kmax;
    for (int k = kts + 1; k <= km1; ++k) {
        if (flg && k < kbm) {
            if (k >= kbcon && A(S_DBYO, k) > 0.0f) {
                kbcon1 = k;
                flg = false;
            }
        }
    }
    if (kbcon1 == kmax) cnvflg = false;
    if (cnvflg) {
        const float tem = A(S_P, kbcon) - A(S_P, kbcon1);
        if (tem > NSAS_DTHK) cnvflg = false;
    }
    if (!cnvflg) return;

    // :2788-2802
    flg = cnvflg;
    ktcon = kbm;
    for (int k = kts + 1; k <= km1; ++k) {
        if (flg && k < kbm) {
            if (k > kbcon1 && A(S_DBYO, k) < 0.0f) {
                ktcon = k;
                flg = false;
            }
        }
    }

    // :2806-2812
    float xmbmax;
    {
        const int k = kbcon;
        const float dp = 1000.0f * A(S_DEL, k);
        xmbmax = dp / (c.g * dt2);
    }

    // :2816-2857
    aa1 = 0.0f;
    A(S_QCKO, kb) = A(S_QO, kb);
    for (int k = kts + 1; k <= km1; ++k) {
        if (k > kb && k < ktcon) {
            const float dz = A(S_ZI, k + 1) - A(S_ZI, k);
            const float tk = A(S_TO, k);
            const float gamma = el2orc * A(S_QESO, k) / (tk * tk);
            const float qrch = A(S_QESO, k) + gamma * A(S_DBYO, k) / (c.hvap * (1.0f + gamma));
            const float tem = 0.5f * (A(S_XLAMUE, k) + A(S_XLAMUE, k - 1)) * dz;
            const float tem1 = 0.5f * xlamud * dz;
            const float factor = 1.0f + tem - tem1;
            A(S_QCKO, k) = ((1.0f - tem1) * A(S_QCKO, k - 1) + tem * 0.5f
                            * (A(S_QO, k) + A(S_QO, k - 1))) / factor;
            const float dq = A(S_ETA, k) * (A(S_QCKO, k) - qrch);
            if (k >= kbcon && dq > 0.0f) {
                const float etah = .5f * (A(S_ETA, k) + A(S_ETA, k - 1));
                float qlk;
                if (ncloud > 0) {
                    const float dp = 1000.0f * A(S_DEL, k);
                    qlk = dq / (A(S_ETA, k) + etah * (NSCV_C0 + NSCV_C1) * dz);
                    A(S_DELLAL, k) = etah * NSCV_C1 * dz * qlk * c.g / dp;
                } else {
                    qlk = dq / (A(S_ETA, k) + etah * NSCV_C0 * dz);
                }
                aa1 = aa1 - dz * c.g * qlk;
                A(S_QCKO, k) = qlk + qrch;
                A(S_PWO, k) = etah * NSCV_C0 * dz * qlk;
            }
        }
    }

    // :2861-2885
    for (int k = kts + 1; k <= km1; ++k) {
        if (k >= kbcon && k < ktcon) {
            const float dz1 = A(S_ZL, k + 1) - A(S_ZL, k);
            const float tk = A(S_TO, k);
            const float gamma = el2orc * A(S_QESO, k) / (tk * tk);
            const float rfact = 1.0f + c.fv * c.cp * gamma * tk / c.hvap;
            aa1 = aa1 + dz1 * (c.g / (c.cp * tk)) * A(S_DBYO, k) / (1.0f + gamma) * rfact;
            aa1 = aa1 + dz1 * c.g * c.fv * fmaxf(0.0f, (A(S_QESO, k) - A(S_QO, k)));
        }
    }
    if (aa1 <= 0.0f) cnvflg = false;
    if (!cnvflg) return;

    // :2891-2920
    aa1 = NSCV_AAFAC * aa1;
    flg = cnvflg;
    int ktcon1 = kbm;
    for (int k = kts + 1; k <= km1; ++k) {
        if (flg) {
            if (k >= ktcon && k < kbm) {
                const float dz1 = A(S_ZL, k + 1) - A(S_ZL, k);
                const float tk = A(S_TO, k);
                const float gamma = el2orc * A(S_QESO, k) / (tk * tk);
                const float rfact = 1.0f + c.fv * c.cp * gamma * tk / c.hvap;
                aa1 = aa1 + dz1 * (c.g / (c.cp * tk)) * A(S_DBYO, k) / (1.0f + gamma) * rfact;
                if (aa1 < 0.0f) {
                    ktcon1 = k;
                    flg = false;
                }
            }
        }
    }

    // :2925-2957
    for (int k = kts + 1; k <= km1; ++k) {
        if (k >= ktcon && k < ktcon1) {
            const float dz = A(S_ZI, k + 1) - A(S_ZI, k);
            const float tk = A(S_TO, k);
            const float gamma = el2orc * A(S_QESO, k) / (tk * tk);
            const float qrch = A(S_QESO, k) + gamma * A(S_DBYO, k) / (c.hvap * (1.0f + gamma));
            const float tem = 0.5f * (A(S_XLAMUE, k) + A(S_XLAMUE, k - 1)) * dz;
            const float tem1 = 0.5f * xlamud * dz;
            const float factor = 1.0f + tem - tem1;
            A(S_QCKO, k) = ((1.0f - tem1) * A(S_QCKO, k - 1) + tem * 0.5f
                            * (A(S_QO, k) + A(S_QO, k - 1))) / factor;
            const float dq = A(S_ETA, k) * (A(S_QCKO, k) - qrch);
            if (dq > 0.0f) {
                const float etah = .5f * (A(S_ETA, k) + A(S_ETA, k - 1));
                float qlk;
                if (ncloud > 0) {
                    const float dp = 1000.0f * A(S_DEL, k);
                    qlk = dq / (A(S_ETA, k) + etah * (NSCV_C0 + NSCV_C1) * dz);
                    A(S_DELLAL, k) = etah * NSCV_C1 * dz * qlk * c.g / dp;
                } else {
                    qlk = dq / (A(S_ETA, k) + etah * NSCV_C0 * dz);
                }
                A(S_QCKO, k) = qlk + qrch;
                A(S_PWO, k) = etah * NSCV_C0 * dz * qlk;
            }
        }
    }

    // :2961-2967
    {
        const int kk = ktcon;
        ktcon = ktcon1;
        ktcon1 = kk;
    }

    // :2971-2992
    if (ncloud > 0) {
        const int k = ktcon - 1;
        const float tk = A(S_TO, k);
        const float gamma = el2orc * A(S_QESO, k) / (tk * tk);
        const float qrch = A(S_QESO, k) + gamma * A(S_DBYO, k) / (c.hvap * (1.0f + gamma));
        const float dq = A(S_QCKO, k) - qrch;
        if (dq > 0.0f) {
            qlko_ktcon = dq;
            A(S_QCKO, k) = qrch;
        }
    }

    // :2996-3024
    vshear = 0.0f;
    for (int k = kts + 1; k <= kte; ++k) {
        if (k > kb && k <= ktcon) {
            const float du = A(S_UO, k) - A(S_UO, k - 1);
            const float dv = A(S_VO, k) - A(S_VO, k - 1);
            const float shear = sqrtf(du * du + dv * dv);
            vshear = vshear + shear;
        }
    }
    {
        vshear = 1.e3f * vshear / (A(S_ZI, ktcon + 1) - A(S_ZI, kb + 1));
        const float e1 = 1.591f - .639f * vshear
            + .0953f * (vshear * vshear) - .00496f * (vshear * vshear * vshear);
        edt = 1.0f - e1;
        edt = fminf(edt, .9f);
        edt = fmaxf(edt, .0f);
    }

    // :3029-3038
    for (int k = kts; k <= kte; ++k) {
        if (k <= kmax) {
            A(S_DELLAH, k) = 0.0f;
            A(S_DELLAQ, k) = 0.0f;
            A(S_DELLAU, k) = 0.0f;
            A(S_DELLAV, k) = 0.0f;
        }
    }

    // :3042-3090
    for (int k = kts + 1; k <= km1; ++k) {
        if (k > kb && k < ktcon) {
            const float dp = 1000.0f * A(S_DEL, k);
            const float dz = A(S_ZI, k + 1) - A(S_ZI, k);
            const float dv1h = A(S_HEO, k);
            const float dv2h = .5f * (A(S_HEO, k) + A(S_HEO, k - 1));
            const float dv3h = A(S_HEO, k - 1);
            const float dv1q = A(S_QO, k);
            const float dv2q = .5f * (A(S_QO, k) + A(S_QO, k - 1));
            const float dv3q = A(S_QO, k - 1);
            const float dv1u = A(S_UO, k);
            const float dv2u = .5f * (A(S_UO, k) + A(S_UO, k - 1));
            const float dv3u = A(S_UO, k - 1);
            const float dv1v = A(S_VO, k);
            const float dv2v = .5f * (A(S_VO, k) + A(S_VO, k - 1));
            const float dv3v = A(S_VO, k - 1);
            const float tem = 0.5f * (A(S_XLAMUE, k) + A(S_XLAMUE, k - 1));
            const float tem1 = xlamud;
            const float eta = A(S_ETA, k), etam = A(S_ETA, k - 1);
            A(S_DELLAH, k) = A(S_DELLAH, k) +
                ((eta * dv1h - etam * dv3h
                  - tem * etam * dv2h * dz
                  + tem1 * etam * .5f * (A(S_HCKO, k) + A(S_HCKO, k - 1)) * dz) * c.g / dp);
            A(S_DELLAQ, k) = A(S_DELLAQ, k) +
                ((eta * dv1q - etam * dv3q
                  - tem * etam * dv2q * dz
                  + tem1 * etam * .5f * (A(S_QCKO, k) + A(S_QCKO, k - 1)) * dz) * c.g / dp);
            A(S_DELLAU, k) = A(S_DELLAU, k) +
                ((eta * dv1u - etam * dv3u
                  - tem * etam * dv2u * dz
                  + tem1 * etam * .5f * (A(S_UCKO, k) + A(S_UCKO, k - 1)) * dz
                  - pgcon * etam * (dv1u - dv3u)) * c.g / dp);
            A(S_DELLAV, k) = A(S_DELLAV, k) +
                ((eta * dv1v - etam * dv3v
                  - tem * etam * dv2v * dz
                  + tem1 * etam * .5f * (A(S_VCKO, k) + A(S_VCKO, k - 1)) * dz
                  - pgcon * etam * (dv1v - dv3v)) * c.g / dp);
        }
    }

    // :3094-3116 cloud top
    {
        const int indx = ktcon;
        const float dp = 1000.0f * A(S_DEL, indx);
        const float dv1h = A(S_HEO, indx - 1);
        A(S_DELLAH, indx) = A(S_ETA, indx - 1) * (A(S_HCKO, indx - 1) - dv1h) * c.g / dp;
        const float dv1q = A(S_QO, indx - 1);
        A(S_DELLAQ, indx) = A(S_ETA, indx - 1) * (A(S_QCKO, indx - 1) - dv1q) * c.g / dp;
        const float dv1u = A(S_UO, indx - 1);
        A(S_DELLAU, indx) = A(S_ETA, indx - 1) * (A(S_UCKO, indx - 1) - dv1u) * c.g / dp;
        const float dv1v = A(S_VO, indx - 1);
        A(S_DELLAV, indx) = A(S_ETA, indx - 1) * (A(S_VCKO, indx - 1) - dv1v) * c.g / dp;
        A(S_DELLAL, indx) = A(S_ETA, indx - 1) * qlko_ktcon * c.g / dp;
    }

    // :3121-3130 cloud-base mass flux (Grant 2001)
    float xmb;
    {
        const int k = kbcon;
        const float ptem = c.g * sflx * hpbl / A(S_T1, 1);
        const float wstar = gfk_pow(ptem, NSCV_H1);
        const float tem = A(S_PO, k) * 100.0f / (c.rd * A(S_T1, k));
        xmb = NSCV_BETAW * tem * wstar;
        xmb = fminf(xmb, xmbmax);
    }

    // :3132-3141
    for (int k = kts; k <= kte; ++k) {
        if (k <= kmax) {
            float q = 0.01f * FPVS(A(S_T1, k));
            q = eps * q / (A(S_P, k) + (eps - 1.0f) * q);
            A(S_QESO, k) = fmaxf(q, 1.e-8f);
        }
    }

    // :3143-3171 feedback
    float delhbar = 0.0f, delqbar = 0.0f, deltbar = 0.0f;
    float delubar = 0.0f, delvbar = 0.0f, qcond = 0.0f;
    for (int k = kts; k <= kte; ++k) {
        if (k > kb && k <= ktcon) {
            const float dellat = (A(S_DELLAH, k) - c.hvap * A(S_DELLAQ, k)) / c.cp;
            A(S_T1, k) = A(S_T1, k) + dellat * xmb * dt2;
            A(S_Q1, k) = A(S_Q1, k) + A(S_DELLAQ, k) * xmb * dt2;
            const float tem = 1.0f / rcs;
            A(S_U1, k) = A(S_U1, k) + A(S_DELLAU, k) * xmb * dt2 * tem;
            A(S_V1, k) = A(S_V1, k) + A(S_DELLAV, k) * xmb * dt2 * tem;
            const float dp = 1000.0f * A(S_DEL, k);
            delhbar = delhbar + A(S_DELLAH, k) * xmb * dp / c.g;
            delqbar = delqbar + A(S_DELLAQ, k) * xmb * dp / c.g;
            deltbar = deltbar + dellat * xmb * dp / c.g;
            delubar = delubar + A(S_DELLAU, k) * xmb * dp / c.g;
            delvbar = delvbar + A(S_DELLAV, k) * xmb * dp / c.g;
        }
    }
    (void)delhbar; (void)delubar; (void)delvbar;

    // :3173-3185
    for (int k = kts; k <= kte; ++k) {
        if (k > kb && k <= ktcon) {
            float q = 0.01f * FPVS(A(S_T1, k));
            q = eps * q / (A(S_P, k) + (eps - 1.0f) * q);
            A(S_QESO, k) = fmaxf(q, 1.e-8f);
        }
    }

    // :3187-3202
    float rntot = 0.0f, delqev = 0.0f, delq2 = 0.0f;
    flg = cnvflg;
    for (int k = kte; k >= kts; --k) {
        if (k < ktcon && k > kb) {
            rntot = rntot + A(S_PWO, k) * xmb * .001f * dt2;
        }
    }

    // :3206-3253 evaporating rain
    for (int k = kte; k >= kts; --k) {
        if (k <= kmax) {
            float deltv = 0.0f;
            float delq = 0.0f;
            float qevap = 0.0f;
            if (k < ktcon && k > kb) {
                *rain = *rain + A(S_PWO, k) * xmb * .001f * dt2;
            }
            if (flg && k < ktcon) {
                float evef = edt * NSCV_EVFACT;
                if (slimsk == 1.0f) evef = edt * NSCV_EVFACTL;
                const float t1k = A(S_T1, k);
                qcond = evef * (A(S_Q1, k) - A(S_QESO, k))
                        / (1.0f + el2orc * A(S_QESO, k) / (t1k * t1k));
                const float dp = 1000.0f * A(S_DEL, k);
                if (*rain > 0.0f && qcond < 0.0f) {
                    qevap = -qcond * (1.0f - gfk_exp(-(.32f * sqrtf(dt2 * *rain))));
                    qevap = fminf(qevap, *rain * 1000.0f * c.g / dp);
                    delq2 = delqev + .001f * qevap * dp / c.g;
                }
                if (*rain > 0.0f && qcond < 0.0f && delq2 > rntot) {
                    qevap = 1000.0f * c.g * (rntot - delqev) / dp;
                    flg = false;
                }
                if (*rain > 0.0f && qevap > 0.0f) {
                    const float tem = .001f * dp / c.g;
                    const float tem1 = qevap * tem;
                    if (tem1 > *rain) {
                        qevap = *rain / tem;
                        *rain = 0.0f;
                    } else {
                        *rain = *rain - tem1;
                    }
                    A(S_Q1, k) = A(S_Q1, k) + qevap;
                    A(S_T1, k) = A(S_T1, k) - (c.hvap / c.cp) * qevap;
                    deltv = -((c.hvap / c.cp) * qevap / dt2);
                    delq = +qevap / dt2;
                    delqev = delqev + .001f * dp * qevap / c.g;
                }
                A(S_DELLAQ, k) = A(S_DELLAQ, k) + delq / xmb;
                delqbar = delqbar + delq * dp / c.g;
                deltbar = deltbar + deltv * dp / c.g;
            }
        }
    }

    // :3255-3262
    if (*rain < 0.0f || !flg) *rain = 0.0f;
    *ktop = ktcon;
    *kbot = kbcon;
    *icps = 0;

    // :3266-3285 cloud water
    if (ncloud > 0) {
        for (int k = kts; k <= km1; ++k) {
            if (k >= kbcon && k <= ktcon) {
                const float tem = A(S_DELLAL, k) * xmb * dt2;
                const float tem1 = fmaxf(0.0f, fminf(1.0f, (NSAS_TCR - A(S_T1, k)) * tcrf));
                if (ncloud >= 2) {
                    A(S_QI2, k) = A(S_QI2, k) + tem * tem1;
                    A(S_QC2, k) = A(S_QC2, k) + tem * (1.0f - tem1);
                } else {
                    A(S_QC2, k) = A(S_QC2, k) + tem;
                }
            }
        }
    }
#undef A
}

// ---------------------------------------------------------------------------
// Per-column scalar tables.  Floats: xland, hpbl, hfx, qfx, dx, dt, pgcon.
// Ints: stepcu, ncloud (module_cu_nsas.F:141-147, mapped on the host),
// dx_factor_nsas.  The engine fills them uniformly; the oracle tests vary
// them column by column.
// ---------------------------------------------------------------------------
#define NSAS_SC_XLAND 0
#define NSAS_SC_HPBL  1
#define NSAS_SC_HFX   2
#define NSAS_SC_QFX   3
#define NSAS_SC_DX    4
#define NSAS_SC_DT    5
#define NSAS_SC_PGCON 6
#define NSAS_NSC      7
#define NSAS_IC_STEPCU   0
#define NSAS_IC_NCLOUD   1
#define NSAS_IC_DXFACTOR 2
#define NSAS_NIC         3

#define LEV(a, k) (a)[(size_t)((k) - 1) * (size_t)ncol + (size_t)i]

// cu_nsas, module_cu_nsas.F:8-308, for one column.  Inputs are level-major
// (k*ncol + i, k from 0) on mass levels; p8w and w carry nz+1 interfaces.
// Outputs: six tendencies (level-major) and sfc[j*ncol + i] for RAINCV,
// PRATEC, HBOT, HTOP.  `prep`, when non-null, receives del, prsl, zl, dot
// (nz each) and prsi (nz+1) per column, level-major, for the oracle tests.
extern "C" __global__ void nsas_row_limits(
    const float* p, const float* p8w, int* limits,
    int ncol, int nz, int row_width)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= ncol / row_width) return;
    int kbmax = nz, kbm = nz, kmax = nz;
    for (int k = 1; k <= nz; ++k) {
        for (int x = 0; x < row_width; ++x) {
            const int i = row * row_width + x;
            const float prsl = p[(size_t)(k-1)*ncol+i] * 0.001f;
            const float prsi = p8w[i] * 0.001f;
            if (prsl > prsi * 0.45f) kbmax = k + 1;
            if (prsl > prsi * 0.70f) kbm = k + 1;
            if (prsl > prsi * 0.04f) kmax = k + 1;
        }
    }
    limits[3*row] = kbmax;
    limits[3*row+1] = kbm;
    limits[3*row+2] = kmax;
}

extern "C" __global__ void nsas_column(
    const float* __restrict__ t3d, const float* __restrict__ qv3d,
    const float* __restrict__ qc3d, const float* __restrict__ qi3d,
    const float* __restrict__ p3d, const float* __restrict__ pi3d,
    const float* __restrict__ rho3d, const float* __restrict__ dz8w,
    const float* __restrict__ u3d, const float* __restrict__ v3d,
    const float* __restrict__ p3di, const float* __restrict__ w,
    const float* __restrict__ sc, const int* __restrict__ ic,
    float* __restrict__ rthcuten, float* __restrict__ rqvcuten,
    float* __restrict__ rqccuten, float* __restrict__ rqicuten,
    float* __restrict__ rucuten, float* __restrict__ rvcuten,
    float* __restrict__ sfc, float* __restrict__ prep,
    float* __restrict__ ws, int col0, int ntile, int ncol, int nz,
    const int* row_limits, int row_width)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= ntile) return;
    const int i = col0 + t;
    if (i >= ncol) return;
    NsasCol col{ws, nz, ntile, t};
    const NsasPhys c = nsas_phys();
#define A(s, k) col.a((s), (k))

    const float xland = sc[(size_t)i * NSAS_NSC + NSAS_SC_XLAND];
    const float hpbl = sc[(size_t)i * NSAS_NSC + NSAS_SC_HPBL];
    const float hfx = sc[(size_t)i * NSAS_NSC + NSAS_SC_HFX];
    const float qfx = sc[(size_t)i * NSAS_NSC + NSAS_SC_QFX];
    const float dx = sc[(size_t)i * NSAS_NSC + NSAS_SC_DX];
    const float dt = sc[(size_t)i * NSAS_NSC + NSAS_SC_DT];
    const float pgcon = sc[(size_t)i * NSAS_NSC + NSAS_SC_PGCON];
    const int stepcu = ic[(size_t)i * NSAS_NIC + NSAS_IC_STEPCU];
    const int ncloud = ic[(size_t)i * NSAS_NIC + NSAS_IC_NCLOUD];
    const int dx_factor = ic[(size_t)i * NSAS_NIC + NSAS_IC_DXFACTOR];
    const int kts = 1, kte = nz;

    // :163-164
    const float delt = dt * (float)stepcu;
    const float rdelt = 1.0f / delt;

    // :168-209
    for (int k = kts; k <= kte; ++k) {
        const int kp = k + 1;
        A(S_DOT, k) = -(5.0e-4f * c.g * LEV(rho3d, k) * (LEV(w, k) + LEV(w, kp)));
        A(S_PRSL, k) = LEV(p3d, k) * 0.001f;
        A(S_PRSI, k) = LEV(p3di, k) * 0.001f;
    }
    A(S_PRSI, kte + 1) = LEV(p3di, kte + 1) * 0.001f;
    A(S_ZII, 1) = 0.0f;
    for (int k = kts; k <= kte; ++k) {
        A(S_ZII, k + 1) = A(S_ZII, k) + LEV(dz8w, k);
    }
    for (int k = kts; k <= kte; ++k) {
        A(S_ZL, k) = 0.5f * (A(S_ZII, k) + A(S_ZII, k + 1));
    }
    for (int k = kts; k <= kte; ++k) {
        A(S_DEL, k) = A(S_PRSL, k) * c.g / c.rd * LEV(dz8w, k) / LEV(t3d, k);
        A(S_U1, k) = LEV(u3d, k);
        A(S_V1, k) = LEV(v3d, k);
        A(S_Q1, k) = LEV(qv3d, k);
        A(S_T1, k) = LEV(t3d, k);
        A(S_QI2, k) = LEV(qi3d, k);
        A(S_QC2, k) = LEV(qc3d, k);
        A(S_PRSLK, k) = LEV(pi3d, k);
    }
    if (prep != nullptr) {
        const size_t lev = (size_t)nz * (size_t)ncol;
        for (int k = kts; k <= kte; ++k) {
            const size_t o = (size_t)(k - 1) * (size_t)ncol + (size_t)i;
            prep[o] = A(S_DEL, k);
            prep[lev + o] = A(S_PRSL, k);
            prep[2 * lev + o] = A(S_ZL, k);
            prep[3 * lev + o] = A(S_DOT, k);
        }
        for (int k = kts; k <= kte + 1; ++k) {
            prep[4 * lev + (size_t)(k - 1) * (size_t)ncol + (size_t)i] = A(S_PRSI, k);
        }
    }

    float rain;
    int kbot, ktop, icps;
    // :213-234
    nsas2d(col, c, delt, dx, ncloud, xland, pgcon, dx_factor, &rain, &kbot,
           &ktop, &icps, row_limits == nullptr ? nullptr : row_limits + 3*(i/row_width));
    const float pratec1 = rain * 1000.0f / ((float)stepcu * dt);
    const float raincv1 = rain * 1000.0f / ((float)stepcu);
    // :238-258
    nscv2d(col, c, delt, ncloud, xland, hpbl, hfx, qfx, pgcon, &rain, &kbot,
           &ktop, &icps);
    const float pratec2 = rain * 1000.0f / ((float)stepcu * dt);
    const float raincv2 = rain * 1000.0f / ((float)stepcu);

    // :260-265
    sfc[0 * (size_t)ncol + i] = raincv1 + raincv2;
    sfc[1 * (size_t)ncol + i] = pratec1 + pratec2;
    sfc[2 * (size_t)ncol + i] = (float)kbot;
    sfc[3 * (size_t)ncol + i] = (float)ktop;

    // :267-303.  F_QC / F_QI are the host's: it passes a null pointer for a
    // tendency WRF would not form.
    for (int k = kts; k <= kte; ++k) {
        LEV(rthcuten, k) = (A(S_T1, k) - LEV(t3d, k)) / LEV(pi3d, k) * rdelt;
        LEV(rqvcuten, k) = (A(S_Q1, k) - LEV(qv3d, k)) * rdelt;
        LEV(rucuten, k) = (A(S_U1, k) - LEV(u3d, k)) * rdelt;
        LEV(rvcuten, k) = (A(S_V1, k) - LEV(v3d, k)) * rdelt;
        if (rqicuten != nullptr) LEV(rqicuten, k) = (A(S_QI2, k) - LEV(qi3d, k)) * rdelt;
        if (rqccuten != nullptr) LEV(rqccuten, k) = (A(S_QC2, k) - LEV(qc3d, k)) * rdelt;
    }
#undef A
}

// The two routines alone, from cu_nsas's prepared column inputs, for the
// stage oracles.  `prep` is del, prsl, zl, dot (nz each) then prsi (nz+1),
// level-major; `state` is t1, q1, u1, v1, qc2, qi2 (nz each), updated in
// place; `ki` is kbot, ktop, icps per column (in for shallow, out for both);
// `rain` per column (out).
__device__ void nsas_stage_load(const NsasCol& col, const float* prep,
                                const float* state, const float* pi3d,
                                int i, int ncol, int nz)
{
    const size_t lev = (size_t)nz * (size_t)ncol;
    for (int k = 1; k <= nz; ++k) {
        const size_t o = (size_t)(k - 1) * (size_t)ncol + (size_t)i;
        col.a(S_DEL, k) = prep[o];
        col.a(S_PRSL, k) = prep[lev + o];
        col.a(S_ZL, k) = prep[2 * lev + o];
        col.a(S_DOT, k) = prep[3 * lev + o];
        col.a(S_T1, k) = state[o];
        col.a(S_Q1, k) = state[lev + o];
        col.a(S_U1, k) = state[2 * lev + o];
        col.a(S_V1, k) = state[3 * lev + o];
        col.a(S_QC2, k) = state[4 * lev + o];
        col.a(S_QI2, k) = state[5 * lev + o];
        col.a(S_PRSLK, k) = pi3d[o];
    }
    for (int k = 1; k <= nz + 1; ++k) {
        col.a(S_PRSI, k) = prep[4 * lev + (size_t)(k - 1) * (size_t)ncol + (size_t)i];
    }
}

__device__ void nsas_stage_store(const NsasCol& col, float* state, int i,
                                 int ncol, int nz)
{
    const size_t lev = (size_t)nz * (size_t)ncol;
    for (int k = 1; k <= nz; ++k) {
        const size_t o = (size_t)(k - 1) * (size_t)ncol + (size_t)i;
        state[o] = col.a(S_T1, k);
        state[lev + o] = col.a(S_Q1, k);
        state[2 * lev + o] = col.a(S_U1, k);
        state[3 * lev + o] = col.a(S_V1, k);
        state[4 * lev + o] = col.a(S_QC2, k);
        state[5 * lev + o] = col.a(S_QI2, k);
    }
}

extern "C" __global__ void nsas_deep_stage(
    const float* __restrict__ prep, float* __restrict__ state,
    const float* __restrict__ pi3d, const float* __restrict__ sc,
    const int* __restrict__ ic, float* __restrict__ rain,
    int* __restrict__ ki, float* __restrict__ ws, int ncol, int nz)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ncol) return;
    NsasCol col{ws, nz, ncol, i};
    const NsasPhys c = nsas_phys();
    nsas_stage_load(col, prep, state, pi3d, i, ncol, nz);
    const float dt = sc[(size_t)i * NSAS_NSC + NSAS_SC_DT];
    const int stepcu = ic[(size_t)i * NSAS_NIC + NSAS_IC_STEPCU];
    float r;
    int kbot, ktop, icps;
    nsas2d(col, c, dt * (float)stepcu, sc[(size_t)i * NSAS_NSC + NSAS_SC_DX],
           ic[(size_t)i * NSAS_NIC + NSAS_IC_NCLOUD],
           sc[(size_t)i * NSAS_NSC + NSAS_SC_XLAND],
           sc[(size_t)i * NSAS_NSC + NSAS_SC_PGCON],
           ic[(size_t)i * NSAS_NIC + NSAS_IC_DXFACTOR], &r, &kbot, &ktop, &icps);
    nsas_stage_store(col, state, i, ncol, nz);
    rain[i] = r;
    ki[(size_t)i * 3 + 0] = kbot;
    ki[(size_t)i * 3 + 1] = ktop;
    ki[(size_t)i * 3 + 2] = icps;
}

extern "C" __global__ void nsas_shallow_stage(
    const float* __restrict__ prep, float* __restrict__ state,
    const float* __restrict__ pi3d, const float* __restrict__ sc,
    const int* __restrict__ ic, float* __restrict__ rain,
    int* __restrict__ ki, float* __restrict__ ws, int ncol, int nz)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ncol) return;
    NsasCol col{ws, nz, ncol, i};
    const NsasPhys c = nsas_phys();
    nsas_stage_load(col, prep, state, pi3d, i, ncol, nz);
    const float dt = sc[(size_t)i * NSAS_NSC + NSAS_SC_DT];
    const int stepcu = ic[(size_t)i * NSAS_NIC + NSAS_IC_STEPCU];
    float r;
    int kbot = ki[(size_t)i * 3 + 0];
    int ktop = ki[(size_t)i * 3 + 1];
    int icps = ki[(size_t)i * 3 + 2];
    nscv2d(col, c, dt * (float)stepcu,
           ic[(size_t)i * NSAS_NIC + NSAS_IC_NCLOUD],
           sc[(size_t)i * NSAS_NSC + NSAS_SC_XLAND],
           sc[(size_t)i * NSAS_NSC + NSAS_SC_HPBL],
           sc[(size_t)i * NSAS_NSC + NSAS_SC_HFX],
           sc[(size_t)i * NSAS_NSC + NSAS_SC_QFX],
           sc[(size_t)i * NSAS_NSC + NSAS_SC_PGCON], &r, &kbot, &ktop, &icps);
    nsas_stage_store(col, state, i, ncol, nz);
    rain[i] = r;
    ki[(size_t)i * 3 + 0] = kbot;
    ki[(size_t)i * 3 + 1] = ktop;
    ki[(size_t)i * 3 + 2] = icps;
}
