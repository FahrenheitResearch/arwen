// gpuwm/core/kernels/morrison.cu
//
// WRF v4.6.1 Morrison two-moment microphysics, with independent level work
// staged around one CUDA sedimentation thread per (j,i) column.  Transcription
// authority:
//   phys/module_mp_morr_two_moment.F
//
// The kernel keeps WRF's column ordering: fixed 250 cm-3 cloud droplets and
// bounded gamma/exponential PSDs (lines 1270-1638, 2101-2267), warm/cold
// conversion with donor conservation (1667-3316), Morrison's own liquid
// saturation adjustment (2048-2070, 3249-3267), internally substepped
// sedimentation of all mass/number moments (3356-3677), and final number
// limits (3683-4055).  The Flatau/Goff-Gratch saturation helper is from
// lines 4066-4149.  gpuwm.verify.npref.np_morrison_column is the float64
// mirror.  Mixing ratios are kg/kg dry air; moments kg-1; bottom fluxes are
// kg m-2, numerically millimetres of precipitation.

#define MORR_KMAX_GENERIC 256
#define MORR_KMAX_SHALLOW 64

#define MPI 3.14159265358979323846f
#define MRHOW 997.0f
#define MRHOI 500.0f
#define MRHOS 100.0f
#define MRHOSU (85000.0f / (287.15f * 273.15f))
#define MQSMALL 1.0e-14f
#define MDCS 125.0e-6f
#define MCI (MRHOI * MPI / 6.0f)
#define MCS (MRHOS * MPI / 6.0f)
#define MMI0 (4.0f / 3.0f * MPI * MRHOI * 1.0e-15f)
#define MMG0 1.6e-10f

struct MorrMoments {
    real lc, lr, li, ls, lg, pg;
};

__device__ __forceinline__ real morr_polysvp(real t, bool ice)
{
    // WRF POLYSVP declares T, DT, every coefficient, and the return value as
    // default REAL.  Under WRF's RWORDSIZE=4 build this is an FP32 Horner
    // chain, not a binary64 polynomial rounded once.  Spell every multiply
    // and add separately because NVRTC otherwise contracts the chain while
    // the -O0 oracle does not.
    real x = __fsub_rn(t, 273.15f);
    if (ice) {
        real p = 0.252751365e-14f;
        p = __fadd_rn(0.146898966e-11f, __fmul_rn(x, p));
        p = __fadd_rn(0.385852041e-9f, __fmul_rn(x, p));
        p = __fadd_rn(0.602588177e-7f, __fmul_rn(x, p));
        p = __fadd_rn(0.615021634e-5f, __fmul_rn(x, p));
        p = __fadd_rn(0.420895665e-3f, __fmul_rn(x, p));
        p = __fadd_rn(0.188439774e-1f, __fmul_rn(x, p));
        p = __fadd_rn(0.503160820f, __fmul_rn(x, p));
        p = __fadd_rn(6.11147274f, __fmul_rn(x, p));
        if (t >= 195.8f) return __fmul_rn(p, 100.0f);
        real ratio = __fdiv_rn(273.16f, t);
        real exponent = __fmul_rn(
            -9.09718f, __fsub_rn(ratio, 1.0f));
        exponent = __fsub_rn(
            exponent, __fmul_rn(3.56654f, log10f(ratio)));
        exponent = __fadd_rn(
            exponent,
            __fmul_rn(
                0.876793f,
                __fsub_rn(1.0f, __fdiv_rn(t, 273.16f))));
        exponent = __fadd_rn(exponent, log10f(6.1071f));
        return __fmul_rn(powf(10.0f, exponent), 100.0f);
    }
    real p = -0.976195544e-15f;
    p = __fadd_rn(-0.952447341e-13f, __fmul_rn(x, p));
    p = __fadd_rn(0.640689451e-10f, __fmul_rn(x, p));
    p = __fadd_rn(0.206739458e-7f, __fmul_rn(x, p));
    p = __fadd_rn(0.302950461e-5f, __fmul_rn(x, p));
    p = __fadd_rn(0.264847430e-3f, __fmul_rn(x, p));
    p = __fadd_rn(0.142986287e-1f, __fmul_rn(x, p));
    p = __fadd_rn(0.443987641f, __fmul_rn(x, p));
    p = __fadd_rn(6.11239921f, __fmul_rn(x, p));
    if (t >= 202.0f) return __fmul_rn(p, 100.0f);
    real ratio = __fdiv_rn(373.16f, t);
    real exponent = __fmul_rn(
        -7.90298f, __fsub_rn(ratio, 1.0f));
    exponent = __fadd_rn(
        exponent, __fmul_rn(5.02808f, log10f(ratio)));
    real inner = __fmul_rn(
        11.344f, __fsub_rn(1.0f, __fdiv_rn(t, 373.16f)));
    exponent = __fsub_rn(
        exponent,
        __fmul_rn(
            1.3816e-7f, __fsub_rn(powf(10.0f, inner), 1.0f)));
    inner = __fmul_rn(-3.49149f, __fsub_rn(ratio, 1.0f));
    exponent = __fadd_rn(
        exponent,
        __fmul_rn(
            8.1328e-3f, __fsub_rn(powf(10.0f, inner), 1.0f)));
    exponent = __fadd_rn(exponent, log10f(1013.246f));
    return __fmul_rn(powf(10.0f, exponent), 100.0f);
}

__device__ __forceinline__ void morr_bound_one(
        real q, real six_c, real lo, real hi,
        real* number, real* lambda)
{
    *number = fmaxf(*number, 0.0f);
    if (q < MQSMALL) {
        *number = 0.0f;
        *lambda = 0.0f;
        return;
    }
    real raw = cbrtf(six_c * (*number) / q);
    *lambda = fminf(fmaxf(raw, lo), hi);
    *number = q * (*lambda) * (*lambda) * (*lambda) / six_c;
}

__device__ __forceinline__ MorrMoments morr_bound(
        real qc, real qr, real qi, real qs, real qg, real rhoa, real temp,
        real* nc, real* nr, real* ni, real* ns, real* ng,
        bool reset_cloud_number, real morr_rhog)
{
    MorrMoments m;
    m.lc = m.lr = m.li = m.ls = m.lg = 0.0f;
    m.pg = 2.0f;
    if (reset_cloud_number) *nc = 250.0e6f / rhoa;
    if (qc >= MQSMALL) {
        // PGAM uses WRF's separate hard-coded 287.15 reference density.
        real rho_cloud = rhoa * RD / 287.15f;
        real pp = 0.0005714f * ((*nc) / 1.0e6f * rho_cloud) + 0.2714f;
        m.pg = fminf(fmaxf(1.0f / (pp * pp) - 1.0f, 2.0f), 10.0f);
        real raw = cbrtf((MPI / 6.0f * MRHOW * (*nc)
                          * tgammaf(m.pg + 4.0f))
                         / (qc * tgammaf(m.pg + 1.0f)));
        real lo = (m.pg + 1.0f) / 60.0e-6f;
        real hi = (m.pg + 1.0f) / 1.0e-6f;
        m.lc = fminf(fmaxf(raw, lo), hi);
        if (m.lc != raw) {
            *nc = m.lc * m.lc * m.lc * qc * tgammaf(m.pg + 1.0f)
                  / ((MPI / 6.0f * MRHOW) * tgammaf(m.pg + 4.0f));
        }
    }
    morr_bound_one(qr, MPI * MRHOW, 1.0f / 2800.0e-6f,
                   1.0f / 20.0e-6f, nr, &m.lr);
    morr_bound_one(qi, 6.0f * MCI, 1.0f / (2.0f * MDCS + 100.0e-6f),
                   1.0f / 1.0e-6f, ni, &m.li);
    morr_bound_one(qs, 6.0f * MCS, 1.0f / 2000.0e-6f,
                   1.0f / 10.0e-6f, ns, &m.ls);
    real morr_cg = morr_rhog * MPI / 6.0f;
    morr_bound_one(qg, 6.0f * morr_cg, 1.0f / 2000.0e-6f,
                   1.0f / 20.0e-6f, ng, &m.lg);
    return m;
}

__device__ __forceinline__ void morr_terminal_velocity(
        int kind, real q, real number, real rhoa, real temp,
        real* bounded_number, real* vm, real* vn,
        real morr_ag, real morr_bg, real morr_rhog)
{
    real nn = fmaxf(number, 0.0f);
    real ll = 0.0f, pg = 2.0f;
    if (kind == 0) {
        if (q >= MQSMALL) {
            real rho_cloud = rhoa * RD / 287.15f;
            real pp = 0.0005714f * (nn / 1.0e6f * rho_cloud) + 0.2714f;
            pg = fminf(fmaxf(1.0f / (pp * pp) - 1.0f, 2.0f), 10.0f);
            ll = cbrtf((MPI / 6.0f * MRHOW * nn * tgammaf(pg + 4.0f))
                        / (q * tgammaf(pg + 1.0f)));
            ll = fminf(fmaxf(ll, (pg + 1.0f) / 60.0e-6f),
                       (pg + 1.0f) / 1.0e-6f);
        }
    } else {
        real six_c, lo, hi;
        if (kind == 1) {
            six_c = MPI * MRHOW; lo = 1.0f / 2800.0e-6f;
            hi = 1.0f / 20.0e-6f;
        } else if (kind == 2) {
            six_c = 6.0f * MCI; lo = 1.0f / (2.0f * MDCS + 100.0e-6f);
            hi = 1.0f / 1.0e-6f;
        } else if (kind == 3) {
            six_c = 6.0f * MCS; lo = 1.0f / 2000.0e-6f;
            hi = 1.0f / 10.0e-6f;
        } else {
            six_c = morr_rhog * MPI; lo = 1.0f / 2000.0e-6f;
            hi = 1.0f / 20.0e-6f;
        }
        morr_bound_one(q, six_c, lo, hi, &nn, &ll);
    }
    *bounded_number = nn;

    if (q < MQSMALL) {
        *vm = *vn = 0.0f;
    } else if (kind == 0) {
        real mu = 1.496e-6f * powf(temp, 1.5f) / (temp + 120.0f);
        real acn = G * MRHOW / (18.0f * mu);
        *vm = acn * tgammaf(6.0f + pg)
              / (ll * ll * tgammaf(pg + 4.0f));
        *vn = acn * tgammaf(3.0f + pg)
              / (ll * ll * tgammaf(pg + 1.0f));
    } else {
        real aa, bb, cap, dexp;
        if (kind == 1) { aa = 841.99667f; bb = 0.8f; cap = 9.1f; dexp = 0.54f; }
        else if (kind == 2) { aa = 700.0f; bb = 1.0f; cap = 1.2f; dexp = 0.35f; }
        else if (kind == 3) { aa = 11.72f; bb = 0.41f; cap = 1.2f; dexp = 0.54f; }
        else { aa = morr_ag; bb = morr_bg; cap = 20.0f; dexp = 0.54f; }
        real an = aa * powf(MRHOSU / rhoa, dexp);
        *vm = an * tgammaf(4.0f + bb) / (6.0f * powf(ll, bb));
        *vn = an * tgammaf(1.0f + bb) / powf(ll, bb);
        real vcap = cap * powf(MRHOSU / rhoa,
                                kind == 2 ? 0.35f : 0.54f);
        *vm = fminf(*vm, vcap);
        *vn = fminf(*vn, vcap);
    }
}

struct MorrRates {
    real prc, nprc, nprc1, pra, npra, nragg, pre;
    real pracs, npracs, pracg, npracg, psmlt, evpms, pgmlt, evpmg;
    real nsmlts, nsmltr, ngmltg, ngmltr, nsubr;
    real mnuccc, nnuccc, nsagg, psacws, npsacws, psacwi, npsacwi;
    real psacwg, npsacwg, qmults, nmults, qmultr, nmultr;
    real qmultg, nmultg, qmultrg, nmultrg, pgsacw, pgracs, psacr;
    real nscng, ngracs, mnuccr, nnuccr, prci, nprci, prai, nprai;
    real nnuccd, mnuccd, prd, prds, prdg, eprd, eprds, eprdg;
    real piacr, niacr, praci, piacrs, niacrs, pracis;
    real nsubi, nsubs, nsubg;
};

__device__ __forceinline__ void morr_process_level(
        real* qv, real* qc, real* qr, real* qi, real* qs, real* qg,
        real* nc, real* nr, real* ni, real* ns, real* ng,
        real* temp, real pressure, real rhoa, real dt,
        real qvs, real qvi, real xlv, real xls, real cpm,
        bool warm, real* stale_lami, real* cloud_nc_for_sedimentation,
        real morr_ag, real morr_bg, real morr_rhog)
{
    real xlf = xls - xlv;
    // WRF 1424-1479 evaluates MU/DV/SC/AB before the warm branch's
    // small-snow/graupel melt (1498-1514).  The melt changes temperature,
    // but these transport/psychrometric coefficients stay stale.
    real mu = 1.496e-6f * powf(*temp, 1.5f) / (*temp + 120.0f);
    real dv = 8.794e-5f * powf(*temp, 1.81f) / pressure;
    real sc = mu / (rhoa * dv);
    real kap = 1.414e3f * mu;
    real ab = 1.0f + xlv * qvs / (RV * *temp * *temp) * xlv / cpm;
    real abi = 1.0f + xls * qvi / (RV * *temp * *temp) * xls / cpm;
    if (warm) {
        if (*qs < 1.0e-6f) {
            *qr += *qs; *nr += *ns; *temp -= *qs * xlf / cpm;
            *qs = 0.0f; *ns = 0.0f;
        }
        if (*qg < 1.0e-6f) {
            *qr += *qg; *nr += *ng; *temp -= *qg * xlf / cpm;
            *qg = 0.0f; *ng = 0.0f;
        }
    }
    MorrMoments m = morr_bound(*qc, *qr, *qi, *qs, *qg, rhoa, *temp,
                                nc, nr, ni, ns, ng, true, morr_rhog);
    *stale_lami = warm ? 0.0f : m.li;
    // INUM=1: DUMFNC=NC3D, without NC3DTEN, at WRF 3367-3374.
    *cloud_nc_for_sedimentation = *nc;
    MorrRates r = {};
    real dens54 = powf(MRHOSU / rhoa, 0.54f);
    real ain = 700.0f * powf(MRHOSU / rhoa, 0.35f);
    real arn = 841.99667f * dens54, asn = 11.72f * dens54;
    real agn = morr_ag * dens54;
    real n0r = *nr * m.lr, n0i = *ni * m.li;
    real n0s = *ns * m.ls, n0g = *ng * m.lg;

    // Zeroed rate-vector evaluation shared by both branches.
    if (*qc >= 1.0e-6f) {
        r.prc = 1350.0f * powf(*qc, 2.47f)
                * powf(*nc / 1.0e6f * rhoa, -1.79f);
        r.nprc1 = r.prc / (4.0f / 3.0f * MPI * MRHOW
                           * 1.5625e-14f);
        r.nprc = fminf(r.prc / (*qc / *nc), *nc / dt);
        r.nprc1 = fminf(r.nprc1, r.nprc);
    }
    if (*qr >= 1.0e-8f && *qc >= 1.0e-8f) {
        r.pra = 67.0f * powf(*qc * *qr, 1.15f);
        r.npra = r.pra / (*qc / *nc);
    }
    if (*qr >= 1.0e-8f) {
        real fb = 1.0f / m.lr < 300.0e-6f ? 1.0f
                  : 2.0f - expf(2300.0f * (1.0f / m.lr - 300.0e-6f));
        r.nragg = -5.78f * fb * *nr * *qr * rhoa;
    }
    if (*qr >= MQSMALL) {
        real epsr = 2.0f * MPI * n0r * rhoa * dv
                    * (0.78f / (m.lr * m.lr)
                       + 0.308f * sqrtf(arn * rhoa / mu)
                       * cbrtf(sc) * tgammaf(2.9f) / powf(m.lr, 2.9f));
        if (*qv < qvs) r.pre = fminf(epsr * (*qv - qvs) / ab, 0.0f);
    }

    if (warm) {
        if (*qr >= 1.0e-8f && *qs >= 1.0e-8f) {
            real ums = fminf(asn * tgammaf(4.41f) / (6.0f * powf(m.ls, 0.41f)),
                             1.2f * dens54);
            real umr = fminf(arn * tgammaf(4.8f) / (6.0f * powf(m.lr, 0.8f)),
                             9.1f * dens54);
            r.pracs = MPI * MPI * MRHOW
                      * sqrtf((1.2f * umr - 0.95f * ums)
                              * (1.2f * umr - 0.95f * ums)
                              + 0.08f * ums * umr)
                      * rhoa * n0r * n0s / (m.lr * m.lr * m.lr)
                      * (5.0f / (powf(m.lr, 3.0f) * m.ls)
                         + 2.0f / (m.lr * m.lr * m.ls * m.ls)
                         + 0.5f / (m.lr * powf(m.ls, 3.0f)));
        }
        if (*qr >= 1.0e-8f && *qg >= 1.0e-8f) {
            real umg = fminf(agn * tgammaf(4.0f + morr_bg)
                             / (6.0f * powf(m.lg, morr_bg)),
                             20.0f * dens54);
            real ung = fminf(agn * tgammaf(1.0f + morr_bg)
                             / powf(m.lg, morr_bg),
                             20.0f * dens54);
            real umr = fminf(arn * tgammaf(4.8f) / (6.0f * powf(m.lr, 0.8f)),
                             9.1f * dens54);
            real unr = fminf(arn * tgammaf(1.8f) / powf(m.lr, 0.8f),
                             9.1f * dens54);
            real vrelm = sqrtf((1.2f * umr - 0.95f * umg)
                               * (1.2f * umr - 0.95f * umg)
                               + 0.08f * umg * umr);
            r.pracg = MPI * MPI * MRHOW * vrelm * rhoa * n0r * n0g
                      / powf(m.lr, 3.0f)
                      * (5.0f / (powf(m.lr, 3.0f) * m.lg)
                         + 2.0f / (m.lr * m.lr * m.lg * m.lg)
                         + 0.5f / (m.lr * powf(m.lg, 3.0f)));
            real vreln = sqrtf(1.7f * (unr - ung) * (unr - ung)
                               + 0.3f * unr * ung);
            real collected_n = MPI / 2.0f * rhoa * vreln * n0r * n0g
                               * (1.0f / (powf(m.lr, 3.0f) * m.lg)
                                  + 1.0f / (m.lr * m.lr * m.lg * m.lg)
                                  + 1.0f / (m.lr * powf(m.lg, 3.0f)));
            r.npracg = collected_n - r.pracg / 5.2e-7f;
        }
        if (*qs >= 1.0e-8f) {
            real accel = -4187.0f / xlf * (*temp - 273.15f) * r.pracs;
            r.psmlt = 2.0f * MPI * n0s * kap * (273.15f - *temp) / xlf
                      * (0.86f / (m.ls * m.ls)
                         + 0.28f * sqrtf(asn * rhoa / mu) * cbrtf(sc)
                         * tgammaf(2.705f) / powf(m.ls, 2.705f)) + accel;
            if (*qv / qvs < 1.0f) {
                real epss = 2.0f * MPI * n0s * rhoa * dv
                            * (0.86f / (m.ls * m.ls)
                               + 0.28f * sqrtf(asn * rhoa / mu) * cbrtf(sc)
                               * tgammaf(2.705f) / powf(m.ls, 2.705f));
                r.evpms = fmaxf((*qv - qvs) * epss / ab, r.psmlt);
                r.psmlt -= r.evpms;
            }
        }
        if (*qg >= 1.0e-8f) {
            real accel = -4187.0f / xlf * (*temp - 273.15f) * r.pracg;
            r.pgmlt = 2.0f * MPI * n0g * kap * (273.15f - *temp) / xlf
                      * (0.86f / (m.lg * m.lg)
                         + 0.28f * sqrtf(agn * rhoa / mu) * cbrtf(sc)
                         * tgammaf(2.5f + 0.5f * morr_bg)
                         / powf(m.lg, 2.5f + 0.5f * morr_bg)) + accel;
            if (*qv / qvs < 1.0f) {
                real epsg = 2.0f * MPI * n0g * rhoa * dv
                            * (0.86f / (m.lg * m.lg)
                               + 0.28f * sqrtf(agn * rhoa / mu) * cbrtf(sc)
                               * tgammaf(2.5f + 0.5f * morr_bg)
                               / powf(m.lg, 2.5f + 0.5f * morr_bg));
                r.evpmg = fmaxf((*qv - qvs) * epsg / ab, r.pgmlt);
                r.pgmlt -= r.evpmg;
            }
        }
        // WRF 1924-1930 collision reset, then mass-only donor ratios.
        r.pracg = 0.0f; r.pracs = 0.0f;
        real loss = (r.prc + r.pra) * dt;
        if (loss > *qc && *qc >= MQSMALL) {
            real ratio = *qc / loss; r.prc *= ratio; r.pra *= ratio;
        }
        loss = (-r.psmlt - r.evpms + r.pracs) * dt;
        if (loss > *qs && *qs >= MQSMALL) {
            real ratio = *qs / loss;
            r.psmlt *= ratio; r.evpms *= ratio; r.pracs *= ratio;
        }
        loss = (-r.pgmlt - r.evpmg + r.pracg) * dt;
        if (loss > *qg && *qg >= MQSMALL) {
            real ratio = *qg / loss;
            r.pgmlt *= ratio; r.evpmg *= ratio; r.pracg *= ratio;
        }
        loss = (-r.pracs - r.pracg - r.pre - r.pra - r.prc
                + r.psmlt + r.pgmlt) * dt;
        if (loss > *qr && *qr >= MQSMALL) {
            real ratio = (*qr / dt + r.pracs + r.pracg + r.pra + r.prc
                          - r.psmlt - r.pgmlt) / (-r.pre);
            r.pre *= ratio;
        }
        real tqv = -r.pre - r.evpms - r.evpmg;
        real tt = (r.pre * xlv + (r.evpms + r.evpmg) * xls
                   + (r.psmlt + r.pgmlt - r.pracs - r.pracg) * xlf) / cpm;
        real tqc = -r.pra - r.prc;
        real tqr = r.pre + r.pra + r.prc - r.psmlt - r.pgmlt
                   + r.pracs + r.pracg;
        real tqi = 0.0f, tqs = r.psmlt + r.evpms - r.pracs;
        real tqg = r.pgmlt + r.evpmg - r.pracg;
        if (r.pre < 0.0f)
            r.nsubr = fmaxf(-1.0f, r.pre * dt / *qr) * *nr / dt;
        if (r.evpms + r.psmlt < 0.0f)
            r.nsmlts = fmaxf(-1.0f, (r.evpms + r.psmlt) * dt / *qs) * *ns / dt;
        if (r.psmlt < 0.0f)
            r.nsmltr = fmaxf(-1.0f, r.psmlt * dt / *qs) * *ns / dt;
        if (r.evpmg + r.pgmlt < 0.0f)
            r.ngmltg = fmaxf(-1.0f, (r.evpmg + r.pgmlt) * dt / *qg) * *ng / dt;
        if (r.pgmlt < 0.0f)
            r.ngmltr = fmaxf(-1.0f, r.pgmlt * dt / *qg) * *ng / dt;
        real tnc = -r.npra - r.nprc;
        real tnr = r.nprc1 + r.nragg - r.npracg + r.nsubr
                   - r.nsmltr - r.ngmltr;
        real tni = 0.0f, tns = r.nsmlts, tng = r.ngmltg;

        real tpred = *temp + dt * tt, qvpred = *qv + dt * tqv;
        real qcpred = fmaxf(*qc + dt * tqc, 0.0f);
        real ew = fminf(0.99f * pressure, morr_polysvp(tpred, false));
        real qss = EP2 * ew / (pressure - ew);
        real pcc = (qvpred - qss)
                   / (1.0f + xlv * xlv * qss / (cpm * RV * tpred * tpred)) / dt;
        if (pcc * dt + qcpred < 0.0f) pcc = -qcpred / dt;
        tqv -= pcc; tqc += pcc; tt += pcc * xlv / cpm;
        *qv += dt * tqv; *qc += dt * tqc; *qr += dt * tqr;
        *qi += dt * tqi; *qs += dt * tqs; *qg += dt * tqg;
        *nc += dt * tnc; *nr += dt * tnr; *ni += dt * tni;
        *ns += dt * tns; *ng += dt * tng; *temp += dt * tt;
        return;
    }

    // Cold rate evaluation from the single begin-of-process snapshot.
    if (*qc >= MQSMALL && *temp < 269.15f) {
        real nacnt = expf(-2.80f + 0.262f * (273.15f - *temp)) * 1000.0f;
        real slip = 7.37f * *temp / (288.0f * 10.0f * pressure) / 100.0f;
        real rin = 0.1e-6f;
        real dap = (4.0f * MPI * 1.38e-23f / (6.0f * MPI * rin)
                    * *temp * (1.0f + slip / rin) / mu);
        real cdist = *nc / tgammaf(m.pg + 1.0f);
        real bigg = expf(0.66f * (273.15f - *temp)) - 1.0f;
        r.mnuccc = MPI * MPI / 3.0f * MRHOW * dap * nacnt * cdist
                   * tgammaf(m.pg + 5.0f) / powf(m.lc, 4.0f)
                   + MPI * MPI / 36.0f * MRHOW * 100.0f * cdist
                   * tgammaf(m.pg + 7.0f) / powf(m.lc, 6.0f) * bigg;
        r.nnuccc = 2.0f * MPI * dap * nacnt * cdist
                   * tgammaf(m.pg + 2.0f) / m.lc
                   + MPI / 6.0f * 100.0f * cdist * tgammaf(m.pg + 4.0f)
                   / powf(m.lc, 3.0f) * bigg;
        r.nnuccc = fminf(r.nnuccc, *nc / dt);
    }
    if (*qs >= 1.0e-8f) {
        real cons15 = -1108.0f * 0.1f * powf(MPI, (1.0f - 0.41f) / 3.0f)
                      * powf(MRHOS, (-2.0f - 0.41f) / 3.0f) / (4.0f * 720.0f);
        r.nsagg = cons15 * asn * powf(rhoa, (2.0f + 0.41f) / 3.0f)
                  * powf(*qs, (2.0f + 0.41f) / 3.0f)
                  * powf(*ns * rhoa, (4.0f - 0.41f) / 3.0f) / rhoa;
    }
    real cons13 = tgammaf(3.41f) * MPI / 4.0f * 0.7f;
    real cons14 = tgammaf(3.0f + morr_bg) * MPI / 4.0f * 0.7f;
    real cons16 = tgammaf(4.0f) * MPI / 4.0f * 0.7f;
    if (*qs >= 1.0e-8f && *qc >= MQSMALL) {
        r.psacws = cons13 * asn * *qc * rhoa * n0s / powf(m.ls, 3.41f);
        r.npsacws = cons13 * asn * *nc * rhoa * n0s / powf(m.ls, 3.41f);
    }
    if (*qg >= 1.0e-8f && *qc >= MQSMALL) {
        r.psacwg = cons14 * agn * *qc * rhoa * n0g
                   / powf(m.lg, 3.0f + morr_bg);
        r.npsacwg = cons14 * agn * *nc * rhoa * n0g
                    / powf(m.lg, 3.0f + morr_bg);
    }
    if (*qi >= 1.0e-8f && *qc >= MQSMALL && 1.0f / m.li >= 100.0e-6f) {
        r.psacwi = cons16 * ain * *qc * rhoa * n0i / powf(m.li, 4.0f);
        r.npsacwi = cons16 * ain * *nc * rhoa * n0i / powf(m.li, 4.0f);
    }
    if (*qr >= 1.0e-8f && *qs >= 1.0e-8f) {
        real ums = fminf(asn * tgammaf(4.41f) / (6.0f * powf(m.ls, 0.41f)),
                         1.2f * dens54);
        real uns = fminf(asn * tgammaf(1.41f) / powf(m.ls, 0.41f),
                         1.2f * dens54);
        real umr = fminf(arn * tgammaf(4.8f) / (6.0f * powf(m.lr, 0.8f)),
                         9.1f * dens54);
        real unr = fminf(arn * tgammaf(1.8f) / powf(m.lr, 0.8f),
                         9.1f * dens54);
        real vrelm = sqrtf((1.2f * umr - 0.95f * ums)
                           * (1.2f * umr - 0.95f * ums) + 0.08f * ums * umr);
        real vreln = sqrtf(1.7f * (unr - uns) * (unr - uns) + 0.3f * unr * uns);
        r.pracs = fminf(MPI * MPI * MRHOW * vrelm * rhoa * n0r * n0s
                        / powf(m.lr, 3.0f)
                        * (5.0f / (powf(m.lr, 3.0f) * m.ls)
                           + 2.0f / (m.lr * m.lr * m.ls * m.ls)
                           + 0.5f / (m.lr * powf(m.ls, 3.0f))), *qr / dt);
        r.npracs = MPI / 2.0f * rhoa * vreln * n0r * n0s
                   * (1.0f / (powf(m.lr, 3.0f) * m.ls)
                      + 1.0f / (m.lr * m.lr * m.ls * m.ls)
                      + 1.0f / (m.lr * powf(m.ls, 3.0f)));
        if (*qs >= 0.1e-3f && *qr >= 0.1e-3f) {
            r.psacr = MPI * MPI * MRHOS * vrelm * rhoa * n0r * n0s
                      / powf(m.ls, 3.0f)
                      * (5.0f / (powf(m.ls, 3.0f) * m.lr)
                         + 2.0f / (m.ls * m.ls * m.lr * m.lr)
                         + 0.5f / (m.ls * powf(m.lr, 3.0f)));
        }
    }
    if (*qr >= 1.0e-8f && *qg >= 1.0e-8f) {
        real umg = fminf(agn * tgammaf(4.0f + morr_bg)
                         / (6.0f * powf(m.lg, morr_bg)),
                         20.0f * dens54);
        real ung = fminf(agn * tgammaf(1.0f + morr_bg)
                         / powf(m.lg, morr_bg),
                         20.0f * dens54);
        real umr = fminf(arn * tgammaf(4.8f) / (6.0f * powf(m.lr, 0.8f)),
                         9.1f * dens54);
        real unr = fminf(arn * tgammaf(1.8f) / powf(m.lr, 0.8f),
                         9.1f * dens54);
        real vrelm = sqrtf((1.2f * umr - 0.95f * umg)
                           * (1.2f * umr - 0.95f * umg) + 0.08f * umg * umr);
        real vreln = sqrtf(1.7f * (unr - ung) * (unr - ung) + 0.3f * unr * ung);
        r.pracg = fminf(MPI * MPI * MRHOW * vrelm * rhoa * n0r * n0g
                        / powf(m.lr, 3.0f)
                        * (5.0f / (powf(m.lr, 3.0f) * m.lg)
                           + 2.0f / (m.lr * m.lr * m.lg * m.lg)
                           + 0.5f / (m.lr * powf(m.lg, 3.0f))), *qr / dt);
        r.npracg = MPI / 2.0f * rhoa * vreln * n0r * n0g
                   * (1.0f / (powf(m.lr, 3.0f) * m.lg)
                      + 1.0f / (m.lr * m.lr * m.lg * m.lg)
                      + 1.0f / (m.lr * powf(m.lg, 3.0f)));
    }

    // Hallett-Mossop rate subtraction, WRF 2601-2713.
    if (*temp > 265.16f && *temp < 270.16f) {
        real fmult = *temp > 268.16f ? (270.16f - *temp) / 2.0f
                                     : (*temp - 265.16f) / 3.0f;
        real mmult = 4.0f / 3.0f * MPI * MRHOI * 1.25e-16f;
        if (*qs >= 0.1e-3f && (*qc >= 0.5e-3f || *qr >= 0.1e-3f)) {
            if (r.psacws > 0.0f) {
                r.nmults = 35.0e4f * r.psacws * fmult * 1000.0f;
                r.qmults = fminf(r.nmults * mmult, r.psacws);
                r.psacws -= r.qmults;
            }
            if (r.pracs > 0.0f) {
                r.nmultr = 35.0e4f * r.pracs * fmult * 1000.0f;
                r.qmultr = fminf(r.nmultr * mmult, r.pracs);
                r.pracs -= r.qmultr;
            }
        }
        if (*qg >= 0.1e-3f && (*qc >= 0.5e-3f || *qr >= 0.1e-3f)) {
            if (r.psacwg > 0.0f) {
                r.nmultg = 35.0e4f * r.psacwg * fmult * 1000.0f;
                r.qmultg = fminf(r.nmultg * mmult, r.psacwg);
                r.psacwg -= r.qmultg;
            }
            if (r.pracg > 0.0f) {
                r.nmultrg = 35.0e4f * r.pracg * fmult * 1000.0f;
                r.qmultrg = fminf(r.nmultrg * mmult, r.pracg);
                r.pracg -= r.qmultrg;
            }
        }
    }
    // Ordered snow-to-graupel redirects, WRF 2719-2763.
    if (r.psacws > 0.0f && *qs >= 0.1e-3f && *qc >= 0.5e-3f) {
        real cons17 = 3.0f * MRHOSU * MPI * 0.7f * 0.7f * tgammaf(2.82f)
                      / (morr_rhog - MRHOS);
        r.pgsacw = fminf(r.psacws, cons17 * dt * n0s * *qc * *qc * asn * asn
                         / (rhoa * powf(m.ls, 2.82f)));
        real embryo = fmaxf(MRHOS / (morr_rhog - MRHOS) * r.pgsacw, 0.0f);
        r.nscng = fminf(embryo / MMG0 * rhoa, *ns / dt);
        r.psacws -= r.pgsacw;
    }
    if (r.pracs > 0.0f && *qs >= 0.1e-3f && *qr >= 0.1e-3f) {
        real snow6 = MRHOS * MRHOS * powf(4.0f / m.ls, 6.0f);
        real rain6 = MRHOW * MRHOW * powf(4.0f / m.lr, 6.0f);
        real fs = fminf(fmaxf(snow6 / (snow6 + rain6), 0.0f), 1.0f);
        r.pgracs = (1.0f - fs) * r.pracs;
        r.ngracs = fminf((1.0f - fs) * r.npracs,
                         fminf(*nr / dt, *ns / dt));
        r.pracs -= r.pgracs; r.npracs -= r.ngracs; r.psacr *= 1.0f - fs;
    }

    if (*temp < 269.15f && *qr >= MQSMALL) {
        real bigg = expf(0.66f * (273.15f - *temp)) - 1.0f;
        r.mnuccr = 20.0f * MPI * MPI * MRHOW * 100.0f * *nr * bigg
                   / powf(m.lr, 6.0f);
        r.nnuccr = fminf(MPI * *nr * 100.0f * bigg / powf(m.lr, 3.0f),
                         *nr / dt);
    }
    if (*qi >= 1.0e-8f && *qv / qvi >= 1.0f) {
        // PRCI forms from the UNCAPPED NPRCI; the NI3D/DT limit applies to
        // the number rate only afterwards (F:2833-2836 statement order).
        r.nprci = 4.0f / (MDCS * MRHOI) * (*qv - qvi) * rhoa * n0i
                   * expf(-m.li * MDCS) * dv / abi;
        r.prci = MPI * MRHOI * powf(MDCS, 3.0f) / 6.0f * r.nprci;
        r.nprci = fminf(r.nprci, *ni / dt);
    }
    if (*qs >= 1.0e-8f && *qi >= MQSMALL) {
        real cons23 = MPI / 4.0f * 0.1f * tgammaf(3.41f);
        r.prai = cons23 * asn * *qi * rhoa * n0s / powf(m.ls, 3.41f);
        r.nprai = fminf(cons23 * asn * *ni * rhoa * n0s
                        / powf(m.ls, 3.41f), *ni / dt);
    }
    if (*qr >= 1.0e-8f && *qi >= 1.0e-8f) {
        real cons24 = MPI / 4.0f * tgammaf(3.8f);
        real cons25 = MPI * MPI / 24.0f * MRHOW * tgammaf(6.8f);
        real niacr = fminf(cons24 * *ni * n0r * arn / powf(m.lr, 3.8f)
                           * rhoa, fminf(*nr / dt, *ni / dt));
        real piacr = cons25 * *ni * n0r * arn / powf(m.lr, 6.8f) * rhoa;
        real praci = cons24 * *qi * n0r * arn / powf(m.lr, 3.8f) * rhoa;
        if (*qr >= 0.1e-3f) {
            r.niacr = niacr; r.piacr = piacr; r.praci = praci;
        } else {
            r.niacrs = niacr; r.piacrs = piacr; r.pracis = praci;
        }
    }
    if ((*qv / qvs >= 0.999f && *temp <= 265.15f) || *qv / qvi >= 1.08f) {
        real target = fminf(0.005f * expf(0.304f * (273.15f - *temp))
                            * 1000.0f, 500.0e3f) / rhoa;
        if (target > *ni + *ns + *ng) {
            r.nnuccd = (target - *ni - *ns - *ng) / dt;
            r.mnuccd = r.nnuccd * MMI0;
            // Deposition nucleation may not consume vapor that is not
            // there.  Below ~159 K POLYSVP's extrapolated liquid curve
            // drops under its ice curve, EIS = min(EW, ...) makes
            // qvi == qvs, and the 0.999*QVS trigger fires with qv AT OR
            // BELOW ice saturation; the FUDGEF rescale below skips the
            // dum <= 0 / sum_dep > 0 sign pair, so the unbounded MNUCCD
            // drove a polar-night model-top level to qv = -1.6e-4 kg/kg
            // (T255 native-suite abort at hour 49 of a 384 h run,
            // 2026-09-01).  Bound the nucleated mass by the vapor excess
            // over ice saturation and scale the number moment with it:
            // both sides of the process shrink together, so no crystal
            // count appears whose seed mass never existed.  Documented
            // divergence: module_mp_morr_two_moment.F carries the same
            // one-sided limiter but has no 156 K levels to reach it with.
            // The bound also engages above the trap, in cold first
            // nucleation whose excess is under the Cooper embryo mass
            // (the 1.08 ice-supersaturation arm near 200 K, a reachable
            // WRF tropopause state): there the dum > 0 FUDGEF branch
            // already caps the MASS in both models, so qv and qi match
            // WRF and only the number moment shrinks -- crystals keep
            // MI0 seed mass instead of WRF's mass-starved count.
            real avail = fmaxf(*qv - qvi, 0.0f) / dt;
            if (r.mnuccd > avail) {
                r.nnuccd *= avail / r.mnuccd;
                r.mnuccd = avail;
            }
        }
    }

    // Harrington deposition tail and the bidirectional collective FUDGEF
    // rescale.  NNUCCD is deliberately not rescaled (WRF 2971-3030).
    real epsi = *qi >= MQSMALL
                ? 2.0f * MPI * n0i * rhoa * dv / (m.li * m.li) : 0.0f;
    real epss = *qs >= MQSMALL
                ? 2.0f * MPI * n0s * rhoa * dv
                  * (0.86f / (m.ls * m.ls)
                     + 0.28f * sqrtf(asn * rhoa / mu) * cbrtf(sc)
                     * tgammaf(2.705f) / powf(m.ls, 2.705f)) : 0.0f;
    real epsg = *qg >= MQSMALL
                ? 2.0f * MPI * n0g * rhoa * dv
                  * (0.86f / (m.lg * m.lg)
                     + 0.28f * sqrtf(agn * rhoa / mu) * cbrtf(sc)
                     * tgammaf(2.5f + 0.5f * morr_bg)
                     / powf(m.lg, 2.5f + 0.5f * morr_bg)) : 0.0f;
    real dep = (*qv - qvi) / abi;
    real tail = *qi >= MQSMALL
                ? 1.0f - expf(-m.li * MDCS) * (1.0f + m.li * MDCS) : 0.0f;
    r.prd = epsi * dep * tail;
    if (*qs >= MQSMALL) r.prds = epss * dep + epsi * dep * (1.0f - tail);
    else r.prd += epsi * dep * (1.0f - tail);
    r.prdg = epsg * dep;
    real sum_dep = r.prd + r.prds + r.prdg + r.mnuccd;
    real dum = (*qv - qvi) / dt;
    if (sum_dep != 0.0f
            && ((dum > 0.0f && sum_dep > dum * 0.9999f)
                || (dum < 0.0f && sum_dep < dum * 0.9999f))) {
        real ratio = 0.9999f * dum / sum_dep;
        r.prd *= ratio; r.prds *= ratio; r.prdg *= ratio; r.mnuccd *= ratio;
    }
    if (r.prd < 0.0f) { r.eprd = r.prd; r.prd = 0.0f; }
    if (r.prds < 0.0f) { r.eprds = r.prds; r.prds = 0.0f; }
    if (r.prdg < 0.0f) { r.eprdg = r.prdg; r.prdg = 0.0f; }

    // WRF 3086-3187 joint mass-only donor limiting.
    real loss = (r.prc + r.pra + r.mnuccc + r.psacws + r.psacwi
                 + r.qmults + r.psacwg + r.pgsacw + r.qmultg) * dt;
    if (loss > *qc && *qc >= MQSMALL) {
        real ratio = *qc / loss;
        r.prc *= ratio; r.pra *= ratio; r.mnuccc *= ratio;
        r.psacws *= ratio; r.psacwi *= ratio; r.qmults *= ratio;
        r.qmultg *= ratio; r.psacwg *= ratio; r.pgsacw *= ratio;
    }
    loss = (-r.prd - r.mnuccc + r.prci + r.prai - r.qmults - r.qmultg
            - r.qmultr - r.qmultrg - r.mnuccd + r.praci + r.pracis
            - r.eprd - r.psacwi) * dt;
    if (loss > *qi && *qi >= MQSMALL) {
        real ratio = (*qi / dt + r.prd + r.mnuccc + r.qmults + r.qmultg
                      + r.qmultr + r.qmultrg + r.mnuccd + r.psacwi)
                     / (r.prci + r.prai + r.praci + r.pracis - r.eprd);
        r.prci *= ratio; r.prai *= ratio; r.praci *= ratio;
        r.pracis *= ratio; r.eprd *= ratio;
    }
    loss = ((r.pracs - r.pre) + (r.qmultr + r.qmultrg - r.prc)
            + (r.mnuccr - r.pra) + r.piacr + r.piacrs + r.pgracs
            + r.pracg) * dt;
    if (loss > *qr && *qr >= MQSMALL) {
        real ratio = (*qr / dt + r.prc + r.pra)
                     / (-r.pre + r.qmultr + r.qmultrg + r.pracs + r.mnuccr
                        + r.piacr + r.piacrs + r.pgracs + r.pracg);
        r.pre *= ratio; r.pracs *= ratio; r.qmultr *= ratio;
        r.qmultrg *= ratio; r.mnuccr *= ratio; r.piacr *= ratio;
        r.piacrs *= ratio; r.pgracs *= ratio; r.pracg *= ratio;
    }
    loss = (-r.prds - r.psacws - r.prai - r.prci - r.pracs - r.eprds
            + r.psacr - r.piacrs - r.pracis) * dt;
    if (loss > *qs && *qs >= MQSMALL) {
        real ratio = (*qs / dt + r.prds + r.psacws + r.prai + r.prci
                      + r.pracs + r.piacrs + r.pracis)
                     / (-r.eprds + r.psacr);
        r.eprds *= ratio; r.psacr *= ratio;
    }
    loss = (-r.psacwg - r.pracg - r.pgsacw - r.pgracs - r.prdg
            - r.mnuccr - r.eprdg - r.piacr - r.praci - r.psacr) * dt;
    if (loss > *qg && *qg >= MQSMALL) {
        real ratio = (*qg / dt + r.psacwg + r.pracg + r.pgsacw + r.pgracs
                      + r.prdg + r.mnuccr + r.psacr + r.piacr + r.praci)
                     / (-r.eprdg);
        r.eprdg *= ratio;
    }

    real tqv = -r.pre - r.prd - r.prds - r.mnuccd - r.eprd - r.eprds
               - r.prdg - r.eprdg;
    real tt = (r.pre * xlv
               + (r.prd + r.prds + r.mnuccd + r.eprd + r.eprds
                  + r.prdg + r.eprdg) * xls
               + (r.psacws + r.psacwi + r.mnuccc + r.mnuccr + r.qmults
                  + r.qmultg + r.qmultr + r.qmultrg + r.pracs + r.psacwg
                  + r.pracg + r.pgsacw + r.pgracs + r.piacr + r.piacrs)
                 * xlf) / cpm;
    real tqc = -r.pra - r.prc - r.mnuccc - r.psacws - r.psacwi
               - r.qmults - r.qmultg - r.psacwg - r.pgsacw;
    real tqi = r.prd + r.eprd + r.psacwi + r.mnuccc - r.prci - r.prai
               + r.qmults + r.qmultg + r.qmultr + r.qmultrg + r.mnuccd
               - r.praci - r.pracis;
    real tqr = r.pre + r.pra + r.prc - r.pracs - r.mnuccr - r.qmultr
               - r.qmultrg - r.piacr - r.piacrs - r.pracg - r.pgracs;
    real tqs = r.prai + r.psacws + r.prds + r.pracs + r.prci + r.eprds
               - r.psacr + r.piacrs + r.pracis;
    real tqg = r.pracg + r.psacwg + r.pgsacw + r.pgracs + r.prdg
               + r.eprdg + r.mnuccr + r.piacr + r.praci + r.psacr;
    real tnc = -r.nnuccc - r.npsacws - r.npra - r.nprc
               - r.npsacwi - r.npsacwg;
    real tni = r.nnuccc - r.nprci - r.nprai + r.nmults + r.nmultg
               + r.nmultr + r.nmultrg + r.nnuccd - r.niacr - r.niacrs;
    real tnr = r.nprc1 - r.npracs - r.nnuccr + r.nragg - r.niacr
               - r.niacrs - r.npracg - r.ngracs;
    real tns = r.nsagg + r.nprci - r.nscng - r.ngracs + r.niacrs;
    real tng = r.nscng + r.ngracs + r.nnuccr + r.niacr;
    if (r.eprd < 0.0f) r.nsubi = fmaxf(-1.0f, r.eprd * dt / *qi) * *ni / dt;
    if (r.eprds < 0.0f) r.nsubs = fmaxf(-1.0f, r.eprds * dt / *qs) * *ns / dt;
    if (r.pre < 0.0f) r.nsubr = fmaxf(-1.0f, r.pre * dt / *qr) * *nr / dt;
    if (r.eprdg < 0.0f) r.nsubg = fmaxf(-1.0f, r.eprdg * dt / *qg) * *ng / dt;
    tni += r.nsubi; tns += r.nsubs; tnr += r.nsubr; tng += r.nsubg;

    // One collective update, including saturation adjustment diagnosed from
    // the collectively predicted state (WRF 3191-3267).
    real tpred = *temp + dt * tt, qvpred = *qv + dt * tqv;
    real qcpred = fmaxf(*qc + dt * tqc, 0.0f);
    real ew = fminf(0.99f * pressure, morr_polysvp(tpred, false));
    real qss = EP2 * ew / (pressure - ew);
    real pcc = (qvpred - qss)
               / (1.0f + xlv * xlv * qss / (cpm * RV * tpred * tpred)) / dt;
    if (pcc * dt + qcpred < 0.0f) pcc = -qcpred / dt;
    tqv -= pcc; tqc += pcc; tt += pcc * xlv / cpm;
    *qv += dt * tqv; *qc += dt * tqc; *qr += dt * tqr;
    *qi += dt * tqi; *qs += dt * tqs; *qg += dt * tqg;
    *nc += dt * tnc; *nr += dt * tnr; *ni += dt * tni;
    *ns += dt * tns; *ng += dt * tng; *temp += dt * tt;
}

__device__ int morr_sediment_nstep(
        const real* qc, const real* qr, const real* qi,
        const real* qs, const real* qg,
        const real* nc, const real* nr, const real* ni,
        const real* ns, const real* ng,
        const real* cloud_nc_for_sedimentation,
        const real* theta, const real* pii, const real* rho_fixed,
        const real* dz, int j, int i, int nz, int ny, int nx, real dt,
        real morr_ag, real morr_bg, real morr_rhog)
{
    real max_courant = 0.0f;
    // Level-outer: rho, theta, pii and dz are one load per level instead of
    // one per level per category.  The reduction is a chain of fmaxf, which
    // is exactly associative and commutative -- max rounds nothing, and
    // fmaxf(NaN, x) == x makes NaN a two-sided identity -- so regrouping it
    // by level cannot move a bit.  Each category still walks downward, so
    // its empty-level speed rebound is unchanged; the five carried pairs
    // stay in registers because the category loop is fully unrolled.
    real vm_above[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    real vn_above[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int k = nz - 1; k >= 0; --k) {
        size_t idx = IDX3(k, j, i);
        real rhoa = fabsf(rho_fixed[idx]);
        real temp = theta[idx] * pii[idx];
        real dzk = dz[idx];
#pragma unroll
        for (int kind = 0; kind < 5; ++kind) {
            const real* mass = kind == 0 ? qc : (kind == 1 ? qr :
                               (kind == 2 ? qi : (kind == 3 ? qs : qg)));
            const real* number = kind == 0 ? cloud_nc_for_sedimentation :
                                 (kind == 1 ? nr :
                                 (kind == 2 ? ni : (kind == 3 ? ns : ng)));
            real nn, vm, vn;
            morr_terminal_velocity(kind, fmaxf(mass[idx], 0.0f),
                                   number[idx], rhoa, temp, &nn, &vm, &vn,
                                   morr_ag, morr_bg, morr_rhog);
            if (k < nz - 1) {
                if (vm < 1.0e-10f) vm = vm_above[kind];
                if (vn < 1.0e-10f) vn = vn_above[kind];
            }
            vm_above[kind] = vm; vn_above[kind] = vn;
            max_courant = fmaxf(max_courant,
                                fmaxf(vm, vn) * dt / dzk);
        }
    }
    return max((int)(max_courant + 1.0f), 1);
}

template <int KMAX>
__device__ __forceinline__ real morr_sediment_pair(
        real* mass, real* number, const real* sediment_number,
        const real* theta, const real* pii,
        const real* pressure, const real* rho_fixed, const real* dz, int j, int i,
        int nz, int ny, int nx, int kind, real dt, int nstep,
        real morr_ag, real morr_bg, real morr_rhog)
{
    real qd[KMAX], nd[KMAX], nd0[KMAX];
    real vm[KMAX], vn[KMAX];
    real vm_above = 0.0f, vn_above = 0.0f;
    int ktop = -1;
    for (int k = nz - 1; k >= 0; --k) {
        size_t idx = IDX3(k, j, i);
        real temp = theta[idx] * pii[idx];
        real rhoa = fabsf(rho_fixed[idx]);
        real q = fmaxf(mass[idx], 0.0f);
        real nn;
        real nsed = sediment_number == nullptr ? number[idx]
                                                : sediment_number[idx];
        morr_terminal_velocity(kind, q, nsed, rhoa, temp,
                               &nn, &vm[k], &vn[k],
                               morr_ag, morr_bg, morr_rhog);
        if (k < nz - 1) {
            if (vm[k] < 1.0e-10f) vm[k] = vm_above;
            if (vn[k] < 1.0e-10f) vn[k] = vn_above;
        }
        vm_above = vm[k]; vn_above = vn[k];
        // Highest level this category can fall from.  The rebound above
        // makes the speeds non-zero all the way down from it, so the
        // sedimenting span is exactly [0, ktop].
        if (ktop < 0 && (vm[k] != 0.0f || vn[k] != 0.0f)) ktop = k;
        qd[k] = q * rhoa;
        // DLAM rebound changes fall speed only; flux the clipped prognostic
        // number moment itself (WRF 3376-3432).
        nd[k] = fmaxf(nsed, 0.0f) * rhoa;
        nd0[k] = nd[k];
    }
    real dts = dt / (real)nstep;
    real exported = 0.0f;
    // Above ktop both speeds are exactly zero, so every flux there is
    // 0*qd == +0 on a quantity that cannot be negative: those levels are the
    // identity and are skipped.  Entering the span with a zero inflow rather
    // than a separate top statement is bit-exact -- (0-x) == -x and
    // ((0-x)*d)/z == -((x*d)/z) -- and keeps the k==0 surface export on the
    // one path that reaches the ground, which a span top of 0 shares.
    for (int nsub = 0; ktop >= 0 && nsub < nstep; ++nsub) {
        // The update walks downward, so only the flux through the interface
        // above the current level must survive.  Carrying those two FP32
        // values preserves every per-level expression while avoiding two
        // KMAX local arrays (512 bytes/thread in the d01 specialization).
        real fm_above = 0.0f, fn_above = 0.0f;
        for (int k = ktop; k >= 0; --k) {
            size_t idx = IDX3(k, j, i);
            real fm_here = vm[k] * qd[k];
            real fn_here = vn[k] * nd[k];
            if (k == 0) exported += fm_here * dts;
            qd[k] += (fm_above - fm_here) * dts / dz[idx];
            nd[k] += (fn_above - fn_here) * dts / dz[idx];
            fm_above = fm_here;
            fn_above = fn_here;
        }
    }
    for (int k = 0; k < nz; ++k) {
        size_t idx = IDX3(k, j, i);
        real temp = theta[idx] * pii[idx];
        real rhoa = rho_fixed[idx];
        mass[idx] = fmaxf(qd[k] / rhoa, 0.0f);
        real sedimented = fmaxf(nd[k] / rhoa, 0.0f);
        if (sediment_number == nullptr) {
            number[idx] = sedimented;
        } else {
            // DUMFNC excludes the local NC3DTEN for INUM=1.  WRF adds the
            // fallout tendency (sedimented DUMFNC minus DUMFNC) to the
            // already-held local tendency; do not replace post-process NC.
            number[idx] = fmaxf(number[idx]
                                 + (nd[k] - nd0[k]) / rhoa, 0.0f);
        }
    }
    return exported;
}

extern "C" __global__
void morrison_process_levels(real* __restrict__ theta,
                             real* __restrict__ qv,
                             real* __restrict__ qc,
                             real* __restrict__ qr,
                             real* __restrict__ qi,
                             real* __restrict__ qs,
                             real* __restrict__ qg,
                             real* __restrict__ nc,
                             real* __restrict__ nr,
                             real* __restrict__ ni,
                             real* __restrict__ ns,
                             real* __restrict__ ng,
                             const real* __restrict__ qrcuten,
                             const real* __restrict__ qscuten,
                             const real* __restrict__ qicuten,
                             real* __restrict__ rho_in,
                             const real* __restrict__ pii,
                             const real* __restrict__ pressure,
                             real* __restrict__ ice_to_snow_scratch,
                             real* __restrict__ effc,
                             real* __restrict__ effi,
                             real* __restrict__ effs,
                             real morr_ag, real morr_bg, real morr_rhog,
                             real dt, int has_cu_tendencies, int ncell)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)ncell) return;
    real piik = pii[idx], p = pressure[idx];
    real temp = theta[idx] * piik;
    real rhoa = p / (RD * temp);
    real qvk = fmaxf(qv[idx], 0.0f), qck = fmaxf(qc[idx], 0.0f);
    real qrk = fmaxf(qr[idx], 0.0f), qik = fmaxf(qi[idx], 0.0f);
    real qsk = fmaxf(qs[idx], 0.0f), qgk = fmaxf(qg[idx], 0.0f);
    real nck = fmaxf(nc[idx], 0.0f), nrk = fmaxf(nr[idx], 0.0f);
    real nik = fmaxf(ni[idx], 0.0f), nsk = fmaxf(ns[idx], 0.0f);
    real ngk = fmaxf(ng[idx], 0.0f);
    real xlv = 3.1484e6f - 2370.0f * temp;
    real xls = 3.15e6f - 2370.0f * temp + 0.3337e6f;
    real cpm = CP * (1.0f + 0.887f * qvk);
    // Reuse output buffers as WRF-statement-order scratch until the
    // post-sedimentation radii are diagnosed below.
    effc[idx] = xlv;
    effi[idx] = cpm;
    real ew = fminf(0.99f * p, morr_polysvp(temp, false));
    real ei = fminf(ew, fminf(0.99f * p, morr_polysvp(temp, true)));
    real qvs = EP2 * ew / (p - ew), qvi = EP2 * ei / (p - ei);

    // WRF v4.6.1 module_mp_morr_two_moment.F:1327-1343.  Apply the raw
    // Kain-Fritsch mass rates in exact rain, snow, ice statement order,
    // after density diagnosis and before entry cleanup/PSD reconstruction.
    if (has_cu_tendencies) {
        real qrcu = qrcuten[idx];
        if (qrcu >= 1.0e-10f) {
            nrk += 1.8e5f * powf(
                qrcu * dt / (MPI * MRHOW * rhoa * rhoa * rhoa), 0.25f);
        }
        real qscu = qscuten[idx];
        if (qscu >= 1.0e-10f) {
            nsk += 3.0e5f * powf(
                qscu * dt / (100.0f * MPI * rhoa * rhoa * rhoa), 0.25f);
        }
        real qicu = qicuten[idx];
        if (qicu >= 1.0e-10f) {
            nik += qicu * dt / (MCI * 80.0e-6f * 80.0e-6f * 80.0e-6f);
        }
    }

    if (qvk / qvs < 0.9f) {
        if (qrk < 1.0e-8f) { qvk += qrk; temp -= qrk * xlv / cpm; qrk = 0.0f; }
        if (qck < 1.0e-8f) { qvk += qck; temp -= qck * xlv / cpm; qck = 0.0f; }
    }
    if (qvk / qvi < 0.9f) {
        if (qik < 1.0e-8f) { qvk += qik; temp -= qik * xls / cpm; qik = 0.0f; }
        if (qsk < 1.0e-8f) { qvk += qsk; temp -= qsk * xls / cpm; qsk = 0.0f; }
        if (qgk < 1.0e-8f) { qvk += qgk; temp -= qgk * xls / cpm; qgk = 0.0f; }
    }
    if (qck < MQSMALL) { qck = 0.0f; nck = 0.0f; }
    if (qrk < MQSMALL) { qrk = 0.0f; nrk = 0.0f; }
    if (qik < MQSMALL) { qik = 0.0f; nik = 0.0f; }
    if (qsk < MQSMALL) { qsk = 0.0f; nsk = 0.0f; }
    if (qgk < MQSMALL) { qgk = 0.0f; ngk = 0.0f; }
    bool warm = temp >= 273.15f;
    real qi_begin = qik, stale_lami = 0.0f;
    real cloud_nc_for_sedimentation = 0.0f;
    morr_process_level(&qvk, &qck, &qrk, &qik, &qsk, &qgk,
                       &nck, &nrk, &nik, &nsk, &ngk,
                       &temp, p, rhoa, dt, qvs, qvi, xlv, xls, cpm,
                       warm, &stale_lami,
                       &cloud_nc_for_sedimentation,
                       morr_ag, morr_bg, morr_rhog);
    bool ice_to_snow = (!warm && qi_begin >= MQSMALL
                        && stale_lami >= 1.0e-10f
                        && 1.0f / stale_lami >= 2.0f * MDCS);
    rho_in[idx] = rhoa;
    ice_to_snow_scratch[idx] = ice_to_snow ? 1.0f : 0.0f;
    effs[idx] = cloud_nc_for_sedimentation;
    theta[idx] = temp / piik;
    qv[idx] = qvk; qc[idx] = qck; qr[idx] = qrk;
    qi[idx] = qik; qs[idx] = qsk; qg[idx] = qgk;
    nc[idx] = nck; nr[idx] = nrk; ni[idx] = nik;
    ns[idx] = nsk; ng[idx] = ngk;
}

template <int KMAX>
__device__ __forceinline__
void morrison_sediment_impl(real* __restrict__ qc,
                            real* __restrict__ qr,
                            real* __restrict__ qi,
                            real* __restrict__ qs,
                            real* __restrict__ qg,
                            real* __restrict__ nc,
                            real* __restrict__ nr,
                            real* __restrict__ ni,
                            real* __restrict__ ns,
                            real* __restrict__ ng,
                            const real* __restrict__ cloud_nc,
                            const real* __restrict__ theta,
                            const real* __restrict__ pii,
                            const real* __restrict__ pressure,
                            const real* __restrict__ rho_in,
                            const real* __restrict__ dz,
                            real* __restrict__ rainnc,
                            real* __restrict__ rainncv,
                            real* __restrict__ snownc,
                            real* __restrict__ snowncv,
                            real* __restrict__ graupelnc,
                            real* __restrict__ graupelncv,
                            real* __restrict__ sr,
                            real dt, real morr_ag, real morr_bg,
                            real morr_rhog, int nz, int ny, int nx)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ny * nx) return;
    int j = col / nx;
    int i = col - j * nx;
    int nstep = morr_sediment_nstep(qc, qr, qi, qs, qg, nc, nr, ni, ns, ng,
                                    cloud_nc,
                                    theta, pii, rho_in, dz,
                                    j, i, nz, ny, nx, dt,
                                    morr_ag, morr_bg, morr_rhog);
    real out_c = morr_sediment_pair<KMAX>(qc, nc, cloud_nc, theta, pii, pressure,
                                          rho_in, dz,
                                          j, i, nz, ny, nx, 0, dt, nstep,
                                          morr_ag, morr_bg, morr_rhog);
    real out_r = morr_sediment_pair<KMAX>(qr, nr, nullptr, theta, pii, pressure,
                                          rho_in, dz,
                                          j, i, nz, ny, nx, 1, dt, nstep,
                                          morr_ag, morr_bg, morr_rhog);
    real out_i = morr_sediment_pair<KMAX>(qi, ni, nullptr, theta, pii, pressure,
                                          rho_in, dz,
                                          j, i, nz, ny, nx, 2, dt, nstep,
                                          morr_ag, morr_bg, morr_rhog);
    real out_s = morr_sediment_pair<KMAX>(qs, ns, nullptr, theta, pii, pressure,
                                          rho_in, dz,
                                          j, i, nz, ny, nx, 3, dt, nstep,
                                          morr_ag, morr_bg, morr_rhog);
    real out_g = morr_sediment_pair<KMAX>(qg, ng, nullptr, theta, pii, pressure,
                                          rho_in, dz,
                                          j, i, nz, ny, nx, 4, dt, nstep,
                                          morr_ag, morr_bg, morr_rhog);

    size_t sidx = (size_t)j * nx + i;
    real total = out_c + out_r + out_i + out_s + out_g;
    real snow = out_i + out_s;
    rainncv[sidx] = total;
    snowncv[sidx] = snow;
    graupelncv[sidx] = out_g;
    rainnc[sidx] += total;
    snownc[sidx] += snow;
    graupelnc[sidx] += out_g;
    sr[sidx] = (snow + out_g) / (total + 1.0e-12f);
}

#define MORRISON_SEDIMENT_PARAMETERS                                         \
    real* __restrict__ qc, real* __restrict__ qr,                            \
    real* __restrict__ qi, real* __restrict__ qs,                            \
    real* __restrict__ qg, real* __restrict__ nc,                            \
    real* __restrict__ nr, real* __restrict__ ni,                            \
    real* __restrict__ ns, real* __restrict__ ng,                            \
    const real* __restrict__ cloud_nc, const real* __restrict__ theta,       \
    const real* __restrict__ pii, const real* __restrict__ pressure,         \
    const real* __restrict__ rho_in, const real* __restrict__ dz,            \
    real* __restrict__ rainnc, real* __restrict__ rainncv,                    \
    real* __restrict__ snownc, real* __restrict__ snowncv,                    \
    real* __restrict__ graupelnc, real* __restrict__ graupelncv,              \
    real* __restrict__ sr, real dt, real morr_ag, real morr_bg,              \
    real morr_rhog, int nz, int ny, int nx

#define MORRISON_SEDIMENT_ARGUMENTS                                          \
    qc, qr, qi, qs, qg, nc, nr, ni, ns, ng, cloud_nc, theta, pii, pressure, \
    rho_in, dz, rainnc, rainncv, snownc, snowncv, graupelnc, graupelncv, sr, \
    dt, morr_ag, morr_bg, morr_rhog, nz, ny, nx

extern "C" __global__
void morrison_sediment_64(MORRISON_SEDIMENT_PARAMETERS)
{
    morrison_sediment_impl<MORR_KMAX_SHALLOW>(MORRISON_SEDIMENT_ARGUMENTS);
}

extern "C" __global__
void morrison_sediment_256(MORRISON_SEDIMENT_PARAMETERS)
{
    morrison_sediment_impl<MORR_KMAX_GENERIC>(MORRISON_SEDIMENT_ARGUMENTS);
}

#undef MORRISON_SEDIMENT_ARGUMENTS
#undef MORRISON_SEDIMENT_PARAMETERS

extern "C" __global__
void morrison_finalize_levels(real* __restrict__ theta,
                              real* __restrict__ qv,
                              real* __restrict__ qc,
                              real* __restrict__ qr,
                              real* __restrict__ qi,
                              real* __restrict__ qs,
                              real* __restrict__ qg,
                              real* __restrict__ nc,
                              real* __restrict__ nr,
                              real* __restrict__ ni,
                              real* __restrict__ ns,
                              real* __restrict__ ng,
                              const real* __restrict__ rho_in,
                              const real* __restrict__ pii,
                              const real* __restrict__ pressure,
                              const real* __restrict__ ice_to_snow_scratch,
                              real* __restrict__ effc,
                              real* __restrict__ effi,
                              real* __restrict__ effs,
                              real* __restrict__ effr,
                              real morr_rhog,
                              int ncell)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)ncell) return;
    real temp = theta[idx] * pii[idx];
    bool ice_to_snow = ice_to_snow_scratch[idx] != 0.0f;
    real rhoa = rho_in[idx];
    real qvk = qv[idx], qck = qc[idx], qrk = qr[idx];
    real qik = qi[idx], qsk = qs[idx], qgk = qg[idx];
    real nck = nc[idx], nrk = nr[idx], nik = ni[idx];
    real nsk = ns[idx], ngk = ng[idx];

    // The conversion mask already latched WRF's pre-update T3D test at
    // 3679-3689; the application does not recheck post-update T.
    if (ice_to_snow && qik > 0.0f) {
        qsk += qik; nsk += nik; qik = 0.0f; nik = 0.0f;
    }
    // XXLV/XXLS/CPM are the stale per-level values diagnosed before
    // any process update (WRF 1298-1304, reused at 3729-3849).
    real xlv = effc[idx];
    real xls = xlv + 0.3353e6f;
    real xlf = xls - xlv;
    real cpm = effi[idx];
    real ew = fminf(0.99f * pressure[idx], morr_polysvp(temp, false));
    real ei = fminf(ew, fminf(0.99f * pressure[idx], morr_polysvp(temp, true)));
    real qvs = EP2 * ew / (pressure[idx] - ew);
    real qvi = EP2 * ei / (pressure[idx] - ei);
    if (qvk / qvs < 0.9f) {
        if (qrk < 1.0e-8f) { qvk += qrk; temp -= qrk * xlv / cpm; qrk = 0.0f; }
        if (qck < 1.0e-8f) { qvk += qck; temp -= qck * xlv / cpm; qck = 0.0f; }
    }
    if (qvk / qvi < 0.9f) {
        if (qik < 1.0e-8f) { qvk += qik; temp -= qik * xls / cpm; qik = 0.0f; }
        if (qsk < 1.0e-8f) { qvk += qsk; temp -= qsk * xls / cpm; qsk = 0.0f; }
        if (qgk < 1.0e-8f) { qvk += qgk; temp -= qgk * xls / cpm; qgk = 0.0f; }
    }
    if (qck < MQSMALL) { qck = 0.0f; nck = 0.0f; }
    if (qrk < MQSMALL) { qrk = 0.0f; nrk = 0.0f; }
    if (qik < MQSMALL) { qik = 0.0f; nik = 0.0f; }
    if (qsk < MQSMALL) { qsk = 0.0f; nsk = 0.0f; }
    if (qgk < MQSMALL) { qgk = 0.0f; ngk = 0.0f; }

    if (qik >= MQSMALL && temp >= 273.15f) {
        qrk += qik; nrk += nik; temp -= qik * xlf / cpm;
        qik = 0.0f; nik = 0.0f;
    }
    if (temp <= 233.15f && qck >= MQSMALL) {
        qik += qck; nik += nck; temp += qck * xlf / cpm;
        qck = 0.0f; nck = 0.0f;
    }
    if (temp <= 233.15f && qrk >= MQSMALL) {
        qgk += qrk; ngk += nrk; temp += qrk * xlf / cpm;
        qrk = 0.0f; nrk = 0.0f;
    }

    qc[idx] = qck; qr[idx] = qrk; qi[idx] = qik;
    qs[idx] = qsk; qg[idx] = qgk;
    nc[idx] = nck; nr[idx] = nrk; ni[idx] = nik;
    ns[idx] = nsk; ng[idx] = ngk;
    // Final PSD reconstruction rebounds LAMC from the transient updated
    // NC3D (WRF 3918-3947).  Fixed 250 cm-3 is restored only after EFFC.
    MorrMoments m = morr_bound(qc[idx], qr[idx], qi[idx], qs[idx], qg[idx],
                                rhoa, temp, &nc[idx], &nr[idx], &ni[idx],
                                &ns[idx], &ng[idx], false, morr_rhog);
    effc[idx] = qc[idx] >= MQSMALL
              ? (m.pg + 3.0f) / (2.0f * m.lc) * 1.0e6f : 25.0f;
    effr[idx] = qr[idx] >= MQSMALL ? 1.5e6f / m.lr : 25.0f;
    effi[idx] = qi[idx] >= MQSMALL ? 1.5e6f / m.li : 25.0f;
    effs[idx] = qs[idx] >= MQSMALL ? 1.5e6f / m.ls : 25.0f;
    ni[idx] = fminf(ni[idx], 0.3e6f / rhoa);
    nc[idx] = 250.0e6f / rhoa;
    theta[idx] = temp / pii[idx];
}
