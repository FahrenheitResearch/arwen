// ==========================================================================
// Grell 3D ensemble cumulus (WRF cu_physics = 5, G3SCHEME), WRF v4.6.1
// phys/module_cu_g3.F, transcribed in WRF's operation order.
//
// WRF is public domain; this is a clean transcription of the scheme, held
// word for word against a column oracle built from the byte-unmodified
// module (tools/g3_wrf461_oracle, tests/test_g3_cu5_parity.py).
//
// SHAPE.  Two entry points, mirroring the two WRF calls:
//
//   g3_column      G3DRV + CUP_enss_3d for one mass column per thread, over
//                  the interior band WRF writes (module_cu_g3.F:485-574).
//                  The column reads its 3 x 3 neighbourhood of w, RTHFTEN,
//                  RQVFTEN (and, under cugd_avedx > 1, the 5 x 5 the
//                  ave_f_t/ave_f_q average needs) straight from the domain
//                  arrays: that is G3DRV's own ens4 = 9 forcing ensemble
//                  (:404-426), not a halo this code invents.
//   g3_spread      conv_grell_spread3d (:5037-5322): the 3 x 3 subsidence
//                  spreading, the 1-2-1 horizontal and vertical smoothing,
//                  and the moistening and heating limiters.  It runs after
//                  every g3_column column has written its cugd_* fields,
//                  which is WRF's own HALO_CUP_G3_OUT ordering.
//
// ARITHMETIC.  Plain float32 C expressions, parenthesised to Fortran's
// left-to-right evaluation, every literal suffixed f.  The SAME source
// compiles both arms through the shared direct-NVRTC cache with
// --fmad=false and --ftz=false. Default FTZ changed closure decisions.
// Both arms require 0 ULP. EXP and ALOG call WOOF's own float32 routines
// gfk_exp/gfk_log.
// sqrt is correctly rounded on both sides.  Integer powers follow
// gfortran's lowering: x**2 = x*x, x**3 = (x*x)*x, x**4 = (x*x)*(x*x).
//
// DEAD WRF WORK NOT CARRIED (no output reads it):
//   * the storm-motion direction (atan2 of the 300-850 hPa mean wind,
//     :427-442): its only reader, cup_direction2, is commented out;
//   * massfln, the "downdraft mass flux ensemble for the next step": G3DRV
//     never copies massflni out;
//   * massflx_stats' per-closure std/skewness/curtosis and the total
//     skewness/curtosis; the total std IS carried (xmb reads it);
//   * sub_mas (G3DRV's subm is never read), cup_dd_aa0 (never called),
//     the random closure picks (irandom is forced to 0 at :3146).
//
// DECLARED DIVERGENCES (WRF reads memory it never wrote; there is no
// reference answer to be exact to):
//   * K22/KBCON/KTOP_SHALLOW with ishallow = 0: G3DRV copies locals
//     CUP_enss_3d only assigns inside the shallow block (:487-490 against
//     :1223).  MEASURED stack residue in WRF (the -O0 and -O2 builds of the
//     pinned source write different words).  This port writes 0, G3DRV's
//     own "no shallow cloud" value (:237-241); the oracle pins the read to
//     0 and a poisoned control proves nothing else depends on it.
//   * kdet when no cloud level lies 1250 m above the terrain (:1008-1014):
//     WRF leaves it unset.  This port sets ktf.  No physical column reaches
//     it, and the poisoned oracle control moves no other word.
// ==========================================================================

struct G3Col {
    float* p;
    int s;
    __device__ __forceinline__ float& operator()(int k) const {
        return p[(long long)k * s];
    }
};

// Fortran MAX/MIN on a tie between signed zeros returns the second
// argument at -O0 (measured, tools/g3_wrf461_oracle); no graded word
// depends on it (the -O0 and -O2 builds, which differ here, agree on every
// output), but the port keeps the -O0 choice.
__device__ __forceinline__ float g3max(float a, float b) { return (a > b) ? a : b; }
__device__ __forceinline__ float g3min(float a, float b) { return (a < b) ? a : b; }
__device__ __forceinline__ int g3imax(int a, int b) { return a > b ? a : b; }
__device__ __forceinline__ int g3imin(int a, int b) { return a < b ? a : b; }
__device__ __forceinline__ float g3p2(float x) { return x * x; }
__device__ __forceinline__ float g3p3(float x) { return (x * x) * x; }
__device__ __forceinline__ float g3p4(float x) { float t = x * x; return t * t; }

#ifndef G3_MAXENS
#define G3_MAXENS  3
#endif
#ifndef G3_MAXENS2
#define G3_MAXENS2 3
#endif
#ifndef G3_MAXENS3
#define G3_MAXENS3 16
#endif
#ifndef G3_ENSDIM
#define G3_ENSDIM  144
#endif
#define G3_ENS4    9

// --------------------------------------------------------------------------
// Workspace slots: one level array per slot, (nz+1) words, lane-interleaved
// so a warp's accesses to one level coalesce.  Index k is Fortran's, 1..nz.
// --------------------------------------------------------------------------
#define G3_SLOTS(X)                                                          \
    X(t2d) X(q2d) X(po) X(us) X(vs) X(tn) X(qo) X(tshall) X(qshall) X(dhdt)  \
    X(outt) X(outq) X(outqc) X(outts) X(outqs) X(subt) X(subq) X(cupclw)     \
    X(z) X(qes) X(he) X(hes) X(zo) X(qeso) X(heo) X(heso)                    \
    X(qes_cup) X(q_cup) X(he_cup) X(hes_cup) X(z_cup) X(p_cup) X(gamma_cup)  \
    X(t_cup) X(qeso_cup) X(qo_cup) X(heo_cup) X(heso_cup) X(zo_cup)          \
    X(po_cup) X(gammao_cup) X(tn_cup)                                        \
    X(cd) X(cdd) X(hc) X(dby) X(hco) X(dbyo) X(zu) X(zuo) X(zd) X(zdo)       \
    X(hcd) X(hcdo) X(dbyd) X(qcd) X(qrcd) X(pwd) X(qcdo) X(qrcdo) X(pwdo)    \
    X(qc) X(qrc) X(pw) X(clw_all) X(qco) X(qrco) X(pwo) X(tv)                \
    X(dellah) X(dellaq) X(dellat) X(dellaqc) X(dsubt) X(dsubq)               \
    X(xhe) X(xq) X(xt) X(xz) X(xqes) X(xhes) X(xqes_cup) X(xq_cup)           \
    X(xhe_cup) X(xhes_cup) X(xz_cup) X(xt_cup) X(xhc) X(xdby) X(xzu) X(xzd)  \
    X(xhcd) X(xqcd) X(xpwd) X(xqrcd) X(xqc) X(xqrc) X(xpw)                   \
    X(he3) X(hes3) X(qes3) X(z3) X(zu3_0) X(hc3_0) X(dby3_0) X(qes3_cup)     \
    X(q3_cup) X(he3_cup) X(hes3_cup) X(z3_cup) X(gamma3_cup) X(t3_cup)       \
    X(xhe3) X(xhes3) X(xqes3) X(xz3) X(xt3) X(xq3) X(xqes3_cup) X(xq3_cup)   \
    X(xhe3_cup) X(xhes3_cup) X(xz3_cup) X(xt3_cup) X(xdby3) X(xhc3) X(xzu3)  \
    X(dby3) X(qc3) X(pw3) X(hc3) X(qrc3) X(zu3) X(cd3) X(dellah3)            \
    X(dellaq3) X(dsubt3) X(dsubq3) X(dellat3) X(zdo3) X(scratch_clw)         \
    X(ax_he) X(ax_hes) X(ax_qes) X(ax_z) X(ax_qes_cup) X(ax_q_cup)           \
    X(ax_he_cup) X(ax_hes_cup) X(ax_z_cup) X(ax_p_cup) X(ax_gamma_cup)       \
    X(ax_t_cup) X(ax_dby) X(ax_hc) X(ax_zu) X(ax_cd) X(ax_tv)

enum {
#define G3_ENUM(n) G3S_##n,
    G3_SLOTS(G3_ENUM)
#undef G3_ENUM
    G3S_omeg,                               // ens4 slots each
    G3S_tx = G3S_omeg + G3_ENS4,
    G3S_qx = G3S_tx + G3_ENS4,
    G3S_subt_ens = G3S_qx + G3_ENS4,        // maxens2 slots each
    G3S_subq_ens = G3S_subt_ens + G3_MAXENS2,
    G3S_dellat_ens = G3S_subq_ens + G3_MAXENS2,
    G3S_dellaq_ens = G3S_dellat_ens + G3_MAXENS2,
    G3S_dellaqc_ens = G3S_dellaq_ens + G3_MAXENS2,
    G3S_pwo_ens = G3S_dellaqc_ens + G3_MAXENS2,
    G3_SLOT_COUNT = G3S_pwo_ens + G3_MAXENS2
};

// Host-visible: the adapter sizes the workspace from this symbol.
extern "C" __global__ void g3_slot_count(int* out) { *out = G3_SLOT_COUNT; }

// --------------------------------------------------------------------------
// cup_env (:2802-2929)
// --------------------------------------------------------------------------
__device__ void g3_cup_env(G3Col z, G3Col qes, G3Col he, G3Col hes, G3Col t,
                           G3Col q, G3Col p, float z1, float psur, int ierr,
                           float tcrit, int itest, float xl, float cp,
                           int ktf, G3Col tv)
{
    const float ht1 = xl / cp;
    const float ht2 = 2.834e6f / cp;
    const float be1 = .622f * ht1 / .286f;
    const float ae1 = be1 / 273.f + gfk_log(610.71f);
    const float be2 = .622f * ht2 / .286f;
    const float ae2 = be2 / 273.f + gfk_log(610.71f);
    if (ierr != 0) return;
    for (int k = 1; k <= ktf; ++k) {
        float ae = ae1, be = be1;
        if (t(k) <= tcrit) { ae = ae2; be = be2; }
        float e = gfk_exp(ae - be / t(k));
        qes(k) = .622f * e / (100.f * p(k) - e);
        if (qes(k) <= 1.e-08f) qes(k) = 1.e-08f;
        if (qes(k) < q(k)) qes(k) = q(k);
        tv(k) = t(k) + .608f * q(k) * t(k);
    }
    if (itest != 2) {
        z(1) = g3max(0.f, z1) - (gfk_log(p(1)) - gfk_log(psur)) * 287.f * tv(1) / 9.81f;
        for (int k = 2; k <= ktf; ++k) {
            float tvbar = .5f * tv(k) + .5f * tv(k - 1);
            z(k) = z(k - 1) - (gfk_log(p(k)) - gfk_log(p(k - 1))) * 287.f * tvbar / 9.81f;
        }
    } else {
        for (int k = 1; k <= ktf; ++k) {
            z(k) = (he(k) - 1004.f * t(k) - 2.5e6f * q(k)) / 9.81f;
            z(k) = g3max(1.e-3f, z(k));
        }
    }
    for (int k = 1; k <= ktf; ++k) {
        if (itest == 0) he(k) = 9.81f * z(k) + 1004.f * t(k) + 2.5E06f * q(k);
        hes(k) = 9.81f * z(k) + 1004.f * t(k) + 2.5E06f * qes(k);
        if (he(k) >= hes(k)) he(k) = hes(k);
    }
}

// --------------------------------------------------------------------------
// cup_env_clev (:2932-3018)
// --------------------------------------------------------------------------
__device__ void g3_cup_env_clev(G3Col t, G3Col qes, G3Col q, G3Col he,
                                G3Col hes, G3Col z, G3Col p, G3Col qes_cup,
                                G3Col q_cup, G3Col he_cup, G3Col hes_cup,
                                G3Col z_cup, G3Col p_cup, G3Col gamma_cup,
                                G3Col t_cup, float psur, int ierr, float z1,
                                float xl, float rv, float cp, int ktf)
{
    if (ierr != 0) return;
    for (int k = 2; k <= ktf; ++k) {
        qes_cup(k) = .5f * (qes(k - 1) + qes(k));
        q_cup(k) = .5f * (q(k - 1) + q(k));
        hes_cup(k) = .5f * (hes(k - 1) + hes(k));
        he_cup(k) = .5f * (he(k - 1) + he(k));
        if (he_cup(k) > hes_cup(k)) he_cup(k) = hes_cup(k);
        z_cup(k) = .5f * (z(k - 1) + z(k));
        p_cup(k) = .5f * (p(k - 1) + p(k));
        t_cup(k) = .5f * (t(k - 1) + t(k));
        gamma_cup(k) = (xl / cp) * (xl / (rv * t_cup(k) * t_cup(k))) * qes_cup(k);
    }
    qes_cup(1) = qes(1);
    q_cup(1) = q(1);
    hes_cup(1) = hes(1);
    he_cup(1) = he(1);
    z_cup(1) = .5f * (z(1) + z1);
    p_cup(1) = .5f * (p(1) + psur);
    t_cup(1) = t(1);
    gamma_cup(1) = xl / cp * (xl / (rv * t_cup(1) * t_cup(1))) * qes_cup(1);
}

// cup_MAXIMI (:3759-3812)
__device__ int g3_maximi(G3Col a, int ks, int ke, int ierr)
{
    int maxx = ks;
    if (ierr == 0) {
        float x = a(ks);
        for (int k = ks; k <= ke; ++k) {
            float xar = a(k);
            if (xar >= x) { x = xar; maxx = k; }
        }
    }
    return maxx;
}

// cup_minimi (:3815-3863)
__device__ int g3_minimi(G3Col a, int ks, int kend, int ierr)
{
    int kt = ks;
    if (ierr == 0) {
        float x = a(ks);
        int kstop = g3imax(ks + 1, kend);
        for (int k = ks + 1; k <= kstop; ++k) {
            if (a(k) < x) { x = a(k); kt = k; }
        }
    }
    return kt;
}

// cup_kbcon (:3605-3689)
__device__ void g3_cup_kbcon(float cap_inc, int iloop, int& k22, int& kbcon,
                             G3Col he_cup, G3Col hes_cup, int& ierr,
                             int kbmax, G3Col p_cup, float cap_max)
{
    kbcon = 1;
    if (ierr != 0) return;
    kbcon = k22;
    for (;;) {
        // label 32
        float hetest = he_cup(k22);
        if (iloop == 5) {
            for (int k = 1; k <= k22; ++k) hetest = g3max(hetest, he_cup(k));
        }
        if (hetest < hes_cup(kbcon)) {
            // label 31
            kbcon = kbcon + 1;
            if (kbcon > kbmax + 2) {
                if (iloop != 4) ierr = 3;
                return;
            }
            continue;
        }
        if (kbcon - k22 == 1) return;
        if (iloop == 5 && (kbcon - k22) == 0) return;
        float pbcdif = -p_cup(kbcon) + p_cup(k22);
        float plus = g3max(25.f, cap_max - (float)(iloop - 1) * cap_inc);
        if (iloop == 4) plus = cap_max;
        if (iloop == 5) plus = 25.f;
        if (iloop == 5 && cap_max > 25.f) pbcdif = -p_cup(kbcon) + cap_max;
        if (pbcdif > plus) {
            k22 = k22 + 1;
            kbcon = k22;
            continue;
        }
        return;
    }
}

// cup_ktop (:3692-3756)
__device__ void g3_cup_ktop(int ilo, G3Col dby, int kbcon, int& ktop,
                            int& ierr, int ktf)
{
    ktop = 1;
    if (ierr != 0) return;
    bool found = false;
    for (int k = kbcon + 1; k <= ktf - 1; ++k) {
        if (dby(k) <= 0.f) { ktop = k - 1; found = true; break; }
    }
    if (!found) {
        if (ilo == 1) ierr = 5;
        return;
    }
    for (int k = ktop + 1; k <= ktf; ++k) dby(k) = 0.f;
    if (kbcon == ktop) ierr = 55;
}

// cup_up_he (:4149-4254)
__device__ void g3_cup_up_he(int k22, float& hkb, G3Col z_cup, G3Col cd,
                             float entr, G3Col he_cup, G3Col hc, int kbcon,
                             int ierr, G3Col dby, G3Col he, G3Col hes_cup,
                             bool shallow, int ktf)
{
    for (int k = 1; k <= ktf; ++k) { hc(k) = 0.f; dby(k) = 0.f; }
    hkb = 0.f;
    if (ierr == 0) {
        hkb = he_cup(k22);
        if (shallow) {
            for (int k = 1; k <= k22; ++k) hkb = g3max(hkb, he_cup(k));
        }
        for (int k = 1; k <= k22; ++k) hc(k) = he_cup(k);
        for (int k = k22; k <= kbcon - 1; ++k) hc(k) = hkb;
        hc(kbcon) = hkb;
        dby(kbcon) = hkb - hes_cup(kbcon);
    }
    for (int k = 2; k <= ktf; ++k) {
        if (k > kbcon && ierr == 0) {
            float dz = z_cup(k) - z_cup(k - 1);
            hc(k) = (hc(k - 1) * (1.f - .5f * cd(k) * dz) + entr * dz * he(k - 1))
                    / (1.f + entr * dz - .5f * cd(k) * dz);
            dby(k) = hc(k) - hes_cup(k);
        }
    }
}

// cup_up_nms (:4401-4473)
__device__ void g3_cup_up_nms(G3Col zu, G3Col z_cup, float entr, G3Col cd,
                              int kbcon, int ktop, int ierr, int k22, int ktf)
{
    for (int k = 1; k <= ktf; ++k) zu(k) = 0.f;
    if (ierr == 0) {
        for (int k = k22; k <= kbcon; ++k) zu(k) = 1.f;
        for (int k = kbcon + 1; k <= ktop; ++k) {
            float dz = z_cup(k) - z_cup(k - 1);
            zu(k) = zu(k - 1) * (1.f + (entr - cd(k)) * dz);
        }
    }
}

// cup_dd_nms (:2300-2401)
__device__ void g3_cup_dd_nms(G3Col zd, G3Col z_cup, G3Col cdd, float entr,
                              int jmin, int ierr, int itest, int kdet,
                              float z1, int ktf)
{
    const float perc = .03f;
    for (int k = 1; k <= ktf; ++k) {
        zd(k) = 0.f;
        if (itest == 0) cdd(k) = 0.f;
    }
    const float a = 1.f - perc;
    if (ierr != 0) return;
    zd(jmin) = 1.f;
    for (int ki = jmin - 1; ki >= 1; --ki) {
        float dz = z_cup(ki + 1) - z_cup(ki);
        if (ki <= kdet && itest == 0) {
            cdd(ki) = entr + (1.f - (a * (z_cup(ki) - z1) + perc * (z_cup(kdet) - z1))
                              / (a * (z_cup(ki + 1) - z1) + perc * (z_cup(kdet) - z1))) / dz;
        }
        zd(ki) = zd(ki + 1) * (1.f + (entr - cdd(ki)) * dz);
    }
}

// cup_dd_he (:2087-2174)
__device__ void g3_cup_dd_he(G3Col hes_cup, G3Col zd, G3Col hcd, G3Col z_cup,
                             G3Col cdd, float entr, int jmin, int ierr,
                             G3Col he, G3Col dby, int ktf)
{
    for (int k = 2; k <= ktf; ++k) {
        dby(k) = 0.f;
        if (ierr == 0) hcd(k) = hes_cup(k);
    }
    if (ierr != 0) return;
    int k = jmin;
    hcd(k) = hes_cup(k);
    dby(k) = hcd(jmin) - hes_cup(k);
    for (int ki = jmin - 1; ki >= 1; --ki) {
        float dz = z_cup(ki + 1) - z_cup(ki);
        hcd(ki) = (hcd(ki + 1) * (1.f - .5f * cdd(ki) * dz) + entr * dz * he(ki))
                  / (1.f + entr * dz - .5f * cdd(ki) * dz);
        dby(ki) = hcd(ki) - hes_cup(ki);
    }
}

// cup_dd_moisture_3d (:2177-2297)
__device__ void g3_cup_dd_moisture(G3Col zd, G3Col hcd, G3Col hes_cup,
                                   G3Col qcd, G3Col qes_cup, G3Col pwd,
                                   G3Col q_cup, G3Col z_cup, G3Col cdd,
                                   float entr, int jmin, int& ierr,
                                   G3Col gamma_cup, float& pwev, float& bu,
                                   G3Col qrcd, G3Col q, int iloop, float xl,
                                   int high_resolution, int ktf)
{
    bu = 0.f;
    pwev = 0.f;
    for (int k = 1; k <= ktf; ++k) { qcd(k) = 0.f; qrcd(k) = 0.f; pwd(k) = 0.f; }
    if (ierr != 0) return;
    int k = jmin;
    float dz = z_cup(k + 1) - z_cup(k);
    qcd(k) = q_cup(k);
    if (high_resolution == 1) qcd(k) = .5f * (qes_cup(k) + q_cup(k));
    qrcd(k) = qes_cup(k);
    pwd(jmin) = g3min(0.f, qcd(k) - qrcd(k));
    pwev = pwev + pwd(jmin);
    qcd(k) = qes_cup(k);
    float dh = hcd(k) - hes_cup(k);
    bu = dz * dh;
    for (int ki = jmin - 1; ki >= 1; --ki) {
        dz = z_cup(ki + 1) - z_cup(ki);
        qcd(ki) = (qcd(ki + 1) * (1.f - .5f * cdd(ki) * dz) + entr * dz * q(ki))
                  / (1.f + entr * dz - .5f * cdd(ki) * dz);
        dh = hcd(ki) - hes_cup(ki);
        bu = bu + dz * dh;
        qrcd(ki) = qes_cup(ki) + (1.f / xl) * (gamma_cup(ki) / (1.f + gamma_cup(ki))) * dh;
        float dqeva = qcd(ki) - qrcd(ki);
        if (dqeva > 0.f) dqeva = 0.f;
        pwd(ki) = zd(ki) * dqeva;
        qcd(ki) = qrcd(ki);
        pwev = pwev + pwd(ki);
    }
    if (pwev == 0.f && iloop == 1) ierr = 7;
    if (bu >= 0.f && iloop == 1) ierr = 7;
}

// cup_up_moisture (:4257-4398)
__device__ void g3_cup_up_moisture(bool shallow, int ierr, G3Col z_cup,
                                   G3Col qc, G3Col qrc, G3Col pw, float& pwav,
                                   int kbcon, int ktop, G3Col cd, G3Col dby,
                                   float mentr_rate, G3Col clw_all, G3Col q,
                                   G3Col gamma_cup, G3Col zu, G3Col qes_cup,
                                   int k22, G3Col qe_cup, float xl, int ktf)
{
    float c0 = .002f;
    if (shallow) c0 = 0.f;
    pwav = 0.f;
    for (int k = 1; k <= ktf; ++k) {
        pw(k) = 0.f;
        qc(k) = 0.f;
        if (ierr == 0) qc(k) = qes_cup(k);
        clw_all(k) = 0.f;
        qrc(k) = 0.f;
    }
    if (ierr != 0) return;
    for (int k = k22; k <= kbcon - 1; ++k) qc(k) = qe_cup(k22);
    for (int k = 2; k <= ktf; ++k) {
        if (k < kbcon) continue;
        if (k > ktop) continue;
        float dz = z_cup(k) - z_cup(k - 1);
        qc(k) = (qc(k - 1) * (1.f - .5f * cd(k) * dz) + mentr_rate * dz * q(k - 1))
                / (1.f + mentr_rate * dz - .5f * cd(k) * dz);
        float qrch = qes_cup(k) + (1.f / xl) * (gamma_cup(k) / (1.f + gamma_cup(k))) * dby(k);
        clw_all(k) = qc(k) - qrch;
        qrc(k) = (qc(k) - qrch) / (1.f + c0 * dz * zu(k));
        if (qrc(k) < 0.f) qrc(k) = 0.f;
        pw(k) = c0 * dz * qrc(k) * zu(k);
        qc(k) = qrc(k) + qrch;
        pwav = pwav + pw(k);
    }
}

// cup_up_aa0 (:4079-4146)
__device__ float g3_cup_up_aa0(G3Col z, G3Col zu, G3Col dby, G3Col gamma_cup,
                               G3Col t_cup, int kbcon, int ktop, int ierr,
                               int ktf)
{
    float aa0 = 0.f;
    if (ierr != 0) return aa0;
    for (int k = 2; k <= ktf; ++k) {
        if (k <= kbcon) continue;
        if (k > ktop) continue;
        float dz = z(k) - z(k - 1);
        float da = zu(k) * dz * (9.81f / (1004.f * (t_cup(k)))) * dby(k - 1)
                   / (1.f + gamma_cup(k));
        if (k == ktop && da <= 0.f) continue;
        aa0 = aa0 + da;
        if (aa0 < 0.f) aa0 = 0.f;
    }
    return aa0;
}

// cup_dd_edt (:1970-2084)
__device__ void g3_cup_dd_edt(int ierr, G3Col us, G3Col vs, G3Col z, int ktop,
                              int kbcon, G3Col p, float pwav, float pwev,
                              float edtmax, float edtmin, float* edtc, int ktf)
{
    float vws = 0.f, sdp = 0.f, vshear = 0.f;
    for (int k = 0; k < G3_MAXENS2; ++k) edtc[k] = 0.f;
    if (ierr != 0) return;
    for (int kk = 1; kk <= ktf - 1; ++kk) {
        if (kk <= g3imin(ktop, ktf) && kk >= kbcon) {
            vws = vws + (fabsf((us(kk + 1) - us(kk)) / (z(kk + 1) - z(kk)))
                         + fabsf((vs(kk + 1) - vs(kk)) / (z(kk + 1) - z(kk))))
                        * (p(kk) - p(kk + 1));
            sdp = sdp + p(kk) - p(kk + 1);
        }
        if (kk == ktf - 1) vshear = 1.e3f * vws / sdp;
    }
    float pef = (1.591f - .639f * vshear + .0953f * g3p2(vshear)
                 - .00496f * g3p3(vshear));
    if (pef > 1.f) pef = 1.f;
    if (pef < 0.f) pef = 0.f;
    float zkbc = z(kbcon) * 3.281e-3f;
    float prezk = .02f;
    if (zkbc > 3.f) {
        prezk = .96729352f + zkbc * (-.70034167f + zkbc * (.162179896f + zkbc
                * (-1.2569798E-2f + zkbc * (4.2772E-4f - zkbc * 5.44E-6f))));
    }
    if (zkbc > 25.f) prezk = 2.4f;
    float pefb = 1.f / (1.f + prezk);
    if (pefb > 1.f) pefb = 1.f;
    if (pefb < 0.f) pefb = 0.f;
    float edt = 1.f - .5f * (pefb + pef);
    float einc = .2f * edt;
    for (int k = 1; k <= G3_MAXENS2; ++k) edtc[k - 1] = edt + (float)(k - 2) * einc;
    for (int k = 1; k <= G3_MAXENS2; ++k) {
        edtc[k - 1] = -(edtc[k - 1] * pwav / pwev);
        if (edtc[k - 1] > edtmax) edtc[k - 1] = edtmax;
        if (edtc[k - 1] < edtmin) edtc[k - 1] = edtmin;
    }
}

// cup_dellabot (:2404-2472)
__device__ void g3_cup_dellabot(G3Col he_cup, int ierr, G3Col z_cup,
                                G3Col p_cup, G3Col hcd, float edt, G3Col zd,
                                G3Col cdd, G3Col he, G3Col della, G3Col subs,
                                float mentrd_rate, float g)
{
    della(1) = 0.f;
    subs(1) = 0.f;
    if (ierr != 0) return;
    float dz = z_cup(2) - z_cup(1);
    float dp = 100.f * (p_cup(1) - p_cup(2));
    float detdo1 = edt * zd(2) * cdd(1) * dz;
    float detdo2 = edt * zd(1);
    float entdo = edt * zd(2) * mentrd_rate * dz;
    float subin = -(edt * zd(2));
    della(1) = (detdo1 * .5f * (hcd(1) + hcd(2)) + detdo2 * hcd(1)
                + subin * he_cup(2) - entdo * he(1)) * g / dp;
    subs(1) = 0.f;
}

// cup_dellas_3d (:2475-2667)
__device__ void g3_cup_dellas(int ierr, G3Col z_cup, G3Col p_cup, G3Col hcd,
                              float edt, G3Col zd, G3Col cdd, G3Col he,
                              G3Col della, G3Col subs, float mentrd_rate,
                              G3Col zu, float g, G3Col cd, G3Col hc, int ktop,
                              int k22, int kbcon, float mentr_rate, int jmin,
                              G3Col he_cup, int kdet, int kpbl, bool shallow,
                              int high_res, int ktf)
{
    int kstart = shallow ? 1 : 2;
    for (int k = kstart; k <= ktf; ++k) { della(k) = 0.f; subs(k) = 0.f; }
    if (ierr != 0) return;
    for (int k = 2; k <= ktf - 1; ++k) {
        if (k > ktop) continue;
        if (k < k22 - 1 && shallow) continue;
        float dz = z_cup(k + 1) - z_cup(k);
        float detdo = edt * cdd(k) * dz * zd(k + 1);
        float entdo = edt * mentrd_rate * dz * zd(k + 1);
        float subin = -(zd(k + 1) * edt);
        float entup = 0.f, detup = 0.f;
        if (k >= kbcon && k < ktop) {
            entup = mentr_rate * dz * zu(k);
            detup = cd(k + 1) * dz * zu(k);
        }
        float subdown = -(zd(k) * edt);
        float entdoj = 0.f, entupk = 0.f, detupk = 0.f;
        if (k == jmin) entdoj = edt * zd(k);
        if (k == k22 - 1) {
            entupk = zu(kpbl);
            subin = zu(k + 1) - zd(k + 1) * edt;
            if (high_res == 1) subin = -(zd(k + 1) * edt);
        }
        if (k > kdet) detdo = 0.f;
        if (k == ktop) {
            detupk = zu(ktop);
            subin = 0.f;
            subdown = 0.f;
        }
        if (k < kbcon) detup = 0.f;
        float dp = 100.f * (p_cup(k - 1) - p_cup(k));
        della(k) = (detup * .5f * (hc(k + 1) + hc(k))
                    + detdo * .5f * (hcd(k + 1) + hcd(k))
                    - entup * he(k)
                    - entdo * he(k)
                    + subin * he_cup(k + 1)
                    - subdown * he_cup(k)
                    + detupk * (hc(ktop) - he_cup(ktop))
                    - entupk * he_cup(k22)
                    - entdoj * he_cup(jmin)) * g / dp;
        if (high_res == 1) {
            della(k) = (detup * .5f * (hc(k + 1) + hc(k)) - entup * he(k)
                        + (entup - detup) * he(k)
                        + detdo * .5f * (hcd(k + 1) + hcd(k))
                        - entdo * he(k)
                        + subin * he_cup(k + 1)
                        - subdown * he_cup(k)
                        + detupk * (hc(ktop) - he(ktop))
                        - entdoj * he_cup(jmin)
                        - entupk * he_cup(k22) + entupk * he(k)) * g / dp;
        }
        if (k >= k22 && k < ktop) {
            subs(k) = (zu(k + 1) * he_cup(k + 1) - zu(k) * he_cup(k)) * g / dp;
        }
        if (high_res == 1) {
            if (k >= k22 && k < ktop) {
                subs(k) = (zu(k + 1) * he_cup(k + 1) - zu(k) * he_cup(k)
                           - (entup - detup) * he(k)) * g / dp;
            } else if (k == ktop) {
                subs(k) = detupk * (he(ktop) - he_cup(ktop)) * g / dp;
            } else if (k == k22 - 1) {
                subs(k) = (entupk * he(k) - entupk * he_cup(k)) * g / dp;
            }
        }
    }
}

// --------------------------------------------------------------------------
// cup_axx (:4826-5035): the ens4 = 9 member cloud work function, one per
// neighbour's forced state.
// --------------------------------------------------------------------------
__device__ void g3_cup_axx(float tcrit, int kbmax, float z1, G3Col p,
                           float psur, float xl, float rv, float cp,
                           int ierr_in, float cap_max,
                           float cap_max_increment, float entr_rate,
                           float mentr_rate, float* axx,
                           G3Col* tx, G3Col* qx,
                           G3Col he, G3Col hes, G3Col qes, G3Col z,
                           G3Col qes_cup, G3Col q_cup, G3Col he_cup,
                           G3Col hes_cup, G3Col z_cup, G3Col p_cup,
                           G3Col gamma_cup, G3Col t_cup, G3Col dby,
                           G3Col hc, G3Col zu, G3Col cd, G3Col tv, int ktf)
{
    for (int n = 0; n < G3_ENS4; ++n) axx[n] = 0.f;
    const float ht1 = xl / cp;
    const float ht2 = 2.834E6f / cp;
    const float be1 = .622f * ht1 / .286f;
    const float ae1 = be1 / 273.f + gfk_log(610.71f);
    const float be2 = .622f * ht2 / .286f;
    const float ae2 = be2 / 273.f + gfk_log(610.71f);
    for (int n = 0; n < G3_ENS4; ++n) {
        G3Col txn = tx[n], qxn = qx[n];
        for (int k = 1; k <= ktf; ++k) cd(k) = 0.1f * entr_rate;
        int ierrxx = ierr_in;
        int k22xx = 1, kbconxx = 1, ktopxx = 1;
        int kstabm = ktf - 1;
        if (ierrxx == 0) {
            for (int k = 1; k <= ktf; ++k) {
                float ae = ae1, be = be1;
                if (txn(k) <= tcrit) { ae = ae2; be = be2; }
                float e = gfk_exp(ae - be / txn(k));
                qes(k) = .622f * e / (100.f * p(k) - e);
                if (qes(k) <= 1.e-08f) qes(k) = 1.e-08f;
                if (qxn(k) > qes(k)) qxn(k) = qes(k);
                tv(k) = txn(k) + .608f * qxn(k) * txn(k);
            }
            z(1) = g3max(0.f, z1) - (gfk_log(p(1)) - gfk_log(psur)) * 287.f * tv(1) / 9.81f;
            for (int k = 2; k <= ktf; ++k) {
                float tvbar = .5f * tv(k) + .5f * tv(k - 1);
                z(k) = z(k - 1) - (gfk_log(p(k)) - gfk_log(p(k - 1))) * 287.f * tvbar / 9.81f;
            }
            for (int k = 1; k <= ktf; ++k) {
                he(k) = 9.81f * z(k) + 1004.f * txn(k) + 2.5E06f * qxn(k);
                hes(k) = 9.81f * z(k) + 1004.f * txn(k) + 2.5E06f * qes(k);
                if (he(k) >= hes(k)) he(k) = hes(k);
            }
            for (int k = 2; k <= ktf; ++k) {
                qes_cup(k) = .5f * (qes(k - 1) + qes(k));
                q_cup(k) = .5f * (qxn(k - 1) + qxn(k));
                hes_cup(k) = .5f * (hes(k - 1) + hes(k));
                he_cup(k) = .5f * (he(k - 1) + he(k));
                if (he_cup(k) > hes_cup(k)) he_cup(k) = hes_cup(k);
                z_cup(k) = .5f * (z(k - 1) + z(k));
                p_cup(k) = .5f * (p(k - 1) + p(k));
                t_cup(k) = .5f * (txn(k - 1) + txn(k));
                gamma_cup(k) = (xl / cp) * (xl / (rv * t_cup(k) * t_cup(k))) * qes_cup(k);
            }
            qes_cup(1) = qes(1);
            q_cup(1) = qxn(1);
            hes_cup(1) = hes(1);
            he_cup(1) = he(1);
            z_cup(1) = .5f * (z(1) + z1);
            p_cup(1) = .5f * (p(1) + psur);
            t_cup(1) = txn(1);
            gamma_cup(1) = xl / cp * (xl / (rv * t_cup(1) * t_cup(1))) * qes_cup(1);
        }
        k22xx = g3_maximi(he_cup, 3, kbmax, ierrxx);
        if (ierrxx == 0 && k22xx >= kbmax) ierrxx = 2;
        g3_cup_kbcon(cap_max_increment, 1, k22xx, kbconxx, he_cup, hes_cup,
                     ierrxx, kbmax, p_cup, cap_max);
        int kstabi = g3_minimi(hes_cup, kbconxx, kstabm, ierrxx);
        if (ierrxx == 0 && kstabm - 1 > kstabi) {
            for (int k = kstabi; k <= kstabm - 1; ++k) {
                cd(k) = cd(k - 1) + 1.5f * entr_rate;
                if (cd(k) > 10.0f * entr_rate) cd(k) = 10.0f * entr_rate;
            }
        }
        float hkb;
        g3_cup_up_he(k22xx, hkb, z_cup, cd, mentr_rate, he_cup, hc, kbconxx,
                     ierrxx, dby, he, hes_cup, false, ktf);
        g3_cup_ktop(1, dby, kbconxx, ktopxx, ierrxx, ktf);
        g3_cup_up_nms(zu, z_cup, mentr_rate, cd, kbconxx, ktopxx, ierrxx, k22xx, ktf);
        float aa0 = g3_cup_up_aa0(z, zu, dby, gamma_cup, t_cup, kbconxx, ktopxx,
                                  ierrxx, ktf);
        if (ierrxx == 0) axx[n] = aa0;
    }
}

// --------------------------------------------------------------------------
// massflx_stats (:4599-4824) for one column; per-closure std/ske/cur and
// the total ske/cur are dead in WRF and are not formed.  Returns the
// ensemble average in xt_ave and the total std in xt_std.
// --------------------------------------------------------------------------
__device__ void g3_massflx_stats(const float* ens, long long es, int ierr,
                                 int itest, float& xt_ave, float& xt_std,
                                 float* apr, float* prv)
{
    const int num = G3_ENSDIM / G3_MAXENS3;
    const int num2 = G3_ENSDIM / G3_MAXENS;
    float x_ave[G3_MAXENS3], x_ave_cap[G3_MAXENS];
    if (itest == 1) {
        for (int m = 0; m < 8; ++m) prv[m] = 0.f;
    }
    for (int k = 0; k < G3_MAXENS; ++k) x_ave_cap[k] = 0.f;
    for (int k = 0; k < G3_MAXENS3; ++k) x_ave[k] = 0.f;
    xt_ave = 0.f;
    xt_std = 0.f;
    if (ierr != 0) return;
    for (int kk = 1; kk <= num; ++kk)
        for (int k = 1; k <= G3_MAXENS3; ++k)
            x_ave[k - 1] = x_ave[k - 1] + ens[(G3_MAXENS3 * (kk - 1) + k - 1) * es];
    for (int iedt = 1; iedt <= G3_MAXENS2; ++iedt)
        for (int k = 1; k <= G3_MAXENS; ++k)
            for (int kk = 1; kk <= G3_MAXENS3; ++kk)
                x_ave_cap[k - 1] = x_ave_cap[k - 1]
                    + ens[(G3_MAXENS3 * (k - 1) + (iedt - 1) * G3_MAXENS * G3_MAXENS3 + kk - 1) * es];
    for (int k = 0; k < G3_MAXENS; ++k) x_ave_cap[k] = x_ave_cap[k] / (float)num2;
    for (int k = 0; k < G3_MAXENS3; ++k) x_ave[k] = x_ave[k] / (float)num;
    for (int k = 0; k < G3_MAXENS3; ++k) xt_ave = xt_ave + x_ave[k];
    xt_ave = xt_ave / (float)G3_MAXENS3;
    if (xt_ave > 0.f) {
        for (int k = 0; k < G3_MAXENS3; ++k) xt_std = xt_std + g3p2(x_ave[k] - xt_ave);
    }
    if (xt_std > 0.f) {
        xt_std = xt_std / (float)G3_MAXENS3;
        float a3 = g3max(1.e-6f, xt_std);
        xt_std = sqrtf(a3);
        if (itest == 1) {
            prv[0] = .25f * (x_ave[0] + x_ave[1] + x_ave[2] + x_ave[12]);
            prv[1] = .25f * (x_ave[3] + x_ave[4] + x_ave[5] + x_ave[13]);
            prv[2] = .25f * (x_ave[6] + x_ave[7] + x_ave[8] + x_ave[14]);
            prv[3] = .333f * (x_ave[9] + x_ave[10] + x_ave[11]);
            prv[4] = x_ave[15];
            prv[5] = x_ave_cap[0];
            prv[6] = x_ave_cap[1];
            prv[7] = x_ave_cap[2];
        } else if (itest == 2) {
            apr[0] = .25f * (x_ave[0] + x_ave[1] + x_ave[2] + x_ave[12]) * 3600.f * prv[0] + apr[0];
            apr[1] = .25f * (x_ave[3] + x_ave[4] + x_ave[5] + x_ave[13]) * 3600.f * prv[1] + apr[1];
            apr[2] = .25f * (x_ave[6] + x_ave[7] + x_ave[8] + x_ave[14]) * 3600.f * prv[2] + apr[2];
            apr[3] = .333f * (x_ave[9] + x_ave[10] + x_ave[11]) * 3600.f * prv[3] + apr[3];
            apr[4] = x_ave[15] * 3600.f * prv[4] + apr[4];
            apr[5] = x_ave_cap[0] * 3600.f * prv[5] + apr[5];
            apr[6] = x_ave_cap[1] * 3600.f * prv[6] + apr[6];
            apr[7] = x_ave_cap[2] * 3600.f * prv[7] + apr[7];
        }
    }
}

// --------------------------------------------------------------------------
// Inputs/outputs.  Level fields are [field][k][j][i] with k 0-based.
// --------------------------------------------------------------------------
enum { G3I_U, G3I_V, G3I_T, G3I_Q, G3I_P, G3I_PI, G3I_RHO, G3I_RTHFTEN,
       G3I_RQVFTEN, G3I_RTHBLTEN, G3I_RQVBLTEN, G3I_NLEV };
enum { G3I_PSFC, G3I_HT, G3I_XLAND, G3I_GSW, G3I_NSFC };
enum { G3O_CUGD_TTEN, G3O_CUGD_QVTEN, G3O_CUGD_QCTEN, G3O_CUGD_TTENS,
       G3O_CUGD_QVTENS, G3O_RQCCUTEN, G3O_RQICUTEN, G3O_GDC, G3O_GDC2,
       G3O_NLEV };
enum { G3O_RAINCV, G3O_PRATEC, G3O_HTOP, G3O_HBOT, G3O_MASS_FLUX,
       G3O_EDT_OUT, G3O_XMB_SHALLOW, G3O_NSFC };
enum { G3O_K22S, G3O_KBCONS, G3O_KTOPS, G3O_KTOP_DEEP, G3O_NISFC };

struct G3Params {
    int nx, ny, nz;
    int ibegc, iendc, jbegc, jendc;   // 0-based inclusive written band
    int ishallow, avedx, ichoice, fqi;
    float dt, dx, xlv, cp, g, rv;
};

// One forced-state value from the high-resolution 3 x 3 average
// (:266-279).  The average is formed for every domain column; WRF's halo
// outside a non-periodic domain is zero-filled. Periodic axes read the same
// wrapped words that WRF supplies through its tendency halo exchange.
__device__ __forceinline__ float g3_ave9(const float* f, int i, int j, int k,
                                         const G3Params& P)
{
    long long plane = (long long)P.nx * P.ny;
    auto at = [&](int ii, int jj) -> float {
        if (P.ibegc == 0 && P.iendc == P.nx - 1) ii = (ii + P.nx) % P.nx;
        if (P.jbegc == 0 && P.jendc == P.ny - 1) jj = (jj + P.ny) % P.ny;
        if (ii < 0 || ii >= P.nx || jj < 0 || jj >= P.ny) return 0.f;
        return f[(long long)k * plane + (long long)jj * P.nx + ii];
    };
    return (at(i - 1, j - 1) + at(i - 1, j) + at(i - 1, j + 1)
            + at(i, j - 1) + at(i, j) + at(i, j + 1)
            + at(i + 1, j - 1) + at(i + 1, j) + at(i + 1, j + 1)) / 9.f;
}

extern "C" __global__ void g3_column(
    const float* __restrict__ lev_in,    // [G3I_NLEV][nz][ny][nx]
    const float* __restrict__ w_in,      // [nz+1][ny][nx]
    const float* __restrict__ sfc_in,    // [G3I_NSFC][ny][nx]
    const int* __restrict__ kpbl_in,     // [ny][nx]
    float* __restrict__ apr,             // [8][ny][nx] in/out
    float* __restrict__ lev_out,         // [G3O_NLEV][nz][ny][nx]
    float* __restrict__ sfc_out,         // [G3O_NSFC][ny][nx]
    int* __restrict__ isfc_out,          // [G3O_NISFC][ny][nx]
    float* __restrict__ xf_ens,          // [144][ny][nx]
    float* __restrict__ pr_ens,          // [144][ny][nx]
    float* __restrict__ ws,              // workspace, G3_SLOT_COUNT*(nz+1)*nt
    int nt, int col0, int nx, int ny, int nz_, int ibegc, int iendc,
    int jbegc, int jendc, int ishallow, int avedx, int ichoice, int fqi,
    float dt_, float dx, float xlv, float cp_, float g_, float rv_)
{
    G3Params P;
    P.nx = nx; P.ny = ny; P.nz = nz_; P.ibegc = ibegc; P.iendc = iendc;
    P.jbegc = jbegc; P.jendc = jendc; P.ishallow = ishallow; P.avedx = avedx;
    P.ichoice = ichoice; P.fqi = fqi; P.dt = dt_; P.dx = dx; P.xlv = xlv;
    P.cp = cp_; P.g = g_; P.rv = rv_;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nt) return;
    const int nbx = P.iendc - P.ibegc + 1;
    const int nby = P.jendc - P.jbegc + 1;
    const int b = col0 + tid;
    if (nbx <= 0 || nby <= 0 || b >= nbx * nby) return;
    const int i = P.ibegc + b % nbx;
    const int j = P.jbegc + b / nbx;
    const int nz = P.nz;
    const int ktf = nz;
    const long long plane = (long long)P.nx * P.ny;
    const long long cix = (long long)j * P.nx + i;
    const long long lev_stride = (long long)nz * plane;

    auto slot = [&](int s) -> G3Col {
        G3Col c;
        c.p = ws + (long long)s * (nz + 1) * nt + tid;
        c.s = nt;
        return c;
    };
#define G3C(name) G3Col name = slot(G3S_##name)
    G3C(t2d); G3C(q2d); G3C(po); G3C(us); G3C(vs); G3C(tn); G3C(qo);
    G3C(tshall); G3C(qshall); G3C(dhdt); G3C(outt); G3C(outq); G3C(outqc);
    G3C(outts); G3C(outqs); G3C(subt); G3C(subq); G3C(cupclw);
    G3Col omeg[G3_ENS4], tx[G3_ENS4], qx[G3_ENS4];
    for (int n = 0; n < G3_ENS4; ++n) {
        omeg[n] = slot(G3S_omeg + n);
        tx[n] = slot(G3S_tx + n);
        qx[n] = slot(G3S_qx + n);
    }

    const float dt = P.dt;
    const float xl = P.xlv, cp = P.cp, g = P.g, rv = P.rv;
    const float tscl_kf = P.dx / 25.f;
    int high_resolution = 0;
    if (P.avedx > 1) high_resolution = 1;
    const float subcenter = 0.f;
    float sub_spread = g3max(1.f, (float)(P.avedx * P.avedx - 1));
    sub_spread = (1.f - subcenter) / sub_spread;
    const float tcrit = 258.f;

    auto lin = [&](int f, int k, int ii, int jj) -> float {
        return lev_in[(long long)f * lev_stride + (long long)(k - 1) * plane
                      + (long long)jj * P.nx + ii];
    };
    // ---- G3DRV column preparation (:343-395) ----------------------------
    const float psur = sfc_in[G3I_PSFC * plane + cix] * .01f;
    const float ter11 = sfc_in[G3I_HT * plane + cix];
    const int kpbli = kpbl_in[cix];
    for (int k = 1; k <= ktf; ++k) {
        float phh = lin(G3I_P, k, i, j);
        po(k) = phh * .01f;
        us(k) = lin(G3I_U, k, i, j);
        vs(k) = lin(G3I_V, k, i, j);
        t2d(k) = lin(G3I_T, k, i, j);
        q2d(k) = lin(G3I_Q, k, i, j);
        if (q2d(k) < 1.E-08f) q2d(k) = 1.E-08f;
        subt(k) = 0.f; subq(k) = 0.f; outt(k) = 0.f; outq(k) = 0.f;
        outqc(k) = 0.f; outts(k) = 0.f; outqs(k) = 0.f;
        float pik = lin(G3I_PI, k, i, j);
        float rthblten = lin(G3I_RTHBLTEN, k, i, j);
        float rqvblten = lin(G3I_RQVBLTEN, k, i, j);
        tn(k) = t2d(k) + lin(G3I_RTHFTEN, k, i, j) * dt;
        qo(k) = q2d(k) + lin(G3I_RQVFTEN, k, i, j) * dt;
        tshall(k) = t2d(k) + rthblten * pik * dt;
        dhdt(k) = cp * rthblten * pik + xl * rqvblten;
        qshall(k) = q2d(k) + rqvblten * dt;
        if (high_resolution == 1) {
            const float* fth = lev_in + (long long)G3I_RTHFTEN * lev_stride;
            const float* fqv = lev_in + (long long)G3I_RQVFTEN * lev_stride;
            tn(k) = t2d(k) + g3_ave9(fth, i, j, k - 1, P) * dt;
            qo(k) = q2d(k) + g3_ave9(fqv, i, j, k - 1, P) * dt;
        }
        if (tn(k) < 200.f) tn(k) = t2d(k);
        if (qo(k) < 1.E-08f) qo(k) = 1.E-08f;
    }
    // ens4 forcing members (:397-426)
    {
        int ens4n = 0;
        for (int nn = -1; nn <= 1; ++nn) {
            int jss = g3imin(g3imax(j + nn, 0), P.ny - 1);
            for (int n = -1; n <= 1; ++n) {
                int iss = g3imin(g3imax(i + n, 0), P.nx - 1);
                G3Col om = omeg[ens4n], txn = tx[ens4n], qxn = qx[ens4n];
                for (int k = 1; k <= ktf; ++k) {
                    float rhok = lin(G3I_RHO, k, i, j);
                    float wv = w_in[(long long)(k - 1) * plane + (long long)jss * P.nx + iss];
                    om(k) = -(g * rhok * wv);
                    txn(k) = t2d(k) + lin(G3I_RTHFTEN, k, iss, jss) * dt;
                    if (high_resolution == 1) {
                        const float* fth = lev_in + (long long)G3I_RTHFTEN * lev_stride;
                        txn(k) = t2d(k) + g3_ave9(fth, iss, jss, k - 1, P) * dt;
                    }
                    if (txn(k) < 200.f) txn(k) = t2d(k);
                    // :419 is overwritten by :420 -- the member's moisture
                    // forcing is the CENTRE column's, unless high-res.
                    qxn(k) = q2d(k) + lin(G3I_RQVFTEN, k, i, j) * dt;
                    if (high_resolution == 1) {
                        const float* fqv = lev_in + (long long)G3I_RQVFTEN * lev_stride;
                        qxn(k) = q2d(k) + g3_ave9(fqv, iss, jss, k - 1, P) * dt;
                    }
                    if (qxn(k) < 1.E-08f) qxn(k) = 1.E-08f;
                }
                ++ens4n;
            }
        }
    }
    float mconv[G3_ENS4];
    for (int n = 0; n < G3_ENS4; ++n) {
        mconv[n] = 0.f;
        for (int k = 1; k <= ktf - 1; ++k) {
            float dq = (q2d(k + 1) - q2d(k));
            mconv[n] = mconv[n] + omeg[n](k) * dq / g;
        }
    }
    for (int n = 0; n < G3_ENS4; ++n)
        if (mconv[n] < 0.f) mconv[n] = 0.f;

    // ======================================================================
    // CUP_enss_3d (:578-1890)
    // ======================================================================
    G3C(z); G3C(qes); G3C(he); G3C(hes); G3C(zo); G3C(qeso); G3C(heo);
    G3C(heso); G3C(qes_cup); G3C(q_cup); G3C(he_cup); G3C(hes_cup);
    G3C(z_cup); G3C(p_cup); G3C(gamma_cup); G3C(t_cup); G3C(qeso_cup);
    G3C(qo_cup); G3C(heo_cup); G3C(heso_cup); G3C(zo_cup); G3C(po_cup);
    G3C(gammao_cup); G3C(tn_cup); G3C(cd); G3C(cdd); G3C(hc); G3C(dby);
    G3C(hco); G3C(dbyo); G3C(zu); G3C(zuo); G3C(zd); G3C(zdo); G3C(hcd);
    G3C(hcdo); G3C(dbyd); G3C(qcd); G3C(qrcd); G3C(pwd); G3C(qcdo);
    G3C(qrcdo); G3C(pwdo); G3C(qc); G3C(qrc); G3C(pw); G3C(clw_all);
    G3C(qco); G3C(qrco); G3C(pwo); G3C(tv); G3C(dellah); G3C(dellaq);
    G3C(dellat); G3C(dellaqc); G3C(dsubt); G3C(dsubq); G3C(xhe); G3C(xq);
    G3C(xt); G3C(xz); G3C(xqes); G3C(xhes); G3C(xqes_cup); G3C(xq_cup);
    G3C(xhe_cup); G3C(xhes_cup); G3C(xz_cup); G3C(xt_cup); G3C(xhc);
    G3C(xdby); G3C(xzu); G3C(xzd); G3C(xhcd); G3C(xqcd); G3C(xpwd);
    G3C(xqrcd); G3C(xqc); G3C(xqrc); G3C(xpw);
    G3C(cd3); G3C(zdo3);

    float* xfc = xf_ens + cix;   // stride plane
    float* prc = pr_ens + cix;
    const long long es = plane;
    float aprc[8];
    for (int m = 0; m < 8; ++m) aprc[m] = apr[m * plane + cix];

    const float xland_in = sfc_in[G3I_XLAND * plane + cix];
    const float gsw = sfc_in[G3I_GSW * plane + cix];
    const float z1 = ter11;
    const float dtime = dt;
    const int iens = 1;

    float xmb3 = 0.f;
    float closure_n = 16.f;
    float xland1 = 1.f;
    if (xland_in > 1.5f) xland1 = 0.f;
    const float cap_max_increment = 25.f;
    float radius;
    if (iens <= 4) radius = 14000.f - (float)iens * 2000.f;
    else radius = 12000.f;
    const float entr_rate = .2f / radius;
    const float entr_rate3 = .2f / 200.f;
    const float mentrd_rate = 0.f;
    const float mentr_rate = entr_rate;
    const float mentr_rate3 = entr_rate3;
    for (int k = 1; k <= ktf; ++k) {
        cupclw(k) = 0.f;
        cd(k) = 0.01f * entr_rate;
        cd3(k) = entr_rate3;
        cdd(k) = 0.f;
        zdo3(k) = 0.f;
        hcdo(k) = 0.f;
        qrcdo(k) = 0.f;
        dellaqc(k) = 0.f;
    }
    const float edtmax = 1.f, edtmin = .2f, depth_min = 500.f;
    const float cap_maxs = 75.f;
    int kbmax = 1;
    float aa0 = 0.f, aa3_0 = 0.f, aa1 = 0.f, aa3 = 0.f;
    float edt = 0.f, edto = 0.f, edt3 = 0.f;
    int kstabm = ktf - 1;
    int ierr = 0, ierr2 = 0, ierr3 = 0, ierr5 = 0;
    float cap_max = cap_maxs;
    float cap_max3 = 25.f;
    if (gsw < 1.f || high_resolution == 1) cap_max = 25.f;
    const float zkbmax = 4000.f, zcutdown = 3000.f, z_detr = 1250.f;
    float mbdt_ens[G3_MAXENS];
    for (int nens = 1; nens <= G3_MAXENS; ++nens)
        mbdt_ens[nens - 1] = ((float)nens - 3.f) * dtime * 1.e-3f + dtime * 5.E-03f;
    for (int k = 0; k < G3_ENSDIM; ++k) { xfc[k * es] = 0.f; prc[k * es] = 0.f; }

    g3_cup_env(z, qes, he, hes, t2d, q2d, po, z1, psur, ierr, tcrit, 0, xl, cp, ktf, tv);
    g3_cup_env(zo, qeso, heo, heso, tn, qo, po, z1, psur, ierr, tcrit, 0, xl, cp, ktf, tv);
    g3_cup_env_clev(t2d, qes, q2d, he, hes, z, po, qes_cup, q_cup, he_cup,
                    hes_cup, z_cup, p_cup, gamma_cup, t_cup, psur, ierr, z1,
                    xl, rv, cp, ktf);
    g3_cup_env_clev(tn, qeso, qo, heo, heso, zo, po, qeso_cup, qo_cup,
                    heo_cup, heso_cup, zo_cup, po_cup, gammao_cup, tn_cup,
                    psur, ierr, z1, xl, rv, cp, ktf);
    int kdet = ktf;   // declared divergence: WRF leaves it unset when unfound
    for (int k = 1; k <= ktf; ++k) {
        if (zo_cup(k) > zkbmax + z1) { kbmax = k; break; }
    }
    for (int k = 1; k <= ktf; ++k) {
        if (zo_cup(k) > z_detr + z1) { kdet = k; break; }
    }
    int k22 = g3_maximi(heo_cup, 3, kbmax, ierr);
    if (ierr == 0 && k22 >= kbmax) ierr = 2;
    int kbcon;
    g3_cup_kbcon(cap_max_increment, 1, k22, kbcon, heo_cup, heso_cup, ierr,
                 kbmax, po_cup, cap_max);
    int kstabi = g3_minimi(heso_cup, kbcon, kstabm, ierr);
    if (ierr == 0 && kstabm - 1 > kstabi) {
        for (int k = kstabi; k <= kstabm - 1; ++k) {
            cd(k) = cd(k - 1) + .15f * entr_rate;
            if (cd(k) > 1.0f * entr_rate) cd(k) = 1.0f * entr_rate;
        }
    }
    float hkb, hkbo;
    g3_cup_up_he(k22, hkb, z_cup, cd, mentr_rate, he_cup, hc, kbcon, ierr,
                 dby, he, hes_cup, false, ktf);
    g3_cup_up_he(k22, hkbo, zo_cup, cd, mentr_rate, heo_cup, hco, kbcon, ierr,
                 dbyo, heo, heso_cup, false, ktf);
    int ktop;
    g3_cup_ktop(1, dbyo, kbcon, ktop, ierr, ktf);
    int kzdown = 0;
    if (ierr == 0) {
        float zktop = (zo_cup(ktop) - z1) * .6f;
        zktop = g3min(zktop + z1, zcutdown + z1);
        for (int k = 1; k <= ktf; ++k) {
            if (zo_cup(k) > zktop) { kzdown = k; break; }
        }
    }
    int jmin = g3_minimi(heso_cup, k22, kzdown, ierr);
    if (ierr == 0) {
        int jmini = jmin;
        bool keep_going = true;
        while (keep_going) {
            keep_going = false;
            if (jmini - 1 < kdet) kdet = jmini - 1;
            if (jmini >= ktop - 1) jmini = ktop - 2;
            int ki = jmini;
            hcdo(ki) = heso_cup(ki);
            float dz = zo_cup(ki + 1) - zo_cup(ki);
            float dh = 0.f;
            for (int k = ki - 1; k >= 1; --k) {
                hcdo(k) = heso_cup(jmini);
                dz = zo_cup(k + 1) - zo_cup(k);
                dh = dh + dz * (hcdo(k) - heso_cup(k));
                if (dh > 0.f) {
                    jmini = jmini - 1;
                    if (jmini > 3) {
                        keep_going = true;
                    } else {
                        ierr = 9;
                        break;
                    }
                }
            }
        }
        jmin = jmini;
        if (jmini <= 3) ierr = 4;
    }
    if (ierr == 0) {
        if (-zo_cup(kbcon) + zo_cup(ktop) < depth_min) ierr = 6;
    }
    g3_cup_up_nms(zu, z_cup, mentr_rate, cd, kbcon, ktop, ierr, k22, ktf);
    g3_cup_up_nms(zuo, zo_cup, mentr_rate, cd, kbcon, ktop, ierr, k22, ktf);
    g3_cup_dd_nms(zd, z_cup, cdd, mentrd_rate, jmin, ierr, 0, kdet, z1, ktf);
    g3_cup_dd_nms(zdo, zo_cup, cdd, mentrd_rate, jmin, ierr, 1, kdet, z1, ktf);
    g3_cup_dd_he(hes_cup, zd, hcd, z_cup, cdd, mentrd_rate, jmin, ierr, he, dbyd, ktf);
    g3_cup_dd_he(heso_cup, zdo, hcdo, zo_cup, cdd, mentrd_rate, jmin, ierr, heo, dbyd, ktf);
    float pwev, bu, pwevo, pwav, pwavo;
    g3_cup_dd_moisture(zd, hcd, hes_cup, qcd, qes_cup, pwd, q_cup, z_cup, cdd,
                       mentrd_rate, jmin, ierr, gamma_cup, pwev, bu, qrcd,
                       q2d, 2, xl, high_resolution, ktf);
    g3_cup_dd_moisture(zdo, hcdo, heso_cup, qcdo, qeso_cup, pwdo, qo_cup,
                       zo_cup, cdd, mentrd_rate, jmin, ierr, gammao_cup,
                       pwevo, bu, qrcdo, qo, 1, xl, high_resolution, ktf);
    g3_cup_up_moisture(false, ierr, z_cup, qc, qrc, pw, pwav, kbcon, ktop, cd,
                       dby, mentr_rate, clw_all, q2d, gamma_cup, zu, qes_cup,
                       k22, q_cup, xl, ktf);
    for (int k = 1; k <= ktf; ++k) cupclw(k) = qrc(k);
    g3_cup_up_moisture(false, ierr, zo_cup, qco, qrco, pwo, pwavo, kbcon, ktop,
                       cd, dbyo, mentr_rate, clw_all, qo, gammao_cup, zuo,
                       qeso_cup, k22, qo_cup, xl, ktf);
    aa0 = g3_cup_up_aa0(z, zu, dby, gamma_cup, t_cup, kbcon, ktop, ierr, ktf);
    aa1 = g3_cup_up_aa0(zo, zuo, dbyo, gammao_cup, tn_cup, kbcon, ktop, ierr, ktf);
    if (ierr == 0 && aa1 == 0.f) ierr = 17;

    // ---- shallow convection (:1223-1522) --------------------------------
    int k23 = 0, kbcon3 = 0, ktop3 = 0;   // declared divergence when ishallow=0
    if (P.ishallow == 1) {
        G3C(he3); G3C(hes3); G3C(qes3); G3C(z3); G3C(zu3_0); G3C(hc3_0);
        G3C(dby3_0); G3C(qes3_cup); G3C(q3_cup); G3C(he3_cup); G3C(hes3_cup);
        G3C(z3_cup); G3C(gamma3_cup); G3C(t3_cup); G3C(xhe3); G3C(xhes3);
        G3C(xqes3); G3C(xz3); G3C(xt3); G3C(xq3); G3C(xqes3_cup);
        G3C(xq3_cup); G3C(xhe3_cup); G3C(xhes3_cup); G3C(xz3_cup);
        G3C(xt3_cup); G3C(xdby3); G3C(xhc3); G3C(xzu3); G3C(dby3); G3C(qc3);
        G3C(pw3); G3C(hc3); G3C(qrc3); G3C(zu3); G3C(dellah3); G3C(dellaq3);
        G3C(dsubt3); G3C(dsubq3); G3C(dellat3); G3C(scratch_clw);
        g3_cup_env(z3, qes3, he3, hes3, tshall, qshall, po, z1, psur, ierr5,
                   tcrit, 0, xl, cp, ktf, tv);
        g3_cup_env_clev(tshall, qes3, qshall, he3, hes3, z3, po, qes3_cup,
                        q3_cup, he3_cup, hes3_cup, z3_cup, po_cup, gamma3_cup,
                        t3_cup, psur, ierr5, z1, xl, rv, cp, ktf);
        k23 = g3_maximi(he3_cup, 1, kbmax, ierr5);
        if (kpbli > 5) cap_max3 = po_cup(kpbli);
        if (ierr5 == 0) {
            if (k23 > kbmax) ierr5 = 2;
            if (kpbli > 5) k23 = kpbli;
        }
        g3_cup_kbcon(cap_max_increment, 5, k23, kbcon3, he3_cup, hes3_cup,
                     ierr5, kbmax, po_cup, cap_max3);
        float hkb3, hkb3_0;
        g3_cup_up_he(k23, hkb3, z3_cup, cd3, mentr_rate3, he3_cup, hc3,
                     kbcon3, ierr5, dby3, he3, hes3_cup, true, ktf);
        g3_cup_up_he(k23, hkb3_0, z_cup, cd3, mentr_rate3, he_cup, hc3_0,
                     kbcon3, ierr5, dby3_0, he, hes_cup, true, ktf);
        g3_cup_ktop(1, dby3, kbcon3, ktop3, ierr5, ktf);
        g3_cup_up_nms(zu3, z3_cup, mentr_rate3, cd3, kbcon3, ktop3, ierr5, k23, ktf);
        g3_cup_up_nms(zu3_0, z_cup, mentr_rate3, cd3, kbcon3, ktop3, ierr5, k23, ktf);
        aa3_0 = g3_cup_up_aa0(z, zu3_0, dby3_0, gamma3_cup, t_cup, kbcon3,
                              ktop3, ierr5, ktf);
        float pwav3;
        g3_cup_up_moisture(true, ierr5, z3_cup, qc3, qrc3, pw3, pwav3, kbcon3,
                           ktop3, cd3, dby3, mentr_rate3, scratch_clw, qshall,
                           gamma3_cup, zu3, qes3_cup, k23, q3_cup, xl, ktf);
        aa3 = g3_cup_up_aa0(z3, zu3, dby3, gamma3_cup, t3_cup, kbcon3, ktop3,
                            ierr5, ktf);
        g3_cup_dellas(ierr5, z3_cup, po_cup, hcdo, edt3, zdo3, cdd, he3,
                      dellah3, dsubt3, mentrd_rate, zu3, g, cd3, hc3, ktop3,
                      k23, kbcon3, mentr_rate3, jmin, he3_cup, kdet, k23,
                      true, 0, ktf);
        g3_cup_dellas(ierr5, z3_cup, po_cup, qrcdo, edt3, zdo3, cdd, qshall,
                      dellaq3, dsubq3, mentrd_rate, zu3, g, cd3, qc3, ktop3,
                      k23, kbcon3, mentr_rate3, jmin, q3_cup, kdet, k23,
                      true, 0, ktf);
        const float mbdt_s = 1.e-1f * mbdt_ens[0];
        for (int k = 1; k <= ktf; ++k) {
            dellat3(k) = 0.f;
            if (ierr5 == 0) {
                xhe3(k) = (dsubt3(k) + dellah3(k)) * mbdt_s + he3(k);
                xq3(k) = (dsubq3(k) + dellaq3(k)) * mbdt_s + qshall(k);
                dellat3(k) = (1.f / cp) * (dellah3(k) - xl * dellaq3(k));
                dsubt3(k) = (1.f / cp) * (dsubt3(k) - xl * dsubq3(k));
                xt3(k) = (dellat3(k) + dsubt3(k)) * mbdt_s + tshall(k);
                if (xq3(k) <= 0.f) xq3(k) = 1.E-08f;
            }
        }
        if (ierr5 == 0) {
            xhe3(ktf) = he3(ktf);
            xq3(ktf) = qshall(ktf);
            xt3(ktf) = tshall(ktf);
            if (xq3(ktf) <= 0.f) xq3(ktf) = 1.E-08f;
        }
        g3_cup_env(xz3, xqes3, xhe3, xhes3, xt3, xq3, po, z1, psur, ierr5,
                   tcrit, 2, xl, cp, ktf, tv);
        g3_cup_env_clev(xt3, xqes3, xq3, xhe3, xhes3, xz3, po, xqes3_cup,
                        xq3_cup, xhe3_cup, xhes3_cup, xz3_cup, po_cup,
                        gamma3_cup, xt3_cup, psur, ierr5, z1, xl, rv, cp, ktf);
        float xhkb3 = 0.f;
        if (ierr5 == 0) xhkb3 = xhe3(k23);
        g3_cup_up_he(k23, xhkb3, xz3_cup, cd3, mentr_rate3, xhe3_cup, xhc3,
                     kbcon3, ierr5, xdby3, xhe3, xhes3_cup, true, ktf);
        g3_cup_up_nms(xzu3, xz3_cup, mentr_rate3, cd3, kbcon3, ktop3, ierr5, k23, ktf);
        float xaa3 = g3_cup_up_aa0(xz3, xzu3, xdby3, gamma3_cup, xt3_cup,
                                   kbcon3, ktop3, ierr5, ktf);
        xmb3 = 0.f;
        float xff_shal[9];
        for (int m = 0; m < 9; ++m) xff_shal[m] = 0.f;
        if (ierr5 == 0) {
            float xkshal = (xaa3 - aa3) / mbdt_s;
            if (xkshal >= 0.f) xkshal = +1.e6f;
            if (xkshal > -1.e-4f && xkshal < 0.f) xkshal = -1.e-4f;
            xff_shal[0] = g3max(0.f, -(aa3 - aa3_0) / (xkshal * dtime));
            xff_shal[1] = g3max(0.f, -(aa3 - aa3_0) / (xkshal * dtime));
            xff_shal[2] = g3max(0.f, -(aa3 - aa3_0) / (xkshal * dtime));
            if (aa3_0 <= 0.f) { xff_shal[0] = 0.f; xff_shal[1] = 0.f; xff_shal[2] = 0.f; }
            if (aa3 - aa3_0 <= 0.f) { xff_shal[0] = 0.f; xff_shal[1] = 0.f; xff_shal[2] = 0.f; }
            float blqe = 0.f;
            float trash = 0.f;
            if (k23 < kpbli + 1) {
                for (int k = 1; k <= kbcon3 - 1; ++k)
                    blqe = blqe + 100.f * dhdt(k) * (p_cup(k) - p_cup(k + 1)) / g;
                trash = g3max((hc3(kbcon3) - he_cup(kbcon3)), 1.e1f);
                xff_shal[6] = g3max(0.f, blqe / trash);
                xff_shal[6] = g3min(0.1f, xff_shal[6]);
            } else {
                xff_shal[6] = 0.f;
            }
            if ((xkshal < -1.1e-04f) && ((aa3 - aa3_0 > 0.f) || (xff_shal[6] > 0.f))) {
                xff_shal[3] = g3max(0.f, -aa3 / (xkshal * tscl_kf));
                xff_shal[3] = g3min(0.1f, xff_shal[3]);
                xff_shal[4] = xff_shal[3];
                xff_shal[5] = xff_shal[3];
            } else {
                xff_shal[3] = 0.f; xff_shal[4] = 0.f; xff_shal[5] = 0.f;
            }
            xff_shal[7] = xff_shal[6];
            xff_shal[8] = xff_shal[6];
            for (int m = 0; m < 9; ++m) xmb3 = xmb3 + xff_shal[m];
            xmb3 = g3min(.1f, xmb3 / 9.f);
            if (xmb3 == 0.f) ierr5 = 22;
            if (xmb3 < 0.f) ierr5 = 21;
        }
        if (ierr5 != 0) {
            k23 = 0; kbcon3 = 0; ktop3 = 0; xmb3 = 0.f;
            for (int k = 1; k <= ktf; ++k) { outts(k) = 0.f; outqs(k) = 0.f; }
        } else {
            float trash = 0.f;
            for (int k = 2; k <= ktop3; ++k)
                trash = g3max(trash, 86400.f * (dsubt3(k) + dellat3(k)) * xmb3);
            if (trash > 150.f) xmb3 = xmb3 * 150.f / trash;
            for (int k = 2; k <= ktop3; ++k) {
                trash = q2d(k) + (dsubq3(k) + dellaq3(k)) * xmb3 * dtime;
                if (trash < 1.e-12f) {
                    trash = ((1.e-12f - q2d(k)) / dtime)
                            / ((dsubq3(k) + dellaq3(k)) * xmb3);
                    trash = g3max(0.f, trash);
                    trash = g3min(1.f, trash);
                    xmb3 = trash * xmb3;
                }
            }
            for (int k = 2; k <= ktop3; ++k) {
                outts(k) = (dsubt3(k) + dellat3(k)) * xmb3;
                outqs(k) = (dsubq3(k) + dellaq3(k)) * xmb3;
            }
        }
    }

    // ---- cup_axx (:1526-1529) -------------------------------------------
    float axx[G3_ENS4];
    {
        G3C(ax_he); G3C(ax_hes); G3C(ax_qes); G3C(ax_z); G3C(ax_qes_cup);
        G3C(ax_q_cup); G3C(ax_he_cup); G3C(ax_hes_cup); G3C(ax_z_cup);
        G3C(ax_p_cup); G3C(ax_gamma_cup); G3C(ax_t_cup); G3C(ax_dby);
        G3C(ax_hc); G3C(ax_zu); G3C(ax_cd); G3C(ax_tv);
        g3_cup_axx(tcrit, kbmax, z1, po, psur, xl, rv, cp, ierr,
                   cap_max, cap_max_increment, entr_rate, mentr_rate, axx, tx,
                   qx, ax_he, ax_hes, ax_qes, ax_z, ax_qes_cup, ax_q_cup,
                   ax_he_cup, ax_hes_cup, ax_z_cup, ax_p_cup, ax_gamma_cup,
                   ax_t_cup, ax_dby, ax_hc, ax_zu, ax_cd, ax_tv, ktf);
    }

    // ---- cup_dd_edt and the iedt ensemble (:1534-1843) -------------------
    float edtc[G3_MAXENS2];
    g3_cup_dd_edt(ierr, us, vs, zo, ktop, kbcon, po, pwavo, pwevo, edtmax,
                  edtmin, edtc, ktf);
    float edt_out = 0.f;
    float xaa0_ens[G3_MAXENS];
    for (int iedt = 1; iedt <= G3_MAXENS2; ++iedt) {
        if (ierr == 0) {
            edt = edtc[iedt - 1];
            edto = edtc[iedt - 1];
            edt_out = edtc[1];
            if (high_resolution == 1) {
                edt = edtc[2];
                edto = edtc[2];
                edt_out = edtc[2];
            }
        }
        G3Col subt_e = slot(G3S_subt_ens + iedt - 1);
        G3Col subq_e = slot(G3S_subq_ens + iedt - 1);
        G3Col dellat_e = slot(G3S_dellat_ens + iedt - 1);
        G3Col dellaq_e = slot(G3S_dellaq_ens + iedt - 1);
        G3Col dellaqc_e = slot(G3S_dellaqc_ens + iedt - 1);
        G3Col pwo_e = slot(G3S_pwo_ens + iedt - 1);
        for (int k = 1; k <= ktf; ++k) {
            subt_e(k) = 0.f; subq_e(k) = 0.f; dellat_e(k) = 0.f;
            dellaq_e(k) = 0.f; dellaqc_e(k) = 0.f; pwo_e(k) = 0.f;
        }
        g3_cup_dellabot(heo_cup, ierr, zo_cup, po, hcdo, edto, zdo, cdd, heo,
                        dellah, dsubt, mentrd_rate, g);
        g3_cup_dellabot(qo_cup, ierr, zo_cup, po, qrcdo, edto, zdo, cdd, qo,
                        dellaq, dsubq, mentrd_rate, g);
        g3_cup_dellas(ierr, zo_cup, po_cup, hcdo, edto, zdo, cdd, heo, dellah,
                      dsubt, mentrd_rate, zuo, g, cd, hco, ktop, k22, kbcon,
                      mentr_rate, jmin, heo_cup, kdet, k22, false,
                      high_resolution, ktf);
        for (int k = 1; k <= ktf - 1; ++k) {
            dellaqc(k) = 0.f;
            if (ierr == 0) {
                if (k == ktop - 0)
                    dellaqc(k) = .01f * zuo(ktop) * qrco(ktop) * 9.81f
                                 / (po_cup(k) - po_cup(k + 1));
                if (k < ktop && k > kbcon) {
                    float dz = zo_cup(k + 1) - zo_cup(k);
                    dellaqc(k) = .01f * 9.81f * cd(k) * dz * zuo(k)
                                 * .5f * (qrco(k) + qrco(k + 1))
                                 / (po_cup(k) - po_cup(k + 1));
                }
            }
        }
        g3_cup_dellas(ierr, zo_cup, po_cup, qrcdo, edto, zdo, cdd, qo, dellaq,
                      dsubq, mentrd_rate, zuo, g, cd, qco, ktop, k22, kbcon,
                      mentr_rate, jmin, qo_cup, kdet, k22, false,
                      high_resolution, ktf);
        const float mbdt = mbdt_ens[1];
        xaa0_ens[0] = 0.f; xaa0_ens[1] = 0.f; xaa0_ens[2] = 0.f;
        for (int k = 1; k <= ktf; ++k) {
            dellat(k) = 0.f;
            if (ierr == 0) {
                xhe(k) = (dsubt(k) + dellah(k)) * mbdt + heo(k);
                xq(k) = (dsubq(k) + dellaq(k)) * mbdt + qo(k);
                dellat(k) = (1.f / cp) * (dellah(k) - xl * dellaq(k));
                dsubt(k) = (1.f / cp) * (dsubt(k) - xl * dsubq(k));
                xt(k) = (dellat(k) + dsubt(k)) * mbdt + tn(k);
                if (xq(k) <= 0.f) xq(k) = 1.E-08f;
            }
        }
        if (ierr == 0) {
            xhe(ktf) = heo(ktf);
            xq(ktf) = qo(ktf);
            xt(ktf) = tn(ktf);
            if (xq(ktf) <= 0.f) xq(ktf) = 1.E-08f;
        }
        g3_cup_env(xz, xqes, xhe, xhes, xt, xq, po, z1, psur, ierr, tcrit, 2,
                   xl, cp, ktf, tv);
        g3_cup_env_clev(xt, xqes, xq, xhe, xhes, xz, po, xqes_cup, xq_cup,
                        xhe_cup, xhes_cup, xz_cup, po_cup, gamma_cup, xt_cup,
                        psur, ierr, z1, xl, rv, cp, ktf);
        float xhkb = 0.f;
        if (ierr == 0) xhkb = xhe(k22);
        g3_cup_up_he(k22, xhkb, xz_cup, cd, mentr_rate, xhe_cup, xhc, kbcon,
                     ierr, xdby, xhe, xhes_cup, false, ktf);
        g3_cup_up_nms(xzu, xz_cup, mentr_rate, cd, kbcon, ktop, ierr, k22, ktf);
        g3_cup_dd_nms(xzd, xz_cup, cdd, mentrd_rate, jmin, ierr, 1, kdet, z1, ktf);
        g3_cup_dd_he(xhes_cup, xzd, xhcd, xz_cup, cdd, mentrd_rate, jmin, ierr,
                     xhe, dbyd, ktf);
        float xpwev, xbu, xpwav;
        g3_cup_dd_moisture(xzd, xhcd, xhes_cup, xqcd, xqes_cup, xpwd, xq_cup,
                           xz_cup, cdd, mentrd_rate, jmin, ierr, gamma_cup,
                           xpwev, xbu, xqrcd, xq, 3, xl, high_resolution, ktf);
        g3_cup_up_moisture(false, ierr, xz_cup, xqc, xqrc, xpw, xpwav, kbcon,
                           ktop, cd, xdby, mentr_rate, clw_all, xq, gamma_cup,
                           xzu, xqes_cup, k22, xq_cup, xl, ktf);
        float xaa0 = g3_cup_up_aa0(xz, xzu, xdby, gamma_cup, xt_cup, kbcon,
                                   ktop, ierr, ktf);
        for (int nens = 1; nens <= G3_MAXENS; ++nens) {
            if (ierr == 0) {
                xaa0_ens[nens - 1] = xaa0;
                int nall = (iens - 1) * G3_MAXENS3 * G3_MAXENS * G3_MAXENS2
                           + (iedt - 1) * G3_MAXENS * G3_MAXENS3
                           + (nens - 1) * G3_MAXENS3;
                for (int k = 1; k <= ktf; ++k) {
                    if (k <= ktop) {
                        for (int nens3 = 1; nens3 <= G3_MAXENS3; ++nens3) {
                            float& pr = prc[(nall + nens3 - 1) * es];
                            if (nens3 == 7) {
                                pr = pr + edto * pwdo(k) + pwo(k);
                            } else if (nens3 == 8) {
                                pr = pr + pwo(k);
                            } else if (nens3 == 9) {
                                pr = pr + .5f * edto * pwdo(k) + pwo(k);
                            } else {
                                pr = pr + pwo(k) + edto * pwdo(k);
                            }
                        }
                    }
                }
                if (prc[(nall + 7 - 1) * es] < 1.e-6f) {
                    ierr = 18;
                    for (int nens3 = 1; nens3 <= G3_MAXENS3; ++nens3)
                        prc[(nall + nens3 - 1) * es] = 0.f;
                }
                for (int nens3 = 1; nens3 <= G3_MAXENS3; ++nens3) {
                    if (prc[(nall + nens3 - 1) * es] < 1.e-4f)
                        prc[(nall + nens3 - 1) * es] = 0.f;
                }
            }
        }
        // ---- check whether aa0 should have been zero (:1792-1806) ---------
        int k22x = g3_maximi(heo_cup, 3, kbmax, ierr);
        ierr2 = ierr;
        ierr3 = ierr;
        int kbconx;
        g3_cup_kbcon(cap_max_increment, 2, k22x, kbconx, heo_cup, heso_cup,
                     ierr2, kbmax, po_cup, cap_max);
        g3_cup_kbcon(cap_max_increment, 3, k22x, kbconx, heo_cup, heso_cup,
                     ierr3, kbmax, po_cup, cap_max);

        // ---- cup_forcing_ens_3d, 'deeps' (:3021-3602) ----------------------
        {
            const float fens4 = (float)G3_ENS4;
            if (ierr > 995) { aa0 = 0.f; ierr = 0; }
            if (ierr == 0) {
                float a_ave = 0.f;
                for (int ne = 0; ne < G3_ENS4; ++ne) a_ave = a_ave + axx[ne];
                a_ave = g3max(0.f, a_ave / fens4);
                a_ave = g3min(a_ave, aa1);
                a_ave = g3max(0.f, a_ave);
                float xff[G3_MAXENS3];
                for (int ne = 0; ne < 16; ++ne) xff[ne] = 0.f;
                float xff0 = (aa1 - aa0) / dtime;
                if (high_resolution == 1) xff0 = (a_ave - aa0) / dtime;
                xff[0] = (aa1 - aa0) / dtime;
                xff[1] = (a_ave - aa0) / dtime;
                xff[2] = (aa1 - aa0) / dtime;
                xff[12] = (aa1 - aa0) / dtime;
                if (high_resolution == 1) {
                    xff[0] = (a_ave - aa0) / dtime;
                    xff[1] = (a_ave - aa0) / dtime;
                    xff[2] = (a_ave - aa0) / dtime;
                    xff[12] = (a_ave - aa0) / dtime;
                }
                xff[13] = 0.f;
                for (int ne = 0; ne < G3_ENS4; ++ne)
                    xff[13] = xff[13] - omeg[ne](k22) / (fens4 * 9.81f);
                if (xff[13] < 0.f) xff[13] = 0.f;
                xff[4] = 0.f;
                for (int ne = 0; ne < G3_ENS4; ++ne)
                    xff[4] = xff[4] - omeg[ne](kbcon) / (fens4 * 9.81f);
                if (xff[4] < 0.f) xff[4] = 0.f;
                if (high_resolution == 0) {
                    xff[3] = -omeg[0](2) / 9.81f;
                    for (int k = 2; k <= kbcon - 1; ++k)
                        for (int ne = 0; ne < G3_ENS4; ++ne) {
                            float xomg = -omeg[ne](k) / 9.81f;
                            if (xomg < xff[3]) xff[3] = xomg;
                        }
                    if (xff[3] < 0.f) xff[3] = 0.f;
                    xff[5] = -omeg[0](2) / 9.81f;
                    for (int k = 2; k <= kbcon - 1; ++k)
                        for (int ne = 0; ne < G3_ENS4; ++ne) {
                            float xomg = -omeg[ne](k) / 9.81f;
                            if (xomg > xff[5]) xff[5] = xomg;
                        }
                    if (xff[5] < 0.f) xff[5] = 0.f;
                }
                if (high_resolution == 1) {
                    xff[4] = g3min(xff[4], xff[13]);
                    xff[3] = xff[4];
                    xff[5] = xff[4];
                }
                xff[6] = mconv[0];
                xff[7] = mconv[0];
                xff[8] = mconv[0];
                for (int ne = 1; ne < G3_ENS4; ++ne)
                    if (mconv[ne] > xff[6]) xff[6] = mconv[ne];
                for (int ne = 1; ne < G3_ENS4; ++ne)
                    if (mconv[ne] < xff[7]) xff[7] = mconv[ne];
                for (int ne = 1; ne < G3_ENS4; ++ne) xff[8] = xff[8] + mconv[ne];
                xff[8] = xff[8] / fens4;
                if (high_resolution == 1) {
                    xff[6] = xff[8];
                    xff[7] = xff[8];
                    xff[14] = xff[8];
                }
                if (high_resolution == 0) xff[14] = mconv[0];
                xff[9] = a_ave / (60.f * 40.f);
                xff[10] = aa1 / (60.f * 40.f);
                xff[11] = aa1 / (60.f * 40.f);
                if (high_resolution == 1) {
                    xff[10] = xff[9];
                    xff[11] = xff[9];
                }
                if (P.ichoice == 0) {
                    if (xff0 < 0.f) {
                        xff[0] = 0.f; xff[1] = 0.f; xff[2] = 0.f; xff[12] = 0.f;
                        xff[9] = 0.f; xff[10] = 0.f; xff[11] = 0.f;
                    }
                }
                float xk[G3_MAXENS];
                for (int nens = 0; nens < G3_MAXENS; ++nens) {
                    xk[nens] = (xaa0_ens[nens] - aa1) / mbdt_ens[1];
                    if (xk[nens] <= 0.f && xk[nens] > -1.e-6f) xk[nens] = -1.e-6f;
                    if (xk[nens] > 0.f && xk[nens] < 1.e-6f) xk[nens] = 1.e-6f;
                }
                for (int ne = 1; ne <= G3_MAXENS; ++ne) {
                    int nall = (iens - 1) * G3_MAXENS3 * G3_MAXENS * G3_MAXENS2
                               + (iedt - 1) * G3_MAXENS * G3_MAXENS3
                               + (ne - 1) * G3_MAXENS3;
                    auto XF = [&](int m) -> float& { return xfc[(nall + m - 1) * es]; };
                    auto PR = [&](int m) -> float { return prc[(nall + m - 1) * es]; };
                    if (xland1 < 0.1f) {
                        if (ierr2 > 0 || ierr3 > 0) {
                            xff[0] = 0.f; xff[1] = 0.f; xff[2] = 0.f;
                            xff[9] = 0.f; xff[10] = 0.f; xff[11] = 0.f;
                            xff[6] = 0.f; xff[7] = 0.f; xff[8] = 0.f;
                            xff[12] = 0.f; xff[14] = 0.f;
                        }
                    }
                    const float massfld = 0.f;
                    const float xkn = xk[ne - 1];
                    if (xff0 >= 0.f) {
                        XF(1) = massfld; XF(2) = massfld; XF(3) = massfld; XF(13) = massfld;
                        if (xff[0] > 0.f) XF(1) = g3max(0.f, -xff[0] / xkn) + massfld;
                        if (xff[1] > 0.f) XF(2) = g3max(0.f, -xff[1] / xkn) + massfld;
                        if (xff[2] > 0.f) XF(3) = g3max(0.f, -xff[2] / xkn) + massfld;
                        if (xff[12] > 0.f) XF(13) = g3max(0.f, -xff[12] / xkn) + massfld;
                    } else {
                        XF(1) = massfld; XF(2) = massfld; XF(3) = massfld; XF(13) = massfld;
                    }
                    XF(4) = g3max(0.f, xff[3] + massfld);
                    XF(5) = g3max(0.f, xff[4] + massfld);
                    XF(6) = g3max(0.f, xff[5] + massfld);
                    XF(14) = g3max(0.f, xff[13] + massfld);
                    float a1 = g3max(1.e-3f, PR(7));
                    XF(7) = g3max(0.f, xff[6] / a1);
                    a1 = g3max(1.e-3f, PR(8));
                    XF(8) = g3max(0.f, xff[7] / a1);
                    a1 = g3max(1.e-3f, PR(9));
                    XF(9) = g3max(0.f, xff[8] / a1);
                    a1 = g3max(1.e-3f, PR(15));
                    XF(15) = g3max(0.f, xff[14] / a1);
                    if (xkn < 0.f) {
                        XF(10) = g3max(0.f, -xff[9] / xkn) + massfld;
                        XF(11) = g3max(0.f, -xff[10] / xkn) + massfld;
                        XF(12) = g3max(0.f, -xff[11] / xkn) + massfld;
                    } else {
                        XF(10) = massfld; XF(11) = massfld; XF(12) = massfld;
                    }
                    // Keep WRF's ordering: member 16 is filled after the selector copy.
                    if (P.ichoice >= 1) {
                        closure_n = 0.f;
                        for (int m = 1; m <= 16; ++m) XF(m) = XF(P.ichoice);
                    }
                    XF(16) = XF(1);
                    if (ne == 2 && ierr2 > 0)
                        for (int m = 1; m <= 16; ++m) XF(m) = 0.f;
                    if (ne == 3 && ierr3 > 0)
                        for (int m = 1; m <= 16; ++m) XF(m) = 0.f;
                }
                int nall = (iens - 1) * G3_MAXENS3 * G3_MAXENS * G3_MAXENS2
                           + (iedt - 1) * G3_MAXENS * G3_MAXENS3;
                int nall2 = nall + (2 - 1) * G3_MAXENS3;
                const int copy[11] = {4, 5, 6, 14, 7, 8, 9, 15, 10, 11, 12};
                for (int c = 0; c < 11; ++c)
                    xfc[(nall + copy[c] - 1) * es] = xfc[(nall2 + copy[c] - 1) * es];
            } else if (ierr != 20 && ierr != 0) {
                for (int n = 0; n < G3_ENSDIM; ++n) xfc[n * es] = 0.f;
            }
        }
        for (int k = 1; k <= ktf; ++k) {
            if (ierr == 0) {
                subt_e(k) = dsubt(k);
                subq_e(k) = dsubq(k);
                dellat_e(k) = dellat(k);
                dellaq_e(k) = dellaq(k);
                dellaqc_e(k) = dellaqc(k);
                pwo_e(k) = pwo(k) + edt * pwdo(k);
            } else {
                subt_e(k) = 0.f; subq_e(k) = 0.f; dellat_e(k) = 0.f;
                dellaq_e(k) = 0.f; dellaqc_e(k) = 0.f; pwo_e(k) = 0.f;
            }
        }
    }

    // ---- cup_output_ens_3d (:3866-4076) ---------------------------------
    float pre = 0.f, xmb = 0.f, xfac1 = 0.f;
    for (int k = 1; k <= ktf; ++k) {
        outt(k) = 0.f; outq(k) = 0.f; outqc(k) = 0.f; subt(k) = 0.f; subq(k) = 0.f;
    }
    if (ierr == 0) {
        for (int n = 1; n <= G3_ENSDIM; ++n)
            if (prc[(n - 1) * es] <= 0.f) xfc[(n - 1) * es] = 0.f;
    }
    float prv[8];
    float xmb_ave, xmb_std, pr_ave, pr_std;
    g3_massflx_stats(xfc, es, ierr, 1, xmb_ave, xmb_std, aprc, prv);
    g3_massflx_stats(prc, es, ierr, 2, pr_ave, pr_std, aprc, prv);
    if (ierr == 0) {
        if (xmb_ave <= 0.f) { ierr = 13; xmb_ave = 0.f; }
        const float tuning = 0.f;
        xmb = g3max(.1f * xmb_ave, xmb_ave - tuning * xmb_std);
        float clos_wei = 16.f / g3max(1.f, closure_n);
        if (xland1 < 0.5f) xmb = xmb * clos_wei;
        if (xmb == 0.f) ierr = 19;
        if (xmb > 100.f) ierr = 19;
        xfac1 = xmb;
    }
    for (int k = 1; k <= ktf; ++k) {
        float dtt = 0.f, dtts = 0.f, dtq = 0.f, dtqs = 0.f, dtqc = 0.f, dtpw = 0.f;
        if (ierr == 0 && k <= ktop) {
            for (int n = 0; n < G3_MAXENS2; ++n) {
                dtt = dtt + slot(G3S_dellat_ens + n)(k);
                dtts = dtts + slot(G3S_subt_ens + n)(k);
                dtq = dtq + slot(G3S_dellaq_ens + n)(k);
                dtqs = dtqs + slot(G3S_subq_ens + n)(k);
                dtqc = dtqc + slot(G3S_dellaqc_ens + n)(k);
                dtpw = dtpw + slot(G3S_pwo_ens + n)(k);
            }
            const float fnx = (float)G3_MAXENS2;
            outt(k) = xmb * dtt / fnx;
            subt(k) = xmb * dtts / fnx;
            outq(k) = xmb * dtq / fnx;
            subq(k) = xmb * dtqs / fnx;
            outqc(k) = xmb * dtqc / fnx;
            pre = pre + xmb * dtpw / fnx;
        }
    }
    if (ierr == 0) {
        for (int k = 1; k <= G3_ENSDIM; ++k)
            xfc[(k - 1) * es] = xfc[(k - 1) * es] * xfac1;
    }
    if (ierr == 0 && ierr5 == 0 && kbcon < ktop3 + 1) {
        if (high_resolution == 1) {
            for (int k = 1; k <= ktf; ++k) { outts(k) = 0.f; outqs(k) = 0.f; }
        }
    }
    // MAX(PRE,0.): a variable-against-literal MAX keeps the variable on a tie
    pre = (0.f > pre) ? 0.f : pre;

    // ======================================================================
    // back in G3DRV: the written band (:485-572)
    // ======================================================================
    const float pret = pre;
    sfc_out[G3O_XMB_SHALLOW * plane + cix] = xmb3;
    isfc_out[G3O_K22S * plane + cix] = k23;
    isfc_out[G3O_KBCONS * plane + cix] = kbcon3;
    isfc_out[G3O_KTOPS * plane + cix] = ktop3;
    isfc_out[G3O_KTOP_DEEP * plane + cix] = ktop;
    float cuten = 0.f;
    if (pret > 0.f) cuten = 1.f;
    const bool use_qi = P.fqi != 0;
    for (int k = 1; k <= ktf; ++k) {
        long long o = (long long)(k - 1) * plane + cix;
        long long fs = (long long)nz * plane;
        float qcten = outqc(k) * cuten;
        lev_out[G3O_CUGD_TTENS * fs + o] = subt(k) * cuten * sub_spread;
        lev_out[G3O_CUGD_QVTENS * fs + o] = subq(k) * cuten * sub_spread;
        lev_out[G3O_CUGD_TTEN * fs + o] = outts(k) + outt(k) * cuten;
        lev_out[G3O_CUGD_QVTEN * fs + o] = outqs(k) + outq(k) * cuten;
        float rqc = qcten, rqi = 0.f;
        float gdc = cupclw(k) * cuten, gdc2 = 0.f;
        if (use_qi) {
            if (t2d(k) < 258.f) {
                rqi = qcten;
                qcten = 0.f;
                rqc = 0.f;
                gdc2 = cupclw(k) * cuten;
            } else {
                rqi = 0.f;
                rqc = outqc(k) * cuten;
                gdc = cupclw(k) * cuten;
            }
        }
        lev_out[G3O_CUGD_QCTEN * fs + o] = qcten;
        lev_out[G3O_RQCCUTEN * fs + o] = rqc;
        lev_out[G3O_RQICUTEN * fs + o] = rqi;
        lev_out[G3O_GDC * fs + o] = gdc;
        lev_out[G3O_GDC2 * fs + o] = gdc2;
    }
    if (pret > 0.f) {
        sfc_out[G3O_RAINCV * plane + cix] = pret * dt;
        sfc_out[G3O_PRATEC * plane + cix] = pret;
        float htop = sfc_out[G3O_HTOP * plane + cix];
        float hbot = sfc_out[G3O_HBOT * plane + cix];
        if ((float)ktop > htop) htop = (float)ktop + .001f;
        if ((float)kbcon < hbot) hbot = (float)kbcon + .001f;
        sfc_out[G3O_HTOP * plane + cix] = htop;
        sfc_out[G3O_HBOT * plane + cix] = hbot;
    }
    for (int m = 0; m < 8; ++m) apr[m * plane + cix] = aprc[m];
    sfc_out[G3O_MASS_FLUX * plane + cix] = 0.f;
    sfc_out[G3O_EDT_OUT * plane + cix] = edt_out;
#undef G3C
}

// ==========================================================================
// conv_grell_spread3d (:5037-5322), one thread per written band column.
// cugd_* are the g3_column outputs; rqccuten/rqicuten/raincv/pratec are
// read and limited in place; rthcuten/rqvcuten are written for the band
// (the host zeroes them elsewhere, :5108-5115).
// ==========================================================================
extern "C" __global__ void g3_spread(
    const float* __restrict__ cugd,      // [G3O_NLEV][nz][ny][nx] (g3_column's lev_out)
    const float* __restrict__ pi_phy,    // [nz][ny][nx]
    const float* __restrict__ qv,        // [nz][ny][nx]
    float* __restrict__ rthcuten,        // [nz][ny][nx]
    float* __restrict__ rqvcuten,        // [nz][ny][nx]
    float* __restrict__ rqccuten,        // [nz][ny][nx] in/out
    float* __restrict__ rqicuten,        // [nz][ny][nx] in/out
    float* __restrict__ raincv,          // [ny][nx] in/out
    float* __restrict__ pratec,          // [ny][nx] in/out
    float* __restrict__ ws,              // [2][nz+1][nt]
    int nt, int col0, int nx, int ny, int nz_, int ibegc, int iendc,
    int jbegc, int jendc, int avedx, int fqi, float dt_)
{
    G3Params P;
    P.nx = nx; P.ny = ny; P.nz = nz_; P.ibegc = ibegc; P.iendc = iendc;
    P.jbegc = jbegc; P.jendc = jendc; P.ishallow = 0; P.avedx = avedx;
    P.ichoice = 0; P.fqi = fqi; P.dt = dt_; P.dx = 0.f; P.xlv = 0.f;
    P.cp = 0.f; P.g = 0.f; P.rv = 0.f;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nt) return;
    // the written band of the smoother/limiters: ids+4..ide-5 (:5211-5229)
    const int nbx = P.iendc - P.ibegc + 1;
    const int nby = P.jendc - P.jbegc + 1;
    const int b = col0 + tid;
    if (nbx <= 0 || nby <= 0 || b >= nbx * nby) return;
    const int i = P.ibegc + b % nbx;
    const int j = P.jbegc + b / nbx;
    const int nz = P.nz;
    const long long plane = (long long)P.nx * P.ny;
    const long long fs = (long long)nz * plane;
    const float* tten = cugd + G3O_CUGD_TTEN * fs;
    const float* qvten = cugd + G3O_CUGD_QVTEN * fs;
    const float* ttens = cugd + G3O_CUGD_TTENS * fs;
    const float* qvtens = cugd + G3O_CUGD_QVTENS * fs;
    const int cugd_spread = P.avedx / 2;
    const float dt = P.dt;
    G3Col rt, rq;
    rt.p = ws + tid; rt.s = nt;
    rq.p = ws + (long long)(nz + 1) * nt + tid; rq.s = nt;

    auto at = [&](const float* f, int k, int ii, int jj) -> float {
        return f[(long long)(k - 1) * plane + (long long)jj * P.nx + ii];
    };
    // RTHcutent(i',k,j') for a neighbour (:5142-5180); i',j' are inside
    // the domain for every band column, and the spreading clamps to it.
    auto rcut = [&](const float* ten, const float* tens, int k, int ii, int jj) -> float {
        float r = at(ten, k, ii, jj);
        if (cugd_spread > 0) {
            for (int nn = -1; nn <= 1; ++nn) {
                int jdo = g3imin(g3imax(jj + nn, 0), P.ny - 1);
                for (int kk = -1; kk <= 1; ++kk) {
                    int ido = g3imin(g3imax(ii + kk, 0), P.nx - 1);
                    r = r + 1.f * at(tens, k, ido, jdo);
                }
            }
        }
        if (cugd_spread == 0) r = r + at(tens, k, ii, jj);
        return r;
    };
    for (int k = 1; k <= nz; ++k) {
        float smt[3], smq[3];
        for (int dj = -1; dj <= 1; ++dj) {
            int jj = j + dj;
            smt[dj + 1] = .25f * (rcut(tten, ttens, k, i - 1, jj)
                                  + 2.f * rcut(tten, ttens, k, i, jj)
                                  + rcut(tten, ttens, k, i + 1, jj));
            smq[dj + 1] = .25f * (rcut(qvten, qvtens, k, i - 1, jj)
                                  + 2.f * rcut(qvten, qvtens, k, i, jj)
                                  + rcut(qvten, qvtens, k, i + 1, jj));
        }
        rt(k) = .25f * (smt[0] + 2.f * smt[1] + smt[2]);
        rq(k) = .25f * (smq[0] + 2.f * smq[1] + smq[2]);
    }
    const long long cix = (long long)j * P.nx + i;
    float* rqc = rqccuten + cix;
    float* rqi = rqicuten + cix;
    // moistening limiter (:5232-5265)
    float qmemf = 1.f;
    float thresh = 1.e-20f;
    for (int k = 1; k <= nz; ++k) {
        if (rq(k) < 0.f) {
            float qv_k = qv[(long long)(k - 1) * plane + cix];
            float qmem1 = qv_k + rq(k) * dt;
            if (qmem1 < thresh) {
                qmem1 = rq(k);
                float qmem2 = (thresh - qv_k) / dt;
                qmemf = g3min(qmemf, qmem2 / qmem1);
                qmemf = g3max(0.f, qmemf);
                qmemf = g3min(1.f, qmemf);
            }
        }
    }
    for (int k = 1; k <= nz; ++k) {
        rq(k) = rq(k) * qmemf;
        rt(k) = rt(k) * qmemf;
    }
    for (int k = 1; k <= nz; ++k) rqc[(k - 1) * plane] = rqc[(k - 1) * plane] * qmemf;
    if (P.fqi) {
        for (int k = 1; k <= nz; ++k) rqi[(k - 1) * plane] = rqi[(k - 1) * plane] * qmemf;
    }
    raincv[cix] = raincv[cix] * qmemf;
    pratec[cix] = pratec[cix] * qmemf;
    // heating limiter (:5270-5302)
    thresh = 200.f;
    qmemf = 1.f;
    float qmem1 = 0.f;
    for (int k = 1; k <= nz; ++k) {
        qmem1 = fabsf(rt(k)) * 86400.f;
        if (qmem1 > thresh) {
            float qmem2 = thresh / qmem1;
            qmemf = g3min(qmemf, qmem2);
            qmemf = g3max(0.f, qmemf);
        }
    }
    raincv[cix] = raincv[cix] * qmemf;
    pratec[cix] = pratec[cix] * qmemf;
    for (int k = 1; k <= nz; ++k) {
        rq(k) = rq(k) * qmemf;
        rt(k) = rt(k) * qmemf;
    }
    for (int k = 1; k <= nz; ++k) rqc[(k - 1) * plane] = rqc[(k - 1) * plane] * qmemf;
    if (P.fqi) {
        for (int k = 1; k <= nz; ++k) rqi[(k - 1) * plane] = rqi[(k - 1) * plane] * qmemf;
    }
    // vertical 1-2-1 smoothing, k = 3..nz-2 (:5303-5315), then /pi
    {
        float pt_km1 = rt(2), pq_km1 = rq(2);
        // conv_TRASHT is formed from the unsmoothed column, then copied.
        for (int k = 3; k <= nz - 2; ++k) {
            float t_new = .25f * (pt_km1 + 2.f * rt(k) + rt(k + 1));
            float q_new = .25f * (pq_km1 + 2.f * rq(k) + rq(k + 1));
            pt_km1 = rt(k);
            pq_km1 = rq(k);
            rt(k) = t_new;
            rq(k) = q_new;
        }
    }
    for (int k = 1; k <= nz; ++k) {
        long long o = (long long)(k - 1) * plane + cix;
        rthcuten[o] = rt(k) / pi_phy[o];
        rqvcuten[o] = rq(k);
    }
}

// WRF v4.6.1 dyn_em/module_convtrans_prep.F:50-205.
// The counter belongs to this domain. WRF's procedure-local SAVE counter
// otherwise mixes domains and tiles. It is carried by the restart inventory.
extern "C" __global__ void g3_diag_clock(int* clock, float dt, float avg,
                                         float curr, int adaptive) {
    int ktau = clock[0] + 1;
    int count = ktau <= 1 ? 0 : clock[1];
    float period = avg * 60.f;
    int stepave = (int)(period / dt);
    bool end, first;
    if (adaptive) {
        end = (double)curr + (double)dt + (double).01f >=
              (double)((long long)((double)curr / (double)period + 1LL)) * (double)period;
        first = (double)curr <= (double)period;
    } else {
        end = ktau % stepave == 0;
        first = ktau <= stepave;
    }
    clock[0] = ktau; clock[1] = count + 1;
    clock[2] = first; clock[3] = end;
}

extern "C" __global__ void g3_diag(
    const float* cloud, const float* ice, const float* rain,
    const float* t, const float* p, const float* qv, const float* qc,
    const float* qi, float* ca, float* cb, float* ia, float* ib,
    float* frac, float* ra, float* rb, const int* clock,
    int fqi, long long nvol, long long ncol) {
    long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    bool first = clock[2] != 0, end = clock[3] != 0;
    float count = (float)clock[1];
    if (c < ncol) {
        if (clock[0] <= 1) { ra[c] = 0.f; rb[c] = 0.f; }
        if (first) rb[c] = rain[c];
        ra[c] = ra[c] + rain[c];
        if (end) { rb[c] = ra[c] / count; ra[c] = 0.f; }
    }
    if (c >= nvol) return;
    if (clock[0] <= 1) { ia[c] = 0.f; frac[c] = 0.f; }
    if (first) { cb[c] = cloud[c]; ib[c] = ice[c]; }
    ca[c] = ca[c] + cloud[c];
    ia[c] = ia[c] + ice[c];
    if (end) {
        cb[c] = .1f * ca[c] / count; ca[c] = 0.f;
        ib[c] = .1f * ia[c] / count; ia[c] = 0.f;
    }
    if (first || end) {
        float satvp = 3.80f * gfk_exp(17.27f * (t[c] - 273.f) / (t[c] - 36.f)) / (.01f * p[c]);
        float rhgrid = g3max(.1f, g3min(.95f, qv[c] / satvp));
        float h2oliq = fqi ? 1000.f * (cb[c] + qc[c] + ib[c] + qi[c])
                           : 1000.f * (cb[c] + qc[c]);
        satvp = 1000.f * satvp;
        float value = (1.f - gfk_exp(-100.f * h2oliq /
                        gfk_pow((1.f - rhgrid) * satvp, .49f))) * gfk_pow(rhgrid, .25f);
        value = g3max(0.f, g3min(1.f, value));
        if (qc[c] == 0.f && (!fqi || qi[c] == 0.f)) value = value * .2f;
        frac[c] = value;
    }
}
extern "C" __global__ void g3_diag_finish(int* clock) {
    if (clock[3]) clock[1] = 0;
}
