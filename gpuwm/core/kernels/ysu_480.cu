// YSU planetary-boundary-layer column scheme.
//
// Transcribed from WRF v4.6.1 phys/module_bl_ysu.F and the scheme body in
// phys/physics_mmm/bl_ysu.F90 (bl_ysu_run), for the non-BEP,
// non-BEP path, including v4.6.1 ysu_topdown_pblmix.  One CUDA thread owns a
// complete surface-to-top column.
// All implicit heat/moisture/momentum systems use the same in-thread Thomas
// pattern as kernels/acoustic.cu.  Float64 verification authority:
// gpuwm.verify.npref.np_ysu_column.

#define YSU_KMAX 128

// ==========================================================================
// Per-thread column workspace
// ==========================================================================
// One thread owns one whole column, so the scheme's column arrays are
// naturally function-scope locals.  CUDA prices those in a way that made
// this kernel the single largest device-memory term a BARE DEFAULT run
// pays: the driver sizes ONE per-context local-memory backing store to the
// widest kernel frame it LAUNCHES times the card's RESIDENT-THREAD
// CAPACITY (multiProcessorCount * maxThreadsPerMultiProcessor), never to
// the occupancy the kernel achieves, and never returns it while the
// context lives.
//
// MEASURED on a test host (RTX 5070 Ti, 70 SMs x 1,536,
// sm_120, NVRTC 13.0.48, CuPy 14.0.1) through the real launcher at nz=49:
// the 9,232 B frame reserved 844.0 MiB.  `bl_pbl_physics = 1` is the
// shipped default (gpuwm/domain_wizard.py:714), so every default run paid
// it.
//
// The same measurement established that the reservation is taken at
// LAUNCH, not at module load: a module holding a 16,384 B kernel reserved
// 0.0 MiB until that kernel was actually launched.  That is why the
// ceiling is set by the kernels a configuration RUNS, not by the widest
// one its modules contain.
//
// So the column arrays live in a caller-provided GLOBAL workspace instead,
// sized to the threads actually in flight.  gpuwm/core/ysu.py sizes the
// tile from the kernel's measured occupancy and launches the columns in
// tiles of that size; the arrays keep their extents and their access
// order, so this is a PLACEMENT change and nothing else.
//
// The per-slot extent is a RUNTIME argument (`wskp`), not YSU_KMAX.  The
// frame had to be sized for the deepest column the kernel would ever
// accept -- 128 levels -- so a 49-level run carried 128 levels of arrays.
// The workspace is allocated when the extent is known, so a 49-level run
// holds 50, which is a 2.6x cut this kernel could not previously express.
#define YSUWS_SLOTS 18

// LANE INTERLEAVING, and why it is not optional.  CUDA lays local memory
// out interleaved across the threads of a warp, so a per-thread array
// access is coalesced by construction.  A workspace that gave each thread
// a contiguous slab would hand that back -- MEASURED on the Grell-Freitas
// cut that this follows, the contiguous form ran 42.1 ms against 19.3 ms
// and got WORSE as the tile grew, the signature of 32-way scatter.  So the
// workspace is laid out exactly the way local memory is: one contiguous
// region per BLOCK, element k of slot s for lane t at
//   block_base + (s * wskp + k) * YSUWS_LANES + t
// and a warp reading arr[k] touches 32 consecutive floats.
//
// YSUWS_LANES is the launch block, fixed: gpuwm/core/ysu.py's _TPB and
// tests/test_ysu_workspace.py both pin it, and a launch at any other block
// width would alias lanes.
#define YSUWS_LANES 32

//: A column array living in the workspace.  Indexes like the
//: `real[YSU_KMAX]` it replaced; the stride is what makes it coalesce.
struct YsuCol {
    real *p;
    __device__ __forceinline__ real &operator[](int k) const {
        return p[(size_t)k * (size_t)YSUWS_LANES];
    }
};
//: The read-only view, so a `const real *` parameter stays const.
struct YsuColC {
    const real *p;
    __device__ __forceinline__ YsuColC(const YsuCol &a) : p(a.p) {}
    __device__ __forceinline__ real operator[](int k) const {
        return p[(size_t)k * (size_t)YSUWS_LANES];
    }
};

#define YSUWS_AT(base, idx, kp) \
    (YsuCol{(base) + (size_t)(idx) * (size_t)(kp) * (size_t)YSUWS_LANES})
//: This thread's lane inside its block's workspace region.
#define YSUWS_LANE_BASE(ws, kp) \
    ((ws) + (size_t)blockIdx.x * (size_t)YSUWS_SLOTS * (size_t)(kp) \
     * (size_t)YSUWS_LANES + (size_t)threadIdx.x)

// NaN-propagating on purpose.  Fortran's amin1/amax1 spelled as `a < b ?`
// silently launders a NaN first argument into the bound: ysu_min(NaN,
// xkzmax) used to return xkzmax, so a poisoned diffusivity became a
// plausible 1000 m2/s and sailed through validate_ysu_outputs.  For every
// non-NaN `a` these are bit-identical to the old `a < b ? a : b` /
// `a > b ? a : b` (ties and signed zeros still select b); only a NaN `a`
// now comes back out, so it reaches the driver's finiteness check instead
// of becoming state.
__device__ __forceinline__ real ysu_min(real a, real b)
{ return (a != a) ? a : (a < b ? a : b); }
__device__ __forceinline__ real ysu_max(real a, real b)
{ return (a != a) ? a : (a > b ? a : b); }
__device__ __forceinline__ real ysu_clip(real x, real lo, real hi) {
    return ysu_min(ysu_max(x, lo), hi);
}

// The Thomas recurrence in register-carried form.  Every operation and both
// of its operands are the ones the plain form uses:
//     gamma[k] = upper[k] / (diag[k] - lower[k]*gamma[k-1])   (as * inv)
//     rhs[k]   = (rhs[k] - lower[k]*rhs[k-1]) * inv
//     rhs[k]  -= gamma[k]*rhs[k+1]                             (back pass)
// Only where the operands come from changes: gamma[k-1], rhs[k-1] and
// rhs[k+1] are the words this thread has just stored, held in registers
// instead of reloaded from the workspace, and the next level's
// coefficients are loaded before this level's stores, so those loads
// overlap the divide instead of waiting behind it.  The workspace ends
// with the same words.  Needs nz >= 2 (YSU's vertical bounds ask for 4).
__device__ void ysu_thomas(YsuColC lower, YsuColC diag,
                           YsuColC upper, YsuCol rhs, YsuCol gamma, int nz) {
    real inv = 1.0f / diag[0];
    real g = upper[0] * inv;
    gamma[0] = g;
    real r = rhs[0] * inv;
    rhs[0] = r;
    real lk = lower[1], dk = diag[1], uk = upper[1], rk = rhs[1];
    for (int k = 1; k < nz - 1; ++k) {
        real ln = lower[k + 1], dn = diag[k + 1], un = upper[k + 1];
        real rn = rhs[k + 1];
        inv = 1.0f / (dk - lk * g);
        g = uk * inv;
        gamma[k] = g;
        r = (rk - lk * r) * inv;
        rhs[k] = r;
        lk = ln; dk = dn; uk = un; rk = rn;
    }
    // lk, dk, rk now hold lower/diag/rhs[nz-1]; g, r hold gamma/rhs[nz-2].
    inv = 1.0f / (dk - lk * g);
    real gb = g, rb = r;
    r = (rk - lk * r) * inv;
    rhs[nz - 1] = r;
    for (int k = nz - 2; k >= 0; --k) {
        int kp = k > 0 ? k - 1 : 0;
        real gp = gamma[kp], rp = rhs[kp];
        r = rb - gb * r;
        rhs[k] = r;
        gb = gp; rb = rp;
    }
}

__device__ void ysu_diagnose(const real *u, const real *v, YsuColC thv,
                             YsuColC za, real thermal, real br0,
                             real brcrit, int nz, int st, int col,
                             real *hpbl, int *kpbl,
                             real *brdn_out, real *brup_out) {
    real brup = br0, brdn = brup;
    int kp = 1;
    bool crossed = false;
    for (int k = 1; k < nz; ++k) {
        if (!crossed) {
            brdn = brup;
            int q = k * st + col;
            real spdk2 = ysu_max(u[q] * u[q] + v[q] * v[q], 1.0f);
            brup = (thv[k] - thermal) * (G * za[k] / thv[0]) / spdk2;
            kp = k + 1;
            crossed = brup > brcrit;
        }
    }
    real frac;
    if (brdn >= brcrit) frac = 0.0f;
    else if (brup <= brcrit) frac = 1.0f;
    else frac = (brcrit - brdn) / (brup - brdn);
    int kh = kp - 1;
    *hpbl = za[kh - 1] + frac * (za[kh] - za[kh - 1]);
    *kpbl = kp;
    *brdn_out = brdn;
    *brup_out = brup;
}

// --------------------------------------------------------------------------
// flag_bep arm (sf_urban_physics 2/3), bl_ysu.F90 at MMM-physics
// 20240626-MPASv8.2, WRF v4.7.1.  That file differs from the v4.6.1 copy
// this kernel was transcribed from by one line (`we(i) = 0.` at :605, which
// the `we` local below already is); the flag_bep arithmetic is identical.
// YsuBep carries the urban source terms exactly as
// module_sf_noahdrv.F:1679-1720 / module_sf_noahmpdrv.F:3700-3740 leave
// them (frc-weighted, rural surface flux folded into level 1):
//   sf    -> sfk2d  (dtodsd/dtodsu factor, :1050-1051, :1138-1139, :1326-1327)
//   vl    -> vlk2d  (au/al divisor, :1070-1071, :1158-1159, :1351-1352)
//   a_t/b_t, a_q/b_q, a_u/b_u, a_v/b_v -> the implicit/explicit forcing
//                    added after assembly (:1078-1083, :1164-1168, :1359-1368)
//   frc   -> frc_urb1d, the urban fraction WRF removes from the surface
//            drag (:1313).  Not read since the declared rural-drag
//            divergence below removes the whole of YSU's own drag; kept in
//            the signature so the launch contract and the WRF-stock arm's
//            inputs are unchanged.
// a_e/b_e/dlg/dl_u are unused by YSU (declared, never read in bl_ysu.F90).
// The body is one template so ysu_column (BEP = false) compiles from exactly
// the statements it always had; every BEP change is under `if constexpr`.
struct YsuBep {
    const real *a_u, *a_v, *a_t, *a_q, *b_u, *b_v, *b_t, *b_q, *sf, *vl;
    const real *frc;
};

// bl_ysu.F90 tridi2n's f2 arm, exactly as WRF solves it: the v column
// pivots on its OWN diagonal cm1 (ad1 = ad - a_v*dt2) but eliminates and
// back-substitutes with au, the factors the u column's diagonal cm produced.
// That is only the true v solve when cm1 == cm, i.e. always outside BEP --
// which is why the plain kernel can re-run ysu_thomas for v -- and it is
// what WRF does whenever a_v_bep differs from a_u_bep.  `gamma_u` must be
// the u solve's gamma, untouched since.
__device__ void ysu_tridi2n_v(YsuColC lower, YsuColC diag_v,
                              YsuColC gamma_u, YsuCol rhs, int nz) {
    real inv = 1.0f / diag_v[0];
    rhs[0] *= inv;
    for (int k = 1; k < nz - 1; ++k) {
        inv = 1.0f / (diag_v[k] - lower[k] * gamma_u[k - 1]);
        rhs[k] = (rhs[k] - lower[k] * rhs[k - 1]) * inv;
    }
    inv = 1.0f / (diag_v[nz - 1] - lower[nz - 1] * gamma_u[nz - 2]);
    rhs[nz - 1] = (rhs[nz - 1] - lower[nz - 1] * rhs[nz - 2]) * inv;
    for (int k = nz - 2; k >= 0; --k)
        rhs[k] -= gamma_u[k] * rhs[k + 1];
}

template <bool BEP, bool TOPO>
__device__ __forceinline__
void ysu_column_slots(const real *__restrict__ u, const real *__restrict__ v, const real *__restrict__ theta,
                const real *__restrict__ qv, const real *__restrict__ qc, const real *__restrict__ qi,
                const real *__restrict__ p, const real *__restrict__ p_interface, const real *__restrict__ exner,
                const real *__restrict__ dz, const real *__restrict__ rthraten,
                const real *__restrict__ psfc, const real *__restrict__ znt,
                const real *__restrict__ ust, const real *__restrict__ hfx, const real *__restrict__ qfx,
                const real *__restrict__ wspd, const real *__restrict__ br, const real *__restrict__ psim,
                const real *__restrict__ psih, const real *__restrict__ xland, const real *u10,
                const real *v10, real *__restrict__ du, real *__restrict__ dv, real *__restrict__ dtheta,
                real *__restrict__ dqv, real *__restrict__ dqc, real *__restrict__ dqi, real *__restrict__ hpbl_out,
                int *__restrict__ kpbl_out, real *__restrict__ exch_h, real *__restrict__ exch_m,
                real *__restrict__ wstar_out, real *__restrict__ delta_out,
                real dt, real *__restrict__ topdown_radsum_out,
                real *__restrict__ wstar3_2_out, int *__restrict__ cloudflg_out,
                int ysu_topdown_pblmix, int nz, int ny, int nx,
                real *__restrict__ ws, int wskp, int col0, const YsuBep bep, const YsuTopo topo,
                real *__restrict__ ws_thv, real *__restrict__ ws_thli, real *__restrict__ ws_zq, real *__restrict__ ws_za,
                real *__restrict__ ws_dza, real *__restrict__ ws_delp, real *__restrict__ ws_xkzm, real *__restrict__ ws_xkzh,
                real *__restrict__ ws_xkzq, real *__restrict__ ws_xkzml, real *__restrict__ ws_xkzhl, real *__restrict__ ws_zfacent,
                real *__restrict__ ws_entfac, real *__restrict__ ws_lower, real *__restrict__ ws_diag, real *__restrict__ ws_upper,
                real *__restrict__ ws_rhs, real *__restrict__ ws_gamma) {
    // `col0` is the first column of this TILE.  The workspace is sized to
    // one tile, so its indexing uses the TILE-LOCAL blockIdx.x while every
    // field index keeps using the global column -- the arrays are
    // (nz, ny, nx) and a column tile is not a contiguous slice of them, so
    // the tile is expressed as an offset rather than as a view.
    //
    // A tile is always a whole number of blocks (gpuwm/core/ysu.py sizes it
    // as SMs x blocks/SM x YSUWS_LANES), so only the FINAL tile launches
    // threads past its end, and those are caught by `col >= st` exactly as
    // the trailing threads of the old single launch were.
    int col = blockDim.x * blockIdx.x + threadIdx.x + col0;
    int st = ny * nx;
    if (col >= st || nz > YSU_KMAX) return;

    // Each column array is its own __restrict__ slot pointer (the slots are
    // disjoint wskp-level regions of this lane's workspace), so the
    // compiler may keep one array's loads ahead of another's stores.
    YsuCol thv     = YsuCol{ws_thv};
    YsuCol thli    = YsuCol{ws_thli};
    YsuCol zq      = YsuCol{ws_zq};
    YsuCol za      = YsuCol{ws_za};
    YsuCol dza     = YsuCol{ws_dza};
    YsuCol delp    = YsuCol{ws_delp};
    YsuCol xkzm    = YsuCol{ws_xkzm};
    YsuCol xkzh    = YsuCol{ws_xkzh};
    YsuCol xkzq    = YsuCol{ws_xkzq};
    YsuCol xkzml   = YsuCol{ws_xkzml};
    YsuCol xkzhl   = YsuCol{ws_xkzhl};
    YsuCol zfacent = YsuCol{ws_zfacent};
    YsuCol entfac  = YsuCol{ws_entfac};
    YsuCol lower   = YsuCol{ws_lower};
    YsuCol diag    = YsuCol{ws_diag};
    YsuCol upper   = YsuCol{ws_upper};
    YsuCol rhs     = YsuCol{ws_rhs};
    YsuCol gamma   = YsuCol{ws_gamma};

    real us = ust[col], hf = hfx[col], qf = qfx[col];
    // No zero-coupling short circuit.  Until lane/verify-ysu this kernel
    // returned zero tendencies, PBLH = dz(1) and KPBL = 1 whenever UST, HFX
    // and QFX were all zero, and -ftz fired the same branch on a subnormal
    // UST.  bl_ysu_run has no such branch: on a stable column it computes a
    // PBL (900 m and ten levels on the wide oracle's probe), and the kernel
    // now runs that column as WRF does.  Where WRF itself divides 0 by 0
    // (sfcflg true, ust**3 == 0: prfac2 below) the kernel keeps its defined
    // value; every other quantity on such a column is finite here because
    // wspd is guarded only where WRF would divide by zero (ysu_pos).
    // Graded by tests/test_ysu_wrf461_wide_parity.py (zero-coupling
    // probes) against tools/ysu_wrf461_oracle/run_bl_ysu_wide.F90.
    //
    //

    const real xkzminm = 0.1f, xkzminh = 0.01f, xkzmax = 1000.0f;
    const real rimin = -100.0f, rlam = 30.0f, prmin = 0.25f, prmax = 4.0f;
    const real brcr_ub = 0.0f, brcr_sb = 0.25f, cori = 1.0e-4f;
    const real afac = 6.8f, bfac = 6.8f, phifac = 8.0f, sfcfrac = 0.1f;
    const real d1 = 0.02f, d2 = 0.05f, d3 = 0.001f;
    const real h1 = 0.333333333333f, h2 = 0.666666666667f;
    const real zfmin = 1.0e-8f, aphi5 = 5.0f, aphi16 = 16.0f;
    const real tmin = 0.01f, gamcrt = 3.0f, gamcrq = 0.002f;
    // ep1 is WRF's EP_1, and WRF forms it in float32: module_model_constants
    // declares `REAL, PARAMETER :: EP_1 = R_v/R_d - 1.` so the quotient is a
    // float32 divide of two float32 values.  CUDA_DEFINES["RVOVRD"] is RV/RD
    // computed in Python doubles and rounded once to float32, which lands 1
    // ULP below WRF's quotient at 1.608 and therefore 2 ULP below it here at
    // 0.608.  Spelling the divide keeps the whole thv/Richardson/PBL-diagnosis
    // chain on WRF's constant; measured worth on the oracle fixture: hpbl
    // 112 -> 1 ULP, exch_h 283 -> 7, exch_m 48 -> 7, dv 46604 -> 23302.
    const real karman = 0.4f, ep1 = RV / RD - 1.0f;
    real dt2 = 2.0f * dt, rdt = 1.0f / dt2;

    // One pass instead of three: zq[k], za[k-1] and the interface pressure
    // ride in registers to the next level.  Each word is the same operation
    // on the same operands as the three separate loops formed:
    //   zq[k+1] = zq[k] + dz,  za[k] = 0.5*(zq[k] + zq[k+1]),
    //   dza[0] = za[0],  dza[k] = za[k] - za[k-1].
    zq[0] = 0.0f;
    real zq_k = 0.0f, za_prev = 0.0f, pi_k = p_interface[col];
    for (int k = 0; k < nz; ++k) {
        int q = k * st + col;
        thv[k] = ysu_thx(theta, exner, q) * (1.0f + ep1 * qv[q]);
        thli[k] = (theta[q] * exner[q] - __fdiv_rn(XLV * qc[q], CP)
                   - __fdiv_rn(2.834e6f * qi[q], CP)) / exner[q];
        real zq_k1 = zq_k + dz[q];
        zq[k + 1] = zq_k1;
        real pi_k1 = p_interface[(k + 1) * st + col];
        delp[k] = pi_k - pi_k1;
        real za_k = 0.5f * (zq_k + zq_k1);
        za[k] = za_k;
        dza[k] = k == 0 ? za_k : za_k - za_prev;
        zq_k = zq_k1;
        za_prev = za_k;
        pi_k = pi_k1;
    }

    real theta0 = ysu_thx(theta, exner, col), qv0 = qv[col];
    real temp0 = theta[col] * exner[col];
    real rho = psfc[col] / (RD * temp0 * (1.0f + ep1 * qv0));
    real govrth = G / theta0;
    real u0 = u[col], v0 = v[col];
    real wspd1 = sqrtf(u0 * u0 + v0 * v0) + 1.0e-9f;
    real sflux = __fdiv_rn(hf / rho, CP) + qf / rho * ep1 * theta0;
    real thermal = thv[0], thermalli = thli[0];
    real hpbl;
    int kpbl;
    real brdn, brup;
    ysu_diagnose(u, v, thv, za, thermal, br[col], brcr_ub, nz, st, col,
                 &hpbl, &kpbl, &brdn, &brup);
    if (hpbl < zq[1]) kpbl = 1;
    bool pblflg = kpbl > 1;
    bool sfcflg = !(ysu_f2d(br[col]) > 0.0);  // :613 in double; ysu_topo.cuh
    real zol1 = ysu_max(br[col] * psim[col] * psim[col] / psih[col], rimin);
    zol1 = sfcflg ? ysu_min(zol1, -zfmin) : ysu_max(zol1, zfmin);
    real hol1 = zol1 * hpbl / za[0] * sfcfrac;
    real phim, phih, wstar3, wstar;
    if (sfcflg) {
        phim = gfk_pow(1.0f - aphi16 * hol1, -0.25f);
        phih = gfk_pow(1.0f - aphi16 * hol1, -0.5f);
        wstar3 = govrth * ysu_max(sflux, 0.0f) * hpbl;
        wstar = gfk_pow(wstar3, h1);
    } else {
        phim = phih = 1.0f + aphi5 * hol1;
        wstar3 = wstar = 0.0f;
    }
    real ust3 = gfk_pow(us, 3.0f);
    real wscale = gfk_pow(ust3 + phifac * karman * wstar3 * 0.5f, h1);
    wscale = ysu_min(wscale, us * aphi16);
    wscale = ysu_max(wscale, __fdiv_rn(us, aphi5));
    real hgamt = 0.0f, hgamq = 0.0f, hgamu = 0.0f, hgamv = 0.0f;
    if (sfcflg && sflux > 0.0f) {
        real gamfac = bfac / rho / wscale;
        hgamt = ysu_min(__fdiv_rn(gamfac * hf, CP), gamcrt);
        hgamq = ysu_min(gamfac * qf, gamcrq);
        real vpert = __fdiv_rn((hgamt + ep1 * theta0 * hgamq), bfac) * afac;
        thermal += ysu_max(vpert, 0.0f)
                 * ysu_min(za[0] / (sfcfrac * hpbl), 1.0f);
        thermalli += ysu_max(vpert, 0.0f)
                   * ysu_min(za[0] / (sfcfrac * hpbl), 1.0f);
        hgamt = ysu_max(hgamt, 0.0f);
        hgamq = ysu_max(hgamq, 0.0f);
        real cg = -15.9f * us * us / ysu_pos(wspd[col], 1.0e-9f) * wstar3
                / ysu_pos(gfk_pow(wscale, 4.0f), 1.0e-20f);
        hgamu = cg * u0;
        hgamv = cg * v0;
        // bl_ysu.F90:703-728 guards all three thermal-enhanced statements
        // with if(pblflg(i)), and :684-698 can only LOWER pblflg -- WRF has
        // no path that raises it here.  A column whose FIRST guess sat
        // below zq(i,2) (:646-647) therefore keeps kpbl=1 and stays in the
        // local-K regime for the whole step, however far the thermal excess
        // could have pushed the enhanced sweep.
        //
        // The sweep is the WHOLE of :703-728: it leaves hpbl at the
        // zq(i,1) of :706 and never touches pblflg.  WRF interpolates
        // hpbl and applies the zq(i,2) clamp exactly once, at :754-768,
        // AFTER the theta-li scan -- so the scan below must see this
        // sweep's kpbl >= 2 and an unchanged pblflg.  ysu_diagnose folds
        // the interpolation in on its way out; that value is dead for
        // this call, because the post-scan if(pblflg) block recomputes it
        // from the same brdn/brup before any reader, exactly as :764 does.
        if (pblflg) {
            ysu_diagnose(u, v, thv, za, thermal, br[col], brcr_ub, nz, st,
                         col, &hpbl, &kpbl, &brdn, &brup);
        }
    } else pblflg = false;

    // WRF v4.6.1 bl_ysu.F90:732-768 runs the theta-li extension for every
    // column.  In-cloud liquid loading can therefore revive pblflg even
    // when the surface is stable and wstar3 remains zero.
    if (ysu_topdown_pblmix) {
        bool definebrup = false;
        int kpblold = kpbl;
        for (int fk = kpblold; fk <= nz - 1; ++fk) {
            int k = fk - 1;
            int q = k * st + col;
            real spdk2 = ysu_max(u[q] * u[q] + v[q] * v[q], 1.0f);
            real bruptmp = (thli[k] - thermalli)
                           * (G * za[k] / thli[0]) / spdk2;
            bool stable_li = bruptmp >= brcr_ub;
            if (definebrup) {
                kpbl = fk;
                brup = bruptmp;
                definebrup = false;
            }
            if (!stable_li) {
                brdn = bruptmp;
                definebrup = true;
                pblflg = true;
            }
        }
    }
    if (pblflg) {
        int kh = kpbl - 1;
        real frac;
        if (brdn >= brcr_ub) frac = 0.0f;
        else if (brup <= brcr_ub) frac = 1.0f;
        else frac = (brcr_ub - brdn) / (brup - brdn);
        hpbl = za[kh - 1] + frac * (za[kh] - za[kh - 1]);
        // bl_ysu.F90:765 and :766 are two INDEPENDENT statements.  The
        // second kills pblflg whenever kpbl <= 1, whatever hpbl came back
        // as -- including the case where hpbl was interpolated from WRF's
        // own za(i,0) overrun above and came back large.  Nesting :766
        // inside :765 let the theta-li revival (:745-749 raises pblflg on a
        // final unstable iteration without ever reassigning kpbl) survive
        // with kpbl == 1, and :833's k = kpbl(i)-1 is then one level below
        // the column.
        if (hpbl < zq[1]) kpbl = 1;
        if (kpbl <= 1) pblflg = false;
    }

    if (!sfcflg && hpbl < zq[1]) {
        real brcrit = brcr_sb;
        if (xland[col] >= 1.5f) {
            real ross = ysu_max(sqrtf(u10[col] * u10[col] + v10[col] * v10[col]),
                                1.0e-9f) / (cori * znt[col]);
            brcrit = ysu_min(0.16f * gfk_pow(1.0e-7f * ross, -0.18f), 0.3f);
        }
        ysu_diagnose(u, v, thv, za, thermal, br[col], brcrit, nz, st, col,
                     &hpbl, &kpbl, &brdn, &brup);
        if (hpbl < zq[1]) kpbl = 1;
        // Keep pblflg false: WRF uses kpbl for the stable K profile but
        // reserves countergradient and entrainment terms for convection.
    }

    real wm2 = 0.0f, we = 0.0f, hfxpbl = 0.0f, qfxpbl = 0.0f;
    real ufxpbl = 0.0f, vfxpbl = 0.0f, delta = 0.0f;
    real wstar3_2 = 0.0f, topdown_radsum = 0.0f;
    bool cloudflg = false;
    if (pblflg) {
        int kt = kpbl - 2;
        real wm3 = wstar3 + 5.0f * ust3;
        wm2 = gfk_pow(wm3, h2);
        real bfxpbl = __fdiv_rn(-0.15f * thv[0], G) * wm3 / hpbl;
        real dthv = ysu_max(thv[kt + 1] - thv[kt], tmin);
        we = ysu_max(bfxpbl / dthv, -sqrtf(wm2));
        // F90 839-897.  The kpbl<nz guard makes the source's k+2 access
        // explicit: a top-level PBL has no above-cloud comparison layer.
        if (ysu_topdown_pblmix && kpbl < nz
                && qc[kt * st + col] + qi[kt * st + col] > 1.0e-5f) {
            cloudflg = true;
            real ptop = p_interface[(kt + 1) * st + col];
            real templ = thli[kt] * gfk_pow(__fdiv_rn(ptop, 100000.0f), RCP);
            real rvls = 100.0f * 6.112f
                      * gfk_exp(17.67f * (templ - 273.16f)
                             / (templ - 29.65f)) * (EP2 / ptop);
            real qsum = qv[kt * st + col] + qc[kt * st + col];
            real temps = templ + (qsum - rvls)
                       / (CP / XLV + EP2 * XLV * rvls
                          / (RD * (templ * templ)));
            rvls = 100.0f * 6.112f
                 * gfk_exp(17.67f * (temps - 273.15f)
                        / (temps - 29.65f)) * (EP2 / ptop);
            real rcldb = ysu_max(qsum - rvls, 0.0f);
            int kabove = kt + 2;
            real dthv_cloud = (thli[kabove]
                    + ysu_thx(theta, exner, kabove * st + col) * ep1
                    * (qv[kabove * st + col] + qc[kabove * st + col]))
                    - (thli[kt] + ysu_thx(theta, exner, kt * st + col) * ep1 * qsum);
            dthv_cloud = ysu_max(dthv_cloud, 0.1f);
            real tmp1 = XLV / CP * rcldb
                      / (exner[kt * st + col] * dthv_cloud);
            // WRF 4.8 MMM 550b5b4:855 includes both upstream fixes.
            // Preserve (0.2 * 8.) * tmp1 before the cap; no offset.
#ifndef YSU_LEGACY_TOPDOWN
            real ent_eff = ysu_min(0.2f * 8.0f * tmp1, 0.4f);
#else
            // Explicit strict-arithmetic legacy arm: the WRF 4.6.1 term
            // with the source-ordered driver and shared no-FTZ math.
            real ent_eff = 0.2f * 8.0f * tmp1 + 0.2f;
#endif
            for (int kk = 0; kk <= kt; ++kk) {
                real radflux = rthraten[kk * st + col]
                             * exner[kk * st + col];
                // :859 radflux*cp/g*(dp) is ((radflux*cp)/g)*dp, not radflux*(cp/g*dp)
                radflux = __fmul_rn(__fdiv_rn(__fmul_rn(radflux, CP), G),
                    p_interface[kk * st + col] - p_interface[(kk + 1) * st + col]);
                if (radflux < 0.0f) topdown_radsum += fabsf(radflux);
            }
            topdown_radsum = ysu_max(topdown_radsum, 0.0f);
            real tk = theta[kt * st + col] * exner[kt * st + col];
            real rho2 = p[kt * st + col]
                      / (RD * tk * (1.0f + ep1 * qv[kt * st + col]));

            // Preserve the two overwritten bfx0 assignments in v4.6.1:
            // the executable values are max(sflux,0) and radsum/rho/cp.
            real bfx0 = ysu_max(sflux, 0.0f);
            wm3 = govrth * bfx0 * hpbl + 5.0f * ust3;
            wm2 = gfk_pow(wm3, h2);
            bfxpbl = __fdiv_rn(-0.15f * thv[0], G) * wm3 / hpbl;
            dthv = ysu_max(thv[kt + 1] - thv[kt], tmin);
            we = ysu_max(bfxpbl / dthv, -sqrtf(wm2));

            bfx0 = ysu_max(__fdiv_rn(topdown_radsum / rho2, CP), 0.0f);
            real wm3_top = G / thv[kt] * bfx0 * hpbl;
            real wm2_top = gfk_pow(wm3_top, h2);
            wm2 += wm2_top;
            bfxpbl = -ent_eff * bfx0;
            dthv = ysu_max(thv[kt + 1] - thv[kt], 0.1f);
            we += ysu_max(bfxpbl / dthv, -sqrtf(wm2_top));
            wstar3_2 = G / thv[kt] * bfx0 * hpbl;
            wscale = gfk_pow(ust3 + phifac * karman
                             * (wstar3 + wstar3_2) * 0.5f, h1);
            wscale = ysu_min(wscale, us * aphi16);
            wscale = ysu_max(wscale, __fdiv_rn(us, aphi5));
            real gamfac = bfac / rho / wscale;
            hgamt = ysu_min(__fdiv_rn(gamfac * hf, CP), gamcrt);
            hgamq = ysu_min(gamfac * qf, gamcrq);
            gamfac = bfac / rho2 / wscale;
            real hgamt2 = ysu_min(__fdiv_rn(gamfac * topdown_radsum, CP), gamcrt);
            hgamt = ysu_max(hgamt, 0.0f) + ysu_max(hgamt2, 0.0f);
            real cg = -15.9f * us * us / ysu_pos(wspd[col], 1.0e-9f)
                    * (wstar3 + wstar3_2)
                    / ysu_pos(gfk_pow(wscale, 4.0f), 1.0e-20f);
            hgamu = cg * u0;
            hgamv = cg * v0;
        }
        real dth = ysu_max(ysu_thx(theta, exner, (kt + 1) * st + col) - ysu_thx(theta, exner, kt * st + col), tmin);
        real dqq = ysu_min(qv[(kt + 1) * st + col] - qv[kt * st + col], 0.0f);
        hfxpbl = we * dth;
        qfxpbl = we * dqq;
        real dux = u[(kt + 1) * st + col] - u[kt * st + col];
        real dvx = v[(kt + 1) * st + col] - v[kt * st + col];
        if (dux > tmin) ufxpbl = ysu_max(we * dux, -us * us);
        else if (dux < -tmin) ufxpbl = ysu_min(we * dux, us * us);
        if (dvx > tmin) vfxpbl = ysu_max(we * dvx, -us * us);
        else if (dvx < -tmin) vfxpbl = ysu_min(we * dvx, us * us);
        real delb = govrth * d3 * hpbl;
        delta = ysu_min(d1 * hpbl + d2 * wm2 / delb, 100.0f);
    }
    // Each word is stored once with the value the overwritten stores left.
    for (int k = 0; k < nz; ++k) {
        real ef = 1.0e30f;
        if (pblflg && delta > 0.0f && k + 1 >= kpbl) {
            real e = (zq[k + 1] - hpbl) / delta;
            ef = e * e;               // :930 **2. is x*x in WRF's -O2 build
        }
        entfac[k] = ef;
        bool below_top = k < nz - 1;
        real xh0 = below_top ? xkzminh : 0.0f;
        xkzm[k] = below_top ? xkzminm : 0.0f;
        xkzh[k] = xh0;
        xkzq[k] = xh0;
        xkzml[k] = 0.0f;
        xkzhl[k] = 0.0f;
        zfacent[k] = 0.0f;
    }

    for (int k = 0; k < nz; ++k) {
        if (k + 1 < kpbl) {
            real zfac = ysu_clip(1.0f - (zq[k + 1] - za[0]) / (hpbl - za[0]),
                                  zfmin, 1.0f);
            zfacent[k] = gfk_pow(1.0f - zfac, 3.0f);
            real wsk = gfk_pow(ust3 + phifac * karman * wstar3
                               * (1.0f - zfac), h1);
            real wsk2 = gfk_pow(phifac * karman * wstar3_2 * zfac,
                                h1);
            real prfac, prfac2, prnumfac;
            if (sfcflg) {
                prfac = bfac * karman * sfcfrac;
                prfac2 = 15.9f * (wstar3 + wstar3_2) / ust3
                       / (1.0f + 4.0f * karman
                          * (wstar3 + wstar3_2) / ust3);
                if (!isfinite(prfac2)) {
                    // Even with the default no-FTZ route, cubing an
                    // extremely small ust can underflow to exactly zero.
                    // On the former FTZ route us*us*us reached zero for
                    // ust < ~2.3e-13 and the quotient above is Inf/Inf or
                    // 0/0 = NaN.  Pre-guard, ysu_min then laundered that
                    // NaN into xkzh = xkzmax: exch_h pinned at 1000 m2/s,
                    // no error raised.  WRF v4.6.1 has no floor here
                    // (bl_ysu.F90:674 cubes ust unguarded and :948 divides
                    // by it; the sfclay land floors, module_sf_sfclay.F:818
                    // and sf_sfclayrev.F90:772, are upstream and land-only),
                    // and on IEEE-subnormal hardware the same expression is
                    // finite: prfac2 -> 15.9/(4*karman) as ust3 -> 0.  So
                    // this is the rrtmg_sw.cu FP64-emulation countermeasure,
                    // not a WRF transcription: re-evaluate the same
                    // expression in float64, where us^3 down to the
                    // smallest normal float is itself normal.  It runs only
                    // when the FP32 result is already Inf/NaN, so every
                    // healthy lane keeps its exact FP32 word.  Where WRF
                    // itself is non-finite (w/ust3 overflowing FP32, or
                    // ust3 == 0 exactly), this produces the defined
                    // algebraic value instead -- never bit-exact to a bug.
                    double u3 = (double)us * (double)us * (double)us;
                    double w = (double)wstar3 + (double)wstar3_2;
                    if (u3 > 0.0)
                        prfac2 = (real)(15.9 * w / u3
                                        / (1.0 + 4.0 * (double)karman
                                           * w / u3));
                    else
                        prfac2 = w > 0.0
                               ? (real)(15.9 / (4.0 * (double)karman))
                               : 0.0f;
                }
                real zz = ysu_max(zq[k + 1] - sfcfrac * hpbl, 0.0f);
                prnumfac = -(3.0f * (zz * zz)) / (hpbl * hpbl);   // :949, x*x
            } else {
                prfac = prfac2 = prnumfac = 0.0f;
                wsk = ysu_max(us / (1.0f + aphi5 * zol1 * zq[k + 1] / za[0]),
                              0.001f);
            }
            real prnum0 = ysu_clip(phih / phim + prfac, prmin, prmax);
            real km = wsk * karman * zq[k + 1] * (zfac * zfac)   // :960-961
                    + wsk2 * karman * (hpbl - zq[k + 1])          // **pfac (2.)
                    * ((1.0f - zfac) * (1.0f - zfac));            // is x*x at -O2
            if (k == kpbl - 2 && cloudflg && we < 0.0f) km = 0.0f;
            real prnum = 1.0f + (prnum0 - 1.0f) * gfk_exp(prnumfac);
            real kqq = km / prnum;
            prnum0 /= 1.0f + prfac2 * karman * sfcfrac;
            prnum = 1.0f + (prnum0 - 1.0f) * gfk_exp(prnumfac);
            real kh = km / prnum;
            xkzm[k] = ysu_min(km + xkzminm, xkzmax);
            xkzh[k] = ysu_min(kh + xkzminh, xkzmax);
            xkzq[k] = ysu_min(kqq + xkzminh, xkzmax);
        }
    }

    for (int k = 0; k < nz - 1; ++k) {
        if (k + 1 >= kpbl) {
            int q0i = k * st + col, q1i = (k + 1) * st + col;
            real ud = u[q1i] - u[q0i], vd = v[q1i] - v[q0i];
            real ss = (ud * ud + vd * vd) / (dza[k + 1] * dza[k + 1]) + 1.0e-9f;
            real ri = (G / (0.5f * (thv[k + 1] + thv[k])))
                    * (thv[k + 1] - thv[k]) / (ss * dza[k + 1]);
            if (qc[q0i] + qi[q0i] > 1.0e-5f && qc[q1i] + qi[q1i] > 1.0e-5f) {
                real qmean = 0.5f * (qv[q0i] + qv[q1i]);
                real tmean = 0.5f * (theta[q0i] * exner[q0i]
                                     + theta[q1i] * exner[q1i]);
                real alph = __fdiv_rn(XLV * qmean, RD) / tmean;
                real chi = __fdiv_rn(__fdiv_rn(XLV * XLV * qmean, CP), RV) / tmean / tmean;
                ri = (1.0f + alph) * (ri - __fdiv_rn(G * G / ss / tmean, CP)
                                      * ((chi - alph) / (1.0f + chi)));
            }
            real zk = karman * zq[k + 1];
            real rlamdz = ysu_min(ysu_max(0.1f * dza[k + 1], rlam), 300.0f);
            rlamdz = ysu_min(dza[k + 1], rlamdz);
            real rl = zk * rlamdz / (rlamdz + zk);
            real dk = rl * rl * sqrtf(ss), km, kh;
            if (ri < 0.0f) {
                ri = ysu_max(ri, rimin);
                real sri = sqrtf(-ri);
                km = dk * (1.0f + 8.0f * (-ri) / (1.0f + 1.746f * sri));
                kh = dk * (1.0f + 8.0f * (-ri) / (1.0f + 1.286f * sri));
            } else {
                kh = dk / ((1.0f + 5.0f * ri) * (1.0f + 5.0f * ri));
                km = kh * ysu_min(1.0f + 2.1f * ri, prmax);
            }
            real xm = ysu_min(km + xkzminm, xkzmax);
            real xh = ysu_min(kh + xkzminh, xkzmax);
            xkzm[k] = xm;
            xkzh[k] = xh;
            xkzml[k] = xm;               // the words just stored, not reloaded
            xkzhl[k] = xh;
        }
    }
    for (int k = kpbl - 1; k < nz - 1; ++k) xkzq[k] = xkzh[k];

#ifndef GPUWM_WRF_EXACT
    // The nonlocal entrainment flux is explicit although local diffusion
    // is implicit.  Its donor is the jump across kpbl-2 -> kpbl-1, not
    // the jump at each interface to which zfacent distributes the flux.
    // Bound its forward-Euler relaxation Courant number by one.  Otherwise
    // a weak virtual inversion permits we*dt to cross several layers and
    // vapor overshoots the donor (then condensation makes a hot layer).
    // Use the actual pressure-mass divergence, including the flux entering
    // the lower member of the donor pair; a geometric dz/dt cap is not the
    // discrete operator's stability limit.  Scale paired scalar fluxes,
    // retaining their conservative interface divergence and the diagnosed
    // turbulent diffusivity, countergradient heat and momentum closure.
    if (pblflg && we < 0.0f && kpbl >= 2) {
        const int kt = kpbl - 2;
        real atop = (p[kt * st + col] - p[(kt + 1) * st + col])
                        / dza[kt + 1] * zfacent[kt];
        real abelow = kt > 0
            ? (p[(kt - 1) * st + col] - p[kt * st + col])
                / dza[kt] * zfacent[kt - 1] : 0.0f;
        if constexpr (BEP) {
            atop *= bep.sf[kt * st + col];
            if (kt > 0) abelow *= bep.sf[(kt - 1) * st + col];
        }
        const real relax = atop / delp[kt + 1]
                         + (atop - abelow) / delp[kt];
        const real courant = dt2 * (-we) * relax;
        if (courant > 1.0f) {
            hfxpbl /= courant;
            qfxpbl /= courant;
        }
    }

#endif

    // Heat matrix, including the countergradient and entrainment fluxes.
    // The assembly below writes diag[0..nz-1], rhs[0..nz-1], upper[0..nz-2]
    // and lower[1..nz-1] before any of them is read, and ysu_thomas never
    // reads lower[0] or upper[nz-1]; zeroing the four whole columns first
    // only cost workspace traffic.  The two unread ends are still zeroed so
    // the workspace holds the same words as before.
    lower[0] = 0.0f;
    upper[nz - 1] = 0.0f;
    diag[0] = 1.0f;
    if constexpr (BEP) {
        // bl_ysu.F90:1045 with bepswitch = 1: (1.0-bepswitch)*hfx/cont/del*dt2
        // is a signed zero, so the surface heat flux enters only through
        // b_t_bep(1), where the couple folded the rural part in.
        rhs[0] = theta0 - 300.0f + __fdiv_rn(0.0f * hf, (CP / G)) / delp[0] * dt2;
    } else {
        rhs[0] = theta0 - 300.0f + __fdiv_rn(hf, (CP / G)) / delp[0] * dt2;
    }
    // diag[k] and rhs[k] ride in registers from the level that set them
    // to the level that finishes them (the same words, not reloaded).
    real diag_k = diag[0], rhs_k = rhs[0];
    for (int k = 0; k < nz - 1; ++k) {
        real dtodsd, dtodsu;
        real rhs_k1, up, lo;
        if constexpr (BEP) {
            // bl_ysu.F90:1050-1051 (and :1138, :1326): sfk2d(i,k)*dt2/del.
            real sfk = bep.sf[k * st + col];
            dtodsd = sfk * dt2 / delp[k];
            dtodsu = sfk * dt2 / delp[k + 1];
        } else {
            dtodsd = dt2 / delp[k], dtodsu = dt2 / delp[k + 1];
        }
        real dsig = p[k * st + col] - p[(k + 1) * st + col];
        real rdz = 1.0f / dza[k + 1];
        real xk_k = xkzh[k];
        real tem1 = dsig * xk_k * rdz;
        if (pblflg && k + 1 < kpbl) {
            real flux = tem1 * (-hgamt / hpbl - hfxpbl * zfacent[k] / xk_k);
            rhs_k += dtodsd * flux;
            rhs_k1 = ysu_thx(theta, exner, (k + 1) * st + col) - 300.0f - dtodsu * flux;
        } else if (pblflg && k + 1 >= kpbl && entfac[k] < 4.6f) {
            real knew = -we * dza[kpbl - 1] * gfk_exp(-entfac[k]);
            xk_k = ysu_clip(sqrtf(ysu_max(knew * xkzhl[k], 0.0f)),
                            xkzminh, xkzmax);
            xkzh[k] = xk_k;
            rhs_k1 = ysu_thx(theta, exner, (k + 1) * st + col) - 300.0f;
        } else rhs_k1 = ysu_thx(theta, exner, (k + 1) * st + col) - 300.0f;
        real dsdz2 = dsig * xk_k * rdz * rdz;
        if constexpr (BEP) {
            // bl_ysu.F90:1070-1071: au = -dtodsd*dsdz2/vlk2d(i,k),
            // al = -dtodsu*dsdz2/vlk2d(i,k) -- both divide by level k.
            real vlk = bep.vl[k * st + col];
            up = -dtodsd * dsdz2 / vlk;
            lo = -dtodsu * dsdz2 / vlk;
        } else {
            up = -dtodsd * dsdz2;
            lo = -dtodsu * dsdz2;
        }
        upper[k] = up;
        lower[k + 1] = lo;
        diag[k] = diag_k - up;          // diag[k] -= upper[k]
        rhs[k] = rhs_k;
        diag_k = 1.0f - lo;             // diag[k+1] = 1 - lower[k+1]
        rhs_k = rhs_k1;
    }
    diag[nz - 1] = diag_k;
    rhs[nz - 1] = rhs_k;
    if constexpr (BEP) {
        // bl_ysu.F90:1078-1083.  __fmul_rn: WRF rounds the product before
        // the add/subtract, NVRTC would otherwise fuse it.
        for (int k = 0; k < nz; ++k) {
            int q = k * st + col;
            diag[k] = diag[k] - __fmul_rn(bep.a_t[q], dt2);
            rhs[k] = rhs[k] + __fmul_rn(bep.b_t[q], dt2);
        }
    }
    ysu_thomas(lower, diag, upper, rhs, gamma, nz);
    for (int k = 0; k < nz; ++k) {
        int q = k * st + col;
        // WRF's own association, bl_ysu.F90:1103: (f1(i,k) - thx(i,k) + 300.).
        // Not cosmetic.  The solve leaves rhs[k] ~ theta - 300, so
        // (rhs - theta) is exactly -300 and adding 300 gives exactly zero,
        // whereas (rhs + 300) rounds at magnitude ~theta -- one ULP coarser
        // than rhs itself -- before the subtract, and the residue survives.
        // Measured: 3.4e-07 K/s where WRF writes exactly 0.0, which is
        // 884345697 ULP of that zero.
        dtheta[q] = __fdiv_rn((rhs[k] - ysu_thx(theta, exner, q) + 300.0f) * rdt * exner[q], exner[q]);
    }

    // Vapor matrix (Kq); cloud and ice reuse its matrix.
    lower[0] = 0.0f;
    upper[nz - 1] = 0.0f;
    diag[0] = 1.0f;
    if constexpr (BEP) {
        // bl_ysu.F90:1135 with bepswitch = 1: a signed zero is added, which
        // is what turns a -0.0 qv into +0.0 exactly as WRF does.
        rhs[0] = qv0 + 0.0f * qf * G / delp[0] * dt2;
    } else {
        rhs[0] = qv0 + qf * G / delp[0] * dt2;
    }
    // diag[k] and rhs[k] ride in registers from the level that set them
    // to the level that finishes them (the same words, not reloaded).
    diag_k = diag[0];
    rhs_k = rhs[0];
    for (int k = 0; k < nz - 1; ++k) {
        real dtodsd, dtodsu;
        real rhs_k1, up, lo;
        if constexpr (BEP) {
            // bl_ysu.F90:1050-1051 (and :1138, :1326): sfk2d(i,k)*dt2/del.
            real sfk = bep.sf[k * st + col];
            dtodsd = sfk * dt2 / delp[k];
            dtodsu = sfk * dt2 / delp[k + 1];
        } else {
            dtodsd = dt2 / delp[k], dtodsu = dt2 / delp[k + 1];
        }
        real dsig = p[k * st + col] - p[(k + 1) * st + col];
        real rdz = 1.0f / dza[k + 1];
        real xk_k = xkzq[k];
        real tem1 = dsig * xk_k * rdz;
        if (pblflg && k + 1 < kpbl) {
            real flux = tem1 * (-qfxpbl * zfacent[k] / xk_k);
            rhs_k += dtodsd * flux;
            rhs_k1 = qv[(k + 1) * st + col] - dtodsu * flux;
        } else if (pblflg && k + 1 >= kpbl && entfac[k] < 4.6f) {
            real knew = -we * dza[kpbl - 1] * gfk_exp(-entfac[k]);
            xk_k = ysu_clip(sqrtf(ysu_max(knew * xkzhl[k], 0.0f)),
                            xkzminh, xkzmax);
            xkzq[k] = xk_k;
            rhs_k1 = qv[(k + 1) * st + col];
        } else rhs_k1 = qv[(k + 1) * st + col];
        real dsdz2 = dsig * xk_k * rdz * rdz;
        if constexpr (BEP) {
            // bl_ysu.F90:1070-1071: au = -dtodsd*dsdz2/vlk2d(i,k),
            // al = -dtodsu*dsdz2/vlk2d(i,k) -- both divide by level k.
            real vlk = bep.vl[k * st + col];
            up = -dtodsd * dsdz2 / vlk;
            lo = -dtodsu * dsdz2 / vlk;
        } else {
            up = -dtodsd * dsdz2;
            lo = -dtodsu * dsdz2;
        }
        upper[k] = up;
        lower[k + 1] = lo;
        diag[k] = diag_k - up;          // diag[k] -= upper[k]
        rhs[k] = rhs_k;
        diag_k = 1.0f - lo;             // diag[k+1] = 1 - lower[k+1]
        rhs_k = rhs_k1;
    }
    diag[nz - 1] = diag_k;
    rhs[nz - 1] = rhs_k;
    if constexpr (BEP) {
        // bl_ysu.F90:1164-1168: vapor alone solves with adv = ad - a_q*dt2;
        // cloud water and ice reuse the unforced ad (:1177, :1190).  xkzhl
        // is dead after the vapor assembly above, so it holds adv.
        for (int k = 0; k < nz; ++k) {
            int q = k * st + col;
            xkzhl[k] = diag[k] - __fmul_rn(bep.a_q[q], dt2);
            rhs[k] = rhs[k] + __fmul_rn(bep.b_q[q], dt2);
        }
        ysu_thomas(lower, xkzhl, upper, rhs, gamma, nz);
    } else {
        ysu_thomas(lower, diag, upper, rhs, gamma, nz);
    }
    for (int k = 0; k < nz; ++k) dqv[k * st + col] = (rhs[k] - qv[k * st + col]) * rdt;
    for (int k = 0; k < nz; ++k) rhs[k] = qc[k * st + col];
    ysu_thomas(lower, diag, upper, rhs, gamma, nz);
    for (int k = 0; k < nz; ++k) dqc[k * st + col] = (rhs[k] - qc[k * st + col]) * rdt;
    for (int k = 0; k < nz; ++k) rhs[k] = qi[k * st + col];
    ysu_thomas(lower, diag, upper, rhs, gamma, nz);
    for (int k = 0; k < nz; ++k) dqi[k * st + col] = (rhs[k] - qi[k * st + col]) * rdt;

    // Momentum matrix and implicit surface stress.
    lower[0] = 0.0f;
    upper[nz - 1] = 0.0f;
    real fric = us * us / wspd1 * rho * G / delp[0] * dt2      // :1306-1307
              * ((wspd1 / ysu_pos(wspd[col], 1.0e-9f))      // (wspd1/wspd)**2
                 * (wspd1 / ysu_pos(wspd[col], 1.0e-9f)));  // is formed first
    if constexpr (TOPO) {
        // bl_ysu.F90:1254-1314, the ctopo-present arm.  The paj TKE profile
        // (:1257-1279) reads the diffusivities as the heat and moisture
        // solves left them and before the momentum assembly below widens
        // xkzm (:1338-1342); thli is dead here and holds it.
        YsuCol tke = thli;
        // WRF writes only 1:kte-1 of this profile. Its get_pblh search can
        // reach the last level in a shallow domain. Define that missing word
        // as zero and retain the bounded searches in ysu_480_topo.cuh.
        // Checked public Fortran exposes qke1d(5) at nz=4, top=100 m.
        tke[nz - 1] = 0.0f;
        for (int k = 0; k < nz - 1; ++k) {
            int q0 = k * st + col, q1 = (k + 1) * st + col;
            real rdza = dza[k + 1];
            real dudz = __fdiv_rn(__fsub_rn(u[q1], u[q0]), rdza);
            real dvdz = __fdiv_rn(__fsub_rn(v[q1], v[q0]), rdza);
            real su = __fdiv_rn(__fmul_rn(__fadd_rn(-__fdiv_rn(hgamu, hpbl), dudz),
                                          __fsub_rn(u[q1], u[q0])), rdza);
            real sv = __fdiv_rn(__fmul_rn(__fadd_rn(-__fdiv_rn(hgamv, hpbl), dvdz),
                                          __fsub_rn(v[q1], v[q0])), rdza);
            real shear = __fmul_rn(xkzm[k], __fadd_rn(su, sv));
            real dthdz = __fdiv_rn(__fsub_rn(ysu_thx(theta, exner, q1), ysu_thx(theta, exner, q0)), rdza);
            real buoy = __fmul_rn(__fmul_rn(__fmul_rn(xkzh[k], G),
                                            __fdiv_rn(1.0f, ysu_thx(theta, exner, q0))),
                                  __fadd_rn(-__fdiv_rn(hgamt, hpbl), dthdz));
            real zk = __fmul_rn(karman, zq[k + 1]);
            real rlamdz;
            if (k + 1 >= kpbl) {
                rlamdz = ysu_min(ysu_max(__fmul_rn(0.1f, rdza), rlam), 300.0f);
                rlamdz = ysu_min(rdza, rlamdz);
            } else {
                rlamdz = 150.0f;
            }
            real el = __fdiv_rn(__fmul_rn(zk, rlamdz), __fadd_rn(rlamdz, zk));
            real t = __fmul_rn(__fmul_rn(16.6f, el), __fsub_rn(shear, buoy));
            tke[k] = (t <= 0.0f) ? 0.0f : gfk_pow(t, 0.66f);
        }
        real pblh_ysu = ysu_get_pblh<YsuColC>(thv, tke, zq, nz, xland[col]);
        real vconv;
        if (xland[col] < 1.5f) {
            real fluxc = ysu_max(sflux, 0.0f);
            vconv = __fmul_rn(1.0f, gfk_pow(__fmul_rn(__fmul_rn(
                        __fdiv_rn(G, thv[0]), pblh_ysu), fluxc), 0.33f));
        } else {
            vconv = 0.0f;
        }
        real vconvnew = __fadd_rn(__fmul_rn(0.9f, vconv),
                                  __fmul_rn(1.5f, ysu_max(__fdiv_rn(
                                      __fsub_rn(pblh_ysu, 500.0f), 1000.0f),
                                      0.0f)));
        real vconvlim = ysu_min(vconvnew, 1.0f);
        // ctopo = 1 on the default and BEP entry points (YsuTopo{} carries
        // null pointers): module_bl_ysu.F:404 passes it on every WRF run.
        real ctopo = topo.ctopo ? topo.ctopo[col] : 1.0f;
        real drag_v = __fmul_rn(fric, vconvlim);
        real drag_c = __fmul_rn(__fmul_rn(ctopo, fric),
                                __fsub_rn(1.0f, vconvlim));
        diag[0] = __fadd_rn(__fadd_rn(1.0f, drag_v), drag_c);
        if constexpr (BEP) {
            // :1313-1314 with the declared fix below: the WHOLE bracket,
            // ad(1) - bepswitch*(fric*vconvlim+ctopo*fric*(1-vconvlim)),
            // bepswitch = 1.  Outside BEP WRF's line subtracts 0*0*bracket,
            // an exact no-op, so it is not spelled.
            diag[0] = __fsub_rn(diag[0], __fadd_rn(drag_v, drag_c));
        }
    } else {
        // bl_ysu.F90:1315, the ctopo-absent arm.  No WRF run reaches it
        // (module_bl_ysu.F:404 always passes ctopo), and no entry point of
        // this kernel instantiates it any longer; kept only so the template
        // stays total.
        diag[0] = 1.0f + fric;
        if constexpr (BEP) diag[0] = diag[0] - fric;
    }
    if constexpr (BEP) {
        // DECLARED DIVERGENCE FROM WRF v4.7.1 (docs/public/PHYSICS.md, "Urban
        // canopy models"; pinned by tests/test_ysu_bep_rural_drag.py).
        // bl_ysu.F90:1313-1314 removes only the URBAN fraction of YSU's own
        // surface drag, ad(1) - bepswitch*frc*(fric*vconvlim+ctopo*fric*
        // (1-vconvlim)), leaving (1-frc)*fric on the diagonal.  But the BEP
        // couple has already folded that same rural drag,
        // (1-frc)*(-ust*ust)/dz8w/|U|, into a_u_bep/a_v_bep at level 1
        // (module_sf_noahdrv.F:1708-1711, module_sf_noahmpdrv.F:3718-3721),
        // and :1359-1368 below adds a_u*dt2 to the same diagonal: WRF counts
        // the rural surface drag twice on every column, ocean included, and
        // the 750 m runs lost 1.25 m/s of 10 m wind over the sea in two
        // hours.  Heat and moisture are counted once (:1045 drops YSU's own
        // flux with (1-bepswitch) and takes the rural flux from b_t/b_q), and
        // WRF's myjurb takes the surface drag only through a_u (its VDIFV
        // never reads AKMS under BEP).  gpuwm removes the WHOLE of YSU's own
        // drag here -- WRF's line without the frc_urb1d factor -- so the
        // surface drag enters once, through a_u_bep, as heat already does.
        // WRF's own driver passes ctopo = 1 (module_bl_ysu.F:404), and so
        // does this kernel now: the removal is spelled on WRF's bracket in
        // the ctopo arm above.  WRF's stock line keeps the frc_urb1d factor:
        // `__fsub_rn(diag[0], __fmul_rn(bep.frc[col], __fadd_rn(drag_v, drag_c)))`.
    }
    rhs[0] = u0;
    // diag[k] and rhs[k] ride in registers from the level that set them
    // to the level that finishes them (the same words, not reloaded).
    diag_k = diag[0];
    rhs_k = rhs[0];
    for (int k = 0; k < nz - 1; ++k) {
        real dtodsd, dtodsu;
        real rhs_k1, up, lo;
        if constexpr (BEP) {
            // bl_ysu.F90:1050-1051 (and :1138, :1326): sfk2d(i,k)*dt2/del.
            real sfk = bep.sf[k * st + col];
            dtodsd = sfk * dt2 / delp[k];
            dtodsu = sfk * dt2 / delp[k + 1];
        } else {
            dtodsd = dt2 / delp[k], dtodsu = dt2 / delp[k + 1];
        }
        real dsig = p[k * st + col] - p[(k + 1) * st + col];
        real rdz = 1.0f / dza[k + 1];
        real xk_k = xkzm[k];
        real tem1 = dsig * xk_k * rdz;
        if (pblflg && k + 1 < kpbl) {
            real flux = tem1 * (-hgamu / hpbl - ufxpbl * zfacent[k] / xk_k);
            rhs_k += dtodsd * flux;
            rhs_k1 = u[(k + 1) * st + col] - dtodsu * flux;
        } else if (pblflg && k + 1 >= kpbl && entfac[k] < 4.6f) {
            xk_k = ysu_clip(sqrtf(ysu_max(xkzh[k] * xkzml[k], 0.0f)),
                            xkzminm, xkzmax);
            xkzm[k] = xk_k;
            rhs_k1 = u[(k + 1) * st + col];
        } else rhs_k1 = u[(k + 1) * st + col];
        real dsdz2 = dsig * xk_k * rdz * rdz;
        if constexpr (BEP) {
            // bl_ysu.F90:1070-1071: au = -dtodsd*dsdz2/vlk2d(i,k),
            // al = -dtodsu*dsdz2/vlk2d(i,k) -- both divide by level k.
            real vlk = bep.vl[k * st + col];
            up = -dtodsd * dsdz2 / vlk;
            lo = -dtodsu * dsdz2 / vlk;
        } else {
            up = -dtodsd * dsdz2;
            lo = -dtodsu * dsdz2;
        }
        upper[k] = up;
        lower[k + 1] = lo;
        diag[k] = diag_k - up;          // diag[k] -= upper[k]
        rhs[k] = rhs_k;
        diag_k = 1.0f - lo;             // diag[k+1] = 1 - lower[k+1]
        rhs_k = rhs_k1;
    }
    diag[nz - 1] = diag_k;
    rhs[nz - 1] = rhs_k;
    if constexpr (BEP) {
        // bl_ysu.F90:1359-1368: ad1 = ad; ad -= a_u*dt2; ad1 -= a_v*dt2;
        // f1 += b_u*dt2 (f2 += b_v*dt2 is added after the v RHS rebuild
        // below).  entfac is dead after the assembly above, so it holds
        // ad1, the v diagonal tridi2n solves with.
        for (int k = 0; k < nz; ++k) {
            int q = k * st + col;
            entfac[k] = diag[k] - __fmul_rn(bep.a_v[q], dt2);
            diag[k] = diag[k] - __fmul_rn(bep.a_u[q], dt2);
            rhs[k] = rhs[k] + __fmul_rn(bep.b_u[q], dt2);
        }
    }
    ysu_thomas(lower, diag, upper, rhs, gamma, nz);
    for (int k = 0; k < nz; ++k) du[k * st + col] = (rhs[k] - u[k * st + col]) * rdt;
    for (int k = 0; k < nz; ++k) rhs[k] = v[k * st + col];
    // tridi2n assembles distinct u/v countergradient right-hand sides while
    // sharing one matrix.  Recreate the v RHS after the u solve.
    for (int k = 0; k < nz - 1; ++k) {
        if (pblflg && k + 1 < kpbl) {
            real dtodsd, dtodsu;
            if constexpr (BEP) {
                real sfk = bep.sf[k * st + col];
                dtodsd = sfk * dt2 / delp[k];
                dtodsu = sfk * dt2 / delp[k + 1];
            } else {
                dtodsd = dt2 / delp[k], dtodsu = dt2 / delp[k + 1];
            }
            real dsig = p[k * st + col] - p[(k + 1) * st + col];
            real rdz = 1.0f / dza[k + 1];
            real tem1 = dsig * xkzm[k] * rdz;
            real flux = tem1 * (-hgamv / hpbl - vfxpbl * zfacent[k] / xkzm[k]);
            rhs[k] += dtodsd * flux;
            rhs[k + 1] = v[(k + 1) * st + col] - dtodsu * flux;
        }
    }
    if constexpr (BEP) {
        for (int k = 0; k < nz; ++k)
            rhs[k] = rhs[k] + __fmul_rn(bep.b_v[k * st + col], dt2);
        ysu_tridi2n_v(lower, entfac, gamma, rhs, nz);
    } else {
        ysu_thomas(lower, diag, upper, rhs, gamma, nz);
    }
    for (int k = 0; k < nz; ++k) dv[k * st + col] = (rhs[k] - v[k * st + col]) * rdt;
    if constexpr (TOPO) ysu_topo_blend_u10(topo, u10, v10, u0, v0, col);

    exch_h[col] = exch_m[col] = 0.0f;
    for (int k = 1; k < nz; ++k) {
        exch_h[k * st + col] = xkzh[k - 1];
        exch_m[k * st + col] = xkzm[k - 1];
    }
    hpbl_out[col] = hpbl;
    kpbl_out[col] = kpbl;
    wstar_out[col] = wstar;
    delta_out[col] = delta;
    topdown_radsum_out[col] = topdown_radsum;
    wstar3_2_out[col] = wstar3_2;
    cloudflg_out[col] = cloudflg ? 1 : 0;
}

// The slot pointers are formed here, exactly where the workspace layout
// put them before (lane-interleaved, YSUWS_LANES floats apart per level).
template <bool BEP, bool TOPO = false>
__device__ __forceinline__
void ysu_column_body(const real *__restrict__ u, const real *__restrict__ v, const real *__restrict__ theta,
                const real *__restrict__ qv, const real *__restrict__ qc, const real *__restrict__ qi,
                const real *__restrict__ p, const real *__restrict__ p_interface, const real *__restrict__ exner,
                const real *__restrict__ dz, const real *__restrict__ rthraten,
                const real *__restrict__ psfc, const real *__restrict__ znt,
                const real *__restrict__ ust, const real *__restrict__ hfx, const real *__restrict__ qfx,
                const real *__restrict__ wspd, const real *__restrict__ br, const real *__restrict__ psim,
                const real *__restrict__ psih, const real *__restrict__ xland, const real *u10,
                const real *v10, real *__restrict__ du, real *__restrict__ dv, real *__restrict__ dtheta,
                real *__restrict__ dqv, real *__restrict__ dqc, real *__restrict__ dqi, real *__restrict__ hpbl_out,
                int *__restrict__ kpbl_out, real *__restrict__ exch_h, real *__restrict__ exch_m,
                real *__restrict__ wstar_out, real *__restrict__ delta_out,
                real dt, real *__restrict__ topdown_radsum_out,
                real *__restrict__ wstar3_2_out, int *__restrict__ cloudflg_out,
                int ysu_topdown_pblmix, int nz, int ny, int nx,
                real *__restrict__ ws, int wskp, int col0, const YsuBep bep, const YsuTopo topo = YsuTopo{}) {
    real *wsb = YSUWS_LANE_BASE(ws, wskp);
    ysu_column_slots<BEP, TOPO>(u, v, theta, qv, qc, qi, p, p_interface, exner, dz, rthraten, psfc, znt, ust, hfx, qfx, wspd, br, psim, psih, xland, u10, v10, du, dv, dtheta, dqv, dqc, dqi, hpbl_out, kpbl_out, exch_h, exch_m, wstar_out, delta_out, dt, topdown_radsum_out, wstar3_2_out, cloudflg_out, ysu_topdown_pblmix, nz, ny, nx, ws, wskp, col0, bep, topo,
        YSUWS_AT(wsb, 0, wskp).p, YSUWS_AT(wsb, 1, wskp).p, YSUWS_AT(wsb, 2, wskp).p, YSUWS_AT(wsb, 3, wskp).p,
        YSUWS_AT(wsb, 4, wskp).p, YSUWS_AT(wsb, 5, wskp).p, YSUWS_AT(wsb, 6, wskp).p, YSUWS_AT(wsb, 7, wskp).p,
        YSUWS_AT(wsb, 8, wskp).p, YSUWS_AT(wsb, 9, wskp).p, YSUWS_AT(wsb, 10, wskp).p, YSUWS_AT(wsb, 11, wskp).p,
        YSUWS_AT(wsb, 12, wskp).p, YSUWS_AT(wsb, 13, wskp).p, YSUWS_AT(wsb, 14, wskp).p, YSUWS_AT(wsb, 15, wskp).p,
        YSUWS_AT(wsb, 16, wskp).p, YSUWS_AT(wsb, 17, wskp).p);
}

extern "C" __global__ __launch_bounds__(32, 16)
void ysu_column(const real *u, const real *v, const real *theta,
                const real *qv, const real *qc, const real *qi,
                const real *p, const real *p_interface, const real *exner,
                const real *dz, const real *rthraten,
                const real *psfc, const real *znt,
                const real *ust, const real *hfx, const real *qfx,
                const real *wspd, const real *br, const real *psim,
                const real *psih, const real *xland, const real *u10,
                const real *v10, real *du, real *dv, real *dtheta,
                real *dqv, real *dqc, real *dqi, real *hpbl_out,
                int *kpbl_out, real *exch_h, real *exch_m,
                real *wstar_out, real *delta_out,
                real dt, real *topdown_radsum_out,
                real *wstar3_2_out, int *cloudflg_out,
                int ysu_topdown_pblmix, int nz, int ny, int nx,
                real *ws, int wskp, int col0) {
    ysu_column_body<false, true>(u, v, theta, qv, qc, qi, p, p_interface, exner, dz, rthraten, psfc, znt, ust, hfx, qfx, wspd, br, psim, psih, xland, u10, v10, du, dv, dtheta, dqv, dqc, dqi, hpbl_out, kpbl_out, exch_h, exch_m, wstar_out, delta_out, dt, topdown_radsum_out, wstar3_2_out, cloudflg_out, ysu_topdown_pblmix, nz, ny, nx, ws, wskp, col0, YsuBep{}, YsuTopo{});
}

// sf_urban_physics 2/3: the same column with WRF's flag_bep = .true. arm.
// BEP arrays are (nz, ny, nx) mass levels (sf_bep is nz+1 in the state; the
// first nz are read, as module_bl_ysu.F:360-375 copies kts:kte), frc (ny, nx).
extern "C" __global__
void ysu_column_bep(const real *u, const real *v, const real *theta,
                const real *qv, const real *qc, const real *qi,
                const real *p, const real *p_interface, const real *exner,
                const real *dz, const real *rthraten,
                const real *psfc, const real *znt,
                const real *ust, const real *hfx, const real *qfx,
                const real *wspd, const real *br, const real *psim,
                const real *psih, const real *xland, const real *u10,
                const real *v10, real *du, real *dv, real *dtheta,
                real *dqv, real *dqc, real *dqi, real *hpbl_out,
                int *kpbl_out, real *exch_h, real *exch_m,
                real *wstar_out, real *delta_out,
                real dt, real *topdown_radsum_out,
                real *wstar3_2_out, int *cloudflg_out,
                int ysu_topdown_pblmix, int nz, int ny, int nx,
                real *ws, int wskp, int col0,
                const real *a_u_bep, const real *a_v_bep,
                const real *a_t_bep, const real *a_q_bep,
                const real *b_u_bep, const real *b_v_bep,
                const real *b_t_bep, const real *b_q_bep,
                const real *sf_bep, const real *vl_bep,
                const real *frc_urb2d) {
    YsuBep bep{a_u_bep, a_v_bep, a_t_bep, a_q_bep, b_u_bep, b_v_bep,
               b_t_bep, b_q_bep, sf_bep, vl_bep, frc_urb2d};
    ysu_column_body<true, true>(u, v, theta, qv, qc, qi, p, p_interface, exner, dz, rthraten, psfc, znt, ust, hfx, qfx, wspd, br, psim, psih, xland, u10, v10, du, dv, dtheta, dqv, dqc, dqi, hpbl_out, kpbl_out, exch_h, exch_m, wstar_out, delta_out, dt, topdown_radsum_out, wstar3_2_out, cloudflg_out, ysu_topdown_pblmix, nz, ny, nx, ws, wskp, col0, bep, YsuTopo{});
}

// topo_wind 1/2: the same column with ctopo/ctopo2 present (bl_ysu.F90
// :1254-1314, :1402-1408).  ctopo, ctopo2 (ny, nx) come from
// terrain_drag.cu::topo_wind_static; u10o/v10o receive the blended 10 m wind
// (u10/v10 stay the surface layer's, which the ocean branch reads).
extern "C" __global__
void ysu_column_topo(const real *u, const real *v, const real *theta,
                const real *qv, const real *qc, const real *qi,
                const real *p, const real *p_interface, const real *exner,
                const real *dz, const real *rthraten,
                const real *psfc, const real *znt,
                const real *ust, const real *hfx, const real *qfx,
                const real *wspd, const real *br, const real *psim,
                const real *psih, const real *xland, const real *u10,
                const real *v10, real *du, real *dv, real *dtheta,
                real *dqv, real *dqc, real *dqi, real *hpbl_out,
                int *kpbl_out, real *exch_h, real *exch_m,
                real *wstar_out, real *delta_out,
                real dt, real *topdown_radsum_out,
                real *wstar3_2_out, int *cloudflg_out,
                int ysu_topdown_pblmix, int nz, int ny, int nx,
                real *ws, int wskp, int col0,
                const real *ctopo, const real *ctopo2,
                real *u10o, real *v10o) {
    YsuTopo topo{ctopo, ctopo2, u10o, v10o};
    ysu_column_body<false, true>(u, v, theta, qv, qc, qi, p, p_interface, exner, dz, rthraten, psfc, znt, ust, hfx, qfx, wspd, br, psim, psih, xland, u10, v10, du, dv, dtheta, dqv, dqc, dqi, hpbl_out, kpbl_out, exch_h, exch_m, wstar_out, delta_out, dt, topdown_radsum_out, wstar3_2_out, cloudflg_out, ysu_topdown_pblmix, nz, ny, nx, ws, wskp, col0, YsuBep{}, topo);
}
