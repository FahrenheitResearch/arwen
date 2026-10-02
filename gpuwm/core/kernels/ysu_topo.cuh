// YSU's topo_wind arm (kernels/ysu.cu ysu_column_topo), prepended to
// ysu.cu by the kernel loader (gpuwm/core/kernels/__init__.py
// _EXTRA_HEADERS["ysu"]) after glibc_flt32.cuh.

// --------------------------------------------------------------------------
// topo_wind arm (WRF v4.7.1 &physics topo_wind = 1 or 2): bl_ysu.F90 with
// ctopo/ctopo2 present, MMM-physics 20240626-MPASv8.2.
//   ctopo  scales the first-level surface drag in the momentum solve,
//          weighted by how convective the column is (:1254-1314: the paj TKE
//          profile, the hybrid PBL height of get_pblh, and the Beljaars
//          convective velocity vconv);
//   ctopo2 blends the 10 m wind toward the first-level wind on hill tops
//          (:1402-1408), which the kernel writes to u10o/v10o.
// Both come from gpuwm/core/kernels/terrain_drag.cu::topo_wind_static
// (start_em.F:1579-1626).  The default kernel (TOPO = false) compiles from
// exactly the statements it always had; everything in ysu.cu is
// `if constexpr`.  This header holds the arm's own pieces so that ysu.cu's
// lines before its momentum assembly keep their numbers (the physics
// registry and the FTZ claim census cite them).
// The new statements spell every product and sum as an IEEE intrinsic (WRF
// rounds each; this module lets NVRTC fuse otherwise) and take glibc's powf
// (gfk_pow) and the correctly rounded tanh glibc 2.43 ships.
struct YsuTopo {
    const real *ctopo, *ctopo2;
    real *u10o, *v10o;
};

// bl_ysu.F90 get_pblh (:1586-1693, "Copied from MYNN PBL"), one column, on
// the kernel's 0-based workspace: thv[k] is thetav1d(k+1), tke[k] is
// qke1d(k+1) (written for k <= nz-2, which is all WRF writes and all the
// two searches below can reach), zq[k] is zw1d(k+1), and dz1d(k) is WRF's
// dzq(i,k) = zq(i,k+1)-zq(i,k) (:522), formed here from zq the same way.
template <class Col>
__device__ real ysu_get_pblh(Col thv, Col tke, Col zq, int nz, real landsea) {
    const real sbl_lim = 200.0f, sbl_damp = 400.0f;
    // Fortran index helpers (1-based, kts = 1, kte = nz).
    #define YT_THV(k) thv[(k) - 1]
    #define YT_QKE(k) tke[(k) - 1]
    #define YT_ZW(k) zq[(k) - 1]
    #define YT_DZ(k) __fsub_rn(zq[(k)], zq[(k) - 1])
    const int kte = nz, kts = 1;
    int k = kts + 1, kthv = 1, ktke = 1;
    real maxqke = 0.0f, minthv = 9.e9f;
    while (k <= kte && YT_ZW(k) <= 500.0f) {
        real qtke = fmaxf(YT_QKE(k), 0.0f);
        if (maxqke < qtke) { maxqke = qtke; ktke = k; }
        if (minthv > YT_THV(k)) { minthv = YT_THV(k); kthv = k; }
        k = k + 1;
    }
    real tkeeps = __fdiv_rn(maxqke, 40.0f);
    tkeeps = fmaxf(tkeeps, 0.025f);
    tkeeps = fminf(tkeeps, 0.25f);
    real delt_thv = (__fsub_rn(landsea, 1.5f) >= 0.0f) ? 0.75f : 1.5f;
    real zi = 0.0f;
    k = kthv + 1;
    real thr = __fadd_rn(minthv, delt_thv);
    while (zi == 0.0f && k <= kte) {
        if (YT_THV(k) >= thr) {
            real frac = fminf(__fdiv_rn(__fsub_rn(YT_THV(k), thr),
                                        fmaxf(__fsub_rn(YT_THV(k), YT_THV(k - 1)),
                                              1e-6f)), 1.0f);
            zi = __fsub_rn(YT_ZW(k), __fmul_rn(YT_DZ(k - 1), frac));
        }
        k = k + 1;
        if (k == kte - 1) zi = YT_ZW(kts + 1);
    }
    real pblh_tke = 0.0f;
    k = ktke + 1;
    while (pblh_tke == 0.0f && k <= kte) {
        real qtke = fmaxf(__fdiv_rn(YT_QKE(k), 2.0f), 0.0f);
        real qtkem1 = fmaxf(__fdiv_rn(YT_QKE(k - 1), 2.0f), 0.0f);
        if (qtke <= tkeeps) {
            real frac = fminf(__fdiv_rn(__fsub_rn(tkeeps, qtke),
                                        fmaxf(__fsub_rn(qtkem1, qtke), 1e-6f)),
                              1.0f);
            pblh_tke = __fsub_rn(YT_ZW(k), __fmul_rn(YT_DZ(k - 1), frac));
            pblh_tke = fmaxf(pblh_tke, YT_ZW(kts + 1));
        }
        k = k + 1;
        if (k == kte - 1) pblh_tke = YT_ZW(kts + 1);
    }
    real th = __double2float_rn(tanh((double)__fdiv_rn(__fsub_rn(zi, sbl_lim),
                                                        sbl_damp)));
    real wt = __fadd_rn(__fmul_rn(0.5f, th), 0.5f);
    #undef YT_THV
    #undef YT_QKE
    #undef YT_ZW
    #undef YT_DZ
    return __fadd_rn(__fmul_rn(pblh_tke, __fsub_rn(1.0f, wt)),
                     __fmul_rn(zi, wt));
}

// bl_ysu.F90:1402-1408: u10 = ctopo2*u10 + (1-ctopo2)*ux(1), the same for v,
// with ux(1), vx(1) the first-level wind the call was handed.  ctopo2 = 1
// everywhere but on hill tops.
__device__ __forceinline__ void ysu_topo_blend_u10(
        const YsuTopo &topo, const real *u10, const real *v10, real u1,
        real v1, int col) {
    real c2 = topo.ctopo2[col];
    topo.u10o[col] = __fadd_rn(__fmul_rn(c2, u10[col]),
                               __fmul_rn(__fsub_rn(1.0f, c2), u1));
    topo.v10o[col] = __fadd_rn(__fmul_rn(c2, v10[col]),
                               __fmul_rn(__fsub_rn(1.0f, c2), v1));
}

