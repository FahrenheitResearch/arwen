// gpuwm/core/kernels/thompson_aerosol_sat.cu
//
// WRF v4.6.1 aerosol-aware Thompson (mp_physics=28): the saturation
// adjustment.  Numerical authority is
// WRF v4.6.1 phys/module_mp_thompson.F, commit
// d66e442fccc04111067e29274c9f9eaccc3cef28, zero local modifications.  Every
// bare line number below refers to that file.
//
// Two kernels live here:
//
//   thompson_aa_saturation_adjust  -- :3399-3494, the condensation /
//       evaporation block.  Carries CCN activation (activ_ncloud), the
//       aerosol-ONLY droplet-evaporation branch that reads tnc_wev, and the
//       one-for-one CCN return.
//   thompson_aa_rain_evaporation   -- :3236-3255 + :3384-3388 + :3500-3574,
//       a direct port including the mean-volume-diameter clamp ArWen's mp=8
//       kernel does not carry, plus WRF's `nwfaten += pnr_rev`.  It ALSO owns
//       the :3237-vs-:3568 decision for the working rain density the caller
//       hands to rain sedimentation -- see "WHICH DENSITY LEAVES THIS KERNEL"
//       above its `reference_density` write; that output is not scratch.
//
// BOTH ARE BITWISE AGAINST WRF, and that is measured rather than asserted by
// construction.  An instrumented copy of the pristine module -- WRITE
// statements only, verified inert by reproducing all 38 committed CSVs under
// gpuwm/data/thompson/oracle-aero/ byte for byte -- dumps the raw float32
// and float64 bit patterns of WRF's own working column.  Driven on those:
//
//   prw_vcd, ncten, nwfaten and qc out of the condensation block agree
//   BITWISE at all 122 (fixture, level) pairs where WRF enters :3401's gate,
//   across the eight fixtures that reach it.
//   prv_rev, pnr_rev and the mvd_r-clamped rain number agree to 3.0e-16
//   (double round-off) or exactly, at all 15 pairs where WRF's rain
//   evaporation runs.
//
// The two residuals that remain are the port's documented in-place-versus-
// accumulate regrouping, not this file's arithmetic: WRF forms
// qv = qv1d + DT*qvten_total and temp = t1d + DT*tten_total from ACCUMULATED
// tendencies (:3488-3489) where a state-carrying port adds this stage's
// delta to a value that already absorbed the earlier stages'.  Measured: qv
// differs at 4 of 122 pairs and temperature at 5, one float32 ulp each,
// <= 8.5e-08 relative.  Every mass and number field is exact.
//
// ---------------------------------------------------------------------------
// WHY THIS IS NOT AN EXTENSION OF thompson_cloud_saturation_adjust_impl
// ---------------------------------------------------------------------------
// The aerosol path changes the MASS answer, not only the number bookkeeping:
// line 3467 applies `prw_vcd = MAX(-rc*0.99*orho*odt, prw_vcd)` on the
// is_aerosol_aware branch only, so a subsaturated cloudy cell can never
// evaporate more than 99% of its liquid in one step under mp=28 while mp=8
// evaporates to R1.  thompson.cu is byte-frozen anyway; this file is a
// separate nvrtc translation unit and cannot perturb it.
//
// ---------------------------------------------------------------------------
// ACCUMULATOR CONTRACT
// ---------------------------------------------------------------------------
// nc / nwfa entry state is READ-ONLY.  This file writes ONLY the ncten and
// nwfaten scratch accumulators (per kg per second, exactly WRF's units) and
// the in-place mass/temperature state.  The single terminal apply with WRF's
// clamps (:3972-4021) belongs to thompson_aerosol_state.cu.  The working
// per-m3 droplet number is recomputed locally the way WRF does at :3216-3223,
// never read back from state.
//
// ---------------------------------------------------------------------------
// FLOAT CONTRACTION
// ---------------------------------------------------------------------------
// The alphsc / t1_evap / Dc_star chain selects idx_d by INT TRUNCATION of
// 1e6*Dc_star, so a single fused-multiply-add rounding can move a bin.  That
// chain is written with thompson_aerosol_common.cuh's __fmul_rn/__fadd_rn
// helpers, which reproduce `gfortran -O2` on baseline x86-64 (no FMA)
// operation for operation.
//
// THE NEWTON SOLVE IS PINNED FOR THE SAME REASON, AND MEASURED.  This file
// originally kept thompson.cu:250-257's plain expressions so that mp=8 and
// mp=28 would evaluate the shared condensation solve identically.  That is
// the wrong tie-break -- the same one the shared header already rejected for
// RSLF/RSIF -- and here it is not a sub-ulp curiosity.  :3405 evaluates
//
//     fcd = qvs(k)*EXP(lvt2(k)*clap) - qv(k) + clap
//
// where qvs*EXP(lvt2*clap) cancels qv to about four significant digits, so
// fcd carries an absolute noise floor of one ulp of qv (~1e-9 at
// qv = 9.7e-3).  Newton converges to the point where fcd/dfcd stops moving
// clap, which fixes clap only to about 1e-9/dfcd ~ 4e-10 absolute -- roughly
// 5e-5 RELATIVE on a clap of 7.5e-6.  clap IS the condensed mass: :3412
// sets prw_vcd = clap*odt, and :3975's qc1d = qc1d + qcten*DT multiplies the
// odt straight back out.  Two independent float32 details each move it by
// that much:
//
//   (a) nvrtc contracts `qvs*e - qv0 + clap` into an FMA and `qvs*lvt2*e +
//       1` into another; `gfortran -O2` on baseline x86-64 has no FMA
//       instruction and rounds every operation separately.
//   (b) CUDA's expf is a ~2-ulp device approximation; gfortran's REAL(4)
//       EXP calls glibc expf, which is faithful to ~0.5 ulp.  Rounding a
//       double-precision exp to float reproduces it.
//
// MEASURED on the aero-ccn-activate entry column (24 levels), against an
// instrumented copy of the pristine Fortran that dumps clap as a raw
// float32 bit pattern at :3408:
//
//     plain chain + expf              24 / 24 levels differ, worst 5.42e-05
//     plain chain + double exp        24 / 24 levels differ
//     pinned chain + expf             24 / 24 levels differ
//     pinned chain + double exp        0 / 24 levels differ  (BIT-EXACT)
//
// Neither change alone is sufficient; both together are exact.  So the
// condensation half is pinned too, and mp=8's plain form stays frozen.
// See tests/test_thompson_aerosol_sat_gpu.py::
// test_newton_condensation_solve_is_bitwise_against_the_instrumented_oracle.
//
// The shared header is prepended by gpuwm/core/kernels/__init__.py's
// _EXTRA_HEADERS allow-list; there is no #include of it below.

// module_mp_thompson.F:185.  The gate epsilon for ssatw and clap; distinct
// from the 1.E-6 second gate on ssatw at :3423 and from the 1.E-9 xsat snap
// at :3437.
#define THOMPSON_AA_SAT_EPS       1.0e-15f
// :3423, the aerosol-only evaporation branch's second gate.
#define THOMPSON_AA_SAT_SSAT_GATE 1.0e-6f
// :3437, `if (abs(xsat).lt. 1.E-9) xsat=0.` -- note this is a SNAP TO ZERO,
// while the rain-evaporation block at :3529 uses MIN(-1.E-9, ssatw) instead.
#define THOMPSON_AA_SAT_XSAT_SNAP 1.0e-9f
#define THOMPSON_AA_ORV           (1.0f / 461.5f)   // oRv, Rv = 461.5
#define THOMPSON_AA_CP            1004.0f           // Cp
#define THOMPSON_AA_LVAP0         2.5e6f            // :217
#define THOMPSON_AA_RHO_W         1000.0f           // :70
#define THOMPSON_AA_RC1           1.0e-6f           // r_c(1), :838
#define THOMPSON_AA_NIC2          (-6)              // NINT(ALOG10(r_c(1))), :820
// rho_not (:69), `101325.0/(287.05*298.0)`.  Written unfolded, not as a
// pre-rounded literal.  gfortran folds PARAMETER expressions round-by-round
// in the declared kind and so does nvrtc here, and the two agree
// (0x3F979E63, verified against the instrumented oracle's own dump); a
// hand-rounded decimal constant moved the answer by 7.6e-07.
#define THOMPSON_AA_RHO_NOT       (101325.0f / (287.05f * 298.0f))


// ---------------------------------------------------------------------------
// Per-cell thermodynamic refresh, module_mp_thompson.F:3199-3210.
// ---------------------------------------------------------------------------
//
// WRF recomputes this block from the post-network temperature and vapour
// immediately before the condensation loop.  ArWen's decomposition carries
// state rather than tendencies, so the incoming temperature/qv ARE
// `t1d + DT*tten` and `MAX(1.E-10, qv1d + DT*qvten)`.
struct thompson_aa_sat_env {
    float rho;
    float qvs;
    float ssatw;
    float lvap;
    float ocp;
    float lvt2;
    float diffu;
    float tcond;
    float otemp;
};

__device__ __forceinline__ thompson_aa_sat_env thompson_aa_sat_environment(
    float temp0, float pres, float qv0)
{
    thompson_aa_sat_env e;
    e.otemp = thompson_aa_div(1.0f, temp0);
    const float tempc = thompson_aa_sub(temp0, 273.15f);
    // :3193.  ((0.622*pres) / ((R*temp)*(qv+0.622))) -- no multiply feeds an
    // add, so contraction cannot reach this one; pinned only for uniformity.
    e.rho = thompson_aa_div(
        thompson_aa_mul(0.622f, pres),
        thompson_aa_mul(thompson_aa_mul(THOMPSON_AA_R_DRY, temp0),
                        thompson_aa_add(qv0, 0.622f)));
    e.qvs = thompson_rslf(pres, temp0);
    float ssatw = thompson_aa_sub(thompson_aa_div(qv0, e.qvs), 1.0f);
    if (fabsf(ssatw) < THOMPSON_AA_SAT_EPS) ssatw = 0.0f;
    e.ssatw = ssatw;
    // :3199.  The correctly-rounded helper, NOT thompson.cu:2199-2201's
    // powf.  MEASURED over the 122 (fixture, level) pairs where WRF runs
    // the condensation block: CUDA's powf misses gfortran's REAL(4) ** at
    // 2 of them (aero-cold-overlap levels 8 and 9, one ulp each) while a
    // double-precision pow rounded to float hits all 122.  Same family of
    // defect as the expf below.
    e.diffu = thompson_aa_mul(
        thompson_aa_mul(2.11e-5f,
                        thompson_aa_powf_cr(
                            thompson_aa_div(temp0, 273.15f), 1.94f)),
        thompson_aa_div(101325.0f, pres));
    // :3206-3209.  Each of these has a multiply feeding an add and would be
    // contracted by nvrtc; the Fortran rounds them separately.
    e.lvap = thompson_aa_add(THOMPSON_AA_LVAP0,
                             thompson_aa_mul(2106.0f - 4218.0f, tempc));
    e.tcond = thompson_aa_mul(
        thompson_aa_mul(thompson_aa_add(5.69f,
                                        thompson_aa_mul(0.0168f, tempc)),
                        1.0e-5f),
        418.936f);
    e.ocp = thompson_aa_div(
        1.0f,
        thompson_aa_mul(THOMPSON_AA_CP,
                        thompson_aa_add(1.0f,
                                        thompson_aa_mul(0.887f, qv0))));
    e.lvt2 = thompson_aa_mul(
        thompson_aa_mul(
            thompson_aa_mul(
                thompson_aa_mul(thompson_aa_mul(e.lvap, e.lvap), e.ocp),
                THOMPSON_AA_ORV),
            e.otemp),
        e.otemp);
    return e;
}


// ---------------------------------------------------------------------------
// Aerosol-only droplet evaporation, module_mp_thompson.F:3423-3471.
// ---------------------------------------------------------------------------
//
// Returns pnc_wcd (per kg per second, negative) and overwrites *prw_vcd with
// WRF's 99%-of-liquid floor.  This is the ONLY consumer of tnc_wev in gpuwm;
// the table has been parsed, SHA-validated and uploaded since the mp=8 port
// and never read, so a latent Fortran/C order defect in the (100,37,100)
// upload would surface here first.  tests/test_thompson_aerosol_sat_gpu.py
// carries an explicit transposition fixture so that failure mode is
// attributed to the table rather than to this kernel.
//
// tpc_wev's use at :3465-3466 is COMMENTED OUT in v4.6.1 and is deliberately
// not implemented; it stays uploaded-but-unread so the pinned classic
// auxiliary asset is unchanged.
__device__ __forceinline__ double thompson_aa_droplet_evaporation(
    const thompson_aa_sat_env& e,
    float rc, float nc_work, float orho, float odt, float dt,
    const double* __restrict__ tnc_wev,
    double* __restrict__ prw_vcd,
    int* __restrict__ out_idx_d,
    int* __restrict__ out_idx_c,
    int* __restrict__ out_idx_n)
{
    // rvs = rho*qvs;  rvs_p = rvs*otemp*(lvap*otemp*oRv - 1.)      :3426-3427
    const float otemp = e.otemp;
    const float rvs = thompson_aa_mul(e.rho, e.qvs);
    const float lv_orv = thompson_aa_mul(
        thompson_aa_mul(e.lvap, otemp), THOMPSON_AA_ORV);
    const float tt = thompson_aa_sub(lv_orv, 1.0f);
    const float rvs_p = thompson_aa_mul(thompson_aa_mul(rvs, otemp), tt);

    // rvs_pp = rvs*( otemp*tt*otemp*tt
    //                + (-2.*lvap*otemp*otemp*otemp*oRv)
    //                + otemp*otemp )                              :3428-3431
    const float pp_a = thompson_aa_mul(
        thompson_aa_mul(thompson_aa_mul(otemp, tt), otemp), tt);
    float pp_b = thompson_aa_mul(-2.0f, e.lvap);
    pp_b = thompson_aa_mul(pp_b, otemp);
    pp_b = thompson_aa_mul(pp_b, otemp);
    pp_b = thompson_aa_mul(pp_b, otemp);
    pp_b = thompson_aa_mul(pp_b, THOMPSON_AA_ORV);
    const float pp_c = thompson_aa_mul(otemp, otemp);
    const float rvs_pp = thompson_aa_mul(
        rvs, thompson_aa_add(thompson_aa_add(pp_a, pp_b), pp_c));

    // gamsc = lvap*diffu/tcond * rvs_p                            :3432
    const float gamsc = thompson_aa_mul(
        thompson_aa_div(thompson_aa_mul(e.lvap, e.diffu), e.tcond), rvs_p);

    // alphsc = 0.5*(g/(1+g))*(g/(1+g)) * rvs_pp/rvs_p * rvs/rvs_p :3433-3435
    const float gr = thompson_aa_div(gamsc, thompson_aa_add(1.0f, gamsc));
    float alphsc = thompson_aa_mul(0.5f, gr);
    alphsc = thompson_aa_mul(alphsc, gr);
    alphsc = thompson_aa_mul(alphsc, rvs_pp);
    alphsc = thompson_aa_div(alphsc, rvs_p);
    alphsc = thompson_aa_mul(alphsc, rvs);
    alphsc = thompson_aa_div(alphsc, rvs_p);
    alphsc = fmaxf(1.0e-9f, alphsc);

    // xsat SNAPS to zero inside a 1e-9 band here (:3436-3437); the rain
    // block at :3529 clips with MIN(-1.E-9, ssatw) instead.  Not the same.
    float xsat = e.ssatw;
    if (fabsf(xsat) < THOMPSON_AA_SAT_XSAT_SNAP) xsat = 0.0f;

    // t1_evap = 2*PI*(1 - a*x + 2*a*a*x*x - 5*a*a*a*x*x*x)/(1+g)  :3438-3441
    const float ax = thompson_aa_mul(alphsc, xsat);
    float t2 = thompson_aa_mul(2.0f, alphsc);
    t2 = thompson_aa_mul(t2, alphsc);
    t2 = thompson_aa_mul(t2, xsat);
    t2 = thompson_aa_mul(t2, xsat);
    float t3 = thompson_aa_mul(5.0f, alphsc);
    t3 = thompson_aa_mul(t3, alphsc);
    t3 = thompson_aa_mul(t3, alphsc);
    t3 = thompson_aa_mul(t3, xsat);
    t3 = thompson_aa_mul(t3, xsat);
    t3 = thompson_aa_mul(t3, xsat);
    float paren = thompson_aa_sub(1.0f, ax);
    paren = thompson_aa_add(paren, t2);
    paren = thompson_aa_sub(paren, t3);
    const float two_pi = thompson_aa_mul(2.0f, THOMPSON_AA_PI);
    float t1_evap = thompson_aa_mul(two_pi, paren);
    t1_evap = thompson_aa_div(t1_evap, thompson_aa_add(1.0f, gamsc));

    // Dc_star = DSQRT(-2.D0*DT * t1_evap/(2.*PI)
    //                 * 4.*diffu*ssatw*rvs/rho_w)                 :3443-3444
    // The leading -2.D0 promotes the whole chain to double, left to right.
    double arg = -2.0 * (double)dt;
    arg = arg * (double)t1_evap;
    arg = arg / (double)two_pi;
    arg = arg * 4.0;
    arg = arg * (double)e.diffu;
    arg = arg * (double)e.ssatw;
    arg = arg * (double)rvs;
    arg = arg / (double)THOMPSON_AA_RHO_W;
    const double dc_star = sqrt(arg);

    // idx_d is an INT TRUNCATION, not a NINT.  Axis 0 of tnc_wev is droplet
    // diameter with Dc(i) = i micron LINEARLY (:831-836), unlike every other
    // bin family in this scheme.
    const int idx_d = max(1, min((int)(1.0e6 * dc_star), THOMPSON_AA_NBC));
    // :3447-3448.  Zero-based out of the shared helper.
    const int idx_n0 = thompson_aa_droplet_bin(nc_work);
    // :3451-3462.  Cloud-water decade bin, 1e-6 .. 1e-2 kg m-3.
    const int idx_c0 = (rc > THOMPSON_AA_RC1)
        ? thompson_aa_decade_index(rc, THOMPSON_AA_NIC2, THOMPSON_AA_NTB_C)
        : 0;

    if (out_idx_d != nullptr) *out_idx_d = idx_d;
    if (out_idx_c != nullptr) *out_idx_c = idx_c0 + 1;
    if (out_idx_n != nullptr) *out_idx_n = idx_n0 + 1;

    // Fortran-order flat offset into tnc_wev(nbc, ntb_c, nbc).
    const int flat = (idx_d - 1)
        + THOMPSON_AA_NBC * (idx_c0 + THOMPSON_AA_NTB_C * idx_n0);
    const double tnc = tnc_wev[flat];

    // :3467.  AEROSOL-ONLY: this floor does not exist on the mp=8 path, so
    // the aerosol scheme's evaporated MASS differs from classic Thompson's
    // even when the droplet number is irrelevant.
    *prw_vcd = fmax(
        (double)thompson_aa_mul(
            thompson_aa_mul(thompson_aa_mul(-rc, 0.99f), orho), odt),
        *prw_vcd);
    // :3468-3469.  tnc_wev is REAL(KIND=R8SIZE) (:391-392), so the right
    // operand is double throughout and DBLE() there is a no-op.
    return fmax((double)thompson_aa_mul(
                    thompson_aa_mul(thompson_aa_mul(-nc_work, 0.99f), orho),
                    odt),
                -tnc * (double)orho * (double)odt);
}


// ---------------------------------------------------------------------------
// module_mp_thompson.F:3399-3494.
// ---------------------------------------------------------------------------
//
// nc_entry / nwfa_work_m3 asymmetry, and why it is not smoothed: nwfa_work
// is WRF's SECOND aerosol snapshot, taken at :3211 as
// MAX(11.1E6, (nwfa1d + nwfaten*DT)*rho) with NO upper bound and no nifa
// counterpart, and it is the only nwfa this block sees.  The entry snapshot
// at :1805 (which does carry the 9999e6 ceiling) feeds scavenging and ice
// nucleation instead.  Conflating them changes activated droplet number
// wherever scavenging was significant.
//
// w is the ENTRY vertical velocity.  mp_gt_driver copies w1d(k) = w(i,k,j)
// once at :1224 with no averaging and never refreshes it, so activation sees
// the vertical velocity from before T and qv were advanced.  ArWen's caller
// passes state.w[:-1], the lower full-level slice: WRF's w_2(k) and ArWen's
// w[k] are the same staggered face, and any interface averaging would be a
// different physical input on activ_ncloud's four-decade log axis.
__device__ __forceinline__ void thompson_aa_saturation_adjust_impl(
    float* __restrict__ temperature,
    const float* __restrict__ pressure,
    float* __restrict__ qv,
    float* __restrict__ qc,
    const float* __restrict__ nc_entry,
    float* __restrict__ ncten,
    float* __restrict__ nwfaten,
    const float* __restrict__ nwfa_work_m3,
    const float* __restrict__ w,
    const double* __restrict__ tnccn_act,
    const double* __restrict__ tnc_wev,
    float* __restrict__ reference_density,
    float* __restrict__ reference_temperature,
    float* __restrict__ condensation_rate,
    float dt, int idx)
{
    const float temp0 = temperature[idx];
    const float pres = pressure[idx];
    const float qv0 = fmaxf(1.0e-10f, qv[idx]);
    const float qc0 = qc[idx];

    const thompson_aa_sat_env e = thompson_aa_sat_environment(
        temp0, pres, qv0);

    if (reference_density != nullptr) reference_density[idx] = e.rho;
    if (reference_temperature != nullptr) reference_temperature[idx] = temp0;
    if (condensation_rate != nullptr) condensation_rate[idx] = 0.0f;

    // :3215-3224.  L_qc and the working droplet number.  nc is rebuilt from
    // the FROZEN entry value plus the accumulator, never read back from a
    // mutated state array.
    const bool l_qc = qc0 > THOMPSON_AA_R1;
    const float rho = e.rho;
    const float orho = thompson_aa_div(1.0f, rho);
    const float odt = thompson_aa_div(1.0f, dt);
    const float rc = l_qc ? thompson_aa_mul(qc0, rho) : THOMPSON_AA_R1;
    // :3217.  `(nc1d(k)+ncten(k)*DT)*rho(k)` -- ncten*DT feeds an add.
    const float nc_work = l_qc
        ? fmaxf(THOMPSON_AA_NC_FLOOR,
                fminf(thompson_aa_mul(
                          thompson_aa_add(nc_entry[idx],
                                          thompson_aa_mul(ncten[idx], dt)),
                          rho),
                      THOMPSON_AA_NT_C_MAX))
        : THOMPSON_AA_NC_FLOOR;

    // :3401-3402.
    const float ssatw = e.ssatw;
    if (!(ssatw > THOMPSON_AA_SAT_EPS
            || (ssatw < -THOMPSON_AA_SAT_EPS && l_qc))) {
        return;
    }

    // :3403-3408.  NOT thompson.cu:250-257's plain form -- see "FLOAT
    // CONTRACTION" in the file header for the 5.4e-05 this costs and the
    // 24/24 bit-exact measurement that closes it.  Fortran's evaluation
    // order is reproduced literally: ((qvs*EXP) - qv) + clap for fcd, and
    // ((qvs*lvt2)*EXP) + 1 for dfcd.
    const float lvt2 = e.lvt2;
    const float qvs = e.qvs;
    float clap = thompson_aa_div(
        thompson_aa_sub(qv0, qvs),
        thompson_aa_add(1.0f, thompson_aa_mul(lvt2, qvs)));
    for (int iteration = 0; iteration < 3; ++iteration) {
        const float exponential = thompson_aa_expf_cr(
            thompson_aa_mul(lvt2, clap));
        const float fcd = thompson_aa_add(
            thompson_aa_sub(thompson_aa_mul(qvs, exponential), qv0), clap);
        const float dfcd = thompson_aa_add(
            thompson_aa_mul(thompson_aa_mul(qvs, lvt2), exponential), 1.0f);
        clap = thompson_aa_sub(clap, thompson_aa_div(fcd, dfcd));
    }

    // :3409.  `rc(k) + clap*rho(k)` is a multiply feeding an add.
    const float xrc = thompson_aa_add(rc, thompson_aa_mul(clap, rho));
    double prw_vcd = 0.0;
    double pnc_wcd = 0.0;

    if (xrc > THOMPSON_AA_R1) {
        // :3412.  Assigned from a REAL(4) product into a DOUBLE PRECISION
        // array, so the float rounding happens first.
        prw_vcd = (double)thompson_aa_mul(clap, odt);

        if (clap > THOMPSON_AA_SAT_EPS) {
            // :3414-3420, DROPLET NUCLEATION.  The only consumer of w and
            // of tnccn_act in the whole scheme.
            const float xnc = fmaxf(
                THOMPSON_AA_NC_FLOOR,
                thompson_activ_ncloud(temp0, w[idx], nwfa_work_m3[idx],
                                      tnccn_act));
            const float diff = thompson_aa_sub(xnc, nc_work);
            pnc_wcd = (double)thompson_aa_mul(
                thompson_aa_mul(
                    thompson_aa_mul(0.5f,
                                    thompson_aa_add(diff, fabsf(diff))),
                    odt),
                orho);
        } else if (clap < -THOMPSON_AA_SAT_EPS
                   && ssatw < -THOMPSON_AA_SAT_SSAT_GATE) {
            // :3423-3471, AEROSOL-ONLY.  mp=8 has no counterpart at all.
            pnc_wcd = thompson_aa_droplet_evaporation(
                e, rc, nc_work, orho, odt, dt, tnc_wev, &prw_vcd,
                nullptr, nullptr, nullptr);
        }
    } else {
        // :3472-3475.
        prw_vcd = (double)thompson_aa_mul(
            thompson_aa_mul(-rc, orho), odt);
        pnc_wcd = (double)thompson_aa_mul(
            thompson_aa_mul(-nc_work, orho), odt);
    }

    // :3479-3489.  WRF rounds each DOUBLE rate into a REAL(4) tendency
    // before multiplying by DT; reproduce that rounding, not the double
    // product.
    const float prw = (float)prw_vcd;
    // :3483.  lvap(k)*ocp(k) is a REAL(4)*REAL(4) product, rounded BEFORE
    // the DOUBLE prw_vcd multiplies it.
    const float tten = (float)(
        (double)thompson_aa_mul(e.lvap, e.ocp) * prw_vcd);

    qv[idx] = fmaxf(1.0e-10f,
                    thompson_aa_sub(qv0, thompson_aa_mul(prw, dt)));
    qc[idx] = thompson_aa_add(qc0, thompson_aa_mul(prw, dt));
    temperature[idx] = thompson_aa_add(temp0, thompson_aa_mul(tten, dt));
    ncten[idx] = (float)((double)ncten[idx] + pnc_wcd);
    // THE aerosol return path (:3482): activation consumes one CCN per
    // droplet nucleated, evaporation regenerates one per droplet lost.
    nwfaten[idx] = (float)((double)nwfaten[idx] - pnc_wcd);
    if (condensation_rate != nullptr) condensation_rate[idx] = prw;
}


extern "C" __global__ void thompson_aa_saturation_adjust(
    float* __restrict__ temperature,
    const float* __restrict__ pressure,
    float* __restrict__ qv,
    float* __restrict__ qc,
    const float* __restrict__ nc_entry,
    float* __restrict__ ncten,
    float* __restrict__ nwfaten,
    const float* __restrict__ nwfa_work_m3,
    const float* __restrict__ w,
    const double* __restrict__ tnccn_act,
    const double* __restrict__ tnc_wev,
    float* __restrict__ reference_density,
    float* __restrict__ reference_temperature,
    float* __restrict__ condensation_rate,
    float dt, int size)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= size) return;
    thompson_aa_saturation_adjust_impl(
        temperature, pressure, qv, qc, nc_entry, ncten, nwfaten,
        nwfa_work_m3, w, tnccn_act, tnc_wev, reference_density,
        reference_temperature, condensation_rate, dt, idx);
}


// ---------------------------------------------------------------------------
// Diagnostic probe of the three droplet-evaporation lookup indices.
// ---------------------------------------------------------------------------
//
// Exists so tests/test_thompson_aerosol_sat_gpu.py can pin (idx_d, idx_c,
// idx_n) and the tnc_wev value the kernel actually read, independently of
// the tendency it produced.  A Fortran/C order defect in the (100,37,100)
// upload then reports as a table bug rather than as new mp=28 physics.
extern "C" __global__ void thompson_aa_droplet_evap_probe(
    const float* __restrict__ temperature,
    const float* __restrict__ pressure,
    const float* __restrict__ qv,
    const float* __restrict__ qc,
    const float* __restrict__ nc_work_m3,
    const double* __restrict__ tnc_wev,
    int* __restrict__ idx_d_out,
    int* __restrict__ idx_c_out,
    int* __restrict__ idx_n_out,
    double* __restrict__ tnc_out,
    double* __restrict__ pnc_wcd_out,
    float dt, int size)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= size) return;
    const float temp0 = temperature[idx];
    const float qv0 = fmaxf(1.0e-10f, qv[idx]);
    const thompson_aa_sat_env e = thompson_aa_sat_environment(
        temp0, pressure[idx], qv0);
    const float qc0 = qc[idx];
    const float rc = (qc0 > THOMPSON_AA_R1)
        ? thompson_aa_mul(qc0, e.rho) : THOMPSON_AA_R1;
    const float orho = thompson_aa_div(1.0f, e.rho);
    const float odt = thompson_aa_div(1.0f, dt);
    double prw_vcd = -1.0e30;
    int id = 0, ic = 0, in = 0;
    const double pnc = thompson_aa_droplet_evaporation(
        e, rc, nc_work_m3[idx], orho, odt, dt, tnc_wev, &prw_vcd,
        &id, &ic, &in);
    idx_d_out[idx] = id;
    idx_c_out[idx] = ic;
    idx_n_out[idx] = in;
    tnc_out[idx] = tnc_wev[(id - 1)
        + THOMPSON_AA_NBC * ((ic - 1) + THOMPSON_AA_NTB_C * (in - 1))];
    pnc_wcd_out[idx] = pnc;
}


// ---------------------------------------------------------------------------
// module_mp_thompson.F:3500-3574 -- rain evaporation with the CCN return.
// ---------------------------------------------------------------------------
//
// This was originally a transcription of ArWen's model-validated mp=8
// thompson_rain_evaporation_impl (thompson.cu:2149-2285) plus WRF's
// `nwfaten += pnr_rev`.  It is now a DIRECT PORT of :3236-3255 + :3384-3388 +
// :3500-3574, for the same reason the shared header pins RSLF/RSIF: the
// authority is WRF, not ArWen's mp=8 sibling.  Three things changed and each
// one is a measured WRF-fidelity gap, not a style preference.  All three were
// measured against an instrumented copy of the pristine module (see
// tests/test_thompson_aerosol_sat_gpu.py::
// test_rain_evaporation_matches_the_instrumented_wrf_oracle for the recipe
// and the pinned oracle values).
//
//   1. THE mvd_r CLAMP, :3242-3250.  WRF bounds the mean volume diameter to
//      [D0r*0.75, 2.5 mm] and REBUILDS the working rain number from the
//      bound before N0_r, ilamr and pnr_rev ever see it.  thompson.cu forms
//      lambda straight from qr/nr and skips this.  The clamp is live in the
//      committed fixtures -- it fires at levels 4, 5 and 6 of
//      aero-cold-overlap (upper bound) and at level 7 of
//      aero-reduces-to-classic (lower bound) -- and moves the working rain
//      number by up to 4.0e-05 relative where it fires.  This was the
//      "KNOWN mp=8 CARRY-OVER, recorded rather than fixed" note that used to
//      stand here.  It is now fixed, because a rain number that differs from
//      WRF's feeds pnr_rev, which feeds nr, which the oracle compares.
//
//   2. Sc3, the Schmidt number to the one third (:663, `Sc3 = Sc**(1./3.)`).
//      CUDA's powf(0.632f, 1/3) returns 0x3F5BB0E8; gfortran's REAL(4) **
//      returns 0x3F5BB0E7, ONE ULP LOWER, and a double-precision pow rounded
//      to float reproduces the Fortran exactly.  That ulp propagates into
//      t2_qr_ev (0x42135208 against WRF's 0x42135207) and straight into the
//      ventilation half of prv_rev.
//
//   3. CONTRACTION.  The rvs_p / rvs_pp / gamsc / alphsc / t1_evap chain is
//      the same chain thompson_aa_droplet_evaporation above already pins,
//      and for the same reason: `gfortran -O2` on baseline x86-64 has no FMA
//      instruction.  Leaving it contracted here while pinning it thirty
//      lines earlier was simply inconsistent.
//
// MEASURED EFFECT of 2+3 together, driving this kernel on WRF's own
// intermediate state at the nine (fixture, level) pairs where WRF's rain
// evaporation actually fires: the pre-existing form disagreed with WRF's
// prv_rev/pnr_rev by 2.0e-08 to 3.9e-07 relative; the form below agrees to
// 0 or 3.0e-16 (double round-off).
//
// WRF's own addition on top of :3500-3574 as mp=8 knows it:
//
//   a. nwfaten += pnr_rev (:3565).  Every fully evaporated raindrop returns
//      exactly one CCN.  Pass `nwfaten = nullptr` to suppress it.
//   b. The optional `condensation_rate` gate reproducing :3502's
//      `.and. (.not.(prw_vcd(k).gt. 0.))` -- WRF skips rain evaporation
//      entirely in a cell that just condensed cloud water.  mp=8's kernel
//      has no such input because its driver sequences the two stages; pass
//      the saturation-adjustment kernel's condensation_rate output here.
//
// UNITS NOTE.  `nr` is per kilogram in ArWen's state and per cubic metre in
// WRF's working column; every clamp below is applied to the per-cubic-metre
// working copy, exactly as WRF does, and only the tendency is written back.

// module_mp_thompson.F:128, `am_r = PI*rho_w/6.0`.  gfortran folds PARAMETER
// expressions ROUND-BY-ROUND in the declared kind, not from the exact
// decimal: the stepwise float32 value is 0x4402E653 and the correctly
// rounded one is 0x4402E652.  0x4402E653 is what the oracle carries.
#define THOMPSON_AA_AM_R   (3.1415926536f * 1000.0f / 6.0f)
#define THOMPSON_AA_CRG3   6.0f            // WGAMMA(bm_r+mu_r+1) = Gamma(4)
#define THOMPSON_AA_ORG2   1.0f            // 1/WGAMMA(mu_r+1) = 1/Gamma(1)
#define THOMPSON_AA_ORG3   (1.0f / 6.0f)   // 1/crg(3)
#define THOMPSON_AA_OBMR   (1.0f / 3.0f)   // 1/bm_r, :721
#define THOMPSON_AA_BM_R   3.0f            // :129
#define THOMPSON_AA_MU_R   0.0f            // :140
#define THOMPSON_AA_D0R    50.0e-6f        // :150
#define THOMPSON_AA_FV_R   195.0f          // :145
#define THOMPSON_AA_AV_R   4854.0f         // :143
#define THOMPSON_AA_SC     0.632f          // :195
#define THOMPSON_AA_R2     1.0e-6f         // :184

// module_mp_thompson.F:3241-3255.  The incoming rain diagnosis, INCLUDING
// the mean-volume-diameter clamp that rebuilds nr.  Returns the working
// per-cubic-metre rain number; *lamr_out receives :3385's slope recomputed
// from whatever number survived the clamp.
__device__ __forceinline__ float thompson_aa_bound_rain_number(
    float rr, float nr_m3, double* __restrict__ lamr_out)
{
    // :3245.  REAL(4) base, REAL(4) exponent -> powf, widened into the
    // DOUBLE PRECISION lamr (:1597).
    float nr_work = fmaxf(THOMPSON_AA_R2, nr_m3);
    double lamr = (double)thompson_aa_powf_cr(
        thompson_aa_div(
            thompson_aa_mul(
                thompson_aa_mul(
                    thompson_aa_mul(THOMPSON_AA_AM_R, THOMPSON_AA_CRG3),
                    THOMPSON_AA_ORG2),
                nr_work),
            rr),
        THOMPSON_AA_OBMR);
    // :3246.  REAL(4) numerator over DOUBLE lamr, rounded into REAL mvd_r.
    const float mvd_num = 3.0f + THOMPSON_AA_MU_R + 0.672f;
    float mvd_r = (float)((double)mvd_num / lamr);
    const bool high = mvd_r > 2.5e-3f;
    const bool low = mvd_r < thompson_aa_mul(THOMPSON_AA_D0R, 0.75f);
    if (high || low) {
        mvd_r = high ? 2.5e-3f : thompson_aa_mul(THOMPSON_AA_D0R, 0.75f);
        // :3249.  REAL(4)/REAL(4), then widened -- NOT a double divide.
        lamr = (double)thompson_aa_div(mvd_num, mvd_r);
        // :3250.  ((crg(2)*org3)*rr) is REAL(4); lamr**bm_r is a DOUBLE
        // pow; the quotient by am_r stays double and rounds on assignment.
        nr_work = (float)(
            (double)thompson_aa_mul(
                thompson_aa_mul(1.0f, THOMPSON_AA_ORG3), rr)
            * pow(lamr, (double)THOMPSON_AA_BM_R)
            / (double)THOMPSON_AA_AM_R);
        // :3385 runs again over the rebuilt number.
        lamr = (double)thompson_aa_powf_cr(
            thompson_aa_div(
                thompson_aa_mul(
                    thompson_aa_mul(
                        thompson_aa_mul(THOMPSON_AA_AM_R, THOMPSON_AA_CRG3),
                        THOMPSON_AA_ORG2),
                    nr_work),
                rr),
            THOMPSON_AA_OBMR);
    }
    *lamr_out = lamr;
    return nr_work;
}

__device__ __forceinline__ void thompson_aa_rain_evaporation_impl(
    float* __restrict__ qr,
    float* __restrict__ nr,
    float* __restrict__ temperature,
    const float* __restrict__ pressure,
    float* __restrict__ qv,
    float* __restrict__ nwfaten,
    float* __restrict__ reference_density,
    float* __restrict__ reference_temperature,
    const float* __restrict__ graupel_melt_marker,
    const float* __restrict__ condensation_rate,
    const float* __restrict__ entry_density,
    double* __restrict__ prv_rev_out,
    double* __restrict__ pnr_rev_out,
    float* __restrict__ nr_bound_out,
    float dt, int idx)
{
    const float temp0 = temperature[idx];
    const float qv0 = fmaxf(1.0e-10f, qv[idx]);
    const float pres = pressure[idx];
    // :3572/:3490 -- the density carried into this loop.
    const float rho = thompson_aa_div(
        thompson_aa_mul(0.622f, pres),
        thompson_aa_mul(thompson_aa_mul(THOMPSON_AA_R_DRY, temp0),
                        thompson_aa_add(qv0, 0.622f)));
    // :3553.  The warm network leaves WRF's held `prr_gml > 0` decision here.
    const bool melting_graupel = graupel_melt_marker != nullptr
        && graupel_melt_marker[idx] > 0.0f;
    // :3242-3243.  TWO DENSITIES ARE IN PLAY AND WRF USES BOTH.  rr and nr
    // are formed in the TAU+1 refresh at :3242-3243 from the :3193 density,
    // i.e. the one diagnosed BEFORE the condensation block; :3490 then
    // overwrites rho(k) with the post-condensation value, and :3505-3520's
    // orho / rhof / vsc2 / rvs all read THAT one.  The mixture is WRF's, not
    // a rounding artefact, and it is worth up to 1.9e-03 on qr and nr where
    // condensation moved the column: it is the entire cause of the 1.9e-03
    // level-6 residual on aero-reduces-to-classic that used to be the port's
    // only carved-out end-to-end tolerance.
    //
    // `entry_density` is that :3193 density.  The saturation-adjustment
    // kernel above writes it unconditionally into its `reference_density`
    // output, at every level, before its own gate -- so the caller has it
    // for free.  Passing nullptr falls back to the local post-condensation
    // rho, which is what ArWen's mp=8 kernel does and what this kernel did
    // before; the fallback is kept only so the aerosol term stays auditable
    // against thompson.cu in isolation.
    const float rho_entry = (entry_density != nullptr)
        ? entry_density[idx] : rho;
    // WHICH DENSITY LEAVES THIS KERNEL, AND WHY IT IS LEVEL-WISE (WP-13a).
    //
    // `reference_density` is not this kernel's own scratch: the caller hands
    // the same buffer to `launch_rain_sedimentation`, which forms the
    // sedimenting rain mass and number as qr*rho / nr*rho.  Those are WRF's
    // rr(k) and nr(k) at :3794-3795, and WRF builds them in TWO places:
    //
    //   * :3237-3238, from the :3193 TAU+1 density, for EVERY level with
    //     L_qr -- and that is the value sedimentation sees unless
    //   * :3568/:3570 rewrites rr(k)/nr(k) from the :3490 POST-condensation
    //     density, which happens ONLY inside the :3501-3502 gate.
    //
    // rho(k) is overwritten again at :3572, but only the fall-speed rhof
    // (:3614) and the flux-divergence orho (:3799) read that, and the
    // sedimentation kernel rediagnoses both from the state it is handed, so
    // the ONLY thing this buffer decides is the :3237-vs-:3568 mixture.
    // Writing the post-condensation rho unconditionally -- what this kernel
    // did before WP-13a -- gives every level the :3568 answer, including the
    // levels WRF never rewrote.  MEASURED on aero-drop-evap, where the
    // :3501-3502 gate fires at NO level of the column: f32(qr*rho_tau1)
    // equals WRF's rr(k) bitwise at all seven rain levels and
    // f32(qr*rho_post) at none, and the error reaches the surface as
    // 5.165e-04 on RAINNC.
    //
    // So the default is the :3237 density and the gated block below
    // overwrites it with the :3568 one, level by level, from the kernel's own
    // three gates.  Pinned by tests/test_thompson_aerosol_sat_gpu.py::
    // test_rain_evaporation_exports_the_sedimentation_density_wrf_actually_used
    // and, end to end, by tests/test_thompson_aerosol_adapter.py::
    // test_rain_sedimentation_gets_wrfs_level_wise_working_rain_density.
    if (reference_density != nullptr) reference_density[idx] = rho_entry;
    if (reference_temperature != nullptr) {
        reference_temperature[idx] = temp0;
    }
    if (prv_rev_out != nullptr) prv_rev_out[idx] = 0.0;
    if (pnr_rev_out != nullptr) pnr_rev_out[idx] = 0.0;
    if (nr_bound_out != nullptr) nr_bound_out[idx] = 0.0f;
    // :3241, L_qr.
    if (qr[idx] <= THOMPSON_AA_R1) return;
    // :3502.
    if (condensation_rate != nullptr && condensation_rate[idx] > 0.0f) return;

    const float qvs = thompson_rslf(pres, temp0);
    float ssatw = thompson_aa_sub(thompson_aa_div(qv0, qvs), 1.0f);
    if (fabsf(ssatw) < THOMPSON_AA_SAT_EPS) ssatw = 0.0f;
    // :3501.
    if (ssatw >= -THOMPSON_AA_SAT_EPS) return;

    // All three of WRF's :3501-3502 conditions hold, so :3568-3570 runs at
    // this level and the working rain mass/number sedimentation sees are
    // rebuilt from the :3490 post-condensation density.  This is the ONLY
    // place WRF replaces the :3237 pair.
    if (reference_density != nullptr) reference_density[idx] = rho;

    const float orho = thompson_aa_div(1.0f, rho);
    const float odt = thompson_aa_div(1.0f, dt);
    // :3242-3243, using the `rho_entry` hoisted above the gates.
    const float rr = thompson_aa_mul(qr[idx], rho_entry);
    double lamr0 = 0.0;
    const float nr_work = thompson_aa_bound_rain_number(
        rr, thompson_aa_mul(nr[idx], rho_entry), &lamr0);
    if (nr_bound_out != nullptr) nr_bound_out[idx] = nr_work;
    // :3386-3388.  ilamr and N0_r are DOUBLE PRECISION (:1587); nr*org2 is
    // a REAL(4) product and lamr**cre(2) with cre(2) = mu_r+1 = 1 is lamr.
    const double ilamr = 1.0 / lamr0;
    const double n0_r = (double)thompson_aa_mul(nr_work, THOMPSON_AA_ORG2)
        * lamr0;
    // :3535.  WRF re-derives the slope from its stored reciprocal here; the
    // double round trip is not always the identity, so reproduce it.
    const double lamr = 1.0 / ilamr;

    // :3503-3517.
    const float tempc = thompson_aa_sub(temp0, 273.15f);
    const float otemp = thompson_aa_div(1.0f, temp0);
    const float rhof = sqrtf(thompson_aa_mul(THOMPSON_AA_RHO_NOT, orho));
    const float rhof2 = sqrtf(rhof);
    const float diffu = thompson_aa_mul(
        thompson_aa_mul(2.11e-5f,
                        thompson_aa_powf_cr(
                            thompson_aa_div(temp0, 273.15f), 1.94f)),
        thompson_aa_div(101325.0f, pres));
    const float visco = tempc >= 0.0f
        ? thompson_aa_mul(
              thompson_aa_add(1.718f, thompson_aa_mul(0.0049f, tempc)),
              1.0e-5f)
        : thompson_aa_mul(
              thompson_aa_sub(
                  thompson_aa_add(1.718f, thompson_aa_mul(0.0049f, tempc)),
                  thompson_aa_mul(thompson_aa_mul(1.2e-5f, tempc), tempc)),
              1.0e-5f);
    const float vsc2 = sqrtf(thompson_aa_div(rho, visco));
    const float lvap = thompson_aa_add(
        THOMPSON_AA_LVAP0, thompson_aa_mul(2106.0f - 4218.0f, tempc));
    const float tcond = thompson_aa_mul(
        thompson_aa_mul(
            thompson_aa_add(5.69f, thompson_aa_mul(0.0168f, tempc)),
            1.0e-5f),
        418.936f);
    const float ocp = thompson_aa_div(
        1.0f,
        thompson_aa_mul(THOMPSON_AA_CP,
                        thompson_aa_add(1.0f,
                                        thompson_aa_mul(0.887f, qv0))));

    // :3519-3533.  Identical chain to :3426-3441 above; pinned identically.
    const float rvs = thompson_aa_mul(rho, qvs);
    const float tt = thompson_aa_sub(
        thompson_aa_mul(thompson_aa_mul(lvap, otemp), THOMPSON_AA_ORV), 1.0f);
    const float rvs_p = thompson_aa_mul(thompson_aa_mul(rvs, otemp), tt);
    const float pp_a = thompson_aa_mul(
        thompson_aa_mul(thompson_aa_mul(otemp, tt), otemp), tt);
    float pp_b = thompson_aa_mul(-2.0f, lvap);
    pp_b = thompson_aa_mul(pp_b, otemp);
    pp_b = thompson_aa_mul(pp_b, otemp);
    pp_b = thompson_aa_mul(pp_b, otemp);
    pp_b = thompson_aa_mul(pp_b, THOMPSON_AA_ORV);
    const float pp_c = thompson_aa_mul(otemp, otemp);
    const float rvs_pp = thompson_aa_mul(
        rvs, thompson_aa_add(thompson_aa_add(pp_a, pp_b), pp_c));
    const float gamsc = thompson_aa_mul(
        thompson_aa_div(thompson_aa_mul(lvap, diffu), tcond), rvs_p);
    const float gr = thompson_aa_div(gamsc, thompson_aa_add(1.0f, gamsc));
    float alphsc = thompson_aa_mul(0.5f, gr);
    alphsc = thompson_aa_mul(alphsc, gr);
    alphsc = thompson_aa_mul(alphsc, rvs_pp);
    alphsc = thompson_aa_div(alphsc, rvs_p);
    alphsc = thompson_aa_mul(alphsc, rvs);
    alphsc = thompson_aa_div(alphsc, rvs_p);
    alphsc = fmaxf(1.0e-9f, alphsc);
    // :3529 -- MIN(-1.E-9, ssatw), the rain block's clip, NOT the droplet
    // block's snap-to-zero at :3437.
    const float xsat = fminf(-1.0e-9f, ssatw);
    const float ax = thompson_aa_mul(alphsc, xsat);
    float t2 = thompson_aa_mul(2.0f, alphsc);
    t2 = thompson_aa_mul(t2, alphsc);
    t2 = thompson_aa_mul(t2, xsat);
    t2 = thompson_aa_mul(t2, xsat);
    float t3 = thompson_aa_mul(5.0f, alphsc);
    t3 = thompson_aa_mul(t3, alphsc);
    t3 = thompson_aa_mul(t3, alphsc);
    t3 = thompson_aa_mul(t3, xsat);
    t3 = thompson_aa_mul(t3, xsat);
    t3 = thompson_aa_mul(t3, xsat);
    float paren = thompson_aa_sub(1.0f, ax);
    paren = thompson_aa_add(paren, t2);
    paren = thompson_aa_sub(paren, t3);
    float t1_evap = thompson_aa_mul(
        thompson_aa_mul(2.0f, THOMPSON_AA_PI), paren);
    t1_evap = thompson_aa_div(t1_evap, thompson_aa_add(1.0f, gamsc));

    // :663 and :800-801.  Sc3 must be the correctly rounded cube root -- see
    // point 2 in the header note.  crg(10) = Gamma(2) = 1 and
    // crg(11) = Gamma(3) = 2 exactly, so t1_qr_ev is literally 0.78.
    const float sc3 = thompson_aa_powf_cr(THOMPSON_AA_SC, 1.0f / 3.0f);
    const float t1_qr_ev = thompson_aa_mul(0.78f, 1.0f);
    const float t2_qr_ev = thompson_aa_mul(
        thompson_aa_mul(thompson_aa_mul(0.308f, sc3),
                        sqrtf(THOMPSON_AA_AV_R)),
        2.0f);

    // :3540-3542.  The leading four factors are REAL(4) until N0_r (DOUBLE)
    // enters; the bracket is DOUBLE throughout.  cre(10) = mu_r+2 = 2 and
    // cre(11) = 0.5*(bv_r+5+2*mu_r) = 3.
    const float rate_prefix = thompson_aa_mul(
        thompson_aa_mul(t1_evap, diffu), -ssatw);
    const double diffusion_term = (double)t1_qr_ev * pow(ilamr, 2.0);
    const float vent_prefix = thompson_aa_mul(
        thompson_aa_mul(t2_qr_ev, vsc2), rhof2);
    const double ventilation_term = (double)vent_prefix
        * pow(lamr + (double)thompson_aa_mul(0.5f, THOMPSON_AA_FV_R), -3.0);
    double prv_rev = (double)rate_prefix * n0_r * (double)rvs
        * (diffusion_term + ventilation_term);

    // :3537-3556.
    if (thompson_aa_div(qv0, qvs) < 0.95f
            && thompson_aa_mul(rr, orho) <= 1.0e-8f) {
        prv_rev = (double)thompson_aa_mul(thompson_aa_mul(rr, orho), odt);
    } else {
        const float rate_max = fminf(
            thompson_aa_mul(thompson_aa_mul(rr, orho), odt),
            thompson_aa_mul(thompson_aa_sub(qvs, qv0), odt));
        prv_rev = fmin((double)rate_max, prv_rev * (double)orho);
        if (melting_graupel) {
            const float eva_factor = fminf(
                1.0f,
                thompson_aa_add(
                    0.01f,
                    thompson_aa_mul(0.99f - 0.01f,
                                    thompson_aa_div(tempc, 20.0f))));
            prv_rev = prv_rev * (double)eva_factor;
        }
    }

    // :3559-3560.
    const double pnr_rev = fmin(
        (double)thompson_aa_mul(
            thompson_aa_mul(thompson_aa_mul(nr_work, 0.99f), orho), odt),
        prv_rev * (double)nr_work / (double)rr);
    if (prv_rev_out != nullptr) prv_rev_out[idx] = prv_rev;
    if (pnr_rev_out != nullptr) pnr_rev_out[idx] = pnr_rev;

    // :3562-3571.  WRF rounds each DOUBLE rate into its REAL(4) tendency
    // accumulator before multiplying by DT; reproduce that rounding.
    const float qr_tendency = (float)(-prv_rev);
    const float qv_tendency = (float)prv_rev;
    const float nr_tendency = (float)(-pnr_rev);
    // :3566.  lvap(k)*ocp(k) is a REAL(4) product formed BEFORE the DOUBLE
    // prv_rev multiplies it.
    const float temperature_tendency = (float)(
        -(double)thompson_aa_mul(lvap, ocp) * prv_rev);

    qr[idx] = thompson_aa_add(qr[idx], thompson_aa_mul(qr_tendency, dt));
    qv[idx] = fmaxf(1.0e-10f,
                    thompson_aa_add(qv0, thompson_aa_mul(qv_tendency, dt)));
    nr[idx] = thompson_aa_add(nr[idx], thompson_aa_mul(nr_tendency, dt));
    temperature[idx] = thompson_aa_add(
        temp0, thompson_aa_mul(temperature_tendency, dt));
    // :3565.  The whole aerosol addition to this process.
    if (nwfaten != nullptr) {
        nwfaten[idx] = (float)((double)nwfaten[idx] + pnr_rev);
    }
}


extern "C" __global__ void thompson_aa_rain_evaporation(
    float* __restrict__ qr,
    float* __restrict__ nr,
    float* __restrict__ temperature,
    const float* __restrict__ pressure,
    float* __restrict__ qv,
    float* __restrict__ nwfaten,
    float* __restrict__ reference_density,
    float* __restrict__ reference_temperature,
    const float* __restrict__ graupel_melt_marker,
    const float* __restrict__ condensation_rate,
    const float* __restrict__ entry_density,
    float dt, int size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    thompson_aa_rain_evaporation_impl(
        qr, nr, temperature, pressure, qv, nwfaten, reference_density,
        reference_temperature, graupel_melt_marker, condensation_rate,
        entry_density, nullptr, nullptr, nullptr, dt, idx);
}


// Diagnostic sibling: exposes prv_rev, pnr_rev and the mvd_r-bounded working
// rain number so the oracle test can pin the three quantities WRF's
// :3540/:3559/:3250 actually produce rather than inferring them from a state
// difference.  The four state arrays are the CALLER'S SCRATCH COPIES and are
// consumed in place exactly as the real kernel would; the launcher hands it
// duplicates so nothing observable changes.
extern "C" __global__ void thompson_aa_rain_evaporation_probe(
    float* __restrict__ qr,
    float* __restrict__ nr,
    float* __restrict__ temperature,
    const float* __restrict__ pressure,
    float* __restrict__ qv,
    const float* __restrict__ graupel_melt_marker,
    const float* __restrict__ entry_density,
    double* __restrict__ prv_rev_out,
    double* __restrict__ pnr_rev_out,
    float* __restrict__ nr_bound_out,
    float dt, int size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    thompson_aa_rain_evaporation_impl(
        qr, nr, temperature, pressure, qv, nullptr, nullptr, nullptr,
        graupel_melt_marker, nullptr, entry_density,
        prv_rev_out, pnr_rev_out, nr_bound_out, dt, idx);
}
