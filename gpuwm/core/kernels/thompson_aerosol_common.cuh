// gpuwm/core/kernels/thompson_aerosol_common.cuh
//
// Shared __device__ helpers for WRF v4.6.1 aerosol-aware Thompson
// (mp_physics=28).  Numerical authority is
// WRF v4.6.1 phys/module_mp_thompson.F, commit
// d66e442fccc04111067e29274c9f9eaccc3cef28, zero local modifications.  Every
// line number in this file refers to that source.
//
// ---------------------------------------------------------------------------
// HOW THIS FILE IS USED
// ---------------------------------------------------------------------------
// There is no #include path under cupy.RawModule, so this header is PREPENDED
// textually by gpuwm/core/kernels/__init__.py::_EXTRA_HEADERS to the six
// aerosol translation units:
//     thompson_aerosol_probe, thompson_aerosol_state, thompson_aerosol_sat,
//     thompson_aerosol_cold,  thompson_aerosol_warm,  thompson_aerosol_sed
// Every other module -- above all thompson.cu -- assembles a BYTE-IDENTICAL
// source string to what it assembled before this header existed.  Do not add
// this header to any other module, and do not #include it from a .cu file.
//
// ===========================================================================
// PUBLISHED SHARED SIGNATURES -- cold.cu, warm.cu and sed.cu MUST CALL THESE
// ===========================================================================
//
// These used to be duplicated, per-network, with DIFFERENT BODIES.  That is
// exactly how the two halves of a scheme drift apart: separate
// cupy.RawModule translation units mean nvrtc never sees the conflict, so the
// drift is silent.  Each now has ONE definition, here, and the .cu files carry
// none.
//
// ---------------------------------------------------------------------------
// WHAT ENFORCES THAT, AND WHAT DOES NOT
// ---------------------------------------------------------------------------
// THE ENFORCEMENT MECHANISM IS THE SOURCE SCAN, NOT THE COMPILER.
// tests/test_thompson_aerosol_device_helpers.py::
// test_shared_helpers_are_defined_exactly_once_and_only_in_the_header greps
// every gpuwm/core/kernels/thompson_aerosol_*.cu for a DEFINITION of each name
// below and fails, naming file and line, if one survives.  That test is the
// contract.  Run it; do not rely on a build failure.
//
// This file used to claim instead that "a surviving local copy will FAIL TO
// COMPILE with a redefinition error -- that error is the enforcement
// mechanism".  THAT CLAIM WAS FALSE FOR ONE OF THE FOUR AND THE FALSE HALF IS
// THE DANGEROUS HALF:
//
//   * thompson_aa_bound_rain_number, thompson_aa_bound_ice_number and
//     thompson_aa_decade_index_double have shared signatures IDENTICAL to the
//     local copies they replaced, so a survivor really is a hard nvrtc
//     "function has already been defined" error.  VERIFIED by compiling the
//     assembled source string for thompson_aerosol_cold and
//     thompson_aerosol_warm with a local thompson_aa_decade_index_double
//     still present: both fail at the local definition, citing the header's
//     line as the previous one.
//
//   * thompson_aa_entry_rain_distribution DOES NOT.  Its shared form takes
//     SEVEN parameters where the deleted local copies took six (they emitted
//     no rain_intercept_n0).  C++ treats a different parameter list as an
//     OVERLOAD, not a redefinition: the translation unit compiles CLEANLY,
//     and every six-argument call site keeps resolving to the LOCAL copy --
//     the plain-powf one that sat ~2.7e-7 away from the oracle -- while the
//     header's contraction-pinned definition sits there unused.  An agent
//     built exactly that variant and nvrtc said nothing.  There is no
//     compiler diagnostic for this; only the source scan catches it.
//
// So: delete the local copy.  Do not rename it, and do not assume the build
// would have told you.
//
//  bool thompson_aa_entry_rain_distribution(
//           float rain_per_kg, float rain_number_per_kg, float density,
//           float* rain_number, double* rain_lambda, float* rain_mvd,
//           double* rain_intercept_n0)
//        module_mp_thompson.F:1878-1898 (entry bound) THEN :2144-2150 (the
//        y-intercept pass).  Returns L_qr, i.e. rain_per_kg > R1.
//        rain_number [m^-3], rain_lambda [m^-1], rain_mvd [m],
//        rain_intercept_n0 = nr*org2*lamr**cre(2) = nr*lamr (cre(2)=mu_r+1=1,
//        org2=1/WGAMMA(1)=1).  THIS IS THE CONTRACTION-PINNED FORM: every
//        product/quotient is thompson_aa_mul/div and every power is
//        thompson_aa_powf_cr.  warm.cu:81-92 records lamr / mvd_r / N0_r and
//        every rate built on them at <= 7e-16 relative over 12348 Fortran-
//        oracle rows -- float32-exact, the residual being the CSV round trip
//        -- where the earlier plain-powf form sat at ~2.7e-7.  Two properties
//        callers depend on:
//          * the :2146-2150 pass ALWAYS re-forms lamr from the bounded nr,
//            for every level, rain or not -- WRF's loop at :2145 has no
//            L_qr guard -- so rain_lambda/rain_mvd/rain_intercept_n0 are
//            defined even when the return value is false (rr=R1, nr=R2);
//          * prr_rcw / pnc_rcw / pna_rca / pnd_rcd read N0_r and
//            (lamr+fv_r)**(-cre(9)) directly, so they MUST use
//            rain_intercept_n0 and rain_lambda from here, never a lambda
//            re-derived from the clamped mvd.
//
//  void thompson_aa_bound_rain_number(float rain_mass, float density,
//                                     float* rain_number_per_kg)
//        module_mp_thompson.F:4032-4046 == thompson.cu:2574-2598.  Terminal
//        size bound; rain_mass [kg m^-3], rain_number_per_kg in/out [kg^-1].
//
//  void thompson_aa_bound_ice_number(float ice_mass, float density,
//                                    float* ice_number_per_kg)
//        module_mp_thompson.F:4029-4039 == thompson.cu:3719-3743.  Terminal
//        size bound; idempotent, so a fused and a terminal application agree.
//
//  int  thompson_aa_decade_index_double(double value, int first_exponent,
//                                       int table_size)
//        thompson.cu:3084-3105.  The DOUBLE form of the base-ten
//        mantissa/decade bin, for the rain and graupel y-intercept lookups,
//        which WRF keeps in DOUBLE PRECISION (N0_r/N0_g, :1587) even though
//        the state and the base-ten scale are default REAL.  Returned
//        ZERO-BASED.  Callers: the idx_r lookup at (first_exponent 6,
//        table_size 37) and the idx_g lookup at (2, 37).
//        PROMOTED FROM cold.cu:206-223 AND warm.cu:272-289, which carried
//        byte-identical copies.  The float sibling thompson_aa_decade_index
//        was already shared; only the double form was not, and two copies of
//        one helper in two translation units is precisely the drift vector
//        that produced the entry_rain_distribution divergence recorded above.
//        The body here is those copies verbatim, and
//        test_promoted_decade_index_double_is_bitwise_identical_to_the_local_
//        copies compiles a golden byte-copy of them alongside this definition
//        in ONE translation unit and asserts bitwise equality on every input,
//        so the promotion cannot have changed a result.
//
//  int  thompson_aa_nu_c_working(float nc_m3_after_rediagnosis)
//        module_mp_thompson.F:2170.  See the nu_c STAGING RULE below.  This
//        is the nu_c every rate in the warm and cold networks must use.
//
// ---------------------------------------------------------------------------
// THE nu_c STAGING RULE  (:1832 entry nu_c  vs  :2170 working nu_c)
// ---------------------------------------------------------------------------
// WRF computes nu_c TWICE from two DIFFERENT droplet numbers:
//
//   :1832  nu_c = MIN(15, NINT(1000.E6/nc(k)) + 2)   <- nc(k) from :1829,
//          the PRE-rediagnosis nc = MAX(2, MIN(nc1d*rho, Nt_c_max)).  Used
//          ONLY to build lamc at :1833 and to run the :1834-1838 droplet-size
//          clamp.  thompson_aa_cloud_dist returns it as nu_c_entry_out.
//
//   :1840  nc(k) is REDIAGNOSED from the clamped lamc.  This is what
//          thompson_aa_cloud_dist RETURNS.
//
//   :2170  nu_c = MIN(15, NINT(1000.E6/nc(k)) + 2)   <- recomputed from the
//          POST-rediagnosis nc(k).  THIS is the nu_c that feeds lamc (:2173),
//          mvd_c (:2174), Dc_g (:2181) and pnr_wau (:2192).
//
// Whenever the :1834-1838 size clamp engages -- thin cloud edges, which the
// cold network reaches at qc > 1e-12 -- the two DIFFER.  MEASURED on an
// RTX 5090 over a 70-point (nc_entry, rc) grid at rho = 1: 13 of 70 states
// diverge, every one of them at rc <= 1e-7.  Worked examples, nc in m^-3:
//     nc_entry=1e9,   rc=1e-8 -> nc = 5.4590136e7, nu_c_entry=3  WRF 15
//     nc_entry=1e8,   rc=1e-8 -> nc = 2.8654908e7, nu_c_entry=12 WRF 15
//     nc_entry=1.5e9, rc=1e-7 -> nc = 5.4590140e8, nu_c_entry=3  WRF 4
// Between nu_c = 3 and nu_c = 15 the individual gamma columns move by more
// than TEN orders of magnitude -- ccg(2,n) by 8.9e12x, ocg1(n) by 2.2e11x --
// and the PRODUCTS the rates are built from move by large finite factors:
// ccg(2,n)*ocg1(n) by 40.8x, so lamc by 3.4x and mvd_c with it;
// ccg(3,n)*ocg2(n) by 15.8x for Dc_g; and pnr_wau, being linear in nu_c,
// rescales by 5.  No tolerance absorbs that.  It is a physics error, not a
// rounding one.
//
//   CORRECT:  nc_m3 = thompson_aa_cloud_dist(rc, nc_per_kg, rho, &nu_c_entry,
//                                            &lamc_entry);
//             const int nu_c = thompson_aa_nu_c_working(nc_m3);
//   WRONG:    use nu_c_entry for anything after :1838.
//
// tests/test_thompson_aerosol_device_helpers.py asserts the divergence over
// the full 70-point grid through thompson_aa_probe_nu_c_staging, so a kernel
// that reuses the entry value can be caught without running a network.
//
// ---------------------------------------------------------------------------
// THE CENTRAL POINT: nc IS PROGNOSTIC
// ---------------------------------------------------------------------------
// mp=8 freezes cloud droplet number at Nt_c = 100e6 m^-3.  thompson.cu
// therefore hardcodes 100.0e6f (12 sites), cloud_number_bin = 65 (3 sites)
// and the gamma ratios 2730.0f / 272.0f (7 sites).  In mp=28 all three are
// live functions of nc.  Every value you compute must trace to nc through
// these helpers, never to a literal.  Two HARD IDENTITY GATES prove the
// generalized forms reduce to what mp=8 froze:
//     thompson_aa_droplet_bin(100.0e6f) == 65
//     thompson_aa_in_bin(1000.0f)       == 27
// They are asserted in tests/test_thompson_aerosol_device_helpers.py.
//
// ---------------------------------------------------------------------------
// GAMMA PARITY, AND A PRE-EXISTING mp=8 DEVIATION THAT MUST NOT BE "FIXED"
// ---------------------------------------------------------------------------
// The THOMPSON_AA_CC* tables below are WRF's REAL(4) WGAMMA/GAMMLN Lanczos
// series (5325-5377) evaluated in float32 by
// gpuwm/core/thompson_aerosol_contract.py and emitted here as exact
// round-tripping float literals.  math.lgamma is NOT equivalent: it gives
// Gamma(16) = 1.30767441e12 where WRF gives 1.30767389e12, and
// 1.30767389e12f is the literal already embedded in thompson.cu:2083.
//
// At nu_c = 12 (i.e. nc = Nt_c) the runtime products are
//     ccg(2,12)*ocg1(12) = 2729.9973    (not 2730)
//     ccg(5,12)*ocg2(12) =  272.00012   (not 272)
// thompson.cu hardcodes 2730.0f at :882, :999, :4005, :4128, :4680 and
// 272.0f at :888, :1006, so mp=8 carries a small deviation at those five/two
// sites.  (thompson.cu:343 is NOT one of them: calc_effectRad genuinely uses
// WRF's exact-integer g_ratio PARAMETER, for which 2730 is correct.)
// mp=28 is RIGHT -- it computes from the series via THOMPSON_AA_CCG*/OCG*.
// mp=8 stays frozen and slightly wrong.  DO NOT reconcile them.
//
// ---------------------------------------------------------------------------
// PUBLISHED HELPER API   (signature | units | valid range | WRF authority)
// ---------------------------------------------------------------------------
//
// -- arithmetic primitives (USE THESE in any new aerosol kernel) ------------
//  float thompson_aa_add/sub/mul/div(float a, float b)
//        __fadd_rn/__fsub_rn/__fmul_rn/__fdiv_rn.  nvrtc defaults to
//        --fmad=true; build_aero.sh compiles the oracle with plain
//        `gfortran -O2` on baseline x86-64, which has NO fma instruction, so
//        every REAL(4) multiply and add in WRF is separately rounded.
//        Leaving contraction on cost 1-2 float digits in activ_ncloud,
//        iceKoop and Eff_aero; pinning it made all three BIT-EXACT against
//        the Fortran probe.  Prefer these wherever a fixture disagrees in the
//        last digits.
//
//  float thompson_aa_expf_cr/logf_cr(float x), thompson_aa_powf_cr(float,
//        float)
//        Correctly-rounded float32 EXP/LOG/**, evaluated in double and
//        rounded ONCE.  gfortran lowers REAL(4) EXP/LOG/** to glibc, which is
//        correctly rounded; CUDA's expf/logf/powf carry up to ~2 ulp.  Every
//        operand and every stored result is still float32.
//
// -- rounding / indexing ----------------------------------------------------
//  int   thompson_aa_nint(float x)
//        Fortran NINT: round half AWAY FROM ZERO.  CUDA __float2int_rn
//        rounds half to even and is NOT a substitute.
//
//  int   thompson_aa_nu_c(float nc_m3)
//        nc_m3 [m^-3], must be > 0 (callers clamp to [2, 1.999e9] first).
//        Returns the cloud shape parameter in [2, 15].  :2170, :1832.
//        This is the integer that indexes every THOMPSON_AA_CC*/OCG* table.
//        WHICH nc YOU PASS IS PHYSICS, NOT PLUMBING -- see the nu_c STAGING
//        RULE above and prefer thompson_aa_nu_c_working at every :2170 site.
//
//  int   thompson_aa_nu_c_working(float nc_m3_after_rediagnosis)
//        :2170.  thompson_aa_nu_c under a name that records WHICH nc is
//        legal to pass: the value thompson_aa_cloud_dist RETURNS, never the
//        nu_c_entry_out it writes.
//
//  int   thompson_aa_droplet_bin(float nc_m3)
//        Zero-based idx_n into tnc_wev's third axis, range [0, 99].
//        :3447-3448.  Computed in DOUBLE precision because WRF uses DLOG of
//        a DOUBLE t_Nc(1); nic1 is the TRUNCATED integer 7, not 7.926.
//        GATE: thompson_aa_droplet_bin(100.0e6f) == 65.
//
//  int   thompson_aa_decade_index(float value, int first_exponent,
//                                 int table_size)
//        Zero-based base-ten mantissa/decade bin.  value > 0.  Transcribed
//        from thompson.cu:3063-3082 (itself :2282-2307); the identical
//        pattern appears at :2581-2589 for idx_IN.
//
//  int   thompson_aa_decade_index_double(double value, int first_exponent,
//                                        int table_size)
//        The DOUBLE spelling of the same rule, thompson.cu:3084-3105, for
//        WRF's DOUBLE PRECISION rain and graupel y-intercepts (N0_r/N0_g,
//        :1587).  Promoted out of cold.cu and warm.cu in wave 4; see the
//        PUBLISHED SHARED SIGNATURES block.  Two spellings exist because WRF
//        has two, not because the rule differs.
//
//  int   thompson_aa_in_bin(float xni)
//        Zero-based idx_IN into the freezeH2O tables, range [0, 54].
//        xni [m^-3] is the ice-nuclei number from thompson_ice_demott.
//        :2579-2591.  GATE: thompson_aa_in_bin(1000.0f) == 27, which is the
//        nuclei_bin thompson.cu:3936 hardcodes.
//
// -- saturation and snow-moment fits (transcribed from WRF, NOT from mp=8) --
//  float thompson_rslf(float p_pa, float t_k)      :5378-5413
//  float thompson_rsif(float p_pa, float t_k)      :5414-5446
//  float thompson_field_a(float tc, float moment)  :2069-2075 (sa, :358-359)
//  float thompson_field_b(float tc, float moment)  :2076-2079 (sb, :361-362)
//        ALL FOUR now diverge from thompson.cu's plain chains BY DESIGN.  The
//        two saturation fits are contraction-pinned (see "THE SHARED FITS"
//        above thompson_aa_add); the two snow-moment fits additionally
//        restore WRF's LEFT-TO-RIGHT operator association, which the copied
//        mp=8 form broke by hoisting tc*tc and moment*moment, and evaluate
//        `10.0**loga_` correctly rounded.  Measured bit-exact against
//        gfortran -O2 on 253 states and against the real calc_effectRad on
//        360 more; the numbers and the two defects are recorded above
//        thompson_field_a.
//
// -- aerosol physics --------------------------------------------------------
//  float thompson_activ_ncloud(float Tt, float Ww, float NCCN,
//                              const double* tnccn_act)
//        Tt [K], Ww [m s^-1], NCCN [m^-3] (water-friendly aerosol).
//        tnccn_act is WP-01's float64 FORTRAN-ORDER (7,9,7,5,4) device
//        array.  Returns activated droplet number [m^-3].  :5178-5253.
//        NEAREST-NEIGHBOUR (not interpolated) in temperature; bilinear in
//        log(N) and log(w); aerosol radius index l=3 and kappa index m=2 are
//        hardcoded by WRF.  Both clamps use ASYMMETRIC epsilons.
//        TOLERANCE NOTE: k is a nearest 10 K bin and i/j are bracket
//        indices, so the result is a STEP function of state near a bin edge.
//
//  float thompson_ice_demott(float tempc, float rho, float nifa_m3)
//        tempc [degC, < 0], rho [kg m^-3], nifa_m3 [m^-3].  Returns ice
//        nuclei number [m^-3].  :5448-5518.  NEGATIVE FINDING: in v4.6.1 the
//        Phillips (2008) branch (5474-5505) is entirely commented out, so
//        this is a PURE function of (tempc, rho, nifa) -- do not pass
//        qv/qvs/qvsi and do not port the commented code.  WRF's two call
//        sites (:2574, :2623) differ only in the dead qv argument and return
//        the identical value; evaluate ONCE per level.
//
//  float thompson_ice_koop(float temp_k, float qv, float qvs,
//                          float nwfa_m3, float dt)
//        Homogeneous freezing of deliquesced haze [m^-3].  :5521-5546.
//        log_J capped at 20; result capped at 1000.e3.  Gated by WRF at
//        :2634-2637 on temp < 238 K, ssati >= 0.4 and ns+ni <= 999.e3 --
//        the caller applies that gate, not this helper.
//
//  float thompson_eff_aero(float D, float Da, float visc, float rhoa,
//                          float temp_k, int species)
//        Aerosol collection efficiency of a collector drop/crystal, Slinn
//        (1983) via Wang et al (2010).  D [m] collector diameter, Da [m]
//        aerosol diameter, visc [kg m^-1 s^-1], rhoa [kg m^-3], temp_k [K].
//        species is THOMPSON_AA_SPECIES_RAIN / _SNOW / _GRAUPEL.  Result is
//        clamped to [1e-5, 1.0].  :4965-5001.  The graupel fall speed uses
//        av_g(idx_bg1)=442.0, bv_g(idx_bg1)=0.89, which is what
//        thompson_init installs when ng is absent (:462-465) -- i.e. the
//        non-hail-aware mp=28 configuration this port targets.
//
//  float thompson_aa_snow_number(float smob, float smoc)
//        WRF's EXPLICIT two-gamma snow number ns(k) [m^-3], :2083-2088.
//        smob is the bm_s-th (== 2nd) snow moment, smoc the (bm_s+1)-th.
//        This is NOT smo0 and is not interchangeable with it.  It is needed
//        only by the Koop homogeneous-freezing gate at :2634, where it enters
//        an ADDITIVE threshold test (xni = ns+ni+... <= 999.e3), so an error
//        here does not perturb a rate, it flips a branch.
//        No WRF procedure exposes it -- Kap0/Kap1/Lam0/Lam1 (:114-117),
//        mu_s (:113) and csg/cse are all PRIVATE -- so the gate is a
//        `gfortran -O2` transcription of :2029-2088 built with build_aero.sh's
//        own flags, over 391 states (23 temperatures x 17 snow contents,
//        ns from 1.5e-1 to 2.6e+08 m^-3).  MEASURED BIT-EXACT on all 391.
//        CALLER NOTE: WRF sets ns(k) = 0 at :1790 and only overwrites it
//        inside the `if (.not. L_qs(k)) CYCLE` loop at :2026-2088, so at a
//        snow-free level the Koop gate sees ns = 0, not a stale value.
//
// -- droplet distribution ---------------------------------------------------
//  float thompson_aa_cloud_dist(float rc, float nc_per_kg, float rho,
//                               int* nu_c_entry_out, double* lamc_entry_out)
//        :1826-1842.  rc [kg m^-3] cloud water CONTENT (qc*rho),
//        nc_per_kg [kg^-1] entry droplet number, rho [kg m^-3].
//        RETURNS the rediagnosed nc [m^-3] (:1840) after the D0c / 2*D0r size
//        clamp.  The two out-parameters are WRF's ENTRY-STAGE values from
//        :1832-1838 and are NOT usable downstream: nu_c_entry_out is computed
//        from the PRE-rediagnosis nc and lamc_entry_out is the clamped lambda
//        that produced the return value.  Everything after :1838 recomputes
//        both from the RETURNED nc -- see the nu_c STAGING RULE above.
//        Caller must have rc > R1.
//
//  int   thompson_aa_inu_c_effrad(float nc_m3)
//        calc_effectRad's THREE-branch shape selector, :5637-5643.  It is
//        deliberately different from thompson_aa_nu_c: nc < 100 -> 15,
//        nc > 1e10 -> 2 (dead code, the preceding Nt_c_max clamp forbids
//        it), else MIN(15, NINT(1000e6/nc)+2).
//
//  float thompson_aa_eff_rad_cloud(float rc, float nc_m3)   :5636-5646
//  float thompson_aa_eff_rad_ice(float ri, float ni)        :5650-5656
//  float thompson_aa_eff_rad_snow(float rs, float t_k)      :5658-5694
//        Effective radii in METRES, already carrying calc_effectRad's own
//        clamps.  mp_gt_driver's second clamp (:1475-1477) and gpuwm's
//        metre->micron convention are the CALLER's job (see
//        thompson.cu:373-381 for the mp=8 precedent).
//
// -- WRF's terminal clamps (:1805-1806, :3217, :3486, :3979-3981) -----------
//  float thompson_aa_clamp_nc(float nc_m3)     -> [2, 1.999e9]
//  float thompson_aa_clamp_nwfa(float nwfa_m3) -> [11.1e6, 9999e6]
//  float thompson_aa_clamp_nifa(float nifa_m3) -> [5.0e3, 9999e6]
//
// -- tables (index 0 unused so device code reads like Fortran) --------------
//  THOMPSON_AA_CCE1..CCE5[16], THOMPSON_AA_CCG1..CCG5[16],
//  THOMPSON_AA_OCG1[16], THOMPSON_AA_OCG2[16]   -- :671-685, WGAMMA-FP32.
//  THOMPSON_AA_G_RATIO[16]                      -- :5611-5613, EXACT
//        integers (n+1)(n+2)(n+3).  Used ONLY by calc_effectRad.  It is NOT
//        interchangeable with CCG2*OCG1 (2730 vs 2729.9973).
//
#pragma once

// ---------------------------------------------------------------------------
// Scalar parameters.  Each is the module_mp_thompson.F REAL(4) value; where
// WRF forms a compile-time product the float32 result is written out.
// ---------------------------------------------------------------------------
#define THOMPSON_AA_PI            3.1415926536f   // :67
#define THOMPSON_AA_R_DRY         287.04f         // module_model_constants R
// DELIBERATELY ABSENT: Nt_c (:88, 100.0e6).  It is the single literal this
// port must never read as a droplet number -- mp=8 freezes nc at it, mp=28
// carries nc prognostically -- and a #define here would make it visible to
// all six aerosol translation units for the benefit of no production kernel.
// The identity gates that prove the generalized forms reduce to what mp=8
// froze consume NT_C from gpuwm/core/thompson_aerosol_contract.py:104, on the
// host, where it cannot be reached by device code.
#define THOMPSON_AA_NT_C_MAX      1999.0e6f       // :89
#define THOMPSON_AA_NC_FLOOR      2.0f            // :1830, :3217, :3486
#define THOMPSON_AA_NWFA_FLOOR    11.1e6f         // :1805
#define THOMPSON_AA_NIFA_FLOOR    5.0e3f          // :1806  (naIN1*0.01)
#define THOMPSON_AA_AERO_CEIL     9999.0e6f       // :1805-1806, :3979-3981
#define THOMPSON_AA_R1            1.0e-12f        // :183
#define THOMPSON_AA_R2            1.0e-6f         // :184
#define THOMPSON_AA_AM_R          5.235988159e+02f  // :128 PI*rho_w/6
#define THOMPSON_AA_BM_R          3.0f            // :129
#define THOMPSON_AA_OBMR          3.333333433e-01f  // :721 1./bm_r in REAL(4)
#define THOMPSON_AA_AM_I          4.660029297e+02f  // :137 PI*rho_i/6
#define THOMPSON_AA_OBMI          3.333333433e-01f  // :703 1./bm_i
#define THOMPSON_AA_CIG2          6.0f            // :695 WGAMMA(4) exactly 6
#define THOMPSON_AA_OIG1          1.0f            // :701 1./WGAMMA(1)
#define THOMPSON_AA_MU_I          0.0f            // :105
#define THOMPSON_AA_D0C           1.0e-6f         // :224
#define THOMPSON_AA_D0R           50.0e-6f        // :225
#define THOMPSON_AA_AV_C          0.316946e8f     // :163
#define THOMPSON_AA_BV_C          2.0f            // :164
#define THOMPSON_AA_AM_S          0.069f          // :130
#define THOMPSON_AA_OAMS          1.449275398e+01f  // :747 1./am_s
#define THOMPSON_AA_BM_S          2.0f            // :131
#define THOMPSON_AA_AV_S          40.0f           // :146
#define THOMPSON_AA_BV_S          0.55f           // :147
#define THOMPSON_AA_AV_G          442.0f          // :149 av_g_old == av_g(5)
#define THOMPSON_AA_BV_G          0.89f           // :150 bv_g_old == bv_g(5)
#define THOMPSON_AA_MU_S          0.6357f         // :113
#define THOMPSON_AA_KAP0          490.6f          // :114
#define THOMPSON_AA_KAP1          17.46f          // :115
#define THOMPSON_AA_LAM0          20.78f          // :116
#define THOMPSON_AA_LAM1          3.29f           // :117
#define THOMPSON_AA_CSE15         1.635699987e+00f  // :741 mu_s + 1.
#define THOMPSON_AA_CSG15         8.980315328e-01f  // WGAMMA(cse(15))
#define THOMPSON_AA_R_UNI         8.314f          // :206
#define THOMPSON_AA_AR_VOLUME     6.544983879e-17f  // :213 4./3.*PI*(2.5e-6)**3
#define THOMPSON_AA_RHO_NOT0      1.292283773e+00f  // :5470 101325/(287.05*273.15)
#define THOMPSON_AA_RE_QC_BG      2.49e-6f        // module_model_constants
#define THOMPSON_AA_RE_QI_BG      4.99e-6f
#define THOMPSON_AA_RE_QS_BG      9.99e-6f

// Table extents, :231-252.
#define THOMPSON_AA_NBC       100
#define THOMPSON_AA_NTB_C     37
#define THOMPSON_AA_NTB_IN    55
#define THOMPSON_AA_NTB_ARC   7
#define THOMPSON_AA_NTB_ARW   9
#define THOMPSON_AA_NTB_ART   7
#define THOMPSON_AA_NTB_ARR   5
#define THOMPSON_AA_NTB_ARK   4

// activ_ncloud:5229-5230.  One-based and hardcoded by WRF: mean aerosol
// radius 0.04 um and hygroscopicity kappa 0.4.  Four fifths of the shipped
// CCN_ACTIVATE.BIN is therefore never read.
#define THOMPSON_AA_ACTIV_L   3
#define THOMPSON_AA_ACTIV_M   2

// Eff_aero collector species selector (:4974-4980).  WRF passes CHARACTER*1.
#define THOMPSON_AA_SPECIES_RAIN     0
#define THOMPSON_AA_SPECIES_SNOW     1
#define THOMPSON_AA_SPECIES_GRAUPEL  2

// t_Nc(1), :720-731.  DOUBLE PRECISION in WRF and DLOG'd there, so the
// droplet-bin index must be formed in double.  nic1 is declared INTEGER at
// :246 and assigned a DOUBLE at :896, so Fortran TRUNCATES 7.926303892 to 7.
#define THOMPSON_AA_T_NC_1    1040843.9118849806
#define THOMPSON_AA_NIC1      7

// ---------------------------------------------------------------------------
// Gamma moment tables, module_mp_thompson.F:671-685, one column per integer
// nu_c in 1..15.  Emitted from gpuwm.core.thompson_aerosol_contract's
// transcription of WRF's REAL(4) WGAMMA/GAMMLN; every literal round-trips
// exactly through float32.  Element 0 is an unused zero.
//     cce(1,n) = n + 1                cce(2,n) = bm_r + n + 1
//     cce(3,n) = bm_r + n + 4         cce(4,n) = n + bv_c + 1
//     cce(5,n) = bm_r + n + bv_c + 1  ccg(r,n) = WGAMMA(cce(r,n))
//     ocg1(n)  = 1./ccg(1,n)          ocg2(n)  = 1./ccg(2,n)
// ---------------------------------------------------------------------------
__constant__ float THOMPSON_AA_CCE1[16] = {
    0.000000000e+00f, 2.000000000e+00f, 3.000000000e+00f, 4.000000000e+00f,
    5.000000000e+00f, 6.000000000e+00f, 7.000000000e+00f, 8.000000000e+00f,
    9.000000000e+00f, 1.000000000e+01f, 1.100000000e+01f, 1.200000000e+01f,
    1.300000000e+01f, 1.400000000e+01f, 1.500000000e+01f, 1.600000000e+01f};

__constant__ float THOMPSON_AA_CCE2[16] = {
    0.000000000e+00f, 5.000000000e+00f, 6.000000000e+00f, 7.000000000e+00f,
    8.000000000e+00f, 9.000000000e+00f, 1.000000000e+01f, 1.100000000e+01f,
    1.200000000e+01f, 1.300000000e+01f, 1.400000000e+01f, 1.500000000e+01f,
    1.600000000e+01f, 1.700000000e+01f, 1.800000000e+01f, 1.900000000e+01f};

__constant__ float THOMPSON_AA_CCE3[16] = {
    0.000000000e+00f, 8.000000000e+00f, 9.000000000e+00f, 1.000000000e+01f,
    1.100000000e+01f, 1.200000000e+01f, 1.300000000e+01f, 1.400000000e+01f,
    1.500000000e+01f, 1.600000000e+01f, 1.700000000e+01f, 1.800000000e+01f,
    1.900000000e+01f, 2.000000000e+01f, 2.100000000e+01f, 2.200000000e+01f};

__constant__ float THOMPSON_AA_CCE4[16] = {
    0.000000000e+00f, 4.000000000e+00f, 5.000000000e+00f, 6.000000000e+00f,
    7.000000000e+00f, 8.000000000e+00f, 9.000000000e+00f, 1.000000000e+01f,
    1.100000000e+01f, 1.200000000e+01f, 1.300000000e+01f, 1.400000000e+01f,
    1.500000000e+01f, 1.600000000e+01f, 1.700000000e+01f, 1.800000000e+01f};

__constant__ float THOMPSON_AA_CCE5[16] = {
    0.000000000e+00f, 7.000000000e+00f, 8.000000000e+00f, 9.000000000e+00f,
    1.000000000e+01f, 1.100000000e+01f, 1.200000000e+01f, 1.300000000e+01f,
    1.400000000e+01f, 1.500000000e+01f, 1.600000000e+01f, 1.700000000e+01f,
    1.800000000e+01f, 1.900000000e+01f, 2.000000000e+01f, 2.100000000e+01f};

__constant__ float THOMPSON_AA_CCG1[16] = {
    0.000000000e+00f, 1.000000000e+00f, 2.000000000e+00f, 6.000000000e+00f,
    2.400000000e+01f, 1.200000076e+02f, 7.200000610e+02f, 5.040001953e+03f,
    4.031999609e+04f, 3.628799688e+05f, 3.628801750e+06f, 3.991680000e+07f,
    4.790018560e+08f, 6.227022336e+09f, 8.717829734e+10f, 1.307673887e+12f};

__constant__ float THOMPSON_AA_CCG2[16] = {
    0.000000000e+00f, 2.400000000e+01f, 1.200000076e+02f, 7.200000610e+02f,
    5.040001953e+03f, 4.031999609e+04f, 3.628799688e+05f, 3.628801750e+06f,
    3.991680000e+07f, 4.790018560e+08f, 6.227022336e+09f, 8.717829734e+10f,
    1.307673887e+12f, 2.092278219e+13f, 3.556874482e+14f, 6.402383731e+15f};

__constant__ float THOMPSON_AA_CCG3[16] = {
    0.000000000e+00f, 5.040001953e+03f, 4.031999609e+04f, 3.628799688e+05f,
    3.628801750e+06f, 3.991680000e+07f, 4.790018560e+08f, 6.227022336e+09f,
    8.717829734e+10f, 1.307673887e+12f, 2.092278219e+13f, 3.556874482e+14f,
    6.402383731e+15f, 1.216452850e+17f, 2.432903398e+18f, 5.109091445e+19f};

__constant__ float THOMPSON_AA_CCG4[16] = {
    0.000000000e+00f, 6.000000000e+00f, 2.400000000e+01f, 1.200000076e+02f,
    7.200000610e+02f, 5.040001953e+03f, 4.031999609e+04f, 3.628799688e+05f,
    3.628801750e+06f, 3.991680000e+07f, 4.790018560e+08f, 6.227022336e+09f,
    8.717829734e+10f, 1.307673887e+12f, 2.092278219e+13f, 3.556874482e+14f};

__constant__ float THOMPSON_AA_CCG5[16] = {
    0.000000000e+00f, 7.200000610e+02f, 5.040001953e+03f, 4.031999609e+04f,
    3.628799688e+05f, 3.628801750e+06f, 3.991680000e+07f, 4.790018560e+08f,
    6.227022336e+09f, 8.717829734e+10f, 1.307673887e+12f, 2.092278219e+13f,
    3.556874482e+14f, 6.402383731e+15f, 1.216452850e+17f, 2.432903398e+18f};

__constant__ float THOMPSON_AA_OCG1[16] = {
    0.000000000e+00f, 1.000000000e+00f, 5.000000000e-01f, 1.666666716e-01f,
    4.166666791e-02f, 8.333332837e-03f, 1.388888806e-03f, 1.984126284e-04f,
    2.480158946e-05f, 2.755732112e-06f, 2.755730577e-07f, 2.505210794e-08f,
    2.087674478e-09f, 1.605904021e-10f, 1.147074449e-11f, 7.647166320e-13f};

__constant__ float THOMPSON_AA_OCG2[16] = {
    0.000000000e+00f, 4.166666791e-02f, 8.333332837e-03f, 1.388888806e-03f,
    1.984126284e-04f, 2.480158946e-05f, 2.755732112e-06f, 2.755730577e-07f,
    2.505210794e-08f, 2.087674478e-09f, 1.605904021e-10f, 1.147074449e-11f,
    7.647166320e-13f, 4.779478950e-14f, 2.811457147e-15f, 1.561918299e-16f};

// calc_effectRad:5611-5613.  EXACT integers (n+1)(n+2)(n+3) for n = 2..16,
// written out verbatim in WRF as a PARAMETER.  Do NOT substitute
// CCG2[n]*OCG1[n]: at n=12 the runtime product is 2729.9973, not 2730.
__constant__ float THOMPSON_AA_G_RATIO[16] = {
    0.000000000e+00f, 2.400000000e+01f, 6.000000000e+01f, 1.200000000e+02f,
    2.100000000e+02f, 3.360000000e+02f, 5.040000000e+02f, 7.200000000e+02f,
    9.900000000e+02f, 1.320000000e+03f, 1.716000000e+03f, 2.184000000e+03f,
    2.730000000e+03f, 3.360000000e+03f, 4.080000000e+03f, 4.896000000e+03f};

// activ_ncloud axis values, module_mp_thompson.F:335-344.  ta_Ra and ta_Ka
// are not needed on device because l and m are hardcoded.
__constant__ float THOMPSON_AA_TA_NA[THOMPSON_AA_NTB_ARC] = {
    10.0f, 31.6f, 100.0f, 316.0f, 1000.0f, 3160.0f, 10000.0f};

__constant__ float THOMPSON_AA_TA_WW[THOMPSON_AA_NTB_ARW] = {
    0.01f, 0.0316f, 0.1f, 0.316f, 1.0f, 3.16f, 10.0f, 31.6f, 100.0f};

// Snow moment fit coefficients sa/sb, module_mp_thompson.F:167-181.  These
// are byte-identical to thompson.cu's thompson_sa/thompson_sb; they are
// duplicated, not shared, because thompson.cu is byte-frozen.
__constant__ float THOMPSON_AA_SA[10] = {
    5.065339f, -0.062659f, -3.032362f, 0.029469f, -0.000285f,
    0.31255f, 0.000204f, 0.003199f, 0.0f, -0.015952f};

__constant__ float THOMPSON_AA_SB[10] = {
    0.476221f, -0.015896f, 0.165977f, 0.007468f, -0.000141f,
    0.060366f, 0.000079f, 0.000594f, 0.0f, -0.003577f};


// ---------------------------------------------------------------------------
// Rounding.
// ---------------------------------------------------------------------------

// Fortran NINT rounds half AWAY FROM ZERO.  CUDA's __float2int_rn rounds half
// to even and would select a different bin at an exact .5 boundary.  Every
// index in this header goes through here or through floorf(x+0.5f) where the
// argument is provably positive.
__device__ __forceinline__ int thompson_aa_nint(float x)
{
    return (int)(x >= 0.0f ? floorf(x + 0.5f) : ceilf(x - 0.5f));
}

__device__ __forceinline__ int thompson_aa_nint_double(double x)
{
    return (int)(x >= 0.0 ? floor(x + 0.5) : ceil(x - 0.5));
}


// ---------------------------------------------------------------------------
// Correctly-rounded float32 transcendentals.
// ---------------------------------------------------------------------------
//
// gfortran lowers a REAL(4) EXP/LOG/** to glibc's expf/logf/powf, which are
// correctly rounded (<= 0.5 ulp).  CUDA's expf/logf/powf carry up to ~2 ulp.
// That is normally invisible, but iceKoop forms `1. - exp(-x)` with x ~ 1e-14,
// where the cancellation amplifies a single expf ulp into an O(1) relative
// error in the returned ice number.  Evaluating in double and rounding ONCE
// to float reproduces the correctly-rounded float32 result, and MEASURES
// bit-exact against all 480 rows of probe-icekoop.csv and all 320 rows of
// probe-icedemott.csv (CUDA's own powf/expf do not).
//
// This is not a change of formula or of precision: every operand and every
// stored result is still float32, exactly as WRF's REAL(4) declarations
// require.  thompson_field_a/thompson_field_b keep thompson.cu's powf; the
// saturation fits thompson_rslf/thompson_rsif no longer do -- see the
// "THE SHARED FITS" note below for why mp=28 diverges from mp=8 there.
__device__ __forceinline__ float thompson_aa_expf_cr(float x)
{
    return (float)exp((double)x);
}

__device__ __forceinline__ float thompson_aa_logf_cr(float x)
{
    return (float)log((double)x);
}

__device__ __forceinline__ float thompson_aa_powf_cr(float x, float y)
{
    return (float)pow((double)x, (double)y);
}


// ---------------------------------------------------------------------------
// Contraction-pinned float32 arithmetic.
// ---------------------------------------------------------------------------
//
// nvrtc defaults to --fmad=true and will fuse a*b+c into one FMA with a
// single rounding.  build_aero.sh compiles the oracle with plain
// `gfortran -O2` and no -march, i.e. baseline x86-64 with no FMA
// instruction, so every REAL(4) multiply and add in WRF is separately
// rounded.  Measured: with contraction left on, activ_ncloud, iceKoop and
// Eff_aero disagree with the Fortran probe in the last one or two float
// digits; with every operation pinned they agree BIT-EXACTLY on all 1320 /
// 480 / 48 probe rows.
//
// Pinning it here rather than with a -fmad=false compile option keeps
// gpuwm/core/kernels/__init__.py's change to the _EXTRA_HEADERS dict alone,
// and gives the five aerosol kernel packages the property automatically.
// Repo precedent: noahmp_bareflux.cu, noahmp_thermal.cu.
//
// These are applied to the aerosol-only helpers (activ_ncloud, iceDeMott,
// iceKoop, Eff_aero), which have no mp=8 counterpart, AND -- deliberately --
// to thompson_rslf/thompson_rsif, which do.
//
// THE SHARED FITS: WHY mp=28 DIVERGES FROM ITS mp=8 SIBLING HERE.
// This header originally kept thompson_rslf/rsif as verbatim copies of
// thompson.cu's plain Horner chains, so that the aerosol port and the
// model-validated mp=8 port would evaluate the shared fits identically.
// That is the wrong tie-break, and mp=28 is where it becomes visible.
//
// The authority is WRF, not ArWen's mp=8.  nvrtc contracts the plain chain
// into FMAs; the Fortran WRF the oracle is built from has no FMA instruction
// at all, so the contracted device fit lands ONE float32 ulp low -- measured
// at 23 of 24 levels of the aero-nc-sed entry column.  In mp=8 that is a
// sub-atol mass error and nothing notices.  In mp=28 it is not a rate
// perturbation at all: module_mp_thompson.F:3400 opens the ENTIRE
// condensation and CCN-activation block on `ssatw(k) .gt. eps` with
// eps = 1.E-15 (:185), so one ulp FLIPS A BRANCH.  The call then condenses
// water at every cloud-free level of a saturated column and activates
// droplets from aerosol that WRF never touches.  Because activation is a
// ONE-WAY SINK, the error does not average out -- it destroyed 33-56% of a
// column's CCN in a single 10-50 s step.
//
// So these two fits are pinned and mp=8's are not.  The two ports therefore
// evaluate RSLF/RSIF differently BY DESIGN: mp=28 matches WRF, mp=8 stays
// frozen and byte-identical to its validated trajectory.  Do not "restore
// consistency" by unpinning these -- consistency with a sibling is not the
// goal, agreement with WRF is.  thompson.cu carries the same unpinned chain
// at :48-58 and is a pre-existing ArWen-wide deviation from WRF that this
// port merely made observable; correcting it there is a separate decision
// about a model-validated trajectory and is NOT in this port's scope.
//
// THE SAME TIE-BREAK NOW APPLIES TO thompson_field_a/thompson_field_b.  They
// used to keep thompson.cu's plain form on the grounds that nothing had
// measured them.  They have now been measured, against a `gfortran -O2`
// transcription of module_mp_thompson.F:2069-2079 AND against the real
// (PUBLIC) calc_effectRad, and the copied mp=8 form was wrong by up to
// 3.267395e-06 on a_ and 5.870704e-06 on the effs_m it feeds -- from a broken
// operator association as much as from FMA and pow.  Both are now WRF's form
// and both are bit-exact.  See the block above thompson_field_a for the
// numbers.  mp=8 keeps thompson.cu:81-107 frozen; mp=28 matches WRF.
__device__ __forceinline__ float thompson_aa_add(float a, float b)
{
    return __fadd_rn(a, b);
}

__device__ __forceinline__ float thompson_aa_sub(float a, float b)
{
    return __fsub_rn(a, b);
}

__device__ __forceinline__ float thompson_aa_mul(float a, float b)
{
    return __fmul_rn(a, b);
}

__device__ __forceinline__ float thompson_aa_div(float a, float b)
{
    return __fdiv_rn(a, b);
}


// ---------------------------------------------------------------------------
// WRF's terminal clamps.  :1805-1806 (entry), :3217/:3486 (working refresh),
// :3979-3981 (terminal apply).  Reproduced here so five packages cannot each
// invent a slightly different one.
// ---------------------------------------------------------------------------

__device__ __forceinline__ float thompson_aa_clamp_nc(float nc_m3)
{
    return fmaxf(THOMPSON_AA_NC_FLOOR, fminf(nc_m3, THOMPSON_AA_NT_C_MAX));
}

__device__ __forceinline__ float thompson_aa_clamp_nwfa(float nwfa_m3)
{
    return fmaxf(THOMPSON_AA_NWFA_FLOOR,
                 fminf(THOMPSON_AA_AERO_CEIL, nwfa_m3));
}

__device__ __forceinline__ float thompson_aa_clamp_nifa(float nifa_m3)
{
    return fmaxf(THOMPSON_AA_NIFA_FLOOR,
                 fminf(THOMPSON_AA_AERO_CEIL, nifa_m3));
}


// ---------------------------------------------------------------------------
// nu_c and the lookup indices mp=8 froze into constants.
// ---------------------------------------------------------------------------

// module_mp_thompson.F:1832, :2171, :3002, :3658.
//     nu_c = MIN(15, NINT(1000.E6/nc(k)) + 2)
// nc is positive at every call site (callers clamp to >= 2 first), so
// floorf(x+0.5f) is the faithful NINT.  The result is in [2, 15]; at
// nc = Nt_c = 100e6 it is exactly 12, which is the value mp=8 froze.
__device__ __forceinline__ int thompson_aa_nu_c(float nc_m3)
{
    return min(15, (int)floorf(1000.0e6f / nc_m3 + 0.5f) + 2);
}

// module_mp_thompson.F:2170, the WORKING nu_c.  Identical arithmetic to
// thompson_aa_nu_c; the separate name exists so that every call site records
// WHICH nc it passed.  WRF recomputes nu_c here from the POST-rediagnosis
// nc(k) assigned at :1840, NOT from the entry nc(k) of :1829 that produced
// thompson_aa_cloud_dist's nu_c_entry_out.  See the nu_c STAGING RULE at the
// top of this header; the two differ wherever the :1834-1838 droplet-size
// clamp engages, and the gamma columns they select move by more than ten
// orders of magnitude between them.
__device__ __forceinline__ int thompson_aa_nu_c_working(
    float nc_m3_after_rediagnosis)
{
    return thompson_aa_nu_c(nc_m3_after_rediagnosis);
}

// module_mp_thompson.F:3447-3448.
//     idx_n = NINT(1.0 + FLOAT(nbc) * DLOG(nc(k)/t_Nc(1)) / nic1)
//     idx_n = MAX(1, MIN(idx_n, nbc))
// t_Nc is DOUBLE PRECISION and nic1 is the truncated INTEGER 7, so the whole
// expression is double.  Returned ZERO-BASED for C indexing.
// HARD GATE: thompson_aa_droplet_bin(100.0e6f) == 65.
__device__ __forceinline__ int thompson_aa_droplet_bin(float nc_m3)
{
    const double raw = 1.0
        + 100.0 * log((double)nc_m3 / THOMPSON_AA_T_NC_1)
          / (double)THOMPSON_AA_NIC1;
    const int one_based = thompson_aa_nint_double(raw);
    return max(1, min(one_based, THOMPSON_AA_NBC)) - 1;
}

// Transcribed from thompson.cu:3063-3082 (mp=8's copy of :2282-2307), which
// is the same NINT(log10)-then-truncate-the-mantissa pattern WRF uses for
// idx_c, idx_r and idx_IN.  Returned ZERO-BASED.
__device__ __forceinline__ int thompson_aa_decade_index(
    float value, int first_exponent, int table_size)
{
    const int center = (int)roundf(log10f(value));
    int exponent = center;
    for (int candidate = center - 1; candidate <= center + 1; ++candidate) {
        const float scale = powf(10.0f, (float)candidate);
        const float mantissa = value / scale;
        if (mantissa >= 1.0f && mantissa < 10.0f) {
            exponent = candidate;
            break;
        }
    }
    const float scale = powf(10.0f, (float)exponent);
    const int digit = (int)(value / scale);
    const int one_based = digit + 9 * (exponent - first_exponent);
    return max(0, min(one_based - 1, table_size - 1));
}

// The DOUBLE form, thompson.cu:3084-3105.  WRF keeps the rain and graupel
// y-intercepts in DOUBLE PRECISION (N0_r, N0_g at :1587) and indexes
// t_Nor/t_Nog with them, so the mantissa split has to be done in double; the
// base-ten SCALE stays default REAL, which is why `scale` below is a float
// and the division that uses it is widened rather than the other way round.
//
// PROMOTED, wave 4.  cold.cu:206-223 and warm.cu:272-289 each carried this
// body, byte-identical to each other and to what is written here (verified by
// diff before the move).  There was no divergence yet -- and that is the
// point: thompson_aa_entry_rain_distribution had none either, right up until
// it did.  The two local copies MUST NOW BE DELETED; until they are, those
// two translation units fail to compile with
//     error: function "thompson_aa_decade_index_double" has already been
//     defined (previous definition at line 23)
// which is the correct and intended outcome for a helper whose shared
// signature matches the local one exactly.  MEASURED: with both local copies
// removed all six aerosol modules compile, and the full 19-fixture G3 table
// is BIT-IDENTICAL to the pre-promotion tree.  Do not rename either copy, and
// do not delete this definition to make the build go green.
__device__ __forceinline__ int thompson_aa_decade_index_double(
    double value, int first_exponent, int table_size)
{
    const int center = (int)round(log10(value));
    int exponent = center;
    for (int candidate = center - 1; candidate <= center + 1; ++candidate) {
        const float scale = powf(10.0f, (float)candidate);
        const double mantissa = value / (double)scale;
        if (mantissa >= 1.0 && mantissa < 10.0) {
            exponent = candidate;
            break;
        }
    }
    const float scale = powf(10.0f, (float)exponent);
    const int digit = (int)(value / (double)scale);
    const int one_based = digit + 9 * (exponent - first_exponent);
    return max(0, min(one_based - 1, table_size - 1));
}

// module_mp_thompson.F:2579-2591.
//     if (xni .gt. Nt_IN(1)) then     ! Nt_IN(1) = 1.0
//        idx_IN = INT(xni/10.**n) + 10*(n-niin2) - (n-niin2)
//        idx_IN = MAX(1, MIN(idx_IN, ntb_IN))
//     else
//        idx_IN = 1
// niin2 = NINT(ALOG10(Nt_IN(1))) = 0 (:828), and
// 10*(n-niin2) - (n-niin2) == 9*(n-niin2), i.e. first_exponent = 0.
// Returned ZERO-BASED.
// HARD GATE: thompson_aa_in_bin(1000.0f) == 27, the nuclei_bin thompson.cu
// hardcodes at :3936 for its fixed 1-per-litre default.
__device__ __forceinline__ int thompson_aa_in_bin(float xni)
{
    if (xni > 1.0f) {
        return thompson_aa_decade_index(xni, 0, THOMPSON_AA_NTB_IN);
    }
    return 0;
}


// ---------------------------------------------------------------------------
// Saturation vapour mixing ratios and the snow-moment power-law fits.
// Transcribed (not shared) from module_mp_thompson.F; byte-for-byte the same
// arithmetic thompson.cu:18-77 performs, because thompson.cu may not be
// edited and there is no #include path under RawModule.
// ---------------------------------------------------------------------------

__device__ __forceinline__ float thompson_rslf(float pressure, float temp)
{
    // module_mp_thompson.F:5378-5413.  Preserve WRF's default-REAL Horner
    // order exactly.
    const float c0 = 0.611583699e3f;
    const float c1 = 0.444606896e2f;
    const float c2 = 0.143177157e1f;
    const float c3 = 0.264224321e-1f;
    const float c4 = 0.299291081e-3f;
    const float c5 = 0.203154182e-5f;
    const float c6 = 0.702620698e-8f;
    const float c7 = 0.379534310e-11f;
    const float c8 = -0.321582393e-13f;
    const float x = fmaxf(-80.0f, temp - 273.16f);
    // CONTRACTION-PINNED, and deliberately NOT identical to thompson.cu's
    // plain chain.  See the "shared fits" note above thompson_aa_add.
    float esl = thompson_aa_add(c0, thompson_aa_mul(x, thompson_aa_add(c1, thompson_aa_mul(x, thompson_aa_add(c2, thompson_aa_mul(x, thompson_aa_add(c3, thompson_aa_mul(x, thompson_aa_add(c4, thompson_aa_mul(x, thompson_aa_add(c5, thompson_aa_mul(x, thompson_aa_add(c6, thompson_aa_mul(x, thompson_aa_add(c7, thompson_aa_mul(x, c8))))))))))))))));
    esl = fminf(esl, pressure * 0.15f);
    return 0.622f * esl / (pressure - esl);
}

__device__ __forceinline__ float thompson_rsif(float pressure, float temp)
{
    // module_mp_thompson.F:5414-5446.
    const float c0 = 0.609868993e3f;
    const float c1 = 0.499320233e2f;
    const float c2 = 0.184672631e1f;
    const float c3 = 0.402737184e-1f;
    const float c4 = 0.565392987e-3f;
    const float c5 = 0.521693933e-5f;
    const float c6 = 0.307839583e-7f;
    const float c7 = 0.105785160e-9f;
    const float c8 = 0.161444444e-12f;
    const float x = fmaxf(-80.0f, temp - 273.16f);
    // CONTRACTION-PINNED, as thompson_rslf.  Same reasoning.
    float esi = thompson_aa_add(c0, thompson_aa_mul(x, thompson_aa_add(c1, thompson_aa_mul(x, thompson_aa_add(c2, thompson_aa_mul(x, thompson_aa_add(c3, thompson_aa_mul(x, thompson_aa_add(c4, thompson_aa_mul(x, thompson_aa_add(c5, thompson_aa_mul(x, thompson_aa_add(c6, thompson_aa_mul(x, thompson_aa_add(c7, thompson_aa_mul(x, c8))))))))))))))));
    esi = fminf(esi, pressure * 0.15f);
    return 0.622f * esi / fmaxf(1.0e-4f, pressure - esi);
}

// THE FIELD ET AL (2005) SNOW-MOMENT FITS.  WRF has no `field_a` procedure:
// the ten-term chain is written out INLINE at :2036-2046, :2050-2053,
// :2056-2064, :2069-2079, :2091-2100, :2103-2112, :2116-2125, :3332-3350,
// :4447-4470, :5670-5680 and :5684-5693, always with the SAME operator tree
// and only the moment symbol changing (bm_s, cse(1), cse(13), cse(14),
// cse(16), cse(17), or a literal 1. / absent factor).  These two helpers are
// that tree with the moment as an argument.
//
// TWO DEFECTS WERE MEASURED HERE AND FIXED, AND THE FORM BELOW IS NOT
// thompson.cu's.  The earlier body was a verbatim copy of thompson.cu:81-107
// and it disagreed with WRF in two independent ways:
//
//   1. ASSOCIATION.  WRF writes `sa(5)*tc0*tc0`, which Fortran evaluates
//      LEFT TO RIGHT as (sa5*tc0)*tc0.  The copied form hoisted `tc2 = tc*tc`
//      and computed sa5*(tc*tc).  In float32 with no FMA those are different
//      numbers.  The same mismatch applied to sa(6)*m*m, sa(7)*tc0*tc0*m,
//      sa(8)*tc0*m*m, sa(9)*tc0**3 and sa(10)*m**3 -- six of the ten terms.
//   2. TRANSCENDENTAL.  `a_ = 10.0**loga_` is REAL(4)**REAL(4), which
//      gfortran lowers to glibc's correctly-rounded powf; CUDA's powf carries
//      several ulp.
//
// Plus nvrtc's default FMA contraction across the nine additions, which the
// baseline-x86-64 `gfortran -O2` the oracle is built with cannot do.
//
// MEASURED on an RTX 5090 against a `gfortran -O2 -ffree-form` transcription
// of :2069-2079 (same compiler and flags as
// tools/thompson_wrf461_oracle/build_aero.sh, i.e. no FMA instruction),
// over 253 states = 23 temperatures from -0.1 to -70 C x 11 moments covering
// every one the mp=28 kernels ask for (0, 1, 1.775, 2.55, 3):
//     a_  old form: 118/253 exact, max 3.267395e-06 relative
//     a_  this form: 253/253 BIT-EXACT
//     b_  old form: 180/253 exact, max 4.411423e-07 relative
//     b_  this form: 253/253 BIT-EXACT
// Restoring only the correctly-rounded pow, without the association fix,
// changes nothing (126/253, still 3.267e-06): the association is the larger
// error and both had to go.
//
// AND AGAINST A TRUE WRF ORACLE, not a transcription: calc_effectRad is
// PUBLIC, so its snow branch (:5658-5695) can be called directly over a
// 24-temperature x 15-snow-mass grid.  360 rows, 299 of them strictly inside
// the [5.01, 999] um clamp:
//     old fits: 138/360 exact, max 5.870704e-06 relative
//     these fits + a correctly-rounded smo2**b_: 360/360 BIT-EXACT
// The committed gpuwm/data/thompson/oracle-aero/probe-effectrad.csv could not
// see this: all 14 of its rows carry t = 285 K and qs = 2e-4, so its effs_m
// column is ONE state repeated fourteen times.
//
// This is the same tie-break the "THE SHARED FITS" note above thompson_aa_add
// records for thompson_rslf/thompson_rsif: the authority is WRF, not ArWen's
// mp=8 port.  thompson.cu keeps its plain chain and stays byte-frozen; mp=28
// matches WRF.  Do not "restore consistency" with thompson.cu:81-107.
__device__ __forceinline__ float thompson_aa_field_loga(float tc0, float mom)
{
    // module_mp_thompson.F:2069-2074, term for term, left to right.
    float v = THOMPSON_AA_SA[0];
    v = thompson_aa_add(v, thompson_aa_mul(THOMPSON_AA_SA[1], tc0));
    v = thompson_aa_add(v, thompson_aa_mul(THOMPSON_AA_SA[2], mom));
    v = thompson_aa_add(v, thompson_aa_mul(
        thompson_aa_mul(THOMPSON_AA_SA[3], tc0), mom));
    v = thompson_aa_add(v, thompson_aa_mul(
        thompson_aa_mul(THOMPSON_AA_SA[4], tc0), tc0));
    v = thompson_aa_add(v, thompson_aa_mul(
        thompson_aa_mul(THOMPSON_AA_SA[5], mom), mom));
    v = thompson_aa_add(v, thompson_aa_mul(thompson_aa_mul(
        thompson_aa_mul(THOMPSON_AA_SA[6], tc0), tc0), mom));
    v = thompson_aa_add(v, thompson_aa_mul(thompson_aa_mul(
        thompson_aa_mul(THOMPSON_AA_SA[7], tc0), mom), mom));
    v = thompson_aa_add(v, thompson_aa_mul(thompson_aa_mul(
        thompson_aa_mul(THOMPSON_AA_SA[8], tc0), tc0), tc0));
    v = thompson_aa_add(v, thompson_aa_mul(thompson_aa_mul(
        thompson_aa_mul(THOMPSON_AA_SA[9], mom), mom), mom));
    return v;
}

__device__ __forceinline__ float thompson_field_a(float tc, float moment)
{
    // :2075  a_ = 10.0**loga_   (REAL(4)**REAL(4) -> glibc powf)
    return thompson_aa_powf_cr(10.0f, thompson_aa_field_loga(tc, moment));
}

__device__ __forceinline__ float thompson_field_b(float tc, float moment)
{
    // module_mp_thompson.F:2076-2079, term for term, left to right.
    float v = THOMPSON_AA_SB[0];
    v = thompson_aa_add(v, thompson_aa_mul(THOMPSON_AA_SB[1], tc));
    v = thompson_aa_add(v, thompson_aa_mul(THOMPSON_AA_SB[2], moment));
    v = thompson_aa_add(v, thompson_aa_mul(
        thompson_aa_mul(THOMPSON_AA_SB[3], tc), moment));
    v = thompson_aa_add(v, thompson_aa_mul(
        thompson_aa_mul(THOMPSON_AA_SB[4], tc), tc));
    v = thompson_aa_add(v, thompson_aa_mul(
        thompson_aa_mul(THOMPSON_AA_SB[5], moment), moment));
    v = thompson_aa_add(v, thompson_aa_mul(thompson_aa_mul(
        thompson_aa_mul(THOMPSON_AA_SB[6], tc), tc), moment));
    v = thompson_aa_add(v, thompson_aa_mul(thompson_aa_mul(
        thompson_aa_mul(THOMPSON_AA_SB[7], tc), moment), moment));
    v = thompson_aa_add(v, thompson_aa_mul(thompson_aa_mul(
        thompson_aa_mul(THOMPSON_AA_SB[8], tc), tc), tc));
    v = thompson_aa_add(v, thompson_aa_mul(thompson_aa_mul(
        thompson_aa_mul(THOMPSON_AA_SB[9], moment), moment), moment));
    return v;
}


// ---------------------------------------------------------------------------
// CCN activation, module_mp_thompson.F:5178-5253.
// ---------------------------------------------------------------------------
//
// tnccn_act is WP-01's float64 Fortran-order (7,9,7,5,4) device array.  The
// Fortran linear index of tnccn_act(i,j,k,l,m) with one-based subscripts is
//     (i-1) + 7*((j-1) + 9*((k-1) + 7*((l-1) + 5*(m-1))))
// Values are cast to float on load; every blend operation is float, matching
// WRF's REAL(KIND=R4SIZE) table (:393) and default-REAL arithmetic.
__device__ __forceinline__ float thompson_activ_ncloud(
    float Tt, float Ww, float NCCN, const double* __restrict__ tnccn_act)
{
    // Asymmetric epsilons: -1.0/+1.0 on the CCN axis, -1.0/+0.001 on the
    // updraft axis.  Reproduce literally; they are not a symmetric guard.
    float n_local = thompson_aa_mul(NCCN, 1.0e-6f);
    if (n_local >= THOMPSON_AA_TA_NA[THOMPSON_AA_NTB_ARC - 1]) {
        n_local = THOMPSON_AA_TA_NA[THOMPSON_AA_NTB_ARC - 1] - 1.0f;
    } else if (n_local <= THOMPSON_AA_TA_NA[0]) {
        n_local = THOMPSON_AA_TA_NA[0] + 1.0f;
    }
    // Fortran DO/GOTO bracket search, i.e. a LINEAR SCAN, not a bisection.
    // WRF's fallthrough leaves n == ntb_arc+1 and would index one past the
    // end; the clamps above make that unreachable for any finite input
    // (11 -> i=2, 9999 -> i=7), so the initializer is pinned at ntb_arc to
    // degrade in bounds rather than read out of bounds on a NaN.
    int i = THOMPSON_AA_NTB_ARC;
    for (int n = 2; n <= THOMPSON_AA_NTB_ARC; ++n) {
        if (n_local >= THOMPSON_AA_TA_NA[n - 2]
                && n_local < THOMPSON_AA_TA_NA[n - 1]) {
            i = n;
            break;
        }
    }
    const float x1 = thompson_aa_logf_cr(THOMPSON_AA_TA_NA[i - 2]);
    const float x2 = thompson_aa_logf_cr(THOMPSON_AA_TA_NA[i - 1]);

    float w_local = Ww;
    if (w_local >= THOMPSON_AA_TA_WW[THOMPSON_AA_NTB_ARW - 1]) {
        w_local = THOMPSON_AA_TA_WW[THOMPSON_AA_NTB_ARW - 1] - 1.0f;
    } else if (w_local <= THOMPSON_AA_TA_WW[0]) {
        w_local = THOMPSON_AA_TA_WW[0] + 0.001f;
    }
    int j = THOMPSON_AA_NTB_ARW;
    for (int n = 2; n <= THOMPSON_AA_NTB_ARW; ++n) {
        if (w_local >= THOMPSON_AA_TA_WW[n - 2]
                && w_local < THOMPSON_AA_TA_WW[n - 1]) {
            j = n;
            break;
        }
    }
    const float y1 = thompson_aa_logf_cr(THOMPSON_AA_TA_WW[j - 2]);
    const float y2 = thompson_aa_logf_cr(THOMPSON_AA_TA_WW[j - 1]);

    // NEAREST-NEIGHBOUR in temperature over a 10 K grid.  There is no
    // interpolation here, so activated number is a STEP function of T.
    const int k = max(1, min(
        thompson_aa_nint(
            thompson_aa_mul(thompson_aa_sub(Tt, 243.15f), 0.1f)) + 1,
        THOMPSON_AA_NTB_ART));

    const int l = THOMPSON_AA_ACTIV_L;
    const int m = THOMPSON_AA_ACTIV_M;
    const int tail = (k - 1)
        + THOMPSON_AA_NTB_ART * ((l - 1) + THOMPSON_AA_NTB_ARR * (m - 1));
    const int base = THOMPSON_AA_NTB_ARC * THOMPSON_AA_NTB_ARW * tail;
    const int row_lo = base + THOMPSON_AA_NTB_ARC * (j - 2);
    const int row_hi = base + THOMPSON_AA_NTB_ARC * (j - 1);

    const float A = (float)tnccn_act[row_lo + (i - 2)];
    const float B = (float)tnccn_act[row_lo + (i - 1)];
    const float C = (float)tnccn_act[row_hi + (i - 1)];
    const float D = (float)tnccn_act[row_hi + (i - 2)];

    const float nx = thompson_aa_logf_cr(n_local);
    const float wy = thompson_aa_logf_cr(w_local);
    const float t = thompson_aa_div(thompson_aa_sub(nx, x1),
                                    thompson_aa_sub(x2, x1));
    const float u = thompson_aa_div(thompson_aa_sub(wy, y1),
                                    thompson_aa_sub(y2, y1));
    const float t1 = thompson_aa_sub(1.0f, t);
    const float u1 = thompson_aa_sub(1.0f, u);
    // Fortran's left-to-right sum of four separately rounded products.
    float fraction = thompson_aa_mul(thompson_aa_mul(t1, u1), A);
    fraction = thompson_aa_add(
        fraction, thompson_aa_mul(thompson_aa_mul(t, u1), B));
    fraction = thompson_aa_add(
        fraction, thompson_aa_mul(thompson_aa_mul(t, u), C));
    fraction = thompson_aa_add(
        fraction, thompson_aa_mul(thompson_aa_mul(t1, u), D));
    return thompson_aa_mul(NCCN, fraction);
}

__device__ __forceinline__ float thompson_aa_activ_ncloud(
    float Tt, float Ww, float NCCN, const double* __restrict__ tnccn_act)
{
    return thompson_activ_ncloud(Tt, Ww, NCCN, tnccn_act);
}


// ---------------------------------------------------------------------------
// DeMott (2010) ice nucleation, module_mp_thompson.F:5448-5518.
// ---------------------------------------------------------------------------
//
// NEGATIVE FINDING, verified in v4.6.1: the Phillips (2008) branch at
// 5474-5505 is entirely commented out, together with the satw/sati/siw
// diagnostics that fed it.  What remains is unconditional and depends only
// on (tempc, rho, nifa); qv, qvs and qvsi are dead formal arguments and are
// deliberately NOT parameters here.
__device__ __forceinline__ float thompson_ice_demott(
    float tempc, float rho, float nifa_m3)
{
    // nifa_cc = MAX(0.5, nifa*RHO_NOT0*1.E-6/rho)
    const float nifa_cc = fmaxf(
        0.5f,
        thompson_aa_div(
            thompson_aa_mul(
                thompson_aa_mul(nifa_m3, THOMPSON_AA_RHO_NOT0), 1.0e-6f),
            rho));
    // xni = (5.94e-5*(-tempc)**3.33) * (nifa_cc**((-0.0264*tempc)+0.0033))
    const float exponent = thompson_aa_add(
        thompson_aa_mul(-0.0264f, tempc), 0.0033f);
    float xni = thompson_aa_mul(
        thompson_aa_mul(5.94e-5f, thompson_aa_powf_cr(-tempc, 3.33f)),
        thompson_aa_powf_cr(nifa_cc, exponent));
    // xni = xni*rho/RHO_NOT0 * 1000.
    xni = thompson_aa_mul(
        thompson_aa_div(thompson_aa_mul(xni, rho), THOMPSON_AA_RHO_NOT0),
        1000.0f);
    return fmaxf(0.0f, xni);
}

__device__ __forceinline__ float thompson_aa_ice_demott(
    float tempc, float rho, float nifa_m3)
{
    return thompson_ice_demott(tempc, rho, nifa_m3);
}


// ---------------------------------------------------------------------------
// Koop et al (2001) homogeneous haze freezing, module_mp_thompson.F:5521-5546.
// ---------------------------------------------------------------------------
//
// The caller owns WRF's gate at :2634-2637 (is_aerosol_aware .AND. homogIce
// .AND. ns+ni <= 999.e3 .AND. temp < 238 .AND. ssati >= 0.4).  This helper
// evaluates the rate unconditionally.
__device__ __forceinline__ float thompson_ice_koop(
    float temp, float qv, float qvs, float naero, float dt)
{
    const float satw = thompson_aa_div(qv, qvs);
    // mu_diff = 210368.0 + (131.438*temp) - (3.32373E6/temp)
    //           - (41729.1*alog(temp))
    float mu_diff = thompson_aa_add(
        210368.0f, thompson_aa_mul(131.438f, temp));
    mu_diff = thompson_aa_sub(mu_diff, thompson_aa_div(3.32373e6f, temp));
    mu_diff = thompson_aa_sub(
        mu_diff, thompson_aa_mul(41729.1f, thompson_aa_logf_cr(temp)));
    const float a_w_i = thompson_aa_expf_cr(
        thompson_aa_div(mu_diff,
                        thompson_aa_mul(THOMPSON_AA_R_UNI, temp)));
    const float d = thompson_aa_sub(satw, a_w_i);
    // log_J_rate = -906.7 + 8502*d - 26924*d*d + 29180*d*d*d, with Fortran's
    // left-to-right products (26924.0*d)*d and ((29180.0*d)*d)*d.
    float log_J_rate = thompson_aa_add(
        -906.7f, thompson_aa_mul(8502.0f, d));
    log_J_rate = thompson_aa_sub(
        log_J_rate, thompson_aa_mul(thompson_aa_mul(26924.0f, d), d));
    log_J_rate = thompson_aa_add(
        log_J_rate,
        thompson_aa_mul(
            thompson_aa_mul(thompson_aa_mul(29180.0f, d), d), d));
    log_J_rate = fminf(20.0f, log_J_rate);
    const float J_rate = thompson_aa_powf_cr(10.0f, log_J_rate);  // cm-3 s-1
    // `1. - exp(-x)` with x ~ 1e-14: prob_h is quantized to multiples of
    // 2^-24 here, so this subtraction is where WRF's own REAL(4) evaluation
    // becomes ulp-sensitive.  See thompson_aa_expf_cr above.
    const float koop_arg = thompson_aa_mul(
        thompson_aa_mul(-J_rate, THOMPSON_AA_AR_VOLUME), dt);
    const float prob_h = fminf(
        thompson_aa_sub(1.0f, thompson_aa_expf_cr(koop_arg)), 1.0f);
    float xni = 0.0f;
    if (prob_h > 0.0f) {
        xni = fminf(thompson_aa_mul(prob_h, naero), 1000.0e3f);
    }
    return fmaxf(0.0f, xni);
}

__device__ __forceinline__ float thompson_aa_ice_koop(
    float temp, float qv, float qvs, float naero, float dt)
{
    return thompson_ice_koop(temp, qv, qvs, naero, dt);
}


// ---------------------------------------------------------------------------
// Aerosol collection efficiency, module_mp_thompson.F:4965-5001.
// ---------------------------------------------------------------------------

__device__ __forceinline__ float thompson_eff_aero(
    float D, float Da, float visc, float rhoa, float Temp, int species)
{
    const float boltzman = 1.3806503e-23f;
    const float meanPath = 0.0256e-6f;

    float vt = 1.0f;
    if (species == THOMPSON_AA_SPECIES_RAIN) {
        // -0.1021 + 4.932E3*D - 0.9551E6*D*D + 0.07934E9*D*D*D
        //         - 0.002362E12*D*D*D*D, all products left-associated.
        vt = thompson_aa_add(-0.1021f, thompson_aa_mul(4.932e3f, D));
        vt = thompson_aa_sub(
            vt, thompson_aa_mul(thompson_aa_mul(0.9551e6f, D), D));
        vt = thompson_aa_add(
            vt, thompson_aa_mul(
                thompson_aa_mul(thompson_aa_mul(0.07934e9f, D), D), D));
        vt = thompson_aa_sub(
            vt, thompson_aa_mul(
                thompson_aa_mul(
                    thompson_aa_mul(thompson_aa_mul(0.002362e12f, D), D), D),
                D));
    } else if (species == THOMPSON_AA_SPECIES_SNOW) {
        vt = thompson_aa_mul(
            THOMPSON_AA_AV_S, thompson_aa_powf_cr(D, THOMPSON_AA_BV_S));
    } else if (species == THOMPSON_AA_SPECIES_GRAUPEL) {
        vt = thompson_aa_mul(
            THOMPSON_AA_AV_G, thompson_aa_powf_cr(D, THOMPSON_AA_BV_G));
    }

    // Cc = 1. + 2.*meanPath/Da * (1.257 + 0.4*exp(-0.55*Da/meanPath))
    const float slip = thompson_aa_add(
        1.257f,
        thompson_aa_mul(
            0.4f,
            thompson_aa_expf_cr(
                thompson_aa_div(thompson_aa_mul(-0.55f, Da), meanPath))));
    const float Cc = thompson_aa_add(
        1.0f,
        thompson_aa_mul(
            thompson_aa_div(thompson_aa_mul(2.0f, meanPath), Da), slip));
    // diff = boltzman*Temp*Cc/(3.*PI*visc*Da)
    const float diff = thompson_aa_div(
        thompson_aa_mul(thompson_aa_mul(boltzman, Temp), Cc),
        thompson_aa_mul(
            thompson_aa_mul(thompson_aa_mul(3.0f, THOMPSON_AA_PI), visc),
            Da));

    // Re = 0.5*rhoa*D*vt/visc ;  Sc = visc/(rhoa*diff)
    const float Re = thompson_aa_div(
        thompson_aa_mul(
            thompson_aa_mul(thompson_aa_mul(0.5f, rhoa), D), vt),
        visc);
    const float Sc = thompson_aa_div(visc, thompson_aa_mul(rhoa, diff));

    // St = Da*Da*vt*1000./(9.*visc*D)
    const float St = thompson_aa_div(
        thompson_aa_mul(
            thompson_aa_mul(thompson_aa_mul(Da, Da), vt), 1000.0f),
        thompson_aa_mul(thompson_aa_mul(9.0f, visc), D));
    const float aval = thompson_aa_add(
        1.0f, thompson_aa_logf_cr(thompson_aa_add(1.0f, Re)));
    // St2 = (1.2 + 1./12.*aval)/(1.+aval)
    const float St2 = thompson_aa_div(
        thompson_aa_add(1.2f, thompson_aa_mul(1.0f / 12.0f, aval)),
        thompson_aa_add(1.0f, aval));

    const float sqrt_re = sqrtf(Re);
    // 1. + 0.4*SQRT(Re)*Sc**0.3333 + 0.16*SQRT(Re)*SQRT(Sc)
    float brownian = thompson_aa_add(
        1.0f,
        thompson_aa_mul(thompson_aa_mul(0.4f, sqrt_re),
                        thompson_aa_powf_cr(Sc, 0.3333f)));
    brownian = thompson_aa_add(
        brownian,
        thompson_aa_mul(thompson_aa_mul(0.16f, sqrt_re), sqrtf(Sc)));
    const float term1 = thompson_aa_mul(
        thompson_aa_div(4.0f, thompson_aa_mul(Re, Sc)), brownian);
    // 4.*Da/D * (0.02 + Da/D*(1.+2.*SQRT(Re)))
    const float ratio = thompson_aa_div(Da, D);
    const float term2 = thompson_aa_mul(
        thompson_aa_div(thompson_aa_mul(4.0f, Da), D),
        thompson_aa_add(
            0.02f,
            thompson_aa_mul(
                ratio,
                thompson_aa_add(1.0f, thompson_aa_mul(2.0f, sqrt_re)))));
    float Eff = thompson_aa_add(term1, term2);

    if (St > St2) {
        const float excess = thompson_aa_sub(St, St2);
        Eff = thompson_aa_add(
            Eff,
            thompson_aa_powf_cr(
                thompson_aa_div(excess,
                                thompson_aa_add(excess, 0.666667f)),
                1.5f));
    }
    return fmaxf(1.0e-5f, fminf(Eff, 1.0f));
}

__device__ __forceinline__ float thompson_aa_eff_aero(
    float D, float Da, float visc, float rhoa, float Temp, int species)
{
    return thompson_eff_aero(D, Da, visc, rhoa, Temp, species);
}


// ---------------------------------------------------------------------------
// Explicit two-gamma snow number, module_mp_thompson.F:2081-2088.
// ---------------------------------------------------------------------------
//
// This is NOT smo0 (the zeroth power-law moment).  It is the analytic
// integral of WRF's bimodal snow distribution and is the ns(k) the Koop
// homogeneous-freezing gate at :2634 tests.  smob is the bm_s-th moment
// (rs*oams, since bm_s == 2 exactly) and smoc the (bm_s+1)-th.
//
// Every operand is REAL(4) in WRF -- ns, M0, Mrat, slam1 and slam2 are all
// declared at :1609-1610 -- so this is float32 throughout, and Fortran's
// left-to-right evaluation of the equal-precedence `*` and `/` chain makes
// the second term ((((Mrat*Kap1)*M0**mu_s)*csg(15))/slam2**cse(15)).
//
// CONTRACTION-PINNED and correctly-rounded, for the same reason every other
// helper here is -- every operation below is __fdiv_rn / __fmul_rn /
// __fadd_rn and both powers are thompson_aa_powf_cr.  (If you are reading
// this because someone told you the body "uses plain *, /, + and CUDA's
// powf": it does not, and has not since wave 4.  Check the body, not the
// claim.)
//
// GATED AGAINST COMPILED WRF, NOT AGAINST A HOST TRANSCRIPTION.  The earlier
// version of this note quoted 391 states from a `gfortran -O2 -ffree-form`
// TRANSCRIPTION of :2029-2088.  Both halves of that are now superseded.  The
// reference is a PUBLIC probe subroutine whose body is a verbatim copy of
// :2028-2088, compiled into the real module_mp_thompson (so Kap0/Kap1,
// Lam0/Lam1, mu_s, csg(15) and cse(15) are thompson_init's own PRIVATE
// values, not retyped constants), and the sweep is 3721 states = 61
// temperatures from 273.05 K to 201.05 K x 61 log-spaced snow contents from
// 1e-12 to 1e-2 kg m^-3, with ns spanning 1.460e-01 to 2.797e+08 m^-3:
//
//     this form, fed WRF's own smoc:   3719/3721 BIT-EXACT
//         the two survivors are (268.25 K, rs = 6.8129234e-06) and
//         (265.84998 K, rs = 4.641592e-05), each exactly ONE float32 ulp,
//         max 6.923721e-08 relative.  Rebuilding the helper with plain CUDA
//         powf reproduces the SAME 3719/3721, so they are not a powf choice;
//         they are the double-rounding limit of `(float)pow(double,double)`
//         against glibc's singly-rounded powf.
//     the composite the callers run, a_ * smob**b_ then ns:
//         plain powf   smoc 3489/3721 (1.663435e-07)  ns 3526/3721 (5.512328e-07)
//         powf_cr      smoc 3717/3721 (1.063558e-07)  ns 3716/3721 (1.344195e-07)
//     thompson_field_a / thompson_field_b are BIT-EXACT on all 3721, so the
//     whole of that difference is the power.  cold.cu and warm.cu both spell
//     the composite with thompson_aa_powf_cr for exactly this reason.
//
// THE HELPER WAS NOT THE ICE-KOOP PROBLEM.  Feeding it WRF's own smoc it was
// already good to 1.2e-07; feeding it the smoc the OLD thompson_field_a /
// thompson_field_b produced it was wrong by up to 1.490356e-05, because the
// composite a_*smo2**b_ inherited the 3.3e-06 fit error and ns is quartic in
// smob/smoc.  The defect was upstream, in the fits, not here.
//
// Gates: tests/test_thompson_aerosol_cold_gpu.py::
// test_two_gamma_snow_number_matches_the_wrf_fortran_integral (92 Fortran
// states, EQUALITY) and ::test_two_gamma_snow_number_survivors_are_exactly_
// one_ulp (the two states above, pinned so the limit cannot silently grow).
__device__ __forceinline__ float thompson_aa_snow_number(
    float smob, float smoc)
{
    const float M0 = thompson_aa_div(smob, smoc);                 // :2083
    const float Mrat = thompson_aa_mul(                           // :2084
        thompson_aa_mul(thompson_aa_mul(smob, M0), M0), M0);
    const float slam1 = thompson_aa_mul(M0, THOMPSON_AA_LAM0);    // :2085
    const float slam2 = thompson_aa_mul(M0, THOMPSON_AA_LAM1);    // :2086
    // :2087-2088
    const float first = thompson_aa_div(
        thompson_aa_mul(Mrat, THOMPSON_AA_KAP0), slam1);
    const float second = thompson_aa_div(
        thompson_aa_mul(
            thompson_aa_mul(
                thompson_aa_mul(Mrat, THOMPSON_AA_KAP1),
                thompson_aa_powf_cr(M0, THOMPSON_AA_MU_S)),
            THOMPSON_AA_CSG15),
        thompson_aa_powf_cr(slam2, THOMPSON_AA_CSE15));
    return thompson_aa_add(first, second);
}


// ---------------------------------------------------------------------------
// Cloud droplet distribution, module_mp_thompson.F:1826-1842.
// ---------------------------------------------------------------------------
//
// Reproduces WRF's entry diagnosis exactly, including its type mixing: the
// first lambda is a REAL power (powf) widened to DOUBLE, the size clamps are
// REAL divisions widened to DOUBLE, and the rediagnosis is a DOUBLE pow.
// rc must already satisfy rc > R1; the caller owns that branch.
//
// CORRECTLY ROUNDED AND CONTRACTION-PINNED, and both halves were MEASURED.
// :1833's `**obmr` is REAL(4)**REAL(4) (nc, am_r, ccg, ocg1, rc and obmr are
// all default REAL), which gfortran lowers to glibc's correctly-rounded
// powf; the float32 products and quotients around it are separately rounded
// on a baseline-x86-64 build with no FMA instruction, and :1841 then CUBES
// the resulting lambda.  MEASURED against a PUBLIC probe subroutine holding
// a verbatim copy of :1826-1842, compiled into the same module_mp_thompson
// object the column oracle uses (so ccg/ocg1/cce/ocg2 are thompson_init's
// own values), over 975 states = 13 qc from 1e-9 to 1e-2 kg kg^-1 x 15 nc
// from 1e6 to 2e9 kg^-1 x 5 densities from 0.2 to 1.3, which spans the whole
// nu_c ladder and both :1835/:1837 size clamps:
//     plain powf, unpinned  (what shipped before)  nc  791/975 exact,
//                                                  max 3.632162e-07 relative
//                                                  lamc 929/975 exact,
//                                                  max 1.108940e-07
//     thompson_aa_powf_cr only                     nc  831/975 exact,
//                                                  max 1.893911e-07
//                                                  lamc 975/975 BIT-EXACT
//     powf_cr + every float32 chain pinned         nc  975/975 BIT-EXACT
//                                                  lamc 975/975 BIT-EXACT
// nu_c was already exact in all three.  Both changes are needed: the power
// alone fixes lamc but not the :1840 rediagnosis, whose REAL(4) prefactor
// ccg(1,nu_c)*ocg2(nu_c)*rc/am_r is rounded to float32 in Fortran before it
// meets the DOUBLE lamc**bm_r and was being widened here.
__device__ __forceinline__ float thompson_aa_cloud_dist(
    float rc, float nc_per_kg, float rho,
    int* nu_c_entry_out, double* lamc_entry_out)
{
    // :1830  nc(k) = MAX(2., MIN(nc1d(k)*rho(k), Nt_c_max))
    float nc = thompson_aa_clamp_nc(thompson_aa_mul(nc_per_kg, rho));
    const int nu_c = thompson_aa_nu_c(nc);
    // :1833  lamc = (nc(k)*am_r*ccg(2,nu_c)*ocg1(nu_c)/rc(k))**obmr
    double lamc = (double)thompson_aa_powf_cr(
        thompson_aa_div(
            thompson_aa_mul(
                thompson_aa_mul(thompson_aa_mul(nc, THOMPSON_AA_AM_R),
                                THOMPSON_AA_CCG2[nu_c]),
                THOMPSON_AA_OCG1[nu_c]),
            rc),
        THOMPSON_AA_OBMR);
    // :1834  xDc = (bm_r + nu_c + 1.) / lamc -- a REAL numerator (exact small
    // integers) over the DOUBLE lambda, the quotient narrowed back to REAL.
    const float xDc = (float)((double)thompson_aa_add(
        thompson_aa_add(THOMPSON_AA_BM_R, (float)nu_c), 1.0f) / lamc);
    // :1836 / :1838  a REAL(4) quotient ASSIGNED to the DOUBLE lamc.  D0c and
    // D0r*2. are not exact in binary, so the rounding is part of the answer.
    if (xDc < THOMPSON_AA_D0C) {
        lamc = (double)thompson_aa_div(THOMPSON_AA_CCE2[nu_c],
                                       THOMPSON_AA_D0C);
    } else if (xDc > THOMPSON_AA_D0R * 2.0f) {
        lamc = (double)thompson_aa_div(THOMPSON_AA_CCE2[nu_c],
                                       THOMPSON_AA_D0R * 2.0f);
    }
    // :1840-1841  MIN(DBLE(Nt_c_max), ccg(1,nu_c)*ocg2(nu_c)*rc(k)/am_r
    //                                 * lamc**bm_r)
    nc = (float)fmin(
        (double)THOMPSON_AA_NT_C_MAX,
        (double)thompson_aa_div(
            thompson_aa_mul(
                thompson_aa_mul(THOMPSON_AA_CCG1[nu_c],
                                THOMPSON_AA_OCG2[nu_c]),
                rc),
            THOMPSON_AA_AM_R)
            * pow(lamc, (double)THOMPSON_AA_BM_R));
    // ENTRY-STAGE values only (:1832-1838).  Everything downstream of :1838
    // recomputes nu_c from the RETURNED nc via thompson_aa_nu_c_working.
    if (nu_c_entry_out != nullptr) *nu_c_entry_out = nu_c;
    if (lamc_entry_out != nullptr) *lamc_entry_out = lamc;
    return nc;
}


// ---------------------------------------------------------------------------
// SHARED NETWORK HELPERS.  One definition each; cold.cu, warm.cu and sed.cu
// carry NONE.  See the PUBLISHED SHARED SIGNATURES block at the top of this
// header for the contract, INCLUDING its correction of what enforces it: a
// surviving local copy of thompson_aa_entry_rain_distribution is NOT a
// redefinition error, because the shared form takes seven parameters and the
// deleted local ones took six, which C++ resolves as an overload in favour of
// the local copy while nvrtc says nothing.  The source scan in
// tests/test_thompson_aerosol_device_helpers.py is the enforcement mechanism.
// ---------------------------------------------------------------------------

// module_mp_thompson.F:4032-4046 == thompson.cu:2574-2598.  Terminal rain
// number size bound.  This is the mp=8 arithmetic verbatim; it carries no nc
// dependence, so mp=8 and mp=28 agree here by construction.
//
// THOMPSON_AA_AM_R is bit-identical to the `3.1415926536f*1000.0f/6.0f`
// product form the deleted cold.cu copy spelled out, and THOMPSON_AA_R1 /
// THOMPSON_AA_R2 are the same float32 values as its 1.0e-12f / 1.0e-6f
// literals.  test_mass_coefficient_constants_equal_the_product_forms proves
// the first of those ON DEVICE so the substitution cannot rot.
__device__ __forceinline__ void thompson_aa_bound_rain_number(
    float rain_mass, float density, float* rain_number_per_kg)
{
    if (rain_mass <= THOMPSON_AA_R1) {
        *rain_number_per_kg = 0.0f;
        return;
    }
    const float am_r = THOMPSON_AA_AM_R;
    float rain_number = fmaxf(THOMPSON_AA_R2, *rain_number_per_kg * density);
    float lambda = powf(am_r * 6.0f * rain_number / rain_mass,
                        1.0f / 3.0f);
    float mvd = 3.672f / lambda;
    if (mvd > 2.5e-3f) {
        mvd = 2.5e-3f;
    } else if (mvd < 37.5e-6f) {
        mvd = 37.5e-6f;
    } else {
        return;
    }
    lambda = 3.672f / mvd;
    rain_number = (1.0f / 6.0f) * rain_mass / am_r
        * lambda * lambda * lambda;
    *rain_number_per_kg = rain_number / density;
}

// module_mp_thompson.F:4029-4039 == thompson.cu:3719-3743.  Terminal ice
// number size bound.  Idempotent, which is what lets the sedimentation kernel
// keep mp=8's fused placement while WP-04's terminal state kernel applies the
// same bound again.  THOMPSON_AA_AM_I is bit-identical to the
// `3.1415926536f*890.0f/6.0f` product form the deleted copies spelled out.
__device__ __forceinline__ void thompson_aa_bound_ice_number(
    float ice_mass, float density, float* ice_number_per_kg)
{
    if (ice_mass <= THOMPSON_AA_R1) {
        *ice_number_per_kg = 0.0f;
        return;
    }
    const float am_i = THOMPSON_AA_AM_I;
    float ice_number = fmaxf(THOMPSON_AA_R2, *ice_number_per_kg * density);
    double lambda = (double)powf(
        am_i * 6.0f * ice_number / ice_mass, 1.0f / 3.0f);
    const float diameter = (float)(4.0 / lambda);
    if (diameter < 5.0e-6f) {
        lambda = 4.0 / 5.0e-6;
        ice_number = fminf(999.0e3f,
            (1.0f / 6.0f) * ice_mass / am_i
            * (float)(lambda * lambda * lambda));
    } else if (diameter > 300.0e-6f) {
        lambda = 4.0 / 300.0e-6;
        ice_number = (1.0f / 6.0f) * ice_mass / am_i
            * (float)(lambda * lambda * lambda);
    }
    *ice_number_per_kg = fminf(ice_number, 999.0e3f) / density;
}

// module_mp_thompson.F:1878-1898 (the bounded LOCAL rain distribution WRF
// diagnoses at entry, deliberately distinct from the prognostic nr1d)
// followed by :2144-2150 (the y-intercept pass, which RE-DERIVES lamr from
// the bounded nr rather than reusing the clamped lambda).  thompson.cu:
// 2607-2645 skips that round trip; mp=28 reproduces it because prr_rcw,
// pnc_rcw, pna_rca and pnd_rcd all read N0_r and (lamr+fv_r)^-cre(9)
// directly.
//
// WRF's :2145 loop has NO L_qr guard, so :2146-2150 runs at every level.
// When rain_per_kg <= R1 the entry block leaves rr = R1 and nr = R2 (:1892-
// 1897) and the y-intercept pass still forms lamr, mvd_r and N0_r from those
// sentinels; this function therefore writes all four outputs unconditionally
// and returns L_qr as its value.
//
// crg(3) = WGAMMA(bm_r+mu_r+1) = WGAMMA(4) = 6 exactly, org2 = 1/WGAMMA(1) =
// 1 exactly, cre(2) = mu_r+1 = 1 exactly, so N0_r = nr*lamr.  crg(2)*org3 =
// WGAMMA(1)/WGAMMA(4) = 1/6.
//
// CONTRACTION-PINNED, and every power is thompson_aa_powf_cr: build_aero.sh
// compiles the oracle with plain `gfortran -O2` on baseline x86-64, which has
// no FMA instruction and lowers REAL(4)** to glibc's correctly-rounded powf,
// while nvrtc defaults to --fmad=true and CUDA's powf carries ~2 ulp.
// MEASURED (warm.cu:81-92, RTX 5090, 12348 oracle rows): with plain powf,
// lamr / N0_r / prr_rcw / pna_rca / pnd_rcd all sit at ~2.7e-7 relative;
// pinned they are BIT-EXACT.  Do not "simplify" this back to powf.
__device__ __forceinline__ bool thompson_aa_entry_rain_distribution(
    float rain_per_kg, float rain_number_per_kg, float density,
    float* rain_number, double* rain_lambda, float* rain_mvd,
    double* rain_intercept_n0)
{
    const float am_r = THOMPSON_AA_AM_R;
    const float obmr = THOMPSON_AA_OBMR;
    // crg(2)*org3 = WGAMMA(mu_r+1)/WGAMMA(bm_r+mu_r+1) = 1/6 exactly.
    const float crg2_org3 = 1.0f / 6.0f;
    float rr;
    float nr;
    bool active;
    if (rain_per_kg > THOMPSON_AA_R1) {
        active = true;
        rr = thompson_aa_mul(rain_per_kg, density);
        nr = fmaxf(THOMPSON_AA_R2,
                   thompson_aa_mul(rain_number_per_kg, density));
        if (nr <= THOMPSON_AA_R2) {
            const double lam = (double)thompson_aa_div(3.672f, 1.0e-3f);
            nr = (float)((double)thompson_aa_div(
                             thompson_aa_mul(crg2_org3, rr), am_r)
                         * lam * lam * lam);
        }
        double lamr = (double)thompson_aa_powf_cr(
            thompson_aa_div(
                thompson_aa_mul(thompson_aa_mul(am_r, 6.0f), nr), rr),
            obmr);
        float mvd = (float)(3.672 / lamr);
        if (mvd > 2.5e-3f) {
            mvd = 2.5e-3f;
            lamr = (double)thompson_aa_div(3.672f, mvd);
            nr = (float)((double)thompson_aa_div(
                             thompson_aa_mul(crg2_org3, rr), am_r)
                         * lamr * lamr * lamr);
        } else if (mvd < THOMPSON_AA_D0R * 0.75f) {
            mvd = THOMPSON_AA_D0R * 0.75f;
            lamr = (double)thompson_aa_div(3.672f, mvd);
            nr = (float)((double)thompson_aa_div(
                             thompson_aa_mul(crg2_org3, rr), am_r)
                         * lamr * lamr * lamr);
        }
    } else {
        active = false;
        rr = THOMPSON_AA_R1;
        nr = THOMPSON_AA_R2;
    }

    // :2146-2150, executed for every level in WRF, rain or not.
    const double lamr = (double)thompson_aa_powf_cr(
        thompson_aa_div(
            thompson_aa_mul(thompson_aa_mul(am_r, 6.0f), nr), rr),
        obmr);
    *rain_number = nr;
    *rain_lambda = lamr;
    *rain_mvd = (float)(3.672 / lamr);
    *rain_intercept_n0 = (double)nr * lamr;
    return active;
}


// ---------------------------------------------------------------------------
// calc_effectRad, module_mp_thompson.F:5594-5699.
// ---------------------------------------------------------------------------

// :5637-5643.  NOT the same selector as thompson_aa_nu_c.  The nc > 1.0e10
// branch is DEAD CODE in v4.6.1 because :5626 clamps nc to Nt_c_max first;
// it is transcribed anyway so a future caller cannot reintroduce it wrongly.
__device__ __forceinline__ int thompson_aa_inu_c_effrad(float nc_m3)
{
    if (nc_m3 < 100.0f) return 15;
    if (nc_m3 > 1.0e10f) return 2;
    return min(15, (int)floorf(1000.0e6f / nc_m3 + 0.5f) + 2);
}

// EVERY `**` IN calc_effectRad IS A REAL(4)**REAL(4), which gfortran lowers
// to glibc's correctly-rounded powf, where CUDA's powf carries several ulp.
// MEASURED against the REAL calc_effectRad -- it is PUBLIC, so it can be
// CALLED rather than transcribed -- over grids far wider than the committed
// probe-effectrad.csv, whose 14 rows all share one t, one qc and one qs and
// therefore pin only the nc ladder:
//
//     effs_m, 360 rows (24 T x 15 qs), 299 strictly inside the clamp
//         old fits + plain powf: 138/360 exact, max 5.870704e-06 relative
//         new fits + plain powf: 342/360 exact, max 1.947693e-07 relative
//         new fits + powf_cr:    360/360 BIT-EXACT      <- SHIPPED
//     effc_m, 378 rows (6 T x 7 qc x 9 nc), 198 strictly inside the clamp
//         plain powf: 373/378 exact, max 1.042121e-07
//         powf_cr:    378/378 BIT-EXACT                 <- SHIPPED
//     effi_m, 378 rows (6 T x 9 qi x 7 ni), 228 strictly inside the clamp
//         plain powf: 374/378 exact, max 6.293743e-08
//         powf_cr:    378/378 BIT-EXACT                 <- SHIPPED
//
// ALL THREE ARE WRF-EXACT, AND THE ICE BRANCH DELIBERATELY DIVERGES FROM
// mp=8 BY ONE ULP.  RECORD, DO NOT REVERT:
//
//   On the aero-reduces-to-classic after-column, thompson.cu and this header
//   disagree on effi at k = 23 and k = 24 by 6.567156e-08 relative -- one
//   float32 ulp.  MEASURED at those two levels, in metres:
//       WRF fixture effi_m   2.9473000e-05      2.9043753e-05
//       mp=28 powf_cr        2.9473000e-05      2.9043753e-05   <- matches
//       mp=8 thompson.cu     2.9473001e-05      2.9043755e-05   <- 1 ulp off
//   i.e. mp=8 is the one that disagrees with WRF there.  Reproducing that
//   disagreement was the only thing plain powf bought, and
//   tests/test_thompson_aerosol_state_gpu.py::
//   test_effective_radius_is_bitwise_against_every_oracle_after_column
//   demands the opposite.
//
//   WP-04's mp=8 identity test,
//   test_effective_radius_reduces_to_the_frozen_mp8_kernel_at_nt_c, has been
//   restated by its owner to EXPECT this: it still requires effc and effs
//   bitwise against thompson.cu, requires the effi divergence to be exactly
//   levels [22, 23] and at most one ulp, and requires mp=28 -- not mp=8 -- to
//   match the Fortran column.  Both tests are green.  If you put CUDA's powf
//   back you will break it from the other side.
//
//   MP28_PORT_SPEC.md's named hard identity gates are
//   thompson_aa_droplet_bin(100.0e6f) == 65 and
//   thompson_aa_in_bin(1000.0f) == 27.  Both are green and both are asserted
//   in tests/test_thompson_aerosol_device_helpers.py.
//
//   This is the same tie-break as "THE SHARED FITS" above thompson_aa_add and
//   as MP28_PORT_SPEC.md's gamma-deviation finding: mp=8 stays frozen and
//   slightly wrong, mp=28 is right.
// tests/test_thompson_aerosol_device_helpers.py::
// test_effect_rad_cloud_and_ice_match_the_real_calc_effect_rad and
// test_effect_rad_cloud_and_ice_diverge_from_mp8_by_one_ulp_on_purpose pin
// both halves of that so neither can rot.

// rc [kg m^-3] = MAX(R1, qc*rho); nc_m3 = MAX(2, MIN(nc*rho, Nt_c_max)).
// Caller must have already applied those, and must skip levels with
// rc <= R1 or nc <= R2.  Returns metres.  :5644-5646.
//
// ---------------------------------------------------------------------------
// CONTRACTION-PINNED AS WELL AS CORRECTLY ROUNDED, AND THE PINS MOVE NOTHING
// TODAY.  SAID PLAINLY BECAUSE IT IS THE POINT.
// ---------------------------------------------------------------------------
// Every operand at :5646, :5654, :5663 and :5694 is REAL(4), so each float32
// product and quotient is separately rounded in the gfortran -O2 baseline-
// x86-64 oracle.  Here each of those chains ends in `(double)<float expr>`,
// and nvrtc is free to evaluate such a chain in DOUBLE and skip the float32
// roundings -- WP-04 MEASURED exactly that happening in
// thompson_aa_state_finalize's :4019 prefactor, where it cost 2 to 3.5
// float32 ulps of droplet number and took 38 of 456 states off a Fortran-
// faithful host transcription.  A named `float` local does not stop it; only
// __fmul_rn / __fdiv_rn do.
//
// MEASURED HERE, against the REAL calc_effectRad (it is PUBLIC, so it is
// CALLED, not transcribed) over 960 states = 40 columns x 24 levels sweeping
// temperature 233-260 K, qc 1e-7..1e-4, nc 3e6..3e9 kg^-1, qi/ni and qs each
// over four decades:
//     unpinned float chains (what shipped)  effc 960/960, effi 958/960
//                                           (max 1.094592e-07), effs 960/960
//     every float32 chain pinned            effc 960/960, effi 958/960
//                                           (max 1.094592e-07), effs 960/960
// i.e. IDENTICAL.  The pins are not a correction; they turn "nvrtc happens
// not to widen this today" into a property of the source, which is what lets
// thompson_aerosol_state.cu delete its three private copies of these
// functions and call these instead.  The two surviving effi states are the
// double-rounding limit of thompson_aa_powf_cr -- `(float)pow(double,double)`
// rounds twice where glibc's powf rounds once -- not a chain-association
// defect; plain CUDA powf does not fix them either.

__device__ __forceinline__ float thompson_aa_eff_rad_cloud(
    float rc, float nc_m3)
{
    const int inu_c = thompson_aa_inu_c_effrad(nc_m3);
    // :5646  lamc = (nc(k)*am_r*g_ratio(inu_c)/rc(k))**obmr.  CORRECTLY
    // ROUNDED: WRF's `**` is REAL(4)**REAL(4) -> glibc powf.  See the block
    // above for what this costs and why it is right.
    const double lamc = (double)thompson_aa_powf_cr(
        thompson_aa_div(
            thompson_aa_mul(thompson_aa_mul(nc_m3, THOMPSON_AA_AM_R),
                            THOMPSON_AA_G_RATIO[inu_c]),
            rc),
        THOMPSON_AA_OBMR);
    // :5647  SNGL(0.5D0 * DBLE(3.+inu_c)/lamc).  `3.+inu_c` is a REAL plus an
    // INTEGER, i.e. a REAL add of exact small integers, widened only after.
    const float diagnosed = (float)(
        0.5 * (double)thompson_aa_add(3.0f, (float)inu_c) / lamc);
    return fmaxf(2.51e-6f, fminf(diagnosed, 50.0e-6f));
}

// ri [kg m^-3] = MAX(R1, qi*rho); ni [m^-3] = MAX(R2, ni*rho).  :5654-5655.
__device__ __forceinline__ float thompson_aa_eff_rad_ice(float ri, float ni)
{
    // :5654  lami = (am_i*cig(2)*oig1*ni(k)/ri(k))**obmi.  CORRECTLY ROUNDED.
    // THIS is the one that costs the mp=8 effective-radius identity test, at
    // aero-reduces-to-classic's top levels, by one float32 ulp -- because
    // mp=8 is 1 ulp off WRF there and mp=28 is not.  See the block above.
    const double lami = (double)thompson_aa_powf_cr(
        thompson_aa_div(
            thompson_aa_mul(
                thompson_aa_mul(
                    thompson_aa_mul(THOMPSON_AA_AM_I, THOMPSON_AA_CIG2),
                    THOMPSON_AA_OIG1),
                ni),
            ri),
        THOMPSON_AA_OBMI);
    // :5655  SNGL(0.5D0 * DBLE(3.+mu_i)/lami); mu_i is REAL (:105).
    const float diagnosed = (float)(
        0.5 * (double)thompson_aa_add(3.0f, THOMPSON_AA_MU_I) / lami);
    return fmaxf(2.51e-6f, fminf(diagnosed, 125.0e-6f));
}

// rs [kg m^-3] = MAX(R1, qs*rho).  bm_s is exactly 2, so WRF's reference
// second moment IS smob and the bm_s.ne.2 branch (:5665-5679) is dead.
// :5661-5695.
__device__ __forceinline__ float thompson_aa_eff_rad_snow(float rs, float t_k)
{
    const float tc0 = fminf(-0.1f, thompson_aa_sub(t_k, 273.15f));   // :5662
    const float smob = thompson_aa_mul(rs, THOMPSON_AA_OAMS);        // :5663
    const float smo2 = smob;                                         // :5668
    const float moment = 3.0f;                // cse(1) = bm_s + 1
    // :5694  smoc = a_ * smo2**b_
    const float smoc = thompson_aa_mul(
        thompson_field_a(tc0, moment),
        thompson_aa_powf_cr(smo2, thompson_field_b(tc0, moment)));
    // :5695  0.5*(smoc/smob)
    const float diagnosed = thompson_aa_mul(
        0.5f, thompson_aa_div(smoc, smob));
    return fmaxf(5.01e-6f, fminf(diagnosed, 999.0e-6f));
}
