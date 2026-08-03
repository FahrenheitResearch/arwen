// gpuwm/core/kernels/thompson_aerosol_cold.cu
//
// WP-06 -- aerosol-aware (mp_physics=28) COLD network for WRF v4.6.1
// Thompson.  One kernel, thompson_aa_cold_network, reproducing the whole
// sub-freezing source group of module_mp_thompson.F:2239-3190 with
// is_aerosol_aware = .TRUE., dustyIce = .TRUE., homogIce = .TRUE.,
// is_hail_aware = .FALSE., wif_input_opt = 0.
//
// Numerical authority: /home/drew/wrf461-pristine/phys/module_mp_thompson.F,
// commit d66e442fccc04111067e29274c9f9eaccc3cef28.  Every bare line number
// below refers to that file.  Structural authority for everything that is
// NOT aerosol-dependent: gpuwm/core/kernels/thompson.cu's
// thompson_frozen_vapor_cloud_network (:6514-7586), which is byte-frozen and
// is transcribed -- not shared -- because cupy.RawModule has no #include
// path.  Shared device helpers come from thompson_aerosol_common.cuh, which
// gpuwm/core/kernels/__init__.py prepends to this translation unit.
//
// ---------------------------------------------------------------------------
// WHAT CHANGES RELATIVE TO mp=8, AND WHY EACH ONE MATTERS
// ---------------------------------------------------------------------------
// 1. nc is PROGNOSTIC.  thompson.cu freezes the droplet number at
//    Nt_c = 100e6 m^-3 (`const float cloud_number = 100.0e6f`, :6928).  Here
//    it is rediagnosed per cell from the entry nc array through
//    thompson_aa_cloud_dist (:1826-1842), and every consumer -- the Bigg
//    freezing number cap, the droplet-bin table index, the mean volume
//    diameter, and the five cloud-number sinks -- reads that value.
//
// 1a. THE nu_c STAGING.  WRF computes nu_c at :1832 from the PRE-rediagnosis
//    nc and AGAIN at :2170 from the POST-rediagnosis nc(k) of :1840.  Only
//    the second is the working shape parameter; it feeds lamc (:2173),
//    mvd_c (:2174), Dc_g (:2181) and pnr_wau (:2191).  This kernel used to
//    take nu_c from thompson_aa_cloud_dist's ENTRY out-parameter, which is
//    the first one.  Whenever the :1834-1838 droplet-size clamp engages the
//    two differ -- measured 689 of 5400 states on a (nc, rc) grid at rho = 1
//    -- and between nu_c = 3 and nu_c = 15 the gamma products the rates are
//    built from move by 40.8x (lamc) and 15.8x (Dc_g) while pnr_wau, being
//    linear in nu_c, rescales by 5.  See thompson_aerosol_common.cuh's
//    "nu_c STAGING RULE" block.
//
//    OBSERVABILITY, MEASURED, because it explains why the committed fixtures
//    could not have caught this: in EVERY divergent state mvd_c lands on a
//    clamp bound.  That is structural, not lucky -- if the entry clamp
//    engaged then lamc = cce(2,nu)/D0c or cce(2,nu)/(2*D0r), so
//    mvd_c = (3.672+nu)/(4+nu) times D0c or 2*D0r, which is below D0c in the
//    first case and above D0r in the second for every nu in [2,15].  The
//    only observables of the staging are therefore Dc_g and pnr_wau (and
//    pnc_wau through prr_wau), all gated on rc > 0.01e-3, and the lowest
//    divergent rc that clears that gate is 3.4365e-2 kg m^-3 (re-measured in
//    wave 4 over a 60 x 80 (nc, qc) sweep: 1259 of 4800 states diverge and
//    EVERY one of them has prr_wau > 0).  No committed fixture goes there.
//
//    WAVE 4 CLOSES THE REGRESSION GAP THAT LEFT.  Wave 3's gate was a PROBE
//    comparison, and the probe/production agreement test compared ncten --
//    which on every divergent state is pinned to the nc*odts cap at :2192
//    and therefore cannot tell the two stages apart.  So nothing failed if
//    the PRODUCTION kernel regressed.  It does now:
//    test_production_kernel_uses_the_working_stage_nu_c drives THIS kernel
//    into that regime with no rain, ice, snow or graupel and compares its
//    own qr/nr/ncten against WRF's :2168-2193.  nr is the discriminator,
//    because prr_wau is pinned to rc*odts by the MIN at :2190 across the
//    whole regime while pnr_wau = prr_wau/(am_r*nu_c*10*D0r**3) stays
//    LINEAR in nu_c.  MEASURED with the working stage reverted to the entry
//    stage: qr and ncten do not move at all, and nr is wrong by 25.0% to
//    80.0% on all 36 divergent rows.
//
// 2. idx_IN IS LIVE.  thompson.cu hardcodes `const int nuclei_bin = 27` at
//    :6820 (rain freezing) and :7008 (cloud freezing), which is WRF's
//    non-aerosol default of 1 ice nucleus per litre.  Under mp=28 the index
//    is thompson_aa_in_bin(iceDeMott(...)) (:2579-2591).  This is the FIRST
//    time gpuwm reads any freezeH2O slice other than 27; 54 of the 55 IN
//    slices have never been exercised by any previous test.
//
// 3. idx_n IS LIVE.  thompson.cu hardcodes `const int cloud_number_bin = 65`
//    at :7005.  Here it is thompson_aa_droplet_bin(nc_work) (:3447-3448).
//    thompson_aa_droplet_bin(100.0e6f) == 65 is a hard identity gate in
//    tests/test_thompson_aerosol_device_helpers.py, so at nc = Nt_c this
//    reduces to what mp=8 froze.
//
// 4. iceDeMott REPLACES Cooper, it does not supplement it.  WRF:2622-2626
//    is an if/else on `dustyIce .AND. is_aerosol_aware`; the Cooper
//    expression MIN(250.E3, TNO*EXP(ATO*(T_0-temp))) that thompson.cu
//    evaluates at :6645-6646 and :7189-7200 is unreachable in mp=28 and
//    appears NOWHERE in this file.  Both of WRF's iceDeMott call sites
//    (:2574 and :2623) differ only in dead formal arguments and return the
//    identical value, so it is evaluated ONCE per cell here.
//
// 5. KOOP HOMOGENEOUS HAZE FREEZING (:2633-2643) is entirely new; classic
//    Thompson has no counterpart.  Its gate needs ns(k), WRF's EXPLICIT
//    two-gamma snow-number integral at :2081-2088 -- NOT smo0, and not
//    gpuwm's mp=8 snow number closure.  ns(k)'s ONLY use in the entire
//    routine is this gate.  pri_iha joins the shared frozen-vapor limiter
//    sum (:2862-2876); pni_iha is deliberately NOT rescaled with it, the
//    same held-number quirk WRF already applies to pni_inu.
//
// 6. FIVE CLOUD-NUMBER SINKS feed ncten: pni_wfz (Bigg immersion freezing),
//    pnc_scw (snow collecting cloud water, :2410-2411), pnc_gcw (graupel
//    collecting cloud water, :2435-2437), pnc_wau (autoconversion,
//    :2192-2193) and pnc_rcw (rain collecting cloud water, :2205-2207).
//    Each collection term is MIN'd against nc*odts individually.
//
// 7. SIX AEROSOL SCAVENGING RATES feed nwfaten/nifaten: pna_sca and pnd_scd
//    (snow, :2444-2450), pna_gca and pnd_gcd (graupel, :2461-2471) and
//    pna_rca and pnd_rcd (rain, :2212-2221), with Eff_aero collection
//    diameters 0.04 um for water-friendly aerosol and 0.8 um for
//    ice-friendly aerosol.  WRF caps each against the FULL available aerosol
//    and deliberately lets the sum overshoot, relying on the terminal floors
//    at :3979-3981; do NOT add a shared limiter across collectors.
//
// 8. TWO AEROSOL SINKS FROM NUCLEATION: nifaten -= pni_inu*orho
//    (:2974-2976, because dustyIce is a PARAMETER .true.) and
//    nwfaten -= pni_iha*orho (:2964-2965 -- Koop consumes WATER-friendly
//    aerosol, not ice nuclei).  CRITICAL NEGATIVE FINDING: Bigg immersion
//    freezing of cloud droplets and raindrops consumes NO aerosol at all.
//    nifa influences it only by shifting the freezing table's temperature
//    index through T_adjust = MAX(-3, MIN(3-LOG10(Nt_IN(m)), 3)) at :4701,
//    i.e. through idx_IN.  Adding a nifa sink for pni_wfz or pni_rfz
//    because it "seems physical" is not in WRF and would deplete ice nuclei
//    roughly an order of magnitude too fast in any glaciating cloud.
//
// ---------------------------------------------------------------------------
// ACCUMULATOR CONTRACT
// ---------------------------------------------------------------------------
// nc/nwfa/nifa state are READ-ONLY entry state for the whole mp=28 call.
// This kernel writes ONLY into the three per-kilogram scratch accumulators
// ncten / nwfaten / nifaten (which the adapter zeroes at entry) with `+=`,
// and a terminal WP-04 kernel applies them once with WRF's clamps
// (:3972-4021).
//
// WHY THE ACCUMULATORS ARE NOT READ BACK HERE.  At the point in WRF where
// this network runs, nwfaten/nifaten/ncten are still identically zero: they
// are first assigned in the tendency loop at :2964-3009, which is AFTER
// every process rate in :2239-2850 has been diagnosed.  The working
// per-m^3 values WRF uses inside the cold network are therefore the ENTRY
// clamps at :1805-1806 alone, and nc(k) is the entry rediagnosis at
// :1826-1842 -- not (nc1d + ncten*dt).  Reading the accumulators here would
// make the answer depend on whether the adapter happens to launch the warm
// network first, which WRF's single-pass structure never does.
//
// ---------------------------------------------------------------------------
// THE TEMPERATURE SPLIT, AND WHY IT IS EXHAUSTIVE
// ---------------------------------------------------------------------------
// iiwarm is a PARAMETER .false. (:59) and mu_g/idx_bg are fixed, so WRF runs
// ONE column loop with two temperature branches inside it.  Only the block
// opened at :2554 is guarded by `if (temp(k).lt.T_0)`.  Everything at
// :2156-2234 (the warm-rain loop) and everything at :2402-2478 (snow/graupel
// collection of cloud water and of aerosol) runs at EVERY level.
//
// gpuwm splits that one loop across two kernels by ENTRY temperature:
//   * this kernel returns at once for temperature >= 273.15 f, and it runs
//     BEFORE any kernel has moved the temperature field, so its gate sees
//     the entry column;
//   * thompson_aerosol_warm.cu returns unless its held entry-temperature
//     mask (cp.greater_equal(temperature, 273.15), seeded before the first
//     source launch) is set.
// The two masks are exact complements of one another over every cell, with
// 273.15 K itself owned by the warm side.  Each of the ten always-run rates
// therefore appears in BOTH translation units and is evaluated EXACTLY ONCE
// per cell.  A gap or an overlap at 273.15 K is the obvious way this goes
// wrong; tests/test_thompson_aerosol_cold_gpu.py pins it at 273.15 and at
// both bracketing floats.
//
// HISTORICAL NOTE, kept because the reasoning was wrong in an instructive
// way.  This file originally omitted pnc_wau, pnc_rcw, pna_rca and pnd_rcd
// on the grounds that they were "WP-07's by the port's exclusive
// file-ownership rule".  File ownership is about who may EDIT a file, not
// about which cells a kernel is responsible for.  WP-07's kernel does own
// those four rates -- for warm cells.  Omitting them here did not delegate
// them, it deleted them for the entire sub-freezing half of the domain,
// while this kernel went on computing their MASS partners prr_wau and
// prr_rcw.  Cloud mass left without its droplet number and rain wet
// scavenging of CCN/IN was absent below freezing.
//
// EVERY level of all six committed aerosol fixtures is sub-freezing (230,
// 240 or 260 K), so WP-07's kernel returns immediately on all of them: the
// four rates reached nobody at all, and the wave-2 reference table had been
// assembled with them subtracted out, so the suite agreed with the defect.
//
// ---------------------------------------------------------------------------
// MEASURED AGREEMENT WITH WRF v4.6.1 (RTX 5090, CuPy 14.1.1, nvrtc c++17)
// ---------------------------------------------------------------------------
// A. Six committed column fixtures, gpuwm/data/thompson/oracle-aero/, max
//    relative difference over levels carrying signal:
//      aero-scav-frozen      nwfa, nifa                        BIT-EXACT
//      aero-ice-demott-dep   nifa                              BIT-EXACT
//      aero-ice-koop         nifa BIT-EXACT, nwfa              1.78e-07
//      aero-ice-demott-idxin qi 1.20e-07 ni 6.60e-08 ncten 3.96e-08
//                            nwfaten 4.22e-08 nifaten 5.24e-08
//      aero-cloud-freeze-nc  qi 2.01e-08 ni 4.03e-08 ncten 4.33e-08
//      aero-cold-overlap     qi 9.95e-08 ni 1.52e-07 ncten 3.43e-07
//                            nwfaten 2.55e-07 nifaten 5.16e-07
//    aero-ice-demott-idxin is the acceptance fixture: it is the only fixture
//    in the port that pins an idx_IN other than 27, i.e. an entire freezeH2O
//    axis gpuwm had never read.
//
// B. 11340 sub-freezing states from tools/thompson_wrf461_oracle/
//    probe_cold_warm_loop_aero.F90 (COMMITTED; build_aero_probes.sh rebuilds
//    it and it reproduces its pinned SHA-256), which links the same compiled
//    module_mp_thompson.o and evaluates :1826-1842 and :2144-2232 verbatim:
//      nu_c (both stages), nc_m3, mvd_c, mvd_r                 EXACT
//      prr_wau, pnr_wau, pnc_wau                               EXACT
//      pnc_rcw, pna_rca, pnd_rcd                   <= 7.6e-16 (CSV round trip)
//    THE LAST FIVE USED TO BE THE UPSTREAM RESIDUAL AND ARE NOW GONE.  This
//    block used to read
//      mvd_c 1.23e-07, nc_m3 3.20e-07, pnc_rcw 3.25e-07
//      pnc_wau 1.97e-06, prr_wau / pnr_wau 2.31e-06
//    and attributed the last line, correctly, to thompson_aa_cloud_dist's
//    CUDA powf in thompson_aerosol_common.cuh, amplified by the Dc_b
//    cancellation at :2182 on the rc = 0.15 kg m^-3 end of the ladder.
//    Contraction-pinning everything this file owns had already taken prr_wau
//    from 1.03e-05 to 2.31e-06; making the shared droplet diagnosis
//    correctly-rounded AND contraction-pinned took the remaining 2.31e-06 to
//    zero.  Nothing in this file changed for it.
//
// C. WAVE 4 -- AGAINST WRF'S OWN POST-SOURCE TENDENCIES, not against a
//    reduction of them.  A scratch build under /home/drew/mp28-oracle-work/
//    wp06/ compiles a COPY of module_mp_thompson.F carrying only added
//    `write` statements (plus three added PUBLIC probe subroutines whose
//    bodies are verbatim transcriptions of statements already in the file),
//    links the same stub_wrf.o / module_mp_radar.o / run_column_aero.o with
//    the same flags and the same four assets, and REPRODUCES ALL 38
//    COMMITTED aerosol fixture CSVs BYTE FOR BYTE.  Its instrumentation
//    publishes qiten/niten/qrten/nrten as WRF has them immediately after the
//    :3021-3092 tendency loop -- i.e. exactly what this kernel is
//    responsible for, before condensation and before any fallout.  Driving
//    this kernel on each fixture's OWN entry column, max relative
//    difference over the levels carrying signal:
//      aero-ice-koop          qi 1.152e-07   ni 8.482e-08
//      aero-cold-overlap      qi 9.950e-08   ni 1.515e-07   qr 8.795e-08
//      aero-ice-demott-idxin  qi 1.203e-07   ni 6.599e-08   qr 1.304e-08
//      aero-ice-demott-dep    qi 1.209e-07   ni 5.907e-08
//      aero-cloud-freeze-nc   qi 2.013e-08   ni 4.026e-08   qr 4.290e-08
//    (nr agrees EXACTLY wherever it moves at all; the apparent 4e-4 in a
//    finite-difference reconstruction of nrten is the reconstruction's own
//    cancellation, not a state difference.)
//
//    WHAT THAT SETTLES.  Two adapter-level G3 fixtures were open against
//    this package.  Neither is this kernel:
//
//    * aero-ice-koop, qi 1.612e-03 / ni 1.764e-03.  The whole residual is
//      ONE 2^-24 quantum of iceKoop's prob_h.  :5539 forms
//      prob_h = 1. - exp(-J*V*DT) with the exponential just below 1, so the
//      subtraction is exact and prob_h can only be an integer multiple of
//      2^-24; at this fixture's level 15 the integer is 561, so one step of
//      that grid is 1/561 = 1.78e-3 of the answer.  log_J_rate is a cubic in
//      delta_aw whose four terms (-906.7, +2.63e3, -2.59e3, +8.7e2) sum to
//      about 10.7, giving d(log10 J)/d(satw) ~ 222, so ONE float32 ulp of
//      pres (9.7e-8 relative) is enough to cross a quantum boundary.
//      tests/test_thompson_aerosol_adapter.py's entry-state reconstruction
//      perturbs p by exactly +1 ulp at that level in order to find a float32
//      theta reproducing temp_k exactly, and that costs 595.97 of 334348.6
//      crystals m^-3 = 1.7825e-3.  MEASURED, and NOT repairable inside the
//      harness: over +-256 ulps of p there is NO pressure at that level that
//      both admits an exact theta and leaves RSLF/RSIF bit-identical, and
//      driving the adapter with the exact p (accepting a 1-ulp temperature
//      instead) makes it WORSE, 3.08e-03.  Gated by
//      test_ice_koop_is_quantised_and_one_pressure_ulp_moves_it_one_quantum.
//
//    * aero-cold-overlap, the terminal droplet number.  Cloud water is
//      consumed to WRF's own R1 floor at these levels (which this port
//      reproduces BIT-EXACTLY), so :4019's rediagnosis collapses onto the
//      MAX(2./rho(k), ...) floor at :3976 and the whole residual is the
//      choice of rho.  WRF's rho(k) in that loop is the TAU+1 density -- the
//      unconditional :3193 refresh, then per level :3490 and :3572 -- and
//      thompson_aa_state_finalize is handed the ENTRY density of :1802.
//      RE-MEASURED on the regenerated fixtures with an instrumented pristine
//      WRF that reproduces all 22 committed column CSVs byte for byte: the
//      two densities differ by up to 7.4672e-03 (this fixture), and
//      recomputing :3976-4021 INSIDE WRF with the entry density substituted
//      moves nc1d at exactly one of 504 fixture levels -- aero-cold-overlap
//      k=5, 1.833336115 -> 1.826136470 kg^-1, 3.9271e-03 relative.
//      (It used to be quoted here as level 7 / 5.73e-05; that level's qc no
//      longer lands on the floor after the harness repaired its own pii.)
//      That is WP-04's kernel and WP-09's call site, not this one; the fix
//      is one added launch of thompson_aa_tau1_density immediately after
//      rain evaporation, which reproduces WRF's terminal rho BITWISE at 501
//      of those 504 levels.  Reported as an integration request.
//
// D. WAVE 4 -- ns(k), the least Fortran-verified helper in the shared
//    header, now gated against compiled WRF.  thompson_aa_snow_number is
//    BIT-EXACT against module_mp_thompson.F:2081-2088 on 3719 of 3721
//    states of a 61-temperature (273.05 to 201.05 K) x 61-snow-content
//    (1e-12 to 1e-2 kg m^-3) sweep run through the module's own
//    csg(15)/cse(15), the other 2 being ONE float32 ulp (6.924e-08) and
//    reproducing identically when the helper is rebuilt fully
//    contraction-pinned with correctly-rounded powers, i.e. not repairable.
//    The COMPOSITE path this kernel actually uses
//    (rho -> smob -> thompson_field_a/b -> smoc -> ns) is BIT-EXACT against
//    WRF's own ns(k) at all 16 snow-bearing levels of the committed
//    fixtures.  ns is NOT interchangeable with smo0: over that sweep
//    ns/smo0 runs from 0.83 to 1226, and
//    test_two_gamma_snow_number_and_not_smo0_decides_the_koop_gate makes the
//    difference PRODUCTION-observable by putting two cells either side of
//    the 999e3 gate where the two closures disagree.


// ---------------------------------------------------------------------------
// Helpers this translation unit consumes from thompson_aerosol_common.cuh.
// thompson.cu is byte-frozen and there is no #include path under
// cupy.RawModule, so anything shared with the other four aerosol kernels
// lives ONCE in the header the loader prepends, never as a local duplicate.
// ---------------------------------------------------------------------------

// thompson_aa_decade_index_double (thompson.cu:3084-3105, the DOUBLE-precision
// base-ten mantissa/decade bin used by the rain and graupel y-intercept
// lookups) USED TO BE DUPLICATED HERE, byte-identically with warm.cu's copy.
// It now lives in thompson_aerosol_common.cuh; see that file's PUBLISHED
// SHARED SIGNATURES block.  Do not reintroduce a local copy -- the shared
// signature is identical, so nvrtc rejects the survivor outright with
// "function has already been defined", which is exactly the diagnostic this
// consolidation exists to guarantee.
//
// thompson_aa_bound_rain_number, thompson_aa_bound_ice_number and
// thompson_aa_entry_rain_distribution USED TO BE DUPLICATED HERE with bodies
// that had already drifted from warm.cu's copies (no rain_intercept_n0, no
// contraction pinning, no unconditional :2146-2150 lamr re-derivation).
// Separate cupy.RawModule translation units mean nvrtc never sees such a
// conflict, so the drift is silent.  All three now live ONCE in
// thompson_aerosol_common.cuh; see its PUBLISHED SHARED SIGNATURES block.
// Do not reintroduce a local copy -- for the two `void` helpers that is a
// hard nvrtc redefinition error, but for the rain distribution the shared
// form takes SEVEN parameters and a six-parameter local copy would quietly
// OVERLOAD rather than collide.  That is exactly the failure mode this
// consolidation exists to remove.


// ---------------------------------------------------------------------------
// The aerosol-aware cold network.
// ---------------------------------------------------------------------------
//
// ABI note: mp=8's kernel carries an include_cold_rain / include_cold_cloud /
// include_snow_rime_conversion / track_graupel_number flag matrix so a
// focused unit gate and the production path can share one translation unit.
// mp=28's production path ALWAYS has cloud water and always includes the
// complete cold-rain group, so those flags are dropped: this kernel is the
// full-group kernel unconditionally.  That is a real scope reduction, not an
// omission -- the disabled branches were unreachable under mp=28.
extern "C" __global__ void thompson_aa_cold_network(
    float* __restrict__ qi,
    float* __restrict__ ni,
    float* __restrict__ qs,
    float* __restrict__ qg,
    float* __restrict__ qr,
    float* __restrict__ nr,
    float* __restrict__ qc,
    float* __restrict__ temperature,
    const float* __restrict__ pressure,
    float* __restrict__ qv,
    // Entry aerosol/droplet state, per kilogram.  READ-ONLY for the whole
    // mp=28 call (:1803-1830).
    const float* __restrict__ nc_entry,
    const float* __restrict__ nwfa_entry,
    const float* __restrict__ nifa_entry,
    // Per-kilogram tendency accumulators, applied once by WP-04's terminal
    // kernel.  This kernel only adds to them.
    float* __restrict__ ncten,
    float* __restrict__ nwfaten,
    float* __restrict__ nifaten,
    float* __restrict__ graupel_number_shadow,
    float* __restrict__ snow_velocity_boost,
    const double* __restrict__ ice_deposition_partition,
    const double* __restrict__ ice_to_snow_mass,
    const double* __restrict__ ice_to_snow_number,
    const double* __restrict__ tcs_racs1,
    const double* __restrict__ tmr_racs1,
    const double* __restrict__ tcs_racs2,
    const double* __restrict__ tmr_racs2,
    const double* __restrict__ tcr_sacr1,
    const double* __restrict__ tms_sacr1,
    const double* __restrict__ tcr_sacr2,
    const double* __restrict__ tms_sacr2,
    const double* __restrict__ tnr_racs1,
    const double* __restrict__ tnr_racs2,
    const double* __restrict__ tnr_sacr1,
    const double* __restrict__ tnr_sacr2,
    const double* __restrict__ tcg_racg,
    const double* __restrict__ tmr_racg,
    const double* __restrict__ tcr_gacr,
    const double* __restrict__ tnr_racg,
    const double* __restrict__ tnr_gacr,
    const double* __restrict__ rain_to_ice_mass,
    const double* __restrict__ rain_to_ice_number,
    const double* __restrict__ rain_to_graupel_mass,
    const double* __restrict__ rain_to_graupel_number,
    const double* __restrict__ rain_cloud_efficiency,
    const double* __restrict__ cloud_to_ice_mass,
    const double* __restrict__ cloud_to_ice_number,
    float dt, int size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    // WRF resets vts_boost on every cold-source call (:2243) before
    // diagnosing the deposition-conditioned rimed-snow conversion.  Warm and
    // empty cells must also receive the neutral value because the later
    // column sedimentation kernel consumes the complete field.
    snow_velocity_boost[idx] = 1.0f;
    if (temperature[idx] >= 273.15f) return;

    const float temp0 = temperature[idx];
    const float qv0 = fmaxf(1.0e-10f, qv[idx]);
    const float qvsi = thompson_rsif(pressure[idx], temp0);
    const float qvs = thompson_rslf(pressure[idx], temp0);
    float ssati = qv0 / qvsi - 1.0f;
    float ssatw = qv0 / qvs - 1.0f;
    if (fabsf(ssati) < 1.0e-15f) ssati = 0.0f;
    if (fabsf(ssatw) < 1.0e-15f) ssatw = 0.0f;
    // :2620-2621.  Koop's own gate (ssati >= 0.4) is strictly stronger than
    // this one, so the fast exit below cannot skip a productive Koop cell.
    const bool nucleation_active = ssati >= 0.25f
        || (ssatw > 1.0e-15f && temp0 < 253.15f);
    if (qi[idx] <= 1.0e-12f && qs[idx] <= 1.0e-12f
            && qg[idx] <= 1.0e-12f && qr[idx] <= 1.0e-12f
            && qc[idx] <= 1.0e-12f && !nucleation_active) return;

    const float rho = 0.622f * pressure[idx]
        / (287.04f * temp0 * (qv0 + 0.622f));
    const float orho = 1.0f / rho;
    // odts = 1./dtsave, :1649.  WRF writes every rate cap as X*odts, never
    // X/dt; with dt = 10 s the two differ because 1/10 is inexact in binary,
    // and the caps are exactly where a MIN can pick a different branch.
    const float inverse_dt = 1.0f / dt;
    const double inverse_dt_d = (double)inverse_dt;
    const float tempc = temp0 - 273.15f;
    const float inverse_temp = 1.0f / temp0;
    const float diffusivity = 2.11e-5f
        * powf(temp0 / 273.15f, 1.94f)
        * (101325.0f / pressure[idx]);
    const float viscosity = (1.718f + 0.0049f * tempc
        - 1.2e-5f * tempc * tempc) * 1.0e-5f;
    const float conductivity = (5.69f + 0.0168f * tempc)
        * 1.0e-5f * 418.936f;
    const float inverse_cp = 1.0f / (1004.0f * (1.0f + 0.887f * qv0));
    const float saturated_density = rho * qvsi;
    const float slope_term = inverse_temp
        * (2.834e6f * inverse_temp * (1.0f / 461.5f) - 1.0f);
    const float saturation_slope = saturated_density * slope_term;
    const float saturation_curvature = saturated_density
        * (slope_term * slope_term
           - 2.0f * 2.834e6f * inverse_temp * inverse_temp
             * inverse_temp * (1.0f / 461.5f)
           + inverse_temp * inverse_temp);
    const float gamma = 2.834e6f * diffusivity / conductivity
        * saturation_slope;
    const float gamma_ratio = gamma / (1.0f + gamma);
    const float alpha = fmaxf(1.0e-9f,
        0.5f * gamma_ratio * gamma_ratio
        * saturation_curvature / saturation_slope
        * saturated_density / saturation_slope);
    float xsat = ssati;
    if (fabsf(xsat) < 1.0e-9f) xsat = 0.0f;
    const float xsat2 = xsat * xsat;
    const float alpha2 = alpha * alpha;
    const float vapor_geometry = 4.0f * 3.1415926536f
        * (1.0f - alpha * xsat
           + 2.0f * alpha2 * xsat2
           - 5.0f * alpha2 * alpha * xsat2 * xsat)
        / (1.0f + gamma);
    // rate_max, :2557 and :2864.
    const float vapor_limit = (qv0 - qvsi) * rho * inverse_dt * 0.999f;

    const float pi = 3.1415926536f;
    const float am_r = pi * 1000.0f / 6.0f;
    // rhof(k) = SQRT(RHO_NOT/rho(k)), :1974.  thompson.cu:6600 writes the
    // algebraically equal RHO_NOT*orho; the two differ by up to an ulp of
    // rhof, which propagates into every collection rate.  mp=28 uses WRF's
    // own division so this kernel and the Fortran oracle agree to the last
    // float bit, exactly as thompson_aerosol_warm.cu already does.
    const float density_factor = sqrtf(
        (101325.0f / (287.05f * 298.0f)) / rho);

    // -------------------------------------------------------------------
    // ENTRY AEROSOL AND DROPLET STATE, module_mp_thompson.F:1803-1842.
    // -------------------------------------------------------------------
    // aer_init_opt < 2 (the v1 scope pin), so both aerosol species take the
    // clamped branch at :1805-1806.
    const float nwfa_work = thompson_aa_clamp_nwfa(nwfa_entry[idx] * rho);
    const float nifa_work = thompson_aa_clamp_nifa(nifa_entry[idx] * rho);

    // L_qc and the entry droplet rediagnosis, :1827-1848.  When qc is at or
    // below R1 WRF sets rc = R1 and nc = 2.
    const bool has_cloud = qc[idx] > 1.0e-12f;
    const float cloud_mass = has_cloud ? qc[idx] * rho : 1.0e-12f;
    // ---- THE nu_c STAGING RULE, :1832 vs :2170 -------------------------
    // WRF computes nu_c TWICE from two DIFFERENT droplet numbers.  :1832
    // uses the PRE-rediagnosis nc of :1829 and exists only to build the
    // entry lamc and run the :1834-1838 droplet-size clamp.  :1840 then
    // REDIAGNOSES nc from that clamped lamc, and :2170 recomputes nu_c from
    // the rediagnosed value.  The :2170 answer is the one that feeds lamc
    // (:2173), mvd_c (:2174), Dc_g (:2181) and pnr_wau (:2191) -- i.e.
    // everything below.  Reusing the entry answer stays finite, stays
    // stable and is grossly wrong wherever the size clamp engages.
    //
    // Both sentinels are 0, never 12: 12 is exactly the value mp=8 froze
    // for Nt_c = 100e6, and a frozen droplet-shape constant sitting in a
    // live kernel is one dropped guard away from becoming the answer.
    // THOMPSON_AA_CCG2[0] and friends are the deliberate zero pad, so a
    // dropped guard produces a division by zero rather than mp=8 physics.
    int nu_c_entry = 0;             // :1832, diagnostic only
    int nu_c = 0;                   // :2170, THE working shape parameter
    double entry_lamc = 0.0;
    float nc_work = THOMPSON_AA_NC_FLOOR;
    if (has_cloud) {
        nc_work = thompson_aa_cloud_dist(
            cloud_mass, nc_entry[idx], rho, &nu_c_entry, &entry_lamc);
        nu_c = thompson_aa_nu_c_working(nc_work);
    }
    // entry_lamc is WRF's clamped entry lamc and nu_c_entry the shape
    // parameter that produced it.  NEITHER is usable after :1838; they are
    // captured so a reader can see which stage is which, and so the
    // readback probe at the bottom of this file can publish both.
    (void)entry_lamc;
    (void)nu_c_entry;

    // Cloud lambda and mean volume diameter, :2173-2175.  thompson.cu:
    // 6939-6944 folds nu_c = 12 into the literals 1.30767389e12f /
    // 2.08767448e-9f and the numerator 15.672f (which IS 3.0 + 12 + 0.672);
    // all three become live.  Written exactly as
    // thompson_aerosol_warm.cu:404-427, including WRF's type mixing: the
    // power is REAL(4) and is WIDENED into the DOUBLE PRECISION lamc
    // (declared at :1597), so mvd_c is a double division rounded once to
    // REAL on assignment -- not a float division.
    double cloud_lambda = 0.0;
    float cloud_mvd = 1.0e-6f;
    if (has_cloud) {
        cloud_lambda = (double)thompson_aa_powf_cr(
            thompson_aa_div(
                thompson_aa_mul(
                    thompson_aa_mul(
                        thompson_aa_mul(nc_work, am_r),
                        THOMPSON_AA_CCG2[nu_c]),
                    THOMPSON_AA_OCG1[nu_c]),
                cloud_mass),
            THOMPSON_AA_OBMR);
        const float mvd = (float)(
            (double)((3.0f + (float)nu_c) + 0.672f) / cloud_lambda);
        cloud_mvd = fmaxf(THOMPSON_AA_D0C,
                          fminf(mvd, THOMPSON_AA_D0R));
    }

    // -------------------------------------------------------------------
    // iceDeMott, evaluated ONCE (:2574 and :2623 return the same value).
    // -------------------------------------------------------------------
    const float xni_demott = thompson_ice_demott(tempc, rho, nifa_work);
    // :2579-2591.  Replaces thompson.cu's hardcoded `nuclei_bin = 27`.
    const int nuclei_bin = thompson_aa_in_bin(xni_demott);
    // :3447-3448.  Replaces thompson.cu's hardcoded `cloud_number_bin = 65`.
    const int cloud_number_bin = thompson_aa_droplet_bin(nc_work);

    // -------------------------------------------------------------------
    // Snow moments, :2027-2101.  Hoisted out of thompson.cu's two separate
    // riming/deposition blocks because the aerosol scavenging terms need
    // smoe and xDs even when there is no cloud water.  Every expression is
    // transcribed unchanged, so the hoist cannot move a bit.
    // -------------------------------------------------------------------
    const float snow_mass = qs[idx] * rho;
    const bool has_snow = qs[idx] > 1.0e-12f;
    const float snow_tc0 = fminf(-0.1f, tempc);
    float smob = 0.0f;
    float smoc = 0.0f;
    float smo0 = 0.0f;
    float smoe = 0.0f;
    float snow_diameter = 0.0f;     // xDs
    float snow_number_ns = 0.0f;    // ns(k), :2081-2088
    if (has_snow) {
        smob = snow_mass * (1.0f / 0.069f);          // rs*oams, bm_s == 2
        // :2054 / :2067 / :2080 / :2101 are all `a_ * smo2**b_` with a_, b_
        // and smo2 REAL(4), which gfortran lowers to glibc's CORRECTLY
        // ROUNDED powf where CUDA's carries several ulp.  MEASURED against a
        // PUBLIC probe holding a verbatim copy of :2028-2088, compiled into
        // the same module_mp_thompson object the column oracle uses, over
        // 3721 states (61 temperatures 273.05 K -> 201.05 K x 61 log-spaced
        // snow contents 1e-12 -> 1e-2 kg m^-3), feeding it WRF's OWN a_ / b_
        // so the fits are not under test:
        //     smoc  plain powf  3489/3721 exact, max 1.663435e-07 relative
        //     smoc  powf_cr     3717/3721 exact, max 1.063558e-07
        //     ns    plain powf  3526/3721 exact, max 5.512328e-07
        //     ns    powf_cr     3716/3721 exact, max 1.344195e-07
        // thompson_field_a / thompson_field_b are BIT-EXACT on all 3721, so
        // the whole of the improvement is the power.  The four survivors are
        // not a fit defect and not repairable here: thompson_aa_powf_cr is
        // `(float)pow((double)x,(double)y)`, which DOUBLE-ROUNDS where
        // glibc's powf rounds once, and at four of these states the double
        // result straddles a float32 midpoint.  Reported, not hidden.
        smoc = thompson_field_a(snow_tc0, 3.0f)
            * thompson_aa_powf_cr(smob, thompson_field_b(snow_tc0, 3.0f));
        smo0 = thompson_field_a(snow_tc0, 0.0f)
            * thompson_aa_powf_cr(smob, thompson_field_b(snow_tc0, 0.0f));
        smoe = thompson_field_a(snow_tc0, 2.55f)
            * thompson_aa_powf_cr(smob, thompson_field_b(snow_tc0, 2.55f));
        snow_diameter = smoc / smob;
        // THE explicit two-gamma integral.  Not smo0, not interchangeable
        // with it, and used by exactly one gate in the whole routine.
        snow_number_ns = thompson_aa_snow_number(smob, smoc);
    }

    // -------------------------------------------------------------------
    // Graupel distribution, hoisted for the same reason.  Transcribed from
    // thompson.cu:7112-7135 (the riming block's reconstruction of WRF's
    // N0_g(k)/ilamg(k), which gpuwm forms from the diagnosed intercept
    // because it carries no prognostic graupel number).
    // -------------------------------------------------------------------
    const float graupel_mass = qg[idx] * rho;
    const float am_g = pi * 400.0f / 6.0f;
    double graupel_lambda = 1.0;
    double graupel_ilam = 1.0;
    double graupel_intercept = 0.0;   // N0_g
    float graupel_diameter = 0.0f;    // xDg = (bm_g+mu_g+1)*ilamg
    // t1_qg_qc = PI*.25*av_g*cgg(9,1), :2432.  cgg(9,1) = 5.23476267.
    const float t1_qg_qc = pi * 0.25f * 442.0f * 5.23476267f;
    // t1_qs_qc = PI*.25*av_s, :794.
    const float t1_qs_qc = pi * 0.25f * 40.0f;
    if (graupel_mass >= 1.0e-6f) {
        const float intercept_power = fmaxf(2.0f, fminf(6.0f,
            3.0f + (2.0f / 7.0f)
                * (log10f(fmaxf(1.0e-9f, graupel_mass)) + 8.0f)));
        const float intercept_guess = powf(10.0f, intercept_power);
        double lambda = (double)powf(
            intercept_guess * am_g * 6.0f / graupel_mass, 0.25f);
        const float mvd = (float)(3.672 / lambda);
        if (mvd > 25.4e-3f) {
            lambda = 3.672 / 25.4e-3;
        } else if (mvd < 50.0e-6f) {
            lambda = 3.672 / 50.0e-6;
        }
        graupel_lambda = lambda;
        graupel_ilam = 1.0 / lambda;
        const float graupel_number = (1.0f / 6.0f) * graupel_mass / am_g
            * (float)(lambda * lambda * lambda);
        graupel_intercept = (double)graupel_number * lambda;
        graupel_diameter = (float)(4.0 * graupel_ilam);
    }

    // -------------------------------------------------------------------
    // Cooper is GONE.  In mp=8 the ice-nucleation target is
    //     MIN(250.0e3f, 5.0f*expf(0.304f*(273.15f - T)))
    // (thompson.cu:6645-6646 and :7189-7200).  Under mp=28 WRF's if/else at
    // :2622-2626 selects iceDeMott instead, so that expression must not
    // appear anywhere in this file.
    // -------------------------------------------------------------------

    double ice_rate = 0.0;
    double ice_to_snow_rate = 0.0;
    double ice_number_rate = 0.0;
    double autoconversion_rate = 0.0;
    double autoconversion_number_rate = 0.0;
    double rain_ice_ice_rate = 0.0;
    double rain_ice_rain_rate = 0.0;
    double rain_ice_ice_number_rate = 0.0;
    double rain_ice_rain_number_rate = 0.0;
    double rain_ice_graupel_rate = 0.0;
    double rain_self_number_rate = 0.0;
    float ice_particle_mass = 1.0e-12f;
    const float ice_mass = qi[idx] * rho;
    // ni(k), :1859-1875.  WRF's entry ice number carries the 5..300 micron
    // mean-diameter clamps below and is R2 = 1e-6 when there is no ice.
    float ice_number = fmaxf(1.0e-6f, ni[idx] * rho);
    const float rain_mass = qr[idx] * rho;
    float rain_number;
    double rain_lambda;
    float rain_mvd;
    // N0_r, :2150.  The shared helper always runs WRF's :2146-2150
    // y-intercept pass, which RE-DERIVES lamr from the bounded nr instead of
    // reusing the clamped lambda, and publishes N0_r = nr*org2*lamr**cre(2).
    // prr_rcw, pnc_rcw, pna_rca and pnd_rcd all read N0_r and
    // (lamr + fv_r)**(-cre(9)) directly, so they must consume these two and
    // never a lambda re-derived from the clamped mvd.
    double rain_intercept_n0;
    const bool rain_active = thompson_aa_entry_rain_distribution(
        qr[idx], nr[idx], rho, &rain_number, &rain_lambda, &rain_mvd,
        &rain_intercept_n0);
    if (rain_active) {
        if (rain_mvd > 50.0e-6f) {
            const float efficiency = 1.0f
                - expf(2300.0f * (rain_mvd - 1950.0e-6f));
            rain_self_number_rate = (double)(
                efficiency * 2.0f * rain_number * rain_mass);
        }
    }
    if (qi[idx] > 1.0e-12f) {
        const float am_i = pi * 890.0f / 6.0f;
        double lambda = (double)powf(
            am_i * 6.0f * ice_number / ice_mass, 1.0f / 3.0f);
        double inverse_lambda = 1.0 / lambda;
        float mean_diameter = (float)(4.0 * inverse_lambda);
        if (mean_diameter < 5.0e-6f) {
            lambda = 4.0 / 5.0e-6;
            inverse_lambda = 1.0 / lambda;
            ice_number = fminf(999.0e3f,
                (1.0f / 6.0f) * ice_mass / am_i
                * (float)(lambda * lambda * lambda));
            mean_diameter = 5.0e-6f;
        } else if (mean_diameter > 300.0e-6f) {
            lambda = 4.0 / 300.0e-6;
            inverse_lambda = 1.0 / lambda;
            ice_number = (1.0f / 6.0f) * ice_mass / am_i
                * (float)(lambda * lambda * lambda);
            mean_diameter = 300.0e-6f;
        }
        ice_particle_mass = am_i * mean_diameter * mean_diameter
            * mean_diameter;
        const int mass_bin = ice_mass > 1.0e-10f
            ? thompson_aa_decade_index(ice_mass, -10, 64) : 0;
        const int number_bin = ice_number > 1.0f
            ? thompson_aa_decade_index(ice_number, 0, 55) : 0;
        double total_rate = (double)(
            0.5f * vapor_geometry * diffusivity * ssati
            * saturated_density * ice_number) * inverse_lambda;
        if (total_rate > 0.0) {
            total_rate = fmin(total_rate, (double)vapor_limit);
            const double ice_fraction = ice_deposition_partition[
                mass_bin + 64 * number_bin];
            ice_rate = ice_fraction * total_rate;
            ice_to_snow_rate = (1.0 - ice_fraction) * total_rate;
        } else {
            total_rate = fmax((double)(-ice_mass * inverse_dt), total_rate);
            total_rate = fmax(total_rate, (double)vapor_limit);
            const float minimum_diameter = powf(1.0e-12f / am_i, 1.0f / 3.0f);
            const float particle_diameter = fmaxf(
                minimum_diameter, mean_diameter);
            const float particle_mass = am_i * particle_diameter
                * particle_diameter * particle_diameter;
            ice_number_rate = total_rate / (double)particle_mass;
            ice_number_rate = fmax(
                (double)(-ice_number * inverse_dt), ice_number_rate);
            ice_rate = total_rate;
        }

        const size_t table_idx = (size_t)mass_bin
            + (size_t)64 * (size_t)number_bin;
        if (mass_bin == 63 || mean_diameter > 1500.0e-6f) {
            autoconversion_rate = (double)(ice_mass * 0.99f)
                * (double)inverse_dt;
            autoconversion_number_rate = (double)(ice_number * 0.95f)
                * (double)inverse_dt;
        } else if (mean_diameter >= 30.0e-6f) {
            autoconversion_rate = fmin(
                (double)(ice_mass * 0.99f) * (double)inverse_dt,
                ice_to_snow_mass[table_idx] * (double)inverse_dt);
            autoconversion_number_rate = fmin(
                (double)(ice_number * 0.95f) * (double)inverse_dt,
                ice_to_snow_number[table_idx] * (double)inverse_dt);
        }

        if (rain_mass >= 1.0e-6f && rain_mvd > 4.0f * mean_diameter) {
            const double rain_intercept = (double)rain_number * rain_lambda;
            const double shifted_lambda = rain_lambda + 195.0;
            const float collection_efficiency = 0.95f;
            const float mass_prefactor = pi * 0.25f * 4854.0f * 6.0f;
            const float rain_mass_prefactor =
                pi * 0.25f * am_r * 4854.0f * 720.0f;
            rain_ice_ice_rate = (double)(
                density_factor * mass_prefactor
                * collection_efficiency * ice_mass)
                * rain_intercept * pow(shifted_lambda, -4.0);
            rain_ice_rain_rate = (double)(
                density_factor * rain_mass_prefactor
                * collection_efficiency * ice_number)
                * rain_intercept * pow(shifted_lambda, -7.0);
            rain_ice_rain_rate = fmin(
                rain_ice_rain_rate, (double)rain_mass * (double)inverse_dt);
            rain_ice_ice_number_rate =
                rain_ice_ice_rate / (double)ice_particle_mass;
            rain_ice_rain_number_rate = (double)(
                density_factor * mass_prefactor
                * collection_efficiency * ice_number)
                * rain_intercept * pow(shifted_lambda, -4.0);
            rain_ice_rain_number_rate = fmin(
                rain_ice_rain_number_rate,
                (double)rain_number * (double)inverse_dt);
            rain_ice_graupel_rate = rain_ice_ice_rate + rain_ice_rain_rate;
        }
    }

    // -------------------------------------------------------------------
    // Cold-rain group, :2492-2553 and :2594-2603.
    // -------------------------------------------------------------------
    (void)tcg_racg;
    double freeze_ice_rate = 0.0;
    double freeze_ice_number_rate = 0.0;
    double freeze_graupel_rate = 0.0;
    double freeze_graupel_number_rate = 0.0;
    double rain_snow_rain_rate = 0.0;
    double rain_snow_category_rate = 0.0;
    double rain_snow_graupel_rate = 0.0;
    double rain_snow_number_rate = 0.0;
    double rain_graupel_rain_rate = 0.0;
    double rain_graupel_graupel_rate = 0.0;
    double rain_graupel_number_rate = 0.0;
    if (rain_active) {
        const int rain_mass_bin = thompson_aa_decade_index(rain_mass, -6, 37);
        const double table_rain_intercept =
            (double)((1.0f / 6.0f) * rain_mass / am_r)
            * rain_lambda * rain_lambda * rain_lambda * rain_lambda;
        const int rain_intercept_bin =
            thompson_aa_decade_index_double(table_rain_intercept, 6, 37);

        // Bigg raindrop freezing, :2594-2603.  idx_IN IS LIVE HERE.
        if (rain_mass > 1.0e-6f) {
            const int temp_bin = max(0, min((int)roundf(-tempc) - 1, 44));
            const size_t table_idx = (size_t)rain_mass_bin
                + (size_t)37 * ((size_t)rain_intercept_bin
                + (size_t)37 * ((size_t)temp_bin
                + (size_t)45 * (size_t)nuclei_bin));
            freeze_ice_rate = rain_to_ice_mass[table_idx] * inverse_dt_d;
            freeze_ice_number_rate =
                rain_to_ice_number[table_idx] * inverse_dt_d;
            freeze_graupel_rate =
                rain_to_graupel_mass[table_idx] * inverse_dt_d;
            freeze_graupel_number_rate = fmin(
                (double)rain_number,
                rain_to_graupel_number[table_idx]) * inverse_dt_d;
        } else if (rain_mass > 1.0e-12f && temp0 < 235.16f) {
            freeze_ice_rate = (double)thompson_aa_mul(rain_mass, inverse_dt);
            freeze_ice_number_rate =
                (double)thompson_aa_mul(rain_number, inverse_dt);
        }

        if (rain_mass >= 1.0e-6f && snow_mass >= 1.0e-6f) {
            const int snow_bin = thompson_aa_decade_index(snow_mass, -6, 37);
            const int raw_temp_bin = (int)((tempc - 2.5f) / 5.0f) - 1;
            const int temp_bin = min(9, max(1, -raw_temp_bin)) - 1;
            const size_t table_idx = (size_t)snow_bin
                + (size_t)37 * ((size_t)temp_bin
                + (size_t)9 * ((size_t)rain_intercept_bin
                + (size_t)37 * (size_t)rain_mass_bin));
            rain_snow_rain_rate = -(
                tmr_racs2[table_idx] + tcr_sacr2[table_idx]
                + tmr_racs1[table_idx] + tcr_sacr1[table_idx]);
            rain_snow_category_rate =
                tmr_racs2[table_idx] + tcr_sacr2[table_idx]
                - tcs_racs1[table_idx] - tms_sacr1[table_idx];
            rain_snow_graupel_rate =
                tmr_racs1[table_idx] + tcr_sacr1[table_idx]
                + tcs_racs1[table_idx] + tms_sacr1[table_idx];
            rain_snow_number_rate =
                tnr_racs1[table_idx] + tnr_racs2[table_idx]
                + tnr_sacr1[table_idx] + tnr_sacr2[table_idx];
            rain_snow_rain_rate = fmax(
                (double)(-thompson_aa_mul(rain_mass, inverse_dt)),
                rain_snow_rain_rate);
            rain_snow_category_rate = fmax(
                (double)(-thompson_aa_mul(snow_mass, inverse_dt)),
                rain_snow_category_rate);
            rain_snow_graupel_rate = fmin(
                (double)thompson_aa_mul(rain_mass + snow_mass, inverse_dt),
                rain_snow_graupel_rate);
            rain_snow_number_rate = fmin(
                (double)thompson_aa_mul(rain_number, inverse_dt),
                rain_snow_number_rate);
        }

        if (rain_mass >= 1.0e-6f && graupel_mass >= 1.0e-6f) {
            const int graupel_mass_bin = thompson_aa_decade_index(
                graupel_mass, -6, 37);
            const float intercept_power = fmaxf(2.0f, fminf(6.0f,
                3.0f + (2.0f / 7.0f)
                    * (log10f(fmaxf(1.0e-9f, graupel_mass)) + 8.0f)));
            const int graupel_intercept_bin =
                thompson_aa_decade_index_double(
                    (double)powf(10.0f, intercept_power), 2, 37);
            const size_t nominal_idx = (size_t)graupel_intercept_bin
                + (size_t)37 * ((size_t)graupel_mass_bin
                + (size_t)37 * ((size_t)0
                + (size_t)1 * ((size_t)rain_intercept_bin
                + (size_t)37 * (size_t)rain_mass_bin)));
            const size_t table_idx = nominal_idx + (size_t)4 * 37 * 37;
            const size_t table_size = (size_t)37 * 37 * 1 * 37 * 37;
            if (table_idx < table_size) {
                rain_graupel_graupel_rate =
                    tmr_racg[table_idx] + tcr_gacr[table_idx];
                rain_graupel_graupel_rate = fmin(
                    (double)thompson_aa_mul(rain_mass, inverse_dt),
                    rain_graupel_graupel_rate);
                rain_graupel_rain_rate = -rain_graupel_graupel_rate;
                rain_graupel_number_rate =
                    tnr_racg[table_idx] + tnr_gacr[table_idx];
                rain_graupel_number_rate = fmin(
                    (double)thompson_aa_mul(rain_number, inverse_dt),
                    rain_graupel_number_rate);
            }
        }
    }

    // -------------------------------------------------------------------
    // Cold cloud-water group, :2179-2222 and :2402-2443 and :2607-2617.
    // -------------------------------------------------------------------
    double cloud_autoconversion_rate = 0.0;
    double cloud_autoconversion_number_rate = 0.0;
    double cloud_rain_accretion_rate = 0.0;
    double cloud_freezing_rate = 0.0;
    double cloud_freezing_number_rate = 0.0;
    double snow_riming_rate = 0.0;
    double graupel_riming_rate = 0.0;
    double hm_number_rate = 0.0;
    double hm_mass_rate = 0.0;
    double snow_hm_rate = 0.0;
    double graupel_hm_rate = 0.0;
    // NEW under mp=28: the two cloud-number collection sinks.
    double cloud_number_snow_rate = 0.0;      // pnc_scw, :2410-2411
    double cloud_number_graupel_rate = 0.0;   // pnc_gcw, :2435-2437
    // -------------------------------------------------------------------
    // THE FOUR :2156-2232 NUMBER/AEROSOL TERMS THAT ALSO RUN SUB-FREEZING.
    // -------------------------------------------------------------------
    // WRF's warm-rain loop opens at :2157 and closes at :2234.  It precedes
    // the `if (.not. iiwarm)` gate at :2239 AND the `if (temp(k).lt.T_0)`
    // guard at :2554, so every rate inside it is evaluated at EVERY level,
    // sub-freezing ones included.  This kernel already carries the MASS
    // members of that loop (prr_wau at :2189-2190 and prr_rcw at :2202-2204)
    // because they are cloud-water sinks the cold group has to compete with;
    // their NUMBER and AEROSOL partners were missing, so cloud mass left
    // without its droplet number and rain wet-scavenging of CCN/IN was
    // absent from the entire sub-freezing half of the domain.
    //
    // NO DOUBLE COUNTING.  thompson_aerosol_warm.cu computes the same four
    // terms and returns early unless its held entry-temperature mask
    // (T_entry >= 273.15 K, seeded before ANY source kernel runs) is set;
    // this kernel returns at :268 for temperature >= 273.15 f and runs
    // FIRST, before any kernel has moved the temperature field.  The two
    // masks are therefore exact complements over every cell, with 273.15 K
    // itself owned by the warm side.  tests/test_thompson_aerosol_cold_gpu.py
    // ::test_cold_and_warm_temperature_masks_are_exact_complements pins that
    // at the two floats bracketing 273.15 as well as at 273.15 itself.
    double cloud_number_autoconversion_rate = 0.0;  // pnc_wau, :2192-2193
    double cloud_number_rain_rate = 0.0;            // pnc_rcw, :2205-2207
    double nwfa_rain_rate = 0.0;                    // pna_rca, :2213-2216
    double nifa_rain_rate = 0.0;                    // pnd_rcd, :2218-2221
    {
        // Berry-Reinhardt (1974) warm-rain autoconversion, :2179-2194.
        //
        // CONTRACTION-PINNED, matching thompson_aerosol_warm.cu:395-445 term
        // for term so the two halves of the scheme cannot disagree about
        // prr_wau.  Dc_b subtracts two nearly equal sixth powers (:2182), so
        // a single fused multiply-add moves it by several float ulps.
        // build_aero.sh compiles the oracle with plain `gfortran -O2` for
        // baseline x86-64, which has NO fma instruction, so every WRF product
        // and sum in this chain is separately rounded, while nvrtc defaults
        // to --fmad=true.  MEASURED over 11340 sub-freezing Fortran-oracle
        // rows spanning rc up to 0.15 kg m^-3: the unpinned form sat at
        // 1.03e-5 relative on prr_wau / pnr_wau / pnc_wau; pinned, and with
        // lamc and Dc_g carrying WRF's REAL-into-DOUBLE type mixing, 2.31e-6.
        // What is left is thompson_aa_cloud_dist's CUDA powf upstream.
        if (cloud_mass > 0.01e-3f && has_cloud) {
            // xDc, :2172.  D0c*1.E6 == 1.0 exactly.
            const float xdc = fmaxf(1.0f,
                thompson_aa_mul(
                    thompson_aa_powf_cr(
                        thompson_aa_div(cloud_mass,
                                        thompson_aa_mul(am_r, nc_work)),
                        THOMPSON_AA_OBMR),
                    1.0e6f));
            // Dc_g = ((ccg(3,nu_c)*ocg2(nu_c))**obmr / lamc) * 1e6, :2181.
            const float dcg = (float)(
                (double)thompson_aa_powf_cr(
                    thompson_aa_mul(THOMPSON_AA_CCG3[nu_c],
                                    THOMPSON_AA_OCG2[nu_c]),
                    THOMPSON_AA_OBMR)
                / cloud_lambda * 1.0e6);
            // WRF writes both sixth powers left-associated and un-grouped
            // (:2182-2183); the grouping is observable in float32.
            const float xdc3 = thompson_aa_mul(
                thompson_aa_mul(xdc, xdc), xdc);
            const float dcb_arg = thompson_aa_sub(
                thompson_aa_mul(thompson_aa_mul(
                    thompson_aa_mul(xdc3, dcg), dcg), dcg),
                thompson_aa_mul(thompson_aa_mul(
                    thompson_aa_mul(xdc3, xdc), xdc), xdc));
            const float dcb = thompson_aa_powf_cr(
                fmaxf(0.0f, dcb_arg), 1.0f / 6.0f);
            const float zeta_term = thompson_aa_sub(
                thompson_aa_mul(thompson_aa_mul(thompson_aa_mul(
                    thompson_aa_mul(6.25e-6f, xdc), dcb), dcb), dcb),
                0.4f);
            const float zeta1 = thompson_aa_mul(
                0.5f, thompson_aa_add(zeta_term, fabsf(zeta_term)));
            const float zeta = thompson_aa_mul(
                thompson_aa_mul(0.027f, cloud_mass), zeta1);
            const float tau_diameter = thompson_aa_sub(
                thompson_aa_mul(0.5f, dcb), 7.5f);
            const float taud = thompson_aa_add(
                thompson_aa_mul(
                    0.5f, thompson_aa_add(tau_diameter, fabsf(tau_diameter))),
                THOMPSON_AA_R1);
            const float tau = thompson_aa_div(
                3.72f, thompson_aa_mul(cloud_mass, taud));
            cloud_autoconversion_rate = fmin(
                (double)thompson_aa_mul(cloud_mass, inverse_dt),
                (double)thompson_aa_div(zeta, tau));
            // pnr_wau, :2191.  nu_c appears EXPLICITLY; thompson.cu:3241 and
            // :6973 fold it into the literal 12.0f.  WRF's denominator is
            // written un-grouped, so it is left-associated here.
            cloud_autoconversion_number_rate = cloud_autoconversion_rate
                / (double)thompson_aa_mul(
                    thompson_aa_mul(
                        thompson_aa_mul(
                            thompson_aa_mul(
                                thompson_aa_mul(am_r, (float)nu_c), 10.0f),
                            THOMPSON_AA_D0R),
                        THOMPSON_AA_D0R),
                    THOMPSON_AA_D0R);
            // pnc_wau, :2192-2193.  The droplet number that leaves with the
            // autoconverted mass, sized by the CLAMPED mvd_c -- not by xDc
            // and not by Dc_g.  prr_wau here is already MIN'd at :2190.
            cloud_number_autoconversion_rate = fmin(
                (double)thompson_aa_mul(nc_work, inverse_dt),
                cloud_autoconversion_rate
                / (double)thompson_aa_mul(
                    thompson_aa_mul(
                        thompson_aa_mul(am_r, cloud_mvd), cloud_mvd),
                    cloud_mvd));
        }

        // Rain collecting cloud water and aerosol, :2196-2222.
        //
        // All four rates share ONE kernel,
        //     rhof(k)*t1_qr_qc*Ef*X*N0_r(k)*((lamr+fv_r)**(-cre(9)))
        // with X in {rc, nc, nwfa, nifa} and Ef the matching efficiency.
        // t1_qr_qc = PI*.25*av_r*crg(9) (:786) and cre(9) = mu_r+bv_r+3 = 4
        // exactly, so the power is pow(lamr+195, -4).  WRF's TYPES are
        // reproduced literally: rhof, t1_qr_qc, Ef and X are REAL and are
        // multiplied left-to-right in float32; N0_r and lamr are DOUBLE
        // PRECISION, so the trailing two factors are double.
        //
        // GATES.  :2197 needs L_qr .and. mvd_r > D0r .and. mvd_c > D0c;
        // :2211 needs only L_qr .and. mvd_r > D0r, so rain scavenges CCN and
        // IN out of clear supercooled air with no cloud droplets present at
        // all -- exactly the state the cold network used to leave untouched.
        // L_qr is the MIXING-RATIO test qr1d > R1 (:1878), not qr*rho > R1;
        // the shared entry helper returns precisely that as rain_active.
        if (rain_active && rain_mvd > 50.0e-6f) {
            const float coefficient = pi * 0.25f * 4854.0f * 6.0f;
            const double tail = rain_intercept_n0
                * pow(rain_lambda + 195.0, -4.0);
            const float prefactor = density_factor * coefficient;

            if (cloud_mvd > 1.0e-6f) {
                const double dr_first = 5.1164649614037726e-05;
                const double dr_last = 0.004886186104779057;
                int rain_bin = 1 + (int)(100.0
                    * log((double)rain_mvd / dr_first)
                    / log(dr_last / dr_first));
                rain_bin = min(rain_bin, 100);
                const int cloud_bin = (int)(cloud_mvd * 1.0e6f);
                const float efficiency = (float)rain_cloud_efficiency[
                    (rain_bin - 1) + 100 * (cloud_bin - 1)];
                cloud_rain_accretion_rate = fmin(
                    (double)(cloud_mass * inverse_dt),
                    (double)(prefactor * efficiency * cloud_mass) * tail);
                // pnc_rcw, :2205-2207.  Same kernel with nc in place of rc,
                // MIN'd against nc*odts.  WRF caps this one even though it
                // deliberately leaves prg_gcw raw.
                cloud_number_rain_rate = fmin(
                    (double)nc_work * (double)inverse_dt,
                    (double)(prefactor * efficiency * nc_work) * tail);
            }

            // pna_rca (:2212-2216) and pnd_rcd (:2218-2221).  Each species is
            // capped against the FULL available aerosol on its own; WRF lets
            // the sum of all six scavenging rates overshoot and relies on the
            // terminal floors at :3979-3981, so no shared limiter here.  The
            // wif_input_opt == 2 black-carbon term (:2223-2230) is out of
            // scope for this port.
            const float ef_ccn = thompson_eff_aero(
                rain_mvd, 0.04e-6f, viscosity, rho, temp0,
                THOMPSON_AA_SPECIES_RAIN);
            nwfa_rain_rate = fmin(
                (double)nwfa_work * (double)inverse_dt,
                (double)(prefactor * ef_ccn * nwfa_work) * tail);
            const float ef_in = thompson_eff_aero(
                rain_mvd, 0.8e-6f, viscosity, rho, temp0,
                THOMPSON_AA_SPECIES_RAIN);
            nifa_rain_rate = fmin(
                (double)nifa_work * (double)inverse_dt,
                (double)(prefactor * ef_in * nifa_work) * tail);
        }

        // Bigg cloud-droplet freezing, :2607-2617.  BOTH table indices that
        // mp=8 freezes are live here: cloud_number_bin (idx_n) and
        // nuclei_bin (idx_IN).
        if (cloud_mass > 1.0e-6f) {
            const int mass_bin = thompson_aa_decade_index(cloud_mass, -6, 37);
            const int temp_bin = max(0, min((int)roundf(-tempc) - 1, 44));
            const size_t table_idx = (size_t)mass_bin
                + (size_t)37 * ((size_t)cloud_number_bin
                + (size_t)100 * ((size_t)temp_bin
                + (size_t)45 * (size_t)nuclei_bin));
            cloud_freezing_rate = fmin(
                (double)thompson_aa_mul(cloud_mass, inverse_dt),
                cloud_to_ice_mass[table_idx] * inverse_dt_d);
            cloud_freezing_number_rate = fmin(
                (double)thompson_aa_mul(nc_work, inverse_dt),
                fmin(cloud_freezing_rate / (2.0 * 1.0e-12),
                     cloud_to_ice_number[table_idx] * inverse_dt_d));
        } else if (has_cloud && cloud_mass > 1.0e-12f && temp0 < 235.16f) {
            cloud_freezing_rate =
                (double)thompson_aa_mul(cloud_mass, inverse_dt);
            // :2616 -- nc(k)*odts, not Nt_c*odts.
            cloud_freezing_number_rate =
                (double)thompson_aa_mul(nc_work, inverse_dt);
        }

        // Snow collecting cloud water, :2402-2412.
        if (has_cloud && has_snow && cloud_mvd > 1.0e-6f
                && snow_diameter > 300.0e-6f) {
            const double diameter_ratio = 0.02 / 300.0e-6;
            const double log_ratio = log(diameter_ratio);
            const double first_snow_bin = 300.0e-6
                * exp(0.5 / 100.0 * log_ratio);
            const double last_snow_bin = 300.0e-6
                * exp(99.5 / 100.0 * log_ratio);
            int snow_bin = 1 + (int)(100.0
                * log((double)snow_diameter / first_snow_bin)
                / log(last_snow_bin / first_snow_bin));
            snow_bin = min(snow_bin, 100);
            const double table_snow_diameter = 300.0e-6
                * exp(((double)snow_bin - 0.5) / 100.0 * log_ratio);
            const int cloud_bin = (int)(cloud_mvd * 1.0e6f);
            const double table_cloud_diameter = (double)cloud_bin * 1.0e-6;
            const double cloud_velocity = 1.19e4
                * (1.0e4 * table_cloud_diameter
                   * table_cloud_diameter * 0.25);
            const double snow_velocity = 40.0
                * pow(table_snow_diameter, 0.55)
                * exp(-100.0 * table_snow_diameter) - cloud_velocity;
            const double melted_snow_diameter = pow(
                0.069 * table_snow_diameter * table_snow_diameter
                    / (pi * 1000.0 / 6.0), 1.0 / 3.0);
            const double diameter_fraction = table_cloud_diameter
                / melted_snow_diameter;
            float efficiency = 0.0f;
            if (diameter_fraction <= 0.25
                    && table_cloud_diameter >= 6.0e-6
                    && snow_velocity >= 1.0e-3) {
                const double stokes_number = table_cloud_diameter
                    * table_cloud_diameter * snow_velocity * 1000.0
                    / (9.0 * 1.718e-5 * melted_snow_diameter);
                const double reynolds_number = 9.0 * stokes_number
                    / (diameter_fraction * diameter_fraction * 1000.0);
                const double log_reynolds = log(reynolds_number);
                const double k0 = exp(-0.1007 - 0.358 * log_reynolds
                    + 0.0261 * log_reynolds * log_reynolds);
                const double z = log(stokes_number / (k0 + 1.0e-15));
                const double h = 0.1465 + 1.302 * z - 0.607 * z * z
                    + 0.293 * z * z * z;
                const double yc0 = 2.0 / 3.14159265358979323846 * atan(h);
                const double value = (yc0 + diameter_fraction)
                    * (yc0 + diameter_fraction)
                    / ((1.0 + diameter_fraction) * (1.0 + diameter_fraction));
                efficiency = fmaxf(0.0f, fminf((float)value, 0.95f));
            }
            snow_riming_rate = (double)(density_factor * t1_qs_qc
                * efficiency * cloud_mass * smoe);
            snow_riming_rate = fmin(
                snow_riming_rate,
                (double)thompson_aa_mul(cloud_mass, inverse_dt));
            // pnc_scw, :2410-2411.  Same coefficients, nc in place of rc.
            cloud_number_snow_rate = (double)(density_factor * t1_qs_qc
                * efficiency * nc_work * smoe);
            cloud_number_snow_rate = fmin(
                cloud_number_snow_rate, (double)nc_work * (double)inverse_dt);
        }

        // Graupel collecting cloud water, :2414-2443.
        if (has_cloud && graupel_mass >= 1.0e-6f && cloud_mvd > 1.0e-6f) {
            const float velocity = (float)(
                (double)(density_factor * 442.0f * 20.3632278f
                         * (1.0f / 6.0f))
                * pow(graupel_ilam, 0.89));
            const float stokes_number = cloud_mvd * cloud_mvd
                * velocity * 1000.0f
                / (9.0f * viscosity * graupel_diameter);
            float efficiency = 0.0f;
            if (stokes_number >= 0.4f && stokes_number <= 10.0f) {
                efficiency = 0.55f * log10f(2.51f * stokes_number);
            } else if (stokes_number > 10.0f) {
                efficiency = 0.77f;
            }
            graupel_riming_rate = (double)(density_factor * t1_qg_qc
                * efficiency * cloud_mass)
                * graupel_intercept * pow(graupel_ilam, 3.89);
            // pnc_gcw, :2435-2437.  MIN'd against nc*odts; WRF leaves the
            // paired MASS rate prg_gcw raw until the joint cloud-water
            // conservation pass, and does NOT bound it here.
            cloud_number_graupel_rate = (double)(density_factor * t1_qg_qc
                * efficiency * nc_work)
                * graupel_intercept * pow(graupel_ilam, 3.89);
            cloud_number_graupel_rate = fmin(
                cloud_number_graupel_rate,
                (double)nc_work * (double)inverse_dt);
        }

        // Hallett-Mossop, :2425-2440 in thompson.cu terms.  Held when the
        // later cloud-water cap rescales the riming mass sources.
        if (graupel_riming_rate > 1.0e-15 && tempc > -8.0f) {
            float factor = 0.0f;
            if (tempc >= -5.0f && tempc < -3.0f) {
                factor = 0.5f * (-3.0f - tempc);
            } else if (tempc > -8.0f && tempc < -5.0f) {
                factor = 0.33333333f * (8.0f + tempc);
            }
            hm_number_rate = 3.5e8 * (double)factor * graupel_riming_rate;
            hm_mass_rate = 1.0e-12 * hm_number_rate;
            const double total_riming = snow_riming_rate + graupel_riming_rate;
            if (total_riming > 0.0) {
                snow_hm_rate = snow_riming_rate / total_riming * hm_mass_rate;
                graupel_hm_rate =
                    graupel_riming_rate / total_riming * hm_mass_rate;
            }
        }
    }

    // -------------------------------------------------------------------
    // FROZEN AEROSOL WET SCAVENGING, :2444-2478.
    // -------------------------------------------------------------------
    // Gated on rs > r_s(1) and rg > r_g(1) alone -- NOT on cloud water and
    // NOT on the riming efficiencies.  Each rate is capped against the FULL
    // available aerosol independently; WRF deliberately allows the four
    // rates to over-deplete in sum and relies on the terminal floors.
    double nwfa_snow_rate = 0.0;      // pna_sca
    double nifa_snow_rate = 0.0;      // pnd_scd
    double nwfa_graupel_rate = 0.0;   // pna_gca
    double nifa_graupel_rate = 0.0;   // pnd_gcd
    if (snow_mass > 1.0e-6f) {
        const float ef_ccn = thompson_eff_aero(
            snow_diameter, 0.04e-6f, viscosity, rho, temp0,
            THOMPSON_AA_SPECIES_SNOW);
        nwfa_snow_rate = (double)(density_factor * t1_qs_qc * ef_ccn
                                  * nwfa_work * smoe);
        nwfa_snow_rate = fmin(nwfa_snow_rate,
                              (double)nwfa_work * (double)inverse_dt);

        const float ef_in = thompson_eff_aero(
            snow_diameter, 0.8e-6f, viscosity, rho, temp0,
            THOMPSON_AA_SPECIES_SNOW);
        nifa_snow_rate = (double)(density_factor * t1_qs_qc * ef_in
                                  * nifa_work * smoe);
        nifa_snow_rate = fmin(nifa_snow_rate,
                              (double)nifa_work * (double)inverse_dt);
    }
    if (graupel_mass > 1.0e-6f) {
        const float ef_ccn = thompson_eff_aero(
            graupel_diameter, 0.04e-6f, viscosity, rho, temp0,
            THOMPSON_AA_SPECIES_GRAUPEL);
        nwfa_graupel_rate = (double)(density_factor * t1_qg_qc * ef_ccn
                                     * nwfa_work)
            * graupel_intercept * pow(graupel_ilam, 3.89);
        nwfa_graupel_rate = fmin(nwfa_graupel_rate,
                                 (double)nwfa_work * (double)inverse_dt);

        const float ef_in = thompson_eff_aero(
            graupel_diameter, 0.8e-6f, viscosity, rho, temp0,
            THOMPSON_AA_SPECIES_GRAUPEL);
        nifa_graupel_rate = (double)(density_factor * t1_qg_qc * ef_in
                                     * nifa_work)
            * graupel_intercept * pow(graupel_ilam, 3.89);
        nifa_graupel_rate = fmin(nifa_graupel_rate,
                                 (double)nifa_work * (double)inverse_dt);
    }

    // -------------------------------------------------------------------
    // DEPOSITION NUCLEATION, :2619-2631.  iceDeMott, not Cooper.
    // -------------------------------------------------------------------
    // The target population subtracts the crystals this same call already
    // created by Bigg freezing of rain AND cloud droplets
    // (xni = ni(k) + (pni_rfz + pni_wfz)*dtsave, :2628), so it must run
    // after both freezing blocks.
    double nucleation_rate = 0.0;         // pri_inu
    double nucleation_number_rate = 0.0;  // pni_inu
    if (nucleation_active) {
        const float existing = ice_number
            + (float)((freeze_ice_number_rate + cloud_freezing_number_rate)
                      * (double)dt);
        const float delta = xni_demott - existing;
        nucleation_number_rate =
            (double)(0.5f * (delta + fabsf(delta)) * inverse_dt);
        nucleation_rate = fmin(
            (double)vapor_limit, 1.0e-12 * nucleation_number_rate);
        nucleation_number_rate = nucleation_rate / 1.0e-12;
    }

    // -------------------------------------------------------------------
    // KOOP (2001) HOMOGENEOUS FREEZING OF DELIQUESCED HAZE, :2633-2643.
    // -------------------------------------------------------------------
    double koop_rate = 0.0;         // pri_iha
    double koop_number_rate = 0.0;  // pni_iha
    {
        const float xni_test = snow_number_ns + ice_number
            + (float)((freeze_ice_number_rate + cloud_freezing_number_rate
                       + nucleation_number_rate) * (double)dt);
        if (xni_test <= 999.0e3f && temp0 < 238.0f && ssati >= 0.4f) {
            const float xnc = thompson_ice_koop(
                temp0, qv0, qvs, nwfa_work, dt);
            koop_number_rate = (double)(xnc * inverse_dt);
            // xm0i*0.1 -- a homogeneously frozen haze droplet starts TEN
            // TIMES smaller than a deposition-nucleated crystal.
            koop_rate = fmin((double)vapor_limit, 1.0e-13 * koop_number_rate);
            koop_number_rate = koop_rate / 1.0e-13;
        }
    }

    // -------------------------------------------------------------------
    // Snow and graupel vapor exchange, :2688-2712.
    // -------------------------------------------------------------------
    double snow_rate = 0.0;
    double snow_collection_rate = 0.0;
    double snow_collection_number_rate = 0.0;
    if (has_snow) {
        const float rho_factor_sqrt = sqrtf(density_factor);
        const float viscosity_factor = sqrtf(rho / viscosity);
        // :2067 and :2114, both `a_ * smo2**b_` -- see the measurement note
        // above the smoc/smo0/smoe block for why these are powf_cr.
        const float snow_first_moment = thompson_field_a(snow_tc0, 1.0f)
            * thompson_aa_powf_cr(smob,
                                  thompson_field_b(snow_tc0, 1.0f));
        const float deposition_moment = 1.0f + (1.0f + 0.55f) * 0.5f;
        const float snow_ventilation_moment =
            thompson_field_a(snow_tc0, deposition_moment)
            * thompson_aa_powf_cr(
                smob, thompson_field_b(snow_tc0, deposition_moment));
        const float snow_capacitance = fmaxf(0.15f, fminf(
            0.15f + (tempc + 1.5f) * (0.5f - 0.15f) / (-30.0f + 1.5f), 0.5f));
        const float ventilation_coefficient = 0.28f
            * powf(0.632f, 1.0f / 3.0f) * sqrtf(40.0f);
        const float moment_sum = 0.86f * snow_first_moment
            + ventilation_coefficient * rho_factor_sqrt
              * viscosity_factor * snow_ventilation_moment;
        snow_rate = (double)(
            snow_capacitance * vapor_geometry * diffusivity * ssati
            * saturated_density * moment_sum);
        if (snow_rate > 0.0) {
            snow_rate = fmin(snow_rate, (double)vapor_limit);
        } else {
            snow_rate = fmax((double)(-snow_mass * inverse_dt), snow_rate);
            snow_rate = fmax(snow_rate, (double)vapor_limit);
        }

        if (qi[idx] > 1.0e-12f && snow_mass >= 1.0e-6f) {
            snow_collection_rate = (double)(
                t1_qs_qc * density_factor * 0.05f * ice_mass * smoe);
            snow_collection_number_rate = snow_collection_rate
                / (double)ice_particle_mass;
        }
    }

    // Deposition-conditioned rimed-snow to graupel conversion, :2754-2777.
    double snow_graupel_conversion_rate = 0.0;
    double snow_graupel_conversion_number_rate = 0.0;
    if (snow_riming_rate > 2.0 * snow_rate && snow_rate > 1.0e-15) {
        const float rime_ratio = (float)fmin(
            30.0, snow_riming_rate / snow_rate);
        float graupel_fraction = fminf(
            0.95f, 0.15f + (rime_ratio - 2.0f) * 0.028f);
        snow_velocity_boost[idx] = fminf(
            1.5f, 1.1f + (rime_ratio - 2.0f) * 0.014f);
        snow_graupel_conversion_rate =
            (double)graupel_fraction * snow_riming_rate;
        snow_graupel_conversion_number_rate = snow_graupel_conversion_rate
            * (double)smo0 / (double)snow_mass;

        const float snow_velocity = 40.0f
            * powf(snow_diameter, 0.55f)
            * expf(-100.0f * snow_diameter);
        float rime_parameter = -(cloud_mvd * 0.5e6f) * snow_velocity
            / fminf(-0.1f, tempc);
        rime_parameter = fmaxf(0.1f, fminf(rime_parameter, 10.0f));
        const float rime_density = (0.051f + 0.114f * rime_parameter
            - 0.0055f * rime_parameter * rime_parameter) * 1000.0f;
        if (rime_density < 150.0f) {
            graupel_fraction = 0.0f;
            snow_graupel_conversion_rate = 0.0;
            snow_graupel_conversion_number_rate = 0.0;
        }
        snow_riming_rate = (1.0 - (double)graupel_fraction) * snow_riming_rate;
    }

    double graupel_rate = 0.0;
    if (qg[idx] > 1.0e-12f && ssati < -1.0e-15f) {
        const float rho_factor_sqrt = sqrtf(density_factor);
        const float viscosity_factor = sqrtf(rho / viscosity);
        const float intercept_power = fmaxf(2.0f, fminf(
            3.0f + (2.0f / 7.0f)
                * (log10f(fmaxf(1.0e-9f, graupel_mass)) + 8.0f),
            6.0f));
        const float diagnosed_intercept = powf(10.0f, intercept_power);
        float lambda = powf(
            diagnosed_intercept * am_g * 6.0f / graupel_mass, 0.25f);
        float number_per_kg = (1.0f / 6.0f) * graupel_mass
            * powf(lambda, 3.0f) / am_g / rho;
        number_per_kg = fmaxf(1.0e-6f, number_per_kg);
        float graupel_number = fmaxf(1.0e-6f, number_per_kg * rho);
        lambda = powf(am_g * 6.0f * graupel_number / graupel_mass,
                      1.0f / 3.0f);
        float mvd = 3.672f / lambda;
        if (mvd > 25.4e-3f) {
            mvd = 25.4e-3f;
            lambda = 3.672f / mvd;
            graupel_number = (1.0f / 6.0f) * graupel_mass
                * powf(lambda, 3.0f) / am_g;
        } else if (mvd < 50.0e-6f) {
            mvd = 50.0e-6f;
            lambda = 3.672f / mvd;
            graupel_number = (1.0f / 6.0f) * graupel_mass
                * powf(lambda, 3.0f) / am_g;
        }
        const float inverse_lambda = 1.0f / lambda;
        const float intercept = graupel_number * lambda;
        const float ventilation_coefficient = 0.28f
            * powf(0.632f, 1.0f / 3.0f)
            * sqrtf(442.0f) * 1.9021706581115723f;
        const float moment_sum = intercept * (
            0.86f * powf(inverse_lambda, 2.0f)
            + ventilation_coefficient * viscosity_factor
              * rho_factor_sqrt * powf(inverse_lambda, 2.945f));
        graupel_rate = (double)(
            0.5f * vapor_geometry * diffusivity * ssati
            * saturated_density * moment_sum);
        graupel_rate = fmax((double)(-graupel_mass * inverse_dt), graupel_rate);
        graupel_rate = fmax(graupel_rate, (double)vapor_limit);
    }

    // -------------------------------------------------------------------
    // SHARED FROZEN-VAPOR LIMITER, :2862-2876.
    // -------------------------------------------------------------------
    // sump = pri_inu + pri_ide + prs_ide + prs_sde + prg_gde + pri_iha.
    // pri_iha JOINS the sum and IS rescaled; pni_iha and pni_inu are
    // deliberately left alone (WRF rescales only the six MASS members plus
    // pni_ide).
    double vapor_sum = nucleation_rate + ice_rate + ice_to_snow_rate
        + snow_rate + graupel_rate + koop_rate;
    if ((vapor_sum > 1.0e-15 && vapor_sum > (double)vapor_limit)
            || (vapor_sum < -1.0e-15 && vapor_sum < (double)vapor_limit)) {
        const double ratio = (double)vapor_limit / vapor_sum;
        nucleation_rate *= ratio;
        ice_rate *= ratio;
        ice_to_snow_rate *= ratio;
        ice_number_rate *= ratio;
        snow_rate *= ratio;
        graupel_rate *= ratio;
        koop_rate *= ratio;
        vapor_sum = (double)vapor_limit;
    }

    // Cloud-water conservation, :2878-2890.  The paired number tendencies
    // (pnc_scw, pnc_gcw, pni_wfz) and the H-M terms intentionally stay held
    // when these mass rates are rescaled.
    const double cloud_limit = (double)(-cloud_mass * inverse_dt);
    const double cloud_sum = -cloud_autoconversion_rate
        - cloud_freezing_rate - cloud_rain_accretion_rate
        - snow_riming_rate - snow_graupel_conversion_rate
        - graupel_riming_rate;
    if (has_cloud && cloud_sum < cloud_limit) {
        const double ratio = cloud_limit / cloud_sum;
        cloud_autoconversion_rate *= ratio;
        cloud_freezing_rate *= ratio;
        cloud_rain_accretion_rate *= ratio;
        snow_riming_rate *= ratio;
        snow_graupel_conversion_rate *= ratio;
        graupel_riming_rate *= ratio;
    }

    // Cloud-ice conservation, :2892-2903.
    const double ice_limit = (double)(-ice_mass * inverse_dt);
    const double ice_sum = ice_rate - autoconversion_rate
        - snow_collection_rate - rain_ice_ice_rate;
    if (qi[idx] > 1.0e-12f && ice_sum < (double)-1.0e-15f
            && ice_sum < ice_limit) {
        const double ratio = ice_limit / ice_sum;
        const double unbounded_ice_rate = ice_rate;
        ice_rate *= ratio;
        autoconversion_rate *= ratio;
        snow_collection_rate *= ratio;
        rain_ice_ice_rate *= ratio;
        vapor_sum += ice_rate - unbounded_ice_rate;
    }

    // Rain conservation, :2905-2917.
    const double rain_limit = (double)(-rain_mass * inverse_dt);
    const double rain_sum = -freeze_graupel_rate - freeze_ice_rate
        - rain_ice_rain_rate + rain_snow_rain_rate + rain_graupel_rain_rate;
    if (rain_active && rain_sum < rain_limit) {
        const double ratio = rain_limit / rain_sum;
        freeze_graupel_rate *= ratio;
        freeze_ice_rate *= ratio;
        rain_ice_rain_rate *= ratio;
        rain_snow_rain_rate *= ratio;
        rain_graupel_rain_rate *= ratio;
    }

    // Snow conservation, :2919-2929.
    const double snow_limit = (double)(-snow_mass * inverse_dt);
    const double snow_sum = snow_rate - snow_hm_rate + rain_snow_category_rate;
    if (has_snow && snow_sum < snow_limit) {
        const double ratio = snow_limit / snow_sum;
        const double unbounded_snow_vapor_rate = snow_rate;
        snow_rate *= ratio;
        snow_hm_rate *= ratio;
        rain_snow_category_rate *= ratio;
        vapor_sum += snow_rate - unbounded_snow_vapor_rate;
    }

    // Graupel conservation, :2931-2941.
    const double graupel_limit = (double)(-graupel_mass * inverse_dt);
    const double graupel_sum = graupel_rate - graupel_hm_rate
        + rain_graupel_graupel_rate;
    if (qg[idx] > 1.0e-12f && graupel_sum < graupel_limit) {
        const double ratio = graupel_limit / graupel_sum;
        const double unbounded_graupel_vapor_rate = graupel_rate;
        graupel_rate *= ratio;
        graupel_hm_rate *= ratio;
        rain_graupel_graupel_rate *= ratio;
        vapor_sum += graupel_rate - unbounded_graupel_vapor_rate;
    }

    // Blossey re-enforcement of the paired rain/graupel transfer.
    const double paired_rate = fmin(
        fabs(rain_graupel_rain_rate), fabs(rain_graupel_graupel_rate));
    rain_graupel_rain_rate = -paired_rate;
    rain_graupel_graupel_rate = paired_rate;
    hm_mass_rate = snow_hm_rate + graupel_hm_rate;

    const double rain_rate = cloud_autoconversion_rate
        + cloud_rain_accretion_rate
        - freeze_graupel_rate - freeze_ice_rate - rain_ice_rain_rate
        + rain_snow_rain_rate + rain_graupel_rain_rate;
    const double rain_number_sink = rain_self_number_rate
        + freeze_graupel_number_rate + freeze_ice_number_rate
        + rain_ice_rain_number_rate + rain_snow_number_rate
        + rain_graupel_number_rate;

    const float latent_vapor = 2.5e6f + (2106.0f - 4218.0f) * tempc;
    const float latent_fusion = 2.834e6f - latent_vapor;
    const double fusion_rate = cloud_freezing_rate
        + snow_riming_rate + snow_graupel_conversion_rate
        + graupel_riming_rate
        + freeze_ice_rate + freeze_graupel_rate
        + rain_ice_rain_rate + rain_snow_category_rate
        + rain_snow_graupel_rate + rain_graupel_graupel_rate;
    // :3974, `qv1d(k) = MAX(1.E-10, qv1d(k) + qvten(k)*DT)`.  Pinned for the
    // reason given at the qc apply below.
    const float post_source_qv = fmaxf(1.0e-10f,
        thompson_aa_sub(qv0,
            thompson_aa_mul((float)(vapor_sum * (double)orho), dt)));
    const float post_source_temperature = temp0 + (float)(
        (double)(2.834e6f * inverse_cp) * vapor_sum
        * (double)orho * (double)dt
        + (double)(latent_fusion * inverse_cp) * fusion_rate
        * (double)orho * (double)dt);

    // -------------------------------------------------------------------
    // State update, :3020-3031 and :3096-3120.
    // -------------------------------------------------------------------
    //
    // CONTRACTION, AND WHY EVERY MULTIPLY-ADD BELOW IS PINNED (WP-13b).
    // WRF applies the accumulated tendencies at :3973-4023 in the form
    // `q1d(k) = q1d(k) + qten(k)*DT`, and the gfortran -O2 baseline-x86-64
    // build the oracle comes from has NO FMA instruction: qten*DT is rounded
    // to REAL(4) BEFORE the add.  Written plainly, nvrtc (--fmad=true)
    // contracts each of these into a single fma and never rounds the product,
    // so the two disagree by up to one ulp of the ENTRY value -- which is a
    // 1e-5 RELATIVE error wherever the level is nearly emptied, and that is
    // exactly where this port's surviving residuals live.
    //
    // MEASURED, wp08-freeze level 0: entry qc = 3.0000001424923540e-04, this
    // kernel's qc tendency -2.727245919231791e-05, dt = 10.  The rounded
    // product gives 2.7275411412119865e-05, which is WRF's own answer; the
    // fused form gives 2.7275422326056287e-05, which is what these lines
    // produced before WP-13b.  Same at aero-cold-overlap level 4:
    // 1.5486410120502114e-05 (WRF, and now this kernel) against
    // 1.5486406482523307e-05 (fused).
    //
    // thompson_aerosol_sat.cu:911-916 already pinned exactly this multiply-add
    // for the rain-evaporation apply, so the source networks were the
    // inconsistent half -- this is not a new tie-break, it is the same one as
    // "THE SHARED FITS" in thompson_aerosol_common.cuh: the authority is WRF,
    // not ArWen's mp=8 sibling.  Pinned by tests/
    // test_thompson_aerosol_adapter.py::
    // test_the_source_networks_round_the_tendency_before_adding_it, which
    // tests the PROPERTY (post - entry must round-trip through float32) and
    // not a constant.
    qi[idx] = fmaxf(0.0f, thompson_aa_add(qi[idx],
        thompson_aa_mul(
            (float)((nucleation_rate + koop_rate + hm_mass_rate
                     + cloud_freezing_rate + freeze_ice_rate + ice_rate
                     - autoconversion_rate
                     - snow_collection_rate - rain_ice_ice_rate)
                    * (double)orho),
            dt)));
    ni[idx] = fmaxf(0.0f, thompson_aa_add(ni[idx],
        thompson_aa_mul(
            (float)((nucleation_number_rate + koop_number_rate
                     + hm_number_rate
                     + cloud_freezing_number_rate + freeze_ice_number_rate
                     + ice_number_rate
                     - autoconversion_number_rate
                     - snow_collection_number_rate
                     - rain_ice_ice_number_rate)
                    * (double)orho),
            dt)));
    qs[idx] = fmaxf(0.0f, thompson_aa_add(qs[idx],
        thompson_aa_mul(
            (float)((ice_to_snow_rate + snow_rate + autoconversion_rate
                     + snow_collection_rate + snow_riming_rate
                     + rain_snow_category_rate - snow_hm_rate)
                    * (double)orho),
            dt)));
    qg[idx] = fmaxf(0.0f, thompson_aa_add(qg[idx],
        thompson_aa_mul(
            (float)((graupel_rate + freeze_graupel_rate
                     + snow_graupel_conversion_rate
                     + graupel_riming_rate - graupel_hm_rate
                     + rain_ice_graupel_rate + rain_snow_graupel_rate
                     + rain_graupel_graupel_rate)
                    * (double)orho),
            dt)));
    {
        // ng1d, the classic wrapper's private, untransported graupel number.
        // Signs mirror :3107-3109.  mp=28 changes nothing here: calc_refl10cm
        // and the graupel sedimentation both read it exactly as mp=8 does.
        const float initial_number_per_kg = graupel_number_shadow[idx];
        const float initial_number = qg[idx] > 0.0f
            ? fmaxf(1.0e-6f, initial_number_per_kg * rho) : 0.0f;
        const double graupel_vapor_number_rate = graupel_mass > 1.0e-12f
            ? graupel_rate * (double)initial_number / (double)graupel_mass
            : 0.0;
        const double number_rate = freeze_graupel_number_rate
            + snow_graupel_conversion_number_rate
            + rain_ice_rain_number_rate
            + rain_snow_number_rate + graupel_vapor_number_rate;
        graupel_number_shadow[idx] = thompson_aa_add(initial_number_per_kg,
            thompson_aa_mul((float)(number_rate * (double)orho), dt));
    }
    qr[idx] = fmaxf(0.0f, thompson_aa_add(qr[idx],
        thompson_aa_mul((float)(rain_rate * (double)orho), dt)));
    nr[idx] = fmaxf(0.0f, thompson_aa_add(nr[idx],
        thompson_aa_mul(
            (float)((cloud_autoconversion_number_rate - rain_number_sink)
                    * (double)orho),
            dt)));
    // :3975, `qc1d(k) = qc1d(k) + qcten(k)*DT`.  See the contraction note at
    // the head of this block; this is the apply the two measured cells there
    // were taken from.
    qc[idx] = fmaxf(0.0f, thompson_aa_sub(qc[idx],
        thompson_aa_mul(
            (float)((cloud_autoconversion_rate + cloud_freezing_rate
                     + cloud_rain_accretion_rate + snow_riming_rate
                     + snow_graupel_conversion_rate
                     + graupel_riming_rate) * (double)orho),
            dt)));
    thompson_aa_bound_ice_number(qi[idx] * rho, rho, &ni[idx]);
    thompson_aa_bound_rain_number(qr[idx] * rho, rho, &nr[idx]);
    qv[idx] = post_source_qv;
    temperature[idx] = post_source_temperature;

    // -------------------------------------------------------------------
    // AEROSOL AND DROPLET NUMBER ACCUMULATORS, :2964-2994.
    // -------------------------------------------------------------------
    // WRF, verbatim and COMPLETE for a sub-freezing level:
    //   ncten   += (-pnc_wau - pnc_rcw - pni_wfz - pnc_scw - pnc_gcw)*orho
    //                                                        (:2993-2995)
    //   nwfaten -= (pna_rca + pna_sca + pna_gca + pni_iha)*orho (:2964-2965)
    //   nifaten -= (pnd_rcd + pnd_scd + pnd_gcd)*orho          (:2966-2967)
    //   nifaten -= pni_inu*orho     (:2974-2976, dustyIce is PARAMETER .true.)
    //
    // pnc_wau, pnc_rcw, pna_rca and pnd_rcd come from WRF's :2157-2234 loop,
    // which runs at EVERY level.  They used to be omitted here on the theory
    // that WP-07's warm kernel owned them; it does, but only for cells whose
    // ENTRY temperature was >= 273.15 K, and this kernel owns exactly the
    // complement.  Leaving them out therefore deleted them outright below
    // freezing rather than delegating them.
    //
    // pni_wfz is the value AFTER its own MIN chain at :2609-2612 but BEFORE
    // the cloud-water mass rescale, exactly as WRF uses it: the limiter at
    // :2880-2890 lists only the six MASS members.
    //
    // pni_iha and pni_inu are likewise the pre-vapor-limiter values.
    //
    // The sums are accumulated in DOUBLE and rounded once, matching WRF's
    // REAL accumulator fed by a DOUBLE right-hand side, and are grouped
    // left-to-right in WRF's own order.
    ncten[idx] = (float)((double)ncten[idx]
        + ((((-cloud_number_autoconversion_rate
              - cloud_number_rain_rate)
              - cloud_freezing_number_rate)
              - cloud_number_snow_rate)
              - cloud_number_graupel_rate) * (double)orho);
    nwfaten[idx] = (float)((double)nwfaten[idx]
        - (((nwfa_rain_rate + nwfa_snow_rate)
            + nwfa_graupel_rate) + koop_number_rate) * (double)orho);
    nifaten[idx] = (float)((double)nifaten[idx]
        - ((nifa_rain_rate + nifa_snow_rate)
           + nifa_graupel_rate) * (double)orho);
    nifaten[idx] = (float)((double)nifaten[idx]
        - nucleation_number_rate * (double)orho);
}


// ---------------------------------------------------------------------------
// READBACK PROBE.
// ---------------------------------------------------------------------------
//
// thompson_aa_cold_network's droplet-distribution staging and its four
// :2157-2234 number/aerosol rates have no end-to-end observable in the
// committed fixtures: every divergent nu_c state clamps mvd_c to D0c or D0r,
// and pnr_wau / Dc_g only switch on above rc = 0.01e-3.  A defect there is
// therefore invisible to a state-level comparison, which is exactly how the
// wrong-stage nu_c survived wave 2.  This kernel publishes the internals so
// the host can compare them against WRF v4.6.1 pointwise.
//
// It re-derives them from the SAME shared helpers the production kernel uses
// (thompson_aa_cloud_dist, thompson_aa_nu_c_working,
// thompson_aa_entry_rain_distribution, thompson_eff_aero) and reproduces the
// same expressions; tests/test_thompson_aerosol_cold_gpu.py additionally
// gates the probe against the production kernel's own ncten/nwfaten/nifaten
// on a shared column, so the two cannot drift apart silently.
extern "C" __global__ void thompson_aa_probe_cold_warm_loop(
    const float* __restrict__ qc,
    const float* __restrict__ nc_entry,
    const float* __restrict__ qr,
    const float* __restrict__ nr,
    const float* __restrict__ nwfa_entry,
    const float* __restrict__ nifa_entry,
    const float* __restrict__ temperature,
    const float* __restrict__ pressure,
    const float* __restrict__ qv,
    const double* __restrict__ rain_cloud_efficiency,
    int* __restrict__ nu_c_entry_out,
    int* __restrict__ nu_c_working_out,
    float* __restrict__ nc_m3_out,
    float* __restrict__ mvd_c_out,
    float* __restrict__ mvd_r_out,
    double* __restrict__ pnc_wau_out,
    double* __restrict__ pnc_rcw_out,
    double* __restrict__ pna_rca_out,
    double* __restrict__ pnd_rcd_out,
    double* __restrict__ prr_wau_out,
    double* __restrict__ pnr_wau_out,
    float dt, int size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    const float temp0 = temperature[idx];
    const float qv0 = fmaxf(1.0e-10f, qv[idx]);
    const float rho = 0.622f * pressure[idx]
        / (287.04f * temp0 * (qv0 + 0.622f));
    const float inverse_dt = 1.0f / dt;
    const float tempc = temp0 - 273.15f;
    const float viscosity = (1.718f + 0.0049f * tempc
        - 1.2e-5f * tempc * tempc) * 1.0e-5f;
    const float density_factor = sqrtf(
        (101325.0f / (287.05f * 298.0f)) / rho);
    const float pi = THOMPSON_AA_PI;
    const float am_r = THOMPSON_AA_AM_R;

    const float nwfa_work = thompson_aa_clamp_nwfa(nwfa_entry[idx] * rho);
    const float nifa_work = thompson_aa_clamp_nifa(nifa_entry[idx] * rho);

    const bool has_cloud = qc[idx] > 1.0e-12f;
    const float cloud_mass = has_cloud ? qc[idx] * rho : 1.0e-12f;
    int nu_c_entry = 0;
    int nu_c = 0;
    double entry_lamc = 0.0;
    float nc_work = THOMPSON_AA_NC_FLOOR;
    double cloud_lambda = 0.0;
    float cloud_mvd = 1.0e-6f;
    if (has_cloud) {
        nc_work = thompson_aa_cloud_dist(
            cloud_mass, nc_entry[idx], rho, &nu_c_entry, &entry_lamc);
        nu_c = thompson_aa_nu_c_working(nc_work);
        cloud_lambda = (double)thompson_aa_powf_cr(
            thompson_aa_div(
                thompson_aa_mul(
                    thompson_aa_mul(
                        thompson_aa_mul(nc_work, am_r),
                        THOMPSON_AA_CCG2[nu_c]),
                    THOMPSON_AA_OCG1[nu_c]),
                cloud_mass),
            THOMPSON_AA_OBMR);
        const float mvd = (float)(
            (double)((3.0f + (float)nu_c) + 0.672f) / cloud_lambda);
        cloud_mvd = fmaxf(THOMPSON_AA_D0C, fminf(mvd, THOMPSON_AA_D0R));
    }

    float rain_number;
    double rain_lambda;
    float rain_mvd;
    double rain_intercept_n0;
    const bool rain_active = thompson_aa_entry_rain_distribution(
        qr[idx], nr[idx], rho, &rain_number, &rain_lambda, &rain_mvd,
        &rain_intercept_n0);

    double prr_wau = 0.0;
    double pnr_wau = 0.0;
    double pnc_wau = 0.0;
    if (cloud_mass > 0.01e-3f && has_cloud) {
        const float xdc = fmaxf(1.0f,
            thompson_aa_mul(
                thompson_aa_powf_cr(
                    thompson_aa_div(cloud_mass,
                                    thompson_aa_mul(am_r, nc_work)),
                    THOMPSON_AA_OBMR),
                1.0e6f));
        const float dcg = (float)(
            (double)thompson_aa_powf_cr(
                thompson_aa_mul(THOMPSON_AA_CCG3[nu_c],
                                THOMPSON_AA_OCG2[nu_c]),
                THOMPSON_AA_OBMR)
            / cloud_lambda * 1.0e6);
        const float xdc3 = thompson_aa_mul(thompson_aa_mul(xdc, xdc), xdc);
        const float dcb_arg = thompson_aa_sub(
            thompson_aa_mul(thompson_aa_mul(
                thompson_aa_mul(xdc3, dcg), dcg), dcg),
            thompson_aa_mul(thompson_aa_mul(
                thompson_aa_mul(xdc3, xdc), xdc), xdc));
        const float dcb = thompson_aa_powf_cr(
            fmaxf(0.0f, dcb_arg), 1.0f / 6.0f);
        const float zeta_term = thompson_aa_sub(
            thompson_aa_mul(thompson_aa_mul(thompson_aa_mul(
                thompson_aa_mul(6.25e-6f, xdc), dcb), dcb), dcb),
            0.4f);
        const float zeta1 = thompson_aa_mul(
            0.5f, thompson_aa_add(zeta_term, fabsf(zeta_term)));
        const float zeta = thompson_aa_mul(
            thompson_aa_mul(0.027f, cloud_mass), zeta1);
        const float tau_diameter = thompson_aa_sub(
            thompson_aa_mul(0.5f, dcb), 7.5f);
        const float taud = thompson_aa_add(
            thompson_aa_mul(
                0.5f, thompson_aa_add(tau_diameter, fabsf(tau_diameter))),
            THOMPSON_AA_R1);
        const float tau = thompson_aa_div(
            3.72f, thompson_aa_mul(cloud_mass, taud));
        prr_wau = fmin((double)thompson_aa_mul(cloud_mass, inverse_dt),
                       (double)thompson_aa_div(zeta, tau));
        pnr_wau = prr_wau / (double)thompson_aa_mul(
            thompson_aa_mul(
                thompson_aa_mul(
                    thompson_aa_mul(
                        thompson_aa_mul(am_r, (float)nu_c), 10.0f),
                    THOMPSON_AA_D0R),
                THOMPSON_AA_D0R),
            THOMPSON_AA_D0R);
        pnc_wau = fmin(
            (double)thompson_aa_mul(nc_work, inverse_dt),
            prr_wau / (double)thompson_aa_mul(
                thompson_aa_mul(
                    thompson_aa_mul(am_r, cloud_mvd), cloud_mvd),
                cloud_mvd));
    }

    double pnc_rcw = 0.0;
    double pna_rca = 0.0;
    double pnd_rcd = 0.0;
    if (rain_active && rain_mvd > 50.0e-6f) {
        const float coefficient = pi * 0.25f * 4854.0f * 6.0f;
        const double tail = rain_intercept_n0
            * pow(rain_lambda + 195.0, -4.0);
        const float prefactor = density_factor * coefficient;
        if (cloud_mvd > 1.0e-6f) {
            const double dr_first = 5.1164649614037726e-05;
            const double dr_last = 0.004886186104779057;
            int rain_bin = 1 + (int)(100.0
                * log((double)rain_mvd / dr_first)
                / log(dr_last / dr_first));
            rain_bin = min(rain_bin, 100);
            const int cloud_bin = (int)(cloud_mvd * 1.0e6f);
            const float efficiency = (float)rain_cloud_efficiency[
                (rain_bin - 1) + 100 * (cloud_bin - 1)];
            pnc_rcw = fmin(
                (double)nc_work * (double)inverse_dt,
                (double)(prefactor * efficiency * nc_work) * tail);
        }
        const float ef_ccn = thompson_eff_aero(
            rain_mvd, 0.04e-6f, viscosity, rho, temp0,
            THOMPSON_AA_SPECIES_RAIN);
        pna_rca = fmin(
            (double)nwfa_work * (double)inverse_dt,
            (double)(prefactor * ef_ccn * nwfa_work) * tail);
        const float ef_in = thompson_eff_aero(
            rain_mvd, 0.8e-6f, viscosity, rho, temp0,
            THOMPSON_AA_SPECIES_RAIN);
        pnd_rcd = fmin(
            (double)nifa_work * (double)inverse_dt,
            (double)(prefactor * ef_in * nifa_work) * tail);
    }

    nu_c_entry_out[idx] = nu_c_entry;
    nu_c_working_out[idx] = nu_c;
    nc_m3_out[idx] = nc_work;
    mvd_c_out[idx] = cloud_mvd;
    mvd_r_out[idx] = rain_mvd;
    pnc_wau_out[idx] = pnc_wau;
    pnc_rcw_out[idx] = pnc_rcw;
    pna_rca_out[idx] = pna_rca;
    pnd_rcd_out[idx] = pnd_rcd;
    prr_wau_out[idx] = prr_wau;
    pnr_wau_out[idx] = pnr_wau;
}
