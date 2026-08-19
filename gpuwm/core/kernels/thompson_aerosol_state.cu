// gpuwm/core/kernels/thompson_aerosol_state.cu
//
// WP-04 -- aerosol state kernels for WRF v4.6.1 aerosol-aware Thompson
// (mp_physics=28).  Numerical authority is
// WRF v4.6.1 phys/module_mp_thompson.F, commit
// d66e442fccc04111067e29274c9f9eaccc3cef28, zero local modifications.  Every
// bare line number below refers to that file.
//
// This translation unit receives gpuwm/core/kernels/thompson_aerosol_common.cuh
// textually, prepended by gpuwm/core/kernels/__init__.py's _EXTRA_HEADERS
// allow-list.  Every nu_c, every gamma moment, every clamp and every lookup
// index comes from that header.  Nothing here re-derives them.
//
// ===========================================================================
// THE ACCUMULATOR CONTRACT -- this file owns it
// ===========================================================================
// WRF runs one monolithic column loop that
//   (a) freezes nc1d/nwfa1d/nifa1d as read-only ENTRY state at :1795-1842,
//   (b) accumulates ncten/nwfaten/nifaten (per KILOGRAM per second; every
//       increment site multiplies by orho -- :2964, :2975, :3008, :3833) in
//       regions separated by thousands of lines,
//   (c) applies them ONCE at :3972-4021 with a single set of clamps.
//
// ArWen's fused network launchers write state in place, so mp=28 must make
// that split explicit:
//
//   * state nc / nwfa / nifa are READ-ONLY entry state for the whole call.
//   * three device scratch accumulators ncten / nwfaten / nifaten are zeroed
//     at adapter entry and written by every aerosol kernel.
//   * thompson_aa_state_finalize below is the ONLY place the accumulators are
//     applied to state, and it carries WRF's ONLY set of clamps.
//   * any kernel needing a working per-m3 value RECOMPUTES it locally exactly
//     the way WRF does (thompson_aa_working_number /
//     thompson_aa_working_cloud below), instead of applying its own delta.
//
// Four other packages depend on there being exactly ONE clamp point, because
// WRF has exactly one.  Clamping four times is a silent physics change that no
// unit test downstream would flag.
//
// ===========================================================================
// TWO DISTINCT AEROSOL SNAPSHOTS -- the asymmetry is intentional
// ===========================================================================
//   :1805-1806  ENTRY   nwfa = MAX(11.1E6, MIN(9999.E6, nwfa1d*rho))
//                       nifa = MAX(naIN1*0.01, MIN(9999.E6, nifa1d*rho))
//               Feeds scavenging, iceDeMott and iceKoop.  BOTH bounds.
//
//   :3211       WORKING nwfa = MAX(11.1E6, (nwfa1d + nwfaten*DT)*rho)
//               Feeds activ_ncloud ONLY.  NO upper bound, and there is NO
//               nifa counterpart at all.  rho here is the TAU+1 density
//               recomputed at :3193 from the updated temp/qv, NOT the entry
//               rho of :1802.
//
// Conflating the two changes activated droplet number wherever scavenging was
// significant.  Reproduce, do not smooth.
//
// ===========================================================================
// WRF UNIT INCONSISTENCIES THAT ARE REPRODUCED LITERALLY
// ===========================================================================
// :3976  nc1d(k) = MAX(2./rho(k), MIN(nc1d(k) + ncten(k)*DT, Nt_c_max))
//        nc1d is PER KILOGRAM but is compared against the volumetric
//        Nt_c_max = 1999.E6 with no density conversion.  Do NOT divide by rho
//        to "fix" the upper bound; the lower bound IS converted and the upper
//        bound is not.  Fixture aero-nc-cap is what pins this.
// :3979-3982  nwfa1d/nifa1d, also per kilogram, clamped against the per-m3
//        constants 11.1E6 / 5.0E3 / 9999.E6.  Same treatment.
// :4020  the terminal droplet rediagnosis caps at DBLE(Nt_c_max)/rho(k),
//        i.e. the SAME constant but converted.  Three different conventions
//        in nine lines; all three are transcribed as written.
//
// ===========================================================================
// HEIGHT FIELD FOR thompson_init's PROFILE FILL  (resolved, see the .py)
// ===========================================================================
// WRF passes hgt=z_at_q (module_physics_init.F:4517-4544).  Despite the name,
// dyn_em/start_em.F:870-876 fills it as
//     z_at_q(i,k,j) = (grid%ph_2(i,k,j)+grid%phb(i,k,j))/g,  k = kts..kte
// and ph/phb are Z-STAGGERED in Registry.EM_COMMON:198-200, i.e. FULL (w)
// levels.  z_at_q is therefore the w-level height above SEA level, truncated
// to the lowest kte entries -- it is NOT the mass-level height.  hgt(i,1,j) is
// the terrain elevation, which is what the ABSOLUTE 1000 m / 2500 m h_01
// thresholds are testing.  ArWen's exact analogue is z8w[:nz] with
// z8w = (phb + php)/G, which gpuwm/core/microphysics.py::_apply_thompson
// already materializes.  Passing the mass-level height 0.5*(z8w[k]+z8w[k+1])
// instead would shift h_01 over terrain and reshape the whole CCN profile.
//
#define THOMPSON_AA_STATE_EPS 1.0e-15f   // :185, thompson_init's fill test

// Correctly-rounded float32 cosine, for thompson_init's h_01 branch only.
// gfortran lowers REAL(4) COS to glibc cosf (correctly rounded); CUDA's cosf
// carries ~2 ulp.  Same rationale as thompson_aa_expf_cr in the shared header.
__device__ __forceinline__ float thompson_aa_cosf_cr(float x)
{
    return (float)cos((double)x);
}

// module_mp_thompson.F:1802, :3193, :5624 -- one definition, three sites.
// qv must already be MAX'd at 1.E-10 by the caller, exactly as :1801 does.
__device__ __forceinline__ float thompson_aa_density(
    float pressure, float temperature, float qv)
{
    return 0.622f * pressure / (287.04f * temperature * (qv + 0.622f));
}


// ---------------------------------------------------------------------------
// 1.  ENTRY SNAPSHOT -- module_mp_thompson.F:1795-1812, aer_init_opt < 2.
// ---------------------------------------------------------------------------
//
// The read-only per-m3 aerosol state for the whole call.  Written once at
// adapter entry, then never again; the scavenging, iceDeMott and iceKoop
// kernels read THIS, not state.nwfa/state.nifa and not the working refresh.
//
// aer_init_opt is pinned at 0 for this port (SCOPE PIN, spec "Strategy"), so
// only the .lt. 2 branch exists here.  wif_input_opt is pinned at 0, so nbca
// is identically zero and is not carried at all.
//
// rho_out is WRF's entry density (:1802).  It is an output rather than an
// input because :1801's MAX(1.E-10, qv1d) must be applied before the density,
// and having one kernel own that ordering is what stops five packages from
// each writing a slightly different rho.
extern "C" __global__ void thompson_aa_entry_snapshot(
    const float* __restrict__ temperature,
    const float* __restrict__ pressure,
    const float* __restrict__ qv,
    const float* __restrict__ nwfa,      // state, per kilogram, READ-ONLY
    const float* __restrict__ nifa,      // state, per kilogram, READ-ONLY
    float* __restrict__ rho_out,
    float* __restrict__ nwfa_entry_m3,
    float* __restrict__ nifa_entry_m3,
    int n)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;

    // :1801  qv(k) = MAX(1.E-10, qv1d(k))   -- the density uses the CLAMPED
    // vapour, so the clamp cannot be hoisted out of this kernel.
    const float qv_local = fmaxf(1.0e-10f, qv[idx]);
    const float rho = thompson_aa_density(pressure[idx], temperature[idx],
                                          qv_local);
    rho_out[idx] = rho;

    // :1805  nwfa(k) = MAX(11.1E6, MIN(9999.E6, nwfa1d(k)*rho(k)))
    // :1806  nifa(k) = MAX(naIN1*0.01, MIN(9999.E6, nifa1d(k)*rho(k)))
    //        naIN1*0.01 = 0.5E6*0.01 = 5.0E3 exactly.
    nwfa_entry_m3[idx] = thompson_aa_clamp_nwfa(nwfa[idx] * rho);
    nifa_entry_m3[idx] = thompson_aa_clamp_nifa(nifa[idx] * rho);
}


// ---------------------------------------------------------------------------
// 1b. ENTRY CLOUD-DROPLET DIAGNOSIS -- module_mp_thompson.F:1826-1842.
// ---------------------------------------------------------------------------
//
// The other half of the entry pack.  Kept in this file so that the entry
// state has exactly one owner; the warm, cold and saturation packages may
// either consume these outputs or call thompson_aa_cloud_dist from the shared
// header themselves -- either way the arithmetic is the header's, so the two
// halves of the scheme cannot drift.
//
// L_qc_out is WRF's L_qc(k) as int32 (1/0).  On the false branch WRF also
// ZEROES qc1d and nc1d in place (:1844-1845); that is done here too, which is
// why qc and nc are mutable.  rc_out is the CONTENT in kg m^-3 (never below
// R1), nc_entry_m3 the rediagnosed droplet number in m^-3.
extern "C" __global__ void thompson_aa_entry_cloud_number(
    float* __restrict__ qc,              // state, per kilogram, zeroed if <= R1
    float* __restrict__ nc,              // state, per kilogram, zeroed if <= R1
    const float* __restrict__ rho,
    float* __restrict__ rc_out,          // kg m^-3
    float* __restrict__ nc_entry_m3,     // m^-3
    int* __restrict__ nu_c_out,
    int* __restrict__ l_qc_out,
    int n)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;

    const float rho_local = rho[idx];
    const float qc_local = qc[idx];

    if (qc_local > THOMPSON_AA_R1) {
        // :1828-1841.  thompson_aa_cloud_dist carries WRF's type mixing:
        // a REAL power widened to DOUBLE, REAL size clamps, a DOUBLE
        // rediagnosis.  Do not re-derive it here.
        const float rc = qc_local * rho_local;
        int nu_c = 0;
        double lamc = 0.0;
        const float nc_m3 = thompson_aa_cloud_dist(rc, nc[idx], rho_local,
                                                   &nu_c, &lamc);
        rc_out[idx] = rc;
        nc_entry_m3[idx] = nc_m3;
        nu_c_out[idx] = nu_c;
        l_qc_out[idx] = 1;
    } else {
        // :1843-1848
        qc[idx] = 0.0f;
        nc[idx] = 0.0f;
        rc_out[idx] = THOMPSON_AA_R1;
        nc_entry_m3[idx] = THOMPSON_AA_NC_FLOOR;
        // nu_c is undefined on this branch in WRF (the whole level is
        // switched off by L_qc).  Publish the nc=2 value so a downstream
        // read of a switched-off level is deterministic rather than stale.
        nu_c_out[idx] = thompson_aa_nu_c(THOMPSON_AA_NC_FLOOR);
        l_qc_out[idx] = 0;
    }
}


// ---------------------------------------------------------------------------
// 2.  WORKING AEROSOL REFRESH -- module_mp_thompson.F:3211.
// ---------------------------------------------------------------------------
//
//     nwfa(k) = MAX(11.1E6, (nwfa1d(k) + nwfaten(k)*DT)*rho(k))
//
// This is the SECOND, DISTINCT snapshot, and it is consumed by activ_ncloud
// alone (:3416-3421).  Differences from the entry snapshot at :1805, all
// deliberate:
//   * no 9999.E6 ceiling,
//   * no nifa counterpart anywhere in the scheme,
//   * rho is the TAU+1 density recomputed at :3193 from the post-tendency
//     temp/qv, not the entry rho of :1802.
// The caller must pass that TAU+1 rho.  Use thompson_aa_tau1_density below to
// build it so the definition stays in one place.
//
// nwfa is the READ-ONLY entry per-kg state.  Nothing here writes state.
extern "C" __global__ void thompson_aa_working_number(
    const float* __restrict__ nwfa,      // state, per kilogram, READ-ONLY
    const float* __restrict__ nwfaten,   // accumulator, per kilogram per s
    const float* __restrict__ rho,       // TAU+1 density, :3193
    float dt,
    float* __restrict__ nwfa_work_m3,
    int n)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;

    const float updated = thompson_aa_add(
        nwfa[idx], thompson_aa_mul(nwfaten[idx], dt));
    nwfa_work_m3[idx] = fmaxf(THOMPSON_AA_NWFA_FLOOR,
                              thompson_aa_mul(updated, rho[idx]));
}


// module_mp_thompson.F:3189-3193 -- the TAU+1 state the working refresh and
// the whole condensation region are evaluated on.
//     temp(k) = t1d(k) + DT*tten(k)
//     qv(k)   = MAX(1.E-10, qv1d(k) + DT*qvten(k))
//     rho(k)  = 0.622*pres(k)/(R*temp(k)*(qv(k)+0.622))
// Supplied here as a kernel so that the working refresh, the saturation
// package and the finalize kernel cannot disagree about which density they
// mean.  ArWen's networks write temperature and qv in place rather than
// carrying tten/qvten, so the caller passes the ALREADY-UPDATED TAU+1 fields
// and this kernel only re-applies :3192's vapour floor and forms the density.
// It is deliberately identical arithmetic to thompson_aa_entry_snapshot's
// density; only the inputs differ.
extern "C" __global__ void thompson_aa_tau1_density(
    const float* __restrict__ temperature,   // TAU+1
    const float* __restrict__ pressure,
    const float* __restrict__ qv,            // TAU+1, before the floor
    float* __restrict__ rho_out,
    int n)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;

    // :3192  qv(k) = MAX(1.E-10, qv1d(k) + DT*qvten(k))
    const float qv_local = fmaxf(1.0e-10f, qv[idx]);
    rho_out[idx] = thompson_aa_density(pressure[idx], temperature[idx],
                                       qv_local);
}


// ---------------------------------------------------------------------------
// 2b. WORKING CLOUD REFRESH -- module_mp_thompson.F:3213-3221 and :3484-3488.
// ---------------------------------------------------------------------------
//
//     if ((qc1d(k) + qcten(k)*DT) .gt. R1) then
//        rc(k) = (qc1d(k) + qcten(k)*DT)*rho(k)
//        nc(k) = MAX(2., MIN((nc1d(k)+ncten(k)*DT)*rho(k), Nt_c_max))
//        L_qc(k) = .true.
//     else
//        rc(k) = R1 ;  nc(k) = 2. ;  L_qc(k) = .false.
//
// Same shape as the entry diagnosis but WITHOUT the lamc/D0c/D0r rediagnosis:
// this one is a plain clamp of the accumulated value.  It runs twice in WRF,
// once before condensation (:3213) and once after the droplet-evaporation
// block (:3484), with the same code; the rho differs (:3193 vs :3490).
//
// Nothing here writes state.  qc/nc are the read-only entry per-kg fields.
extern "C" __global__ void thompson_aa_working_cloud(
    const float* __restrict__ qc,        // state, per kilogram, READ-ONLY
    const float* __restrict__ qcten,     // accumulator, per kilogram per s
    const float* __restrict__ nc,        // state, per kilogram, READ-ONLY
    const float* __restrict__ ncten,     // accumulator, per kilogram per s
    const float* __restrict__ rho,
    float dt,
    float* __restrict__ rc_work,         // kg m^-3
    float* __restrict__ nc_work_m3,      // m^-3
    int* __restrict__ l_qc_out,
    int n)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;

    const float rho_local = rho[idx];
    const float qc_updated = thompson_aa_add(
        qc[idx], thompson_aa_mul(qcten[idx], dt));

    if (qc_updated > THOMPSON_AA_R1) {
        rc_work[idx] = thompson_aa_mul(qc_updated, rho_local);
        const float nc_updated = thompson_aa_add(
            nc[idx], thompson_aa_mul(ncten[idx], dt));
        nc_work_m3[idx] = thompson_aa_clamp_nc(
            thompson_aa_mul(nc_updated, rho_local));
        l_qc_out[idx] = 1;
    } else {
        rc_work[idx] = THOMPSON_AA_R1;
        nc_work_m3[idx] = THOMPSON_AA_NC_FLOOR;
        l_qc_out[idx] = 0;
    }
}


// ---------------------------------------------------------------------------
// 3.  TERMINAL APPLY AND CLAMP -- module_mp_thompson.F:3972-4021.
// ---------------------------------------------------------------------------
//
// THE single point at which the three accumulators reach state.  WRF's order
// is reproduced exactly:
//
//   (a) nc1d = MAX(2./rho, MIN(nc1d + ncten*DT, Nt_c_max))
//       -- per-kg value against the volumetric ceiling, see the unit note at
//          the top of this file.
//   (b) nwfa1d = MAX(11.1E6, MIN(9999.E6, nwfa1d + nwfaten*DT))
//       nifa1d = MAX(naIN1*0.01, MIN(9999.E6, nifa1d + nifaten*DT))
//       -- again per-kg values against per-m3 constants, and NOTE that unlike
//          the entry snapshot there is no *rho here at all.
//   (c) if qc1d <= R1 then qc1d = 0, nc1d = 0
//       else rediagnose nc1d through nu_c / lamc / D0c / 2*D0r.
//
// (c) uses the PER-KILOGRAM form
//       lamc = (am_r*ccg(2,nu_c)*ocg1(nu_c)*nc1d/qc1d)**obmr
// in which the densities cancel.  Do NOT algebraically simplify it to the
// entry form: the two differ in operand association
// (entry :1832 is nc*am_r*ccg2*ocg1/rc, terminal :4014 is am_r*ccg2*ocg1*nc/qc)
// and float32 is not associative.
//
// qc is mutable because WRF zeroes it on the (c) false branch.  Every output
// pointer MAY alias its corresponding input (each thread reads its own
// element before writing it), so the adapter can legally pass state.nc for
// both nc and nc_out.
//
// ---------------------------------------------------------------------------
// WHICH DENSITY.  IT IS NOT THE ENTRY DENSITY, AND THE DIFFERENCE IS MEASURED.
// ---------------------------------------------------------------------------
// `rho` MUST be WRF's TAU+1 density as the terminal loop finds it, NOT the
// :1802 entry density.  rho(k) is written in exactly four places in
// mp_thompson -- :1802 (entry), :3193 (the unconditional TAU+1 refresh before
// condensation), :3490 (inside the condensation block, per level) and :3572
// (inside the rain-evaporation block, per level) -- and nothing after :3574
// touches it, so what :3976, :4011, :4019 and :4020 read is whichever of the
// last three ran at that level.
//
// MEASURED, from a build of PRISTINE module_mp_thompson.F carrying only added
// `write` statements (it reproduces all 22 committed column fixtures BYTE FOR
// BYTE), over all 21 fixtures x 24 levels:
//     max |rho_terminal - rho_entry| / rho_terminal = 7.4672e-03
//         (aero-cold-overlap; also 4.1e-03 on aero-reduces-to-classic and
//          7.3e-04 on aero-cloud-freeze-nc)
// and, recomputing :3976-:4021 with the entry density substituted for the
// terminal one inside WRF itself, nc1d changes at exactly one of those 504
// levels -- aero-cold-overlap k=5, 1.833336115 -> 1.826136470 kg^-1, i.e.
//     3.9271e-03 relative, 1963x the 2.0e-06 end-to-end gate.
// The path is :3976's `2./rho(k)` floor; :4011's nu_c and :4020's
// Nt_c_max/rho(k) ceiling read the same rho and are integer-selector and
// ceiling respectively, so they are latent rather than quiet.
//
// HOW A CALLER GETS IT.  ArWen's `temperature` and `qv` arrays hold WRF's
// temp(k) / qv(k) exactly at ONE moment: immediately after the rain-
// evaporation launcher returns and before sedimentation, because WRF's
// :3569-3572 is the last statement that refreshes them and everything after
// :3574 accumulates into tten/qvten without writing temp/qv back.  MEASURED:
// launching thompson_aa_tau1_density there reproduces WRF's terminal rho
// BITWISE at 501 of those 504 levels, worst 1.2334e-07 (inherited float32
// noise on two fixtures), against 7.4672e-03 for the entry density.
// Recomputing it later -- e.g. at the finalize call itself -- is 5.39e-05
// off, because ArWen's temperature keeps absorbing the melt/freeze cleanup's
// tten while WRF's temp(k) snapshot does not.
extern "C" __global__ void thompson_aa_state_finalize(
    float* __restrict__ qc,              // final per-kg qc, zeroed if <= R1
    const float* __restrict__ nc,        // ENTRY per-kg nc, READ-ONLY
    const float* __restrict__ nwfa,      // ENTRY per-kg nwfa, READ-ONLY
    const float* __restrict__ nifa,      // ENTRY per-kg nifa, READ-ONLY
    const float* __restrict__ ncten,
    const float* __restrict__ nwfaten,
    const float* __restrict__ nifaten,
    // TAU+1 density as of :3972 -- :3193/:3490/:3572, whichever ran last at
    // this level.  NOT the :1802 entry density; see the block above.
    const float* __restrict__ rho,
    float dt,
    float* __restrict__ nc_out,
    float* __restrict__ nwfa_out,
    float* __restrict__ nifa_out,
    int n)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;

    const float rho_local = rho[idx];

    // (a) :3976
    float nc_new = fmaxf(
        THOMPSON_AA_NC_FLOOR / rho_local,
        fminf(thompson_aa_add(nc[idx], thompson_aa_mul(ncten[idx], dt)),
              THOMPSON_AA_NT_C_MAX));

    // (b) :3979-3982.  aer_init_opt < 2 branch; wif_input_opt = 0 so nbca is
    // identically zero and is not carried.
    nwfa_out[idx] = thompson_aa_clamp_nwfa(
        thompson_aa_add(nwfa[idx], thompson_aa_mul(nwfaten[idx], dt)));
    nifa_out[idx] = thompson_aa_clamp_nifa(
        thompson_aa_add(nifa[idx], thompson_aa_mul(nifaten[idx], dt)));

    // (c) :4008-4021
    const float qc_local = qc[idx];
    if (qc_local <= THOMPSON_AA_R1) {
        qc[idx] = 0.0f;
        nc_out[idx] = 0.0f;
        return;
    }

    const int nu_c = thompson_aa_nu_c(nc_new * rho_local);

    // -----------------------------------------------------------------------
    // EVERY float32 SUB-EXPRESSION BELOW THAT FEEDS A DOUBLE IS PINNED WITH
    // thompson_aa_mul / thompson_aa_div, AND THAT IS LOAD-BEARING.
    // -----------------------------------------------------------------------
    // MEASURED on an RTX 5090 with nvrtc from CUDA 12, options ("-std=c++17",)
    // -- exactly what gpuwm/core/kernels/__init__.py::load_module passes.  For
    // nu_c = 6, qc = 8.9721725e-06 and lamc = 2184250.25 the three spellings
    //     (double)(C1[u]*O2[u]*qc/AM_R) * pow(lamc, 3.0)
    //     float pref = C1[u]*O2[u]*qc/AM_R;  (double)pref * pow(lamc, 3.0)
    //     (double)__fdiv_rn(__fmul_rn(__fmul_rn(C1[u],O2[u]),qc),AM_R) * ...
    // give 3.5430368e+08, 3.5430368e+08 and 3.5430370e+08.  The first two are
    // bit-for-bit the value you get by evaluating the PREFACTOR IN DOUBLE:
    // nvrtc widens the float32 chain when its result is consumed by a double
    // expression, and a named `float` local does NOT stop it.  Only the
    // rounding intrinsics do.
    //
    // WRF has no such freedom.  :4019's ccg(1,nu_c), ocg2(nu_c), qc1d(k) and
    // am_r are all REAL(4), so the prefactor IS rounded to float32 before it
    // meets the DOUBLE lamc**bm_r; same at :4012 for the lambda base and at
    // :4015/:4017 where a REAL(4) quotient is assigned to the DOUBLE lamc.
    // Reproducing that is not pedantry: the terminal rediagnosis CUBES lamc,
    // so the skipped roundings surfaced as a 2.4e-07 to 4.2e-07 end-to-end
    // nc_per_kg residual on nearly every fixture -- 2 to 3.5 float32 ulps,
    // and 38 of 456 fixture states disagreed with a Fortran-faithful host
    // transcription.  With the pins the same comparison is BITWISE.

    // :4012  lamc = (am_r*ccg(2,nu_c)*ocg1(nu_c)*nc1d(k)/qc1d(k))**obmr
    //        REAL base to a REAL power, the REAL result widened to DOUBLE.
    //        thompson_aa_powf_cr, not CUDA's powf: gfortran lowers REAL(4) **
    //        to glibc powf, which is correctly rounded where libdevice's powf
    //        is not, and :4019 then CUBES this lambda.
    double lamc = (double)thompson_aa_powf_cr(
        thompson_aa_div(
            thompson_aa_mul(
                thompson_aa_mul(
                    thompson_aa_mul(THOMPSON_AA_AM_R,
                                    THOMPSON_AA_CCG2[nu_c]),
                    THOMPSON_AA_OCG1[nu_c]),
                nc_new),
            qc_local),
        THOMPSON_AA_OBMR);
    // :4013  REAL xDc from a DOUBLE division.  (bm_r + nu_c + 1.) is an exact
    // small integer, so only the division needs care.
    const float xDc = (float)((double)(THOMPSON_AA_BM_R + (float)nu_c + 1.0f)
                              / lamc);
    // :4015 / :4017  cce(2,nu_c)/D0c is a REAL(4) quotient ASSIGNED to the
    // DOUBLE lamc.  D0c = 1.E-6 and D0r*2. = 1.E-4 are not exact in binary, so
    // evaluating these in double instead of float32 shifts lamc: at nu_c = 6,
    // 10.0f/1.0e-6f is 10000000.0 in float32 and 10000000.025 in double.
    if (xDc < THOMPSON_AA_D0C) {
        lamc = (double)thompson_aa_div(THOMPSON_AA_CCE2[nu_c],
                                       THOMPSON_AA_D0C);
    } else if (xDc > THOMPSON_AA_D0R * 2.0f) {
        lamc = (double)thompson_aa_div(THOMPSON_AA_CCE2[nu_c],
                                       THOMPSON_AA_D0R * 2.0f);
    }
    // :4019-4020  MIN(<double>, DBLE(Nt_c_max)/rho(k)) -- here the ceiling IS
    // density-converted, unlike (a).  The prefactor is REAL(4).
    nc_new = (float)fmin(
        (double)thompson_aa_div(
            thompson_aa_mul(
                thompson_aa_mul(THOMPSON_AA_CCG1[nu_c],
                                THOMPSON_AA_OCG2[nu_c]),
                qc_local),
            THOMPSON_AA_AM_R)
            * pow(lamc, (double)THOMPSON_AA_BM_R),
        (double)THOMPSON_AA_NT_C_MAX / (double)rho_local);
    nc_out[idx] = nc_new;
}


// ---------------------------------------------------------------------------
// 4.  SURFACE AEROSOL EMISSION -- mp_gt_driver:1310-1327.
// ---------------------------------------------------------------------------
//
//     nwfa1d(kts) = nwfa1d(kts) + nwfa2d(i,j)*dt_in
//     nifa1d(kts) = nifa1d(kts) + nifa2d(i,j)*dt_in
//
// LOWEST MODEL LEVEL ONLY, and DELIBERATELY NOT CLAMPED.  This runs AFTER
// mp_thompson has returned, i.e. after thompson_aa_state_finalize has already
// applied its ceiling, so between this kernel and the next call's entry pack
// nwfa/nifa may legitimately exceed 9999.E6.  The ceiling reappears only at
// the next call's :1805-1806.
//
// Adding MIN(9999.E6, ...) here "for safety" would silently change the
// boundary-layer aerosol budget on EVERY step and would make the model
// diverge from WRF in exactly the regime the emission exists to represent.
// Do not add it.  Fixture aero-sfc-emit is what pins the arithmetic.
//
// nwfa2d / nifa2d are number tendencies, per kilogram per second (the comment
// at :1313-1315 says so explicitly; the field was redefined from a
// concentration to a tendency on 13 May 2013).
//
// Launched over (ny*nx) columns.  Field layout is C-contiguous (nz, ny, nx),
// so the lowest level of column (j,i) is element j*nx + i.
extern "C" __global__ void thompson_aa_surface_emission(
    float* __restrict__ nwfa,            // per kilogram, updated in place
    float* __restrict__ nifa,            // per kilogram, updated in place
    const float* __restrict__ nwfa2d,    // per kilogram per second
    const float* __restrict__ nifa2d,    // per kilogram per second
    float dt,
    int ncolumns)
{
    const int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (col >= ncolumns) return;

    nwfa[col] = thompson_aa_add(nwfa[col],
                                thompson_aa_mul(nwfa2d[col], dt));
    nifa[col] = thompson_aa_add(nifa[col],
                                thompson_aa_mul(nifa2d[col], dt));
}


// ---------------------------------------------------------------------------
// 5.  thompson_init's SYNTHETIC CCN / IN PROFILE -- thompson_init:493-551.
// ---------------------------------------------------------------------------
//
//     if (hgt(i,1,j) <= 1000.)      h_01 = 0.8
//     elseif (hgt(i,1,j) >= 2500.)  h_01 = 0.01
//     else                          h_01 = 0.8*cos(hgt(i,1,j)*0.001 - 1.0)
//     niCCN3 = -1.0*ALOG(naCCN1/naCCN0)/h_01
//     nwfa(i,1,j) = naCCN1 + naCCN0*exp(-((hgt(i,2,j)-hgt(i,1,j))/1000.)*niCCN3)
//     z1 = hgt(i,2,j)-hgt(i,1,j)
//     nwfa2d(i,j) = nwfa(i,1,j) * 0.000196 * (50./z1)
//     do k = 2, kte
//        nwfa(i,k,j) = naCCN1 + naCCN0*exp(-((hgt(i,k,j)-hgt(i,1,j))/1000.)*niCCN3)
//
// FOUR THINGS THAT ARE EASY TO GET WRONG AND ARE PINNED BY aero-init-profile:
//   * the k=1 level uses the LEVEL-2 height difference, not zero.  WRF is
//     deliberately not evaluating the profile at the surface.
//   * hgt is ABSOLUTE (above sea level) in the h_01 branch but the profile
//     itself uses the AGL difference hgt(k)-hgt(1).  Both, in one formula.
//   * nifa gets the identical shape with naIN0=1.5E6 / naIN1=0.5E6 but there
//     is NO 2-D flux: WRF never derives a nifa2d, and it stays exactly zero.
//   * nc is NEVER touched by thompson_init.  It stays 0 and is bootstrapped
//     by the first call's terminal rediagnosis.
//
// The CCN and IN fills are INDEPENDENT: WRF tests MAXVAL(nwfa) and MAXVAL(nifa)
// separately against eps=1.E-15, so a domain can get one and not the other.
// Those two domain-wide reductions are the launcher's job; this kernel takes
// the two decisions as flags.
//
// Launched over (ny*nx) columns; requires nz >= 2.
extern "C" __global__ void thompson_aa_init_profile(
    const float* __restrict__ hgt,       // (nz, ny, nx), w-level height ASL
    float* __restrict__ nwfa,
    float* __restrict__ nifa,
    float* __restrict__ nwfa2d,
    int fill_ccn,
    int fill_in,
    int nz, int ncolumns)
{
    const int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (col >= ncolumns) return;

    // CONTRACTION IS PINNED THROUGHOUT THIS KERNEL.  `naCCN1 + naCCN0*exp(x)`
    // is exactly the a*b+c shape nvrtc fuses into a single-rounded FMA, while
    // build_aero.sh compiles the oracle with plain `gfortran -O2` on baseline
    // x86-64, which has no FMA instruction and rounds the multiply and the
    // add separately.  MEASURED: with contraction left on, the nifa profile
    // over a 1500 m terrain column differs from the Fortran-equivalent host
    // transcription by 1 ulp (928957.75 vs 928957.625) at one level; with
    // every operation pinned the two agree exactly.  The same applies to
    // `hgt*0.001 - 1.0` in the h_01 branch.
    const float hgt1 = hgt[col];
    const float hgt2 = hgt[ncolumns + col];

    float h_01;
    if (hgt1 <= 1000.0f) {
        h_01 = 0.8f;
    } else if (hgt1 >= 2500.0f) {
        h_01 = 0.01f;
    } else {
        h_01 = thompson_aa_mul(
            0.8f,
            thompson_aa_cosf_cr(
                thompson_aa_sub(thompson_aa_mul(hgt1, 0.001f), 1.0f)));
    }

    const float z1 = thompson_aa_sub(hgt2, hgt1);

    if (fill_ccn != 0) {
        // naCCN1 = 50.0E6, naCCN0 = 300.0E6  (:96-97)
        const float niCCN3 = thompson_aa_div(
            thompson_aa_mul(
                -1.0f, thompson_aa_logf_cr(thompson_aa_div(50.0e6f,
                                                           300.0e6f))),
            h_01);
        const float first = thompson_aa_add(
            50.0e6f,
            thompson_aa_mul(
                300.0e6f,
                thompson_aa_expf_cr(thompson_aa_mul(
                    -thompson_aa_div(z1, 1000.0f), niCCN3))));
        nwfa[col] = first;
        nwfa2d[col] = thompson_aa_mul(
            thompson_aa_mul(first, 0.000196f),
            thompson_aa_div(50.0f, z1));
        for (int k = 1; k < nz; ++k) {
            const float dz = thompson_aa_sub(hgt[k * ncolumns + col], hgt1);
            nwfa[k * ncolumns + col] = thompson_aa_add(
                50.0e6f,
                thompson_aa_mul(
                    300.0e6f,
                    thompson_aa_expf_cr(thompson_aa_mul(
                        -thompson_aa_div(dz, 1000.0f), niCCN3))));
        }
    }

    if (fill_in != 0) {
        // naIN1 = 0.5E6, naIN0 = 1.5E6  (:94-95).  No 2-D counterpart.
        const float niIN3 = thompson_aa_div(
            thompson_aa_mul(
                -1.0f, thompson_aa_logf_cr(thompson_aa_div(0.5e6f, 1.5e6f))),
            h_01);
        nifa[col] = thompson_aa_add(
            0.5e6f,
            thompson_aa_mul(
                1.5e6f,
                thompson_aa_expf_cr(thompson_aa_mul(
                    -thompson_aa_div(z1, 1000.0f), niIN3))));
        for (int k = 1; k < nz; ++k) {
            const float dz = thompson_aa_sub(hgt[k * ncolumns + col], hgt1);
            nifa[k * ncolumns + col] = thompson_aa_add(
                0.5e6f,
                thompson_aa_mul(
                    1.5e6f,
                    thompson_aa_expf_cr(thompson_aa_mul(
                        -thompson_aa_div(dz, 1000.0f), niIN3))));
        }
    }
}


// ---------------------------------------------------------------------------
// 6.  EFFECTIVE RADIUS -- calc_effectRad:5594-5699.
// ---------------------------------------------------------------------------
//
// The cloud branch is the ONLY place in the whole scheme with a THREE-way
// shape selector (:5637-5643):
//     nc < 100.      -> inu_c = 15
//     nc > 1.E10     -> inu_c = 2       (dead: :5626 already capped nc)
//     otherwise      -> inu_c = MIN(15, NINT(1000.E6/nc) + 2)
// thompson_aa_nu_c is NOT a substitute; thompson_aa_inu_c_effrad is.
//
// It is also the only consumer of the EXACT-INTEGER g_ratio PARAMETER
// (:5611-5613) rather than the runtime ccg(2,n)*ocg1(n) product.  At nu_c=12
// those differ: 2730 exactly versus 2729.9973.  That is precisely why
// thompson.cu:343's hardcoded 2730.0f is CORRECT for mp=8's effective radius
// while its 2730.0f at :882/:999/:4005/:4128/:4680 is a 1e-6 deviation.
//
// OUTPUT CONVENTION: metres are converted to MICRONS as the very last
// operation, after mp_gt_driver's second clamp at :1476-1478, exactly as
// thompson.cu:373-381 does.  gpuwm's state contract for effc/effi/effs is
// microns; WRF's own driver does re*1.E6 on the way into radiation.
//
// ===========================================================================
// WHAT THIS PACKAGE MEASURED ABOUT calc_effectRad, AND WHERE THE FIX LIVES
// ===========================================================================
// The three branch helpers below come from thompson_aerosol_common.cuh.  When
// WP-04 first gated them they were verbatim transcriptions of
// thompson.cu:339-368 and the header said so ("thompson_field_a/
// thompson_field_b still keep thompson.cu's plain form; nothing has measured
// them").  Driving THIS kernel with the ORACLE's own post-step column -- so
// that nothing but calc_effectRad is under test and no upstream residual can
// be blamed -- measured two deviations from WRF v4.6.1:
//
//   1. THE sa/sb POLYNOMIAL ASSOCIATION.  :5684-5688 spells the 5th, 7th,
//      8th, 9th and 10th terms as sa(5)*tc0*tc0, sa(7)*tc0*tc0*cse(1),
//      sa(8)*tc0*cse(1)*cse(1), sa(9)*tc0*tc0*tc0, sa(10)*cse(1)**3, and
//      Fortran multiplication is LEFT-associative; thompson.cu's
//      thompson_field_a precomputes tc2 = tc*tc / moment2 = moment*moment and
//      forms sa[4]*tc2, sa[6]*(tc2*moment), sa[7]*(tc*moment2),
//      sa[8]*(tc2*tc), sa[9]*(moment2*moment).  float32 multiplication is not
//      associative, and :5689's a_ = 10.0**loga_ AMPLIFIES the difference by
//      ln(10).  MEASURED on oracle-aero/aero-scav-frozen's after-column
//      (T = 260 K exactly at every level, so the fixture's temperature
//      round-trip is exact and nothing else can be blamed): the old spelling
//      gives loga_ = -2.3101425 where gfortran -O2 gives -2.3101428, and
//      re_qs came out 4.0e-7 to 6.4e-7 HIGH at all 9 snowy levels.  On
//      aero-cold-overlap the same defect reached 2.305e-6 -- it BREACHED the
//      port's 2e-6 end-to-end gate.
//
//   2. CUDA powf.  gfortran lowers REAL(4) ** to glibc powf (<= ~0.5 ulp);
//      CUDA's powf carries several.  MEASURED across the 19 committed
//      after-columns: 7 cloud levels and 5 ice levels differed from the
//      oracle by exactly 1 ulp.  Independently: glibc powf and
//      thompson_aa_powf_cr (evaluate in double, round once) agree on 19983 of
//      20000 random cube-root arguments and 19987 of 20000 random snow
//      (smob, b) pairs, so the correctly-rounded helper is the right stand-in
//      for the oracle's libm and CUDA's powf is not.
//
// Both are now repaired in thompson_aerosol_common.cuh -- WP-02's file, fixed
// concurrently by the package that owns it and reached independently from the
// cold network's snow moments -- so this file does NOT carry a private copy.
// A private copy would be a second place for the same arithmetic to drift.
// What this file owns is the composition and the receipts:
// tests/test_thompson_aerosol_state_gpu.py gates all three branches BITWISE
// against all 19 committed after-columns and against probe-effectrad.csv.
//
// CONSEQUENCE FOR THE mp=8 IDENTITY: at nc = Nt_c the two kernels no longer
// agree BITWISE on every level.  On aero-reduces-to-classic effc and effs are
// still bitwise equal and effi differs at 2 of 24 levels by exactly 1 float32
// ulp (6.567e-08) -- and at those two levels mp=28 is bitwise equal to the
// WRF oracle while frozen mp=8 is not.  That is the same tie-break the shared
// header records for thompson_rslf/thompson_rsif: the authority is WRF, not
// ArWen's mp=8, and thompson.cu stays byte-frozen.

// -------------------------------------------------------------------------
// THE THREE BRANCHES COME FROM THE SHARED HEADER.  THEY USED NOT TO.
// -------------------------------------------------------------------------
// This file used to carry private copies named thompson_aa_state_eff_rad_
// cloud / _ice / _snow, on two grounds that were true when they were written
// and are no longer true:
//
// (1) THE SHARED ONES' float32 CHAINS WERE UNPINNED, AND nvrtc WIDENS SUCH
//     CHAINS WHEN IT FEELS LIKE IT.  Every branch ends in `(double)<float
//     expression>`.  MEASURED, in this very file: thompson_aa_state_finalize's
//     :4019 prefactor was spelled exactly that way and nvrtc evaluated it in
//     DOUBLE, which cost 2 to 3.5 float32 ulps of droplet number on nearly
//     every fixture and took 38 of 456 states off a Fortran-faithful host
//     transcription.  A named `float` local does not stop it; only __fmul_rn
//     / __fdiv_rn do.  WRF's operands at :5646, :5654, :5663 and :5694 are
//     all REAL(4), so every one of those roundings is part of the answer.
//     (Receipt: tests/test_thompson_aerosol_state_gpu.py::
//     test_state_finalize_rounds_every_real4_subexpression_that_feeds_a_double
//     compiles the three spellings side by side and measures the split.)
//     thompson_aerosol_common.cuh NOW PINS ALL THREE with the identical
//     __fmul_rn / __fdiv_rn / __fadd_rn spelling these copies used, and its
//     own header block carries the 960-state measurement showing the pins are
//     bit-for-bit inert on today's toolchain and load-bearing as a guarantee.
//
// (2) THE ICE BRANCH'S POWER WAS A LIVE DISAGREEMENT.  :5654 is
//     REAL(4)**REAL(4), which gfortran lowers to glibc powf.  MEASURED over
//     the 19 committed after-columns: with CUDA's powf the ice branch misses
//     the Fortran by exactly 1 float32 ulp at several levels; with
//     thompson_aa_powf_cr it is bitwise wherever the fixture's own
//     temperature is exact.  It also costs the mp=8 effective-radius
//     identity, and the shared header was flipped between the two spellings
//     twice while this package was measuring.  The header now pins
//     thompson_aa_powf_cr there for good, with MP28_PORT_SPEC.md's tie-break
//     -- the authority is WRF, not ArWen's mp=8 -- written next to it.
//
// So the two copies are gone.  Two implementations of one WRF formula is how
// a port acquires a silent split, and the source scan in
// tests/test_thompson_aerosol_device_helpers.py::
// test_shared_helpers_are_defined_exactly_once_and_only_in_the_header is what
// keeps them gone.  thompson_field_a / thompson_field_b were already shared
// for the same reason.

// calc_effectRad's whole body for one level, :5624-5695.  Both kernels below
// share it so the micron and metre gates cannot drift apart.
__device__ __forceinline__ void thompson_aa_calc_effect_rad(
    float t_k, float p_pa, float qv_kg, float qc_kg, float nc_kg,
    float qi_kg, float ni_kg, float qs_kg,
    float* __restrict__ reqc, float* __restrict__ reqi,
    float* __restrict__ reqs)
{
    // :5624-5634.  Every constant is single precision because the WRF
    // declarations are default REAL; only the two lambdas are DOUBLE.
    const float rho = thompson_aa_density(p_pa, t_k, qv_kg);
    const float rc = fmaxf(THOMPSON_AA_R1, qc_kg * rho);
    // :5626 + :5627.  is_aerosol_aware is TRUE for mp=28, so :5627's
    // `nc(k) = Nt_c` override does NOT run and the prognostic value stands.
    const float nc_m3 = thompson_aa_clamp_nc(nc_kg * rho);
    const float ri = fmaxf(THOMPSON_AA_R1, qi_kg * rho);
    const float ni_m3 = fmaxf(THOMPSON_AA_R2, ni_kg * rho);
    const float rs = fmaxf(THOMPSON_AA_R1, qs_kg * rho);

    // :5619-5621, the background values every level starts at.
    *reqc = THOMPSON_AA_RE_QC_BG;
    *reqi = THOMPSON_AA_RE_QI_BG;
    *reqs = THOMPSON_AA_RE_QS_BG;

    // :5636-5648 / :5651-5656 / :5659-5696.  WRF's column-wide has_qc /
    // has_qi / has_qs flags are exactly the disjunction of these per-level
    // tests, and every level that fails them CYCLEs, so a pointwise kernel
    // reproduces the column loop exactly.
    if (rc > THOMPSON_AA_R1 && nc_m3 > THOMPSON_AA_R2) {
        *reqc = thompson_aa_eff_rad_cloud(rc, nc_m3);
    }
    if (ri > THOMPSON_AA_R1 && ni_m3 > THOMPSON_AA_R2) {
        *reqi = thompson_aa_eff_rad_ice(ri, ni_m3);
    }
    if (rs > THOMPSON_AA_R1) {
        *reqs = thompson_aa_eff_rad_snow(rs, t_k);
    }
}

extern "C" __global__ void thompson_aa_effective_radius(
    const float* __restrict__ temperature,
    const float* __restrict__ pressure,
    const float* __restrict__ qv,
    const float* __restrict__ qc,
    const float* __restrict__ nc,        // per kilogram
    const float* __restrict__ qi,
    const float* __restrict__ ni,        // per kilogram
    const float* __restrict__ qs,
    float* __restrict__ effc,
    float* __restrict__ effi,
    float* __restrict__ effs,
    int n)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;

    float reqc, reqi, reqs;
    thompson_aa_calc_effect_rad(
        temperature[idx], pressure[idx], qv[idx], qc[idx], nc[idx],
        qi[idx], ni[idx], qs[idx], &reqc, &reqi, &reqs);

    // mp_gt_driver:1476-1478, then the metre->micron convention.
    effc[idx] = fmaxf(THOMPSON_AA_RE_QC_BG, fminf(reqc, 50.0e-6f)) * 1.0e6f;
    effi[idx] = fmaxf(THOMPSON_AA_RE_QI_BG, fminf(reqi, 125.0e-6f)) * 1.0e6f;
    effs[idx] = fmaxf(THOMPSON_AA_RE_QS_BG, fminf(reqs, 999.0e-6f)) * 1.0e6f;
}


// Metre-valued variant of the same computation, for gating directly against
// gpuwm/data/thompson/oracle-aero/probe-effectrad.csv, which is a call to
// calc_effectRad itself and therefore carries neither mp_gt_driver's second
// clamp nor gpuwm's micron convention.  Same arithmetic, different tail.
extern "C" __global__ void thompson_aa_effective_radius_metres(
    const float* __restrict__ temperature,
    const float* __restrict__ pressure,
    const float* __restrict__ qv,
    const float* __restrict__ qc,
    const float* __restrict__ nc,
    const float* __restrict__ qi,
    const float* __restrict__ ni,
    const float* __restrict__ qs,
    float* __restrict__ effc,
    float* __restrict__ effi,
    float* __restrict__ effs,
    int n)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;

    thompson_aa_calc_effect_rad(
        temperature[idx], pressure[idx], qv[idx], qc[idx], nc[idx],
        qi[idx], ni[idx], qs[idx], &effc[idx], &effi[idx], &effs[idx]);
}
