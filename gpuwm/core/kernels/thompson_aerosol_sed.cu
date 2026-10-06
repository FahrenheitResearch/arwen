// gpuwm/core/kernels/thompson_aerosol_sed.cu
//
// Aerosol-aware Thompson (mp_physics=28): number-weighted cloud-water
// sedimentation and the number-conserving final phase cleanup.
//
// Numerical authority is WRF v4.6.1 phys/module_mp_thompson.F
// (WRF v4.6.1, commit d66e442fccc04111067e29274c9f9eaccc3cef28, zero local
// modifications).  Every line number below refers to that file.
//
// gpuwm/core/kernels/thompson_aerosol_common.cuh is PREPENDED to this
// translation unit by gpuwm/core/kernels/__init__.py::_EXTRA_HEADERS.  Do not
// #include it and do not duplicate any helper it publishes.
//
// ---------------------------------------------------------------------------
// WHY THIS IS A SEPARATE TRANSLATION UNIT
// ---------------------------------------------------------------------------
// gpuwm/core/kernels/thompson.cu is byte-frozen: its compiled source string is
// the entire mp=8 numerics guarantee.  Its
// thompson_cloud_sediment_held_density_impl (thompson.cu:944-1042) is the
// structural template for the mass channel here, but mp=8 has NO cloud-number
// fallout at all, and it hardcodes
//     100.0e6f              (the constant Nt_c)
//     2730.0f == ccg(2,12)*ocg1(12)
//      272.0f == ccg(5,12)*ocg2(12)
// where mp=28 needs live per-cell nu_c-indexed gamma moments.  A shared
// implementation is impossible without editing thompson.cu, so the mass
// channel is transcribed here and mp=8 stays frozen.
//
// ---------------------------------------------------------------------------
// THE TWO THINGS THAT ARE EASY TO GET SILENTLY WRONG
// ---------------------------------------------------------------------------
// 1. NO SUBSTEPPING.  Rain (:3790-3820), ice (:3840-3870), snow (:3871-3902)
//    and graupel (:3903-3937) all wrap their apply loop in
//    `do n = 1, nstep` and scale every term by onstep(1..4).  CLOUD DOES NOT
//    (:3823-3838): one pass, no onstep factor, no k=kte export term and no
//    surface accumulation.  Copying a rain/ice launcher as a template and
//    keeping its substep loop is a silent rate error that no bounds check
//    would catch.
//
// 2. THE FLOOR IS 10, NOT 2.  :3835 is
//        nc(k) = MAX(10., nc(k) + (sed_n(k+1)-sed_n(k))*odzq*DT)
//    It is the ONLY use of 10 as a droplet-number floor anywhere in the
//    scheme; every other site floors at 2 (THOMPSON_AA_NC_FLOOR).
//
// A third, quieter trap: cloud mass and number leaving level kts are simply
// DISCARDED.  There is no `pptrain`-style accumulation for cloud, so a
// number-budget test must not expect closure.
//
// ---------------------------------------------------------------------------
// ACCUMULATOR CONTRACT
// ---------------------------------------------------------------------------
// state.nc is READ-ONLY entry state (nc1d) for the whole mp=28 call.  These
// kernels never write it.  They write the shared per-kilogram-per-second
// accumulator ncten, which a terminal kernel (WP-04) applies once with WRF's
// clamps at :3972-4021.  Any working per-m3 droplet number is recomputed
// locally the way WRF does at :3216/:3486:
//     nc = MAX(2, MIN((nc1d + ncten*DT)*rho, Nt_c_max))
// Cloud MASS is different: gpuwm applies mass tendencies in place, exactly as
// the frozen mp=8 pipeline does, so qc is read-modify-written here.
//
// ENTRY STATE IS NOT THE RAW STATE ARRAY.  :1844-1846 and :1870-1871 rewrite
// the caller's own column on the way in:
//     else                    ! qc1d(k) .le. R1        (and qi1d(k) .le. R1)
//        qc1d(k) = 0.0                                  qi1d(k) = 0.0
//        nc1d(k) = 0.0                                  ni1d(k) = 0.0
// so cloud_number_entry and ice_number_entry are state.nc / state.ni ZEROED
// wherever the matching condensate was absent at call entry.  Feeding the raw
// arrays instead produces a non-zero working droplet number in air that has no
// droplets -- bounded, finite, and wrong.
//
// ---------------------------------------------------------------------------
// FLOATING-POINT CONTRACTION
// ---------------------------------------------------------------------------
// nvrtc defaults to --fmad=true; the oracle is `gfortran -O2` on baseline
// x86-64 with no FMA instruction, so every REAL(4) multiply and add in WRF is
// separately rounded.  Every expression below that could contract into an FMA
// goes through thompson_aa_add/sub/mul/div.  Pure multiply/divide chains do
// not contract and are written plainly, but still respect Fortran's
// left-to-right association: WRF's
//     nc(k)*am_r*ccg(2,nu_c)*ocg1(nu_c)/rc(k)
// is (((nc*am_r)*ccg2)*ocg1)/rc -- four separately rounded operations, NOT
// mp=8's fused nc*am_r*2730/rc.
//
// MEASURED (tests/test_thompson_aerosol_sed_gpu.py): with this pinning the
// kernel reproduces WRF's ncten, qcten, vtck and vtnck BIT-EXACTLY on the
// aero-nc-sed, aero-reduces-to-classic, aero-nc-cap, aero-warm-overlap and
// aero-cold-overlap columns dumped from an instrumented build of the pristine
// source.
//
// ---------------------------------------------------------------------------
// THE NUMBER CHANNEL HAS NO mp=8 EVIDENCE BEHIND IT, SO IT CARRIES ITS OWN
// ---------------------------------------------------------------------------
// Classic Thompson has no cloud-water number flux at all, so nothing in the
// model-validated mp=8 trajectory constrains vtnck.  If this kernel silently
// reused the MASS fall speed for the number -- the obvious copy-paste -- every
// bound would still hold, the column budget would still close against its own
// fluxes, and the scheme would simply drift nc with no error visible anywhere.
// Three gates in the test module close that hole, none of them fixture-bound:
//   * vtck/vtnck must equal (nu_c+5)(nu_c+4)/((nu_c+2)(nu_c+1)) at every
//     reachable nu_c.  That is an identity, not a measurement: :673-684 with
//     bm_r = 3 and bv_c = 2 makes ccg(5,n)*ocg2(n) = WGAMMA(n+6)/WGAMMA(n+4)
//     and ccg(4,n)*ocg1(n) = WGAMMA(n+3)/WGAMMA(n+1).  A copied mass velocity
//     reads 1.0 against true values from 1.397 to 2.8.
//   * both channels must reproduce (F[k+1]-F[k])*odzq*orho per level, bit for
//     bit, which catches a stray onstep factor in one channel only.
//   * on a uniform slab a copied velocity leaves the mean droplet mass
//     bit-unchanged; the real kernel drops it by ~40% in one 30 s step.
// Both mutations -- swapping CCG4/OCG1 for CCG5/OCG2, and halving the number
// divergence -- were injected and confirmed to fail those gates.
//
// ---------------------------------------------------------------------------
// SHARED HELPERS ARE NOT DEFINED HERE
// ---------------------------------------------------------------------------
// thompson_aa_bound_ice_number used to be duplicated in this file with am_i
// spelled as `3.1415926536f*890.0f/6.0f` where thompson_aerosol_common.cuh
// uses THOMPSON_AA_AM_I.  The two constants are bit-identical in float32 and
// the whole bound is now measured against an independent NumPy transcription
// of :4029-4039, so deleting the copy moved nothing.  It is deleted because
// two definitions in two cupy.RawModule translation units are how the halves
// of a scheme drift apart: nvrtc never diffs them.

#define THOMPSON_AA_KMAX_SHALLOW 64
#define THOMPSON_AA_KMAX_GENERIC 256

//! module_mp_thompson.F:3835.  The single site in the scheme that floors the
//! droplet number at 10 m^-3 instead of THOMPSON_AA_NC_FLOOR (2 m^-3).
#define THOMPSON_AA_NC_SED_FLOOR 10.0f

//! :3656.  Cloud fallout is suppressed in rising air.
#define THOMPSON_AA_SED_W_LIMIT 1.0e-1f

//! :3650.  Fallout depth search stops at 500 m above ground.
#define THOMPSON_AA_SED_HGT_AGL 500.0f


// ---------------------------------------------------------------------------
// Cloud-water sedimentation with the number channel, :3644-3666 + :3823-3838.
// ---------------------------------------------------------------------------
//
// Structural template: thompson.cu:944-1042
// (thompson_cloud_sediment_held_density_impl).  The gating is kept VERBATIM
// from mp=8, because WRF's is identical for both options:
//   * the ksed1(5) fallout-depth search is keyed on cloud MASS rc > R2 and
//     still breaks at 500 m AGL (:3646-3652).  Cloud NUMBER does not extend
//     the fallout depth.
//   * the per-level velocity gate is still rc > R1 .AND. w1d(k) < 1.E-1
//     (:3656).
//
// DENSITY RULE (two distinct densities, and using one for both is an
// invisible error):
//   * reference_density is WRF's HELD pre-adjustment rho -- the value rc and
//     nc were both formed on at :3216/:3486, before rho is refreshed at
//     :3489.  cloud_mass and cloud_number are built on it.
//   * density[k], recomputed here from the current temperature/pressure/qv,
//     is WRF's refreshed rho(k), and it is what converts the flux divergence
//     back into a tendency at :3831-3833.
//   * the rhof fall-speed factor (:3194, :3506, :3614) stays on the held
//     density UNLESS a post-source rain column caused the rain fall-speed
//     pass to refresh rhof for every level first; rain_active_columns carries
//     WRF's ANY(L_qr) for that.  This is mp=8's model, unchanged.
//
// Optional diagnostic outputs (any may be null) expose the exact intermediate
// columns the oracle comparison pins: vtck, vtnck, rc and nc.
template <int KMAX>
__device__ __forceinline__ void thompson_aa_cloud_sediment_impl(
    float* __restrict__ qc,
    const float* __restrict__ cloud_number_entry,
    float* __restrict__ cloud_number_tendency,
    const float* __restrict__ temperature,
    const float* __restrict__ pressure,
    const float* __restrict__ qv,
    const float* __restrict__ reference_density,
    const float* __restrict__ rain_active_columns,
    const float* __restrict__ cloud_active_columns,
    const float* __restrict__ vertical_velocity,
    const float* __restrict__ dz,
    float* __restrict__ out_mass_velocity,
    float* __restrict__ out_number_velocity,
    float* __restrict__ out_cloud_mass,
    float* __restrict__ out_cloud_number,
    float dt, int nz, int ny, int nx)
{
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= ny * nx) return;
    if (cloud_active_columns != nullptr
            && cloud_active_columns[column] == 0.0f) return;
    const int j = column / nx;
    const int i = column - j * nx;

#ifndef THOMPSON_NO_EXACT_SHORTCUTS
    // Diagnostic entry points retain the full working-column outputs.
    bool empty = out_mass_velocity == nullptr && out_number_velocity == nullptr
        && out_cloud_mass == nullptr && out_cloud_number == nullptr
        && dt >= 0.0f && dt <= 100000.0f;
    for (int k = 0; empty && k < nz; ++k) {
        const size_t idx = IDX3(k, j, i);
        empty = qc[idx] >= -1.0f && qc[idx] <= THOMPSON_AA_R1
            && isfinite(cloud_number_entry[idx])
            && isfinite(cloud_number_tendency[idx])
            && reference_density[idx] >= 0.00001f
            && reference_density[idx] <= 100.0f
            && temperature[idx] >= 100.0f && temperature[idx] <= 400.0f
            && pressure[idx] >= 1.0f && pressure[idx] <= 120000.0f
            && qv[idx] >= -1.0f && qv[idx] <= 1.0f
            && dz[idx] >= 1.0f && dz[idx] <= 100000.0f;
    }
    if (empty) {
        // Only the bottom level receives the zero divergence at sediment_top=0.
        const size_t bottom = IDX3(0, j, i);
        cloud_number_tendency[bottom] = thompson_aa_add(
            cloud_number_tendency[bottom], 0.0f);
        for (int k = 0; k < nz; ++k) {
            const size_t idx = IDX3(k, j, i);
            qc[idx] = thompson_aa_add(qc[idx], thompson_aa_mul(0.0f, dt));
        }
        return;
    }
#endif  // THOMPSON_NO_EXACT_SHORTCUTS

    float density[KMAX];
    float cloud_mass[KMAX];
    float cloud_number[KMAX];
    float mass_velocity[KMAX];
    float number_velocity[KMAX];
    float mass_flux[KMAX];
    float number_flux[KMAX];
    float qc_tendency[KMAX];
    float qc_initial[KMAX];

    // module_mp_thompson.F:192, rho_not = 101325.0/(287.05*298.0).  Written
    // as the same runtime expression thompson.cu:970 uses.
    const float rho_not = 101325.0f / (287.05f * 298.0f);
    const bool rain_refreshes_rhof = rain_active_columns != nullptr
        && rain_active_columns[column] != 0.0f;

    for (int k = 0; k < nz; ++k) {
        const size_t idx = IDX3(k, j, i);
        const float qvk = fmaxf(1.0e-10f, qv[idx]);
        density[k] = 0.622f * pressure[idx]
            / (287.04f * temperature[idx] * (qvk + 0.622f));
        qc_initial[k] = qc[idx];
        qc_tendency[k] = 0.0f;
        const float held = reference_density[idx];
        // :3215-3216 / :3484 rc(k) = MAX(R1, (qc1d(k) + qcten(k)*DT)*rho(k))
        cloud_mass[k] = qc[idx] > THOMPSON_AA_R1
            ? qc[idx] * held : THOMPSON_AA_R1;
        // :3217 / :3486 nc(k) = MAX(2., MIN((nc1d(k)+ncten(k)*DT)*rho(k),
        //                                   Nt_c_max))
        //
        // The clamp form is applied UNCONDITIONALLY, unlike the mass, even
        // though :3222 assigns a bare 2.0 on the rc <= R1 side.  WRF's own
        // droplet number at this point is whichever of :3222 and :3486 ran
        // last for that level, and the difference is provably dead: sed_n is
        // vtnck*nc, and vtnck is left at zero for every level with
        // rc <= R1 (:3656).  Reproducing the clamp everywhere removes a
        // divergent branch and MEASURES equal to WRF on every level of every
        // aerosol fixture, including the levels the saturation adjustment
        // evaporated down to R1 with a cancellation residue in
        // nc1d + ncten*DT.
        cloud_number[k] = thompson_aa_clamp_nc(
            thompson_aa_mul(
                thompson_aa_add(
                    cloud_number_entry[idx],
                    thompson_aa_mul(cloud_number_tendency[idx], dt)),
                held));
        mass_velocity[k] = 0.0f;
        number_velocity[k] = 0.0f;
    }

    // :3646-3652.  ksed1(:) = 1 at :3598, i.e. sediment_top starts at kts.
    int sediment_top = 0;
    float height_agl = 0.0f;
    for (int k = 0; k < nz - 1; ++k) {
        const size_t idx = IDX3(k, j, i);
        if (cloud_mass[k] > THOMPSON_AA_R2) sediment_top = k;
        height_agl += dz[idx];
        if (height_agl > THOMPSON_AA_SED_HGT_AGL) break;
    }

    // :3654-3665.
    for (int k = sediment_top; k >= 0; --k) {
        const size_t idx = IDX3(k, j, i);
        if (cloud_mass[k] > THOMPSON_AA_R1
                && vertical_velocity[idx] < THOMPSON_AA_SED_W_LIMIT) {
            // nu_c = MIN(15, NINT(1000.E6/nc(k)) + 2)
            const int nu_c = thompson_aa_nu_c(cloud_number[k]);
            // lamc = (nc(k)*am_r*ccg(2,nu_c)*ocg1(nu_c)/rc(k))**obmr
            const float lambda_arg = thompson_aa_div(
                thompson_aa_mul(
                    thompson_aa_mul(
                        thompson_aa_mul(cloud_number[k], THOMPSON_AA_AM_R),
                        THOMPSON_AA_CCG2[nu_c]),
                    THOMPSON_AA_OCG1[nu_c]),
                cloud_mass[k]);
            // lamc and ilamc are DOUBLE PRECISION in WRF (:1597-1598); the
            // power itself is a REAL**REAL, i.e. a single-precision powf,
            // widened afterwards.  thompson_aa_powf_cr, not CUDA's powf:
            // gfortran lowers REAL**REAL to glibc's correctly-rounded powf
            // while CUDA's carries up to ~2 ulp, and MEASURED over the
            // nu_c = 3..15 ladder that costs up to 2.1e-7 relative on vtck
            // and 1.2e-5 on the resulting ncten.  With the correctly-rounded
            // form every level of every fixture is bit-exact.  (thompson.cu
            // uses the plain powf here; it is frozen, and this is one of the
            // places mp=28 is simply closer to WRF than mp=8 is.)
            const double lambda =
                (double)thompson_aa_powf_cr(lambda_arg, THOMPSON_AA_OBMR);
            const double inverse_lambda = 1.0 / lambda;

            const float velocity_density = rain_refreshes_rhof
                ? density[k] : reference_density[idx];
            const float rhof = sqrtf(rho_not / velocity_density);

            // MASS: vtc = rhof(k)*av_c*ccg(5,nu_c)*ocg2(nu_c) * ilamc**bv_c
            const float mass_prefix = thompson_aa_mul(
                thompson_aa_mul(
                    thompson_aa_mul(rhof, THOMPSON_AA_AV_C),
                    THOMPSON_AA_CCG5[nu_c]),
                THOMPSON_AA_OCG2[nu_c]);
            // NUMBER (:3663, no mp=8 counterpart):
            //     vtc = rhof(k)*av_c*ccg(4,nu_c)*ocg1(nu_c) * ilamc**bv_c
            const float number_prefix = thompson_aa_mul(
                thompson_aa_mul(
                    thompson_aa_mul(rhof, THOMPSON_AA_AV_C),
                    THOMPSON_AA_CCG4[nu_c]),
                THOMPSON_AA_OCG1[nu_c]);
            // bv_c is exactly 2.0 (:164), and gfortran -O2 expands the
            // DOUBLE**REAL(2.0) to one exact multiply pair.
            mass_velocity[k] = (float)((double)mass_prefix
                * inverse_lambda * inverse_lambda);
            number_velocity[k] = (float)((double)number_prefix
                * inverse_lambda * inverse_lambda);
        }
    }

    // :3825-3828.  sed_c/sed_n are filled over the whole column, not just the
    // fallout depth, because the apply loop reads index k+1.
    for (int k = nz - 1; k >= 0; --k) {
        mass_flux[k] = mass_velocity[k] * cloud_mass[k];
        number_flux[k] = number_velocity[k] * cloud_number[k];
    }

    if (out_mass_velocity != nullptr || out_number_velocity != nullptr
            || out_cloud_mass != nullptr) {
        for (int k = 0; k < nz; ++k) {
            const size_t idx = IDX3(k, j, i);
            if (out_mass_velocity != nullptr) {
                out_mass_velocity[idx] = mass_velocity[k];
            }
            if (out_number_velocity != nullptr) {
                out_number_velocity[idx] = number_velocity[k];
            }
            if (out_cloud_mass != nullptr) out_cloud_mass[idx] = cloud_mass[k];
        }
    }

    // :3829-3836.  SINGLE PASS.  No nstep loop, no onstep factor, no k=kte
    // export term, no surface accumulation.
    for (int k = sediment_top; k >= 0; --k) {
        const size_t idx = IDX3(k, j, i);
        const float odzq = 1.0f / dz[idx];
        const float orho = 1.0f / density[k];
        const float mass_divergence = mass_flux[k + 1] - mass_flux[k];
        const float number_divergence = number_flux[k + 1] - number_flux[k];
        qc_tendency[k] = thompson_aa_add(
            qc_tendency[k],
            thompson_aa_mul(thompson_aa_mul(mass_divergence, odzq), orho));
        cloud_number_tendency[idx] = thompson_aa_add(
            cloud_number_tendency[idx],
            thompson_aa_mul(thompson_aa_mul(number_divergence, odzq), orho));
        cloud_mass[k] = fmaxf(
            THOMPSON_AA_R1,
            thompson_aa_add(
                cloud_mass[k],
                thompson_aa_mul(thompson_aa_mul(mass_divergence, odzq), dt)));
        cloud_number[k] = fmaxf(
            THOMPSON_AA_NC_SED_FLOOR,
            thompson_aa_add(
                cloud_number[k],
                thompson_aa_mul(thompson_aa_mul(number_divergence, odzq),
                                dt)));
    }

    for (int k = 0; k < nz; ++k) {
        const size_t idx = IDX3(k, j, i);
        // Cloud at or below R1 is carried to the phase cleanup, which
        // freezes any positive cloud below HGFR and adds melted ice to it
        // above 0 C before it removes what is left at or below R1, as WRF's
        // :3943-3966 and terminal :4007-4009 do.  Removing it here took it
        // out of both: cloud that WRF keeps at 1.6e-12 to 2.0e-12 kg/kg,
        // with its droplets, came back as zero on saved real-data columns
        // (tools/thompson_real_column_parity).
        qc[idx] = thompson_aa_add(
            qc_initial[k], thompson_aa_mul(qc_tendency[k], dt));
        if (out_cloud_number != nullptr) out_cloud_number[idx] = cloud_number[k];
    }
}


#define THOMPSON_AA_CLOUD_SEDIMENT_PARAMETERS                             \
    float* __restrict__ qc,                                               \
    const float* __restrict__ cloud_number_entry,                         \
    float* __restrict__ cloud_number_tendency,                            \
    const float* __restrict__ temperature,                                \
    const float* __restrict__ pressure,                                   \
    const float* __restrict__ qv,                                         \
    const float* __restrict__ reference_density,                          \
    const float* __restrict__ vertical_velocity,                          \
    const float* __restrict__ dz,                                         \
    float dt, int nz, int ny, int nx

#define THOMPSON_AA_CLOUD_SEDIMENT_ARGUMENTS                              \
    qc, cloud_number_entry, cloud_number_tendency, temperature, pressure, \
    qv, reference_density, (const float*)0, (const float*)0,              \
    vertical_velocity, dz, (float*)0, (float*)0, (float*)0, (float*)0,    \
    dt, nz, ny, nx

extern "C" __global__ void thompson_aa_cloud_sediment_64(
    THOMPSON_AA_CLOUD_SEDIMENT_PARAMETERS)
{
    thompson_aa_cloud_sediment_impl<THOMPSON_AA_KMAX_SHALLOW>(
        THOMPSON_AA_CLOUD_SEDIMENT_ARGUMENTS);
}

extern "C" __global__ void thompson_aa_cloud_sediment_256(
    THOMPSON_AA_CLOUD_SEDIMENT_PARAMETERS)
{
    thompson_aa_cloud_sediment_impl<THOMPSON_AA_KMAX_GENERIC>(
        THOMPSON_AA_CLOUD_SEDIMENT_ARGUMENTS);
}


#define THOMPSON_AA_CLOUD_SEDIMENT_RAIN_PARAMETERS                        \
    float* __restrict__ qc,                                               \
    const float* __restrict__ cloud_number_entry,                         \
    float* __restrict__ cloud_number_tendency,                            \
    const float* __restrict__ temperature,                                \
    const float* __restrict__ pressure,                                   \
    const float* __restrict__ qv,                                         \
    const float* __restrict__ reference_density,                          \
    const float* __restrict__ rain_active_columns,                        \
    const float* __restrict__ vertical_velocity,                          \
    const float* __restrict__ dz,                                         \
    float dt, int nz, int ny, int nx

#define THOMPSON_AA_CLOUD_SEDIMENT_RAIN_ARGUMENTS                         \
    qc, cloud_number_entry, cloud_number_tendency, temperature, pressure, \
    qv, reference_density, rain_active_columns, (const float*)0,          \
    vertical_velocity, dz, (float*)0, (float*)0, (float*)0, (float*)0,    \
    dt, nz, ny, nx

extern "C" __global__ void thompson_aa_cloud_sediment_64_with_rain(
    THOMPSON_AA_CLOUD_SEDIMENT_RAIN_PARAMETERS)
{
    thompson_aa_cloud_sediment_impl<THOMPSON_AA_KMAX_SHALLOW>(
        THOMPSON_AA_CLOUD_SEDIMENT_RAIN_ARGUMENTS);
}

extern "C" __global__ void thompson_aa_cloud_sediment_256_with_rain(
    THOMPSON_AA_CLOUD_SEDIMENT_RAIN_PARAMETERS)
{
    thompson_aa_cloud_sediment_impl<THOMPSON_AA_KMAX_GENERIC>(
        THOMPSON_AA_CLOUD_SEDIMENT_RAIN_ARGUMENTS);
}


#define THOMPSON_AA_CLOUD_SEDIMENT_MASKS_PARAMETERS                       \
    float* __restrict__ qc,                                               \
    const float* __restrict__ cloud_number_entry,                         \
    float* __restrict__ cloud_number_tendency,                            \
    const float* __restrict__ temperature,                                \
    const float* __restrict__ pressure,                                   \
    const float* __restrict__ qv,                                         \
    const float* __restrict__ reference_density,                          \
    const float* __restrict__ rain_active_columns,                        \
    const float* __restrict__ cloud_active_columns,                       \
    const float* __restrict__ vertical_velocity,                          \
    const float* __restrict__ dz,                                         \
    float dt, int nz, int ny, int nx

#define THOMPSON_AA_CLOUD_SEDIMENT_MASKS_ARGUMENTS                        \
    qc, cloud_number_entry, cloud_number_tendency, temperature, pressure, \
    qv, reference_density, rain_active_columns, cloud_active_columns,     \
    vertical_velocity, dz, (float*)0, (float*)0, (float*)0, (float*)0,    \
    dt, nz, ny, nx

extern "C" __global__ void thompson_aa_cloud_sediment_64_with_masks(
    THOMPSON_AA_CLOUD_SEDIMENT_MASKS_PARAMETERS)
{
    thompson_aa_cloud_sediment_impl<THOMPSON_AA_KMAX_SHALLOW>(
        THOMPSON_AA_CLOUD_SEDIMENT_MASKS_ARGUMENTS);
}

extern "C" __global__ void thompson_aa_cloud_sediment_256_with_masks(
    THOMPSON_AA_CLOUD_SEDIMENT_MASKS_PARAMETERS)
{
    thompson_aa_cloud_sediment_impl<THOMPSON_AA_KMAX_GENERIC>(
        THOMPSON_AA_CLOUD_SEDIMENT_MASKS_ARGUMENTS);
}


// Diagnostic entry point.  Same physics, same code path, but it also writes
// vtck, vtnck, the working rc and the post-fallout nc so the oracle test can
// pin WRF's intermediate columns instead of only the endpoints.  It is not
// used by the forecast adapter.
#define THOMPSON_AA_CLOUD_SEDIMENT_DIAG_PARAMETERS                        \
    float* __restrict__ qc,                                               \
    const float* __restrict__ cloud_number_entry,                         \
    float* __restrict__ cloud_number_tendency,                            \
    const float* __restrict__ temperature,                                \
    const float* __restrict__ pressure,                                   \
    const float* __restrict__ qv,                                         \
    const float* __restrict__ reference_density,                          \
    const float* __restrict__ rain_active_columns,                        \
    const float* __restrict__ cloud_active_columns,                       \
    const float* __restrict__ vertical_velocity,                          \
    const float* __restrict__ dz,                                         \
    float* __restrict__ out_mass_velocity,                                \
    float* __restrict__ out_number_velocity,                              \
    float* __restrict__ out_cloud_mass,                                   \
    float* __restrict__ out_cloud_number,                                 \
    float dt, int nz, int ny, int nx

#define THOMPSON_AA_CLOUD_SEDIMENT_DIAG_ARGUMENTS                         \
    qc, cloud_number_entry, cloud_number_tendency, temperature, pressure, \
    qv, reference_density, rain_active_columns, cloud_active_columns,     \
    vertical_velocity, dz, out_mass_velocity, out_number_velocity,        \
    out_cloud_mass, out_cloud_number, dt, nz, ny, nx

extern "C" __global__ void thompson_aa_cloud_sediment_64_diagnostic(
    THOMPSON_AA_CLOUD_SEDIMENT_DIAG_PARAMETERS)
{
    thompson_aa_cloud_sediment_impl<THOMPSON_AA_KMAX_SHALLOW>(
        THOMPSON_AA_CLOUD_SEDIMENT_DIAG_ARGUMENTS);
}

extern "C" __global__ void thompson_aa_cloud_sediment_256_diagnostic(
    THOMPSON_AA_CLOUD_SEDIMENT_DIAG_PARAMETERS)
{
    thompson_aa_cloud_sediment_impl<THOMPSON_AA_KMAX_GENERIC>(
        THOMPSON_AA_CLOUD_SEDIMENT_DIAG_ARGUMENTS);
}


// ---------------------------------------------------------------------------
// Terminal ice-number size bound, :4029-4039.
// ---------------------------------------------------------------------------
//
// NOT DEFINED HERE.  thompson_aa_bound_ice_number lives in
// thompson_aerosol_common.cuh and is prepended to this translation unit; see
// its PUBLISHED SHARED SIGNATURES block.  This file used to carry a local
// copy whose only textual difference was spelling am_i as the product
// `3.1415926536f*890.0f/6.0f` where the header uses THOMPSON_AA_AM_I
// (4.660029297e+02f).  The two constants are BIT-IDENTICAL in float32
// (test_am_i_product_form_is_bit_identical_to_the_header_constant proves it,
// and test_bound_ice_number_matches_an_independent_wrf_transcription proves
// the whole bound value-by-value against a NumPy transcription of :4029-4039
// that never touches the CUDA source), so removing the copy moved nothing.
//
// It is deleted because a second definition is exactly how the two halves of
// a scheme drift apart: separate cupy.RawModule translation units mean nvrtc
// never diffs them.  A re-added local copy is now a hard nvrtc redefinition
// error, and test_sed_defines_no_published_shared_helper catches a renamed
// one.
//
// ---------------------------------------------------------------------------
// Number-conserving final phase cleanup, :3943-3966.
// ---------------------------------------------------------------------------
//
// Structural template: thompson.cu:3745-3790 (thompson_final_phase_cleanup).
// The two instantaneous phase transfers run after every fallout tendency and
// before the terminal category bounds.  mp=8 exposes only the carried ICE
// number; mp=28 exposes the DROPLET number on both sides:
//
//   MELT   (temp > T_0 = 273.15, xri > 0), :3947-3953
//     qcten(k) = qcten(k) + xri*odt
//     ncten(k) = ncten(k) + ni1d(k)*odt      <-- the melted ice number
//                                                becomes DROPLET number.
//     qiten(k) = qiten(k) - xri*odt
//     niten(k) = -ni1d(k)*odt                <-- assignment, so the final ni
//                                                is exactly zero.
//   NOTE THE ARGUMENT: WRF credits ncten with ni1d(k), the ENTRY ice number,
//   NOT the current ni.  gpuwm applies ice-number tendencies in place, so the
//   entry value has to be carried in explicitly; passing the live ni here
//   would be a plausible-looking, silently wrong droplet source.
//
//   FREEZE (temp < HGFR = 235.16, xrc > 0), :3956-3965
//     xnc = nc1d(k) + ncten(k)*DT            <-- the TRUE running per-kg
//                                                droplet number: unclamped,
//                                                not multiplied by rho, and
//                                                read AFTER the melt branch.
//     niten(k) = niten(k) + xnc*odt
//     ncten(k) = ncten(k) - xnc*odt
//
// FLAGGED FOR THE RECORD, DELIBERATELY NOT FIXED: thompson.cu:3780 uses
// 100.0e6f/rho for that xnc, ignoring the accumulated ncten -- which is NOT
// zero even in mp=8, since mp=8 accumulates all five droplet sinks plus the
// balance limiter plus sedimentation into it and only discards it at
// mp_gt_driver's writeback.  That is a real pre-existing mp=8 deviation.
// Correcting thompson.cu would move a model-validated trajectory and confound
// the mp=28 gate, so it is recorded, not repaired.
//
// The two branches are written as WRF writes them -- two sequential IFs, not
// an IF/ELSE -- even though T_0 > HGFR makes them mutually exclusive, because
// the freeze branch deliberately reads the qc and ncten the melt branch just
// wrote.
extern "C" __global__ void thompson_aa_final_phase_cleanup(
    float* __restrict__ qc,
    float* __restrict__ qi,
    float* __restrict__ ni,
    float* __restrict__ temperature,
    const float* __restrict__ cloud_number_entry,
    const float* __restrict__ ice_number_entry,
    float* __restrict__ cloud_number_tendency,
    const float* __restrict__ pressure,
    const float* __restrict__ qv,
    float dt, int size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    const float temp0 = temperature[idx];
    const float qv0 = fmaxf(1.0e-10f, qv[idx]);
    const float rho = 0.622f * pressure[idx]
        / (287.04f * temp0 * (qv0 + 0.622f));
    const float inverse_cp = 1.0f / (1004.0f * (1.0f + 0.887f * qv0));
    const float odt = 1.0f / dt;

    if (temp0 > 273.15f && qi[idx] > 0.0f) {
        const float transferred = fmaxf(0.0f, qi[idx]);
        qc[idx] += transferred;
        cloud_number_tendency[idx] = thompson_aa_add(
            cloud_number_tendency[idx],
            thompson_aa_mul(ice_number_entry[idx], odt));
        qi[idx] = 0.0f;
        ni[idx] = 0.0f;
        temperature[idx] -= 334000.0f * inverse_cp * transferred;
    }

    if (temp0 < 235.16f && qc[idx] > 0.0f) {
        const float transferred = fmaxf(0.0f, qc[idx]);
        // lfus2 = lsub - lvap(k); lvap(k) = lvap0 + (2106.0 - 4218.0)*tempc.
        const float latent_vapor = 2.5e6f
            + (2106.0f - 4218.0f) * (temp0 - 273.15f);
        const float latent_fusion = 2.834e6f - latent_vapor;
        const float xnc = thompson_aa_add(
            cloud_number_entry[idx],
            thompson_aa_mul(cloud_number_tendency[idx], dt));
        qc[idx] = 0.0f;
        qi[idx] += transferred;
        ni[idx] += xnc;
        cloud_number_tendency[idx] = thompson_aa_sub(
            cloud_number_tendency[idx], thompson_aa_mul(xnc, odt));
        temperature[idx] += latent_fusion * inverse_cp * transferred;
    }

    // :3990-3991 and :4025-4039, kept in mp=8's fused position.  Both are
    // idempotent, so WP-04's terminal state kernel may repeat them.  Droplet
    // number is deliberately absent: nc is entry state and is only ever
    // written by that terminal kernel.
    if (qc[idx] <= THOMPSON_AA_R1) qc[idx] = 0.0f;
    if (qi[idx] <= THOMPSON_AA_R1) {
        qi[idx] = 0.0f;
        ni[idx] = 0.0f;
    } else if (qi[idx] * rho > THOMPSON_AA_R1) {
#if defined(THOMPSON_AA_WRF39)
        // fork :3733, 499.D3 (audit T8).
        thompson_aa_wrf39_bound_ice_number(qi[idx] * rho, rho, &ni[idx]);
#else
        thompson_aa_bound_ice_number(qi[idx] * rho, rho, &ni[idx]);
#endif
    } else {
        // :4025-4039 tests the MIXING RATIO and keeps both mass and number.
        // The shared bound tests the concentration (right for the source
        // stage at :3036-3055, wrong here) and zeroed the number of ice that
        // sediments into thin air aloft while keeping its mass: 12 to 72
        // levels of every saved 19,600-column real-data frame.  WRF's
        // per-kilogram form:
        const float am_i = THOMPSON_AA_AM_I;
        const float qi_local = qi[idx];
        const float ni_local = fmaxf(thompson_aa_div(THOMPSON_AA_R2, rho),
                                     ni[idx]);
        double lami = (double)thompson_aa_powf_cr(
            thompson_aa_div(thompson_aa_mul(thompson_aa_mul(am_i, 6.0f),
                                            ni_local), qi_local),
            1.0f / 3.0f);
        const float xdi = (float)(4.0 * (1.0 / lami));
        if (xdi < 5.0e-6f) {
            lami = (double)thompson_aa_div(4.0f, 5.0e-6f);
        } else if (xdi > 300.0e-6f) {
            lami = (double)thompson_aa_div(4.0f, 300.0e-6f);
        }
#if defined(THOMPSON_AA_WRF39)
        // fork :3733, 499.D3/rho (audit T8).
        ni[idx] = (float)fmin(
            (double)thompson_aa_div(thompson_aa_mul(1.0f / 6.0f, qi_local),
                                    am_i) * pow(lami, 3.0),
            (double)THOMPSON_AA_WRF39_NI_MAX / (double)rho);
#else
        ni[idx] = (float)fmin(
            (double)thompson_aa_div(thompson_aa_mul(1.0f / 6.0f, qi_local),
                                    am_i) * pow(lami, 3.0),
            999.0e3 / (double)rho);
#endif
    }
}

// Level-parallel cloud fallout for columns of at most 64 levels, eight
// columns per block and one level per thread.  It keeps the column
// kernel's separate-rounding helpers, number floor, masks and serial
// height sum; the diagnostic entry points keep the column kernel.

// Device only: a block barrier has no meaning in the serial host
// build (tools/thompson_real_column_parity), which runs the column
// kernels instead (gpuwm/core/thompson.py LEVEL_PARALLEL_FALLOUT).
#ifdef __CUDACC_RTC__
extern "C" __global__ void thompson_aa_cloud_sediment_levels_64_with_masks(
    float* __restrict__ qc,
    const float* __restrict__ cloud_number_entry,
    float* __restrict__ cloud_number_tendency,
    const float* __restrict__ temperature,
    const float* __restrict__ pressure,
    const float* __restrict__ qv,
    const float* __restrict__ reference_density,
    const float* __restrict__ rain_active_columns,
    const float* __restrict__ cloud_active_columns,
    const float* __restrict__ vertical_velocity,
    const float* __restrict__ dz,
    float dt, int nz, int ny, int nx)
{
    const int c = threadIdx.x;
    const int column = blockIdx.x * 8 + c;
    const int j = column / nx;
    const int i = column - j * nx;
    const int k = threadIdx.y;
    const bool live = column < ny * nx
        && (cloud_active_columns == nullptr || cloud_active_columns[column] != 0.0f);
    __shared__ float depth[64][8], mass[64][8], mass_flux[64][8], number_flux[64][8];
    __shared__ int top[8];
    float density, cloud_mass, cloud_number, mass_velocity, number_velocity;
    float qc_tendency, qc_initial;
    const float rho_not = 101325.0f / (287.05f * 298.0f);
    const bool rain_refreshes_rhof = live && rain_active_columns != nullptr
        && rain_active_columns[column] != 0.0f;

    if (live && k < nz) {
        const size_t idx = IDX3(k, j, i);
        const float qvk = fmaxf(1.0e-10f, qv[idx]);
        density = 0.622f * pressure[idx]
            / (287.04f * temperature[idx] * (qvk + 0.622f));
        qc_initial = qc[idx];
        qc_tendency = 0.0f;
        const float held = reference_density[idx];
        cloud_mass = qc[idx] > THOMPSON_AA_R1
            ? qc[idx] * held : THOMPSON_AA_R1;
        cloud_number = thompson_aa_clamp_nc(
            thompson_aa_mul(
                thompson_aa_add(
                    cloud_number_entry[idx],
                    thompson_aa_mul(cloud_number_tendency[idx], dt)),
                held));
        mass_velocity = 0.0f;
        number_velocity = 0.0f;
        depth[k][c] = dz[idx];
        mass[k][c] = cloud_mass;
    }
    __syncthreads();
    if (k == 0) {
        int sediment_top = 0;
        float height_agl = 0.0f;
        if (live) for (int level = 0; level < nz - 1; ++level) {
            if (mass[level][c] > THOMPSON_AA_R2) sediment_top = level;
            height_agl += depth[level][c];
            if (height_agl > THOMPSON_AA_SED_HGT_AGL) break;
        }
        top[c] = sediment_top;
    }
    __syncthreads();
    const int sediment_top = top[c];
    if (live && k <= sediment_top) {
        const size_t idx = IDX3(k, j, i);
        if (cloud_mass > THOMPSON_AA_R1
                && vertical_velocity[idx] < THOMPSON_AA_SED_W_LIMIT) {
            const int nu_c = thompson_aa_nu_c(cloud_number);
            const float lambda_arg = thompson_aa_div(
                thompson_aa_mul(
                    thompson_aa_mul(
                        thompson_aa_mul(cloud_number, THOMPSON_AA_AM_R),
                        THOMPSON_AA_CCG2[nu_c]),
                    THOMPSON_AA_OCG1[nu_c]),
                cloud_mass);
            const double lambda =
                (double)thompson_aa_powf_cr(lambda_arg, THOMPSON_AA_OBMR);
            const double inverse_lambda = 1.0 / lambda;

            const float velocity_density = rain_refreshes_rhof
                ? density : reference_density[idx];
            const float rhof = sqrtf(rho_not / velocity_density);
            const float mass_prefix = thompson_aa_mul(
                thompson_aa_mul(
                    thompson_aa_mul(rhof, THOMPSON_AA_AV_C),
                    THOMPSON_AA_CCG5[nu_c]),
                THOMPSON_AA_OCG2[nu_c]);
            const float number_prefix = thompson_aa_mul(
                thompson_aa_mul(
                    thompson_aa_mul(rhof, THOMPSON_AA_AV_C),
                    THOMPSON_AA_CCG4[nu_c]),
                THOMPSON_AA_OCG1[nu_c]);
            mass_velocity = (float)((double)mass_prefix
                * inverse_lambda * inverse_lambda);
            number_velocity = (float)((double)number_prefix
                * inverse_lambda * inverse_lambda);
        }
    }
    if (live && k < nz) {
        mass_flux[k][c] = mass_velocity * cloud_mass;
        number_flux[k][c] = number_velocity * cloud_number;
    }
    __syncthreads();
    if (live && k <= sediment_top) {
        const size_t idx = IDX3(k, j, i);
        const float odzq = 1.0f / dz[idx];
        const float orho = 1.0f / density;
        const float mass_divergence = mass_flux[k + 1][c] - mass_flux[k][c];
        const float number_divergence = number_flux[k + 1][c] - number_flux[k][c];
        qc_tendency = thompson_aa_add(
            qc_tendency,
            thompson_aa_mul(thompson_aa_mul(mass_divergence, odzq), orho));
        cloud_number_tendency[idx] = thompson_aa_add(
            cloud_number_tendency[idx],
            thompson_aa_mul(thompson_aa_mul(number_divergence, odzq), orho));
        cloud_mass = fmaxf(
            THOMPSON_AA_R1,
            thompson_aa_add(
                cloud_mass,
                thompson_aa_mul(thompson_aa_mul(mass_divergence, odzq), dt)));
        cloud_number = fmaxf(
            THOMPSON_AA_NC_SED_FLOOR,
            thompson_aa_add(
                cloud_number,
                thompson_aa_mul(thompson_aa_mul(number_divergence, odzq),
                                dt)));
    }
    if (live && k < nz) {
        const size_t idx = IDX3(k, j, i);
        qc[idx] = thompson_aa_add(qc_initial, thompson_aa_mul(qc_tendency, dt));
    }
}
#endif  // __CUDACC_RTC__


#if defined(THOMPSON_AA_WRF39)
// ---------------------------------------------------------------------------
// THE FORK'S ICE AND GRAUPEL FALLOUT (RunConfig.thompson_version =
// "wrf_39_noaa").
// ---------------------------------------------------------------------------
//
// thompson.cu's classic ice and graupel fallout (thompson_ice_sediment_impl,
// thompson_graupel_sediment_impl) carry v4.6.1's constants and its graupel
// number, and that unit stays byte-frozen, so the fork's two passes live
// here.  Each is the classic pass statement for statement with only the
// fork's own differences:
//
//   ice     av_i = 1847.5 (fork :140, audit T9), the 499.D3 number ceiling
//           (fork :2879, :3733, audit T8), surface ice counted above R1*10
//           (fork :3603, audit T21);
//   graupel the slope from the fork's column intercept (fork :3110-3133,
//           thompson_aa_wrf39_graupel_intercept on the post-source state,
//           audit T1) with am_g at rho_g = 500 (audit T11), no number and
//           no size clamp; above 0 C the fall speed is at least the rain's,
//           vtgk = MAX(vtg, vtrk) (fork :3500-3505, audit T15); surface
//           graupel counted above R1*10 (fork :3653, audit T21).
//
// The density decisions are the classic passes': the fallout state is
// formed with the held pre-adjustment density, the fall speed and the
// tendency with the current one.

template <int KMAX>
__device__ __forceinline__ void thompson_aa_wrf39_ice_sediment_impl(
    float* __restrict__ qi,
    float* __restrict__ ni,
    const float* __restrict__ temperature,
    const float* __restrict__ pressure,
    const float* __restrict__ qv,
    const float* __restrict__ reference_density,
    const float* __restrict__ dz,
    float* __restrict__ rainnc,
    float* __restrict__ rainncv,
    float* __restrict__ snownc,
    float* __restrict__ snowncv,
    float dt, int nz, int ny, int nx)
{
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= ny * nx) return;
    const int j = column / nx;
    const int i = column - j * nx;

    float density[KMAX];
    float ice_mass[KMAX];
    float ice_number[KMAX];
    float mass_velocity[KMAX];
    float number_velocity[KMAX];
    float mass_flux[KMAX];
    float number_flux[KMAX];
    float qi_tendency[KMAX];
    float ni_tendency[KMAX];
    float qi_initial[KMAX];
    float ni_initial[KMAX];

    const float am_i = THOMPSON_AA_AM_I;
    const float oig2 = 0.16666667163372040f;
    const float rho_not = __fdiv_rn(101325.0f, 287.05f * 298.0f);
    const float av_i = 1847.5f;
    int sediment_top = 0;
    int nstep = 0;
    float velocity_above_mass = 0.0f;
    float velocity_above_number = 0.0f;

    for (int k = nz - 1; k >= 0; --k) {
        const size_t idx = IDX3(k, j, i);
        const float qvk = fmaxf(1.0e-10f, qv[idx]);
        const float rho = 0.622f * pressure[idx]
            / (287.04f * temperature[idx] * (qvk + 0.622f));
        density[k] = rho;
        qi_initial[k] = qi[idx];
        ni_initial[k] = ni[idx];
        qi_tendency[k] = 0.0f;
        ni_tendency[k] = 0.0f;
        const float ice_state_rho = reference_density == nullptr
            ? rho : reference_density[idx];
        if (qi[idx] > 1.0e-12f && qi[idx] * ice_state_rho > 1.0e-12f) {
            const float ri = qi[idx] * ice_state_rho;
            float nn = fmaxf(1.0e-6f, ni[idx] * ice_state_rho);
            if (nn <= 1.0e-6f) {
                const double lambda = 4.0 / 5.0e-6;
                const float prefix = __fdiv_rn(oig2 * ri, am_i);
                nn = fminf(THOMPSON_AA_WRF39_NI_MAX,
                           (float)((double)prefix * pow(lambda, 3.0)));
            }
            const float lambda_arg = am_i * 6.0f * nn / ri;
            double lambda = (double)powf(lambda_arg, 0.33333334326744080f);
            float diameter = (float)(4.0 / lambda);
            if (diameter < 5.0e-6f) {
                diameter = 5.0e-6f;
                lambda = 4.0 / (double)diameter;
                const float prefix = __fdiv_rn(oig2 * ri, am_i);
                nn = fminf(THOMPSON_AA_WRF39_NI_MAX,
                           (float)((double)prefix * pow(lambda, 3.0)));
            } else if (diameter > 300.0e-6f) {
                diameter = 300.0e-6f;
                lambda = 4.0 / (double)diameter;
                const float prefix = __fdiv_rn(oig2 * ri, am_i);
                nn = (float)((double)prefix * pow(lambda, 3.0));
            }
            ice_mass[k] = ri;
            ice_number[k] = nn;
            const float rhof = sqrtf(rho_not / rho);
            const double inverse_lambda = 1.0 / lambda;
            const float mass_prefix = rhof * av_i * 24.0f * oig2;
            mass_velocity[k] = (float)((double)mass_prefix * inverse_lambda);
            const float number_prefix = __fdiv_rn(rhof * av_i
                * 3.3233511f, 1.3293403f);
            number_velocity[k] = (float)((double)number_prefix
                                          * inverse_lambda);
        } else {
            const bool l_qi = qi[idx] > 1.0e-12f;
            ice_mass[k] = l_qi ? qi[idx] * ice_state_rho : 1.0e-12f;
            ice_number[k] = l_qi
                ? fmaxf(1.0e-6f, ni[idx] * ice_state_rho) : 1.0e-6f;
            mass_velocity[k] = velocity_above_mass;
            number_velocity[k] = velocity_above_number;
        }
        velocity_above_mass = mass_velocity[k];
        velocity_above_number = number_velocity[k];
        if (mass_velocity[k] > 1.0e-3f) {
            sediment_top = max(sediment_top, k);
            const float delta_tp = dz[idx] / mass_velocity[k];
            nstep = max(nstep, (int)(dt / delta_tp + 1.0f));
        }
    }
    if (sediment_top == nz - 1) sediment_top = nz - 2;
    nstep = max(nstep, 1);
    const float onstep = 1.0f / (float)nstep;
    const float dt_substep = dt * onstep;
    float exported = 0.0f;

    for (int step = 0; step < nstep; ++step) {
        for (int k = nz - 1; k >= 0; --k) {
            mass_flux[k] = mass_velocity[k] * ice_mass[k];
            number_flux[k] = number_velocity[k] * ice_number[k];
        }
        int k = nz - 1;
        size_t idx = IDX3(k, j, i);
        float inv_dz = 1.0f / dz[idx];
        float inv_rho = 1.0f / density[k];
        qi_tendency[k] -= mass_flux[k] * inv_dz * onstep * inv_rho;
        ni_tendency[k] -= number_flux[k] * inv_dz * onstep * inv_rho;
        ice_mass[k] = fmaxf(1.0e-12f,
            ice_mass[k] - mass_flux[k] * inv_dz * dt_substep);
        ice_number[k] = fmaxf(1.0e-6f,
            ice_number[k] - number_flux[k] * inv_dz * dt_substep);
        for (k = sediment_top; k >= 0; --k) {
            idx = IDX3(k, j, i);
            inv_dz = 1.0f / dz[idx];
            inv_rho = 1.0f / density[k];
            const float mass_divergence = mass_flux[k + 1] - mass_flux[k];
            const float number_divergence =
                number_flux[k + 1] - number_flux[k];
            qi_tendency[k] += mass_divergence * inv_dz * onstep * inv_rho;
            ni_tendency[k] += number_divergence * inv_dz * onstep * inv_rho;
            ice_mass[k] = fmaxf(1.0e-12f,
                ice_mass[k] + mass_divergence * inv_dz * dt_substep);
            ice_number[k] = fmaxf(1.0e-6f,
                ice_number[k] + number_divergence * inv_dz * dt_substep);
        }
        // fork :3603, ri(kts) > R1*10.
        if (ice_mass[0] > 1.0e-11f) {
            exported += mass_flux[0] * dt_substep;
        }
    }

    for (int k = 0; k < nz; ++k) {
        const size_t idx = IDX3(k, j, i);
        const float qi_new = qi_initial[k] + qi_tendency[k] * dt;
        float ni_new = fmaxf(1.0e-6f / density[k],
                             ni_initial[k] + ni_tendency[k] * dt);
        if (qi_new <= 1.0e-12f) {
            qi[idx] = qi_new;
            ni[idx] = ni_initial[k] + ni_tendency[k] * dt;
            continue;
        }
        const float lambda_arg = am_i * 6.0f * ni_new / qi_new;
        double lambda = (double)powf(lambda_arg, 0.33333334326744080f);
        float diameter = (float)(4.0 / lambda);
        if (diameter < 5.0e-6f) diameter = 5.0e-6f;
        else if (diameter > 300.0e-6f) diameter = 300.0e-6f;
        lambda = 4.0 / (double)diameter;
        const float prefix = __fdiv_rn(oig2 * qi_new, am_i);
        // fork :3733, 499.D3/rho.
        ni_new = fminf((float)((double)prefix * pow(lambda, 3.0)),
                       THOMPSON_AA_WRF39_NI_MAX / density[k]);
        qi[idx] = qi_new;
        ni[idx] = ni_new;
    }
    rainncv[column] = exported;
    snowncv[column] = exported;
    rainnc[column] += exported;
    snownc[column] += exported;
}

#define THOMPSON_AA_WRF39_ICE_SEDIMENT_PARAMETERS                        \
    float* __restrict__ qi, float* __restrict__ ni,                      \
    const float* __restrict__ temperature,                               \
    const float* __restrict__ pressure, const float* __restrict__ qv,    \
    const float* __restrict__ reference_density,                         \
    const float* __restrict__ dz, float* __restrict__ rainnc,            \
    float* __restrict__ rainncv, float* __restrict__ snownc,             \
    float* __restrict__ snowncv, float dt, int nz, int ny, int nx

#define THOMPSON_AA_WRF39_ICE_SEDIMENT_ARGUMENTS                         \
    qi, ni, temperature, pressure, qv, reference_density, dz, rainnc,    \
    rainncv, snownc, snowncv, dt, nz, ny, nx

extern "C" __global__ void thompson_aa_wrf39_ice_sediment_64(
    THOMPSON_AA_WRF39_ICE_SEDIMENT_PARAMETERS)
{
    thompson_aa_wrf39_ice_sediment_impl<THOMPSON_AA_KMAX_SHALLOW>(
        THOMPSON_AA_WRF39_ICE_SEDIMENT_ARGUMENTS);
}

extern "C" __global__ void thompson_aa_wrf39_ice_sediment_256(
    THOMPSON_AA_WRF39_ICE_SEDIMENT_PARAMETERS)
{
    thompson_aa_wrf39_ice_sediment_impl<THOMPSON_AA_KMAX_GENERIC>(
        THOMPSON_AA_WRF39_ICE_SEDIMENT_ARGUMENTS);
}

// graupel_intercept: N0_exp per level from thompson_aa_wrf39_graupel_
// intercept on the post-source state.  melt_rain_qr/nr and
// melt_rain_density are the rain fallout's own inputs (the rain evaporation
// writes the density with L_qr carried in its sign, as the classic snow
// pass reads it), so vtrk(k) is the rain pass's speed, inherited from above
// where a level has no rain.
template <int KMAX>
__device__ __forceinline__ void thompson_aa_wrf39_graupel_sediment_impl(
    float* __restrict__ qg,
    const float* __restrict__ graupel_intercept,
    const float* __restrict__ temperature,
    const float* __restrict__ pressure,
    const float* __restrict__ qv,
    const float* __restrict__ reference_density,
    const float* __restrict__ melt_rain_qr,
    const float* __restrict__ melt_rain_nr,
    const float* __restrict__ melt_rain_density,
    const float* __restrict__ dz,
    float* __restrict__ rainnc,
    float* __restrict__ rainncv,
    float* __restrict__ graupelnc,
    float* __restrict__ graupelncv,
    const float* __restrict__ active_columns,
    float dt, int nz, int ny, int nx)
{
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= ny * nx || active_columns[column] == 0.0f) return;
    const int j = column / nx;
    const int i = column - j * nx;

    float density[KMAX];
    float graupel_mass[KMAX];
    float mass_velocity[KMAX];
    float mass_flux[KMAX];
    float qg_tendency[KMAX];
    float qg_initial[KMAX];

    const float ogg3 = 0.16666667163372040f;
    const float rho_not = __fdiv_rn(101325.0f, 287.05f * 298.0f);
    // bv_g = 0.89 as the REAL(4) PARAMETER holds it, cgg(6) = WGAMMA(4.89).
    const double bv_g = (double)0.89f;
    int sediment_top = 0;
    int nstep = 0;
    float velocity_above = 0.0f;
    float rain_velocity_above = 0.0f;

    for (int k = nz - 1; k >= 0; --k) {
        const size_t idx = IDX3(k, j, i);
        const float qvk = fmaxf(1.0e-10f, qv[idx]);
        const float rho = 0.622f * pressure[idx]
            / (287.04f * temperature[idx] * (qvk + 0.622f));
        density[k] = rho;
        qg_initial[k] = qg[idx];
        qg_tendency[k] = 0.0f;

        // The rain pass's vtrk(k), as thompson.cu's snow fallout forms it.
        float rain_velocity = 0.0f;
        {
            const float carried = melt_rain_density[idx];
            const float rain_density = fabsf(carried);
            const bool l_qr = carried != 0.0f;
            const bool rewritten = carried < 0.0f;
            const float rain_rr = !l_qr ? 1.0e-12f
                : rewritten ? fmaxf(1.0e-12f, melt_rain_qr[idx] * rain_density)
                : melt_rain_qr[idx] * rain_density;
            if (rain_rr > 1.0e-12f) {
                const float am_r = THOMPSON_AA_AM_R;
                const float rain_number = fmaxf(
                    1.0e-6f, melt_rain_nr[idx] * rain_density);
                const double rain_lambda = (double)powf(
                    am_r * 6.0f * rain_number / rain_rr,
                    0.33333334326744080f);
                const float rain_rhof = sqrtf(rho_not / rho);
                rain_velocity = (float)(
                    (double)(rain_rhof * 4854.0f * 24.0f * ogg3)
                    * pow(rain_lambda, 4.0)
                    * pow(rain_lambda + 195.0, -5.0));
            } else {
                rain_velocity = rain_velocity_above;
            }
            rain_velocity_above = rain_velocity;
        }

        if (qg[idx] > 1.0e-12f) {
            const float state_rho = reference_density == nullptr
                ? rho : reference_density[idx];
            const float rg = qg[idx] * state_rho;
            double lamg, ilamg, n0_g;
            thompson_aa_wrf39_graupel_slope(
                graupel_intercept[idx], rg, &lamg, &ilamg, &n0_g);
            const float rhof = sqrtf(rho_not / rho);
            // vtg = rhof*av_g*cgg(6)*ogg3 * ilamg**bv_g, fork :3500.
            const float prefix = rhof * 442.0f * 20.3632278f * ogg3;
            float velocity = (float)((double)prefix * pow(ilamg, bv_g));
            // fork :3501-3505.
            if (temperature[idx] > 273.15f) {
                velocity = fmaxf(velocity, rain_velocity);
            }
            mass_velocity[k] = velocity;
            graupel_mass[k] = rg;
        } else {
            graupel_mass[k] = 1.0e-12f;
            mass_velocity[k] = velocity_above;
        }
        velocity_above = mass_velocity[k];

        if (mass_velocity[k] > 1.0e-3f) {
            sediment_top = max(sediment_top, k);
            const float delta_tp = dz[idx] / mass_velocity[k];
            nstep = max(nstep, (int)(dt / delta_tp + 1.0f));
        }
    }
    if (sediment_top == nz - 1) sediment_top = nz - 2;
    nstep = max(nstep, 1);
    const float onstep = 1.0f / (float)nstep;
    const float dt_substep = dt * onstep;
    float exported = 0.0f;

    for (int step = 0; step < nstep; ++step) {
        for (int k = nz - 1; k >= 0; --k) {
            mass_flux[k] = mass_velocity[k] * graupel_mass[k];
        }
        int k = nz - 1;
        size_t idx = IDX3(k, j, i);
        float inv_dz = 1.0f / dz[idx];
        float inv_rho = 1.0f / density[k];
        qg_tendency[k] -= mass_flux[k] * inv_dz * onstep * inv_rho;
        graupel_mass[k] = fmaxf(1.0e-12f,
            graupel_mass[k] - mass_flux[k] * inv_dz * dt_substep);
        for (k = sediment_top; k >= 0; --k) {
            idx = IDX3(k, j, i);
            inv_dz = 1.0f / dz[idx];
            inv_rho = 1.0f / density[k];
            const float divergence = mass_flux[k + 1] - mass_flux[k];
            qg_tendency[k] += divergence * inv_dz * onstep * inv_rho;
            graupel_mass[k] = fmaxf(1.0e-12f,
                graupel_mass[k] + divergence * inv_dz * dt_substep);
        }
        // fork :3653, rg(kts) > R1*10.
        if (graupel_mass[0] > 1.0e-11f) {
            exported += mass_flux[0] * dt_substep;
        }
    }

    for (int k = 0; k < nz; ++k) {
        const size_t idx = IDX3(k, j, i);
        const float qg_new = qg_initial[k] + qg_tendency[k] * dt;
        qg[idx] = qg_new <= 1.0e-12f ? 0.0f : qg_new;
    }
    rainncv[column] += exported;
    graupelncv[column] += exported;
    rainnc[column] += exported;
    graupelnc[column] += exported;
}

#define THOMPSON_AA_WRF39_GRAUPEL_SEDIMENT_PARAMETERS                    \
    float* __restrict__ qg, const float* __restrict__ graupel_intercept, \
    const float* __restrict__ temperature,                               \
    const float* __restrict__ pressure, const float* __restrict__ qv,    \
    const float* __restrict__ reference_density,                         \
    const float* __restrict__ melt_rain_qr,                              \
    const float* __restrict__ melt_rain_nr,                              \
    const float* __restrict__ melt_rain_density,                         \
    const float* __restrict__ dz, float* __restrict__ rainnc,            \
    float* __restrict__ rainncv, float* __restrict__ graupelnc,          \
    float* __restrict__ graupelncv,                                      \
    const float* __restrict__ active_columns,                            \
    float dt, int nz, int ny, int nx

#define THOMPSON_AA_WRF39_GRAUPEL_SEDIMENT_ARGUMENTS                     \
    qg, graupel_intercept, temperature, pressure, qv, reference_density, \
    melt_rain_qr, melt_rain_nr, melt_rain_density, dz, rainnc, rainncv,  \
    graupelnc, graupelncv, active_columns, dt, nz, ny, nx

extern "C" __global__ void thompson_aa_wrf39_graupel_sediment_64(
    THOMPSON_AA_WRF39_GRAUPEL_SEDIMENT_PARAMETERS)
{
    thompson_aa_wrf39_graupel_sediment_impl<THOMPSON_AA_KMAX_SHALLOW>(
        THOMPSON_AA_WRF39_GRAUPEL_SEDIMENT_ARGUMENTS);
}

extern "C" __global__ void thompson_aa_wrf39_graupel_sediment_256(
    THOMPSON_AA_WRF39_GRAUPEL_SEDIMENT_PARAMETERS)
{
    thompson_aa_wrf39_graupel_sediment_impl<THOMPSON_AA_KMAX_GENERIC>(
        THOMPSON_AA_WRF39_GRAUPEL_SEDIMENT_ARGUMENTS);
}
#endif  // THOMPSON_AA_WRF39


#if defined(THOMPSON_AA_WRF39)
// ---------------------------------------------------------------------------
// THE FORK'S SNOW FALLOUT (RunConfig.thompson_version = "wrf_39_noaa").
// ---------------------------------------------------------------------------
//
// thompson.cu's thompson_snow_sediment_impl (the RAIN_PRESENCE arm the
// coupled adapter launches) statement for statement, with the fork's two
// differences:
//
//   surface snow is counted above R1*10 (fork :3628, audit T21);
//   the speed of snow on a level warmer than 0 C follows RunConfig.
//   thompson_fork_snow_fall, passed as ``singular_fall``:
//     0 ("blend", the default): WRF v4.6.1's rain-share blend where snow
//        melts, vtsk = vts*SR + (1-SR)*vtrk with SR = rs/(rs+rr) (:3722-
//        3724), the later upstream form the fork itself carries commented
//        out at fork :3475-3476 as an upstream bug fix;
//     1 ("wrf_39_noaa"): the fork's own expression (fork :3472-3478,
//        audit T4), above 0.1 C
//          vtsk = MAX(vts*vts_boost, vts*((vtrk - vts*vts_boost)/(T - T_0)))
//        and vts*vts_boost between 0 and 0.1 C, with vts_boost = 1.5 on
//        every level the source stage found at or above 0 C (fork :2151).
//        The divisor goes to zero just above +0.1 C, where 1 m/s snow
//        under 5 m/s rain falls at about 35 m/s: a defect kept by name for
//        the owner's ruling, never the default.
template <int KMAX>
__device__ __forceinline__ void thompson_aa_wrf39_snow_sediment_impl(
    float* __restrict__ qs,
    const float* __restrict__ snow_melt_marker,
    const float* __restrict__ melt_rain_qr,
    const float* __restrict__ melt_rain_nr,
    const float* __restrict__ temperature,
    const float* __restrict__ pressure,
    const float* __restrict__ qv,
    const float* __restrict__ reference_density,
    const float* __restrict__ reference_temperature,
    const float* __restrict__ velocity_boost,
    const float* __restrict__ melt_rain_density,
    const float* __restrict__ dz,
    float* __restrict__ rainnc,
    float* __restrict__ rainncv,
    float* __restrict__ snownc,
    float* __restrict__ snowncv,
    int singular_fall, float dt, int nz, int ny, int nx)
{
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= ny * nx) return;
    const int j = column / nx;
    const int i = column - j * nx;

    float density[KMAX];
    float snow_mass[KMAX];
    float mass_velocity[KMAX];
    float mass_flux[KMAX];
    float qs_tendency[KMAX];
    float qs_initial[KMAX];

    const float rho_not = __fdiv_rn(101325.0f, 287.05f * 298.0f);
    int sediment_top = 0;
    int nstep = 0;
    float velocity_above = 0.0f;
    float rain_velocity_above = 0.0f;

    for (int k = nz - 1; k >= 0; --k) {
        const size_t idx = IDX3(k, j, i);
        const float qvk = fmaxf(1.0e-10f, qv[idx]);
        const float rho = 0.622f * pressure[idx]
            / (287.04f * temperature[idx] * (qvk + 0.622f));
        density[k] = rho;
        qs_initial[k] = qs[idx];
        qs_tendency[k] = 0.0f;

        float rain_rr = 1.0e-12f;
        float rain_velocity = 0.0f;
        {
            const float carried = melt_rain_density[idx];
            const float rain_density = fabsf(carried);
            const bool l_qr = carried != 0.0f;
            const bool rewritten = carried < 0.0f;
            rain_rr = !l_qr ? 1.0e-12f
                : rewritten ? fmaxf(1.0e-12f, melt_rain_qr[idx] * rain_density)
                : melt_rain_qr[idx] * rain_density;
            if (rain_rr > 1.0e-12f) {
                const float am_r = THOMPSON_AA_AM_R;
                const float rain_number = fmaxf(
                    1.0e-6f, melt_rain_nr[idx] * rain_density);
                const double rain_lambda = (double)powf(
                    am_r * 6.0f * rain_number / rain_rr,
                    0.33333334326744080f);
                const float rain_rhof = sqrtf(rho_not / rho);
                rain_velocity = (float)(
                    (double)(rain_rhof * 4854.0f * 24.0f
                             * 0.16666667163372040f)
                    * pow(rain_lambda, 4.0)
                    * pow(rain_lambda + 195.0, -5.0));
            } else {
                rain_velocity = rain_velocity_above;
            }
            rain_velocity_above = rain_velocity;
        }

        const float snow_state_rho = reference_density[idx];
        if (qs[idx] > 1.0e-12f && qs[idx] * snow_state_rho > 1.0e-12f) {
            const float rs = qs[idx] * snow_state_rho;
            const float smob = rs * THOMPSON_AA_OAMS;      // rs*oams
            const float tc0 = fminf(-0.1f,
                                    reference_temperature[idx] - 273.15f);
            const float moment = 3.0f;
            const float tc02 = tc0 * tc0;
            const float moment2 = moment * moment;
            const float loga = 5.065339f + -0.062659f * tc0
                + -3.032362f * moment + 0.029469f * tc0 * moment
                + -0.000285f * tc02 + 0.31255f * moment2
                + 0.000204f * tc02 * moment
                + 0.003199f * tc0 * moment2
                + 0.0f * tc02 * tc0
                + -0.015952f * moment2 * moment;
            const float exponent = 0.476221f + -0.015896f * tc0
                + 0.165977f * moment + 0.007468f * tc0 * moment
                + -0.000141f * tc02 + 0.060366f * moment2
                + 0.000079f * tc02 * moment
                + 0.000594f * tc0 * moment2
                + 0.0f * tc02 * tc0
                + -0.003577f * moment2 * moment;
            const float smoc = powf(10.0f, loga) * powf(smob, exponent);
            const float mean_ratio = smob / smoc;
            float ils1 = 1.0f / (mean_ratio * 20.78f + 100.0f);
            float ils2 = 1.0f / (mean_ratio * 3.29f + 100.0f);
            const float ratio_power = powf(mean_ratio, 0.6357f);
            const float numerator1 = 490.6f * 3.51325202f
                * powf(ils1, 3.55f);
            const float numerator2 = 17.46f * ratio_power * 7.61279917f
                * powf(ils2, 4.1857f);
            ils1 = 1.0f / (mean_ratio * 20.78f);
            ils2 = 1.0f / (mean_ratio * 3.29f);
            const float denominator1 = 490.6f * 2.0f
                * powf(ils1, 3.0f);
            const float denominator2 = 17.46f * ratio_power * 3.87160635f
                * powf(ils2, 3.6357f);
            const float rhof = sqrtf(rho_not / rho);
            const float vts = rhof * 40.0f
                * (numerator1 + numerator2)
                / (denominator1 + denominator2);
            const float boost = velocity_boost[idx];
            float snow_velocity;
            const float temp_k = temperature[idx];
            if (singular_fall) {
                if (temp_k > 273.15f + 0.1f) {
                    snow_velocity = fmaxf(vts * boost,
                        vts * ((rain_velocity - vts * boost)
                               / (temp_k - 273.15f)));
                } else {
                    snow_velocity = vts * boost;
                }
            } else if (snow_melt_marker[idx] != 0.0f) {
                const float solid_fraction = rs / (rs + rain_rr);
                snow_velocity = vts * boost * solid_fraction
                    + rain_velocity * (1.0f - solid_fraction);
            } else {
                snow_velocity = vts * boost;
            }
            mass_velocity[k] = snow_velocity;
            snow_mass[k] = rs;
        } else {
            snow_mass[k] = qs[idx] > 1.0e-12f
                ? qs[idx] * snow_state_rho : 1.0e-12f;
            mass_velocity[k] = velocity_above;
        }
        velocity_above = mass_velocity[k];

        if (mass_velocity[k] > 1.0e-3f) {
            sediment_top = max(sediment_top, k);
            const float delta_tp = dz[idx] / mass_velocity[k];
            nstep = max(nstep, (int)(dt / delta_tp + 1.0f));
        }
    }
    if (sediment_top == nz - 1) sediment_top = nz - 2;
    nstep = max(nstep, 1);
    const float onstep = 1.0f / (float)nstep;
    const float dt_substep = dt * onstep;
    float exported = 0.0f;

    for (int step = 0; step < nstep; ++step) {
        for (int k = nz - 1; k >= 0; --k) {
            mass_flux[k] = mass_velocity[k] * snow_mass[k];
        }
        int k = nz - 1;
        size_t idx = IDX3(k, j, i);
        float inv_dz = 1.0f / dz[idx];
        float inv_rho = 1.0f / density[k];
        qs_tendency[k] -= mass_flux[k] * inv_dz * onstep * inv_rho;
        snow_mass[k] = fmaxf(1.0e-12f,
            snow_mass[k] - mass_flux[k] * inv_dz * dt_substep);
        for (k = sediment_top; k >= 0; --k) {
            idx = IDX3(k, j, i);
            inv_dz = 1.0f / dz[idx];
            inv_rho = 1.0f / density[k];
            const float divergence = mass_flux[k + 1] - mass_flux[k];
            qs_tendency[k] += divergence * inv_dz * onstep * inv_rho;
            snow_mass[k] = fmaxf(1.0e-12f,
                snow_mass[k] + divergence * inv_dz * dt_substep);
        }
        // fork :3628, rs(kts) > R1*10.
        if (snow_mass[0] > 1.0e-11f) {
            exported += mass_flux[0] * dt_substep;
        }
    }

    for (int k = 0; k < nz; ++k) {
        const size_t idx = IDX3(k, j, i);
        const float qs_new = qs_initial[k] + qs_tendency[k] * dt;
        qs[idx] = qs_new <= 1.0e-12f ? 0.0f : qs_new;
    }
    rainncv[column] += exported;
    snowncv[column] += exported;
    rainnc[column] += exported;
    snownc[column] += exported;
}

#define THOMPSON_AA_WRF39_SNOW_SEDIMENT_PARAMETERS                       \
    float* __restrict__ qs, const float* __restrict__ snow_melt_marker,  \
    const float* __restrict__ melt_rain_qr,                              \
    const float* __restrict__ melt_rain_nr,                              \
    const float* __restrict__ temperature,                               \
    const float* __restrict__ pressure, const float* __restrict__ qv,    \
    const float* __restrict__ reference_density,                         \
    const float* __restrict__ reference_temperature,                     \
    const float* __restrict__ velocity_boost,                            \
    const float* __restrict__ melt_rain_density,                         \
    const float* __restrict__ dz, float* __restrict__ rainnc,            \
    float* __restrict__ rainncv, float* __restrict__ snownc,             \
    float* __restrict__ snowncv, int singular_fall, float dt,            \
    int nz, int ny, int nx

#define THOMPSON_AA_WRF39_SNOW_SEDIMENT_ARGUMENTS                        \
    qs, snow_melt_marker, melt_rain_qr, melt_rain_nr, temperature,       \
    pressure, qv, reference_density, reference_temperature,              \
    velocity_boost, melt_rain_density, dz, rainnc, rainncv, snownc,      \
    snowncv, singular_fall, dt, nz, ny, nx

extern "C" __global__ void thompson_aa_wrf39_snow_sediment_64(
    THOMPSON_AA_WRF39_SNOW_SEDIMENT_PARAMETERS)
{
    thompson_aa_wrf39_snow_sediment_impl<THOMPSON_AA_KMAX_SHALLOW>(
        THOMPSON_AA_WRF39_SNOW_SEDIMENT_ARGUMENTS);
}

extern "C" __global__ void thompson_aa_wrf39_snow_sediment_256(
    THOMPSON_AA_WRF39_SNOW_SEDIMENT_PARAMETERS)
{
    thompson_aa_wrf39_snow_sediment_impl<THOMPSON_AA_KMAX_GENERIC>(
        THOMPSON_AA_WRF39_SNOW_SEDIMENT_ARGUMENTS);
}

// The fork's vts_boost on the levels its source stage found at or above
// 0 C (fork :2151): 1.5 where the singular fall is selected, 1.0 (the
// v4.6.1 start value the cold network already wrote) otherwise.
extern "C" __global__ void thompson_aa_wrf39_warm_snow_boost(
    const float* __restrict__ entry_warm_mask,
    float* __restrict__ velocity_boost, int size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    if (entry_warm_mask[idx] != 0.0f) velocity_boost[idx] = 1.5f;
}
#endif  // THOMPSON_AA_WRF39
