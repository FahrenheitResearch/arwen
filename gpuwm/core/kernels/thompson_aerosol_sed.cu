// gpuwm/core/kernels/thompson_aerosol_sed.cu
//
// Aerosol-aware Thompson (mp_physics=28): number-weighted cloud-water
// sedimentation and the number-conserving final phase cleanup.
//
// Numerical authority is /home/drew/wrf461-pristine/phys/module_mp_thompson.F
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
        const float qc_new = thompson_aa_add(
            qc_initial[k], thompson_aa_mul(qc_tendency[k], dt));
        qc[idx] = qc_new <= THOMPSON_AA_R1 ? 0.0f : qc_new;
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
    } else {
        thompson_aa_bound_ice_number(qi[idx] * rho, rho, &ni[idx]);
    }
}
