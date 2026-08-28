// WRF v4.6.1 RUC LSM dominant-category surface/soil parameter setup.
// One thread transcribes one call to module_sf_ruclsm.F:soilvegin with
// mosaic_lu=0 and mosaic_soil=0.  Explicit round-to-nearest intrinsics keep
// the operation boundaries used by WRF default REAL arithmetic.

// PROVISIONAL float32 transcendentals, pending a verified glibc transcription.
//
// gfortran lowers `**`, exp and log10 on default REAL to glibc's powf/expf/
// log10f, and glibc's are NOT correctly rounded (2.39 still ships the 1993
// SunPro float32 reduction for log10f), so rounding a float64 evaluation once
// is a *different* function, not merely a more accurate one.  Measured against
// glibc 2.39 over the 30720 distinct float32 arguments the RUC snow-covered
// land column reaches (see the report accompanying this branch):
//
//   float64-then-round-once (below)     783 / 30720  = 2.55 %, all 1 ULP
//   CUDA device libm (powf/expf/log10f) 3030 / 30720 = 9.86 %, up to 2 ULP
//   numpy 2.2.6 float32 loop            4420 / 30720 = 14.4 %, up to 2 ULP
//   numpy 2.4.3 float32 loop            8901 / 30720 = 29.0 %, up to 2 ULP
//
// So this is the closest of the four and, unlike the other three, it is
// identical on the host and the device, which is why the CPU and CUDA RUC
// lanes now agree bit for bit.  It is still not glibc: 723 of the residual 783
// are the freezing-curve log, whose only consumer is a `tln < 0.` sign test
// that never flips, and 34 more are the log10 feeding WRF's discarded legacy
// McCumber conductivity, leaving ~26 consequential 1-ULP deviations that no
// pinned fixture reaches.  Replace all three with the glibc transcription when
// it lands.

// >>> RUC_NZS TIER LADDER >>>
// The soil column's level count.  9 is the geometry this source keeps when
// nothing overrides it, and every nine-level configuration compiles the
// UNSPECIALIZED module -- no define is injected, so the string handed to
// NVRTC is byte-identical to what module_source("ruc") assembles, and the
// manifest key stays 'gpuwm.core.kernels:ruc'.  Exactly acoustic.cu's
// WPHI_MAX_LEV ladder (:625-627), and selected the same way, through
// gpuwm.core.kernels.get_kernel_int_defines.  gpuwm/core/ruc_gpu.py owns the
// dispatch; tests/test_ruc_nzs_tier.py pins both halves of the claim.
//
// The M1/M2/M3/DTDZS_LEN macros are DERIVED HERE, as bare literals, and never
// as arithmetic on RUC_NZS.  That is deliberate: `RUC_NZS_M2` expands to the
// token `7` at the shipped geometry, where `RUC_NZS - 2` would expand to
// `9 - 2`.  Bare literals make the nine-level preprocessed translation unit
// token-for-token what it was before this ladder existed, which is how the
// re-pinned mp=8 freeze digest is justified without a device measurement.
#ifndef RUC_NZS
#define RUC_NZS 9
#endif
#if RUC_NZS == 9
#define RUC_NZS_M1 8
#define RUC_NZS_M2 7
#define RUC_NZS_M3 6
#define RUC_DTDZS_LEN 14
#elif RUC_NZS == 6
#define RUC_NZS_M1 5
#define RUC_NZS_M2 4
#define RUC_NZS_M3 3
#define RUC_DTDZS_LEN 8
#else
#error "RUC_NZS must be 6 or 9: share/module_soil_pre.F:init_soil_depth_3 tabulates zs for those lengths only"
#endif
// <<< RUC_NZS TIER LADDER <<<

// RUC nine-level soil-layer midpoints, transcribed from WRF v4.6.1
// share/module_soil_pre.F:1175 (SUBROUTINE init_soil_depth_3, the
// num_soil_layers .EQ. 9 branch):
//     zs = (/ 0.00 , 0.01 , 0.04 , 0.10 , 0.30, 0.60, 1.00 , 1.60, 3.00 /)
// WRF default REAL is float32, so each literal is the float32 nearest value.
//
// This table is deliberately __constant__ rather than a local initializer.
// A local initializer is a compile-time constant, so the PTX->SASS backend
// constant-folds the whole zshalf/dtdzs derivation built from it -- and that
// fold is a different implementation of the arithmetic from the one the SM
// runs.  Measured on sm_120, ptxas 12.8.93 and 12.9.86 fold
//     __fsub_rn(0x3CCCCCCC, 0x3BA3D70A)   ==  zshalf[2] - zshalf[1]
// to 0x3CA3D709, where the exact difference is an FP32 halfway case whose
// round-to-nearest-even answer, and the word the hardware produces, is
// 0x3CA3D70A.  ptxas 13.0.88 and 13.1.115 fold it correctly, as does
// 12.8.93 at -O0.  __constant__ memory is not foldable, so the derivation
// runs on the device under the hardware rounding mode on every toolkit.
//
// Note the defect is narrower than "ptxas mis-rounds 0.025f - 0.005f": that
// bare pair folds *correctly* under 12.8.93.  The toolchain guard in
// tests/test_fp32_tie_folding_gpu.py therefore sweeps tie pairs rather than
// asserting one.  Measured alternatives that do NOT stop the fold under
// 12.8.93: a plain local literal array, asm()/asm volatile() movs, a
// volatile local array, and __device__ static const -- the last of these
// reproduces all 16 tests/test_ruc_gpu.py failures exactly.
// >>> RUC_NZS DEPTH TABLE >>>
#if RUC_NZS == 9
__constant__ real ruc_soil_layer_depth[RUC_NZS] = {
    0.00f, 0.01f, 0.04f, 0.10f, 0.30f,
    0.60f, 1.00f, 1.60f, 3.00f
};
#elif RUC_NZS == 6
// share/module_soil_pre.F:1153-1194 (init_soil_depth_3), the
// num_soil_layers .EQ. 6 branch:
//     zs = (/ 0.00 , 0.05 , 0.20 , 0.40 , 1.60, 3.00 /)
// The exact line of that branch is not cited because the WRF tree is not
// vendored here and an unverified line number is worse than none; the
// routine range is the one gpuwm/core/ruc_contract.py already carries.
// These are the same float32 nearest values
// gpuwm.ingest.ruc_soil.RUC_LEVEL_DEPTHS_M[6] carries and
// gpuwm.core.ruc.ruc_soil_geometry(6) returns -- that table is the one
// oracle-matched against WRF 4.7.1 real.exe (ZS 0 0.05 0.2 0.4 1.6 3), and
// tests/test_ruc_nzs_tier.py pins the literals below against it bit for
// bit, so the two transcriptions cannot drift.
__constant__ real ruc_soil_layer_depth[RUC_NZS] = {
    0.00f, 0.05f, 0.20f, 0.40f, 1.60f, 3.00f
};
#endif
// <<< RUC_NZS DEPTH TABLE <<<

// The 0.05 m and 0.01 m snow-layer thicknesses of
// phys/module_sf_ruclsm.F:3387-3388, deltsn's first and snth's second.
// __constant__ for the same reason, and against the same backend defect, as
// the table above: ruc_snow_layer_thresholds derives deltsn and snth from
// them, and a local literal pair would hand that arithmetic to the folder.
__constant__ real ruc_snow_layer_threshold_depth[2] = { 0.05f, 0.01f };

__device__ __forceinline__
real ruc_powf_rn(real base, real exponent)
{
    return __double2float_rn(pow((double)base, (double)exponent));
}

__device__ __forceinline__
real ruc_log10f_rn(real value)
{
    return __double2float_rn(log10((double)value));
}

__device__ __forceinline__
real ruc_expf_rn(real value)
{
    return __double2float_rn(exp((double)value));
}

extern "C" __global__
void ruc_surface_parameters(
    const int* __restrict__ isltyp,
    const int* __restrict__ ivgtyp,
    const real* __restrict__ shdmin,
    const real* __restrict__ shdmax,
    const real* __restrict__ vegfrac,
    const real* __restrict__ znt_in,
    const real* __restrict__ lai_in,
    const int* __restrict__ ifortbl,
    const real* __restrict__ z0tbl,
    const real* __restrict__ lemitbl,
    const real* __restrict__ pctbl,
    const real* __restrict__ laitbl,
    const real* __restrict__ bb,
    const real* __restrict__ drysmc,
    const real* __restrict__ hc,
    const real* __restrict__ maxsmc,
    const real* __restrict__ refsmc,
    const real* __restrict__ satpsi,
    const real* __restrict__ satdk,
    const real* __restrict__ wltsmc,
    const real* __restrict__ qtz,
    int* __restrict__ iforest_out,
    real* __restrict__ emiss_out,
    real* __restrict__ pc_out,
    real* __restrict__ znt_out,
    real* __restrict__ lai_out,
    real* __restrict__ qwrtz_out,
    real* __restrict__ rhocs_out,
    real* __restrict__ bclh_out,
    real* __restrict__ dqm_out,
    real* __restrict__ ksat_out,
    real* __restrict__ psis_out,
    real* __restrict__ qmin_out,
    real* __restrict__ ref_out,
    real* __restrict__ wilt_out,
    int iswater, int rdlai2d, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int vegetation_index = ivgtyp[idx] - 1;
    int soil_index = isltyp[idx] - 1;
    int forest_class = ifortbl[vegetation_index];
    iforest_out[idx] = forest_class;

    real green_range = __fsub_rn(shdmax[idx], shdmin[idx]);
    real factor;
    if (green_range < 1.0f) {
        factor = 1.0f;
    } else {
        real numerator = __fsub_rn(vegfrac[idx], shdmin[idx]);
        real denominator = fmaxf(1.0f, green_range);
        real ratio = __fdiv_rn(numerator, denominator);
        factor = __fsub_rn(1.0f, fmaxf(0.0f, fminf(1.0f, ratio)));
    }

    real table_lai = laitbl[vegetation_index];
    real scaled_lai = __fmul_rn(0.8f, table_lai);
    real delta_lai = 0.0f;
    if (forest_class == 1) {
        delta_lai = fminf(0.2f, scaled_lai);
    } else if (forest_class == 2 || forest_class == 7) {
        delta_lai = fminf(0.5f, scaled_lai);
    } else if (forest_class == 3) {
        delta_lai = fminf(0.45f, scaled_lai);
    } else if (forest_class == 4) {
        delta_lai = fminf(0.75f, scaled_lai);
    } else if (forest_class == 5) {
        delta_lai = fminf(0.86f, scaled_lai);
    }

    real roughness = znt_in[idx];
    real leaf_area = lai_in[idx];
    if (ivgtyp[idx] == iswater) {
        if (!rdlai2d) leaf_area = table_lai;
    } else {
        if (!rdlai2d) {
            leaf_area = __fsub_rn(
                table_lai, __fmul_rn(delta_lai, factor));
        }
        roughness = z0tbl[vegetation_index];
        if (forest_class == 7) {
            roughness = __fsub_rn(
                roughness, __fmul_rn(0.125f, factor));
        }
    }
    emiss_out[idx] = lemitbl[vegetation_index];
    pc_out[idx] = pctbl[vegetation_index];
    znt_out[idx] = roughness;
    lai_out[idx] = leaf_area;

    qwrtz_out[idx] = 0.0f;
    rhocs_out[idx] = 0.0f;
    bclh_out[idx] = 0.0f;
    dqm_out[idx] = 0.0f;
    ksat_out[idx] = 0.0f;
    psis_out[idx] = 0.0f;
    qmin_out[idx] = 0.0f;
    ref_out[idx] = 0.0f;
    wilt_out[idx] = 0.0f;
    if (isltyp[idx] == 14) return;

    qwrtz_out[idx] = qtz[soil_index];
    rhocs_out[idx] = __fmul_rn(hc[soil_index], 1.0e6f);
    bclh_out[idx] = bb[soil_index];
    dqm_out[idx] = __fsub_rn(maxsmc[soil_index], drysmc[soil_index]);
    ksat_out[idx] = satdk[soil_index];
    psis_out[idx] = -satpsi[soil_index];
    qmin_out[idx] = drysmc[soil_index];
    ref_out[idx] = refsmc[soil_index];
    wilt_out[idx] = wltsmc[soil_index];
}


// Freezing-curve partition used on both sides of WRF's snow-free soiltemp
// call.  One thread owns a complete nine-level soil column.
extern "C" __global__
void ruc_soil_phase_partition(
    const real* __restrict__ soilmois,
    const real* __restrict__ tso,
    real* __restrict__ smfrkeep,
    const real* __restrict__ keepfr,
    const real* __restrict__ dqm_a,
    const real* __restrict__ qmin_a,
    const real* __restrict__ psis_a,
    const real* __restrict__ bclh_a,
    int update_smfrkeep,
    real* __restrict__ soiliqw,
    real* __restrict__ soilice,
    real* __restrict__ tav,
    real* __restrict__ soilmoism,
    real* __restrict__ soiliqwm,
    real* __restrict__ soilicem,
    real* __restrict__ lwsat,
    real* __restrict__ fwsat,
    int ncolumn)
{
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= ncolumn) return;

    const real zero = 0.0f;
    const real half = 0.5f;
    const real one = 1.0f;
    const real freeze = 273.15f;
    const real xlmelt = 3.35e5f;
    const real gravity = 9.81f;
    // WRF forms RIW as RHOICE*1.E-3.  The product is one float32 ULP
    // above a source literal 0.9f, so retain the operation boundary.
    const real riw = __fmul_rn(900.0f, 1.0e-3f);
    real dqm = dqm_a[column];
    real qmin = qmin_a[column];
    real psis = psis_a[column];
    real bclh = bclh_a[column];
    real maximum = __fadd_rn(dqm, qmin);
    real exponent = __fdiv_rn(-one, bclh);

    for (int level = 0; level < RUC_NZS; ++level) {
        int index = level * ncolumn + column;
        soiliqw[index] = zero;
        soilice[index] = zero;
        tav[index] = zero;
        soilmoism[index] = zero;
        soiliqwm[index] = zero;
        soilicem[index] = zero;
        lwsat[index] = zero;
        fwsat[index] = zero;

        real temperature = tso[index];
        real tln = logf(__fdiv_rn(temperature, freeze));
        if (tln < zero) {
            real base = __fmul_rn(
                xlmelt, __fsub_rn(temperature, freeze));
            base = __fdiv_rn(base, temperature);
            base = __fdiv_rn(base, gravity);
            base = __fdiv_rn(base, psis);
            // ruc_powf_rn, not powf: plain CUDA powf drifts ~4 ULP into
            // soilice on cold deep layers relative to gfortran.  See the
            // provisional-transcendental note at the top of this file.
            real liquid = __fsub_rn(
                __fmul_rn(
                    maximum,
                    __double2float_rn(pow((double)base, (double)exponent))),
                qmin);
            liquid = fmaxf(zero, liquid);
            liquid = fminf(liquid, soilmois[index]);
            real ice = __fdiv_rn(
                __fsub_rn(soilmois[index], liquid), riw);
            if (keepfr[index] == one) {
                ice = fminf(ice, smfrkeep[index]);
                liquid = fmaxf(
                    zero,
                    __fsub_rn(soilmois[index], __fmul_rn(ice, riw)));
            }
            soiliqw[index] = liquid;
            soilice[index] = ice;
        } else {
            soiliqw[index] = soilmois[index];
        }
    }

    for (int level = 0; level < RUC_NZS_M1; ++level) {
        int index = level * ncolumn + column;
        int below = (level + 1) * ncolumn + column;
        real middle_temperature = __fmul_rn(
            half, __fadd_rn(tso[index], tso[below]));
        real middle_moisture = __fmul_rn(
            half, __fadd_rn(soilmois[index], soilmois[below]));
        tav[index] = middle_temperature;
        soilmoism[index] = middle_moisture;
        real tln = logf(__fdiv_rn(middle_temperature, freeze));
        if (tln < zero) {
            real base = __fmul_rn(
                xlmelt, __fsub_rn(middle_temperature, freeze));
            base = __fdiv_rn(base, middle_temperature);
            base = __fdiv_rn(base, gravity);
            base = __fdiv_rn(base, psis);
            // Same ruc_powf_rn substitution as the full-level loop above.
            real liquid = __fsub_rn(
                __fmul_rn(
                    maximum,
                    __double2float_rn(pow((double)base, (double)exponent))),
                qmin);
            fwsat[index] = __fsub_rn(dqm, liquid);
            lwsat[index] = __fadd_rn(liquid, qmin);
            liquid = fmaxf(zero, liquid);
            liquid = fminf(liquid, middle_moisture);
            real ice = __fdiv_rn(
                __fsub_rn(middle_moisture, liquid), riw);
            if (keepfr[index] == one) {
                real memory = __fmul_rn(
                    half, __fadd_rn(smfrkeep[index], smfrkeep[below]));
                ice = fminf(ice, memory);
                liquid = fmaxf(
                    zero,
                    __fsub_rn(middle_moisture, __fmul_rn(ice, riw)));
                fwsat[index] = __fsub_rn(dqm, liquid);
                lwsat[index] = __fadd_rn(liquid, qmin);
            }
            soiliqwm[index] = liquid;
            soilicem[index] = ice;
        } else {
            soiliqwm[index] = middle_moisture;
            lwsat[index] = maximum;
        }
    }

    if (update_smfrkeep) {
        for (int level = 0; level < RUC_NZS; ++level) {
            int index = level * ncolumn + column;
            smfrkeep[index] = soilice[index] > zero
                ? soilice[index]
                : __fdiv_rn(soilmois[index], riw);
        }
    }
}


// Canopy fractions, initial dew, and dry-soil resistance immediately before
// WRF's transpiration and heat solves.
extern "C" __global__
void ruc_soil_canopy_setup(
    const real* __restrict__ soilmois,
    const real* __restrict__ qvatm_a,
    const real* __restrict__ qsg_a,
    const real* __restrict__ qvg_a,
    const real* __restrict__ qkms_a,
    const real* __restrict__ cst_a,
    const real* __restrict__ sat_a,
    const real* __restrict__ cn_a,
    const real* __restrict__ qmin_a,
    const real* __restrict__ reference_a,
    real* __restrict__ dew_out,
    real* __restrict__ wetcan_out,
    real* __restrict__ drycan_out,
    real* __restrict__ soilres_out,
    int ncolumn)
{
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= ncolumn) return;

    const real zero = 0.0f;
    const real one = 1.0f;
    const real quarter = 0.25f;
    real dew = zero;
    if (qvatm_a[column] >= qsg_a[column]) {
        dew = __fmul_rn(
            qkms_a[column],
            __fsub_rn(qvatm_a[column], qsg_a[column]));
    }
    real ratio = __fdiv_rn(cst_a[column], sat_a[column]);
    ratio = fmaxf(zero, ratio);
    // float64 evaluated and rounded once, matching the host transcription,
    // ruc_snow_soil_canopy_setup below, and every other ** in this file.  The
    // CUDA device libm powf and numpy's float32 pow are two DIFFERENT
    // non-glibc functions -- they disagree on 6.3% of the arguments this site
    // spans -- so leaving one on each side is a divergence waiting for a wide
    // enough grid.  See _RUC_PROVISIONAL_TRANSCENDENTALS.
    real wetcan = fminf(quarter, ruc_powf_rn(ratio, cn_a[column]));
    real drycan = __fsub_rn(one, wetcan);

    real fc = fmaxf(
        qmin_a[column], __fmul_rn(reference_a[column], 0.5f));
    real total_top = __fadd_rn(soilmois[column], qmin_a[column]);
    real soilres;
    if (total_top > fc
            || __fsub_rn(qvatm_a[column], qvg_a[column]) > zero) {
        soilres = one;
    } else {
        real fex = __fdiv_rn(total_top, fc);
        fex = fmaxf(0.01f, fminf(one, fex));
        real resistance = __fsub_rn(
            one, cosf(__fmul_rn(3.141592653589793f, fex)));
        resistance = __fmul_rn(resistance, resistance);
        soilres = __fmul_rn(quarter, resistance);
    }
    dew_out[column] = dew;
    wetcan_out[column] = wetcan;
    drycan_out[column] = drycan;
    soilres_out[column] = soilres;
}


// Deterministic module_sf_ruclsm.F:soilprop.  Profiles are stored in gpuwm's
// soil-first order, so level k for column idx is k*ncolumn + idx.
extern "C" __global__
void ruc_soil_properties(
    const real* __restrict__ fwsat,
    const real* __restrict__ lwsat,
    const real* __restrict__ tav,
    const real* __restrict__ keepfr,
    const real* __restrict__ soilmois,
    const real* __restrict__ soiliqw,
    const real* __restrict__ soilice,
    const real* __restrict__ soilmoism,
    const real* __restrict__ soiliqwm,
    const real* __restrict__ soilicem,
    const real* __restrict__ qwrtz_a,
    const real* __restrict__ rhocs_a,
    const real* __restrict__ dqm_a,
    const real* __restrict__ qmin_a,
    const real* __restrict__ psis_a,
    const real* __restrict__ bclh_a,
    const real* __restrict__ ksat_a,
    real riw,
    real* __restrict__ thdif,
    real* __restrict__ diffu,
    real* __restrict__ hydro,
    real* __restrict__ cap,
    int ncolumn)
{
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= ncolumn) return;

    const real xlmelt = 3.35e5f;
    const real heat_air = 1004.5f;
    const real gravity = 9.81f;
    const real heat_water = 4.183e6f;
    const real heat_ice = 1.89e6f;
    const real conductivity_quartz = 7.7f;
    const real conductivity_ice = 2.2f;
    const real conductivity_water = 0.57f;
    const real minimum = 1.0e-8f;

    real qwrtz = qwrtz_a[column];
    real rhocs = rhocs_a[column];
    real dqm = dqm_a[column];
    real qmin = qmin_a[column];
    real psis = psis_a[column];
    real bclh = bclh_a[column];
    real ksat = ksat_a[column];
    real ws = __fadd_rn(dqm, qmin);
    real x1 = __fdiv_rn(xlmelt, __fmul_rn(gravity, psis));
    real x2 = __fmul_rn(__fdiv_rn(x1, bclh), ws);
    real x4 = __fdiv_rn(__fadd_rn(bclh, 1.0f), bclh);
    real gamd = __fmul_rn(__fsub_rn(1.0f, ws), 2700.0f);
    real kdry = __fdiv_rn(
        __fadd_rn(__fmul_rn(0.135f, gamd), 64.7f),
        __fsub_rn(2700.0f, __fmul_rn(0.947f, gamd)));
    real mineral = qwrtz > 0.2f ? 2.0f : 3.0f;
    real kas = __fmul_rn(
        ruc_powf_rn(conductivity_quartz, qwrtz),
        ruc_powf_rn(mineral, __fsub_rn(1.0f, qwrtz)));

    for (int level = 0; level < RUC_NZS; ++level) {
        int index = level * ncolumn + column;
        thdif[index] = 0.0f;
        diffu[index] = 0.0f;
        hydro[index] = 0.0f;
        cap[index] = 0.0f;
    }

    for (int level = 0; level < RUC_NZS_M1; ++level) {
        int index = level * ncolumn + column;
        real middle_temperature = tav[index];
        real tn = __fsub_rn(middle_temperature, 273.15f);
        real middle_ice = soilicem[index];
        real wd = __fsub_rn(ws, __fmul_rn(riw, middle_ice));
        real middle_liquid = soiliqwm[index];
        real first_ratio = __fdiv_rn(
            wd, __fadd_rn(middle_liquid, qmin));
        real psif = __fmul_rn(
            __fmul_rn(__fmul_rn(psis, 100.0f), ruc_powf_rn(first_ratio, bclh)),
            ruc_powf_rn(__fdiv_rn(ws, wd), 3.0f));
        real pf = ruc_log10f_rn(fabsf(psif));
        // WRF also evaluates the legacy McCumber conductivity here, but the
        // active Johansen path below is the only consumer of conductivity.

        real detal = 0.0f;
        if (middle_ice != 0.0f && tn < 0.0f) {
            detal = __fmul_rn(
                __fdiv_rn(
                    __fmul_rn(273.15f, x2),
                    __fmul_rn(middle_temperature, middle_temperature)),
                ruc_powf_rn(
                    __fdiv_rn(
                        middle_temperature, __fmul_rn(x1, tn)),
                    x4));
            if (keepfr[index] == 1.0f) detal = 0.0f;
        }

        real kasat = ruc_powf_rn(kas, __fsub_rn(1.0f, ws));
        kasat = __fmul_rn(kasat, ruc_powf_rn(conductivity_ice, fwsat[index]));
        kasat = __fmul_rn(kasat, ruc_powf_rn(conductivity_water, lwsat[index]));
        real middle_moisture = soilmoism[index];
        real x5 = __fdiv_rn(__fadd_rn(middle_moisture, qmin), ws);
        real ke = middle_ice == 0.0f
            ? __fadd_rn(ruc_log10f_rn(fmaxf(0.101f, x5)), 1.0f)
            : x5;
        real kjpl = __fadd_rn(
            __fmul_rn(ke, __fsub_rn(kasat, kdry)), kdry);

        real capacity = __fmul_rn(__fsub_rn(1.0f, ws), rhocs);
        capacity = __fadd_rn(
            capacity,
            __fmul_rn(__fadd_rn(middle_liquid, qmin), heat_water));
        capacity = __fadd_rn(capacity, __fmul_rn(middle_ice, heat_ice));
        capacity = __fadd_rn(
            capacity,
            __fmul_rn(
                __fmul_rn(__fsub_rn(dqm, middle_moisture), heat_air),
                1.2f));
        capacity = __fsub_rn(
            capacity, __fmul_rn(__fmul_rn(detal, 1.0e3f), xlmelt));
        cap[index] = capacity;

        real ice = __fmul_rn(riw, middle_ice);
        real diffusivity = 0.0f;
        if (__fsub_rn(ws, ice) >= 0.12f) {
            real h = fmaxf(
                0.0f,
                __fdiv_rn(
                    __fsub_rn(__fadd_rn(middle_moisture, qmin), ice),
                    fmaxf(minimum, __fsub_rn(ws, ice))));
            real facd = 1.0f;
            if (ice != 0.0f) {
                facd = __fsub_rn(
                    1.0f, __fdiv_rn(ice, fmaxf(minimum, middle_moisture)));
            }
            real ame = fmaxf(minimum, __fsub_rn(ws, ice));
            diffusivity = __fmul_rn(__fmul_rn(-bclh, ksat), psis);
            diffusivity = __fdiv_rn(diffusivity, ame);
            diffusivity = __fmul_rn(
                diffusivity, ruc_powf_rn(__fdiv_rn(ws, ame), 3.0f));
            diffusivity = __fmul_rn(
                diffusivity, ruc_powf_rn(h, __fadd_rn(bclh, 2.0f)));
            diffusivity = __fmul_rn(diffusivity, facd);
        }
        diffu[index] = diffusivity;
        thdif[index] = __fdiv_rn(kjpl, capacity);
    }

    for (int level = 0; level < RUC_NZS; ++level) {
        int index = level * ncolumn + column;
        real level_ice = soilice[index];
        real ice = __fmul_rn(riw, level_ice);
        if (__fsub_rn(ws, ice) < 0.12f) continue;
        real fach = 1.0f;
        if (level_ice != 0.0f) {
            fach = __fsub_rn(
                1.0f,
                __fdiv_rn(ice, fmaxf(minimum, soilmois[index])));
        }
        real am = fmaxf(minimum, __fsub_rn(ws, ice));
        real conductivity = __fdiv_rn(ksat, am);
        real exponent = __fadd_rn(__fmul_rn(2.0f, bclh), 2.0f);
        conductivity = __fmul_rn(
            conductivity,
            ruc_powf_rn(__fdiv_rn(soiliqw[index], am), exponent));
        conductivity = __fmul_rn(conductivity, fach);
        conductivity = fminf(ksat, conductivity);
        if (conductivity < 1.0e-10f) conductivity = 0.0f;
        hydro[index] = conductivity;
    }
}


// module_sf_ruclsm.F:transf for WRF's per-land-class 1..8-level root zone.
extern "C" __global__
void ruc_transpiration(
    const real* __restrict__ soiliqw,
    const real* __restrict__ tabs,
    const real* __restrict__ lai,
    const real* __restrict__ gswin,
    const real* __restrict__ dqm,
    const real* __restrict__ qmin,
    const real* __restrict__ reference,
    const real* __restrict__ wilt,
    const real* __restrict__ pc,
    const int* __restrict__ iland,
    const real* __restrict__ rstbl,
    const real* __restrict__ rgltbl,
    const real* __restrict__ zshalf,
    const int* __restrict__ nroot_a,
    real rsmax,
    real* __restrict__ tranf,
    real* __restrict__ transum,
    int ncolumn)
{
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= ncolumn) return;
    (void)dqm;
    int nroot = nroot_a[column];

    for (int level = 0; level < RUC_NZS; ++level) {
        tranf[level * ncolumn + column] = 0.0f;
    }
    real minimum_moisture = qmin[column];
    real field_capacity = reference[column];
    real wilting = wilt[column];
    for (int level = 0; level < nroot; ++level) {
        int index = level * ncolumn + column;
        real total_liquid = __fadd_rn(soiliqw[index], minimum_moisture);
        real depth = level == 0
            ? zshalf[1]
            : __fsub_rn(zshalf[level + 1], zshalf[level]);
        bool saturated = level == 0
            ? total_liquid > field_capacity
            : total_liquid >= field_capacity;
        real weight;
        if (saturated) {
            weight = depth;
        } else if (total_liquid <= wilting) {
            weight = 0.0f;
        } else {
            weight = __fmul_rn(
                __fdiv_rn(
                    __fsub_rn(total_liquid, wilting),
                    __fsub_rn(field_capacity, wilting)),
                depth);
        }
        tranf[index] = weight;
    }

    real leaf_area = lai[column];
    real pctot = leaf_area > 4.0f ? 0.8f : pc[column];
    real temperature = tabs[column];
    real exponent = temperature <= 302.15f
        ? __fmul_rn(-0.41f, __fsub_rn(temperature, 282.05f))
        : __fmul_rn(0.5f, __fsub_rn(temperature, 314.0f));
    real ftem = __fdiv_rn(1.0f, __fadd_rn(1.0f, ruc_expf_rn(exponent)));

    int vegetation_index = iland[column] - 1;
    real resistance = rstbl[vegetation_index];
    real light_threshold = rgltbl[vegetation_index];
    real cmin = __fdiv_rn(1.0f, rsmax);
    real cmax = __fdiv_rn(1.0f, resistance);
    if (leaf_area > 1.0f) cmax = __fdiv_rn(leaf_area, resistance);
    real fsol;
    if (gswin[column] < light_threshold) {
        real light_exponent = __fmul_rn(
            -0.034f, __fsub_rn(gswin[column], 3.5f));
        fsol = __fdiv_rn(
            1.0f, __fadd_rn(1.0f, ruc_expf_rn(light_exponent)));
    } else {
        fsol = 1.0f;
    }
    real conductance = __fsub_rn(cmax, cmin);
    conductance = __fmul_rn(conductance, pctot);
    conductance = __fmul_rn(conductance, ftem);
    conductance = __fmul_rn(conductance, fsol);
    conductance = __fadd_rn(cmin, conductance);
    conductance = __fdiv_rn(conductance, cmax);

    real total = 0.0f;
    for (int level = 0; level < nroot; ++level) {
        int index = level * ncolumn + column;
        real weight = fmaxf(cmin, __fmul_rn(tranf[index], conductance));
        tranf[index] = weight;
        total = __fadd_rn(total, weight);
    }
    transum[column] = total;
}


// Convert WRF's root weights and post-soiltemp humidity into the profile
// extraction flux consumed by soilmoist.  It also pins the no-snow water
// inputs used by the assembled land-column lane.
extern "C" __global__
void ruc_soil_prepare_moisture(
    const real* __restrict__ qvatm_a,
    const real* __restrict__ qsg_a,
    const real* __restrict__ qkms_a,
    const real* __restrict__ rho_a,
    const real* __restrict__ vegfrac_a,
    const real* __restrict__ drycan_a,
    const real* __restrict__ tranf,
    const int* __restrict__ nroot_a,
    const real* __restrict__ infwater_a,
    real* __restrict__ transp_out,
    real* __restrict__ ett1_out,
    real* __restrict__ dew_out,
    real* __restrict__ prcp_out,
    real* __restrict__ ras_out,
    int ncolumn)
{
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= ncolumn) return;

    const real zero = 0.0f;
    const real* zsmain = ruc_soil_layer_depth;
    real zshalf[RUC_NZS];
    zshalf[0] = zero;
    for (int level = 1; level < RUC_NZS; ++level) {
        zshalf[level] = __fmul_rn(
            __fadd_rn(zsmain[level - 1], zsmain[level]), 0.5f);
    }
    for (int level = 0; level < RUC_NZS; ++level) {
        transp_out[level * ncolumn + column] = zero;
    }

    real ett1 = zero;
    real dew = zero;
    real ras = __fmul_rn(rho_a[column], 1.0e-3f);
    if (qvatm_a[column] >= qsg_a[column]) {
        dew = __fmul_rn(
            qkms_a[column],
            __fsub_rn(qvatm_a[column], qsg_a[column]));
    } else {
        int nroot = nroot_a[column];
        for (int level = 0; level < nroot; ++level) {
            int index = level * ncolumn + column;
            real flux = __fmul_rn(vegfrac_a[column], ras);
            flux = __fmul_rn(flux, qkms_a[column]);
            flux = __fmul_rn(
                flux, __fsub_rn(qvatm_a[column], qsg_a[column]));
            flux = __fmul_rn(flux, tranf[index]);
            flux = __fmul_rn(flux, drycan_a[column]);
            flux = __fdiv_rn(flux, zshalf[nroot]);
            if (flux > zero) flux = zero;
            transp_out[index] = flux;
            ett1 = __fsub_rn(ett1, flux);
        }
    }
    ett1_out[column] = ett1;
    dew_out[column] = dew;
    prcp_out[column] = -infwater_a[column];
    ras_out[column] = ras;
}


// Full nine-level module_sf_ruclsm.F:soilmoist implicit water solve.
extern "C" __global__
void ruc_soil_moisture_step(
    const real* __restrict__ diffu,
    const real* __restrict__ hydro,
    const real* __restrict__ transp,
    const real* __restrict__ soilice,
    const real* __restrict__ soilmois_in,
    const real* __restrict__ soiliqw_in,
    const real* __restrict__ qsg_a,
    const real* __restrict__ qvg_a,
    const real* __restrict__ qcg_a,
    const real* __restrict__ qcatm_a,
    const real* __restrict__ qvatm_a,
    const real* __restrict__ prcp_a,
    const real* __restrict__ qkms_a,
    const real* __restrict__ drip_a,
    const real* __restrict__ dew_a,
    const real* __restrict__ smelt_a,
    const real* __restrict__ vegfrac_a,
    const real* __restrict__ snowfrac_a,
    const real* __restrict__ soilres_a,
    const real* __restrict__ dqm_a,
    const real* __restrict__ qmin_a,
    const real* __restrict__ reference_a,
    const real* __restrict__ ksat_a,
    const real* __restrict__ ras_a,
    real delt,
    real* __restrict__ soilmois_out,
    real* __restrict__ soiliqw_out,
    real* __restrict__ mavail_out,
    real* __restrict__ runoff_out,
    real* __restrict__ runoff2_out,
    real* __restrict__ infiltrp_out,
    real* __restrict__ infmax_out,
    int ncolumn)
{
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= ncolumn) return;

    const real one = 1.0f;
    const real minimum = 1.0e-8f;
    const real* zsmain = ruc_soil_layer_depth;
    real zshalf[RUC_NZS];
    real dtdzs[RUC_DTDZS_LEN];
    real dtdzs2[RUC_NZS];
    zshalf[0] = 0.0f;
    for (int level = 1; level < RUC_NZS; ++level) {
        zshalf[level] = __fmul_rn(
            __fadd_rn(zsmain[level - 1], zsmain[level]), 0.5f);
    }
    for (int index = 0; index < RUC_DTDZS_LEN; ++index) dtdzs[index] = 0.0f;
    for (int index = 0; index < RUC_NZS; ++index) dtdzs2[index] = 0.0f;
    for (int fortran_level = 2; fortran_level < RUC_NZS; ++fortran_level) {
        int first = 2 * fortran_level - 3;
        int second = first + 1;
        int level = fortran_level - 1;
        real x = __fdiv_rn(
            __fdiv_rn(delt, 2.0f),
            __fsub_rn(zshalf[level + 1], zshalf[level]));
        dtdzs[first - 1] = __fdiv_rn(
            x, __fsub_rn(zsmain[level], zsmain[level - 1]));
        dtdzs2[level - 1] = x;
        dtdzs[second - 1] = __fdiv_rn(
            x, __fsub_rn(zsmain[level + 1], zsmain[level]));
    }

    for (int level = 0; level < RUC_NZS; ++level) {
        int index = level * ncolumn + column;
        soilmois_out[index] = soilmois_in[index];
        soiliqw_out[index] = soiliqw_in[index];
    }
    real cosmc[RUC_NZS] = {0.0f};
    real rhsmc[RUC_NZS] = {0.0f};
    cosmc[0] = 0.0f;
    rhsmc[0] = soilmois_in[RUC_NZS_M1 * ncolumn + column];
    for (int step = 1; step < RUC_NZS_M1; ++step) {
        int kn = RUC_NZS - step;
        int first = 2 * kn - 3;
        real x4 = __fmul_rn(
            __fmul_rn(2.0f, dtdzs[first - 1]),
            diffu[(kn - 2) * ncolumn + column]);
        real x2 = __fmul_rn(
            __fmul_rn(2.0f, dtdzs[first]),
            diffu[(kn - 1) * ncolumn + column]);
        real q4 = __fadd_rn(
            x4,
            __fmul_rn(
                hydro[(kn - 2) * ncolumn + column], dtdzs2[kn - 2]));
        real q2 = __fsub_rn(
            x2,
            __fmul_rn(
                hydro[kn * ncolumn + column], dtdzs2[kn - 2]));
        real denominator = __fadd_rn(1.0f, x2);
        denominator = __fadd_rn(denominator, x4);
        denominator = __fsub_rn(
            denominator, __fmul_rn(q2, cosmc[step - 1]));
        cosmc[step] = __fdiv_rn(q4, denominator);
        real rhs = __fadd_rn(
            soilmois_in[(kn - 1) * ncolumn + column],
            __fmul_rn(q2, rhsmc[step - 1]));
        real root_flux = __fdiv_rn(
            transp[(kn - 1) * ncolumn + column],
            __fsub_rn(zshalf[kn], zshalf[kn - 1]));
        rhs = __fadd_rn(rhs, __fmul_rn(root_flux, delt));
        rhsmc[step] = __fdiv_rn(rhs, denominator);
    }

    real vegfrac = vegfrac_a[column];
    real umveg = __fmul_rn(__fsub_rn(one, vegfrac), soilres_a[column]);
    real runoff = 0.0f;
    real runoff2 = 0.0f;
    real dzs = zsmain[1];
    real r1 = cosmc[RUC_NZS_M2];
    real r2 = rhsmc[RUC_NZS_M2];
    real r3 = __fdiv_rn(diffu[column], dzs);
    real r4 = __fadd_rn(r3, __fmul_rn(hydro[column], 0.5f));
    real r5 = __fsub_rn(
        r3, __fmul_rn(hydro[ncolumn + column], 0.5f));
    real r6 = __fmul_rn(qkms_a[column], ras_a[column]);
    real total_liquid = prcp_a[column];
    total_liquid = __fsub_rn(
        total_liquid, __fdiv_rn(drip_a[column], delt));
    total_liquid = __fsub_rn(
        total_liquid,
        __fmul_rn(__fmul_rn(umveg, dew_a[column]), ras_a[column]));
    total_liquid = __fsub_rn(total_liquid, smelt_a[column]);
    real flux = total_liquid;

    real dqm = dqm_a[column];
    real qmin = qmin_a[column];
    real reference = reference_a[column];
    real ksat = ksat_a[column];
    real delt1 = __fdiv_rn(delt, 86400.0f);
    real f1max = __fmul_rn(dqm, zshalf[1]);
    real free_storage = __fmul_rn(
        f1max,
        __fsub_rn(
            one, __fdiv_rn(soilmois_in[column], dqm)));
    real frozen_depth = __fmul_rn(soilice[column], zshalf[1]);
    for (int level = 1; level < RUC_NZS_M1; ++level) {
        int index = level * ncolumn + column;
        real thickness = __fsub_rn(zshalf[level + 1], zshalf[level]);
        frozen_depth = __fadd_rn(
            frozen_depth, __fmul_rn(soilice[index], thickness));
        real maximum = __fmul_rn(dqm, thickness);
        real available = __fmul_rn(
            maximum,
            __fsub_rn(one, __fdiv_rn(soilmois_in[index], dqm)));
        free_storage = __fadd_rn(free_storage, available);
    }
    real kdt = __fdiv_rn(__fmul_rn(3.0f, ksat), 3.4341e-6f);
    real infiltration_fraction = __fsub_rn(
        one, ruc_expf_rn(__fmul_rn(-kdt, delt1)));
    real ddt = __fmul_rn(free_storage, infiltration_fraction);
    real water_input = __fmul_rn(-total_liquid, delt);
    if (water_input < 0.0f) water_input = 0.0f;
    real infmax1 = 0.0f;
    if (water_input > 0.0f) {
        infmax1 = __fdiv_rn(
            __fmul_rn(
                water_input,
                __fdiv_rn(ddt, __fadd_rn(water_input, ddt))),
            delt);
    }
    real frzx = __fmul_rn(
        0.15f, __fdiv_rn(__fadd_rn(dqm, qmin), reference));
    frzx = __fmul_rn(frzx, __fdiv_rn(0.412f, 0.468f));
    real frozen_factor = 1.0f;
    if (frozen_depth > 1.0e-2f) {
        real acrt = __fdiv_rn(__fmul_rn(3.0f, frzx), frozen_depth);
        // ``SUM = SUM + (ACRT ** (CVFRZ-JK)) / FLOAT(K)`` at
        // module_sf_ruclsm.F:5983.  CVFRZ is declared ``real`` at :5826 and
        // set to 3. at :5936, so ``CVFRZ-JK`` is REAL and gfortran lowers the
        // whole thing to glibc powf -- it does NOT expand to a multiplication
        // the way Noah-MP's INTEGER PARAMETER CVFRZ does.  Neither float32
        // pow available here is glibc's, so this site takes the same
        // float64-evaluated, rounded-once form as every other ** in this file
        // and in the host transcription (see _RUC_PROVISIONAL_TRANSCENDENTALS
        // and gpuwm/core/ruc.py:1533); for an exponent of 2 that form is the
        // exactly-rounded square, because a float32 squared is exact in
        // float64.
        //
        // This was the CUDA device libm ``powf`` until 2026-07-26.  It differs
        // from the host by 1 ULP on the acrt a frozen column reaches (1.03 to
        // 1.47 on a snow grid), and the ``1 - exp(-acrt)*SUM`` cancellation at
        // :5985 amplifies that into 10 ULP of ``infiltr``.  No fixture could
        // see it: three of soilmoist.csv's four cases carry soilice = 0 so
        // :5972 is false, and the fourth multiplies the result by an
        // ``infmax1`` of exactly zero.  See
        // tests/test_ruc_gpu.py::test_ruc_soil_moisture_cuda_agrees_off_fixture.
        real series = __fadd_rn(
            1.0f, __fdiv_rn(ruc_powf_rn(acrt, 2.0f), 2.0f));
        series = __fadd_rn(series, acrt);
        frozen_factor = __fsub_rn(
            1.0f, __fmul_rn(ruc_expf_rn(-acrt), series));
    }
    infmax1 = __fmul_rn(infmax1, frozen_factor);
    real infmax = fmaxf(
        infmax1, __fmul_rn(hydro[column], soilmois_in[column]));
    // Fortran MIN preserves the first +0 operand when the upper bound is
    // -0; fminf would manufacture -0 and break exact WRF state identity.
    real infiltration_upper = -total_liquid;
    if (infiltration_upper < infmax) infmax = infiltration_upper;
    if (-total_liquid > infmax) {
        runoff = __fsub_rn(-total_liquid, infmax);
        flux = -infmax;
    }
    real infiltrp = flux;

    real r7 = __fdiv_rn(__fmul_rn(0.5f, dzs), delt);
    r4 = __fadd_rn(r4, r7);
    flux = __fsub_rn(
        flux, __fmul_rn(soilmois_in[column], r7));
    real r8 = __fmul_rn(umveg, r6);
    r8 = __fmul_rn(r8, __fsub_rn(one, snowfrac_a[column]));
    real qtot = __fadd_rn(qvatm_a[column], qcatm_a[column]);
    real r9 = transp[column];
    real r10 = __fsub_rn(qtot, qsg_a[column]);
    real denominator, numerator, candidate, saturated_flux;
    if (r10 <= 0.0f) {
        denominator = __fsub_rn(r4, __fmul_rn(r5, r1));
        denominator = __fsub_rn(
            denominator,
            __fdiv_rn(
                __fmul_rn(r10, r8), __fsub_rn(reference, qmin)));
        numerator = __fsub_rn(__fmul_rn(r5, r2), flux);
        numerator = __fadd_rn(numerator, r9);
        candidate = __fdiv_rn(numerator, denominator);
        saturated_flux = __fmul_rn(-dqm, denominator);
        saturated_flux = __fadd_rn(
            saturated_flux, __fmul_rn(r5, r2));
        saturated_flux = __fadd_rn(saturated_flux, r9);
    } else {
        denominator = __fsub_rn(r4, __fmul_rn(r1, r5));
        real humidity = __fsub_rn(qtot, qcg_a[column]);
        humidity = __fsub_rn(humidity, qvg_a[column]);
        numerator = __fsub_rn(__fmul_rn(r2, r5), flux);
        numerator = __fadd_rn(numerator, __fmul_rn(r8, humidity));
        numerator = __fadd_rn(numerator, r9);
        candidate = __fdiv_rn(numerator, denominator);
        saturated_flux = __fmul_rn(-dqm, denominator);
        saturated_flux = __fadd_rn(
            saturated_flux, __fmul_rn(r2, r5));
        saturated_flux = __fadd_rn(
            saturated_flux, __fmul_rn(r8, humidity));
        saturated_flux = __fadd_rn(saturated_flux, r9);
    }
    if (candidate < 0.0f) {
        soilmois_out[column] = minimum;
    } else if (candidate > dqm) {
        soilmois_out[column] = dqm;
        runoff = __fadd_rn(
            runoff, __fsub_rn(saturated_flux, flux));
    } else {
        soilmois_out[column] = fminf(dqm, fmaxf(minimum, candidate));
    }

    for (int level = 1; level < RUC_NZS; ++level) {
        int index = level * ncolumn + column;
        int previous = (level - 1) * ncolumn + column;
        int coefficient = RUC_NZS_M1 - level;
        candidate = __fadd_rn(
            __fmul_rn(cosmc[coefficient], soilmois_out[previous]),
            rhsmc[coefficient]);
        if (candidate < 0.0f) {
            soilmois_out[index] = minimum;
        } else if (candidate > dqm) {
            soilmois_out[index] = dqm;
            real thickness = level == RUC_NZS_M1
                ? __fsub_rn(zsmain[level], zshalf[level])
                : __fsub_rn(zshalf[level + 1], zshalf[level]);
            runoff2 = __fadd_rn(
                runoff2,
                __fdiv_rn(
                    __fmul_rn(__fsub_rn(candidate, dqm), thickness),
                    delt));
        } else {
            soilmois_out[index] = fminf(dqm, fmaxf(minimum, candidate));
        }
    }
    real availability = __fdiv_rn(
        soilmois_out[column], __fsub_rn(reference, qmin));
    availability = __fmul_rn(
        availability, __fsub_rn(one, snowfrac_a[column]));
    availability = __fadd_rn(availability, snowfrac_a[column]);
    mavail_out[column] = fmaxf(0.00001f, fminf(one, availability));
    runoff_out[column] = runoff;
    runoff2_out[column] = runoff2;
    infiltrp_out[column] = infiltrp;
    infmax_out[column] = infmax;
}


// module_sf_ruclsm.F:vilka.  Returns false instead of calling WRF's fatal
// handler when an invalid surface state leaves the pinned lookup table.
__device__ __forceinline__
bool ruc_vilka(
    real tn,
    real d1,
    real d2,
    real pp,
    const real* __restrict__ table,
    real* qs,
    real* ts)
{
    real raw_index = __fsub_rn(tn, 173.15f);
    raw_index = __fdiv_rn(raw_index, 0.05f);
    raw_index = __fadd_rn(raw_index, 1.0f);
    int index = __float2int_rz(raw_index);
    if (index < 1 || index > 5000) return false;

    real t1 = __fadd_rn(
        173.1f, __fmul_rn(__int2float_rn(index), 0.05f));
    real delta = __fsub_rn(table[index], table[index - 1]);
    real f1 = __fadd_rn(t1, __fmul_rn(d1, table[index - 1]));
    f1 = __fsub_rn(f1, d2);
    real denominator = __fadd_rn(0.05f, __fmul_rn(d1, delta));
    real ratio = __fdiv_rn(f1, denominator);
    index = __float2int_rz(
        __fsub_rn(__int2float_rn(index), ratio));
    if (index < 1 || index > 5000) return false;

    real rn = 0.0f;
    bool converged = false;
    for (int iteration = 0; iteration < 5001; ++iteration) {
        int previous = index;
        t1 = __fadd_rn(
            173.1f, __fmul_rn(__int2float_rn(index), 0.05f));
        delta = __fsub_rn(table[index], table[index - 1]);
        f1 = __fadd_rn(t1, __fmul_rn(d1, table[index - 1]));
        f1 = __fsub_rn(f1, d2);
        denominator = __fadd_rn(0.05f, __fmul_rn(d1, delta));
        rn = __fdiv_rn(f1, denominator);
        index -= __float2int_rz(rn);
        if (index < 1 || index > 5000) return false;
        if (index == previous) {
            converged = true;
            break;
        }
    }
    if (!converged) return false;

    *ts = __fsub_rn(t1, __fmul_rn(0.05f, rn));
    real numerator = __fadd_rn(
        table[index - 1],
        __fmul_rn(
            __fsub_rn(table[index - 1], table[index]), rn));
    *qs = __fdiv_rn(numerator, pp);
    return true;
}


// Full snow-free module_sf_ruclsm.F:soiltemp heat and skin solve.
extern "C" __global__
void ruc_soil_temperature_step(
    const real* __restrict__ thdif,
    const real* __restrict__ cap,
    const real* __restrict__ tso_in,
    const real* __restrict__ prcpms_a,
    const real* __restrict__ rainf_a,
    const real* __restrict__ patm_a,
    const real* __restrict__ tabs_a,
    const real* __restrict__ qvatm_a,
    const real* __restrict__ emiss_a,
    const real* __restrict__ rnet_a,
    const real* __restrict__ qkms_a,
    const real* __restrict__ tkms_a,
    const real* __restrict__ rho_a,
    const real* __restrict__ vegfrac_a,
    const real* __restrict__ drycan_a,
    const real* __restrict__ wetcan_a,
    const real* __restrict__ transum_a,
    const real* __restrict__ mavail_a,
    const real* __restrict__ soilres_a,
    const real* __restrict__ soilt_in,
    const real* __restrict__ qvg_in,
    const int* __restrict__ nroot_a,
    const real* __restrict__ tbq,
    real delt,
    // ``0.5*dz8w(i,1,j)`` -- half the lowest model layer, which is a
    // PER-COLUMN depth on a terrain-following coordinate.  WRF passes it
    // as a scalar only because ``sfctmp`` is called from inside DO j/DO i.
    const real* __restrict__ conflx,
    real cvw,
    real* __restrict__ tso_out,
    real* __restrict__ soilt_out,
    real* __restrict__ qvg_out,
    real* __restrict__ qsg_out,
    real* __restrict__ qcg_out,
    real* __restrict__ storage_out,
    int ncolumn)
{
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= ncolumn) return;

    const real one = 1.0f;
    const real half = 0.5f;
    const real cp_air = 1004.5f;
    const real xlv = 2.5e6f;
    const real stbolt = 5.67051e-8f;
    const real freeze = 273.15f;
    const real* zsmain = ruc_soil_layer_depth;
    real zshalf[RUC_NZS];
    real dtdzs[RUC_DTDZS_LEN];
    zshalf[0] = 0.0f;
    for (int level = 1; level < RUC_NZS; ++level) {
        zshalf[level] = __fmul_rn(
            __fadd_rn(zsmain[level - 1], zsmain[level]), half);
    }
    for (int index = 0; index < RUC_DTDZS_LEN; ++index) dtdzs[index] = 0.0f;
    for (int fortran_level = 2; fortran_level < RUC_NZS; ++fortran_level) {
        int first = 2 * fortran_level - 3;
        int second = first + 1;
        int level = fortran_level - 1;
        real x = __fdiv_rn(
            __fdiv_rn(delt, 2.0f),
            __fsub_rn(zshalf[level + 1], zshalf[level]));
        dtdzs[first - 1] = __fdiv_rn(
            x, __fsub_rn(zsmain[level], zsmain[level - 1]));
        dtdzs[second - 1] = __fdiv_rn(
            x, __fsub_rn(zsmain[level + 1], zsmain[level]));
    }

    real cotso[RUC_NZS] = {0.0f};
    real rhtso[RUC_NZS] = {0.0f};
    rhtso[0] = tso_in[RUC_NZS_M1 * ncolumn + column];
    for (int step = 0; step < RUC_NZS_M2; ++step) {
        int kn = RUC_NZS_M1 - step;
        int first = 2 * kn - 3;
        real x1 = __fmul_rn(
            dtdzs[first - 1], thdif[(kn - 2) * ncolumn + column]);
        real x2 = __fmul_rn(
            dtdzs[first], thdif[(kn - 1) * ncolumn + column]);
        real ft = __fadd_rn(
            tso_in[(kn - 1) * ncolumn + column],
            __fmul_rn(
                x1,
                __fsub_rn(
                    tso_in[(kn - 2) * ncolumn + column],
                    tso_in[(kn - 1) * ncolumn + column])));
        ft = __fsub_rn(
            ft,
            __fmul_rn(
                x2,
                __fsub_rn(
                    tso_in[(kn - 1) * ncolumn + column],
                    tso_in[kn * ncolumn + column])));
        real denominator = __fadd_rn(one, x1);
        denominator = __fadd_rn(denominator, x2);
        denominator = __fsub_rn(
            denominator, __fmul_rn(x2, cotso[step]));
        cotso[step + 1] = __fdiv_rn(x1, denominator);
        rhtso[step + 1] = __fdiv_rn(
            __fadd_rn(ft, __fmul_rn(x2, rhtso[step])), denominator);
    }

    real rhcs = cap[column];
    real h = mavail_a[column];
    real trans = __fdiv_rn(
        __fmul_rn(transum_a[column], drycan_a[column]),
        zshalf[nroot_a[column]]);
    real can = __fadd_rn(wetcan_a[column], trans);
    real umveg = __fmul_rn(
        __fsub_rn(one, vegfrac_a[column]), soilres_a[column]);
    real d1 = cotso[RUC_NZS_M2];
    real d2 = rhtso[RUC_NZS_M2];
    real tn = soilt_in[column];
    real qgold = qvg_in[column];
    real dzstop = __fdiv_rn(
        one, __fsub_rn(zsmain[1], zsmain[0]));
    real d9 = __fmul_rn(
        __fmul_rn(thdif[column], rhcs), dzstop);
    real d10 = __fmul_rn(
        __fmul_rn(tkms_a[column], cp_air), rho_a[column]);
    real r211 = __fdiv_rn(__fmul_rn(half, conflx[column]), delt);
    real r21 = __fmul_rn(__fmul_rn(r211, cp_air), rho_a[column]);
    real depth_square = __fmul_rn(dzstop, dzstop);
    real denominator = __fmul_rn(
        __fmul_rn(thdif[column], delt), depth_square);
    real r22 = __fdiv_rn(half, denominator);
    real tn2 = __fmul_rn(tn, tn);
    real tn4 = __fmul_rn(tn2, tn2);
    real r6 = __fmul_rn(
        __fmul_rn(__fmul_rn(emiss_a[column], stbolt), half), tn4);
    real r7 = __fdiv_rn(r6, tn);
    real d11 = __fadd_rn(rnet_a[column], r6);
    real tdenom = __fmul_rn(
        d9, __fadd_rn(__fsub_rn(one, d1), r22));
    tdenom = __fadd_rn(tdenom, d10);
    tdenom = __fadd_rn(tdenom, r21);
    tdenom = __fadd_rn(tdenom, r7);
    real rain_heat_capacity = __fmul_rn(
        __fmul_rn(rainf_a[column], cvw), prcpms_a[column]);
    tdenom = __fadd_rn(tdenom, rain_heat_capacity);
    real fkq = __fmul_rn(qkms_a[column], rho_a[column]);
    real r210 = __fmul_rn(r211, rho_a[column]);
    real c = __fmul_rn(
        __fmul_rn(vegfrac_a[column], fkq), can);
    real cc = __fdiv_rn(__fmul_rn(c, xlv), tdenom);
    real aa_inner = __fadd_rn(__fmul_rn(fkq, umveg), r210);
    real aa = __fdiv_rn(__fmul_rn(xlv, aa_inner), tdenom);

    real humidity_inner = __fadd_rn(__fmul_rn(fkq, umveg), c);
    humidity_inner = __fmul_rn(qvatm_a[column], humidity_inner);
    humidity_inner = __fadd_rn(
        humidity_inner, __fmul_rn(r210, qgold));
    real numerator = __fmul_rn(d10, tabs_a[column]);
    numerator = __fadd_rn(numerator, __fmul_rn(r21, tn));
    numerator = __fadd_rn(numerator, __fmul_rn(xlv, humidity_inner));
    numerator = __fadd_rn(numerator, d11);
    real conduction = __fadd_rn(d2, __fmul_rn(r22, tn));
    numerator = __fadd_rn(numerator, __fmul_rn(d9, conduction));
    real rain_temperature = fmaxf(freeze, tabs_a[column]);
    numerator = __fadd_rn(
        numerator, __fmul_rn(rain_heat_capacity, rain_temperature));
    real bb = __fdiv_rn(numerator, tdenom);
    real aa1 = __fadd_rn(aa, cc);
    real pp = __fmul_rn(patm_a[column], 1.0e3f);
    aa1 = __fdiv_rn(aa1, pp);
    real qs1, ts1;
    bool valid = ruc_vilka(tn, aa1, bb, pp, tbq, &qs1, &ts1);
    real tx2 = __fmul_rn(qvatm_a[column], __fsub_rn(one, h));
    real q1 = __fadd_rn(tx2, __fmul_rn(h, qs1));
    real qvg, qsg, qcg;
    if (valid && q1 >= qs1) {
        qvg = qs1;
        qsg = qs1;
        qcg = fmaxf(0.0f, __fsub_rn(q1, qs1));
    } else if (valid) {
        bb = __fsub_rn(bb, __fmul_rn(aa, tx2));
        aa = __fdiv_rn(__fadd_rn(__fmul_rn(aa, h), cc), pp);
        valid = ruc_vilka(tn, aa, bb, pp, tbq, &qs1, &ts1);
        q1 = __fadd_rn(tx2, __fmul_rn(h, qs1));
        if (valid && q1 >= qs1) {
            qvg = qs1;
            qsg = qs1;
            qcg = fmaxf(0.0f, __fsub_rn(q1, qs1));
        } else if (valid) {
            qsg = qs1;
            qvg = q1;
            qcg = 0.0f;
        }
    }
    if (!valid) {
        real invalid = nanf("");
        for (int level = 0; level < RUC_NZS; ++level) {
            tso_out[level * ncolumn + column] = invalid;
        }
        soilt_out[column] = invalid;
        qvg_out[column] = invalid;
        qsg_out[column] = invalid;
        qcg_out[column] = invalid;
        storage_out[column] = invalid;
        return;
    }

    tso_out[column] = ts1;
    for (int level = 1; level < RUC_NZS; ++level) {
        int coefficient = RUC_NZS_M1 - level;
        tso_out[level * ncolumn + column] = __fadd_rn(
            rhtso[coefficient],
            __fmul_rn(
                cotso[coefficient],
                tso_out[(level - 1) * ncolumn + column]));
    }
    real storage_coefficient = __fadd_rn(
        __fmul_rn(__fmul_rn(cp_air, rho_a[column]), r211),
        __fdiv_rn(
            __fmul_rn(__fmul_rn(rhcs, zsmain[1]), half), delt));
    real storage = __fmul_rn(
        storage_coefficient, __fsub_rn(ts1, tn));
    storage = __fadd_rn(
        storage,
        __fmul_rn(
            __fmul_rn(__fmul_rn(xlv, rho_a[column]), r211),
            __fsub_rn(qvg, qgold)));
    storage = __fsub_rn(
        storage,
        __fmul_rn(
            rain_heat_capacity, __fsub_rn(rain_temperature, ts1)));
    soilt_out[column] = ts1;
    qvg_out[column] = qvg;
    qsg_out[column] = qsg;
    qcg_out[column] = qcg;
    storage_out[column] = storage;
}


// Final snow-free WRF soil bookkeeping and surface-flux diagnostics.  The
// prognostic heat and water solves remain in their separately oracled kernels;
// this kernel preserves the source ordering around their returned values.
extern "C" __global__
void ruc_soil_finalize(
    const real* __restrict__ soilice,
    const real* __restrict__ tso,
    const real* __restrict__ told,
    const real* __restrict__ soilmois,
    const real* __restrict__ smold,
    real* __restrict__ keepfr,
    const real* __restrict__ thdif,
    const real* __restrict__ cap,
    const real* __restrict__ cst_in,
    const real* __restrict__ dew,
    const real* __restrict__ soilt,
    const real* __restrict__ qvg,
    const real* __restrict__ qsg,
    const real* __restrict__ qcg,
    const real* __restrict__ ett1_in,
    const real* __restrict__ wetcan,
    const real* __restrict__ soilres,
    const real* __restrict__ ras,
    const real* __restrict__ tkms,
    const real* __restrict__ rho,
    const real* __restrict__ tabs,
    const real* __restrict__ patm,
    const real* __restrict__ qkms,
    const real* __restrict__ qvatm,
    const real* __restrict__ vegfrac,
    const real* __restrict__ rnet,
    const real* __restrict__ prcpms,
    const real* __restrict__ storage,
    real delt,
    real* __restrict__ cst_out,
    real* __restrict__ edir1_out,
    real* __restrict__ ec1_out,
    real* __restrict__ ett1_out,
    real* __restrict__ eeta_out,
    real* __restrict__ qfx_out,
    real* __restrict__ hfx_out,
    real* __restrict__ s_out,
    real* __restrict__ evapl_out,
    real* __restrict__ prcpl_out,
    real* __restrict__ fltot_out,
    real* __restrict__ smf_out,
    int ncolumn)
{
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= ncolumn) return;

    const real zero = 0.0f;
    const real one = 1.0f;
    const real xlv = 2.5e6f;
    const real cp_air = 1004.5f;
    const real rovcp = __fdiv_rn(287.0f, cp_air);
// >>> RUC_NZS DZSTOP >>>
    // The ONE soil literal in this file that is a DEPTH rather than an
    // extent, and the one the RUC_NZS sweep therefore walked past: it was
    // `__fsub_rn(0.01f, 0.0f)`, WRF's nine-level zsmain(2) - zsmain(1)
    // written out.  Correct while nine was the only geometry; at six levels
    // zsmain(2) - zsmain(1) is 0.05, so this kernel divided by 0.01 and
    // returned a ground heat flux five times too large -- MEASURED, before
    // the fix, as grdflx = -337.1 W m-2 against the host lane's -67.4 on
    // the same column, exactly a factor of 5.  Every other site in this
    // file already reads the table (:1667, :2702, :3495); this one now
    // does too, which is all the fix is.
    //
    // At nine levels ruc_soil_layer_depth[1] IS 0.01f and [0] IS 0.00f, so
    // the subtraction, the divide and every number downstream of them are
    // unchanged -- tests/test_ruc_nzs_device.py measures that on the
    // hardware.  The PTX is not unchanged (a __constant__ load where an
    // immediate was), which is why the sentinel exists: it keeps
    // tests/test_ruc_nzs_tier.py's inversion to the pre-lift file exact,
    // and keeps the ladder's own no-op claim separate from this fix's.
    const real* zsmain = ruc_soil_layer_depth;
    const real dzstop = __fdiv_rn(one, __fsub_rn(zsmain[1], zsmain[0]));
// <<< RUC_NZS DZSTOP <<<

    for (int level = 0; level < RUC_NZS; ++level) {
        int index = level * ncolumn + column;
        if (soilice[index] > zero) {
            keepfr[index] = tso[index] > told[index]
                    && soilmois[index] > smold[index]
                ? one : zero;
        }
    }

    real hft = __fmul_rn(tkms[column], cp_air);
    hft = __fmul_rn(hft, rho[column]);
    hft = __fmul_rn(-hft, __fsub_rn(tabs[column], soilt[column]));
    // Same substitution, and this is the site that was MEASURED to diverge:
    // hfx moved 1 ULP on 1 of 512 warm columns and 2 ULP on 13 of 4,096,
    // which no 64-column gate could see.  See _RUC_PROVISIONAL_TRANSCENDENTALS.
    real pressure_factor = ruc_powf_rn(
        __fdiv_rn(one, patm[column]), rovcp);
    real hfx = __fmul_rn(hft, pressure_factor);
    real q1 = __fmul_rn(qkms[column], ras[column]);
    q1 = __fmul_rn(-q1, __fsub_rn(qvatm[column], qsg[column]));

    real cst = cst_in[column];
    real edir1;
    real ec1;
    real ett1 = ett1_in[column];
    real eeta;
    if (q1 <= zero) {
        ec1 = zero;
        edir1 = zero;
        ett1 = zero;
        eeta = -__fmul_rn(rho[column], dew[column]);
        real canopy_dew = __fmul_rn(delt, dew[column]);
        canopy_dew = __fmul_rn(canopy_dew, ras[column]);
        canopy_dew = __fmul_rn(canopy_dew, vegfrac[column]);
        cst = __fadd_rn(cst, canopy_dew);
    } else {
        edir1 = __fmul_rn(
            soilres[column], __fsub_rn(one, vegfrac[column]));
        edir1 = __fmul_rn(edir1, qkms[column]);
        edir1 = __fmul_rn(edir1, ras[column]);
        edir1 = __fmul_rn(
            -edir1, __fsub_rn(qvatm[column], qvg[column]));
        ec1 = __fmul_rn(q1, wetcan[column]);
        ec1 = __fmul_rn(ec1, vegfrac[column]);
        cst = fmaxf(zero, __fsub_rn(cst, __fmul_rn(ec1, delt)));
        real total_evaporation = __fadd_rn(edir1, ec1);
        total_evaporation = __fadd_rn(total_evaporation, ett1);
        eeta = __fmul_rn(total_evaporation, 1.0e3f);
    }

    real soil_heat = __fmul_rn(thdif[column], cap[column]);
    soil_heat = __fmul_rn(soil_heat, dzstop);
    soil_heat = __fmul_rn(
        soil_heat, __fsub_rn(tso[column], tso[ncolumn + column]));
    real balance = __fsub_rn(rnet[column], hft);
    balance = __fsub_rn(balance, __fmul_rn(xlv, eeta));
    balance = __fsub_rn(balance, soil_heat);
    balance = __fsub_rn(balance, storage[column]);

    cst_out[column] = cst;
    edir1_out[column] = edir1;
    ec1_out[column] = ec1;
    ett1_out[column] = ett1;
    eeta_out[column] = eeta;
    qfx_out[column] = __fmul_rn(xlv, eeta);
    hfx_out[column] = hfx;
    s_out[column] = soil_heat;
    evapl_out[column] = eeta;
    prcpl_out[column] = prcpms[column];
    fltot_out[column] = balance;
    smf_out[column] = zero;
}


// module_sf_ruclsm.F:2202-2225 function qsn: the tbq saturation lookup.
// WRF's INT() truncates toward zero, reproduced by __float2int_rz.  Below
// the first node (:2212-2214) the index is forced to 1 and the raw weight
// to 1., so the result collapses onto t(1); at or past the last node
// (:2215-2217) the index becomes 5000 with a raw weight of 5001., which
// collapses onto t(5001).  The final form (:2220) is a*b+c, the exact shape
// NVRTC would contract into an FMA, so it is pinned term by term.
__device__ __forceinline__
real ruc_qsn_lookup(real tn, const real* __restrict__ table)
{
    real raw = __fsub_rn(tn, 173.15f);
    raw = __fdiv_rn(raw, 0.05f);
    raw = __fadd_rn(raw, 1.0f);
    int index = __float2int_rz(raw);
    if (index < 1) {
        index = 1;
        raw = 1.0f;
    }
    if (index > 5000) {
        index = 5000;
        raw = 5001.0f;
    }
    real lower = table[index - 1];
    real weight = __fsub_rn(raw, __int2float_rn(index));
    real value = __fsub_rn(table[index], lower);
    value = __fmul_rn(value, weight);
    return __fadd_rn(value, lower);
}


// One thread evaluates one module_sf_ruclsm.F:2202-2225 qsn lookup.
extern "C" __global__
void ruc_qsn(
    const real* __restrict__ tn,
    const real* __restrict__ tbq,
    real* __restrict__ qsn_out,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    qsn_out[idx] = ruc_qsn_lookup(tn[idx], tbq);
}


// module_sf_ruclsm.F:2853-3115 subroutine sice: the snow-free sea-ice
// column.  One thread carries one independent column: the nine-level heat
// diffusion sweep (:2977-2989), the surface energy balance closed by vilka
// (:2991-3028), the 271.4 K melting cap on every level (:3032, :3038-3041)
// and the surface flux diagnostics (:3045-3108).
//
// WRF's t3/upflux/xinet block (:3046-3048) is dead - xinet is assigned and
// never read - so glw never reaches an output and is not an argument here.
// qcatm, gsw, tice, rhosice, zshalf, dtdzs2, nroot and xlv are likewise
// never read by sice, and the incoming qcg is overwritten at :3033 before
// any use.
//
// fltot (:3108) subtracts icemelt (:3103) from the identical expression
// that produced it, so the FP32 cancellation is exact and fltot is zero.
extern "C" __global__
void ruc_sea_ice_step(
    const real* __restrict__ capice,
    const real* __restrict__ thdifice,
    const real* __restrict__ tso_in,
    const real* __restrict__ prcpms_a,
    const real* __restrict__ rainf_a,
    const real* __restrict__ patm_a,
    const real* __restrict__ qvatm_a,
    const real* __restrict__ emiss_a,
    const real* __restrict__ rnet_a,
    const real* __restrict__ qkms_a,
    const real* __restrict__ tkms_a,
    const real* __restrict__ rho_a,
    const real* __restrict__ tabs_a,
    const real* __restrict__ soilt_in,
    const real* __restrict__ qvg_in,
    const real* __restrict__ qsg_in,
    const real* __restrict__ tbq,
    real delt,
    // ``0.5*dz8w(i,1,j)`` -- half the lowest model layer, which is a
    // PER-COLUMN depth on a terrain-following coordinate.  WRF passes it
    // as a scalar only because ``sfctmp`` is called from inside DO j/DO i.
    const real* __restrict__ conflx,
    real cvw,
    int myj,
    real* __restrict__ tso_out,
    real* __restrict__ dew_out,
    real* __restrict__ soilt_out,
    real* __restrict__ qvg_out,
    real* __restrict__ qsg_out,
    real* __restrict__ qcg_out,
    real* __restrict__ eeta_out,
    real* __restrict__ qfx_out,
    real* __restrict__ hfx_out,
    real* __restrict__ s_out,
    real* __restrict__ evapl_out,
    real* __restrict__ prcpl_out,
    real* __restrict__ fltot_out,
    int ncolumn)
{
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= ncolumn) return;

    const real zero = 0.0f;
    const real one = 1.0f;
    const real half = 0.5f;
    // cp and rovcp arrive from module_model_constants; the pinned RUC lane
    // fixes them at these values and the oracle carries the same bytes.
    const real cp_air = 1004.5f;
    const real rovcp = __fdiv_rn(287.0f, cp_air);
    // module_model_constants XLS, the latent heat of sublimation.
    const real xls = 2.85e6f;
    const real stbolt = 5.67051e-8f;
    const real freeze = 273.15f;
    // module_sf_ruclsm.F:3032 - sice caps every ice temperature here and
    // does not model the melt itself.
    const real ice_cap = 271.4f;
    // p1000mb*0.00001 from the :3050-3051 Exner factor.
    const real reference = __fmul_rn(100000.0f, 0.00001f);

    // share/module_soil_pre.F:1153-1194 nine-level RUC depths, and the
    // lsmruc-side zshalf/dtdzs construction sice consumes.
    const real* zsmain = ruc_soil_layer_depth;
    real zshalf[RUC_NZS];
    real dtdzs[RUC_DTDZS_LEN];
    zshalf[0] = zero;
    for (int level = 1; level < RUC_NZS; ++level) {
        zshalf[level] = __fmul_rn(
            __fadd_rn(zsmain[level - 1], zsmain[level]), half);
    }
    for (int index = 0; index < RUC_DTDZS_LEN; ++index) dtdzs[index] = zero;
    for (int fortran_level = 2; fortran_level < RUC_NZS; ++fortran_level) {
        int first = 2 * fortran_level - 3;
        int second = first + 1;
        int level = fortran_level - 1;
        real x = __fdiv_rn(
            __fdiv_rn(delt, 2.0f),
            __fsub_rn(zshalf[level + 1], zshalf[level]));
        dtdzs[first - 1] = __fdiv_rn(
            x, __fsub_rn(zsmain[level], zsmain[level - 1]));
        dtdzs[second - 1] = __fdiv_rn(
            x, __fsub_rn(zsmain[level + 1], zsmain[level]));
    }

    // module_sf_ruclsm.F:2964-2967 prologue.
    real prcpl = prcpms_a[column];
    real ras = __fmul_rn(rho_a[column], 1.0e-3f);
    real dzstop = __fdiv_rn(one, __fsub_rn(zsmain[1], zsmain[0]));

    // module_sf_ruclsm.F:2973-2989 upward tridiagonal sweep.
    real cotso[RUC_NZS] = {0.0f};
    real rhtso[RUC_NZS] = {0.0f};
    rhtso[0] = tso_in[RUC_NZS_M1 * ncolumn + column];
    for (int step = 0; step < RUC_NZS_M2; ++step) {
        int kn = RUC_NZS_M1 - step;
        int first = 2 * kn - 3;
        real x1 = __fmul_rn(
            dtdzs[first - 1], thdifice[(kn - 2) * ncolumn + column]);
        real x2 = __fmul_rn(
            dtdzs[first], thdifice[(kn - 1) * ncolumn + column]);
        real ft = __fadd_rn(
            tso_in[(kn - 1) * ncolumn + column],
            __fmul_rn(
                x1,
                __fsub_rn(
                    tso_in[(kn - 2) * ncolumn + column],
                    tso_in[(kn - 1) * ncolumn + column])));
        ft = __fsub_rn(
            ft,
            __fmul_rn(
                x2,
                __fsub_rn(
                    tso_in[(kn - 1) * ncolumn + column],
                    tso_in[kn * ncolumn + column])));
        real denominator = __fadd_rn(one, x1);
        denominator = __fadd_rn(denominator, x2);
        denominator = __fsub_rn(
            denominator, __fmul_rn(x2, cotso[step]));
        cotso[step + 1] = __fdiv_rn(x1, denominator);
        rhtso[step + 1] = __fdiv_rn(
            __fadd_rn(ft, __fmul_rn(x2, rhtso[step])), denominator);
    }

    // module_sf_ruclsm.F:2991-3016 heat balance (Smirnova et al. 1996).
    real rhcs = capice[column];
    real d1 = cotso[RUC_NZS_M2];
    real d2 = rhtso[RUC_NZS_M2];
    real tn = soilt_in[column];
    real d9 = __fmul_rn(
        __fmul_rn(thdifice[column], rhcs), dzstop);
    real d10 = __fmul_rn(
        __fmul_rn(tkms_a[column], cp_air), rho_a[column]);
    real r211 = __fdiv_rn(__fmul_rn(half, conflx[column]), delt);
    real r21 = __fmul_rn(__fmul_rn(r211, cp_air), rho_a[column]);
    real depth_square = __fmul_rn(dzstop, dzstop);
    real denominator = __fmul_rn(
        __fmul_rn(thdifice[column], delt), depth_square);
    real r22 = __fdiv_rn(half, denominator);
    real tn2 = __fmul_rn(tn, tn);
    real tn4 = __fmul_rn(tn2, tn2);
    real r6 = __fmul_rn(
        __fmul_rn(__fmul_rn(emiss_a[column], stbolt), half), tn4);
    real r7 = __fdiv_rn(r6, tn);
    real d11 = __fadd_rn(rnet_a[column], r6);
    real tdenom = __fmul_rn(
        d9, __fadd_rn(__fsub_rn(one, d1), r22));
    tdenom = __fadd_rn(tdenom, d10);
    tdenom = __fadd_rn(tdenom, r21);
    tdenom = __fadd_rn(tdenom, r7);
    real rain_heat_capacity = __fmul_rn(
        __fmul_rn(rainf_a[column], cvw), prcpms_a[column]);
    tdenom = __fadd_rn(tdenom, rain_heat_capacity);
    real fkq = __fmul_rn(qkms_a[column], rho_a[column]);
    real r210 = __fmul_rn(r211, rho_a[column]);
    real aa = __fdiv_rn(__fmul_rn(xls, __fadd_rn(fkq, r210)), tdenom);

    real humidity_inner = __fmul_rn(qvatm_a[column], fkq);
    humidity_inner = __fadd_rn(
        humidity_inner, __fmul_rn(r210, qvg_in[column]));
    real numerator = __fmul_rn(d10, tabs_a[column]);
    numerator = __fadd_rn(numerator, __fmul_rn(r21, tn));
    numerator = __fadd_rn(numerator, __fmul_rn(xls, humidity_inner));
    numerator = __fadd_rn(numerator, d11);
    real conduction = __fadd_rn(d2, __fmul_rn(r22, tn));
    numerator = __fadd_rn(numerator, __fmul_rn(d9, conduction));
    real rain_temperature = fmaxf(freeze, tabs_a[column]);
    numerator = __fadd_rn(
        numerator, __fmul_rn(rain_heat_capacity, rain_temperature));
    real bb = __fdiv_rn(numerator, tdenom);
    real aa1 = aa;
    real pp = __fmul_rn(patm_a[column], 1.0e3f);
    aa1 = __fdiv_rn(aa1, pp);

    // module_sf_ruclsm.F:3027-3033.  Sea ice is always saturated, so sice
    // takes the single vilka solution without soiltemp's h-weighted retry.
    real qgold = qsg_in[column];
    real qs1, ts1;
    if (!ruc_vilka(tn, aa1, bb, pp, tbq, &qs1, &ts1)) {
        real invalid = nanf("");
        for (int level = 0; level < RUC_NZS; ++level) {
            tso_out[level * ncolumn + column] = invalid;
        }
        dew_out[column] = invalid;
        soilt_out[column] = invalid;
        qvg_out[column] = invalid;
        qsg_out[column] = invalid;
        qcg_out[column] = invalid;
        eeta_out[column] = invalid;
        qfx_out[column] = invalid;
        hfx_out[column] = invalid;
        s_out[column] = invalid;
        evapl_out[column] = invalid;
        prcpl_out[column] = invalid;
        fltot_out[column] = invalid;
        return;
    }
    real qvg = qs1;
    real qsg = qs1;
    tso_out[column] = fminf(ice_cap, ts1);
    real qcg = zero;
    real soilt = tso_out[column];

    // module_sf_ruclsm.F:3037-3041 downward substitution, capped level by
    // level exactly as WRF caps it.
    for (int level = 1; level < RUC_NZS; ++level) {
        int coefficient = RUC_NZS_M1 - level;
        tso_out[level * ncolumn + column] = fminf(
            ice_cap,
            __fadd_rn(
                rhtso[coefficient],
                __fmul_rn(
                    cotso[coefficient],
                    tso_out[(level - 1) * ncolumn + column])));
    }
    real dew = zero;

    // module_sf_ruclsm.F:3049-3052 surface flux diagnostics.
    real hft = __fmul_rn(tkms_a[column], cp_air);
    hft = __fmul_rn(hft, rho_a[column]);
    hft = __fmul_rn(hft, __fsub_rn(tabs_a[column], soilt));
    // Same substitution as ruc_soil_finalize, and as ruc_snow_sea_ice_step
    // below; see _RUC_PROVISIONAL_TRANSCENDENTALS.
    real exner = ruc_powf_rn(__fdiv_rn(reference, patm_a[column]), rovcp);
    real hfx = -__fmul_rn(hft, exner);
    hft = -hft;
    real q1 = __fmul_rn(qkms_a[column], ras);
    q1 = -__fmul_rn(q1, __fsub_rn(qvatm_a[column], qsg));

    // module_sf_ruclsm.F:3053-3089.  Both branches recompute eeta after
    // qfx, so the qfx-side value never leaves the branch.
    real eeta;
    real qfx;
    if (q1 <= zero) {
        if (myj) {
            eeta = __fmul_rn(qkms_a[column], ras);
            eeta = -__fmul_rn(
                eeta,
                __fsub_rn(
                    __fdiv_rn(
                        qvatm_a[column],
                        __fadd_rn(one, qvatm_a[column])),
                    __fdiv_rn(qsg, __fadd_rn(one, qsg))));
            eeta = __fmul_rn(eeta, 1.0e3f);
        } else {
            dew = __fmul_rn(
                qkms_a[column], __fsub_rn(qvatm_a[column], qsg));
            eeta = -__fmul_rn(rho_a[column], dew);
        }
        qfx = __fmul_rn(xls, eeta);
        eeta = -__fmul_rn(rho_a[column], dew);
    } else {
        if (myj) {
            eeta = __fmul_rn(qkms_a[column], ras);
            eeta = -__fmul_rn(
                eeta,
                __fsub_rn(
                    __fdiv_rn(
                        qvatm_a[column],
                        __fadd_rn(one, qvatm_a[column])),
                    __fdiv_rn(qvg, __fadd_rn(one, qvg))));
            eeta = __fmul_rn(eeta, 1.0e3f);
        } else {
            eeta = __fmul_rn(q1, 1.0e3f);
        }
        qfx = __fmul_rn(xls, eeta);
        eeta = __fmul_rn(q1, 1.0e3f);
    }
    real evapl = eeta;

    // module_sf_ruclsm.F:3092-3100 storage and the surface heat residual.
    real storage = __fmul_rn(
        __fmul_rn(thdifice[column], rhcs), dzstop);
    storage = __fmul_rn(
        storage,
        __fsub_rn(tso_out[column], tso_out[ncolumn + column]));
    real x = __fadd_rn(
        __fmul_rn(__fmul_rn(cp_air, rho_a[column]), r211),
        __fdiv_rn(
            __fmul_rn(__fmul_rn(rhcs, zsmain[1]), half), delt));
    x = __fmul_rn(x, __fsub_rn(soilt, tn));
    x = __fadd_rn(
        x,
        __fmul_rn(
            __fmul_rn(__fmul_rn(xls, rho_a[column]), r211),
            __fsub_rn(qsg, qgold)));
    x = __fsub_rn(
        x,
        __fmul_rn(
            rain_heat_capacity, __fsub_rn(rain_temperature, soilt)));

    // module_sf_ruclsm.F:3103 icemelt absorbs the entire residual, so the
    // :3108 fltot difference cancels to an exact zero.
    real residual = __fsub_rn(rnet_a[column], __fmul_rn(xls, eeta));
    residual = __fsub_rn(residual, hft);
    residual = __fsub_rn(residual, storage);
    residual = __fsub_rn(residual, x);
    real icemelt = residual;
    real fltot = __fsub_rn(residual, icemelt);

    dew_out[column] = dew;
    soilt_out[column] = soilt;
    qvg_out[column] = qvg;
    qsg_out[column] = qsg;
    qcg_out[column] = qcg;
    eeta_out[column] = eeta;
    qfx_out[column] = qfx;
    hfx_out[column] = hfx;
    s_out[column] = storage;
    evapl_out[column] = evapl;
    prcpl_out[column] = prcpl;
    fltot_out[column] = fltot;
}


// module_sf_ruclsm.F:63-69 sncovfac, read only when isncovr_opt==3.
__device__ static const real ruc_sncovfac[30] = {
    0.030f, 0.030f, 0.030f, 0.030f, 0.030f,
    0.016f, 0.016f, 0.020f, 0.020f, 0.020f,
    0.020f, 0.014f, 0.042f, 0.026f, 0.030f,
    0.016f, 0.030f, 0.030f, 0.030f, 0.030f,
    0.000f, 0.000f, 0.000f, 0.000f, 0.000f,
    0.000f, 0.000f, 0.000f, 0.000f, 0.000f
};

// gfortran emits calls to glibc expf/tanhf for EXP/TANH on a default REAL,
// and neither CUDA's expf nor CUDA's tanhf reproduces those bit for bit.
//
// The device's own expf is documented at 2 ULP, which sfctmp cannot tolerate:
// the snow compaction evaluates (exp(x)-1)/x for x ~ 2e-4 and turns 1 ULP of
// exp into ~7600 ULP of rhosn (module_sf_ruclsm.F:1497).  That much is
// measured and is why this shim exists.
//
// The name is a promise this function does not keep, and an earlier comment
// here made the promise explicit: "glibc's expf is correctly rounded, so
// rounding a double exp once reproduces it".  It is not.  glibc 2.39's expf
// is a float32 reduction with its own error, so rounding a double once gives
// a third function -- gpuwm/core/noahmp_libm.py measures the round-once shim
// missing glibc expf on 21,750 of 34,902,602 arguments.  What is true is
// that this matches the host's _f32_exp exactly, so CPU and GPU agree.
// Renaming it, and moving both onto the verified noahmp_libm transcription,
// is deferred: it changes results on arguments no fixture reaches, and no
// RUC fixture varies pressure, so the suite cannot referee it.
__device__ __forceinline__
real ruc_expf_glibc(real x)
{
    return (real)exp((double)x);
}

__device__ __forceinline__
real ruc_expm1f_glibc(real x)
{
    return (real)expm1((double)x);
}

// glibc's tanhf, unlike its expf, is still fdlibm's expm1-based reduction
// evaluated in float32 and is NOT correctly rounded - on this lane's fixture
// it lands 2 ULP above the correctly rounded value for one snow-fraction
// argument.  The reduction is therefore spelled out here exactly as in
// gpuwm.core.ruc._f32_tanh, so host and device share one definition instead of
// inheriting two different libm implementations.
__device__ __forceinline__
real ruc_tanhf_glibc(real x)
{
    const real one = 1.0f;
    const real two = 2.0f;
    real magnitude = fabsf(x);
    real z;
    if (magnitude < 22.0f) {
        if (magnitude < 3.7252902984619141e-09f) {
            // tanh(tiny) == tiny, in fdlibm's inexact-flag form.
            return __fmul_rn(x, __fadd_rn(one, x));
        }
        real doubled = __fmul_rn(two, magnitude);
        real t;
        if (magnitude >= one) {
            t = ruc_expm1f_glibc(doubled);
            z = __fsub_rn(one, __fdiv_rn(two, __fadd_rn(t, two)));
        } else {
            t = ruc_expm1f_glibc(-doubled);
            z = __fdiv_rn(-t, __fadd_rn(t, two));
        }
    } else {
        // fdlibm returns one-tiny here, which rounds to exactly one.
        z = one;
    }
    return (x >= 0.0f) ? z : -z;
}

// ruc_tanhf_glibc over a column field.
//
// module_sf_ruclsm.F:2087 and :2098 - the snow-cover rebuild inside sfctmp's
// melt-out block, which is in the DISPATCH and not in any leaf, so no leaf
// conversion reaches it.  On the host that TANH is gpuwm.core.ruc._f32_tanh,
// which spells fdlibm's reduction with Python control flow and is therefore
// SCALAR: a fully snow-covered column field pays one Python call per column
// there.  This is the same function, one thread per column, so the whole
// dispatch can stay on the card.
extern "C" __global__
void ruc_tanhf_glibc_array(
    const real* __restrict__ x,
    real* __restrict__ out,
    const int ncolumn)
{
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= ncolumn) {
        return;
    }
    out[column] = ruc_tanhf_glibc(x[column]);
}

// module_sf_ruclsm.F:1400-1766 - the straight-line snow-preparation prologue
// of subroutine sfctmp, stopping immediately before the :1767 dispatch to
// soil/snowsoil/sice/snowseaice.  One thread carries one independent column.
//
// Every arithmetic boundary is pinned with round-to-nearest intrinsics, so the
// result does not depend on whether the compiler is allowed to contract a*b+c
// into an FMA.
//
// keep_snow_albedo and snowfrac2 are locals WRF only assigns inside the
// snhei>0. branch (:1626, :1658); when that branch does not run WRF leaves
// them undefined and nothing reads them, so 0 is written instead of stack
// residue - matching the CPU port.
extern "C" __global__
void ruc_snow_preparation(
    const real* __restrict__ ts1d,
    const real* __restrict__ seaice_a,
    const real* __restrict__ gsw_a,
    const real* __restrict__ tabs_a,
    const real* __restrict__ tsnav_a,
    const real* __restrict__ prcpms_a,
    const real* __restrict__ newsnms_a,
    const real* __restrict__ vegfra_a,
    const real* __restrict__ lai_a,
    const real* __restrict__ sat_a,
    const real* __restrict__ soilt_a,
    const real* __restrict__ snowfallac_a,
    const real* __restrict__ alb_snow_a,
    const real* __restrict__ alb_snow_free_a,
    const real* __restrict__ snowrat_a,
    const real* __restrict__ grauprat_a,
    const real* __restrict__ icerat_a,
    const real* __restrict__ curat_a,
    const real* __restrict__ snwe_in,
    const real* __restrict__ snhei_in,
    const real* __restrict__ snowfrac_in,
    const real* __restrict__ rhosn_in,
    const real* __restrict__ rhosnfall_in,
    const real* __restrict__ cst_in,
    const real* __restrict__ alb_in,
    const real* __restrict__ emiss_in,
    const real* __restrict__ znt_in,
    const int* __restrict__ ivgtyp_a,
    const int* __restrict__ iland_in,
    const real* __restrict__ z0tbl,
    const real* __restrict__ lemitbl,
    real delt,
    real c1sn,
    real c2sn,
    int isice,
    int urban,
    int isncovr_opt,
    real* __restrict__ tice_out,
    real* __restrict__ rhosice_out,
    real* __restrict__ capice_out,
    real* __restrict__ thdifice_out,
    real* __restrict__ snhei_crit_out,
    real* __restrict__ snhei_crit_newsn_out,
    real* __restrict__ zntsn_out,
    real* __restrict__ snow_mosaic_out,
    real* __restrict__ snfr_out,
    real* __restrict__ newsn_out,
    real* __restrict__ newsnowratio_out,
    real* __restrict__ snowfracnewsn_out,
    real* __restrict__ rhonewsn_out,
    real* __restrict__ smelt_out,
    real* __restrict__ rainf_out,
    real* __restrict__ rsm_out,
    real* __restrict__ dd1_out,
    real* __restrict__ infiltr_out,
    real* __restrict__ vegfrac_out,
    real* __restrict__ drip_out,
    real* __restrict__ dripsn_out,
    real* __restrict__ dripliq_out,
    real* __restrict__ smf_out,
    real* __restrict__ interw_out,
    real* __restrict__ intersn_out,
    real* __restrict__ infwater_out,
    real* __restrict__ intwratio_out,
    real* __restrict__ gswnew_out,
    real* __restrict__ gswin_out,
    real* __restrict__ albice_out,
    real* __restrict__ albsn_out,
    real* __restrict__ emissn_out,
    real* __restrict__ emiss_snowfree_out,
    real* __restrict__ keep_snow_albedo_out,
    real* __restrict__ snowfrac2_out,
    real* __restrict__ snwe_out,
    real* __restrict__ snhei_out,
    real* __restrict__ snowfrac_out,
    real* __restrict__ rhosn_out,
    real* __restrict__ rhosnfall_out,
    real* __restrict__ cst_out,
    real* __restrict__ alb_out,
    real* __restrict__ emiss_out,
    real* __restrict__ znt_out,
    int* __restrict__ iland_out,
    int ncolumn)
{
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= ncolumn) return;

    const real zero = 0.0f;
    const real one = 1.0f;
    // module_model_constants rhowater.
    const real rhowater = 1000.0f;
    // :1418-1419 and :1493 fold their leading products at compile time in
    // default REAL; the same products are formed here.
    const real critical_depth_coefficient = __fmul_rn(0.01601f, rhowater);
    const real new_snow_depth_coefficient = __fmul_rn(0.0005f, rhowater);
    const real aging_depth_coefficient = __fmul_rn(0.0081f, 1.0e3f);
    const real freeze = 273.15f;
    const real snow_emissivity = 0.98f;
    // (273.15-263.15) in the :1735/:1761 albedo forms.
    const real albedo_span = __fsub_rn(273.15f, 263.15f);

    real seaice = seaice_a[column];
    real gsw = gsw_a[column];
    real tabs = tabs_a[column];
    real tsnav = tsnav_a[column];
    real prcpms = prcpms_a[column];
    real newsnms = newsnms_a[column];
    real vegfra = vegfra_a[column];
    real lai = lai_a[column];
    real sat = sat_a[column];
    real soilt = soilt_a[column];
    real snowfallac = snowfallac_a[column];
    real alb_snow = alb_snow_a[column];
    real alb_snow_free = alb_snow_free_a[column];
    real snowrat = snowrat_a[column];
    real grauprat = grauprat_a[column];
    real icerat = icerat_a[column];
    real curat = curat_a[column];
    real snwe = snwe_in[column];
    real snhei = snhei_in[column];
    real snowfrac = snowfrac_in[column];
    real rhosn = rhosn_in[column];
    real rhosnfall = rhosnfall_in[column];
    real cst = cst_in[column];
    real alb = alb_in[column];
    real emiss = emiss_in[column];
    real znt = znt_in[column];
    int ivgtyp = ivgtyp_a[column];
    int iland = iland_in[column];

    // :1418-1419
    real snhei_crit = __fdiv_rn(critical_depth_coefficient, rhosn);
    real snhei_crit_newsn = __fdiv_rn(new_snow_depth_coefficient, rhosn);
    // :1421 - assigned and never read again by sfctmp.
    real zntsn = z0tbl[isice - 1];
    // :1423-1434
    real snow_mosaic = zero;
    real snfr = one;
    real newsn = zero;
    real newsnowratio = zero;
    real snowfracnewsn = zero;
    real rhonewsn = 100.0f;
    if (snhei == zero) snowfrac = zero;
    real smelt = zero;
    real rainf = zero;
    real rsm = zero;
    real dd1 = zero;
    real infiltr = zero;
    // :1442-1449
    real vegfrac = __fmul_rn(0.01f, vegfra);
    real drip = zero;
    real dripsn = zero;
    real dripliq = zero;
    real smf = zero;
    real interw = zero;
    real intersn = zero;
    real infwater = zero;
    // :1452-1458 - the ice column starts at zero on every point.
    for (int level = 0; level < RUC_NZS; ++level) {
        int index = level * ncolumn + column;
        tice_out[index] = zero;
        rhosice_out[index] = zero;
        capice_out[index] = zero;
        thdifice_out[index] = zero;
    }
    // :1460-1465
    real gswnew = gsw;
    real gswin = __fdiv_rn(gsw, __fsub_rn(one, alb));
    real albice = alb_snow_free;
    real albsn = alb_snow;
    real emissn = snow_emissivity;
    real emiss_snowfree = lemitbl[ivgtyp - 1];

    // :1471-1485 sea-ice column properties (Zubov) and the ice albedo.
    if (seaice >= 0.5f) {
        for (int level = 0; level < RUC_NZS; ++level) {
            int index = level * ncolumn + column;
            real tice = __fsub_rn(ts1d[index], freeze);
            real rhosice = __fdiv_rn(
                917.6f, __fsub_rn(one, __fmul_rn(0.000165f, tice)));
            real cice = __fadd_rn(2115.85f, __fmul_rn(7.7948f, tice));
            real capice = __fmul_rn(cice, rhosice);
            tice_out[index] = tice;
            rhosice_out[index] = rhosice;
            capice_out[index] = capice;
            thdifice_out[index] = __fdiv_rn(2.260872f, capice);
        }
        real surface_tice = tice_out[column];
        albice = fminf(
            alb_snow_free,
            fmaxf(
                __fsub_rn(alb_snow_free, 0.05f),
                __fsub_rn(
                    alb_snow_free,
                    __fdiv_rn(
                        __fmul_rn(0.1f, __fadd_rn(surface_tice, 10.0f)),
                        10.0f))));
    }

    // :1493-1501 Koren et al. (1999) compaction of the existing pack.
    if (snhei > __fdiv_rn(aging_depth_coefficient, rhosn)) {
        real bsn = __fmul_rn(
            __fmul_rn(__fdiv_rn(delt, 3600.0f), c1sn),
            ruc_expf_glibc(
                __fsub_rn(
                    __fmul_rn(0.08f, fminf(zero, tsnav)),
                    __fmul_rn(__fmul_rn(c2sn, rhosn), 1.0e-3f))));
        real compaction = __fmul_rn(__fmul_rn(bsn, snwe), 100.0f);
        // :1496 - goto 777 leaves rhosn untouched.
        if (!(compaction < 1.0e-4f)) {
            real xsn = __fdiv_rn(
                __fmul_rn(rhosn, __fsub_rn(ruc_expf_glibc(compaction), one)),
                compaction);
            rhosn = fminf(fmaxf(58.8f, xsn), 500.0f);
        }
    }

    // :1504 - the mosaic flag from the previous step's snow fraction.
    if (snowfrac < 0.75f) snow_mosaic = one;

    // :1506-1540 fresh snowfall and its density.
    newsn = __fmul_rn(newsnms, delt);
    if (newsn > zero) {
        newsnowratio = fminf(one, __fdiv_rn(newsn, __fadd_rn(snwe, newsn)));
        rhonewsn = fminf(
            125.0f,
            __fdiv_rn(
                1000.0f,
                fmaxf(
                    8.0f,
                    __fmul_rn(
                        17.0f,
                        ruc_tanhf_glibc(
                            __fmul_rn(__fsub_rn(276.65f, tabs), 0.15f))))));
        real rhonewgr = fminf(
            500.0f,
            __fdiv_rn(
                rhowater,
                fmaxf(
                    2.0f,
                    __fmul_rn(
                        3.5f,
                        ruc_tanhf_glibc(
                            __fmul_rn(__fsub_rn(274.15f, tabs), 0.3333f))))));
        real rhonewice = rhonewsn;
        real weighted = __fmul_rn(rhonewsn, snowrat);
        weighted = __fadd_rn(weighted, __fmul_rn(rhonewgr, grauprat));
        weighted = __fadd_rn(weighted, __fmul_rn(rhonewice, icerat));
        weighted = __fadd_rn(weighted, __fmul_rn(rhonewgr, curat));
        rhosnfall = fminf(500.0f, fmaxf(58.8f, weighted));
        // :1531 - from here rhonewsn is the density of the falling frozen
        // precipitation, not of fresh snow alone.
        rhonewsn = rhosnfall;
        real xsn = __fdiv_rn(
            __fadd_rn(__fmul_rn(rhosn, snwe), __fmul_rn(rhonewsn, newsn)),
            __fadd_rn(snwe, newsn));
        rhosn = fminf(fmaxf(58.8f, xsn), 500.0f);
    }

    // :1542-1551
    if (prcpms != zero) rainf = one;

    // :1553-1578 canopy interception (Lawrence et al. 2006 CLM eq. 1).
    drip = zero;
    real intwratio = zero;
    if (vegfrac > 0.01f) {
        real canopy_gap = __fsub_rn(
            one, ruc_expf_glibc(__fmul_rn(-0.5f, lai)));
        interw = __fmul_rn(
            __fmul_rn(__fmul_rn(__fmul_rn(0.25f, delt), prcpms), canopy_gap),
            vegfrac);
        intersn = __fmul_rn(
            __fmul_rn(__fmul_rn(0.25f, newsn), canopy_gap), vegfrac);
        infwater = __fsub_rn(prcpms, __fdiv_rn(interw, delt));
        real intercepted = __fadd_rn(interw, intersn);
        if (intercepted > zero) intwratio = __fdiv_rn(interw, intercepted);
        dd1 = __fadd_rn(__fadd_rn(cst, interw), intersn);
        cst = dd1;
        if (cst > sat) {
            cst = sat;
            drip = __fsub_rn(dd1, sat);
        }
    } else {
        cst = zero;
        drip = zero;
        interw = zero;
        intersn = zero;
        infwater = prcpms;
    }

    // :1580-1598 fresh snow onto the ground.
    if (newsn > zero) {
        snwe = fmaxf(zero, __fsub_rn(__fadd_rn(snwe, newsn), intersn));
        if (drip > zero) {
            if (snow_mosaic == one) {
                dripliq = __fmul_rn(drip, intwratio);
                dripsn = __fsub_rn(drip, dripliq);
                snwe = __fadd_rn(snwe, dripsn);
                infwater = __fadd_rn(infwater, dripliq);
                dripliq = zero;
                dripsn = zero;
            } else {
                snwe = __fadd_rn(snwe, drip);
            }
        }
        snhei = __fdiv_rn(__fmul_rn(snwe, rhowater), rhosn);
        newsn = __fdiv_rn(__fmul_rn(newsn, rhowater), rhonewsn);
    }

    // WRF leaves both undefined when the snhei>0. branch does not run.
    real keep_snow_albedo = zero;
    real snowfrac2 = zero;

    if (snhei > zero) {
        // :1603 - snow-covered points use the snow/ice land-use class.
        iland = isice;
        if (isncovr_opt == 1) {
            snowfrac = fminf(
                one, __fdiv_rn(snhei, __fmul_rn(2.0f, snhei_crit)));
        } else if (isncovr_opt == 2) {
            snowfrac = fminf(
                one, __fdiv_rn(snhei, __fmul_rn(2.0f, snhei_crit)));
            // (rhosn/rhonewsn)**1. is the quotient itself; a real exponent of
            // exactly 1 is an identity, so no pow is issued.
            snowfrac2 = ruc_tanhf_glibc(
                __fdiv_rn(
                    snhei,
                    __fmul_rn(
                        __fmul_rn(2.5f, fminf(0.2f, znt)),
                        __fdiv_rn(rhosn, rhonewsn))));
            snowfrac = __fmul_rn(0.5f, __fadd_rn(snowfrac, snowfrac2));
        } else {
            // :1633 - m is the Noah-MP facsnf exponent, held at 1.
            snowfrac = ruc_tanhf_glibc(
                __fdiv_rn(
                    snhei,
                    __fmul_rn(
                        __fmul_rn(10.0f, ruc_sncovfac[ivgtyp - 1]),
                        __fdiv_rn(rhosn, rhonewsn))));
        }
        if (newsn > zero) {
            snowfracnewsn = fminf(
                one,
                __fdiv_rn(
                    __fmul_rn(snowfallac, 1.0e-3f), snhei_crit_newsn));
        }
        // :1645
        if (ivgtyp == urban) snowfrac = fminf(0.75f, snowfrac);
        // :1656
        if (snowfrac < 0.75f) snow_mosaic = one;
        // :1658-1663
        keep_snow_albedo = zero;
        if (snowfracnewsn > 0.99f && rhosnfall < 450.0f) {
            keep_snow_albedo = one;
            snow_mosaic = zero;
        }
        // :1672-1680 roughness blend toward the snow/ice class.
        if (newsn == zero && znt <= 0.2f && ivgtyp != isice) {
            real snow_roughness = z0tbl[iland - 1];
            if (snhei <= __fmul_rn(2.0f, znt)) {
                znt = __fadd_rn(
                    __fmul_rn(0.55f, znt), __fmul_rn(0.45f, snow_roughness));
            } else if (snhei > __fmul_rn(2.0f, znt)
                       && snhei <= __fmul_rn(4.0f, znt)) {
                znt = __fadd_rn(
                    __fmul_rn(0.2f, znt), __fmul_rn(0.8f, snow_roughness));
            } else if (snhei > __fmul_rn(4.0f, znt)) {
                znt = snow_roughness;
            }
        }

        if (seaice < 0.5f) {
            // :1682-1737 snow on soil.
            if (snow_mosaic == one) {
                albsn = alb_snow;
                // :1690-1696 is unreachable: :1662 clears snow_mosaic
                // whenever keep_snow_albedo is set, so this test can never
                // pass.  Transcribed to stay faithful.
                if (keep_snow_albedo > 0.9f && albsn < 0.4f) albsn = 0.7f;
                emiss = emissn;
            } else {
                albsn = fmaxf(
                    __fmul_rn(keep_snow_albedo, alb_snow),
                    fminf(
                        __fadd_rn(
                            alb_snow_free,
                            __fmul_rn(
                                __fsub_rn(alb_snow, alb_snow_free), snowfrac)),
                        alb_snow));
                if (newsn > zero && keep_snow_albedo > 0.9f && albsn < 0.4f) {
                    albsn = 0.7f;
                }
                emiss = fmaxf(
                    __fmul_rn(keep_snow_albedo, emissn),
                    fminf(
                        __fadd_rn(
                            emiss_snowfree,
                            __fmul_rn(
                                __fsub_rn(emissn, emiss_snowfree), snowfrac)),
                        emissn));
            }
            if (albsn < 0.4f || keep_snow_albedo == one) {
                alb = albsn;
            } else {
                alb = fminf(
                    albsn,
                    fmaxf(
                        __fsub_rn(
                            albsn,
                            __fmul_rn(
                                __fdiv_rn(
                                    __fmul_rn(
                                        0.1f, __fsub_rn(soilt, 263.15f)),
                                    albedo_span),
                                albsn)),
                        __fsub_rn(albsn, 0.05f)));
            }
        } else {
            // :1738-1765 snow on ice.
            if (snow_mosaic == one) {
                albsn = alb_snow;
                emiss = emissn;
            } else {
                albsn = fmaxf(
                    __fmul_rn(keep_snow_albedo, alb_snow),
                    fminf(
                        __fadd_rn(
                            albice,
                            __fmul_rn(__fsub_rn(alb_snow, albice), snowfrac)),
                        alb_snow));
                emiss = fmaxf(
                    __fmul_rn(keep_snow_albedo, emissn),
                    fminf(
                        __fadd_rn(
                            emiss_snowfree,
                            __fmul_rn(
                                __fsub_rn(emissn, emiss_snowfree), snowfrac)),
                        emissn));
            }
            if (albsn < alb_snow || keep_snow_albedo == one) {
                alb = albsn;
            } else {
                alb = fminf(
                    albsn,
                    fmaxf(
                        __fsub_rn(
                            albsn,
                            __fdiv_rn(
                                __fmul_rn(
                                    __fmul_rn(0.15f, albsn),
                                    __fsub_rn(soilt, 263.15f)),
                                albedo_span)),
                        __fsub_rn(albsn, 0.1f)));
            }
        }
    }

    snhei_crit_out[column] = snhei_crit;
    snhei_crit_newsn_out[column] = snhei_crit_newsn;
    zntsn_out[column] = zntsn;
    snow_mosaic_out[column] = snow_mosaic;
    snfr_out[column] = snfr;
    newsn_out[column] = newsn;
    newsnowratio_out[column] = newsnowratio;
    snowfracnewsn_out[column] = snowfracnewsn;
    rhonewsn_out[column] = rhonewsn;
    smelt_out[column] = smelt;
    rainf_out[column] = rainf;
    rsm_out[column] = rsm;
    dd1_out[column] = dd1;
    infiltr_out[column] = infiltr;
    vegfrac_out[column] = vegfrac;
    drip_out[column] = drip;
    dripsn_out[column] = dripsn;
    dripliq_out[column] = dripliq;
    smf_out[column] = smf;
    interw_out[column] = interw;
    intersn_out[column] = intersn;
    infwater_out[column] = infwater;
    intwratio_out[column] = intwratio;
    gswnew_out[column] = gswnew;
    gswin_out[column] = gswin;
    albice_out[column] = albice;
    albsn_out[column] = albsn;
    emissn_out[column] = emissn;
    emiss_snowfree_out[column] = emiss_snowfree;
    keep_snow_albedo_out[column] = keep_snow_albedo;
    snowfrac2_out[column] = snowfrac2;
    snwe_out[column] = snwe;
    snhei_out[column] = snhei;
    snowfrac_out[column] = snowfrac;
    rhosn_out[column] = rhosn;
    rhosnfall_out[column] = rhosnfall;
    cst_out[column] = cst;
    alb_out[column] = alb;
    emiss_out[column] = emiss;
    znt_out[column] = znt;
    iland_out[column] = iland;
}


// module_sf_ruclsm.F:3789-4526 subroutine snowseaice: the snow energy budget
// and coupled snow/sea-ice heat diffusion.  One thread solves one column.
//
// Every arithmetic boundary is an explicit round-to-nearest intrinsic.  The
// solve is a recurrence -- the tridiagonal sweep feeds tdenom, tdenom feeds
// vilka, vilka feeds the melt pass -- so a single fused a*b+c anywhere would
// propagate.  powf matches the other powf sites in this file, and the tbq
// lookups reuse ruc_qsn_lookup and ruc_vilka above.
extern "C" __global__
void ruc_snow_sea_ice_step(
    const real* __restrict__ capice,
    const real* __restrict__ thdifice,
    const real* __restrict__ tso_in,
    const real* __restrict__ meltfactor_a,
    const real* __restrict__ rhonewsn_a,
    const real* __restrict__ prcpms_a,
    const real* __restrict__ rainf_a,
    const real* __restrict__ newsnow_a,
    const real* __restrict__ snhei_in,
    const real* __restrict__ snwe_in,
    const real* __restrict__ snowfrac_a,
    const real* __restrict__ rhosn_in,
    const real* __restrict__ patm_a,
    const real* __restrict__ qvatm_a,
    const real* __restrict__ emiss_in,
    const real* __restrict__ rnet_a,
    const real* __restrict__ qkms_a,
    const real* __restrict__ tkms_a,
    const real* __restrict__ rho_a,
    const real* __restrict__ alb_in,
    const real* __restrict__ znt_in,
    const real* __restrict__ tabs_a,
    const real* __restrict__ soilt_in,
    const real* __restrict__ soilt1_in,
    const real* __restrict__ tsnav_in,
    const real* __restrict__ qvg_in,
    const real* __restrict__ qsg_in,
    const real* __restrict__ snom_in,
    const real* __restrict__ s_in,
    const int* __restrict__ ilnb_in,
    const real* __restrict__ tbq,
    real delt,
    // ``0.5*dz8w(i,1,j)`` -- half the lowest model layer, which is a
    // PER-COLUMN depth on a terrain-following coordinate.  WRF passes it
    // as a scalar only because ``sfctmp`` is called from inside DO j/DO i.
    const real* __restrict__ conflx,
    real cvw,
    real xlv,
    int myj,
    real* __restrict__ tso_out,
    int* __restrict__ ilnb_out,
    real* __restrict__ snweprint_out,
    real* __restrict__ snheiprint_out,
    real* __restrict__ rsm_out,
    real* __restrict__ dew_out,
    real* __restrict__ soilt_out,
    real* __restrict__ soilt1_out,
    real* __restrict__ tsnav_out,
    real* __restrict__ qvg_out,
    real* __restrict__ qsg_out,
    real* __restrict__ qcg_out,
    real* __restrict__ smelt_out,
    real* __restrict__ snoh_out,
    real* __restrict__ snflx_out,
    real* __restrict__ snom_out,
    real* __restrict__ eeta_out,
    real* __restrict__ qfx_out,
    real* __restrict__ hfx_out,
    real* __restrict__ s_out,
    real* __restrict__ sublim_out,
    real* __restrict__ prcpl_out,
    real* __restrict__ fltot_out,
    real* __restrict__ snwe_out,
    real* __restrict__ snhei_out,
    real* __restrict__ rhosn_out,
    real* __restrict__ emiss_out,
    real* __restrict__ alb_out,
    real* __restrict__ znt_out,
    int ncolumn)
{
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= ncolumn) return;

    const real zero = 0.0f;
    const real one = 1.0f;
    const real half = 0.5f;
    const real two = 2.0f;
    // cp and rovcp arrive from module_model_constants; the pinned RUC lane
    // fixes them at these values and the oracle carries the same bytes.
    const real cp_air = 1004.5f;
    const real rovcp = __fdiv_rn(287.0f, cp_air);
    const real stbolt = 5.67051e-8f;
    const real freeze = 273.15f;
    // module_sf_ruclsm.F:4211 caps every ice temperature here.
    const real ice_cap = 271.4f;
    // :3930 latent heat of fusion, :3932 the sublimation sum.
    const real xlmelt = 3.35e5f;
    const real xlvm = __fadd_rn(xlv, xlmelt);
    const real thousand = 1.0e3f;
    const real milli = 1.0e-3f;
    const real snow_heat = 2090.0f;
    const real snow_cond = 0.265f;
    // :3945-3946 two-layer and blending depth thresholds.
    const real deltsn_scale = __fmul_rn(0.05f, thousand);
    const real snth_scale = __fmul_rn(0.01f, thousand);
    // :4318 Egglston melt-rate limit.
    const real egglston = 5.6e-8f;
    // :4341 Koren et al. (1999) retained-liquid fraction.
    const real rsm_low = 0.08f;
    const real rsm_high = 0.18f;
    const real rsm_depth = 0.10f;
    const real rsm_share = 0.13f;
    const real rsm_snhei = 0.01f;
    const real rhosn_low = 58.8f;
    const real rhosn_high = 500.0f;
    const real epdt_floor = 1.0e-8f;
    // (p1000mb*0.00001) from the :4438 Exner factor.
    const real reference = __fmul_rn(100000.0f, 0.00001f);
    // :4519-4521 bare sea-ice surface properties.
    const real bare_emiss = 0.98f;
    const real bare_znt = 0.011f;
    const real bare_alb = 0.55f;

    // share/module_soil_pre.F:1153-1194 nine-level RUC depths, and the
    // lsmruc-side zshalf/dtdzs construction snowseaice consumes.
    const real* zsmain = ruc_soil_layer_depth;
    real zshalf[RUC_NZS];
    real dtdzs[RUC_DTDZS_LEN];
    zshalf[0] = zero;
    for (int level = 1; level < RUC_NZS; ++level) {
        zshalf[level] = __fmul_rn(
            __fadd_rn(zsmain[level - 1], zsmain[level]), half);
    }
    for (int index = 0; index < RUC_DTDZS_LEN; ++index) dtdzs[index] = zero;
    for (int fortran_level = 2; fortran_level < RUC_NZS; ++fortran_level) {
        int first = 2 * fortran_level - 3;
        int second = first + 1;
        int level = fortran_level - 1;
        real x = __fdiv_rn(
            __fdiv_rn(delt, two),
            __fsub_rn(zshalf[level + 1], zshalf[level]));
        dtdzs[first - 1] = __fdiv_rn(
            x, __fsub_rn(zsmain[level], zsmain[level - 1]));
        dtdzs[second - 1] = __fdiv_rn(
            x, __fsub_rn(zsmain[level + 1], zsmain[level]));
    }

    real rhosn = rhosn_in[column];
    real rhonewsn = rhonewsn_a[column];
    real snhei = snhei_in[column];
    real snwe = snwe_in[column];
    real soilt = soilt_in[column];
    real soilt1 = soilt1_in[column];
    real tsnav = tsnav_in[column];
    real qvg = qvg_in[column];
    real qsg = qsg_in[column];
    real emiss = emiss_in[column];
    real alb = alb_in[column];
    real znt = znt_in[column];
    real snom = snom_in[column];
    real storage = s_in[column];
    int ilnb = ilnb_in[column];
    real rho = rho_a[column];
    real qkms = qkms_a[column];
    real tkms = tkms_a[column];
    real qvatm = qvatm_a[column];
    real tabs = tabs_a[column];
    real patm = patm_a[column];
    real rnet = rnet_a[column];
    real prcpms = prcpms_a[column];
    real rainf = rainf_a[column];
    real newsnow = newsnow_a[column];
    real meltfactor = meltfactor_a[column];
    real snowfrac = snowfrac_a[column];

    // module_sf_ruclsm.F:3945-3956.  The snhei tested here is the incoming
    // one; :4004 below rebuilds it from snwe.
    real deltsn = __fdiv_rn(deltsn_scale, rhosn);
    real snth = __fdiv_rn(snth_scale, rhosn);
    if (snhei >= __fadd_rn(deltsn, snth)) {
        if (__fsub_rn(__fsub_rn(snhei, deltsn), snth) < snth) {
            deltsn = __fmul_rn(half, __fsub_rn(snhei, snth));
        }
    }

    // module_sf_ruclsm.F:3958-3993.
    real ras = __fmul_rn(rho, milli);
    real rhocsn = __fmul_rn(snow_heat, rhosn);
    real rhonewcsn = __fmul_rn(snow_heat, rhonewsn);
    real thdifsn = __fdiv_rn(snow_cond, rhocsn);
    real smelt = zero;
    real snoh = zero;
    real rsm = zero;
    real fsn = one;
    real fso = zero;
    real qgold = qsg;
    real tnold = soilt;
    real dzstop = __fdiv_rn(one, __fsub_rn(zsmain[1], zsmain[0]));
    real prcpl = prcpms;

    // module_sf_ruclsm.F:4003-4014.
    real fq = qkms;
    snhei = __fdiv_rn(__fmul_rn(snwe, thousand), rhosn);
    real snwepr = snwe;
    real beta = one;
    real epot = -__fmul_rn(fq, __fsub_rn(qvatm, qsg));
    real epdt = __fmul_rn(__fmul_rn(epot, ras), delt);
    if (epdt > zero && snwepr <= epdt) {
        beta = __fdiv_rn(snwepr, fmaxf(epdt_floor, epdt));
        snwe = zero;
    }

    // module_sf_ruclsm.F:4020-4032 upward tridiagonal sweep in the ice.
    real cotso[RUC_NZS] = {0.0f};
    real rhtso[RUC_NZS] = {0.0f};
    rhtso[0] = tso_in[RUC_NZS_M1 * ncolumn + column];
    for (int step = 0; step < RUC_NZS_M2; ++step) {
        int kn = RUC_NZS_M1 - step;
        int first = 2 * kn - 3;
        real x1 = __fmul_rn(
            dtdzs[first - 1], thdifice[(kn - 2) * ncolumn + column]);
        real x2 = __fmul_rn(
            dtdzs[first], thdifice[(kn - 1) * ncolumn + column]);
        real ft = __fadd_rn(
            tso_in[(kn - 1) * ncolumn + column],
            __fmul_rn(
                x1,
                __fsub_rn(
                    tso_in[(kn - 2) * ncolumn + column],
                    tso_in[(kn - 1) * ncolumn + column])));
        ft = __fsub_rn(
            ft,
            __fmul_rn(
                x2,
                __fsub_rn(
                    tso_in[(kn - 1) * ncolumn + column],
                    tso_in[kn * ncolumn + column])));
        real denominator = __fadd_rn(one, x1);
        denominator = __fadd_rn(denominator, x2);
        denominator = __fsub_rn(denominator, __fmul_rn(x2, cotso[step]));
        cotso[step + 1] = __fdiv_rn(x1, denominator);
        rhtso[step + 1] = __fdiv_rn(
            __fadd_rn(ft, __fmul_rn(x2, rhtso[step])), denominator);
    }

    // WRF leaves snprim/tsob/cotsn/rhtsn undefined when snhei == 0; no
    // consumer is reached on that path, so zero stands in for them here.
    real snprim = zero;
    real tsob = zero;
    real cotsn = zero;
    real rhtsn = zero;
    real d1sn = zero;
    real d2sn = zero;
    real d9sn = zero;
    real r22sn = zero;
    bool thin = (snhei < snth) && (snhei > zero);
    real thdifice_top = thdifice[column];
    real tso1_in = tso_in[column];
    real tso2_in = tso_in[ncolumn + column];
    real tso3_in = tso_in[2 * ncolumn + column];
    if (snhei >= snth) {
        if (snhei <= __fadd_rn(deltsn, snth)) {
            // module_sf_ruclsm.F:4037-4055 one-layer snow.
            ilnb = 1;
            snprim = fmaxf(snth, snhei);
            soilt1 = tso1_in;
            tsob = tso1_in;
            real xsn = __fdiv_rn(
                __fdiv_rn(delt, two),
                __fadd_rn(zshalf[1], __fmul_rn(half, snprim)));
            real ddzsn = __fdiv_rn(xsn, snprim);
            real x1sn = __fmul_rn(ddzsn, thdifsn);
            real x2 = __fmul_rn(dtdzs[0], thdifice_top);
            real ft = __fadd_rn(
                tso1_in, __fmul_rn(x1sn, __fsub_rn(soilt, tso1_in)));
            ft = __fsub_rn(ft, __fmul_rn(x2, __fsub_rn(tso1_in, tso2_in)));
            real denominator = __fadd_rn(one, x1sn);
            denominator = __fadd_rn(denominator, x2);
            denominator = __fsub_rn(denominator, __fmul_rn(x2, cotso[RUC_NZS_M2]));
            cotso[RUC_NZS_M1] = __fdiv_rn(x1sn, denominator);
            rhtso[RUC_NZS_M1] = __fdiv_rn(
                __fadd_rn(ft, __fmul_rn(x2, rhtso[RUC_NZS_M2])), denominator);
            cotsn = cotso[RUC_NZS_M1];
            rhtsn = rhtso[RUC_NZS_M1];
            tsnav = __fsub_rn(
                __fmul_rn(half, __fadd_rn(soilt, tso1_in)), freeze);
        } else {
            // module_sf_ruclsm.F:4058-4082 two-layer snow.
            ilnb = 2;
            snprim = deltsn;
            tsob = soilt1;
            real xsn = __fdiv_rn(
                __fdiv_rn(delt, two), __fmul_rn(half, snhei));
            real xsn1 = __fdiv_rn(
                __fdiv_rn(delt, two),
                __fadd_rn(
                    zshalf[1],
                    __fmul_rn(half, __fsub_rn(snhei, deltsn))));
            real ddzsn = __fdiv_rn(xsn, deltsn);
            real ddzsn1 = __fdiv_rn(xsn1, __fsub_rn(snhei, deltsn));
            real x1sn = __fmul_rn(ddzsn, thdifsn);
            real x1sn1 = __fmul_rn(ddzsn1, thdifsn);
            real x2 = __fmul_rn(dtdzs[0], thdifice_top);
            real ft = __fadd_rn(
                tso1_in, __fmul_rn(x1sn1, __fsub_rn(soilt1, tso1_in)));
            ft = __fsub_rn(ft, __fmul_rn(x2, __fsub_rn(tso1_in, tso2_in)));
            real denominator = __fadd_rn(one, x1sn1);
            denominator = __fadd_rn(denominator, x2);
            denominator = __fsub_rn(denominator, __fmul_rn(x2, cotso[RUC_NZS_M2]));
            cotso[RUC_NZS_M1] = __fdiv_rn(x1sn1, denominator);
            rhtso[RUC_NZS_M1] = __fdiv_rn(
                __fadd_rn(ft, __fmul_rn(x2, rhtso[RUC_NZS_M2])), denominator);
            real ftsnow = __fadd_rn(
                soilt1, __fmul_rn(x1sn, __fsub_rn(soilt, soilt1)));
            ftsnow = __fsub_rn(
                ftsnow, __fmul_rn(x1sn1, __fsub_rn(soilt1, tso1_in)));
            real denomsn = __fadd_rn(one, x1sn);
            denomsn = __fadd_rn(denomsn, x1sn1);
            denomsn = __fsub_rn(denomsn, __fmul_rn(x1sn1, cotso[RUC_NZS_M1]));
            cotsn = __fdiv_rn(x1sn, denomsn);
            rhtsn = __fdiv_rn(
                __fadd_rn(ftsnow, __fmul_rn(x1sn1, rhtso[RUC_NZS_M1])), denomsn);
            tsnav = __fsub_rn(
                __fmul_rn(
                    __fdiv_rn(half, snhei),
                    __fadd_rn(
                        __fmul_rn(__fadd_rn(soilt, soilt1), deltsn),
                        __fmul_rn(
                            __fadd_rn(soilt1, tso1_in),
                            __fsub_rn(snhei, deltsn)))),
                freeze);
        }
    }

    if (thin) {
        // module_sf_ruclsm.F:4089-4108 snow blended into the top ice layer.
        snprim = __fadd_rn(snhei, zsmain[1]);
        fsn = __fdiv_rn(snhei, snprim);
        fso = __fsub_rn(one, fsn);
        soilt1 = tso1_in;
        tsob = tso2_in;
        real xsn = __fdiv_rn(
            __fdiv_rn(delt, two),
            __fadd_rn(
                __fsub_rn(zshalf[2], zsmain[1]),
                __fmul_rn(half, snprim)));
        real ddzsn = __fdiv_rn(xsn, snprim);
        real x1sn = __fmul_rn(
            ddzsn,
            __fadd_rn(
                __fmul_rn(fsn, thdifsn), __fmul_rn(fso, thdifice_top)));
        real x2 = __fmul_rn(dtdzs[1], thdifice[ncolumn + column]);
        real ft = __fadd_rn(
            tso2_in, __fmul_rn(x1sn, __fsub_rn(soilt, tso2_in)));
        ft = __fsub_rn(ft, __fmul_rn(x2, __fsub_rn(tso2_in, tso3_in)));
        real denominator = __fadd_rn(one, x1sn);
        denominator = __fadd_rn(denominator, x2);
        denominator = __fsub_rn(denominator, __fmul_rn(x2, cotso[RUC_NZS_M3]));
        cotso[RUC_NZS_M2] = __fdiv_rn(x1sn, denominator);
        rhtso[RUC_NZS_M2] = __fdiv_rn(
            __fadd_rn(ft, __fmul_rn(x2, rhtso[RUC_NZS_M3])), denominator);
        tsnav = __fsub_rn(
            __fmul_rn(half, __fadd_rn(soilt, tso1_in)), freeze);
        cotso[RUC_NZS_M1] = cotso[RUC_NZS_M2];
        rhtso[RUC_NZS_M1] = rhtso[RUC_NZS_M2];
        cotsn = cotso[RUC_NZS_M1];
        rhtsn = rhtso[RUC_NZS_M1];
    }

    // module_sf_ruclsm.F:4114-4131 heat balance coefficients.
    epot = -__fmul_rn(qkms, __fsub_rn(qvatm, qsg));
    real rhcs = capice[column];
    real d1 = cotso[RUC_NZS_M2];
    real d2 = rhtso[RUC_NZS_M2];
    real tn = soilt;
    real d9 = __fmul_rn(__fmul_rn(thdifice_top, rhcs), dzstop);
    real d10 = __fmul_rn(__fmul_rn(tkms, cp_air), rho);
    real r211 = __fdiv_rn(__fmul_rn(half, conflx[column]), delt);
    real r21 = __fmul_rn(__fmul_rn(r211, cp_air), rho);
    real depth_square = __fmul_rn(dzstop, dzstop);
    real r22 = __fdiv_rn(
        half, __fmul_rn(__fmul_rn(thdifice_top, delt), depth_square));
    real tn2 = __fmul_rn(tn, tn);
    real tn4 = __fmul_rn(tn2, tn2);
    real r6 = __fmul_rn(__fmul_rn(__fmul_rn(emiss, stbolt), half), tn4);
    real r7 = __fdiv_rn(r6, tn);
    real d11 = __fadd_rn(rnet, r6);

    // module_sf_ruclsm.F:4133-4163 snow-side coefficients.
    if (snhei >= snth) {
        if (snhei <= __fadd_rn(deltsn, snth)) {
            d1sn = cotso[RUC_NZS_M1];
            d2sn = rhtso[RUC_NZS_M1];
        } else {
            d1sn = cotsn;
            d2sn = rhtsn;
        }
        d9sn = __fdiv_rn(__fmul_rn(thdifsn, rhocsn), snprim);
        r22sn = __fdiv_rn(
            __fmul_rn(__fmul_rn(snprim, snprim), half),
            __fmul_rn(thdifsn, delt));
    }
    if (thin) {
        d1sn = d1;
        d2sn = d2;
        d9sn = __fdiv_rn(
            __fadd_rn(
                __fmul_rn(__fmul_rn(fsn, thdifsn), rhocsn),
                __fmul_rn(__fmul_rn(fso, thdifice_top), rhcs)),
            snprim);
        r22sn = __fdiv_rn(
            __fmul_rn(__fmul_rn(snprim, snprim), half),
            __fmul_rn(
                __fadd_rn(
                    __fmul_rn(fsn, thdifsn),
                    __fmul_rn(fso, thdifice_top)),
                delt));
    }
    if (snhei == zero) {
        d9sn = d9;
        r22sn = r22;
        d1sn = d1;
        d2sn = d2;
    }

    // module_sf_ruclsm.F:4167-4182.
    real rain_heat_capacity = __fmul_rn(__fmul_rn(rainf, cvw), prcpms);
    real newsnow_heat_capacity = __fdiv_rn(
        __fmul_rn(rhonewcsn, newsnow), delt);
    real tdenom = __fmul_rn(
        d9sn, __fadd_rn(__fsub_rn(one, d1sn), r22sn));
    tdenom = __fadd_rn(tdenom, d10);
    tdenom = __fadd_rn(tdenom, r21);
    tdenom = __fadd_rn(tdenom, r7);
    tdenom = __fadd_rn(tdenom, rain_heat_capacity);
    tdenom = __fadd_rn(tdenom, newsnow_heat_capacity);
    real fkq = __fmul_rn(qkms, rho);
    real r210 = __fmul_rn(r211, rho);
    real beta_fkq = __fmul_rn(beta, fkq);
    real aa = __fdiv_rn(
        __fmul_rn(xlvm, __fadd_rn(beta_fkq, r210)), tdenom);
    real humidity_inner = __fmul_rn(qvatm, beta_fkq);
    humidity_inner = __fadd_rn(humidity_inner, __fmul_rn(r210, qvg));
    real rain_temperature = fmaxf(freeze, tabs);
    real newsnow_temperature = fminf(freeze, tabs);
    real numerator = __fmul_rn(d10, tabs);
    numerator = __fadd_rn(numerator, __fmul_rn(r21, tn));
    numerator = __fadd_rn(numerator, __fmul_rn(xlvm, humidity_inner));
    numerator = __fadd_rn(numerator, d11);
    real conduction = __fadd_rn(d2sn, __fmul_rn(r22sn, tn));
    numerator = __fadd_rn(numerator, __fmul_rn(d9sn, conduction));
    numerator = __fadd_rn(
        numerator, __fmul_rn(rain_heat_capacity, rain_temperature));
    numerator = __fadd_rn(
        numerator,
        __fmul_rn(newsnow_heat_capacity, newsnow_temperature));
    real bb = __fdiv_rn(numerator, tdenom);
    real aa1 = aa;
    real pp = __fmul_rn(patm, thousand);
    aa1 = __fdiv_rn(aa1, pp);

    // module_sf_ruclsm.F:4184-4200.  snoh is still zero on the only pass WRF
    // makes; the :4374 second iteration is commented out upstream.
    bb = __fsub_rn(bb, __fdiv_rn(snoh, tdenom));
    real qs1, ts1;
    if (!ruc_vilka(tn, aa1, bb, pp, tbq, &qs1, &ts1)) {
        real invalid = nanf("");
        for (int level = 0; level < RUC_NZS; ++level) {
            tso_out[level * ncolumn + column] = invalid;
        }
        ilnb_out[column] = ilnb;
        snweprint_out[column] = invalid;
        snheiprint_out[column] = invalid;
        rsm_out[column] = invalid;
        dew_out[column] = invalid;
        soilt_out[column] = invalid;
        soilt1_out[column] = invalid;
        tsnav_out[column] = invalid;
        qvg_out[column] = invalid;
        qsg_out[column] = invalid;
        qcg_out[column] = invalid;
        smelt_out[column] = invalid;
        snoh_out[column] = invalid;
        snflx_out[column] = invalid;
        snom_out[column] = invalid;
        eeta_out[column] = invalid;
        qfx_out[column] = invalid;
        hfx_out[column] = invalid;
        s_out[column] = invalid;
        sublim_out[column] = invalid;
        prcpl_out[column] = invalid;
        fltot_out[column] = invalid;
        snwe_out[column] = invalid;
        snhei_out[column] = invalid;
        rhosn_out[column] = invalid;
        emiss_out[column] = invalid;
        alb_out[column] = invalid;
        znt_out[column] = invalid;
        return;
    }
    qvg = qs1;
    qsg = qs1;
    real qcg = zero;
    soilt = ts1;

    // module_sf_ruclsm.F:4207-4230 snow interior and top ice temperature.
    if (snhei >= snth) {
        if (snhei > __fadd_rn(deltsn, snth)) {
            soilt1 = fminf(freeze, __fadd_rn(rhtsn, __fmul_rn(cotsn, soilt)));
            tso_out[column] = fminf(
                ice_cap, __fadd_rn(rhtso[RUC_NZS_M1], __fmul_rn(cotso[RUC_NZS_M1], soilt1)));
            tsob = soilt1;
        } else {
            tso_out[column] = fminf(
                ice_cap, __fadd_rn(rhtso[RUC_NZS_M1], __fmul_rn(cotso[RUC_NZS_M1], soilt)));
            soilt1 = tso_out[column];
            tsob = tso_out[column];
        }
    } else if (thin) {
        tso_out[ncolumn + column] = fminf(
            ice_cap, __fadd_rn(rhtso[RUC_NZS_M2], __fmul_rn(cotso[RUC_NZS_M2], soilt)));
        tso_out[column] = fminf(
            ice_cap,
            __fadd_rn(
                tso_out[ncolumn + column],
                __fmul_rn(
                    __fsub_rn(soilt, tso_out[ncolumn + column]), fso)));
        soilt1 = tso_out[column];
        tsob = tso_out[ncolumn + column];
    } else {
        tso_out[column] = fminf(ice_cap, soilt);
        soilt1 = fminf(ice_cap, soilt);
        tsob = tso_out[column];
    }

    // module_sf_ruclsm.F:4232-4243 downward substitution through the ice.
    for (int level = thin ? 2 : 1; level < RUC_NZS; ++level) {
        int coefficient = RUC_NZS_M1 - level;
        tso_out[level * ncolumn + column] = fminf(
            ice_cap,
            __fadd_rn(
                rhtso[coefficient],
                __fmul_rn(
                    cotso[coefficient],
                    tso_out[(level - 1) * ncolumn + column])));
    }

    // module_sf_ruclsm.F:4257 melting test on the pre-vilka epot.
    real dew = zero;
    real eeta;
    real qfx;
    real hfx;
    real snflx;
    real melt_supply = __fsub_rn(
        snwepr, __fmul_rn(__fmul_rn(__fmul_rn(beta, epot), ras), delt));
    if (soilt > freeze && melt_supply > zero && snhei > zero) {
        // module_sf_ruclsm.F:4259-4360 the single melt pass.
        real soiltfrac = __fadd_rn(
            __fmul_rn(snowfrac, freeze),
            __fmul_rn(
                __fsub_rn(one, snowfrac), fminf(ice_cap, soilt)));
        qsg = __fdiv_rn(ruc_qsn_lookup(soiltfrac, tbq), pp);
        epot = -__fmul_rn(qkms, __fsub_rn(qvatm, qsg));
        real q1 = __fmul_rn(epot, ras);
        if (q1 <= zero) {
            dew = -epot;
            qfx = __fmul_rn(__fmul_rn(xlvm, rho), dew);
            eeta = __fdiv_rn(qfx, xlvm);
        } else {
            eeta = __fmul_rn(__fmul_rn(q1, beta), thousand);
            qfx = -__fmul_rn(xlvm, eeta);
        }
        hfx = __fmul_rn(d10, __fsub_rn(tabs, soiltfrac));
        real soh;
        if (snhei >= snth) {
            soh = __fdiv_rn(
                __fmul_rn(
                    __fmul_rn(thdifsn, rhocsn),
                    __fsub_rn(soiltfrac, tsob)),
                snprim);
        } else {
            soh = __fdiv_rn(
                __fmul_rn(
                    __fadd_rn(
                        __fmul_rn(__fmul_rn(fsn, thdifsn), rhocsn),
                        __fmul_rn(__fmul_rn(fso, thdifice_top), rhcs)),
                    __fsub_rn(soiltfrac, tsob)),
                snprim);
        }
        snflx = soh;
        real x = __fmul_rn(
            __fadd_rn(r21, __fmul_rn(d9sn, r22sn)),
            __fsub_rn(soiltfrac, tnold));
        x = __fadd_rn(
            x,
            __fmul_rn(__fmul_rn(xlvm, r210), __fsub_rn(qsg, qgold)));
        snoh = __fadd_rn(rnet, qfx);
        snoh = __fadd_rn(snoh, hfx);
        snoh = __fadd_rn(
            snoh,
            __fmul_rn(
                newsnow_heat_capacity,
                __fsub_rn(newsnow_temperature, soiltfrac)));
        snoh = __fsub_rn(snoh, soh);
        snoh = __fsub_rn(snoh, x);
        snoh = __fadd_rn(
            snoh,
            __fmul_rn(
                rain_heat_capacity,
                __fsub_rn(rain_temperature, soiltfrac)));
        snoh = fmaxf(zero, snoh);
        smelt = __fmul_rn(__fdiv_rn(snoh, xlmelt), milli);
        real potential = __fsub_rn(
            __fdiv_rn(snwepr, delt),
            __fmul_rn(__fmul_rn(beta, epot), ras));
        smelt = fminf(smelt, potential);
        smelt = fmaxf(zero, smelt);
        smelt = fminf(
            smelt,
            __fmul_rn(
                __fmul_rn(egglston, meltfactor),
                fmaxf(one, __fsub_rn(soilt, freeze))));
        real rr = __fsub_rn(
            __fdiv_rn(snwepr, delt),
            __fmul_rn(__fmul_rn(beta, epot), ras));
        smelt = fminf(smelt, rr);
        // WRF's snodif here only feeds debug prints.
        snoh = __fmul_rn(__fmul_rn(smelt, xlmelt), thousand);
        real rsmfrac = fminf(
            rsm_high,
            fmaxf(
                rsm_low,
                __fmul_rn(__fdiv_rn(snwepr, rsm_depth), rsm_share)));
        if (snhei > rsm_snhei) {
            rsm = __fmul_rn(__fmul_rn(rsmfrac, smelt), delt);
        } else {
            rsm = zero;
        }
        smelt = fmaxf(zero, __fsub_rn(smelt, __fdiv_rn(rsm, delt)));
        snwe = fmaxf(
            zero,
            __fsub_rn(
                snwepr,
                __fmul_rn(
                    __fadd_rn(
                        smelt, __fmul_rn(__fmul_rn(beta, epot), ras)),
                    delt)));
        soilt = soiltfrac;
    } else {
        // module_sf_ruclsm.F:4363-4370 evaporation-only update.
        if (snhei != zero) {
            epot = -__fmul_rn(qkms, __fsub_rn(qvatm, qsg));
            snwe = fmaxf(
                zero,
                __fsub_rn(
                    snwepr,
                    __fmul_rn(
                        __fmul_rn(__fmul_rn(beta, epot), ras), delt)));
        }
    }

    // module_sf_ruclsm.F:4377-4396 melt-driven snow densification.
    if (smelt > zero && rsm > zero) {
        if (snwe > rsm) {
            real xsn = __fdiv_rn(
                __fadd_rn(
                    __fmul_rn(rhosn, __fsub_rn(snwe, rsm)),
                    __fmul_rn(thousand, rsm)),
                snwe);
            rhosn = fminf(fmaxf(rhosn_low, xsn), rhosn_high);
            rhocsn = __fmul_rn(snow_heat, rhosn);
            thdifsn = __fdiv_rn(snow_cond, rhocsn);
        }
    }

    // module_sf_ruclsm.F:4398-4403 diagnostic copies.
    real snweprint = snwe;
    real snheiprint = __fdiv_rn(__fmul_rn(snweprint, thousand), rhosn);

    // module_sf_ruclsm.F:4409-4417 snow-pack mean temperature.
    if (snhei > zero) {
        if (ilnb > 1) {
            tsnav = __fsub_rn(
                __fmul_rn(
                    __fdiv_rn(half, snhei),
                    __fadd_rn(
                        __fmul_rn(__fadd_rn(soilt, soilt1), deltsn),
                        __fmul_rn(
                            __fadd_rn(soilt1, tso_out[column]),
                            __fsub_rn(snhei, deltsn)))),
                freeze);
        } else {
            tsnav = __fsub_rn(
                __fmul_rn(half, __fadd_rn(soilt, tso_out[column])), freeze);
        }
    }

    // module_sf_ruclsm.F:4419-4428.
    dew = zero;
    pp = __fmul_rn(patm, thousand);
    qsg = __fdiv_rn(ruc_qsn_lookup(soilt, tbq), pp);
    epot = -__fmul_rn(fq, __fsub_rn(qvatm, qsg));
    if (epot < zero) {
        dew = -epot;
    }
    snom = __fadd_rn(snom, __fmul_rn(__fmul_rn(smelt, delt), thousand));

    // module_sf_ruclsm.F:4432-4466 surface flux diagnostics.  WRF's
    // t3/upflux/xinet block here is dead: xinet is assigned, never read.
    real sensible = __fmul_rn(__fmul_rn(tkms, cp_air), rho);
    sensible = __fmul_rn(sensible, __fsub_rn(tabs, soilt));
    real hft = -sensible;
    // Same substitution; see _RUC_PROVISIONAL_TRANSCENDENTALS.
    real exner = ruc_powf_rn(__fdiv_rn(reference, patm), rovcp);
    hfx = -__fmul_rn(sensible, exner);
    real q1 = -__fmul_rn(__fmul_rn(fq, ras), __fsub_rn(qvatm, qsg));
    if (q1 < zero) {
        if (myj) {
            eeta = -__fmul_rn(
                __fmul_rn(
                    __fmul_rn(qkms, ras),
                    __fsub_rn(
                        __fdiv_rn(qvatm, __fadd_rn(one, qvatm)),
                        __fdiv_rn(qsg, __fadd_rn(one, qsg)))),
                thousand);
        } else {
            dew = __fmul_rn(qkms, __fsub_rn(qvatm, qsg));
            eeta = -__fmul_rn(rho, dew);
        }
        qfx = __fmul_rn(xlvm, eeta);
        eeta = -__fmul_rn(rho, dew);
    } else {
        if (myj) {
            eeta = -__fmul_rn(
                __fmul_rn(
                    __fmul_rn(__fmul_rn(qkms, ras), beta),
                    __fsub_rn(
                        __fdiv_rn(qvatm, __fadd_rn(one, qvatm)),
                        __fdiv_rn(qvg, __fadd_rn(one, qvg)))),
                thousand);
        } else {
            eeta = __fmul_rn(__fmul_rn(q1, beta), thousand);
        }
        qfx = __fmul_rn(xlvm, eeta);
        eeta = __fmul_rn(__fmul_rn(q1, beta), thousand);
    }
    real sublim = eeta;

    // module_sf_ruclsm.F:4468-4486.  The snhei tested here is still the
    // pre-melt depth; :4486 rebuilds it from the updated snwe.
    if (snhei >= snth) {
        storage = __fdiv_rn(
            __fmul_rn(
                __fmul_rn(thdifsn, rhocsn), __fsub_rn(soilt, tsob)),
            snprim);
        snflx = storage;
    } else if (thin) {
        storage = __fdiv_rn(
            __fmul_rn(
                __fadd_rn(
                    __fmul_rn(__fmul_rn(fsn, thdifsn), rhocsn),
                    __fmul_rn(__fmul_rn(fso, thdifice_top), rhcs)),
                __fsub_rn(soilt, tsob)),
            snprim);
        snflx = storage;
    } else {
        snflx = __fmul_rn(d9sn, __fsub_rn(soilt, tsob));
    }
    snhei = __fdiv_rn(__fmul_rn(snwe, thousand), rhosn);

    // module_sf_ruclsm.F:4492-4509.
    real x = __fmul_rn(
        __fadd_rn(r21, __fmul_rn(d9sn, r22sn)), __fsub_rn(soilt, tnold));
    x = __fadd_rn(
        x, __fmul_rn(__fmul_rn(xlvm, r210), __fsub_rn(qsg, qgold)));
    x = __fsub_rn(
        x,
        __fmul_rn(
            newsnow_heat_capacity,
            __fsub_rn(newsnow_temperature, soilt)));
    x = __fsub_rn(
        x,
        __fmul_rn(
            rain_heat_capacity, __fsub_rn(rain_temperature, soilt)));
    real residual = __fsub_rn(rnet, hft);
    residual = __fsub_rn(residual, __fmul_rn(xlvm, eeta));
    residual = __fsub_rn(residual, storage);
    residual = __fsub_rn(residual, snoh);
    residual = __fsub_rn(residual, x);
    real icemelt = residual;
    real fltot = __fsub_rn(residual, icemelt);

    // module_sf_ruclsm.F:4517-4522 restore the bare sea-ice surface.
    if (snhei == zero) {
        tsnav = __fsub_rn(soilt, freeze);
        emiss = bare_emiss;
        znt = bare_znt;
        alb = bare_alb;
    }

    ilnb_out[column] = ilnb;
    snweprint_out[column] = snweprint;
    snheiprint_out[column] = snheiprint;
    rsm_out[column] = rsm;
    dew_out[column] = dew;
    soilt_out[column] = soilt;
    soilt1_out[column] = soilt1;
    tsnav_out[column] = tsnav;
    qvg_out[column] = qvg;
    qsg_out[column] = qsg;
    qcg_out[column] = qcg;
    smelt_out[column] = smelt;
    snoh_out[column] = snoh;
    snflx_out[column] = snflx;
    snom_out[column] = snom;
    eeta_out[column] = eeta;
    qfx_out[column] = qfx;
    hfx_out[column] = hfx;
    s_out[column] = storage;
    sublim_out[column] = sublim;
    prcpl_out[column] = prcpl;
    fltot_out[column] = fltot;
    snwe_out[column] = snwe;
    snhei_out[column] = snhei;
    rhosn_out[column] = rhosn;
    emiss_out[column] = emiss;
    alb_out[column] = alb;
    znt_out[column] = znt;
}


// module_sf_ruclsm.F:5046-5072, repeated verbatim at :5587-5610.  :49 fixes
// isncond_opt = 2, so the constant 0.265/rhocsn branch is dead and the Sturm
// et al. (1997) effective conductivity always applies.
__device__ __forceinline__
real ruc_snow_thermal_diffusivity(
    real rhosn,
    real rhocsn,
    real rhonewsn,
    real newsnow,
    real snhei)
{
    const real fact = 1.0f;
    real keff;
    if (rhosn < 156.0f || (newsnow > 0.0f && rhonewsn < 156.0f)) {
        keff = __fadd_rn(
            0.023f, __fmul_rn(__fmul_rn(0.234f, rhosn), 1.0e-3f));
    } else {
        keff = __fsub_rn(
            0.138f, __fmul_rn(__fmul_rn(1.01f, rhosn), 1.0e-3f));
        keff = __fadd_rn(
            keff,
            __fmul_rn(
                __fmul_rn(3.233f, __fmul_rn(rhosn, rhosn)), 1.0e-6f));
    }
    if (newsnow <= 0.0f && snhei > 1.0f && rhosn > 250.0f) {
        // :5600-5606 hard slabs under deep packs get a fixed diffusivity.
        return 4.431718e-7f;
    }
    return __fmul_rn(__fdiv_rn(keff, rhocsn), fact);
}


// module_sf_ruclsm.F:4836-5728 subroutine snowtemp: the snow energy budget
// and the snow/soil heat-diffusion solve.  One thread carries one
// independent nine-level column: the upward tridiagonal sweep (:5103-5115),
// the extra snow coefficient row for a one-layer pack (:5119-5141), a
// two-layer pack (:5143-5171) or snow blended into the top soil layer
// (:5174-5198), the surface energy balance closed by vilka (:5277-5329), the
// downward substitution (:5347-5394), the melt iteration (:5414-5569) and the
// bottom-melt, density and flux epilogue (:5572-5725).
//
// i, j, ktau, iland and isoil only reach debug prints and vilka's fatal
// handler; qcatm, gsw, pc, dqm, qmin, psis, bclh, mavail, rovcp and g0_p are
// dead; glw reaches only the local xinet (:5421) which is never read; cst
// reaches only cmc2ms (:5447) which is likewise dead; and the incoming qsg,
// qcg and tsnav are overwritten before they are read.  None of them are
// arguments here.  iter is zero at :5030 and never reassigned, so the
// iter==1 disjunct at :5425 is unreachable.
//
// h is fixed at 1 (:5209), so tx2 is exactly zero and q1 equals qs1 bit for
// bit: the unsaturated retry at :5315-5328 cannot be entered.  It is
// transcribed anyway.
extern "C" __global__
void ruc_snow_temperature_step(
    const real* __restrict__ cap,
    const real* __restrict__ thdif,
    const real* __restrict__ tranf,
    const real* __restrict__ tso_in,
    const real* __restrict__ snwe_in,
    const real* __restrict__ snwepr_a,
    const real* __restrict__ snhei_in,
    const real* __restrict__ newsnow_a,
    const real* __restrict__ snowfrac_a,
    const real* __restrict__ beta_in,
    const real* __restrict__ deltsn_a,
    const real* __restrict__ snth_a,
    const real* __restrict__ rhosn_in,
    const real* __restrict__ rhonewsn_a,
    const real* __restrict__ meltfactor_a,
    const real* __restrict__ prcpms_a,
    const real* __restrict__ rainf_a,
    const real* __restrict__ patm_a,
    const real* __restrict__ tabs_a,
    const real* __restrict__ qvatm_a,
    const real* __restrict__ emiss_a,
    const real* __restrict__ rnet_a,
    const real* __restrict__ qkms_a,
    const real* __restrict__ tkms_a,
    const real* __restrict__ rho_a,
    const real* __restrict__ vegfrac_a,
    const real* __restrict__ drycan_a,
    const real* __restrict__ wetcan_a,
    const real* __restrict__ transum_a,
    const real* __restrict__ dew_in,
    const real* __restrict__ soilt_in,
    const real* __restrict__ soilt1_in,
    const real* __restrict__ qvg_in,
    const int* __restrict__ nroot_a,
    const int* __restrict__ ilnb_in,
    const real* __restrict__ tbq,
    real delt,
    // ``0.5*dz8w(i,1,j)`` -- half the lowest model layer, which is a
    // PER-COLUMN depth on a terrain-following coordinate.  WRF passes it
    // as a scalar only because ``sfctmp`` is called from inside DO j/DO i.
    const real* __restrict__ conflx,
    real xlvm,
    real cvw,
    real* __restrict__ tso_out,
    real* __restrict__ soilt_out,
    real* __restrict__ soilt1_out,
    real* __restrict__ tsnav_out,
    real* __restrict__ qvg_out,
    real* __restrict__ qsg_out,
    real* __restrict__ qcg_out,
    real* __restrict__ dew_out,
    real* __restrict__ snwe_out,
    real* __restrict__ snhei_out,
    real* __restrict__ rhosn_out,
    real* __restrict__ beta_out,
    real* __restrict__ smelt_out,
    real* __restrict__ snoh_out,
    real* __restrict__ snflx_out,
    real* __restrict__ s_out,
    real* __restrict__ rsm_out,
    real* __restrict__ snweprint_out,
    real* __restrict__ snheiprint_out,
    real* __restrict__ storage_out,
    int* __restrict__ ilnb_out,
    int ncolumn)
{
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= ncolumn) return;

    const real zero = 0.0f;
    const real one = 1.0f;
    const real half = 0.5f;
    // module_model_constants cp and stbolt; the pinned RUC lane fixes them at
    // these values and the oracle carries the same bytes.
    const real cp_air = 1004.5f;
    const real stbolt = 5.67051e-8f;
    const real freeze = 273.15f;
    // :5045 latent heat of fusion.
    const real xlmelt = 3.35e5f;
    const real thousand = 1.0e3f;
    const real milli = 1.0e-3f;

    // share/module_soil_pre.F:1153-1194 nine-level RUC depths, and the
    // lsmruc-side zshalf/dtdzs construction snowtemp consumes.
    const real* zsmain = ruc_soil_layer_depth;
    real zshalf[RUC_NZS];
    real dtdzs[RUC_DTDZS_LEN];
    zshalf[0] = zero;
    for (int level = 1; level < RUC_NZS; ++level) {
        zshalf[level] = __fmul_rn(
            __fadd_rn(zsmain[level - 1], zsmain[level]), half);
    }
    for (int index = 0; index < RUC_DTDZS_LEN; ++index) dtdzs[index] = zero;
    for (int fortran_level = 2; fortran_level < RUC_NZS; ++fortran_level) {
        int first = 2 * fortran_level - 3;
        int second = first + 1;
        int level = fortran_level - 1;
        real x = __fdiv_rn(
            __fdiv_rn(delt, 2.0f),
            __fsub_rn(zshalf[level + 1], zshalf[level]));
        dtdzs[first - 1] = __fdiv_rn(
            x, __fsub_rn(zsmain[level], zsmain[level - 1]));
        dtdzs[second - 1] = __fdiv_rn(
            x, __fsub_rn(zsmain[level + 1], zsmain[level]));
    }
    real dzstop = __fdiv_rn(one, __fsub_rn(zsmain[1], zsmain[0]));

    int root_count = nroot_a[column];
    int snow_layers = ilnb_in[column];
    real tso[RUC_NZS];
    for (int level = 0; level < RUC_NZS; ++level) {
        tso[level] = tso_in[level * ncolumn + column];
    }
    real snwe = snwe_in[column];
    real snwepr = snwepr_a[column];
    real snhei = snhei_in[column];
    real newsnow = newsnow_a[column];
    real snowfrac = snowfrac_a[column];
    real beta = beta_in[column];
    real deltsn = deltsn_a[column];
    real snth = snth_a[column];
    real rhosn = rhosn_in[column];
    real rhonewsn = rhonewsn_a[column];
    real meltfactor = meltfactor_a[column];
    real prcpms = prcpms_a[column];
    real rainf = rainf_a[column];
    real patm = patm_a[column];
    real tabs = tabs_a[column];
    real qvatm = qvatm_a[column];
    real emiss = emiss_a[column];
    real rnet = rnet_a[column];
    real qkms = qkms_a[column];
    real tkms = tkms_a[column];
    real rho = rho_a[column];
    real vegfrac = vegfrac_a[column];
    real drycan = drycan_a[column];
    real wetcan = wetcan_a[column];
    real transum = transum_a[column];
    real dew = dew_in[column];
    real soilt = soilt_in[column];
    real soilt1 = soilt1_in[column];
    real qvg = qvg_in[column];
    real thdif0 = thdif[column];

    // :5046-5072 snow heat capacity and thermal diffusivity.
    real rhocsn = __fmul_rn(2090.0f, rhosn);
    real rhonewcsn = __fmul_rn(2090.0f, rhonewsn);
    real thdifsn = ruc_snow_thermal_diffusivity(
        rhosn, rhocsn, rhonewsn, newsnow, snhei);
    // :5074-5091 prologue.
    real ras = __fmul_rn(rho, milli);
    real soiltfrac = soilt;
    real smelt = zero;
    real rsm = zero;
    real fsn = one;
    real fso = zero;
    real qgold = qvg;
    real snprim = zero;
    real tsob = zero;
    real cotsn = zero;
    real rhtsn = zero;
    real tsnav = zero;

    // :5103-5115 upward tridiagonal sweep through the soil.
    real cotso[RUC_NZS];
    real rhtso[RUC_NZS];
    for (int level = 0; level < RUC_NZS; ++level) {
        cotso[level] = zero;
        rhtso[level] = zero;
    }
    rhtso[0] = tso[RUC_NZS_M1];
    for (int step = 0; step < RUC_NZS_M2; ++step) {
        int kn = RUC_NZS_M1 - step;
        int first = 2 * kn - 3;
        real x1 = __fmul_rn(dtdzs[first - 1], thdif[(kn - 2) * ncolumn + column]);
        real x2 = __fmul_rn(dtdzs[first], thdif[(kn - 1) * ncolumn + column]);
        real ft = __fadd_rn(
            tso[kn - 1],
            __fmul_rn(x1, __fsub_rn(tso[kn - 2], tso[kn - 1])));
        ft = __fsub_rn(ft, __fmul_rn(x2, __fsub_rn(tso[kn - 1], tso[kn])));
        real denominator = __fadd_rn(one, x1);
        denominator = __fadd_rn(denominator, x2);
        denominator = __fsub_rn(denominator, __fmul_rn(x2, cotso[step]));
        cotso[step + 1] = __fdiv_rn(x1, denominator);
        rhtso[step + 1] = __fdiv_rn(
            __fadd_rn(ft, __fmul_rn(x2, rhtso[step])), denominator);
    }

    real half_step = __fdiv_rn(delt, 2.0f);
    if (snhei >= snth) {
        if (snhei <= __fadd_rn(deltsn, snth)) {
            // :5124-5141 one snow layer.
            snow_layers = 1;
            snprim = fmaxf(snth, snhei);
            tsob = tso[0];
            soilt1 = tso[0];
            real xsn = __fdiv_rn(
                half_step, __fadd_rn(zshalf[1], __fmul_rn(half, snprim)));
            real ddzsn = __fdiv_rn(xsn, snprim);
            real x1sn = __fmul_rn(ddzsn, thdifsn);
            real x2 = __fmul_rn(dtdzs[0], thdif0);
            real ft = __fadd_rn(
                tso[0], __fmul_rn(x1sn, __fsub_rn(soilt, tso[0])));
            ft = __fsub_rn(ft, __fmul_rn(x2, __fsub_rn(tso[0], tso[1])));
            real denominator = __fadd_rn(one, x1sn);
            denominator = __fadd_rn(denominator, x2);
            denominator = __fsub_rn(denominator, __fmul_rn(x2, cotso[RUC_NZS_M2]));
            cotso[RUC_NZS_M1] = __fdiv_rn(x1sn, denominator);
            rhtso[RUC_NZS_M1] = __fdiv_rn(
                __fadd_rn(ft, __fmul_rn(x2, rhtso[RUC_NZS_M2])), denominator);
            cotsn = cotso[RUC_NZS_M1];
            rhtsn = rhtso[RUC_NZS_M1];
            tsnav = __fsub_rn(
                __fmul_rn(half, __fadd_rn(soilt, tso[0])), freeze);
        } else {
            // :5148-5171 two snow layers, soilt1 sits at deltsn depth.
            snow_layers = 2;
            snprim = deltsn;
            tsob = soilt1;
            real xsn = __fdiv_rn(half_step, __fmul_rn(half, deltsn));
            real xsn1 = __fdiv_rn(
                half_step,
                __fadd_rn(
                    zshalf[1],
                    __fmul_rn(half, __fsub_rn(snhei, deltsn))));
            real ddzsn = __fdiv_rn(xsn, deltsn);
            real ddzsn1 = __fdiv_rn(xsn1, __fsub_rn(snhei, deltsn));
            real x1sn = __fmul_rn(ddzsn, thdifsn);
            real x1sn1 = __fmul_rn(ddzsn1, thdifsn);
            real x2 = __fmul_rn(dtdzs[0], thdif0);
            real ft = __fadd_rn(
                tso[0], __fmul_rn(x1sn1, __fsub_rn(soilt1, tso[0])));
            ft = __fsub_rn(ft, __fmul_rn(x2, __fsub_rn(tso[0], tso[1])));
            real denominator = __fadd_rn(one, x1sn1);
            denominator = __fadd_rn(denominator, x2);
            denominator = __fsub_rn(denominator, __fmul_rn(x2, cotso[RUC_NZS_M2]));
            cotso[RUC_NZS_M1] = __fdiv_rn(x1sn1, denominator);
            rhtso[RUC_NZS_M1] = __fdiv_rn(
                __fadd_rn(ft, __fmul_rn(x2, rhtso[RUC_NZS_M2])), denominator);
            real ftsnow = __fadd_rn(
                soilt1, __fmul_rn(x1sn, __fsub_rn(soilt, soilt1)));
            ftsnow = __fsub_rn(
                ftsnow, __fmul_rn(x1sn1, __fsub_rn(soilt1, tso[0])));
            real denomsn = __fadd_rn(one, x1sn);
            denomsn = __fadd_rn(denomsn, x1sn1);
            denomsn = __fsub_rn(denomsn, __fmul_rn(x1sn1, cotso[RUC_NZS_M1]));
            cotsn = __fdiv_rn(x1sn, denomsn);
            rhtsn = __fdiv_rn(
                __fadd_rn(ftsnow, __fmul_rn(x1sn1, rhtso[RUC_NZS_M1])), denomsn);
            tsnav = __fmul_rn(__fadd_rn(soilt, soilt1), deltsn);
            tsnav = __fadd_rn(
                tsnav,
                __fmul_rn(
                    __fadd_rn(soilt1, tso[0]),
                    __fsub_rn(snhei, deltsn)));
            tsnav = __fsub_rn(
                __fmul_rn(__fdiv_rn(half, snhei), tsnav), freeze);
        }
    }
    if (snhei < snth && snhei > zero) {
        // :5174-5198 snow too thin for its own layer: blend it with the top
        // soil layer.
        snprim = __fadd_rn(snhei, zsmain[1]);
        fsn = __fdiv_rn(snhei, snprim);
        fso = __fsub_rn(one, fsn);
        soilt1 = tso[0];
        tsob = tso[1];
        real xsn = __fdiv_rn(
            half_step,
            __fadd_rn(
                __fsub_rn(zshalf[2], zsmain[1]),
                __fmul_rn(half, snprim)));
        real ddzsn = __fdiv_rn(xsn, snprim);
        real x1sn = __fmul_rn(
            ddzsn,
            __fadd_rn(__fmul_rn(fsn, thdifsn), __fmul_rn(fso, thdif0)));
        real x2 = __fmul_rn(dtdzs[1], thdif[ncolumn + column]);
        real ft = __fadd_rn(
            tso[1], __fmul_rn(x1sn, __fsub_rn(soilt, tso[1])));
        ft = __fsub_rn(ft, __fmul_rn(x2, __fsub_rn(tso[1], tso[2])));
        real denominator = __fadd_rn(one, x1sn);
        denominator = __fadd_rn(denominator, x2);
        denominator = __fsub_rn(denominator, __fmul_rn(x2, cotso[RUC_NZS_M3]));
        cotso[RUC_NZS_M2] = __fdiv_rn(x1sn, denominator);
        rhtso[RUC_NZS_M2] = __fdiv_rn(
            __fadd_rn(ft, __fmul_rn(x2, rhtso[RUC_NZS_M3])), denominator);
        tsnav = __fsub_rn(
            __fmul_rn(half, __fadd_rn(soilt, tso[0])), freeze);
        cotso[RUC_NZS_M1] = cotso[RUC_NZS_M2];
        rhtso[RUC_NZS_M1] = rhtso[RUC_NZS_M2];
        cotsn = cotso[RUC_NZS_M1];
        rhtsn = rhtso[RUC_NZS_M1];
    }

    // :5203-5224 heat balance (Smirnova et al. 1996, eq. 21, 26).
    int nmelt = 0;
    real snoh = zero;
    real ett1 = zero;
    real epot = -__fmul_rn(qkms, __fsub_rn(qvatm, qgold));
    real rhcs = cap[column];
    real h = one;
    real trans = __fdiv_rn(
        __fmul_rn(transum, drycan), zshalf[root_count]);
    real can = __fadd_rn(wetcan, trans);
    real umveg = __fsub_rn(one, vegfrac);
    real d1 = cotso[RUC_NZS_M2];
    real d2 = rhtso[RUC_NZS_M2];
    real tn = soilt;
    real d9 = __fmul_rn(__fmul_rn(thdif0, rhcs), dzstop);
    real d10 = __fmul_rn(__fmul_rn(tkms, cp_air), rho);
    real r211 = __fdiv_rn(__fmul_rn(half, conflx[column]), delt);
    real r21 = __fmul_rn(__fmul_rn(r211, cp_air), rho);
    real r22 = __fdiv_rn(
        half,
        __fmul_rn(__fmul_rn(thdif0, delt), __fmul_rn(dzstop, dzstop)));
    real tn2 = __fmul_rn(tn, tn);
    real tn4 = __fmul_rn(tn2, tn2);
    real r6 = __fmul_rn(__fmul_rn(__fmul_rn(emiss, stbolt), half), tn4);
    real r7 = __fdiv_rn(r6, tn);
    real d11 = __fadd_rn(rnet, r6);

    // :5226-5270 the snow row of the tridiagonal system.
    real d1sn = zero;
    real d2sn = zero;
    real d9sn = zero;
    real r22sn = zero;
    if (snhei >= snth) {
        if (snhei <= __fadd_rn(deltsn, snth)) {
            d1sn = cotso[RUC_NZS_M1];
            d2sn = rhtso[RUC_NZS_M1];
        } else {
            d1sn = cotsn;
            d2sn = rhtsn;
        }
        d9sn = __fdiv_rn(__fmul_rn(thdifsn, rhocsn), snprim);
        r22sn = __fdiv_rn(
            __fmul_rn(__fmul_rn(snprim, snprim), half),
            __fmul_rn(thdifsn, delt));
    }
    if (snhei < snth && snhei > zero) {
        d1sn = d1;
        d2sn = d2;
        d9sn = __fdiv_rn(
            __fadd_rn(
                __fmul_rn(__fmul_rn(fsn, thdifsn), rhocsn),
                __fmul_rn(__fmul_rn(fso, thdif0), rhcs)),
            snprim);
        r22sn = __fdiv_rn(
            __fmul_rn(__fmul_rn(snprim, snprim), half),
            __fmul_rn(
                __fadd_rn(
                    __fmul_rn(fsn, thdifsn), __fmul_rn(fso, thdif0)),
                delt));
    }

    // :5279-5280 and :5290-5291.  These rain and new-snow heat terms do not
    // depend on anything the melt iteration updates, so WRF's second pass
    // through :5275 recomputes them identically.
    real rain_heat_capacity = __fmul_rn(__fmul_rn(rainf, cvw), prcpms);
    real snow_heat_capacity = __fdiv_rn(
        __fmul_rn(rhonewcsn, newsnow), delt);
    real rain_temperature = fmaxf(freeze, tabs);
    real snow_temperature = fminf(freeze, tabs);

    real qsg = zero;
    real qcg = zero;
    real snflx = zero;
    real storage = zero;
    real r210 = zero;
    bool failed = false;

    while (true) {
        // :5275 the melt iteration entry point.
        real tdenom = __fmul_rn(
            d9sn, __fadd_rn(__fsub_rn(one, d1sn), r22sn));
        tdenom = __fadd_rn(tdenom, d10);
        tdenom = __fadd_rn(tdenom, r21);
        tdenom = __fadd_rn(tdenom, r7);
        tdenom = __fadd_rn(tdenom, rain_heat_capacity);
        tdenom = __fadd_rn(tdenom, snow_heat_capacity);
        real fkq = __fmul_rn(qkms, rho);
        r210 = __fmul_rn(r211, rho);
        real c = __fmul_rn(__fmul_rn(vegfrac, fkq), can);
        real cc = __fdiv_rn(__fmul_rn(c, xlvm), tdenom);
        real evaporation = __fmul_rn(__fmul_rn(beta, fkq), umveg);
        real aa = __fdiv_rn(
            __fmul_rn(xlvm, __fadd_rn(evaporation, r210)), tdenom);
        real humidity_inner = __fmul_rn(qvatm, __fadd_rn(evaporation, c));
        humidity_inner = __fadd_rn(
            humidity_inner, __fmul_rn(r210, qgold));
        real numerator = __fmul_rn(d10, tabs);
        numerator = __fadd_rn(numerator, __fmul_rn(r21, tn));
        numerator = __fadd_rn(numerator, __fmul_rn(xlvm, humidity_inner));
        numerator = __fadd_rn(numerator, d11);
        numerator = __fadd_rn(
            numerator,
            __fmul_rn(d9sn, __fadd_rn(d2sn, __fmul_rn(r22sn, tn))));
        numerator = __fadd_rn(
            numerator, __fmul_rn(rain_heat_capacity, rain_temperature));
        numerator = __fadd_rn(
            numerator, __fmul_rn(snow_heat_capacity, snow_temperature));
        real bb = __fdiv_rn(numerator, tdenom);
        real aa1 = __fadd_rn(aa, cc);
        real pp = __fmul_rn(patm, thousand);
        aa1 = __fdiv_rn(aa1, pp);
        bb = __fsub_rn(bb, __fdiv_rn(snoh, tdenom));

        real qs1, ts1;
        if (!ruc_vilka(tn, aa1, bb, pp, tbq, &qs1, &ts1)) {
            failed = true;
            break;
        }
        real tq2 = qvatm;
        real tx2 = __fmul_rn(tq2, __fsub_rn(one, h));
        real q1 = __fadd_rn(tx2, __fmul_rn(h, qs1));
        bool saturated = !(q1 < qs1);
        if (!saturated) {
            // :5315-5328.  Unreachable while h == 1; transcribed anyway.
            bb = __fsub_rn(bb, __fmul_rn(aa, tx2));
            aa = __fdiv_rn(__fadd_rn(__fmul_rn(aa, h), cc), pp);
            if (!ruc_vilka(tn, aa, bb, pp, tbq, &qs1, &ts1)) {
                failed = true;
                break;
            }
            q1 = __fadd_rn(tx2, __fmul_rn(h, qs1));
            saturated = q1 > qs1;
        }
        if (saturated) {
            qvg = qs1;
            qsg = qs1;
            qcg = fmaxf(zero, __fsub_rn(q1, qs1));
        } else {
            qsg = qs1;
            qvg = q1;
            qcg = zero;
        }

        // :5332-5340 skin temperature.
        soilt = ts1;
        if (nmelt == 1 && snowfrac == one && snwe > zero && soilt > freeze) {
            soilt = fminf(freeze, soilt);
        }

        // :5348-5378 the snow-soil interface and the 7.5 cm level.
        if (snhei >= snth) {
            if (snhei > __fadd_rn(deltsn, snth)) {
                soilt1 = fminf(
                    freeze, __fadd_rn(rhtsn, __fmul_rn(cotsn, soilt)));
                tso[0] = __fadd_rn(rhtso[RUC_NZS_M1], __fmul_rn(cotso[RUC_NZS_M1], soilt1));
                tsob = soilt1;
            } else {
                tso[0] = __fadd_rn(rhtso[RUC_NZS_M1], __fmul_rn(cotso[RUC_NZS_M1], soilt));
                soilt1 = tso[0];
                tsob = tso[0];
            }
        } else if (snhei > zero && snhei < snth) {
            tso[1] = __fadd_rn(rhtso[RUC_NZS_M2], __fmul_rn(cotso[RUC_NZS_M2], soilt));
            tso[0] = __fadd_rn(
                tso[1], __fmul_rn(__fsub_rn(soilt, tso[1]), fso));
            soilt1 = tso[0];
            tsob = tso[1];
        } else {
            tso[0] = soilt;
            soilt1 = soilt;
            tsob = tso[0];
        }
        if (nmelt == 1 && snowfrac == one) {
            soilt1 = fminf(freeze, soilt1);
            tso[0] = fminf(freeze, tso[0]);
            tsob = fminf(freeze, tsob);
        }

        // :5382-5394 downward substitution.
        int start = (snhei > zero && snhei < snth) ? 2 : 1;
        for (int level = start; level < RUC_NZS; ++level) {
            int coefficient = RUC_NZS_M1 - level;
            tso[level] = __fadd_rn(
                rhtso[coefficient],
                __fmul_rn(cotso[coefficient], tso[level - 1]));
        }

        if (nmelt == 1) break;

        if (soilt > freeze && beta == one && snhei > zero) {
            // :5414-5553 top melt.
            nmelt = 1;
            soiltfrac = __fadd_rn(
                __fmul_rn(snowfrac, freeze),
                __fmul_rn(__fsub_rn(one, snowfrac), soilt));
            qsg = fminf(
                qsg, __fdiv_rn(ruc_qsn_lookup(soiltfrac, tbq), pp));
            qvg = __fadd_rn(
                __fmul_rn(snowfrac, qsg),
                __fmul_rn(__fsub_rn(one, snowfrac), qvg));
            // :5419-5421 t3/upflux/xinet are dead: xinet is never read.
            epot = -__fmul_rn(qkms, __fsub_rn(qvatm, qsg));
            q1 = __fmul_rn(epot, ras);
            real qfx;
            if (q1 <= zero) {
                dew = -epot;
                qfx = -__fmul_rn(__fmul_rn(xlvm, rho), dew);
            } else {
                for (int level = 0; level < root_count; ++level) {
                    real transp = -__fdiv_rn(
                        __fmul_rn(
                            __fmul_rn(
                                __fmul_rn(vegfrac, q1),
                                tranf[level * ncolumn + column]),
                            drycan),
                        zshalf[root_count]);
                    ett1 = __fsub_rn(ett1, transp);
                }
                real edir1 = __fmul_rn(__fmul_rn(q1, umveg), beta);
                real ec1 = __fmul_rn(__fmul_rn(q1, wetcan), vegfrac);
                real eeta = __fmul_rn(
                    __fadd_rn(__fadd_rn(edir1, ec1), ett1), thousand);
                qfx = __fmul_rn(xlvm, eeta);
            }
            real hfx = -__fmul_rn(d10, __fsub_rn(tabs, soiltfrac));
            real soh;
            if (snhei >= snth) {
                soh = __fdiv_rn(
                    __fmul_rn(
                        __fmul_rn(thdifsn, rhocsn),
                        __fsub_rn(soiltfrac, tsob)),
                    snprim);
            } else {
                soh = __fdiv_rn(
                    __fmul_rn(
                        __fadd_rn(
                            __fmul_rn(__fmul_rn(fsn, thdifsn), rhocsn),
                            __fmul_rn(__fmul_rn(fso, thdif0), rhcs)),
                        __fsub_rn(soiltfrac, tsob)),
                    snprim);
            }
            snflx = soh;
            storage = __fmul_rn(
                __fadd_rn(r21, __fmul_rn(d9sn, r22sn)),
                __fsub_rn(soiltfrac, tn));
            storage = __fadd_rn(
                storage,
                __fmul_rn(
                    __fmul_rn(xlvm, r210), __fsub_rn(qvg, qgold)));
            snoh = __fsub_rn(rnet, qfx);
            snoh = __fsub_rn(snoh, hfx);
            snoh = __fsub_rn(snoh, soh);
            snoh = __fsub_rn(snoh, storage);
            snoh = __fadd_rn(
                snoh,
                __fmul_rn(
                    snow_heat_capacity,
                    __fsub_rn(snow_temperature, soiltfrac)));
            snoh = __fadd_rn(
                snoh,
                __fmul_rn(
                    rain_heat_capacity,
                    __fsub_rn(rain_temperature, soiltfrac)));
            snoh = fmaxf(zero, snoh);
            smelt = __fmul_rn(__fdiv_rn(snoh, xlmelt), milli);
            real potential = __fmul_rn(__fmul_rn(epot, ras), delt);
            if (epot > zero && snwepr <= potential) {
                // :5483-5491 all the snow can evaporate; jump to :5518.
                beta = __fdiv_rn(snwepr, potential);
                smelt = fminf(
                    smelt,
                    __fsub_rn(
                        __fdiv_rn(snwepr, delt),
                        __fmul_rn(__fmul_rn(beta, epot), ras)));
                snwe = zero;
            } else {
                smelt = fmaxf(zero, smelt);
                // :5499-5501 the Egglston melt limiter.
                if ((rhosn < 350.0f
                     || (newsnow > zero && rhonewsn < 450.0f))
                    && soilt < 283.0f) {
                    real limit = __fdiv_rn(delt, 60.0f);
                    limit = __fmul_rn(limit, 5.6e-8f);
                    limit = __fmul_rn(limit, meltfactor);
                    limit = __fmul_rn(
                        limit, fmaxf(one, __fsub_rn(soilt, freeze)));
                    smelt = fminf(smelt, limit);
                }
                real rr = fmaxf(
                    zero,
                    __fsub_rn(
                        __fdiv_rn(snwepr, delt),
                        __fmul_rn(__fmul_rn(beta, epot), ras)));
                if (smelt > rr) {
                    smelt = fminf(smelt, rr);
                    snwe = zero;
                }
            }
            // :5518-5522.
            snoh = __fmul_rn(__fmul_rn(smelt, xlmelt), thousand);
            if (smelt > zero) {
                // :5529-5543 Koren et al. (1999) liquid retention.
                real rsmfrac = fminf(
                    0.18f,
                    fmaxf(
                        0.08f,
                        __fmul_rn(__fdiv_rn(snwepr, 0.10f), 0.13f)));
                if (snhei > 0.01f && rhosn < 350.0f) {
                    rsm = __fmul_rn(__fmul_rn(rsmfrac, smelt), delt);
                } else {
                    rsm = zero;
                }
                if (rsm > zero) {
                    smelt = fmaxf(
                        zero, __fsub_rn(smelt, __fdiv_rn(rsm, delt)));
                }
            }
            if (snwe > zero) {
                snwe = fmaxf(
                    zero,
                    __fsub_rn(
                        snwepr,
                        __fmul_rn(
                            __fadd_rn(
                                smelt,
                                __fmul_rn(__fmul_rn(beta, epot), ras)),
                            delt)));
            }
        } else {
            // :5557-5567 no melt: sublimation or condensation only.
            if (snhei != zero && beta == one) {
                epot = -__fmul_rn(qkms, __fsub_rn(qvatm, qsg));
                snwe = fmaxf(
                    zero,
                    __fsub_rn(
                        snwepr,
                        __fmul_rn(
                            __fmul_rn(__fmul_rn(beta, epot), ras), delt)));
            } else {
                snwe = zero;
            }
        }

        if (nmelt == 1) continue;
        break;
    }

    if (failed) {
        real invalid = nanf("");
        for (int level = 0; level < RUC_NZS; ++level) {
            tso_out[level * ncolumn + column] = invalid;
        }
        soilt_out[column] = invalid;
        soilt1_out[column] = invalid;
        tsnav_out[column] = invalid;
        qvg_out[column] = invalid;
        qsg_out[column] = invalid;
        qcg_out[column] = invalid;
        dew_out[column] = invalid;
        snwe_out[column] = invalid;
        snhei_out[column] = invalid;
        rhosn_out[column] = invalid;
        beta_out[column] = invalid;
        smelt_out[column] = invalid;
        snoh_out[column] = invalid;
        snflx_out[column] = invalid;
        s_out[column] = invalid;
        rsm_out[column] = invalid;
        snweprint_out[column] = invalid;
        snheiprint_out[column] = invalid;
        storage_out[column] = invalid;
        ilnb_out[column] = snow_layers;
        return;
    }

    // :5572-5613 melt water changes the snow density.  WRF only prints a
    // diagnostic when snwe <= rsm and leaves the density alone.
    if (smelt > zero && rsm > zero && snwe > rsm) {
        real xsn = __fdiv_rn(
            __fadd_rn(
                __fmul_rn(rhosn, __fsub_rn(snwe, rsm)),
                __fmul_rn(thousand, rsm)),
            snwe);
        rhosn = fminf(fmaxf(58.8f, xsn), 500.0f);
        rhocsn = __fmul_rn(2090.0f, rhosn);
        thdifsn = ruc_snow_thermal_diffusivity(
            rhosn, rhocsn, rhonewsn, newsnow, snhei);
    }

    // :5616-5629 flux in the top snow layer, then in the top soil layer.
    real s;
    if (snhei >= snth) {
        s = __fdiv_rn(
            __fmul_rn(
                __fmul_rn(thdifsn, rhocsn), __fsub_rn(soilt, tsob)),
            snprim);
    } else if (snhei < snth && snhei > zero) {
        s = __fdiv_rn(
            __fmul_rn(
                __fadd_rn(
                    __fmul_rn(__fmul_rn(fsn, thdifsn), rhocsn),
                    __fmul_rn(__fmul_rn(fso, thdif0), rhcs)),
                __fsub_rn(soilt, tsob)),
            snprim);
    } else {
        s = __fmul_rn(d9sn, __fsub_rn(soilt, tsob));
    }
    snflx = s;
    s = __fmul_rn(d9, __fsub_rn(tso[0], tso[1]));

    snhei = __fdiv_rn(__fmul_rn(snwe, thousand), rhosn);

    // :5636-5684 melt from the bottom of the pack on thawed ground.
    if (tso[0] > freeze && snhei > zero) {
        real hsn;
        if (snhei > __fadd_rn(deltsn, snth)) {
            hsn = __fsub_rn(snhei, deltsn);
        } else {
            hsn = snhei;
        }
        soiltfrac = __fadd_rn(
            __fmul_rn(snowfrac, freeze),
            __fmul_rn(__fsub_rn(one, snowfrac), tso[0]));
        real snohg = __fadd_rn(
            __fmul_rn(cap[column], zshalf[1]),
            __fmul_rn(__fmul_rn(rhocsn, half), hsn));
        snohg = __fdiv_rn(
            __fmul_rn(__fsub_rn(tso[0], soiltfrac), snohg), delt);
        snohg = fmaxf(zero, snohg);
        real smeltg = __fmul_rn(__fdiv_rn(snohg, xlmelt), milli);
        // :5658-5660 the Egglston bottom-melt limit.
        if ((rhosn < 350.0f || (newsnow > zero && rhonewsn < 450.0f))
            && soilt < 283.0f) {
            smeltg = fminf(smeltg, 5.8e-9f);
        }
        real rr = __fdiv_rn(snwe, delt);
        smeltg = fminf(smeltg, rr);
        snwe = fmaxf(zero, __fsub_rn(snwe, __fmul_rn(smeltg, delt)));
        snhei = __fdiv_rn(__fmul_rn(snwe, thousand), rhosn);
        smelt = __fadd_rn(smelt, smeltg);
        if (snhei > zero) tso[0] = soiltfrac;
    }

    real snweprint = snwe;
    real snheiprint = __fdiv_rn(__fmul_rn(snweprint, thousand), rhosn);

    // :5697-5708 surface heat storage.
    storage = __fmul_rn(
        __fadd_rn(r21, __fmul_rn(d9sn, r22sn)), __fsub_rn(soilt, tn));
    storage = __fadd_rn(
        storage,
        __fmul_rn(__fmul_rn(xlvm, r210), __fsub_rn(qsg, qgold)));
    storage = __fsub_rn(
        storage,
        __fmul_rn(
            snow_heat_capacity, __fsub_rn(snow_temperature, soilt)));
    storage = __fsub_rn(
        storage,
        __fmul_rn(
            rain_heat_capacity, __fsub_rn(rain_temperature, soilt)));

    // :5715-5725 mean snow-pack temperature in Celsius.
    if (snhei > zero) {
        if (snow_layers > 1) {
            tsnav = __fmul_rn(__fadd_rn(soilt, soilt1), deltsn);
            tsnav = __fadd_rn(
                tsnav,
                __fmul_rn(
                    __fadd_rn(soilt1, tso[0]),
                    __fsub_rn(snhei, deltsn)));
            tsnav = __fsub_rn(
                __fmul_rn(__fdiv_rn(half, snhei), tsnav), freeze);
        } else {
            tsnav = __fsub_rn(
                __fmul_rn(half, __fadd_rn(soilt, tso[0])), freeze);
        }
    } else {
        tsnav = __fsub_rn(soilt, freeze);
    }

    for (int level = 0; level < RUC_NZS; ++level) {
        tso_out[level * ncolumn + column] = tso[level];
    }
    soilt_out[column] = soilt;
    soilt1_out[column] = soilt1;
    tsnav_out[column] = tsnav;
    qvg_out[column] = qvg;
    qsg_out[column] = qsg;
    qcg_out[column] = qcg;
    dew_out[column] = dew;
    snwe_out[column] = snwe;
    snhei_out[column] = snhei;
    rhosn_out[column] = rhosn;
    beta_out[column] = beta;
    smelt_out[column] = smelt;
    snoh_out[column] = snoh;
    snflx_out[column] = snflx;
    s_out[column] = s;
    rsm_out[column] = rsm;
    snweprint_out[column] = snweprint;
    snheiprint_out[column] = snheiprint;
    storage_out[column] = storage;
    ilnb_out[column] = snow_layers;
}


// ---------------------------------------------------------------------------
// module_sf_ruclsm.F:3120-3786 subroutine snowsoil, the snow-covered land
// column, split into the four column-local kernels that surround the already
// oracled soilprop/transf/soilmoist launches.  The :4836-5728 snowtemp solve
// is ruc_snow_temperature_step above, which snowsoil launches directly;
// ruc_snow_layer_thresholds below supplies the two arguments LSMRUC derives
// for it at :3387-3398.
//
// Every arithmetic boundary is pinned with __fadd_rn/__fsub_rn/__fmul_rn/
// __fdiv_rn: NVRTC contracts a*b+c into an FMA where gfortran does not, and
// the melt iteration feeds each rounding into the next pass.
//
// snhei_crit, gsw, alb, znt, ivgtyp and drip are arguments snowsoil never
// reads (the :3654-3666 soilmoist call passes a literal 0. where drip would
// go), and glw reaches only the :3701 xinet that is assigned and never used,
// so none of them appear here.
// ---------------------------------------------------------------------------


// module_sf_ruclsm.F:3535-3554 the all-snow-can-evaporate limiter beta, the
// canopy wet/dry split, and the ras and post-limiter snwe the later kernels
// consume.
extern "C" __global__
void ruc_snow_soil_canopy_setup(
    const real* __restrict__ qvatm_a,
    const real* __restrict__ qsg_a,
    const real* __restrict__ qkms_a,
    const real* __restrict__ rho_a,
    const real* __restrict__ vegfrac_a,
    const real* __restrict__ snwe_a,
    const real* __restrict__ cst_a,
    const real* __restrict__ sat_a,
    const real* __restrict__ cn_a,
    real delt,
    real* __restrict__ beta_out,
    real* __restrict__ wetcan_out,
    real* __restrict__ drycan_out,
    real* __restrict__ snwe_out,
    real* __restrict__ ras_out,
    int ncolumn)
{
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= ncolumn) return;

    const real one = 1.0f;
    real ras = __fmul_rn(rho_a[column], 1.0e-3f);
    real umveg = __fsub_rn(one, vegfrac_a[column]);
    real epot = -__fmul_rn(
        qkms_a[column], __fsub_rn(qvatm_a[column], qsg_a[column]));
    real snwepr = snwe_a[column];
    real snwe = snwepr;
    real beta = one;
    real epdt = __fmul_rn(epot, ras);
    epdt = __fmul_rn(epdt, delt);
    epdt = __fmul_rn(epdt, umveg);
    if (epdt > 0.0f && snwepr <= epdt) {
        beta = __fdiv_rn(snwepr, fmaxf(1.0e-8f, epdt));
        snwe = 0.0f;
    }
    real ratio = fmaxf(0.0f, __fdiv_rn(cst_a[column], sat_a[column]));
    real wetcan = fminf(0.25f, ruc_powf_rn(ratio, cn_a[column]));

    beta_out[column] = beta;
    wetcan_out[column] = wetcan;
    drycan_out[column] = __fsub_rn(one, wetcan);
    snwe_out[column] = snwe;
    ras_out[column] = ras;
}


// module_sf_ruclsm.F:3387-3398 -- the snow-layer thresholds LSMRUC's snow
// branch derives from the entry pack density and hands to snowtemp: deltsn,
// the thickness of the upper snow layer, and snth, the minimum thickness a
// layer may have.  A pack only just deeper than the two together is split
// evenly instead (:3394-3397).
//
// The two source depths live in __constant__ memory for the reason recorded
// on ruc_soil_layer_depth above: as local literals they are compile-time
// constants, so the PTX->SASS backend, not the SM, would evaluate the
// products that build deltsn and snth.
extern "C" __global__
void ruc_snow_layer_thresholds(
    const real* __restrict__ rhosn_a,
    const real* __restrict__ snhei_a,
    real* __restrict__ deltsn_out,
    real* __restrict__ snth_out,
    int ncolumn)
{
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= ncolumn) return;

    const real half = 0.5f;
    const real thousand = 1.0e3f;
    real rhosn = rhosn_a[column];
    real snhei = snhei_a[column];
    real deltsn = __fdiv_rn(
        __fmul_rn(ruc_snow_layer_threshold_depth[0], thousand), rhosn);
    real snth = __fdiv_rn(
        __fmul_rn(ruc_snow_layer_threshold_depth[1], thousand), rhosn);
    if (snhei >= __fadd_rn(deltsn, snth)) {
        if (__fsub_rn(__fsub_rn(snhei, deltsn), snth) < snth) {
            deltsn = __fmul_rn(half, __fsub_rn(snhei, snth));
        }
    }
    deltsn_out[column] = deltsn;
    snth_out[column] = snth;
}


// module_sf_ruclsm.F:3604-3626 recompute dew or root-zone transpiration
// against the qsg snowtemp returned, plus the -infwater that :3658 passes as prcp.
extern "C" __global__
void ruc_snow_soil_prepare_moisture(
    const real* __restrict__ qvatm_a,
    const real* __restrict__ qsg_a,
    const real* __restrict__ qkms_a,
    const real* __restrict__ vegfrac_a,
    const real* __restrict__ drycan_a,
    const real* __restrict__ ras_a,
    const real* __restrict__ tranf,
    const int* __restrict__ nroot_a,
    const real* __restrict__ infwater_a,
    real* __restrict__ transp,
    real* __restrict__ ett1_out,
    real* __restrict__ dew_out,
    real* __restrict__ prcp_out,
    int ncolumn)
{
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= ncolumn) return;

    const real zero = 0.0f;
    real zshalf[RUC_NZS];
    const real* zsmain = ruc_soil_layer_depth;
    zshalf[0] = zero;
    for (int level = 1; level < RUC_NZS; ++level) {
        zshalf[level] = __fmul_rn(
            __fadd_rn(zsmain[level - 1], zsmain[level]), 0.5f);
    }

    real dew = zero;
    real ett1 = zero;
    real deficit = __fsub_rn(qvatm_a[column], qsg_a[column]);
    real epot = -__fmul_rn(qkms_a[column], deficit);
    int nroot = nroot_a[column];
    for (int level = 0; level < RUC_NZS; ++level) {
        transp[level * ncolumn + column] = zero;
    }
    if (epot > zero) {
        for (int level = 0; level < nroot; ++level) {
            real flux = __fmul_rn(vegfrac_a[column], ras_a[column]);
            flux = __fmul_rn(flux, qkms_a[column]);
            flux = __fmul_rn(flux, deficit);
            flux = __fmul_rn(flux, tranf[level * ncolumn + column]);
            flux = __fmul_rn(flux, drycan_a[column]);
            flux = __fdiv_rn(flux, zshalf[nroot]);
            transp[level * ncolumn + column] = flux;
            ett1 = __fsub_rn(ett1, flux);
        }
    } else {
        dew = -epot;
    }
    ett1_out[column] = ett1;
    dew_out[column] = dew;
    prcp_out[column] = -infwater_a[column];
}


// module_sf_ruclsm.F:3671-3771 snowsoil's closing bookkeeping: the melted-out
// snow average, accumulated melt, the keepfr latch, and the surface fluxes.
extern "C" __global__
void ruc_snow_soil_finalize(
    const real* __restrict__ soilice,
    const real* __restrict__ tso,
    const real* __restrict__ told,
    const real* __restrict__ soilmois,
    const real* __restrict__ smold,
    real* __restrict__ keepfr,
    const real* __restrict__ snhei_a,
    const real* __restrict__ smelt_a,
    const real* __restrict__ snom_in,
    const real* __restrict__ snflx_a,
    const real* __restrict__ snoh_a,
    const real* __restrict__ storage_a,
    const real* __restrict__ soilt_a,
    const real* __restrict__ qsg_a,
    const real* __restrict__ tsnav_in,
    const real* __restrict__ beta_a,
    const real* __restrict__ wetcan_a,
    const real* __restrict__ ett1_in,
    const real* __restrict__ dew_in,
    const real* __restrict__ ras_a,
    const real* __restrict__ cst_in,
    const real* __restrict__ tkms_a,
    const real* __restrict__ rho_a,
    const real* __restrict__ tabs_a,
    const real* __restrict__ patm_a,
    const real* __restrict__ qkms_a,
    const real* __restrict__ qvatm_a,
    const real* __restrict__ vegfrac_a,
    const real* __restrict__ rnet_a,
    real delt,
    real xlvm,
    real* __restrict__ tsnav_out,
    real* __restrict__ snom_out,
    real* __restrict__ cst_out,
    real* __restrict__ dew_out,
    real* __restrict__ ett1_out,
    real* __restrict__ edir1_out,
    real* __restrict__ ec1_out,
    real* __restrict__ eeta_out,
    real* __restrict__ qfx_out,
    real* __restrict__ hfx_out,
    real* __restrict__ sublim_out,
    real* __restrict__ fltot_out,
    int ncolumn)
{
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= ncolumn) return;

    const real zero = 0.0f;
    const real one = 1.0f;
    const real freeze = 273.15f;
    const real cp_air = 1004.5f;
    const real rovcp = __fdiv_rn(287.0f, cp_air);
    // p1000mb*0.00001 from the :3703 Exner factor; exactly 1 in float32.
    const real reference = __fmul_rn(100000.0f, 0.00001f);

    real soilt = soilt_a[column];
    real qsg = qsg_a[column];

    // :3671-3677 restore the snow-free average when the pack is gone.
    tsnav_out[column] = (snhei_a[column] == zero)
        ? __fsub_rn(soilt, freeze) : tsnav_in[column];
    snom_out[column] = __fadd_rn(
        snom_in[column],
        __fmul_rn(__fmul_rn(smelt_a[column], delt), 1.0e3f));

    // :3688-3696 latch keepfr where rain fell on frozen soil.
    for (int level = 0; level < RUC_NZS; ++level) {
        int index = level * ncolumn + column;
        if (soilice[index] > zero) {
            keepfr[index] =
                (tso[index] > told[index] && soilmois[index] > smold[index])
                ? one : zero;
        }
    }

    // :3699-3701 t3/upflux/xinet are assigned and never read.
    real sensible = __fmul_rn(
        __fmul_rn(tkms_a[column], cp_air), rho_a[column]);
    sensible = __fmul_rn(sensible, __fsub_rn(tabs_a[column], soilt));
    real hft = -sensible;
    real exner = ruc_powf_rn(__fdiv_rn(reference, patm_a[column]), rovcp);
    hfx_out[column] = -__fmul_rn(sensible, exner);

    real deficit = __fsub_rn(qvatm_a[column], qsg);
    real q1 = -__fmul_rn(
        __fmul_rn(qkms_a[column], ras_a[column]), deficit);
    real umveg = __fsub_rn(one, vegfrac_a[column]);
    real edir1 = zero;
    real ec1 = zero;
    real ett1 = ett1_in[column];
    // The evaporation branch never touches dew, so it keeps whatever the
    // :3604-3626 recomputation left there.
    real dew = dew_in[column];
    real eeta;
    real cst;
    if (q1 < zero) {
        ett1 = zero;
        dew = __fmul_rn(qkms_a[column], deficit);
        eeta = -__fmul_rn(rho_a[column], dew);
        cst = __fadd_rn(
            cst_in[column],
            __fmul_rn(
                __fmul_rn(__fmul_rn(delt, dew), ras_a[column]),
                vegfrac_a[column]));
        qfx_out[column] = __fmul_rn(xlvm, eeta);
        eeta = -__fmul_rn(rho_a[column], dew);
    } else {
        edir1 = __fmul_rn(__fmul_rn(q1, umveg), beta_a[column]);
        ec1 = __fmul_rn(
            __fmul_rn(q1, wetcan_a[column]), vegfrac_a[column]);
        cst = fmaxf(
            zero, __fsub_rn(cst_in[column], __fmul_rn(ec1, delt)));
        real total = __fadd_rn(__fadd_rn(edir1, ec1), ett1);
        eeta = __fmul_rn(total, 1.0e3f);
        qfx_out[column] = __fmul_rn(xlvm, eeta);
        eeta = __fmul_rn(total, 1.0e3f);
    }
    cst_out[column] = cst;
    dew_out[column] = dew;
    ett1_out[column] = ett1;
    edir1_out[column] = edir1;
    ec1_out[column] = ec1;
    eeta_out[column] = eeta;
    sublim_out[column] = __fmul_rn(edir1, 1.0e3f);

    real balance = __fsub_rn(rnet_a[column], hft);
    balance = __fsub_rn(balance, __fmul_rn(xlvm, eeta));
    balance = __fsub_rn(balance, snflx_a[column]);
    balance = __fsub_rn(balance, snoh_a[column]);
    balance = __fsub_rn(balance, storage_a[column]);
    fltot_out[column] = balance;
}
