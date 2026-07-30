// Exact WRF v4.6.1 NSSL QVEXCESS and its default max-supersaturation caller.
//
// Numerical authority:
//   phys/module_mp_nssl_2mom.F:6052-6199  (pure-return QVEXCESS)
//   phys/module_mp_nssl_2mom.F:11537-11567 (caller update, imaxsupopt=4)
//
// QVEXCESS mutates only private trial variables and returns qvex. The pure
// kernels below never write qv, qc, theta, number, or CCN inputs. The separate
// caller kernel applies the WRF update explicitly. Number fields in that
// caller kernel are already NSSL-native concentrations (#/m3); there is no
// Registry gather/scatter or density unit conversion.

__device__ __forceinline__ int nssl2_qvexcess_table_index(float temperature)
{
    const float index_value = __fadd_rn(
        __fdiv_rn(__fsub_rn(temperature, 163.15f), 0.002f), 1.5f);
    // The official single-precision gfortran oracle converts a positive
    // out-of-int32 trial index to INT_MIN before applying MIN/MAX, hence the
    // low table endpoint.  Preserve that source/compiler result explicitly;
    // CUDA otherwise optimizes the later clamp to the high endpoint.
    if (!(index_value < 2147483648.0f)) return 1;
    if (index_value <= 1.0f) return 1;
    if (index_value >= 1000001.0f) return 1000001;
    return (int)index_value;
}

__device__ __forceinline__ float nssl2_qvexcess_table_value(
    float temperature)
{
    const int index = nssl2_qvexcess_table_index(temperature);
    const float table_temperature = __fadd_rn(
        163.15f, __fmul_rn((float)(index - 1), 0.002f));
    const float exponent = __fdiv_rn(
        __fmul_rn(
            17.2693882f,
            __fsub_rn(table_temperature, 273.15f)),
        __fsub_rn(table_temperature, 35.86f));
    // WRF fills TABQVS with the host single-precision EXP intrinsic.  A
    // correctly rounded double evaluation followed by an FP32 cast matches
    // that table, while CUDA's fast expf differs by an ulp at branch ties.
    return (float)exp((double)exponent);
}

__device__ __forceinline__ float nssl2_qvexcess_exact(
    float theta_base,
    float theta_perturbation,
    float pressure,
    float exner,
    float qv_base,
    float qv_perturbation,
    float cloud_initial,
    float condensation_factor,
    float latent_over_cp,
    float target_supersaturation_percent,
    int* trace_branch,
    float* trace_target_qv,
    float* trace_qv,
    float* trace_qc,
    float* trace_theta_perturbation)
{
    const float pressure_factor = __fdiv_rn(380.0f, pressure);
    float trial_theta_perturbation = theta_perturbation;
    float theta = __fadd_rn(trial_theta_perturbation, theta_base);
    float trial_qv_perturbation = qv_perturbation;
    float vapor = fmaxf(
        __fadd_rn(trial_qv_perturbation, qv_base), 0.0f);
    float temperature = __fmul_rn(theta, exner);
    float trial_vapor = fmaxf(0.0f, vapor);
    float trial_cloud = fmaxf(0.0f, cloud_initial);
    float saturation = __fmul_rn(
        pressure_factor, nssl2_qvexcess_table_value(temperature));
    const float target_factor = __fadd_rn(
        __fmul_rn(0.01f, target_supersaturation_percent), 1.0f);
    float target_vapor = __fmul_rn(target_factor, saturation);

    for (int iteration = 0; iteration < 2; ++iteration) {
        float cloud_delta = 0.0f;
        float vapor_delta = __fsub_rn(trial_vapor, target_vapor);
        int branch = 0;

        if (vapor_delta < 0.0f) {
            if (trial_cloud > -vapor_delta) {
                cloud_delta = vapor_delta;
                vapor_delta = 0.0f;
                branch = -1;
            } else {
                cloud_delta = -trial_cloud;
                vapor_delta = __fadd_rn(vapor_delta, trial_cloud);
                branch = -2;
            }
            trial_qv_perturbation = __fsub_rn(
                trial_qv_perturbation, cloud_delta);
            trial_cloud = __fadd_rn(trial_cloud, cloud_delta);
            // The evaporation branch is written in WRF as
            // (1/pi0) * (felvcp*dqcw), unlike the later condensation branch.
            const float theta_delta = __fmul_rn(
                __fdiv_rn(1.0f, exner),
                __fmul_rn(latent_over_cp, cloud_delta));
            trial_theta_perturbation = __fadd_rn(
                trial_theta_perturbation, theta_delta);
        }

        if (vapor_delta >= 0.0f) {
            const float temperature_offset = __fsub_rn(
                temperature, 35.86f);
            const float condensed_vapor = __fdiv_rn(
                vapor_delta,
                __fadd_rn(
                    1.0f,
                    __fdiv_rn(
                        __fmul_rn(condensation_factor, target_vapor),
                        __fmul_rn(
                            temperature_offset, temperature_offset))));
            cloud_delta = condensed_vapor;
            if (vapor_delta > 0.0f) branch = 1;
            trial_theta_perturbation = __fadd_rn(
                trial_theta_perturbation,
                __fdiv_rn(
                    __fmul_rn(latent_over_cp, cloud_delta), exner));
            trial_qv_perturbation = __fsub_rn(
                trial_qv_perturbation, condensed_vapor);
            trial_cloud = __fadd_rn(trial_cloud, cloud_delta);
        }

        theta = __fadd_rn(trial_theta_perturbation, theta_base);
        temperature = __fmul_rn(theta, exner);
        vapor = fmaxf(
            __fadd_rn(trial_qv_perturbation, qv_base), 0.0f);
        saturation = __fmul_rn(
            pressure_factor, nssl2_qvexcess_table_value(temperature));
        trial_cloud = fmaxf(0.0f, trial_cloud);
        trial_vapor = fmaxf(0.0f, vapor);
        target_vapor = __fmul_rn(target_factor, saturation);

        if (trace_branch != nullptr) {
            trace_branch[iteration] = branch;
            trace_target_qv[iteration] = target_vapor;
            trace_qv[iteration] = trial_vapor;
            trace_qc[iteration] = trial_cloud;
            trace_theta_perturbation[iteration] =
                trial_theta_perturbation;
        }
    }

    return fmaxf(0.0f, __fsub_rn(trial_cloud, cloud_initial));
}

extern "C" __global__ void nssl2_qvexcess_split(
    const float* __restrict__ theta_base,
    const float* __restrict__ theta_perturbation,
    const float* __restrict__ pressure_pa,
    const float* __restrict__ exner,
    const float* __restrict__ qv_base,
    const float* __restrict__ qv_perturbation,
    const float* __restrict__ qc,
    const float* __restrict__ condensation_factor,
    const float* __restrict__ latent_over_cp,
    float target_supersaturation_percent,
    float* __restrict__ qvex_output,
    int n)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;
    qvex_output[idx] = nssl2_qvexcess_exact(
        theta_base[idx], theta_perturbation[idx],
        pressure_pa[idx], exner[idx],
        qv_base[idx], qv_perturbation[idx], qc[idx],
        condensation_factor[idx], latent_over_cp[idx],
        target_supersaturation_percent,
        nullptr, nullptr, nullptr, nullptr, nullptr);
}

extern "C" __global__ void nssl2_qvexcess_workspace(
    const float* __restrict__ full_theta,
    const float* __restrict__ pressure_pa,
    const float* __restrict__ exner,
    const float* __restrict__ qv,
    const float* __restrict__ qc,
    const float* __restrict__ condensation_factor,
    const float* __restrict__ latent_over_cp,
    float target_supersaturation_percent,
    float* __restrict__ qvex_output,
    int n)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;
    qvex_output[idx] = nssl2_qvexcess_exact(
        full_theta[idx], 0.0f, pressure_pa[idx], exner[idx],
        qv[idx], 0.0f, qc[idx],
        condensation_factor[idx], latent_over_cp[idx],
        target_supersaturation_percent,
        nullptr, nullptr, nullptr, nullptr, nullptr);
}

extern "C" __global__ void nssl2_qvexcess_trace_split(
    const float* __restrict__ theta_base,
    const float* __restrict__ theta_perturbation,
    const float* __restrict__ pressure_pa,
    const float* __restrict__ exner,
    const float* __restrict__ qv_base,
    const float* __restrict__ qv_perturbation,
    const float* __restrict__ qc,
    const float* __restrict__ condensation_factor,
    const float* __restrict__ latent_over_cp,
    float target_supersaturation_percent,
    float* __restrict__ qvex_output,
    float* __restrict__ branch_iteration1,
    float* __restrict__ target_qv_iteration1,
    float* __restrict__ qv_iteration1,
    float* __restrict__ qc_iteration1,
    float* __restrict__ theta_perturbation_iteration1,
    float* __restrict__ branch_iteration2,
    float* __restrict__ target_qv_iteration2,
    float* __restrict__ qv_iteration2,
    float* __restrict__ qc_iteration2,
    float* __restrict__ theta_perturbation_iteration2,
    int n)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;
    int branch[2];
    float target_qv[2];
    float trial_qv[2];
    float trial_qc[2];
    float trial_theta_perturbation[2];
    qvex_output[idx] = nssl2_qvexcess_exact(
        theta_base[idx], theta_perturbation[idx],
        pressure_pa[idx], exner[idx],
        qv_base[idx], qv_perturbation[idx], qc[idx],
        condensation_factor[idx], latent_over_cp[idx],
        target_supersaturation_percent,
        branch, target_qv, trial_qv, trial_qc,
        trial_theta_perturbation);
    branch_iteration1[idx] = (float)branch[0];
    target_qv_iteration1[idx] = target_qv[0];
    qv_iteration1[idx] = trial_qv[0];
    qc_iteration1[idx] = trial_qc[0];
    theta_perturbation_iteration1[idx] =
        trial_theta_perturbation[0];
    branch_iteration2[idx] = (float)branch[1];
    target_qv_iteration2[idx] = target_qv[1];
    qv_iteration2[idx] = trial_qv[1];
    qc_iteration2[idx] = trial_qc[1];
    theta_perturbation_iteration2[idx] =
        trial_theta_perturbation[1];
}

extern "C" __global__ void nssl2_qvexcess_apply_maxsup_default(
    float* __restrict__ full_theta,
    const float* __restrict__ air_density,
    const float* __restrict__ exner,
    float* __restrict__ qv,
    float* __restrict__ qc,
    float* __restrict__ cloud_number,
    float* __restrict__ ccn_number,
    const float* __restrict__ background_ccn,
    const float* __restrict__ cloud_mean_mass,
    const float* __restrict__ latent_over_cp,
    const float* __restrict__ qvex,
    float* __restrict__ new_cloud_number_output,
    int couple_number,
    int n)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;

    const float vapor_excess = qvex[idx];
    new_cloud_number_output[idx] = 0.0f;
    if (!(vapor_excess > 0.0f)) return;

    full_theta[idx] =
        full_theta[idx]
        + (latent_over_cp[idx] * vapor_excess) / exner[idx];
    qv[idx] = qv[idx] - vapor_excess;
    qc[idx] = qc[idx] + vapor_excess;

    if (couple_number != 0) {
        const float cloud_five_micron_mass =
            1000.0f * 0.523599f
            * (10.0e-6f * 10.0e-6f * 10.0e-6f);
        const float cloud_twenty_micron_mass =
            1000.0f * 0.523599f
            * (40.0e-6f * 40.0e-6f * 40.0e-6f);
        const float limiting_mass = fmaxf(
            cloud_five_micron_mass,
            fmaxf(cloud_twenty_micron_mass, cloud_mean_mass[idx]));
        const float new_cloud_number = fminf(
            fmaxf(ccn_number[idx], background_ccn[idx]),
            air_density[idx] * vapor_excess / limiting_mass);
        cloud_number[idx] = cloud_number[idx] + new_cloud_number;
        ccn_number[idx] = fmaxf(0.0f, ccn_number[idx] - new_cloud_number);
        new_cloud_number_output[idx] = new_cloud_number;
    }
}
