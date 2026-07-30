/* Kain--Fritsch cumulus, one CUDA thread per mass-grid column.
 *
 * Mirror authority: gpuwm.verify.npref.np_kf_column.
 * WRF v4.6.1 transcription anchors (phys/module_cu_kfeta.F):
 *   763-815    saturation and 50-hPa updraft-source search
 *   914-1046   LCL and grid-scale-vertical-velocity trigger
 *   1071-1320  entraining/detraining updraft and condensate loading
 *   1585-1822  advective timescale and humidity-scaled downdraft
 *   1875-2281  mass-flux/CAPE closure (90% removal)
 *   2503-2646  tendencies and convective-rain output
 *   3009-3039  KF_LUTAB bilinear interpolation
 */

/* Compile-time bound on kf_column's ~47 per-thread column arrays.  The
 * launcher (gpuwm/core/kf.py) specializes it to the configuration's own nz
 * through get_kernel_int_defines; 128 is the unspecialized ceiling this
 * source keeps when nothing overrides it, and the value gpuwm/core/kf.py
 * validates nz against.  The bound is a pure array extent -- no expression
 * in this kernel reads it -- so specializing it moves no arithmetic; the
 * frame it costs per thread is 188 B per level, and on this card the driver
 * reserves (frame - 1024) * 1536 * 170 bytes at the kernel's first launch,
 * i.e. 5,738 MiB at 128 against 2,039 MiB at 49.
 */
#ifndef KF_KMAX
#define KF_KMAX 128
#endif
#define KF_PHASE_WARM_RAIN 0
#define KF_PHASE_NO_SEPARATE_SNOW 1
#define KF_PHASE_SEPARATE_SNOW 2
#define KF_PHASE_SEPARATE_ICE_SNOW 3
#define KF_RLF 3.339e5f

__device__ __forceinline__ float kf_clip(float value, float lower,
                                         float upper) {
    return fminf(fmaxf(value, lower), upper);
}

__device__ __forceinline__ float kf_qsat(float temperature, float pressure) {
    float es = 611.2f * expf((17.67f * temperature - 17.67f * 273.15f)
                            / (temperature - 29.65f));
    es = fminf(es, 0.99f * pressure);
    return 0.622f * es / (pressure - es);
}

__device__ __forceinline__ float kf_thetae(
        float pressure, float temperature, float qv,
        const float* __restrict__ log_ratio) {
    float q = fmaxf(qv, 1.0e-9f);
    float ee = q * pressure / (0.622f + q);
    float a1 = fmaxf(ee / 611.2f, 0.001f);
    float position = (a1 - 0.001f) / 0.075f;
    int index = min(max((int)truncf(position), 0), 198);
    float base = 0.001f + 0.075f * index;
    float fraction = kf_clip((a1 - base) / 0.075f, 0.0f, 1.0f);
    float tlog = (1.0f - fraction) * log_ratio[index]
                 + fraction * log_ratio[index + 1];
    float dewpoint = (17.67f * 273.15f - 29.65f * tlog) / (17.67f - tlog);
    float tsat = dewpoint
        - (0.212f + 1.571e-3f * (dewpoint - 273.16f)
           - 4.36e-4f * (temperature - 273.16f))
          * (temperature - dewpoint);
    float theta = temperature * powf(1.0e5f / pressure,
                                      0.2854f * (1.0f - 0.28f * qv));
    return theta * expf((3374.6525f / fmaxf(tsat, 150.0f) - 2.5403f)
                        * qv * (1.0f + 0.81f * qv));
}

__device__ __forceinline__ void kf_table_parcel(
        float pressure, float thetae,
        const float* __restrict__ temperature_table,
        const float* __restrict__ qsat_table,
        const float* __restrict__ thetae_base,
        float pressure_top, float pressure_reciprocal,
        float thetae_reciprocal, float* temperature, float* qsat) {
    float pressure_position = (pressure - pressure_top) * pressure_reciprocal;
    float fp = pressure_position - truncf(pressure_position);
    int ip = (int)pressure_position;
    float base = (thetae_base[ip + 1] - thetae_base[ip]) * fp
                 + thetae_base[ip];
    float theta_position = (thetae - base) * thetae_reciprocal;
    float ft = theta_position - truncf(theta_position);
    int it = (int)theta_position;
    int i00 = it * 220 + ip;
    int i10 = (it + 1) * 220 + ip;
    int i01 = i00 + 1;
    int i11 = i10 + 1;
    float t00 = temperature_table[i00], t10 = temperature_table[i10];
    float t01 = temperature_table[i01], t11 = temperature_table[i11];
    float q00 = qsat_table[i00], q10 = qsat_table[i10];
    float q01 = qsat_table[i01], q11 = qsat_table[i11];
    *temperature = t00 + (t10-t00)*ft + (t01-t00)*fp
                   + (t00-t10-t01+t11)*ft*fp;
    *qsat = q00 + (q10-q00)*ft + (q01-q00)*fp
            + (q00-q10-q01+q11)*ft*fp;
}

__device__ __forceinline__ void kf_prof5(float equilibrium,
                                          float* entrainment,
                                          float* detrainment) {
    const float sqrt_two_pi = 2.506628f;
    const float a1 = 0.4361836f, a2 = -0.1201676f, a3 = 0.9372980f;
    const float p = 0.33267f, sigma = 0.166666667f;
    const float normalization = 0.202765151f;
    float y = 6.0f * equilibrium - 3.0f;
    float ey = expf(-0.5f * y * y);
    float e45 = expf(-4.5f);
    float t2 = 1.0f / (1.0f + p * fabsf(y));
    float t1 = 0.500498f;
    float c1 = a1 * t1 + a2 * t1 * t1 + a3 * t1 * t1 * t1;
    float c2 = a1 * t2 + a2 * t2 * t2 + a3 * t2 * t2 * t2;
    if (y >= 0.0f) {
        *entrainment = (sigma * (0.5f * (sqrt_two_pi - e45 * c1 - ey * c2)
                                  + sigma * (e45 - ey))
                        - e45 * equilibrium * equilibrium / 2.0f);
        *detrainment = (sigma * (0.5f * (ey * c2 - e45 * c1)
                                  + sigma * (e45 - ey))
                        - e45 * (0.5f + equilibrium * equilibrium / 2.0f
                                 - equilibrium));
    } else {
        *entrainment = (sigma * (0.5f * (ey * c2 - e45 * c1)
                                  + sigma * (e45 - ey))
                        - e45 * equilibrium * equilibrium / 2.0f);
        *detrainment = (sigma * (0.5f * (sqrt_two_pi - e45 * c1 - ey * c2)
                                  + sigma * (e45 - ey))
                        - e45 * (0.5f + equilibrium * equilibrium / 2.0f
                                 - equilibrium));
    }
    *entrainment /= normalization;
    *detrainment /= normalization;
}

__device__ __forceinline__ void kf_tpmix2(
        float pressure, float thetae,
        const float* __restrict__ temperature_table,
        const float* __restrict__ qsat_table,
        const float* __restrict__ thetae_base,
        float pressure_top, float pressure_reciprocal,
        float thetae_reciprocal, float* temperature, float* qv,
        float* liquid, float* ice, float* qnew_liquid, float* qnew_ice) {
    float qsat;
    kf_table_parcel(pressure, thetae, temperature_table, qsat_table,
                    thetae_base, pressure_top, pressure_reciprocal,
                    thetae_reciprocal, temperature, &qsat);
    float deficit = qsat - *qv;
    if (deficit <= 0.0f) {
        *qnew_liquid = *qv - qsat;
        *qv = qsat;
    } else {
        *qnew_liquid = 0.0f;
        float total = *liquid + *ice;
        if (total >= deficit) {
            *liquid -= deficit * *liquid / (total + 1.0e-10f);
            *ice -= deficit * *ice / (total + 1.0e-10f);
            *qv = qsat;
        } else {
            float latent = 3.15e6f - 2370.0f * *temperature;
            float cp = 1004.5f * (1.0f + 0.89f * *qv);
            if (total < 1.0e-10f) {
                *temperature += latent * (deficit / (1.0f + deficit)) / cp;
            } else {
                float remainder = deficit - total;
                *temperature += latent * (remainder / (1.0f + remainder)) / cp;
                *qv += total;
                *liquid = 0.0f;
                *ice = 0.0f;
            }
        }
    }
    *qnew_ice = 0.0f;
}

__device__ __forceinline__ void kf_dtfrznew(
        float pressure, float frozen, float* temperature, float* thetae,
        float* qv, float* ice) {
    float rlc = 2.5e6f - 2369.276f * (*temperature - 273.16f);
    float rls = 2833922.0f - 259.532f * (*temperature - 273.16f);
    float rlf = rls - rlc;
    float cp = 1004.5f * (1.0f + 0.89f * *qv);
    float a = (17.67f*273.15f - 17.67f*29.65f)
              / ((*temperature-29.65f)*(*temperature-29.65f));
    *temperature += rlf * frozen / (cp + rls * *qv * a);
    float es = 611.2f * expf((17.67f * *temperature - 17.67f * 273.15f)
                             / (*temperature - 29.65f));
    float qs = 0.622f * es / (pressure - es);
    float evaporated = qs - *qv;
    *ice -= evaporated;
    *qv += evaporated;
    float pii = powf(1.0e5f / pressure, 0.2854f * (1.0f - 0.28f * *qv));
    *thetae = *temperature * pii
        * expf((3374.6525f / *temperature - 2.5403f)
               * *qv * (1.0f + 0.81f * *qv));
}

__device__ __forceinline__ void kf_condload(
        float layer_depth, float buoyancy_term, float entrainment_term,
        float* liquid, float* ice, float* w2,
        float* qnew_liquid, float* qnew_ice,
        float* liquid_out, float* ice_out) {
    float total = *liquid + *ice;
    float fresh = *qnew_liquid + *qnew_ice;
    float estimated = 0.5f * (total + fresh);
    float g1 = fmaxf(*w2 + buoyancy_term - entrainment_term
                     - 2.0f*9.81f*layer_depth*estimated/1.5f, 0.0f);
    float wavg = 0.5f * (sqrtf(*w2) + sqrtf(g1));
    float conversion = 0.03f * layer_depth / wavg;
    float fresh_liquid_ratio = *qnew_liquid / (fresh + 1.0e-8f);
    total += 0.6f * fresh;
    float old_total = total;
    float liquid_ratio = (0.6f * *qnew_liquid + *liquid)
                         / (total + 1.0e-8f);
    total *= expf(-conversion);
    float fallout = old_total - total;
    *liquid_out = liquid_ratio * fallout;
    *ice_out = (1.0f - liquid_ratio) * fallout;
    float drag = 0.5f * (old_total + total - 0.2f * fresh);
    *w2 += buoyancy_term - entrainment_term
           - 2.0f*9.81f*layer_depth*drag/1.5f;
    if (fabsf(*w2) < 1.0e-4f) *w2 = 1.0e-4f;
    *liquid = liquid_ratio*total + fresh_liquid_ratio*0.4f*fresh;
    *ice = (1.0f-liquid_ratio)*total
           + (1.0f-fresh_liquid_ratio)*0.4f*fresh;
    *qnew_liquid = 0.0f;
    *qnew_ice = 0.0f;
}

__device__ __forceinline__ float kf_mixed_virtual_temperature(
        float pressure, float thetae, float qv, float liquid, float ice,
        const float* __restrict__ temperature_table,
        const float* __restrict__ qsat_table,
        const float* __restrict__ thetae_base,
        float pressure_top, float pressure_reciprocal,
        float thetae_reciprocal) {
    float temperature = 0.0f, qnew_liquid, qnew_ice;
    kf_tpmix2(pressure, thetae, temperature_table, qsat_table, thetae_base,
              pressure_top, pressure_reciprocal, thetae_reciprocal,
              &temperature, &qv, &liquid, &ice, &qnew_liquid, &qnew_ice);
    return temperature * (1.0f + 0.608f*qv - liquid - ice);
}

extern "C" __global__
void kf_column(
        const float* __restrict__ u,
        const float* __restrict__ v,
        const float* __restrict__ temperature,
        const float* __restrict__ qv,
        const float* __restrict__ qc,
        const float* __restrict__ pressure,
        const float* __restrict__ exner,
        const float* __restrict__ dz,
        const float* __restrict__ w,
        const float* __restrict__ temperature_table,
        const float* __restrict__ qsat_table,
        const float* __restrict__ thetae_base,
        const float* __restrict__ log_ratio,
        float* __restrict__ rthcuten,
        float* __restrict__ rqvcuten,
        float* __restrict__ rqccuten,
        float* __restrict__ rqicuten,
        float* __restrict__ rqrcuten,
        float* __restrict__ rqscuten,
        float* __restrict__ rainc,
        int* __restrict__ triggered,
        float* __restrict__ cape_before,
        float* __restrict__ cape_after,
        float* __restrict__ closure_time,
        float* __restrict__ nca_seconds,
        int* __restrict__ shallow_out,
        int* __restrict__ cloud_base,
        int* __restrict__ cloud_top_out,
        float* __restrict__ updraft_out,
        float* __restrict__ downdraft_out,
        float pressure_top, float pressure_reciprocal,
        float thetae_reciprocal, float dx, float dt, float cudt,
        int phase_mode,
        int nz, int ny, int nx) {
    int column = blockDim.x * blockIdx.x + threadIdx.x;
    int ncol = ny * nx;
    if (column >= ncol) return;
    if (nz < 8 || nz > KF_KMAX) return;

    float z[KF_KMAX], dp[KF_KMAX], qsat_env[KF_KMAX];
    float qenv[KF_KMAX], tv_env[KF_KMAX];
    float parcel_t[KF_KMAX], parcel_q[KF_KMAX];
    float thetaeu[KF_KMAX], qliq[KF_KMAX], qice[KF_KMAX];
    float qlqout[KF_KMAX], qicout[KF_KMAX], pptliq[KF_KMAX];
    float pptice[KF_KMAX], detlq[KF_KMAX], detice[KF_KMAX];
    float uer[KF_KMAX], udr[KF_KMAX], eqfrc[KF_KMAX];
    float dilfrc[KF_KMAX], qdt[KF_KMAX];
    float der[KF_KMAX], ddr[KF_KMAX], thetaed[KF_KMAX];
    float theta_ad[KF_KMAX], downdraft_q[KF_KMAX];
    float downdraft_t[KF_KMAX], qsd[KF_KMAX];
    float positive_energy[KF_KMAX], updraft[KF_KMAX], downdraft[KF_KMAX];
    float theta_env[KF_KMAX], theta_up[KF_KMAX], cell_mass[KF_KMAX];
    float unit_updraft[KF_KMAX], unit_downdraft[KF_KMAX];
    float unit_detlq[KF_KMAX], unit_detice[KF_KMAX];
    float unit_udr[KF_KMAX], unit_uer[KF_KMAX];
    float unit_der[KF_KMAX], unit_ddr[KF_KMAX];
    float tg[KF_KMAX], qg[KF_KMAX], theta_pa[KF_KMAX], qpa[KF_KMAX];
    float resolved_precip[KF_KMAX];
    float omega[KF_KMAX], fxm[KF_KMAX];
    float thfxin[KF_KMAX], thfxout[KF_KMAX];
    float qfxin[KF_KMAX], qfxout[KF_KMAX];
    float z_interface = 0.0f;
    for (int k = 0; k < nz; ++k) {
        int index = k * ncol + column;
        float depth = dz[index];
        z[k] = z_interface + 0.5f * depth;
        z_interface += depth;
        qsat_env[k] = kf_qsat(temperature[index], pressure[index]);
        qenv[k] = kf_clip(fminf(qv[index], qsat_env[k]), 1.0e-6f, 1.0f);
        tv_env[k] = temperature[index] * (1.0f + 0.608f * qenv[k]);
        float rho = pressure[index] / (287.0f * tv_env[k]);
        dp[k] = rho * 9.81f * depth;
        parcel_t[k] = parcel_q[k] = thetaeu[k] = 0.0f;
        qliq[k] = qice[k] = qlqout[k] = qicout[k] = 0.0f;
        pptliq[k] = pptice[k] = detlq[k] = detice[k] = 0.0f;
        uer[k] = udr[k] = qdt[k] = 0.0f;
        der[k] = ddr[k] = thetaed[k] = theta_ad[k] = 0.0f;
        downdraft_q[k] = downdraft_t[k] = qsd[k] = 0.0f;
        eqfrc[k] = dilfrc[k] = 1.0f;
        positive_energy[k] = updraft[k] = downdraft[k] = 0.0f;
        theta_env[k] = theta_up[k] = cell_mass[k] = 0.0f;
        tg[k] = temperature[index]; qg[k] = qenv[k];
        theta_pa[k] = qpa[k] = omega[k] = fxm[k] = 0.0f;
        resolved_precip[k] = 0.0f;
        rthcuten[index] = rqvcuten[index] = 0.0f;
        rqccuten[index] = rqicuten[index] = 0.0f;
        rqrcuten[index] = rqscuten[index] = 0.0f;
        updraft_out[index] = downdraft_out[index] = 0.0f;
    }
    rainc[column] = 0.0f;
    triggered[column] = 0;
    cape_before[column] = cape_after[column] = closure_time[column] = 0.0f;
    nca_seconds[column] = 0.0f;
    shallow_out[column] = 0;
    cloud_base[column] = cloud_top_out[column] = -1;
    int source_bottom = -1, source_top = -1, klcl = -1;
    int kbase = -1, let = -1, cloud_top = -1;
    float source_dp = 0.0f, tmix = 0.0f, qmix = 0.0f;
    float zmix = 0.0f, pmix = 0.0f, tlcl = 0.0f, zlcl = 0.0f;
    float dt_lcl = 0.0f, trigger_w = 0.0f, plcl = 0.0f;
    float trigger_environment_t = 0.0f, trigger_environment_tv = 0.0f;
    float thetae = 0.0f, radius = 0.0f, tv_lcl = 0.0f;
    float rho_lcl = 0.0f, base_mass_flux = 0.0f, w2 = 0.0f;
    float upold = 0.0f, upnew = 0.0f, cape = 0.0f, trppt = 0.0f;
    float cloud_depth = 0.0f, cloud_minimum = 0.0f;
    float surface_pressure = pressure[column];
    int candidates[KF_KMAX], ncandidates = 1;
    candidates[0] = 0;
    float threshold = surface_pressure - 1500.0f;
    for (int candidate = 1; candidate < nz; ++candidate) {
        int ci = candidate * ncol + column;
        if (pressure[ci] < surface_pressure - 30000.0f) break;
        if (pressure[ci] < threshold) {
            candidates[ncandidates++] = candidate;
            threshold -= 1500.0f;
        }
    }
    int candidate_cursor = 0, selected_candidate = -1;
    int shallow_candidate = -1;
    float shallow_max_depth = -1.0f;
    bool shallow = false;
    for (;;) {
        for (int k=0; k<nz; ++k) {
            parcel_t[k] = parcel_q[k] = thetaeu[k] = 0.0f;
            qliq[k] = qice[k] = qlqout[k] = qicout[k] = 0.0f;
            pptliq[k] = pptice[k] = detlq[k] = detice[k] = 0.0f;
            uer[k] = udr[k] = qdt[k] = 0.0f;
            eqfrc[k] = dilfrc[k] = 1.0f;
            positive_energy[k] = updraft[k] = 0.0f;
        }
        source_bottom = -1;
        while (source_bottom < 0) {
            if (shallow) {
                if (selected_candidate >= 0) return;
                selected_candidate = shallow_candidate;
            } else {
                if (candidate_cursor >= ncandidates) {
                    if (shallow_candidate < 0) return;
                    shallow = true;
                    selected_candidate = -1;
                    continue;
                }
                selected_candidate = candidate_cursor++;
            }
            int candidate = candidates[selected_candidate];
            int ci = candidate * ncol + column;
        float sum_dp = 0.0f;
        int top = candidate;
        while (top < nz && sum_dp <= 5000.0f) sum_dp += dp[top++];
        if (sum_dp <= 5000.0f || top >= nz) continue;
        float sum_t = 0.0f, sum_q = 0.0f, sum_z = 0.0f, sum_p = 0.0f;
        for (int k = candidate; k < top; ++k) {
            int index = k * ncol + column;
            sum_t += dp[k] * temperature[index];
            sum_q += dp[k] * qenv[k];
            sum_z += dp[k] * z[k];
            sum_p += dp[k] * pressure[index];
        }
        float tm = sum_t / sum_dp;
        float qm = sum_q / sum_dp;
        float zm = sum_z / sum_dp;
        float pm = sum_p / sum_dp;
        float emix = fmaxf(qm * pm / (0.622f + qm), 0.6112f);
        float a1 = fmaxf(emix / 611.2f, 0.001f);
        float position = (a1 - 0.001f) / 0.075f;
        int li = min(max((int)truncf(position), 0), 198);
        float base = 0.001f + 0.075f * li;
        float lf = kf_clip((a1 - base) / 0.075f, 0.0f, 1.0f);
        float tlog = (1.0f - lf) * log_ratio[li] + lf * log_ratio[li + 1];
        float dewpoint = (17.67f * 273.15f - 29.65f * tlog)
                         / (17.67f - tlog);
        float lcl_t = dewpoint
            - (0.212f + 1.571e-3f * (dewpoint - 273.16f)
               - 4.36e-4f * (tm - 273.16f)) * (tm - dewpoint);
        lcl_t = fminf(lcl_t, tm);
        float lcl_z = zm + (lcl_t - tm) / (-9.81f / 1004.5f);
        int lk = 0;
        while (lk < nz && z[lk] < lcl_z) ++lk;
        if (lk <= 0 || lk >= nz - 2) continue;
        float fz = kf_clip((lcl_z - z[lk - 1]) / (z[lk] - z[lk - 1]),
                           0.0f, 1.0f);
        float env_lcl = ((1.0f - fz) * temperature[(lk - 1) * ncol + column]
                         + fz * temperature[lk * ncol + column]);
        float qenv_lcl = ((1.0f - fz) * qenv[lk - 1]
                          + fz * qenv[lk]);
        float wlcl = ((1.0f - fz) * w[(lk - 1) * ncol + column]
                      + fz * w[lk * ncol + column]);
        float w_threshold = 0.02f * fminf(lcl_z / 2000.0f, 1.0f);
        float w_scaled = wlcl * dx / 25000.0f - w_threshold;
        float perturbation = (w_scaled < 1.0e-4f)
            ? 0.0f : 4.64f * powf(w_scaled, 0.33f);
        if (lcl_t + perturbation < env_lcl) continue;
        source_bottom = candidate;
        source_top = top;
        source_dp = sum_dp;
        tmix = tm; qmix = qm; zmix = zm; pmix = pm;
        tlcl = lcl_t; zlcl = lcl_z; klcl = lk; dt_lcl = perturbation;
        trigger_w = w_scaled;
        trigger_environment_t = env_lcl;
        trigger_environment_tv = env_lcl * (1.0f + 0.608f * qenv_lcl);
        plcl = ((1.0f - fz) * pressure[(lk - 1) * ncol + column]
                + fz * pressure[lk * ncol + column]);
        break;
        }

    thetae = kf_thetae(pmix, tmix, qmix, log_ratio);
    radius = trigger_w < 0.0f ? 1000.0f
        : (trigger_w > 0.1f ? 2000.0f : 1000.0f + 1000.0f*trigger_w/0.1f);
    tv_lcl = tlcl * (1.0f + 0.608f * qmix);
    rho_lcl = plcl / (287.0f * tv_lcl);
    base_mass_flux = rho_lcl * 0.01f * dx * dx;
    kbase = klcl - 1;
    updraft[kbase] = base_mass_flux;
    float initial_w = dt_lcl <= 1.0e-4f ? 1.0f
        : fminf(1.0f + 0.5f*sqrtf(2.0f*9.81f*dt_lcl*500.0f
                                  / trigger_environment_tv), 3.0f);
    w2 = initial_w * initial_w;
    thetaeu[kbase] = thetae;
    parcel_t[kbase] = tlcl;
    parcel_q[kbase] = qmix;
    float ee1 = 1.0f, ud1 = 0.0f, rei = 0.0f;
    upold = base_mass_flux; upnew = upold;
    cape = 0.0f; trppt = 0.0f;
    float ttemp = 268.16f;
    let = klcl; cloud_top = kbase;
    for (int nk = kbase; nk < nz - 1; ++nk) {
        int nk1 = nk + 1;
        int index = nk1*ncol + column;
        parcel_t[nk1] = temperature[index];
        thetaeu[nk1] = thetaeu[nk];
        parcel_q[nk1] = parcel_q[nk];
        qliq[nk1] = qliq[nk];
        qice[nk1] = qice[nk];
        float qnewlq, qnewice;
        kf_tpmix2(pressure[index], thetaeu[nk1], temperature_table,
                  qsat_table, thetae_base, pressure_top,
                  pressure_reciprocal, thetae_reciprocal, &parcel_t[nk1],
                  &parcel_q[nk1], &qliq[nk1], &qice[nk1],
                  &qnewlq, &qnewice);
        if (parcel_t[nk1] <= 268.16f) {
            float frc1;
            if (parcel_t[nk1] > 248.16f) {
                if (ttemp > 268.16f) ttemp = 268.16f;
                frc1 = (ttemp-parcel_t[nk1])/(ttemp-248.16f);
            } else {
                frc1 = 1.0f;
            }
            ttemp = parcel_t[nk1];
            float frozen = (qliq[nk1]+qnewlq)*frc1;
            qnewice += qnewlq*frc1;
            qnewlq -= qnewlq*frc1;
            qice[nk1] += qliq[nk1]*frc1;
            qliq[nk1] -= qliq[nk1]*frc1;
            kf_dtfrznew(pressure[index], frozen, &parcel_t[nk1],
                        &thetaeu[nk1], &parcel_q[nk1], &qice[nk1]);
        }
        float tvu = parcel_t[nk1]*(1.0f+0.608f*parcel_q[nk1]);
        float be, layer_depth;
        if (nk == kbase) {
            be = (tv_lcl+tvu)/(trigger_environment_tv+tv_env[nk1])-1.0f;
            layer_depth = z[nk1]-zlcl;
        } else {
            float tvu_below = parcel_t[nk]*(1.0f+0.608f*parcel_q[nk]);
            be = (tvu_below+tvu)/(tv_env[nk]+tv_env[nk1])-1.0f;
            layer_depth = z[nk1]-z[nk];
        }
        float boterm = 2.0f*layer_depth*9.81f*be/1.5f;
        float enterm = 2.0f*rei*w2/upold;
        kf_condload(layer_depth, boterm, enterm, &qliq[nk1], &qice[nk1],
                    &w2, &qnewlq, &qnewice, &qlqout[nk1], &qicout[nk1]);
        cloud_top = nk;
        if (w2 < 1.0e-3f) break;
        float environment_thetae = kf_thetae(
            pressure[index], temperature[index], qenv[nk1], log_ratio);
        rei = base_mass_flux*dp[nk1]*0.03f/radius;
        float tvqu = parcel_t[nk1]
            *(1.0f+0.608f*parcel_q[nk1]-qliq[nk1]-qice[nk1]);
        float dilbe;
        if (nk == kbase) {
            dilbe = ((tv_lcl+tvqu)/(trigger_environment_tv+tv_env[nk1])-1.0f)
                    *layer_depth;
        } else {
            float tvqu_below = parcel_t[nk]
                *(1.0f+0.608f*parcel_q[nk]-qliq[nk]-qice[nk]);
            dilbe = ((tvqu_below+tvqu)/(tv_env[nk]+tv_env[nk1])-1.0f)
                    *layer_depth;
        }
        if (dilbe > 0.0f) {
            positive_energy[nk1] = dilbe*9.81f;
            cape += positive_energy[nk1];
        }
        float ee2, ud2;
        if (tvqu <= tv_env[nk1]) {
            ee2 = 0.5f; ud2 = 1.0f; eqfrc[nk1] = 0.0f;
        } else {
            let = nk1;
            float f1 = 0.95f, f2 = 0.05f;
            float mixed_virtual = kf_mixed_virtual_temperature(
                pressure[index], f1*environment_thetae+f2*thetaeu[nk1],
                f1*qenv[nk1]+f2*parcel_q[nk1], f2*qliq[nk1], f2*qice[nk1],
                temperature_table, qsat_table, thetae_base, pressure_top,
                pressure_reciprocal, thetae_reciprocal);
            if (mixed_virtual > tv_env[nk1]) {
                ee2 = 1.0f; ud2 = 0.0f; eqfrc[nk1] = 1.0f;
            } else {
                f1 = 0.10f; f2 = 0.90f;
                mixed_virtual = kf_mixed_virtual_temperature(
                    pressure[index], f1*environment_thetae+f2*thetaeu[nk1],
                    f1*qenv[nk1]+f2*parcel_q[nk1], f2*qliq[nk1],
                    f2*qice[nk1], temperature_table, qsat_table, thetae_base,
                    pressure_top, pressure_reciprocal, thetae_reciprocal);
                if (fabsf(mixed_virtual-tvqu) < 1.0e-3f) {
                    ee2 = 1.0f; ud2 = 0.0f; eqfrc[nk1] = 1.0f;
                } else {
                    eqfrc[nk1] = kf_clip((tv_env[nk1]-tvqu)*f1
                                         /(mixed_virtual-tvqu), 0.0f, 1.0f);
                    if (eqfrc[nk1] == 1.0f) {
                        ee2 = 1.0f; ud2 = 0.0f;
                    } else if (eqfrc[nk1] == 0.0f) {
                        ee2 = 0.0f; ud2 = 1.0f;
                    } else {
                        kf_prof5(eqfrc[nk1], &ee2, &ud2);
                    }
                }
            }
        }
        ee2 = fmaxf(ee2, 0.5f);
        ud2 *= 1.5f;
        uer[nk1] = 0.5f*rei*(ee1+ee2);
        udr[nk1] = 0.5f*rei*(ud1+ud2);
        if (updraft[nk]-udr[nk1] < 10.0f) {
            if (dilbe > 0.0f) {
                cape -= dilbe*9.81f;
                positive_energy[nk1] = 0.0f;
            }
            let = nk;
            break;
        }
        ee1 = ee2; ud1 = ud2;
        upold = updraft[nk]-udr[nk1];
        upnew = upold+uer[nk1];
        updraft[nk1] = upnew;
        dilfrc[nk1] = upnew/upold;
        detlq[nk1] = qliq[nk1]*udr[nk1];
        detice[nk1] = qice[nk1]*udr[nk1];
        qdt[nk1] = parcel_q[nk1];
        parcel_q[nk1] = (upold*parcel_q[nk1]+uer[nk1]*qenv[nk1])/upnew;
        thetaeu[nk1] = (thetaeu[nk1]*upold+environment_thetae*uer[nk1])/upnew;
        qliq[nk1] *= upold/upnew;
        qice[nk1] *= upold/upnew;
        pptliq[nk1] = qlqout[nk1]*updraft[nk];
        pptice[nk1] = qicout[nk1]*updraft[nk];
        trppt += pptliq[nk1]+pptice[nk1];
        if (nk1 <= source_top-1)
            uer[nk1] += base_mass_flux*dp[nk1]/source_dp;
    }
    cloud_depth = z[cloud_top] - zlcl;
    cloud_minimum = tlcl > 293.0f ? 4000.0f
        : 2000.0f + 100.0f * kf_clip(tlcl - 273.0f, 0.0f, 20.0f);
    int kpbl = source_top - 1;
    if (cloud_top <= klcl || cloud_top <= kpbl || let + 1 <= kpbl) {
        if (shallow) return;
        continue;
    }
    if (!shallow && (cape <= 1.0f || cloud_depth <= cloud_minimum)) {
        if (cloud_depth > shallow_max_depth) {
            shallow_max_depth = cloud_depth;
            shallow_candidate = selected_candidate;
        }
        continue;
    }
    if (shallow) let = max(kpbl, klcl);
    break;
    }

    if (let == cloud_top) {
        udr[cloud_top] = updraft[cloud_top]+udr[cloud_top]-uer[cloud_top];
        detlq[cloud_top] = qliq[cloud_top]*udr[cloud_top]*upnew/upold;
        detice[cloud_top] = qice[cloud_top]*udr[cloud_top]*upnew/upold;
        uer[cloud_top] = 0.0f;
        updraft[cloud_top] = 0.0f;
    } else {
        float top_dp = 0.0f;
        for (int nk=let+1; nk<=cloud_top; ++nk) top_dp += dp[nk];
        float dumfdp = updraft[let]/top_dp;
        for (int nk=let+1; nk<=cloud_top; ++nk) {
            if (nk == cloud_top) {
                udr[nk] = updraft[nk-1]; uer[nk] = 0.0f;
                detlq[nk] = udr[nk]*qliq[nk]*dilfrc[nk];
                detice[nk] = udr[nk]*qice[nk]*dilfrc[nk];
            } else {
                updraft[nk] = updraft[nk-1]-dp[nk]*dumfdp;
                uer[nk] = updraft[nk]*(1.0f-1.0f/dilfrc[nk]);
                udr[nk] = updraft[nk-1]-updraft[nk]+uer[nk];
                detlq[nk] = udr[nk]*qliq[nk]*dilfrc[nk];
                detice[nk] = udr[nk]*qice[nk]*dilfrc[nk];
            }
            if (nk >= let+2) {
                trppt -= pptliq[nk]+pptice[nk];
                pptliq[nk] = updraft[nk-1]*qlqout[nk];
                pptice[nk] = updraft[nk-1]*qicout[nk];
                trppt += pptliq[nk]+pptice[nk];
            }
        }
    }
    for (int nk=0; nk<=kbase; ++nk) {
        if (nk >= source_bottom) {
            if (nk == source_bottom) {
                updraft[nk] = base_mass_flux*dp[nk]/source_dp;
                uer[nk] = updraft[nk];
            } else if (nk <= source_top-1) {
                uer[nk] = base_mass_flux*dp[nk]/source_dp;
                updraft[nk] = updraft[nk-1]+uer[nk];
            } else {
                updraft[nk] = base_mass_flux; uer[nk] = 0.0f;
            }
            parcel_t[nk] = tmix+(z[nk]-zmix)*(-9.81f/1004.5f);
            parcel_q[nk] = qmix;
        }
        udr[nk] = qdt[nk] = 0.0f;
        qliq[nk] = qice[nk] = qlqout[nk] = qicout[nk] = 0.0f;
        pptliq[nk] = pptice[nk] = detlq[nk] = detice[nk] = 0.0f;
        eqfrc[nk] = 1.0f;
    }
    for (int nk=0; nk<=cloud_top; ++nk) {
        int index = nk*ncol+column;
        if (thetaeu[nk] == 0.0f)
            thetaeu[nk] = kf_thetae(pressure[index], temperature[index],
                                     qenv[nk], log_ratio);
        theta_up[nk] = parcel_t[nk]
            *powf(1.0e5f/pressure[index], 0.2854f*(1.0f-0.28f*qdt[nk]));
        theta_env[nk] = temperature[index]
            *powf(1.0e5f/pressure[index], 0.2854f*(1.0f-0.28f*qenv[nk]));
    }

    int l5 = 0;
    for (int k = 0; k < nz; ++k)
        if (pressure[k * ncol + column] >= 0.5f * surface_pressure) l5 = k;
    float wind_mid = hypotf(u[l5 * ncol + column], v[l5 * ncol + column]);
    float wind_lcl = hypotf(u[klcl * ncol + column], v[klcl * ncol + column]);
    float velocity = 0.5f * (wind_lcl + wind_mid);
    float advective_time = dx / velocity;
    float timec = shallow ? 2400.0f
                          : kf_clip(advective_time, 1800.0f, 3600.0f);
    timec = fmaxf(floorf(timec / dt + 0.5f), 1.0f) * dt;

    float wind_top = hypotf(u[cloud_top * ncol + column],
                            v[cloud_top * ncol + column]);
    float shear_sign = wind_top > wind_lcl ? 1.0f : -1.0f;
    float shear = (1000.0f * shear_sign
                   * hypotf(u[cloud_top * ncol + column]
                                - u[klcl * ncol + column],
                            v[cloud_top * ncol + column]
                                - v[klcl * ncol + column])
                   / (z[cloud_top] - z[klcl]));
    float shear_efficiency = kf_clip(
        1.591f + shear * (-0.639f + shear * (9.53e-2f
                                             - shear * 4.96e-3f)),
        0.2f, 0.9f);
    float cloud_base_kft = fmaxf((zlcl - z[0]) * 3.281e-3f, 0.0f);
    float cloud_base_response;
    if (cloud_base_kft < 3.0f) {
        cloud_base_response = 0.02f;
    } else {
        cloud_base_response = 0.96729352f + cloud_base_kft
            * (-0.70034167f + cloud_base_kft
               * (0.162179896f + cloud_base_kft
                  * (-1.2569798e-2f + cloud_base_kft
                     * (4.2772e-4f - cloud_base_kft * 5.44e-6f))));
    }
    if (cloud_base_kft > 25.0f) cloud_base_response = 2.4f;
    float cloud_base_efficiency = fminf(
        1.0f / (1.0f + cloud_base_response), 0.9f);
    float efficiency = 0.5f * (shear_efficiency + cloud_base_efficiency);

    // WRF module_cu_kfeta.F:1642-1873 downdraft.
    float tder = 0.0f;
    float pptflx = trppt;
    int lfs = shallow ? 0 : max(let-1, 0), ldb = 0;
    int kstart = source_top;
    if (!shallow && kstart < nz-1) {
        for (int nk=kstart+1; nk<nz; ++nk) {
            if (pressure[kstart*ncol+column]-pressure[nk*ncol+column]
                    > 15000.0f) {
                lfs = nk;
                break;
            }
        }
        lfs = min(lfs, let-1);
        if (lfs > kstart
                && pressure[kstart*ncol+column]-pressure[lfs*ncol+column]
                   > 5000.0f) {
            int ilfs = lfs*ncol+column;
            thetaed[lfs] = kf_thetae(pressure[ilfs], temperature[ilfs],
                                      qenv[lfs], log_ratio);
            downdraft_q[lfs] = qenv[lfs];
            float qss;
            kf_table_parcel(pressure[ilfs], thetaed[lfs], temperature_table,
                            qsat_table, thetae_base, pressure_top,
                            pressure_reciprocal, thetae_reciprocal,
                            &downdraft_t[lfs], &qss);
            theta_ad[lfs] = downdraft_t[lfs]
                * powf(1.0e5f/pressure[ilfs], 0.2854f*(1.0f-0.28f*qss));
            float rdd = pressure[ilfs]
                /(287.0f*downdraft_t[lfs]*(1.0f+0.608f*qss));
            downdraft[lfs] = -(1.0f-efficiency)*0.01f*dx*dx*rdd;
            der[lfs] = downdraft[lfs];
            float rhnum = qenv[lfs]/qsat_env[lfs]*dp[lfs];
            float rhden = dp[lfs];
            for (int nd=lfs-1; nd>=kstart; --nd) {
                int index = nd*ncol+column;
                der[nd] = der[lfs]*dp[nd]/dp[lfs];
                downdraft[nd] = downdraft[nd+1]+der[nd];
                float env_thetae = kf_thetae(
                    pressure[index], temperature[index], qenv[nd], log_ratio);
                thetaed[nd] = (thetaed[nd+1]*downdraft[nd+1]
                               + env_thetae*der[nd])/downdraft[nd];
                downdraft_q[nd] = (downdraft_q[nd+1]*downdraft[nd+1]
                                   + qenv[nd]*der[nd])/downdraft[nd];
                rhden += dp[nd];
                rhnum += qenv[nd]/qsat_env[nd]*dp[nd];
            }
            float dmffrc = 2.0f*(1.0f-rhnum/rhden);
            float melting_precip = 0.0f;
            for (int nk=klcl; nk<=cloud_top; ++nk)
                melting_precip += pptice[nk];
            int ml = 0;
            for (int nk=0; nk<=cloud_top; ++nk)
                if (temperature[nk*ncol+column] > 273.16f) ml = nk;
            float dtmelt = (source_bottom < ml && updraft[klcl] != 0.0f)
                ? 3.339e5f*melting_precip/(1004.5f*updraft[klcl]) : 0.0f;
            int ldt = min(lfs-1, kstart-1);
            int iks = kstart*ncol+column;
            kf_table_parcel(pressure[iks], thetaed[kstart], temperature_table,
                            qsat_table, thetae_base, pressure_top,
                            pressure_reciprocal, thetae_reciprocal,
                            &downdraft_t[kstart], &qss);
            downdraft_t[kstart] -= dtmelt;
            float es = 611.2f*expf((17.67f*downdraft_t[kstart]
                                    -17.67f*273.15f)
                                   /(downdraft_t[kstart]-29.65f));
            qss = 0.622f*es/(pressure[iks]-es);
            thetaed[kstart] = downdraft_t[kstart]
                * powf(1.0e5f/pressure[iks], 0.2854f*(1.0f-0.28f*qss))
                * expf((3374.6525f/downdraft_t[kstart]-2.5403f)
                       *qss*(1.0f+0.81f*qss));
            float dpdd = 0.0f;
            for (int nd=ldt; nd>=0; --nd) {
                int index = nd*ncol+column;
                dpdd += dp[nd];
                thetaed[nd] = thetaed[kstart];
                downdraft_q[nd] = downdraft_q[kstart];
                kf_table_parcel(pressure[index], thetaed[nd], temperature_table,
                                qsat_table, thetae_base, pressure_top,
                                pressure_reciprocal, thetae_reciprocal,
                                &downdraft_t[nd], &qss);
                qsd[nd] = qss;
                float rhh = 1.0f-0.2e-3f*(z[kstart]-z[nd]);
                if (rhh < 1.0f) {
                    float dssdt = (17.67f*273.15f-17.67f*29.65f)
                        /((downdraft_t[nd]-29.65f)*(downdraft_t[nd]-29.65f));
                    float latent = 3.15e6f-2370.0f*downdraft_t[nd];
                    float dtmp = latent*qss*(1.0f-rhh)
                        /(1004.5f+latent*rhh*qss*dssdt);
                    float t1rh = downdraft_t[nd]+dtmp;
                    es = rhh*611.2f*expf((17.67f*t1rh-17.67f*273.15f)
                                         /(t1rh-29.65f));
                    float qsrh = 0.622f*es/(pressure[index]-es);
                    if (qsrh < downdraft_q[nd]) {
                        qsrh = downdraft_q[nd];
                        t1rh = downdraft_t[nd]+(qss-qsrh)*latent/1004.5f;
                    }
                    downdraft_t[nd] = t1rh;
                    qss = qsrh;
                    qsd[nd] = qss;
                }
                float tvd = downdraft_t[nd]*(1.0f+0.608f*qsd[nd]);
                if (tvd > tv_env[nd] || nd == 0) {
                    ldb = nd;
                    break;
                }
            }
            if (pressure[ldb*ncol+column]-pressure[lfs*ncol+column]
                    > 5000.0f) {
                for (int nd=ldt; nd>=ldb; --nd) {
                    int index = nd*ncol+column;
                    ddr[nd] = -downdraft[kstart]*dp[nd]/dpdd;
                    der[nd] = 0.0f;
                    downdraft[nd] = downdraft[nd+1]+ddr[nd];
                    tder += (qsd[nd]-downdraft_q[nd])*ddr[nd];
                    downdraft_q[nd] = qsd[nd];
                    theta_ad[nd] = downdraft_t[nd]
                        *powf(1.0e5f/pressure[index],
                              0.2854f*(1.0f-0.28f*downdraft_q[nd]));
                }
            }
            if (tder >= 1.0f) {
                float ddinc = -dmffrc*updraft[klcl]/downdraft[kstart];
                if (tder*ddinc > trppt) ddinc = trppt/tder;
                tder *= ddinc;
                pptflx = trppt-tder;
                for (int nk=ldb; nk<=lfs; ++nk) {
                    downdraft[nk] *= ddinc;
                    der[nk] *= ddinc;
                    ddr[nk] *= ddinc;
                }
                efficiency = (trppt-tder)/trppt;
                for (int nk=0; nk<ldb; ++nk) {
                    downdraft[nk] = der[nk] = ddr[nk] = 0.0f;
                    theta_ad[nk] = downdraft_q[nk] = downdraft_t[nk] = 0.0f;
                }
                for (int nk=lfs+1; nk<nz; ++nk) {
                    downdraft[nk] = der[nk] = ddr[nk] = 0.0f;
                    theta_ad[nk] = downdraft_q[nk] = downdraft_t[nk] = 0.0f;
                }
                for (int nk=ldt+1; nk<lfs; ++nk)
                    theta_ad[nk] = downdraft_q[nk] = downdraft_t[nk] = 0.0f;
            } else {
                // WRF 1789-1794 suppresses the entire downdraft branch.
                pptflx = trppt;
                tder = 0.0f;
                for (int nk=0; nk<nz; ++nk) {
                    downdraft[nk] = der[nk] = ddr[nk] = 0.0f;
                    theta_ad[nk] = downdraft_q[nk] = downdraft_t[nk] = 0.0f;
                }
            }
        }
    }

    // WRF module_cu_kfeta.F:1875-2281 stabilization closure.
    float dxsq = dx*dx;
    float aincmx = 1000.0f;
    int lmax = max(klcl, lfs);
    for (int nk=0; nk<nz; ++nk) cell_mass[nk] = dp[nk]*dxsq/9.81f;
    for (int nk=source_bottom; nk<=lmax; ++nk) {
        float draft_inflow = uer[nk]-der[nk];
        if (draft_inflow > 1.0e-3f)
            aincmx = fminf(aincmx, cell_mass[nk]/(draft_inflow*timec));
    }
    float ainc = fminf(1.0f, aincmx);
    float unit_tder = tder;
    float unit_pptflx = pptflx;
    for (int nk=0; nk<nz; ++nk) {
        unit_updraft[nk] = updraft[nk];
        unit_downdraft[nk] = downdraft[nk];
        unit_detlq[nk] = detlq[nk];
        unit_detice[nk] = detice[nk];
        unit_udr[nk] = udr[nk];
        unit_uer[nk] = uer[nk];
        unit_der[nk] = der[nk];
        unit_ddr[nk] = ddr[nk];
    }
    if (shallow) {
        // WRF 1911-1948: TKEMAX=5 gives EVAC=0.25.
        ainc = 0.25f*source_dp*dxsq/(base_mass_flux*9.81f*timec);
        tder = unit_tder*ainc;
        pptflx = unit_pptflx*ainc;
        for (int nk=0; nk<nz; ++nk) {
            updraft[nk] = unit_updraft[nk]*ainc;
            downdraft[nk] = unit_downdraft[nk]*ainc;
            detlq[nk] = unit_detlq[nk]*ainc;
            detice[nk] = unit_detice[nk]*ainc;
            udr[nk] = unit_udr[nk]*ainc;
            uer[nk] = unit_uer[nk]*ainc;
            der[nk] = unit_der[nk]*ainc;
            ddr[nk] = unit_ddr[nk]*ainc;
        }
    }
    float fabe = 1.0f, fabeold = 1.0f, aincold = 0.0f;
    float adjusted_cape = cape;
    bool noitr = false;
    int nstep = 1;
    float dtime = timec;
    for (int ncount=0; ncount<10; ++ncount) {
        float dtt = timec;
        omega[0] = 0.0f;
        for (int nk=0; nk<=cloud_top; ++nk) {
            if (nk > 0) {
                float previous_domgdp = -(uer[nk-1]-der[nk-1]
                    -udr[nk-1]-ddr[nk-1])/cell_mass[nk-1];
                omega[nk] = omega[nk-1]-dp[nk-1]*previous_domgdp;
                float absolute_omega = fabsf(omega[nk]);
                if (absolute_omega*timec > 0.75f*dp[nk-1])
                    dtt = fminf(dtt, 0.75f*dp[nk-1]/absolute_omega);
            }
        }
        nstep = max((int)floorf(timec/dtt+1.5f), 1);
        dtime = timec/(float)nstep;
        for (int nk=0; nk<nz; ++nk) {
            theta_pa[nk] = theta_env[nk];
            qpa[nk] = qenv[nk];
            fxm[nk] = omega[nk]*dxsq/9.81f;
        }
        for (int ntc=0; ntc<nstep; ++ntc) {
            for (int nk=0; nk<=cloud_top; ++nk)
                thfxin[nk] = thfxout[nk] = qfxin[nk] = qfxout[nk] = 0.0f;
            for (int nk=1; nk<=cloud_top; ++nk) {
                if (omega[nk] <= 0.0f) {
                    thfxin[nk] = -fxm[nk]*theta_pa[nk-1];
                    qfxin[nk] = -fxm[nk]*qpa[nk-1];
                    thfxout[nk-1] += thfxin[nk];
                    qfxout[nk-1] += qfxin[nk];
                } else {
                    thfxout[nk] = fxm[nk]*theta_pa[nk];
                    qfxout[nk] = fxm[nk]*qpa[nk];
                    thfxin[nk-1] += thfxout[nk];
                    qfxin[nk-1] += qfxout[nk];
                }
            }
            for (int nk=0; nk<=cloud_top; ++nk) {
                theta_pa[nk] += (thfxin[nk]+udr[nk]*theta_up[nk]
                    +ddr[nk]*theta_ad[nk]-thfxout[nk]
                    -(uer[nk]-der[nk])*theta_env[nk])
                    *dtime/cell_mass[nk];
                qpa[nk] += (qfxin[nk]+udr[nk]*qdt[nk]
                    +ddr[nk]*downdraft_q[nk]-qfxout[nk]
                    -(uer[nk]-der[nk])*qenv[nk])
                    *dtime/cell_mass[nk];
            }
        }
        for (int nk=0; nk<=cloud_top; ++nk) {
            if (qpa[nk] >= 0.0f) continue;
            if (nk == 0) return;
            int neighbor = nk == cloud_top ? klcl : nk+1;
            float tma = qpa[neighbor]*cell_mass[neighbor];
            float tmb = qpa[nk-1]*cell_mass[nk-1];
            float tmm = (qpa[nk]-1.0e-9f)*cell_mass[nk];
            if (tma == 0.0f || tmb == 0.0f) return;
            float bcoeff = -tmm/(tma*tma/tmb+tmb);
            float acoeff = bcoeff*tma/tmb;
            tmb *= 1.0f-bcoeff;
            tma *= 1.0f-acoeff;
            qpa[nk] = 1.0e-9f;
            qpa[neighbor] = tma/cell_mass[neighbor];
            qpa[nk-1] = tmb/cell_mass[nk-1];
        }
        float top_omega = (udr[cloud_top]-uer[cloud_top])*dp[cloud_top]
                          /cell_mass[cloud_top];
        if (fabsf(top_omega-omega[cloud_top]) > 1.0e-3f) return;
        for (int nk=0; nk<=cloud_top; ++nk) {
            int index = nk*ncol+column;
            qg[nk] = qpa[nk];
            float moist_exponent = 0.2854f*(1.0f-0.28f*qg[nk]);
            tg[nk] = theta_pa[nk]
                /powf(1.0e5f/pressure[index], moist_exponent);
        }
        if (shallow) break;

        float tmix_g = 0.0f, qmix_g = 0.0f;
        for (int nk=source_bottom; nk<source_top; ++nk) {
            tmix_g += dp[nk]*tg[nk];
            qmix_g += dp[nk]*qg[nk];
        }
        tmix_g /= source_dp;
        qmix_g /= source_dp;
        float qss = kf_qsat(tmix_g, pmix);
        float tlcl_g;
        if (qmix_g > qss) {
            float latent = 3.15e6f-2370.0f*tmix_g;
            float cpm = 1004.5f*(1.0f+0.887f*qmix_g);
            float dssdt = qss*(17.67f*273.15f-17.67f*29.65f)
                           /((tmix_g-29.65f)*(tmix_g-29.65f));
            float dq = (qmix_g-qss)/(1.0f+latent*dssdt/cpm);
            tmix_g += latent/1004.5f*dq;
            qmix_g -= dq;
            tlcl_g = tmix_g;
        } else {
            qmix_g = fmaxf(qmix_g, 0.0f);
            float emix = qmix_g*pmix/(0.622f+qmix_g);
            float a1 = emix/611.2f;
            float position = (a1-0.001f)/0.075f;
            int li = (int)position;
            float value = li*0.075f+0.001f;
            float fraction = (a1-value)/0.075f;
            float tlog = fraction*log_ratio[li+1]
                         +(1.0f-fraction)*log_ratio[li];
            float dewpoint = (17.67f*273.15f-29.65f*tlog)/(17.67f-tlog);
            tlcl_g = dewpoint-(0.212f+1.571e-3f*(dewpoint-273.16f)
                -4.36e-4f*(tmix_g-273.16f))*(tmix_g-dewpoint);
            tlcl_g = fminf(tlcl_g, tmix_g);
        }
        float tvlcl_g = tlcl_g*(1.0f+0.608f*qmix_g);
        float zlcl_g = zmix+(tlcl_g-tmix_g)/(-9.81f/1004.5f);
        int klcl_g = 0;
        while (klcl_g<nz && z[klcl_g]<zlcl_g) ++klcl_g;
        if (klcl_g <= 0 || klcl_g > cloud_top) return;
        int kbase_g = klcl_g-1;
        float lcl_fraction = (zlcl_g-z[kbase_g])/(z[klcl_g]-z[kbase_g]);
        float tenv_g = tg[kbase_g]+(tg[klcl_g]-tg[kbase_g])*lcl_fraction;
        float qenv_g = qg[kbase_g]+(qg[klcl_g]-qg[kbase_g])*lcl_fraction;
        float tven_g = tenv_g*(1.0f+0.608f*qenv_g);
        float thetae_parcel = tmix_g
            *powf(1.0e5f/pmix, 0.2854f*(1.0f-0.28f*qmix_g))
            *expf((3374.6525f/tlcl_g-2.5403f)*qmix_g*(1.0f+0.81f*qmix_g));
        adjusted_cape = 0.0f;
        float tvqu_below = 0.0f;
        for (int nk=kbase_g; nk<cloud_top; ++nk) {
            int nk1 = nk+1;
            int index = nk1*ncol+column;
            float tgu, qgu;
            kf_table_parcel(pressure[index], thetae_parcel, temperature_table,
                            qsat_table, thetae_base, pressure_top,
                            pressure_reciprocal, thetae_reciprocal, &tgu, &qgu);
            float tvqu = tgu*(1.0f+0.608f*qgu-qliq[nk1]-qice[nk1]);
            float dilbe;
            if (nk == kbase_g) {
                float depth = z[klcl_g]-zlcl_g;
                dilbe = ((tvlcl_g+tvqu)/(tven_g
                    +tg[nk1]*(1.0f+0.608f*qg[nk1]))-1.0f)*depth;
            } else {
                float depth = z[nk1]-z[nk];
                dilbe = ((tvqu_below+tvqu)
                    /(tg[nk]*(1.0f+0.608f*qg[nk])
                      +tg[nk1]*(1.0f+0.608f*qg[nk1]))-1.0f)*depth;
            }
            if (dilbe > 0.0f) adjusted_cape += dilbe*9.81f;
            float environment_thetae = kf_thetae(
                pressure[index], tg[nk1], qg[nk1], log_ratio);
            thetae_parcel = thetae_parcel/dilfrc[nk1]
                +environment_thetae*(1.0f-1.0f/dilfrc[nk1]);
            tvqu_below = tvqu;
        }
        if (noitr) break;
        float dabe = fmaxf(cape-adjusted_cape, 0.1f*cape);
        fabe = adjusted_cape/cape;
        if (fabe > 1.0f) return;
        if (ncount != 0) {
            if (fabsf(ainc-aincold) < 1.0e-4f) {
                noitr = true; ainc = aincold; continue;
            }
            float dfda = (fabe-fabeold)/(ainc-aincold);
            if (dfda > 0.0f) {
                noitr = true; ainc = aincold; continue;
            }
        }
        aincold = ainc;
        fabeold = fabe;
        if (ainc/aincmx > 0.999f && fabe > 0.10f) break;
        if ((fabe >= 0.0f && fabe <= 0.10f) || ncount == 9) break;
        if (fabe == 0.0f) {
            ainc *= 0.5f;
        } else {
            if (dabe < 1.0e-4f) {
                noitr = true; ainc = aincold; continue;
            }
            ainc *= 0.95f*cape/dabe;
        }
        ainc = fminf(aincmx, ainc);
        if (ainc < 0.05f) return;
        tder = unit_tder*ainc;
        pptflx = unit_pptflx*ainc;
        for (int nk=0; nk<nz; ++nk) {
            updraft[nk] = unit_updraft[nk]*ainc;
            downdraft[nk] = unit_downdraft[nk]*ainc;
            detlq[nk] = unit_detlq[nk]*ainc;
            detice[nk] = unit_detice[nk]*ainc;
            udr[nk] = unit_udr[nk]*ainc;
            uer[nk] = unit_uer[nk]*ainc;
            der[nk] = unit_der[nk]*ainc;
            ddr[nk] = unit_ddr[nk]*ainc;
        }
    }
    // PPTFLX is assigned only when the mass flux is rescaled (WRF 2262).
    // A NOITR AINC revert intentionally keeps the pre-revert flux.
    for (int nk=0; nk<=cloud_top; ++nk) {
        int index = nk*ncol+column;
        rqvcuten[index] = (qg[nk]-qenv[nk])/timec;
    }
    // WRF 2311-2382: preserve QL/QI/QR/QS independently.  These arrays are
    // dead after the closure calculation and are intentionally reused here
    // so the corrected mixed-phase interface does not grow per-thread stack:
    // parcel_t=QLPA, parcel_q=QIPA, resolved_precip=QRPA, thetaeu=QSPA;
    // TH/Q flux arrays carry QL/QI, QLQOUT/QICOUT carry QR, and QLIQ/QICE
    // carry QS.  Shallow FBFRC=1 returns liquid and frozen fallout to the
    // resolved rain and snow categories separately.
    float fbfrc = shallow ? 1.0f : 0.0f;
    float frc2 = trppt > 0.0f ? pptflx/(trppt*ainc) : 0.0f;
    for (int nk=0; nk<nz; ++nk) {
        parcel_t[nk] = 0.0f;
        parcel_q[nk] = 0.0f;
        resolved_precip[nk] = 0.0f;
        thetaeu[nk] = 0.0f;
    }
    for (int ntc=0; ntc<nstep; ++ntc) {
        for (int nk=0; nk<=cloud_top; ++nk) {
            thfxin[nk] = thfxout[nk] = qfxin[nk] = qfxout[nk] = 0.0f;
            qlqout[nk] = qicout[nk] = 0.0f;
            qliq[nk] = qice[nk] = 0.0f;
        }
        for (int nk=1; nk<=cloud_top; ++nk) {
            if (omega[nk] <= 0.0f) {
                thfxin[nk] = -fxm[nk]*parcel_t[nk-1];
                qfxin[nk] = -fxm[nk]*parcel_q[nk-1];
                qlqout[nk] = -fxm[nk]*resolved_precip[nk-1];
                qliq[nk] = -fxm[nk]*thetaeu[nk-1];
                thfxout[nk-1] += thfxin[nk];
                qfxout[nk-1] += qfxin[nk];
                qicout[nk-1] += qlqout[nk];
                qice[nk-1] += qliq[nk];
            } else {
                thfxout[nk] = fxm[nk]*parcel_t[nk];
                qfxout[nk] = fxm[nk]*parcel_q[nk];
                qicout[nk] = fxm[nk]*resolved_precip[nk];
                qice[nk] = fxm[nk]*thetaeu[nk];
                thfxin[nk-1] += thfxout[nk];
                qfxin[nk-1] += qfxout[nk];
                qlqout[nk-1] += qicout[nk];
                qliq[nk-1] += qice[nk];
            }
        }
        for (int nk=0; nk<=cloud_top; ++nk) {
            parcel_t[nk] += (thfxin[nk]+detlq[nk]-thfxout[nk])
                            *dtime/cell_mass[nk];
            parcel_q[nk] += (qfxin[nk]+detice[nk]-qfxout[nk])
                            *dtime/cell_mass[nk];
            resolved_precip[nk] += (qlqout[nk]
                +pptliq[nk]*ainc*fbfrc*frc2-qicout[nk])
                *dtime/cell_mass[nk];
            thetaeu[nk] += (qliq[nk]
                +pptice[nk]*ainc*fbfrc*frc2-qice[nk])
                *dtime/cell_mass[nk];
        }
    }
    // WRF module_cu_kfeta.F:2599-2640.  Phase closure is not a mass-only
    // remap: WARM_RAIN and !F_QS return frozen condensate through QC/QR and
    // apply latent fusion to TG before DTDT is diagnosed.  F_QS,!F_QI keeps
    // snow separate and folds only cloud ice into snow without changing TG.
    int melting_level = -1;
    if (phase_mode == KF_PHASE_NO_SEPARATE_SNOW) {
        for (int nk=0; nk<=cloud_top; ++nk) {
            int index = nk*ncol+column;
            if (temperature[index] > 273.16f) melting_level = nk;
        }
    }
    for (int nk=0; nk<=cloud_top; ++nk) {
        int index = nk*ncol+column;
        float cpm = 1004.5f*(1.0f+0.887f*qg[nk]);
        if (phase_mode == KF_PHASE_WARM_RAIN) {
            tg[nk] -= (parcel_q[nk]+thetaeu[nk])*KF_RLF/cpm;
            rqccuten[index] = (parcel_t[nk]+parcel_q[nk])/timec;
            rqicuten[index] = 0.0f;
            rqrcuten[index] = (resolved_precip[nk]+thetaeu[nk])/timec;
            rqscuten[index] = 0.0f;
        } else if (phase_mode == KF_PHASE_NO_SEPARATE_SNOW) {
            if (nk <= melting_level)
                tg[nk] -= (parcel_q[nk]+thetaeu[nk])*KF_RLF/cpm;
            else
                tg[nk] += (parcel_t[nk]+resolved_precip[nk])*KF_RLF/cpm;
            rqccuten[index] = (parcel_t[nk]+parcel_q[nk])/timec;
            rqicuten[index] = 0.0f;
            rqrcuten[index] = (resolved_precip[nk]+thetaeu[nk])/timec;
            rqscuten[index] = 0.0f;
        } else if (phase_mode == KF_PHASE_SEPARATE_SNOW) {
            rqccuten[index] = parcel_t[nk]/timec;
            rqicuten[index] = 0.0f;
            rqrcuten[index] = resolved_precip[nk]/timec;
            rqscuten[index] = (thetaeu[nk]+parcel_q[nk])/timec;
        } else {
            rqccuten[index] = parcel_t[nk]/timec;
            rqicuten[index] = parcel_q[nk]/timec;
            rqrcuten[index] = resolved_precip[nk]/timec;
            rqscuten[index] = thetaeu[nk]/timec;
        }
        rthcuten[index] = (tg[nk]-temperature[index])/(exner[index]*timec);
    }

    for (int k = 0; k < nz; ++k) {
        int index = k * ncol + column;
        updraft_out[index] = updraft[k];
        downdraft_out[index] = downdraft[k];
    }
    float precip_rate = pptflx*(1.0f-fbfrc)/dxsq;
    float feedback_time = timec;
    if (!shallow && advective_time < timec)
        feedback_time = fmaxf(floorf(advective_time/dt+0.5f), 0.0f)*dt;
    rainc[column] = precip_rate*fminf(cudt, feedback_time);
    triggered[column] = 1;
    cape_before[column] = cape;
    cape_after[column] = adjusted_cape;
    closure_time[column] = timec;
    nca_seconds[column] = shallow ? cudt : feedback_time;
    shallow_out[column] = shallow ? 1 : 0;
    cloud_base[column] = klcl;
    cloud_top_out[column] = cloud_top;
}
