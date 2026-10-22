// SPDX-License-Identifier: Apache-2.0
// CUDA orchestration around the pinned NOAA PET numerical functions.
// One independent catchment per thread; no BMI, host runtime or file access.
#define M_PI 3.14159265358979323846
// Original evaluation of Borges (2019), section 4.2, Algorithms 4/5:
// https://arxiv.org/abs/1904.09481 . Retain the balanced-operand correction;
// otherwise evaluate h*h-a*a-b*b with explicit fused residuals to avoid
// cancellation in the expanded correction. Rounding is contraction-independent.
// Power-of-two scaling handles finite extremes without changing normal
// expression rounding. No glibc hypot source is incorporated.
__device__ double pet_wind_norm(double u, double v) {
  double a=fmax(fabs(u),fabs(v)), b=fmin(fabs(u),fabs(v));
  if (b==0.0 || b<=a*0x1p-54) return a;
  int exponent;
  frexp(a,&exponent);
  a=ldexp(a,-exponent); b=ldexp(b,-exponent);
  const double h=__dsqrt_rn(__dadd_rn(__dmul_rn(a,a),__dmul_rn(b,b)));
  double correction;
  if (h<=2.0*b) {
    const double d=h-b;
    correction=__dadd_rn(__dmul_rn(a,__dsub_rn(2.0*d,a)),
        __dmul_rn(__dsub_rn(d,2.0*(a-b)),d));
  } else {
    const double hh=__dmul_rn(h,h), aa=__dmul_rn(a,a);
    correction=__dsub_rn(
        __dadd_rn(__fma_rn(-b,b,__dsub_rn(hh,aa)),__fma_rn(h,h,-hh)),
        __fma_rn(a,a,-aa));
  }
  return ldexp(__dsub_rn(h,correction/(2.0*h)),exponent);
}
// NVRTC lowers constant-exponent pow through libdevice, while GCC uses x*x
// for exponent two and libm pow for four. Keep the square exact and evaluate
// fourth powers with a compensated product to avoid an extra rounding ulp.
__device__ double pet_pow(double x, double exponent) {
  if (exponent == 2.0) return x*x;
  #ifdef PET_STRICT
  return glibc_pow(x,exponent);
  #else
  if (exponent == 4.0) {
    const double a=x*x, ae=fma(x,x,-a);
    const double p=a*a;
    const double pe=fma(a,a,-p) + (2.0*a*ae + ae*ae);
    return p+pe;
  }
  return pow(x,exponent);
  #endif
}
#define exp glibc_exp
#ifdef PET_STRICT
#define log glibc_log
#endif
#define pow pet_pow
#include "pet_numeric.cuh"
#undef pow

#ifndef FORCING_TYPE
#define FORCING_TYPE float
#endif

extern "C" __global__ void catchment_pet(
    const long long n, const int *methods,
    const FORCING_TYPE *temperature, const FORCING_TYPE *humidity,
    const FORCING_TYPE *uwind, const FORCING_TYPE *vwind,
    const FORCING_TYPE *pressure, const FORCING_TYPE *longwave,
    const FORCING_TYPE *shortwave, double *parameters,
    double *potential_et_m_s, double *diagnostics, int *status) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int method = methods[i];
  double *p = parameters + i * 12;
  pet_model value = {};
  pet_model *model = &value;
  const double t = temperature[i], q = humidity[i], pres = pressure[i];
  if (!isfinite(t) || t <= 35.85 || !isfinite(q) || q < 0.0 ||
      !isfinite(pres) || pres <= 0.0 || !isfinite(uwind[i]) ||
      !isfinite(vwind[i]) || !isfinite(longwave[i]) || !isfinite(shortwave[i])) {
    potential_et_m_s[i] = nan(""); status[i] |= 1; return;
  }
  model->pet_options.use_aerodynamic_method = method == 2;
  model->pet_options.use_combination_method = method == 3;
  model->pet_options.use_penman_monteith_method = method == 5;
  model->pet_params.wind_speed_measurement_height_m = p[0];
  model->pet_params.humidity_measurement_height_m = p[1];
  model->pet_params.vegetation_height_m = p[2];
  model->pet_params.zero_plane_displacement_height_m = p[3];
  model->pet_params.momentum_transfer_roughness_length_m = p[4];
  model->pet_params.heat_transfer_roughness_length_m = p[5];
  model->surf_rad_params.surface_longwave_emissivity = p[6];
  model->surf_rad_params.surface_shortwave_albedo = p[7];
  model->pet_forcing.canopy_resistance_sec_per_m = p[8];
  model->pet_forcing.water_temperature_C = p[9];
  model->pet_forcing.ground_heat_flux_W_per_sq_m = p[10];
  model->surf_rad_forcing.surface_skin_temperature_C = p[11];
  model->pet_forcing.air_temperature_C = t - TK;
  model->pet_forcing.relative_humidity_percent = -99.9;
  model->pet_forcing.specific_humidity_2m_kg_per_kg = q;
  model->pet_forcing.air_pressure_Pa = pres;
  const double u = uwind[i], v = vwind[i];
  model->pet_forcing.wind_speed_m_per_s = pet_wind_norm(u,v);
  if (model->pet_params.wind_speed_measurement_height_m >
      model->pet_params.zero_plane_displacement_height_m) {
    const double numerator = log(2.0 / model->pet_params.zero_plane_displacement_height_m);
    const double denominator = log(model->pet_params.wind_speed_measurement_height_m /
                                   model->pet_params.zero_plane_displacement_height_m);
    model->pet_forcing.wind_speed_m_per_s =
        model->pet_forcing.wind_speed_m_per_s * numerator / denominator;
  }
  // Native run_pet mutates this after every call, including the first.
  model->pet_params.wind_speed_measurement_height_m = 2.0;
  model->surf_rad_forcing.incoming_shortwave_radiation_W_per_sq_m = shortwave[i];
  model->surf_rad_forcing.incoming_longwave_radiation_W_per_sq_m = longwave[i];
  // Use the supplied shortwave for every PET method.
  if (method != 2)
    model->pet_forcing.net_radiation_W_per_sq_m = calculate_net_radiation_W_per_sq_m(model);
  double result;
  switch (method) {
    case 1: result = pevapotranspiration_energy_balance_method(model); break;
    case 2: result = pevapotranspiration_aerodynamic_method(model); break;
    case 3: result = pevapotranspiration_combination_method(model); break;
    case 4: result = pevapotranspiration_priestley_taylor_method(model); break;
    default: result = pevapotranspiration_penman_monteith_method(model); break;
  }
  if (result < 0.0) result = 0.0;
  potential_et_m_s[i] = result;
  p[0] = model->pet_params.wind_speed_measurement_height_m;
  p[2] = model->pet_params.vegetation_height_m;
  p[4] = model->pet_params.momentum_transfer_roughness_length_m;
  p[5] = model->pet_params.heat_transfer_roughness_length_m;
  p[9] = model->pet_forcing.water_temperature_C;
  const intermediate_vars *x = &model->inter_vars;
  diagnostics[i*6+0] = model->pet_forcing.net_radiation_W_per_sq_m;
  diagnostics[i*6+1] = model->pet_forcing.wind_speed_m_per_s;
  diagnostics[i*6+2] = x->air_saturation_vapor_pressure_Pa;
  diagnostics[i*6+3] = x->air_actual_vapor_pressure_Pa;
  diagnostics[i*6+4] = x->moist_air_density_kg_per_m3;
  diagnostics[i*6+5] = x->slope_sat_vap_press_curve_Pa_s;
  status[i] |= isfinite(result) ? 0 : 2;
}
