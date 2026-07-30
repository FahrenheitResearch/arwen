use rayon::prelude::*;
use rustwx_core::GridShape;

use crate::ecape::{EcapeVolumeInputs, SurfaceInputs, validate_len, validate_len_or_absent};
use crate::error::CalcError;
use crate::severe::{
    CapeCinOutputs, WindGridInputs, compute_cape_cin, compute_ehi, compute_shear, compute_srh,
    compute_srh_hemispheric,
};

#[derive(Debug, Clone, Copy)]
pub struct TemperatureAdvectionInputs<'a> {
    pub grid: GridShape,
    pub temperature_2d: &'a [f64],
    pub u_2d_ms: &'a [f64],
    pub v_2d_ms: &'a [f64],
    pub dx_m: f64,
    pub dy_m: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SurfaceThermoOutputs {
    pub dewpoint_2m_c: Vec<f64>,
    pub relative_humidity_2m_pct: Vec<f64>,
    pub theta_e_2m_k: Vec<f64>,
    pub wetbulb_2m_c: Vec<f64>,
    pub dewpoint_depression_2m_c: Vec<f64>,
    pub vpd_2m_hpa: Vec<f64>,
    pub fire_weather_composite: Vec<f64>,
    pub apparent_temperature_2m_c: Vec<f64>,
    pub heat_index_2m_c: Vec<f64>,
    pub wind_chill_2m_c: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EhiLayerOutputs {
    pub ehi_01km: Vec<f64>,
    pub ehi_03km: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
struct SurfaceThermoState {
    dewpoint_2m_c: Vec<f64>,
    relative_humidity_2m_pct: Vec<f64>,
    theta_e_2m_k: Vec<f64>,
    wetbulb_2m_c: Vec<f64>,
    dewpoint_depression_2m_c: Vec<f64>,
    vpd_2m_hpa: Vec<f64>,
    fire_weather_composite: Vec<f64>,
    heat_index_2m_c: Vec<f64>,
    wind_chill_2m_c: Vec<f64>,
    apparent_temperature_2m_c: Vec<f64>,
}

pub fn compute_surface_thermo(
    grid: GridShape,
    surface: SurfaceInputs<'_>,
) -> Result<SurfaceThermoOutputs, CalcError> {
    let state = compute_surface_thermo_state(grid, surface)?;

    Ok(SurfaceThermoOutputs {
        dewpoint_2m_c: state.dewpoint_2m_c,
        relative_humidity_2m_pct: state.relative_humidity_2m_pct,
        theta_e_2m_k: state.theta_e_2m_k,
        wetbulb_2m_c: state.wetbulb_2m_c,
        dewpoint_depression_2m_c: state.dewpoint_depression_2m_c,
        vpd_2m_hpa: state.vpd_2m_hpa,
        fire_weather_composite: state.fire_weather_composite,
        apparent_temperature_2m_c: state.apparent_temperature_2m_c,
        heat_index_2m_c: state.heat_index_2m_c,
        wind_chill_2m_c: state.wind_chill_2m_c,
    })
}

pub fn compute_2m_dewpoint(
    grid: GridShape,
    surface: SurfaceInputs<'_>,
) -> Result<Vec<f64>, CalcError> {
    Ok(compute_surface_thermo(grid, surface)?.dewpoint_2m_c)
}

pub fn compute_2m_relative_humidity(
    grid: GridShape,
    surface: SurfaceInputs<'_>,
) -> Result<Vec<f64>, CalcError> {
    Ok(compute_surface_thermo(grid, surface)?.relative_humidity_2m_pct)
}

pub fn compute_2m_theta_e(
    grid: GridShape,
    surface: SurfaceInputs<'_>,
) -> Result<Vec<f64>, CalcError> {
    Ok(compute_surface_thermo(grid, surface)?.theta_e_2m_k)
}

pub fn compute_dewpoint_from_pressure_and_mixing_ratio(
    pressure_hpa: &[f64],
    mixing_ratio_kgkg: &[f64],
) -> Result<Vec<f64>, CalcError> {
    let n = pressure_hpa.len();
    validate_len("mixing_ratio_kgkg", mixing_ratio_kgkg.len(), n)?;

    Ok(pressure_hpa
        .iter()
        .zip(mixing_ratio_kgkg.iter())
        .map(|(&pressure_hpa, &mixing_ratio_kgkg)| {
            dewpoint_from_mixing_ratio(pressure_hpa, mixing_ratio_kgkg)
        })
        .collect())
}

pub fn compute_relative_humidity_from_pressure_temperature_and_mixing_ratio(
    pressure_hpa: &[f64],
    temperature_c: &[f64],
    mixing_ratio_kgkg: &[f64],
) -> Result<Vec<f64>, CalcError> {
    let n = pressure_hpa.len();
    validate_len("temperature_c", temperature_c.len(), n)?;
    let dewpoint_c =
        compute_dewpoint_from_pressure_and_mixing_ratio(pressure_hpa, mixing_ratio_kgkg)?;

    Ok(temperature_c
        .iter()
        .zip(dewpoint_c.iter())
        .map(|(&temperature_c, &dewpoint_c)| {
            metrust::calc::thermo::relative_humidity_from_dewpoint(temperature_c, dewpoint_c)
        })
        .collect())
}

pub fn compute_theta_e_from_pressure_temperature_and_mixing_ratio(
    pressure_hpa: &[f64],
    temperature_c: &[f64],
    mixing_ratio_kgkg: &[f64],
) -> Result<Vec<f64>, CalcError> {
    let n = pressure_hpa.len();
    validate_len("temperature_c", temperature_c.len(), n)?;
    let dewpoint_c =
        compute_dewpoint_from_pressure_and_mixing_ratio(pressure_hpa, mixing_ratio_kgkg)?;

    Ok(pressure_hpa
        .iter()
        .zip(temperature_c.iter())
        .zip(dewpoint_c.iter())
        .map(|((&pressure_hpa, &temperature_c), &dewpoint_c)| {
            metrust::calc::thermo::equivalent_potential_temperature(
                pressure_hpa,
                temperature_c,
                dewpoint_c,
            )
        })
        .collect())
}

pub fn compute_2m_heat_index(
    grid: GridShape,
    surface: SurfaceInputs<'_>,
) -> Result<Vec<f64>, CalcError> {
    Ok(compute_surface_thermo(grid, surface)?.heat_index_2m_c)
}

pub fn compute_2m_wind_chill(
    grid: GridShape,
    surface: SurfaceInputs<'_>,
) -> Result<Vec<f64>, CalcError> {
    Ok(compute_surface_thermo(grid, surface)?.wind_chill_2m_c)
}

pub fn compute_2m_apparent_temperature(
    grid: GridShape,
    surface: SurfaceInputs<'_>,
) -> Result<Vec<f64>, CalcError> {
    Ok(compute_surface_thermo_state(grid, surface)?.apparent_temperature_2m_c)
}

pub fn compute_temperature_advection(
    inputs: TemperatureAdvectionInputs<'_>,
) -> Result<Vec<f64>, CalcError> {
    validate_temperature_advection_inputs(inputs)?;
    Ok(metrust::calc::kinematics::temperature_advection(
        inputs.temperature_2d,
        inputs.u_2d_ms,
        inputs.v_2d_ms,
        inputs.grid.nx,
        inputs.grid.ny,
        inputs.dx_m,
        inputs.dy_m,
    ))
}

pub fn compute_temperature_advection_700mb(
    inputs: TemperatureAdvectionInputs<'_>,
) -> Result<Vec<f64>, CalcError> {
    compute_temperature_advection(inputs)
}

pub fn compute_temperature_advection_850mb(
    inputs: TemperatureAdvectionInputs<'_>,
) -> Result<Vec<f64>, CalcError> {
    compute_temperature_advection(inputs)
}

pub fn compute_lifted_index(
    grid: GridShape,
    volume: EcapeVolumeInputs<'_>,
    surface: SurfaceInputs<'_>,
) -> Result<Vec<f64>, CalcError> {
    validate_volume_inputs(grid, volume)?;
    validate_surface_inputs(grid, surface)?;

    let nxy = grid.len();
    Ok((0..nxy)
        .into_par_iter()
        .map(|ij| {
            let p_prof =
                column_with_surface_hpa(volume.pressure_pa, surface.psfc_pa, nxy, volume.nz, ij);
            if !profile_contains_pressure(&p_prof, 500.0) {
                return f64::NAN;
            }
            let mut t_prof = Vec::with_capacity(volume.nz + 1);
            let mut td_prof = Vec::with_capacity(volume.nz + 1);
            t_prof.push(surface.t2_k[ij] - 273.15);
            td_prof.push(dewpoint_from_mixing_ratio(
                surface.psfc_pa[ij] / 100.0,
                surface.q2_kgkg[ij],
            ));
            for k in 0..volume.nz {
                let idx = k * nxy + ij;
                let p_hpa = pressure_hpa_at(volume, nxy, k, ij);
                t_prof.push(volume.temperature_c[idx]);
                td_prof.push(dewpoint_from_mixing_ratio(p_hpa, volume.qvapor_kgkg[idx]));
            }

            metrust::calc::thermo::lifted_index(&p_prof, &t_prof, &td_prof)
        })
        .collect())
}

pub fn compute_dcape(
    grid: GridShape,
    volume: EcapeVolumeInputs<'_>,
    surface: SurfaceInputs<'_>,
) -> Result<Vec<f64>, CalcError> {
    validate_volume_inputs(grid, volume)?;
    validate_surface_inputs(grid, surface)?;

    let nxy = grid.len();
    Ok((0..nxy)
        .into_par_iter()
        .map(|ij| {
            let p_prof =
                column_with_surface_hpa(volume.pressure_pa, surface.psfc_pa, nxy, volume.nz, ij);
            let mut t_prof = Vec::with_capacity(volume.nz + 1);
            let mut td_prof = Vec::with_capacity(volume.nz + 1);
            t_prof.push(surface.t2_k[ij] - 273.15);
            td_prof.push(dewpoint_from_mixing_ratio(
                surface.psfc_pa[ij] / 100.0,
                surface.q2_kgkg[ij],
            ));
            for k in 0..volume.nz {
                let idx = k * nxy + ij;
                let p_hpa = pressure_hpa_at(volume, nxy, k, ij);
                t_prof.push(volume.temperature_c[idx]);
                td_prof.push(dewpoint_from_mixing_ratio(p_hpa, volume.qvapor_kgkg[idx]));
            }

            if p_prof.len() < 3
                || p_prof.iter().any(|value| !value.is_finite())
                || t_prof.iter().any(|value| !value.is_finite())
                || td_prof.iter().any(|value| !value.is_finite())
            {
                return f64::NAN;
            }

            metrust::calc::thermo::downdraft_cape(&p_prof, &t_prof, &td_prof)
        })
        .collect())
}

pub fn compute_lapse_rate_700_500(
    grid: GridShape,
    volume: EcapeVolumeInputs<'_>,
) -> Result<Vec<f64>, CalcError> {
    validate_volume_inputs(grid, volume)?;
    let nxy = grid.len();
    Ok((0..nxy)
        .into_par_iter()
        .map(|ij| {
            let pressures_hpa = column_hpa(volume.pressure_pa, nxy, volume.nz, ij);
            let temperatures_c = column(volume.temperature_c, nxy, volume.nz, ij);
            let qvapor_kgkg = column(volume.qvapor_kgkg, nxy, volume.nz, ij);
            let heights_m = column(volume.height_agl_m, nxy, volume.nz, ij);

            let dewpoints_c = pressures_hpa
                .iter()
                .zip(qvapor_kgkg.iter())
                .map(|(pressure_hpa, mixing_ratio_kgkg)| {
                    dewpoint_from_mixing_ratio(*pressure_hpa, *mixing_ratio_kgkg)
                })
                .collect::<Vec<_>>();
            let virtual_temperatures_c = pressures_hpa
                .iter()
                .zip(temperatures_c.iter())
                .zip(dewpoints_c.iter())
                .map(|((pressure_hpa, temperature_c), dewpoint_c)| {
                    metrust::calc::thermo::virtual_temperature_from_dewpoint(
                        *temperature_c,
                        *dewpoint_c,
                        *pressure_hpa,
                    )
                })
                .collect::<Vec<_>>();

            let Some(tv700) = interp_at_pressure(&pressures_hpa, &virtual_temperatures_c, 700.0)
            else {
                return f64::NAN;
            };
            let Some(tv500) = interp_at_pressure(&pressures_hpa, &virtual_temperatures_c, 500.0)
            else {
                return f64::NAN;
            };
            let Some(z700) = interp_at_pressure(&pressures_hpa, &heights_m, 700.0) else {
                return f64::NAN;
            };
            let Some(z500) = interp_at_pressure(&pressures_hpa, &heights_m, 500.0) else {
                return f64::NAN;
            };

            let dz_km = (z500 - z700) / 1000.0;
            if dz_km > 0.0 {
                (tv700 - tv500) / dz_km
            } else {
                f64::NAN
            }
        })
        .collect())
}

pub fn compute_lapse_rate_0_3km(
    grid: GridShape,
    volume: EcapeVolumeInputs<'_>,
    surface: SurfaceInputs<'_>,
) -> Result<Vec<f64>, CalcError> {
    validate_volume_inputs(grid, volume)?;
    validate_surface_inputs(grid, surface)?;
    let nxy = grid.len();
    Ok((0..nxy)
        .into_par_iter()
        .map(|ij| {
            let mut heights_m = Vec::with_capacity(volume.nz + 1);
            let mut temperatures_c = Vec::with_capacity(volume.nz + 1);
            heights_m.push(0.0);
            temperatures_c.push(surface.t2_k[ij] - 273.15);
            for k in 0..volume.nz {
                let idx = k * nxy + ij;
                heights_m.push(volume.height_agl_m[idx]);
                temperatures_c.push(volume.temperature_c[idx]);
            }

            let Some(t_3km) = interp_at_height(&heights_m, &temperatures_c, 3000.0) else {
                return f64::NAN;
            };
            (temperatures_c[0] - t_3km) / 3.0
        })
        .collect())
}

pub fn compute_shear_01km(wind: WindGridInputs<'_>) -> Result<Vec<f64>, CalcError> {
    compute_shear(wind, 0.0, 1000.0)
}

pub fn compute_shear_06km(wind: WindGridInputs<'_>) -> Result<Vec<f64>, CalcError> {
    compute_shear(wind, 0.0, 6000.0)
}

pub fn compute_srh_01km(wind: WindGridInputs<'_>) -> Result<Vec<f64>, CalcError> {
    compute_srh(wind, 1000.0)
}

pub fn compute_srh_03km(wind: WindGridInputs<'_>) -> Result<Vec<f64>, CalcError> {
    compute_srh(wind, 3000.0)
}

pub fn compute_srh_01km_hemispheric(
    wind: WindGridInputs<'_>,
    lat_deg: &[f64],
) -> Result<Vec<f64>, CalcError> {
    compute_srh_hemispheric(wind, lat_deg, 1000.0)
}

pub fn compute_srh_03km_hemispheric(
    wind: WindGridInputs<'_>,
    lat_deg: &[f64],
) -> Result<Vec<f64>, CalcError> {
    compute_srh_hemispheric(wind, lat_deg, 3000.0)
}

pub fn compute_ehi_01km(
    grid: GridShape,
    cape_jkg: &[f64],
    srh_01km_m2s2: &[f64],
) -> Result<Vec<f64>, CalcError> {
    compute_ehi(grid, cape_jkg, srh_01km_m2s2)
}

pub fn compute_ehi_03km(
    grid: GridShape,
    cape_jkg: &[f64],
    srh_03km_m2s2: &[f64],
) -> Result<Vec<f64>, CalcError> {
    compute_ehi(grid, cape_jkg, srh_03km_m2s2)
}

pub fn compute_ehi_layers(
    grid: GridShape,
    cape_jkg: &[f64],
    srh_01km_m2s2: &[f64],
    srh_03km_m2s2: &[f64],
) -> Result<EhiLayerOutputs, CalcError> {
    Ok(EhiLayerOutputs {
        ehi_01km: compute_ehi_01km(grid, cape_jkg, srh_01km_m2s2)?,
        ehi_03km: compute_ehi_03km(grid, cape_jkg, srh_03km_m2s2)?,
    })
}

pub fn compute_sbcape_cin(
    grid: GridShape,
    volume: EcapeVolumeInputs<'_>,
    surface: SurfaceInputs<'_>,
    top_m: Option<f64>,
) -> Result<CapeCinOutputs, CalcError> {
    compute_cape_cin(grid, volume, surface, "sb", top_m)
}

pub fn compute_mlcape_cin(
    grid: GridShape,
    volume: EcapeVolumeInputs<'_>,
    surface: SurfaceInputs<'_>,
    top_m: Option<f64>,
) -> Result<CapeCinOutputs, CalcError> {
    compute_cape_cin(grid, volume, surface, "ml", top_m)
}

pub fn compute_mucape_cin(
    grid: GridShape,
    volume: EcapeVolumeInputs<'_>,
    surface: SurfaceInputs<'_>,
    top_m: Option<f64>,
) -> Result<CapeCinOutputs, CalcError> {
    compute_cape_cin(grid, volume, surface, "mu", top_m)
}

pub fn compute_sbcape(
    grid: GridShape,
    volume: EcapeVolumeInputs<'_>,
    surface: SurfaceInputs<'_>,
    top_m: Option<f64>,
) -> Result<Vec<f64>, CalcError> {
    Ok(compute_sbcape_cin(grid, volume, surface, top_m)?.cape_jkg)
}

pub fn compute_sbcin(
    grid: GridShape,
    volume: EcapeVolumeInputs<'_>,
    surface: SurfaceInputs<'_>,
    top_m: Option<f64>,
) -> Result<Vec<f64>, CalcError> {
    Ok(compute_sbcape_cin(grid, volume, surface, top_m)?.cin_jkg)
}

pub fn compute_sblcl(
    grid: GridShape,
    volume: EcapeVolumeInputs<'_>,
    surface: SurfaceInputs<'_>,
    top_m: Option<f64>,
) -> Result<Vec<f64>, CalcError> {
    Ok(compute_sbcape_cin(grid, volume, surface, top_m)?.lcl_m)
}

pub fn compute_mlcape(
    grid: GridShape,
    volume: EcapeVolumeInputs<'_>,
    surface: SurfaceInputs<'_>,
    top_m: Option<f64>,
) -> Result<Vec<f64>, CalcError> {
    Ok(compute_mlcape_cin(grid, volume, surface, top_m)?.cape_jkg)
}

pub fn compute_mlcin(
    grid: GridShape,
    volume: EcapeVolumeInputs<'_>,
    surface: SurfaceInputs<'_>,
    top_m: Option<f64>,
) -> Result<Vec<f64>, CalcError> {
    Ok(compute_mlcape_cin(grid, volume, surface, top_m)?.cin_jkg)
}

pub fn compute_mucape(
    grid: GridShape,
    volume: EcapeVolumeInputs<'_>,
    surface: SurfaceInputs<'_>,
    top_m: Option<f64>,
) -> Result<Vec<f64>, CalcError> {
    Ok(compute_mucape_cin(grid, volume, surface, top_m)?.cape_jkg)
}

pub fn compute_mucin(
    grid: GridShape,
    volume: EcapeVolumeInputs<'_>,
    surface: SurfaceInputs<'_>,
    top_m: Option<f64>,
) -> Result<Vec<f64>, CalcError> {
    Ok(compute_mucape_cin(grid, volume, surface, top_m)?.cin_jkg)
}

fn validate_surface_inputs(grid: GridShape, surface: SurfaceInputs<'_>) -> Result<(), CalcError> {
    let n = grid.len();
    validate_len("psfc_pa", surface.psfc_pa.len(), n)?;
    validate_len("t2_k", surface.t2_k.len(), n)?;
    validate_len("q2_kgkg", surface.q2_kgkg.len(), n)?;
    validate_len("u10_ms", surface.u10_ms.len(), n)?;
    validate_len("v10_ms", surface.v10_ms.len(), n)?;
    Ok(())
}

/// Volume validation for the purely thermodynamic kernels in this module
/// (lifted index, DCAPE, lapse rates), none of which read the wind
/// members: winds may be absent (empty) so the store-ingest derived lane
/// can free them before these kernels run. Wrong-length non-empty winds
/// are still rejected.
fn validate_volume_inputs(grid: GridShape, volume: EcapeVolumeInputs<'_>) -> Result<(), CalcError> {
    let n3d = grid.len() * volume.nz;
    if pressure_is_levels(volume) {
        validate_len("pressure_levels_pa", volume.pressure_pa.len(), volume.nz)?;
    } else {
        validate_len("pressure_pa", volume.pressure_pa.len(), n3d)?;
    }
    validate_len("temperature_c", volume.temperature_c.len(), n3d)?;
    validate_len("qvapor_kgkg", volume.qvapor_kgkg.len(), n3d)?;
    validate_len("height_agl_m", volume.height_agl_m.len(), n3d)?;
    validate_len_or_absent("u_ms", volume.u_ms.len(), n3d)?;
    validate_len_or_absent("v_ms", volume.v_ms.len(), n3d)?;
    Ok(())
}

fn pressure_is_levels(volume: EcapeVolumeInputs<'_>) -> bool {
    volume.pressure_pa.len() == volume.nz
}

fn pressure_hpa_at(volume: EcapeVolumeInputs<'_>, nxy: usize, k: usize, ij: usize) -> f64 {
    if pressure_is_levels(volume) {
        volume.pressure_pa[k] / 100.0
    } else {
        volume.pressure_pa[k * nxy + ij] / 100.0
    }
}

fn validate_temperature_advection_inputs(
    inputs: TemperatureAdvectionInputs<'_>,
) -> Result<(), CalcError> {
    let n = inputs.grid.len();
    validate_len("temperature_2d", inputs.temperature_2d.len(), n)?;
    validate_len("u_2d_ms", inputs.u_2d_ms.len(), n)?;
    validate_len("v_2d_ms", inputs.v_2d_ms.len(), n)?;
    Ok(())
}

/// One grid point's surface-thermo bundle, computed in a single pass so the
/// shared intermediates (dewpoint, RH, wind speed, VPD) are evaluated once.
/// Per-point math is exactly the former serial loop's; only the iteration
/// is rayon-parallel now (each point is independent, results are identical).
#[derive(Debug, Clone, Copy)]
struct SurfaceThermoPoint {
    dewpoint_c: f64,
    relative_humidity_pct: f64,
    theta_e_k: f64,
    wetbulb_c: f64,
    dewpoint_depression_c: f64,
    vpd_hpa: f64,
    fire_weather: f64,
    heat_index_c: f64,
    wind_chill_c: f64,
    apparent_temperature_c: f64,
}

fn compute_surface_thermo_state(
    grid: GridShape,
    surface: SurfaceInputs<'_>,
) -> Result<SurfaceThermoState, CalcError> {
    validate_surface_inputs(grid, surface)?;

    let points: Vec<SurfaceThermoPoint> = (0..grid.len())
        .into_par_iter()
        .map(|idx| {
            let pressure_hpa = surface.psfc_pa[idx] / 100.0;
            let temperature_c = surface.t2_k[idx] - 273.15;
            let dewpoint_c = dewpoint_from_mixing_ratio(pressure_hpa, surface.q2_kgkg[idx]);
            let relative_humidity_pct =
                metrust::calc::thermo::relative_humidity_from_dewpoint(temperature_c, dewpoint_c);
            let wind_speed_ms = (surface.u10_ms[idx] * surface.u10_ms[idx]
                + surface.v10_ms[idx] * surface.v10_ms[idx])
                .sqrt();
            let vpd_hpa = vapor_pressure_deficit_hpa(temperature_c, dewpoint_c);
            SurfaceThermoPoint {
                dewpoint_c,
                relative_humidity_pct,
                theta_e_k: metrust::calc::thermo::equivalent_potential_temperature(
                    pressure_hpa,
                    temperature_c,
                    dewpoint_c,
                ),
                wetbulb_c: metrust::calc::wet_bulb_temperature(
                    pressure_hpa,
                    temperature_c,
                    dewpoint_c,
                ),
                dewpoint_depression_c: (temperature_c - dewpoint_c).max(0.0),
                vpd_hpa,
                fire_weather: fire_weather_composite_value(
                    temperature_c,
                    relative_humidity_pct,
                    wind_speed_ms,
                    vpd_hpa,
                ),
                heat_index_c: metrust::calc::atmo::heat_index(temperature_c, relative_humidity_pct),
                wind_chill_c: metrust::calc::atmo::windchill(temperature_c, wind_speed_ms),
                apparent_temperature_c: metrust::calc::atmo::apparent_temperature(
                    temperature_c,
                    relative_humidity_pct,
                    wind_speed_ms,
                ),
            }
        })
        .collect();

    let unzip = |field: fn(&SurfaceThermoPoint) -> f64| -> Vec<f64> {
        points.par_iter().map(field).collect()
    };
    Ok(SurfaceThermoState {
        dewpoint_2m_c: unzip(|p| p.dewpoint_c),
        relative_humidity_2m_pct: unzip(|p| p.relative_humidity_pct),
        theta_e_2m_k: unzip(|p| p.theta_e_k),
        wetbulb_2m_c: unzip(|p| p.wetbulb_c),
        dewpoint_depression_2m_c: unzip(|p| p.dewpoint_depression_c),
        vpd_2m_hpa: unzip(|p| p.vpd_hpa),
        fire_weather_composite: unzip(|p| p.fire_weather),
        heat_index_2m_c: unzip(|p| p.heat_index_c),
        wind_chill_2m_c: unzip(|p| p.wind_chill_c),
        apparent_temperature_2m_c: unzip(|p| p.apparent_temperature_c),
    })
}

fn dewpoint_from_mixing_ratio(pressure_hpa: f64, mixing_ratio_kgkg: f64) -> f64 {
    let q = mixing_ratio_kgkg.max(0.0);
    let vapor_pressure_hpa = (q * pressure_hpa / (0.622 + q)).max(1.0e-10);
    let ln_e = (vapor_pressure_hpa / 6.112).ln();
    (243.5 * ln_e) / (17.67 - ln_e)
}

fn vapor_pressure_deficit_hpa(temperature_c: f64, dewpoint_c: f64) -> f64 {
    let saturation_hpa = metrust::calc::thermo::saturation_vapor_pressure(temperature_c);
    let actual_hpa = metrust::calc::thermo::vapor_pressure(dewpoint_c);
    (saturation_hpa - actual_hpa).max(0.0)
}

fn fire_weather_composite_value(
    temperature_c: f64,
    relative_humidity_pct: f64,
    wind_speed_ms: f64,
    vpd_hpa: f64,
) -> f64 {
    const MPH_PER_MS: f64 = 2.236_936_292_054_4;

    // Blend Fosberg and capped HDW so the public-facing composite stays on
    // a stable 0-100 scale while still responding directly to VPD.
    let fosberg = metrust::calc::fosberg_fire_weather_index(
        temperature_c * 9.0 / 5.0 + 32.0,
        relative_humidity_pct,
        wind_speed_ms * MPH_PER_MS,
    );
    let hdw =
        metrust::calc::hot_dry_windy(temperature_c, relative_humidity_pct, wind_speed_ms, vpd_hpa)
            .clamp(0.0, 100.0);

    (0.5 * fosberg + 0.5 * hdw).clamp(0.0, 100.0)
}

fn column(values: &[f64], nxy: usize, nz: usize, ij: usize) -> Vec<f64> {
    (0..nz).map(|k| values[k * nxy + ij]).collect()
}

fn column_hpa(values_pa: &[f64], nxy: usize, nz: usize, ij: usize) -> Vec<f64> {
    if values_pa.len() == nz {
        values_pa.iter().map(|value| value / 100.0).collect()
    } else {
        (0..nz).map(|k| values_pa[k * nxy + ij] / 100.0).collect()
    }
}

fn column_with_surface_hpa(
    values_pa: &[f64],
    surface_pa: &[f64],
    nxy: usize,
    nz: usize,
    ij: usize,
) -> Vec<f64> {
    let mut column = Vec::with_capacity(nz + 1);
    column.push(surface_pa[ij] / 100.0);
    if values_pa.len() == nz {
        column.extend(values_pa.iter().map(|value| value / 100.0));
    } else {
        column.extend((0..nz).map(|k| values_pa[k * nxy + ij] / 100.0));
    }
    column
}

fn profile_contains_pressure(pressures_hpa: &[f64], target_hpa: f64) -> bool {
    let Some((&surface_hpa, rest)) = pressures_hpa.split_first() else {
        return false;
    };
    let Some(&top_hpa) = rest.last().or(Some(&surface_hpa)) else {
        return false;
    };
    surface_hpa >= target_hpa && top_hpa <= target_hpa
}

fn interp_at_pressure(pressures_hpa: &[f64], values: &[f64], target_hpa: f64) -> Option<f64> {
    if pressures_hpa.len() != values.len() || pressures_hpa.is_empty() {
        return None;
    }
    for i in 0..pressures_hpa.len() - 1 {
        let p0 = pressures_hpa[i];
        let p1 = pressures_hpa[i + 1];
        if (p0 >= target_hpa && p1 <= target_hpa) || (p0 <= target_hpa && p1 >= target_hpa) {
            let frac = (target_hpa - p0) / (p1 - p0);
            return Some(values[i] + frac * (values[i + 1] - values[i]));
        }
    }
    None
}

fn interp_at_height(heights_m: &[f64], values: &[f64], target_m: f64) -> Option<f64> {
    if heights_m.len() != values.len() || heights_m.is_empty() {
        return None;
    }
    if target_m < heights_m[0] || target_m > *heights_m.last().unwrap() {
        return None;
    }
    for i in 0..heights_m.len() - 1 {
        let h0 = heights_m[i];
        let h1 = heights_m[i + 1];
        if h0 <= target_m && h1 >= target_m {
            let frac = if (h1 - h0).abs() < f64::EPSILON {
                0.0
            } else {
                (target_m - h0) / (h1 - h0)
            };
            return Some(values[i] + frac * (values[i + 1] - values[i]));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dewpoint_from_mixing_ratio_matches_expected_surface_value() {
        let td = dewpoint_from_mixing_ratio(1000.0, 0.014);
        assert!(td > 18.0 && td < 21.0, "dewpoint={td}");
    }

    #[test]
    fn pressure_profile_thermo_wrappers_return_finite_values() {
        let pressure_hpa = [1000.0, 850.0];
        let temperature_c = [28.0, 14.0];
        let mixing_ratio_kgkg = [0.014, 0.009];

        let dewpoint_c =
            compute_dewpoint_from_pressure_and_mixing_ratio(&pressure_hpa, &mixing_ratio_kgkg)
                .unwrap();
        let relative_humidity_pct =
            compute_relative_humidity_from_pressure_temperature_and_mixing_ratio(
                &pressure_hpa,
                &temperature_c,
                &mixing_ratio_kgkg,
            )
            .unwrap();
        let theta_e_k = compute_theta_e_from_pressure_temperature_and_mixing_ratio(
            &pressure_hpa,
            &temperature_c,
            &mixing_ratio_kgkg,
        )
        .unwrap();

        assert_eq!(dewpoint_c.len(), 2);
        assert!(dewpoint_c.iter().all(|value| value.is_finite()));
        assert!(relative_humidity_pct.iter().all(|value| value.is_finite()));
        assert!(theta_e_k.iter().all(|value| value.is_finite()));
        assert!(relative_humidity_pct.iter().all(|value| *value > 0.0));
        assert!(theta_e_k[0] > 330.0);
    }

    #[test]
    fn pressure_profile_thermo_wrappers_validate_lengths() {
        let err = compute_relative_humidity_from_pressure_temperature_and_mixing_ratio(
            &[1000.0, 850.0],
            &[20.0],
            &[0.01, 0.008],
        )
        .unwrap_err();

        assert!(matches!(
            err,
            CalcError::LengthMismatch {
                field: "temperature_c",
                expected: 2,
                actual: 1,
            }
        ));
    }

    #[test]
    fn pressure_interpolation_handles_standard_level() {
        let pressures = [1000.0, 850.0, 700.0, 500.0];
        let values = [20.0, 14.0, 8.0, -8.0];
        assert_eq!(interp_at_pressure(&pressures, &values, 700.0), Some(8.0));
        assert!(interp_at_pressure(&pressures, &values, 925.0).is_some());
    }

    #[test]
    fn dcape_grid_wrapper_returns_non_negative_values() {
        let grid = GridShape::new(1, 1).unwrap();
        let volume = EcapeVolumeInputs {
            pressure_pa: &[92500.0, 85000.0, 70000.0, 50000.0, 30000.0],
            temperature_c: &[25.0, 20.0, 5.0, -15.0, -40.0],
            qvapor_kgkg: &[0.008, 0.005, 0.003, 0.001, 0.0004],
            height_agl_m: &[700.0, 1500.0, 3000.0, 5500.0, 9000.0],
            u_ms: &[5.0, 10.0, 15.0, 20.0, 25.0],
            v_ms: &[0.0, 3.0, 8.0, 12.0, 15.0],
            nz: 5,
        };
        let surface = SurfaceInputs {
            psfc_pa: &[100000.0],
            t2_k: &[303.15],
            q2_kgkg: &[0.014],
            u10_ms: &[5.0],
            v10_ms: &[0.0],
        };

        let dcape = compute_dcape(grid, volume, surface).unwrap();
        assert_eq!(dcape.len(), 1);
        assert!(dcape[0].is_finite());
        assert!(dcape[0] >= 0.0);
    }

    #[test]
    fn height_interpolation_requires_target_inside_profile() {
        let heights = [0.0, 1500.0, 3000.0];
        let values = [25.0, 15.0, 5.0];
        assert_eq!(interp_at_height(&heights, &values, 3000.0), Some(5.0));
        assert!(interp_at_height(&heights, &values, 4000.0).is_none());
    }

    #[test]
    fn lapse_rate_700_500_uses_virtual_temperature() {
        let grid = GridShape::new(1, 1).unwrap();
        let volume = EcapeVolumeInputs {
            pressure_pa: &[70000.0, 50000.0],
            temperature_c: &[8.0, -8.0],
            qvapor_kgkg: &[0.006, 0.001],
            height_agl_m: &[3000.0, 5600.0],
            u_ms: &[10.0, 20.0],
            v_ms: &[0.0, 10.0],
            nz: 2,
        };

        let lapse_rate = compute_lapse_rate_700_500(grid, volume).unwrap();
        let td700 = dewpoint_from_mixing_ratio(700.0, 0.006);
        let td500 = dewpoint_from_mixing_ratio(500.0, 0.001);
        let tv700 = metrust::calc::thermo::virtual_temperature_from_dewpoint(8.0, td700, 700.0);
        let tv500 = metrust::calc::thermo::virtual_temperature_from_dewpoint(-8.0, td500, 500.0);
        let expected = (tv700 - tv500) / 2.6;

        assert_eq!(lapse_rate.len(), 1);
        assert!((lapse_rate[0] - expected).abs() < 1e-6);
    }

    #[test]
    fn temperature_advection_wrapper_matches_metrust_kernel() {
        let inputs = TemperatureAdvectionInputs {
            grid: GridShape::new(3, 1).unwrap(),
            temperature_2d: &[0.0, 1.0, 2.0],
            u_2d_ms: &[2.0, 2.0, 2.0],
            v_2d_ms: &[0.0, 0.0, 0.0],
            dx_m: 1000.0,
            dy_m: 1000.0,
        };
        let wrapper = compute_temperature_advection(inputs).unwrap();
        let direct = metrust::calc::kinematics::temperature_advection(
            inputs.temperature_2d,
            inputs.u_2d_ms,
            inputs.v_2d_ms,
            inputs.grid.nx,
            inputs.grid.ny,
            inputs.dx_m,
            inputs.dy_m,
        );
        assert_eq!(wrapper, direct);
    }

    #[test]
    fn surface_thermo_outputs_include_fire_weather_family() {
        let pressure_hpa = 1000.0;
        let cool_moist_q =
            metrust::calc::thermo::specific_humidity_from_dewpoint(pressure_hpa, 18.0);
        let hot_dry_q = metrust::calc::thermo::specific_humidity_from_dewpoint(pressure_hpa, 5.0);
        let outputs = compute_surface_thermo(
            GridShape::new(2, 1).unwrap(),
            SurfaceInputs {
                psfc_pa: &[100000.0, 100000.0],
                t2_k: &[293.15, 308.15],
                q2_kgkg: &[cool_moist_q, hot_dry_q],
                u10_ms: &[1.0, 12.0],
                v10_ms: &[0.0, 4.0],
            },
        )
        .unwrap();

        assert_eq!(outputs.wetbulb_2m_c.len(), 2);
        assert_eq!(outputs.dewpoint_depression_2m_c.len(), 2);
        assert_eq!(outputs.vpd_2m_hpa.len(), 2);
        assert_eq!(outputs.fire_weather_composite.len(), 2);

        for idx in 0..2 {
            let temperature_c = [20.0, 35.0][idx];
            assert!(outputs.wetbulb_2m_c[idx] <= temperature_c + 1.0e-6);
            assert!(outputs.wetbulb_2m_c[idx] >= outputs.dewpoint_2m_c[idx] - 1.0e-6);
            assert!(outputs.dewpoint_depression_2m_c[idx] >= 0.0);
            assert!(outputs.vpd_2m_hpa[idx] >= 0.0);
            assert!((0.0..=100.0).contains(&outputs.fire_weather_composite[idx]));
        }

        assert!(outputs.dewpoint_depression_2m_c[1] > outputs.dewpoint_depression_2m_c[0]);
        assert!(outputs.vpd_2m_hpa[1] > outputs.vpd_2m_hpa[0]);
        assert!(outputs.fire_weather_composite[1] > outputs.fire_weather_composite[0]);
    }
}
