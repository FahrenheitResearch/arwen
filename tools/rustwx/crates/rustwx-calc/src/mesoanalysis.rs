use rayon::prelude::*;
use rustwx_core::GridShape;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::str::FromStr;

use crate::derived::compute_dewpoint_from_pressure_and_mixing_ratio;
use crate::ecape::validate_len;
use crate::error::CalcError;

const EARTH_RADIUS_KM: f64 = 6371.0;
const KTS_TO_MS: f64 = 0.514_444;
const EPSILON: f64 = 0.622;
const MAX_MSLP_INCREMENT_HPA: f64 = 40.0;

#[derive(Debug, Clone, Copy)]
pub struct SurfaceMesoBackground<'a> {
    pub grid: GridShape,
    pub lat_deg: &'a [f64],
    pub lon_deg: &'a [f64],
    pub psfc_pa: &'a [f64],
    pub t2_k: &'a [f64],
    pub q2_kgkg: &'a [f64],
    pub u10_ms: &'a [f64],
    pub v10_ms: &'a [f64],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MesoObservation {
    pub station_id: String,
    pub source: String,
    pub timestamp: Option<String>,
    pub latitude_deg: f64,
    pub longitude_deg: f64,
    pub temperature_c: Option<f64>,
    pub dewpoint_c: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mean_sea_level_pressure_hpa: Option<f64>,
    pub wind_direction_deg: Option<f64>,
    pub wind_speed_ms: Option<f64>,
    pub quality_weight: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_quality_class: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub representativeness_class: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub correction_role: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observation_age_minutes: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub time_weight: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature_error_c: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dewpoint_error_c: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wind_error_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mean_sea_level_pressure_error_hpa: Option<f64>,
}

impl MesoObservation {
    pub fn new(station_id: impl Into<String>, latitude_deg: f64, longitude_deg: f64) -> Self {
        Self {
            station_id: station_id.into(),
            source: String::new(),
            timestamp: None,
            latitude_deg,
            longitude_deg,
            temperature_c: None,
            dewpoint_c: None,
            mean_sea_level_pressure_hpa: None,
            wind_direction_deg: None,
            wind_speed_ms: None,
            quality_weight: 1.0,
            source_quality_class: None,
            representativeness_class: None,
            correction_role: None,
            observation_age_minutes: None,
            time_weight: None,
            temperature_error_c: None,
            dewpoint_error_c: None,
            wind_error_ms: None,
            mean_sea_level_pressure_error_hpa: None,
        }
    }

    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = source.into();
        self
    }

    pub fn with_timestamp(mut self, timestamp: impl Into<String>) -> Self {
        self.timestamp = Some(timestamp.into());
        self
    }

    pub fn with_temperature_c(mut self, value: f64) -> Self {
        self.temperature_c = Some(value);
        self
    }

    pub fn with_dewpoint_c(mut self, value: f64) -> Self {
        self.dewpoint_c = Some(value);
        self
    }

    pub fn with_mean_sea_level_pressure_hpa(mut self, value: f64) -> Self {
        self.mean_sea_level_pressure_hpa = Some(value);
        self
    }

    pub fn with_wind(mut self, direction_deg: f64, speed_ms: f64) -> Self {
        self.wind_direction_deg = Some(direction_deg);
        self.wind_speed_ms = Some(speed_ms);
        self
    }

    pub fn with_wind_kts(self, direction_deg: f64, speed_kts: f64) -> Self {
        self.with_wind(direction_deg, speed_kts * KTS_TO_MS)
    }

    pub fn with_quality_weight(mut self, quality_weight: f64) -> Self {
        self.quality_weight = quality_weight;
        self
    }

    pub fn with_temperature_error_c(mut self, value: f64) -> Self {
        self.temperature_error_c = Some(value);
        self
    }

    pub fn with_dewpoint_error_c(mut self, value: f64) -> Self {
        self.dewpoint_error_c = Some(value);
        self
    }

    pub fn with_wind_error_ms(mut self, value: f64) -> Self {
        self.wind_error_ms = Some(value);
        self
    }

    pub fn with_mean_sea_level_pressure_error_hpa(mut self, value: f64) -> Self {
        self.mean_sea_level_pressure_error_hpa = Some(value);
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MesoanalysisMethod {
    Barnes,
    OptimalInterpolation,
}

impl Default for MesoanalysisMethod {
    fn default() -> Self {
        Self::Barnes
    }
}

impl FromStr for MesoanalysisMethod {
    type Err = String;

    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        match raw.trim().to_ascii_lowercase().replace('-', "_").as_str() {
            "barnes" | "objective_analysis" => Ok(Self::Barnes),
            "oi"
            | "optimal_interpolation"
            | "optimal_interpolation_style"
            | "kriging"
            | "full_matrix_oi"
            | "full_matrix_optimal_interpolation" => Ok(Self::OptimalInterpolation),
            other => Err(format!(
                "unknown mesoanalysis method '{other}'; expected barnes or optimal_interpolation"
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MesoanalysisCovarianceKernel {
    Gaussian,
    Exponential,
    Matern32,
}

impl Default for MesoanalysisCovarianceKernel {
    fn default() -> Self {
        Self::Gaussian
    }
}

impl FromStr for MesoanalysisCovarianceKernel {
    type Err = String;

    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        match raw.trim().to_ascii_lowercase().replace('-', "_").as_str() {
            "gaussian" | "squared_exponential" => Ok(Self::Gaussian),
            "exponential" => Ok(Self::Exponential),
            "matern32" | "matern_32" | "matern_3_2" => Ok(Self::Matern32),
            other => Err(format!(
                "unknown covariance kernel '{other}'; expected gaussian, exponential, or matern32"
            )),
        }
    }
}

impl MesoanalysisCovarianceKernel {
    fn correlation(self, normalized_distance: f64) -> f64 {
        if !(normalized_distance.is_finite() && normalized_distance >= 0.0) {
            return 0.0;
        }
        match self {
            Self::Gaussian => (-0.5 * normalized_distance * normalized_distance).exp(),
            Self::Exponential => (-normalized_distance).exp(),
            Self::Matern32 => {
                let scaled = 3.0_f64.sqrt() * normalized_distance;
                (1.0 + scaled) * (-scaled).exp()
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MesoanalysisConfig {
    pub method: MesoanalysisMethod,
    pub barnes_radius_km: f64,
    pub barnes_kappa_km2: f64,
    pub barnes_passes: u8,
    pub barnes_second_pass_gamma: f64,
    pub min_neighbors: usize,
    pub background_search_radius_km: f64,
    pub max_temperature_increment_c: f64,
    pub max_dewpoint_increment_c: f64,
    pub max_wind_increment_ms: f64,
    pub oi_length_scale_km: f64,
    pub oi_background_error_temperature_c: f64,
    pub oi_background_error_dewpoint_c: f64,
    pub oi_background_error_wind_ms: f64,
    pub oi_background_error_mslp_hpa: f64,
    pub oi_covariance_kernel: MesoanalysisCovarianceKernel,
    pub oi_observation_error_temperature_c: f64,
    pub oi_observation_error_dewpoint_c: f64,
    pub oi_observation_error_wind_ms: f64,
    pub oi_observation_error_mslp_hpa: f64,
    pub oi_flow_anisotropy_ratio: f64,
    pub oi_terrain_pressure_scale_hpa: f64,
    pub oi_max_observations_per_grid_cell: usize,
    pub oi_min_target_correlation: f64,
    pub oi_matrix_jitter_fraction: f64,
    pub oi_gross_error_sigma: f64,
    pub oi_gross_error_buddy_radius_km: f64,
    pub oi_gross_error_buddy_min_neighbors: usize,
    pub oi_gross_error_buddy_agreement_sigma: f64,
    pub oi_max_local_innovation_factor: f64,
}

impl Default for MesoanalysisConfig {
    fn default() -> Self {
        Self {
            method: MesoanalysisMethod::Barnes,
            barnes_radius_km: 75.0,
            barnes_kappa_km2: 2500.0,
            barnes_passes: 2,
            barnes_second_pass_gamma: 0.3,
            min_neighbors: 1,
            background_search_radius_km: 35.0,
            max_temperature_increment_c: 20.0,
            max_dewpoint_increment_c: 25.0,
            max_wind_increment_ms: 30.0,
            oi_length_scale_km: 15.0,
            oi_background_error_temperature_c: 1.0,
            oi_background_error_dewpoint_c: 1.2,
            oi_background_error_wind_ms: 1.5,
            oi_background_error_mslp_hpa: 3.0,
            oi_covariance_kernel: MesoanalysisCovarianceKernel::Exponential,
            oi_observation_error_temperature_c: 1.2,
            oi_observation_error_dewpoint_c: 1.6,
            oi_observation_error_wind_ms: 2.0,
            oi_observation_error_mslp_hpa: 1.5,
            oi_flow_anisotropy_ratio: 2.5,
            oi_terrain_pressure_scale_hpa: 75.0,
            oi_max_observations_per_grid_cell: 32,
            oi_min_target_correlation: 1.0e-4,
            oi_matrix_jitter_fraction: 1.0e-8,
            oi_gross_error_sigma: 6.0,
            oi_gross_error_buddy_radius_km: 25.0,
            oi_gross_error_buddy_min_neighbors: 1,
            oi_gross_error_buddy_agreement_sigma: 2.5,
            oi_max_local_innovation_factor: 1.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MesoanalysisVariableDiagnostics {
    pub variable: String,
    pub candidate_observations: usize,
    pub accepted_observations: usize,
    pub rejected_observations: usize,
    pub covered_grid_cells: usize,
    pub solver_failed_grid_cells: usize,
    pub truncated_neighbor_grid_cells: usize,
    #[serde(default)]
    pub gross_error_rescued_observations: usize,
    pub mean_neighbor_count: Option<f64>,
    pub max_neighbor_count: u16,
    pub mean_confidence: Option<f64>,
    pub max_confidence: Option<f64>,
    pub mean_abs_increment: Option<f64>,
    pub max_abs_increment: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MesoanalysisFields {
    pub temperature_2m_c: Vec<f64>,
    pub dewpoint_2m_c: Vec<f64>,
    pub q2_kgkg: Vec<f64>,
    pub u10_ms: Vec<f64>,
    pub v10_ms: Vec<f64>,
    pub mean_sea_level_pressure_hpa: Option<Vec<f64>>,
    pub temperature_increment_c: Vec<f64>,
    pub dewpoint_increment_c: Vec<f64>,
    pub u10_increment_ms: Vec<f64>,
    pub v10_increment_ms: Vec<f64>,
    pub mean_sea_level_pressure_increment_hpa: Option<Vec<f64>>,
    pub neighbor_count: Vec<u16>,
    pub temperature_confidence: Vec<f64>,
    pub dewpoint_confidence: Vec<f64>,
    pub u10_confidence: Vec<f64>,
    pub v10_confidence: Vec<f64>,
    pub mean_sea_level_pressure_confidence: Option<Vec<f64>>,
    pub diagnostics: Vec<MesoanalysisVariableDiagnostics>,
}

#[derive(Debug, Clone, Copy)]
enum Variable {
    Temperature,
    Dewpoint,
    MeanSeaLevelPressure,
    UWind,
    VWind,
}

impl Variable {
    fn label(self) -> &'static str {
        match self {
            Self::Temperature => "temperature_2m_c",
            Self::Dewpoint => "dewpoint_2m_c",
            Self::MeanSeaLevelPressure => "mean_sea_level_pressure_hpa",
            Self::UWind => "u10_ms",
            Self::VWind => "v10_ms",
        }
    }

    fn max_increment(self, config: MesoanalysisConfig) -> f64 {
        match self {
            Self::Temperature => config.max_temperature_increment_c,
            Self::Dewpoint => config.max_dewpoint_increment_c,
            Self::MeanSeaLevelPressure => MAX_MSLP_INCREMENT_HPA,
            Self::UWind | Self::VWind => config.max_wind_increment_ms,
        }
    }

    fn oi_background_error(self, config: MesoanalysisConfig) -> f64 {
        match self {
            Self::Temperature => config.oi_background_error_temperature_c,
            Self::Dewpoint => config.oi_background_error_dewpoint_c,
            Self::MeanSeaLevelPressure => config.oi_background_error_mslp_hpa,
            Self::UWind | Self::VWind => config.oi_background_error_wind_ms,
        }
    }

    fn oi_observation_error(self, config: MesoanalysisConfig) -> f64 {
        match self {
            Self::Temperature => config.oi_observation_error_temperature_c,
            Self::Dewpoint => config.oi_observation_error_dewpoint_c,
            Self::MeanSeaLevelPressure => config.oi_observation_error_mslp_hpa,
            Self::UWind | Self::VWind => config.oi_observation_error_wind_ms,
        }
    }
}

#[derive(Debug, Clone)]
struct IncrementObservation {
    lat_deg: f64,
    lon_deg: f64,
    increment: f64,
    weight: f64,
    observation_error: f64,
    background_index: usize,
}

#[derive(Debug, Clone)]
struct IncrementSet {
    variable: Variable,
    candidate_observations: usize,
    rejected_observations: usize,
    gross_error_rescued_observations: usize,
    observations: Vec<IncrementObservation>,
}

#[derive(Debug, Clone)]
struct BarnesOutput {
    increments: Vec<f64>,
    neighbor_count: Vec<u16>,
    confidence: Vec<f64>,
    covered_grid_cells: usize,
    solver_failed_grid_cells: usize,
    truncated_neighbor_grid_cells: usize,
}

#[derive(Debug, Clone, Copy)]
struct LocalOiObservation {
    observation_index: usize,
    target_covariance: f64,
}

#[derive(Debug, Default)]
struct OiWorkspace {
    local: Vec<LocalOiObservation>,
    matrix: Vec<f64>,
    rhs: Vec<f64>,
    target_covariances: Vec<f64>,
    factor: Vec<f64>,
    forward_solution: Vec<f64>,
    alpha: Vec<f64>,
    beta: Vec<f64>,
}

#[derive(Debug, Clone, Copy)]
struct AnalysisContext<'a> {
    grid_lat_deg: &'a [f64],
    grid_lon_deg: &'a [f64],
    grid_psfc_hpa: &'a [f64],
    grid_u10_ms: &'a [f64],
    grid_v10_ms: &'a [f64],
    grid_index: &'a SpatialBins,
    config: MesoanalysisConfig,
}

impl<'a> SurfaceMesoBackground<'a> {
    pub fn compute_with_mean_sea_level_pressure_hpa(
        self,
        mean_sea_level_pressure_hpa: &[f64],
        observations: &[MesoObservation],
        config: MesoanalysisConfig,
    ) -> Result<MesoanalysisFields, CalcError> {
        compute_surface_mesoanalysis_inner(
            self,
            Some(mean_sea_level_pressure_hpa),
            observations,
            config,
        )
    }
}

pub fn compute_surface_mesoanalysis(
    background: SurfaceMesoBackground<'_>,
    observations: &[MesoObservation],
    config: MesoanalysisConfig,
) -> Result<MesoanalysisFields, CalcError> {
    compute_surface_mesoanalysis_inner(background, None, observations, config)
}

fn compute_surface_mesoanalysis_inner(
    background: SurfaceMesoBackground<'_>,
    mean_sea_level_pressure_hpa: Option<&[f64]>,
    observations: &[MesoObservation],
    config: MesoanalysisConfig,
) -> Result<MesoanalysisFields, CalcError> {
    validate_background(background)?;
    if let Some(mslp) = mean_sea_level_pressure_hpa {
        validate_len(
            "mean_sea_level_pressure_hpa",
            mslp.len(),
            background.grid.len(),
        )?;
    }
    validate_config(config)?;

    let len = background.grid.len();
    let pressure_hpa: Vec<f64> = background
        .psfc_pa
        .iter()
        .map(|value| *value / 100.0)
        .collect();
    let background_temperature_c: Vec<f64> = background
        .t2_k
        .iter()
        .map(|value| *value - 273.15)
        .collect();
    let background_dewpoint_c =
        compute_dewpoint_from_pressure_and_mixing_ratio(&pressure_hpa, background.q2_kgkg)?;

    let grid_index = SpatialBins::new_grid(
        background.lat_deg,
        background.lon_deg,
        (config.background_search_radius_km / 111.0).clamp(0.10, 1.0),
    );

    let temperature_set = build_increment_set(
        Variable::Temperature,
        observations,
        background.lat_deg,
        background.lon_deg,
        &background_temperature_c,
        &grid_index,
        config,
    );
    let dewpoint_set = build_increment_set(
        Variable::Dewpoint,
        observations,
        background.lat_deg,
        background.lon_deg,
        &background_dewpoint_c,
        &grid_index,
        config,
    );
    let u_set = build_increment_set(
        Variable::UWind,
        observations,
        background.lat_deg,
        background.lon_deg,
        background.u10_ms,
        &grid_index,
        config,
    );
    let v_set = build_increment_set(
        Variable::VWind,
        observations,
        background.lat_deg,
        background.lon_deg,
        background.v10_ms,
        &grid_index,
        config,
    );
    let mean_sea_level_pressure_set = mean_sea_level_pressure_hpa.map(|background_mslp_hpa| {
        build_increment_set(
            Variable::MeanSeaLevelPressure,
            observations,
            background.lat_deg,
            background.lon_deg,
            background_mslp_hpa,
            &grid_index,
            config,
        )
    });

    let analysis_context = AnalysisContext {
        grid_lat_deg: background.lat_deg,
        grid_lon_deg: background.lon_deg,
        grid_psfc_hpa: &pressure_hpa,
        grid_u10_ms: background.u10_ms,
        grid_v10_ms: background.v10_ms,
        grid_index: &grid_index,
        config,
    };
    let temperature_analysis = increment_grid(&analysis_context, &temperature_set);
    let dewpoint_analysis = increment_grid(&analysis_context, &dewpoint_set);
    let u_analysis = increment_grid(&analysis_context, &u_set);
    let v_analysis = increment_grid(&analysis_context, &v_set);
    let mean_sea_level_pressure_analysis = mean_sea_level_pressure_set
        .as_ref()
        .map(|increment_set| increment_grid(&analysis_context, increment_set));

    let temperature_2m_c = apply_increment(&background_temperature_c, &temperature_analysis);
    let mut dewpoint_2m_c = apply_increment(&background_dewpoint_c, &dewpoint_analysis);
    for (dewpoint, temperature) in dewpoint_2m_c.iter_mut().zip(temperature_2m_c.iter()) {
        if dewpoint.is_finite() && temperature.is_finite() && *dewpoint > *temperature {
            *dewpoint = *temperature;
        }
    }

    let u10_ms = apply_increment(background.u10_ms, &u_analysis);
    let v10_ms = apply_increment(background.v10_ms, &v_analysis);
    let q2_kgkg: Vec<f64> = pressure_hpa
        .iter()
        .zip(dewpoint_2m_c.iter())
        .map(|(&pressure_hpa, &dewpoint_c)| mixing_ratio_from_dewpoint_c(pressure_hpa, dewpoint_c))
        .collect();
    let mean_sea_level_pressure_analysis_hpa = mean_sea_level_pressure_hpa
        .zip(mean_sea_level_pressure_analysis.as_ref())
        .map(|(background_mslp_hpa, analysis)| apply_increment(background_mslp_hpa, analysis));

    let mut neighbor_count = vec![0u16; len];
    for source in [
        &temperature_analysis,
        &dewpoint_analysis,
        &u_analysis,
        &v_analysis,
    ] {
        for (target, count) in neighbor_count.iter_mut().zip(source.neighbor_count.iter()) {
            *target = (*target).max(*count);
        }
    }
    if let Some(source) = &mean_sea_level_pressure_analysis {
        for (target, count) in neighbor_count.iter_mut().zip(source.neighbor_count.iter()) {
            *target = (*target).max(*count);
        }
    }

    let mut diagnostics = vec![
        variable_diagnostics(&temperature_set, &temperature_analysis),
        variable_diagnostics(&dewpoint_set, &dewpoint_analysis),
        variable_diagnostics(&u_set, &u_analysis),
        variable_diagnostics(&v_set, &v_analysis),
    ];
    if let (Some(increment_set), Some(analysis)) = (
        mean_sea_level_pressure_set.as_ref(),
        mean_sea_level_pressure_analysis.as_ref(),
    ) {
        diagnostics.push(variable_diagnostics(increment_set, analysis));
    }
    let (mean_sea_level_pressure_increment_hpa, mean_sea_level_pressure_confidence) =
        if let Some(analysis) = mean_sea_level_pressure_analysis {
            (Some(analysis.increments), Some(analysis.confidence))
        } else {
            (None, None)
        };

    Ok(MesoanalysisFields {
        temperature_2m_c,
        dewpoint_2m_c,
        q2_kgkg,
        u10_ms,
        v10_ms,
        mean_sea_level_pressure_hpa: mean_sea_level_pressure_analysis_hpa,
        temperature_increment_c: temperature_analysis.increments,
        dewpoint_increment_c: dewpoint_analysis.increments,
        u10_increment_ms: u_analysis.increments,
        v10_increment_ms: v_analysis.increments,
        mean_sea_level_pressure_increment_hpa,
        neighbor_count,
        temperature_confidence: temperature_analysis.confidence,
        dewpoint_confidence: dewpoint_analysis.confidence,
        u10_confidence: u_analysis.confidence,
        v10_confidence: v_analysis.confidence,
        mean_sea_level_pressure_confidence,
        diagnostics,
    })
}

fn validate_background(background: SurfaceMesoBackground<'_>) -> Result<(), CalcError> {
    let len = background.grid.len();
    validate_len("lat_deg", background.lat_deg.len(), len)?;
    validate_len("lon_deg", background.lon_deg.len(), len)?;
    validate_len("psfc_pa", background.psfc_pa.len(), len)?;
    validate_len("t2_k", background.t2_k.len(), len)?;
    validate_len("q2_kgkg", background.q2_kgkg.len(), len)?;
    validate_len("u10_ms", background.u10_ms.len(), len)?;
    validate_len("v10_ms", background.v10_ms.len(), len)?;
    Ok(())
}

fn validate_config(config: MesoanalysisConfig) -> Result<(), CalcError> {
    validate_positive("barnes_radius_km", config.barnes_radius_km)?;
    validate_positive("barnes_kappa_km2", config.barnes_kappa_km2)?;
    if !(1..=2).contains(&config.barnes_passes) {
        return Err(CalcError::InvalidConfig {
            field: "barnes_passes",
            reason: "must be 1 or 2",
        });
    }
    validate_positive("barnes_second_pass_gamma", config.barnes_second_pass_gamma)?;
    if config.barnes_second_pass_gamma > 1.0 {
        return Err(CalcError::InvalidConfig {
            field: "barnes_second_pass_gamma",
            reason: "must be less than or equal to 1",
        });
    }
    validate_positive(
        "background_search_radius_km",
        config.background_search_radius_km,
    )?;
    validate_positive(
        "max_temperature_increment_c",
        config.max_temperature_increment_c,
    )?;
    validate_positive("max_dewpoint_increment_c", config.max_dewpoint_increment_c)?;
    validate_positive("max_wind_increment_ms", config.max_wind_increment_ms)?;
    validate_positive("oi_length_scale_km", config.oi_length_scale_km)?;
    validate_positive(
        "oi_background_error_temperature_c",
        config.oi_background_error_temperature_c,
    )?;
    validate_positive(
        "oi_background_error_dewpoint_c",
        config.oi_background_error_dewpoint_c,
    )?;
    validate_positive(
        "oi_background_error_wind_ms",
        config.oi_background_error_wind_ms,
    )?;
    validate_positive(
        "oi_background_error_mslp_hpa",
        config.oi_background_error_mslp_hpa,
    )?;
    validate_positive(
        "oi_observation_error_temperature_c",
        config.oi_observation_error_temperature_c,
    )?;
    validate_positive(
        "oi_observation_error_dewpoint_c",
        config.oi_observation_error_dewpoint_c,
    )?;
    validate_positive(
        "oi_observation_error_wind_ms",
        config.oi_observation_error_wind_ms,
    )?;
    validate_positive(
        "oi_observation_error_mslp_hpa",
        config.oi_observation_error_mslp_hpa,
    )?;
    validate_positive("oi_flow_anisotropy_ratio", config.oi_flow_anisotropy_ratio)?;
    validate_positive(
        "oi_terrain_pressure_scale_hpa",
        config.oi_terrain_pressure_scale_hpa,
    )?;
    if config.oi_max_observations_per_grid_cell == 0 {
        return Err(CalcError::InvalidConfig {
            field: "oi_max_observations_per_grid_cell",
            reason: "must be at least 1",
        });
    }
    if !(config.oi_min_target_correlation.is_finite()
        && (0.0..1.0).contains(&config.oi_min_target_correlation))
    {
        return Err(CalcError::InvalidConfig {
            field: "oi_min_target_correlation",
            reason: "must be finite and in the range [0, 1)",
        });
    }
    validate_positive(
        "oi_matrix_jitter_fraction",
        config.oi_matrix_jitter_fraction,
    )?;
    validate_positive("oi_gross_error_sigma", config.oi_gross_error_sigma)?;
    validate_positive(
        "oi_gross_error_buddy_radius_km",
        config.oi_gross_error_buddy_radius_km,
    )?;
    validate_positive(
        "oi_gross_error_buddy_agreement_sigma",
        config.oi_gross_error_buddy_agreement_sigma,
    )?;
    validate_positive(
        "oi_max_local_innovation_factor",
        config.oi_max_local_innovation_factor,
    )?;
    if config.min_neighbors == 0 {
        return Err(CalcError::InvalidConfig {
            field: "min_neighbors",
            reason: "must be at least 1",
        });
    }
    Ok(())
}

fn validate_positive(field: &'static str, value: f64) -> Result<(), CalcError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(CalcError::InvalidConfig {
            field,
            reason: "must be finite and positive",
        })
    }
}

fn build_increment_set(
    variable: Variable,
    observations: &[MesoObservation],
    grid_lat_deg: &[f64],
    grid_lon_deg: &[f64],
    background_values: &[f64],
    grid_index: &SpatialBins,
    config: MesoanalysisConfig,
) -> IncrementSet {
    let mut candidate_observations = 0usize;
    let mut rejected_observations = 0usize;
    let mut gross_error_rescued_observations = 0usize;
    let mut increments = Vec::new();

    for observation in observations {
        let Some(value) = observation_value(variable, observation) else {
            continue;
        };
        candidate_observations += 1;
        if !observation_has_valid_location(observation) || !value_is_plausible(variable, value) {
            rejected_observations += 1;
            continue;
        }
        let Some(model_index) = nearest_grid_index(
            grid_lat_deg,
            grid_lon_deg,
            grid_index,
            observation.latitude_deg,
            observation.longitude_deg,
            config.background_search_radius_km,
        ) else {
            rejected_observations += 1;
            continue;
        };
        let background = background_values[model_index];
        if !background.is_finite() {
            rejected_observations += 1;
            continue;
        }
        let increment = value - background;
        if !increment.is_finite() || increment.abs() > variable.max_increment(config) {
            rejected_observations += 1;
            continue;
        }
        let observation_error = observation_error_for_variable(variable, observation, config);
        increments.push(IncrementObservation {
            lat_deg: observation.latitude_deg,
            lon_deg: observation.longitude_deg,
            increment,
            weight: observation.quality_weight.clamp(0.05, 10.0),
            observation_error,
            background_index: model_index,
        });
    }

    let observations = if config.method == MesoanalysisMethod::OptimalInterpolation {
        let mut accepted = Vec::with_capacity(increments.len());
        for (index, observation) in increments.iter().enumerate() {
            if innovation_is_gross_error(
                variable,
                config,
                observation.observation_error,
                observation.increment,
            ) {
                if gross_error_has_buddy_support(variable, config, index, &increments) {
                    gross_error_rescued_observations += 1;
                    accepted.push(observation.clone());
                } else {
                    rejected_observations += 1;
                }
            } else {
                accepted.push(observation.clone());
            }
        }
        accepted
    } else {
        increments
    };

    IncrementSet {
        variable,
        candidate_observations,
        rejected_observations,
        gross_error_rescued_observations,
        observations,
    }
}

fn observation_value(variable: Variable, observation: &MesoObservation) -> Option<f64> {
    match variable {
        Variable::Temperature => observation.temperature_c,
        Variable::Dewpoint => observation.dewpoint_c,
        Variable::MeanSeaLevelPressure => observation.mean_sea_level_pressure_hpa,
        Variable::UWind => observation_wind_components(observation).map(|(u, _)| u),
        Variable::VWind => observation_wind_components(observation).map(|(_, v)| v),
    }
}

fn innovation_is_gross_error(
    variable: Variable,
    config: MesoanalysisConfig,
    observation_error: f64,
    increment: f64,
) -> bool {
    let background_error = variable.oi_background_error(config).max(1.0e-6);
    let combined_error = (background_error * background_error
        + observation_error * observation_error)
        .sqrt()
        .max(1.0e-6);
    increment.abs() > config.oi_gross_error_sigma * combined_error
}

fn gross_error_has_buddy_support(
    variable: Variable,
    config: MesoanalysisConfig,
    candidate_index: usize,
    observations: &[IncrementObservation],
) -> bool {
    if config.oi_gross_error_buddy_min_neighbors == 0 {
        return false;
    }
    let Some(candidate) = observations.get(candidate_index) else {
        return false;
    };
    if !candidate.increment.is_finite() || candidate.increment.abs() <= 1.0e-9 {
        return false;
    }
    let background_error = variable.oi_background_error(config).max(1.0e-6);
    let candidate_combined_error = (background_error * background_error
        + candidate.observation_error * candidate.observation_error)
        .sqrt()
        .max(1.0e-6);
    let mut supporting_neighbors = 0usize;
    for (index, neighbor) in observations.iter().enumerate() {
        if index == candidate_index || !neighbor.increment.is_finite() {
            continue;
        }
        if candidate.increment.signum() != neighbor.increment.signum() {
            continue;
        }
        let distance = haversine_km(
            candidate.lat_deg,
            candidate.lon_deg,
            neighbor.lat_deg,
            neighbor.lon_deg,
        );
        if distance > config.oi_gross_error_buddy_radius_km {
            continue;
        }
        let neighbor_combined_error = (background_error * background_error
            + neighbor.observation_error * neighbor.observation_error)
            .sqrt()
            .max(1.0e-6);
        let agreement_limit = config.oi_gross_error_buddy_agreement_sigma
            * candidate_combined_error.max(neighbor_combined_error);
        if (candidate.increment - neighbor.increment).abs() <= agreement_limit {
            supporting_neighbors += 1;
            if supporting_neighbors >= config.oi_gross_error_buddy_min_neighbors {
                return true;
            }
        }
    }
    false
}

fn observation_error_for_variable(
    variable: Variable,
    observation: &MesoObservation,
    config: MesoanalysisConfig,
) -> f64 {
    let specific_error = match variable {
        Variable::Temperature => observation.temperature_error_c,
        Variable::Dewpoint => observation.dewpoint_error_c,
        Variable::MeanSeaLevelPressure => observation.mean_sea_level_pressure_error_hpa,
        Variable::UWind | Variable::VWind => observation.wind_error_ms,
    };
    if let Some(error) = specific_error.filter(|value| value.is_finite() && *value > 0.0) {
        error
    } else {
        variable.oi_observation_error(config).max(1.0e-6)
            / observation.quality_weight.clamp(0.05, 10.0).sqrt()
    }
}

fn observation_wind_components(observation: &MesoObservation) -> Option<(f64, f64)> {
    let direction_deg = observation.wind_direction_deg?;
    let speed_ms = observation.wind_speed_ms?;
    if !(direction_deg.is_finite() && speed_ms.is_finite() && speed_ms >= 0.0) {
        return None;
    }
    let direction_rad = direction_deg.to_radians();
    Some((
        -speed_ms * direction_rad.sin(),
        -speed_ms * direction_rad.cos(),
    ))
}

fn observation_has_valid_location(observation: &MesoObservation) -> bool {
    observation.latitude_deg.is_finite()
        && observation.longitude_deg.is_finite()
        && (-90.0..=90.0).contains(&observation.latitude_deg)
        && (-180.0..=180.0).contains(&observation.longitude_deg)
}

fn value_is_plausible(variable: Variable, value: f64) -> bool {
    if !value.is_finite() {
        return false;
    }
    match variable {
        Variable::Temperature => (-90.0..=65.0).contains(&value),
        Variable::Dewpoint => (-100.0..=45.0).contains(&value),
        Variable::MeanSeaLevelPressure => (850.0..=1100.0).contains(&value),
        Variable::UWind | Variable::VWind => value.abs() <= 90.0,
    }
}

fn nearest_grid_index(
    grid_lat_deg: &[f64],
    grid_lon_deg: &[f64],
    grid_index: &SpatialBins,
    lat_deg: f64,
    lon_deg: f64,
    radius_km: f64,
) -> Option<usize> {
    let mut best: Option<(usize, f64)> = None;
    grid_index.for_each_candidate(lat_deg, lon_deg, radius_km, |index| {
        let distance = haversine_km(lat_deg, lon_deg, grid_lat_deg[index], grid_lon_deg[index]);
        if distance <= radius_km && best.map(|(_, best)| distance < best).unwrap_or(true) {
            best = Some((index, distance));
        }
    });
    best.map(|(index, _)| index)
}

fn increment_grid(context: &AnalysisContext<'_>, increments: &IncrementSet) -> BarnesOutput {
    match context.config.method {
        MesoanalysisMethod::Barnes => barnes_increment_grid(
            context.grid_lat_deg,
            context.grid_lon_deg,
            increments,
            context.grid_index,
            context.config,
        ),
        MesoanalysisMethod::OptimalInterpolation => oi_increment_grid(context, increments),
    }
}

fn barnes_increment_grid(
    grid_lat_deg: &[f64],
    grid_lon_deg: &[f64],
    increments: &IncrementSet,
    grid_index: &SpatialBins,
    config: MesoanalysisConfig,
) -> BarnesOutput {
    let len = grid_lat_deg.len();
    if increments.observations.is_empty() {
        return BarnesOutput {
            increments: vec![f64::NAN; len],
            neighbor_count: vec![0u16; len],
            confidence: vec![0.0; len],
            covered_grid_cells: 0,
            solver_failed_grid_cells: 0,
            truncated_neighbor_grid_cells: 0,
        };
    }

    let first_pass = barnes_single_pass_grid(
        grid_lat_deg,
        grid_lon_deg,
        &increments.observations,
        config,
        config.barnes_kappa_km2,
    );
    if config.barnes_passes == 1 {
        return first_pass;
    }

    let residuals = residual_increment_observations(
        grid_lat_deg,
        grid_lon_deg,
        grid_index,
        increments,
        &first_pass,
        config,
    );
    if residuals.is_empty() {
        return first_pass;
    }
    let second_pass = barnes_single_pass_grid(
        grid_lat_deg,
        grid_lon_deg,
        &residuals,
        config,
        config.barnes_kappa_km2 * config.barnes_second_pass_gamma,
    );
    combine_barnes_outputs(
        first_pass,
        second_pass,
        increments.variable.max_increment(config),
    )
}

fn barnes_single_pass_grid(
    grid_lat_deg: &[f64],
    grid_lon_deg: &[f64],
    observations: &[IncrementObservation],
    config: MesoanalysisConfig,
    kappa_km2: f64,
) -> BarnesOutput {
    let len = grid_lat_deg.len();
    if observations.is_empty() {
        return BarnesOutput {
            increments: vec![f64::NAN; len],
            neighbor_count: vec![0u16; len],
            confidence: vec![0.0; len],
            covered_grid_cells: 0,
            solver_failed_grid_cells: 0,
            truncated_neighbor_grid_cells: 0,
        };
    }

    let obs_index = SpatialBins::new_observations(
        observations,
        (config.barnes_radius_km / 111.0).clamp(0.10, 2.0),
    );
    let paired: Vec<(f64, u16, f64)> = (0..len)
        .into_par_iter()
        .map(|idx| {
            let lat = grid_lat_deg[idx];
            let lon = grid_lon_deg[idx];
            let mut weighted_sum = 0.0;
            let mut weight_sum = 0.0;
            let mut count = 0u16;
            obs_index.for_each_candidate(lat, lon, config.barnes_radius_km, |obs_idx| {
                let observation = &observations[obs_idx];
                let distance = haversine_km(lat, lon, observation.lat_deg, observation.lon_deg);
                if distance > config.barnes_radius_km {
                    return;
                }
                let weight = (-(distance * distance) / kappa_km2).exp() * observation.weight;
                if weight.is_finite() && weight > 0.0 {
                    weighted_sum += weight * observation.increment;
                    weight_sum += weight;
                    count = count.saturating_add(1);
                }
            });
            if count as usize >= config.min_neighbors && weight_sum > 0.0 {
                let confidence = (weight_sum / (weight_sum + 1.0)).clamp(0.0, 1.0);
                (weighted_sum / weight_sum, count, confidence)
            } else {
                (f64::NAN, count, 0.0)
            }
        })
        .collect();

    let mut covered_grid_cells = 0usize;
    let mut output = Vec::with_capacity(len);
    let mut neighbor_count = Vec::with_capacity(len);
    let mut confidence = Vec::with_capacity(len);
    for (increment, count, cell_confidence) in paired {
        if increment.is_finite() {
            covered_grid_cells += 1;
        }
        output.push(increment);
        neighbor_count.push(count);
        confidence.push(cell_confidence);
    }

    BarnesOutput {
        increments: output,
        neighbor_count,
        confidence,
        covered_grid_cells,
        solver_failed_grid_cells: 0,
        truncated_neighbor_grid_cells: 0,
    }
}

fn oi_increment_grid(context: &AnalysisContext<'_>, increments: &IncrementSet) -> BarnesOutput {
    let len = context.grid_lat_deg.len();
    if increments.observations.is_empty() {
        return BarnesOutput {
            increments: vec![f64::NAN; len],
            neighbor_count: vec![0u16; len],
            confidence: vec![0.0; len],
            covered_grid_cells: 0,
            solver_failed_grid_cells: 0,
            truncated_neighbor_grid_cells: 0,
        };
    }

    let obs_index = SpatialBins::new_observations(
        &increments.observations,
        (context.config.barnes_radius_km / 111.0).clamp(0.10, 2.0),
    );
    let background_error = increments
        .variable
        .oi_background_error(context.config)
        .max(1.0e-6);
    let background_variance = background_error * background_error;
    let max_local_observations = context.config.oi_max_observations_per_grid_cell;
    let paired: Vec<(f64, u16, f64, bool, bool)> = (0..len)
        .into_par_iter()
        .map_init(OiWorkspace::default, |workspace, idx| {
            let lat = context.grid_lat_deg[idx];
            let lon = context.grid_lon_deg[idx];
            workspace.local.clear();
            obs_index.for_each_candidate(lat, lon, context.config.barnes_radius_km, |obs_idx| {
                let observation = &increments.observations[obs_idx];
                let distance = haversine_km(lat, lon, observation.lat_deg, observation.lon_deg);
                if distance > context.config.barnes_radius_km {
                    return;
                }
                let target_correlation =
                    oi_background_covariance(context, idx, observation, distance);
                if target_correlation < context.config.oi_min_target_correlation {
                    return;
                }
                let target_covariance = target_correlation * background_variance;
                if !(target_covariance.is_finite() && target_covariance > 0.0) {
                    return;
                }
                workspace.local.push(LocalOiObservation {
                    observation_index: obs_idx,
                    target_covariance,
                });
            });
            let truncated = workspace.local.len() > max_local_observations;
            if truncated {
                let retained = max_local_observations - 1;
                workspace
                    .local
                    .select_nth_unstable_by(retained, compare_local_oi_observation_priority);
                workspace.local.truncate(max_local_observations);
                workspace
                    .local
                    .sort_unstable_by(compare_local_oi_observation_priority);
            }
            let count = workspace.local.len().min(u16::MAX as usize) as u16;
            if workspace.local.len() < context.config.min_neighbors {
                return (f64::NAN, count, 0.0, truncated, false);
            }
            if let Some((increment, confidence)) =
                solve_local_oi_increment(context, increments, background_variance, workspace)
            {
                (
                    increment.clamp(
                        -increments.variable.max_increment(context.config),
                        increments.variable.max_increment(context.config),
                    ),
                    count,
                    confidence,
                    truncated,
                    false,
                )
            } else {
                (f64::NAN, count, 0.0, truncated, true)
            }
        })
        .collect();

    let mut covered_grid_cells = 0usize;
    let mut solver_failed_grid_cells = 0usize;
    let mut truncated_neighbor_grid_cells = 0usize;
    let mut output = Vec::with_capacity(len);
    let mut neighbor_count = Vec::with_capacity(len);
    let mut confidence = Vec::with_capacity(len);
    for (increment, count, cell_confidence, truncated, solver_failed) in paired {
        if increment.is_finite() {
            covered_grid_cells += 1;
        }
        if truncated {
            truncated_neighbor_grid_cells += 1;
        }
        if solver_failed {
            solver_failed_grid_cells += 1;
        }
        output.push(increment);
        neighbor_count.push(count);
        confidence.push(cell_confidence);
    }

    BarnesOutput {
        increments: output,
        neighbor_count,
        confidence,
        covered_grid_cells,
        solver_failed_grid_cells,
        truncated_neighbor_grid_cells,
    }
}

fn solve_local_oi_increment(
    context: &AnalysisContext<'_>,
    increments: &IncrementSet,
    background_variance: f64,
    workspace: &mut OiWorkspace,
) -> Option<(f64, f64)> {
    let n = workspace.local.len();
    workspace.matrix.clear();
    workspace.matrix.resize(n * n, 0.0);
    workspace.rhs.clear();
    workspace.target_covariances.clear();

    for (row, local_row) in workspace.local.iter().enumerate() {
        let row_observation = &increments.observations[local_row.observation_index];
        workspace.rhs.push(row_observation.increment);
        workspace
            .target_covariances
            .push(local_row.target_covariance);
        for col in 0..=row {
            let col_observation = &increments.observations[workspace.local[col].observation_index];
            let covariance = if row == col {
                background_variance
                    + row_observation.observation_error * row_observation.observation_error
            } else {
                let distance = haversine_km(
                    row_observation.lat_deg,
                    row_observation.lon_deg,
                    col_observation.lat_deg,
                    col_observation.lon_deg,
                );
                oi_observation_covariance(
                    context,
                    row_observation,
                    col_observation,
                    distance,
                    background_variance,
                )
            };
            if !covariance.is_finite() {
                return None;
            }
            workspace.matrix[row * n + col] = covariance;
            workspace.matrix[col * n + row] = covariance;
        }
    }

    let base_jitter = (background_variance * context.config.oi_matrix_jitter_fraction).max(1.0e-12);
    if !cholesky_factor_with_jitter_into(&workspace.matrix, n, base_jitter, &mut workspace.factor) {
        return None;
    }
    if !cholesky_solve_into(
        &workspace.factor,
        n,
        &workspace.rhs,
        &mut workspace.forward_solution,
        &mut workspace.alpha,
    ) {
        return None;
    }
    if !cholesky_solve_into(
        &workspace.factor,
        n,
        &workspace.target_covariances,
        &mut workspace.forward_solution,
        &mut workspace.beta,
    ) {
        return None;
    }
    let increment = workspace
        .target_covariances
        .iter()
        .zip(workspace.alpha.iter())
        .map(|(covariance, alpha)| covariance * alpha)
        .sum::<f64>();
    let max_local_innovation = workspace
        .local
        .iter()
        .map(|local| {
            increments.observations[local.observation_index]
                .increment
                .abs()
        })
        .filter(|value| value.is_finite())
        .fold(0.0, f64::max);
    let increment = if max_local_innovation > 0.0 {
        let limit = max_local_innovation * context.config.oi_max_local_innovation_factor;
        increment.clamp(-limit, limit)
    } else {
        increment
    };
    let confidence = workspace
        .target_covariances
        .iter()
        .zip(workspace.beta.iter())
        .map(|(covariance, beta)| covariance * beta)
        .sum::<f64>()
        / background_variance;
    if increment.is_finite() && confidence.is_finite() {
        Some((increment, confidence.clamp(0.0, 1.0)))
    } else {
        None
    }
}

fn compare_local_oi_observation_priority(
    left: &LocalOiObservation,
    right: &LocalOiObservation,
) -> std::cmp::Ordering {
    right
        .target_covariance
        .total_cmp(&left.target_covariance)
        .then_with(|| left.observation_index.cmp(&right.observation_index))
}

fn oi_observation_covariance(
    context: &AnalysisContext<'_>,
    left: &IncrementObservation,
    right: &IncrementObservation,
    distance_km: f64,
    background_variance: f64,
) -> f64 {
    let left_to_right =
        oi_background_covariance(context, left.background_index, right, distance_km);
    let right_to_left =
        oi_background_covariance(context, right.background_index, left, distance_km);
    (0.5 * (left_to_right + right_to_left)).clamp(0.0, 1.0) * background_variance
}

fn cholesky_factor_with_jitter_into(
    matrix: &[f64],
    n: usize,
    base_jitter: f64,
    factor: &mut Vec<f64>,
) -> bool {
    if matrix.len() != n * n {
        return false;
    }
    for attempt in 0..6 {
        factor.clear();
        factor.extend_from_slice(matrix);
        let jitter = base_jitter * 10.0_f64.powi(attempt);
        for idx in 0..n {
            factor[idx * n + idx] += jitter;
        }
        if cholesky_decompose_lower(factor, n) {
            return true;
        }
    }
    false
}

fn cholesky_decompose_lower(matrix: &mut [f64], n: usize) -> bool {
    if matrix.len() != n * n {
        return false;
    }
    for row in 0..n {
        for col in 0..=row {
            let mut sum = matrix[row * n + col];
            for k in 0..col {
                sum -= matrix[row * n + k] * matrix[col * n + k];
            }
            if row == col {
                if !(sum.is_finite() && sum > 0.0) {
                    return false;
                }
                matrix[row * n + col] = sum.sqrt();
            } else {
                let diagonal = matrix[col * n + col];
                if !(diagonal.is_finite() && diagonal > 0.0) {
                    return false;
                }
                matrix[row * n + col] = sum / diagonal;
            }
        }
        for col in (row + 1)..n {
            matrix[row * n + col] = 0.0;
        }
    }
    true
}

fn cholesky_solve_into(
    factor: &[f64],
    n: usize,
    rhs: &[f64],
    forward_solution: &mut Vec<f64>,
    output: &mut Vec<f64>,
) -> bool {
    if factor.len() != n * n || rhs.len() != n {
        return false;
    }
    forward_solution.clear();
    forward_solution.resize(n, 0.0);
    for row in 0..n {
        let mut sum = rhs[row];
        for col in 0..row {
            sum -= factor[row * n + col] * forward_solution[col];
        }
        let diagonal = factor[row * n + row];
        if !(diagonal.is_finite() && diagonal > 0.0) {
            return false;
        }
        forward_solution[row] = sum / diagonal;
    }

    output.clear();
    output.resize(n, 0.0);
    for row in (0..n).rev() {
        let mut sum = forward_solution[row];
        for col in (row + 1)..n {
            sum -= factor[col * n + row] * output[col];
        }
        let diagonal = factor[row * n + row];
        if !(diagonal.is_finite() && diagonal > 0.0) {
            return false;
        }
        output[row] = sum / diagonal;
    }
    true
}

fn oi_background_covariance(
    context: &AnalysisContext<'_>,
    target_index: usize,
    observation: &IncrementObservation,
    distance_km: f64,
) -> f64 {
    let effective_distance = oi_effective_distance(context, target_index, observation, distance_km);
    let spatial_factor = context
        .config
        .oi_covariance_kernel
        .correlation(effective_distance);
    let terrain_factor = oi_terrain_pressure_factor(context, target_index, observation);
    spatial_factor * terrain_factor
}

fn oi_effective_distance(
    context: &AnalysisContext<'_>,
    target_index: usize,
    observation: &IncrementObservation,
    distance_km: f64,
) -> f64 {
    let length_scale = context.config.oi_length_scale_km.max(1.0e-6);
    let ratio = context.config.oi_flow_anisotropy_ratio.max(1.0);
    if ratio <= 1.0 + 1.0e-9 {
        return distance_km / length_scale;
    }
    let u = context.grid_u10_ms[target_index];
    let v = context.grid_v10_ms[target_index];
    let speed = (u * u + v * v).sqrt();
    if !(speed.is_finite() && speed >= 1.0) {
        return distance_km / length_scale;
    }
    let (dx_km, dy_km) = local_delta_km(
        observation.lat_deg,
        observation.lon_deg,
        context.grid_lat_deg[target_index],
        context.grid_lon_deg[target_index],
    );
    let unit_x = u / speed;
    let unit_y = v / speed;
    let along = dx_km * unit_x + dy_km * unit_y;
    let cross = dx_km * -unit_y + dy_km * unit_x;
    let along_scale = length_scale * ratio;
    let cross_scale = length_scale;
    ((along / along_scale).powi(2) + (cross / cross_scale).powi(2)).sqrt()
}

fn oi_terrain_pressure_factor(
    context: &AnalysisContext<'_>,
    target_index: usize,
    observation: &IncrementObservation,
) -> f64 {
    let target_pressure = context.grid_psfc_hpa[target_index];
    let observation_pressure = context.grid_psfc_hpa[observation.background_index];
    let scale = context.config.oi_terrain_pressure_scale_hpa.max(1.0e-6);
    if !(target_pressure.is_finite() && observation_pressure.is_finite()) {
        return 1.0;
    }
    let delta = target_pressure - observation_pressure;
    (-0.5 * (delta / scale).powi(2)).exp()
}

fn local_delta_km(
    from_lat_deg: f64,
    from_lon_deg: f64,
    to_lat_deg: f64,
    to_lon_deg: f64,
) -> (f64, f64) {
    let mean_lat = ((from_lat_deg + to_lat_deg) * 0.5).to_radians();
    let dx = normalize_lon_delta(to_lon_deg - from_lon_deg) * 111.0 * mean_lat.cos().abs().max(0.2);
    let dy = (to_lat_deg - from_lat_deg) * 111.0;
    (dx, dy)
}

fn residual_increment_observations(
    grid_lat_deg: &[f64],
    grid_lon_deg: &[f64],
    grid_index: &SpatialBins,
    increments: &IncrementSet,
    first_pass: &BarnesOutput,
    config: MesoanalysisConfig,
) -> Vec<IncrementObservation> {
    let mut residuals = Vec::with_capacity(increments.observations.len());
    for observation in &increments.observations {
        let Some(nearest_idx) = nearest_grid_index(
            grid_lat_deg,
            grid_lon_deg,
            grid_index,
            observation.lat_deg,
            observation.lon_deg,
            config.background_search_radius_km,
        ) else {
            continue;
        };
        let first_pass_increment = first_pass.increments[nearest_idx];
        if !first_pass_increment.is_finite() {
            continue;
        }
        let residual = observation.increment - first_pass_increment;
        if residual.is_finite() {
            residuals.push(IncrementObservation {
                lat_deg: observation.lat_deg,
                lon_deg: observation.lon_deg,
                increment: residual,
                weight: observation.weight,
                observation_error: observation.observation_error,
                background_index: observation.background_index,
            });
        }
    }
    residuals
}

fn combine_barnes_outputs(
    first: BarnesOutput,
    second: BarnesOutput,
    max_abs_increment: f64,
) -> BarnesOutput {
    let mut increments = Vec::with_capacity(first.increments.len());
    let mut neighbor_count = Vec::with_capacity(first.neighbor_count.len());
    let mut covered_grid_cells = 0usize;
    for (((first_increment, second_increment), first_count), second_count) in first
        .increments
        .into_iter()
        .zip(second.increments.into_iter())
        .zip(first.neighbor_count.into_iter())
        .zip(second.neighbor_count.into_iter())
    {
        let increment = match (first_increment.is_finite(), second_increment.is_finite()) {
            (true, true) => first_increment + second_increment,
            (true, false) => first_increment,
            (false, true) => second_increment,
            (false, false) => f64::NAN,
        };
        let increment = if increment.is_finite() {
            increment.clamp(-max_abs_increment, max_abs_increment)
        } else {
            increment
        };
        if increment.is_finite() {
            covered_grid_cells += 1;
        }
        increments.push(increment);
        neighbor_count.push(first_count.max(second_count));
    }
    let confidence = first
        .confidence
        .into_iter()
        .zip(second.confidence)
        .map(|(first, second)| (1.0 - (1.0 - first) * (1.0 - second)).clamp(0.0, 1.0))
        .collect();
    BarnesOutput {
        increments,
        neighbor_count,
        confidence,
        covered_grid_cells,
        solver_failed_grid_cells: first.solver_failed_grid_cells + second.solver_failed_grid_cells,
        truncated_neighbor_grid_cells: first.truncated_neighbor_grid_cells
            + second.truncated_neighbor_grid_cells,
    }
}

fn apply_increment(background: &[f64], increments: &BarnesOutput) -> Vec<f64> {
    background
        .iter()
        .zip(increments.increments.iter())
        .map(|(&background, &increment)| {
            if background.is_finite() && increment.is_finite() {
                background + increment
            } else {
                background
            }
        })
        .collect()
}

fn variable_diagnostics(
    increment_set: &IncrementSet,
    analysis: &BarnesOutput,
) -> MesoanalysisVariableDiagnostics {
    let accepted_observations = increment_set.observations.len();
    let mut max_abs_increment = None::<f64>;
    let mut abs_sum = 0.0;
    for observation in &increment_set.observations {
        let abs = observation.increment.abs();
        abs_sum += abs;
        max_abs_increment = Some(
            max_abs_increment
                .map(|current| current.max(abs))
                .unwrap_or(abs),
        );
    }
    let mut confidence_count = 0usize;
    let mut confidence_sum = 0.0;
    let mut max_confidence = None::<f64>;
    for &confidence in &analysis.confidence {
        if !confidence.is_finite() || confidence <= 0.0 {
            continue;
        }
        confidence_count += 1;
        confidence_sum += confidence;
        max_confidence = Some(
            max_confidence
                .map(|current| current.max(confidence))
                .unwrap_or(confidence),
        );
    }
    let mut neighbor_cell_count = 0usize;
    let mut neighbor_count_sum = 0u64;
    let mut max_neighbor_count = 0u16;
    for &count in &analysis.neighbor_count {
        max_neighbor_count = max_neighbor_count.max(count);
        if count == 0 {
            continue;
        }
        neighbor_cell_count += 1;
        neighbor_count_sum += u64::from(count);
    }

    MesoanalysisVariableDiagnostics {
        variable: increment_set.variable.label().to_string(),
        candidate_observations: increment_set.candidate_observations,
        accepted_observations,
        rejected_observations: increment_set.rejected_observations,
        covered_grid_cells: analysis.covered_grid_cells,
        solver_failed_grid_cells: analysis.solver_failed_grid_cells,
        truncated_neighbor_grid_cells: analysis.truncated_neighbor_grid_cells,
        gross_error_rescued_observations: increment_set.gross_error_rescued_observations,
        mean_neighbor_count: if neighbor_cell_count > 0 {
            Some(neighbor_count_sum as f64 / neighbor_cell_count as f64)
        } else {
            None
        },
        max_neighbor_count,
        mean_confidence: if confidence_count > 0 {
            Some(confidence_sum / confidence_count as f64)
        } else {
            None
        },
        max_confidence,
        mean_abs_increment: if accepted_observations > 0 {
            Some(abs_sum / accepted_observations as f64)
        } else {
            None
        },
        max_abs_increment,
    }
}

fn mixing_ratio_from_dewpoint_c(pressure_hpa: f64, dewpoint_c: f64) -> f64 {
    if !(pressure_hpa.is_finite() && dewpoint_c.is_finite()) || pressure_hpa <= 0.0 {
        return f64::NAN;
    }
    let vapor_pressure_hpa = 6.112 * ((17.67 * dewpoint_c) / (dewpoint_c + 243.5)).exp();
    if vapor_pressure_hpa >= pressure_hpa {
        return f64::NAN;
    }
    EPSILON * vapor_pressure_hpa / (pressure_hpa - vapor_pressure_hpa)
}

fn haversine_km(lat1_deg: f64, lon1_deg: f64, lat2_deg: f64, lon2_deg: f64) -> f64 {
    let lat1 = lat1_deg.to_radians();
    let lat2 = lat2_deg.to_radians();
    let dlat = (lat2_deg - lat1_deg).to_radians();
    let dlon = (normalize_lon_delta(lon2_deg - lon1_deg)).to_radians();
    let h = (dlat * 0.5).sin().powi(2) + lat1.cos() * lat2.cos() * (dlon * 0.5).sin().powi(2);
    2.0 * EARTH_RADIUS_KM * h.sqrt().asin()
}

fn normalize_lon_delta(delta_deg: f64) -> f64 {
    let mut delta = delta_deg;
    while delta > 180.0 {
        delta -= 360.0;
    }
    while delta < -180.0 {
        delta += 360.0;
    }
    delta
}

#[derive(Debug, Clone)]
struct SpatialBins {
    bin_deg: f64,
    entries: HashMap<(i32, i32), Vec<usize>>,
}

impl SpatialBins {
    fn new_grid(lat_deg: &[f64], lon_deg: &[f64], bin_deg: f64) -> Self {
        let mut entries = HashMap::<(i32, i32), Vec<usize>>::new();
        for (index, (&lat, &lon)) in lat_deg.iter().zip(lon_deg.iter()).enumerate() {
            entries
                .entry(Self::key(lat, lon, bin_deg))
                .or_default()
                .push(index);
        }
        Self { bin_deg, entries }
    }

    fn new_observations(observations: &[IncrementObservation], bin_deg: f64) -> Self {
        let mut entries = HashMap::<(i32, i32), Vec<usize>>::new();
        for (index, observation) in observations.iter().enumerate() {
            entries
                .entry(Self::key(observation.lat_deg, observation.lon_deg, bin_deg))
                .or_default()
                .push(index);
        }
        Self { bin_deg, entries }
    }

    fn for_each_candidate(
        &self,
        lat_deg: f64,
        lon_deg: f64,
        radius_km: f64,
        mut visit: impl FnMut(usize),
    ) {
        let lat_bins = ((radius_km / 111.0) / self.bin_deg).ceil() as i32 + 1;
        let cos_lat = lat_deg.to_radians().cos().abs().max(0.2);
        let lon_bins = ((radius_km / (111.0 * cos_lat)) / self.bin_deg).ceil() as i32 + 1;
        let (lat_key, lon_key) = Self::key(lat_deg, lon_deg, self.bin_deg);
        for y in (lat_key - lat_bins)..=(lat_key + lat_bins) {
            for x in (lon_key - lon_bins)..=(lon_key + lon_bins) {
                let wrapped_x = Self::wrapped_lon_key(x, self.bin_deg);
                if let Some(indices) = self.entries.get(&(y, wrapped_x)) {
                    for &index in indices {
                        visit(index);
                    }
                }
            }
        }
    }

    fn key(lat_deg: f64, lon_deg: f64, bin_deg: f64) -> (i32, i32) {
        (
            (lat_deg / bin_deg).floor() as i32,
            (normalize_lon(lon_deg) / bin_deg).floor() as i32,
        )
    }

    fn wrapped_lon_key(lon_key: i32, bin_deg: f64) -> i32 {
        (normalize_lon(lon_key as f64 * bin_deg) / bin_deg).floor() as i32
    }
}

fn normalize_lon(lon_deg: f64) -> f64 {
    let mut lon = lon_deg;
    while lon < -180.0 {
        lon += 360.0;
    }
    while lon >= 180.0 {
        lon -= 360.0;
    }
    lon
}

#[cfg(test)]
mod tests;
