use chrono::{Duration, NaiveDate};
use rustwx_calc::{GridShape as CalcGridShape, SurfaceInputs, compute_surface_thermo};
use rustwx_core::{
    CanonicalBundleDescriptor, CanonicalField, FieldPointSampleMethod, FieldSelector, GeoBounds,
    GeoPoint, GridShape, LatLonGrid, ModelId, ModelRunRequest, SelectedField2D, SourceId,
    VerticalSelector,
};
use rustwx_io::{
    FetchRequest, extract_fields_partial_from_model_bytes, fetch_bytes_with_cache,
    load_cached_selected_field, store_cached_selected_field,
};
use rustwx_models::{
    LatestRun, latest_available_run_for_products_at_forecast_hour, resolve_canonical_bundle_product,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::PathBuf;
use std::time::Instant;

pub const POINT_TIMESERIES_SCHEMA_VERSION: u32 = 1;

fn default_sample_method() -> FieldPointSampleMethod {
    FieldPointSampleMethod::Nearest
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PointTimeseriesRequest {
    pub model: ModelId,
    pub date_yyyymmdd: String,
    pub cycle_override_utc: Option<u8>,
    pub source: SourceId,
    pub point: GeoPoint,
    pub forecast_hours: Vec<u16>,
    pub variables: Vec<String>,
    pub cache_root: PathBuf,
    pub use_cache: bool,
    #[serde(default = "default_sample_method")]
    pub method: FieldPointSampleMethod,
}

#[derive(Debug, Clone, Serialize)]
pub struct PointTimeseriesRun {
    pub model: ModelId,
    pub date_yyyymmdd: String,
    pub cycle_utc: u8,
    pub source: SourceId,
    pub surface_product: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct PointTimeseriesVariableDescriptor {
    pub slug: String,
    pub label: &'static str,
    pub units: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct PointTimeseriesGridPoint {
    pub grid_index: usize,
    pub i: usize,
    pub j: usize,
    pub lat_deg: f64,
    pub lon_deg: f64,
    pub distance_score: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct PointTimeseriesFetchInfo {
    pub forecast_hour: u16,
    pub product: String,
    pub resolved_source: SourceId,
    pub resolved_url: String,
    pub fetch_cache_hit: bool,
    pub bytes_len: usize,
    pub fetch_ms: u128,
    pub extract_ms: u128,
    pub selected_field_cache_hits: usize,
    pub selected_field_cache_misses: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct PointTimeseriesHour {
    pub forecast_hour: u16,
    pub valid_time_utc: String,
    pub grid_point: Option<PointTimeseriesGridPoint>,
    pub values: BTreeMap<String, Option<f64>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub missing: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PointTimeseriesBlocker {
    pub forecast_hour: Option<u16>,
    pub variable: Option<String>,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct PointTimeseriesReport {
    pub schema_version: u32,
    pub run: PointTimeseriesRun,
    pub point: GeoPoint,
    pub method: FieldPointSampleMethod,
    pub variables: Vec<PointTimeseriesVariableDescriptor>,
    pub hours: Vec<PointTimeseriesHour>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fetches: Vec<PointTimeseriesFetchInfo>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub blockers: Vec<PointTimeseriesBlocker>,
    pub total_ms: u128,
}

#[derive(Debug, Clone)]
pub struct PointTimeseriesGridStoreRequest {
    pub model: ModelId,
    pub date_yyyymmdd: String,
    pub cycle_override_utc: Option<u8>,
    pub source: SourceId,
    pub forecast_hours: Vec<u16>,
    pub variables: Vec<String>,
    pub bounds: GeoBounds,
    pub cache_root: PathBuf,
    pub use_cache: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct PointTimeseriesGridStoreBuildReport {
    pub schema_version: u32,
    pub store_id: String,
    pub run: PointTimeseriesRun,
    pub bounds: GeoBounds,
    pub variables: Vec<PointTimeseriesVariableDescriptor>,
    pub requested_forecast_hours: Vec<u16>,
    pub loaded_forecast_hours: Vec<u16>,
    pub grid_points: usize,
    pub fetches: Vec<PointTimeseriesFetchInfo>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub blockers: Vec<PointTimeseriesBlocker>,
    pub memory_bytes_estimate: usize,
    pub total_ms: u128,
}

#[derive(Debug, Clone)]
pub struct PointTimeseriesGridStore {
    store_id: String,
    run: PointTimeseriesRun,
    variables: Vec<PointVariable>,
    requested_forecast_hours: Vec<u16>,
    grid_nx: usize,
    grid_indices: Vec<usize>,
    grid_lat_deg: Vec<f32>,
    grid_lon_deg: Vec<f32>,
    hours: BTreeMap<u16, GridStoreHour>,
    memory_bytes_estimate: usize,
}

#[derive(Debug, Clone)]
struct GridStoreHour {
    forecast_hour: u16,
    valid_time_utc: String,
    values: HashMap<FieldSelector, Vec<f32>>,
}

impl PointTimeseriesGridStore {
    pub fn store_id(&self) -> &str {
        &self.store_id
    }

    pub fn run(&self) -> &PointTimeseriesRun {
        &self.run
    }

    pub fn requested_forecast_hours(&self) -> &[u16] {
        &self.requested_forecast_hours
    }

    pub fn grid_points(&self) -> usize {
        self.grid_indices.len()
    }

    pub fn memory_bytes_estimate(&self) -> usize {
        self.memory_bytes_estimate
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum PointVariable {
    Temperature2mC,
    Dewpoint2mC,
    Wetbulb2mC,
    RelativeHumidity2mPct,
    WindU10mMs,
    WindV10mMs,
    WindSpeed10mMs,
    WindDirection10mDeg,
    WindGust10mMs,
    PrecipAccumMm,
    PrecipHourlyMm,
    LowCloudPct,
    MiddleCloudPct,
    HighCloudPct,
    MslpHpa,
    Vpd2mHpa,
    Hdw,
    FireWeatherComposite,
}

impl PointVariable {
    fn slug(self) -> &'static str {
        match self {
            Self::Temperature2mC => "temperature_2m_c",
            Self::Dewpoint2mC => "dewpoint_2m_c",
            Self::Wetbulb2mC => "wetbulb_2m_c",
            Self::RelativeHumidity2mPct => "relative_humidity_2m_pct",
            Self::WindU10mMs => "wind_u_10m_ms",
            Self::WindV10mMs => "wind_v_10m_ms",
            Self::WindSpeed10mMs => "wind_speed_10m_ms",
            Self::WindDirection10mDeg => "wind_direction_10m_deg",
            Self::WindGust10mMs => "wind_gust_10m_ms",
            Self::PrecipAccumMm => "precip_accum_mm",
            Self::PrecipHourlyMm => "precip_hourly_mm",
            Self::LowCloudPct => "cloud_low_pct",
            Self::MiddleCloudPct => "cloud_middle_pct",
            Self::HighCloudPct => "cloud_high_pct",
            Self::MslpHpa => "mslp_hpa",
            Self::Vpd2mHpa => "vpd_2m_hpa",
            Self::Hdw => "hdw",
            Self::FireWeatherComposite => "fire_weather_composite",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Temperature2mC => "2 m Temperature",
            Self::Dewpoint2mC => "2 m Dewpoint",
            Self::Wetbulb2mC => "2 m Wet Bulb",
            Self::RelativeHumidity2mPct => "2 m Relative Humidity",
            Self::WindU10mMs => "10 m U Wind",
            Self::WindV10mMs => "10 m V Wind",
            Self::WindSpeed10mMs => "10 m Wind Speed",
            Self::WindDirection10mDeg => "10 m Wind Direction",
            Self::WindGust10mMs => "10 m Wind Gust",
            Self::PrecipAccumMm => "Accumulated Precipitation",
            Self::PrecipHourlyMm => "Hourly Precipitation",
            Self::LowCloudPct => "Low Cloud Cover",
            Self::MiddleCloudPct => "Middle Cloud Cover",
            Self::HighCloudPct => "High Cloud Cover",
            Self::MslpHpa => "Mean Sea-Level Pressure",
            Self::Vpd2mHpa => "2 m Vapor Pressure Deficit",
            Self::Hdw => "Hot-Dry-Windy Index",
            Self::FireWeatherComposite => "Fire Weather Composite",
        }
    }

    fn units(self) -> &'static str {
        match self {
            Self::Temperature2mC | Self::Dewpoint2mC | Self::Wetbulb2mC => "degC",
            Self::RelativeHumidity2mPct
            | Self::LowCloudPct
            | Self::MiddleCloudPct
            | Self::HighCloudPct => "%",
            Self::WindU10mMs | Self::WindV10mMs | Self::WindSpeed10mMs | Self::WindGust10mMs => {
                "m/s"
            }
            Self::WindDirection10mDeg => "degrees",
            Self::PrecipAccumMm | Self::PrecipHourlyMm => "mm",
            Self::MslpHpa | Self::Vpd2mHpa => "hPa",
            Self::Hdw | Self::FireWeatherComposite => "unitless",
        }
    }

    fn descriptor(self) -> PointTimeseriesVariableDescriptor {
        PointTimeseriesVariableDescriptor {
            slug: self.slug().to_string(),
            label: self.label(),
            units: self.units(),
        }
    }
}

#[derive(Debug, Clone)]
struct HourSamples {
    forecast_hour: u16,
    valid_time_utc: String,
    grid_point: Option<PointTimeseriesGridPoint>,
    values: HashMap<FieldSelector, f64>,
}

pub fn sample_point_timeseries(
    request: &PointTimeseriesRequest,
) -> Result<PointTimeseriesReport, Box<dyn std::error::Error>> {
    let total_start = Instant::now();
    if request.forecast_hours.is_empty() {
        return Err("point timeseries request requires at least one forecast hour".into());
    }
    let variables = resolve_variables(&request.variables)?;
    let latest = resolve_timeseries_latest(request)?;
    let surface_product = resolve_canonical_bundle_product(
        request.model,
        CanonicalBundleDescriptor::SurfaceAnalysis,
        None,
    )
    .native_product;
    let mut blockers = Vec::new();
    let mut fetches = Vec::new();
    let selectors = selectors_for_variables(&variables);
    let fetch_hours = fetch_hours_for_variables(&request.forecast_hours, &variables);
    let mut samples_by_hour = BTreeMap::<u16, HourSamples>::new();

    for forecast_hour in fetch_hours {
        match load_hour_samples(
            &latest,
            forecast_hour,
            &surface_product,
            &selectors,
            request,
        ) {
            Ok((samples, fetch_info, hour_blockers)) => {
                fetches.push(fetch_info);
                samples_by_hour.insert(forecast_hour, samples);
                blockers.extend(hour_blockers);
            }
            Err(err) => blockers.push(PointTimeseriesBlocker {
                forecast_hour: Some(forecast_hour),
                variable: None,
                reason: err.to_string(),
            }),
        }
    }

    let mut hours = Vec::new();
    for &forecast_hour in &request.forecast_hours {
        if let Some(samples) = samples_by_hour.get(&forecast_hour) {
            hours.push(build_report_hour(
                samples,
                samples_by_hour.get(&forecast_hour.saturating_sub(1)),
                &variables,
            ));
        } else {
            let mut values = BTreeMap::new();
            let mut missing = Vec::new();
            for variable in &variables {
                values.insert(variable.slug().to_string(), None);
                missing.push(variable.slug().to_string());
            }
            hours.push(PointTimeseriesHour {
                forecast_hour,
                valid_time_utc: valid_time_utc(&latest, forecast_hour)?,
                grid_point: None,
                values,
                missing,
            });
        }
    }

    Ok(PointTimeseriesReport {
        schema_version: POINT_TIMESERIES_SCHEMA_VERSION,
        run: PointTimeseriesRun {
            model: latest.model,
            date_yyyymmdd: latest.cycle.date_yyyymmdd,
            cycle_utc: latest.cycle.hour_utc,
            source: latest.source,
            surface_product,
        },
        point: request.point,
        method: request.method,
        variables: variables
            .iter()
            .copied()
            .map(PointVariable::descriptor)
            .collect(),
        hours,
        fetches,
        blockers,
        total_ms: total_start.elapsed().as_millis(),
    })
}

pub fn build_point_timeseries_grid_store(
    request: &PointTimeseriesGridStoreRequest,
) -> Result<
    (
        PointTimeseriesGridStore,
        PointTimeseriesGridStoreBuildReport,
    ),
    Box<dyn std::error::Error>,
> {
    let total_start = Instant::now();
    if request.forecast_hours.is_empty() {
        return Err("point timeseries grid store requires at least one forecast hour".into());
    }
    let variables = resolve_variables(&request.variables)?;
    let latest = resolve_grid_store_latest(request, &variables)?;
    let surface_product = resolve_canonical_bundle_product(
        request.model,
        CanonicalBundleDescriptor::SurfaceAnalysis,
        None,
    )
    .native_product;
    let selectors = selectors_for_variables(&variables);
    let fetch_hours = fetch_hours_for_variables(&request.forecast_hours, &variables);
    let mut fetches = Vec::new();
    let mut blockers = Vec::new();
    let mut grid_indices: Vec<usize> = Vec::new();
    let mut grid_nx = 0usize;
    let mut grid_lat_deg: Vec<f32> = Vec::new();
    let mut grid_lon_deg: Vec<f32> = Vec::new();
    let mut hours = BTreeMap::<u16, GridStoreHour>::new();

    for forecast_hour in fetch_hours {
        match load_hour_fields(
            &latest,
            forecast_hour,
            &surface_product,
            &selectors,
            &request.cache_root,
            request.use_cache,
        ) {
            Ok((fields, fetch_info, hour_blockers)) => {
                fetches.push(fetch_info);
                blockers.extend(hour_blockers);
                if fields.is_empty() {
                    blockers.push(PointTimeseriesBlocker {
                        forecast_hour: Some(forecast_hour),
                        variable: None,
                        reason: "no fields loaded for store hour".to_string(),
                    });
                    continue;
                }
                if grid_indices.is_empty() {
                    grid_nx = fields[0].grid.shape.nx;
                    grid_indices = indices_for_bounds(&fields[0].grid, request.bounds);
                    if grid_indices.is_empty() {
                        return Err(
                            "point timeseries grid store bounds selected no grid points".into()
                        );
                    }
                    grid_lat_deg = grid_indices
                        .iter()
                        .map(|&idx| fields[0].grid.lat_deg[idx])
                        .collect();
                    grid_lon_deg = grid_indices
                        .iter()
                        .map(|&idx| fields[0].grid.lon_deg[idx])
                        .collect();
                }

                let mut values = HashMap::new();
                for field in fields {
                    if field.values.len() <= *grid_indices.iter().max().unwrap_or(&0) {
                        blockers.push(PointTimeseriesBlocker {
                            forecast_hour: Some(forecast_hour),
                            variable: Some(field.selector.key()),
                            reason: "field grid is smaller than the store grid".to_string(),
                        });
                        continue;
                    }
                    let cropped = grid_indices
                        .iter()
                        .map(|&idx| field.values[idx])
                        .collect::<Vec<_>>();
                    values.insert(field.selector, cropped);
                }
                hours.insert(
                    forecast_hour,
                    GridStoreHour {
                        forecast_hour,
                        valid_time_utc: valid_time_utc(&latest, forecast_hour)?,
                        values,
                    },
                );
            }
            Err(err) => blockers.push(PointTimeseriesBlocker {
                forecast_hour: Some(forecast_hour),
                variable: None,
                reason: err.to_string(),
            }),
        }
    }

    if hours.is_empty() {
        return Err("point timeseries grid store loaded no hours".into());
    }

    let loaded_forecast_hours = hours.keys().copied().collect::<Vec<_>>();
    let run = PointTimeseriesRun {
        model: latest.model,
        date_yyyymmdd: latest.cycle.date_yyyymmdd,
        cycle_utc: latest.cycle.hour_utc,
        source: latest.source,
        surface_product,
    };
    let store_id = grid_store_id(&run, request.bounds, &request.forecast_hours, &variables);
    let memory_bytes_estimate = grid_lat_deg.len() * std::mem::size_of::<f32>() * 2
        + grid_indices.len() * std::mem::size_of::<usize>()
        + hours
            .values()
            .flat_map(|hour| hour.values.values())
            .map(|values| values.len() * std::mem::size_of::<f32>())
            .sum::<usize>();
    let store = PointTimeseriesGridStore {
        store_id: store_id.clone(),
        run: run.clone(),
        variables: variables.clone(),
        requested_forecast_hours: request.forecast_hours.clone(),
        grid_nx,
        grid_indices,
        grid_lat_deg,
        grid_lon_deg,
        hours,
        memory_bytes_estimate,
    };
    let report = PointTimeseriesGridStoreBuildReport {
        schema_version: POINT_TIMESERIES_SCHEMA_VERSION,
        store_id,
        run,
        bounds: request.bounds,
        variables: variables
            .iter()
            .copied()
            .map(PointVariable::descriptor)
            .collect(),
        requested_forecast_hours: request.forecast_hours.clone(),
        loaded_forecast_hours,
        grid_points: store.grid_points(),
        fetches,
        blockers,
        memory_bytes_estimate,
        total_ms: total_start.elapsed().as_millis(),
    };
    Ok((store, report))
}

pub fn sample_point_timeseries_grid_store(
    store: &PointTimeseriesGridStore,
    point: GeoPoint,
    method: FieldPointSampleMethod,
    forecast_hours: Option<&[u16]>,
) -> Result<PointTimeseriesReport, Box<dyn std::error::Error>> {
    let total_start = Instant::now();
    if method != FieldPointSampleMethod::Nearest {
        return Err("point timeseries grid store currently supports nearest sampling only".into());
    }
    let (cropped_index, grid_point) = nearest_grid_store_point(store, point)
        .ok_or("point timeseries grid store contains no grid points")?;
    let mut hours = Vec::new();
    let requested_hours = forecast_hours.unwrap_or(&store.requested_forecast_hours);
    for &forecast_hour in requested_hours {
        if let Some(store_hour) = store.hours.get(&forecast_hour) {
            let samples = hour_samples_from_store_hour(store_hour, &grid_point, cropped_index);
            let previous = forecast_hour
                .checked_sub(1)
                .and_then(|previous_hour| store.hours.get(&previous_hour))
                .map(|hour| hour_samples_from_store_hour(hour, &grid_point, cropped_index));
            hours.push(build_report_hour(
                &samples,
                previous.as_ref(),
                &store.variables,
            ));
        } else {
            let mut values = BTreeMap::new();
            let mut missing = Vec::new();
            for variable in &store.variables {
                values.insert(variable.slug().to_string(), None);
                missing.push(variable.slug().to_string());
            }
            hours.push(PointTimeseriesHour {
                forecast_hour,
                valid_time_utc: valid_time_from_run(&store.run, forecast_hour)?,
                grid_point: Some(grid_point.clone()),
                values,
                missing,
            });
        }
    }

    Ok(PointTimeseriesReport {
        schema_version: POINT_TIMESERIES_SCHEMA_VERSION,
        run: store.run.clone(),
        point,
        method,
        variables: store
            .variables
            .iter()
            .copied()
            .map(PointVariable::descriptor)
            .collect(),
        hours,
        fetches: Vec::new(),
        blockers: Vec::new(),
        total_ms: total_start.elapsed().as_millis(),
    })
}

fn resolve_timeseries_latest(
    request: &PointTimeseriesRequest,
) -> Result<LatestRun, Box<dyn std::error::Error>> {
    if let Some(cycle_hour) = request.cycle_override_utc {
        return Ok(LatestRun {
            model: request.model,
            cycle: rustwx_core::CycleSpec::new(&request.date_yyyymmdd, cycle_hour)?,
            source: request.source,
        });
    }

    let max_hour = request
        .forecast_hours
        .iter()
        .copied()
        .max()
        .ok_or("point timeseries request has no forecast hours")?;
    let surface_product = resolve_canonical_bundle_product(
        request.model,
        CanonicalBundleDescriptor::SurfaceAnalysis,
        None,
    )
    .native_product;
    Ok(latest_available_run_for_products_at_forecast_hour(
        request.model,
        Some(request.source),
        &request.date_yyyymmdd,
        &[surface_product.as_str()],
        max_hour,
    )?)
}

fn resolve_grid_store_latest(
    request: &PointTimeseriesGridStoreRequest,
    variables: &[PointVariable],
) -> Result<LatestRun, Box<dyn std::error::Error>> {
    if let Some(cycle_hour) = request.cycle_override_utc {
        return Ok(LatestRun {
            model: request.model,
            cycle: rustwx_core::CycleSpec::new(&request.date_yyyymmdd, cycle_hour)?,
            source: request.source,
        });
    }

    let fetch_hours = fetch_hours_for_variables(&request.forecast_hours, variables);
    let max_hour = fetch_hours
        .iter()
        .copied()
        .max()
        .ok_or("point timeseries grid store request has no forecast hours")?;
    let surface_product = resolve_canonical_bundle_product(
        request.model,
        CanonicalBundleDescriptor::SurfaceAnalysis,
        None,
    )
    .native_product;
    Ok(latest_available_run_for_products_at_forecast_hour(
        request.model,
        Some(request.source),
        &request.date_yyyymmdd,
        &[surface_product.as_str()],
        max_hour,
    )?)
}

fn indices_for_bounds(grid: &LatLonGrid, bounds: GeoBounds) -> Vec<usize> {
    (0..grid.shape.len())
        .filter(|&idx| {
            bounds.contains(GeoPoint::new(
                f64::from(grid.lat_deg[idx]),
                f64::from(grid.lon_deg[idx]),
            ))
        })
        .collect()
}

fn grid_store_id(
    run: &PointTimeseriesRun,
    bounds: GeoBounds,
    forecast_hours: &[u16],
    variables: &[PointVariable],
) -> String {
    let first_hour = forecast_hours.iter().copied().min().unwrap_or(0);
    let last_hour = forecast_hours.iter().copied().max().unwrap_or(0);
    let vars = variables
        .iter()
        .map(|variable| variable.slug())
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{}:{}:{:02}:{}:{:.3}:{:.3}:{:.3}:{:.3}:f{:03}-f{:03}:{}",
        run.model.as_str(),
        run.date_yyyymmdd,
        run.cycle_utc,
        run.source.as_str(),
        bounds.west_lon_deg,
        bounds.east_lon_deg,
        bounds.south_lat_deg,
        bounds.north_lat_deg,
        first_hour,
        last_hour,
        vars
    )
}

fn nearest_grid_store_point(
    store: &PointTimeseriesGridStore,
    point: GeoPoint,
) -> Option<(usize, PointTimeseriesGridPoint)> {
    let mut best: Option<(usize, f64)> = None;
    for idx in 0..store.grid_lat_deg.len() {
        let distance = geographic_distance_score_for_lat_lon(
            f64::from(store.grid_lat_deg[idx]),
            f64::from(store.grid_lon_deg[idx]),
            point,
        );
        match best {
            Some((best_idx, best_distance))
                if distance > best_distance
                    || ((distance - best_distance).abs() <= 1.0e-12 && idx >= best_idx) => {}
            _ => best = Some((idx, distance)),
        }
    }
    let (cropped_index, distance_score) = best?;
    let grid_index = store.grid_indices[cropped_index];
    Some((
        cropped_index,
        PointTimeseriesGridPoint {
            grid_index,
            i: grid_index % store.grid_nx,
            j: grid_index / store.grid_nx,
            lat_deg: f64::from(store.grid_lat_deg[cropped_index]),
            lon_deg: f64::from(store.grid_lon_deg[cropped_index]),
            distance_score,
        },
    ))
}

fn hour_samples_from_store_hour(
    hour: &GridStoreHour,
    grid_point: &PointTimeseriesGridPoint,
    cropped_index: usize,
) -> HourSamples {
    let values = hour
        .values
        .iter()
        .filter_map(|(&selector, values)| {
            values
                .get(cropped_index)
                .copied()
                .filter(|value| value.is_finite())
                .map(|value| (selector, f64::from(value)))
        })
        .collect::<HashMap<_, _>>();
    HourSamples {
        forecast_hour: hour.forecast_hour,
        valid_time_utc: hour.valid_time_utc.clone(),
        grid_point: Some(grid_point.clone()),
        values,
    }
}

fn load_hour_samples(
    latest: &LatestRun,
    forecast_hour: u16,
    surface_product: &str,
    selectors: &[FieldSelector],
    request: &PointTimeseriesRequest,
) -> Result<
    (
        HourSamples,
        PointTimeseriesFetchInfo,
        Vec<PointTimeseriesBlocker>,
    ),
    Box<dyn std::error::Error>,
> {
    let (fields, fetch_info, blockers) = load_hour_fields(
        latest,
        forecast_hour,
        surface_product,
        selectors,
        &request.cache_root,
        request.use_cache,
    )?;

    let grid_point = fields
        .first()
        .and_then(|field| nearest_grid_point(&field.grid, request.point));
    let mut values = HashMap::new();
    for field in fields {
        let value = match request.method {
            FieldPointSampleMethod::Nearest => grid_point
                .as_ref()
                .and_then(|point| field.values.get(point.grid_index))
                .copied()
                .filter(|value| value.is_finite())
                .map(f64::from),
            FieldPointSampleMethod::InverseDistance4 => field
                .sample_point(request.point, request.method)
                .value
                .filter(|value| value.is_finite())
                .map(f64::from),
        };
        if let Some(value) = value {
            values.insert(field.selector, value);
        }
    }

    Ok((
        HourSamples {
            forecast_hour,
            valid_time_utc: valid_time_utc(latest, forecast_hour)?,
            grid_point,
            values,
        },
        fetch_info,
        blockers,
    ))
}

fn load_hour_fields(
    latest: &LatestRun,
    forecast_hour: u16,
    surface_product: &str,
    selectors: &[FieldSelector],
    cache_root: &std::path::Path,
    use_cache: bool,
) -> Result<
    (
        Vec<SelectedField2D>,
        PointTimeseriesFetchInfo,
        Vec<PointTimeseriesBlocker>,
    ),
    Box<dyn std::error::Error>,
> {
    let fetch_request = FetchRequest {
        request: ModelRunRequest::new(
            latest.model,
            latest.cycle.clone(),
            forecast_hour,
            surface_product,
        )?,
        source_override: Some(latest.source),
        variable_patterns: timeseries_fetch_patterns(latest.model, surface_product),
    };
    let fetch_start = Instant::now();
    let fetched = fetch_bytes_with_cache(&fetch_request, cache_root, use_cache)?;
    let fetch_ms = fetch_start.elapsed().as_millis();
    let extract_start = Instant::now();

    let mut fields = Vec::new();
    let mut missing_for_extract = Vec::new();
    let mut cache_hits = 0usize;
    if use_cache {
        for selector in selectors {
            if let Some(cached) = load_cached_selected_field(cache_root, &fetch_request, *selector)?
            {
                fields.push(cached.field);
                cache_hits += 1;
            } else {
                missing_for_extract.push(*selector);
            }
        }
    } else {
        missing_for_extract.extend(selectors.iter().copied());
    }

    let mut blockers = Vec::new();
    if !missing_for_extract.is_empty() {
        let partial = extract_fields_partial_from_model_bytes(
            latest.model,
            &fetched.result.bytes,
            Some(fetched.bytes_path.as_path()),
            &missing_for_extract,
        )?;
        if use_cache {
            for field in &partial.extracted {
                store_cached_selected_field(cache_root, &fetch_request, field)?;
            }
        }
        fields.extend(partial.extracted);
        for missing in partial.missing {
            blockers.push(PointTimeseriesBlocker {
                forecast_hour: Some(forecast_hour),
                variable: None,
                reason: format!("missing GRIB message for selector {}", missing.key()),
            });
        }
    }
    let extract_ms = extract_start.elapsed().as_millis();

    Ok((
        fields,
        PointTimeseriesFetchInfo {
            forecast_hour,
            product: surface_product.to_string(),
            resolved_source: fetched.result.source,
            resolved_url: fetched.result.url,
            fetch_cache_hit: fetched.cache_hit,
            bytes_len: fetched.result.bytes.len(),
            fetch_ms,
            extract_ms,
            selected_field_cache_hits: cache_hits,
            selected_field_cache_misses: missing_for_extract.len(),
        },
        blockers,
    ))
}

fn build_report_hour(
    samples: &HourSamples,
    previous: Option<&HourSamples>,
    variables: &[PointVariable],
) -> PointTimeseriesHour {
    let mut derived = DerivedPointState::from_samples(samples);
    let mut values = BTreeMap::new();
    let mut missing = Vec::new();
    for variable in variables {
        let value = match variable {
            PointVariable::Temperature2mC => derived.temperature_c(),
            PointVariable::Dewpoint2mC => derived.dewpoint_c(),
            PointVariable::Wetbulb2mC => derived.surface_thermo().map(|thermo| thermo.wetbulb_c),
            PointVariable::RelativeHumidity2mPct => derived.relative_humidity_pct(),
            PointVariable::WindU10mMs => derived.u10_ms(),
            PointVariable::WindV10mMs => derived.v10_ms(),
            PointVariable::WindSpeed10mMs => derived.wind_speed_ms(),
            PointVariable::WindDirection10mDeg => derived.wind_direction_deg(),
            PointVariable::WindGust10mMs => derived.wind_gust_ms(),
            PointVariable::PrecipAccumMm => derived.precip_accum_mm(),
            PointVariable::PrecipHourlyMm => {
                let current = derived.precip_accum_mm();
                let previous = previous
                    .and_then(|hour| DerivedPointState::from_samples(hour).precip_accum_mm());
                match (current, previous) {
                    (Some(current), Some(previous)) => Some((current - previous).max(0.0)),
                    (Some(current), None) if samples.forecast_hour == 0 => Some(current.max(0.0)),
                    _ => None,
                }
            }
            PointVariable::LowCloudPct => derived.low_cloud_pct(),
            PointVariable::MiddleCloudPct => derived.middle_cloud_pct(),
            PointVariable::HighCloudPct => derived.high_cloud_pct(),
            PointVariable::MslpHpa => derived.mslp_hpa(),
            PointVariable::Vpd2mHpa => derived.surface_thermo().map(|thermo| thermo.vpd_hpa),
            PointVariable::Hdw => derived
                .surface_thermo()
                .and_then(|thermo| derived.wind_speed_ms().map(|wind| thermo.vpd_hpa * wind)),
            PointVariable::FireWeatherComposite => derived
                .surface_thermo()
                .map(|thermo| thermo.fire_weather_composite),
        };
        if value.is_none() {
            missing.push(variable.slug().to_string());
        }
        values.insert(variable.slug().to_string(), value);
    }

    PointTimeseriesHour {
        forecast_hour: samples.forecast_hour,
        valid_time_utc: samples.valid_time_utc.clone(),
        grid_point: samples.grid_point.clone(),
        values,
        missing,
    }
}

#[derive(Debug, Clone, Copy)]
struct SurfaceThermoPoint {
    wetbulb_c: f64,
    vpd_hpa: f64,
    fire_weather_composite: f64,
}

struct DerivedPointState<'a> {
    samples: &'a HourSamples,
    surface_thermo: Option<Option<SurfaceThermoPoint>>,
}

impl<'a> DerivedPointState<'a> {
    fn from_samples(samples: &'a HourSamples) -> Self {
        Self {
            samples,
            surface_thermo: None,
        }
    }

    fn value(&self, selector: FieldSelector) -> Option<f64> {
        self.samples.values.get(&selector).copied()
    }

    fn temperature_k(&self) -> Option<f64> {
        self.value(FieldSelector::height_agl(CanonicalField::Temperature, 2))
    }

    fn temperature_c(&self) -> Option<f64> {
        self.temperature_k().map(|value| value - 273.15)
    }

    fn dewpoint_k(&self) -> Option<f64> {
        self.value(FieldSelector::height_agl(CanonicalField::Dewpoint, 2))
    }

    fn dewpoint_c(&self) -> Option<f64> {
        self.dewpoint_k()
            .map(|value| value - 273.15)
            .or_else(|| dewpoint_from_temperature_rh(self.temperature_c()?, self.native_rh_pct()?))
    }

    fn native_rh_pct(&self) -> Option<f64> {
        self.value(FieldSelector::height_agl(
            CanonicalField::RelativeHumidity,
            2,
        ))
        .map(|value| value.clamp(0.0, 100.0))
    }

    fn relative_humidity_pct(&self) -> Option<f64> {
        self.native_rh_pct().or_else(|| {
            let temp_c = self.temperature_c()?;
            let dewpoint_c = self.dewpoint_c()?;
            let saturation = saturation_vapor_pressure_hpa(temp_c);
            if saturation <= 0.0 {
                return None;
            }
            Some((saturation_vapor_pressure_hpa(dewpoint_c) / saturation * 100.0).clamp(0.0, 100.0))
        })
    }

    fn surface_pressure_pa(&self) -> Option<f64> {
        self.value(FieldSelector::surface(CanonicalField::Pressure))
            .or_else(|| {
                self.value(FieldSelector::mean_sea_level(
                    CanonicalField::PressureReducedToMeanSeaLevel,
                ))
            })
    }

    fn u10_ms(&self) -> Option<f64> {
        self.value(FieldSelector::height_agl(CanonicalField::UWind, 10))
    }

    fn v10_ms(&self) -> Option<f64> {
        self.value(FieldSelector::height_agl(CanonicalField::VWind, 10))
    }

    fn wind_speed_ms(&self) -> Option<f64> {
        let u = self.u10_ms()?;
        let v = self.v10_ms()?;
        Some((u * u + v * v).sqrt())
    }

    fn wind_direction_deg(&self) -> Option<f64> {
        let u = self.u10_ms()?;
        let v = self.v10_ms()?;
        let mut degrees = (-u).atan2(-v).to_degrees();
        if degrees < 0.0 {
            degrees += 360.0;
        }
        Some(degrees)
    }

    fn wind_gust_ms(&self) -> Option<f64> {
        self.value(FieldSelector::height_agl(CanonicalField::WindGust, 10))
    }

    fn precip_accum_mm(&self) -> Option<f64> {
        self.value(FieldSelector::surface(CanonicalField::TotalPrecipitation))
    }

    fn low_cloud_pct(&self) -> Option<f64> {
        self.value(FieldSelector::entire_atmosphere(
            CanonicalField::LowCloudCover,
        ))
    }

    fn middle_cloud_pct(&self) -> Option<f64> {
        self.value(FieldSelector::entire_atmosphere(
            CanonicalField::MiddleCloudCover,
        ))
    }

    fn high_cloud_pct(&self) -> Option<f64> {
        self.value(FieldSelector::entire_atmosphere(
            CanonicalField::HighCloudCover,
        ))
    }

    fn mslp_hpa(&self) -> Option<f64> {
        self.value(FieldSelector::mean_sea_level(
            CanonicalField::PressureReducedToMeanSeaLevel,
        ))
        .map(|value| value / 100.0)
    }

    fn surface_thermo(&mut self) -> Option<SurfaceThermoPoint> {
        if let Some(cached) = self.surface_thermo {
            return cached;
        }
        let computed = self.compute_surface_thermo();
        self.surface_thermo = Some(computed);
        computed
    }

    fn compute_surface_thermo(&self) -> Option<SurfaceThermoPoint> {
        let pressure_pa = self.surface_pressure_pa()?;
        let temperature_k = self.temperature_k()?;
        let dewpoint_c = self.dewpoint_c()?;
        let q2_kgkg = mixing_ratio_from_dewpoint_c(pressure_pa / 100.0, dewpoint_c);
        let u10 = self.u10_ms()?;
        let v10 = self.v10_ms()?;
        let psfc = [pressure_pa];
        let t2 = [temperature_k];
        let q2 = [q2_kgkg];
        let u = [u10];
        let v = [v10];
        let thermo = compute_surface_thermo(
            CalcGridShape::new(1, 1).ok()?,
            SurfaceInputs {
                psfc_pa: &psfc,
                t2_k: &t2,
                q2_kgkg: &q2,
                u10_ms: &u,
                v10_ms: &v,
            },
        )
        .ok()?;
        Some(SurfaceThermoPoint {
            wetbulb_c: thermo.wetbulb_2m_c[0],
            vpd_hpa: thermo.vpd_2m_hpa[0],
            fire_weather_composite: thermo.fire_weather_composite[0],
        })
    }
}

fn selectors_for_variables(variables: &[PointVariable]) -> Vec<FieldSelector> {
    let mut selectors = Vec::new();
    let needs_temperature = variables.iter().any(|variable| {
        matches!(
            variable,
            PointVariable::Temperature2mC
                | PointVariable::Dewpoint2mC
                | PointVariable::Wetbulb2mC
                | PointVariable::RelativeHumidity2mPct
                | PointVariable::Vpd2mHpa
                | PointVariable::Hdw
                | PointVariable::FireWeatherComposite
        )
    });
    let needs_moisture = variables.iter().any(|variable| {
        matches!(
            variable,
            PointVariable::Dewpoint2mC
                | PointVariable::Wetbulb2mC
                | PointVariable::RelativeHumidity2mPct
                | PointVariable::Vpd2mHpa
                | PointVariable::Hdw
                | PointVariable::FireWeatherComposite
        )
    });
    let needs_wind = variables.iter().any(|variable| {
        matches!(
            variable,
            PointVariable::WindU10mMs
                | PointVariable::WindV10mMs
                | PointVariable::WindSpeed10mMs
                | PointVariable::WindDirection10mDeg
                | PointVariable::Hdw
                | PointVariable::FireWeatherComposite
        )
    });
    let needs_pressure = variables.iter().any(|variable| {
        matches!(
            variable,
            PointVariable::Wetbulb2mC
                | PointVariable::Vpd2mHpa
                | PointVariable::Hdw
                | PointVariable::FireWeatherComposite
        )
    });

    if needs_temperature {
        push_selector(
            &mut selectors,
            FieldSelector::height_agl(CanonicalField::Temperature, 2),
        );
    }
    if needs_moisture {
        push_selector(
            &mut selectors,
            FieldSelector::height_agl(CanonicalField::Dewpoint, 2),
        );
        push_selector(
            &mut selectors,
            FieldSelector::height_agl(CanonicalField::RelativeHumidity, 2),
        );
    }
    if needs_wind {
        push_selector(
            &mut selectors,
            FieldSelector::height_agl(CanonicalField::UWind, 10),
        );
        push_selector(
            &mut selectors,
            FieldSelector::height_agl(CanonicalField::VWind, 10),
        );
    }
    if needs_pressure {
        push_selector(
            &mut selectors,
            FieldSelector::surface(CanonicalField::Pressure),
        );
    }
    for variable in variables {
        match variable {
            PointVariable::WindGust10mMs => {
                push_selector(
                    &mut selectors,
                    FieldSelector::height_agl(CanonicalField::WindGust, 10),
                );
            }
            PointVariable::PrecipAccumMm | PointVariable::PrecipHourlyMm => {
                push_selector(
                    &mut selectors,
                    FieldSelector::surface(CanonicalField::TotalPrecipitation),
                );
            }
            PointVariable::LowCloudPct => {
                push_selector(
                    &mut selectors,
                    FieldSelector::entire_atmosphere(CanonicalField::LowCloudCover),
                );
            }
            PointVariable::MiddleCloudPct => {
                push_selector(
                    &mut selectors,
                    FieldSelector::entire_atmosphere(CanonicalField::MiddleCloudCover),
                );
            }
            PointVariable::HighCloudPct => {
                push_selector(
                    &mut selectors,
                    FieldSelector::entire_atmosphere(CanonicalField::HighCloudCover),
                );
            }
            PointVariable::MslpHpa => {
                push_selector(
                    &mut selectors,
                    FieldSelector::mean_sea_level(CanonicalField::PressureReducedToMeanSeaLevel),
                );
            }
            _ => {}
        }
    }
    selectors
}

fn push_selector(selectors: &mut Vec<FieldSelector>, selector: FieldSelector) {
    if !selectors.contains(&selector) {
        selectors.push(selector);
    }
}

fn fetch_hours_for_variables(requested: &[u16], variables: &[PointVariable]) -> Vec<u16> {
    let mut hours = BTreeSet::new();
    let needs_previous = variables.contains(&PointVariable::PrecipHourlyMm);
    for &hour in requested {
        hours.insert(hour);
        if needs_previous && hour > 0 {
            hours.insert(hour - 1);
        }
    }
    hours.into_iter().collect()
}

fn timeseries_fetch_patterns(model: ModelId, surface_product: &str) -> Vec<String> {
    if model == ModelId::Hrrr && surface_product == "sfc" {
        return [
            "PRES:surface",
            "APCP:surface",
            "TMP:2 m above ground",
            "DPT:2 m above ground",
            "RH:2 m above ground",
            "UGRD:10 m above ground",
            "VGRD:10 m above ground",
            "GUST:surface",
            "GUST:10 m above ground",
            "MSLMA:mean sea level",
            "PRMSL:mean sea level",
            "MSLET:mean sea level",
            "LCDC:low cloud layer",
            "MCDC:middle cloud layer",
            "HCDC:high cloud layer",
        ]
        .into_iter()
        .map(str::to_string)
        .collect();
    }
    Vec::new()
}

fn resolve_variables(
    requested: &[String],
) -> Result<Vec<PointVariable>, Box<dyn std::error::Error>> {
    let requested = if requested.is_empty() {
        default_variables()
    } else {
        requested.to_vec()
    };
    let mut seen = BTreeSet::new();
    let mut variables = Vec::new();
    for raw in requested {
        let variable = parse_variable(&raw)
            .ok_or_else(|| format!("unsupported point-timeseries variable '{raw}'"))?;
        if seen.insert(variable) {
            variables.push(variable);
        }
    }
    Ok(variables)
}

fn default_variables() -> Vec<String> {
    [
        "temperature_2m_c",
        "dewpoint_2m_c",
        "wetbulb_2m_c",
        "relative_humidity_2m_pct",
        "wind_speed_10m_ms",
        "wind_direction_10m_deg",
        "wind_gust_10m_ms",
        "precip_hourly_mm",
        "precip_accum_mm",
        "cloud_low_pct",
        "cloud_middle_pct",
        "cloud_high_pct",
        "mslp_hpa",
        "vpd_2m_hpa",
        "hdw",
        "fire_weather_composite",
    ]
    .into_iter()
    .map(str::to_string)
    .collect()
}

fn parse_variable(raw: &str) -> Option<PointVariable> {
    match normalize_slug(raw).as_str() {
        "temperature_2m" | "temperature_2m_c" | "temp_2m" | "temp_2m_c" | "t2m" | "t2m_c" => {
            Some(PointVariable::Temperature2mC)
        }
        "dewpoint_2m" | "dewpoint_2m_c" | "2m_dewpoint" | "td2m" | "td2m_c" => {
            Some(PointVariable::Dewpoint2mC)
        }
        "wetbulb_2m" | "wetbulb_2m_c" | "wet_bulb_2m" | "tw2m" | "tw2m_c" => {
            Some(PointVariable::Wetbulb2mC)
        }
        "relative_humidity_2m" | "relative_humidity_2m_pct" | "rh_2m" | "rh2m" => {
            Some(PointVariable::RelativeHumidity2mPct)
        }
        "wind_u_10m" | "wind_u_10m_ms" | "u10" | "u10m" => Some(PointVariable::WindU10mMs),
        "wind_v_10m" | "wind_v_10m_ms" | "v10" | "v10m" => Some(PointVariable::WindV10mMs),
        "wind_speed_10m" | "wind_speed_10m_ms" | "10m_wind_speed" | "wspd10m" => {
            Some(PointVariable::WindSpeed10mMs)
        }
        "wind_direction_10m" | "wind_direction_10m_deg" | "10m_wind_direction" | "wdir10m" => {
            Some(PointVariable::WindDirection10mDeg)
        }
        "wind_gust_10m" | "wind_gust_10m_ms" | "10m_wind_gust" | "gust" => {
            Some(PointVariable::WindGust10mMs)
        }
        "precip_accum" | "precip_accum_mm" | "qpf_accum" | "total_qpf" => {
            Some(PointVariable::PrecipAccumMm)
        }
        "precip_hourly" | "precip_hourly_mm" | "qpf_hourly" | "qpf_1h" | "hourly_qpf" => {
            Some(PointVariable::PrecipHourlyMm)
        }
        "cloud_low" | "cloud_low_pct" | "low_cloud" | "lcc" => Some(PointVariable::LowCloudPct),
        "cloud_middle" | "cloud_middle_pct" | "cloud_mid" | "cloud_mid_pct" | "mcc" => {
            Some(PointVariable::MiddleCloudPct)
        }
        "cloud_high" | "cloud_high_pct" | "high_cloud" | "hcc" => Some(PointVariable::HighCloudPct),
        "mslp" | "mslp_hpa" | "mean_sea_level_pressure" => Some(PointVariable::MslpHpa),
        "vpd" | "vpd_2m" | "vpd_2m_hpa" | "vapor_pressure_deficit_2m" => {
            Some(PointVariable::Vpd2mHpa)
        }
        "hdw" | "hot_dry_windy" => Some(PointVariable::Hdw),
        "fire_weather_composite" | "fwc" => Some(PointVariable::FireWeatherComposite),
        _ => None,
    }
}

fn normalize_slug(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    let mut last_was_separator = false;
    for ch in value.trim().chars() {
        let normalized = ch.to_ascii_lowercase();
        if normalized.is_ascii_alphanumeric() {
            out.push(normalized);
            last_was_separator = false;
        } else if !last_was_separator {
            out.push('_');
            last_was_separator = true;
        }
    }
    out.trim_matches('_').to_string()
}

fn valid_time_utc(
    latest: &LatestRun,
    forecast_hour: u16,
) -> Result<String, Box<dyn std::error::Error>> {
    valid_time_parts(
        &latest.cycle.date_yyyymmdd,
        latest.cycle.hour_utc,
        forecast_hour,
    )
}

fn valid_time_from_run(
    run: &PointTimeseriesRun,
    forecast_hour: u16,
) -> Result<String, Box<dyn std::error::Error>> {
    valid_time_parts(&run.date_yyyymmdd, run.cycle_utc, forecast_hour)
}

fn valid_time_parts(
    date_yyyymmdd: &str,
    cycle_utc: u8,
    forecast_hour: u16,
) -> Result<String, Box<dyn std::error::Error>> {
    let date = NaiveDate::parse_from_str(date_yyyymmdd, "%Y%m%d")?;
    let cycle_time = date
        .and_hms_opt(u32::from(cycle_utc), 0, 0)
        .ok_or("invalid cycle hour")?;
    let valid_time = cycle_time + Duration::hours(i64::from(forecast_hour));
    Ok(valid_time.format("%Y-%m-%dT%H:%M:%SZ").to_string())
}

fn nearest_grid_point(grid: &LatLonGrid, point: GeoPoint) -> Option<PointTimeseriesGridPoint> {
    let mut best: Option<(usize, f64)> = None;
    for idx in 0..grid.shape.len() {
        let distance = geographic_distance_score(grid, idx, point);
        match best {
            Some((best_idx, best_distance))
                if distance > best_distance
                    || ((distance - best_distance).abs() <= 1.0e-12 && idx >= best_idx) => {}
            _ => best = Some((idx, distance)),
        }
    }
    let (grid_index, distance_score) = best?;
    Some(PointTimeseriesGridPoint {
        grid_index,
        i: grid_index % grid.shape.nx,
        j: grid_index / grid.shape.nx,
        lat_deg: f64::from(grid.lat_deg[grid_index]),
        lon_deg: f64::from(grid.lon_deg[grid_index]),
        distance_score,
    })
}

fn geographic_distance_score(grid: &LatLonGrid, idx: usize, point: GeoPoint) -> f64 {
    geographic_distance_score_for_lat_lon(
        f64::from(grid.lat_deg[idx]),
        f64::from(grid.lon_deg[idx]),
        point,
    )
}

fn geographic_distance_score_for_lat_lon(lat_deg: f64, lon_deg: f64, point: GeoPoint) -> f64 {
    let cos_lat = point.lat_deg.to_radians().cos().abs().max(0.2);
    let dlat = lat_deg - point.lat_deg;
    let dlon = normalized_longitude_delta(lon_deg - point.lon_deg) * cos_lat;
    dlat * dlat + dlon * dlon
}

fn normalized_longitude_delta(delta_deg: f64) -> f64 {
    let mut delta = delta_deg;
    while delta <= -180.0 {
        delta += 360.0;
    }
    while delta > 180.0 {
        delta -= 360.0;
    }
    delta
}

fn dewpoint_from_temperature_rh(temperature_c: f64, rh_pct: f64) -> Option<f64> {
    let vapor_pressure = saturation_vapor_pressure_hpa(temperature_c) * (rh_pct / 100.0);
    dewpoint_from_vapor_pressure_hpa(vapor_pressure)
}

fn dewpoint_from_vapor_pressure_hpa(vapor_pressure_hpa: f64) -> Option<f64> {
    if vapor_pressure_hpa <= 0.0 || !vapor_pressure_hpa.is_finite() {
        return None;
    }
    let ln_e = (vapor_pressure_hpa / 6.112).ln();
    Some((243.5 * ln_e) / (17.67 - ln_e))
}

fn mixing_ratio_from_dewpoint_c(pressure_hpa: f64, dewpoint_c: f64) -> f64 {
    let vapor_pressure_hpa = saturation_vapor_pressure_hpa(dewpoint_c);
    let e = vapor_pressure_hpa
        .max(1.0e-10)
        .min((pressure_hpa - 1.0e-6).max(1.0e-6));
    0.622 * e / (pressure_hpa - e).max(1.0e-6)
}

fn saturation_vapor_pressure_hpa(temp_c: f64) -> f64 {
    6.112 * ((17.67 * temp_c) / (temp_c + 243.5)).exp()
}

#[allow(dead_code)]
fn _assert_grid_shape_send(_: GridShape, _: VerticalSelector) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_variables_are_unique_and_parseable() {
        let variables = resolve_variables(&[]).unwrap();
        let unique = variables.iter().copied().collect::<BTreeSet<_>>();
        assert_eq!(variables.len(), unique.len());
        assert!(variables.contains(&PointVariable::PrecipHourlyMm));
        assert!(variables.contains(&PointVariable::Hdw));
    }

    #[test]
    fn hourly_precip_adds_previous_fetch_hour() {
        let hours = fetch_hours_for_variables(&[0, 1, 2], &[PointVariable::PrecipHourlyMm]);
        assert_eq!(hours, vec![0, 1, 2]);
        let hours = fetch_hours_for_variables(&[6], &[PointVariable::PrecipHourlyMm]);
        assert_eq!(hours, vec![5, 6]);
    }

    #[test]
    fn variable_aliases_cover_meteogram_terms() {
        assert_eq!(
            parse_variable("qpf_1h"),
            Some(PointVariable::PrecipHourlyMm)
        );
        assert_eq!(parse_variable("gust"), Some(PointVariable::WindGust10mMs));
        assert_eq!(parse_variable("hot-dry-windy"), Some(PointVariable::Hdw));
        assert_eq!(
            parse_variable("cloud_mid"),
            Some(PointVariable::MiddleCloudPct)
        );
    }

    #[test]
    #[ignore]
    fn hrrr_point_timeseries_smoke_from_env() {
        let date = std::env::var("RUSTWX_POINT_TS_DATE")
            .expect("set RUSTWX_POINT_TS_DATE=YYYYMMDD for smoke test");
        let cycle = std::env::var("RUSTWX_POINT_TS_CYCLE")
            .expect("set RUSTWX_POINT_TS_CYCLE=HH for smoke test")
            .parse::<u8>()
            .expect("cycle must be an integer hour");
        let source = std::env::var("RUSTWX_POINT_TS_SOURCE")
            .unwrap_or_else(|_| "aws".to_string())
            .parse::<SourceId>()
            .expect("source must parse");
        let cache_root = std::env::var("RUSTWX_POINT_TS_CACHE")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("target/point-timeseries-smoke-cache"));
        let request = PointTimeseriesRequest {
            model: ModelId::Hrrr,
            date_yyyymmdd: date,
            cycle_override_utc: Some(cycle),
            source,
            point: GeoPoint::new(40.802, -124.164),
            forecast_hours: vec![0, 1],
            variables: vec![
                "temperature_2m_c".to_string(),
                "relative_humidity_2m_pct".to_string(),
                "wind_speed_10m_ms".to_string(),
                "wind_gust_10m_ms".to_string(),
                "mslp_hpa".to_string(),
                "cloud_low_pct".to_string(),
                "precip_hourly_mm".to_string(),
                "hdw".to_string(),
            ],
            cache_root,
            use_cache: true,
            method: FieldPointSampleMethod::Nearest,
        };
        let report = sample_point_timeseries(&request).expect("point timeseries smoke should run");
        assert_eq!(report.hours.len(), 2);
        assert!(
            report.hours.iter().any(|hour| {
                hour.values
                    .get("temperature_2m_c")
                    .and_then(|value| *value)
                    .is_some()
            }),
            "at least one hour should have a temperature sample"
        );
    }
}
