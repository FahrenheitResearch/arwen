use crate::cache::{load_bincode, store_bincode};
use crate::direct::build_projected_map_with_projection;
use crate::shared_context::PreparedProjectedContext;
use grib_core::grib2::{
    Grib2File, Grib2Message, GridDefinition, flip_rows, grid_latlon,
    unpack_message_normalized as unpack_message_scan_normalized,
    unpack_message_scan_normalized_row_window,
};
use rayon::prelude::*;
use rustwx_calc::{GridShape as CalcGridShape, VolumeShape};
use rustwx_core::{
    CanonicalBundleDescriptor, CanonicalDataFamily, CycleSpec, GridProjection, GridShape,
    LatLonGrid, ModelId, RustwxError, SourceId,
};
use rustwx_io::{
    CachedFetchResult, FetchRequest, artifact_cache_dir, grid_projection_from_grib2_grid,
};
use rustwx_models::{
    LatestRun, ResolvedCanonicalBundleProduct, latest_available_run_at_forecast_hour,
    latest_available_run_for_products_at_forecast_hour, resolve_canonical_bundle_product,
};
#[cfg(test)]
use rustwx_render::ProjectedExtent;
use rustwx_render::map_frame_aspect_ratio;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::Instant;

const GEOPOTENTIAL_M2S2_TO_M: f64 = 1.0 / 9.806_65;
const MAX_DECODE_CACHE_WRITE_BYTES: usize = 512 * 1024 * 1024;
const PRESSURE_OPTIONAL_FIELDS_ENV: &str = "RUSTWX_PRESSURE_OPTIONAL_FIELDS";

mod crop;
mod fetch;

pub use crop::{
    CroppedHeavyDomain, GridCrop, ProjectedGridIntersection, classify_projected_grid_intersection,
    crop_heavy_domain, crop_heavy_domain_for_projected_extent, crop_heavy_domain_with,
    crop_latlon_grid, crop_values_f32, crop_values_f64, grid_crop_for_bounds,
};
use crop::{crop_2d_values, crop_rect_for_layout, cropped_decode_cache_path};
pub(crate) use fetch::{
    bundle_fetch_variable_patterns, fetch_family_file, fetch_family_file_with_patterns,
};
use fetch::{fetch_surface_pressure_files_parallel, merge_variable_patterns, thermo_bundles};
#[cfg(test)]
use fetch::{
    hrrr_pressure_analysis_fetch_patterns, pressure_analysis_fetch_patterns,
    surface_analysis_fetch_patterns,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurfaceFields {
    pub lat: Vec<f64>,
    pub lon: Vec<f64>,
    pub nx: usize,
    pub ny: usize,
    pub projection: Option<GridProjection>,
    pub psfc_pa: Vec<f64>,
    pub orog_m: Vec<f64>,
    pub orog_is_proxy: bool,
    pub t2_k: Vec<f64>,
    pub q2_kgkg: Vec<f64>,
    pub u10_ms: Vec<f64>,
    pub v10_ms: Vec<f64>,
    pub native_sbcape_jkg: Option<Vec<f64>>,
    pub native_mlcape_jkg: Option<Vec<f64>>,
    pub native_mucape_jkg: Option<Vec<f64>>,
    pub native_pblh_m: Option<Vec<f64>>,
}

impl SurfaceFields {
    pub fn core_grid(&self) -> Result<LatLonGrid, RustwxError> {
        LatLonGrid::new(
            GridShape::new(self.nx, self.ny)?,
            self.lat.iter().map(|&v| v as f32).collect(),
            self.lon.iter().map(|&v| v as f32).collect(),
        )
    }

    pub fn decoded_bytes_estimate(&self) -> usize {
        let len = self.lat.len();
        let required_f64_fields = 8usize;
        let optional_f64_fields = [
            self.native_sbcape_jkg.as_ref(),
            self.native_mlcape_jkg.as_ref(),
            self.native_mucape_jkg.as_ref(),
            self.native_pblh_m.as_ref(),
        ]
        .into_iter()
        .filter(|field| field.is_some())
        .count();
        len * (required_f64_fields + optional_f64_fields) * std::mem::size_of::<f64>()
            + std::mem::size_of::<bool>()
    }
}

#[derive(Debug, Clone)]
pub(crate) struct SurfaceGridLayout {
    pub lat: Vec<f64>,
    pub lon: Vec<f64>,
    pub nx: usize,
    pub ny: usize,
    pub projection: Option<GridProjection>,
    pub longitude_row_wraps: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PressureFields {
    pub pressure_levels_hpa: Vec<f64>,
    pub pressure_3d_pa: Option<Vec<f64>>,
    pub temperature_c_3d: Vec<f64>,
    pub qvapor_kgkg_3d: Vec<f64>,
    pub u_ms_3d: Vec<f64>,
    pub v_ms_3d: Vec<f64>,
    pub gh_m_3d: Vec<f64>,
    #[serde(default)]
    pub omega_pa_s_3d: Option<Vec<f64>>,
    #[serde(default)]
    pub absolute_vorticity_s_3d: Option<Vec<f64>>,
    #[serde(default)]
    pub cloud_liquid_kgkg_3d: Option<Vec<f64>>,
    #[serde(default)]
    pub cloud_ice_kgkg_3d: Option<Vec<f64>>,
    #[serde(default)]
    pub rain_kgkg_3d: Option<Vec<f64>>,
    #[serde(default)]
    pub snow_kgkg_3d: Option<Vec<f64>>,
    #[serde(default)]
    pub graupel_kgkg_3d: Option<Vec<f64>>,
}

impl PressureFields {
    pub fn decoded_bytes_estimate(&self) -> usize {
        let level_count = self.pressure_levels_hpa.len();
        let volume_len = self.temperature_c_3d.len();
        let pressure_3d_len = self
            .pressure_3d_pa
            .as_ref()
            .map(|values| values.len())
            .unwrap_or(0);
        let optional_volume_len: usize = [
            self.omega_pa_s_3d.as_ref(),
            self.absolute_vorticity_s_3d.as_ref(),
            self.cloud_liquid_kgkg_3d.as_ref(),
            self.cloud_ice_kgkg_3d.as_ref(),
            self.rain_kgkg_3d.as_ref(),
            self.snow_kgkg_3d.as_ref(),
            self.graupel_kgkg_3d.as_ref(),
        ]
        .into_iter()
        .map(|values| values.map(Vec::len).unwrap_or(0))
        .sum();
        level_count * std::mem::size_of::<f64>()
            + (volume_len * 5usize + pressure_3d_len + optional_volume_len)
                * std::mem::size_of::<f64>()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FetchRuntimeInfo {
    pub planned_bundle: CanonicalBundleDescriptor,
    pub planned_family: CanonicalDataFamily,
    pub planned_product: String,
    pub resolved_native_product: String,
    pub fetched_product: String,
    pub requested_source: SourceId,
    pub resolved_source: SourceId,
    pub resolved_url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharedTiming {
    pub fetch_surface_ms: u128,
    pub fetch_pressure_ms: u128,
    pub decode_surface_ms: u128,
    pub decode_pressure_ms: u128,
    pub fetch_surface_cache_hit: bool,
    pub fetch_pressure_cache_hit: bool,
    pub decode_surface_cache_hit: bool,
    pub decode_pressure_cache_hit: bool,
    pub surface_fetch: FetchRuntimeInfo,
    pub pressure_fetch: FetchRuntimeInfo,
}

#[derive(Debug, Clone)]
pub struct CachedDecode<T> {
    pub value: T,
    pub cache_hit: bool,
    pub path: PathBuf,
}

#[derive(Debug, Clone)]
pub struct FetchedModelFile {
    pub request: FetchRequest,
    pub fetched: CachedFetchResult,
    pub bytes: Vec<u8>,
}

impl FetchedModelFile {
    pub fn runtime_info(
        &self,
        planned_bundle: &ResolvedCanonicalBundleProduct,
    ) -> FetchRuntimeInfo {
        FetchRuntimeInfo {
            planned_bundle: planned_bundle.bundle,
            planned_family: planned_bundle.family,
            planned_product: planned_bundle.native_product.clone(),
            resolved_native_product: planned_bundle.native_product.clone(),
            fetched_product: self.request.request.product.clone(),
            requested_source: self
                .request
                .source_override
                .unwrap_or(self.fetched.result.source),
            resolved_source: self.fetched.result.source,
            resolved_url: self.fetched.result.url.clone(),
        }
    }
}

#[derive(Debug)]
pub struct LoadedModelTimestep {
    pub latest: LatestRun,
    pub model: ModelId,
    pub surface_file: FetchedModelFile,
    pub pressure_file: FetchedModelFile,
    pub surface_decode: CachedDecode<SurfaceFields>,
    pub pressure_decode: CachedDecode<PressureFields>,
    pub grid: LatLonGrid,
    pub shared_timing: SharedTiming,
}

#[derive(Debug, Clone)]
pub struct PreparedHeavyVolume {
    pub grid: CalcGridShape,
    pub shape: VolumeShape,
    pub pressure_levels_pa: Vec<f64>,
    pub pressure_3d_pa: Option<Vec<f64>>,
    pub height_agl_3d: Vec<f64>,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct PreparedHeavyVolumeTiming {
    pub prepare_height_agl_ms: u128,
    pub broadcast_pressure_ms: u128,
    pub pressure_3d_bytes: usize,
}

#[derive(Debug, Clone)]
pub struct LoadedSurfaceGeometry {
    pub latest: LatestRun,
    pub model: ModelId,
    pub surface_bundle: ResolvedCanonicalBundleProduct,
    pub surface_file: FetchedModelFile,
    pub surface_decode: CachedDecode<SurfaceFields>,
    pub grid: LatLonGrid,
    pub fetch_ms: u128,
    pub decode_ms: u128,
}

pub fn resolve_model_run(
    model: ModelId,
    date: &str,
    cycle_override: Option<u8>,
    forecast_hour: u16,
    source: SourceId,
) -> Result<LatestRun, Box<dyn std::error::Error>> {
    match cycle_override {
        Some(hour) => Ok(LatestRun {
            model,
            cycle: CycleSpec::new(date, hour)?,
            source,
        }),
        None => Ok(latest_available_run_at_forecast_hour(
            model,
            Some(source),
            date,
            forecast_hour,
        )?),
    }
}

pub fn resolve_thermo_pair_run(
    model: ModelId,
    date: &str,
    cycle_override: Option<u8>,
    forecast_hour: u16,
    source: SourceId,
    surface_product_override: Option<&str>,
    pressure_product_override: Option<&str>,
) -> Result<LatestRun, Box<dyn std::error::Error>> {
    match cycle_override {
        Some(hour) => Ok(LatestRun {
            model,
            cycle: CycleSpec::new(date, hour)?,
            source,
        }),
        None => {
            let (surface_bundle, pressure_bundle) =
                thermo_bundles(model, surface_product_override, pressure_product_override);
            let required_products = [
                surface_bundle.native_product.as_str(),
                pressure_bundle.native_product.as_str(),
            ];
            Ok(latest_available_run_for_products_at_forecast_hour(
                model,
                Some(source),
                date,
                &required_products,
                forecast_hour,
            )?)
        }
    }
}

pub fn load_model_timestep_from_parts(
    model: ModelId,
    date_yyyymmdd: &str,
    cycle_override_utc: Option<u8>,
    forecast_hour: u16,
    source: SourceId,
    surface_product_override: Option<&str>,
    pressure_product_override: Option<&str>,
    cache_root: &Path,
    use_cache: bool,
) -> Result<LoadedModelTimestep, Box<dyn std::error::Error>> {
    let latest = resolve_model_run(
        model,
        date_yyyymmdd,
        cycle_override_utc,
        forecast_hour,
        source,
    )?;
    load_model_timestep_from_latest(
        latest,
        forecast_hour,
        surface_product_override,
        pressure_product_override,
        cache_root,
        use_cache,
    )
}

pub fn load_model_timestep_from_parts_cropped(
    model: ModelId,
    date_yyyymmdd: &str,
    cycle_override_utc: Option<u8>,
    forecast_hour: u16,
    source: SourceId,
    surface_product_override: Option<&str>,
    pressure_product_override: Option<&str>,
    cache_root: &Path,
    use_cache: bool,
    bounds: (f64, f64, f64, f64),
) -> Result<LoadedModelTimestep, Box<dyn std::error::Error>> {
    let latest = resolve_model_run(
        model,
        date_yyyymmdd,
        cycle_override_utc,
        forecast_hour,
        source,
    )?;
    load_model_timestep_from_latest_cropped(
        latest,
        forecast_hour,
        surface_product_override,
        pressure_product_override,
        cache_root,
        use_cache,
        bounds,
    )
}

pub fn load_surface_geometry_from_latest(
    latest: LatestRun,
    forecast_hour: u16,
    surface_product_override: Option<&str>,
    cache_root: &Path,
    use_cache: bool,
) -> Result<LoadedSurfaceGeometry, Box<dyn std::error::Error>> {
    let surface_bundle = resolve_canonical_bundle_product(
        latest.model,
        CanonicalBundleDescriptor::SurfaceAnalysis,
        surface_product_override,
    );
    let fetch_start = Instant::now();
    let surface_file = fetch_family_file(
        latest.model,
        latest.cycle.clone(),
        forecast_hour,
        latest.source,
        &surface_bundle,
        cache_root,
        use_cache,
    )?;
    let fetch_ms = fetch_start.elapsed().as_millis();
    let decode_start = Instant::now();
    let surface_decode = load_or_decode_surface(
        &decode_cache_path(cache_root, &surface_file.request, "surface"),
        surface_file.bytes.as_slice(),
        use_cache,
    )?;
    let decode_ms = decode_start.elapsed().as_millis();
    let grid = surface_decode.value.core_grid()?;
    let model = latest.model;
    Ok(LoadedSurfaceGeometry {
        latest,
        model,
        surface_bundle,
        surface_file,
        surface_decode,
        grid,
        fetch_ms,
        decode_ms,
    })
}

pub fn load_model_timestep_from_latest(
    latest: LatestRun,
    forecast_hour: u16,
    surface_product_override: Option<&str>,
    pressure_product_override: Option<&str>,
    cache_root: &Path,
    use_cache: bool,
) -> Result<LoadedModelTimestep, Box<dyn std::error::Error>> {
    let model = latest.model;
    let (surface_bundle, pressure_bundle) =
        thermo_bundles(model, surface_product_override, pressure_product_override);

    let ((mut surface_file, fetch_surface_ms), (mut pressure_file, fetch_pressure_ms)) =
        if surface_bundle.native_product == pressure_bundle.native_product {
            let fetch_start = Instant::now();
            let fetched = fetch_family_file_with_patterns(
                model,
                latest.cycle.clone(),
                forecast_hour,
                latest.source,
                &surface_bundle,
                merge_variable_patterns([
                    bundle_fetch_variable_patterns(
                        model,
                        surface_bundle.bundle,
                        &surface_bundle.native_product,
                    ),
                    bundle_fetch_variable_patterns(
                        model,
                        pressure_bundle.bundle,
                        &pressure_bundle.native_product,
                    ),
                ]),
                cache_root,
                use_cache,
            )?;
            let elapsed = fetch_start.elapsed().as_millis();
            ((fetched.clone(), elapsed), (fetched, elapsed))
        } else {
            let ((surface_result, fetch_surface_ms), (pressure_result, fetch_pressure_ms)) =
                fetch_surface_pressure_files_parallel(
                    model,
                    latest.cycle.clone(),
                    forecast_hour,
                    latest.source,
                    &surface_bundle,
                    &pressure_bundle,
                    cache_root,
                    use_cache,
                );
            let surface = surface_result?;
            let pressure = pressure_result?;
            ((surface, fetch_surface_ms), (pressure, fetch_pressure_ms))
        };

    let surface_cache_path = decode_cache_path(cache_root, &surface_file.request, "surface");
    let pressure_cache_path = decode_cache_path(
        cache_root,
        &pressure_file.request,
        pressure_decode_cache_name(),
    );
    let surface_bytes = surface_file.bytes.as_slice();
    let pressure_bytes = pressure_file.bytes.as_slice();
    let decode_surface_start = Instant::now();
    let surface_decode = load_or_decode_surface(&surface_cache_path, surface_bytes, use_cache)?;
    let decode_surface_ms = decode_surface_start.elapsed().as_millis();
    let decode_pressure_start = Instant::now();
    let (pressure_decode, pressure_shape) =
        load_or_decode_pressure_with_shape(&pressure_cache_path, pressure_bytes, use_cache)?;
    let decode_pressure_ms = decode_pressure_start.elapsed().as_millis();

    validate_pressure_decode_against_surface(
        &pressure_decode,
        pressure_shape,
        surface_decode.value.nx,
        surface_decode.value.ny,
    )?;
    surface_file.bytes.clear();
    surface_file.bytes.shrink_to_fit();
    pressure_file.bytes.clear();
    pressure_file.bytes.shrink_to_fit();
    let grid = surface_decode.value.core_grid()?;
    let surface_fetch = surface_file.runtime_info(&surface_bundle);
    let pressure_fetch = pressure_file.runtime_info(&pressure_bundle);

    Ok(LoadedModelTimestep {
        latest,
        model,
        surface_file,
        pressure_file,
        surface_decode,
        pressure_decode,
        grid,
        shared_timing: SharedTiming {
            fetch_surface_ms,
            fetch_pressure_ms,
            decode_surface_ms,
            decode_pressure_ms,
            fetch_surface_cache_hit: false,
            fetch_pressure_cache_hit: false,
            decode_surface_cache_hit: false,
            decode_pressure_cache_hit: false,
            surface_fetch,
            pressure_fetch,
        },
    }
    .with_cache_flags())
}

pub fn load_model_timestep_from_latest_cropped(
    latest: LatestRun,
    forecast_hour: u16,
    surface_product_override: Option<&str>,
    pressure_product_override: Option<&str>,
    cache_root: &Path,
    use_cache: bool,
    bounds: (f64, f64, f64, f64),
) -> Result<LoadedModelTimestep, Box<dyn std::error::Error>> {
    let model = latest.model;
    let (surface_bundle, pressure_bundle) =
        thermo_bundles(model, surface_product_override, pressure_product_override);

    let ((mut surface_file, fetch_surface_ms), (mut pressure_file, fetch_pressure_ms)) =
        if surface_bundle.native_product == pressure_bundle.native_product {
            let fetch_start = Instant::now();
            let fetched = fetch_family_file_with_patterns(
                model,
                latest.cycle.clone(),
                forecast_hour,
                latest.source,
                &surface_bundle,
                merge_variable_patterns([
                    bundle_fetch_variable_patterns(
                        model,
                        surface_bundle.bundle,
                        &surface_bundle.native_product,
                    ),
                    bundle_fetch_variable_patterns(
                        model,
                        pressure_bundle.bundle,
                        &pressure_bundle.native_product,
                    ),
                ]),
                cache_root,
                use_cache,
            )?;
            let elapsed = fetch_start.elapsed().as_millis();
            ((fetched.clone(), elapsed), (fetched, elapsed))
        } else {
            let ((surface_result, fetch_surface_ms), (pressure_result, fetch_pressure_ms)) =
                fetch_surface_pressure_files_parallel(
                    model,
                    latest.cycle.clone(),
                    forecast_hour,
                    latest.source,
                    &surface_bundle,
                    &pressure_bundle,
                    cache_root,
                    use_cache,
                );
            let surface = surface_result?;
            let pressure = pressure_result?;
            ((surface, fetch_surface_ms), (pressure, fetch_pressure_ms))
        };

    let surface_layout = decode_surface_grid(surface_file.bytes.as_slice())?;
    let crop = crop_rect_for_layout(&surface_layout, bounds)?
        .ok_or("requested cropped load produced an empty domain")?;
    let surface_cache_path =
        cropped_decode_cache_path(cache_root, &surface_file.request, "surface", crop);
    let pressure_cache_path = cropped_decode_cache_path(
        cache_root,
        &pressure_file.request,
        pressure_decode_cache_name(),
        crop,
    );

    let decode_surface_start = Instant::now();
    let surface_decode = load_or_decode_surface_cropped(
        &surface_cache_path,
        surface_file.bytes.as_slice(),
        use_cache,
        crop,
    )?;
    let decode_surface_ms = decode_surface_start.elapsed().as_millis();

    let decode_pressure_start = Instant::now();
    let (pressure_decode, pressure_shape) = load_or_decode_pressure_cropped_with_shape(
        &pressure_cache_path,
        pressure_file.bytes.as_slice(),
        use_cache,
        crop,
    )?;
    let decode_pressure_ms = decode_pressure_start.elapsed().as_millis();

    validate_pressure_decode_against_surface(
        &pressure_decode,
        pressure_shape,
        surface_decode.value.nx,
        surface_decode.value.ny,
    )?;
    surface_file.bytes.clear();
    surface_file.bytes.shrink_to_fit();
    pressure_file.bytes.clear();
    pressure_file.bytes.shrink_to_fit();
    let grid = surface_decode.value.core_grid()?;
    let surface_fetch = surface_file.runtime_info(&surface_bundle);
    let pressure_fetch = pressure_file.runtime_info(&pressure_bundle);

    Ok(LoadedModelTimestep {
        latest,
        model,
        surface_file,
        pressure_file,
        surface_decode,
        pressure_decode,
        grid,
        shared_timing: SharedTiming {
            fetch_surface_ms,
            fetch_pressure_ms,
            decode_surface_ms,
            decode_pressure_ms,
            fetch_surface_cache_hit: false,
            fetch_pressure_cache_hit: false,
            decode_surface_cache_hit: false,
            decode_pressure_cache_hit: false,
            surface_fetch,
            pressure_fetch,
        },
    }
    .with_cache_flags())
}

pub fn build_projected_maps_for_sizes(
    surface: &SurfaceFields,
    bounds: (f64, f64, f64, f64),
    sizes: &[(u32, u32)],
) -> Result<PreparedProjectedContext, Box<dyn std::error::Error>> {
    let mut context = PreparedProjectedContext::new();
    for &(width, height) in sizes {
        if width == 0 || height == 0 || context.contains_size(width, height) {
            continue;
        }
        let projected = build_projected_map_with_projection(
            &surface
                .lat
                .iter()
                .copied()
                .map(|v| v as f32)
                .collect::<Vec<_>>(),
            &surface
                .lon
                .iter()
                .copied()
                .map(|v| v as f32)
                .collect::<Vec<_>>(),
            surface.projection.as_ref(),
            bounds,
            map_frame_aspect_ratio(width, height, true, true),
        )?;
        context.insert(width, height, projected);
    }
    Ok(context)
}

pub fn prepare_heavy_volume(
    surface: &SurfaceFields,
    pressure: &PressureFields,
    include_pressure_3d: bool,
) -> Result<PreparedHeavyVolume, Box<dyn std::error::Error>> {
    let (prepared, _) = prepare_heavy_volume_timed(surface, pressure, include_pressure_3d)?;
    Ok(prepared)
}

pub fn prepare_heavy_volume_timed(
    surface: &SurfaceFields,
    pressure: &PressureFields,
    include_pressure_3d: bool,
) -> Result<(PreparedHeavyVolume, PreparedHeavyVolumeTiming), Box<dyn std::error::Error>> {
    let grid = CalcGridShape::new(surface.nx, surface.ny)?;
    let shape = VolumeShape::new(grid, pressure.pressure_levels_hpa.len())?;
    let pressure_levels_pa = pressure
        .pressure_levels_hpa
        .iter()
        .map(|level_hpa| level_hpa * 100.0)
        .collect::<Vec<_>>();
    let height_agl_start = Instant::now();
    let height_agl_3d = compute_height_agl_3d(surface, pressure, grid, shape);
    let prepare_height_agl_ms = height_agl_start.elapsed().as_millis();
    let broadcast_start = Instant::now();
    let pressure_3d_pa = include_pressure_3d.then(|| {
        pressure
            .pressure_3d_pa
            .clone()
            .unwrap_or_else(|| broadcast_levels_pa(&pressure.pressure_levels_hpa, grid.len()))
    });
    let broadcast_pressure_ms = if include_pressure_3d {
        broadcast_start.elapsed().as_millis()
    } else {
        0
    };
    let pressure_3d_bytes = pressure_3d_pa
        .as_ref()
        .map(|values| values.len() * std::mem::size_of::<f64>())
        .unwrap_or(0);
    Ok((
        PreparedHeavyVolume {
            grid,
            shape,
            pressure_levels_pa,
            pressure_3d_pa,
            height_agl_3d,
        },
        PreparedHeavyVolumeTiming {
            prepare_height_agl_ms,
            broadcast_pressure_ms,
            pressure_3d_bytes,
        },
    ))
}

pub fn compute_height_agl_3d(
    surface: &SurfaceFields,
    pressure: &PressureFields,
    grid: CalcGridShape,
    shape: VolumeShape,
) -> Vec<f64> {
    let fallback_orog = surface
        .orog_is_proxy
        .then(|| proxy_orography_from_pressure(pressure, grid, shape));
    let mut height_agl_3d = pressure
        .gh_m_3d
        .iter()
        .enumerate()
        .map(|(idx, &value)| {
            let ij = idx % grid.len();
            let orog = fallback_orog
                .as_ref()
                .map(|values| values[ij])
                .unwrap_or(surface.orog_m[ij]);
            (value - orog).max(0.0)
        })
        .collect::<Vec<_>>();

    for k in 1..shape.nz {
        let level_offset = k * grid.len();
        let prev_offset = (k - 1) * grid.len();
        for ij in 0..grid.len() {
            let min_height = height_agl_3d[prev_offset + ij] + 1.0;
            if height_agl_3d[level_offset + ij] < min_height {
                height_agl_3d[level_offset + ij] = min_height;
            }
        }
    }

    height_agl_3d
}

fn proxy_orography_from_pressure(
    pressure: &PressureFields,
    grid: CalcGridShape,
    shape: VolumeShape,
) -> Vec<f64> {
    let mut proxy = vec![f64::INFINITY; grid.len()];
    for k in 0..shape.nz {
        let level_offset = k * grid.len();
        for ij in 0..grid.len() {
            proxy[ij] = proxy[ij].min(pressure.gh_m_3d[level_offset + ij]);
        }
    }
    proxy
        .into_iter()
        .map(|value| if value.is_finite() { value } else { 0.0 })
        .collect()
}

pub fn broadcast_levels_pa(levels_hpa: &[f64], n2d: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(levels_hpa.len() * n2d);
    for level in levels_hpa {
        out.extend(std::iter::repeat_n(*level * 100.0, n2d));
    }
    out
}

pub(crate) fn decode_cache_path(cache_root: &Path, fetch: &FetchRequest, name: &str) -> PathBuf {
    artifact_cache_dir(cache_root, fetch)
        .join("decoded")
        .join(format!("{name}.bin"))
}

pub(crate) fn load_or_decode_surface(
    path: &Path,
    bytes: &[u8],
    use_cache: bool,
) -> Result<CachedDecode<SurfaceFields>, Box<dyn std::error::Error>> {
    if use_cache {
        if let Some(cached) = load_bincode::<SurfaceFields>(path)? {
            return Ok(CachedDecode {
                value: cached,
                cache_hit: true,
                path: path.to_path_buf(),
            });
        }
    }
    let decoded = decode_surface(bytes)?;
    if use_cache && decoded.decoded_bytes_estimate() <= MAX_DECODE_CACHE_WRITE_BYTES {
        store_bincode(path, &decoded)?;
    }
    Ok(CachedDecode {
        value: decoded,
        cache_hit: false,
        path: path.to_path_buf(),
    })
}

pub(crate) fn load_or_decode_surface_from_file(
    path: &Path,
    file: &FetchedModelFile,
    use_cache: bool,
) -> Result<CachedDecode<SurfaceFields>, Box<dyn std::error::Error>> {
    load_or_decode_surface(path, file.bytes.as_slice(), use_cache)
}

pub(crate) fn load_or_decode_pressure_with_shape(
    path: &Path,
    bytes: &[u8],
    use_cache: bool,
) -> Result<(CachedDecode<PressureFields>, Option<(usize, usize)>), Box<dyn std::error::Error>> {
    if use_cache {
        if let Some(cached) = load_bincode::<PressureFields>(path)? {
            return Ok((
                CachedDecode {
                    value: cached,
                    cache_hit: true,
                    path: path.to_path_buf(),
                },
                None,
            ));
        }
    }
    let (decoded, nx, ny) = decode_pressure_with_shape(bytes)?;
    if use_cache && decoded.decoded_bytes_estimate() <= MAX_DECODE_CACHE_WRITE_BYTES {
        store_bincode(path, &decoded)?;
    }
    Ok((
        CachedDecode {
            value: decoded,
            cache_hit: false,
            path: path.to_path_buf(),
        },
        Some((nx, ny)),
    ))
}

pub(crate) fn load_or_decode_pressure_from_file_with_shape(
    path: &Path,
    file: &FetchedModelFile,
    use_cache: bool,
) -> Result<(CachedDecode<PressureFields>, Option<(usize, usize)>), Box<dyn std::error::Error>> {
    load_or_decode_pressure_with_shape(path, file.bytes.as_slice(), use_cache)
}

pub(crate) fn decode_surface_grid(
    bytes: &[u8],
) -> Result<SurfaceGridLayout, Box<dyn std::error::Error>> {
    let file = Grib2File::from_bytes(bytes)?;
    let sample = file
        .messages
        .first()
        .ok_or("surface family GRIB had no messages")?;
    Ok(decode_surface_grid_from_sample(sample))
}

pub(crate) fn load_or_decode_surface_cropped(
    path: &Path,
    bytes: &[u8],
    use_cache: bool,
    crop: GridCrop,
) -> Result<CachedDecode<SurfaceFields>, Box<dyn std::error::Error>> {
    if use_cache {
        if let Some(cached) = load_bincode::<SurfaceFields>(path)? {
            return Ok(CachedDecode {
                value: cached,
                cache_hit: true,
                path: path.to_path_buf(),
            });
        }
    }
    let decoded = decode_surface_cropped(bytes, crop)?;
    if use_cache && decoded.decoded_bytes_estimate() <= MAX_DECODE_CACHE_WRITE_BYTES {
        store_bincode(path, &decoded)?;
    }
    Ok(CachedDecode {
        value: decoded,
        cache_hit: false,
        path: path.to_path_buf(),
    })
}

pub(crate) fn load_or_decode_pressure_cropped_with_shape(
    path: &Path,
    bytes: &[u8],
    use_cache: bool,
    crop: GridCrop,
) -> Result<(CachedDecode<PressureFields>, Option<(usize, usize)>), Box<dyn std::error::Error>> {
    if use_cache {
        if let Some(cached) = load_bincode::<PressureFields>(path)? {
            return Ok((
                CachedDecode {
                    value: cached,
                    cache_hit: true,
                    path: path.to_path_buf(),
                },
                None,
            ));
        }
    }
    let (decoded, nx, ny) = decode_pressure_cropped_with_shape(bytes, crop)?;
    if use_cache && decoded.decoded_bytes_estimate() <= MAX_DECODE_CACHE_WRITE_BYTES {
        store_bincode(path, &decoded)?;
    }
    Ok((
        CachedDecode {
            value: decoded,
            cache_hit: false,
            path: path.to_path_buf(),
        },
        Some((nx, ny)),
    ))
}

fn decode_surface(bytes: &[u8]) -> Result<SurfaceFields, Box<dyn std::error::Error>> {
    let file = Grib2File::from_bytes(bytes)?;
    let sample = file
        .messages
        .first()
        .ok_or("surface family GRIB had no messages")?;
    let SurfaceGridLayout {
        lat,
        lon,
        nx,
        ny,
        projection,
        longitude_row_wraps: _,
    } = decode_surface_grid_from_sample(sample);

    let psfc_pa = unpack_message_normalized(find_message(
        &file.messages,
        &[(0, 3, 0, 1, Some(0.0)), (0, 3, 0, 1, None)],
    )?)?;
    let (orog_m, orog_is_proxy) = match decode_orography(&file.messages) {
        Ok(values) => (values, false),
        Err(_) => (vec![0.0; nx * ny], true),
    };
    let t2_k =
        unpack_message_normalized(find_message(&file.messages, &[(0, 0, 0, 103, Some(2.0))])?)?;
    let q2_kgkg = decode_surface_mixing_ratio(&file.messages, &psfc_pa, &t2_k)?;
    let u10_ms =
        unpack_message_normalized(find_message(&file.messages, &[(0, 2, 2, 103, Some(10.0))])?)?;
    let v10_ms =
        unpack_message_normalized(find_message(&file.messages, &[(0, 2, 3, 103, Some(10.0))])?)?;
    let native_sbcape_jkg = decode_optional_native_cape(&file.messages, NativeCapeLayer::Surface)?;
    let native_mlcape_jkg =
        decode_optional_native_cape(&file.messages, NativeCapeLayer::MixedLayer)?;
    let native_mucape_jkg =
        decode_optional_native_cape(&file.messages, NativeCapeLayer::MostUnstable)?;
    let native_pblh_m = decode_optional_native_pblh(&file.messages)?;

    Ok(SurfaceFields {
        lat,
        lon,
        nx,
        ny,
        projection,
        psfc_pa,
        orog_m,
        orog_is_proxy,
        t2_k,
        q2_kgkg,
        u10_ms,
        v10_ms,
        native_sbcape_jkg,
        native_mlcape_jkg,
        native_mucape_jkg,
        native_pblh_m,
    })
}

fn decode_surface_cropped(
    bytes: &[u8],
    crop: GridCrop,
) -> Result<SurfaceFields, Box<dyn std::error::Error>> {
    let file = Grib2File::from_bytes(bytes)?;
    let sample = file
        .messages
        .first()
        .ok_or("surface family GRIB had no messages")?;
    let SurfaceGridLayout {
        lat,
        lon,
        nx,
        ny: _,
        projection,
        longitude_row_wraps,
    } = decode_surface_grid_from_sample(sample);

    let psfc_pa = unpack_message_normalized_cropped(
        find_message(
            &file.messages,
            &[(0, 3, 0, 1, Some(0.0)), (0, 3, 0, 1, None)],
        )?,
        nx,
        crop,
        &longitude_row_wraps,
    )?;
    let (orog_m, orog_is_proxy) =
        match decode_orography_cropped(&file.messages, nx, crop, &longitude_row_wraps) {
            Ok(values) => (values, false),
            Err(_) => (vec![0.0; crop.width() * crop.height()], true),
        };
    let t2_k = unpack_message_normalized_cropped(
        find_message(&file.messages, &[(0, 0, 0, 103, Some(2.0))])?,
        nx,
        crop,
        &longitude_row_wraps,
    )?;
    let q2_kgkg = decode_surface_mixing_ratio_cropped(
        &file.messages,
        &psfc_pa,
        &t2_k,
        nx,
        crop,
        &longitude_row_wraps,
    )?;
    let u10_ms = unpack_message_normalized_cropped(
        find_message(&file.messages, &[(0, 2, 2, 103, Some(10.0))])?,
        nx,
        crop,
        &longitude_row_wraps,
    )?;
    let v10_ms = unpack_message_normalized_cropped(
        find_message(&file.messages, &[(0, 2, 3, 103, Some(10.0))])?,
        nx,
        crop,
        &longitude_row_wraps,
    )?;
    let native_sbcape_jkg = decode_optional_native_cape_cropped(
        &file.messages,
        NativeCapeLayer::Surface,
        nx,
        crop,
        &longitude_row_wraps,
    )?;
    let native_mlcape_jkg = decode_optional_native_cape_cropped(
        &file.messages,
        NativeCapeLayer::MixedLayer,
        nx,
        crop,
        &longitude_row_wraps,
    )?;
    let native_mucape_jkg = decode_optional_native_cape_cropped(
        &file.messages,
        NativeCapeLayer::MostUnstable,
        nx,
        crop,
        &longitude_row_wraps,
    )?;
    let native_pblh_m =
        decode_optional_native_pblh_cropped(&file.messages, nx, crop, &longitude_row_wraps)?;

    Ok(SurfaceFields {
        lat: crop_2d_values(&lat, nx, crop),
        lon: crop_2d_values(&lon, nx, crop),
        nx: crop.width(),
        ny: crop.height(),
        projection,
        psfc_pa,
        orog_m,
        orog_is_proxy,
        t2_k,
        q2_kgkg,
        u10_ms,
        v10_ms,
        native_sbcape_jkg,
        native_mlcape_jkg,
        native_mucape_jkg,
        native_pblh_m,
    })
}

fn decode_pressure_with_shape(
    bytes: &[u8],
) -> Result<(PressureFields, usize, usize), Box<dyn std::error::Error>> {
    decode_pressure_with_shape_opts(bytes, pressure_optional_decode_enabled())
}

fn decode_pressure_with_shape_opts(
    bytes: &[u8],
    include_optional: bool,
) -> Result<(PressureFields, usize, usize), Box<dyn std::error::Error>> {
    let file = Grib2File::from_bytes(bytes)?;
    decode_pressure_file_with_shape_opts(file, include_optional)
}

/// Owned-bytes variant for the store-ingest lane: the raw pressure GRIB
/// bytes are freed as soon as the parser has its own copy of every
/// message, instead of staying resident through the whole decode. Same
/// parse, same decode, byte-identical `PressureFields`.
fn decode_pressure_with_shape_opts_owned(
    bytes: Vec<u8>,
    include_optional: bool,
) -> Result<(PressureFields, usize, usize), Box<dyn std::error::Error>> {
    let file = Grib2File::from_bytes(&bytes)?;
    drop(bytes);
    decode_pressure_file_with_shape_opts(file, include_optional)
}

fn decode_pressure_file_with_shape_opts(
    mut file: Grib2File,
    include_optional: bool,
) -> Result<(PressureFields, usize, usize), Box<dyn std::error::Error>> {
    let (nx, ny) = pressure_grid_shape_from_messages(&file.messages)?;
    let omega = if include_optional {
        collect_optional_levels(&file.messages, 0, 2, 8, 100)?
    } else {
        None
    };
    let absolute_vorticity = if include_optional {
        collect_optional_levels(&file.messages, 0, 2, 10, 100)?
    } else {
        None
    };
    let cloud_liquid = if include_optional {
        collect_optional_levels(&file.messages, 0, 1, 22, 100)?
    } else {
        None
    };
    let cloud_ice = if include_optional {
        collect_optional_levels_any(&file.messages, &[(0, 1, 82), (0, 1, 23)], 100)?
    } else {
        None
    };
    let rain = if include_optional {
        collect_optional_levels(&file.messages, 0, 1, 24, 100)?
    } else {
        None
    };
    let snow = if include_optional {
        collect_optional_levels(&file.messages, 0, 1, 25, 100)?
    } else {
        None
    };
    let graupel = if include_optional {
        collect_optional_levels_any(&file.messages, &[(0, 1, 32), (0, 1, 74)], 100)?
    } else {
        None
    };

    // Every optional volume is decoded; from here on only the five
    // required variables' messages are touched. Free the raw bytes of
    // everything else (~60% of an HRRR prs file's parsed copy).
    strip_unneeded_required_var_raws(&mut file.messages);

    // The five required volumes decode by direct write: each matched
    // message unpacks straight into its slot of a preallocated flat
    // volume, so the per-level collections (one full extra copy of the
    // f64 volumes, the measured global ingest peak) never exist. Level
    // selection, fallback order and failure semantics replicate the
    // collect-then-flatten lane exactly; see `decode_required_volumes`.
    let decoded = decode_required_volumes(&file, nx, ny)?;
    let levels = decoded.common_levels;
    // The parsed Grib2File (each message owns a copy of its raw bytes) is
    // not consulted past this point; free it before the optional flatten.
    drop(file);

    let expected = nx * ny;
    let flatten_optional =
        |records: Option<Vec<(f64, Vec<f64>)>>| -> Result<Option<Vec<f64>>, Box<dyn std::error::Error>> {
            let Some(records) = records else {
                return Ok(None);
            };
            let mut out = Vec::with_capacity(levels.len() * expected);
            for &level in &levels {
                let Some(values) = level_values(&records, level) else {
                    return Ok(None);
                };
                if values.len() != expected {
                    return Ok(None);
                }
                out.extend_from_slice(values);
            }
            Ok(Some(out))
        };

    Ok((
        PressureFields {
            pressure_levels_hpa: levels
                .iter()
                .copied()
                .map(normalize_pressure_level_hpa)
                .collect(),
            pressure_3d_pa: None,
            temperature_c_3d: decoded.temperature_c_3d,
            qvapor_kgkg_3d: decoded.qvapor_kgkg_3d,
            u_ms_3d: decoded.u_ms_3d,
            v_ms_3d: decoded.v_ms_3d,
            gh_m_3d: decoded.gh_m_3d,
            omega_pa_s_3d: flatten_optional(omega)?,
            absolute_vorticity_s_3d: flatten_optional(absolute_vorticity)?,
            cloud_liquid_kgkg_3d: flatten_optional(cloud_liquid)?,
            cloud_ice_kgkg_3d: flatten_optional(cloud_ice)?,
            rain_kgkg_3d: flatten_optional(rain)?,
            snow_kgkg_3d: flatten_optional(snow)?,
            graupel_kgkg_3d: flatten_optional(graupel)?,
        },
        nx,
        ny,
    ))
}

fn decode_pressure_cropped_with_shape(
    bytes: &[u8],
    crop: GridCrop,
) -> Result<(PressureFields, usize, usize), Box<dyn std::error::Error>> {
    let file = Grib2File::from_bytes(bytes)?;
    let (nx, _ny) = pressure_grid_shape_from_messages(&file.messages)?;
    let longitude_row_wraps = normalized_longitude_row_wraps_from_messages(&file.messages)?;
    let temperature =
        collect_levels_cropped(&file.messages, 0, 0, 0, 100, nx, crop, &longitude_row_wraps)?;
    let u_wind =
        collect_levels_cropped(&file.messages, 0, 2, 2, 100, nx, crop, &longitude_row_wraps)?;
    let v_wind =
        collect_levels_cropped(&file.messages, 0, 2, 3, 100, nx, crop, &longitude_row_wraps)?;
    let gh = decode_height_levels_cropped(&file.messages, nx, crop, &longitude_row_wraps)?;
    let moisture = decode_pressure_mixing_ratio_levels_cropped(
        &file.messages,
        &temperature,
        nx,
        crop,
        &longitude_row_wraps,
    )?;
    let include_optional = pressure_optional_decode_enabled();
    let omega = if include_optional {
        collect_optional_levels_cropped(
            &file.messages,
            0,
            2,
            8,
            100,
            nx,
            crop,
            &longitude_row_wraps,
        )?
    } else {
        None
    };
    let absolute_vorticity = if include_optional {
        collect_optional_levels_cropped(
            &file.messages,
            0,
            2,
            10,
            100,
            nx,
            crop,
            &longitude_row_wraps,
        )?
    } else {
        None
    };
    let cloud_liquid = if include_optional {
        collect_optional_levels_cropped(
            &file.messages,
            0,
            1,
            22,
            100,
            nx,
            crop,
            &longitude_row_wraps,
        )?
    } else {
        None
    };
    let cloud_ice = if include_optional {
        collect_optional_levels_any_cropped(
            &file.messages,
            &[(0, 1, 82), (0, 1, 23)],
            100,
            nx,
            crop,
            &longitude_row_wraps,
        )?
    } else {
        None
    };
    let rain = if include_optional {
        collect_optional_levels_cropped(
            &file.messages,
            0,
            1,
            24,
            100,
            nx,
            crop,
            &longitude_row_wraps,
        )?
    } else {
        None
    };
    let snow = if include_optional {
        collect_optional_levels_cropped(
            &file.messages,
            0,
            1,
            25,
            100,
            nx,
            crop,
            &longitude_row_wraps,
        )?
    } else {
        None
    };
    let graupel = if include_optional {
        collect_optional_levels_any_cropped(
            &file.messages,
            &[(0, 1, 32), (0, 1, 74)],
            100,
            nx,
            crop,
            &longitude_row_wraps,
        )?
    } else {
        None
    };

    let levels = common_isobaric_levels(&temperature, &[&moisture, &u_wind, &v_wind, &gh]);
    if levels.is_empty() {
        return Err("pressure family had no common thermodynamic levels".into());
    }
    // See decode_pressure_with_shape_opts: free the parsed message copies
    // before the flatten pass.
    drop(file);

    let expected = crop.width() * crop.height();
    // Consuming flatten — identical values, per-variable collection drop
    // (see decode_pressure_with_shape_opts).
    let flatten = |records: Vec<(f64, Vec<f64>)>| -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        let mut out = Vec::with_capacity(levels.len() * expected);
        for &level in &levels {
            let values = level_values(&records, level)
                .ok_or_else(|| format!("missing aligned pressure level {level}"))?;
            if values.len() != expected {
                return Err("decoded cropped pressure field had unexpected grid size".into());
            }
            out.extend_from_slice(values);
        }
        Ok(out)
    };
    let flatten_optional =
        |records: Option<Vec<(f64, Vec<f64>)>>| -> Result<Option<Vec<f64>>, Box<dyn std::error::Error>> {
            let Some(records) = records else {
                return Ok(None);
            };
            let mut out = Vec::with_capacity(levels.len() * expected);
            for &level in &levels {
                let Some(values) = level_values(&records, level) else {
                    return Ok(None);
                };
                if values.len() != expected {
                    return Ok(None);
                }
                out.extend_from_slice(values);
            }
            Ok(Some(out))
        };

    let pressure_levels_hpa = levels
        .iter()
        .copied()
        .map(normalize_pressure_level_hpa)
        .collect();

    Ok((
        PressureFields {
            pressure_levels_hpa,
            pressure_3d_pa: None,
            temperature_c_3d: flatten(temperature)?
                .into_iter()
                .map(|value| value - 273.15)
                .collect(),
            qvapor_kgkg_3d: flatten(moisture)?,
            u_ms_3d: flatten(u_wind)?,
            v_ms_3d: flatten(v_wind)?,
            gh_m_3d: flatten(gh)?,
            omega_pa_s_3d: flatten_optional(omega)?,
            absolute_vorticity_s_3d: flatten_optional(absolute_vorticity)?,
            cloud_liquid_kgkg_3d: flatten_optional(cloud_liquid)?,
            cloud_ice_kgkg_3d: flatten_optional(cloud_ice)?,
            rain_kgkg_3d: flatten_optional(rain)?,
            snow_kgkg_3d: flatten_optional(snow)?,
            graupel_kgkg_3d: flatten_optional(graupel)?,
        },
        crop.width(),
        crop.height(),
    ))
}

fn decode_surface_grid_from_sample(sample: &Grib2Message) -> SurfaceGridLayout {
    let (mut lat_raw, mut lon_raw) = grid_latlon(&sample.grid);
    if sample.grid.scan_mode & 0x40 != 0 {
        flip_rows(
            &mut lat_raw,
            sample.grid.nx as usize,
            sample.grid.ny as usize,
        );
        flip_rows(
            &mut lon_raw,
            sample.grid.nx as usize,
            sample.grid.ny as usize,
        );
    }
    let longitude_row_wraps = normalize_longitude_rows(
        &mut lat_raw,
        &mut lon_raw,
        sample.grid.nx as usize,
        sample.grid.ny as usize,
    );
    SurfaceGridLayout {
        lat: lat_raw,
        lon: lon_raw
            .into_iter()
            .map(normalize_longitude)
            .collect::<Vec<_>>(),
        nx: sample.grid.nx as usize,
        ny: sample.grid.ny as usize,
        projection: grid_projection_from_grib2_grid(&sample.grid),
        longitude_row_wraps,
    }
}

fn unpack_message_normalized(
    message: &Grib2Message,
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let mut values = unpack_message_scan_normalized(message)?;
    rotate_values_to_normalized_longitude_rows(message, &mut values);
    Ok(values)
}

fn unpack_message_normalized_cropped(
    message: &Grib2Message,
    source_nx: usize,
    crop: GridCrop,
    longitude_row_wraps: &[usize],
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    if source_nx == message.grid.nx as usize {
        if let Ok(mut rows) =
            unpack_message_scan_normalized_row_window(message, crop.y_start, crop.y_end)
        {
            rotate_window_values_to_normalized_longitude_rows(
                &mut rows,
                source_nx,
                crop.y_start,
                crop.y_end,
                longitude_row_wraps,
            );
            return Ok(crop_window_x_values(&rows, source_nx, crop));
        }
    }

    let values = unpack_message_normalized(message)?;
    Ok(crop_2d_values(&values, source_nx, crop))
}

fn crop_window_x_values(values: &[f64], source_nx: usize, crop: GridCrop) -> Vec<f64> {
    let mut cropped = Vec::with_capacity(crop.width() * crop.height());
    for window_y in 0..crop.height() {
        let start = window_y * source_nx + crop.x_start;
        let end = window_y * source_nx + crop.x_end;
        cropped.extend_from_slice(&values[start..end]);
    }
    cropped
}

fn rotate_window_values_to_normalized_longitude_rows(
    values: &mut [f64],
    nx: usize,
    y_start: usize,
    y_end: usize,
    longitude_row_wraps: &[usize],
) {
    if nx == 0 || y_start > y_end || values.len() != nx * (y_end - y_start) {
        return;
    }
    for y in y_start..y_end {
        let wrap_idx = longitude_row_wraps.get(y).copied().unwrap_or(0) % nx;
        if wrap_idx == 0 {
            continue;
        }
        let row_start = (y - y_start) * nx;
        let row_end = row_start + nx;
        values[row_start..row_end].rotate_left(wrap_idx);
    }
}

fn rotate_values_to_normalized_longitude_rows(message: &Grib2Message, values: &mut [f64]) {
    let nx = message.grid.nx as usize;
    let ny = message.grid.ny as usize;
    if nx == 0 || ny == 0 || values.len() != nx * ny {
        return;
    }

    let (_lat_raw, mut lon_raw) = grid_latlon(&message.grid);
    if message.grid.scan_mode & 0x40 != 0 {
        flip_rows(&mut lon_raw, nx, ny);
    }
    for row in 0..ny {
        let start = row * nx;
        let end = start + nx;
        let lon_row = &mut lon_raw[start..end];
        for lon_value in lon_row.iter_mut() {
            *lon_value = normalize_longitude(*lon_value);
        }
        if let Some(wrap_idx) = first_longitude_wrap(lon_row) {
            values[start..end].rotate_left(wrap_idx);
        }
    }
}

fn decode_orography(messages: &[Grib2Message]) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    if let Ok(message) = find_message(messages, &[(0, 3, 5, 1, Some(0.0)), (0, 3, 5, 1, None)]) {
        return Ok(unpack_message_normalized(message)?);
    }
    if let Ok(message) = find_message(messages, &[(0, 3, 4, 1, Some(0.0)), (0, 3, 4, 1, None)]) {
        return Ok(unpack_message_normalized(message)?
            .into_iter()
            .map(|value| value * GEOPOTENTIAL_M2S2_TO_M)
            .collect());
    }
    Err("missing surface orography/geopotential-height field".into())
}

fn decode_orography_cropped(
    messages: &[Grib2Message],
    source_nx: usize,
    crop: GridCrop,
    longitude_row_wraps: &[usize],
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    if let Ok(message) = find_message(messages, &[(0, 3, 5, 1, Some(0.0)), (0, 3, 5, 1, None)]) {
        return Ok(unpack_message_normalized_cropped(
            message,
            source_nx,
            crop,
            longitude_row_wraps,
        )?);
    }
    if let Ok(message) = find_message(messages, &[(0, 3, 4, 1, Some(0.0)), (0, 3, 4, 1, None)]) {
        return Ok(unpack_message_normalized_cropped(
            message,
            source_nx,
            crop,
            longitude_row_wraps,
        )?
        .into_iter()
        .map(|value| value * GEOPOTENTIAL_M2S2_TO_M)
        .collect());
    }
    Err("missing surface orography/geopotential-height field".into())
}

fn decode_surface_mixing_ratio(
    messages: &[Grib2Message],
    psfc_pa: &[f64],
    t2_k: &[f64],
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    if let Ok(message) = find_message(messages, &[(0, 1, 0, 103, Some(2.0))]) {
        return Ok(q_to_mixing_ratio(&unpack_message_normalized(message)?));
    }
    if let Ok(message) = find_message(messages, &[(0, 0, 6, 103, Some(2.0))]) {
        let dewpoint_k = unpack_message_normalized(message)?;
        return Ok(psfc_pa
            .iter()
            .zip(dewpoint_k.iter())
            .map(|(&psfc, &td_k)| mixing_ratio_from_dewpoint_k(psfc / 100.0, td_k))
            .collect());
    }
    if let Ok(message) = find_message(messages, &[(0, 1, 1, 103, Some(2.0))]) {
        let rh_pct = unpack_message_normalized(message)?;
        return Ok(psfc_pa
            .iter()
            .zip(t2_k.iter())
            .zip(rh_pct.iter())
            .map(|((&psfc, &t_k), &rh)| mixing_ratio_from_relative_humidity(psfc / 100.0, t_k, rh))
            .collect());
    }
    Err("missing 2m specific humidity/dewpoint/RH field for surface thermodynamics".into())
}

fn decode_surface_mixing_ratio_cropped(
    messages: &[Grib2Message],
    psfc_pa: &[f64],
    t2_k: &[f64],
    source_nx: usize,
    crop: GridCrop,
    longitude_row_wraps: &[usize],
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    if let Ok(message) = find_message(messages, &[(0, 1, 0, 103, Some(2.0))]) {
        let values =
            unpack_message_normalized_cropped(message, source_nx, crop, longitude_row_wraps)?;
        return Ok(q_to_mixing_ratio(&values));
    }
    if let Ok(message) = find_message(messages, &[(0, 0, 6, 103, Some(2.0))]) {
        let dewpoint_k =
            unpack_message_normalized_cropped(message, source_nx, crop, longitude_row_wraps)?;
        return Ok(psfc_pa
            .iter()
            .zip(dewpoint_k.iter())
            .map(|(&psfc, &td_k)| mixing_ratio_from_dewpoint_k(psfc / 100.0, td_k))
            .collect());
    }
    if let Ok(message) = find_message(messages, &[(0, 1, 1, 103, Some(2.0))]) {
        let rh_pct =
            unpack_message_normalized_cropped(message, source_nx, crop, longitude_row_wraps)?;
        return Ok(psfc_pa
            .iter()
            .zip(t2_k.iter())
            .zip(rh_pct.iter())
            .map(|((&psfc, &t_k), &rh)| mixing_ratio_from_relative_humidity(psfc / 100.0, t_k, rh))
            .collect());
    }
    Err("missing 2m specific humidity/dewpoint/RH field for surface thermodynamics".into())
}

fn decode_pressure_mixing_ratio_levels_cropped(
    messages: &[Grib2Message],
    temperature: &Vec<(f64, Vec<f64>)>,
    source_nx: usize,
    crop: GridCrop,
    longitude_row_wraps: &[usize],
) -> Result<Vec<(f64, Vec<f64>)>, Box<dyn std::error::Error>> {
    if let Ok(levels) =
        collect_levels_cropped(messages, 0, 1, 0, 100, source_nx, crop, longitude_row_wraps)
    {
        return Ok(levels
            .into_iter()
            .map(|(level, values)| (level, q_to_mixing_ratio(&values)))
            .collect());
    }
    if let Ok(dewpoint) =
        collect_levels_cropped(messages, 0, 0, 6, 100, source_nx, crop, longitude_row_wraps)
    {
        let mut out = Vec::with_capacity(dewpoint.len());
        for (level, td_k) in dewpoint {
            out.push((
                level,
                td_k.into_iter()
                    .map(|td_k| {
                        mixing_ratio_from_dewpoint_k(normalize_pressure_level_hpa(level), td_k)
                    })
                    .collect(),
            ));
        }
        return Ok(out);
    }
    if let Ok(rh) =
        collect_levels_cropped(messages, 0, 1, 1, 100, source_nx, crop, longitude_row_wraps)
    {
        let mut out = Vec::with_capacity(rh.len());
        for (level, rh_pct) in rh {
            let temperature_k = level_values(temperature, level)
                .ok_or_else(|| format!("missing temperature level {level} for RH fallback"))?;
            out.push((
                level,
                temperature_k
                    .iter()
                    .zip(rh_pct.iter())
                    .map(|(&t_k, &rh)| {
                        mixing_ratio_from_relative_humidity(
                            normalize_pressure_level_hpa(level),
                            t_k,
                            rh,
                        )
                    })
                    .collect(),
            ));
        }
        return Ok(out);
    }
    Err("missing pressure-level specific humidity/dewpoint/RH field for thermodynamics".into())
}

fn decode_height_levels_cropped(
    messages: &[Grib2Message],
    source_nx: usize,
    crop: GridCrop,
    longitude_row_wraps: &[usize],
) -> Result<Vec<(f64, Vec<f64>)>, Box<dyn std::error::Error>> {
    if let Ok(levels) =
        collect_levels_cropped(messages, 0, 3, 5, 100, source_nx, crop, longitude_row_wraps)
    {
        return Ok(levels);
    }
    if let Ok(levels) =
        collect_levels_cropped(messages, 0, 3, 4, 100, source_nx, crop, longitude_row_wraps)
    {
        return Ok(levels
            .into_iter()
            .map(|(level, values)| {
                (
                    level,
                    values
                        .into_iter()
                        .map(|value| value * GEOPOTENTIAL_M2S2_TO_M)
                        .collect(),
                )
            })
            .collect());
    }
    Err("missing pressure-level height/geopotential field".into())
}

fn collect_levels(
    messages: &[Grib2Message],
    discipline: u8,
    category: u8,
    number: u8,
    level_type: u8,
) -> Result<Vec<(f64, Vec<f64>)>, Box<dyn std::error::Error>> {
    let mut records = messages
        .par_iter()
        .filter(|msg| {
            msg.discipline == discipline
                && msg.product.parameter_category == category
                && msg.product.parameter_number == number
                && msg.product.level_type == level_type
        })
        .map(|msg| {
            unpack_message_normalized(msg)
                .map(|values| (msg.product.level_value, values))
                .map_err(|err| err.to_string())
        })
        .collect::<Result<Vec<_>, String>>()
        .map_err(|err| std::io::Error::other(err))?;

    if records.is_empty() {
        return Err(format!(
            "missing GRIB records for discipline={discipline} category={category} number={number} level_type={level_type}"
        )
        .into());
    }
    records.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    Ok(records)
}

fn collect_optional_levels(
    messages: &[Grib2Message],
    discipline: u8,
    category: u8,
    number: u8,
    level_type: u8,
) -> Result<Option<Vec<(f64, Vec<f64>)>>, Box<dyn std::error::Error>> {
    match collect_levels(messages, discipline, category, number, level_type) {
        Ok(records) => Ok(Some(records)),
        Err(_) => Ok(None),
    }
}

fn collect_optional_levels_any(
    messages: &[Grib2Message],
    candidates: &[(u8, u8, u8)],
    level_type: u8,
) -> Result<Option<Vec<(f64, Vec<f64>)>>, Box<dyn std::error::Error>> {
    for &(discipline, category, number) in candidates {
        if let Some(records) =
            collect_optional_levels(messages, discipline, category, number, level_type)?
        {
            return Ok(Some(records));
        }
    }
    Ok(None)
}

fn collect_levels_cropped(
    messages: &[Grib2Message],
    discipline: u8,
    category: u8,
    number: u8,
    level_type: u8,
    source_nx: usize,
    crop: GridCrop,
    longitude_row_wraps: &[usize],
) -> Result<Vec<(f64, Vec<f64>)>, Box<dyn std::error::Error>> {
    let mut records = messages
        .par_iter()
        .filter(|msg| {
            msg.discipline == discipline
                && msg.product.parameter_category == category
                && msg.product.parameter_number == number
                && msg.product.level_type == level_type
        })
        .map(|msg| {
            unpack_message_normalized_cropped(msg, source_nx, crop, longitude_row_wraps)
                .map(|values| (msg.product.level_value, values))
                .map_err(|err| err.to_string())
        })
        .collect::<Result<Vec<_>, String>>()
        .map_err(|err| std::io::Error::other(err))?;

    if records.is_empty() {
        return Err(format!(
            "missing GRIB records for discipline={discipline} category={category} number={number} level_type={level_type}"
        )
        .into());
    }
    records.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    Ok(records)
}

fn collect_optional_levels_cropped(
    messages: &[Grib2Message],
    discipline: u8,
    category: u8,
    number: u8,
    level_type: u8,
    source_nx: usize,
    crop: GridCrop,
    longitude_row_wraps: &[usize],
) -> Result<Option<Vec<(f64, Vec<f64>)>>, Box<dyn std::error::Error>> {
    match collect_levels_cropped(
        messages,
        discipline,
        category,
        number,
        level_type,
        source_nx,
        crop,
        longitude_row_wraps,
    ) {
        Ok(records) => Ok(Some(records)),
        Err(_) => Ok(None),
    }
}

fn collect_optional_levels_any_cropped(
    messages: &[Grib2Message],
    candidates: &[(u8, u8, u8)],
    level_type: u8,
    source_nx: usize,
    crop: GridCrop,
    longitude_row_wraps: &[usize],
) -> Result<Option<Vec<(f64, Vec<f64>)>>, Box<dyn std::error::Error>> {
    for &(discipline, category, number) in candidates {
        if let Some(records) = collect_optional_levels_cropped(
            messages,
            discipline,
            category,
            number,
            level_type,
            source_nx,
            crop,
            longitude_row_wraps,
        )? {
            return Ok(Some(records));
        }
    }
    Ok(None)
}

// ---------------------------------------------------------------------------
// Direct-write decode of the five required pressure volumes.
//
// The collect-then-flatten lane materialized every variable's per-level
// collection AND its flattened volume — a full extra copy of the ~2.9 GB of
// f64 inputs, the measured global peak of a store ingest, with an allocator
// churn profile (~200 15 MB frees racing five 580 MB commits) that kept the
// working set inflated well past the live set. This lane unpacks each
// matched message straight into its slot of a preallocated flat volume.
//
// Semantics replicate the collect lane exactly:
// * level records filter + stable sort descending == collect_levels;
// * the common level set is computed from the level keys with the same
//   0.25-tolerance first-match (values never affected which levels were
//   common — only unpack FAILURES could, see below);
// * EVERY filtered message of a chosen branch is unpacked (results outside
//   the common set are discarded), so an unpack failure anywhere in a
//   variable fails that variable exactly as collect_levels did: a hard
//   error for temperature/u/v, branch fallback for gh (height ->
//   geopotential) and moisture (q -> dewpoint -> RH) via the retry loop;
// * per-element conversions (q/dewpoint/RH -> mixing ratio, geopotential
//   -> height, K -> C) are the identical formulas applied at the identical
//   per-record granularity.
// ---------------------------------------------------------------------------

/// The five required volumes plus the aligned (common) level set, in the
/// exact values and order the collect-then-flatten lane produced.
struct DecodedRequiredVolumes {
    common_levels: Vec<f64>,
    temperature_c_3d: Vec<f64>,
    qvapor_kgkg_3d: Vec<f64>,
    u_ms_3d: Vec<f64>,
    v_ms_3d: Vec<f64>,
    gh_m_3d: Vec<f64>,
}

/// `(level_value, message_index)` records for one variable: the same
/// filter and the same stable descending sort as `collect_levels`, without
/// unpacking anything.
fn level_records(
    messages: &[Grib2Message],
    discipline: u8,
    category: u8,
    number: u8,
    level_type: u8,
) -> Vec<(f64, usize)> {
    let mut records: Vec<(f64, usize)> = messages
        .iter()
        .enumerate()
        .filter(|(_, msg)| {
            msg.discipline == discipline
                && msg.product.parameter_category == category
                && msg.product.parameter_number == number
                && msg.product.level_type == level_type
        })
        .map(|(index, msg)| (msg.product.level_value, index))
        .collect();
    records.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    records
}

/// First record within the same 0.25 tolerance `level_values` used.
fn find_record(records: &[(f64, usize)], level: f64) -> Option<usize> {
    records
        .iter()
        .position(|(candidate, _)| (candidate - level).abs() < 0.25)
}

fn missing_records_error(discipline: u8, category: u8, number: u8, level_type: u8) -> String {
    format!(
        "missing GRIB records for discipline={discipline} category={category} number={number} level_type={level_type}"
    )
}

/// Free the raw payload (and bitmap) of every message the required-volume
/// decode can never touch. Called after the optional volumes are decoded,
/// so only the five variables' candidate messages keep their bytes
/// (~40% of an HRRR prs file's parsed copy).
fn strip_unneeded_required_var_raws(messages: &mut [Grib2Message]) {
    const KEEP: [(u8, u8, u8); 8] = [
        (0, 0, 0), // temperature
        (0, 2, 2), // u wind
        (0, 2, 3), // v wind
        (0, 3, 5), // geopotential height
        (0, 3, 4), // geopotential (height fallback)
        (0, 1, 0), // specific humidity
        (0, 0, 6), // dewpoint (moisture fallback)
        (0, 1, 1), // relative humidity (moisture fallback)
    ];
    for msg in messages {
        let keep = msg.product.level_type == 100
            && KEEP.contains(&(
                msg.discipline,
                msg.product.parameter_category,
                msg.product.parameter_number,
            ));
        if !keep {
            msg.raw_data = Vec::new();
            msg.bitmap = None;
        }
    }
}

/// Total-field grid-definition equality (f64 by bit pattern) — the gate
/// for reusing one precomputed set of longitude row wraps across messages.
fn grid_definitions_identical(a: &GridDefinition, b: &GridDefinition) -> bool {
    a.template == b.template
        && a.nx == b.nx
        && a.ny == b.ny
        && a.lat1.to_bits() == b.lat1.to_bits()
        && a.lon1.to_bits() == b.lon1.to_bits()
        && a.lat2.to_bits() == b.lat2.to_bits()
        && a.lon2.to_bits() == b.lon2.to_bits()
        && a.dx.to_bits() == b.dx.to_bits()
        && a.dy.to_bits() == b.dy.to_bits()
        && a.latin1.to_bits() == b.latin1.to_bits()
        && a.latin2.to_bits() == b.latin2.to_bits()
        && a.lov.to_bits() == b.lov.to_bits()
        && a.scan_mode == b.scan_mode
        && a.lad.to_bits() == b.lad.to_bits()
        && a.projection_center_flag == b.projection_center_flag
        && a.n_parallel == b.n_parallel
        && a.south_pole_lat.to_bits() == b.south_pole_lat.to_bits()
        && a.south_pole_lon.to_bits() == b.south_pole_lon.to_bits()
        && a.rotation_angle.to_bits() == b.rotation_angle.to_bits()
        && a.satellite_lat.to_bits() == b.satellite_lat.to_bits()
        && a.satellite_lon.to_bits() == b.satellite_lon.to_bits()
        && a.xp.to_bits() == b.xp.to_bits()
        && a.yp.to_bits() == b.yp.to_bits()
        && a.altitude.to_bits() == b.altitude.to_bits()
        && a.pl == b.pl
        && a.is_reduced == b.is_reduced
        && a.num_data_points == b.num_data_points
        && a.shape_of_earth == b.shape_of_earth
        && a.resolution_flags == b.resolution_flags
}

/// The per-row rotate-left amounts `rotate_values_to_normalized_longitude_rows`
/// derives from one grid definition (identical derivation: lat/lon from the
/// grid, scan-0x40 row flip, per-row longitude normalization + first wrap).
fn row_wraps_for_grid(grid: &GridDefinition) -> Vec<usize> {
    let nx = grid.nx as usize;
    let ny = grid.ny as usize;
    let (_lat, mut lon) = grid_latlon(grid);
    if grid.scan_mode & 0x40 != 0 {
        flip_rows(&mut lon, nx, ny);
    }
    normalized_longitude_row_wraps(&mut lon, nx, ny)
}

/// One precomputed row-wrap set, valid for every message whose grid
/// definition is bit-identical to the primed one (true for every message
/// of a single-grid file like HRRR prs). Messages with a different grid
/// fall back to the per-message derivation — never a wrong wrap.
#[derive(Default)]
struct RowWrapCache {
    def: Option<GridDefinition>,
    wraps: Vec<usize>,
}

impl RowWrapCache {
    fn prime(&mut self, grid: &GridDefinition) {
        if self
            .def
            .as_ref()
            .map(|have| grid_definitions_identical(have, grid))
            .unwrap_or(false)
        {
            return;
        }
        self.wraps = row_wraps_for_grid(grid);
        self.def = Some(grid.clone());
    }

    fn get(&self, grid: &GridDefinition) -> Option<&[usize]> {
        match &self.def {
            Some(have) if grid_definitions_identical(have, grid) => Some(&self.wraps),
            _ => None,
        }
    }
}

/// `unpack_message_normalized` with the row-wrap derivation hoisted out:
/// identical unpack, identical early-out, identical per-row rotation.
fn unpack_message_normalized_cached(
    message: &Grib2Message,
    cache: &RowWrapCache,
) -> Result<Vec<f64>, String> {
    let mut values = unpack_message_scan_normalized(message).map_err(|err| err.to_string())?;
    let nx = message.grid.nx as usize;
    let ny = message.grid.ny as usize;
    if nx == 0 || ny == 0 || values.len() != nx * ny {
        return Ok(values);
    }
    match cache.get(&message.grid) {
        Some(wraps) => rotate_value_rows_left(&mut values, nx, wraps),
        None => {
            let wraps = row_wraps_for_grid(&message.grid);
            rotate_value_rows_left(&mut values, nx, &wraps);
        }
    }
    Ok(values)
}

fn rotate_value_rows_left(values: &mut [f64], nx: usize, wraps: &[usize]) {
    for (row, &wrap) in wraps.iter().enumerate() {
        if wrap == 0 {
            continue;
        }
        let start = row * nx;
        values[start..start + nx].rotate_left(wrap);
    }
}

/// Per-element conversion applied while a record's values are written into
/// its volume slot — each is the exact formula the collect lane applied
/// per level (temperature stays K here; the K -> C map runs on the whole
/// volume afterwards, as the flatten lane did).
enum RequiredConvert<'a> {
    Identity,
    GeopotentialToHeightM,
    QToMixingRatio,
    DewpointToMixingRatio,
    RhToMixingRatio {
        temperature_records: &'a [(f64, usize)],
        temperature_volume_k: &'a [f64],
        file: &'a Grib2File,
    },
}

fn decode_required_volumes(
    file: &Grib2File,
    nx: usize,
    ny: usize,
) -> Result<DecodedRequiredVolumes, Box<dyn std::error::Error>> {
    let expected = nx * ny;
    // Branch-failure flags: an unpack failure inside a fallback-capable
    // variable marks its branch failed and restarts the decode with the
    // next branch — the exact outcome of collect_levels' Err falling
    // through to the next `if let Ok(..)` arm.
    let mut gh_height_failed = false;
    let mut q_failed = false;
    let mut dewpoint_failed = false;
    let mut rh_failed = false;
    let mut wrap_cache = RowWrapCache::default();
    loop {
        let temperature = level_records(&file.messages, 0, 0, 0, 100);
        if temperature.is_empty() {
            return Err(missing_records_error(0, 0, 0, 100).into());
        }
        let u_wind = level_records(&file.messages, 0, 2, 2, 100);
        if u_wind.is_empty() {
            return Err(missing_records_error(0, 2, 2, 100).into());
        }
        let v_wind = level_records(&file.messages, 0, 2, 3, 100);
        if v_wind.is_empty() {
            return Err(missing_records_error(0, 2, 3, 100).into());
        }
        let gh_height = if gh_height_failed {
            Vec::new()
        } else {
            level_records(&file.messages, 0, 3, 5, 100)
        };
        let (gh_records, gh_is_height_branch) = if !gh_height.is_empty() {
            (gh_height, true)
        } else {
            let gh_geo = level_records(&file.messages, 0, 3, 4, 100);
            if gh_geo.is_empty() {
                return Err("missing pressure-level height/geopotential field".into());
            }
            (gh_geo, false)
        };
        #[derive(Clone, Copy, PartialEq)]
        enum MoistureBranch {
            Q,
            Dewpoint,
            Rh,
        }
        let (moisture_records, moisture_branch) = {
            let q = if q_failed {
                Vec::new()
            } else {
                level_records(&file.messages, 0, 1, 0, 100)
            };
            if !q.is_empty() {
                (q, MoistureBranch::Q)
            } else {
                let dewpoint = if dewpoint_failed {
                    Vec::new()
                } else {
                    level_records(&file.messages, 0, 0, 6, 100)
                };
                if !dewpoint.is_empty() {
                    (dewpoint, MoistureBranch::Dewpoint)
                } else {
                    let rh = if rh_failed {
                        Vec::new()
                    } else {
                        level_records(&file.messages, 0, 1, 1, 100)
                    };
                    if !rh.is_empty() {
                        (rh, MoistureBranch::Rh)
                    } else {
                        return Err(
                            "missing pressure-level specific humidity/dewpoint/RH field \
                             for thermodynamics"
                                .into(),
                        );
                    }
                }
            }
        };
        // RH branch: every moisture record's level must have a temperature
        // record — the collect lane's conversion loop errored on the first
        // missing one, before the common-level computation.
        if moisture_branch == MoistureBranch::Rh {
            for &(level, _) in &moisture_records {
                if find_record(&temperature, level).is_none() {
                    return Err(format!("missing temperature level {level} for RH fallback").into());
                }
            }
        }

        // Common levels from the level keys: the same base order (the
        // temperature records), the same first-match 0.25 tolerance, and
        // duplicates preserved — `common_isobaric_levels` over metadata.
        let common: Vec<f64> = temperature
            .iter()
            .map(|&(level, _)| level)
            .filter(|&level| {
                [&moisture_records, &u_wind, &v_wind, &gh_records]
                    .iter()
                    .all(|records| find_record(records, level).is_some())
            })
            .collect();
        if common.is_empty() {
            return Err("pressure family had no common thermodynamic levels".into());
        }

        // Build the volumes in the collect lane's variable order, so on a
        // multi-variable failure the same variable's error surfaces.
        let mut temperature_volume_k = match build_required_volume(
            file,
            &temperature,
            &common,
            expected,
            &RequiredConvert::Identity,
            &mut wrap_cache,
        ) {
            Ok(volume) => volume,
            Err(err) => return Err(Box::new(std::io::Error::other(err))),
        };
        let u_ms_3d = match build_required_volume(
            file,
            &u_wind,
            &common,
            expected,
            &RequiredConvert::Identity,
            &mut wrap_cache,
        ) {
            Ok(volume) => volume,
            Err(err) => return Err(Box::new(std::io::Error::other(err))),
        };
        let v_ms_3d = match build_required_volume(
            file,
            &v_wind,
            &common,
            expected,
            &RequiredConvert::Identity,
            &mut wrap_cache,
        ) {
            Ok(volume) => volume,
            Err(err) => return Err(Box::new(std::io::Error::other(err))),
        };
        let gh_convert = if gh_is_height_branch {
            RequiredConvert::Identity
        } else {
            RequiredConvert::GeopotentialToHeightM
        };
        let gh_m_3d = match build_required_volume(
            file,
            &gh_records,
            &common,
            expected,
            &gh_convert,
            &mut wrap_cache,
        ) {
            Ok(volume) => volume,
            Err(_) if gh_is_height_branch => {
                gh_height_failed = true;
                continue;
            }
            // A geopotential-branch failure fell through to the generic
            // error in the collect lane.
            Err(_) => return Err("missing pressure-level height/geopotential field".into()),
        };
        let moisture_convert = match moisture_branch {
            MoistureBranch::Q => RequiredConvert::QToMixingRatio,
            MoistureBranch::Dewpoint => RequiredConvert::DewpointToMixingRatio,
            MoistureBranch::Rh => RequiredConvert::RhToMixingRatio {
                temperature_records: &temperature,
                temperature_volume_k: &temperature_volume_k,
                file,
            },
        };
        let qvapor_kgkg_3d = match build_required_volume(
            file,
            &moisture_records,
            &common,
            expected,
            &moisture_convert,
            &mut wrap_cache,
        ) {
            Ok(volume) => volume,
            Err(_) => {
                // The collect lane's moisture chain swallowed the branch
                // error and tried the next branch (or the generic error,
                // produced by the next loop iteration once every branch
                // is marked failed).
                match moisture_branch {
                    MoistureBranch::Q => q_failed = true,
                    MoistureBranch::Dewpoint => dewpoint_failed = true,
                    MoistureBranch::Rh => rh_failed = true,
                }
                continue;
            }
        };

        // K -> C, the same per-element op the flatten lane applied.
        for value in &mut temperature_volume_k {
            *value -= 273.15;
        }

        return Ok(DecodedRequiredVolumes {
            common_levels: common,
            temperature_c_3d: temperature_volume_k,
            qvapor_kgkg_3d,
            u_ms_3d,
            v_ms_3d,
            gh_m_3d,
        });
    }
}

/// Unpack every record of one variable in parallel; records selected for a
/// common level write (with conversion) straight into their slot of the
/// preallocated flat volume, the rest are unpacked and discarded (failure
/// parity with collect_levels, which unpacked everything).
fn build_required_volume(
    file: &Grib2File,
    records: &[(f64, usize)],
    common: &[f64],
    expected: usize,
    convert: &RequiredConvert<'_>,
    wrap_cache: &mut RowWrapCache,
) -> Result<Vec<f64>, String> {
    // slot k (common level k) -> the record `level_values` would pick.
    let slot_records: Vec<usize> = common
        .iter()
        .map(|&level| {
            find_record(records, level).expect("common levels have a record by construction")
        })
        .collect();
    if let Some(&(_, message_index)) = records.first() {
        wrap_cache.prime(&file.messages[message_index].grid);
    }

    let mut volume = vec![0.0f64; common.len() * expected];
    let mut record_slots: Vec<Vec<usize>> = vec![Vec::new(); records.len()];
    for (k, &record_index) in slot_records.iter().enumerate() {
        record_slots[record_index].push(k);
    }
    // Disjoint per-slot mutable windows handed to the parallel jobs.
    let mut slot_slices: Vec<Option<&mut [f64]>> = volume.chunks_mut(expected).map(Some).collect();
    let mut jobs: Vec<(usize, Vec<(usize, &mut [f64])>)> = Vec::with_capacity(records.len());
    for (record_index, slots) in record_slots.iter().enumerate() {
        let targets: Vec<(usize, &mut [f64])> = slots
            .iter()
            .map(|&k| (k, slot_slices[k].take().expect("each slot is taken once")))
            .collect();
        jobs.push((record_index, targets));
    }

    let cache: &RowWrapCache = wrap_cache;
    jobs.into_par_iter()
        .try_for_each(|(record_index, mut targets)| -> Result<(), String> {
            let (record_level, message_index) = records[record_index];
            let message = &file.messages[message_index];
            let values = unpack_message_normalized_cached(message, cache)?;
            if targets.is_empty() {
                return Ok(()); // unpacked for failure parity; value discarded
            }
            if values.len() != expected {
                return Err("decoded pressure field had unexpected grid size".to_string());
            }
            let (first, rest) = targets.split_at_mut(1);
            let (slot_k, dst0) = &mut first[0];
            convert_into_slot(
                dst0,
                &values,
                record_level,
                *slot_k,
                common[*slot_k],
                expected,
                convert,
                cache,
            )?;
            for (_, dst) in rest {
                dst.copy_from_slice(dst0);
            }
            Ok(())
        })?;
    Ok(volume)
}

/// Apply one record's conversion into its slot — each arm is the exact
/// per-element formula the collect lane applied to that record's plane.
#[allow(clippy::too_many_arguments)]
fn convert_into_slot(
    dst: &mut [f64],
    values: &[f64],
    record_level: f64,
    slot_k: usize,
    slot_common_level: f64,
    expected: usize,
    convert: &RequiredConvert<'_>,
    cache: &RowWrapCache,
) -> Result<(), String> {
    match convert {
        RequiredConvert::Identity => dst.copy_from_slice(values),
        RequiredConvert::GeopotentialToHeightM => {
            for (out, &value) in dst.iter_mut().zip(values.iter()) {
                *out = value * GEOPOTENTIAL_M2S2_TO_M;
            }
        }
        RequiredConvert::QToMixingRatio => {
            for (out, &q) in dst.iter_mut().zip(values.iter()) {
                *out = (q / (1.0 - q).max(1.0e-12)).max(1.0e-10);
            }
        }
        RequiredConvert::DewpointToMixingRatio => {
            let level_hpa = normalize_pressure_level_hpa(record_level);
            for (out, &td_k) in dst.iter_mut().zip(values.iter()) {
                *out = mixing_ratio_from_dewpoint_k(level_hpa, td_k);
            }
        }
        RequiredConvert::RhToMixingRatio {
            temperature_records,
            temperature_volume_k,
            file,
        } => {
            // The collect lane looked temperature up at the RH RECORD's
            // level. The temperature volume row for this slot holds the
            // record matched at the slot's COMMON level; whenever the two
            // searches pick the same record (always, for real level
            // spacings) the row is reused, otherwise the exact record the
            // collect lane would have used is unpacked on demand.
            let record_t = find_record(temperature_records, record_level)
                .expect("RH temperature presence pre-checked");
            let slot_record_t = find_record(temperature_records, slot_common_level);
            let t_owned;
            let t_values: &[f64] = if slot_record_t == Some(record_t) {
                &temperature_volume_k[slot_k * expected..(slot_k + 1) * expected]
            } else {
                t_owned = unpack_message_normalized_cached(
                    &file.messages[temperature_records[record_t].1],
                    cache,
                )?;
                &t_owned
            };
            // The collect lane zipped temperature against RH (truncating)
            // and the flatten length check then rejected short rows.
            if t_values.len() < expected {
                return Err("decoded pressure field had unexpected grid size".to_string());
            }
            let level_hpa = normalize_pressure_level_hpa(record_level);
            for ((out, &rh), &t_k) in dst.iter_mut().zip(values.iter()).zip(t_values.iter()) {
                *out = mixing_ratio_from_relative_humidity(level_hpa, t_k, rh);
            }
        }
    }
    Ok(())
}

fn pressure_optional_decode_enabled() -> bool {
    pressure_optional_decode_enabled_from_env_value(
        std::env::var(PRESSURE_OPTIONAL_FIELDS_ENV).ok(),
    )
}

fn pressure_decode_cache_name() -> &'static str {
    pressure_decode_cache_name_from_optional_enabled(pressure_optional_decode_enabled())
}

fn pressure_decode_cache_name_from_optional_enabled(include_optional: bool) -> &'static str {
    if include_optional {
        "pressure_optional"
    } else {
        "pressure_core"
    }
}

fn pressure_optional_decode_enabled_from_env_value(value: Option<String>) -> bool {
    let Some(value) = value else {
        return true;
    };
    !matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "0" | "false" | "no" | "off"
    )
}

fn common_isobaric_levels(
    base: &Vec<(f64, Vec<f64>)>,
    others: &[&Vec<(f64, Vec<f64>)>],
) -> Vec<f64> {
    base.iter()
        .map(|(level, _)| *level)
        .filter(|&level| {
            others
                .iter()
                .all(|records| level_values(records, level).is_some())
        })
        .collect()
}

fn level_values<'a>(records: &'a [(f64, Vec<f64>)], level: f64) -> Option<&'a [f64]> {
    records
        .iter()
        .find(|(candidate, _)| (*candidate - level).abs() < 0.25)
        .map(|(_, values)| values.as_slice())
}

fn pressure_grid_shape_from_messages(
    messages: &[Grib2Message],
) -> Result<(usize, usize), Box<dyn std::error::Error>> {
    let mut matching = messages.iter().filter(|msg| msg.product.level_type == 100);
    let sample = matching
        .next()
        .ok_or("pressure family had no isobaric GRIB messages")?;
    let nx = sample.grid.nx as usize;
    let ny = sample.grid.ny as usize;
    for message in matching {
        let message_nx = message.grid.nx as usize;
        let message_ny = message.grid.ny as usize;
        if message_nx != nx || message_ny != ny {
            return Err("pressure family contained inconsistent grid shapes".into());
        }
    }
    Ok((nx, ny))
}

fn normalized_longitude_row_wraps_from_messages(
    messages: &[Grib2Message],
) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    let sample = messages
        .iter()
        .find(|msg| msg.product.level_type == 100)
        .or_else(|| messages.first())
        .ok_or("GRIB family had no messages for row-wrap detection")?;
    let nx = sample.grid.nx as usize;
    let ny = sample.grid.ny as usize;
    let (_lat_raw, mut lon_raw) = grid_latlon(&sample.grid);
    if sample.grid.scan_mode & 0x40 != 0 {
        flip_rows(&mut lon_raw, nx, ny);
    }
    Ok(normalized_longitude_row_wraps(&mut lon_raw, nx, ny))
}

/// The three native model CAPE planes a surface-product GRIB file may
/// carry (HRRR `sfc` does), decoded with the exact message matching and
/// scan/longitude normalization the surface decode lane uses for
/// [`SurfaceFields::native_sbcape_jkg`] and friends. Planes the file does
/// not carry come back `None` — same optionality as the decode lane.
///
/// This exists for the store-ingest path: `rw_ingest` extracts its 2D
/// fields through rustwx-io (which applies the same scan-mode flip and
/// per-row longitude rotation), so these planes line up row-for-row with
/// the extracted grid, letting the heavy ECAPE stage compute the
/// native-CAPE ratio recipes without a second decode lane.
#[derive(Debug, Clone, Default)]
pub struct NativeCapePlanes {
    pub sbcape_jkg: Option<Vec<f64>>,
    pub mlcape_jkg: Option<Vec<f64>>,
    pub mucape_jkg: Option<Vec<f64>>,
}

/// Decode the native CAPE planes from raw surface-product GRIB bytes.
/// Only the (up to three) matching CAPE messages unpack; the rest of the
/// file is index-parsed only.
pub fn decode_native_cape_planes(
    bytes: &[u8],
) -> Result<NativeCapePlanes, Box<dyn std::error::Error>> {
    let file = Grib2File::from_bytes(bytes)?;
    Ok(NativeCapePlanes {
        sbcape_jkg: decode_optional_native_cape(&file.messages, NativeCapeLayer::Surface)?,
        mlcape_jkg: decode_optional_native_cape(&file.messages, NativeCapeLayer::MixedLayer)?,
        mucape_jkg: decode_optional_native_cape(&file.messages, NativeCapeLayer::MostUnstable)?,
    })
}

/// Decode the surface + pressure thermodynamic pair from raw family GRIB
/// bytes exactly as the derived/heavy render lanes decode them: the same
/// message matching, the same moisture preference (2 m and isobaric
/// specific humidity first, then dewpoint, then RH), the same f64
/// precision, and the same level alignment. The optional pressure volumes
/// (omega, absolute vorticity, cloud species) are skipped — no
/// store-computed recipe consumes them and the required arrays are
/// unaffected by their presence (level alignment uses only the required
/// five). The store ingest computes its derived and heavy grids from this
/// pair so the stored grids are bit-identical to what the render lanes
/// compute from the same files.
pub fn decode_store_thermo_pair(
    surface_bytes: &[u8],
    pressure_bytes: &[u8],
) -> Result<(SurfaceFields, PressureFields), Box<dyn std::error::Error>> {
    let surface = decode_surface(surface_bytes)?;
    let (pressure, nx, ny) = decode_pressure_with_shape_opts(pressure_bytes, false)?;
    validate_store_thermo_pair(&surface, &pressure, nx, ny)?;
    Ok((surface, pressure))
}

/// [`decode_store_thermo_pair`] taking ownership of both raw GRIB buffers
/// so each is freed at its true last use — the surface bytes right after
/// the surface decode, the pressure bytes as soon as the pressure parser
/// holds its own copy of the messages — instead of staying resident
/// through the whole thermo decode. Identical decode, identical output.
pub fn decode_store_thermo_pair_owned(
    surface_bytes: Vec<u8>,
    pressure_bytes: Vec<u8>,
) -> Result<(SurfaceFields, PressureFields), Box<dyn std::error::Error>> {
    let surface = decode_surface(&surface_bytes)?;
    drop(surface_bytes);
    let (pressure, nx, ny) = decode_pressure_with_shape_opts_owned(pressure_bytes, false)?;
    validate_store_thermo_pair(&surface, &pressure, nx, ny)?;
    Ok((surface, pressure))
}

fn validate_store_thermo_pair(
    surface: &SurfaceFields,
    pressure: &PressureFields,
    nx: usize,
    ny: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    if nx != surface.nx || ny != surface.ny {
        return Err(format!(
            "pressure decode shape {nx}x{ny} did not match surface shape {}x{}",
            surface.nx, surface.ny
        )
        .into());
    }
    let expected = surface.nx * surface.ny * pressure.pressure_levels_hpa.len();
    if pressure.temperature_c_3d.len() != expected
        || pressure.qvapor_kgkg_3d.len() != expected
        || pressure.u_ms_3d.len() != expected
        || pressure.v_ms_3d.len() != expected
        || pressure.gh_m_3d.len() != expected
    {
        return Err("pressure decode fields did not match the surface grid shape".into());
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
enum NativeCapeLayer {
    Surface,
    MixedLayer,
    MostUnstable,
}

impl NativeCapeLayer {
    fn candidates(self) -> &'static [(u8, u8, u8, u8, Option<f64>)] {
        match self {
            Self::Surface => &[(0, 7, 6, 1, Some(0.0))],
            Self::MixedLayer => &[(0, 7, 6, 108, Some(9000.0))],
            Self::MostUnstable => &[(0, 7, 6, 108, Some(25500.0))],
        }
    }
}

fn decode_optional_native_cape(
    messages: &[Grib2Message],
    layer: NativeCapeLayer,
) -> Result<Option<Vec<f64>>, Box<dyn std::error::Error>> {
    let Some(message) = find_optional_message(messages, layer.candidates()) else {
        return Ok(None);
    };
    Ok(Some(unpack_message_normalized(message)?))
}

fn decode_optional_native_cape_cropped(
    messages: &[Grib2Message],
    layer: NativeCapeLayer,
    source_nx: usize,
    crop: GridCrop,
    longitude_row_wraps: &[usize],
) -> Result<Option<Vec<f64>>, Box<dyn std::error::Error>> {
    let Some(message) = find_optional_message(messages, layer.candidates()) else {
        return Ok(None);
    };
    Ok(Some(unpack_message_normalized_cropped(
        message,
        source_nx,
        crop,
        longitude_row_wraps,
    )?))
}

fn decode_optional_native_pblh(
    messages: &[Grib2Message],
) -> Result<Option<Vec<f64>>, Box<dyn std::error::Error>> {
    let candidates = &[
        (0, 3, 18, 1, Some(0.0)),
        (0, 3, 18, 1, None),
        (0, 3, 196, 1, Some(0.0)),
        (0, 3, 196, 1, None),
    ];
    let Some(message) = find_optional_message(messages, candidates) else {
        return Ok(None);
    };
    Ok(Some(unpack_message_normalized(message)?))
}

fn decode_optional_native_pblh_cropped(
    messages: &[Grib2Message],
    source_nx: usize,
    crop: GridCrop,
    longitude_row_wraps: &[usize],
) -> Result<Option<Vec<f64>>, Box<dyn std::error::Error>> {
    let candidates = &[
        (0, 3, 18, 1, Some(0.0)),
        (0, 3, 18, 1, None),
        (0, 3, 196, 1, Some(0.0)),
        (0, 3, 196, 1, None),
    ];
    let Some(message) = find_optional_message(messages, candidates) else {
        return Ok(None);
    };
    Ok(Some(unpack_message_normalized_cropped(
        message,
        source_nx,
        crop,
        longitude_row_wraps,
    )?))
}

fn find_optional_message<'a>(
    messages: &'a [Grib2Message],
    candidates: &[(u8, u8, u8, u8, Option<f64>)],
) -> Option<&'a Grib2Message> {
    for &(discipline, category, number, level_type, level_value) in candidates {
        if let Some(message) = messages.iter().find(|msg| {
            msg.discipline == discipline
                && msg.product.parameter_category == category
                && msg.product.parameter_number == number
                && msg.product.level_type == level_type
                && level_value
                    .map(|level| (msg.product.level_value - level).abs() < 0.25)
                    .unwrap_or(true)
        }) {
            return Some(message);
        }
    }
    None
}

fn find_message<'a>(
    messages: &'a [Grib2Message],
    candidates: &[(u8, u8, u8, u8, Option<f64>)],
) -> Result<&'a Grib2Message, Box<dyn std::error::Error>> {
    for &(discipline, category, number, level_type, level_value) in candidates {
        if let Some(message) = messages.iter().find(|msg| {
            msg.discipline == discipline
                && msg.product.parameter_category == category
                && msg.product.parameter_number == number
                && msg.product.level_type == level_type
                && level_value
                    .map(|level| (msg.product.level_value - level).abs() < 0.25)
                    .unwrap_or(true)
        }) {
            return Ok(message);
        }
    }
    Err("missing GRIB message for requested candidates".into())
}

fn q_to_mixing_ratio(values: &[f64]) -> Vec<f64> {
    values
        .iter()
        .map(|&q| (q / (1.0 - q).max(1.0e-12)).max(1.0e-10))
        .collect()
}

/// Water-vapor mixing ratio (kg/kg) from dewpoint via Bolton (1980)
/// saturation vapor pressure. Public so store-ingest derived precompute
/// (`rw_ingest`) derives moisture with exactly the formula this crate's
/// GRIB decode lane uses — do not duplicate it.
pub fn mixing_ratio_from_dewpoint_k(pressure_hpa: f64, dewpoint_k: f64) -> f64 {
    let td_c = dewpoint_k - 273.15;
    let vapor_pressure_hpa = 6.112 * ((17.67 * td_c) / (td_c + 243.5)).exp();
    mixing_ratio_from_vapor_pressure(pressure_hpa, vapor_pressure_hpa)
}

/// Water-vapor mixing ratio (kg/kg) from relative humidity (%), the decode
/// lane's last-resort moisture fallback. Public for the same store-ingest
/// reuse as [`mixing_ratio_from_dewpoint_k`].
pub fn mixing_ratio_from_relative_humidity(
    pressure_hpa: f64,
    temperature_k: f64,
    rh_pct: f64,
) -> f64 {
    let t_c = temperature_k - 273.15;
    let saturation_vapor_pressure_hpa = 6.112 * ((17.67 * t_c) / (t_c + 243.5)).exp();
    let vapor_pressure_hpa = (rh_pct / 100.0).clamp(0.0, 1.5) * saturation_vapor_pressure_hpa;
    mixing_ratio_from_vapor_pressure(pressure_hpa, vapor_pressure_hpa)
}

fn mixing_ratio_from_vapor_pressure(pressure_hpa: f64, vapor_pressure_hpa: f64) -> f64 {
    let epsilon = 0.622;
    let e = vapor_pressure_hpa
        .max(0.0)
        .min((pressure_hpa - 1.0).max(0.0));
    (epsilon * e / (pressure_hpa - e).max(1.0e-6)).max(1.0e-10)
}

// GRIB2 level type 100 (isobaric surface) values are always pascals; see the
// matching note in rustwx_io. The old "only divide when > 2000" heuristic
// aliased stratospheric Pa levels onto tropospheric hectopascal numbers, so
// GFS/RRFS-A moisture columns picked up bogus mid-level RH values.
fn normalize_pressure_level_hpa(level_value_pa: f64) -> f64 {
    level_value_pa / 100.0
}

pub(crate) fn validate_pressure_decode_against_surface(
    decoded: &CachedDecode<PressureFields>,
    decoded_shape: Option<(usize, usize)>,
    nx: usize,
    ny: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some((found_nx, found_ny)) = decoded_shape {
        if found_nx != nx || found_ny != ny {
            return Err(format!(
                "pressure decode shape {found_nx}x{found_ny} did not match surface shape {nx}x{ny}"
            )
            .into());
        }
    }
    let expected = nx * ny * decoded.value.pressure_levels_hpa.len();
    if decoded.value.temperature_c_3d.len() != expected
        || decoded.value.qvapor_kgkg_3d.len() != expected
        || decoded.value.u_ms_3d.len() != expected
        || decoded.value.v_ms_3d.len() != expected
        || decoded.value.gh_m_3d.len() != expected
    {
        return Err("pressure decode fields did not match the surface grid shape".into());
    }
    for (name, values) in [
        ("omega_pa_s_3d", decoded.value.omega_pa_s_3d.as_ref()),
        (
            "absolute_vorticity_s_3d",
            decoded.value.absolute_vorticity_s_3d.as_ref(),
        ),
        (
            "cloud_liquid_kgkg_3d",
            decoded.value.cloud_liquid_kgkg_3d.as_ref(),
        ),
        (
            "cloud_ice_kgkg_3d",
            decoded.value.cloud_ice_kgkg_3d.as_ref(),
        ),
        ("rain_kgkg_3d", decoded.value.rain_kgkg_3d.as_ref()),
        ("snow_kgkg_3d", decoded.value.snow_kgkg_3d.as_ref()),
        ("graupel_kgkg_3d", decoded.value.graupel_kgkg_3d.as_ref()),
    ] {
        if let Some(values) = values {
            if values.len() != expected {
                return Err(format!(
                    "pressure decode optional field {name} did not match the surface grid shape"
                )
                .into());
            }
        }
    }
    Ok(())
}

fn normalize_longitude(lon: f64) -> f64 {
    if lon > 180.0 { lon - 360.0 } else { lon }
}

fn point_in_geographic_bounds(lon: f64, lat: f64, bounds: (f64, f64, f64, f64)) -> bool {
    if !lon.is_finite() || !lat.is_finite() || lat < bounds.2 || lat > bounds.3 {
        return false;
    }
    let west = normalize_longitude_for_bounds(bounds.0);
    let east = normalize_longitude_for_bounds(bounds.1);
    let lon = normalize_longitude_for_bounds(lon);
    if west <= east {
        lon >= west && lon <= east
    } else {
        lon >= west || lon <= east
    }
}

fn normalize_longitude_for_bounds(lon: f64) -> f64 {
    let mut lon = lon % 360.0;
    if lon > 180.0 {
        lon -= 360.0;
    } else if lon <= -180.0 {
        lon += 360.0;
    }
    lon
}

fn normalize_longitude_rows(lat: &mut [f64], lon: &mut [f64], nx: usize, ny: usize) -> Vec<usize> {
    if nx == 0 || ny == 0 {
        return Vec::new();
    }

    let row_wraps = normalized_longitude_row_wraps(lon, nx, ny);
    for row in 0..ny {
        let start = row * nx;
        let end = start + nx;
        let lat_row = &mut lat[start..end];
        let lon_row = &mut lon[start..end];
        let wrap_idx = row_wraps[row];
        if wrap_idx > 0 {
            lat_row.rotate_left(wrap_idx);
            lon_row.rotate_left(wrap_idx);
        }
    }
    row_wraps
}

fn normalized_longitude_row_wraps(lon: &mut [f64], nx: usize, ny: usize) -> Vec<usize> {
    if nx == 0 || ny == 0 {
        return Vec::new();
    }

    let mut row_wraps = Vec::with_capacity(ny);
    for row in 0..ny {
        let start = row * nx;
        let end = start + nx;
        let lon_row = &mut lon[start..end];
        for lon_value in lon_row.iter_mut() {
            *lon_value = normalize_longitude(*lon_value);
        }
        row_wraps.push(first_longitude_wrap(lon_row).unwrap_or(0));
    }
    row_wraps
}

fn first_longitude_wrap(lon_row: &[f64]) -> Option<usize> {
    lon_row
        .windows(2)
        .position(|pair| pair[1] < pair[0])
        .map(|idx| idx + 1)
}

impl LoadedModelTimestep {
    pub fn grid(&self) -> &LatLonGrid {
        &self.grid
    }

    pub fn shared_timing(&self) -> &SharedTiming {
        &self.shared_timing
    }

    fn with_cache_flags(mut self) -> Self {
        self.shared_timing.fetch_surface_cache_hit = self.surface_file.fetched.cache_hit;
        self.shared_timing.fetch_pressure_cache_hit = self.pressure_file.fetched.cache_hit;
        self.shared_timing.decode_surface_cache_hit = self.surface_decode.cache_hit;
        self.shared_timing.decode_pressure_cache_hit = self.pressure_decode.cache_hit;
        self
    }
}

#[cfg(test)]
mod tests;
