use crate::direct::{
    build_projected_map, build_projected_map_with_projection,
    build_requested_projected_map_with_projection, inverse_raster_projection_for_grid,
};
use rustwx_core::{CanonicalBundleDescriptor, Field2D, ModelId, ProductKey, SourceId};
use rustwx_render::{
    Color, ColorScale, DerivedProductStyle, DiscreteColorScale, ExtendMode, LevelDensity,
    MapRenderRequest, PngWriteOptions, ProjectedContourLineStyle, ProjectedDomain, ProjectedExtent,
    ProjectedMap, RasterSampleMode, RenderImageTiming, WeatherPalette, WeatherProduct,
    WindBarbLayer, build_projected_contour_geometry_profile, densify_discrete_scale,
    map_frame_aspect_ratio, save_png_profile_with_options, weather::temperature_palette_cropped_f,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io;
use std::path::PathBuf;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use std::thread;
use std::time::Instant;

use crate::ecape::compute_ecape_map_fields_with_prepared_volume;
use crate::gridded::{
    GridCrop, PressureFields as GenericPressureFields, ProjectedGridIntersection,
    SharedTiming as GenericSharedTiming, SurfaceFields as GenericSurfaceFields,
    classify_projected_grid_intersection, crop_latlon_grid, crop_values_f64, decode_cache_path,
    decode_surface_grid, fetch_family_file, load_or_decode_pressure_cropped_with_shape,
    load_or_decode_surface_cropped, prepare_heavy_volume_timed,
};
use crate::heavy::{HeavyComputeTiming, crop_and_guard_heavy_domain};
use crate::planner::PlannedBundle;
use crate::publication::{PublishedFetchIdentity, artifact_identity_from_path};
use crate::runtime::{
    BundleLoaderConfig, CroppedDecodeProfile, FetchedBundleBytes, LoadedBundleSet,
    LoadedBundleTiming, load_execution_plan,
};
use crate::severe::{build_planned_input_fetches, build_shared_timing_for_pair};
use crate::shared_context::{
    WeatherPanelField, build_weather_map_request, model_time_subtitle, source_subtitle,
    static_chrome_scale, static_supersample_factor, static_supersample_sharpen,
};
use crate::source::{ProductSourceMode, ProductSourceRoute};
use crate::thermo_native::extract_native_thermo_field;

mod compute;
mod inventory;
mod planning;
mod presentation;
mod query;
mod recipes;
mod store;
mod store_render;
mod types;

use compute::{
    DerivedComputedFields, SurfaceFieldSet, compute_derived_fields_generic,
    compute_surface_only_derived_fields,
};
pub use inventory::{
    BlockedDerivedRecipeInventoryEntry, DerivedRecipeInventoryEntry,
    blocked_derived_recipe_inventory, supported_derived_recipe_inventory,
};
pub(crate) use planning::{
    NativeDerivedRecipe, PlannedDerivedSourceRoutes, plan_derived_recipes,
    plan_native_thermo_routes_with_surface_product,
};
use planning::{build_derived_execution_plan, cheap_fastest_route, resolve_derived_run};
use presentation::{
    DerivedRenderOverrides, derived_output_suffix, derived_title_for_model,
    derived_title_for_request,
};
pub(crate) use query::{load_derived_sampled_fields_from_latest, required_derived_fetch_products};
use recipes::DerivedRequirements;
pub(crate) use recipes::{DerivedRecipe, derived_compute_recipes_need_pressure};
pub use store::{
    StoreComputeInputs, StoreDerivedGrid, StoreDerivedGridF32, StoreHeavyGrids, StoreHeavyGridsF32,
    StoreHeavySkip, StoreHeavyTiming, compute_store_derived_grids, compute_store_derived_grids_f32,
    compute_store_heavy_grids, compute_store_heavy_grids_f32, prepare_store_compute_inputs,
    store_derived_recipe_slugs, store_heavy_recipe_slugs,
};
pub use store_render::{StoreProductGrid, render_derived_recipes_from_store_grids};
pub use types::{
    DerivedBatchReport, DerivedBatchRequest, DerivedMemoryProfile, DerivedRecipeBlocker,
    DerivedRecipeTiming, DerivedRenderedRecipe, DerivedSharedTiming, HrrrDerivedBatchReport,
    HrrrDerivedBatchRequest, HrrrDerivedRecipeTiming, HrrrDerivedRenderedRecipe,
    HrrrDerivedSharedTiming, NativeContourRenderMode, NativeThermoArtifactReport,
};
use types::{OUTPUT_HEIGHT, OUTPUT_WIDTH};

const KNOTS_PER_MS: f64 = 1.943_844_5;

fn build_derived_projected_map_with_projection(
    model: ModelId,
    lat_deg: &[f32],
    lon_deg: &[f32],
    projection: Option<&rustwx_core::GridProjection>,
    bounds: (f64, f64, f64, f64),
    target_ratio: f64,
) -> Result<ProjectedMap, Box<dyn std::error::Error>> {
    if matches!(model, ModelId::RrfsA) {
        build_requested_projected_map_with_projection(
            lat_deg,
            lon_deg,
            projection,
            bounds,
            target_ratio,
        )
    } else {
        build_projected_map_with_projection(lat_deg, lon_deg, projection, bounds, target_ratio)
    }
}

pub fn is_heavy_derived_recipe_slug(slug: &str) -> bool {
    DerivedRecipe::parse(slug)
        .map(|recipe| recipe.is_heavy())
        .unwrap_or(false)
}

fn derived_data_layer_draw_ms(image_timing: &RenderImageTiming) -> u128 {
    image_timing.polygon_fill_ms
        + image_timing.projected_pixel_ms
        + image_timing.rasterize_ms
        + image_timing.raster_blit_ms
}

fn derived_overlay_draw_ms(image_timing: &RenderImageTiming) -> u128 {
    image_timing.linework_ms + image_timing.contour_ms + image_timing.barb_ms
}

#[derive(Debug, Clone)]
pub struct HrrrDerivedLiveArtifact {
    pub recipe_slug: String,
    pub title: String,
    pub field: Field2D,
    pub request: MapRenderRequest,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct DerivedLiveArtifactBuildTiming {
    pub compute_fields_ms: u128,
    pub request_base_build_ms: u128,
    pub native_contour_fill_ms: u128,
    #[serde(default)]
    pub native_contour_projected_points_ms: u128,
    #[serde(default)]
    pub native_contour_scalar_field_ms: u128,
    #[serde(default)]
    pub native_contour_fill_topology_ms: u128,
    #[serde(default)]
    pub native_contour_fill_geometry_ms: u128,
    #[serde(default)]
    pub native_contour_line_topology_ms: u128,
    #[serde(default)]
    pub native_contour_line_geometry_ms: u128,
    pub wind_overlay_build_ms: u128,
    pub total_ms: u128,
}

#[derive(Debug, Clone, Copy, Default)]
struct NativeContourBuildTiming {
    total_ms: u128,
    projected_points_ms: u128,
    scalar_field_ms: u128,
    fill_topology_ms: u128,
    fill_geometry_ms: u128,
    line_topology_ms: u128,
    line_geometry_ms: u128,
}

#[derive(Debug, Clone)]
pub struct ProfiledHrrrDerivedLiveArtifact {
    pub artifact: HrrrDerivedLiveArtifact,
    pub timing: DerivedLiveArtifactBuildTiming,
}

#[derive(Debug, Clone)]
pub(crate) struct PreparedSharedDerivedFields {
    grid: rustwx_core::LatLonGrid,
    projection: Option<rustwx_core::GridProjection>,
    computed: DerivedComputedFields,
    fetch_decode: Option<GenericSharedTiming>,
}

#[derive(Debug, Clone)]
struct NativeDerivedField {
    grid: rustwx_core::LatLonGrid,
    values: Vec<f64>,
}

impl DerivedBatchRequest {
    pub(crate) fn from_hrrr(request: &HrrrDerivedBatchRequest) -> Self {
        Self {
            model: ModelId::Hrrr,
            date_yyyymmdd: request.date_yyyymmdd.clone(),
            cycle_override_utc: request.cycle_override_utc,
            forecast_hour: request.forecast_hour,
            source: request.source,
            domain: request.domain.clone(),
            out_dir: request.out_dir.clone(),
            cache_root: request.cache_root.clone(),
            use_cache: request.use_cache,
            recipe_slugs: request.recipe_slugs.clone(),
            surface_product_override: None,
            pressure_product_override: None,
            source_mode: request.source_mode,
            allow_large_heavy_domain: request.allow_large_heavy_domain,
            contour_mode: request.contour_mode,
            native_fill_level_multiplier: request.native_fill_level_multiplier,
            output_width: request.output_width,
            output_height: request.output_height,
            png_compression: request.png_compression,
            place_label_overlay: request.place_label_overlay.clone(),
            output_suffix: None,
            subtitle_left_override: None,
            subtitle_right_override: None,
        }
    }

    fn png_write_options(&self) -> PngWriteOptions {
        PngWriteOptions {
            compression: self.png_compression,
        }
    }
}

pub fn supported_derived_recipe_slugs(model: ModelId) -> Vec<String> {
    match model {
        ModelId::Hrrr
        | ModelId::HrrrAk
        | ModelId::Gfs
        | ModelId::Gdas
        | ModelId::Gefs
        | ModelId::Aigfs
        | ModelId::Aigefs
        | ModelId::Hgefs
        | ModelId::EcmwfOpenData
        | ModelId::Aifs
        | ModelId::Rap
        | ModelId::Nam
        | ModelId::Hiresw
        | ModelId::Sref
        | ModelId::RrfsA
        | ModelId::RrfsPublic
        | ModelId::RrfsFireWx
        | ModelId::WrfGdex => supported_derived_recipe_inventory()
            .iter()
            .filter(|recipe| derived_recipe_supported_for_model(recipe, model))
            .map(|recipe| recipe.slug.to_string())
            .collect(),
        ModelId::Rtma | ModelId::Urma | ModelId::Href | ModelId::Nbm | ModelId::Refs => Vec::new(),
    }
}

fn derived_recipe_supported_for_model(
    recipe: &DerivedRecipeInventoryEntry,
    model: ModelId,
) -> bool {
    if model == ModelId::Aifs {
        return !matches!(
            recipe.slug,
            "sb_ecape_native_cape_ratio"
                | "ml_ecape_native_cape_ratio"
                | "mu_ecape_native_cape_ratio"
        );
    }
    true
}

pub fn run_derived_batch(
    request: &DerivedBatchRequest,
) -> Result<DerivedBatchReport, Box<dyn std::error::Error>> {
    let recipes = plan_derived_recipes(&request.recipe_slugs)?;
    let planned_routes = plan_native_thermo_routes_with_surface_product(
        request.model,
        &recipes,
        request.source_mode,
        request.surface_product_override.as_deref(),
    )?;
    let latest = resolve_derived_run(
        request,
        &planned_routes.compute_recipes,
        &planned_routes.heavy_recipes,
        &planned_routes.native_routes,
    )?;
    if planned_routes.output_recipes.is_empty() {
        return Ok(empty_derived_report(
            request,
            &latest,
            planned_routes.blockers,
        ));
    }
    if let Some(loaded) =
        maybe_load_rrfs_cropped_pair_for_derived(request, &latest, &planned_routes)?
    {
        return run_derived_batch_from_loaded_bundles(request, &recipes, &loaded);
    }
    let plan = build_derived_execution_plan(
        &latest,
        request.forecast_hour,
        request.surface_product_override.as_deref(),
        request.pressure_product_override.as_deref(),
        derived_compute_recipes_need_pressure(&planned_routes.compute_recipes)
            || !planned_routes.heavy_recipes.is_empty(),
        !planned_routes.compute_recipes.is_empty(),
        &planned_routes.native_routes,
    );
    let loaded = load_execution_plan(
        plan,
        &BundleLoaderConfig {
            cache_root: request.cache_root.clone(),
            use_cache: request.use_cache,
        },
    )?;
    run_derived_batch_from_loaded_bundles(request, &recipes, &loaded)
}

fn maybe_load_rrfs_cropped_pair_for_derived(
    request: &DerivedBatchRequest,
    latest: &rustwx_models::LatestRun,
    planned_routes: &PlannedDerivedSourceRoutes,
) -> Result<Option<LoadedBundleSet>, Box<dyn std::error::Error>> {
    if !matches!(
        request.model,
        ModelId::RrfsA | ModelId::RrfsPublic | ModelId::RrfsFireWx
    ) || planned_routes.compute_recipes.is_empty()
        || !derived_compute_recipes_need_pressure(&planned_routes.compute_recipes)
        || !planned_routes.native_routes.is_empty()
    {
        return Ok(None);
    }

    let plan = build_derived_execution_plan(
        latest,
        request.forecast_hour,
        request.surface_product_override.as_deref(),
        request.pressure_product_override.as_deref(),
        true,
        true,
        &[],
    );
    let surface_planned = plan
        .bundle_for(
            CanonicalBundleDescriptor::SurfaceAnalysis,
            request.forecast_hour,
        )
        .ok_or("rrfs derived crop path missing surface bundle")?;
    let pressure_planned = plan
        .bundle_for(
            CanonicalBundleDescriptor::PressureAnalysis,
            request.forecast_hour,
        )
        .ok_or("rrfs derived crop path missing pressure bundle")?;

    let surface_fetch_start = Instant::now();
    let mut surface_file = fetch_family_file(
        request.model,
        latest.cycle.clone(),
        request.forecast_hour,
        latest.source,
        &surface_planned.resolved,
        &request.cache_root,
        request.use_cache,
    )?;
    let fetch_surface_ms = surface_fetch_start.elapsed().as_millis();

    let pressure_fetch_start = Instant::now();
    let mut pressure_file = fetch_family_file(
        request.model,
        latest.cycle.clone(),
        request.forecast_hour,
        latest.source,
        &pressure_planned.resolved,
        &request.cache_root,
        request.use_cache,
    )?;
    let fetch_pressure_ms = pressure_fetch_start.elapsed().as_millis();

    let surface_grid = decode_surface_grid(surface_file.bytes.as_slice())?;
    let projected = build_derived_projected_map_with_projection(
        request.model,
        &surface_grid
            .lat
            .iter()
            .copied()
            .map(|value| value as f32)
            .collect::<Vec<_>>(),
        &surface_grid
            .lon
            .iter()
            .copied()
            .map(|value| value as f32)
            .collect::<Vec<_>>(),
        surface_grid.projection.as_ref(),
        request.domain.bounds,
        map_frame_aspect_ratio(request.output_width, request.output_height, true, true),
    )?;

    let crop = match classify_projected_grid_intersection(
        surface_grid.nx,
        surface_grid.ny,
        &projected.projected_x,
        &projected.projected_y,
        &projected.extent,
        2,
    )? {
        ProjectedGridIntersection::Empty => {
            return Err(format!(
                "rrfs derived projected crop for domain '{}' produced an empty domain",
                request.domain.slug
            )
            .into());
        }
        ProjectedGridIntersection::Full => return Ok(None),
        ProjectedGridIntersection::Crop(crop) => crop,
    };

    let surface_decode_start = Instant::now();
    let surface_decode = load_or_decode_surface_cropped(
        &cropped_decode_cache_path(&request.cache_root, &surface_file.request, "surface", crop),
        surface_file.bytes.as_slice(),
        request.use_cache,
        crop,
    )?;
    let decode_surface_ms = surface_decode_start.elapsed().as_millis();

    let pressure_decode_start = Instant::now();
    let (pressure_decode, pressure_shape) = load_or_decode_pressure_cropped_with_shape(
        &cropped_decode_cache_path(
            &request.cache_root,
            &pressure_file.request,
            "pressure",
            crop,
        ),
        pressure_file.bytes.as_slice(),
        request.use_cache,
        crop,
    )?;
    let decode_pressure_ms = pressure_decode_start.elapsed().as_millis();

    crate::gridded::validate_pressure_decode_against_surface(
        &pressure_decode,
        pressure_shape,
        surface_decode.value.nx,
        surface_decode.value.ny,
    )?;

    surface_file.bytes.clear();
    surface_file.bytes.shrink_to_fit();
    pressure_file.bytes.clear();
    pressure_file.bytes.shrink_to_fit();
    let surface_fetch_bytes_len = surface_file.fetched.result.bytes.len();
    let pressure_fetch_bytes_len = pressure_file.fetched.result.bytes.len();

    let mut fetched = BTreeMap::new();
    fetched.insert(
        surface_planned.fetch_key(),
        FetchedBundleBytes {
            key: surface_planned.fetch_key(),
            file: surface_file,
            fetch_ms: fetch_surface_ms,
        },
    );
    fetched.insert(
        pressure_planned.fetch_key(),
        FetchedBundleBytes {
            key: pressure_planned.fetch_key(),
            file: pressure_file,
            fetch_ms: fetch_pressure_ms,
        },
    );

    let mut surface_decodes = BTreeMap::new();
    surface_decodes.insert(surface_planned.id.clone(), surface_decode);
    let mut pressure_decodes = BTreeMap::new();
    pressure_decodes.insert(pressure_planned.id.clone(), pressure_decode);

    Ok(Some(LoadedBundleSet {
        plan,
        latest: latest.clone(),
        forecast_hour: request.forecast_hour,
        fetched,
        fetch_failures: BTreeMap::new(),
        surface_decodes,
        pressure_decodes,
        bundle_failures: BTreeMap::new(),
        timing: LoadedBundleTiming {
            fetch_ms_total: fetch_surface_ms + fetch_pressure_ms,
            decode_surface_ms_total: decode_surface_ms,
            decode_pressure_ms_total: decode_pressure_ms,
            cropped_decode_profile: Some(CroppedDecodeProfile {
                source_grid_nx: surface_grid.nx,
                source_grid_ny: surface_grid.ny,
                crop_x_start: crop.x_start,
                crop_x_end: crop.x_end,
                crop_y_start: crop.y_start,
                crop_y_end: crop.y_end,
                cropped_grid_nx: crop.width(),
                cropped_grid_ny: crop.height(),
                surface_fetch_bytes_len,
                pressure_fetch_bytes_len,
            }),
        },
    }))
}

pub(crate) fn maybe_load_special_pair_for_derived(
    request: &DerivedBatchRequest,
    latest: &rustwx_models::LatestRun,
    planned_routes: &PlannedDerivedSourceRoutes,
) -> Result<Option<LoadedBundleSet>, Box<dyn std::error::Error>> {
    maybe_load_rrfs_cropped_pair_for_derived(request, latest, planned_routes)
}

fn cropped_decode_cache_path(
    cache_root: &std::path::Path,
    fetch: &rustwx_io::FetchRequest,
    name: &str,
    crop: crate::gridded::GridCrop,
) -> PathBuf {
    let mut path = decode_cache_path(cache_root, fetch, name);
    let stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or(name)
        .to_string();
    let suffix = format!(
        "{stem}_crop_{}_{}_{}_{}",
        crop.x_start, crop.x_end, crop.y_start, crop.y_end
    );
    path.set_file_name(format!("{suffix}.bin"));
    path
}

pub fn run_hrrr_derived_batch(
    request: &HrrrDerivedBatchRequest,
) -> Result<HrrrDerivedBatchReport, Box<dyn std::error::Error>> {
    Ok(into_hrrr_report(run_derived_batch(
        &DerivedBatchRequest::from_hrrr(request),
    )?))
}

fn run_derived_batch_from_loaded_bundles(
    request: &DerivedBatchRequest,
    recipes: &[DerivedRecipe],
    loaded: &LoadedBundleSet,
) -> Result<DerivedBatchReport, Box<dyn std::error::Error>> {
    run_derived_batch_from_loaded_bundles_with_precomputed(request, recipes, loaded, None)
}

fn run_derived_batch_from_loaded_bundles_with_precomputed(
    request: &DerivedBatchRequest,
    recipes: &[DerivedRecipe],
    loaded: &LoadedBundleSet,
    shared_precomputed: Option<&PreparedSharedDerivedFields>,
) -> Result<DerivedBatchReport, Box<dyn std::error::Error>> {
    fs::create_dir_all(&request.out_dir)?;
    if request.use_cache {
        fs::create_dir_all(&request.cache_root)?;
    }
    let total_start = Instant::now();
    let planned_routes = plan_native_thermo_routes_with_surface_product(
        request.model,
        recipes,
        request.source_mode,
        request.surface_product_override.as_deref(),
    )?;
    if planned_routes.output_recipes.is_empty() {
        return Ok(empty_derived_report(
            request,
            &loaded.latest,
            planned_routes.blockers,
        ));
    }
    let mut computed = DerivedComputedFields::default();
    let mut fetch_decode = None;
    let mut compute_ms = 0u128;
    let mut project_ms = 0u128;
    let mut native_extract_ms = 0u128;
    let native_compare_ms = 0u128;
    let mut heavy_timing = None;
    let mut memory_profile = None;
    let mut grid: Option<rustwx_core::LatLonGrid> = None;
    let mut grid_projection: Option<rustwx_core::GridProjection> = None;
    let mut projected: Option<ProjectedMap> = None;
    let input_fetches = build_planned_input_fetches(loaded);
    let input_fetch_keys = unique_input_fetch_keys(&input_fetches);
    let date_yyyymmdd = request.date_yyyymmdd.as_str();
    let cycle_utc = loaded.latest.cycle.hour_utc;
    let forecast_hour = request.forecast_hour;
    let source = loaded.latest.source;
    let model = request.model;
    let mut rendered_by_recipe = HashMap::<DerivedRecipe, DerivedRenderedRecipe>::new();
    let compute_needs_pressure =
        derived_compute_recipes_need_pressure(&planned_routes.compute_recipes);
    let needs_pair = compute_needs_pressure || !planned_routes.heavy_recipes.is_empty();

    if !planned_routes.compute_recipes.is_empty()
        && !compute_needs_pressure
        && planned_routes.heavy_recipes.is_empty()
    {
        let surface_planned = loaded
            .plan
            .bundle_for(
                CanonicalBundleDescriptor::SurfaceAnalysis,
                request.forecast_hour,
            )
            .ok_or("derived surface-only compute missing planned surface bundle")?;
        let surface_decode = loaded
            .surface_decodes
            .get(&surface_planned.id)
            .ok_or("derived surface-only compute missing decoded surface bundle")?;
        let surface = &surface_decode.value;
        let surface_grid = surface.core_grid()?;
        let project_start = Instant::now();
        let surface_projected = build_derived_projected_map_with_projection(
            request.model,
            &surface_grid.lat_deg,
            &surface_grid.lon_deg,
            surface.projection.as_ref(),
            request.domain.bounds,
            map_frame_aspect_ratio(request.output_width, request.output_height, true, true),
        )?;
        project_ms += project_start.elapsed().as_millis();
        match shared_precomputed {
            Some(shared) => {
                match classify_projected_grid_intersection(
                    shared.grid.shape.nx,
                    shared.grid.shape.ny,
                    &surface_projected.projected_x,
                    &surface_projected.projected_y,
                    &surface_projected.extent,
                    2,
                )? {
                    ProjectedGridIntersection::Empty => {
                        return Err(format!(
                            "derived projected crop for domain '{}' produced an empty domain",
                            request.domain.slug
                        )
                        .into());
                    }
                    ProjectedGridIntersection::Full => {
                        grid = Some(shared.grid.clone());
                        grid_projection = shared.projection.clone();
                        projected = Some(surface_projected.clone());
                        computed = shared.computed.clone();
                    }
                    ProjectedGridIntersection::Crop(crop) => {
                        let derived_grid = crop_latlon_grid(&shared.grid, crop)?;
                        let derived_projected = build_derived_projected_map_with_projection(
                            request.model,
                            &derived_grid.lat_deg,
                            &derived_grid.lon_deg,
                            shared.projection.as_ref(),
                            request.domain.bounds,
                            map_frame_aspect_ratio(
                                request.output_width,
                                request.output_height,
                                true,
                                true,
                            ),
                        )?;
                        grid = Some(derived_grid);
                        grid_projection = shared.projection.clone();
                        projected = Some(derived_projected);
                        computed =
                            crop_computed_fields(&shared.computed, shared.grid.shape.nx, crop);
                    }
                }
                fetch_decode = shared.fetch_decode.clone();
            }
            None => {
                let compute_start = Instant::now();
                computed =
                    compute_surface_only_derived_fields(surface, &planned_routes.compute_recipes)?;
                compute_ms += compute_start.elapsed().as_millis();
                grid = Some(surface_grid);
                grid_projection = surface.projection.clone();
                projected = Some(surface_projected);
            }
        }
    }

    if needs_pair {
        let (surface_planned, surface_decode, pressure_planned, pressure_decode) = loaded
            .require_surface_pressure_pair()
            .map_err(|err| format!("derived surface/pressure pair unavailable: {err}"))?;
        let full_surface = &surface_decode.value;
        let full_pressure = &pressure_decode.value;
        if !planned_routes.compute_recipes.is_empty() {
            memory_profile = build_derived_memory_profile(
                request.model,
                &planned_routes.compute_recipes,
                full_surface,
                full_pressure,
                loaded.timing.cropped_decode_profile,
            );
        }
        let owned_full_grid = full_surface.core_grid()?;
        let project_start = Instant::now();
        let full_projected = build_derived_projected_map_with_projection(
            request.model,
            &owned_full_grid.lat_deg,
            &owned_full_grid.lon_deg,
            full_surface.projection.as_ref(),
            request.domain.bounds,
            map_frame_aspect_ratio(request.output_width, request.output_height, true, true),
        )?;
        match shared_precomputed {
            Some(shared) => {
                match classify_projected_grid_intersection(
                    shared.grid.shape.nx,
                    shared.grid.shape.ny,
                    &full_projected.projected_x,
                    &full_projected.projected_y,
                    &full_projected.extent,
                    2,
                )? {
                    ProjectedGridIntersection::Empty => {
                        return Err(format!(
                            "derived projected crop for domain '{}' produced an empty domain",
                            request.domain.slug
                        )
                        .into());
                    }
                    ProjectedGridIntersection::Full => {
                        grid = Some(shared.grid.clone());
                        grid_projection = shared.projection.clone();
                        projected = Some(full_projected.clone());
                        computed = shared.computed.clone();
                    }
                    ProjectedGridIntersection::Crop(crop) => {
                        let derived_grid = crop_latlon_grid(&shared.grid, crop)?;
                        let derived_projected = build_derived_projected_map_with_projection(
                            request.model,
                            &derived_grid.lat_deg,
                            &derived_grid.lon_deg,
                            full_surface.projection.as_ref(),
                            request.domain.bounds,
                            map_frame_aspect_ratio(
                                request.output_width,
                                request.output_height,
                                true,
                                true,
                            ),
                        )?;
                        grid = Some(derived_grid);
                        grid_projection = shared.projection.clone();
                        projected = Some(derived_projected);
                        computed =
                            crop_computed_fields(&shared.computed, shared.grid.shape.nx, crop);
                    }
                }
                fetch_decode = shared.fetch_decode.clone();
            }
            None => {
                if !planned_routes.compute_recipes.is_empty() {
                    let cropped = crate::gridded::crop_heavy_domain_for_projected_extent(
                        full_surface,
                        full_pressure,
                        &full_projected.projected_x,
                        &full_projected.projected_y,
                        &full_projected.extent,
                        2,
                    )?;
                    let (surface, pressure, derived_grid) = match cropped.as_ref() {
                        Some(cropped) => {
                            (&cropped.surface, &cropped.pressure, cropped.grid.clone())
                        }
                        None => (full_surface, full_pressure, owned_full_grid.clone()),
                    };

                    let derived_projected = if cropped.is_some() {
                        build_derived_projected_map_with_projection(
                            request.model,
                            &derived_grid.lat_deg,
                            &derived_grid.lon_deg,
                            surface.projection.as_ref(),
                            request.domain.bounds,
                            map_frame_aspect_ratio(
                                request.output_width,
                                request.output_height,
                                true,
                                true,
                            ),
                        )?
                    } else {
                        full_projected.clone()
                    };

                    let compute_start = Instant::now();
                    computed = compute_derived_fields_generic(
                        surface,
                        pressure,
                        &planned_routes.compute_recipes,
                    )?;
                    compute_ms += compute_start.elapsed().as_millis();
                    grid = Some(derived_grid);
                    grid_projection = surface.projection.clone();
                    projected = Some(derived_projected);
                }
                fetch_decode = Some(build_shared_timing_for_pair(
                    loaded,
                    surface_planned,
                    pressure_planned,
                )?);
            }
        }
        if !planned_routes.heavy_recipes.is_empty() {
            let (heavy_rendered, timing) = render_derived_heavy_recipes(
                request,
                &planned_routes.heavy_recipes,
                full_surface,
                full_pressure,
                &owned_full_grid,
                &full_projected,
                date_yyyymmdd,
                cycle_utc,
                forecast_hour,
                source,
                model,
                input_fetch_keys.clone(),
                DerivedRenderOverrides::default(),
            )?;
            heavy_timing = Some(timing);
            for recipe in heavy_rendered {
                let parsed = DerivedRecipe::parse(&recipe.recipe_slug).map_err(io::Error::other)?;
                rendered_by_recipe.insert(parsed, recipe);
            }
        }
        project_ms += project_start.elapsed().as_millis();
    }

    let computed = &computed;
    let mut native_thermo_artifacts = Vec::<NativeThermoArtifactReport>::new();

    for route in &planned_routes.native_routes {
        let native_planned = find_loaded_native_bundle(loaded, route.candidate.fetch_product)
            .ok_or_else(|| {
                format!(
                    "native thermo planner missed fetch for '{}' ({})",
                    route.recipe.slug(),
                    route.candidate.fetch_product
                )
            })?;
        let fetched = loaded
            .fetched_for(native_planned)
            .ok_or_else(|| format!("native thermo fetch missing for {}", route.recipe.slug()))?;
        let extract_start = Instant::now();
        let native_field =
            extract_native_derived_field(request.model, route.native_recipe, fetched)?.ok_or_else(
                || {
                    format!(
                        "native derived field '{}' not found in {}",
                        route.recipe.slug(),
                        route.candidate.fetch_product
                    )
                },
            )?;
        let native_field = crop_native_derived_field(&native_field, request.domain.bounds)?;
        native_extract_ms += extract_start.elapsed().as_millis();

        let needs_native_projection = grid
            .as_ref()
            .map(|existing| !latlon_grids_match(existing, &native_field.grid))
            .unwrap_or(true);
        let native_projected = if needs_native_projection {
            let project_start = Instant::now();
            let native_projected = build_projected_map(
                &native_field.grid.lat_deg,
                &native_field.grid.lon_deg,
                request.domain.bounds,
                map_frame_aspect_ratio(request.output_width, request.output_height, true, true),
            )?;
            project_ms += project_start.elapsed().as_millis();
            if grid.is_none() {
                grid = Some(native_field.grid.clone());
                projected = Some(native_projected.clone());
            }
            native_projected
        } else {
            projected
                .as_ref()
                .ok_or("native thermo projection missing during main render")?
                .clone()
        };
        let output_path = request.out_dir.join(format!(
            "rustwx_{}_{}_{}z_f{:03}_{}_{}.png",
            model.as_str().replace('-', "_"),
            request.date_yyyymmdd,
            cycle_utc,
            request.forecast_hour,
            request.domain.slug,
            route.recipe.slug(),
        ));
        let render_start = Instant::now();
        let render_artifact = build_native_render_artifact(
            route.recipe,
            &native_field.grid,
            &native_projected,
            request.domain.bounds,
            date_yyyymmdd,
            cycle_utc,
            forecast_hour,
            source,
            model,
            request.output_width,
            request.output_height,
            native_field.values.clone(),
            request.contour_mode,
            request.native_fill_level_multiplier,
        )?;
        let HrrrDerivedLiveArtifact {
            recipe_slug,
            title: _,
            field: _,
            request: mut render_request,
        } = render_artifact;
        let title = derived_title_for_request(request, route.recipe.title());
        render_request.title = Some(title.clone());
        if let Some(overlay) = request.place_label_overlay.as_ref() {
            crate::apply_place_label_overlay_with_density_styling(
                &mut render_request,
                overlay,
                &request.domain,
                &native_field.grid.lat_deg,
                &native_field.grid.lon_deg,
                None,
            )?;
        }
        let save_timing = save_png_profile_with_options(
            &render_request,
            &output_path,
            &request.png_write_options(),
        )?;
        let render_ms = render_start.elapsed().as_millis();
        let content_identity = artifact_identity_from_path(&output_path)?;
        rendered_by_recipe.insert(
            route.recipe,
            DerivedRenderedRecipe {
                recipe_slug,
                title,
                source_route: route.source_route,
                output_path,
                content_identity,
                input_fetch_keys: input_fetch_keys.clone(),
                timing: DerivedRecipeTiming {
                    render_to_image_ms: save_timing.png_timing.render_to_image_ms,
                    data_layer_draw_ms: derived_data_layer_draw_ms(
                        &save_timing.png_timing.image_timing,
                    ),
                    overlay_draw_ms: derived_overlay_draw_ms(&save_timing.png_timing.image_timing),
                    render_state_prep_ms: save_timing.state_timing.state_prep_ms,
                    png_encode_ms: save_timing.png_timing.png_encode_ms,
                    file_write_ms: save_timing.file_write_ms,
                    render_ms,
                    total_ms: render_ms,
                    state_timing: save_timing.state_timing,
                    image_timing: save_timing.png_timing.image_timing,
                },
            },
        );

        native_thermo_artifacts.push(NativeThermoArtifactReport {
            recipe_slug: route.recipe.slug().to_string(),
            source_route: route.source_route,
            semantics: route.candidate.semantics,
            auto_eligible: route.candidate.auto_eligible,
            native_label: route.candidate.label.to_string(),
            native_detail: route.candidate.detail.to_string(),
            native_fetch_product: route.candidate.fetch_product.to_string(),
        });
    }

    let derived_output_recipes = planned_routes
        .output_recipes
        .iter()
        .copied()
        .filter(|recipe| !rendered_by_recipe.contains_key(recipe))
        .collect::<Vec<_>>();
    if !derived_output_recipes.is_empty() {
        let render_parallelism = png_render_parallelism(derived_output_recipes.len());
        let grid_ref = grid
            .as_ref()
            .ok_or("derived render requested but no grid was prepared")?;
        let projection_ref = grid_projection.as_ref();
        let projected_ref = projected
            .as_ref()
            .ok_or("derived render requested but no projection was prepared")?;
        let rendered = if render_parallelism <= 1 {
            let mut rendered = Vec::with_capacity(derived_output_recipes.len());
            for recipe in derived_output_recipes.iter().copied() {
                rendered.push(render_derived_output_recipe(
                    request,
                    recipe,
                    grid_ref,
                    projection_ref,
                    projected_ref,
                    date_yyyymmdd,
                    cycle_utc,
                    forecast_hour,
                    source,
                    model,
                    computed,
                    input_fetch_keys.clone(),
                    DerivedRenderOverrides::default(),
                )?);
            }
            rendered
        } else {
            thread::scope(|scope| -> Result<Vec<DerivedRenderedRecipe>, io::Error> {
                let next_index = Arc::new(AtomicUsize::new(0));
                let recipe_count = derived_output_recipes.len();
                let mut handles = Vec::with_capacity(render_parallelism);
                for _ in 0..render_parallelism {
                    let next_index = Arc::clone(&next_index);
                    let lane_projection = grid_projection.clone();
                    let worker_fetch_keys = input_fetch_keys.clone();
                    let recipes = &derived_output_recipes;
                    handles.push(scope.spawn(move || {
                        let mut rendered = Vec::new();
                        loop {
                            let index = next_index.fetch_add(1, Ordering::Relaxed);
                            let Some(recipe) = recipes.get(index).copied() else {
                                break;
                            };
                            let lane_fetch_keys = worker_fetch_keys.clone();
                            rendered.push(render_derived_output_recipe(
                                request,
                                recipe,
                                grid_ref,
                                lane_projection.as_ref(),
                                projected_ref,
                                date_yyyymmdd,
                                cycle_utc,
                                forecast_hour,
                                source,
                                model,
                                computed,
                                lane_fetch_keys,
                                DerivedRenderOverrides::default(),
                            )?);
                        }
                        Ok(rendered)
                    }));
                }

                let mut rendered = Vec::with_capacity(recipe_count);
                for handle in handles {
                    rendered.extend(join_render_job(handle)?);
                }
                Ok(rendered)
            })
            .map_err(|err| -> Box<dyn std::error::Error> { Box::new(err) })?
        };
        for recipe in rendered {
            let parsed = DerivedRecipe::parse(&recipe.recipe_slug).map_err(io::Error::other)?;
            rendered_by_recipe.insert(parsed, recipe);
        }
    }

    let rendered = planned_routes
        .output_recipes
        .iter()
        .map(|recipe| {
            rendered_by_recipe
                .remove(recipe)
                .ok_or_else(|| format!("derived renderer missed recipe '{}'", recipe.slug()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(DerivedBatchReport {
        model: request.model,
        date_yyyymmdd: request.date_yyyymmdd.clone(),
        cycle_utc,
        forecast_hour: request.forecast_hour,
        source: loaded.latest.source,
        domain: request.domain.clone(),
        input_fetches,
        shared_timing: DerivedSharedTiming {
            fetch_decode,
            compute_ms,
            project_ms,
            native_extract_ms,
            native_compare_ms,
            memory_profile,
            heavy_timing,
        },
        recipes: rendered,
        source_mode: request.source_mode,
        blockers: planned_routes.blockers,
        native_thermo_artifacts,
        total_ms: total_start.elapsed().as_millis(),
    })
}

/// Run the HRRR derived lane consuming a planner-loaded bundle set.
/// Used by the unified `hrrr_non_ecape_hour` runner so direct + derived
/// + windowed all share one fetch+decode pass.
pub(crate) fn run_model_derived_batch_from_loaded(
    request: &DerivedBatchRequest,
    recipes: &[DerivedRecipe],
    loaded: &LoadedBundleSet,
) -> Result<HrrrDerivedBatchReport, Box<dyn std::error::Error>> {
    let report = run_derived_batch_from_loaded_bundles(request, recipes, loaded)?;
    Ok(into_hrrr_report(report))
}

pub(crate) fn run_model_derived_batch_from_loaded_with_precomputed(
    request: &DerivedBatchRequest,
    recipes: &[DerivedRecipe],
    loaded: &LoadedBundleSet,
    prepared: &PreparedSharedDerivedFields,
) -> Result<HrrrDerivedBatchReport, Box<dyn std::error::Error>> {
    let mut report = run_derived_batch_from_loaded_bundles_with_precomputed(
        request,
        recipes,
        loaded,
        Some(prepared),
    )?;
    report.shared_timing.compute_ms = 0;
    Ok(into_hrrr_report(report))
}

pub(crate) fn run_model_derived_batch_without_loaded(
    request: &DerivedBatchRequest,
    recipes: &[DerivedRecipe],
    latest: &rustwx_models::LatestRun,
) -> Result<HrrrDerivedBatchReport, Box<dyn std::error::Error>> {
    let planned_routes = plan_native_thermo_routes_with_surface_product(
        request.model,
        recipes,
        request.source_mode,
        request.surface_product_override.as_deref(),
    )?;
    let report = empty_derived_report(request, latest, planned_routes.blockers);
    Ok(into_hrrr_report(report))
}

pub(crate) fn prepare_shared_derived_fields(
    request: &DerivedBatchRequest,
    recipes: &[DerivedRecipe],
    loaded: &LoadedBundleSet,
) -> Result<Option<PreparedSharedDerivedFields>, Box<dyn std::error::Error>> {
    let planned_routes = plan_native_thermo_routes_with_surface_product(
        request.model,
        recipes,
        request.source_mode,
        request.surface_product_override.as_deref(),
    )?;
    if planned_routes.compute_recipes.is_empty() {
        return Ok(None);
    }

    if !derived_compute_recipes_need_pressure(&planned_routes.compute_recipes) {
        let surface_planned = loaded
            .plan
            .bundle_for(
                CanonicalBundleDescriptor::SurfaceAnalysis,
                request.forecast_hour,
            )
            .ok_or("derived surface-only shared prepare missing planned surface bundle")?;
        let surface_decode = loaded
            .surface_decodes
            .get(&surface_planned.id)
            .ok_or("derived surface-only shared prepare missing decoded surface bundle")?;
        let computed = compute_surface_only_derived_fields(
            &surface_decode.value,
            &planned_routes.compute_recipes,
        )?;
        return Ok(Some(PreparedSharedDerivedFields {
            grid: surface_decode.value.core_grid()?,
            projection: surface_decode.value.projection.clone(),
            computed,
            fetch_decode: None,
        }));
    }

    let (surface_planned, surface_decode, pressure_planned, pressure_decode) = loaded
        .require_surface_pressure_pair()
        .map_err(|err| format!("derived surface/pressure pair unavailable: {err}"))?;
    let computed = compute_derived_fields_generic(
        &surface_decode.value,
        &pressure_decode.value,
        &planned_routes.compute_recipes,
    )?;
    let fetch_decode = build_shared_timing_for_pair(loaded, surface_planned, pressure_planned)?;
    Ok(Some(PreparedSharedDerivedFields {
        grid: surface_decode.value.core_grid()?,
        projection: surface_decode.value.projection.clone(),
        computed,
        fetch_decode: Some(GenericSharedTiming {
            fetch_surface_ms: 0,
            fetch_pressure_ms: 0,
            decode_surface_ms: 0,
            decode_pressure_ms: 0,
            fetch_surface_cache_hit: fetch_decode.fetch_surface_cache_hit,
            fetch_pressure_cache_hit: fetch_decode.fetch_pressure_cache_hit,
            decode_surface_cache_hit: fetch_decode.decode_surface_cache_hit,
            decode_pressure_cache_hit: fetch_decode.decode_pressure_cache_hit,
            surface_fetch: fetch_decode.surface_fetch,
            pressure_fetch: fetch_decode.pressure_fetch,
        }),
    }))
}

fn into_hrrr_report(report: DerivedBatchReport) -> HrrrDerivedBatchReport {
    HrrrDerivedBatchReport {
        date_yyyymmdd: report.date_yyyymmdd,
        cycle_utc: report.cycle_utc,
        forecast_hour: report.forecast_hour,
        source: report.source,
        domain: report.domain,
        input_fetches: report.input_fetches,
        shared_timing: report.shared_timing,
        recipes: report.recipes,
        source_mode: report.source_mode,
        blockers: report.blockers,
        native_thermo_artifacts: report.native_thermo_artifacts,
        total_ms: report.total_ms,
    }
}

fn derived_compute_source_route(
    recipe: DerivedRecipe,
    mode: ProductSourceMode,
) -> Option<ProductSourceRoute> {
    match mode {
        ProductSourceMode::Canonical => Some(ProductSourceRoute::CanonicalDerived),
        ProductSourceMode::Fastest => cheap_fastest_route(recipe),
    }
}

fn build_derived_memory_profile(
    model: ModelId,
    compute_recipes: &[DerivedRecipe],
    surface: &GenericSurfaceFields,
    pressure: &GenericPressureFields,
    cropped_profile: Option<CroppedDecodeProfile>,
) -> Option<DerivedMemoryProfile> {
    if model != ModelId::RrfsA {
        return None;
    }
    let cropped = cropped_profile?;
    let requirements = DerivedRequirements::from_recipes(compute_recipes);
    let pressure_level_count = pressure.pressure_levels_hpa.len();
    let thermo_volume_points = surface.nx * surface.ny * pressure_level_count;
    let canonical_pressure_3d_pa_bytes_estimate = if requirements.needs_volume() {
        pressure
            .pressure_3d_pa
            .as_ref()
            .map(|values| values.len() * std::mem::size_of::<f64>())
            .unwrap_or(pressure_level_count * std::mem::size_of::<f64>())
    } else {
        0
    };
    let canonical_height_agl_3d_bytes_estimate = if requirements.needs_height_agl() {
        thermo_volume_points * std::mem::size_of::<f64>()
    } else {
        0
    };
    Some(DerivedMemoryProfile {
        source_grid_nx: cropped.source_grid_nx,
        source_grid_ny: cropped.source_grid_ny,
        cropped_grid_nx: cropped.cropped_grid_nx,
        cropped_grid_ny: cropped.cropped_grid_ny,
        crop_x_start: cropped.crop_x_start,
        crop_x_end: cropped.crop_x_end,
        crop_y_start: cropped.crop_y_start,
        crop_y_end: cropped.crop_y_end,
        surface_fetch_bytes_len: cropped.surface_fetch_bytes_len,
        pressure_fetch_bytes_len: cropped.pressure_fetch_bytes_len,
        cropped_surface_decoded_bytes_estimate: surface.decoded_bytes_estimate(),
        cropped_pressure_decoded_bytes_estimate: pressure.decoded_bytes_estimate(),
        cropped_decoded_total_bytes_estimate: surface.decoded_bytes_estimate()
            + pressure.decoded_bytes_estimate(),
        pressure_level_count,
        thermo_volume_points,
        compute_recipe_count: compute_recipes.len(),
        needs_volume: requirements.needs_volume(),
        needs_height_agl: requirements.needs_height_agl(),
        canonical_pressure_3d_pa_bytes_estimate,
        canonical_height_agl_3d_bytes_estimate,
        canonical_shared_volume_work_bytes_estimate: canonical_pressure_3d_pa_bytes_estimate
            + canonical_height_agl_3d_bytes_estimate,
    })
}

fn empty_derived_report(
    request: &DerivedBatchRequest,
    latest: &rustwx_models::LatestRun,
    blockers: Vec<DerivedRecipeBlocker>,
) -> DerivedBatchReport {
    DerivedBatchReport {
        model: request.model,
        date_yyyymmdd: latest.cycle.date_yyyymmdd.clone(),
        cycle_utc: latest.cycle.hour_utc,
        forecast_hour: request.forecast_hour,
        source: latest.source,
        domain: request.domain.clone(),
        input_fetches: Vec::new(),
        shared_timing: DerivedSharedTiming {
            fetch_decode: None,
            compute_ms: 0,
            project_ms: 0,
            native_extract_ms: 0,
            native_compare_ms: 0,
            memory_profile: None,
            heavy_timing: None,
        },
        recipes: Vec::new(),
        source_mode: request.source_mode,
        blockers,
        native_thermo_artifacts: Vec::new(),
        total_ms: 0,
    }
}

fn unique_input_fetch_keys(fetches: &[PublishedFetchIdentity]) -> Vec<String> {
    let mut keys = Vec::with_capacity(fetches.len());
    for fetch in fetches {
        if !keys.contains(&fetch.fetch_key) {
            keys.push(fetch.fetch_key.clone());
        }
    }
    keys
}

fn find_loaded_native_bundle<'a>(
    loaded: &'a LoadedBundleSet,
    fetch_product: &str,
) -> Option<&'a PlannedBundle> {
    loaded.plan.bundles.iter().find(|bundle| {
        bundle.id.bundle == CanonicalBundleDescriptor::NativeAnalysis
            && bundle.fetch_key().native_product == fetch_product
    })
}

fn extract_native_derived_field(
    model: ModelId,
    native_recipe: NativeDerivedRecipe,
    fetched: &FetchedBundleBytes,
) -> Result<Option<NativeDerivedField>, Box<dyn std::error::Error>> {
    match native_recipe {
        NativeDerivedRecipe::Thermo(recipe) => {
            let Some(field) = extract_native_thermo_field(model, recipe, &fetched.file.bytes)?
            else {
                return Ok(None);
            };
            Ok(Some(NativeDerivedField {
                grid: field.grid,
                values: field.values,
            }))
        }
        NativeDerivedRecipe::WrfGdexScalar { .. } => {
            if model == ModelId::WrfGdex {
                return Err("WRF/GDEX NetCDF support is not available in this build".into());
            }
            Ok(None)
        }
        NativeDerivedRecipe::WrfGdexVectorMagnitude { .. } => {
            if model == ModelId::WrfGdex {
                return Err("WRF/GDEX NetCDF support is not available in this build".into());
            }
            Ok(None)
        }
    }
}

fn crop_native_derived_field(
    field: &NativeDerivedField,
    bounds: (f64, f64, f64, f64),
) -> Result<NativeDerivedField, Box<dyn std::error::Error>> {
    let nx = field.grid.shape.nx;
    let ny = field.grid.shape.ny;
    let mut min_x = nx;
    let mut max_x = 0usize;
    let mut min_y = ny;
    let mut max_y = 0usize;
    let mut found = false;

    for y in 0..ny {
        let row_offset = y * nx;
        for x in 0..nx {
            let idx = row_offset + x;
            let lat = f64::from(field.grid.lat_deg[idx]);
            let lon = f64::from(field.grid.lon_deg[idx]);
            if point_in_geographic_bounds(lon, lat, bounds) {
                min_x = min_x.min(x);
                max_x = max_x.max(x);
                min_y = min_y.min(y);
                max_y = max_y.max(y);
                found = true;
            }
        }
    }

    if !found {
        return Err("requested native derived crop produced an empty domain".into());
    }

    if min_x == 0 && max_x + 1 == nx && min_y == 0 && max_y + 1 == ny {
        return Ok(field.clone());
    }

    let pad_cells = if inverse_raster_projection_for_grid(None, bounds, &field.grid).is_some() {
        inverse_raster_crop_pad_cells()
    } else {
        0
    };
    let crop = GridCrop {
        x_start: min_x.saturating_sub(pad_cells),
        x_end: (max_x + 1 + pad_cells).min(nx),
        y_start: min_y.saturating_sub(pad_cells),
        y_end: (max_y + 1 + pad_cells).min(ny),
    };

    Ok(NativeDerivedField {
        grid: crop_latlon_grid(&field.grid, crop)?,
        values: crop_values_f64(&field.values, field.grid.shape.nx, crop),
    })
}

fn inverse_raster_crop_pad_cells() -> usize {
    std::env::var("RUSTWX_INVERSE_RASTER_CROP_PAD_CELLS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(1000)
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

fn build_native_render_artifact(
    recipe: DerivedRecipe,
    grid: &rustwx_core::LatLonGrid,
    projected: &ProjectedMap,
    domain_bounds: (f64, f64, f64, f64),
    date_yyyymmdd: &str,
    cycle_utc: u8,
    forecast_hour: u16,
    source: SourceId,
    model: ModelId,
    output_width: u32,
    output_height: u32,
    values: Vec<f64>,
    contour_mode: NativeContourRenderMode,
    native_fill_level_multiplier: usize,
) -> Result<HrrrDerivedLiveArtifact, Box<dyn std::error::Error>> {
    let computed = computed_from_native_values(recipe, values)?;
    build_render_artifact(
        recipe,
        grid,
        projected,
        domain_bounds,
        date_yyyymmdd,
        cycle_utc,
        forecast_hour,
        source,
        model,
        output_width,
        output_height,
        &computed,
        contour_mode,
        native_fill_level_multiplier,
    )
}

fn computed_from_native_values(
    recipe: DerivedRecipe,
    values: Vec<f64>,
) -> Result<DerivedComputedFields, Box<dyn std::error::Error>> {
    let mut computed = DerivedComputedFields::default();
    match recipe {
        DerivedRecipe::Sbcape => computed.sbcape_jkg = Some(values),
        DerivedRecipe::Sbcin => computed.sbcin_jkg = Some(values),
        DerivedRecipe::Sblcl => computed.sblcl_m = Some(values),
        DerivedRecipe::Mlcape => computed.mlcape_jkg = Some(values),
        DerivedRecipe::Mlcin => computed.mlcin_jkg = Some(values),
        DerivedRecipe::Mucape => computed.mucape_jkg = Some(values),
        DerivedRecipe::Mucin => computed.mucin_jkg = Some(values),
        DerivedRecipe::LiftedIndex => computed.lifted_index_c = Some(values),
        DerivedRecipe::BulkShear01km => computed.shear_01km_kt = Some(values),
        DerivedRecipe::BulkShear06km => computed.shear_06km_kt = Some(values),
        DerivedRecipe::Srh01km => computed.srh_01km_m2s2 = Some(values),
        DerivedRecipe::Srh03km => computed.srh_03km_m2s2 = Some(values),
        _ => {
            return Err(format!(
                "recipe '{}' does not support native derived rendering",
                recipe.slug()
            )
            .into());
        }
    }
    Ok(computed)
}

/// Build a single derived render artifact for an HRRR live-preview
/// surface. Takes the planner-decoded generic surface/pressure types so
/// callers can reuse a `LoadedBundleSet` rather than re-decoding HRRR
/// natively. Reroutes through the same generic compute kernel as the
/// batched derived lane.
pub fn build_hrrr_live_derived_artifact(
    recipe_slug: &str,
    surface: &GenericSurfaceFields,
    pressure: &GenericPressureFields,
    grid: &rustwx_core::LatLonGrid,
    projected: &ProjectedMap,
    domain_bounds: (f64, f64, f64, f64),
    date_yyyymmdd: &str,
    cycle_utc: u8,
    forecast_hour: u16,
    source: SourceId,
) -> Result<HrrrDerivedLiveArtifact, Box<dyn std::error::Error>> {
    build_hrrr_live_derived_artifact_with_render_mode(
        recipe_slug,
        surface,
        pressure,
        grid,
        projected,
        domain_bounds,
        date_yyyymmdd,
        cycle_utc,
        forecast_hour,
        source,
        NativeContourRenderMode::Automatic,
        1,
    )
}

pub fn build_hrrr_live_derived_artifact_with_render_mode(
    recipe_slug: &str,
    surface: &GenericSurfaceFields,
    pressure: &GenericPressureFields,
    grid: &rustwx_core::LatLonGrid,
    projected: &ProjectedMap,
    domain_bounds: (f64, f64, f64, f64),
    date_yyyymmdd: &str,
    cycle_utc: u8,
    forecast_hour: u16,
    source: SourceId,
    contour_mode: NativeContourRenderMode,
    native_fill_level_multiplier: usize,
) -> Result<HrrrDerivedLiveArtifact, Box<dyn std::error::Error>> {
    let recipe =
        DerivedRecipe::parse(recipe_slug).map_err(|err| format!("{recipe_slug}: {err}"))?;
    with_prepared_live_derived_domain(
        surface,
        pressure,
        grid,
        projected,
        domain_bounds,
        OUTPUT_WIDTH,
        OUTPUT_HEIGHT,
        |surface, pressure, grid, projected| {
            let computed = compute_derived_fields_generic(surface, pressure, &[recipe])?;
            build_render_artifact_with_contour_mode(
                recipe,
                grid,
                projected,
                domain_bounds,
                date_yyyymmdd,
                cycle_utc,
                forecast_hour,
                source,
                ModelId::Hrrr,
                OUTPUT_WIDTH,
                OUTPUT_HEIGHT,
                &computed,
                contour_mode,
                native_fill_level_multiplier,
            )
        },
    )
}

pub fn build_hrrr_live_derived_artifact_profiled(
    recipe_slug: &str,
    surface: &GenericSurfaceFields,
    pressure: &GenericPressureFields,
    grid: &rustwx_core::LatLonGrid,
    projected: &ProjectedMap,
    domain_bounds: (f64, f64, f64, f64),
    date_yyyymmdd: &str,
    cycle_utc: u8,
    forecast_hour: u16,
    source: SourceId,
    contour_mode: NativeContourRenderMode,
) -> Result<ProfiledHrrrDerivedLiveArtifact, Box<dyn std::error::Error>> {
    let total_start = Instant::now();
    let recipe =
        DerivedRecipe::parse(recipe_slug).map_err(|err| format!("{recipe_slug}: {err}"))?;
    with_prepared_live_derived_domain(
        surface,
        pressure,
        grid,
        projected,
        domain_bounds,
        OUTPUT_WIDTH,
        OUTPUT_HEIGHT,
        |surface, pressure, grid, projected| {
            let compute_start = Instant::now();
            let computed = compute_derived_fields_generic(surface, pressure, &[recipe])?;
            let compute_fields_ms = compute_start.elapsed().as_millis();
            let (artifact, mut timing) = build_render_artifact_with_contour_mode_profiled(
                recipe,
                grid,
                projected,
                domain_bounds,
                date_yyyymmdd,
                cycle_utc,
                forecast_hour,
                source,
                ModelId::Hrrr,
                OUTPUT_WIDTH,
                OUTPUT_HEIGHT,
                &computed,
                contour_mode,
                1,
            )?;
            timing.compute_fields_ms = compute_fields_ms;
            timing.total_ms = total_start.elapsed().as_millis();
            Ok(ProfiledHrrrDerivedLiveArtifact { artifact, timing })
        },
    )
}

fn with_prepared_live_derived_domain<T>(
    surface: &GenericSurfaceFields,
    pressure: &GenericPressureFields,
    grid: &rustwx_core::LatLonGrid,
    projected: &ProjectedMap,
    domain_bounds: (f64, f64, f64, f64),
    output_width: u32,
    output_height: u32,
    build: impl FnOnce(
        &GenericSurfaceFields,
        &GenericPressureFields,
        &rustwx_core::LatLonGrid,
        &ProjectedMap,
    ) -> Result<T, Box<dyn std::error::Error>>,
) -> Result<T, Box<dyn std::error::Error>> {
    let cropped = crate::gridded::crop_heavy_domain_for_projected_extent(
        surface,
        pressure,
        &projected.projected_x,
        &projected.projected_y,
        &projected.extent,
        2,
    )?;
    if let Some(cropped) = cropped {
        let cropped_projected = build_projected_map_with_projection(
            &cropped.grid.lat_deg,
            &cropped.grid.lon_deg,
            cropped.surface.projection.as_ref(),
            domain_bounds,
            map_frame_aspect_ratio(output_width, output_height, true, true),
        )?;
        build(
            &cropped.surface,
            &cropped.pressure,
            &cropped.grid,
            &cropped_projected,
        )
    } else {
        build(surface, pressure, grid, projected)
    }
}

fn build_render_artifact(
    recipe: DerivedRecipe,
    grid: &rustwx_core::LatLonGrid,
    projected: &ProjectedMap,
    domain_bounds: (f64, f64, f64, f64),
    date_yyyymmdd: &str,
    cycle_utc: u8,
    forecast_hour: u16,
    source: SourceId,
    model: ModelId,
    output_width: u32,
    output_height: u32,
    computed: &DerivedComputedFields,
    contour_mode: NativeContourRenderMode,
    native_fill_level_multiplier: usize,
) -> Result<HrrrDerivedLiveArtifact, Box<dyn std::error::Error>> {
    build_render_artifact_with_contour_mode(
        recipe,
        grid,
        projected,
        domain_bounds,
        date_yyyymmdd,
        cycle_utc,
        forecast_hour,
        source,
        model,
        output_width,
        output_height,
        computed,
        contour_mode,
        native_fill_level_multiplier,
    )
}

fn build_render_artifact_with_contour_mode(
    recipe: DerivedRecipe,
    grid: &rustwx_core::LatLonGrid,
    projected: &ProjectedMap,
    domain_bounds: (f64, f64, f64, f64),
    date_yyyymmdd: &str,
    cycle_utc: u8,
    forecast_hour: u16,
    source: SourceId,
    model: ModelId,
    output_width: u32,
    output_height: u32,
    computed: &DerivedComputedFields,
    contour_mode: NativeContourRenderMode,
    native_fill_level_multiplier: usize,
) -> Result<HrrrDerivedLiveArtifact, Box<dyn std::error::Error>> {
    let (field, mut request) = match recipe {
        DerivedRecipe::Sbcape => weather_request(
            recipe,
            grid,
            "J/kg",
            required_values(&computed.sbcape_jkg, recipe, "sbcape_jkg")?.clone(),
            WeatherProduct::Sbcape,
        )?,
        DerivedRecipe::Sbcin => weather_request(
            recipe,
            grid,
            "J/kg",
            required_values(&computed.sbcin_jkg, recipe, "sbcin_jkg")?.clone(),
            WeatherProduct::Sbcin,
        )?,
        DerivedRecipe::Sblcl => weather_request(
            recipe,
            grid,
            "m",
            required_values(&computed.sblcl_m, recipe, "sblcl_m")?.clone(),
            WeatherProduct::Lcl,
        )?,
        DerivedRecipe::Mlcape => weather_request(
            recipe,
            grid,
            "J/kg",
            required_values(&computed.mlcape_jkg, recipe, "mlcape_jkg")?.clone(),
            WeatherProduct::Mlcape,
        )?,
        DerivedRecipe::Mlcin => weather_request(
            recipe,
            grid,
            "J/kg",
            required_values(&computed.mlcin_jkg, recipe, "mlcin_jkg")?.clone(),
            WeatherProduct::Mlcin,
        )?,
        DerivedRecipe::Mucape => weather_request(
            recipe,
            grid,
            "J/kg",
            required_values(&computed.mucape_jkg, recipe, "mucape_jkg")?.clone(),
            WeatherProduct::Mucape,
        )?,
        DerivedRecipe::Mucin => weather_request(
            recipe,
            grid,
            "J/kg",
            required_values(&computed.mucin_jkg, recipe, "mucin_jkg")?.clone(),
            WeatherProduct::Mucin,
        )?,
        DerivedRecipe::Dcape => custom_scale_request(
            recipe,
            grid,
            "J/kg",
            required_values(&computed.dcape_jkg, recipe, "dcape_jkg")?.clone(),
            range_step(0.0, 2501.0, 100.0),
            dcape_scale_colors(),
            ExtendMode::Max,
            Some(250.0),
        )?,
        DerivedRecipe::ThetaE2m10mWinds => palette_request(
            recipe,
            grid,
            "K",
            required_values(&computed.theta_e_2m_k, recipe, "theta_e_2m_k")?.clone(),
            WeatherPalette::Temperature,
            range_step(280.0, 381.0, 4.0),
            ExtendMode::Both,
            Some(8.0),
        )?,
        DerivedRecipe::Vpd2m => custom_scale_request(
            recipe,
            grid,
            "hPa",
            required_values(&computed.vpd_2m_hpa, recipe, "vpd_2m_hpa")?.clone(),
            range_step(0.0, 41.0, 2.0),
            vpd_scale_colors(),
            ExtendMode::Max,
            Some(4.0),
        )?,
        DerivedRecipe::DewpointDepression2m => custom_scale_request(
            recipe,
            grid,
            "degC",
            required_values(
                &computed.dewpoint_depression_2m_c,
                recipe,
                "dewpoint_depression_2m_c",
            )?
            .clone(),
            range_step(0.0, 41.0, 4.0),
            dewpoint_depression_scale_colors(),
            ExtendMode::Max,
            Some(8.0),
        )?,
        DerivedRecipe::Wetbulb2m => scale_request(
            recipe,
            grid,
            "degC",
            required_values(&computed.wetbulb_2m_c, recipe, "wetbulb_2m_c")?.clone(),
            surface_temperature_scale_c(0.5),
            Some(5.0),
        )?,
        DerivedRecipe::FireWeatherComposite => custom_scale_request(
            recipe,
            grid,
            "index",
            required_values(
                &computed.fire_weather_composite,
                recipe,
                "fire_weather_composite",
            )?
            .clone(),
            range_step(0.0, 101.0, 10.0),
            fire_weather_composite_scale_colors(),
            ExtendMode::Neither,
            Some(20.0),
        )?,
        DerivedRecipe::ApparentTemperature2m => derived_style_request(
            recipe,
            grid,
            "degC",
            required_values(
                &computed.apparent_temperature_2m_c,
                recipe,
                "apparent_temperature_2m_c",
            )?
            .clone(),
            DerivedProductStyle::ApparentTemperature,
        )?,
        DerivedRecipe::HeatIndex2m => palette_request(
            recipe,
            grid,
            "degC",
            required_values(&computed.heat_index_2m_c, recipe, "heat_index_2m_c")?.clone(),
            WeatherPalette::Temperature,
            range_step(-30.0, 51.0, 5.0),
            ExtendMode::Both,
            Some(5.0),
        )?,
        DerivedRecipe::WindChill2m => palette_request(
            recipe,
            grid,
            "degC",
            required_values(&computed.wind_chill_2m_c, recipe, "wind_chill_2m_c")?.clone(),
            WeatherPalette::Temperature,
            range_step(-40.0, 31.0, 5.0),
            ExtendMode::Both,
            Some(5.0),
        )?,
        DerivedRecipe::LiftedIndex => palette_request(
            recipe,
            grid,
            "degC",
            required_values(&computed.lifted_index_c, recipe, "lifted_index_c")?.clone(),
            WeatherPalette::Temperature,
            range_step(-12.0, 13.0, 1.0),
            ExtendMode::Both,
            Some(1.0),
        )?,
        DerivedRecipe::LapseRate700500 => weather_lapse_request(
            recipe,
            grid,
            required_values(
                &computed.lapse_rate_700_500_cpkm,
                recipe,
                "lapse_rate_700_500_cpkm",
            )?
            .clone(),
        )?,
        DerivedRecipe::LapseRate03km => weather_lapse_request(
            recipe,
            grid,
            required_values(
                &computed.lapse_rate_0_3km_cpkm,
                recipe,
                "lapse_rate_0_3km_cpkm",
            )?
            .clone(),
        )?,
        DerivedRecipe::BulkShear01km => palette_request(
            recipe,
            grid,
            "kt",
            required_values(&computed.shear_01km_kt, recipe, "shear_01km_kt")?.clone(),
            WeatherPalette::Winds,
            range_step(0.0, 85.0, 5.0),
            ExtendMode::Max,
            Some(5.0),
        )?,
        DerivedRecipe::BulkShear06km => palette_request(
            recipe,
            grid,
            "kt",
            required_values(&computed.shear_06km_kt, recipe, "shear_06km_kt")?.clone(),
            WeatherPalette::Winds,
            range_step(0.0, 85.0, 5.0),
            ExtendMode::Max,
            Some(5.0),
        )?,
        DerivedRecipe::Srh01km => weather_request(
            recipe,
            grid,
            "m^2/s^2",
            required_values(&computed.srh_01km_m2s2, recipe, "srh_01km_m2s2")?.clone(),
            WeatherProduct::Srh01km,
        )?,
        DerivedRecipe::Srh03km => weather_request(
            recipe,
            grid,
            "m^2/s^2",
            required_values(&computed.srh_03km_m2s2, recipe, "srh_03km_m2s2")?.clone(),
            WeatherProduct::Srh03km,
        )?,
        DerivedRecipe::Ehi01km => weather_request(
            recipe,
            grid,
            "dimensionless",
            required_values(&computed.ehi_01km, recipe, "ehi_01km")?.clone(),
            WeatherProduct::Ehi,
        )?,
        DerivedRecipe::Ehi03km => weather_request(
            recipe,
            grid,
            "dimensionless",
            required_values(&computed.ehi_03km, recipe, "ehi_03km")?.clone(),
            WeatherProduct::Ehi,
        )?,
        DerivedRecipe::StpFixed => weather_request(
            recipe,
            grid,
            "dimensionless",
            required_values(&computed.stp_fixed, recipe, "stp_fixed")?.clone(),
            WeatherProduct::StpFixed,
        )?,
        DerivedRecipe::ScpMu03km06kmProxy => weather_request(
            recipe,
            grid,
            "dimensionless",
            required_values(
                &computed.scp_mu_03km_06km_proxy,
                recipe,
                "scp_mu_03km_06km_proxy",
            )?
            .clone(),
            WeatherProduct::Scp,
        )?,
        DerivedRecipe::TemperatureAdvection700mb => palette_request(
            recipe,
            grid,
            "degC/hr",
            required_values(
                &computed.temperature_advection_700mb_cph,
                recipe,
                "temperature_advection_700mb_cph",
            )?
            .clone(),
            WeatherPalette::Temperature,
            range_step(-12.0, 13.0, 1.0),
            ExtendMode::Both,
            Some(1.0),
        )?,
        DerivedRecipe::TemperatureAdvection850mb => palette_request(
            recipe,
            grid,
            "degC/hr",
            required_values(
                &computed.temperature_advection_850mb_cph,
                recipe,
                "temperature_advection_850mb_cph",
            )?
            .clone(),
            WeatherPalette::Temperature,
            range_step(-12.0, 13.0, 1.0),
            ExtendMode::Both,
            Some(1.0),
        )?,
        DerivedRecipe::Sbecape
        | DerivedRecipe::Mlecape
        | DerivedRecipe::Muecape
        | DerivedRecipe::SbEcapeDerivedCapeRatio
        | DerivedRecipe::MlEcapeDerivedCapeRatio
        | DerivedRecipe::MuEcapeDerivedCapeRatio
        | DerivedRecipe::SbEcapeNativeCapeRatio
        | DerivedRecipe::MlEcapeNativeCapeRatio
        | DerivedRecipe::MuEcapeNativeCapeRatio
        | DerivedRecipe::Sbncape
        | DerivedRecipe::Sbecin
        | DerivedRecipe::Mlecin
        | DerivedRecipe::EcapeScp
        | DerivedRecipe::EcapeEhi01km
        | DerivedRecipe::EcapeEhi03km
        | DerivedRecipe::EcapeStp => {
            return Err(format!(
                "heavy derived recipe '{}' must render through the cropped ECAPE path",
                recipe.slug()
            )
            .into());
        }
    };

    request.width = output_width;
    request.height = output_height;
    request.chrome_scale = static_chrome_scale();
    request.supersample_factor = static_supersample_factor();
    request.supersample_sharpen = static_supersample_sharpen();
    crate::plot_design::StaticPlotDesign::new(domain_bounds, recipe.visual_mode())
        .apply_to_request(&mut request);
    request.title = Some(derived_title_for_model(model, recipe.title()));
    request.subtitle_left = Some(model_time_subtitle(
        model,
        date_yyyymmdd,
        cycle_utc,
        forecast_hour,
    ));
    request.subtitle_right = Some(source_subtitle(source));
    request.projected_domain = Some(ProjectedDomain {
        x: projected.projected_x.clone(),
        y: projected.projected_y.clone(),
        extent: projected.extent.clone(),
    });
    request.projected_lines = projected.lines.clone();
    request.projected_polygons = projected.polygons.clone();
    apply_source_raster_policy(source, &mut request);
    maybe_apply_native_contour_fill_for_mode(
        recipe,
        &mut request,
        contour_mode,
        native_fill_level_multiplier,
    )?;
    if matches!(recipe, DerivedRecipe::ThetaE2m10mWinds) {
        let u_kt = computed_surface_u10(computed, recipe)?;
        let v_kt = computed_surface_v10(computed, recipe)?;
        request.wind_barbs.push(surface_wind_barb_layer(
            grid,
            &projected.extent,
            &projected.projected_x,
            &projected.projected_y,
            &u_kt,
            &v_kt,
        ));
    }
    Ok(HrrrDerivedLiveArtifact {
        recipe_slug: recipe.slug().to_string(),
        title: recipe.title().to_string(),
        field,
        request,
    })
}

fn build_render_artifact_with_contour_mode_profiled(
    recipe: DerivedRecipe,
    grid: &rustwx_core::LatLonGrid,
    projected: &ProjectedMap,
    domain_bounds: (f64, f64, f64, f64),
    date_yyyymmdd: &str,
    cycle_utc: u8,
    forecast_hour: u16,
    source: SourceId,
    model: ModelId,
    output_width: u32,
    output_height: u32,
    computed: &DerivedComputedFields,
    contour_mode: NativeContourRenderMode,
    native_fill_level_multiplier: usize,
) -> Result<(HrrrDerivedLiveArtifact, DerivedLiveArtifactBuildTiming), Box<dyn std::error::Error>> {
    let total_start = Instant::now();
    let request_base_build_start = Instant::now();
    let (field, mut request) = match recipe {
        DerivedRecipe::Sbcape => weather_request(
            recipe,
            grid,
            "J/kg",
            required_values(&computed.sbcape_jkg, recipe, "sbcape_jkg")?.clone(),
            WeatherProduct::Sbcape,
        )?,
        DerivedRecipe::Sbcin => weather_request(
            recipe,
            grid,
            "J/kg",
            required_values(&computed.sbcin_jkg, recipe, "sbcin_jkg")?.clone(),
            WeatherProduct::Sbcin,
        )?,
        DerivedRecipe::Sblcl => weather_request(
            recipe,
            grid,
            "m",
            required_values(&computed.sblcl_m, recipe, "sblcl_m")?.clone(),
            WeatherProduct::Lcl,
        )?,
        DerivedRecipe::Mlcape => weather_request(
            recipe,
            grid,
            "J/kg",
            required_values(&computed.mlcape_jkg, recipe, "mlcape_jkg")?.clone(),
            WeatherProduct::Mlcape,
        )?,
        DerivedRecipe::Mlcin => weather_request(
            recipe,
            grid,
            "J/kg",
            required_values(&computed.mlcin_jkg, recipe, "mlcin_jkg")?.clone(),
            WeatherProduct::Mlcin,
        )?,
        DerivedRecipe::Mucape => weather_request(
            recipe,
            grid,
            "J/kg",
            required_values(&computed.mucape_jkg, recipe, "mucape_jkg")?.clone(),
            WeatherProduct::Mucape,
        )?,
        DerivedRecipe::Mucin => weather_request(
            recipe,
            grid,
            "J/kg",
            required_values(&computed.mucin_jkg, recipe, "mucin_jkg")?.clone(),
            WeatherProduct::Mucin,
        )?,
        DerivedRecipe::Dcape => custom_scale_request(
            recipe,
            grid,
            "J/kg",
            required_values(&computed.dcape_jkg, recipe, "dcape_jkg")?.clone(),
            range_step(0.0, 2501.0, 100.0),
            dcape_scale_colors(),
            ExtendMode::Max,
            Some(250.0),
        )?,
        DerivedRecipe::ThetaE2m10mWinds => palette_request(
            recipe,
            grid,
            "K",
            required_values(&computed.theta_e_2m_k, recipe, "theta_e_2m_k")?.clone(),
            WeatherPalette::Temperature,
            range_step(280.0, 381.0, 4.0),
            ExtendMode::Both,
            Some(8.0),
        )?,
        DerivedRecipe::Vpd2m => custom_scale_request(
            recipe,
            grid,
            "hPa",
            required_values(&computed.vpd_2m_hpa, recipe, "vpd_2m_hpa")?.clone(),
            range_step(0.0, 11.0, 1.0),
            vpd_scale_colors(),
            ExtendMode::Max,
            Some(2.0),
        )?,
        DerivedRecipe::DewpointDepression2m => custom_scale_request(
            recipe,
            grid,
            "degC",
            required_values(
                &computed.dewpoint_depression_2m_c,
                recipe,
                "dewpoint_depression_2m_c",
            )?
            .clone(),
            range_step(0.0, 41.0, 4.0),
            dewpoint_depression_scale_colors(),
            ExtendMode::Max,
            Some(8.0),
        )?,
        DerivedRecipe::Wetbulb2m => scale_request(
            recipe,
            grid,
            "degC",
            required_values(&computed.wetbulb_2m_c, recipe, "wetbulb_2m_c")?.clone(),
            surface_temperature_scale_c(0.5),
            Some(5.0),
        )?,
        DerivedRecipe::FireWeatherComposite => custom_scale_request(
            recipe,
            grid,
            "index",
            required_values(
                &computed.fire_weather_composite,
                recipe,
                "fire_weather_composite",
            )?
            .clone(),
            range_step(0.0, 101.0, 10.0),
            fire_weather_composite_scale_colors(),
            ExtendMode::Neither,
            Some(20.0),
        )?,
        DerivedRecipe::ApparentTemperature2m => derived_style_request(
            recipe,
            grid,
            "degC",
            required_values(
                &computed.apparent_temperature_2m_c,
                recipe,
                "apparent_temperature_2m_c",
            )?
            .clone(),
            DerivedProductStyle::ApparentTemperature,
        )?,
        DerivedRecipe::HeatIndex2m => palette_request(
            recipe,
            grid,
            "degC",
            required_values(&computed.heat_index_2m_c, recipe, "heat_index_2m_c")?.clone(),
            WeatherPalette::Temperature,
            range_step(-30.0, 51.0, 5.0),
            ExtendMode::Both,
            Some(5.0),
        )?,
        DerivedRecipe::WindChill2m => palette_request(
            recipe,
            grid,
            "degC",
            required_values(&computed.wind_chill_2m_c, recipe, "wind_chill_2m_c")?.clone(),
            WeatherPalette::Temperature,
            range_step(-40.0, 31.0, 5.0),
            ExtendMode::Both,
            Some(5.0),
        )?,
        DerivedRecipe::LiftedIndex => palette_request(
            recipe,
            grid,
            "degC",
            required_values(&computed.lifted_index_c, recipe, "lifted_index_c")?.clone(),
            WeatherPalette::Temperature,
            range_step(-12.0, 13.0, 1.0),
            ExtendMode::Both,
            Some(1.0),
        )?,
        DerivedRecipe::LapseRate700500 => weather_lapse_request(
            recipe,
            grid,
            required_values(
                &computed.lapse_rate_700_500_cpkm,
                recipe,
                "lapse_rate_700_500_cpkm",
            )?
            .clone(),
        )?,
        DerivedRecipe::LapseRate03km => weather_lapse_request(
            recipe,
            grid,
            required_values(
                &computed.lapse_rate_0_3km_cpkm,
                recipe,
                "lapse_rate_0_3km_cpkm",
            )?
            .clone(),
        )?,
        DerivedRecipe::BulkShear01km => palette_request(
            recipe,
            grid,
            "kt",
            required_values(&computed.shear_01km_kt, recipe, "shear_01km_kt")?.clone(),
            WeatherPalette::Winds,
            range_step(0.0, 85.0, 5.0),
            ExtendMode::Max,
            Some(5.0),
        )?,
        DerivedRecipe::BulkShear06km => palette_request(
            recipe,
            grid,
            "kt",
            required_values(&computed.shear_06km_kt, recipe, "shear_06km_kt")?.clone(),
            WeatherPalette::Winds,
            range_step(0.0, 85.0, 5.0),
            ExtendMode::Max,
            Some(5.0),
        )?,
        DerivedRecipe::Srh01km => weather_request(
            recipe,
            grid,
            "m^2/s^2",
            required_values(&computed.srh_01km_m2s2, recipe, "srh_01km_m2s2")?.clone(),
            WeatherProduct::Srh01km,
        )?,
        DerivedRecipe::Srh03km => weather_request(
            recipe,
            grid,
            "m^2/s^2",
            required_values(&computed.srh_03km_m2s2, recipe, "srh_03km_m2s2")?.clone(),
            WeatherProduct::Srh03km,
        )?,
        DerivedRecipe::Ehi01km => weather_request(
            recipe,
            grid,
            "dimensionless",
            required_values(&computed.ehi_01km, recipe, "ehi_01km")?.clone(),
            WeatherProduct::Ehi,
        )?,
        DerivedRecipe::Ehi03km => weather_request(
            recipe,
            grid,
            "dimensionless",
            required_values(&computed.ehi_03km, recipe, "ehi_03km")?.clone(),
            WeatherProduct::Ehi,
        )?,
        DerivedRecipe::StpFixed => weather_request(
            recipe,
            grid,
            "dimensionless",
            required_values(&computed.stp_fixed, recipe, "stp_fixed")?.clone(),
            WeatherProduct::StpFixed,
        )?,
        DerivedRecipe::ScpMu03km06kmProxy => weather_request(
            recipe,
            grid,
            "dimensionless",
            required_values(
                &computed.scp_mu_03km_06km_proxy,
                recipe,
                "scp_mu_03km_06km_proxy",
            )?
            .clone(),
            WeatherProduct::Scp,
        )?,
        DerivedRecipe::TemperatureAdvection700mb => palette_request(
            recipe,
            grid,
            "degC/hr",
            required_values(
                &computed.temperature_advection_700mb_cph,
                recipe,
                "temperature_advection_700mb_cph",
            )?
            .clone(),
            WeatherPalette::Temperature,
            range_step(-12.0, 13.0, 1.0),
            ExtendMode::Both,
            Some(1.0),
        )?,
        DerivedRecipe::TemperatureAdvection850mb => palette_request(
            recipe,
            grid,
            "degC/hr",
            required_values(
                &computed.temperature_advection_850mb_cph,
                recipe,
                "temperature_advection_850mb_cph",
            )?
            .clone(),
            WeatherPalette::Temperature,
            range_step(-12.0, 13.0, 1.0),
            ExtendMode::Both,
            Some(1.0),
        )?,
        DerivedRecipe::Sbecape
        | DerivedRecipe::Mlecape
        | DerivedRecipe::Muecape
        | DerivedRecipe::SbEcapeDerivedCapeRatio
        | DerivedRecipe::MlEcapeDerivedCapeRatio
        | DerivedRecipe::MuEcapeDerivedCapeRatio
        | DerivedRecipe::SbEcapeNativeCapeRatio
        | DerivedRecipe::MlEcapeNativeCapeRatio
        | DerivedRecipe::MuEcapeNativeCapeRatio
        | DerivedRecipe::Sbncape
        | DerivedRecipe::Sbecin
        | DerivedRecipe::Mlecin
        | DerivedRecipe::EcapeScp
        | DerivedRecipe::EcapeEhi01km
        | DerivedRecipe::EcapeEhi03km
        | DerivedRecipe::EcapeStp => {
            return Err(format!(
                "heavy derived recipe '{}' must render through the cropped ECAPE path",
                recipe.slug()
            )
            .into());
        }
    };

    request.width = output_width;
    request.height = output_height;
    request.chrome_scale = static_chrome_scale();
    request.supersample_factor = static_supersample_factor();
    request.supersample_sharpen = static_supersample_sharpen();
    crate::plot_design::StaticPlotDesign::new(domain_bounds, recipe.visual_mode())
        .apply_to_request(&mut request);
    request.title = Some(derived_title_for_model(model, recipe.title()));
    request.subtitle_left = Some(model_time_subtitle(
        model,
        date_yyyymmdd,
        cycle_utc,
        forecast_hour,
    ));
    request.subtitle_right = Some(source_subtitle(source));
    request.projected_domain = Some(ProjectedDomain {
        x: projected.projected_x.clone(),
        y: projected.projected_y.clone(),
        extent: projected.extent.clone(),
    });
    request.projected_lines = projected.lines.clone();
    request.projected_polygons = projected.polygons.clone();
    apply_source_raster_policy(source, &mut request);
    let request_base_build_ms = request_base_build_start.elapsed().as_millis();

    let native_contour_timing = maybe_apply_native_contour_fill_for_mode_profiled(
        recipe,
        &mut request,
        contour_mode,
        native_fill_level_multiplier,
    )?;

    let mut wind_overlay_build_ms = 0;
    if matches!(recipe, DerivedRecipe::ThetaE2m10mWinds) {
        let wind_overlay_start = Instant::now();
        let u_kt = computed_surface_u10(computed, recipe)?;
        let v_kt = computed_surface_v10(computed, recipe)?;
        request.wind_barbs.push(surface_wind_barb_layer(
            grid,
            &projected.extent,
            &projected.projected_x,
            &projected.projected_y,
            &u_kt,
            &v_kt,
        ));
        wind_overlay_build_ms = wind_overlay_start.elapsed().as_millis();
    }

    Ok((
        HrrrDerivedLiveArtifact {
            recipe_slug: recipe.slug().to_string(),
            title: recipe.title().to_string(),
            field,
            request,
        },
        DerivedLiveArtifactBuildTiming {
            compute_fields_ms: 0,
            request_base_build_ms,
            native_contour_fill_ms: native_contour_timing.total_ms,
            native_contour_projected_points_ms: native_contour_timing.projected_points_ms,
            native_contour_scalar_field_ms: native_contour_timing.scalar_field_ms,
            native_contour_fill_topology_ms: native_contour_timing.fill_topology_ms,
            native_contour_fill_geometry_ms: native_contour_timing.fill_geometry_ms,
            native_contour_line_topology_ms: native_contour_timing.line_topology_ms,
            native_contour_line_geometry_ms: native_contour_timing.line_geometry_ms,
            wind_overlay_build_ms,
            total_ms: total_start.elapsed().as_millis(),
        },
    ))
}

struct NativeContourProductConfig {
    scale: rustwx_render::ColorScale,
    line_levels: &'static [f64],
    line_style: ProjectedContourLineStyle,
    tick_step: Option<f64>,
}

const STP_NATIVE_LINE_LEVELS: &[f64] = &[1.0, 3.0, 5.0];
const CAPE_NATIVE_LINE_LEVELS: &[f64] = &[500.0, 1000.0, 2000.0, 3000.0, 4000.0];
const DCAPE_NATIVE_LINE_LEVELS: &[f64] = &[500.0, 1000.0, 1500.0, 2000.0];
const SRH_NATIVE_LINE_LEVELS: &[f64] = &[150.0, 250.0, 350.0, 450.0];
const EHI_NATIVE_LINE_LEVELS: &[f64] = &[1.0, 2.0, 3.0, 5.0];

fn weather_preset_masked_scale(
    preset: rustwx_render::weather::WeatherPreset,
    mask_below: Option<f64>,
) -> ColorScale {
    let mut scale = preset.scale();
    scale.mask_below = mask_below;
    ColorScale::Discrete(scale)
}

fn native_contour_product_config(recipe: DerivedRecipe) -> Option<NativeContourProductConfig> {
    match recipe {
        DerivedRecipe::StpFixed => Some(NativeContourProductConfig {
            scale: weather_preset_masked_scale(
                rustwx_render::weather::WeatherPreset::Stp,
                Some(1.0),
            ),
            line_levels: STP_NATIVE_LINE_LEVELS,
            line_style: ProjectedContourLineStyle {
                color: Color::rgba(55, 16, 16, 210),
                width: 2,
            },
            tick_step: Some(1.0),
        }),
        DerivedRecipe::Sbcape | DerivedRecipe::Mlcape => Some(NativeContourProductConfig {
            scale: weather_preset_masked_scale(
                rustwx_render::weather::WeatherPreset::Cape,
                Some(250.0),
            ),
            line_levels: CAPE_NATIVE_LINE_LEVELS,
            line_style: ProjectedContourLineStyle {
                color: Color::rgba(84, 44, 18, 215),
                width: 2,
            },
            tick_step: Some(500.0),
        }),
        DerivedRecipe::Dcape => Some(NativeContourProductConfig {
            scale: rustwx_render::ColorScale::Discrete(rustwx_render::DiscreteColorScale {
                levels: range_step(0.0, 2501.0, 100.0),
                colors: dcape_scale_colors(),
                extend: ExtendMode::Max,
                mask_below: Some(500.0),
            }),
            line_levels: DCAPE_NATIVE_LINE_LEVELS,
            line_style: ProjectedContourLineStyle {
                color: Color::rgba(70, 40, 20, 215),
                width: 2,
            },
            tick_step: Some(250.0),
        }),
        DerivedRecipe::Srh01km | DerivedRecipe::Srh03km => Some(NativeContourProductConfig {
            scale: weather_preset_masked_scale(
                rustwx_render::weather::WeatherPreset::Srh,
                Some(100.0),
            ),
            line_levels: SRH_NATIVE_LINE_LEVELS,
            line_style: ProjectedContourLineStyle {
                color: Color::rgba(15, 35, 56, 220),
                width: 2,
            },
            tick_step: Some(50.0),
        }),
        DerivedRecipe::Ehi01km | DerivedRecipe::Ehi03km => Some(NativeContourProductConfig {
            scale: weather_preset_masked_scale(
                rustwx_render::weather::WeatherPreset::Ehi,
                Some(0.5),
            ),
            line_levels: EHI_NATIVE_LINE_LEVELS,
            line_style: ProjectedContourLineStyle {
                color: Color::rgba(44, 18, 66, 220),
                width: 2,
            },
            tick_step: Some(0.5),
        }),
        _ => None,
    }
}

pub fn native_contour_line_levels_for_recipe_slug(
    recipe_slug: &str,
) -> Result<Option<Vec<f64>>, String> {
    let recipe = DerivedRecipe::parse(recipe_slug)?;
    Ok(native_contour_product_config(recipe).map(|config| config.line_levels.to_vec()))
}

fn maybe_apply_native_contour_fill_for_mode(
    recipe: DerivedRecipe,
    request: &mut MapRenderRequest,
    contour_mode: NativeContourRenderMode,
    native_fill_level_multiplier: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    maybe_apply_native_contour_fill_for_mode_profiled(
        recipe,
        request,
        contour_mode,
        native_fill_level_multiplier,
    )
    .map(|_| ())
}

fn maybe_apply_native_contour_fill_for_mode_profiled(
    recipe: DerivedRecipe,
    request: &mut MapRenderRequest,
    contour_mode: NativeContourRenderMode,
    native_fill_level_multiplier: usize,
) -> Result<NativeContourBuildTiming, Box<dyn std::error::Error>> {
    let total_start = Instant::now();
    if matches!(
        contour_mode,
        NativeContourRenderMode::Automatic | NativeContourRenderMode::LegacyRaster
    ) {
        return Ok(NativeContourBuildTiming::default());
    }
    let Some(projected_domain) = request.projected_domain.as_ref() else {
        return Ok(NativeContourBuildTiming::default());
    };
    let config = match contour_mode {
        NativeContourRenderMode::Signature => {
            let Some(config) = native_contour_product_config(recipe) else {
                return Ok(NativeContourBuildTiming::default());
            };
            config
        }
        NativeContourRenderMode::ExperimentalAllProjected => native_contour_product_config(recipe)
            .unwrap_or_else(|| NativeContourProductConfig {
                scale: request.scale.clone(),
                line_levels: &[],
                line_style: ProjectedContourLineStyle::default(),
                tick_step: request.cbar_tick_step,
            }),
        NativeContourRenderMode::Automatic | NativeContourRenderMode::LegacyRaster => {
            unreachable!()
        }
    };
    request.scale = native_projected_contour_scale(
        config.scale,
        config.tick_step,
        native_fill_level_multiplier,
    );
    if config.tick_step.is_some() {
        request.cbar_tick_step = config.tick_step;
    }
    let (geometry, geometry_timing) = build_projected_contour_geometry_profile(
        &request.field,
        projected_domain,
        &request.scale,
        config.line_levels,
        config.line_style,
    )?;
    request.projected_data_polygons.extend(geometry.fills);
    request.projected_lines.extend(geometry.lines);
    request.field.values.fill(f32::NAN);
    Ok(NativeContourBuildTiming {
        total_ms: total_start.elapsed().as_millis(),
        projected_points_ms: geometry_timing.projected_points_ms,
        scalar_field_ms: geometry_timing.scalar_field_ms,
        fill_topology_ms: geometry_timing.fill_topology_ms,
        fill_geometry_ms: geometry_timing.fill_geometry_ms,
        line_topology_ms: geometry_timing.line_topology_ms,
        line_geometry_ms: geometry_timing.line_geometry_ms,
    })
}

fn native_projected_contour_scale(
    scale: rustwx_render::ColorScale,
    tick_step: Option<f64>,
    native_fill_level_multiplier: usize,
) -> rustwx_render::ColorScale {
    if let Some(tick_step) = tick_step.filter(|value| value.is_finite() && *value > 0.0) {
        let mut discrete = scale.resolved_discrete();
        let multiplier = native_fill_level_multiplier.max(1) as f64;
        discrete.levels = coarsen_native_contour_levels(
            &discrete.levels,
            tick_step / multiplier,
            discrete.mask_below,
        );
        return rustwx_render::ColorScale::Discrete(discrete);
    }

    if native_fill_level_multiplier <= 1 {
        return scale;
    }
    let discrete = scale.resolved_discrete();
    rustwx_render::ColorScale::Discrete(densify_discrete_scale(
        &discrete,
        LevelDensity {
            multiplier: native_fill_level_multiplier,
            min_source_level_count: 2,
        },
    ))
}

fn coarsen_native_contour_levels(
    levels: &[f64],
    min_step: f64,
    mask_below: Option<f64>,
) -> Vec<f64> {
    if levels.len() <= 2 || !min_step.is_finite() || min_step <= 0.0 {
        return levels.to_vec();
    }

    let mut coarsened = Vec::new();
    let push_level = |levels_out: &mut Vec<f64>, level: f64| {
        if level.is_finite()
            && levels_out
                .last()
                .is_none_or(|last| (level - *last).abs() > 1.0e-9)
        {
            levels_out.push(level);
        }
    };

    let mut last_kept = levels[0];
    push_level(&mut coarsened, last_kept);
    for &level in levels.iter().skip(1) {
        if level - last_kept >= min_step - 1.0e-9 {
            push_level(&mut coarsened, level);
            last_kept = level;
        }
    }
    if let Some(&last) = levels.last() {
        push_level(&mut coarsened, last);
    }
    if let Some(mask) = mask_below.filter(|value| value.is_finite()) {
        if let (Some(&first), Some(&last)) = (levels.first(), levels.last()) {
            if mask > first && mask < last {
                push_level(&mut coarsened, mask);
            }
        }
    }

    coarsened.sort_by(|left, right| left.total_cmp(right));
    coarsened.dedup_by(|left, right| (*left - *right).abs() <= 1.0e-9);
    coarsened
}

fn heavy_ecape_subtitle_right(recipe: DerivedRecipe, source: SourceId) -> String {
    let source_label = source_subtitle(source);
    match recipe {
        DerivedRecipe::SbEcapeDerivedCapeRatio
        | DerivedRecipe::MlEcapeDerivedCapeRatio
        | DerivedRecipe::MuEcapeDerivedCapeRatio => {
            format!("{source_label} | EXP | derived")
        }
        DerivedRecipe::SbEcapeNativeCapeRatio
        | DerivedRecipe::MlEcapeNativeCapeRatio
        | DerivedRecipe::MuEcapeNativeCapeRatio => {
            format!("{source_label} | EXP | native")
        }
        DerivedRecipe::EcapeScp
        | DerivedRecipe::EcapeEhi01km
        | DerivedRecipe::EcapeEhi03km
        | DerivedRecipe::EcapeStp => {
            format!("{source_label} | experimental")
        }
        _ => source_label,
    }
}

fn render_derived_heavy_recipe(
    request: &DerivedBatchRequest,
    recipe: DerivedRecipe,
    field: &WeatherPanelField,
    grid: &rustwx_core::LatLonGrid,
    projection: Option<&rustwx_core::GridProjection>,
    projected: &ProjectedMap,
    date_yyyymmdd: &str,
    cycle_utc: u8,
    forecast_hour: u16,
    source: SourceId,
    model: ModelId,
    input_fetch_keys: Vec<String>,
    render_overrides: DerivedRenderOverrides<'_>,
) -> Result<DerivedRenderedRecipe, Box<dyn std::error::Error>> {
    let filename_suffix = derived_output_suffix(render_overrides.output_suffix);
    let output_path = request.out_dir.join(format!(
        "rustwx_{}_{}_{}z_f{:03}_{}_{}{}.png",
        model.as_str().replace('-', "_"),
        date_yyyymmdd,
        cycle_utc,
        forecast_hour,
        request.domain.slug,
        recipe.slug(),
        filename_suffix
    ));
    let subtitle_left = model_time_subtitle(model, date_yyyymmdd, cycle_utc, forecast_hour);
    let render_start = Instant::now();
    let mut render_request = build_weather_map_request(
        grid,
        projected,
        field,
        request.output_width,
        request.output_height,
        Some(subtitle_left),
        Some(heavy_ecape_subtitle_right(recipe, source)),
    )?;
    render_request.chrome_scale = static_chrome_scale();
    render_request.title = Some(derived_title_for_request(request, recipe.title()));
    if let Some(subtitle_left) = render_overrides.subtitle_left {
        render_request.subtitle_left = Some(subtitle_left.to_string());
    }
    if let Some(subtitle_right) = render_overrides.subtitle_right {
        render_request.subtitle_right = Some(subtitle_right.to_string());
    }
    maybe_apply_native_contour_fill_for_mode(
        recipe,
        &mut render_request,
        request.contour_mode,
        request.native_fill_level_multiplier,
    )?;
    if let Some(overlay) = request.place_label_overlay.as_ref() {
        crate::apply_place_label_overlay_with_density_styling(
            &mut render_request,
            overlay,
            &request.domain,
            &grid.lat_deg,
            &grid.lon_deg,
            projection,
        )?;
    }
    let save_timing =
        save_png_profile_with_options(&render_request, &output_path, &request.png_write_options())?;
    let render_ms = render_start.elapsed().as_millis();
    let content_identity = artifact_identity_from_path(&output_path)?;
    Ok(DerivedRenderedRecipe {
        recipe_slug: recipe.slug().to_string(),
        title: recipe.title().to_string(),
        source_route: ProductSourceRoute::CanonicalDerived,
        output_path,
        content_identity,
        input_fetch_keys,
        timing: DerivedRecipeTiming {
            render_to_image_ms: save_timing.png_timing.render_to_image_ms,
            data_layer_draw_ms: derived_data_layer_draw_ms(&save_timing.png_timing.image_timing),
            overlay_draw_ms: derived_overlay_draw_ms(&save_timing.png_timing.image_timing),
            render_state_prep_ms: save_timing.state_timing.state_prep_ms,
            png_encode_ms: save_timing.png_timing.png_encode_ms,
            file_write_ms: save_timing.file_write_ms,
            render_ms,
            total_ms: render_ms,
            state_timing: save_timing.state_timing,
            image_timing: save_timing.png_timing.image_timing,
        },
    })
}

fn render_derived_heavy_recipes(
    request: &DerivedBatchRequest,
    heavy_recipes: &[DerivedRecipe],
    full_surface: &GenericSurfaceFields,
    full_pressure: &GenericPressureFields,
    full_grid: &rustwx_core::LatLonGrid,
    full_projected: &ProjectedMap,
    date_yyyymmdd: &str,
    cycle_utc: u8,
    forecast_hour: u16,
    source: SourceId,
    model: ModelId,
    input_fetch_keys: Vec<String>,
    render_overrides: DerivedRenderOverrides<'_>,
) -> Result<(Vec<DerivedRenderedRecipe>, HeavyComputeTiming), Box<dyn std::error::Error>> {
    let total_start = Instant::now();
    let heavy_domain = crop_and_guard_heavy_domain(
        full_surface,
        full_pressure,
        full_projected,
        &request.domain,
        2,
        request.allow_large_heavy_domain,
    )?;
    let (surface, pressure, grid) = heavy_domain.bind(full_surface, full_pressure, full_grid);
    let projected = if heavy_domain.cropped.is_some() {
        build_derived_projected_map_with_projection(
            model,
            &grid.lat_deg,
            &grid.lon_deg,
            surface.projection.as_ref(),
            request.domain.bounds,
            map_frame_aspect_ratio(request.output_width, request.output_height, true, true),
        )?
    } else {
        full_projected.clone()
    };

    let (prepared, prep_timing) = prepare_heavy_volume_timed(surface, pressure, false)?;
    let ecape_start = Instant::now();
    let (ecape_fields, _failure_count) =
        compute_ecape_map_fields_with_prepared_volume(surface, pressure, &prepared)?;
    let ecape_triplet_ms = ecape_start.elapsed().as_millis();

    let mut rendered = Vec::with_capacity(heavy_recipes.len());
    let mut render_ms = 0u128;
    for recipe in heavy_recipes {
        let field = ecape_fields
            .iter()
            .find(|field| field.artifact_slug() == recipe.slug())
            .ok_or_else(|| {
                format!(
                    "heavy derived ECAPE renderer missing field for recipe '{}'",
                    recipe.slug()
                )
            })?;
        let artifact = render_derived_heavy_recipe(
            request,
            *recipe,
            field,
            &grid,
            surface.projection(),
            &projected,
            date_yyyymmdd,
            cycle_utc,
            forecast_hour,
            source,
            model,
            input_fetch_keys.clone(),
            render_overrides,
        )?;
        render_ms += artifact.timing.render_ms;
        rendered.push(artifact);
    }

    Ok((
        rendered,
        HeavyComputeTiming {
            full_cells: heavy_domain.stats.full_cells,
            cropped_cells: heavy_domain.stats.cropped_cells,
            pressure_levels: heavy_domain.stats.pressure_levels,
            crop_kind: heavy_domain.stats.crop_kind,
            crop_ms: heavy_domain.crop_ms,
            prepare_height_agl_ms: prep_timing.prepare_height_agl_ms,
            broadcast_pressure_ms: prep_timing.broadcast_pressure_ms,
            pressure_3d_bytes: prep_timing.pressure_3d_bytes,
            ecape_triplet_ms,
            severe_fields_ms: 0,
            render_ms,
            total_ms: total_start.elapsed().as_millis(),
        },
    ))
}

fn required_values<'a>(
    values: &'a Option<Vec<f64>>,
    recipe: DerivedRecipe,
    field_name: &str,
) -> Result<&'a Vec<f64>, Box<dyn std::error::Error>> {
    values.as_ref().ok_or_else(|| {
        format!(
            "derived field '{field_name}' was not computed for requested recipe '{}'",
            recipe.slug()
        )
        .into()
    })
}

fn crop_optional_values(
    values: &Option<Vec<f64>>,
    source_nx: usize,
    crop: crate::gridded::GridCrop,
) -> Option<Vec<f64>> {
    values
        .as_ref()
        .map(|values| crop_values_f64(values, source_nx, crop))
}

fn crop_computed_fields(
    computed: &DerivedComputedFields,
    source_nx: usize,
    crop: crate::gridded::GridCrop,
) -> DerivedComputedFields {
    DerivedComputedFields {
        sbcape_jkg: crop_optional_values(&computed.sbcape_jkg, source_nx, crop),
        sbcin_jkg: crop_optional_values(&computed.sbcin_jkg, source_nx, crop),
        sblcl_m: crop_optional_values(&computed.sblcl_m, source_nx, crop),
        mlcape_jkg: crop_optional_values(&computed.mlcape_jkg, source_nx, crop),
        mlcin_jkg: crop_optional_values(&computed.mlcin_jkg, source_nx, crop),
        mucape_jkg: crop_optional_values(&computed.mucape_jkg, source_nx, crop),
        mucin_jkg: crop_optional_values(&computed.mucin_jkg, source_nx, crop),
        dcape_jkg: crop_optional_values(&computed.dcape_jkg, source_nx, crop),
        theta_e_2m_k: crop_optional_values(&computed.theta_e_2m_k, source_nx, crop),
        vpd_2m_hpa: crop_optional_values(&computed.vpd_2m_hpa, source_nx, crop),
        dewpoint_depression_2m_c: crop_optional_values(
            &computed.dewpoint_depression_2m_c,
            source_nx,
            crop,
        ),
        wetbulb_2m_c: crop_optional_values(&computed.wetbulb_2m_c, source_nx, crop),
        fire_weather_composite: crop_optional_values(
            &computed.fire_weather_composite,
            source_nx,
            crop,
        ),
        apparent_temperature_2m_c: crop_optional_values(
            &computed.apparent_temperature_2m_c,
            source_nx,
            crop,
        ),
        heat_index_2m_c: crop_optional_values(&computed.heat_index_2m_c, source_nx, crop),
        wind_chill_2m_c: crop_optional_values(&computed.wind_chill_2m_c, source_nx, crop),
        surface_u10_ms: crop_optional_values(&computed.surface_u10_ms, source_nx, crop),
        surface_v10_ms: crop_optional_values(&computed.surface_v10_ms, source_nx, crop),
        lifted_index_c: crop_optional_values(&computed.lifted_index_c, source_nx, crop),
        lapse_rate_700_500_cpkm: crop_optional_values(
            &computed.lapse_rate_700_500_cpkm,
            source_nx,
            crop,
        ),
        lapse_rate_0_3km_cpkm: crop_optional_values(
            &computed.lapse_rate_0_3km_cpkm,
            source_nx,
            crop,
        ),
        shear_01km_kt: crop_optional_values(&computed.shear_01km_kt, source_nx, crop),
        shear_06km_kt: crop_optional_values(&computed.shear_06km_kt, source_nx, crop),
        srh_01km_m2s2: crop_optional_values(&computed.srh_01km_m2s2, source_nx, crop),
        srh_03km_m2s2: crop_optional_values(&computed.srh_03km_m2s2, source_nx, crop),
        ehi_01km: crop_optional_values(&computed.ehi_01km, source_nx, crop),
        ehi_03km: crop_optional_values(&computed.ehi_03km, source_nx, crop),
        stp_fixed: crop_optional_values(&computed.stp_fixed, source_nx, crop),
        scp_mu_03km_06km_proxy: crop_optional_values(
            &computed.scp_mu_03km_06km_proxy,
            source_nx,
            crop,
        ),
        temperature_advection_700mb_cph: crop_optional_values(
            &computed.temperature_advection_700mb_cph,
            source_nx,
            crop,
        ),
        temperature_advection_850mb_cph: crop_optional_values(
            &computed.temperature_advection_850mb_cph,
            source_nx,
            crop,
        ),
    }
}

fn latlon_grids_match(a: &rustwx_core::LatLonGrid, b: &rustwx_core::LatLonGrid) -> bool {
    a.shape == b.shape && a.lat_deg == b.lat_deg && a.lon_deg == b.lon_deg
}

fn computed_surface_u10(
    computed: &DerivedComputedFields,
    recipe: DerivedRecipe,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    Ok(
        required_values(&computed.surface_u10_ms, recipe, "surface_u10_ms")?
            .iter()
            .map(|value| (*value * KNOTS_PER_MS) as f32)
            .collect(),
    )
}

fn computed_surface_v10(
    computed: &DerivedComputedFields,
    recipe: DerivedRecipe,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    Ok(
        required_values(&computed.surface_v10_ms, recipe, "surface_v10_ms")?
            .iter()
            .map(|value| (*value * KNOTS_PER_MS) as f32)
            .collect(),
    )
}

fn surface_wind_barb_layer(
    grid: &rustwx_core::LatLonGrid,
    extent: &ProjectedExtent,
    projected_x: &[f64],
    projected_y: &[f64],
    u_kt: &[f32],
    v_kt: &[f32],
) -> WindBarbLayer {
    let (visible_nx, visible_ny) = visible_projected_grid_span(
        grid.shape.nx,
        grid.shape.ny,
        projected_x,
        projected_y,
        extent,
    );
    let stride_x = ((visible_nx as f64 / 30.0).round() as usize).clamp(3, 128);
    let stride_y = ((visible_ny as f64 / 18.0).round() as usize).clamp(3, 96);
    WindBarbLayer {
        u: u_kt.to_vec(),
        v: v_kt.to_vec(),
        stride_x,
        stride_y,
        spacing_px: 56.0,
        color: Color::BLACK,
        halo_color: Color::WHITE,
        halo_width: 2,
        width: 1,
        length_px: 20.0,
    }
}

fn visible_projected_grid_span(
    nx: usize,
    ny: usize,
    projected_x: &[f64],
    projected_y: &[f64],
    extent: &ProjectedExtent,
) -> (usize, usize) {
    let mut min_i = usize::MAX;
    let mut max_i = 0usize;
    let mut min_j = usize::MAX;
    let mut max_j = 0usize;

    for j in 0..ny {
        for i in 0..nx {
            let idx = j * nx + i;
            let x = projected_x[idx];
            let y = projected_y[idx];
            if x >= extent.x_min && x <= extent.x_max && y >= extent.y_min && y <= extent.y_max {
                min_i = min_i.min(i);
                max_i = max_i.max(i);
                min_j = min_j.min(j);
                max_j = max_j.max(j);
            }
        }
    }

    if min_i == usize::MAX || min_j == usize::MAX {
        return (nx.max(1), ny.max(1));
    }

    (max_i - min_i + 1, max_j - min_j + 1)
}

fn weather_request(
    recipe: DerivedRecipe,
    grid: &rustwx_core::LatLonGrid,
    units: &str,
    values: Vec<f64>,
    product: WeatherProduct,
) -> Result<(Field2D, MapRenderRequest), Box<dyn std::error::Error>> {
    let field = core_field(recipe, units, grid, values)?;
    let mut request = MapRenderRequest::for_core_weather_product(field.clone(), product)
        .with_visual_mode(recipe.visual_mode());
    apply_operational_raster_scale(&mut request, product);
    Ok((field, request))
}

fn apply_operational_raster_scale(request: &mut MapRenderRequest, product: WeatherProduct) {
    let mask_below = match product {
        WeatherProduct::Sbcape
        | WeatherProduct::Mlcape
        | WeatherProduct::Mucape
        | WeatherProduct::Sbecape
        | WeatherProduct::Mlecape
        | WeatherProduct::Muecape
        | WeatherProduct::Sbncape
        | WeatherProduct::Mlncape
        | WeatherProduct::Muncape
        | WeatherProduct::EcapeCape => Some(250.0),
        WeatherProduct::Srh01km | WeatherProduct::Srh03km => Some(100.0),
        WeatherProduct::Stp
        | WeatherProduct::StpFixed
        | WeatherProduct::StpEffective
        | WeatherProduct::Tehi
        | WeatherProduct::Tts
        | WeatherProduct::VtpMod
        | WeatherProduct::EcapeStpExperimental => Some(1.0),
        WeatherProduct::Scp | WeatherProduct::EcapeScpExperimental => Some(1.0),
        WeatherProduct::Ehi
        | WeatherProduct::EcapeEhi01kmExperimental
        | WeatherProduct::EcapeEhi03kmExperimental => Some(0.5),
        _ => None,
    };
    if let Some(mask_below) = mask_below {
        request.scale = weather_preset_masked_scale(product.scale_preset(), Some(mask_below));
    }
}

fn apply_source_raster_policy(source: SourceId, request: &mut MapRenderRequest) {
    if matches!(source, SourceId::AifsInference)
        && request.raster_sample_mode == RasterSampleMode::Nearest
    {
        request.raster_sample_mode = RasterSampleMode::Linear;
    }
}

fn weather_lapse_request(
    recipe: DerivedRecipe,
    grid: &rustwx_core::LatLonGrid,
    values: Vec<f64>,
) -> Result<(Field2D, MapRenderRequest), Box<dyn std::error::Error>> {
    let field = core_field(recipe, "degC/km", grid, values)?;
    let mut request = MapRenderRequest::for_palette_fill(
        field.clone().into(),
        WeatherPalette::LapseRate,
        range_step(3.0, 10.1, 0.1),
        ExtendMode::Both,
    )
    .with_visual_mode(recipe.visual_mode());
    request.cbar_tick_step = Some(1.0);
    Ok((field, request))
}

fn palette_request(
    recipe: DerivedRecipe,
    grid: &rustwx_core::LatLonGrid,
    units: &str,
    values: Vec<f64>,
    palette: WeatherPalette,
    levels: Vec<f64>,
    extend: ExtendMode,
    tick_step: Option<f64>,
) -> Result<(Field2D, MapRenderRequest), Box<dyn std::error::Error>> {
    let field = core_field(recipe, units, grid, values)?;
    let mut request =
        MapRenderRequest::for_palette_fill(field.clone().into(), palette, levels, extend)
            .with_visual_mode(recipe.visual_mode());
    request.cbar_tick_step = tick_step;
    Ok((field, request))
}

fn scale_request(
    recipe: DerivedRecipe,
    grid: &rustwx_core::LatLonGrid,
    units: &str,
    values: Vec<f64>,
    scale: ColorScale,
    tick_step: Option<f64>,
) -> Result<(Field2D, MapRenderRequest), Box<dyn std::error::Error>> {
    let field = core_field(recipe, units, grid, values)?;
    let mut request =
        MapRenderRequest::new(field.clone().into(), scale).with_visual_mode(recipe.visual_mode());
    request.cbar_tick_step = tick_step;
    Ok((field, request))
}

fn custom_scale_request(
    recipe: DerivedRecipe,
    grid: &rustwx_core::LatLonGrid,
    units: &str,
    values: Vec<f64>,
    levels: Vec<f64>,
    colors: Vec<Color>,
    extend: ExtendMode,
    tick_step: Option<f64>,
) -> Result<(Field2D, MapRenderRequest), Box<dyn std::error::Error>> {
    let field = core_field(recipe, units, grid, values)?;
    let mut request = MapRenderRequest::new(
        field.clone().into(),
        rustwx_render::ColorScale::Discrete(rustwx_render::DiscreteColorScale {
            levels,
            colors,
            extend,
            mask_below: None,
        }),
    )
    .with_visual_mode(recipe.visual_mode());
    request.cbar_tick_step = tick_step;
    Ok((field, request))
}

fn derived_style_request(
    recipe: DerivedRecipe,
    grid: &rustwx_core::LatLonGrid,
    units: &str,
    values: Vec<f64>,
    style: DerivedProductStyle,
) -> Result<(Field2D, MapRenderRequest), Box<dyn std::error::Error>> {
    let field = core_field(recipe, units, grid, values)?;
    let request = MapRenderRequest::for_derived_product(field.clone().into(), style)
        .with_visual_mode(recipe.visual_mode());
    Ok((field, request))
}

fn surface_temperature_scale_c(level_step_c: f64) -> ColorScale {
    let lo = -50.0;
    let hi = 50.5;
    ColorScale::Discrete(DiscreteColorScale {
        levels: range_step(lo, hi, level_step_c),
        colors: temperature_palette_cropped_f(
            Some((-40.0, 120.0)),
            (((hi - lo) / level_step_c).round() as usize).max(2),
        ),
        extend: ExtendMode::Both,
        mask_below: None,
    })
}

fn core_field(
    recipe: DerivedRecipe,
    units: &str,
    grid: &rustwx_core::LatLonGrid,
    values: Vec<f64>,
) -> Result<Field2D, Box<dyn std::error::Error>> {
    Ok(Field2D::new(
        ProductKey::named(recipe.slug()),
        units,
        grid.clone(),
        values.into_iter().map(|value| value as f32).collect(),
    )?)
}

fn vpd_scale_colors() -> Vec<Color> {
    vec![
        Color::rgba(24, 90, 145, 255),
        Color::rgba(39, 129, 172, 255),
        Color::rgba(67, 164, 184, 255),
        Color::rgba(110, 190, 168, 255),
        Color::rgba(154, 211, 142, 255),
        Color::rgba(196, 226, 126, 255),
        Color::rgba(229, 232, 126, 255),
        Color::rgba(247, 219, 118, 255),
        Color::rgba(248, 195, 102, 255),
        Color::rgba(240, 163, 85, 255),
        Color::rgba(226, 130, 72, 255),
        Color::rgba(207, 100, 65, 255),
        Color::rgba(184, 74, 61, 255),
        Color::rgba(157, 53, 60, 255),
        Color::rgba(128, 37, 63, 255),
    ]
}

fn dewpoint_depression_scale_colors() -> Vec<Color> {
    vec![
        Color::rgba(0, 104, 55, 255),
        Color::rgba(26, 152, 80, 255),
        Color::rgba(102, 189, 99, 255),
        Color::rgba(166, 217, 106, 255),
        Color::rgba(217, 239, 139, 255),
        Color::rgba(254, 224, 139, 255),
        Color::rgba(253, 174, 97, 255),
        Color::rgba(244, 109, 67, 255),
        Color::rgba(215, 48, 39, 255),
        Color::rgba(165, 0, 38, 255),
    ]
}

fn dcape_scale_colors() -> Vec<Color> {
    vec![
        Color::rgba(245, 250, 255, 255),
        Color::rgba(218, 237, 251, 255),
        Color::rgba(185, 219, 241, 255),
        Color::rgba(142, 195, 222, 255),
        Color::rgba(102, 170, 200, 255),
        Color::rgba(83, 157, 176, 255),
        Color::rgba(95, 169, 139, 255),
        Color::rgba(132, 188, 103, 255),
        Color::rgba(184, 205, 82, 255),
        Color::rgba(226, 211, 77, 255),
        Color::rgba(245, 186, 70, 255),
        Color::rgba(238, 145, 61, 255),
        Color::rgba(220, 100, 57, 255),
        Color::rgba(190, 63, 62, 255),
        Color::rgba(150, 42, 72, 255),
        Color::rgba(105, 31, 80, 255),
    ]
}

fn fire_weather_composite_scale_colors() -> Vec<Color> {
    vec![
        Color::rgba(250, 250, 247, 255),
        Color::rgba(224, 236, 214, 255),
        Color::rgba(169, 220, 139, 255),
        Color::rgba(91, 179, 93, 255),
        Color::rgba(238, 232, 94, 255),
        Color::rgba(252, 196, 67, 255),
        Color::rgba(247, 145, 45, 255),
        Color::rgba(231, 76, 41, 255),
        Color::rgba(184, 28, 38, 255),
        Color::rgba(119, 18, 35, 255),
    ]
}

fn png_render_parallelism(job_count: usize) -> usize {
    let override_threads = std::env::var("RUSTWX_RENDER_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0);

    thread::available_parallelism()
        .map(|parallelism| override_threads.unwrap_or((parallelism.get() / 2).max(1)))
        .unwrap_or(1)
        .min(job_count.max(1))
}

fn thread_render_error(err: impl std::fmt::Display) -> io::Error {
    io::Error::other(err.to_string())
}

fn render_derived_output_recipe(
    request: &DerivedBatchRequest,
    recipe: DerivedRecipe,
    grid_ref: &rustwx_core::LatLonGrid,
    projection: Option<&rustwx_core::GridProjection>,
    projected_ref: &ProjectedMap,
    date_yyyymmdd: &str,
    cycle_utc: u8,
    forecast_hour: u16,
    source: SourceId,
    model: ModelId,
    computed: &DerivedComputedFields,
    lane_fetch_keys: Vec<String>,
    render_overrides: DerivedRenderOverrides<'_>,
) -> Result<DerivedRenderedRecipe, io::Error> {
    let model_slug = request.model.as_str().replace('-', "_");
    let filename_suffix = derived_output_suffix(render_overrides.output_suffix);
    let output_path = request.out_dir.join(format!(
        "rustwx_{}_{}_{}z_f{:03}_{}_{}{}.png",
        model_slug,
        request.date_yyyymmdd,
        cycle_utc,
        request.forecast_hour,
        request.domain.slug,
        recipe.slug(),
        filename_suffix
    ));
    let render_start = Instant::now();
    let render_artifact = build_render_artifact(
        recipe,
        grid_ref,
        projected_ref,
        request.domain.bounds,
        date_yyyymmdd,
        cycle_utc,
        forecast_hour,
        source,
        model,
        request.output_width,
        request.output_height,
        computed,
        request.contour_mode,
        request.native_fill_level_multiplier,
    )
    .map_err(thread_render_error)?;
    let HrrrDerivedLiveArtifact {
        recipe_slug,
        title: _,
        field: _,
        request: mut render_request,
    } = render_artifact;
    render_request.domain_frame =
        crate::plot_design::static_domain_frame_for_bounds(request.domain.bounds);
    render_request.inverse_raster_projection =
        inverse_raster_projection_for_grid(projection, request.domain.bounds, grid_ref);
    let title = derived_title_for_request(request, recipe.title());
    render_request.title = Some(title.clone());
    if let Some(subtitle_left) = render_overrides.subtitle_left {
        render_request.subtitle_left = Some(subtitle_left.to_string());
    }
    if let Some(subtitle_right) = render_overrides.subtitle_right {
        render_request.subtitle_right = Some(subtitle_right.to_string());
    }
    if let Some(overlay) = request.place_label_overlay.as_ref() {
        crate::apply_place_label_overlay_with_density_styling(
            &mut render_request,
            overlay,
            &request.domain,
            &grid_ref.lat_deg,
            &grid_ref.lon_deg,
            projection,
        )
        .map_err(thread_render_error)?;
    }
    let save_timing =
        save_png_profile_with_options(&render_request, &output_path, &request.png_write_options())
            .map_err(thread_render_error)?;
    let render_ms = render_start.elapsed().as_millis();
    let content_identity =
        artifact_identity_from_path(&output_path).map_err(thread_render_error)?;
    Ok(DerivedRenderedRecipe {
        recipe_slug,
        title,
        source_route: derived_compute_source_route(recipe, request.source_mode).ok_or_else(
            || {
                io::Error::other(format!(
                    "missing compute source route for '{}'",
                    recipe.slug()
                ))
            },
        )?,
        output_path,
        content_identity,
        input_fetch_keys: lane_fetch_keys,
        timing: DerivedRecipeTiming {
            render_to_image_ms: save_timing.png_timing.render_to_image_ms,
            data_layer_draw_ms: derived_data_layer_draw_ms(&save_timing.png_timing.image_timing),
            overlay_draw_ms: derived_overlay_draw_ms(&save_timing.png_timing.image_timing),
            render_state_prep_ms: save_timing.state_timing.state_prep_ms,
            png_encode_ms: save_timing.png_timing.png_encode_ms,
            file_write_ms: save_timing.file_write_ms,
            render_ms,
            total_ms: render_ms,
            state_timing: save_timing.state_timing,
            image_timing: save_timing.png_timing.image_timing,
        },
    })
}

fn join_render_job<T>(
    handle: thread::ScopedJoinHandle<'_, Result<T, io::Error>>,
) -> Result<T, io::Error> {
    match handle.join() {
        Ok(result) => result,
        Err(panic) => Err(io::Error::other(format!(
            "render worker panicked: {}",
            panic_message(panic)
        ))),
    }
}

fn panic_message(panic: Box<dyn std::any::Any + Send + 'static>) -> String {
    if let Some(message) = panic.downcast_ref::<&'static str>() {
        (*message).to_string()
    } else if let Some(message) = panic.downcast_ref::<String>() {
        message.clone()
    } else {
        "unknown panic".to_string()
    }
}

fn range_step(start: f64, stop: f64, step: f64) -> Vec<f64> {
    let mut values = Vec::new();
    let mut current = start;
    while current < stop - step * 1.0e-9 {
        values.push(current);
        current += step;
    }
    values
}

/// The render styling one derived/heavy STORE variable carries in its
/// production PNG: the request's color scale, tick step, density, and
/// legend controls plus the recipe title.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DerivedLaneStyle {
    pub scale: ColorScale,
    pub cbar_tick_step: Option<f64>,
    pub render_density: rustwx_render::RenderDensity,
    pub legend: rustwx_render::LegendControls,
    pub title: String,
}

/// Resolve the production render styling for one derived/heavy store slug
/// by building a REAL render request over a tiny probe grid through the
/// exact builders the render lanes run — non-heavy recipes through
/// [`build_render_artifact_with_contour_mode`] (raster mode, the default),
/// heavy recipes through [`build_weather_map_request`] — and reading the
/// styling off the request. Identity with the plot output is therefore by
/// construction, not by a hand-maintained scale table.
pub(crate) fn derived_store_variable_style(
    slug: &str,
) -> Result<DerivedLaneStyle, Box<dyn std::error::Error>> {
    /// Any non-global domain yields the operational regional styling
    /// (`apply_static_map_design`'s regional branch), matching the CONUS
    /// domains the production HRRR products render with.
    const PROBE_DOMAIN_BOUNDS: (f64, f64, f64, f64) = (-125.0, -66.0, 24.0, 50.0);

    let recipe = DerivedRecipe::parse(slug).map_err(std::io::Error::other)?;
    let grid = rustwx_core::LatLonGrid {
        shape: rustwx_core::GridShape::new(2, 2)?,
        lat_deg: vec![35.0, 35.0, 36.0, 36.0],
        lon_deg: vec![-100.0, -99.0, -100.0, -99.0],
    };
    let projected = ProjectedMap {
        projected_x: grid.lon_deg.iter().map(|&v| f64::from(v)).collect(),
        projected_y: grid.lat_deg.iter().map(|&v| f64::from(v)).collect(),
        extent: ProjectedExtent {
            x_min: -100.0,
            x_max: -99.0,
            y_min: 35.0,
            y_max: 36.0,
        },
        lines: Vec::new(),
        polygons: Vec::new(),
        inverse_raster_projection: None,
    };
    let values = vec![0.0_f64; grid.shape.len()];

    if recipe.is_heavy() {
        let product = store_render::weather_product_for_heavy_recipe(recipe)?;
        let field = WeatherPanelField::new(product, "probe", values);
        let request = build_weather_map_request(&grid, &projected, &field, 320, 240, None, None)?;
        return Ok(DerivedLaneStyle {
            scale: request.scale,
            cbar_tick_step: request.cbar_tick_step,
            render_density: request.render_density,
            legend: request.legend,
            title: recipe.title().to_string(),
        });
    }

    let mut computed = DerivedComputedFields::default();
    store_render::assign_store_values(&mut computed, recipe, values.clone())?;
    if matches!(recipe, DerivedRecipe::ThetaE2m10mWinds) {
        // The theta-e request build also attaches its 10 m barb overlay.
        computed.surface_u10_ms = Some(values.clone());
        computed.surface_v10_ms = Some(values);
    }
    let artifact = build_render_artifact_with_contour_mode(
        recipe,
        &grid,
        &projected,
        PROBE_DOMAIN_BOUNDS,
        "20260101",
        0,
        1,
        SourceId::Aws,
        ModelId::Hrrr,
        320,
        240,
        &computed,
        NativeContourRenderMode::Automatic,
        1,
    )?;
    Ok(DerivedLaneStyle {
        scale: artifact.request.scale,
        cbar_tick_step: artifact.request.cbar_tick_step,
        render_density: artifact.request.render_density,
        legend: artifact.request.legend,
        title: recipe.title().to_string(),
    })
}

#[cfg(test)]
mod tests;
