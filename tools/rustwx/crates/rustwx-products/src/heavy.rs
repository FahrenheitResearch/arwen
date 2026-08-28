use crate::direct::build_projected_map_with_projection;
use crate::ecape::compute_ecape_map_fields_with_prepared_volume;
use crate::gridded::{
    CroppedHeavyDomain, PressureFields, ProjectedGridIntersection, SharedTiming, SurfaceFields,
    classify_projected_grid_intersection, crop_heavy_domain_for_projected_extent,
    prepare_heavy_volume_timed, resolve_thermo_pair_run,
};
use crate::publication::{
    ArtifactContentIdentity, PublishedFetchIdentity, artifact_identity_from_path,
};
use crate::runtime::{BundleLoaderConfig, load_execution_plan};
use crate::severe::{
    build_planned_input_fetches, build_severe_execution_plan, build_shared_timing_for_pair,
    compute_severe_panel_fields_with_prepared_volume,
};
use crate::shared_context::{
    DomainSpec, ProjectedMap, WeatherPanelField, build_weather_map_request, model_time_subtitle,
    source_subtitle,
};
use rustwx_core::{LatLonGrid, ModelId, SourceId};
use rustwx_render::{ProductVisualMode, map_frame_aspect_ratio_for_mode, save_png_profile};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::time::Instant;

// Default allows regional heavy products like Midwest HRRR while still
// forcing explicit opt-in for CONUS-scale parcel diagnostics.
//
// The audit of 2026-08-25 asked whether the crop had removed this
// guard's premise. It has NOT, and the reason is one branch below:
// `crop_and_guard_heavy_domain` classifies the intersection first, and a
// view that COVERS the model grid takes `ProjectedGridIntersection::Full`,
// where `cropped_cells = full_cells` -- the crop bounds nothing at all in
// exactly the case this ceiling exists for (a CONUS-wide view of a
// 1799x1059 = 1.9M-cell source grid). The trigger condition is live.
//
// What is NOT settled is the THRESHOLD VALUE: 1.5M was never timed
// against 3M. That measurement needs the production HRRR product path on
// an otherwise idle box (an in-tree timing on a 117k-cell wrfout, taken
// beside concurrent GPU work, was contaminated and is deliberately not
// quoted here). Protocol recorded in
// evidence/2026-08-25-stale-guards-engine/named-follow-ups.md: same
// cycle twice over a ~3M-cell extent, RUSTWX_MAX_HEAVY_CELLS at 1.5M vs
// 3M, compared on HeavyComputeTiming.
const DEFAULT_MAX_HEAVY_CELLS: usize = 1_500_000;
const MAX_HEAVY_CELLS_ENV: &str = "RUSTWX_MAX_HEAVY_CELLS";
pub const HEAVY_MAP_WIDTH: u32 = 1200;
pub const HEAVY_MAP_HEIGHT: u32 = 900;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HeavyCropKind {
    Full,
    Crop,
    Empty,
}

impl Default for HeavyCropKind {
    fn default() -> Self {
        Self::Full
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeavyDomainStats {
    pub full_cells: usize,
    pub cropped_cells: usize,
    pub pressure_levels: usize,
    pub crop_kind: HeavyCropKind,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HeavyComputeTiming {
    pub full_cells: usize,
    pub cropped_cells: usize,
    pub pressure_levels: usize,
    pub crop_kind: HeavyCropKind,
    pub crop_ms: u128,
    pub prepare_height_agl_ms: u128,
    pub broadcast_pressure_ms: u128,
    pub pressure_3d_bytes: usize,
    pub ecape_triplet_ms: u128,
    pub severe_fields_ms: u128,
    pub render_ms: u128,
    pub total_ms: u128,
}

#[derive(Debug, Clone)]
pub struct HeavyDomainSelection {
    pub cropped: Option<CroppedHeavyDomain>,
    pub stats: HeavyDomainStats,
    pub crop_ms: u128,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeavyPanelHourRequest {
    pub model: ModelId,
    pub date_yyyymmdd: String,
    pub cycle_override_utc: Option<u8>,
    pub forecast_hour: u16,
    pub source: SourceId,
    pub domain: DomainSpec,
    pub out_dir: PathBuf,
    pub cache_root: PathBuf,
    pub use_cache: bool,
    pub surface_product_override: Option<String>,
    pub pressure_product_override: Option<String>,
    pub allow_large_heavy_domain: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeavyRenderedArtifact {
    pub product: String,
    pub title: String,
    pub output_path: PathBuf,
    pub output_identity: ArtifactContentIdentity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeavyRenderedArtifactGroup {
    pub family: String,
    pub outputs: Vec<HeavyRenderedArtifact>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure_count: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeavyPanelHourReport {
    pub model: ModelId,
    pub date_yyyymmdd: String,
    pub cycle_utc: u8,
    pub forecast_hour: u16,
    pub source: SourceId,
    pub domain: DomainSpec,
    pub input_fetches: Vec<PublishedFetchIdentity>,
    pub shared_timing: SharedTiming,
    pub project_ms: u128,
    pub compute_ms: u128,
    pub heavy_timing: HeavyComputeTiming,
    pub severe: HeavyRenderedArtifactGroup,
    pub ecape: HeavyRenderedArtifactGroup,
    pub total_ms: u128,
}

impl HeavyDomainSelection {
    pub fn bind<'a>(
        &'a self,
        full_surface: &'a SurfaceFields,
        full_pressure: &'a PressureFields,
        full_grid: &'a LatLonGrid,
    ) -> (&'a SurfaceFields, &'a PressureFields, LatLonGrid) {
        match self.cropped.as_ref() {
            Some(cropped) => (&cropped.surface, &cropped.pressure, cropped.grid.clone()),
            None => (full_surface, full_pressure, full_grid.clone()),
        }
    }
}

impl HeavyComputeTiming {
    pub fn compute_ms(&self) -> u128 {
        self.prepare_height_agl_ms
            + self.broadcast_pressure_ms
            + self.ecape_triplet_ms
            + self.severe_fields_ms
    }
}

pub fn heavy_map_target_aspect_ratio() -> f64 {
    map_frame_aspect_ratio_for_mode(
        ProductVisualMode::SevereDiagnostic,
        HEAVY_MAP_WIDTH,
        HEAVY_MAP_HEIGHT,
        true,
        true,
    )
}

pub fn render_heavy_map_group(
    out_dir: &Path,
    model_slug: &str,
    date_yyyymmdd: &str,
    cycle_utc: u8,
    forecast_hour: u16,
    domain_slug: &str,
    family_slug: &str,
    grid: &LatLonGrid,
    projected: &ProjectedMap,
    fields: &[WeatherPanelField],
    subtitle_left: &str,
    subtitle_right: impl Fn(&WeatherPanelField) -> Option<String>,
) -> Result<(Vec<HeavyRenderedArtifact>, u128), Box<dyn std::error::Error>> {
    let mut outputs = Vec::with_capacity(fields.len());
    let mut render_ms = 0;
    for field in fields {
        let output_path = out_dir.join(format!(
            "rustwx_{}_{}_{}z_f{:03}_{}_{}_{}.png",
            model_slug,
            date_yyyymmdd,
            cycle_utc,
            forecast_hour,
            domain_slug,
            family_slug,
            field.artifact_slug()
        ));
        let request = build_weather_map_request(
            grid,
            projected,
            field,
            HEAVY_MAP_WIDTH,
            HEAVY_MAP_HEIGHT,
            Some(subtitle_left.to_string()),
            subtitle_right(field),
        )?;
        render_ms += save_png_profile(&request, &output_path)?.total_ms;
        outputs.push(HeavyRenderedArtifact {
            product: field.artifact_slug().to_string(),
            title: field.display_title().to_string(),
            output_identity: artifact_identity_from_path(&output_path)?,
            output_path,
        });
    }
    Ok((outputs, render_ms))
}

pub fn heavy_domain_cell_limit() -> usize {
    std::env::var(MAX_HEAVY_CELLS_ENV)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(DEFAULT_MAX_HEAVY_CELLS)
}

pub fn heavy_domain_limit_error(stats: &HeavyDomainStats, domain: &DomainSpec) -> String {
    format!(
        "Refusing heavy ECAPE/severe compute on {} cells \u{00D7} {} levels for domain {}. Use a smaller region or pass --allow-large-heavy-domain.",
        stats.cropped_cells, stats.pressure_levels, domain.slug
    )
}

pub fn crop_and_guard_heavy_domain(
    full_surface: &SurfaceFields,
    full_pressure: &PressureFields,
    full_projected: &ProjectedMap,
    domain: &DomainSpec,
    pad_cells: usize,
    allow_large_heavy_domain: bool,
) -> Result<HeavyDomainSelection, Box<dyn std::error::Error>> {
    let crop_start = Instant::now();
    let full_cells = full_surface.nx * full_surface.ny;
    let pressure_levels = full_pressure.pressure_levels_hpa.len();
    let intersection = classify_projected_grid_intersection(
        full_surface.nx,
        full_surface.ny,
        &full_projected.projected_x,
        &full_projected.projected_y,
        &full_projected.extent,
        pad_cells,
    )?;
    let stats = match intersection {
        ProjectedGridIntersection::Empty => HeavyDomainStats {
            full_cells,
            cropped_cells: 0,
            pressure_levels,
            crop_kind: HeavyCropKind::Empty,
        },
        ProjectedGridIntersection::Full => HeavyDomainStats {
            full_cells,
            cropped_cells: full_cells,
            pressure_levels,
            crop_kind: HeavyCropKind::Full,
        },
        ProjectedGridIntersection::Crop(crop) => HeavyDomainStats {
            full_cells,
            cropped_cells: crop.width() * crop.height(),
            pressure_levels,
            crop_kind: HeavyCropKind::Crop,
        },
    };

    if matches!(stats.crop_kind, HeavyCropKind::Empty) {
        return Err("requested projected crop produced an empty heavy-compute domain".into());
    }
    if !allow_large_heavy_domain && stats.cropped_cells > heavy_domain_cell_limit() {
        return Err(heavy_domain_limit_error(&stats, domain).into());
    }

    let cropped = match stats.crop_kind {
        HeavyCropKind::Full => None,
        HeavyCropKind::Crop => crop_heavy_domain_for_projected_extent(
            full_surface,
            full_pressure,
            &full_projected.projected_x,
            &full_projected.projected_y,
            &full_projected.extent,
            pad_cells,
        )?,
        HeavyCropKind::Empty => None,
    };

    Ok(HeavyDomainSelection {
        cropped,
        stats,
        crop_ms: crop_start.elapsed().as_millis(),
    })
}

pub fn run_heavy_panel_hour(
    request: &HeavyPanelHourRequest,
) -> Result<HeavyPanelHourReport, Box<dyn std::error::Error>> {
    fs::create_dir_all(&request.out_dir)?;
    if request.use_cache {
        fs::create_dir_all(&request.cache_root)?;
    }

    let total_start = Instant::now();
    let latest = resolve_thermo_pair_run(
        request.model,
        &request.date_yyyymmdd,
        request.cycle_override_utc,
        request.forecast_hour,
        request.source,
        request.surface_product_override.as_deref(),
        request.pressure_product_override.as_deref(),
    )?;
    let plan = build_severe_execution_plan(
        &latest,
        request.forecast_hour,
        request.surface_product_override.as_deref(),
        request.pressure_product_override.as_deref(),
    );
    let loaded = load_execution_plan(
        plan,
        &BundleLoaderConfig {
            cache_root: request.cache_root.clone(),
            use_cache: request.use_cache,
        },
    )?;

    let (surface_planned, surface_decode, pressure_planned, pressure_decode) = loaded
        .require_surface_pressure_pair()
        .map_err(|err| format!("heavy surface/pressure pair unavailable: {err}"))?;
    let full_surface = &surface_decode.value;
    let full_pressure = &pressure_decode.value;
    let target_ratio = heavy_map_target_aspect_ratio();
    let owned_full_grid = full_surface.core_grid()?;
    let project_start = Instant::now();
    let full_projected = build_projected_map_with_projection(
        &owned_full_grid.lat_deg,
        &owned_full_grid.lon_deg,
        full_surface.projection.as_ref(),
        request.domain.bounds,
        target_ratio,
    )?;
    let heavy_domain = crop_and_guard_heavy_domain(
        full_surface,
        full_pressure,
        &full_projected,
        &request.domain,
        2,
        request.allow_large_heavy_domain,
    )?;
    let (surface, pressure, grid) =
        heavy_domain.bind(full_surface, full_pressure, &owned_full_grid);
    let projected = if heavy_domain.cropped.is_some() {
        build_projected_map_with_projection(
            &grid.lat_deg,
            &grid.lon_deg,
            surface.projection.as_ref(),
            request.domain.bounds,
            target_ratio,
        )?
    } else {
        full_projected
    };
    let project_ms = project_start.elapsed().as_millis();

    let compute_start = Instant::now();
    let (prepared, prep_timing) = prepare_heavy_volume_timed(surface, pressure, false)?;
    let ecape_triplet_start = Instant::now();
    let (ecape_fields, failure_count) =
        compute_ecape_map_fields_with_prepared_volume(surface, pressure, &prepared)?;
    let ecape_triplet_ms = ecape_triplet_start.elapsed().as_millis();
    let severe_fields_start = Instant::now();
    let severe_fields =
        compute_severe_panel_fields_with_prepared_volume(surface, pressure, &prepared)?;
    let severe_fields_ms = severe_fields_start.elapsed().as_millis();
    let compute_ms = compute_start.elapsed().as_millis();

    let model_slug = request.model.as_str().replace('-', "_");
    let subtitle_left = model_time_subtitle(
        request.model,
        &request.date_yyyymmdd,
        loaded.latest.cycle.hour_utc,
        request.forecast_hour,
    );
    let source_label = source_subtitle(loaded.latest.source);
    let (ecape_outputs, ecape_render_ms) = render_heavy_map_group(
        &request.out_dir,
        &model_slug,
        &request.date_yyyymmdd,
        loaded.latest.cycle.hour_utc,
        request.forecast_hour,
        &request.domain.slug,
        "ecape",
        &grid,
        &projected,
        &ecape_fields,
        &subtitle_left,
        |field| match field.artifact_slug() {
            "sb_ecape_derived_cape_ratio"
            | "ml_ecape_derived_cape_ratio"
            | "mu_ecape_derived_cape_ratio" => Some(format!("{source_label} | EXP | derived")),
            "sb_ecape_native_cape_ratio"
            | "ml_ecape_native_cape_ratio"
            | "mu_ecape_native_cape_ratio" => Some(format!("{source_label} | EXP | native")),
            "ecape_scp" | "ecape_ehi_0_1km" | "ecape_ehi_0_3km" | "ecape_stp" => {
                Some(format!("{source_label} | experimental"))
            }
            _ => Some(source_label.clone()),
        },
    )?;
    let (severe_outputs, severe_render_ms) = render_heavy_map_group(
        &request.out_dir,
        &model_slug,
        &request.date_yyyymmdd,
        loaded.latest.cycle.hour_utc,
        request.forecast_hour,
        &request.domain.slug,
        "severe",
        &grid,
        &projected,
        &severe_fields,
        &subtitle_left,
        |field| match field.artifact_slug() {
            "stp_fixed" => Some(format!("{source_label} | fixed-layer composite")),
            "scp_mu_0_3km_0_6km_proxy" => Some(format!(
                "{source_label} | proxy: MUCAPE + 0-3 km SRH + 0-6 km shear"
            )),
            "ehi_0_1km" => Some(format!("{source_label} | proxy: SBCAPE + 0-1 km SRH")),
            _ => Some(source_label.clone()),
        },
    )?;
    let render_ms = ecape_render_ms + severe_render_ms;

    let shared_timing = build_shared_timing_for_pair(&loaded, surface_planned, pressure_planned)?;
    let input_fetches = build_planned_input_fetches(&loaded);
    let total_ms = total_start.elapsed().as_millis();
    let heavy_timing = HeavyComputeTiming {
        full_cells: heavy_domain.stats.full_cells,
        cropped_cells: heavy_domain.stats.cropped_cells,
        pressure_levels: heavy_domain.stats.pressure_levels,
        crop_kind: heavy_domain.stats.crop_kind,
        crop_ms: heavy_domain.crop_ms,
        prepare_height_agl_ms: prep_timing.prepare_height_agl_ms,
        broadcast_pressure_ms: prep_timing.broadcast_pressure_ms,
        pressure_3d_bytes: prep_timing.pressure_3d_bytes,
        ecape_triplet_ms,
        severe_fields_ms,
        render_ms,
        total_ms,
    };

    Ok(HeavyPanelHourReport {
        model: request.model,
        date_yyyymmdd: request.date_yyyymmdd.clone(),
        cycle_utc: loaded.latest.cycle.hour_utc,
        forecast_hour: request.forecast_hour,
        source: loaded.latest.source,
        domain: request.domain.clone(),
        input_fetches,
        shared_timing,
        project_ms,
        compute_ms,
        heavy_timing,
        severe: HeavyRenderedArtifactGroup {
            family: "severe".to_string(),
            outputs: severe_outputs,
            failure_count: None,
        },
        ecape: HeavyRenderedArtifactGroup {
            family: "ecape".to_string(),
            outputs: ecape_outputs,
            failure_count: Some(failure_count),
        },
        total_ms,
    })
}
