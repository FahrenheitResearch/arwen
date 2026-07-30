use crate::direct::build_projected_map_with_projection;
use crate::gridded::{
    PressureFields, SharedTiming, SurfaceFields, prepare_heavy_volume, prepare_heavy_volume_timed,
    resolve_thermo_pair_run,
};
use crate::heavy::{
    HeavyComputeTiming, HeavyRenderedArtifact, crop_and_guard_heavy_domain,
    heavy_map_target_aspect_ratio, render_heavy_map_group,
};
use crate::publication::{PublishedFetchIdentity, artifact_identity_from_path};
use crate::runtime::{BundleLoaderConfig, load_execution_plan};
use crate::severe::{
    build_planned_input_fetches, build_severe_execution_plan, build_shared_timing_for_pair,
};
use crate::shared_context::{
    DomainSpec, ProjectedMap, WeatherPanelField, build_weather_map_request,
};
use rustwx_calc::{
    EcapeTripletOptions, EcapeVolumeInputs, EffectiveStpInputs, ScpEhiInputs, SurfaceInputs,
    WindGridInputs, compute_ecape_triplet_with_failure_mask_from_parts, compute_ehi,
    compute_mlcape_cin, compute_scp_ehi, compute_stp_effective, compute_wind_diagnostics_bundle,
};
use rustwx_core::{ModelId, SourceId};
use rustwx_render::{
    Color, ContourStyle, Field2D as RenderField2D, ProductKey as RenderProductKey, WeatherProduct,
    save_png_profile,
};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

const ECAPE_CAPE_RATIO_MIN_DENOMINATOR_JKG: f64 = 100.0;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EcapeBatchRequest {
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
pub struct EcapeBatchReport {
    pub model: ModelId,
    pub date_yyyymmdd: String,
    pub cycle_utc: u8,
    pub forecast_hour: u16,
    pub source: SourceId,
    pub domain: DomainSpec,
    pub outputs: Vec<HeavyRenderedArtifact>,
    pub input_fetches: Vec<PublishedFetchIdentity>,
    pub shared_timing: SharedTiming,
    pub heavy_timing: HeavyComputeTiming,
    pub project_ms: u128,
    pub compute_ms: u128,
    pub render_ms: u128,
    pub total_ms: u128,
    pub failure_count: usize,
}

pub fn run_ecape_batch(
    request: &EcapeBatchRequest,
) -> Result<EcapeBatchReport, Box<dyn std::error::Error>> {
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
    // ECAPE consumes the same surface+pressure pair as the severe panel,
    // so we reuse the same execution-plan builder; the planner dedupes if
    // both products run in the same pass.
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
        .map_err(|err| format!("ECAPE surface/pressure pair unavailable: {err}"))?;
    let full_surface = &surface_decode.value;
    let full_pressure = &pressure_decode.value;
    let owned_full_grid = full_surface.core_grid()?;
    let project_start = Instant::now();
    let full_projected = build_projected_map_with_projection(
        &owned_full_grid.lat_deg,
        &owned_full_grid.lon_deg,
        full_surface.projection.as_ref(),
        request.domain.bounds,
        heavy_map_target_aspect_ratio(),
    )?;

    // Same rationale as severe_batch: crop before compute so ECAPE's
    // per-cell parcel ascent runs on ~300×300 midwest cells instead of
    // ~1800×1000 CONUS.
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
            heavy_map_target_aspect_ratio(),
        )?
    } else {
        full_projected
    };
    let project_ms = project_start.elapsed().as_millis();

    let compute_start = Instant::now();
    let (prepared, prep_timing) = prepare_heavy_volume_timed(surface, pressure, false)?;
    let ecape_triplet_start = Instant::now();
    let (fields, failure_count) =
        compute_ecape_map_fields_with_prepared_volume(surface, pressure, &prepared)?;
    let ecape_triplet_ms = ecape_triplet_start.elapsed().as_millis();
    let compute_ms = compute_start.elapsed().as_millis();

    let model_slug = request.model.as_str().replace('-', "_");
    let subtitle_left = format!(
        "{} {}Z F{:03}  {}",
        request.date_yyyymmdd, loaded.latest.cycle.hour_utc, request.forecast_hour, request.model
    );
    let source_label = format!("source: {}", loaded.latest.source.as_str());
    let (outputs, render_ms) = render_heavy_map_group(
        &request.out_dir,
        &model_slug,
        &request.date_yyyymmdd,
        loaded.latest.cycle.hour_utc,
        request.forecast_hour,
        &request.domain.slug,
        "ecape",
        &grid,
        &projected,
        &fields,
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
        severe_fields_ms: 0,
        render_ms,
        total_ms,
    };

    Ok(EcapeBatchReport {
        model: request.model,
        date_yyyymmdd: request.date_yyyymmdd.clone(),
        cycle_utc: loaded.latest.cycle.hour_utc,
        forecast_hour: request.forecast_hour,
        source: loaded.latest.source,
        domain: request.domain.clone(),
        outputs,
        input_fetches,
        shared_timing,
        heavy_timing,
        project_ms,
        compute_ms,
        render_ms,
        total_ms,
        failure_count,
    })
}

pub fn run_ecape_ratio_display_batch(
    request: &EcapeBatchRequest,
) -> Result<EcapeBatchReport, Box<dyn std::error::Error>> {
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
        .map_err(|err| format!("ECAPE ratio display surface/pressure pair unavailable: {err}"))?;
    let full_surface = &surface_decode.value;
    let full_pressure = &pressure_decode.value;
    let owned_full_grid = full_surface.core_grid()?;
    let project_start = Instant::now();
    let full_projected = build_projected_map_with_projection(
        &owned_full_grid.lat_deg,
        &owned_full_grid.lon_deg,
        full_surface.projection.as_ref(),
        request.domain.bounds,
        heavy_map_target_aspect_ratio(),
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
            heavy_map_target_aspect_ratio(),
        )?
    } else {
        full_projected
    };
    let project_ms = project_start.elapsed().as_millis();

    let compute_start = Instant::now();
    let (prepared, prep_timing) = prepare_heavy_volume_timed(surface, pressure, false)?;
    let ecape_triplet_start = Instant::now();
    let (fields, failure_count) =
        compute_ecape_map_fields_with_prepared_volume(surface, pressure, &prepared)?;
    let ecape_triplet_ms = ecape_triplet_start.elapsed().as_millis();
    let compute_ms = compute_start.elapsed().as_millis();

    let model_slug = request.model.as_str().replace('-', "_");
    let subtitle_left = format!(
        "{} {}Z F{:03}  {}",
        request.date_yyyymmdd, loaded.latest.cycle.hour_utc, request.forecast_hour, request.model
    );
    let source_label = format!("source: {}", loaded.latest.source.as_str());
    let (outputs, render_ms) = render_ecape_ratio_display_group(
        &request.out_dir,
        &model_slug,
        &request.date_yyyymmdd,
        loaded.latest.cycle.hour_utc,
        request.forecast_hour,
        &request.domain.slug,
        &grid,
        &projected,
        &fields,
        &subtitle_left,
        &source_label,
    )?;
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
        severe_fields_ms: 0,
        render_ms,
        total_ms,
    };

    Ok(EcapeBatchReport {
        model: request.model,
        date_yyyymmdd: request.date_yyyymmdd.clone(),
        cycle_utc: loaded.latest.cycle.hour_utc,
        forecast_hour: request.forecast_hour,
        source: loaded.latest.source,
        domain: request.domain.clone(),
        outputs,
        input_fetches,
        shared_timing,
        heavy_timing,
        project_ms,
        compute_ms,
        render_ms,
        total_ms,
        failure_count,
    })
}

fn render_ecape_ratio_display_group(
    out_dir: &Path,
    model_slug: &str,
    date_yyyymmdd: &str,
    cycle_utc: u8,
    forecast_hour: u16,
    domain_slug: &str,
    grid: &rustwx_core::LatLonGrid,
    projected: &ProjectedMap,
    fields: &[WeatherPanelField],
    subtitle_left: &str,
    source_label: &str,
) -> Result<(Vec<HeavyRenderedArtifact>, u128), Box<dyn std::error::Error>> {
    let mut outputs = Vec::with_capacity(6);
    let mut render_ms = 0;
    for parcel in ECAPE_RATIO_DISPLAY_PARCELS {
        let ecape = required_panel_field(fields, parcel.ecape_slug)?;
        let ratio = required_panel_field(fields, parcel.ratio_slug)?;

        let mut ratio_fill = ratio
            .clone()
            .with_title_override(format!("{} ECAPE/CAPE Ratio + Contours", parcel.label));
        ratio_fill.artifact_slug = Some(format!("{}_ratio_fill_contours", parcel.slug));
        render_ms += render_ecape_ratio_display_plot(
            out_dir,
            model_slug,
            date_yyyymmdd,
            cycle_utc,
            forecast_hour,
            domain_slug,
            grid,
            projected,
            &ratio_fill,
            ratio,
            subtitle_left,
            &format!("{source_label} | derived ratio fill"),
            &mut outputs,
        )?;

        let mut ecape_fill = ecape
            .clone()
            .with_title_override(format!("{}ECAPE + Ratio Contours", parcel.label));
        ecape_fill.artifact_slug = Some(format!("{}_ecape_fill_ratio_contours", parcel.slug));
        render_ms += render_ecape_ratio_display_plot(
            out_dir,
            model_slug,
            date_yyyymmdd,
            cycle_utc,
            forecast_hour,
            domain_slug,
            grid,
            projected,
            &ecape_fill,
            ratio,
            subtitle_left,
            &format!("{source_label} | ratio ctrs .50/.75/1.00"),
            &mut outputs,
        )?;
    }
    Ok((outputs, render_ms))
}

fn render_ecape_ratio_display_plot(
    out_dir: &Path,
    model_slug: &str,
    date_yyyymmdd: &str,
    cycle_utc: u8,
    forecast_hour: u16,
    domain_slug: &str,
    grid: &rustwx_core::LatLonGrid,
    projected: &ProjectedMap,
    field: &WeatherPanelField,
    ratio: &WeatherPanelField,
    subtitle_left: &str,
    subtitle_right: &str,
    outputs: &mut Vec<HeavyRenderedArtifact>,
) -> Result<u128, Box<dyn std::error::Error>> {
    let output_path = out_dir.join(format!(
        "rustwx_{}_{}_{}z_f{:03}_{}_ecape_ratio_display_{}.png",
        model_slug,
        date_yyyymmdd,
        cycle_utc,
        forecast_hour,
        domain_slug,
        field.artifact_slug()
    ));
    let mut request = build_weather_map_request(
        grid,
        projected,
        field,
        crate::heavy::HEAVY_MAP_WIDTH,
        crate::heavy::HEAVY_MAP_HEIGHT,
        Some(subtitle_left.to_string()),
        Some(subtitle_right.to_string()),
    )?;
    let contour_field = ratio_contour_field(&request.field, ratio);
    request.add_contour_field(
        &contour_field,
        ECAPE_RATIO_CONTOUR_LEVELS.to_vec(),
        ContourStyle {
            color: Color::rgba(15, 15, 15, 235),
            width: 1,
            labels: false,
            show_extrema: false,
            ..Default::default()
        },
    )?;
    let timing = save_png_profile(&request, &output_path)?;
    outputs.push(HeavyRenderedArtifact {
        product: field.artifact_slug().to_string(),
        title: field.display_title().to_string(),
        output_identity: artifact_identity_from_path(&output_path)?,
        output_path,
    });
    Ok(timing.total_ms)
}

fn ratio_contour_field(base: &RenderField2D, ratio: &WeatherPanelField) -> RenderField2D {
    let mut field = base.clone();
    field.product = RenderProductKey::named(format!("{}_contours", ratio.artifact_slug()));
    field.units = "ratio".to_string();
    field.values = ratio.values.iter().map(|&value| value as f32).collect();
    field
}

fn required_panel_field<'a>(
    fields: &'a [WeatherPanelField],
    slug: &str,
) -> Result<&'a WeatherPanelField, Box<dyn std::error::Error>> {
    fields
        .iter()
        .find(|field| field.artifact_slug() == slug)
        .ok_or_else(|| format!("missing ECAPE field '{slug}'").into())
}

#[derive(Debug, Clone, Copy)]
struct EcapeRatioDisplayParcel {
    slug: &'static str,
    label: &'static str,
    ecape_slug: &'static str,
    ratio_slug: &'static str,
}

const ECAPE_RATIO_CONTOUR_LEVELS: [f64; 3] = [0.5, 0.75, 1.0];

const ECAPE_RATIO_DISPLAY_PARCELS: [EcapeRatioDisplayParcel; 3] = [
    EcapeRatioDisplayParcel {
        slug: "sb",
        label: "SB",
        ecape_slug: "sbecape",
        ratio_slug: "sb_ecape_derived_cape_ratio",
    },
    EcapeRatioDisplayParcel {
        slug: "ml",
        label: "ML",
        ecape_slug: "mlecape",
        ratio_slug: "ml_ecape_derived_cape_ratio",
    },
    EcapeRatioDisplayParcel {
        slug: "mu",
        label: "MU",
        ecape_slug: "muecape",
        ratio_slug: "mu_ecape_derived_cape_ratio",
    },
];

pub fn compute_ecape_map_fields(
    surface: &SurfaceFields,
    pressure: &PressureFields,
) -> Result<(Vec<WeatherPanelField>, usize), Box<dyn std::error::Error>> {
    let prepared = prepare_heavy_volume(surface, pressure, false)?;
    compute_ecape_map_fields_with_prepared_volume(surface, pressure, &prepared)
}

/// Per-kernel wall times inside [`compute_ecape_map_fields_with_prepared_volume`],
/// for callers (the store-ingest lane) that need an honest breakdown of
/// where the heavy stage's time goes. Pure observation: the timed variant
/// runs the exact same kernels in the exact same order.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct EcapeMapFieldsTiming {
    /// The SB/ML/MU entraining-parcel ECAPE triplet (the dominant cost).
    pub ecape_triplet_ms: u128,
    /// SRH 0-1/0-3 km + 0-6 km bulk shear for the experimental composites.
    pub wind_diagnostics_ms: u128,
    /// The classic (non-entraining) ML parcel pass feeding the STP's LCL.
    pub ml_classic_ms: u128,
    /// Elementwise composites + ratios (SCP/EHI/STP, derived/native ratios).
    pub composites_ms: u128,
}

pub fn compute_ecape_map_fields_with_prepared_volume(
    surface: &SurfaceFields,
    pressure: &PressureFields,
    prepared: &crate::gridded::PreparedHeavyVolume,
) -> Result<(Vec<WeatherPanelField>, usize), Box<dyn std::error::Error>> {
    let (fields, failure_count, _) =
        compute_ecape_map_fields_with_prepared_volume_timed(surface, pressure, prepared)?;
    Ok((fields, failure_count))
}

pub fn compute_ecape_map_fields_with_prepared_volume_timed(
    surface: &SurfaceFields,
    pressure: &PressureFields,
    prepared: &crate::gridded::PreparedHeavyVolume,
) -> Result<(Vec<WeatherPanelField>, usize, EcapeMapFieldsTiming), Box<dyn std::error::Error>> {
    let mut timing = EcapeMapFieldsTiming::default();
    let triplet_start = Instant::now();
    let triplet = compute_ecape_triplet_with_failure_mask_from_parts(
        prepared.grid,
        EcapeVolumeInputs {
            pressure_pa: prepared
                .pressure_3d_pa
                .as_deref()
                .unwrap_or(&prepared.pressure_levels_pa),
            temperature_c: &pressure.temperature_c_3d,
            qvapor_kgkg: &pressure.qvapor_kgkg_3d,
            height_agl_m: &prepared.height_agl_3d,
            u_ms: &pressure.u_ms_3d,
            v_ms: &pressure.v_ms_3d,
            nz: prepared.shape.nz,
        },
        SurfaceInputs {
            psfc_pa: &surface.psfc_pa,
            t2_k: &surface.t2_k,
            q2_kgkg: &surface.q2_kgkg,
            u10_ms: &surface.u10_ms,
            v10_ms: &surface.v10_ms,
        },
        EcapeTripletOptions::new("right_moving"),
    )?;
    timing.ecape_triplet_ms = triplet_start.elapsed().as_millis();
    let wind_start = Instant::now();
    let wind = WindGridInputs {
        shape: prepared.shape,
        u_3d_ms: &pressure.u_ms_3d,
        v_3d_ms: &pressure.v_ms_3d,
        height_agl_3d_m: &prepared.height_agl_3d,
    };
    let wind_diagnostics = compute_wind_diagnostics_bundle(wind)?;
    timing.wind_diagnostics_ms = wind_start.elapsed().as_millis();
    let composites_start = Instant::now();
    let experimental = compute_scp_ehi(ScpEhiInputs {
        grid: prepared.grid,
        scp_cape_jkg: &triplet.mu.fields.ecape_jkg,
        scp_srh_m2s2: &wind_diagnostics.srh_03km_m2s2,
        scp_bulk_wind_difference_ms: &wind_diagnostics.shear_06km_ms,
        ehi_cape_jkg: &triplet.sb.fields.ecape_jkg,
        ehi_srh_m2s2: &wind_diagnostics.srh_01km_m2s2,
    })?;
    let ecape_ehi_03km = compute_ehi(
        prepared.grid,
        &triplet.sb.fields.ecape_jkg,
        &wind_diagnostics.srh_03km_m2s2,
    )?;
    timing.composites_ms += composites_start.elapsed().as_millis();
    let ml_classic_start = Instant::now();
    let ml_classic = compute_mlcape_cin(
        prepared.grid,
        EcapeVolumeInputs {
            pressure_pa: prepared
                .pressure_3d_pa
                .as_deref()
                .unwrap_or(&prepared.pressure_levels_pa),
            temperature_c: &pressure.temperature_c_3d,
            qvapor_kgkg: &pressure.qvapor_kgkg_3d,
            height_agl_m: &prepared.height_agl_3d,
            u_ms: &pressure.u_ms_3d,
            v_ms: &pressure.v_ms_3d,
            nz: prepared.shape.nz,
        },
        SurfaceInputs {
            psfc_pa: &surface.psfc_pa,
            t2_k: &surface.t2_k,
            q2_kgkg: &surface.q2_kgkg,
            u10_ms: &surface.u10_ms,
            v10_ms: &surface.v10_ms,
        },
        None,
    )?;
    timing.ml_classic_ms = ml_classic_start.elapsed().as_millis();
    let tail_start = Instant::now();
    let ecape_stp = compute_stp_effective(EffectiveStpInputs {
        grid: prepared.grid,
        mlcape_jkg: &triplet.ml.fields.ecape_jkg,
        mlcin_jkg: &triplet.ml.fields.cin_jkg,
        ml_lcl_m: &ml_classic.lcl_m,
        effective_srh_m2s2: &wind_diagnostics.srh_01km_m2s2,
        effective_bulk_wind_difference_ms: &wind_diagnostics.shear_06km_ms,
    })?;
    let failure_count = triplet.total_failure_count();

    let sb_derived_ratio =
        ecape_cape_ratio(&triplet.sb.fields.ecape_jkg, &triplet.sb.fields.cape_jkg);
    let ml_derived_ratio =
        ecape_cape_ratio(&triplet.ml.fields.ecape_jkg, &triplet.ml.fields.cape_jkg);
    let mu_derived_ratio =
        ecape_cape_ratio(&triplet.mu.fields.ecape_jkg, &triplet.mu.fields.cape_jkg);

    let mut fields = vec![
        WeatherPanelField::new(WeatherProduct::Sbecape, "J/kg", triplet.sb.fields.ecape_jkg),
        WeatherPanelField::new(WeatherProduct::Mlecape, "J/kg", triplet.ml.fields.ecape_jkg),
        WeatherPanelField::new(WeatherProduct::Muecape, "J/kg", triplet.mu.fields.ecape_jkg),
        WeatherPanelField::new(
            WeatherProduct::SbEcapeDerivedCapeRatio,
            "ratio",
            sb_derived_ratio,
        ),
        WeatherPanelField::new(
            WeatherProduct::MlEcapeDerivedCapeRatio,
            "ratio",
            ml_derived_ratio,
        ),
        WeatherPanelField::new(
            WeatherProduct::MuEcapeDerivedCapeRatio,
            "ratio",
            mu_derived_ratio,
        ),
        WeatherPanelField::new(WeatherProduct::Sbncape, "J/kg", triplet.sb.fields.ncape_jkg),
        WeatherPanelField::new(WeatherProduct::Sbecin, "J/kg", triplet.sb.fields.cin_jkg),
        WeatherPanelField::new(WeatherProduct::Mlecin, "J/kg", triplet.ml.fields.cin_jkg),
        WeatherPanelField::new(
            WeatherProduct::EcapeScpExperimental,
            "dimensionless",
            experimental.scp,
        ),
        WeatherPanelField::new(
            WeatherProduct::EcapeEhi01kmExperimental,
            "dimensionless",
            experimental.ehi,
        ),
        WeatherPanelField::new(
            WeatherProduct::EcapeEhi03kmExperimental,
            "dimensionless",
            ecape_ehi_03km,
        ),
        WeatherPanelField::new(
            WeatherProduct::EcapeStpExperimental,
            "dimensionless",
            ecape_stp,
        ),
    ];
    if let Some(native_sbcape_jkg) = surface.native_sbcape_jkg.as_ref() {
        fields.push(WeatherPanelField::new(
            WeatherProduct::SbEcapeNativeCapeRatio,
            "ratio",
            ecape_cape_ratio(&fields[0].values, native_sbcape_jkg),
        ));
    }
    if let Some(native_mlcape_jkg) = surface.native_mlcape_jkg.as_ref() {
        fields.push(WeatherPanelField::new(
            WeatherProduct::MlEcapeNativeCapeRatio,
            "ratio",
            ecape_cape_ratio(&fields[1].values, native_mlcape_jkg),
        ));
    }
    if let Some(native_mucape_jkg) = surface.native_mucape_jkg.as_ref() {
        fields.push(WeatherPanelField::new(
            WeatherProduct::MuEcapeNativeCapeRatio,
            "ratio",
            ecape_cape_ratio(&fields[2].values, native_mucape_jkg),
        ));
    }
    timing.composites_ms += tail_start.elapsed().as_millis();
    Ok((fields, failure_count, timing))
}

fn ecape_cape_ratio(ecape_jkg: &[f64], cape_jkg: &[f64]) -> Vec<f64> {
    ecape_jkg
        .iter()
        .zip(cape_jkg.iter())
        .map(|(&ecape, &cape)| {
            if ecape.is_finite() && cape.is_finite() && cape >= ECAPE_CAPE_RATIO_MIN_DENOMINATOR_JKG
            {
                ecape / cape
            } else {
                f64::NAN
            }
        })
        .collect()
}
