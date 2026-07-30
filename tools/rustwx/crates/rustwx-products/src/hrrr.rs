use crate::gridded::{
    PreparedHeavyVolume as GenericPreparedHeavyVolume, PressureFields as GenericPressureFields,
    SurfaceFields as GenericSurfaceFields,
    prepare_heavy_volume_timed as prepare_generic_heavy_volume_timed,
};
use crate::heavy::crop_and_guard_heavy_domain;
use crate::publication::{
    ArtifactContentIdentity, PublishedFetchIdentity, artifact_identity_from_path,
};
use crate::runtime::{BundleLoaderConfig, LoadedBundleSet, load_execution_plan};
use crate::severe::{
    build_planned_input_fetches, build_severe_execution_plan,
    compute_severe_panel_fields_with_prepared_volume as compute_generic_severe_panel_fields_with_prepared_volume,
    severe_panel_fields_from_supported as generic_severe_panel_fields_from_supported,
};
pub use crate::shared_context::{
    DomainSpec, PreparedProjectedContext, ProjectedMap, WeatherPanelField, WeatherPanelHeader,
    WeatherPanelLayout, layout_key, render_two_by_four_weather_panel,
};
use rustwx_calc::SupportedSevereFields;
use rustwx_core::{CycleSpec, ModelId, SourceId};
use rustwx_models::{LatestRun, latest_available_run_at_forecast_hour};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::fs;
use std::io;
use std::path::PathBuf;
use std::thread;
use std::time::Instant;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum HrrrBatchProduct {
    SevereProofPanel,
}

impl HrrrBatchProduct {
    pub fn slug(self) -> &'static str {
        match self {
            Self::SevereProofPanel => "severe_proof_panel",
        }
    }

    pub fn layout(self) -> WeatherPanelLayout {
        match self {
            Self::SevereProofPanel => WeatherPanelLayout {
                top_padding: 86,
                ..Default::default()
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HrrrBatchRequest {
    pub date_yyyymmdd: String,
    pub cycle_override_utc: Option<u8>,
    pub forecast_hour: u16,
    pub source: SourceId,
    pub domain: DomainSpec,
    pub out_dir: PathBuf,
    pub cache_root: PathBuf,
    pub use_cache: bool,
    pub products: Vec<HrrrBatchProduct>,
    pub allow_large_heavy_domain: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HrrrFetchRuntimeInfo {
    pub planned_product: String,
    pub fetched_product: String,
    pub requested_source: SourceId,
    pub resolved_source: SourceId,
    pub resolved_url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HrrrSharedTiming {
    pub fetch_surface_ms: u128,
    pub fetch_pressure_ms: u128,
    pub decode_surface_ms: u128,
    pub decode_pressure_ms: u128,
    pub fetch_surface_cache_hit: bool,
    pub fetch_pressure_cache_hit: bool,
    pub decode_surface_cache_hit: bool,
    pub decode_pressure_cache_hit: bool,
    pub surface_fetch: HrrrFetchRuntimeInfo,
    pub pressure_fetch: HrrrFetchRuntimeInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HrrrProductTiming {
    pub project_ms: u128,
    pub compute_ms: u128,
    pub render_ms: u128,
    pub total_ms: u128,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HrrrRenderedProduct {
    pub product: HrrrBatchProduct,
    pub output_path: PathBuf,
    pub timing: HrrrProductTiming,
    pub metadata: HrrrProductMetadata,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_identity: Option<ArtifactContentIdentity>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub input_fetch_keys: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HrrrProductMetadata {
    pub failure_count: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HrrrBatchReport {
    pub date_yyyymmdd: String,
    pub cycle_utc: u8,
    pub forecast_hour: u16,
    pub source: SourceId,
    pub domain: DomainSpec,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub input_fetches: Vec<PublishedFetchIdentity>,
    pub products: Vec<HrrrRenderedProduct>,
    pub shared_timing: HrrrSharedTiming,
    pub total_ms: u128,
}

pub fn resolve_hrrr_run(
    date: &str,
    cycle_override: Option<u8>,
    forecast_hour: u16,
    source: SourceId,
) -> Result<LatestRun, Box<dyn std::error::Error>> {
    match cycle_override {
        Some(hour) => Ok(LatestRun {
            model: ModelId::Hrrr,
            cycle: CycleSpec::new(date, hour)?,
            source,
        }),
        None => Ok(latest_available_run_at_forecast_hour(
            ModelId::Hrrr,
            Some(source),
            date,
            forecast_hour,
        )?),
    }
}

pub fn run_hrrr_batch(
    request: &HrrrBatchRequest,
) -> Result<HrrrBatchReport, Box<dyn std::error::Error>> {
    fs::create_dir_all(&request.out_dir)?;
    if request.use_cache {
        fs::create_dir_all(&request.cache_root)?;
    }

    let latest = resolve_hrrr_run(
        &request.date_yyyymmdd,
        request.cycle_override_utc,
        request.forecast_hour,
        request.source,
    )?;
    // Build a planner-driven execution plan for this hour: surface +
    // pressure analyses are the only bundles severe/ECAPE need, and the
    // planner dedupes if both products are requested in the same pass.
    let plan = build_severe_execution_plan(&latest, request.forecast_hour, None, None);
    let loaded = load_execution_plan(
        plan,
        &BundleLoaderConfig {
            cache_root: request.cache_root.clone(),
            use_cache: request.use_cache,
        },
    )?;
    run_hrrr_batch_from_loaded(request, &loaded)
}

/// Internal entry point used by the unified non-ECAPE-hour runner: the
/// planner has already loaded surface+pressure bundles, so the bundled
/// HRRR products consume the same `LoadedBundleSet` instead of doing
/// their own fetch/decode.
pub(crate) fn run_hrrr_batch_from_loaded(
    request: &HrrrBatchRequest,
    loaded: &LoadedBundleSet,
) -> Result<HrrrBatchReport, Box<dyn std::error::Error>> {
    let total_start = Instant::now();
    let (_, surface_decode, _, pressure_decode) = loaded
        .surface_pressure_pair()
        .ok_or("HRRR batch planner missed surface or pressure bundle")?;
    let surface_full = &surface_decode.value;
    let pressure_full = &pressure_decode.value;
    let unique_products = dedupe_products(&request.products);
    let crop_target_ratio = unique_products
        .iter()
        .map(|product| product.layout().target_aspect_ratio())
        .reduce(f64::max)
        .unwrap_or(1.0);
    let owned_full_grid;
    let base_grid = surface_full.core_grid()?;
    let full_projected_for_crop = crate::direct::build_projected_map_with_projection(
        &base_grid.lat_deg,
        &base_grid.lon_deg,
        surface_full.projection.as_ref(),
        request.domain.bounds,
        crop_target_ratio,
    )?;
    let cropped_heavy_domain = crop_and_guard_heavy_domain(
        surface_full,
        pressure_full,
        &full_projected_for_crop,
        &request.domain,
        2,
        request.allow_large_heavy_domain,
    )?;
    let (surface, pressure, grid) = match cropped_heavy_domain.cropped.as_ref() {
        Some(cropped) => (&cropped.surface, &cropped.pressure, &cropped.grid),
        None => {
            owned_full_grid = base_grid;
            (surface_full, pressure_full, &owned_full_grid)
        }
    };
    let render_parallelism = self::png_render_parallelism(unique_products.len());
    let mut projected_maps = HashMap::<(u32, u32, u32), ProjectedMap>::new();
    let mut project_timings = Vec::with_capacity(unique_products.len());

    for product in &unique_products {
        let layout = product.layout();
        let key = self::layout_key(layout);
        let project_start = Instant::now();
        if !projected_maps.contains_key(&key) {
            let projected = crate::direct::build_projected_map_with_projection(
                &grid.lat_deg,
                &grid.lon_deg,
                surface.projection.as_ref(),
                request.domain.bounds,
                layout.target_aspect_ratio(),
            )?;
            projected_maps.insert(key, projected);
        }
        project_timings.push(project_start.elapsed().as_millis());
    }

    let date_yyyymmdd = request.date_yyyymmdd.as_str();
    let cycle_utc = loaded.latest.cycle.hour_utc;
    let forecast_hour = request.forecast_hour;
    let domain_slug = request.domain.slug.as_str();
    let input_fetches = build_planned_input_fetches(loaded);
    let input_fetch_keys = input_fetches
        .iter()
        .map(|fetch| fetch.fetch_key.clone())
        .collect::<Vec<_>>();
    let needs_heavy = unique_products
        .iter()
        .any(|product| matches!(product, HrrrBatchProduct::SevereProofPanel));
    let prepared_heavy_volume = if needs_heavy {
        let (prepared, _prep_timing) =
            prepare_generic_heavy_volume_timed(surface, pressure, false)?;
        Some(prepared)
    } else {
        None
    };
    let shared_timing_for_report = build_hrrr_shared_timing_from_loaded(loaded)?;
    let products = thread::scope(|scope| -> Result<Vec<HrrrRenderedProduct>, io::Error> {
        let mut products = Vec::with_capacity(unique_products.len());
        let mut pending = VecDeque::new();

        for (idx, product) in unique_products.iter().copied().enumerate() {
            let product_start = Instant::now();
            let project_ms = project_timings[idx];
            let layout = product.layout();
            let lane_fetch_keys = input_fetch_keys.clone();

            let compute_start = Instant::now();
            let computed = compute_hrrr_batch_product(
                product,
                date_yyyymmdd,
                cycle_utc,
                forecast_hour,
                surface,
                pressure,
                prepared_heavy_volume.as_ref(),
            )
            .map_err(self::thread_render_error)?;
            let compute_ms = compute_start.elapsed().as_millis();

            let output_path = request.out_dir.join(format!(
                "rustwx_hrrr_{}_{}z_f{:02}_{}_{}.png",
                date_yyyymmdd,
                cycle_utc,
                forecast_hour,
                domain_slug,
                product.slug()
            ));
            let projected = projected_maps
                .get(&self::layout_key(layout))
                .ok_or_else(|| io::Error::other("missing projected map for HRRR batch render"))?;

            pending.push_back(
                scope.spawn(move || -> Result<HrrrRenderedProduct, io::Error> {
                    let render_start = Instant::now();
                    render_two_by_four_weather_panel(
                        &output_path,
                        grid,
                        projected,
                        &computed.fields,
                        &computed.header,
                        layout,
                    )
                    .map_err(self::thread_render_error)?;
                    let render_ms = render_start.elapsed().as_millis();
                    let content_identity = artifact_identity_from_path(&output_path)
                        .map_err(self::thread_render_error)?;

                    Ok(HrrrRenderedProduct {
                        product,
                        output_path,
                        timing: HrrrProductTiming {
                            project_ms,
                            compute_ms,
                            render_ms,
                            total_ms: product_start.elapsed().as_millis(),
                        },
                        metadata: computed.metadata,
                        content_identity: Some(content_identity),
                        input_fetch_keys: lane_fetch_keys,
                    })
                }),
            );

            if pending.len() >= render_parallelism {
                products.push(self::join_render_job(pending.pop_front().unwrap())?);
            }
        }

        while let Some(handle) = pending.pop_front() {
            products.push(self::join_render_job(handle)?);
        }

        Ok(products)
    })
    .map_err(|err| -> Box<dyn std::error::Error> { Box::new(err) })?;

    Ok(HrrrBatchReport {
        date_yyyymmdd: request.date_yyyymmdd.clone(),
        cycle_utc,
        forecast_hour,
        source: loaded.latest.source,
        domain: request.domain.clone(),
        input_fetches,
        products,
        shared_timing: shared_timing_for_report,
        total_ms: total_start.elapsed().as_millis(),
    })
}

/// Bridge: the legacy `HrrrSharedTiming` block stays in the public
/// report so existing manifest/archive consumers keep working, but the
/// data now comes straight off the planner's `LoadedBundleSet`.
fn build_hrrr_shared_timing_from_loaded(
    loaded: &LoadedBundleSet,
) -> Result<HrrrSharedTiming, Box<dyn std::error::Error>> {
    let surface_planned = loaded
        .plan
        .bundle_for(
            rustwx_core::CanonicalBundleDescriptor::SurfaceAnalysis,
            loaded.forecast_hour,
        )
        .ok_or("loaded bundle set missing surface analysis for HRRR shared timing")?;
    let pressure_planned = loaded
        .plan
        .bundle_for(
            rustwx_core::CanonicalBundleDescriptor::PressureAnalysis,
            loaded.forecast_hour,
        )
        .ok_or("loaded bundle set missing pressure analysis for HRRR shared timing")?;
    let surface_fetched = loaded
        .fetched_for(surface_planned)
        .ok_or("loader missing surface fetch for HRRR shared timing")?;
    let pressure_fetched = loaded
        .fetched_for(pressure_planned)
        .ok_or("loader missing pressure fetch for HRRR shared timing")?;
    let surface_decode = loaded
        .surface_decode_for(
            rustwx_core::CanonicalBundleDescriptor::SurfaceAnalysis,
            loaded.forecast_hour,
        )
        .ok_or("loader missing surface decode for HRRR shared timing")?;
    let pressure_decode = loaded
        .pressure_decode_for(
            rustwx_core::CanonicalBundleDescriptor::PressureAnalysis,
            loaded.forecast_hour,
        )
        .ok_or("loader missing pressure decode for HRRR shared timing")?;

    Ok(HrrrSharedTiming {
        fetch_surface_ms: surface_fetched.fetch_ms,
        fetch_pressure_ms: pressure_fetched.fetch_ms,
        decode_surface_ms: 0,
        decode_pressure_ms: 0,
        fetch_surface_cache_hit: surface_fetched.file.fetched.cache_hit,
        fetch_pressure_cache_hit: pressure_fetched.file.fetched.cache_hit,
        decode_surface_cache_hit: surface_decode.cache_hit,
        decode_pressure_cache_hit: pressure_decode.cache_hit,
        surface_fetch: HrrrFetchRuntimeInfo {
            planned_product: surface_planned.resolved.native_product.clone(),
            fetched_product: surface_fetched.file.request.request.product.clone(),
            requested_source: surface_fetched
                .file
                .request
                .source_override
                .unwrap_or(surface_fetched.file.fetched.result.source),
            resolved_source: surface_fetched.file.fetched.result.source,
            resolved_url: surface_fetched.file.fetched.result.url.clone(),
        },
        pressure_fetch: HrrrFetchRuntimeInfo {
            planned_product: pressure_planned.resolved.native_product.clone(),
            fetched_product: pressure_fetched.file.request.request.product.clone(),
            requested_source: pressure_fetched
                .file
                .request
                .source_override
                .unwrap_or(pressure_fetched.file.fetched.result.source),
            resolved_source: pressure_fetched.file.fetched.result.source,
            resolved_url: pressure_fetched.file.fetched.result.url.clone(),
        },
    })
}

#[derive(Debug)]
struct ComputedHrrrProduct {
    fields: Vec<WeatherPanelField>,
    header: WeatherPanelHeader,
    metadata: HrrrProductMetadata,
}

fn compute_hrrr_batch_product(
    product: HrrrBatchProduct,
    date_yyyymmdd: &str,
    cycle_utc: u8,
    forecast_hour: u16,
    surface: &GenericSurfaceFields,
    pressure: &GenericPressureFields,
    prepared_heavy_volume: Option<&GenericPreparedHeavyVolume>,
) -> Result<ComputedHrrrProduct, Box<dyn std::error::Error>> {
    match product {
        HrrrBatchProduct::SevereProofPanel => {
            let fields = match prepared_heavy_volume {
                Some(prepared) => compute_generic_severe_panel_fields_with_prepared_volume(
                    surface, pressure, prepared,
                )?,
                None => crate::severe::compute_severe_panel_fields(surface, pressure)?,
            };
            let header = WeatherPanelHeader::new(format!(
                "HRRR Severe Proof Panel  Run: {} {:02}:00 UTC  Forecast Hour: F{:02}",
                date_yyyymmdd, cycle_utc, forecast_hour
            ))
            .with_subtitle_line(
                "STP is fixed-layer only: sbCAPE + sbLCL + 0-1 km SRH + 0-6 km bulk shear.",
            )
            .with_subtitle_line(
                "SCP stays a fixed-depth proxy here: muCAPE + 0-3 km SRH + 0-6 km shear. EHI 0-1 km uses sbCAPE + 0-1 km SRH. Effective-layer derivation is not wired yet.",
            );
            Ok(ComputedHrrrProduct {
                fields,
                header,
                metadata: HrrrProductMetadata::default(),
            })
        }
    }
}

fn dedupe_products(products: &[HrrrBatchProduct]) -> Vec<HrrrBatchProduct> {
    let mut seen = std::collections::HashSet::new();
    let mut unique = Vec::new();
    for product in products {
        if seen.insert(*product) {
            unique.push(*product);
        }
    }
    unique
}

/// Re-export for callers that still want the supported-severe → panel
/// fields conversion under the HRRR-pinned name. Implementation lives in
/// `crate::severe`; preserved here so external consumers do not break.
pub fn severe_panel_fields_from_supported(fields: SupportedSevereFields) -> Vec<WeatherPanelField> {
    generic_severe_panel_fields_from_supported(fields)
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

fn join_scoped_job<T>(
    handle: thread::ScopedJoinHandle<'_, Result<T, io::Error>>,
) -> Result<T, io::Error> {
    match handle.join() {
        Ok(result) => result,
        Err(panic) => Err(io::Error::other(format!(
            "worker panicked: {}",
            panic_message(panic)
        ))),
    }
}

fn join_render_job<T>(
    handle: thread::ScopedJoinHandle<'_, Result<T, io::Error>>,
) -> Result<T, io::Error> {
    join_scoped_job(handle).map_err(|err| io::Error::other(format!("render worker failed: {err}")))
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

#[cfg(test)]
mod tests {
    use super::*;
    use rustwx_calc::SupportedSevereFields;
    use rustwx_render::WeatherProduct;

    #[test]
    fn explicit_hrrr_cycle_avoids_latest_probe() {
        let latest = resolve_hrrr_run("20260414", Some(19), 0, SourceId::Aws).unwrap();
        assert_eq!(latest.model, ModelId::Hrrr);
        assert_eq!(latest.cycle.date_yyyymmdd, "20260414");
        assert_eq!(latest.cycle.hour_utc, 19);
        assert_eq!(latest.source, SourceId::Aws);
    }

    #[test]
    fn fetch_runtime_info_keeps_planned_and_actual_fetch_truth() {
        // `HrrrFetchRuntimeInfo` is still serialized into
        // `HrrrSharedTiming` and therefore part of the downstream
        // report wire format. Assert that the struct preserves the
        // "planned vs actual" truth (including nat->sfc aliasing and
        // requested-vs-resolved source) that manifest aliasing relies
        // on.
        let runtime = HrrrFetchRuntimeInfo {
            planned_product: "nat".to_string(),
            fetched_product: "sfc".to_string(),
            requested_source: SourceId::Nomads,
            resolved_source: SourceId::Nomads,
            resolved_url: "https://example.test/hrrr.t23z.wrfsfcf06.grib2".to_string(),
        };

        assert_eq!(runtime.planned_product, "nat");
        assert_eq!(runtime.fetched_product, "sfc");
        assert_ne!(runtime.planned_product, runtime.fetched_product);
        assert_eq!(runtime.requested_source, SourceId::Nomads);
        assert_eq!(runtime.resolved_source, SourceId::Nomads);
        assert!(runtime.resolved_url.contains("wrfsfc"));
    }

    #[test]
    fn panel_field_keeps_title_override() {
        let field = WeatherPanelField::new(WeatherProduct::StpFixed, "dimensionless", vec![1.0])
            .with_title_override("STP (FIXED)");
        assert_eq!(field.title_override.as_deref(), Some("STP (FIXED)"));
    }

    #[test]
    fn batch_product_dedupe_preserves_first_seen_order() {
        let products = dedupe_products(&[
            HrrrBatchProduct::SevereProofPanel,
            HrrrBatchProduct::SevereProofPanel,
        ]);
        assert_eq!(products, vec![HrrrBatchProduct::SevereProofPanel]);
    }

    #[test]
    fn severe_field_titles_keep_current_labels_explicit() {
        let fields = severe_panel_fields_from_supported(SupportedSevereFields {
            sbcape_jkg: vec![1.0],
            mlcin_jkg: vec![-25.0],
            mucape_jkg: vec![2.0],
            srh_01km_m2s2: vec![100.0],
            srh_03km_m2s2: vec![200.0],
            shear_06km_ms: vec![20.0],
            stp_fixed: vec![1.5],
            scp_mu_03km_06km_proxy: vec![5.0],
            ehi_sb_01km_proxy: vec![2.0],
            tehi: vec![3.0],
            tts: vec![4.0],
            vtp_mod: vec![5.0],
        });

        assert_eq!(fields.len(), 11);
        assert_eq!(fields[5].product, WeatherProduct::StpFixed);
        assert_eq!(
            fields[6].title_override.as_deref(),
            Some("SCP (MU / 0-3 KM / 0-6 KM PROXY)")
        );
        assert_eq!(fields[7].title_override.as_deref(), Some("EHI 0-1 KM"));
        assert_eq!(fields[8].product, WeatherProduct::Tehi);
        assert_eq!(fields[9].product, WeatherProduct::Tts);
        assert_eq!(fields[10].product, WeatherProduct::VtpMod);
    }
}
