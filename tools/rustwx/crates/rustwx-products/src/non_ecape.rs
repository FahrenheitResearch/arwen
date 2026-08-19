use crate::derived::{
    DerivedBatchRequest, DerivedRecipeBlocker, HrrrDerivedBatchReport, PlannedDerivedSourceRoutes,
    derived_compute_recipes_need_pressure, is_heavy_derived_recipe_slug,
    maybe_load_special_pair_for_derived, plan_derived_recipes,
    plan_native_thermo_routes_with_surface_product, prepare_shared_derived_fields,
    run_model_derived_batch_from_loaded, run_model_derived_batch_from_loaded_with_precomputed,
    run_model_derived_batch_without_loaded,
};
use crate::direct::{
    DirectBatchRequest, FetchGroup, PreparedDirectBatch, prepare_direct_batch_from_loaded,
    run_direct_batch_from_loaded, run_direct_batch_from_prepared,
};
use crate::hrrr::{DomainSpec, resolve_hrrr_run};
use crate::shared_context::TitleProvenance;
use crate::planner::ExecutionPlanBuilder;
use crate::publication::{
    default_run_manifest_path, finalize_and_publish_run_manifest, publish_run_manifest_with_attempt,
};
use crate::runtime::{BundleLoaderConfig, load_execution_plan};
use crate::severe::build_severe_execution_plan;
use crate::source::{ProductSourceMode, ProductSourceRoute};
use crate::windowed::{
    HrrrWindowedBatchRequest, HrrrWindowedProduct, PreparedWindowedBatch,
    prepare_hrrr_windowed_batch_with_context, run_hrrr_windowed_batch_from_prepared,
    run_hrrr_windowed_batch_with_context,
};
use rustwx_core::{BundleRequirement, CanonicalBundleDescriptor, ModelId, SourceId};
use rustwx_models::{LatestRun, latest_available_run_at_forecast_hour, plot_recipe};
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs;
use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use std::time::Instant;

mod manifest;
mod summary;
mod types;

use manifest::{
    apply_derived_manifest_updates, apply_direct_manifest_updates, apply_windowed_manifest_updates,
    build_run_manifest, collect_input_fetches,
};
use summary::{build_static_domain_timings, build_static_product_timings, build_summary};
pub use types::{
    HrrrNonEcapeDomainReport, HrrrNonEcapeFanoutTiming, HrrrNonEcapeHourReport,
    HrrrNonEcapeHourRequest, HrrrNonEcapeHourRequestedProducts, HrrrNonEcapeHourSummary,
    HrrrNonEcapeMultiDomainReport, HrrrNonEcapeMultiDomainRequest, HrrrNonEcapeSharedTiming,
    NonEcapeBuildDomainTiming, NonEcapeBuildProductTiming, NonEcapeDomainReport,
    NonEcapeFanoutTiming, NonEcapeHourBuildReport, NonEcapeHourReport, NonEcapeHourRequest,
    NonEcapeHourSummary, NonEcapeMultiDomainReport, NonEcapeMultiDomainRequest,
    NonEcapeRequestedProducts, NonEcapeSharedTiming,
};
use types::{non_ecape_derived_contour_mode, non_ecape_native_fill_level_multiplier};

#[cfg(test)]
use crate::direct::HrrrDirectBatchReport;
#[cfg(test)]
use crate::publication::{ArtifactPublicationState, PublishedFetchIdentity};
#[cfg(test)]
use crate::windowed::{HrrrWindowedBatchReport, windowed_product_input_fetch_keys};
#[cfg(test)]
use manifest::count_blocked_artifacts;
#[cfg(test)]
use std::path::PathBuf;

struct PreparedNonEcapeHour {
    normalized: NonEcapeRequestedProducts,
    latest: LatestRun,
    derived_recipes: Vec<crate::derived::DerivedRecipe>,
    precomputed_direct: Option<Arc<PreparedDirectBatch>>,
    precomputed_derived: Option<crate::derived::PreparedSharedDerivedFields>,
    precomputed_windowed: Option<Arc<PreparedWindowedBatch>>,
    direct_loaded: Option<Arc<crate::runtime::LoadedBundleSet>>,
    derived_loaded: Option<Arc<crate::runtime::LoadedBundleSet>>,
    derived_lane_blocker: Option<String>,
    timing: NonEcapeSharedTiming,
}

#[derive(Debug, Default)]
struct SharedLoaderProfile {
    fetch_ms_total: u128,
    decode_surface_ms_total: u128,
    decode_pressure_ms_total: u128,
    fetched_bundle_count: usize,
    surface_decode_count: usize,
    pressure_decode_count: usize,
}

impl SharedLoaderProfile {
    fn record(&mut self, loaded: &crate::runtime::LoadedBundleSet) {
        self.fetch_ms_total += loaded.timing.fetch_ms_total;
        self.decode_surface_ms_total += loaded.timing.decode_surface_ms_total;
        self.decode_pressure_ms_total += loaded.timing.decode_pressure_ms_total;
        self.fetched_bundle_count += loaded.fetched.len();
        self.surface_decode_count += loaded.surface_decodes.len();
        self.pressure_decode_count += loaded.pressure_decodes.len();
    }
}

fn non_ecape_request_from_hrrr(request: &HrrrNonEcapeHourRequest) -> NonEcapeHourRequest {
    NonEcapeHourRequest {
        model: ModelId::Hrrr,
        date_yyyymmdd: request.date_yyyymmdd.clone(),
        cycle_override_utc: request.cycle_override_utc,
        forecast_hour: request.forecast_hour,
        source: request.source,
        domain: request.domain.clone(),
        out_dir: request.out_dir.clone(),
        cache_root: request.cache_root.clone(),
        use_cache: request.use_cache,
        source_mode: request.source_mode,
        direct_recipe_slugs: request.direct_recipe_slugs.clone(),
        derived_recipe_slugs: request.derived_recipe_slugs.clone(),
        direct_product_overrides: HashMap::new(),
        surface_product_override: None,
        pressure_product_override: None,
        allow_large_heavy_domain: false,
        windowed_products: request.windowed_products.clone(),
        output_width: request.output_width,
        output_height: request.output_height,
        png_compression: request.png_compression,
        place_label_overlay: request.place_label_overlay.clone(),
    }
}

fn non_ecape_multi_request_from_hrrr(
    request: &HrrrNonEcapeMultiDomainRequest,
) -> NonEcapeMultiDomainRequest {
    NonEcapeMultiDomainRequest {
        model: ModelId::Hrrr,
        date_yyyymmdd: request.date_yyyymmdd.clone(),
        cycle_override_utc: request.cycle_override_utc,
        forecast_hour: request.forecast_hour,
        source: request.source,
        domains: request.domains.clone(),
        out_dir: request.out_dir.clone(),
        cache_root: request.cache_root.clone(),
        use_cache: request.use_cache,
        source_mode: request.source_mode,
        direct_recipe_slugs: request.direct_recipe_slugs.clone(),
        derived_recipe_slugs: request.derived_recipe_slugs.clone(),
        direct_product_overrides: HashMap::new(),
        surface_product_override: None,
        pressure_product_override: None,
        allow_large_heavy_domain: false,
        windowed_products: request.windowed_products.clone(),
        output_width: request.output_width,
        output_height: request.output_height,
        png_compression: request.png_compression,
        place_label_overlay: request.place_label_overlay.clone(),
        domain_jobs: request.domain_jobs,
    }
}

fn hrrr_hour_report_from_generic(report: NonEcapeHourReport) -> HrrrNonEcapeHourReport {
    HrrrNonEcapeHourReport {
        date_yyyymmdd: report.date_yyyymmdd,
        cycle_utc: report.cycle_utc,
        forecast_hour: report.forecast_hour,
        source: report.source,
        domain: report.domain,
        out_dir: report.out_dir,
        cache_root: report.cache_root,
        use_cache: report.use_cache,
        source_mode: report.source_mode,
        publication_manifest_path: report.publication_manifest_path,
        attempt_manifest_path: report.attempt_manifest_path,
        requested: report.requested,
        shared_timing: report.shared_timing,
        summary: report.summary,
        direct: report.direct,
        derived: report.derived,
        windowed: report.windowed,
        total_ms: report.total_ms,
    }
}

fn hrrr_multi_domain_report_from_generic(
    report: NonEcapeMultiDomainReport,
) -> HrrrNonEcapeMultiDomainReport {
    HrrrNonEcapeMultiDomainReport {
        date_yyyymmdd: report.date_yyyymmdd,
        cycle_utc: report.cycle_utc,
        forecast_hour: report.forecast_hour,
        source: report.source,
        out_dir: report.out_dir,
        cache_root: report.cache_root,
        use_cache: report.use_cache,
        source_mode: report.source_mode,
        requested: report.requested,
        shared_timing: report.shared_timing,
        fanout_timing: report.fanout_timing,
        domains: report
            .domains
            .into_iter()
            .map(|domain| HrrrNonEcapeDomainReport {
                domain: domain.domain,
                publication_manifest_path: domain.publication_manifest_path,
                attempt_manifest_path: domain.attempt_manifest_path,
                summary: domain.summary,
                direct: domain.direct,
                derived: domain.derived,
                windowed: domain.windowed,
                total_ms: domain.total_ms,
            })
            .collect(),
        total_ms: report.total_ms,
    }
}

pub fn run_model_non_ecape_hour(
    request: &NonEcapeHourRequest,
) -> Result<NonEcapeHourReport, Box<dyn std::error::Error>> {
    let multi_request = NonEcapeMultiDomainRequest {
        model: request.model,
        date_yyyymmdd: request.date_yyyymmdd.clone(),
        cycle_override_utc: request.cycle_override_utc,
        forecast_hour: request.forecast_hour,
        source: request.source,
        domains: vec![request.domain.clone()],
        out_dir: request.out_dir.clone(),
        cache_root: request.cache_root.clone(),
        use_cache: request.use_cache,
        source_mode: request.source_mode,
        direct_recipe_slugs: request.direct_recipe_slugs.clone(),
        derived_recipe_slugs: request.derived_recipe_slugs.clone(),
        direct_product_overrides: request.direct_product_overrides.clone(),
        surface_product_override: request.surface_product_override.clone(),
        pressure_product_override: request.pressure_product_override.clone(),
        allow_large_heavy_domain: request.allow_large_heavy_domain,
        windowed_products: request.windowed_products.clone(),
        output_width: request.output_width,
        output_height: request.output_height,
        png_compression: request.png_compression,
        place_label_overlay: request.place_label_overlay.clone(),
        domain_jobs: None,
    };
    let report = run_model_non_ecape_hour_multi_domain(&multi_request)?;
    let domain_report = report
        .domains
        .into_iter()
        .next()
        .ok_or("multi-domain runner returned no domain reports for single-domain request")?;
    Ok(NonEcapeHourReport {
        model: report.model,
        date_yyyymmdd: report.date_yyyymmdd,
        cycle_utc: report.cycle_utc,
        forecast_hour: report.forecast_hour,
        source: report.source,
        domain: domain_report.domain,
        out_dir: report.out_dir,
        cache_root: report.cache_root,
        use_cache: report.use_cache,
        source_mode: report.source_mode,
        publication_manifest_path: domain_report.publication_manifest_path,
        attempt_manifest_path: domain_report.attempt_manifest_path,
        requested: report.requested,
        shared_timing: report.shared_timing,
        summary: domain_report.summary,
        direct: domain_report.direct,
        derived: domain_report.derived,
        windowed: domain_report.windowed,
        total_ms: domain_report.total_ms,
    })
}

pub fn run_hrrr_non_ecape_hour(
    request: &HrrrNonEcapeHourRequest,
) -> Result<HrrrNonEcapeHourReport, Box<dyn std::error::Error>> {
    let report = run_model_non_ecape_hour(&non_ecape_request_from_hrrr(request))?;
    Ok(hrrr_hour_report_from_generic(report))
}

pub fn run_model_non_ecape_hour_multi_domain(
    request: &NonEcapeMultiDomainRequest,
) -> Result<NonEcapeMultiDomainReport, Box<dyn std::error::Error>> {
    validate_requested_domains(&request.domains)?;
    let total_start = Instant::now();
    let prepared = prepare_non_ecape_hour(request)?;
    run_prepared_model_non_ecape_hour_multi_domain(request, &prepared, total_start)
}

pub fn run_model_non_ecape_hour_build(
    request: &NonEcapeMultiDomainRequest,
) -> Result<NonEcapeHourBuildReport, Box<dyn std::error::Error>> {
    validate_requested_domains(&request.domains)?;
    let total_start = Instant::now();
    let static_start = Instant::now();
    let prepared = prepare_non_ecape_hour(request)?;

    let static_report =
        run_prepared_model_non_ecape_hour_multi_domain(request, &prepared, static_start)?;
    let static_domain_timings = build_static_domain_timings(&static_report);
    let static_product_timings = build_static_product_timings(&static_report);
    Ok(NonEcapeHourBuildReport {
        static_report,
        static_domain_timings,
        static_product_timings,
        total_ms: total_start.elapsed().as_millis(),
    })
}

fn run_prepared_model_non_ecape_hour_multi_domain(
    request: &NonEcapeMultiDomainRequest,
    prepared: &PreparedNonEcapeHour,
    total_start: Instant,
) -> Result<NonEcapeMultiDomainReport, Box<dyn std::error::Error>> {
    let worker_count = domain_worker_count(request.domain_jobs, request.domains.len());
    let domain_context_build_ms = 0;
    let domain_fanout_start = Instant::now();
    let mut domain_reports = Vec::with_capacity(request.domains.len());
    if worker_count <= 1 || request.domains.len() <= 1 {
        for domain in &request.domains {
            domain_reports.push(run_prepared_non_ecape_domain(request, &prepared, domain)?);
        }
    } else {
        let queue = Arc::new(Mutex::new(
            (0..request.domains.len()).collect::<VecDeque<usize>>(),
        ));
        let (tx, rx) = mpsc::channel::<(usize, Result<NonEcapeDomainReport, String>)>();
        let mut ordered = vec![None; request.domains.len()];
        let request_ref = request;
        let prepared_ref = &prepared;
        let domains_ref = &request.domains;
        thread::scope(|scope| {
            for _ in 0..worker_count {
                let queue = Arc::clone(&queue);
                let tx = tx.clone();
                let request_ref = request_ref;
                let prepared_ref = prepared_ref;
                let domains_ref = domains_ref;
                scope.spawn(move || {
                    loop {
                        let next = {
                            let mut queue = queue.lock().expect("domain queue poisoned");
                            queue.pop_front()
                        };
                        let Some(index) = next else {
                            break;
                        };
                        let result = run_prepared_non_ecape_domain(
                            request_ref,
                            prepared_ref,
                            &domains_ref[index],
                        )
                        .map_err(|err| err.to_string());
                        if tx.send((index, result)).is_err() {
                            break;
                        }
                    }
                });
            }
            drop(tx);
            for (index, result) in rx {
                ordered[index] = Some(result);
            }
        });
        for result in ordered {
            let report = result
                .ok_or("domain worker dropped a result")?
                .map_err(|err| -> Box<dyn std::error::Error> { err.into() })?;
            domain_reports.push(report);
        }
    }
    let domain_fanout_wall_ms = domain_fanout_start.elapsed().as_millis();
    let domain_render_sum_ms = domain_reports.iter().map(|report| report.total_ms).sum();
    let domain_render_max_ms = domain_reports
        .iter()
        .map(|report| report.total_ms)
        .max()
        .unwrap_or(0);
    let conus_wall_ms = domain_reports
        .iter()
        .find(|report| report.domain.slug == "conus")
        .map(|report| report.total_ms)
        .unwrap_or(0);
    let city_domains_sum_ms = domain_reports
        .iter()
        .filter(|report| report.domain.slug != "conus")
        .map(|report| report.total_ms)
        .sum();
    let city_domains_max_ms = domain_reports
        .iter()
        .filter(|report| report.domain.slug != "conus")
        .map(|report| report.total_ms)
        .max()
        .unwrap_or(0);
    Ok(NonEcapeMultiDomainReport {
        model: request.model,
        date_yyyymmdd: prepared.latest.cycle.date_yyyymmdd.clone(),
        cycle_utc: prepared.latest.cycle.hour_utc,
        forecast_hour: request.forecast_hour,
        source: prepared.latest.source,
        out_dir: request.out_dir.clone(),
        cache_root: request.cache_root.clone(),
        use_cache: request.use_cache,
        source_mode: request.source_mode,
        requested: prepared.normalized.clone(),
        shared_timing: prepared.timing.clone(),
        fanout_timing: HrrrNonEcapeFanoutTiming {
            domain_context_build_ms,
            domain_fanout_wall_ms,
            domain_render_sum_ms,
            domain_render_max_ms,
            conus_wall_ms,
            city_domains_sum_ms,
            city_domains_max_ms,
        },
        domains: domain_reports,
        total_ms: total_start.elapsed().as_millis(),
    })
}

pub fn run_hrrr_non_ecape_hour_multi_domain(
    request: &HrrrNonEcapeMultiDomainRequest,
) -> Result<HrrrNonEcapeMultiDomainReport, Box<dyn std::error::Error>> {
    let report =
        run_model_non_ecape_hour_multi_domain(&non_ecape_multi_request_from_hrrr(request))?;
    Ok(hrrr_multi_domain_report_from_generic(report))
}

fn resolve_model_run(
    model: ModelId,
    date: &str,
    cycle_override: Option<u8>,
    forecast_hour: u16,
    source: SourceId,
) -> Result<LatestRun, Box<dyn std::error::Error>> {
    if model == ModelId::Hrrr {
        return resolve_hrrr_run(date, cycle_override, forecast_hour, source);
    }
    match cycle_override {
        Some(hour) => Ok(LatestRun {
            model,
            cycle: rustwx_core::CycleSpec::new(date, hour)?,
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

fn prepare_non_ecape_hour(
    request: &NonEcapeMultiDomainRequest,
) -> Result<PreparedNonEcapeHour, Box<dyn std::error::Error>> {
    let total_prepare_start = Instant::now();
    let normalized = normalize_requested_products_from_parts(
        request.model,
        &request.direct_recipe_slugs,
        &request.derived_recipe_slugs,
        &request.windowed_products,
    );
    validate_requested_work(request.model, &normalized)?;

    fs::create_dir_all(&request.out_dir)?;
    if request.use_cache {
        fs::create_dir_all(&request.cache_root)?;
    }

    let resolve_start = Instant::now();
    let latest = resolve_model_run(
        request.model,
        &request.date_yyyymmdd,
        request.cycle_override_utc,
        request.forecast_hour,
        request.source,
    )?;
    let resolve_run_ms = resolve_start.elapsed().as_millis();
    let pinned_date = latest.cycle.date_yyyymmdd.clone();
    let pinned_cycle = Some(latest.cycle.hour_utc);
    let pinned_source = latest.source;
    let first_domain = request
        .domains
        .first()
        .cloned()
        .ok_or("multi-domain HRRR hour runner needs at least one domain")?;
    let planning_domain = source_preparation_domain(request.model).unwrap_or(first_domain);

    let direct_groups = if normalized.direct_recipe_slugs.is_empty() {
        Vec::new()
    } else {
        let direct_request = DirectBatchRequest {
            model: request.model,
            date_yyyymmdd: pinned_date,
            cycle_override_utc: pinned_cycle,
            forecast_hour: request.forecast_hour,
            source: pinned_source,
            domain: planning_domain.clone(),
            out_dir: request.out_dir.clone(),
            cache_root: request.cache_root.clone(),
            use_cache: request.use_cache,
            recipe_slugs: normalized.direct_recipe_slugs.clone(),
            product_overrides: request.direct_product_overrides.clone(),
            contour_mode: crate::derived::NativeContourRenderMode::Automatic,
            native_fill_level_multiplier: 1,
            output_width: request.output_width,
            output_height: request.output_height,
            png_compression: request.png_compression,
            place_label_overlay: request.place_label_overlay.clone(),
            output_suffix: None,
            subtitle_left_override: None,
            subtitle_right_override: None,
            title_provenance: TitleProvenance::default(),
            geographic_overlays: None,
            panel_annotations: None,
        };
        crate::direct::plan_direct_fetch_groups(&direct_request)?
    };
    let derived_recipes = if normalized.derived_recipe_slugs.is_empty() {
        Vec::new()
    } else {
        plan_derived_recipes(&normalized.derived_recipe_slugs)?
    };
    let derived_routes = if derived_recipes.is_empty() {
        None
    } else {
        Some(plan_native_thermo_routes_with_surface_product(
            request.model,
            &derived_recipes,
            request.source_mode,
            request.surface_product_override.as_deref(),
        )?)
    };

    let derived_request = (!derived_recipes.is_empty()).then(|| DerivedBatchRequest {
        model: request.model,
        date_yyyymmdd: latest.cycle.date_yyyymmdd.clone(),
        cycle_override_utc: Some(latest.cycle.hour_utc),
        forecast_hour: request.forecast_hour,
        source: latest.source,
        domain: planning_domain.clone(),
        out_dir: request.out_dir.clone(),
        cache_root: request.cache_root.clone(),
        use_cache: request.use_cache,
        recipe_slugs: normalized.derived_recipe_slugs.clone(),
        surface_product_override: request.surface_product_override.clone(),
        pressure_product_override: request.pressure_product_override.clone(),
        source_mode: request.source_mode,
        allow_large_heavy_domain: request.allow_large_heavy_domain,
        contour_mode: non_ecape_derived_contour_mode(),
        native_fill_level_multiplier: non_ecape_native_fill_level_multiplier(),
        output_width: request.output_width,
        output_height: request.output_height,
        png_compression: request.png_compression,
        place_label_overlay: request.place_label_overlay.clone(),
        output_suffix: None,
        subtitle_left_override: None,
        subtitle_right_override: None,
        title_provenance: TitleProvenance::default(),
    });

    let mut shared_load_decode_ms = 0u128;
    let mut shared_loader_profile = SharedLoaderProfile::default();
    let mut derived_loaded_override: Option<Arc<crate::runtime::LoadedBundleSet>> = None;
    let mut derived_lane_blocker = None::<String>;
    if let (Some(derived_request), Some(routes)) =
        (derived_request.as_ref(), derived_routes.as_ref())
    {
        let special_load_start = Instant::now();
        match maybe_load_special_pair_for_derived(derived_request, &latest, routes) {
            Ok(Some(loaded)) => {
                shared_load_decode_ms += special_load_start.elapsed().as_millis();
                shared_loader_profile.record(&loaded);
                derived_loaded_override = Some(Arc::new(loaded));
            }
            Ok(None) => {}
            Err(err) => {
                shared_load_decode_ms += special_load_start.elapsed().as_millis();
                derived_lane_blocker = Some(format!("derived shared load failed: {err}"));
            }
        }
    }

    let mut main_loaded: Option<Arc<crate::runtime::LoadedBundleSet>> = None;
    let mut direct_loaded: Option<Arc<crate::runtime::LoadedBundleSet>> = None;

    let derived_routes_for_plan = if derived_lane_blocker.is_some() {
        None
    } else {
        derived_routes.as_ref()
    };
    let build_shared_loaded = request.model == ModelId::Hrrr
        || (derived_loaded_override.is_none() && derived_lane_blocker.is_none());
    if build_shared_loaded {
        let plan = build_shared_non_ecape_execution_plan(
            &latest,
            request.forecast_hour,
            &direct_groups,
            derived_routes_for_plan,
            derived_loaded_override.is_none(),
            request.surface_product_override.as_deref(),
            request.pressure_product_override.as_deref(),
        );
        let load_start = Instant::now();
        main_loaded = if plan.bundles.is_empty() {
            None
        } else {
            match load_execution_plan(
                plan,
                &BundleLoaderConfig {
                    cache_root: request.cache_root.clone(),
                    use_cache: request.use_cache,
                },
            ) {
                Ok(loaded) => Some(Arc::new(loaded)),
                Err(err) if derived_routes_for_plan.is_some() => {
                    derived_lane_blocker =
                        Some(format!("derived/shared execution plan failed: {err}"));
                    let direct_plan = build_shared_non_ecape_execution_plan(
                        &latest,
                        request.forecast_hour,
                        &direct_groups,
                        None,
                        false,
                        request.surface_product_override.as_deref(),
                        request.pressure_product_override.as_deref(),
                    );
                    if direct_plan.bundles.is_empty() {
                        None
                    } else {
                        Some(Arc::new(load_execution_plan(
                            direct_plan,
                            &BundleLoaderConfig {
                                cache_root: request.cache_root.clone(),
                                use_cache: request.use_cache,
                            },
                        )?))
                    }
                }
                Err(err) => return Err(err),
            }
        };
        shared_load_decode_ms += load_start.elapsed().as_millis();
        if let Some(loaded) = main_loaded.as_ref() {
            shared_loader_profile.record(loaded);
        }
        direct_loaded = main_loaded.clone();
    } else {
        if !direct_groups.is_empty() {
            let mut direct_plan_builder = ExecutionPlanBuilder::new(&latest, request.forecast_hour);
            for group in &direct_groups {
                let requirement = rustwx_core::BundleRequirement::new(
                    rustwx_core::CanonicalBundleDescriptor::NativeAnalysis,
                    request.forecast_hour,
                )
                .with_native_override(group.product.clone());
                for alias in &group.planned_family_aliases {
                    if should_attach_direct_idx_patterns(latest.source) {
                        direct_plan_builder.require_with_logical_family_and_patterns(
                            &requirement,
                            Some(alias),
                            group.variable_patterns.clone(),
                        );
                    } else {
                        direct_plan_builder.require_with_logical_family(&requirement, Some(alias));
                    }
                }
            }
            let direct_plan = direct_plan_builder.build();
            let direct_load_start = Instant::now();
            direct_loaded = if direct_plan.bundles.is_empty() {
                None
            } else {
                Some(Arc::new(load_execution_plan(
                    direct_plan,
                    &BundleLoaderConfig {
                        cache_root: request.cache_root.clone(),
                        use_cache: request.use_cache,
                    },
                )?))
            };
            shared_load_decode_ms += direct_load_start.elapsed().as_millis();
            if let Some(loaded) = direct_loaded.as_ref() {
                shared_loader_profile.record(loaded);
            }
        }
    }

    let derived_loaded = if derived_recipes.is_empty() {
        None
    } else if let Some(loaded) = derived_loaded_override {
        Some(loaded)
    } else if derived_lane_blocker.is_some() {
        None
    } else {
        main_loaded.clone()
    };

    let shared_direct_prepare_start = Instant::now();
    let precomputed_direct =
        if normalized.direct_recipe_slugs.is_empty() || request.domains.len() <= 1 {
            None
        } else if let Some(direct_loaded_ref) = direct_loaded.as_ref() {
            let direct_request = DirectBatchRequest {
                model: request.model,
                date_yyyymmdd: latest.cycle.date_yyyymmdd.clone(),
                cycle_override_utc: Some(latest.cycle.hour_utc),
                forecast_hour: request.forecast_hour,
                source: latest.source,
                domain: planning_domain.clone(),
                out_dir: request.out_dir.clone(),
                cache_root: request.cache_root.clone(),
                use_cache: request.use_cache,
                recipe_slugs: normalized.direct_recipe_slugs.clone(),
                product_overrides: request.direct_product_overrides.clone(),
                contour_mode: non_ecape_derived_contour_mode(),
                native_fill_level_multiplier: non_ecape_native_fill_level_multiplier(),
                output_width: request.output_width,
                output_height: request.output_height,
                png_compression: request.png_compression,
                place_label_overlay: request.place_label_overlay.clone(),
                output_suffix: None,
                subtitle_left_override: None,
                subtitle_right_override: None,
                title_provenance: TitleProvenance::default(),
                geographic_overlays: None,
                panel_annotations: None,
            };
            Some(Arc::new(prepare_direct_batch_from_loaded(
                &direct_request,
                direct_loaded_ref,
                &request.cache_root,
                request.use_cache,
            )?))
        } else {
            None
        };
    let shared_direct_prepare_ms = shared_direct_prepare_start.elapsed().as_millis();

    let shared_derived_prepare_start = Instant::now();
    let precomputed_derived = if derived_recipes.is_empty() || request.domains.len() <= 1 {
        None
    } else if let (Some(derived_request), Some(derived_loaded_ref)) =
        (derived_request.as_ref(), derived_loaded.as_ref())
    {
        prepare_shared_derived_fields(derived_request, &derived_recipes, derived_loaded_ref)?
    } else {
        None
    };
    let shared_derived_prepare_ms = shared_derived_prepare_start.elapsed().as_millis();

    let shared_windowed_prepare_start = Instant::now();
    let precomputed_windowed =
        if normalized.windowed_products.is_empty() || request.domains.len() <= 1 {
            None
        } else {
            let windowed_request = HrrrWindowedBatchRequest {
                model: request.model,
                date_yyyymmdd: latest.cycle.date_yyyymmdd.clone(),
                cycle_override_utc: Some(latest.cycle.hour_utc),
                forecast_hour: request.forecast_hour,
                source: latest.source,
                domain: planning_domain.clone(),
                out_dir: request.out_dir.clone(),
                cache_root: request.cache_root.clone(),
                use_cache: request.use_cache,
                products: normalized.windowed_products.clone(),
                output_width: request.output_width,
                output_height: request.output_height,
                png_compression: request.png_compression,
                place_label_overlay: None,
                subtitle_left_suffix: None,
                subtitle_right_override: None,
                title_provenance: TitleProvenance::default(),
            };
            Some(Arc::new(prepare_hrrr_windowed_batch_with_context(
                &windowed_request,
                &latest,
            )?))
        };
    let shared_windowed_prepare_ms = shared_windowed_prepare_start.elapsed().as_millis();

    Ok(PreparedNonEcapeHour {
        normalized,
        latest,
        derived_recipes,
        precomputed_direct,
        precomputed_derived,
        precomputed_windowed,
        direct_loaded,
        derived_loaded,
        derived_lane_blocker,
        timing: HrrrNonEcapeSharedTiming {
            resolve_run_ms,
            shared_load_decode_ms,
            shared_fetch_ms_total: shared_loader_profile.fetch_ms_total,
            shared_decode_surface_ms_total: shared_loader_profile.decode_surface_ms_total,
            shared_decode_pressure_ms_total: shared_loader_profile.decode_pressure_ms_total,
            shared_fetched_bundle_count: shared_loader_profile.fetched_bundle_count,
            shared_surface_decode_count: shared_loader_profile.surface_decode_count,
            shared_pressure_decode_count: shared_loader_profile.pressure_decode_count,
            shared_direct_prepare_ms,
            shared_derived_prepare_ms,
            shared_windowed_prepare_ms,
            total_prepare_ms: total_prepare_start.elapsed().as_millis(),
        },
    })
}

fn build_shared_non_ecape_execution_plan(
    latest: &LatestRun,
    forecast_hour: u16,
    direct_groups: &[FetchGroup],
    derived_routes: Option<&PlannedDerivedSourceRoutes>,
    include_pair_compute: bool,
    surface_product_override: Option<&str>,
    pressure_product_override: Option<&str>,
) -> crate::planner::ExecutionPlan {
    let mut plan_builder = ExecutionPlanBuilder::new(latest, forecast_hour);
    if include_pair_compute
        && derived_routes
            .map(|routes| derived_compute_recipes_need_pressure(&routes.compute_recipes))
            .unwrap_or(false)
    {
        add_pair_requirements(
            &mut plan_builder,
            latest,
            forecast_hour,
            surface_product_override,
            pressure_product_override,
        );
    } else if include_pair_compute
        && derived_routes
            .map(|routes| !routes.compute_recipes.is_empty())
            .unwrap_or(false)
    {
        add_surface_requirement(
            &mut plan_builder,
            latest,
            forecast_hour,
            surface_product_override,
        );
    }
    if let Some(routes) = derived_routes {
        add_native_route_requirements(&mut plan_builder, forecast_hour, routes);
    }
    add_direct_fetch_group_requirements(
        &mut plan_builder,
        latest.source,
        forecast_hour,
        direct_groups,
    );
    plan_builder.build()
}

fn should_attach_direct_idx_patterns(source: SourceId) -> bool {
    matches!(source, SourceId::Aws | SourceId::Google)
}

fn add_pair_requirements(
    plan_builder: &mut ExecutionPlanBuilder,
    latest: &LatestRun,
    forecast_hour: u16,
    surface_product_override: Option<&str>,
    pressure_product_override: Option<&str>,
) {
    let pair_plan = build_severe_execution_plan(
        latest,
        forecast_hour,
        surface_product_override,
        pressure_product_override,
    );
    for bundle in &pair_plan.bundles {
        for alias in &bundle.aliases {
            let mut requirement =
                rustwx_core::BundleRequirement::new(alias.bundle, bundle.id.forecast_hour);
            if let Some(ref over) = alias.native_override {
                requirement = requirement.with_native_override(over.clone());
            }
            plan_builder.require_with_logical_family(&requirement, alias.logical_family.as_deref());
        }
    }
}

fn add_native_route_requirements(
    plan_builder: &mut ExecutionPlanBuilder,
    forecast_hour: u16,
    routes: &PlannedDerivedSourceRoutes,
) {
    let mut native_products = std::collections::BTreeSet::<String>::new();
    for route in &routes.native_routes {
        if native_products.insert(route.candidate.fetch_product.to_string()) {
            let requirement =
                BundleRequirement::new(CanonicalBundleDescriptor::NativeAnalysis, forecast_hour)
                    .with_native_override(route.candidate.fetch_product);
            plan_builder.require_with_logical_family(
                &requirement,
                Some(&format!("thermo-native:{}", route.candidate.fetch_product)),
            );
        }
    }
}

fn add_surface_requirement(
    plan_builder: &mut ExecutionPlanBuilder,
    latest: &LatestRun,
    forecast_hour: u16,
    surface_product_override: Option<&str>,
) {
    let native_product = rustwx_models::resolve_canonical_bundle_product(
        latest.model,
        CanonicalBundleDescriptor::SurfaceAnalysis,
        surface_product_override,
    )
    .native_product;
    let requirement =
        BundleRequirement::new(CanonicalBundleDescriptor::SurfaceAnalysis, forecast_hour)
            .with_native_override(native_product);
    plan_builder.require_with_logical_family(&requirement, Some("sfc"));
}

fn add_direct_fetch_group_requirements(
    plan_builder: &mut ExecutionPlanBuilder,
    source: SourceId,
    forecast_hour: u16,
    direct_groups: &[FetchGroup],
) {
    for group in direct_groups {
        let requirement =
            BundleRequirement::new(CanonicalBundleDescriptor::NativeAnalysis, forecast_hour)
                .with_native_override(group.product.clone());
        for alias in &group.planned_family_aliases {
            if should_attach_direct_idx_patterns(source) {
                plan_builder.require_with_logical_family_and_patterns(
                    &requirement,
                    Some(alias),
                    group.variable_patterns.clone(),
                );
            } else {
                plan_builder.require_with_logical_family(&requirement, Some(alias));
            }
        }
    }
}

fn run_prepared_non_ecape_domain(
    request: &NonEcapeMultiDomainRequest,
    prepared: &PreparedNonEcapeHour,
    domain: &DomainSpec,
) -> Result<NonEcapeDomainReport, Box<dyn std::error::Error>> {
    let total_start = Instant::now();
    let domain_out_dir = request.out_dir.join(&domain.slug);
    fs::create_dir_all(&domain_out_dir)?;
    let pinned_date = prepared.latest.cycle.date_yyyymmdd.clone();
    let pinned_cycle = Some(prepared.latest.cycle.hour_utc);
    let pinned_source = prepared.latest.source;
    let pinned_cycle_utc = prepared.latest.cycle.hour_utc;
    let run_slug = format!(
        "rustwx_{}_{}_{}z_f{:03}_{}_non_ecape_hour",
        request.model.as_str().replace('-', "_"),
        pinned_date,
        pinned_cycle_utc,
        request.forecast_hour,
        domain.slug
    );
    let manifest_path = default_run_manifest_path(&domain_out_dir, &run_slug);
    let mut manifest = build_run_manifest(
        request.model,
        &prepared.normalized,
        &domain_out_dir,
        &run_slug,
        &pinned_date,
        pinned_cycle_utc,
        request.forecast_hour,
        pinned_source,
        &domain.slug,
    );
    manifest.mark_running();
    crate::publication::publish_run_manifest(&manifest_path, &manifest)?;

    let direct_loaded_ref = prepared.direct_loaded.as_deref();
    let derived_loaded_ref = prepared.derived_loaded.as_deref();
    let direct_request =
        (!prepared.normalized.direct_recipe_slugs.is_empty()).then(|| DirectBatchRequest {
            model: request.model,
            date_yyyymmdd: pinned_date.clone(),
            cycle_override_utc: pinned_cycle,
            forecast_hour: request.forecast_hour,
            source: pinned_source,
            domain: domain.clone(),
            out_dir: domain_out_dir.clone(),
            cache_root: request.cache_root.clone(),
            use_cache: request.use_cache,
            recipe_slugs: prepared.normalized.direct_recipe_slugs.clone(),
            product_overrides: request.direct_product_overrides.clone(),
            contour_mode: crate::derived::NativeContourRenderMode::Automatic,
            native_fill_level_multiplier: 1,
            output_width: request.output_width,
            output_height: request.output_height,
            png_compression: request.png_compression,
            place_label_overlay: request.place_label_overlay.clone(),
            output_suffix: None,
            subtitle_left_override: None,
            subtitle_right_override: None,
            title_provenance: TitleProvenance::default(),
            geographic_overlays: None,
            panel_annotations: None,
        });

    let derived_request = (!prepared.normalized.derived_recipe_slugs.is_empty()).then(|| {
        (
            DerivedBatchRequest {
                model: request.model,
                date_yyyymmdd: pinned_date.clone(),
                cycle_override_utc: pinned_cycle,
                forecast_hour: request.forecast_hour,
                source: pinned_source,
                domain: domain.clone(),
                out_dir: domain_out_dir.clone(),
                cache_root: request.cache_root.clone(),
                use_cache: request.use_cache,
                recipe_slugs: prepared.normalized.derived_recipe_slugs.clone(),
                surface_product_override: request.surface_product_override.clone(),
                pressure_product_override: request.pressure_product_override.clone(),
                source_mode: request.source_mode,
                allow_large_heavy_domain: request.allow_large_heavy_domain,
                contour_mode: non_ecape_derived_contour_mode(),
                native_fill_level_multiplier: non_ecape_native_fill_level_multiplier(),
                output_width: request.output_width,
                output_height: request.output_height,
                png_compression: request.png_compression,
                place_label_overlay: request.place_label_overlay.clone(),
                output_suffix: None,
                subtitle_left_override: None,
                subtitle_right_override: None,
                title_provenance: TitleProvenance::default(),
            },
            prepared.derived_recipes.clone(),
        )
    });
    let derived_latest = prepared.latest.clone();
    let precomputed_derived = prepared.precomputed_derived.as_ref();
    let derived_lane_blocker = prepared.derived_lane_blocker.as_deref();

    let windowed_request =
        (!prepared.normalized.windowed_products.is_empty()).then(|| HrrrWindowedBatchRequest {
            model: request.model,
            date_yyyymmdd: pinned_date.clone(),
            cycle_override_utc: pinned_cycle,
            forecast_hour: request.forecast_hour,
            source: pinned_source,
            domain: domain.clone(),
            out_dir: domain_out_dir.clone(),
            cache_root: request.cache_root.clone(),
            use_cache: request.use_cache,
            products: prepared.normalized.windowed_products.clone(),
            output_width: request.output_width,
            output_height: request.output_height,
            png_compression: request.png_compression,
            place_label_overlay: request.place_label_overlay.clone(),
            subtitle_left_suffix: None,
            subtitle_right_override: None,
            title_provenance: TitleProvenance::default(),
        });

    let lane_result = run_fanout3(
        should_run_prepared_lanes_concurrently(request, prepared, pinned_source),
        direct_request.as_ref().map(|lane_request| {
            lane("direct", move || {
                if let Some(precomputed) = prepared.precomputed_direct.as_deref() {
                    run_direct_batch_from_prepared(lane_request, precomputed, None)
                } else {
                    run_direct_batch_from_loaded(
                        lane_request,
                        direct_loaded_ref
                            .expect("planner must load bundles when direct is requested"),
                        &lane_request.cache_root,
                        lane_request.use_cache,
                        None,
                    )
                }
            })
        }),
        derived_request.as_ref().map(|(lane_request, recipes)| {
            lane("derived", move || {
                let report = if let Some(reason) = derived_lane_blocker {
                    derived_lane_failure_report(lane_request, recipes, &derived_latest, reason)
                } else if let Some(loaded) = derived_loaded_ref {
                    if let Some(precomputed) = precomputed_derived {
                        run_model_derived_batch_from_loaded_with_precomputed(
                            lane_request,
                            recipes,
                            loaded,
                            precomputed,
                        )
                    } else {
                        run_model_derived_batch_from_loaded(lane_request, recipes, loaded)
                    }
                } else {
                    run_model_derived_batch_without_loaded(lane_request, recipes, &derived_latest)
                };
                match report {
                    Ok(report) => Ok(report),
                    Err(err) => derived_lane_failure_report(
                        lane_request,
                        recipes,
                        &derived_latest,
                        &format!("derived lane failed: {err}"),
                    ),
                }
            })
        }),
        windowed_request.as_ref().map(|lane_request| {
            let windowed_latest = prepared.latest.clone();
            lane("windowed", move || {
                if let Some(precomputed) = prepared.precomputed_windowed.as_deref() {
                    run_hrrr_windowed_batch_from_prepared(lane_request, precomputed)
                } else {
                    run_hrrr_windowed_batch_with_context(lane_request, &windowed_latest)
                }
            })
        }),
    );

    let (direct, derived, windowed) = match lane_result {
        Ok(reports) => reports,
        Err(err) => {
            manifest.mark_failed(err.to_string());
            let _ = publish_run_manifest_with_attempt(
                &manifest_path,
                &domain_out_dir,
                &run_slug,
                &manifest,
            );
            return Err(err);
        }
    };

    let summary = build_summary(&direct, &derived, &windowed);
    manifest.input_fetches = collect_input_fetches(&direct, &derived, &windowed);
    apply_direct_manifest_updates(&mut manifest, &direct);
    apply_derived_manifest_updates(&mut manifest, &derived);
    apply_windowed_manifest_updates(&mut manifest, &windowed);
    let (canonical_manifest_path, attempt_manifest_path) =
        finalize_and_publish_run_manifest(&mut manifest, &domain_out_dir, &run_slug)?;

    Ok(NonEcapeDomainReport {
        domain: domain.clone(),
        publication_manifest_path: canonical_manifest_path,
        attempt_manifest_path: Some(attempt_manifest_path),
        summary,
        direct,
        derived,
        windowed,
        total_ms: total_start.elapsed().as_millis(),
    })
}

fn derived_lane_failure_report(
    request: &DerivedBatchRequest,
    recipes: &[crate::derived::DerivedRecipe],
    latest: &LatestRun,
    reason: &str,
) -> Result<HrrrDerivedBatchReport, Box<dyn std::error::Error>> {
    let mut report = run_model_derived_batch_without_loaded(request, recipes, latest)?;
    let existing = report
        .blockers
        .iter()
        .map(|blocker| blocker.recipe_slug.clone())
        .collect::<HashSet<_>>();
    let source_route = match request.source_mode {
        ProductSourceMode::Canonical => ProductSourceRoute::CanonicalDerived,
        ProductSourceMode::Fastest => ProductSourceRoute::BlockedNoFastRoute,
    };
    report.blockers.extend(
        request
            .recipe_slugs
            .iter()
            .filter(|slug| !existing.contains(*slug))
            .map(|slug| DerivedRecipeBlocker {
                recipe_slug: slug.clone(),
                source_route,
                reason: reason.to_string(),
            }),
    );
    Ok(report)
}

fn validate_requested_work(
    model: ModelId,
    request: &NonEcapeRequestedProducts,
) -> Result<(), Box<dyn std::error::Error>> {
    if request.direct_recipe_slugs.is_empty()
        && request.derived_recipe_slugs.is_empty()
        && request.windowed_products.is_empty()
    {
        return Err(
            "unified non-ECAPE hour runner needs at least one direct recipe, derived recipe, or windowed product"
                .into(),
        );
    }
    if !windowed_products_supported_for_model(model, &request.windowed_products) {
        return Err(format!(
            "requested windowed products are not supported by model {}; HRRR supports the full windowed family, while the v0.5 cross-model GRIB path supports qpf_total only",
            model,
        )
        .into());
    }
    if let Some(heavy_slug) = request
        .derived_recipe_slugs
        .iter()
        .find(|slug| is_heavy_derived_recipe_slug(slug))
    {
        return Err(format!(
            "derived recipe '{}' is a heavy ECAPE product; use derived_batch or a heavy runner instead of non_ecape_hour",
            heavy_slug
        )
        .into());
    }
    Ok(())
}

fn windowed_products_supported_for_model(model: ModelId, products: &[HrrrWindowedProduct]) -> bool {
    if products.is_empty() {
        return true;
    }
    if model == ModelId::Hrrr {
        return true;
    }
    products
        .iter()
        .all(|product| matches!(product, HrrrWindowedProduct::QpfTotal))
        && matches!(
            model,
            ModelId::HrrrAk
                | ModelId::Gfs
                | ModelId::Gdas
                | ModelId::Gefs
                | ModelId::Aigfs
                | ModelId::Aigefs
                | ModelId::Rap
                | ModelId::Nam
                | ModelId::Hiresw
                | ModelId::Sref
                | ModelId::Nbm
                | ModelId::RrfsA
        )
}

fn validate_requested_domains(domains: &[DomainSpec]) -> Result<(), Box<dyn std::error::Error>> {
    if domains.is_empty() {
        return Err("multi-domain non-ECAPE hour runner needs at least one domain".into());
    }
    let mut seen = HashSet::<&str>::new();
    for domain in domains {
        if !seen.insert(domain.slug.as_str()) {
            return Err(format!("duplicate multi-domain slug '{}'", domain.slug).into());
        }
    }
    Ok(())
}

fn source_preparation_domain(model: ModelId) -> Option<DomainSpec> {
    match model {
        ModelId::Hrrr
        | ModelId::Rap
        | ModelId::RrfsA
        | ModelId::RrfsPublic
        | ModelId::RrfsFireWx
        | ModelId::Nam
        | ModelId::Hiresw
        | ModelId::Nbm => Some(DomainSpec::new("conus_source", (-127.0, -66.0, 23.0, 51.5))),
        ModelId::HrrrAk => Some(DomainSpec::new(
            "alaska_source",
            (-180.0, -100.0, 40.0, 75.0),
        )),
        ModelId::Gfs
        | ModelId::Gdas
        | ModelId::Gefs
        | ModelId::Aigfs
        | ModelId::Aigefs
        | ModelId::EcmwfOpenData
        | ModelId::Aifs => Some(DomainSpec::new(
            "global_source",
            (-180.0, 179.999, -90.0, 90.0),
        )),
        _ => None,
    }
}

#[cfg(test)]
fn normalize_requested_products(
    request: &HrrrNonEcapeHourRequest,
) -> HrrrNonEcapeHourRequestedProducts {
    normalize_requested_products_from_parts(
        ModelId::Hrrr,
        &request.direct_recipe_slugs,
        &request.derived_recipe_slugs,
        &request.windowed_products,
    )
}

fn normalize_requested_products_from_parts(
    model: ModelId,
    direct_recipe_slugs: &[String],
    derived_recipe_slugs: &[String],
    windowed_products: &[HrrrWindowedProduct],
) -> HrrrNonEcapeHourRequestedProducts {
    let mut normalized_direct_recipe_slugs = Vec::new();
    let mut normalized_windowed_products = windowed_products.to_vec();

    for slug in direct_recipe_slugs {
        let normalized_slug = plot_recipe(slug)
            .map(|recipe| recipe.slug)
            .unwrap_or(slug.as_str());
        if model == ModelId::Hrrr && normalized_slug == "1h_qpf" {
            if !normalized_windowed_products.contains(&HrrrWindowedProduct::Qpf1h) {
                normalized_windowed_products.push(HrrrWindowedProduct::Qpf1h);
            }
            continue;
        }
        normalized_direct_recipe_slugs.push(slug.clone());
    }

    HrrrNonEcapeHourRequestedProducts {
        direct_recipe_slugs: normalized_direct_recipe_slugs,
        derived_recipe_slugs: derived_recipe_slugs.to_vec(),
        windowed_products: normalized_windowed_products,
    }
}

fn should_run_lanes_concurrently(model: ModelId, source: SourceId) -> bool {
    matches!(model, ModelId::Hrrr | ModelId::RrfsA | ModelId::WrfGdex)
        && !matches!(source, SourceId::Nomads)
}

fn should_run_prepared_lanes_concurrently(
    request: &NonEcapeMultiDomainRequest,
    prepared: &PreparedNonEcapeHour,
    source: SourceId,
) -> bool {
    if let Some(enabled) = env_bool("RUSTWX_PREPARED_LANE_CONCURRENCY") {
        return enabled;
    }
    if request.domains.len() > 1
        && (prepared.precomputed_direct.is_some()
            || prepared.precomputed_derived.is_some()
            || prepared.precomputed_windowed.is_some())
    {
        return true;
    }
    should_run_lanes_concurrently(request.model, source)
}

fn env_bool(name: &str) -> Option<bool> {
    let value = std::env::var(name).ok()?;
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

fn domain_worker_count(requested_jobs: Option<usize>, domain_count: usize) -> usize {
    if domain_count <= 1 {
        return 1;
    }

    let env_override = std::env::var("RUSTWX_DOMAIN_JOBS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0);
    let requested = requested_jobs.or(env_override).filter(|&value| value > 0);
    let default_jobs = 1;
    requested.unwrap_or(default_jobs).clamp(1, domain_count)
}

struct FanoutLane<'scope, T> {
    name: &'static str,
    job: Box<dyn FnOnce() -> Result<T, Box<dyn std::error::Error>> + Send + 'scope>,
}

impl<'scope, T> FanoutLane<'scope, T> {
    fn new<F>(name: &'static str, job: F) -> Self
    where
        F: FnOnce() -> Result<T, Box<dyn std::error::Error>> + Send + 'scope,
    {
        Self {
            name,
            job: Box::new(job),
        }
    }

    fn run(self) -> Result<T, Box<dyn std::error::Error>> {
        (self.job)()
    }
}

fn lane<'scope, T, F>(name: &'static str, job: F) -> FanoutLane<'scope, T>
where
    F: FnOnce() -> Result<T, Box<dyn std::error::Error>> + Send + 'scope,
{
    FanoutLane::new(name, job)
}

fn run_fanout3<'scope, A, B, C>(
    concurrent: bool,
    first: Option<FanoutLane<'scope, A>>,
    second: Option<FanoutLane<'scope, B>>,
    third: Option<FanoutLane<'scope, C>>,
) -> Result<(Option<A>, Option<B>, Option<C>), Box<dyn std::error::Error>>
where
    A: Send + 'scope,
    B: Send + 'scope,
    C: Send + 'scope,
{
    if concurrent {
        thread::scope(|scope| {
            let first_handle = first.map(|lane| {
                let name = lane.name;
                (
                    name,
                    scope.spawn(move || lane.run().map_err(|err| lane_error(name, err))),
                )
            });
            let second_handle = second.map(|lane| {
                let name = lane.name;
                (
                    name,
                    scope.spawn(move || lane.run().map_err(|err| lane_error(name, err))),
                )
            });
            let third_handle = third.map(|lane| {
                let name = lane.name;
                (
                    name,
                    scope.spawn(move || lane.run().map_err(|err| lane_error(name, err))),
                )
            });

            let first = first_handle
                .map(|(name, handle)| join_lane(name, handle))
                .transpose()?;
            let second = second_handle
                .map(|(name, handle)| join_lane(name, handle))
                .transpose()?;
            let third = third_handle
                .map(|(name, handle)| join_lane(name, handle))
                .transpose()?;

            Ok::<_, Box<dyn std::error::Error>>((first, second, third))
        })
    } else {
        Ok((
            first.map(FanoutLane::run).transpose()?,
            second.map(FanoutLane::run).transpose()?,
            third.map(FanoutLane::run).transpose()?,
        ))
    }
}

fn join_lane<T>(
    name: &'static str,
    handle: thread::ScopedJoinHandle<'_, std::io::Result<T>>,
) -> Result<T, Box<dyn std::error::Error>> {
    match handle.join() {
        Ok(result) => result.map_err(Box::<dyn std::error::Error>::from),
        Err(panic) => Err(Box::new(std::io::Error::other(format!(
            "{name} lane panicked: {}",
            panic_message(panic)
        )))),
    }
}

fn lane_error(name: &'static str, err: Box<dyn std::error::Error>) -> std::io::Error {
    std::io::Error::other(format!("{name} lane failed: {err}"))
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
mod tests;
