use super::fetch::{extract_direct_fetch_group_from_loaded, find_loaded_bytes_for_group};
use super::planning::{
    build_direct_execution_plan, group_direct_fetches, partition_recipes_by_selector_availability,
    plan_direct_recipes, recipe_slugs_depending_on_group,
};
use super::render_direct_recipes;
use super::types::{
    DirectBatchReport, DirectBatchRequest, DirectFetchRuntimeInfo, DirectRecipeBlocker,
    HrrrDirectBatchReport, HrrrDirectBatchRequest, PreparedDirectBatch,
};
use crate::runtime::{BundleLoaderConfig, LoadedBundleSet, load_execution_plan};
use crate::shared_context::ProjectedMapProvider;
use rustwx_core::{CycleSpec, FieldSelector, ModelId, SelectedField2D, SourceId};
use rustwx_models::{LatestRun, latest_available_run_at_forecast_hour};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::time::Instant;

fn resolve_direct_run(
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

pub fn run_direct_batch(
    request: &DirectBatchRequest,
) -> Result<DirectBatchReport, Box<dyn std::error::Error>> {
    let latest = resolve_direct_run(
        request.model,
        &request.date_yyyymmdd,
        request.cycle_override_utc,
        request.forecast_hour,
        request.source,
    )?;
    run_direct_batch_with_context(request, &latest, None)
}

pub fn run_hrrr_direct_batch(
    request: &HrrrDirectBatchRequest,
) -> Result<HrrrDirectBatchReport, Box<dyn std::error::Error>> {
    run_direct_batch(&DirectBatchRequest::from_hrrr(request))
}

pub(crate) fn run_direct_batch_from_loaded(
    request: &DirectBatchRequest,
    loaded: &LoadedBundleSet,
    cache_root: &std::path::Path,
    use_cache: bool,
    shared_context: Option<&dyn ProjectedMapProvider>,
) -> Result<DirectBatchReport, Box<dyn std::error::Error>> {
    let total_start = Instant::now();
    let prepared = prepare_direct_batch_from_loaded(request, loaded, cache_root, use_cache)?;
    run_direct_batch_from_prepared_with_total_start(request, &prepared, shared_context, total_start)
}

pub(crate) fn prepare_direct_batch_from_loaded(
    request: &DirectBatchRequest,
    loaded: &LoadedBundleSet,
    cache_root: &std::path::Path,
    use_cache: bool,
) -> Result<PreparedDirectBatch, Box<dyn std::error::Error>> {
    fs::create_dir_all(&request.out_dir)?;
    if use_cache {
        fs::create_dir_all(cache_root)?;
    }
    let planned = plan_direct_recipes(request.model, &request.recipe_slugs)?;
    let groups = group_direct_fetches(request, &planned);
    let mut extracted = HashMap::<FieldSelector, SelectedField2D>::new();
    let mut fetches = Vec::with_capacity(groups.len());
    let mut fetch_truth_by_actual_product = HashMap::<String, DirectFetchRuntimeInfo>::new();
    let mut missing_selectors = HashSet::<FieldSelector>::new();
    let mut blockers = Vec::<DirectRecipeBlocker>::new();

    for group in &groups {
        let fetched = match find_loaded_bytes_for_group(loaded, group) {
            Ok(bytes) => bytes,
            Err(err) => {
                let reason = err.to_string();
                for selector in &group.selectors {
                    missing_selectors.insert(*selector);
                }
                for recipe_slug in recipe_slugs_depending_on_group(&planned, group) {
                    blockers.push(DirectRecipeBlocker {
                        recipe_slug,
                        reason: reason.clone(),
                    });
                }
                continue;
            }
        };
        let (fields, unmatched, timing) =
            match extract_direct_fetch_group_from_loaded(request, group, fetched, use_cache) {
                Ok(result) => result,
                Err(err) => {
                    let reason = err.to_string();
                    for selector in &group.selectors {
                        missing_selectors.insert(*selector);
                    }
                    for recipe_slug in recipe_slugs_depending_on_group(&planned, group) {
                        blockers.push(DirectRecipeBlocker {
                            recipe_slug,
                            reason: reason.clone(),
                        });
                    }
                    continue;
                }
            };
        extracted.extend(fields.into_iter().map(|field| (field.selector, field)));
        for selector in unmatched {
            missing_selectors.insert(selector);
        }
        fetch_truth_by_actual_product.insert(group.product.clone(), timing.runtime_fetch.clone());
        fetches.push(timing);
    }

    let (renderable, selector_blockers) =
        partition_recipes_by_selector_availability(&planned, &missing_selectors);
    blockers.extend(selector_blockers);

    Ok(PreparedDirectBatch {
        latest: loaded.latest.clone(),
        renderable,
        extracted,
        fetches,
        fetch_truth_by_actual_product,
        blockers,
    })
}

pub(crate) fn run_direct_batch_from_prepared(
    request: &DirectBatchRequest,
    prepared: &PreparedDirectBatch,
    shared_context: Option<&dyn ProjectedMapProvider>,
) -> Result<DirectBatchReport, Box<dyn std::error::Error>> {
    run_direct_batch_from_prepared_with_total_start(
        request,
        prepared,
        shared_context,
        Instant::now(),
    )
}

fn run_direct_batch_from_prepared_with_total_start(
    request: &DirectBatchRequest,
    prepared: &PreparedDirectBatch,
    shared_context: Option<&dyn ProjectedMapProvider>,
    total_start: Instant,
) -> Result<DirectBatchReport, Box<dyn std::error::Error>> {
    fs::create_dir_all(&request.out_dir)?;

    let rendered = render_direct_recipes(
        request,
        &prepared.latest,
        &prepared.renderable,
        &prepared.extracted,
        &prepared.fetch_truth_by_actual_product,
        shared_context,
    )?;

    Ok(DirectBatchReport {
        model: request.model,
        date_yyyymmdd: request.date_yyyymmdd.clone(),
        cycle_utc: prepared.latest.cycle.hour_utc,
        forecast_hour: request.forecast_hour,
        source: prepared.latest.source,
        domain: request.domain.clone(),
        fetches: prepared.fetches.clone(),
        recipes: rendered,
        blockers: prepared.blockers.clone(),
        total_ms: total_start.elapsed().as_millis(),
    })
}

fn run_direct_batch_with_context(
    request: &DirectBatchRequest,
    latest: &LatestRun,
    shared_context: Option<&dyn ProjectedMapProvider>,
) -> Result<DirectBatchReport, Box<dyn std::error::Error>> {
    fs::create_dir_all(&request.out_dir)?;
    if request.use_cache {
        fs::create_dir_all(&request.cache_root)?;
    }

    let total_start = Instant::now();
    let planned = plan_direct_recipes(request.model, &request.recipe_slugs)?;
    let groups = group_direct_fetches(request, &planned);
    let plan = build_direct_execution_plan(latest, request.forecast_hour, &groups);
    let loaded = load_execution_plan(
        plan,
        &BundleLoaderConfig {
            cache_root: request.cache_root.clone(),
            use_cache: request.use_cache,
        },
    )?;

    let mut extracted = HashMap::<FieldSelector, SelectedField2D>::new();
    let mut fetches = Vec::with_capacity(groups.len());
    let mut fetch_truth_by_actual_product = HashMap::<String, DirectFetchRuntimeInfo>::new();
    let mut missing_selectors = HashSet::<FieldSelector>::new();
    let mut blockers = Vec::<DirectRecipeBlocker>::new();

    for group in &groups {
        let fetched = match find_loaded_bytes_for_group(&loaded, group) {
            Ok(bytes) => bytes,
            Err(err) => {
                let reason = err.to_string();
                for selector in &group.selectors {
                    missing_selectors.insert(*selector);
                }
                for recipe_slug in recipe_slugs_depending_on_group(&planned, group) {
                    blockers.push(DirectRecipeBlocker {
                        recipe_slug,
                        reason: reason.clone(),
                    });
                }
                continue;
            }
        };
        let (fields, unmatched, timing) = match extract_direct_fetch_group_from_loaded(
            request,
            group,
            fetched,
            request.use_cache,
        ) {
            Ok(result) => result,
            Err(err) => {
                let reason = err.to_string();
                for selector in &group.selectors {
                    missing_selectors.insert(*selector);
                }
                for recipe_slug in recipe_slugs_depending_on_group(&planned, group) {
                    blockers.push(DirectRecipeBlocker {
                        recipe_slug,
                        reason: reason.clone(),
                    });
                }
                continue;
            }
        };
        extracted.extend(fields.into_iter().map(|field| (field.selector, field)));
        for selector in unmatched {
            missing_selectors.insert(selector);
        }
        fetch_truth_by_actual_product.insert(group.product.clone(), timing.runtime_fetch.clone());
        fetches.push(timing);
    }

    let (renderable, selector_blockers) =
        partition_recipes_by_selector_availability(&planned, &missing_selectors);
    blockers.extend(selector_blockers);

    let rendered = render_direct_recipes(
        request,
        latest,
        &renderable,
        &extracted,
        &fetch_truth_by_actual_product,
        shared_context,
    )?;

    Ok(DirectBatchReport {
        model: request.model,
        date_yyyymmdd: request.date_yyyymmdd.clone(),
        cycle_utc: latest.cycle.hour_utc,
        forecast_hour: request.forecast_hour,
        source: latest.source,
        domain: request.domain.clone(),
        fetches,
        recipes: rendered,
        blockers,
        total_ms: total_start.elapsed().as_millis(),
    })
}
