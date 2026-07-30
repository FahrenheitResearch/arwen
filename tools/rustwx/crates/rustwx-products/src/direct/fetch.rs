use super::planning::FetchGroup;
use super::types::{DirectBatchRequest, DirectFetchRuntimeInfo, DirectFetchTiming};
use crate::publication::fetch_identity_from_cached_result_with_aliases;
use crate::runtime::{FetchedBundleBytes, LoadedBundleSet};
use rustwx_core::{FieldSelector, SelectedField2D};
use rustwx_io::{
    extract_fields_partial_from_model_bytes_at_forecast_hour, load_cached_selected_field,
    store_cached_selected_field,
};
use std::time::Instant;

pub(super) fn find_loaded_bytes_for_group<'a>(
    loaded: &'a LoadedBundleSet,
    group: &FetchGroup,
) -> Result<&'a FetchedBundleBytes, Box<dyn std::error::Error>> {
    if let Some(bundle) = loaded
        .fetched
        .values()
        .find(|bundle| bundle.key.native_product == group.product)
    {
        return Ok(bundle);
    }
    if let Some((key, reason)) = loaded
        .fetch_failures
        .iter()
        .find(|(key, _)| key.native_product == group.product)
    {
        return Err(format!(
            "direct fetch failed for canonical family '{}' from {:?}: {}",
            group.product, key.source, reason
        )
        .into());
    }
    Err(format!(
        "direct planner missed fetch for canonical family '{}'",
        group.product
    )
    .into())
}

pub(super) fn extract_direct_fetch_group_from_loaded(
    request: &DirectBatchRequest,
    group: &FetchGroup,
    fetched: &FetchedBundleBytes,
    use_cache: bool,
) -> Result<(Vec<SelectedField2D>, Vec<FieldSelector>, DirectFetchTiming), Box<dyn std::error::Error>>
{
    let total_start = Instant::now();
    let fetch_request = &fetched.file.request;
    let cached_result = &fetched.file.fetched;
    let fetch_ms = fetched.fetch_ms;

    let extract_start = Instant::now();
    let mut extracted = Vec::<SelectedField2D>::new();
    let mut missing = Vec::<FieldSelector>::new();
    let mut extract_cache_hits = 0usize;
    if use_cache {
        for selector in &group.selectors {
            if let Some(cached) =
                load_cached_selected_field(&request.cache_root, fetch_request, *selector)?
            {
                extracted.push(cached.field);
                extract_cache_hits += 1;
            } else {
                missing.push(*selector);
            }
        }
    } else {
        missing.extend(group.selectors.iter().copied());
    }

    let mut unmatched = Vec::<FieldSelector>::new();
    let parse_start = Instant::now();
    if !missing.is_empty() {
        let partial = extract_fields_partial_from_model_bytes_at_forecast_hour(
            fetch_request.request.model,
            &fetched.file.bytes,
            Some(cached_result.bytes_path.as_path()),
            &missing,
            Some(fetch_request.request.forecast_hour),
        )?;
        if use_cache {
            for field in &partial.extracted {
                store_cached_selected_field(&request.cache_root, fetch_request, field)?;
            }
        }
        let fetched_count = partial.extracted.len();
        extracted.extend(partial.extracted);
        unmatched = partial.missing;
        // extract_cache_misses was previously "count of selectors we
        // had to decode from GRIB"; keep that meaning by subtracting
        // truly-unmatched selectors from the count we actually pulled.
        let _ = fetched_count;
    }
    let parse_ms = parse_start.elapsed().as_millis();
    let extract_ms = extract_start.elapsed().as_millis();

    let extract_cache_misses = missing.len().saturating_sub(unmatched.len());

    Ok((
        extracted,
        unmatched,
        DirectFetchTiming {
            product: group.product.clone(),
            fetch_mode: group.fetch_mode,
            fetch_ms,
            parse_ms,
            extract_ms,
            total_ms: total_start.elapsed().as_millis(),
            fetch_cache_hit: cached_result.cache_hit,
            extract_cache_hits,
            extract_cache_misses,
            runtime_fetch: DirectFetchRuntimeInfo {
                fetch_key: crate::publication::fetch_key(
                    group.product.as_str(),
                    &fetch_request.request,
                ),
                planned_product: group.product.clone(),
                fetched_product: fetch_request.request.product.clone(),
                planned_family_aliases: group.planned_family_aliases.iter().cloned().collect(),
                requested_source: fetch_request
                    .source_override
                    .unwrap_or(cached_result.result.source),
                resolved_source: cached_result.result.source,
                resolved_url: cached_result.result.url.clone(),
            },
            input_fetch: fetch_identity_from_cached_result_with_aliases(
                group.product.as_str(),
                group
                    .planned_family_aliases
                    .iter()
                    .cloned()
                    .collect::<Vec<_>>(),
                fetch_request,
                cached_result,
            ),
        },
    ))
}
