use std::collections::{HashMap, HashSet};

use rustwx_core::{FieldSelector, ModelId, SelectedField2D, SourceId};
use rustwx_models::LatestRun;

use crate::publication::PublishedFetchIdentity;
use crate::runtime::{BundleLoaderConfig, LoadedBundleSet, load_execution_plan};
use crate::source::direct_route_for_recipe_slug;

use super::composite::composite_panel_spec;
use super::planning::{
    build_direct_execution_plan, canonical_fetch_product_for_selectors, group_direct_fetches,
    partition_recipes_by_selector_availability, plan_direct_recipes,
    recipe_slugs_depending_on_group,
};
use super::rendering::render_filled_field;
use super::types::{
    DirectBatchRequest, DirectRecipeBlocker, DirectSampledProductField, DirectSampledProductSet,
};
use super::{
    extract_direct_fetch_group_from_loaded, find_loaded_bytes_for_group, sampling_direct_request,
};

pub(crate) fn required_direct_fetch_products(
    model: ModelId,
    recipe_slugs: &[String],
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let planned = plan_direct_recipes(model, recipe_slugs)?;
    let request =
        sampling_direct_request(model, SourceId::Aws, 0, std::path::Path::new("."), false);
    Ok(group_direct_fetches(&request, &planned)
        .into_iter()
        .map(|group| group.product)
        .collect())
}

pub(crate) fn load_direct_sampled_fields_from_latest(
    latest: &LatestRun,
    forecast_hour: u16,
    cache_root: &std::path::Path,
    use_cache: bool,
    recipe_slugs: &[String],
) -> Result<DirectSampledProductSet, Box<dyn std::error::Error>> {
    let request = sampling_direct_request(
        latest.model,
        latest.source,
        forecast_hour,
        cache_root,
        use_cache,
    );
    let planned = plan_direct_recipes(latest.model, recipe_slugs)?;
    if planned.is_empty() {
        return Ok(DirectSampledProductSet {
            fields: Vec::new(),
            blockers: Vec::new(),
        });
    }

    let groups = group_direct_fetches(&request, &planned);
    let plan = build_direct_execution_plan(latest, forecast_hour, &groups);
    let loaded = load_execution_plan(
        plan,
        &BundleLoaderConfig::new(cache_root.to_path_buf(), use_cache),
    )?;
    load_direct_sampled_fields_from_loaded_request(&request, &loaded, recipe_slugs)
}

fn load_direct_sampled_fields_from_loaded_request(
    request: &DirectBatchRequest,
    loaded: &LoadedBundleSet,
    recipe_slugs: &[String],
) -> Result<DirectSampledProductSet, Box<dyn std::error::Error>> {
    let planned = plan_direct_recipes(request.model, recipe_slugs)?;
    if planned.is_empty() {
        return Ok(DirectSampledProductSet {
            fields: Vec::new(),
            blockers: Vec::new(),
        });
    }

    let groups = group_direct_fetches(request, &planned);

    let mut extracted = HashMap::<FieldSelector, SelectedField2D>::new();
    let mut missing_selectors = HashSet::<FieldSelector>::new();
    let mut blockers = Vec::<DirectRecipeBlocker>::new();
    let mut fetches_by_product = HashMap::<String, PublishedFetchIdentity>::new();

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
        let (fields, unmatched, timing) =
            extract_direct_fetch_group_from_loaded(request, group, fetched, request.use_cache)?;
        extracted.extend(fields.into_iter().map(|field| (field.selector, field)));
        for selector in unmatched {
            missing_selectors.insert(selector);
        }
        fetches_by_product.insert(group.product.clone(), timing.input_fetch.clone());
    }

    let (renderable, selector_blockers) =
        partition_recipes_by_selector_availability(&planned, &missing_selectors);
    blockers.extend(selector_blockers);

    let mut fields = Vec::new();
    for item in renderable {
        let canonical_product = canonical_fetch_product_for_selectors(
            request,
            item.plan.product.as_ref(),
            &item.plan.selectors(),
        );
        let input_fetches: Vec<PublishedFetchIdentity> = fetches_by_product
            .get(&canonical_product)
            .cloned()
            .into_iter()
            .collect();
        if composite_panel_spec(item.recipe.slug).is_some() {
            continue;
        }
        let Some(filled_selector) = item.recipe.filled.selector else {
            blockers.push(DirectRecipeBlocker {
                recipe_slug: item.recipe.slug.to_string(),
                reason: "direct recipe is missing a filled selector binding".to_string(),
            });
            continue;
        };
        let Some(filled) = extracted.get(&filled_selector) else {
            blockers.push(DirectRecipeBlocker {
                recipe_slug: item.recipe.slug.to_string(),
                reason: format!(
                    "direct recipe '{}' was renderable but missing selector {}",
                    item.recipe.slug,
                    filled_selector.key()
                ),
            });
            continue;
        };
        let field = render_filled_field(item.recipe, filled, &extracted)?;
        fields.push(DirectSampledProductField {
            recipe_slug: item.recipe.slug.to_string(),
            source_route: direct_route_for_recipe_slug(item.recipe.slug),
            field_selector: Some(filled_selector),
            field,
            input_fetches,
        });
    }

    Ok(DirectSampledProductSet { fields, blockers })
}
