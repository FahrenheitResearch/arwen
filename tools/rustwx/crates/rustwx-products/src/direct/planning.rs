use std::collections::{HashMap, HashSet};

use rustwx_core::{
    BundleRequirement, CanonicalBundleDescriptor, CanonicalField, FieldSelector, ModelId, SourceId,
    VerticalSelector,
};
use rustwx_models::{
    LatestRun, ModelError, PlotRecipe, PlotRecipeFetchMode, PlotRecipeFetchPlan, plot_recipe,
    plot_recipe_fetch_plan,
};

use crate::planner::{ExecutionPlan, ExecutionPlanBuilder};
use crate::spec::direct_product_specs;

use super::{
    DirectBatchRequest, DirectRecipeBlocker, composite::composite_panel_spec,
    titles::is_gdex_dataset_token,
};

#[derive(Debug, Clone)]
pub(super) struct PlannedDirectRecipe {
    pub(super) recipe: &'static PlotRecipe,
    pub(super) plan: PlotRecipeFetchPlan,
}

#[derive(Debug, Clone)]
pub struct FetchGroup {
    pub product: String,
    pub fetch_mode: PlotRecipeFetchMode,
    // Retained for recipe-level coverage/debugging; the direct/native batch
    // path intentionally pulls full family GRIB bytes and extracts grouped
    // selectors from the parsed full file.
    pub variable_patterns: Vec<String>,
    pub selectors: Vec<FieldSelector>,
    /// Sorted set of logical planned-family names that collapsed into this
    /// canonical fetch. For HRRR this preserves the "nat" logical identity even
    /// when it reroutes to the physical "sfc" file.
    pub planned_family_aliases: std::collections::BTreeSet<String>,
}

pub fn supported_direct_recipe_slugs(model: ModelId) -> Vec<String> {
    direct_product_specs()
        .into_iter()
        .filter(|spec| !direct_recipe_requires_explicit_opt_in(&spec.slug))
        .filter(|spec| plot_recipe_fetch_plan(&spec.slug, model).is_ok())
        .map(|spec| spec.slug)
        .collect()
}

/// The store-lane counterpart of [`supported_direct_recipe_slugs`]:
/// every non-opt-in direct recipe, with NO model-identity filter
/// (gpuwm vendor divergence).  Local stores decide availability by
/// resolving each recipe's field selectors against the stored fields;
/// filtering candidates by a fetch plan here would smuggle a source
/// identity gate back into the render path.
pub fn store_direct_recipe_slugs() -> Vec<String> {
    direct_product_specs()
        .into_iter()
        .filter(|spec| !direct_recipe_requires_explicit_opt_in(&spec.slug))
        .map(|spec| spec.slug)
        .collect()
}

/// Ensemble/probabilistic recipe families that never render implicitly:
/// the "all"/catalog paths skip them, and callers (including gpuwm's
/// vendored catalog lister) must name them explicitly.  Public so
/// catalog listers can apply the exact same filter the planner does.
pub fn direct_recipe_requires_explicit_opt_in(slug: &str) -> bool {
    slug.starts_with("nbm_qmd_")
        || slug.starts_with("sref_prob_")
        || slug.starts_with("gefs_avg_")
        || slug.starts_with("gefs_spr_")
        || slug.starts_with("aigefs_spr_")
        || slug.starts_with("hgefs_spr_")
        || slug.starts_with("href_sprd_")
        || slug.starts_with("href_prob_")
        || slug.starts_with("href_mean_")
        || slug.starts_with("refs_sprd_")
        || slug.starts_with("refs_prob_")
}

pub(super) fn plan_direct_recipes(
    model: ModelId,
    recipe_slugs: &[String],
) -> Result<Vec<PlannedDirectRecipe>, Box<dyn std::error::Error>> {
    let mut planned = Vec::new();
    let mut seen = HashSet::<String>::new();
    for slug in recipe_slugs {
        let recipe = plot_recipe(slug).ok_or_else(|| format!("unknown recipe '{slug}'"))?;
        if !seen.insert(recipe.slug.to_string()) {
            continue;
        }
        let plan = match plot_recipe_fetch_plan(recipe.slug, model) {
            Ok(plan) => plan,
            Err(ModelError::UnsupportedPlotRecipeModel { reason, .. }) => {
                return Err(format!(
                    "plot recipe '{}' is not supported for {}: {}",
                    recipe.slug, model, reason
                )
                .into());
            }
            Err(err) => return Err(err.into()),
        };
        planned.push(PlannedDirectRecipe { recipe, plan });
    }
    Ok(planned)
}

/// Which planned recipe slugs route their fetches through this group?
pub(super) fn recipe_slugs_depending_on_group(
    planned: &[PlannedDirectRecipe],
    group: &FetchGroup,
) -> Vec<String> {
    planned
        .iter()
        .filter(|item| {
            item.plan
                .selectors()
                .into_iter()
                .any(|sel| group.selectors.contains(&sel))
        })
        .map(|item| item.recipe.slug.to_string())
        .collect()
}

pub(super) fn partition_recipes_by_selector_availability(
    planned: &[PlannedDirectRecipe],
    missing: &HashSet<FieldSelector>,
) -> (Vec<PlannedDirectRecipe>, Vec<DirectRecipeBlocker>) {
    let mut renderable = Vec::with_capacity(planned.len());
    let mut blockers = Vec::new();
    for item in planned {
        let reason = recipe_block_reason(item.recipe, missing);
        match reason {
            Some(reason) => blockers.push(DirectRecipeBlocker {
                recipe_slug: item.recipe.slug.to_string(),
                reason,
            }),
            None => renderable.push(item.clone()),
        }
    }
    (renderable, blockers)
}

pub(super) fn recipe_block_reason(
    recipe: &PlotRecipe,
    missing: &HashSet<FieldSelector>,
) -> Option<String> {
    if let Some(spec) = composite_panel_spec(recipe.slug) {
        for component_slug in spec.component_slugs {
            let Some(component) = plot_recipe(component_slug) else {
                continue;
            };
            if let Some(selector) = component.filled.selector {
                if missing.contains(&selector) {
                    return Some(format!(
                        "composite component '{}' missing selector {}",
                        component_slug,
                        selector.key()
                    ));
                }
            }
        }
        return None;
    }
    if let Some(selector) = recipe.filled.selector {
        if missing.contains(&selector) {
            return Some(format!(
                "missing GRIB message for filled selector {}",
                selector.key()
            ));
        }
    }
    None
}

pub(super) fn group_direct_fetches(
    request: &DirectBatchRequest,
    recipes: &[PlannedDirectRecipe],
) -> Vec<FetchGroup> {
    let mut grouped = HashMap::<String, FetchGroup>::new();
    for item in recipes {
        let planned_family = item.plan.product.to_string();
        let selectors = item.plan.selectors();
        let key =
            canonical_fetch_product_for_selectors(request, planned_family.as_str(), &selectors);
        let entry = grouped.entry(key.clone()).or_insert_with(|| FetchGroup {
            product: key.clone(),
            fetch_mode: PlotRecipeFetchMode::WholeFileStructuredExtract,
            variable_patterns: Vec::new(),
            selectors: Vec::new(),
            planned_family_aliases: std::collections::BTreeSet::new(),
        });
        entry.planned_family_aliases.insert(planned_family);
        for pattern in item.plan.variable_patterns() {
            if !entry.variable_patterns.iter().any(|value| value == pattern) {
                entry.variable_patterns.push(pattern.to_string());
            }
        }
        for selector in selectors {
            if !entry.selectors.contains(&selector) {
                entry.selectors.push(selector);
            }
        }
        for (product, selector) in
            extra_direct_selectors(request, item.plan.product.as_ref(), item.recipe)
        {
            let extra_key = canonical_fetch_product_for_selectors(request, &product, &[selector]);
            let extra_entry = grouped
                .entry(extra_key.clone())
                .or_insert_with(|| FetchGroup {
                    product: extra_key.clone(),
                    fetch_mode: PlotRecipeFetchMode::WholeFileStructuredExtract,
                    variable_patterns: Vec::new(),
                    selectors: Vec::new(),
                    planned_family_aliases: std::collections::BTreeSet::new(),
                });
            extra_entry.planned_family_aliases.insert(product);
            if !extra_entry.selectors.contains(&selector) {
                extra_entry.selectors.push(selector);
            }
        }
    }
    let mut groups = grouped.into_values().collect::<Vec<_>>();
    groups.sort_by(|left, right| left.product.cmp(&right.product));
    groups
}

fn extra_direct_selectors(
    request: &DirectBatchRequest,
    planned_product: &str,
    recipe: &PlotRecipe,
) -> Vec<(String, FieldSelector)> {
    if request.model == ModelId::WrfGdex {
        if let Some(FieldSelector {
            vertical: VerticalSelector::IsobaricHpa(_),
            ..
        }) = recipe.filled.selector
        {
            return vec![(
                wrf_gdex_surface_pressure_product(request, planned_product),
                FieldSelector::surface(CanonicalField::Pressure),
            )];
        }
    }
    Vec::new()
}

pub(super) fn canonical_fetch_product(
    request: &DirectBatchRequest,
    planned_product: &str,
) -> String {
    canonical_fetch_product_for_selectors(request, planned_product, &[])
}

fn wrf_gdex_surface_pressure_product(
    request: &DirectBatchRequest,
    planned_product: &str,
) -> String {
    let product = canonical_fetch_product(request, planned_product);
    let normalized = product.replace('_', "-").to_ascii_lowercase();
    let Some((dataset, suffix)) = normalized.split_once('-') else {
        return product;
    };
    if !is_gdex_dataset_token(dataset) {
        return product;
    }
    match suffix {
        "hist3d" => format!("{dataset}-hist2d"),
        "future3d" => format!("{dataset}-future2d"),
        _ => product,
    }
}

pub(super) fn canonical_fetch_product_for_selectors(
    request: &DirectBatchRequest,
    planned_product: &str,
    selectors: &[FieldSelector],
) -> String {
    if let Some(overridden) = request.product_overrides.get(planned_product) {
        return overridden.clone();
    }

    match (request.model, planned_product) {
        (ModelId::Hrrr, "nat") if hrrr_native_selectors_require_wrfnat(selectors) => {
            "nat".to_string()
        }
        (ModelId::Hrrr, "nat") => "sfc".to_string(),
        _ => planned_product.to_string(),
    }
}

fn hrrr_native_selectors_require_wrfnat(selectors: &[FieldSelector]) -> bool {
    selectors.iter().any(|selector| {
        matches!(
            selector.field,
            CanonicalField::SmokeMassDensity | CanonicalField::ColumnIntegratedSmoke
        )
    })
}

pub(super) fn build_direct_execution_plan(
    latest: &LatestRun,
    forecast_hour: u16,
    groups: &[FetchGroup],
) -> ExecutionPlan {
    let mut builder = ExecutionPlanBuilder::new(latest, forecast_hour);
    for group in groups {
        let requirement =
            BundleRequirement::new(CanonicalBundleDescriptor::NativeAnalysis, forecast_hour)
                .with_native_override(group.product.clone());
        for alias in &group.planned_family_aliases {
            if should_attach_direct_idx_patterns(latest.source) {
                builder.require_with_logical_family_and_patterns(
                    &requirement,
                    Some(alias),
                    group.variable_patterns.clone(),
                );
            } else {
                builder.require_with_logical_family(&requirement, Some(alias));
            }
        }
    }
    builder.build()
}

pub(super) fn should_attach_direct_idx_patterns(source: SourceId) -> bool {
    matches!(source, SourceId::Aws | SourceId::Google)
}
