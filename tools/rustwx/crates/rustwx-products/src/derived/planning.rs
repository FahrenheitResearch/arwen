use std::collections::{BTreeSet, HashSet};

use rustwx_core::{BundleRequirement, CanonicalBundleDescriptor, ModelId};
use rustwx_models::{
    latest_available_run_at_forecast_hour, latest_available_run_for_products_at_forecast_hour,
    resolve_canonical_bundle_product,
};

use crate::gridded::resolve_thermo_pair_run;
use crate::planner::ExecutionPlanBuilder;
use crate::severe::build_severe_execution_plan;
use crate::source::{ProductSourceMode, ProductSourceRoute};
use crate::thermo_native::{NativeSemantics, NativeThermoRecipe, native_candidate};

use super::presentation::is_gdex_dataset_token;
use super::recipes::{DerivedRecipe, derived_compute_recipes_need_pressure};
use super::types::{DerivedBatchRequest, DerivedRecipeBlocker};

#[derive(Debug, Clone, Copy)]
pub(crate) enum NativeDerivedRecipe {
    Thermo(NativeThermoRecipe),
    WrfGdexScalar,
    WrfGdexVectorMagnitude,
}

#[derive(Debug, Clone)]
pub(crate) struct PlannedNativeDerivedCandidate {
    pub(crate) label: String,
    pub(crate) semantics: NativeSemantics,
    pub(crate) auto_eligible: bool,
    pub(crate) detail: String,
    pub(crate) fetch_product: &'static str,
}

#[derive(Debug, Clone)]
pub(crate) struct PlannedNativeThermoRoute {
    pub(crate) recipe: DerivedRecipe,
    pub(crate) native_recipe: NativeDerivedRecipe,
    pub(crate) candidate: PlannedNativeDerivedCandidate,
    pub(crate) source_route: ProductSourceRoute,
}

#[derive(Debug, Clone)]
pub(crate) struct PlannedDerivedSourceRoutes {
    pub(crate) output_recipes: Vec<DerivedRecipe>,
    pub(crate) compute_recipes: Vec<DerivedRecipe>,
    pub(crate) heavy_recipes: Vec<DerivedRecipe>,
    pub(crate) native_routes: Vec<PlannedNativeThermoRoute>,
    pub(crate) blockers: Vec<DerivedRecipeBlocker>,
}

pub(crate) fn plan_derived_recipes(
    recipe_slugs: &[String],
) -> Result<Vec<DerivedRecipe>, Box<dyn std::error::Error>> {
    let mut seen = HashSet::<DerivedRecipe>::new();
    let mut planned = Vec::new();
    for slug in recipe_slugs {
        let recipe = DerivedRecipe::parse(slug).map_err(|err| format!("{slug}: {err}"))?;
        if seen.insert(recipe) {
            planned.push(recipe);
        }
    }
    Ok(planned)
}

fn native_recipe_for_derived(recipe: DerivedRecipe) -> Option<NativeThermoRecipe> {
    match recipe {
        DerivedRecipe::Sbcape => Some(NativeThermoRecipe::Sbcape),
        DerivedRecipe::Sbcin => Some(NativeThermoRecipe::Sbcin),
        DerivedRecipe::Sblcl => Some(NativeThermoRecipe::Sblcl),
        DerivedRecipe::Mlcape => Some(NativeThermoRecipe::Mlcape),
        DerivedRecipe::Mlcin => Some(NativeThermoRecipe::Mlcin),
        DerivedRecipe::Mucape => Some(NativeThermoRecipe::Mucape),
        DerivedRecipe::Mucin => Some(NativeThermoRecipe::Mucin),
        DerivedRecipe::LiftedIndex => Some(NativeThermoRecipe::LiftedIndex),
        _ => None,
    }
}

fn planned_candidate_from_native(
    model: ModelId,
    recipe: DerivedRecipe,
    surface_product_override: Option<&str>,
) -> Option<(NativeDerivedRecipe, PlannedNativeDerivedCandidate)> {
    if model == ModelId::WrfGdex {
        if let Some(candidate) = wrf_gdex_native_candidate(recipe, surface_product_override) {
            return Some(candidate);
        }
    }

    let native_recipe = native_recipe_for_derived(recipe)?;
    let candidate = native_candidate(model, native_recipe)?;
    Some((
        NativeDerivedRecipe::Thermo(native_recipe),
        PlannedNativeDerivedCandidate {
            label: candidate.label.to_string(),
            semantics: candidate.semantics,
            auto_eligible: candidate.auto_eligible,
            detail: candidate.detail.to_string(),
            fetch_product: candidate.fetch_product,
        },
    ))
}

fn wrf_gdex_native_candidate(
    recipe: DerivedRecipe,
    surface_product_override: Option<&str>,
) -> Option<(NativeDerivedRecipe, PlannedNativeDerivedCandidate)> {
    let fetch_product = resolve_canonical_bundle_product(
        ModelId::WrfGdex,
        CanonicalBundleDescriptor::SurfaceAnalysis,
        surface_product_override,
    )
    .native_product;
    if !wrf_gdex_native_surface_product(&fetch_product) {
        return None;
    }
    let fetch_product = leak_static_str(fetch_product);

    let (native_recipe, label, detail) = match recipe {
        DerivedRecipe::Sbcape => (
            NativeDerivedRecipe::WrfGdexScalar,
            "surface CAPE",
            "WRF native SBCAPE from model diagnostics",
        ),
        DerivedRecipe::Sbcin => (
            NativeDerivedRecipe::WrfGdexScalar,
            "surface CIN",
            "WRF native SBCINH from model diagnostics",
        ),
        DerivedRecipe::Sblcl => (
            NativeDerivedRecipe::WrfGdexScalar,
            "surface LCL height",
            "WRF native SBLCL from model diagnostics",
        ),
        DerivedRecipe::Mlcape => (
            NativeDerivedRecipe::WrfGdexScalar,
            "mixed-layer CAPE",
            "WRF native MLCAPE from model diagnostics",
        ),
        DerivedRecipe::Mlcin => (
            NativeDerivedRecipe::WrfGdexScalar,
            "mixed-layer CIN",
            "WRF native MLCINH from model diagnostics",
        ),
        DerivedRecipe::Mucape => (
            NativeDerivedRecipe::WrfGdexScalar,
            "most-unstable CAPE",
            "WRF native MUCAPE from model diagnostics",
        ),
        DerivedRecipe::Mucin => (
            NativeDerivedRecipe::WrfGdexScalar,
            "most-unstable CIN",
            "WRF native MUCINH from model diagnostics",
        ),
        DerivedRecipe::Srh01km => (
            NativeDerivedRecipe::WrfGdexScalar,
            "0-1 km SRH",
            "WRF native SRH01 from model diagnostics",
        ),
        DerivedRecipe::Srh03km => (
            NativeDerivedRecipe::WrfGdexScalar,
            "0-3 km SRH",
            "WRF native SRH03 from model diagnostics",
        ),
        DerivedRecipe::BulkShear01km => (
            NativeDerivedRecipe::WrfGdexVectorMagnitude,
            "0-1 km bulk shear",
            "WRF native 0-1 km shear magnitude from model diagnostics",
        ),
        DerivedRecipe::BulkShear06km => (
            NativeDerivedRecipe::WrfGdexVectorMagnitude,
            "0-6 km bulk shear",
            "WRF native 0-6 km shear magnitude from model diagnostics",
        ),
        _ => return None,
    };

    Some((
        native_recipe,
        PlannedNativeDerivedCandidate {
            label: label.to_string(),
            semantics: NativeSemantics::ExactEquivalent,
            auto_eligible: true,
            detail: detail.to_string(),
            fetch_product,
        },
    ))
}

fn wrf_gdex_native_surface_product(product: &str) -> bool {
    let normalized = product.replace('_', "-").to_ascii_lowercase();
    let Some((dataset, suffix)) = normalized.split_once('-') else {
        return false;
    };
    is_gdex_dataset_token(dataset)
        && (matches!(suffix, "hist2d" | "future2d")
            || (suffix.starts_with('d')
                && suffix.len() == 3
                && suffix[1..].chars().all(|ch| ch.is_ascii_digit())))
}

pub(crate) fn plan_native_thermo_routes_with_surface_product(
    model: ModelId,
    recipes: &[DerivedRecipe],
    mode: ProductSourceMode,
    surface_product_override: Option<&str>,
) -> Result<PlannedDerivedSourceRoutes, Box<dyn std::error::Error>> {
    let mut output_recipes = Vec::new();
    let mut compute_recipes = Vec::new();
    let mut heavy_recipes = Vec::new();
    let mut native_routes = Vec::new();
    let mut blockers = Vec::new();

    for &recipe in recipes {
        if recipe.is_heavy() {
            match mode {
                ProductSourceMode::Canonical => {
                    output_recipes.push(recipe);
                    heavy_recipes.push(recipe);
                }
                ProductSourceMode::Fastest => blockers.push(DerivedRecipeBlocker {
                    recipe_slug: recipe.slug().to_string(),
                    source_route: ProductSourceRoute::BlockedNoFastRoute,
                    reason: format!(
                        "recipe '{}' uses the cropped heavy ECAPE path; fastest mode will not fall back to canonical-derived compute",
                        recipe.slug()
                    ),
                }),
            }
            continue;
        }

        let candidate = planned_candidate_from_native(model, recipe, surface_product_override);

        match mode {
            ProductSourceMode::Canonical => {
                if let Some((native_recipe, candidate)) = candidate {
                    if use_native_route_in_canonical_mode(model, &candidate) {
                        output_recipes.push(recipe);
                        native_routes.push(PlannedNativeThermoRoute {
                            recipe,
                            native_recipe,
                            source_route: native_source_route(candidate.semantics),
                            candidate,
                        });
                        continue;
                    }
                }
                output_recipes.push(recipe);
                compute_recipes.push(recipe);
            }
            ProductSourceMode::Fastest => {
                if let Some((native_recipe, candidate)) = candidate {
                    output_recipes.push(recipe);
                    native_routes.push(PlannedNativeThermoRoute {
                        recipe,
                        native_recipe,
                        source_route: native_source_route(candidate.semantics),
                        candidate,
                    });
                } else if let Some(source_route) = cheap_fastest_route(recipe) {
                    output_recipes.push(recipe);
                    if matches!(source_route, ProductSourceRoute::CheapDerived) {
                        compute_recipes.push(recipe);
                    }
                } else {
                    blockers.push(DerivedRecipeBlocker {
                        recipe_slug: recipe.slug().to_string(),
                        source_route: ProductSourceRoute::BlockedNoFastRoute,
                        reason: format!(
                            "recipe '{}' has no fast native/cheap route; fastest mode will not fall back to canonical-derived compute",
                            recipe.slug()
                        ),
                    });
                }
            }
        }
    }

    Ok(PlannedDerivedSourceRoutes {
        output_recipes,
        compute_recipes,
        heavy_recipes,
        native_routes,
        blockers,
    })
}

fn leak_static_str(value: String) -> &'static str {
    Box::leak(value.into_boxed_str())
}

fn native_source_route(semantics: NativeSemantics) -> ProductSourceRoute {
    match semantics {
        NativeSemantics::ExactEquivalent => ProductSourceRoute::NativeExact,
        NativeSemantics::ProxyEquivalent => ProductSourceRoute::NativeProxy,
    }
}

fn use_native_route_in_canonical_mode(
    model: ModelId,
    candidate: &PlannedNativeDerivedCandidate,
) -> bool {
    model == ModelId::WrfGdex
        || (model == ModelId::Gfs
            && matches!(candidate.semantics, NativeSemantics::ExactEquivalent))
}

pub(super) fn cheap_fastest_route(_recipe: DerivedRecipe) -> Option<ProductSourceRoute> {
    // The current derived kernel still routes every non-native recipe
    // through the canonical surface+pressure pair compute path. Until a
    // recipe can be satisfied from already-loaded native/direct inputs
    // without forcing that pair, fastest mode blocks it explicitly.
    None
}

pub(super) fn resolve_derived_run(
    request: &DerivedBatchRequest,
    derived_compute_recipes: &[DerivedRecipe],
    heavy_recipes: &[DerivedRecipe],
    native_routes: &[PlannedNativeThermoRoute],
) -> Result<rustwx_models::LatestRun, Box<dyn std::error::Error>> {
    let needs_pair =
        derived_compute_recipes_need_pressure(derived_compute_recipes) || !heavy_recipes.is_empty();
    if let Some(hour_utc) = request.cycle_override_utc {
        if !needs_pair {
            return Ok(rustwx_models::LatestRun {
                model: request.model,
                cycle: rustwx_core::CycleSpec::new(request.date_yyyymmdd.clone(), hour_utc)?,
                source: request.source,
            });
        }
        return resolve_thermo_pair_run(
            request.model,
            &request.date_yyyymmdd,
            Some(hour_utc),
            request.forecast_hour,
            request.source,
            request.surface_product_override.as_deref(),
            request.pressure_product_override.as_deref(),
        )
        .map_err(Into::into);
    }

    if !needs_pair && native_routes.is_empty() {
        return latest_available_run_at_forecast_hour(
            request.model,
            Some(request.source),
            &request.date_yyyymmdd,
            request.forecast_hour,
        )
        .map_err(Into::into);
    }

    let mut required_products = BTreeSet::<String>::new();
    if needs_pair {
        required_products.insert(
            resolve_canonical_bundle_product(
                request.model,
                CanonicalBundleDescriptor::SurfaceAnalysis,
                request.surface_product_override.as_deref(),
            )
            .native_product,
        );
        required_products.insert(
            resolve_canonical_bundle_product(
                request.model,
                CanonicalBundleDescriptor::PressureAnalysis,
                request.pressure_product_override.as_deref(),
            )
            .native_product,
        );
    }
    for route in native_routes {
        required_products.insert(route.candidate.fetch_product.to_string());
    }
    if required_products.is_empty() {
        return resolve_thermo_pair_run(
            request.model,
            &request.date_yyyymmdd,
            request.cycle_override_utc,
            request.forecast_hour,
            request.source,
            request.surface_product_override.as_deref(),
            request.pressure_product_override.as_deref(),
        )
        .map_err(Into::into);
    }

    let required_refs = required_products
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();
    latest_available_run_for_products_at_forecast_hour(
        request.model,
        Some(request.source),
        &request.date_yyyymmdd,
        &required_refs,
        request.forecast_hour,
    )
    .map_err(Into::into)
}

pub(super) fn build_derived_execution_plan(
    latest: &rustwx_models::LatestRun,
    forecast_hour: u16,
    surface_product_override: Option<&str>,
    pressure_product_override: Option<&str>,
    include_pair: bool,
    include_surface: bool,
    native_routes: &[PlannedNativeThermoRoute],
) -> crate::planner::ExecutionPlan {
    let mut builder = ExecutionPlanBuilder::new(latest, forecast_hour);
    if include_pair {
        let pair_plan = build_severe_execution_plan(
            latest,
            forecast_hour,
            surface_product_override,
            pressure_product_override,
        );
        for bundle in &pair_plan.bundles {
            for alias in &bundle.aliases {
                let mut requirement = BundleRequirement::new(alias.bundle, bundle.id.forecast_hour);
                if let Some(ref over) = alias.native_override {
                    requirement = requirement.with_native_override(over.clone());
                }
                builder.require_with_logical_family(&requirement, alias.logical_family.as_deref());
            }
        }
    } else if include_surface {
        let native_product = resolve_canonical_bundle_product(
            latest.model,
            CanonicalBundleDescriptor::SurfaceAnalysis,
            surface_product_override,
        )
        .native_product;
        let requirement =
            BundleRequirement::new(CanonicalBundleDescriptor::SurfaceAnalysis, forecast_hour)
                .with_native_override(native_product);
        builder.require_with_logical_family(&requirement, Some("sfc"));
    }
    let mut seen_native_products = BTreeSet::<String>::new();
    for route in native_routes {
        if seen_native_products.insert(route.candidate.fetch_product.to_string()) {
            let requirement =
                BundleRequirement::new(CanonicalBundleDescriptor::NativeAnalysis, forecast_hour)
                    .with_native_override(route.candidate.fetch_product);
            builder.require_with_logical_family(
                &requirement,
                Some(&format!("thermo-native:{}", route.candidate.fetch_product)),
            );
        }
    }
    builder.build()
}
