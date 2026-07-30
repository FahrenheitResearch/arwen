use rustwx_core::{CanonicalBundleDescriptor, Field2D, ModelId, ProductKey};
use rustwx_models::{LatestRun, resolve_canonical_bundle_product};
use serde::{Deserialize, Serialize};

use crate::gridded::{
    PressureFields as GenericPressureFields, SurfaceFields as GenericSurfaceFields,
};
use crate::publication::PublishedFetchIdentity;
use crate::runtime::{BundleLoaderConfig, LoadedBundleSet, load_execution_plan};
use crate::severe::build_planned_input_fetches;
use crate::source::ProductSourceRoute;

use super::compute::{DerivedComputedFields, compute_derived_fields_generic};
use super::planning::{build_derived_execution_plan, plan_derived_recipes};
use super::recipes::DerivedRecipe;
use super::types::DerivedRecipeBlocker;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct DerivedQueryField {
    pub recipe_slug: String,
    pub title: String,
    pub units: String,
    pub values: Vec<f64>,
    pub nx: usize,
    pub ny: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct DerivedSampledProductField {
    pub recipe_slug: String,
    pub source_route: ProductSourceRoute,
    pub field: Field2D,
    pub input_fetches: Vec<PublishedFetchIdentity>,
}

#[derive(Debug, Clone)]
pub(crate) struct DerivedSampledProductSet {
    pub fields: Vec<DerivedSampledProductField>,
    pub blockers: Vec<DerivedRecipeBlocker>,
}

pub(crate) fn required_derived_fetch_products(
    model: ModelId,
    recipe_slugs: &[String],
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let recipes = plan_derived_recipes(recipe_slugs)?;
    if recipes.is_empty() {
        return Ok(Vec::new());
    }
    Ok(vec![
        resolve_canonical_bundle_product(model, CanonicalBundleDescriptor::SurfaceAnalysis, None)
            .native_product,
        resolve_canonical_bundle_product(model, CanonicalBundleDescriptor::PressureAnalysis, None)
            .native_product,
    ])
}

pub(crate) fn load_derived_sampled_fields_from_latest(
    latest: &LatestRun,
    forecast_hour: u16,
    cache_root: &std::path::Path,
    use_cache: bool,
    recipe_slugs: &[String],
) -> Result<DerivedSampledProductSet, Box<dyn std::error::Error>> {
    let recipes = plan_derived_recipes(recipe_slugs)?;
    if recipes.is_empty() {
        return Ok(DerivedSampledProductSet {
            fields: Vec::new(),
            blockers: Vec::new(),
        });
    }

    let plan =
        build_derived_execution_plan(latest, forecast_hour, None, None, true, true, &Vec::new());
    let loaded = load_execution_plan(
        plan,
        &BundleLoaderConfig::new(cache_root.to_path_buf(), use_cache),
    )?;
    load_derived_sampled_fields_from_loaded(recipe_slugs, &loaded)
}

pub(crate) fn load_derived_sampled_fields_from_loaded(
    recipe_slugs: &[String],
    loaded: &LoadedBundleSet,
) -> Result<DerivedSampledProductSet, Box<dyn std::error::Error>> {
    let recipes = plan_derived_recipes(recipe_slugs)?;
    if recipes.is_empty() {
        return Ok(DerivedSampledProductSet {
            fields: Vec::new(),
            blockers: Vec::new(),
        });
    }

    let (_, surface_decode, _, pressure_decode) = loaded
        .require_surface_pressure_pair()
        .map_err(|err| format!("derived sampling surface/pressure pair unavailable: {err}"))?;
    let input_fetches = build_planned_input_fetches(loaded);
    let mut fields = Vec::new();
    let mut blockers = Vec::new();

    match compute_derived_fields_generic(&surface_decode.value, &pressure_decode.value, &recipes) {
        Ok(mut computed) => {
            for recipe in recipes {
                match derived_query_field_from_computed(
                    surface_decode.value.nx,
                    surface_decode.value.ny,
                    recipe,
                    &mut computed,
                ) {
                    Ok(query) => {
                        let field = Field2D::new(
                            ProductKey::named(query.recipe_slug.clone()),
                            query.units.clone(),
                            surface_decode.value.core_grid()?,
                            query.values.into_iter().map(|value| value as f32).collect(),
                        )?;
                        fields.push(DerivedSampledProductField {
                            recipe_slug: query.recipe_slug,
                            source_route: ProductSourceRoute::CanonicalDerived,
                            field,
                            input_fetches: input_fetches.clone(),
                        });
                    }
                    Err(err) => blockers.push(DerivedRecipeBlocker {
                        recipe_slug: recipe.slug().to_string(),
                        source_route: ProductSourceRoute::CanonicalDerived,
                        reason: err.to_string(),
                    }),
                }
            }
        }
        Err(shared_err) => {
            for recipe in recipes {
                match compute_derived_query_field(
                    &surface_decode.value,
                    &pressure_decode.value,
                    recipe.slug(),
                ) {
                    Ok(query) => {
                        let field = Field2D::new(
                            ProductKey::named(query.recipe_slug.clone()),
                            query.units.clone(),
                            surface_decode.value.core_grid()?,
                            query.values.into_iter().map(|value| value as f32).collect(),
                        )?;
                        fields.push(DerivedSampledProductField {
                            recipe_slug: query.recipe_slug,
                            source_route: ProductSourceRoute::CanonicalDerived,
                            field,
                            input_fetches: input_fetches.clone(),
                        });
                    }
                    Err(err) => blockers.push(DerivedRecipeBlocker {
                        recipe_slug: recipe.slug().to_string(),
                        source_route: ProductSourceRoute::CanonicalDerived,
                        reason: format!(
                            "shared derived compute failed: {shared_err}; recipe failed: {err}"
                        ),
                    }),
                }
            }
        }
    }

    Ok(DerivedSampledProductSet { fields, blockers })
}

pub(crate) fn compute_derived_query_field(
    surface: &GenericSurfaceFields,
    pressure: &GenericPressureFields,
    recipe_slug: &str,
) -> Result<DerivedQueryField, Box<dyn std::error::Error>> {
    let recipe = DerivedRecipe::parse(recipe_slug).map_err(std::io::Error::other)?;
    if recipe.is_heavy() {
        return Err(format!(
            "heavy derived recipe '{}' is not exposed through the lightweight query path",
            recipe.slug()
        )
        .into());
    }

    let mut computed = compute_derived_fields_generic(surface, pressure, &[recipe])?;
    derived_query_field_from_computed(surface.nx, surface.ny, recipe, &mut computed)
}

/// The single recipe -> computed-field mapping behind both query shapes:
/// a mutable slot reference plus the display units and the field name used
/// in the not-computed error.
fn computed_recipe_slot<'a>(
    recipe: DerivedRecipe,
    computed: &'a mut DerivedComputedFields,
) -> (&'a mut Option<Vec<f64>>, &'static str, &'static str) {
    match recipe {
        DerivedRecipe::Sbcape => (&mut computed.sbcape_jkg, "J/kg", "sbcape_jkg"),
        DerivedRecipe::Sbcin => (&mut computed.sbcin_jkg, "J/kg", "sbcin_jkg"),
        DerivedRecipe::Sblcl => (&mut computed.sblcl_m, "m", "sblcl_m"),
        DerivedRecipe::Mlcape => (&mut computed.mlcape_jkg, "J/kg", "mlcape_jkg"),
        DerivedRecipe::Mlcin => (&mut computed.mlcin_jkg, "J/kg", "mlcin_jkg"),
        DerivedRecipe::Mucape => (&mut computed.mucape_jkg, "J/kg", "mucape_jkg"),
        DerivedRecipe::Mucin => (&mut computed.mucin_jkg, "J/kg", "mucin_jkg"),
        DerivedRecipe::Dcape => (&mut computed.dcape_jkg, "J/kg", "dcape_jkg"),
        DerivedRecipe::ThetaE2m10mWinds => (&mut computed.theta_e_2m_k, "K", "theta_e_2m_k"),
        DerivedRecipe::Vpd2m => (&mut computed.vpd_2m_hpa, "hPa", "vpd_2m_hpa"),
        DerivedRecipe::DewpointDepression2m => (
            &mut computed.dewpoint_depression_2m_c,
            "degC",
            "dewpoint_depression_2m_c",
        ),
        DerivedRecipe::Wetbulb2m => (&mut computed.wetbulb_2m_c, "degC", "wetbulb_2m_c"),
        DerivedRecipe::FireWeatherComposite => (
            &mut computed.fire_weather_composite,
            "index",
            "fire_weather_composite",
        ),
        DerivedRecipe::ApparentTemperature2m => (
            &mut computed.apparent_temperature_2m_c,
            "degC",
            "apparent_temperature_2m_c",
        ),
        DerivedRecipe::HeatIndex2m => (&mut computed.heat_index_2m_c, "degC", "heat_index_2m_c"),
        DerivedRecipe::WindChill2m => (&mut computed.wind_chill_2m_c, "degC", "wind_chill_2m_c"),
        DerivedRecipe::LiftedIndex => (&mut computed.lifted_index_c, "degC", "lifted_index_c"),
        DerivedRecipe::LapseRate700500 => (
            &mut computed.lapse_rate_700_500_cpkm,
            "degC/km",
            "lapse_rate_700_500_cpkm",
        ),
        DerivedRecipe::LapseRate03km => (
            &mut computed.lapse_rate_0_3km_cpkm,
            "degC/km",
            "lapse_rate_0_3km_cpkm",
        ),
        DerivedRecipe::BulkShear01km => (&mut computed.shear_01km_kt, "kt", "shear_01km_kt"),
        DerivedRecipe::BulkShear06km => (&mut computed.shear_06km_kt, "kt", "shear_06km_kt"),
        DerivedRecipe::Srh01km => (&mut computed.srh_01km_m2s2, "m^2/s^2", "srh_01km_m2s2"),
        DerivedRecipe::Srh03km => (&mut computed.srh_03km_m2s2, "m^2/s^2", "srh_03km_m2s2"),
        DerivedRecipe::Ehi01km => (&mut computed.ehi_01km, "dimensionless", "ehi_01km"),
        DerivedRecipe::Ehi03km => (&mut computed.ehi_03km, "dimensionless", "ehi_03km"),
        DerivedRecipe::StpFixed => (&mut computed.stp_fixed, "dimensionless", "stp_fixed"),
        DerivedRecipe::ScpMu03km06kmProxy => (
            &mut computed.scp_mu_03km_06km_proxy,
            "dimensionless",
            "scp_mu_03km_06km_proxy",
        ),
        DerivedRecipe::TemperatureAdvection700mb => (
            &mut computed.temperature_advection_700mb_cph,
            "degC/hr",
            "temperature_advection_700mb_cph",
        ),
        DerivedRecipe::TemperatureAdvection850mb => (
            &mut computed.temperature_advection_850mb_cph,
            "degC/hr",
            "temperature_advection_850mb_cph",
        ),
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
        | DerivedRecipe::EcapeStp => unreachable!("heavy recipes are blocked above"),
    }
}

fn missing_computed_field(recipe: DerivedRecipe, field_name: &str) -> Box<dyn std::error::Error> {
    format!(
        "derived field '{field_name}' was not computed for requested recipe '{}'",
        recipe.slug()
    )
    .into()
}

pub(super) fn derived_query_field_from_computed(
    nx: usize,
    ny: usize,
    recipe: DerivedRecipe,
    computed: &mut DerivedComputedFields,
) -> Result<DerivedQueryField, Box<dyn std::error::Error>> {
    let (slot, units, field_name) = computed_recipe_slot(recipe, computed);
    let values = slot
        .clone()
        .ok_or_else(|| missing_computed_field(recipe, field_name))?;
    Ok(DerivedQueryField {
        recipe_slug: recipe.slug().to_string(),
        title: recipe.title().to_string(),
        units: units.to_string(),
        values,
        nx,
        ny,
    })
}

/// Take-semantics sibling of [`derived_query_field_from_computed`] for the
/// store-ingest lane, where every recipe is realized exactly once: the
/// computed field MOVES out (its slot becomes `None`), so the full f64
/// grid set is never duplicated during the recipe mapping.
pub(super) fn derived_query_field_take_from_computed(
    nx: usize,
    ny: usize,
    recipe: DerivedRecipe,
    computed: &mut DerivedComputedFields,
) -> Result<DerivedQueryField, Box<dyn std::error::Error>> {
    let (slot, units, field_name) = computed_recipe_slot(recipe, computed);
    let values = slot
        .take()
        .ok_or_else(|| missing_computed_field(recipe, field_name))?;
    Ok(DerivedQueryField {
        recipe_slug: recipe.slug().to_string(),
        title: recipe.title().to_string(),
        units: units.to_string(),
        values,
        nx,
        ny,
    })
}
