//! Store-render lane: render derived and heavy recipe PNGs from grids that
//! were precomputed at ingest and read back from the rw-store, through the
//! exact render path the GRIB lanes use.
//!
//! This mirrors the proven "precomputed" branch of the derived batch runner
//! (`run_derived_batch_from_loaded_bundles_with_precomputed` with
//! `Some(shared)`): full-grid values are cropped to the projected domain
//! intersection (`classify_projected_grid_intersection`, margin 2 — the
//! same crop `crop_heavy_domain_for_projected_extent` and
//! `crop_and_guard_heavy_domain` compute from the same projected map), the
//! projected map is rebuilt for the cropped grid, and each recipe renders
//! through [`render_derived_output_recipe`] (non-heavy) or
//! [`render_derived_heavy_recipe`] (heavy) — the same functions, styles,
//! scales, titles, and subtitles as the GRIB lanes. No compute and no
//! domain-size guard happens here: the grids already exist.

use std::collections::HashMap;

use rayon::prelude::*;
use rustwx_render::map_frame_aspect_ratio;

use crate::gridded::{
    ProjectedGridIntersection, classify_projected_grid_intersection, crop_latlon_grid,
    crop_values_f64,
};
use crate::shared_context::WeatherPanelField;

use super::compute::DerivedComputedFields;
use super::presentation::DerivedRenderOverrides;
use super::recipes::DerivedRecipe;
use super::types::{DerivedBatchRequest, DerivedRenderedRecipe};
use super::{
    build_derived_projected_map_with_projection, render_derived_heavy_recipe,
    render_derived_output_recipe,
};

/// One full-grid product plane read back from the store: the recipe slug
/// (also the store variable name), the stored display units, and the
/// row-major `ny * nx` values.
#[derive(Debug, Clone)]
pub struct StoreProductGrid {
    pub slug: String,
    pub units: String,
    pub values: Vec<f64>,
}

/// Render every recipe in `request.recipe_slugs` from store-read grids.
///
/// `full_grid`/`projection` describe the full hour grid the values sit on;
/// `grids` must contain one entry per requested recipe slug (the caller
/// resolves store coverage and reports unresolvable slugs itself).
/// `surface_winds_10m_ms` carries the full-grid 10 m u/v planes (m/s) that
/// the `theta_e_2m_10m_winds` barb overlay needs; it is only required when
/// that recipe is requested. `input_fetch_keys` is provenance metadata for
/// the rendered-recipe report (it never reaches pixels).
#[allow(clippy::too_many_arguments)]
pub fn render_derived_recipes_from_store_grids(
    request: &DerivedBatchRequest,
    cycle_utc: u8,
    full_grid: &rustwx_core::LatLonGrid,
    projection: Option<&rustwx_core::GridProjection>,
    grids: &[StoreProductGrid],
    surface_winds_10m_ms: Option<(&[f64], &[f64])>,
    input_fetch_keys: Vec<String>,
) -> Result<Vec<DerivedRenderedRecipe>, Box<dyn std::error::Error>> {
    let recipes = request
        .recipe_slugs
        .iter()
        .map(|slug| DerivedRecipe::parse(slug).map_err(|err| format!("{slug}: {err}")))
        .collect::<Result<Vec<_>, _>>()?;
    if recipes.is_empty() {
        return Ok(Vec::new());
    }
    std::fs::create_dir_all(&request.out_dir)?;
    let by_slug: HashMap<&str, &StoreProductGrid> = grids
        .iter()
        .map(|grid| (grid.slug.as_str(), grid))
        .collect();
    let cells = full_grid.shape.len();
    for recipe in &recipes {
        let grid = by_slug.get(recipe.slug()).ok_or_else(|| {
            format!(
                "store render requested recipe '{}' without its store grid",
                recipe.slug()
            )
        })?;
        if grid.values.len() != cells {
            return Err(format!(
                "store grid '{}' holds {} values, expected {cells}",
                recipe.slug(),
                grid.values.len()
            )
            .into());
        }
    }

    // Same projected map + crop the GRIB derived/heavy lanes derive from the
    // full grid for this domain (classify margin 2).
    let full_projected = build_derived_projected_map_with_projection(
        request.model,
        &full_grid.lat_deg,
        &full_grid.lon_deg,
        projection,
        request.domain.bounds,
        map_frame_aspect_ratio(request.output_width, request.output_height, true, true),
    )?;
    let projected_crop_pad_cells = store_render_projected_crop_pad_cells(request.domain.bounds);
    let (grid, projected, crop) = match classify_projected_grid_intersection(
        full_grid.shape.nx,
        full_grid.shape.ny,
        &full_projected.projected_x,
        &full_projected.projected_y,
        &full_projected.extent,
        projected_crop_pad_cells,
    )? {
        ProjectedGridIntersection::Empty => {
            return Err(format!(
                "store render projected crop for domain '{}' produced an empty domain",
                request.domain.slug
            )
            .into());
        }
        ProjectedGridIntersection::Full => (full_grid.clone(), full_projected, None),
        ProjectedGridIntersection::Crop(crop) => {
            let cropped_grid = crop_latlon_grid(full_grid, crop)?;
            let cropped_projected = build_derived_projected_map_with_projection(
                request.model,
                &cropped_grid.lat_deg,
                &cropped_grid.lon_deg,
                projection,
                request.domain.bounds,
                map_frame_aspect_ratio(request.output_width, request.output_height, true, true),
            )?;
            (cropped_grid, cropped_projected, Some(crop))
        }
    };
    let crop_plane = |values: &[f64]| match crop {
        Some(crop) => crop_values_f64(values, full_grid.shape.nx, crop),
        None => values.to_vec(),
    };

    let mut computed = DerivedComputedFields::default();
    for recipe in recipes.iter().filter(|recipe| !recipe.is_heavy()) {
        let grid = by_slug[recipe.slug()];
        assign_store_values(&mut computed, *recipe, crop_plane(&grid.values))?;
    }
    if recipes
        .iter()
        .any(|recipe| matches!(recipe, DerivedRecipe::ThetaE2m10mWinds))
    {
        let (u10, v10) = surface_winds_10m_ms.ok_or(
            "store render of 'theta_e_2m_10m_winds' needs the stored 10 m u/v planes \
             for its barb overlay",
        )?;
        if u10.len() != cells || v10.len() != cells {
            return Err("store 10 m wind planes do not match the hour grid".into());
        }
        computed.surface_u10_ms = Some(crop_plane(u10));
        computed.surface_v10_ms = Some(crop_plane(v10));
    }

    let render_overrides = || DerivedRenderOverrides {
        output_suffix: request.output_suffix.as_deref(),
        subtitle_left: request.subtitle_left_override.as_deref(),
        subtitle_right: request.subtitle_right_override.as_deref(),
    };
    let render_one = |recipe: &DerivedRecipe| -> Result<DerivedRenderedRecipe, String> {
        if recipe.is_heavy() {
            let store_grid = by_slug[recipe.slug()];
            let field = WeatherPanelField::new(
                weather_product_for_heavy_recipe(*recipe).map_err(|err| err.to_string())?,
                store_grid.units.clone(),
                crop_plane(&store_grid.values),
            );
            render_derived_heavy_recipe(
                request,
                *recipe,
                &field,
                &grid,
                projection,
                &projected,
                &request.date_yyyymmdd,
                cycle_utc,
                request.forecast_hour,
                request.source,
                request.model,
                input_fetch_keys.clone(),
                render_overrides(),
            )
            .map_err(|err| err.to_string())
        } else {
            render_derived_output_recipe(
                request,
                *recipe,
                &grid,
                projection,
                &projected,
                &request.date_yyyymmdd,
                cycle_utc,
                request.forecast_hour,
                request.source,
                request.model,
                &computed,
                input_fetch_keys.clone(),
                render_overrides(),
            )
            .map_err(|err| err.to_string())
        }
    };

    let worker_count = derived_render_worker_count(recipes.len());
    let rendered = if worker_count <= 1 {
        recipes
            .iter()
            .map(render_one)
            .collect::<Result<Vec<_>, _>>()
    } else {
        rayon::ThreadPoolBuilder::new()
            .num_threads(worker_count)
            .thread_name(|index| format!("rustwx-derived-render-{index}"))
            .build()
            .map_err(|err| err.to_string())?
            .install(|| {
                recipes
                    .par_iter()
                    .map(render_one)
                    .collect::<Result<Vec<_>, _>>()
            })
    }
    .map_err(std::io::Error::other)?;
    Ok(rendered)
}

fn store_render_projected_crop_pad_cells(bounds: (f64, f64, f64, f64)) -> usize {
    std::env::var("RUSTWX_STORE_RENDER_PROJECTED_CROP_PAD_CELLS")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or_else(|| {
            if is_straight_western_store_render_domain(bounds) {
                192
            } else {
                2
            }
        })
}

fn is_straight_western_store_render_domain(bounds: (f64, f64, f64, f64)) -> bool {
    let west = normalize_longitude_for_store_bounds(bounds.0);
    let east = normalize_longitude_for_store_bounds(bounds.1);
    if west > east {
        return false;
    }
    let lat_span = (bounds.3 - bounds.2).abs();
    let lon_span = longitude_store_bounds_span_deg(bounds);
    bounds.2 >= 25.0
        && bounds.3 <= 55.0
        && west >= -130.0
        && west <= -115.0
        && east >= -123.0
        && east <= -104.0
        && lat_span >= 4.0
        && lon_span <= 28.0
}

fn longitude_store_bounds_span_deg(bounds: (f64, f64, f64, f64)) -> f64 {
    let raw_span = (bounds.1 - bounds.0).abs();
    if raw_span >= 359.0 {
        return raw_span.min(360.0);
    }

    let west = normalize_longitude_for_store_bounds(bounds.0);
    let east = normalize_longitude_for_store_bounds(bounds.1);
    if west <= east {
        east - west
    } else {
        east + 360.0 - west
    }
}

fn normalize_longitude_for_store_bounds(lon: f64) -> f64 {
    let mut lon = lon % 360.0;
    if lon > 180.0 {
        lon -= 360.0;
    } else if lon <= -180.0 {
        lon += 360.0;
    }
    lon
}

fn derived_render_worker_count(recipe_count: usize) -> usize {
    if recipe_count <= 1 {
        return 1;
    }

    let override_threads = std::env::var("RUSTWX_DERIVED_RENDER_THREADS")
        .ok()
        .or_else(|| std::env::var("RUSTWX_RENDER_THREADS").ok())
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0);

    std::thread::available_parallelism()
        .map(|count| override_threads.unwrap_or((count.get() / 2).max(1)))
        .unwrap_or(1)
        .min(recipe_count)
}

/// Place one store-read plane into the computed-fields slot the render
/// builder consumes for this recipe — the inverse of
/// `derived_query_field_from_computed`'s recipe -> field mapping.
pub(super) fn assign_store_values(
    computed: &mut DerivedComputedFields,
    recipe: DerivedRecipe,
    values: Vec<f64>,
) -> Result<(), Box<dyn std::error::Error>> {
    let slot = match recipe {
        DerivedRecipe::Sbcape => &mut computed.sbcape_jkg,
        DerivedRecipe::Sbcin => &mut computed.sbcin_jkg,
        DerivedRecipe::Sblcl => &mut computed.sblcl_m,
        DerivedRecipe::Mlcape => &mut computed.mlcape_jkg,
        DerivedRecipe::Mlcin => &mut computed.mlcin_jkg,
        DerivedRecipe::Mucape => &mut computed.mucape_jkg,
        DerivedRecipe::Mucin => &mut computed.mucin_jkg,
        DerivedRecipe::Dcape => &mut computed.dcape_jkg,
        DerivedRecipe::ThetaE2m10mWinds => &mut computed.theta_e_2m_k,
        DerivedRecipe::Vpd2m => &mut computed.vpd_2m_hpa,
        DerivedRecipe::DewpointDepression2m => &mut computed.dewpoint_depression_2m_c,
        DerivedRecipe::Wetbulb2m => &mut computed.wetbulb_2m_c,
        DerivedRecipe::FireWeatherComposite => &mut computed.fire_weather_composite,
        DerivedRecipe::ApparentTemperature2m => &mut computed.apparent_temperature_2m_c,
        DerivedRecipe::HeatIndex2m => &mut computed.heat_index_2m_c,
        DerivedRecipe::WindChill2m => &mut computed.wind_chill_2m_c,
        DerivedRecipe::LiftedIndex => &mut computed.lifted_index_c,
        DerivedRecipe::LapseRate700500 => &mut computed.lapse_rate_700_500_cpkm,
        DerivedRecipe::LapseRate03km => &mut computed.lapse_rate_0_3km_cpkm,
        DerivedRecipe::BulkShear01km => &mut computed.shear_01km_kt,
        DerivedRecipe::BulkShear06km => &mut computed.shear_06km_kt,
        DerivedRecipe::Srh01km => &mut computed.srh_01km_m2s2,
        DerivedRecipe::Srh03km => &mut computed.srh_03km_m2s2,
        DerivedRecipe::Ehi01km => &mut computed.ehi_01km,
        DerivedRecipe::Ehi03km => &mut computed.ehi_03km,
        DerivedRecipe::StpFixed => &mut computed.stp_fixed,
        DerivedRecipe::ScpMu03km06kmProxy => &mut computed.scp_mu_03km_06km_proxy,
        DerivedRecipe::TemperatureAdvection700mb => &mut computed.temperature_advection_700mb_cph,
        DerivedRecipe::TemperatureAdvection850mb => &mut computed.temperature_advection_850mb_cph,
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
        | DerivedRecipe::EcapeStp => {
            return Err(format!(
                "heavy recipe '{}' renders from a WeatherPanelField, not the computed slots",
                recipe.slug()
            )
            .into());
        }
    };
    *slot = Some(values);
    Ok(())
}

/// The `WeatherProduct` each heavy recipe's panel field carries — the same
/// constructors `compute_ecape_map_fields_with_prepared_volume` uses, whose
/// product slug equals the recipe slug (pinned by test below).
pub(super) fn weather_product_for_heavy_recipe(
    recipe: DerivedRecipe,
) -> Result<rustwx_render::WeatherProduct, Box<dyn std::error::Error>> {
    use rustwx_render::WeatherProduct;
    Ok(match recipe {
        DerivedRecipe::Sbecape => WeatherProduct::Sbecape,
        DerivedRecipe::Mlecape => WeatherProduct::Mlecape,
        DerivedRecipe::Muecape => WeatherProduct::Muecape,
        DerivedRecipe::SbEcapeDerivedCapeRatio => WeatherProduct::SbEcapeDerivedCapeRatio,
        DerivedRecipe::MlEcapeDerivedCapeRatio => WeatherProduct::MlEcapeDerivedCapeRatio,
        DerivedRecipe::MuEcapeDerivedCapeRatio => WeatherProduct::MuEcapeDerivedCapeRatio,
        DerivedRecipe::SbEcapeNativeCapeRatio => WeatherProduct::SbEcapeNativeCapeRatio,
        DerivedRecipe::MlEcapeNativeCapeRatio => WeatherProduct::MlEcapeNativeCapeRatio,
        DerivedRecipe::MuEcapeNativeCapeRatio => WeatherProduct::MuEcapeNativeCapeRatio,
        DerivedRecipe::Sbncape => WeatherProduct::Sbncape,
        DerivedRecipe::Sbecin => WeatherProduct::Sbecin,
        DerivedRecipe::Mlecin => WeatherProduct::Mlecin,
        DerivedRecipe::EcapeScp => WeatherProduct::EcapeScpExperimental,
        DerivedRecipe::EcapeEhi01km => WeatherProduct::EcapeEhi01kmExperimental,
        DerivedRecipe::EcapeEhi03km => WeatherProduct::EcapeEhi03kmExperimental,
        DerivedRecipe::EcapeStp => WeatherProduct::EcapeStpExperimental,
        other => {
            return Err(format!(
                "recipe '{}' is not a heavy ECAPE-class recipe",
                other.slug()
            )
            .into());
        }
    })
}

#[cfg(test)]
mod tests {
    use super::super::store::{store_derived_recipe_slugs, store_heavy_recipe_slugs};
    use super::*;

    /// Every non-heavy store slug has a computed-fields slot, and assigning
    /// it round-trips through the same field the render builder reads
    /// (`derived_query_field_from_computed` is its forward direction).
    #[test]
    fn assign_covers_every_non_heavy_store_slug() {
        for slug in store_derived_recipe_slugs() {
            let recipe = DerivedRecipe::parse(slug).expect("store slug parses");
            let mut computed = DerivedComputedFields::default();
            assign_store_values(&mut computed, recipe, vec![1.0, 2.0])
                .unwrap_or_else(|err| panic!("assign '{slug}': {err}"));
            let query =
                super::super::query::derived_query_field_from_computed(2, 1, recipe, &mut computed)
                    .unwrap_or_else(|err| panic!("query '{slug}': {err}"));
            assert_eq!(query.values, vec![1.0, 2.0], "slot mismatch for '{slug}'");
        }
    }

    /// Heavy recipes refuse the computed-slot path: they render from a
    /// WeatherPanelField like the ECAPE lane.
    #[test]
    fn assign_rejects_heavy_recipes() {
        for slug in store_heavy_recipe_slugs() {
            let recipe = DerivedRecipe::parse(slug).expect("heavy slug parses");
            let mut computed = DerivedComputedFields::default();
            assert!(
                assign_store_values(&mut computed, recipe, vec![0.0]).is_err(),
                "heavy '{slug}' must not assign a computed slot"
            );
        }
    }

    /// The heavy product mapping reproduces the ECAPE lane's field identity:
    /// product slug == recipe slug for all 16 heavy store recipes, so
    /// titles, scales, and artifact naming match the GRIB lane exactly.
    #[test]
    fn heavy_product_mapping_pins_slug_identity() {
        for slug in store_heavy_recipe_slugs() {
            let recipe = DerivedRecipe::parse(slug).expect("heavy slug parses");
            let product = weather_product_for_heavy_recipe(recipe)
                .unwrap_or_else(|err| panic!("mapping '{slug}': {err}"));
            assert_eq!(
                product.slug(),
                slug,
                "WeatherProduct slug must equal the heavy recipe slug"
            );
        }
        let non_heavy = DerivedRecipe::parse("sbcape").unwrap();
        assert!(weather_product_for_heavy_recipe(non_heavy).is_err());
    }
}
