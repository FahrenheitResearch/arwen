//! Store-viewer style resolver: map one STORED variable (its name + the
//! selector JSON the store carries for it) to the production render styling
//! its plot counterpart uses — the same `ColorScale`, colormap build
//! options, tick step, legend mode, unit conversion, and title the PNG
//! lanes render with.
//!
//! Identity with production is by construction, not by a parallel table:
//!
//! * derived/heavy slugs resolve through
//!   [`crate::derived::derived_store_variable_style`], which builds a REAL
//!   render request via the same builders the render lanes run and reads
//!   the styling off it;
//! * direct planes resolve their recipe by reverse-matching the stored
//!   `FieldSelector` against the supported recipe catalog (the same
//!   resolution `store_render` performs forward) and then call the direct
//!   lane's own scale/controls/conversion functions
//!   ([`crate::plot_design::operational_fill_scale_for_recipe`],
//!   [`crate::direct::direct_recipe_render_controls`],
//!   [`crate::direct::direct_fill_unit_conversion`]);
//! * the trailing windowed-source planes (`uh_2to5km_max_1h`,
//!   `wind_speed_10m_max_1h`) mirror `build_windowed_render_request`.
//!
//! Variables with NO production fill counterpart (u/v wind components,
//! geopotential height planes — production only contours heights —
//! `mslp` — production contours mslp and fills the companion 10 m wind
//! speed — `surface_pressure`, `orography`, 3D volumes) resolve to
//! `None`: the viewer keeps its clearly-labeled generic ramp for those.

use rustwx_core::{CanonicalField, FieldSelector, ModelId, VerticalSelector};
use rustwx_models::{PlotRecipe, built_in_plot_recipes};
use rustwx_render::{
    ColorScale, ColormapBuildOptions, DiscreteColorScale, ExtendMode, LegendControls, LegendMode,
    MapRenderRequest, ProductVisualMode, RenderDensity, StaticPlotStyle, WeatherProduct,
};
use std::collections::HashSet;

use crate::direct::{
    direct_fill_unit_conversion, direct_recipe_render_controls, supported_direct_recipe_slugs,
};
use crate::windowed::HrrrWindowedProduct;

/// The unit conversion applied to raw stored values before the color scale
/// — mirrors the direct lane's `convert_filled_field` arithmetic exactly
/// (same f32 expressions), so converted values color identically to the
/// production fill.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnitConvert {
    None,
    /// 2 m temperature/dewpoint: `(K - 273.15) * 9/5 + 32`.
    KelvinToFahrenheit,
    /// Isobaric temperature/dewpoint: `K - 273.15`.
    KelvinToCelsius,
    /// MSLP: `Pa * 0.01`.
    PaToHpa,
    /// QPF / precipitable water: `mm / 25.4` (kg/m^2 == mm of water).
    MmToInches,
    /// Visibility: `m * 0.0006213712`.
    MetersToMiles,
    /// Absolute vorticity: `s^-1 * 1e5`.
    PerSecondToE5PerSecond,
    /// Wind speed/gust: `m/s * 1.9438445`.
    MsToKnots,
    /// Near-surface smoke: `kg/m^3 * 1e9`.
    KgM3ToUgM3,
    /// Column smoke: `kg/m^2 * 1e6`.
    KgM2ToMgM2,
}

impl UnitConvert {
    /// Convert one raw stored value into display units with the SAME f32
    /// arithmetic the direct render lane applies before its color scale.
    pub fn apply(self, value: f32) -> f32 {
        match self {
            Self::None => value,
            Self::KelvinToFahrenheit => (value - 273.15) * 9.0 / 5.0 + 32.0,
            Self::KelvinToCelsius => value - 273.15,
            Self::PaToHpa => value * 0.01,
            Self::MmToInches => value / 25.4,
            Self::MetersToMiles => value * 0.000_621_371_2,
            Self::PerSecondToE5PerSecond => value * 100_000.0,
            Self::MsToKnots => value * 1.943_844_5,
            Self::KgM3ToUgM3 => value * 1_000_000_000.0,
            Self::KgM2ToMgM2 => value * 1_000_000.0,
        }
    }

    pub fn is_none(self) -> bool {
        matches!(self, Self::None)
    }
}

/// The production styling for one stored variable: everything a viewer
/// needs to color pixels and draw a legend that match the variable's plot
/// counterpart. Build the colormap with
/// `rustwx_render::build_colormap(&style.scale, style.colormap_options)`,
/// color CONVERTED values with `cmap.map(...)` (NaN and masked values map
/// to transparent), and label ticks from
/// `rustwx_render::colorbar_ticks(&cmap, style.cbar_tick_step)` +
/// `rustwx_render::format_tick`.
#[derive(Debug, Clone, PartialEq)]
pub struct StoreVariableStyle {
    /// Production product title (recipe/preset title, no time suffixes).
    pub title: String,
    /// Display units AFTER `convert` (the units the legend is labeled in).
    pub display_units: String,
    /// Conversion from raw stored values to display units.
    pub convert: UnitConvert,
    /// The production color scale (apply to CONVERTED values).
    pub scale: ColorScale,
    /// The colormap build options the production render uses for this
    /// variable's lane, already filtered through the active
    /// `StaticPlotStyle` exactly as the renderer does.
    pub colormap_options: ColormapBuildOptions,
    /// The production colorbar tick step (`None` = auto "nice" ticks).
    pub cbar_tick_step: Option<f64>,
    /// Stepped vs smooth-ramp colorbar painting (also carried inside
    /// `colormap_options.legend.mode`).
    pub legend_mode: LegendMode,
}

/// One operational color-table template a UI can clone into a user-editable
/// table. The style is copied from the same production code path used by
/// rendered products and the native viewer.
#[derive(Debug, Clone, PartialEq)]
pub struct StoreVariableStyleTemplate {
    pub id: String,
    pub slug: String,
    pub label: String,
    pub category: String,
    pub style: StoreVariableStyle,
}

/// Resolve the production styling for the stored variable `var_name`
/// carrying `stored_selector` (the store's per-variable selector JSON:
/// either a `FieldSelector` or a `{"derived": slug}` marker) and
/// `stored_units`. Returns `None` for variables with no production fill
/// counterpart — the caller should fall back to a clearly-labeled generic
/// ramp.
pub fn operational_style_for_store_variable(
    var_name: &str,
    stored_selector: &serde_json::Value,
    stored_units: &str,
    model: ModelId,
) -> Option<StoreVariableStyle> {
    if let Some(slug) = stored_selector.get("derived").and_then(|v| v.as_str()) {
        let slug = normalize_wrf_store_slug(slug);
        return derived_style(&slug, stored_units)
            .or_else(|| weather_product_style(&slug, stored_units));
    }
    let selector: FieldSelector = serde_json::from_value(stored_selector.clone()).ok()?;

    // Trailing (h-1)->h window planes: their formal plot counterpart is the
    // windowed product family, mirroring `build_windowed_render_request`
    // (`1h_qpf` is deliberately NOT a direct recipe — it routes to the
    // windowed `qpf_1h` product, see `LEGACY_PRODUCT_ALIASES`).
    match var_name {
        "uh_2to5km_max_1h" => return Some(windowed_uh_style(stored_units)),
        "wind_speed_10m_max_1h" => return Some(windowed_wind10m_style()),
        "apcp_1h" => return Some(windowed_qpf_1h_style()),
        _ => {}
    }

    // Production never color-fills ISOBARIC geopotential height (height
    // recipes are contour-only / fill companion wind speed).  The surface
    // member of the same canonical field is orography, which now has a
    // production fill of its own, so it keeps its plot identity instead of
    // the generic ramp.
    if selector.field == CanonicalField::GeopotentialHeight
        && !matches!(selector.vertical, VerticalSelector::Surface)
    {
        return None;
    }

    // Production never color-fills mslp values either: the `mslp_10m_winds`
    // plot fills the companion 10 m wind speed (kt) and only CONTOURS mslp,
    // so no production colorbar exists for the stored pressure plane.
    // Claiming that plot's identity over the catalog WeatherPressure scale
    // would show legend values (960..1044 hPa) that match no production
    // colorbar — keep the clearly-labeled generic ramp instead.
    if selector.field == CanonicalField::PressureReducedToMeanSeaLevel {
        return None;
    }

    let recipe = direct_recipe_for_selector(var_name, selector, model)?;
    let scale = crate::plot_design::operational_fill_scale_for_recipe(recipe, selector);
    let (convert, units_override) = direct_fill_unit_conversion(recipe, selector);
    let (legend_mode_override, cbar_tick_step) = direct_recipe_render_controls(recipe, selector);
    let (render_density, mut legend) = operational_request_chrome();
    if let Some(mode) = legend_mode_override {
        legend.mode = mode;
    }
    Some(StoreVariableStyle {
        title: recipe.title.to_string(),
        display_units: units_override.map_or_else(|| stored_units.to_string(), str::to_string),
        convert,
        scale,
        colormap_options: filtered_options(render_density, legend),
        cbar_tick_step,
        legend_mode: legend.mode,
    })
}

/// Derived/heavy slugs: styling read off a real render request built by the
/// production builders (see `derived_store_variable_style`). Stored grids
/// already carry display units, so no conversion applies.
fn derived_style(slug: &str, stored_units: &str) -> Option<StoreVariableStyle> {
    let lane = crate::derived::derived_store_variable_style(slug).ok()?;
    Some(StoreVariableStyle {
        title: lane.title,
        display_units: stored_units.to_string(),
        convert: UnitConvert::None,
        scale: lane.scale,
        colormap_options: filtered_options(lane.render_density, lane.legend),
        cbar_tick_step: lane.cbar_tick_step,
        legend_mode: lane.legend.mode,
    })
}

/// Every production color table currently reachable through the operational
/// render/style system, packaged as cloneable user-table templates.
pub fn operational_style_templates(model: ModelId) -> Vec<StoreVariableStyleTemplate> {
    let mut out = Vec::new();
    let mut seen = HashSet::<String>::new();

    let supported = supported_direct_recipe_slugs(model);
    for recipe in built_in_plot_recipes()
        .iter()
        .filter(|recipe| supported.iter().any(|slug| slug == recipe.slug))
    {
        let Some(selector) = recipe.filled.selector else {
            continue;
        };
        if selector.field == CanonicalField::GeopotentialHeight
            || selector.field == CanonicalField::PressureReducedToMeanSeaLevel
        {
            continue;
        }
        let Ok(selector_json) = serde_json::to_value(selector) else {
            continue;
        };
        if let Some(mut style) = operational_style_for_store_variable(
            recipe.slug,
            &selector_json,
            selector.native_units(),
            model,
        ) {
            style.title = recipe.title.to_string();
            push_template(
                &mut out,
                &mut seen,
                "direct",
                recipe.slug,
                recipe.title,
                style,
            );
        }
    }

    for entry in crate::derived::supported_derived_recipe_inventory() {
        if let Some(mut style) = operational_style_for_store_variable(
            entry.slug,
            &serde_json::json!({ "derived": entry.slug }),
            template_units_for_slug(entry.slug),
            model,
        ) {
            style.title = entry.title.to_string();
            push_template(
                &mut out,
                &mut seen,
                if entry.heavy { "heavy" } else { "derived" },
                entry.slug,
                entry.title,
                style,
            );
        }
    }

    for product in ALL_WEATHER_PRODUCTS {
        if let Some(style) =
            weather_product_style(product.slug(), template_units_for_slug(product.slug()))
        {
            push_template(
                &mut out,
                &mut seen,
                "weather",
                product.slug(),
                product.display_title(),
                style,
            );
        }
    }

    for &product in HrrrWindowedProduct::supported_products() {
        if let Some(style) = windowed_product_style(product) {
            push_template(
                &mut out,
                &mut seen,
                "windowed",
                product.slug(),
                product.title(),
                style,
            );
        }
    }

    out.sort_by(|a, b| {
        a.category
            .cmp(&b.category)
            .then_with(|| a.label.cmp(&b.label))
            .then_with(|| a.slug.cmp(&b.slug))
    });
    out
}

fn push_template(
    out: &mut Vec<StoreVariableStyleTemplate>,
    seen: &mut HashSet<String>,
    category: &str,
    slug: &str,
    label: &str,
    style: StoreVariableStyle,
) {
    let id = format!("{category}:{slug}");
    if seen.insert(id.clone()) {
        out.push(StoreVariableStyleTemplate {
            id,
            slug: slug.to_string(),
            label: label.to_string(),
            category: category.to_string(),
            style,
        });
    }
}

/// Local WRF processing initially stored `wrf_*` derived markers, while the
/// existing operational style catalog uses model-agnostic product slugs.
/// Normalize those aliases before resolving a derived/weather palette.
fn normalize_wrf_store_slug(slug: &str) -> String {
    let slug = slug.strip_prefix("wrf_").unwrap_or(slug);
    match slug {
        "srh1" => "srh_0_1km".to_string(),
        "srh3" => "srh_0_3km".to_string(),
        "shear_0_1km" => "bulk_shear_0_1km".to_string(),
        "shear_0_6km" => "bulk_shear_0_6km".to_string(),
        "uhel" => "uhel".to_string(),
        other => other.to_string(),
    }
}

/// WRF-core exposes a few already-computed severe parameters that are not
/// HRRR-derived recipe slugs but do have first-class WeatherProduct palettes
/// (STP/SCP/EHI/LCL/LFC/EL/UH, etc.). Use the same weather request builder
/// as the production weather lanes for those.
fn weather_product_style(slug: &str, stored_units: &str) -> Option<StoreVariableStyle> {
    let product = WeatherProduct::from_product_name(slug)?;
    let mut request = MapRenderRequest::for_core_weather_product(probe_core_field(), product);
    apply_probe_static_design(&mut request);
    Some(StoreVariableStyle {
        title: product.display_title().to_string(),
        display_units: stored_units.to_string(),
        convert: UnitConvert::None,
        scale: request.scale,
        colormap_options: filtered_options(request.render_density, request.legend),
        cbar_tick_step: request.cbar_tick_step,
        legend_mode: request.legend.mode,
    })
}

/// `uh_2to5km_max_1h`: the windowed UH family request —
/// `for_core_weather_product(WeatherProduct::Uh)` + static map design.
fn windowed_uh_style(stored_units: &str) -> StoreVariableStyle {
    let mut request =
        MapRenderRequest::for_core_weather_product(probe_core_field(), WeatherProduct::Uh);
    apply_probe_static_design(&mut request);
    StoreVariableStyle {
        title: HrrrWindowedProduct::Uh25km1h.title().to_string(),
        display_units: stored_units.to_string(),
        convert: UnitConvert::None,
        scale: request.scale,
        colormap_options: filtered_options(request.render_density, request.legend),
        cbar_tick_step: request.cbar_tick_step,
        legend_mode: request.legend.mode,
    }
}

/// `wind_speed_10m_max_1h`: the windowed 10 m wind family request —
/// `from_core_field(windowed_product_scale(...))` + static map design. The
/// stored plane is m/s; the windowed lane displays knots.
fn windowed_wind10m_style() -> StoreVariableStyle {
    let scale = crate::windowed_decoder::windowed_product_scale(HrrrWindowedProduct::Wind10m1hMax);
    let mut request = MapRenderRequest::from_core_field(probe_core_field(), scale);
    apply_probe_static_design(&mut request);
    StoreVariableStyle {
        title: HrrrWindowedProduct::Wind10m1hMax.title().to_string(),
        display_units: "kt".to_string(),
        convert: UnitConvert::MsToKnots,
        scale: request.scale,
        colormap_options: filtered_options(request.render_density, request.legend),
        cbar_tick_step: request.cbar_tick_step,
        legend_mode: request.legend.mode,
    }
}

/// `apcp_1h`: the trailing 1 h QPF window — the windowed `qpf_1h` product
/// (its legacy `1h_qpf` recipe slug deliberately aliases to the windowed
/// lane). Stored kg/m^2 == mm; displayed in inches like all QPF products.
fn windowed_qpf_1h_style() -> StoreVariableStyle {
    let scale = crate::windowed_decoder::windowed_product_scale(HrrrWindowedProduct::Qpf1h);
    let mut request = MapRenderRequest::from_core_field(probe_core_field(), scale);
    apply_probe_static_design(&mut request);
    StoreVariableStyle {
        title: HrrrWindowedProduct::Qpf1h.title().to_string(),
        display_units: "in".to_string(),
        convert: UnitConvert::MmToInches,
        scale: request.scale,
        colormap_options: filtered_options(request.render_density, request.legend),
        cbar_tick_step: request.cbar_tick_step,
        legend_mode: request.legend.mode,
    }
}

/// Reverse-resolve one stored direct plane to its plot recipe: the first
/// supported recipe whose `filled.selector` equals the stored selector —
/// except `apcp_run_total`, whose store name pins the run-total window
/// identity for the shared plain TotalPrecipitation selector.
fn windowed_product_style(product: HrrrWindowedProduct) -> Option<StoreVariableStyle> {
    let scale = crate::windowed_decoder::windowed_product_scale(product);
    let mut request = MapRenderRequest::from_core_field(probe_core_field(), scale);
    apply_probe_static_design(&mut request);
    let (convert, display_units) = windowed_product_units(product);
    Some(StoreVariableStyle {
        title: product.title().to_string(),
        display_units: display_units.to_string(),
        convert,
        scale: request.scale,
        colormap_options: filtered_options(request.render_density, request.legend),
        cbar_tick_step: request.cbar_tick_step,
        legend_mode: request.legend.mode,
    })
}

fn windowed_product_units(product: HrrrWindowedProduct) -> (UnitConvert, &'static str) {
    let slug = product.slug();
    if slug.starts_with("qpf_") {
        (UnitConvert::MmToInches, "in")
    } else if slug.starts_with("10m_wind_") {
        (UnitConvert::MsToKnots, "kt")
    } else if slug.starts_with("2m_temp_") || slug.starts_with("2m_dewpoint_") {
        (UnitConvert::KelvinToFahrenheit, "degF")
    } else if slug.starts_with("2m_rh_") {
        (UnitConvert::None, "%")
    } else if slug.starts_with("2m_vpd_") {
        (UnitConvert::None, "hPa")
    } else if slug.starts_with("uh_") {
        (UnitConvert::None, "m^2/s^2")
    } else {
        (UnitConvert::None, "")
    }
}

const ALL_WEATHER_PRODUCTS: &[WeatherProduct] = &[
    WeatherProduct::Sbcape,
    WeatherProduct::Mlcape,
    WeatherProduct::Mucape,
    WeatherProduct::Sbecape,
    WeatherProduct::Mlecape,
    WeatherProduct::Muecape,
    WeatherProduct::SbEcapeDerivedCapeRatio,
    WeatherProduct::MlEcapeDerivedCapeRatio,
    WeatherProduct::MuEcapeDerivedCapeRatio,
    WeatherProduct::SbEcapeNativeCapeRatio,
    WeatherProduct::MlEcapeNativeCapeRatio,
    WeatherProduct::MuEcapeNativeCapeRatio,
    WeatherProduct::Sbncape,
    WeatherProduct::Mlncape,
    WeatherProduct::Muncape,
    WeatherProduct::Sbcin,
    WeatherProduct::Mlcin,
    WeatherProduct::Mucin,
    WeatherProduct::Sbecin,
    WeatherProduct::Mlecin,
    WeatherProduct::Muecin,
    WeatherProduct::EcapeCape,
    WeatherProduct::EcapeCin,
    WeatherProduct::Lcl,
    WeatherProduct::Lfc,
    WeatherProduct::El,
    WeatherProduct::EcapeLfc,
    WeatherProduct::EcapeEl,
    WeatherProduct::Srh01km,
    WeatherProduct::Srh03km,
    WeatherProduct::Stp,
    WeatherProduct::StpFixed,
    WeatherProduct::StpEffective,
    WeatherProduct::Scp,
    WeatherProduct::Ehi,
    WeatherProduct::Tehi,
    WeatherProduct::Tts,
    WeatherProduct::VtpMod,
    WeatherProduct::Uh,
    WeatherProduct::EcapeScpExperimental,
    WeatherProduct::EcapeEhi01kmExperimental,
    WeatherProduct::EcapeEhi03kmExperimental,
    WeatherProduct::EcapeStpExperimental,
];

fn template_units_for_slug(slug: &str) -> &'static str {
    let slug = normalize_wrf_store_slug(slug);
    if slug.contains("cape") || slug.contains("cin") || slug.contains("ncape") || slug == "dcape" {
        "J/kg"
    } else if slug.contains("lcl") || slug.contains("lfc") || slug == "el" {
        "m"
    } else if slug.contains("srh") || slug.contains("uhel") || slug == "uh" {
        "m^2/s^2"
    } else if slug.contains("shear") {
        "kt"
    } else if slug.contains("stp")
        || slug.contains("scp")
        || slug.contains("ehi")
        || slug.contains("tts")
        || slug.contains("vtp")
        || slug.contains("ratio")
    {
        "dimensionless"
    } else if slug.contains("temperature")
        || slug.contains("apparent")
        || slug.contains("heat_index")
        || slug.contains("wind_chill")
        || slug.contains("wetbulb")
    {
        "degF"
    } else if slug.contains("vpd") {
        "hPa"
    } else {
        ""
    }
}

fn direct_recipe_for_selector(
    var_name: &str,
    selector: FieldSelector,
    model: ModelId,
) -> Option<&'static PlotRecipe> {
    direct_recipe_for_selector_with_supported(
        var_name,
        selector,
        &supported_direct_recipe_slugs(model),
    )
    .or_else(|| {
        (model == ModelId::WrfGdex).then(|| {
            direct_recipe_for_selector_with_supported(
                var_name,
                selector,
                &supported_direct_recipe_slugs(ModelId::Hrrr),
            )
        })?
    })
}

fn direct_recipe_for_selector_with_supported(
    var_name: &str,
    selector: FieldSelector,
    supported: &[String],
) -> Option<&'static PlotRecipe> {
    let is_supported = |slug: &str| supported.iter().any(|s| s == slug);
    if var_name == "apcp_run_total" {
        if let Some(recipe) =
            rustwx_models::plot_recipe("total_qpf").filter(|r| is_supported(r.slug))
        {
            return Some(recipe);
        }
    }
    built_in_plot_recipes()
        .iter()
        .find(|recipe| recipe.filled.selector == Some(selector) && is_supported(recipe.slug))
}

/// The render density + legend controls a single-product operational static
/// plot request ends with — read off a real request run through
/// `StaticPlotDesign` (the same code path every PNG lane applies), over a
/// regional domain like the production CONUS products.
fn operational_request_chrome() -> (RenderDensity, LegendControls) {
    let mut request = MapRenderRequest::from_core_field(
        probe_core_field(),
        ColorScale::Discrete(DiscreteColorScale {
            levels: vec![0.0, 1.0],
            colors: vec![rustwx_render::Color::rgba(0, 0, 0, 255)],
            extend: ExtendMode::Neither,
            mask_below: None,
        }),
    );
    apply_probe_static_design(&mut request);
    (request.render_density, request.legend)
}

/// Any non-global bounds select `apply_static_map_design`'s regional
/// branch, matching the production CONUS domains.
fn apply_probe_static_design(request: &mut MapRenderRequest) {
    crate::plot_design::StaticPlotDesign::new(
        (-125.0, -66.0, 24.0, 50.0),
        ProductVisualMode::FilledMeteorology,
    )
    .apply_to_request(request);
}

/// Filter the request's density through the active plot style exactly as
/// the renderer does when it builds the colormap
/// (`plot_style.render_density(request.render_density)`).
fn filtered_options(render_density: RenderDensity, legend: LegendControls) -> ColormapBuildOptions {
    ColormapBuildOptions {
        render_density: StaticPlotStyle::from_env().render_density(render_density),
        legend,
    }
}

fn probe_core_field() -> rustwx_core::Field2D {
    let shape = rustwx_core::GridShape::new(2, 2).expect("probe grid shape");
    let grid = rustwx_core::LatLonGrid::new(
        shape,
        vec![35.0, 35.0, 36.0, 36.0],
        vec![-100.0, -99.0, -100.0, -99.0],
    )
    .expect("probe grid");
    rustwx_core::Field2D::new(
        rustwx_core::ProductKey::named("style-probe"),
        "probe",
        grid,
        vec![0.0, 0.0, 0.0, 0.0],
    )
    .expect("probe field")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::derived::{store_derived_recipe_slugs, store_heavy_recipe_slugs};
    use rustwx_render::{LevelDensity, Rgba, build_colormap, colorbar_ticks};

    fn style_for(
        var_name: &str,
        selector: &FieldSelector,
        units: &str,
    ) -> Option<StoreVariableStyle> {
        operational_style_for_store_variable(
            var_name,
            &serde_json::to_value(selector).expect("selector json"),
            units,
            ModelId::Hrrr,
        )
    }

    fn derived_marker(slug: &str) -> serde_json::Value {
        serde_json::json!({ "derived": slug })
    }

    #[test]
    fn every_store_derived_and_heavy_slug_resolves() {
        for slug in store_derived_recipe_slugs()
            .into_iter()
            .chain(store_heavy_recipe_slugs())
        {
            let style = operational_style_for_store_variable(
                slug,
                &derived_marker(slug),
                "units",
                ModelId::Hrrr,
            )
            .unwrap_or_else(|| panic!("derived slug '{slug}' must resolve"));
            assert!(!style.title.is_empty(), "'{slug}' carries a title");
            assert!(
                style.convert.is_none(),
                "derived grids are stored in display units ('{slug}')"
            );
        }
    }

    #[test]
    fn sbcape_resolves_to_masked_cape_preset_with_production_ticks() {
        let style = operational_style_for_store_variable(
            "sbcape",
            &derived_marker("sbcape"),
            "J/kg",
            ModelId::Hrrr,
        )
        .expect("sbcape resolves");
        // The derived lane overrides the CAPE preset with mask_below 250
        // (apply_operational_raster_scale) and ticks every 500 J/kg.
        let discrete = style.scale.resolved_discrete();
        assert_eq!(discrete.mask_below, Some(250.0));
        assert_eq!(style.cbar_tick_step, Some(500.0));
        let cmap = build_colormap(&style.scale, style.colormap_options);
        let ticks = colorbar_ticks(&cmap, style.cbar_tick_step);
        assert_eq!(ticks.first().copied(), Some(discrete.levels[0]));
        assert!(ticks.windows(2).all(|w| (w[1] - w[0] - 500.0).abs() < 1e-9));
        // Masked (below 250) and NaN values are transparent, never clamped.
        assert_eq!(cmap.map(100.0), Rgba::TRANSPARENT);
        assert_eq!(cmap.map(f64::NAN), Rgba::TRANSPARENT);
    }

    #[test]
    fn wrf_prefixed_derived_markers_reuse_product_palettes() {
        let sbcape = operational_style_for_store_variable(
            "wrf_sbcape",
            &derived_marker("wrf_sbcape"),
            "J/kg",
            ModelId::Hrrr,
        )
        .expect("legacy wrf_sbcape resolves");
        assert_eq!(sbcape.title, "SBCAPE");
        assert_eq!(
            sbcape.scale.resolved_discrete().mask_below,
            Some(250.0),
            "wrf_sbcape must use the same CAPE masking as sbcape"
        );

        let srh = operational_style_for_store_variable(
            "wrf_srh1",
            &derived_marker("wrf_srh1"),
            "m2/s2",
            ModelId::Hrrr,
        )
        .expect("legacy wrf_srh1 resolves");
        assert_eq!(srh.title, "0-1 km SRH");
        assert_eq!(srh.cbar_tick_step, Some(50.0));

        let shear = operational_style_for_store_variable(
            "wrf_shear_0_6km",
            &derived_marker("wrf_shear_0_6km"),
            "m/s",
            ModelId::Hrrr,
        )
        .expect("legacy wrf_shear_0_6km resolves");
        assert_eq!(shear.title, "0-6 km Bulk Shear");

        let stp = operational_style_for_store_variable(
            "wrf_stp",
            &derived_marker("wrf_stp"),
            "dimensionless",
            ModelId::Hrrr,
        )
        .expect("legacy wrf_stp resolves via WeatherProduct");
        assert_eq!(stp.title, "STP");
        assert!(matches!(stp.scale, ColorScale::Weather(_)));
    }

    #[test]
    fn operational_template_catalog_exposes_common_starting_palettes() {
        let templates = operational_style_templates(ModelId::Hrrr);
        assert!(
            templates
                .iter()
                .any(|template| template.slug == "2m_temperature"),
            "template catalog must include the operational 2 m temperature palette"
        );
        assert!(
            templates
                .iter()
                .any(|template| template.slug == "2m_dewpoint"),
            "template catalog must include the operational dewpoint palette"
        );
        assert!(
            templates.iter().any(|template| template.slug == "stp"),
            "template catalog must include an STP palette"
        );
    }

    #[test]
    fn heavy_slugs_use_the_weather_preset_lane_without_densification() {
        let style = operational_style_for_store_variable(
            "sbecape",
            &derived_marker("sbecape"),
            "J/kg",
            ModelId::Hrrr,
        )
        .expect("sbecape resolves");
        assert!(
            matches!(style.scale, ColorScale::Weather(_)),
            "heavy lane keeps the Weather preset scale"
        );
        assert_eq!(style.cbar_tick_step, Some(500.0));
        assert_eq!(style.legend_mode, LegendMode::Stepped);
        // for_weather_product's reference-discrete defaults: no fill/palette
        // densification requested (the plot style may still bump it; compare
        // against the identically-filtered reference request).
        let reference = filtered_options(
            RenderDensity {
                fill: LevelDensity::default(),
                palette_multiplier: 1,
            },
            LegendControls {
                density: LevelDensity::default(),
                mode: LegendMode::Stepped,
            },
        );
        assert_eq!(style.colormap_options, reference);
    }

    #[test]
    fn direct_planes_resolve_with_production_conversions_and_controls() {
        let temp = style_for(
            "temperature_2m",
            &FieldSelector::height_agl(CanonicalField::Temperature, 2),
            "K",
        )
        .expect("temperature_2m resolves");
        assert_eq!(temp.convert, UnitConvert::KelvinToFahrenheit);
        assert_eq!(temp.display_units, "degF");
        assert_eq!(temp.cbar_tick_step, None);
        assert_eq!(temp.legend_mode, LegendMode::SmoothRamp);
        let discrete = temp.scale.resolved_discrete();
        assert_eq!(discrete.levels.first().copied(), Some(-60.0));
        assert_eq!(discrete.levels.last().copied(), Some(120.0));

        let dewpoint = style_for(
            "dewpoint_2m",
            &FieldSelector::height_agl(CanonicalField::Dewpoint, 2),
            "K",
        )
        .expect("dewpoint_2m resolves");
        assert_eq!(dewpoint.cbar_tick_step, Some(10.0));
        assert_eq!(dewpoint.legend_mode, LegendMode::Stepped);
        assert_eq!(
            dewpoint.colormap_options.legend.mode,
            LegendMode::Stepped,
            "the legend-mode override must reach the colormap options"
        );

        let rh = style_for(
            "rh_2m",
            &FieldSelector::height_agl(CanonicalField::RelativeHumidity, 2),
            "%",
        )
        .expect("rh_2m resolves");
        assert_eq!(rh.cbar_tick_step, Some(25.0));
        assert_eq!(rh.display_units, "%");
        assert_eq!(rh.convert, UnitConvert::None);

        let reflectivity = style_for(
            "composite_reflectivity",
            &FieldSelector::entire_atmosphere(CanonicalField::CompositeReflectivity),
            "dBZ",
        )
        .expect("composite_reflectivity resolves");
        let scale = reflectivity.scale.resolved_discrete();
        assert_eq!(scale.levels.first().copied(), Some(10.0));
        assert_eq!(scale.levels.last().copied(), Some(70.0));
        assert_eq!(scale.mask_below, Some(10.0));
    }

    #[test]
    fn wrf_gdex_direct_planes_resolve_standard_product_palettes() {
        let temp = operational_style_for_store_variable(
            "temperature_2m",
            &serde_json::to_value(FieldSelector::height_agl(CanonicalField::Temperature, 2))
                .expect("selector json"),
            "K",
            ModelId::WrfGdex,
        )
        .expect("WRF 2m temperature should use the standard temperature palette");
        assert_eq!(temp.convert, UnitConvert::KelvinToFahrenheit);
        assert_eq!(temp.display_units, "degF");
    }

    #[test]
    fn mslp_falls_back_to_the_generic_ramp() {
        // The production `mslp_10m_winds` plot fills the companion 10 m wind
        // speed (legend 10..60 kt) and only contours mslp — there is no
        // production colorbar for the stored pressure values, so claiming
        // production parity with ANY pressure-valued legend would be false.
        assert!(
            style_for(
                "mslp",
                &FieldSelector::mean_sea_level(CanonicalField::PressureReducedToMeanSeaLevel),
                "Pa",
            )
            .is_none(),
            "mslp must keep the generic ramp (no production fill counterpart)"
        );
    }

    #[test]
    fn windowed_source_planes_resolve_through_the_windowed_family() {
        let uh = style_for(
            "uh_2to5km_max_1h",
            &FieldSelector::height_layer_agl(CanonicalField::UpdraftHelicity, 2000, 5000),
            "m^2/s^2",
        )
        .expect("uh_2to5km_max_1h resolves");
        assert!(matches!(uh.scale, ColorScale::Weather(_)));
        assert_eq!(
            uh.cbar_tick_step,
            WeatherProduct::Uh.default_tick_step(),
            "windowed UH carries the UH product tick step"
        );

        let wind = style_for(
            "wind_speed_10m_max_1h",
            &FieldSelector::height_agl(CanonicalField::WindSpeed, 10),
            "m/s",
        )
        .expect("wind_speed_10m_max_1h resolves");
        assert_eq!(wind.convert, UnitConvert::MsToKnots);
        assert_eq!(wind.display_units, "kt");
        let scale = wind.scale.resolved_discrete();
        assert_eq!(scale.levels.first().copied(), Some(10.0));
        assert_eq!(scale.levels.last().copied(), Some(70.0));
    }

    #[test]
    fn qpf_planes_pin_their_window_identity_by_store_name() {
        let selector = FieldSelector::surface(CanonicalField::TotalPrecipitation);
        let total = style_for("apcp_run_total", &selector, "kg/m^2").expect("run total resolves");
        let hourly = style_for("apcp_1h", &selector, "kg/m^2").expect("1h resolves");
        assert_eq!(total.convert, UnitConvert::MmToInches);
        assert_eq!(total.display_units, "in");
        assert_eq!(
            total.scale, hourly.scale,
            "both windows share the QPF scale"
        );
        assert_eq!(total.title, "Total QPF");
        assert_eq!(hourly.title, "1-h QPF", "titles pin the window identity");
    }

    #[test]
    fn unmapped_variables_fall_back_to_none() {
        // Barb inputs, compute inputs, contour-only heights, and the
        // contour-only mslp plane have no production fill counterpart.
        for (name, selector) in [
            (
                "u_10m",
                FieldSelector::height_agl(CanonicalField::UWind, 10),
            ),
            (
                "v_10m",
                FieldSelector::height_agl(CanonicalField::VWind, 10),
            ),
            (
                "surface_pressure",
                FieldSelector::surface(CanonicalField::Pressure),
            ),
            (
                "mslp",
                FieldSelector::mean_sea_level(CanonicalField::PressureReducedToMeanSeaLevel),
            ),
            (
                "geopotential_height_500hpa",
                FieldSelector::isobaric(CanonicalField::GeopotentialHeight, 500),
            ),
            (
                "u_wind_500hpa",
                FieldSelector::isobaric(CanonicalField::UWind, 500),
            ),
        ] {
            assert!(
                style_for(name, &selector, "units").is_none(),
                "'{name}' must keep the generic ramp"
            );
        }
        // `orography` used to be on that list, for want of a production
        // colorbar.  It has one now, and the browser must show it rather
        // than the generic ramp -- this assertion is what stops the
        // isobaric-height exclusion from silently swallowing it again.
        let terrain = style_for(
            "orography",
            &FieldSelector::surface(CanonicalField::GeopotentialHeight),
            "m",
        )
        .expect("orography now has a production fill");
        assert_eq!(terrain.title, "Terrain Height");
        assert_eq!(terrain.display_units, "m");

        // Unknown derived markers fall back too.
        assert!(
            operational_style_for_store_variable(
                "mystery",
                &serde_json::json!({ "derived": "not_a_recipe" }),
                "units",
                ModelId::Hrrr,
            )
            .is_none()
        );
    }

    #[test]
    fn shared_filled_selectors_agree_on_style_and_scale() {
        // `direct_recipe_for_selector` resolves a stored selector by
        // FIRST-MATCH over the supported recipe catalog. That is only safe
        // while every pair of supported recipes sharing a filled selector
        // agrees on render style and operational fill scale — otherwise the
        // viewer would silently pick whichever recipe happens to come first.
        // Pin the invariant over ALL pairs (zero offending pairs required;
        // the sweep guards future catalog additions even if no pair exists
        // today).
        let supported = supported_direct_recipe_slugs(ModelId::Hrrr);
        let recipes: Vec<&PlotRecipe> = built_in_plot_recipes()
            .iter()
            .filter(|recipe| supported.iter().any(|slug| slug == recipe.slug))
            .collect();
        assert!(
            !recipes.is_empty(),
            "HRRR must support at least one direct recipe"
        );
        let mut shared_pairs = 0usize;
        for (i, a) in recipes.iter().enumerate() {
            for b in &recipes[i + 1..] {
                let (Some(selector), Some(other)) = (a.filled.selector, b.filled.selector) else {
                    continue;
                };
                if selector != other {
                    continue;
                }
                shared_pairs += 1;
                assert_eq!(
                    a.style, b.style,
                    "supported recipes '{}' and '{}' share filled selector {selector:?} but \
                     disagree on render style — first-match resolution in \
                     direct_recipe_for_selector is no longer safe",
                    a.slug, b.slug,
                );
                assert_eq!(
                    crate::plot_design::operational_fill_scale_for_recipe(a, selector),
                    crate::plot_design::operational_fill_scale_for_recipe(b, selector),
                    "supported recipes '{}' and '{}' share filled selector {selector:?} but \
                     disagree on operational fill scale — first-match resolution in \
                     direct_recipe_for_selector is no longer safe",
                    a.slug,
                    b.slug,
                );
            }
        }
        eprintln!(
            "shared_filled_selectors_agree_on_style_and_scale: checked {shared_pairs} \
             shared-selector pair(s) across {} supported recipes",
            recipes.len()
        );
    }

    #[test]
    fn unit_conversions_match_the_direct_lane_arithmetic() {
        assert_eq!(UnitConvert::KelvinToFahrenheit.apply(273.15), 32.0);
        assert_eq!(
            UnitConvert::KelvinToFahrenheit.apply(300.0),
            (300.0 - 273.15) * 9.0 / 5.0 + 32.0
        );
        assert_eq!(UnitConvert::KelvinToCelsius.apply(273.15), 0.0);
        assert_eq!(UnitConvert::PaToHpa.apply(101_325.0), 101_325.0 * 0.01);
        assert_eq!(UnitConvert::MmToInches.apply(25.4), 1.0);
        assert_eq!(UnitConvert::MsToKnots.apply(10.0), 10.0_f32 * 1.943_844_5);
        assert_eq!(UnitConvert::KgM3ToUgM3.apply(1.0e-9), 1.0);
        assert!(UnitConvert::KelvinToFahrenheit.apply(f32::NAN).is_nan());
    }
}
