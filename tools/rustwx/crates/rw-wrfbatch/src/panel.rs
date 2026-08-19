//! One 2-D plane -> one production PNG, for planes that are not a stored
//! model variable.
//!
//! `store_render::render_generic_store_variable` already does this for a
//! plane that IS a stored variable, and this is deliberately the same
//! sequence of calls: the same projected map and basemap
//! (`build_projected_map_with_projection`), the same frame aspect
//! (`direct_map_frame_aspect_ratio` at `FilledMeteorology`), the same
//! `StaticPlotDesign`, the same `save_png_profile_with_options`.  What it
//! adds is that the values may be anything -- an ensemble reduction, an
//! observation grid -- and that overlays and annotations can ride on top.
//!
//! **What this is not.**  It is not a layout engine.  Multi-panel sheets,
//! lead rows, verification pairs and evolution strips are composition over
//! finished panels, and `gpuwm/pair_compose.py` already does that with
//! Pillow ("pixels are pasted, never recomputed").  Nor is there a footer
//! BAND: `MapRenderRequest` has a title and three subtitle slots and no
//! fourth text region, so a multi-paragraph honesty footer has nowhere
//! honest to go and is not silently squeezed into a subtitle.

use std::path::{Path, PathBuf};

use rustwx_core::{Field2D, GridProjection, GridShape, LatLonGrid, ProductKey};
use rustwx_products::direct::{build_projected_map_with_projection, direct_map_frame_aspect_ratio};
use rustwx_products::plot_design::StaticPlotDesign;
use rustwx_render::{
    ColorScale, ContourLayer, LegendControls, MapRenderRequest, PngCompressionMode,
    PngWriteOptions, ProductVisualMode, ProjectedDomain, RenderDensity,
    save_png_profile_with_options,
};

use crate::annotate::{MapOverlays, PanelAnnotations};

/// Everything one panel needs that is not the map itself.
pub struct PanelRequest<'a> {
    pub lat_deg: &'a [f32],
    pub lon_deg: &'a [f32],
    pub projection: Option<&'a GridProjection>,
    pub ny: usize,
    pub nx: usize,
    /// The filled plane, row-major, `ny * nx`.
    pub values: Vec<f32>,
    /// Filename/product token (`ens_mean_refl`, `obs_z_composite`).
    pub product_slug: String,
    pub title: String,
    pub display_units: String,
    pub scale: ColorScale,
    pub cbar_tick_step: Option<f64>,
    pub legend: LegendControls,
    pub render_density: RenderDensity,
    pub subtitle_left: String,
    pub subtitle_center: Option<String>,
    pub subtitle_right: String,
    pub width: u32,
    pub height: u32,
    /// Contour layers on the SAME grid (paintball members, analysis lines).
    pub contours: Vec<ContourLayer>,
    /// Draw the colourbar.  `false` for a panel whose fill carries no
    /// quantity -- a paintball sheet is contours over nothing, and a
    /// colourbar beside it invites reading a value off an empty scale.
    pub colorbar: bool,
    pub overlays: Option<&'a MapOverlays>,
    pub annotations: Option<&'a PanelAnnotations>,
    pub out_path: PathBuf,
}

/// Render, and return the path written.
pub fn render_panel(request: PanelRequest<'_>) -> Result<PathBuf, String> {
    let points = request.ny * request.nx;
    if request.lat_deg.len() != points || request.lon_deg.len() != points {
        return Err(format!(
            "{}: the {}x{} grid needs {points} coordinate pair(s); got lat {} lon {}",
            request.product_slug,
            request.ny,
            request.nx,
            request.lat_deg.len(),
            request.lon_deg.len()
        ));
    }
    if request.values.len() != points {
        return Err(format!(
            "{}: the {}x{} grid needs {points} value(s); got {}",
            request.product_slug,
            request.ny,
            request.nx,
            request.values.len()
        ));
    }
    for layer in &request.contours {
        if layer.data.len() != points {
            return Err(format!(
                "{}: a contour layer carries {} value(s) on a {points}-point grid",
                request.product_slug,
                layer.data.len()
            ));
        }
    }

    let domain = rusty_weather::batch_render::native_grid_domain_from_coordinates(
        request.lat_deg,
        request.lon_deg,
    )?;
    let target_ratio = direct_map_frame_aspect_ratio(
        ProductVisualMode::FilledMeteorology,
        request.width,
        request.height,
        request.projection,
    );
    let projected = build_projected_map_with_projection(
        request.lat_deg,
        request.lon_deg,
        request.projection,
        domain.bounds,
        target_ratio,
    )
    .map_err(|err| format!("{}: project map: {err}", request.product_slug))?;

    let grid = LatLonGrid {
        shape: GridShape {
            nx: request.nx,
            ny: request.ny,
        },
        lat_deg: request.lat_deg.to_vec(),
        lon_deg: request.lon_deg.to_vec(),
    };
    let field = Field2D::new(
        ProductKey::named(request.product_slug.clone()),
        request.display_units.clone(),
        grid,
        request.values,
    )
    .map_err(|err| format!("{}: build field: {err}", request.product_slug))?;

    let mut map_request = MapRenderRequest::from_core_field(field, request.scale);
    StaticPlotDesign::new(domain.bounds, ProductVisualMode::FilledMeteorology)
        .apply_to_request(&mut map_request);
    map_request.width = request.width;
    map_request.height = request.height;
    map_request.title = Some(request.title);
    map_request.cbar_tick_step = request.cbar_tick_step;
    map_request.render_density = request.render_density;
    map_request.legend = request.legend;
    map_request.subtitle_left = Some(request.subtitle_left);
    map_request.subtitle_center = request.subtitle_center;
    map_request.subtitle_right = Some(request.subtitle_right);
    map_request.projected_domain = Some(ProjectedDomain {
        x: projected.projected_x,
        y: projected.projected_y,
        extent: projected.extent,
    });
    map_request.projected_lines = projected.lines;
    map_request.projected_polygons = projected.polygons;
    map_request.inverse_raster_projection = projected.inverse_raster_projection;
    map_request.contours = request.contours;
    map_request.colorbar = request.colorbar;

    if let Some(overlays) = request.overlays {
        overlays.apply(
            &mut map_request,
            request.lat_deg,
            request.lon_deg,
            request.projection,
            domain.bounds,
            target_ratio,
        )?;
    }
    if let Some(annotations) = request.annotations {
        annotations.apply(&mut map_request);
    }

    if let Some(parent) = request.out_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|err| format!("create {}: {err}", parent.display()))?;
    }
    save_png_profile_with_options(
        &map_request,
        &request.out_path,
        &PngWriteOptions {
            compression: PngCompressionMode::default(),
        },
    )
    .map_err(|err| format!("{}: write PNG: {err}", request.product_slug))?;
    Ok(request.out_path)
}

/// `<out_dir>/<domain>/<product>/<valid-day>/<stem>.png`.
///
/// The render folder layout is a standing ruling (CLAUDE.md: "case folder
/// -> domain -> product subfolders, organised at render time"), and 2.5.0's
/// `<out>/<domain>/<product>/<valid-day>/` extends it.  Filing at write
/// time rather than sorting afterwards is the whole point of the ruling, so
/// these binaries do it themselves rather than leaning on the Python
/// re-filer that only knows the `rw_wrfbatch` filename grammar.
pub fn layout_path(
    out_dir: &Path,
    domain_token: &str,
    product: &str,
    valid_day: &str,
    stem: &str,
) -> PathBuf {
    out_dir
        .join(safe_component(domain_token, "native_grid"))
        .join(safe_component(product, "product"))
        .join(safe_component(valid_day, "undated"))
        .join(format!("{}.png", safe_component(stem, "panel")))
}

/// A filename component with every separator collapsed.
///
/// `.` survives because a 3:1 nest of a 12 km parent is 1.333 km and
/// `d03-1_333km` reads as a typo (the same rule `batch_render::safe_slug`
/// applies).  But a RUN of dots does not survive: these components come
/// from a caller-supplied product name and a file-derived valid day, and
/// `..` in a path element is a directory traversal.  Runs collapse to one
/// dot and leading/trailing dots are trimmed, so `../../etc` is `etc` and
/// `..` alone is the fallback.
pub fn safe_component(value: &str, fallback: &str) -> String {
    let mut out = String::with_capacity(value.len());
    let mut underscore = false;
    let mut dot = false;
    for character in value.trim().chars() {
        if character == '.' {
            if !dot {
                dot = true;
                underscore = false;
                out.push('.');
            }
            continue;
        }
        dot = false;
        if character.is_ascii_alphanumeric() || matches!(character, '-' | '_') {
            underscore = false;
            out.push(character.to_ascii_lowercase());
        } else if !underscore {
            underscore = true;
            out.push('_');
        }
    }
    let trimmed = out.trim_matches(|c| c == '_' || c == '.').to_string();
    if trimmed.is_empty() {
        fallback.to_string()
    } else {
        trimmed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_layout_is_domain_then_product_then_valid_day() {
        let path = layout_path(
            Path::new("out"),
            "d02-1km",
            "ens_mean_refl",
            "1974-04-03",
            "arwen_ens_mean_refl_19740403_1800z",
        );
        assert_eq!(
            path,
            Path::new("out")
                .join("d02-1km")
                .join("ens_mean_refl")
                .join("1974-04-03")
                .join("arwen_ens_mean_refl_19740403_1800z.png")
        );
    }

    #[test]
    fn a_component_never_becomes_a_path_element() {
        assert_eq!(safe_component("../../etc", "x"), "etc");
        assert_eq!(safe_component("..", "fallback"), "fallback");
        assert_eq!(safe_component("d03-1.333km", "x"), "d03-1.333km");
        assert_eq!(safe_component("   ", "fallback"), "fallback");
        assert_eq!(safe_component("A B/C", "x"), "a_b_c");
        for probe in ["..", "../x", "a/../b", "\\..\\", "..."] {
            let component = safe_component(probe, "fallback");
            assert!(!component.contains(".."), "{probe} -> {component}");
            assert!(!component.contains(['/', '\\']), "{probe} -> {component}");
        }
    }
}
