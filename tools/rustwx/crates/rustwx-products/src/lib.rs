//! Product orchestration APIs.
//!
//! This crate currently keeps a broad public module surface for compatibility.
//! Before changing any module visibility, update the product-surface map in
//! the guardrail tests in `src/tests.rs` so consumers and CLI tools get
//! audited intentionally.

pub mod cache;
pub mod catalog;
pub mod derived;
pub mod direct;
pub mod ecape;
pub mod geographic_overlays;
pub mod gridded;
pub mod heavy;
pub mod hrrr;
pub mod named_geometry;
pub mod non_ecape;
pub mod places;
pub mod planner;
pub mod plot_design;
pub mod point_timeseries;
pub mod publication;
pub(crate) mod qpf;
pub mod runtime;
pub mod sampling;
pub mod severe;
pub mod shared_context;
pub mod source;
pub mod spec;
pub mod thermo_native;
pub mod viewer;
pub mod windowed;
pub mod windowed_decoder;

pub use named_geometry::{
    NamedGeoBounds, NamedGeoPoint, NamedGeometry, NamedGeometryAsset, NamedGeometryCatalog,
    NamedGeometryKind, NamedGeometrySelector,
};
pub use shared_context::{
    DomainSpec, PreparedProjectedContext, ProjectedMap, ProjectedMapProvider, WeatherPanelField,
    WeatherPanelHeader, WeatherPanelLayout, layout_key, render_two_by_four_weather_panel,
};

pub(crate) fn apply_place_label_overlay_with_density_styling(
    render_request: &mut rustwx_render::MapRenderRequest,
    overlay: &crate::places::PlaceLabelOverlay,
    domain: &crate::shared_context::DomainSpec,
    grid_lat_deg: &[f32],
    grid_lon_deg: &[f32],
    projection: Option<&rustwx_core::GridProjection>,
) -> Result<(), Box<dyn std::error::Error>> {
    let selected = overlay.selected_places_for_domain(domain);
    let start = render_request.projected_place_labels.len();
    crate::places::apply_place_label_overlay(
        render_request,
        overlay,
        domain,
        grid_lat_deg,
        grid_lon_deg,
        projection,
    )?;
    style_added_place_labels_for_density(render_request, overlay, &selected, start);
    Ok(())
}

fn style_added_place_labels_for_density(
    render_request: &mut rustwx_render::MapRenderRequest,
    overlay: &crate::places::PlaceLabelOverlay,
    selected: &[crate::places::SelectedPlace],
    start: usize,
) {
    let aux_total = selected
        .iter()
        .filter(|place| !is_major_catalog_place(place.slug.as_str()))
        .count();
    let auxiliary_budget = if aux_total <= 2 {
        aux_total
    } else {
        aux_total.div_ceil(2).min(4)
    };
    let mut aux_seen = 0usize;

    for (place, label) in selected
        .iter()
        .zip(render_request.projected_place_labels.iter_mut().skip(start))
    {
        let is_auxiliary_catalog_place = !is_major_catalog_place(place.slug.as_str());
        label.priority = match overlay.density {
            crate::places::PlaceLabelDensityTier::None
            | crate::places::PlaceLabelDensityTier::Major => {
                rustwx_render::ProjectedPlaceLabelPriority::Primary
            }
            crate::places::PlaceLabelDensityTier::MajorAndAux => {
                if is_auxiliary_catalog_place {
                    rustwx_render::ProjectedPlaceLabelPriority::Auxiliary
                } else {
                    rustwx_render::ProjectedPlaceLabelPriority::Primary
                }
            }
            crate::places::PlaceLabelDensityTier::Dense => {
                if !is_auxiliary_catalog_place {
                    rustwx_render::ProjectedPlaceLabelPriority::Primary
                } else if aux_seen < auxiliary_budget {
                    rustwx_render::ProjectedPlaceLabelPriority::Auxiliary
                } else {
                    rustwx_render::ProjectedPlaceLabelPriority::Micro
                }
            }
        };

        if is_auxiliary_catalog_place {
            aux_seen = aux_seen.saturating_add(1);
        }
    }
}

fn is_major_catalog_place(slug: &str) -> bool {
    crate::places::major_us_city_places()
        .iter()
        .any(|preset| preset.slug == slug)
}

#[cfg(test)]
mod tests;
