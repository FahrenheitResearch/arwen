use rustwx_core::GridProjection;
use rustwx_render::{
    BasemapDetail, DomainFrame, DomainFrameSource, GeographicClipBounds, InverseRasterProjection,
    ProductVisualMode, ProjectedMap, ProjectedMapBuildOptions,
};

use crate::shared_context::static_chrome_scale;

use super::domain::{
    is_global_scale_domain, longitude_bounds_span_deg, normalize_longitude_for_bounds,
};

pub fn build_projected_map(
    lat_deg: &[f32],
    lon_deg: &[f32],
    bounds: (f64, f64, f64, f64),
    target_ratio: f64,
) -> Result<ProjectedMap, Box<dyn std::error::Error>> {
    build_projected_map_with_projection(lat_deg, lon_deg, None, bounds, target_ratio)
}

pub fn build_projected_map_with_projection(
    lat_deg: &[f32],
    lon_deg: &[f32],
    projection: Option<&GridProjection>,
    bounds: (f64, f64, f64, f64),
    target_ratio: f64,
) -> Result<ProjectedMap, Box<dyn std::error::Error>> {
    if full_domain_projected_frame_enabled(projection, bounds) {
        return build_full_domain_projected_map_with_projection(
            lat_deg,
            lon_deg,
            projection,
            bounds,
            target_ratio,
        );
    }

    let variant = projection_presentation_variant();
    let presentation_projection = presentation_projection_for_bounds(projection, bounds, variant);
    let frame_bounds = presentation_frame_bounds_for_projection(
        bounds,
        presentation_projection.as_ref(),
        target_ratio,
    );
    let mut options = ProjectedMapBuildOptions::from_bounds(frame_bounds, target_ratio);
    if straight_western_projection_enabled(bounds) {
        options = options
            .with_geographic_grid_intersection_frame(frame_bounds)
            .with_natural_frame_aspect();
    }
    if let Some(presentation_projection) = presentation_projection {
        let reference_latitude =
            reference_latitude_for_projection_variant(variant, projection, frame_bounds);
        options = options.with_projection(presentation_projection);
        if let Some(reference_latitude) = reference_latitude {
            options.domain.reference_latitude_deg = Some(reference_latitude);
        }
    }
    options = options.with_basemap_detail(basemap_detail_for_bounds(frame_bounds));
    options.domain.pad_fraction = presentation_pad_fraction_for_bounds(frame_bounds);
    let mut projected =
        rustwx_render::build_projected_map_with_options(lat_deg, lon_deg, &options)?;
    projected.inverse_raster_projection =
        inverse_raster_projection_for_latlon_mesh(projection, frame_bounds, lat_deg, lon_deg);
    Ok(projected)
}

pub fn build_requested_projected_map_with_projection(
    lat_deg: &[f32],
    lon_deg: &[f32],
    projection: Option<&GridProjection>,
    bounds: (f64, f64, f64, f64),
    target_ratio: f64,
) -> Result<ProjectedMap, Box<dyn std::error::Error>> {
    let variant = projection_presentation_variant();
    let presentation_projection = presentation_projection_for_bounds(projection, bounds, variant);
    let frame_bounds = presentation_frame_bounds_for_projection(
        bounds,
        presentation_projection.as_ref(),
        target_ratio,
    );
    let mut options = ProjectedMapBuildOptions::from_bounds(frame_bounds, target_ratio);
    if straight_western_projection_enabled(bounds) {
        options = options
            .with_geographic_grid_intersection_frame(frame_bounds)
            .with_natural_frame_aspect();
    }
    if let Some(presentation_projection) = presentation_projection {
        let reference_latitude =
            reference_latitude_for_projection_variant(variant, projection, frame_bounds);
        options = options.with_projection(presentation_projection);
        if let Some(reference_latitude) = reference_latitude {
            options.domain.reference_latitude_deg = Some(reference_latitude);
        }
    }
    options = options.with_basemap_detail(basemap_detail_for_bounds(frame_bounds));
    options.domain.pad_fraction = presentation_pad_fraction_for_bounds(frame_bounds);
    let mut projected =
        rustwx_render::build_projected_map_with_options(lat_deg, lon_deg, &options)?;
    projected.inverse_raster_projection =
        inverse_raster_projection_for_latlon_mesh(projection, frame_bounds, lat_deg, lon_deg);
    Ok(projected)
}

pub fn build_natural_projected_map_with_projection(
    lat_deg: &[f32],
    lon_deg: &[f32],
    projection: Option<&GridProjection>,
    bounds: (f64, f64, f64, f64),
    target_ratio: f64,
) -> Result<ProjectedMap, Box<dyn std::error::Error>> {
    build_natural_projected_map_with_projection_inner(
        lat_deg,
        lon_deg,
        projection,
        bounds,
        target_ratio,
        None,
    )
}

pub fn build_natural_projected_map_with_projection_and_basemap_padding(
    lat_deg: &[f32],
    lon_deg: &[f32],
    projection: Option<&GridProjection>,
    bounds: (f64, f64, f64, f64),
    target_ratio: f64,
    line_pad_fraction: f64,
    polygon_pad_fraction: f64,
) -> Result<ProjectedMap, Box<dyn std::error::Error>> {
    build_natural_projected_map_with_projection_inner(
        lat_deg,
        lon_deg,
        projection,
        bounds,
        target_ratio,
        Some((line_pad_fraction, polygon_pad_fraction)),
    )
}

fn build_natural_projected_map_with_projection_inner(
    lat_deg: &[f32],
    lon_deg: &[f32],
    projection: Option<&GridProjection>,
    bounds: (f64, f64, f64, f64),
    target_ratio: f64,
    basemap_padding: Option<(f64, f64)>,
) -> Result<ProjectedMap, Box<dyn std::error::Error>> {
    let variant = projection_presentation_variant();
    let presentation_projection = presentation_projection_for_bounds(projection, bounds, variant);
    let frame_bounds = bounds;
    let mut options = ProjectedMapBuildOptions::from_bounds(frame_bounds, target_ratio)
        .with_geographic_grid_intersection_frame(frame_bounds)
        .with_natural_frame_aspect();
    if let Some(presentation_projection) = presentation_projection {
        let reference_latitude =
            reference_latitude_for_projection_variant(variant, projection, frame_bounds);
        options = options.with_projection(presentation_projection);
        if let Some(reference_latitude) = reference_latitude {
            options.domain.reference_latitude_deg = Some(reference_latitude);
        }
    }
    options = options.with_basemap_detail(basemap_detail_for_bounds(frame_bounds));
    if let Some((line_pad_fraction, polygon_pad_fraction)) = basemap_padding {
        options = options.with_basemap_padding(line_pad_fraction, polygon_pad_fraction);
    }
    options.domain.pad_fraction = presentation_pad_fraction_for_bounds(frame_bounds);
    let mut projected =
        rustwx_render::build_projected_map_with_options(lat_deg, lon_deg, &options)?;
    projected.inverse_raster_projection =
        inverse_raster_projection_for_latlon_mesh(projection, frame_bounds, lat_deg, lon_deg);
    Ok(projected)
}

fn build_full_domain_projected_map_with_projection(
    lat_deg: &[f32],
    lon_deg: &[f32],
    projection: Option<&GridProjection>,
    bounds: (f64, f64, f64, f64),
    target_ratio: f64,
) -> Result<ProjectedMap, Box<dyn std::error::Error>> {
    let mut options = ProjectedMapBuildOptions::full_domain(target_ratio);
    if let Some(projection) = projection {
        options = options.with_projection(projection.clone());
    }
    let basemap_bounds = latlon_mesh_bounds(lat_deg, lon_deg).unwrap_or(bounds);
    options = options.with_basemap_detail(basemap_detail_for_bounds(basemap_bounds));
    options.domain.pad_fraction = full_domain_projected_frame_pad_fraction();
    let mut projected =
        rustwx_render::build_projected_map_with_options(lat_deg, lon_deg, &options)?;
    projected.inverse_raster_projection =
        inverse_raster_projection_for_latlon_mesh(projection, basemap_bounds, lat_deg, lon_deg);
    Ok(projected)
}

fn full_domain_projected_frame_enabled(
    projection: Option<&GridProjection>,
    bounds: (f64, f64, f64, f64),
) -> bool {
    let auto = full_domain_projected_frame_default(projection, bounds);
    std::env::var("RUSTWX_PROJECTED_FRAME_SOURCE")
        .ok()
        .map(|value| match value.trim().to_ascii_lowercase().as_str() {
            "full-domain" | "full_domain" | "native" | "native-domain" | "native_domain" => true,
            "requested" | "request" | "bounds" | "domain" | "map-bounds" | "map_bounds" => false,
            "auto" | "" => auto,
            other => matches!(other, "1" | "true" | "yes" | "on"),
        })
        .unwrap_or(auto)
}

pub(super) fn full_domain_projected_frame_default(
    projection: Option<&GridProjection>,
    bounds: (f64, f64, f64, f64),
) -> bool {
    let lat_span = (bounds.3 - bounds.2).abs();
    let lon_span = longitude_bounds_span_deg(bounds);
    !matches!(projection, Some(GridProjection::Geographic) | None)
        && (lat_span >= 25.0 || lon_span >= 45.0)
}

fn full_domain_projected_frame_pad_fraction() -> f64 {
    std::env::var("RUSTWX_PROJECTED_FRAME_PAD_FRACTION")
        .ok()
        .and_then(|value| value.trim().parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value >= 0.0)
        .unwrap_or(0.02)
        .clamp(0.0, 0.25)
}

fn latlon_mesh_bounds(lat_deg: &[f32], lon_deg: &[f32]) -> Option<(f64, f64, f64, f64)> {
    let mut west = f64::INFINITY;
    let mut east = f64::NEG_INFINITY;
    let mut south = f64::INFINITY;
    let mut north = f64::NEG_INFINITY;
    for (&lat, &lon) in lat_deg.iter().zip(lon_deg.iter()) {
        let lat = lat as f64;
        let lon = lon as f64;
        if !lat.is_finite() || !lon.is_finite() {
            continue;
        }
        south = south.min(lat);
        north = north.max(lat);
        west = west.min(lon);
        east = east.max(lon);
    }
    (west.is_finite() && east.is_finite() && south.is_finite() && north.is_finite())
        .then_some((west, east, south, north))
}

pub(crate) fn inverse_raster_projection_for_grid(
    projection: Option<&GridProjection>,
    bounds: (f64, f64, f64, f64),
    grid: &rustwx_core::LatLonGrid,
) -> Option<InverseRasterProjection> {
    inverse_raster_projection_for_latlon_mesh(projection, bounds, &grid.lat_deg, &grid.lon_deg)
}

fn inverse_raster_projection_for_latlon_mesh(
    projection: Option<&GridProjection>,
    bounds: (f64, f64, f64, f64),
    lat_deg: &[f32],
    lon_deg: &[f32],
) -> Option<InverseRasterProjection> {
    let regular_latlon = matches!(projection, Some(GridProjection::Geographic) | None)
        && rectilinear_latlon_mesh_for_inverse(lat_deg, lon_deg);
    if !regular_latlon {
        return None;
    }
    let variant = projection_presentation_variant();
    let projection =
        presentation_projection_for_bounds(Some(&GridProjection::Geographic), bounds, variant)?;
    let reference_longitude_deg = match projection {
        rustwx_render::ProjectionSpec::Geographic => Some(center_longitude_for_bounds(bounds)),
        _ => None,
    };
    match projection {
        rustwx_render::ProjectionSpec::AlbersEqualArea { .. }
        | rustwx_render::ProjectionSpec::Geographic
        | rustwx_render::ProjectionSpec::LambertConformal { .. }
        | rustwx_render::ProjectionSpec::Mercator { .. }
        | rustwx_render::ProjectionSpec::Robinson { .. } => {
            let clip_bounds = inverse_raster_clip_bounds(bounds, &projection);
            Some(InverseRasterProjection {
                projection,
                reference_latitude_deg: reference_latitude_for_projection_variant(
                    variant,
                    Some(&GridProjection::Geographic),
                    bounds,
                ),
                reference_longitude_deg,
                clip_bounds,
            })
        }
        _ => None,
    }
}

pub(super) fn inverse_raster_clip_bounds(
    bounds: (f64, f64, f64, f64),
    projection: &rustwx_render::ProjectionSpec,
) -> Option<GeographicClipBounds> {
    if !env_flag_enabled("RUSTWX_INVERSE_RASTER_GEO_CLIP", true) {
        return None;
    }
    if !matches!(projection, rustwx_render::ProjectionSpec::Geographic) {
        return None;
    }
    Some(GeographicClipBounds::new(
        bounds.0, bounds.1, bounds.2, bounds.3,
    ))
}

fn env_flag_enabled(name: &str, default: bool) -> bool {
    std::env::var(name)
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(default)
}

pub(super) fn rectilinear_latlon_mesh_for_inverse(lat_deg: &[f32], lon_deg: &[f32]) -> bool {
    if lat_deg.len() != lon_deg.len() || lat_deg.len() < 9 {
        return false;
    }
    let len = lat_deg.len();
    let mut nx = 0usize;
    for idx in 1..len {
        if (lat_deg[idx] - lat_deg[0]).abs() > 1.0e-4 {
            nx = idx;
            break;
        }
    }
    if nx < 2 || len % nx != 0 {
        return false;
    }
    let ny = len / nx;
    if ny < 2 {
        return false;
    }
    let sample_rows = [0, ny / 2, ny - 1];
    let sample_cols = [0, nx / 2, nx - 1];
    for &row in &sample_rows {
        let row_offset = row * nx;
        let row_lat = lat_deg[row_offset];
        for &col in &sample_cols {
            if (lat_deg[row_offset + col] - row_lat).abs() > 1.0e-3 {
                return false;
            }
        }
    }
    for &col in &sample_cols {
        let col_lon = lon_deg[col];
        for &row in &sample_rows {
            if longitude_delta_abs_deg(lon_deg[row * nx + col], col_lon) > 1.0e-3 {
                return false;
            }
        }
    }
    true
}

fn longitude_delta_abs_deg(a: f32, b: f32) -> f32 {
    let mut delta = (a - b).abs();
    while delta > 180.0 {
        delta = (delta - 360.0).abs();
    }
    delta
}

pub fn model_data_domain_frame_for_projection(
    _projection: Option<&GridProjection>,
) -> Option<DomainFrame> {
    Some(DomainFrame {
        inset_px: 2,
        outline_width: 2,
        source: DomainFrameSource::ProjectedGrid,
        ..DomainFrame::map_viewport_default()
    })
}

/// The direct lane's map-frame target ratio: visual-mode, domain-frame,
/// and chrome-scale aware.  Public so the generic store-variable lane can
/// build its projected frame with the SAME geometry named products get --
/// a plain `map_frame_aspect_ratio` produced a narrower frame whose
/// subtitle row truncated text that named products fit at the same size.
pub fn direct_map_frame_aspect_ratio(
    visual_mode: ProductVisualMode,
    width: u32,
    height: u32,
    projection: Option<&GridProjection>,
) -> f64 {
    rustwx_render::map_frame_aspect_ratio_for_mode_with_domain_frame_and_chrome_scale(
        visual_mode,
        width,
        height,
        true,
        true,
        model_data_domain_frame_for_projection(projection).is_some(),
        static_chrome_scale(),
    )
}

fn basemap_detail_for_bounds(bounds: (f64, f64, f64, f64)) -> BasemapDetail {
    let lat_span = (bounds.3 - bounds.2).abs();
    let lon_span = longitude_bounds_span_deg(bounds);
    if is_global_scale_domain(bounds) {
        BasemapDetail::Global
    } else if lat_span >= 60.0 || lon_span >= 120.0 {
        BasemapDetail::Broad
    } else {
        BasemapDetail::Regional
    }
}

fn presentation_pad_fraction_for_bounds(bounds: (f64, f64, f64, f64)) -> f64 {
    if let Ok(value) = std::env::var("RUSTWX_PRESENTATION_PAD_FRACTION") {
        if let Ok(parsed) = value.trim().parse::<f64>() {
            return parsed.clamp(0.0, 0.25);
        }
    }
    let lat_span = (bounds.3 - bounds.2).abs();
    let lon_span = longitude_bounds_span_deg(bounds);
    if is_global_scale_domain(bounds) {
        0.06
    } else if lat_span >= 45.0 || lon_span >= 65.0 {
        0.045
    } else {
        0.025
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ProjectionPresentationVariant {
    Adaptive,
    AlbersEqualArea,
    RectangularGeographic,
    Mercator,
    PivotalLambert,
    Robinson,
}

pub(super) const PIVOTAL_CONUS_STANDARD_PARALLEL_1_DEG: f64 = 33.0;
pub(super) const PIVOTAL_CONUS_STANDARD_PARALLEL_2_DEG: f64 = 45.0;
pub(super) const PIVOTAL_CONUS_CENTRAL_MERIDIAN_DEG: f64 = -96.0;
pub(super) const PIVOTAL_CONUS_REFERENCE_LATITUDE_DEG: f64 = 39.0;
const NORTH_AMERICA_LAMBERT_REFERENCE_LATITUDE_DEG: f64 = 45.0;
pub(super) const PIVOTAL_GEOGRAPHIC_CROP_PAD_DEG: f64 = 18.0;

pub(super) fn projection_presentation_variant() -> ProjectionPresentationVariant {
    std::env::var("RUSTWX_PROJECTION_VARIANT")
        .ok()
        .map(
            |value| match normalize_projection_variant_name(&value).as_str() {
                "albers" | "albersequalarea" | "aea" => {
                    ProjectionPresentationVariant::AlbersEqualArea
                }
                "rectangular" | "geographic" | "platecarree" | "crop" => {
                    ProjectionPresentationVariant::RectangularGeographic
                }
                "mercator" | "webmap" | "webmercator" | "straight" | "northup"
                | "northupmercator" | "weststraight" | "straightwest" | "westernstraight" => {
                    ProjectionPresentationVariant::Mercator
                }
                "pivotallambert" | "pivotal" => ProjectionPresentationVariant::PivotalLambert,
                "robinson" | "atlas" => ProjectionPresentationVariant::Robinson,
                _ => ProjectionPresentationVariant::Adaptive,
            },
        )
        .unwrap_or(ProjectionPresentationVariant::Adaptive)
}

fn normalize_projection_variant_name(value: &str) -> String {
    value.trim().to_ascii_lowercase().replace(['-', '_'], "")
}

pub(super) fn presentation_projection_for_bounds(
    native_projection: Option<&GridProjection>,
    bounds: (f64, f64, f64, f64),
    variant: ProjectionPresentationVariant,
) -> Option<rustwx_render::ProjectionSpec> {
    if is_global_scale_domain(bounds) {
        return Some(rustwx_render::ProjectionSpec::Robinson {
            central_meridian_deg: center_longitude_for_bounds(bounds),
        });
    }

    if !matches!(variant, ProjectionPresentationVariant::Adaptive) {
        return Some(regional_latlon_presentation_projection(bounds, variant));
    }

    if straight_western_projection_enabled(bounds) {
        return Some(straight_western_presentation_projection(bounds));
    }

    match native_projection {
        Some(GridProjection::Geographic) | None => {
            Some(regional_latlon_presentation_projection(bounds, variant))
        }
        Some(projection) => Some(projection.clone().into()),
    }
}

fn regional_latlon_presentation_projection(
    bounds: (f64, f64, f64, f64),
    variant: ProjectionPresentationVariant,
) -> rustwx_render::ProjectionSpec {
    match variant {
        ProjectionPresentationVariant::AlbersEqualArea => conus_albers_presentation_projection(),
        ProjectionPresentationVariant::RectangularGeographic => {
            rustwx_render::ProjectionSpec::Geographic
        }
        ProjectionPresentationVariant::Mercator => {
            regional_mercator_presentation_projection(bounds)
        }
        ProjectionPresentationVariant::PivotalLambert if is_conus_lambert_candidate(bounds) => {
            pivotal_lambert_conus_projection()
        }
        ProjectionPresentationVariant::PivotalLambert
            if is_north_america_projection_candidate(bounds) =>
        {
            north_america_lambert_presentation_projection()
        }
        ProjectionPresentationVariant::Robinson => robinson_presentation_projection(bounds),
        _ => regional_presentation_projection(bounds),
    }
}

fn presentation_frame_bounds_for_projection(
    bounds: (f64, f64, f64, f64),
    projection: Option<&rustwx_render::ProjectionSpec>,
    target_ratio: f64,
) -> (f64, f64, f64, f64) {
    if !matches!(projection, Some(rustwx_render::ProjectionSpec::Geographic))
        || is_global_scale_domain(bounds)
    {
        return bounds;
    }
    expand_geographic_bounds_to_aspect(bounds, target_ratio)
}

pub(super) fn presentation_frame_bounds_for_grid(
    native_projection: Option<&GridProjection>,
    bounds: (f64, f64, f64, f64),
    variant: ProjectionPresentationVariant,
    target_ratio: f64,
) -> (f64, f64, f64, f64) {
    let presentation_projection =
        presentation_projection_for_bounds(native_projection, bounds, variant);
    presentation_frame_bounds_for_projection(bounds, presentation_projection.as_ref(), target_ratio)
}

fn expand_geographic_bounds_to_aspect(
    bounds: (f64, f64, f64, f64),
    target_ratio: f64,
) -> (f64, f64, f64, f64) {
    let safe_ratio = target_ratio.max(1.0e-6);
    let mut south = bounds.2.min(bounds.3).clamp(-89.5, 89.5);
    let mut north = bounds.2.max(bounds.3).clamp(-89.5, 89.5);
    if north <= south {
        south = (south - 0.5).clamp(-89.5, 89.0);
        north = (north + 0.5).clamp(-89.0, 89.5);
    }
    let lat_span = (north - south).max(1.0e-6);
    let lon_span = longitude_bounds_span_deg(bounds).max(1.0e-6);
    let current_ratio = lon_span / lat_span;
    if current_ratio < safe_ratio {
        let wanted_lon_span = (lat_span * safe_ratio).min(360.0);
        let center = center_longitude_for_bounds(bounds);
        let west = normalize_longitude_for_bounds(center - wanted_lon_span / 2.0);
        let east_unwrapped = center + wanted_lon_span / 2.0;
        let east = if east_unwrapped > 180.0 {
            east_unwrapped
        } else {
            normalize_longitude_for_bounds(east_unwrapped)
        };
        (west, east, south, north)
    } else {
        let wanted_lat_span = lon_span / safe_ratio;
        let center = ((south + north) / 2.0).clamp(-89.0, 89.0);
        south = (center - wanted_lat_span / 2.0).clamp(-89.5, 89.5);
        north = (center + wanted_lat_span / 2.0).clamp(-89.5, 89.5);
        if north - south < wanted_lat_span {
            if south <= -89.5 {
                north = (south + wanted_lat_span).clamp(-89.5, 89.5);
            } else if north >= 89.5 {
                south = (north - wanted_lat_span).clamp(-89.5, 89.5);
            }
        }
        (bounds.0, bounds.1, south, north)
    }
}

fn conus_albers_presentation_projection() -> rustwx_render::ProjectionSpec {
    rustwx_render::ProjectionSpec::AlbersEqualArea {
        standard_parallel_1_deg: 29.5,
        standard_parallel_2_deg: 45.5,
        central_meridian_deg: -96.0,
        latitude_of_origin_deg: 23.0,
    }
}

fn pivotal_lambert_conus_projection() -> rustwx_render::ProjectionSpec {
    rustwx_render::ProjectionSpec::LambertConformal {
        standard_parallel_1_deg: PIVOTAL_CONUS_STANDARD_PARALLEL_1_DEG,
        standard_parallel_2_deg: PIVOTAL_CONUS_STANDARD_PARALLEL_2_DEG,
        central_meridian_deg: PIVOTAL_CONUS_CENTRAL_MERIDIAN_DEG,
    }
}

fn north_america_lambert_presentation_projection() -> rustwx_render::ProjectionSpec {
    rustwx_render::ProjectionSpec::LambertConformal {
        standard_parallel_1_deg: 25.0,
        standard_parallel_2_deg: 60.0,
        central_meridian_deg: -100.0,
    }
}

fn regional_mercator_presentation_projection(
    bounds: (f64, f64, f64, f64),
) -> rustwx_render::ProjectionSpec {
    if bounds.3 <= -55.0 || bounds.2 >= 55.0 {
        return regional_presentation_projection(bounds);
    }

    rustwx_render::ProjectionSpec::Mercator {
        latitude_of_true_scale_deg: ((bounds.2 + bounds.3) / 2.0).clamp(-85.0, 85.0),
        central_meridian_deg: center_longitude_for_bounds(bounds),
    }
}

pub(super) fn reference_latitude_for_projection_variant(
    variant: ProjectionPresentationVariant,
    native_projection: Option<&GridProjection>,
    bounds: (f64, f64, f64, f64),
) -> Option<f64> {
    let _ = variant;
    match native_projection {
        Some(GridProjection::Geographic) | None if !is_global_scale_domain(bounds) => {
            if is_conus_lambert_candidate(bounds) {
                Some(PIVOTAL_CONUS_REFERENCE_LATITUDE_DEG)
            } else if is_north_america_projection_candidate(bounds) {
                Some(NORTH_AMERICA_LAMBERT_REFERENCE_LATITUDE_DEG)
            } else {
                None
            }
        }
        _ => None,
    }
}

fn is_conus_lambert_candidate(bounds: (f64, f64, f64, f64)) -> bool {
    let west = normalize_longitude_for_bounds(bounds.0);
    let east = normalize_longitude_for_bounds(bounds.1);
    if west > east {
        return false;
    }

    let lat_span = (bounds.3 - bounds.2).abs();
    let lon_span = longitude_bounds_span_deg(bounds);
    bounds.2 >= 20.0
        && bounds.3 <= 56.0
        && west >= -132.0
        && east <= -60.0
        && lat_span >= 5.0
        && lat_span <= 38.0
        && lon_span >= 8.0
        && lon_span <= 75.0
}

fn is_north_america_projection_candidate(bounds: (f64, f64, f64, f64)) -> bool {
    let west = normalize_longitude_for_bounds(bounds.0);
    let east = normalize_longitude_for_bounds(bounds.1);
    if west > east {
        return false;
    }

    let lat_span = (bounds.3 - bounds.2).abs();
    let lon_span = longitude_bounds_span_deg(bounds);
    bounds.2 >= -5.0
        && bounds.3 <= 88.0
        && west >= -180.0
        && east <= -35.0
        && lat_span >= 45.0
        && lon_span >= 80.0
        && lon_span <= 155.0
}

fn regional_presentation_projection(bounds: (f64, f64, f64, f64)) -> rustwx_render::ProjectionSpec {
    let center_lat = ((bounds.2 + bounds.3) / 2.0).clamp(-85.0, 85.0);
    let center_lon = center_longitude_for_bounds(bounds);
    let lat_span = (bounds.3 - bounds.2).abs();

    if bounds.3 <= -55.0 {
        return rustwx_render::ProjectionSpec::PolarStereographic {
            true_latitude_deg: -71.0,
            central_meridian_deg: center_lon,
            south_pole_on_projection_plane: true,
        };
    }
    if bounds.2 >= 55.0 {
        return rustwx_render::ProjectionSpec::PolarStereographic {
            true_latitude_deg: 71.0,
            central_meridian_deg: center_lon,
            south_pole_on_projection_plane: false,
        };
    }
    if is_north_america_projection_candidate(bounds) {
        return north_america_lambert_presentation_projection();
    }
    if is_broad_continent_scale_domain(bounds) {
        return rustwx_render::ProjectionSpec::Geographic;
    }
    if bounds.2 < -25.0 && bounds.3 > 25.0 {
        return rustwx_render::ProjectionSpec::Mercator {
            latitude_of_true_scale_deg: center_lat,
            central_meridian_deg: center_lon,
        };
    }

    let inset = (lat_span / 6.0).clamp(2.0, 12.0);
    let sp1 = stabilize_presentation_parallel(bounds.2 + inset);
    let sp2 = stabilize_presentation_parallel(bounds.3 - inset);
    rustwx_render::ProjectionSpec::LambertConformal {
        standard_parallel_1_deg: sp1,
        standard_parallel_2_deg: if (sp2 - sp1).abs() < 0.25 { sp1 } else { sp2 },
        central_meridian_deg: center_lon,
    }
}

pub(super) fn is_broad_continent_scale_domain(bounds: (f64, f64, f64, f64)) -> bool {
    let lat_span = (bounds.3 - bounds.2).abs();
    let lon_span = longitude_bounds_span_deg(bounds);
    !is_conus_lambert_candidate(bounds) && (lat_span >= 50.0 || lon_span >= 90.0)
}

fn robinson_presentation_projection(bounds: (f64, f64, f64, f64)) -> rustwx_render::ProjectionSpec {
    rustwx_render::ProjectionSpec::Robinson {
        central_meridian_deg: center_longitude_for_bounds(bounds),
    }
}

pub(super) fn center_longitude_for_bounds(bounds: (f64, f64, f64, f64)) -> f64 {
    if longitude_bounds_span_deg(bounds) >= 359.0 {
        return 0.0;
    }
    let west = normalize_longitude_for_bounds(bounds.0);
    let mut east = normalize_longitude_for_bounds(bounds.1);
    if east < west {
        east += 360.0;
    }
    normalize_longitude_for_bounds((west + east) / 2.0)
}

fn stabilize_presentation_parallel(lat_deg: f64) -> f64 {
    let lat = lat_deg.clamp(-80.0, 80.0);
    if lat.abs() < 1.0 {
        10.0_f64.copysign(if lat < 0.0 { -1.0 } else { 1.0 })
    } else {
        lat
    }
}

fn straight_western_projection_enabled(bounds: (f64, f64, f64, f64)) -> bool {
    let default = is_straight_western_projection_candidate(bounds);
    env_flag_enabled("RUSTWX_STRAIGHT_WEST_PROJECTION", default)
}

fn is_straight_western_projection_candidate(bounds: (f64, f64, f64, f64)) -> bool {
    let west = normalize_longitude_for_bounds(bounds.0);
    let east = normalize_longitude_for_bounds(bounds.1);
    if west > east {
        return false;
    }
    let lat_span = (bounds.3 - bounds.2).abs();
    let lon_span = longitude_bounds_span_deg(bounds);
    bounds.2 >= 25.0
        && bounds.3 <= 55.0
        && west >= -130.0
        && west <= -115.0
        && east >= -123.0
        && east <= -104.0
        && lat_span >= 4.0
        && lon_span <= 28.0
}

fn straight_western_presentation_projection(
    bounds: (f64, f64, f64, f64),
) -> rustwx_render::ProjectionSpec {
    regional_mercator_presentation_projection(bounds)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conus_sized_domains_keep_regional_basemap_detail() {
        assert_eq!(
            basemap_detail_for_bounds((-127.0, -66.0, 23.0, 51.5)),
            BasemapDetail::Regional
        );
        assert_eq!(
            basemap_detail_for_bounds((-134.0, -60.0, 21.0, 54.0)),
            BasemapDetail::Regional
        );
    }

    #[test]
    fn continent_sized_domains_use_broad_basemap_detail() {
        assert_eq!(
            basemap_detail_for_bounds((-170.0, -50.0, 5.0, 84.0)),
            BasemapDetail::Broad
        );
    }
}
