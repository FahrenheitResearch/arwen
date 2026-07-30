use std::collections::HashMap;

use rustwx_core::{FieldSelector, GridProjection, SelectedField2D};
use rustwx_render::ProductVisualMode;

use crate::gridded::{GridCrop, crop_latlon_grid, crop_values_f32};

use super::planning::PlannedDirectRecipe;
use super::types::DirectBatchRequest;
use super::{
    projection::{
        PIVOTAL_GEOGRAPHIC_CROP_PAD_DEG, ProjectionPresentationVariant,
        center_longitude_for_bounds, direct_map_frame_aspect_ratio,
        inverse_raster_projection_for_grid, presentation_frame_bounds_for_grid,
        projection_presentation_variant, rectilinear_latlon_mesh_for_inverse,
    },
    rendering::{should_render_overlay_only, visual_mode_for_direct_recipe},
};

pub(super) fn crop_bounds_for_direct_request(
    request: &DirectBatchRequest,
    planned: &[PlannedDirectRecipe],
    extracted: &HashMap<FieldSelector, SelectedField2D>,
) -> (f64, f64, f64, f64) {
    let Some((recipe, field)) = planned.iter().find_map(|item| {
        let selector = item.recipe.filled.selector?;
        extracted.get(&selector).map(|field| (item.recipe, field))
    }) else {
        return request.domain.bounds;
    };
    let overlay_only = should_render_overlay_only(field.selector, recipe.contours.is_some());
    let visual_mode = visual_mode_for_direct_recipe(recipe, field.selector, overlay_only);
    render_bounds_for_direct_field(
        request.domain.bounds,
        field,
        visual_mode,
        request.output_width,
        request.output_height,
    )
}

pub(super) fn render_bounds_for_direct_field(
    bounds: (f64, f64, f64, f64),
    field: &SelectedField2D,
    visual_mode: ProductVisualMode,
    width: u32,
    height: u32,
) -> (f64, f64, f64, f64) {
    let target_ratio =
        direct_map_frame_aspect_ratio(visual_mode, width, height, field.projection.as_ref());
    presentation_frame_bounds_for_grid(
        field.projection.as_ref(),
        bounds,
        projection_presentation_variant(),
        target_ratio,
    )
}

fn direct_domain_crop_pad_cells_override() -> Option<usize> {
    std::env::var("RUSTWX_DOMAIN_CROP_PAD_CELLS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
}

fn direct_domain_crop_pad_cells_for_field(field: &SelectedField2D) -> usize {
    let base = direct_domain_crop_pad_cells_override().unwrap_or(6);
    if !matches!(field.projection.as_ref(), Some(GridProjection::Geographic)) {
        return base.max(128);
    }

    let variant = projection_presentation_variant();
    let pad_deg = std::env::var("RUSTWX_GEOGRAPHIC_DOMAIN_CROP_PAD_DEG")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(match variant {
            ProjectionPresentationVariant::PivotalLambert => PIVOTAL_GEOGRAPHIC_CROP_PAD_DEG,
            _ => 12.0,
        });
    let Some(spacing_deg) = estimate_geographic_grid_spacing_deg(&field.grid) else {
        return base.max(24);
    };
    let cells = (pad_deg / spacing_deg.max(1.0e-6)).ceil() as usize;
    let max_cells = match variant {
        ProjectionPresentationVariant::PivotalLambert => 128,
        _ => 96,
    };
    base.max(cells.clamp(12, max_cells))
}

pub(super) fn crop_direct_fields_for_domain(
    extracted: &HashMap<FieldSelector, SelectedField2D>,
    bounds: (f64, f64, f64, f64),
) -> Result<HashMap<FieldSelector, SelectedField2D>, Box<dyn std::error::Error>> {
    let mut cropped = HashMap::with_capacity(extracted.len());
    for (&selector, field) in extracted {
        let mut pad_cells = direct_domain_crop_pad_cells_for_field(field);
        let uses_inverse_raster =
            inverse_raster_projection_for_grid(field.projection.as_ref(), bounds, &field.grid)
                .is_some();
        if uses_inverse_raster {
            pad_cells = pad_cells.max(inverse_raster_crop_pad_cells());
        }
        let preserve_full_longitude_axis =
            uses_inverse_raster && direct_field_has_periodic_geographic_axis(field);
        cropped.insert(
            selector,
            crop_selected_field_for_domain(field, bounds, pad_cells, preserve_full_longitude_axis)?,
        );
    }
    Ok(cropped)
}

fn inverse_raster_crop_pad_cells() -> usize {
    std::env::var("RUSTWX_INVERSE_RASTER_CROP_PAD_CELLS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(1000)
}

fn estimate_geographic_grid_spacing_deg(grid: &rustwx_core::LatLonGrid) -> Option<f64> {
    let nx = grid.shape.nx;
    let ny = grid.shape.ny;
    if nx < 2 && ny < 2 {
        return None;
    }

    let mut best = f64::INFINITY;
    let row_candidates = [0usize, ny / 2, ny.saturating_sub(1)];
    for y in row_candidates {
        if y >= ny || nx < 2 {
            continue;
        }
        let offset = y * nx;
        for x in 0..nx - 1 {
            let a = grid.lon_deg[offset + x] as f64;
            let b = grid.lon_deg[offset + x + 1] as f64;
            let delta = longitude_delta_deg(a, b);
            if delta.is_finite() && delta > 0.0 && delta < best {
                best = delta;
            }
        }
    }

    let col_candidates = [0usize, nx / 2, nx.saturating_sub(1)];
    for x in col_candidates {
        if x >= nx || ny < 2 {
            continue;
        }
        for y in 0..ny - 1 {
            let a = grid.lat_deg[y * nx + x] as f64;
            let b = grid.lat_deg[(y + 1) * nx + x] as f64;
            let delta = (b - a).abs();
            if delta.is_finite() && delta > 0.0 && delta < best {
                best = delta;
            }
        }
    }

    best.is_finite().then_some(best)
}

fn grid_has_full_periodic_longitude_axis(grid: &rustwx_core::LatLonGrid) -> bool {
    let nx = grid.shape.nx;
    let ny = grid.shape.ny;
    if nx < 2 || ny == 0 || grid.lon_deg.len() < nx {
        return false;
    }

    let lon0 = grid.lon_deg[0] as f64;
    let lon1 = grid.lon_deg[1] as f64;
    let mut step = lon1 - lon0;
    if step > 180.0 {
        step -= 360.0;
    } else if step < -180.0 {
        step += 360.0;
    }
    let step = step.abs();
    if !step.is_finite() || step < 1.0e-9 {
        return false;
    }

    let tol = (step * 1.5).max(1.0e-6);
    ((nx as f64 * step) - 360.0).abs() <= tol || (((nx - 1) as f64 * step) - 360.0).abs() <= tol
}

fn longitude_delta_deg(a: f64, b: f64) -> f64 {
    let mut delta = (normalize_longitude_for_bounds(b) - normalize_longitude_for_bounds(a)).abs();
    if delta > 180.0 {
        delta = 360.0 - delta;
    }
    delta
}

pub(super) fn crop_selected_field_for_domain(
    field: &SelectedField2D,
    bounds: (f64, f64, f64, f64),
    pad_cells: usize,
    preserve_full_longitude_axis: bool,
) -> Result<SelectedField2D, Box<dyn std::error::Error>> {
    let Some(crop) =
        crop_for_direct_grid(&field.grid, bounds, pad_cells, preserve_full_longitude_axis)?
    else {
        return Ok(field.clone());
    };
    let mut cropped_grid = crop_latlon_grid_for_direct(&field.grid, crop)?;
    if should_normalize_periodic_direct_crop_longitudes(field, bounds, preserve_full_longitude_axis)
    {
        normalize_grid_longitudes_around(&mut cropped_grid, center_longitude_for_bounds(bounds));
    }
    let mut cropped = SelectedField2D::new(
        field.selector,
        field.units.clone(),
        cropped_grid,
        crop_values_f32_for_direct(&field.values, field.grid.shape.nx, crop),
    )?;
    if let Some(projection) = field.projection.clone() {
        cropped = cropped.with_projection(projection);
    }
    Ok(cropped)
}

fn should_normalize_periodic_direct_crop_longitudes(
    field: &SelectedField2D,
    bounds: (f64, f64, f64, f64),
    preserve_full_longitude_axis: bool,
) -> bool {
    preserve_full_longitude_axis
        && longitude_bounds_span_deg(bounds) < 359.0
        && direct_field_has_periodic_geographic_axis(field)
}

fn normalize_grid_longitudes_around(grid: &mut rustwx_core::LatLonGrid, center_lon: f64) {
    for lon in &mut grid.lon_deg {
        let mut value = *lon as f64;
        while value - center_lon > 180.0 {
            value -= 360.0;
        }
        while value - center_lon < -180.0 {
            value += 360.0;
        }
        *lon = value as f32;
    }
}

fn direct_field_has_periodic_geographic_axis(field: &SelectedField2D) -> bool {
    let regular_latlon = matches!(field.projection.as_ref(), Some(GridProjection::Geographic))
        || (field.projection.is_none()
            && rectilinear_latlon_mesh_for_inverse(&field.grid.lat_deg, &field.grid.lon_deg));
    regular_latlon && grid_has_full_periodic_longitude_axis(&field.grid)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DirectGridCrop {
    Contiguous(GridCrop),
    Wrapped {
        x_start: usize,
        x_end: usize,
        y_start: usize,
        y_end: usize,
    },
}

impl DirectGridCrop {
    fn width(self, source_nx: usize) -> usize {
        match self {
            Self::Contiguous(crop) => crop.width(),
            Self::Wrapped { x_start, x_end, .. } => source_nx - x_start + x_end,
        }
    }

    fn height(self) -> usize {
        match self {
            Self::Contiguous(crop) => crop.height(),
            Self::Wrapped { y_start, y_end, .. } => y_end - y_start,
        }
    }

    fn is_full(self, source_nx: usize, source_ny: usize) -> bool {
        match self {
            Self::Contiguous(crop) => {
                crop.x_start == 0
                    && crop.x_end == source_nx
                    && crop.y_start == 0
                    && crop.y_end == source_ny
            }
            Self::Wrapped { .. } => false,
        }
    }
}

pub(super) fn crop_latlon_grid_for_direct(
    grid: &rustwx_core::LatLonGrid,
    crop: DirectGridCrop,
) -> Result<rustwx_core::LatLonGrid, Box<dyn std::error::Error>> {
    if let DirectGridCrop::Contiguous(crop) = crop {
        return crop_latlon_grid(grid, crop);
    }

    Ok(rustwx_core::LatLonGrid::new(
        rustwx_core::GridShape::new(crop.width(grid.shape.nx), crop.height())?,
        crop_values_f32_for_direct(&grid.lat_deg, grid.shape.nx, crop),
        crop_values_f32_for_direct(&grid.lon_deg, grid.shape.nx, crop),
    )?)
}

fn crop_values_f32_for_direct(values: &[f32], source_nx: usize, crop: DirectGridCrop) -> Vec<f32> {
    match crop {
        DirectGridCrop::Contiguous(crop) => crop_values_f32(values, source_nx, crop),
        DirectGridCrop::Wrapped {
            x_start,
            x_end,
            y_start,
            y_end,
        } => {
            let mut cropped = Vec::with_capacity(crop.width(source_nx) * crop.height());
            for y in y_start..y_end {
                let row_start = y * source_nx;
                cropped.extend_from_slice(&values[row_start + x_start..row_start + source_nx]);
                cropped.extend_from_slice(&values[row_start..row_start + x_end]);
            }
            cropped
        }
    }
}

pub(super) fn crop_for_direct_grid(
    grid: &rustwx_core::LatLonGrid,
    bounds: (f64, f64, f64, f64),
    pad_cells: usize,
    preserve_full_longitude_axis: bool,
) -> Result<Option<DirectGridCrop>, Box<dyn std::error::Error>> {
    let nx = grid.shape.nx;
    let ny = grid.shape.ny;
    if nx == 0 || ny == 0 {
        return Ok(None);
    }

    let mut hit_columns = vec![false; nx];
    let mut min_x = nx;
    let mut max_x = 0usize;
    let mut min_y = ny;
    let mut max_y = 0usize;
    let mut found = false;

    for y in 0..ny {
        let row_offset = y * nx;
        for x in 0..nx {
            let idx = row_offset + x;
            let lat = grid.lat_deg[idx] as f64;
            let lon = grid.lon_deg[idx] as f64;
            if point_in_geographic_bounds(lon, lat, bounds) {
                min_x = min_x.min(x);
                max_x = max_x.max(x);
                min_y = min_y.min(y);
                max_y = max_y.max(y);
                hit_columns[x] = true;
                found = true;
            }
        }
    }

    if !found {
        return Ok(None);
    }

    let y_start = min_y.saturating_sub(pad_cells);
    let y_end = (max_y + 1 + pad_cells).min(ny);

    let crop = if preserve_full_longitude_axis
        && longitude_bounds_span_deg(bounds) < 359.0
        && grid_has_full_periodic_longitude_axis(grid)
    {
        circular_crop_for_hit_columns(&hit_columns, y_start, y_end, pad_cells).unwrap_or(
            DirectGridCrop::Contiguous(GridCrop {
                x_start: 0,
                x_end: nx,
                y_start,
                y_end,
            }),
        )
    } else {
        DirectGridCrop::Contiguous(GridCrop {
            x_start: if preserve_full_longitude_axis {
                0
            } else {
                min_x.saturating_sub(pad_cells)
            },
            x_end: if preserve_full_longitude_axis {
                nx
            } else {
                (max_x + 1 + pad_cells).min(nx)
            },
            y_start,
            y_end,
        })
    };

    if crop.is_full(nx, ny) {
        Ok(None)
    } else {
        Ok(Some(crop))
    }
}

fn circular_crop_for_hit_columns(
    hit_columns: &[bool],
    y_start: usize,
    y_end: usize,
    pad_cells: usize,
) -> Option<DirectGridCrop> {
    let nx = hit_columns.len();
    if nx == 0 {
        return None;
    }
    let hits = hit_columns
        .iter()
        .enumerate()
        .filter_map(|(index, hit)| hit.then_some(index))
        .collect::<Vec<_>>();
    if hits.is_empty() {
        return None;
    }
    if hits.len() == nx {
        return Some(DirectGridCrop::Contiguous(GridCrop {
            x_start: 0,
            x_end: nx,
            y_start,
            y_end,
        }));
    }

    let mut largest_gap = 0usize;
    let mut gap_before_hit = hits[0];
    let mut gap_after_hit = *hits.last().unwrap();
    for idx in 0..hits.len() {
        let current = hits[idx];
        let next = hits[(idx + 1) % hits.len()];
        let gap = if next > current {
            next - current - 1
        } else {
            next + nx - current - 1
        };
        if gap > largest_gap {
            largest_gap = gap;
            gap_after_hit = current;
            gap_before_hit = next;
        }
    }

    if largest_gap <= pad_cells.saturating_mul(2) {
        return Some(DirectGridCrop::Contiguous(GridCrop {
            x_start: 0,
            x_end: nx,
            y_start,
            y_end,
        }));
    }

    let x_start = (gap_before_hit + nx - (pad_cells % nx)) % nx;
    let x_end_inclusive = (gap_after_hit + pad_cells) % nx;
    if x_start <= x_end_inclusive {
        Some(DirectGridCrop::Contiguous(GridCrop {
            x_start,
            x_end: x_end_inclusive + 1,
            y_start,
            y_end,
        }))
    } else {
        Some(DirectGridCrop::Wrapped {
            x_start,
            x_end: x_end_inclusive + 1,
            y_start,
            y_end,
        })
    }
}

pub(super) fn range_step(start: f64, stop: f64, step: f64) -> Vec<f64> {
    let mut values = Vec::new();
    let mut current = start;
    while current < stop - step * 1.0e-9 {
        values.push(current);
        current += step;
    }
    values
}

pub(super) fn visible_grid_span(
    grid: &rustwx_core::LatLonGrid,
    bounds: (f64, f64, f64, f64),
) -> (usize, usize) {
    let mut min_i = usize::MAX;
    let mut max_i = 0usize;
    let mut min_j = usize::MAX;
    let mut max_j = 0usize;

    for j in 0..grid.shape.ny {
        for i in 0..grid.shape.nx {
            let idx = j * grid.shape.nx + i;
            let lat = grid.lat_deg[idx] as f64;
            let lon = grid.lon_deg[idx] as f64;
            if point_in_geographic_bounds(lon, lat, bounds) {
                min_i = min_i.min(i);
                max_i = max_i.max(i);
                min_j = min_j.min(j);
                max_j = max_j.max(j);
            }
        }
    }

    if min_i == usize::MAX || min_j == usize::MAX {
        return (grid.shape.nx.max(1), grid.shape.ny.max(1));
    }

    (max_i - min_i + 1, max_j - min_j + 1)
}

pub(super) fn point_in_geographic_bounds(lon: f64, lat: f64, bounds: (f64, f64, f64, f64)) -> bool {
    if !lon.is_finite() || !lat.is_finite() || lat < bounds.2 || lat > bounds.3 {
        return false;
    }
    if longitude_bounds_span_deg(bounds) >= 359.0 {
        return true;
    }
    let west = normalize_longitude_for_bounds(bounds.0);
    let east = normalize_longitude_for_bounds(bounds.1);
    let lon = normalize_longitude_for_bounds(lon);
    if west <= east {
        lon >= west && lon <= east
    } else {
        lon >= west || lon <= east
    }
}

pub(super) fn normalize_longitude_for_bounds(lon: f64) -> f64 {
    let mut lon = lon % 360.0;
    if lon > 180.0 {
        lon -= 360.0;
    } else if lon <= -180.0 {
        lon += 360.0;
    }
    lon
}

pub(crate) fn is_global_scale_domain(bounds: (f64, f64, f64, f64)) -> bool {
    crate::plot_design::is_global_scale_domain(bounds)
}

pub(super) fn longitude_bounds_span_deg(bounds: (f64, f64, f64, f64)) -> f64 {
    let raw_span = (bounds.1 - bounds.0).abs();
    if raw_span >= 359.0 {
        return raw_span.min(360.0);
    }

    let west = normalize_longitude_for_bounds(bounds.0);
    let east = normalize_longitude_for_bounds(bounds.1);
    if west <= east {
        east - west
    } else {
        east + 360.0 - west
    }
}
