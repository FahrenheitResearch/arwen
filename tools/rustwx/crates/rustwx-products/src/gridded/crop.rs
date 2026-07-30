use std::path::{Path, PathBuf};

use rustwx_core::{GridShape, LatLonGrid};
use rustwx_io::FetchRequest;
use rustwx_render::ProjectedExtent;

use super::{
    PressureFields, SurfaceFields, SurfaceGridLayout, decode_cache_path, point_in_geographic_bounds,
};

/// Subrectangle of a generic surface/pressure grid that the heavy
/// compute kernels can operate on. Used to keep ECAPE/severe runs fast
/// on regional renders that only need a fraction of the source domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GridCrop {
    pub x_start: usize,
    pub x_end: usize,
    pub y_start: usize,
    pub y_end: usize,
}

impl GridCrop {
    pub fn width(self) -> usize {
        self.x_end - self.x_start
    }

    pub fn height(self) -> usize {
        self.y_end - self.y_start
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectedGridIntersection {
    Empty,
    Full,
    Crop(GridCrop),
}

/// Cropped surface+pressure pair plus the recomputed `LatLonGrid`. The
/// generic loader returns full-domain decoded fields; lane runners that
/// know they only need a regional slice (mainly `hrrr_batch` for
/// severe/ECAPE on a small bounding box) crop with this helper.
#[derive(Debug, Clone)]
pub struct CroppedHeavyDomain {
    pub surface: SurfaceFields,
    pub pressure: PressureFields,
    pub grid: LatLonGrid,
}

pub fn crop_heavy_domain(
    surface: &SurfaceFields,
    pressure: &PressureFields,
    bounds: (f64, f64, f64, f64),
) -> Result<Option<CroppedHeavyDomain>, Box<dyn std::error::Error>> {
    let Some(crop) = crop_rect_for_bounds(surface, bounds)? else {
        return Ok(None);
    };
    let cropped_surface = crop_surface_fields(surface, crop);
    let cropped_pressure = crop_pressure_fields(pressure, surface.nx, surface.ny, crop)?;
    let grid = cropped_surface.core_grid()?;
    Ok(Some(CroppedHeavyDomain {
        surface: cropped_surface,
        pressure: cropped_pressure,
        grid,
    }))
}

pub fn crop_heavy_domain_for_projected_extent(
    surface: &SurfaceFields,
    pressure: &PressureFields,
    projected_x: &[f64],
    projected_y: &[f64],
    extent: &ProjectedExtent,
    pad_cells: usize,
) -> Result<Option<CroppedHeavyDomain>, Box<dyn std::error::Error>> {
    let crop = match classify_projected_grid_intersection(
        surface.nx,
        surface.ny,
        projected_x,
        projected_y,
        extent,
        pad_cells,
    )? {
        ProjectedGridIntersection::Empty => {
            return Err("requested projected crop produced an empty heavy-compute domain".into());
        }
        ProjectedGridIntersection::Full => return Ok(None),
        ProjectedGridIntersection::Crop(crop) => crop,
    };
    let cropped_surface = crop_surface_fields(surface, crop);
    let cropped_pressure = crop_pressure_fields(pressure, surface.nx, surface.ny, crop)?;
    let grid = cropped_surface.core_grid()?;
    Ok(Some(CroppedHeavyDomain {
        surface: cropped_surface,
        pressure: cropped_pressure,
        grid,
    }))
}

pub fn classify_projected_grid_intersection(
    nx: usize,
    ny: usize,
    projected_x: &[f64],
    projected_y: &[f64],
    extent: &ProjectedExtent,
    pad_cells: usize,
) -> Result<ProjectedGridIntersection, Box<dyn std::error::Error>> {
    let expected_len = nx * ny;
    if projected_x.len() != expected_len || projected_y.len() != expected_len {
        return Err("projected crop inputs did not match surface grid size".into());
    }

    let mut min_x = nx;
    let mut max_x = 0usize;
    let mut min_y = ny;
    let mut max_y = 0usize;
    let mut found = false;

    for y in 0..ny {
        let row_offset = y * nx;
        for x in 0..nx {
            let idx = row_offset + x;
            let px = projected_x[idx];
            let py = projected_y[idx];
            if px >= extent.x_min && px <= extent.x_max && py >= extent.y_min && py <= extent.y_max
            {
                min_x = min_x.min(x);
                max_x = max_x.max(x);
                min_y = min_y.min(y);
                max_y = max_y.max(y);
                found = true;
            }
        }
    }

    if !found {
        return Ok(ProjectedGridIntersection::Empty);
    }

    let crop = GridCrop {
        x_start: min_x.saturating_sub(pad_cells),
        x_end: (max_x + 1 + pad_cells).min(nx),
        y_start: min_y.saturating_sub(pad_cells),
        y_end: (max_y + 1 + pad_cells).min(ny),
    };
    if crop.x_start == 0 && crop.x_end == nx && crop.y_start == 0 && crop.y_end == ny {
        Ok(ProjectedGridIntersection::Full)
    } else {
        Ok(ProjectedGridIntersection::Crop(crop))
    }
}

pub fn crop_values_f64(values: &[f64], source_nx: usize, crop: GridCrop) -> Vec<f64> {
    crop_2d_values(values, source_nx, crop)
}

pub fn crop_values_f32(values: &[f32], source_nx: usize, crop: GridCrop) -> Vec<f32> {
    let mut cropped = Vec::with_capacity(crop.width() * crop.height());
    for y in crop.y_start..crop.y_end {
        let start = y * source_nx + crop.x_start;
        let end = y * source_nx + crop.x_end;
        cropped.extend_from_slice(&values[start..end]);
    }
    cropped
}

pub fn crop_latlon_grid(
    grid: &LatLonGrid,
    crop: GridCrop,
) -> Result<LatLonGrid, Box<dyn std::error::Error>> {
    Ok(LatLonGrid::new(
        GridShape::new(crop.width(), crop.height())?,
        crop_values_f32(&grid.lat_deg, grid.shape.nx, crop),
        crop_values_f32(&grid.lon_deg, grid.shape.nx, crop),
    )?)
}

/// Compute the tightest [`GridCrop`] (contiguous index block) whose cells'
/// geographic coordinates intersect `bounds = (west, east, south, north)`, on a
/// per-cell `LatLonGrid` (regular, Lambert, or rotated-pole-unrotated — same
/// generic per-cell scan the heavy-compute crop uses). `None` when no cell is
/// in bounds; the full grid (no crop) is reported as `Some(full)` so callers
/// can compare. The crop index range is a deterministic pure function of the
/// grid's per-cell lat/lon — no per-run floating-point branch — so re-ingesting
/// an hour yields a bit-identical cropped grid.
pub fn grid_crop_for_bounds(grid: &LatLonGrid, bounds: (f64, f64, f64, f64)) -> Option<GridCrop> {
    let nx = grid.shape.nx;
    let ny = grid.shape.ny;
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
                found = true;
            }
        }
    }
    if !found {
        return None;
    }
    Some(GridCrop {
        x_start: min_x,
        x_end: max_x + 1,
        y_start: min_y,
        y_end: max_y + 1,
    })
}

/// Crop a decoded surface+pressure pair by an explicit [`GridCrop`] (the index
/// block the caller already computed from the store grid), plus the recomputed
/// `LatLonGrid`. Used by the ingest crop, where ONE crop spec — computed from
/// the extraction reference grid — must drive both the extracted field planes
/// AND the derived/heavy compute domain in lock-step.
pub fn crop_heavy_domain_with(
    surface: &SurfaceFields,
    pressure: &PressureFields,
    crop: GridCrop,
) -> Result<CroppedHeavyDomain, Box<dyn std::error::Error>> {
    let cropped_surface = crop_surface_fields(surface, crop);
    let cropped_pressure = crop_pressure_fields(pressure, surface.nx, surface.ny, crop)?;
    let grid = cropped_surface.core_grid()?;
    Ok(CroppedHeavyDomain {
        surface: cropped_surface,
        pressure: cropped_pressure,
        grid,
    })
}

pub(super) fn crop_rect_for_layout(
    layout: &SurfaceGridLayout,
    bounds: (f64, f64, f64, f64),
) -> Result<Option<GridCrop>, Box<dyn std::error::Error>> {
    let mut min_x = layout.nx;
    let mut max_x = 0usize;
    let mut min_y = layout.ny;
    let mut max_y = 0usize;
    let mut found = false;

    for y in 0..layout.ny {
        let row_offset = y * layout.nx;
        for x in 0..layout.nx {
            let idx = row_offset + x;
            let lat = layout.lat[idx];
            let lon = layout.lon[idx];
            if point_in_geographic_bounds(lon, lat, bounds) {
                min_x = min_x.min(x);
                max_x = max_x.max(x);
                min_y = min_y.min(y);
                max_y = max_y.max(y);
                found = true;
            }
        }
    }

    if !found {
        return Ok(None);
    }

    Ok(Some(GridCrop {
        x_start: min_x,
        x_end: max_x + 1,
        y_start: min_y,
        y_end: max_y + 1,
    }))
}

fn crop_rect_for_bounds(
    surface: &SurfaceFields,
    bounds: (f64, f64, f64, f64),
) -> Result<Option<GridCrop>, Box<dyn std::error::Error>> {
    let mut min_x = surface.nx;
    let mut max_x = 0usize;
    let mut min_y = surface.ny;
    let mut max_y = 0usize;
    let mut found = false;

    for y in 0..surface.ny {
        let row_offset = y * surface.nx;
        for x in 0..surface.nx {
            let idx = row_offset + x;
            let lat = surface.lat[idx];
            let lon = surface.lon[idx];
            if point_in_geographic_bounds(lon, lat, bounds) {
                min_x = min_x.min(x);
                max_x = max_x.max(x);
                min_y = min_y.min(y);
                max_y = max_y.max(y);
                found = true;
            }
        }
    }

    if !found {
        return Err("requested crop produced an empty heavy-compute domain".into());
    }

    let crop = GridCrop {
        x_start: min_x,
        x_end: max_x + 1,
        y_start: min_y,
        y_end: max_y + 1,
    };

    if crop.x_start == 0
        && crop.x_end == surface.nx
        && crop.y_start == 0
        && crop.y_end == surface.ny
    {
        Ok(None)
    } else {
        Ok(Some(crop))
    }
}

pub(super) fn crop_surface_fields(surface: &SurfaceFields, crop: GridCrop) -> SurfaceFields {
    SurfaceFields {
        lat: crop_2d_values(&surface.lat, surface.nx, crop),
        lon: crop_2d_values(&surface.lon, surface.nx, crop),
        nx: crop.width(),
        ny: crop.height(),
        projection: surface.projection.clone(),
        psfc_pa: crop_2d_values(&surface.psfc_pa, surface.nx, crop),
        orog_m: crop_2d_values(&surface.orog_m, surface.nx, crop),
        orog_is_proxy: surface.orog_is_proxy,
        t2_k: crop_2d_values(&surface.t2_k, surface.nx, crop),
        q2_kgkg: crop_2d_values(&surface.q2_kgkg, surface.nx, crop),
        u10_ms: crop_2d_values(&surface.u10_ms, surface.nx, crop),
        v10_ms: crop_2d_values(&surface.v10_ms, surface.nx, crop),
        native_sbcape_jkg: crop_optional_2d_values(&surface.native_sbcape_jkg, surface.nx, crop),
        native_mlcape_jkg: crop_optional_2d_values(&surface.native_mlcape_jkg, surface.nx, crop),
        native_mucape_jkg: crop_optional_2d_values(&surface.native_mucape_jkg, surface.nx, crop),
        native_pblh_m: crop_optional_2d_values(&surface.native_pblh_m, surface.nx, crop),
    }
}

pub(super) fn crop_pressure_fields(
    pressure: &PressureFields,
    source_nx: usize,
    source_ny: usize,
    crop: GridCrop,
) -> Result<PressureFields, Box<dyn std::error::Error>> {
    let level_count = pressure.pressure_levels_hpa.len();
    let expected_len = source_nx
        .checked_mul(source_ny)
        .and_then(|n2d| n2d.checked_mul(level_count))
        .ok_or("pressure crop expected length overflowed")?;
    if let Some(values) = pressure.pressure_3d_pa.as_ref() {
        if values.len() != expected_len {
            return Err(format!(
                "pressure field pressure_3d_pa length {} did not match expected source volume length {expected_len}",
                values.len()
            )
            .into());
        }
    }
    for (name, values) in [
        ("temperature_c_3d", &pressure.temperature_c_3d),
        ("qvapor_kgkg_3d", &pressure.qvapor_kgkg_3d),
        ("u_ms_3d", &pressure.u_ms_3d),
        ("v_ms_3d", &pressure.v_ms_3d),
        ("gh_m_3d", &pressure.gh_m_3d),
    ] {
        if values.len() != expected_len {
            return Err(format!(
                "pressure field {name} length {} did not match expected source volume length {expected_len}",
                values.len()
            )
            .into());
        }
    }
    for (name, values) in [
        ("omega_pa_s_3d", pressure.omega_pa_s_3d.as_ref()),
        (
            "absolute_vorticity_s_3d",
            pressure.absolute_vorticity_s_3d.as_ref(),
        ),
        (
            "cloud_liquid_kgkg_3d",
            pressure.cloud_liquid_kgkg_3d.as_ref(),
        ),
        ("cloud_ice_kgkg_3d", pressure.cloud_ice_kgkg_3d.as_ref()),
        ("rain_kgkg_3d", pressure.rain_kgkg_3d.as_ref()),
        ("snow_kgkg_3d", pressure.snow_kgkg_3d.as_ref()),
        ("graupel_kgkg_3d", pressure.graupel_kgkg_3d.as_ref()),
    ] {
        if let Some(values) = values {
            if values.len() != expected_len {
                return Err(format!(
                    "pressure field {name} length {} did not match expected source volume length {expected_len}",
                    values.len()
                )
                .into());
            }
        }
    }

    Ok(PressureFields {
        pressure_levels_hpa: pressure.pressure_levels_hpa.clone(),
        pressure_3d_pa: pressure
            .pressure_3d_pa
            .as_ref()
            .map(|values| crop_3d_values(values, source_nx, source_ny, level_count, crop)),
        temperature_c_3d: crop_3d_values(
            &pressure.temperature_c_3d,
            source_nx,
            source_ny,
            level_count,
            crop,
        ),
        qvapor_kgkg_3d: crop_3d_values(
            &pressure.qvapor_kgkg_3d,
            source_nx,
            source_ny,
            level_count,
            crop,
        ),
        u_ms_3d: crop_3d_values(&pressure.u_ms_3d, source_nx, source_ny, level_count, crop),
        v_ms_3d: crop_3d_values(&pressure.v_ms_3d, source_nx, source_ny, level_count, crop),
        gh_m_3d: crop_3d_values(&pressure.gh_m_3d, source_nx, source_ny, level_count, crop),
        omega_pa_s_3d: pressure
            .omega_pa_s_3d
            .as_ref()
            .map(|values| crop_3d_values(values, source_nx, source_ny, level_count, crop)),
        absolute_vorticity_s_3d: pressure
            .absolute_vorticity_s_3d
            .as_ref()
            .map(|values| crop_3d_values(values, source_nx, source_ny, level_count, crop)),
        cloud_liquid_kgkg_3d: pressure
            .cloud_liquid_kgkg_3d
            .as_ref()
            .map(|values| crop_3d_values(values, source_nx, source_ny, level_count, crop)),
        cloud_ice_kgkg_3d: pressure
            .cloud_ice_kgkg_3d
            .as_ref()
            .map(|values| crop_3d_values(values, source_nx, source_ny, level_count, crop)),
        rain_kgkg_3d: pressure
            .rain_kgkg_3d
            .as_ref()
            .map(|values| crop_3d_values(values, source_nx, source_ny, level_count, crop)),
        snow_kgkg_3d: pressure
            .snow_kgkg_3d
            .as_ref()
            .map(|values| crop_3d_values(values, source_nx, source_ny, level_count, crop)),
        graupel_kgkg_3d: pressure
            .graupel_kgkg_3d
            .as_ref()
            .map(|values| crop_3d_values(values, source_nx, source_ny, level_count, crop)),
    })
}

pub(super) fn crop_2d_values(values: &[f64], source_nx: usize, crop: GridCrop) -> Vec<f64> {
    let mut out = Vec::with_capacity(crop.width() * crop.height());
    for y in crop.y_start..crop.y_end {
        let row_start = y * source_nx + crop.x_start;
        let row_end = row_start + crop.width();
        out.extend_from_slice(&values[row_start..row_end]);
    }
    out
}

pub(super) fn crop_optional_2d_values(
    values: &Option<Vec<f64>>,
    source_nx: usize,
    crop: GridCrop,
) -> Option<Vec<f64>> {
    values
        .as_ref()
        .map(|values| crop_2d_values(values, source_nx, crop))
}

pub(super) fn cropped_decode_cache_path(
    cache_root: &Path,
    fetch: &FetchRequest,
    name: &str,
    crop: GridCrop,
) -> PathBuf {
    let mut path = decode_cache_path(cache_root, fetch, name);
    let stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or(name)
        .to_string();
    let suffix = format!(
        "{stem}_crop_{}_{}_{}_{}",
        crop.x_start, crop.x_end, crop.y_start, crop.y_end
    );
    path.set_file_name(format!("{suffix}.bin"));
    path
}

fn crop_3d_values(
    values: &[f64],
    source_nx: usize,
    source_ny: usize,
    level_count: usize,
    crop: GridCrop,
) -> Vec<f64> {
    let source_n2d = source_nx * source_ny;
    let mut out = Vec::with_capacity(crop.width() * crop.height() * level_count);
    for level in 0..level_count {
        let level_offset = level * source_n2d;
        for y in crop.y_start..crop.y_end {
            let row_start = level_offset + y * source_nx + crop.x_start;
            let row_end = row_start + crop.width();
            out.extend_from_slice(&values[row_start..row_end]);
        }
    }
    out
}
