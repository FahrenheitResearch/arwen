use crate::color::Rgba;
use crate::colormap::LeveledColormap;
use crate::overlay::MapExtent;
use crate::projection::ProjectionProjector;
use crate::request::{GeographicClipBounds, RasterSampleMode};
use image::RgbaImage;

fn projected_pixel_to_f64(point: (f32, f32)) -> (f64, f64) {
    (point.0 as f64, point.1 as f64)
}

// API-compat stubs mirroring the upstream no-cuda build; always empty.
pub fn cuda_rasterize_stats() -> [(&'static str, usize); 0] {
    []
}

pub fn print_cuda_rasterize_stats_if_enabled() {}

/// Rasterize a 2D grid into an RGBA image using bilinear sampling.
///
/// `data` is row-major `[ny][nx]`. The image maps grid row 0 to the
/// bottom of the image (geographic convention: south at bottom).
pub fn rasterize_grid(
    data: &[f64],
    ny: usize,
    nx: usize,
    cmap: &LeveledColormap,
    sample_mode: RasterSampleMode,
    img_w: u32,
    img_h: u32,
) -> RgbaImage {
    let mut img = RgbaImage::new(img_w, img_h);

    if ny == 0 || nx == 0 {
        return img;
    }

    let x_den = img_w.saturating_sub(1).max(1) as f64;
    let y_den = img_h.saturating_sub(1).max(1) as f64;
    let gx_den = nx.saturating_sub(1).max(1) as f64;
    let gy_den = ny.saturating_sub(1).max(1) as f64;

    for py in 0..img_h {
        for px in 0..img_w {
            let gx = px as f64 / x_den * gx_den;
            let gy = (img_h.saturating_sub(1) - py) as f64 / y_den * gy_den;

            let value = match sample_mode {
                RasterSampleMode::Nearest => {
                    let i = (gx.round() as usize).min(nx - 1);
                    let j = (gy.round() as usize).min(ny - 1);
                    data[j * nx + i]
                }
                RasterSampleMode::Linear => {
                    let i0 = gx.floor() as usize;
                    let j0 = gy.floor() as usize;
                    let i1 = (i0 + 1).min(nx - 1);
                    let j1 = (j0 + 1).min(ny - 1);
                    let fx = gx - i0 as f64;
                    let fy = gy - j0 as f64;

                    let v00 = data[j0 * nx + i0];
                    let v10 = data[j0 * nx + i1];
                    let v01 = data[j1 * nx + i0];
                    let v11 = data[j1 * nx + i1];

                    bilinear(v00, v10, v01, v11, fx, fy)
                }
            };
            let color = cmap.map(value);
            img.put_pixel(px, py, color.to_image_rgba());
        }
    }

    img
}

/// Rasterize a 2D grid on a projected mesh.
///
/// `pixel_points` contains local image coordinates in map space, one per grid
/// point, or `None` when the projected point falls outside the valid extent.
///
/// `sample_mode` decides what a pixel between grid points shows: the
/// interpolated value, or the value of the nearest grid point. Nearest is
/// what a category field needs, because a value between two codes is not a
/// code, so both the rectilinear and the triangle path honour it.
pub fn rasterize_projected_grid(
    data: &[f64],
    ny: usize,
    nx: usize,
    pixel_points: &[Option<(f32, f32)>],
    cmap: &LeveledColormap,
    sample_mode: RasterSampleMode,
    img_w: u32,
    img_h: u32,
) -> RgbaImage {
    if let Some(mut img) = rasterize_rectilinear_projected_grid(
        data,
        ny,
        nx,
        pixel_points,
        cmap,
        sample_mode,
        img_w,
        img_h,
    ) {
        feather_projected_raster_edges(&mut img);
        return img;
    }

    let mut img = RgbaImage::new(img_w, img_h);

    if ny < 2 || nx < 2 || pixel_points.len() != ny * nx {
        return img;
    }

    for j in 0..(ny - 1) {
        for i in 0..(nx - 1) {
            let idx = |jj: usize, ii: usize| jj * nx + ii;
            let p00 = pixel_points[idx(j, i)];
            let p10 = pixel_points[idx(j, i + 1)];
            let p01 = pixel_points[idx(j + 1, i)];
            let p11 = pixel_points[idx(j + 1, i + 1)];

            let (p00, p10, p01, p11) = match (p00, p10, p01, p11) {
                (Some(a), Some(b), Some(c), Some(d)) => (a, b, c, d),
                _ => continue,
            };

            let v00 = data[idx(j, i)];
            let v10 = data[idx(j, i + 1)];
            let v01 = data[idx(j + 1, i)];
            let v11 = data[idx(j + 1, i + 1)];

            let p00 = projected_pixel_to_f64(p00);
            let p10 = projected_pixel_to_f64(p10);
            let p01 = projected_pixel_to_f64(p01);
            let p11 = projected_pixel_to_f64(p11);
            rasterize_triangle(&mut img, p00, v00, p10, v10, p11, v11, cmap, sample_mode);
            rasterize_triangle(&mut img, p00, v00, p11, v11, p01, v01, cmap, sample_mode);
        }
    }

    feather_projected_raster_edges(&mut img);
    img
}

#[derive(Clone, Copy)]
struct RectilinearPixelAxes {
    x0: f64,
    y0: f64,
    dx: f64,
    dy: f64,
}

fn rectilinear_pixel_axes(
    ny: usize,
    nx: usize,
    pixel_points: &[Option<(f32, f32)>],
) -> Option<RectilinearPixelAxes> {
    if ny < 2 || nx < 2 || pixel_points.len() != ny * nx {
        return None;
    }
    let first = pixel_points[0]?;
    let east = pixel_points[nx - 1]?;
    let south = pixel_points[(ny - 1) * nx]?;
    let first = projected_pixel_to_f64(first);
    let east = projected_pixel_to_f64(east);
    let south = projected_pixel_to_f64(south);
    let dx = (east.0 - first.0) / (nx - 1) as f64;
    let dy = (south.1 - first.1) / (ny - 1) as f64;
    if !dx.is_finite() || !dy.is_finite() || dx.abs() < 1.0e-9 || dy.abs() < 1.0e-9 {
        return None;
    }

    let tolerance = 0.08_f64;
    let sample_rows = [0, ny / 2, ny - 1];
    let sample_cols = [0, nx / 2, nx - 1];
    for &row in &sample_rows {
        for &col in &sample_cols {
            let point = projected_pixel_to_f64(pixel_points[row * nx + col]?);
            let expected_x = first.0 + dx * col as f64;
            let expected_y = first.1 + dy * row as f64;
            if (point.0 - expected_x).abs() > tolerance || (point.1 - expected_y).abs() > tolerance
            {
                return None;
            }
        }
    }

    Some(RectilinearPixelAxes {
        x0: first.0,
        y0: first.1,
        dx,
        dy,
    })
}

#[derive(Clone, Copy)]
struct AxisSample {
    lower: usize,
    upper: usize,
    fraction: f64,
}

fn axis_samples(
    origin: f64,
    delta: f64,
    count: usize,
    pixel_count: u32,
) -> Vec<Option<AxisSample>> {
    let mut samples = Vec::with_capacity(pixel_count as usize);
    let max_grid = count.saturating_sub(1) as f64;
    for pixel in 0..pixel_count {
        let grid = (pixel as f64 - origin) / delta;
        if !grid.is_finite() || grid < 0.0 || grid > max_grid {
            samples.push(None);
            continue;
        }
        let lower = grid.floor().min(max_grid) as usize;
        let upper = (lower + 1).min(count - 1);
        let fraction = if lower == upper {
            0.0
        } else {
            grid - lower as f64
        };
        samples.push(Some(AxisSample {
            lower,
            upper,
            fraction,
        }));
    }
    samples
}

fn rasterize_rectilinear_projected_grid(
    data: &[f64],
    ny: usize,
    nx: usize,
    pixel_points: &[Option<(f32, f32)>],
    cmap: &LeveledColormap,
    sample_mode: RasterSampleMode,
    img_w: u32,
    img_h: u32,
) -> Option<RgbaImage> {
    if data.len() != ny * nx {
        return None;
    }
    let axes = rectilinear_pixel_axes(ny, nx, pixel_points)?;
    let x_samples = axis_samples(axes.x0, axes.dx, nx, img_w);
    let y_samples = axis_samples(axes.y0, axes.dy, ny, img_h);
    let mut img = RgbaImage::new(img_w, img_h);

    for (py, y_sample) in y_samples.iter().enumerate() {
        let Some(y_sample) = y_sample else {
            continue;
        };
        let j0 = y_sample.lower;
        let j1 = y_sample.upper;
        let fy = y_sample.fraction;
        for (px, x_sample) in x_samples.iter().enumerate() {
            let Some(x_sample) = x_sample else {
                continue;
            };
            let i0 = x_sample.lower;
            let i1 = x_sample.upper;
            let fx = x_sample.fraction;
            let idx = |j: usize, i: usize| j * nx + i;
            let value = match sample_mode {
                RasterSampleMode::Linear => {
                    let cell_j0 = j0.min(ny - 2);
                    let cell_i0 = i0.min(nx - 2);
                    if !data[idx(cell_j0, cell_i0)].is_finite()
                        || !data[idx(cell_j0, cell_i0 + 1)].is_finite()
                        || !data[idx(cell_j0 + 1, cell_i0)].is_finite()
                        || !data[idx(cell_j0 + 1, cell_i0 + 1)].is_finite()
                    {
                        continue;
                    }
                    let v00 = data[idx(j0, i0)];
                    let v10 = data[idx(j0, i1)];
                    let v01 = data[idx(j1, i0)];
                    let v11 = data[idx(j1, i1)];
                    bilinear(v00, v10, v01, v11, fx, fy)
                }
                // The grid point this pixel is closest to; a non-finite one
                // maps to transparent and leaves the pixel empty.
                RasterSampleMode::Nearest => {
                    let j = if fy < 0.5 { j0 } else { j1 };
                    let i = if fx < 0.5 { i0 } else { i1 };
                    data[idx(j, i)]
                }
            };
            let color = cmap.map(value).to_image_rgba();
            if color.0[3] > 0 {
                img.put_pixel(px as u32, py as u32, color);
            }
        }
    }

    Some(img)
}

pub(crate) fn rasterize_inverse_projected_grid(
    data: &[f64],
    ny: usize,
    nx: usize,
    lat_deg: &[f64],
    lon_deg: &[f64],
    projector: ProjectionProjector,
    clip_bounds: Option<GeographicClipBounds>,
    extent: &MapExtent,
    cmap: &LeveledColormap,
    sample_mode: RasterSampleMode,
    img_w: u32,
    img_h: u32,
) -> RgbaImage {
    let mut img = RgbaImage::new(img_w, img_h);
    if ny < 2
        || nx < 2
        || data.len() != ny * nx
        || lat_deg.len() != ny * nx
        || lon_deg.len() != ny * nx
    {
        return img;
    }

    let Some(axes) = RegularLatLonAxes::from_grid(lat_deg, lon_deg, ny, nx) else {
        return img;
    };

    let x_den = img_w.saturating_sub(1).max(1) as f64;
    let y_den = img_h.saturating_sub(1).max(1) as f64;
    for py in 0..img_h {
        let y = extent.y_max - (py as f64 / y_den) * (extent.y_max - extent.y_min);
        for px in 0..img_w {
            let x = extent.x_min + (px as f64 / x_den) * (extent.x_max - extent.x_min);
            let Some((lat, lon)) = projector.unproject(x, y) else {
                continue;
            };
            if clip_bounds.is_some_and(|bounds| !bounds.contains(lat, lon)) {
                continue;
            }
            let Some(value) =
                sample_regular_latlon_grid_with_mode(data, &axes, lat, lon, sample_mode)
            else {
                continue;
            };
            let color = cmap.map(value);
            let rgba = color.to_image_rgba();
            if rgba.0[3] > 0 {
                img.put_pixel(px, py, rgba);
            }
        }
    }
    img
}

pub fn rasterize_rgba_grid(
    pixels: &[Rgba],
    ny: usize,
    nx: usize,
    img_w: u32,
    img_h: u32,
) -> RgbaImage {
    let mut img = RgbaImage::new(img_w, img_h);

    if ny == 0 || nx == 0 || pixels.len() != ny * nx {
        return img;
    }

    let x_den = img_w.saturating_sub(1).max(1) as f64;
    let y_den = img_h.saturating_sub(1).max(1) as f64;
    let gx_den = nx.saturating_sub(1).max(1) as f64;
    let gy_den = ny.saturating_sub(1).max(1) as f64;

    for py in 0..img_h {
        for px in 0..img_w {
            let gx = px as f64 / x_den * gx_den;
            let gy = (img_h.saturating_sub(1) - py) as f64 / y_den * gy_den;

            let i0 = gx.floor() as usize;
            let j0 = gy.floor() as usize;
            let i1 = (i0 + 1).min(nx - 1);
            let j1 = (j0 + 1).min(ny - 1);
            let fx = gx - i0 as f64;
            let fy = gy - j0 as f64;

            let c00 = pixels[j0 * nx + i0];
            let c10 = pixels[j0 * nx + i1];
            let c01 = pixels[j1 * nx + i0];
            let c11 = pixels[j1 * nx + i1];
            let color = bilinear_rgba(c00, c10, c01, c11, fx, fy);
            if color.a > 0 {
                img.put_pixel(px, py, color.to_image_rgba());
            }
        }
    }

    img
}

#[derive(Clone, Copy)]
struct RegularLatLonAxes {
    nx: usize,
    ny: usize,
    lat0: f64,
    lat_step: f64,
    lon0: f64,
    lon_step: f64,
    periodic_lon: bool,
    period_points: f64,
}

impl RegularLatLonAxes {
    fn from_grid(lat_deg: &[f64], lon_deg: &[f64], ny: usize, nx: usize) -> Option<Self> {
        if nx < 2 || ny < 2 {
            return None;
        }
        let lat0 = lat_deg[0];
        let lat_last = lat_deg[(ny - 1) * nx];
        let lon0 = lon_deg[0];
        let lon1 = lon_deg[1];
        let lat_step = (lat_last - lat0) / (ny - 1) as f64;
        let lon_step = normalize_axis_lon_delta(lon1 - lon0);
        if !lat_step.is_finite()
            || !lon_step.is_finite()
            || lat_step.abs() < 1.0e-9
            || lon_step.abs() < 1.0e-9
        {
            return None;
        }
        let step = lon_step.abs();
        let tol = (step * 1.5).max(1.0e-6);
        let no_duplicate_full = ((nx as f64 * step) - 360.0).abs() <= tol;
        let duplicate_endpoint_full = (((nx - 1) as f64 * step) - 360.0).abs() <= tol;
        Some(Self {
            nx,
            ny,
            lat0,
            lat_step,
            lon0,
            lon_step,
            periodic_lon: no_duplicate_full || duplicate_endpoint_full,
            period_points: if duplicate_endpoint_full && !no_duplicate_full {
                (nx - 1) as f64
            } else {
                nx as f64
            },
        })
    }
}

#[cfg(test)]
fn sample_regular_latlon_grid(
    data: &[f64],
    axes: &RegularLatLonAxes,
    lat: f64,
    lon: f64,
) -> Option<f64> {
    sample_regular_latlon_grid_with_mode(data, axes, lat, lon, RasterSampleMode::Linear)
}

fn sample_regular_latlon_grid_with_mode(
    data: &[f64],
    axes: &RegularLatLonAxes,
    lat: f64,
    lon: f64,
    sample_mode: RasterSampleMode,
) -> Option<f64> {
    if !lat.is_finite() || !lon.is_finite() {
        return None;
    }
    let gy = (lat - axes.lat0) / axes.lat_step;
    if gy < 0.0 || gy > (axes.ny - 1) as f64 {
        return None;
    }
    let gx = grid_x_for_axis_lon(lon, *axes)?;

    if matches!(sample_mode, RasterSampleMode::Nearest) {
        let i = if axes.periodic_lon {
            (gx.round()).rem_euclid(axes.period_points) as usize
        } else {
            (gx.round() as usize).min(axes.nx - 1)
        };
        let j = (gy.round() as usize).min(axes.ny - 1);
        return Some(data[j * axes.nx + i]);
    }

    let i0 = (gx.floor() as usize).min(axes.nx - 1);
    let j0 = gy.floor() as usize;
    let i1 = if axes.periodic_lon {
        ((i0 + 1) as f64).rem_euclid(axes.period_points) as usize
    } else {
        (i0 + 1).min(axes.nx - 1)
    };
    let j1 = (j0 + 1).min(axes.ny - 1);
    let fx = gx - i0 as f64;
    let fy = gy - j0 as f64;
    let idx = |j: usize, i: usize| j * axes.nx + i;
    Some(bilinear(
        data[idx(j0, i0)],
        data[idx(j0, i1)],
        data[idx(j1, i0)],
        data[idx(j1, i1)],
        fx,
        fy,
    ))
}

fn grid_x_for_axis_lon(lon: f64, axes: RegularLatLonAxes) -> Option<f64> {
    if axes.periodic_lon {
        return Some(((lon - axes.lon0) / axes.lon_step).rem_euclid(axes.period_points));
    }

    let mut adjusted = normalize_longitude_deg(lon);
    let axis_center = axes.lon0 + axes.lon_step * (axes.nx - 1) as f64 / 2.0;
    while adjusted - axis_center > 180.0 {
        adjusted -= 360.0;
    }
    while adjusted - axis_center < -180.0 {
        adjusted += 360.0;
    }

    let gx = (adjusted - axes.lon0) / axes.lon_step;
    (gx >= 0.0 && gx <= (axes.nx - 1) as f64).then_some(gx)
}

fn normalize_axis_lon_delta(delta: f64) -> f64 {
    if delta > 180.0 {
        delta - 360.0
    } else if delta < -180.0 {
        delta + 360.0
    } else {
        delta
    }
}

fn normalize_longitude_deg(mut lon: f64) -> f64 {
    while lon < -180.0 {
        lon += 360.0;
    }
    while lon >= 180.0 {
        lon -= 360.0;
    }
    lon
}

pub fn rasterize_projected_rgba_grid(
    pixels: &[Rgba],
    ny: usize,
    nx: usize,
    pixel_points: &[Option<(f32, f32)>],
    img_w: u32,
    img_h: u32,
) -> RgbaImage {
    let mut img = RgbaImage::new(img_w, img_h);

    if ny < 2 || nx < 2 || pixels.len() != ny * nx || pixel_points.len() != ny * nx {
        return img;
    }

    for j in 0..(ny - 1) {
        for i in 0..(nx - 1) {
            let idx = |jj: usize, ii: usize| jj * nx + ii;
            let p00 = pixel_points[idx(j, i)];
            let p10 = pixel_points[idx(j, i + 1)];
            let p01 = pixel_points[idx(j + 1, i)];
            let p11 = pixel_points[idx(j + 1, i + 1)];

            let (p00, p10, p01, p11) = match (p00, p10, p01, p11) {
                (Some(a), Some(b), Some(c), Some(d)) => (a, b, c, d),
                _ => continue,
            };

            let c00 = pixels[idx(j, i)];
            let c10 = pixels[idx(j, i + 1)];
            let c01 = pixels[idx(j + 1, i)];
            let c11 = pixels[idx(j + 1, i + 1)];

            let p00 = projected_pixel_to_f64(p00);
            let p10 = projected_pixel_to_f64(p10);
            let p01 = projected_pixel_to_f64(p01);
            let p11 = projected_pixel_to_f64(p11);
            rasterize_rgba_triangle(&mut img, p00, c00, p10, c10, p11, c11);
            rasterize_rgba_triangle(&mut img, p00, c00, p11, c11, p01, c01);
        }
    }

    feather_projected_raster_edges(&mut img);
    img
}

/// Rasterize projected grid coverage independent of fill values or colormap
/// alpha so frame calculations track the valid mesh footprint, not the weather
/// values currently painted into it.
pub fn rasterize_projected_coverage_mask(
    ny: usize,
    nx: usize,
    pixel_points: &[Option<(f32, f32)>],
    img_w: u32,
    img_h: u32,
) -> RgbaImage {
    let mut img = RgbaImage::new(img_w, img_h);

    if ny < 2 || nx < 2 || pixel_points.len() != ny * nx {
        return img;
    }

    for j in 0..(ny - 1) {
        for i in 0..(nx - 1) {
            let idx = |jj: usize, ii: usize| jj * nx + ii;
            let p00 = pixel_points[idx(j, i)];
            let p10 = pixel_points[idx(j, i + 1)];
            let p01 = pixel_points[idx(j + 1, i)];
            let p11 = pixel_points[idx(j + 1, i + 1)];

            let (p00, p10, p01, p11) = match (p00, p10, p01, p11) {
                (Some(a), Some(b), Some(c), Some(d)) => (a, b, c, d),
                _ => continue,
            };

            let p00 = projected_pixel_to_f64(p00);
            let p10 = projected_pixel_to_f64(p10);
            let p01 = projected_pixel_to_f64(p01);
            let p11 = projected_pixel_to_f64(p11);
            rasterize_mask_triangle(&mut img, p00, p10, p11);
            rasterize_mask_triangle(&mut img, p00, p11, p01);
        }
    }

    img
}

fn squared_distance(a: (f64, f64), b: (f64, f64)) -> f64 {
    (a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)
}

fn bilinear(v00: f64, v10: f64, v01: f64, v11: f64, fx: f64, fy: f64) -> f64 {
    if v00.is_finite() && v10.is_finite() && v01.is_finite() && v11.is_finite() {
        // A uniform cell stays exactly at its source value. Weighted
        // summation can put a constant display-floor value one ULP below
        // its mask and punch transparent stripes into an opaque field.
        if [v10, v01, v11].iter().all(|value| value.to_bits() == v00.to_bits()) {
            return v00;
        }
        let south = v00 * (1.0 - fx) + v10 * fx;
        let north = v01 * (1.0 - fx) + v11 * fx;
        south * (1.0 - fy) + north * fy
    } else {
        for value in [v00, v10, v01, v11] {
            if value.is_finite() {
                return value;
            }
        }
        f64::NAN
    }
}

fn bilinear_rgba(c00: Rgba, c10: Rgba, c01: Rgba, c11: Rgba, fx: f64, fy: f64) -> Rgba {
    if c00.a == 0 && c10.a == 0 && c01.a == 0 && c11.a == 0 {
        return Rgba::TRANSPARENT;
    }
    let south = lerp_rgba_pair(c00, c10, fx);
    let north = lerp_rgba_pair(c01, c11, fx);
    lerp_rgba_pair(south, north, fy)
}

fn rasterize_triangle(
    img: &mut RgbaImage,
    p0: (f64, f64),
    v0: f64,
    p1: (f64, f64),
    v1: f64,
    p2: (f64, f64),
    v2: f64,
    cmap: &LeveledColormap,
    sample_mode: RasterSampleMode,
) {
    // An interpolated triangle needs all three values. A nearest one needs
    // only the vertex a pixel takes, and a non-finite one maps to
    // transparent, so the finite vertices keep their share of the triangle
    // exactly as the rectilinear path keeps theirs.
    if matches!(sample_mode, RasterSampleMode::Linear)
        && (!v0.is_finite() || !v1.is_finite() || !v2.is_finite())
    {
        return;
    }
    if !v0.is_finite() && !v1.is_finite() && !v2.is_finite() {
        return;
    }

    let min_x = p0.0.min(p1.0).min(p2.0).floor().max(0.0) as i32;
    let max_x =
        p0.0.max(p1.0)
            .max(p2.0)
            .ceil()
            .min(img.width() as f64 - 1.0) as i32;
    let min_y = p0.1.min(p1.1).min(p2.1).floor().max(0.0) as i32;
    let max_y =
        p0.1.max(p1.1)
            .max(p2.1)
            .ceil()
            .min(img.height() as f64 - 1.0) as i32;

    if min_x > max_x || min_y > max_y {
        return;
    }

    let area = edge_fn(p0, p1, p2);
    if area.abs() < 1e-9 {
        return;
    }

    let inv_area = 1.0 / area;
    let constant = v0.is_finite() && v0.to_bits() == v1.to_bits() && v0.to_bits() == v2.to_bits();

    for py in min_y..=max_y {
        for px in min_x..=max_x {
            let p = (px as f64 + 0.5, py as f64 + 0.5);
            let w0 = edge_fn(p1, p2, p) * inv_area;
            let w1 = edge_fn(p2, p0, p) * inv_area;
            let w2 = edge_fn(p0, p1, p) * inv_area;

            if w0 < -1e-6 || w1 < -1e-6 || w2 < -1e-6 {
                continue;
            }

            let value = match sample_mode {
                RasterSampleMode::Linear if constant => v0,
                RasterSampleMode::Linear => v0 * w0 + v1 * w1 + v2 * w2,
                // The vertex nearest the pixel centre in pixel space. Each
                // grid quad is split along its p00-p11 diagonal, which is
                // the perpendicular bisector of p10-p01 on a square cell, so
                // the nearest of a triangle's three vertices is the nearest
                // of the quad's four and every grid point keeps a whole
                // cell around it.
                RasterSampleMode::Nearest => {
                    let d0 = squared_distance(p, p0);
                    let d1 = squared_distance(p, p1);
                    let d2 = squared_distance(p, p2);
                    if d0 <= d1 && d0 <= d2 {
                        v0
                    } else if d1 <= d2 {
                        v1
                    } else {
                        v2
                    }
                }
            };
            let color = cmap.map(value).to_image_rgba();
            if color.0[3] > 0 {
                img.put_pixel(px as u32, py as u32, color);
            }
        }
    }
}

fn rasterize_rgba_triangle(
    img: &mut RgbaImage,
    p0: (f64, f64),
    c0: Rgba,
    p1: (f64, f64),
    c1: Rgba,
    p2: (f64, f64),
    c2: Rgba,
) {
    if c0.a == 0 && c1.a == 0 && c2.a == 0 {
        return;
    }

    let min_x = p0.0.min(p1.0).min(p2.0).floor().max(0.0) as i32;
    let max_x =
        p0.0.max(p1.0)
            .max(p2.0)
            .ceil()
            .min(img.width() as f64 - 1.0) as i32;
    let min_y = p0.1.min(p1.1).min(p2.1).floor().max(0.0) as i32;
    let max_y =
        p0.1.max(p1.1)
            .max(p2.1)
            .ceil()
            .min(img.height() as f64 - 1.0) as i32;

    if min_x > max_x || min_y > max_y {
        return;
    }

    let area = edge_fn(p0, p1, p2);
    if area.abs() < 1e-9 {
        return;
    }

    let inv_area = 1.0 / area;

    for py in min_y..=max_y {
        for px in min_x..=max_x {
            let p = (px as f64 + 0.5, py as f64 + 0.5);
            let w0 = edge_fn(p1, p2, p) * inv_area;
            let w1 = edge_fn(p2, p0, p) * inv_area;
            let w2 = edge_fn(p0, p1, p) * inv_area;

            if w0 < -1e-6 || w1 < -1e-6 || w2 < -1e-6 {
                continue;
            }

            let color = weighted_rgba(c0, w0, c1, w1, c2, w2);
            if color.a > 0 {
                img.put_pixel(px as u32, py as u32, color.to_image_rgba());
            }
        }
    }
}

fn lerp_rgba_pair(left: Rgba, right: Rgba, t: f64) -> Rgba {
    let t = t.clamp(0.0, 1.0);
    Rgba::with_alpha(
        lerp_u8(left.r, right.r, t),
        lerp_u8(left.g, right.g, t),
        lerp_u8(left.b, right.b, t),
        lerp_u8(left.a, right.a, t),
    )
}

fn weighted_rgba(c0: Rgba, w0: f64, c1: Rgba, w1: f64, c2: Rgba, w2: f64) -> Rgba {
    Rgba::with_alpha(
        weighted_u8(c0.r, w0, c1.r, w1, c2.r, w2),
        weighted_u8(c0.g, w0, c1.g, w1, c2.g, w2),
        weighted_u8(c0.b, w0, c1.b, w1, c2.b, w2),
        weighted_u8(c0.a, w0, c1.a, w1, c2.a, w2),
    )
}

fn lerp_u8(left: u8, right: u8, t: f64) -> u8 {
    (left as f64 + (right as f64 - left as f64) * t)
        .round()
        .clamp(0.0, 255.0) as u8
}

fn weighted_u8(v0: u8, w0: f64, v1: u8, w1: f64, v2: u8, w2: f64) -> u8 {
    (v0 as f64 * w0 + v1 as f64 * w1 + v2 as f64 * w2)
        .round()
        .clamp(0.0, 255.0) as u8
}

fn rasterize_mask_triangle(img: &mut RgbaImage, p0: (f64, f64), p1: (f64, f64), p2: (f64, f64)) {
    let min_x = p0.0.min(p1.0).min(p2.0).floor().max(0.0) as i32;
    let max_x =
        p0.0.max(p1.0)
            .max(p2.0)
            .ceil()
            .min(img.width() as f64 - 1.0) as i32;
    let min_y = p0.1.min(p1.1).min(p2.1).floor().max(0.0) as i32;
    let max_y =
        p0.1.max(p1.1)
            .max(p2.1)
            .ceil()
            .min(img.height() as f64 - 1.0) as i32;

    if min_x > max_x || min_y > max_y {
        return;
    }

    let area = edge_fn(p0, p1, p2);
    if area.abs() < 1e-9 {
        return;
    }

    let inv_area = 1.0 / area;
    let opaque = image::Rgba([255, 255, 255, 255]);

    for py in min_y..=max_y {
        for px in min_x..=max_x {
            let p = (px as f64 + 0.5, py as f64 + 0.5);
            let w0 = edge_fn(p1, p2, p) * inv_area;
            let w1 = edge_fn(p2, p0, p) * inv_area;
            let w2 = edge_fn(p0, p1, p) * inv_area;

            if w0 < -1e-6 || w1 < -1e-6 || w2 < -1e-6 {
                continue;
            }

            img.put_pixel(px as u32, py as u32, opaque);
        }
    }
}

fn edge_fn(a: (f64, f64), b: (f64, f64), p: (f64, f64)) -> f64 {
    (p.0 - a.0) * (b.1 - a.1) - (p.1 - a.1) * (b.0 - a.0)
}

fn feather_projected_raster_edges(img: &mut RgbaImage) {
    if img.width() < 2 || img.height() < 2 {
        return;
    }

    let src = img.clone();
    let width = src.width() as i32;
    let height = src.height() as i32;

    for py in 0..height {
        for px in 0..width {
            let center = src.get_pixel(px as u32, py as u32);
            let center_alpha = center.0[3];
            let mut transparent_neighbor = false;

            for dy in -1..=1 {
                for dx in -1..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let nx = px + dx;
                    let ny = py + dy;
                    if nx < 0 || ny < 0 || nx >= width || ny >= height {
                        transparent_neighbor = true;
                        continue;
                    }
                    let neighbor = src.get_pixel(nx as u32, ny as u32);
                    if neighbor.0[3] > 0 {
                    } else {
                        transparent_neighbor = true;
                    }
                }
            }

            if center_alpha > 0 && transparent_neighbor {
                let mut softened = *center;
                softened.0[3] = softened.0[3].min(216);
                img.put_pixel(px as u32, py as u32, softened);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        RegularLatLonAxes, rasterize_grid, rasterize_projected_grid, sample_regular_latlon_grid,
        sample_regular_latlon_grid_with_mode,
    };
    use crate::color::Rgba;
    use crate::colormap::{Extend, LeveledColormap};
    use crate::request::RasterSampleMode;

    #[test]
    fn projected_raster_keeps_nan_mask_holes_transparent() {
        let data = [f64::NAN, 1.0, 1.0, 1.0];
        let pixel_points = [
            Some((0.0, 0.0)),
            Some((3.0, 0.0)),
            Some((0.0, 3.0)),
            Some((3.0, 3.0)),
        ];
        let cmap = LeveledColormap::from_palette(
            &[Rgba::new(255, 0, 0)],
            &[0.0, 2.0],
            Extend::Neither,
            None,
        );

        let image = rasterize_projected_grid(
            &data,
            2,
            2,
            &pixel_points,
            &cmap,
            RasterSampleMode::Linear,
            4,
            4,
        );

        assert!(
            image.pixels().all(|px| px.0[3] == 0),
            "mixed-validity projected cells should remain masked instead of bleeding a nearby finite value",
        );
    }

    #[test]
    fn nearest_raster_sampling_preserves_source_bins() {
        let data = [0.0, 100.0];
        let cmap = LeveledColormap::from_palette(
            &[
                Rgba::new(0, 0, 255),
                Rgba::new(0, 255, 0),
                Rgba::new(255, 0, 0),
            ],
            &[0.0, 25.0, 75.0, 100.0],
            Extend::Neither,
            None,
        );

        let linear = rasterize_grid(&data, 1, 2, &cmap, RasterSampleMode::Linear, 9, 1);
        let nearest = rasterize_grid(&data, 1, 2, &cmap, RasterSampleMode::Nearest, 9, 1);

        let linear_px = linear.get_pixel(3, 0).0;
        let nearest_px = nearest.get_pixel(3, 0).0;
        assert_ne!(linear_px, nearest_px);
        assert_eq!(nearest_px[3], 255);
    }

    #[test]
    fn nearest_regular_latlon_sampling_keeps_gridpoint_value() {
        let nx = 2;
        let ny = 2;
        let lat = vec![0.0, 0.0, 1.0, 1.0];
        let lon = vec![0.0, 1.0, 0.0, 1.0];
        let data = vec![0.0, 100.0, 200.0, 300.0];
        let axes = RegularLatLonAxes::from_grid(&lat, &lon, ny, nx).unwrap();

        let linear = sample_regular_latlon_grid_with_mode(
            &data,
            &axes,
            0.49,
            0.49,
            RasterSampleMode::Linear,
        )
        .unwrap();
        let nearest = sample_regular_latlon_grid_with_mode(
            &data,
            &axes,
            0.49,
            0.49,
            RasterSampleMode::Nearest,
        )
        .unwrap();

        assert!(linear > 100.0);
        assert_eq!(nearest, 0.0);
    }

    #[test]
    fn partial_wide_longitude_crop_is_not_periodic() {
        let nx = 321;
        let ny = 2;
        let row_lon = (0..nx).map(|i| -140.0 + i as f64).collect::<Vec<_>>();
        let lon = row_lon
            .iter()
            .copied()
            .chain(row_lon.iter().copied())
            .collect::<Vec<_>>();
        let lat = vec![-10.0; nx]
            .into_iter()
            .chain(vec![0.0; nx])
            .collect::<Vec<_>>();
        let data = vec![1.0; nx * ny];

        let axes = RegularLatLonAxes::from_grid(&lat, &lon, ny, nx).unwrap();
        assert!(!axes.periodic_lon);
        assert!(sample_regular_latlon_grid(&data, &axes, -5.0, -170.0).is_none());
    }

    #[test]
    fn full_0_360_axis_samples_negative_longitudes_periodically() {
        let nx = 360;
        let ny = 2;
        let row_lon = (0..nx).map(|i| i as f64).collect::<Vec<_>>();
        let lon = row_lon
            .iter()
            .copied()
            .chain(row_lon.iter().copied())
            .collect::<Vec<_>>();
        let lat = vec![0.0; nx]
            .into_iter()
            .chain(vec![1.0; nx])
            .collect::<Vec<_>>();
        let data = lon.clone();

        let axes = RegularLatLonAxes::from_grid(&lat, &lon, ny, nx).unwrap();
        assert!(axes.periodic_lon);
        let value = sample_regular_latlon_grid(&data, &axes, 0.5, -170.0).unwrap();
        assert!((value - 190.0).abs() < 1.0);
    }
}
#[cfg(test)]
mod uniform_sampling_tests {
    use super::*;
    use crate::colormap::Extend;

    #[test]
    fn bilinear_sampling_keeps_every_uniform_cell_value_exact() {
        for value in [10.0f64, 9.999, 95.0, 0.0, -0.0] {
            for x in 0..101 {
                for y in 0..101 {
                    let sampled = bilinear(value, value, value, value,
                                           x as f64 / 100.0, y as f64 / 100.0);
                    assert_eq!(sampled.to_bits(), value.to_bits(),
                               "uniform value {value} changed at ({x}, {y})");
                }
            }
        }
    }

    #[test]
    fn a_uniform_mask_floor_has_no_holes_in_rectilinear_or_projected_pixels() {
        let masked = LeveledColormap::from_palette(
            &[Rgba::WHITE, Rgba::new(120, 120, 120)],
            &[10.0, 50.0, 100.0], Extend::Max, Some(10.0));
        let unmasked = LeveledColormap::from_palette(
            &[Rgba::WHITE, Rgba::new(120, 120, 120)],
            &[10.0, 50.0, 100.0], Extend::Both, None);
        let data = [10.0; 4];
        let drawn = rasterize_grid(&data, 2, 2, &masked, RasterSampleMode::Linear, 257, 193);
        let control = rasterize_grid(&data, 2, 2, &unmasked, RasterSampleMode::Linear, 257, 193);
        assert_eq!(drawn, control, "the mask punched a hole into a uniform first level");
        let points = [Some((1.2, 2.3)), Some((254.7, 1.5)),
                      Some((2.5, 190.2)), Some((255.2, 191.0))];
        let drawn = rasterize_projected_grid(&data, 2, 2, &points, &masked,
                                             RasterSampleMode::Linear, 257, 193);
        let control = rasterize_projected_grid(&data, 2, 2, &points, &unmasked,
                                               RasterSampleMode::Linear, 257, 193);
        assert_eq!(drawn, control, "the projected mask punched a hole into a uniform first level");
        let below = rasterize_grid(&[9.999; 4], 2, 2, &masked,
                                   RasterSampleMode::Linear, 257, 193);
        assert!(below.pixels().all(|pixel| pixel[3] == 0),
                "the uniform-cell fix must not widen the display floor");
    }
}
