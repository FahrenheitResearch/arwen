//! Map overlay rendering: contour lines, wind barbs, and streamlines
//! drawn on top of existing RGBA pixel buffers.

use super::contour::ContourLine;

/// Draw contour lines on top of a raster image.
///
/// # Arguments
/// * `pixels` - Existing RGBA buffer to draw on (length = width * height * 4)
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
/// * `contours` - Contour lines from `contour_lines()`
/// * `nx` - Data grid width
/// * `ny` - Data grid height
/// * `color` - RGB color for contour lines
/// * `line_width` - Line width in pixels
pub fn overlay_contours(
    pixels: &mut [u8],
    width: u32,
    height: u32,
    contours: &[ContourLine],
    nx: usize,
    ny: usize,
    color: (u8, u8, u8),
    line_width: u32,
) {
    let w = width as usize;
    let h = height as usize;
    assert_eq!(pixels.len(), w * h * 4);

    let scale_x = (w - 1) as f64 / (nx - 1).max(1) as f64;
    let scale_y = (h - 1) as f64 / (ny - 1).max(1) as f64;

    for contour in contours {
        for &(x1, y1, x2, y2) in &contour.segments {
            // Map grid coords to pixel coords
            let px1 = x1 * scale_x;
            let py1 = y1 * scale_y;
            let px2 = x2 * scale_x;
            let py2 = y2 * scale_y;

            draw_thick_line(pixels, w, h, px1, py1, px2, py2, color, line_width);
        }
    }
}

/// Draw wind barbs on a raster image at regular grid intervals.
///
/// Standard meteorological wind barbs:
/// - Short barb = 5 kt, long barb = 10 kt, pennant (flag) = 50 kt
///
/// # Arguments
/// * `pixels` - RGBA buffer to draw on
/// * `width`, `height` - Image dimensions
/// * `u`, `v` - Wind component arrays (row-major, same grid as image)
/// * `nx`, `ny` - Data grid dimensions
/// * `skip` - Plot every Nth grid point
/// * `color` - RGB color for barbs
/// * `barb_length` - Length of the barb staff in pixels
pub fn overlay_wind_barbs(
    pixels: &mut [u8],
    width: u32,
    height: u32,
    u: &[f64],
    v: &[f64],
    nx: usize,
    ny: usize,
    skip: usize,
    color: (u8, u8, u8),
    barb_length: u32,
) {
    assert_eq!(u.len(), nx * ny);
    assert_eq!(v.len(), nx * ny);
    let w = width as usize;
    let h = height as usize;
    assert_eq!(pixels.len(), w * h * 4);

    let scale_x = (w - 1) as f64 / (nx - 1).max(1) as f64;
    let scale_y = (h - 1) as f64 / (ny - 1).max(1) as f64;
    let skip = skip.max(1);

    for gy in (0..ny).step_by(skip) {
        for gx in (0..nx).step_by(skip) {
            let idx = gy * nx + gx;
            let uu = u[idx];
            let vv = v[idx];

            if uu.is_nan() || vv.is_nan() {
                continue;
            }

            let speed = (uu * uu + vv * vv).sqrt();
            if speed < 0.5 {
                // Calm wind: draw a circle
                let cx = (gx as f64 * scale_x) as i32;
                let cy = (gy as f64 * scale_y) as i32;
                draw_circle(pixels, w, h, cx, cy, 3, color);
                continue;
            }

            // Wind direction: barb points into the wind (from direction)
            let angle = vv.atan2(uu); // meteorological: direction wind is FROM
            let cx = gx as f64 * scale_x;
            let cy = gy as f64 * scale_y;

            draw_wind_barb(pixels, w, h, cx, cy, angle, speed, barb_length, color);
        }
    }
}

/// Draw streamlines on a raster image.
///
/// # Arguments
/// * `pixels` - RGBA buffer to draw on
/// * `width`, `height` - Image dimensions
/// * `u`, `v` - Wind component arrays (row-major)
/// * `nx`, `ny` - Data grid dimensions
/// * `density` - Streamline density (1.0 = normal, higher = more lines)
/// * `color` - RGB color for streamlines
pub fn overlay_streamlines(
    pixels: &mut [u8],
    width: u32,
    height: u32,
    u: &[f64],
    v: &[f64],
    nx: usize,
    ny: usize,
    density: f64,
    color: (u8, u8, u8),
) {
    assert_eq!(u.len(), nx * ny);
    assert_eq!(v.len(), nx * ny);
    let w = width as usize;
    let h = height as usize;
    assert_eq!(pixels.len(), w * h * 4);

    let density = density.max(0.1);
    // Grid spacing for seed points
    let spacing = ((nx.min(ny) as f64) / (5.0 * density)).max(2.0) as usize;

    let scale_x = (w - 1) as f64 / (nx - 1).max(1) as f64;
    let scale_y = (h - 1) as f64 / (ny - 1).max(1) as f64;

    // Track which pixel cells have been visited to avoid overlapping streamlines
    let cell_nx = (w / 4).max(1);
    let cell_ny = (h / 4).max(1);
    let mut visited = vec![false; cell_nx * cell_ny];

    // Seed streamlines at regular grid intervals
    for gy in (spacing / 2..ny).step_by(spacing) {
        for gx in (spacing / 2..nx).step_by(spacing) {
            let start_x = gx as f64;
            let start_y = gy as f64;

            // Integrate forward
            trace_streamline(
                pixels, w, h,
                u, v, nx, ny,
                start_x, start_y,
                scale_x, scale_y,
                color, 1.0,
                &mut visited, cell_nx, cell_ny,
            );

            // Integrate backward
            trace_streamline(
                pixels, w, h,
                u, v, nx, ny,
                start_x, start_y,
                scale_x, scale_y,
                color, -1.0,
                &mut visited, cell_nx, cell_ny,
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────
// Drawing primitives
// ─────────────────────────────────────────────────────────────

/// Draw a thick line using Bresenham with offset passes.
fn draw_thick_line(
    pixels: &mut [u8],
    w: usize,
    h: usize,
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    color: (u8, u8, u8),
    thickness: u32,
) {
    let half = thickness as i32 / 2;
    for dy in -half..=half {
        for dx in -half..=half {
            // Approximate circular kernel
            if dx * dx + dy * dy <= half * half + half {
                draw_line(
                    pixels, w, h,
                    (x1 as i32 + dx, y1 as i32 + dy),
                    (x2 as i32 + dx, y2 as i32 + dy),
                    color,
                );
            }
        }
    }
}

/// Bresenham line drawing.
fn draw_line(
    pixels: &mut [u8],
    w: usize,
    h: usize,
    p0: (i32, i32),
    p1: (i32, i32),
    color: (u8, u8, u8),
) {
    let (mut x0, mut y0) = p0;
    let (x1, y1) = p1;

    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;

    loop {
        if x0 >= 0 && x0 < w as i32 && y0 >= 0 && y0 < h as i32 {
            let offset = (y0 as usize * w + x0 as usize) * 4;
            pixels[offset] = color.0;
            pixels[offset + 1] = color.1;
            pixels[offset + 2] = color.2;
            pixels[offset + 3] = 255;
        }

        if x0 == x1 && y0 == y1 {
            break;
        }

        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            err += dx;
            y0 += sy;
        }
    }
}

/// Draw a small circle (for calm wind).
fn draw_circle(
    pixels: &mut [u8],
    w: usize,
    h: usize,
    cx: i32,
    cy: i32,
    r: i32,
    color: (u8, u8, u8),
) {
    for dy in -r..=r {
        for dx in -r..=r {
            if dx * dx + dy * dy <= r * r && dx * dx + dy * dy >= (r - 1) * (r - 1) {
                let px = cx + dx;
                let py = cy + dy;
                if px >= 0 && px < w as i32 && py >= 0 && py < h as i32 {
                    let offset = (py as usize * w + px as usize) * 4;
                    pixels[offset] = color.0;
                    pixels[offset + 1] = color.1;
                    pixels[offset + 2] = color.2;
                    pixels[offset + 3] = 255;
                }
            }
        }
    }
}

/// Draw a single wind barb at the given position.
fn draw_wind_barb(
    pixels: &mut [u8],
    w: usize,
    h: usize,
    cx: f64,
    cy: f64,
    angle: f64,
    speed: f64,
    barb_length: u32,
    color: (u8, u8, u8),
) {
    let bl = barb_length as f64;
    let cos_a = angle.cos();
    let sin_a = angle.sin();

    // Staff: from center pointing into wind direction
    let staff_x = cx - bl * cos_a;
    let staff_y = cy - bl * sin_a;
    draw_line(
        pixels, w, h,
        (cx as i32, cy as i32),
        (staff_x as i32, staff_y as i32),
        color,
    );

    // Convert speed to knots (assume m/s input, 1 m/s = 1.94384 kt)
    let knots = speed * 1.94384;

    // Draw barb elements from tip of staff inward
    let barb_tick_len = bl * 0.4;
    let barb_spacing = bl * 0.12;
    // Perpendicular direction (to the left of the staff looking from center to tip)
    let perp_x = sin_a;
    let perp_y = -cos_a;

    let mut remaining = knots;
    let mut pos = 0.0; // distance from tip along staff

    // Pennants (50 kt)
    while remaining >= 47.5 {
        let base_x = staff_x + pos * cos_a;
        let base_y = staff_y + pos * sin_a;
        let tip_x = base_x + barb_tick_len * perp_x;
        let tip_y = base_y + barb_tick_len * perp_y;
        let next_x = staff_x + (pos + barb_spacing * 2.0) * cos_a;
        let next_y = staff_y + (pos + barb_spacing * 2.0) * sin_a;

        // Draw filled triangle (pennant)
        draw_line(pixels, w, h, (base_x as i32, base_y as i32), (tip_x as i32, tip_y as i32), color);
        draw_line(pixels, w, h, (tip_x as i32, tip_y as i32), (next_x as i32, next_y as i32), color);
        draw_line(pixels, w, h, (next_x as i32, next_y as i32), (base_x as i32, base_y as i32), color);

        remaining -= 50.0;
        pos += barb_spacing * 2.5;
    }

    // Long barbs (10 kt)
    while remaining >= 7.5 {
        let base_x = staff_x + pos * cos_a;
        let base_y = staff_y + pos * sin_a;
        let tip_x = base_x + barb_tick_len * perp_x;
        let tip_y = base_y + barb_tick_len * perp_y;

        draw_line(pixels, w, h, (base_x as i32, base_y as i32), (tip_x as i32, tip_y as i32), color);

        remaining -= 10.0;
        pos += barb_spacing;
    }

    // Short barb (5 kt)
    if remaining >= 2.5 {
        let base_x = staff_x + pos * cos_a;
        let base_y = staff_y + pos * sin_a;
        let tip_x = base_x + barb_tick_len * 0.5 * perp_x;
        let tip_y = base_y + barb_tick_len * 0.5 * perp_y;

        draw_line(pixels, w, h, (base_x as i32, base_y as i32), (tip_x as i32, tip_y as i32), color);
    }
}

/// Bilinear interpolation of a grid value at fractional coordinates.
#[inline]
fn interp_grid(data: &[f64], nx: usize, ny: usize, x: f64, y: f64) -> f64 {
    if x < 0.0 || y < 0.0 || x >= (nx - 1) as f64 || y >= (ny - 1) as f64 {
        return f64::NAN;
    }
    let ix = x.floor() as usize;
    let iy = y.floor() as usize;
    let fx = x - ix as f64;
    let fy = y - iy as f64;

    let v00 = data[iy * nx + ix];
    let v10 = data[iy * nx + ix + 1];
    let v01 = data[(iy + 1) * nx + ix];
    let v11 = data[(iy + 1) * nx + ix + 1];

    if v00.is_nan() || v10.is_nan() || v01.is_nan() || v11.is_nan() {
        return f64::NAN;
    }

    v00 * (1.0 - fx) * (1.0 - fy)
        + v10 * fx * (1.0 - fy)
        + v01 * (1.0 - fx) * fy
        + v11 * fx * fy
}

/// Trace a single streamline from a seed point using RK2 integration.
fn trace_streamline(
    pixels: &mut [u8],
    w: usize,
    h: usize,
    u: &[f64],
    v: &[f64],
    nx: usize,
    ny: usize,
    start_x: f64,
    start_y: f64,
    scale_x: f64,
    scale_y: f64,
    color: (u8, u8, u8),
    direction: f64,
    visited: &mut [bool],
    cell_nx: usize,
    cell_ny: usize,
) {
    let max_steps = (nx + ny) * 2;
    let step_size = 0.3;

    let mut x = start_x;
    let mut y = start_y;

    for _ in 0..max_steps {
        // Check bounds
        if x < 0.5 || y < 0.5 || x >= (nx - 1) as f64 - 0.5 || y >= (ny - 1) as f64 - 0.5 {
            break;
        }

        let uu = interp_grid(u, nx, ny, x, y);
        let vv = interp_grid(v, nx, ny, x, y);
        if uu.is_nan() || vv.is_nan() {
            break;
        }

        let speed = (uu * uu + vv * vv).sqrt();
        if speed < 1e-6 {
            break;
        }

        // Check if this pixel cell is already visited
        let px = (x * scale_x) as usize;
        let py = (y * scale_y) as usize;
        let cell_x = (px * cell_nx / w).min(cell_nx - 1);
        let cell_y = (py * cell_ny / h).min(cell_ny - 1);
        let cell_idx = cell_y * cell_nx + cell_x;

        if visited[cell_idx] {
            break;
        }
        visited[cell_idx] = true;

        // RK2 integration
        let dx = direction * uu / speed * step_size;
        let dy = direction * vv / speed * step_size;

        let mid_x = x + dx * 0.5;
        let mid_y = y + dy * 0.5;

        let uu2 = interp_grid(u, nx, ny, mid_x, mid_y);
        let vv2 = interp_grid(v, nx, ny, mid_x, mid_y);
        if uu2.is_nan() || vv2.is_nan() {
            break;
        }
        let speed2 = (uu2 * uu2 + vv2 * vv2).sqrt();
        if speed2 < 1e-6 {
            break;
        }

        let new_x = x + direction * uu2 / speed2 * step_size;
        let new_y = y + direction * vv2 / speed2 * step_size;

        // Draw the line segment
        let px1 = (x * scale_x) as i32;
        let py1 = (y * scale_y) as i32;
        let px2 = (new_x * scale_x) as i32;
        let py2 = (new_y * scale_y) as i32;

        draw_line(pixels, w, h, (px1, py1), (px2, py2), color);

        x = new_x;
        y = new_y;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_overlay_contours() {
        let mut pixels = vec![0u8; 10 * 10 * 4];
        let values = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let contours = super::super::contour::contour_lines(&values, 3, 3, &[4.0]);
        overlay_contours(&mut pixels, 10, 10, &contours, 3, 3, (255, 0, 0), 1);
        // Should have drawn some red pixels
        let any_red = pixels.chunks(4).any(|c| c[0] == 255 && c[3] == 255);
        assert!(any_red, "Should have drawn red contour pixels");
    }

    #[test]
    fn test_overlay_wind_barbs() {
        let mut pixels = vec![128u8; 20 * 20 * 4];
        // 5x5 grid with constant westerly wind
        let nx = 5;
        let ny = 5;
        let u = vec![10.0; nx * ny];
        let v = vec![0.0; nx * ny];
        overlay_wind_barbs(&mut pixels, 20, 20, &u, &v, nx, ny, 2, (0, 0, 0), 8);
        // Should not panic and should modify some pixels
    }

    #[test]
    fn test_overlay_streamlines() {
        let mut pixels = vec![255u8; 20 * 20 * 4];
        let nx = 5;
        let ny = 5;
        let u = vec![5.0; nx * ny];
        let v = vec![3.0; nx * ny];
        overlay_streamlines(&mut pixels, 20, 20, &u, &v, nx, ny, 1.0, (0, 0, 0));
        // Should not panic
    }

    #[test]
    fn test_draw_line() {
        let mut pixels = vec![0u8; 10 * 10 * 4];
        draw_line(&mut pixels, 10, 10, (0, 0), (9, 9), (255, 255, 255));
        // Diagonal line should have some white pixels
        let white_count = pixels.chunks(4).filter(|c| c[0] == 255 && c[3] == 255).count();
        assert!(white_count >= 10, "Diagonal line should have at least 10 pixels");
    }
}
