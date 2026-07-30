//! Vertical cross-section rendering.
//!
//! Renders filled-contour vertical cross-sections from 2D scalar fields
//! (distance x pressure level) onto RGBA pixel buffers. Commonly used
//! for temperature, moisture, or wind cross-sections through a model domain.

use super::colormap;

/// Configuration for cross-section rendering.
#[derive(Debug, Clone)]
pub struct CrossSectionConfig {
    /// Output image width in pixels
    pub width: u32,
    /// Output image height in pixels
    pub height: u32,
    /// Top pressure level (minimum, e.g. 100 hPa)
    pub p_min: f64,
    /// Bottom pressure level (maximum, e.g. 1000 hPa)
    pub p_max: f64,
}

impl Default for CrossSectionConfig {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            p_min: 100.0,
            p_max: 1000.0,
        }
    }
}

/// Input data for a vertical cross-section.
#[derive(Debug, Clone)]
pub struct CrossSectionData {
    /// Values at each point and level: `values[point_idx][level_idx]`
    /// Shape: `[n_points][n_levels]`
    pub values: Vec<Vec<f64>>,
    /// Pressure levels in hPa, ordered from surface (highest) to top (lowest).
    /// Length: `n_levels`
    pub pressure_levels: Vec<f64>,
    /// Distances along the cross-section in km.
    /// Length: `n_points`
    pub distances: Vec<f64>,
}

/// Convert a pressure level to a vertical pixel coordinate.
///
/// Uses a log-pressure scale, which is more representative of the atmosphere
/// than linear pressure spacing. Higher pressure (surface) maps to the
/// bottom of the image (larger y), lower pressure (upper atmosphere) maps
/// to the top (y=0).
#[inline]
fn pressure_to_y(p: f64, p_min: f64, p_max: f64, height: u32) -> f64 {
    let log_p = p.ln();
    let log_min = p_min.ln();
    let log_max = p_max.ln();
    // log_max (surface, large p) -> y = height-1 (bottom)
    // log_min (top, small p) -> y = 0 (top)
    let frac = (log_p - log_min) / (log_max - log_min);
    frac * (height as f64 - 1.0)
}

/// Convert a horizontal distance to a pixel x-coordinate.
#[inline]
#[allow(dead_code)]
fn distance_to_x(d: f64, d_min: f64, d_max: f64, width: u32) -> f64 {
    let frac = (d - d_min) / (d_max - d_min);
    frac * (width as f64 - 1.0)
}

/// Bilinearly interpolate the data value at a given (distance_frac, pressure_frac).
///
/// `dist_idx_f` is the fractional index into the distance dimension.
/// `plev_idx_f` is the fractional index into the pressure level dimension.
fn bilinear_interp(data: &CrossSectionData, dist_idx_f: f64, plev_idx_f: f64) -> f64 {
    let n_pts = data.values.len();
    let n_levs = if n_pts > 0 { data.values[0].len() } else { 0 };

    if n_pts == 0 || n_levs == 0 {
        return f64::NAN;
    }

    let di = dist_idx_f.floor() as i64;
    let pi = plev_idx_f.floor() as i64;

    let di0 = di.max(0).min(n_pts as i64 - 1) as usize;
    let di1 = (di + 1).max(0).min(n_pts as i64 - 1) as usize;
    let pi0 = pi.max(0).min(n_levs as i64 - 1) as usize;
    let pi1 = (pi + 1).max(0).min(n_levs as i64 - 1) as usize;

    let fd = dist_idx_f - dist_idx_f.floor();
    let fp = plev_idx_f - plev_idx_f.floor();

    let v00 = data.values[di0][pi0];
    let v10 = data.values[di1][pi0];
    let v01 = data.values[di0][pi1];
    let v11 = data.values[di1][pi1];

    // If any corner is NaN, return NaN
    if v00.is_nan() || v10.is_nan() || v01.is_nan() || v11.is_nan() {
        return f64::NAN;
    }

    let top = v00 * (1.0 - fd) + v10 * fd;
    let bot = v01 * (1.0 - fd) + v11 * fd;
    top * (1.0 - fp) + bot * fp
}

/// Render a vertical cross-section as filled contours (colormapped raster).
///
/// # Arguments
/// * `data` - Cross-section data (values, pressure levels, distances)
/// * `config` - Rendering configuration (size, pressure bounds)
/// * `colormap_name` - Name of a built-in colormap
/// * `vmin` - Value mapped to the bottom of the colormap
/// * `vmax` - Value mapped to the top of the colormap
///
/// # Returns
/// RGBA pixel buffer of length `config.width * config.height * 4`.
///
/// The vertical axis uses a logarithmic pressure scale. The horizontal
/// axis is linear in distance. Data is bilinearly interpolated from the
/// input grid to each pixel.
pub fn render_cross_section(
    data: &CrossSectionData,
    config: &CrossSectionConfig,
    colormap_name: &str,
    vmin: f64,
    vmax: f64,
) -> Vec<u8> {
    let w = config.width;
    let h = config.height;
    let mut buf = vec![0u8; (w * h * 4) as usize];

    let cmap = colormap::get_colormap(colormap_name).unwrap_or(colormap::TEMPERATURE);
    let range = vmax - vmin;
    let inv_range = if range.abs() < 1e-12 { 0.0 } else { 1.0 / range };

    let n_pts = data.values.len();
    let n_levs = if n_pts > 0 { data.values[0].len() } else { 0 };

    if n_pts < 2 || n_levs < 2 || data.distances.len() != n_pts || data.pressure_levels.len() != n_levs {
        // Return transparent image if data is insufficient
        return buf;
    }

    let d_min = data.distances[0];
    let d_max = data.distances[n_pts - 1];

    // Precompute: for each pixel row, find the corresponding pressure
    // and fractional index into pressure_levels
    let mut row_plev_idx: Vec<f64> = Vec::with_capacity(h as usize);
    for py in 0..h {
        // Invert pressure_to_y: given py, find p
        let log_min = config.p_min.ln();
        let log_max = config.p_max.ln();
        let frac = py as f64 / (h as f64 - 1.0);
        let log_p = log_min + frac * (log_max - log_min);
        let p = log_p.exp();

        // Find fractional index into pressure_levels
        // pressure_levels might be in descending order (surface first) or ascending
        // We need to handle both. Find where p falls.
        let mut plev_frac = f64::NAN;
        for k in 0..n_levs - 1 {
            let p0 = data.pressure_levels[k];
            let p1 = data.pressure_levels[k + 1];
            let (lo, hi) = if p0 < p1 { (p0, p1) } else { (p1, p0) };
            if p >= lo && p <= hi {
                let f = (p - p0) / (p1 - p0);
                plev_frac = k as f64 + f;
                break;
            }
        }
        // Clamp to edges if outside range
        if plev_frac.is_nan() {
            // Check if above or below
            let p_top = data.pressure_levels[0].min(data.pressure_levels[n_levs - 1]);
            let p_bot = data.pressure_levels[0].max(data.pressure_levels[n_levs - 1]);
            if p <= p_top {
                plev_frac = if data.pressure_levels[0] < data.pressure_levels[n_levs - 1] {
                    0.0
                } else {
                    (n_levs - 1) as f64
                };
            } else if p >= p_bot {
                plev_frac = if data.pressure_levels[0] > data.pressure_levels[n_levs - 1] {
                    0.0
                } else {
                    (n_levs - 1) as f64
                };
            }
        }
        row_plev_idx.push(plev_frac);
    }

    for py in 0..h {
        let plev_f = row_plev_idx[py as usize];
        if plev_f.is_nan() {
            continue; // outside data range
        }

        for px in 0..w {
            // Map pixel x to distance, then to fractional index
            let d = d_min + (px as f64 / (w as f64 - 1.0)) * (d_max - d_min);
            // Find fractional index in distances
            let mut dist_f = 0.0f64;
            let mut found = false;
            for k in 0..n_pts - 1 {
                let d0 = data.distances[k];
                let d1 = data.distances[k + 1];
                if (d >= d0 && d <= d1) || (d >= d1 && d <= d0) {
                    let f = if (d1 - d0).abs() < 1e-12 { 0.0 } else { (d - d0) / (d1 - d0) };
                    dist_f = k as f64 + f;
                    found = true;
                    break;
                }
            }
            if !found {
                if d <= data.distances[0] {
                    dist_f = 0.0;
                } else {
                    dist_f = (n_pts - 1) as f64;
                }
            }

            let val = bilinear_interp(data, dist_f, plev_f);
            let offset = (py * w + px) as usize * 4;

            if val.is_nan() {
                buf[offset] = 0;
                buf[offset + 1] = 0;
                buf[offset + 2] = 0;
                buf[offset + 3] = 0;
            } else {
                let t = ((val - vmin) * inv_range).clamp(0.0, 1.0);
                let (r, g, b) = colormap::interpolate_color(cmap, t);
                buf[offset] = r;
                buf[offset + 1] = g;
                buf[offset + 2] = b;
                buf[offset + 3] = 255;
            }
        }
    }

    // Draw axes: left and bottom borders
    draw_axis_line(&mut buf, w, h, 0, 0, 0, h as i32 - 1); // left
    draw_axis_line_h(&mut buf, w, h, 0, w as i32 - 1, h as i32 - 1); // bottom

    // Pressure tick marks on the left axis
    let standard_levels = [1000.0, 925.0, 850.0, 700.0, 500.0, 300.0, 200.0, 100.0];
    for &plev in &standard_levels {
        if plev >= config.p_min && plev <= config.p_max {
            let y = pressure_to_y(plev, config.p_min, config.p_max, h).round() as i32;
            // Small tick
            for tx in 0..5 {
                set_pixel_black(&mut buf, w, h, tx, y);
            }
        }
    }

    buf
}

/// Set a pixel to black.
#[inline]
fn set_pixel_black(buf: &mut [u8], w: u32, h: u32, x: i32, y: i32) {
    if x >= 0 && y >= 0 && (x as u32) < w && (y as u32) < h {
        let idx = ((y as u32) * w + x as u32) as usize * 4;
        buf[idx] = 0;
        buf[idx + 1] = 0;
        buf[idx + 2] = 0;
        buf[idx + 3] = 255;
    }
}

/// Draw a vertical axis line.
fn draw_axis_line(buf: &mut [u8], w: u32, h: u32, x: i32, y0: i32, _x2: i32, y1: i32) {
    for y in y0..=y1 {
        set_pixel_black(buf, w, h, x, y);
    }
}

/// Draw a horizontal axis line.
fn draw_axis_line_h(buf: &mut [u8], w: u32, h: u32, x0: i32, x1: i32, y: i32) {
    for x in x0..=x1 {
        set_pixel_black(buf, w, h, x, y);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> CrossSectionData {
        // 5 points along the cross-section, 4 pressure levels
        let pressure_levels = vec![1000.0, 850.0, 500.0, 200.0];
        let distances = vec![0.0, 100.0, 200.0, 300.0, 400.0];
        let mut values = Vec::new();
        for i in 0..5 {
            let mut col = Vec::new();
            for j in 0..4 {
                // Temperature decreasing with height, varying along distance
                col.push(300.0 - j as f64 * 20.0 + i as f64 * 2.0);
            }
            values.push(col);
        }
        CrossSectionData {
            values,
            pressure_levels,
            distances,
        }
    }

    #[test]
    fn test_render_cross_section_basic() {
        let data = make_test_data();
        let config = CrossSectionConfig {
            width: 200,
            height: 150,
            p_min: 200.0,
            p_max: 1000.0,
        };

        let pixels = render_cross_section(&data, &config, "temperature", 220.0, 310.0);
        assert_eq!(pixels.len(), 200 * 150 * 4);

        // Check that some pixels are opaque (data was rendered)
        let opaque = pixels.chunks(4).filter(|p| p[3] == 255).count();
        assert!(opaque > 0, "Should have rendered some data pixels");
    }

    #[test]
    fn test_pressure_to_y() {
        let y_surface = pressure_to_y(1000.0, 100.0, 1000.0, 600);
        let y_top = pressure_to_y(100.0, 100.0, 1000.0, 600);
        assert!(y_surface > y_top, "Surface should be below (larger y) than top");
        assert!((y_top - 0.0).abs() < 1.0, "Top should be near y=0");
        assert!((y_surface - 599.0).abs() < 1.0, "Surface should be near y=height-1");
    }

    #[test]
    fn test_empty_data() {
        let data = CrossSectionData {
            values: vec![],
            pressure_levels: vec![],
            distances: vec![],
        };
        let config = CrossSectionConfig::default();
        let pixels = render_cross_section(&data, &config, "temperature", 0.0, 1.0);
        // Should return a transparent buffer without panicking
        assert_eq!(pixels.len(), (config.width * config.height * 4) as usize);
    }

    #[test]
    fn test_unknown_colormap_fallback() {
        let data = make_test_data();
        let config = CrossSectionConfig {
            width: 100,
            height: 100,
            p_min: 200.0,
            p_max: 1000.0,
        };
        // Should not panic, falls back to temperature
        let pixels = render_cross_section(&data, &config, "nonexistent", 220.0, 310.0);
        assert_eq!(pixels.len(), 100 * 100 * 4);
    }
}
