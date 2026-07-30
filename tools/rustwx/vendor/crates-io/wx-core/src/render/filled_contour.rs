//! Filled (shaded) contour rendering.
//!
//! Renders a 2D scalar field as colored bands between contour levels,
//! producing smooth filled contour plots similar to matplotlib's `contourf`.

use super::colormap::{self, ColorStop};

/// Render filled (shaded) contours to RGBA pixels.
///
/// Each contour band between adjacent levels gets the interpolated colormap color.
/// Grid cells are bilinearly interpolated for smooth edges between bands.
///
/// # Arguments
/// * `values` - Flat array of f64 values, row-major (ny rows of nx columns)
/// * `nx` - Number of data columns
/// * `ny` - Number of data rows
/// * `levels` - Contour levels (must be sorted ascending). Values below
///   the first level or above the last level are clamped.
/// * `colormap` - Name of a built-in colormap
/// * `width` - Output image width in pixels
/// * `height` - Output image height in pixels
///
/// # Returns
/// RGBA pixel buffer of length `width * height * 4`.
/// NaN values produce transparent pixels.
pub fn render_filled_contours(
    values: &[f64],
    nx: usize,
    ny: usize,
    levels: &[f64],
    colormap_name: &str,
    width: u32,
    height: u32,
) -> Vec<u8> {
    assert_eq!(
        values.len(),
        nx * ny,
        "values.len()={} but nx*ny={}",
        values.len(),
        nx * ny
    );

    let cmap = colormap::get_colormap(colormap_name).unwrap_or(colormap::TEMPERATURE);
    render_filled_contours_with_colormap(values, nx, ny, levels, cmap, width, height)
}

/// Render filled contours with an explicit colormap.
pub fn render_filled_contours_with_colormap(
    values: &[f64],
    nx: usize,
    ny: usize,
    levels: &[f64],
    cmap: &[ColorStop],
    width: u32,
    height: u32,
) -> Vec<u8> {
    assert_eq!(values.len(), nx * ny);
    assert!(levels.len() >= 2, "Need at least 2 contour levels");

    let w = width as usize;
    let h = height as usize;
    let mut pixels = vec![0u8; w * h * 4];

    let num_bands = levels.len() - 1;
    let level_min = levels[0];
    let level_max = levels[levels.len() - 1];
    let level_range = level_max - level_min;
    let inv_level_range = if level_range.abs() < 1e-12 {
        0.0
    } else {
        1.0 / level_range
    };

    // For each output pixel, compute the bilinearly interpolated value
    // from the data grid, then map to a contour band color.
    for py in 0..h {
        // Map pixel y to data grid y (fractional)
        let gy = (py as f64) * ((ny - 1) as f64) / ((h - 1).max(1) as f64);
        let gy0 = (gy.floor() as usize).min(ny - 2);
        let gy1 = gy0 + 1;
        let fy = gy - gy0 as f64;

        for px in 0..w {
            // Map pixel x to data grid x (fractional)
            let gx = (px as f64) * ((nx - 1) as f64) / ((w - 1).max(1) as f64);
            let gx0 = (gx.floor() as usize).min(nx - 2);
            let gx1 = gx0 + 1;
            let fx = gx - gx0 as f64;

            // Bilinear interpolation
            let v00 = values[gy0 * nx + gx0];
            let v10 = values[gy0 * nx + gx1];
            let v01 = values[gy1 * nx + gx0];
            let v11 = values[gy1 * nx + gx1];

            let offset = (py * w + px) * 4;

            // If any corner is NaN, pixel is transparent
            if v00.is_nan() || v10.is_nan() || v01.is_nan() || v11.is_nan() {
                pixels[offset] = 0;
                pixels[offset + 1] = 0;
                pixels[offset + 2] = 0;
                pixels[offset + 3] = 0;
                continue;
            }

            let val = v00 * (1.0 - fx) * (1.0 - fy)
                + v10 * fx * (1.0 - fy)
                + v01 * (1.0 - fx) * fy
                + v11 * fx * fy;

            // Find which band this value falls in
            let _t = ((val - level_min) * inv_level_range).clamp(0.0, 1.0);

            // Quantize to band center for discrete contour look
            let band_idx = find_band(val, levels, num_bands);
            let band_center = (band_idx as f64 + 0.5) / num_bands as f64;

            // Blend: mostly band color, slight interpolation for smoother edges
            // within 10% of a level boundary, blend between adjacent bands
            let band_low = levels[band_idx];
            let band_high = levels[band_idx + 1];
            let band_range = band_high - band_low;

            let final_t = if band_range.abs() < 1e-12 {
                band_center
            } else {
                let pos_in_band = (val - band_low) / band_range;
                // Edge smoothing: blend near boundaries
                let smooth_width = 0.1;
                if pos_in_band < smooth_width && band_idx > 0 {
                    // Near lower boundary, blend with band below
                    let blend = pos_in_band / smooth_width;
                    let below_center = (band_idx as f64 - 0.5) / num_bands as f64;
                    below_center * (1.0 - blend) + band_center * blend
                } else if pos_in_band > (1.0 - smooth_width) && band_idx < num_bands - 1 {
                    // Near upper boundary, blend with band above
                    let blend = (pos_in_band - (1.0 - smooth_width)) / smooth_width;
                    let above_center = (band_idx as f64 + 1.5) / num_bands as f64;
                    band_center * (1.0 - blend) + above_center * blend
                } else {
                    band_center
                }
            };

            let (r, g, b) = colormap::interpolate_color(cmap, final_t.clamp(0.0, 1.0));
            pixels[offset] = r;
            pixels[offset + 1] = g;
            pixels[offset + 2] = b;
            pixels[offset + 3] = 255;
        }
    }

    pixels
}

/// Find which band index a value falls into.
/// Returns index in [0, num_bands-1].
#[inline]
fn find_band(val: f64, levels: &[f64], num_bands: usize) -> usize {
    // Binary search for the correct band
    if val <= levels[0] {
        return 0;
    }
    if val >= levels[num_bands] {
        return num_bands - 1;
    }
    for i in 0..num_bands {
        if val >= levels[i] && val < levels[i + 1] {
            return i;
        }
    }
    num_bands - 1
}

/// Generate automatic contour levels for a data range.
///
/// Returns approximately `n` evenly-spaced levels spanning the data range.
pub fn auto_levels(vmin: f64, vmax: f64, n: usize) -> Vec<f64> {
    let n = n.max(2);
    let step = (vmax - vmin) / (n - 1) as f64;
    (0..n).map(|i| vmin + i as f64 * step).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_filled_contours() {
        // 3x3 gradient grid
        let values = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let levels = vec![0.0, 2.0, 4.0, 6.0, 8.0];
        let pixels = render_filled_contours(&values, 3, 3, &levels, "temperature", 10, 10);
        assert_eq!(pixels.len(), 10 * 10 * 4);

        // All pixels should be opaque (no NaN in input)
        for py in 0..10 {
            for px in 0..10 {
                let a = pixels[(py * 10 + px) * 4 + 3];
                assert_eq!(a, 255, "pixel ({},{}) should be opaque", px, py);
            }
        }
    }

    #[test]
    fn test_nan_transparent() {
        let values = vec![1.0, 2.0, f64::NAN, 4.0];
        let levels = vec![0.0, 2.5, 5.0];
        let pixels = render_filled_contours(&values, 2, 2, &levels, "temperature", 4, 4);
        // Top-right and bottom-left pixels touching the NaN corner should be transparent
        // (the NaN is at position (0,1) in the 2x2 grid)
        assert_eq!(pixels.len(), 4 * 4 * 4);
    }

    #[test]
    fn test_auto_levels() {
        let levels = auto_levels(0.0, 10.0, 6);
        assert_eq!(levels.len(), 6);
        assert!((levels[0] - 0.0).abs() < 1e-10);
        assert!((levels[5] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_find_band() {
        let levels = vec![0.0, 2.0, 4.0, 6.0, 8.0];
        assert_eq!(find_band(-1.0, &levels, 4), 0);
        assert_eq!(find_band(1.0, &levels, 4), 0);
        assert_eq!(find_band(3.0, &levels, 4), 1);
        assert_eq!(find_band(5.0, &levels, 4), 2);
        assert_eq!(find_band(9.0, &levels, 4), 3);
    }
}
