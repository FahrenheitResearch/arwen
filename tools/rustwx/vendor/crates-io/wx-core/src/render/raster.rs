//! Raster rendering: convert 2D grids to RGBA pixel buffers.
//!
//! Produces row-major RGBA pixel data suitable for PNG encoding or numpy arrays.
//! Each grid cell maps to exactly one pixel (nearest-neighbor).

use super::colormap::{self, ColorStop};

/// Render a 2D grid of values to an RGBA pixel buffer.
///
/// # Arguments
/// * `values` - Flat array of f64 values, row-major (ny rows of nx columns)
/// * `nx` - Number of columns (width)
/// * `ny` - Number of rows (height)
/// * `colormap` - Name of a built-in colormap (e.g. "temperature", "wind")
/// * `vmin` - Value mapped to the bottom of the colormap
/// * `vmax` - Value mapped to the top of the colormap
///
/// # Returns
/// RGBA pixel buffer: `Vec<u8>` of length `nx * ny * 4`.
/// Row-major order, top row first. Each pixel is [R, G, B, A].
/// NaN values produce transparent pixels (alpha = 0).
///
/// # Panics
/// Panics if `values.len() != nx * ny`.
pub fn render_raster(
    values: &[f64],
    nx: usize,
    ny: usize,
    colormap_name: &str,
    vmin: f64,
    vmax: f64,
) -> Vec<u8> {
    assert_eq!(
        values.len(),
        nx * ny,
        "values.len()={} but nx*ny={}",
        values.len(),
        nx * ny
    );

    let cmap = colormap::get_colormap(colormap_name)
        .unwrap_or(colormap::TEMPERATURE);

    render_raster_with_colormap(values, nx, ny, cmap, vmin, vmax)
}

/// Render using an explicit colormap slice (for custom colormaps).
pub fn render_raster_with_colormap(
    values: &[f64],
    nx: usize,
    ny: usize,
    cmap: &[ColorStop],
    vmin: f64,
    vmax: f64,
) -> Vec<u8> {
    assert_eq!(
        values.len(),
        nx * ny,
        "values.len()={} but nx*ny={}",
        values.len(),
        nx * ny
    );

    let range = vmax - vmin;
    let inv_range = if range.abs() < 1e-12 { 0.0 } else { 1.0 / range };

    let mut pixels = vec![0u8; nx * ny * 4];

    for i in 0..values.len() {
        let v = values[i];
        let offset = i * 4;

        if v.is_nan() {
            // Transparent pixel
            pixels[offset] = 0;
            pixels[offset + 1] = 0;
            pixels[offset + 2] = 0;
            pixels[offset + 3] = 0;
        } else {
            let t = ((v - vmin) * inv_range).clamp(0.0, 1.0);
            let (r, g, b) = colormap::interpolate_color(cmap, t);
            pixels[offset] = r;
            pixels[offset + 1] = g;
            pixels[offset + 2] = b;
            pixels[offset + 3] = 255;
        }
    }

    pixels
}

/// Render with parallel processing using rayon (for large grids).
pub fn render_raster_par(
    values: &[f64],
    nx: usize,
    ny: usize,
    colormap_name: &str,
    vmin: f64,
    vmax: f64,
) -> Vec<u8> {
    use rayon::prelude::*;

    assert_eq!(values.len(), nx * ny);

    let cmap = colormap::get_colormap(colormap_name)
        .unwrap_or(colormap::TEMPERATURE);
    let range = vmax - vmin;
    let inv_range = if range.abs() < 1e-12 { 0.0 } else { 1.0 / range };

    let pixels: Vec<u8> = values
        .par_iter()
        .flat_map(|&v| {
            if v.is_nan() {
                [0u8, 0, 0, 0]
            } else {
                let t = ((v - vmin) * inv_range).clamp(0.0, 1.0);
                let (r, g, b) = colormap::interpolate_color(cmap, t);
                [r, g, b, 255]
            }
        })
        .collect();

    pixels
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_render() {
        let values = vec![0.0, 0.5, 1.0, f64::NAN];
        let pixels = render_raster(&values, 2, 2, "temperature", 0.0, 1.0);
        assert_eq!(pixels.len(), 16); // 4 pixels * 4 bytes

        // First pixel: value 0.0 -> opaque
        assert_eq!(pixels[3], 255);

        // Last pixel: NaN -> transparent
        assert_eq!(pixels[12], 0);
        assert_eq!(pixels[13], 0);
        assert_eq!(pixels[14], 0);
        assert_eq!(pixels[15], 0);
    }

    #[test]
    fn test_unknown_colormap_falls_back() {
        let values = vec![0.5; 4];
        // Should not panic, falls back to temperature
        let pixels = render_raster(&values, 2, 2, "nonexistent_colormap", 0.0, 1.0);
        assert_eq!(pixels.len(), 16);
    }

    #[test]
    #[should_panic(expected = "values.len()")]
    fn test_size_mismatch_panics() {
        let values = vec![0.0; 5];
        render_raster(&values, 2, 2, "temperature", 0.0, 1.0);
    }
}
