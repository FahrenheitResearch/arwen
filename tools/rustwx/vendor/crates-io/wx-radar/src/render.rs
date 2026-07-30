//! CPU PPI (Plan Position Indicator) radar renderer.
//! Inverse-mapped with bilinear interpolation between adjacent radials and gates.

use rayon::prelude::*;
use crate::level2::{Level2Sweep, RadialData};
use crate::products::RadarProduct;
use crate::color_table::ColorTable;

/// Rendered PPI output.
pub struct RenderedPPI {
    pub pixels: Vec<u8>,  // RGBA
    pub size: u32,
    pub range_km: f64,
}

/// Render a sweep for a given product into a square RGBA image.
pub fn render_ppi(
    sweep: &Level2Sweep,
    product: RadarProduct,
    image_size: u32,
) -> Option<RenderedPPI> {
    let color_table = ColorTable::for_product(product);
    render_ppi_with_table(sweep, product, image_size, &color_table)
}

pub fn render_ppi_with_table(
    sweep: &Level2Sweep,
    product: RadarProduct,
    image_size: u32,
    color_table: &ColorTable,
) -> Option<RenderedPPI> {
    // Find max range from the data
    let max_range_m = sweep.radials.iter()
        .filter_map(|r| {
            r.moments.iter()
                .filter(|m| m.product == product)
                .map(|m| m.first_gate_range as f64 + m.gate_count as f64 * m.gate_size as f64)
                .next()
        })
        .fold(0.0f64, f64::max);

    if max_range_m <= 0.0 { return None; }

    let range_km = max_range_m / 1000.0;
    let size = image_size as usize;
    let center = size as f64 / 2.0;
    let scale = center / max_range_m;

    // Build sorted azimuth lookup
    let mut radial_indices: Vec<(f32, usize)> = sweep.radials.iter()
        .enumerate()
        .map(|(i, r)| (r.azimuth, i))
        .collect();
    radial_indices.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let azimuths: Vec<f32> = radial_indices.iter().map(|(az, _)| *az).collect();
    let indices: Vec<usize> = radial_indices.iter().map(|(_, i)| *i).collect();
    let n_az = azimuths.len();
    if n_az < 2 { return None; }

    // Inverse rendering with bilinear interpolation, parallelized per row
    let row_chunks: Vec<Vec<u8>> = (0..size).into_par_iter().map(|py| {
        let mut row = vec![0u8; size * 4];
        let dy = center - py as f64;

        for px in 0..size {
            let dx = px as f64 - center;
            let range_m = (dx * dx + dy * dy).sqrt() / scale;
            if range_m <= 0.0 || range_m > max_range_m { continue; }

            // Azimuth: 0° = north, clockwise
            let mut az_deg = (dx.atan2(dy)).to_degrees();
            if az_deg < 0.0 { az_deg += 360.0; }
            let az_f32 = az_deg as f32;

            // Binary search for bracketing radials
            let insert_pos = match azimuths.binary_search_by(|a| {
                a.partial_cmp(&az_f32).unwrap_or(std::cmp::Ordering::Equal)
            }) {
                Ok(i) => i,
                Err(i) => i,
            };

            let lo_sorted = if insert_pos == 0 { n_az - 1 } else { insert_pos - 1 };
            let hi_sorted = if insert_pos >= n_az { 0 } else { insert_pos };

            let lo_idx = indices[lo_sorted];
            let hi_idx = indices[hi_sorted];
            let az_lo = azimuths[lo_sorted];
            let az_hi = azimuths[hi_sorted];

            let mut az_span = az_hi - az_lo;
            if az_span < 0.0 { az_span += 360.0; }
            let mut az_off = az_f32 - az_lo;
            if az_off < 0.0 { az_off += 360.0; }

            if az_span > 5.0 { continue; }
            let az_t = if az_span > 0.001 { (az_off / az_span) as f64 } else { 0.0 };

            let val_lo = sample_radial_interp(&sweep.radials[lo_idx], product, range_m);
            let val_hi = sample_radial_interp(&sweep.radials[hi_idx], product, range_m);

            let value = match (val_lo, val_hi) {
                (Some(v0), Some(v1)) => v0 + (v1 - v0) * az_t as f32,
                (Some(v), None) | (None, Some(v)) => v,
                (None, None) => continue,
            };

            if value.is_nan() { continue; }

            let color = color_table.color_for_value(value);
            if color[3] == 0 { continue; }

            let idx = px * 4;
            row[idx] = color[0];
            row[idx + 1] = color[1];
            row[idx + 2] = color[2];
            row[idx + 3] = color[3];
        }
        row
    }).collect();

    let mut pixels = vec![0u8; size * size * 4];
    for (py, row) in row_chunks.into_iter().enumerate() {
        let start = py * size * 4;
        pixels[start..start + size * 4].copy_from_slice(&row);
    }

    Some(RenderedPPI { pixels, size: image_size, range_km })
}

/// Sample a radial at a given range with linear gate interpolation.
#[inline]
fn sample_radial_interp(radial: &RadialData, product: RadarProduct, range_m: f64) -> Option<f32> {
    let moment = radial.moments.iter().find(|m| m.product == product)?;
    let gate_offset = range_m - moment.first_gate_range as f64;
    if gate_offset < 0.0 { return None; }
    let gate_f = gate_offset / moment.gate_size as f64;
    let gate_lo = gate_f as usize;
    if gate_lo >= moment.data.len() { return None; }
    let v0 = moment.data[gate_lo];
    if v0.is_nan() { return None; }

    let gate_hi = gate_lo + 1;
    if gate_hi < moment.data.len() {
        let v1 = moment.data[gate_hi];
        if !v1.is_nan() {
            let t = (gate_f - gate_lo as f64) as f32;
            return Some(v0 + (v1 - v0) * t);
        }
    }
    Some(v0)
}
