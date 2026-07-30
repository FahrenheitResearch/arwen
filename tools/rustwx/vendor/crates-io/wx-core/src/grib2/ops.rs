//! GRIB2 file manipulation operations — merge, subset, filter, split,
//! field math, smoothing, wind rotation, and unit conversion.
//!
//! Mirrors common wgrib2 file operations in a composable Rust API.

use super::parser::{Grib2File, Grib2Message};
use super::writer::{Grib2Writer, MessageBuilder, PackingMethod};
use super::unpack::unpack_message;

// ═══════════════════════════════════════════════════════════
// File-level operations
// ═══════════════════════════════════════════════════════════

/// Merge multiple GRIB2 files into one by concatenating all messages.
pub fn merge(files: &[&Grib2File]) -> Grib2File {
    let mut messages = Vec::new();
    for file in files {
        messages.extend(file.messages.iter().cloned());
    }
    Grib2File { messages }
}

/// Extract specific messages by index (0-based).
///
/// Out-of-range indices are silently skipped.
pub fn subset(file: &Grib2File, indices: &[usize]) -> Grib2File {
    let messages = indices
        .iter()
        .filter_map(|&i| file.messages.get(i).cloned())
        .collect();
    Grib2File { messages }
}

/// Filter messages matching a predicate.
pub fn filter<F>(file: &Grib2File, predicate: F) -> Grib2File
where
    F: Fn(&Grib2Message) -> bool,
{
    let messages = file
        .messages
        .iter()
        .filter(|m| predicate(m))
        .cloned()
        .collect();
    Grib2File { messages }
}

/// Split a multi-message GRIB2 file into individual single-message files.
///
/// Returns `Vec<(filename, bytes)>` where each entry is one message
/// re-encoded as a standalone GRIB2 file. Filenames are generated as
/// `"msg_NNN.grib2"`.
pub fn split(file: &Grib2File) -> Vec<(String, Vec<u8>)> {
    let mut results = Vec::with_capacity(file.messages.len());

    for (i, msg) in file.messages.iter().enumerate() {
        let filename = format!("msg_{:03}.grib2", i);

        // Re-encode the single message using the writer
        let values = match unpack_message(msg) {
            Ok(v) => v,
            Err(_) => {
                // If we can't unpack, write an empty message
                results.push((filename, Vec::new()));
                continue;
            }
        };

        let builder = MessageBuilder::new(msg.discipline, values)
            .grid(msg.grid.clone())
            .product(msg.product.clone())
            .reference_time(msg.reference_time)
            .packing(PackingMethod::Simple { bits_per_value: 16 });

        let builder = if let Some(ref bm) = msg.bitmap {
            builder.bitmap(bm.clone())
        } else {
            builder
        };

        let writer = Grib2Writer::new().add_message(builder);
        match writer.to_bytes() {
            Ok(bytes) => results.push((filename, bytes)),
            Err(_) => results.push((filename, Vec::new())),
        }
    }

    results
}

// ═══════════════════════════════════════════════════════════
// Field math and statistics
// ═══════════════════════════════════════════════════════════

/// Compute element-wise difference between two fields (a - b).
///
/// Both arrays must have the same length. NaN values propagate.
pub fn field_diff(a: &[f64], b: &[f64]) -> Vec<f64> {
    assert_eq!(a.len(), b.len(), "field_diff: arrays must have same length");
    a.iter().zip(b.iter()).map(|(&x, &y)| x - y).collect()
}

/// Statistics for a data field.
#[derive(Debug, Clone)]
pub struct FieldStats {
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub std_dev: f64,
    pub count: usize,
    pub nan_count: usize,
}

/// Compute statistics over a field, ignoring NaN values.
pub fn field_stats(values: &[f64]) -> FieldStats {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    let mut count: usize = 0;
    let mut nan_count: usize = 0;

    for &v in values {
        if v.is_nan() {
            nan_count += 1;
            continue;
        }
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
        sum += v;
        sum_sq += v * v;
        count += 1;
    }

    let mean = if count > 0 { sum / count as f64 } else { f64::NAN };
    let std_dev = if count > 1 {
        let variance = (sum_sq - sum * sum / count as f64) / (count as f64 - 1.0);
        if variance > 0.0 {
            variance.sqrt()
        } else {
            0.0
        }
    } else {
        0.0
    };

    if count == 0 {
        return FieldStats {
            min: f64::NAN,
            max: f64::NAN,
            mean: f64::NAN,
            std_dev: f64::NAN,
            count: 0,
            nan_count,
        };
    }

    FieldStats {
        min,
        max,
        mean,
        std_dev,
        count,
        nan_count,
    }
}

/// Compute statistics over a subregion defined by a lat/lon bounding box.
///
/// `values`, `lats`, `lons` are all 1D arrays of length `nx * ny`
/// (row-major order).
pub fn field_stats_region(
    values: &[f64],
    lats: &[f64],
    lons: &[f64],
    nx: usize,
    ny: usize,
    lat_min: f64,
    lat_max: f64,
    lon_min: f64,
    lon_max: f64,
) -> FieldStats {
    let n = nx * ny;
    assert_eq!(values.len(), n, "values length mismatch");
    assert_eq!(lats.len(), n, "lats length mismatch");
    assert_eq!(lons.len(), n, "lons length mismatch");

    let region_values: Vec<f64> = values
        .iter()
        .zip(lats.iter().zip(lons.iter()))
        .filter_map(|(&v, (&lat, &lon))| {
            if lat >= lat_min && lat <= lat_max && lon >= lon_min && lon <= lon_max {
                Some(v)
            } else {
                None
            }
        })
        .collect();

    field_stats(&region_values)
}

// ═══════════════════════════════════════════════════════════
// Field operations (apply_op)
// ═══════════════════════════════════════════════════════════

/// Mathematical operations that can be applied to field values.
#[derive(Debug, Clone)]
pub enum FieldOp {
    Add(f64),
    Multiply(f64),
    KelvinToCelsius,
    CelsiusToFahrenheit,
    MetersToFeet,
    MsToKnots,
    PaToHpa,
    /// kg/m^2 to inches (precipitation, assuming 1 kg/m^2 = 1 mm water)
    KgM2ToInches,
    Clamp(f64, f64),
    Log,
    Sqrt,
    Abs,
}

/// Apply a mathematical operation to all values in a field (in-place).
pub fn apply_op(values: &mut [f64], op: FieldOp) {
    match op {
        FieldOp::Add(x) => {
            for v in values.iter_mut() {
                *v += x;
            }
        }
        FieldOp::Multiply(x) => {
            for v in values.iter_mut() {
                *v *= x;
            }
        }
        FieldOp::KelvinToCelsius => {
            for v in values.iter_mut() {
                *v -= 273.15;
            }
        }
        FieldOp::CelsiusToFahrenheit => {
            for v in values.iter_mut() {
                *v = *v * 9.0 / 5.0 + 32.0;
            }
        }
        FieldOp::MetersToFeet => {
            for v in values.iter_mut() {
                *v *= 3.28084;
            }
        }
        FieldOp::MsToKnots => {
            for v in values.iter_mut() {
                *v *= 1.94384;
            }
        }
        FieldOp::PaToHpa => {
            for v in values.iter_mut() {
                *v /= 100.0;
            }
        }
        FieldOp::KgM2ToInches => {
            // 1 kg/m^2 = 1 mm; 1 inch = 25.4 mm
            for v in values.iter_mut() {
                *v /= 25.4;
            }
        }
        FieldOp::Clamp(lo, hi) => {
            for v in values.iter_mut() {
                *v = v.clamp(lo, hi);
            }
        }
        FieldOp::Log => {
            for v in values.iter_mut() {
                *v = v.ln();
            }
        }
        FieldOp::Sqrt => {
            for v in values.iter_mut() {
                *v = v.sqrt();
            }
        }
        FieldOp::Abs => {
            for v in values.iter_mut() {
                *v = v.abs();
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════
// Smoothing
// ═══════════════════════════════════════════════════════════

/// Smooth a 2D field using a Gaussian kernel.
///
/// `values` is row-major, `nx` columns, `ny` rows. `sigma` is the
/// Gaussian standard deviation in grid-point units.
pub fn smooth_gaussian(values: &[f64], nx: usize, ny: usize, sigma: f64) -> Vec<f64> {
    assert_eq!(values.len(), nx * ny, "smooth_gaussian: length mismatch");

    if sigma <= 0.0 {
        return values.to_vec();
    }

    // Build 1D Gaussian kernel (truncate at 3*sigma)
    let radius = (3.0 * sigma).ceil() as usize;
    let ksize = 2 * radius + 1;
    let mut kernel = vec![0.0; ksize];
    let mut ksum = 0.0;
    for i in 0..ksize {
        let x = i as f64 - radius as f64;
        let w = (-0.5 * (x / sigma).powi(2)).exp();
        kernel[i] = w;
        ksum += w;
    }
    for k in kernel.iter_mut() {
        *k /= ksum;
    }

    let has_nan = values.iter().any(|v| v.is_nan());

    // Separable filter: horizontal pass, then vertical pass.
    // The vertical pass uses a transpose so both passes access memory row-major.
    let mut temp = vec![0.0; nx * ny];

    // Helper: 1D convolution on a contiguous slice (interior only, no bounds checks).
    #[inline(always)]
    fn convolve_1d_interior(src: &[f64], dst: &mut [f64], kernel: &[f64], radius: usize) {
        let ksize = kernel.len();
        let n = src.len();
        // Boundary: renormalize
        for i in 0..radius.min(n) {
            let mut sum = 0.0;
            let mut wsum = 0.0;
            let i_max = (i + radius + 1).min(n);
            for ii in 0..i_max {
                let ki = ii + radius - i;
                sum += src[ii] * kernel[ki];
                wsum += kernel[ki];
            }
            dst[i] = sum / wsum;
        }
        // Interior: full kernel, tight loop, no bounds checks
        for i in radius..(n.saturating_sub(radius)) {
            let mut sum = 0.0;
            let src_start = i - radius;
            for ki in 0..ksize {
                unsafe {
                    sum += *src.get_unchecked(src_start + ki) * *kernel.get_unchecked(ki);
                }
            }
            unsafe { *dst.get_unchecked_mut(i) = sum; }
        }
        // Right/bottom boundary
        for i in n.saturating_sub(radius)..n {
            if i < radius { continue; } // already handled above for tiny n
            let mut sum = 0.0;
            let mut wsum = 0.0;
            let i_min = i - radius;
            for ii in i_min..n {
                let ki = ii + radius - i;
                sum += src[ii] * kernel[ki];
                wsum += kernel[ki];
            }
            dst[i] = sum / wsum;
        }
    }

    if has_nan {
        // NaN-aware horizontal pass
        for j in 0..ny {
            let row_start = j * nx;
            for i in 0..nx {
                let mut sum = 0.0;
                let mut wsum = 0.0;
                let i_min = if i >= radius { i - radius } else { 0 };
                let i_max = (i + radius + 1).min(nx);
                for ii in i_min..i_max {
                    let v = values[row_start + ii];
                    if !v.is_nan() {
                        let ki = ii + radius - i;
                        sum += v * kernel[ki];
                        wsum += kernel[ki];
                    }
                }
                temp[row_start + i] = if wsum > 0.0 { sum / wsum } else { f64::NAN };
            }
        }

        // NaN-aware vertical pass using column buffers for cache locality
        let mut result = vec![0.0; nx * ny];
        let mut col_in = vec![0.0; ny];
        for i in 0..nx {
            // Gather column into contiguous buffer
            for j in 0..ny { col_in[j] = temp[j * nx + i]; }
            // Convolve with NaN awareness
            for j in 0..ny {
                let mut sum = 0.0;
                let mut wsum = 0.0;
                let j_min = if j >= radius { j - radius } else { 0 };
                let j_max = (j + radius + 1).min(ny);
                for jj in j_min..j_max {
                    let v = col_in[jj];
                    if !v.is_nan() {
                        let kj = jj + radius - j;
                        sum += v * kernel[kj];
                        wsum += kernel[kj];
                    }
                }
                result[j * nx + i] = if wsum > 0.0 { sum / wsum } else { f64::NAN };
            }
        }
        result
    } else {
        // Fast path: no NaN values.
        // Horizontal pass — row-major, cache-friendly
        for j in 0..ny {
            let row = &values[j * nx..(j + 1) * nx];
            let dst = &mut temp[j * nx..(j + 1) * nx];
            convolve_1d_interior(row, dst, &kernel, radius);
        }

        // Vertical pass — gather column into contiguous buffer, convolve, scatter back.
        // Column buffer stays in L1 cache (ny * 8 bytes, typically < 8KB).
        let mut result = vec![0.0; nx * ny];
        let mut col_in = vec![0.0; ny];
        let mut col_out = vec![0.0; ny];
        for i in 0..nx {
            // Gather column i from temp (one strided read per column)
            for j in 0..ny {
                unsafe { *col_in.get_unchecked_mut(j) = *temp.get_unchecked(j * nx + i); }
            }
            // Convolve on contiguous buffer (L1-resident)
            convolve_1d_interior(&col_in, &mut col_out, &kernel, radius);
            // Scatter back to result (one strided write per column)
            for j in 0..ny {
                unsafe { *result.get_unchecked_mut(j * nx + i) = *col_out.get_unchecked(j); }
            }
        }
        result
    }
}

/// Smooth using an N-point smoother (like wgrib2's `-smooth`).
///
/// Supported values for `n`: 5 or 9. Other values are treated as 9.
/// `passes` is the number of smoothing iterations to apply.
pub fn smooth_n_point(values: &[f64], nx: usize, ny: usize, n: usize, passes: usize) -> Vec<f64> {
    assert_eq!(values.len(), nx * ny, "smooth_n_point: length mismatch");

    let mut current = values.to_vec();
    let mut scratch = vec![0.0; nx * ny];

    for _ in 0..passes {
        for j in 0..ny {
            for i in 0..nx {
                let c = current[j * nx + i];
                if c.is_nan() {
                    scratch[j * nx + i] = f64::NAN;
                    continue;
                }

                let mut sum = c;
                let mut cnt = 1.0;

                // Cardinal neighbors (N, S, E, W)
                let neighbors_4: [(isize, isize); 4] = [
                    (0, -1),
                    (0, 1),
                    (-1, 0),
                    (1, 0),
                ];
                for &(di, dj) in &neighbors_4 {
                    let ii = i as isize + di;
                    let jj = j as isize + dj;
                    if ii >= 0 && (ii as usize) < nx && jj >= 0 && (jj as usize) < ny {
                        let v = current[jj as usize * nx + ii as usize];
                        if !v.is_nan() {
                            sum += v;
                            cnt += 1.0;
                        }
                    }
                }

                // Diagonal neighbors for 9-point smoother
                if n != 5 {
                    let neighbors_diag: [(isize, isize); 4] = [
                        (-1, -1),
                        (1, -1),
                        (-1, 1),
                        (1, 1),
                    ];
                    for &(di, dj) in &neighbors_diag {
                        let ii = i as isize + di;
                        let jj = j as isize + dj;
                        if ii >= 0 && (ii as usize) < nx && jj >= 0 && (jj as usize) < ny {
                            let v = current[jj as usize * nx + ii as usize];
                            if !v.is_nan() {
                                sum += v;
                                cnt += 1.0;
                            }
                        }
                    }
                }

                scratch[j * nx + i] = sum / cnt;
            }
        }
        std::mem::swap(&mut current, &mut scratch);
    }

    current
}

/// Smooth a 2D field using a rectangular (box) moving average kernel.
///
/// `window_size` is the side length of the square window (must be odd;
/// if even it is incremented by 1). NaN values are excluded from the average.
pub fn smooth_window(values: &[f64], nx: usize, ny: usize, window_size: usize) -> Vec<f64> {
    assert_eq!(values.len(), nx * ny, "smooth_window: length mismatch");

    let ws = if window_size % 2 == 0 { window_size + 1 } else { window_size };
    let half = (ws / 2) as isize;

    if ws <= 1 {
        return values.to_vec();
    }

    let mut result = vec![0.0; nx * ny];

    for j in 0..ny {
        for i in 0..nx {
            let mut sum = 0.0;
            let mut cnt = 0.0;
            for dj in -half..=half {
                let jj = j as isize + dj;
                if jj < 0 || jj as usize >= ny {
                    continue;
                }
                for di in -half..=half {
                    let ii = i as isize + di;
                    if ii < 0 || ii as usize >= nx {
                        continue;
                    }
                    let v = values[jj as usize * nx + ii as usize];
                    if !v.is_nan() {
                        sum += v;
                        cnt += 1.0;
                    }
                }
            }
            result[j * nx + i] = if cnt > 0.0 { sum / cnt } else { f64::NAN };
        }
    }
    result
}

/// Smooth a 2D field using a circular kernel of the given `radius`
/// (in grid-point units).
///
/// Points within `radius` of each grid cell contribute equally to the
/// average. NaN values are excluded.
pub fn smooth_circular(values: &[f64], nx: usize, ny: usize, radius: f64) -> Vec<f64> {
    assert_eq!(values.len(), nx * ny, "smooth_circular: length mismatch");

    if radius <= 0.0 {
        return values.to_vec();
    }

    let half = radius.ceil() as isize;
    let r2 = radius * radius;
    let mut result = vec![0.0; nx * ny];

    for j in 0..ny {
        for i in 0..nx {
            let mut sum = 0.0;
            let mut cnt = 0.0;
            for dj in -half..=half {
                let jj = j as isize + dj;
                if jj < 0 || jj as usize >= ny {
                    continue;
                }
                for di in -half..=half {
                    let ii = i as isize + di;
                    if ii < 0 || ii as usize >= nx {
                        continue;
                    }
                    let dist2 = (di * di + dj * dj) as f64;
                    if dist2 > r2 {
                        continue;
                    }
                    let v = values[jj as usize * nx + ii as usize];
                    if !v.is_nan() {
                        sum += v;
                        cnt += 1.0;
                    }
                }
            }
            result[j * nx + i] = if cnt > 0.0 { sum / cnt } else { f64::NAN };
        }
    }
    result
}

// ═══════════════════════════════════════════════════════════
// Masking
// ═══════════════════════════════════════════════════════════

/// Mask values outside (or inside) a lat/lon bounding box by setting them
/// to NaN.
///
/// `values`, `lats`, `lons` are 1D arrays of length `nx * ny`.
/// When `invert` is false, values OUTSIDE the box are set to NaN.
/// When `invert` is true, values INSIDE the box are set to NaN.
pub fn mask_region(
    values: &mut [f64],
    lats: &[f64],
    lons: &[f64],
    nx: usize,
    ny: usize,
    lat_min: f64,
    lat_max: f64,
    lon_min: f64,
    lon_max: f64,
    invert: bool,
) {
    let n = nx * ny;
    assert_eq!(values.len(), n, "mask_region: values length mismatch");
    assert_eq!(lats.len(), n, "mask_region: lats length mismatch");
    assert_eq!(lons.len(), n, "mask_region: lons length mismatch");

    for i in 0..n {
        let inside =
            lats[i] >= lat_min && lats[i] <= lat_max && lons[i] >= lon_min && lons[i] <= lon_max;
        if invert {
            if inside {
                values[i] = f64::NAN;
            }
        } else if !inside {
            values[i] = f64::NAN;
        }
    }
}

// ═══════════════════════════════════════════════════════════
// Wind operations
// ═══════════════════════════════════════════════════════════

/// Compute wind speed and meteorological direction from U and V components.
///
/// Returns `(speed, direction)` where direction is in degrees (0 = from north,
/// 90 = from east), following the meteorological convention.
pub fn wind_speed_dir(u: &[f64], v: &[f64]) -> (Vec<f64>, Vec<f64>) {
    assert_eq!(u.len(), v.len(), "wind_speed_dir: arrays must have same length");

    let mut speed = Vec::with_capacity(u.len());
    let mut direction = Vec::with_capacity(u.len());

    for (&uu, &vv) in u.iter().zip(v.iter()) {
        let spd = (uu * uu + vv * vv).sqrt();
        speed.push(spd);

        if spd < 1e-10 {
            direction.push(0.0);
        } else {
            // Meteorological convention: direction wind is FROM
            let mut dir = 270.0 - vv.atan2(uu).to_degrees();
            if dir < 0.0 {
                dir += 360.0;
            }
            if dir >= 360.0 {
                dir -= 360.0;
            }
            direction.push(dir);
        }
    }

    (speed, direction)
}

/// Rotate wind components from grid-relative to earth-relative.
///
/// For Lambert Conformal and similar projected grids, the U/V components
/// are relative to the grid axes. This function rotates them to be
/// relative to true east/north.
///
/// `center_lon` is the grid's central longitude (LoV for Lambert Conformal).
pub fn rotate_winds(
    u: &[f64],
    v: &[f64],
    lons: &[f64],
    center_lon: f64,
) -> (Vec<f64>, Vec<f64>) {
    assert_eq!(u.len(), v.len(), "rotate_winds: u and v must have same length");
    assert_eq!(u.len(), lons.len(), "rotate_winds: u and lons must have same length");

    let mut u_earth = Vec::with_capacity(u.len());
    let mut v_earth = Vec::with_capacity(v.len());

    for i in 0..u.len() {
        let angle = (lons[i] - center_lon).to_radians();
        let cos_a = angle.cos();
        let sin_a = angle.sin();
        u_earth.push(u[i] * cos_a - v[i] * sin_a);
        v_earth.push(u[i] * sin_a + v[i] * cos_a);
    }

    (u_earth, v_earth)
}

// ═══════════════════════════════════════════════════════════
// Unit conversion
// ═══════════════════════════════════════════════════════════

/// Convert values between common meteorological units (in-place).
///
/// Supported conversions:
///   - Temperature: K, C, F
///   - Speed: m/s, kt, mph, km/h
///   - Pressure: Pa, hPa, mb, inHg
///   - Distance: m, ft, km
///   - Precipitation: kg/m2, mm, in
///   - Energy: m2/s2, J/kg
pub fn convert_units(values: &mut [f64], from: &str, to: &str) -> Result<(), String> {
    let from_lower = from.to_lowercase();
    let to_lower = to.to_lowercase();

    if from_lower == to_lower {
        return Ok(());
    }

    // Normalize aliases
    let from_norm = normalize_unit(&from_lower);
    let to_norm = normalize_unit(&to_lower);

    if from_norm == to_norm {
        return Ok(());
    }

    // Convert via a canonical intermediate form for each unit category
    match (from_norm.as_str(), to_norm.as_str()) {
        // ── Temperature ──
        ("k", "c") => {
            for v in values.iter_mut() {
                *v -= 273.15;
            }
        }
        ("k", "f") => {
            for v in values.iter_mut() {
                *v = (*v - 273.15) * 9.0 / 5.0 + 32.0;
            }
        }
        ("c", "k") => {
            for v in values.iter_mut() {
                *v += 273.15;
            }
        }
        ("c", "f") => {
            for v in values.iter_mut() {
                *v = *v * 9.0 / 5.0 + 32.0;
            }
        }
        ("f", "c") => {
            for v in values.iter_mut() {
                *v = (*v - 32.0) * 5.0 / 9.0;
            }
        }
        ("f", "k") => {
            for v in values.iter_mut() {
                *v = (*v - 32.0) * 5.0 / 9.0 + 273.15;
            }
        }

        // ── Speed ──
        ("m/s", "kt") => scale(values, 1.94384),
        ("m/s", "mph") => scale(values, 2.23694),
        ("m/s", "km/h") => scale(values, 3.6),
        ("kt", "m/s") => scale(values, 1.0 / 1.94384),
        ("kt", "mph") => scale(values, 1.15078),
        ("kt", "km/h") => scale(values, 1.852),
        ("mph", "m/s") => scale(values, 1.0 / 2.23694),
        ("mph", "kt") => scale(values, 1.0 / 1.15078),
        ("mph", "km/h") => scale(values, 1.60934),
        ("km/h", "m/s") => scale(values, 1.0 / 3.6),
        ("km/h", "kt") => scale(values, 1.0 / 1.852),
        ("km/h", "mph") => scale(values, 1.0 / 1.60934),

        // ── Pressure ──
        ("pa", "hpa") | ("pa", "mb") => scale(values, 0.01),
        ("pa", "inhg") => scale(values, 0.0002953),
        ("hpa", "pa") | ("mb", "pa") => scale(values, 100.0),
        ("hpa", "mb") | ("mb", "hpa") => {} // identical
        ("hpa", "inhg") | ("mb", "inhg") => scale(values, 0.02953),
        ("inhg", "pa") => scale(values, 3386.39),
        ("inhg", "hpa") | ("inhg", "mb") => scale(values, 33.8639),

        // ── Distance ──
        ("m", "ft") => scale(values, 3.28084),
        ("m", "km") => scale(values, 0.001),
        ("ft", "m") => scale(values, 0.3048),
        ("ft", "km") => scale(values, 0.0003048),
        ("km", "m") => scale(values, 1000.0),
        ("km", "ft") => scale(values, 3280.84),

        // ── Precipitation ──
        ("kg/m2", "mm") | ("mm", "kg/m2") => {} // 1:1
        ("kg/m2", "in") | ("mm", "in") => scale(values, 1.0 / 25.4),
        ("in", "kg/m2") | ("in", "mm") => scale(values, 25.4),

        // ── Energy ──
        ("m2/s2", "j/kg") | ("j/kg", "m2/s2") => {} // numerically identical

        _ => {
            return Err(format!(
                "Unsupported unit conversion: '{}' -> '{}'",
                from, to
            ));
        }
    }

    Ok(())
}

/// Normalize unit string to canonical form.
fn normalize_unit(unit: &str) -> String {
    match unit {
        "kelvin" => "k".into(),
        "celsius" | "degc" | "deg_c" => "c".into(),
        "fahrenheit" | "degf" | "deg_f" => "f".into(),
        "knots" | "knot" | "kts" => "kt".into(),
        "meters/second" | "meters/sec" | "ms" | "m_s" => "m/s".into(),
        "miles/hour" | "miles_per_hour" => "mph".into(),
        "km/hr" | "kph" | "kilometers/hour" => "km/h".into(),
        "pascal" | "pascals" => "pa".into(),
        "hectopascal" | "hectopascals" => "hpa".into(),
        "millibar" | "millibars" | "mbar" => "mb".into(),
        "inches_hg" | "inches_of_mercury" => "inhg".into(),
        "meter" | "meters" => "m".into(),
        "foot" | "feet" => "ft".into(),
        "kilometer" | "kilometers" => "km".into(),
        "kilogram/m2" | "kg_m2" | "kgm2" => "kg/m2".into(),
        "millimeter" | "millimeters" => "mm".into(),
        "inch" | "inches" => "in".into(),
        other => other.into(),
    }
}

/// Helper: multiply all values by a constant.
fn scale(values: &mut [f64], factor: f64) {
    for v in values.iter_mut() {
        *v *= factor;
    }
}

// ═══════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grib2::parser::{
        DataRepresentation, GridDefinition, ProductDefinition,
    };

    /// Helper: build a minimal Grib2File with `n` messages, each containing
    /// `values` on a simple lat/lon grid.
    fn make_file(n: usize, values: &[f64]) -> Grib2File {
        let nx = 3u32;
        let ny = (values.len() / nx as usize) as u32;
        let grid = GridDefinition {
            template: 0,
            nx,
            ny,
            lat1: 0.0,
            lon1: 0.0,
            lat2: (ny - 1) as f64,
            lon2: (nx - 1) as f64,
            dx: 1.0,
            dy: 1.0,
            scan_mode: 0,
            ..GridDefinition::default()
        };

        let mut messages = Vec::new();
        for i in 0..n {
            // Create via writer -> parse roundtrip so raw_data is valid
            let product = ProductDefinition {
                template: 0,
                parameter_category: i as u8,
                parameter_number: 0,
                level_type: 103,
                level_value: 2.0,
                ..ProductDefinition::default()
            };
            let builder = MessageBuilder::new(0, values.to_vec())
                .grid(grid.clone())
                .product(product)
                .packing(PackingMethod::Simple { bits_per_value: 16 });
            let writer = Grib2Writer::new().add_message(builder);
            let bytes = writer.to_bytes().unwrap();
            let parsed = Grib2File::from_bytes(&bytes).unwrap();
            messages.push(parsed.messages.into_iter().next().unwrap());
        }

        Grib2File { messages }
    }

    #[test]
    fn test_merge() {
        let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let f1 = make_file(2, &vals);
        let f2 = make_file(1, &vals);
        let merged = merge(&[&f1, &f2]);
        assert_eq!(merged.messages.len(), 3);
    }

    #[test]
    fn test_subset() {
        let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let f = make_file(5, &vals);
        let sub = subset(&f, &[0, 2, 4]);
        assert_eq!(sub.messages.len(), 3);
        assert_eq!(sub.messages[0].product.parameter_category, 0);
        assert_eq!(sub.messages[1].product.parameter_category, 2);
        assert_eq!(sub.messages[2].product.parameter_category, 4);
    }

    #[test]
    fn test_subset_out_of_range() {
        let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let f = make_file(2, &vals);
        let sub = subset(&f, &[0, 99]);
        assert_eq!(sub.messages.len(), 1);
    }

    #[test]
    fn test_filter() {
        let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let f = make_file(5, &vals);
        let filtered = filter(&f, |m| m.product.parameter_category < 3);
        assert_eq!(filtered.messages.len(), 3);
    }

    #[test]
    fn test_field_stats() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = field_stats(&values);
        assert_eq!(stats.count, 5);
        assert_eq!(stats.nan_count, 0);
        assert!((stats.min - 1.0).abs() < 1e-10);
        assert!((stats.max - 5.0).abs() < 1e-10);
        assert!((stats.mean - 3.0).abs() < 1e-10);
        // std_dev of [1,2,3,4,5] (sample) = sqrt(2.5) ≈ 1.5811
        assert!((stats.std_dev - 1.5811388).abs() < 1e-4);
    }

    #[test]
    fn test_field_stats_with_nan() {
        let values = vec![1.0, f64::NAN, 3.0, f64::NAN, 5.0];
        let stats = field_stats(&values);
        assert_eq!(stats.count, 3);
        assert_eq!(stats.nan_count, 2);
        assert!((stats.min - 1.0).abs() < 1e-10);
        assert!((stats.max - 5.0).abs() < 1e-10);
        assert!((stats.mean - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_field_stats_empty() {
        let values: Vec<f64> = vec![];
        let stats = field_stats(&values);
        assert_eq!(stats.count, 0);
        assert!(stats.mean.is_nan());
    }

    #[test]
    fn test_apply_op_kelvin_to_celsius() {
        let mut values = vec![273.15, 300.0, 250.0];
        apply_op(&mut values, FieldOp::KelvinToCelsius);
        assert!((values[0] - 0.0).abs() < 1e-10);
        assert!((values[1] - 26.85).abs() < 1e-10);
        assert!((values[2] - (-23.15)).abs() < 1e-10);
    }

    #[test]
    fn test_apply_op_clamp() {
        let mut values = vec![-5.0, 0.0, 5.0, 10.0, 15.0];
        apply_op(&mut values, FieldOp::Clamp(0.0, 10.0));
        assert_eq!(values, vec![0.0, 0.0, 5.0, 10.0, 10.0]);
    }

    #[test]
    fn test_apply_op_add_multiply() {
        let mut values = vec![1.0, 2.0, 3.0];
        apply_op(&mut values, FieldOp::Add(10.0));
        assert_eq!(values, vec![11.0, 12.0, 13.0]);

        apply_op(&mut values, FieldOp::Multiply(2.0));
        assert_eq!(values, vec![22.0, 24.0, 26.0]);
    }

    #[test]
    fn test_smooth_gaussian() {
        // 3x3 field, center value is high
        let values = vec![0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0];
        let smoothed = smooth_gaussian(&values, 3, 3, 1.0);
        // Center should be reduced, neighbors should increase
        assert!(smoothed[4] < 100.0);
        assert!(smoothed[0] > 0.0);
        // The sum should be approximately conserved (within edge effects)
        let original_sum: f64 = values.iter().sum();
        let smoothed_sum: f64 = smoothed.iter().sum();
        assert!(
            (smoothed_sum - original_sum).abs() / original_sum < 0.5,
            "sum conservation: {} vs {}",
            smoothed_sum,
            original_sum
        );
    }

    #[test]
    fn test_smooth_n_point() {
        let values = vec![0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0];
        let smoothed = smooth_n_point(&values, 3, 3, 5, 1);
        // Center should be reduced
        assert!(smoothed[4] < 100.0);
        assert!(smoothed[4] > 0.0);
    }

    #[test]
    fn test_convert_units_temperature() {
        let mut values = vec![300.0];
        convert_units(&mut values, "K", "C").unwrap();
        assert!((values[0] - 26.85).abs() < 0.01);

        convert_units(&mut values, "C", "F").unwrap();
        assert!((values[0] - 80.33).abs() < 0.01);

        convert_units(&mut values, "F", "K").unwrap();
        assert!((values[0] - 300.0).abs() < 0.01);
    }

    #[test]
    fn test_convert_units_speed() {
        let mut values = vec![10.0]; // 10 m/s
        convert_units(&mut values, "m/s", "kt").unwrap();
        assert!((values[0] - 19.4384).abs() < 0.01);

        convert_units(&mut values, "kt", "mph").unwrap();
        assert!((values[0] - 22.3694).abs() < 0.02);
    }

    #[test]
    fn test_convert_units_pressure() {
        let mut values = vec![101325.0]; // standard atmosphere in Pa
        convert_units(&mut values, "Pa", "hPa").unwrap();
        assert!((values[0] - 1013.25).abs() < 0.01);

        convert_units(&mut values, "hPa", "inHg").unwrap();
        assert!((values[0] - 29.92).abs() < 0.02);
    }

    #[test]
    fn test_convert_units_noop() {
        let mut values = vec![42.0];
        convert_units(&mut values, "K", "K").unwrap();
        assert_eq!(values[0], 42.0);
    }

    #[test]
    fn test_convert_units_unsupported() {
        let mut values = vec![1.0];
        let result = convert_units(&mut values, "furlongs", "cubits");
        assert!(result.is_err());
    }

    #[test]
    fn test_wind_speed_dir() {
        // Pure east wind (u=10, v=0) -> speed=10, dir=270 (from west)
        let u = vec![10.0];
        let v = vec![0.0];
        let (speed, dir) = wind_speed_dir(&u, &v);
        assert!((speed[0] - 10.0).abs() < 1e-10);
        assert!((dir[0] - 270.0).abs() < 0.1);

        // Pure north wind (u=0, v=10) -> speed=10, dir=180 (from south)
        let u = vec![0.0];
        let v = vec![10.0];
        let (speed, dir) = wind_speed_dir(&u, &v);
        assert!((speed[0] - 10.0).abs() < 1e-10);
        assert!((dir[0] - 180.0).abs() < 0.1);

        // Calm wind
        let u = vec![0.0];
        let v = vec![0.0];
        let (speed, dir) = wind_speed_dir(&u, &v);
        assert!(speed[0] < 1e-10);
        assert!((dir[0] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_rotate_winds() {
        // No rotation at center longitude
        let u = vec![10.0];
        let v = vec![5.0];
        let lons = vec![262.5]; // same as center_lon
        let (ue, ve) = rotate_winds(&u, &v, &lons, 262.5);
        assert!((ue[0] - 10.0).abs() < 1e-10);
        assert!((ve[0] - 5.0).abs() < 1e-10);

        // Rotation should change values when lon != center_lon
        let lons2 = vec![272.5]; // 10 degrees east of center
        let (ue2, ve2) = rotate_winds(&u, &v, &lons2, 262.5);
        // Should differ from original
        assert!((ue2[0] - 10.0).abs() > 0.01 || (ve2[0] - 5.0).abs() > 0.01);
        // But magnitude should be preserved
        let orig_mag = (u[0] * u[0] + v[0] * v[0]).sqrt();
        let rot_mag = (ue2[0] * ue2[0] + ve2[0] * ve2[0]).sqrt();
        assert!((orig_mag - rot_mag).abs() < 1e-10);
    }

    #[test]
    fn test_field_diff() {
        let a = vec![10.0, 20.0, 30.0];
        let b = vec![1.0, 2.0, 3.0];
        let diff = field_diff(&a, &b);
        assert_eq!(diff, vec![9.0, 18.0, 27.0]);
    }

    #[test]
    fn test_mask_region() {
        let mut values = vec![1.0, 2.0, 3.0, 4.0];
        let lats = vec![30.0, 30.0, 40.0, 40.0];
        let lons = vec![-100.0, -90.0, -100.0, -90.0];

        // Mask outside the box: keep only lat >= 35
        mask_region(
            &mut values, &lats, &lons, 2, 2,
            35.0, 45.0, -105.0, -85.0, false,
        );
        assert!(values[0].is_nan()); // lat=30, outside
        assert!(values[1].is_nan()); // lat=30, outside
        assert!((values[2] - 3.0).abs() < 1e-10); // lat=40, inside
        assert!((values[3] - 4.0).abs() < 1e-10); // lat=40, inside
    }

    #[test]
    fn test_mask_region_invert() {
        let mut values = vec![1.0, 2.0, 3.0, 4.0];
        let lats = vec![30.0, 30.0, 40.0, 40.0];
        let lons = vec![-100.0, -90.0, -100.0, -90.0];

        // Invert: mask INSIDE the box
        mask_region(
            &mut values, &lats, &lons, 2, 2,
            35.0, 45.0, -105.0, -85.0, true,
        );
        assert!((values[0] - 1.0).abs() < 1e-10); // outside, kept
        assert!((values[1] - 2.0).abs() < 1e-10); // outside, kept
        assert!(values[2].is_nan()); // inside, masked
        assert!(values[3].is_nan()); // inside, masked
    }

    #[test]
    fn test_smooth_window() {
        // 3x3 field, center value is high
        let values = vec![0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0];
        let smoothed = smooth_window(&values, 3, 3, 3);
        // Center: average of all 9 values = 100/9 ≈ 11.11
        assert!(
            (smoothed[4] - 100.0 / 9.0).abs() < 1e-10,
            "center = {}, expected {}",
            smoothed[4],
            100.0 / 9.0
        );
        // Corner (0,0): average of 4 values (0,0), (1,0), (0,1), (1,1) = 100/4 = 25
        // Actually corner (0,0) sees (0,0),(1,0),(0,1),(1,1) = 0+0+0+100 = 100, /4 = 25
        assert!(
            (smoothed[0] - 25.0).abs() < 1e-10,
            "corner = {}, expected 25.0",
            smoothed[0]
        );
    }

    #[test]
    fn test_smooth_circular() {
        // 5x5, center spike
        let mut values = vec![0.0; 25];
        values[12] = 100.0; // center
        let smoothed = smooth_circular(&values, 5, 5, 1.0);
        // With radius 1.0, the kernel includes the center + 4 cardinal neighbors
        // (diagonals have dist sqrt(2) > 1.0)
        // So center = (100 + 0 + 0 + 0 + 0) / 5 = 20
        assert!(
            (smoothed[12] - 20.0).abs() < 1e-10,
            "center = {}, expected 20.0",
            smoothed[12]
        );
    }

    // =========================================================================
    // smooth_gaussian: sigma=0 returns input unchanged
    // =========================================================================

    #[test]
    fn test_smooth_gaussian_sigma_zero() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let result = smooth_gaussian(&values, 3, 3, 0.0);
        for (i, (&orig, &smoothed)) in values.iter().zip(result.iter()).enumerate() {
            assert!(
                (orig - smoothed).abs() < 1e-10,
                "sigma=0: index {}: orig={}, smoothed={}",
                i, orig, smoothed
            );
        }
    }

    #[test]
    fn test_smooth_gaussian_negative_sigma() {
        // Negative sigma should also return input unchanged (same as sigma=0)
        let values = vec![10.0, 20.0, 30.0, 40.0];
        let result = smooth_gaussian(&values, 2, 2, -1.0);
        for (i, (&orig, &smoothed)) in values.iter().zip(result.iter()).enumerate() {
            assert!(
                (orig - smoothed).abs() < 1e-10,
                "sigma<0: index {}: orig={}, smoothed={}",
                i, orig, smoothed
            );
        }
    }

    // =========================================================================
    // smooth_gaussian: constant field returns the same constant
    // =========================================================================

    #[test]
    fn test_smooth_gaussian_constant_field() {
        let val = 42.0;
        let nx = 7;
        let ny = 7;
        let values = vec![val; nx * ny];
        let result = smooth_gaussian(&values, nx, ny, 2.0);
        for (i, &v) in result.iter().enumerate() {
            assert!(
                (v - val).abs() < 1e-10,
                "constant field: index {}: got={}, expected={}",
                i, v, val
            );
        }
    }

    #[test]
    fn test_smooth_gaussian_constant_large_sigma() {
        // Even with a very large sigma, a constant field should remain constant
        let val = -7.5;
        let nx = 10;
        let ny = 10;
        let values = vec![val; nx * ny];
        let result = smooth_gaussian(&values, nx, ny, 10.0);
        for (i, &v) in result.iter().enumerate() {
            assert!(
                (v - val).abs() < 1e-9,
                "constant/large sigma: index {}: got={}, expected={}",
                i, v, val
            );
        }
    }

    // =========================================================================
    // smooth_gaussian: preserves mean (interior)
    // =========================================================================

    #[test]
    fn test_smooth_gaussian_preserves_mean_interior() {
        // Gaussian smoothing preserves the global sum for interior pixels
        // (boundary renormalization can shift things slightly).
        // Use a large enough grid that interior dominates.
        let nx = 21;
        let ny = 21;
        let n = nx * ny;
        let mut values = vec![0.0; n];
        // Create a pattern
        for j in 0..ny {
            for i in 0..nx {
                values[j * nx + i] = ((i + j) as f64).sin() * 10.0;
            }
        }
        let sigma = 1.5;
        let result = smooth_gaussian(&values, nx, ny, sigma);

        // Compare mean of interior (skip 4 pixels of border)
        let border = 4;
        let mut sum_orig = 0.0;
        let mut sum_smooth = 0.0;
        let mut count = 0;
        for j in border..ny - border {
            for i in border..nx - border {
                sum_orig += values[j * nx + i];
                sum_smooth += result[j * nx + i];
                count += 1;
            }
        }
        let mean_orig = sum_orig / count as f64;
        let mean_smooth = sum_smooth / count as f64;
        assert!(
            (mean_orig - mean_smooth).abs() < 0.1,
            "mean not preserved: orig={}, smooth={}",
            mean_orig, mean_smooth
        );
    }

    // =========================================================================
    // smooth_gaussian: NaN handling
    // =========================================================================

    #[test]
    fn test_smooth_gaussian_nan_isolated() {
        // A single NaN in the middle should not spread to fill the entire output.
        let nx = 5;
        let ny = 5;
        let n = nx * ny;
        let mut values = vec![1.0; n];
        values[12] = f64::NAN; // center
        let result = smooth_gaussian(&values, nx, ny, 1.0);

        // Corners should still be finite (far from NaN)
        assert!(result[0].is_finite(), "corner (0,0) should be finite");
        assert!(result[4].is_finite(), "corner (4,0) should be finite");
        assert!(result[20].is_finite(), "corner (0,4) should be finite");
        assert!(result[24].is_finite(), "corner (4,4) should be finite");
    }

    #[test]
    fn test_smooth_gaussian_all_nan() {
        // All NaN input should return all NaN
        let nx = 3;
        let ny = 3;
        let values = vec![f64::NAN; nx * ny];
        let result = smooth_gaussian(&values, nx, ny, 1.0);
        for (i, &v) in result.iter().enumerate() {
            assert!(v.is_nan(), "all-NaN: index {} should be NaN, got {}", i, v);
        }
    }

    #[test]
    fn test_smooth_gaussian_nan_does_not_corrupt_distant() {
        // Place NaN at (0,0); check that far corner (nx-1, ny-1) is unaffected
        let nx = 11;
        let ny = 11;
        let n = nx * ny;
        let mut values = vec![5.0; n];
        values[0] = f64::NAN;
        let result = smooth_gaussian(&values, nx, ny, 1.0);
        let far_corner = result[(ny - 1) * nx + (nx - 1)];
        assert!(
            far_corner.is_finite() && (far_corner - 5.0).abs() < 0.1,
            "far corner should be ~5.0, got {}",
            far_corner
        );
    }

    // =========================================================================
    // merge, subset, filter additional tests
    // =========================================================================

    #[test]
    fn test_merge_empty() {
        let merged = merge(&[]);
        assert_eq!(merged.messages.len(), 0);
    }

    #[test]
    fn test_subset_empty_indices() {
        let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let f = make_file(3, &vals);
        let sub = subset(&f, &[]);
        assert_eq!(sub.messages.len(), 0);
    }

    #[test]
    fn test_filter_none_match() {
        let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let f = make_file(3, &vals);
        let filtered = filter(&f, |_| false);
        assert_eq!(filtered.messages.len(), 0);
    }

    #[test]
    fn test_filter_all_match() {
        let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let f = make_file(4, &vals);
        let filtered = filter(&f, |_| true);
        assert_eq!(filtered.messages.len(), 4);
    }

    #[test]
    fn test_field_diff_basic() {
        let a = vec![5.0, 10.0, 15.0];
        let b = vec![1.0, 2.0, 3.0];
        let diff = field_diff(&a, &b);
        assert!((diff[0] - 4.0).abs() < 1e-10);
        assert!((diff[1] - 8.0).abs() < 1e-10);
        assert!((diff[2] - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_field_stats_with_nan_extended() {
        let values = vec![1.0, f64::NAN, 3.0, f64::NAN, 5.0];
        let stats = field_stats(&values);
        assert_eq!(stats.count, 3);
        assert_eq!(stats.nan_count, 2);
        assert!((stats.min - 1.0).abs() < 1e-10);
        assert!((stats.max - 5.0).abs() < 1e-10);
        assert!((stats.mean - 3.0).abs() < 1e-10);
    }
}
