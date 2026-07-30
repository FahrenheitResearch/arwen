use grib_core::grib2::{
    Grib2File, Grib2Message, flip_rows, grid_latlon, level_name, parameter_name, parameter_units,
    unpack_message,
};
use rustwx_core::{GridShape, LatLonGrid, ModelId};
use serde::{Deserialize, Serialize};

use crate::source::ProductSourceMode;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeThermoRecipe {
    Sbcape,
    Sbcin,
    Sblcl,
    Mlcape,
    Mlcin,
    Mucape,
    Mucin,
    LiftedIndex,
}

impl NativeThermoRecipe {
    pub const fn slug(self) -> &'static str {
        match self {
            Self::Sbcape => "sbcape",
            Self::Sbcin => "sbcin",
            Self::Sblcl => "sblcl",
            Self::Mlcape => "mlcape",
            Self::Mlcin => "mlcin",
            Self::Mucape => "mucape",
            Self::Mucin => "mucin",
            Self::LiftedIndex => "lifted_index",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeSemantics {
    ExactEquivalent,
    ProxyEquivalent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeRoute {
    Derived,
    NativeExact,
    NativeProxy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeComparisonVerdict {
    Pass,
    Review,
    Reject,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldSummaryStats {
    pub min: f64,
    pub p01: f64,
    pub p05: f64,
    pub p25: f64,
    pub median: f64,
    pub p75: f64,
    pub p95: f64,
    pub p99: f64,
    pub max: f64,
    pub mean: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NativeDerivedComparisonStats {
    pub native: FieldSummaryStats,
    pub derived: FieldSummaryStats,
    pub valid_points: usize,
    pub correlation_r: f64,
    pub mean_signed_diff: f64,
    pub mean_abs_diff: f64,
    pub rmse: f64,
    pub max_abs_diff: f64,
    pub domain_mean_native: f64,
    pub domain_mean_derived: f64,
    pub verdict: NativeComparisonVerdict,
    pub verdict_reason: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct NativeThermoCandidate {
    pub recipe: NativeThermoRecipe,
    pub label: &'static str,
    pub semantics: NativeSemantics,
    pub auto_eligible: bool,
    pub detail: &'static str,
    pub fetch_product: &'static str,
    pub resolved_parameter_name: &'static str,
    pub resolved_level_name: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct NativeThermoField {
    pub recipe: NativeThermoRecipe,
    pub candidate: NativeThermoCandidate,
    pub units: String,
    pub grid: LatLonGrid,
    pub values: Vec<f64>,
    pub parameter_name: String,
    pub level_name: String,
    pub level_type: u8,
    pub level_value: f64,
}

#[derive(Debug, Clone, Copy)]
struct NativeThermoSelector {
    discipline: u8,
    category: u8,
    number: u8,
    level_type: u8,
    level_value: Option<f64>,
}

impl NativeThermoSelector {
    fn matches(self, message: &Grib2Message) -> bool {
        message.discipline == self.discipline
            && message.product.parameter_category == self.category
            && message.product.parameter_number == self.number
            && message.product.level_type == self.level_type
            && self
                .level_value
                .map(|level| (message.product.level_value - level).abs() < 0.25)
                .unwrap_or(true)
    }
}

pub fn native_candidate(
    model: ModelId,
    recipe: NativeThermoRecipe,
) -> Option<NativeThermoCandidate> {
    let (selector, label, semantics, auto_eligible, detail, fetch_product) =
        candidate_spec(model, recipe)?;
    Some(NativeThermoCandidate {
        recipe,
        label,
        semantics,
        auto_eligible,
        detail,
        fetch_product,
        resolved_parameter_name: parameter_name(
            selector.discipline,
            selector.category,
            selector.number,
        ),
        resolved_level_name: level_name(selector.level_type),
    })
}

pub fn native_candidate_for_slug(model: ModelId, slug: &str) -> Option<NativeThermoCandidate> {
    let recipe = match slug {
        "sbcape" => NativeThermoRecipe::Sbcape,
        "sbcin" => NativeThermoRecipe::Sbcin,
        "sblcl" => NativeThermoRecipe::Sblcl,
        "mlcape" => NativeThermoRecipe::Mlcape,
        "mlcin" => NativeThermoRecipe::Mlcin,
        "mucape" => NativeThermoRecipe::Mucape,
        "mucin" => NativeThermoRecipe::Mucin,
        "lifted_index" => NativeThermoRecipe::LiftedIndex,
        _ => return None,
    };
    native_candidate(model, recipe)
}

pub fn native_candidate_allowed_in_fastest(
    mode: ProductSourceMode,
    model: ModelId,
    recipe: NativeThermoRecipe,
) -> bool {
    matches!(mode, ProductSourceMode::Fastest) && native_candidate(model, recipe).is_some()
}

pub fn extract_native_thermo_field(
    model: ModelId,
    recipe: NativeThermoRecipe,
    bytes: &[u8],
) -> Result<Option<NativeThermoField>, Box<dyn std::error::Error>> {
    let Some((selector, label, semantics, auto_eligible, detail, fetch_product)) =
        candidate_spec(model, recipe)
    else {
        return Ok(None);
    };
    let candidate = NativeThermoCandidate {
        recipe,
        label,
        semantics,
        auto_eligible,
        detail,
        fetch_product,
        resolved_parameter_name: parameter_name(
            selector.discipline,
            selector.category,
            selector.number,
        ),
        resolved_level_name: level_name(selector.level_type),
    };
    let grib = Grib2File::from_bytes(bytes)?;
    let Some(message) = grib
        .messages
        .iter()
        .find(|message| selector.matches(message))
    else {
        return Ok(None);
    };
    Ok(Some(build_native_field(recipe, candidate, message)?))
}

pub fn crop_native_field(
    field: &NativeThermoField,
    bounds: (f64, f64, f64, f64),
) -> Result<NativeThermoField, Box<dyn std::error::Error>> {
    let nx = field.grid.shape.nx;
    let ny = field.grid.shape.ny;
    let mut min_x = nx;
    let mut max_x = 0usize;
    let mut min_y = ny;
    let mut max_y = 0usize;
    let mut found = false;

    for y in 0..ny {
        let row_offset = y * nx;
        for x in 0..nx {
            let idx = row_offset + x;
            let lat = f64::from(field.grid.lat_deg[idx]);
            let lon = f64::from(field.grid.lon_deg[idx]);
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
        return Err("requested native thermo crop produced an empty domain".into());
    }

    if min_x == 0 && max_x + 1 == nx && min_y == 0 && max_y + 1 == ny {
        return Ok(field.clone());
    }

    let x_start = min_x;
    let x_end = max_x + 1;
    let y_start = min_y;
    let y_end = max_y + 1;
    let width = x_end - x_start;
    let height = y_end - y_start;

    let crop_2d = |values: &[f32]| -> Vec<f32> {
        let mut cropped = Vec::with_capacity(width * height);
        for y in y_start..y_end {
            let start = y * nx + x_start;
            let end = y * nx + x_end;
            cropped.extend_from_slice(&values[start..end]);
        }
        cropped
    };
    let crop_2d_f64 = |values: &[f64]| -> Vec<f64> {
        let mut cropped = Vec::with_capacity(width * height);
        for y in y_start..y_end {
            let start = y * nx + x_start;
            let end = y * nx + x_end;
            cropped.extend_from_slice(&values[start..end]);
        }
        cropped
    };

    let grid = LatLonGrid::new(
        GridShape::new(width, height)?,
        crop_2d(&field.grid.lat_deg),
        crop_2d(&field.grid.lon_deg),
    )?;

    Ok(NativeThermoField {
        recipe: field.recipe,
        candidate: field.candidate.clone(),
        units: field.units.clone(),
        grid,
        values: crop_2d_f64(&field.values),
        parameter_name: field.parameter_name.clone(),
        level_name: field.level_name.clone(),
        level_type: field.level_type,
        level_value: field.level_value,
    })
}

fn point_in_geographic_bounds(lon: f64, lat: f64, bounds: (f64, f64, f64, f64)) -> bool {
    if !lon.is_finite() || !lat.is_finite() || lat < bounds.2 || lat > bounds.3 {
        return false;
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

fn normalize_longitude_for_bounds(lon: f64) -> f64 {
    let mut lon = lon % 360.0;
    if lon > 180.0 {
        lon -= 360.0;
    } else if lon <= -180.0 {
        lon += 360.0;
    }
    lon
}

pub fn compare_native_vs_derived(
    recipe_slug: &str,
    native: &[f64],
    derived: &[f64],
) -> Result<NativeDerivedComparisonStats, Box<dyn std::error::Error>> {
    if native.len() != derived.len() {
        return Err(format!(
            "native/derived length mismatch for {recipe_slug}: {} vs {}",
            native.len(),
            derived.len()
        )
        .into());
    }

    let paired = native
        .iter()
        .copied()
        .zip(derived.iter().copied())
        .filter(|(native_value, derived_value)| {
            native_value.is_finite() && derived_value.is_finite()
        })
        .collect::<Vec<_>>();
    if paired.is_empty() {
        return Err(format!("no finite native/derived comparison points for {recipe_slug}").into());
    }

    let native_values = paired
        .iter()
        .map(|(native_value, _)| *native_value)
        .collect::<Vec<_>>();
    let derived_values = paired
        .iter()
        .map(|(_, derived_value)| *derived_value)
        .collect::<Vec<_>>();
    let mean_native = mean(&native_values);
    let mean_derived = mean(&derived_values);

    let mut sum_cov = 0.0;
    let mut sum_native_var = 0.0;
    let mut sum_derived_var = 0.0;
    let mut sum_signed_diff = 0.0;
    let mut sum_abs_diff = 0.0;
    let mut sum_sq_diff = 0.0;
    let mut max_abs_diff: f64 = 0.0;

    for (native_value, derived_value) in &paired {
        let native_delta = *native_value - mean_native;
        let derived_delta = *derived_value - mean_derived;
        let diff = *native_value - *derived_value;
        sum_cov += native_delta * derived_delta;
        sum_native_var += native_delta * native_delta;
        sum_derived_var += derived_delta * derived_delta;
        sum_signed_diff += diff;
        sum_abs_diff += diff.abs();
        sum_sq_diff += diff * diff;
        max_abs_diff = max_abs_diff.max(diff.abs());
    }

    let valid_points = paired.len();
    let correlation_r = if sum_native_var > 0.0 && sum_derived_var > 0.0 {
        sum_cov / (sum_native_var.sqrt() * sum_derived_var.sqrt())
    } else {
        0.0
    };
    let mean_signed_diff = sum_signed_diff / valid_points as f64;
    let mean_abs_diff = sum_abs_diff / valid_points as f64;
    let rmse = (sum_sq_diff / valid_points as f64).sqrt();

    let verdict = if correlation_r >= 0.98 && mean_abs_diff <= tolerance_for_recipe(recipe_slug) {
        NativeComparisonVerdict::Pass
    } else if correlation_r >= 0.90 && mean_abs_diff <= tolerance_for_recipe(recipe_slug) * 2.0 {
        NativeComparisonVerdict::Review
    } else {
        NativeComparisonVerdict::Reject
    };
    let verdict_reason = format!(
        "corr={:.4}, mad={:.3}, rmse={:.3}, tolerance={:.3}",
        correlation_r,
        mean_abs_diff,
        rmse,
        tolerance_for_recipe(recipe_slug)
    );

    Ok(NativeDerivedComparisonStats {
        native: summarize(&native_values),
        derived: summarize(&derived_values),
        valid_points,
        correlation_r,
        mean_signed_diff,
        mean_abs_diff,
        rmse,
        max_abs_diff,
        domain_mean_native: mean_native,
        domain_mean_derived: mean_derived,
        verdict,
        verdict_reason,
    })
}

fn candidate_spec(
    model: ModelId,
    recipe: NativeThermoRecipe,
) -> Option<(
    NativeThermoSelector,
    &'static str,
    NativeSemantics,
    bool,
    &'static str,
    &'static str,
)> {
    use NativeSemantics::{ExactEquivalent, ProxyEquivalent};
    use NativeThermoRecipe::*;

    match (model, recipe) {
        (ModelId::Hrrr, Sbcape) => Some((
            NativeThermoSelector {
                discipline: 0,
                category: 7,
                number: 6,
                level_type: 1,
                level_value: Some(0.0),
            },
            "surface CAPE",
            ExactEquivalent,
            true,
            "native surface CAPE validated against canonical-derived baseline",
            "sfc",
        )),
        (ModelId::Gfs, Sbcape) | (ModelId::RrfsA, Sbcape) => Some((
            NativeThermoSelector {
                discipline: 0,
                category: 7,
                number: 6,
                level_type: 1,
                level_value: Some(0.0),
            },
            "surface CAPE",
            ExactEquivalent,
            false,
            if matches!(model, ModelId::RrfsA) {
                "RRFS surface CAPE candidate; compare before promotion"
            } else {
                "GFS surface CAPE candidate; compare before promotion"
            },
            if matches!(model, ModelId::RrfsA) {
                "nat-na"
            } else {
                "pgrb2.0p25"
            },
        )),
        (ModelId::Hrrr, Sbcin) => Some((
            NativeThermoSelector {
                discipline: 0,
                category: 7,
                number: 7,
                level_type: 1,
                level_value: Some(0.0),
            },
            "surface CIN",
            ExactEquivalent,
            false,
            "HRRR surface CIN candidate; compare before promotion",
            "sfc",
        )),
        (ModelId::Gfs, Sbcin) | (ModelId::RrfsA, Sbcin) => Some((
            NativeThermoSelector {
                discipline: 0,
                category: 7,
                number: 7,
                level_type: 1,
                level_value: Some(0.0),
            },
            "surface CIN",
            ExactEquivalent,
            false,
            if matches!(model, ModelId::RrfsA) {
                "RRFS surface CIN candidate; compare before promotion"
            } else {
                "GFS surface CIN candidate; compare before promotion"
            },
            if matches!(model, ModelId::RrfsA) {
                "nat-na"
            } else {
                "pgrb2.0p25"
            },
        )),
        (ModelId::Hrrr, Sblcl) => Some((
            NativeThermoSelector {
                discipline: 0,
                category: 3,
                number: 5,
                level_type: 5,
                level_value: Some(0.0),
            },
            "surface LCL height",
            ExactEquivalent,
            false,
            "HRRR surface LCL-height candidate; compare before promotion",
            "sfc",
        )),
        (ModelId::RrfsA, Sblcl) => Some((
            NativeThermoSelector {
                discipline: 0,
                category: 3,
                number: 5,
                level_type: 2,
                level_value: Some(0.0),
            },
            "cloud-base height",
            ProxyEquivalent,
            false,
            "RRFS cloud-base height proxy for surface LCL height",
            "nat-na",
        )),
        (ModelId::Hrrr, Mlcape) | (ModelId::Gfs, Mlcape) | (ModelId::RrfsA, Mlcape) => Some((
            NativeThermoSelector {
                discipline: 0,
                category: 7,
                number: 6,
                level_type: 108,
                level_value: Some(9000.0),
            },
            "90-0 mb CAPE",
            ProxyEquivalent,
            false,
            "native mixed-layer CAPE proxy",
            if matches!(model, ModelId::Hrrr) {
                "sfc"
            } else if matches!(model, ModelId::RrfsA) {
                "nat-na"
            } else {
                "pgrb2.0p25"
            },
        )),
        (ModelId::Hrrr, Mlcin) | (ModelId::Gfs, Mlcin) | (ModelId::RrfsA, Mlcin) => Some((
            NativeThermoSelector {
                discipline: 0,
                category: 7,
                number: 7,
                level_type: 108,
                level_value: Some(9000.0),
            },
            "90-0 mb CIN",
            ProxyEquivalent,
            false,
            "native mixed-layer CIN proxy",
            if matches!(model, ModelId::Hrrr) {
                "sfc"
            } else if matches!(model, ModelId::RrfsA) {
                "nat-na"
            } else {
                "pgrb2.0p25"
            },
        )),
        (ModelId::Hrrr, Mucape) | (ModelId::Gfs, Mucape) | (ModelId::RrfsA, Mucape) => Some((
            NativeThermoSelector {
                discipline: 0,
                category: 7,
                number: 6,
                level_type: 108,
                level_value: Some(25500.0),
            },
            "255-0 mb CAPE",
            ProxyEquivalent,
            false,
            "native most-unstable CAPE proxy",
            if matches!(model, ModelId::Hrrr) {
                "sfc"
            } else if matches!(model, ModelId::RrfsA) {
                "nat-na"
            } else {
                "pgrb2.0p25"
            },
        )),
        (ModelId::Hrrr, Mucin) | (ModelId::Gfs, Mucin) | (ModelId::RrfsA, Mucin) => Some((
            NativeThermoSelector {
                discipline: 0,
                category: 7,
                number: 7,
                level_type: 108,
                level_value: Some(25500.0),
            },
            "255-0 mb CIN",
            ProxyEquivalent,
            false,
            "native most-unstable CIN proxy",
            if matches!(model, ModelId::Hrrr) {
                "sfc"
            } else if matches!(model, ModelId::RrfsA) {
                "nat-na"
            } else {
                "pgrb2.0p25"
            },
        )),
        (ModelId::Hrrr, LiftedIndex) => Some((
            NativeThermoSelector {
                discipline: 0,
                category: 7,
                number: 198,
                level_type: 1,
                level_value: Some(0.0),
            },
            "surface lifted index",
            ExactEquivalent,
            false,
            "HRRR surface LI candidate; compare before promotion",
            "sfc",
        )),
        (ModelId::Gfs, LiftedIndex) => Some((
            NativeThermoSelector {
                discipline: 0,
                category: 7,
                number: 192,
                level_type: 1,
                level_value: Some(0.0),
            },
            "surface lifted index",
            ExactEquivalent,
            false,
            "GFS surface LI candidate; compare before promotion",
            "pgrb2.0p25",
        )),
        (ModelId::RrfsA, LiftedIndex) => Some((
            NativeThermoSelector {
                discipline: 0,
                category: 7,
                number: 10,
                level_type: 100,
                level_value: Some(50000.0),
            },
            "500-1000 mb lifted index",
            ProxyEquivalent,
            false,
            "RRFS 500-1000 mb lifted-index proxy; compare before promotion",
            "nat-na",
        )),
        (ModelId::EcmwfOpenData, Mucape) => Some((
            NativeThermoSelector {
                discipline: 0,
                category: 7,
                number: 6,
                level_type: 17,
                level_value: None,
            },
            "most unstable CAPE",
            ExactEquivalent,
            false,
            "ECMWF open-data MUCAPE candidate; compare before promotion",
            "oper",
        )),
        _ => None,
    }
}

fn build_native_field(
    recipe: NativeThermoRecipe,
    candidate: NativeThermoCandidate,
    message: &Grib2Message,
) -> Result<NativeThermoField, Box<dyn std::error::Error>> {
    let nx = message.grid.nx as usize;
    let ny = message.grid.ny as usize;
    let shape = GridShape::new(nx, ny)?;
    let (mut lat, mut lon) = grid_latlon(&message.grid);
    let mut values = unpack_message(message)?;
    if message.grid.scan_mode & 0x40 != 0 {
        flip_rows(&mut lat, nx, ny);
        flip_rows(&mut lon, nx, ny);
        flip_rows(&mut values, nx, ny);
    }
    normalize_and_rotate_longitude_rows(&mut lat, &mut lon, &mut values, nx, ny);

    let grid = LatLonGrid::new(
        shape,
        lat.into_iter().map(|value| value as f32).collect(),
        lon.into_iter().map(|value| value as f32).collect(),
    )?;

    Ok(NativeThermoField {
        recipe,
        candidate,
        units: parameter_units(
            message.discipline,
            message.product.parameter_category,
            message.product.parameter_number,
        )
        .to_string(),
        grid,
        values,
        parameter_name: parameter_name(
            message.discipline,
            message.product.parameter_category,
            message.product.parameter_number,
        )
        .to_string(),
        level_name: level_name(message.product.level_type).to_string(),
        level_type: message.product.level_type,
        level_value: message.product.level_value,
    })
}

fn tolerance_for_recipe(recipe_slug: &str) -> f64 {
    match recipe_slug {
        "sbcape" | "mlcape" | "mucape" => 50.0,
        "sbcin" | "mlcin" | "mucin" => 25.0,
        "sblcl" => 100.0,
        "lifted_index" => 1.0,
        _ => 10.0,
    }
}

fn summarize(values: &[f64]) -> FieldSummaryStats {
    let mut sorted = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    sorted.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    FieldSummaryStats {
        min: percentile(&sorted, 0.0),
        p01: percentile(&sorted, 0.01),
        p05: percentile(&sorted, 0.05),
        p25: percentile(&sorted, 0.25),
        median: percentile(&sorted, 0.50),
        p75: percentile(&sorted, 0.75),
        p95: percentile(&sorted, 0.95),
        p99: percentile(&sorted, 0.99),
        max: percentile(&sorted, 1.0),
        mean: mean(&sorted),
    }
}

fn percentile(sorted: &[f64], fraction: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let clamped = fraction.clamp(0.0, 1.0);
    let index = ((sorted.len() - 1) as f64 * clamped).round() as usize;
    sorted[index]
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn normalize_longitude(lon: f64) -> f64 {
    if lon > 180.0 { lon - 360.0 } else { lon }
}

fn normalize_and_rotate_longitude_rows(
    lat: &mut [f64],
    lon: &mut [f64],
    values: &mut [f64],
    nx: usize,
    ny: usize,
) {
    if nx == 0 || ny == 0 {
        return;
    }

    for row in 0..ny {
        let start = row * nx;
        let end = start + nx;
        let lat_row = &mut lat[start..end];
        let lon_row = &mut lon[start..end];
        let value_row = &mut values[start..end];

        for lon_value in lon_row.iter_mut() {
            *lon_value = normalize_longitude(*lon_value);
        }

        if let Some(wrap_idx) = lon_row
            .windows(2)
            .position(|pair| pair[1] < pair[0])
            .map(|idx| idx + 1)
        {
            lat_row.rotate_left(wrap_idx);
            lon_row.rotate_left(wrap_idx);
            value_row.rotate_left(wrap_idx);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    fn sample_file(path: &[&str]) -> Option<Vec<u8>> {
        let mut full = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        full.pop();
        full.pop();
        full.push("proof");
        for part in path {
            full.push(part);
        }
        if !full.exists() {
            eprintln!(
                "skipping thermo_native sample test; missing fixture {}",
                full.display()
            );
            return None;
        }
        Some(fs::read(full).expect("sample grib should exist"))
    }

    #[test]
    fn hrrr_surface_cape_and_cin_extract_from_sample() {
        let Some(bytes) = sample_file(&[
            "model_samples_20260416",
            "hrrr",
            "derived",
            "cache",
            "hrrr",
            "20260414",
            "23z",
            "f001",
            "sfc",
            "nomads",
            "full",
            "fetch.grib2",
        ]) else {
            return;
        };
        let sbcape = extract_native_thermo_field(ModelId::Hrrr, NativeThermoRecipe::Sbcape, &bytes)
            .unwrap()
            .unwrap();
        let sbcin = extract_native_thermo_field(ModelId::Hrrr, NativeThermoRecipe::Sbcin, &bytes)
            .unwrap()
            .unwrap();
        let sblcl = extract_native_thermo_field(ModelId::Hrrr, NativeThermoRecipe::Sblcl, &bytes)
            .unwrap()
            .unwrap();
        assert_eq!(sbcape.level_type, 1);
        assert_eq!(sbcin.level_type, 1);
        assert_eq!(sblcl.level_type, 5);
        assert_eq!(
            sbcape.parameter_name,
            "Convective Available Potential Energy"
        );
        assert_eq!(sbcin.parameter_name, "Convective Inhibition");
        assert_eq!(sblcl.parameter_name, "Geopotential Height");
    }

    #[test]
    fn gfs_surface_cape_cin_and_lifted_index_extract_from_sample() {
        let Some(bytes) = sample_file(&[
            "model_samples_20260416",
            "gfs",
            "derived",
            "cache",
            "gfs",
            "20260414",
            "18z",
            "f012",
            "pgrb2_0p25",
            "nomads",
            "full",
            "fetch.grib2",
        ]) else {
            return;
        };
        let sbcape = extract_native_thermo_field(ModelId::Gfs, NativeThermoRecipe::Sbcape, &bytes)
            .unwrap()
            .unwrap();
        let sbcin = extract_native_thermo_field(ModelId::Gfs, NativeThermoRecipe::Sbcin, &bytes)
            .unwrap()
            .unwrap();
        let lifted =
            extract_native_thermo_field(ModelId::Gfs, NativeThermoRecipe::LiftedIndex, &bytes)
                .unwrap()
                .unwrap();
        assert_eq!(sbcape.level_type, 1);
        assert_eq!(sbcin.level_type, 1);
        assert_eq!(lifted.level_type, 1);
        assert_eq!(lifted.level_name, "Ground or Water Surface");
    }

    #[test]
    fn ecmwf_mucape_extracts_from_sample() {
        let Some(bytes) = sample_file(&[
            "model_samples_20260416",
            "ecmwf_open_data",
            "derived",
            "cache",
            "ecmwf_open_data",
            "20260414",
            "12z",
            "f006",
            "oper",
            "ecmwf",
            "full",
            "fetch.grib2",
        ]) else {
            return;
        };
        let mucape =
            extract_native_thermo_field(ModelId::EcmwfOpenData, NativeThermoRecipe::Mucape, &bytes)
                .unwrap()
                .unwrap();
        assert_eq!(
            mucape.parameter_name,
            "Convective Available Potential Energy"
        );
        assert_eq!(mucape.level_type, 17);
    }

    #[test]
    fn compare_stats_report_correlation_and_error() {
        let native = vec![10.0, 20.0, 30.0, 40.0];
        let derived = vec![11.0, 19.0, 31.0, 39.0];
        let stats = compare_native_vs_derived("lifted_index", &native, &derived).unwrap();
        assert_eq!(stats.valid_points, 4);
        assert!(stats.correlation_r > 0.99);
        assert!(stats.mean_abs_diff > 0.0);
    }

    #[test]
    fn compare_stats_separate_pass_review_and_reject_verdicts() {
        let native = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let pass =
            compare_native_vs_derived("lifted_index", &native, &[1.5, 2.5, 3.5, 4.5, 5.5]).unwrap();
        let review =
            compare_native_vs_derived("lifted_index", &native, &[2.25, 3.25, 4.25, 5.25, 6.25])
                .unwrap();
        let reject =
            compare_native_vs_derived("lifted_index", &native, &[5.0, 4.0, 3.0, 2.0, 1.0]).unwrap();

        assert_eq!(pass.verdict, NativeComparisonVerdict::Pass);
        assert_eq!(review.verdict, NativeComparisonVerdict::Review);
        assert_eq!(reject.verdict, NativeComparisonVerdict::Reject);
        assert!(pass.mean_abs_diff < review.mean_abs_diff);
        assert!(reject.correlation_r < 0.0);
    }

    #[test]
    fn compare_stats_ignore_non_finite_points() {
        let native = vec![10.0, f64::NAN, 30.0, f64::INFINITY];
        let derived = vec![10.5, 20.0, 29.5, 40.0];
        let stats = compare_native_vs_derived("lifted_index", &native, &derived).unwrap();

        assert_eq!(stats.valid_points, 2);
        assert_eq!(stats.domain_mean_native, 20.0);
        assert_eq!(stats.domain_mean_derived, 20.0);
        assert_eq!(stats.native.min, 10.0);
        assert_eq!(stats.native.max, 30.0);
    }

    #[test]
    fn rrfs_exposes_all_supported_native_thermo_candidates() {
        let supported = [
            "sbcape",
            "sbcin",
            "sblcl",
            "mlcape",
            "mlcin",
            "mucape",
            "mucin",
            "lifted_index",
        ];
        for slug in supported {
            let candidate = native_candidate_for_slug(ModelId::RrfsA, slug)
                .unwrap_or_else(|| panic!("rrfs native candidate missing for {slug}"));
            assert_eq!(candidate.fetch_product, "nat-na");
        }
    }
}
