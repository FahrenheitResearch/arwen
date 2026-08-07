//! GOES ABI RGB composite recipes — the pure per-pixel core extracted from
//! the old rustwx `satellite/rgb.rs`, decoupled from the map renderer: colors
//! are plain `[u8; 4]` RGBA and the only inputs are per-band CMI values.
//!
//! The recipes are unchanged from the production code. GeoColor here is the
//! DAYTIME pseudo-natural-color recipe (no nighttime IR / city-lights blend
//! like CIRA GeoColor); at night it renders dark/transparent. Use
//! `DayNightCloudMicroCombo` for 24h loops.

use std::collections::HashMap;
use std::error::Error;
use std::io;

use crate::abi::{GoesAbiField, GoesAbiScene};

/// Plain RGBA color, `[r, g, b, a]`.
pub type Rgba = [u8; 4];

/// Fully transparent pixel (off-earth / missing data).
pub const TRANSPARENT: Rgba = [0, 0, 0, 0];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoesAbiRgbCompositeStyle {
    GeoColor,
    FireTemperature,
    AirMass,
    Dust,
    Sandwich,
    DayCloudPhase,
    DayNightCloudMicroCombo,
    NaturalColor,
}

impl GoesAbiRgbCompositeStyle {
    pub const ALL: [GoesAbiRgbCompositeStyle; 8] = [
        Self::GeoColor,
        Self::FireTemperature,
        Self::AirMass,
        Self::Dust,
        Self::Sandwich,
        Self::DayCloudPhase,
        Self::DayNightCloudMicroCombo,
        Self::NaturalColor,
    ];

    pub fn slug(self) -> &'static str {
        match self {
            Self::GeoColor => "geocolor",
            Self::FireTemperature => "fire_temperature",
            Self::AirMass => "airmass",
            Self::Dust => "dust",
            Self::Sandwich => "sandwich",
            Self::DayCloudPhase => "day_cloud_phase",
            Self::DayNightCloudMicroCombo => "day_night_cloud_micro_combo",
            Self::NaturalColor => "natural_color",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        let normalized = value.trim().to_ascii_lowercase().replace('-', "_");
        Self::ALL
            .into_iter()
            .find(|style| style.slug() == normalized)
    }

    pub fn title(self) -> &'static str {
        match self {
            Self::GeoColor => "GeoColor",
            Self::FireTemperature => "Fire Temperature",
            Self::AirMass => "AirMass RGB",
            Self::Dust => "Dust RGB",
            Self::Sandwich => "Sandwich RGB",
            Self::DayCloudPhase => "GOES Day Cloud Phase RGB",
            Self::DayNightCloudMicroCombo => "Day Night Cloud Micro Combo RGB",
            Self::NaturalColor => "GeoColor",
        }
    }

    /// The channel whose fixed grid the composite is rendered on.
    pub fn base_channel(self) -> u8 {
        match self {
            Self::GeoColor => 2,
            Self::FireTemperature => 7,
            Self::AirMass => 8,
            Self::Dust => 13,
            Self::Sandwich => 13,
            Self::DayCloudPhase => 13,
            Self::DayNightCloudMicroCombo => 13,
            Self::NaturalColor => 2,
        }
    }

    pub fn required_channels(self) -> &'static [u8] {
        match self {
            Self::GeoColor => &[1, 2, 3],
            Self::FireTemperature => &[5, 6, 7],
            Self::AirMass => &[8, 10, 12, 13],
            Self::Dust => &[11, 13, 14, 15],
            Self::Sandwich => &[3, 13],
            Self::DayCloudPhase => &[2, 5, 13],
            Self::DayNightCloudMicroCombo => &[2, 5, 7, 13, 15],
            Self::NaturalColor => &[1, 2, 3],
        }
    }
}

/// Compose one pixel of `style` from per-band CMI values supplied by
/// `band_value` (reflectance factor 0..1 for C01-06, brightness temperature
/// Kelvin for C07-16). Identical math to the old production code.
pub fn compose_goes_abi_rgb_pixel<F>(
    style: GoesAbiRgbCompositeStyle,
    mut band_value: F,
) -> Result<Rgba, Box<dyn Error>>
where
    F: FnMut(u8) -> Result<f32, Box<dyn Error>>,
{
    Ok(match style {
        GoesAbiRgbCompositeStyle::GeoColor | GoesAbiRgbCompositeStyle::NaturalColor => {
            let c01 = reflectance_pct(band_value(1)?);
            let c02 = reflectance_pct(band_value(2)?);
            let c03 = reflectance_pct(band_value(3)?);
            let r = visible_component(c02);
            let g = visible_component(0.45 * c02 + 0.10 * c03 + 0.45 * c01);
            let b = visible_component(c01);
            color_or_transparent([r, g, b])
        }
        GoesAbiRgbCompositeStyle::FireTemperature => {
            let r = component(k_to_c(band_value(7)?), 0.0, 60.0, 0.4);
            let g = component(reflectance_pct(band_value(6)?), 0.0, 100.0, 1.0);
            let b = component(reflectance_pct(band_value(5)?), 0.0, 75.0, 1.0);
            color_or_transparent([r, g, b])
        }
        GoesAbiRgbCompositeStyle::AirMass => {
            let c08 = band_value(8)?;
            let c10 = band_value(10)?;
            let c12 = band_value(12)?;
            let c13 = band_value(13)?;
            let r = component((c08 - c10) as f64, -26.2, 0.6, 1.0);
            let g = component((c12 - c13) as f64, -43.2, 6.7, 1.0);
            let b = component(k_to_c(c08), -29.25, -64.65, 1.0);
            color_or_transparent([r, g, b])
        }
        GoesAbiRgbCompositeStyle::Dust => {
            let c11 = band_value(11)?;
            let c13 = band_value(13)?;
            let c14 = band_value(14)?;
            let c15 = band_value(15)?;
            let r = component((c15 - c13) as f64, -6.7, 2.6, 1.0);
            let g = component((c14 - c11) as f64, -0.5, 20.0, 2.5);
            let b = component(k_to_c(c13), -11.95, 15.55, 1.0);
            color_or_transparent([r, g, b])
        }
        GoesAbiRgbCompositeStyle::Sandwich => {
            let visible = component(reflectance_pct(band_value(3)?), 0.0, 95.0, 1.0);
            let ir_cold = normalized(k_to_c(band_value(13)?), 30.0, -70.0, 1.0);
            sandwich_color(visible, ir_cold)
        }
        GoesAbiRgbCompositeStyle::DayCloudPhase => {
            let r = component(k_to_c(band_value(13)?), 7.5, -53.5, 1.0);
            let g = component(reflectance_pct(band_value(2)?), 0.0, 78.0, 1.0);
            let b = component(reflectance_pct(band_value(5)?), 1.0, 59.0, 1.0);
            color_or_transparent([r, g, b])
        }
        GoesAbiRgbCompositeStyle::DayNightCloudMicroCombo => {
            let day_green = reflectance_pct(band_value(2)?);
            let day_blue = reflectance_pct(band_value(5)?);
            let c07 = band_value(7)?;
            let c13 = band_value(13)?;
            let c15 = band_value(15)?;
            let daylight = normalized(day_green, 0.0, 18.0, 1.0).unwrap_or(0.0);
            let r = component(k_to_c(c13), 12.0, -60.0, 1.0);
            let g_day = normalized(day_green, 0.0, 80.0, 1.0).unwrap_or(0.0);
            let b_day = normalized(day_blue, 0.0, 65.0, 1.0).unwrap_or(0.0);
            let g_night = normalized((c15 - c13) as f64, -5.0, 12.0, 1.0).unwrap_or(0.0);
            let b_night = normalized(k_to_c(c07), 30.0, -45.0, 1.0).unwrap_or(0.0);
            let g = Some(((g_night * (1.0 - daylight) + g_day * daylight) * 255.0).round() as u8);
            let b = Some(((b_night * (1.0 - daylight) + b_day * daylight) * 255.0).round() as u8);
            color_or_transparent([r, g, b])
        }
    })
}

/// Compose a full plane of pixels from full-grid band planes (all on the
/// same fixed grid, e.g. after [`values_on_base_grid`]).
pub fn compose_rgb_pixels(
    style: GoesAbiRgbCompositeStyle,
    bands: &HashMap<u8, Vec<f32>>,
    len: usize,
) -> Result<Vec<Rgba>, Box<dyn Error>> {
    let mut pixels = Vec::with_capacity(len);
    for idx in 0..len {
        pixels.push(compose_goes_abi_rgb_pixel(style, |channel| {
            band_value(bands, channel, idx)
        })?);
    }
    Ok(pixels)
}

fn band_value(
    bands: &HashMap<u8, Vec<f32>>,
    channel: u8,
    idx: usize,
) -> Result<f32, Box<dyn Error>> {
    bands
        .get(&channel)
        .and_then(|values| values.get(idx).copied())
        .ok_or_else(|| boxed_error(format!("missing C{channel:02} value at index {idx}")))
}

/// Resample `field` onto `base_scene`'s fixed grid (bilinear in scan-angle
/// space; identity when the grids already match). This is how 0.5/1 km bands
/// land on a 2 km composite base grid and vice versa.
pub fn values_on_base_grid(
    field: &GoesAbiField,
    base_scene: &GoesAbiScene,
) -> Result<Vec<f32>, Box<dyn Error>> {
    if same_fixed_grid(&field.scene, base_scene) {
        return Ok(field.values.clone());
    }
    resample_fixed_grid_to_scene(field, base_scene)
}

fn same_fixed_grid(left: &GoesAbiScene, right: &GoesAbiScene) -> bool {
    left.fixed_grid.nx == right.fixed_grid.nx
        && left.fixed_grid.ny == right.fixed_grid.ny
        && left.fixed_grid.x_scan_rad.len() == right.fixed_grid.x_scan_rad.len()
        && left.fixed_grid.y_scan_rad.len() == right.fixed_grid.y_scan_rad.len()
        && left
            .fixed_grid
            .x_scan_rad
            .iter()
            .zip(right.fixed_grid.x_scan_rad.iter())
            .all(|(a, b)| (a - b).abs() <= 1.0e-12)
        && left
            .fixed_grid
            .y_scan_rad
            .iter()
            .zip(right.fixed_grid.y_scan_rad.iter())
            .all(|(a, b)| (a - b).abs() <= 1.0e-12)
}

fn resample_fixed_grid_to_scene(
    field: &GoesAbiField,
    base_scene: &GoesAbiScene,
) -> Result<Vec<f32>, Box<dyn Error>> {
    let source = &field.scene.fixed_grid;
    let target = &base_scene.fixed_grid;
    let x_map = build_axis_resample(&source.x_scan_rad, &target.x_scan_rad);
    let y_map = build_axis_resample(&source.y_scan_rad, &target.y_scan_rad);
    let mut out = vec![f32::NAN; target.nx.saturating_mul(target.ny)];

    for (j, y_bracket) in y_map.iter().enumerate() {
        let Some((j0, j1, fy)) = *y_bracket else {
            continue;
        };
        for (i, x_bracket) in x_map.iter().enumerate() {
            let Some((i0, i1, fx)) = *x_bracket else {
                continue;
            };
            let idx = |yy: usize, xx: usize| yy * source.nx + xx;
            out[j * target.nx + i] = bilinear_f32(
                field.values[idx(j0, i0)],
                field.values[idx(j0, i1)],
                field.values[idx(j1, i0)],
                field.values[idx(j1, i1)],
                fx,
                fy,
            );
        }
    }
    Ok(out)
}

fn build_axis_resample(source: &[f64], target: &[f64]) -> Vec<Option<(usize, usize, f32)>> {
    target
        .iter()
        .map(|&value| bracket_axis(source, value))
        .collect()
}

/// Bracket `value` between two axis samples (binary search; the axis may be
/// ascending or descending, as GOES y scan axes are). Returns
/// `(lo, hi, t)` with `t` the interpolation weight toward `hi`.
pub fn bracket_axis(axis: &[f64], value: f64) -> Option<(usize, usize, f32)> {
    if axis.is_empty() || !value.is_finite() {
        return None;
    }
    if axis.len() == 1 {
        return ((value - axis[0]).abs() <= 1.0e-10).then_some((0, 0, 0.0));
    }
    let ascending = axis[axis.len() - 1] >= axis[0];
    let first = axis[0];
    let last = axis[axis.len() - 1];
    if ascending {
        if value < first || value > last {
            return None;
        }
    } else if value > first || value < last {
        return None;
    }

    let mut lo = 0usize;
    let mut hi = axis.len() - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        let mid_value = axis[mid];
        if (ascending && mid_value <= value) || (!ascending && mid_value >= value) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let a = axis[lo];
    let b = axis[hi];
    let t = if (b - a).abs() <= 1.0e-15 {
        0.0
    } else {
        ((value - a) / (b - a)).clamp(0.0, 1.0)
    };
    Some((lo, hi, t as f32))
}

/// Bilinear blend that degrades to the first finite corner when any corner
/// is NaN (limb pixels), and NaN when every corner is.
pub fn bilinear_f32(v00: f32, v10: f32, v01: f32, v11: f32, fx: f32, fy: f32) -> f32 {
    if v00.is_finite() && v10.is_finite() && v01.is_finite() && v11.is_finite() {
        let south = v00 * (1.0 - fx) + v10 * fx;
        let north = v01 * (1.0 - fx) + v11 * fx;
        south * (1.0 - fy) + north * fy
    } else {
        [v00, v10, v01, v11]
            .into_iter()
            .find(|value| value.is_finite())
            .unwrap_or(f32::NAN)
    }
}

fn k_to_c(value: f32) -> f64 {
    value as f64 - 273.15
}

fn reflectance_pct(value: f32) -> f64 {
    value as f64 * 100.0
}

fn visible_component(value_pct: f64) -> Option<u8> {
    component(value_pct, 0.0, 100.0, 2.2)
}

fn component(value: f64, min: f64, max: f64, gamma: f64) -> Option<u8> {
    normalized(value, min, max, gamma).map(|value| (value * 255.0).round() as u8)
}

fn normalized(value: f64, min: f64, max: f64, gamma: f64) -> Option<f64> {
    if !value.is_finite() || !min.is_finite() || !max.is_finite() || (max - min).abs() <= 1.0e-12 {
        return None;
    }
    let raw = if max >= min {
        (value - min) / (max - min)
    } else {
        (min - value) / (min - max)
    };
    let scaled = raw.clamp(0.0, 1.0);
    let gamma = gamma.max(1.0e-6);
    Some(scaled.powf(1.0 / gamma))
}

fn color_or_transparent(channels: [Option<u8>; 3]) -> Rgba {
    match channels {
        [Some(r), Some(g), Some(b)] => [r, g, b, 255],
        _ => TRANSPARENT,
    }
}

fn sandwich_color(visible: Option<u8>, ir_cold: Option<f64>) -> Rgba {
    let (Some(gray), Some(cold)) = (visible, ir_cold) else {
        return TRANSPARENT;
    };
    let ir = cold_cloud_tint(cold);
    let weight = (cold * 0.65).clamp(0.0, 0.65);
    [
        blend_u8(gray, ir[0], weight),
        blend_u8(gray, ir[1], weight),
        blend_u8(gray, ir[2], weight),
        255,
    ]
}

fn cold_cloud_tint(t: f64) -> Rgba {
    let t = t.clamp(0.0, 1.0);
    if t < 0.33 {
        let local = t / 0.33;
        return lerp_color([70, 78, 92, 255], [135, 105, 78, 255], local);
    }
    if t < 0.66 {
        let local = (t - 0.33) / 0.33;
        return lerp_color([135, 105, 78, 255], [220, 215, 184, 255], local);
    }
    let local = (t - 0.66) / 0.34;
    lerp_color([220, 215, 184, 255], [235, 250, 255, 255], local)
}

fn blend_u8(base: u8, top: u8, weight: f64) -> u8 {
    (base as f64 * (1.0 - weight) + top as f64 * weight)
        .round()
        .clamp(0.0, 255.0) as u8
}

fn lerp_color(left: Rgba, right: Rgba, t: f64) -> Rgba {
    [
        blend_u8(left[0], right[0], t),
        blend_u8(left[1], right[1], t),
        blend_u8(left[2], right[2], t),
        blend_u8(left[3], right[3], t),
    ]
}

fn boxed_error(message: impl Into<String>) -> Box<dyn Error> {
    Box::new(io::Error::new(io::ErrorKind::InvalidData, message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgb_styles_advertise_required_channels() {
        assert_eq!(
            GoesAbiRgbCompositeStyle::FireTemperature.required_channels(),
            &[5, 6, 7]
        );
        assert_eq!(
            GoesAbiRgbCompositeStyle::GeoColor.required_channels(),
            &[1, 2, 3]
        );
        assert_eq!(
            GoesAbiRgbCompositeStyle::Dust.required_channels(),
            &[11, 13, 14, 15]
        );
        assert_eq!(
            GoesAbiRgbCompositeStyle::DayNightCloudMicroCombo.required_channels(),
            &[2, 5, 7, 13, 15]
        );
        assert_eq!(GoesAbiRgbCompositeStyle::AirMass.base_channel(), 8);
    }

    #[test]
    fn style_slugs_round_trip_through_parse() {
        for style in GoesAbiRgbCompositeStyle::ALL {
            assert_eq!(GoesAbiRgbCompositeStyle::parse(style.slug()), Some(style));
        }
        assert_eq!(
            GoesAbiRgbCompositeStyle::parse("Fire-Temperature"),
            Some(GoesAbiRgbCompositeStyle::FireTemperature)
        );
        assert_eq!(GoesAbiRgbCompositeStyle::parse("nope"), None);
    }

    #[test]
    fn reversed_component_range_maps_cold_values_high() {
        let warm = component(7.5, 7.5, -53.5, 1.0).unwrap();
        let cold = component(-53.5, 7.5, -53.5, 1.0).unwrap();
        assert!(cold > warm);
    }

    #[test]
    fn resample_bracket_supports_descending_axis() {
        let axis = [3.0, 2.0, 1.0, 0.0];
        let (lo, hi, t) = bracket_axis(&axis, 1.5).unwrap();
        assert_eq!((lo, hi), (1, 2));
        assert!((t - 0.5).abs() < 1.0e-6);
    }

    #[test]
    fn fire_temperature_rgb_turns_hot_pixels_bright() {
        let mut bands = HashMap::<u8, Vec<f32>>::new();
        bands.insert(7, vec![273.15 + 60.0]);
        bands.insert(6, vec![1.0]);
        bands.insert(5, vec![0.75]);
        let pixels = compose_rgb_pixels(GoesAbiRgbCompositeStyle::FireTemperature, &bands, 1)
            .expect("fire rgb");
        assert_eq!(pixels[0], [255, 255, 255, 255]);
    }

    #[test]
    fn nan_band_values_compose_transparent() {
        let mut bands = HashMap::<u8, Vec<f32>>::new();
        for channel in [1u8, 2, 3] {
            bands.insert(channel, vec![f32::NAN]);
        }
        let pixels =
            compose_rgb_pixels(GoesAbiRgbCompositeStyle::GeoColor, &bands, 1).expect("geocolor");
        assert_eq!(pixels[0], TRANSPARENT);
    }
}
