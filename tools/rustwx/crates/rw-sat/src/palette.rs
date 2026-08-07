//! Per-band false-color palettes ported as plain data from the old rustwx
//! `satellite/render.rs` anchor tables (the durable science; the old
//! ColorScale / MapRenderRequest plumbing stays behind).
//!
//! Anchors are `(value, rgb)` pairs: reflectance factor (0..1) for the
//! visible/near-IR bands C01-06, brightness temperature in Kelvin for
//! C07-16. Values are clamped to the anchor range; NaN renders transparent.

use crate::composite::{Rgba, TRANSPARENT};

/// Anchor table: `(value, [r, g, b])`, strictly ascending by value.
pub type Anchors = &'static [(f32, [u8; 3])];

/// Grayscale gamma-ish ramp over reflectance 0..1 (C01-06), from the old
/// `grayscale_visible_scale(0.0, 1.0)`.
pub const VISIBLE_REFLECTANCE: Anchors = &[
    (0.00, [3, 5, 8]),
    (0.18, [24, 29, 35]),
    (0.36, [61, 69, 78]),
    (0.56, [118, 128, 138]),
    (0.76, [190, 197, 204]),
    (1.00, [252, 253, 254]),
];

/// C07 shortwave window IR (240..430 K).
pub const SHORTWAVE_WINDOW_IR: Anchors = &[
    (240.0, [5, 9, 18]),
    (270.0, [28, 44, 78]),
    (295.0, [58, 85, 103]),
    (315.0, [98, 104, 88]),
    (335.0, [156, 130, 58]),
    (355.0, [219, 135, 34]),
    (375.0, [224, 61, 31]),
    (395.0, [196, 28, 72]),
    (415.0, [255, 236, 180]),
    (430.0, [255, 255, 255]),
];

/// C08 upper-level water vapor (184..268 K), old `water_vapor_scale` with
/// the upper-channel blue.
pub const UPPER_WATER_VAPOR: Anchors = &[
    (184.0, [252, 253, 255]),
    (196.0, [200, 229, 242]),
    (210.0, [66, 129, 195]),
    (226.0, [68, 80, 151]),
    (240.0, [112, 78, 128]),
    (254.0, [151, 113, 82]),
    (260.0, [90, 80, 65]),
    (268.0, [38, 38, 36]),
];

/// C09 mid-level water vapor (188..276 K).
pub const MID_WATER_VAPOR: Anchors = &[
    (188.0, [252, 253, 255]),
    (200.0, [200, 229, 242]),
    (214.0, [78, 145, 198]),
    (230.0, [68, 80, 151]),
    (244.0, [112, 78, 128]),
    (258.0, [151, 113, 82]),
    (268.0, [90, 80, 65]),
    (276.0, [38, 38, 36]),
];

/// C10 lower-level water vapor (196..286 K).
pub const LOWER_WATER_VAPOR: Anchors = &[
    (196.0, [252, 252, 255]),
    (208.0, [194, 224, 239]),
    (222.0, [97, 158, 205]),
    (236.0, [70, 91, 160]),
    (250.0, [116, 79, 135]),
    (264.0, [159, 117, 84]),
    (276.0, [98, 85, 68]),
    (286.0, [43, 42, 39]),
];

/// C11 cloud-top phase IR (188..325 K).
pub const CLOUD_TOP_IR: Anchors = &[
    (188.0, [255, 255, 255]),
    (202.0, [207, 234, 252]),
    (216.0, [122, 191, 231]),
    (230.0, [76, 124, 190]),
    (244.0, [84, 78, 139]),
    (258.0, [103, 93, 111]),
    (274.0, [95, 95, 95]),
    (292.0, [61, 61, 61]),
    (310.0, [31, 31, 31]),
    (325.0, [8, 8, 8]),
];

/// C12 ozone IR (190..320 K).
pub const OZONE_IR: Anchors = &[
    (190.0, [252, 251, 255]),
    (205.0, [201, 223, 245]),
    (220.0, [129, 169, 218]),
    (235.0, [94, 104, 174]),
    (250.0, [119, 82, 139]),
    (265.0, [153, 112, 91]),
    (282.0, [107, 100, 91]),
    (300.0, [59, 59, 59]),
    (320.0, [12, 12, 12]),
];

/// C13 clean window IR (188..328 K) — the canonical IR look.
pub const CLEAN_WINDOW_IR: Anchors = &[
    (188.0, [255, 255, 255]),
    (202.0, [218, 239, 254]),
    (216.0, [143, 204, 235]),
    (230.0, [83, 146, 202]),
    (244.0, [67, 91, 154]),
    (258.0, [87, 76, 122]),
    (272.0, [99, 95, 102]),
    (288.0, [72, 72, 72]),
    (306.0, [36, 36, 36]),
    (328.0, [4, 4, 4]),
];

/// C14 longwave IR (188..330 K).
pub const LONGWAVE_IR: Anchors = &[
    (188.0, [255, 255, 255]),
    (204.0, [225, 238, 250]),
    (220.0, [157, 196, 222]),
    (236.0, [91, 137, 185]),
    (252.0, [86, 85, 132]),
    (268.0, [102, 96, 101]),
    (286.0, [76, 76, 76]),
    (306.0, [37, 37, 37]),
    (330.0, [5, 5, 5]),
];

/// C15 dirty window IR (188..330 K).
pub const DIRTY_WINDOW_IR: Anchors = &[
    (188.0, [255, 255, 255]),
    (204.0, [224, 237, 248]),
    (220.0, [158, 193, 216]),
    (236.0, [104, 137, 178]),
    (252.0, [96, 86, 132]),
    (268.0, [117, 97, 94]),
    (286.0, [83, 78, 72]),
    (306.0, [39, 39, 37]),
    (330.0, [5, 5, 5]),
];

/// C16 CO2 longwave IR (188..315 K).
pub const CO2_LONGWAVE_IR: Anchors = &[
    (188.0, [255, 255, 255]),
    (202.0, [218, 232, 253]),
    (216.0, [148, 188, 231]),
    (230.0, [91, 120, 194]),
    (244.0, [95, 80, 151]),
    (258.0, [125, 89, 124]),
    (274.0, [96, 91, 88]),
    (294.0, [48, 48, 48]),
    (315.0, [7, 7, 7]),
];

/// The anchor table the old render path assigned to each ABI band.
pub fn band_anchors(channel: u8) -> Anchors {
    match channel {
        1..=6 => VISIBLE_REFLECTANCE,
        7 => SHORTWAVE_WINDOW_IR,
        8 => UPPER_WATER_VAPOR,
        9 => MID_WATER_VAPOR,
        10 => LOWER_WATER_VAPOR,
        11 => CLOUD_TOP_IR,
        12 => OZONE_IR,
        13 => CLEAN_WINDOW_IR,
        14 => LONGWAVE_IR,
        15 => DIRTY_WINDOW_IR,
        16 => CO2_LONGWAVE_IR,
        _ => CLEAN_WINDOW_IR,
    }
}

/// Linear interpolation through an anchor table; values outside the range
/// clamp to the end colors, NaN renders transparent.
pub fn anchor_color(value: f32, anchors: Anchors) -> Rgba {
    if !value.is_finite() || anchors.is_empty() {
        return TRANSPARENT;
    }
    if value <= anchors[0].0 {
        let [r, g, b] = anchors[0].1;
        return [r, g, b, 255];
    }
    for window in anchors.windows(2) {
        let (lo, lo_color) = window[0];
        let (hi, hi_color) = window[1];
        if value <= hi {
            let span = hi - lo;
            let t = if span > 0.0 {
                ((value - lo) / span).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let channel = |a: u8, b: u8| -> u8 {
                (a as f32 + (b as f32 - a as f32) * t)
                    .round()
                    .clamp(0.0, 255.0) as u8
            };
            return [
                channel(lo_color[0], hi_color[0]),
                channel(lo_color[1], hi_color[1]),
                channel(lo_color[2], hi_color[2]),
                255,
            ];
        }
    }
    let [r, g, b] = anchors[anchors.len() - 1].1;
    [r, g, b, 255]
}

/// False-color one band value with its production palette.
pub fn band_color(channel: u8, value: f32) -> Rgba {
    anchor_color(value, band_anchors(channel))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anchor_tables_are_strictly_ascending() {
        for channel in 1..=16u8 {
            let anchors = band_anchors(channel);
            assert!(anchors.len() >= 2, "C{channel:02} needs >= 2 anchors");
            for pair in anchors.windows(2) {
                assert!(
                    pair[0].0 < pair[1].0,
                    "C{channel:02} anchors must ascend: {} then {}",
                    pair[0].0,
                    pair[1].0
                );
            }
        }
    }

    #[test]
    fn band_color_clamps_and_interpolates() {
        // Cold cloud tops render bright white on the clean IR ramp.
        assert_eq!(band_color(13, 150.0), [255, 255, 255, 255]);
        // Hot surface clamps to the dark end.
        assert_eq!(band_color(13, 400.0), [4, 4, 4, 255]);
        // NaN is transparent.
        assert_eq!(band_color(13, f32::NAN), TRANSPARENT);
        // Midpoint of the first clean-IR segment blends the two anchors.
        let mid = band_color(13, 195.0);
        assert!(mid[0] > 218 && mid[0] < 255, "blended red: {}", mid[0]);
        assert_eq!(mid[3], 255);
    }

    #[test]
    fn visible_band_uses_reflectance_ramp() {
        assert_eq!(band_color(2, 0.0), [3, 5, 8, 255]);
        assert_eq!(band_color(2, 1.0), [252, 253, 254, 255]);
        let mid = band_color(2, 0.5);
        assert!(mid[0] > 61 && mid[0] < 190, "mid gray: {}", mid[0]);
    }
}
