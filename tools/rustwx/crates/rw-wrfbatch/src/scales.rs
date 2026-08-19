//! Colour scales for the products that are NOT a model field.
//!
//! An ensemble mean and an ensemble PMM are the field itself and must
//! therefore wear the field's own operational ladder -- reflectivity on the
//! NWS dBZ steps, precipitation on the QPF steps -- which
//! `rustwx_products::viewer::operational_style_for_store_variable` already
//! resolves from the stored variable's selector.  This module holds only
//! the scales for quantities the catalog has no ladder for, because they
//! are not weather variables:
//!
//! * **spread** -- a standard deviation.  Deliberately never the field's
//!   own ladder: a 6 dBZ spread painted on the NWS reflectivity table
//!   reads as "light rain", and a reader who glances at two panels drawn
//!   in the same colours will compare them as if they were the same
//!   quantity.
//! * **probability** -- a fraction in `[0, 1]`, on 10% steps with the
//!   zero bin left transparent so the panel shows where the ensemble said
//!   something rather than where it did not.
//! * **counts** -- small integer tallies (radar overlap, observed levels
//!   per column), where a continuous ramp implies a precision the integers
//!   do not have.

use rustwx_render::{Color, ColorScale, DiscreteColorScale, ExtendMode};

/// `magma`, 9 stops, dark for no spread through bright for a lot.
const SPREAD_COLORS: [[u8; 3]; 9] = [
    [0, 0, 4],
    [28, 16, 68],
    [79, 18, 123],
    [129, 37, 129],
    [181, 54, 122],
    [229, 80, 100],
    [251, 135, 97],
    [254, 194, 135],
    [252, 253, 191],
];

/// `plasma`-like, 10 stops for the 10 probability bins.
const PROBABILITY_COLORS: [[u8; 3]; 10] = [
    [13, 8, 135],
    [70, 3, 159],
    [114, 1, 168],
    [156, 23, 158],
    [189, 55, 134],
    [216, 87, 107],
    [237, 121, 83],
    [251, 159, 58],
    [253, 202, 38],
    [240, 249, 33],
];

/// The exceedance-probability ladder: 10% steps, zero transparent.
///
/// `mask_below` is what keeps an all-zero region out of the picture: a
/// solid dark wash over every point no member reached would say the
/// ensemble had an opinion there.
pub fn probability_scale() -> ColorScale {
    let levels: Vec<f64> = (0..=10).map(|step| f64::from(step) / 10.0).collect();
    ColorScale::Discrete(DiscreteColorScale {
        levels,
        colors: PROBABILITY_COLORS
            .iter()
            .map(|[r, g, b]| Color::rgba(*r, *g, *b, 255))
            .collect(),
        extend: ExtendMode::Neither,
        mask_below: Some(0.1),
    })
}

/// Ensemble spread, `0 .. upper` in nine steps.
///
/// `upper` is the caller's, not the data's: an auto-ranged spread panel
/// cannot be compared with the next valid time's, and the point of a
/// spread sequence is exactly that comparison.
pub fn spread_scale(upper: f64) -> ColorScale {
    let upper = if upper.is_finite() && upper > 0.0 {
        upper
    } else {
        1.0
    };
    let steps = SPREAD_COLORS.len();
    let levels: Vec<f64> = (0..=steps)
        .map(|index| upper * index as f64 / steps as f64)
        .collect();
    ColorScale::Discrete(DiscreteColorScale {
        levels,
        colors: SPREAD_COLORS
            .iter()
            .map(|[r, g, b]| Color::rgba(*r, *g, *b, 255))
            .collect(),
        extend: ExtendMode::Max,
        mask_below: None,
    })
}

/// A discrete count ladder `1, 2, ... n, n+` with zero transparent.
///
/// Used for the observation grids whose value IS a tally -- how many
/// radars see a cell, how many levels a column has -- where a smooth ramp
/// would invite reading 2.5 radars off the colourbar.
pub fn count_scale(max_count: usize) -> ColorScale {
    let max_count = max_count.clamp(1, 12);
    let levels: Vec<f64> = (0..=max_count + 1).map(|step| step as f64).collect();
    let colors: Vec<Color> = (0..=max_count)
        .map(|index| {
            let fraction = if max_count == 0 {
                0.0
            } else {
                index as f64 / max_count as f64
            };
            ramp(&SPREAD_COLORS, fraction)
        })
        .collect();
    ColorScale::Discrete(DiscreteColorScale {
        levels,
        colors,
        extend: ExtendMode::Max,
        mask_below: Some(1.0),
    })
}

/// Radial velocity: a symmetric diverging scale about zero.
///
/// Symmetric on purpose -- an inbound/outbound couplet is the signal, and
/// a scale whose zero is not the colour break hides it.
pub fn radial_velocity_scale(half_range: f64) -> ColorScale {
    const COLORS: [[u8; 3]; 10] = [
        [5, 48, 97],
        [33, 102, 172],
        [67, 147, 195],
        [146, 197, 222],
        [209, 229, 240],
        [253, 219, 199],
        [244, 165, 130],
        [214, 96, 77],
        [178, 24, 43],
        [103, 0, 31],
    ];
    let half_range = if half_range.is_finite() && half_range > 0.0 {
        half_range
    } else {
        30.0
    };
    let steps = COLORS.len();
    let levels: Vec<f64> = (0..=steps)
        .map(|index| -half_range + 2.0 * half_range * index as f64 / steps as f64)
        .collect();
    ColorScale::Discrete(DiscreteColorScale {
        levels,
        colors: COLORS
            .iter()
            .map(|[r, g, b]| Color::rgba(*r, *g, *b, 255))
            .collect(),
        extend: ExtendMode::Both,
        mask_below: None,
    })
}

/// The NWS composite-reflectivity ladder, sub-5 dBZ transparent.
///
/// This is the fallback for an observation grid, which has no stored model
/// variable and therefore no selector for
/// `operational_style_for_store_variable` to resolve.  The steps are the
/// 5 dBZ NWS breaks both matplotlib render modules in this repository
/// already draw, so an observed panel and a forecast panel are read off the
/// same colours.
pub fn reflectivity_scale() -> ColorScale {
    const BREAKS: [f64; 15] = [
        5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0,
    ];
    const COLORS: [[u8; 3]; 14] = [
        [4, 233, 231],
        [1, 159, 244],
        [3, 0, 244],
        [2, 253, 2],
        [1, 197, 1],
        [0, 142, 0],
        [253, 248, 2],
        [229, 188, 0],
        [253, 149, 0],
        [253, 0, 0],
        [212, 0, 0],
        [188, 0, 0],
        [248, 0, 253],
        [152, 84, 198],
    ];
    ColorScale::Discrete(DiscreteColorScale {
        levels: BREAKS.to_vec(),
        colors: COLORS
            .iter()
            .map(|[r, g, b]| Color::rgba(*r, *g, *b, 255))
            .collect(),
        extend: ExtendMode::Max,
        mask_below: Some(5.0),
    })
}

fn ramp(stops: &[[u8; 3]], fraction: f64) -> Color {
    if stops.is_empty() {
        return Color::BLACK;
    }
    let position = fraction.clamp(0.0, 1.0) * (stops.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = (lower + 1).min(stops.len() - 1);
    let weight = position - lower as f64;
    let blend = |a: u8, b: u8| (f64::from(a) + (f64::from(b) - f64::from(a)) * weight).round() as u8;
    Color::rgba(
        blend(stops[lower][0], stops[upper][0]),
        blend(stops[lower][1], stops[upper][1]),
        blend(stops[lower][2], stops[upper][2]),
        255,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn discrete(scale: &ColorScale) -> DiscreteColorScale {
        match scale {
            ColorScale::Discrete(scale) => scale.clone(),
            ColorScale::Weather(preset) => preset.scale(),
        }
    }

    /// Every discrete scale must have exactly one colour per interval, or
    /// the renderer silently drops the top bin.
    fn levels_and_colors_agree(scale: &ColorScale, what: &str) {
        let scale = discrete(scale);
        assert_eq!(
            scale.levels.len(),
            scale.colors.len() + 1,
            "{what}: {} level(s) do not bound {} colour(s)",
            scale.levels.len(),
            scale.colors.len()
        );
        assert!(
            scale.levels.windows(2).all(|pair| pair[1] > pair[0]),
            "{what}: levels are not strictly ascending"
        );
    }

    #[test]
    fn every_scale_is_well_formed() {
        levels_and_colors_agree(&probability_scale(), "probability");
        levels_and_colors_agree(&spread_scale(12.0), "spread");
        levels_and_colors_agree(&count_scale(4), "count");
        levels_and_colors_agree(&radial_velocity_scale(30.0), "radial velocity");
        levels_and_colors_agree(&reflectivity_scale(), "reflectivity");
    }

    #[test]
    fn zero_probability_is_transparent_not_a_dark_wash() {
        assert_eq!(discrete(&probability_scale()).mask_below, Some(0.1));
    }

    #[test]
    fn sub_five_dbz_is_transparent_as_both_render_modules_draw_it() {
        let scale = discrete(&reflectivity_scale());
        assert_eq!(scale.mask_below, Some(5.0));
        assert_eq!(scale.levels[0], 5.0);
    }

    #[test]
    fn a_degenerate_upper_bound_does_not_produce_a_collapsed_scale() {
        for upper in [0.0, -3.0, f64::NAN, f64::INFINITY] {
            levels_and_colors_agree(&spread_scale(upper), "spread");
        }
        for half in [0.0, -1.0, f64::NAN] {
            levels_and_colors_agree(&radial_velocity_scale(half), "vr");
        }
    }

    #[test]
    fn radial_velocity_breaks_exactly_at_zero() {
        let scale = discrete(&radial_velocity_scale(30.0));
        assert!(scale.levels.iter().any(|level| level.abs() < 1.0e-9));
    }

    #[test]
    fn a_count_ladder_is_one_bin_per_integer() {
        let scale = discrete(&count_scale(4));
        assert_eq!(scale.levels, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(scale.mask_below, Some(1.0), "zero radars is not a colour");
        // A request for more bins than the ladder has is clamped, never
        // silently truncated into a mismatched pair.
        levels_and_colors_agree(&count_scale(50), "count");
    }
}
