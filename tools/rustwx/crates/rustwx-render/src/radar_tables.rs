//! Colour tables for radar moments: reflectivity, radial velocity and the
//! dual-polarization moments (ZDR, CC, KDP, PHIDP).
//!
//! Every stop below is transcribed value for value from the owner's radar
//! application, BowEcho, file `crates/color_tables/src/lib.rs` at commit
//! `66ceb9c4aea5da2bd857d3178f08a13c4c08a8d0`. The provenance file
//! `tools/rustwx/vendor/RADAR-COLOR-TABLES.md` names the BowEcho constant
//! or function each table comes from. Sampling reproduces BowEcho's own:
//! [`Sampling::Interval`] is its `.pal` interval mode (each row ramps to
//! its own end colour, or to the next row's colour when it has none) and
//! [`Sampling::Linear`] is its interpolated mode. Arithmetic is `f32` with
//! the same rounding, so a value samples to the byte BowEcho draws.
//!
//! A scale here is a [`DiscreteColorScale`] whose bins are narrow enough
//! for the ramps to read as ramps; each bin takes the table's colour at
//! the bin centre.

use crate::request::{Color, DiscreteColorScale, ExtendMode};

/// One table row: a value in the table's own units, its colour and, for an
/// interval row, the colour its interval ramps to.
#[derive(Debug, Clone, Copy)]
struct Stop {
    value: f32,
    rgba: [u8; 4],
    end: Option<[u8; 4]>,
}

const fn row(value: f32, r: u8, g: u8, b: u8) -> Stop {
    Stop {
        value,
        rgba: [r, g, b, 255],
        end: None,
    }
}

const fn ramp(value: f32, r: u8, g: u8, b: u8, r2: u8, g2: u8, b2: u8) -> Stop {
    Stop {
        value,
        rgba: [r, g, b, 255],
        end: Some([r2, g2, b2, 255]),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Sampling {
    /// `.pal` interval semantics: a row ramps from its colour to its end
    /// colour (or the next row's colour) across its own interval.
    Interval,
    /// Linear interpolation between neighbouring rows.
    Linear,
}

struct Table {
    /// Ascending rows.
    stops: &'static [Stop],
    sampling: Sampling,
    /// Table units per display unit (the `.pal` `Scale:` header); the rows'
    /// values are divided by it, exactly as BowEcho's parser does.
    units_per_display_unit: f32,
}

/// Reflectivity, dBZ. BowEcho's AWIPS reflectivity preset
/// (the `.pal` table at line 2637 of that file), an interval table.
/// The -30 dBZ row's start colour is transparent there; below the display
/// floor of every scale built here, so it never reaches a picture.
const REFLECTIVITY_ROWS: &[Stop] = &[
    Stop {
        value: -30.0,
        rgba: [116, 78, 173, 0],
        end: Some([147, 141, 117, 255]),
    },
    ramp(-20.0, 150, 145, 83, 210, 212, 180),
    ramp(-10.0, 204, 207, 180, 65, 91, 158),
    ramp(10.0, 67, 97, 162, 106, 208, 228),
    ramp(18.0, 111, 214, 232, 53, 213, 91),
    ramp(22.0, 17, 213, 24, 9, 94, 9),
    ramp(35.0, 29, 104, 9, 234, 210, 4),
    ramp(40.0, 255, 226, 0, 255, 128, 0),
    ramp(50.0, 255, 0, 0, 113, 0, 0),
    ramp(60.0, 255, 255, 255, 255, 146, 255),
    ramp(65.0, 255, 117, 255, 225, 11, 227),
    ramp(70.0, 178, 0, 255, 99, 0, 214),
    ramp(75.0, 5, 236, 240, 1, 32, 32),
    row(85.0, 1, 32, 32),
    row(95.0, 1, 32, 32),
];

/// Radial velocity, knots in the table (`Scale: 1.9426`, so a row's m/s
/// value is its knot value divided by 1.9426). BowEcho's green-red velocity
/// `.pal` preset (the table at line 2875 of that file): dark to bright greens
/// toward the radar, dark to bright reds away, a narrow dark grey band
/// either side of zero, pale and violet extremes.
const VELOCITY_ROWS: &[Stop] = &[
    row(-200.0, 255, 220, 220),
    row(-140.0, 255, 20, 180),
    ramp(-120.0, 250, 4, 130, 114, 3, 141),
    ramp(-100.0, 105, 2, 142, 32, 1, 141),
    ramp(-90.0, 25, 1, 142, 47, 215, 225),
    ramp(-70.0, 55, 226, 229, 172, 239, 242),
    ramp(-50.0, 180, 240, 243, 33, 253, 50),
    ramp(-40.0, 10, 248, 35, 15, 99, 20),
    ramp(-10.0, 72, 112, 71, 106, 125, 105),
    ramp(0.0, 130, 106, 120, 122, 48, 57),
    ramp(10.0, 105, 0, 0, 242, 1, 6),
    ramp(40.0, 249, 58, 84, 255, 142, 212),
    ramp(55.0, 255, 157, 206, 255, 221, 176),
    ramp(60.0, 255, 230, 169, 255, 151, 86),
    row(80.0, 254, 137, 80),
    row(120.0, 97, 6, 2),
    row(140.0, 60, 0, 0),
    row(200.0, 45, 0, 0),
];

/// Differential reflectivity, dB. BowEcho's default ZDR table
/// (`builtin_differential_reflectivity_table`): grey at 0 dB, cool
/// negatives, warm positives, resolution packed into 0..4 dB.
const ZDR_ROWS: &[Stop] = &[
    row(-4.0, 60, 30, 96),
    row(-2.0, 56, 70, 168),
    row(-0.5, 96, 150, 196),
    row(0.0, 140, 140, 140),
    row(0.5, 150, 168, 120),
    row(1.0, 120, 192, 88),
    row(2.0, 224, 220, 60),
    row(3.0, 245, 158, 32),
    row(4.0, 226, 52, 40),
    row(5.5, 176, 28, 92),
    row(7.0, 206, 86, 200),
    row(8.0, 240, 200, 240),
];

/// Correlation coefficient (rhoHV), unitless. BowEcho's default CC table
/// (`builtin_correlation_coefficient_table`): cool non-meteorological lows,
/// warm precipitation above 0.95, near white at 1.
const CC_ROWS: &[Stop] = &[
    row(0.20, 48, 48, 56),
    row(0.45, 72, 60, 150),
    row(0.65, 46, 96, 200),
    row(0.80, 0, 168, 196),
    row(0.88, 64, 196, 92),
    row(0.92, 208, 216, 52),
    row(0.95, 245, 158, 32),
    row(0.97, 226, 46, 40),
    row(0.99, 150, 22, 30),
    row(1.00, 236, 236, 244),
    row(1.05, 255, 255, 255),
];

/// Specific differential phase, deg/km. BowEcho's default KDP table
/// (`builtin_specific_differential_phase_table`): dark near zero, cool
/// negatives, positive KDP warming green to red.
const KDP_ROWS: &[Stop] = &[
    row(-1.0, 70, 96, 170),
    row(-0.3, 90, 110, 140),
    row(0.0, 60, 64, 70),
    row(0.3, 70, 120, 80),
    row(0.75, 90, 180, 70),
    row(1.5, 210, 210, 50),
    row(2.5, 244, 158, 32),
    row(4.0, 230, 70, 44),
    row(7.0, 180, 30, 96),
];

/// Differential phase, degrees. BowEcho's default PHIDP table
/// (`builtin_differential_phase_table`): a monotonic ramp over 0..360.
const PHIDP_ROWS: &[Stop] = &[
    row(0.0, 40, 44, 78),
    row(30.0, 36, 96, 180),
    row(60.0, 0, 158, 170),
    row(90.0, 70, 184, 70),
    row(120.0, 210, 206, 50),
    row(150.0, 240, 150, 32),
    row(180.0, 226, 60, 44),
    row(270.0, 170, 40, 110),
    row(360.0, 232, 200, 230),
];

/// Which tables the reflectivity and radial velocity products draw with.
///
/// `Standard` is the radar tables in this module. `Classic` is the pair of
/// scales those products wore before them (`reflectivity_classic` and
/// `radial_velocity_classic`). The dual-polarization moments have one table
/// each, so the set does not change them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RadarColorSet {
    #[default]
    Standard,
    Classic,
}

/// The environment spelling of the set, for every Rust door that draws a
/// radar product (`rw_wrfbatch --radar-colors NAME` outranks it).
pub const RADAR_COLORS_ENV: &str = "RUSTWX_RADAR_COLORS";

impl RadarColorSet {
    pub const NAMES: [&'static str; 2] = ["standard", "classic"];

    pub fn name(self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::Classic => "classic",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "standard" => Some(Self::Standard),
            "classic" => Some(Self::Classic),
            _ => None,
        }
    }

    /// `from_name`, or a refusal that lists the names.
    pub fn parse(name: &str) -> Result<Self, String> {
        Self::from_name(name).ok_or_else(|| {
            format!(
                "radar colours must be one of {:?}, got {name:?}",
                Self::NAMES
            )
        })
    }
}

static ACTIVE_COLOR_SET: std::sync::OnceLock<RadarColorSet> = std::sync::OnceLock::new();

/// Select the set for this process, before the first radar product is
/// drawn. A second, different selection is refused: one process draws one
/// look.
pub fn install_radar_color_set(set: RadarColorSet) -> Result<(), String> {
    let active = *ACTIVE_COLOR_SET.get_or_init(|| set);
    if active == set {
        Ok(())
    } else {
        Err(format!(
            "radar colours are already {}; one process draws one set",
            active.name()
        ))
    }
}

/// The set radar products draw with: what `install_radar_color_set`
/// selected, else `RUSTWX_RADAR_COLORS`, else `Standard`. An unknown
/// environment value is refused by `radar_color_set_from_env`, which the
/// binaries call at start; here it reads as `Standard`.
pub fn active_radar_color_set() -> RadarColorSet {
    *ACTIVE_COLOR_SET.get_or_init(|| radar_color_set_from_env().unwrap_or_default())
}

/// The set named by `RUSTWX_RADAR_COLORS`: `Standard` when it is unset or
/// blank, a refusal when it names no set.
pub fn radar_color_set_from_env() -> Result<RadarColorSet, String> {
    match std::env::var(RADAR_COLORS_ENV) {
        Ok(value) if !value.trim().is_empty() => RadarColorSet::parse(value.trim())
            .map_err(|error| format!("{RADAR_COLORS_ENV}: {error}")),
        _ => Ok(RadarColorSet::Standard),
    }
}

/// The radar moment tables, by what they colour.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RadarTable {
    Reflectivity,
    RadialVelocity,
    DifferentialReflectivity,
    CorrelationCoefficient,
    SpecificDifferentialPhase,
    DifferentialPhase,
}

impl RadarTable {
    pub const ALL: [RadarTable; 6] = [
        Self::Reflectivity,
        Self::RadialVelocity,
        Self::DifferentialReflectivity,
        Self::CorrelationCoefficient,
        Self::SpecificDifferentialPhase,
        Self::DifferentialPhase,
    ];

    /// The name a caller selects the table by.
    pub fn name(self) -> &'static str {
        match self {
            Self::Reflectivity => "radar_reflectivity",
            Self::RadialVelocity => "radar_velocity",
            Self::DifferentialReflectivity => "differential_reflectivity",
            Self::CorrelationCoefficient => "correlation_coefficient",
            Self::SpecificDifferentialPhase => "specific_differential_phase",
            Self::DifferentialPhase => "differential_phase",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|table| table.name() == name)
    }

    fn table(self) -> Table {
        let (stops, sampling, units_per_display_unit) = match self {
            Self::Reflectivity => (REFLECTIVITY_ROWS, Sampling::Interval, 1.0),
            Self::RadialVelocity => (VELOCITY_ROWS, Sampling::Interval, 1.9426),
            Self::DifferentialReflectivity => (ZDR_ROWS, Sampling::Linear, 1.0),
            Self::CorrelationCoefficient => (CC_ROWS, Sampling::Linear, 1.0),
            Self::SpecificDifferentialPhase => (KDP_ROWS, Sampling::Linear, 1.0),
            Self::DifferentialPhase => (PHIDP_ROWS, Sampling::Linear, 1.0),
        };
        Table {
            stops,
            sampling,
            units_per_display_unit,
        }
    }

    /// The table's colour at `value`, in display units (dBZ, m/s, dB,
    /// unitless, deg/km, deg).
    pub fn sample(self, value: f64) -> Color {
        let [r, g, b, a] = self.table().sample(value as f32);
        Color::rgba(r, g, b, a)
    }

    /// A discrete scale over `[low, high]` in bins of `step`, each bin the
    /// table's colour at its centre.
    pub fn scale(
        self,
        low: f64,
        high: f64,
        step: f64,
        extend: ExtendMode,
        mask_below: Option<f64>,
    ) -> DiscreteColorScale {
        let bins = ((high - low) / step).round().max(1.0) as usize;
        let levels: Vec<f64> = (0..=bins)
            .map(|index| round_level(low + step * index as f64))
            .collect();
        let colors = levels
            .windows(2)
            .map(|pair| self.sample(0.5 * (pair[0] + pair[1])))
            .collect();
        DiscreteColorScale {
            levels,
            colors,
            extend,
            mask_below,
        }
    }
}

/// Bin edges stay on their decimal values (0.1 steps sum to 0.30000000000000004).
fn round_level(value: f64) -> f64 {
    (value * 1.0e6).round() / 1.0e6
}

impl Table {
    fn sample(&self, display_value: f32) -> [u8; 4] {
        let unit_scale = 1.0f32 / self.units_per_display_unit;
        let value_of = |stop: &Stop| stop.value * unit_scale;
        let first = &self.stops[0];
        if display_value <= value_of(first) {
            return first.rgba;
        }
        match self.sampling {
            Sampling::Interval => {
                let index = self
                    .stops
                    .partition_point(|stop| value_of(stop) <= display_value);
                let stop = &self.stops[index.saturating_sub(1)];
                let Some(next) = self.stops.get(index) else {
                    return stop.rgba;
                };
                let end = stop
                    .end
                    .unwrap_or(if stop.rgba[3] == 0 { stop.rgba } else { next.rgba });
                let span = (value_of(next) - value_of(stop)).max(f32::EPSILON);
                let t = ((display_value - value_of(stop)) / span).clamp(0.0, 1.0);
                lerp(stop.rgba, end, t)
            }
            Sampling::Linear => {
                let index = self
                    .stops
                    .partition_point(|stop| value_of(stop) < display_value);
                let Some(right) = self.stops.get(index) else {
                    return self.stops[self.stops.len() - 1].rgba;
                };
                if display_value == value_of(right) {
                    return right.rgba;
                }
                let left = &self.stops[index - 1];
                let span = (value_of(right) - value_of(left)).max(f32::EPSILON);
                lerp(left.rgba, right.rgba, (display_value - value_of(left)) / span)
            }
        }
    }
}

fn lerp(left: [u8; 4], right: [u8; 4], amount: f32) -> [u8; 4] {
    let amount = amount.clamp(0.0, 1.0);
    let channel = |a: u8, b: u8| {
        ((a as f32 + (b as f32 - a as f32) * amount).round()).clamp(0.0, 255.0) as u8
    };
    [
        channel(left[0], right[0]),
        channel(left[1], right[1]),
        channel(left[2], right[2]),
        channel(left[3], right[3]),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rgb(color: Color) -> [u8; 3] {
        [color.r, color.g, color.b]
    }

    #[test]
    fn the_colour_set_is_named_and_refuses_unknown_names() {
        for name in RadarColorSet::NAMES {
            assert_eq!(RadarColorSet::parse(name).unwrap().name(), name);
        }
        assert_eq!(RadarColorSet::default(), RadarColorSet::Standard);
        let refusal = RadarColorSet::parse("rainbow").unwrap_err();
        assert!(refusal.contains("standard") && refusal.contains("classic"), "{refusal}");
    }

    #[test]
    fn every_table_is_ascending_and_selectable_by_name() {
        for table in RadarTable::ALL {
            let rows = table.table().stops;
            assert!(rows.len() >= 2, "{}", table.name());
            assert!(
                rows.windows(2).all(|pair| pair[1].value > pair[0].value),
                "{} rows are not ascending",
                table.name()
            );
            assert_eq!(RadarTable::from_name(table.name()), Some(table));
        }
        assert_eq!(RadarTable::from_name("reflectivity"), None);
    }

    /// Values sampled by hand from the BowEcho rows, the way its sampler
    /// reads them: an interval row starts at its own colour, a row with no
    /// end colour ramps to the next row's.
    #[test]
    fn reflectivity_samples_the_interval_rows() {
        let table = RadarTable::Reflectivity;
        assert_eq!(rgb(table.sample(10.0)), [67, 97, 162]);
        assert_eq!(rgb(table.sample(14.0)), [87, 153, 195]);
        assert_eq!(rgb(table.sample(40.0)), [255, 226, 0]);
        assert_eq!(rgb(table.sample(45.0)), [255, 177, 0]);
        assert_eq!(rgb(table.sample(50.0)), [255, 0, 0]);
        assert_eq!(rgb(table.sample(60.0)), [255, 255, 255]);
        assert_eq!(rgb(table.sample(90.0)), [1, 32, 32]);
        assert_eq!(table.sample(-31.0).a, 0, "the -30 dBZ start is transparent");
    }

    #[test]
    fn velocity_greens_toward_reds_away_and_dark_near_zero() {
        let table = RadarTable::RadialVelocity;
        // Just inside 40 kt inbound (20.5909 m/s): the bright green row start.
        assert_eq!(rgb(table.sample(-20.59)), [10, 248, 35]);
        // Just away from the radar: the zero row's grey.
        assert_eq!(rgb(table.sample(0.0)), [130, 106, 120]);
        // Just past 10 kt outbound (5.1478 m/s): dark red, brightening outward.
        assert_eq!(rgb(table.sample(5.148)), [105, 0, 0]);
        for speed in [2.0, 5.0, 10.0, 15.0, 20.0] {
            let toward = table.sample(-speed);
            let away = table.sample(speed);
            assert!(toward.g > toward.r, "{speed} m/s toward the radar is green");
            assert!(away.r > away.g, "{speed} m/s away from the radar is red");
        }
        let near_zero = [table.sample(-0.25), table.sample(0.25)];
        for color in near_zero {
            let spread = color.r.max(color.g).max(color.b) - color.r.min(color.g).min(color.b);
            assert!(spread < 40, "near zero is a muted grey band: {color:?}");
        }
    }

    #[test]
    fn dual_pol_tables_follow_their_rows() {
        assert_eq!(rgb(RadarTable::DifferentialReflectivity.sample(0.0)), [140, 140, 140]);
        assert_eq!(rgb(RadarTable::DifferentialReflectivity.sample(2.5)), [235, 189, 46]);
        assert_eq!(rgb(RadarTable::CorrelationCoefficient.sample(0.95)), [245, 158, 32]);
        assert_eq!(rgb(RadarTable::CorrelationCoefficient.sample(1.0)), [236, 236, 244]);
        assert_eq!(rgb(RadarTable::SpecificDifferentialPhase.sample(0.0)), [60, 64, 70]);
        assert_eq!(rgb(RadarTable::SpecificDifferentialPhase.sample(9.0)), [180, 30, 96]);
        assert_eq!(rgb(RadarTable::DifferentialPhase.sample(45.0)), [18, 127, 175]);
    }

    #[test]
    fn scales_have_one_colour_per_bin_on_decimal_edges() {
        let scale = RadarTable::DifferentialReflectivity.scale(-4.0, 8.0, 0.1, ExtendMode::Both, None);
        assert_eq!(scale.levels.len(), scale.colors.len() + 1);
        assert_eq!(scale.levels.len(), 121);
        assert_eq!(scale.levels[43], 0.3);
        assert!(scale.levels.windows(2).all(|pair| pair[1] > pair[0]));
        let velocity = RadarTable::RadialVelocity.scale(-60.0, 60.0, 0.5, ExtendMode::Both, None);
        let zero = velocity.levels.iter().position(|level| *level == 0.0);
        assert_eq!(zero, Some(120), "zero is a bin edge, so the sign is the colour break");
    }
}
