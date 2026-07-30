// Unit conversion module — a lightweight, zero-cost alternative to Python's
// `pint` library, covering the meteorological units used by MetPy.

/// Meteorological unit identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Units {
    // Temperature
    Kelvin,
    Celsius,
    Fahrenheit,

    // Speed
    MetersPerSecond,
    Knots,
    MPH,

    // Pressure
    Pascal,
    Hectopascal,
    Millibar,
    InchesOfMercury,

    // Length
    Meters,
    Feet,
    Kilometers,
    Miles,

    // Dimensionless / ratio
    Percent,
    KgPerKg,
    GramsPerKg,

    // Radar
    Dbz,

    // Angular
    Degrees,
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Returned when a conversion between incompatible unit categories is
/// requested (e.g. temperature -> speed).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnitConversionError {
    pub from: Units,
    pub to: Units,
    pub message: String,
}

impl std::fmt::Display for UnitConversionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for UnitConversionError {}

// ---------------------------------------------------------------------------
// Temperature
// ---------------------------------------------------------------------------

/// Convert a temperature `value` from unit `from` to unit `to`.
///
/// Supported units: `Kelvin`, `Celsius`, `Fahrenheit`.
pub fn convert_temperature(value: f64, from: Units, to: Units) -> Result<f64, UnitConversionError> {
    if from == to {
        return Ok(value);
    }
    // Normalise to Kelvin first.
    let kelvin = match from {
        Units::Kelvin => value,
        Units::Celsius => value + 273.15,
        Units::Fahrenheit => (value - 32.0) * 5.0 / 9.0 + 273.15,
        other => {
            return Err(UnitConversionError {
                from: other,
                to,
                message: format!("{other:?} is not a temperature unit"),
            })
        }
    };
    match to {
        Units::Kelvin => Ok(kelvin),
        Units::Celsius => Ok(kelvin - 273.15),
        Units::Fahrenheit => Ok((kelvin - 273.15) * 9.0 / 5.0 + 32.0),
        other => Err(UnitConversionError {
            from,
            to: other,
            message: format!("{other:?} is not a temperature unit"),
        }),
    }
}

// ---------------------------------------------------------------------------
// Speed
// ---------------------------------------------------------------------------

/// Metres-per-second equivalent of one knot.
const KNOT_TO_MS: f64 = 0.514444;
/// Metres-per-second equivalent of one mile-per-hour.
const MPH_TO_MS: f64 = 0.44704;

/// Convert a speed `value` from unit `from` to unit `to`.
///
/// Supported units: `MetersPerSecond`, `Knots`, `MPH`.
pub fn convert_speed(value: f64, from: Units, to: Units) -> Result<f64, UnitConversionError> {
    if from == to {
        return Ok(value);
    }
    let ms = match from {
        Units::MetersPerSecond => value,
        Units::Knots => value * KNOT_TO_MS,
        Units::MPH => value * MPH_TO_MS,
        other => {
            return Err(UnitConversionError {
                from: other,
                to,
                message: format!("{other:?} is not a speed unit"),
            })
        }
    };
    match to {
        Units::MetersPerSecond => Ok(ms),
        Units::Knots => Ok(ms / KNOT_TO_MS),
        Units::MPH => Ok(ms / MPH_TO_MS),
        other => Err(UnitConversionError {
            from,
            to: other,
            message: format!("{other:?} is not a speed unit"),
        }),
    }
}

// ---------------------------------------------------------------------------
// Pressure
// ---------------------------------------------------------------------------

/// Pascals per inch of mercury (1 inHg = 3386.39 Pa = 33.8639 hPa).
const INHG_TO_PA: f64 = 3386.39;

/// Convert a pressure `value` from unit `from` to unit `to`.
///
/// Supported units: `Pascal`, `Hectopascal`, `Millibar`, `InchesOfMercury`.
/// Note: 1 hPa == 1 mbar.
pub fn convert_pressure(value: f64, from: Units, to: Units) -> Result<f64, UnitConversionError> {
    if from == to {
        return Ok(value);
    }
    let pa = match from {
        Units::Pascal => value,
        Units::Hectopascal | Units::Millibar => value * 100.0,
        Units::InchesOfMercury => value * INHG_TO_PA,
        other => {
            return Err(UnitConversionError {
                from: other,
                to,
                message: format!("{other:?} is not a pressure unit"),
            })
        }
    };
    match to {
        Units::Pascal => Ok(pa),
        Units::Hectopascal | Units::Millibar => Ok(pa / 100.0),
        Units::InchesOfMercury => Ok(pa / INHG_TO_PA),
        other => Err(UnitConversionError {
            from,
            to: other,
            message: format!("{other:?} is not a pressure unit"),
        }),
    }
}

// ---------------------------------------------------------------------------
// Length
// ---------------------------------------------------------------------------

/// Metres equivalent of one foot.
const FOOT_TO_M: f64 = 0.3048;
/// Metres in one kilometre.
const KM_TO_M: f64 = 1000.0;
/// Metres in one statute mile.
const MILE_TO_M: f64 = 1609.344;

/// Convert a length `value` from unit `from` to unit `to`.
///
/// Supported units: `Meters`, `Feet`, `Kilometers`, `Miles`.
pub fn convert_length(value: f64, from: Units, to: Units) -> Result<f64, UnitConversionError> {
    if from == to {
        return Ok(value);
    }
    let meters = match from {
        Units::Meters => value,
        Units::Feet => value * FOOT_TO_M,
        Units::Kilometers => value * KM_TO_M,
        Units::Miles => value * MILE_TO_M,
        other => {
            return Err(UnitConversionError {
                from: other,
                to,
                message: format!("{other:?} is not a length unit"),
            })
        }
    };
    match to {
        Units::Meters => Ok(meters),
        Units::Feet => Ok(meters / FOOT_TO_M),
        Units::Kilometers => Ok(meters / KM_TO_M),
        Units::Miles => Ok(meters / MILE_TO_M),
        other => Err(UnitConversionError {
            from,
            to: other,
            message: format!("{other:?} is not a length unit"),
        }),
    }
}

// ---------------------------------------------------------------------------
// Convenience helpers (infallible, since the unit pair is known at compile
// time).
// ---------------------------------------------------------------------------

/// Celsius to Kelvin.
#[inline]
pub fn celsius_to_kelvin(c: f64) -> f64 {
    c + 273.15
}

/// Kelvin to Celsius.
#[inline]
pub fn kelvin_to_celsius(k: f64) -> f64 {
    k - 273.15
}

/// Celsius to Fahrenheit.
#[inline]
pub fn celsius_to_fahrenheit(c: f64) -> f64 {
    c * 9.0 / 5.0 + 32.0
}

/// Fahrenheit to Celsius.
#[inline]
pub fn fahrenheit_to_celsius(f: f64) -> f64 {
    (f - 32.0) * 5.0 / 9.0
}

/// Knots to metres per second.
#[inline]
pub fn knots_to_ms(kt: f64) -> f64 {
    kt * KNOT_TO_MS
}

/// Metres per second to knots.
#[inline]
pub fn ms_to_knots(ms: f64) -> f64 {
    ms / KNOT_TO_MS
}

/// Hectopascals to Pascals.
#[inline]
pub fn hpa_to_pa(hpa: f64) -> f64 {
    hpa * 100.0
}

/// Pascals to hectopascals.
#[inline]
pub fn pa_to_hpa(pa: f64) -> f64 {
    pa / 100.0
}

/// Feet to metres.
#[inline]
pub fn feet_to_meters(ft: f64) -> f64 {
    ft * FOOT_TO_M
}

/// Metres to feet.
#[inline]
pub fn meters_to_feet(m: f64) -> f64 {
    m / FOOT_TO_M
}

// ---------------------------------------------------------------------------
// Mixing ratio
// ---------------------------------------------------------------------------

/// Convert a mixing-ratio `value` between `KgPerKg` and `GramsPerKg`.
pub fn convert_mixing_ratio(value: f64, from: Units, to: Units) -> Result<f64, UnitConversionError> {
    if from == to {
        return Ok(value);
    }
    match (from, to) {
        (Units::KgPerKg, Units::GramsPerKg) => Ok(value * 1000.0),
        (Units::GramsPerKg, Units::KgPerKg) => Ok(value / 1000.0),
        _ => Err(UnitConversionError {
            from,
            to,
            message: format!(
                "convert_mixing_ratio only supports KgPerKg <-> GramsPerKg, got {from:?} -> {to:?}"
            ),
        }),
    }
}

// ---------------------------------------------------------------------------
// Generic dispatcher
// ---------------------------------------------------------------------------

/// Determine the category of a unit for dispatching.
fn unit_category(u: Units) -> &'static str {
    match u {
        Units::Kelvin | Units::Celsius | Units::Fahrenheit => "temperature",
        Units::MetersPerSecond | Units::Knots | Units::MPH => "speed",
        Units::Pascal | Units::Hectopascal | Units::Millibar | Units::InchesOfMercury => "pressure",
        Units::Meters | Units::Feet | Units::Kilometers | Units::Miles => "length",
        Units::KgPerKg | Units::GramsPerKg => "mixing_ratio",
        Units::Percent | Units::Dbz | Units::Degrees => "other",
    }
}

/// Convert `value` from unit `from` to unit `to`, automatically selecting the
/// correct category-specific converter.
///
/// This is the main entry-point that mirrors MetPy's `units.Quantity.to()`
/// workflow: callers do not need to know which category a unit belongs to.
pub fn convert(value: f64, from: Units, to: Units) -> Result<f64, UnitConversionError> {
    if from == to {
        return Ok(value);
    }

    let cat_from = unit_category(from);
    let cat_to = unit_category(to);

    if cat_from != cat_to {
        return Err(UnitConversionError {
            from,
            to,
            message: format!(
                "incompatible unit categories: {from:?} ({cat_from}) -> {to:?} ({cat_to})"
            ),
        });
    }

    match cat_from {
        "temperature" => convert_temperature(value, from, to),
        "speed" => convert_speed(value, from, to),
        "pressure" => convert_pressure(value, from, to),
        "length" => convert_length(value, from, to),
        "mixing_ratio" => convert_mixing_ratio(value, from, to),
        _ => Err(UnitConversionError {
            from,
            to,
            message: format!("conversion not supported for {from:?} -> {to:?}"),
        }),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-6;

    fn approx(a: f64, b: f64) {
        let diff = (a - b).abs();
        assert!(
            diff < TOL,
            "approx failed: {a} vs {b} (diff {diff})"
        );
    }

    fn approx_rel(a: f64, b: f64, rel: f64) {
        let diff = (a - b).abs();
        let mag = a.abs().max(b.abs()).max(1e-30);
        assert!(
            diff / mag < rel,
            "approx_rel failed: {a} vs {b} (diff {diff}, rel {rel})"
        );
    }

    // -- Temperature -------------------------------------------------------

    #[test]
    fn test_celsius_to_kelvin() {
        approx(celsius_to_kelvin(0.0), 273.15);
        approx(celsius_to_kelvin(100.0), 373.15);
        approx(celsius_to_kelvin(-40.0), 233.15);
    }

    #[test]
    fn test_kelvin_to_celsius() {
        approx(kelvin_to_celsius(273.15), 0.0);
        approx(kelvin_to_celsius(0.0), -273.15);
    }

    #[test]
    fn test_celsius_to_fahrenheit() {
        approx(celsius_to_fahrenheit(0.0), 32.0);
        approx(celsius_to_fahrenheit(100.0), 212.0);
        approx(celsius_to_fahrenheit(-40.0), -40.0);
    }

    #[test]
    fn test_fahrenheit_to_celsius() {
        approx(fahrenheit_to_celsius(32.0), 0.0);
        approx(fahrenheit_to_celsius(212.0), 100.0);
        approx(fahrenheit_to_celsius(-40.0), -40.0);
    }

    #[test]
    fn test_convert_temperature_roundtrip() {
        let original = 20.0;
        let k = convert_temperature(original, Units::Celsius, Units::Kelvin).unwrap();
        let f = convert_temperature(k, Units::Kelvin, Units::Fahrenheit).unwrap();
        let c = convert_temperature(f, Units::Fahrenheit, Units::Celsius).unwrap();
        approx(c, original);
    }

    #[test]
    fn test_convert_temperature_identity() {
        approx(
            convert_temperature(300.0, Units::Kelvin, Units::Kelvin).unwrap(),
            300.0,
        );
    }

    #[test]
    fn test_convert_temperature_bad_unit() {
        assert!(convert_temperature(1.0, Units::Pascal, Units::Kelvin).is_err());
        assert!(convert_temperature(1.0, Units::Kelvin, Units::Meters).is_err());
    }

    // -- Speed -------------------------------------------------------------

    #[test]
    fn test_knots_to_ms() {
        approx_rel(knots_to_ms(1.0), 0.514444, 1e-5);
        approx_rel(knots_to_ms(100.0), 51.4444, 1e-5);
    }

    #[test]
    fn test_ms_to_knots() {
        approx_rel(ms_to_knots(0.514444), 1.0, 1e-4);
    }

    #[test]
    fn test_convert_speed_mph() {
        let ms = convert_speed(60.0, Units::MPH, Units::MetersPerSecond).unwrap();
        approx_rel(ms, 26.8224, 1e-5);
    }

    #[test]
    fn test_convert_speed_roundtrip() {
        let original = 50.0;
        let ms = convert_speed(original, Units::Knots, Units::MetersPerSecond).unwrap();
        let mph = convert_speed(ms, Units::MetersPerSecond, Units::MPH).unwrap();
        let kt = convert_speed(mph, Units::MPH, Units::Knots).unwrap();
        approx_rel(kt, original, 1e-10);
    }

    #[test]
    fn test_convert_speed_identity() {
        approx(
            convert_speed(42.0, Units::Knots, Units::Knots).unwrap(),
            42.0,
        );
    }

    #[test]
    fn test_convert_speed_bad_unit() {
        assert!(convert_speed(1.0, Units::Celsius, Units::Knots).is_err());
    }

    // -- Pressure ----------------------------------------------------------

    #[test]
    fn test_hpa_to_pa() {
        approx(hpa_to_pa(1013.25), 101325.0);
    }

    #[test]
    fn test_pa_to_hpa() {
        approx(pa_to_hpa(101325.0), 1013.25);
    }

    #[test]
    fn test_convert_pressure_millibar_hpa() {
        // 1 hPa == 1 mbar exactly
        approx(
            convert_pressure(500.0, Units::Hectopascal, Units::Millibar).unwrap(),
            500.0,
        );
    }

    #[test]
    fn test_convert_pressure_roundtrip() {
        let original = 850.0;
        let pa = convert_pressure(original, Units::Hectopascal, Units::Pascal).unwrap();
        let mb = convert_pressure(pa, Units::Pascal, Units::Millibar).unwrap();
        approx(mb, original);
    }

    #[test]
    fn test_convert_pressure_bad_unit() {
        assert!(convert_pressure(1.0, Units::Meters, Units::Pascal).is_err());
    }

    // -- Length -------------------------------------------------------------

    #[test]
    fn test_feet_to_meters() {
        approx(feet_to_meters(1.0), 0.3048);
        approx_rel(feet_to_meters(1000.0), 304.8, 1e-10);
    }

    #[test]
    fn test_meters_to_feet() {
        approx_rel(meters_to_feet(1.0), 3.28084, 1e-4);
    }

    #[test]
    fn test_convert_length_km() {
        approx(
            convert_length(1.0, Units::Kilometers, Units::Meters).unwrap(),
            1000.0,
        );
    }

    #[test]
    fn test_convert_length_miles() {
        approx_rel(
            convert_length(1.0, Units::Miles, Units::Kilometers).unwrap(),
            1.609344,
            1e-6,
        );
    }

    #[test]
    fn test_convert_length_roundtrip() {
        let original = 5280.0;
        let m = convert_length(original, Units::Feet, Units::Meters).unwrap();
        let km = convert_length(m, Units::Meters, Units::Kilometers).unwrap();
        let mi = convert_length(km, Units::Kilometers, Units::Miles).unwrap();
        let ft = convert_length(mi, Units::Miles, Units::Feet).unwrap();
        approx_rel(ft, original, 1e-10);
    }

    #[test]
    fn test_convert_length_bad_unit() {
        assert!(convert_length(1.0, Units::Kelvin, Units::Meters).is_err());
    }

    // -- Pressure: InchesOfMercury ----------------------------------------

    #[test]
    fn test_inhg_to_hpa() {
        // Standard atmosphere: 29.9212 inHg ≈ 1013.25 hPa
        let hpa = convert_pressure(29.9212, Units::InchesOfMercury, Units::Hectopascal).unwrap();
        approx_rel(hpa, 1013.25, 1e-3);
    }

    #[test]
    fn test_hpa_to_inhg() {
        let inhg = convert_pressure(1013.25, Units::Hectopascal, Units::InchesOfMercury).unwrap();
        approx_rel(inhg, 29.9212, 1e-3);
    }

    #[test]
    fn test_inhg_roundtrip() {
        let original = 30.00;
        let pa = convert_pressure(original, Units::InchesOfMercury, Units::Pascal).unwrap();
        let back = convert_pressure(pa, Units::Pascal, Units::InchesOfMercury).unwrap();
        approx_rel(back, original, 1e-10);
    }

    #[test]
    fn test_inhg_to_millibar() {
        // 1 inHg = 33.8639 hPa = 33.8639 mbar
        let mb = convert_pressure(1.0, Units::InchesOfMercury, Units::Millibar).unwrap();
        approx_rel(mb, 33.8639, 1e-4);
    }

    // -- Mixing ratio ------------------------------------------------------

    #[test]
    fn test_kg_to_g_per_kg() {
        approx(
            convert_mixing_ratio(0.012, Units::KgPerKg, Units::GramsPerKg).unwrap(),
            12.0,
        );
    }

    #[test]
    fn test_g_to_kg_per_kg() {
        approx(
            convert_mixing_ratio(12.0, Units::GramsPerKg, Units::KgPerKg).unwrap(),
            0.012,
        );
    }

    #[test]
    fn test_mixing_ratio_identity() {
        approx(
            convert_mixing_ratio(5.0, Units::GramsPerKg, Units::GramsPerKg).unwrap(),
            5.0,
        );
    }

    #[test]
    fn test_mixing_ratio_bad_unit() {
        assert!(convert_mixing_ratio(1.0, Units::Kelvin, Units::GramsPerKg).is_err());
    }

    // -- Generic convert ---------------------------------------------------

    #[test]
    fn test_convert_dispatches_temperature() {
        let k = convert(100.0, Units::Celsius, Units::Kelvin).unwrap();
        approx(k, 373.15);
    }

    #[test]
    fn test_convert_dispatches_speed() {
        let ms = convert(1.0, Units::Knots, Units::MetersPerSecond).unwrap();
        approx_rel(ms, 0.514444, 1e-5);
    }

    #[test]
    fn test_convert_dispatches_pressure() {
        let pa = convert(1013.25, Units::Hectopascal, Units::Pascal).unwrap();
        approx(pa, 101325.0);
    }

    #[test]
    fn test_convert_dispatches_pressure_inhg() {
        let hpa = convert(29.9212, Units::InchesOfMercury, Units::Hectopascal).unwrap();
        approx_rel(hpa, 1013.25, 1e-3);
    }

    #[test]
    fn test_convert_dispatches_length() {
        approx(
            convert(1.0, Units::Kilometers, Units::Meters).unwrap(),
            1000.0,
        );
    }

    #[test]
    fn test_convert_dispatches_mixing_ratio() {
        approx(
            convert(0.010, Units::KgPerKg, Units::GramsPerKg).unwrap(),
            10.0,
        );
    }

    #[test]
    fn test_convert_identity() {
        approx(convert(42.0, Units::Kelvin, Units::Kelvin).unwrap(), 42.0);
    }

    #[test]
    fn test_convert_incompatible() {
        assert!(convert(1.0, Units::Kelvin, Units::Meters).is_err());
        assert!(convert(1.0, Units::Pascal, Units::Knots).is_err());
        assert!(convert(1.0, Units::GramsPerKg, Units::Fahrenheit).is_err());
    }

    // -- Error display -----------------------------------------------------

    #[test]
    fn test_error_display() {
        let err = UnitConversionError {
            from: Units::Pascal,
            to: Units::Kelvin,
            message: "incompatible units".to_string(),
        };
        assert_eq!(format!("{err}"), "incompatible units");
    }
}
