use crate::error::{WrfError, WrfResult};

/// Unit types supported by wrf-rust.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WrfUnits {
    // Temperature
    Kelvin,
    Celsius,
    Fahrenheit,
    // Pressure
    Pascal,
    Hectopascal,
    Millibar,
    InchesOfMercury,
    // Speed
    MetersPerSecond,
    Knots,
    Mph,
    Kph,
    // Length / height
    Meters,
    Decameters,
    Feet,
    Kilometers,
    Miles,
    // Moisture / dimensionless
    KgPerKg,
    GramsPerKg,
    Percent,
    // Energy
    JoulesPerKg,
    // Angular
    Degrees,
    Radians,
    // Radar
    Dbz,
    // Vertical velocity
    PascalPerSecond,
    MicrobarsPerSecond,
    // Geopotential
    M2PerS2,
    // Vorticity
    PerSecond,
    // Dimensionless (STP, SCP, etc.)
    Dimensionless,
    // Precipitable water
    Millimeters,
    Inches,
}

/// Parse a unit string (case-insensitive) into a WrfUnits variant.
pub fn parse_units(s: &str) -> WrfResult<WrfUnits> {
    match s.to_lowercase().trim() {
        // Temperature
        "k" | "kelvin" => Ok(WrfUnits::Kelvin),
        "c" | "degc" | "celsius" | "degrees_celsius" | "degree_celsius" => Ok(WrfUnits::Celsius),
        "f" | "degf" | "fahrenheit" | "degrees_fahrenheit" => Ok(WrfUnits::Fahrenheit),
        // Pressure
        "pa" | "pascal" | "pascals" => Ok(WrfUnits::Pascal),
        "hpa" | "hectopascal" | "hectopascals" => Ok(WrfUnits::Hectopascal),
        "mb" | "mbar" | "millibar" | "millibars" => Ok(WrfUnits::Millibar),
        "inhg" | "inches_hg" | "incheshg" => Ok(WrfUnits::InchesOfMercury),
        // Speed
        "m/s" | "m s-1" | "ms-1" | "m_s-1" | "meters_per_second" | "mps" => {
            Ok(WrfUnits::MetersPerSecond)
        }
        "kt" | "kts" | "knots" | "knot" => Ok(WrfUnits::Knots),
        "mph" | "mi/h" => Ok(WrfUnits::Mph),
        "kph" | "km/h" | "kmh" => Ok(WrfUnits::Kph),
        // Length
        "m" | "meters" | "meter" => Ok(WrfUnits::Meters),
        "dam" | "decameters" | "decameter" | "dkm" => Ok(WrfUnits::Decameters),
        "ft" | "feet" | "foot" => Ok(WrfUnits::Feet),
        "km" | "kilometers" => Ok(WrfUnits::Kilometers),
        "mi" | "miles" => Ok(WrfUnits::Miles),
        // Moisture
        "kg/kg" | "kg kg-1" => Ok(WrfUnits::KgPerKg),
        "g/kg" | "g kg-1" => Ok(WrfUnits::GramsPerKg),
        "%" | "percent" => Ok(WrfUnits::Percent),
        // Energy
        "j/kg" | "j kg-1" | "joules_per_kg" => Ok(WrfUnits::JoulesPerKg),
        // Angular
        "deg" | "degrees" | "degree" => Ok(WrfUnits::Degrees),
        "rad" | "radians" => Ok(WrfUnits::Radians),
        // Radar
        "dbz" => Ok(WrfUnits::Dbz),
        // Vertical velocity
        "pa/s" | "pa s-1" => Ok(WrfUnits::PascalPerSecond),
        "ub/s" | "microbar/s" => Ok(WrfUnits::MicrobarsPerSecond),
        // Geopotential
        "m2/s2" | "m2 s-2" | "m^2/s^2" => Ok(WrfUnits::M2PerS2),
        // Vorticity
        "s-1" | "1/s" | "/s" => Ok(WrfUnits::PerSecond),
        // Dimensionless
        "" | "dimensionless" | "none" | "unitless" => Ok(WrfUnits::Dimensionless),
        // Precipitable water / depth
        "mm" | "millimeters" => Ok(WrfUnits::Millimeters),
        "in" | "inches" => Ok(WrfUnits::Inches),
        other => Err(WrfError::UnitConversion(format!(
            "unrecognized unit string: '{other}'"
        ))),
    }
}

/// Convert a single value between compatible units.
pub fn convert_value(value: f64, from: WrfUnits, to: WrfUnits) -> WrfResult<f64> {
    if from == to {
        return Ok(value);
    }
    match (from, to) {
        // ── Temperature ──
        (WrfUnits::Kelvin, WrfUnits::Celsius) => Ok(value - 273.15),
        (WrfUnits::Celsius, WrfUnits::Kelvin) => Ok(value + 273.15),
        (WrfUnits::Celsius, WrfUnits::Fahrenheit) => Ok(value * 9.0 / 5.0 + 32.0),
        (WrfUnits::Fahrenheit, WrfUnits::Celsius) => Ok((value - 32.0) * 5.0 / 9.0),
        (WrfUnits::Kelvin, WrfUnits::Fahrenheit) => Ok((value - 273.15) * 9.0 / 5.0 + 32.0),
        (WrfUnits::Fahrenheit, WrfUnits::Kelvin) => Ok((value - 32.0) * 5.0 / 9.0 + 273.15),

        // ── Pressure ──
        (WrfUnits::Pascal, WrfUnits::Hectopascal | WrfUnits::Millibar) => Ok(value / 100.0),
        (WrfUnits::Hectopascal | WrfUnits::Millibar, WrfUnits::Pascal) => Ok(value * 100.0),
        (WrfUnits::Hectopascal, WrfUnits::Millibar)
        | (WrfUnits::Millibar, WrfUnits::Hectopascal) => Ok(value),
        (WrfUnits::Pascal, WrfUnits::InchesOfMercury) => Ok(value / 3386.39),
        (WrfUnits::InchesOfMercury, WrfUnits::Pascal) => Ok(value * 3386.39),
        (WrfUnits::Hectopascal | WrfUnits::Millibar, WrfUnits::InchesOfMercury) => {
            Ok(value * 100.0 / 3386.39)
        }
        (WrfUnits::InchesOfMercury, WrfUnits::Hectopascal | WrfUnits::Millibar) => {
            Ok(value * 3386.39 / 100.0)
        }

        // ── Speed ──
        (WrfUnits::MetersPerSecond, WrfUnits::Knots) => Ok(value / 0.514444),
        (WrfUnits::Knots, WrfUnits::MetersPerSecond) => Ok(value * 0.514444),
        (WrfUnits::MetersPerSecond, WrfUnits::Mph) => Ok(value / 0.44704),
        (WrfUnits::Mph, WrfUnits::MetersPerSecond) => Ok(value * 0.44704),
        (WrfUnits::MetersPerSecond, WrfUnits::Kph) => Ok(value * 3.6),
        (WrfUnits::Kph, WrfUnits::MetersPerSecond) => Ok(value / 3.6),
        (WrfUnits::Knots, WrfUnits::Mph) => Ok(value * 0.514444 / 0.44704),
        (WrfUnits::Mph, WrfUnits::Knots) => Ok(value * 0.44704 / 0.514444),
        (WrfUnits::Knots, WrfUnits::Kph) => Ok(value * 0.514444 * 3.6),
        (WrfUnits::Kph, WrfUnits::Knots) => Ok(value / (0.514444 * 3.6)),
        (WrfUnits::Mph, WrfUnits::Kph) => Ok(value * 1.60934),
        (WrfUnits::Kph, WrfUnits::Mph) => Ok(value / 1.60934),

        // ── Length ──
        (WrfUnits::Meters, WrfUnits::Decameters) => Ok(value / 10.0),
        (WrfUnits::Decameters, WrfUnits::Meters) => Ok(value * 10.0),
        (WrfUnits::Meters, WrfUnits::Feet) => Ok(value / 0.3048),
        (WrfUnits::Feet, WrfUnits::Meters) => Ok(value * 0.3048),
        (WrfUnits::Meters, WrfUnits::Kilometers) => Ok(value / 1000.0),
        (WrfUnits::Kilometers, WrfUnits::Meters) => Ok(value * 1000.0),
        (WrfUnits::Meters, WrfUnits::Miles) => Ok(value / 1609.344),
        (WrfUnits::Miles, WrfUnits::Meters) => Ok(value * 1609.344),
        (WrfUnits::Decameters, WrfUnits::Feet) => Ok(value * 10.0 / 0.3048),
        (WrfUnits::Feet, WrfUnits::Decameters) => Ok(value * 0.3048 / 10.0),
        (WrfUnits::Decameters, WrfUnits::Kilometers) => Ok(value / 100.0),
        (WrfUnits::Kilometers, WrfUnits::Decameters) => Ok(value * 100.0),
        (WrfUnits::Decameters, WrfUnits::Miles) => Ok(value * 10.0 / 1609.344),
        (WrfUnits::Miles, WrfUnits::Decameters) => Ok(value * 1609.344 / 10.0),
        (WrfUnits::Feet, WrfUnits::Kilometers) => Ok(value * 0.3048 / 1000.0),
        (WrfUnits::Kilometers, WrfUnits::Feet) => Ok(value * 1000.0 / 0.3048),
        (WrfUnits::Feet, WrfUnits::Miles) => Ok(value / 5280.0),
        (WrfUnits::Miles, WrfUnits::Feet) => Ok(value * 5280.0),
        (WrfUnits::Kilometers, WrfUnits::Miles) => Ok(value / 1.60934),
        (WrfUnits::Miles, WrfUnits::Kilometers) => Ok(value * 1.60934),

        // ── Moisture ──
        (WrfUnits::KgPerKg, WrfUnits::GramsPerKg) => Ok(value * 1000.0),
        (WrfUnits::GramsPerKg, WrfUnits::KgPerKg) => Ok(value / 1000.0),

        // ── Depth (precipitable water) ──
        (WrfUnits::Millimeters, WrfUnits::Inches) => Ok(value / 25.4),
        (WrfUnits::Inches, WrfUnits::Millimeters) => Ok(value * 25.4),
        (WrfUnits::Meters, WrfUnits::Millimeters) => Ok(value * 1000.0),
        (WrfUnits::Millimeters, WrfUnits::Meters) => Ok(value / 1000.0),

        // ── Angular ──
        (WrfUnits::Degrees, WrfUnits::Radians) => Ok(value * std::f64::consts::PI / 180.0),
        (WrfUnits::Radians, WrfUnits::Degrees) => Ok(value * 180.0 / std::f64::consts::PI),

        // ── Vertical velocity ──
        (WrfUnits::PascalPerSecond, WrfUnits::MicrobarsPerSecond) => Ok(value * 10.0),
        (WrfUnits::MicrobarsPerSecond, WrfUnits::PascalPerSecond) => Ok(value / 10.0),

        _ => Err(WrfError::UnitConversion(format!(
            "cannot convert {from:?} to {to:?}"
        ))),
    }
}

/// Convert an array of values in-place.
pub fn convert_array(values: &mut [f64], from: WrfUnits, to: WrfUnits) -> WrfResult<()> {
    if from == to {
        return Ok(());
    }
    // Validate the conversion is possible with a test value first
    convert_value(0.0, from, to)?;
    for v in values.iter_mut() {
        // Safe to unwrap: we validated above
        *v = convert_value(*v, from, to).unwrap();
    }
    Ok(())
}
