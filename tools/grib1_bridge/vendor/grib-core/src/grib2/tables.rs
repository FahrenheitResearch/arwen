/// Look up the human-readable name of a GRIB2 parameter.
/// Based on WMO GRIB2 Code Table 4.2 and NCEP local-use extensions.
/// Returns "Unknown" for unrecognized combinations.
pub fn parameter_name(discipline: u8, category: u8, number: u8) -> &'static str {
    match (discipline, category, number) {
        // =====================================================================
        // Discipline 0: Meteorological Products
        // =====================================================================

        // Category 0: Temperature
        (0, 0, 0) => "Temperature",
        (0, 0, 1) => "Virtual Temperature",
        (0, 0, 2) => "Potential Temperature",
        (0, 0, 3) => "Pseudo-Adiabatic Potential Temperature",
        (0, 0, 4) => "Maximum Temperature",
        (0, 0, 5) => "Minimum Temperature",
        (0, 0, 6) => "Dewpoint Temperature",
        (0, 0, 7) => "Dewpoint Depression",
        (0, 0, 8) => "Lapse Rate",
        (0, 0, 9) => "Temperature Anomaly",
        (0, 0, 10) => "Latent Heat Net Flux",
        (0, 0, 11) => "Sensible Heat Net Flux",
        (0, 0, 12) => "Heat Index",
        (0, 0, 13) => "Wind Chill Factor",
        (0, 0, 14) => "Minimum Dewpoint Depression",
        (0, 0, 15) => "Virtual Potential Temperature",
        (0, 0, 16) => "Snow Phase Change Heat Flux",
        (0, 0, 17) => "Skin Temperature",
        (0, 0, 18) => "Snow Temperature (top of snow)",
        (0, 0, 19) => "Turbulent Transfer Coefficient for Heat",
        (0, 0, 20) => "Turbulent Diffusion Coefficient for Heat",
        (0, 0, 21) => "Apparent Temperature",
        (0, 0, 22) => "Temperature Tendency due to Short-Wave Radiation",
        (0, 0, 23) => "Temperature Tendency due to Long-Wave Radiation",
        (0, 0, 24) => "Temperature Tendency due to Short-Wave Radiation, Clear Sky",
        (0, 0, 25) => "Temperature Tendency due to Long-Wave Radiation, Clear Sky",
        (0, 0, 26) => "Temperature Tendency due to Parameterizations",
        (0, 0, 27) => "Wet Bulb Temperature",
        (0, 0, 28) => "Unbalanced Component of Temperature",
        (0, 0, 29) => "Temperature Advection",
        (0, 0, 30) => "Latent Heat Net Flux Due to Evaporation",
        (0, 0, 31) => "Latent Heat Net Flux Due to Sublimation",
        (0, 0, 32) => "Wet-Bulb Potential Temperature",
        // NCEP Local Use
        (0, 0, 192) => "Snow Phase Change Heat Flux",
        (0, 0, 193) => "Temperature Tendency by All Radiation",
        (0, 0, 194) => "Relative Error Variance",
        (0, 0, 195) => "Large Scale Condensate Heating Rate",
        (0, 0, 196) => "Deep Convective Heating Rate",
        (0, 0, 197) => "Total Downward Heat Flux at Surface",
        (0, 0, 198) => "Temperature Tendency by All Physics",
        (0, 0, 199) => "Temperature Tendency by Non-radiation Physics",
        (0, 0, 200) => "Standard Dev. of IR Temp. over 1x1 deg. area",
        (0, 0, 201) => "Shallow Convective Heating Rate",
        (0, 0, 202) => "Vertical Diffusion Heating rate",
        (0, 0, 203) => "Potential Temperature at Top of Viscous Sublayer",
        (0, 0, 204) => "Tropical Cyclone Heat Potential",

        // Category 1: Moisture
        (0, 1, 0) => "Specific Humidity",
        (0, 1, 1) => "Relative Humidity",
        (0, 1, 2) => "Humidity Mixing Ratio",
        (0, 1, 3) => "Precipitable Water",
        (0, 1, 4) => "Vapor Pressure",
        (0, 1, 5) => "Saturation Deficit",
        (0, 1, 6) => "Evaporation",
        (0, 1, 7) => "Precipitation Rate",
        (0, 1, 8) => "Total Precipitation",
        (0, 1, 9) => "Large-Scale Precipitation (non-convective)",
        (0, 1, 10) => "Convective Precipitation",
        (0, 1, 11) => "Snow Depth",
        (0, 1, 12) => "Snowfall Rate Water Equivalent",
        (0, 1, 13) => "Water Equivalent of Accumulated Snow Depth",
        (0, 1, 192) => "Categorical Rain (yes=1; no=0)",

        // Category 2: Momentum
        (0, 2, 0) => "Wind Direction (from which blowing)",
        (0, 2, 1) => "Wind Speed",
        (0, 2, 2) => "U-Component of Wind",
        (0, 2, 3) => "V-Component of Wind",
        (0, 2, 8) => "Vertical Velocity (Pressure)",
        (0, 2, 10) => "Absolute Vorticity",
        (0, 2, 12) => "Relative Vorticity",
        (0, 2, 22) => "Wind Speed (Gust)",

        // Category 3: Mass
        (0, 3, 0) => "Pressure",
        (0, 3, 1) => "Pressure Reduced to MSL",
        (0, 3, 5) => "Geopotential Height",
        (0, 3, 18) => "Planetary Boundary Layer Height",

        // Category 6: Cloud
        (0, 6, 1) => "Total Cloud Cover",
        (0, 6, 3) => "Low Cloud Cover",
        (0, 6, 4) => "Medium Cloud Cover",
        (0, 6, 5) => "High Cloud Cover",

        // Category 7: Thermodynamic Stability
        (0, 7, 6) => "Convective Available Potential Energy",
        (0, 7, 7) => "Convective Inhibition",
        (0, 7, 8) => "Storm Relative Helicity",

        // Category 17: Electrodynamics
        (0, 17, 0) => "Lightning Strike Density",
        (0, 17, 192) => "Lightning",

        // Category 19: Physical Atmospheric Properties
        (0, 19, 0) => "Visibility",

        // Discipline 2: Land Surface Products
        (2, 0, 0) => "Land Cover (0=sea, 1=land)",
        (2, 0, 2) => "Soil Temperature",

        // Discipline 3: Satellite Remote Sensing Products
        (3, 192, 7) => "Simulated Brightness Temperature for GOES 11, Channel 3",

        // Discipline 10: Oceanographic Products
        (10, 0, 3) => "Significant Height of Combined Wind Waves and Swell",

        _ => "Unknown",
    }
}

/// Look up the units of a GRIB2 parameter.
/// Based on WMO GRIB2 Code Table 4.2 and NCEP local-use extensions.
/// Returns "?" for unrecognized combinations.
pub fn parameter_units(discipline: u8, category: u8, number: u8) -> &'static str {
    match (discipline, category, number) {
        // Category 0: Temperature
        (0, 0, 0) | (0, 0, 1) | (0, 0, 2) | (0, 0, 3) => "K",
        (0, 0, 4) | (0, 0, 5) | (0, 0, 6) | (0, 0, 7) => "K",
        (0, 0, 8) => "K/m",
        (0, 0, 9) => "K",
        (0, 0, 10) | (0, 0, 11) => "W/m\u{b2}",
        (0, 0, 12) | (0, 0, 13) | (0, 0, 14) | (0, 0, 15) => "K",
        (0, 0, 16) => "W/m\u{b2}",
        (0, 0, 17) | (0, 0, 18) => "K",
        (0, 0, 27) => "K",
        (0, 0, 32) => "K",

        // Category 1: Moisture
        (0, 1, 0) => "kg/kg",
        (0, 1, 1) => "%",
        (0, 1, 2) => "kg/kg",
        (0, 1, 3) => "kg/m\u{b2}",
        (0, 1, 7) => "kg/m\u{b2}/s",
        (0, 1, 8) | (0, 1, 9) | (0, 1, 10) => "kg/m\u{b2}",
        (0, 1, 11) => "m",
        (0, 1, 13) => "kg/m\u{b2}",

        // Category 2: Momentum
        (0, 2, 0) => "degrees",
        (0, 2, 1) | (0, 2, 2) | (0, 2, 3) => "m/s",
        (0, 2, 8) => "Pa/s",
        (0, 2, 10) | (0, 2, 12) => "1/s",
        (0, 2, 22) => "m/s",

        // Category 3: Mass
        (0, 3, 0) | (0, 3, 1) => "Pa",
        (0, 3, 5) => "gpm",
        (0, 3, 18) => "m",

        // Category 4-5: Radiation
        (0, 4, _) => "W/m\u{b2}",
        (0, 5, _) => "W/m\u{b2}",

        // Category 6: Cloud
        (0, 6, 1) | (0, 6, 3) | (0, 6, 4) | (0, 6, 5) => "%",

        // Category 7: Thermodynamic Stability
        (0, 7, 6) | (0, 7, 7) => "J/kg",
        (0, 7, 8) => "m\u{b2}/s\u{b2}",

        // Category 17
        (0, 17, 0) => "m^-2 s^-1",
        (0, 17, 192) => "non-dim",

        // Category 19
        (0, 19, 0) => "m",

        // Discipline 3: Satellite Remote Sensing Products
        (3, 192, 7) => "K",

        _ => "?",
    }
}

/// Look up the human-readable name of a GRIB2 level type.
/// Based on WMO GRIB2 Code Table 4.5 and NCEP local-use extensions.
pub fn level_name(level_type: u8) -> &'static str {
    match level_type {
        1 => "Ground or Water Surface",
        2 => "Cloud Base Level",
        3 => "Cloud Top Level",
        4 => "Level of 0\u{b0}C Isotherm",
        5 => "Level of Adiabatic Condensation Lifted from Surface",
        6 => "Maximum Wind Level",
        7 => "Tropopause",
        8 => "Nominal Top of Atmosphere",
        9 => "Sea Bottom",
        10 => "Entire Atmosphere",
        11 => "Cumulonimbus Base",
        12 => "Cumulonimbus Top",
        100 => "Isobaric Surface",
        101 => "Mean Sea Level",
        102 => "Specific Altitude Above Mean Sea Level",
        103 => "Specified Height Level Above Ground",
        104 => "Sigma Level",
        105 => "Hybrid Level",
        106 => "Depth Below Land Surface",
        107 => "Isentropic (Theta) Level",
        108 => "Level at Specified Pressure Difference from Ground to Level",
        109 => "Potential Vorticity Surface",
        111 => "Eta Level",
        117 => "Mixed Layer Depth (m)",
        160 => "Depth Below Sea Level",
        200 => "Entire Atmosphere (as single layer)",
        201 => "Entire Ocean (as single layer)",
        204 => "Highest Tropospheric Freezing Level",
        215 => "Cloud Ceiling",
        220 => "Planetary Boundary Layer",
        242 => "Convective Cloud Bottom Level",
        243 => "Convective Cloud Top Level",
        _ => "Unknown Level Type",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameter_name_temperature() {
        assert_eq!(parameter_name(0, 0, 0), "Temperature");
    }

    #[test]
    fn test_parameter_name_dewpoint() {
        assert_eq!(parameter_name(0, 0, 6), "Dewpoint Temperature");
    }

    #[test]
    fn test_parameter_name_relative_humidity() {
        assert_eq!(parameter_name(0, 1, 1), "Relative Humidity");
    }

    #[test]
    fn test_parameter_name_u_wind() {
        assert_eq!(parameter_name(0, 2, 2), "U-Component of Wind");
    }

    #[test]
    fn test_parameter_name_v_wind() {
        assert_eq!(parameter_name(0, 2, 3), "V-Component of Wind");
    }

    #[test]
    fn test_parameter_name_total_precipitation() {
        assert_eq!(parameter_name(0, 1, 8), "Total Precipitation");
    }

    #[test]
    fn test_parameter_name_geopotential_height() {
        assert_eq!(parameter_name(0, 3, 5), "Geopotential Height");
    }

    #[test]
    fn test_parameter_name_specific_humidity() {
        assert_eq!(parameter_name(0, 1, 0), "Specific Humidity");
    }

    #[test]
    fn test_parameter_name_wind_speed() {
        assert_eq!(parameter_name(0, 2, 1), "Wind Speed");
    }

    #[test]
    fn test_parameter_name_ncep_local_categorical_rain() {
        assert_eq!(parameter_name(0, 1, 192), "Categorical Rain (yes=1; no=0)");
    }

    #[test]
    fn test_parameter_name_lightning_strike_density() {
        assert_eq!(parameter_name(0, 17, 0), "Lightning Strike Density");
    }

    #[test]
    fn test_parameter_name_lightning() {
        assert_eq!(parameter_name(0, 17, 192), "Lightning");
    }

    #[test]
    fn test_parameter_name_simulated_ir() {
        assert_eq!(
            parameter_name(3, 192, 7),
            "Simulated Brightness Temperature for GOES 11, Channel 3"
        );
    }

    #[test]
    fn test_parameter_name_unknown() {
        assert_eq!(parameter_name(255, 255, 255), "Unknown");
    }

    #[test]
    fn test_parameter_units_temperature_kelvin() {
        assert_eq!(parameter_units(0, 0, 0), "K");
    }

    #[test]
    fn test_parameter_units_relative_humidity_percent() {
        assert_eq!(parameter_units(0, 1, 1), "%");
    }

    #[test]
    fn test_parameter_units_wind_ms() {
        assert_eq!(parameter_units(0, 2, 1), "m/s");
    }

    #[test]
    fn test_parameter_units_precipitation_kgm2() {
        assert_eq!(parameter_units(0, 1, 8), "kg/m\u{b2}");
    }

    #[test]
    fn test_parameter_units_pressure_pa() {
        assert_eq!(parameter_units(0, 3, 0), "Pa");
    }

    #[test]
    fn test_parameter_units_geopotential_height() {
        assert_eq!(parameter_units(0, 3, 5), "gpm");
    }

    #[test]
    fn test_parameter_units_lapse_rate() {
        assert_eq!(parameter_units(0, 0, 8), "K/m");
    }

    #[test]
    fn test_parameter_units_heat_flux() {
        assert_eq!(parameter_units(0, 0, 10), "W/m\u{b2}");
    }

    #[test]
    fn test_parameter_units_lightning_strike_density() {
        assert_eq!(parameter_units(0, 17, 0), "m^-2 s^-1");
    }

    #[test]
    fn test_parameter_units_lightning() {
        assert_eq!(parameter_units(0, 17, 192), "non-dim");
    }

    #[test]
    fn test_parameter_units_simulated_ir() {
        assert_eq!(parameter_units(3, 192, 7), "K");
    }

    #[test]
    fn test_parameter_units_unknown() {
        assert_eq!(parameter_units(255, 255, 255), "?");
    }

    #[test]
    fn test_level_name_surface() {
        assert_eq!(level_name(1), "Ground or Water Surface");
    }

    #[test]
    fn test_level_name_isobaric() {
        assert_eq!(level_name(100), "Isobaric Surface");
    }

    #[test]
    fn test_level_name_msl() {
        assert_eq!(level_name(101), "Mean Sea Level");
    }

    #[test]
    fn test_level_name_height_above_ground() {
        assert_eq!(level_name(103), "Specified Height Level Above Ground");
    }

    #[test]
    fn test_level_name_hybrid() {
        assert_eq!(level_name(105), "Hybrid Level");
    }

    #[test]
    fn test_level_name_entire_atmosphere() {
        assert_eq!(level_name(200), "Entire Atmosphere (as single layer)");
    }

    #[test]
    fn test_level_name_tropopause() {
        assert_eq!(level_name(7), "Tropopause");
    }

    #[test]
    fn test_level_name_pbl() {
        assert_eq!(level_name(220), "Planetary Boundary Layer");
    }

    #[test]
    fn test_level_name_cloud_ceiling() {
        assert_eq!(level_name(215), "Cloud Ceiling");
    }

    #[test]
    fn test_level_name_unknown() {
        assert_eq!(level_name(199), "Unknown Level Type");
    }

    #[test]
    fn test_level_name_unknown_high_value() {
        assert_eq!(level_name(255), "Unknown Level Type");
    }
}
