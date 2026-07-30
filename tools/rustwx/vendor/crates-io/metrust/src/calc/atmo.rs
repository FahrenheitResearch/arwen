//! Standard atmosphere, pressure conversions, and apparent temperature indices.
//!
//! Implements the US Standard Atmosphere 1976 for height-pressure conversions,
//! altimeter/station pressure conversions, sigma coordinate transforms, and
//! the NWS heat index, wind chill, and apparent temperature formulas.
//!
//! All temperatures in degrees Celsius, all pressures in hPa, heights in meters,
//! wind speeds in m/s, and relative humidity in percent [0-100].

/// Standard sea-level pressure (hPa).
const P0: f64 = 1013.25;

/// Standard sea-level temperature (K) — matches MetPy's value.
const T0: f64 = 288.0;

/// Temperature lapse rate in the troposphere (K/m).
const LAPSE_RATE: f64 = 0.0065;

/// Gravitational acceleration (m/s^2).
const G: f64 = 9.80665;

/// Molar mass of dry air (kg/mol).
const M_AIR: f64 = 0.0289644;

/// Universal gas constant (J/(mol*K)).
const R_STAR: f64 = 8.31447;

/// Exponent G*M / (R*L) used in the barometric formula.
const BARO_EXP: f64 = G * M_AIR / (R_STAR * LAPSE_RATE);

// ─────────────────────────────────────────────
// US Standard Atmosphere 1976
// ─────────────────────────────────────────────

/// Convert geometric height to pressure using the US Standard Atmosphere 1976.
///
/// Valid for the troposphere (0 to ~11 km). Uses the barometric formula with
/// a constant lapse rate of 6.5 K/km:
///
/// ```text
/// P = P0 * (1 - L*h / T0) ^ (g*M / (R*L))
/// ```
///
/// # Arguments
/// * `height_m` — Geometric height above mean sea level (m)
///
/// # Returns
/// Pressure in hPa.
///
/// # Examples
/// ```
/// use metrust::calc::atmo::height_to_pressure_std;
/// let p = height_to_pressure_std(0.0);
/// assert!((p - 1013.25).abs() < 0.01);
/// ```
pub fn height_to_pressure_std(height_m: f64) -> f64 {
    P0 * (1.0 - LAPSE_RATE * height_m / T0).powf(BARO_EXP)
}

/// Convert pressure to geometric height using the US Standard Atmosphere 1976.
///
/// Inverse of [`height_to_pressure_std`]. Valid in the troposphere.
///
/// ```text
/// h = (T0 / L) * (1 - (P / P0) ^ (R*L / (g*M)))
/// ```
///
/// # Arguments
/// * `pressure_hpa` — Pressure (hPa)
///
/// # Returns
/// Geometric height above mean sea level (m).
///
/// # Examples
/// ```
/// use metrust::calc::atmo::pressure_to_height_std;
/// let h = pressure_to_height_std(500.0);
/// assert!((h - 5574.0).abs() < 10.0);
/// ```
pub fn pressure_to_height_std(pressure_hpa: f64) -> f64 {
    (T0 / LAPSE_RATE) * (1.0 - (pressure_hpa / P0).powf(1.0 / BARO_EXP))
}

// ─────────────────────────────────────────────
// Altimeter / station pressure conversions
// ─────────────────────────────────────────────

/// Convert altimeter setting to station pressure.
///
/// Uses the Smithsonian Meteorological Tables (1951) formula, matching MetPy's
/// implementation:
///
/// ```text
/// n = Rd * gamma / g        (≈ 0.190284)
/// P_stn = (A^n - p0^n * gamma * H / T0) ^ (1/n) + 0.3
/// ```
///
/// # Arguments
/// * `altimeter_hpa` — Altimeter setting (hPa)
/// * `elevation_m` — Station elevation above MSL (m)
///
/// # Returns
/// Station pressure (hPa).
///
/// # References
/// Smithsonian Meteorological Tables (1951), p. 269.
pub fn altimeter_to_station_pressure(altimeter_hpa: f64, elevation_m: f64) -> f64 {
    let n = 1.0 / BARO_EXP;
    (altimeter_hpa.powf(n) - P0.powf(n) * LAPSE_RATE * elevation_m / T0).powf(1.0 / n) + 0.3
}

/// Convert station pressure to altimeter setting.
///
/// Inverse of [`altimeter_to_station_pressure`] (Smithsonian formula).
///
/// ```text
/// n = Rd * gamma / g
/// A = ((P_stn - 0.3)^n + p0^n * gamma * H / T0) ^ (1/n)
/// ```
///
/// # Arguments
/// * `station_hpa` — Station pressure (hPa)
/// * `elevation_m` — Station elevation above MSL (m)
///
/// # Returns
/// Altimeter setting (hPa).
pub fn station_to_altimeter_pressure(station_hpa: f64, elevation_m: f64) -> f64 {
    let n = 1.0 / BARO_EXP;
    ((station_hpa - 0.3).powf(n) + P0.powf(n) * LAPSE_RATE * elevation_m / T0).powf(1.0 / n)
}

// ─────────────────────────────────────────────
// Altimeter to sea-level pressure
// ─────────────────────────────────────────────

/// Convert altimeter setting to sea-level pressure accounting for temperature.
///
/// First reduces the altimeter setting to station pressure using the Smithsonian
/// formula ([`altimeter_to_station_pressure`]), then applies a temperature-corrected
/// hypsometric equation to obtain the sea-level pressure.
///
/// ```text
/// P_stn = (A^n - p0^n * gamma * H / T0)^(1/n) + 0.3   (Smithsonian 1951)
/// SLP   = P_stn * exp( g * elev / (Rd * T_mean) )
/// ```
///
/// where `T_mean` is the mean column virtual temperature approximated as
/// `T_sfc + 0.5 * L * elev` (the average of surface and sea-level temperature).
///
/// # Arguments
/// * `alt_hpa`      — Altimeter setting (hPa)
/// * `elevation_m`  — Station elevation above MSL (m)
/// * `t_c`          — Station temperature (degrees Celsius)
///
/// # Returns
/// Estimated sea-level pressure (hPa).
///
/// # Examples
/// ```
/// use metrust::calc::atmo::altimeter_to_sea_level_pressure;
/// // At sea level, SLP equals the Smithsonian station value (altimeter + 0.3)
/// let slp = altimeter_to_sea_level_pressure(1013.25, 0.0, 15.0);
/// assert!((slp - 1013.55).abs() < 0.01);
/// ```
pub fn altimeter_to_sea_level_pressure(alt_hpa: f64, elevation_m: f64, t_c: f64) -> f64 {
    // Step 1: reduce altimeter to station pressure via standard atmosphere
    let p_stn = altimeter_to_station_pressure(alt_hpa, elevation_m);

    // Step 2: compute mean virtual temperature of the fictitious air column
    // between station and sea level.  Approximate Tv ≈ T since moisture
    // correction is small.  Use the average of station T and estimated
    // sea-level T (assuming standard lapse rate through the column).
    let t_sfc_k = t_c + 273.15;
    let t_mean_k = t_sfc_k + 0.5 * LAPSE_RATE * elevation_m;

    // Step 3: hypsometric equation  SLP = P_stn * exp( g * h / (Rd * T_mean) )
    const RD: f64 = 287.058; // specific gas constant for dry air (J/(kg·K))
    p_stn * (G * elevation_m / (RD * t_mean_k)).exp()
}

// ─────────────────────────────────────────────
// Sigma coordinate
// ─────────────────────────────────────────────

/// Convert a sigma coordinate to pressure.
///
/// ```text
/// P = sigma * (P_sfc - P_top) + P_top
/// ```
///
/// # Arguments
/// * `sigma` — Sigma value in [0, 1] (0 = model top, 1 = surface)
/// * `psfc_hpa` — Surface pressure (hPa)
/// * `ptop_hpa` — Model top pressure (hPa)
///
/// # Returns
/// Pressure (hPa).
pub fn sigma_to_pressure(sigma: f64, psfc_hpa: f64, ptop_hpa: f64) -> f64 {
    sigma * (psfc_hpa - ptop_hpa) + ptop_hpa
}

// ─────────────────────────────────────────────
// Apparent temperature indices
// ─────────────────────────────────────────────

/// Heat index using the Rothfusz regression.
///
/// Implements the full NWS heat index with Rothfusz (1990) polynomial regression
/// and the low/high-RH adjustments specified by the NWS.
///
/// # Arguments
/// * `temperature_c` — Air temperature (degrees Celsius). The regression is only
///   applicable when T >= 27 C (80 F); returns a simpler Steadman approximation
///   below that threshold.
/// * `relative_humidity_pct` — Relative humidity (percent, 0-100)
///
/// # Returns
/// Heat index in degrees Celsius.
///
/// # References
/// Rothfusz, L. P., 1990: The Heat Index "Equation" (or, More Than You Ever
/// Wanted to Know About Heat Index). NWS Technical Attachment SR 90-23.
pub fn heat_index(temperature_c: f64, relative_humidity_pct: f64) -> f64 {
    // Convert to Fahrenheit for the regression
    let t_f = temperature_c * 9.0 / 5.0 + 32.0;
    let rh = relative_humidity_pct;

    // NWS two-step: compute Steadman, average with T, then decide
    let steadman = 0.5 * (t_f + 61.0 + (t_f - 68.0) * 1.2 + rh * 0.094);
    let hi_avg = (steadman + t_f) / 2.0;

    if hi_avg < 80.0 {
        // Below threshold, return the averaged Steadman result
        return (hi_avg - 32.0) * 5.0 / 9.0;
    }

    // Rothfusz regression
    let mut hi_f = -42.379
        + 2.04901523 * t_f
        + 10.14333127 * rh
        - 0.22475541 * t_f * rh
        - 0.00683783 * t_f * t_f
        - 0.05481717 * rh * rh
        + 0.00122874 * t_f * t_f * rh
        + 0.00085282 * t_f * rh * rh
        - 0.00000199 * t_f * t_f * rh * rh;

    // Adjustment for low humidity at high temperatures
    if rh < 13.0 && t_f >= 80.0 && t_f <= 112.0 {
        let adjustment = -((13.0 - rh) / 4.0)
            * ((17.0 - (t_f - 95.0).abs()) / 17.0).sqrt();
        hi_f += adjustment;
    }

    // Adjustment for high humidity at moderate temperatures
    if rh > 85.0 && t_f >= 80.0 && t_f <= 87.0 {
        let adjustment = ((rh - 85.0) / 10.0) * ((87.0 - t_f) / 5.0);
        hi_f += adjustment;
    }

    // Convert back to Celsius
    (hi_f - 32.0) * 5.0 / 9.0
}

/// Wind chill index using the NWS/Environment Canada formula.
///
/// Computes the Wind Chill Temperature Index (WCTI) per the FCM formula.
/// The formula is always evaluated (matching MetPy's default behavior).
///
/// # Arguments
/// * `temperature_c` — Air temperature (degrees Celsius)
/// * `wind_speed_ms` — Wind speed at 10 m height (m/s)
///
/// # Returns
/// Wind chill temperature in degrees Celsius. The formula is applied
/// unconditionally; callers may wish to mask values where T > 10 C or
/// wind <= ~1.34 m/s (3 mph).
///
/// # References
/// NWS Wind Chill Temperature Index, adopted 2001 (Osczevski and Bluestein).
pub fn windchill(temperature_c: f64, wind_speed_ms: f64) -> f64 {
    let wind_kmh = wind_speed_ms * 3.6;
    let speed_factor = wind_kmh.powf(0.16);

    (0.6215 + 0.3965 * speed_factor) * temperature_c
        - 11.37 * speed_factor
        + 13.12
}

/// Apparent temperature combining heat index and wind chill.
///
/// Selects the appropriate index based on conditions:
/// - If T >= 27 C (80 F): returns the heat index
/// - If T <= 10 C (50 F) and wind > 1.34 m/s: returns wind chill
/// - Otherwise: returns the air temperature
///
/// # Arguments
/// * `temperature_c` — Air temperature (degrees Celsius)
/// * `rh_pct` — Relative humidity (percent, 0-100)
/// * `wind_speed_ms` — Wind speed at 10 m height (m/s)
///
/// # Returns
/// Apparent temperature in degrees Celsius.
pub fn apparent_temperature(temperature_c: f64, rh_pct: f64, wind_speed_ms: f64) -> f64 {
    let t_f = temperature_c * 9.0 / 5.0 + 32.0;
    let wind_mph = wind_speed_ms * 2.23694;

    if t_f >= 80.0 {
        heat_index(temperature_c, rh_pct)
    } else if t_f <= 50.0 && wind_mph > 3.0 {
        windchill(temperature_c, wind_speed_ms)
    } else {
        temperature_c
    }
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── height_to_pressure_std / pressure_to_height_std ──

    #[test]
    fn test_sea_level_pressure() {
        let p = height_to_pressure_std(0.0);
        assert!((p - 1013.25).abs() < 0.01, "sea level = {p} hPa");
    }

    #[test]
    fn test_500hpa_height() {
        // 500 hPa is approximately 5574 m in the standard atmosphere
        let h = pressure_to_height_std(500.0);
        assert!((h - 5574.0).abs() < 50.0, "500 hPa height = {h} m");
    }

    #[test]
    fn test_roundtrip_height_pressure() {
        for &h in &[0.0, 1000.0, 3000.0, 5000.0, 8000.0, 10000.0] {
            let p = height_to_pressure_std(h);
            let h2 = pressure_to_height_std(p);
            assert!(
                (h - h2).abs() < 0.01,
                "roundtrip failed at h={h}: got {h2}"
            );
        }
    }

    #[test]
    fn test_pressure_decreases_with_height() {
        let p0 = height_to_pressure_std(0.0);
        let p1 = height_to_pressure_std(1000.0);
        let p5 = height_to_pressure_std(5000.0);
        assert!(p0 > p1);
        assert!(p1 > p5);
    }

    #[test]
    fn test_known_std_atmo_levels() {
        // At ~1500 m, pressure should be about 845 hPa
        let p = height_to_pressure_std(1500.0);
        assert!((p - 845.6).abs() < 2.0, "1500m = {p} hPa");

        // At ~3000 m, pressure should be about 701 hPa
        let p = height_to_pressure_std(3000.0);
        assert!((p - 701.1).abs() < 2.0, "3000m = {p} hPa");
    }

    // ── altimeter / station pressure ──

    #[test]
    fn test_altimeter_station_roundtrip() {
        let elev = 1609.0; // Denver, ~1 mile high
        let altimeter = 1013.25;
        let station = altimeter_to_station_pressure(altimeter, elev);
        let alt_back = station_to_altimeter_pressure(station, elev);
        assert!(
            (alt_back - altimeter).abs() < 0.01,
            "roundtrip: {altimeter} -> {station} -> {alt_back}"
        );
    }

    #[test]
    fn test_station_pressure_lower_at_elevation() {
        let altimeter = 1013.25;
        let station = altimeter_to_station_pressure(altimeter, 1000.0);
        assert!(
            station < altimeter,
            "station pressure ({station}) should be lower than altimeter ({altimeter}) at elevation"
        );
    }

    #[test]
    fn test_altimeter_at_sea_level() {
        // At sea level, the Smithsonian formula gives station = altimeter + 0.3
        // (matching MetPy's behavior)
        let station = altimeter_to_station_pressure(1013.25, 0.0);
        assert!((station - 1013.55).abs() < 0.01, "sea-level station = {station}");
    }

    // ── sigma_to_pressure ──

    #[test]
    fn test_sigma_surface() {
        let p = sigma_to_pressure(1.0, 1000.0, 100.0);
        assert!((p - 1000.0).abs() < 1e-10);
    }

    #[test]
    fn test_sigma_top() {
        let p = sigma_to_pressure(0.0, 1000.0, 100.0);
        assert!((p - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_sigma_midlevel() {
        let p = sigma_to_pressure(0.5, 1000.0, 100.0);
        assert!((p - 550.0).abs() < 1e-10, "mid-sigma = {p}");
    }

    #[test]
    fn test_sigma_monotonic() {
        let psfc = 1013.25;
        let ptop = 10.0;
        let p1 = sigma_to_pressure(0.2, psfc, ptop);
        let p2 = sigma_to_pressure(0.5, psfc, ptop);
        let p3 = sigma_to_pressure(0.8, psfc, ptop);
        assert!(p1 < p2 && p2 < p3);
    }

    // ── heat_index ──

    #[test]
    fn test_heat_index_below_threshold() {
        // Below 80 F (26.7 C), heat index should be close to air temperature
        let hi = heat_index(20.0, 50.0);
        assert!(
            (hi - 20.0).abs() < 3.0,
            "heat index at 20C should be near 20C, got {hi}"
        );
    }

    #[test]
    fn test_heat_index_hot_humid() {
        // 35 C (95 F), 80% RH => heat index should be significantly above 35 C
        let hi = heat_index(35.0, 80.0);
        assert!(hi > 40.0, "heat index at 35C/80%RH = {hi}, expected > 40C");
    }

    #[test]
    fn test_heat_index_hot_dry() {
        // 40 C (104 F), 10% RH => heat index should be lower than the humid case
        let hi_dry = heat_index(40.0, 10.0);
        let hi_humid = heat_index(40.0, 80.0);
        assert!(
            hi_dry < hi_humid,
            "dry ({hi_dry}) should be less than humid ({hi_humid})"
        );
    }

    #[test]
    fn test_heat_index_nws_reference_point() {
        // NWS chart: 90 F (32.2 C) / 65% RH => HI ~ 103 F (39.4 C)
        let hi = heat_index(32.2, 65.0);
        assert!(
            (hi - 39.4).abs() < 2.0,
            "heat index at 32.2C/65% = {hi}, expected ~39.4C"
        );
    }

    // ── windchill ──

    #[test]
    fn test_windchill_warm_computed() {
        // Formula is always applied (matching MetPy), even above 10 C
        let wc = windchill(15.0, 10.0);
        // Should not just return 15.0; the formula yields a different value
        assert!(
            (wc - 15.0).abs() > 0.01,
            "windchill should compute formula even for warm temps, got {wc}"
        );
    }

    #[test]
    fn test_windchill_calm_computed() {
        // Formula is always applied, even with low wind
        let wc = windchill(-10.0, 1.0);
        assert!(
            (wc - (-10.0)).abs() > 0.01,
            "windchill should compute formula even for calm wind, got {wc}"
        );
    }

    #[test]
    fn test_windchill_cold_windy() {
        // -10 C, 10 m/s => significant wind chill
        let wc = windchill(-10.0, 10.0);
        assert!(wc < -10.0, "wind chill should be below air temp, got {wc}");
        assert!(wc > -30.0, "wind chill shouldn't be unreasonably low, got {wc}");
    }

    #[test]
    fn test_windchill_colder_with_more_wind() {
        let wc_low = windchill(-5.0, 5.0);
        let wc_high = windchill(-5.0, 15.0);
        assert!(
            wc_high < wc_low,
            "more wind should give lower wind chill: {wc_high} vs {wc_low}"
        );
    }

    #[test]
    fn test_windchill_nws_reference_point() {
        // NWS chart: 0 F (-17.8 C), 15 mph (6.7 m/s) => WC ~ -19 F (-28.3 C)
        let wc = windchill(-17.8, 6.7);
        assert!(
            (wc - (-28.3)).abs() < 2.0,
            "wind chill at -17.8C / 6.7 m/s = {wc}, expected ~-28.3C"
        );
    }

    // ── apparent_temperature ──

    #[test]
    fn test_apparent_temperature_hot() {
        // Hot conditions => uses heat index
        let at = apparent_temperature(35.0, 70.0, 2.0);
        let hi = heat_index(35.0, 70.0);
        assert!(
            (at - hi).abs() < 1e-10,
            "hot: apparent={at}, heat_index={hi}"
        );
    }

    #[test]
    fn test_apparent_temperature_cold() {
        // Cold + windy => uses wind chill
        let at = apparent_temperature(-10.0, 50.0, 10.0);
        let wc = windchill(-10.0, 10.0);
        assert!(
            (at - wc).abs() < 1e-10,
            "cold: apparent={at}, windchill={wc}"
        );
    }

    #[test]
    fn test_apparent_temperature_mild() {
        // Mild conditions => returns air temperature
        let at = apparent_temperature(18.0, 50.0, 3.0);
        assert!(
            (at - 18.0).abs() < 1e-10,
            "mild: apparent={at}, expected 18.0"
        );
    }

    #[test]
    fn test_apparent_temperature_cold_calm() {
        // Cold but calm wind => wind chill not applicable, returns air temp
        let at = apparent_temperature(-5.0, 50.0, 1.0);
        assert!(
            (at - (-5.0)).abs() < 1e-10,
            "cold calm: apparent={at}, expected -5.0"
        );
    }
}
