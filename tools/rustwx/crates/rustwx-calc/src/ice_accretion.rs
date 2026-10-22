//! Independent FRAM equations (1)-(8), doi:10.1175/WAF-D-15-0118.1.
//! NOAA/NWS published model. Rain is mm water equivalent; ice is mm.
//! Elevated horizontal ice converts to equivalent radial ice by 0.394.
//! Ground icing, melting, shedding and structural loads are not represented.
use rayon::prelude::*;

#[derive(Debug, Clone, Copy)]
pub struct IceAccretionIncrement {
    pub freezing_rain_mm: f64,
    pub horizontal_ice_mm: f64,
    pub radial_ice_mm: f64,
}

/// Rate in mm/hour, wind in m/s, wet-bulb in Celsius.
/// The published cold predictor is capped at -7 C. Missing inputs stay missing.
pub fn fram_ice_liquid_ratio(rate_mm_h: f64, wind_ms: f64, wetbulb_c: f64) -> f64 {
    if !rate_mm_h.is_finite()
        || !wind_ms.is_finite()
        || !wetbulb_c.is_finite()
        || rate_mm_h < 0.0
        || wind_ms < 0.0
    {
        return f64::NAN;
    }
    if rate_mm_h == 0.0 || wetbulb_c >= 1.0 {
        return 0.0;
    }
    let p = rate_mm_h / 25.4;
    let v = wind_ms / (1852.0 / 3600.0);
    let t = wetbulb_c.max(-7.0);
    let ip = 0.1395 * p.powf(-0.541);
    let it = ((-0.0071 * t - 0.1039) * t - 0.3904) * t + 0.5545;
    let iv = (0.0014 * v + 0.0027) * v + 0.7574;
    let (wp, wt, wv) = if t > -0.35 {
        (0.70, 0.29, 0.01)
    } else if v > 12.0 {
        (0.73, 0.01, 0.26)
    } else {
        (0.79, 0.20, 0.01)
    };
    (wp * ip + wt * it + wv * iv).max(0.0)
}

/// Microphysics precipitation-type flag AND subfreezing skin or 2 m.
/// Saved-frame callers must explicitly label their temporal sampling.
pub fn freezing_rain_increment(
    precipitation_mm: f64,
    hours: f64,
    freezing_rain: bool,
    temperature2_k: f64,
    skin_temperature_k: Option<f64>,
    wetbulb_c: f64,
    wind_ms: f64,
) -> IceAccretionIncrement {
    let missing = IceAccretionIncrement {
        freezing_rain_mm: f64::NAN,
        horizontal_ice_mm: f64::NAN,
        radial_ice_mm: f64::NAN,
    };
    if !hours.is_finite()
        || hours <= 0.0
        || !precipitation_mm.is_finite()
        || precipitation_mm < 0.0
        || !temperature2_k.is_finite()
        || temperature2_k <= 0.0
        || skin_temperature_k.is_some_and(|x| !x.is_finite() || x <= 0.0)
    {
        return missing;
    }
    let cold = temperature2_k < 273.15 || skin_temperature_k.is_some_and(|x| x < 273.15);
    if !freezing_rain || !cold || precipitation_mm == 0.0 {
        return IceAccretionIncrement {
            freezing_rain_mm: 0.0,
            horizontal_ice_mm: 0.0,
            radial_ice_mm: 0.0,
        };
    }
    let ilr = fram_ice_liquid_ratio(precipitation_mm / hours, wind_ms, wetbulb_c);
    IceAccretionIncrement {
        freezing_rain_mm: precipitation_mm,
        horizontal_ice_mm: precipitation_mm * ilr,
        radial_ice_mm: precipitation_mm * ilr * 0.394,
    }
}

/// Full-grid, all-core reference for the CUDA implementation.
pub fn fram_radial_grid(
    rate: &[f64],
    wind: &[f64],
    wetbulb: &[f64],
) -> Result<Vec<f64>, &'static str> {
    if rate.len() != wind.len() || rate.len() != wetbulb.len() {
        return Err("FRAM rate, wind and wet-bulb grids have different lengths");
    }
    Ok((0..rate.len())
        .into_par_iter()
        .map(|i| rate[i] * fram_ice_liquid_ratio(rate[i], wind[i], wetbulb[i]) * 0.394)
        .collect())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn publication_units_branches_radial_conversion() {
        let p = 0.1395_f64 * 0.1_f64.powf(-0.541);
        for (t, v, w) in [
            (-0.2, 10.0, (0.70, 0.29, 0.01)),
            (-0.35, 12.0, (0.79, 0.20, 0.01)),
            (-2.0, 20.0, (0.73, 0.01, 0.26)),
        ] {
            let it = -0.0071 * t * t * t - 0.1039 * t * t - 0.3904 * t + 0.5545;
            let iv = 0.0014 * v * v + 0.0027 * v + 0.7574;
            let expected = w.0 * p + w.1 * it + w.2 * iv;
            let actual = fram_ice_liquid_ratio(2.54, v * 1852.0 / 3600.0, t);
            assert!((actual - expected).abs() < 1e-12);
            let ice = freezing_rain_increment(
                2.54,
                1.0,
                true,
                271.0,
                Some(270.0),
                t,
                v * 1852.0 / 3600.0,
            );
            assert!((ice.radial_ice_mm - 2.54 * expected * 0.394).abs() < 1e-12);
        }
        assert_eq!(
            fram_ice_liquid_ratio(2.54, 5.0, -20.0),
            fram_ice_liquid_ratio(2.54, 5.0, -7.0)
        );
    }
    #[test]
    fn surface_type_dry_missing_gates() {
        assert_eq!(
            freezing_rain_increment(2.54, 1.0, false, 270.0, None, -3.0, 5.0).radial_ice_mm,
            0.0
        );
        assert_eq!(
            freezing_rain_increment(2.54, 1.0, true, 274.0, None, 0.0, 5.0).radial_ice_mm,
            0.0
        );
        assert!(
            freezing_rain_increment(2.54, 1.0, true, 274.0, Some(272.0), 0.0, 5.0).radial_ice_mm
                > 0.0
        );
        assert_eq!(fram_ice_liquid_ratio(0.0, 0.0, 0.0), 0.0);
        assert!(
            freezing_rain_increment(-1.0, 1.0, true, 270.0, None, -2.0, 5.0)
                .radial_ice_mm
                .is_nan()
        );
        assert!(fram_ice_liquid_ratio(1.0, f64::NAN, -2.0).is_nan());
    }
}
