// Physical constants mirroring MetPy's metpy.constants module.
//
// All values use SI base units unless otherwise noted. Sources follow
// the same references as MetPy: CODATA 2018, the U.S. Standard
// Atmosphere (1976), and the WMO International Meteorological Tables.
//
// Derived constants follow MetPy's derivation chain:
//   Rd = R / Md,  Rv = R / Mw,  kappa = 2/7,  Cp_d = Rd / kappa,
//   Cv_d = Cp_d - Rd,  epsilon = Mw / Md

// ---------------------------------------------------------------------------
// Universal
// ---------------------------------------------------------------------------

/// Universal gas constant (J mol^-1 K^-1).
pub const R: f64 = 8.314462618;

/// Stefan-Boltzmann constant (W m^-2 K^-4).
pub const STEFAN_BOLTZMANN: f64 = 5.670374419e-8;

/// Newtonian gravitational constant (m^3 kg^-1 s^-2).
pub const GRAVITATIONAL_CONSTANT: f64 = 6.6743e-11;

// ---------------------------------------------------------------------------
// Earth
// ---------------------------------------------------------------------------

/// Mean radius of the Earth (m).
pub const EARTH_AVG_RADIUS: f64 = 6_371_008.7714;

/// Standard acceleration of gravity (m s^-2).
pub const EARTH_GRAVITY: f64 = 9.80665;

/// Angular velocity of Earth's rotation (rad s^-1).
pub const OMEGA: f64 = 7.292115e-5;

/// Mean density of the Earth (kg m^-3).
/// Derived from Earth's mass (5.9722e24 kg) and mean radius.
pub const EARTH_AVG_DENSITY: f64 = 5515.0;

/// Maximum solar declination angle (degrees).
pub const EARTH_MAX_DECLINATION: f64 = 23.45;

/// Geocentric gravitational constant GM (m^3 s^-2).
pub const EARTH_GM: f64 = 3.986005e14;

/// Mass of the Earth (kg).
pub const EARTH_MASS: f64 = 5.972169366075844e24;

/// Eccentricity of Earth's orbit (dimensionless).
pub const EARTH_ORBIT_ECCENTRICITY: f64 = 0.0167;

/// Mean distance from Earth's surface to the Sun (m).
pub const EARTH_SFC_AVG_DIST_SUN: f64 = 1.495978707e11;

/// Total solar irradiance (W m^-2).
pub const EARTH_SOLAR_IRRADIANCE: f64 = 1360.8;

// ---------------------------------------------------------------------------
// Dry air
// ---------------------------------------------------------------------------

/// Mean molecular weight of dry air (kg mol^-1).
pub const MOLECULAR_WEIGHT_DRY_AIR: f64 = 0.02896546;

/// Specific gas constant for dry air (J kg^-1 K^-1).
/// Derived as R / Md.
pub const RD: f64 = R / MOLECULAR_WEIGHT_DRY_AIR;

/// Poisson constant for dry air (dimensionless).
/// MetPy uses the exact fraction 2/7.
pub const KAPPA: f64 = 2.0 / 7.0;

/// Specific heat at constant pressure for dry air (J kg^-1 K^-1).
/// Derived as Rd / kappa.
pub const CP_D: f64 = RD / KAPPA;

/// Specific heat at constant volume for dry air (J kg^-1 K^-1).
/// Derived as Cp_d - Rd.
pub const CV_D: f64 = CP_D - RD;

/// Density of dry air at STP (kg m^-3).
/// Calculated as P_stp / (Rd * T_stp).
pub const RHO_D_STP: f64 = P_STP / (RD * T_STP);

/// Ratio of the molecular weight of water to the molecular weight of dry
/// air, also equal to Rd / Rv (dimensionless).
pub const EPSILON: f64 = MOLECULAR_WEIGHT_WATER / MOLECULAR_WEIGHT_DRY_AIR;

/// Specific heat ratio for dry air (Cp_d / Cv_d, dimensionless).
pub const DRY_AIR_SPEC_HEAT_RATIO: f64 = CP_D / CV_D;

/// Dry adiabatic lapse rate (K m^-1).
/// Derived as g / Cp_d.
pub const DRY_ADIABATIC_LAPSE_RATE: f64 = EARTH_GRAVITY / CP_D;

// ---------------------------------------------------------------------------
// Water / moist thermodynamics
// ---------------------------------------------------------------------------

/// Mean molecular weight of water (kg mol^-1).
pub const MOLECULAR_WEIGHT_WATER: f64 = 0.018015268;

/// Specific gas constant for water vapour (J kg^-1 K^-1).
/// Derived as R / Mw.
pub const RV: f64 = R / MOLECULAR_WEIGHT_WATER;

/// Specific heat at constant pressure for water vapour (J kg^-1 K^-1).
pub const CP_V: f64 = 1860.078011865639;

/// Specific heat at constant volume for water vapour (J kg^-1 K^-1).
pub const CV_V: f64 = 1398.554896139578;

/// Density of liquid water at 0 degC (kg m^-3).
pub const RHO_L: f64 = 999.97495;

/// Density of ice (kg m^-3).
pub const RHO_I: f64 = 917.0;

/// Latent heat of vaporisation at 0 degC (J kg^-1).
pub const LV: f64 = 2_500_840.0;

/// Latent heat of fusion at 0 degC (J kg^-1).
pub const LF: f64 = 333_700.0;

/// Latent heat of sublimation at 0 degC (J kg^-1).
pub const LS: f64 = 2_834_540.0;

/// Specific heat of liquid water at 0 degC (J kg^-1 K^-1).
pub const CP_L: f64 = 4219.4;

/// Specific heat of ice (J kg^-1 K^-1).
pub const CP_I: f64 = 2090.0;

/// Freezing point of water (K).
pub const T_FREEZE: f64 = 273.15;

/// Triple point temperature of water (K).
pub const WATER_TRIPLE_POINT_TEMPERATURE: f64 = 273.16;

/// Saturation water vapour pressure at 0 degC (Pa).
pub const SAT_PRESSURE_0C: f64 = 611.2;

/// Specific heat ratio for water vapour (Cp_v / Cv_v, dimensionless).
pub const WV_SPECIFIC_HEAT_RATIO: f64 = CP_V / CV_V;

// ---------------------------------------------------------------------------
// Standard atmosphere reference values
// ---------------------------------------------------------------------------

/// Standard atmospheric pressure at sea level (Pa).
pub const P_STP: f64 = 101_325.0;

/// Standard temperature at sea level (K).
pub const T_STP: f64 = 288.15;

/// Reference pressure for potential temperature (Pa).
/// MetPy uses 1000 hPa = 100000 Pa.
pub const POT_TEMP_REF_PRESS: f64 = 100_000.0;

// ---------------------------------------------------------------------------
// MetPy-compatible named constants
// ---------------------------------------------------------------------------
// These mirror the names MetPy exposes (e.g. `mpconsts.water_heat_vaporization`).
// Where a value duplicates an existing constant the compiler will fold them.

/// Latent heat of vaporisation at 0 degC (J kg^-1). Same as `LV`.
pub const WATER_HEAT_VAPORIZATION: f64 = LV;

/// Latent heat of fusion at 0 degC (J kg^-1). Same as `LF`.
pub const WATER_HEAT_FUSION: f64 = LF;

/// Latent heat of sublimation at 0 degC (J kg^-1). Same as `LS`.
pub const WATER_HEAT_SUBLIMATION: f64 = LS;

/// Specific heat of liquid water at 0 degC (J kg^-1 K^-1). Same as `CP_L`.
pub const WATER_SPECIFIC_HEAT_LIQUID: f64 = CP_L;

/// Specific heat of water vapour at constant pressure (J kg^-1 K^-1). Same as `CP_V`.
pub const WATER_SPECIFIC_HEAT_VAPOR: f64 = CP_V;

/// Mean molecular weight of water (kg mol^-1). Same as `MOLECULAR_WEIGHT_WATER`.
pub const WATER_MOLECULAR_WEIGHT: f64 = MOLECULAR_WEIGHT_WATER;

/// Mean molecular weight of dry air (kg mol^-1). Same as `MOLECULAR_WEIGHT_DRY_AIR`.
pub const DRY_AIR_MOLECULAR_WEIGHT: f64 = MOLECULAR_WEIGHT_DRY_AIR;

/// Specific heat at constant pressure for dry air (J kg^-1 K^-1). Same as `CP_D`.
pub const DRY_AIR_SPEC_HEAT_PRESS: f64 = CP_D;

/// Specific heat at constant volume for dry air (J kg^-1 K^-1). Same as `CV_D`.
pub const DRY_AIR_SPEC_HEAT_VOL: f64 = CV_D;

/// Density of dry air at STP (kg m^-3).
pub const DRY_AIR_DENSITY_STP: f64 = RHO_D_STP;

/// Specific gas constant for dry air (J kg^-1 K^-1). Same as `RD`.
pub const DRY_AIR_GAS_CONSTANT: f64 = RD;

/// Specific gas constant for water vapour (J kg^-1 K^-1). Same as `RV`.
pub const WATER_GAS_CONSTANT: f64 = RV;

/// NOAA mean radius of the Earth (m). Same as `EARTH_AVG_RADIUS`.
pub const NOAA_MEAN_EARTH_RADIUS: f64 = EARTH_AVG_RADIUS;

/// Standard acceleration of gravity (m s^-2). Same as `EARTH_GRAVITY`.
pub const EARTH_GRAVITATIONAL_ACCELERATION: f64 = EARTH_GRAVITY;

/// Poisson exponent for dry air (Rd / Cp_d, dimensionless). Same as `KAPPA`.
pub const POISSON_EXPONENT_DRY_AIR: f64 = KAPPA;

// ---------------------------------------------------------------------------
// MetPy short aliases (mpconsts.g, mpconsts.Rd, etc.)
// ---------------------------------------------------------------------------

#[allow(non_upper_case_globals)]
/// Standard acceleration of gravity (m s^-2). Alias for `EARTH_GRAVITY`.
pub const g: f64 = EARTH_GRAVITY;

#[allow(non_upper_case_globals)]
/// Specific gas constant for dry air (J kg^-1 K^-1). Alias for `RD`.
pub const Rd: f64 = RD;

#[allow(non_upper_case_globals)]
/// Specific gas constant for water vapour (J kg^-1 K^-1). Alias for `RV`.
pub const Rv: f64 = RV;

#[allow(non_upper_case_globals)]
/// Specific heat at constant pressure for dry air (J kg^-1 K^-1). Alias for `CP_D`.
pub const Cp_d: f64 = CP_D;

#[allow(non_upper_case_globals)]
/// Specific heat at constant volume for dry air (J kg^-1 K^-1). Alias for `CV_D`.
pub const Cv_d: f64 = CV_D;

#[allow(non_upper_case_globals)]
/// Latent heat of vaporisation at 0 degC (J kg^-1). Alias for `LV`.
pub const Lv: f64 = LV;

#[allow(non_upper_case_globals)]
/// Latent heat of fusion at 0 degC (J kg^-1). Alias for `LF`.
pub const Lf: f64 = LF;

#[allow(non_upper_case_globals)]
/// Latent heat of sublimation at 0 degC (J kg^-1). Alias for `LS`.
pub const Ls: f64 = LS;

#[allow(non_upper_case_globals)]
/// Mean radius of the Earth (m). Alias for `EARTH_AVG_RADIUS`.
pub const Re: f64 = EARTH_AVG_RADIUS;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: assert approximate equality within a relative tolerance.
    fn approx_eq(a: f64, b: f64, rel_tol: f64) {
        let diff = (a - b).abs();
        let magnitude = a.abs().max(b.abs()).max(1e-30);
        assert!(
            diff / magnitude < rel_tol,
            "approx_eq failed: {a} vs {b} (diff {diff}, rel_tol {rel_tol})"
        );
    }

    #[test]
    fn test_earth_constants() {
        approx_eq(EARTH_AVG_RADIUS, 6_371_008.7714, 1e-12);
        assert_eq!(EARTH_GRAVITY, 9.80665);
        assert_eq!(OMEGA, 7.292115e-5);
        assert_eq!(EARTH_AVG_DENSITY, 5515.0);
        assert_eq!(EARTH_MAX_DECLINATION, 23.45);
    }

    #[test]
    fn test_new_earth_constants() {
        assert_eq!(GRAVITATIONAL_CONSTANT, 6.6743e-11);
        assert_eq!(EARTH_GM, 3.986005e14);
        approx_eq(EARTH_MASS, 5.972169366075844e24, 1e-12);
        assert_eq!(EARTH_ORBIT_ECCENTRICITY, 0.0167);
        assert_eq!(EARTH_SFC_AVG_DIST_SUN, 1.495978707e11);
        assert_eq!(EARTH_SOLAR_IRRADIANCE, 1360.8);
    }

    #[test]
    fn test_dry_air_constants() {
        // MetPy derives Rd = R / Md
        approx_eq(RD, 287.04749097718457, 1e-12);
        // MetPy derives Cp_d = Rd / kappa where kappa = 2/7
        approx_eq(CP_D, 1004.6662184201462, 1e-12);
        // MetPy derives Cv_d = Cp_d - Rd
        approx_eq(CV_D, 717.6187274429616, 1e-12);
        approx_eq(KAPPA, 2.0 / 7.0, 1e-15);
        assert_eq!(MOLECULAR_WEIGHT_DRY_AIR, 0.02896546);
    }

    #[test]
    fn test_epsilon_ratio() {
        // epsilon = Mw / Md, which should also approximate Rd / Rv
        let rd_rv = RD / RV;
        approx_eq(EPSILON, rd_rv, 1e-12);
        approx_eq(EPSILON, 0.6219569100577033, 1e-12);
    }

    #[test]
    fn test_water_constants() {
        approx_eq(RV, 461.52311572606084, 1e-12);
        approx_eq(CP_V, 1860.078011865639, 1e-12);
        approx_eq(CV_V, 1398.554896139578, 1e-12);
        approx_eq(RHO_L, 999.97495, 1e-12);
        assert_eq!(RHO_I, 917.0);
        assert_eq!(LV, 2_500_840.0);
        assert_eq!(LF, 333_700.0);
        assert_eq!(LS, 2_834_540.0);
        approx_eq(CP_L, 4219.4, 1e-12);
        assert_eq!(CP_I, 2090.0);
        assert_eq!(T_FREEZE, 273.15);
        assert_eq!(WATER_TRIPLE_POINT_TEMPERATURE, 273.16);
        assert_eq!(SAT_PRESSURE_0C, 611.2);
        assert_eq!(MOLECULAR_WEIGHT_WATER, 0.018015268);
    }

    #[test]
    fn test_standard_atmosphere() {
        assert_eq!(P_STP, 101_325.0);
        assert_eq!(T_STP, 288.15);
        assert_eq!(POT_TEMP_REF_PRESS, 100_000.0);
    }

    #[test]
    fn test_rho_d_stp() {
        // rho = P / (Rd * T)
        let expected = 101_325.0 / (RD * 288.15);
        approx_eq(RHO_D_STP, expected, 1e-10);
    }

    #[test]
    fn test_universal_constants() {
        approx_eq(R, 8.314462618, 1e-10);
        approx_eq(STEFAN_BOLTZMANN, 5.670374419e-8, 1e-10);
    }

    #[test]
    fn test_specific_heat_ratios() {
        approx_eq(DRY_AIR_SPEC_HEAT_RATIO, 1.4, 1e-12);
        approx_eq(WV_SPECIFIC_HEAT_RATIO, 1.33, 1e-2);
    }

    #[test]
    fn test_dry_adiabatic_lapse_rate() {
        approx_eq(DRY_ADIABATIC_LAPSE_RATE, 0.009761102563417645, 1e-12);
    }

    #[test]
    fn test_named_water_constants() {
        assert_eq!(WATER_HEAT_VAPORIZATION, LV);
        assert_eq!(WATER_HEAT_FUSION, LF);
        assert_eq!(WATER_HEAT_SUBLIMATION, LS);
        assert_eq!(WATER_SPECIFIC_HEAT_LIQUID, CP_L);
        assert_eq!(WATER_SPECIFIC_HEAT_VAPOR, CP_V);
        assert_eq!(WATER_MOLECULAR_WEIGHT, MOLECULAR_WEIGHT_WATER);
        assert_eq!(WATER_GAS_CONSTANT, RV);
    }

    #[test]
    fn test_named_dry_air_constants() {
        assert_eq!(DRY_AIR_MOLECULAR_WEIGHT, MOLECULAR_WEIGHT_DRY_AIR);
        assert_eq!(DRY_AIR_SPEC_HEAT_PRESS, CP_D);
        assert_eq!(DRY_AIR_SPEC_HEAT_VOL, CV_D);
        assert_eq!(DRY_AIR_GAS_CONSTANT, RD);
        assert_eq!(DRY_AIR_DENSITY_STP, RHO_D_STP);
    }

    #[test]
    fn test_named_earth_constants() {
        assert_eq!(NOAA_MEAN_EARTH_RADIUS, EARTH_AVG_RADIUS);
        assert_eq!(EARTH_GRAVITATIONAL_ACCELERATION, EARTH_GRAVITY);
    }

    #[test]
    fn test_poisson_exponent() {
        approx_eq(POISSON_EXPONENT_DRY_AIR, KAPPA, 1e-15);
        approx_eq(POISSON_EXPONENT_DRY_AIR, 2.0 / 7.0, 1e-15);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_metpy_short_aliases() {
        assert_eq!(g, EARTH_GRAVITY);
        assert_eq!(Rd, RD);
        assert_eq!(Rv, RV);
        assert_eq!(Cp_d, CP_D);
        assert_eq!(Cv_d, CV_D);
        assert_eq!(Lv, LV);
        assert_eq!(Lf, LF);
        assert_eq!(Ls, LS);
        assert_eq!(Re, EARTH_AVG_RADIUS);
    }
}
