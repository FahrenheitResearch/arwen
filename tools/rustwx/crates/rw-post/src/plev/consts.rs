//! Physical constants.  The dynamical core's values (WRF ARW technical note
//! [R1]) so post-processed heights and pressures stay consistent with the
//! forecast that wrote the history.  The CUDA kernels carry the same
//! literals (kernels/group_b.cu); a test pins the two lists together.

/// Gravity (m s-2), [R1].
pub const G: f64 = 9.81;
/// Dry-air gas constant (J kg-1 K-1), [R1].
pub const RD: f64 = 287.0;
/// Water-vapour gas constant (J kg-1 K-1), [R1].
pub const RV: f64 = 461.6;
/// Specific heat at constant pressure, 7 Rd / 2, [R1].
pub const CP: f64 = 7.0 * RD / 2.0;
/// Reference pressure (Pa), [R1].
pub const P0: f64 = 100000.0;
/// Rd / Rv.
pub const EPS: f64 = RD / RV;
/// Rd / cp.
pub const KAPPA: f64 = RD / CP;
/// Base potential temperature of the perturbation `T` carrier (K), [R1].
pub const THETA_BASE: f64 = 300.0;
/// Standard-atmosphere lapse rate (K m-1), [R13].
pub const GAMMA: f64 = 0.0065;
/// Lapse rate in pressure form, gamma Rd / g: T is proportional to
/// p^ALPHA along the standard lapse rate.
pub const ALPHA: f64 = GAMMA * RD / G;
/// 0 C in kelvin.
pub const T_MELT: f64 = 273.15;

/// Improved Magnus form over liquid water [R6]: e_w = A exp(B Tc / (Tc + C)) Pa.
pub const MAGNUS_A: f64 = 610.94;
pub const MAGNUS_B: f64 = 17.625;
pub const MAGNUS_C: f64 = 243.04;

/// Specific-humidity floor (kg kg-1) used only to keep vapour pressure and
/// dewpoint finite in a bone-dry cell.
pub const Q_FLOOR: f64 = 1.0e-10;

/// NCAR/TN-396 [R12] high-terrain and warm-column constants.
pub const TN396_Z_LOW: f64 = 2000.0;
pub const TN396_Z_HIGH: f64 = 2500.0;
pub const TN396_T_PLATEAU: f64 = 298.0;
pub const TN396_T_WARM: f64 = 290.5;
pub const TN396_T_COLD: f64 = 255.0;

/// The NMC reduction's warm-temperature threshold and coefficient as
/// published in [R15]: 290.66 K and 0.005 K-1.
pub const NMC_T_WARM: f64 = 290.66;
pub const NMC_WARM_COEF: f64 = 0.005;

/// MAPS reference pressure [R17] (Pa).
pub const MAPS_P_REF: f64 = 70000.0;

/// Membrane reduction levels [R14]: 300 to 1000 hPa by 25 hPa (Pa),
/// ascending pressure.  Fixed so the reduction does not depend on which
/// output levels a request asks for.
pub fn membrane_levels() -> Vec<f64> {
    (0..29).map(|i| 30000.0 + 2500.0 * i as f64).collect()
}

/// Membrane solve schedule: Jacobi sweeps at the coarsest grid and at every
/// finer grid of the cascade (fixed counts, no convergence test).
pub const MEMBRANE_COARSEST_MAX_DIM: usize = 32;
pub const MEMBRANE_COARSE_SWEEPS: usize = 1000;
pub const MEMBRANE_LEVEL_SWEEPS: usize = 40;

/// MAPS smoothing scale (m): passes = round(2 (L / DX)^2), the number of
/// 1-2-1 passes whose combined response has a standard deviation of L
/// (each pass adds DX^2 / 2 of variance per direction).
pub const MAPS_SMOOTH_SCALE_M: f64 = 15000.0;
pub const MAPS_SMOOTH_MAX_PASSES: usize = 1000;

/// The 1-2-1 pass count for a native grid spacing `dx` (m).
pub fn maps_smoothing_passes(dx: f64) -> usize {
    if !(dx > 0.0) {
        return 0;
    }
    let r = MAPS_SMOOTH_SCALE_M / dx;
    let n = (2.0 * r * r).round();
    if n > MAPS_SMOOTH_MAX_PASSES as f64 { MAPS_SMOOTH_MAX_PASSES } else { n as usize }
}
