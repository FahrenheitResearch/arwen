//! Radar diagnostic variables: dbz, maxdbz
//!
//! Simulated reflectivity from hydrometeor mixing ratios following
//! wrf-python's `wrf_user_dbz.f90` (`CALCDBZ`) subroutine.
//!
//! Uses constant intercept parameters (ivarint=0) and no bright-band
//! correction (iliqskin=0), matching the wrf-python defaults.

use crate::compute::ComputeOpts;
use crate::error::{WrfError, WrfResult};
use crate::file::WrfFile;
use rayon::prelude::*;

// --- Physical constants (wrf_constants) ---
const GAMMA_SEVEN: f64 = 720.0;
const PI: f64 = std::f64::consts::PI;
const RD: f64 = 287.0;
const CELKEL: f64 = 273.15;
const RHOWAT: f64 = 1000.0;
const ALPHA: f64 = 0.224; // |K_ice|^2 / |K_water|^2

// --- Hydrometeor densities (kg m^-3) ---
const RHO_R: f64 = 1000.0; // rain
const RHO_S: f64 = 100.0; // snow
const RHO_G: f64 = 400.0; // graupel

// --- Constant intercept parameters (m^-4) ---
const RN0_R: f64 = 8.0e6;
const RN0_S: f64 = 2.0e7;
const RN0_G: f64 = 4.0e6;

// --- Variable intercept parameters (Thompson microphysics) ---
const R1: f64 = 1.0e-15;
const RON: f64 = 8.0e6;
const RON2: f64 = 1.0e10;
const GON: f64 = 5.0e7;
const RON_MIN: f64 = 8.0e6;
const RON_QR0: f64 = 0.00010;
const RON_DELQR0: f64 = 0.25 * RON_QR0;
const RON_CONST1R: f64 = (RON2 - RON_MIN) * 0.5;
const RON_CONST2R: f64 = (RON2 + RON_MIN) * 0.5;

/// Match NumPy's `qs.any()` check used by wrf-python to set CALCDBZ's
/// `sn0` flag. An existing but entirely zero QSNOW field is treated the
/// same as a missing QSNOW field.
#[inline]
fn snow_field_present(qs: &[f64]) -> bool {
    qs.iter().any(|&value| value != 0.0)
}

#[inline]
fn clamp_negative_to_zero(value: f64) -> f64 {
    if value < 0.0 {
        0.0
    } else {
        value
    }
}

#[inline]
fn rain_and_snow_for_dbz(qrain: f64, qsnow: f64, sn0: bool, temperature_k: f64) -> (f64, f64) {
    let mut rain = clamp_negative_to_zero(qrain);
    let mut snow = clamp_negative_to_zero(qsnow);

    if !sn0 && temperature_k < CELKEL {
        snow = rain;
        rain = 0.0;
    }

    (rain, snow)
}

fn read_optional_hydrometeor(
    f: &WrfFile,
    name: &str,
    t: usize,
    field_len: usize,
) -> WrfResult<Vec<f64>> {
    if f.has_var(name) {
        f.read_var(name, t)
    } else {
        Ok(vec![0.0; field_len])
    }
}

fn validate_field_len(name: &str, actual: usize, expected: usize) -> WrfResult<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(WrfError::DimMismatch(format!(
            "{name} has {actual} values, expected {expected} for the WRF mass grid"
        )))
    }
}

#[derive(Clone, Copy)]
struct ReflectivityFactors {
    rain: f64,
    snow: f64,
    graupel: f64,
    bright_snow: f64,
    bright_graupel: f64,
}

impl ReflectivityFactors {
    fn new() -> Self {
        let rain = GAMMA_SEVEN * 1.0e18 * (1.0 / (PI * RHO_R)).powf(1.75);
        let snow = GAMMA_SEVEN
            * 1.0e18
            * (1.0 / (PI * RHO_S)).powf(1.75)
            * (RHO_S / RHOWAT).powi(2)
            * ALPHA;
        let graupel = GAMMA_SEVEN
            * 1.0e18
            * (1.0 / (PI * RHO_G)).powf(1.75)
            * (RHO_G / RHOWAT).powi(2)
            * ALPHA;
        Self {
            rain,
            snow,
            graupel,
            bright_snow: snow / ALPHA,
            bright_graupel: graupel / ALPHA,
        }
    }
}

#[derive(Clone, Copy)]
struct ReflectivityOptions {
    sn0: bool,
    use_varint: bool,
    use_liqskin: bool,
}

#[derive(Clone, Copy)]
struct ReflectivityPoint {
    pressure_pa: f64,
    temperature_k: f64,
    qvapor: f64,
    qrain: f64,
    qsnow: f64,
    qgraupel: f64,
}

#[inline]
fn dbz_at_point(
    point: ReflectivityPoint,
    options: ReflectivityOptions,
    factors: ReflectivityFactors,
) -> f64 {
    let qvp = clamp_negative_to_zero(point.qvapor);

    // Virtual temperature: full formula from CALCDBZ.
    let virtual_t = point.temperature_k * (0.622 + qvp) / (0.622 * (1.0 + qvp));
    let rhoair = point.pressure_pa / (RD * virtual_t);

    let (qra, qsn) =
        rain_and_snow_for_dbz(point.qrain, point.qsnow, options.sn0, point.temperature_k);
    let qgr = clamp_negative_to_zero(point.qgraupel);

    // Above freezing, the liquid-skin option makes frozen particles scatter
    // as liquid by dropping ALPHA from their factors.
    let (factor_s, factor_g) = if options.use_liqskin && point.temperature_k > CELKEL {
        (factors.bright_snow, factors.bright_graupel)
    } else {
        (factors.snow, factors.graupel)
    };

    let (ronv, sonv, gonv) = if options.use_varint {
        let temp_c = (point.temperature_k - CELKEL).min(-0.001);
        let sonv_v = (2.0e6 * (-0.12 * temp_c).exp()).min(2.0e8);

        let gonv_v = if qgr > R1 {
            let g = 2.38 * (PI * RHO_G / (rhoair * qgr)).powf(0.92);
            g.max(1.0e4).min(GON)
        } else {
            GON
        };

        let ronv_v = if qra > R1 {
            RON_CONST1R * ((RON_QR0 - qra) / RON_DELQR0).tanh() + RON_CONST2R
        } else {
            RON2
        };

        (ronv_v, sonv_v, gonv_v)
    } else {
        (RN0_R, RN0_S, RN0_G)
    };

    let z_r = factors.rain * (rhoair * qra).powf(1.75) / ronv.powf(0.75);
    let z_s = factor_s * (rhoair * qsn).powf(1.75) / sonv.powf(0.75);
    let z_g = factor_g * (rhoair * qgr).powf(1.75) / gonv.powf(0.75);

    // Rust and the wrf-python Fortran build both select the finite floor when
    // z_e is NaN, yielding -30 dBZ for non-finite input cells.
    let z_e = (z_r + z_s + z_g).max(0.001);
    10.0 * z_e.log10()
}

/// Simulated reflectivity (dBZ). `[nz, ny, nx]`
///
/// Matches wrf-python's `CALCDBZ` from `wrf_user_dbz.f90`.
/// Defaults: constant intercepts (ivarint=0), no bright-band (iliqskin=0).
/// Set `opts.use_varint=true` for Thompson variable intercepts.
/// Set `opts.use_liqskin=true` for bright-band correction.
///
/// When QSNOW is missing or entirely zero (sn0=0 behavior), rain mixing
/// ratio is reassigned to snow below freezing.
pub fn compute_dbz(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let tk = f.temperature(t)?;
    let pres = f.full_pressure(t)?;
    let qv = f.qvapor(t)?;
    let n = f.nxyz();

    // QRAIN is required by wrf-python. QSNOW and QGRAUP are optional, but
    // errors while reading an existing field must not be hidden as zeros.
    let qr = f.read_var("QRAIN", t)?;
    let qs = read_optional_hydrometeor(f, "QSNOW", t, n)?;
    let qg = read_optional_hydrometeor(f, "QGRAUP", t, n)?;

    for (name, actual) in [
        ("temperature", tk.len()),
        ("pressure", pres.len()),
        ("QVAPOR", qv.len()),
        ("QRAIN", qr.len()),
        ("QSNOW", qs.len()),
        ("QGRAUP", qg.len()),
    ] {
        validate_field_len(name, actual, n)?;
    }

    let sn0 = snow_field_present(&qs);
    let factors = ReflectivityFactors::new();
    let options = ReflectivityOptions {
        sn0,
        use_varint: opts.use_varint.unwrap_or(false),
        use_liqskin: opts.use_liqskin.unwrap_or(false),
    };

    let dbz: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|i| {
            dbz_at_point(
                ReflectivityPoint {
                    pressure_pa: pres[i],
                    temperature_k: tk[i],
                    qvapor: qv[i],
                    qrain: qr[i],
                    qsnow: qs[i],
                    qgraupel: qg[i],
                },
                options,
                factors,
            )
        })
        .collect();

    Ok(dbz)
}

/// Maximum (composite) reflectivity (dBZ). `[ny, nx]`
pub fn compute_maxdbz(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let dbz_3d = compute_dbz(f, t, opts)?;
    let nxy = f.nxy();
    let nz = f.nz;

    let mut maxdbz = vec![f64::NEG_INFINITY; nxy];
    for k in 0..nz {
        let offset = k * nxy;
        for ij in 0..nxy {
            let val = dbz_3d[offset + ij];
            if val > maxdbz[ij] {
                maxdbz[ij] = val;
            }
        }
    }
    Ok(maxdbz)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_point_matches_wrf_python(
        point: ReflectivityPoint,
        sn0: bool,
        use_varint: bool,
        use_liqskin: bool,
        expected: f64,
    ) {
        let actual = dbz_at_point(
            point,
            ReflectivityOptions {
                sn0,
                use_varint,
                use_liqskin,
            },
            ReflectivityFactors::new(),
        );
        assert!(
            (actual - expected).abs() < 2.0e-12,
            "expected {expected:.15}, got {actual:.15}"
        );
    }

    #[test]
    fn all_zero_qsnow_uses_wrf_python_no_snow_path() {
        let qs = [0.0, -0.0, 0.0];
        let sn0 = snow_field_present(&qs);
        assert!(!sn0);

        let (rain, snow) = rain_and_snow_for_dbz(1.0e-3, 0.0, sn0, 263.15);
        assert_eq!(rain, 0.0);
        assert_eq!(snow, 1.0e-3);
    }

    #[test]
    fn any_nonzero_qsnow_keeps_separate_species() {
        let qs = [0.0, 1.0e-12, 0.0];
        let sn0 = snow_field_present(&qs);
        assert!(sn0);

        let (rain, snow) = rain_and_snow_for_dbz(1.0e-3, 0.0, sn0, 263.15);
        assert_eq!(rain, 1.0e-3);
        assert_eq!(snow, 0.0);
    }

    #[test]
    fn radar_gas_constant_matches_wrf_python() {
        assert_eq!(RD, 287.0);
    }

    #[test]
    fn negative_mixing_ratios_are_zeroed_but_nan_is_preserved() {
        assert_eq!(clamp_negative_to_zero(-1.0e-6), 0.0);
        assert_eq!(clamp_negative_to_zero(1.0e-6), 1.0e-6);
        assert!(clamp_negative_to_zero(f64::NAN).is_nan());
    }

    #[test]
    fn point_formula_matches_wrf_python_for_all_option_combinations() {
        // Frozen outputs from NCAR wrf-python 1.3.4.1's _dbz wrapper.
        let point = ReflectivityPoint {
            pressure_pa: 90_000.0,
            temperature_k: 280.0,
            qvapor: 0.01,
            qrain: 0.001,
            qsnow: 0.0003,
            qgraupel: 0.0005,
        };
        for (use_varint, use_liqskin, expected) in [
            (false, false, 44.316_654_841_042_38),
            (false, true, 45.475_004_486_572_445),
            (true, false, 44.761_410_567_933_34),
            (true, true, 46.838_110_502_884_554),
        ] {
            assert_point_matches_wrf_python(point, true, use_varint, use_liqskin, expected);
        }
    }

    #[test]
    fn point_formula_matches_wrf_python_for_no_snow_and_nan_paths() {
        let cold_rain = ReflectivityPoint {
            pressure_pa: 80_000.0,
            temperature_k: 260.0,
            qvapor: 0.001,
            qrain: 0.001,
            qsnow: 0.0,
            qgraupel: 0.0,
        };
        assert_point_matches_wrf_python(cold_rain, false, false, false, 31.642_452_669_493_88);
        assert_point_matches_wrf_python(cold_rain, true, false, false, 43.624_522_551_192_54);

        let mut non_finite = cold_rain;
        non_finite.qvapor = f64::NAN;
        assert_eq!(
            dbz_at_point(
                non_finite,
                ReflectivityOptions {
                    sn0: true,
                    use_varint: false,
                    use_liqskin: false,
                },
                ReflectivityFactors::new(),
            ),
            -30.0
        );
    }

    #[test]
    fn field_length_mismatch_is_an_error() {
        assert!(validate_field_len("QRAIN", 3, 4).is_err());
        assert!(validate_field_len("QRAIN", 4, 4).is_ok());
    }
}
