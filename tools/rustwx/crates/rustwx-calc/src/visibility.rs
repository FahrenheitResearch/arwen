//! Stoelinga-Warner hydrometeor extinction plus the NOAA GSD/RUC RH limit.
//! Concentrations: g/m3. Extinction: /km. Output: metres, 2% contrast.
//! Lowest three mass-level species maxima, lowest two water-relative RH maxima.
//! Missing optional species mean absent species; bad present values mask output.
use crate::{CalcError, GridShape};
use rayon::prelude::*;
pub const VISIBILITY_CAP_M: f64 = 90000.0;
pub const FOG_THRESHOLD_M: f64 = 1000.0;
pub const DENSE_FOG_THRESHOLD_M: f64 = 400.0;
pub const CONTRAST: f64 = 3.912023005428146;
#[derive(Clone, Copy)]
pub struct VisibilityInputs<'a> {
    pub grid: GridShape,
    pub nz: usize,
    pub pressure_pa: &'a [f64],
    pub temperature_k: &'a [f64],
    pub vapor_kgkg: &'a [f64],
    pub cloud_kgkg: &'a [f64],
    pub rain_kgkg: Option<&'a [f64]>,
    pub ice_kgkg: Option<&'a [f64]>,
    pub snow_kgkg: Option<&'a [f64]>,
    pub graupel_kgkg: Option<&'a [f64]>,
}
#[derive(Debug)]
pub struct VisibilityOutputs {
    pub visibility_m: Vec<f64>,
    /// One iff visibility <1000 m, zero otherwise. Unknown stays NaN.
    /// Low visibility alone does not identify the obscurant as fog.
    pub fog_flag: Vec<f64>,
}
pub fn rh_limit_m(rh: f64) -> f64 {
    if !rh.is_finite() {
        return f64::NAN;
    }
    VISIBILITY_CAP_M * (-2.5 * (rh / 100.0 - 0.15).clamp(0.0, 0.8)).exp()
}
pub fn relative_humidity(p: f64, t: f64, r: f64) -> f64 {
    let x = t - 273.15;
    // NOAA GSD water saturation-pressure polynomial, hPa.
    let coeff = [
        0.99999683,
        -0.0090826951,
        0.000078736169,
        -0.00000061117958,
        4.3884187e-9,
        -2.9883885e-11,
        2.1874425e-13,
        -1.7892321e-15,
        1.1112018e-17,
        -3.0994571e-20,
    ];
    let pol = coeff.into_iter().rev().fold(0.0, |v, c| v * x + c);
    let es = 6.1078 / pol.powi(8);
    (100.0 * (p / 100.0 * r / (0.62197 + r)) / es).clamp(0.0, 100.0)
}
fn valid(v: f64) -> bool {
    v.is_finite() && v.abs() < 1e30
}
pub fn compute_visibility(x: VisibilityInputs<'_>) -> Result<VisibilityOutputs, CalcError> {
    if x.nz == 0 {
        return Err(CalcError::InvalidConfig {
            field: "nz",
            reason: "must be positive",
        });
    }
    let n = x.grid.len();
    let count = n.checked_mul(x.nz).ok_or(CalcError::InvalidConfig {
        field: "visibility shape",
        reason: "overflow",
    })?;
    let species = [
        Some(x.cloud_kgkg),
        x.rain_kgkg,
        x.ice_kgkg,
        x.snow_kgkg,
        x.graupel_kgkg,
    ];
    for (name, a) in [
        ("pressure_pa", Some(x.pressure_pa)),
        ("temperature_k", Some(x.temperature_k)),
        ("vapor_kgkg", Some(x.vapor_kgkg)),
        ("cloud_kgkg", species[0]),
        ("rain_kgkg", species[1]),
        ("ice_kgkg", species[2]),
        ("snow_kgkg", species[3]),
        ("graupel_kgkg", species[4]),
    ] {
        if let Some(a) = a {
            if a.len() != count {
                return Err(CalcError::LengthMismatch {
                    field: name,
                    expected: count,
                    actual: a.len(),
                });
            }
        }
    }
    let rows: Vec<(f64, f64)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut maxq = [0.0f64; 5];
            let mut rh = 0.0f64;
            for k in 0..x.nz.min(3) {
                let j = k * n + i;
                let p = x.pressure_pa[j];
                let t = x.temperature_k[j];
                let q = x.vapor_kgkg[j];
                if !valid(p)
                    || !valid(t)
                    || !valid(q)
                    || p <= 0.0
                    || !(150.0..=350.0).contains(&t)
                    || q < 0.0
                {
                    return (f64::NAN, f64::NAN);
                }
                if k < 2 {
                    rh = rh.max(relative_humidity(p, t, q));
                }
                for (s, a) in species.iter().enumerate() {
                    if let Some(a) = a {
                        if !valid(a[j]) {
                            return (f64::NAN, f64::NAN);
                        }
                        maxq[s] = maxq[s].max(a[j].max(0.0));
                    }
                }
            }
            let q = x.vapor_kgkg[i];
            let tv = x.temperature_k[i] * (1.0 + q / 0.62197) / (1.0 + q);
            let rho = x.pressure_pa[i] / (287.0 * tv);
            let volume = (1.0 + q) / rho
                + (maxq[0] + maxq[1]) / 1000.0
                + (maxq[2] + maxq[3] + maxq[4]) / 917.0;
            let c = maxq.map(|q| 1000.0 * q / volume);
            let beta = 144.7 * c[0].powf(0.88)
                + 2.24 * c[1].powf(0.75)
                + 327.8 * c[2]
                + 10.36 * c[3].powf(0.7776)
                + 8.0 * c[4].powf(0.75)
                + 1e-10;
            let v = (1000.0 * CONTRAST / beta)
                .min(rh_limit_m(rh))
                .min(VISIBILITY_CAP_M);
            (v, if v < FOG_THRESHOLD_M { 1.0 } else { 0.0 })
        })
        .collect();
    let (visibility_m, fog_flag) = rows.into_iter().unzip();
    Ok(VisibilityOutputs {
        visibility_m,
        fog_flag,
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn saturated_clear_air_is_not_dense_fog() {
        assert!((rh_limit_m(100.0) - 12180.17549129514).abs() < 1e-8);
        assert_eq!(rh_limit_m(0.0), 90000.0);
        assert!(rh_limit_m(f64::NAN).is_nan());
    }
    #[test]
    fn species_maximum_and_bad_present_carrier() {
        let grid = GridShape::new(2, 1).unwrap();
        let p = [100000.0; 6];
        let t = [280.0; 6];
        let q = [0.006; 6];
        let cloud = [0.0, 0.0, 0.0001, 0.0, 0.0, 0.0];
        let mut x = VisibilityInputs {
            grid,
            nz: 3,
            pressure_pa: &p,
            temperature_k: &t,
            vapor_kgkg: &q,
            cloud_kgkg: &cloud,
            rain_kgkg: None,
            ice_kgkg: None,
            snow_kgkg: None,
            graupel_kgkg: None,
        };
        let v = compute_visibility(x).unwrap();
        assert!(v.visibility_m[0] < 400.0);
        assert_eq!(v.fog_flag, [1.0, 0.0]);
        let bad = [f64::NAN; 6];
        x.rain_kgkg = Some(&bad);
        assert!(
            compute_visibility(x)
                .unwrap()
                .visibility_m
                .iter()
                .all(|v| v.is_nan())
        );
        x.rain_kgkg = Some(&[0.0]);
        assert!(compute_visibility(x).is_err());
    }
}
