//! Instantaneous lightning proxies, not observed flashes or probabilities.
//! LTG1/2/3 follow the 2009 graupel-flux/ice-mass calibration; LTG1 uses
//! graupel in g/kg (also documented in doi:10.1029/2023JD039492, section 2.6).
//! Outputs are flashes km^-2 (5 min)^-1.
//! PR92 follows WRF's domain-wide maximum-updraft implementation, including
//! allocation at the widest reflectivity level and conversion to flashes/s.
//! References and redistribution notices: DERIVED-DIAGNOSTICS-NOTICES.txt.
use crate::derived_heat::ThermoColumns;
use rayon::prelude::*;

pub struct LightningThreat {
    pub ltg1: Vec<f64>,
    pub ltg2: Vec<f64>,
    pub ltg3: Vec<f64>,
}
/// All volumes are level-major native mass levels, mixing ratios kg/kg dry air.
/// Missing ice species must be an error upstream, never assumed to be zero.
pub fn mc_caul(
    c: ThermoColumns<'_>,
    w: &[f64],
    graupel: &[f64],
    snow: &[f64],
    ice: &[f64],
) -> Result<LightningThreat, String> {
    c.validate()?;
    if [w, graupel, snow, ice]
        .iter()
        .any(|a| a.len() != c.cells * c.nz)
    {
        return Err("lightning volumes must match native [nz,ny,nx]".into());
    }
    let pairs: Vec<(f64, f64)> = (0..c.cells)
        .into_par_iter()
        .map(|i| {
            let mut vii = 0.0;
            let mut flux = None;
            for k in 0..c.nz {
                let j = k * c.cells + i;
                let p = c.pressure_pa[j];
                let t = c.temperature_k[j];
                let r = c.vapor_kgkg[j];
                let dz =
                    c.interface_agl_m[(k + 1) * c.cells + i] - c.interface_agl_m[k * c.cells + i];
                let vals = [p, t, r, dz, w[j], graupel[j], snow[j], ice[j]];
                if vals.iter().any(|v| !v.is_finite())
                    || p <= 0.0
                    || t <= 0.0
                    || r < 0.0
                    || dz <= 0.0
                    || [graupel[j], snow[j], ice[j]].iter().any(|&v| v < 0.0)
                {
                    return (f64::NAN, f64::NAN);
                }
                // Hydrometeors are mixing ratios per dry air mass.
                let rho_d = p / (287.04 * t * (1.0 + r / 0.622));
                vii += rho_d * (graupel[j] + snow[j] + ice[j]) * dz;
                if k > 0 && flux.is_none() {
                    let j0 = j - c.cells;
                    let t0 = c.temperature_k[j0];
                    if t0 >= 258.15 && t <= 258.15 && t0 > t {
                        let f = (t0 - 258.15) / (t0 - t);
                        let ww = w[j0] + f * (w[j] - w[j0]);
                        let qg = graupel[j0] + f * (graupel[j] - graupel[j0]);
                        flux = Some(ww.max(0.0) * qg * 1000.0);
                    }
                }
            }
            // An entirely warm column has no mixed-phase lightning. A cold
            // truncated column without the isotherm cannot support LTG1.
            let f = flux.unwrap_or_else(|| {
                if (0..c.nz).all(|k| c.temperature_k[k * c.cells + i] > 258.15) {
                    0.0
                } else {
                    f64::NAN
                }
            });
            (0.042 * f, 0.20 * vii)
        })
        .collect();
    Ok(LightningThreat {
        ltg1: pairs.iter().map(|p| p.0).collect(),
        ltg2: pairs.iter().map(|p| p.1).collect(),
        ltg3: pairs.iter().map(|p| 0.95 * p.0 + 0.05 * p.1).collect(),
    })
}

/// WRF PR92 maximum-updraft option. Threshold is explicit, in dBZ. Domain
/// partitioning changes this parameterization, so this routine always sees
/// the entire native horizontal domain. No cell-area multiplier is applied.
pub fn pr92(
    w: &[f64],
    refl: &[f64],
    nz: usize,
    cells: usize,
    threshold: f64,
) -> Result<Vec<f64>, String> {
    if cells == 0
        || nz == 0
        || nz.checked_mul(cells) != Some(w.len())
        || refl.len() != w.len()
        || !threshold.is_finite()
    {
        return Err("PR92 needs matching nonempty native volumes and finite threshold".into());
    }
    if w.par_iter()
        .chain(refl.par_iter())
        .any(|v| !v.is_finite() || v.abs() > 1e30)
    {
        return Err("PR92 domain contains missing updraft or reflectivity".into());
    }
    let wmax = w.par_iter().copied().reduce(|| 0.0, f64::max);
    let counts: Vec<usize> = refl
        .par_chunks(cells)
        .map(|row| row.iter().filter(|&&r| r > threshold).count())
        .collect();
    // Match WRF's strict-greater update: first level wins a tie.
    let mut level = 0;
    for k in 1..nz {
        if counts[k] > counts[level] {
            level = k;
        }
    }
    if counts[level] == 0 {
        return Ok(vec![0.0; cells]);
    }
    let rate = 5.7e-6 * wmax.powf(4.5) / 60.0 / (counts[level] as f64);
    Ok(refl[level * cells..(level + 1) * cells]
        .par_iter()
        .map(|&r| if r > threshold { rate } else { 0.0 })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mixed_phase_flux_uses_grams_and_suppresses_downdrafts() {
        let p = [90000., 50000.];
        let t = [268.15, 248.15];
        let q = [0., 0.];
        let z = [1000., 3000.];
        let zi = [0., 2000., 4000.];
        let c = ThermoColumns {
            cells: 1,
            nz: 2,
            pressure_pa: &p,
            temperature_k: &t,
            vapor_kgkg: &q,
            height_agl_m: &z,
            interface_agl_m: &zi,
        };
        let result = mc_caul(c, &[20., 20.], &[0.001, 0.001], &[0., 0.], &[0., 0.]).unwrap();
        assert!((result.ltg1[0] - 0.84).abs() < 1e-12);
        let expected_ice =
            0.20 * 2000. * 0.001 * (90000. / (287.04 * 268.15) + 50000. / (287.04 * 248.15));
        assert!((result.ltg2[0] - expected_ice).abs() < 1e-12);
        assert!((result.ltg3[0] - (0.95 * 0.84 + 0.05 * expected_ice)).abs() < 1e-12);
        let down = mc_caul(c, &[-20., -20.], &[0.001, 0.001], &[0., 0.], &[0., 0.]).unwrap();
        assert_eq!(down.ltg1[0], 0.);
        assert!(down.ltg2[0] > 0.);
    }
    #[test]
    fn pr92_conserves_domain_rate_and_does_not_allocate_outside_widest_level() {
        let w = [10., 20., 30., 40., 50., 60.];
        let r = [40., 40., 0., 40., 40., 40.];
        let rate = pr92(&w, &r, 2, 3, 20.).unwrap();
        assert!((rate.iter().sum::<f64>() - 5.7e-6 * 60_f64.powf(4.5) / 60.).abs() < 1e-10);
        assert!(rate.iter().all(|x| *x > 0.));
        assert_eq!(pr92(&w, &[0.; 6], 2, 3, 20.).unwrap(), vec![0.; 3]);
        let mut missing = w;
        missing[0] = f64::NAN;
        assert!(pr92(&missing, &r, 2, 3, 20.).is_err());
    }
}
