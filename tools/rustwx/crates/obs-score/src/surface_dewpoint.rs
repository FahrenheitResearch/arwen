//! Pinned liquid-water dewpoint to Q2 mixing ratio, in SI units.
//!
//! Inverse of the public Q2/PSFC rendered dewpoint contract:
//! e_hPa = 6.112 exp(17.67 Td_C / (Td_C + 243.5));
//! Q2 = 0.622 e / (p-e), kg water per kg dry air.
//! Observation pressure is a stated ensemble-mean forecast PSFC proxy.
//! Sigma propagation holds this pressure fixed. No pressure-error term,
//! saturation clipping, humidity floor or temperature-unit conversion occurs.

use crate::{call, input, output};

const E0_PA: f64 = 611.2;
const A: f64 = 17.67;
const B_C: f64 = 243.5;
const KELVIN_ZERO: f64 = 273.15;
const EPSILON: f64 = 0.622;

fn pressure_pa(value: f64) -> Result<(), String> {
    if !value.is_finite() || !(20_000.0..=120_000.0).contains(&value) {
        return Err(format!(
            "surface dewpoint requires PSFC in Pa within 20000..120000, got {value}; hPa inputs are not converted"
        ));
    }
    Ok(())
}

fn dewpoint_to_q2(td_k: f64, p_pa: f64, sigma_k: f64) -> Result<(f64, f64), String> {
    pressure_pa(p_pa)?;
    if !td_k.is_finite() || !(150.0..=350.0).contains(&td_k) {
        return Err(format!(
            "surface dewpoint requires Td in K within 150..350, got {td_k}; Celsius inputs are not converted"
        ));
    }
    if !sigma_k.is_finite() || sigma_k <= 0.0 {
        return Err("surface dewpoint sigma must be finite and positive in K".into());
    }
    let td_c = td_k - KELVIN_ZERO;
    let e = E0_PA * (A * td_c / (td_c + B_C)).exp();
    if !e.is_finite() || e >= p_pa {
        return Err("surface dewpoint vapor pressure must be finite and below PSFC".into());
    }
    let denominator = p_pa - e;
    let q = EPSILON * e / denominator;
    let dln_e_dtd = A * B_C / (td_c + B_C).powi(2);
    let sigma_q = EPSILON * p_pa * e * dln_e_dtd / denominator.powi(2) * sigma_k;
    if !q.is_finite()
        || !(0.0..=0.2).contains(&q)
        || q == 0.0
        || !sigma_q.is_finite()
        || sigma_q <= 0.0
    {
        return Err(
            "surface dewpoint gives Q2 outside 0..0.2 kg/kg or invalid propagated sigma".into(),
        );
    }
    Ok((q, sigma_q))
}

/// Points are the already age/elevation/dedup accepted station columns.
/// PSFC is member-major [members, points], all in Pa. Mean pressure is
/// reduced in fixed member order, independent of worker completion order.
/// Outputs are written only after every input has validated.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_obsscore_surface_dewpoint_q2(
    td_k: *const f64,
    member_psfc_pa: *const f64,
    points: usize,
    members: usize,
    sigma_td_k: f64,
    error_inflation: f64,
    q2: *mut f64,
    sigma_q2: *mut f64,
    mean_psfc_pa: *mut f64,
) -> i32 {
    call(|| {
        if members == 0 {
            return Err("surface dewpoint needs at least one PSFC member".into());
        }
        if !sigma_td_k.is_finite() || sigma_td_k <= 0.0 {
            return Err("surface dewpoint sigma must be finite and positive in K".into());
        }
        if !error_inflation.is_finite() || error_inflation < 1.0 {
            return Err("surface dewpoint error inflation must be finite and >= 1".into());
        }
        let count = points
            .checked_mul(members)
            .ok_or("surface dewpoint input size overflow")?;
        let td = unsafe { input(td_k, points)? };
        let pressure = unsafe { input(member_psfc_pa, count)? };
        let mut computed = Vec::with_capacity(points);
        for point in 0..points {
            let mut total = 0.0;
            for member in 0..members {
                let p = pressure[member * points + point];
                pressure_pa(p).map_err(|e| format!("point {point}, member {member}: {e}"))?;
                total += p;
            }
            let mean_p = total / members as f64;
            let (q, sigma) = dewpoint_to_q2(td[point], mean_p, sigma_td_k * error_inflation)
                .map_err(|e| format!("point {point}: {e}"))?;
            computed.push((q, sigma, mean_p));
        }
        let qs = unsafe { output(q2, points)? };
        let errors = unsafe { output(sigma_q2, points)? };
        let means = unsafe { output(mean_psfc_pa, points)? };
        for (index, (q, sigma, p)) in computed.into_iter().enumerate() {
            qs[index] = q;
            errors[index] = sigma;
            means[index] = p;
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn freezing_dewpoint_matches_pinned_mixing_ratio() {
        let (q, _) = dewpoint_to_q2(273.15, 100_000.0, 2.0).unwrap();
        assert_eq!(q, 0.622 * 611.2 / (100_000.0 - 611.2));
    }

    #[test]
    fn analytic_sigma_matches_independent_centered_derivative() {
        for td in [233.15, 273.15, 293.15, 303.15] {
            let (_, sigma) = dewpoint_to_q2(td, 93_000.0, 1.7).unwrap();
            let step = 1e-3;
            let plus = dewpoint_to_q2(td + step, 93_000.0, 1.0).unwrap().0;
            let minus = dewpoint_to_q2(td - step, 93_000.0, 1.0).unwrap().0;
            let expected = (plus - minus) / (2.0 * step) * 1.7;
            assert!((sigma / expected - 1.0).abs() < 1e-8);
        }
    }

    #[test]
    fn units_and_physical_domain_are_refused() {
        assert!(
            dewpoint_to_q2(20.0, 100_000.0, 2.0)
                .unwrap_err()
                .contains("in K")
        );
        assert!(
            dewpoint_to_q2(293.15, 1000.0, 2.0)
                .unwrap_err()
                .contains("in Pa")
        );
        assert!(dewpoint_to_q2(350.0, 20_000.0, 2.0).is_err());
        assert!(dewpoint_to_q2(f64::NAN, 100_000.0, 2.0).is_err());
        assert!(dewpoint_to_q2(293.15, 100_000.0, 0.0).is_err());
    }

    #[test]
    fn batch_mean_is_fixed_order_and_bad_input_cannot_write_partial_results() {
        let td = [273.15, 293.15];
        let mut pressures = [90_000.0, 92_000.0, 100_000.0, 98_000.0];
        let mut q = [-1.0; 2];
        let mut sigma = [-1.0; 2];
        let mut mean = [-1.0; 2];
        let status = unsafe {
            gpuwm_obsscore_surface_dewpoint_q2(
                td.as_ptr(),
                pressures.as_ptr(),
                2,
                2,
                2.0,
                1.0,
                q.as_mut_ptr(),
                sigma.as_mut_ptr(),
                mean.as_mut_ptr(),
            )
        };
        assert_eq!(status, 0);
        assert_eq!(mean, [95_000.0, 95_000.0]);
        assert_eq!(q[0], 0.622 * 611.2 / (95_000.0 - 611.2));
        pressures[3] = 980.0;
        q = [-2.0; 2];
        sigma = [-2.0; 2];
        mean = [-2.0; 2];
        let status = unsafe {
            gpuwm_obsscore_surface_dewpoint_q2(
                td.as_ptr(),
                pressures.as_ptr(),
                2,
                2,
                2.0,
                1.0,
                q.as_mut_ptr(),
                sigma.as_mut_ptr(),
                mean.as_mut_ptr(),
            )
        };
        assert_eq!(status, -1);
        assert_eq!(q, [-2.0; 2]);
        assert_eq!(sigma, [-2.0; 2]);
        assert_eq!(mean, [-2.0; 2]);
    }
}
