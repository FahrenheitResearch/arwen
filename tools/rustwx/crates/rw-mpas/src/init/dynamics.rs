//! Moisture, the base state, hydrostatic balance and the wind fields.
//!
//! Everything from the vertically interpolated `t`, `pressure` and `relhum`
//! to the model state the dycore starts from, in the order
//! `init_atm_case_gfs` does it — the order is load-bearing, because `relhum`
//! is used for `qv` and *then* rewritten with respect to ice, and because the
//! hydrostatic fixed point runs on a `pp` that the base-state step has
//! already coupled to the vertical metric.
//!
//! ## The inherited arithmetic quirk, carried deliberately, and what it is not
//! The virtual-temperature factor is `rvord - 1` (0.60836...) when `rho_zz` is
//! first formed, and the literal `1.61` in the level-1 seed, in the
//! hydrostatic loop, in `theta_m`, and in the recovery of `theta` from
//! `theta_m`.
//!
//! One correction to the design note, found by test rather than by reading:
//! **the `theta` round trip does close exactly.**  Step 4 builds `theta` as
//! `T*(p0/p)^(R/cp)` with no virtual factor at all — the `rvord - 1` there
//! sits in `rho_zz`'s denominator, not in `theta` — and steps 9 and 12
//! multiply and divide by the *same* `1.61` factor.  What the mix actually
//! moves is `rho_zz`: the value formed at step 4 with `rvord - 1` is
//! recomputed inside the hydrostatic loop with `1.61`, so the density (and
//! through it `pressure`, `rho_p`, `theta_m`, `ru` and `w`) carries the
//! inconsistency while `theta` does not.
//!
//! [`VirtualFactor`] carries both arms.  `ReproduceFortran` is the comparison
//! arm; `Consistent` uses `rvord - 1` in all five places.  Neither is silently
//! made the other.

#![allow(clippy::needless_range_loop)]

use crate::init::vinterp::{vertical_interp, Extrap};

/// MPAS's own constants, from `mpas_constants.F`.
pub mod constants {
    pub const GRAVITY: f32 = 9.80616;
    pub const RGAS: f32 = 287.0;
    pub const RV: f32 = 461.6;
    pub const CP: f32 = 7.0 * RGAS / 2.0;
    pub const CV: f32 = CP - RGAS;
    pub const RVORD: f32 = RV / RGAS;
    pub const P0: f32 = 1.0e5;
    /// The isothermal base state's temperature, a `parameter` inside
    /// `init_atm_case_gfs`.
    pub const T0B: f32 = 250.0;
    pub const SVP1: f32 = 0.6112;
    pub const SVP2: f32 = 17.67;
    pub const SVP3: f32 = 29.65;
    pub const SVPT0: f32 = 273.15;
}

use constants::*;

/// Which virtual-temperature factor the chain uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VirtualFactor {
    /// `rvord - 1` at first formation, `1.61` in the three later places.
    /// What the Fortran does; the comparison arm.
    ReproduceFortran,
    /// `rvord - 1` throughout, so the `theta` round trip closes.
    Consistent,
}

impl VirtualFactor {
    /// The factor the *later* three uses take.
    #[inline]
    fn late(self) -> f32 {
        match self {
            VirtualFactor::ReproduceFortran => 1.61,
            VirtualFactor::Consistent => RVORD - 1.0,
        }
    }
}

/// Liquid saturation mixing ratio, the Thompson polynomial MPAS carries.
///
/// The comment at the call site is load-bearing: ungrib's RH is always with
/// respect to liquid water (see `fix_gfs_rh` in `WPS/ungrib/src/rrpr.F`), so
/// this is the right saturation to divide by and an ice saturation is not.
#[allow(clippy::excessive_precision)]
pub fn rslf(p: f32, t: f32) -> f32 {
    // Transcribed at the Fortran's own written precision, not rounded to what
    // f32 can hold: the digits are the provenance.
    const C: [f32; 9] = [
        0.611583699e03,
        0.444606896e02,
        0.143177157e01,
        0.264224321e-1,
        0.299291081e-3,
        0.203154182e-5,
        0.702620698e-8,
        0.379534310e-11,
        -0.321582393e-13,
    ];
    let x = (t - 273.16).max(-80.0);
    let mut esl = C[8];
    for c in C.iter().take(8).rev() {
        esl = c + x * esl;
    }
    let esl = esl.min(p * 0.15);
    0.622 * esl / (p - esl)
}

/// Rewrite RH with respect to ice below freezing, after it has been used for
/// `qv`.  The ungrib reconstruction, including the linear blend GFS applies
/// between -20 C and 0 C.
pub fn convert_relhum_wrt_ice(t: &[f32], relhum: &mut [f32]) {
    for (i, &temp) in t.iter().enumerate() {
        if temp <= 273.15 {
            let eis = 0.01
                * (9.550426 - (5723.265 / temp) + (3.53068 * temp.ln()) - (0.00728332 * temp))
                    .exp();
            let mut ews = 6.112 * (17.67 * (temp - 273.15) / ((temp - 273.15) + 243.5)).exp();
            let r1 = if temp > 253.15 {
                let f = (273.15 - temp) / 20.0;
                (f * eis) + ((1.0 - f) * ews)
            } else {
                eis
            };
            let r1 = r1.max(1.0e-12);
            ews = ews.max(0.0);
            relhum[i] = (ews / r1 * relhum[i]).clamp(0.0, 100.0);
        }
    }
}

/// Two-metre specific humidity from the two-metre temperature, surface
/// pressure and two-metre relative humidity.
pub fn diagnose_q2(t2m: f32, psfc: f32, rh2: f32) -> f32 {
    let es = 6.112 * ((17.27 * (t2m - 273.16)) / (t2m - 35.86)).exp();
    let rs = 0.622 * es * 100.0 / (psfc - es * 100.0);
    0.01 * rs * rh2
}

/// The first-guess surface value of `q2` the case code computes while the
/// surface level is still in hand, with `svp1/svp2/svp3` rather than the
/// 6.112/17.27/35.86 triple used later.  Both are in the Fortran; both are
/// kept, because the later one overwrites the earlier one and a reader
/// comparing the two needs to see that.
pub fn diagnose_q2_surface_pass(t2m: f32, psfc: f32, rh2: f32) -> f32 {
    let mut es = SVP1 * 10.0 * (SVP2 * (t2m - SVPT0) / (t2m - SVP3)).exp();
    es = es.min(0.99 * 0.01 * psfc);
    let rs = 0.622 * es * 100.0 / (psfc - es * 100.0);
    0.01 * rs * rh2
}

/// One column's worth of the state the chain builds.
#[derive(Debug, Default, Clone)]
pub struct ColumnState {
    pub qv: Vec<f32>,
    pub theta: Vec<f32>,
    pub theta_m: Vec<f32>,
    pub pressure: Vec<f32>,
    pub pressure_base: Vec<f32>,
    pub pressure_p: Vec<f32>,
    pub exner: Vec<f32>,
    pub exner_base: Vec<f32>,
    pub rho_zz: Vec<f32>,
    pub rho: Vec<f32>,
    pub rho_base: Vec<f32>,
    pub rho_p: Vec<f32>,
    pub theta_base: Vec<f32>,
    pub rtheta_base: Vec<f32>,
    pub precipw: f32,
    pub surface_pressure: f32,
    /// Hydrostatic passes actually used, per level.  Reported as a
    /// distribution so a delta traceable to the fixed point can be attributed
    /// to it rather than absorbed into a tolerance.
    pub hydrostatic_iterations: Vec<u32>,
}

/// Build one cell's column, from the vertically interpolated fields up.
///
/// `t` enters as temperature and leaves the function having been turned into
/// potential temperature and then `theta_m`, exactly as the Fortran reuses its
/// own array.
#[allow(clippy::too_many_arguments)]
pub fn build_column(
    t_in: &[f32],
    pressure_in: &[f32],
    relhum: &[f32],
    spechum: &[f32],
    use_spechumd: bool,
    zgrid: &[f32],
    zz: &[f32],
    fzm: &[f32],
    fzp: &[f32],
    dzu: &[f32],
    rdzw0: f32,
    factor: VirtualFactor,
) -> ColumnState {
    let nz = t_in.len();
    let mut s = ColumnState {
        qv: vec![0.0; nz],
        theta: vec![0.0; nz],
        theta_m: vec![0.0; nz],
        pressure: pressure_in.to_vec(),
        pressure_base: vec![0.0; nz],
        pressure_p: vec![0.0; nz],
        exner: vec![0.0; nz],
        exner_base: vec![0.0; nz],
        rho_zz: vec![0.0; nz],
        rho: vec![0.0; nz],
        rho_base: vec![0.0; nz],
        rho_p: vec![0.0; nz],
        theta_base: vec![0.0; nz],
        rtheta_base: vec![0.0; nz],
        precipw: 0.0,
        surface_pressure: 0.0,
        hydrostatic_iterations: vec![0; nz],
    };

    // 1. Water vapour.
    for k in 0..nz {
        s.qv[k] = if use_spechumd {
            spechum[k] / (1.0 - spechum[k])
        } else {
            let rs = rslf(s.pressure[k], t_in[k]);
            0.01 * rs * relhum[k]
        };
    }

    // 4. Exner, potential temperature, dry density.
    let mut theta = vec![0.0f32; nz];
    for k in 0..nz {
        s.exner[k] = (s.pressure[k] / P0).powf(RGAS / CP);
        theta[k] = t_in[k] * (P0 / s.pressure[k]).powf(RGAS / CP);
        let early = match factor {
            VirtualFactor::ReproduceFortran => RVORD - 1.0,
            VirtualFactor::Consistent => RVORD - 1.0,
        };
        s.rho_zz[k] = s.pressure[k] / RGAS / (s.exner[k] * theta[k] * (1.0 + early * s.qv[k]));
        s.rho_zz[k] /= 1.0 + s.qv[k];
    }

    // 5. Precipitable water, before the metric coupling.
    for k in 0..nz {
        s.precipw += s.rho_zz[k] * s.qv[k] * (zgrid[k + 1] - zgrid[k]);
    }

    // 6. Base state: a dry isothermal atmosphere at t0b.
    for k in 0..nz {
        let ztemp = 0.5 * (zgrid[k + 1] + zgrid[k]);
        s.pressure_base[k] = P0 * (-GRAVITY * ztemp / (RGAS * T0B)).exp();
        s.exner_base[k] = (s.pressure_base[k] / P0).powf(RGAS / CP);
        s.rho_base[k] = s.pressure_base[k] / (RGAS * T0B);
        s.theta_base[k] = T0B / s.exner_base[k];
        s.rtheta_base[k] = s.rho_base[k] * s.theta_base[k];
        s.exner[k] = s.exner_base[k];
        s.pressure_p[k] = 0.0;
        s.rho_p[k] = 0.0;
    }

    // 7. Couple to the vertical metric.
    for k in 0..nz {
        s.rho_base[k] /= zz[k];
        s.rho_zz[k] /= zz[k];
        s.pressure_p[k] = s.pressure[k] - s.pressure_base[k];
        s.rho_p[k] = s.rho_zz[k] - s.rho_base[k];
    }

    // 8. Hydrostatic rebalance.  Level 1 is seeded, then each level above runs
    //    a fixed point of at most thirty passes, converged when |dpp| <= 1e-4 Pa.
    let late = factor.late();
    s.rho_zz[0] = ((s.pressure[0] / P0).powf(CV / CP)) * (P0 / RGAS)
        / (theta[0] * (1.0 + late * s.qv[0]))
        / zz[0];
    s.rho_p[0] = s.rho_zz[0] - s.rho_base[0];
    for k in 1..nz {
        let mut it = 0u32;
        let mut p_check = 2.0 * 0.0001f32;
        while it < 30 && p_check > 0.0001 {
            p_check = s.pressure_p[k];
            s.pressure_p[k] = s.pressure_p[k - 1]
                - (fzm[k] * s.rho_p[k] + fzp[k] * s.rho_p[k - 1]) * GRAVITY * dzu[k]
                - (fzm[k] * s.rho_zz[k] * s.qv[k] + fzp[k] * s.rho_zz[k - 1] * s.qv[k - 1])
                    * GRAVITY
                    * dzu[k];
            s.pressure[k] = s.pressure_p[k] + s.pressure_base[k];
            s.exner[k] = (s.pressure[k] / P0).powf(RGAS / CP);
            s.rho_zz[k] = s.pressure[k]
                / RGAS
                / (s.exner[k] * theta[k] * (1.0 + late * s.qv[k]))
                / zz[k];
            s.rho_p[k] = s.rho_zz[k] - s.rho_base[k];
            p_check = (p_check - s.pressure_p[k]).abs();
            it += 1;
        }
        s.hydrostatic_iterations[k] = it;
    }

    // 9. theta_m, and decouple rho_p from the metric.
    for k in 0..nz {
        s.theta_m[k] = theta[k] * (1.0 + late * s.qv[k]);
        s.rho_p[k] *= zz[k];
    }

    // 11. The explicitly ad-hoc diagnostic surface pressure.
    s.surface_pressure = 0.5 * GRAVITY / rdzw0
        * (1.25 * s.rho_zz[0] * (1.0 + s.qv[0]) - 0.25 * s.rho_zz[1] * (1.0 + s.qv[1]))
        + s.pressure_p[0]
        + s.pressure_base[0];

    // 12. rho and the recovered theta.
    for k in 0..nz {
        s.rho[k] = s.rho_zz[k] * zz[k];
        s.theta[k] = s.theta_m[k] / (1.0 + late * s.qv[k]);
    }

    s
}

/// Rotate a first-guess wind pair to the edge normal.
#[inline]
pub fn edge_normal_wind(angle_edge: f32, u: f32, v: f32) -> f32 {
    angle_edge.cos() * u + angle_edge.sin() * v
}

/// The two-cell mean coordinate an edge column is interpolated on.
#[inline]
pub fn edge_column_coordinate(z1: f32, z2: f32) -> f32 {
    0.5 * (z1 + z2)
}

/// The four-point mean an edge's target height is.
#[inline]
pub fn edge_target_height(zk1: f32, zkp1: f32, zk2: f32, zkp2: f32) -> f32 {
    0.25 * (zk1 + zkp1 + zk2 + zkp2)
}

/// Interpolate one edge column with `extrap = 0`.
///
/// Case 7 uses constant extrapolation at edges; `init_atm_case_lbc` uses
/// linear for the same field.  That difference is real and the two are not
/// unified here.
pub fn interp_edge_column(target: f32, coords: &[f32], values: &[f32]) -> f32 {
    vertical_interp(target, coords, values, Extrap::Constant).unwrap_or(values[values.len() - 1])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rslf_is_near_the_textbook_saturation_at_twenty_celsius() {
        // 1000 hPa, 293.15 K: about 14.9 g/kg.
        let r = rslf(100_000.0, 293.15);
        assert!((r - 0.0149).abs() < 5e-4, "{r}");
    }

    #[test]
    fn relhum_over_ice_is_left_alone_above_freezing() {
        let t = [280.0f32];
        let mut rh = [50.0f32];
        convert_relhum_wrt_ice(&t, &mut rh);
        assert_eq!(rh[0], 50.0);
    }

    #[test]
    fn relhum_over_ice_is_raised_below_freezing_and_capped_at_a_hundred() {
        let t = [250.0f32];
        let mut rh = [95.0f32];
        convert_relhum_wrt_ice(&t, &mut rh);
        assert!(rh[0] > 95.0, "{}", rh[0]);
        assert!(rh[0] <= 100.0, "{}", rh[0]);
    }

    #[test]
    fn the_virtual_factor_mix_moves_density_and_leaves_theta_alone() {
        let nz = 4usize;
        let t = vec![288.0f32, 285.0, 280.0, 270.0];
        let p = vec![100_000.0f32, 95_000.0, 90_000.0, 85_000.0];
        let rh = vec![90.0f32; nz];
        let sh = vec![0.0f32; nz];
        let zgrid: Vec<f32> = (0..=nz).map(|k| k as f32 * 200.0).collect();
        let zz = vec![1.0f32; nz];
        let fzm = vec![0.5f32; nz];
        let fzp = vec![0.5f32; nz];
        let dzu = vec![200.0f32; nz];

        let a = build_column(
            &t,
            &p,
            &rh,
            &sh,
            false,
            &zgrid,
            &zz,
            &fzm,
            &fzp,
            &dzu,
            1.0 / 200.0,
            VirtualFactor::ReproduceFortran,
        );
        let b = build_column(
            &t,
            &p,
            &rh,
            &sh,
            false,
            &zgrid,
            &zz,
            &fzm,
            &fzp,
            &dzu,
            1.0 / 200.0,
            VirtualFactor::Consistent,
        );
        // theta is untouched by the factor: steps 9 and 12 multiply and
        // divide by the same number, whichever number that is.
        for k in 0..nz {
            assert_eq!(
                a.theta[k].to_bits(),
                b.theta[k].to_bits(),
                "theta must round-trip exactly at level {k}"
            );
        }
        // Density is where the mix actually lands.
        let rel = ((a.rho_zz[0] - b.rho_zz[0]) / b.rho_zz[0]).abs();
        assert!(rel > 1e-6, "the two arms must differ in rho_zz: {rel:e}");
        assert!(rel < 1e-2, "and not by this much: {rel:e}");
        let rel_m = ((a.theta_m[0] - b.theta_m[0]) / b.theta_m[0]).abs();
        assert!(rel_m > 1e-6, "and in theta_m: {rel_m:e}");
    }

    #[test]
    fn the_hydrostatic_fixed_point_reports_its_pass_count() {
        let nz = 3usize;
        let t = vec![288.0f32, 285.0, 280.0];
        let p = vec![100_000.0f32, 97_000.0, 94_000.0];
        let rh = vec![50.0f32; nz];
        let sh = vec![0.0f32; nz];
        let zgrid: Vec<f32> = (0..=nz).map(|k| k as f32 * 250.0).collect();
        let s = build_column(
            &t,
            &p,
            &rh,
            &sh,
            false,
            &zgrid,
            &vec![1.0; nz],
            &vec![0.5; nz],
            &vec![0.5; nz],
            &vec![250.0; nz],
            1.0 / 250.0,
            VirtualFactor::ReproduceFortran,
        );
        assert_eq!(s.hydrostatic_iterations[0], 0, "level 1 is seeded, not iterated");
        assert!(s.hydrostatic_iterations[1] >= 1);
        assert!(s.hydrostatic_iterations.iter().all(|&i| i <= 30));
    }

    #[test]
    fn specific_humidity_becomes_a_mixing_ratio_when_that_path_is_chosen() {
        let nz = 2usize;
        let sh = vec![0.01f32, 0.005];
        let s = build_column(
            &[288.0, 285.0],
            &[100_000.0, 97_000.0],
            &vec![0.0; nz],
            &sh,
            true,
            &[0.0, 250.0, 500.0],
            &vec![1.0; nz],
            &vec![0.5; nz],
            &vec![0.5; nz],
            &vec![250.0; nz],
            1.0 / 250.0,
            VirtualFactor::ReproduceFortran,
        );
        assert!((s.qv[0] - 0.01 / 0.99).abs() < 1e-7, "{}", s.qv[0]);
    }
}
