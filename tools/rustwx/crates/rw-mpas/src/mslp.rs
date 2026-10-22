//! Sea-level pressure for MPAS-mesh output, reduced the way the GFS analysis
//! reduces it (NCEP's Shuell method), so a hex map and the analysis it is
//! compared with mean the same thing by "sea-level pressure".
//!
//! THE DEFECT THIS FIXES (hex global plots, 2026-10-10; measured again
//! 2026-10-14).  Hex products diagnosed MSLP through wrf-core's `slp`, the
//! RIP/wrf-python DCOMPUTESEAPRS reduction.  That reduction reads its
//! temperature 100 hPa above the ground and then applies the "ridiculous MM5
//! test" to every column that is not hot:
//! `T_sl = 290.66 - 0.005 (T_s - 290.66)^2`.  The test was written to cool
//! columns whose sea-level temperature comes out above 290.66 K; as coded it
//! also WARMS every cold column toward 290.66 K -- a 230 K Antarctic plateau
//! column is reduced through a 272 K sea-level temperature instead of 250 K --
//! and a warmer fictitious column gives a lower sea-level pressure.  Against
//! the GFS 0.25 degree PRMSL on the x1.40962 init of 2026-10-11 12Z
//! (40,962 cells): mean -0.63 hPa, RMS 4.15, max 53.8; land above 3 km mean
//! -14.6, Antarctica -8.2, Greenland -14.6 (on the 15 km mesh the same
//! reduction read mean -3.18, RMS 10.85, max 79 hPa on a lat-lon window that
//! over-weights the polar caps).  The state was right: the init's surface
//! pressure matched GFS's surface pressure moved to the model terrain to
//! 0.05 hPa mean, 0.16 RMS.
//!
//! THE REDUCTION HERE (NCEP Shuell, as the GFS post computes PRMSL):
//!
//! * `T_g`, the ground temperature, is the lowest model level's virtual
//!   temperature carried down to the terrain at 6.5 K/km;
//! * `T_0 = T_g + 0.0065 z_s`, the sea-level temperature of a fictitious
//!   column at the standard lapse rate;
//! * ONLY when `T_0 > 290.66 K` is it adjusted: to 290.66 K when
//!   `T_g <= 290.66 K`, else to `290.66 - 0.005 (T_g - 290.66)^2`;
//! * `p_msl = p_s exp(2 g z_s / (R_d (T_0 + T_g)))`.
//!
//! On the same init this reads mean -0.02 hPa, RMS 0.64, max 15.7 against
//! PRMSL; land above 3 km -1.5, Antarctica -0.8, Greenland -2.0, Tibet -0.03.
//! Over the ocean (terrain under 5 m) both reductions agree with PRMSL to
//! 0.14 hPa RMS, because there is no fictitious column to argue about.

/// Dry-air gas constant, J/(kg K), as the NCEP post uses it.
pub const RD: f64 = 287.04;
/// Gravity, m/s^2.
pub const G: f64 = 9.80665;
/// The standard-atmosphere lapse rate, K/m.
pub const LAPSE: f64 = 0.0065;
/// The Shuell warm-column threshold, K.
pub const SHUELL_TC: f64 = 290.66;

/// The NCEP Shuell sea-level temperature for a ground temperature `t_ground`
/// (K) under terrain `terrain_m` (m).
pub fn shuell_sea_level_temperature(t_ground: f64, terrain_m: f64) -> f64 {
    let t0 = t_ground + LAPSE * terrain_m;
    if t0 > SHUELL_TC {
        if t_ground <= SHUELL_TC {
            SHUELL_TC
        } else {
            SHUELL_TC - 0.005 * (t_ground - SHUELL_TC) * (t_ground - SHUELL_TC)
        }
    } else {
        t0
    }
}

/// Sea-level pressure, Pa, by the NCEP Shuell reduction.
///
/// * `surface_pressure_pa`, `terrain_m`: the column's surface;
/// * `t_lowest_k`, `qv_lowest`, `z_lowest_m`: the lowest model level's
///   temperature (K), water-vapour mixing ratio (kg/kg) and height above sea
///   level (m).
pub fn shuell_mslp_pa(
    surface_pressure_pa: f64,
    terrain_m: f64,
    t_lowest_k: f64,
    qv_lowest: f64,
    z_lowest_m: f64,
) -> f64 {
    let tv = t_lowest_k * (1.0 + 0.608 * qv_lowest.max(0.0));
    let t_ground = tv + LAPSE * (z_lowest_m - terrain_m).max(0.0);
    let t_sea = shuell_sea_level_temperature(t_ground, terrain_m);
    surface_pressure_pa * (2.0 * G * terrain_m / (RD * (t_sea + t_ground))).exp()
}

/// The reduction over planes: every argument holds one value per column.
pub fn shuell_mslp_plane(
    surface_pressure_pa: &[f64],
    terrain_m: &[f64],
    t_lowest_k: &[f64],
    qv_lowest: &[f64],
    z_lowest_m: &[f64],
) -> Vec<f64> {
    surface_pressure_pa
        .iter()
        .zip(terrain_m)
        .zip(t_lowest_k)
        .zip(qv_lowest)
        .zip(z_lowest_m)
        .map(|((((&ps, &zs), &t), &q), &z)| shuell_mslp_pa(ps, zs, t, q, z))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn at_sea_level_the_reduction_is_the_surface_pressure() {
        let p = shuell_mslp_pa(101_000.0, 0.0, 288.0, 0.01, 30.0);
        assert_eq!(p, 101_000.0);
    }

    #[test]
    fn a_cold_plateau_column_is_not_warmed() {
        // 230 K over 3,000 m: the fictitious column runs 230 -> 249.5 K.  The
        // wrf-python test would have set the sea-level temperature to
        // 290.66 - 0.005 (230 - 290.66)^2 = 272.26 K and reduced through it.
        let t_ground = 230.0;
        let t0 = shuell_sea_level_temperature(t_ground, 3_000.0);
        assert!((t0 - 249.5).abs() < 1e-9, "{t0}");
        let shuell = shuell_mslp_pa(70_000.0, 3_000.0, t_ground, 0.0, 3_000.0);
        let mean_t = 0.5 * (272.2577 + t_ground);
        let wrf_python_like = 70_000.0 * (G * 3_000.0 / (RD * mean_t)).exp();
        // the warmed column reads lower; the size is the defect (about 13 hPa here)
        assert!(shuell - wrf_python_like > 1_000.0, "{shuell} {wrf_python_like}");
    }

    #[test]
    fn a_hot_column_is_capped_the_two_ways_shuell_states() {
        // ground at 285 K under 2 km: T0 = 298 K > 290.66 and T_g <= 290.66 -> 290.66
        assert_eq!(shuell_sea_level_temperature(285.0, 2_000.0), SHUELL_TC);
        // ground at 300 K under 500 m: both hot -> 290.66 - 0.005 (300 - 290.66)^2
        let t0 = shuell_sea_level_temperature(300.0, 500.0);
        assert!((t0 - (SHUELL_TC - 0.005 * 9.34 * 9.34)).abs() < 1e-9);
        // a temperate column is untouched
        assert!((shuell_sea_level_temperature(280.0, 1_000.0) - 286.5).abs() < 1e-9);
    }

    #[test]
    fn the_lowest_level_is_carried_to_the_ground_and_never_up() {
        // lowest level 40 m above a 1,000 m surface at 280 K: ground 280.26 K
        let p = shuell_mslp_pa(90_000.0, 1_000.0, 280.0, 0.0, 1_040.0);
        let tg = 280.0 + 0.26;
        let expected = 90_000.0 * (2.0 * G * 1_000.0 / (RD * (tg + LAPSE * 1_000.0 + tg))).exp();
        assert!((p - expected).abs() < 1e-6);
        // a level reported below the terrain is not extrapolated upward
        let q = shuell_mslp_pa(90_000.0, 1_000.0, 280.0, 0.0, 990.0);
        let expected = 90_000.0 * (2.0 * G * 1_000.0 / (RD * (280.0 + LAPSE * 1_000.0 + 280.0))).exp();
        assert!((q - expected).abs() < 1e-6);
    }

    #[test]
    fn the_plane_form_is_the_point_form() {
        let ps = [101_000.0, 70_000.0];
        let zs = [0.0, 3_000.0];
        let t = [290.0, 230.0];
        let q = [0.01, 0.0];
        let z = [20.0, 3_030.0];
        let plane = shuell_mslp_plane(&ps, &zs, &t, &q, &z);
        for i in 0..2 {
            assert_eq!(plane[i], shuell_mslp_pa(ps[i], zs[i], t[i], q[i], z[i]));
        }
    }
}
