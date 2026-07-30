//! Severe weather composite parameters.
//!
//! Implements the significant tornado parameter (STP), supercell composite
//! parameter (SCP), and critical angle — standard indices used in operational
//! severe convective weather forecasting.
//!
//! All inputs use SI units: CAPE in J/kg, heights in meters, helicity in
//! m^2/s^2, and bulk shear in m/s.
//!
//! ## Grid-based composites
//!
//! The [`grid`] submodule re-exports grid-oriented functions from
//! `wx_math::composite` that operate on flattened 2-D/3-D arrays and use
//! `rayon` for parallelism. These have different signatures from the
//! point-based functions at the top level of this module.

use std::f64::consts::PI;

// ─────────────────────────────────────────────
// Re-exports from wx_math::composite (point-based helpers)
// ─────────────────────────────────────────────

pub use wx_math::composite::boyden_index;
pub use wx_math::composite::bulk_richardson_number;
pub use wx_math::composite::convective_inhibition_depth;
pub use wx_math::composite::dendritic_growth_zone;
pub use wx_math::composite::fosberg_fire_weather_index;
pub use wx_math::composite::freezing_rain_composite;
pub use wx_math::composite::haines_index;
pub use wx_math::composite::hot_dry_windy;
pub use wx_math::composite::warm_nose_check;

// Re-export galvez_davison_index from wx_math::thermo (it lives there, not composite)
pub use wx_math::thermo::galvez_davison_index;

/// Grid-oriented composite parameters from `wx_math::composite`.
///
/// These functions accept flattened 2-D or 3-D arrays (`&[f64]`) and grid
/// dimensions (`nx`, `ny`, `nz`) and return `Vec<f64>` result grids. They
/// use `rayon` internally for parallel computation.
///
/// Functions whose names overlap with point-based versions in the parent
/// module (e.g. `supercell_composite_parameter`, `critical_angle`) have
/// different signatures here — they take grid slices rather than scalar
/// values.
pub mod grid {
    use ecape_rs::{
        calc_ecape_ncape, calc_ecape_parcel, CapeType as EcapeCapeType, ParcelOptions,
        StormMotionType as EcapeStormMotionType,
    };
    use rayon::prelude::*;

    // ── Stability indices (profile-based, different arg order from thermo) ──
    pub use wx_math::composite::cross_totals;
    pub use wx_math::composite::k_index;
    pub use wx_math::composite::lifted_index;
    pub use wx_math::composite::showalter_index;
    pub use wx_math::composite::sweat_index;
    pub use wx_math::composite::total_totals;
    pub use wx_math::composite::vertical_totals;

    // ── 3-D grid compute functions ──
    pub use wx_math::composite::compute_cape_cin;
    pub use wx_math::composite::compute_lapse_rate;
    pub use wx_math::composite::compute_pw;
    pub use wx_math::composite::compute_shear;
    pub use wx_math::composite::{compute_srh, compute_srh_hemispheric};

    #[derive(Debug, Clone, PartialEq)]
    pub struct WindDiagnosticsBundle {
        pub srh_01km_m2s2: Vec<f64>,
        pub srh_03km_m2s2: Vec<f64>,
        pub shear_06km_ms: Vec<f64>,
    }

    #[derive(Debug, Clone, Copy, Default)]
    struct EcapeSummary {
        ecape: f64,
        ncape: f64,
        cape: f64,
        cin: f64,
        lfc: f64,
        el: f64,
    }

    #[derive(Debug, Clone, Copy, Default)]
    struct EcapeColumnResult {
        summary: EcapeSummary,
        failed: bool,
    }

    #[derive(Debug, Clone, PartialEq)]
    pub struct EcapeGridFields {
        pub ecape: Vec<f64>,
        pub ncape: Vec<f64>,
        pub cape: Vec<f64>,
        pub cin: Vec<f64>,
        pub lfc: Vec<f64>,
        pub el: Vec<f64>,
    }

    #[derive(Debug, Clone, PartialEq)]
    pub struct EcapeGridFieldsWithFailureMask {
        pub fields: EcapeGridFields,
        pub failure_mask: Vec<u8>,
    }

    impl EcapeGridFieldsWithFailureMask {
        pub fn failure_count(&self) -> usize {
            self.failure_mask.iter().filter(|&&flag| flag != 0).count()
        }
    }

    #[derive(Debug, Clone, PartialEq)]
    pub struct EcapeGridTripletWithFailureMask {
        pub sb: EcapeGridFieldsWithFailureMask,
        pub ml: EcapeGridFieldsWithFailureMask,
        pub mu: EcapeGridFieldsWithFailureMask,
    }

    #[derive(Debug, Clone, PartialEq)]
    pub struct EcapeGridTriplet {
        pub sb: EcapeGridFields,
        pub ml: EcapeGridFields,
        pub mu: EcapeGridFields,
    }

    #[derive(Debug, Clone, Copy, Default)]
    struct EcapeTripletColumnResult {
        sb: EcapeColumnResult,
        ml: EcapeColumnResult,
        mu: EcapeColumnResult,
    }

    fn resolve_ecape_cape_type(parcel_type: &str) -> Result<EcapeCapeType, String> {
        let cape_type = EcapeCapeType::parse_normalized(parcel_type).map_err(|err| {
            format!(
                "unsupported ECAPE parcel_type '{}'; expected 'surface', 'sb', 'mixed_layer', 'ml', 'most_unstable', or 'mu'",
                err.value()
            )
        })?;
        match cape_type {
            EcapeCapeType::UserDefined => Err(
                "ECAPE grid calculations do not support user_defined custom parcel thermodynamics"
                    .to_string(),
            ),
            _ => Ok(cape_type),
        }
    }

    fn resolve_ecape_storm_motion_type(
        storm_motion_type: &str,
    ) -> Result<EcapeStormMotionType, String> {
        EcapeStormMotionType::parse_normalized(storm_motion_type).map_err(|err| {
            format!(
                "unsupported ECAPE storm_motion_type '{}'; expected 'right_moving', 'bunkers_rm', 'left_moving', 'bunkers_lm', or 'mean_wind'",
                err.value()
            )
        })
    }

    fn dewpoint_k_from_q(q_kgkg: f64, p_pa: f64, temp_k: f64) -> f64 {
        let q = q_kgkg.max(1.0e-10);
        let p_hpa = p_pa / 100.0;
        let e = (q * p_hpa / (0.622 + q)).max(1.0e-10);
        let ln_e = (e / 6.112).ln();
        let td_c = (243.5 * ln_e) / (17.67 - ln_e);
        (td_c + 273.15).min(temp_k)
    }

    fn interp_at_height(target_h: f64, heights: &[f64], values: &[f64]) -> f64 {
        if heights.is_empty() {
            return f64::NAN;
        }
        if target_h <= heights[0] {
            return values[0];
        }
        if target_h >= heights[heights.len() - 1] {
            return values[values.len() - 1];
        }
        for k in 0..heights.len() - 1 {
            if heights[k] <= target_h && heights[k + 1] >= target_h {
                let frac = (target_h - heights[k]) / (heights[k + 1] - heights[k]);
                return values[k] + frac * (values[k + 1] - values[k]);
            }
        }
        values[values.len() - 1]
    }

    fn extract_column(
        data: &[f64],
        nz: usize,
        ny: usize,
        nx: usize,
        j: usize,
        i: usize,
    ) -> Vec<f64> {
        let mut col = Vec::with_capacity(nz);
        for k in 0..nz {
            col.push(data[k * ny * nx + j * nx + i]);
        }
        col
    }

    fn ensure_surface_up_profile(
        h_col: Vec<f64>,
        u_col: Vec<f64>,
        v_col: Vec<f64>,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        if h_col.len() > 1 && h_col[0] > h_col[h_col.len() - 1] {
            let mut h = h_col;
            let mut u = u_col;
            let mut v = v_col;
            h.reverse();
            u.reverse();
            v.reverse();
            (h, u, v)
        } else {
            (h_col, u_col, v_col)
        }
    }

    fn compute_srh_column(heights: &[f64], u_prof: &[f64], v_prof: &[f64], top_m: f64) -> f64 {
        let nz = heights.len();
        if nz < 2 {
            return 0.0;
        }

        let mean_depth = 6000.0;
        let mut sum_u = 0.0;
        let mut sum_v = 0.0;
        let mut sum_dz = 0.0;

        for k in 0..nz - 1 {
            if heights[k] >= mean_depth {
                break;
            }
            let h_bot = heights[k];
            let h_top = heights[k + 1].min(mean_depth);
            let dz = h_top - h_bot;
            if dz <= 0.0 {
                continue;
            }
            let u_mid = 0.5 * (u_prof[k] + u_prof[k + 1]);
            let v_mid = 0.5 * (v_prof[k] + v_prof[k + 1]);
            sum_u += u_mid * dz;
            sum_v += v_mid * dz;
            sum_dz += dz;
        }

        if sum_dz <= 0.0 {
            return 0.0;
        }

        let mean_u = sum_u / sum_dz;
        let mean_v = sum_v / sum_dz;

        let u_sfc = u_prof[0];
        let v_sfc = v_prof[0];
        let u_6km = interp_at_height(mean_depth, heights, u_prof);
        let v_6km = interp_at_height(mean_depth, heights, v_prof);
        let shear_u = u_6km - u_sfc;
        let shear_v = v_6km - v_sfc;

        let shear_mag = (shear_u * shear_u + shear_v * shear_v).sqrt();
        let (dev_u, dev_v) = if shear_mag > 0.1 {
            let scale = 7.5 / shear_mag;
            (shear_v * scale, -shear_u * scale)
        } else {
            (0.0, 0.0)
        };

        let storm_u = mean_u + dev_u;
        let storm_v = mean_v + dev_v;

        let mut srh = 0.0;
        for k in 0..nz - 1 {
            if heights[k] >= top_m {
                break;
            }

            let h_bot = heights[k];
            let h_top = heights[k + 1].min(top_m);
            if h_top <= h_bot {
                continue;
            }

            let u_bot = u_prof[k];
            let v_bot = v_prof[k];
            let (u_top_val, v_top_val) = if h_top < heights[k + 1] {
                let frac = (h_top - heights[k]) / (heights[k + 1] - heights[k]);
                (
                    u_prof[k] + frac * (u_prof[k + 1] - u_prof[k]),
                    v_prof[k] + frac * (v_prof[k + 1] - v_prof[k]),
                )
            } else {
                (u_prof[k + 1], v_prof[k + 1])
            };

            let sr_u_bot = u_bot - storm_u;
            let sr_v_bot = v_bot - storm_v;
            let sr_u_top = u_top_val - storm_u;
            let sr_v_top = v_top_val - storm_v;
            srh += sr_u_top * sr_v_bot - sr_u_bot * sr_v_top;
        }

        srh
    }

    fn compute_shear_column(
        heights: &[f64],
        u_prof: &[f64],
        v_prof: &[f64],
        bottom_m: f64,
        top_m: f64,
    ) -> f64 {
        let u_bot = interp_at_height(bottom_m, heights, u_prof);
        let v_bot = interp_at_height(bottom_m, heights, v_prof);
        let u_top = interp_at_height(top_m, heights, u_prof);
        let v_top = interp_at_height(top_m, heights, v_prof);

        let du = u_top - u_bot;
        let dv = v_top - v_bot;
        (du * du + dv * dv).sqrt()
    }

    pub fn compute_wind_diagnostics_bundle(
        u_3d: &[f64],
        v_3d: &[f64],
        height_agl_3d: &[f64],
        nx: usize,
        ny: usize,
        nz: usize,
    ) -> WindDiagnosticsBundle {
        let n2d = ny * nx;
        let results: Vec<(f64, f64, f64)> = (0..n2d)
            .into_par_iter()
            .map(|idx| {
                let j = idx / nx;
                let i = idx % nx;

                let u_col = extract_column(u_3d, nz, ny, nx, j, i);
                let v_col = extract_column(v_3d, nz, ny, nx, j, i);
                let h_col = extract_column(height_agl_3d, nz, ny, nx, j, i);
                let (h_prof, u_prof, v_prof) = ensure_surface_up_profile(h_col, u_col, v_col);

                (
                    compute_srh_column(&h_prof, &u_prof, &v_prof, 1000.0),
                    compute_srh_column(&h_prof, &u_prof, &v_prof, 3000.0),
                    compute_shear_column(&h_prof, &u_prof, &v_prof, 0.0, 6000.0),
                )
            })
            .collect();

        let mut srh_01km_m2s2 = Vec::with_capacity(n2d);
        let mut srh_03km_m2s2 = Vec::with_capacity(n2d);
        let mut shear_06km_ms = Vec::with_capacity(n2d);
        for (srh_01, srh_03, shear_06) in results {
            srh_01km_m2s2.push(srh_01);
            srh_03km_m2s2.push(srh_03);
            shear_06km_ms.push(shear_06);
        }

        WindDiagnosticsBundle {
            srh_01km_m2s2,
            srh_03km_m2s2,
            shear_06km_ms,
        }
    }

    fn push_ecape_level(
        pressure_pa: &mut Vec<f64>,
        height_m: &mut Vec<f64>,
        temp_k: &mut Vec<f64>,
        dewpoint_k: &mut Vec<f64>,
        u_ms: &mut Vec<f64>,
        v_ms: &mut Vec<f64>,
        p: f64,
        z: f64,
        t: f64,
        td: f64,
        u: f64,
        v: f64,
    ) {
        if !p.is_finite()
            || !z.is_finite()
            || !t.is_finite()
            || !td.is_finite()
            || !u.is_finite()
            || !v.is_finite()
        {
            return;
        }

        if let (Some(&last_p), Some(&last_z)) = (pressure_pa.last(), height_m.last()) {
            if p >= last_p || z <= last_z {
                return;
            }
        }

        pressure_pa.push(p);
        height_m.push(z);
        temp_k.push(t);
        dewpoint_k.push(td.min(t));
        u_ms.push(u);
        v_ms.push(v);
    }

    fn push_ecape_level_with_qv(
        pressure_pa: &mut Vec<f64>,
        height_m: &mut Vec<f64>,
        temp_k: &mut Vec<f64>,
        qv_kgkg: &mut Vec<f64>,
        u_ms: &mut Vec<f64>,
        v_ms: &mut Vec<f64>,
        p: f64,
        z: f64,
        t: f64,
        qv: f64,
        u: f64,
        v: f64,
    ) {
        if !p.is_finite()
            || !z.is_finite()
            || !t.is_finite()
            || !qv.is_finite()
            || !u.is_finite()
            || !v.is_finite()
        {
            return;
        }

        if let (Some(&last_p), Some(&last_z)) = (pressure_pa.last(), height_m.last()) {
            if p >= last_p || z <= last_z {
                return;
            }
        }

        pressure_pa.push(p);
        height_m.push(z);
        temp_k.push(t);
        qv_kgkg.push(qv.max(0.0));
        u_ms.push(u);
        v_ms.push(v);
    }

    fn build_surface_augmented_ecape_column(
        pressure_3d: &[f64],
        temperature_c_3d: &[f64],
        qvapor_3d: &[f64],
        height_agl_3d: &[f64],
        u_3d: &[f64],
        v_3d: &[f64],
        psfc_pa: f64,
        t2_k: f64,
        q2_kgkg: f64,
        u10_ms: f64,
        v10_ms: f64,
        nz: usize,
        nxy: usize,
        ij: usize,
        model_bottom_up: bool,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut pressure_pa = Vec::with_capacity(nz + 1);
        let mut height_m = Vec::with_capacity(nz + 1);
        let mut temp_k = Vec::with_capacity(nz + 1);
        let mut dewpoint_k = Vec::with_capacity(nz + 1);
        let mut u_ms = Vec::with_capacity(nz + 1);
        let mut v_ms = Vec::with_capacity(nz + 1);

        push_ecape_level(
            &mut pressure_pa,
            &mut height_m,
            &mut temp_k,
            &mut dewpoint_k,
            &mut u_ms,
            &mut v_ms,
            psfc_pa,
            0.0,
            t2_k,
            dewpoint_k_from_q(q2_kgkg, psfc_pa, t2_k),
            u10_ms,
            v10_ms,
        );

        let push_model_level = |k: usize,
                                pressure_pa: &mut Vec<f64>,
                                height_m: &mut Vec<f64>,
                                temp_k: &mut Vec<f64>,
                                dewpoint_k: &mut Vec<f64>,
                                u_ms: &mut Vec<f64>,
                                v_ms: &mut Vec<f64>| {
            let idx = k * nxy + ij;
            let tk = temperature_c_3d[idx] + 273.15;
            push_ecape_level(
                pressure_pa,
                height_m,
                temp_k,
                dewpoint_k,
                u_ms,
                v_ms,
                pressure_3d[idx],
                height_agl_3d[idx],
                tk,
                dewpoint_k_from_q(qvapor_3d[idx], pressure_3d[idx], tk),
                u_3d[idx],
                v_3d[idx],
            );
        };

        if model_bottom_up {
            for k in 0..nz {
                push_model_level(
                    k,
                    &mut pressure_pa,
                    &mut height_m,
                    &mut temp_k,
                    &mut dewpoint_k,
                    &mut u_ms,
                    &mut v_ms,
                );
            }
        } else {
            for k in (0..nz).rev() {
                push_model_level(
                    k,
                    &mut pressure_pa,
                    &mut height_m,
                    &mut temp_k,
                    &mut dewpoint_k,
                    &mut u_ms,
                    &mut v_ms,
                );
            }
        }

        (pressure_pa, height_m, temp_k, dewpoint_k, u_ms, v_ms)
    }

    fn build_surface_augmented_ecape_column_levels(
        pressure_levels_pa: &[f64],
        temperature_c_3d: &[f64],
        qvapor_3d: &[f64],
        height_agl_3d: &[f64],
        u_3d: &[f64],
        v_3d: &[f64],
        psfc_pa: f64,
        t2_k: f64,
        q2_kgkg: f64,
        u10_ms: f64,
        v10_ms: f64,
        nz: usize,
        nxy: usize,
        ij: usize,
        model_bottom_up: bool,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut pressure_pa = Vec::with_capacity(nz + 1);
        let mut height_m = Vec::with_capacity(nz + 1);
        let mut temp_k = Vec::with_capacity(nz + 1);
        let mut dewpoint_k = Vec::with_capacity(nz + 1);
        let mut u_ms = Vec::with_capacity(nz + 1);
        let mut v_ms = Vec::with_capacity(nz + 1);

        push_ecape_level(
            &mut pressure_pa,
            &mut height_m,
            &mut temp_k,
            &mut dewpoint_k,
            &mut u_ms,
            &mut v_ms,
            psfc_pa,
            0.0,
            t2_k,
            dewpoint_k_from_q(q2_kgkg, psfc_pa, t2_k),
            u10_ms,
            v10_ms,
        );

        let push_model_level = |k: usize,
                                pressure_pa: &mut Vec<f64>,
                                height_m: &mut Vec<f64>,
                                temp_k: &mut Vec<f64>,
                                dewpoint_k: &mut Vec<f64>,
                                u_ms: &mut Vec<f64>,
                                v_ms: &mut Vec<f64>| {
            let idx = k * nxy + ij;
            let pressure_k = pressure_levels_pa[k];
            let tk = temperature_c_3d[idx] + 273.15;
            push_ecape_level(
                pressure_pa,
                height_m,
                temp_k,
                dewpoint_k,
                u_ms,
                v_ms,
                pressure_k,
                height_agl_3d[idx],
                tk,
                dewpoint_k_from_q(qvapor_3d[idx], pressure_k, tk),
                u_3d[idx],
                v_3d[idx],
            );
        };

        if model_bottom_up {
            for k in 0..nz {
                push_model_level(
                    k,
                    &mut pressure_pa,
                    &mut height_m,
                    &mut temp_k,
                    &mut dewpoint_k,
                    &mut u_ms,
                    &mut v_ms,
                );
            }
        } else {
            for k in (0..nz).rev() {
                push_model_level(
                    k,
                    &mut pressure_pa,
                    &mut height_m,
                    &mut temp_k,
                    &mut dewpoint_k,
                    &mut u_ms,
                    &mut v_ms,
                );
            }
        }

        (pressure_pa, height_m, temp_k, dewpoint_k, u_ms, v_ms)
    }

    fn build_surface_augmented_ecape_column_with_qv(
        pressure_3d: &[f64],
        temperature_c_3d: &[f64],
        qvapor_3d: &[f64],
        height_agl_3d: &[f64],
        u_3d: &[f64],
        v_3d: &[f64],
        psfc_pa: f64,
        t2_k: f64,
        q2_kgkg: f64,
        u10_ms: f64,
        v10_ms: f64,
        nz: usize,
        nxy: usize,
        ij: usize,
        model_bottom_up: bool,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut pressure_pa = Vec::with_capacity(nz + 1);
        let mut height_m = Vec::with_capacity(nz + 1);
        let mut temp_k = Vec::with_capacity(nz + 1);
        let mut qv_kgkg = Vec::with_capacity(nz + 1);
        let mut u_ms = Vec::with_capacity(nz + 1);
        let mut v_ms = Vec::with_capacity(nz + 1);

        push_ecape_level_with_qv(
            &mut pressure_pa,
            &mut height_m,
            &mut temp_k,
            &mut qv_kgkg,
            &mut u_ms,
            &mut v_ms,
            psfc_pa,
            0.0,
            t2_k,
            q2_kgkg,
            u10_ms,
            v10_ms,
        );

        let push_model_level = |k: usize,
                                pressure_pa: &mut Vec<f64>,
                                height_m: &mut Vec<f64>,
                                temp_k: &mut Vec<f64>,
                                qv_kgkg: &mut Vec<f64>,
                                u_ms: &mut Vec<f64>,
                                v_ms: &mut Vec<f64>| {
            let idx = k * nxy + ij;
            push_ecape_level_with_qv(
                pressure_pa,
                height_m,
                temp_k,
                qv_kgkg,
                u_ms,
                v_ms,
                pressure_3d[idx],
                height_agl_3d[idx],
                temperature_c_3d[idx] + 273.15,
                qvapor_3d[idx],
                u_3d[idx],
                v_3d[idx],
            );
        };

        if model_bottom_up {
            for k in 0..nz {
                push_model_level(
                    k,
                    &mut pressure_pa,
                    &mut height_m,
                    &mut temp_k,
                    &mut qv_kgkg,
                    &mut u_ms,
                    &mut v_ms,
                );
            }
        } else {
            for k in (0..nz).rev() {
                push_model_level(
                    k,
                    &mut pressure_pa,
                    &mut height_m,
                    &mut temp_k,
                    &mut qv_kgkg,
                    &mut u_ms,
                    &mut v_ms,
                );
            }
        }

        (pressure_pa, height_m, temp_k, qv_kgkg, u_ms, v_ms)
    }

    fn build_surface_augmented_ecape_column_levels_with_qv(
        pressure_levels_pa: &[f64],
        temperature_c_3d: &[f64],
        qvapor_3d: &[f64],
        height_agl_3d: &[f64],
        u_3d: &[f64],
        v_3d: &[f64],
        psfc_pa: f64,
        t2_k: f64,
        q2_kgkg: f64,
        u10_ms: f64,
        v10_ms: f64,
        nz: usize,
        nxy: usize,
        ij: usize,
        model_bottom_up: bool,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut pressure_pa = Vec::with_capacity(nz + 1);
        let mut height_m = Vec::with_capacity(nz + 1);
        let mut temp_k = Vec::with_capacity(nz + 1);
        let mut qv_kgkg = Vec::with_capacity(nz + 1);
        let mut u_ms = Vec::with_capacity(nz + 1);
        let mut v_ms = Vec::with_capacity(nz + 1);

        push_ecape_level_with_qv(
            &mut pressure_pa,
            &mut height_m,
            &mut temp_k,
            &mut qv_kgkg,
            &mut u_ms,
            &mut v_ms,
            psfc_pa,
            0.0,
            t2_k,
            q2_kgkg,
            u10_ms,
            v10_ms,
        );

        let push_model_level = |k: usize,
                                pressure_pa: &mut Vec<f64>,
                                height_m: &mut Vec<f64>,
                                temp_k: &mut Vec<f64>,
                                qv_kgkg: &mut Vec<f64>,
                                u_ms: &mut Vec<f64>,
                                v_ms: &mut Vec<f64>| {
            let idx = k * nxy + ij;
            push_ecape_level_with_qv(
                pressure_pa,
                height_m,
                temp_k,
                qv_kgkg,
                u_ms,
                v_ms,
                pressure_levels_pa[k],
                height_agl_3d[idx],
                temperature_c_3d[idx] + 273.15,
                qvapor_3d[idx],
                u_3d[idx],
                v_3d[idx],
            );
        };

        if model_bottom_up {
            for k in 0..nz {
                push_model_level(
                    k,
                    &mut pressure_pa,
                    &mut height_m,
                    &mut temp_k,
                    &mut qv_kgkg,
                    &mut u_ms,
                    &mut v_ms,
                );
            }
        } else {
            for k in (0..nz).rev() {
                push_model_level(
                    k,
                    &mut pressure_pa,
                    &mut height_m,
                    &mut temp_k,
                    &mut qv_kgkg,
                    &mut u_ms,
                    &mut v_ms,
                );
            }
        }

        (pressure_pa, height_m, temp_k, qv_kgkg, u_ms, v_ms)
    }

    fn build_column_parcel_options(
        cape_type: EcapeCapeType,
        entrainment_rate: Option<f64>,
        pseudoadiabatic: Option<bool>,
        storm_motion_u_ms: f64,
        storm_motion_v_ms: f64,
    ) -> ParcelOptions {
        ParcelOptions {
            cape_type,
            storm_motion_type: EcapeStormMotionType::UserDefined,
            entrainment_rate,
            pseudoadiabatic,
            storm_motion_u_ms: Some(storm_motion_u_ms),
            storm_motion_v_ms: Some(storm_motion_v_ms),
            ..ParcelOptions::default()
        }
    }

    fn resolve_grid_storm_motion(
        pressure_pa: &[f64],
        height_m: &[f64],
        u_ms: &[f64],
        v_ms: &[f64],
        motion_type: EcapeStormMotionType,
        storm_u: Option<f64>,
        storm_v: Option<f64>,
    ) -> (f64, f64) {
        if let (Some(storm_motion_u_ms), Some(storm_motion_v_ms)) = (storm_u, storm_v) {
            return (storm_motion_u_ms, storm_motion_v_ms);
        }

        let pressure_hpa = pressure_pa.iter().map(|p| *p / 100.0).collect::<Vec<_>>();
        let (rm, lm, mean) =
            crate::calc::wind::bunkers_storm_motion(&pressure_hpa, u_ms, v_ms, height_m);
        match motion_type {
            EcapeStormMotionType::RightMoving => rm,
            EcapeStormMotionType::LeftMoving => lm,
            EcapeStormMotionType::MeanWind => mean,
            EcapeStormMotionType::UserDefined => rm,
        }
    }

    fn compute_ecape_column_result(
        pressure_pa: &[f64],
        height_m: &[f64],
        temp_k: &[f64],
        dewpoint_k: &[f64],
        u_ms: &[f64],
        v_ms: &[f64],
        options: &ParcelOptions,
    ) -> EcapeColumnResult {
        match calc_ecape_parcel(
            height_m,
            pressure_pa,
            temp_k,
            dewpoint_k,
            u_ms,
            v_ms,
            options,
        ) {
            Ok(result) => EcapeColumnResult {
                summary: EcapeSummary {
                    ecape: result.ecape_jkg,
                    ncape: result.ncape_jkg,
                    cape: result.cape_jkg,
                    cin: result.cin_jkg,
                    lfc: result.lfc_m.unwrap_or(0.0),
                    el: result.el_m.unwrap_or(0.0),
                },
                failed: false,
            },
            Err(_) => EcapeColumnResult {
                summary: EcapeSummary::default(),
                failed: true,
            },
        }
    }

    fn compute_analytic_ecape_column_result(
        pressure_pa: &[f64],
        height_m: &[f64],
        temp_k: &[f64],
        qv_kgkg: &[f64],
        u_ms: &[f64],
        v_ms: &[f64],
        options: &ParcelOptions,
    ) -> EcapeColumnResult {
        match calc_ecape_ncape(height_m, pressure_pa, temp_k, qv_kgkg, u_ms, v_ms, options) {
            Ok(result) => EcapeColumnResult {
                summary: EcapeSummary {
                    ecape: result.ecape_jkg,
                    ncape: result.ncape_jkg,
                    cape: result.cape_jkg,
                    cin: 0.0,
                    lfc: result.lfc_m.unwrap_or(0.0),
                    el: result.el_m.unwrap_or(0.0),
                },
                failed: false,
            },
            Err(_) => EcapeColumnResult {
                summary: EcapeSummary::default(),
                failed: true,
            },
        }
    }

    fn pack_ecape_results(results: Vec<EcapeColumnResult>) -> EcapeGridFieldsWithFailureMask {
        let mut ecape = Vec::with_capacity(results.len());
        let mut ncape = Vec::with_capacity(results.len());
        let mut cape = Vec::with_capacity(results.len());
        let mut cin = Vec::with_capacity(results.len());
        let mut lfc = Vec::with_capacity(results.len());
        let mut el = Vec::with_capacity(results.len());
        let mut failure_mask = Vec::with_capacity(results.len());

        for result in results {
            ecape.push(result.summary.ecape);
            ncape.push(result.summary.ncape);
            cape.push(result.summary.cape);
            cin.push(result.summary.cin);
            lfc.push(result.summary.lfc);
            el.push(result.summary.el);
            failure_mask.push(u8::from(result.failed));
        }

        EcapeGridFieldsWithFailureMask {
            fields: EcapeGridFields {
                ecape,
                ncape,
                cape,
                cin,
                lfc,
                el,
            },
            failure_mask,
        }
    }

    fn strip_failure_mask(fields: EcapeGridFieldsWithFailureMask) -> EcapeGridFields {
        fields.fields
    }

    /// Compute ECAPE-family diagnostics for every grid point.
    ///
    /// Returns `(ecape, ncape, cape, cin, lfc, el)` as six 1-D arrays of length `nx*ny`.
    pub fn compute_ecape(
        pressure_3d: &[f64],
        temperature_c_3d: &[f64],
        qvapor_3d: &[f64],
        height_agl_3d: &[f64],
        u_3d: &[f64],
        v_3d: &[f64],
        psfc: &[f64],
        t2: &[f64],
        q2: &[f64],
        u10: &[f64],
        v10: &[f64],
        nx: usize,
        ny: usize,
        nz: usize,
        parcel_type: &str,
        storm_motion_type: &str,
        entrainment_rate: Option<f64>,
        pseudoadiabatic: Option<bool>,
        storm_u: Option<f64>,
        storm_v: Option<f64>,
    ) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>), String> {
        let (ecape, ncape, cape, cin, lfc, el, _) = compute_ecape_with_failure_mask(
            pressure_3d,
            temperature_c_3d,
            qvapor_3d,
            height_agl_3d,
            u_3d,
            v_3d,
            psfc,
            t2,
            q2,
            u10,
            v10,
            nx,
            ny,
            nz,
            parcel_type,
            storm_motion_type,
            entrainment_rate,
            pseudoadiabatic,
            storm_u,
            storm_v,
        )?;
        Ok((ecape, ncape, cape, cin, lfc, el))
    }

    /// Compute ECAPE-family diagnostics and return a per-column failure mask.
    ///
    /// The first six return arrays match [`compute_ecape`]. The final `u8`
    /// array is `1` where the column fell back to zero-fill because it was too
    /// short after filtering or the ECAPE solver returned an error.
    pub fn compute_ecape_with_failure_mask(
        pressure_3d: &[f64],
        temperature_c_3d: &[f64],
        qvapor_3d: &[f64],
        height_agl_3d: &[f64],
        u_3d: &[f64],
        v_3d: &[f64],
        psfc: &[f64],
        t2: &[f64],
        q2: &[f64],
        u10: &[f64],
        v10: &[f64],
        nx: usize,
        ny: usize,
        nz: usize,
        parcel_type: &str,
        storm_motion_type: &str,
        entrainment_rate: Option<f64>,
        pseudoadiabatic: Option<bool>,
        storm_u: Option<f64>,
        storm_v: Option<f64>,
    ) -> Result<
        (
            Vec<f64>,
            Vec<f64>,
            Vec<f64>,
            Vec<f64>,
            Vec<f64>,
            Vec<f64>,
            Vec<u8>,
        ),
        String,
    > {
        let n2d = nx * ny;
        let expected_3d = n2d * nz;
        if pressure_3d.len() != expected_3d
            || temperature_c_3d.len() != expected_3d
            || qvapor_3d.len() != expected_3d
            || height_agl_3d.len() != expected_3d
            || u_3d.len() != expected_3d
            || v_3d.len() != expected_3d
        {
            return Err("ECAPE 3-D inputs must all have length nx*ny*nz".into());
        }
        if psfc.len() != n2d
            || t2.len() != n2d
            || q2.len() != n2d
            || u10.len() != n2d
            || v10.len() != n2d
        {
            return Err("ECAPE surface inputs must all have length nx*ny".into());
        }
        if storm_u.is_some() ^ storm_v.is_some() {
            return Err(
                "storm_u and storm_v must either both be provided or both be omitted".into(),
            );
        }

        let cape_type = resolve_ecape_cape_type(parcel_type)?;
        let mut motion_type = resolve_ecape_storm_motion_type(storm_motion_type)?;
        if storm_u.is_some() && storm_v.is_some() {
            motion_type = EcapeStormMotionType::UserDefined;
        }

        let results: Vec<EcapeColumnResult> = (0..n2d)
            .into_par_iter()
            .map(|ij| {
                let top_idx = ij;
                let bottom_idx = (nz - 1) * n2d + ij;
                let model_bottom_up = pressure_3d[top_idx] >= pressure_3d[bottom_idx]
                    || height_agl_3d[top_idx] <= height_agl_3d[bottom_idx];

                let (pressure_pa, height_m, temp_k, dewpoint_k, u_ms, v_ms) =
                    build_surface_augmented_ecape_column(
                        pressure_3d,
                        temperature_c_3d,
                        qvapor_3d,
                        height_agl_3d,
                        u_3d,
                        v_3d,
                        psfc[ij],
                        t2[ij],
                        q2[ij],
                        u10[ij],
                        v10[ij],
                        nz,
                        n2d,
                        ij,
                        model_bottom_up,
                    );

                if pressure_pa.len() < 2 {
                    return EcapeColumnResult {
                        summary: EcapeSummary::default(),
                        failed: true,
                    };
                }

                let mut options = ParcelOptions {
                    cape_type,
                    storm_motion_type: motion_type,
                    entrainment_rate,
                    pseudoadiabatic,
                    ..ParcelOptions::default()
                };

                if let (Some(storm_motion_u_ms), Some(storm_motion_v_ms)) = (storm_u, storm_v) {
                    options.storm_motion_type = EcapeStormMotionType::UserDefined;
                    options.storm_motion_u_ms = Some(storm_motion_u_ms);
                    options.storm_motion_v_ms = Some(storm_motion_v_ms);
                }

                match calc_ecape_parcel(
                    &height_m,
                    &pressure_pa,
                    &temp_k,
                    &dewpoint_k,
                    &u_ms,
                    &v_ms,
                    &options,
                ) {
                    Ok(result) => EcapeColumnResult {
                        summary: EcapeSummary {
                            ecape: result.ecape_jkg,
                            ncape: result.ncape_jkg,
                            cape: result.cape_jkg,
                            cin: result.cin_jkg,
                            lfc: result.lfc_m.unwrap_or(0.0),
                            el: result.el_m.unwrap_or(0.0),
                        },
                        failed: false,
                    },
                    Err(_) => EcapeColumnResult {
                        summary: EcapeSummary::default(),
                        failed: true,
                    },
                }
            })
            .collect();

        let packed = pack_ecape_results(results);
        Ok((
            packed.fields.ecape,
            packed.fields.ncape,
            packed.fields.cape,
            packed.fields.cin,
            packed.fields.lfc,
            packed.fields.el,
            packed.failure_mask,
        ))
    }

    /// Compute SB/ML/MU ECAPE-family diagnostics together with one shared
    /// surface-augmented column build per grid point.
    pub fn compute_ecape_triplet_with_failure_mask(
        pressure_3d: &[f64],
        temperature_c_3d: &[f64],
        qvapor_3d: &[f64],
        height_agl_3d: &[f64],
        u_3d: &[f64],
        v_3d: &[f64],
        psfc: &[f64],
        t2: &[f64],
        q2: &[f64],
        u10: &[f64],
        v10: &[f64],
        nx: usize,
        ny: usize,
        nz: usize,
        storm_motion_type: &str,
        entrainment_rate: Option<f64>,
        pseudoadiabatic: Option<bool>,
        storm_u: Option<f64>,
        storm_v: Option<f64>,
    ) -> Result<EcapeGridTripletWithFailureMask, String> {
        let n2d = nx * ny;
        let expected_3d = n2d * nz;
        if pressure_3d.len() != expected_3d
            || temperature_c_3d.len() != expected_3d
            || qvapor_3d.len() != expected_3d
            || height_agl_3d.len() != expected_3d
            || u_3d.len() != expected_3d
            || v_3d.len() != expected_3d
        {
            return Err("ECAPE 3-D inputs must all have length nx*ny*nz".into());
        }
        if psfc.len() != n2d
            || t2.len() != n2d
            || q2.len() != n2d
            || u10.len() != n2d
            || v10.len() != n2d
        {
            return Err("ECAPE surface inputs must all have length nx*ny".into());
        }
        if storm_u.is_some() ^ storm_v.is_some() {
            return Err(
                "storm_u and storm_v must either both be provided or both be omitted".into(),
            );
        }

        let motion_type = resolve_ecape_storm_motion_type(storm_motion_type)?;
        let failed_triplet = EcapeTripletColumnResult {
            sb: EcapeColumnResult {
                summary: EcapeSummary::default(),
                failed: true,
            },
            ml: EcapeColumnResult {
                summary: EcapeSummary::default(),
                failed: true,
            },
            mu: EcapeColumnResult {
                summary: EcapeSummary::default(),
                failed: true,
            },
        };

        let results: Vec<EcapeTripletColumnResult> = (0..n2d)
            .into_par_iter()
            .map(|ij| {
                let top_idx = ij;
                let bottom_idx = (nz - 1) * n2d + ij;
                let model_bottom_up = pressure_3d[top_idx] >= pressure_3d[bottom_idx]
                    || height_agl_3d[top_idx] <= height_agl_3d[bottom_idx];

                let (pressure_pa, height_m, temp_k, dewpoint_k, u_ms, v_ms) =
                    build_surface_augmented_ecape_column(
                        pressure_3d,
                        temperature_c_3d,
                        qvapor_3d,
                        height_agl_3d,
                        u_3d,
                        v_3d,
                        psfc[ij],
                        t2[ij],
                        q2[ij],
                        u10[ij],
                        v10[ij],
                        nz,
                        n2d,
                        ij,
                        model_bottom_up,
                    );

                if pressure_pa.len() < 2 {
                    return failed_triplet;
                }

                let (storm_motion_u_ms, storm_motion_v_ms) = resolve_grid_storm_motion(
                    &pressure_pa,
                    &height_m,
                    &u_ms,
                    &v_ms,
                    motion_type,
                    storm_u,
                    storm_v,
                );

                let sb_options = build_column_parcel_options(
                    EcapeCapeType::SurfaceBased,
                    entrainment_rate,
                    pseudoadiabatic,
                    storm_motion_u_ms,
                    storm_motion_v_ms,
                );
                let ml_options = build_column_parcel_options(
                    EcapeCapeType::MixedLayer,
                    entrainment_rate,
                    pseudoadiabatic,
                    storm_motion_u_ms,
                    storm_motion_v_ms,
                );
                let mu_options = build_column_parcel_options(
                    EcapeCapeType::MostUnstable,
                    entrainment_rate,
                    pseudoadiabatic,
                    storm_motion_u_ms,
                    storm_motion_v_ms,
                );

                EcapeTripletColumnResult {
                    sb: compute_ecape_column_result(
                        &pressure_pa,
                        &height_m,
                        &temp_k,
                        &dewpoint_k,
                        &u_ms,
                        &v_ms,
                        &sb_options,
                    ),
                    ml: compute_ecape_column_result(
                        &pressure_pa,
                        &height_m,
                        &temp_k,
                        &dewpoint_k,
                        &u_ms,
                        &v_ms,
                        &ml_options,
                    ),
                    mu: compute_ecape_column_result(
                        &pressure_pa,
                        &height_m,
                        &temp_k,
                        &dewpoint_k,
                        &u_ms,
                        &v_ms,
                        &mu_options,
                    ),
                }
            })
            .collect();

        let mut sb_results = Vec::with_capacity(n2d);
        let mut ml_results = Vec::with_capacity(n2d);
        let mut mu_results = Vec::with_capacity(n2d);
        for result in results {
            sb_results.push(result.sb);
            ml_results.push(result.ml);
            mu_results.push(result.mu);
        }

        Ok(EcapeGridTripletWithFailureMask {
            sb: pack_ecape_results(sb_results),
            ml: pack_ecape_results(ml_results),
            mu: pack_ecape_results(mu_results),
        })
    }

    /// Compute SB/ML/MU ECAPE-family diagnostics together using one
    /// pressure-per-level vector shared across the whole grid.
    pub fn compute_ecape_triplet_with_failure_mask_levels(
        pressure_levels_pa: &[f64],
        temperature_c_3d: &[f64],
        qvapor_3d: &[f64],
        height_agl_3d: &[f64],
        u_3d: &[f64],
        v_3d: &[f64],
        psfc: &[f64],
        t2: &[f64],
        q2: &[f64],
        u10: &[f64],
        v10: &[f64],
        nx: usize,
        ny: usize,
        nz: usize,
        storm_motion_type: &str,
        entrainment_rate: Option<f64>,
        pseudoadiabatic: Option<bool>,
        storm_u: Option<f64>,
        storm_v: Option<f64>,
    ) -> Result<EcapeGridTripletWithFailureMask, String> {
        let n2d = nx * ny;
        let expected_3d = n2d * nz;
        if pressure_levels_pa.len() != nz {
            return Err("ECAPE pressure levels input must have length nz".into());
        }
        if temperature_c_3d.len() != expected_3d
            || qvapor_3d.len() != expected_3d
            || height_agl_3d.len() != expected_3d
            || u_3d.len() != expected_3d
            || v_3d.len() != expected_3d
        {
            return Err("ECAPE 3-D inputs must all have length nx*ny*nz".into());
        }
        if psfc.len() != n2d
            || t2.len() != n2d
            || q2.len() != n2d
            || u10.len() != n2d
            || v10.len() != n2d
        {
            return Err("ECAPE surface inputs must all have length nx*ny".into());
        }
        if storm_u.is_some() ^ storm_v.is_some() {
            return Err(
                "storm_u and storm_v must either both be provided or both be omitted".into(),
            );
        }

        let motion_type = resolve_ecape_storm_motion_type(storm_motion_type)?;
        let failed_triplet = EcapeTripletColumnResult {
            sb: EcapeColumnResult {
                summary: EcapeSummary::default(),
                failed: true,
            },
            ml: EcapeColumnResult {
                summary: EcapeSummary::default(),
                failed: true,
            },
            mu: EcapeColumnResult {
                summary: EcapeSummary::default(),
                failed: true,
            },
        };

        let model_bottom_up = pressure_levels_pa[0] >= pressure_levels_pa[nz - 1];
        let results: Vec<EcapeTripletColumnResult> = (0..n2d)
            .into_par_iter()
            .map(|ij| {
                let top_idx = ij;
                let bottom_idx = (nz - 1) * n2d + ij;
                let column_bottom_up =
                    model_bottom_up || height_agl_3d[top_idx] <= height_agl_3d[bottom_idx];

                let (pressure_pa, height_m, temp_k, dewpoint_k, u_ms, v_ms) =
                    build_surface_augmented_ecape_column_levels(
                        pressure_levels_pa,
                        temperature_c_3d,
                        qvapor_3d,
                        height_agl_3d,
                        u_3d,
                        v_3d,
                        psfc[ij],
                        t2[ij],
                        q2[ij],
                        u10[ij],
                        v10[ij],
                        nz,
                        n2d,
                        ij,
                        column_bottom_up,
                    );

                if pressure_pa.len() < 2 {
                    return failed_triplet;
                }

                let (storm_motion_u_ms, storm_motion_v_ms) = resolve_grid_storm_motion(
                    &pressure_pa,
                    &height_m,
                    &u_ms,
                    &v_ms,
                    motion_type,
                    storm_u,
                    storm_v,
                );

                let sb_options = build_column_parcel_options(
                    EcapeCapeType::SurfaceBased,
                    entrainment_rate,
                    pseudoadiabatic,
                    storm_motion_u_ms,
                    storm_motion_v_ms,
                );
                let ml_options = build_column_parcel_options(
                    EcapeCapeType::MixedLayer,
                    entrainment_rate,
                    pseudoadiabatic,
                    storm_motion_u_ms,
                    storm_motion_v_ms,
                );
                let mu_options = build_column_parcel_options(
                    EcapeCapeType::MostUnstable,
                    entrainment_rate,
                    pseudoadiabatic,
                    storm_motion_u_ms,
                    storm_motion_v_ms,
                );

                EcapeTripletColumnResult {
                    sb: compute_ecape_column_result(
                        &pressure_pa,
                        &height_m,
                        &temp_k,
                        &dewpoint_k,
                        &u_ms,
                        &v_ms,
                        &sb_options,
                    ),
                    ml: compute_ecape_column_result(
                        &pressure_pa,
                        &height_m,
                        &temp_k,
                        &dewpoint_k,
                        &u_ms,
                        &v_ms,
                        &ml_options,
                    ),
                    mu: compute_ecape_column_result(
                        &pressure_pa,
                        &height_m,
                        &temp_k,
                        &dewpoint_k,
                        &u_ms,
                        &v_ms,
                        &mu_options,
                    ),
                }
            })
            .collect();

        let mut sb_results = Vec::with_capacity(n2d);
        let mut ml_results = Vec::with_capacity(n2d);
        let mut mu_results = Vec::with_capacity(n2d);
        for result in results {
            sb_results.push(result.sb);
            ml_results.push(result.ml);
            mu_results.push(result.mu);
        }

        Ok(EcapeGridTripletWithFailureMask {
            sb: pack_ecape_results(sb_results),
            ml: pack_ecape_results(ml_results),
            mu: pack_ecape_results(mu_results),
        })
    }

    /// Compute Peters-style analytic ECAPE/NCAPE diagnostics for SB/ML/MU
    /// parcels without following the entraining parcel path.
    pub fn compute_analytic_ecape_triplet_with_failure_mask(
        pressure_3d: &[f64],
        temperature_c_3d: &[f64],
        qvapor_3d: &[f64],
        height_agl_3d: &[f64],
        u_3d: &[f64],
        v_3d: &[f64],
        psfc: &[f64],
        t2: &[f64],
        q2: &[f64],
        u10: &[f64],
        v10: &[f64],
        nx: usize,
        ny: usize,
        nz: usize,
        storm_motion_type: &str,
        pseudoadiabatic: Option<bool>,
        storm_u: Option<f64>,
        storm_v: Option<f64>,
    ) -> Result<EcapeGridTripletWithFailureMask, String> {
        let n2d = nx * ny;
        let expected_3d = n2d * nz;
        if pressure_3d.len() != expected_3d
            || temperature_c_3d.len() != expected_3d
            || qvapor_3d.len() != expected_3d
            || height_agl_3d.len() != expected_3d
            || u_3d.len() != expected_3d
            || v_3d.len() != expected_3d
        {
            return Err("analytic ECAPE 3-D inputs must all have length nx*ny*nz".into());
        }
        if psfc.len() != n2d
            || t2.len() != n2d
            || q2.len() != n2d
            || u10.len() != n2d
            || v10.len() != n2d
        {
            return Err("analytic ECAPE surface inputs must all have length nx*ny".into());
        }
        if storm_u.is_some() ^ storm_v.is_some() {
            return Err(
                "storm_u and storm_v must either both be provided or both be omitted".into(),
            );
        }

        let motion_type = resolve_ecape_storm_motion_type(storm_motion_type)?;
        let failed_triplet = EcapeTripletColumnResult {
            sb: EcapeColumnResult {
                summary: EcapeSummary::default(),
                failed: true,
            },
            ml: EcapeColumnResult {
                summary: EcapeSummary::default(),
                failed: true,
            },
            mu: EcapeColumnResult {
                summary: EcapeSummary::default(),
                failed: true,
            },
        };

        let results: Vec<EcapeTripletColumnResult> = (0..n2d)
            .into_par_iter()
            .map(|ij| {
                let top_idx = ij;
                let bottom_idx = (nz - 1) * n2d + ij;
                let model_bottom_up = pressure_3d[top_idx] >= pressure_3d[bottom_idx]
                    || height_agl_3d[top_idx] <= height_agl_3d[bottom_idx];

                let (pressure_pa, height_m, temp_k, qv_kgkg, u_ms, v_ms) =
                    build_surface_augmented_ecape_column_with_qv(
                        pressure_3d,
                        temperature_c_3d,
                        qvapor_3d,
                        height_agl_3d,
                        u_3d,
                        v_3d,
                        psfc[ij],
                        t2[ij],
                        q2[ij],
                        u10[ij],
                        v10[ij],
                        nz,
                        n2d,
                        ij,
                        model_bottom_up,
                    );

                if pressure_pa.len() < 2 {
                    return failed_triplet;
                }

                let (storm_motion_u_ms, storm_motion_v_ms) = resolve_grid_storm_motion(
                    &pressure_pa,
                    &height_m,
                    &u_ms,
                    &v_ms,
                    motion_type,
                    storm_u,
                    storm_v,
                );

                let sb_options = build_column_parcel_options(
                    EcapeCapeType::SurfaceBased,
                    None,
                    pseudoadiabatic,
                    storm_motion_u_ms,
                    storm_motion_v_ms,
                );
                let ml_options = build_column_parcel_options(
                    EcapeCapeType::MixedLayer,
                    None,
                    pseudoadiabatic,
                    storm_motion_u_ms,
                    storm_motion_v_ms,
                );
                let mu_options = build_column_parcel_options(
                    EcapeCapeType::MostUnstable,
                    None,
                    pseudoadiabatic,
                    storm_motion_u_ms,
                    storm_motion_v_ms,
                );

                EcapeTripletColumnResult {
                    sb: compute_analytic_ecape_column_result(
                        &pressure_pa,
                        &height_m,
                        &temp_k,
                        &qv_kgkg,
                        &u_ms,
                        &v_ms,
                        &sb_options,
                    ),
                    ml: compute_analytic_ecape_column_result(
                        &pressure_pa,
                        &height_m,
                        &temp_k,
                        &qv_kgkg,
                        &u_ms,
                        &v_ms,
                        &ml_options,
                    ),
                    mu: compute_analytic_ecape_column_result(
                        &pressure_pa,
                        &height_m,
                        &temp_k,
                        &qv_kgkg,
                        &u_ms,
                        &v_ms,
                        &mu_options,
                    ),
                }
            })
            .collect();

        let mut sb_results = Vec::with_capacity(n2d);
        let mut ml_results = Vec::with_capacity(n2d);
        let mut mu_results = Vec::with_capacity(n2d);
        for result in results {
            sb_results.push(result.sb);
            ml_results.push(result.ml);
            mu_results.push(result.mu);
        }

        Ok(EcapeGridTripletWithFailureMask {
            sb: pack_ecape_results(sb_results),
            ml: pack_ecape_results(ml_results),
            mu: pack_ecape_results(mu_results),
        })
    }

    /// Compute Peters-style analytic ECAPE/NCAPE diagnostics when pressure is
    /// supplied as one shared level vector for the whole grid.
    pub fn compute_analytic_ecape_triplet_with_failure_mask_levels(
        pressure_levels_pa: &[f64],
        temperature_c_3d: &[f64],
        qvapor_3d: &[f64],
        height_agl_3d: &[f64],
        u_3d: &[f64],
        v_3d: &[f64],
        psfc: &[f64],
        t2: &[f64],
        q2: &[f64],
        u10: &[f64],
        v10: &[f64],
        nx: usize,
        ny: usize,
        nz: usize,
        storm_motion_type: &str,
        pseudoadiabatic: Option<bool>,
        storm_u: Option<f64>,
        storm_v: Option<f64>,
    ) -> Result<EcapeGridTripletWithFailureMask, String> {
        let n2d = nx * ny;
        let expected_3d = n2d * nz;
        if pressure_levels_pa.len() != nz {
            return Err("analytic ECAPE pressure levels input must have length nz".into());
        }
        if temperature_c_3d.len() != expected_3d
            || qvapor_3d.len() != expected_3d
            || height_agl_3d.len() != expected_3d
            || u_3d.len() != expected_3d
            || v_3d.len() != expected_3d
        {
            return Err("analytic ECAPE 3-D inputs must all have length nx*ny*nz".into());
        }
        if psfc.len() != n2d
            || t2.len() != n2d
            || q2.len() != n2d
            || u10.len() != n2d
            || v10.len() != n2d
        {
            return Err("analytic ECAPE surface inputs must all have length nx*ny".into());
        }
        if storm_u.is_some() ^ storm_v.is_some() {
            return Err(
                "storm_u and storm_v must either both be provided or both be omitted".into(),
            );
        }

        let motion_type = resolve_ecape_storm_motion_type(storm_motion_type)?;
        let failed_triplet = EcapeTripletColumnResult {
            sb: EcapeColumnResult {
                summary: EcapeSummary::default(),
                failed: true,
            },
            ml: EcapeColumnResult {
                summary: EcapeSummary::default(),
                failed: true,
            },
            mu: EcapeColumnResult {
                summary: EcapeSummary::default(),
                failed: true,
            },
        };

        let model_bottom_up = pressure_levels_pa[0] >= pressure_levels_pa[nz - 1];
        let results: Vec<EcapeTripletColumnResult> = (0..n2d)
            .into_par_iter()
            .map(|ij| {
                let top_idx = ij;
                let bottom_idx = (nz - 1) * n2d + ij;
                let column_bottom_up =
                    model_bottom_up || height_agl_3d[top_idx] <= height_agl_3d[bottom_idx];

                let (pressure_pa, height_m, temp_k, qv_kgkg, u_ms, v_ms) =
                    build_surface_augmented_ecape_column_levels_with_qv(
                        pressure_levels_pa,
                        temperature_c_3d,
                        qvapor_3d,
                        height_agl_3d,
                        u_3d,
                        v_3d,
                        psfc[ij],
                        t2[ij],
                        q2[ij],
                        u10[ij],
                        v10[ij],
                        nz,
                        n2d,
                        ij,
                        column_bottom_up,
                    );

                if pressure_pa.len() < 2 {
                    return failed_triplet;
                }

                let (storm_motion_u_ms, storm_motion_v_ms) = resolve_grid_storm_motion(
                    &pressure_pa,
                    &height_m,
                    &u_ms,
                    &v_ms,
                    motion_type,
                    storm_u,
                    storm_v,
                );

                let sb_options = build_column_parcel_options(
                    EcapeCapeType::SurfaceBased,
                    None,
                    pseudoadiabatic,
                    storm_motion_u_ms,
                    storm_motion_v_ms,
                );
                let ml_options = build_column_parcel_options(
                    EcapeCapeType::MixedLayer,
                    None,
                    pseudoadiabatic,
                    storm_motion_u_ms,
                    storm_motion_v_ms,
                );
                let mu_options = build_column_parcel_options(
                    EcapeCapeType::MostUnstable,
                    None,
                    pseudoadiabatic,
                    storm_motion_u_ms,
                    storm_motion_v_ms,
                );

                EcapeTripletColumnResult {
                    sb: compute_analytic_ecape_column_result(
                        &pressure_pa,
                        &height_m,
                        &temp_k,
                        &qv_kgkg,
                        &u_ms,
                        &v_ms,
                        &sb_options,
                    ),
                    ml: compute_analytic_ecape_column_result(
                        &pressure_pa,
                        &height_m,
                        &temp_k,
                        &qv_kgkg,
                        &u_ms,
                        &v_ms,
                        &ml_options,
                    ),
                    mu: compute_analytic_ecape_column_result(
                        &pressure_pa,
                        &height_m,
                        &temp_k,
                        &qv_kgkg,
                        &u_ms,
                        &v_ms,
                        &mu_options,
                    ),
                }
            })
            .collect();

        let mut sb_results = Vec::with_capacity(n2d);
        let mut ml_results = Vec::with_capacity(n2d);
        let mut mu_results = Vec::with_capacity(n2d);
        for result in results {
            sb_results.push(result.sb);
            ml_results.push(result.ml);
            mu_results.push(result.mu);
        }

        Ok(EcapeGridTripletWithFailureMask {
            sb: pack_ecape_results(sb_results),
            ml: pack_ecape_results(ml_results),
            mu: pack_ecape_results(mu_results),
        })
    }

    /// Compute SB/ML/MU ECAPE-family diagnostics together without returning
    /// per-column failure masks.
    pub fn compute_ecape_triplet(
        pressure_3d: &[f64],
        temperature_c_3d: &[f64],
        qvapor_3d: &[f64],
        height_agl_3d: &[f64],
        u_3d: &[f64],
        v_3d: &[f64],
        psfc: &[f64],
        t2: &[f64],
        q2: &[f64],
        u10: &[f64],
        v10: &[f64],
        nx: usize,
        ny: usize,
        nz: usize,
        storm_motion_type: &str,
        entrainment_rate: Option<f64>,
        pseudoadiabatic: Option<bool>,
        storm_u: Option<f64>,
        storm_v: Option<f64>,
    ) -> Result<EcapeGridTriplet, String> {
        let triplet = compute_ecape_triplet_with_failure_mask(
            pressure_3d,
            temperature_c_3d,
            qvapor_3d,
            height_agl_3d,
            u_3d,
            v_3d,
            psfc,
            t2,
            q2,
            u10,
            v10,
            nx,
            ny,
            nz,
            storm_motion_type,
            entrainment_rate,
            pseudoadiabatic,
            storm_u,
            storm_v,
        )?;

        Ok(EcapeGridTriplet {
            sb: strip_failure_mask(triplet.sb),
            ml: strip_failure_mask(triplet.ml),
            mu: strip_failure_mask(triplet.mu),
        })
    }

    /// Compute SB/ML/MU ECAPE-family diagnostics together without returning
    /// per-column failure masks, using pressure specified once per model level.
    pub fn compute_ecape_triplet_levels(
        pressure_levels_pa: &[f64],
        temperature_c_3d: &[f64],
        qvapor_3d: &[f64],
        height_agl_3d: &[f64],
        u_3d: &[f64],
        v_3d: &[f64],
        psfc: &[f64],
        t2: &[f64],
        q2: &[f64],
        u10: &[f64],
        v10: &[f64],
        nx: usize,
        ny: usize,
        nz: usize,
        storm_motion_type: &str,
        entrainment_rate: Option<f64>,
        pseudoadiabatic: Option<bool>,
        storm_u: Option<f64>,
        storm_v: Option<f64>,
    ) -> Result<EcapeGridTriplet, String> {
        let triplet = compute_ecape_triplet_with_failure_mask_levels(
            pressure_levels_pa,
            temperature_c_3d,
            qvapor_3d,
            height_agl_3d,
            u_3d,
            v_3d,
            psfc,
            t2,
            q2,
            u10,
            v10,
            nx,
            ny,
            nz,
            storm_motion_type,
            entrainment_rate,
            pseudoadiabatic,
            storm_u,
            storm_v,
        )?;

        Ok(EcapeGridTriplet {
            sb: strip_failure_mask(triplet.sb),
            ml: strip_failure_mask(triplet.ml),
            mu: strip_failure_mask(triplet.mu),
        })
    }

    // ── 2-D grid composite parameters ──
    pub use wx_math::composite::compute_ehi;
    pub use wx_math::composite::compute_scp;
    pub use wx_math::composite::compute_stp;
    pub use wx_math::composite::critical_angle;
    pub use wx_math::composite::derecho_composite_parameter;
    pub use wx_math::composite::significant_hail_parameter;
    pub use wx_math::composite::supercell_composite_parameter;

    // ── Reflectivity composites ──
    pub use wx_math::composite::composite_reflectivity_from_hydrometeors;
    pub use wx_math::composite::composite_reflectivity_from_refl;
}

/// Significant Tornado Parameter (STP).
///
/// STP combines mixed-layer CAPE, LCL height, 0-1 km storm-relative helicity,
/// and 0-6 km bulk shear magnitude into a single composite favoring significant
/// (EF2+) tornadoes.
///
/// ```text
/// STP = (mlCAPE / 1500) * ((2000 - mlLCL) / 1000) * (SRH / 150) * (shear / 20)
/// ```
///
/// Each term is clamped so it cannot go below 0 (and the LCL term is also
/// capped at 1.0 when LCL <= 1000 m, per Thompson et al. 2003).
///
/// # Arguments
/// * `mlcape` — Mixed-layer CAPE (J/kg)
/// * `lcl_height_m` — Mixed-layer LCL height AGL (m)
/// * `srh_0_1km` — 0-1 km storm-relative helicity (m^2/s^2)
/// * `bulk_shear_0_6km_ms` — 0-6 km bulk wind shear magnitude (m/s)
///
/// # Returns
/// Dimensionless STP value. Values above 1.0 are increasingly favorable for
/// significant tornadoes.
///
/// # References
/// Thompson, R. L., R. Edwards, J. A. Hart, K. L. Elmore, and P. Markowski,
/// 2003: Close proximity soundings within supercell environments obtained from
/// the Rapid Update Cycle. *Wea. Forecasting*, **18**, 1243-1261.
pub fn significant_tornado_parameter(
    sbcape: f64,
    lcl_height_m: f64,
    srh_0_1km: f64,
    bulk_shear_0_6km_ms: f64,
) -> f64 {
    // CAPE term: SBCAPE / 1500, floored at 0
    // Fixed-layer STP uses surface-based CAPE (not MLCAPE)
    let cape_term = (sbcape / 1500.0).max(0.0);

    // LCL term: (2000 - LCL) / 1000
    //   - Capped at 1.0 when LCL <= 1000 m (very low LCL is always favorable)
    //   - Floored at 0.0 when LCL >= 2000 m (too high, unfavorable)
    let lcl_term = if lcl_height_m <= 1000.0 {
        1.0
    } else {
        ((2000.0 - lcl_height_m) / 1000.0).clamp(0.0, 1.0)
    };

    // SRH term: SRH / 150, floored at 0
    let srh_term = (srh_0_1km / 150.0).max(0.0);

    // Shear term: zero when < 12.5 m/s, capped at 30 m/s, then / 20
    // (per Thompson et al. 2003 / MetPy: shear < 12.5 m/s => 0)
    let shear_term = if bulk_shear_0_6km_ms < 12.5 {
        0.0
    } else {
        (bulk_shear_0_6km_ms.min(30.0) / 20.0).max(0.0)
    };

    cape_term * lcl_term * srh_term * shear_term
}

/// Supercell Composite Parameter (SCP).
///
/// SCP combines most-unstable CAPE, effective-layer storm-relative helicity,
/// and effective bulk shear magnitude.
///
/// ```text
/// SCP = (muCAPE / 1000) * (SRH / 50) * (shear / 20)
/// ```
///
/// Each term is floored at 0.
///
/// # Arguments
/// * `mucape` — Most-unstable CAPE (J/kg)
/// * `srh_eff` — Effective-layer storm-relative helicity (m^2/s^2)
/// * `bulk_shear_eff_ms` — Effective bulk shear magnitude (m/s)
///
/// # Returns
/// Dimensionless SCP value. Values >= 1.0 favor supercells.
///
/// # References
/// Thompson, R. L., R. Edwards, and C. M. Mead, 2004: An update to the
/// supercell composite and significant tornado parameters. Preprints, 22nd
/// Conf. on Severe Local Storms, Hyannis, MA.
pub fn supercell_composite_parameter(mucape: f64, srh_eff: f64, bulk_shear_eff_ms: f64) -> f64 {
    let cape_term = (mucape / 1000.0).max(0.0);
    let srh_term = (srh_eff / 50.0).max(0.0);
    // Shear term: zero when < 10 m/s, capped at 1.0 (i.e. clipped to 20 m/s then / 20)
    // (per Thompson et al. 2004 / MetPy)
    let shear_term = if bulk_shear_eff_ms < 10.0 {
        0.0
    } else {
        (bulk_shear_eff_ms.min(20.0) / 20.0).max(0.0)
    };

    cape_term * srh_term * shear_term
}

/// Critical angle between the storm-relative inflow vector and the 0-500 m
/// shear vector.
///
/// A critical angle near 90 degrees is most favorable for low-level
/// mesocyclone development because it means the storm-relative inflow is
/// perpendicular to the low-level shear, maximizing streamwise vorticity
/// tilting.
///
/// # Arguments
/// * `storm_u`, `storm_v` — Storm motion components (m/s)
/// * `u_sfc`, `v_sfc` — Surface wind components (m/s)
/// * `u_500m`, `v_500m` — Wind at 500 m AGL (m/s)
///
/// # Returns
/// The angle in degrees [0, 180] between the two vectors. Returns 0.0 if
/// either vector has near-zero magnitude.
///
/// # References
/// Esterheld, J. M., and D. J. Giuliano, 2008: Discriminating between
/// tornadic and non-tornadic supercells: A new hodograph technique. *E-Journal
/// of Severe Storms Meteorology*, **3(2)**, 1-13.
pub fn critical_angle(
    storm_u: f64,
    storm_v: f64,
    u_sfc: f64,
    v_sfc: f64,
    u_500m: f64,
    v_500m: f64,
) -> f64 {
    // Storm-relative inflow vector (surface wind relative to storm)
    let inflow_u = u_sfc - storm_u;
    let inflow_v = v_sfc - storm_v;

    // 0-500 m shear vector
    let shear_u = u_500m - u_sfc;
    let shear_v = v_500m - v_sfc;

    let mag_inflow = (inflow_u * inflow_u + inflow_v * inflow_v).sqrt();
    let mag_shear = (shear_u * shear_u + shear_v * shear_v).sqrt();

    if mag_inflow < 1e-10 || mag_shear < 1e-10 {
        return 0.0;
    }

    let cos_angle = (inflow_u * shear_u + inflow_v * shear_v) / (mag_inflow * mag_shear);
    // Clamp to [-1, 1] to avoid NaN from floating-point rounding
    cos_angle.clamp(-1.0, 1.0).acos() * (180.0 / PI)
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ecape_rs::{calc_ecape_parcel, CapeType, ParcelOptions, StormMotionType};

    // ── significant_tornado_parameter ──

    #[test]
    fn test_stp_nominal() {
        // All terms at their normalizing values => STP = 1.0
        let stp = significant_tornado_parameter(1500.0, 1000.0, 150.0, 20.0);
        assert!(
            (stp - 1.0).abs() < 1e-10,
            "STP with nominal values = {stp}, expected 1.0"
        );
    }

    #[test]
    fn test_stp_zero_cape() {
        let stp = significant_tornado_parameter(0.0, 800.0, 200.0, 25.0);
        assert!((stp - 0.0).abs() < 1e-10, "STP should be 0 with zero CAPE");
    }

    #[test]
    fn test_stp_high_lcl_zero() {
        // LCL at 2000 m makes the LCL term 0 => STP = 0
        let stp = significant_tornado_parameter(2000.0, 2000.0, 200.0, 25.0);
        assert!((stp - 0.0).abs() < 1e-10, "STP should be 0 with LCL=2000m");
    }

    #[test]
    fn test_stp_very_low_lcl_capped() {
        // LCL <= 1000m => LCL term capped at 1.0
        let stp_500 = significant_tornado_parameter(1500.0, 500.0, 150.0, 20.0);
        let stp_1000 = significant_tornado_parameter(1500.0, 1000.0, 150.0, 20.0);
        assert!(
            (stp_500 - stp_1000).abs() < 1e-10,
            "LCL 500m and 1000m should give same STP"
        );
    }

    #[test]
    fn test_stp_shear_capped() {
        // Shear term capped at 1.5 when shear = 30 m/s (30/20 = 1.5)
        let stp_30 = significant_tornado_parameter(1500.0, 1000.0, 150.0, 30.0);
        let stp_50 = significant_tornado_parameter(1500.0, 1000.0, 150.0, 50.0);
        assert!(
            (stp_30 - stp_50).abs() < 1e-10,
            "STP should cap shear at 1.5: stp_30={stp_30}, stp_50={stp_50}"
        );
        assert!((stp_30 - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_stp_weak_shear_zero() {
        // Shear below 12.5 m/s => shear_term = 0 => STP = 0 (per MetPy)
        let stp = significant_tornado_parameter(2000.0, 800.0, 150.0, 10.0);
        assert!(
            (stp - 0.0).abs() < 1e-10,
            "STP should be 0 with weak shear (<12.5 m/s)"
        );
    }

    #[test]
    fn test_stp_negative_inputs_floored() {
        let stp = significant_tornado_parameter(-500.0, 3000.0, -100.0, -5.0);
        assert!(
            (stp - 0.0).abs() < 1e-10,
            "STP should be 0 with all negative inputs"
        );
    }

    #[test]
    fn test_stp_strong_case() {
        // High CAPE=4000, low LCL=500, strong SRH=400, strong shear=35
        let stp = significant_tornado_parameter(4000.0, 500.0, 400.0, 35.0);
        let expected = (4000.0 / 1500.0) * 1.0 * (400.0 / 150.0) * 1.5;
        assert!(
            (stp - expected).abs() < 1e-10,
            "STP = {stp}, expected {expected}"
        );
    }

    // ── supercell_composite_parameter ──

    #[test]
    fn test_scp_nominal() {
        let scp = supercell_composite_parameter(1000.0, 50.0, 20.0);
        assert!((scp - 1.0).abs() < 1e-10, "SCP with nominal values = {scp}");
    }

    #[test]
    fn test_scp_zero_cape() {
        let scp = supercell_composite_parameter(0.0, 200.0, 30.0);
        assert!((scp - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_fused_wind_diagnostics_bundle_matches_individual_grid_calls() {
        let u_3d = [0.0, 10.0, 20.0];
        let v_3d = [0.0, 0.0, 0.0];
        let height_agl_3d = [0.0, 3000.0, 6000.0];

        let fused = grid::compute_wind_diagnostics_bundle(&u_3d, &v_3d, &height_agl_3d, 1, 1, 3);
        assert_eq!(
            fused.srh_01km_m2s2,
            grid::compute_srh(&u_3d, &v_3d, &height_agl_3d, 1, 1, 3, 1000.0)
        );
        assert_eq!(
            fused.srh_03km_m2s2,
            grid::compute_srh(&u_3d, &v_3d, &height_agl_3d, 1, 1, 3, 3000.0)
        );
        assert_eq!(
            fused.shear_06km_ms,
            grid::compute_shear(&u_3d, &v_3d, &height_agl_3d, 1, 1, 3, 0.0, 6000.0)
        );
    }

    #[test]
    fn test_scp_strong_case() {
        // Shear 30 m/s is capped at 20 m/s => shear_term = 1.0
        let scp = supercell_composite_parameter(3000.0, 300.0, 30.0);
        let expected = (3000.0 / 1000.0) * (300.0 / 50.0) * 1.0;
        assert!(
            (scp - expected).abs() < 1e-10,
            "SCP = {scp}, expected {expected}"
        );
    }

    #[test]
    fn test_scp_weak_shear_zero() {
        // Shear below 10 m/s => shear_term = 0 => SCP = 0 (per MetPy)
        let scp = supercell_composite_parameter(3000.0, 200.0, 8.0);
        assert!((scp - 0.0).abs() < 1e-10, "SCP should be 0 with weak shear");
    }

    #[test]
    fn test_scp_negative_inputs_floored() {
        let scp = supercell_composite_parameter(-500.0, -100.0, -10.0);
        assert!((scp - 0.0).abs() < 1e-10);
    }

    // ── critical_angle ──

    #[test]
    fn test_critical_angle_perpendicular() {
        // Inflow from east (storm east of surface obs), shear points north
        // => 90 degrees
        let angle = critical_angle(
            10.0, 0.0, // storm at (10, 0)
            0.0, 0.0, // sfc wind calm
            0.0, 5.0, // 500m wind northward
        );
        assert!(
            (angle - 90.0).abs() < 1e-10,
            "expected 90 degrees, got {angle}"
        );
    }

    #[test]
    fn test_critical_angle_parallel() {
        // Inflow and shear both pointing north => 0 degrees
        let angle = critical_angle(
            0.0, -10.0, // storm south of sfc
            0.0, 0.0, // sfc calm
            0.0, 5.0, // 500m northward
        );
        assert!(angle.abs() < 1e-10, "expected 0 degrees, got {angle}");
    }

    #[test]
    fn test_critical_angle_antiparallel() {
        // Inflow and shear 180 degrees apart
        let angle = critical_angle(
            0.0, 10.0, // storm north of sfc
            0.0, 0.0, // sfc calm
            0.0, 5.0, // 500m northward
        );
        assert!(
            (angle - 180.0).abs() < 1e-10,
            "expected 180 degrees, got {angle}"
        );
    }

    #[test]
    fn test_critical_angle_zero_inflow() {
        // Storm motion equals surface wind => zero inflow vector
        let angle = critical_angle(5.0, 3.0, 5.0, 3.0, 10.0, 8.0);
        assert!((angle - 0.0).abs() < 1e-10, "zero inflow should give 0 deg");
    }

    #[test]
    fn test_critical_angle_zero_shear() {
        // 500m wind equals surface wind => zero shear vector
        let angle = critical_angle(10.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert!((angle - 0.0).abs() < 1e-10, "zero shear should give 0 deg");
    }

    #[test]
    fn test_critical_angle_45_degrees() {
        // Inflow along x-axis, shear at 45 degrees
        let angle = critical_angle(
            10.0, 0.0, // storm motion
            0.0, 0.0, // sfc
            -5.0, 5.0, // 500m: shear = (-5, 5), inflow = (0,0)-(10,0) = (-10, 0)
        );
        // inflow = (-10, 0), shear = (-5, 5)
        // cos(theta) = ((-10)*(-5) + 0*5) / (10 * sqrt(50)) = 50 / (10*sqrt(50))
        //            = 5 / sqrt(50) = sqrt(50)/10 = 1/sqrt(2)
        // theta = 45 degrees
        assert!(
            (angle - 45.0).abs() < 1e-10,
            "expected 45 degrees, got {angle}"
        );
    }

    fn dewpoint_k_from_q(q_kgkg: f64, p_pa: f64, temp_k: f64) -> f64 {
        let q = q_kgkg.max(1.0e-10);
        let p_hpa = p_pa / 100.0;
        let e = (q * p_hpa / (0.622 + q)).max(1.0e-10);
        let ln_e = (e / 6.112).ln();
        let td_c = (243.5 * ln_e) / (17.67 - ln_e);
        (td_c + 273.15).min(temp_k)
    }

    fn q_from_dewpoint_k(td_k: f64, p_pa: f64) -> f64 {
        let td_c = td_k - 273.15;
        let e_hpa = 6.112 * ((17.67 * td_c) / (td_c + 243.5)).exp();
        let p_hpa = p_pa / 100.0;
        0.622 * e_hpa / (p_hpa - e_hpa)
    }

    fn assert_close(actual: f64, expected: f64) {
        let tolerance = 1e-6_f64.max(expected.abs() * 1e-10);
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual={actual}, expected={expected}, tolerance={tolerance}"
        );
    }

    #[test]
    fn test_compute_ecape_single_column_matches_direct_solver() {
        let nx = 1;
        let ny = 1;
        let nz = 6;

        let pressure_3d = vec![95000.0, 90000.0, 85000.0, 70000.0, 50000.0, 30000.0];
        let temperature_c_3d = vec![26.0, 22.0, 18.0, 8.0, -10.0, -38.0];
        let qvapor_3d = vec![0.016, 0.013, 0.010, 0.005, 0.0015, 0.0003];
        let height_agl_3d = vec![150.0, 800.0, 1500.0, 3000.0, 5600.0, 9200.0];
        let u_3d = vec![6.0, 9.0, 12.0, 18.0, 26.0, 33.0];
        let v_3d = vec![2.0, 5.0, 8.0, 13.0, 20.0, 28.0];
        let psfc = vec![100000.0];
        let t2 = vec![303.15];
        let q2 = vec![0.018];
        let u10 = vec![5.0];
        let v10 = vec![1.5];

        let (ecape, ncape, cape, cin, lfc, el) = grid::compute_ecape(
            &pressure_3d,
            &temperature_c_3d,
            &qvapor_3d,
            &height_agl_3d,
            &u_3d,
            &v_3d,
            &psfc,
            &t2,
            &q2,
            &u10,
            &v10,
            nx,
            ny,
            nz,
            "ml",
            "bunkers_rm",
            None,
            Some(true),
            None,
            None,
        )
        .unwrap();

        let pressure_pa = vec![
            100000.0, 95000.0, 90000.0, 85000.0, 70000.0, 50000.0, 30000.0,
        ];
        let height_m = vec![0.0, 150.0, 800.0, 1500.0, 3000.0, 5600.0, 9200.0];
        let temp_k = vec![303.15, 299.15, 295.15, 291.15, 281.15, 263.15, 235.15];
        let dewpoint_k = vec![
            dewpoint_k_from_q(0.018, 100000.0, 303.15),
            dewpoint_k_from_q(0.016, 95000.0, 299.15),
            dewpoint_k_from_q(0.013, 90000.0, 295.15),
            dewpoint_k_from_q(0.010, 85000.0, 291.15),
            dewpoint_k_from_q(0.005, 70000.0, 281.15),
            dewpoint_k_from_q(0.0015, 50000.0, 263.15),
            dewpoint_k_from_q(0.0003, 30000.0, 235.15),
        ];
        let u_ms = vec![5.0, 6.0, 9.0, 12.0, 18.0, 26.0, 33.0];
        let v_ms = vec![1.5, 2.0, 5.0, 8.0, 13.0, 20.0, 28.0];
        let options = ParcelOptions {
            cape_type: CapeType::MixedLayer,
            storm_motion_type: StormMotionType::RightMoving,
            pseudoadiabatic: Some(true),
            ..ParcelOptions::default()
        };
        let direct = calc_ecape_parcel(
            &height_m,
            &pressure_pa,
            &temp_k,
            &dewpoint_k,
            &u_ms,
            &v_ms,
            &options,
        )
        .unwrap();

        assert!((ecape[0] - direct.ecape_jkg).abs() < 1.0);
        assert!((ncape[0] - direct.ncape_jkg).abs() < 1.0);
        assert!((cape[0] - direct.cape_jkg).abs() < 1.0);
        assert!((cin[0] - direct.cin_jkg).abs() < 1.0);
        assert!((lfc[0] - direct.lfc_m.unwrap_or(0.0)).abs() < 1.0e-6);
        assert!((el[0] - direct.el_m.unwrap_or(0.0)).abs() < 1.0e-6);
    }

    #[test]
    fn test_compute_ecape_matches_shared_parity_fixture() {
        let height_m = [
            0.0, 250.0, 500.0, 750.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0, 4000.0, 5000.0,
            6000.0, 7500.0, 9000.0, 10500.0, 12000.0, 14000.0, 16000.0,
        ];
        let pressure_pa = [
            100000.0,
            96923.32344763441,
            93941.30628134758,
            91051.03613800342,
            88249.69025845954,
            82902.91181804004,
            77880.07830714049,
            73161.56289466418,
            68728.92787909723,
            60653.06597126334,
            53526.142851899036,
            47236.65527410147,
            39160.5626676799,
            32465.24673583497,
            26914.634872918385,
            22313.016014842982,
            17377.394345044515,
            13533.52832366127,
        ];
        let temperature_k = [
            302.0, 300.2, 298.4, 296.6, 294.8, 291.2, 287.6, 284.0, 280.4, 273.2, 266.0, 258.8,
            248.0, 237.2, 226.4, 215.6, 215.6, 215.6,
        ];
        let dewpoint_k = [
            296.0, 295.625, 295.25, 294.875, 294.3, 290.7, 287.1, 283.5, 279.9, 272.7, 265.5,
            258.3, 247.5, 236.7, 225.9, 215.1, 215.1, 215.1,
        ];
        let u_wind_ms = [
            4.0, 4.625, 5.25, 5.875, 6.5, 7.75, 9.0, 10.25, 11.5, 14.0, 16.5, 19.0, 22.75, 26.5,
            30.25, 34.0, 39.0, 44.0,
        ];
        let v_wind_ms = [
            1.0, 1.375, 1.75, 2.125, 2.5, 3.25, 4.0, 4.75, 5.5, 7.0, 8.5, 10.0, 12.25, 14.5, 16.75,
            19.0, 22.0, 25.0,
        ];
        let qvapor: Vec<f64> = pressure_pa
            .iter()
            .zip(dewpoint_k)
            .map(|(&p, td)| q_from_dewpoint_k(td, p))
            .collect();
        let temperature_c_3d: Vec<f64> = temperature_k[1..].iter().map(|t| t - 273.15).collect();

        let cases = [
            (
                "surface_based",
                (
                    2011.5445493759416,
                    0.0,
                    2846.0409852115004,
                    -44.991140025487326,
                    1360.0,
                    12220.0,
                ),
            ),
            (
                "mixed_layer",
                (
                    2115.38982529213,
                    0.0,
                    3040.829940651471,
                    -8.677832569217891,
                    1180.0,
                    12240.0,
                ),
            ),
            (
                "most_unstable",
                (
                    2097.6810414544825,
                    0.0,
                    3010.1256185714574,
                    -0.23088078138348503,
                    1100.0,
                    12200.0,
                ),
            ),
        ];

        for (parcel_type, expected) in cases {
            let (ecape, ncape, cape, cin, lfc, el) = grid::compute_ecape(
                &pressure_pa[1..],
                &temperature_c_3d,
                &qvapor[1..],
                &height_m[1..],
                &u_wind_ms[1..],
                &v_wind_ms[1..],
                &[pressure_pa[0]],
                &[temperature_k[0]],
                &[qvapor[0]],
                &[u_wind_ms[0]],
                &[v_wind_ms[0]],
                1,
                1,
                pressure_pa.len() - 1,
                parcel_type,
                "bunkers_rm",
                None,
                Some(true),
                Some(12.0),
                Some(6.0),
            )
            .unwrap();

            assert_close(ecape[0], expected.0);
            assert_close(ncape[0], expected.1);
            assert_close(cape[0], expected.2);
            assert_close(cin[0], expected.3);
            assert_close(lfc[0], expected.4);
            assert_close(el[0], expected.5);
        }
    }

    #[test]
    fn test_compute_ecape_triplet_matches_individual_parity_fixture() {
        let height_m = [
            0.0, 250.0, 500.0, 750.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0, 4000.0, 5000.0,
            6000.0, 7500.0, 9000.0, 10500.0, 12000.0, 14000.0, 16000.0,
        ];
        let pressure_pa = [
            100000.0,
            96923.32344763441,
            93941.30628134758,
            91051.03613800342,
            88249.69025845954,
            82902.91181804004,
            77880.07830714049,
            73161.56289466418,
            68728.92787909723,
            60653.06597126334,
            53526.142851899036,
            47236.65527410147,
            39160.5626676799,
            32465.24673583497,
            26914.634872918385,
            22313.016014842982,
            17377.394345044515,
            13533.52832366127,
        ];
        let temperature_k = [
            302.0, 300.2, 298.4, 296.6, 294.8, 291.2, 287.6, 284.0, 280.4, 273.2, 266.0, 258.8,
            248.0, 237.2, 226.4, 215.6, 215.6, 215.6,
        ];
        let dewpoint_k = [
            296.0, 295.625, 295.25, 294.875, 294.3, 290.7, 287.1, 283.5, 279.9, 272.7, 265.5,
            258.3, 247.5, 236.7, 225.9, 215.1, 215.1, 215.1,
        ];
        let u_wind_ms = [
            4.0, 4.625, 5.25, 5.875, 6.5, 7.75, 9.0, 10.25, 11.5, 14.0, 16.5, 19.0, 22.75, 26.5,
            30.25, 34.0, 39.0, 44.0,
        ];
        let v_wind_ms = [
            1.0, 1.375, 1.75, 2.125, 2.5, 3.25, 4.0, 4.75, 5.5, 7.0, 8.5, 10.0, 12.25, 14.5, 16.75,
            19.0, 22.0, 25.0,
        ];
        let qvapor: Vec<f64> = pressure_pa
            .iter()
            .zip(dewpoint_k)
            .map(|(&p, td)| q_from_dewpoint_k(td, p))
            .collect();
        let temperature_c_3d: Vec<f64> = temperature_k[1..].iter().map(|t| t - 273.15).collect();

        let triplet = grid::compute_ecape_triplet_with_failure_mask(
            &pressure_pa[1..],
            &temperature_c_3d,
            &qvapor[1..],
            &height_m[1..],
            &u_wind_ms[1..],
            &v_wind_ms[1..],
            &[pressure_pa[0]],
            &[temperature_k[0]],
            &[qvapor[0]],
            &[u_wind_ms[0]],
            &[v_wind_ms[0]],
            1,
            1,
            pressure_pa.len() - 1,
            "bunkers_rm",
            None,
            Some(true),
            Some(12.0),
            Some(6.0),
        )
        .unwrap();

        let cases = [
            (
                &triplet.sb,
                (
                    2011.5445493759416,
                    0.0,
                    2846.0409852115004,
                    -44.991140025487326,
                    1360.0,
                    12220.0,
                ),
            ),
            (
                &triplet.ml,
                (
                    2115.38982529213,
                    0.0,
                    3040.829940651471,
                    -8.677832569217891,
                    1180.0,
                    12240.0,
                ),
            ),
            (
                &triplet.mu,
                (
                    2097.6810414544825,
                    0.0,
                    3010.1256185714574,
                    -0.23088078138348503,
                    1100.0,
                    12200.0,
                ),
            ),
        ];

        for (result, expected) in cases {
            assert_eq!(result.failure_mask, vec![0]);
            assert_close(result.fields.ecape[0], expected.0);
            assert_close(result.fields.ncape[0], expected.1);
            assert_close(result.fields.cape[0], expected.2);
            assert_close(result.fields.cin[0], expected.3);
            assert_close(result.fields.lfc[0], expected.4);
            assert_close(result.fields.el[0], expected.5);
        }
    }

    #[test]
    fn test_compute_ecape_triplet_levels_matches_broadcast_pressure_path() {
        let height_m = [
            0.0, 250.0, 500.0, 750.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0, 4000.0, 5000.0,
            6000.0, 7500.0, 9000.0, 10500.0, 12000.0, 14000.0, 16000.0,
        ];
        let pressure_pa = [
            100000.0,
            96923.32344763441,
            93941.30628134758,
            91051.03613800342,
            88249.69025845954,
            82902.91181804004,
            77880.07830714049,
            73161.56289466418,
            68728.92787909723,
            60653.06597126334,
            53526.142851899036,
            47236.65527410147,
            39160.5626676799,
            32465.24673583497,
            26914.634872918385,
            22313.016014842982,
            17377.394345044515,
            13533.52832366127,
        ];
        let temperature_k = [
            302.0, 300.2, 298.4, 296.6, 294.8, 291.2, 287.6, 284.0, 280.4, 273.2, 266.0, 258.8,
            248.0, 237.2, 226.4, 215.6, 215.6, 215.6,
        ];
        let dewpoint_k = [
            296.0, 295.625, 295.25, 294.875, 294.3, 290.7, 287.1, 283.5, 279.9, 272.7, 265.5,
            258.3, 247.5, 236.7, 225.9, 215.1, 215.1, 215.1,
        ];
        let u_wind_ms = [
            4.0, 4.625, 5.25, 5.875, 6.5, 7.75, 9.0, 10.25, 11.5, 14.0, 16.5, 19.0, 22.75, 26.5,
            30.25, 34.0, 39.0, 44.0,
        ];
        let v_wind_ms = [
            1.0, 1.375, 1.75, 2.125, 2.5, 3.25, 4.0, 4.75, 5.5, 7.0, 8.5, 10.0, 12.25, 14.5, 16.75,
            19.0, 22.0, 25.0,
        ];
        let qvapor: Vec<f64> = pressure_pa
            .iter()
            .zip(dewpoint_k)
            .map(|(&p, td)| q_from_dewpoint_k(td, p))
            .collect();
        let temperature_c_3d: Vec<f64> = temperature_k[1..].iter().map(|t| t - 273.15).collect();

        let broadcast = grid::compute_ecape_triplet_with_failure_mask(
            &pressure_pa[1..],
            &temperature_c_3d,
            &qvapor[1..],
            &height_m[1..],
            &u_wind_ms[1..],
            &v_wind_ms[1..],
            &[pressure_pa[0]],
            &[temperature_k[0]],
            &[qvapor[0]],
            &[u_wind_ms[0]],
            &[v_wind_ms[0]],
            1,
            1,
            pressure_pa.len() - 1,
            "bunkers_rm",
            None,
            Some(true),
            Some(12.0),
            Some(6.0),
        )
        .unwrap();

        let levels = grid::compute_ecape_triplet_with_failure_mask_levels(
            &pressure_pa[1..],
            &temperature_c_3d,
            &qvapor[1..],
            &height_m[1..],
            &u_wind_ms[1..],
            &v_wind_ms[1..],
            &[pressure_pa[0]],
            &[temperature_k[0]],
            &[qvapor[0]],
            &[u_wind_ms[0]],
            &[v_wind_ms[0]],
            1,
            1,
            pressure_pa.len() - 1,
            "bunkers_rm",
            None,
            Some(true),
            Some(12.0),
            Some(6.0),
        )
        .unwrap();

        assert_eq!(levels, broadcast);
    }

    #[test]
    fn test_compute_ecape_triplet_without_failure_mask_matches_masked_fields() {
        let height_m = [
            0.0, 250.0, 500.0, 750.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0, 4000.0, 5000.0,
            6000.0, 7500.0, 9000.0, 10500.0, 12000.0, 14000.0, 16000.0,
        ];
        let pressure_pa = [
            100000.0,
            96923.32344763441,
            93941.30628134758,
            91051.03613800342,
            88249.69025845954,
            82902.91181804004,
            77880.07830714049,
            73161.56289466418,
            68728.92787909723,
            60653.06597126334,
            53526.142851899036,
            47236.65527410147,
            39160.5626676799,
            32465.24673583497,
            26914.634872918385,
            22313.016014842982,
            17377.394345044515,
            13533.52832366127,
        ];
        let temperature_k = [
            302.0, 300.2, 298.4, 296.6, 294.8, 291.2, 287.6, 284.0, 280.4, 273.2, 266.0, 258.8,
            248.0, 237.2, 226.4, 215.6, 215.6, 215.6,
        ];
        let dewpoint_k = [
            296.0, 295.625, 295.25, 294.875, 294.3, 290.7, 287.1, 283.5, 279.9, 272.7, 265.5,
            258.3, 247.5, 236.7, 225.9, 215.1, 215.1, 215.1,
        ];
        let u_wind_ms = [
            4.0, 4.625, 5.25, 5.875, 6.5, 7.75, 9.0, 10.25, 11.5, 14.0, 16.5, 19.0, 22.75, 26.5,
            30.25, 34.0, 39.0, 44.0,
        ];
        let v_wind_ms = [
            1.0, 1.375, 1.75, 2.125, 2.5, 3.25, 4.0, 4.75, 5.5, 7.0, 8.5, 10.0, 12.25, 14.5, 16.75,
            19.0, 22.0, 25.0,
        ];
        let qvapor: Vec<f64> = pressure_pa
            .iter()
            .zip(dewpoint_k)
            .map(|(&p, td)| q_from_dewpoint_k(td, p))
            .collect();
        let temperature_c_3d: Vec<f64> = temperature_k[1..].iter().map(|t| t - 273.15).collect();

        let masked = grid::compute_ecape_triplet_with_failure_mask(
            &pressure_pa[1..],
            &temperature_c_3d,
            &qvapor[1..],
            &height_m[1..],
            &u_wind_ms[1..],
            &v_wind_ms[1..],
            &[pressure_pa[0]],
            &[temperature_k[0]],
            &[qvapor[0]],
            &[u_wind_ms[0]],
            &[v_wind_ms[0]],
            1,
            1,
            pressure_pa.len() - 1,
            "bunkers_rm",
            None,
            Some(true),
            Some(12.0),
            Some(6.0),
        )
        .unwrap();

        let unmasked = grid::compute_ecape_triplet(
            &pressure_pa[1..],
            &temperature_c_3d,
            &qvapor[1..],
            &height_m[1..],
            &u_wind_ms[1..],
            &v_wind_ms[1..],
            &[pressure_pa[0]],
            &[temperature_k[0]],
            &[qvapor[0]],
            &[u_wind_ms[0]],
            &[v_wind_ms[0]],
            1,
            1,
            pressure_pa.len() - 1,
            "bunkers_rm",
            None,
            Some(true),
            Some(12.0),
            Some(6.0),
        )
        .unwrap();

        assert_eq!(masked.sb.failure_mask, vec![0]);
        assert_eq!(masked.ml.failure_mask, vec![0]);
        assert_eq!(masked.mu.failure_mask, vec![0]);

        assert_eq!(unmasked.sb, masked.sb.fields);
        assert_eq!(unmasked.ml, masked.ml.fields);
        assert_eq!(unmasked.mu, masked.mu.fields);
    }

    #[test]
    fn test_compute_ecape_requires_both_storm_motion_components() {
        let err = grid::compute_ecape(
            &[95000.0, 90000.0],
            &[20.0, 10.0],
            &[0.010, 0.004],
            &[500.0, 2000.0],
            &[5.0, 10.0],
            &[0.0, 5.0],
            &[100000.0],
            &[300.0],
            &[0.014],
            &[4.0],
            &[1.0],
            1,
            1,
            2,
            "sb",
            "mean_wind",
            None,
            Some(true),
            Some(8.0),
            None,
        )
        .unwrap_err();
        assert!(err.contains("storm_u and storm_v"));
    }

    #[test]
    fn test_compute_ecape_with_failure_mask_flags_zero_fill_columns() {
        let (ecape, ncape, cape, cin, lfc, el, failures) = grid::compute_ecape_with_failure_mask(
            &[f64::NAN, f64::NAN],
            &[f64::NAN, f64::NAN],
            &[f64::NAN, f64::NAN],
            &[f64::NAN, f64::NAN],
            &[f64::NAN, f64::NAN],
            &[f64::NAN, f64::NAN],
            &[100000.0],
            &[300.0],
            &[0.014],
            &[4.0],
            &[1.0],
            1,
            1,
            2,
            "sb",
            "mean_wind",
            None,
            Some(true),
            None,
            None,
        )
        .unwrap();

        assert_eq!(failures, vec![1]);
        assert_eq!(ecape, vec![0.0]);
        assert_eq!(ncape, vec![0.0]);
        assert_eq!(cape, vec![0.0]);
        assert_eq!(cin, vec![0.0]);
        assert_eq!(lfc, vec![0.0]);
        assert_eq!(el, vec![0.0]);
    }
}
