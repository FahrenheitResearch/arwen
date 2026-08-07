//! Cloud water path (CWP) derived from the ABI L2 retrieval trio: cloud
//! optical depth (COD), cloud particle size (CPS, effective radius in µm)
//! and cloud-top phase (ACTP). This is the WoFS-style satellite cloud
//! observable: CWP columns (plus clear-sky zeros) assimilated against a
//! model's integrated liquid/ice water paths — the forward-operator
//! lineage of Jones et al. (2013) and the GOES CWP assimilation line in
//! the Warn-on-Forecast System.
//!
//! ## The relation and what is (and is not) sourced
//!
//! For a vertically homogeneous cloud of spheres in the geometric-optics
//! limit (extinction efficiency `Q_ext ≈ 2`), optical depth relates to
//! condensed water path as `τ = (3/2)·CWP/(ρ·r_e)`, i.e.
//!
//! ```text
//! CWP [g m⁻²] = (2/3) · τ · r_e [µm] · ρ [g cm⁻³]
//! ```
//!
//! * **Liquid** (`ρ_w = 1.0 g cm⁻³`): the standard liquid-water-path
//!   relation (Stephens 1978; the same `Q_ext = 2` geometric-optics form
//!   used by the Minnis-lineage GOES cloud retrievals that feed WoFS CWP
//!   assimilation). Applied to the `liquid_water` and
//!   `supercooled_liquid_water` phases.
//! * **Ice** (`ρ_i = 0.917 g cm⁻³`): **PROVISIONAL.** The same algebraic
//!   form with the bulk density of ice. Published IWP–τ–r_e relations for
//!   ice are empirical and habit/temperature dependent (e.g. Heymsfield et
//!   al. 2003) and do not reduce to one verified coefficient; this
//!   implementation deliberately uses the transparent spherical-particle
//!   form and flags it, rather than citing a coefficient that could not be
//!   verified. Applied to the `ice` phase.
//! * **Mixed phase**: **PROVISIONAL.** Treated with the ice branch (a
//!   mixed-phase cloud *top* at these wavelengths is ice-dominated).
//! * **Clear sky**: CWP = 0 by definition — a genuine zero observation
//!   (the clear-sky-zero half of WoFS-style CWP assimilation), emitted
//!   even where COD/CPS are fill, because the phase product observed
//!   clear sky.
//! * **Unknown phase, missing inputs, negative inputs**: NaN, counted.
//!   Fail-closed; nothing is fabricated.

use std::error::Error;
use std::io;

/// Density of liquid water, g cm⁻³.
pub const LIQUID_WATER_DENSITY_G_CM3: f32 = 1.0;

/// Bulk density of ice, g cm⁻³ (0.917). Its use inside the CWP relation
/// is PROVISIONAL — see the module docs.
pub const BULK_ICE_DENSITY_G_CM3: f32 = 0.917;

/// ABI ACTP cloud-top phase codes, as encoded in the `Phase` variable
/// (`flag_values` 0..=5, verified against real GOES-19 granules).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CloudPhase {
    ClearSky,
    LiquidWater,
    SupercooledLiquidWater,
    MixedPhase,
    Ice,
    Unknown,
}

impl CloudPhase {
    /// Map a decoded `Phase` pixel to its phase. Returns `None` for NaN
    /// (fill / gated) and for any value outside the documented 0..=5.
    pub fn from_decoded(value: f32) -> Option<Self> {
        if !value.is_finite() || !(0.0..=5.0).contains(&value) || value.fract() != 0.0 {
            return None;
        }
        Some(match value as u8 {
            0 => Self::ClearSky,
            1 => Self::LiquidWater,
            2 => Self::SupercooledLiquidWater,
            3 => Self::MixedPhase,
            4 => Self::Ice,
            _ => Self::Unknown,
        })
    }
}

/// CWP in g m⁻² for one pixel: `(2/3)·τ·r_e·ρ(phase)`, with the phase
/// dispatch and fail-closed cases described in the module docs.
///
/// `cod` is unitless optical depth; `particle_radius_um` is the effective
/// radius in µm (the native CPS unit).
pub fn cloud_water_path_g_m2(cod: f32, particle_radius_um: f32, phase: CloudPhase) -> f32 {
    let density = match phase {
        CloudPhase::ClearSky => return 0.0,
        CloudPhase::LiquidWater | CloudPhase::SupercooledLiquidWater => {
            LIQUID_WATER_DENSITY_G_CM3
        }
        // PROVISIONAL: mixed-phase tops take the ice branch.
        CloudPhase::MixedPhase | CloudPhase::Ice => BULK_ICE_DENSITY_G_CM3,
        CloudPhase::Unknown => return f32::NAN,
    };
    if !(cod.is_finite() && particle_radius_um.is_finite()) || cod < 0.0 || particle_radius_um < 0.0
    {
        return f32::NAN;
    }
    (2.0f32 / 3.0) * density * cod * particle_radius_um
}

/// Per-pixel accounting of one [`cloud_water_path_plane`] pass. Every
/// pixel lands in exactly one bucket.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CwpCounts {
    /// Clear-sky pixels emitted as CWP = 0.
    pub clear_zero: usize,
    pub liquid: usize,
    pub supercooled: usize,
    pub mixed: usize,
    pub ice: usize,
    /// Phase = unknown: NaN.
    pub unknown: usize,
    /// Phase pixel NaN (fill or DQF-gated) or outside 0..=5: NaN.
    pub phase_missing: usize,
    /// Cloudy phase but COD or CPS missing/negative: NaN.
    pub input_missing: usize,
}

impl CwpCounts {
    /// Pixels that produced a finite CWP value.
    pub fn finite(&self) -> usize {
        self.clear_zero + self.liquid + self.supercooled + self.mixed + self.ice
    }
}

/// Derive a CWP plane from equally shaped, already DQF-gated COD, CPS and
/// Phase planes (same fixed grid — the caller asserts that; the ABI L2
/// CONUS cloud suite shares one 2 km grid, verified on real granules).
pub fn cloud_water_path_plane(
    cod: &[f32],
    particle_radius_um: &[f32],
    phase: &[f32],
) -> Result<(Vec<f32>, CwpCounts), Box<dyn Error>> {
    if cod.len() != phase.len() || particle_radius_um.len() != phase.len() {
        return Err(boxed_error(format!(
            "CWP input plane lengths differ: cod={} cps={} phase={}",
            cod.len(),
            particle_radius_um.len(),
            phase.len()
        )));
    }
    let mut counts = CwpCounts::default();
    let mut out = Vec::with_capacity(phase.len());
    for ((&tau, &radius), &phase_value) in cod.iter().zip(particle_radius_um).zip(phase) {
        let Some(phase) = CloudPhase::from_decoded(phase_value) else {
            counts.phase_missing += 1;
            out.push(f32::NAN);
            continue;
        };
        let value = cloud_water_path_g_m2(tau, radius, phase);
        match phase {
            CloudPhase::ClearSky => counts.clear_zero += 1,
            CloudPhase::Unknown => counts.unknown += 1,
            _ if value.is_nan() => counts.input_missing += 1,
            CloudPhase::LiquidWater => counts.liquid += 1,
            CloudPhase::SupercooledLiquidWater => counts.supercooled += 1,
            CloudPhase::MixedPhase => counts.mixed += 1,
            CloudPhase::Ice => counts.ice += 1,
        }
        out.push(value);
    }
    Ok((out, counts))
}

fn boxed_error(message: impl Into<String>) -> Box<dyn Error> {
    Box::new(io::Error::new(io::ErrorKind::InvalidData, message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f32, expected: f32) {
        let scale = expected.abs().max(1e-6);
        assert!(
            (actual - expected).abs() / scale < 1e-5,
            "{actual} != {expected}"
        );
    }

    #[test]
    fn liquid_cwp_is_two_thirds_tau_re() {
        // 2/3 · 10 · 12 · 1.0 = 80 g/m².
        assert_close(
            cloud_water_path_g_m2(10.0, 12.0, CloudPhase::LiquidWater),
            80.0,
        );
        // Supercooled liquid is liquid.
        assert_close(
            cloud_water_path_g_m2(10.0, 12.0, CloudPhase::SupercooledLiquidWater),
            80.0,
        );
    }

    #[test]
    fn ice_cwp_uses_bulk_ice_density() {
        // 2/3 · 10 · 12 · 0.917 = 73.36 g/m² (PROVISIONAL coefficient).
        assert_close(cloud_water_path_g_m2(10.0, 12.0, CloudPhase::Ice), 73.36);
        assert_close(
            cloud_water_path_g_m2(10.0, 12.0, CloudPhase::MixedPhase),
            73.36,
        );
    }

    #[test]
    fn clear_sky_is_a_zero_observation_even_without_cod() {
        assert_eq!(
            cloud_water_path_g_m2(f32::NAN, f32::NAN, CloudPhase::ClearSky),
            0.0
        );
    }

    #[test]
    fn missing_or_negative_inputs_and_unknown_phase_are_nan() {
        assert!(cloud_water_path_g_m2(f32::NAN, 12.0, CloudPhase::Ice).is_nan());
        assert!(cloud_water_path_g_m2(10.0, f32::NAN, CloudPhase::LiquidWater).is_nan());
        assert!(cloud_water_path_g_m2(-1.0, 12.0, CloudPhase::LiquidWater).is_nan());
        assert!(cloud_water_path_g_m2(10.0, -1.0, CloudPhase::Ice).is_nan());
        assert!(cloud_water_path_g_m2(10.0, 12.0, CloudPhase::Unknown).is_nan());
    }

    #[test]
    fn phase_codes_match_the_actp_encoding() {
        assert_eq!(CloudPhase::from_decoded(0.0), Some(CloudPhase::ClearSky));
        assert_eq!(CloudPhase::from_decoded(1.0), Some(CloudPhase::LiquidWater));
        assert_eq!(
            CloudPhase::from_decoded(2.0),
            Some(CloudPhase::SupercooledLiquidWater)
        );
        assert_eq!(CloudPhase::from_decoded(3.0), Some(CloudPhase::MixedPhase));
        assert_eq!(CloudPhase::from_decoded(4.0), Some(CloudPhase::Ice));
        assert_eq!(CloudPhase::from_decoded(5.0), Some(CloudPhase::Unknown));
        assert_eq!(CloudPhase::from_decoded(f32::NAN), None);
        assert_eq!(CloudPhase::from_decoded(255.0), None);
        assert_eq!(CloudPhase::from_decoded(1.5), None);
    }

    #[test]
    fn plane_accounts_for_every_pixel() {
        let cod = [10.0, 10.0, f32::NAN, 3.0, 1.0, 2.0, 5.0];
        let cps = [12.0, 12.0, 8.0, 6.0, 4.0, 10.0, 7.0];
        let phase = [1.0, 4.0, 4.0, 0.0, 5.0, f32::NAN, 2.0];
        let (cwp, counts) = cloud_water_path_plane(&cod, &cps, &phase).unwrap();
        assert_eq!(
            counts,
            CwpCounts {
                clear_zero: 1,
                liquid: 1,
                supercooled: 1,
                mixed: 0,
                ice: 1,
                unknown: 1,
                phase_missing: 1,
                input_missing: 1,
            }
        );
        assert_eq!(counts.finite(), 4);
        assert_close(cwp[0], 80.0);
        assert_close(cwp[1], 73.36);
        assert!(cwp[2].is_nan());
        assert_eq!(cwp[3], 0.0);
        assert!(cwp[4].is_nan());
        assert!(cwp[5].is_nan());
        assert_close(cwp[6], 23.333334);

        assert!(cloud_water_path_plane(&cod[..2], &cps, &phase).is_err());
    }
}
