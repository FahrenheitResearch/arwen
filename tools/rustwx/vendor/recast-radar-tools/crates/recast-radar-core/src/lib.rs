//! Core data model for recast-radar-tools.
//!
//! # The data model
//!
//! The WMO FM301 / CfRadial 2 data model (`docs/design/fm301-model.md`), with
//! compact raw storage and lazy scaling:
//!
//! - [`Volume`] is the root group of an FM301 file: global attributes, the
//!   station location, radar parameters and calibration, and the sweeps.
//! - [`Sweep`] is one `sweep_<n>` group: ray coordinates (`time`, `azimuth`,
//!   `elevation`), one `range` coordinate, per-ray instrument variables and the
//!   dataset variables ([`Field`]).
//! - [`Field`] holds one variable's values row-major `[nrays × ngates]` in the
//!   source's own encoding (`u8`, `u16`, `i8`, `i16`, `i32`, `f32`, `f64`) with
//!   its CF packing. Physical values are computed on demand; decoders never
//!   expand raw storage to floats.
//!
//! The model keeps rays in the source's storage order and each field in its
//! native gate geometry ([`GateMapping`] onto the sweep range). The FM301 view
//! ([`fm301`]) applies ray order, padding and gate repetition when a caller
//! reads a variable.
//!
//! The model's items are documented here, at the crate root. Each is also
//! reachable as `model::<item>` (`recast_radar_core::model::Volume`), the path
//! the member crates use; that module is left out of the documentation so
//! that every item has one page.
//!
//! # The rest of the crate
//!
//! - [`fm301`]: the FM301 group view over a [`Volume`], the conformance
//!   surface for xradar `DataTree` and CfRadial 2 output.
//! - Beam geometry, refractivity and bounded decompression helpers shared by
//!   every crate.
//!
//! # Limits
//!
//! [`bounded_read`] holds the resource limits every decoder crate shares
//! (expanded input size, decoded volume and batch budgets, gates per radial,
//! sweeps per volume) and the [`bounded_read::DecodeBudget`] that enforces
//! them. Each decoder crate documents its format-specific limits in its own
//! `# Limits` section.

#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

pub mod bounded_read;
pub mod fm301;
// Hidden from the documentation only (`cfg(doc)` is set by rustdoc alone):
// its items are documented at the crate root, where the glob below re-exports
// them. A plain `#[doc(hidden)]` would also switch off `missing_docs` for
// every item in the module, so an undocumented model item would build.
#[cfg_attr(doc, doc(hidden))]
pub mod model;
mod refractivity;

// Every public item of the model, so that each has one short path
// (`recast_radar_core::RowRef`, `recast_radar_tools::model::RowRef`) as well
// as its `model::` path. A glob, so that a type added to the model is
// reachable here without a second edit.
#[doc(inline)]
pub use model::*;
pub use refractivity::{
    EARTH_DUCTING_GRADIENT_N_PER_KM, PropagationRegime, RefractedBeamError, RefractedBeamPoint,
    RefractedBeamTrace, RefractivityLevel, RefractivityProfile, RefractivityProfileError,
    STANDARD_REFRACTIVITY_GRADIENT_N_PER_KM, propagation_regime, radio_refractivity_n_units,
    trace_refracted_beam,
};

/// Earth's mean radius (m).
pub const EARTH_RADIUS_M: f64 = 6_371_000.0;
/// Effective Earth radius under the standard "4/3 Earth" refraction model
/// (Bean & Dutton 1968; the standard-atmosphere refractivity gradient).
pub const EFFECTIVE_EARTH_RADIUS_M: f64 = EARTH_RADIUS_M * 4.0 / 3.0;

/// Center height of the radar beam **above the antenna**, in metres, under the
/// 4/3-Earth-radius effective-radius approximation for atmospheric refraction.
///
/// Doviak & Zrnić (1993), *Doppler Radar and Weather Observations* (2nd ed.),
/// eq. 2.28b: `h = sqrt(r² + aₑ² + 2·r·aₑ·sin θ) − aₑ`, with `aₑ = 4/3·a`.
///
/// `slant_range_m` is range along the beam; `elevation_deg` the antenna
/// elevation angle. Add the antenna's MSL altitude to get beam MSL height.
pub fn beam_height_above_radar_m(slant_range_m: f64, elevation_deg: f64) -> f64 {
    let ae = EFFECTIVE_EARTH_RADIUS_M;
    let r = slant_range_m;
    let theta = elevation_deg.to_radians();
    (r * r + ae * ae + 2.0 * r * ae * theta.sin()).sqrt() - ae
}

/// Great-circle (ground) distance from the radar to the gate, in metres, under
/// the same 4/3-Earth model. Doviak & Zrnić (1993) eq. 2.28c.
pub fn beam_ground_range_m(slant_range_m: f64, elevation_deg: f64) -> f64 {
    let ae = EFFECTIVE_EARTH_RADIUS_M;
    let r = slant_range_m;
    let theta = elevation_deg.to_radians();
    let h = beam_height_above_radar_m(r, elevation_deg);
    ae * ((r * theta.cos()) / (ae + h)).asin()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beam_height_matches_four_thirds_earth_reference() {
        // At 0° elevation, h ≈ r²/(2·aₑ): 100 km -> ~588 m.
        let h0 = beam_height_above_radar_m(100_000.0, 0.0);
        assert!((h0 - 588.6).abs() < 3.0, "0° 100km height was {h0}");

        // At 0.5° elevation, add ~r·sin(0.5°) ≈ 873 m -> ~1461 m.
        let h05 = beam_height_above_radar_m(100_000.0, 0.5);
        assert!((h05 - 1461.0).abs() < 5.0, "0.5° 100km height was {h05}");

        // Origin and monotonicity in range.
        assert!(beam_height_above_radar_m(0.0, 0.5).abs() < 1.0);
        assert!(
            beam_height_above_radar_m(200_000.0, 0.5) > beam_height_above_radar_m(100_000.0, 0.5)
        );
    }

    #[test]
    fn ground_range_close_to_slant_range_at_low_tilt() {
        // At low elevation the ground range is only slightly less than slant range.
        let s = beam_ground_range_m(100_000.0, 0.5);
        assert!(s > 99_000.0 && s < 100_000.0, "ground range was {s}");
    }
}
