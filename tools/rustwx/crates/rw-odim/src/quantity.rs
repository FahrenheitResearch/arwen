//! The ODIM quantity vocabulary.
//!
//! ODIM names its moments in `/datasetN/dataM/what/quantity` with a
//! vocabulary of its own -- `DBZH`, `VRADH`, `RHOHV` -- which does *not*
//! overlap NEXRAD's (`REF`, `VEL`, `RHO`). `wx_radar::products::RadarProduct`
//! only knows the NEXRAD spelling and answers `Unknown` for all of these, so
//! this crate carries its own table rather than mistranslating through that
//! one.
//!
//! The table exists mainly to answer two questions the assimilating side
//! actually asks: what unit is this in, and is it a velocity (and therefore
//! folded, and therefore in need of a Nyquist interval before it can be
//! used)?
//!
//! An unrecognised quantity is **not** an error. National writers ship vendor
//! quantities, and refusing a whole volume because one extra moment was not in
//! a hardcoded list would throw away the moments that were. Unknown quantities
//! decode normally and report [`QuantityKind::Other`] with an empty unit.

use serde::{Deserialize, Serialize};

/// What kind of measurement a quantity is, coarsely.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuantityKind {
    /// Reflectivity factor, corrected or total.
    Reflectivity,
    /// Doppler radial velocity. Folded at the Nyquist interval.
    RadialVelocity,
    /// Doppler spectrum width.
    SpectralWidth,
    /// A dual-polarisation moment.
    DualPol,
    /// A quality or housekeeping field.
    Quality,
    /// An accumulation or derived surface product.
    Derived,
    /// Not in this table.
    Other,
}

/// What is known about one ODIM quantity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuantityInfo {
    /// The unit the decoded physical values are in. Empty when unitless or
    /// unknown.
    pub unit: &'static str,
    pub kind: QuantityKind,
}

const TABLE: &[(&str, &str, QuantityKind)] = &[
    // Reflectivity. TH/TV are the total (uncorrected) factors; DBZH/DBZV are
    // the corrected ones. Both are dBZ and both appear in the same file.
    ("DBZH", "dBZ", QuantityKind::Reflectivity),
    ("DBZV", "dBZ", QuantityKind::Reflectivity),
    ("TH", "dBZ", QuantityKind::Reflectivity),
    ("TV", "dBZ", QuantityKind::Reflectivity),
    ("DBZ", "dBZ", QuantityKind::Reflectivity),
    // Doppler velocity. VRADDH/VRADDV are the dealiased siblings some
    // national writers ship alongside the folded field.
    ("VRAD", "m s-1", QuantityKind::RadialVelocity),
    ("VRADH", "m s-1", QuantityKind::RadialVelocity),
    ("VRADV", "m s-1", QuantityKind::RadialVelocity),
    ("VRADDH", "m s-1", QuantityKind::RadialVelocity),
    ("VRADDV", "m s-1", QuantityKind::RadialVelocity),
    // Spectrum width.
    ("WRAD", "m s-1", QuantityKind::SpectralWidth),
    ("WRADH", "m s-1", QuantityKind::SpectralWidth),
    ("WRADV", "m s-1", QuantityKind::SpectralWidth),
    // Dual polarisation.
    ("ZDR", "dB", QuantityKind::DualPol),
    ("RHOHV", "", QuantityKind::DualPol),
    ("LDR", "dB", QuantityKind::DualPol),
    ("PHIDP", "deg", QuantityKind::DualPol),
    ("UPHIDP", "deg", QuantityKind::DualPol),
    ("KDP", "deg km-1", QuantityKind::DualPol),
    // Quality and housekeeping.
    ("SQI", "", QuantityKind::Quality),
    ("SQIH", "", QuantityKind::Quality),
    ("SQIV", "", QuantityKind::Quality),
    ("SNR", "dB", QuantityKind::Quality),
    ("SNRH", "dB", QuantityKind::Quality),
    ("SNRV", "dB", QuantityKind::Quality),
    ("CCOR", "dB", QuantityKind::Quality),
    ("CCORH", "dB", QuantityKind::Quality),
    ("CCORV", "dB", QuantityKind::Quality),
    ("QIND", "", QuantityKind::Quality),
    ("CLASS", "", QuantityKind::Quality),
    ("DBZH_CLEAN", "dBZ", QuantityKind::Quality),
    // Derived surface fields, which appear in composites more than volumes.
    ("RATE", "mm h-1", QuantityKind::Derived),
    ("ACRR", "mm", QuantityKind::Derived),
    ("HGHT", "km", QuantityKind::Derived),
    ("VIL", "kg m-2", QuantityKind::Derived),
];

/// Look up a quantity. Unknown names answer `Other` with an empty unit rather
/// than failing.
pub fn describe(quantity: &str) -> QuantityInfo {
    for (name, unit, kind) in TABLE {
        if *name == quantity {
            return QuantityInfo { unit, kind: *kind };
        }
    }
    QuantityInfo {
        unit: "",
        kind: QuantityKind::Other,
    }
}

/// Whether this quantity is a Doppler radial velocity, and therefore folded.
///
/// This is the predicate the dealiasing handoff keys on: a moment for which
/// this is true is unusable by the assimilating side until it is paired with
/// the sweep's Nyquist interval.
pub fn is_radial_velocity(quantity: &str) -> bool {
    describe(quantity).kind == QuantityKind::RadialVelocity
}

/// Whether this quantity is a reflectivity factor.
pub fn is_reflectivity(quantity: &str) -> bool {
    describe(quantity).kind == QuantityKind::Reflectivity
}

/// The pack's canonical reflectivity token, and the one the superob layer
/// keys on.
pub const CANONICAL_REFLECTIVITY: &str = "REF";

/// The pack's canonical radial-velocity token.
pub const CANONICAL_VELOCITY: &str = "VEL";

/// Which ODIM quantity claims each canonical token, best first.
///
/// The pack has one consumer vocabulary — `REF` and `VEL`, the tokens
/// `gpuwm.obs.superob` filters on — and a producer that wrote its archive's
/// own spellings instead would have every gate silently skipped by a
/// consumer that still counted the sweep as used. So the translation happens
/// here, at the writer, and it is an ordered preference rather than a kind
/// test because `QuantityKind::Reflectivity` covers four ODIM names at once.
///
/// The order is the physics. `DBZH` is the corrected horizontal factor and is
/// what an assimilation wants; `TH` is the same beam uncorrected, which is
/// diagnostic. A German volume carries both, so choosing by kind would make
/// the canonical moment depend on file order.
const CANONICAL_PREFERENCE: [(&str, &[&str]); 2] = [
    (CANONICAL_REFLECTIVITY, &["DBZH", "DBZ", "DBZV", "TH", "TV"]),
    (
        CANONICAL_VELOCITY,
        &["VRADH", "VRAD", "VRADV", "VRADDH", "VRADDV"],
    ),
];

/// The canonical token `quantity` would claim, and its rank among the
/// contenders for it. `None` for a quantity no consumer reads.
///
/// Rank is what keeps the choice from depending on file order: of two
/// quantities claiming one token, the lower rank takes it and the other keeps
/// its ODIM spelling, so it is still in the pack and still readable but is
/// not mistaken for the assimilable one.
pub fn canonical_claim(quantity: &str) -> Option<(&'static str, usize)> {
    for (token, names) in CANONICAL_PREFERENCE {
        if let Some(rank) = names.iter().position(|name| *name == quantity) {
            return Some((token, rank));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_velocity_moments_seen_on_the_feed_are_recognised_as_folded() {
        // Every velocity spelling the MeteoGate site table advertises.
        for name in ["VRAD", "VRADH", "VRADV"] {
            assert!(is_radial_velocity(name), "{name}");
            assert_eq!(describe(name).unit, "m s-1");
        }
    }

    #[test]
    fn spectrum_width_shares_the_unit_but_is_not_a_velocity() {
        // WRADH is m s-1 too, and it does not fold. Treating it as a velocity
        // would send it to the dealiaser.
        assert_eq!(describe("WRADH").unit, "m s-1");
        assert!(!is_radial_velocity("WRADH"));
        assert_eq!(describe("WRADH").kind, QuantityKind::SpectralWidth);
    }

    #[test]
    fn total_and_corrected_reflectivity_are_both_reflectivity() {
        for name in ["DBZH", "TH"] {
            assert!(is_reflectivity(name));
            assert_eq!(describe(name).unit, "dBZ");
        }
    }

    #[test]
    fn an_unknown_vendor_quantity_is_described_not_refused() {
        let info = describe("XYZZY");
        assert_eq!(info.kind, QuantityKind::Other);
        assert_eq!(info.unit, "");
        assert!(!is_radial_velocity("XYZZY"));
    }

    #[test]
    fn the_dutch_nine_moment_sweep_is_fully_described() {
        // The moments a KNMI volume actually carries, all nine of them.
        let quantities = [
            "DBZH", "TH", "VRADH", "WRADH", "ZDR", "PHIDP", "RHOHV", "SQIH", "CCORH",
        ];
        for name in quantities {
            assert_ne!(
                describe(name).kind,
                QuantityKind::Other,
                "{name} fell through the table"
            );
        }
        assert_eq!(
            quantities.iter().filter(|q| is_radial_velocity(q)).count(),
            1
        );
    }

    #[test]
    fn the_two_moments_a_consumer_reads_claim_the_two_canonical_tokens() {
        assert_eq!(
            canonical_claim("DBZH").map(|(token, _)| token),
            Some(CANONICAL_REFLECTIVITY)
        );
        assert_eq!(
            canonical_claim("VRADH").map(|(token, _)| token),
            Some(CANONICAL_VELOCITY)
        );
    }

    #[test]
    fn corrected_reflectivity_outranks_the_uncorrected_beam() {
        // A German volume carries DBZH and TH on one cut. Choosing by
        // QuantityKind would make the canonical moment depend on file order,
        // and TH is the diagnostic one.
        let (dbzh_token, dbzh_rank) = canonical_claim("DBZH").expect("DBZH claims REF");
        let (th_token, th_rank) = canonical_claim("TH").expect("TH claims REF");
        assert_eq!(dbzh_token, th_token);
        assert!(
            dbzh_rank < th_rank,
            "DBZH must outrank TH for the canonical reflectivity slot"
        );
    }

    #[test]
    fn a_moment_no_consumer_reads_claims_nothing() {
        // Dual-pol and quality fields ride along in the pack and are readable,
        // but nothing downstream treats them as observations, so they must not
        // take a canonical slot from a moment that is one.
        for name in ["ZDR", "PHIDP", "RHOHV", "SQIH", "CCORH", "WRADH", "XYZZY"] {
            assert!(
                canonical_claim(name).is_none(),
                "{name} must not claim a canonical token"
            );
        }
    }
}
