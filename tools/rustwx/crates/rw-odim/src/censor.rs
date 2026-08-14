//! Why a gate in [`Moment::values`](crate::Moment::values) is not a number.
//!
//! ODIM_H5 reserves two raw storage values per moment, declared in
//! `/datasetN/dataM/what`, and they are **opposites**:
//!
//! * `undetect` -- the radar illuminated this gate and the return did not
//!   clear the detection threshold. That is a detection of *nothing*, which
//!   is a real observation of clear air and the thing every correct negative
//!   in a skill score is built from.
//! * `nodata` -- the radar did not measure this gate at all. It is evidence
//!   of nothing, and may well be hiding a storm.
//!
//! [`Moment::values`](crate::Moment::values) maps both to `f64::NAN`, because
//! a caller that only wants numbers should not have to know the vocabulary.
//! The *reason* is carried beside it in
//! [`Moment::censor`](crate::Moment::censor) instead -- one code per gate,
//! parallel to `values` and always the same length. That is the whole point
//! of this module: a consumer that needs the distinction can have it, and one
//! that does not is untouched, but the distinction is never *lost*.
//!
//! This mirrors `wx_radar::level2::censor`, which does the same job for
//! NEXRAD Message-31, and it deliberately shares that module's code space.
//!
//! # Why these particular numbers
//!
//! The two vocabularies are not the same, so the codes are chosen so that the
//! values which *mean* the same thing *are* the same, and the values which
//! mean different things are never confused:
//!
//! | code | this module | `wx_radar::level2::censor` | same meaning? |
//! |-----:|-------------|----------------------------|---------------|
//! | 0 | [`MEASURED`] | `MEASURED` | yes |
//! | 1 | [`UNDETECT`] | `BELOW_THRESHOLD` | yes -- both are "looked, found nothing" |
//! | 2 | *never minted* | `RANGE_FOLDED` | n/a -- ODIM has no such state |
//! | 3 | [`NOT_COLLECTED`] | `NOT_COLLECTED` | yes |
//! | 4 | [`NODATA`] | -- | ODIM-only |
//! | 5 | [`SENTINEL_AMBIGUOUS`] | -- | ODIM-only |
//!
//! Code 2 is the load-bearing gap. NEXRAD's range-folded gate has no ODIM
//! counterpart, and ODIM's `nodata` is *not* range folding -- so `nodata`
//! takes a fresh code rather than borrowing 2. A consumer that reads both
//! planes with one table therefore cannot mistake a European unmeasured gate
//! for an American second-trip ambiguity. See [`RESERVED_RANGE_FOLDED`].
//!
//! The codes are `u8` and deliberately not a Rust `enum`: they are meant to
//! be transcribed verbatim into a `|u1` plane that crosses a file boundary
//! into an observation pack, and a numeric contract that crosses a file
//! boundary should be stated as numbers.

/// `values` holds a decoded measurement.
pub const MEASURED: u8 = 0;

/// ODIM `undetect`: the radar looked at this gate and detected nothing.
///
/// Clear air. Numerically equal to `wx_radar::level2::censor::BELOW_THRESHOLD`
/// because it means the same thing.
pub const UNDETECT: u8 = 1;

/// Not minted by this decoder, and reserved so that it stays not-minted.
///
/// In `wx_radar::level2::censor` the value 2 is `RANGE_FOLDED` -- a return the
/// RDA cannot place in range. ODIM_H5 has no equivalent state: a European
/// radar's second-trip handling happens before the file is written, and
/// nothing in `/datasetN/dataM/what` declares it. Leaving 2 unused is what
/// lets the two planes share one lookup table safely.
pub const RESERVED_RANGE_FOLDED: u8 = 2;

/// A gate this decoder never saw a value for.
///
/// Not produced by this decoder: every gate it reports came from a payload
/// whose shape it checked against the sweep's declared `nrays`/`nbins`.
/// Reserved for a consumer that widens moments into a common rectangle -- a
/// sweep that carried no such moment at all -- so that "not collected" stays
/// distinct from "collected, and empty". Numerically equal to
/// `wx_radar::level2::censor::NOT_COLLECTED`.
pub const NOT_COLLECTED: u8 = 3;

/// ODIM `nodata`: the radar did not measure this gate.
///
/// Never clear air, and never admissible as a correct negative.
pub const NODATA: u8 = 4;

/// The file declared `nodata` and `undetect` as the *same* raw value, so a
/// gate holding it cannot be told apart.
///
/// This is not a decoder defect, it is a property of the file, and it is
/// observed in the wild: Finnish `VRADH` scans on the MeteoGate feed declare
/// `nodata = 0.0` and `undetect = 0.0`. Collapsing such a gate silently into
/// either neighbour would be inventing an observation in one direction or
/// discarding a correct negative in the other, so it gets its own code and
/// its own count in [`Census::sentinel_ambiguous`](crate::Census).
///
/// A consumer must treat this as **not** clear air: it may be `nodata`.
pub const SENTINEL_AMBIGUOUS: u8 = 5;

/// Human-readable name for a code, for provenance records and CLI output.
pub fn name(code: u8) -> &'static str {
    match code {
        MEASURED => "measured",
        UNDETECT => "undetect",
        RESERVED_RANGE_FOLDED => "reserved_range_folded",
        NOT_COLLECTED => "not_collected",
        NODATA => "nodata",
        SENTINEL_AMBIGUOUS => "sentinel_ambiguous",
        _ => "unknown",
    }
}

/// Whether a code denotes a gate the radar actually observed, i.e. one that
/// carries information -- either a measurement or a genuine correct negative.
///
/// [`SENTINEL_AMBIGUOUS`] is deliberately **not** observed: the file cannot
/// say whether the radar looked.
pub fn is_observed(code: u8) -> bool {
    code == MEASURED || code == UNDETECT
}

/// The per-gate census of a decoded moment.
///
/// Every gate is counted exactly once, and the total always equals the gate
/// count, which is what makes the classification auditable rather than merely
/// asserted.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Census {
    /// Gates holding a decoded physical value.
    pub measured: usize,
    /// Gates the radar looked at and found empty (ODIM `undetect`).
    pub undetect: usize,
    /// Gates the radar did not measure (ODIM `nodata`).
    pub nodata: usize,
    /// Gates no payload was read for. Always zero from this decoder.
    pub not_collected: usize,
    /// Gates whose raw value matched two sentinels the file declared equal.
    pub sentinel_ambiguous: usize,
}

impl Census {
    /// Total gates accounted for.
    pub fn total(&self) -> usize {
        self.measured + self.undetect + self.nodata + self.not_collected + self.sentinel_ambiguous
    }

    /// Fraction of gates that carry information: measured plus correct
    /// negatives, over the total.
    pub fn observed_fraction(&self) -> f64 {
        let total = self.total();
        if total == 0 {
            return 0.0;
        }
        (self.measured + self.undetect) as f64 / total as f64
    }

    pub(crate) fn tally(&mut self, code: u8) {
        match code {
            MEASURED => self.measured += 1,
            UNDETECT => self.undetect += 1,
            NODATA => self.nodata += 1,
            NOT_COLLECTED => self.not_collected += 1,
            SENTINEL_AMBIGUOUS => self.sentinel_ambiguous += 1,
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_shared_codes_line_up_with_the_nexrad_plane() {
        // These three mean the same thing in both vocabularies and must keep
        // the same numbers, or a consumer reading both planes with one table
        // silently mis-labels gates.
        assert_eq!(MEASURED, 0);
        assert_eq!(UNDETECT, 1);
        assert_eq!(NOT_COLLECTED, 3);
    }

    #[test]
    fn the_odim_only_codes_avoid_the_range_folded_slot() {
        // 2 is RANGE_FOLDED on the NEXRAD plane and ODIM has no such state,
        // so nothing here may take it.
        assert_eq!(RESERVED_RANGE_FOLDED, 2);
        assert_ne!(NODATA, RESERVED_RANGE_FOLDED);
        assert_ne!(SENTINEL_AMBIGUOUS, RESERVED_RANGE_FOLDED);
        for code in [
            MEASURED,
            UNDETECT,
            NOT_COLLECTED,
            NODATA,
            SENTINEL_AMBIGUOUS,
        ] {
            assert_ne!(code, RESERVED_RANGE_FOLDED, "{} collides", name(code));
        }
    }

    #[test]
    fn only_measurement_and_clear_air_count_as_observed() {
        assert!(is_observed(MEASURED));
        assert!(is_observed(UNDETECT));
        // An unmeasured gate is not an observation, and an ambiguous one
        // cannot be shown to be one.
        assert!(!is_observed(NODATA));
        assert!(!is_observed(NOT_COLLECTED));
        assert!(!is_observed(SENTINEL_AMBIGUOUS));
    }

    #[test]
    fn the_census_accounts_for_every_gate_exactly_once() {
        let mut census = Census::default();
        for code in [MEASURED, MEASURED, UNDETECT, NODATA, SENTINEL_AMBIGUOUS] {
            census.tally(code);
        }
        assert_eq!(census.total(), 5);
        assert_eq!(census.measured, 2);
        assert_eq!(census.undetect, 1);
        assert_eq!(census.nodata, 1);
        assert_eq!(census.sentinel_ambiguous, 1);
        // Two of the five carry information; the ambiguous one deliberately
        // does not count towards it.
        assert!((census.observed_fraction() - 0.6).abs() < 1e-12);
    }

    #[test]
    fn every_code_has_a_name() {
        for code in [
            MEASURED,
            UNDETECT,
            RESERVED_RANGE_FOLDED,
            NOT_COLLECTED,
            NODATA,
            SENTINEL_AMBIGUOUS,
        ] {
            assert_ne!(name(code), "unknown");
        }
        assert_eq!(name(200), "unknown");
    }
}
