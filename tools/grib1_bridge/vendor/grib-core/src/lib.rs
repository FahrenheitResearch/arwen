//! GRIB1/GRIB2 decoding for this repository.
//!
//! # Convergence notes (2026-08-17)
//!
//! This crate is the ONE grib-core: the superset of gpuwm's hardened
//! descendant and the `rusty-weather` consolidation's descendant, which had
//! diverged in nine files.  Behaviour unique to either side is present here;
//! where the two disagreed the resolution is fail-closed and recorded below,
//! keyed by the marker each conflict site carries.
//!
//! **C1 — unknown grid-definition template.**  `grib2::grid_latlon` REFUSES,
//! naming the template.  The donor returned empty coordinate vectors, which
//! let an unsupported input decode "successfully" to nothing and reach
//! sha-bound artifacts as a good decode.  The donor's
//! `test_unknown_template_returns_empty` survives here re-expressed as
//! `test_unknown_template_fails_closed_naming_the_template`; the
//! returns-empty behaviour is what is refused, not a fallback kept for
//! non-decode probes.  A reduced grid with no `pl` array is refused the same
//! way.
//!
//! **C2 — IEEE (DRT 5.4) payload length.**  The Section-5 point count is the
//! authority.  Trailing Section-7 bytes beyond `count * width` are TOLERATED
//! (donor semantics) and dropped by `unpack_message`, which is where the
//! tolerance is recorded; a payload SHORT of the declared count is REFUSED
//! (gpuwm semantics), because a short IEEE payload silently shortens the
//! field.  The struct keeps `section5_num_data_points`, so the declared
//! count stays auditable after the trim.
//!
//! **C3 — parameter-table lookups.**  Unknown cited or local tables refuse
//! (gpuwm semantics; `grib1::parameter_entry` returns `Err` naming version,
//! center and parameter).  Donor rows that resolve through WMO 4.2/4.5 plus
//! the NCEP extensions merge additively — no row was dropped in either
//! direction.
//!
//! **C4 — mode 1/2 missing values with `bits_per_value == 0`.**  The refusal
//! is retained but SCOPED to where the ambiguity exists: a zero-width
//! (constant) group, whose reference is simultaneously the only representable
//! value and the all-ones marker.  It used to refuse every such field,
//! including fields whose groups all carry a real width from
//! `group_width_ref`; those decode unambiguously and the donor's complex and
//! complex+spatial missing-value tests exercise them.  See
//! `grib2::unpack::refuse_ambiguous_constant_group`.
//!
//! **C5 — longitude spacing on Templates 3.0 and 3.40.**  The other side
//! preferred the DECLARED i-direction increment whenever one was present, to
//! fix cyclic grids whose first and last longitudes are equal.  Adopted
//! whole, that regressed real operational placement: a 3072-point Gaussian
//! row states `Di = 0.117188` for a true `0.1171875` spacing, and taking the
//! rounded octet at face value walks the last column ~170 m east.  The rule
//! kept here is the narrow one — endpoints place the row wherever they
//! describe a span; the declared increment supplies the spacing only where
//! they do not (equal endpoints) and the sign on a -i scan.  Both halves are
//! pinned by tests in `grib2::grid`.

pub mod grib1;
pub mod grib2;

/// Error types for GRIB file operations.
#[derive(Debug)]
pub enum GribError {
    /// I/O error reading a file.
    Io(std::io::Error),
    /// Error parsing a GRIB message structure.
    Parse(String),
    /// Error unpacking data values.
    Unpack(String),
    /// Unsupported template number.
    UnsupportedTemplate { template: u16, detail: String },
}

impl std::fmt::Display for GribError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GribError::Io(e) => write!(f, "I/O error: {}", e),
            GribError::Parse(msg) => write!(f, "Parse error: {}", msg),
            GribError::Unpack(msg) => write!(f, "Unpack error: {}", msg),
            GribError::UnsupportedTemplate { template, detail } => {
                write!(f, "Unsupported {} template: {}", detail, template)
            }
        }
    }
}

impl std::error::Error for GribError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            GribError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for GribError {
    fn from(e: std::io::Error) -> Self {
        GribError::Io(e)
    }
}

/// Convenience type alias for Results using GribError.
pub type Result<T> = std::result::Result<T, GribError>;
