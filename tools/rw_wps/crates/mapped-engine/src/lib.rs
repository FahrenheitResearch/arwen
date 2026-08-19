//! `gpuwm_mapped_engine` — the decode engine behind gpuwm's mapped prep.
//!
//! The seam is normative in `docs/dev/decode-vendor-design.md` §3, and this
//! crate and `gpuwm/mapped_engine_bridge.py` both encode it.  What moved
//! here is DATA PATH: selector→record matching, declared-grid cross-checks,
//! wind rotation, record assembly, transposition, unit transforms, the
//! closed derivation catalog, canonical-frame invariants, missing-count
//! accounting, and the NetCDF variable resolution above the reader.  What
//! did NOT move is orchestration: authority snapshotting, manifest
//! verification, the decoder ladder, receipts, member grammar, CLI parsing
//! and exception-type selection all stay in Python, which is why this
//! binary never invokes Python and never shells out to another tool.
//!
//! Outputs: `frames.json` (schema [`FRAMESET_SCHEMA`], the compiled ABI
//! marker) beside `frames.f64` (one little-endian float64 stream, fields
//! packed row-major in manifest order).  Refusals leave as one JSON object
//! on the last stderr line, schema [`REFUSAL_SCHEMA`].

pub mod array;
pub mod assemble;
pub mod codec;
pub mod compose;
pub mod derive;
pub mod digest;
pub mod engine;
pub mod frames;
pub mod grib;
pub mod grib1;
pub mod join;
pub mod lambert;
pub mod model;
pub mod ncdf;
pub mod node;
pub mod refusal;
pub mod threads;

/// The ABI marker: the OUTPUT SCHEMA name, so it changes exactly when the
/// frameset contract changes and a stale staged binary fails the static
/// handshake instead of writing a shape the Python side no longer reads.
pub const FRAMESET_SCHEMA: &str = "gpuwm-mapped-frameset-v1";
pub const REFUSAL_SCHEMA: &str = "gpuwm-mapped-refusal-v1";
pub const PROGRESS_SCHEMA: &str = "gpuwm-mapped-engine-progress-v1";
pub const INSPECTION_SCHEMA: &str = "gpuwm-mapped-source-inspection-v1";
pub const CAPABILITIES_SCHEMA: &str = "gpuwm-mapped-engine-capabilities-v1";
/// The raw per-record product-identity surface (`inventory`): every
/// GRIB2 identity octet the subprocess `grib2_inventory` renders, in the
/// same string spellings, so a product-identity gate reads ONE spelling
/// whichever instrument measured it.
pub const RECORD_INVENTORY_SCHEMA: &str = "gpuwm-mapped-record-inventory-v1";

pub const ENGINE_NAME: &str = "gpuwm_mapped_engine";
pub const ENGINE_VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    #[test]
    fn the_marker_is_the_output_schema_name() {
        // Stated as a test because the two are the same string BY DESIGN:
        // if a future change gives the marker its own literal, the reason
        // the contract works — a frameset shape change forces a marker
        // change — quietly stops holding.
        assert_eq!(super::FRAMESET_SCHEMA, "gpuwm-mapped-frameset-v1");
    }
}
