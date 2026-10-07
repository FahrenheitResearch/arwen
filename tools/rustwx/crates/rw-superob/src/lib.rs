//! Gate-to-cell superobbing: many range gates in, one observation per cell
//! out.  The Rust port of `gpuwm/obs/superob.py`.
//!
//! The Python module is the reference and its docstrings hold the science
//! rulings (why reflectivity is averaged in linear Z, why velocity never
//! merges across radars, which gates may be called clear air, what each of
//! the four alias masks can and cannot see, the CC QC ruling and its
//! debris-fringe exemption).  This crate does the same arithmetic in the
//! same order so the two agree to the bit wherever the elementary
//! functions agree, and it says so where they cannot (see
//! `tests/` and the parity receipts).
//!
//! Determinism: every accumulator is reduced in a fixed order -- sweeps in
//! pack order, moments in pack order, radials ascending, gates ascending,
//! and contributions in caller order at the merge -- which is the order
//! `numpy.add.at` applies the reference's additions in.  Parallelism is
//! only ever over work whose results do not depend on order (gate
//! geometry, disjoint output cells).

pub mod capi;
pub mod ccqc;
pub mod dealias;
pub mod geometry;
pub mod grid;
pub mod merge;
pub mod num;
pub mod request;
pub mod volume;

/// The error every operation refuses with.  `kind` is a stable token the
/// Python seam maps onto its own exception classes.
#[derive(Debug, Clone)]
pub struct SuperobFailure {
    pub kind: &'static str,
    pub message: String,
}

impl SuperobFailure {
    pub fn request(message: impl Into<String>) -> Self {
        Self { kind: "request", message: message.into() }
    }

    pub fn window(message: impl Into<String>) -> Self {
        Self { kind: "window", message: message.into() }
    }
}

pub type Result<T> = std::result::Result<T, SuperobFailure>;
