//! Putting an observation and a forecast on one grid, in Rust.
//!
//! The port target is `gpuwm/verify/obs/regrid.py`, the remap the
//! observation battery scores through.  Drew's Python boundary names
//! "regrid/transform" as data-path processing, and this crate is where
//! that half of the battery now runs; the Python module keeps its
//! signatures and routes here by default.
//!
//! **Seeded from Drew's Rust.**  `crates/rustwx-regrid` in the
//! consolidated workspace supplies the shape of this crate rather than
//! its arithmetic: the plan-as-data discipline (a remap between two
//! fixed grids is a fixed mapping, built once and applied many times),
//! the caller-owned output buffer on apply, the bounded-distance
//! validation and its refusal class, and the error taxonomy in
//! [`error`].  What that crate does NOT carry is this operator pair --
//! it regrids structured grids with sparse weights and a NaN missing
//! policy, and the observation battery remaps scattered curvilinear
//! swaths with an explicit validity field and a reverse-assignment
//! cell average.  Those are written here, against the Python.
//!
//! **The parity contract is bitwise.**  Every value this crate produces
//! is compared against goldens extracted from the real scipy/numpy path
//! on real observation and model grids, and the comparison is on IEEE
//! bit patterns, not a tolerance.  The single documented divergence is
//! nearest-neighbour tie-breaking, which scipy leaves to traversal
//! order and this crate defines as lowest-index-wins; see [`kdtree`] for
//! the measurements and the breakage that rule prevents.
//!
//! Three properties carried over from the Python module verbatim,
//! because they are the reasons it exists:
//!
//! * **The plan is data, computed once.**  Every arm of a case is
//!   remapped by the identical integer array, so no score can differ
//!   because a neighbour search broke a tie differently.
//! * **Distance is bounded and the bound is enforced.**  A destination
//!   cell with no source centre within `max_distance_m` is marked
//!   invalid, not filled from far away.
//! * **Validity is remapped with the values.**  A destination cell built
//!   only from invalid sources is invalid.  Missing does not become zero
//!   at a grid change.

pub mod capi;
pub mod error;
pub mod geometry;
pub mod kdtree;
pub mod plan;

pub use error::RegridError;
pub use geometry::{EARTH_RADIUS_M, arc_from_chord, chord_from_arc, unit_vectors};
pub use kdtree::{KdTree, Neighbour};
pub use plan::{Method, RegridPlan, apply_plan, build_plan, unreachable_destination_cells};
