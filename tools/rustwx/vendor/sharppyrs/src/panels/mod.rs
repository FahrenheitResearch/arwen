//! The non-skew-T panels of the full SPC window. Every panel exposes
//! `draw(painter, rect, prof, derived, style)` and paints in the exact style
//! of its Python counterpart (see PORTING.md for the source of each).

pub mod advection;
pub mod hazard;
pub mod hodo;
pub(crate) mod hodo_map_data;
pub mod index_board;
pub mod locator;
pub mod ship_inset;
pub mod slinky;
pub mod speed;
pub mod srwinds;
pub mod stp;
pub mod streamwiseness;
pub mod thetae;
