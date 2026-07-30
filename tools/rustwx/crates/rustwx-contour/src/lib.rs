//! Renderer-agnostic contour and fill topology for the `rustwx` plotting stack.
//!
//! This crate is intended to sit between scalar-field preparation and image
//! rendering. It understands grid geometry, contour levels, and band bins, then
//! produces line segments and polygon topology that a downstream renderer can
//! stroke or fill.
//!
//! `rustwx-render` can integrate by building either a [`RectilinearGrid`] or a
//! [`ProjectedGrid`], wrapping it in [`ScalarField2D`], then extracting
//! topology with [`ContourEngine`]. The resulting [`ContourTopology`] and
//! [`FillTopology`] are renderer-agnostic and do not assume any raster backend.
//!
//! ```rust
//! use rustwx_contour::{
//!     ContourEngine, ContourLevels, ExtendMode, LevelBins, RectilinearGrid, ScalarField2D,
//! };
//!
//! let grid = RectilinearGrid::new(vec![0.0, 1.0, 2.0], vec![0.0, 1.0, 2.0])?;
//! let field = ScalarField2D::new(
//!     grid,
//!     vec![
//!         0.0, 1.0, 2.0,
//!         1.0, 2.0, 3.0,
//!         2.0, 3.0, 4.0,
//!     ],
//! )?;
//!
//! let engine = ContourEngine::new();
//! let lines = engine.extract_isolines(&field, &ContourLevels::new(vec![1.5, 2.5])?);
//! let fills = engine.extract_filled_bands(
//!     &field,
//!     &LevelBins::with_extend(vec![1.0, 2.0, 3.0], ExtendMode::Both)?,
//! );
//!
//! assert_eq!(lines.layers.len(), 2);
//! assert!(!fills.polygons.is_empty());
//! # Ok::<(), rustwx_contour::ContourError>(())
//! ```

mod engine;
mod error;
mod field;
mod geometry;
mod grid;
mod levels;
mod topology;

pub use engine::{ContourEngine, SaddleResolver};
pub use error::ContourError;
pub use field::{ScalarField2D, ScalarMetadata};
pub use geometry::{LineSegment, Point2, Polygon, Ring};
pub use grid::{Grid2D, GridShape, ProjectedGrid, RectilinearGrid};
pub use levels::{ContourLevels, ExtendMode, LevelBin, LevelBins, LevelBound};
pub use topology::{BandPolygon, ContourLayer, ContourSegment, ContourTopology, FillTopology};
