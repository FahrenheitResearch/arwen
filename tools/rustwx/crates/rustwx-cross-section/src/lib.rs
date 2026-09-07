//! Foundational utilities for extracting and rendering atmospheric cross-sections.
//!
//! The crate is intentionally small and dependency-light. It focuses on the stable
//! pieces that multiple clients tend to need:
//! - path construction and equal-distance sampling
//! - cumulative along-section distance and bearings
//! - pressure and height vertical-axis handling
//! - terrain and metadata containers
//! - wind decomposition into along/across-section components
//! - a small RGBA renderer for quick previews and tests
//!
//! # Example
//! ```rust
//! use rustwx_cross_section::{
//!     render_scalar_section, Color, CrossSectionRenderRequest, CrossSectionRequest, GeoPoint,
//!     SamplingStrategy, ScalarSection, SectionMetadata, SectionPath, TerrainProfile, VerticalAxis,
//! };
//!
//! let path = SectionPath::endpoints(
//!     GeoPoint::new(39.7392, -104.9903)?,
//!     GeoPoint::new(41.8781, -87.6298)?,
//! )?;
//! let layout = CrossSectionRequest::new(path)
//!     .with_sampling(SamplingStrategy::Count(5))
//!     .build_layout()?;
//!
//! let axis = VerticalAxis::pressure_hpa(vec![1000.0, 850.0, 700.0, 500.0])?;
//! let mut values = Vec::new();
//! for level in 0..axis.len() {
//!     for point in 0..layout.sampled_path.len() {
//!         values.push(280.0 + point as f32 * 2.0 - level as f32 * 8.0);
//!     }
//! }
//!
//! let terrain = TerrainProfile::from_surface_pressure_hpa(
//!     layout.sampled_path.distances_km().to_vec(),
//!     vec![960.0, 930.0, 900.0, 920.0, 950.0],
//! )?;
//!
//! let section = ScalarSection::new(layout.sampled_path.distances_km().to_vec(), axis, values)?
//!     .with_metadata(SectionMetadata::new().field("temperature", "K"))
//!     .with_terrain(terrain)?;
//!
//! let image = render_scalar_section(
//!     &section,
//!     &CrossSectionRenderRequest::default().with_palette(vec![
//!         Color::rgb(49, 54, 149),
//!         Color::rgb(69, 117, 180),
//!         Color::rgb(116, 173, 209),
//!         Color::rgb(254, 224, 144),
//!         Color::rgb(215, 48, 39),
//!     ]),
//! )?;
//! assert_eq!(image.rgba().len(), (image.width() * image.height() * 4) as usize);
//! # Ok::<(), rustwx_cross_section::CrossSectionError>(())
//! ```

mod data;
mod error;
mod geo;
mod palette;
mod render;
mod request;
mod style;
mod vertical;
mod wind;

pub use data::{ScalarSection, SectionMetadata, TerrainProfile};
pub use error::CrossSectionError;
pub use geo::{
    GeoBounds, GeoPoint, RepresentativeRoute, RepresentativeRouteStrategy, SampledPath,
    SectionPath, haversine_distance_km, initial_bearing_deg, intermediate_point,
    representative_route_for_bounds, representative_route_for_cluster,
};
pub use palette::{ALL_CROSS_SECTION_PALETTES, CrossSectionPalette, PaletteStop};
pub use render::{
    Color, CrossSectionRenderRequest, CrossSectionRenderTiming, Insets, RenderedCrossSection,
    ScalarContourOverlayBundle, WindOverlayBundle, WindOverlayStyle, install_cross_section_fonts,
    render_scalar_section, render_scalar_section_profile,
};
pub use request::{
    CrossSectionRequest, HorizontalInterpolation, SamplingStrategy, SectionLayout, VerticalWindow,
};
pub use style::{
    ALL_CROSS_SECTION_PRODUCTS, CrossSectionProduct, CrossSectionProductGroup, CrossSectionStyle,
};
pub use vertical::{
    VerticalAxis, VerticalKind, VerticalScale, VerticalUnits, standard_atmosphere_height_m,
};
pub use wind::{DecomposedWindGrid, WindDecomposition, decompose_wind, decompose_wind_grid};
