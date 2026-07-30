//! wx-field -- Shared type foundation for the unified atmospheric engine.
//!
//! Provides core data structures for weather fields, radar data, soundings,
//! map projections, and metadata. This crate has minimal dependencies and
//! serves as the common type layer that all other wx-* crates depend on.

pub mod error;
pub mod field;
pub mod meta;
pub mod projection;
pub mod radial;
pub mod site;
pub mod sounding;
pub mod time;

// Re-export the most commonly used types at crate root.
pub use error::{WxFieldError, Result};
pub use field::Field2D;
pub use meta::{FieldMeta, Units, Level, DataSource};
pub use projection::{
    Projection, LambertProjection, LatLonProjection,
    PolarStereoProjection, MercatorProjection, GaussianProjection,
};
pub use radial::{Radial, RadialSweep, RadialField};
pub use site::RadarSite;
pub use sounding::{SoundingLevel, SoundingProfile};
pub use time::{ValidTime, ForecastHour, ModelRun};
