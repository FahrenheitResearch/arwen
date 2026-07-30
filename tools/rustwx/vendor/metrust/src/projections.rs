//! Map projections and spatial data types.
//!
//! Re-exports projection implementations, field containers, metadata,
//! sounding profiles, radial (radar) data types, and radar site info
//! from the `wx-field` crate.

// ── Projections ──────────────────────────────────────────────────────
pub use wx_field::projection::{
    GaussianProjection, LambertProjection, LatLonProjection, MercatorProjection,
    PolarStereoProjection, Projection,
};

// ── 2-D gridded field ────────────────────────────────────────────────
pub use wx_field::field::Field2D;

// ── Metadata ─────────────────────────────────────────────────────────
pub use wx_field::meta::{DataSource, FieldMeta, Level, Units};

// ── Sounding ─────────────────────────────────────────────────────────
pub use wx_field::sounding::{SoundingLevel, SoundingProfile};

// ── Radial (radar) data ──────────────────────────────────────────────
pub use wx_field::radial::{Radial, RadialField, RadialSweep};

// ── Radar site ───────────────────────────────────────────────────────
pub use wx_field::site::RadarSite;

// ── Time types ───────────────────────────────────────────────────────
pub use wx_field::time::{ForecastHour, ModelRun, ValidTime};
