//! Map projections and spatial data types.
//!
//! Re-exports projection implementations, field containers, metadata,
//! sounding profiles, radial (radar) data types, and radar site info
//! from the `wx-field` crate.

// ── Projections ──────────────────────────────────────────────────────
pub use wx_field::projection::{
    Projection, LambertProjection, LatLonProjection,
    PolarStereoProjection, MercatorProjection, GaussianProjection,
};

// ── 2-D gridded field ────────────────────────────────────────────────
pub use wx_field::field::Field2D;

// ── Metadata ─────────────────────────────────────────────────────────
pub use wx_field::meta::{FieldMeta, Units, Level, DataSource};

// ── Sounding ─────────────────────────────────────────────────────────
pub use wx_field::sounding::{SoundingProfile, SoundingLevel};

// ── Radial (radar) data ──────────────────────────────────────────────
pub use wx_field::radial::{RadialField, RadialSweep, Radial};

// ── Radar site ───────────────────────────────────────────────────────
pub use wx_field::site::RadarSite;

// ── Time types ───────────────────────────────────────────────────────
pub use wx_field::time::{ValidTime, ForecastHour, ModelRun};
