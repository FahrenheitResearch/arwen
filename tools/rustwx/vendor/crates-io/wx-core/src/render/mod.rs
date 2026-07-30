//! Lightweight rendering module for weather maps.
//!
//! Provides zero-dependency plotting primitives:
//! - Colormaps: standard meteorological color scales
//! - Raster: grid-to-RGBA pixel rendering
//! - Contour: marching squares isopleths
//! - PNG: minimal PNG encoder for output
//!
//! This is a library-level module suitable for embedding in Python bindings
//! or any consumer of wx-core. No windowing, no GPU, no matplotlib.

pub mod colormap;
pub mod raster;
pub mod contour;
pub mod filled_contour;
pub mod overlay;
pub mod encode;
pub mod ansi;
pub mod skewt;
pub mod hodograph;
pub mod station;
pub mod cross_section;

// ── Re-exports for convenient access ──────────────────────────────

// colormap
pub use colormap::{ColorStop, interpolate_color, get_colormap, list_colormaps};
pub use colormap::{
    TEMPERATURE, DEWPOINT, PRECIPITATION, WIND, REFLECTIVITY, CAPE,
    RELATIVE_HUMIDITY, VORTICITY, PRESSURE, SNOW, ICE, VISIBILITY,
    CLOUD_COVER, HELICITY, DIVERGENCE, THETA_E, NWS_REFLECTIVITY,
    NWS_PRECIP, GOES_IR,
    TEMPERATURE_NWS, TEMPERATURE_PIVOTAL, CAPE_PIVOTAL,
    WIND_PIVOTAL, REFLECTIVITY_CLEAN,
};

// raster
pub use raster::{render_raster, render_raster_with_colormap, render_raster_par};

// contour
pub use contour::{ContourLine, LabeledContour, contour_lines, contour_lines_labeled};

// filled_contour
pub use filled_contour::{
    render_filled_contours, render_filled_contours_with_colormap, auto_levels,
};

// overlay
pub use overlay::{overlay_contours, overlay_wind_barbs, overlay_streamlines};

// encode
pub use encode::{write_png, encode_png};

// ansi
pub use ansi::{rgba_to_ansi, rgba_to_ansi_mode, AnsiMode};

// skewt
pub use skewt::{SkewTConfig, SkewTData, render_skewt};

// hodograph
pub use hodograph::{HodographConfig, HodographData, render_hodograph};

// station
pub use station::{StationObs, StationPlotConfig, render_station_plot};

// cross_section
pub use cross_section::{CrossSectionConfig, CrossSectionData, render_cross_section};
