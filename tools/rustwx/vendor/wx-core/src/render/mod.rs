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

pub mod ansi;
pub mod colormap;
pub mod contour;
pub mod cross_section;
pub mod encode;
pub mod filled_contour;
pub mod hodograph;
pub mod overlay;
pub mod raster;
pub mod skewt;
pub mod station;

// ── Re-exports for convenient access ──────────────────────────────

// colormap
pub use colormap::{get_colormap, interpolate_color, list_colormaps, ColorStop};
pub use colormap::{
    CAPE, CAPE_PIVOTAL, CLOUD_COVER, DEWPOINT, DIVERGENCE, GOES_IR, HELICITY, ICE, NWS_PRECIP,
    NWS_REFLECTIVITY, PRECIPITATION, PRESSURE, REFLECTIVITY, REFLECTIVITY_CLEAN, RELATIVE_HUMIDITY,
    SNOW, TEMPERATURE, TEMPERATURE_NWS, TEMPERATURE_PIVOTAL, THETA_E, VISIBILITY, VORTICITY, WIND,
    WIND_PIVOTAL,
};

// raster
pub use raster::{render_raster, render_raster_par, render_raster_with_colormap};

// contour
pub use contour::{contour_lines, contour_lines_labeled, ContourLine, LabeledContour};

// filled_contour
pub use filled_contour::{
    auto_levels, render_filled_contours, render_filled_contours_with_colormap,
};

// overlay
pub use overlay::{overlay_contours, overlay_streamlines, overlay_wind_barbs};

// encode
pub use encode::{encode_png, write_png};

// ansi
pub use ansi::{rgba_to_ansi, rgba_to_ansi_mode, AnsiMode};

// skewt
pub use skewt::{render_skewt, SkewTConfig, SkewTData};

// hodograph
pub use hodograph::{render_hodograph, HodographConfig, HodographData};

// station
pub use station::{render_station_plot, StationObs, StationPlotConfig};

// cross_section
pub use cross_section::{render_cross_section, CrossSectionConfig, CrossSectionData};
