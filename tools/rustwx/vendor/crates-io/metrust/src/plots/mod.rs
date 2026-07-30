//! Plotting module -- MetPy-compatible rendering primitives.
//!
//! Re-exports skew-T, hodograph, station plot, colormap, contour,
//! raster, cross-section, and radar PPI renderers.

// ── Skew-T / Hodograph / Station ─────────────────────────────────────
pub use wx_core::render::skewt::{SkewTConfig, SkewTData, render_skewt};
pub use wx_core::render::hodograph::{HodographConfig, HodographData, render_hodograph};
pub use wx_core::render::station::{StationObs, StationPlotConfig, render_station_plot};

// ── Colormaps ────────────────────────────────────────────────────────
pub use wx_core::render::colormap::{
    interpolate_color, get_colormap, list_colormaps, ColorStop,
};
pub use wx_core::render::colormap::{
    TEMPERATURE, DEWPOINT, PRECIPITATION, WIND, REFLECTIVITY, CAPE,
    RELATIVE_HUMIDITY, VORTICITY, PRESSURE, SNOW, ICE, VISIBILITY,
    CLOUD_COVER, HELICITY, DIVERGENCE, THETA_E, NWS_REFLECTIVITY,
    NWS_PRECIP, GOES_IR,
    TEMPERATURE_NWS, TEMPERATURE_PIVOTAL, CAPE_PIVOTAL,
    WIND_PIVOTAL, REFLECTIVITY_CLEAN,
};

// ── Contours ─────────────────────────────────────────────────────────
pub use wx_core::render::contour::{
    contour_lines, contour_lines_labeled, ContourLine, LabeledContour,
};
pub use wx_core::render::filled_contour::{
    render_filled_contours, render_filled_contours_with_colormap, auto_levels,
};

// ── Overlays ─────────────────────────────────────────────────────────
pub use wx_core::render::overlay::{
    overlay_contours, overlay_wind_barbs, overlay_streamlines,
};

// ── Raster ───────────────────────────────────────────────────────────
pub use wx_core::render::raster::{
    render_raster, render_raster_with_colormap, render_raster_par,
};

// ── Cross-section ────────────────────────────────────────────────────
pub use wx_core::render::cross_section::{
    CrossSectionConfig, CrossSectionData, render_cross_section,
};

// ── Encoding ─────────────────────────────────────────────────────────
pub use wx_core::render::encode::{write_png, encode_png};
pub use wx_core::render::ansi::{rgba_to_ansi, rgba_to_ansi_mode, AnsiMode};

// ── Radar rendering ──────────────────────────────────────────────────
pub use wx_radar::render::{render_ppi, render_ppi_with_table, RenderedPPI};
pub use wx_radar::color_table::ColorTable;
pub use wx_radar::cells::{identify_cells, StormCell};
