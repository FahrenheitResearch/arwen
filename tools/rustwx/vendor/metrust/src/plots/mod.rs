//! Plotting module -- MetPy-compatible rendering primitives.
//!
//! Re-exports skew-T, hodograph, station plot, colormap, contour,
//! raster, cross-section, and radar PPI renderers.

// ── Skew-T / Hodograph / Station ─────────────────────────────────────
pub use wx_core::render::hodograph::{render_hodograph, HodographConfig, HodographData};
pub use wx_core::render::skewt::{render_skewt, SkewTConfig, SkewTData};
pub use wx_core::render::station::{render_station_plot, StationObs, StationPlotConfig};

// ── Colormaps ────────────────────────────────────────────────────────
pub use wx_core::render::colormap::{get_colormap, interpolate_color, list_colormaps, ColorStop};
pub use wx_core::render::colormap::{
    CAPE, CAPE_PIVOTAL, CLOUD_COVER, DEWPOINT, DIVERGENCE, GOES_IR, HELICITY, ICE, NWS_PRECIP,
    NWS_REFLECTIVITY, PRECIPITATION, PRESSURE, REFLECTIVITY, REFLECTIVITY_CLEAN, RELATIVE_HUMIDITY,
    SNOW, TEMPERATURE, TEMPERATURE_NWS, TEMPERATURE_PIVOTAL, THETA_E, VISIBILITY, VORTICITY, WIND,
    WIND_PIVOTAL,
};

// ── Contours ─────────────────────────────────────────────────────────
pub use wx_core::render::contour::{
    contour_lines, contour_lines_labeled, ContourLine, LabeledContour,
};
pub use wx_core::render::filled_contour::{
    auto_levels, render_filled_contours, render_filled_contours_with_colormap,
};

// ── Overlays ─────────────────────────────────────────────────────────
pub use wx_core::render::overlay::{overlay_contours, overlay_streamlines, overlay_wind_barbs};

// ── Raster ───────────────────────────────────────────────────────────
pub use wx_core::render::raster::{render_raster, render_raster_par, render_raster_with_colormap};

// ── Cross-section ────────────────────────────────────────────────────
pub use wx_core::render::cross_section::{
    render_cross_section, CrossSectionConfig, CrossSectionData,
};

// ── Encoding ─────────────────────────────────────────────────────────
pub use wx_core::render::ansi::{rgba_to_ansi, rgba_to_ansi_mode, AnsiMode};
pub use wx_core::render::encode::{encode_png, write_png};

// ── Radar rendering ──────────────────────────────────────────────────
pub use wx_radar::cells::{identify_cells, StormCell};
pub use wx_radar::color_table::ColorTable;
pub use wx_radar::render::{render_ppi, render_ppi_with_table, RenderedPPI};
