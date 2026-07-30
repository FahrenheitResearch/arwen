//! Data I/O module -- MetPy-compatible structure for reading weather data.
//!
//! Re-exports GRIB2, NEXRAD Level-II, and data download primitives from
//! wx-core and wx-radar.  Also provides native parsers for Level-III
//! products, METAR text reports, and station lookup.

// ── GRIB2 ────────────────────────────────────────────────────────────
pub use wx_core::grib2::{
    Grib2File, Grib2Message, GridDefinition, ProductDefinition, DataRepresentation,
};
pub use wx_core::grib2::{unpack_message, unpack_message_normalized, flip_rows, BitReader};
pub use wx_core::grib2::tables::{parameter_name, parameter_units, level_name};
pub use wx_core::grib2::grid::{grid_latlon, rotated_to_geographic};
pub use wx_core::grib2::{search_messages, StreamingParser};
pub use wx_core::grib2::{
    Grib2Writer, MessageBuilder, PackingMethod,
};
pub use wx_core::grib2::{
    merge, subset, filter, split, field_diff, field_stats, field_stats_region,
    FieldStats, FieldOp, apply_op, smooth_gaussian, smooth_n_point,
    mask_region, wind_speed_dir, rotate_winds, convert_units,
    smooth_window, smooth_circular,
};

// ── NEXRAD ───────────────────────────────────────────────────────────
pub use wx_radar::level2::Level2File;
pub use wx_radar::products::RadarProduct;
pub use wx_radar::sites;

// ── Level-III (NIDS) ────────────────────────────────────────────────
pub mod level3;
pub use level3::Level3File;

// ── METAR ───────────────────────────────────────────────────────────
pub mod metar;
pub use metar::{Metar, parse_metar_file};

// ── Station lookup ──────────────────────────────────────────────────
pub mod station;
pub use station::{StationInfo, StationLookup};

// ── GEMPAK grid files ──────────────────────────────────────────────
pub mod gempak;
pub use gempak::GempakGrid;

// ── GEMPAK shared DM format infrastructure ──────────────────────────
pub mod gempak_dm;

// ── GEMPAK sounding files ─────────────────────────────────────────
pub mod gempak_sounding;
pub use gempak_sounding::{GempakSounding, GempakStation as GempakSoundingStation, SoundingData};

// ── GEMPAK surface files ──────────────────────────────────────────
pub mod gempak_surface;
pub use gempak_surface::{GempakSurface, SurfaceStation, SurfaceObs};

// ── GINI satellite images ──────────────────────────────────────────
pub mod gini;
pub use gini::GiniFile;

// ── WPC coded surface bulletins ────────────────────────────────────
pub mod wpc;
pub use wpc::{parse_wpc_surface_bulletin, SurfaceBulletinFeature};

// ── NEXRAD VCP helpers ─────────────────────────────────────────────

/// Return `true` if the given NEXRAD Volume Coverage Pattern (VCP)
/// number corresponds to a precipitation (storm) scanning mode.
///
/// Precipitation VCPs use shorter update cycles and more tilts in the
/// lower atmosphere to better resolve convective features.
pub fn is_precip_mode(vcp: u16) -> bool {
    matches!(vcp, 11 | 12 | 21 | 121 | 211 | 212 | 215 | 221)
}

// ── Download ─────────────────────────────────────────────────────────
pub use wx_core::download::{
    DownloadClient, DownloadConfig, Cache, DiskCache,
};
pub use wx_core::download::{
    IdxEntry, parse_idx, find_entries, find_entries_regex,
    find_entries_criteria, SearchCriteria, byte_ranges,
};
pub use wx_core::download::{fetch_with_fallback, probe_sources, FetchResult};
pub use wx_core::download::{fetch_streaming, fetch_streaming_full};
pub use wx_core::download::{
    DataSource as DownloadSource, model_sources, model_sources_filtered, source_names,
};
pub use wx_core::download::{
    VariableGroup, variable_groups, expand_var_group, expand_vars, group_names, get_group,
};
