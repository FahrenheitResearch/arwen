//! Data I/O module -- MetPy-compatible structure for reading weather data.
//!
//! Re-exports GRIB2, NEXRAD Level-II, and data download primitives from
//! wx-core and wx-radar.  Also provides native parsers for Level-III
//! products, METAR text reports, and station lookup.

// ── GRIB2 ────────────────────────────────────────────────────────────
pub use wx_core::grib2::grid::{grid_latlon, rotated_to_geographic};
pub use wx_core::grib2::tables::{level_name, parameter_name, parameter_units};
pub use wx_core::grib2::{
    apply_op, convert_units, field_diff, field_stats, field_stats_region, filter, mask_region,
    merge, rotate_winds, smooth_circular, smooth_gaussian, smooth_n_point, smooth_window, split,
    subset, wind_speed_dir, FieldOp, FieldStats,
};
pub use wx_core::grib2::{flip_rows, unpack_message, unpack_message_normalized, BitReader};
pub use wx_core::grib2::{search_messages, StreamingParser};
pub use wx_core::grib2::{
    DataRepresentation, Grib2File, Grib2Message, GridDefinition, ProductDefinition,
};
pub use wx_core::grib2::{Grib2Writer, MessageBuilder, PackingMethod};

// ── NEXRAD ───────────────────────────────────────────────────────────
pub use wx_radar::level2::Level2File;
pub use wx_radar::products::RadarProduct;
pub use wx_radar::sites;

// ── Level-III (NIDS) ────────────────────────────────────────────────
pub mod level3;
pub use level3::Level3File;

// ── METAR ───────────────────────────────────────────────────────────
pub mod metar;
pub use metar::{parse_metar_file, Metar};

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
pub use gempak_surface::{GempakSurface, SurfaceObs, SurfaceStation};

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
    byte_ranges, find_entries, find_entries_criteria, find_entries_regex, parse_idx, IdxEntry,
    SearchCriteria,
};
pub use wx_core::download::{
    expand_var_group, expand_vars, get_group, group_names, variable_groups, VariableGroup,
};
pub use wx_core::download::{fetch_streaming, fetch_streaming_full};
pub use wx_core::download::{fetch_with_fallback, probe_sources, FetchResult};
pub use wx_core::download::{
    model_sources, model_sources_filtered, source_names, DataSource as DownloadSource,
};
pub use wx_core::download::{Cache, DiskCache, DownloadClient, DownloadConfig};
