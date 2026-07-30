pub mod cache;
pub mod catalog;
pub mod client;
pub mod fallback;
pub mod idx;
pub mod sources;
pub mod streaming;

pub use cache::{Cache, DiskCache};
pub use catalog::{
    expand_var_group, expand_vars, get_group, group_names, variable_groups, VariableGroup,
};
pub use client::{DownloadClient, DownloadConfig};
pub use fallback::{fetch_with_fallback, probe_sources, FetchResult};
#[cfg(feature = "network")]
pub use idx::available_fhours;
pub use idx::{
    byte_ranges, find_entries, find_entries_criteria, find_entries_regex, parse_idx, IdxEntry,
    SearchCriteria,
};
pub use sources::{model_sources, model_sources_filtered, source_names, DataSource};
pub use streaming::{fetch_streaming, fetch_streaming_full};
