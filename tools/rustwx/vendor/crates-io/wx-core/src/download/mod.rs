pub mod idx;
pub mod client;
pub mod cache;
pub mod streaming;
pub mod sources;
pub mod fallback;
pub mod catalog;

pub use idx::{IdxEntry, parse_idx, find_entries, find_entries_regex, find_entries_criteria, SearchCriteria, byte_ranges};
#[cfg(feature = "network")]
pub use idx::available_fhours;
pub use client::{DownloadClient, DownloadConfig};
pub use cache::{Cache, DiskCache};
pub use streaming::{fetch_streaming, fetch_streaming_full};
pub use sources::{DataSource, model_sources, model_sources_filtered, source_names};
pub use fallback::{fetch_with_fallback, probe_sources, FetchResult};
pub use catalog::{VariableGroup, variable_groups, expand_var_group, expand_vars, group_names, get_group};
