use crate::derived::HrrrDerivedBatchReport;
use crate::direct::HrrrDirectBatchReport;
use crate::hrrr::DomainSpec;
use crate::places::PlaceLabelOverlay;
use crate::source::{ProductSourceMode, ProductSourceRoute};
use crate::windowed::{HrrrWindowedBatchReport, HrrrWindowedProduct};
use rustwx_core::{ModelId, SourceId};
use rustwx_render::PngCompressionMode;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
fn default_output_width() -> u32 {
    1200
}

fn default_output_height() -> u32 {
    900
}

fn default_png_compression() -> PngCompressionMode {
    PngCompressionMode::Default
}

pub(super) fn non_ecape_derived_contour_mode() -> crate::derived::NativeContourRenderMode {
    crate::derived::NativeContourRenderMode::Automatic
}

pub(super) fn non_ecape_native_fill_level_multiplier() -> usize {
    1
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HrrrNonEcapeHourRequest {
    pub date_yyyymmdd: String,
    pub cycle_override_utc: Option<u8>,
    pub forecast_hour: u16,
    pub source: SourceId,
    pub domain: DomainSpec,
    pub out_dir: PathBuf,
    pub cache_root: PathBuf,
    pub use_cache: bool,
    #[serde(default)]
    pub source_mode: ProductSourceMode,
    pub direct_recipe_slugs: Vec<String>,
    pub derived_recipe_slugs: Vec<String>,
    pub windowed_products: Vec<HrrrWindowedProduct>,
    #[serde(default = "default_output_width")]
    pub output_width: u32,
    #[serde(default = "default_output_height")]
    pub output_height: u32,
    #[serde(default = "default_png_compression")]
    pub png_compression: PngCompressionMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub place_label_overlay: Option<PlaceLabelOverlay>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HrrrNonEcapeMultiDomainRequest {
    pub date_yyyymmdd: String,
    pub cycle_override_utc: Option<u8>,
    pub forecast_hour: u16,
    pub source: SourceId,
    pub domains: Vec<DomainSpec>,
    pub out_dir: PathBuf,
    pub cache_root: PathBuf,
    pub use_cache: bool,
    #[serde(default)]
    pub source_mode: ProductSourceMode,
    pub direct_recipe_slugs: Vec<String>,
    pub derived_recipe_slugs: Vec<String>,
    pub windowed_products: Vec<HrrrWindowedProduct>,
    #[serde(default = "default_output_width")]
    pub output_width: u32,
    #[serde(default = "default_output_height")]
    pub output_height: u32,
    #[serde(default = "default_png_compression")]
    pub png_compression: PngCompressionMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub place_label_overlay: Option<PlaceLabelOverlay>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub domain_jobs: Option<usize>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HrrrNonEcapeSharedTiming {
    pub resolve_run_ms: u128,
    pub shared_load_decode_ms: u128,
    #[serde(default)]
    pub shared_fetch_ms_total: u128,
    #[serde(default)]
    pub shared_decode_surface_ms_total: u128,
    #[serde(default)]
    pub shared_decode_pressure_ms_total: u128,
    #[serde(default)]
    pub shared_fetched_bundle_count: usize,
    #[serde(default)]
    pub shared_surface_decode_count: usize,
    #[serde(default)]
    pub shared_pressure_decode_count: usize,
    #[serde(default)]
    pub shared_direct_prepare_ms: u128,
    pub shared_derived_prepare_ms: u128,
    #[serde(default)]
    pub shared_windowed_prepare_ms: u128,
    pub total_prepare_ms: u128,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HrrrNonEcapeFanoutTiming {
    pub domain_context_build_ms: u128,
    pub domain_fanout_wall_ms: u128,
    pub domain_render_sum_ms: u128,
    pub domain_render_max_ms: u128,
    pub conus_wall_ms: u128,
    pub city_domains_sum_ms: u128,
    pub city_domains_max_ms: u128,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HrrrNonEcapeHourRequestedProducts {
    pub direct_recipe_slugs: Vec<String>,
    pub derived_recipe_slugs: Vec<String>,
    pub windowed_products: Vec<HrrrWindowedProduct>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HrrrNonEcapeHourSummary {
    pub runner_count: usize,
    pub direct_rendered_count: usize,
    pub derived_rendered_count: usize,
    pub windowed_rendered_count: usize,
    pub windowed_blocker_count: usize,
    pub output_count: usize,
    pub output_paths: Vec<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HrrrNonEcapeHourReport {
    pub date_yyyymmdd: String,
    pub cycle_utc: u8,
    pub forecast_hour: u16,
    pub source: SourceId,
    pub domain: DomainSpec,
    pub out_dir: PathBuf,
    pub cache_root: PathBuf,
    pub use_cache: bool,
    #[serde(default)]
    pub source_mode: ProductSourceMode,
    /// Canonical (latest-attempt) run manifest path — stable across
    /// reruns and therefore clobberable.
    pub publication_manifest_path: PathBuf,
    /// Immutable attempt-stamped sibling manifest path. Always present
    /// on completed runs; paired with [`publication_manifest_path`] it
    /// forms the `(current truth, immutable attempt)` contract.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attempt_manifest_path: Option<PathBuf>,
    pub requested: HrrrNonEcapeHourRequestedProducts,
    #[serde(default)]
    pub shared_timing: HrrrNonEcapeSharedTiming,
    pub summary: HrrrNonEcapeHourSummary,
    pub direct: Option<HrrrDirectBatchReport>,
    pub derived: Option<HrrrDerivedBatchReport>,
    pub windowed: Option<HrrrWindowedBatchReport>,
    pub total_ms: u128,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HrrrNonEcapeDomainReport {
    pub domain: DomainSpec,
    pub publication_manifest_path: PathBuf,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attempt_manifest_path: Option<PathBuf>,
    pub summary: HrrrNonEcapeHourSummary,
    pub direct: Option<HrrrDirectBatchReport>,
    pub derived: Option<HrrrDerivedBatchReport>,
    pub windowed: Option<HrrrWindowedBatchReport>,
    pub total_ms: u128,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HrrrNonEcapeMultiDomainReport {
    pub date_yyyymmdd: String,
    pub cycle_utc: u8,
    pub forecast_hour: u16,
    pub source: SourceId,
    pub out_dir: PathBuf,
    pub cache_root: PathBuf,
    pub use_cache: bool,
    #[serde(default)]
    pub source_mode: ProductSourceMode,
    pub requested: HrrrNonEcapeHourRequestedProducts,
    #[serde(default)]
    pub shared_timing: HrrrNonEcapeSharedTiming,
    #[serde(default)]
    pub fanout_timing: HrrrNonEcapeFanoutTiming,
    pub domains: Vec<HrrrNonEcapeDomainReport>,
    pub total_ms: u128,
}

pub type NonEcapeSharedTiming = HrrrNonEcapeSharedTiming;
pub type NonEcapeFanoutTiming = HrrrNonEcapeFanoutTiming;
pub type NonEcapeRequestedProducts = HrrrNonEcapeHourRequestedProducts;
pub type NonEcapeHourSummary = HrrrNonEcapeHourSummary;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NonEcapeHourRequest {
    pub model: ModelId,
    pub date_yyyymmdd: String,
    pub cycle_override_utc: Option<u8>,
    pub forecast_hour: u16,
    pub source: SourceId,
    pub domain: DomainSpec,
    pub out_dir: PathBuf,
    pub cache_root: PathBuf,
    pub use_cache: bool,
    #[serde(default)]
    pub source_mode: ProductSourceMode,
    pub direct_recipe_slugs: Vec<String>,
    pub derived_recipe_slugs: Vec<String>,
    #[serde(default)]
    pub direct_product_overrides: HashMap<String, String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub surface_product_override: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pressure_product_override: Option<String>,
    #[serde(default)]
    pub allow_large_heavy_domain: bool,
    #[serde(default)]
    pub windowed_products: Vec<HrrrWindowedProduct>,
    #[serde(default = "default_output_width")]
    pub output_width: u32,
    #[serde(default = "default_output_height")]
    pub output_height: u32,
    #[serde(default = "default_png_compression")]
    pub png_compression: PngCompressionMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub place_label_overlay: Option<PlaceLabelOverlay>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NonEcapeMultiDomainRequest {
    pub model: ModelId,
    pub date_yyyymmdd: String,
    pub cycle_override_utc: Option<u8>,
    pub forecast_hour: u16,
    pub source: SourceId,
    pub domains: Vec<DomainSpec>,
    pub out_dir: PathBuf,
    pub cache_root: PathBuf,
    pub use_cache: bool,
    #[serde(default)]
    pub source_mode: ProductSourceMode,
    pub direct_recipe_slugs: Vec<String>,
    pub derived_recipe_slugs: Vec<String>,
    #[serde(default)]
    pub direct_product_overrides: HashMap<String, String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub surface_product_override: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pressure_product_override: Option<String>,
    #[serde(default)]
    pub allow_large_heavy_domain: bool,
    #[serde(default)]
    pub windowed_products: Vec<HrrrWindowedProduct>,
    #[serde(default = "default_output_width")]
    pub output_width: u32,
    #[serde(default = "default_output_height")]
    pub output_height: u32,
    #[serde(default = "default_png_compression")]
    pub png_compression: PngCompressionMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub place_label_overlay: Option<PlaceLabelOverlay>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub domain_jobs: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NonEcapeHourReport {
    pub model: ModelId,
    pub date_yyyymmdd: String,
    pub cycle_utc: u8,
    pub forecast_hour: u16,
    pub source: SourceId,
    pub domain: DomainSpec,
    pub out_dir: PathBuf,
    pub cache_root: PathBuf,
    pub use_cache: bool,
    #[serde(default)]
    pub source_mode: ProductSourceMode,
    pub publication_manifest_path: PathBuf,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attempt_manifest_path: Option<PathBuf>,
    pub requested: NonEcapeRequestedProducts,
    #[serde(default)]
    pub shared_timing: NonEcapeSharedTiming,
    pub summary: NonEcapeHourSummary,
    pub direct: Option<HrrrDirectBatchReport>,
    pub derived: Option<HrrrDerivedBatchReport>,
    pub windowed: Option<HrrrWindowedBatchReport>,
    pub total_ms: u128,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NonEcapeDomainReport {
    pub domain: DomainSpec,
    pub publication_manifest_path: PathBuf,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attempt_manifest_path: Option<PathBuf>,
    pub summary: NonEcapeHourSummary,
    pub direct: Option<HrrrDirectBatchReport>,
    pub derived: Option<HrrrDerivedBatchReport>,
    pub windowed: Option<HrrrWindowedBatchReport>,
    pub total_ms: u128,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NonEcapeMultiDomainReport {
    pub model: ModelId,
    pub date_yyyymmdd: String,
    pub cycle_utc: u8,
    pub forecast_hour: u16,
    pub source: SourceId,
    pub out_dir: PathBuf,
    pub cache_root: PathBuf,
    pub use_cache: bool,
    #[serde(default)]
    pub source_mode: ProductSourceMode,
    pub requested: NonEcapeRequestedProducts,
    #[serde(default)]
    pub shared_timing: NonEcapeSharedTiming,
    #[serde(default)]
    pub fanout_timing: NonEcapeFanoutTiming,
    pub domains: Vec<NonEcapeDomainReport>,
    pub total_ms: u128,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NonEcapeBuildDomainTiming {
    pub domain_slug: String,
    pub total_ms: u128,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub direct_total_ms: Option<u128>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub derived_total_ms: Option<u128>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub windowed_total_ms: Option<u128>,
    pub output_count: usize,
    pub direct_count: usize,
    pub derived_count: usize,
    pub windowed_count: usize,
    pub windowed_blocker_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NonEcapeBuildProductTiming {
    pub domain_slug: String,
    pub lane: String,
    pub product_slug: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    pub output_path: PathBuf,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_route: Option<ProductSourceRoute>,
    pub render_ms: u128,
    pub total_ms: u128,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub render_to_image_ms: Option<u128>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data_layer_draw_ms: Option<u128>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub overlay_draw_ms: Option<u128>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub png_encode_ms: Option<u128>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_write_ms: Option<u128>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NonEcapeHourBuildReport {
    pub static_report: NonEcapeMultiDomainReport,
    #[serde(default)]
    pub static_domain_timings: Vec<NonEcapeBuildDomainTiming>,
    #[serde(default)]
    pub static_product_timings: Vec<NonEcapeBuildProductTiming>,
    pub total_ms: u128,
}
