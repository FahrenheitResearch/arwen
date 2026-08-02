use std::collections::HashMap;
use std::path::PathBuf;

use rustwx_core::{FieldSelector, ModelId, SelectedField2D, SourceId};
use rustwx_models::{LatestRun, PlotRecipeFetchMode};
use rustwx_render::{PngCompressionMode, RenderImageTiming, RenderStateTiming};
use serde::{Deserialize, Serialize};

use crate::derived::NativeContourRenderMode;
use crate::places::PlaceLabelOverlay;
use crate::publication::{ArtifactContentIdentity, PublishedFetchIdentity};
use crate::shared_context::{DomainSpec, TitleProvenance};
use crate::source::ProductSourceRoute;

use super::planning::PlannedDirectRecipe;

pub(super) const OUTPUT_WIDTH: u32 = 1600;
pub(super) const OUTPUT_HEIGHT: u32 = 900;
pub(super) const CLOUD_LEVEL_COMPONENT_SLUGS: &[&str] =
    &["low_cloud_cover", "middle_cloud_cover", "high_cloud_cover"];
pub(super) const PRECIPITATION_TYPE_COMPONENT_SLUGS: &[&str] = &[
    "categorical_rain",
    "categorical_freezing_rain",
    "categorical_ice_pellets",
    "categorical_snow",
];

fn default_output_width() -> u32 {
    OUTPUT_WIDTH
}

fn default_output_height() -> u32 {
    OUTPUT_HEIGHT
}

fn default_png_compression() -> PngCompressionMode {
    PngCompressionMode::Default
}

fn default_native_fill_level_multiplier() -> usize {
    1
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectBatchRequest {
    pub model: ModelId,
    pub date_yyyymmdd: String,
    pub cycle_override_utc: Option<u8>,
    pub forecast_hour: u16,
    pub source: SourceId,
    pub domain: DomainSpec,
    pub out_dir: PathBuf,
    pub cache_root: PathBuf,
    pub use_cache: bool,
    pub recipe_slugs: Vec<String>,
    pub product_overrides: HashMap<String, String>,
    #[serde(default)]
    pub contour_mode: NativeContourRenderMode,
    #[serde(default = "default_native_fill_level_multiplier")]
    pub native_fill_level_multiplier: usize,
    #[serde(default = "default_output_width")]
    pub output_width: u32,
    #[serde(default = "default_output_height")]
    pub output_height: u32,
    #[serde(default = "default_png_compression")]
    pub png_compression: PngCompressionMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub place_label_overlay: Option<PlaceLabelOverlay>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_suffix: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subtitle_left_override: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subtitle_right_override: Option<String>,
    /// Where these frames came from, for the headline parenthetical.
    #[serde(default)]
    pub title_provenance: TitleProvenance,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HrrrDirectBatchRequest {
    pub date_yyyymmdd: String,
    pub cycle_override_utc: Option<u8>,
    pub forecast_hour: u16,
    pub source: SourceId,
    pub domain: DomainSpec,
    pub out_dir: PathBuf,
    pub cache_root: PathBuf,
    pub use_cache: bool,
    pub recipe_slugs: Vec<String>,
    #[serde(default)]
    pub contour_mode: NativeContourRenderMode,
    #[serde(default = "default_native_fill_level_multiplier")]
    pub native_fill_level_multiplier: usize,
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
pub struct DirectFetchRuntimeInfo {
    pub fetch_key: String,
    /// Canonical physical family name that was actually fetched.
    ///
    /// Kept equal to `fetched_product` for backward compatibility with existing
    /// manifest consumers. Logical families that contributed to this canonical
    /// fetch are surfaced separately in `planned_family_aliases`.
    pub planned_product: String,
    pub fetched_product: String,
    /// Sorted de-duplicated logical planned-family names before
    /// canonicalization.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub planned_family_aliases: Vec<String>,
    pub requested_source: SourceId,
    pub resolved_source: SourceId,
    pub resolved_url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectRecipeTiming {
    pub project_ms: u128,
    #[serde(default)]
    pub field_prepare_ms: u128,
    #[serde(default)]
    pub contour_prepare_ms: u128,
    #[serde(default)]
    pub barb_prepare_ms: u128,
    #[serde(default)]
    pub render_to_image_ms: u128,
    #[serde(default)]
    pub data_layer_draw_ms: u128,
    #[serde(default)]
    pub overlay_draw_ms: u128,
    #[serde(default)]
    pub panel_compose_ms: u128,
    pub request_build_ms: u128,
    pub render_state_prep_ms: u128,
    pub png_encode_ms: u128,
    pub file_write_ms: u128,
    pub render_ms: u128,
    pub total_ms: u128,
    pub state_timing: RenderStateTiming,
    pub image_timing: RenderImageTiming,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectFetchTiming {
    pub product: String,
    pub fetch_mode: PlotRecipeFetchMode,
    pub fetch_ms: u128,
    pub parse_ms: u128,
    pub extract_ms: u128,
    pub total_ms: u128,
    pub fetch_cache_hit: bool,
    pub extract_cache_hits: usize,
    pub extract_cache_misses: usize,
    pub runtime_fetch: DirectFetchRuntimeInfo,
    pub input_fetch: PublishedFetchIdentity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectRenderedRecipe {
    pub recipe_slug: String,
    pub title: String,
    pub source_route: ProductSourceRoute,
    pub grib_product: String,
    pub fetched_grib_product: String,
    pub resolved_source: SourceId,
    pub resolved_url: String,
    pub output_path: PathBuf,
    pub content_identity: ArtifactContentIdentity,
    pub input_fetch_keys: Vec<String>,
    pub timing: DirectRecipeTiming,
}

/// Per-recipe failure that does not abort the whole batch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectRecipeBlocker {
    pub recipe_slug: String,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectBatchReport {
    pub model: ModelId,
    pub date_yyyymmdd: String,
    pub cycle_utc: u8,
    pub forecast_hour: u16,
    pub source: SourceId,
    pub domain: DomainSpec,
    pub fetches: Vec<DirectFetchTiming>,
    pub recipes: Vec<DirectRenderedRecipe>,
    /// Recipes that could not render. Populated instead of short-circuiting the
    /// batch, so orchestration callers get per-recipe signal rather than one
    /// hard error on the first problem.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub blockers: Vec<DirectRecipeBlocker>,
    pub total_ms: u128,
}

pub type HrrrDirectFetchRuntimeInfo = DirectFetchRuntimeInfo;
pub type HrrrDirectRecipeTiming = DirectRecipeTiming;
pub type HrrrDirectFetchTiming = DirectFetchTiming;
pub type HrrrDirectRenderedRecipe = DirectRenderedRecipe;
pub type HrrrDirectRecipeBlocker = DirectRecipeBlocker;
pub type HrrrDirectBatchReport = DirectBatchReport;

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct DirectRequestBuildTiming {
    pub(super) field_prepare_ms: u128,
    pub(super) contour_prepare_ms: u128,
    pub(super) barb_prepare_ms: u128,
}

#[derive(Debug, Clone)]
pub(crate) struct DirectSampledProductField {
    pub recipe_slug: String,
    pub source_route: ProductSourceRoute,
    pub field_selector: Option<FieldSelector>,
    pub field: rustwx_core::Field2D,
    pub input_fetches: Vec<PublishedFetchIdentity>,
}

#[derive(Debug, Clone)]
pub(crate) struct DirectSampledProductSet {
    pub fields: Vec<DirectSampledProductField>,
    pub blockers: Vec<DirectRecipeBlocker>,
}

#[derive(Debug, Clone)]
pub(crate) struct PreparedDirectBatch {
    pub(super) latest: LatestRun,
    pub(super) renderable: Vec<PlannedDirectRecipe>,
    pub(super) extracted: HashMap<FieldSelector, SelectedField2D>,
    pub(super) fetches: Vec<DirectFetchTiming>,
    pub(super) fetch_truth_by_actual_product: HashMap<String, DirectFetchRuntimeInfo>,
    pub(super) blockers: Vec<DirectRecipeBlocker>,
}
