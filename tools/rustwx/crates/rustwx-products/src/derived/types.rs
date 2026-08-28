use std::path::PathBuf;

use rustwx_core::{ModelId, SourceId};
use rustwx_render::{PngCompressionMode, RenderImageTiming, RenderStateTiming};
use serde::{Deserialize, Serialize};

use crate::gridded::SharedTiming as GenericSharedTiming;
use crate::heavy::HeavyComputeTiming;
use crate::places::PlaceLabelOverlay;
use crate::publication::{ArtifactContentIdentity, PublishedFetchIdentity};
use crate::shared_context::{DomainSpec, TitleProvenance};
use crate::source::{ProductSourceMode, ProductSourceRoute};
use crate::thermo_native::NativeSemantics;

pub(crate) const OUTPUT_WIDTH: u32 = 1200;
pub(crate) const OUTPUT_HEIGHT: u32 = 900;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeContourRenderMode {
    #[default]
    Automatic,
    Signature,
    LegacyRaster,
    ExperimentalAllProjected,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DerivedBatchRequest {
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
    pub surface_product_override: Option<String>,
    pub pressure_product_override: Option<String>,
    #[serde(default)]
    pub source_mode: ProductSourceMode,
    #[serde(default)]
    pub allow_large_heavy_domain: bool,
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
    /// Optional exact-time filename discriminator used by store-backed
    /// rendering when an ordinal slot is not a whole forecast hour.
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
pub struct HrrrDerivedBatchRequest {
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
    pub source_mode: ProductSourceMode,
    #[serde(default)]
    pub allow_large_heavy_domain: bool,
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
pub struct DerivedSharedTiming {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fetch_decode: Option<GenericSharedTiming>,
    pub compute_ms: u128,
    pub project_ms: u128,
    #[serde(default)]
    pub native_extract_ms: u128,
    #[serde(default)]
    pub native_compare_ms: u128,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_profile: Option<DerivedMemoryProfile>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub heavy_timing: Option<HeavyComputeTiming>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DerivedMemoryProfile {
    pub source_grid_nx: usize,
    pub source_grid_ny: usize,
    pub cropped_grid_nx: usize,
    pub cropped_grid_ny: usize,
    pub crop_x_start: usize,
    pub crop_x_end: usize,
    pub crop_y_start: usize,
    pub crop_y_end: usize,
    pub surface_fetch_bytes_len: usize,
    pub pressure_fetch_bytes_len: usize,
    pub cropped_surface_decoded_bytes_estimate: usize,
    pub cropped_pressure_decoded_bytes_estimate: usize,
    pub cropped_decoded_total_bytes_estimate: usize,
    pub pressure_level_count: usize,
    pub thermo_volume_points: usize,
    pub compute_recipe_count: usize,
    pub needs_volume: bool,
    pub needs_height_agl: bool,
    pub canonical_pressure_3d_pa_bytes_estimate: usize,
    pub canonical_height_agl_3d_bytes_estimate: usize,
    pub canonical_shared_volume_work_bytes_estimate: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DerivedRecipeTiming {
    #[serde(default)]
    pub render_to_image_ms: u128,
    #[serde(default)]
    pub data_layer_draw_ms: u128,
    #[serde(default)]
    pub overlay_draw_ms: u128,
    pub render_state_prep_ms: u128,
    pub png_encode_ms: u128,
    pub file_write_ms: u128,
    pub render_ms: u128,
    pub total_ms: u128,
    pub state_timing: RenderStateTiming,
    pub image_timing: RenderImageTiming,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DerivedRenderedRecipe {
    pub recipe_slug: String,
    pub title: String,
    pub source_route: ProductSourceRoute,
    pub output_path: PathBuf,
    pub content_identity: ArtifactContentIdentity,
    pub input_fetch_keys: Vec<String>,
    /// What the finished PNG maps to on the Earth, when the save path
    /// could publish it (gpuwm addition, VENDOR.md).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub georeference: Option<rustwx_render::PanelGeoReference>,
    /// Why `georeference` is `None`, from the save path itself; a panel
    /// listed without a transform must say what was missing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub georeference_absent_reason: Option<String>,
    pub timing: DerivedRecipeTiming,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DerivedRecipeBlocker {
    pub recipe_slug: String,
    pub source_route: ProductSourceRoute,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NativeThermoArtifactReport {
    pub recipe_slug: String,
    pub source_route: ProductSourceRoute,
    pub semantics: NativeSemantics,
    pub auto_eligible: bool,
    pub native_label: String,
    pub native_detail: String,
    pub native_fetch_product: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DerivedBatchReport {
    pub model: ModelId,
    pub date_yyyymmdd: String,
    pub cycle_utc: u8,
    pub forecast_hour: u16,
    pub source: SourceId,
    pub domain: DomainSpec,
    pub input_fetches: Vec<PublishedFetchIdentity>,
    pub shared_timing: DerivedSharedTiming,
    pub recipes: Vec<DerivedRenderedRecipe>,
    #[serde(default)]
    pub source_mode: ProductSourceMode,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub blockers: Vec<DerivedRecipeBlocker>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub native_thermo_artifacts: Vec<NativeThermoArtifactReport>,
    pub total_ms: u128,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HrrrDerivedBatchReport {
    pub date_yyyymmdd: String,
    pub cycle_utc: u8,
    pub forecast_hour: u16,
    pub source: SourceId,
    pub domain: DomainSpec,
    pub input_fetches: Vec<PublishedFetchIdentity>,
    pub shared_timing: DerivedSharedTiming,
    pub recipes: Vec<DerivedRenderedRecipe>,
    #[serde(default)]
    pub source_mode: ProductSourceMode,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub blockers: Vec<DerivedRecipeBlocker>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub native_thermo_artifacts: Vec<NativeThermoArtifactReport>,
    pub total_ms: u128,
}

pub type HrrrDerivedSharedTiming = DerivedSharedTiming;
pub type HrrrDerivedRecipeTiming = DerivedRecipeTiming;
pub type HrrrDerivedRenderedRecipe = DerivedRenderedRecipe;
