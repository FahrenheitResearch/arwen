#![allow(dead_code)]

//! Store -> render glue shared by the render-side bins (`rw_render`, and
//! Task 6's `rw_batch`): open one stored hour, resolve plot-recipe
//! `FieldSelector`s and derived recipe slugs against the stored variable
//! metadata (built from each variable's selector JSON, not hardcoded
//! tables, so coverage is provable), and feed the EXISTING rustwx-products
//! render paths — `render_direct_recipes_from_selected_fields` for direct
//! recipes and `render_derived_recipes_from_store_grids` for derived/heavy
//! recipes. No render logic lives here; this module only reads fields and
//! reports which requested products cannot resolve against the store.
//!
//! Reads are full-field (`read_full_2d` via `read_field_2d`/`read_grid_2d`,
//! ~3.6 ms per field): the direct render path crops in render space from
//! the full grid (`crop_direct_fields_for_domain` inside the products
//! crate) and the derived path crops values after the projected-domain
//! classification, so a store-side windowed read would change the data the
//! proven render paths see. Windowed reads stay a later optimization for
//! when a render lane learns to consume pre-windowed grids.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use rustwx_core::{Field2D, FieldSelector, ProductKey, SelectedField2D};
use rustwx_models::{LatestRun, plot_recipe_store_requirements};
use rustwx_products::derived::{
    DerivedBatchRequest, DerivedRenderedRecipe, StoreProductGrid,
    render_derived_recipes_from_store_grids,
};
use rustwx_products::direct::{
    DirectBatchRequest, DirectRenderedRecipe, build_projected_map_with_projection,
    direct_map_frame_aspect_ratio, direct_render_chunk_size,
    render_direct_recipes_chunked_from_loader,
};
use rustwx_products::plot_design::StaticPlotDesign;
use rustwx_products::viewer::{
    curated_style_for_store_variable, generic_style_for_store_variable,
    operational_style_for_store_variable,
};
use rustwx_render::{
    MapRenderRequest, PngWriteOptions, ProductVisualMode, ProjectedDomain,
    save_png_profile_with_options,
};
use rw_store::RwsExactTime;
use rw_store::error::RwStoreError;
use rw_store::format::RwsVariableMeta;
use rw_store::grid::GridFile;
use rw_store::ingest::{StoredField2D, derived_selector_slug, read_field_2d, read_grid_2d};
use rw_store::reader::HourReader;
use rw_store::run::{RwsRunManifest, validate_store_component};

fn canonical_contained_path(
    run_dir: &Path,
    path: &Path,
    label: &str,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let canonical = std::fs::canonicalize(path)
        .map_err(|error| format!("cannot resolve {label} {}: {error}", path.display()))?;
    if !canonical.starts_with(run_dir) {
        return Err(format!(
            "{label} {} resolves outside run directory {}",
            path.display(),
            run_dir.display()
        )
        .into());
    }
    Ok(canonical)
}

/// One stored hour opened for rendering: the hour reader, the run grid,
/// and the resolution maps built from the stored variable metadata.
pub struct StoreFieldSource {
    hour_path: PathBuf,
    reader: HourReader,
    grid: GridFile,
    /// `FieldSelector` -> stored 2D variable name. First write wins on
    /// selector collisions, mirroring GRIB file-order semantics: the sfc
    /// file's two APCP accumulations share the plain TotalPrecipitation
    /// selector, and the run total (`apcp_run_total`, stored first) is the
    /// one the direct lane's extraction would have picked.
    selector_vars: HashMap<FieldSelector, String>,
    /// Derived/heavy variable slugs present in this hour (store order),
    /// keyed from each variable's `{"derived": slug}` selector marker.
    derived_slugs: Vec<String>,
    /// The COMPLETE `surface2d` inventory in stable file order, including
    /// variables whose selector is neither a canonical `FieldSelector` nor
    /// a derived marker.  The generic render lane serves these; the
    /// canonical maps above deliberately never learn about them.
    surface_variables: Vec<RwsVariableMeta>,
    /// Physical timing carried by an exact-time v2 store. `None` for the
    /// legacy whole-hour v1 axis.
    exact_time: Option<RwsExactTime>,
}

impl StoreFieldSource {
    /// Open one manifest entry plus the run's `grid.rwg` and build the
    /// selector/derived resolution maps. For v2 stores, `hour` is the ordinal
    /// storage slot and [`Self::exact_time`] exposes its physical timing.
    pub fn open(
        store_root: &Path,
        model_slug: &str,
        run_slug: &str,
        hour: u16,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        validate_store_component("model", model_slug)?;
        validate_store_component("run", run_slug)?;
        let root = std::fs::canonicalize(store_root).map_err(|error| {
            format!(
                "cannot resolve store root {}: {error}",
                store_root.display()
            )
        })?;
        let requested_run = root.join(model_slug).join(run_slug);
        let run_dir = std::fs::canonicalize(&requested_run).map_err(|error| {
            format!(
                "cannot resolve run directory {}: {error}",
                requested_run.display()
            )
        })?;
        if !run_dir.starts_with(&root) {
            return Err(format!(
                "run directory {} resolves outside store root {}",
                requested_run.display(),
                root.display()
            )
            .into());
        }
        let manifest_path =
            canonical_contained_path(&run_dir, &run_dir.join("run.json"), "run manifest")?;
        let manifest = RwsRunManifest::load_for_run(&manifest_path, model_slug, run_slug)?;
        let entry = manifest.hours.get(&hour).ok_or_else(|| {
            RwStoreError::Meta(format!(
                "forecast hour f{hour:03} is absent from {model_slug}/{run_slug}/run.json"
            ))
        })?;
        let exact_time = entry.exact_time();
        let hour_path = canonical_contained_path(
            &run_dir,
            &run_dir.join(&entry.file),
            &format!("forecast hour f{hour:03}"),
        )?;
        let reader = HourReader::open(&hour_path)
            .map_err(|err| format!("open {}: {err}", hour_path.display()))?;
        let grid_path = canonical_contained_path(&run_dir, &run_dir.join("grid.rwg"), "grid file")?;
        let grid = GridFile::open(&grid_path)
            .map_err(|err| format!("open {}: {err}", grid_path.display()))?;
        manifest.validate_grid(&grid.hash, grid.nx, grid.ny)?;
        manifest.validate_hour_meta(hour, reader.meta())?;

        let mut selector_vars = HashMap::new();
        let mut derived_slugs = Vec::new();
        let mut surface_variables = Vec::new();
        for var in &reader.meta().variables {
            if var.kind != "surface2d" {
                continue;
            }
            surface_variables.push(var.clone());
            if let Some(slug) = derived_selector_slug(&var.selector) {
                derived_slugs.push(slug.to_string());
                continue;
            }
            // A selector that is neither a canonical FieldSelector nor a
            // derived marker is NOT fatal: one unknown diagnostic used to
            // abort the whole render hour.  The variable stays in the
            // surface inventory (renderable via the generic lane) and only
            // skips canonical selector indexing.
            if let Ok(selector) = serde_json::from_value::<FieldSelector>(var.selector.clone()) {
                selector_vars
                    .entry(selector)
                    .or_insert_with(|| var.name.clone());
            }
        }

        Ok(Self {
            hour_path,
            reader,
            grid,
            selector_vars,
            derived_slugs,
            surface_variables,
            exact_time,
        })
    }

    pub fn hour_path(&self) -> &Path {
        &self.hour_path
    }

    /// Exact lead/valid time for a v2 store slot.
    pub fn exact_time(&self) -> Option<RwsExactTime> {
        self.exact_time
    }

    /// The stored 2D variable carrying this selector, if any.
    pub fn resolve(&self, selector: &FieldSelector) -> Option<&str> {
        self.selector_vars.get(selector).map(String::as_str)
    }

    /// Read one selector's full field back as the `SelectedField2D` the
    /// render lanes consume (bit-exact f32 round-trip of the extraction).
    pub fn fetch(
        &self,
        selector: &FieldSelector,
    ) -> Result<SelectedField2D, Box<dyn std::error::Error>> {
        let name = self
            .resolve(selector)
            .ok_or_else(|| format!("no stored variable carries selector {}", selector.key()))?;
        Ok(read_field_2d(&self.reader, &self.grid, name)?)
    }

    /// Derived/heavy recipe slugs stored in this hour, in store order.
    pub fn derived_slugs(&self) -> &[String] {
        &self.derived_slugs
    }

    /// All stored 2-D variables in stable file order, including variables
    /// whose selector is intentionally opaque to production recipes.
    pub fn surface_variables(&self) -> &[RwsVariableMeta] {
        &self.surface_variables
    }

    pub fn surface_variable(&self, name: &str) -> Option<&RwsVariableMeta> {
        self.surface_variables.iter().find(|var| var.name == name)
    }

    /// Read an arbitrary stored 2-D plane without pretending it is a known
    /// operational recipe.
    pub fn generic_grid(&self, name: &str) -> Result<StoredField2D, RwStoreError> {
        read_grid_2d(&self.reader, &self.grid, name)
    }

    /// Read one precomputed derived/heavy grid by recipe slug.
    pub fn derived_grid(&self, slug: &str) -> Result<StoredField2D, RwStoreError> {
        read_grid_2d(&self.reader, &self.grid, slug)
    }

    /// The full hour grid (coordinates from `grid.rwg`).
    pub fn full_grid(&self) -> rustwx_core::LatLonGrid {
        rustwx_core::LatLonGrid {
            shape: rustwx_core::GridShape {
                nx: self.grid.nx,
                ny: self.grid.ny,
            },
            lat_deg: self.grid.lat.clone(),
            lon_deg: self.grid.lon.clone(),
        }
    }

    pub fn projection(&self) -> Option<&rustwx_core::GridProjection> {
        self.grid.projection.as_ref()
    }

    /// Borrow the run grid coordinates without cloning the full arrays.
    /// Batch orchestration uses this to derive a native-domain extent once;
    /// the render paths continue to receive their existing owned grid value.
    pub fn grid_coordinates(&self) -> (&[f32], &[f32]) {
        (&self.grid.lat, &self.grid.lon)
    }

    /// Provenance strings for the rendered-recipe reports (never pixels).
    pub fn fetch_key(&self) -> String {
        let meta = self.reader.meta();
        format!(
            "rw-store:{}:{}:f{:03}",
            meta.model, meta.run, meta.forecast_hour
        )
    }
}

/// One requested product that cannot render from this store hour, with the
/// reason (missing selector(s) or an unstored derived slug). The blocker
/// pattern: record why, never invent a substitute.
#[derive(Debug, Clone)]
pub struct StoreRenderSkip {
    pub slug: String,
    pub reason: String,
}

/// Outcome of one direct-lane store render pass.
pub struct DirectStoreOutcome {
    pub rendered: Vec<DirectRenderedRecipe>,
    pub skipped: Vec<StoreRenderSkip>,
}

/// Render the requested direct recipes from stored fields through the
/// direct lane's chunked render entry (the same planning, crop,
/// render-request build, and save path the GRIB lane runs — recipes render
/// in bounded chunks whose fields are loaded from the store on demand, so
/// the full ~2 GB selector set never sits in RAM at once). A recipe is
/// renderable iff every selector in its model-identity-free store
/// requirements resolves in the store; unresolvable recipes are returned
/// as skips naming the missing fields (never a model-identity reason).
pub fn render_direct_recipes_from_store(
    source: &StoreFieldSource,
    request: &DirectBatchRequest,
    latest: &LatestRun,
    recipe_slugs: &[String],
    chunk_gate: Option<&dyn Fn()>,
) -> Result<DirectStoreOutcome, Box<dyn std::error::Error>> {
    let mut renderable = Vec::new();
    let mut skipped = Vec::new();
    for slug in recipe_slugs {
        let requirements = match plot_recipe_store_requirements(slug) {
            Ok(requirements) => requirements,
            Err(err) => {
                skipped.push(StoreRenderSkip {
                    slug: slug.clone(),
                    reason: format!("unknown plot recipe: {err}"),
                });
                continue;
            }
        };
        let missing: Vec<String> = requirements
            .iter()
            .filter_map(|requirement| match requirement.selector {
                Some(selector) => source
                    .resolve(&selector)
                    .is_none()
                    .then(|| selector.key()),
                None => Some(format!(
                    "{} (no canonical store selector exists for this field)",
                    requirement.field_key
                )),
            })
            .collect();
        if missing.is_empty() {
            renderable.push(slug.clone());
        } else {
            skipped.push(StoreRenderSkip {
                slug: slug.clone(),
                reason: format!("missing stored selector(s): {}", missing.join(", ")),
            });
        }
    }
    if renderable.is_empty() {
        return Ok(DirectStoreOutcome {
            rendered: Vec::new(),
            skipped,
        });
    }

    let mut load_field = |selector: &FieldSelector| source.fetch(selector);
    let rendered = render_direct_recipes_chunked_from_loader(
        request,
        latest,
        &renderable,
        &mut load_field,
        direct_render_chunk_size(),
        chunk_gate,
        "rw-store",
        source.hour_path().display().to_string(),
        source.fetch_key(),
    )?;
    Ok(DirectStoreOutcome { rendered, skipped })
}

/// Outcome of one derived/heavy-lane store render pass.
pub struct DerivedStoreOutcome {
    pub rendered: Vec<DerivedRenderedRecipe>,
    pub skipped: Vec<StoreRenderSkip>,
}

/// Render metadata for an arbitrary stored 2-D variable.
pub struct GenericRenderedVariable {
    pub variable: String,
    pub input_units: String,
    pub display_units: String,
    pub output_path: PathBuf,
    pub total_ms: u128,
}

/// Render an arbitrary `surface2d` variable through the same native map,
/// basemap, static design, and PNG writer used by production recipes.
/// Operationally recognized selector metadata retains its production style;
/// otherwise a deterministic full-finite-range generic style is used.
///
/// Vendor divergence from upstream fe56726: the finite range is computed
/// from the decoded plane itself rather than the writer-recorded tile
/// statistics (`stats_2d`), because this lane always decodes the full
/// plane anyway and the vendored rw-store predates the stats reader.
pub fn render_generic_store_variable(
    source: &StoreFieldSource,
    config: &super::StoreRenderConfig,
    storage_slot: u16,
    variable: &str,
) -> Result<GenericRenderedVariable, Box<dyn std::error::Error>> {
    let started = Instant::now();
    let meta = source
        .surface_variable(variable)
        .ok_or_else(|| format!("stored 2-D variable {variable:?} does not exist"))?
        .clone();
    let mut stored = source.generic_grid(variable)?;
    let mut finite_range: Option<(f32, f32)> = None;
    for value in &stored.values {
        if value.is_finite() {
            finite_range = Some(match finite_range {
                None => (*value, *value),
                Some((lo, hi)) => (lo.min(*value), hi.max(*value)),
            });
        }
    }
    // Style resolution ladder: the variable's own production style, then
    // the curated same-quantity mapping, then the generic full-range ramp.
    let style =
        operational_style_for_store_variable(variable, &meta.selector, &meta.units, config.model)
            .or_else(|| {
                curated_style_for_store_variable(
                    variable,
                    &meta.selector,
                    &meta.units,
                    config.model,
                )
            })
            .unwrap_or_else(|| {
                // The generic ramp is a correct fallback, not an error --
                // but say so, per variable, so an unstyled gallery frame is
                // traceable to a missing colortable mapping rather than to
                // a silent style decision.
                rustwx_render::advisory::advise(format!(
                    "GENERIC_STYLE\t{variable}\tno operational or curated colortable \
                     resolved; using the full-finite-range generic fill"
                ));
                generic_style_for_store_variable(variable, &meta.units, finite_range)
            });
    if !style.convert.is_none() {
        for value in &mut stored.values {
            if value.is_finite() {
                *value = style.convert.apply(*value);
            }
        }
    }

    // The SAME frame geometry the direct lane gives named products
    // (visual-mode, domain-frame, and chrome-scale aware).  The plain
    // `map_frame_aspect_ratio` built a narrower frame whose subtitle row
    // truncated text that named products fit at the same canvas size.
    // Borrowed before `stored.grid` is moved into the field below; the
    // overlay pass needs the same mesh the projection was built from.
    let stored_lat = stored.grid.lat_deg.clone();
    let stored_lon = stored.grid.lon_deg.clone();
    let projected = build_projected_map_with_projection(
        &stored.grid.lat_deg,
        &stored.grid.lon_deg,
        source.projection(),
        config.domain.bounds,
        direct_map_frame_aspect_ratio(
            ProductVisualMode::FilledMeteorology,
            config.output_width,
            config.output_height,
            source.projection(),
        ),
    )?;
    let field = Field2D::new(
        ProductKey::named(generic_variable_output_slug(variable)),
        style.display_units.clone(),
        stored.grid,
        stored.values,
    )?;
    let presentation = super::hour_presentation(config, storage_slot, source.exact_time())?;
    let (subtitle_left, subtitle_right) = generic_subtitle_row(config, &presentation);
    let mut request = MapRenderRequest::from_core_field(field, style.scale);
    StaticPlotDesign::new(config.domain.bounds, ProductVisualMode::FilledMeteorology)
        .apply_to_request(&mut request);
    request.width = config.output_width;
    request.height = config.output_height;
    request.title = Some(style.title);
    request.cbar_tick_step = style.cbar_tick_step;
    request.render_density = style.colormap_options.render_density;
    request.legend = style.colormap_options.legend;
    request.subtitle_left = Some(subtitle_left);
    request.subtitle_right = Some(subtitle_right);
    request.projected_domain = Some(ProjectedDomain {
        x: projected.projected_x,
        y: projected.projected_y,
        extent: projected.extent,
    });
    request.projected_lines = projected.lines;
    request.projected_polygons = projected.polygons;
    request.inverse_raster_projection = projected.inverse_raster_projection;

    // gpuwm addition (VENDOR.md): the same `--overlays`/`--annotate`
    // payload the direct lane applies, projected with the SAME bounds and
    // aspect ratio this function's own `build_projected_map_with_projection`
    // call above used.  A generic `var:` panel and a named product panel
    // of the same run must carry the overlay in the same place.  Both are
    // `None` unless the flags were passed, so the default render is
    // byte-unchanged.
    if let Some(overlays) = config.geographic_overlays.as_ref() {
        overlays.apply(
            &mut request,
            &stored_lat,
            &stored_lon,
            source.projection(),
            config.domain.bounds,
            direct_map_frame_aspect_ratio(
                ProductVisualMode::FilledMeteorology,
                config.output_width,
                config.output_height,
                source.projection(),
            ),
        )?;
    }
    if let Some(annotations) = config.panel_annotations.as_ref() {
        annotations.apply(&mut request);
    }

    std::fs::create_dir_all(&config.out_dir)?;
    // Same shape as the direct lane's artifact name --
    // `rustwx_{model}_{date}_{cycle}z_f{hour:03}_{domain}_{slug}{_suffix}.png`
    // -- so one gallery sorts named and generic products together.
    let suffix = presentation
        .output_suffix
        .as_deref()
        .map(|value| format!("_{value}"))
        .unwrap_or_default();
    let output_path = config.out_dir.join(format!(
        "rustwx_{}_{}_{}z_f{:03}_{}_{}{}.png",
        config.model.as_str().replace('-', "_"),
        config.date_yyyymmdd,
        config.cycle_utc,
        presentation.forecast_hour,
        config.domain.slug,
        generic_variable_output_slug(variable),
        suffix,
    ));
    save_png_profile_with_options(
        &request,
        &output_path,
        &PngWriteOptions {
            compression: config.png_compression,
        },
    )?;
    Ok(GenericRenderedVariable {
        variable: variable.to_string(),
        input_units: meta.units.clone(),
        display_units: style.display_units,
        output_path,
        total_ms: started.elapsed().as_millis(),
    })
}

/// The subtitle row for a generic render: the same shape named products
/// carry, so the same text always survives the renderer's width budget.
///
/// The right half is EXACTLY the provenance stamp and nothing else.  The
/// renderer measures the right subtitle and end-ellipsizes it inside a
/// half-row cap, so any variable/units segment sharing that string made
/// the stamp the casualty ("rw-store variable el [m] | sour...").  The
/// variable identity and stored units live in the TITLE row instead,
/// which has the full row width and its own ellipsis -- a long variable
/// name gives way there, never the stamp, and the left half (init /
/// lead / full valid time) keeps the width named products get.
pub(crate) fn generic_subtitle_row(
    config: &super::StoreRenderConfig,
    presentation: &super::HourPresentation,
) -> (String, String) {
    let left = presentation.subtitle_left.clone().unwrap_or_else(|| {
        rustwx_products::shared_context::model_time_subtitle(
            config.model,
            &config.date_yyyymmdd,
            config.cycle_utc,
            presentation.forecast_hour,
        )
    });
    let right = super::source_subtitle_override(config)
        .unwrap_or_else(|| format!("source: {}", config.source));
    (left, right)
}

/// Deterministic filename token for an arbitrary stored variable name: the
/// sanitized name (truncated) plus an FNV-1a hash of the exact original, so
/// two names that sanitize identically can never overwrite each other.
fn generic_variable_output_slug(variable: &str) -> String {
    let mut safe = safe_artifact_component(variable);
    safe.truncate(56);
    let hash = variable
        .as_bytes()
        .iter()
        .fold(0xcbf2_9ce4_8422_2325u64, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
        });
    format!("var_{safe}_{hash:016x}")
}

fn safe_artifact_component(value: &str) -> String {
    let mut safe = String::with_capacity(value.len().min(64));
    let mut underscore = false;
    for byte in value.bytes() {
        let next = if byte.is_ascii_alphanumeric() {
            underscore = false;
            Some((byte as char).to_ascii_lowercase())
        } else if !underscore {
            underscore = true;
            Some('_')
        } else {
            None
        };
        if let Some(next) = next {
            safe.push(next);
        }
    }
    let safe = safe.trim_matches('_');
    if safe.is_empty() {
        "unnamed".to_string()
    } else {
        safe.to_string()
    }
}

/// Render the requested derived/heavy recipes from their precomputed store
/// grids through the derived lane's store-render seam (the same projected
/// crop, styles, scales, and save path the GRIB lane runs). A recipe is
/// renderable iff its slug-named grid exists in the hour; `theta_e_2m_10m_winds`
/// additionally needs the stored 10 m u/v planes for its barb overlay.
pub fn render_derived_recipes_from_store(
    source: &StoreFieldSource,
    request: &DerivedBatchRequest,
    cycle_utc: u8,
    recipe_slugs: &[String],
) -> Result<DerivedStoreOutcome, Box<dyn std::error::Error>> {
    use rustwx_core::CanonicalField;

    let mut grids = Vec::new();
    let mut renderable = Vec::new();
    let mut skipped = Vec::new();
    let mut winds: Option<(Vec<f64>, Vec<f64>)> = None;
    for slug in recipe_slugs {
        let stored = match source.derived_grid(slug) {
            Ok(stored) => stored,
            Err(RwStoreError::UnknownVariable(_)) => {
                skipped.push(StoreRenderSkip {
                    slug: slug.clone(),
                    reason: "not stored: ingest did not realize this recipe grid".to_string(),
                });
                continue;
            }
            Err(err) => return Err(format!("read derived grid '{slug}': {err}").into()),
        };
        if slug == "theta_e_2m_10m_winds" && winds.is_none() {
            let u10 = FieldSelector::height_agl(CanonicalField::UWind, 10);
            let v10 = FieldSelector::height_agl(CanonicalField::VWind, 10);
            if source.resolve(&u10).is_none() || source.resolve(&v10).is_none() {
                skipped.push(StoreRenderSkip {
                    slug: slug.clone(),
                    reason: "stored grid present but the 10 m u/v planes its barb overlay \
                             needs are not stored"
                        .to_string(),
                });
                continue;
            }
            let to_f64 =
                |field: SelectedField2D| field.values.iter().map(|&v| f64::from(v)).collect();
            winds = Some((to_f64(source.fetch(&u10)?), to_f64(source.fetch(&v10)?)));
        }
        grids.push(StoreProductGrid {
            slug: slug.clone(),
            units: stored.units,
            values: stored.values.iter().map(|&v| f64::from(v)).collect(),
        });
        renderable.push(slug.clone());
    }
    if renderable.is_empty() {
        return Ok(DerivedStoreOutcome {
            rendered: Vec::new(),
            skipped,
        });
    }

    let mut sub_request = request.clone();
    sub_request.recipe_slugs = renderable;
    let full_grid = source.full_grid();
    let rendered = render_derived_recipes_from_store_grids(
        &sub_request,
        cycle_utc,
        &full_grid,
        source.projection(),
        &grids,
        winds
            .as_ref()
            .map(|(u10, v10)| (u10.as_slice(), v10.as_slice())),
        vec![source.fetch_key()],
    )?;
    Ok(DerivedStoreOutcome { rendered, skipped })
}
