//! Bounded, callback-driven batch rendering over an `rw-store` run.
//!
//! This is orchestration only.  Every image goes through [`crate::render_all`],
//! which in turn uses the production direct, derived, heavy, and windowed
//! render paths.  Callers own the worker thread and receive plain-data events,
//! making the API suitable for egui without ever blocking its frame thread.

use std::collections::{HashMap, HashSet};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use rustwx_core::{CycleSpec, ModelId, SourceId};
use rustwx_models::{model_summary, plot_recipe_store_requirements};
use rustwx_products::derived::{
    is_heavy_derived_recipe_slug, store_derived_recipe_slugs, store_heavy_recipe_slugs,
};
use rustwx_products::direct::{direct_recipe_is_time_invariant, store_direct_recipe_slugs};
use rustwx_products::shared_context::{DomainSpec, TitleProvenance};
use rustwx_products::windowed::HrrrWindowedProduct;
use rustwx_render::PngCompressionMode;

use crate::render_all::{
    StoreFieldSource, StoreRenderConfig, partition_products, render_hour_products,
    render_windowed_products,
};

/// Hard defaults used by the GUI lane.  The default job is still one product
/// for one hour; these ceilings merely prevent an accidental unbounded click.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchRenderLimits {
    pub max_hours: usize,
    pub max_products_per_hour: usize,
    pub max_work_items: usize,
    pub max_output_width: u32,
    pub max_output_height: u32,
    pub max_output_pixels: u64,
}

impl Default for BatchRenderLimits {
    fn default() -> Self {
        Self {
            // Covers long global-model runs while the product-hour ceiling
            // still prevents a giant cross product.
            max_hours: 256,
            max_products_per_hour: 32,
            max_work_items: 512,
            max_output_width: 1_920,
            max_output_height: 1_440,
            max_output_pixels: 2_764_800,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchHourScope {
    Current(u16),
    AllStored,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BatchRenderDomain {
    /// Tight finite lat/lon extent of the run's `grid.rwg`.
    NativeGrid,
    Bounds {
        slug: String,
        west: f64,
        east: f64,
        south: f64,
        north: f64,
    },
}

#[derive(Debug, Clone)]
pub struct BatchRenderRequest {
    pub store_root: PathBuf,
    /// Exact rw-store directory component (for example `hrrr` or `wrf`).
    pub model_slug: String,
    pub run_slug: String,
    pub hours: BatchHourScope,
    /// Comma-separated production slugs, or a `render_all` catalog keyword.
    pub product_spec: String,
    pub out_dir: PathBuf,
    pub domain: BatchRenderDomain,
    /// Slug written into every output filename in place of the generic
    /// `native_grid`, for [`BatchRenderDomain::NativeGrid`] runs whose
    /// caller knows the grid's identity (a WRF nest's `d02-3km`).  One
    /// run's nests are otherwise indistinguishable in the filename --
    /// same model, same cycle, same forecast hour -- and the second
    /// domain rendered at a lead overwrites the first.
    pub native_domain_slug: Option<String>,
    /// Grid spacing appended to the plot's time subtitle (`Δx 3 km`).
    pub subtitle_spacing: Option<String>,
    /// Provenance label for the subtitle, replacing the store model's
    /// registered fetch source.  A locally-imported run was never
    /// fetched from anywhere.
    pub source_label: Option<String>,
    /// Where these frames came from, for the plot headline.  Defaults to
    /// the source catalog, so a fetched run's titles are untouched; a
    /// locally-imported run declares itself here and names its own grid.
    pub title_provenance: TitleProvenance,
    /// Optional label overrides for nonstandard/local run names.
    pub date_yyyymmdd: Option<String>,
    pub cycle_utc: Option<u8>,
    /// Provenance subtitle override.  The model's primary source is used when
    /// omitted because rw-store v1 does not retain fetch-source identity.
    pub source: Option<SourceId>,
    pub output_width: u32,
    pub output_height: u32,
    pub limits: BatchRenderLimits,
}

impl BatchRenderRequest {
    pub fn conservative(
        store_root: impl Into<PathBuf>,
        model_slug: impl Into<String>,
        run_slug: impl Into<String>,
        hour: u16,
        product_slug: impl Into<String>,
        out_dir: impl Into<PathBuf>,
    ) -> Self {
        Self {
            store_root: store_root.into(),
            model_slug: model_slug.into(),
            run_slug: run_slug.into(),
            hours: BatchHourScope::Current(hour),
            product_spec: product_slug.into(),
            out_dir: out_dir.into(),
            domain: BatchRenderDomain::NativeGrid,
            native_domain_slug: None,
            subtitle_spacing: None,
            source_label: None,
            title_provenance: TitleProvenance::default(),
            date_yyyymmdd: None,
            cycle_utc: None,
            source: None,
            output_width: 1_200,
            output_height: 900,
            limits: BatchRenderLimits::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum BatchProductKind {
    Direct,
    Derived,
    Heavy,
    Generic,
    Windowed,
}

impl BatchProductKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::Direct => "direct",
            Self::Derived => "derived",
            Self::Heavy => "heavy",
            Self::Generic => "generic",
            Self::Windowed => "windowed",
        }
    }
}

/// One recipe available to the inspected run. Direct recipes are proven from
/// selector metadata and derived recipes from stored slug grids. Windowed
/// candidates are listed for multi-hour HRRR runs and may still report an
/// honest blocker when their exact contiguous window is incomplete.
/// `source_fields` maps a store-browser selection to recipes that consume it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchProductOption {
    pub slug: String,
    pub kind: BatchProductKind,
    pub source_fields: Vec<String>,
    /// Stored units of the backing variable, when the product IS one
    /// stored variable (derived/heavy grids and generic `var:` rows).
    pub units: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchRenderCatalog {
    pub products: Vec<BatchProductOption>,
    pub stored_hours: Vec<u16>,
}

/// Inspect one hour without decoding any field payloads.  Selector metadata is
/// enough to prove direct-recipe coverage; derived grids are slug-addressed.
/// Model-identity-free (gpuwm vendor divergence): every non-opt-in direct
/// recipe is a candidate and the STORED FIELDS decide, never a per-model
/// fetch plan.
pub fn inspect_renderable_products(
    store_root: &Path,
    model_slug: &str,
    run_slug: &str,
    hour: u16,
) -> Result<BatchRenderCatalog, String> {
    validate_store_component("model", model_slug)?;
    validate_store_component("run", run_slug)?;
    let store = StoreFieldSource::open(store_root, model_slug, run_slug, hour)
        .map_err(|err| err.to_string())?;
    let mut products = Vec::new();

    for slug in store_direct_recipe_slugs() {
        let Ok(requirements) = plot_recipe_store_requirements(&slug) else {
            continue;
        };
        let mut source_fields = Vec::new();
        let mut renderable = true;
        for requirement in requirements {
            let resolved = requirement
                .selector
                .and_then(|selector| store.resolve(&selector));
            let Some(name) = resolved else {
                renderable = false;
                break;
            };
            if !source_fields.iter().any(|existing| existing == name) {
                source_fields.push(name.to_string());
            }
        }
        if renderable {
            products.push(BatchProductOption {
                slug,
                kind: BatchProductKind::Direct,
                source_fields,
                units: None,
            });
        }
    }

    let known_derived: HashSet<&str> = store_derived_recipe_slugs()
        .into_iter()
        .chain(store_heavy_recipe_slugs())
        .collect();
    for slug in store.derived_slugs() {
        if !known_derived.contains(slug.as_str()) {
            continue;
        }
        products.push(BatchProductOption {
            slug: slug.clone(),
            kind: if is_heavy_derived_recipe_slug(slug) {
                BatchProductKind::Heavy
            } else {
                BatchProductKind::Derived
            },
            source_fields: vec![slug.clone()],
            units: store.surface_variable(slug).map(|var| var.units.clone()),
        });
    }

    // Every remaining stored 2-D variable is renderable through the generic
    // lane as `var:<name>`.  A variable that is already the SOLE source
    // field of a named product above is excluded -- listing it would render
    // the same grid twice under two slugs -- and every exclusion says so on
    // stderr rather than silently thinning the catalog.
    let sole_source_of_named: HashSet<String> = products
        .iter()
        .filter_map(|product| match product.source_fields.as_slice() {
            [only] => Some(only.clone()),
            _ => None,
        })
        .collect();
    for variable in store.surface_variables() {
        if let Err(reason) = crate::render_all::validate_generic_variable_name(&variable.name) {
            eprintln!(
                "GENERIC_EXCLUDED\t{}\tname is not request-safe: {reason}",
                variable.name.escape_debug()
            );
            continue;
        }
        if sole_source_of_named.contains(&variable.name) {
            eprintln!(
                "GENERIC_EXCLUDED\t{}\talready rendered by a named product using exactly \
                 this stored grid",
                variable.name
            );
            continue;
        }
        products.push(BatchProductOption {
            slug: format!("var:{}", variable.name),
            kind: BatchProductKind::Generic,
            source_fields: vec![variable.name.clone()],
            units: Some(variable.units.clone()),
        });
    }

    let stored_hours =
        crate::render_all::windowed_store::stored_run_hours(store_root, model_slug, run_slug)
            .map_err(|err| err.to_string())?;
    // Windowed products are axis-gated, not model-gated: any run with more
    // than one WHOLE-hour stored frame lists them, and each product's
    // per-plane availability is proven (or blocked with a reason) when the
    // windows compute.  Exact-time ordinal axes stay out -- fixed-hour
    // windows are undefined on them.
    let windowed_ready =
        crate::render_all::windowed_store::windowed_axis_ready(store_root, model_slug, run_slug)
            .map_err(|err| err.to_string())?;
    if windowed_ready {
        products.extend(
            HrrrWindowedProduct::supported_products()
                .iter()
                .map(|product| BatchProductOption {
                    slug: product.slug().to_string(),
                    kind: BatchProductKind::Windowed,
                    source_fields: Vec::new(),
                    units: None,
                }),
        );
    }

    let mut seen = HashSet::new();
    products.retain(|product| seen.insert(product.slug.clone()));
    products.sort_by(|left, right| {
        left.kind
            .cmp(&right.kind)
            .then_with(|| left.slug.cmp(&right.slug))
    });
    Ok(BatchRenderCatalog {
        products,
        stored_hours,
    })
}

#[derive(Debug, Clone)]
pub enum BatchRenderEvent {
    Started {
        planned_items: usize,
        hours: Vec<u16>,
        products: Vec<String>,
        output_dir: PathBuf,
    },
    HourStarted {
        hour: u16,
        index: usize,
        total: usize,
    },
    ItemStarted {
        hour: Option<u16>,
        slug: String,
        kind: BatchProductKind,
        completed: usize,
        total: usize,
    },
    ItemRendered {
        hour: Option<u16>,
        slug: String,
        output_path: PathBuf,
        render_ms: u128,
        completed: usize,
        total: usize,
    },
    ItemSkipped {
        hour: Option<u16>,
        slug: String,
        reason: String,
        completed: usize,
        total: usize,
    },
    ItemFailed {
        hour: Option<u16>,
        slug: String,
        error: String,
        completed: usize,
        total: usize,
    },
    Finished(BatchRenderSummary),
}

#[derive(Debug, Clone, Default)]
pub struct BatchRenderSummary {
    pub planned: usize,
    pub rendered: usize,
    pub skipped: usize,
    pub failed: usize,
    pub cancelled: bool,
    pub elapsed_ms: u128,
    pub outputs: Vec<PathBuf>,
}

enum ProductOutcome {
    Rendered {
        output_path: PathBuf,
        render_ms: u128,
    },
    Skipped(String),
}

/// Run a validated job synchronously on the caller's worker thread.
/// Cancellation is observed between products; an in-flight native render is
/// allowed to finish so renderer state is never abandoned halfway through a
/// file write.
pub fn run_batch_render(
    request: BatchRenderRequest,
    cancel: &AtomicBool,
    mut emit: impl FnMut(BatchRenderEvent),
) -> Result<BatchRenderSummary, String> {
    let started = Instant::now();
    validate_store_component("model", &request.model_slug)?;
    validate_store_component("run", &request.run_slug)?;
    if request.product_spec.len() > 16 * 1024 {
        return Err("product selection exceeds the 16 KiB GUI request limit".to_string());
    }
    let model = parse_model(&request.model_slug)?;
    validate_dimensions(&request)?;
    let cycle = resolve_cycle(&request)?;
    let source = match request.source {
        Some(source) => source,
        None => model_summary(model)
            .sources
            .first()
            .map(|descriptor| descriptor.id)
            .ok_or_else(|| format!("model {model} has no configured provenance source"))?,
    };

    let mut product_request =
        partition_products(&request.product_spec).map_err(|err| err.to_string())?;
    dedup(&mut product_request.direct);
    dedup(&mut product_request.derived);
    dedup(&mut product_request.generic);
    dedup(&mut product_request.windowed);

    let stored_hours = crate::render_all::windowed_store::stored_run_hours(
        &request.store_root,
        &request.model_slug,
        &request.run_slug,
    )
    .map_err(|err| err.to_string())?;
    // The "all" keyword only pulls the windowed lane onto runs whose axis
    // can serve it (multiple WHOLE-hour frames); explicit windowed slug
    // requests are left to fail loudly with the axis reason.
    if product_request.windowed_auto
        && !crate::render_all::windowed_store::windowed_axis_ready(
            &request.store_root,
            &request.model_slug,
            &request.run_slug,
        )
        .map_err(|err| err.to_string())?
    {
        product_request.windowed.clear();
    }
    let hours = match request.hours {
        BatchHourScope::Current(hour) => vec![hour],
        BatchHourScope::AllStored => stored_hours.clone(),
    };
    if hours.is_empty() {
        return Err(format!(
            "run {}/{} has no stored hours",
            request.model_slug, request.run_slug
        ));
    }

    let mut per_hour = Vec::new();
    per_hour.extend(
        product_request
            .direct
            .iter()
            .cloned()
            .map(|slug| (BatchProductKind::Direct, slug)),
    );
    per_hour.extend(product_request.derived.iter().cloned().map(|slug| {
        let kind = if is_heavy_derived_recipe_slug(&slug) {
            BatchProductKind::Heavy
        } else {
            BatchProductKind::Derived
        };
        (kind, slug)
    }));
    per_hour.extend(
        product_request
            .generic
            .iter()
            .map(|name| (BatchProductKind::Generic, format!("var:{name}"))),
    );
    validate_work(
        &request,
        &hours,
        per_hour.len(),
        product_request.windowed.len(),
    )?;
    std::fs::create_dir_all(&request.out_dir).map_err(|err| {
        format!(
            "create output directory {}: {err}",
            request.out_dir.display()
        )
    })?;

    let planned = hours
        .len()
        .checked_mul(per_hour.len())
        .and_then(|count| count.checked_add(product_request.windowed.len()))
        .ok_or_else(|| "batch work-item count overflowed usize".to_string())?;
    let all_products = per_hour
        .iter()
        .map(|(_, slug)| slug.clone())
        .chain(product_request.windowed.iter().cloned())
        .collect::<Vec<_>>();
    emit(BatchRenderEvent::Started {
        planned_items: planned,
        hours: hours.clone(),
        products: all_products,
        output_dir: request.out_dir.clone(),
    });

    let mut summary = BatchRenderSummary {
        planned,
        ..BatchRenderSummary::default()
    };
    let mut completed = 0usize;
    let mut native_domain = None;

    'hours: for (hour_index, &hour) in hours.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            break;
        }
        emit(BatchRenderEvent::HourStarted {
            hour,
            index: hour_index + 1,
            total: hours.len(),
        });
        let store = match protected(|| {
            StoreFieldSource::open(
                &request.store_root,
                &request.model_slug,
                &request.run_slug,
                hour,
            )
            .map_err(|err| err.to_string())
        }) {
            Ok(store) => store,
            Err(error) => {
                for (kind, slug) in &per_hour {
                    if cancel.load(Ordering::Relaxed) {
                        break 'hours;
                    }
                    emit(BatchRenderEvent::ItemStarted {
                        hour: Some(hour),
                        slug: slug.clone(),
                        kind: *kind,
                        completed,
                        total: planned,
                    });
                    completed += 1;
                    summary.failed += 1;
                    emit(BatchRenderEvent::ItemFailed {
                        hour: Some(hour),
                        slug: slug.clone(),
                        error: format!("open stored hour: {error}"),
                        completed,
                        total: planned,
                    });
                }
                continue;
            }
        };

        let domain = resolve_render_domain(
            &request.domain,
            request.native_domain_slug.as_deref(),
            &store,
            &mut native_domain,
        )?;
        let config = render_config(&request, model, &cycle, source, domain);

        for (kind, slug) in &per_hour {
            if cancel.load(Ordering::Relaxed) {
                break 'hours;
            }
            emit(BatchRenderEvent::ItemStarted {
                hour: Some(hour),
                slug: slug.clone(),
                kind: *kind,
                completed,
                total: planned,
            });
            // A grid property, not a forecast: render it on the first
            // selected hour and say so for the rest, rather than writing
            // one identical image per lead.  Announced, because a silent
            // drop and a deliberate single render look the same in a
            // directory listing.
            if matches!(kind, BatchProductKind::Direct)
                && hour_index > 0
                && direct_recipe_is_time_invariant(slug)
            {
                completed += 1;
                summary.skipped += 1;
                emit(BatchRenderEvent::ItemSkipped {
                    hour: Some(hour),
                    slug: slug.clone(),
                    reason: format!(
                        "static field: does not vary with forecast time, rendered once for this domain at F{:03}",
                        hours[0]
                    ),
                    completed,
                    total: planned,
                });
                continue;
            }
            let outcome = protected(|| render_hour_item(&config, &store, hour, *kind, slug));
            completed += 1;
            match outcome {
                Ok(ProductOutcome::Rendered {
                    output_path,
                    render_ms,
                }) => {
                    summary.rendered += 1;
                    summary.outputs.push(output_path.clone());
                    emit(BatchRenderEvent::ItemRendered {
                        hour: Some(hour),
                        slug: slug.clone(),
                        output_path,
                        render_ms,
                        completed,
                        total: planned,
                    });
                }
                Ok(ProductOutcome::Skipped(reason)) => {
                    summary.skipped += 1;
                    emit(BatchRenderEvent::ItemSkipped {
                        hour: Some(hour),
                        slug: slug.clone(),
                        reason,
                        completed,
                        total: planned,
                    });
                }
                Err(error) => {
                    summary.failed += 1;
                    emit(BatchRenderEvent::ItemFailed {
                        hour: Some(hour),
                        slug: slug.clone(),
                        error,
                        completed,
                        total: planned,
                    });
                }
            }
        }
    }

    if !cancel.load(Ordering::Relaxed) && !product_request.windowed.is_empty() {
        let anchor_hour = stored_hours
            .last()
            .copied()
            .ok_or_else(|| "windowed render needs at least one stored hour".to_string())?;
        let store = protected(|| {
            StoreFieldSource::open(
                &request.store_root,
                &request.model_slug,
                &request.run_slug,
                anchor_hour,
            )
            .map_err(|err| err.to_string())
        });
        match store {
            Ok(store) => {
                let domain = resolve_render_domain(
                    &request.domain,
                    request.native_domain_slug.as_deref(),
                    &store,
                    &mut native_domain,
                )?;
                let config = render_config(&request, model, &cycle, source, domain);
                for slug in &product_request.windowed {
                    emit(BatchRenderEvent::ItemStarted {
                        hour: None,
                        slug: slug.clone(),
                        kind: BatchProductKind::Windowed,
                        completed,
                        total: planned,
                    });
                }
                // One multi-product call is load-bearing: windowed_store can
                // read each (hour, source plane) once and fold it into every
                // selected accumulator. Calling once per slug would turn a
                // 40-product export into 40 full passes over the run.
                let outcomes = protected(|| {
                    render_windowed_items(
                        &config,
                        &store,
                        &request.store_root,
                        &request.model_slug,
                        &request.run_slug,
                        &product_request.windowed,
                    )
                });
                match outcomes {
                    Ok(outcomes) => {
                        for (slug, outcome) in outcomes {
                            completed += 1;
                            match outcome {
                                ProductOutcome::Rendered {
                                    output_path,
                                    render_ms,
                                } => {
                                    summary.rendered += 1;
                                    summary.outputs.push(output_path.clone());
                                    emit(BatchRenderEvent::ItemRendered {
                                        hour: None,
                                        slug,
                                        output_path,
                                        render_ms,
                                        completed,
                                        total: planned,
                                    });
                                }
                                ProductOutcome::Skipped(reason) => {
                                    summary.skipped += 1;
                                    emit(BatchRenderEvent::ItemSkipped {
                                        hour: None,
                                        slug,
                                        reason,
                                        completed,
                                        total: planned,
                                    });
                                }
                            }
                        }
                    }
                    Err(error) => {
                        for slug in &product_request.windowed {
                            completed += 1;
                            summary.failed += 1;
                            emit(BatchRenderEvent::ItemFailed {
                                hour: None,
                                slug: slug.clone(),
                                error: error.clone(),
                                completed,
                                total: planned,
                            });
                        }
                    }
                }
            }
            Err(error) => {
                for slug in &product_request.windowed {
                    if cancel.load(Ordering::Relaxed) {
                        break;
                    }
                    emit(BatchRenderEvent::ItemStarted {
                        hour: None,
                        slug: slug.clone(),
                        kind: BatchProductKind::Windowed,
                        completed,
                        total: planned,
                    });
                    completed += 1;
                    summary.failed += 1;
                    emit(BatchRenderEvent::ItemFailed {
                        hour: None,
                        slug: slug.clone(),
                        error: format!("open window anchor hour: {error}"),
                        completed,
                        total: planned,
                    });
                }
            }
        }
    }

    summary.cancelled = cancel.load(Ordering::Relaxed);
    summary.elapsed_ms = started.elapsed().as_millis();
    emit(BatchRenderEvent::Finished(summary.clone()));
    Ok(summary)
}

fn render_hour_item(
    config: &StoreRenderConfig,
    store: &StoreFieldSource,
    hour: u16,
    kind: BatchProductKind,
    slug: &str,
) -> Result<ProductOutcome, String> {
    if kind == BatchProductKind::Generic {
        let variable = slug.strip_prefix("var:").ok_or_else(|| {
            format!("internal generic product identity {slug:?} is missing the 'var:' prefix")
        })?;
        let rendered = crate::render_all::store_render::render_generic_store_variable(
            store, config, hour, variable,
        )
        .map_err(|err| err.to_string())?;
        return Ok(ProductOutcome::Rendered {
            output_path: rendered.output_path,
            render_ms: rendered.total_ms,
        });
    }
    let single = vec![slug.to_string()];
    let empty: &[String] = &[];
    let (direct, derived) = if kind == BatchProductKind::Direct {
        (single.as_slice(), empty)
    } else {
        (empty, single.as_slice())
    };
    let mut outcome = render_hour_products(config, store, hour, direct, derived, empty, None)
        .map_err(|err| err.to_string())?;
    if let Some(rendered) = outcome.rendered.pop() {
        return Ok(ProductOutcome::Rendered {
            output_path: rendered.output_path,
            render_ms: rendered.total_ms,
        });
    }
    if let Some(skipped) = outcome.skipped.pop() {
        return Ok(ProductOutcome::Skipped(skipped.reason));
    }
    Err(format!(
        "renderer returned no image and no blocker for '{slug}'"
    ))
}

fn render_windowed_items(
    config: &StoreRenderConfig,
    store: &StoreFieldSource,
    store_root: &Path,
    model_slug: &str,
    run_slug: &str,
    requested: &[String],
) -> Result<Vec<(String, ProductOutcome)>, String> {
    let Some(mut outcome) = render_windowed_products(
        config, store, store_root, model_slug, run_slug, requested, false,
    )
    .map_err(|err| err.to_string())?
    else {
        return Ok(requested
            .iter()
            .cloned()
            .map(|slug| {
                (
                    slug,
                    ProductOutcome::Skipped(
                        "windowed renderer found no usable multi-hour window".to_string(),
                    ),
                )
            })
            .collect());
    };
    let mut by_slug = HashMap::new();
    for rendered in outcome.rendered.drain(..) {
        by_slug.insert(
            rendered.slug,
            ProductOutcome::Rendered {
                output_path: rendered.output_path,
                render_ms: rendered.total_ms,
            },
        );
    }
    for blocked in outcome.blocked.drain(..) {
        by_slug.insert(blocked.slug, ProductOutcome::Skipped(blocked.reason));
    }
    Ok(requested
        .iter()
        .cloned()
        .map(|slug| {
            let outcome = by_slug.remove(&slug).unwrap_or_else(|| {
                ProductOutcome::Skipped(
                    "windowed renderer returned no image and no blocker".to_string(),
                )
            });
            (slug, outcome)
        })
        .collect())
}

fn render_config(
    request: &BatchRenderRequest,
    model: ModelId,
    cycle: &CycleSpec,
    source: SourceId,
    domain: DomainSpec,
) -> StoreRenderConfig {
    StoreRenderConfig {
        model,
        date_yyyymmdd: cycle.date_yyyymmdd.clone(),
        cycle_utc: cycle.hour_utc,
        source,
        subtitle_spacing: request.subtitle_spacing.clone(),
        source_label: request.source_label.clone(),
        title_provenance: request.title_provenance.clone(),
        domain,
        out_dir: request.out_dir.clone(),
        contour_mode: Default::default(),
        native_fill_level_multiplier: 1,
        output_width: request.output_width,
        output_height: request.output_height,
        png_compression: PngCompressionMode::Fast,
        place_label_overlay: None,
    }
}

fn parse_model(model_slug: &str) -> Result<ModelId, String> {
    model_slug.parse::<ModelId>().map_err(|err| {
        format!("store model '{model_slug}' has no production renderer identity: {err}")
    })
}

fn validate_store_component(label: &str, value: &str) -> Result<(), String> {
    let mut components = Path::new(value).components();
    let valid = matches!(components.next(), Some(std::path::Component::Normal(_)))
        && components.next().is_none();
    if valid {
        Ok(())
    } else {
        Err(format!(
            "store {label} must be one non-empty path component, got '{value}'"
        ))
    }
}

fn validate_dimensions(request: &BatchRenderRequest) -> Result<(), String> {
    if request.output_width < 320 || request.output_height < 240 {
        return Err("output dimensions must be at least 320 x 240".to_string());
    }
    if request.output_width > request.limits.max_output_width
        || request.output_height > request.limits.max_output_height
    {
        return Err(format!(
            "output {}x{} exceeds the GUI ceiling {}x{}",
            request.output_width,
            request.output_height,
            request.limits.max_output_width,
            request.limits.max_output_height
        ));
    }
    let pixels = u64::from(request.output_width) * u64::from(request.output_height);
    if pixels > request.limits.max_output_pixels {
        return Err(format!(
            "output has {pixels} pixels; GUI ceiling is {}",
            request.limits.max_output_pixels
        ));
    }
    Ok(())
}

fn validate_work(
    request: &BatchRenderRequest,
    hours: &[u16],
    per_hour_products: usize,
    windowed_products: usize,
) -> Result<(), String> {
    if hours.len() > request.limits.max_hours {
        return Err(format!(
            "{} hours selected; GUI ceiling is {} (split the run into smaller jobs)",
            hours.len(),
            request.limits.max_hours
        ));
    }
    if per_hour_products > request.limits.max_products_per_hour {
        return Err(format!(
            "{per_hour_products} per-hour products selected; GUI ceiling is {}",
            request.limits.max_products_per_hour
        ));
    }
    let work = hours
        .len()
        .checked_mul(per_hour_products)
        .and_then(|count| count.checked_add(windowed_products))
        .ok_or_else(|| "batch work-item count overflowed usize".to_string())?;
    if work == 0 {
        return Err("select at least one product".to_string());
    }
    if work > request.limits.max_work_items {
        return Err(format!(
            "{work} product-hours selected; GUI ceiling is {} (split the job)",
            request.limits.max_work_items
        ));
    }
    Ok(())
}

fn resolve_cycle(request: &BatchRenderRequest) -> Result<CycleSpec, String> {
    let inferred = infer_run_cycle(&request.run_slug);
    let date = request
        .date_yyyymmdd
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .or_else(|| inferred.as_ref().map(|(date, _)| date.clone()))
        .ok_or_else(|| {
            format!(
                "cannot infer an init date from run '{}'; provide YYYYMMDD",
                request.run_slug
            )
        })?;
    let hour = request
        .cycle_utc
        .or_else(|| inferred.map(|(_, hour)| hour))
        .ok_or_else(|| {
            format!(
                "cannot infer a cycle hour from run '{}'; provide 0-23",
                request.run_slug
            )
        })?;
    CycleSpec::new(date, hour).map_err(|err| err.to_string())
}

/// Infer standard `YYYYMMDD_CCz` and local-WRF `...YYYYMMDD_HHMMSS` names.
pub fn infer_run_cycle(run_slug: &str) -> Option<(String, u8)> {
    let bytes = run_slug.as_bytes();
    if bytes.len() < 10 {
        return None;
    }
    for start in 0..=bytes.len().saturating_sub(8) {
        let date_bytes = &bytes[start..start + 8];
        if !date_bytes.iter().all(|byte| byte.is_ascii_digit()) {
            continue;
        }
        let Ok(date) = std::str::from_utf8(date_bytes) else {
            continue;
        };
        if CycleSpec::new(date.to_string(), 0).is_err() {
            continue;
        }
        let search_end = (start + 16).min(bytes.len());
        for hour_start in start + 8..search_end.saturating_sub(1) {
            let pair = &bytes[hour_start..hour_start + 2];
            if !pair.iter().all(|byte| byte.is_ascii_digit()) {
                continue;
            }
            let hour = (pair[0] - b'0') * 10 + (pair[1] - b'0');
            if hour < 24 {
                return Some((date.to_string(), hour));
            }
        }
    }
    None
}

fn explicit_domain(domain: &BatchRenderDomain) -> Result<DomainSpec, String> {
    let BatchRenderDomain::Bounds {
        slug,
        west,
        east,
        south,
        north,
    } = domain
    else {
        return Err("internal error: native domain was not resolved".to_string());
    };
    let (west, east, south, north) = (*west, *east, *south, *north);
    if ![west, east, south, north]
        .iter()
        .all(|value| value.is_finite())
    {
        return Err("custom domain bounds must be finite".to_string());
    }
    let raw_lon_span = (east - west).abs();
    if raw_lon_span == 0.0 || raw_lon_span > 360.0 || south >= north {
        return Err(
            "custom domain needs distinct west/east bounds no more than 360 degrees apart and south < north"
                .to_string(),
        );
    }
    if south < -90.0 || north > 90.0 {
        return Err("custom domain latitude must stay inside -90..90".to_string());
    }
    let slug = safe_slug(slug, "custom");
    Ok(DomainSpec::new(slug, (west, east, south, north)))
}

fn resolve_render_domain(
    requested: &BatchRenderDomain,
    native_slug: Option<&str>,
    store: &StoreFieldSource,
    native_cache: &mut Option<DomainSpec>,
) -> Result<DomainSpec, String> {
    match requested {
        BatchRenderDomain::NativeGrid => {
            if native_cache.is_none() {
                let mut domain = native_grid_domain(store)?;
                if let Some(slug) = native_slug {
                    domain.slug = safe_slug(slug, "native_grid");
                }
                *native_cache = Some(domain);
            }
            native_cache
                .clone()
                .ok_or_else(|| "native render domain was not cached".to_string())
        }
        custom => explicit_domain(custom),
    }
}

fn native_grid_domain(store: &StoreFieldSource) -> Result<DomainSpec, String> {
    let (latitudes, longitudes) = store.grid_coordinates();
    if latitudes.is_empty() || latitudes.len() != longitudes.len() {
        return Err(format!(
            "grid.rwg coordinate lengths are invalid (lat {}, lon {})",
            latitudes.len(),
            longitudes.len()
        ));
    }
    let mut south = f64::INFINITY;
    let mut north = f64::NEG_INFINITY;
    let mut west_180 = f64::INFINITY;
    let mut east_180 = f64::NEG_INFINITY;
    let mut west_360 = f64::INFINITY;
    let mut east_360 = f64::NEG_INFINITY;
    for (&lat, &lon) in latitudes.iter().zip(longitudes.iter()) {
        if !lat.is_finite() || !lon.is_finite() {
            continue;
        }
        let lat = f64::from(lat);
        let lon = f64::from(lon);
        south = south.min(lat);
        north = north.max(lat);
        let lon_180 = (lon + 180.0).rem_euclid(360.0) - 180.0;
        let lon_360 = lon.rem_euclid(360.0);
        west_180 = west_180.min(lon_180);
        east_180 = east_180.max(lon_180);
        west_360 = west_360.min(lon_360);
        east_360 = east_360.max(lon_360);
    }
    if ![south, north, west_180, east_180, west_360, east_360]
        .iter()
        .all(|value| value.is_finite())
    {
        return Err("grid.rwg has no finite latitude/longitude coordinates".to_string());
    }
    let (mut west, mut east) = if east_180 - west_180 <= east_360 - west_360 {
        (west_180, east_180)
    } else {
        (west_360, east_360)
    };
    if (east - west).abs() < 0.01 {
        west -= 0.5;
        east += 0.5;
    } else {
        west -= 0.05;
        east += 0.05;
    }
    if (north - south).abs() < 0.01 {
        south -= 0.5;
        north += 0.5;
    } else {
        south -= 0.05;
        north += 0.05;
    }
    south = south.max(-90.0);
    north = north.min(90.0);
    Ok(DomainSpec::new("native_grid", (west, east, south, north)))
}

fn dedup(values: &mut Vec<String>) {
    let mut seen = HashSet::new();
    values.retain(|value| seen.insert(value.clone()));
}

/// A domain slug is only ever a filename *component* (never a path
/// element), so `.` is admissible and load-bearing: a 3:1 nest of a 12 km
/// parent is 1.333 km, and `d03-1_333km` would read as a typo.  Every
/// separator character still collapses to `_`.
fn safe_slug(value: &str, fallback: &str) -> String {
    let slug = value
        .trim()
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.') {
                character
            } else {
                '_'
            }
        })
        .collect::<String>();
    if slug.is_empty() {
        fallback.to_string()
    } else {
        slug
    }
}

fn protected<T>(run: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    match catch_unwind(AssertUnwindSafe(run)) {
        Ok(result) => result,
        Err(payload) => Err(format!("renderer panicked: {}", panic_message(payload))),
    }
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "unknown panic payload".to_string()
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::atomic::AtomicBool;

    use rustwx_core::{CanonicalField, FieldSelector, GridShape, LatLonGrid, SelectedField2D};
    use rw_store::ingest::{DerivedFieldInput, write_hour_from_fields_with_derived};
    use rw_store::run::RwsRunManifest;
    use rw_store::writer::HourWriter;

    use super::*;

    #[test]
    fn infers_operational_and_local_wrf_run_names() {
        assert_eq!(
            infer_run_cycle("20260608_00z"),
            Some(("20260608".to_string(), 0))
        );
        assert_eq!(
            infer_run_cycle("local_wrf_20110524_180000"),
            Some(("20110524".to_string(), 18))
        );
        assert_eq!(infer_run_cycle("local_wrf_no_time"), None);
    }

    fn test_dir(label: &str) -> PathBuf {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "rusty-weather-batch-{label}-{}-{unique}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A store whose one hour carries: a canonical plane that is the sole
    /// source of a named direct product, a known derived grid, an unknown
    /// derived-marker diagnostic, and a plane whose selector is neither a
    /// FieldSelector nor a derived marker.  The last one used to abort
    /// `StoreFieldSource::open` for the WHOLE hour.
    fn write_mixed_store(root: &std::path::Path, run: &str) -> (String, Vec<String>) {
        let shape = GridShape::new(4, 3).unwrap();
        let lat = vec![
            35.0, 35.0, 35.0, 35.0, 36.0, 36.0, 36.0, 36.0, 37.0, 37.0, 37.0, 37.0,
        ];
        let lon = vec![
            -100.0, -99.0, -98.0, -97.0, -100.0, -99.0, -98.0, -97.0, -100.0, -99.0, -98.0, -97.0,
        ];
        let grid = LatLonGrid::new(shape, lat, lon).unwrap();
        let temperature_selector = FieldSelector::height_agl(CanonicalField::Temperature, 2);
        let temperature_values: Vec<f32> = (0..12).map(|value| 270.0 + value as f32).collect();
        let temperature = SelectedField2D::new(
            temperature_selector,
            "K",
            grid,
            temperature_values.clone(),
        )
        .unwrap();
        let sbcape_values: Vec<f32> = (0..12).map(|value| 500.0 + value as f32 * 100.0).collect();
        let custom_values: Vec<f32> = (0..12).map(|value| value as f32).collect();
        write_hour_from_fields_with_derived(
            root,
            "wrf",
            run,
            0,
            &[("temperature_2m", &temperature)],
            &[
                DerivedFieldInput {
                    name: "sbcape",
                    units: "J/kg",
                    values: &sbcape_values,
                },
                DerivedFieldInput {
                    name: "custom_diagnostic_plane",
                    units: "widgets",
                    values: &custom_values,
                },
            ],
            &[],
            "batch-render-test",
            1_800_000_000,
        )
        .unwrap();

        // Rewrite the hour file with one extra variable whose selector is
        // deliberately opaque.  The writer API cannot spell it, which is
        // the point: only files, not this build's vocabulary, decide what
        // a store contains.
        let manifest =
            RwsRunManifest::load(&root.join("wrf").join(run).join("run.json")).unwrap();
        let mystery_values: Vec<f32> = (0..12).map(|value| -2.0 + value as f32 * 0.5).collect();
        let mut writer = HourWriter::new(
            "wrf",
            run,
            0,
            manifest.nx,
            manifest.ny,
            &manifest.grid_hash,
            "batch-render-test",
        );
        writer
            .add_surface2d(
                "temperature_2m",
                "K",
                serde_json::to_value(temperature_selector).unwrap(),
                &temperature_values,
            )
            .unwrap();
        writer
            .add_surface2d(
                "sbcape",
                "J/kg",
                rw_store::ingest::derived_selector("sbcape"),
                &sbcape_values,
            )
            .unwrap();
        writer
            .add_surface2d(
                "custom_diagnostic_plane",
                "widgets",
                rw_store::ingest::derived_selector("custom_diagnostic_plane"),
                &custom_values,
            )
            .unwrap();
        writer
            .add_surface2d(
                "mystery_plane",
                "widgets",
                serde_json::json!({ "bespoke": "wrfout attribute soup" }),
                &mystery_values,
            )
            .unwrap();
        let entry_file = manifest.hours.get(&0).unwrap().file.clone();
        writer
            .finish(&root.join("wrf").join(run).join(&entry_file))
            .unwrap();
        (
            "mystery_plane".to_string(),
            vec![
                "temperature_2m".to_string(),
                "sbcape".to_string(),
                "custom_diagnostic_plane".to_string(),
                "mystery_plane".to_string(),
            ],
        )
    }

    #[test]
    fn unknown_selectors_are_nonfatal_and_render_through_the_generic_lane() {
        let root = test_dir("generic-lane");
        let out = root.join("out");
        let run = "local_wrf_20200102_030000";
        let (mystery, all_variables) = write_mixed_store(&root, run);

        // (a) The opaque selector no longer aborts the hour.
        let store = StoreFieldSource::open(&root, "wrf", run, 0)
            .expect("one unknown diagnostic must not abort the whole render hour");
        assert_eq!(
            store
                .surface_variables()
                .iter()
                .map(|var| var.name.clone())
                .collect::<Vec<_>>(),
            all_variables,
            "the surface inventory must retain every stored 2-D variable in file order"
        );
        assert_eq!(store.surface_variable(&mystery).unwrap().units, "widgets");

        // (b) The catalog lists the opaque plane as a generic product.
        let catalog = inspect_renderable_products(&root, "wrf", run, 0).unwrap();
        let generic_row = catalog
            .products
            .iter()
            .find(|product| product.slug == format!("var:{mystery}"))
            .expect("var:mystery_plane must be in the catalog");
        assert_eq!(generic_row.kind, BatchProductKind::Generic);
        assert_eq!(generic_row.units.as_deref(), Some("widgets"));

        // (c) Rendering it through the production batch path yields a PNG.
        let request = BatchRenderRequest::conservative(
            &root,
            "wrf",
            run,
            0,
            format!("var:{mystery}"),
            &out,
        );
        let summary =
            run_batch_render(request, &AtomicBool::new(false), |_event| {}).unwrap();
        assert_eq!(summary.rendered, 1, "{summary:?}");
        assert_eq!(summary.failed, 0);
        let output = &summary.outputs[0];
        assert!(output.exists(), "{}", output.display());
        assert!(
            std::fs::metadata(output).unwrap().len() > 1_000,
            "suspiciously small PNG at {}",
            output.display()
        );
        let name = output.file_name().unwrap().to_string_lossy().into_owned();
        assert!(name.contains("var_mystery_plane_"), "{name}");
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn generic_rows_excluded_are_exactly_the_sole_sources_of_named_products() {
        let root = test_dir("generic-dedup");
        let run = "local_wrf_20200102_030000";
        let (_, all_variables) = write_mixed_store(&root, run);
        let catalog = inspect_renderable_products(&root, "wrf", run, 0).unwrap();

        let sole_sources: std::collections::HashSet<&str> = catalog
            .products
            .iter()
            .filter(|product| product.kind != BatchProductKind::Generic)
            .filter_map(|product| match product.source_fields.as_slice() {
                [only] => Some(only.as_str()),
                _ => None,
            })
            .collect();
        assert!(
            sole_sources.contains("temperature_2m"),
            "2m_temperature must claim its stored plane: {catalog:?}"
        );
        assert!(sole_sources.contains("sbcape"), "{catalog:?}");

        for variable in &all_variables {
            let listed = catalog
                .products
                .iter()
                .any(|product| product.slug == format!("var:{variable}"));
            let excluded = sole_sources.contains(variable.as_str());
            assert!(
                listed != excluded,
                "variable '{variable}': listed={listed} excluded-by-dedup={excluded}; \
                 a stored grid must render under exactly one slug"
            );
        }
        let _ = std::fs::remove_dir_all(root);
    }
}
