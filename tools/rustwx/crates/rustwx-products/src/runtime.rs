//! Planner-backed runtime: takes an [`ExecutionPlan`] and materializes
//! every bundle into fetched bytes plus, where applicable, decoded
//! surface and pressure fields. Heavy/derived/severe/ECAPE/direct kernels
//! consume this `LoadedBundleSet` instead of running their own ad hoc
//! fetch wiring.
//!
//! The loader honors the planner's two-level identity:
//! - Each `BundleFetchKey` is fetched once even when several
//!   `CanonicalBundleId`s decode out of the same physical file (for
//!   example, GFS / ECMWF surface + pressure).
//! - Each surface or pressure `CanonicalBundleId` records a typed
//!   decode that the kernels can borrow without re-parsing GRIB bytes.

use rustwx_core::{
    BundleRequirement, CanonicalBundleDescriptor, CanonicalBundleId, CycleSpec, LatLonGrid,
    ModelRunRequest, RustwxError, SourceId,
};
use rustwx_io::{FetchRequest, fetch_bytes_with_cache};
use rustwx_models::{LatestRun, default_bundle_product};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::gridded::{
    CachedDecode, FetchedModelFile, PressureFields, SurfaceFields, decode_cache_path,
    load_or_decode_pressure_from_file_with_shape, load_or_decode_surface_from_file,
    validate_pressure_decode_against_surface,
};
use crate::planner::{BundleFetchKey, ExecutionPlan, PlannedBundle};

/// Outcome of running a fetch+decode pass over an `ExecutionPlan`.
///
/// Partial-success: a fetch or decode failure on one `BundleFetchKey` /
/// `CanonicalBundleId` does not abort the whole plan. Successful fetches
/// are kept in `fetched`; failures are recorded in `fetch_failures`
/// (keyed by physical file) and `bundle_failures` (keyed by decoded
/// bundle, including bundles that couldn't decode because their fetch
/// failed). Lanes consult the `bundle_available` / `bundle_failure`
/// accessors before assuming a bundle is ready; lanes that need a
/// composite (e.g. surface+pressure) and lose one half emit a per-lane
/// blocker without taking down sibling lanes.
#[derive(Debug)]
pub struct LoadedBundleSet {
    pub plan: ExecutionPlan,
    pub latest: LatestRun,
    pub forecast_hour: u16,
    pub fetched: BTreeMap<BundleFetchKey, FetchedBundleBytes>,
    /// Physical-file failures: a `BundleFetchKey` that could not be
    /// fetched (404, network error, cache corruption).
    pub fetch_failures: BTreeMap<BundleFetchKey, String>,
    pub surface_decodes: BTreeMap<CanonicalBundleId, CachedDecode<SurfaceFields>>,
    pub pressure_decodes: BTreeMap<CanonicalBundleId, CachedDecode<PressureFields>>,
    /// Bundle-level failures: a `CanonicalBundleId` that could not be
    /// made ready, whether because its physical fetch failed or its
    /// decode raised an error. The string is a human-readable reason
    /// suitable for surfacing through per-lane blockers.
    pub bundle_failures: BTreeMap<CanonicalBundleId, String>,
    pub timing: LoadedBundleTiming,
}

/// Aggregated timing surfaced into per-lane reports.
///
/// Note on `fetch_ms_total`: this is the **sum of per-worker elapsed
/// time across fetches**, not the wall-clock cost of the fetch phase.
/// When fetches run in parallel, wall-clock is roughly `max(per_fetch_ms)`
/// while this field is the sum across workers and will be larger. Callers
/// that want wall-clock fetch cost should measure around
/// `load_execution_plan` directly.
#[derive(Debug, Default, Clone, Copy)]
pub struct LoadedBundleTiming {
    /// Summed worker-elapsed fetch time across all distinct fetch keys.
    /// See the struct-level note: this is not wall-clock time when the
    /// loader fetches in parallel.
    pub fetch_ms_total: u128,
    pub decode_surface_ms_total: u128,
    pub decode_pressure_ms_total: u128,
    pub cropped_decode_profile: Option<CroppedDecodeProfile>,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct CroppedDecodeProfile {
    pub source_grid_nx: usize,
    pub source_grid_ny: usize,
    pub crop_x_start: usize,
    pub crop_x_end: usize,
    pub crop_y_start: usize,
    pub crop_y_end: usize,
    pub cropped_grid_nx: usize,
    pub cropped_grid_ny: usize,
    pub surface_fetch_bytes_len: usize,
    pub pressure_fetch_bytes_len: usize,
}

/// Raw fetched bytes for a single physical fetch key, plus the original
/// `FetchRequest`/`CachedFetchResult` so manifest code can build
/// `PublishedFetchIdentity` records.
#[derive(Debug, Clone)]
pub struct FetchedBundleBytes {
    pub key: BundleFetchKey,
    pub file: FetchedModelFile,
    pub fetch_ms: u128,
}

/// Configuration for the loader.
#[derive(Debug, Clone)]
pub struct BundleLoaderConfig {
    pub cache_root: PathBuf,
    pub use_cache: bool,
}

impl BundleLoaderConfig {
    pub fn new(cache_root: PathBuf, use_cache: bool) -> Self {
        Self {
            cache_root,
            use_cache,
        }
    }
}

impl LoadedBundleSet {
    pub fn fetched_for(&self, bundle: &PlannedBundle) -> Option<&FetchedBundleBytes> {
        self.fetched.get(&bundle.fetch_key())
    }

    /// Human-readable failure reason for a physical fetch key, if the
    /// loader captured one. `None` means the fetch succeeded.
    pub fn fetch_failure(&self, key: &BundleFetchKey) -> Option<&str> {
        self.fetch_failures.get(key).map(String::as_str)
    }

    /// Human-readable failure reason for a canonical bundle. This
    /// covers both underlying fetch failures and decode errors. `None`
    /// means the bundle is ready (or was never attempted).
    pub fn bundle_failure(&self, id: &CanonicalBundleId) -> Option<&str> {
        self.bundle_failures.get(id).map(String::as_str)
    }

    /// `true` when every fetch the plan requested succeeded.
    pub fn all_fetches_succeeded(&self) -> bool {
        self.fetch_failures.is_empty()
    }

    /// Convenience for kernels that want a (surface, pressure) pair at
    /// the run's nominal forecast hour. Returns the decoded fields and
    /// the matching planned bundles in one call.
    pub fn surface_pressure_pair(
        &self,
    ) -> Option<(
        &PlannedBundle,
        &CachedDecode<SurfaceFields>,
        &PlannedBundle,
        &CachedDecode<PressureFields>,
    )> {
        let surface = self.plan.bundle_for(
            CanonicalBundleDescriptor::SurfaceAnalysis,
            self.forecast_hour,
        )?;
        let pressure = self.plan.bundle_for(
            CanonicalBundleDescriptor::PressureAnalysis,
            self.forecast_hour,
        )?;
        let surface_decode = self.surface_decodes.get(&surface.id)?;
        let pressure_decode = self.pressure_decodes.get(&pressure.id)?;
        Some((surface, surface_decode, pressure, pressure_decode))
    }

    pub fn require_surface_pressure_pair(
        &self,
    ) -> Result<
        (
            &PlannedBundle,
            &CachedDecode<SurfaceFields>,
            &PlannedBundle,
            &CachedDecode<PressureFields>,
        ),
        String,
    > {
        if let Some(pair) = self.surface_pressure_pair() {
            return Ok(pair);
        }

        let mut issues = Vec::new();
        for bundle in [
            CanonicalBundleDescriptor::SurfaceAnalysis,
            CanonicalBundleDescriptor::PressureAnalysis,
        ] {
            let label = match bundle {
                CanonicalBundleDescriptor::SurfaceAnalysis => "surface",
                CanonicalBundleDescriptor::PressureAnalysis => "pressure",
                CanonicalBundleDescriptor::NativeAnalysis => "native",
            };
            match self.plan.bundle_for(bundle, self.forecast_hour) {
                Some(planned) => {
                    if let Some(reason) = self.bundle_failure(&planned.id) {
                        issues.push(format!("{label} bundle failed: {reason}"));
                    } else {
                        let fetched = self.fetched_for(planned).is_some();
                        let decoded = match bundle {
                            CanonicalBundleDescriptor::SurfaceAnalysis => {
                                self.surface_decodes.contains_key(&planned.id)
                            }
                            CanonicalBundleDescriptor::PressureAnalysis => {
                                self.pressure_decodes.contains_key(&planned.id)
                            }
                            CanonicalBundleDescriptor::NativeAnalysis => false,
                        };
                        issues.push(format!(
                            "{label} bundle missing decoded payload (fetched={fetched}, decoded={decoded}, fetch_key={})",
                            planned.fetch_key().native_product
                        ));
                    }
                }
                None => issues.push(format!(
                    "{label} bundle not planned for f{:03}",
                    self.forecast_hour
                )),
            }
        }

        if issues.is_empty() {
            Err("surface/pressure pair unavailable".to_string())
        } else {
            Err(issues.join("; "))
        }
    }

    pub fn surface_decode_for(
        &self,
        bundle: CanonicalBundleDescriptor,
        forecast_hour: u16,
    ) -> Option<&CachedDecode<SurfaceFields>> {
        let planned = self.plan.bundle_for(bundle, forecast_hour)?;
        self.surface_decodes.get(&planned.id)
    }

    pub fn pressure_decode_for(
        &self,
        bundle: CanonicalBundleDescriptor,
        forecast_hour: u16,
    ) -> Option<&CachedDecode<PressureFields>> {
        let planned = self.plan.bundle_for(bundle, forecast_hour)?;
        self.pressure_decodes.get(&planned.id)
    }

    /// Convenience for derived/severe/ECAPE: returns the decoded surface
    /// grid (uses the surface bundle at the run's nominal forecast hour).
    pub fn surface_grid(&self) -> Result<LatLonGrid, RustwxError> {
        let surface_decode = self
            .surface_decode_for(
                CanonicalBundleDescriptor::SurfaceAnalysis,
                self.forecast_hour,
            )
            .expect("surface bundle missing from loaded plan");
        surface_decode.value.core_grid()
    }
}

fn empty_loaded_bundle_set(plan: ExecutionPlan) -> LoadedBundleSet {
    let latest = plan.latest();
    let forecast_hour = plan.forecast_hour;
    LoadedBundleSet {
        plan,
        latest,
        forecast_hour,
        fetched: BTreeMap::new(),
        fetch_failures: BTreeMap::new(),
        surface_decodes: BTreeMap::new(),
        pressure_decodes: BTreeMap::new(),
        bundle_failures: BTreeMap::new(),
        timing: LoadedBundleTiming::default(),
    }
}

fn fetch_execution_plan_into(
    loaded: &mut LoadedBundleSet,
    config: &BundleLoaderConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    // Phase 1: fetch each unique physical file. The planner already
    // deduped, so each worker corresponds to one distinct GRIB file.
    // NOMADS still needs bounded concurrency; fully serial full-GRIB
    // pulls make HRRR latest-cycle processing miss the target budget.
    let fetch_keys = loaded.plan.fetch_keys();
    let fetch_concurrency = fetch_concurrency_for_source(loaded.plan.source, fetch_keys.len());
    let cache_root = config.cache_root.clone();
    let use_cache = config.use_cache;
    let fetch_results: Vec<(
        BundleFetchKey,
        Result<FetchedBundleBytes, Box<dyn std::error::Error + Send + Sync>>,
    )> = if fetch_concurrency > 1 && fetch_keys.len() > 1 {
        let mut results = Vec::with_capacity(fetch_keys.len());
        for chunk in fetch_keys.chunks(fetch_concurrency) {
            let chunk_results = std::thread::scope(|scope| -> Vec<_> {
                let handles: Vec<_> = chunk
                    .iter()
                    .cloned()
                    .map(|key| {
                        let cache_root = cache_root.clone();
                        let use_cache = use_cache;
                        let key_for_worker = key.clone();
                        let plan = &loaded.plan;
                        let handle = scope
                            .spawn(move || fetch_one(plan, key_for_worker, &cache_root, use_cache));
                        (key, handle)
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|(key, handle)| {
                        let result = handle.join().unwrap_or_else(|_| {
                            Err(Box::<dyn std::error::Error + Send + Sync>::from(
                                "planner fetch worker panicked",
                            ))
                        });
                        (key, result)
                    })
                    .collect()
            });
            results.extend(chunk_results);
        }
        results
    } else {
        fetch_keys
            .iter()
            .cloned()
            .map(|key| {
                let result = fetch_one(&loaded.plan, key.clone(), &cache_root, use_cache);
                (key, result)
            })
            .collect()
    };

    loaded.fetched.clear();
    loaded.fetch_failures.clear();
    loaded.timing.fetch_ms_total = 0;

    for (key, entry) in fetch_results {
        match entry {
            Ok(bundle) => {
                loaded.timing.fetch_ms_total += bundle.fetch_ms;
                loaded.fetched.insert(bundle.key.clone(), bundle);
            }
            Err(err) => {
                loaded.fetch_failures.insert(key, err.to_string());
            }
        }
    }

    Ok(())
}

fn fetch_concurrency_for_source(source: SourceId, fetch_key_count: usize) -> usize {
    if fetch_key_count <= 1 {
        return fetch_key_count;
    }
    match source {
        SourceId::Nomads => fetch_key_count.min(3),
        _ => fetch_key_count,
    }
}

fn decode_execution_plan_into(
    loaded: &mut LoadedBundleSet,
    config: &BundleLoaderConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    // Phase 2: decode surface + pressure bundles. Bundles whose
    // underlying fetch failed are recorded in `bundle_failures` so
    // lanes can report "missing bundle" with the actual upstream error;
    // decode errors are caught and captured the same way.
    loaded.surface_decodes.clear();
    loaded.pressure_decodes.clear();
    loaded.bundle_failures.clear();
    loaded.timing.decode_surface_ms_total = 0;
    loaded.timing.decode_pressure_ms_total = 0;

    for bundle in &loaded.plan.bundles {
        let fetched_bytes = match loaded.fetched.get(&bundle.fetch_key()) {
            Some(bytes) => bytes,
            None => {
                let reason = loaded
                    .fetch_failures
                    .get(&bundle.fetch_key())
                    .cloned()
                    .unwrap_or_else(|| format!("planner missed fetch for bundle {}", bundle.id));
                loaded.bundle_failures.insert(bundle.id.clone(), reason);
                continue;
            }
        };
        match bundle.id.bundle {
            CanonicalBundleDescriptor::SurfaceAnalysis => {
                let cache_path =
                    decode_cache_path(&config.cache_root, &fetched_bytes.file.request, "surface");
                let start = Instant::now();
                match load_or_decode_surface_from_file(
                    &cache_path,
                    &fetched_bytes.file,
                    config.use_cache,
                ) {
                    Ok(decoded) => {
                        loaded.timing.decode_surface_ms_total += start.elapsed().as_millis();
                        loaded.surface_decodes.insert(bundle.id.clone(), decoded);
                    }
                    Err(err) => {
                        loaded.timing.decode_surface_ms_total += start.elapsed().as_millis();
                        loaded
                            .bundle_failures
                            .insert(bundle.id.clone(), err.to_string());
                    }
                }
            }
            CanonicalBundleDescriptor::PressureAnalysis => {
                let cache_path =
                    decode_cache_path(&config.cache_root, &fetched_bytes.file.request, "pressure");
                let start = Instant::now();
                let decode_outcome = load_or_decode_pressure_from_file_with_shape(
                    &cache_path,
                    &fetched_bytes.file,
                    config.use_cache,
                );
                loaded.timing.decode_pressure_ms_total += start.elapsed().as_millis();
                match decode_outcome {
                    Ok((decoded, shape)) => {
                        if let Some(matching_surface) = loaded.plan.bundle_for(
                            CanonicalBundleDescriptor::SurfaceAnalysis,
                            bundle.id.forecast_hour,
                        ) {
                            if let Some(matching) = loaded.surface_decodes.get(&matching_surface.id)
                            {
                                if let Err(err) = validate_pressure_decode_against_surface(
                                    &decoded,
                                    shape,
                                    matching.value.nx,
                                    matching.value.ny,
                                ) {
                                    loaded
                                        .bundle_failures
                                        .insert(bundle.id.clone(), err.to_string());
                                    continue;
                                }
                            }
                        }
                        loaded.pressure_decodes.insert(bundle.id.clone(), decoded);
                    }
                    Err(err) => {
                        loaded
                            .bundle_failures
                            .insert(bundle.id.clone(), err.to_string());
                    }
                }
            }
            CanonicalBundleDescriptor::NativeAnalysis => {
                // Native bundles surface as raw bytes only; kernels
                // (windowed UH/QPF, native composite-direct decode) walk
                // the GRIB messages on demand.
            }
        }
    }

    Ok(())
}

/// Materialize only the fetch stage of the execution plan. This is the
/// entry point for staged cache-warming flows that want planner-deduped
/// network/disk fetches without paying decode cost yet.
pub fn fetch_execution_plan(
    plan: ExecutionPlan,
    config: &BundleLoaderConfig,
) -> Result<LoadedBundleSet, Box<dyn std::error::Error>> {
    let mut loaded = empty_loaded_bundle_set(plan);
    fetch_execution_plan_into(&mut loaded, config)?;
    Ok(loaded)
}

/// Complete the decode stage for a previously fetched `LoadedBundleSet`.
///
/// The caller is responsible for passing a set whose `fetched` /
/// `fetch_failures` came from the same plan. Existing fetch outcomes are
/// preserved; surface/pressure decodes and bundle failures are rebuilt
/// from the fetched bytes using the current loader config.
pub fn decode_loaded_execution_plan(
    mut loaded: LoadedBundleSet,
    config: &BundleLoaderConfig,
) -> Result<LoadedBundleSet, Box<dyn std::error::Error>> {
    decode_execution_plan_into(&mut loaded, config)?;
    Ok(loaded)
}

/// Materialize the plan: fetch each unique fetch key once, then decode
/// surface and pressure bundles. Other bundle types (e.g. NativeAnalysis
/// at extra forecast hours used by windowed) are surfaced as raw bytes
/// only — kernels that need them call into `fetched_for` to access the
/// `FetchedModelFile`.
///
/// The fetch phase runs in parallel across distinct fetch keys, except
/// for NOMADS-sourced runs (which serialize to avoid the well-known
/// rate-limiting that the windowed lane has historically guarded
/// against).
///
/// Failure model: this loader is partial-success. A failed fetch or
/// decode on one `BundleFetchKey` / `CanonicalBundleId` is recorded in
/// `LoadedBundleSet::fetch_failures` / `bundle_failures` and the rest
/// of the plan still runs. Lane code inspects the returned set and
/// decides per product whether a missing bundle is fatal — critical on
/// the windowed lane, where a single 404 on one contributing hour
/// shouldn't nuke a 24-hour QPF sum when the other 23 hours are fine.
/// This function only returns `Err` for genuinely unrecoverable
/// conditions (currently there are none in steady state; the signature
/// is kept for forward flexibility and to avoid a breaking API change
/// for every caller).
pub fn load_execution_plan(
    plan: ExecutionPlan,
    config: &BundleLoaderConfig,
) -> Result<LoadedBundleSet, Box<dyn std::error::Error>> {
    let loaded = fetch_execution_plan(plan, config)?;
    decode_loaded_execution_plan(loaded, config)
}

fn build_fetch_request(
    plan: &ExecutionPlan,
    key: &BundleFetchKey,
) -> Result<FetchRequest, RustwxError> {
    let sharing_bundles: Vec<_> = plan
        .bundles
        .iter()
        .filter(|bundle| bundle.fetch_key() == *key)
        .collect();
    let variable_patterns = if key.source == SourceId::Nomads {
        Vec::new()
    } else {
        sharing_bundles
            .iter()
            .map(|bundle| {
                if bundle.aliases.iter().any(|alias| {
                    alias.variable_patterns.is_empty()
                        && alias
                            .logical_family
                            .as_deref()
                            .is_some_and(|family| family.starts_with("thermo-native:"))
                }) {
                    return Vec::new();
                }
                if let Some(patterns) = explicit_subset_patterns_for_bundle(bundle) {
                    return patterns;
                }
                let mut patterns = crate::gridded::bundle_fetch_variable_patterns(
                    bundle.id.model,
                    bundle.id.bundle,
                    bundle.resolved.native_product.as_str(),
                );
                for alias in &bundle.aliases {
                    for pattern in &alias.variable_patterns {
                        if !patterns.contains(pattern) {
                            patterns.push(pattern.clone());
                        }
                    }
                }
                patterns
            })
            // Indexed subsetting is only safe when every consumer that
            // shares this physical GRIB explicitly declares a safe subset.
            // If even one bundle on the fetch key lacks a subset contract,
            // keep the whole-file path so we don't truncate sibling lanes.
            .try_fold(Vec::<String>::new(), |mut merged, patterns| {
                if patterns.is_empty() {
                    return None;
                }
                for pattern in patterns {
                    if !merged.contains(&pattern) {
                        merged.push(pattern);
                    }
                }
                Some(merged)
            })
            .unwrap_or_default()
    };
    Ok(FetchRequest {
        request: ModelRunRequest::new(
            key.model,
            key.cycle.clone(),
            key.forecast_hour,
            key.native_product.as_str(),
        )?,
        source_override: Some(key.source),
        variable_patterns,
    })
}

fn explicit_subset_patterns_for_bundle(bundle: &PlannedBundle) -> Option<Vec<String>> {
    if bundle.aliases.is_empty()
        || !bundle
            .aliases
            .iter()
            .all(|alias| !alias.variable_patterns.is_empty())
    {
        return None;
    }

    let mut patterns = Vec::new();
    for alias in &bundle.aliases {
        for pattern in &alias.variable_patterns {
            if !patterns.contains(pattern) {
                patterns.push(pattern.clone());
            }
        }
    }
    (!patterns.is_empty()).then_some(patterns)
}

/// Worker used by `load_execution_plan` to fetch a single bundle's
/// physical bytes. Returns a Send + Sync error so it composes with
/// `std::thread::scope`.
fn fetch_one(
    plan: &ExecutionPlan,
    key: BundleFetchKey,
    cache_root: &Path,
    use_cache: bool,
) -> Result<FetchedBundleBytes, Box<dyn std::error::Error + Send + Sync>> {
    let request = build_fetch_request(plan, &key)
        .map_err(|err| Box::<dyn std::error::Error + Send + Sync>::from(err.to_string()))?;
    let start = Instant::now();
    let cached = fetch_bytes_with_cache(&request, cache_root, use_cache)
        .map_err(|err| Box::<dyn std::error::Error + Send + Sync>::from(err.to_string()))?;
    let fetch_ms = start.elapsed().as_millis();
    let bytes = cached.result.bytes.clone();
    Ok(FetchedBundleBytes {
        key,
        file: FetchedModelFile {
            request,
            bytes,
            fetched: cached,
        },
        fetch_ms,
    })
}

/// Helper used by the per-lane reports: build a deduped list of
/// `(planned_family, BundleFetchKey)` pairs that captures every alias
/// that asked for each fetch.
pub fn planned_family_aliases_for(bundle: &PlannedBundle) -> Vec<String> {
    bundle.planned_family_slugs()
}

/// Convenience for callers that want to build a one-product plan with a
/// single requirement set (used by severe / ECAPE / single-direct
/// runners). The latest run is supplied externally because resolving the
/// "latest" sometimes requires network probes that the lane batches
/// already perform.
pub fn build_single_pair_plan(
    latest: &LatestRun,
    forecast_hour: u16,
    surface_override: Option<String>,
    pressure_override: Option<String>,
) -> ExecutionPlan {
    let mut builder = crate::planner::ExecutionPlanBuilder::new(latest, forecast_hour);
    let mut surface =
        BundleRequirement::new(CanonicalBundleDescriptor::SurfaceAnalysis, forecast_hour);
    if let Some(value) = surface_override {
        surface = surface.with_native_override(value);
    }
    let mut pressure =
        BundleRequirement::new(CanonicalBundleDescriptor::PressureAnalysis, forecast_hour);
    if let Some(value) = pressure_override {
        pressure = pressure.with_native_override(value);
    }
    builder.require_with_logical_family(
        &surface,
        Some(default_planned_family_slug(
            latest.model,
            CanonicalBundleDescriptor::SurfaceAnalysis,
        )),
    );
    builder.require_with_logical_family(
        &pressure,
        Some(default_planned_family_slug(
            latest.model,
            CanonicalBundleDescriptor::PressureAnalysis,
        )),
    );
    builder.build()
}

fn default_planned_family_slug(
    model: rustwx_core::ModelId,
    bundle: CanonicalBundleDescriptor,
) -> &'static str {
    default_bundle_product(model, bundle)
}

#[cfg(test)]
mod tests;

// Re-export the path helper so tests in other modules don't need to
// dive into gridded.rs internals.
#[doc(hidden)]
pub fn cache_root_decode_path(cache_root: &Path, fetch: &FetchRequest, name: &str) -> PathBuf {
    decode_cache_path(cache_root, fetch, name)
}

#[doc(hidden)]
#[allow(unused)]
fn _force_use_unused_imports(_: SourceId, _: CycleSpec) {}
