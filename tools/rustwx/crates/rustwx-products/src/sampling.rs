use crate::derived::{load_derived_sampled_fields_from_latest, required_derived_fetch_products};
use crate::direct::{load_direct_sampled_fields_from_latest, required_direct_fetch_products};
use crate::publication::PublishedFetchIdentity;
use crate::source::ProductSourceRoute;
use crate::spec::{
    ProductSpec, direct_product_specs, supported_derived_product_specs, windowed_product_specs,
};
use crate::windowed::{
    HrrrWindowedBlocker, HrrrWindowedProduct, load_windowed_sampled_fields_from_latest,
    required_windowed_fetch_products,
};
use rustwx_core::{
    CycleSpec, FieldAreaSummary, FieldAreaSummaryMethod, FieldPointSample, FieldPointSampleMethod,
    FieldSelector, GeoPoint, GeoPolygon, ModelId, ProductId, ProductKey, ProductKeyMetadata,
    ProductKind, SourceId,
};
use rustwx_models::{LatestRun, latest_available_run_for_products_at_forecast_hour, model_summary};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

pub const PRODUCT_SAMPLING_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PreparedRunMetadata {
    pub model: ModelId,
    pub date_yyyymmdd: String,
    pub cycle_utc: u8,
    pub forecast_hour: u16,
    pub source: SourceId,
}

impl PreparedRunMetadata {
    pub fn from_latest(latest: &LatestRun, forecast_hour: u16) -> Self {
        Self {
            model: latest.model,
            date_yyyymmdd: latest.cycle.date_yyyymmdd.clone(),
            cycle_utc: latest.cycle.hour_utc,
            forecast_hour,
            source: latest.source,
        }
    }
}

fn default_point_sample_method() -> FieldPointSampleMethod {
    FieldPointSampleMethod::InverseDistance4
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductSamplingRunRequest {
    pub model: ModelId,
    pub date_yyyymmdd: String,
    pub cycle_override_utc: Option<u8>,
    pub forecast_hour: u16,
    pub source: SourceId,
    pub cache_root: PathBuf,
    pub use_cache: bool,
    pub product_slugs: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductPointSamplingRequest {
    #[serde(flatten)]
    pub run: ProductSamplingRunRequest,
    pub point: GeoPoint,
    #[serde(default = "default_point_sample_method")]
    pub method: FieldPointSampleMethod,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductAreaSummaryRequest {
    #[serde(flatten)]
    pub run: ProductSamplingRunRequest,
    pub polygon: GeoPolygon,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampledProductDescriptor {
    pub requested_slug: String,
    pub canonical_id: ProductId,
    pub title: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub product_metadata: Option<ProductKeyMetadata>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_route: Option<ProductSourceRoute>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub field_selector: Option<FieldSelector>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductPointSampleResult {
    pub product: SampledProductDescriptor,
    pub sampled_field: ProductKey,
    pub units: String,
    pub sample: FieldPointSample,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub input_fetches: Vec<PublishedFetchIdentity>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductAreaSummaryResult {
    pub product: SampledProductDescriptor,
    pub sampled_field: ProductKey,
    pub units: String,
    pub summary: FieldAreaSummary,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub input_fetches: Vec<PublishedFetchIdentity>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductSamplingBlocker {
    pub requested_slug: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub canonical_id: Option<ProductId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_route: Option<ProductSourceRoute>,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductPointSamplingReport {
    pub schema_version: u32,
    pub run: PreparedRunMetadata,
    pub point: GeoPoint,
    pub method: FieldPointSampleMethod,
    pub results: Vec<ProductPointSampleResult>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub blockers: Vec<ProductSamplingBlocker>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductAreaSummaryReport {
    pub schema_version: u32,
    pub run: PreparedRunMetadata,
    pub polygon: GeoPolygon,
    pub method: FieldAreaSummaryMethod,
    pub results: Vec<ProductAreaSummaryResult>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub blockers: Vec<ProductSamplingBlocker>,
}

#[derive(Debug, Clone)]
struct ResolvedSamplingTarget {
    requested_slug: String,
    spec: ProductSpec,
    windowed_product: Option<HrrrWindowedProduct>,
}

impl ResolvedSamplingTarget {
    fn key(&self) -> String {
        format!("{}:{}", self.spec.kind.as_str(), self.spec.slug)
    }

    fn descriptor(
        &self,
        source_route: Option<ProductSourceRoute>,
        field_selector: Option<FieldSelector>,
    ) -> SampledProductDescriptor {
        SampledProductDescriptor {
            requested_slug: self.requested_slug.clone(),
            canonical_id: self.spec.id.clone(),
            title: self.spec.title.clone(),
            product_metadata: self.spec.product_metadata.clone(),
            source_route,
            field_selector,
        }
    }

    fn blocker(
        &self,
        reason: impl Into<String>,
        source_route: Option<ProductSourceRoute>,
    ) -> ProductSamplingBlocker {
        ProductSamplingBlocker {
            requested_slug: self.requested_slug.clone(),
            canonical_id: Some(self.spec.id.clone()),
            source_route,
            reason: reason.into(),
        }
    }
}

pub fn sample_products_at_point(
    request: &ProductPointSamplingRequest,
) -> Result<ProductPointSamplingReport, Box<dyn std::error::Error>> {
    let (resolved, mut blockers) =
        resolve_sampling_targets(request.run.model, &request.run.product_slugs);
    let latest = resolve_sampling_latest(&request.run, &resolved)?;
    let direct_targets = targets_by_kind(&resolved, ProductKind::Direct);
    let derived_targets = targets_by_kind(&resolved, ProductKind::Derived);
    let windowed_targets = targets_by_kind(&resolved, ProductKind::Windowed);

    let mut results = Vec::new();

    if !direct_targets.is_empty() {
        let slugs = direct_targets.keys().cloned().collect::<Vec<_>>();
        let sampled = load_direct_sampled_fields_from_latest(
            &latest,
            request.run.forecast_hour,
            &request.run.cache_root,
            request.run.use_cache,
            &slugs,
        )?;
        for field in sampled.fields {
            if let Some(target) = direct_targets.get(&field.recipe_slug) {
                results.push(ProductPointSampleResult {
                    product: target.descriptor(Some(field.source_route), field.field_selector),
                    sampled_field: field.field.product.clone(),
                    units: field.field.units.clone(),
                    sample: field.field.sample_point(request.point, request.method),
                    input_fetches: field.input_fetches,
                });
            }
        }
        blockers.extend(sampled.blockers.into_iter().map(|blocker| {
            if let Some(target) = direct_targets.get(&blocker.recipe_slug) {
                target.blocker(blocker.reason, Some(direct_route_for(target)))
            } else {
                ProductSamplingBlocker {
                    requested_slug: blocker.recipe_slug,
                    canonical_id: None,
                    source_route: None,
                    reason: blocker.reason,
                }
            }
        }));
    }

    if !derived_targets.is_empty() {
        let slugs = derived_targets.keys().cloned().collect::<Vec<_>>();
        let sampled = load_derived_sampled_fields_from_latest(
            &latest,
            request.run.forecast_hour,
            &request.run.cache_root,
            request.run.use_cache,
            &slugs,
        )?;
        for field in sampled.fields {
            if let Some(target) = derived_targets.get(&field.recipe_slug) {
                results.push(ProductPointSampleResult {
                    product: target.descriptor(Some(field.source_route), None),
                    sampled_field: field.field.product.clone(),
                    units: field.field.units.clone(),
                    sample: field.field.sample_point(request.point, request.method),
                    input_fetches: field.input_fetches,
                });
            }
        }
        blockers.extend(sampled.blockers.into_iter().map(|blocker| {
            if let Some(target) = derived_targets.get(&blocker.recipe_slug) {
                target.blocker(blocker.reason, Some(blocker.source_route))
            } else {
                ProductSamplingBlocker {
                    requested_slug: blocker.recipe_slug,
                    canonical_id: None,
                    source_route: Some(blocker.source_route),
                    reason: blocker.reason,
                }
            }
        }));
    }

    if !windowed_targets.is_empty() {
        let products = windowed_targets
            .values()
            .filter_map(|target| target.windowed_product)
            .collect::<Vec<_>>();
        let sampled = load_windowed_sampled_fields_from_latest(
            &latest,
            request.run.forecast_hour,
            &request.run.cache_root,
            request.run.use_cache,
            &products,
        )?;
        for field in sampled.fields {
            if let Some(target) = windowed_targets.get(field.product.slug()) {
                results.push(ProductPointSampleResult {
                    product: target.descriptor(None, None),
                    sampled_field: field.field.product.clone(),
                    units: field.field.units.clone(),
                    sample: field.field.sample_point(request.point, request.method),
                    input_fetches: field.input_fetches,
                });
            }
        }
        blockers.extend(
            sampled
                .blockers
                .into_iter()
                .map(|blocker| windowed_blocker_to_report(&windowed_targets, blocker)),
        );
    }

    Ok(ProductPointSamplingReport {
        schema_version: PRODUCT_SAMPLING_SCHEMA_VERSION,
        run: PreparedRunMetadata::from_latest(&latest, request.run.forecast_hour),
        point: request.point,
        method: request.method,
        results,
        blockers,
    })
}

pub fn summarize_products_over_polygon(
    request: &ProductAreaSummaryRequest,
) -> Result<ProductAreaSummaryReport, Box<dyn std::error::Error>> {
    let (resolved, mut blockers) =
        resolve_sampling_targets(request.run.model, &request.run.product_slugs);
    let latest = resolve_sampling_latest(&request.run, &resolved)?;
    let direct_targets = targets_by_kind(&resolved, ProductKind::Direct);
    let derived_targets = targets_by_kind(&resolved, ProductKind::Derived);
    let windowed_targets = targets_by_kind(&resolved, ProductKind::Windowed);

    let mut results = Vec::new();

    if !direct_targets.is_empty() {
        let slugs = direct_targets.keys().cloned().collect::<Vec<_>>();
        let sampled = load_direct_sampled_fields_from_latest(
            &latest,
            request.run.forecast_hour,
            &request.run.cache_root,
            request.run.use_cache,
            &slugs,
        )?;
        for field in sampled.fields {
            if let Some(target) = direct_targets.get(&field.recipe_slug) {
                results.push(ProductAreaSummaryResult {
                    product: target.descriptor(Some(field.source_route), field.field_selector),
                    sampled_field: field.field.product.clone(),
                    units: field.field.units.clone(),
                    summary: field.field.summarize_polygon(&request.polygon),
                    input_fetches: field.input_fetches,
                });
            }
        }
        blockers.extend(sampled.blockers.into_iter().map(|blocker| {
            if let Some(target) = direct_targets.get(&blocker.recipe_slug) {
                target.blocker(blocker.reason, Some(direct_route_for(target)))
            } else {
                ProductSamplingBlocker {
                    requested_slug: blocker.recipe_slug,
                    canonical_id: None,
                    source_route: None,
                    reason: blocker.reason,
                }
            }
        }));
    }

    if !derived_targets.is_empty() {
        let slugs = derived_targets.keys().cloned().collect::<Vec<_>>();
        let sampled = load_derived_sampled_fields_from_latest(
            &latest,
            request.run.forecast_hour,
            &request.run.cache_root,
            request.run.use_cache,
            &slugs,
        )?;
        for field in sampled.fields {
            if let Some(target) = derived_targets.get(&field.recipe_slug) {
                results.push(ProductAreaSummaryResult {
                    product: target.descriptor(Some(field.source_route), None),
                    sampled_field: field.field.product.clone(),
                    units: field.field.units.clone(),
                    summary: field.field.summarize_polygon(&request.polygon),
                    input_fetches: field.input_fetches,
                });
            }
        }
        blockers.extend(sampled.blockers.into_iter().map(|blocker| {
            if let Some(target) = derived_targets.get(&blocker.recipe_slug) {
                target.blocker(blocker.reason, Some(blocker.source_route))
            } else {
                ProductSamplingBlocker {
                    requested_slug: blocker.recipe_slug,
                    canonical_id: None,
                    source_route: Some(blocker.source_route),
                    reason: blocker.reason,
                }
            }
        }));
    }

    if !windowed_targets.is_empty() {
        let products = windowed_targets
            .values()
            .filter_map(|target| target.windowed_product)
            .collect::<Vec<_>>();
        let sampled = load_windowed_sampled_fields_from_latest(
            &latest,
            request.run.forecast_hour,
            &request.run.cache_root,
            request.run.use_cache,
            &products,
        )?;
        for field in sampled.fields {
            if let Some(target) = windowed_targets.get(field.product.slug()) {
                results.push(ProductAreaSummaryResult {
                    product: target.descriptor(None, None),
                    sampled_field: field.field.product.clone(),
                    units: field.field.units.clone(),
                    summary: field.field.summarize_polygon(&request.polygon),
                    input_fetches: field.input_fetches,
                });
            }
        }
        blockers.extend(
            sampled
                .blockers
                .into_iter()
                .map(|blocker| windowed_blocker_to_report(&windowed_targets, blocker)),
        );
    }

    Ok(ProductAreaSummaryReport {
        schema_version: PRODUCT_SAMPLING_SCHEMA_VERSION,
        run: PreparedRunMetadata::from_latest(&latest, request.run.forecast_hour),
        polygon: request.polygon.clone(),
        method: FieldAreaSummaryMethod::CellCentersWithinPolygon,
        results,
        blockers,
    })
}

fn resolve_sampling_targets(
    model: ModelId,
    product_slugs: &[String],
) -> (Vec<ResolvedSamplingTarget>, Vec<ProductSamplingBlocker>) {
    let direct_specs = direct_product_specs();
    let derived_specs = supported_derived_product_specs();
    let windowed_specs = (model == ModelId::Hrrr)
        .then(windowed_product_specs)
        .unwrap_or_default();
    let mut resolved = Vec::new();
    let mut blockers = Vec::new();
    let mut seen = BTreeSet::<String>::new();

    for requested_slug in product_slugs {
        let resolved_target = resolve_target_from_specs(requested_slug, &direct_specs, None)
            .or_else(|| resolve_target_from_specs(requested_slug, &derived_specs, None))
            .or_else(|| {
                resolve_target_from_specs(
                    requested_slug,
                    &windowed_specs,
                    parse_windowed_product_slug(requested_slug),
                )
            });
        match resolved_target {
            Some(target) => {
                if seen.insert(target.key()) {
                    resolved.push(target);
                }
            }
            None => blockers.push(ProductSamplingBlocker {
                requested_slug: requested_slug.clone(),
                canonical_id: None,
                source_route: None,
                reason: format!(
                    "unsupported sampling product '{}' for model {}",
                    requested_slug, model
                ),
            }),
        }
    }

    (resolved, blockers)
}

fn resolve_target_from_specs(
    requested_slug: &str,
    specs: &[ProductSpec],
    windowed_product: Option<HrrrWindowedProduct>,
) -> Option<ResolvedSamplingTarget> {
    let wanted = normalize_sampling_slug(requested_slug);
    specs
        .iter()
        .find(|spec| {
            normalize_sampling_slug(&spec.slug) == wanted
                || spec
                    .aliases
                    .iter()
                    .any(|alias| normalize_sampling_slug(&alias.slug) == wanted)
        })
        .cloned()
        .map(|spec| ResolvedSamplingTarget {
            requested_slug: requested_slug.to_string(),
            windowed_product: windowed_product.or_else(|| parse_windowed_product_slug(&spec.slug)),
            spec,
        })
}

fn targets_by_kind(
    resolved: &[ResolvedSamplingTarget],
    kind: ProductKind,
) -> BTreeMap<String, ResolvedSamplingTarget> {
    resolved
        .iter()
        .filter(|target| target.spec.kind == kind)
        .map(|target| (target.spec.slug.clone(), target.clone()))
        .collect()
}

fn resolve_sampling_latest(
    request: &ProductSamplingRunRequest,
    resolved: &[ResolvedSamplingTarget],
) -> Result<LatestRun, Box<dyn std::error::Error>> {
    if let Some(cycle_hour) = request.cycle_override_utc {
        return Ok(LatestRun {
            model: request.model,
            cycle: CycleSpec::new(&request.date_yyyymmdd, cycle_hour)?,
            source: request.source,
        });
    }

    let direct_slugs = resolved
        .iter()
        .filter(|target| target.spec.kind == ProductKind::Direct)
        .map(|target| target.spec.slug.clone())
        .collect::<Vec<_>>();
    let derived_slugs = resolved
        .iter()
        .filter(|target| target.spec.kind == ProductKind::Derived)
        .map(|target| target.spec.slug.clone())
        .collect::<Vec<_>>();
    let windowed_products = resolved
        .iter()
        .filter(|target| target.spec.kind == ProductKind::Windowed)
        .filter_map(|target| target.windowed_product)
        .collect::<Vec<_>>();

    let mut required_products = BTreeSet::<String>::new();
    required_products.extend(required_direct_fetch_products(
        request.model,
        &direct_slugs,
    )?);
    required_products.extend(required_derived_fetch_products(
        request.model,
        &derived_slugs,
    )?);
    required_products.extend(required_windowed_fetch_products(&windowed_products));
    if required_products.is_empty() {
        required_products.insert(model_summary(request.model).default_product.to_string());
    }
    let required_refs = required_products
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();
    Ok(latest_available_run_for_products_at_forecast_hour(
        request.model,
        Some(request.source),
        &request.date_yyyymmdd,
        &required_refs,
        request.forecast_hour,
    )?)
}

fn windowed_blocker_to_report(
    targets: &BTreeMap<String, ResolvedSamplingTarget>,
    blocker: HrrrWindowedBlocker,
) -> ProductSamplingBlocker {
    if let Some(target) = targets.get(blocker.product.slug()) {
        ProductSamplingBlocker {
            requested_slug: target.requested_slug.clone(),
            canonical_id: Some(target.spec.id.clone()),
            source_route: None,
            reason: blocker.reason,
        }
    } else {
        ProductSamplingBlocker {
            requested_slug: blocker.product.slug().to_string(),
            canonical_id: None,
            source_route: None,
            reason: blocker.reason,
        }
    }
}

fn direct_route_for(target: &ResolvedSamplingTarget) -> ProductSourceRoute {
    target
        .spec
        .product_metadata
        .as_ref()
        .and_then(|metadata| metadata.provenance.as_ref())
        .map(|_| crate::source::direct_route_for_recipe_slug(&target.spec.slug))
        .unwrap_or_else(|| crate::source::direct_route_for_recipe_slug(&target.spec.slug))
}

fn parse_windowed_product_slug(slug: &str) -> Option<HrrrWindowedProduct> {
    let wanted = normalize_sampling_slug(slug);
    HrrrWindowedProduct::supported_products()
        .iter()
        .copied()
        .find(|product| normalize_sampling_slug(product.slug()) == wanted)
}

fn normalize_sampling_slug(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    let mut last_was_separator = false;
    for ch in value.trim().chars() {
        let normalized = ch.to_ascii_lowercase();
        if normalized.is_ascii_alphanumeric() {
            out.push(normalized);
            last_was_separator = false;
        } else if !last_was_separator {
            out.push('_');
            last_was_separator = true;
        }
    }
    out.trim_matches('_').to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolver_prefers_canonical_product_specs_and_aliases() {
        let (resolved, blockers) = resolve_sampling_targets(
            ModelId::Hrrr,
            &[
                "2m_heat_index".to_string(),
                "smoke_pm25_native".to_string(),
                "qpf_1h".to_string(),
            ],
        );
        assert!(blockers.is_empty());
        assert_eq!(resolved.len(), 3);
        assert!(resolved.iter().any(|target| {
            target.spec.kind == ProductKind::Derived && target.spec.slug == "heat_index_2m"
        }));
        assert!(resolved.iter().any(|target| {
            target.spec.kind == ProductKind::Direct && target.spec.slug == "smoke_pm25_native"
        }));
        assert!(resolved.iter().any(|target| {
            target.spec.kind == ProductKind::Windowed && target.spec.slug == "qpf_1h"
        }));
    }

    #[test]
    fn normalize_sampling_slug_collapses_common_separator_variants() {
        assert_eq!(
            normalize_sampling_slug("500MB temperature height winds"),
            "500mb_temperature_height_winds"
        );
        assert_eq!(
            normalize_sampling_slug("500mb-temperature-height-winds"),
            "500mb_temperature_height_winds"
        );
    }
}
