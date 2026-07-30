use rustwx_core::{ModelId, ProductId, ProductKeyMetadata, ProductKind};
use rustwx_models::{
    PlotRecipeFetchMode, built_in_models, plot_recipe_fetch_blockers, plot_recipe_fetch_plan,
    selector_supported_for_model,
};
use rustwx_render::{ProductMaturity, ProductSemanticFlag};
use serde::{Deserialize, Serialize};

use crate::derived::is_heavy_derived_recipe_slug;
use crate::source::{ProductSourceRoute, direct_route_for_recipe_slug};
use crate::spec::{
    ProductSpec, blocked_derived_product_specs, direct_product_specs, heavy_product_specs,
    supported_derived_product_specs, windowed_product_specs,
};
use crate::thermo_native::{NativeSemantics, native_candidate_for_slug};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProductCatalogKind {
    Direct,
    Derived,
    Heavy,
    Windowed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProductCatalogStatus {
    Supported,
    Partial,
    Blocked,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProductTargetStatus {
    Supported,
    Blocked,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductCatalogSummary {
    pub total_entries: usize,
    pub direct_entries: usize,
    pub derived_entries: usize,
    pub heavy_entries: usize,
    pub windowed_entries: usize,
    pub supported_entries: usize,
    pub partial_entries: usize,
    pub blocked_entries: usize,
    pub experimental_entries: usize,
    pub proof_entries: usize,
    pub proxy_entries: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductTargetSupport {
    pub target: String,
    pub model: Option<ModelId>,
    pub status: ProductTargetStatus,
    pub fetch_mode: Option<PlotRecipeFetchMode>,
    pub grib_product: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub source_routes: Vec<ProductSourceRoute>,
    pub blockers: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductCatalogAlias {
    pub id: ProductId,
    pub slug: String,
    pub title: String,
    pub note: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductCatalogEntry {
    pub id: ProductId,
    pub slug: String,
    pub title: String,
    pub kind: ProductCatalogKind,
    pub status: ProductCatalogStatus,
    pub product_metadata: Option<ProductKeyMetadata>,
    pub maturity: ProductMaturity,
    pub flags: Vec<ProductSemanticFlag>,
    pub experimental: bool,
    pub render_style: Option<String>,
    pub runners: Vec<String>,
    pub aliases: Vec<ProductCatalogAlias>,
    pub notes: Vec<String>,
    pub support: Vec<ProductTargetSupport>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupportedProductsCatalog {
    pub summary: ProductCatalogSummary,
    pub direct: Vec<ProductCatalogEntry>,
    pub derived: Vec<ProductCatalogEntry>,
    pub heavy: Vec<ProductCatalogEntry>,
    pub windowed: Vec<ProductCatalogEntry>,
}

pub fn build_supported_products_catalog() -> SupportedProductsCatalog {
    let direct = build_direct_entries();
    let derived = build_derived_entries();
    let heavy = build_heavy_entries();
    let windowed = build_windowed_entries();

    let all = direct
        .iter()
        .chain(derived.iter())
        .chain(heavy.iter())
        .chain(windowed.iter());

    let mut summary = ProductCatalogSummary {
        total_entries: direct.len() + derived.len() + heavy.len() + windowed.len(),
        direct_entries: direct.len(),
        derived_entries: derived.len(),
        heavy_entries: heavy.len(),
        windowed_entries: windowed.len(),
        supported_entries: 0,
        partial_entries: 0,
        blocked_entries: 0,
        experimental_entries: 0,
        proof_entries: 0,
        proxy_entries: 0,
    };
    for entry in all {
        match entry.status {
            ProductCatalogStatus::Supported => summary.supported_entries += 1,
            ProductCatalogStatus::Partial => summary.partial_entries += 1,
            ProductCatalogStatus::Blocked => summary.blocked_entries += 1,
        }
        match entry.maturity {
            ProductMaturity::Operational => {}
            ProductMaturity::Experimental => summary.experimental_entries += 1,
            ProductMaturity::Proof => summary.proof_entries += 1,
        }
        if entry.flags.contains(&ProductSemanticFlag::Proxy) {
            summary.proxy_entries += 1;
        }
    }

    SupportedProductsCatalog {
        summary,
        direct,
        derived,
        heavy,
        windowed,
    }
}

fn build_direct_entries() -> Vec<ProductCatalogEntry> {
    direct_product_specs()
        .into_iter()
        .map(|spec| {
            let support = built_in_models()
                .iter()
                .map(|model| match plot_recipe_fetch_plan(&spec.slug, model.id) {
                    Ok(plan) => ProductTargetSupport {
                        target: model.id.to_string(),
                        model: Some(model.id),
                        status: ProductTargetStatus::Supported,
                        fetch_mode: Some(plan.fetch_mode),
                        grib_product: Some(plan.product.to_string()),
                        source_routes: vec![direct_route_for_recipe_slug(&spec.slug)],
                        blockers: Vec::new(),
                    },
                    Err(_) => ProductTargetSupport {
                        target: model.id.to_string(),
                        model: Some(model.id),
                        status: ProductTargetStatus::Blocked,
                        fetch_mode: None,
                        grib_product: None,
                        source_routes: Vec::new(),
                        blockers: plot_recipe_fetch_blockers(&spec.slug, model.id)
                            .unwrap_or_default()
                            .into_iter()
                            .map(|blocker| format!("{}: {}", blocker.field_label, blocker.reason))
                            .collect(),
                    },
                })
                .collect::<Vec<_>>();

            let mut runners = vec!["plot_recipe_proof".to_string()];
            if support
                .iter()
                .any(|target| matches!(target.status, ProductTargetStatus::Supported))
            {
                runners.push("direct_batch".to_string());
                runners.push("non_ecape_hour".to_string());
            }
            if support.iter().any(|target| {
                target.model == Some(ModelId::Hrrr)
                    && matches!(target.status, ProductTargetStatus::Supported)
            }) {
                runners.push("hrrr_direct_batch".to_string());
                runners.push("hrrr_non_ecape_hour".to_string());
            }

            build_catalog_entry(spec, collapse_entry_status(&support), runners, support)
        })
        .collect()
}

fn build_derived_entries() -> Vec<ProductCatalogEntry> {
    let mut entries = supported_derived_product_specs()
        .into_iter()
        .map(|spec| {
            let heavy = is_heavy_derived_recipe_slug(&spec.slug);
            let support = built_in_models()
                .iter()
                .map(|model| {
                    let mut source_routes = vec![ProductSourceRoute::CanonicalDerived];
                    if let Some(candidate) = native_candidate_for_slug(model.id, &spec.slug) {
                        source_routes.push(match candidate.semantics {
                            NativeSemantics::ExactEquivalent => ProductSourceRoute::NativeExact,
                            NativeSemantics::ProxyEquivalent => ProductSourceRoute::NativeProxy,
                        });
                    }
                    ProductTargetSupport {
                        target: model.id.to_string(),
                        model: Some(model.id),
                        status: ProductTargetStatus::Supported,
                        fetch_mode: None,
                        grib_product: None,
                        source_routes,
                        blockers: Vec::new(),
                    }
                })
                .collect::<Vec<_>>();
            let mut runners = vec!["derived_batch".to_string()];
            if support
                .iter()
                .any(|target| target.model == Some(ModelId::Hrrr))
            {
                runners.push("hrrr_derived_batch".to_string());
            }
            if !heavy
                && support
                    .iter()
                    .any(|target| target.model == Some(ModelId::Hrrr))
            {
                runners.push("hrrr_non_ecape_hour".to_string());
            }
            if !heavy
                && support
                    .iter()
                    .any(|target| matches!(target.status, ProductTargetStatus::Supported))
            {
                runners.push("non_ecape_hour".to_string());
            }
            build_catalog_entry(spec, ProductCatalogStatus::Supported, runners, support)
        })
        .collect::<Vec<_>>();

    entries.extend(blocked_derived_product_specs().into_iter().map(|spec| {
        let blockers = spec.blocked_reasons.clone();
        let blocked_support = built_in_models()
            .iter()
            .map(|model| ProductTargetSupport {
                target: model.id.to_string(),
                model: Some(model.id),
                status: ProductTargetStatus::Blocked,
                fetch_mode: None,
                grib_product: None,
                source_routes: Vec::new(),
                blockers: blockers.clone(),
            })
            .collect::<Vec<_>>();
        build_catalog_entry(
            spec,
            ProductCatalogStatus::Blocked,
            vec![
                "derived_batch".to_string(),
                "hrrr_derived_batch".to_string(),
            ],
            blocked_support,
        )
    }));

    entries
}

fn build_heavy_entries() -> Vec<ProductCatalogEntry> {
    heavy_product_specs()
        .into_iter()
        .map(|spec| {
            let (runners, support) = match spec.slug.as_str() {
                "severe_proof_panel" => {
                    let mut runners =
                        vec!["severe_batch".to_string(), "heavy_panel_hour".to_string()];
                    let support = built_in_models()
                        .iter()
                        .map(|model| ProductTargetSupport {
                            target: model.id.to_string(),
                            model: Some(model.id),
                            status: ProductTargetStatus::Supported,
                            fetch_mode: None,
                            grib_product: None,
                            source_routes: vec![ProductSourceRoute::CanonicalDerived],
                            blockers: Vec::new(),
                        })
                        .collect::<Vec<_>>();
                    if support
                        .iter()
                        .any(|target| target.model == Some(ModelId::Hrrr))
                    {
                        runners.push("hrrr_batch".to_string());
                        runners.push("hrrr_severe_proof".to_string());
                    }
                    (runners, support)
                }
                _ => (
                    vec!["hrrr_batch".to_string()],
                    vec![ProductTargetSupport {
                        target: ModelId::Hrrr.to_string(),
                        model: Some(ModelId::Hrrr),
                        status: ProductTargetStatus::Supported,
                        fetch_mode: None,
                        grib_product: None,
                        source_routes: vec![ProductSourceRoute::CanonicalDerived],
                        blockers: Vec::new(),
                    }],
                ),
            };
            build_catalog_entry(spec, ProductCatalogStatus::Supported, runners, support)
        })
        .collect()
}

fn build_windowed_entries() -> Vec<ProductCatalogEntry> {
    use crate::windowed::HrrrWindowedProduct;

    windowed_product_specs()
        .into_iter()
        .map(|spec| {
            // QPF windows fold 1-hour APCP bucket increments, a fact
            // about the source's precipitation accumulation windows
            // that the input-field selector cannot express; their
            // support lists stay curated.  Every NON-QPF windowed
            // entry derives per-source support from its declared
            // input-field requirements (gpuwm vendor divergence: no
            // hardcoded HRRR-only capability rows -- a source that
            // serves the input fields is supported, one that does not
            // is blocked with the missing fields named).
            let support = if spec.slug == "qpf_total" {
                qpf_total_windowed_model_support()
            } else {
                match HrrrWindowedProduct::from_slug(&spec.slug) {
                    Some(product) if !product.is_qpf() => {
                        windowed_input_requirement_support(product)
                    }
                    _ => vec![windowed_model_support(ModelId::Hrrr)],
                }
            };
            build_catalog_entry(
                spec,
                collapse_entry_status(&support),
                vec![
                    "hrrr_windowed_batch".to_string(),
                    "hrrr_non_ecape_hour".to_string(),
                    "non_ecape_hour".to_string(),
                ],
                support,
            )
        })
        .collect()
}

fn windowed_model_support(model: ModelId) -> ProductTargetSupport {
    ProductTargetSupport {
        target: model.to_string(),
        model: Some(model),
        status: ProductTargetStatus::Supported,
        fetch_mode: None,
        grib_product: None,
        source_routes: vec![ProductSourceRoute::CheapDerived],
        blockers: Vec::new(),
    }
}

/// Input-requirement gate for a non-QPF windowed product: one support
/// row per built-in source, Supported exactly when the source's files
/// serve every input-field selector the window folds, Blocked with the
/// unserved fields named otherwise.  Model identity never appears as a
/// reason -- only fields do.
fn windowed_input_requirement_support(
    product: crate::windowed::HrrrWindowedProduct,
) -> Vec<ProductTargetSupport> {
    let selectors = product.input_selectors();
    built_in_models()
        .iter()
        .map(|model| {
            let blockers: Vec<String> = selectors
                .iter()
                .filter(|selector| !selector_supported_for_model(**selector, model.id))
                .map(|selector| {
                    format!(
                        "{}: input field not served by this source's files",
                        selector.key()
                    )
                })
                .collect();
            if blockers.is_empty() {
                windowed_model_support(model.id)
            } else {
                ProductTargetSupport {
                    target: model.id.to_string(),
                    model: Some(model.id),
                    status: ProductTargetStatus::Blocked,
                    fetch_mode: None,
                    grib_product: None,
                    source_routes: Vec::new(),
                    blockers,
                }
            }
        })
        .collect()
}

fn qpf_total_windowed_model_support() -> Vec<ProductTargetSupport> {
    [
        ModelId::Hrrr,
        ModelId::HrrrAk,
        ModelId::Gfs,
        ModelId::Gdas,
        ModelId::Gefs,
        ModelId::Aigfs,
        ModelId::Aigefs,
        ModelId::Rap,
        ModelId::Nam,
        ModelId::Hiresw,
        ModelId::Sref,
        ModelId::Nbm,
        ModelId::RrfsA,
    ]
    .into_iter()
    .map(windowed_model_support)
    .collect()
}

fn collapse_entry_status(support: &[ProductTargetSupport]) -> ProductCatalogStatus {
    let supported = support
        .iter()
        .filter(|target| matches!(target.status, ProductTargetStatus::Supported))
        .count();
    if supported == 0 {
        ProductCatalogStatus::Blocked
    } else if supported == support.len() {
        ProductCatalogStatus::Supported
    } else {
        ProductCatalogStatus::Partial
    }
}

fn build_catalog_entry(
    spec: ProductSpec,
    status: ProductCatalogStatus,
    runners: Vec<String>,
    support: Vec<ProductTargetSupport>,
) -> ProductCatalogEntry {
    let experimental = spec.experimental();
    ProductCatalogEntry {
        id: spec.id,
        slug: spec.slug,
        title: spec.title,
        kind: catalog_kind(spec.kind),
        status,
        product_metadata: spec.product_metadata,
        maturity: spec.maturity,
        flags: spec.flags,
        experimental,
        render_style: spec.render_style,
        runners,
        aliases: spec
            .aliases
            .into_iter()
            .map(|alias| ProductCatalogAlias {
                id: alias.id,
                slug: alias.slug,
                title: alias.title,
                note: alias.note,
            })
            .collect(),
        notes: spec.notes,
        support,
    }
}

fn catalog_kind(kind: ProductKind) -> ProductCatalogKind {
    match kind {
        ProductKind::Direct => ProductCatalogKind::Direct,
        ProductKind::Derived => ProductCatalogKind::Derived,
        ProductKind::Bundled => ProductCatalogKind::Heavy,
        ProductKind::Windowed => ProductCatalogKind::Windowed,
    }
}

#[cfg(test)]
mod tests;
