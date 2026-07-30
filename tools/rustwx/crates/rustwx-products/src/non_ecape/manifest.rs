use super::types::NonEcapeRequestedProducts;
use crate::derived::HrrrDerivedBatchReport;
use crate::direct::HrrrDirectBatchReport;
use crate::publication::{
    ArtifactPublicationState, PublishedArtifactRecord, PublishedFetchIdentity,
    RunPublicationManifest, artifact_identity_from_path,
};
use crate::windowed::{
    HrrrWindowedBatchReport, HrrrWindowedProduct, HrrrWindowedRenderedProduct,
    collect_windowed_input_fetches, windowed_product_input_fetch_keys,
};
use rustwx_core::{ModelId, SourceId};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

pub(super) fn build_run_manifest(
    model: ModelId,
    request: &NonEcapeRequestedProducts,
    out_dir: &Path,
    run_slug: &str,
    date_yyyymmdd: &str,
    cycle_utc: u8,
    forecast_hour: u16,
    source: SourceId,
    domain_slug: &str,
) -> RunPublicationManifest {
    let mut seen = HashSet::new();
    let mut artifacts = Vec::new();

    for slug in &request.direct_recipe_slugs {
        let key = direct_artifact_key(slug);
        if seen.insert(key.clone()) {
            artifacts.push(PublishedArtifactRecord::planned(
                key,
                expected_output_relative_path(
                    model,
                    date_yyyymmdd,
                    cycle_utc,
                    forecast_hour,
                    domain_slug,
                    slug,
                ),
            ));
        }
    }

    for slug in &request.derived_recipe_slugs {
        let key = derived_artifact_key(slug);
        if seen.insert(key.clone()) {
            artifacts.push(PublishedArtifactRecord::planned(
                key,
                expected_output_relative_path(
                    model,
                    date_yyyymmdd,
                    cycle_utc,
                    forecast_hour,
                    domain_slug,
                    slug,
                ),
            ));
        }
    }

    for product in &request.windowed_products {
        let slug = product.slug();
        let key = windowed_artifact_key(slug);
        if seen.insert(key.clone()) {
            artifacts.push(PublishedArtifactRecord::planned(
                key,
                expected_output_relative_path(
                    model,
                    date_yyyymmdd,
                    cycle_utc,
                    forecast_hour,
                    domain_slug,
                    slug,
                ),
            ));
        }
    }

    let runner_name = if model == ModelId::Hrrr {
        "hrrr_non_ecape_hour".to_string()
    } else {
        format!("{}_non_ecape_hour", model.as_str().replace('-', "_"))
    };
    RunPublicationManifest::new(&runner_name, run_slug.to_string(), out_dir.to_path_buf())
        .with_run_metadata(
            model.as_str(),
            date_yyyymmdd,
            cycle_utc,
            forecast_hour,
            source.as_str(),
            domain_slug,
        )
        .with_artifacts(artifacts)
}

fn expected_output_relative_path(
    model: ModelId,
    date_yyyymmdd: &str,
    cycle_utc: u8,
    forecast_hour: u16,
    domain_slug: &str,
    product_slug: &str,
) -> PathBuf {
    PathBuf::from(format!(
        "rustwx_{}_{}_{}z_f{:03}_{}_{}.png",
        model.as_str().replace('-', "_"),
        date_yyyymmdd,
        cycle_utc,
        forecast_hour,
        domain_slug,
        product_slug
    ))
}

fn direct_artifact_key(slug: &str) -> String {
    format!("direct:{slug}")
}

fn derived_artifact_key(slug: &str) -> String {
    format!("derived:{slug}")
}

fn windowed_artifact_key(slug: &str) -> String {
    format!("windowed:{slug}")
}

pub(super) fn apply_direct_manifest_updates(
    manifest: &mut RunPublicationManifest,
    direct: &Option<HrrrDirectBatchReport>,
) {
    let Some(report) = direct else {
        return;
    };
    for recipe in &report.recipes {
        manifest.update_artifact_state(
            &direct_artifact_key(&recipe.recipe_slug),
            ArtifactPublicationState::Complete,
            Some(format!(
                "source_route={} planned_family={} fetched_family={} resolved_source={} resolved_url={}",
                recipe.source_route.as_str(),
                recipe.grib_product,
                recipe.fetched_grib_product,
                recipe.resolved_source,
                recipe.resolved_url
            )),
        );
        manifest.update_artifact_identity(
            &direct_artifact_key(&recipe.recipe_slug),
            recipe.content_identity.clone(),
        );
        manifest.update_artifact_input_fetch_keys(
            &direct_artifact_key(&recipe.recipe_slug),
            recipe.input_fetch_keys.clone(),
        );
    }
    for blocker in &report.blockers {
        manifest.update_artifact_state(
            &direct_artifact_key(&blocker.recipe_slug),
            ArtifactPublicationState::Blocked,
            Some(blocker.reason.clone()),
        );
    }
}

pub(super) fn apply_derived_manifest_updates(
    manifest: &mut RunPublicationManifest,
    derived: &Option<HrrrDerivedBatchReport>,
) {
    let Some(report) = derived else {
        return;
    };
    for recipe in &report.recipes {
        let detail = if let Some(fetch_decode) = &report.shared_timing.fetch_decode {
            format!(
                "source_mode={} source_route={} shared_surface planned_family={} fetched_family={} resolved_source={}; shared_pressure planned_family={} fetched_family={} resolved_source={}",
                report.source_mode.as_str(),
                recipe.source_route.as_str(),
                fetch_decode.surface_fetch.planned_product,
                fetch_decode.surface_fetch.fetched_product,
                fetch_decode.surface_fetch.resolved_source,
                fetch_decode.pressure_fetch.planned_product,
                fetch_decode.pressure_fetch.fetched_product,
                fetch_decode.pressure_fetch.resolved_source
            )
        } else {
            format!(
                "source_mode={} source_route={} native_thermo_only native_extract_ms={} native_compare_ms={}",
                report.source_mode.as_str(),
                recipe.source_route.as_str(),
                report.shared_timing.native_extract_ms,
                report.shared_timing.native_compare_ms
            )
        };
        manifest.update_artifact_state(
            &derived_artifact_key(&recipe.recipe_slug),
            ArtifactPublicationState::Complete,
            Some(detail),
        );
        manifest.update_artifact_identity(
            &derived_artifact_key(&recipe.recipe_slug),
            recipe.content_identity.clone(),
        );
        manifest.update_artifact_input_fetch_keys(
            &derived_artifact_key(&recipe.recipe_slug),
            recipe.input_fetch_keys.clone(),
        );
    }
    for blocker in &report.blockers {
        manifest.update_artifact_state(
            &derived_artifact_key(&blocker.recipe_slug),
            ArtifactPublicationState::Blocked,
            Some(format!(
                "source_mode={} source_route={} {}",
                report.source_mode.as_str(),
                blocker.source_route.as_str(),
                blocker.reason
            )),
        );
    }
}

pub(super) fn apply_windowed_manifest_updates(
    manifest: &mut RunPublicationManifest,
    windowed: &Option<HrrrWindowedBatchReport>,
) {
    let Some(report) = windowed else {
        return;
    };
    for product in &report.products {
        let detail = windowed_artifact_detail(product, &report.shared_timing);
        manifest.update_artifact_state(
            &windowed_artifact_key(product.product.slug()),
            ArtifactPublicationState::Complete,
            Some(detail),
        );
        if let Ok(identity) = artifact_identity_from_path(&product.output_path) {
            manifest
                .update_artifact_identity(&windowed_artifact_key(product.product.slug()), identity);
        }
        let input_fetch_keys = windowed_product_input_fetch_keys(product, &report.shared_timing);
        if !input_fetch_keys.is_empty() {
            manifest.update_artifact_input_fetch_keys(
                &windowed_artifact_key(product.product.slug()),
                input_fetch_keys,
            );
        }
    }
    for blocker in &report.blockers {
        manifest.update_artifact_state(
            &windowed_artifact_key(blocker.product.slug()),
            ArtifactPublicationState::Blocked,
            Some(blocker.reason.clone()),
        );
    }
}

fn windowed_artifact_detail(
    product: &HrrrWindowedRenderedProduct,
    shared_timing: &crate::windowed::HrrrWindowedSharedTiming,
) -> String {
    let is_qpf = matches!(
        product.product,
        HrrrWindowedProduct::Qpf1h
            | HrrrWindowedProduct::Qpf6h
            | HrrrWindowedProduct::Qpf12h
            | HrrrWindowedProduct::Qpf24h
            | HrrrWindowedProduct::QpfTotal
    );
    let is_wind = matches!(
        product.product,
        HrrrWindowedProduct::Wind10m1hMax
            | HrrrWindowedProduct::Wind10mRunMax
            | HrrrWindowedProduct::Wind10m0to24hMax
            | HrrrWindowedProduct::Wind10m24to48hMax
            | HrrrWindowedProduct::Wind10m0to48hMax
    );
    let is_surface_snapshot = product.product.is_surface_snapshot();
    let fetches = windowed_runtime_fetches_for_product(product, shared_timing);
    let planned_family = fetches
        .first()
        .map(|fetch| fetch.planned_product.as_str())
        .unwrap_or(if is_qpf || is_wind || is_surface_snapshot {
            "sfc"
        } else {
            "nat"
        });
    let fetched_families = unique_join(fetches.iter().map(|fetch| fetch.fetched_product.as_str()));
    let resolved_sources = unique_join(fetches.iter().map(|fetch| fetch.resolved_source.as_str()));
    let hours = fetches
        .iter()
        .map(|fetch| fetch.hour.to_string())
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "planned_family={} fetched_families={} resolved_sources={} contributing_fetch_hours=[{}]",
        planned_family, fetched_families, resolved_sources, hours
    )
}

fn unique_join<'a>(values: impl IntoIterator<Item = &'a str>) -> String {
    let mut unique = Vec::<&'a str>::new();
    for value in values {
        if !unique.contains(&value) {
            unique.push(value);
        }
    }
    unique.join(",")
}

#[cfg(test)]
pub(super) fn count_blocked_artifacts(manifest: &RunPublicationManifest) -> usize {
    manifest
        .artifacts
        .iter()
        .filter(|artifact| artifact.state == ArtifactPublicationState::Blocked)
        .count()
}

pub(super) fn collect_input_fetches(
    direct: &Option<HrrrDirectBatchReport>,
    derived: &Option<HrrrDerivedBatchReport>,
    windowed: &Option<HrrrWindowedBatchReport>,
) -> Vec<PublishedFetchIdentity> {
    let mut by_key = HashMap::<String, PublishedFetchIdentity>::new();

    if let Some(report) = direct {
        for fetch in &report.fetches {
            by_key
                .entry(fetch.input_fetch.fetch_key.clone())
                .or_insert_with(|| fetch.input_fetch.clone());
        }
    }

    if let Some(report) = derived {
        for fetch in &report.input_fetches {
            by_key
                .entry(fetch.fetch_key.clone())
                .or_insert_with(|| fetch.clone());
        }
    }

    if let Some(report) = windowed {
        for identity in collect_windowed_input_fetches(report) {
            by_key.entry(identity.fetch_key.clone()).or_insert(identity);
        }
    }

    let mut fetches = by_key.into_values().collect::<Vec<_>>();
    fetches.sort_by(|left, right| left.fetch_key.cmp(&right.fetch_key));
    fetches
}

fn windowed_runtime_fetches_for_product<'a>(
    product: &HrrrWindowedRenderedProduct,
    shared_timing: &'a crate::windowed::HrrrWindowedSharedTiming,
) -> Vec<&'a crate::windowed::HrrrWindowedHourFetchInfo> {
    let is_qpf = matches!(
        product.product,
        HrrrWindowedProduct::Qpf1h
            | HrrrWindowedProduct::Qpf6h
            | HrrrWindowedProduct::Qpf12h
            | HrrrWindowedProduct::Qpf24h
            | HrrrWindowedProduct::QpfTotal
    );
    let is_wind = matches!(
        product.product,
        HrrrWindowedProduct::Wind10m1hMax
            | HrrrWindowedProduct::Wind10mRunMax
            | HrrrWindowedProduct::Wind10m0to24hMax
            | HrrrWindowedProduct::Wind10m24to48hMax
            | HrrrWindowedProduct::Wind10m0to48hMax
    );
    let is_surface_snapshot = product.product.is_surface_snapshot();
    let contributing_hours = &product.metadata.contributing_forecast_hours;
    let fetches = if is_qpf {
        &shared_timing.surface_hour_fetches
    } else if is_wind {
        &shared_timing.wind_hour_fetches
    } else if is_surface_snapshot {
        &shared_timing.temp_hour_fetches
    } else {
        &shared_timing.uh_hour_fetches
    };
    fetches
        .iter()
        .filter(|fetch| contributing_hours.contains(&fetch.hour))
        .collect()
}
