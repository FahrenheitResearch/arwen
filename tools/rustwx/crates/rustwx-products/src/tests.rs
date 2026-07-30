use super::*;
use crate::places::{PlaceLabelDensityTier, PlaceLabelOverlay};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProductModuleSurfaceKind {
    StablePublic,
    OperationalPublic,
    CompatibilityPublic,
    ProofResearchPublic,
    InternalCandidatePublic,
    LegacyPublic,
    CratePrivate,
}

use ProductModuleSurfaceKind::*;

const PRODUCT_MODULE_SURFACE: &[(&str, ProductModuleSurfaceKind)] = &[
    ("cache", CompatibilityPublic),
    ("catalog", StablePublic),
    ("derived", StablePublic),
    ("direct", StablePublic),
    ("ecape", OperationalPublic),
    ("gridded", CompatibilityPublic),
    ("heavy", OperationalPublic),
    ("hrrr", LegacyPublic),
    ("named_geometry", StablePublic),
    ("non_ecape", OperationalPublic),
    ("places", CompatibilityPublic),
    ("planner", CompatibilityPublic),
    ("plot_design", InternalCandidatePublic),
    ("point_timeseries", OperationalPublic),
    ("publication", OperationalPublic),
    ("qpf", CratePrivate),
    ("runtime", CompatibilityPublic),
    ("sampling", StablePublic),
    ("severe", OperationalPublic),
    ("shared_context", StablePublic),
    ("source", CompatibilityPublic),
    ("spec", CompatibilityPublic),
    ("thermo_native", ProofResearchPublic),
    ("viewer", StablePublic),
    ("windowed", OperationalPublic),
    ("windowed_decoder", InternalCandidatePublic),
];

const CALIFORNIA_SQUARE: (f64, f64, f64, f64) = (-124.9, -113.7, 31.8, 42.7);

fn declared_product_modules() -> Vec<(&'static str, bool)> {
    include_str!("lib.rs")
        .lines()
        .filter_map(|line| {
            let trimmed = line.trim();
            if let Some(rest) = trimmed.strip_prefix("pub mod ") {
                Some((rest.split(';').next()?.trim(), true))
            } else if let Some(rest) = trimmed.strip_prefix("pub(crate) mod ") {
                Some((rest.split(';').next()?.trim(), false))
            } else {
                None
            }
        })
        .collect()
}

fn assert_no_duplicate_names(sorted_names: &[&str], label: &str) {
    for pair in sorted_names.windows(2) {
        assert_ne!(pair[0], pair[1], "duplicate {label}: {}", pair[0]);
    }
}

fn sample_place_label_request() -> rustwx_render::MapRenderRequest {
    let grid = rustwx_render::LatLonGrid::new(
        rustwx_render::GridShape::new(2, 2).unwrap(),
        vec![31.8, 31.8, 42.7, 42.7],
        vec![-124.9, -113.7, -124.9, -113.7],
    )
    .unwrap();
    let field = rustwx_render::Field2D::new(
        rustwx_render::ProductKey::named("place_label_density_style_test"),
        "unitless",
        grid,
        vec![0.0, 0.0, 0.0, 0.0],
    )
    .unwrap();
    let mut request = rustwx_render::MapRenderRequest::contour_only(field);
    request.projected_domain = Some(rustwx_render::ProjectedDomain {
        x: vec![0.0, 1.0, 0.0, 1.0],
        y: vec![0.0, 0.0, 1.0, 1.0],
        extent: rustwx_render::ProjectedExtent {
            x_min: 0.0,
            x_max: 1.0,
            y_min: 0.0,
            y_max: 1.0,
        },
    });
    request
}

#[test]
fn public_surface_classification_covers_declared_modules() {
    let mut declared = declared_product_modules()
        .into_iter()
        .map(|(name, _)| name)
        .collect::<Vec<_>>();
    let mut classified = PRODUCT_MODULE_SURFACE
        .iter()
        .map(|(name, _)| *name)
        .collect::<Vec<_>>();

    declared.sort_unstable();
    classified.sort_unstable();

    assert_no_duplicate_names(&declared, "declared module");
    assert_no_duplicate_names(&classified, "classified module");
    assert_eq!(classified, declared);
}

#[test]
fn public_surface_classification_marks_crate_private_modules() {
    for (module_name, is_public) in declared_product_modules() {
        let kind = PRODUCT_MODULE_SURFACE
            .iter()
            .find(|(name, _)| *name == module_name)
            .map(|(_, kind)| *kind)
            .unwrap_or_else(|| panic!("{module_name} missing from product surface map"));
        assert_eq!(
            matches!(kind, CratePrivate),
            !is_public,
            "{module_name} classification should match crate-root visibility"
        );
    }
}

#[test]
fn major_and_aux_place_label_glue_marks_auxiliary_catalog_entries_as_auxiliary() {
    let overlay = PlaceLabelOverlay::major_us_cities()
        .with_density(PlaceLabelDensityTier::MajorAndAux)
        .with_included_place_slugs([
            "ca_los_angeles",
            "ca_san_diego",
            "ca_bakersfield",
            "ca_santa_barbara",
        ]);
    let domain = DomainSpec::new("california_square", CALIFORNIA_SQUARE);
    let selected = overlay.selected_places_for_domain(&domain);

    assert!(
        selected
            .iter()
            .any(|place| !is_major_catalog_place(place.slug.as_str()))
    );

    let mut request = sample_place_label_request();
    let grid_lat_deg = request.field.grid.lat_deg.clone();
    let grid_lon_deg = request.field.grid.lon_deg.clone();
    apply_place_label_overlay_with_density_styling(
        &mut request,
        &overlay,
        &domain,
        &grid_lat_deg,
        &grid_lon_deg,
        None,
    )
    .expect("overlay should project");

    assert_eq!(request.projected_place_labels.len(), selected.len());
    for (place, label) in selected.iter().zip(request.projected_place_labels.iter()) {
        let expected = if is_major_catalog_place(place.slug.as_str()) {
            rustwx_render::ProjectedPlaceLabelPriority::Primary
        } else {
            rustwx_render::ProjectedPlaceLabelPriority::Auxiliary
        };
        assert_eq!(label.priority, expected);
    }
}

#[test]
fn dense_place_label_glue_pushes_lower_rank_auxiliary_entries_to_micro() {
    let overlay = PlaceLabelOverlay::major_us_cities()
        .with_density(PlaceLabelDensityTier::Dense)
        .with_included_place_slugs([
            "ca_los_angeles",
            "ca_san_diego",
            "ca_bakersfield",
            "ca_fresno",
            "ca_san_jose",
            "ca_santa_barbara",
            "ca_san_luis_obispo",
        ]);
    let domain = DomainSpec::new("california_square", CALIFORNIA_SQUARE);
    let selected = overlay.selected_places_for_domain(&domain);
    let aux_total = selected
        .iter()
        .filter(|place| !is_major_catalog_place(place.slug.as_str()))
        .count();

    assert!(
        aux_total >= 3,
        "test should include enough auxiliary labels"
    );

    let mut request = sample_place_label_request();
    let grid_lat_deg = request.field.grid.lat_deg.clone();
    let grid_lon_deg = request.field.grid.lon_deg.clone();
    apply_place_label_overlay_with_density_styling(
        &mut request,
        &overlay,
        &domain,
        &grid_lat_deg,
        &grid_lon_deg,
        None,
    )
    .expect("overlay should project");

    let micro_count = request
        .projected_place_labels
        .iter()
        .filter(|label| label.priority == rustwx_render::ProjectedPlaceLabelPriority::Micro)
        .count();
    assert!(micro_count > 0);
    assert!(
        request.projected_place_labels.iter().any(|label| {
            label.priority == rustwx_render::ProjectedPlaceLabelPriority::Auxiliary
        })
    );
    assert!(
        selected
            .iter()
            .zip(request.projected_place_labels.iter())
            .filter(|(place, _)| is_major_catalog_place(place.slug.as_str()))
            .all(|(_, label)| {
                label.priority == rustwx_render::ProjectedPlaceLabelPriority::Primary
            })
    );
}
