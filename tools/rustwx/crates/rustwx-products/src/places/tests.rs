use super::*;
use std::collections::HashSet;

const CALIFORNIA_SQUARE: (f64, f64, f64, f64) = (-124.9, -113.7, 31.8, 42.7);

fn sample_places() -> [PlacePreset; 4] {
    [
        PlacePreset {
            slug: "center",
            label: "Center",
            center_lon: -100.0,
            center_lat: 40.0,
            half_height_deg: 1.2,
        },
        PlacePreset {
            slug: "near_center",
            label: "Near Center",
            center_lon: -99.5,
            center_lat: 40.2,
            half_height_deg: 1.2,
        },
        PlacePreset {
            slug: "edge",
            label: "Edge",
            center_lon: -108.0,
            center_lat: 49.0,
            half_height_deg: 1.2,
        },
        PlacePreset {
            slug: "outside",
            label: "Outside",
            center_lon: -130.0,
            center_lat: 20.0,
            half_height_deg: 1.2,
        },
    ]
}

fn sample_place_label_request() -> MapRenderRequest {
    let grid = rustwx_render::LatLonGrid::new(
        rustwx_render::GridShape::new(2, 2).unwrap(),
        vec![31.8, 31.8, 42.7, 42.7],
        vec![-124.9, -113.7, -124.9, -113.7],
    )
    .unwrap();
    let field = rustwx_render::Field2D::new(
        rustwx_render::ProductKey::named("place_label_overlay_test"),
        "unitless",
        grid,
        vec![0.0, 0.0, 0.0, 0.0],
    )
    .unwrap();
    let mut request = MapRenderRequest::contour_only(field);
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

fn place_slugs(places: &[SelectedPlace]) -> Vec<&str> {
    places.iter().map(|place| place.slug.as_str()).collect()
}

#[test]
fn major_city_slugs_are_unique() {
    let mut seen = HashSet::new();
    for city in MAJOR_US_CITY_PRESETS {
        assert!(seen.insert(city.slug), "duplicate city slug {}", city.slug);
    }
}

#[test]
fn auxiliary_and_micro_place_slugs_are_unique_and_do_not_overlap_other_catalogs() {
    let mut seen = HashSet::new();
    for city in MAJOR_US_CITY_PRESETS {
        assert!(seen.insert(city.slug), "duplicate city slug {}", city.slug);
    }
    for city in AUX_US_CITY_PRESETS {
        assert!(seen.insert(city.slug), "duplicate city slug {}", city.slug);
    }
    for city in MICRO_US_PLACE_PRESETS {
        assert!(seen.insert(city.slug), "duplicate city slug {}", city.slug);
    }
}

#[test]
fn new_york_city_bounds_stay_centered() {
    let domain = MAJOR_US_CITY_PRESETS
        .iter()
        .find(|city| city.slug == "ny_new_york_city")
        .expect("NYC preset should exist")
        .domain();
    let (west, east, south, north) = domain.bounds;
    assert!((((west + east) / 2.0) + 74.0).abs() < 0.1);
    assert!((((south + north) / 2.0) - 40.71).abs() < 0.1);
}

#[test]
fn centered_domain_preserves_requested_physical_aspect_ratio() {
    let domain = centered_domain("custom_place", -104.99, 39.74, 1.9);
    let (west, east, south, north) = domain.bounds;
    let center_lat = (south + north) / 2.0;
    let width_deg = (east - west) * center_lat.to_radians().cos().abs();
    let height_deg = north - south;
    let aspect_ratio = width_deg / height_deg;

    assert!((aspect_ratio - PLACE_OUTPUT_ASPECT_RATIO).abs() < 1.0e-6);
}

#[test]
fn major_us_city_domains_follow_source_order() {
    let domains = major_us_city_domains();

    assert_eq!(domains.len(), MAJOR_US_CITY_PRESETS.len());
    assert_eq!(domains[0].slug, MAJOR_US_CITY_PRESETS[0].slug);
    assert_eq!(
        domains.last().map(|domain| domain.slug.as_str()),
        Some(MAJOR_US_CITY_PRESETS.last().unwrap().slug)
    );
}

#[test]
fn crop_selection_keeps_centers_within_bounds_and_includes_major_california_metros() {
    let selected =
        select_major_us_city_places(CALIFORNIA_SQUARE, PlaceSelectionOptions::for_city_crops());
    let slugs = selected
        .iter()
        .map(|place| place.slug.as_str())
        .collect::<Vec<_>>();

    assert!(
        selected
            .iter()
            .all(|place| { contains_point(CALIFORNIA_SQUARE, place.center_lon, place.center_lat) })
    );
    assert!(slugs.contains(&"ca_los_angeles"));
    assert!(slugs.contains(&"ca_san_francisco_bay"));
    assert!(slugs.contains(&"ca_sacramento"));
    assert!(slugs.contains(&"ca_san_diego"));
}

#[test]
fn ranked_selection_prefers_central_places_before_edges() {
    let bounds = (-110.0, -90.0, 30.0, 50.0);
    let selected = select_places_for_bounds(
        &sample_places(),
        bounds,
        PlaceSelectionOptions::for_city_crops().with_max_count(1),
    );

    assert_eq!(selected.len(), 1);
    assert_eq!(selected[0].slug, "center");
}

#[test]
fn overlay_selection_declutters_close_neighbors() {
    let bounds = (-110.0, -90.0, 30.0, 50.0);
    let selected = select_places_for_bounds(
        &sample_places(),
        bounds,
        PlaceSelectionOptions::for_overlay_labels().with_max_count(3),
    );
    let slugs = selected
        .iter()
        .map(|place| place.slug.as_str())
        .collect::<Vec<_>>();

    assert!(slugs.contains(&"center"));
    assert!(!slugs.contains(&"near_center"));
    assert!(!slugs.contains(&"outside"));
}

#[test]
fn overlay_selection_can_relax_declutter_knobs() {
    let bounds = (-110.0, -90.0, 30.0, 50.0);
    let selected = select_places_for_bounds(
        &sample_places(),
        bounds,
        PlaceSelectionOptions::for_overlay_labels()
            .with_min_center_spacing_km(0.0)
            .with_max_crop_overlap_fraction(1.0)
            .with_max_count(3),
    );
    let slugs = selected
        .iter()
        .map(|place| place.slug.as_str())
        .collect::<Vec<_>>();

    assert!(slugs.contains(&"center"));
    assert!(slugs.contains(&"near_center"));
    assert!(!slugs.contains(&"outside"));
}

#[test]
fn non_major_selection_uses_smaller_declutter_bounds_but_keeps_public_crop_bounds() {
    let aux = PlacePreset {
        slug: "ca_san_jose",
        label: "Aux Probe",
        center_lon: -100.0,
        center_lat: 40.0,
        half_height_deg: DEFAULT_PLACE_HALF_HEIGHT_DEG,
    };
    let micro = PlacePreset {
        slug: "ca_oxnard",
        label: "Micro Probe",
        center_lon: -97.4,
        center_lat: 40.0,
        half_height_deg: DEFAULT_PLACE_HALF_HEIGHT_DEG,
    };
    let bounds = (-103.5, -94.0, 37.5, 42.5);

    assert!(
        crop_overlap_fraction(aux.bounds(), micro.bounds()) > 0.35,
        "full crop footprints should still overlap enough to be decluttered"
    );
    assert!(
        crop_overlap_fraction(effective_place_bounds(aux), effective_place_bounds(micro)) < 0.35,
        "selection-only footprints should be small enough to admit both labels"
    );

    let selected = select_places_for_bounds(
        &[aux, micro],
        bounds,
        PlaceSelectionOptions::for_overlay_labels()
            .with_min_center_spacing_km(0.0)
            .with_max_count(2),
    );

    assert_eq!(selected.len(), 2);
    assert_eq!(selected[0].bounds, aux.bounds());
    assert_eq!(selected[1].bounds, micro.bounds());
}

#[test]
fn crop_within_bounds_can_filter_edge_places() {
    let catalog = [
        PlacePreset {
            slug: "inside",
            label: "Inside",
            center_lon: -100.0,
            center_lat: 40.0,
            half_height_deg: 0.8,
        },
        PlacePreset {
            slug: "edge_crop",
            label: "Edge Crop",
            center_lon: -100.8,
            center_lat: 40.0,
            half_height_deg: 1.4,
        },
    ];
    let selected = select_places_for_bounds(
        &catalog,
        (-102.0, -98.0, 38.5, 41.5),
        PlaceSelectionOptions::for_city_crops()
            .with_containment(PlaceContainmentMode::CropWithinBounds),
    );
    let slugs = selected
        .iter()
        .map(|place| place.slug.as_str())
        .collect::<Vec<_>>();

    assert_eq!(slugs, vec!["inside"]);
}

#[test]
fn city_label_plan_keeps_anchor_city_first() {
    let domain = MAJOR_US_CITY_PRESETS
        .iter()
        .find(|preset| preset.slug == "ca_los_angeles")
        .expect("los angeles preset should exist")
        .domain();
    let selected = select_places_for_label_plan(
        &domain,
        place_label_plan_for_domain(&domain).expect("city crop should produce a plan"),
        &PlaceLabelOverlay::major_us_cities(),
    );

    assert!(!selected.is_empty());
    assert_eq!(selected[0].slug, "ca_los_angeles");
    assert!(selected.len() <= 4);
}

#[test]
fn apply_place_label_overlay_compacts_region_labels_and_places_them_interior() {
    let mut request = sample_place_label_request();
    let grid_lat_deg = request.field.grid.lat_deg.clone();
    let grid_lon_deg = request.field.grid.lon_deg.clone();
    let overlay = PlaceLabelOverlay::major_us_cities()
        .with_included_place_slugs(["ca_los_angeles", "ca_san_francisco_bay"]);
    let domain = DomainSpec::new("california_square", CALIFORNIA_SQUARE);

    apply_place_label_overlay(
        &mut request,
        &overlay,
        &domain,
        &grid_lat_deg,
        &grid_lon_deg,
        None,
    )
    .expect("overlay should project");

    assert_eq!(request.projected_place_labels.len(), 2);
    assert!(
        request
            .projected_place_labels
            .iter()
            .all(|label| label.x.is_finite() && label.y.is_finite())
    );
    assert!(
        request
            .projected_place_labels
            .iter()
            .all(|label| { !label.label.as_deref().unwrap_or_default().contains(',') })
    );

    let los_angeles = request
        .projected_place_labels
        .iter()
        .find(|label| label.label.as_deref() == Some("Los Angeles"))
        .expect("Los Angeles should be included");
    assert_eq!(
        los_angeles.style.label_placement,
        ProjectedLabelPlacement::AboveLeft
    );
    assert_eq!(los_angeles.style.marker_radius_px, 3);
    assert!(!los_angeles.style.label_bold);

    let bay = request
        .projected_place_labels
        .iter()
        .find(|label| label.label.as_deref() == Some("San Francisco Bay"))
        .expect("San Francisco Bay should be included");
    assert_eq!(
        bay.style.label_placement,
        ProjectedLabelPlacement::BelowRight
    );
    assert_eq!(bay.style.label_scale, 1);
}

#[test]
fn apply_place_label_overlay_emphasizes_anchor_city_for_city_crops() {
    let mut request = sample_place_label_request();
    let grid_lat_deg = request.field.grid.lat_deg.clone();
    let grid_lon_deg = request.field.grid.lon_deg.clone();
    let domain = MAJOR_US_CITY_PRESETS
        .iter()
        .find(|preset| preset.slug == "ca_los_angeles")
        .expect("los angeles preset should exist")
        .domain();

    apply_place_label_overlay(
        &mut request,
        &PlaceLabelOverlay::major_us_cities(),
        &domain,
        &grid_lat_deg,
        &grid_lon_deg,
        None,
    )
    .expect("overlay should project");

    let anchor = request
        .projected_place_labels
        .first()
        .expect("city crop should include an anchor label");
    assert_eq!(anchor.label.as_deref(), Some("Los Angeles"));
    assert_eq!(anchor.style.marker_radius_px, 4);
    assert_eq!(anchor.style.marker_outline_width, 2);
    assert_eq!(anchor.style.label_scale, 2);
    assert_eq!(
        anchor.style.label_placement,
        ProjectedLabelPlacement::AboveRight
    );
    assert!(anchor.style.label_bold);
}

#[test]
fn place_label_overlay_can_limit_the_major_city_catalog() {
    let overlay = PlaceLabelOverlay::major_us_cities()
        .with_included_place_slugs(["ca_los_angeles", "ca_san_diego"]);
    let domain = DomainSpec::new("california_square", CALIFORNIA_SQUARE);
    let selected = overlay.selected_places_for_domain(&domain);
    let slugs = selected
        .iter()
        .map(|place| place.slug.as_str())
        .collect::<Vec<_>>();

    assert!(!slugs.is_empty());
    assert!(
        slugs
            .iter()
            .all(|slug| { matches!(*slug, "ca_los_angeles" | "ca_san_diego") })
    );
}

#[test]
fn denser_overlay_tiers_expand_city_crop_neighbors() {
    let domain = MAJOR_US_CITY_PRESETS
        .iter()
        .find(|preset| preset.slug == "ca_los_angeles")
        .expect("los angeles preset should exist")
        .domain();

    let major = PlaceLabelOverlay::major_us_cities().selected_places_for_domain(&domain);
    let aux = PlaceLabelOverlay::major_us_cities()
        .with_density(PlaceLabelDensityTier::MajorAndAux)
        .selected_places_for_domain(&domain);
    let dense = PlaceLabelOverlay::major_us_cities()
        .with_density(PlaceLabelDensityTier::Dense)
        .selected_places_for_domain(&domain);

    assert!(major.len() <= 4);
    assert!(aux.len() >= major.len());
    assert!(dense.len() >= aux.len());
    assert!(
        dense.len() > aux.len()
            || dense
                .iter()
                .any(|place| place_catalog_tier_for_slug(&place.slug) == PlaceCatalogTier::Micro)
    );
}

#[test]
fn dense_overlay_can_pull_micro_places_into_california_square() {
    let domain = DomainSpec::new("california_square", CALIFORNIA_SQUARE);
    let dense = PlaceLabelOverlay::major_us_cities()
        .with_density(PlaceLabelDensityTier::Dense)
        .selected_places_for_domain(&domain);
    assert!(
        dense
            .iter()
            .any(|place| place_catalog_tier_for_slug(&place.slug) == PlaceCatalogTier::Micro)
    );
}

#[test]
fn dense_overlay_can_select_micro_only_places_beyond_major_and_aux() {
    let domain = MAJOR_US_CITY_PRESETS
        .iter()
        .find(|preset| preset.slug == "ca_los_angeles")
        .expect("los angeles preset should exist")
        .domain();

    let major_and_aux = PlaceLabelOverlay::major_us_cities()
        .with_density(PlaceLabelDensityTier::MajorAndAux)
        .with_included_place_slugs(["ca_oxnard"])
        .selected_places_for_domain(&domain);
    let dense = PlaceLabelOverlay::major_us_cities()
        .with_density(PlaceLabelDensityTier::Dense)
        .with_included_place_slugs(["ca_oxnard"])
        .selected_places_for_domain(&domain);

    assert!(major_and_aux.is_empty());
    assert_eq!(
        dense
            .iter()
            .map(|place| place.slug.as_str())
            .collect::<Vec<_>>(),
        vec!["ca_oxnard"]
    );
}

#[test]
fn dense_overlay_keeps_region_selection_distinct_from_major_and_aux() {
    for (slug, bounds) in [
        ("california_square", CALIFORNIA_SQUARE),
        ("southern_plains", (-109.5, -89.5, 24.5, 40.5)),
        ("oklahoma", (-103.75, -93.5, 32.75, 38.25)),
    ] {
        let domain = DomainSpec::new(slug, bounds);
        let major_and_aux = PlaceLabelOverlay::major_us_cities()
            .with_density(PlaceLabelDensityTier::MajorAndAux)
            .selected_places_for_domain(&domain);
        let dense = PlaceLabelOverlay::major_us_cities()
            .with_density(PlaceLabelDensityTier::Dense)
            .selected_places_for_domain(&domain);
        let major_and_aux_slugs = place_slugs(&major_and_aux);
        let dense_slugs = place_slugs(&dense);

        assert!(
            dense.len() > major_and_aux.len(),
            "{slug} should admit more dense labels than tier 2; tier2={major_and_aux_slugs:?} dense={dense_slugs:?}"
        );
        assert_ne!(
            major_and_aux_slugs, dense_slugs,
            "{slug} tier 2 and tier 3 selections should not collapse to the same slug set"
        );
        assert!(
            dense
                .iter()
                .any(|place| place_catalog_tier_for_slug(&place.slug) == PlaceCatalogTier::Micro),
            "{slug} dense selection should pull at least one micro place"
        );
    }
}

#[test]
fn dense_overlay_keeps_crowded_city_crop_selection_distinct_from_major_and_aux() {
    let domain = MAJOR_US_CITY_PRESETS
        .iter()
        .find(|preset| preset.slug == "ok_oklahoma_city")
        .expect("oklahoma city preset should exist")
        .domain();

    let major_and_aux = PlaceLabelOverlay::major_us_cities()
        .with_density(PlaceLabelDensityTier::MajorAndAux)
        .selected_places_for_domain(&domain);
    let dense = PlaceLabelOverlay::major_us_cities()
        .with_density(PlaceLabelDensityTier::Dense)
        .selected_places_for_domain(&domain);
    let major_and_aux_slugs = place_slugs(&major_and_aux);
    let dense_slugs = place_slugs(&dense);
    let dense_only = dense
        .iter()
        .filter(|place| !major_and_aux.iter().any(|other| other.slug == place.slug))
        .map(|place| place.slug.as_str())
        .collect::<Vec<_>>();

    assert!(
        dense.len() > major_and_aux.len(),
        "ok_oklahoma_city should gain labels at tier 3; tier2={major_and_aux_slugs:?} dense={dense_slugs:?}"
    );
    assert_ne!(
        major_and_aux_slugs, dense_slugs,
        "ok_oklahoma_city tier 2 and tier 3 selections should not collapse"
    );
    assert!(
        dense_only.len() >= 2,
        "ok_oklahoma_city tier 3 should add multiple labels beyond tier 2; dense_only={dense_only:?}"
    );
}

#[test]
fn zero_density_disables_default_overlay_selection() {
    let domain = DomainSpec::new("california_square", CALIFORNIA_SQUARE);
    let overlay = default_place_label_overlay_for_domain(&domain, PlaceLabelDensityTier::None)
        .expect("known domain should still produce an overlay config");
    assert!(overlay.selected_places_for_domain(&domain).is_empty());
}

#[test]
fn split_region_slugs_produce_region_place_label_plans() {
    for (slug, bounds) in [
        ("pacific_northwest", (-125.0, -110.0, 41.0, 49.5)),
        ("california_southwest", (-125.0, -108.0, 31.0, 41.5)),
        ("rockies_high_plains", (-112.0, -96.0, 37.0, 49.5)),
    ] {
        let domain = DomainSpec::new(slug, bounds);
        let plan = place_label_plan_for_domain(&domain)
            .expect("split-region domain should produce a place-label plan");
        assert_eq!(plan.kind, PlaceLabelDomainKind::Region);
    }
}

#[test]
fn region_and_city_labels_use_compact_names() {
    assert_eq!(
        display_label_for_domain(PlaceLabelDomainKind::Region, "Los Angeles, CA"),
        "Los Angeles"
    );
    assert_eq!(
        display_label_for_domain(PlaceLabelDomainKind::CityCrop, "San Diego, CA"),
        "San Diego"
    );
    assert_eq!(
        display_label_for_domain(PlaceLabelDomainKind::Conus, "Phoenix, AZ"),
        "Phoenix, AZ"
    );
}
