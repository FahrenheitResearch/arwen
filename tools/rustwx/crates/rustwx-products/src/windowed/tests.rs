use super::*;
use rustwx_render::ChromeScale;

#[test]
fn plan_windowed_products_blocks_short_forecast_hours() {
    let (planned, blockers, surface_hours, nat_hours, wind_hours, temp_hours) =
        plan_windowed_products(
            &[HrrrWindowedProduct::Qpf24h, HrrrWindowedProduct::Uh25km3h],
            2,
            Some(0),
        );
    assert!(planned.is_empty());
    assert_eq!(blockers.len(), 2);
    assert!(surface_hours.is_empty());
    assert!(nat_hours.is_empty());
    assert!(wind_hours.is_empty());
    assert!(temp_hours.is_empty());
}

#[test]
fn qpf_hourly_fallback_is_limited_to_hourly_cadence_models() {
    assert!(qpf_hourly_fallback_supported(ModelId::Hrrr, 48));
    assert!(qpf_hourly_fallback_supported(ModelId::Rap, 51));
    assert!(qpf_hourly_fallback_supported(ModelId::Gfs, 120));
    assert!(!qpf_hourly_fallback_supported(ModelId::Gfs, 123));
    assert!(!qpf_hourly_fallback_supported(ModelId::Gefs, 6));
    assert!(!qpf_hourly_fallback_supported(ModelId::EcmwfOpenData, 6));
}

#[test]
fn plan_windowed_products_adds_wind_max_hours_for_any_extended_cycle() {
    let (planned, blockers, surface_hours, nat_hours, wind_hours, temp_hours) =
        plan_windowed_products(
            &[
                HrrrWindowedProduct::Wind10m1hMax,
                HrrrWindowedProduct::Wind10m0to24hMax,
                HrrrWindowedProduct::Wind10m24to48hMax,
                HrrrWindowedProduct::Wind10m0to48hMax,
            ],
            48,
            Some(0),
        );
    assert_eq!(planned.len(), 4);
    assert!(blockers.is_empty());
    assert!(surface_hours.is_empty());
    assert!(nat_hours.is_empty());
    assert!(temp_hours.is_empty());
    assert_eq!(wind_hours.first(), Some(&1));
    assert_eq!(wind_hours.last(), Some(&48));

    let (planned, blockers, _, _, wind_hours, temp_hours) =
        plan_windowed_products(&[HrrrWindowedProduct::Wind10m0to24hMax], 24, Some(18));
    assert_eq!(planned, vec![HrrrWindowedProduct::Wind10m0to24hMax]);
    assert!(blockers.is_empty());
    assert_eq!(wind_hours.first(), Some(&1));
    assert_eq!(wind_hours.last(), Some(&24));
    assert!(temp_hours.is_empty());
}

#[test]
fn plan_windowed_products_adds_diurnal_temperature_hours() {
    let (planned, blockers, surface_hours, nat_hours, wind_hours, temp_hours) =
        plan_windowed_products(
            &[
                HrrrWindowedProduct::Temp2m0to24hMax,
                HrrrWindowedProduct::Temp2m24to48hMin,
                HrrrWindowedProduct::Temp2m0to48hMax,
                HrrrWindowedProduct::Temp2m0to48hRange,
                HrrrWindowedProduct::Rh2m0to24hMin,
                HrrrWindowedProduct::Dewpoint2m24to48hMax,
                HrrrWindowedProduct::Vpd2m0to48hRange,
            ],
            48,
            Some(0),
        );
    assert_eq!(planned.len(), 7);
    assert!(blockers.is_empty());
    assert!(surface_hours.is_empty());
    assert!(nat_hours.is_empty());
    assert!(wind_hours.is_empty());
    assert_eq!(temp_hours.first(), Some(&1));
    assert_eq!(temp_hours.last(), Some(&48));

    let (planned, blockers, _, _, _, temp_hours) =
        plan_windowed_products(&[HrrrWindowedProduct::Temp2m0to24hMax], 24, Some(18));
    assert_eq!(planned, vec![HrrrWindowedProduct::Temp2m0to24hMax]);
    assert!(blockers.is_empty());
    assert_eq!(temp_hours.first(), Some(&1));
    assert_eq!(temp_hours.last(), Some(&24));
}

#[test]
fn windowed_fetch_truth_can_show_nat_planned_but_sfc_fetched() {
    let fetch = HrrrWindowedHourFetchInfo {
        hour: 1,
        planned_product: "nat".into(),
        fetched_product: "sfc".into(),
        requested_source: SourceId::Nomads,
        resolved_source: SourceId::Nomads,
        resolved_url: "https://example.test/hrrr.t23z.wrfsfcf01.grib2".into(),
        fetch_cache_hit: false,
        input_fetch: None,
    };
    assert_eq!(fetch.planned_product, "nat");
    assert_eq!(fetch.fetched_product, "sfc");
    assert_eq!(fetch.resolved_source, SourceId::Nomads);
    assert!(fetch.resolved_url.contains("wrfsfc"));
}

#[test]
fn windowed_render_request_uses_modern_map_chrome() {
    let shape = rustwx_core::GridShape::new(2, 2).unwrap();
    let grid = rustwx_core::LatLonGrid::new(
        shape,
        vec![36.0, 36.0, 35.0, 35.0],
        vec![-98.0, -97.0, -98.0, -97.0],
    )
    .unwrap();
    let field = rustwx_core::Field2D::new(
        rustwx_core::ProductKey::named("qpf_1h"),
        "in",
        grid,
        vec![0.0, 0.1, 0.2, 0.3],
    )
    .unwrap();
    let computed = crate::windowed_decoder::ComputedWindowedField {
        field,
        title: "1-h QPF".to_string(),
        metadata: HrrrWindowedProductMetadata {
            strategy: "test window".to_string(),
            contributing_forecast_hours: vec![1],
            window_hours: Some(1),
        },
        scale: rustwx_render::ColorScale::Discrete(crate::windowed_decoder::qpf_scale()),
    };
    let request = HrrrWindowedBatchRequest {
        model: ModelId::Hrrr,
        date_yyyymmdd: "20260424".to_string(),
        cycle_override_utc: Some(22),
        forecast_hour: 1,
        source: SourceId::Nomads,
        domain: DomainSpec::new("southern_plains", (-109.0, -90.0, 25.0, 40.5)),
        out_dir: PathBuf::new(),
        cache_root: PathBuf::new(),
        use_cache: false,
        products: vec![HrrrWindowedProduct::Qpf1h],
        output_width: 1200,
        output_height: 900,
        png_compression: PngCompressionMode::Default,
        place_label_overlay: None,
    };
    let projected = ProjectedMap {
        projected_x: vec![0.0, 1.0, 0.0, 1.0],
        projected_y: vec![1.0, 1.0, 0.0, 0.0],
        extent: rustwx_render::ProjectedExtent {
            x_min: 0.0,
            x_max: 1.0,
            y_min: 0.0,
            y_max: 1.0,
        },
        lines: Vec::new(),
        polygons: Vec::new(),
        inverse_raster_projection: None,
    };

    let render_request = build_windowed_render_request(
        HrrrWindowedProduct::Qpf1h,
        &computed,
        &request,
        &projected,
        "20260424",
        22,
        1,
        ModelId::Hrrr,
        SourceId::Nomads,
    );

    assert_eq!(render_request.width, 1200);
    assert_eq!(render_request.height, 900);
    assert_eq!(render_request.chrome_scale, ChromeScale::Fixed(0.9));
    assert_eq!(render_request.supersample_factor, 1);
    assert_eq!(
        render_request.subtitle_left.as_deref(),
        Some("Init 04/24 22Z | F001 | Valid 04/24 23Z | HRRR")
    );
    assert_eq!(
        render_request.subtitle_right.as_deref(),
        Some("source: nomads")
    );
    assert_eq!(
        render_request.visual_mode,
        ProductVisualMode::FilledMeteorology
    );
    assert_eq!(
        render_request.legend.mode,
        rustwx_render::LegendMode::SmoothRamp
    );
    assert!(render_request.domain_frame.is_some());
    assert!(render_request.projected_domain.is_some());
}

#[test]
fn windowed_render_request_labels_fixed_window_instead_of_requested_end_hour() {
    let shape = rustwx_core::GridShape::new(2, 2).unwrap();
    let grid = rustwx_core::LatLonGrid::new(
        shape,
        vec![36.0, 36.0, 35.0, 35.0],
        vec![-98.0, -97.0, -98.0, -97.0],
    )
    .unwrap();
    let field = rustwx_core::Field2D::new(
        rustwx_core::ProductKey::named("2m_rh_24_48h_range"),
        "%",
        grid,
        vec![10.0, 20.0, 30.0, 40.0],
    )
    .unwrap();
    let computed = crate::windowed_decoder::ComputedWindowedField {
        field,
        title: "2 m Relative Humidity Range (24-48 h)".to_string(),
        metadata: HrrrWindowedProductMetadata {
            strategy:
                "pointwise max-min range of hourly 2 m relative humidity snapshots across F025-F048"
                    .to_string(),
            contributing_forecast_hours: (25..=48).collect(),
            window_hours: Some(24),
        },
        scale: rustwx_render::ColorScale::Discrete(crate::windowed_decoder::rh2m_scale(true)),
    };
    let request = HrrrWindowedBatchRequest {
        model: ModelId::Hrrr,
        date_yyyymmdd: "20260424".to_string(),
        cycle_override_utc: Some(0),
        forecast_hour: 48,
        source: SourceId::Aws,
        domain: DomainSpec::new("california", (-124.9, -113.8, 31.9, 42.5)),
        out_dir: PathBuf::new(),
        cache_root: PathBuf::new(),
        use_cache: false,
        products: vec![HrrrWindowedProduct::Rh2m24to48hRange],
        output_width: 1200,
        output_height: 900,
        png_compression: PngCompressionMode::Default,
        place_label_overlay: None,
    };
    let projected = ProjectedMap {
        projected_x: vec![0.0, 1.0, 0.0, 1.0],
        projected_y: vec![1.0, 1.0, 0.0, 0.0],
        extent: rustwx_render::ProjectedExtent {
            x_min: 0.0,
            x_max: 1.0,
            y_min: 0.0,
            y_max: 1.0,
        },
        lines: Vec::new(),
        polygons: Vec::new(),
        inverse_raster_projection: None,
    };

    let render_request = build_windowed_render_request(
        HrrrWindowedProduct::Rh2m24to48hRange,
        &computed,
        &request,
        &projected,
        "20260424",
        0,
        48,
        ModelId::Hrrr,
        SourceId::Aws,
    );

    assert_eq!(
        render_request.subtitle_left.as_deref(),
        Some("Init 04/24 00Z | F025-F048 | Valid 04/26 00Z | HRRR")
    );
    assert_eq!(
        render_request.subtitle_right.as_deref(),
        Some("source: aws")
    );
}

#[test]
fn from_slug_round_trips_every_supported_windowed_product() {
    for &product in HrrrWindowedProduct::supported_products() {
        assert_eq!(
            HrrrWindowedProduct::from_slug(product.slug()),
            Some(product),
            "slug '{}' must parse back to its product",
            product.slug()
        );
    }
    assert_eq!(HrrrWindowedProduct::from_slug("not_a_windowed_slug"), None);
}

/// The store-render scale helper reproduces the per-family scales the GRIB
/// compute kernels attach: QPF/wind/snapshot products get the family's
/// discrete scale, UH products the Uh weather preset — for every supported
/// product, so store-rendered windowed PNGs style identically.
#[test]
fn windowed_product_scale_matches_the_kernel_families() {
    use rustwx_render::ColorScale;
    for &product in HrrrWindowedProduct::supported_products() {
        let scale = crate::windowed_decoder::windowed_product_scale(product);
        if product.is_qpf() {
            let ColorScale::Discrete(scale) = scale else {
                panic!("{}: QPF must use a discrete scale", product.slug());
            };
            assert_eq!(scale.levels, crate::windowed_decoder::qpf_scale().levels);
        } else if product.is_wind10m() {
            let ColorScale::Discrete(scale) = scale else {
                panic!("{}: wind must use a discrete scale", product.slug());
            };
            assert_eq!(
                scale.levels,
                crate::windowed_decoder::wind10m_scale().levels
            );
        } else if product.is_surface_snapshot() {
            assert!(
                matches!(scale, ColorScale::Discrete(_)),
                "{}: snapshot windows use discrete scales",
                product.slug()
            );
        } else {
            assert!(
                matches!(scale, ColorScale::Weather(_)),
                "{}: UH products use the Uh weather preset",
                product.slug()
            );
        }
    }
}
