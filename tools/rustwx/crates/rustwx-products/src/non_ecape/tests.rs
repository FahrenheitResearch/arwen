use super::*;
use crate::shared_context::TitleProvenance;
use crate::derived::{HrrrDerivedRecipeTiming, HrrrDerivedRenderedRecipe, HrrrDerivedSharedTiming};
use crate::direct::{
    DirectBatchRequest, HrrrDirectRecipeTiming, HrrrDirectRenderedRecipe, plan_direct_fetch_groups,
};
use crate::hrrr::HrrrFetchRuntimeInfo;
use crate::windowed::{
    HrrrWindowedBlocker, HrrrWindowedHourFetchInfo, HrrrWindowedProductMetadata,
    HrrrWindowedProductTiming, HrrrWindowedRenderedProduct, HrrrWindowedSharedTiming,
};
use rustwx_render::PngCompressionMode;

fn domain() -> DomainSpec {
    DomainSpec::new("conus", (-127.0, -66.0, 23.0, 51.5))
}

fn empty_request() -> HrrrNonEcapeHourRequest {
    HrrrNonEcapeHourRequest {
        date_yyyymmdd: "20260415".into(),
        cycle_override_utc: Some(12),
        forecast_hour: 6,
        source: SourceId::Aws,
        domain: domain(),
        out_dir: PathBuf::from("C:\\temp\\proof"),
        cache_root: PathBuf::from("C:\\temp\\proof\\cache"),
        use_cache: true,
        source_mode: ProductSourceMode::Canonical,
        direct_recipe_slugs: Vec::new(),
        derived_recipe_slugs: Vec::new(),
        windowed_products: Vec::new(),
        output_width: 1200,
        output_height: 900,
        png_compression: PngCompressionMode::Default,
        place_label_overlay: None,
    }
}

fn latest_global(model: ModelId) -> LatestRun {
    LatestRun {
        model,
        cycle: rustwx_core::CycleSpec::new("20260415", 12).unwrap(),
        source: SourceId::Aws,
    }
}

#[test]
fn duplicate_multi_domain_slugs_are_rejected() {
    let err = validate_requested_domains(&[domain(), domain()]).unwrap_err();
    assert!(err.to_string().contains("duplicate multi-domain slug"));
}

fn windowed_fetch_identity(
    planned_family: &str,
    fetched_product: &str,
    hour: u16,
) -> PublishedFetchIdentity {
    let request = rustwx_core::ModelRunRequest::new(
        rustwx_core::ModelId::Hrrr,
        rustwx_core::CycleSpec::new("20260415", 12).unwrap(),
        hour,
        fetched_product,
    )
    .unwrap();
    PublishedFetchIdentity {
        fetch_key: crate::publication::fetch_key(planned_family, &request),
        planned_family: planned_family.to_string(),
        planned_family_aliases: Vec::new(),
        request,
        source_override: Some(SourceId::Aws),
        resolved_source: SourceId::Aws,
        resolved_url: format!(
            "https://example.test/hrrr.t12z.wrf{}f{:02}.grib2",
            fetched_product, hour
        ),
        resolved_family: fetched_product.to_string(),
        bytes_len: 3,
        bytes_sha256: "abc123".into(),
    }
}

#[test]
fn validation_rejects_empty_requests() {
    let err = validate_requested_work(
        ModelId::Hrrr,
        &normalize_requested_products(&empty_request()),
    )
    .expect_err("empty request should be rejected")
    .to_string();
    assert!(err.contains("at least one direct recipe"));
}

#[test]
fn validation_rejects_heavy_derived_recipes() {
    let mut request = empty_request();
    request.derived_recipe_slugs = vec!["sbecape".into()];
    let err = validate_requested_work(ModelId::Hrrr, &normalize_requested_products(&request))
        .expect_err("heavy derived recipes should be rejected by non_ecape_hour")
        .to_string();
    assert!(err.contains("heavy ECAPE product"));
}

#[test]
fn validation_allows_cross_model_total_qpf_windowed_product() {
    let request = NonEcapeRequestedProducts {
        direct_recipe_slugs: Vec::new(),
        derived_recipe_slugs: Vec::new(),
        windowed_products: vec![HrrrWindowedProduct::QpfTotal],
    };
    validate_requested_work(ModelId::Gfs, &request)
        .expect("GFS qpf_total should use the v0.5 cross-model windowed path");
}

#[test]
fn validation_rejects_cross_model_hrrr_specific_windowed_products() {
    let request = NonEcapeRequestedProducts {
        direct_recipe_slugs: Vec::new(),
        derived_recipe_slugs: Vec::new(),
        windowed_products: vec![HrrrWindowedProduct::Qpf6h],
    };
    let err = validate_requested_work(ModelId::Gfs, &request)
        .expect_err("GFS qpf_6h should remain blocked until explicitly validated")
        .to_string();
    assert!(err.contains("qpf_total only"));
}

#[test]
fn normalization_routes_legacy_one_hour_qpf_to_windowed_lane() {
    let mut request = empty_request();
    request.direct_recipe_slugs = vec!["1h_qpf".into(), "cloud_cover".into()];
    let normalized = normalize_requested_products(&request);
    assert_eq!(
        normalized.direct_recipe_slugs,
        vec!["cloud_cover".to_string()]
    );
    assert_eq!(
        normalized.windowed_products,
        vec![HrrrWindowedProduct::Qpf1h]
    );
}

#[test]
fn nomads_runs_lanes_sequentially() {
    assert!(!should_run_lanes_concurrently(
        ModelId::Hrrr,
        SourceId::Nomads
    ));
    assert!(should_run_lanes_concurrently(ModelId::Hrrr, SourceId::Aws));
    assert!(should_run_lanes_concurrently(ModelId::RrfsA, SourceId::Aws));
    assert!(should_run_lanes_concurrently(
        ModelId::WrfGdex,
        SourceId::Gdex
    ));
}

#[test]
fn shared_non_ecape_plan_collapses_gfs_direct_and_pair_to_one_fetch_key() {
    let latest = latest_global(ModelId::Gfs);
    let direct_request = DirectBatchRequest {
        model: ModelId::Gfs,
        date_yyyymmdd: latest.cycle.date_yyyymmdd.clone(),
        cycle_override_utc: Some(latest.cycle.hour_utc),
        forecast_hour: 12,
        source: latest.source,
        domain: domain(),
        out_dir: PathBuf::from("C:\\temp\\proof"),
        cache_root: PathBuf::from("C:\\temp\\proof\\cache"),
        use_cache: true,
        recipe_slugs: vec!["mslp_10m_winds".into()],
        product_overrides: HashMap::new(),
        contour_mode: crate::derived::NativeContourRenderMode::Automatic,
        native_fill_level_multiplier: 1,
        output_width: 1200,
        output_height: 900,
        png_compression: PngCompressionMode::Default,
        place_label_overlay: None,
        output_suffix: None,
        subtitle_left_override: None,
        subtitle_right_override: None,
        title_provenance: TitleProvenance::default(),
        geographic_overlays: None,
        panel_annotations: None,
    };
    let direct_groups = plan_direct_fetch_groups(&direct_request).unwrap();
    let derived_recipes = plan_derived_recipes(&["sbcape".to_string()]).unwrap();
    let derived_routes = plan_native_thermo_routes_with_surface_product(
        ModelId::Gfs,
        &derived_recipes,
        ProductSourceMode::Canonical,
        None,
    )
    .unwrap();

    let plan = build_shared_non_ecape_execution_plan(
        &latest,
        12,
        &direct_groups,
        Some(&derived_routes),
        true,
        None,
        None,
    );

    assert_eq!(plan.fetch_keys().len(), 1);
    assert_eq!(plan.fetch_keys()[0].native_product, "pgrb2.0p25");
}

#[test]
fn shared_non_ecape_plan_strips_nomads_hrrr_direct_idx_patterns() {
    let latest = LatestRun {
        model: ModelId::Hrrr,
        cycle: rustwx_core::CycleSpec::new("20260415", 12).unwrap(),
        source: SourceId::Nomads,
    };
    let direct_request = DirectBatchRequest {
        model: ModelId::Hrrr,
        date_yyyymmdd: latest.cycle.date_yyyymmdd.clone(),
        cycle_override_utc: Some(latest.cycle.hour_utc),
        forecast_hour: 6,
        source: latest.source,
        domain: domain(),
        out_dir: PathBuf::from("C:\\temp\\proof"),
        cache_root: PathBuf::from("C:\\temp\\proof\\cache"),
        use_cache: true,
        recipe_slugs: vec!["500mb_temperature_height_winds".into()],
        product_overrides: HashMap::new(),
        contour_mode: crate::derived::NativeContourRenderMode::Automatic,
        native_fill_level_multiplier: 1,
        output_width: 1200,
        output_height: 900,
        png_compression: PngCompressionMode::Default,
        place_label_overlay: None,
        output_suffix: None,
        subtitle_left_override: None,
        subtitle_right_override: None,
        title_provenance: TitleProvenance::default(),
        geographic_overlays: None,
        panel_annotations: None,
    };
    let direct_groups = plan_direct_fetch_groups(&direct_request).unwrap();
    let plan =
        build_shared_non_ecape_execution_plan(&latest, 6, &direct_groups, None, false, None, None);
    let prs_bundle = plan
        .bundles
        .iter()
        .find(|bundle| bundle.id.native_product == "prs")
        .expect("expected HRRR prs direct bundle");
    let patterns = prs_bundle
        .aliases
        .iter()
        .flat_map(|alias| alias.variable_patterns.iter())
        .collect::<Vec<_>>();
    assert!(
        patterns.is_empty(),
        "NOMADS production aliases should not carry .idx subset patterns"
    );
}

#[test]
fn shared_non_ecape_plan_collapses_ecmwf_direct_and_pair_to_one_fetch_key() {
    let latest = latest_global(ModelId::EcmwfOpenData);
    let direct_request = DirectBatchRequest {
        model: ModelId::EcmwfOpenData,
        date_yyyymmdd: latest.cycle.date_yyyymmdd.clone(),
        cycle_override_utc: Some(latest.cycle.hour_utc),
        forecast_hour: 6,
        source: latest.source,
        domain: domain(),
        out_dir: PathBuf::from("C:\\temp\\proof"),
        cache_root: PathBuf::from("C:\\temp\\proof\\cache"),
        use_cache: true,
        recipe_slugs: vec!["500mb_height_winds".into()],
        product_overrides: HashMap::new(),
        contour_mode: crate::derived::NativeContourRenderMode::Automatic,
        native_fill_level_multiplier: 1,
        output_width: 1200,
        output_height: 900,
        png_compression: PngCompressionMode::Default,
        place_label_overlay: None,
        output_suffix: None,
        subtitle_left_override: None,
        subtitle_right_override: None,
        title_provenance: TitleProvenance::default(),
        geographic_overlays: None,
        panel_annotations: None,
    };
    let direct_groups = plan_direct_fetch_groups(&direct_request).unwrap();
    let derived_recipes = plan_derived_recipes(&["sbcape".to_string()]).unwrap();
    let derived_routes = plan_native_thermo_routes_with_surface_product(
        ModelId::EcmwfOpenData,
        &derived_recipes,
        ProductSourceMode::Canonical,
        None,
    )
    .unwrap();

    let plan = build_shared_non_ecape_execution_plan(
        &latest,
        6,
        &direct_groups,
        Some(&derived_routes),
        true,
        None,
        None,
    );

    assert_eq!(plan.fetch_keys().len(), 1);
    assert_eq!(plan.fetch_keys()[0].native_product, "oper");
}

#[test]
fn summary_flattens_outputs_across_all_runners() {
    let direct = HrrrDirectBatchReport {
        model: rustwx_core::ModelId::Hrrr,
        date_yyyymmdd: "20260415".into(),
        cycle_utc: 12,
        forecast_hour: 6,
        source: SourceId::Aws,
        domain: domain(),
        fetches: Vec::new(),
        recipes: vec![HrrrDirectRenderedRecipe {
            recipe_slug: "composite_reflectivity".into(),
            title: "Composite Reflectivity".into(),
            source_route: crate::source::ProductSourceRoute::DirectNativeExact,
            grib_product: "nat".into(),
            fetched_grib_product: "sfc".into(),
            resolved_source: SourceId::Aws,
            resolved_url: "https://example.test/hrrr.t12z.wrfsfcf06.grib2".into(),
            output_path: PathBuf::from("C:\\proof\\direct.png"),
            content_identity: crate::publication::artifact_identity_from_bytes(b"direct"),
            input_fetch_keys: vec!["direct:nat->sfc".into()],
            timing: HrrrDirectRecipeTiming {
                project_ms: 1,
                field_prepare_ms: 0,
                contour_prepare_ms: 0,
                barb_prepare_ms: 0,
                render_to_image_ms: 0,
                data_layer_draw_ms: 0,
                overlay_draw_ms: 0,
                panel_compose_ms: 0,
                request_build_ms: 0,
                render_state_prep_ms: 0,
                png_encode_ms: 0,
                file_write_ms: 0,
                render_ms: 2,
                total_ms: 3,
                state_timing: Default::default(),
                image_timing: Default::default(),
            },
        }],
        blockers: Vec::new(),
        total_ms: 10,
    };
    let derived = HrrrDerivedBatchReport {
        date_yyyymmdd: "20260415".into(),
        cycle_utc: 12,
        forecast_hour: 6,
        source: SourceId::Aws,
        domain: domain(),
        input_fetches: Vec::new(),
        shared_timing: HrrrDerivedSharedTiming {
            fetch_decode: Some(crate::gridded::SharedTiming {
                fetch_surface_ms: 0,
                fetch_pressure_ms: 0,
                decode_surface_ms: 0,
                decode_pressure_ms: 0,
                fetch_surface_cache_hit: false,
                fetch_pressure_cache_hit: false,
                decode_surface_cache_hit: false,
                decode_pressure_cache_hit: false,
                surface_fetch: crate::gridded::FetchRuntimeInfo {
                    planned_bundle: rustwx_core::CanonicalBundleDescriptor::SurfaceAnalysis,
                    planned_family: rustwx_core::CanonicalDataFamily::Surface,
                    planned_product: "sfc".into(),
                    resolved_native_product: "sfc".into(),
                    fetched_product: "sfc".into(),
                    requested_source: SourceId::Aws,
                    resolved_source: SourceId::Aws,
                    resolved_url: "https://example.test/hrrr.t12z.wrfsfcf06.grib2".into(),
                },
                pressure_fetch: crate::gridded::FetchRuntimeInfo {
                    planned_bundle: rustwx_core::CanonicalBundleDescriptor::PressureAnalysis,
                    planned_family: rustwx_core::CanonicalDataFamily::Pressure,
                    planned_product: "prs".into(),
                    resolved_native_product: "prs".into(),
                    fetched_product: "prs".into(),
                    requested_source: SourceId::Aws,
                    resolved_source: SourceId::Aws,
                    resolved_url: "https://example.test/hrrr.t12z.wrfprsf06.grib2".into(),
                },
            }),
            compute_ms: 4,
            project_ms: 5,
            native_extract_ms: 0,
            native_compare_ms: 0,
            memory_profile: None,
            heavy_timing: None,
        },
        recipes: vec![HrrrDerivedRenderedRecipe {
            recipe_slug: "sbcape".into(),
            title: "SBCAPE".into(),
            source_route: crate::source::ProductSourceRoute::CanonicalDerived,
            output_path: PathBuf::from("C:\\proof\\derived.png"),
            content_identity: crate::publication::artifact_identity_from_bytes(b"derived"),
            input_fetch_keys: vec!["derived:sfc".into(), "derived:prs".into()],
            timing: HrrrDerivedRecipeTiming {
                render_to_image_ms: 0,
                data_layer_draw_ms: 0,
                overlay_draw_ms: 0,
                render_state_prep_ms: 0,
                png_encode_ms: 0,
                file_write_ms: 0,
                render_ms: 6,
                total_ms: 6,
                state_timing: Default::default(),
                image_timing: Default::default(),
            },
        }],
        source_mode: ProductSourceMode::Canonical,
        blockers: Vec::new(),
        native_thermo_artifacts: Vec::new(),
        total_ms: 11,
    };
    let windowed = HrrrWindowedBatchReport {
        model: ModelId::Hrrr,
        date_yyyymmdd: "20260415".into(),
        cycle_utc: 12,
        forecast_hour: 6,
        source: SourceId::Aws,
        domain: domain(),
        shared_timing: HrrrWindowedSharedTiming {
            fetch_geometry_ms: 0,
            decode_geometry_ms: 0,
            project_ms: 0,
            fetch_surface_ms: 0,
            decode_surface_ms: 0,
            fetch_nat_ms: 0,
            decode_nat_ms: 0,
            fetch_wind_ms: 0,
            decode_wind_ms: 0,
            fetch_temp_ms: 0,
            decode_temp_ms: 0,
            compute_products_ms: 0,
            geometry_fetch_cache_hit: false,
            geometry_decode_cache_hit: false,
            surface_hours_loaded: vec![6],
            nat_hours_loaded: vec![6],
            wind_hours_loaded: Vec::new(),
            temp_hours_loaded: Vec::new(),
            geometry_fetch: Some(HrrrFetchRuntimeInfo {
                planned_product: "sfc".into(),
                fetched_product: "sfc".into(),
                requested_source: SourceId::Aws,
                resolved_source: SourceId::Aws,
                resolved_url: "https://example.test/hrrr.t12z.wrfsfcf06.grib2".into(),
            }),
            geometry_input_fetch: Some(windowed_fetch_identity("sfc", "sfc", 6)),
            surface_hour_fetches: vec![HrrrWindowedHourFetchInfo {
                hour: 6,
                planned_product: "sfc".into(),
                fetched_product: "sfc".into(),
                requested_source: SourceId::Aws,
                resolved_source: SourceId::Aws,
                resolved_url: "https://example.test/hrrr.t12z.wrfsfcf06.grib2".into(),
                fetch_cache_hit: false,
                input_fetch: Some(windowed_fetch_identity("sfc", "sfc", 6)),
            }],
            uh_hour_fetches: vec![HrrrWindowedHourFetchInfo {
                hour: 6,
                planned_product: "nat".into(),
                fetched_product: "sfc".into(),
                requested_source: SourceId::Aws,
                resolved_source: SourceId::Aws,
                resolved_url: "https://example.test/hrrr.t12z.wrfsfcf06.grib2".into(),
                fetch_cache_hit: false,
                input_fetch: Some(windowed_fetch_identity("nat", "sfc", 6)),
            }],
            wind_hour_fetches: Vec::new(),
            temp_hour_fetches: Vec::new(),
        },
        products: vec![HrrrWindowedRenderedProduct {
            product: HrrrWindowedProduct::Qpf6h,
            output_path: PathBuf::from("C:\\proof\\windowed.png"),
            timing: HrrrWindowedProductTiming {
                compute_ms: 7,
                render_ms: 8,
                total_ms: 15,
            },
            metadata: HrrrWindowedProductMetadata {
                strategy: "direct APCP 6h accumulation".into(),
                contributing_forecast_hours: vec![1, 2, 3, 4, 5, 6],
                window_hours: Some(6),
            },
        }],
        blockers: vec![HrrrWindowedBlocker {
            product: HrrrWindowedProduct::Uh25kmRunMax,
            reason: "demo blocker".into(),
        }],
        total_ms: 12,
    };

    let summary = build_summary(&Some(direct), &Some(derived), &Some(windowed));
    assert_eq!(summary.runner_count, 3);
    assert_eq!(summary.direct_rendered_count, 1);
    assert_eq!(summary.derived_rendered_count, 1);
    assert_eq!(summary.windowed_rendered_count, 1);
    assert_eq!(summary.windowed_blocker_count, 1);
    assert_eq!(summary.output_count, 3);
    assert_eq!(
        summary.output_paths,
        vec![
            PathBuf::from("C:\\proof\\direct.png"),
            PathBuf::from("C:\\proof\\derived.png"),
            PathBuf::from("C:\\proof\\windowed.png"),
        ]
    );
}

#[test]
fn run_manifest_tracks_planned_complete_and_blocked_artifacts() {
    let requested = HrrrNonEcapeHourRequestedProducts {
        direct_recipe_slugs: vec!["500mb_height_winds".into()],
        derived_recipe_slugs: vec!["sbcape".into()],
        windowed_products: vec![HrrrWindowedProduct::Qpf6h, HrrrWindowedProduct::Qpf12h],
    };
    let mut manifest = build_run_manifest(
        ModelId::Hrrr,
        &requested,
        std::path::Path::new("C:\\proof\\run"),
        "rustwx_hrrr_20260415_12z_f006_conus_non_ecape_hour",
        "20260415",
        12,
        6,
        SourceId::Aws,
        "conus",
    );
    manifest.mark_running();

    let direct = HrrrDirectBatchReport {
        model: rustwx_core::ModelId::Hrrr,
        date_yyyymmdd: "20260415".into(),
        cycle_utc: 12,
        forecast_hour: 6,
        source: SourceId::Aws,
        domain: domain(),
        fetches: Vec::new(),
        recipes: vec![HrrrDirectRenderedRecipe {
            recipe_slug: "500mb_height_winds".into(),
            title: "500mb Height / Winds".into(),
            source_route: crate::source::ProductSourceRoute::DirectNativeExact,
            grib_product: "prs".into(),
            fetched_grib_product: "prs".into(),
            resolved_source: SourceId::Aws,
            resolved_url: "https://example.test/hrrr.t12z.wrfprsf06.grib2".into(),
            output_path: PathBuf::from(
                "C:\\proof\\run\\rustwx_hrrr_20260415_12z_f006_conus_500mb_height_winds.png",
            ),
            content_identity: crate::publication::artifact_identity_from_bytes(b"direct-run"),
            input_fetch_keys: vec!["direct:prs".into()],
            timing: HrrrDirectRecipeTiming {
                project_ms: 1,
                field_prepare_ms: 0,
                contour_prepare_ms: 0,
                barb_prepare_ms: 0,
                render_to_image_ms: 0,
                data_layer_draw_ms: 0,
                overlay_draw_ms: 0,
                panel_compose_ms: 0,
                request_build_ms: 0,
                render_state_prep_ms: 0,
                png_encode_ms: 0,
                file_write_ms: 0,
                render_ms: 2,
                total_ms: 3,
                state_timing: Default::default(),
                image_timing: Default::default(),
            },
        }],
        blockers: Vec::new(),
        total_ms: 10,
    };
    let derived = HrrrDerivedBatchReport {
        date_yyyymmdd: "20260415".into(),
        cycle_utc: 12,
        forecast_hour: 6,
        source: SourceId::Aws,
        domain: domain(),
        input_fetches: Vec::new(),
        shared_timing: HrrrDerivedSharedTiming {
            fetch_decode: Some(crate::gridded::SharedTiming {
                fetch_surface_ms: 0,
                fetch_pressure_ms: 0,
                decode_surface_ms: 0,
                decode_pressure_ms: 0,
                fetch_surface_cache_hit: false,
                fetch_pressure_cache_hit: false,
                decode_surface_cache_hit: false,
                decode_pressure_cache_hit: false,
                surface_fetch: crate::gridded::FetchRuntimeInfo {
                    planned_bundle: rustwx_core::CanonicalBundleDescriptor::SurfaceAnalysis,
                    planned_family: rustwx_core::CanonicalDataFamily::Surface,
                    planned_product: "sfc".into(),
                    resolved_native_product: "sfc".into(),
                    fetched_product: "sfc".into(),
                    requested_source: SourceId::Aws,
                    resolved_source: SourceId::Aws,
                    resolved_url: "https://example.test/hrrr.t12z.wrfsfcf06.grib2".into(),
                },
                pressure_fetch: crate::gridded::FetchRuntimeInfo {
                    planned_bundle: rustwx_core::CanonicalBundleDescriptor::PressureAnalysis,
                    planned_family: rustwx_core::CanonicalDataFamily::Pressure,
                    planned_product: "prs".into(),
                    resolved_native_product: "prs".into(),
                    fetched_product: "prs".into(),
                    requested_source: SourceId::Aws,
                    resolved_source: SourceId::Aws,
                    resolved_url: "https://example.test/hrrr.t12z.wrfprsf06.grib2".into(),
                },
            }),
            compute_ms: 1,
            project_ms: 1,
            native_extract_ms: 0,
            native_compare_ms: 0,
            memory_profile: None,
            heavy_timing: None,
        },
        recipes: vec![HrrrDerivedRenderedRecipe {
            recipe_slug: "sbcape".into(),
            title: "SBCAPE".into(),
            source_route: crate::source::ProductSourceRoute::CanonicalDerived,
            output_path: PathBuf::from(
                "C:\\proof\\run\\rustwx_hrrr_20260415_12z_f006_conus_sbcape.png",
            ),
            content_identity: crate::publication::artifact_identity_from_bytes(b"derived-run"),
            input_fetch_keys: vec!["derived:sfc".into(), "derived:prs".into()],
            timing: HrrrDerivedRecipeTiming {
                render_to_image_ms: 0,
                data_layer_draw_ms: 0,
                overlay_draw_ms: 0,
                render_state_prep_ms: 0,
                png_encode_ms: 0,
                file_write_ms: 0,
                render_ms: 1,
                total_ms: 1,
                state_timing: Default::default(),
                image_timing: Default::default(),
            },
        }],
        source_mode: ProductSourceMode::Canonical,
        blockers: Vec::new(),
        native_thermo_artifacts: Vec::new(),
        total_ms: 5,
    };
    let windowed = HrrrWindowedBatchReport {
        model: ModelId::Hrrr,
        date_yyyymmdd: "20260415".into(),
        cycle_utc: 12,
        forecast_hour: 6,
        source: SourceId::Aws,
        domain: domain(),
        shared_timing: HrrrWindowedSharedTiming {
            fetch_geometry_ms: 0,
            decode_geometry_ms: 0,
            project_ms: 0,
            fetch_surface_ms: 0,
            decode_surface_ms: 0,
            fetch_nat_ms: 0,
            decode_nat_ms: 0,
            fetch_wind_ms: 0,
            decode_wind_ms: 0,
            fetch_temp_ms: 0,
            decode_temp_ms: 0,
            compute_products_ms: 0,
            geometry_fetch_cache_hit: false,
            geometry_decode_cache_hit: false,
            surface_hours_loaded: vec![6],
            nat_hours_loaded: vec![6],
            wind_hours_loaded: Vec::new(),
            temp_hours_loaded: Vec::new(),
            geometry_fetch: Some(HrrrFetchRuntimeInfo {
                planned_product: "sfc".into(),
                fetched_product: "sfc".into(),
                requested_source: SourceId::Aws,
                resolved_source: SourceId::Aws,
                resolved_url: "https://example.test/hrrr.t12z.wrfsfcf06.grib2".into(),
            }),
            geometry_input_fetch: Some(windowed_fetch_identity("sfc", "sfc", 6)),
            surface_hour_fetches: vec![HrrrWindowedHourFetchInfo {
                hour: 6,
                planned_product: "sfc".into(),
                fetched_product: "sfc".into(),
                requested_source: SourceId::Aws,
                resolved_source: SourceId::Aws,
                resolved_url: "https://example.test/hrrr.t12z.wrfsfcf06.grib2".into(),
                fetch_cache_hit: false,
                input_fetch: Some(windowed_fetch_identity("sfc", "sfc", 6)),
            }],
            uh_hour_fetches: vec![HrrrWindowedHourFetchInfo {
                hour: 6,
                planned_product: "nat".into(),
                fetched_product: "sfc".into(),
                requested_source: SourceId::Aws,
                resolved_source: SourceId::Aws,
                resolved_url: "https://example.test/hrrr.t12z.wrfsfcf06.grib2".into(),
                fetch_cache_hit: false,
                input_fetch: Some(windowed_fetch_identity("nat", "sfc", 6)),
            }],
            wind_hour_fetches: Vec::new(),
            temp_hour_fetches: Vec::new(),
        },
        products: vec![HrrrWindowedRenderedProduct {
            product: HrrrWindowedProduct::Qpf6h,
            output_path: PathBuf::from(
                "C:\\proof\\run\\rustwx_hrrr_20260415_12z_f006_conus_qpf_6h.png",
            ),
            timing: HrrrWindowedProductTiming {
                compute_ms: 1,
                render_ms: 1,
                total_ms: 2,
            },
            metadata: HrrrWindowedProductMetadata {
                strategy: "test".into(),
                contributing_forecast_hours: vec![1, 2, 3, 4, 5, 6],
                window_hours: Some(6),
            },
        }],
        blockers: vec![HrrrWindowedBlocker {
            product: HrrrWindowedProduct::Qpf12h,
            reason: "not enough hours".into(),
        }],
        total_ms: 2,
    };

    apply_direct_manifest_updates(&mut manifest, &Some(direct));
    apply_derived_manifest_updates(&mut manifest, &Some(derived));
    apply_windowed_manifest_updates(&mut manifest, &Some(windowed));
    assert_eq!(count_blocked_artifacts(&manifest), 1);

    let direct_record = manifest
        .artifacts
        .iter()
        .find(|artifact| artifact.artifact_key == "direct:500mb_height_winds")
        .unwrap();
    assert_eq!(direct_record.state, ArtifactPublicationState::Complete);
    assert!(
        direct_record
            .detail
            .as_deref()
            .unwrap()
            .contains("planned_family=prs fetched_family=prs resolved_source=aws")
    );

    let derived_record = manifest
        .artifacts
        .iter()
        .find(|artifact| artifact.artifact_key == "derived:sbcape")
        .unwrap();
    assert_eq!(derived_record.state, ArtifactPublicationState::Complete);
    assert!(
        derived_record
            .detail
            .as_deref()
            .unwrap()
            .contains("shared_surface planned_family=sfc fetched_family=sfc resolved_source=aws")
    );

    let blocked_record = manifest
        .artifacts
        .iter()
        .find(|artifact| artifact.artifact_key == "windowed:qpf_12h")
        .unwrap();
    assert_eq!(blocked_record.state, ArtifactPublicationState::Blocked);
    assert_eq!(blocked_record.detail.as_deref(), Some("not enough hours"));
}

#[test]
fn windowed_input_fetch_keys_follow_contributing_hours_without_cache() {
    let product = HrrrWindowedRenderedProduct {
        product: HrrrWindowedProduct::Qpf1h,
        output_path: PathBuf::from("C:\\proof\\qpf_1h.png"),
        timing: HrrrWindowedProductTiming {
            compute_ms: 1,
            render_ms: 1,
            total_ms: 2,
        },
        metadata: HrrrWindowedProductMetadata {
            strategy: "direct APCP 1h accumulation".into(),
            contributing_forecast_hours: vec![6],
            window_hours: Some(1),
        },
    };
    let shared_timing = HrrrWindowedSharedTiming {
        fetch_geometry_ms: 0,
        decode_geometry_ms: 0,
        project_ms: 0,
        fetch_surface_ms: 0,
        decode_surface_ms: 0,
        fetch_nat_ms: 0,
        decode_nat_ms: 0,
        fetch_wind_ms: 0,
        decode_wind_ms: 0,
        fetch_temp_ms: 0,
        decode_temp_ms: 0,
        compute_products_ms: 0,
        geometry_fetch_cache_hit: false,
        geometry_decode_cache_hit: false,
        surface_hours_loaded: vec![5, 6],
        nat_hours_loaded: Vec::new(),
        wind_hours_loaded: Vec::new(),
        temp_hours_loaded: Vec::new(),
        geometry_fetch: None,
        geometry_input_fetch: None,
        surface_hour_fetches: vec![
            HrrrWindowedHourFetchInfo {
                hour: 5,
                planned_product: "sfc".into(),
                fetched_product: "sfc".into(),
                requested_source: SourceId::Aws,
                resolved_source: SourceId::Aws,
                resolved_url: "https://example.test/hrrr.t12z.wrfsfcf05.grib2".into(),
                fetch_cache_hit: false,
                input_fetch: Some(windowed_fetch_identity("sfc", "sfc", 5)),
            },
            HrrrWindowedHourFetchInfo {
                hour: 6,
                planned_product: "sfc".into(),
                fetched_product: "sfc".into(),
                requested_source: SourceId::Aws,
                resolved_source: SourceId::Aws,
                resolved_url: "https://example.test/hrrr.t12z.wrfsfcf06.grib2".into(),
                fetch_cache_hit: false,
                input_fetch: Some(windowed_fetch_identity("sfc", "sfc", 6)),
            },
        ],
        uh_hour_fetches: Vec::new(),
        wind_hour_fetches: Vec::new(),
        temp_hour_fetches: Vec::new(),
    };

    let keys = windowed_product_input_fetch_keys(&product, &shared_timing);
    assert_eq!(
        keys,
        vec![windowed_fetch_identity("sfc", "sfc", 6).fetch_key]
    );
}

#[test]
fn collect_input_fetches_keeps_windowed_lineage_when_cache_is_off() {
    let report = HrrrWindowedBatchReport {
        model: ModelId::Hrrr,
        date_yyyymmdd: "20260415".into(),
        cycle_utc: 12,
        forecast_hour: 6,
        source: SourceId::Aws,
        domain: domain(),
        shared_timing: HrrrWindowedSharedTiming {
            fetch_geometry_ms: 0,
            decode_geometry_ms: 0,
            project_ms: 0,
            fetch_surface_ms: 0,
            decode_surface_ms: 0,
            fetch_nat_ms: 0,
            decode_nat_ms: 0,
            fetch_wind_ms: 0,
            decode_wind_ms: 0,
            fetch_temp_ms: 0,
            decode_temp_ms: 0,
            compute_products_ms: 0,
            geometry_fetch_cache_hit: false,
            geometry_decode_cache_hit: false,
            surface_hours_loaded: vec![6],
            nat_hours_loaded: vec![6],
            wind_hours_loaded: Vec::new(),
            temp_hours_loaded: Vec::new(),
            geometry_fetch: None,
            geometry_input_fetch: Some(windowed_fetch_identity("sfc", "sfc", 6)),
            surface_hour_fetches: vec![HrrrWindowedHourFetchInfo {
                hour: 6,
                planned_product: "sfc".into(),
                fetched_product: "sfc".into(),
                requested_source: SourceId::Aws,
                resolved_source: SourceId::Aws,
                resolved_url: "https://example.test/hrrr.t12z.wrfsfcf06.grib2".into(),
                fetch_cache_hit: false,
                input_fetch: Some(windowed_fetch_identity("sfc", "sfc", 6)),
            }],
            uh_hour_fetches: vec![HrrrWindowedHourFetchInfo {
                hour: 6,
                planned_product: "nat".into(),
                fetched_product: "sfc".into(),
                requested_source: SourceId::Aws,
                resolved_source: SourceId::Aws,
                resolved_url: "https://example.test/hrrr.t12z.wrfsfcf06.grib2".into(),
                fetch_cache_hit: false,
                input_fetch: Some(windowed_fetch_identity("nat", "sfc", 6)),
            }],
            wind_hour_fetches: Vec::new(),
            temp_hour_fetches: Vec::new(),
        },
        products: Vec::new(),
        blockers: Vec::new(),
        total_ms: 1,
    };

    let fetches = collect_input_fetches(&None, &None, &Some(report));
    let keys = fetches
        .into_iter()
        .map(|fetch| fetch.fetch_key)
        .collect::<Vec<_>>();
    assert!(keys.contains(&windowed_fetch_identity("sfc", "sfc", 6).fetch_key));
    assert!(keys.contains(&windowed_fetch_identity("nat", "sfc", 6).fetch_key));
}

#[test]
fn non_ecape_report_serialization_keeps_cache_mode_for_benchmarks() {
    let report = HrrrNonEcapeHourReport {
        date_yyyymmdd: "20260415".into(),
        cycle_utc: 12,
        forecast_hour: 6,
        source: SourceId::Aws,
        domain: domain(),
        out_dir: PathBuf::from("C:\\proof\\bench"),
        cache_root: PathBuf::from("C:\\proof\\bench\\cache"),
        use_cache: false,
        source_mode: ProductSourceMode::Canonical,
        publication_manifest_path: PathBuf::from("C:\\proof\\bench\\run_manifest.json"),
        attempt_manifest_path: None,
        requested: HrrrNonEcapeHourRequestedProducts {
            direct_recipe_slugs: vec!["500mb_height_winds".into()],
            derived_recipe_slugs: vec!["sbcape".into()],
            windowed_products: vec![HrrrWindowedProduct::Qpf6h],
        },
        shared_timing: HrrrNonEcapeSharedTiming::default(),
        summary: HrrrNonEcapeHourSummary {
            runner_count: 1,
            direct_rendered_count: 1,
            derived_rendered_count: 0,
            windowed_rendered_count: 0,
            windowed_blocker_count: 0,
            output_count: 1,
            output_paths: vec![PathBuf::from("C:\\proof\\bench\\out.png")],
        },
        direct: None,
        derived: None,
        windowed: None,
        total_ms: 1234,
    };

    let json = serde_json::to_string(&report).unwrap();
    assert!(
        json.contains("\"use_cache\":false"),
        "cold benchmark reports should serialize cache mode explicitly"
    );
}
