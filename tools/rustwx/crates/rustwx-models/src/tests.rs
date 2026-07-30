use super::*;

#[test]
fn built_in_models_are_real() {
    assert_eq!(built_in_models().len(), 23);
    assert_eq!(model_summary(ModelId::HrrrAk).default_product, "sfc");
    assert_eq!(model_summary(ModelId::Gdas).default_product, "pgrb2.0p25");
    assert_eq!(
        model_summary(ModelId::Gefs).default_product,
        "pgrb2ap5/gec00"
    );
    assert_eq!(model_summary(ModelId::Aigfs).default_product, "sfc");
    assert_eq!(model_summary(ModelId::Aigefs).default_product, "sfc/avg");
    assert_eq!(model_summary(ModelId::Hgefs).default_product, "sfc/avg");
    assert_eq!(model_summary(ModelId::Hgefs).max_forecast_hour, 240);
    assert_eq!(model_summary(ModelId::Aifs).max_forecast_hour, 43_848);
    assert_eq!(
        model_summary(ModelId::Href).default_product,
        "ensprod/conus/sprd"
    );
    assert_eq!(
        model_summary(ModelId::Href).cycle_hours_utc,
        HREF_CYCLE_HOURS
    );
    assert_eq!(model_summary(ModelId::Rtma).max_forecast_hour, 0);
    assert_eq!(model_summary(ModelId::Nbm).default_product, "core/co");
    assert_eq!(model_summary(ModelId::RrfsA).default_product, "prs-conus");
    assert_eq!(
        model_summary(ModelId::RrfsPublic).cycle_hours_utc,
        RRFS_PUBLIC_CYCLE_HOURS
    );
    assert_eq!(model_summary(ModelId::Refs).default_product, "mean-conus");
    assert_eq!(
        model_summary(ModelId::Refs).cycle_hours_utc,
        REFS_CYCLE_HOURS
    );
    assert_eq!(model_summary(ModelId::Refs).max_forecast_hour, 60);
    assert_eq!(
        model_summary(ModelId::RrfsFireWx).cycle_hours_utc,
        RRFS_FIREWX_CYCLE_HOURS
    );
    assert_eq!(model_summary(ModelId::RrfsFireWx).max_forecast_hour, 36);
    assert_eq!(
        model_summary(ModelId::WrfGdex).default_product,
        WRF_GDEX_DEFAULT_SURFACE_PRODUCT
    );
}

#[test]
fn catalog_exposes_the_user_facing_supported_models() {
    let expected = [
        ModelId::Hrrr,
        ModelId::HrrrAk,
        ModelId::Rap,
        ModelId::Gfs,
        ModelId::Gdas,
        ModelId::Gefs,
        ModelId::Aigfs,
        ModelId::Aigefs,
        ModelId::Hgefs,
        ModelId::EcmwfOpenData,
        ModelId::Nam,
        ModelId::RrfsA,
        ModelId::Refs,
        ModelId::Nbm,
    ];
    assert_eq!(supported_models(), expected);

    // Every supported model must have real registry plumbing behind it.
    for model in supported_models() {
        let summary = model_summary(model);
        assert_eq!(summary.id, model);
        assert!(!summary.default_product.is_empty());
    }
}

#[test]
fn canonical_bundle_products_resolve_through_model_adapter() {
    let hrrr_surface = resolve_canonical_bundle_product(
        ModelId::Hrrr,
        CanonicalBundleDescriptor::SurfaceAnalysis,
        None,
    );
    assert_eq!(hrrr_surface.family, CanonicalDataFamily::Surface);
    assert_eq!(hrrr_surface.native_product, "sfc");

    let hrrr_pressure = resolve_canonical_bundle_product(
        ModelId::Hrrr,
        CanonicalBundleDescriptor::PressureAnalysis,
        None,
    );
    assert_eq!(hrrr_pressure.family, CanonicalDataFamily::Pressure);
    assert_eq!(hrrr_pressure.native_product, "prs");

    let nam_surface = resolve_canonical_bundle_product(
        ModelId::Nam,
        CanonicalBundleDescriptor::SurfaceAnalysis,
        None,
    );
    assert_eq!(nam_surface.native_product, "awip3d");

    let nam_pressure = resolve_canonical_bundle_product(
        ModelId::Nam,
        CanonicalBundleDescriptor::PressureAnalysis,
        None,
    );
    assert_eq!(nam_pressure.native_product, "awip3d");

    let rrfs_pressure = resolve_canonical_bundle_product(
        ModelId::RrfsA,
        CanonicalBundleDescriptor::PressureAnalysis,
        Some("prs-na"),
    );
    assert_eq!(
        rrfs_pressure.bundle,
        CanonicalBundleDescriptor::PressureAnalysis
    );
    assert_eq!(rrfs_pressure.family, CanonicalDataFamily::Pressure);
    assert_eq!(rrfs_pressure.native_product, "prs-na");

    let wrf_surface = resolve_canonical_bundle_product(
        ModelId::WrfGdex,
        CanonicalBundleDescriptor::SurfaceAnalysis,
        None,
    );
    assert_eq!(wrf_surface.native_product, WRF_GDEX_DEFAULT_SURFACE_PRODUCT);

    let wrf_pressure = resolve_canonical_bundle_product(
        ModelId::WrfGdex,
        CanonicalBundleDescriptor::PressureAnalysis,
        None,
    );
    assert_eq!(
        wrf_pressure.native_product,
        WRF_GDEX_DEFAULT_PRESSURE_PRODUCT
    );
}

#[test]
fn built_in_plot_recipes_cover_current_direct_atmos_surface_and_radar_maps() {
    assert!(plot_recipe("200mb_height_winds").is_some());
    assert!(plot_recipe("300mb_height_winds").is_some());
    assert!(plot_recipe("250mb_height_winds").is_some());
    assert!(plot_recipe("500mb_height_winds").is_some());
    assert!(plot_recipe("700mb_height_winds").is_some());
    assert!(plot_recipe("850mb_height_winds").is_some());
    assert!(plot_recipe("200mb_temperature_height_winds").is_some());
    assert!(plot_recipe("300mb_temperature_height_winds").is_some());
    assert!(plot_recipe("250mb_temperature_height_winds").is_some());
    assert!(plot_recipe("500mb_temperature_height_winds").is_some());
    assert!(plot_recipe("700mb_temperature_height_winds").is_some());
    assert!(plot_recipe("850mb_temperature_height_winds").is_some());
    assert!(plot_recipe("2m_relative_humidity").is_some());
    assert!(plot_recipe("2m_relative_humidity_10m_winds").is_some());
    assert!(plot_recipe("2m_temperature").is_some());
    assert!(plot_recipe("2m_temperature_10m_winds").is_some());
    assert!(plot_recipe("2m_dewpoint").is_some());
    assert!(plot_recipe("2m_dewpoint_10m_winds").is_some());
    assert!(plot_recipe("mslp_10m_winds").is_some());
    assert!(plot_recipe("10m_wind_gusts").is_some());
    assert!(plot_recipe("precipitable_water").is_some());
    assert!(plot_recipe("cloud_cover").is_some());
    assert!(plot_recipe("visibility").is_some());
    assert!(plot_recipe("simulated_ir_satellite").is_some());
    assert!(plot_recipe("700mb_dewpoint_height_winds").is_some());
    assert!(plot_recipe("850mb_dewpoint_height_winds").is_some());
    assert!(plot_recipe("200mb_rh_height_winds").is_some());
    assert!(plot_recipe("300mb_rh_height_winds").is_some());
    assert!(plot_recipe("500mb_absolute_vorticity_height_winds").is_some());
    assert!(plot_recipe("200mb_absolute_vorticity_height_winds").is_some());
    assert!(plot_recipe("300mb_absolute_vorticity_height_winds").is_some());
    assert!(plot_recipe("500mb_rh_height_winds").is_some());
    assert!(plot_recipe("700mb_rh_height_winds").is_some());
    assert!(plot_recipe("700mb_absolute_vorticity_height_winds").is_some());
    assert!(plot_recipe("1km_reflectivity").is_some());
    assert!(plot_recipe("composite_reflectivity").is_some());
    assert!(plot_recipe("composite_reflectivity_uh").is_some());
    assert!(plot_recipe("smoke_pm25_native").is_some());
    assert!(plot_recipe("smoke_column").is_some());
}

#[test]
fn grib_field_spec_exposes_typed_product_metadata() {
    let metadata = FIELD_500_TEMP.product_metadata();
    assert_eq!(metadata.display_name, "500mb Temperature");
    assert_eq!(metadata.category.as_deref(), Some("pressure"));
    assert_eq!(metadata.native_units.as_deref(), Some("K"));
    let provenance = metadata
        .provenance
        .expect("field metadata should carry provenance");
    assert_eq!(provenance.lineage, ProductLineage::Direct);
    assert_eq!(provenance.maturity, ProductMaturity::Operational);
    assert_eq!(
        provenance.selector,
        Some(FieldSelector::isobaric(CanonicalField::Temperature, 500))
    );

    let smoke_metadata = FIELD_SMOKE_MASS_DENSITY_8M.product_metadata();
    assert_eq!(smoke_metadata.display_name, "8m AGL Smoke Mass Density");
    assert_eq!(smoke_metadata.category.as_deref(), Some("native"));
    assert_eq!(smoke_metadata.native_units.as_deref(), Some("kg/m^3"));

    let column_metadata = FIELD_COLUMN_INTEGRATED_SMOKE.product_metadata();
    assert_eq!(column_metadata.display_name, "Column-Integrated Smoke");
    assert_eq!(column_metadata.category.as_deref(), Some("native"));
    assert_eq!(column_metadata.native_units.as_deref(), Some("kg/m^2"));
}

#[test]
fn plot_recipe_metadata_marks_derived_windowed_and_composite_routes() {
    let heat_index = plot_recipe("2m_heat_index").expect("heat index recipe should exist");
    assert_eq!(heat_index.provenance().lineage, ProductLineage::Derived);

    let qpf_1h = plot_recipe("1h_qpf").expect("1h qpf recipe should exist");
    let qpf_provenance = qpf_1h.provenance();
    assert_eq!(qpf_provenance.lineage, ProductLineage::Windowed);
    assert_eq!(
        qpf_provenance.window,
        Some(ProductWindowSpec::accumulation(Some(1)))
    );
    assert!(qpf_provenance.flags.contains(&ProductSemanticFlag::Alias));

    let refl_uh =
        plot_recipe("composite_reflectivity_uh").expect("reflectivity+UH recipe should exist");
    assert!(
        refl_uh
            .provenance()
            .flags
            .contains(&ProductSemanticFlag::Composite)
    );
}

#[test]
fn experimental_recipe_metadata_is_explicit() {
    let simulated_ir =
        plot_recipe("simulated_ir_satellite").expect("simulated ir recipe should exist");
    assert_eq!(
        simulated_ir.product_metadata().provenance.unwrap().maturity,
        ProductMaturity::Experimental
    );

    let lightning = plot_recipe("lightning_flash_density").expect("lightning recipe should exist");
    assert_eq!(
        lightning.product_metadata().provenance.unwrap().maturity,
        ProductMaturity::Experimental
    );

    for slug in [
        "href_sprd_2m_temperature",
        "href_prob_2m_temperature_below_273p15k",
        "href_mean_2m_temperature",
        "refs_sprd_2m_temperature",
        "refs_prob_2m_temperature_below_273p15k",
    ] {
        let recipe = plot_recipe(slug).expect("experimental recipe should exist");
        let provenance = recipe.product_metadata().provenance.unwrap();
        assert_eq!(provenance.maturity, ProductMaturity::Experimental, "{slug}");
        assert!(
            provenance
                .flags
                .contains(&ProductSemanticFlag::ProofOriented),
            "{slug}"
        );
    }
}

#[test]
fn plot_recipe_alias_lookup_normalizes_tokens() {
    let recipe = plot_recipe("500MB temperature height winds").unwrap();
    assert_eq!(recipe.slug, "500mb_temperature_height_winds");
    assert_eq!(recipe.filled.level_value, Some(500));
    assert_eq!(recipe.barbs_u.as_ref().unwrap().key, "u_500mb");
    assert_eq!(
        recipe.filled.selector,
        Some(FieldSelector::isobaric(CanonicalField::Temperature, 500))
    );

    let absolute_vorticity = plot_recipe("500MB vorticity height winds").unwrap();
    assert_eq!(
        absolute_vorticity.slug,
        "500mb_absolute_vorticity_height_winds"
    );
    assert_eq!(
        absolute_vorticity.filled.selector,
        Some(FieldSelector::isobaric(
            CanonicalField::AbsoluteVorticity,
            500,
        ))
    );

    let temp_2m = plot_recipe("2m temperature 10m winds").unwrap();
    assert_eq!(temp_2m.slug, "2m_temperature_10m_winds");
    assert_eq!(
        temp_2m.filled.selector,
        Some(FieldSelector::height_agl(CanonicalField::Temperature, 2))
    );

    let reflectivity_1km = plot_recipe("1km reflectivity").unwrap();
    assert_eq!(reflectivity_1km.slug, "1km_reflectivity");
    assert_eq!(
        reflectivity_1km.filled.selector,
        Some(FieldSelector::height_agl(
            CanonicalField::RadarReflectivity,
            1000,
        ))
    );
}

#[test]
fn composite_reflectivity_uh_recipe_requires_native_reflectivity_and_uh() {
    let recipe = plot_recipe("composite_reflectivity_uh").unwrap();
    assert_eq!(recipe.filled.family, ProductFamily::Native);
    assert_eq!(recipe.filled.idx_patterns()[0], "REFC:entire atmosphere");
    assert_eq!(
        recipe.contours.as_ref().unwrap().idx_patterns()[0],
        "MXUPHL:5000-2000"
    );
    assert!(recipe.barbs_u.is_none());
    assert!(recipe.barbs_v.is_none());
}

#[test]
fn surface_wind_recipes_include_mslp_contours() {
    for slug in [
        "2m_relative_humidity_10m_winds",
        "2m_temperature_10m_winds",
        "2m_dewpoint_10m_winds",
        "mslp_10m_winds",
    ] {
        let recipe = plot_recipe(slug).unwrap();
        assert_eq!(
            recipe.contours.as_ref().unwrap().selector,
            Some(FieldSelector::mean_sea_level(
                CanonicalField::PressureReducedToMeanSeaLevel,
            ))
        );
        assert_eq!(
            recipe.barbs_u.as_ref().unwrap().selector,
            Some(FieldSelector::height_agl(CanonicalField::UWind, 10))
        );
        assert_eq!(
            recipe.barbs_v.as_ref().unwrap().selector,
            Some(FieldSelector::height_agl(CanonicalField::VWind, 10))
        );
    }
}

#[test]
fn smoke_recipes_are_selector_backed_native_hrrr_products() {
    let smoke = plot_recipe("smoke_pm25_native").unwrap();
    assert_eq!(smoke.filled.family, ProductFamily::Native);
    assert_eq!(
        smoke.filled.selector,
        Some(FieldSelector::height_agl(
            CanonicalField::SmokeMassDensity,
            8
        ))
    );
    assert_eq!(smoke.filled.idx_patterns()[0], "MASSDEN:8 m above ground");

    let column = plot_recipe("smoke_column").unwrap();
    assert_eq!(column.filled.family, ProductFamily::Native);
    assert_eq!(
        column.filled.selector,
        Some(FieldSelector::entire_atmosphere(
            CanonicalField::ColumnIntegratedSmoke
        ))
    );
    assert_eq!(
        column.filled.idx_patterns()[0],
        "COLMD:entire atmosphere (considered as a single layer)"
    );
}

#[test]
fn selector_backed_temperature_recipe_produces_gfs_fetch_plan() {
    let plan = plot_recipe_fetch_plan("500mb_temperature_height_winds", ModelId::Gfs).unwrap();
    assert_eq!(plan.product, "pgrb2.0p25");
    assert_eq!(
        plan.fetch_policy,
        PlotRecipeFetchPolicy::PreferIndexedSubset
    );
    assert_eq!(plan.fetch_mode, PlotRecipeFetchMode::IndexedSubset);
    assert_eq!(plan.fields.len(), 4);
    assert_eq!(
        plan.selectors(),
        vec![
            FieldSelector::isobaric(CanonicalField::Temperature, 500),
            FieldSelector::isobaric(CanonicalField::GeopotentialHeight, 500),
            FieldSelector::isobaric(CanonicalField::UWind, 500),
            FieldSelector::isobaric(CanonicalField::VWind, 500),
        ]
    );
    assert!(!plan.variable_patterns().is_empty());
}

#[test]
fn selector_backed_200mb_temperature_recipe_produces_gfs_fetch_plan() {
    let plan = plot_recipe_fetch_plan("200mb_temperature_height_winds", ModelId::Gfs).unwrap();
    assert_eq!(plan.product, "pgrb2.0p25");
    assert_eq!(
        plan.selectors(),
        vec![
            FieldSelector::isobaric(CanonicalField::Temperature, 200),
            FieldSelector::isobaric(CanonicalField::GeopotentialHeight, 200),
            FieldSelector::isobaric(CanonicalField::UWind, 200),
            FieldSelector::isobaric(CanonicalField::VWind, 200),
        ]
    );
    assert_eq!(
        plan.fetch_policy,
        PlotRecipeFetchPolicy::PreferIndexedSubset
    );
    assert_eq!(plan.fetch_mode, PlotRecipeFetchMode::IndexedSubset);
    assert!(!plan.variable_patterns().is_empty());
}

#[test]
fn selector_backed_temperature_recipe_produces_ecmwf_whole_file_fetch_plan() {
    let plan =
        plot_recipe_fetch_plan("500mb_temperature_height_winds", ModelId::EcmwfOpenData).unwrap();
    assert_eq!(plan.product, "oper");
    assert_eq!(plan.fetch_policy, PlotRecipeFetchPolicy::WholeFile);
    assert_eq!(
        plan.fetch_mode,
        PlotRecipeFetchMode::WholeFileStructuredExtract
    );
    assert_eq!(
        plan.selectors(),
        vec![
            FieldSelector::isobaric(CanonicalField::Temperature, 500),
            FieldSelector::isobaric(CanonicalField::GeopotentialHeight, 500),
            FieldSelector::isobaric(CanonicalField::UWind, 500),
            FieldSelector::isobaric(CanonicalField::VWind, 500),
        ]
    );
    assert!(plan.variable_patterns().is_empty());
}

#[test]
fn rh_recipe_blocker_is_explicit_for_gfs() {
    let blockers = plot_recipe_fetch_blockers("500mb_rh_height_winds", ModelId::Gfs).unwrap();
    assert!(blockers.is_empty());
}

#[test]
fn selector_gap_is_explicit_for_rh_recipe() {
    let plan = plot_recipe_fetch_plan("500mb_rh_height_winds", ModelId::Gfs).unwrap();
    assert_eq!(
        plan.selectors(),
        vec![
            FieldSelector::isobaric(CanonicalField::RelativeHumidity, 500),
            FieldSelector::isobaric(CanonicalField::GeopotentialHeight, 500),
            FieldSelector::isobaric(CanonicalField::UWind, 500),
            FieldSelector::isobaric(CanonicalField::VWind, 500),
        ]
    );
}

#[test]
fn selector_backed_300mb_rh_recipe_produces_gfs_fetch_plan() {
    let plan = plot_recipe_fetch_plan("300mb_rh_height_winds", ModelId::Gfs).unwrap();
    assert_eq!(
        plan.selectors(),
        vec![
            FieldSelector::isobaric(CanonicalField::RelativeHumidity, 300),
            FieldSelector::isobaric(CanonicalField::GeopotentialHeight, 300),
            FieldSelector::isobaric(CanonicalField::UWind, 300),
            FieldSelector::isobaric(CanonicalField::VWind, 300),
        ]
    );
    assert_eq!(
        plan.fetch_policy,
        PlotRecipeFetchPolicy::PreferIndexedSubset
    );
    assert_eq!(plan.fetch_mode, PlotRecipeFetchMode::IndexedSubset);
    assert!(!plan.variable_patterns().is_empty());
}

#[test]
fn temperature_700_recipe_tracks_model_support() {
    let selectors = vec![
        FieldSelector::isobaric(CanonicalField::Temperature, 700),
        FieldSelector::isobaric(CanonicalField::GeopotentialHeight, 700),
        FieldSelector::isobaric(CanonicalField::UWind, 700),
        FieldSelector::isobaric(CanonicalField::VWind, 700),
    ];

    for model in [
        ModelId::Hrrr,
        ModelId::HrrrAk,
        ModelId::Gfs,
        ModelId::Gdas,
        ModelId::Gefs,
        ModelId::Aigfs,
        ModelId::Aigefs,
        ModelId::EcmwfOpenData,
        ModelId::Aifs,
        ModelId::Rap,
        ModelId::Nam,
        ModelId::Hiresw,
        ModelId::Href,
        ModelId::Sref,
        ModelId::Rtma,
        ModelId::Urma,
        ModelId::Nbm,
        ModelId::RrfsA,
        ModelId::RrfsPublic,
        ModelId::Refs,
        ModelId::WrfGdex,
    ] {
        if selectors
            .iter()
            .all(|selector| selector_supported_for_model(*selector, model))
        {
            let plan = plot_recipe_fetch_plan("700mb_temperature_height_winds", model).unwrap();
            assert_eq!(plan.selectors(), selectors);
            assert!(
                plot_recipe_fetch_blockers("700mb_temperature_height_winds", model)
                    .unwrap()
                    .is_empty()
            );
        } else {
            let blockers =
                plot_recipe_fetch_blockers("700mb_temperature_height_winds", model).unwrap();
            assert_eq!(
                blockers
                    .iter()
                    .map(|blocker| blocker.field_key)
                    .collect::<Vec<_>>(),
                vec!["temperature_700mb", "height_700mb", "u_700mb", "v_700mb"]
            );
            let reason = &blockers[0].reason;
            if matches!(model, ModelId::Rtma | ModelId::Urma | ModelId::Nbm) {
                assert!(reason.contains("surface/core grids"));
            } else if model == ModelId::Href {
                assert!(reason.contains("limited to explicit `href_sprd_*`"));
            } else if model == ModelId::Refs {
                assert!(reason.contains("limited to explicit `refs_sprd_*`"));
            } else {
                assert!(reason.contains("700 hPa temperature/height/wind selectors"));
            }
            match model {
                ModelId::EcmwfOpenData | ModelId::WrfGdex => {
                    assert!(reason.contains("whole-file structured extraction"));
                }
                ModelId::Hrrr
                | ModelId::HrrrAk
                | ModelId::Gfs
                | ModelId::Gdas
                | ModelId::Gefs
                | ModelId::Aigfs
                | ModelId::Aigefs
                | ModelId::Hgefs
                | ModelId::Rap
                | ModelId::Nam
                | ModelId::Hiresw
                | ModelId::Sref
                | ModelId::RrfsA
                | ModelId::RrfsPublic
                | ModelId::RrfsFireWx => {
                    assert!(reason.contains("idx subsetting can stage the GRIB messages"));
                }
                ModelId::Refs => {
                    assert!(reason.contains("limited to explicit `refs_sprd_*`"));
                }
                ModelId::Rtma | ModelId::Urma | ModelId::Nbm => {
                    assert!(reason.contains("surface/core grids"));
                }
                ModelId::Href => {
                    assert!(reason.contains("limited to explicit `href_sprd_*`"));
                }
                ModelId::Aifs => {
                    assert!(reason.contains("whole-file structured extraction"));
                }
            }
        }
    }
}

#[test]
fn dewpoint_and_700mb_recipe_blockers_are_explicit() {
    for model in [ModelId::Hrrr, ModelId::RrfsA] {
        let dewpoint_850 =
            plot_recipe_fetch_blockers("850mb_dewpoint_height_winds", model).unwrap();
        assert!(dewpoint_850.is_empty());

        let dewpoint_700 =
            plot_recipe_fetch_blockers("700mb_dewpoint_height_winds", model).unwrap();
        assert!(dewpoint_700.is_empty());
    }

    let gfs_dewpoint =
        plot_recipe_fetch_blockers("700mb_dewpoint_height_winds", ModelId::Gfs).unwrap();
    assert_eq!(
        gfs_dewpoint,
        vec![PlotRecipeBlocker {
            field_key: "dewpoint_700mb",
            field_label: "700mb Dewpoint",
            reason: "700mb Dewpoint is not present in the GFS 0.25-degree pgrb2 file currently wired by rustwx-models; keep it blocked until a verified direct field or derived dewpoint path is implemented".to_string(),
        }]
    );
    let err = plot_recipe_fetch_plan("700mb_dewpoint_height_winds", ModelId::Gfs).unwrap_err();
    assert!(matches!(err, ModelError::UnsupportedPlotRecipeModel { .. }));

    let ecmwf_dewpoint =
        plot_recipe_fetch_blockers("700mb_dewpoint_height_winds", ModelId::EcmwfOpenData).unwrap();
    assert_eq!(
        ecmwf_dewpoint,
        vec![PlotRecipeBlocker {
            field_key: "dewpoint_700mb",
            field_label: "700mb Dewpoint",
            reason: "700mb Dewpoint is not present in the ECMWF open-data 'oper' pressure product currently wired by rustwx-models; use RH/TMP or add derived dewpoint support for this model".to_string(),
        }]
    );
}

#[test]
fn absolute_vorticity_recipe_blocker_is_explicit_for_gfs() {
    let blockers =
        plot_recipe_fetch_blockers("850mb_absolute_vorticity_height_winds", ModelId::Gfs).unwrap();
    assert!(blockers.is_empty());
}

#[test]
fn absolute_vorticity_recipe_retains_explicit_primary_blocker() {
    let plan =
        plot_recipe_fetch_plan("700mb_absolute_vorticity_height_winds", ModelId::Gfs).unwrap();
    assert_eq!(
        plan.selectors(),
        vec![
            FieldSelector::isobaric(CanonicalField::AbsoluteVorticity, 700),
            FieldSelector::isobaric(CanonicalField::GeopotentialHeight, 700),
            FieldSelector::isobaric(CanonicalField::UWind, 700),
            FieldSelector::isobaric(CanonicalField::VWind, 700),
        ]
    );
    assert_eq!(
        plan.fetch_policy,
        PlotRecipeFetchPolicy::PreferIndexedSubset
    );
    assert_eq!(plan.fetch_mode, PlotRecipeFetchMode::IndexedSubset);
    assert!(!plan.variable_patterns().is_empty());

    let blockers =
        plot_recipe_fetch_blockers("700mb_absolute_vorticity_height_winds", ModelId::Gfs).unwrap();
    assert!(blockers.is_empty());

    let err = plot_recipe_fetch_plan(
        "500mb_absolute_vorticity_height_winds",
        ModelId::EcmwfOpenData,
    )
    .unwrap_err();
    assert!(matches!(
        err,
        ModelError::UnsupportedPlotRecipeModel {
            recipe: "500mb_absolute_vorticity_height_winds",
            model: ModelId::EcmwfOpenData,
            reason,
        } if reason == "500mb Absolute Vorticity: 500mb Absolute Vorticity is not present in the ECMWF open-data 'oper' pressure product currently wired by rustwx-models"
    ));

    let ecmwf_blockers = plot_recipe_fetch_blockers(
        "500mb_absolute_vorticity_height_winds",
        ModelId::EcmwfOpenData,
    )
    .unwrap();
    assert_eq!(
        ecmwf_blockers,
        vec![PlotRecipeBlocker {
            field_key: "absolute_vorticity_500mb",
            field_label: "500mb Absolute Vorticity",
            reason: "500mb Absolute Vorticity is not present in the ECMWF open-data 'oper' pressure product currently wired by rustwx-models".to_string(),
        }]
    );

    let err = plot_recipe_fetch_plan(
        "300mb_absolute_vorticity_height_winds",
        ModelId::EcmwfOpenData,
    )
    .unwrap_err();
    assert!(matches!(
        err,
        ModelError::UnsupportedPlotRecipeModel {
            recipe: "300mb_absolute_vorticity_height_winds",
            model: ModelId::EcmwfOpenData,
            reason,
        } if reason == "300mb Absolute Vorticity: 300mb Absolute Vorticity is not present in the ECMWF open-data 'oper' pressure product currently wired by rustwx-models"
    ));
}

#[test]
fn latest_available_run_prefers_newest_cycle_over_source_priority() {
    let latest = latest_available_run_with_probe(ModelId::Gfs, None, "20260414", |resolved| {
        resolved.source == SourceId::Aws
            && resolved
                .availability_probe_url()
                .contains("gfs.t18z.pgrb2.0p25.f000")
    })
    .unwrap();

    assert_eq!(latest.cycle.hour_utc, 18);
    assert_eq!(latest.source, SourceId::Aws);
}

#[test]
fn latest_available_run_prefers_source_priority_within_same_cycle() {
    let latest = latest_available_run_with_probe(ModelId::Gfs, None, "20260414", |resolved| {
        resolved
            .availability_probe_url()
            .contains("gfs.t18z.pgrb2.0p25.f000")
    })
    .unwrap();

    assert_eq!(latest.cycle.hour_utc, 18);
    assert_eq!(latest.source, SourceId::Nomads);
}

#[test]
fn ecmwf_summary_matches_current_open_data_cycles_and_horizon() {
    let summary = model_summary(ModelId::EcmwfOpenData);
    assert!(summary.description.contains("50r1"));
    assert_eq!(summary.cycle_hours_utc, &[0, 6, 12, 18]);
    assert_eq!(summary.max_forecast_hour, 360);
}

#[test]
fn aifs_summary_defaults_to_operational_v2_open_data() {
    let summary = model_summary(ModelId::Aifs);
    assert!(summary.description.contains("v2"));
    assert_eq!(summary.sources[0].id, SourceId::Ecmwf);
    assert_eq!(summary.sources[1].id, SourceId::AifsInference);
    assert_eq!(summary.sources[2].id, SourceId::Earth2Archive);
    assert_eq!(summary.runtime_family, ModelRuntimeFamily::Grib2Forecast);
    assert_eq!(summary.ensemble_mode, EnsembleMode::Deterministic);
}

#[test]
fn hiresw_summary_matches_default_conus_arw_cycles_and_horizon() {
    let summary = model_summary(ModelId::Hiresw);
    assert_eq!(summary.default_product, "arw_2p5km/conus");
    assert_eq!(summary.cycle_hours_utc, &[0, 12]);
    assert_eq!(summary.max_forecast_hour, 48);
}

#[test]
fn ncep_regional_summaries_include_aws_archive_sources() {
    for model in [ModelId::Rap, ModelId::Nam] {
        let summary = model_summary(model);
        assert!(
            summary
                .sources
                .iter()
                .any(|source| source.id == SourceId::Aws),
            "{model} should advertise AWS so operational fetches can avoid NOMADS"
        );
    }
}

#[test]
fn ecmwf_supported_forecast_hours_follow_open_data_cadence() {
    let hours_00z = supported_forecast_hours(ModelId::EcmwfOpenData, 0);
    assert!(hours_00z.contains(&0));
    assert!(hours_00z.contains(&144));
    assert!(hours_00z.contains(&150));
    assert!(hours_00z.contains(&360));
    assert!(!hours_00z.contains(&145));
    assert!(!hours_00z.contains(&147));

    let hours_06z = supported_forecast_hours(ModelId::EcmwfOpenData, 6);
    assert!(hours_06z.contains(&0));
    assert!(hours_06z.contains(&144));
    assert!(!hours_06z.contains(&145));
    assert!(!hours_06z.contains(&150));
    assert!(!hours_06z.contains(&360));
}

#[test]
fn gefs_supported_forecast_hours_follow_operational_high_hour_cadence() {
    let hours = supported_forecast_hours(ModelId::Gefs, 12);
    assert!(hours.contains(&0));
    assert!(hours.contains(&240));
    assert!(!hours.contains(&243));
    assert!(hours.contains(&246));
    assert!(!hours.contains(&249));
    assert!(hours.contains(&384));
}

#[test]
fn rap_supported_forecast_hours_are_cycle_aware() {
    let short_cycle = supported_forecast_hours(ModelId::Rap, 0);
    assert!(short_cycle.contains(&21));
    assert!(!short_cycle.contains(&22));
    assert!(!short_cycle.contains(&51));

    let extended_cycle = supported_forecast_hours(ModelId::Rap, 3);
    assert!(extended_cycle.contains(&21));
    assert!(extended_cycle.contains(&22));
    assert!(extended_cycle.contains(&51));
}

#[test]
fn rap_url_builder_rejects_extended_hours_on_short_cycles() {
    let short_cycle = rustwx_core::CycleSpec::new("20260502", 0).unwrap();
    let request = ModelRunRequest::new(ModelId::Rap, short_cycle, 22, "awp130pgrb").unwrap();
    let err = build_grib_url(SourceId::Nomads, &request).unwrap_err();
    assert!(matches!(err, ModelError::UnsupportedForecastHour { .. }));

    let extended_cycle = rustwx_core::CycleSpec::new("20260502", 3).unwrap();
    let request = ModelRunRequest::new(ModelId::Rap, extended_cycle, 51, "awp130pgrb").unwrap();
    assert_eq!(
        build_grib_url(SourceId::Nomads, &request).unwrap(),
        "https://nomads.ncep.noaa.gov/pub/data/nccf/com/rap/prod/rap.20260502/rap.t03z.awp130pgrbf51.grib2"
    );
}

#[test]
fn gefs_urls_use_member_product_and_operational_layout() {
    let request = ModelRunRequest::new(
        ModelId::Gefs,
        rustwx_core::CycleSpec::new("20260502", 0).unwrap(),
        24,
        "pgrb2ap5/gep03",
    )
    .unwrap();

    assert_eq!(
        build_grib_url(SourceId::Aws, &request).unwrap(),
        "https://noaa-gefs-pds.s3.amazonaws.com/gefs.20260502/00/atmos/pgrb2ap5/gep03.t00z.pgrb2a.0p50.f024"
    );
    assert_eq!(
        build_grib_url(SourceId::Nomads, &request).unwrap(),
        "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gens/prod/gefs.20260502/00/atmos/pgrb2ap5/gep03.t00z.pgrb2a.0p50.f024"
    );
}

#[test]
fn easy_ncep_model_urls_use_current_operational_layouts() {
    let cycle = rustwx_core::CycleSpec::new("20260502", 0).unwrap();
    let hrrr_ak = ModelRunRequest::new(ModelId::HrrrAk, cycle.clone(), 24, "sfc").unwrap();
    assert_eq!(
        build_grib_url(SourceId::Nomads, &hrrr_ak).unwrap(),
        "https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod/hrrr.20260502/alaska/hrrr.t00z.wrfsfcf24.ak.grib2"
    );

    let gdas = ModelRunRequest::new(ModelId::Gdas, cycle.clone(), 3, "pgrb2.0p25").unwrap();
    assert_eq!(
        build_grib_url(SourceId::Aws, &gdas).unwrap(),
        "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gdas.20260502/00/atmos/gdas.t00z.pgrb2.0p25.f003"
    );

    let aigfs = ModelRunRequest::new(ModelId::Aigfs, cycle.clone(), 24, "pres").unwrap();
    assert_eq!(
        build_grib_url(SourceId::Nomads, &aigfs).unwrap(),
        "https://nomads.ncep.noaa.gov/pub/data/nccf/com/aigfs/prod/aigfs.20260502/00/model/atmos/grib2/aigfs.t00z.pres.f024.grib2"
    );

    let aigefs = ModelRunRequest::new(ModelId::Aigefs, cycle.clone(), 24, "sfc/avg").unwrap();
    assert_eq!(
        build_grib_url(SourceId::Nomads, &aigefs).unwrap(),
        "https://nomads.ncep.noaa.gov/pub/data/nccf/com/aigefs/prod/aigefs.20260502/00/ensstat/products/atmos/grib2/aigefs.t00z.sfc.avg.f024.grib2"
    );

    let hgefs = ModelRunRequest::new(ModelId::Hgefs, cycle.clone(), 24, "pres/spr").unwrap();
    assert_eq!(
        build_grib_url(SourceId::Nomads, &hgefs).unwrap(),
        "https://nomads.ncep.noaa.gov/pub/data/nccf/com/hgefs/prod/hgefs.20260502/00/ensstat/products/atmos/grib2/hgefs.t00z.pres.spr.f024.grib2"
    );

    let rrfs_firewx =
        ModelRunRequest::new(ModelId::RrfsFireWx, cycle.clone(), 24, "2dfld-firewx").unwrap();
    assert_eq!(
        build_grib_url(SourceId::Aws, &rrfs_firewx).unwrap(),
        "https://noaa-rrfs-pds.s3.amazonaws.com/rrfs_public/firewx.20260502/00/rrfs.t00z.2dfld.1p5km.f024.firewx_lcc.grib2"
    );

    let rap = ModelRunRequest::new(ModelId::Rap, cycle.clone(), 21, "awp130pgrb").unwrap();
    assert_eq!(
        build_grib_url(SourceId::Aws, &rap).unwrap(),
        "https://noaa-rap-pds.s3.amazonaws.com/rap.20260502/rap.t00z.awp130pgrbf21.grib2"
    );
    assert_eq!(
        build_grib_url(SourceId::Nomads, &rap).unwrap(),
        "https://nomads.ncep.noaa.gov/pub/data/nccf/com/rap/prod/rap.20260502/rap.t00z.awp130pgrbf21.grib2"
    );

    let nam = ModelRunRequest::new(ModelId::Nam, cycle.clone(), 24, "awip12").unwrap();
    assert_eq!(
        build_grib_url(SourceId::Aws, &nam).unwrap(),
        "https://noaa-nam-pds.s3.amazonaws.com/nam.20260502/nam.t00z.awip1224.tm00.grib2"
    );
    assert_eq!(
        build_grib_url(SourceId::Nomads, &nam).unwrap(),
        "https://nomads.ncep.noaa.gov/pub/data/nccf/com/nam/prod/nam.20260502/nam.t00z.awip1224.tm00.grib2"
    );

    let nam_3d = ModelRunRequest::new(ModelId::Nam, cycle.clone(), 24, "awip3d").unwrap();
    assert_eq!(
        build_grib_url(SourceId::Aws, &nam_3d).unwrap(),
        "https://noaa-nam-pds.s3.amazonaws.com/nam.20260502/nam.t00z.awip3d24.tm00.grib2"
    );

    let hiresw =
        ModelRunRequest::new(ModelId::Hiresw, cycle.clone(), 24, "arw_2p5km/conus").unwrap();
    assert_eq!(
        build_grib_url(SourceId::Nomads, &hiresw).unwrap(),
        "https://nomads.ncep.noaa.gov/pub/data/nccf/com/hiresw/prod/hiresw.20260502/hiresw.t00z.arw_2p5km.f24.conus.grib2"
    );

    let hiresw_arw_mem2 =
        ModelRunRequest::new(ModelId::Hiresw, cycle.clone(), 24, "arw_mem2/conus").unwrap();
    assert_eq!(
        build_grib_url(SourceId::Nomads, &hiresw_arw_mem2).unwrap(),
        "https://nomads.ncep.noaa.gov/pub/data/nccf/com/hiresw/prod/hiresw.20260502/hiresw.t00z.arw_5km.f24.conusmem2.grib2"
    );

    let hiresw_arw_5km_mem2 =
        ModelRunRequest::new(ModelId::Hiresw, cycle.clone(), 24, "arw_5km/conusmem2").unwrap();
    assert_eq!(
        build_grib_url(SourceId::Nomads, &hiresw_arw_5km_mem2).unwrap(),
        "https://nomads.ncep.noaa.gov/pub/data/nccf/com/hiresw/prod/hiresw.20260502/hiresw.t00z.arw_5km.f24.conusmem2.grib2"
    );

    let href =
        ModelRunRequest::new(ModelId::Href, cycle.clone(), 24, "ensprod/conus/sprd").unwrap();
    assert_eq!(
        build_grib_url(SourceId::Nomads, &href).unwrap(),
        "https://nomads.ncep.noaa.gov/pub/data/nccf/com/href/prod/href.20260502/ensprod/href.t00z.conus.sprd.f24.grib2"
    );

    let href_mean =
        ModelRunRequest::new(ModelId::Href, cycle.clone(), 24, "ensprod/conus/mean").unwrap();
    assert_eq!(
        build_grib_url(SourceId::Nomads, &href_mean).unwrap(),
        "https://nomads.ncep.noaa.gov/pub/data/nccf/com/href/prod/href.20260502/ensprod/href.t00z.conus.mean.f24.grib2"
    );

    let refs = ModelRunRequest::new(ModelId::Refs, cycle.clone(), 24, "prob-conus").unwrap();
    assert_eq!(
        build_grib_url(SourceId::Aws, &refs).unwrap(),
        "https://noaa-rrfs-pds.s3.amazonaws.com/rrfs_public/refs.20260502/00/enspost/refs.t00z.prob.f24.conus.grib2"
    );

    let refs_hi = ModelRunRequest::new(ModelId::Refs, cycle.clone(), 24, "mean-hi").unwrap();
    assert_eq!(
        build_grib_url(SourceId::Aws, &refs_hi).unwrap(),
        "https://noaa-rrfs-pds.s3.amazonaws.com/rrfs_public/refs.20260502/00/enspost/refs.t00z.mean.f24.hi.grib2"
    );

    let refs_pr =
        ModelRunRequest::new(ModelId::Refs, cycle.clone(), 24, "pmmn-puerto-rico").unwrap();
    assert_eq!(
        build_grib_url(SourceId::Aws, &refs_pr).unwrap(),
        "https://noaa-rrfs-pds.s3.amazonaws.com/rrfs_public/refs.20260502/00/enspost/refs.t00z.pmmn.f24.pr.grib2"
    );

    let refs_ffri_ak = ModelRunRequest::new(ModelId::Refs, cycle.clone(), 24, "ffri-ak").unwrap();
    assert!(matches!(
        build_grib_url(SourceId::Aws, &refs_ffri_ak),
        Err(ModelError::UnsupportedProduct { .. })
    ));

    let nbm = ModelRunRequest::new(ModelId::Nbm, cycle.clone(), 24, "core/co").unwrap();
    assert_eq!(
        build_grib_url(SourceId::Aws, &nbm).unwrap(),
        "https://noaa-nbm-grib2-pds.s3.amazonaws.com/blend.20260502/00/core/blend.t00z.core.f024.co.grib2"
    );

    let rtma = ModelRunRequest::new(ModelId::Rtma, cycle.clone(), 0, "2dvaranl_ndfd").unwrap();
    assert_eq!(
        build_grib_url(SourceId::Nomads, &rtma).unwrap(),
        "https://nomads.ncep.noaa.gov/pub/data/nccf/com/rtma/prod/rtma2p5.20260502/rtma2p5.t00z.2dvaranl_ndfd.grb2_wexp"
    );

    let sref_cycle = rustwx_core::CycleSpec::new("20260502", 3).unwrap();
    let sref = ModelRunRequest::new(ModelId::Sref, sref_cycle, 24, "arw/ctl/pgrb132").unwrap();
    assert_eq!(
        build_grib_url(SourceId::Nomads, &sref).unwrap(),
        "https://nomads.ncep.noaa.gov/pub/data/nccf/com/sref/prod/sref.20260502/03/pgrb/sref_arw.t03z.pgrb132.ctl.f24.grib2"
    );
}

#[test]
fn aifs_supports_local_long_runs_and_ecmwf_open_data_urls() {
    let local = ModelRunRequest::new(
        ModelId::Aifs,
        rustwx_core::CycleSpec::new("20260502", 0).unwrap(),
        43_848,
        "oper",
    )
    .unwrap();
    assert_eq!(
        build_grib_url(SourceId::Earth2Archive, &local).unwrap(),
        "earth2-archive://aifs/20260502T00Z/lead43848.nc"
    );
    assert_eq!(
        build_grib_url(SourceId::AifsInference, &local).unwrap(),
        "aifs-inference://aifs/20260502T00Z/lead43848.nc"
    );

    let open_data = ModelRunRequest::new(
        ModelId::Aifs,
        rustwx_core::CycleSpec::new("20260502", 0).unwrap(),
        24,
        "oper",
    )
    .unwrap();
    assert_eq!(
        build_grib_url(SourceId::Ecmwf, &open_data).unwrap(),
        "https://data.ecmwf.int/forecasts/20260502/00z/aifs-single/0p25/oper/20260502000000-24h-oper-fc.grib2"
    );

    let wave = ModelRunRequest::new(
        ModelId::Aifs,
        rustwx_core::CycleSpec::new("20260502", 0).unwrap(),
        24,
        "wave",
    )
    .unwrap();
    assert_eq!(
        build_grib_url(SourceId::Ecmwf, &wave).unwrap(),
        "https://data.ecmwf.int/forecasts/20260502/00z/aifs-single/0p25/wave/20260502000000-24h-wave-fc.grib2"
    );

    assert!(matches!(
        build_grib_url(SourceId::Ecmwf, &local),
        Err(ModelError::UnsupportedForecastHour {
            model: ModelId::Aifs,
            forecast_hour: 43_848,
            ..
        })
    ));
}

#[test]
fn latest_available_run_considers_ecmwf_six_hour_cycles() {
    let latest =
        latest_available_run_with_probe(ModelId::EcmwfOpenData, None, "20260414", |resolved| {
            resolved
                .availability_probe_url()
                .contains("/20260414/18z/ifs/0p25/oper/")
        })
        .unwrap();

    assert_eq!(latest.cycle.hour_utc, 18);
    assert_eq!(latest.source, SourceId::Ecmwf);
}

#[test]
fn latest_available_run_at_forecast_hour_prefers_ecmwf_long_cycle_when_needed() {
    let latest = latest_available_run_for_products_with_probe_at_forecast_hour(
        ModelId::EcmwfOpenData,
        None,
        "20260414",
        &["oper"],
        150,
        |resolved| {
            let url = resolved.availability_probe_url();
            url.contains("/20260414/12z/ifs/0p25/oper/") && url.contains("150h-oper-fc.grib2")
        },
    )
    .unwrap();

    assert_eq!(latest.cycle.date_yyyymmdd, "20260414");
    assert_eq!(latest.cycle.hour_utc, 12);
}

#[test]
fn latest_available_run_rolls_back_to_previous_day_when_today_is_unpublished() {
    // Simulate a UTC-rollover / publication window where no cycle of the
    // requested date has been published yet, but yesterday's 18z is still
    // serving. The probe returns true only for yesterday's 18z URL.
    let latest = latest_available_run_with_probe(ModelId::Gfs, None, "20260415", |resolved| {
        resolved
            .availability_probe_url()
            .contains("gfs.20260414/18/atmos/gfs.t18z.pgrb2.0p25.f000")
    })
    .unwrap();

    assert_eq!(latest.cycle.date_yyyymmdd, "20260414");
    assert_eq!(latest.cycle.hour_utc, 18);
}

#[test]
fn latest_available_run_prefers_today_even_with_rollback_enabled() {
    // If both today and yesterday are available, today wins — the
    // rollback must not demote a published current-day run.
    let latest =
        latest_available_run_with_probe(ModelId::EcmwfOpenData, None, "20260414", |resolved| {
            let url = resolved.availability_probe_url();
            url.contains("/20260414/06z/") || url.contains("/20260413/18z/")
        })
        .unwrap();

    assert_eq!(latest.cycle.date_yyyymmdd, "20260414");
    assert_eq!(latest.cycle.hour_utc, 6);
}

#[test]
fn latest_available_run_for_products_requires_all_products_on_same_cycle() {
    let latest = latest_available_run_for_products_with_probe(
        ModelId::RrfsA,
        Some(SourceId::Aws),
        "20260417",
        &["prs-conus", "nat-na", "prs-na"],
        |resolved| {
            let url = resolved.availability_probe_url();
            url.contains("20260417/22/")
                || (url.contains("20260417/23/") && url.contains("prslev.3km.f000.conus"))
        },
    )
    .unwrap();

    assert_eq!(latest.cycle.date_yyyymmdd, "20260417");
    assert_eq!(latest.cycle.hour_utc, 22);
    assert_eq!(latest.source, SourceId::Aws);
}

#[test]
fn previous_day_yyyymmdd_handles_month_and_year_boundaries() {
    assert_eq!(
        previous_day_yyyymmdd("20260415").as_deref(),
        Some("20260414")
    );
    assert_eq!(
        previous_day_yyyymmdd("20260401").as_deref(),
        Some("20260331")
    );
    assert_eq!(
        previous_day_yyyymmdd("20260101").as_deref(),
        Some("20251231")
    );
    // Leap-year awareness: 2028 is a leap year, so 2028-03-01 rolls back
    // to 2028-02-29, while 2027-03-01 rolls back to 2027-02-28.
    assert_eq!(
        previous_day_yyyymmdd("20280301").as_deref(),
        Some("20280229")
    );
    assert_eq!(
        previous_day_yyyymmdd("20270301").as_deref(),
        Some("20270228")
    );
    assert_eq!(previous_day_yyyymmdd("notadate"), None);
}

#[test]
fn nomads_uses_range_probe_policy() {
    assert!(should_use_range_probe(SourceId::Nomads));
    assert!(!should_use_range_probe(SourceId::Aws));
}

#[test]
fn hrrr_native_reflectivity_recipe_produces_nat_fetch_plan() {
    let plan = plot_recipe_fetch_plan("composite_reflectivity", ModelId::Hrrr).unwrap();
    assert_eq!(plan.product, "nat");
    assert_eq!(
        plan.fetch_policy,
        PlotRecipeFetchPolicy::PreferIndexedSubset
    );
    assert_eq!(plan.fetch_mode, PlotRecipeFetchMode::IndexedSubset);
    assert_eq!(
        plan.selectors(),
        vec![FieldSelector::entire_atmosphere(
            CanonicalField::CompositeReflectivity
        )]
    );
    assert!(!plan.variable_patterns().is_empty());
}

#[test]
fn native_reflectivity_uh_recipe_tracks_supported_models() {
    let hrrr_plan = plot_recipe_fetch_plan("composite_reflectivity_uh", ModelId::Hrrr).unwrap();
    assert_eq!(hrrr_plan.product, "nat");
    assert_eq!(
        hrrr_plan.selectors(),
        vec![
            FieldSelector::entire_atmosphere(CanonicalField::CompositeReflectivity),
            FieldSelector::height_layer_agl(CanonicalField::UpdraftHelicity, 2000, 5000),
        ]
    );

    let rrfs_plan = plot_recipe_fetch_plan("composite_reflectivity_uh", ModelId::RrfsA).unwrap();
    assert_eq!(rrfs_plan.product, "prs-conus");
    assert_eq!(
        rrfs_plan.fetch_policy,
        PlotRecipeFetchPolicy::PreferIndexedSubset
    );
    assert_eq!(rrfs_plan.fetch_mode, PlotRecipeFetchMode::IndexedSubset);
    assert_eq!(
        rrfs_plan.selectors(),
        vec![
            FieldSelector::entire_atmosphere(CanonicalField::CompositeReflectivity),
            FieldSelector::height_layer_agl(CanonicalField::UpdraftHelicity, 2000, 5000),
        ]
    );
    assert!(!rrfs_plan.variable_patterns().is_empty());

    let rrfs_public_plan =
        plot_recipe_fetch_plan("composite_reflectivity_uh", ModelId::RrfsPublic).unwrap();
    assert_eq!(rrfs_public_plan.product, "2dfld-conus");
    assert_eq!(
        rrfs_public_plan.fetch_policy,
        PlotRecipeFetchPolicy::PreferIndexedSubset
    );
    assert_eq!(
        rrfs_public_plan.fetch_mode,
        PlotRecipeFetchMode::IndexedSubset
    );
    assert_eq!(rrfs_public_plan.selectors(), rrfs_plan.selectors());
    assert!(!rrfs_public_plan.variable_patterns().is_empty());

    let rrfs_firewx_plan =
        plot_recipe_fetch_plan("composite_reflectivity_uh", ModelId::RrfsFireWx).unwrap();
    assert_eq!(rrfs_firewx_plan.product, "2dfld-firewx");
    assert_eq!(
        rrfs_firewx_plan.fetch_policy,
        PlotRecipeFetchPolicy::PreferIndexedSubset
    );
    assert_eq!(
        rrfs_firewx_plan.fetch_mode,
        PlotRecipeFetchMode::IndexedSubset
    );
    assert_eq!(rrfs_firewx_plan.selectors(), rrfs_plan.selectors());
    assert!(!rrfs_firewx_plan.variable_patterns().is_empty());
}

#[test]
fn rrfs_public_variants_direct_fetches_use_matching_public_products() {
    let rrfs_a_surface =
        plot_recipe_fetch_plan("2m_temperature_10m_winds", ModelId::RrfsA).unwrap();
    assert_eq!(rrfs_a_surface.product, "nat-na");
    assert_eq!(
        rrfs_a_surface.fetch_policy,
        PlotRecipeFetchPolicy::PreferIndexedSubset
    );
    assert_eq!(
        rrfs_a_surface.fetch_mode,
        PlotRecipeFetchMode::IndexedSubset
    );

    let rrfs_a_pressure = plot_recipe_fetch_plan("500mb_height_winds", ModelId::RrfsA).unwrap();
    assert_eq!(rrfs_a_pressure.product, "prs-conus");
    assert_eq!(
        rrfs_a_pressure.fetch_policy,
        PlotRecipeFetchPolicy::PreferIndexedSubset
    );
    assert_eq!(
        rrfs_a_pressure.fetch_mode,
        PlotRecipeFetchMode::IndexedSubset
    );

    let cases = [
        (ModelId::RrfsPublic, "2dfld-conus", "prs-conus"),
        (ModelId::RrfsFireWx, "2dfld-firewx", "prs-firewx"),
    ];
    for (model, surface_product, pressure_product) in cases {
        let surface = plot_recipe_fetch_plan("2m_temperature_10m_winds", model).unwrap();
        assert_eq!(surface.product, surface_product);
        assert_eq!(surface.fetch_mode, PlotRecipeFetchMode::IndexedSubset);

        let pressure = plot_recipe_fetch_plan("500mb_height_winds", model).unwrap();
        assert_eq!(pressure.product, pressure_product);
        assert_eq!(pressure.fetch_mode, PlotRecipeFetchMode::IndexedSubset);
    }
}

#[test]
fn simulated_ir_recipe_is_supported_for_hrrr_native_fetch() {
    let plan = plot_recipe_fetch_plan("simulated_ir_satellite", ModelId::Hrrr).unwrap();
    assert_eq!(plan.product, "nat");
    assert_eq!(
        plan.fetch_policy,
        PlotRecipeFetchPolicy::PreferIndexedSubset
    );
    assert_eq!(plan.fetch_mode, PlotRecipeFetchMode::IndexedSubset);
    assert_eq!(
        plan.selectors(),
        vec![FieldSelector::nominal_top(
            CanonicalField::SimulatedInfraredBrightnessTemperature
        )]
    );
    assert!(!plan.variable_patterns().is_empty());
    assert!(
        plot_recipe_fetch_blockers("simulated_ir_satellite", ModelId::Hrrr)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn smoke_recipes_use_hrrr_native_fetch_plan() {
    let smoke = plot_recipe_fetch_plan("smoke_pm25_native", ModelId::Hrrr).unwrap();
    assert_eq!(smoke.product, "nat");
    assert_eq!(
        smoke.fetch_policy,
        PlotRecipeFetchPolicy::PreferIndexedSubset
    );
    assert_eq!(smoke.fetch_mode, PlotRecipeFetchMode::IndexedSubset);
    assert_eq!(
        smoke.selectors(),
        vec![FieldSelector::height_agl(
            CanonicalField::SmokeMassDensity,
            8
        )]
    );
    assert!(!smoke.variable_patterns().is_empty());

    let column = plot_recipe_fetch_plan("smoke_column", ModelId::Hrrr).unwrap();
    assert_eq!(column.product, "nat");
    assert_eq!(
        column.fetch_policy,
        PlotRecipeFetchPolicy::PreferIndexedSubset
    );
    assert_eq!(column.fetch_mode, PlotRecipeFetchMode::IndexedSubset);
    assert_eq!(
        column.selectors(),
        vec![FieldSelector::entire_atmosphere(
            CanonicalField::ColumnIntegratedSmoke
        )]
    );
    assert!(!column.variable_patterns().is_empty());
}

#[test]
fn supported_recipe_has_no_fetch_blockers() {
    let blockers =
        plot_recipe_fetch_blockers("850mb_temperature_height_winds", ModelId::EcmwfOpenData)
            .unwrap();
    assert!(blockers.is_empty());
}

#[test]
fn supported_native_recipe_has_no_fetch_blockers() {
    assert!(
        plot_recipe_fetch_blockers("composite_reflectivity", ModelId::Hrrr)
            .unwrap()
            .is_empty()
    );
    assert!(
        plot_recipe_fetch_blockers("composite_reflectivity_uh", ModelId::RrfsA)
            .unwrap()
            .is_empty()
    );
    assert!(
        plot_recipe_fetch_blockers("composite_reflectivity_uh", ModelId::RrfsPublic)
            .unwrap()
            .is_empty()
    );
    assert!(
        plot_recipe_fetch_blockers("uh_2to5km", ModelId::Hrrr)
            .unwrap()
            .is_empty()
    );
    assert!(
        plot_recipe_fetch_blockers("smoke_pm25_native", ModelId::Hrrr)
            .unwrap()
            .is_empty()
    );
    assert!(
        plot_recipe_fetch_blockers("smoke_column", ModelId::Hrrr)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn global_models_get_explicit_native_recipe_blockers() {
    let blockers = plot_recipe_fetch_blockers("composite_reflectivity", ModelId::Gfs).unwrap();
    assert_eq!(
        blockers,
        vec![PlotRecipeBlocker {
            field_key: "composite_reflectivity",
            field_label: "Composite Reflectivity",
            reason: "Composite Reflectivity is not wired for model 'gfs'; rustwx-models only has native convective product fetch planning for HRRR/RRFS-A right now".to_string(),
        }]
    );

    let reflectivity_1km = plot_recipe_fetch_blockers("1km_reflectivity", ModelId::Gfs).unwrap();
    assert_eq!(
        reflectivity_1km,
        vec![PlotRecipeBlocker {
            field_key: "radar_reflectivity_1km_agl",
            field_label: "1km AGL Reflectivity",
            reason: "1km AGL Reflectivity is not wired for model 'gfs'; rustwx-models only has native convective product fetch planning for HRRR/RRFS-A right now".to_string(),
        }]
    );

    let uh = plot_recipe_fetch_blockers("uh_2to5km", ModelId::Gfs).unwrap();
    assert_eq!(
        uh,
        vec![PlotRecipeBlocker {
            field_key: "updraft_helicity",
            field_label: "Updraft Helicity",
            reason: "Updraft Helicity is not wired for model 'gfs'; rustwx-models only has native convective product fetch planning for HRRR/RRFS-A right now".to_string(),
        }]
    );
}

#[test]
fn selector_support_policy_lives_in_models() {
    assert!(selector_supported_for_model(
        FieldSelector::isobaric(CanonicalField::Temperature, 500),
        ModelId::Gfs,
    ));
    assert!(selector_supported_for_model(
        FieldSelector::isobaric(CanonicalField::Temperature, 200),
        ModelId::Gfs,
    ));
    assert!(selector_supported_for_model(
        FieldSelector::isobaric(CanonicalField::Temperature, 250),
        ModelId::WrfGdex,
    ));
    assert!(selector_supported_for_model(
        FieldSelector::isobaric(CanonicalField::Dewpoint, 700),
        ModelId::RrfsA,
    ));
    assert!(!selector_supported_for_model(
        FieldSelector::isobaric(CanonicalField::RelativeVorticity, 500),
        ModelId::Gfs,
    ));
    assert!(selector_supported_for_model(
        FieldSelector::surface(CanonicalField::LandSeaMask),
        ModelId::EcmwfOpenData,
    ));
    assert!(!selector_supported_for_model(
        FieldSelector::surface(CanonicalField::LandSeaMask),
        ModelId::Hrrr,
    ));
    assert!(selector_supported_for_model(
        FieldSelector::height_agl(CanonicalField::Temperature, 2),
        ModelId::Hrrr,
    ));
    assert!(selector_supported_for_model(
        FieldSelector::height_agl(CanonicalField::RadarReflectivity, 1000),
        ModelId::RrfsA,
    ));
    assert!(selector_supported_for_model(
        FieldSelector::height_agl(CanonicalField::RadarReflectivity, 1000),
        ModelId::WrfGdex,
    ));
    assert!(selector_supported_for_model(
        FieldSelector::mean_sea_level(CanonicalField::PressureReducedToMeanSeaLevel),
        ModelId::Hrrr,
    ));
    assert!(selector_supported_for_model(
        FieldSelector::surface(CanonicalField::Visibility),
        ModelId::Gfs,
    ));
    assert!(selector_supported_for_model(
        FieldSelector::nominal_top(CanonicalField::SimulatedInfraredBrightnessTemperature),
        ModelId::Hrrr,
    ));
    assert!(!selector_supported_for_model(
        FieldSelector::nominal_top(CanonicalField::SimulatedInfraredBrightnessTemperature),
        ModelId::Gfs,
    ));
    assert!(selector_supported_for_model(
        FieldSelector::height_agl(CanonicalField::SmokeMassDensity, 8),
        ModelId::Hrrr,
    ));
    assert!(selector_supported_for_model(
        FieldSelector::entire_atmosphere(CanonicalField::ColumnIntegratedSmoke),
        ModelId::Hrrr,
    ));
    assert!(selector_supported_for_model(
        FieldSelector::hybrid_level(CanonicalField::SmokeMassDensity, 50),
        ModelId::Hrrr,
    ));
    assert!(selector_supported_for_model(
        FieldSelector::hybrid_level(CanonicalField::Pressure, 1),
        ModelId::Hrrr,
    ));
    assert!(!selector_supported_for_model(
        FieldSelector::height_agl(CanonicalField::SmokeMassDensity, 8),
        ModelId::RrfsA,
    ));
    assert!(!selector_supported_for_model(
        FieldSelector::hybrid_level(CanonicalField::SmokeMassDensity, 51),
        ModelId::Hrrr,
    ));
}

#[test]
fn direct_surface_recipe_uses_surface_fetch_plan_when_supported() {
    let blockers = plot_recipe_fetch_blockers("2m_temperature", ModelId::Hrrr).unwrap();
    assert!(blockers.is_empty());

    let plan = plot_recipe_fetch_plan("2m_temperature_10m_winds", ModelId::Hrrr).unwrap();
    assert_eq!(plan.product, "sfc");
    assert_eq!(
        plan.fetch_policy,
        PlotRecipeFetchPolicy::PreferIndexedSubset
    );
    assert_eq!(plan.fetch_mode, PlotRecipeFetchMode::IndexedSubset);
    assert!(!plan.variable_patterns().is_empty());
    assert_eq!(
        plan.selectors(),
        vec![
            FieldSelector::height_agl(CanonicalField::Temperature, 2),
            FieldSelector::mean_sea_level(CanonicalField::PressureReducedToMeanSeaLevel),
            FieldSelector::height_agl(CanonicalField::UWind, 10),
            FieldSelector::height_agl(CanonicalField::VWind, 10),
        ]
    );
}

#[test]
fn hrrr_pressure_recipe_prefers_indexed_subset_fetches() {
    let plan = plot_recipe_fetch_plan("500mb_temperature_height_winds", ModelId::Hrrr).unwrap();
    assert_eq!(plan.product, "prs");
    assert_eq!(
        plan.fetch_policy,
        PlotRecipeFetchPolicy::PreferIndexedSubset
    );
    assert_eq!(plan.fetch_mode, PlotRecipeFetchMode::IndexedSubset);
    assert!(!plan.variable_patterns().is_empty());
}

#[test]
fn wrf_gdex_pressure_recipe_prefers_whole_file_fetches() {
    let plan = plot_recipe_fetch_plan("500mb_temperature_height_winds", ModelId::WrfGdex).unwrap();
    assert_eq!(plan.product, WRF_GDEX_DEFAULT_PRESSURE_PRODUCT);
    assert_eq!(plan.fetch_policy, PlotRecipeFetchPolicy::WholeFile);
    assert_eq!(
        plan.fetch_mode,
        PlotRecipeFetchMode::WholeFileStructuredExtract
    );
    assert!(plan.variable_patterns().is_empty());
}

#[test]
fn wrf_gdex_native_reflectivity_recipe_uses_whole_file_structured_extract() {
    let plan = plot_recipe_fetch_plan("composite_reflectivity", ModelId::WrfGdex).unwrap();
    assert_eq!(plan.product, WRF_GDEX_DEFAULT_SURFACE_PRODUCT);
    assert_eq!(plan.fetch_policy, PlotRecipeFetchPolicy::WholeFile);
    assert_eq!(
        plan.fetch_mode,
        PlotRecipeFetchMode::WholeFileStructuredExtract
    );
    assert_eq!(
        plan.selectors(),
        vec![FieldSelector::entire_atmosphere(
            CanonicalField::CompositeReflectivity
        )]
    );
}

#[test]
fn wrf_gdex_native_reflectivity_recipe_uses_surface_fetch_plan() {
    for slug in ["1km_reflectivity"] {
        let plan = plot_recipe_fetch_plan(slug, ModelId::WrfGdex).unwrap();
        assert_eq!(plan.product, WRF_GDEX_DEFAULT_SURFACE_PRODUCT, "{slug}");
        assert_eq!(
            plan.fetch_policy,
            PlotRecipeFetchPolicy::WholeFile,
            "{slug}"
        );
        assert_eq!(
            plan.fetch_mode,
            PlotRecipeFetchMode::WholeFileStructuredExtract,
            "{slug}"
        );
    }
}

#[test]
fn wrf_gdex_uh_recipes_use_3d_fetch_plan() {
    for slug in ["uh_2to5km", "composite_reflectivity_uh"] {
        let plan = plot_recipe_fetch_plan(slug, ModelId::WrfGdex).unwrap();
        assert_eq!(plan.product, WRF_GDEX_DEFAULT_PRESSURE_PRODUCT, "{slug}");
        assert_eq!(
            plan.fetch_policy,
            PlotRecipeFetchPolicy::WholeFile,
            "{slug}"
        );
        assert_eq!(
            plan.fetch_mode,
            PlotRecipeFetchMode::WholeFileStructuredExtract,
            "{slug}"
        );
    }
}

#[test]
fn wrf_gdex_wind_gust_recipe_is_unblocked_and_uses_surface_fetch_plan() {
    let blockers = plot_recipe_fetch_blockers("10m_wind_gusts", ModelId::WrfGdex).unwrap();
    assert!(blockers.is_empty());

    let recipe = plot_recipe("10m_wind_gusts").expect("gust recipe should exist");
    assert!(recipe.filled.idx_patterns().contains(&"WSPD10MAX"));

    let plan = plot_recipe_fetch_plan("10m_wind_gusts", ModelId::WrfGdex).unwrap();
    assert_eq!(plan.product, WRF_GDEX_DEFAULT_SURFACE_PRODUCT);
    assert_eq!(plan.fetch_policy, PlotRecipeFetchPolicy::WholeFile);
    assert_eq!(
        plan.selectors(),
        vec![FieldSelector::height_agl(CanonicalField::WindGust, 10)]
    );
}

#[test]
fn hrrr_indexed_subset_fetch_plans_cover_pressure_surface_and_native_lanes() {
    let pressure = plot_recipe_fetch_plan("500mb_temperature_height_winds", ModelId::Hrrr)
        .expect("pressure recipe should plan");
    let surface = plot_recipe_fetch_plan("2m_temperature_10m_winds", ModelId::Hrrr)
        .expect("surface recipe should plan");
    let native = plot_recipe_fetch_plan("composite_reflectivity_uh", ModelId::Hrrr)
        .expect("native recipe should plan");

    for plan in [pressure, surface, native] {
        assert_eq!(
            plan.fetch_policy,
            PlotRecipeFetchPolicy::PreferIndexedSubset
        );
        assert_eq!(plan.fetch_mode, PlotRecipeFetchMode::IndexedSubset);
        assert!(
            !plan.variable_patterns().is_empty(),
            "indexed HRRR plans should carry idx variable patterns"
        );
    }
}

#[test]
fn hrrr_direct_composite_layout_recipes_expand_to_selector_backed_components() {
    let cloud_levels = plot_recipe_fetch_plan("cloud_cover_levels", ModelId::Hrrr).unwrap();
    assert_eq!(cloud_levels.product, "sfc");
    assert_eq!(
        cloud_levels.selectors(),
        vec![
            FieldSelector::entire_atmosphere(CanonicalField::LowCloudCover),
            FieldSelector::entire_atmosphere(CanonicalField::MiddleCloudCover),
            FieldSelector::entire_atmosphere(CanonicalField::HighCloudCover),
        ]
    );

    let precipitation_type = plot_recipe_fetch_plan("precipitation_type", ModelId::Hrrr).unwrap();
    assert_eq!(precipitation_type.product, "sfc");
    assert_eq!(
        precipitation_type.selectors(),
        vec![
            FieldSelector::surface(CanonicalField::CategoricalRain),
            FieldSelector::surface(CanonicalField::CategoricalFreezingRain),
            FieldSelector::surface(CanonicalField::CategoricalIcePellets),
            FieldSelector::surface(CanonicalField::CategoricalSnow),
        ]
    );
}

#[test]
fn hrrr_blockers_point_non_native_surface_products_to_honest_lanes() {
    let theta_e = plot_recipe_fetch_blockers("2m_theta_e_10m_winds", ModelId::Hrrr).unwrap();
    assert!(theta_e.iter().any(|blocker| {
        blocker.reason.contains("theta_e_2m_10m_winds")
            && blocker.reason.contains("derived product")
    }));

    let heat_index = plot_recipe_fetch_blockers("2m_heat_index", ModelId::Hrrr).unwrap();
    assert!(heat_index.iter().any(|blocker| {
        blocker.reason.contains("heat_index_2m") && blocker.reason.contains("derived product")
    }));

    let wind_chill = plot_recipe_fetch_blockers("2m_wind_chill", ModelId::Hrrr).unwrap();
    assert!(wind_chill.iter().any(|blocker| {
        blocker.reason.contains("wind_chill_2m") && blocker.reason.contains("derived product")
    }));

    let qpf = plot_recipe_fetch_blockers("1h_qpf", ModelId::Hrrr).unwrap();
    assert!(qpf.iter().any(|blocker| {
        blocker.reason.contains("qpf_1h") && blocker.reason.contains("windowed lane")
    }));
}

#[test]
fn generic_direct_composite_layouts_follow_selector_support() {
    let cloud_levels = plot_recipe_fetch_blockers("cloud_cover_levels", ModelId::Gfs).unwrap();
    assert!(cloud_levels.is_empty());

    let precipitation_type =
        plot_recipe_fetch_blockers("precipitation_type", ModelId::EcmwfOpenData).unwrap();
    assert!(precipitation_type.iter().any(|blocker| {
        blocker.reason.contains("selector") || blocker.reason.contains("not yet supported")
    }));
}

#[test]
fn nbm_probability_of_precipitation_recipe_uses_core_surface_plan() {
    let plan = plot_recipe_fetch_plan("probability_of_precipitation", ModelId::Nbm).unwrap();
    assert_eq!(plan.product, "core/co");
    assert_eq!(
        plan.fetch_policy,
        PlotRecipeFetchPolicy::PreferIndexedSubset
    );
    assert_eq!(
        plan.selectors(),
        vec![FieldSelector::surface(
            CanonicalField::ProbabilityOfPrecipitation
        )]
    );
    assert!(plan.variable_patterns().contains(&"APCP:surface"));
    assert!(
        plot_recipe_fetch_blockers("probability_of_precipitation", ModelId::Nbm)
            .unwrap()
            .is_empty()
    );

    let gfs_blockers =
        plot_recipe_fetch_blockers("probability_of_precipitation", ModelId::Gfs).unwrap();
    assert!(
        gfs_blockers
            .iter()
            .any(|blocker| blocker.reason.contains("only verified for NBM"))
    );
}

#[test]
fn nbm_qmd_stat_recipes_use_qmd_product_and_exact_selectors() {
    let cases = [
        (
            "nbm_qmd_2m_temperature_mean",
            FieldSelector::height_agl(CanonicalField::Temperature, 2).with_ensemble_mean(),
        ),
        (
            "nbm_qmd_2m_temperature_stddev",
            FieldSelector::height_agl(CanonicalField::Temperature, 2)
                .with_ensemble_standard_deviation(),
        ),
        (
            "nbm_qmd_2m_temperature_p05",
            FieldSelector::height_agl(CanonicalField::Temperature, 2).with_percentile(5),
        ),
        (
            "nbm_qmd_2m_temperature_p10",
            FieldSelector::height_agl(CanonicalField::Temperature, 2).with_percentile(10),
        ),
        (
            "nbm_qmd_2m_temperature_p25",
            FieldSelector::height_agl(CanonicalField::Temperature, 2).with_percentile(25),
        ),
        (
            "nbm_qmd_2m_temperature_p50",
            FieldSelector::height_agl(CanonicalField::Temperature, 2).with_percentile(50),
        ),
        (
            "nbm_qmd_2m_temperature_p75",
            FieldSelector::height_agl(CanonicalField::Temperature, 2).with_percentile(75),
        ),
        (
            "nbm_qmd_2m_temperature_p90",
            FieldSelector::height_agl(CanonicalField::Temperature, 2).with_percentile(90),
        ),
        (
            "nbm_qmd_2m_temperature_p95",
            FieldSelector::height_agl(CanonicalField::Temperature, 2).with_percentile(95),
        ),
        (
            "nbm_qmd_prob_2m_temperature_below_270p928k",
            FieldSelector::height_agl(CanonicalField::Temperature, 2)
                .with_probability(ProbabilitySelection::below_milli(270_928)),
        ),
        (
            "nbm_qmd_prob_2m_temperature_below_273p15k",
            FieldSelector::height_agl(CanonicalField::Temperature, 2)
                .with_probability(ProbabilitySelection::below_milli(273_150)),
        ),
        (
            "nbm_qmd_prob_2m_temperature_above_299p817k",
            FieldSelector::height_agl(CanonicalField::Temperature, 2)
                .with_probability(ProbabilitySelection::above_milli(299_817)),
        ),
        (
            "nbm_qmd_prob_2m_temperature_above_305p372k",
            FieldSelector::height_agl(CanonicalField::Temperature, 2)
                .with_probability(ProbabilitySelection::above_milli(305_372)),
        ),
        (
            "nbm_qmd_prob_2m_temperature_above_310p928k",
            FieldSelector::height_agl(CanonicalField::Temperature, 2)
                .with_probability(ProbabilitySelection::above_milli(310_928)),
        ),
        (
            "nbm_qmd_prob_2m_temperature_above_316p483k",
            FieldSelector::height_agl(CanonicalField::Temperature, 2)
                .with_probability(ProbabilitySelection::above_milli(316_483)),
        ),
        (
            "nbm_qmd_2m_dewpoint_mean",
            FieldSelector::height_agl(CanonicalField::Dewpoint, 2).with_ensemble_mean(),
        ),
        (
            "nbm_qmd_2m_dewpoint_stddev",
            FieldSelector::height_agl(CanonicalField::Dewpoint, 2)
                .with_ensemble_standard_deviation(),
        ),
        (
            "nbm_qmd_2m_dewpoint_p05",
            FieldSelector::height_agl(CanonicalField::Dewpoint, 2).with_percentile(5),
        ),
        (
            "nbm_qmd_2m_dewpoint_p10",
            FieldSelector::height_agl(CanonicalField::Dewpoint, 2).with_percentile(10),
        ),
        (
            "nbm_qmd_2m_dewpoint_p25",
            FieldSelector::height_agl(CanonicalField::Dewpoint, 2).with_percentile(25),
        ),
        (
            "nbm_qmd_2m_dewpoint_p50",
            FieldSelector::height_agl(CanonicalField::Dewpoint, 2).with_percentile(50),
        ),
        (
            "nbm_qmd_2m_dewpoint_p75",
            FieldSelector::height_agl(CanonicalField::Dewpoint, 2).with_percentile(75),
        ),
        (
            "nbm_qmd_2m_dewpoint_p90",
            FieldSelector::height_agl(CanonicalField::Dewpoint, 2).with_percentile(90),
        ),
        (
            "nbm_qmd_2m_dewpoint_p95",
            FieldSelector::height_agl(CanonicalField::Dewpoint, 2).with_percentile(95),
        ),
        (
            "nbm_qmd_prob_2m_dewpoint_below_273p15k",
            FieldSelector::height_agl(CanonicalField::Dewpoint, 2)
                .with_probability(ProbabilitySelection::below_milli(273_150)),
        ),
        (
            "nbm_qmd_prob_2m_dewpoint_above_288p706k",
            FieldSelector::height_agl(CanonicalField::Dewpoint, 2)
                .with_probability(ProbabilitySelection::above_milli(288_706)),
        ),
        (
            "nbm_qmd_prob_2m_dewpoint_above_291p483k",
            FieldSelector::height_agl(CanonicalField::Dewpoint, 2)
                .with_probability(ProbabilitySelection::above_milli(291_483)),
        ),
        (
            "nbm_qmd_prob_2m_dewpoint_above_294p261k",
            FieldSelector::height_agl(CanonicalField::Dewpoint, 2)
                .with_probability(ProbabilitySelection::above_milli(294_261)),
        ),
        (
            "nbm_qmd_prob_2m_dewpoint_above_297p039k",
            FieldSelector::height_agl(CanonicalField::Dewpoint, 2)
                .with_probability(ProbabilitySelection::above_milli(297_039)),
        ),
        (
            "nbm_qmd_2m_relative_humidity_p05",
            FieldSelector::height_agl(CanonicalField::RelativeHumidity, 2).with_percentile(5),
        ),
        (
            "nbm_qmd_2m_relative_humidity_p10",
            FieldSelector::height_agl(CanonicalField::RelativeHumidity, 2).with_percentile(10),
        ),
        (
            "nbm_qmd_2m_relative_humidity_p25",
            FieldSelector::height_agl(CanonicalField::RelativeHumidity, 2).with_percentile(25),
        ),
        (
            "nbm_qmd_2m_relative_humidity_p50",
            FieldSelector::height_agl(CanonicalField::RelativeHumidity, 2).with_percentile(50),
        ),
        (
            "nbm_qmd_2m_relative_humidity_p75",
            FieldSelector::height_agl(CanonicalField::RelativeHumidity, 2).with_percentile(75),
        ),
        (
            "nbm_qmd_2m_relative_humidity_p90",
            FieldSelector::height_agl(CanonicalField::RelativeHumidity, 2).with_percentile(90),
        ),
        (
            "nbm_qmd_2m_relative_humidity_p95",
            FieldSelector::height_agl(CanonicalField::RelativeHumidity, 2).with_percentile(95),
        ),
        (
            "nbm_qmd_10m_wind_gust_mean",
            FieldSelector::height_agl(CanonicalField::WindGust, 10).with_ensemble_mean(),
        ),
        (
            "nbm_qmd_10m_wind_gust_stddev",
            FieldSelector::height_agl(CanonicalField::WindGust, 10)
                .with_ensemble_standard_deviation(),
        ),
        (
            "nbm_qmd_prob_10m_wind_gust_above_17p4911ms",
            FieldSelector::height_agl(CanonicalField::WindGust, 10)
                .with_probability(ProbabilitySelection::above_milli(17_491)),
        ),
        (
            "nbm_qmd_prob_10m_wind_gust_above_21p0922ms",
            FieldSelector::height_agl(CanonicalField::WindGust, 10)
                .with_probability(ProbabilitySelection::above_milli(21_092)),
        ),
        (
            "nbm_qmd_10m_wind_gust_p05",
            FieldSelector::height_agl(CanonicalField::WindGust, 10).with_percentile(5),
        ),
        (
            "nbm_qmd_10m_wind_gust_p10",
            FieldSelector::height_agl(CanonicalField::WindGust, 10).with_percentile(10),
        ),
        (
            "nbm_qmd_10m_wind_gust_p25",
            FieldSelector::height_agl(CanonicalField::WindGust, 10).with_percentile(25),
        ),
        (
            "nbm_qmd_10m_wind_gust_p50",
            FieldSelector::height_agl(CanonicalField::WindGust, 10).with_percentile(50),
        ),
        (
            "nbm_qmd_10m_wind_gust_p75",
            FieldSelector::height_agl(CanonicalField::WindGust, 10).with_percentile(75),
        ),
        (
            "nbm_qmd_10m_wind_gust_p90",
            FieldSelector::height_agl(CanonicalField::WindGust, 10).with_percentile(90),
        ),
        (
            "nbm_qmd_10m_wind_gust_p95",
            FieldSelector::height_agl(CanonicalField::WindGust, 10).with_percentile(95),
        ),
        (
            "nbm_qmd_prob_10m_wind_gust_above_24p6933ms",
            FieldSelector::height_agl(CanonicalField::WindGust, 10)
                .with_probability(ProbabilitySelection::above_milli(24_693)),
        ),
        (
            "nbm_qmd_prob_10m_wind_gust_above_28p8089ms",
            FieldSelector::height_agl(CanonicalField::WindGust, 10)
                .with_probability(ProbabilitySelection::above_milli(28_809)),
        ),
        (
            "nbm_qmd_prob_10m_wind_gust_above_32p9244ms",
            FieldSelector::height_agl(CanonicalField::WindGust, 10)
                .with_probability(ProbabilitySelection::above_milli(32_924)),
        ),
        (
            "nbm_qmd_10m_wind_speed_mean",
            FieldSelector::height_agl(CanonicalField::WindSpeed, 10).with_ensemble_mean(),
        ),
        (
            "nbm_qmd_10m_wind_speed_stddev",
            FieldSelector::height_agl(CanonicalField::WindSpeed, 10)
                .with_ensemble_standard_deviation(),
        ),
        (
            "nbm_qmd_prob_10m_wind_speed_above_8p7456ms",
            FieldSelector::height_agl(CanonicalField::WindSpeed, 10)
                .with_probability(ProbabilitySelection::above_milli(8_746)),
        ),
        (
            "nbm_qmd_prob_10m_wind_speed_above_11p3177ms",
            FieldSelector::height_agl(CanonicalField::WindSpeed, 10)
                .with_probability(ProbabilitySelection::above_milli(11_318)),
        ),
        (
            "nbm_qmd_10m_wind_speed_p05",
            FieldSelector::height_agl(CanonicalField::WindSpeed, 10).with_percentile(5),
        ),
        (
            "nbm_qmd_10m_wind_speed_p10",
            FieldSelector::height_agl(CanonicalField::WindSpeed, 10).with_percentile(10),
        ),
        (
            "nbm_qmd_10m_wind_speed_p25",
            FieldSelector::height_agl(CanonicalField::WindSpeed, 10).with_percentile(25),
        ),
        (
            "nbm_qmd_10m_wind_speed_p50",
            FieldSelector::height_agl(CanonicalField::WindSpeed, 10).with_percentile(50),
        ),
        (
            "nbm_qmd_10m_wind_speed_p75",
            FieldSelector::height_agl(CanonicalField::WindSpeed, 10).with_percentile(75),
        ),
        (
            "nbm_qmd_10m_wind_speed_p90",
            FieldSelector::height_agl(CanonicalField::WindSpeed, 10).with_percentile(90),
        ),
        (
            "nbm_qmd_10m_wind_speed_p95",
            FieldSelector::height_agl(CanonicalField::WindSpeed, 10).with_percentile(95),
        ),
        (
            "nbm_qmd_prob_10m_wind_speed_above_15p4333ms",
            FieldSelector::height_agl(CanonicalField::WindSpeed, 10)
                .with_probability(ProbabilitySelection::above_milli(15_433)),
        ),
        (
            "nbm_qmd_prob_10m_wind_speed_above_17p4911ms",
            FieldSelector::height_agl(CanonicalField::WindSpeed, 10)
                .with_probability(ProbabilitySelection::above_milli(17_491)),
        ),
        (
            "nbm_qmd_prob_10m_wind_speed_above_24p6933ms",
            FieldSelector::height_agl(CanonicalField::WindSpeed, 10)
                .with_probability(ProbabilitySelection::above_milli(24_693)),
        ),
    ];

    for (slug, selector) in cases {
        let plan = plot_recipe_fetch_plan(slug, ModelId::Nbm).unwrap();
        assert_eq!(plan.product, "qmd/co", "{slug}");
        assert_eq!(
            plan.fetch_policy,
            PlotRecipeFetchPolicy::PreferIndexedSubset,
            "{slug}"
        );
        assert_eq!(plan.selectors(), vec![selector], "{slug}");
    }
}

#[test]
fn nbm_qmd_temperature_stat_recipes_block_for_non_nbm_models() {
    let blockers = plot_recipe_fetch_blockers("nbm_qmd_2m_temperature_p50", ModelId::Gfs).unwrap();
    assert!(
        blockers
            .iter()
            .any(|blocker| { blocker.reason.contains("not yet supported for model 'gfs'") })
    );
}

#[test]
fn aigefs_spread_recipes_use_spr_products_and_stddev_selectors() {
    let cases = [
        (
            "aigefs_spr_2m_temperature_stddev",
            "sfc/spr",
            FieldSelector::height_agl(CanonicalField::Temperature, 2)
                .with_ensemble_standard_deviation(),
        ),
        (
            "aigefs_spr_500mb_height_stddev",
            "pres/spr",
            FieldSelector::isobaric(CanonicalField::GeopotentialHeight, 500)
                .with_ensemble_standard_deviation(),
        ),
        (
            "aigefs_spr_500mb_temperature_stddev",
            "pres/spr",
            FieldSelector::isobaric(CanonicalField::Temperature, 500)
                .with_ensemble_standard_deviation(),
        ),
        (
            "aigefs_spr_mslp_stddev",
            "sfc/spr",
            FieldSelector::mean_sea_level(CanonicalField::PressureReducedToMeanSeaLevel)
                .with_ensemble_standard_deviation(),
        ),
        (
            "aigefs_spr_6h_qpf_stddev",
            "sfc/spr",
            FieldSelector::surface(CanonicalField::TotalPrecipitation)
                .with_ensemble_standard_deviation(),
        ),
    ];

    for (slug, product, selector) in cases {
        let plan = plot_recipe_fetch_plan(slug, ModelId::Aigefs).unwrap();
        assert_eq!(plan.product, product, "{slug}");
        assert_eq!(
            plan.fetch_policy,
            PlotRecipeFetchPolicy::PreferIndexedSubset,
            "{slug}"
        );
        assert_eq!(plan.selectors(), vec![selector], "{slug}");

        let blockers = plot_recipe_fetch_blockers(slug, ModelId::Aigfs).unwrap();
        assert!(
            blockers
                .iter()
                .any(|blocker| blocker.reason.contains("model 'aigfs'")),
            "{slug}"
        );
    }
}

#[test]
fn gefs_native_stat_recipes_use_geavg_gespr_products_and_exact_selectors() {
    let cases = [
        (
            "gefs_avg_2m_temperature_10m_winds",
            "pgrb2ap5/geavg",
            vec![
                FieldSelector::height_agl(CanonicalField::Temperature, 2).with_ensemble_mean(),
                FieldSelector::mean_sea_level(CanonicalField::PressureReducedToMeanSeaLevel)
                    .with_ensemble_mean(),
                FieldSelector::height_agl(CanonicalField::UWind, 10).with_ensemble_mean(),
                FieldSelector::height_agl(CanonicalField::VWind, 10).with_ensemble_mean(),
            ],
        ),
        (
            "gefs_avg_500mb_height_winds",
            "pgrb2ap5/geavg",
            vec![
                FieldSelector::isobaric(CanonicalField::GeopotentialHeight, 500)
                    .with_ensemble_mean(),
                FieldSelector::isobaric(CanonicalField::UWind, 500).with_ensemble_mean(),
                FieldSelector::isobaric(CanonicalField::VWind, 500).with_ensemble_mean(),
            ],
        ),
        (
            "gefs_avg_500mb_temperature_height_winds",
            "pgrb2ap5/geavg",
            vec![
                FieldSelector::isobaric(CanonicalField::Temperature, 500).with_ensemble_mean(),
                FieldSelector::isobaric(CanonicalField::GeopotentialHeight, 500)
                    .with_ensemble_mean(),
                FieldSelector::isobaric(CanonicalField::UWind, 500).with_ensemble_mean(),
                FieldSelector::isobaric(CanonicalField::VWind, 500).with_ensemble_mean(),
            ],
        ),
        (
            "gefs_avg_500mb_rh_height_winds",
            "pgrb2ap5/geavg",
            vec![
                FieldSelector::isobaric(CanonicalField::RelativeHumidity, 500).with_ensemble_mean(),
                FieldSelector::isobaric(CanonicalField::GeopotentialHeight, 500)
                    .with_ensemble_mean(),
                FieldSelector::isobaric(CanonicalField::UWind, 500).with_ensemble_mean(),
                FieldSelector::isobaric(CanonicalField::VWind, 500).with_ensemble_mean(),
            ],
        ),
        (
            "gefs_avg_mslp_10m_winds",
            "pgrb2ap5/geavg",
            vec![
                FieldSelector::mean_sea_level(CanonicalField::PressureReducedToMeanSeaLevel)
                    .with_ensemble_mean(),
                FieldSelector::height_agl(CanonicalField::UWind, 10).with_ensemble_mean(),
                FieldSelector::height_agl(CanonicalField::VWind, 10).with_ensemble_mean(),
            ],
        ),
        (
            "gefs_avg_precipitable_water",
            "pgrb2ap5/geavg",
            vec![
                FieldSelector::entire_atmosphere(CanonicalField::PrecipitableWater)
                    .with_ensemble_mean(),
            ],
        ),
        (
            "gefs_spr_2m_temperature_stddev",
            "pgrb2ap5/gespr",
            vec![
                FieldSelector::height_agl(CanonicalField::Temperature, 2)
                    .with_ensemble_standard_deviation(),
            ],
        ),
        (
            "gefs_avg_2m_relative_humidity",
            "pgrb2ap5/geavg",
            vec![
                FieldSelector::height_agl(CanonicalField::RelativeHumidity, 2).with_ensemble_mean(),
            ],
        ),
        (
            "gefs_spr_2m_relative_humidity_stddev",
            "pgrb2ap5/gespr",
            vec![
                FieldSelector::height_agl(CanonicalField::RelativeHumidity, 2)
                    .with_ensemble_standard_deviation(),
            ],
        ),
        (
            "gefs_spr_500mb_height_stddev",
            "pgrb2ap5/gespr",
            vec![
                FieldSelector::isobaric(CanonicalField::GeopotentialHeight, 500)
                    .with_ensemble_standard_deviation(),
            ],
        ),
        (
            "gefs_spr_500mb_temperature_stddev",
            "pgrb2ap5/gespr",
            vec![
                FieldSelector::isobaric(CanonicalField::Temperature, 500)
                    .with_ensemble_standard_deviation(),
            ],
        ),
        (
            "gefs_spr_500mb_rh_stddev",
            "pgrb2ap5/gespr",
            vec![
                FieldSelector::isobaric(CanonicalField::RelativeHumidity, 500)
                    .with_ensemble_standard_deviation(),
            ],
        ),
        (
            "gefs_spr_mslp_stddev",
            "pgrb2ap5/gespr",
            vec![
                FieldSelector::mean_sea_level(CanonicalField::PressureReducedToMeanSeaLevel)
                    .with_ensemble_standard_deviation(),
            ],
        ),
        (
            "gefs_spr_precipitable_water_stddev",
            "pgrb2ap5/gespr",
            vec![
                FieldSelector::entire_atmosphere(CanonicalField::PrecipitableWater)
                    .with_ensemble_standard_deviation(),
            ],
        ),
        (
            "gefs_avg_cloud_cover",
            "pgrb2ap5/geavg",
            vec![
                FieldSelector::entire_atmosphere(CanonicalField::TotalCloudCover)
                    .with_ensemble_mean(),
            ],
        ),
        (
            "gefs_spr_cloud_cover_stddev",
            "pgrb2ap5/gespr",
            vec![
                FieldSelector::entire_atmosphere(CanonicalField::TotalCloudCover)
                    .with_ensemble_standard_deviation(),
            ],
        ),
        (
            "gefs_avg_6h_qpf",
            "pgrb2ap5/geavg",
            vec![FieldSelector::surface(CanonicalField::TotalPrecipitation).with_ensemble_mean()],
        ),
        (
            "gefs_spr_6h_qpf_stddev",
            "pgrb2ap5/gespr",
            vec![
                FieldSelector::surface(CanonicalField::TotalPrecipitation)
                    .with_ensemble_standard_deviation(),
            ],
        ),
    ];

    for (slug, product, selectors) in cases {
        let plan = plot_recipe_fetch_plan(slug, ModelId::Gefs).unwrap();
        assert_eq!(plan.product, product, "{slug}");
        assert_eq!(
            plan.fetch_policy,
            PlotRecipeFetchPolicy::PreferIndexedSubset,
            "{slug}"
        );
        assert_eq!(plan.selectors(), selectors, "{slug}");

        let blockers = plot_recipe_fetch_blockers(slug, ModelId::Gfs).unwrap();
        assert!(
            blockers
                .iter()
                .any(|blocker| blocker.reason.contains("model 'gfs'")),
            "{slug}"
        );
    }
}

#[test]
fn hgefs_spread_recipes_use_spr_products_and_stddev_selectors() {
    let cases = [
        (
            "hgefs_spr_2m_temperature_stddev",
            "sfc/spr",
            FieldSelector::height_agl(CanonicalField::Temperature, 2)
                .with_ensemble_standard_deviation(),
        ),
        (
            "hgefs_spr_500mb_height_stddev",
            "pres/spr",
            FieldSelector::isobaric(CanonicalField::GeopotentialHeight, 500)
                .with_ensemble_standard_deviation(),
        ),
        (
            "hgefs_spr_500mb_temperature_stddev",
            "pres/spr",
            FieldSelector::isobaric(CanonicalField::Temperature, 500)
                .with_ensemble_standard_deviation(),
        ),
        (
            "hgefs_spr_mslp_stddev",
            "sfc/spr",
            FieldSelector::mean_sea_level(CanonicalField::PressureReducedToMeanSeaLevel)
                .with_ensemble_standard_deviation(),
        ),
        (
            "hgefs_spr_6h_qpf_stddev",
            "sfc/spr",
            FieldSelector::surface(CanonicalField::TotalPrecipitation)
                .with_ensemble_standard_deviation(),
        ),
    ];

    for (slug, product, selector) in cases {
        let plan = plot_recipe_fetch_plan(slug, ModelId::Hgefs).unwrap();
        assert_eq!(plan.product, product, "{slug}");
        assert_eq!(
            plan.fetch_policy,
            PlotRecipeFetchPolicy::PreferIndexedSubset,
            "{slug}"
        );
        assert_eq!(plan.selectors(), vec![selector], "{slug}");

        let blockers = plot_recipe_fetch_blockers(slug, ModelId::Aigefs).unwrap();
        assert!(
            blockers
                .iter()
                .any(|blocker| blocker.reason.contains("model 'aigefs'")),
            "{slug}"
        );
    }
}

#[test]
fn href_spread_recipes_use_sprd_product_and_spread_selectors() {
    let cases = [
        (
            "href_sprd_2m_temperature",
            FieldSelector::height_agl(CanonicalField::Temperature, 2)
                .with_product(FieldProduct::EnsembleSpread),
        ),
        (
            "href_sprd_2m_dewpoint",
            FieldSelector::height_agl(CanonicalField::Dewpoint, 2)
                .with_product(FieldProduct::EnsembleSpread),
        ),
        (
            "href_sprd_10m_wind_speed",
            FieldSelector::height_agl(CanonicalField::WindSpeed, 10)
                .with_product(FieldProduct::EnsembleSpread),
        ),
        (
            "href_sprd_mslp",
            FieldSelector::mean_sea_level(CanonicalField::PressureReducedToMeanSeaLevel)
                .with_product(FieldProduct::EnsembleSpread),
        ),
        (
            "href_sprd_precipitable_water",
            FieldSelector::entire_atmosphere(CanonicalField::PrecipitableWater)
                .with_product(FieldProduct::EnsembleSpread),
        ),
        (
            "href_sprd_500mb_height",
            FieldSelector::isobaric(CanonicalField::GeopotentialHeight, 500)
                .with_product(FieldProduct::EnsembleSpread),
        ),
        (
            "href_sprd_500mb_temperature",
            FieldSelector::isobaric(CanonicalField::Temperature, 500)
                .with_product(FieldProduct::EnsembleSpread),
        ),
    ];

    for (slug, selector) in cases {
        let plan = plot_recipe_fetch_plan(slug, ModelId::Href).unwrap();
        assert_eq!(plan.product, "ensprod/conus/sprd", "{slug}");
        assert_eq!(
            plan.fetch_policy,
            PlotRecipeFetchPolicy::PreferIndexedSubset,
            "{slug}"
        );
        assert_eq!(plan.selectors(), vec![selector], "{slug}");

        let blockers = plot_recipe_fetch_blockers(slug, ModelId::Gefs).unwrap();
        assert!(
            blockers
                .iter()
                .any(|blocker| blocker.reason.contains("model 'gefs'")),
            "{slug}"
        );
    }

    let generic = plot_recipe_fetch_blockers("2m_temperature", ModelId::Href).unwrap();
    assert!(
        generic
            .iter()
            .any(|blocker| { blocker.reason.contains("limited to explicit `href_sprd_*`") })
    );
}

#[test]
fn href_probability_recipes_use_prob_product_and_exact_selectors() {
    let cases = [
        (
            "href_prob_uh_2to5km_above_25",
            FieldSelector::height_layer_agl(CanonicalField::UpdraftHelicity, 2000, 5000)
                .with_probability(ProbabilitySelection::above_milli(25_000)),
        ),
        (
            "href_prob_uh_2to5km_above_75",
            FieldSelector::height_layer_agl(CanonicalField::UpdraftHelicity, 2000, 5000)
                .with_probability(ProbabilitySelection::above_milli(75_000)),
        ),
        (
            "href_prob_uh_2to5km_above_150",
            FieldSelector::height_layer_agl(CanonicalField::UpdraftHelicity, 2000, 5000)
                .with_probability(ProbabilitySelection::above_milli(150_000)),
        ),
        (
            "href_prob_2m_temperature_below_273p15k",
            FieldSelector::height_agl(CanonicalField::Temperature, 2)
                .with_probability(ProbabilitySelection::below_milli(273_150)),
        ),
        (
            "href_prob_2m_dewpoint_above_291p48k",
            FieldSelector::height_agl(CanonicalField::Dewpoint, 2)
                .with_probability(ProbabilitySelection::above_milli(291_480)),
        ),
        (
            "href_prob_2m_dewpoint_above_294p26k",
            FieldSelector::height_agl(CanonicalField::Dewpoint, 2)
                .with_probability(ProbabilitySelection::above_milli(294_260)),
        ),
        (
            "href_prob_pwat_above_25mm",
            FieldSelector::entire_atmosphere(CanonicalField::PrecipitableWater)
                .with_probability(ProbabilitySelection::above_milli(25_000)),
        ),
        (
            "href_prob_pwat_above_37p5mm",
            FieldSelector::entire_atmosphere(CanonicalField::PrecipitableWater)
                .with_probability(ProbabilitySelection::above_milli(37_500)),
        ),
        (
            "href_prob_pwat_above_50mm",
            FieldSelector::entire_atmosphere(CanonicalField::PrecipitableWater)
                .with_probability(ProbabilitySelection::above_milli(50_000)),
        ),
        (
            "href_prob_visibility_below_1600m",
            FieldSelector::surface(CanonicalField::Visibility)
                .with_probability(ProbabilitySelection::below_milli(1_600_000)),
        ),
        (
            "href_prob_visibility_below_3200m",
            FieldSelector::surface(CanonicalField::Visibility)
                .with_probability(ProbabilitySelection::below_milli(3_200_000)),
        ),
        (
            "href_prob_visibility_below_6400m",
            FieldSelector::surface(CanonicalField::Visibility)
                .with_probability(ProbabilitySelection::below_milli(6_400_000)),
        ),
        (
            "href_prob_10m_wind_speed_above_15p4ms",
            FieldSelector::height_agl(CanonicalField::WindSpeed, 10)
                .with_probability(ProbabilitySelection::above_milli(15_400)),
        ),
        (
            "href_prob_10m_wind_speed_above_20p6ms",
            FieldSelector::height_agl(CanonicalField::WindSpeed, 10)
                .with_probability(ProbabilitySelection::above_milli(20_600)),
        ),
        (
            "href_prob_10m_wind_speed_above_25p72ms",
            FieldSelector::height_agl(CanonicalField::WindSpeed, 10)
                .with_probability(ProbabilitySelection::above_milli(25_720)),
        ),
    ];

    for (slug, selector) in cases {
        let plan = plot_recipe_fetch_plan(slug, ModelId::Href).unwrap();
        assert_eq!(plan.product, "ensprod/conus/prob", "{slug}");
        assert_eq!(
            plan.fetch_policy,
            PlotRecipeFetchPolicy::PreferIndexedSubset,
            "{slug}"
        );
        assert_eq!(plan.selectors(), vec![selector], "{slug}");

        let blockers = plot_recipe_fetch_blockers(slug, ModelId::Gfs).unwrap();
        assert!(
            blockers
                .iter()
                .any(|blocker| blocker.reason.contains("only verified for HREF")),
            "{slug}"
        );
    }
}

#[test]
fn href_mean_recipes_use_mean_product_and_exact_selectors() {
    let cases = [
        (
            "href_mean_2m_temperature",
            FieldSelector::height_agl(CanonicalField::Temperature, 2).with_ensemble_mean(),
        ),
        (
            "href_mean_2m_dewpoint",
            FieldSelector::height_agl(CanonicalField::Dewpoint, 2).with_ensemble_mean(),
        ),
        (
            "href_mean_10m_wind_speed",
            FieldSelector::height_agl(CanonicalField::WindSpeed, 10).with_ensemble_mean(),
        ),
        (
            "href_mean_mslp",
            FieldSelector::mean_sea_level(CanonicalField::PressureReducedToMeanSeaLevel)
                .with_ensemble_mean(),
        ),
        (
            "href_mean_precipitable_water",
            FieldSelector::entire_atmosphere(CanonicalField::PrecipitableWater)
                .with_ensemble_mean(),
        ),
        (
            "href_mean_visibility",
            FieldSelector::surface(CanonicalField::Visibility).with_ensemble_mean(),
        ),
        (
            "href_mean_cloud_cover",
            FieldSelector::entire_atmosphere(CanonicalField::TotalCloudCover).with_ensemble_mean(),
        ),
        (
            "href_mean_500mb_height",
            FieldSelector::isobaric(CanonicalField::GeopotentialHeight, 500).with_ensemble_mean(),
        ),
        (
            "href_mean_500mb_temperature",
            FieldSelector::isobaric(CanonicalField::Temperature, 500).with_ensemble_mean(),
        ),
    ];

    for (slug, selector) in cases {
        let plan = plot_recipe_fetch_plan(slug, ModelId::Href).unwrap();
        assert_eq!(plan.product, "ensprod/conus/mean", "{slug}");
        assert_eq!(
            plan.fetch_policy,
            PlotRecipeFetchPolicy::PreferIndexedSubset,
            "{slug}"
        );
        assert_eq!(plan.selectors(), vec![selector], "{slug}");

        let blockers = plot_recipe_fetch_blockers(slug, ModelId::Gfs).unwrap();
        assert!(
            blockers
                .iter()
                .any(|blocker| blocker.reason.contains("only verified for HREF")),
            "{slug}"
        );
    }
}

#[test]
fn refs_spread_recipes_use_sprd_product_and_spread_selectors() {
    let cases = [
        (
            "refs_sprd_2m_temperature",
            FieldSelector::height_agl(CanonicalField::Temperature, 2)
                .with_product(FieldProduct::EnsembleSpread),
        ),
        (
            "refs_sprd_2m_dewpoint",
            FieldSelector::height_agl(CanonicalField::Dewpoint, 2)
                .with_product(FieldProduct::EnsembleSpread),
        ),
        (
            "refs_sprd_10m_wind_speed",
            FieldSelector::height_agl(CanonicalField::WindSpeed, 10)
                .with_product(FieldProduct::EnsembleSpread),
        ),
        (
            "refs_sprd_mslp",
            FieldSelector::mean_sea_level(CanonicalField::PressureReducedToMeanSeaLevel)
                .with_product(FieldProduct::EnsembleSpread),
        ),
        (
            "refs_sprd_precipitable_water",
            FieldSelector::entire_atmosphere(CanonicalField::PrecipitableWater)
                .with_product(FieldProduct::EnsembleSpread),
        ),
        (
            "refs_sprd_visibility",
            FieldSelector::surface(CanonicalField::Visibility)
                .with_product(FieldProduct::EnsembleSpread),
        ),
        (
            "refs_sprd_cloud_cover",
            FieldSelector::entire_atmosphere(CanonicalField::TotalCloudCover)
                .with_product(FieldProduct::EnsembleSpread),
        ),
        (
            "refs_sprd_500mb_height",
            FieldSelector::isobaric(CanonicalField::GeopotentialHeight, 500)
                .with_product(FieldProduct::EnsembleSpread),
        ),
        (
            "refs_sprd_500mb_temperature",
            FieldSelector::isobaric(CanonicalField::Temperature, 500)
                .with_product(FieldProduct::EnsembleSpread),
        ),
    ];

    for (slug, selector) in cases {
        let plan = plot_recipe_fetch_plan(slug, ModelId::Refs).unwrap();
        assert_eq!(plan.product, "sprd-conus", "{slug}");
        assert_eq!(
            plan.fetch_policy,
            PlotRecipeFetchPolicy::PreferIndexedSubset,
            "{slug}"
        );
        assert_eq!(plan.selectors(), vec![selector], "{slug}");

        let blockers = plot_recipe_fetch_blockers(slug, ModelId::RrfsPublic).unwrap();
        assert!(
            blockers
                .iter()
                .any(|blocker| blocker.reason.contains("model 'rrfs-public'")),
            "{slug}"
        );
    }

    let generic = plot_recipe_fetch_plan("2m_temperature_10m_winds", ModelId::Refs).unwrap();
    assert_eq!(generic.product, "mean-conus");

    let pressure = plot_recipe_fetch_plan("700mb_temperature_height_winds", ModelId::Refs).unwrap();
    assert_eq!(pressure.product, "mean-conus");

    let radar = plot_recipe_fetch_plan("composite_reflectivity", ModelId::Refs).unwrap();
    assert_eq!(radar.product, "pmmn-conus");
}

#[test]
fn refs_probability_recipes_use_prob_product_and_exact_selectors() {
    let cases = [
        (
            "refs_prob_uh_2to5km_above_25",
            FieldSelector::height_layer_agl(CanonicalField::UpdraftHelicity, 2000, 5000)
                .with_probability(ProbabilitySelection::above_milli(25_000)),
        ),
        (
            "refs_prob_uh_2to5km_above_75",
            FieldSelector::height_layer_agl(CanonicalField::UpdraftHelicity, 2000, 5000)
                .with_probability(ProbabilitySelection::above_milli(75_000)),
        ),
        (
            "refs_prob_uh_2to5km_above_150",
            FieldSelector::height_layer_agl(CanonicalField::UpdraftHelicity, 2000, 5000)
                .with_probability(ProbabilitySelection::above_milli(150_000)),
        ),
        (
            "refs_prob_2m_temperature_below_273p15k",
            FieldSelector::height_agl(CanonicalField::Temperature, 2)
                .with_probability(ProbabilitySelection::below_milli(273_150)),
        ),
        (
            "refs_prob_2m_dewpoint_above_291p48k",
            FieldSelector::height_agl(CanonicalField::Dewpoint, 2)
                .with_probability(ProbabilitySelection::above_milli(291_480)),
        ),
        (
            "refs_prob_2m_dewpoint_above_294p26k",
            FieldSelector::height_agl(CanonicalField::Dewpoint, 2)
                .with_probability(ProbabilitySelection::above_milli(294_260)),
        ),
        (
            "refs_prob_pwat_above_25mm",
            FieldSelector::entire_atmosphere(CanonicalField::PrecipitableWater)
                .with_probability(ProbabilitySelection::above_milli(25_000)),
        ),
        (
            "refs_prob_pwat_above_37p5mm",
            FieldSelector::entire_atmosphere(CanonicalField::PrecipitableWater)
                .with_probability(ProbabilitySelection::above_milli(37_500)),
        ),
        (
            "refs_prob_pwat_above_50mm",
            FieldSelector::entire_atmosphere(CanonicalField::PrecipitableWater)
                .with_probability(ProbabilitySelection::above_milli(50_000)),
        ),
        (
            "refs_prob_qpf_above_1mm",
            FieldSelector::surface(CanonicalField::TotalPrecipitation)
                .with_probability(ProbabilitySelection::above_milli(1_000)),
        ),
        (
            "refs_prob_qpf_above_2mm",
            FieldSelector::surface(CanonicalField::TotalPrecipitation)
                .with_probability(ProbabilitySelection::above_milli(2_000)),
        ),
        (
            "refs_prob_qpf_above_5mm",
            FieldSelector::surface(CanonicalField::TotalPrecipitation)
                .with_probability(ProbabilitySelection::above_milli(5_000)),
        ),
        (
            "refs_prob_qpf_above_10mm",
            FieldSelector::surface(CanonicalField::TotalPrecipitation)
                .with_probability(ProbabilitySelection::above_milli(10_000)),
        ),
        (
            "refs_prob_qpf_above_12p7mm",
            FieldSelector::surface(CanonicalField::TotalPrecipitation)
                .with_probability(ProbabilitySelection::above_milli(12_700)),
        ),
        (
            "refs_prob_qpf_above_25mm",
            FieldSelector::surface(CanonicalField::TotalPrecipitation)
                .with_probability(ProbabilitySelection::above_milli(25_000)),
        ),
        (
            "refs_prob_qpf_above_25p4mm",
            FieldSelector::surface(CanonicalField::TotalPrecipitation)
                .with_probability(ProbabilitySelection::above_milli(25_400)),
        ),
        (
            "refs_prob_qpf_above_50mm",
            FieldSelector::surface(CanonicalField::TotalPrecipitation)
                .with_probability(ProbabilitySelection::above_milli(50_000)),
        ),
        (
            "refs_prob_qpf_above_50p8mm",
            FieldSelector::surface(CanonicalField::TotalPrecipitation)
                .with_probability(ProbabilitySelection::above_milli(50_800)),
        ),
        (
            "refs_prob_qpf_above_76p2mm",
            FieldSelector::surface(CanonicalField::TotalPrecipitation)
                .with_probability(ProbabilitySelection::above_milli(76_200)),
        ),
        (
            "refs_prob_qpf_above_100mm",
            FieldSelector::surface(CanonicalField::TotalPrecipitation)
                .with_probability(ProbabilitySelection::above_milli(100_000)),
        ),
        (
            "refs_prob_qpf_above_127mm",
            FieldSelector::surface(CanonicalField::TotalPrecipitation)
                .with_probability(ProbabilitySelection::above_milli(127_000)),
        ),
        (
            "refs_prob_visibility_below_1600m",
            FieldSelector::surface(CanonicalField::Visibility)
                .with_probability(ProbabilitySelection::below_milli(1_600_000)),
        ),
        (
            "refs_prob_visibility_below_3200m",
            FieldSelector::surface(CanonicalField::Visibility)
                .with_probability(ProbabilitySelection::below_milli(3_200_000)),
        ),
        (
            "refs_prob_visibility_below_8049m",
            FieldSelector::surface(CanonicalField::Visibility)
                .with_probability(ProbabilitySelection::below_milli(8_049_000)),
        ),
        (
            "refs_prob_10m_wind_speed_above_15p4ms",
            FieldSelector::height_agl(CanonicalField::WindSpeed, 10)
                .with_probability(ProbabilitySelection::above_milli(15_400)),
        ),
        (
            "refs_prob_10m_wind_speed_above_20p6ms",
            FieldSelector::height_agl(CanonicalField::WindSpeed, 10)
                .with_probability(ProbabilitySelection::above_milli(20_600)),
        ),
        (
            "refs_prob_10m_wind_speed_above_25p72ms",
            FieldSelector::height_agl(CanonicalField::WindSpeed, 10)
                .with_probability(ProbabilitySelection::above_milli(25_720)),
        ),
    ];

    for (slug, selector) in cases {
        let plan = plot_recipe_fetch_plan(slug, ModelId::Refs).unwrap();
        assert_eq!(plan.product, "prob-conus", "{slug}");
        assert_eq!(
            plan.fetch_policy,
            PlotRecipeFetchPolicy::PreferIndexedSubset,
            "{slug}"
        );
        assert_eq!(plan.selectors(), vec![selector], "{slug}");

        let blockers = plot_recipe_fetch_blockers(slug, ModelId::Gfs).unwrap();
        assert!(
            blockers
                .iter()
                .any(|blocker| blocker.reason.contains("only verified for REFS")),
            "{slug}"
        );
    }
}

#[test]
fn sref_probability_recipes_use_prob_product_and_exact_selectors() {
    let cases = [
        (
            "sref_prob_2m_temperature_below_273k",
            FieldSelector::height_agl(CanonicalField::Temperature, 2)
                .with_probability(ProbabilitySelection::below_milli(273_000)),
        ),
        (
            "sref_prob_2m_temperature_above_298p8k",
            FieldSelector::height_agl(CanonicalField::Temperature, 2)
                .with_probability(ProbabilitySelection::above_milli(298_800)),
        ),
        (
            "sref_prob_850mb_temperature_below_273k",
            FieldSelector::isobaric(CanonicalField::Temperature, 850)
                .with_probability(ProbabilitySelection::below_milli(273_000)),
        ),
        (
            "sref_prob_visibility_below_1609m",
            FieldSelector::surface(CanonicalField::Visibility)
                .with_probability(ProbabilitySelection::below_milli(1_609_000)),
        ),
        (
            "sref_prob_visibility_below_402m",
            FieldSelector::surface(CanonicalField::Visibility)
                .with_probability(ProbabilitySelection::below_milli(402_000)),
        ),
        (
            "sref_prob_visibility_below_804m",
            FieldSelector::surface(CanonicalField::Visibility)
                .with_probability(ProbabilitySelection::below_milli(804_000)),
        ),
        (
            "sref_prob_visibility_below_3218m",
            FieldSelector::surface(CanonicalField::Visibility)
                .with_probability(ProbabilitySelection::below_milli(3_218_000)),
        ),
        (
            "sref_prob_visibility_below_4827m",
            FieldSelector::surface(CanonicalField::Visibility)
                .with_probability(ProbabilitySelection::below_milli(4_827_000)),
        ),
        (
            "sref_prob_visibility_below_8046m",
            FieldSelector::surface(CanonicalField::Visibility)
                .with_probability(ProbabilitySelection::below_milli(8_046_000)),
        ),
        (
            "sref_prob_visibility_below_9654m",
            FieldSelector::surface(CanonicalField::Visibility)
                .with_probability(ProbabilitySelection::below_milli(9_654_000)),
        ),
        (
            "sref_prob_10m_wind_speed_above_12p89ms",
            FieldSelector::height_agl(CanonicalField::WindSpeed, 10)
                .with_probability(ProbabilitySelection::above_milli(12_890)),
        ),
        (
            "sref_prob_10m_wind_speed_above_17p5ms",
            FieldSelector::height_agl(CanonicalField::WindSpeed, 10)
                .with_probability(ProbabilitySelection::above_milli(17_500)),
        ),
        (
            "sref_prob_10m_wind_speed_above_25p78ms",
            FieldSelector::height_agl(CanonicalField::WindSpeed, 10)
                .with_probability(ProbabilitySelection::above_milli(25_780)),
        ),
    ];

    for (slug, selector) in cases {
        let plan = plot_recipe_fetch_plan(slug, ModelId::Sref).unwrap();
        assert_eq!(plan.product, "ensprod/pgrb212/prob_3hrly", "{slug}");
        assert_eq!(
            plan.fetch_policy,
            PlotRecipeFetchPolicy::PreferIndexedSubset,
            "{slug}"
        );
        assert_eq!(plan.selectors(), vec![selector], "{slug}");
    }
}

#[test]
fn sref_probability_recipes_block_for_non_sref_models() {
    for model in [ModelId::Gfs, ModelId::Nbm] {
        let blockers =
            plot_recipe_fetch_blockers("sref_prob_2m_temperature_below_273k", model).unwrap();
        assert!(
            blockers
                .iter()
                .any(|blocker| blocker.reason.contains("only verified for SREF")),
            "{model:?}: {blockers:?}"
        );
    }
}

#[test]
fn direct_upper_air_200mb_recipe_is_now_supported() {
    let blockers = plot_recipe_fetch_blockers("200mb_height_winds", ModelId::Gfs).unwrap();
    assert!(blockers.is_empty());

    let plan = plot_recipe_fetch_plan("200mb_height_winds", ModelId::Gfs).unwrap();
    assert_eq!(plan.product, "pgrb2.0p25");
}

#[test]
fn simulated_ir_recipe_remains_blocked_for_unverified_models() {
    let blockers =
        plot_recipe_fetch_blockers("simulated_ir_satellite", ModelId::EcmwfOpenData).unwrap();
    assert_eq!(blockers.len(), 1);
    assert!(
        blockers[0]
            .reason
            .contains("GRIB signature is not verified yet")
    );
}

#[test]
fn smoke_recipes_remain_blocked_for_unverified_models() {
    let rrfs_smoke = plot_recipe_fetch_blockers("smoke_pm25_native", ModelId::RrfsA).unwrap();
    assert_eq!(rrfs_smoke.len(), 1);
    assert!(rrfs_smoke[0].reason.contains("HRRR wrfnat"));

    let gfs_column = plot_recipe_fetch_blockers("smoke_column", ModelId::Gfs).unwrap();
    assert_eq!(gfs_column.len(), 1);
    assert!(gfs_column[0].reason.contains("native smoke GRIB signature"));
}

#[test]
fn lightning_flash_density_blocker_uses_verified_hrrr_message_evidence() {
    let blockers = plot_recipe_fetch_blockers("lightning_flash_density", ModelId::Hrrr).unwrap();
    assert_eq!(blockers.len(), 1);
    let reason = &blockers[0].reason;
    assert!(reason.contains("LTNGSD"));
    assert!(reason.contains("discipline 0/category 17/number 0"));
    assert!(reason.contains("m^-2 s^-1"));
    assert!(reason.contains("LTNG"));
    assert!(reason.contains("flash-density parameters 2/3/4"));
}

#[test]
fn hrrr_urls_match_expected_operational_paths() {
    let request = ModelRunRequest::new(
        ModelId::Hrrr,
        CycleSpec::new("20260414", 19).unwrap(),
        2,
        "sfc",
    )
    .unwrap();
    let urls = resolve_urls(&request).unwrap();
    assert_eq!(
        urls[0].grib_url,
        "https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod/hrrr.20260414/conus/hrrr.t19z.wrfsfcf02.grib2"
    );
}

#[test]
fn gfs_urls_match_expected_operational_paths() {
    let request = ModelRunRequest::new(
        ModelId::Gfs,
        CycleSpec::new("20260414", 18).unwrap(),
        12,
        "pgrb2.0p25",
    )
    .unwrap();
    let urls = resolve_urls(&request).unwrap();
    assert_eq!(
        urls[0].grib_url,
        "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gfs.20260414/18/atmos/gfs.t18z.pgrb2.0p25.f012"
    );
}

#[test]
fn ecmwf_urls_match_open_data_feed() {
    let request = ModelRunRequest::new(
        ModelId::EcmwfOpenData,
        CycleSpec::new("20260414", 12).unwrap(),
        6,
        "oper",
    )
    .unwrap();
    let urls = resolve_urls(&request).unwrap();
    assert_eq!(
        urls[0].grib_url,
        "https://data.ecmwf.int/forecasts/20260414/12z/ifs/0p25/oper/20260414120000-6h-oper-fc.grib2"
    );

    let wave = ModelRunRequest::new(
        ModelId::EcmwfOpenData,
        CycleSpec::new("20260414", 12).unwrap(),
        6,
        "wave",
    )
    .unwrap();
    let urls = resolve_urls(&wave).unwrap();
    assert_eq!(
        urls[0].grib_url,
        "https://data.ecmwf.int/forecasts/20260414/12z/ifs/0p25/wave/20260414120000-6h-wave-fc.grib2"
    );
}

#[test]
fn ecmwf_urls_support_six_utc_cycles() {
    let request = ModelRunRequest::new(
        ModelId::EcmwfOpenData,
        CycleSpec::new("20260414", 6).unwrap(),
        144,
        "oper",
    )
    .unwrap();
    let urls = resolve_urls(&request).unwrap();
    assert_eq!(
        urls[0].grib_url,
        "https://data.ecmwf.int/forecasts/20260414/06z/ifs/0p25/oper/20260414060000-144h-oper-fc.grib2"
    );
}

#[test]
fn ecmwf_urls_reject_unsupported_cycle_hour_cadence() {
    let request = ModelRunRequest::new(
        ModelId::EcmwfOpenData,
        CycleSpec::new("20260414", 6).unwrap(),
        150,
        "oper",
    )
    .unwrap();
    let err = resolve_urls(&request).unwrap_err();
    assert!(matches!(
        err,
        ModelError::UnsupportedForecastHour {
            model: ModelId::EcmwfOpenData,
            cycle_hour: 6,
            forecast_hour: 150,
            ..
        }
    ));
}

#[test]
fn rrfs_a_urls_match_live_bucket_pattern() {
    let request = ModelRunRequest::new(
        ModelId::RrfsA,
        CycleSpec::new("20260414", 20).unwrap(),
        2,
        "prs-conus",
    )
    .unwrap();
    let urls = resolve_urls(&request).unwrap();
    assert_eq!(
        urls[0].grib_url,
        "https://noaa-rrfs-pds.s3.amazonaws.com/rrfs_a/rrfs.20260414/20/rrfs.t20z.prslev.3km.f002.conus.grib2"
    );
    assert_eq!(
        urls[0].idx_url.as_deref(),
        Some(
            "https://noaa-rrfs-pds.s3.amazonaws.com/rrfs_a/rrfs.20260414/20/rrfs.t20z.prslev.3km.f002.conus.grib2.idx"
        )
    );
}

#[test]
fn rrfs_public_urls_match_public_bucket_pattern() {
    let request = ModelRunRequest::new(
        ModelId::RrfsPublic,
        CycleSpec::new("20260507", 0).unwrap(),
        24,
        "prs-conus",
    )
    .unwrap();
    let urls = resolve_urls(&request).unwrap();
    assert_eq!(
        urls[0].grib_url,
        "https://noaa-rrfs-pds.s3.amazonaws.com/rrfs_public/rrfs.20260507/00/rrfs.t00z.prslev.3km.f024.conus.grib2"
    );
    assert_eq!(
        urls[0].idx_url.as_deref(),
        Some(
            "https://noaa-rrfs-pds.s3.amazonaws.com/rrfs_public/rrfs.20260507/00/rrfs.t00z.prslev.3km.f024.conus.grib2.idx"
        )
    );
}

#[test]
fn rrfs_firewx_urls_match_public_firewx_bucket_pattern() {
    let request = ModelRunRequest::new(
        ModelId::RrfsFireWx,
        CycleSpec::new("20260507", 6).unwrap(),
        24,
        "prs-firewx",
    )
    .unwrap();
    let urls = resolve_urls(&request).unwrap();
    assert_eq!(
        urls[0].grib_url,
        "https://noaa-rrfs-pds.s3.amazonaws.com/rrfs_public/firewx.20260507/06/rrfs.t06z.prslev.1p5km.f024.firewx_lcc.grib2"
    );
    assert_eq!(
        urls[0].idx_url.as_deref(),
        Some(
            "https://noaa-rrfs-pds.s3.amazonaws.com/rrfs_public/firewx.20260507/06/rrfs.t06z.prslev.1p5km.f024.firewx_lcc.grib2.idx"
        )
    );
}

#[test]
fn rrfs_a_subhourly_hi_urls_match_live_bucket_pattern() {
    let request = ModelRunRequest::new(
        ModelId::RrfsA,
        CycleSpec::new("20260414", 20).unwrap(),
        2,
        "subh-hi",
    )
    .unwrap();
    let urls = resolve_urls(&request).unwrap();
    assert_eq!(
        urls[0].grib_url,
        "https://noaa-rrfs-pds.s3.amazonaws.com/rrfs_a/rrfs.20260414/20/rrfs.t20z.prslev.2p5km.subh.f002.hi.grib2"
    );
}

#[test]
fn wrf_gdex_urls_match_gdex_wrfout_pattern() {
    let request = ModelRunRequest::new(
        ModelId::WrfGdex,
        CycleSpec::new("20150101", 0).unwrap(),
        0,
        "d010047-d01",
    )
    .unwrap();
    let urls = resolve_urls(&request).unwrap();
    assert_eq!(
        urls[0].grib_url,
        "https://tds.gdex.ucar.edu/thredds/fileServer/files/d010047/201501/wrfout_d01_2015-01-01_00:00:00.nc"
    );
    assert!(urls[0].idx_url.is_none());
}

#[test]
fn wrf_gdex_urls_roll_forward_by_forecast_hour() {
    let request = ModelRunRequest::new(
        ModelId::WrfGdex,
        CycleSpec::new("20150101", 0).unwrap(),
        15,
        "d010047-d01",
    )
    .unwrap();
    let urls = resolve_urls(&request).unwrap();
    assert_eq!(
        urls[0].grib_url,
        "https://tds.gdex.ucar.edu/thredds/fileServer/files/d010047/201501/wrfout_d01_2015-01-01_15:00:00.nc"
    );
}

#[test]
fn wrf_gdex_hist2d_urls_match_osdf_pattern() {
    let request = ModelRunRequest::new(
        ModelId::WrfGdex,
        CycleSpec::new("19950101", 0).unwrap(),
        12,
        "d612005-hist2d",
    )
    .unwrap();
    let urls = resolve_urls(&request).unwrap();
    assert_eq!(
        urls[0].grib_url,
        "https://tds.gdex.ucar.edu/thredds/fileServer/files/g/d612005/hist2D/199501/wrf2d_d01_1995-01-01_12:00:00.nc"
    );
}

#[test]
fn wrf_gdex_hist3d_urls_match_osdf_pattern() {
    let request = ModelRunRequest::new(
        ModelId::WrfGdex,
        CycleSpec::new("19950101", 0).unwrap(),
        12,
        "d612005-hist3d",
    )
    .unwrap();
    let urls = resolve_urls(&request).unwrap();
    assert_eq!(
        urls[0].grib_url,
        "https://tds.gdex.ucar.edu/thredds/fileServer/files/g/d612005/hist3D/199501/wrf3d_d01_1995-01-01_12:00:00.nc"
    );
}

#[test]
fn wrf_gdex_future2d_urls_match_osdf_pattern() {
    let request = ModelRunRequest::new(
        ModelId::WrfGdex,
        CycleSpec::new("20800101", 0).unwrap(),
        0,
        "d612005-future2d",
    )
    .unwrap();
    let urls = resolve_urls(&request).unwrap();
    assert_eq!(
        urls[0].grib_url,
        "https://tds.gdex.ucar.edu/thredds/fileServer/files/g/d612005/future2D/208001/wrf2d_d01_2080-01-01_00:00:00.nc"
    );
}

#[test]
fn wrf_gdex_future3d_urls_match_osdf_pattern() {
    let request = ModelRunRequest::new(
        ModelId::WrfGdex,
        CycleSpec::new("20800101", 0).unwrap(),
        0,
        "d612005-future3d",
    )
    .unwrap();
    let urls = resolve_urls(&request).unwrap();
    assert_eq!(
        urls[0].grib_url,
        "https://tds.gdex.ucar.edu/thredds/fileServer/files/g/d612005/future3D/208001/wrf3d_d01_2080-01-01_00:00:00.nc"
    );
}

#[test]
fn wrf_gdex_hist3d_rejects_off_cadence_valid_times() {
    let request = ModelRunRequest::new(
        ModelId::WrfGdex,
        CycleSpec::new("19950101", 0).unwrap(),
        1,
        "d612005-hist3d",
    )
    .unwrap();
    let err = resolve_urls(&request).unwrap_err();
    assert!(
        err.to_string().contains("every 3 hours"),
        "unexpected error: {err}"
    );
}
