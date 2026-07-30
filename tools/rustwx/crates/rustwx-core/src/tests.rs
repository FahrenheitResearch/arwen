use super::*;

#[test]
fn grid_shape_len_matches() {
    let shape = GridShape::new(3, 2).unwrap();
    assert_eq!(shape.len(), 6);
}

#[test]
fn grid_shape_rejects_overflow_ceiling_and_invalid_deserialization() {
    let unchecked_overflow = GridShape {
        nx: usize::MAX,
        ny: 2,
    };
    assert_eq!(
        unchecked_overflow.len(),
        0,
        "legacy len must not panic or wrap"
    );
    assert!(unchecked_overflow.checked_len().is_err());
    assert!(matches!(
        GridShape::new(usize::MAX, 2),
        Err(RustwxError::InvalidGridShape {
            nx: usize::MAX,
            ny: 2
        })
    ));
    assert!(matches!(
        GridShape::new(MAX_GRID_CELLS + 1, 1),
        Err(RustwxError::GridTooLarge {
            cells,
            max_cells: MAX_GRID_CELLS,
            ..
        }) if cells == MAX_GRID_CELLS + 1
    ));

    let largest_square = GridShape::new(5_000, 5_000).expect("ceiling is accepted");
    assert_eq!(largest_square.len(), MAX_GRID_CELLS);

    #[derive(serde::Serialize)]
    struct UncheckedShape {
        nx: usize,
        ny: usize,
    }
    let encoded = bincode::serialize(&UncheckedShape {
        nx: usize::MAX,
        ny: 2,
    })
    .unwrap();
    assert!(bincode::deserialize::<GridShape>(&encoded).is_err());

    #[derive(serde::Serialize)]
    struct UncheckedGrid {
        shape: GridShape,
        lat_deg: Vec<f32>,
        lon_deg: Vec<f32>,
    }
    let encoded = bincode::serialize(&UncheckedGrid {
        shape: GridShape::new(2, 2).unwrap(),
        lat_deg: vec![0.0],
        lon_deg: vec![0.0],
    })
    .unwrap();
    assert!(bincode::deserialize::<LatLonGrid>(&encoded).is_err());
}

#[test]
fn dense_volume_products_are_checked_and_bounded() {
    assert_eq!(checked_volume_elements(37, 1_000).unwrap(), 37_000);
    let max_37_level_cells = MAX_VOLUME_ELEMENTS / 37;
    assert_eq!(
        checked_volume_elements(37, max_37_level_cells).unwrap(),
        37 * max_37_level_cells
    );
    assert!(matches!(
        checked_volume_elements(37, max_37_level_cells + 1),
        Err(RustwxError::VolumeTooLarge {
            levels: 37,
            cells,
            max_elements: MAX_VOLUME_ELEMENTS,
        }) if cells == max_37_level_cells + 1
    ));
    assert!(matches!(
        checked_volume_elements(usize::MAX, 2),
        Err(RustwxError::VolumeTooLarge { .. })
    ));
    assert!(matches!(
        checked_volume_elements(6, MAX_GRID_CELLS),
        Err(RustwxError::VolumeTooLarge {
            levels: 6,
            cells: MAX_GRID_CELLS,
            max_elements: MAX_VOLUME_ELEMENTS
        })
    ));
}

#[test]
fn model_id_aliases_round_trip() {
    assert_eq!("rrfs_a".parse::<ModelId>().unwrap(), ModelId::RrfsA);
    assert_eq!(
        "rrfs_public".parse::<ModelId>().unwrap(),
        ModelId::RrfsPublic
    );
    assert_eq!("rrfs_ensemble".parse::<ModelId>().unwrap(), ModelId::Refs);
    assert_eq!("hybrid_gefs".parse::<ModelId>().unwrap(), ModelId::Hgefs);
    assert_eq!("firewx".parse::<ModelId>().unwrap(), ModelId::RrfsFireWx);
    assert_eq!("ecmwf".parse::<ModelId>().unwrap(), ModelId::EcmwfOpenData);
    assert_eq!("euro".parse::<ModelId>().unwrap(), ModelId::EcmwfOpenData);
    assert_eq!("aifs-v2".parse::<ModelId>().unwrap(), ModelId::Aifs);
    assert_eq!("wrf".parse::<ModelId>().unwrap(), ModelId::WrfGdex);
    assert_eq!(ModelId::Hrrr.to_string(), "hrrr");
    assert_eq!(ModelId::Hgefs.to_string(), "hgefs");
    assert_eq!(ModelId::RrfsFireWx.to_string(), "rrfs-firewx");
    assert_eq!(ModelId::RrfsPublic.to_string(), "rrfs-public");
    assert_eq!(ModelId::Refs.to_string(), "refs");
    assert_eq!(ModelId::WrfGdex.to_string(), "wrf");
    assert_eq!("wrf-gdex".parse::<ModelId>().unwrap(), ModelId::WrfGdex);
    assert_eq!("gdex".parse::<SourceId>().unwrap(), SourceId::Gdex);
    assert_eq!(SourceId::Gdex.to_string(), "gdex");
    assert_eq!(
        "aifsv2-inference".parse::<SourceId>().unwrap(),
        SourceId::AifsInference
    );
    assert_eq!(SourceId::AifsInference.to_string(), "aifs-inference");
}

#[test]
fn cycle_spec_validates_inputs() {
    assert!(CycleSpec::new("20260414", 20).is_ok());
    assert!(CycleSpec::new("20240229", 20).is_ok());
    assert!(matches!(
        CycleSpec::new("20260229", 20),
        Err(RustwxError::InvalidCycleDate(_))
    ));
    assert!(matches!(
        CycleSpec::new("2026-04-14", 20),
        Err(RustwxError::InvalidCycleDate(_))
    ));
    assert!(matches!(
        CycleSpec::new("20260414", 24),
        Err(RustwxError::InvalidCycleHour(24))
    ));
}

#[test]
fn product_key_helpers_expose_name() {
    let key = ProductKey::named("cape_sfc");
    assert_eq!(key.as_named(), Some("cape_sfc"));
    assert_eq!(key.to_string(), "cape_sfc");
}

#[test]
fn timestamp_validates_basic_utc_format() {
    assert!(TimeStamp::new("2026-04-15T00:00:00Z").is_ok());
    assert!(matches!(
        TimeStamp::new("2026-04-15 00:00:00Z"),
        Err(RustwxError::InvalidTimeStamp(_))
    ));
    assert!(matches!(
        TimeStamp::new("2026-02-29T00:00:00Z"),
        Err(RustwxError::InvalidTimeStamp(_))
    ));
}

#[test]
fn field_selector_builds_keys_and_units() {
    let hybrid_pressure = FieldSelector::hybrid_level(CanonicalField::Pressure, 17);
    assert_eq!(hybrid_pressure.key(), "pressure_hybrid_level_17");
    assert_eq!(hybrid_pressure.native_units(), "Pa");

    let selector = FieldSelector::isobaric(CanonicalField::Temperature, 500);
    assert_eq!(selector.to_string(), "temperature@500hpa");
    assert_eq!(selector.key(), "temperature_500hpa");
    assert_eq!(
        selector.product_key().as_named(),
        Some("temperature_500hpa")
    );

    let temp_700 = FieldSelector::isobaric(CanonicalField::Temperature, 700);
    assert_eq!(temp_700.key(), "temperature_700hpa");

    let rh_700 = FieldSelector::isobaric(CanonicalField::RelativeHumidity, 700);
    assert_eq!(rh_700.key(), "relative_humidity_700hpa");
    assert_eq!(rh_700.native_units(), "%");

    let dewpoint_850 = FieldSelector::isobaric(CanonicalField::Dewpoint, 850);
    assert_eq!(dewpoint_850.key(), "dewpoint_850hpa");
    assert_eq!(dewpoint_850.native_units(), "K");

    let temp_2m = FieldSelector::height_agl(CanonicalField::Temperature, 2);
    assert_eq!(temp_2m.key(), "temperature_2m_agl");
    assert_eq!(temp_2m.native_units(), "K");
    let temp_2m_p50 = temp_2m.with_percentile(50);
    assert_eq!(temp_2m_p50.key(), "temperature_2m_agl_p50");
    assert_eq!(temp_2m_p50.to_string(), "temperature@2m_agl:p50");
    let temp_2m_prob = temp_2m.with_probability(ProbabilitySelection::below_milli(273_000));
    assert_eq!(temp_2m_prob.native_units(), "%");

    let dewpoint_2m = FieldSelector::height_agl(CanonicalField::Dewpoint, 2);
    assert_eq!(dewpoint_2m.key(), "dewpoint_2m_agl");
    assert_eq!(dewpoint_2m.native_units(), "K");

    let rh_2m = FieldSelector::height_agl(CanonicalField::RelativeHumidity, 2);
    assert_eq!(rh_2m.key(), "relative_humidity_2m_agl");
    assert_eq!(rh_2m.native_units(), "%");

    let wind_10m = FieldSelector::height_agl(CanonicalField::UWind, 10);
    assert_eq!(wind_10m.key(), "u_wind_10m_agl");
    assert_eq!(wind_10m.native_units(), "m/s");

    let wind_speed_10m = FieldSelector::height_agl(CanonicalField::WindSpeed, 10);
    assert_eq!(wind_speed_10m.key(), "wind_speed_10m_agl");
    assert_eq!(wind_speed_10m.native_units(), "m/s");

    let wind_gust_10m = FieldSelector::height_agl(CanonicalField::WindGust, 10);
    assert_eq!(wind_gust_10m.key(), "wind_gust_10m_agl");
    assert_eq!(wind_gust_10m.native_units(), "m/s");

    let mslp = FieldSelector::mean_sea_level(CanonicalField::PressureReducedToMeanSeaLevel);
    assert_eq!(
        mslp.key(),
        "pressure_reduced_to_mean_sea_level_mean_sea_level"
    );
    assert_eq!(mslp.native_units(), "Pa");

    let absolute_vorticity_500 = FieldSelector::isobaric(CanonicalField::AbsoluteVorticity, 500);
    assert_eq!(absolute_vorticity_500.key(), "absolute_vorticity_500hpa");
    assert_eq!(absolute_vorticity_500.native_units(), "s^-1");

    let relative_vorticity_500 = FieldSelector::isobaric(CanonicalField::RelativeVorticity, 500);
    assert_eq!(relative_vorticity_500.key(), "relative_vorticity_500hpa");
    assert_eq!(relative_vorticity_500.native_units(), "s^-1");

    let reflectivity = FieldSelector::entire_atmosphere(CanonicalField::CompositeReflectivity);
    assert_eq!(
        reflectivity.key(),
        "composite_reflectivity_entire_atmosphere"
    );

    let reflectivity_1km = FieldSelector::height_agl(CanonicalField::RadarReflectivity, 1000);
    assert_eq!(reflectivity_1km.key(), "radar_reflectivity_1000m_agl");
    assert_eq!(reflectivity_1km.native_units(), "dBZ");

    let pwat = FieldSelector::entire_atmosphere(CanonicalField::PrecipitableWater);
    assert_eq!(pwat.key(), "precipitable_water_entire_atmosphere");
    assert_eq!(pwat.native_units(), "kg/m^2");

    let cloud_cover = FieldSelector::entire_atmosphere(CanonicalField::TotalCloudCover);
    assert_eq!(cloud_cover.key(), "total_cloud_cover_entire_atmosphere");
    assert_eq!(cloud_cover.native_units(), "%");

    let simulated_ir =
        FieldSelector::nominal_top(CanonicalField::SimulatedInfraredBrightnessTemperature);
    assert_eq!(
        simulated_ir.key(),
        "simulated_infrared_brightness_temperature_nominal_top"
    );
    assert_eq!(simulated_ir.native_units(), "K");

    let visibility = FieldSelector::surface(CanonicalField::Visibility);
    assert_eq!(visibility.key(), "visibility_surface");
    assert_eq!(visibility.native_units(), "m");

    let lsm = FieldSelector::surface(CanonicalField::LandSeaMask);
    assert_eq!(lsm.key(), "land_sea_mask_surface");
    assert_eq!(lsm.native_units(), "fraction");

    let lightning = FieldSelector::height_agl(CanonicalField::LightningFlashDensity, 2);
    assert_eq!(lightning.key(), "lightning_flash_density_2m_agl");
    assert_eq!(lightning.native_units(), "km^-2 day^-1");

    let uh = FieldSelector::height_layer_agl(CanonicalField::UpdraftHelicity, 2000, 5000);
    assert_eq!(uh.key(), "updraft_helicity_2000m_to_5000m_agl");

    let smoke_8m = FieldSelector::height_agl(CanonicalField::SmokeMassDensity, 8);
    assert_eq!(smoke_8m.key(), "smoke_mass_density_8m_agl");
    assert_eq!(smoke_8m.native_units(), "kg/m^3");

    let smoke_hybrid = FieldSelector::hybrid_level(CanonicalField::SmokeMassDensity, 50);
    assert_eq!(smoke_hybrid.key(), "smoke_mass_density_hybrid_level_50");
    assert_eq!(smoke_hybrid.native_units(), "kg/m^3");

    let smoke_column = FieldSelector::entire_atmosphere(CanonicalField::ColumnIntegratedSmoke);
    assert_eq!(
        smoke_column.key(),
        "column_integrated_smoke_entire_atmosphere"
    );
    assert_eq!(smoke_column.native_units(), "kg/m^2");
}

#[test]
fn model_timestep_builds_requests_and_descriptors() {
    let timestep = ModelTimestep::with_source(
        ModelId::RrfsA,
        CycleSpec::new("20260414", 18).unwrap(),
        6,
        TimeStamp::new("2026-04-15T00:00:00Z").unwrap(),
        Some(SourceId::Aws),
    )
    .unwrap();

    let request = timestep.request("prs-conus").unwrap();
    assert_eq!(request.model, ModelId::RrfsA);
    assert_eq!(request.forecast_hour, 6);
    assert_eq!(request.product, "prs-conus");
    assert_eq!(timestep.descriptor().cycle.date_yyyymmdd, "20260414");
    assert_eq!(timestep.descriptor().cycle.hour_utc, 18);
    assert_eq!(
        timestep.descriptor().valid_time.as_str(),
        "2026-04-15T00:00:00Z"
    );
    assert_eq!(timestep.source, Some(SourceId::Aws));
}

#[test]
fn canonical_bundle_descriptors_are_typed() {
    assert_eq!(
        CanonicalBundleDescriptor::SurfaceAnalysis.as_str(),
        "surface_analysis"
    );
    assert_eq!(
        CanonicalBundleDescriptor::SurfaceAnalysis.family(),
        CanonicalDataFamily::Surface
    );
    assert_eq!(
        CanonicalBundleDescriptor::PressureAnalysis.as_str(),
        "pressure_analysis"
    );
    assert_eq!(
        CanonicalBundleDescriptor::PressureAnalysis.family(),
        CanonicalDataFamily::Pressure
    );
    assert_eq!(
        CanonicalBundleDescriptor::NativeAnalysis.as_str(),
        "native_analysis"
    );
    assert_eq!(
        CanonicalBundleDescriptor::NativeAnalysis.family(),
        CanonicalDataFamily::Native
    );
}

#[test]
fn canonical_bundle_id_dedupes_by_full_identity() {
    let cycle = CycleSpec::new("20260415", 18).unwrap();
    let surface = CanonicalBundleId::new(
        ModelId::Hrrr,
        cycle.clone(),
        6,
        SourceId::Aws,
        CanonicalBundleDescriptor::SurfaceAnalysis,
        "sfc",
    );
    let surface_clone = CanonicalBundleId::new(
        ModelId::Hrrr,
        cycle.clone(),
        6,
        SourceId::Aws,
        CanonicalBundleDescriptor::SurfaceAnalysis,
        "sfc",
    );
    let surface_other_hour = CanonicalBundleId::new(
        ModelId::Hrrr,
        cycle.clone(),
        7,
        SourceId::Aws,
        CanonicalBundleDescriptor::SurfaceAnalysis,
        "sfc",
    );
    assert_eq!(surface, surface_clone);
    assert_ne!(surface, surface_other_hour);
    assert_eq!(surface.family(), CanonicalDataFamily::Surface);
    assert!(
        surface
            .to_string()
            .contains("surface_analysis@hrrr:2026041518z+f006:sfc")
    );
}

#[test]
fn bundle_requirement_carries_native_override() {
    let plain = BundleRequirement::new(CanonicalBundleDescriptor::PressureAnalysis, 12);
    assert!(plain.native_override.is_none());
    let overridden = plain.clone().with_native_override("prs-na");
    assert_eq!(overridden.native_override.as_deref(), Some("prs-na"));
    assert_ne!(plain, overridden);
}

#[test]
fn resolved_url_prefers_idx_when_probing_availability() {
    let with_idx = ResolvedUrl {
        source: SourceId::Aws,
        grib_url: "https://example.test/file.grib2".to_string(),
        idx_url: Some("https://example.test/file.grib2.idx".to_string()),
    };
    assert_eq!(
        with_idx.availability_probe_url(),
        "https://example.test/file.grib2.idx"
    );

    let without_idx = ResolvedUrl {
        source: SourceId::Azure,
        grib_url: "https://example.test/file.grib2".to_string(),
        idx_url: None,
    };
    assert_eq!(
        without_idx.availability_probe_url(),
        "https://example.test/file.grib2"
    );
}

#[test]
fn model_field_2d_round_trips_to_legacy_field() {
    let shape = GridShape::new(2, 2).unwrap();
    let grid = LatLonGrid::new(
        shape,
        vec![35.0, 35.0, 36.0, 36.0],
        vec![-99.0, -98.0, -99.0, -98.0],
    )
    .unwrap();
    let metadata = ModelFieldMetadata::new(
        ModelTimestep::new(
            ModelId::Hrrr,
            CycleSpec::new("20260414", 18).unwrap(),
            1,
            TimeStamp::new("2026-04-14T19:00:00Z").unwrap(),
        )
        .unwrap(),
        ProductKey::named("sbcape"),
        "J/kg",
    )
    .with_product_metadata(ProductKeyMetadata::new("Surface-Based CAPE"));

    let field =
        ModelField2D::new(metadata.clone(), grid.clone(), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let legacy: Field2D = field.into();

    assert_eq!(legacy.product, metadata.product);
    assert_eq!(legacy.units, "J/kg");
    assert_eq!(legacy.grid, grid);
    assert_eq!(legacy.values, vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(
        metadata.product_metadata.unwrap().display_name,
        "Surface-Based CAPE"
    );
}

#[test]
fn selected_field_2d_round_trips_to_legacy_field() {
    let shape = GridShape::new(2, 1).unwrap();
    let grid = LatLonGrid::new(shape, vec![35.0, 35.0], vec![-99.0, -98.0]).unwrap();
    let selector = FieldSelector::isobaric(CanonicalField::GeopotentialHeight, 500);

    let selected = SelectedField2D::new(selector, "gpm", grid.clone(), vec![5700.0, 5712.0])
        .unwrap()
        .with_projection(GridProjection::LambertConformal {
            standard_parallel_1_deg: 38.5,
            standard_parallel_2_deg: 38.5,
            central_meridian_deg: -97.5,
        });
    let legacy: Field2D = selected.into();

    assert_eq!(
        legacy.product.as_named(),
        Some("geopotential_height_500hpa")
    );
    assert_eq!(legacy.units, "gpm");
    assert_eq!(legacy.grid, grid);
    assert_eq!(legacy.values, vec![5700.0, 5712.0]);
}

#[test]
fn selected_field_keeps_projection_metadata() {
    let shape = GridShape::new(2, 1).unwrap();
    let grid = LatLonGrid::new(shape, vec![35.0, 35.0], vec![-99.0, -98.0]).unwrap();
    let selector = FieldSelector::surface(CanonicalField::Temperature);

    let selected = SelectedField2D::new(selector, "K", grid, vec![290.0, 291.0])
        .unwrap()
        .with_projection(GridProjection::Mercator {
            latitude_of_true_scale_deg: 20.0,
            central_meridian_deg: -95.0,
        });

    assert_eq!(
        selected.projection,
        Some(GridProjection::Mercator {
            latitude_of_true_scale_deg: 20.0,
            central_meridian_deg: -95.0,
        })
    );
}

#[test]
fn selected_hybrid_level_volume_tracks_levels_slices_and_projection() {
    let shape = GridShape::new(2, 1).unwrap();
    let grid = LatLonGrid::new(shape, vec![35.0, 35.0], vec![-99.0, -98.0]).unwrap();
    let volume = SelectedHybridLevelVolume::new(
        CanonicalField::SmokeMassDensity,
        vec![1, 2],
        "kg/m^3",
        grid,
        vec![1.0, 2.0, 3.0, 4.0],
    )
    .unwrap()
    .with_projection(GridProjection::LambertConformal {
        standard_parallel_1_deg: 38.5,
        standard_parallel_2_deg: 38.5,
        central_meridian_deg: -97.5,
    });

    assert_eq!(volume.level_count(), 2);
    assert_eq!(volume.level_slice(0), Some(&[1.0, 2.0][..]));
    assert_eq!(volume.level_slice(1), Some(&[3.0, 4.0][..]));
    assert_eq!(
        volume.selector_at(1),
        Some(FieldSelector::hybrid_level(
            CanonicalField::SmokeMassDensity,
            2
        ))
    );
    assert_eq!(
        volume.projection,
        Some(GridProjection::LambertConformal {
            standard_parallel_1_deg: 38.5,
            standard_parallel_2_deg: 38.5,
            central_meridian_deg: -97.5,
        })
    );
}

#[test]
fn grid_projection_bincode_round_trip_works() {
    let projection = GridProjection::PolarStereographic {
        true_latitude_deg: 60.0,
        central_meridian_deg: -105.0,
        south_pole_on_projection_plane: false,
    };

    let bytes = bincode::serialize(&projection).unwrap();
    let round_trip = bincode::deserialize::<GridProjection>(&bytes).unwrap();

    assert_eq!(round_trip, projection);
}

#[test]
fn selector_product_metadata_carries_typed_provenance() {
    let selector = FieldSelector::isobaric(CanonicalField::Temperature, 500);
    let metadata = selector.product_metadata();

    assert_eq!(metadata.display_name, "Temperature (500hpa)");
    assert_eq!(metadata.native_units.as_deref(), Some("K"));
    assert_eq!(
        metadata
            .identity
            .as_ref()
            .expect("selector metadata should expose canonical identity")
            .canonical,
        ProductId::new(ProductKind::Direct, "temperature_500hpa")
    );
    let provenance = metadata
        .provenance
        .as_ref()
        .expect("selector metadata should carry provenance");
    assert_eq!(provenance.lineage, ProductLineage::Direct);
    assert_eq!(provenance.maturity, ProductMaturity::Operational);
    assert_eq!(provenance.selector, Some(selector));
    assert!(provenance.flags.is_empty());
    assert!(provenance.window.is_none());
}

#[test]
fn product_key_metadata_builder_keeps_additive_provenance_fields() {
    let metadata = ProductKeyMetadata::new("Run-Max UH")
        .with_description("Trailing native hourly 2-5 km updraft-helicity maxima")
        .with_category("windowed")
        .with_native_units("m^2/s^2")
        .with_identity(
            CanonicalProductIdentity::new(ProductId::new(
                ProductKind::Windowed,
                "uh_2to5km_run_max",
            ))
            .with_alias_slug("run_max_uh_2to5km"),
        )
        .with_provenance(
            ProductProvenance::new(ProductLineage::Windowed, ProductMaturity::Operational)
                .with_flag(ProductSemanticFlag::Composite)
                .with_window(ProductWindowSpec::accumulation(Some(3))),
        );

    assert_eq!(metadata.category.as_deref(), Some("windowed"));
    let identity = metadata
        .identity
        .clone()
        .expect("builder should keep canonical identity");
    assert_eq!(
        identity.canonical,
        ProductId::new(ProductKind::Windowed, "uh_2to5km_run_max")
    );
    assert!(
        identity
            .alias_slugs
            .contains(&"run_max_uh_2to5km".to_string())
    );
    let provenance = metadata.provenance.expect("builder should keep provenance");
    assert_eq!(provenance.lineage, ProductLineage::Windowed);
    assert!(provenance.flags.contains(&ProductSemanticFlag::Composite));
    assert_eq!(
        provenance.window,
        Some(ProductWindowSpec::accumulation(Some(3)))
    );
}

#[test]
fn pressure_level_volume_exposes_level_slices() {
    let shape = GridShape::new(2, 2).unwrap();
    let grid = LatLonGrid::new(
        shape,
        vec![35.0, 35.0, 36.0, 36.0],
        vec![-99.0, -98.0, -99.0, -98.0],
    )
    .unwrap();
    let metadata = ModelFieldMetadata::new(
        ModelTimestep::new(
            ModelId::Gfs,
            CycleSpec::new("20260414", 12).unwrap(),
            9,
            TimeStamp::new("2026-04-14T21:00:00Z").unwrap(),
        )
        .unwrap(),
        ProductKey::named("temperature"),
        "degC",
    );

    let volume = PressureLevelVolume::new(
        metadata.clone(),
        vec![850.0, 700.0],
        grid.clone(),
        vec![1.0, 2.0, 3.0, 4.0, -5.0, -4.0, -3.0, -2.0],
    )
    .unwrap();

    assert_eq!(volume.level_count(), 2);
    assert_eq!(volume.level_slice(0), Some(&[1.0, 2.0, 3.0, 4.0][..]));
    assert_eq!(volume.level_slice(1), Some(&[-5.0, -4.0, -3.0, -2.0][..]));

    let legacy: Field3D = volume.into();
    assert_eq!(legacy.product, metadata.product);
    assert_eq!(legacy.units, "degC");
    assert_eq!(legacy.levels, vec![850.0, 700.0]);
    assert_eq!(legacy.grid, grid);
}

#[test]
fn pressure_level_volume_validates_levels_and_lengths() {
    let shape = GridShape::new(2, 1).unwrap();
    let grid = LatLonGrid::new(shape, vec![35.0, 35.0], vec![-99.0, -98.0]).unwrap();
    let metadata = ModelFieldMetadata::new(
        ModelTimestep::new(
            ModelId::EcmwfOpenData,
            CycleSpec::new("20260414", 0).unwrap(),
            12,
            TimeStamp::new("2026-04-14T12:00:00Z").unwrap(),
        )
        .unwrap(),
        ProductKey::named("rh"),
        "%",
    );

    assert!(matches!(
        PressureLevelVolume::new(metadata.clone(), Vec::new(), grid.clone(), vec![1.0, 2.0]),
        Err(RustwxError::EmptyPressureLevels)
    ));
    assert!(matches!(
        PressureLevelVolume::new(
            metadata.clone(),
            vec![850.0, -700.0],
            grid.clone(),
            vec![1.0, 2.0, 3.0, 4.0],
        ),
        Err(RustwxError::InvalidPressureLevel {
            index: 1,
            value: -700.0
        })
    ));
    assert!(matches!(
        PressureLevelVolume::new(metadata, vec![850.0, 700.0], grid, vec![1.0, 2.0, 3.0]),
        Err(RustwxError::InvalidFieldDataLength {
            expected: 4,
            actual: 3
        })
    ));
}

#[test]
fn hybrid_level_volume_validates_levels_and_lengths() {
    let shape = GridShape::new(2, 1).unwrap();
    let grid = LatLonGrid::new(shape, vec![35.0, 35.0], vec![-99.0, -98.0]).unwrap();

    assert!(matches!(
        SelectedHybridLevelVolume::new(
            CanonicalField::SmokeMassDensity,
            Vec::new(),
            "kg/m^3",
            grid.clone(),
            vec![1.0, 2.0],
        ),
        Err(RustwxError::EmptyHybridLevels)
    ));
    assert!(matches!(
        SelectedHybridLevelVolume::new(
            CanonicalField::SmokeMassDensity,
            vec![1, 0],
            "kg/m^3",
            grid.clone(),
            vec![1.0, 2.0, 3.0, 4.0],
        ),
        Err(RustwxError::InvalidHybridLevel { index: 1, value: 0 })
    ));
    assert!(matches!(
        SelectedHybridLevelVolume::new(
            CanonicalField::Pressure,
            vec![1, 2],
            "Pa",
            grid,
            vec![1.0, 2.0, 3.0],
        ),
        Err(RustwxError::InvalidFieldDataLength {
            expected: 4,
            actual: 3
        })
    ));
}

#[test]
fn field_point_sampling_uses_nearest_and_inverse_distance_modes() {
    let grid = LatLonGrid::new(
        GridShape::new(2, 2).unwrap(),
        vec![0.0, 0.0, 1.0, 1.0],
        vec![0.0, 1.0, 0.0, 1.0],
    )
    .unwrap();
    let field = Field2D::new(
        ProductKey::named("sample"),
        "unitless",
        grid,
        vec![0.0, 10.0, 20.0, 30.0],
    )
    .unwrap();

    let nearest = field.sample_point(GeoPoint::new(0.95, 0.95), FieldPointSampleMethod::Nearest);
    assert_eq!(nearest.value, Some(30.0));
    assert_eq!(nearest.contributors.len(), 1);
    assert_eq!(nearest.contributors[0].grid_index, 3);

    let blended = field.sample_point(
        GeoPoint::new(0.5, 0.5),
        FieldPointSampleMethod::InverseDistance4,
    );
    assert_eq!(blended.contributors.len(), 4);
    assert!((blended.value.expect("blended value") - 15.0).abs() < 1.0e-5);
    let total_weight = blended
        .contributors
        .iter()
        .map(|entry| entry.weight)
        .sum::<f64>();
    assert!((total_weight - 1.0).abs() < 1.0e-9);
}

#[test]
fn field_polygon_summary_counts_finite_cells_inside_polygon() {
    let grid = LatLonGrid::new(
        GridShape::new(3, 2).unwrap(),
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
    )
    .unwrap();
    let field = Field2D::new(
        ProductKey::named("sample"),
        "unitless",
        grid,
        vec![1.0, 2.0, f32::NAN, 4.0, 5.0, 6.0],
    )
    .unwrap();
    let polygon = GeoPolygon::new(
        vec![
            GeoPoint::new(-0.5, -0.5),
            GeoPoint::new(-0.5, 1.5),
            GeoPoint::new(1.5, 1.5),
            GeoPoint::new(1.5, -0.5),
        ],
        Vec::new(),
    );

    let summary = field.summarize_polygon(&polygon);
    assert_eq!(summary.included_cell_count, 4);
    assert_eq!(summary.valid_cell_count, 4);
    assert_eq!(summary.missing_cell_count, 0);
    assert_eq!(summary.min, Some(1.0));
    assert_eq!(summary.max, Some(5.0));
    assert_eq!(summary.mean, Some(3.0));
}

#[test]
fn polygon_holes_exclude_cells_from_area_summary() {
    let grid = LatLonGrid::new(
        GridShape::new(3, 1).unwrap(),
        vec![0.0, 0.0, 0.0],
        vec![0.0, 1.0, 2.0],
    )
    .unwrap();
    let field = Field2D::new(
        ProductKey::named("sample"),
        "unitless",
        grid,
        vec![10.0, 20.0, 30.0],
    )
    .unwrap();
    let polygon = GeoPolygon::new(
        vec![
            GeoPoint::new(-1.0, -1.0),
            GeoPoint::new(-1.0, 3.0),
            GeoPoint::new(1.0, 3.0),
            GeoPoint::new(1.0, -1.0),
        ],
        vec![vec![
            GeoPoint::new(-0.5, 0.5),
            GeoPoint::new(-0.5, 1.5),
            GeoPoint::new(0.5, 1.5),
            GeoPoint::new(0.5, 0.5),
        ]],
    );

    let summary = field.summarize_polygon(&polygon);
    assert_eq!(summary.included_cell_count, 2);
    assert_eq!(summary.valid_cell_count, 2);
    assert_eq!(summary.mean, Some(20.0));
}
