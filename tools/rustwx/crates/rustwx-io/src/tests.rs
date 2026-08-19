use super::*;
use grib_core::grib2::{DataRepresentation, GridDefinition, ProductDefinition};
use std::path::PathBuf;

const SAMPLE_IDX: &str = "\
1:0:d=2026041420:TMP:2 m above ground:anl:
2:47843:d=2026041420:SPFH:2 m above ground:anl:
3:96542:d=2026041420:CAPE:surface:anl:
4:143210:d=2026041420:UGRD:10 m above ground:anl:
5:200000:d=2026041420:VGRD:10 m above ground:anl:
";

#[test]
fn gzip_grib_payloads_are_decompressed_before_decode() {
    let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    encoder.write_all(b"GRIBtest").unwrap();
    let compressed = encoder.finish().unwrap();

    let decoded = maybe_decompress_grib_payload(
        "https://example.invalid/MRMS_ReflectivityAtLowestAltitude.latest.grib2.gz",
        compressed,
    )
    .expect("gzip payload decodes");

    assert_eq!(decoded, b"GRIBtest");
}

#[test]
fn plain_grib_payloads_are_left_unchanged() {
    let decoded = maybe_decompress_grib_payload(
        "https://example.invalid/hrrr.t00z.wrfsfcf006.grib2",
        b"GRIBtest".to_vec(),
    )
    .expect("plain payload passes through");

    assert_eq!(decoded, b"GRIBtest");
}

#[test]
fn mrms_latest_product_url_rejects_pathlike_tokens() {
    assert_eq!(
        mrms_latest_product_url("ReflectivityAtLowestAltitude").unwrap(),
        "https://mrms.ncep.noaa.gov/2D/ReflectivityAtLowestAltitude/MRMS_ReflectivityAtLowestAltitude.latest.grib2.gz"
    );
    assert!(mrms_latest_product_url("../ReflectivityAtLowestAltitude").is_err());
}

#[test]
fn mrms_lowest_altitude_reflectivity_maps_to_500m_msl_selector() {
    let message = ieee_f32_message(
        ParameterCode {
            discipline: 209,
            category: 3,
            number: 57,
        },
        102,
        500.0,
        &[10.0, 20.0],
        -130.0,
        -129.0,
    );
    let grib = Grib2File {
        messages: vec![message],
    };

    let field = extract_field_from_grib2(
        &grib,
        FieldSelector::altitude_msl(CanonicalField::RadarReflectivity, 500),
    )
    .expect("MRMS lowest-altitude reflectivity selector matches");

    assert_eq!(field.values, vec![10.0, 20.0]);
}

#[test]
fn mrms_composite_reflectivity_maps_to_composite_selector() {
    let message = ieee_f32_message(
        ParameterCode {
            discipline: 209,
            category: 10,
            number: 0,
        },
        102,
        500.0,
        &[30.0, 40.0],
        -130.0,
        -129.0,
    );
    let grib = Grib2File {
        messages: vec![message],
    };

    let field = extract_field_from_grib2(
        &grib,
        FieldSelector::altitude_msl(CanonicalField::CompositeReflectivity, 500),
    )
    .expect("MRMS composite reflectivity selector matches");

    assert_eq!(field.values, vec![30.0, 40.0]);
}

#[test]
fn eumetnet_opera_dbzh_coverage_url_encodes_datetime_range() {
    let url = eumetnet_opera_dbzh_coverage_url("2026-06-27T05:00Z/2026-06-27T05:30Z")
        .expect("range encodes");
    assert!(url.contains("datetime=2026-06-27T05%3A00Z%2F2026-06-27T05%3A30Z"));
    assert!(url.contains("standard_name=DBZH"));
    assert!(eumetnet_opera_dbzh_coverage_url("2026-06-27T05:00Z\\bad").is_err());
}

#[test]
fn eumetnet_opera_coverage_json_extracts_odim_links_and_meta() {
    let json = br#"{
        "type": "Coverage",
        "links": [
            {"href":"https://eumetnet.eu/","type":"text/html","title":"Website"},
            {"href":"https://s3.waw3-1.cloudferro.com/openradar-24h/2026/06/27/OPERA/COMP/OPERA@20260627T0530@0@DBZH.h5","rel":"items","type":"application/x-odim","title":"Data download link.","length":1940764},
            {"href":"https://s3.waw3-1.cloudferro.com/openradar-24h/2026/06/27/OPERA/COMP/OPERA@20260627T0535@0@DBZH.h5","rel":"items","type":"application/x-odim","title":"Data download link.","length":1953411}
        ],
        "metocean:radar_meta": {
            "projdef":"+proj=laea +lat_0=55.0 +lon_0=10.0 +x_0=1950000.0 +y_0=-2100000.0 +units=m +ellps=WGS84",
            "xsize":3800,
            "ysize":4400,
            "xscale":1000.0,
            "yscale":1000.0,
            "LL_lon":-10.4345768386404,
            "LL_lat":31.7462153182675,
            "UL_lon":-39.5357864125034,
            "UL_lat":67.0228327624372,
            "UR_lon":57.8119647501499,
            "UR_lat":67.6210371071631,
            "LR_lon":29.421038635578,
            "LR_lat":31.987650276733
        }
    }"#;

    let coverage = parse_eumetnet_opera_dbzh_coverage_json(json).expect("coverage parses");

    assert_eq!(coverage.download_links.len(), 2);
    assert_eq!(
        coverage.latest_odim_link().map(|link| link.length),
        Some(Some(1_953_411))
    );
    let meta = coverage.radar_meta.expect("radar metadata exists");
    assert_eq!(meta.xsize, 3800);
    assert_eq!(meta.ysize, 4400);
    assert_eq!(meta.xscale_m, 1000.0);
}

#[test]
fn opera_spherical_laea_inverse_is_close_to_documented_corners() {
    let lat0 = 55.0_f64.to_radians();
    let lon0 = 10.0_f64.to_radians();
    let false_easting = 1_950_000.0;
    let false_northing = -2_100_000.0;

    let (ll_lat, ll_lon) =
        inverse_spherical_laea(0.0, -4_400_000.0, lat0, lon0, false_easting, false_northing);
    let (ul_lat, ul_lon) =
        inverse_spherical_laea(0.0, 0.0, lat0, lon0, false_easting, false_northing);
    let (ur_lat, ur_lon) =
        inverse_spherical_laea(3_800_000.0, 0.0, lat0, lon0, false_easting, false_northing);

    assert!((ll_lat - 31.7462).abs() < 0.05, "{ll_lat}");
    assert!((ll_lon - -10.4346).abs() < 0.25, "{ll_lon}");
    assert!((ul_lat - 67.0228).abs() < 0.05, "{ul_lat}");
    assert!((ul_lon - -39.5358).abs() < 0.25, "{ul_lon}");
    assert!((ur_lat - 67.6210).abs() < 0.05, "{ur_lat}");
    assert!((ur_lon - 57.8120).abs() < 0.25, "{ur_lon}");
}

#[test]
#[ignore = "network smoke test against live MRMS realtime feed"]
fn live_mrms_latest_reflectivity_gzip_parses_grib2() {
    let url = "https://mrms.ncep.noaa.gov/2D/ReflectivityAtLowestAltitude/MRMS_ReflectivityAtLowestAltitude.latest.grib2.gz";
    let compressed = client()
        .expect("download client")
        .get_bytes(url)
        .expect("MRMS latest reflectivity downloads");
    let bytes = maybe_decompress_grib_payload(url, compressed).expect("MRMS gzip decompresses");
    let grib = Grib2File::from_bytes(&bytes).expect("MRMS payload parses as GRIB2");
    assert_eq!(grib.messages.len(), 1, "MRMS reflectivity is one field");
    let message = &grib.messages[0];
    eprintln!(
        "MRMS latest: discipline={} category={} parameter={} level_type={} level_value={} grid={}x{} template={}",
        message.discipline,
        message.product.parameter_category,
        message.product.parameter_number,
        message.product.level_type,
        message.product.level_value,
        message.grid.nx,
        message.grid.ny,
        message.grid.template
    );

    assert_eq!(message.discipline, 209);
    assert_eq!(message.product.parameter_category, 3);
    assert_eq!(message.product.parameter_number, 57);
}

#[test]
#[ignore = "network smoke test against live MRMS realtime feed"]
fn live_mrms_latest_reflectivity_helpers_extract_fields() {
    let lowest = retry_live(|| extract_mrms_latest_reflectivity_at_lowest_altitude())
        .expect("MRMS lowest-altitude reflectivity extracts");
    eprintln!(
        "MRMS lowest-altitude: {} values on {}x{}",
        lowest.values.len(),
        lowest.grid.shape.nx,
        lowest.grid.shape.ny
    );
    assert_eq!(
        lowest.selector,
        FieldSelector::altitude_msl(CanonicalField::RadarReflectivity, 500)
    );

    let composite_bytes = retry_live(|| fetch_mrms_latest_product("MergedReflectivityQCComposite"))
        .expect("MRMS composite reflectivity downloads");
    let composite_grib =
        Grib2File::from_bytes(&composite_bytes).expect("MRMS composite parses as GRIB2");
    let composite_message = &composite_grib.messages[0];
    eprintln!(
        "MRMS composite metadata: discipline={} category={} parameter={} level_type={} level_value={} grid={}x{} template={}",
        composite_message.discipline,
        composite_message.product.parameter_category,
        composite_message.product.parameter_number,
        composite_message.product.level_type,
        composite_message.product.level_value,
        composite_message.grid.nx,
        composite_message.grid.ny,
        composite_message.grid.template
    );

    let composite = retry_live(|| extract_mrms_latest_composite_reflectivity())
        .expect("MRMS composite reflectivity extracts");
    eprintln!(
        "MRMS composite: {} values on {}x{}",
        composite.values.len(),
        composite.grid.shape.nx,
        composite.grid.shape.ny
    );
    assert_eq!(
        composite.selector,
        FieldSelector::altitude_msl(CanonicalField::CompositeReflectivity, 500)
    );
}

#[test]
#[ignore = "network smoke test against live EUMETNET OPERA ODIM HDF5 feed"]
fn live_eumetnet_opera_latest_dbzh_helpers_extract_field() {
    let now = chrono::Utc::now();
    let start = now - chrono::Duration::minutes(45);
    let end = now - chrono::Duration::minutes(5);
    let range = format!(
        "{}/{}",
        start.format("%Y-%m-%dT%H:%MZ"),
        end.format("%Y-%m-%dT%H:%MZ")
    );
    let coverage =
        retry_live(|| fetch_eumetnet_opera_dbzh_coverage(&range)).expect("OPERA coverage resolves");
    let link = coverage
        .latest_odim_link()
        .expect("OPERA coverage has ODIM HDF5 link");
    eprintln!("OPERA latest link: {}", link.href);
    let bytes =
        retry_live(|| fetch_eumetnet_opera_odim_h5(&link.href)).expect("OPERA HDF5 downloads");
    let field = extract_eumetnet_opera_dbzh_from_odim_h5(&bytes).expect("OPERA HDF5 extracts");
    eprintln!(
        "OPERA DBZH: {} values on {}x{}",
        field.values.len(),
        field.grid.shape.nx,
        field.grid.shape.ny
    );
    assert_eq!(
        field.selector,
        FieldSelector::entire_atmosphere(CanonicalField::CompositeReflectivity)
    );
    assert_eq!(field.units, "dBZ");
    assert_eq!(field.grid.shape.nx, 3800);
    assert_eq!(field.grid.shape.ny, 4400);
    assert!(field.values.iter().any(|value| value.is_finite()));
}

fn retry_live<T>(mut f: impl FnMut() -> Result<T, IoError>) -> Result<T, IoError> {
    let mut last = None;
    for attempt in 1..=3 {
        match f() {
            Ok(value) => return Ok(value),
            Err(err) => {
                eprintln!("live attempt {attempt} failed: {err}");
                last = Some(err);
                std::thread::sleep(std::time::Duration::from_millis(500));
            }
        }
    }
    Err(last.expect("retry loop ran at least once"))
}

#[test]
fn alternating_i_scan_rows_are_normalized_to_row_major_order() {
    let mut values = vec![
        1.0, 2.0, 3.0, 4.0, //
        8.0, 7.0, 6.0, 5.0, //
        9.0, 10.0, 11.0, 12.0,
    ];

    normalize_alternating_i_scan_rows(&mut values, 4, 3, 0x50);

    assert_eq!(
        values,
        vec![
            1.0, 2.0, 3.0, 4.0, //
            5.0, 6.0, 7.0, 8.0, //
            9.0, 10.0, 11.0, 12.0,
        ]
    );
}

#[test]
fn plain_i_scan_rows_are_left_unchanged() {
    let mut values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    normalize_alternating_i_scan_rows(&mut values, 3, 2, 0x40);

    assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

fn ieee_f32_message(
    parameter: ParameterCode,
    level_type: u8,
    level_value: f64,
    values: &[f32],
    lon1: f64,
    lon2: f64,
) -> Grib2Message {
    let raw_data = values
        .iter()
        .flat_map(|value| value.to_be_bytes())
        .collect::<Vec<_>>();
    Grib2Message {
        discipline: parameter.discipline,
        // The converged decoder carries Section 1 as a value object; a
        // synthetic message has no originating centre to name.
        identification: grib_core::grib2::Identification::default(),
        reference_time: chrono::NaiveDate::from_ymd_opt(2026, 4, 14)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap(),
        grid: GridDefinition {
            template: 0,
            nx: values.len() as u32,
            ny: 1,
            lat1: 35.0,
            lon1,
            lat2: 35.0,
            lon2,
            dx: 1.0,
            dy: 0.0,
            scan_mode: 0,
            num_data_points: values.len() as u32,
            ..Default::default()
        },
        product: ProductDefinition {
            template: 0,
            parameter_category: parameter.category,
            parameter_number: parameter.number,
            level_type,
            level_value,
            ..Default::default()
        },
        data_rep: DataRepresentation {
            template: 4,
            bits_per_value: 32,
            section5_num_data_points: values.len() as u32,
            ..Default::default()
        },
        bitmap: None,
        raw_data,
    }
}

#[test]
fn projection_metadata_is_inferred_from_grib_grid_templates() {
    let lambert = GridDefinition {
        template: 30,
        latin1: 38.5,
        latin2: 38.5,
        lov: 262.5,
        ..Default::default()
    };
    assert_eq!(
        grid_projection_from_grib2_grid(&lambert),
        Some(GridProjection::LambertConformal {
            standard_parallel_1_deg: 38.5,
            standard_parallel_2_deg: 38.5,
            central_meridian_deg: -97.5,
        })
    );

    let polar = GridDefinition {
        template: 20,
        lad: 60.0,
        lov: 210.0,
        projection_center_flag: 1,
        ..Default::default()
    };
    assert_eq!(
        grid_projection_from_grib2_grid(&polar),
        Some(GridProjection::PolarStereographic {
            true_latitude_deg: 60.0,
            central_meridian_deg: -150.0,
            south_pole_on_projection_plane: true,
        })
    );
}

fn sample_pressure_subset_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("proof")
        .join("rustwx_hrrr_20260414_22z_f00_prs_subset.grib2")
}

#[test]
fn candidate_hours_match_model_rules() {
    assert_eq!(candidate_hours(ModelId::Hrrr, 20).last().copied(), Some(18));
    assert_eq!(candidate_hours(ModelId::Hrrr, 18).last().copied(), Some(48));
    assert_eq!(
        candidate_hours(ModelId::RrfsA, 20).last().copied(),
        Some(60)
    );
    // ECMWF open-data 00/12z stream reaches f360; 06/18z stops at f144.
    assert_eq!(
        candidate_hours(ModelId::EcmwfOpenData, 0).last().copied(),
        Some(360)
    );
    assert_eq!(
        candidate_hours(ModelId::EcmwfOpenData, 12).last().copied(),
        Some(360)
    );
    assert_eq!(
        candidate_hours(ModelId::EcmwfOpenData, 6).last().copied(),
        Some(144)
    );
    assert_eq!(
        candidate_hours(ModelId::EcmwfOpenData, 18).last().copied(),
        Some(144)
    );
}

#[test]
fn nomads_hour_probes_are_serialized() {
    assert!(!should_parallelize_hour_availability_probes(
        Some(SourceId::Nomads),
        model_summary(ModelId::Hrrr)
    ));
    assert!(should_parallelize_hour_availability_probes(
        Some(SourceId::Aws),
        model_summary(ModelId::Hrrr)
    ));
    assert!(!should_parallelize_hour_availability_probes(
        None,
        model_summary(ModelId::Hrrr)
    ));
}

#[test]
fn aws_fetches_can_use_idx_subsets_and_parallel_whole_file_fallback() {
    assert!(should_use_idx_subset_fetch(SourceId::Aws));
    assert!(should_use_parallel_whole_file_fetch(SourceId::Aws));
}

#[test]
fn nomads_skips_idx_subsets_and_fetches_full_grib_files() {
    assert!(!should_use_idx_subset_fetch(SourceId::Nomads));
    assert!(!should_use_parallel_whole_file_fetch(SourceId::Nomads));
}

#[test]
fn nomads_fetch_strategy_ignores_variable_patterns() {
    let resolved = ResolvedUrl {
        source: SourceId::Nomads,
        grib_url: "https://nomads.ncep.noaa.gov/file.grib2".to_string(),
        idx_url: Some("https://nomads.ncep.noaa.gov/file.grib2.idx".to_string()),
    };

    assert!(!should_use_idx_subset_fetch(resolved.source));
    assert_eq!(resolved.grib_url, "https://nomads.ncep.noaa.gov/file.grib2");
}

#[test]
fn nomads_probe_uses_grib_url_for_availability() {
    let resolved = ResolvedUrl {
        source: SourceId::Nomads,
        grib_url: "https://nomads.ncep.noaa.gov/file.grib2".to_string(),
        idx_url: Some("https://nomads.ncep.noaa.gov/file.grib2.idx".to_string()),
    };
    assert_eq!(
        resolved.availability_probe_url(),
        "https://nomads.ncep.noaa.gov/file.grib2.idx"
    );
    assert_eq!(resolved.grib_url, "https://nomads.ncep.noaa.gov/file.grib2");
}

#[test]
fn source_probe_uses_fallback_sources_in_registry_order() {
    let urls = vec![
        ResolvedUrl {
            source: SourceId::Nomads,
            grib_url: "https://nomads.ncep.noaa.gov/primary.grib2".to_string(),
            idx_url: None,
        },
        ResolvedUrl {
            source: SourceId::Aws,
            grib_url: "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/fallback.grib2".to_string(),
            idx_url: None,
        },
    ];
    let seen = std::sync::Mutex::new(Vec::new());
    let available = any_source_available(&urls, |resolved| {
        seen.lock().unwrap().push(resolved.source);
        matches!(resolved.source, SourceId::Aws)
    });
    assert!(available);
    assert_eq!(*seen.lock().unwrap(), vec![SourceId::Nomads, SourceId::Aws]);
}

#[test]
fn matching_ranges_uses_idx_patterns() {
    let ranges = idx_subset_ranges(SAMPLE_IDX, &["TMP:2 m above ground", "CAPE:surface"])
        .unwrap()
        .expect("idx subset ranges should exist");
    assert_eq!(ranges.len(), 2);
    assert_eq!(ranges[0].0, 0);
    assert_eq!(ranges[1].0, 96542);
}

#[test]
fn matching_ranges_dedupes_duplicate_selector_hits() {
    let ranges = idx_subset_ranges(
        SAMPLE_IDX,
        &["TMP:2 m above ground", "TMP:2 m above ground"],
    )
    .unwrap()
    .expect("idx subset ranges should exist");
    assert_eq!(ranges.len(), 1);
    assert_eq!(ranges[0].0, 0);
}

#[test]
fn idx_subset_ranges_coalesces_contiguous_messages_only() {
    let ranges = idx_subset_ranges(
        SAMPLE_IDX,
        &["TMP:2 m above ground", "SPFH:2 m above ground"],
    )
    .unwrap()
    .expect("idx subset ranges should exist");
    assert_eq!(ranges, vec![(0, 96541)]);
}

#[test]
fn idx_subset_ranges_falls_back_when_patterns_do_not_match() {
    assert_eq!(
        idx_subset_ranges(SAMPLE_IDX, &["TMP:850 mb"]).unwrap(),
        None
    );
}

#[test]
fn idx_subset_ranges_falls_back_when_idx_is_unparseable() {
    assert_eq!(
        idx_subset_ranges("not an idx", &["TMP:2 m above ground"]).unwrap(),
        None
    );
}

#[test]
fn resolve_fetch_urls_uses_registry_order() {
    let request = ModelRunRequest::new(
        ModelId::RrfsA,
        rustwx_core::CycleSpec::new("20260414", 20).unwrap(),
        2,
        "prs-conus",
    )
    .unwrap();
    let fetch = FetchRequest {
        request,
        source_override: None,
        variable_patterns: Vec::new(),
    };
    let urls = filtered_urls(&fetch).unwrap();
    assert_eq!(urls.len(), 1);
    assert!(urls[0].grib_url.contains("noaa-rrfs-pds.s3.amazonaws.com/rrfs_a/rrfs.20260414/20/rrfs.t20z.prslev.3km.f002.conus.grib2"));
}

#[test]
fn fetch_request_from_timestep_builds_request() {
    let timestep = ModelTimestep::with_source(
        ModelId::Hrrr,
        rustwx_core::CycleSpec::new("20260414", 18).unwrap(),
        3,
        rustwx_core::TimeStamp::new("2026-04-14T21:00:00Z").unwrap(),
        Some(SourceId::Nomads),
    )
    .unwrap();

    let fetch = FetchRequest::from_timestep(
        &timestep,
        "prs",
        timestep.source,
        ["TMP:500 mb", "RH:500 mb"],
    )
    .unwrap();

    assert_eq!(fetch.request.model, ModelId::Hrrr);
    assert_eq!(fetch.request.forecast_hour, 3);
    assert_eq!(fetch.request.product, "prs");
    assert_eq!(fetch.source_override, Some(SourceId::Nomads));
    assert_eq!(
        fetch.variable_patterns,
        vec!["TMP:500 mb".to_string(), "RH:500 mb".to_string()]
    );
}

#[test]
fn structured_selector_matches_supported_upper_air_subset() {
    let height_200 = StructuredMessageSelector::try_from(FieldSelector::isobaric(
        CanonicalField::GeopotentialHeight,
        200,
    ))
    .unwrap();
    let height_200_message =
        ieee_f32_message(PARAMETER_HGT[0], 100, 20_000.0, &[12_040.0], -99.0, -99.0);
    assert!(height_200.matches(&height_200_message));

    let height_250 = StructuredMessageSelector::try_from(FieldSelector::isobaric(
        CanonicalField::GeopotentialHeight,
        250,
    ))
    .unwrap();
    let height_250_message =
        ieee_f32_message(PARAMETER_HGT[0], 100, 25_000.0, &[10_540.0], -99.0, -99.0);
    assert!(height_250.matches(&height_250_message));

    let wind_300 =
        StructuredMessageSelector::try_from(FieldSelector::isobaric(CanonicalField::VWind, 300))
            .unwrap();
    let wind_300_message =
        ieee_f32_message(PARAMETER_VGRD[0], 100, 30_000.0, &[36.0], -99.0, -99.0);
    assert!(wind_300.matches(&wind_300_message));

    let wind_selector =
        StructuredMessageSelector::try_from(FieldSelector::isobaric(CanonicalField::UWind, 850))
            .unwrap();
    let wind_message = ieee_f32_message(
        PARAMETER_UGRD[0],
        100,
        85_000.0,
        &[12.0, 15.0],
        -99.0,
        -98.0,
    );
    assert!(wind_selector.matches(&wind_message));

    let temp_700 = StructuredMessageSelector::try_from(FieldSelector::isobaric(
        CanonicalField::Temperature,
        700,
    ))
    .unwrap();
    let temp_message = ieee_f32_message(PARAMETER_TMP[0], 100, 70_000.0, &[274.0], -99.0, -99.0);
    assert!(temp_700.matches(&temp_message));
    // Stratospheric 7 hPa (level_value=700 Pa) must NOT alias onto 700 hPa.
    let stratospheric_tmp_message =
        ieee_f32_message(PARAMETER_TMP[0], 100, 700.0, &[210.0], -99.0, -99.0);
    assert!(!temp_700.matches(&stratospheric_tmp_message));

    let rh_700 = StructuredMessageSelector::try_from(FieldSelector::isobaric(
        CanonicalField::RelativeHumidity,
        700,
    ))
    .unwrap();
    let rh_message = ieee_f32_message(PARAMETER_RH[0], 100, 70_000.0, &[61.0], -99.0, -99.0);
    assert!(rh_700.matches(&rh_message));
    // GFS/RRFS carry stratospheric RH at level_value=700 Pa (7 hPa). With the
    // old "divide by 100 only when > 2000" heuristic this collided with 700
    // hPa and the first-match extraction picked up the near-zero
    // stratospheric RH, producing a flat-brown 700 mb render.
    let stratospheric_rh_message =
        ieee_f32_message(PARAMETER_RH[0], 100, 700.0, &[0.1], -99.0, -99.0);
    assert!(!rh_700.matches(&stratospheric_rh_message));

    let dewpoint_850 =
        StructuredMessageSelector::try_from(FieldSelector::isobaric(CanonicalField::Dewpoint, 850))
            .unwrap();
    let dewpoint_message =
        ieee_f32_message(PARAMETER_DPT[0], 100, 85_000.0, &[281.0], -99.0, -99.0);
    assert!(dewpoint_850.matches(&dewpoint_message));

    let dewpoint_700 =
        StructuredMessageSelector::try_from(FieldSelector::isobaric(CanonicalField::Dewpoint, 700))
            .unwrap();
    let dewpoint_700_message =
        ieee_f32_message(PARAMETER_DPT[0], 100, 70_000.0, &[270.0], -99.0, -99.0);
    assert!(dewpoint_700.matches(&dewpoint_700_message));

    let vorticity_500 = StructuredMessageSelector::try_from(FieldSelector::isobaric(
        CanonicalField::AbsoluteVorticity,
        500,
    ))
    .unwrap();
    let vorticity_message = ieee_f32_message(
        PARAMETER_ABSOLUTE_VORTICITY[0],
        100,
        50_000.0,
        &[0.00012],
        -99.0,
        -99.0,
    );
    assert!(vorticity_500.matches(&vorticity_message));

    let lsm_surface =
        StructuredMessageSelector::try_from(FieldSelector::surface(CanonicalField::LandSeaMask))
            .unwrap();
    let lsm_message = ieee_f32_message(PARAMETER_LANDSEA_MASK[0], 1, 0.0, &[1.0], -99.0, -99.0);
    assert!(lsm_surface.matches(&lsm_message));

    let terrain_surface = StructuredMessageSelector::try_from(FieldSelector::surface(
        CanonicalField::GeopotentialHeight,
    ))
    .unwrap();
    let terrain_message = ieee_f32_message(PARAMETER_HGT[0], 1, 0.0, &[326.0], -99.0, -99.0);
    assert!(terrain_surface.matches(&terrain_message));

    let temp_2m = StructuredMessageSelector::try_from(FieldSelector::height_agl(
        CanonicalField::Temperature,
        2,
    ))
    .unwrap();
    let temp_2m_message = ieee_f32_message(PARAMETER_TMP[0], 103, 2.0, &[293.2], -99.0, -99.0);
    assert!(temp_2m.matches(&temp_2m_message));

    let dewpoint_2m =
        StructuredMessageSelector::try_from(FieldSelector::height_agl(CanonicalField::Dewpoint, 2))
            .unwrap();
    let dewpoint_2m_message = ieee_f32_message(PARAMETER_DPT[0], 103, 2.0, &[286.4], -99.0, -99.0);
    assert!(dewpoint_2m.matches(&dewpoint_2m_message));

    let rh_2m = StructuredMessageSelector::try_from(FieldSelector::height_agl(
        CanonicalField::RelativeHumidity,
        2,
    ))
    .unwrap();
    let rh_2m_message = ieee_f32_message(PARAMETER_RH[0], 103, 2.0, &[64.0], -99.0, -99.0);
    assert!(rh_2m.matches(&rh_2m_message));

    let hybrid_pressure = StructuredMessageSelector::try_from(FieldSelector::hybrid_level(
        CanonicalField::Pressure,
        7,
    ))
    .unwrap();
    let hybrid_pressure_message =
        ieee_f32_message(PARAMETER_PRESSURE[0], 105, 7.0, &[81_500.0], -99.0, -99.0);
    assert!(hybrid_pressure.matches(&hybrid_pressure_message));

    let hybrid_smoke = StructuredMessageSelector::try_from(FieldSelector::hybrid_level(
        CanonicalField::SmokeMassDensity,
        7,
    ))
    .unwrap();
    let hybrid_smoke_message = ieee_f32_message(
        PARAMETER_SMOKE_MASS_DENSITY[0],
        105,
        7.0,
        &[0.000_012],
        -99.0,
        -99.0,
    );
    assert!(hybrid_smoke.matches(&hybrid_smoke_message));
    let wrong_hybrid_smoke_message = ieee_f32_message(
        PARAMETER_SMOKE_MASS_DENSITY[0],
        105,
        8.0,
        &[0.000_012],
        -99.0,
        -99.0,
    );
    assert!(!hybrid_smoke.matches(&wrong_hybrid_smoke_message));

    let smoke_8m = StructuredMessageSelector::try_from(FieldSelector::height_agl(
        CanonicalField::SmokeMassDensity,
        8,
    ))
    .unwrap();
    let smoke_8m_message = ieee_f32_message(
        PARAMETER_SMOKE_MASS_DENSITY[0],
        103,
        8.0,
        &[0.000_025],
        -99.0,
        -99.0,
    );
    assert!(smoke_8m.matches(&smoke_8m_message));

    let smoke_column = StructuredMessageSelector::try_from(FieldSelector::entire_atmosphere(
        CanonicalField::ColumnIntegratedSmoke,
    ))
    .unwrap();
    let smoke_column_message = ieee_f32_message(
        PARAMETER_COLUMN_INTEGRATED_SMOKE[0],
        200,
        0.0,
        &[0.003],
        -99.0,
        -99.0,
    );
    assert!(smoke_column.matches(&smoke_column_message));

    let u_10m =
        StructuredMessageSelector::try_from(FieldSelector::height_agl(CanonicalField::UWind, 10))
            .unwrap();
    let u_10m_message = ieee_f32_message(PARAMETER_UGRD[0], 103, 10.0, &[8.0], -99.0, -99.0);
    assert!(u_10m.matches(&u_10m_message));

    let wind_speed_10m = StructuredMessageSelector::try_from(FieldSelector::height_agl(
        CanonicalField::WindSpeed,
        10,
    ))
    .unwrap();
    let wind_speed_10m_message =
        ieee_f32_message(PARAMETER_WIND_SPEED[0], 103, 10.0, &[12.0], -99.0, -99.0);
    assert!(wind_speed_10m.matches(&wind_speed_10m_message));

    let gust_10m = StructuredMessageSelector::try_from(FieldSelector::height_agl(
        CanonicalField::WindGust,
        10,
    ))
    .unwrap();
    let gust_surface_message =
        ieee_f32_message(PARAMETER_WIND_GUST[0], 1, 0.0, &[18.0], -99.0, -99.0);
    assert!(gust_10m.matches(&gust_surface_message));
    let gust_10m_message =
        ieee_f32_message(PARAMETER_WIND_GUST[0], 103, 10.0, &[18.0], -99.0, -99.0);
    assert!(gust_10m.matches(&gust_10m_message));

    let mslp = StructuredMessageSelector::try_from(FieldSelector::mean_sea_level(
        CanonicalField::PressureReducedToMeanSeaLevel,
    ))
    .unwrap();
    let mslp_message = ieee_f32_message(PARAMETER_MSLP[0], 101, 0.0, &[100_925.0], -99.0, -99.0);
    assert!(mslp.matches(&mslp_message));
    let mslma_message = ieee_f32_message(PARAMETER_MSLP[2], 101, 0.0, &[100_830.0], -99.0, -99.0);
    assert!(mslp.matches(&mslma_message));

    let pwat = StructuredMessageSelector::try_from(FieldSelector::entire_atmosphere(
        CanonicalField::PrecipitableWater,
    ))
    .unwrap();
    let pwat_message = ieee_f32_message(PARAMETER_PWAT[0], 200, 0.0, &[31.0], -99.0, -99.0);
    assert!(pwat.matches(&pwat_message));

    let qpf = StructuredMessageSelector::try_from(FieldSelector::surface(
        CanonicalField::TotalPrecipitation,
    ))
    .unwrap();
    let qpf_message = ieee_f32_message(
        PARAMETER_TOTAL_PRECIPITATION[0],
        1,
        0.0,
        &[12.0],
        -99.0,
        -99.0,
    );
    assert!(qpf.matches(&qpf_message));

    let pop = StructuredMessageSelector::try_from(FieldSelector::surface(
        CanonicalField::ProbabilityOfPrecipitation,
    ))
    .unwrap();
    let pop_message = ieee_f32_message(
        PARAMETER_PROBABILITY_OF_PRECIPITATION[0],
        1,
        0.0,
        &[80.0],
        -99.0,
        -99.0,
    );
    assert!(pop.matches(&pop_message));

    let tcdc = StructuredMessageSelector::try_from(FieldSelector::entire_atmosphere(
        CanonicalField::TotalCloudCover,
    ))
    .unwrap();
    let tcdc_message = ieee_f32_message(
        PARAMETER_TOTAL_CLOUD_COVER[0],
        200,
        0.0,
        &[84.0],
        -99.0,
        -99.0,
    );
    assert!(tcdc.matches(&tcdc_message));

    let lcdc = StructuredMessageSelector::try_from(FieldSelector::entire_atmosphere(
        CanonicalField::LowCloudCover,
    ))
    .unwrap();
    let lcdc_message = ieee_f32_message(
        PARAMETER_LOW_CLOUD_COVER[0],
        214,
        0.0,
        &[40.0],
        -99.0,
        -99.0,
    );
    assert!(lcdc.matches(&lcdc_message));

    let mcdc = StructuredMessageSelector::try_from(FieldSelector::entire_atmosphere(
        CanonicalField::MiddleCloudCover,
    ))
    .unwrap();
    let mcdc_message = ieee_f32_message(
        PARAMETER_MIDDLE_CLOUD_COVER[0],
        224,
        0.0,
        &[55.0],
        -99.0,
        -99.0,
    );
    assert!(mcdc.matches(&mcdc_message));

    let hcdc = StructuredMessageSelector::try_from(FieldSelector::entire_atmosphere(
        CanonicalField::HighCloudCover,
    ))
    .unwrap();
    let hcdc_message = ieee_f32_message(
        PARAMETER_HIGH_CLOUD_COVER[0],
        234,
        0.0,
        &[70.0],
        -99.0,
        -99.0,
    );
    assert!(hcdc.matches(&hcdc_message));

    let visibility =
        StructuredMessageSelector::try_from(FieldSelector::surface(CanonicalField::Visibility))
            .unwrap();
    let visibility_message =
        ieee_f32_message(PARAMETER_VISIBILITY[0], 1, 0.0, &[16_000.0], -99.0, -99.0);
    assert!(visibility.matches(&visibility_message));

    let simulated_ir = StructuredMessageSelector::try_from(FieldSelector::nominal_top(
        CanonicalField::SimulatedInfraredBrightnessTemperature,
    ))
    .unwrap();
    let simulated_ir_message =
        ieee_f32_message(PARAMETER_SIMULATED_IR[0], 8, 0.0, &[234.5], -99.0, -99.0);
    let simulated_ir_wrong_level =
        ieee_f32_message(PARAMETER_SIMULATED_IR[0], 10, 0.0, &[234.5], -99.0, -99.0);
    assert!(simulated_ir.matches(&simulated_ir_message));
    assert!(!simulated_ir.matches(&simulated_ir_wrong_level));

    let categorical_rain = StructuredMessageSelector::try_from(FieldSelector::surface(
        CanonicalField::CategoricalRain,
    ))
    .unwrap();
    let categorical_rain_message =
        ieee_f32_message(PARAMETER_CATEGORICAL_RAIN[0], 1, 0.0, &[1.0], -99.0, -99.0);
    assert!(categorical_rain.matches(&categorical_rain_message));
    let categorical_rain_hrrr_message =
        ieee_f32_message(PARAMETER_CATEGORICAL_RAIN[1], 1, 0.0, &[1.0], -99.0, -99.0);
    assert!(categorical_rain.matches(&categorical_rain_hrrr_message));

    let categorical_freezing_rain = StructuredMessageSelector::try_from(FieldSelector::surface(
        CanonicalField::CategoricalFreezingRain,
    ))
    .unwrap();
    let categorical_freezing_rain_message = ieee_f32_message(
        PARAMETER_CATEGORICAL_FREEZING_RAIN[0],
        1,
        0.0,
        &[1.0],
        -99.0,
        -99.0,
    );
    assert!(categorical_freezing_rain.matches(&categorical_freezing_rain_message));
    let categorical_freezing_rain_hrrr_message = ieee_f32_message(
        PARAMETER_CATEGORICAL_FREEZING_RAIN[1],
        1,
        0.0,
        &[1.0],
        -99.0,
        -99.0,
    );
    assert!(categorical_freezing_rain.matches(&categorical_freezing_rain_hrrr_message));

    let categorical_ice_pellets = StructuredMessageSelector::try_from(FieldSelector::surface(
        CanonicalField::CategoricalIcePellets,
    ))
    .unwrap();
    let categorical_ice_pellets_message = ieee_f32_message(
        PARAMETER_CATEGORICAL_ICE_PELLETS[0],
        1,
        0.0,
        &[1.0],
        -99.0,
        -99.0,
    );
    assert!(categorical_ice_pellets.matches(&categorical_ice_pellets_message));
    let categorical_ice_pellets_hrrr_message = ieee_f32_message(
        PARAMETER_CATEGORICAL_ICE_PELLETS[1],
        1,
        0.0,
        &[1.0],
        -99.0,
        -99.0,
    );
    assert!(categorical_ice_pellets.matches(&categorical_ice_pellets_hrrr_message));

    let categorical_snow = StructuredMessageSelector::try_from(FieldSelector::surface(
        CanonicalField::CategoricalSnow,
    ))
    .unwrap();
    let categorical_snow_message =
        ieee_f32_message(PARAMETER_CATEGORICAL_SNOW[0], 1, 0.0, &[1.0], -99.0, -99.0);
    assert!(categorical_snow.matches(&categorical_snow_message));
    let categorical_snow_hrrr_message =
        ieee_f32_message(PARAMETER_CATEGORICAL_SNOW[1], 1, 0.0, &[1.0], -99.0, -99.0);
    assert!(categorical_snow.matches(&categorical_snow_hrrr_message));

    let reflectivity_1km = StructuredMessageSelector::try_from(FieldSelector::height_agl(
        CanonicalField::RadarReflectivity,
        1000,
    ))
    .unwrap();
    let reflectivity_message = ieee_f32_message(
        PARAMETER_RADAR_REFLECTIVITY[0],
        103,
        1000.0,
        &[42.0],
        -99.0,
        -99.0,
    );
    assert!(reflectivity_1km.matches(&reflectivity_message));

    let uh_2_5km = StructuredMessageSelector::try_from(FieldSelector::height_layer_agl(
        CanonicalField::UpdraftHelicity,
        2000,
        5000,
    ))
    .unwrap();
    let uh_message = ieee_f32_message(
        PARAMETER_UPDRAFT_HELICITY[0],
        103,
        5000.0,
        &[125.0],
        -99.0,
        -99.0,
    );
    assert!(uh_2_5km.matches(&uh_message));

    // Off-grid isobaric levels (not a 25 hPa multiple in 100..=1000) stay
    // unsupported; 500 mb dewpoint and 925 mb vorticity are now on-grid.
    assert!(matches!(
        StructuredMessageSelector::try_from(FieldSelector::isobaric(CanonicalField::Dewpoint, 510)),
        Err(IoError::UnsupportedStructuredSelector { .. })
    ));
    assert!(matches!(
        StructuredMessageSelector::try_from(FieldSelector::isobaric(
            CanonicalField::AbsoluteVorticity,
            935
        )),
        Err(IoError::UnsupportedStructuredSelector { .. })
    ));
    assert!(matches!(
        StructuredMessageSelector::try_from(FieldSelector::isobaric(
            CanonicalField::RelativeVorticity,
            500
        )),
        Err(IoError::UnsupportedStructuredSelector { .. })
    ));
    assert!(matches!(
        StructuredMessageSelector::try_from(FieldSelector::height_layer_agl(
            CanonicalField::UpdraftHelicity,
            0,
            3000
        )),
        Err(IoError::UnsupportedStructuredSelector { .. })
    ));
    assert!(matches!(
        StructuredMessageSelector::try_from(FieldSelector::hybrid_level(
            CanonicalField::SmokeMassDensity,
            51
        )),
        Err(IoError::UnsupportedStructuredSelector { .. })
    ));
    assert!(matches!(
        StructuredMessageSelector::try_from(FieldSelector::height_agl(
            CanonicalField::SmokeMassDensity,
            2
        )),
        Err(IoError::UnsupportedStructuredSelector { .. })
    ));
    assert!(matches!(
        StructuredMessageSelector::try_from(FieldSelector::entire_atmosphere(
            CanonicalField::SimulatedInfraredBrightnessTemperature
        )),
        Err(IoError::UnsupportedStructuredSelector { .. })
    ));
}

#[test]
fn extract_ignores_stratospheric_pa_alias_of_tropospheric_level() {
    // GFS/RRFS-A carry both 7 hPa (level_value = 700 Pa) and 700 hPa
    // (level_value = 70_000 Pa) messages in the same file. The 7 hPa one
    // appears first. The extractor must return the 700 hPa message.
    let stratospheric = ieee_f32_message(PARAMETER_RH[0], 100, 700.0, &[0.1, 0.2], 261.0, 262.0);
    let tropospheric =
        ieee_f32_message(PARAMETER_RH[0], 100, 70_000.0, &[55.0, 65.0], 261.0, 262.0);
    let grib = Grib2File {
        messages: vec![stratospheric, tropospheric],
    };

    let field =
        extract_pressure_field_from_grib2(&grib, CanonicalField::RelativeHumidity, 700).unwrap();

    assert_eq!(field.values, vec![55.0, 65.0]);
}

#[test]
fn extract_prefers_instantaneous_temperature_over_statistical_alias() {
    // ECMWF Open Data can carry PDT 4.8 statistical 2 m temperature
    // messages before the instantaneous PDT 4.0 2 m temperature message.
    // The statistical fields can be zero at f000, which becomes -273.15 C
    // downstream if we take the first parameter/level match.
    let mut statistical = ieee_f32_message(PARAMETER_TMP[0], 103, 2.0, &[0.0, 0.0], 261.0, 262.0);
    statistical.product.template = 8;
    statistical.product.statistical_process_type = Some(2);
    statistical.product.statistical_time_range_unit = Some(1);
    statistical.product.time_range_length = Some(6);

    let instantaneous = ieee_f32_message(PARAMETER_TMP[0], 103, 2.0, &[280.0, 281.5], 261.0, 262.0);
    let grib = Grib2File {
        messages: vec![statistical, instantaneous],
    };

    let field = extract_field_from_grib2(
        &grib,
        FieldSelector::height_agl(CanonicalField::Temperature, 2),
    )
    .unwrap();

    assert_eq!(field.values, vec![280.0, 281.5]);
}

#[test]
fn partial_extract_at_forecast_hour_uses_requested_lead_time() {
    let mut f003 = ieee_f32_message(PARAMETER_TMP[0], 103, 2.0, &[273.0], -99.0, -99.0);
    f003.product.time_range_unit = 1;
    f003.product.forecast_time = 3;

    let mut f024 = ieee_f32_message(PARAMETER_TMP[0], 103, 2.0, &[294.0], -99.0, -99.0);
    f024.product.time_range_unit = 1;
    f024.product.forecast_time = 24;

    let selector = FieldSelector::height_agl(CanonicalField::Temperature, 2);
    let grib = Grib2File {
        messages: vec![f003, f024],
    };

    let partial =
        extract_fields_from_grib2_partial_at_forecast_hour(&grib, &[selector], 24).unwrap();
    assert!(partial.missing.is_empty());
    assert_eq!(partial.extracted[0].values, vec![294.0]);

    let missing =
        extract_fields_from_grib2_partial_at_forecast_hour(&grib, &[selector], 12).unwrap();
    assert!(missing.extracted.is_empty());
    assert_eq!(missing.missing, vec![selector]);
}

#[test]
fn partial_extract_at_forecast_hour_matches_statistical_window_end() {
    let mut qpf = ieee_f32_message(
        PARAMETER_TOTAL_PRECIPITATION[0],
        1,
        0.0,
        &[7.5],
        -99.0,
        -99.0,
    );
    qpf.product.template = 8;
    qpf.product.time_range_unit = 1;
    qpf.product.forecast_time = 18;
    qpf.product.statistical_time_range_unit = Some(1);
    qpf.product.time_range_length = Some(6);

    let selector = FieldSelector::surface(CanonicalField::TotalPrecipitation);
    let grib = Grib2File {
        messages: vec![qpf],
    };

    let partial =
        extract_fields_from_grib2_partial_at_forecast_hour(&grib, &[selector], 24).unwrap();
    assert!(partial.missing.is_empty());
    assert_eq!(partial.extracted[0].values, vec![7.5]);
}

/// REGRESSION (found live on RRFS-A f002, 2026-06-11): a surface file may
/// carry BOTH the run-total (0→h) and the trailing-window ((h−1)→h) APCP
/// accumulation, and both end at hour h so both tie on the end-hour forecast
/// score. Selection of the run total must NOT depend on which message comes
/// first in the file: HRRR orders the run total first (accidentally correct),
/// RRFS-A orders the window first — which silently stored the 1 h window as
/// `apcp_run_total`. The run-total selection must prefer the accumulation
/// that starts at the run start (hour 0) in BOTH file orders, and the
/// trailing re-select at h−1 must still find the window.
#[test]
fn qpf_run_total_prefers_zero_start_accumulation_in_either_file_order() {
    let make_apcp = |start_hour: u32, length: u32, values: &[f32]| {
        let mut message = ieee_f32_message(
            PARAMETER_TOTAL_PRECIPITATION[0],
            1,
            0.0,
            values,
            -99.0,
            -99.0,
        );
        message.product.template = 8;
        message.product.time_range_unit = 1;
        message.product.forecast_time = start_hour;
        message.product.statistical_time_range_unit = Some(1);
        message.product.time_range_length = Some(length);
        message
    };
    // f002: window = 1→2 hour acc, run total = 0→2 hour acc.
    let window = make_apcp(1, 1, &[1.5]);
    let run_total = make_apcp(0, 2, &[9.0]);
    let selector = FieldSelector::surface(CanonicalField::TotalPrecipitation);

    // RRFS-A file order: window FIRST (this is the order that bit live).
    let rrfs_order = Grib2File {
        messages: vec![window.clone(), run_total.clone()],
    };
    let picked = extract_fields_from_grib2_partial_at_forecast_hour(&rrfs_order, &[selector], 2)
        .unwrap()
        .extracted
        .swap_remove(0);
    assert_eq!(
        picked.values,
        vec![9.0],
        "run total must be the 0->2 accumulation even when the window comes first"
    );

    // HRRR file order: run total first (the historical accidental pass).
    let hrrr_order = Grib2File {
        messages: vec![run_total.clone(), window.clone()],
    };
    let picked = extract_fields_from_grib2_partial_at_forecast_hour(&hrrr_order, &[selector], 2)
        .unwrap()
        .extracted
        .swap_remove(0);
    assert_eq!(picked.values, vec![9.0]);

    // The trailing-window re-select at h−1 = 1 must still find the WINDOW
    // (its start hour matches exactly; the run total's start and end both
    // miss) — in both orders.
    for messages in [
        vec![window.clone(), run_total.clone()],
        vec![run_total, window],
    ] {
        let grib = Grib2File { messages };
        let picked = extract_fields_from_grib2_partial_at_forecast_hour(&grib, &[selector], 1)
            .unwrap()
            .extracted
            .swap_remove(0);
        assert_eq!(
            picked.values,
            vec![1.5],
            "the h-1 re-select must still pick the trailing window"
        );
    }
}

#[test]
fn extract_distinguishes_pop_from_accumulated_qpf() {
    let mut probability = ieee_f32_message(
        PARAMETER_TOTAL_PRECIPITATION[0],
        1,
        0.0,
        &[80.0, 90.0],
        -99.0,
        -99.0,
    );
    probability.product.template = 9;

    let mut accumulation = ieee_f32_message(
        PARAMETER_TOTAL_PRECIPITATION[0],
        1,
        0.0,
        &[2.0, 4.0],
        -99.0,
        -99.0,
    );
    accumulation.product.template = 8;

    let grib = Grib2File {
        messages: vec![probability, accumulation],
    };

    let qpf = extract_field_from_grib2(
        &grib,
        FieldSelector::surface(CanonicalField::TotalPrecipitation),
    )
    .unwrap();
    let pop = extract_field_from_grib2(
        &grib,
        FieldSelector::surface(CanonicalField::ProbabilityOfPrecipitation),
    )
    .unwrap();

    assert_eq!(qpf.values, vec![2.0, 4.0]);
    assert_eq!(pop.values, vec![80.0, 90.0]);
    assert_eq!(pop.units, "%");
}

#[test]
fn extract_qmd_percentile_uses_exact_percentile_metadata() {
    let mut p10 = ieee_f32_message(PARAMETER_TMP[0], 103, 2.0, &[270.0], -99.0, -99.0);
    p10.product.template = 6;
    p10.product.percentile_value = Some(10);
    let mut p50 = ieee_f32_message(PARAMETER_TMP[0], 103, 2.0, &[280.0], -99.0, -99.0);
    p50.product.template = 6;
    p50.product.percentile_value = Some(50);
    let mut p90 = ieee_f32_message(PARAMETER_TMP[0], 103, 2.0, &[290.0], -99.0, -99.0);
    p90.product.template = 6;
    p90.product.percentile_value = Some(90);
    let grib = Grib2File {
        messages: vec![p10, p50, p90],
    };

    let field = extract_field_from_grib2(
        &grib,
        FieldSelector::height_agl(CanonicalField::Temperature, 2).with_percentile(50),
    )
    .unwrap();

    assert_eq!(field.values, vec![280.0]);
    assert_eq!(
        field.selector,
        FieldSelector::height_agl(CanonicalField::Temperature, 2).with_percentile(50)
    );
}

#[test]
fn extract_qmd_percentile_does_not_fallback_to_wrong_percentile() {
    let mut p50 = ieee_f32_message(PARAMETER_TMP[0], 103, 2.0, &[280.0], -99.0, -99.0);
    p50.product.template = 6;
    p50.product.percentile_value = Some(50);
    let grib = Grib2File {
        messages: vec![p50],
    };

    let err = extract_field_from_grib2(
        &grib,
        FieldSelector::height_agl(CanonicalField::Temperature, 2).with_percentile(90),
    )
    .unwrap_err();

    assert!(matches!(err, IoError::FieldNotFound { .. }));
}

#[test]
fn extract_qmd_probability_uses_exact_threshold_metadata() {
    let mut freeze = ieee_f32_message(PARAMETER_TMP[0], 103, 2.0, &[70.0], -99.0, -99.0);
    freeze.product.template = 5;
    freeze.product.probability_type = Some(0);
    freeze.product.probability_lower_limit = Some(273.0);
    let mut hot = ieee_f32_message(PARAMETER_TMP[0], 103, 2.0, &[30.0], -99.0, -99.0);
    hot.product.template = 5;
    hot.product.probability_type = Some(1);
    hot.product.probability_upper_limit = Some(298.8);
    let grib = Grib2File {
        messages: vec![freeze, hot],
    };

    let freezing_probability = extract_field_from_grib2(
        &grib,
        FieldSelector::height_agl(CanonicalField::Temperature, 2)
            .with_probability(ProbabilitySelection::below_milli(273_000)),
    )
    .unwrap();
    let hot_probability = extract_field_from_grib2(
        &grib,
        FieldSelector::height_agl(CanonicalField::Temperature, 2)
            .with_probability(ProbabilitySelection::above_milli(298_800)),
    )
    .unwrap();

    assert_eq!(freezing_probability.values, vec![70.0]);
    assert_eq!(hot_probability.values, vec![30.0]);
    assert_eq!(freezing_probability.units, "%");
    assert_eq!(hot_probability.units, "%");
}

#[test]
fn extract_qmd_derived_mean_and_stddev_do_not_alias() {
    let mut mean = ieee_f32_message(PARAMETER_TMP[0], 103, 2.0, &[279.0], -99.0, -99.0);
    mean.product.template = 2;
    mean.product.derived_forecast_type = Some(0);
    let mut stddev = ieee_f32_message(PARAMETER_TMP[0], 103, 2.0, &[3.5], -99.0, -99.0);
    stddev.product.template = 2;
    stddev.product.derived_forecast_type = Some(2);
    let grib = Grib2File {
        messages: vec![stddev, mean],
    };

    let mean_field = extract_field_from_grib2(
        &grib,
        FieldSelector::height_agl(CanonicalField::Temperature, 2).with_ensemble_mean(),
    )
    .unwrap();
    let stddev_field = extract_field_from_grib2(
        &grib,
        FieldSelector::height_agl(CanonicalField::Temperature, 2)
            .with_ensemble_standard_deviation(),
    )
    .unwrap();

    assert_eq!(mean_field.values, vec![279.0]);
    assert_eq!(stddev_field.values, vec![3.5]);
}

#[test]
fn ensemble_mean_selector_accepts_weighted_mean_product() {
    let mut weighted_mean = ieee_f32_message(PARAMETER_TMP[0], 103, 2.0, &[281.0], -99.0, -99.0);
    weighted_mean.product.template = 2;
    weighted_mean.product.derived_forecast_type = Some(1);
    let grib = Grib2File {
        messages: vec![weighted_mean],
    };

    let field = extract_field_from_grib2(
        &grib,
        FieldSelector::height_agl(CanonicalField::Temperature, 2).with_ensemble_mean(),
    )
    .unwrap();

    assert_eq!(field.values, vec![281.0]);
}

#[test]
fn default_selector_can_fallback_to_ensemble_mean_when_file_is_mean_product() {
    let mut mean = ieee_f32_message(PARAMETER_TMP[0], 103, 2.0, &[279.0], -99.0, -99.0);
    mean.product.template = 2;
    mean.product.derived_forecast_type = Some(0);
    let grib = Grib2File {
        messages: vec![mean],
    };

    let field = extract_field_from_grib2(
        &grib,
        FieldSelector::height_agl(CanonicalField::Temperature, 2),
    )
    .unwrap();

    assert_eq!(field.selector.product, FieldProduct::Default);
    assert_eq!(field.values, vec![279.0]);
}

#[test]
fn default_selector_can_fallback_to_weighted_ensemble_mean_product() {
    let mut mean = ieee_f32_message(PARAMETER_TMP[0], 103, 2.0, &[281.0], -99.0, -99.0);
    mean.product.template = 2;
    mean.product.derived_forecast_type = Some(1);
    let grib = Grib2File {
        messages: vec![mean],
    };

    let field = extract_field_from_grib2(
        &grib,
        FieldSelector::height_agl(CanonicalField::Temperature, 2),
    )
    .unwrap();

    assert_eq!(field.selector.product, FieldProduct::Default);
    assert_eq!(field.values, vec![281.0]);
}

#[test]
fn default_qpf_selector_can_fallback_to_ensemble_mean_accumulation() {
    let mut qpf = ieee_f32_message(
        PARAMETER_TOTAL_PRECIPITATION[0],
        1,
        0.0,
        &[12.7],
        -99.0,
        -99.0,
    );
    qpf.product.template = 8;
    qpf.product.derived_forecast_type = Some(1);
    let grib = Grib2File {
        messages: vec![qpf],
    };

    let field = extract_field_from_grib2(
        &grib,
        FieldSelector::surface(CanonicalField::TotalPrecipitation),
    )
    .unwrap();

    assert_eq!(field.selector.product, FieldProduct::Default);
    assert_eq!(field.values, vec![12.7]);
}

#[test]
fn default_temperature_selector_does_not_fallback_to_qmd_percentiles() {
    let mut percentile = ieee_f32_message(PARAMETER_TMP[0], 103, 2.0, &[280.0], -99.0, -99.0);
    percentile.product.template = 6;
    percentile.product.percentile_value = Some(50);
    let grib = Grib2File {
        messages: vec![percentile],
    };

    let err = extract_field_from_grib2(
        &grib,
        FieldSelector::height_agl(CanonicalField::Temperature, 2),
    )
    .unwrap_err();

    assert!(matches!(err, IoError::FieldNotFound { .. }));
}

#[test]
fn structured_selector_accepts_standard_mslp_parameter_zero() {
    let message = ieee_f32_message(
        PARAMETER_MSLP[0],
        101,
        0.0,
        &[101000.0, 100750.0],
        261.0,
        262.0,
    );
    let grib = Grib2File {
        messages: vec![message],
    };

    let field = extract_field_from_grib2(
        &grib,
        FieldSelector::mean_sea_level(CanonicalField::PressureReducedToMeanSeaLevel),
    )
    .unwrap();

    assert_eq!(field.values, vec![101000.0, 100750.0]);
}

#[test]
fn extract_field_from_grib2_returns_selector_backed_field() {
    // 500 hPa is encoded as 50_000 Pa per GRIB2 Code Table 4.5 level 100.
    let message = ieee_f32_message(
        PARAMETER_TMP[0],
        100,
        50_000.0,
        &[255.0, 256.5],
        261.0,
        262.0,
    );
    let grib = Grib2File {
        messages: vec![message],
    };

    let field = extract_pressure_field_from_grib2(&grib, CanonicalField::Temperature, 500).unwrap();

    assert_eq!(
        field.selector,
        FieldSelector::isobaric(CanonicalField::Temperature, 500)
    );
    assert_eq!(field.units, "K");
    assert_eq!(field.grid.shape.nx, 2);
    assert_eq!(field.grid.shape.ny, 1);
    assert_eq!(field.grid.lon_deg, vec![-99.0, -98.0]);
    assert_eq!(field.values, vec![255.0, 256.5]);
}

#[test]
fn nbm_speed_direction_messages_synthesize_10m_uv_components() {
    let direction = ieee_f32_message(
        PARAMETER_WIND_DIRECTION[0],
        103,
        10.0,
        &[0.0, 90.0, 180.0, 270.0],
        261.0,
        264.0,
    );
    let speed = ieee_f32_message(
        PARAMETER_WIND_SPEED[0],
        103,
        10.0,
        &[10.0, 10.0, 10.0, 10.0],
        261.0,
        264.0,
    );
    let grib = Grib2File {
        messages: vec![direction, speed],
    };
    let u_selector = FieldSelector::height_agl(CanonicalField::UWind, 10);
    let v_selector = FieldSelector::height_agl(CanonicalField::VWind, 10);

    let mut partial = extract_fields_from_grib2_partial(&grib, &[u_selector, v_selector])
        .expect("standard U/V messages are absent but partial extraction should soft-fail");
    assert_eq!(partial.missing, vec![u_selector, v_selector]);

    synthesize_nbm_10m_wind_components_from_speed_direction(&grib, &mut partial).unwrap();
    assert!(partial.missing.is_empty());

    let u = partial
        .extracted
        .iter()
        .find(|field| field.selector == u_selector)
        .expect("synthesized U component");
    let v = partial
        .extracted
        .iter()
        .find(|field| field.selector == v_selector)
        .expect("synthesized V component");

    assert_component_values(&u.values, &[0.0, -10.0, 0.0, 10.0]);
    assert_component_values(&v.values, &[-10.0, 0.0, 10.0, 0.0]);
}

fn assert_component_values(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert!(
            (*actual - *expected).abs() < 1.0e-4,
            "actual={actual} expected={expected}"
        );
    }
}

#[test]
fn extract_hybrid_level_volume_from_grib2_stacks_requested_levels() {
    let smoke_level_2 = ieee_f32_message(
        PARAMETER_SMOKE_MASS_DENSITY[0],
        105,
        2.0,
        &[0.3, 0.4],
        -99.0,
        -98.0,
    );
    let smoke_level_1 = ieee_f32_message(
        PARAMETER_SMOKE_MASS_DENSITY[0],
        105,
        1.0,
        &[0.1, 0.2],
        -99.0,
        -98.0,
    );
    let grib = Grib2File {
        messages: vec![smoke_level_2, smoke_level_1],
    };

    let volume =
        extract_hybrid_level_volume_from_grib2(&grib, CanonicalField::SmokeMassDensity, &[1, 2])
            .unwrap();

    assert_eq!(volume.field, CanonicalField::SmokeMassDensity);
    assert_eq!(volume.levels_hybrid, vec![1, 2]);
    assert_eq!(volume.units, "kg/m^3");
    assert_eq!(volume.level_slice(0), Some(&[0.1, 0.2][..]));
    assert_eq!(volume.level_slice(1), Some(&[0.3, 0.4][..]));
    assert_eq!(
        volume.selector_at(0),
        Some(FieldSelector::hybrid_level(
            CanonicalField::SmokeMassDensity,
            1
        ))
    );
}

#[test]
fn extract_hrrr_wrfnat_smoke_fields_returns_surface_column_and_hybrid_pairs() {
    let mut messages = Vec::new();
    for level in 1..=HRRR_WRFNAT_HYBRID_LEVEL_COUNT {
        messages.push(ieee_f32_message(
            PARAMETER_PRESSURE[0],
            105,
            f64::from(level),
            &[80_000.0 - level as f32, 79_000.0 - level as f32],
            -99.0,
            -98.0,
        ));

        let smoke_values = match level {
            1 => vec![0.1, 0.2],
            2 => vec![0.3, 0.4],
            _ => vec![level as f32, level as f32 + 0.5],
        };
        messages.push(ieee_f32_message(
            PARAMETER_SMOKE_MASS_DENSITY[0],
            105,
            f64::from(level),
            &smoke_values,
            -99.0,
            -98.0,
        ));
    }
    messages.push(ieee_f32_message(
        PARAMETER_SMOKE_MASS_DENSITY[0],
        103,
        8.0,
        &[1.5, 2.5],
        -99.0,
        -98.0,
    ));
    messages.push(ieee_f32_message(
        PARAMETER_COLUMN_INTEGRATED_SMOKE[0],
        200,
        0.0,
        &[3.5, 4.5],
        -99.0,
        -98.0,
    ));
    let grib = Grib2File { messages };

    let extracted = extract_hrrr_wrfnat_smoke_fields_from_grib2(&grib).unwrap();

    assert_eq!(extracted.hybrid_smoke.level_count(), 50);
    assert_eq!(extracted.hybrid_pressure.level_count(), 50);
    assert_eq!(
        extracted.near_surface_smoke.selector,
        FieldSelector::height_agl(CanonicalField::SmokeMassDensity, 8)
    );
    assert_eq!(
        extracted.column_smoke.selector,
        FieldSelector::entire_atmosphere(CanonicalField::ColumnIntegratedSmoke)
    );
    assert_eq!(extracted.hybrid_smoke.level_slice(0), Some(&[0.1, 0.2][..]));
    assert_eq!(extracted.hybrid_smoke.level_slice(1), Some(&[0.3, 0.4][..]));
    assert_eq!(
        extracted.hybrid_pressure.selector_at(49),
        Some(FieldSelector::hybrid_level(CanonicalField::Pressure, 50))
    );
    assert_eq!(extracted.near_surface_smoke.values, vec![1.5, 2.5]);
    assert_eq!(extracted.column_smoke.values, vec![3.5, 4.5]);
}

#[test]
fn extract_field_from_real_pressure_bytes_uses_structured_matching() {
    let path = sample_pressure_subset_path();
    if !path.exists() {
        eprintln!(
            "skipping real pressure subset test; fixture is not present at {}",
            path.display()
        );
        return;
    }
    let bytes = std::fs::read(&path).unwrap();

    let temp_500 =
        extract_pressure_field_from_bytes(&bytes, CanonicalField::Temperature, 500).unwrap();
    let temp_700 =
        extract_pressure_field_from_bytes(&bytes, CanonicalField::Temperature, 700).unwrap();
    let hgt_700 =
        extract_pressure_field_from_bytes(&bytes, CanonicalField::GeopotentialHeight, 700).unwrap();
    let hgt_850 =
        extract_pressure_field_from_bytes(&bytes, CanonicalField::GeopotentialHeight, 850).unwrap();
    let u_700 = extract_pressure_field_from_bytes(&bytes, CanonicalField::UWind, 700).unwrap();
    let v_700 = extract_pressure_field_from_bytes(&bytes, CanonicalField::VWind, 700).unwrap();

    assert_eq!(
        temp_500.selector,
        FieldSelector::isobaric(CanonicalField::Temperature, 500)
    );
    assert_eq!(
        temp_700.selector,
        FieldSelector::isobaric(CanonicalField::Temperature, 700)
    );
    assert_eq!(temp_500.units, "K");
    assert_eq!(temp_700.units, "K");
    assert_eq!(hgt_700.units, "gpm");
    assert_eq!(hgt_850.units, "gpm");
    assert_eq!(u_700.units, "m/s");
    assert_eq!(v_700.units, "m/s");
    assert_eq!(temp_700.grid.shape, hgt_700.grid.shape);
    assert_eq!(temp_700.grid.shape, u_700.grid.shape);
    assert_eq!(u_700.grid.shape, v_700.grid.shape);
    assert_eq!(temp_500.grid.shape, hgt_850.grid.shape);
    assert_eq!(temp_500.values.len(), temp_500.grid.shape.len());
    assert_eq!(temp_700.values.len(), temp_700.grid.shape.len());
    assert_eq!(hgt_700.values.len(), hgt_700.grid.shape.len());
    assert_eq!(hgt_850.values.len(), hgt_850.grid.shape.len());
    assert_eq!(u_700.values.len(), u_700.grid.shape.len());
    assert_eq!(v_700.values.len(), v_700.grid.shape.len());
    assert!(temp_500.values.iter().any(|value| value.is_finite()));
    assert!(temp_700.values.iter().any(|value| value.is_finite()));
    assert!(hgt_700.values.iter().any(|value| value.is_finite()));
    assert!(hgt_850.values.iter().any(|value| value.is_finite()));
    assert!(u_700.values.iter().any(|value| value.is_finite()));
    assert!(v_700.values.iter().any(|value| value.is_finite()));
}

#[test]
fn extract_fields_from_real_pressure_bytes_batches_parse_and_matching() {
    let path = sample_pressure_subset_path();
    if !path.exists() {
        eprintln!(
            "skipping real pressure subset batch test; fixture is not present at {}",
            path.display()
        );
        return;
    }
    let bytes = std::fs::read(&path).unwrap();
    let selectors = [
        FieldSelector::isobaric(CanonicalField::Temperature, 500),
        FieldSelector::isobaric(CanonicalField::Temperature, 700),
        FieldSelector::isobaric(CanonicalField::GeopotentialHeight, 700),
        FieldSelector::isobaric(CanonicalField::UWind, 700),
        FieldSelector::isobaric(CanonicalField::VWind, 700),
    ];

    let batched = extract_fields_from_bytes(&bytes, &selectors).unwrap();

    assert_eq!(batched.len(), selectors.len());
    for (selector, field) in selectors.iter().zip(batched.iter()) {
        assert_eq!(&field.selector, selector);
    }

    let single_temp_500 =
        extract_pressure_field_from_bytes(&bytes, CanonicalField::Temperature, 500).unwrap();
    let single_hgt_700 =
        extract_pressure_field_from_bytes(&bytes, CanonicalField::GeopotentialHeight, 700).unwrap();
    let single_u_700 =
        extract_pressure_field_from_bytes(&bytes, CanonicalField::UWind, 700).unwrap();

    assert_eq!(batched[0], single_temp_500);
    assert_eq!(batched[2], single_hgt_700);
    assert_eq!(batched[3], single_u_700);
}

#[test]
fn normalize_and_rotate_longitude_rows_keeps_rows_monotone() {
    let mut lat = vec![40.0, 40.0, 40.0, 40.0, 39.0, 39.0, 39.0, 39.0];
    let mut lon = vec![0.0, 90.0, 180.0, 270.0, 0.0, 90.0, 180.0, 270.0];
    let mut values = vec![1.0, 2.0, 3.0, 4.0, 11.0, 12.0, 13.0, 14.0];

    let row_wraps = normalize_and_rotate_longitude_grid_rows(&mut lat, &mut lon, 4, 2);
    rotate_rows_left(&mut values, 4, &row_wraps);

    assert_eq!(row_wraps, [3, 3]);
    assert_eq!(lon[..4], [-90.0, 0.0, 90.0, 180.0]);
    assert_eq!(lon[4..], [-90.0, 0.0, 90.0, 180.0]);
    assert_eq!(values[..4], [4.0, 1.0, 2.0, 3.0]);
    assert_eq!(values[4..], [14.0, 11.0, 12.0, 13.0]);
    assert_eq!(lat[..4], [40.0, 40.0, 40.0, 40.0]);
    assert_eq!(lat[4..], [39.0, 39.0, 39.0, 39.0]);
}
