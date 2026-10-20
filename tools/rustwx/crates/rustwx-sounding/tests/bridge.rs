use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use ecape_rs::{CapeType, ParcelOptions, StormMotionType, calc_ecape_ncape, calc_ecape_parcel};
use image::ImageFormat;
use rustwx_sounding::{
    EcapeIntegrationStatus, ExternalEcapeSummary, ExternalEcapeValue, NativeParcelContext,
    NativeSounding, ParcelFlavor, PendingEcapeRequest, SoundingColumn, SoundingMetadata,
    ecape_status, render_full_sounding_png, render_full_sounding_with_ecape_png,
    render_sharppy_sounding_png, require_future_ecape_bridge,
};

fn sample_column() -> SoundingColumn {
    let pressure_hpa = vec![1000.0, 925.0, 850.0, 700.0, 500.0, 300.0, 200.0];
    let height_m_msl = vec![100.0, 800.0, 1500.0, 3100.0, 5600.0, 9200.0, 12000.0];
    let temperature_c = vec![30.0, 24.0, 18.0, 4.0, -15.0, -40.0, -55.0];
    let dewpoint_c = vec![22.0, 18.0, 12.0, -4.0, -30.0, -50.0, -65.0];
    let u_knots = vec![0.0, 5.1, 12.9, 28.2, 49.2, 59.1, 68.9];
    let v_knots = vec![10.0, 14.1, 15.3, 10.3, 0.0, -10.4, -12.2];
    SoundingColumn {
        pressure_hpa,
        height_m_msl,
        temperature_c,
        dewpoint_c,
        u_ms: u_knots
            .iter()
            .map(|value| value * 0.514_444_444_444_444_5)
            .collect(),
        v_ms: v_knots
            .iter()
            .map(|value| value * 0.514_444_444_444_444_5)
            .collect(),
        omega_pa_s: Vec::new(),
        metadata: SoundingMetadata {
            station_id: "TST".into(),
            valid_time: "2026-04-14T20:00:00Z".into(),
            latitude_deg: Some(35.22),
            longitude_deg: Some(-97.44),
            elevation_m: Some(397.0),
            sample_method: Some("nearest".into()),
            box_radius_lat_deg: None,
            box_radius_lon_deg: None,
        },
    }
}

fn sample_external_ecape() -> ExternalEcapeSummary {
    ExternalEcapeSummary {
        source: Some("rustwx-calc / metrust".into()),
        storm_motion: Some("Bunkers RM".into()),
        values: vec![
            ExternalEcapeValue {
                parcel: ParcelFlavor::SurfaceBased,
                ecape_j_kg: 2125.0,
            },
            ExternalEcapeValue {
                parcel: ParcelFlavor::MixedLayer,
                ecape_j_kg: 1840.0,
            },
            ExternalEcapeValue {
                parcel: ParcelFlavor::MostUnstable,
                ecape_j_kg: 2480.0,
            },
        ],
        notes: vec![
            "Caller-supplied ECAPE values are shown here as an external summary block.".into(),
        ],
    }
}

fn sample_native_context() -> Vec<NativeParcelContext> {
    vec![
        NativeParcelContext {
            parcel: ParcelFlavor::SurfaceBased,
            cape_j_kg: 2850.0,
            cin_j_kg: -42.0,
            lcl_m_agl: 920.0,
            lfc_m_agl: 1420.0,
            el_m_agl: 12100.0,
        },
        NativeParcelContext {
            parcel: ParcelFlavor::MixedLayer,
            cape_j_kg: 2310.0,
            cin_j_kg: -28.0,
            lcl_m_agl: 1100.0,
            lfc_m_agl: 1880.0,
            el_m_agl: 11600.0,
        },
        NativeParcelContext {
            parcel: ParcelFlavor::MostUnstable,
            cape_j_kg: 3020.0,
            cin_j_kg: -15.0,
            lcl_m_agl: 760.0,
            lfc_m_agl: 1180.0,
            el_m_agl: 12400.0,
        },
    ]
}

#[test]
fn converts_generic_column_into_sharprs_profile() {
    let column = sample_column();
    let profile = column.to_sharprs_profile().expect("column should bridge");

    assert_eq!(profile.num_levels(), column.len());
    assert_eq!(profile.station.station_id, "TST");
    assert_eq!(profile.station.datetime, "2026-04-14T20:00:00Z");
    assert!((profile.u[3] - 28.2).abs() < 0.2);
    assert!((profile.v[3] - 10.3).abs() < 0.2);
}

#[test]
fn validate_allows_saturated_levels() {
    let mut column = sample_column();
    column.dewpoint_c[1] = column.temperature_c[1];

    column
        .validate()
        .expect("saturated level should remain valid");
}

#[test]
fn rejects_non_finite_values() {
    let mut column = sample_column();
    column.u_ms[2] = f64::NAN;

    let error = column.validate().expect_err("NaN wind should be rejected");
    let message = error.to_string();

    assert!(message.contains("u_ms"));
    assert!(message.contains("finite"));
}

#[test]
fn rejects_pressure_that_increases_with_height() {
    let mut column = sample_column();
    column.pressure_hpa[2] = 930.0;

    let error = column
        .validate()
        .expect_err("pressure increase should be rejected");
    let message = error.to_string();

    assert!(message.contains("pressure_hpa"));
    assert!(message.contains("non-increasing"));
}

#[test]
fn rejects_height_that_decreases() {
    let mut column = sample_column();
    column.height_m_msl[3] = 1400.0;

    let error = column
        .validate()
        .expect_err("height decrease should be rejected");
    let message = error.to_string();

    assert!(message.contains("height_m_msl"));
    assert!(message.contains("non-decreasing"));
}

#[test]
fn rejects_dewpoint_above_temperature() {
    let mut column = sample_column();
    column.dewpoint_c[1] = column.temperature_c[1] + 0.5;

    let error = column
        .validate()
        .expect_err("dewpoint above temperature should be rejected");
    let message = error.to_string();

    assert!(message.contains("dewpoint_c"));
    assert!(message.contains("above temperature"));
}

#[test]
fn roundtrip_from_sharprs_profile_preserves_station_and_level_count() {
    let native = NativeSounding::from_column(&sample_column()).expect("bridge should succeed");
    let roundtrip = SoundingColumn::from_sharprs_profile(&native.profile);

    assert_eq!(roundtrip.metadata.station_id, "TST");
    assert_eq!(roundtrip.metadata.valid_time, "2026-04-14T20:00:00Z");
    assert_eq!(roundtrip.len(), 7);
    assert!((roundtrip.u_ms[4] - sample_column().u_ms[4]).abs() < 0.05);
}

#[test]
fn native_sounding_populates_verified_ecape_table_params() {
    let native = NativeSounding::from_column(&sample_column()).expect("bridge should succeed");

    assert!(native.verified_ecape.surface_based.ecape.is_finite());
    assert!(native.verified_ecape.surface_based.ncape.is_finite());
    assert!(native.verified_ecape.surface_based.cape.is_finite());
    assert!(native.verified_ecape.mixed_layer.ecape.is_finite());
    assert!(native.verified_ecape.most_unstable.ecape.is_finite());
}

#[test]
fn native_sounding_ecape_column_selects_the_analytic_quantity() {
    let column = sample_column();
    let native = NativeSounding::from_column(&column).expect("bridge should succeed");
    let pressure_pa = column
        .pressure_hpa
        .iter()
        .map(|pressure| pressure * 100.0)
        .collect::<Vec<_>>();
    let temperature_k = column
        .temperature_c
        .iter()
        .map(|temperature| temperature + 273.15)
        .collect::<Vec<_>>();
    let dewpoint_k = column
        .dewpoint_c
        .iter()
        .map(|dewpoint| dewpoint + 273.15)
        .collect::<Vec<_>>();
    let qv_kgkg = column
        .pressure_hpa
        .iter()
        .zip(column.dewpoint_c.iter())
        .map(|(&pressure, &dewpoint)| {
            wx_math::thermo::specific_humidity_from_dewpoint(pressure, dewpoint)
        })
        .collect::<Vec<_>>();
    let options = ParcelOptions {
        cape_type: CapeType::SurfaceBased,
        storm_motion_type: StormMotionType::RightMoving,
        pseudoadiabatic: Some(true),
        ..ParcelOptions::default()
    };
    let analytic = calc_ecape_ncape(
        &column.height_m_msl,
        &pressure_pa,
        &temperature_k,
        &qv_kgkg,
        &column.u_ms,
        &column.v_ms,
        &options,
    )
    .expect("analytic ECAPE should solve");
    let post_path = calc_ecape_parcel(
        &column.height_m_msl,
        &pressure_pa,
        &temperature_k,
        &dewpoint_k,
        &column.u_ms,
        &column.v_ms,
        &options,
    )
    .expect("entraining parcel path should solve");

    let displayed = native.verified_ecape.surface_based.ecape;
    assert!((displayed - analytic.ecape_jkg).abs() < 1.0e-6);
    assert!(
        (displayed - post_path.ecape_jkg).abs() > 1.0,
        "ECAPE column must not select calc_ecape_parcel().ecape_jkg"
    );
}

#[test]
fn renders_full_sounding_png_bytes() {
    let png = render_full_sounding_png(&sample_column()).expect("render should succeed");

    assert!(png.len() > 1000, "png too small: {}", png.len());
    assert_eq!(&png[..4], &[0x89, 0x50, 0x4e, 0x47]);
}

#[test]
fn renders_classic_sharppy_layout_png_bytes() {
    let classic =
        render_sharppy_sounding_png(&sample_column()).expect("classic render should succeed");
    let full = render_full_sounding_png(&sample_column()).expect("full render should succeed");

    assert!(classic.len() > 1000, "png too small: {}", classic.len());
    assert_eq!(&classic[..4], &[0x89, 0x50, 0x4e, 0x47]);
    assert_ne!(
        classic, full,
        "classic SHARPpy layout should not include the rustwx replacement table"
    );
}

#[test]
fn writes_full_sounding_png_to_disk() {
    let native = NativeSounding::from_column(&sample_column()).expect("bridge should succeed");
    let output_path = temp_png_path("rustwx_sounding_bridge");

    native
        .write_full_png(&output_path)
        .expect("png write should succeed");

    let png = fs::read(&output_path).expect("written png should exist");
    assert!(png.len() > 1000);
    assert_eq!(&png[..4], &[0x89, 0x50, 0x4e, 0x47]);

    let _ = fs::remove_file(output_path);
}

#[test]
fn ecape_bridge_status_reports_native_verified_table_path() {
    assert!(matches!(
        ecape_status(),
        EcapeIntegrationStatus::NativeVerifiedTableAndExternalAnnotationBridge
    ));

    let error = require_future_ecape_bridge(
        &sample_column(),
        PendingEcapeRequest {
            parcel: ParcelFlavor::MixedLayer,
        },
    )
    .expect_err("internal sharprs ECAPE should remain unavailable");

    let message = error.to_string();
    assert!(message.contains("rustwx-sounding"));
    assert!(message.contains("ecape-rs"));
}

#[test]
fn rejects_duplicate_external_ecape_values() {
    let error = ExternalEcapeSummary {
        source: Some("rustwx-calc".into()),
        storm_motion: None,
        values: vec![
            ExternalEcapeValue {
                parcel: ParcelFlavor::MixedLayer,
                ecape_j_kg: 1800.0,
            },
            ExternalEcapeValue {
                parcel: ParcelFlavor::MixedLayer,
                ecape_j_kg: 1900.0,
            },
        ],
        notes: Vec::new(),
    }
    .validate()
    .expect_err("duplicate parcel values should be rejected");

    assert!(error.to_string().contains("duplicate ML ECAPE value"));
}

#[test]
fn builds_annotation_context_with_native_cape_cin_companions() {
    let annotation = sample_external_ecape()
        .annotation_context(&sample_native_context())
        .expect("annotation context should build");

    assert_eq!(annotation.source_label, "rustwx-calc / metrust");
    assert_eq!(annotation.storm_motion_label.as_deref(), Some("Bunkers RM"));
    assert_eq!(annotation.rows.len(), 3);
    assert_eq!(annotation.rows[0].parcel, ParcelFlavor::SurfaceBased);
    assert_eq!(annotation.rows[0].native_cape_j_kg, 2850.0);
    assert_eq!(annotation.rows[0].native_cin_j_kg, -42.0);
    let ratio = annotation.rows[0]
        .ecape_fraction_of_cape
        .expect("ratio should exist");
    assert!((ratio - (2125.0 / 2850.0)).abs() < 1e-6);
}

#[test]
fn renders_full_sounding_png_with_external_ecape_block() {
    let base_png = render_full_sounding_png(&sample_column()).expect("base render should succeed");
    let annotated_png =
        render_full_sounding_with_ecape_png(&sample_column(), &sample_external_ecape())
            .expect("annotated render should succeed");

    let base_image = image::load_from_memory_with_format(&base_png, ImageFormat::Png)
        .expect("base png should decode")
        .to_rgba8();
    let annotated_image = image::load_from_memory_with_format(&annotated_png, ImageFormat::Png)
        .expect("annotated png should decode")
        .to_rgba8();

    assert_eq!(annotated_image.width(), base_image.width());
    assert!(
        annotated_image.height() > base_image.height(),
        "annotated image height {} should exceed base {}",
        annotated_image.height(),
        base_image.height()
    );

    let base_raw_len = base_image.as_raw().len();
    let appended = &annotated_image.as_raw()[base_raw_len..];
    let non_background_pixels = appended
        .chunks_exact(4)
        .filter(|pixel| pixel[0] != 10 || pixel[1] != 10 || pixel[2] != 22 || pixel[3] != 255)
        .count();

    assert!(
        non_background_pixels > 1000,
        "external ECAPE block looks blank: only {} non-background pixels",
        non_background_pixels
    );
}

fn temp_png_path(prefix: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be after epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("{prefix}_{nanos}.png"))
}
