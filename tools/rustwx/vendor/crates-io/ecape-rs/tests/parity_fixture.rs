use ecape_rs::{CapeType, ParcelOptions, StormMotionType, calc_ecape_parcel};
use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Debug, Deserialize)]
struct Fixture {
    height_m: Vec<f64>,
    pressure_pa: Vec<f64>,
    temperature_k: Vec<f64>,
    dewpoint_k: Vec<f64>,
    u_wind_ms: Vec<f64>,
    v_wind_ms: Vec<f64>,
    storm_motion_u_ms: f64,
    storm_motion_v_ms: f64,
    expected: BTreeMap<String, Expected>,
}

#[derive(Debug, Deserialize)]
struct Expected {
    ecape_jkg: f64,
    ncape_jkg: f64,
    cape_jkg: f64,
    cin_jkg: f64,
    lfc_m: Option<f64>,
    el_m: Option<f64>,
}

fn fixture() -> Fixture {
    serde_json::from_str(include_str!("fixtures/parity_column.json")).unwrap()
}

fn assert_close(label: &str, actual: f64, expected: f64) {
    let tolerance = 1e-6_f64.max(expected.abs() * 1e-10);
    assert!(
        (actual - expected).abs() <= tolerance,
        "{label}: actual={actual}, expected={expected}, tolerance={tolerance}"
    );
}

fn assert_option_close(label: &str, actual: Option<f64>, expected: Option<f64>) {
    match (actual, expected) {
        (Some(actual), Some(expected)) => assert_close(label, actual, expected),
        (None, None) => {}
        _ => panic!("{label}: actual={actual:?}, expected={expected:?}"),
    }
}

#[test]
fn defaults_remain_stable() {
    let defaults = ParcelOptions::default();

    assert_eq!(defaults.cape_type, CapeType::SurfaceBased);
    assert_eq!(defaults.storm_motion_type, StormMotionType::RightMoving);
    assert_eq!(defaults.mixed_layer_depth_pa, Some(10000.0));
    assert_eq!(defaults.inflow_layer_bottom_m, Some(0.0));
    assert_eq!(defaults.inflow_layer_top_m, Some(1000.0));
    assert_eq!(defaults.pseudoadiabatic, Some(true));
}

#[test]
fn option_parsing_accepts_wrapper_aliases() {
    assert_eq!("sb".parse::<CapeType>().unwrap(), CapeType::SurfaceBased);
    assert_eq!(
        "surface".parse::<CapeType>().unwrap(),
        CapeType::SurfaceBased
    );
    assert_eq!(
        "surface_based".parse::<CapeType>().unwrap(),
        CapeType::SurfaceBased
    );
    assert_eq!("ml".parse::<CapeType>().unwrap(), CapeType::MixedLayer);
    assert_eq!(
        "mixed_layer".parse::<CapeType>().unwrap(),
        CapeType::MixedLayer
    );
    assert_eq!("mu".parse::<CapeType>().unwrap(), CapeType::MostUnstable);
    assert_eq!(
        "most_unstable".parse::<CapeType>().unwrap(),
        CapeType::MostUnstable
    );

    assert_eq!(
        "right_moving".parse::<StormMotionType>().unwrap(),
        StormMotionType::RightMoving
    );
    assert_eq!(
        "bunkers_rm".parse::<StormMotionType>().unwrap(),
        StormMotionType::RightMoving
    );
    assert_eq!(
        "right".parse::<StormMotionType>().unwrap(),
        StormMotionType::RightMoving
    );
    assert_eq!(
        "rm".parse::<StormMotionType>().unwrap(),
        StormMotionType::RightMoving
    );
    assert_eq!(
        "left_moving".parse::<StormMotionType>().unwrap(),
        StormMotionType::LeftMoving
    );
    assert_eq!(
        "bunkers_lm".parse::<StormMotionType>().unwrap(),
        StormMotionType::LeftMoving
    );
    assert_eq!(
        "left".parse::<StormMotionType>().unwrap(),
        StormMotionType::LeftMoving
    );
    assert_eq!(
        "lm".parse::<StormMotionType>().unwrap(),
        StormMotionType::LeftMoving
    );
    assert_eq!(
        "mean_wind".parse::<StormMotionType>().unwrap(),
        StormMotionType::MeanWind
    );
    assert_eq!(
        "mean".parse::<StormMotionType>().unwrap(),
        StormMotionType::MeanWind
    );
    assert_eq!(
        "custom".parse::<StormMotionType>().unwrap(),
        StormMotionType::UserDefined
    );

    assert!("nonsense".parse::<CapeType>().is_err());
    assert!("nonsense".parse::<StormMotionType>().is_err());
    assert_eq!(
        CapeType::parse_or_default(Some("nonsense")),
        CapeType::SurfaceBased
    );
    assert_eq!(
        StormMotionType::parse_or_default(Some("nonsense")),
        StormMotionType::RightMoving
    );

    let options: ParcelOptions =
        serde_json::from_str(r#"{"cape_type":"surface-based","storm_motion_type":"bunkers-rm"}"#)
            .unwrap();
    assert_eq!(options.cape_type, CapeType::SurfaceBased);
    assert_eq!(options.storm_motion_type, StormMotionType::RightMoving);
}

#[test]
fn parity_fixture_outputs_remain_stable() {
    let fixture = fixture();

    for (cape_type_raw, expected) in &fixture.expected {
        let cape_type = cape_type_raw.parse::<CapeType>().unwrap();
        let options = ParcelOptions {
            cape_type,
            storm_motion_type: StormMotionType::UserDefined,
            storm_motion_u_ms: Some(fixture.storm_motion_u_ms),
            storm_motion_v_ms: Some(fixture.storm_motion_v_ms),
            ..ParcelOptions::default()
        };

        let result = calc_ecape_parcel(
            &fixture.height_m,
            &fixture.pressure_pa,
            &fixture.temperature_k,
            &fixture.dewpoint_k,
            &fixture.u_wind_ms,
            &fixture.v_wind_ms,
            &options,
        )
        .unwrap();

        assert_close(cape_type_raw, result.ecape_jkg, expected.ecape_jkg);
        assert_close(cape_type_raw, result.ncape_jkg, expected.ncape_jkg);
        assert_close(cape_type_raw, result.cape_jkg, expected.cape_jkg);
        assert_close(cape_type_raw, result.cin_jkg, expected.cin_jkg);
        assert_option_close(cape_type_raw, result.lfc_m, expected.lfc_m);
        assert_option_close(cape_type_raw, result.el_m, expected.el_m);
    }
}
