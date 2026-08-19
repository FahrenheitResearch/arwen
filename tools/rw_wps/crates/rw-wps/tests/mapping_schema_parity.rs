use rw_wps::mapping::{
    AxisRole, GridLocation, MissingPolicy, NativeMapping, VariableSelector, read_mapping,
    validate_mapping,
};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

const GFS_MAPPING_SHA256: &str = "5b0f41a7f4ddee1116ce8310dfd67827761413908d45402e1f55f32facc61d86";
const ERA5_MAPPING_SHA256: &str =
    "d2c9ee08e45478a64e4d2bba689e9bad1d2e97bde713477ee2a4de26e31d7ad3";

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

fn assert_fixture_sha256(path: &Path, expected: &str) {
    // Git may materialize text fixtures with CRLF on Windows. Pin the exact
    // repository content while making that checkout-only transformation inert.
    let text = std::fs::read_to_string(path).unwrap().replace("\r\n", "\n");
    let digest = Sha256::digest(text.as_bytes());
    assert_eq!(format!("{digest:x}"), expected);
}

fn load_fixture(name: &str) -> NativeMapping {
    read_mapping(&fixture(name)).unwrap()
}

#[test]
fn exact_gpuwm_gfs_grib2_mapping_validates_with_bounded_soil_layers() {
    let path = fixture("rw-wps-gfs-pressure-grib2.mapping.json");
    assert_fixture_sha256(&path, GFS_MAPPING_SHA256);
    let mapping = read_mapping(&path).unwrap();

    assert!(validate_mapping(&mapping).passed());
    let soil = &mapping.fields["soil_temperature"];
    assert_eq!(soil.missing, MissingPolicy::PreserveMask);
    let bounds = soil
        .selectors
        .iter()
        .map(|selector| match selector {
            VariableSelector::Grib2 {
                level_type,
                level_value,
                second_level_type,
                second_level_value,
                ..
            } => (
                *level_type,
                *level_value,
                *second_level_type,
                *second_level_value,
            ),
            other => panic!("expected GRIB2 soil selector, got {other:?}"),
        })
        .collect::<Vec<_>>();
    assert_eq!(
        bounds,
        vec![
            (Some(106), Some(0.0), Some(106), Some(0.1)),
            (Some(106), Some(0.1), Some(106), Some(0.4)),
            (Some(106), Some(0.4), Some(106), Some(1.0)),
            (Some(106), Some(1.0), Some(106), Some(2.0)),
        ]
    );
}

#[test]
fn exact_gpuwm_era5_netcdf_mapping_validates_with_soil_selector_stacks() {
    let path = fixture("rw-wps-era5-netcdf.mapping.json");
    assert_fixture_sha256(&path, ERA5_MAPPING_SHA256);
    let mapping = read_mapping(&path).unwrap();

    assert!(validate_mapping(&mapping).passed());
    for name in ["soil_temperature", "volumetric_soil_moisture"] {
        let field = &mapping.fields[name];
        assert_eq!(field.selector_stack_axis, Some(AxisRole::Soil));
        assert_eq!(field.selectors.len(), 4);
        assert!(field.source_axes.contains(&AxisRole::Soil));
        assert_eq!(field.location, GridLocation::Soil);
    }
}

#[test]
fn grib2_second_fixed_surface_selector_is_an_atomic_pair() {
    for missing_key in ["second_level_type", "second_level_value"] {
        let mut json: serde_json::Value = serde_json::from_slice(
            &std::fs::read(fixture("rw-wps-gfs-pressure-grib2.mapping.json")).unwrap(),
        )
        .unwrap();
        json["fields"]["soil_temperature"]["selectors"][0]
            .as_object_mut()
            .unwrap()
            .remove(missing_key);
        let mapping: NativeMapping = serde_json::from_value(json).unwrap();
        let report = validate_mapping(&mapping);
        assert!(report.errors.iter().any(|diagnostic| {
            diagnostic.code == "incomplete_second_fixed_surface"
                && diagnostic.field.as_deref() == Some("soil_temperature")
        }));
    }
}

#[test]
fn netcdf_selector_stack_axis_constraints_match_the_engine() {
    let mut wrong_format = load_fixture("rw-wps-gfs-pressure-grib2.mapping.json");
    wrong_format
        .fields
        .get_mut("soil_temperature")
        .unwrap()
        .selector_stack_axis = Some(AxisRole::Soil);
    assert!(
        validate_mapping(&wrong_format)
            .errors
            .iter()
            .any(|diagnostic| diagnostic.code == "selector_stack_axis_source")
    );

    let mut unsupported_axis = load_fixture("rw-wps-era5-netcdf.mapping.json");
    unsupported_axis
        .fields
        .get_mut("soil_temperature")
        .unwrap()
        .selector_stack_axis = Some(AxisRole::Vertical);
    assert!(
        validate_mapping(&unsupported_axis)
            .errors
            .iter()
            .any(|diagnostic| { diagnostic.code == "selector_stack_axis_unsupported" })
    );

    let mut one_selector = load_fixture("rw-wps-era5-netcdf.mapping.json");
    one_selector
        .fields
        .get_mut("soil_temperature")
        .unwrap()
        .selectors
        .truncate(1);
    assert!(
        validate_mapping(&one_selector)
            .errors
            .iter()
            .any(|diagnostic| { diagnostic.code == "selector_stack_axis_cardinality" })
    );

    let mut missing_axis = load_fixture("rw-wps-era5-netcdf.mapping.json");
    missing_axis
        .fields
        .get_mut("soil_temperature")
        .unwrap()
        .source_axes
        .retain(|axis| *axis != AxisRole::Soil);
    assert!(
        validate_mapping(&missing_axis)
            .errors
            .iter()
            .any(|diagnostic| {
                diagnostic.code == "selector_stack_axis_missing_from_source_axes"
            })
    );
}

#[test]
fn preserve_mask_is_restricted_to_land_aware_soil_fields() {
    let mut mapping = load_fixture("rw-wps-gfs-pressure-grib2.mapping.json");
    mapping.fields.get_mut("surface_pressure").unwrap().missing = MissingPolicy::PreserveMask;
    let report = validate_mapping(&mapping);
    assert!(report.errors.iter().any(|diagnostic| {
        diagnostic.code == "preserve_mask_location"
            && diagnostic.field.as_deref() == Some("surface_pressure")
    }));
}
