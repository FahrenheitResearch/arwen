//! Exercise the shipped executable on a hand-encoded Zarr v2 store and read
//! its NetCDF output with the independent netcrust reader.
use serde_json::{json, Value};
use std::{fs, path::{Path, PathBuf}, process::{Command, Output}, time::{SystemTime, UNIX_EPOCH}};

struct Fixture(PathBuf);

impl Fixture {
    fn new() -> Self {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let path = std::env::temp_dir().join(format!("rw-zarr-cli-{}-{nonce}", std::process::id()));
        fs::create_dir(&path).unwrap();
        let fixture = Self(path);
        let store = fixture.0.join("store");
        fs::create_dir(&store).unwrap();
        write_json(&store.join(".zgroup"), json!({"zarr_format": 2}));
        write_json(&store.join(".zattrs"), json!({
            "valid_time_start": "2020-01-01", "valid_time_stop": "2020-01-01"}));
        array(&store, "time", &[2], &[0.0, 1.0], json!({
            "units": "hours since 2020-01-01 00:00:00", "_ARRAY_DIMENSIONS": ["time"]}));
        array(&store, "latitude", &[2], &[-1.0, 1.0], json!({"units": "degrees_north"}));
        array(&store, "longitude", &[4], &[0.0, 90.0, 180.0, 270.0], json!({"units": "degrees_east"}));
        array(&store, "level", &[2], &[500.0, 1000.0], json!({"units": "hPa"}));
        let mut packed = Vec::new();
        for start in [0, 80, 20, 100] {
            packed.extend((start..start + 8).map(f64::from));
        }
        packed[16] = -999.0;
        array(&store, "temperature", &[2, 2, 2, 4], &packed, json!({
            "units": "K", "_ARRAY_DIMENSIONS": ["time", "level", "latitude", "longitude"],
            "scale_factor": 0.5, "add_offset": 200.0, "_FillValue": -999.0}));
        write_json(&fixture.0.join("request.json"), json!({
            "store": store, "times": ["2020-01-01T00:00:00Z", "2020-01-01T01:00:00Z"],
            "area": [-1.0, 0.0, 1.0, 90.0], "expected_levels_hpa": [500.0, 1000.0],
            "fields": [{"source": "temperature", "output": "T", "units": "K", "pressure": true}]
        }));
        fixture
    }

    fn extract(&self) -> Output {
        Command::new(env!("CARGO_BIN_EXE_rw_zarr")).arg("extract")
            .arg(self.0.join("request.json")).arg(self.0.join("forcing.nc"))
            .output().unwrap()
    }
}

impl Drop for Fixture {
    fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); }
}

fn write_json(path: &Path, value: Value) {
    fs::write(path, serde_json::to_vec(&value).unwrap()).unwrap();
}

fn array(store: &Path, name: &str, shape: &[usize], values: &[f64], attrs: Value) {
    let directory = store.join(name);
    fs::create_dir(&directory).unwrap();
    write_json(&directory.join(".zarray"), json!({
        "zarr_format": 2, "shape": shape, "chunks": shape, "dtype": "<f8",
        "compressor": null, "fill_value": null, "order": "C", "filters": null}));
    write_json(&directory.join(".zattrs"), attrs);
    let chunk = vec!["0"; shape.len()].join(".");
    let bytes: Vec<u8> = values.iter().flat_map(|value| value.to_le_bytes()).collect();
    fs::write(directory.join(chunk), bytes).unwrap();
}

#[test]
fn executable_preserves_time_level_packing_missingness_and_record_decode() {
    let fixture = Fixture::new();
    let result = fixture.extract();
    assert!(result.status.success(), "{}", String::from_utf8_lossy(&result.stderr));
    let report: Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(report["schema"], "arwen.regular-forcing.v1");
    assert_eq!(report["levels_hpa"], json!([1000.0, 500.0]));
    let file = netcrust::File::open(fixture.0.join("forcing.nc")).unwrap();
    let record = file.read_array_f64_record_or_all("T", 1).unwrap();
    let expected = [251.5, 250.0, 250.5, 251.0, 253.5, 252.0, 252.5, 253.0,
                    211.5, f64::NAN, 210.5, 211.0, 213.5, 212.0, 212.5, 213.0];
    assert_eq!(record.values().len(), expected.len());
    for (actual, wanted) in record.values().iter().zip(expected) {
        assert!(if wanted.is_nan() { actual.is_nan() } else { *actual == wanted }, "{actual} != {wanted}");
    }
    let mask = file.read_array_f64_record_or_all("missing__T", 1).unwrap();
    assert_eq!(mask.values().iter().filter(|value| **value == 1.0).count(), 1);
    assert_eq!(mask.values()[9], 1.0);
    let dump = Command::new(env!("CARGO_BIN_EXE_rw_zarr")).arg("dump-record")
        .arg(fixture.0.join("forcing.nc")).arg("1").arg(fixture.0.join("decoded"))
        .arg("T").output().unwrap();
    assert!(dump.status.success(), "{}", String::from_utf8_lossy(&dump.stderr));
    let document: Value = serde_json::from_slice(&dump.stdout).unwrap();
    assert_eq!(document["schema"], "arwen.regular-forcing-record.v1");
    let raw = fs::read(fixture.0.join("decoded/variable-0000.bin")).unwrap();
    let expected_bytes: Vec<u8> = record.values().iter().flat_map(|value| value.to_le_bytes()).collect();
    assert_eq!(raw, expected_bytes);
}

#[test]
fn extraction_refuses_to_clobber_an_existing_output() {
    let fixture = Fixture::new();
    let output = fixture.0.join("forcing.nc");
    fs::write(&output, b"existing user data").unwrap();
    let result = fixture.extract();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("output already exists"));
    assert_eq!(fs::read(output).unwrap(), b"existing user data");
}

#[test]
fn inconsistent_units_refuse_before_creating_output() {
    let fixture = Fixture::new();
    let path = fixture.0.join("store/temperature/.zattrs");
    let mut attrs: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    attrs["units"] = json!("Pa");
    write_json(&path, attrs);
    let result = fixture.extract();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("disagree with declared units"));
    assert!(!fixture.0.join("forcing.nc").exists());
}
