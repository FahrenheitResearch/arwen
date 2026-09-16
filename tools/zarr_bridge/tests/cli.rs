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

impl Fixture {
    fn set_attr(&self, array: &str, name: &str, value: Value) {
        let path = self.0.join(format!("store/{array}/.zattrs"));
        let mut attrs: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        attrs[name] = value;
        write_json(&path, attrs);
    }

    fn set_coverage(&self, attrs: Value) {
        write_json(&self.0.join("store/.zattrs"), attrs);
    }
}

/// THE ARCO DOOR'S FIRST BYTE.  Google's public ERA5 Zarr publishes its
/// `level` coordinate as `Hectopascal(hPa)`: the unit's written-out name
/// with its symbol in brackets.  Compared as a string against the declared
/// `hPa` it disagreed, and `gpuwm fetch --source era5 --era5-provider arco
/// --retrieve` downloaded nothing at all -- it refused before the first
/// chunk read, on every request, for every date.
#[test]
fn a_name_bracket_symbol_unit_spelling_is_read_as_its_symbol() {
    let fixture = Fixture::new();
    fixture.set_attr("level", "units", json!("Hectopascal(hPa)"));
    fixture.set_attr("temperature", "units", json!("Kelvin(K)"));
    let result = fixture.extract();
    assert!(result.status.success(), "{}", String::from_utf8_lossy(&result.stderr));
}

/// A unit that is ITSELF bracketed keeps being read whole: `(0-1)` is the
/// spelling ERA5 gives fractional fields and the dimensionless row carries
/// it verbatim, so reducing it to its interior must not be what admits it.
#[test]
fn a_wholly_bracketed_unit_is_still_matched_as_the_whole_string() {
    let fixture = Fixture::new();
    fixture.set_attr("temperature", "units", json!("(0-1)"));
    let result = fixture.extract();
    assert!(!result.status.success());
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(stderr.contains("\"(0-1)\""), "{stderr}");
    assert!(stderr.contains("\"K\""), "{stderr}");
}

/// The refusal that remains names BOTH strings and the way out.
#[test]
fn a_unit_no_spelling_matches_names_both_strings_and_the_way_out() {
    let fixture = Fixture::new();
    fixture.set_attr("temperature", "units", json!("Pa"));
    let result = fixture.extract();
    assert!(!result.status.success());
    let stderr = String::from_utf8_lossy(&result.stderr);
    for expected in ["\"Pa\"", "\"K\"", "kelvin", "Name(symbol)"] {
        assert!(stderr.contains(expected), "{expected} missing from {stderr}");
    }
    assert!(!fixture.0.join("forcing.nc").exists());
}

/// ONE AUTHORITY FOR BOTH BOUNDARIES.  The notice used to name the
/// finalized stop while the refusal named coverage running to the
/// preliminary one, so the same store answered "how far does this go" with
/// two different dates and neither said what kind of boundary it was.
#[test]
fn the_preliminary_notice_names_the_finalized_and_the_accepted_boundary() {
    let fixture = Fixture::new();
    fixture.set_coverage(json!({
        "valid_time_start": "2019-01-01",
        "valid_time_stop": "2019-12-31",
        "valid_time_stop_era5t": "2020-01-01"}));
    let result = fixture.extract();
    assert!(result.status.success(), "{}", String::from_utf8_lossy(&result.stderr));
    let stderr = String::from_utf8_lossy(&result.stderr);
    for expected in [
        "2 requested time(s)",
        "2019-12-31",
        "valid_time_stop)",
        "2020-01-01",
        "valid_time_stop_era5t",
        "preliminary ERA5T",
    ] {
        assert!(stderr.contains(expected), "{expected} missing from {stderr}");
    }
}

#[test]
fn a_time_past_the_preliminary_boundary_is_refused_by_that_boundary_and_its_reason() {
    let fixture = Fixture::new();
    fixture.set_coverage(json!({
        "valid_time_start": "2019-01-01",
        "valid_time_stop": "2019-11-30",
        "valid_time_stop_era5t": "2019-12-31"}));
    let result = fixture.extract();
    assert!(!result.status.success());
    let stderr = String::from_utf8_lossy(&result.stderr);
    for expected in [
        "2020-01-01 00:00:00",
        "2019-01-01 (valid_time_start)",
        "2019-12-31 (valid_time_stop_era5t)",
        "last hour of the preliminary ERA5T stream",
        "2019-11-30 (valid_time_stop)",
        "Request a time inside that window",
    ] {
        assert!(stderr.contains(expected), "{expected} missing from {stderr}");
    }
    assert!(!fixture.0.join("forcing.nc").exists());
}

/// A store with no preliminary stream refuses on the finalized boundary and
/// says that is what it is, so the two cases are never read as one.
#[test]
fn a_store_without_an_era5t_stream_refuses_on_the_finalized_boundary() {
    let fixture = Fixture::new();
    fixture.set_coverage(json!({
        "valid_time_start": "2019-01-01",
        "valid_time_stop": "2019-12-31"}));
    let result = fixture.extract();
    assert!(!result.status.success());
    let stderr = String::from_utf8_lossy(&result.stderr);
    for expected in [
        "2019-12-31 (valid_time_stop)",
        "last hour of the finalized reanalysis",
        "declares no preliminary ERA5T stream past it",
    ] {
        assert!(stderr.contains(expected), "{expected} missing from {stderr}");
    }
}
