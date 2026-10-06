//! Real NCO output with dense attributes, checked against native netCDF values.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::Value;

const SOURCE: &[u8] = include_bytes!("fixtures/dense-attrs-nco.nc4");

fn native() -> Value {
    serde_json::from_str(include_str!("fixtures/dense-attrs-nco.native.json"))
        .expect("native fixture receipt")
}

fn source_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/dense-attrs-nco.nc4")
}

fn bytes_from_hex(value: &Value) -> Vec<u8> {
    let text = value.as_str().expect("hex string");
    assert_eq!(text.len() % 2, 0);
    (0..text.len())
        .step_by(2)
        .map(|index| u8::from_str_radix(&text[index..index + 2], 16).expect("hex byte"))
        .collect()
}

fn shape(value: &Value) -> Vec<usize> {
    value
        .as_array()
        .expect("shape")
        .iter()
        .map(|length| length.as_u64().expect("length") as usize)
        .collect()
}

fn assert_native_dimensions(file: &netcrust::File, expected: &Value) {
    let dimensions = file.dimensions().expect("dimensions");
    assert_eq!(dimensions.len(), expected.as_array().unwrap().len());
    for dimension in expected.as_array().unwrap() {
        let name = dimension["name"].as_str().unwrap();
        let actual = dimensions
            .iter()
            .find(|item| item.name() == name)
            .unwrap_or_else(|| panic!("missing dimension {name}"));
        assert_eq!(
            actual.len(),
            dimension["len"].as_u64().unwrap() as usize,
            "{name}"
        );
        assert_eq!(
            actual.is_unlimited(),
            dimension["unlimited"].as_bool().unwrap(),
            "{name}"
        );
    }
}

fn assert_native_values(file: &netcrust::File) {
    let expected = native();
    assert_native_dimensions(file, &expected["dimensions"]);
    for index in 0..16 {
        let attribute = format!("metadata_{index:02}");
        let actual = file.attribute(&attribute).expect("dense global attribute");
        assert_eq!(
            actual.as_string(),
            Some(format!("global value {index:02}").as_str())
        );
        let variable = file.variable("t2m").expect("t2m metadata");
        assert_eq!(
            variable
                .attribute(&attribute)
                .expect("dense variable attribute")
                .as_string(),
            Some(format!("variable value {index:02}").as_str())
        );
    }
    for (name, variable) in expected["variables"].as_object().unwrap() {
        assert!(file.has_hdf5_dataset(name), "missing dataset {name}");
        if variable["dtype"] == "|S1" {
            let strings = file
                .read_strings(name)
                .expect("read native character variable");
            let expected_bytes = bytes_from_hex(&variable["bytes_hex"]);
            assert_eq!(
                strings.len(),
                expected_bytes.len(),
                "{name}: fixed-width HDF5 characters"
            );
            let actual: Vec<u8> = strings
                .iter()
                .map(|text| {
                    assert!(text.is_ascii() && text.len() <= 1);
                    text.as_bytes().first().copied().unwrap_or(0)
                })
                .collect();
            assert_eq!(actual, expected_bytes, "{name}");
            let metadata = file.variable(name).expect("character variable metadata");
            assert_eq!(metadata.shape(), shape(&variable["shape"]), "{name}");
        } else {
            let actual = file
                .read_array_f64(name)
                .expect("read native numeric variable");
            assert_eq!(actual.shape(), shape(&variable["shape"]), "{name}");
            let actual_bytes: Vec<u8> = actual
                .values()
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect();
            assert_eq!(
                actual_bytes,
                bytes_from_hex(&variable["f64_le_hex"]),
                "{name}"
            );
        }
    }
}

#[test]
fn disk_and_memory_reads_match_every_native_variable() {
    let disk = netcrust::File::open(source_path()).expect("open real NCO output from disk");
    assert_native_values(&disk);
    let memory = netcrust::File::from_bytes(SOURCE).expect("open real NCO output from bytes");
    assert_native_values(&memory);
}

struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        let path =
            std::env::temp_dir().join(format!("rw-netcdf-dense-{}-{tag}", std::process::id()));
        std::fs::create_dir(&path).expect("create test scratch");
        Self(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn run(arguments: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_rw_netcdf"))
        .args(arguments)
        .output()
        .expect("run rw_netcdf")
}

fn document(output: &Output) -> Value {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).expect("inventory JSON")
}

fn read_metadata(file: netcrust::File) -> netcrust::Result<()> {
    file.dimensions()?;
    file.variables()?;
    file.attributes()?;
    Ok(())
}

#[test]
fn cli_inventory_and_dump_match_every_native_variable() {
    let expected = native();
    let source = source_path();
    let inventory = document(&run(&["inventory", source.to_str().unwrap()]));
    assert_eq!(inventory["metadata"]["mode"], "strict");
    let variables = inventory["variables"].as_array().unwrap();
    assert_eq!(
        variables.len(),
        expected["variables"].as_object().unwrap().len()
    );
    for (name, variable) in expected["variables"].as_object().unwrap() {
        let actual = variables
            .iter()
            .find(|item| item["name"] == name.as_str())
            .unwrap_or_else(|| panic!("missing inventory variable {name}"));
        assert_eq!(actual["shape"], variable["shape"], "{name}");
        assert_eq!(actual["dimensions"], variable["dimensions"], "{name}");
    }
    for dimension in expected["dimensions"].as_array().unwrap() {
        assert!(inventory["dimensions"]
            .as_array()
            .unwrap()
            .contains(dimension));
    }
    assert_eq!(inventory["dimensions"].as_array().unwrap().len(), 3);
    for index in 0..16 {
        assert_eq!(
            inventory["global_attributes"][format!("metadata_{index:02}")],
            format!("global value {index:02}")
        );
    }

    let scratch = Scratch::new("dump");
    let names: Vec<&str> = expected["variables"]
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    let mut arguments = vec![
        "dump",
        "--raw",
        source.to_str().unwrap(),
        scratch.0.to_str().unwrap(),
    ];
    arguments.extend(names.iter().copied());
    let output = run(&arguments);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let metadata: Value = serde_json::from_slice(
        &std::fs::read(scratch.0.join("metadata.json")).expect("dump metadata"),
    )
    .expect("dump metadata JSON");
    for (index, name) in names.iter().enumerate() {
        let variable = &expected["variables"][name];
        let record = &metadata["variables"][index];
        assert_eq!(record["name"], *name);
        assert_eq!(record["shape"], variable["shape"]);
        let expected_bytes = if variable["dtype"] == "|S1" {
            bytes_from_hex(&variable["bytes_hex"])
        } else {
            bytes_from_hex(&variable["f64_le_hex"])
        };
        let actual = std::fs::read(scratch.0.join(record["filename"].as_str().unwrap()))
            .expect("dump bytes");
        assert_eq!(actual, expected_bytes, "{name}");
    }
}

#[test]
fn a_corrupt_fractal_heap_checksum_is_rejected_by_memory_disk_and_cli() {
    let expected = native();
    let offset = expected["corruption_checksum_offset"].as_u64().unwrap() as usize;
    let heap = expected["fractal_heap_offsets"][0].as_u64().unwrap() as usize;
    assert_eq!(&SOURCE[heap..heap + 4], b"FRHP");
    assert_eq!(
        &SOURCE[offset..offset + 4],
        bytes_from_hex(&expected["corruption_checksum_le_hex"])
    );
    let mut bytes = SOURCE.to_vec();
    bytes[offset] ^= 1;
    let memory_error = match netcrust::File::from_bytes(&bytes).and_then(read_metadata) {
        Ok(_) => panic!("memory read accepted a corrupt fractal heap checksum"),
        Err(error) => error.to_string(),
    };
    assert!(memory_error.contains("checksum mismatch"), "{memory_error}");

    let scratch = Scratch::new("corruption");
    let path = scratch.0.join("corrupt.nc4");
    std::fs::write(&path, bytes).expect("write corrupt fixture");
    let disk_error = match netcrust::File::open(&path).and_then(read_metadata) {
        Ok(_) => panic!("disk read accepted a corrupt fractal heap checksum"),
        Err(error) => error.to_string(),
    };
    assert!(disk_error.contains("checksum mismatch"), "{disk_error}");
    let output = run(&["inventory", path.to_str().unwrap()]);
    assert_eq!(output.status.code(), Some(2));
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("checksum mismatch"),
        "{output:?}"
    );
}
