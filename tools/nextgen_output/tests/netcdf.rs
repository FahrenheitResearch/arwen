use rw_nextgen_output::{
    checkpoint,
    netcdf::{self, Attribute, Dataset, Dimension, Variable},
};
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

static SERIAL: AtomicU64 = AtomicU64::new(0);
struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "hydro-netcdf-test-{}-{}",
            std::process::id(),
            SERIAL.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

#[test]
fn checkpoint_preserves_every_primitive_and_float_bits() {
    let scratch = Scratch::new();
    let mut payload = Vec::new();
    let fields: Vec<(&str, Vec<u8>)> = vec![
        ("i8", vec![0x80, 0x7f]),
        ("u8", vec![0, 255]),
        (
            "i16le",
            [i16::MIN.to_le_bytes(), i16::MAX.to_le_bytes()].concat(),
        ),
        (
            "u16le",
            [0u16.to_le_bytes(), u16::MAX.to_le_bytes()].concat(),
        ),
        (
            "i32le",
            [i32::MIN.to_le_bytes(), i32::MAX.to_le_bytes()].concat(),
        ),
        (
            "u32le",
            [0u32.to_le_bytes(), u32::MAX.to_le_bytes()].concat(),
        ),
        (
            "i64le",
            [i64::MIN.to_le_bytes(), i64::MAX.to_le_bytes()].concat(),
        ),
        (
            "u64le",
            [0u64.to_le_bytes(), u64::MAX.to_le_bytes()].concat(),
        ),
        (
            "f32le",
            [0x80000000u32.to_le_bytes(), 0x7fc01234u32.to_le_bytes()].concat(),
        ),
        (
            "f64le",
            [
                0x8000000000000000u64.to_le_bytes(),
                0x7ff8000000001234u64.to_le_bytes(),
            ]
            .concat(),
        ),
    ];
    let mut numeric_payload = fields
        .iter()
        .flat_map(|(_, b)| b.clone())
        .collect::<Vec<_>>();
    numeric_payload.extend_from_slice(&[0, 255]);
    let mut offset = 0;
    let numeric = Dataset {
        format: "netcdf4".into(),
        dimensions: vec![Dimension {
            name: "sample".into(),
            len: 2,
            unlimited: false,
        }],
        attributes: vec![],
        variables: fields
            .iter()
            .map(|(dtype, b)| {
                let var = Variable {
                    name: format!("value_{dtype}"),
                    dtype: (*dtype).into(),
                    dimensions: vec!["sample".into()],
                    attributes: vec![],
                    offset,
                    strings: vec![],
                };
                offset += b.len();
                var
            })
            .chain(std::iter::once(Variable {
                name: "raw_chars".into(),
                dtype: "char".into(),
                dimensions: vec!["sample".into()],
                attributes: vec![],
                offset: numeric_payload.len() - 2,
                strings: vec![],
            }))
            .collect(),
    };
    let numeric_path = scratch.0.join("numeric.nc");
    netcdf::write(&numeric_path, &numeric, &numeric_payload).unwrap();
    assert_eq!(
        netcdf::read(&numeric_path).unwrap(),
        (numeric, numeric_payload)
    );
    let arrays = fields
        .iter()
        .map(|(dtype, bytes)| {
            payload.extend_from_slice(bytes);
            checkpoint::Array {
                name: format!("routing/{dtype}"),
                dtype: (*dtype).into(),
                shape: vec![2],
            }
        })
        .chain(std::iter::once(checkpoint::Array {
            name: "empty".into(),
            dtype: "i64le".into(),
            shape: vec![0],
        }))
        .collect::<Vec<_>>();
    let metadata = serde_json::json!({"clock":3600,"identity":"sealed inventory"});
    let path = scratch.0.join("restart.nc");
    let receipt = checkpoint::write(&path, &metadata, &arrays, &payload).unwrap();
    assert_eq!(receipt["container_backend"], "netcdf-writer (Rust)");
    assert_eq!(&fs::read(&path).unwrap()[..4], b"CDF\x05");
    let (receipt, actual) = checkpoint::read(&path).unwrap();
    assert_eq!(receipt["metadata"], metadata);
    assert_eq!(actual, payload);
    let before = fs::read(&path).unwrap();
    assert!(checkpoint::write(&path, &metadata, &arrays, &payload).is_err());
    assert_eq!(fs::read(&path).unwrap(), before);
    assert_eq!(fs::read_dir(&scratch.0).unwrap().count(), 2);
}

#[test]
fn record_dimension_and_typed_attributes_roundtrip() {
    let scratch = Scratch::new();
    let schema = Dataset {
        format: "cdf5".into(),
        dimensions: vec![
            Dimension {
                name: "time".into(),
                len: 2,
                unlimited: true,
            },
            Dimension {
                name: "reach".into(),
                len: 2,
                unlimited: false,
            },
        ],
        attributes: vec![
            Attribute::text("title", "hydro\0"),
            Attribute::f64("scale", -0.),
        ],
        variables: vec![
            Variable {
                name: "flow".into(),
                dtype: "f32le".into(),
                dimensions: vec!["time".into(), "reach".into()],
                attributes: vec![
                    Attribute::text("units", "m3 s-1"),
                    Attribute::f32("_FillValue", -9999.),
                    Attribute::f64("missing_value", -9999.),
                ],
                offset: 0,
                strings: vec![],
            },
            Variable {
                name: "type".into(),
                dtype: "char".into(),
                dimensions: vec!["reach".into()],
                attributes: vec![],
                offset: 16,
                strings: vec![],
            },
        ],
    };
    let mut payload = [0.25f32, -0., f32::from_bits(0x7fc01234), 125.]
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect::<Vec<_>>();
    payload.extend_from_slice(b"cr");
    let path = scratch.0.join("history.nc");
    netcdf::write(&path, &schema, &payload).unwrap();
    let (actual, bytes) = netcdf::read(&path).unwrap();
    assert_eq!(actual, schema);
    assert_eq!(bytes, payload);
}

#[test]
fn invalid_payload_publishes_nothing() {
    let scratch = Scratch::new();
    let schema = Dataset {
        format: "cdf5".into(),
        dimensions: vec![],
        attributes: vec![],
        variables: vec![Variable {
            name: "flow".into(),
            dtype: "f64le".into(),
            dimensions: vec![],
            attributes: vec![],
            offset: 0,
            strings: vec![],
        }],
    };
    assert!(netcdf::write(&scratch.0.join("history.nc"), &schema, &[0; 7]).is_err());
    assert_eq!(fs::read_dir(&scratch.0).unwrap().count(), 0);
}

#[test]
fn netcdf4_history_preserves_strings_dimensions_and_bits() {
    let scratch = Scratch::new();
    let payload = [1i64.to_le_bytes(), i64::MAX.to_le_bytes()].concat();
    let schema = Dataset {
        format: "netcdf4".into(),
        dimensions: vec![
            Dimension {
                name: "feature_id".into(),
                len: 2,
                unlimited: false,
            },
            Dimension {
                name: "type_strlen".into(),
                len: 2,
                unlimited: false,
            },
        ],
        attributes: vec![Attribute::text("code_version", "\0")],
        variables: vec![
            Variable {
                name: "feature_id".into(),
                dtype: "i64le".into(),
                dimensions: vec!["feature_id".into()],
                attributes: vec![],
                offset: 0,
                strings: vec![],
            },
            Variable {
                name: "type".into(),
                dtype: "string".into(),
                dimensions: vec!["feature_id".into()],
                attributes: vec![Attribute::text("long_name", "Reach type")],
                offset: 16,
                strings: vec!["wb".into(), "é".into()],
            },
        ],
    };
    let path = scratch.0.join("history.nc");
    netcdf::write(&path, &schema, &payload).unwrap();
    if let Some(destination) = std::env::var_os("HYDRO_NETCDF_TEST_ARTIFACT") {
        fs::copy(&path, destination).unwrap();
    }
    let (actual, bytes) = netcdf::read(&path).unwrap();
    assert_eq!(actual, schema);
    assert_eq!(bytes, payload);
}
#[test]
fn complete_hydro_history_contract_and_reference_fixture() {
    use rw_nextgen_output::series;
    let scratch = Scratch::new();
    let mut payload = [
        17i64.to_le_bytes(),
        29i64.to_le_bytes(),
        i64::MAX.to_le_bytes(),
    ]
    .concat();
    let mut offsets = vec![];
    for field in 0..4 {
        offsets.push(payload.len());
        for sample in 0..12 {
            payload.extend_from_slice(
                &(if sample == 0 {
                    -0.0f64
                } else {
                    (field * 12 + sample) as f64 * 0.125
                })
                .to_le_bytes(),
            );
        }
    }
    let plan = series::Plan {
        output_directory: scratch.0.join("products"),
        origin: "2026-10-21 12:00:00".into(),
        interval_seconds: 1200,
        cadence_steps: 1,
        frames: 3,
        capacity: 4,
        reaches: 3,
        dtype: "f64le".into(),
        ids_offset: 0,
        flow_offset: offsets[0],
        velocity_offset: Some(offsets[1]),
        depth_offset: Some(offsets[2]),
        nudge_offset: Some(offsets[3]),
        types: vec!["wb".into(), "r".into(), "wb".into()],
        formats: vec!["troute_netcdf".into()],
        nexus_ids: vec![],
        candidate_ids: vec![],
        provenance: Default::default(),
        parquet: None,
        renderer: None,
    };
    series::write(&plan, &payload).unwrap();
    let (header, raw) = netcdf::read(&plan.output_directory.join("flowveldepth.nc")).unwrap();
    assert_eq!(header.format, "netcdf4");
    assert_eq!(
        header
            .dimensions
            .iter()
            .map(|d| (&*d.name, d.len))
            .collect::<Vec<_>>(),
        [("feature_id", 3), ("time", 3), ("type_strlen", 2)]
    );
    assert_eq!(
        header
            .variables
            .iter()
            .map(|v| &*v.name)
            .collect::<Vec<_>>(),
        [
            "time",
            "feature_id",
            "type",
            "flow",
            "velocity",
            "depth",
            "nudge"
        ]
    );
    assert_eq!(header.variables[2].dtype, "string");
    assert_eq!(header.variables[2].strings, plan.types);
    assert_eq!(
        header
            .attributes
            .iter()
            .find(|a| a.name == "code_version")
            .unwrap()
            .text,
        "\0"
    );
    assert_eq!(raw.len(), 3 * 8 + 3 * 8 + 4 * 9 * 4);
    // Generated by the pre-port netCDF-C 4.9.3 bridge from this same plan/payload.
    let reference =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/netcdf-c-history.nc");
    let (old_header, old_raw) = netcdf::read(&reference).unwrap();
    assert_eq!(header, old_header);
    assert_eq!(raw, old_raw);
    if let Some(dir) = std::env::var_os("HYDRO_NETCDF_REFERENCE_FIXTURE") {
        let dir = PathBuf::from(dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("payload.raw"), &payload).unwrap();
        let mut request = serde_json::json!({"operation":"series-write","plan":plan});
        request["plan"]["output_directory"] = "reference".into();
        fs::write(
            dir.join("request.json"),
            serde_json::to_vec_pretty(&request).unwrap(),
        )
        .unwrap();
        fs::copy(
            plan.output_directory.join("flowveldepth.nc"),
            dir.join("rust-history.nc"),
        )
        .unwrap();
    }
}

#[test]
fn netcdf4_string_heaps_cross_the_collection_index_limit() {
    let scratch = Scratch::new();
    let strings = (0..65540).map(|i| format!("reach-{i}")).collect::<Vec<_>>();
    let schema = Dataset {
        format: "netcdf4".into(),
        dimensions: vec![Dimension {
            name: "reach".into(),
            len: strings.len(),
            unlimited: false,
        }],
        attributes: vec![],
        variables: vec![Variable {
            name: "labels".into(),
            dtype: "string".into(),
            dimensions: vec!["reach".into()],
            attributes: vec![],
            offset: 0,
            strings,
        }],
    };
    let path = scratch.0.join("large-history.nc");
    netcdf::write(&path, &schema, &[]).unwrap();
    assert_eq!(netcdf::read(&path).unwrap(), (schema, vec![]));
    assert_eq!(rw_nextgen_output::rw_nextgen_output_abi_version(), 1);
}
