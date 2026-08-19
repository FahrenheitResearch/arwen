//! Crate goldens: the engine against identities the REAL Python engine
//! measured on REAL bytes.
//!
//! Each golden in `tests/goldens/` was produced by
//! `gpuwm.mapped_source.inspect_mapped_source` (see `extract.py`), so a
//! green run here means the two engines agree on the sha256 of every
//! field's array bytes, on the shapes and axes, on the missing counts, on
//! the grid fingerprint, and — where the frames materialize — on the
//! canonical frame header digest.
//!
//! These tests read bytes that are not part of the repository (staged
//! agency objects under the model-gauntlet staging tree).  A machine
//! without the staging tree SKIPS them with a printed reason rather than
//! failing: a missing sample is an environment fact, and turning it into a
//! red suite would train the next reader to ignore red.  The one case
//! whose sample lives beside the golden never skips.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde_json::Value;

fn repository() -> PathBuf {
    // tests/ -> mapped-engine/ -> crates/ -> rw_wps/ -> tools/ -> root
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(4)
        .expect("the crate sits five levels below the repository root")
        .to_path_buf()
}

/// This account's home directory, on either platform.
///
/// The staging default is COMPOSED from it rather than spelled out: a
/// written-out default is one developer's absolute path, and the
/// release snapshot's machine-path gate refuses to build a tree that
/// ships one (`tests/test_release_snapshot_machine_paths.py`).
fn home() -> PathBuf {
    std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .map(PathBuf::from)
        .unwrap_or_default()
}

fn staging() -> PathBuf {
    std::env::var("GPUWM_MODEL_GAUNTLET_STAGING")
        .map(PathBuf::from)
        .unwrap_or_else(|_| home().join("gpuwm-model-gauntlet-staging"))
}

fn goldens() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("goldens")
}

/// The inputs a case decodes, mirroring `extract.py`'s selection exactly.
fn inputs(directory: &Path, prefix: &str, suffix: &str, limit: Option<usize>) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(directory) else {
        return Vec::new();
    };
    let mut files: Vec<PathBuf> = entries
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            let name = path.file_name().and_then(|name| name.to_str()).unwrap_or("");
            name.starts_with(prefix) && name.ends_with(suffix)
        })
        .collect();
    files.sort();
    if let Some(limit) = limit {
        files.truncate(limit);
    }
    files
}

fn run_case(name: &str, directory: PathBuf, prefix: &str, suffix: &str, limit: Option<usize>) {
    let golden_path = goldens().join(format!("{name}.json"));
    let golden: Value = serde_json::from_slice(
        &std::fs::read(&golden_path).expect("golden is committed beside this test"),
    )
    .expect("golden is JSON");

    let files = inputs(&directory, prefix, suffix, limit);
    let expected_names: Vec<&str> = golden["input_names"]
        .as_array()
        .expect("golden names its inputs")
        .iter()
        .map(|value| value.as_str().expect("input names are strings"))
        .collect();
    let observed_names: Vec<String> = files
        .iter()
        .map(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .unwrap_or_default()
                .to_owned()
        })
        .collect();
    if observed_names != expected_names {
        println!(
            "SKIP {name}: the staged inputs this golden was measured on are not \
             present at {} ({} of {} found)",
            directory.display(),
            observed_names.len(),
            expected_names.len()
        );
        return;
    }

    let scratch = std::env::temp_dir().join(format!("gpuwm-mapped-engine-golden-{name}"));
    std::fs::create_dir_all(&scratch).expect("scratch directory");
    let listing = scratch.join("inputs.txt");
    let body: Vec<String> = files
        .iter()
        .map(|path| path.display().to_string())
        .collect();
    std::fs::write(&listing, body.join("\n") + "\n").expect("input list");

    let mapping = repository().join(
        golden["mapping"]
            .as_str()
            .expect("golden names its mapping"),
    );
    let output = mapped_engine::engine::run_inspect(
        &mapped_engine::engine::Invocation {
            subcommand: "inspect".to_owned(),
            mapping: mapping.display().to_string(),
            input_list: listing.display().to_string(),
            ..Default::default()
        },
        &mut |_event| {},
    )
    .unwrap_or_else(|refusal| panic!("{name}: engine refused: {refusal}"));

    assert_eq!(
        output["grid"]["fingerprint"], golden["grid"]["fingerprint"],
        "{name}: grid fingerprint"
    );
    for key in ["nx", "ny", "vertical_count"] {
        assert_eq!(output["grid"][key], golden["grid"][key], "{name}: grid.{key}");
    }
    assert_eq!(output["status"], golden["status"], "{name}: status");

    let expected_frames = golden["frames"].as_array().expect("frames");
    let observed_frames = output["frames"].as_array().expect("frames");
    assert_eq!(
        observed_frames.len(),
        expected_frames.len(),
        "{name}: frame count"
    );
    let mut compared = 0usize;
    for (index, (expected, observed)) in expected_frames.iter().zip(observed_frames).enumerate() {
        for key in [
            "valid_time",
            "source_cycle",
            "member",
            "decoded_direct_fields",
            "unresolved_direct_fields",
        ] {
            assert_eq!(observed[key], expected[key], "{name}: frames[{index}].{key}");
        }
        let expected_fields: &serde_json::Map<String, Value> =
            expected["fields"].as_object().expect("fields object");
        let observed_fields: &serde_json::Map<String, Value> =
            observed["fields"].as_object().expect("fields object");
        let mut differences: BTreeMap<String, (Value, Value)> = BTreeMap::new();
        for (field, expected_field) in expected_fields {
            let observed_field = observed_fields
                .get(field)
                .unwrap_or_else(|| panic!("{name}: frames[{index}] lacks {field}"));
            for key in ["axes", "shape", "missing", "sha256"] {
                if observed_field[key] != expected_field[key] {
                    differences.insert(
                        format!("{field}.{key}"),
                        (expected_field[key].clone(), observed_field[key].clone()),
                    );
                }
            }
            compared += 1;
        }
        assert!(
            differences.is_empty(),
            "{name}: frames[{index}] differs from the Python engine: {differences:?}"
        );
    }
    assert_eq!(
        output["materialization"]["verdict"], golden["materialization"]["verdict"],
        "{name}: materialization verdict"
    );
    if golden["materialization"]["verdict"] == "PASS" {
        // The canonical frame header quotes each field's source references,
        // and those are ABSOLUTE input paths -- so a raw header digest is
        // an identity of the machine that measured it, not of the decode.
        // The two staged GRIB cases hide that (their inputs sit at a fixed
        // path outside the repository), but this case's sample lives beside
        // the golden and therefore moves with the checkout: the digest
        // differed between two worktrees of the SAME commit while every
        // field array matched to the byte.  Both sides are compared under
        // the repository's one mask instead -- input paths reduced to
        // basenames, per tests/test_mapped_engine_parity.py -- so the
        // assertion stays exact on every other member of the header.
        assert_eq!(
            masked_header_digests(&mapping, &files),
            golden["materialization"]["frame_header_sha256_masked"]
                .as_array()
                .expect("golden records masked frame header digests")
                .iter()
                .map(|value| value.as_str().expect("digests are strings").to_owned())
                .collect::<Vec<String>>(),
            "{name}: canonical frame header digests (input paths masked)"
        );
    }
    println!("PASS {name}: {compared} field identities match the Python engine");
}

/// Replace every occurrence of an input's absolute path, wherever it
/// appears in a JSON document, with that input's file name.
fn mask_input_paths(value: &Value, files: &[PathBuf]) -> Value {
    match value {
        Value::String(text) => {
            let mut masked = text.clone();
            for file in files {
                let absolute = file.display().to_string();
                let Some(name) = file.file_name().and_then(|name| name.to_str()) else {
                    continue;
                };
                masked = masked.replace(&absolute, name);
            }
            Value::String(masked)
        }
        Value::Array(items) => Value::Array(
            items.iter().map(|item| mask_input_paths(item, files)).collect(),
        ),
        Value::Object(members) => Value::Object(
            members
                .iter()
                .map(|(key, item)| (key.clone(), mask_input_paths(item, files)))
                .collect(),
        ),
        other => other.clone(),
    }
}

/// The engine's own canonical frame headers, digested under the mask.
///
/// This decodes in process rather than through `run_decode` on purpose:
/// the headers are what is being compared, and writing the `frames.f64`
/// stream to get at them would cost 700 MB on the Lambert case for
/// numbers already in memory.
fn masked_header_digests(mapping: &Path, files: &[PathBuf]) -> Vec<String> {
    let mapping = mapped_engine::model::Mapping::load(&mapping.display().to_string())
        .expect("the golden names a mapping the engine loads");
    let names: Vec<String> = files.iter().map(|file| file.display().to_string()).collect();
    let collection = mapped_engine::engine::decode_collection(&mapping, &names, &mut |_| {})
        .expect("the engine decodes the golden's inputs");
    mapped_engine::frames::materialize_frames(&mapping, &collection)
        .expect("the golden's verdict is PASS, so the frames materialize")
        .iter()
        .map(|frame| {
            mapped_engine::digest::bytes_sha256(
                mapped_engine::engine::canonical_json(&mask_input_paths(
                    &frame.header,
                    files,
                ))
                .as_bytes(),
            )
        })
        .collect()
}

/// DWD regular-lat-lon GRIB2, bzip2-wrapped: acquisition codec, GDT-0
/// frame export, longitude unwrap, declared level ladder.
#[test]
fn regular_latlon_grib2_matches_the_python_engine() {
    run_case(
        "icon-eu",
        staging().join("icon").join("eu-regular-latlon"),
        "",
        ".grib2.bz2",
        None,
    );
}

/// NCEP AWIPS-grid GRIB2 on a declared Lambert grid: projected axes, the
/// declared-grid octet cross-check, JPEG2000 unpacking, grid-relative wind
/// rotation, the derivation catalog, and the canonical frame header.
#[test]
fn declared_lambert_grib2_matches_the_python_engine() {
    run_case(
        "lambert-awips",
        staging().join("rap"),
        "rap.t00z.awip32f0",
        ".grib2",
        Some(3),
    );
}

/// NetCDF through gpuwm's own checked-in NetCDF mapping authority: CF
/// decode, selector resolution, the level ladder, the soil selector stack,
/// the humidity derivations, and the canonical frame header.  Its sample
/// lives beside the golden, so this case never skips.
#[test]
fn netcdf_matches_the_python_engine() {
    run_case(
        "netcdf-pressure-level",
        goldens(),
        "netcdf-pressure-level-sample",
        ".nc",
        None,
    );
}
