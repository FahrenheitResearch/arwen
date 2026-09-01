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

    let libm_dependent: std::collections::BTreeSet<String> = golden["libm_dependent_fields"]
        .as_array()
        .map(|names| {
            names
                .iter()
                .map(|value| value.as_str().expect("field names are strings").to_owned())
                .collect()
        })
        .unwrap_or_default();

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
            // A field the mapping produces through a transcendental --
            // a declared derivation, or the sin/cos grid-relative wind
            // rotation -- carries its last bits from the box's libm, so
            // its array digest is a per-box value.  Shape, axes and the
            // missing count stay exact here, and the VALUES are compared
            // against the golden's recorded statistics, within the
            // declared tolerance, in compare_libm_dependent_fields.
            let exact: &[&str] = if libm_dependent.contains(field.as_str()) {
                &["axes", "shape", "missing"]
            } else {
                &["axes", "shape", "missing", "sha256"]
            };
            for key in exact.iter().copied() {
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
        // The canonical frame header is an exact identity of the decode,
        // and two of its members are identities of the BOX: each field
        // descriptor quotes the ABSOLUTE input paths, and every field
        // produced through a TRANSCENDENTAL quotes an array digest whose
        // last bits are the box's libm.  Both showed up as red here, in
        // that order -- first between two worktrees of the same commit
        // (paths), then between the Windows desktop and weather-node-1:
        // three of 38 netCDF arrays at most 3 ULP apart from the two
        // `exp`-based humidity derivations, and all four Lambert wind
        // components from the `sin`/`cos` rotation.  So the digest
        // asserted is the PORTABLE one both engines publish: paths
        // reduced to basenames, libm-dependent digests elided by name,
        // everything else -- grid, vertical coordinates, times, policies,
        // shapes, dtypes, and the exact array digest of every other
        // field -- still exact.
        assert_eq!(
            output["materialization"]["portable_rule"],
            golden["materialization"]["portable_rule"],
            "{name}: the golden was recorded under a different portable rule; \
             re-run extract.py"
        );
        assert_eq!(
            output["materialization"]["frame_header_sha256_portable"],
            golden["materialization"]["frame_header_sha256_portable"],
            "{name}: portable canonical frame header digests"
        );
        // And the derived arrays the portable digest steps around are
        // compared by VALUE, under the tolerance the golden declares --
        // so a derivation that changed formula, units, level order or
        // dependency is still caught, by name.
        compare_libm_dependent_fields(name, &golden, &mapping, &files);
    }
    println!("PASS {name}: {compared} field identities match the Python engine");
}

/// Compare each libm-dependent field against the golden's recorded value.
///
/// Such an array's bytes are not comparable across boxes (its last bits
/// are the box's libm), so the golden records five statistics per field
/// and this asserts them within the golden's declared relative tolerance.
/// Five and not one: a single sum can be cancelled by a sign flip, while
/// the minimum, the maximum, the sum, the sum of magnitudes and the sum
/// of squares together cannot be moved by any rearrangement that leaves
/// the array wrong.  Shape, axes and the non-finite count stay EXACT.
fn compare_libm_dependent_fields(name: &str, golden: &Value, mapping: &Path, files: &[PathBuf]) {
    let Some(expected_frames) = golden["libm_dependent_statistics"].as_array() else {
        panic!(
            "{name}: the golden records no libm-dependent statistics; \
             re-run extract.py to re-measure it"
        );
    };
    let tolerance = golden["libm_relative_tolerance"]
        .as_f64()
        .expect("the golden declares its libm tolerance");

    let mapping_model = mapped_engine::model::Mapping::load(&mapping.display().to_string())
        .expect("the golden names a mapping the engine loads");
    let names: Vec<String> = files.iter().map(|file| file.display().to_string()).collect();
    let collection =
        mapped_engine::engine::decode_collection(&mapping_model, &names, &mut |_| {})
            .expect("the engine decodes the golden's inputs");
    let frames = mapped_engine::frames::materialize_frames(&mapping_model, &collection)
        .expect("the golden's verdict is PASS, so the frames materialize");
    assert_eq!(frames.len(), expected_frames.len(), "{name}: libm-dependent frame count");

    let mut compared = 0usize;
    for (index, expected_frame) in expected_frames.iter().enumerate() {
        let expected_fields = expected_frame
            .as_object()
            .expect("each statistics frame is an object");
        for (field, expected) in expected_fields {
            let observed = frames[index]
                .fields
                .iter()
                .find(|candidate| &candidate.name == field)
                .unwrap_or_else(|| {
                    panic!("{name}: frames[{index}] no longer produces {field}")
                });
            let values = mapped_engine::array::contiguous(&observed.values);
            let finite: Vec<f64> = values.iter().copied().filter(|value| value.is_finite()).collect();
            let shape: Vec<u64> = observed.values.shape().iter().map(|size| *size as u64).collect();
            let expected_shape: Vec<u64> = expected["shape"]
                .as_array()
                .expect("shape is an array")
                .iter()
                .map(|size| size.as_u64().expect("shape entries are counts"))
                .collect();
            assert_eq!(shape, expected_shape, "{name}: frames[{index}].{field}.shape");
            assert_eq!(
                observed.axes,
                expected["axes"]
                    .as_array()
                    .expect("axes is an array")
                    .iter()
                    .map(|axis| axis.as_str().expect("axes are strings").to_owned())
                    .collect::<Vec<String>>(),
                "{name}: frames[{index}].{field}.axes"
            );
            assert_eq!(
                (values.len() - finite.len()) as u64,
                expected["nonfinite"].as_u64().expect("a count"),
                "{name}: frames[{index}].{field}.nonfinite"
            );
            let statistics: [(&str, Option<f64>); 5] = [
                ("minimum", finite.iter().copied().reduce(f64::min)),
                ("maximum", finite.iter().copied().reduce(f64::max)),
                ("sum", Some(finite.iter().sum())),
                ("sum_absolute", Some(finite.iter().map(|value| value.abs()).sum())),
                ("sum_square", Some(finite.iter().map(|value| value * value).sum())),
            ];
            for (statistic, observed_value) in statistics {
                let expected_value = expected[statistic].as_f64();
                match (expected_value, observed_value) {
                    (None, None) => {}
                    (Some(expected_value), Some(observed_value)) => {
                        let difference = (expected_value - observed_value).abs();
                        let scale = expected_value.abs().max(observed_value.abs());
                        assert!(
                            difference <= tolerance * scale,
                            "{name}: frames[{index}].{field}.{statistic} moved beyond the \
                             declared {tolerance:e} relative tolerance: golden \
                             {expected_value:e} vs engine {observed_value:e}"
                        );
                    }
                    (expected_value, observed_value) => panic!(
                        "{name}: frames[{index}].{field}.{statistic} presence disagrees: \
                         golden {expected_value:?} vs engine {observed_value:?}"
                    ),
                }
            }
            compared += 1;
        }
    }
    println!("PASS {name}: {compared} libm-dependent value identities within {tolerance:e}");
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

/// The two decodes of the same bytes must write the SAME frameset.
///
/// The engine decodes a valid time, writes it and drops it before it
/// reads the next -- the whole remedy for a preparation whose host
/// memory grew with the number of forcing times.  A remedy on the decode
/// path that moved a number would be the worst kind of defect, so the
/// gate is byte equality of the written frameset (`frames.json` and
/// every byte of `frames.f64`) between:
///
///   * the WHOLE-SERIES decode, assembled at once and carved per valid
///     time as the writer asks for it, and
///   * the STREAMED decode, which reads one valid time's messages when
///     that valid time is written.
///
/// Both arms go through the shipped writer, so this compares the two
/// decode strategies and nothing else.
fn compare_streamed_against_whole(
    name: &str,
    directory: PathBuf,
    prefix: &str,
    suffix: &str,
    limit: Option<usize>,
) {
    let golden_path = goldens().join(format!("{name}.json"));
    let golden: Value = serde_json::from_slice(
        &std::fs::read(&golden_path).expect("golden is committed beside this test"),
    )
    .expect("golden is JSON");
    let files = inputs(&directory, prefix, suffix, limit);
    let expected: Vec<&str> = golden["input_names"]
        .as_array()
        .expect("golden names its inputs")
        .iter()
        .map(|value| value.as_str().expect("input names are strings"))
        .collect();
    let observed: Vec<String> = files
        .iter()
        .map(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .unwrap_or_default()
                .to_owned()
        })
        .collect();
    if observed != expected {
        println!(
            "SKIP {name}: the staged inputs are not present at {}",
            directory.display()
        );
        return;
    }
    let mapping_path =
        repository().join(golden["mapping"].as_str().expect("golden names its mapping"));
    let mapping = mapped_engine::model::Mapping::load(&mapping_path.display().to_string())
        .expect("the golden names a mapping the engine loads");
    let names: Vec<String> = files.iter().map(|file| file.display().to_string()).collect();
    let digests = mapped_engine::engine::input_digests(&names).expect("inputs hash");

    let scratch = std::env::temp_dir().join(format!("gpuwm-mapped-engine-stream-{name}"));
    let _ = std::fs::remove_dir_all(&scratch);
    let whole_out = scratch.join("whole");
    let streamed_out = scratch.join("streamed");

    let collection = mapped_engine::engine::decode_collection(&mapping, &names, &mut |_| {})
        .unwrap_or_else(|refusal| panic!("{name}: whole decode refused: {refusal}"));
    let series = mapped_engine::frames::SeriesSummary {
        source_cycles: collection.source_cycles.clone(),
        grid_fingerprint: collection.grid_fingerprint.clone(),
    };
    let mut carved = collection;
    mapped_engine::frames::write_frameset(&whole_out, &mapping, &series, &digests, |key| {
        Ok(mapped_engine::engine::carve_valid_time(&mut carved, key))
    })
    .unwrap_or_else(|refusal| panic!("{name}: whole frameset refused: {refusal}"));

    let mut stream = mapped_engine::engine::DecodeStream::open(&mapping, &names, &mut |_| {})
        .unwrap_or_else(|refusal| panic!("{name}: streamed decode refused: {refusal}"));
    let streamed_series = stream.summary().clone();
    mapped_engine::frames::write_frameset(
        &streamed_out,
        &mapping,
        &streamed_series,
        &digests,
        |key| stream.slice(key),
    )
    .unwrap_or_else(|refusal| panic!("{name}: streamed frameset refused: {refusal}"));

    for file in ["frames.json", "frames.f64"] {
        let left = std::fs::read(whole_out.join(file)).expect("the whole arm wrote it");
        let right = std::fs::read(streamed_out.join(file)).expect("the streamed arm wrote it");
        assert_eq!(
            left.len(),
            right.len(),
            "{name}: {file} length differs between the whole and streamed decodes"
        );
        assert!(
            left == right,
            "{name}: {file} differs between the whole and streamed decodes"
        );
    }
    let document: Value = serde_json::from_slice(
        &std::fs::read(streamed_out.join("frames.json")).expect("frames.json"),
    )
    .expect("frames.json is JSON");
    let frames = document["frames"]
        .as_array()
        .expect("frames is an array")
        .len();
    assert!(
        frames > 1,
        "{name}: a one-frame case cannot prove per-valid-time streaming"
    );
    println!("PASS {name}: {frames} frames byte-identical, streamed against whole");
    let _ = std::fs::remove_dir_all(&scratch);
}

/// Multi-object GRIB2 on a declared Lambert grid, three valid times.
#[test]
fn streamed_grib2_writes_the_same_frameset_as_the_whole_decode() {
    compare_streamed_against_whole(
        "lambert-awips",
        staging().join("rap"),
        "rap.t00z.awip32f0",
        ".grib2",
        Some(3),
    );
}

/// The same equality one level lower, on the DECODED COLLECTION.
///
/// For a mapping whose frames cannot materialize alone -- a composed
/// source's primary carries no terrain, so the bare decode is
/// deliberately incomplete -- the frameset cannot be the subject, and
/// the decode itself is compared instead: every key, every array, every
/// missing count, every source reference, per valid time.
fn compare_streamed_collections(
    name: &str,
    directory: PathBuf,
    prefix: &str,
    suffix: &str,
    limit: Option<usize>,
) {
    let golden_path = goldens().join(format!("{name}.json"));
    let golden: Value = serde_json::from_slice(
        &std::fs::read(&golden_path).expect("golden is committed beside this test"),
    )
    .expect("golden is JSON");
    let files = inputs(&directory, prefix, suffix, limit);
    let expected: Vec<&str> = golden["input_names"]
        .as_array()
        .expect("golden names its inputs")
        .iter()
        .map(|value| value.as_str().expect("input names are strings"))
        .collect();
    let observed: Vec<String> = files
        .iter()
        .map(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .unwrap_or_default()
                .to_owned()
        })
        .collect();
    if observed != expected {
        println!(
            "SKIP {name}: the staged inputs are not present at {}",
            directory.display()
        );
        return;
    }
    let mapping_path =
        repository().join(golden["mapping"].as_str().expect("golden names its mapping"));
    let mapping = mapped_engine::model::Mapping::load(&mapping_path.display().to_string())
        .expect("the golden names a mapping the engine loads");
    let names: Vec<String> = files.iter().map(|file| file.display().to_string()).collect();

    let whole = mapped_engine::engine::decode_collection(&mapping, &names, &mut |_| {})
        .unwrap_or_else(|refusal| panic!("{name}: whole decode refused: {refusal}"));
    let mut stream = mapped_engine::engine::DecodeStream::open(&mapping, &names, &mut |_| {})
        .unwrap_or_else(|refusal| panic!("{name}: streamed decode refused: {refusal}"));
    let keys: Vec<(chrono::NaiveDateTime, Option<String>)> = stream.keys().to_vec();
    assert_eq!(
        keys,
        whole.source_cycles.keys().cloned().collect::<Vec<_>>(),
        "{name}: the streamed and whole decodes disagree about the forcing axis"
    );
    let mut compared = 0usize;
    for key in &keys {
        let slice = stream
            .slice(key)
            .unwrap_or_else(|refusal| panic!("{name}: streamed slice refused: {refusal}"));
        assert_eq!(
            slice.source_cycles.get(key),
            whole.source_cycles.get(key),
            "{name}: source cycle at {key:?}"
        );
        assert_eq!(
            slice.latitude, whole.latitude,
            "{name}: latitude axis at {key:?}"
        );
        assert_eq!(
            slice.longitude, whole.longitude,
            "{name}: longitude axis at {key:?}"
        );
        assert_eq!(
            slice.vertical_values, whole.vertical_values,
            "{name}: vertical ladder at {key:?}"
        );
        assert_eq!(
            slice.grid_fingerprint, whole.grid_fingerprint,
            "{name}: grid fingerprint at {key:?}"
        );
        let mine: Vec<&String> = whole
            .direct
            .keys()
            .filter(|(time, member, _field)| (*time, member.clone()) == *key)
            .map(|(_time, _member, field)| field)
            .collect();
        let streamed: Vec<&String> = slice
            .direct
            .keys()
            .filter(|(time, member, _field)| (*time, member.clone()) == *key)
            .map(|(_time, _member, field)| field)
            .collect();
        assert_eq!(mine, streamed, "{name}: field inventory at {key:?}");
        for field in mine {
            let entry = (key.0, key.1.clone(), field.clone());
            let left = &whole.direct[&entry];
            let right = &slice.direct[&entry];
            assert_eq!(left.axes, right.axes, "{name}: {field} axes at {key:?}");
            assert_eq!(
                left.source_cycle, right.source_cycle,
                "{name}: {field} source cycle at {key:?}"
            );
            assert_eq!(
                left.missing_count, right.missing_count,
                "{name}: {field} missing count at {key:?}"
            );
            assert_eq!(
                left.references, right.references,
                "{name}: {field} source references at {key:?}"
            );
            assert_eq!(
                mapped_engine::digest::array_sha256(
                    left.values.shape(),
                    &mapped_engine::array::contiguous(&left.values)
                ),
                mapped_engine::digest::array_sha256(
                    right.values.shape(),
                    &mapped_engine::array::contiguous(&right.values)
                ),
                "{name}: {field} array bytes at {key:?}"
            );
            compared += 1;
        }
    }
    assert!(
        keys.len() > 1,
        "{name}: a one-time case cannot prove per-valid-time streaming"
    );
    println!("PASS {name}: {compared} decoded field identities, streamed against whole");
}

/// The same equality on a mapping that declares a `cycle_invariant`
/// field: one broadcast record answers every valid time, so a streamed
/// decode that read only its own valid time's messages would drop it
/// from every frame but one.
#[test]
fn streamed_cycle_invariant_grib2_decodes_the_same_collection() {
    compare_streamed_collections(
        "icon-eu",
        staging().join("icon").join("eu-regular-latlon"),
        "",
        ".grib2.bz2",
        None,
    );
}
