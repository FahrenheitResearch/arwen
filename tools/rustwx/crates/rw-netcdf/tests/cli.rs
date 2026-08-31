//! The front door itself: argument parsing, flag filtering, and the
//! inventory document, exercised through the real binary.
//!
//! `gpuwm.mapped_source` drives `rw_netcdf` as a subprocess, so the
//! argv contract IS the product interface: a flag that stops being
//! filtered out of the positionals, or a switch whose sense inverts,
//! changes every NetCDF initial condition read through it without
//! failing a single in-process test.  These tests run the executable
//! the way Python does and read what it wrote.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_rw_netcdf")
}

fn run(args: &[&str]) -> Output {
    Command::new(binary())
        .args(args)
        .output()
        .expect("spawn rw_netcdf")
}

fn stdout(output: &Output) -> String {
    String::from_utf8(output.stdout.clone()).expect("stdout is UTF-8")
}

fn stderr(output: &Output) -> String {
    String::from_utf8(output.stderr.clone()).expect("stderr is UTF-8")
}

fn fixture(name: &str) -> String {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
        .to_string_lossy()
        .into_owned()
}

/// A scratch directory unique to one test, wiped at entry.
fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir()
        .join(format!("rw-netcdf-cli-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

/// The classic packed fixture, built by the workspace's own writer --
/// same layout as the unit suite's: stored shorts
/// [-32768 (_FillValue), -32767 (missing_value), 4, 6] with
/// scale_factor 0.5 and add_offset 100.
fn write_packed_classic(path: &Path) {
    use netcdf_writer::{AttrValue, NcFormat, NcType, NcWriter, Schema, VarData};
    let mut schema = Schema::new(NcFormat::Classic);
    let x = schema.def_dim("x", 4, false).expect("dim");
    schema
        .put_global_attr("title", AttrValue::Text("cli fixture".into()))
        .expect("gattr");
    let packed = schema.def_var("packed", NcType::Short, &[x]).expect("var");
    schema
        .put_var_attr(packed, "scale_factor", AttrValue::Doubles(vec![0.5]))
        .expect("attr");
    schema
        .put_var_attr(packed, "add_offset", AttrValue::Doubles(vec![100.0]))
        .expect("attr");
    schema
        .put_var_attr(packed, "_FillValue", AttrValue::Shorts(vec![-32768]))
        .expect("attr");
    schema
        .put_var_attr(packed, "missing_value", AttrValue::Shorts(vec![-32767]))
        .expect("attr");
    let mut writer = NcWriter::create(path, schema).expect("create");
    writer
        .write_var(packed, VarData::I16(&[-32768, -32767, 4, 6]))
        .expect("write");
    writer.finish().expect("finish");
}

fn read_f64_plane(path: &Path) -> Vec<f64> {
    let bytes = std::fs::read(path).expect("read plane");
    bytes
        .chunks_exact(8)
        .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

#[test]
fn no_arguments_is_usage_on_stderr_and_exit_2() {
    let output = run(&[]);
    assert_eq!(output.status.code(), Some(2));
    assert!(stderr(&output).contains("usage:"), "{output:?}");
}

#[test]
fn abi_prints_both_schema_markers() {
    let output = run(&["--abi"]);
    assert_eq!(output.status.code(), Some(0));
    let text = stdout(&output);
    assert!(text.contains("gpuwm-rw-netcdf-inventory-v1"), "{text}");
    assert!(text.contains("gpuwm-rw-netcdf-dump-v1"), "{text}");
}

#[test]
fn help_prints_usage_on_stdout() {
    let output = run(&["--help"]);
    assert_eq!(output.status.code(), Some(0));
    assert!(stdout(&output).contains("usage:"), "{output:?}");
}

#[test]
fn unknown_subcommands_are_refused_by_name() {
    let output = run(&["decode"]);
    assert_eq!(output.status.code(), Some(2));
    let text = stderr(&output);
    assert!(text.contains("unknown subcommand"), "{text}");
    assert!(text.contains("decode"), "{text}");
}

#[test]
fn inventory_takes_exactly_one_file() {
    for arguments in [vec!["inventory"], vec!["inventory", "a.nc", "b.nc"]] {
        let output = run(&arguments);
        assert_eq!(output.status.code(), Some(2), "{arguments:?}");
        assert!(
            stderr(&output).contains("exactly one FILE"),
            "{arguments:?}: {output:?}"
        );
    }
}

#[test]
fn inventory_reports_the_whole_document() {
    let dir = scratch("inventory");
    let source = dir.join("packed.nc");
    write_packed_classic(&source);
    let output = run(&["inventory", source.to_str().expect("path")]);
    assert_eq!(output.status.code(), Some(0), "{output:?}");
    let document: serde_json::Value =
        serde_json::from_str(&stdout(&output)).expect("inventory JSON");

    assert_eq!(document["schema"], "gpuwm-rw-netcdf-inventory-v1");
    assert_eq!(document["metadata"]["mode"], "strict");
    assert_eq!(
        document["dimensions"],
        serde_json::json!([{"name": "x", "len": 4, "unlimited": false}])
    );
    assert_eq!(document["global_attributes"]["title"], "cli fixture");
    let variables = document["variables"].as_array().expect("variables");
    assert_eq!(variables.len(), 1);
    let packed = &variables[0];
    assert_eq!(packed["name"], "packed");
    assert_eq!(packed["dimensions"], serde_json::json!(["x"]));
    assert_eq!(packed["shape"], serde_json::json!([4]));
    assert_eq!(packed["attributes"]["scale_factor"], 0.5);
    assert_eq!(packed["attributes"]["add_offset"], 100.0);
    assert_eq!(packed["attributes"]["_FillValue"], -32768.0);
    let _ = std::fs::remove_dir_all(&dir);
}

/// The provenance travels through the CLI too: a NetCDF-4 file that
/// strict reconstruction refuses arrives marked size-inferred, with the
/// strict error and the length-collision confession beside it.
#[test]
fn inventory_confesses_size_inferred_metadata() {
    let output = run(&["inventory", &fixture("mixed.nc4")]);
    assert_eq!(output.status.code(), Some(0), "{output:?}");
    let document: serde_json::Value =
        serde_json::from_str(&stdout(&output)).expect("inventory JSON");
    assert_eq!(document["metadata"]["mode"], "size-inferred");
    assert!(
        document["metadata"]["strict_error"]
            .as_str()
            .expect("strict_error")
            .contains("DIMENSION_LIST"),
        "{document}"
    );
    assert_eq!(document["metadata"]["dimension_lengths_ambiguous"], true);
}

#[test]
fn inventory_refuses_a_missing_file_with_exit_2() {
    let output = run(&["inventory", "no-such-file.nc"]);
    assert_eq!(output.status.code(), Some(2));
    assert!(stderr(&output).contains("cannot open"), "{output:?}");
}

#[test]
fn dump_requires_file_output_dir_and_a_variable() {
    let dir = scratch("dump-argc");
    let source = dir.join("packed.nc");
    write_packed_classic(&source);
    let out = dir.join("out");
    let arguments = [
        "dump",
        source.to_str().expect("path"),
        out.to_str().expect("path"),
    ];
    let output = run(&arguments);
    assert_eq!(output.status.code(), Some(2), "{output:?}");
    assert!(
        stderr(&output).contains("at least one VARIABLE"),
        "{output:?}"
    );
    assert!(
        !out.join("metadata.json").exists(),
        "a refused dump must write nothing"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_bare_dump_masks_and_scales_by_default() {
    let dir = scratch("dump-default");
    let source = dir.join("packed.nc");
    write_packed_classic(&source);
    let out = dir.join("out");
    let output = run(&[
        "dump",
        source.to_str().expect("path"),
        out.to_str().expect("path"),
        "packed",
    ]);
    assert_eq!(output.status.code(), Some(0), "{output:?}");
    let values = read_f64_plane(&out.join("0000.f64"));
    assert!(values[0].is_nan() && values[1].is_nan(), "{values:?}");
    assert_eq!(&values[2..], &[102.0, 103.0]);
    let _ = std::fs::remove_dir_all(&dir);
}

/// `--raw` is netCDF4's set_auto_maskandscale(False): the stored bytes
/// survive untouched.  The flag must also be FILTERED out of the
/// positionals wherever it appears.
#[test]
fn dump_raw_preserves_stored_sentinels() {
    let dir = scratch("dump-raw");
    let source = dir.join("packed.nc");
    write_packed_classic(&source);
    let out = dir.join("out");
    let output = run(&[
        "dump",
        "--raw",
        source.to_str().expect("path"),
        out.to_str().expect("path"),
        "packed",
    ]);
    assert_eq!(output.status.code(), Some(0), "{output:?}");
    assert_eq!(
        read_f64_plane(&out.join("0000.f64")),
        &[-32768.0, -32767.0, 4.0, 6.0]
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// `--no-mask` is set_auto_mask(False): sentinels survive but arrive
/// SCALED like every other element, exactly as netCDF4-python does it.
#[test]
fn dump_no_mask_scales_the_surviving_sentinels() {
    let dir = scratch("dump-nomask");
    let source = dir.join("packed.nc");
    write_packed_classic(&source);
    let out = dir.join("out");
    let output = run(&[
        "dump",
        source.to_str().expect("path"),
        out.to_str().expect("path"),
        "packed",
        "--no-mask",
    ]);
    assert_eq!(output.status.code(), Some(0), "{output:?}");
    assert_eq!(
        read_f64_plane(&out.join("0000.f64")),
        &[-16284.0, -16283.5, 102.0, 103.0]
    );
    let _ = std::fs::remove_dir_all(&dir);
}
