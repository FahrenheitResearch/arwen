#![cfg(feature = "rewrite")]

use netcdf_writer::{AttrValue, NcFormat, NcType, NcWriter, Schema, VarData};
use std::path::Path;
use std::process::Command;

fn file(path: &Path, name: &str, width: usize, values: &[f32]) {
    let mut schema = Schema::new(NcFormat::Offset64);
    let time = schema.def_dim("Time", 0, true).unwrap();
    let x = schema.def_dim("x", width, false).unwrap();
    schema
        .put_global_attr("TITLE", AttrValue::Text("history".into()))
        .unwrap();
    let var = schema.def_var(name, NcType::Float, &[time, x]).unwrap();
    schema
        .put_var_attr(var, "units", AttrValue::Text("K".into()))
        .unwrap();
    let mut writer = NcWriter::create(path, schema).unwrap();
    writer.write_record(0, var, VarData::F32(values)).unwrap();
    writer.finish().unwrap();
}

#[test]
fn copy_static_records_keeps_history_values_and_refuses_replacement_or_wrong_grid() {
    let root = std::env::temp_dir().join(format!("copy-fields-{}", std::process::id()));
    std::fs::create_dir(&root).unwrap();
    let history = root.join("history.nc");
    let template = root.join("template.nc");
    let bad = root.join("bad-grid.nc");
    let output = root.join("ready.nc");
    file(&history, "T", 2, &[301.25, -1.5]);
    file(&template, "STATIC", 2, &[7.0, 8.0]);
    file(&bad, "STATIC", 3, &[7.0, 8.0, 9.0]);
    let invoke = |source: &Path, target: &Path, names: &str| {
        Command::new(env!("CARGO_BIN_EXE_nc_rewrite"))
            .arg(&history)
            .arg(target)
            .arg("--fields-from")
            .arg(source)
            .arg(names)
            .output()
            .unwrap()
    };
    let result = invoke(&template, &output, "STATIC");
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let reader = netcdf_reader::NcFile::open(&output).unwrap();
    let actual = reader.read_variable::<f32>("T").unwrap();
    assert_eq!(actual.as_slice().unwrap(), &[301.25, -1.5]);
    let actual = reader.read_variable::<f32>("STATIC").unwrap();
    assert_eq!(actual.as_slice().unwrap(), &[7.0, 8.0]);
    drop(reader);
    let refused = root.join("refused.nc");
    let result = invoke(&history, &refused, "T");
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("must not overwrite"));
    assert!(!refused.exists());
    let result = invoke(&bad, &refused, "STATIC");
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("incompatible dimension"));
    assert!(!refused.exists());
    let result = invoke(&template, &refused, "STATIC,STATIC");
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("duplicate template field"));
    assert!(!refused.exists());
    for path in [history, template, bad, output] {
        std::fs::remove_file(path).unwrap();
    }
    std::fs::remove_dir(root).unwrap();
}
