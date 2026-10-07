//! The DA sheet modes (`rw_compare --mode increment|spread|stamp|omb`)
//! through the real binary, on small wrfout-shaped files whose values are
//! known, so every number a receipt reports can be checked by hand.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use netcdf_writer::{AttrValue, NcFormat, NcType, NcWriter, Schema, VarData};
use serde_json::Value;

const NX: usize = 24;
const NY: usize = 18;
const NZ: usize = 4;
const TERRAIN_M: f32 = 320.0;
const WIDTH: u32 = 400;
const HEIGHT: u32 = 300;

struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        let root = std::env::temp_dir().join(format!("rw-da-sheets-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        Self(root)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The values one fixture file carries, as functions of `(level, y, x)`.
struct Member {
    theta: Box<dyn Fn(usize, usize, usize) -> f32>,
    qv: Box<dyn Fn(usize, usize, usize) -> f32>,
    hydrometeor: Box<dyn Fn(usize, usize, usize) -> f32>,
    refl: Box<dyn Fn(usize, usize, usize) -> f32>,
    heights: bool,
}

impl Member {
    fn constant(qv: f32, refl: f32) -> Self {
        Self {
            theta: Box::new(|_, _, _| 10.0),
            qv: Box::new(move |_, _, _| qv),
            hydrometeor: Box::new(|_, _, _| 0.0),
            refl: Box::new(move |level, _, _| if level == 1 { refl } else { refl - 20.0 }),
            heights: true,
        }
    }
}

/// A wrfout-shaped file at 2026-08-19 01Z from a 00Z start: Lambert
/// metadata, mass levels 500, 1500, 2500 and 3500 m above 320 m terrain.
fn write_member(path: &Path, member: &Member) {
    let cells = NX * NY;
    let mut schema = Schema::new(NcFormat::Offset64);
    let time = schema.def_dim("Time", 0, true).unwrap();
    let strlen = schema.def_dim("DateStrLen", 19, false).unwrap();
    let bt = schema.def_dim("bottom_top", NZ, false).unwrap();
    let bts = schema.def_dim("bottom_top_stag", NZ + 1, false).unwrap();
    let sn = schema.def_dim("south_north", NY, false).unwrap();
    let we = schema.def_dim("west_east", NX, false).unwrap();
    for (name, value) in [("START_DATE", "2026-08-19_00:00:00"), ("SIMULATION_START_DATE", "2026-08-19_00:00:00")] {
        schema.put_global_attr(name, AttrValue::Text(value.into())).unwrap();
    }
    for (name, value) in [("MAP_PROJ", 1i32), ("GRID_ID", 1)] {
        schema.put_global_attr(name, AttrValue::Ints(vec![value])).unwrap();
    }
    for (name, value) in [("DX", 3000.0f32), ("DY", 3000.0), ("TRUELAT1", 38.0), ("TRUELAT2", 38.0),
        ("STAND_LON", -95.0), ("CEN_LAT", 38.0), ("CEN_LON", -95.0)] {
        schema.put_global_attr(name, AttrValue::Floats(vec![value])).unwrap();
    }
    let times = schema.def_var("Times", NcType::Char, &[time, strlen]).unwrap();
    let var = |schema: &mut Schema, name: &str, dims: &[usize], units: &str| {
        let id = schema.def_var(name, NcType::Float, dims).unwrap();
        schema.put_var_attr(id, "units", AttrValue::Text(units.into())).unwrap();
        id
    };
    let xlat = var(&mut schema, "XLAT", &[time, sn, we], "degree_north");
    let xlong = var(&mut schema, "XLONG", &[time, sn, we], "degree_east");
    let t2 = var(&mut schema, "T2", &[time, sn, we], "K");
    let t = var(&mut schema, "T", &[time, bt, sn, we], "K");
    let qv = var(&mut schema, "QVAPOR", &[time, bt, sn, we], "kg kg-1");
    let qr = var(&mut schema, "QRAIN", &[time, bt, sn, we], "kg kg-1");
    let qs = var(&mut schema, "QSNOW", &[time, bt, sn, we], "kg kg-1");
    let qg = var(&mut schema, "QGRAUP", &[time, bt, sn, we], "kg kg-1");
    let refl = var(&mut schema, "REFL_10CM", &[time, bt, sn, we], "dBZ");
    let w = var(&mut schema, "W", &[time, bts, sn, we], "m s-1");
    let heights = member.heights.then(|| {
        (
            var(&mut schema, "HGT", &[time, sn, we], "m"),
            var(&mut schema, "PH", &[time, bts, sn, we], "m2 s-2"),
            var(&mut schema, "PHB", &[time, bts, sn, we], "m2 s-2"),
        )
    });
    let mut writer = NcWriter::create(path, schema).unwrap();
    writer.write_record(0, times, VarData::Char(b"2026-08-19_01:00:00")).unwrap();
    let plane = |f: &dyn Fn(usize, usize) -> f32| -> Vec<f32> {
        (0..NY).flat_map(|y| (0..NX).map(move |x| (y, x))).map(|(y, x)| f(y, x)).collect()
    };
    let volume = |levels: usize, f: &dyn Fn(usize, usize, usize) -> f32| -> Vec<f32> {
        (0..levels)
            .flat_map(|k| (0..NY).flat_map(move |y| (0..NX).map(move |x| (k, y, x))))
            .map(|(k, y, x)| f(k, y, x))
            .collect()
    };
    writer.write_record(0, xlat, VarData::F32(&plane(&|y, _| 36.0 + 0.05 * y as f32))).unwrap();
    writer.write_record(0, xlong, VarData::F32(&plane(&|_, x| -98.0 + 0.05 * x as f32))).unwrap();
    writer.write_record(0, t2, VarData::F32(&vec![295.0; cells])).unwrap();
    writer.write_record(0, t, VarData::F32(&volume(NZ, &*member.theta))).unwrap();
    writer.write_record(0, qv, VarData::F32(&volume(NZ, &*member.qv))).unwrap();
    for id in [qr, qs, qg] {
        writer.write_record(0, id, VarData::F32(&volume(NZ, &*member.hydrometeor))).unwrap();
    }
    writer.write_record(0, refl, VarData::F32(&volume(NZ, &*member.refl))).unwrap();
    writer.write_record(0, w, VarData::F32(&volume(NZ + 1, &|k, _, _| k as f32))).unwrap();
    if let Some((hgt, ph, phb)) = heights {
        writer.write_record(0, hgt, VarData::F32(&vec![TERRAIN_M; cells])).unwrap();
        writer.write_record(0, ph, VarData::F32(&vec![0.0; cells * (NZ + 1)])).unwrap();
        // Full levels at 0, 1000, 2000, 3000, 4000 m above ground.
        let phb_values = volume(NZ + 1, &|k, _, _| 9.81 * (TERRAIN_M + 1_000.0 * k as f32));
        writer.write_record(0, phb, VarData::F32(&phb_values)).unwrap();
    }
    writer.finish().unwrap();
}

fn rw_compare(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_rw_compare"))
        .args(args)
        .env("RAYON_NUM_THREADS", "2")
        .env("CUDA_VISIBLE_DEVICES", "")
        .output()
        .unwrap()
}

fn ok(output: &Output) -> (String, PathBuf, Value) {
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    assert!(output.status.success(), "{stdout}\n{}", String::from_utf8_lossy(&output.stderr));
    let line = stdout.lines().find(|line| line.starts_with("RENDERED\t")).unwrap_or_else(|| panic!("{stdout}"));
    let png = PathBuf::from(line.split('\t').last().unwrap());
    let receipt: Value = serde_json::from_slice(&std::fs::read(png.with_extension("json")).unwrap()).unwrap();
    (stdout, png, receipt)
}

fn failure(output: &Output) -> String {
    assert!(!output.status.success(), "{}", String::from_utf8_lossy(&output.stdout));
    String::from_utf8_lossy(&output.stderr).into_owned()
}

fn sheet_size(columns: u32, rows: u32) -> (u32, u32) {
    let metrics = rw_wrfbatch::compare::sheet_metrics(WIDTH);
    (
        columns * WIDTH + (columns - 1) * metrics.gap,
        metrics.header_height + rows * HEIGHT + (rows - 1) * metrics.gap,
    )
}

fn s(path: &Path) -> &str {
    path.to_str().unwrap()
}

#[test]
fn increment_sheet_draws_prior_posterior_and_their_difference() {
    let scratch = Scratch::new("increment");
    let dir = &scratch.0;
    for member in 0..2 {
        let prior = 0.008 + 0.001 * member as f32;
        write_member(&dir.join(format!("prior_{member}.nc")), &Member::constant(prior, 30.0));
        let mut posterior = Member::constant(prior, 30.0);
        // Two grams per kilogram added over the western half, nothing east.
        posterior.qv = Box::new(move |_, _, x| if x < NX / 2 { prior + 0.002 } else { prior });
        write_member(&dir.join(format!("posterior_{member}.nc")), &posterior);
    }
    let width = WIDTH.to_string();
    let height = HEIGHT.to_string();
    let output = rw_compare(&[
        "--mode", "increment", "--out-dir", s(&dir.join("out")), "--name", "inc",
        "--field", "qv,qr+qs+qg", "--height-agl", "1000,2000",
        "--prior", s(&dir.join("prior_*.nc")), "--posterior", s(&dir.join("posterior_*.nc")),
        "--width", &width, "--height", &height,
    ]);
    let (_, png, receipt) = ok(&output);
    assert_eq!(png.file_name().unwrap(), "inc.png");
    assert_eq!(image::image_dimensions(&png).unwrap(), sheet_size(3, 4));
    let rows = receipt["rows"].as_array().unwrap();
    assert_eq!(rows.len(), 4, "two fields at two heights");
    assert_eq!(rows[0]["level"], "1000 m AGL");
    assert_eq!(rows[2]["title"], "Rain + snow + graupel mixing ratio");
    let increment = &rows[0]["increment"];
    assert!((increment["max"].as_f64().unwrap() - 2.0).abs() < 1e-4, "{increment}");
    assert!(increment["min"].as_f64().unwrap().abs() < 1e-4, "{increment}");
    assert!((rows[0]["prior"]["mean"].as_f64().unwrap() - 8.5).abs() < 1e-4, "the prior mean of 8 and 9 g/kg");
    assert!(rows[0]["half_range"].as_f64().unwrap() >= 2.0);
}

#[test]
fn increment_input_is_drawn_without_state_offsets_and_refusals_are_named() {
    let scratch = Scratch::new("increment-only");
    let dir = &scratch.0;
    let mut member = Member::constant(0.0005, 0.0);
    member.theta = Box::new(|_, y, _| if y < NY / 2 { 1.5 } else { -0.5 });
    let increment = dir.join("increment.nc");
    write_member(&increment, &member);
    let width = WIDTH.to_string();
    let height = HEIGHT.to_string();
    let out = dir.join("out");
    let output = rw_compare(&[
        "--mode", "increment", "--out-dir", s(&out), "--name", "only",
        "--field", "theta", "--height-agl", "1000", "--increment", s(&increment),
        "--width", &width, "--height", &height,
    ]);
    let (_, png, receipt) = ok(&output);
    assert_eq!(image::image_dimensions(&png).unwrap(), sheet_size(1, 1));
    let stats = &receipt["rows"][0]["increment"];
    assert_eq!(stats["max"], 1.5, "theta's 300 K is a state offset, not part of an increment");
    assert_eq!(stats["min"], -0.5);

    let refused = failure(&rw_compare(&["--mode", "increment", "--out-dir", s(&out), "--field", "qv", "--increment", s(&increment)]));
    assert!(refused.contains("--height-agl"), "{refused}");
    // Several heights of an increment input: fields down, heights across,
    // one ladder per field row.
    let (_, png, receipt) = ok(&rw_compare(&["--mode", "increment", "--out-dir", s(&out), "--name", "grid",
        "--field", "qv,theta", "--height-agl", "1000,2000,3000", "--increment", s(&increment),
        "--width", &width, "--height", &height]));
    assert_eq!(image::image_dimensions(&png).unwrap(), sheet_size(3, 2));
    let cells = receipt["rows"].as_array().unwrap();
    assert_eq!(cells.len(), 6);
    assert_eq!(cells[3]["level"], "1000 m AGL");
    assert_eq!(cells[3]["half_range"], cells[5]["half_range"], "one ladder per field row");
    // A plane listed beside 3-D fields is drawn once, as stored.
    let (_, _, receipt) = ok(&rw_compare(&["--mode", "increment", "--out-dir", s(&out), "--name", "mixed",
        "--field", "theta,t2", "--height-agl", "1000,2000", "--increment", s(&increment),
        "--width", &width, "--height", &height]));
    let levels: Vec<&str> = receipt["rows"].as_array().unwrap().iter().map(|row| row["level"].as_str().unwrap()).collect();
    assert_eq!(levels, vec!["1000 m AGL", "2000 m AGL", "as stored"]);
    // A single-level mode asked for a height of a plane refuses by name.
    let refused = failure(&rw_compare(&["--mode", "stamp", "--out-dir", s(&out), "--field", "t2", "--height-agl", "1000",
        "--members", &format!("{},{}", s(&increment), s(&increment))]));
    assert!(refused.contains("is a plane"), "{refused}");
    let refused = failure(&rw_compare(&["--mode", "increment", "--out-dir", s(&out), "--field", "nonesuch", "--increment", s(&increment)]));
    assert!(refused.contains("neither a table field"), "{refused}");

    let mut flat = Member::constant(0.0005, 0.0);
    flat.heights = false;
    let no_heights = dir.join("no-heights.nc");
    write_member(&no_heights, &flat);
    let args = ["--mode", "increment", "--out-dir", s(&out), "--field", "qv", "--height-agl", "1000", "--increment", s(&no_heights)];
    let refused = failure(&rw_compare(&args));
    assert!(refused.contains("--heights-from"), "{refused}");
    let mut with = args.to_vec();
    with.extend(["--heights-from", s(&increment), "--width", &width, "--height", &height]);
    let (_, _, receipt) = ok(&rw_compare(&with));
    assert!((receipt["rows"][0]["increment"]["mean"].as_f64().unwrap() - 0.5).abs() < 1e-4);
}

#[test]
fn spread_sheet_reports_the_member_standard_deviation() {
    let scratch = Scratch::new("spread");
    let dir = &scratch.0;
    for (member, refl) in [10.0f32, 20.0, 30.0].iter().enumerate() {
        write_member(&dir.join(format!("m_{member}.nc")), &Member::constant(0.008, *refl));
    }
    let width = WIDTH.to_string();
    let height = HEIGHT.to_string();
    let (_, png, receipt) = ok(&rw_compare(&[
        "--mode", "spread", "--out-dir", s(&dir.join("out")), "--members", s(&dir.join("m_*.nc")),
        "--field", "refc", "--columns", "pmm,mean,spread", "--width", &width, "--height", &height,
    ]));
    assert_eq!(image::image_dimensions(&png).unwrap(), sheet_size(3, 1));
    let row = &receipt["rows"][0];
    assert_eq!(row["level"], "column maximum");
    assert!((row["spread"]["mean"].as_f64().unwrap() - 10.0).abs() < 1e-6, "{row}");
    assert!((row["mean"]["mean"].as_f64().unwrap() - 20.0).abs() < 1e-6, "{row}");
    assert_eq!(row["field_scale"], "production colour table");
    let refused = failure(&rw_compare(&["--mode", "spread", "--out-dir", s(&dir.join("out")),
        "--members", s(&dir.join("m_0.nc"))]));
    assert!(refused.contains("at least two members"), "{refused}");
}

#[test]
fn stamp_sheet_lays_out_stamps_paintball_maximum_and_pmm() {
    let scratch = Scratch::new("stamp");
    let dir = &scratch.0;
    for member in 0..4 {
        let mut stamp = Member::constant(0.008, 0.0);
        stamp.refl = Box::new(move |_, y, x| if x > 4 * member && y > 4 { 50.0 } else { 0.0 });
        write_member(&dir.join(format!("prior_{member}.nc")), &stamp);
        write_member(&dir.join(format!("post_{member}.nc")), &stamp);
    }
    let width = WIDTH.to_string();
    let height = HEIGHT.to_string();
    let (stdout, png, receipt) = ok(&rw_compare(&[
        "--mode", "stamp", "--out-dir", s(&dir.join("out")), "--stamps", "2",
        "--members", s(&dir.join("prior_*.nc")), "--row-label", "Prior",
        "--members", s(&dir.join("post_*.nc")), "--row-label", "Analysis",
        "--width", &width, "--height", &height,
    ]));
    // Two stamps, paintball at 35 and 45 dBZ, the maximum and the PMM.
    assert_eq!(image::image_dimensions(&png).unwrap(), sheet_size(6, 2));
    assert_eq!(receipt["thresholds"], serde_json::json!([35.0, 45.0]));
    assert_eq!(receipt["rows"][0]["member_numbers"], serde_json::json!([0, 1, 2, 3]));
    assert!(stdout.contains("PAINTBALL_LEGEND\t0=#1f77b4 1=#aec7e8"), "{stdout}");
}

#[test]
fn omb_sheet_marks_rejected_observations_and_splits_types() {
    let scratch = Scratch::new("omb");
    let dir = &scratch.0;
    let grid = dir.join("grid.nc");
    write_member(&grid, &Member::constant(0.008, 40.0));
    let obs = dir.join("omb.json");
    std::fs::write(&obs, serde_json::to_vec(&serde_json::json!({"valid": "2026-08-19T01:00:00", "observations": [
        {"lat": 36.2, "lon": -97.8, "value": 1.5, "type": "temperature_2m", "units": "K"},
        {"lat": 36.4, "lon": -97.4, "value": -2.0, "type": "temperature_2m", "units": "K"},
        {"lat": 36.5, "lon": -97.5, "value": 9.0, "type": "temperature_2m", "units": "K", "rejected": true},
        {"lat": 36.3, "lon": -97.6, "value": 0.5, "type": "wind_speed_10m", "units": "m/s"},
    ]})).unwrap()).unwrap();
    let width = WIDTH.to_string();
    let height = HEIGHT.to_string();
    let (_, png, receipt) = ok(&rw_compare(&[
        "--mode", "omb", "--out-dir", s(&dir.join("out")), "--obs", s(&obs), "--grid", s(&grid),
        "--width", &width, "--height", &height,
    ]));
    assert_eq!(image::image_dimensions(&png).unwrap(), sheet_size(2, 1));
    let panels = receipt["panels"].as_array().unwrap();
    assert_eq!(panels[0]["type"], "temperature_2m");
    assert_eq!((panels[0]["used"].as_u64(), panels[0]["rejected"].as_u64()), (Some(2), Some(1)));
    assert_eq!(panels[0]["departure"]["mean"], -0.25);
    assert_eq!(panels[1]["units"], "m/s");

    // Over a shaded background each dot ladder gets its own key under the sheet.
    let (_, png, _) = ok(&rw_compare(&[
        "--mode", "omb", "--out-dir", s(&dir.join("out")), "--obs", s(&obs), "--background", s(&grid),
        "--field", "refc", "--types", "temperature_2m", "--name", "over", "--width", &width, "--height", &height,
    ]));
    let (w, h) = sheet_size(1, 1);
    assert_eq!(image::image_dimensions(&png).unwrap(), (w, h + 78));
    let refused = failure(&rw_compare(&["--mode", "omb", "--out-dir", s(&dir.join("out")), "--obs", s(&obs),
        "--grid", s(&grid), "--all"]));
    assert!(refused.contains("one unit"), "{refused}");
    let refused = failure(&rw_compare(&["--mode", "omb", "--out-dir", s(&dir.join("out")), "--obs", s(&obs),
        "--grid", s(&grid), "--types", "radar_z"]));
    assert!(refused.contains("holds no such observation"), "{refused}");
}
