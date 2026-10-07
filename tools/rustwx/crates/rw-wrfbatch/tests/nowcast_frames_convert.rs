//! `rw_nowcast_frames from-netcdf` through the real binary, and the frames
//! it writes through the real `rw_compare`: a NetCDF nowcast whose frame
//! holds exactly a history file's composite reflectivity must draw exactly
//! the history panel, and a frame set that cannot say its issue time or
//! whether it saw the future must be refused.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use netcdf_writer::{AttrValue, NcFormat, NcType, NcWriter, Schema, VarData};
use rw_wrfbatch::nowcast_frames::{LambertLattice, Lattice, read_frames_receipt};
use serde_json::Value;

const NX: usize = 24;
const NY: usize = 18;
const NZ: usize = 4;
const WIDTH: u32 = 400;
const HEIGHT: u32 = 300;

struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        let root = std::env::temp_dir().join(format!("rw-nowcast-convert-{tag}-{}", std::process::id()));
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

/// The composite reflectivity every fixture holds at cell (y, x).
fn composite(_y: usize, x: usize) -> f32 {
    5.0 + 2.0 * x as f32
}

/// A wrfout-shaped file valid 2026-08-19 01Z from a 00Z start, on the
/// regular 0.05 degree lattice from 36 N, 98 W; its REFL_10CM column
/// maximum is `composite`.  (The fixture `heat_eval_panels.rs` draws.)
fn write_history(path: &Path) {
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
    let hgt = var(&mut schema, "HGT", &[time, sn, we], "m");
    let ph = var(&mut schema, "PH", &[time, bts, sn, we], "m2 s-2");
    let phb = var(&mut schema, "PHB", &[time, bts, sn, we], "m2 s-2");
    let mut writer = NcWriter::create(path, schema).unwrap();
    writer.write_record(0, times, VarData::Char(b"2026-08-19_01:00:00")).unwrap();
    let plane = |f: &dyn Fn(usize, usize) -> f32| -> Vec<f32> {
        (0..NY).flat_map(|y| (0..NX).map(move |x| (y, x))).map(|(y, x)| f(y, x)).collect()
    };
    writer.write_record(0, xlat, VarData::F32(&plane(&|y, _| 36.0 + 0.05 * y as f32))).unwrap();
    writer.write_record(0, xlong, VarData::F32(&plane(&|_, x| -98.0 + 0.05 * x as f32))).unwrap();
    writer.write_record(0, t2, VarData::F32(&vec![295.0; cells])).unwrap();
    let volume: Vec<f32> = (0..NZ)
        .flat_map(|k| (0..NY).flat_map(move |y| (0..NX).map(move |x| (k, y, x))))
        .map(|(k, y, x)| if k == 1 { composite(y, x) } else { composite(y, x) - 20.0 })
        .collect();
    writer.write_record(0, refl, VarData::F32(&volume)).unwrap();
    writer.write_record(0, t, VarData::F32(&vec![10.0; cells * NZ])).unwrap();
    writer.write_record(0, qv, VarData::F32(&vec![0.01; cells * NZ])).unwrap();
    for id in [qr, qs, qg] {
        writer.write_record(0, id, VarData::F32(&vec![0.0; cells * NZ])).unwrap();
    }
    writer.write_record(0, w, VarData::F32(&vec![0.0; cells * (NZ + 1)])).unwrap();
    writer.write_record(0, hgt, VarData::F32(&vec![320.0; cells])).unwrap();
    writer.write_record(0, ph, VarData::F32(&vec![0.0; cells * (NZ + 1)])).unwrap();
    let phb_values: Vec<f32> = (0..NZ + 1)
        .flat_map(|k| std::iter::repeat_n(9.81 * (320.0 + 1_000.0 * k as f32), cells))
        .collect();
    writer.write_record(0, phb, VarData::F32(&phb_values)).unwrap();
    writer.finish().unwrap();
}

/// What a nowcast runner writes: `refc[valid_time, member, y, x]` in dBZ
/// with CF hours since an epoch, cell-centre `lat`/`lon`, and the run's
/// facts as global attributes.  `value(t, member, y, x)`.
#[allow(clippy::too_many_arguments)]
fn write_nowcast_netcdf(
    path: &Path,
    lat: &[f32],
    lon: &[f32],
    ny: usize,
    nx: usize,
    hours: &[i32],
    members: usize,
    attrs: &[(&str, &str)],
    value: &dyn Fn(usize, usize, usize, usize) -> f32,
) {
    let mut schema = Schema::new(NcFormat::Offset64);
    let time = schema.def_dim("valid_time", 0, true).unwrap();
    let member = schema.def_dim("member", members, false).unwrap();
    let y = schema.def_dim("y", ny, false).unwrap();
    let x = schema.def_dim("x", nx, false).unwrap();
    for (name, text) in attrs {
        schema.put_global_attr(name, AttrValue::Text((*text).into())).unwrap();
    }
    let refc = schema.def_var("refc", NcType::Float, &[time, member, y, x]).unwrap();
    schema.put_var_attr(refc, "units", AttrValue::Text("dBZ".into())).unwrap();
    schema.put_var_attr(refc, "_FillValue", AttrValue::Floats(vec![-9999.0])).unwrap();
    let valid = schema.def_var("valid_time", NcType::Int, &[time]).unwrap();
    schema.put_var_attr(valid, "units", AttrValue::Text("hours since 2026-08-19 00:00:00".into())).unwrap();
    let member_var = schema.def_var("member", NcType::Int, &[member]).unwrap();
    let lat_var = schema.def_var("lat", NcType::Float, &[y, x]).unwrap();
    let lon_var = schema.def_var("lon", NcType::Float, &[y, x]).unwrap();
    let mut writer = NcWriter::create(path, schema).unwrap();
    writer.write_var(member_var, VarData::I32(&(0..members as i32).collect::<Vec<_>>())).unwrap();
    writer.write_var(lat_var, VarData::F32(lat)).unwrap();
    writer.write_var(lon_var, VarData::F32(lon)).unwrap();
    for (t, hour) in hours.iter().enumerate() {
        writer.write_record(t as u64, valid, VarData::I32(&[*hour])).unwrap();
        let block: Vec<f32> = (0..members)
            .flat_map(|m| (0..ny).flat_map(move |j| (0..nx).map(move |i| (m, j, i))))
            .map(|(m, j, i)| value(t, m, j, i))
            .collect();
        writer.write_record(t as u64, refc, VarData::F32(&block)).unwrap();
    }
    writer.finish().unwrap();
}

fn regular_grid() -> (Vec<f32>, Vec<f32>) {
    let lat = (0..NY).flat_map(|y| (0..NX).map(move |_| 36.0 + 0.05 * y as f32)).collect();
    let lon = (0..NY).flat_map(|_| (0..NX).map(|x| -98.0 + 0.05 * x as f32)).collect();
    (lat, lon)
}

fn convert(args: &[String]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_rw_nowcast_frames"))
        .arg("from-netcdf")
        .args(args)
        .env("CUDA_VISIBLE_DEVICES", "")
        .output()
        .unwrap()
}

fn rw_compare(args: &[String]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_rw_compare"))
        .args(args)
        .env("RAYON_NUM_THREADS", "2")
        .env("CUDA_VISIBLE_DEVICES", "")
        .output()
        .unwrap()
}

fn strings(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| s.to_string()).collect()
}

fn converted(output: &Output) -> Value {
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_str(String::from_utf8_lossy(&output.stdout).trim()).unwrap()
}

fn refused(output: &Output) -> String {
    assert!(!output.status.success(), "{}", String::from_utf8_lossy(&output.stdout));
    assert_eq!(output.status.code(), Some(3), "a refusal exits 3");
    String::from_utf8_lossy(&output.stderr).into_owned()
}

fn rendered(output: &Output) -> PathBuf {
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "{stdout}\n{}", String::from_utf8_lossy(&output.stderr));
    let line = stdout
        .lines()
        .find(|line| line.starts_with("RENDERED\trefc\t"))
        .unwrap_or_else(|| panic!("no RENDERED line: {stdout}"));
    PathBuf::from(line.split('\t').last().unwrap())
}

fn panel(sheet: &image::RgbaImage, row: u32, column: u32) -> image::RgbaImage {
    let (x, y) = rw_wrfbatch::compare::sheet_panel_origin(row, column, WIDTH, HEIGHT);
    image::imageops::crop_imm(sheet, x, y, WIDTH, HEIGHT).to_image()
}

/// The map below the title and subtitle strip.
fn body(image: &image::RgbaImage) -> image::RgbaImage {
    image::imageops::crop_imm(image, 0, HEIGHT / 4, WIDTH, HEIGHT - HEIGHT / 4).to_image()
}

#[test]
fn a_converted_frame_holding_the_history_composite_draws_the_history_panel() {
    let scratch = Scratch::new("identity");
    let history = scratch.0.join("wrfout_d01_2026-08-19_01_00_00");
    write_history(&history);
    let (lat, lon) = regular_grid();
    let netcdf = scratch.0.join("refc_run.nc");
    // Hour 0 is 20 dBZ stronger and cell (0, 0) of its member 1 carries the
    // fill value; hour 1 is the history composite in both members, so its
    // probability-matched mean is that composite exactly.
    write_nowcast_netcdf(
        &netcdf, &lat, &lon, NY, NX, &[0, 1], 2,
        &[("init", "2026-08-19 00:00"), ("causal", "True"), ("seed", "7"), ("variant", "synthetic hourly")],
        &|t, m, y, x| if t == 0 && m == 1 && y == 0 && x == 0 { -9999.0 } else { composite(y, x) + if t == 0 { 20.0 } else { 0.0 } },
    );
    let root = scratch.0.join("frames");
    let record = converted(&convert(&strings(&[
        "--netcdf", netcdf.to_str().unwrap(), "--out", root.to_str().unwrap(),
        "--source-id", "synthetic-nowcast", "--lattice", "latlon",
    ])));
    assert_eq!(record["frames"], 2);
    assert_eq!(record["members"], 2);
    assert_eq!(record["causal"], true);
    assert_eq!(record["issue_time"], "2026-08-19T00:00:00+00:00");
    assert_eq!(record["lattice"]["kind"], "latlon");

    let receipt = read_frames_receipt(&root).unwrap();
    assert_eq!(receipt.members, 2);
    assert_eq!(receipt.source_id, "synthetic-nowcast");
    assert!(receipt.causal);
    assert_eq!(receipt.frames.len(), 2);
    assert_eq!(receipt.frames[0].lead_minutes, 0);
    assert_eq!(receipt.frames[1].lead_minutes, 60);
    match &receipt.lattice {
        Lattice::LatLon(l) => {
            assert!((l.lat0 - 36.0).abs() < 1e-5 && (l.lon0 + 98.0).abs() < 1e-5);
            assert!((l.dlat - 0.05).abs() < 1e-6 && (l.dlon - 0.05).abs() < 1e-6);
            assert_eq!((l.ny, l.nx), (NY, NX));
        }
        other => panic!("{other:?}"),
    }
    let doc: Value = serde_json::from_slice(&std::fs::read(root.join("nowcast.json")).unwrap()).unwrap();
    assert_eq!(doc["seed"], 7);
    assert_eq!(doc["source"]["package"], "synthetic hourly");
    assert_eq!(doc["converter"]["tool"], "rw_nowcast_frames from-netcdf");
    assert_eq!(doc["history"]["no_coverage"]["applied"], false);
    assert!(root.join("inputs.json").is_file());
    assert!(!root.join("20260819T0100Z").join("refc.f32.partial").exists());

    // The bytes are the file's values, member-major, with the fill value as NaN.
    let bytes = std::fs::read(root.join("20260819T0100Z").join("refc.f32")).unwrap();
    assert_eq!(bytes.len(), 2 * NY * NX * 4);
    let at = |index: usize| f32::from_le_bytes(bytes[index * 4..index * 4 + 4].try_into().unwrap());
    assert_eq!(at(1), composite(0, 1));
    assert_eq!(at(NY * NX + 5), composite(0, 5));
    let hour0 = std::fs::read(root.join("20260819T0000Z").join("refc.f32")).unwrap();
    let at0 = |index: usize| f32::from_le_bytes(hour0[index * 4..index * 4 + 4].try_into().unwrap());
    assert_eq!(at0(1), composite(0, 1) + 20.0);
    assert!(at0(NY * NX).is_nan(), "the fill value is no coverage");

    // Through the renderer: the 01Z frame draws the history panel's map.
    let mut args = strings(&[
        "--store-root", scratch.0.join("store").to_str().unwrap(),
        "--out-dir", scratch.0.join("png").to_str().unwrap(),
        "--products", "refc", "--offline", "--layout", "flat",
        "--width", &WIDTH.to_string(), "--height", &HEIGHT.to_string(), "--sheet-name", "identity",
    ]);
    args.extend(strings(&["--panel", &format!("history={}", history.display())]));
    args.extend(strings(&["--panel", &format!("nowcast=frames:{}@2026-08-19T01:00Z", root.display())]));
    let path = rendered(&rw_compare(&args));
    let sheet = image::open(&path).unwrap().to_rgba8();
    let history_body = body(&panel(&sheet, 0, 0));
    assert!(history_body.pixels().any(|p| p.0 != history_body.get_pixel(0, 0).0), "the history panel drew nothing");
    assert!(body(&panel(&sheet, 0, 1)) == history_body, "the converted frame is not the history panel's map");
    let receipt: Value = serde_json::from_slice(&std::fs::read(path.with_extension("json")).unwrap()).unwrap();
    let column = &receipt["rows"][0]["columns"][1];
    assert_eq!(column["kind"], "external");
    assert_eq!(column["source"]["record"]["source_id"], "synthetic-nowcast");
    assert_eq!(column["source"]["record"]["members"], 2);
    assert_eq!(column["source"]["record"]["causal"], true);

    // The same root again: refused, the READY receipt stays.
    assert!(refused(&convert(&strings(&[
        "--netcdf", netcdf.to_str().unwrap(), "--out", root.to_str().unwrap(),
        "--source-id", "synthetic-nowcast", "--lattice", "latlon",
    ])))
    .contains("already READY"));
}

#[test]
fn a_lambert_grid_is_fitted_and_a_receipt_supplies_what_the_file_lacks() {
    let scratch = Scratch::new("lambert");
    // A 3 km HRRR-like lattice, cell centres stored as float32 as a runner writes them.
    let mut truth = LambertLattice {
        truelat1: 38.5, truelat2: 38.5, stand_lon: -97.5, ref_lat: 38.5, ref_lon: -97.5,
        earth_radius_m: 6_371_229.0, nx: 40, ny: 30, dx_m: 3000.0, dy_m: 3000.0, x0_m: 0.0, y0_m: 0.0,
    };
    let (x0, y0) = truth.forward(21.138123, -122.719528);
    truth.x0_m = x0 + 600.0 * 3000.0;
    truth.y0_m = y0 + 400.0 * 3000.0;
    let lattice = Lattice::Lambert(truth.clone());
    let (ny, nx) = lattice.shape();
    let mut lat = Vec::new();
    let mut lon = Vec::new();
    for j in 0..ny {
        for i in 0..nx {
            let (la, lo) = lattice.cell_latlon(j, i);
            lat.push(la as f32);
            // Longitudes in [0, 360), as some runners write them.
            lon.push((lo + 360.0) as f32);
        }
    }
    let netcdf = scratch.0.join("refc_lambert.nc");
    write_nowcast_netcdf(&netcdf, &lat, &lon, ny, nx, &[1, 2, 3], 1, &[], &|t, _, y, x| composite(y, x) + t as f32);
    let receipt_path = scratch.0.join("receipt.json");
    std::fs::write(
        &receipt_path,
        serde_json::to_vec_pretty(&serde_json::json!({
            "issue_time": "2026-08-19T00:00:00Z", "causal": false, "latest_input_time": "2026-08-19T00:30:00Z",
            "members": 1, "seed": 3, "variant": "6km hourly",
            "objects": [{"stream": "mrms", "key": "noaa-mrms-pds/CONUS/x/y.grib2.gz", "stamp": "2026-08-19T00:30:00Z"}],
            "step_s": [1.5, 1.4], "load_s": 2.0, "torch_peak_reserved_gib": 1.0,
        }))
        .unwrap(),
    )
    .unwrap();
    let root = scratch.0.join("frames");
    let projection = "lambert:truelat1=38.5,truelat2=38.5,stand_lon=-97.5,ref_lat=38.5,earth_radius_m=6371229";
    let record = converted(&convert(&strings(&[
        "--netcdf", netcdf.to_str().unwrap(), "--out", root.to_str().unwrap(), "--source-id", "lambert-run",
        "--lattice", projection, "--receipt", receipt_path.to_str().unwrap(),
    ])));
    assert_eq!(record["lattice"]["kind"], "lambert");
    assert_eq!(record["lattice"]["row_order"], "south_to_north");
    assert_eq!(record["causal"], false);
    let receipt = read_frames_receipt(&root).unwrap();
    match &receipt.lattice {
        Lattice::Lambert(fitted) => {
            assert!((fitted.dx_m - 3000.0).abs() < 0.05 && (fitted.dy_m - 3000.0).abs() < 0.05, "{fitted:?}");
            assert!((fitted.x0_m - truth.x0_m).abs() < 1.0 && (fitted.y0_m - truth.y0_m).abs() < 1.0, "{fitted:?}");
            assert_eq!(fitted.earth_radius_m, 6_371_229.0);
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(receipt.frames.len(), 3);
    assert_eq!(receipt.frames[0].lead_minutes, 60);
    assert!(!receipt.causal);
    let doc: Value = serde_json::from_slice(&std::fs::read(root.join("nowcast.json")).unwrap()).unwrap();
    assert_eq!(doc["latest_input_time"], "2026-08-19T00:30:00Z");
    assert_eq!(doc["peak_device_bytes"], 1_073_741_824u64);
    assert_eq!(doc["timing"]["load_s"], 2.0);
    let inputs: Value = serde_json::from_slice(&std::fs::read(root.join("inputs.json")).unwrap()).unwrap();
    assert_eq!(inputs["objects"][0]["bucket"], "noaa-mrms-pds");
    assert_eq!(inputs["objects"][0]["key"], "CONUS/x/y.grib2.gz");

    // The receipt's member count must match the file.
    let wrong = scratch.0.join("wrong-members.json");
    std::fs::write(&wrong, br#"{"issue_time": "2026-08-19T00:00:00Z", "causal": true, "members": 8}"#).unwrap();
    assert!(refused(&convert(&strings(&[
        "--netcdf", netcdf.to_str().unwrap(), "--out", scratch.0.join("w").to_str().unwrap(),
        "--source-id", "x", "--lattice", projection, "--receipt", wrong.to_str().unwrap(),
    ])))
    .contains("8 members"));

    // A Lambert grid asked for as a latitude-longitude lattice is refused.
    assert!(refused(&convert(&strings(&[
        "--netcdf", netcdf.to_str().unwrap(), "--out", scratch.0.join("ll").to_str().unwrap(),
        "--source-id", "x", "--lattice", "latlon", "--issue", "2026-08-19T00:00Z", "--causal", "true",
    ])))
    .contains("not a latitude-longitude lattice"));
}

#[test]
fn a_frame_set_that_cannot_say_its_issue_time_or_causality_is_refused() {
    let scratch = Scratch::new("refusals");
    let (lat, lon) = regular_grid();
    let netcdf = scratch.0.join("bare.nc");
    write_nowcast_netcdf(&netcdf, &lat, &lon, NY, NX, &[1], 1, &[], &|_, _, y, x| composite(y, x));
    let base = |tag: &str| {
        strings(&[
            "--netcdf", netcdf.to_str().unwrap(), "--out", scratch.0.join(tag).to_str().unwrap(),
            "--source-id", "bare", "--lattice", "latlon",
        ])
    };
    assert!(refused(&convert(&base("no-issue"))).contains("no issue time"));
    let mut no_causal = base("no-causal");
    no_causal.extend(strings(&["--issue", "2026-08-19T00:00Z"]));
    assert!(refused(&convert(&no_causal)).contains("whether it saw the future"));
    let mut both = base("both");
    both.extend(strings(&["--issue", "2026-08-19T00:00Z", "--causal", "false"]));
    let record = converted(&convert(&both));
    assert_eq!(record["causal"], false);
    assert_eq!(record["frames"], 1);
    // A regular lat/lon grid asked for as a Lambert lattice is refused.
    let mut lambert = base("lambert");
    lambert.extend(strings(&[
        "--issue", "2026-08-19T00:00Z", "--causal", "true", "--lattice",
        "lambert:truelat1=38.5,truelat2=38.5,stand_lon=-97.5,ref_lat=38.5,earth_radius_m=6371229",
    ]));
    let error = refused(&convert(&lambert));
    assert!(error.contains("not rectilinear") || error.contains("not uniformly spaced"), "{error}");
    // A receipt that disagrees with the file's own init or causal attribute describes another run.
    let stamped = scratch.0.join("stamped.nc");
    write_nowcast_netcdf(
        &stamped, &lat, &lon, NY, NX, &[1], 1, &[("init", "2026-08-19 00:00"), ("causal", "True")],
        &|_, _, y, x| composite(y, x),
    );
    for (tag, receipt, needle) in [
        ("other-issue", r#"{"issue_time": "2026-08-18T23:00:00Z", "causal": true}"#, "wrong issue"),
        ("other-causal", r#"{"issue_time": "2026-08-19T00:00:00Z", "causal": false}"#, "scored as a forecast"),
    ] {
        let path = scratch.0.join(format!("{tag}.json"));
        std::fs::write(&path, receipt).unwrap();
        let error = refused(&convert(&strings(&[
            "--netcdf", stamped.to_str().unwrap(), "--out", scratch.0.join(tag).to_str().unwrap(),
            "--source-id", "stamped", "--lattice", "latlon", "--receipt", path.to_str().unwrap(),
        ])));
        assert!(error.contains(needle), "{tag}: {error}");
        assert!(!scratch.0.join(tag).join("nowcast.json").exists());
    }
    let agreeing = scratch.0.join("agreeing.json");
    std::fs::write(&agreeing, r#"{"issue_time": "2026-08-19T00:00:00Z", "causal": true}"#).unwrap();
    let record = converted(&convert(&strings(&[
        "--netcdf", stamped.to_str().unwrap(), "--out", scratch.0.join("agreeing").to_str().unwrap(),
        "--source-id", "stamped", "--lattice", "latlon", "--receipt", agreeing.to_str().unwrap(),
    ])));
    assert_eq!(record["causal"], true);
    // Arguments short of a conversion exit 2, not 3.
    let output = Command::new(env!("CARGO_BIN_EXE_rw_nowcast_frames")).arg("from-netcdf").output().unwrap();
    assert_eq!(output.status.code(), Some(2));
    let abi = Command::new(env!("CARGO_BIN_EXE_rw_nowcast_frames")).arg("--abi").output().unwrap();
    assert!(String::from_utf8_lossy(&abi.stdout).starts_with("gpuwm-rw-nowcast-frames-v1\t"));
}

/// A NetCDF-4 (HDF5) file shaped like a StormScope run, whose `valid_time`
/// coordinate is an HDF5 dimension scale.  The NetCDF index netcrust
/// answers `variable()` from omits those datasets, so a converter that reads
/// the time coordinate's units only through `variable()` refuses the file
/// as having no time dimension (every real StormScope run on 2026-10-06
/// did).  The classic-format fixtures above cannot reproduce that.
#[test]
fn a_netcdf4_file_whose_time_coordinate_is_a_dimension_scale_converts() {
    let scratch = Scratch::new("netcdf4");
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("nowcast_frames_fixtures")
        .join("netcdf4-dimension-scale-nowcast.nc");
    let root = scratch.0.join("frames");
    let record = converted(&convert(&strings(&[
        "--netcdf", fixture.to_str().unwrap(), "--out", root.to_str().unwrap(),
        "--source-id", "stormscope-shaped", "--lattice", "latlon",
    ])));
    assert_eq!(record["frames"], 2);
    assert_eq!(record["members"], 2);
    assert_eq!(record["causal"], true);
    assert_eq!(record["issue_time"], "2026-08-19T00:00:00+00:00");
    assert_eq!(record["first_valid"], "2026-08-19T00:00:00+00:00");
    assert_eq!(record["last_valid"], "2026-08-19T01:00:00+00:00");
    let receipt = read_frames_receipt(&root).unwrap();
    match &receipt.lattice {
        Lattice::LatLon(l) => {
            // Longitudes were written in [0, 360): 262 E is 98 W.
            assert!((l.lat0 - 36.0).abs() < 1e-5 && (l.lon0 + 98.0).abs() < 1e-4, "{l:?}");
            assert!((l.dlat - 0.05).abs() < 1e-5 && (l.dlon - 0.05).abs() < 1e-5, "{l:?}");
            assert_eq!((l.ny, l.nx), (3, 4));
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(receipt.frames[1].lead_minutes, 60);
    let doc: Value = serde_json::from_slice(&std::fs::read(root.join("nowcast.json")).unwrap()).unwrap();
    assert_eq!(doc["seed"], 7);
    assert_eq!(doc["source"]["code"]["precision"], "bf16");
    assert_eq!(doc["source"]["package"], "6km hourly");
    let hour1 = std::fs::read(root.join("20260819T0100Z").join("refc.f32")).unwrap();
    let at = |bytes: &[u8], index: usize| f32::from_le_bytes(bytes[index * 4..index * 4 + 4].try_into().unwrap());
    assert_eq!(hour1.len(), 2 * 3 * 4 * 4);
    assert_eq!(at(&hour1, 1), 7.0);
    assert_eq!(at(&hour1, 12 + 3), 11.0);
    let hour0 = std::fs::read(root.join("20260819T0000Z").join("refc.f32")).unwrap();
    assert_eq!(at(&hour0, 1), 27.0);
    assert!(at(&hour0, 12).is_nan(), "the NaN fill value is no coverage");
}
