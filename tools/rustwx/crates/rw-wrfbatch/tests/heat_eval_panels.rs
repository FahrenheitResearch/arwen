//! The two radar-heating evaluation panel sources through the real
//! `rw_compare` binary: `frames:ROOT@VALID` (a nowcast or MRMS frame set in
//! the `gpuwm-obs.nowcast-frames.v1` contract) and `ttenref:WINDOW` (a
//! heating window's column maximum), on small synthetic inputs whose
//! pictures are known: a frame or window that holds exactly the history
//! file's composite reflectivity must draw exactly the history panel.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use netcdf_writer::{AttrValue, NcFormat, NcType, NcWriter, Schema, VarData};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const NX: usize = 24;
const NY: usize = 18;
const NZ: usize = 4;
const WIDTH: u32 = 400;
const HEIGHT: u32 = 300;

struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        let root = std::env::temp_dir().join(format!("rw-heat-panels-{tag}-{}", std::process::id()));
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
/// maximum is `composite`.
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

fn hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

/// A latitude-longitude lattice block covering the history grid exactly.
fn latlon_lattice(corner_shift_deg: f64) -> Value {
    let corner = |row: usize, column: usize| {
        json!([36.0 + 0.05 * row as f64 + corner_shift_deg, -98.0 + 0.05 * column as f64])
    };
    json!({
        "kind": "latlon", "lat0": 36.0, "lon0": -98.0, "dlat": 0.05, "dlon": 0.05,
        "nx": NX, "ny": NY, "row_order": "south_to_north",
        "corners_latlon": {"j0_i0": corner(0, 0), "j0_in": corner(0, NX - 1),
                           "jn_i0": corner(NY - 1, 0), "jn_in": corner(NY - 1, NX - 1)},
    })
}

/// A frames root with one frame per `(valid, members)` entry; each member
/// plane is `value(member, y, x)`.
fn write_frames(
    root: &Path,
    causal: bool,
    lattice: Value,
    frames: &[(&str, usize)],
    value: &dyn Fn(usize, usize, usize) -> f32,
) {
    std::fs::create_dir_all(root).unwrap();
    let mut rows = Vec::new();
    let members = frames[0].1;
    for (valid, count) in frames {
        let stamp = valid.replace(['-', ':'], "");
        let dir = root.join(&stamp);
        std::fs::create_dir_all(&dir).unwrap();
        let mut bytes = Vec::new();
        for member in 0..*count {
            for y in 0..NY {
                for x in 0..NX {
                    bytes.extend_from_slice(&value(member, y, x).to_le_bytes());
                }
            }
        }
        std::fs::write(dir.join("refc.f32"), &bytes).unwrap();
        rows.push(json!({"valid": valid, "file": format!("{stamp}/refc.f32"),
            "shape": [count, NY, NX], "dtype": "float32-le", "sha256": hex(&bytes)}));
    }
    let receipt = json!({
        "schema": "gpuwm-obs.nowcast-frames.v1", "status": "READY",
        "source": {"kind": if causal { "nowcast" } else { "observed" }, "id": "synthetic-nowcast"},
        "issue_time": "2026-08-19T00:00:00Z", "latest_input_time": "2026-08-19T00:00:00Z",
        "causal": causal, "variable": "refc", "units": "dBZ", "members": members,
        "lattice": lattice, "frames": rows,
    });
    std::fs::write(root.join("nowcast.json"), serde_json::to_vec_pretty(&receipt).unwrap()).unwrap();
}

/// A heating window ending at `end` on an `ny x nx` mass grid; level 2
/// holds `column(y, x)` and every other level is no coverage.
fn write_window(dir: &Path, end: &str, ny: usize, nx: usize, lead_class: &str, column: &dyn Fn(usize, usize) -> f32) {
    std::fs::create_dir_all(dir).unwrap();
    let mut bytes = Vec::new();
    for k in 0..NZ {
        for y in 0..ny {
            for x in 0..nx {
                let value = if k == 2 { column(y, x) } else { -99999.0f32 };
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
    }
    std::fs::write(dir.join("ref.f32"), &bytes).unwrap();
    let receipt = json!({
        "schema": "gpuwm-obs.radar-tten-ref.v1", "status": "READY", "shape": [NZ, ny, nx],
        "data": {"file": "ref.f32", "bytes": bytes.len(), "sha256": hex(&bytes)},
        "grid": {"identity_sha256": "0".repeat(64), "nx": nx, "ny": ny, "nz": NZ},
        "window": {"end": end},
        "source": {"lead_class": lead_class, "id": "synthetic", "issue_time": "2026-08-19T00:00:00Z"},
    });
    std::fs::write(dir.join("ref.json"), serde_json::to_vec_pretty(&receipt).unwrap()).unwrap();
}

fn rw_compare(args: &[String]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_rw_compare"))
        .args(args)
        .env("RAYON_NUM_THREADS", "2")
        .env("CUDA_VISIBLE_DEVICES", "")
        .output()
        .unwrap()
}

fn base(root: &Path, tag: &str) -> Vec<String> {
    [
        "--store-root", root.join(format!("store-{tag}")).to_str().unwrap(),
        "--out-dir", root.join(format!("png-{tag}")).to_str().unwrap(),
        "--products", "refc", "--offline", "--layout", "flat",
        "--width", &WIDTH.to_string(), "--height", &HEIGHT.to_string(),
        "--sheet-name", tag,
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
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

fn refused(output: &Output) -> String {
    assert!(!output.status.success(), "{}", String::from_utf8_lossy(&output.stdout));
    format!("{}{}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr))
}

fn panel(sheet: &image::RgbaImage, row: u32, column: u32) -> image::RgbaImage {
    let (x, y) = rw_wrfbatch::compare::sheet_panel_origin(row, column, WIDTH, HEIGHT);
    image::imageops::crop_imm(sheet, x, y, WIDTH, HEIGHT).to_image()
}

/// The map below the title and subtitle strip.
fn body(image: &image::RgbaImage) -> image::RgbaImage {
    image::imageops::crop_imm(image, 0, HEIGHT / 4, WIDTH, HEIGHT - HEIGHT / 4).to_image()
}

fn panel_arg(text: String) -> [String; 2] {
    ["--panel".to_string(), text]
}

#[test]
fn a_frame_or_window_holding_the_history_composite_draws_the_history_panel() {
    let scratch = Scratch::new("identity");
    let history = scratch.0.join("wrfout_d01_2026-08-19_01_00_00");
    write_history(&history);
    let frames = scratch.0.join("frames");
    write_frames(&frames, true, latlon_lattice(0.0), &[("2026-08-19T01:00:00Z", 1)], &|_, y, x| composite(y, x));
    let shifted = scratch.0.join("frames-shifted");
    write_frames(&shifted, true, latlon_lattice(0.0), &[("2026-08-19T01:00:00Z", 1)], &|_, y, x| composite(y, x) + 20.0);
    let window = scratch.0.join("windows").join("20260819T0100Z");
    write_window(&window, "2026-08-19T01:00:00Z", NY, NX, "forecast", &|y, x| composite(y, x));

    let mut args = base(&scratch.0, "identity");
    args.extend(panel_arg(format!("history={}", history.display())));
    args.extend(panel_arg(format!("frames=frames:{}@2026-08-19T01:00Z", frames.display())));
    args.extend(panel_arg(format!("window=ttenref:{}", window.display())));
    args.extend(panel_arg(format!("shifted=frames:{}@20260819T0100Z", shifted.display())));
    let path = rendered(&rw_compare(&args));
    let sheet = image::open(&path).unwrap().to_rgba8();
    let history_body = body(&panel(&sheet, 0, 0));
    assert!(history_body.pixels().any(|p| p.0 != history_body.get_pixel(0, 0).0), "the history panel drew nothing");
    assert!(body(&panel(&sheet, 0, 1)) == history_body, "the frame panel is not the history panel's map");
    assert!(body(&panel(&sheet, 0, 2)) == history_body, "the window panel is not the history panel's map");
    assert!(body(&panel(&sheet, 0, 3)) != history_body, "a frame 20 dBZ stronger drew alike");

    let receipt: Value = serde_json::from_slice(&std::fs::read(path.with_extension("json")).unwrap()).unwrap();
    let columns = receipt["rows"][0]["columns"].as_array().unwrap();
    assert_eq!(columns[0]["kind"], "model");
    assert_eq!(columns[1]["kind"], "external");
    assert_eq!(columns[1]["op"], "frames");
    assert_eq!(columns[1]["source"]["record"]["causal"], true);
    assert_eq!(columns[1]["source"]["record"]["covered_cells"], NX * NY);
    assert_eq!(columns[1]["run_lead_seconds"], 3600);
    assert_eq!(columns[2]["op"], "ttenref");
    assert_eq!(columns[2]["source"]["record"]["lead_class"], "forecast");
    assert_eq!(columns[2]["source"]["record"]["echo_columns"], NX * NY);
}

#[test]
fn a_row_of_frames_and_windows_alone_takes_its_grid_from_panel_grid() {
    let scratch = Scratch::new("alone");
    let history = scratch.0.join("wrfout_d01_2026-08-19_01_00_00");
    write_history(&history);
    let frames = scratch.0.join("frames");
    // Three members: the reduction is the probability-matched mean, and a
    // cell uncovered in two of three members is uncovered.
    write_frames(&frames, false, latlon_lattice(0.0), &[("2026-08-19T00:30:00Z", 3), ("2026-08-19T01:00:00Z", 3)],
        &|member, y, x| if member > 0 && x == 0 { f32::NAN } else { composite(y, x) + member as f32 });
    let window = scratch.0.join("w").join("20260819T0030Z");
    write_window(&window, "2026-08-19T00:30:00Z", NY, NX, "oracle", &|_, x| if x < 4 { -99.0 } else { 40.0 });

    let mut args = base(&scratch.0, "alone");
    args.extend(["--panel-grid".to_string(), history.display().to_string()]);
    args.extend(panel_arg(format!("nowcast=frames:{}@2026-08-19T00:30Z", frames.display())));
    args.extend(panel_arg(format!("windows=ttenref:{}", window.display())));
    let path = rendered(&rw_compare(&args));
    let receipt: Value = serde_json::from_slice(&std::fs::read(path.with_extension("json")).unwrap()).unwrap();
    let columns = receipt["rows"][0]["columns"].as_array().unwrap();
    assert_eq!(receipt["rows"][0]["valid"], "2026-08-19T00:30:00+00:00");
    assert_eq!(columns[0]["source"]["record"]["causal"], false);
    assert_eq!(columns[0]["source"]["record"]["members"], 3);
    assert_eq!(columns[0]["source"]["record"]["covered_cells"], NX * NY - NY, "column 0 is uncovered in 2 of 3 members");
    assert_eq!(columns[1]["source"]["record"]["clear_columns"], 4 * NY);
    assert_eq!(columns[1]["source"]["record"]["lead_class"], "oracle");

    let mut no_grid = base(&scratch.0, "nogrid");
    no_grid.extend(panel_arg(format!("nowcast=frames:{}@2026-08-19T00:30Z", frames.display())));
    assert!(refused(&rw_compare(&no_grid)).contains("--panel-grid"));
}

#[test]
fn wrong_times_grids_corners_and_bytes_are_refused() {
    let scratch = Scratch::new("refusals");
    let history = scratch.0.join("wrfout_d01_2026-08-19_01_00_00");
    write_history(&history);

    // A frame of another valid time in a history file's row.
    let frames = scratch.0.join("frames");
    write_frames(&frames, true, latlon_lattice(0.0), &[("2026-08-19T02:00:00Z", 1)], &|_, y, x| composite(y, x));
    let mut late = base(&scratch.0, "late");
    late.extend(panel_arg(format!("history={}", history.display())));
    late.extend(panel_arg(format!("frames=frames:{}@2026-08-19T02:00Z", frames.display())));
    assert!(refused(&rw_compare(&late)).contains("one row of a sheet is one valid time"));

    // A valid time the frame set does not hold.
    let mut missing = base(&scratch.0, "missing");
    missing.extend(panel_arg(format!("history={}", history.display())));
    missing.extend(panel_arg(format!("frames=frames:{}@2026-08-19T01:00Z", frames.display())));
    assert!(refused(&rw_compare(&missing)).contains("holds no frame valid at"));

    // Corners that do not reproduce from the lattice numbers.
    let bent = scratch.0.join("bent");
    write_frames(&bent, true, latlon_lattice(0.01), &[("2026-08-19T01:00:00Z", 1)], &|_, y, x| composite(y, x));
    let mut corners = base(&scratch.0, "corners");
    corners.extend(panel_arg(format!("history={}", history.display())));
    corners.extend(panel_arg(format!("frames=frames:{}@2026-08-19T01:00Z", bent.display())));
    assert!(refused(&rw_compare(&corners)).contains("drawn in the wrong place"));

    // Frame bytes that are not the bytes the receipt hashed.
    let tampered = scratch.0.join("tampered");
    write_frames(&tampered, true, latlon_lattice(0.0), &[("2026-08-19T01:00:00Z", 1)], &|_, y, x| composite(y, x));
    let file = tampered.join("20260819T010000Z").join("refc.f32");
    let mut bytes = std::fs::read(&file).unwrap();
    bytes[0] ^= 0x40;
    std::fs::write(&file, bytes).unwrap();
    let mut hashed = base(&scratch.0, "hashed");
    hashed.extend(panel_arg(format!("history={}", history.display())));
    hashed.extend(panel_arg(format!("frames=frames:{}@2026-08-19T01:00Z", tampered.display())));
    assert!(refused(&rw_compare(&hashed)).contains("do not hash"));

    // A window made for another grid.
    let small = scratch.0.join("small").join("20260819T0100Z");
    write_window(&small, "2026-08-19T01:00:00Z", 10, 10, "forecast", &|_, _| 30.0);
    let mut grid = base(&scratch.0, "grid");
    grid.extend(panel_arg(format!("history={}", history.display())));
    grid.extend(panel_arg(format!("window=ttenref:{}", small.display())));
    assert!(refused(&rw_compare(&grid)).contains("nothing is regridded here"));

    // A frames panel on a product that is not composite reflectivity.
    let mut product = base(&scratch.0, "product");
    product[5] = "t2m".to_string();
    product.extend(panel_arg(format!("history={}", history.display())));
    product.extend(panel_arg(format!("frames=frames:{}@2026-08-19T02:00Z", frames.display())));
    assert!(refused(&rw_compare(&product)).contains("composite reflectivity only"));
}

#[test]
fn the_reduction_keeps_the_member_distribution_and_the_majority_rule() {
    let scratch = Scratch::new("reduce");
    let frames = scratch.0.join("frames");
    write_frames(&frames, true, latlon_lattice(0.0), &[("2026-08-19T01:00:00Z", 3)],
        &|member, y, x| if y == 0 && member < 2 { f32::NAN } else { (member * 10) as f32 + x as f32 });
    let receipt = rw_wrfbatch::nowcast_frames::read_frames_receipt(&frames).unwrap();
    let lat: Vec<f32> = (0..NY).flat_map(|y| (0..NX).map(move |_| 36.0 + 0.05 * y as f32)).collect();
    let lon: Vec<f32> = (0..NY).flat_map(|_| (0..NX).map(|x| -98.0 + 0.05 * x as f32)).collect();
    let valid = rw_wrfbatch::nowcast_frames::parse_utc("2026-08-19T01:00Z").unwrap();
    let plane = rw_wrfbatch::nowcast_frames::frame_on_grid(&receipt, valid, &lat, &lon, NY, NX).unwrap();
    // Row 0 is uncovered in two of three members: uncovered.
    assert!(plane.values[..NX].iter().all(|v| v.is_nan()));
    // Elsewhere the PMM's values are drawn from the pooled member values:
    // its maximum is the largest member value, never a smoothed mean.
    let finite: Vec<f32> = plane.values.iter().copied().filter(|v| v.is_finite()).collect();
    let max = finite.iter().copied().fold(f32::MIN, f32::max);
    assert_eq!(max, 20.0 + (NX - 1) as f32);
    assert_eq!(plane.covered_cells, NX * (NY - 1));
}
