//! The N-panel comparison sheet through the real `rw_compare` binary: any
//! number of labelled model panels beside the references in a caller-given
//! column order, the model panel identical to the panel the two-input sheet
//! draws for the same file, ensemble reductions, rows and refusals.

mod stored_plane_fixture;

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use chrono::NaiveDate;
use serde_json::Value;
use wx_core::grib2::writer::{Grib2Writer, MessageBuilder, PackingMethod};
use wx_core::grib2::{GridDefinition, ProductDefinition};

struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        let root = std::env::temp_dir().join(format!("rw-compare-panels-{tag}-{}", std::process::id()));
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

const WIDTH: u32 = 480;
const HEIGHT: u32 = 360;

/// An instantaneous surface DSWRF message at `hour`.
fn flux_message(hour: u32, value: f64) -> MessageBuilder {
    let grid = GridDefinition {
        template: 0,
        nx: stored_plane_fixture::NX as u32,
        ny: stored_plane_fixture::NY as u32,
        lat1: 36.0,
        lon1: -98.0,
        lat2: 36.0 + 0.05 * (stored_plane_fixture::NY - 1) as f64,
        lon2: -98.0 + 0.05 * (stored_plane_fixture::NX - 1) as f64,
        dx: 0.05,
        dy: 0.05,
        scan_mode: 0x40,
        ..Default::default()
    };
    MessageBuilder::new(0, vec![value; stored_plane_fixture::NX * stored_plane_fixture::NY])
        .center(7, 0)
        .reference_time(NaiveDate::from_ymd_opt(2026, 8, 19).unwrap().and_hms_opt(0, 0, 0).unwrap())
        .grid(grid)
        .product(ProductDefinition {
            template: 0,
            parameter_category: 4,
            parameter_number: 7,
            time_range_unit: 1,
            forecast_time: hour,
            level_type: 1,
            level_value: 0.0,
            ..Default::default()
        })
        .packing(PackingMethod::Simple { bits_per_value: 20 })
}

/// A reference directory holding the 00Z cycle's f01 and f02 files.
fn reference_dir(root: &Path) -> PathBuf {
    let dir = root.join("reference");
    std::fs::create_dir_all(&dir).unwrap();
    for lead in [1u32, 2] {
        let bytes = Grib2Writer::new()
            .add_message(flux_message(1, 510.0))
            .add_message(flux_message(2, 480.0))
            .to_bytes()
            .unwrap();
        std::fs::write(dir.join(format!("hrrr.t00z.wrfsfcf{lead:02}.grib2")), bytes).unwrap();
    }
    dir
}

/// A shortwave frame at `lead_seconds` with a constant flux, in its own
/// directory so two members at one valid time keep their own file names.
fn frame(root: &Path, member: &str, lead_seconds: i64, flux: f32) -> PathBuf {
    let dir = root.join(member);
    std::fs::create_dir_all(&dir).unwrap();
    stored_plane_fixture::write_regular_shortwave_frame(&dir, lead_seconds, flux)
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
        "--products", "swdown", "--offline", "--layout", "flat",
        "--width", &WIDTH.to_string(), "--height", &HEIGHT.to_string(),
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

fn rendered(output: &Output) -> PathBuf {
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "{stdout}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let line = stdout
        .lines()
        .find(|line| line.starts_with("RENDERED\tswdown\t"))
        .unwrap_or_else(|| panic!("no RENDERED line: {stdout}"));
    PathBuf::from(line.split('\t').last().unwrap())
}

/// Panel `(row, column)` of a sheet, cut at the composition's own origin.
fn panel(sheet: &image::RgbaImage, row: u32, column: u32) -> image::RgbaImage {
    let (x, y) = rw_wrfbatch::compare::sheet_panel_origin(row, column, WIDTH, HEIGHT);
    image::imageops::crop_imm(sheet, x, y, WIDTH, HEIGHT).to_image()
}

#[test]
fn a_file_panel_is_the_panel_the_two_input_sheet_draws_for_that_file() {
    let scratch = Scratch::new("identity");
    let refs = reference_dir(&scratch.0);
    let a = frame(&scratch.0, "a", 3600, 535.0);
    let b = frame(&scratch.0, "b", 3600, 600.0);

    // The existing two-input invocation: run | HRRR | run minus HRRR.
    let mut old = base(&scratch.0, "old");
    old.extend(["--reference", "hrrr", "--reference-dir", refs.to_str().unwrap(), "--run-label", "Member A"].map(String::from));
    old.push(a.to_str().unwrap().to_string());
    let old_sheet = image::open(rendered(&rw_compare(&old))).unwrap().to_rgba8();

    // The N-panel sheet: HRRR first, then the two members.
    let mut new = base(&scratch.0, "new");
    new.extend(["--reference", "hrrr", "--reference-dir", refs.to_str().unwrap(), "--order", "hrrr,1,2", "--sheet-name", "three"].map(String::from));
    new.extend(["--panel".to_string(), format!("Member A={}", a.display())]);
    new.extend(["--panel".to_string(), format!("Member B={}", b.display())]);
    let path = rendered(&rw_compare(&new));
    assert_eq!(path.file_name().unwrap(), "three.png");
    let sheet = image::open(&path).unwrap().to_rgba8();
    let metrics = rw_wrfbatch::compare::sheet_metrics(WIDTH);
    assert_eq!(sheet.dimensions(), (3 * WIDTH + 2 * metrics.gap, HEIGHT + metrics.header_height));
    assert!(panel(&sheet, 0, 1) == panel(&old_sheet, 0, 0), "the member A panel is not the two-input sheet's run panel");
    assert!(panel(&sheet, 0, 1) != panel(&sheet, 0, 2), "two members with different planes drew alike");

    let receipt: Value = serde_json::from_slice(&std::fs::read(path.with_extension("json")).unwrap()).unwrap();
    assert_eq!(receipt["schema"], "gpuwm.compare-sheet.v3");
    assert_eq!(receipt["order"], serde_json::json!(["hrrr", "1", "2"]));
    let columns = receipt["rows"][0]["columns"].as_array().unwrap();
    assert_eq!(columns[0]["name"], "hrrr");
    assert_eq!(columns[1]["label"], "Member A");
    assert_eq!(columns[2]["label"], "Member B");
    assert_eq!(columns[2]["op"], "field");
}

#[test]
fn reductions_rows_and_model_only_sheets() {
    let scratch = Scratch::new("reductions");
    let refs = reference_dir(&scratch.0);
    let members: Vec<PathBuf> = [("m0", 500.0f32), ("m1", 600.0), ("m2", 700.0)]
        .iter()
        .map(|(name, flux)| frame(&scratch.0, name, 3600, *flux))
        .collect();
    let mean_file = frame(&scratch.0, "mean", 3600, 600.0);
    let list = members.iter().map(|m| m.display().to_string()).collect::<Vec<_>>().join(",");

    // An ensemble mean of 500, 600, 700 is the 600 member's map; only the
    // panel's left subtitle (which names the reduction) may differ.
    let mut args = base(&scratch.0, "mean");
    args.extend(["--sheet-name", "mean"].map(String::from));
    args.extend(["--panel".to_string(), format!("X=mean:{list}")]);
    args.extend(["--panel".to_string(), format!("X={}", mean_file.display())]);
    args.extend(["--panel".to_string(), format!("X=max:{list}")]);
    args.extend(["--panel".to_string(), format!("X=pmm:{list}")]);
    let sheet = image::open(rendered(&rw_compare(&args))).unwrap().to_rgba8();
    let body = |image: &image::RgbaImage| image::imageops::crop_imm(image, 0, HEIGHT / 4, WIDTH, HEIGHT - HEIGHT / 4).to_image();
    assert!(body(&panel(&sheet, 0, 0)) == body(&panel(&sheet, 0, 1)), "the mean of 500, 600, 700 is not drawn as 600");
    assert!(body(&panel(&sheet, 0, 2)) != body(&panel(&sheet, 0, 1)), "the maximum drew as the mean");

    // Two rows, one per valid time, each with its own reference lead.
    let late: Vec<PathBuf> = ["n0", "n1"].iter().map(|name| frame(&scratch.0, name, 7200, 650.0)).collect();
    let mut rows = base(&scratch.0, "rows");
    rows.extend(["--reference", "hrrr", "--reference-dir", refs.to_str().unwrap(), "--sheet-name", "rows"].map(String::from));
    rows.extend(["--panel".to_string(), format!("member={}", members[0].display())]);
    rows.extend(["--panel".to_string(), format!("mean=mean:{},{}", members[0].display(), members[1].display())]);
    rows.push("--row".to_string());
    rows.extend(["--panel".to_string(), format!("member={}", late[0].display())]);
    rows.extend(["--panel".to_string(), format!("mean=mean:{},{}", late[0].display(), late[1].display())]);
    let path = rendered(&rw_compare(&rows));
    let metrics = rw_wrfbatch::compare::sheet_metrics(WIDTH);
    assert_eq!(
        image::image_dimensions(&path).unwrap(),
        (3 * WIDTH + 2 * metrics.gap, 2 * HEIGHT + metrics.gap + metrics.header_height)
    );
    let receipt: Value = serde_json::from_slice(&std::fs::read(path.with_extension("json")).unwrap()).unwrap();
    assert_eq!(receipt["rows"][0]["columns"][2]["forecast_hour"], 1);
    assert_eq!(receipt["rows"][1]["columns"][2]["forecast_hour"], 2);

    // No reference at all: model panels compared with each other.
    let mut only = base(&scratch.0, "only");
    only.extend(["--panel".to_string(), format!("one={}", members[0].display())]);
    only.extend(["--panel".to_string(), format!("two={}", members[1].display())]);
    let path = rendered(&rw_compare(&only));
    assert_eq!(image::image_dimensions(&path).unwrap().0, 2 * WIDTH + metrics.gap);
}

#[test]
fn an_order_that_drops_a_panel_or_a_row_of_another_valid_time_is_refused() {
    let scratch = Scratch::new("refusals");
    let refs = reference_dir(&scratch.0);
    let a = frame(&scratch.0, "a", 3600, 535.0);
    let late = frame(&scratch.0, "late", 7200, 535.0);
    let mut args = base(&scratch.0, "drop");
    args.extend(["--reference", "hrrr", "--reference-dir", refs.to_str().unwrap(), "--order", "hrrr,1"].map(String::from));
    args.extend(["--panel".to_string(), format!("A={}", a.display())]);
    args.extend(["--panel".to_string(), format!("B={}", a.display())]);
    let output = rw_compare(&args);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("leaves out 2"), "{}", String::from_utf8_lossy(&output.stderr));

    let mut mixed = base(&scratch.0, "mixed");
    mixed.extend(["--panel".to_string(), format!("A={}", a.display())]);
    mixed.extend(["--panel".to_string(), format!("B={}", late.display())]);
    let output = rw_compare(&mixed);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("one row of a sheet is one valid time"));

    let mut positional = base(&scratch.0, "positional");
    positional.extend(["--panel".to_string(), format!("A={}", a.display()), a.display().to_string()]);
    let output = rw_compare(&positional);
    assert_eq!(output.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&output.stderr).contains("a panel with no label"));
}
