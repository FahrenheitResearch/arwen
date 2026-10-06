//! The native comparison executable must reach the stored SWDOWN plane
//! and the reference's instantaneous surface DSWRF, on one physical scale.

mod stored_plane_fixture;

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use chrono::NaiveDate;
use serde_json::{Value, json};
use wx_core::grib2::writer::{Grib2Writer, MessageBuilder, PackingMethod, StatisticalInterval};
use wx_core::grib2::{GridDefinition, ProductDefinition};

struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        let root = std::env::temp_dir().join(format!("rw-sw-compare-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        Self(root)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn reference_message(parameter: u8, level: u8, hour: u32, value: f64) -> MessageBuilder {
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
    MessageBuilder::new(
        0,
        vec![value; stored_plane_fixture::NX * stored_plane_fixture::NY],
    )
    .center(7, 0)
    .reference_time(
        NaiveDate::from_ymd_opt(2026, 8, 19)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap(),
    )
    .grid(grid)
    .product(ProductDefinition {
        template: 0,
        parameter_category: 4,
        parameter_number: parameter,
        time_range_unit: 1,
        forecast_time: hour,
        level_type: level,
        level_value: 0.0,
        ..Default::default()
    })
    .packing(PackingMethod::Simple { bits_per_value: 20 })
}

fn write_reference(path: &Path, alias: u8, instant: bool) {
    let end = NaiveDate::from_ymd_opt(2026, 8, 19)
        .unwrap()
        .and_hms_opt(1, 0, 0)
        .unwrap();
    let average = reference_message(alias, 1, 0, 950.0).statistical_interval(StatisticalInterval {
        end_time: end,
        statistical_process: 0,
        time_unit: 1,
        length: 1,
    });
    let mut writer = Grib2Writer::new()
        .add_message(average)
        .add_message(reference_message(alias, 8, 1, 900.0))
        .add_message(reference_message(alias, 1, 2, 850.0));
    if instant {
        writer = writer.add_message(reference_message(alias, 1, 1, 510.0));
    }
    std::fs::write(path, writer.to_bytes().unwrap()).unwrap();
}

fn compare(root: &Path, frame: &Path, reference: &Path) -> Output {
    compare_product(
        Path::new(env!("CARGO_BIN_EXE_rw_compare")),
        root,
        frame,
        reference,
        "swdown",
    )
}

fn compare_product(
    binary: &Path,
    root: &Path,
    frame: &Path,
    reference: &Path,
    product: &str,
) -> Output {
    Command::new(binary)
        .args([
            "--store-root",
            root.join("store").to_str().unwrap(),
            "--out-dir",
            root.join("png").to_str().unwrap(),
            "--reference",
            "hrrr",
            "--reference-file",
            reference.to_str().unwrap(),
            "--products",
            product,
            "--offline",
            "--width",
            "480",
            "--height",
            "360",
            frame.to_str().unwrap(),
        ])
        .env("RAYON_NUM_THREADS", "2")
        .env("CUDA_VISIBLE_DEVICES", "")
        .output()
        .unwrap()
}

#[test]
fn native_sw_sheet_reads_both_dswrf_aliases_without_averages_or_wrong_leads() {
    for alias in [7, 192] {
        let scratch = Scratch::new(&format!("alias-{alias}"));
        let frame = stored_plane_fixture::write_regular_shortwave_frame(&scratch.0, 3600, 535.0);
        let reference = scratch.0.join("hrrr.t00z.wrfsfcf01.grib2");
        write_reference(&reference, alias, true);
        let output = compare(&scratch.0, &frame, &reference);
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(output.status.success(), "{stdout}\n{stderr}");
        assert!(
            stdout.contains("FINISHED rendered=1 skipped=0 failed=0"),
            "{stdout}"
        );
        let rendered = stdout
            .lines()
            .find(|line| line.starts_with("RENDERED\tswdown\t"))
            .unwrap();
        let png = PathBuf::from(rendered.split('\t').nth(3).unwrap());
        let sidecar = png.with_extension("json");
        let receipt: Value = serde_json::from_slice(&std::fs::read(&sidecar).unwrap()).unwrap();
        assert_eq!(receipt["product"], "swdown");
        assert_eq!(receipt["display_units"], "W m-2");
        assert_eq!(receipt["reference_message"], "DSWRF, surface, instant");
        assert_eq!(receipt["reference_forecast_hour"], 1);
        assert_eq!(
            receipt["difference"]["points"],
            stored_plane_fixture::NX * stored_plane_fixture::NY
        );
        assert_eq!(receipt["difference"]["mean"], 25.0);
        assert_eq!(receipt["difference"]["rms"], 25.0);
        let metrics = rw_wrfbatch::compare::sheet_metrics(480);
        assert_eq!(
            image::image_dimensions(&png).unwrap(),
            (1440 + 2 * metrics.gap, 360 + metrics.header_height)
        );
        // Keep only renderer artifacts and receipts when the artifact proof
        // names an evidence folder. The raw analytic fixtures stay scratch.
        if let Some(dir) = std::env::var_os("RW_SW_COMPARE_RECEIPTS") {
            let dir = PathBuf::from(dir);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::copy(&png, dir.join(format!("swdown-alias-{alias}.png"))).unwrap();
            std::fs::copy(&sidecar, dir.join(format!("swdown-alias-{alias}.json"))).unwrap();
            std::fs::write(
                dir.join(format!("swdown-alias-{alias}.log")),
                format!("{stdout}\n{stderr}"),
            )
            .unwrap();
            std::fs::write(dir.join(format!("swdown-alias-{alias}-fixture.json")), serde_json::to_vec_pretty(&json!({
                "fixture": "analytic renderer regression, not forecast evidence",
                "run_swdown_w_m2": 535.0, "reference_instantaneous_w_m2": 510.0,
                "wrong_average_w_m2": 950.0, "wrong_level_w_m2": 900.0, "wrong_lead_w_m2": 850.0,
                "wrfout_deleted_bytes": std::fs::metadata(&frame).unwrap().len(),
                "grib2_deleted_bytes": std::fs::metadata(&reference).unwrap().len(),
            })).unwrap()).unwrap();
        }
    }
}

#[test]
fn native_sw_sheet_refuses_average_without_instantaneous_surface_flux() {
    let scratch = Scratch::new("average-only");
    let frame = stored_plane_fixture::write_regular_shortwave_frame(&scratch.0, 3600, 535.0);
    let reference = scratch.0.join("hrrr.t00z.wrfsfcf01.grib2");
    write_reference(&reference, 7, false);
    let output = compare(&scratch.0, &frame, &reference);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        !output.status.success(),
        "an averaged reference was drawn: {stdout}"
    );
    assert!(stdout.contains("SKIPPED\tswdown\t"), "{stdout}");
    assert!(
        stdout.contains("no downward_shortwave_radiation_flux"),
        "{stdout}"
    );
    assert!(
        stdout.contains("FINISHED rendered=0 skipped=1 failed=0"),
        "{stdout}"
    );
}

#[test]
fn native_catalog_exposes_sw_product() {
    let output = Command::new(env!("CARGO_BIN_EXE_rw_compare"))
        .arg("--list-products")
        .output()
        .unwrap();
    assert!(output.status.success());
    assert!(String::from_utf8_lossy(&output.stdout).contains(
        "PRODUCT\tswdown\tsurface_downward_shortwave\tSurface downward shortwave\tdifference=auto"
    ));
}

#[test]
#[ignore = "requires RW_SW_COMPARE_BASELINE pointing at the staging executable"]
fn existing_temperature_sheet_is_byte_identical_to_staging() {
    let baseline =
        PathBuf::from(std::env::var_os("RW_SW_COMPARE_BASELINE").expect("staging executable"));
    let scratch = Scratch::new("old-temperature");
    let frame = stored_plane_fixture::write_regular_shortwave_frame(&scratch.0, 3600, 535.0);
    let reference = scratch.0.join("hrrr.t00z.wrfsfcf01.grib2");
    let temperature = reference_message(0, 103, 1, 295.0).product(ProductDefinition {
        template: 0,
        parameter_category: 0,
        parameter_number: 0,
        time_range_unit: 1,
        forecast_time: 1,
        level_type: 103,
        level_value: 2.0,
        ..Default::default()
    });
    std::fs::write(
        &reference,
        Grib2Writer::new()
            .add_message(temperature)
            .add_message(reference_message(7, 1, 1, 510.0))
            .to_bytes()
            .unwrap(),
    )
    .unwrap();
    let mut pngs = Vec::new();
    for (label, binary) in [
        ("staging", baseline),
        ("candidate", PathBuf::from(env!("CARGO_BIN_EXE_rw_compare"))),
    ] {
        let products = if label == "candidate" {
            "t2m,swdown"
        } else {
            "t2m"
        };
        let output = compare_product(
            &binary,
            &scratch.0.join(label),
            &frame,
            &reference,
            products,
        );
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            output.status.success(),
            "{label}: {stdout}\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let rendered = stdout
            .lines()
            .find(|line| line.starts_with("RENDERED\tt2m\t"))
            .unwrap();
        pngs.push(std::fs::read(rendered.split('\t').nth(3).unwrap()).unwrap());
    }
    assert_eq!(pngs[0], pngs[1], "the existing temperature picture changed");
    println!("BYTE_IDENTICAL\tt2m\t{} bytes", pngs[0].len());
}
