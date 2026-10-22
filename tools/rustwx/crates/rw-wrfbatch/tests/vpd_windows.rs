//! Default WRF history must supply the three default day-window VPD products.
//! The rolling 2.9.4 GFS run stored hourly history but imported no vpd_2m plane.
use std::path::{Path, PathBuf};
use std::process::Command;

use netcdf_writer::{AttrValue, NcFormat, NcType, NcWriter, Schema, VarData};
use rusty_weather::render_all::windowed_store::compute_windowed_products;
use rw_wrfbatch::wrf_process::{
    WrfProcessMessage, WrfProcessOptions, required_fields_for_products, spawn_process_paths,
};

mod support;
use support::Scratch;

const NX: usize = 6;
const NY: usize = 4;
const PRODUCTS: [&str; 3] = ["2m_vpd_0_24h_max", "2m_vpd_0_24h_min", "2m_vpd_0_24h_range"];

// Only the three native thermodynamic surface inputs, with no wind/radiation
// dependency. T exists solely for WrfFile's dimension probe.
fn history(root: &Path) -> PathBuf {
    let path = root.join("wrfout_d01_2026-01-01_00_00_00");
    let mut schema = Schema::new(NcFormat::Offset64);
    for name in ["START_DATE", "SIMULATION_START_DATE"] {
        schema
            .put_global_attr(name, AttrValue::Text("2026-01-01_00:00:00".into()))
            .unwrap();
    }
    let time = schema.def_dim("Time", 0, true).unwrap();
    let strlen = schema.def_dim("DateStrLen", 19, false).unwrap();
    let z = schema.def_dim("bottom_top", 2, false).unwrap();
    let y = schema.def_dim("south_north", NY, false).unwrap();
    let x = schema.def_dim("west_east", NX, false).unwrap();
    let times = schema
        .def_var("Times", NcType::Char, &[time, strlen])
        .unwrap();
    let theta = schema
        .def_var("T", NcType::Float, &[time, z, y, x])
        .unwrap();
    let mut vars = Vec::new();
    for (name, units) in [
        ("XLAT", "degrees_north"),
        ("XLONG", "degrees_east"),
        ("T2", "K"),
        ("Q2", "kg kg-1"),
        ("PSFC", "Pa"),
        ("HGT", "m"),
    ] {
        let id = schema.def_var(name, NcType::Float, &[time, y, x]).unwrap();
        schema
            .put_var_attr(id, "units", AttrValue::Text(units.into()))
            .unwrap();
        vars.push((name, id));
    }
    let mut writer = NcWriter::create(&path, schema).unwrap();
    let init = chrono::NaiveDate::from_ymd_opt(2026, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();
    for hour in 0..=24 {
        let stamp = (init + chrono::Duration::hours(hour))
            .format("%Y-%m-%d_%H:%M:%S")
            .to_string();
        writer
            .write_record(hour as u64, times, VarData::Char(stamp.as_bytes()))
            .unwrap();
        writer
            .write_record(hour as u64, theta, VarData::F32(&vec![0.0; 2 * NX * NY]))
            .unwrap();
        for &(name, id) in &vars {
            let values: Vec<f32> = (0..NX * NY)
                .map(|i| match name {
                    "XLAT" => 35.0 + (i / NX) as f32 * 0.1,
                    "XLONG" => -97.0 + (i % NX) as f32 * 0.1,
                    "T2" if i == 21 => 1.0e35, // finite WRF fill
                    "Q2" if i == 20 => 0.5,    // supersaturated, zero deficit
                    "Q2" if i == 22 => f32::NAN,
                    "PSFC" if i == 23 => 1.0e35,
                    "T2" => 280.0 + hour as f32 + i as f32 * 0.1,
                    "Q2" => 0.005,
                    "PSFC" => 100_000.0,
                    _ => 0.0,
                })
                .collect();
            writer
                .write_record(hour as u64, id, VarData::F32(&values))
                .unwrap();
        }
    }
    writer.finish().unwrap();
    path
}

fn expected(hour: u16, cell: usize) -> f64 {
    // Independent Bolton approximation, with native f32 inputs promoted.
    // Ambaum differs by at most 0.157 hPa across this fixture (up to 32.75 C).
    // Allow 0.2 hPa for that independent approximation; wrong units, missing
    // hours, and a zero-filled plane still fail by much larger margins.
    let tc = f64::from(280.0_f32 + hour as f32 + cell as f32 * 0.1) - 273.15;
    let q = f64::from(if cell == 20 { 0.5_f32 } else { 0.005_f32 });
    (6.112 * (17.67 * tc / (tc + 243.5)).exp() - q * 1000.0 / (0.622 + q)).max(0.0)
}

#[test]
fn default_and_scoped_imports_supply_all_three_vpd_windows() {
    let scratch = Scratch::new("vpd-windows", "import");
    let path = history(&scratch.0);
    for (label, options) in [
        ("default", WrfProcessOptions::default()),
        (
            "windowed",
            WrfProcessOptions {
                required_fields: required_fields_for_products("windowed"),
                ..WrfProcessOptions::default()
            },
        ),
        (
            "named",
            WrfProcessOptions {
                required_fields: required_fields_for_products(&PRODUCTS.join(",")),
                ..WrfProcessOptions::default()
            },
        ),
    ] {
        let store = scratch.0.join(label);
        let task = spawn_process_paths(vec![path.clone()], store.clone(), options);
        let summary = loop {
            if let WrfProcessMessage::Done(result) = task.rx.recv().unwrap() {
                break result.unwrap();
            }
        };
        for hour in 0..=24 {
            let reader = rw_store::reader::HourReader::open(
                &store
                    .join(&summary.model)
                    .join(&summary.run)
                    .join(format!("f{hour:03}.rws")),
            )
            .unwrap();
            let values = reader
                .read_full_2d("vpd_2m")
                .expect("WRF import must store VPD in hPa");
            for (i, value) in values.iter().enumerate() {
                if i >= 21 {
                    assert!(
                        value.is_nan(),
                        "{label} missing input became valid VPD at {i}"
                    );
                } else {
                    assert!(
                        (f64::from(*value) - expected(hour, i)).abs() < 0.2,
                        "{label} F{hour:03} cell {i}: {value}"
                    );
                }
            }
        }
        let outcome = compute_windowed_products(
            &store,
            &summary.model,
            &summary.run,
            &(0..=24).collect::<Vec<_>>(),
            &PRODUCTS.map(str::to_string),
        )
        .unwrap();
        assert!(
            outcome.blockers.is_empty(),
            "{label}: {:?}",
            outcome.blockers
        );
        assert_eq!(outcome.grids.len(), 3);
        for grid in outcome.grids {
            assert_eq!(grid.units, "hPa");
            assert_eq!(grid.hours_used, (1..=24).collect::<Vec<_>>());
            for (i, value) in grid.values.iter().enumerate() {
                if i >= 21 {
                    assert!(value.is_nan());
                    continue;
                }
                let want = match grid.slug.as_str() {
                    "2m_vpd_0_24h_max" => expected(24, i),
                    "2m_vpd_0_24h_min" => expected(1, i),
                    _ => expected(24, i) - expected(1, i),
                };
                assert!(
                    (value - want).abs() < 0.2,
                    "{label} {} cell {i}: {value} vs {want}",
                    grid.slug
                );
            }
        }
    }
}

#[test]
fn real_renderer_draws_all_three_day_vpd_products_from_wrf_history() {
    let scratch = Scratch::new("vpd-windows", "renderer");
    let path = history(&scratch.0);
    let pictures = std::env::var_os("VPD_WINDOW_PICTURE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| scratch.0.join("png"));
    let output = Command::new(env!("CARGO_BIN_EXE_rw_wrfbatch"))
        .args([
            "--products",
            &PRODUCTS.join(","),
            "--width",
            "480",
            "--height",
            "360",
        ])
        .arg("--store-root")
        .arg(scratch.0.join("store"))
        .arg("--out-dir")
        .arg(pictures)
        .arg(path)
        .env("CUDA_VISIBLE_DEVICES", "")
        .env("RAYON_NUM_THREADS", "1")
        .env("RUSTWX_BATCH_RENDER_THREADS", "1")
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "{stdout}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    for product in PRODUCTS {
        assert!(
            stdout
                .lines()
                .any(|line| line.starts_with(&format!("RENDERED {product} "))),
            "{stdout}"
        );
    }
}
