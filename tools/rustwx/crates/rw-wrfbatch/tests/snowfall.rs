use netcdf_writer::{AttrValue, NcFormat, NcType, NcWriter, Schema, VarData};
use rw_wrfbatch::wrf_process::compute_var;
use rw_wrfbatch::wrf_process::{WrfProcessMessage, WrfProcessOptions, spawn_process_paths};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::process::Command;
use wrf_core::WrfFile;

// Independent source-patch arithmetic, kept in the test only. This uses
// the original f64 maximum walk, f32 frozen-liquid sum, clamp and ratio.
fn patch_interval(file: &WrfFile, ti: usize, prev: &mut Vec<f32>, acc: &mut [f32]) {
    let cells = acc.len();
    let snow = compute_var(file, "SNOWNC", ti, None).unwrap();
    let graupel = compute_var(file, "GRAUPELNC", ti, None).ok();
    let current: Vec<f32> = snow
        .data
        .iter()
        .enumerate()
        .map(|(i, &s)| s as f32 + graupel.as_ref().map(|g| g.data[i] as f32).unwrap_or(0.))
        .collect();
    let t = compute_var(file, "temp", ti, Some("K")).unwrap();
    let p = compute_var(file, "pres", ti, Some("Pa")).unwrap();
    for i in 0..cells {
        let mut best = f64::NEG_INFINITY;
        for k in 0..file.nz {
            if k > 0 && p.data[k * cells + i] < 50000. {
                break;
            }
            let v = t.data[k * cells + i];
            if v.is_finite() && v > best {
                best = v;
            }
        }
        let tmax = best as f32;
        let d = 271.16 - tmax;
        let ratio = (if tmax > 271.16 { 12. + 2. * d } else { 12. + d }).max(0.);
        acc[i] += (current[i] - prev[i]).max(0.) * ratio;
    }
    *prev = current;
}

fn hash(v: &[f32]) -> String {
    let mut h = Sha256::new();
    for x in v {
        h.update(x.to_le_bytes());
    }
    format!("{:x}", h.finalize())
}

struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        let p = std::env::temp_dir().join(format!(
            "snowfall-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&p).unwrap();
        Self(p)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn frame(dir: &Path, hour: u32, snow: f32, graupel: f32, depth: f32, temp: f64) -> PathBuf {
    frame_series(dir, &[(hour, snow, graupel, depth, temp)])
}

fn frame_series(dir: &Path, records: &[(u32, f32, f32, f32, f64)]) -> PathBuf {
    const NX: usize = 24;
    const NY: usize = 18;
    const NZ: usize = 4;
    let mut s = Schema::new(NcFormat::Offset64);
    let time = s.def_dim("Time", 0, true).unwrap();
    let strlen = s.def_dim("DateStrLen", 19, false).unwrap();
    let z = s.def_dim("bottom_top", NZ, false).unwrap();
    let y = s.def_dim("south_north", NY, false).unwrap();
    let x = s.def_dim("west_east", NX, false).unwrap();
    for (n, v) in [
        ("START_DATE", "2026-01-01_00:00:00"),
        ("SIMULATION_START_DATE", "2026-01-01_00:00:00"),
    ] {
        s.put_global_attr(n, AttrValue::Text(v.into())).unwrap();
    }
    s.put_global_attr("MAP_PROJ", AttrValue::Ints(vec![6]))
        .unwrap();
    s.put_global_attr("GRID_ID", AttrValue::Ints(vec![2]))
        .unwrap();
    s.put_global_attr("DX", AttrValue::Floats(vec![750.]))
        .unwrap();
    let times = s.def_var("Times", NcType::Char, &[time, strlen]).unwrap();
    let mut surfaces = Vec::new();
    for (n, u) in [
        ("XLAT", "degree_north"),
        ("XLONG", "degree_east"),
        ("SNOWNC", "mm"),
        ("GRAUPELNC", "mm"),
        ("SNOWFALLAC", ""),
    ] {
        let id = s.def_var(n, NcType::Float, &[time, y, x]).unwrap();
        s.put_var_attr(id, "units", AttrValue::Text(u.into()))
            .unwrap();
        surfaces.push(id);
    }
    let t = s.def_var("T", NcType::Float, &[time, z, y, x]).unwrap();
    let p = s.def_var("P", NcType::Float, &[time, z, y, x]).unwrap();
    let pb = s.def_var("PB", NcType::Float, &[time, z, y, x]).unwrap();
    let label = format!("2026-01-01_{:02}:00:00", records[0].0);
    let path = dir.join(format!("wrfout_d02_{}", label.replace(':', "_")));
    let mut w = NcWriter::create(&path, s).unwrap();
    for (rec, &(hour, snow, graupel, depth, temp)) in records.iter().enumerate() {
        let rec = rec as u64;
        let label = format!("2026-01-01_{hour:02}:00:00");
        w.write_record(rec, times, VarData::Char(label.as_bytes()))
            .unwrap();
        let lat = (0..NY)
            .flat_map(|j| (0..NX).map(move |_| 47. + j as f32 * 0.01))
            .collect();
        let lon = (0..NY)
            .flat_map(|_| (0..NX).map(|i| -112. + i as f32 * 0.01))
            .collect();
        for (&id, v) in surfaces.iter().zip([
            lat,
            lon,
            vec![snow; NX * NY],
            vec![graupel; NX * NY],
            vec![depth; NX * NY],
        ]) {
            w.write_record(rec, id, VarData::F32(&v)).unwrap();
        }
        let pressures = [90000.0f64, 70000., 50000., 30000.];
        let theta: Vec<f32> = pressures
            .iter()
            .enumerate()
            .flat_map(|(k, &p)| {
                let tk = if k == 3 { 310. } else { temp - k as f64 };
                vec![(tk / (p / 100000.).powf(0.2857142857) - 300.) as f32; NX * NY]
            })
            .collect();
        w.write_record(rec, t, VarData::F32(&theta)).unwrap();
        w.write_record(rec, p, VarData::F32(&vec![0.; NX * NY * NZ]))
            .unwrap();
        w.write_record(
            rec,
            pb,
            VarData::F32(
                &pressures
                    .into_iter()
                    .flat_map(|p| vec![p as f32; NX * NY])
                    .collect::<Vec<_>>(),
            ),
        )
        .unwrap();
    }
    w.finish().unwrap();
    path
}

fn import(path: &Path, root: &Path, since: Option<&str>) -> (PathBuf, Vec<String>) {
    let task = spawn_process_paths(
        vec![path.to_path_buf()],
        root.to_path_buf(),
        WrfProcessOptions {
            core_fields: false,
            raw_extras: false,
            stored_planes: false,
            only: vec![
                "snowfall_window".into(),
                "snow_10to1_window".into(),
                "snow_kuchera_window".into(),
            ],
            snow_since: since.map(str::to_string),
            ..Default::default()
        },
    );
    loop {
        match task.rx.recv().unwrap() {
            WrfProcessMessage::Progress(_) => {}
            WrfProcessMessage::Done(r) => {
                let s = r.unwrap();
                return (root.join(s.model).join(s.run).join("f002.rws"), s.notes);
            }
        }
    }
}

#[test]
fn real_import_registry_render_and_points() {
    let s = Scratch::new();
    frame(&s.0, 0, 0., 0., 0., 272.16);
    let middle = frame(&s.0, 1, 1., 1., 18., 272.16);
    let end = frame(&s.0, 2, 2., 1., 30., 269.16);
    let (hour, notes) = import(&end, &s.0.join("store"), None);
    let reader = rw_store::reader::HourReader::open(&hour).unwrap();
    for (name, want) in [
        ("snowfall_window", 30.),
        ("snow_10to1_window", 30.),
        ("snow_kuchera_window", 34.),
    ] {
        let values = reader
            .read_full_2d(name)
            .unwrap_or_else(|e| panic!("{name}: {e}; {notes:?}"));
        assert_eq!(values.len(), 24 * 18);
        assert!(
            values.iter().all(|v| (*v - want).abs() < 0.002),
            "{name}: {}",
            values[0]
        );
        assert!(
            rusty_weather::render_all::known_product_slugs()
                .iter()
                .any(|s| s == name)
        );
        assert!(
            rustwx_products::direct::store_direct_recipe_slugs()
                .iter()
                .any(|s| s == name)
        );
    }
    let mut expected = vec![0.; 24 * 18];
    let mut prev = vec![0.; 24 * 18];
    for path in [&middle, &end] {
        patch_interval(&WrfFile::open(path).unwrap(), 0, &mut prev, &mut expected);
    }
    let actual = reader.read_full_2d("snow_kuchera_window").unwrap();
    assert_eq!(
        hash(&actual),
        hash(&expected),
        "full-field source-patch comparison"
    );
    println!(
        "synthetic_source_patch_sha256={} cells={}",
        hash(&actual),
        actual.len()
    );
    let (hour, _) = import(&end, &s.0.join("store"), Some("2026-01-01_01:00:00"));
    let reader = rw_store::reader::HourReader::open(&hour).unwrap();
    assert!((reader.read_full_2d("snow_kuchera_window").unwrap()[0] - 14.).abs() < 0.002);
    // A changed intermediate accumulator must invalidate cached totals.
    let old_run = hour.parent().unwrap().to_path_buf();
    let _ = middle;
    frame(&s.0, 1, 1.5, 1., 18., 272.16);
    let (new_hour, _) = import(&end, &s.0.join("store"), Some("2026-01-01_01:00:00"));
    assert_ne!(old_run, new_hour.parent().unwrap());
    assert!(
        (rw_store::reader::HourReader::open(&new_hour)
            .unwrap()
            .read_full_2d("snow_kuchera_window")
            .unwrap()[0]
            - 7.)
            .abs()
            < 0.002
    );

    let output = Command::new(env!("CARGO_BIN_EXE_rw_wrfbatch"))
        .args([
            "--products",
            "snowfall_window,snow_10to1_window,snow_kuchera_window",
            "--width",
            "480",
            "--height",
            "360",
        ])
        .arg("--store-root")
        .arg(s.0.join("cli-store"))
        .arg("--out-dir")
        .arg(s.0.join("png"))
        .arg(&end)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let point = Command::new(env!("CARGO_BIN_EXE_rw_points"))
        .args(["--point", "site,47.01,-111.99", "--out"])
        .arg(s.0.join("points.csv"))
        .arg(&end)
        .output()
        .unwrap();
    assert!(
        point.status.success(),
        "{}",
        String::from_utf8_lossy(&point.stderr)
    );
    let csv = std::fs::read_to_string(s.0.join("points.csv")).unwrap();
    assert!(csv.contains("TMAXCOL,KUCH_SLR,SNOWLVL_FT,GUST10"));
    let row: Vec<_> = csv.lines().nth(1).unwrap().split(',').collect();
    assert_eq!((row[5], row[6]), ("1", "1"));
    assert!((row[7].parse::<f64>().unwrap() - 47.01).abs() < 0.00001);
    assert!((row[21].parse::<f64>().unwrap() - 269.16).abs() < 0.0001);
    assert!((row[22].parse::<f64>().unwrap() - 14.).abs() < 0.0001);
    // Preserve a small fixture on the box only when explicitly requested.
    if let Ok(dir) = std::env::var("RW_SNOW_RECEIPT_DIR") {
        let dir = PathBuf::from(dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::copy(&end, dir.join(end.file_name().unwrap())).unwrap();
        for h in [0, 1] {
            let p = s.0.join(format!("wrfout_d02_2026-01-01_{h:02}_00_00"));
            std::fs::copy(&p, dir.join(p.file_name().unwrap())).unwrap();
        }
    }
}

#[test]
fn incomplete_history_cannot_publish_a_total() {
    let s = Scratch::new();
    let end = frame(&s.0, 2, 2., 1., 30., 269.16);
    let (_, notes) = import(&end, &s.0.join("store"), None);
    assert!(
        notes
            .iter()
            .any(|s| s.contains("exact snowfall baseline frame missing")),
        "{notes:?}"
    );
}

#[test]
fn an_incomplete_future_frame_does_not_break_a_completed_window() {
    let s = Scratch::new();
    frame(&s.0, 0, 0., 0., 0., 272.16);
    let end = frame(&s.0, 2, 2., 1., 30., 269.16);
    std::fs::write(s.0.join("wrfout_d02_2026-01-01_03_00_00"), []).unwrap();
    let file = WrfFile::open(&end).unwrap();
    let values = rw_wrfbatch::snowfall::window(&file, &end, 0, None, true).unwrap();
    assert!(values.iter().all(|v| (*v - 42.).abs() < 0.002));
    let (_, notes) = import(&end, &s.0.join("store"), Some("2026-01-01_00:00:00"));
    assert!(
        !notes
            .iter()
            .any(|n| n.contains("snowfall_window unavailable")),
        "{notes:?}"
    );
}

#[test]
fn supplied_planes_are_renderable_without_claiming_download_support() {
    for &(slug, field) in rw_wrfbatch::snowfall::PRODUCTS {
        for model in [
            rustwx_core::ModelId::WrfGdex,
            rustwx_core::ModelId::Hrrr,
            rustwx_core::ModelId::Aifs,
        ] {
            let plan = rustwx_models::plot_recipe_store_plan(slug, model).unwrap();
            assert_eq!(
                plan.selectors(),
                [rustwx_core::FieldSelector::surface(field)]
            );
            assert!(rustwx_models::plot_recipe_fetch_plan(slug, model).is_err());
        }
        let default = rw_wrfbatch::wrf_process::WrfProcessOptions::default();
        let selector = serde_json::to_value(rustwx_core::FieldSelector::surface(field)).unwrap();
        let style = rustwx_products::viewer::operational_style_for_store_variable(
            slug,
            &selector,
            "mm",
            rustwx_core::ModelId::WrfGdex,
        )
        .unwrap();
        assert_eq!(style.legend_mode, rustwx_render::LegendMode::Thresholds);
        assert_eq!(
            style.convert,
            rustwx_products::viewer::UnitConvert::MmToInches
        );
        assert!(default.planned_store_fields().contains(&slug.to_string()));
        let profile = rw_wrfbatch::viewer_profile::ViewerProfile::new(&[slug.into()]).unwrap();
        assert!(
            profile
                .options
                .planned_store_fields()
                .contains(&slug.to_string())
        );
    }
}

#[test]
fn multi_record_history_uses_every_interval_and_points_record() {
    let s = Scratch::new();
    let path = frame_series(
        &s.0,
        &[
            (0, 0., 0., 0., 272.16),
            (1, 1., 1., 18., 272.16),
            (2, 2., 1., 30., 269.16),
        ],
    );
    let file = WrfFile::open(&path).unwrap();
    let v = rw_wrfbatch::snowfall::window(&file, &path, 2, None, true).unwrap();
    assert!(v.iter().all(|v| (*v - 34.).abs() < 0.002));
    let (hour, notes) = import(&path, &s.0.join("store"), None);
    let stored = rw_store::reader::HourReader::open(&hour)
        .unwrap()
        .read_full_2d("snow_kuchera_window")
        .unwrap_or_else(|e| panic!("{e}: {notes:?}"));
    assert_eq!(
        hash(&v),
        hash(&stored),
        "incremental import must retain exact interval order"
    );
    let v =
        rw_wrfbatch::snowfall::window(&file, &path, 2, Some("2026-01-01_01_00_00"), true).unwrap();
    assert!(v.iter().all(|v| (*v - 14.).abs() < 0.002));
    let point = Command::new(env!("CARGO_BIN_EXE_rw_points"))
        .args(["--point", "site,47.01,-111.99", "--out"])
        .arg(s.0.join("points.csv"))
        .arg(&path)
        .output()
        .unwrap();
    assert!(
        point.status.success(),
        "{}",
        String::from_utf8_lossy(&point.stderr)
    );
    let text = std::fs::read_to_string(s.0.join("points.csv")).unwrap();
    assert_eq!(text.lines().count(), 4);
    assert!(text.contains("2026-01-01_02:00:00"));
    let bad = Command::new(env!("CARGO_BIN_EXE_rw_points"))
        .args(["--point", "site,47.01,-111.99", "--timeidx", "3", "--out"])
        .arg(s.0.join("bad.csv"))
        .arg(&path)
        .output()
        .unwrap();
    assert!(!bad.status.success());
    assert!(!s.0.join("bad.csv").exists());
}

#[test]
#[ignore = "requires an authorized real history sample on the sprint box"]
fn real_history_matches_source_patch() {
    let path = PathBuf::from(std::env::var("RW_SNOW_REAL_CURRENT").expect("RW_SNOW_REAL_CURRENT"));
    let since = std::env::var("RW_SNOW_REAL_SINCE").expect("RW_SNOW_REAL_SINCE");
    let file = WrfFile::open(&path).unwrap();
    let cells = file.ny * file.nx;
    let actual = rw_wrfbatch::snowfall::window(&file, &path, 0, Some(&since), true).unwrap();
    let name = path.file_name().unwrap().to_str().unwrap();
    let prefix = &name[..11];
    let now = name[11..].replace(':', "_");
    let since_name = since.replace(':', "_");
    let mut frames: Vec<_> = std::fs::read_dir(path.parent().unwrap())
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| {
            let n = p.file_name().unwrap().to_string_lossy();
            n.starts_with(prefix)
                && n.len() == 30
                && n[11..].replace(':', "_") >= since_name
                && n[11..].replace(':', "_") <= now
        })
        .collect();
    frames.sort_by_key(|p| p.file_name().unwrap().to_string_lossy().replace(':', "_"));
    let base = WrfFile::open(&frames[0]).unwrap();
    let snow = compute_var(&base, "SNOWNC", 0, None).unwrap();
    let graupel = compute_var(&base, "GRAUPELNC", 0, None).ok();
    let mut prev: Vec<_> = snow
        .data
        .iter()
        .enumerate()
        .map(|(i, &s)| s as f32 + graupel.as_ref().map(|g| g.data[i] as f32).unwrap_or(0.))
        .collect();
    let mut expected = vec![0.; cells];
    for p in &frames[1..] {
        patch_interval(&WrfFile::open(p).unwrap(), 0, &mut prev, &mut expected);
    }
    let mismatch = actual
        .iter()
        .zip(&expected)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    let max_error = actual
        .iter()
        .zip(&expected)
        .filter(|(a, b)| a.is_finite() && b.is_finite())
        .map(|(a, b)| (a - b).abs())
        .fold(0., f32::max);
    let receipt = serde_json::json!({"schema":"rustwx.snowfall-patch-comparison.v1","source":path,"since":since,
        "cells":cells,"frames":frames.len(),"mismatched_cells":mismatch,"max_abs_error_mm":max_error,
        "actual_sha256":hash(&actual),"source_patch_sha256":hash(&expected)});
    println!("{receipt}");
    if let Ok(p) = std::env::var("RW_SNOW_REAL_RECEIPT") {
        std::fs::write(p, serde_json::to_vec_pretty(&receipt).unwrap()).unwrap();
    }
    assert_eq!(mismatch, 0, "{receipt}");
}
