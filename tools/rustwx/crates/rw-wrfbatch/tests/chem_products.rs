//! Optional products use raw source presence and the first mass-level plane.
use std::path::{Path, PathBuf};
use std::process::Command;
use netcdf_writer::{AttrValue, NcFormat, NcType, NcWriter, Schema, VarData};
use rw_wrfbatch::wrf_process::{CHEM_CORE_FIELD_CATALOG, WrfProcessMessage,
                             WrfProcessOptions, spawn_process_paths};
use wrf_core::WrfFile;

const NX: usize = 8;
const NY: usize = 6;
const NZ: usize = 2;

struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        let nonce = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
            .unwrap().as_nanos();
        let root = std::env::temp_dir().join(format!("chem-products-{}-{nonce}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        Self(root)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); }
}

fn write_fixture(root: &Path, chem: bool, only: Option<&str>) -> PathBuf {
    let path = root.join("wrfout_d01_2026-01-01_00_00_00");
    let mut schema = Schema::new(NcFormat::Offset64);
    // The run origin every wrfout carries (WRF's START_DATE and
    // SIMULATION_START_DATE); the renderer refuses a file without one.
    for name in ["START_DATE", "SIMULATION_START_DATE"] {
        schema.put_global_attr(name, AttrValue::Text("2026-01-01_00:00:00".into())).unwrap();
    }
    let time = schema.def_dim("Time", 0, true).unwrap();
    let strlen = schema.def_dim("DateStrLen", 19, false).unwrap();
    let z = schema.def_dim("bottom_top", NZ, false).unwrap();
    let y = schema.def_dim("south_north", NY, false).unwrap();
    let x = schema.def_dim("west_east", NX, false).unwrap();
    let times = schema.def_var("Times", NcType::Char, &[time, strlen]).unwrap();
    let theta = schema.def_var("T", NcType::Float, &[time, z, y, x]).unwrap();
    let lat = schema.def_var("XLAT", NcType::Float, &[time, y, x]).unwrap();
    let lon = schema.def_var("XLONG", NcType::Float, &[time, y, x]).unwrap();
    let t2 = schema.def_var("T2", NcType::Float, &[time, y, x]).unwrap();
    let mut rows = Vec::new();
    for (index, row) in CHEM_CORE_FIELD_CATALOG.iter().enumerate() {
        if !chem || only.is_some_and(|name| name != row.source) { continue; }
        let dims = if row.lowest_level { vec![time, z, y, x] } else { vec![time, y, x] };
        let var = schema.def_var(row.source, NcType::Float, &dims).unwrap();
        let mut values = vec![index as f32 + 0.125; NX * NY];
        if row.lowest_level { values.extend(vec![999.0; NX * NY]); }
        rows.push((var, values));
    }
    let mut writer = NcWriter::create(&path, schema).unwrap();
    writer.write_record(0, times, VarData::Char(b"2026-01-01_00:00:00")).unwrap();
    writer.write_record(0, theta, VarData::F32(&vec![0.0; NZ * NY * NX])).unwrap();
    writer.write_record(0, t2, VarData::F32(&vec![290.0; NY * NX])).unwrap();
    let lats: Vec<f32> = (0..NY).flat_map(|j| std::iter::repeat_n(35.0 + j as f32 * 0.1, NX)).collect();
    let lons: Vec<f32> = (0..NY).flat_map(|_| (0..NX).map(|i| -97.0 + i as f32 * 0.1)).collect();
    writer.write_record(0, lat, VarData::F32(&lats)).unwrap();
    writer.write_record(0, lon, VarData::F32(&lons)).unwrap();
    for (var, values) in rows { writer.write_record(0, var, VarData::F32(&values)).unwrap(); }
    writer.finish().unwrap();
    path
}

#[test]
fn chem_off_wrfout_has_the_base_store_and_selector_plan() {
    let scratch = Scratch::new();
    let path = write_fixture(&scratch.0, false, None);
    let file = WrfFile::open(path).unwrap();
    let options = WrfProcessOptions::default();
    // The no-file methods are the base's unconditional meteorological plan.
    assert_eq!(options.planned_store_fields_for(&file), options.planned_store_fields());
    assert_eq!(options.planned_store_selectors_for(&file), options.planned_store_selectors());
    for row in CHEM_CORE_FIELD_CATALOG {
        assert!(!options.planned_store_fields_for(&file).contains(&row.store_name.to_owned()));
        assert!(!options.planned_store_selectors_for(&file).contains(&row.selector));
    }
}

#[test]
fn chem_presence_is_checked_per_source() {
    for row in CHEM_CORE_FIELD_CATALOG {
        let scratch = Scratch::new();
        let path = write_fixture(&scratch.0, true, Some(row.source));
        let file = WrfFile::open(path).unwrap();
        let options = WrfProcessOptions::default();
        let added: Vec<_> = options.planned_store_fields_for(&file).into_iter()
            .filter(|name| !options.planned_store_fields().contains(name)).collect();
        assert_eq!(added, vec![row.store_name.to_string()]);
    }
}

#[test]
fn chem_level_one_and_ppb_scaling_are_bit_exact() {
    let scratch = Scratch::new();
    let path = write_fixture(&scratch.0, true, None);
    let store = scratch.0.join("store");
    let options = WrfProcessOptions {
        diagnostics: false, raw_extras: false, stored_planes: false, viewer_2d: true,
        only: CHEM_CORE_FIELD_CATALOG.iter().map(|row| row.store_name.to_string()).collect(),
        ..WrfProcessOptions::default()
    };
    let task = spawn_process_paths(vec![path], store.clone(), options);
    let summary = loop {
        match task.rx.recv().unwrap() {
            WrfProcessMessage::Progress(_) => (),
            WrfProcessMessage::Done(result) => break result.unwrap(),
        }
    };
    let hour = rw_store::reader::HourReader::open(
        &store.join(summary.model).join(summary.run).join("f000.rws")).unwrap();
    for (index, row) in CHEM_CORE_FIELD_CATALOG.iter().enumerate() {
        let values = hour.read_full_2d(row.store_name).unwrap();
        let expected = [0.125_f32 * 1.0e-9, 1.125_f32 * 1.0e-6,
                        2.125, 3.125, 4.125, 5.125_f32 * 1000.0][index];
        let max_ulp = values.iter().map(|v| v.to_bits().abs_diff(expected.to_bits())).max().unwrap();
        println!("ULP synthetic-lowest-level {} cells={} max={max_ulp}", row.store_name, values.len());
        assert_eq!(values.len(), NX * NY);
        assert_eq!(max_ulp, 0, "{}", row.store_name);
    }
}

#[test]
fn synthetic_chem_wrfout_renders_all_six_products() {
    let scratch = Scratch::new();
    let path = write_fixture(&scratch.0, true, None);
    let products = CHEM_CORE_FIELD_CATALOG.iter().map(|row| row.store_name).collect::<Vec<_>>().join(",");
    let output = Command::new(env!("CARGO_BIN_EXE_rw_wrfbatch"))
        .args(["--products", &products, "--width", "480", "--height", "360"])
        .arg("--store-root").arg(scratch.0.join("store"))
        .arg("--out-dir").arg(scratch.0.join("png"))
        .arg(path).env("CUDA_VISIBLE_DEVICES", "").env("RAYON_NUM_THREADS", "1")
        .env("RUSTWX_BATCH_RENDER_THREADS", "1").output().unwrap();
    assert!(output.status.success(), "{}\n{}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
    let stdout = String::from_utf8(output.stdout).unwrap();
    for row in CHEM_CORE_FIELD_CATALOG {
        assert!(stdout.lines().any(|line| line.starts_with(&format!("RENDERED {} ", row.store_name))), "{stdout}");
    }
}


/// A wrfout carrying only the named 2-D sources at the given values.
fn write_sources(root: &Path, sources: &[(&str, f32)]) -> PathBuf {
    let path = root.join("wrfout_d01_2026-01-01_00_00_00");
    let mut schema = Schema::new(NcFormat::Offset64);
    for name in ["START_DATE", "SIMULATION_START_DATE"] {
        schema.put_global_attr(name, AttrValue::Text("2026-01-01_00:00:00".into())).unwrap();
    }
    let time = schema.def_dim("Time", 0, true).unwrap();
    let strlen = schema.def_dim("DateStrLen", 19, false).unwrap();
    let z = schema.def_dim("bottom_top", NZ, false).unwrap();
    let y = schema.def_dim("south_north", NY, false).unwrap();
    let x = schema.def_dim("west_east", NX, false).unwrap();
    let times = schema.def_var("Times", NcType::Char, &[time, strlen]).unwrap();
    let theta = schema.def_var("T", NcType::Float, &[time, z, y, x]).unwrap();
    let lat = schema.def_var("XLAT", NcType::Float, &[time, y, x]).unwrap();
    let lon = schema.def_var("XLONG", NcType::Float, &[time, y, x]).unwrap();
    let t2 = schema.def_var("T2", NcType::Float, &[time, y, x]).unwrap();
    let vars: Vec<_> = sources.iter()
        .map(|(name, value)| (schema.def_var(name, NcType::Float, &[time, y, x]).unwrap(), *value))
        .collect();
    let mut writer = NcWriter::create(&path, schema).unwrap();
    writer.write_record(0, times, VarData::Char(b"2026-01-01_00:00:00")).unwrap();
    writer.write_record(0, theta, VarData::F32(&vec![0.0; NZ * NY * NX])).unwrap();
    writer.write_record(0, t2, VarData::F32(&vec![290.0; NY * NX])).unwrap();
    let lats: Vec<f32> = (0..NY).flat_map(|j| std::iter::repeat_n(35.0 + j as f32 * 0.1, NX)).collect();
    let lons: Vec<f32> = (0..NY).flat_map(|_| (0..NX).map(|i| -97.0 + i as f32 * 0.1)).collect();
    writer.write_record(0, lat, VarData::F32(&lats)).unwrap();
    writer.write_record(0, lon, VarData::F32(&lons)).unwrap();
    for (var, value) in vars { writer.write_record(0, var, VarData::F32(&vec![value; NX * NY])).unwrap(); }
    writer.finish().unwrap();
    path
}

fn stored(scratch: &Scratch, path: PathBuf, names: &[&str]) -> Vec<Vec<f32>> {
    let store = scratch.0.join("store");
    let options = WrfProcessOptions {
        diagnostics: false, raw_extras: false, stored_planes: false, viewer_2d: true,
        only: names.iter().map(|name| name.to_string()).collect(),
        ..WrfProcessOptions::default()
    };
    let task = spawn_process_paths(vec![path], store.clone(), options);
    let summary = loop {
        match task.rx.recv().unwrap() {
            WrfProcessMessage::Progress(_) => (),
            WrfProcessMessage::Done(result) => break result.unwrap(),
        }
    };
    let hour = rw_store::reader::HourReader::open(
        &store.join(summary.model).join(summary.run).join("f000.rws")).unwrap();
    names.iter().map(|name| hour.read_full_2d(name).unwrap()).collect()
}

#[test]
fn the_coupled_fire_smoke_draws_on_the_smoke_maps_and_adds_to_aq_smoke() {
    // A fire-only run: SFIRE's bulk smoke alone reaches both smoke products.
    let scratch = Scratch::new();
    let path = write_sources(&scratch.0, &[("SFIRE_SMOKE_SFC", 40.0), ("SFIRE_SMOKE_COLUMN", 3.0)]);
    let planes = stored(&scratch, path, &["smoke_near_surface", "smoke_column"]);
    assert!(planes[0].iter().all(|v| *v == 40.0_f32 * 1.0e-9));
    assert!(planes[1].iter().all(|v| *v == 3.0_f32 * 1.0e-6));
    // A run carrying both: the map is their total, in the same units.
    let scratch = Scratch::new();
    let path = write_sources(&scratch.0, &[("SMOKE_SFC", 10.0), ("SFIRE_SMOKE_SFC", 40.0)]);
    let planes = stored(&scratch, path, &["smoke_near_surface"]);
    assert!(planes[0].iter().all(|v| *v == 10.0_f32 * 1.0e-9 + 40.0_f32 * 1.0e-9));
}
