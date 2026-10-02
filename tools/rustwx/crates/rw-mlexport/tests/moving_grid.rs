//! Moving native grids keep geography and terrain for every frame.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use netcdf_writer::{AttrValue, NcFormat, NcType, NcWriter, Schema, VarData};
use rw_mlexport::blosc;
use rw_mlexport::export::execute;
use rw_mlexport::request::{GridKind, GridRequest, Layout, LevelKind, LevelSpec, Mode, Provenance, RegridMethod, Request, VariableRow};
use serde_json::Value;
use static_fields::projection::{GridSpec, ProjectedGrid, ProjectionKind};

const NX: usize = 4;
const NY: usize = 3;

fn label(hour: usize) -> String { format!("2026-09-29_{hour:02}:00:00") }

/// The frame's file name.  WRF spells history files `..._HH:MM:SS` and the
/// reader takes that and `..._HH_MM_SS`; a colon is not a legal Windows
/// file-name character (os error 123), so Windows writes the underscore
/// spelling and every other platform keeps the colon form under test.
fn history_name(hour: usize) -> String {
    let name = format!("wrfout_d01_{}", label(hour));
    if cfg!(windows) { name.replace(':', "_") } else { name }
}

fn write_frame(path: &Path, hour: usize, projected: bool, moving: bool) {
    let shift = if moving { hour as f64 } else { 0.0 };
    write_frame_shift(path, hour, projected, shift);
}

fn write_frame_shift(path: &Path, hour: usize, projected: bool, shift: f64) {
    let mut schema = Schema::new(NcFormat::Offset64);
    let time = schema.def_dim("Time", 0, true).unwrap();
    let strlen = schema.def_dim("DateStrLen", 19, false).unwrap();
    let we = schema.def_dim("west_east", NX, false).unwrap();
    let sn = schema.def_dim("south_north", NY, false).unwrap();
    schema.def_dim("bottom_top", 2, false).unwrap();
    schema.def_dim("west_east_stag", NX + 1, false).unwrap();
    schema.def_dim("south_north_stag", NY + 1, false).unwrap();
    let bt_s = schema.def_dim("bottom_top_stag", 3, false).unwrap();
    for (name, value) in [
        ("MAP_PROJ", AttrValue::Ints(vec![if projected { 1 } else { 6 }])),
        ("GRID_ID", AttrValue::Ints(vec![1])),
        ("DX", AttrValue::Floats(vec![12_000.0])),
        ("DY", AttrValue::Floats(vec![12_000.0])),
        ("TRUELAT1", AttrValue::Floats(vec![30.0])),
        ("TRUELAT2", AttrValue::Floats(vec![60.0])),
        ("STAND_LON", AttrValue::Floats(vec![-98.0])),
        ("CEN_LAT", AttrValue::Floats(vec![38.0])),
        ("MOAD_CEN_LAT", AttrValue::Floats(vec![38.0])),
        ("POLE_LAT", AttrValue::Floats(vec![90.0])),
        ("POLE_LON", AttrValue::Floats(vec![0.0])),
        ("SIMULATION_START_DATE", AttrValue::Text(label(0))),
    ] { schema.put_global_attr(name, value).unwrap(); }
    let times = schema.def_var("Times", NcType::Char, &[time, strlen]).unwrap();
    let mut ids = BTreeMap::new();
    for name in ["XLAT", "XLONG", "T2", "LANDMASK", "RAINNC"] {
        ids.insert(name, schema.def_var(name, NcType::Float, &[time, sn, we]).unwrap());
    }
    for name in ["PH", "PHB"] {
        ids.insert(name, schema.def_var(name, NcType::Float, &[time, bt_s, sn, we]).unwrap());
    }
    let grid = ProjectedGrid::new(GridSpec {
        kind: ProjectionKind::Lambert, ref_lat: 38.0, ref_lon: -98.0,
        truelat1: 30.0, truelat2: 60.0, stand_lon: -98.0, dx: 12_000.0, dy: 12_000.0,
        e_we: NX as i64 + 1, e_sn: NY as i64 + 1,
        known_x: (NX as f64 + 1.0) / 2.0 - shift,
        known_y: (NY as f64 + 1.0) / 2.0,
        moad_cen_lat: 38.0, moad_cen_lon: -98.0,
        lat_deg: vec![], lon0_deg: 0.0, dlon_deg: 0.0,
    }).unwrap();
    let mut latitude = Vec::new();
    let mut longitude = Vec::new();
    for j in 0..NY { for i in 0..NX {
        let (lat, lon) = if projected {
            grid.ij_to_latlon(i as f64 + 1.0, j as f64 + 1.0)
        } else {
            (-45.0 + 45.0 * j as f64 + shift, -180.0 + 90.0 * i as f64)
        };
        latitude.push(lat as f32); longitude.push(lon as f32);
    }}
    let terrain: Vec<f32> = (0..3 * NX * NY).map(|k| {
        (100.0 * shift + (k % (NX * NY)) as f64 + (k / (NX * NY)) as f64 * 1000.0) as f32
    }).collect();
    let landmask: Vec<f32> = (0..NX * NY).map(|c| ((c + shift as usize) % 2) as f32).collect();
    let rain: Vec<f32> = (0..NX * NY).map(|c| (10.0 * ((c % NX) as f64 + shift)) as f32).collect();
    let mut writer = NcWriter::create(path, schema).unwrap();
    writer.write_record(0, times, VarData::Char(label(hour).as_bytes())).unwrap();
    for (name, values) in [
        ("XLAT", latitude), ("XLONG", longitude), ("T2", vec![280.0 + hour as f32; NX * NY]),
        ("LANDMASK", landmask), ("RAINNC", rain), ("PH", vec![0.0; 3 * NX * NY]), ("PHB", terrain),
    ] { writer.write_record(0, ids[name], VarData::F32(&values)).unwrap(); }
    writer.finish().unwrap();
}

fn request(inputs: Vec<PathBuf>, out: &Path, layout: Layout) -> Request {
    let table = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../../gpuwm/data/ml_export/variables.json");
    let document: Value = serde_json::from_str(&std::fs::read_to_string(table).unwrap()).unwrap();
    let rows: Vec<VariableRow> = serde_json::from_value(document["rows"].clone()).unwrap();
    Request {
        schema: "ml-export.request/v1".into(), mode: Mode::Run, inputs, out: out.to_path_buf(),
        overwrite: false, zip: false,
        variables: rows.into_iter().filter(|r| ["2m_temperature", "geopotential_at_surface", "land_sea_mask"].contains(&r.id.as_str())).collect(),
        levels: LevelSpec { set: "model".into(), kind: LevelKind::Model, hpa: vec![], model_levels: vec![] },
        grid: GridRequest { kind: GridKind::Native, deg: None, method: RegridMethod::Bilinear },
        names: "wb2".into(), layout, domains: None, every_hours: None, start: None, end: None,
        skip_unavailable: false, threads: None, spacings_deg: vec![0.1],
        provenance: Provenance { engine: "engine".into(), exporter_version: "0-test".into(), created_utc: "2026-10-01T00:00:00Z".into(), ..Default::default() },
    }
}

fn json(path: &Path) -> Value { serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap() }
fn f32_values(path: &Path) -> Vec<f32> {
    blosc::decompress(&std::fs::read(path).unwrap()).unwrap().chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect()
}
fn f64_values(path: &Path) -> Vec<f64> {
    blosc::decompress(&std::fs::read(path).unwrap()).unwrap().chunks_exact(8)
        .map(|b| f64::from_le_bytes(b.try_into().unwrap())).collect()
}

fn exercise(projected: bool, moving: bool, layout: Layout, append: bool) {
    let scratch = std::env::temp_dir().join(format!("mlx-moving-{}-{projected}-{moving}-{layout:?}-{append}", std::process::id()));
    std::fs::create_dir_all(&scratch).unwrap();
    let mut inputs = Vec::new();
    for hour in 0..2 {
        let path = scratch.join(history_name(hour));
        write_frame(&path, hour, projected, moving); inputs.push(path);
    }
    let out = scratch.join("export");
    let mut sink = |_: Value| {};
    if append {
        for input in &inputs {
            let mut req = request(vec![input.clone()], &out, layout); req.mode = Mode::Append;
            execute(req, &mut sink).unwrap();
        }
        let mut req = request(vec![], &out, layout); req.mode = Mode::Finalize;
        execute(req, &mut sink).unwrap();
    } else { execute(request(inputs, &out, layout), &mut sink).unwrap(); }
    let store = out.join("d01.zarr");
    if moving {
        let expected = if layout == Layout::Analysis { serde_json::json!(["time", "y", "x"]) }
            else { serde_json::json!(["time", "prediction_timedelta", "y", "x"]) };
        for name in ["latitude", "longitude", "geopotential_at_surface", "land_sea_mask", "2m_temperature"] {
            assert_eq!(json(&store.join(name).join(".zattrs"))["_ARRAY_DIMENSIONS"], expected, "{name}");
        }
        let key0 = if layout == Layout::Analysis { "0.0.0" } else { "0.0.0.0" };
        let key1 = if layout == Layout::Analysis { "1.0.0" } else { "0.1.0.0" };
        let old_phi = f32_values(&store.join("geopotential_at_surface").join(key0));
        let new_phi = f32_values(&store.join("geopotential_at_surface").join(key1));
        assert!(old_phi.iter().zip(&new_phi).all(|(a,b)| (*b - *a - 100.0).abs() < 1e-6));
        let old_land = f32_values(&store.join("land_sea_mask").join(key0));
        let new_land = f32_values(&store.join("land_sea_mask").join(key1));
        assert!(old_land.iter().zip(&new_land).all(|(a,b)| *a + *b == 1.0));
        let old_lat = f32_values(&store.join("latitude").join(key0));
        let new_lat = f32_values(&store.join("latitude").join(key1));
        assert_ne!(old_lat, new_lat);
        if projected {
            let pkey0 = if layout == Layout::Analysis { "0.0" } else { "0.0.0" };
            let pkey1 = if layout == Layout::Analysis { "1.0" } else { "0.1.0" };
            let x0 = f64_values(&store.join("projection_x_coordinate").join(pkey0));
            let x1 = f64_values(&store.join("projection_x_coordinate").join(pkey1));
            assert!(x0.iter().zip(&x1).all(|(a,b)| (*b - *a - 12_000.0).abs() < 1.0));
            assert!(json(&store.join("2m_temperature/.zattrs"))["coordinates"].as_str().unwrap().contains("projection_x_coordinate"));
            assert!(!store.join("x").exists());
        } else {
            assert_eq!(old_phi[0], 2.0, "coordinates and fields use the same longitude reorder");
            assert!(old_lat.iter().zip(&new_lat).all(|(a,b)| *b - *a == 1.0));
        }
    } else {
        for name in ["geopotential_at_surface", "land_sea_mask"] {
            assert_eq!(json(&store.join(name).join(".zattrs"))["_ARRAY_DIMENSIONS"], serde_json::json!(["y", "x"]));
            assert!(store.join(name).join("0.0").is_file());
            assert_eq!(std::fs::read_dir(store.join(name)).unwrap().count(), 3);
        }
    }
    std::fs::remove_dir_all(scratch).unwrap();
}

#[test]
fn moving_projected_analysis_preserves_terrain_and_absolute_axes() {
    exercise(true, true, Layout::Analysis, false);
}

#[test]
fn moving_projected_forecast_append_preserves_terrain_and_absolute_axes() {
    exercise(true, true, Layout::Forecast, true);
}

#[test]
fn moving_regular_grid_carries_each_times_reordered_coordinates() {
    exercise(false, true, Layout::Analysis, false);
}

#[test]
fn stationary_analysis_keeps_static_two_dimensional_arrays() {
    exercise(true, false, Layout::Analysis, false);
}

#[test]
fn stationary_forecast_append_keeps_static_two_dimensional_arrays() {
    exercise(true, false, Layout::Forecast, true);
}

#[test]
fn moving_precipitation_differences_the_same_ground_for_interval_and_six_hours() {
    let scratch = std::env::temp_dir().join(format!("mlx-moving-rain-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).unwrap();
    let mut inputs = Vec::new();
    for (hour, shift) in [(0, 0.0), (1, 1.0), (6, 2.0)] {
        let path = scratch.join(history_name(hour));
        write_frame_shift(&path, hour, true, shift);
        inputs.push(path);
    }
    let out = scratch.join("export");
    let mut req = request(inputs, &out, Layout::Analysis);
    let table = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../../gpuwm/data/ml_export/variables.json");
    let document: Value = serde_json::from_str(&std::fs::read_to_string(table).unwrap()).unwrap();
    let rows: Vec<VariableRow> = serde_json::from_value(document["rows"].clone()).unwrap();
    req.variables.extend(rows.into_iter().filter(|r| ["total_precipitation", "total_precipitation_6hr"].contains(&r.id.as_str())));
    let mut sink = |_: Value| {};
    execute(req, &mut sink).unwrap();
    let store = out.join("d01.zarr");
    for (name, overlap) in [("total_precipitation", NX - 1), ("total_precipitation_6hr", NX - 2)] {
        let values = f32_values(&store.join(name).join("2.0.0"));
        for j in 0..NY { for i in 0..NX {
            let value = values[j * NX + i];
            if i < overlap { assert_eq!(value, 0.0, "no rain fell at the overlapping ground"); }
            else { assert!(value.is_nan(), "new ground has no earlier observation"); }
        }}
        assert!(json(&store.join(name).join(".zattrs"))["moving_grid_accumulation"].as_str().unwrap().contains("newly exposed ground"));
    }
    std::fs::remove_dir_all(scratch).unwrap();
}
