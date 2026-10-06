//! End-to-end: synthetic wrfout files with analytic columns, written by the
//! workspace's classic NetCDF writer, exported, and read back chunk by
//! chunk.
//!
//! The columns are built so the right answer is known without the
//! exporter: temperature, geopotential and both wind components are linear
//! in ln(p) and the mass levels sit at the geometric mean of their faces, so
//! log-pressure interpolation reproduces the analytic value exactly (up to
//! the file's float32 storage).  Every check below compares against that
//! formula, not against the exporter's own arithmetic.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use netcdf_writer::{AttrValue, NcFormat, NcType, NcWriter, Schema, VarData};
use rw_isobaric::{ecmwf_temperature, ColumnBase, ECMWF_RULE};
use rw_mlexport::blosc;
use rw_mlexport::export::{execute, Outcome};
use rw_mlexport::request::{GridKind, GridRequest, Layout, LevelKind, LevelSpec, Mode, Provenance, RegridMethod, Request, VariableRow};
use serde_json::Value;
use static_fields::projection::{GridSpec, ProjectedGrid, ProjectionKind};

const P_TOP: f64 = 5000.0;
const KAPPA: f64 = 0.2857142857;

struct Domain {
    id: u32,
    nx: usize,
    ny: usize,
    nz: usize,
    dx: f64,
}

fn temperature(p: f64) -> f64 {
    280.0 + 40.0 * (p / 1e5).ln()
}
fn u_grid(p: f64) -> f64 {
    10.0 + 3.0 * (p / 1e5).ln()
}
/// Constant, so unstaggering V across rows of different surface pressure
/// is exact.
fn v_grid(_p: f64) -> f64 {
    -5.0
}
fn qv(p: f64) -> f64 {
    0.006 * (1.0 + (p / 1e5).ln() / 4.0)
}
fn surface_pressure(j: usize) -> f64 {
    101_000.0 - 900.0 * j as f64
}
fn terrain(i: usize) -> f64 {
    50.0 * i as f64
}
fn angle(i: usize) -> f64 {
    0.02 * (i as f64 - 10.0)
}
const PHI_SCALE: f64 = 287.0 * 260.0;
fn phi(ps: f64, hgt: f64, p: f64) -> f64 {
    9.81 * hgt + PHI_SCALE * (ps / p).ln()
}
fn face_pressure(ps: f64, k: usize, nz: usize) -> f64 {
    let eta = 1.0 - k as f64 / nz as f64;
    P_TOP + eta * (ps - P_TOP)
}
fn mass_pressure(ps: f64, k: usize, nz: usize) -> f64 {
    (face_pressure(ps, k, nz) * face_pressure(ps, k + 1, nz)).sqrt()
}
/// A WRF eta core's mass level: the middle of its layer in eta, so the
/// arithmetic mean of its faces' pressures.
fn eta_mass_pressure(ps: f64, k: usize, nz: usize) -> f64 {
    0.5 * (face_pressure(ps, k, nz) + face_pressure(ps, k + 1, nz))
}

fn lambert(d: &Domain, shift_cells: f64) -> ProjectedGrid {
    ProjectedGrid::new(GridSpec {
        kind: ProjectionKind::Lambert,
        ref_lat: 38.0,
        ref_lon: -98.0,
        truelat1: 30.0,
        truelat2: 60.0,
        stand_lon: -98.0,
        dx: d.dx,
        dy: d.dx,
        e_we: d.nx as i64 + 1,
        e_sn: d.ny as i64 + 1,
        known_x: (d.nx as f64 + 1.0) / 2.0 - shift_cells,
        known_y: (d.ny as f64 + 1.0) / 2.0,
        moad_cen_lat: 38.0,
        moad_cen_lon: -98.0,
        lat_deg: vec![],
        lon0_deg: 0.0,
        dlon_deg: 0.0,
    })
    .unwrap()
}

fn label(hour: i64) -> String {
    format!("2026-09-29_{:02}:00:00", hour)
}

/// Write one wrfout holding `hours` (one record each).
fn write_wrfout(path: &Path, d: &Domain, hours: &[i64], with_qv: bool) {
    write_wrfout_ladder(path, d, hours, with_qv, false)
}

fn write_wrfout_ladder(path: &Path, d: &Domain, hours: &[i64], with_qv: bool, eta: bool) {
    write_wrfout_full(path, d, hours, with_qv, eta, 0.0, &|_, _| 290.0)
}

/// `eta`: mass levels at the arithmetic mean of their faces and ZNW
/// written, as a WRF eta core's history file is.
/// `shift_cells`: the grid moved east by this many cells, as a moving
/// nest's is between two frames.
/// `t2`: the 2 m temperature as a function of (latitude, longitude) in
/// degrees, longitude in -180..180.
fn write_wrfout_full(
    path: &Path,
    d: &Domain,
    hours: &[i64],
    with_qv: bool,
    eta: bool,
    shift_cells: f64,
    t2: &dyn Fn(f64, f64) -> f64,
) {
    write_wrfout_metadata(path, d, hours, with_qv, eta, shift_cells, t2, "GPUWM", false)
}

fn write_wrfout_metadata(
    path: &Path, d: &Domain, hours: &[i64], with_qv: bool, eta: bool,
    shift_cells: f64, t2: &dyn Fn(f64, f64) -> f64, prefix: &str, missing: bool,
) {
    let mp = move |ps: f64, k: usize, nz: usize| if eta { eta_mass_pressure(ps, k, nz) } else { mass_pressure(ps, k, nz) };
    let (nx, ny, nz) = (d.nx, d.ny, d.nz);
    let cells = nx * ny;
    let mut s = Schema::new(NcFormat::Offset64);
    let time = s.def_dim("Time", 0, true).unwrap();
    let strlen = s.def_dim("DateStrLen", 19, false).unwrap();
    let we = s.def_dim("west_east", nx, false).unwrap();
    let sn = s.def_dim("south_north", ny, false).unwrap();
    let bt = s.def_dim("bottom_top", nz, false).unwrap();
    let we_s = s.def_dim("west_east_stag", nx + 1, false).unwrap();
    let sn_s = s.def_dim("south_north_stag", ny + 1, false).unwrap();
    let bt_s = s.def_dim("bottom_top_stag", nz + 1, false).unwrap();
    let version_attr = format!("{prefix}_VERSION");
    let source_attr = format!("{prefix}_INITIAL_CONDITION_SOURCE");
    let cycle_attr = format!("{prefix}_INITIAL_CONDITION_CYCLE");
    for (name, value) in [
        ("MAP_PROJ", AttrValue::Ints(vec![1])),
        ("GRID_ID", AttrValue::Ints(vec![d.id as i32])),
        ("PARENT_ID", AttrValue::Ints(vec![if d.id == 1 { 0 } else { 1 }])),
        ("DX", AttrValue::Floats(vec![d.dx as f32])),
        ("DY", AttrValue::Floats(vec![d.dx as f32])),
        ("TRUELAT1", AttrValue::Floats(vec![30.0])),
        ("TRUELAT2", AttrValue::Floats(vec![60.0])),
        ("STAND_LON", AttrValue::Floats(vec![-98.0])),
        ("CEN_LAT", AttrValue::Floats(vec![38.0])),
        ("CEN_LON", AttrValue::Floats(vec![-98.0])),
        ("MOAD_CEN_LAT", AttrValue::Floats(vec![38.0])),
        ("POLE_LAT", AttrValue::Floats(vec![90.0])),
        ("POLE_LON", AttrValue::Floats(vec![0.0])),
        ("SIMULATION_START_DATE", AttrValue::Text(label(0))),
        (version_attr.as_str(), AttrValue::Text("9.9.9-test".into())),
        ("TITLE", AttrValue::Text(format!("{prefix} model history"))),
        (source_attr.as_str(), AttrValue::Text("ERA5".into())),
        (cycle_attr.as_str(), AttrValue::Text("2026-09-29T00:00:00Z".into())),
    ] {
        s.put_global_attr(name, value).unwrap();
    }
    let mut ids: BTreeMap<&str, usize> = BTreeMap::new();
    ids.insert("Times", s.def_var("Times", NcType::Char, &[time, strlen]).unwrap());
    for name in ["XLAT", "XLONG", "PSFC", "T2", "U10", "V10", "HGT", "LANDMASK", "SINALPHA", "COSALPHA", "RAINNC", "RAINC"] {
        ids.insert(name, s.def_var(name, NcType::Float, &[time, sn, we]).unwrap());
    }
    let mut mass = vec!["P", "PB", "T"];
    if with_qv {
        mass.push("QVAPOR");
    }
    for name in mass {
        ids.insert(name, s.def_var(name, NcType::Float, &[time, bt, sn, we]).unwrap());
    }
    ids.insert("U", s.def_var("U", NcType::Float, &[time, bt, sn, we_s]).unwrap());
    ids.insert("V", s.def_var("V", NcType::Float, &[time, bt, sn_s, we]).unwrap());
    for name in ["W", "PH", "PHB"] {
        ids.insert(name, s.def_var(name, NcType::Float, &[time, bt_s, sn, we]).unwrap());
    }
    ids.insert("P_TOP", s.def_var("P_TOP", NcType::Float, &[time]).unwrap());
    ids.insert("ZNU", s.def_var("ZNU", NcType::Float, &[time, bt]).unwrap());
    if eta {
        ids.insert("ZNW", s.def_var("ZNW", NcType::Float, &[time, bt_s]).unwrap());
    }
    let mut w = NcWriter::create(path, s).unwrap();
    let grid = lambert(d, shift_cells);
    let mut xlat = vec![0f32; cells];
    let mut xlong = vec![0f32; cells];
    for j in 0..ny {
        for i in 0..nx {
            let (lat, lon) = grid.ij_to_latlon(i as f64 + 1.0, j as f64 + 1.0);
            xlat[j * nx + i] = lat as f32;
            xlong[j * nx + i] = lon as f32;
        }
    }
    let plane = |f: &dyn Fn(usize, usize) -> f64| -> Vec<f32> {
        (0..cells).map(|c| f(c % nx, c / nx) as f32).collect()
    };
    for (rec, &hour) in hours.iter().enumerate() {
        let rec = rec as u64;
        w.write_record(rec, ids["Times"], VarData::Char(label(hour).as_bytes())).unwrap();
        w.write_record(rec, ids["XLAT"], VarData::F32(&xlat)).unwrap();
        w.write_record(rec, ids["XLONG"], VarData::F32(&xlong)).unwrap();
        w.write_record(rec, ids["PSFC"], VarData::F32(&plane(&|_, j| surface_pressure(j)))).unwrap();
        let t2_plane: Vec<f32> = (0..cells).map(|c| t2(xlat[c] as f64, xlong[c] as f64) as f32).collect();
        w.write_record(rec, ids["T2"], VarData::F32(&t2_plane)).unwrap();
        w.write_record(rec, ids["U10"], VarData::F32(&plane(&|_, _| 5.0))).unwrap();
        w.write_record(rec, ids["V10"], VarData::F32(&plane(&|_, _| 2.0))).unwrap();
        w.write_record(rec, ids["HGT"], VarData::F32(&plane(&|i, _| terrain(i)))).unwrap();
        w.write_record(rec, ids["LANDMASK"], VarData::F32(&plane(&|i, _| if i > 12 { 1.0 } else { 0.0 }))).unwrap();
        w.write_record(rec, ids["SINALPHA"], VarData::F32(&plane(&|i, _| angle(i).sin()))).unwrap();
        w.write_record(rec, ids["COSALPHA"], VarData::F32(&plane(&|i, _| angle(i).cos()))).unwrap();
        w.write_record(rec, ids["RAINNC"], VarData::F32(&plane(&|i, j| if missing && i == 0 && j == 0 { f64::NAN } else { 2.0 * hour as f64 }))).unwrap();
        w.write_record(rec, ids["RAINC"], VarData::F32(&plane(&|_, _| 0.5 * hour as f64))).unwrap();
        let volume = |levels: usize, f: &dyn Fn(usize, usize, usize) -> f64| -> Vec<f32> {
            let mut v = Vec::with_capacity(levels * cells);
            for k in 0..levels {
                for c in 0..cells {
                    v.push(f(k, c % nx, c / nx) as f32);
                }
            }
            v
        };
        w.write_record(rec, ids["P"], VarData::F32(&volume(nz, &|_, _, _| 0.0))).unwrap();
        w.write_record(rec, ids["PB"], VarData::F32(&volume(nz, &|k, _, j| mp(surface_pressure(j), k, nz)))).unwrap();
        w.write_record(rec, ids["T"], VarData::F32(&volume(nz, &|k, _, j| {
            let p = mp(surface_pressure(j), k, nz);
            temperature(p) / (p / 1e5).powf(KAPPA) - 300.0
        }))).unwrap();
        if with_qv {
            w.write_record(rec, ids["QVAPOR"], VarData::F32(&volume(nz, &|k, i, j| if missing && i == 0 && j == 0 { f64::NAN } else { qv(mp(surface_pressure(j), k, nz)) }))).unwrap();
        }
        // U, V constant along the staggered direction: unstaggering is exact.
        let u: Vec<f32> = (0..nz)
            .flat_map(|k| (0..ny).flat_map(move |j| (0..=nx).map(move |_| u_grid(mp(surface_pressure(j), k, nz)) as f32)))
            .collect();
        w.write_record(rec, ids["U"], VarData::F32(&u)).unwrap();
        let v: Vec<f32> = (0..nz)
            .flat_map(|k| {
                (0..=ny).flat_map(move |j| {
                    let jj = j.min(ny - 1);
                    (0..nx).map(move |_| v_grid(mp(surface_pressure(jj), k, nz)) as f32)
                })
            })
            .collect();
        w.write_record(rec, ids["V"], VarData::F32(&v)).unwrap();
        w.write_record(rec, ids["W"], VarData::F32(&volume(nz + 1, &|_, _, _| 0.0))).unwrap();
        w.write_record(rec, ids["PH"], VarData::F32(&volume(nz + 1, &|_, _, _| 0.0))).unwrap();
        w.write_record(rec, ids["PHB"], VarData::F32(&volume(nz + 1, &|k, i, j| {
            let ps = surface_pressure(j);
            phi(ps, terrain(i), face_pressure(ps, k, nz))
        }))).unwrap();
        w.write_record(rec, ids["P_TOP"], VarData::F32(&[P_TOP as f32])).unwrap();
        let znu: Vec<f32> = (0..nz).map(|k| (1.0 - (k as f64 + 0.5) / nz as f64) as f32).collect();
        w.write_record(rec, ids["ZNU"], VarData::F32(&znu)).unwrap();
        if eta {
            let znw: Vec<f32> = (0..=nz).map(|k| (1.0 - k as f64 / nz as f64) as f32).collect();
            w.write_record(rec, ids["ZNW"], VarData::F32(&znw)).unwrap();
        }
    }
    w.finish().unwrap();
}

fn table_rows() -> Vec<VariableRow> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../../gpuwm/data/ml_export/variables.json");
    let text = std::fs::read_to_string(&path).expect("the variables table");
    let doc: Value = serde_json::from_str(&text).unwrap();
    serde_json::from_value(doc["rows"].clone()).unwrap()
}

fn rows(ids: &[&str]) -> Vec<VariableRow> {
    let all = table_rows();
    ids.iter().map(|id| all.iter().find(|r| r.id == *id).unwrap().clone()).collect()
}

fn request(inputs: Vec<PathBuf>, out: &Path, variables: Vec<VariableRow>, hpa: Vec<u32>) -> Request {
    Request {
        schema: "ml-export.request/v1".into(),
        mode: Mode::Run,
        inputs,
        out: out.to_path_buf(),
        overwrite: false,
        zip: false,
        variables,
        levels: LevelSpec { set: "custom".into(), kind: LevelKind::Pressure, hpa, model_levels: vec![] },
        grid: GridRequest { kind: GridKind::Native, deg: None, method: RegridMethod::Bilinear },
        names: "wb2".into(),
        layout: Layout::Analysis,
        domains: None,
        every_hours: None,
        start: None,
        end: None,
        skip_unavailable: false,
        threads: None,
        spacings_deg: vec![0.05, 0.1, 0.25],
        provenance: Provenance {
            history_attributes: BTreeMap::new(),
            history_engines: BTreeMap::new(),
            history_engine_titles: BTreeMap::new(),
            engine: "engine".into(),
            exporter_version: "0.0.0-test".into(),
            config_sha256: None,
            created_utc: "2026-10-01T00:00:00Z".into(),
            options: "test".into(),
        },
    }
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mlx-test-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn run(req: Request) -> Outcome {
    let mut sink = |_: Value| {};
    execute(req, &mut sink).unwrap()
}

fn read_f32(store: &Path, array: &str, key: &str) -> Vec<f32> {
    let frame = std::fs::read(store.join(array).join(key)).unwrap();
    blosc::decompress(&frame).unwrap().chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect()
}

fn read_u8(store: &Path, array: &str, key: &str) -> Vec<u8> {
    blosc::decompress(&std::fs::read(store.join(array).join(key)).unwrap()).unwrap()
}

fn json(path: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

const D01: Domain = Domain { id: 1, nx: 24, ny: 18, nz: 12, dx: 12_000.0 };
const D02: Domain = Domain { id: 2, nx: 16, ny: 14, nz: 12, dx: 4_000.0 };

fn inputs(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    for hour in 0..3 {
        let path = dir.join(format!("wrfout_d01_{}", label(hour)).replace(':', "_"));
        write_wrfout(&path, &D01, &[hour], true);
        files.push(path);
    }
    // The nest: two records in one file.
    let path = dir.join(format!("wrfout_d02_{}", label(0)).replace(':', "_"));
    write_wrfout(&path, &D02, &[0, 1], true);
    files.push(path);
    files
}

const LEVELS: [u32; 7] = [1000, 925, 850, 500, 200, 50, 30];

#[test]
fn pressure_levels_match_the_analytic_columns() {
    let dir = scratch("analytic");
    let files = inputs(&dir);
    let out = dir.join("export");
    let ids = ["geopotential", "temperature", "u_component_of_wind", "v_component_of_wind", "specific_humidity", "2m_temperature", "surface_pressure", "total_precipitation", "geopotential_at_surface", "land_sea_mask"];
    let outcome = run(request(files, &out, rows(&ids), LEVELS.to_vec()));
    assert_eq!(outcome.domains, vec!["d01".to_string(), "d02".to_string()]);
    assert_eq!(outcome.frames, 5);

    // Every domain is its own finished dataset.
    let store = out.join("d01.zarr");
    assert!(store.join(".zmetadata").is_file());
    assert!(out.join("d02.zarr/.zmetadata").is_file());
    assert!(!out.join(".ml-export-state").exists());
    let attrs = json(&store.join(".zattrs"));
    assert_eq!(attrs["levels_dropped_above_model_top"], serde_json::json!([30]));
    assert_eq!(attrs["model_top_source"], "P_TOP");
    assert_eq!(attrs["initial_condition_source"], "ERA5");
    assert_eq!(attrs["source"], "model history 9.9.9-test");
    let kept = [1000u32, 925, 850, 500, 200, 50];
    let level_meta = json(&store.join("level/.zarray"));
    assert_eq!(level_meta["shape"], serde_json::json!([6]));
    let t_meta = json(&store.join("temperature/.zarray"));
    assert_eq!(t_meta["shape"], serde_json::json!([3, 6, 18, 24]));
    assert_eq!(t_meta["chunks"], serde_json::json!([1, 6, 18, 24]));
    assert_eq!(json(&store.join("temperature/.zattrs"))["_ARRAY_DIMENSIONS"], serde_json::json!(["time", "level", "y", "x"]));
    assert_eq!(json(&out.join("d02.zarr/temperature/.zarray"))["shape"], serde_json::json!([2, 6, 14, 16]));

    let (nx, ny, nz) = (D01.nx, D01.ny, D01.nz);
    let cells = nx * ny;
    let t = read_f32(&store, "temperature", "1.0.0.0");
    let z = read_f32(&store, "geopotential", "1.0.0.0");
    let u = read_f32(&store, "u_component_of_wind", "1.0.0.0");
    let mask = read_u8(&store, "below_ground", "1.0.0.0");
    let mut interior = 0;
    let mut below = 0;
    for (l, &hpa) in kept.iter().enumerate() {
        let target = f64::from(hpa) * 100.0;
        for j in 0..ny {
            for i in 0..nx {
                let c = j * nx + i;
                let ps = surface_pressure(j);
                let p_bot = mass_pressure(ps, 0, nz);
                let p_topm = mass_pressure(ps, nz - 1, nz);
                let got_t = f64::from(t[l * cells + c]);
                let got_z = f64::from(z[l * cells + c]);
                assert_eq!(mask[l * cells + c], u8::from(target > ps), "mask at {hpa} hPa, row {j}");
                if target <= p_bot && target >= p_topm {
                    interior += 1;
                    assert!((got_t - temperature(target)).abs() < 2e-3, "T at {hpa} hPa: {got_t} vs {}", temperature(target));
                    assert!((got_z - phi(ps, terrain(i), target)).abs() < 0.2, "z at {hpa} hPa");
                    let a = angle(i);
                    let expected_u = u_grid(target) * a.cos() - v_grid(target) * a.sin();
                    assert!((f64::from(u[l * cells + c]) - expected_u).abs() < 1e-3, "u at {hpa} hPa");
                } else if target > p_bot {
                    below += 1;
                    // The ECMWF rule from the lowest level's state.
                    let base = ColumnBase {
                        t_bot: temperature(p_bot),
                        p_bot,
                        p_sfc: ps,
                        phi_sfc: 9.81 * terrain(i),
                    };
                    let expected = ecmwf_temperature(&base, target, &ECMWF_RULE);
                    assert!((got_t - expected).abs() < 2e-3, "below-ground T at {hpa} hPa row {j}: {got_t} vs {expected}");
                } else {
                    // Between the top mass level and the lid: temperature held.
                    assert!((got_t - temperature(p_topm)).abs() < 2e-3);
                }
            }
        }
    }
    assert!(interior > 0 && below > 0, "the test must exercise both regimes");

    // Precipitation over the interval: none at the first time, 2.5 mm after.
    let tp0 = read_f32(&store, "total_precipitation", "0.0.0");
    assert!(tp0.iter().all(|v| v.is_nan()));
    let tp1 = read_f32(&store, "total_precipitation", "1.0.0");
    assert!(tp1.iter().all(|v| (v - 0.0025).abs() < 1e-7));
    // Static fields once, as (y, x).
    let zs = read_f32(&store, "geopotential_at_surface", "0.0");
    assert!((f64::from(zs[5]) - 9.81 * terrain(5)).abs() < 1e-3);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn appending_frame_by_frame_writes_the_same_bytes_as_one_run() {
    let dir = scratch("append");
    let files = inputs(&dir);
    let ids = ["temperature", "total_precipitation", "total_precipitation_6hr", "surface_pressure"];
    let one = dir.join("one");
    run(request(files.clone(), &one, rows(&ids), vec![850, 500]));
    let many = dir.join("many");
    for file in &files {
        let mut r = request(vec![file.clone()], &many, rows(&ids), vec![850, 500]);
        r.mode = Mode::Append;
        run(r);
    }
    let mut r = request(vec![], &many, rows(&ids), vec![850, 500]);
    r.mode = Mode::Finalize;
    run(r);
    for store in ["d01.zarr", "d02.zarr"] {
        let a = rw_mlexport::zipout::collect(&one.join(store), "s").unwrap();
        let b = rw_mlexport::zipout::collect(&many.join(store), "s").unwrap();
        assert_eq!(a.iter().map(|x| &x.0).collect::<Vec<_>>(), b.iter().map(|x| &x.0).collect::<Vec<_>>());
        for ((name, pa), (_, pb)) in a.iter().zip(&b) {
            assert_eq!(std::fs::read(pa).unwrap(), std::fs::read(pb).unwrap(), "{store} {name}");
        }
    }
    // Three hours apart at most: no 6 h partner, so that variable is left
    // out and says why.
    let attrs = json(&one.join("d01.zarr/.zattrs"));
    let omitted = attrs["variables_omitted"].as_array().unwrap();
    assert!(omitted[0].as_str().unwrap().starts_with("total_precipitation_6hr"));
    assert!(!one.join("d01.zarr/total_precipitation_6hr").exists());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn compressed_multi_record_inputs_keep_every_time_and_chunk() {
    use std::io::Write;
    let dir = scratch("compressed-records");
    let path = dir.join("wrfout_d01_2026-09-29_00_00_00");
    write_wrfout(&path, &D01, &[0, 1, 6], true);
    let gz = dir.join("wrfout_d01_2026-09-29_00_00_00.gz");
    let mut encoder = flate2::write::GzEncoder::new(std::fs::File::create(&gz).unwrap(), flate2::Compression::fast());
    encoder.write_all(&std::fs::read(&path).unwrap()).unwrap();
    encoder.finish().unwrap();
    let zip = dir.join("history.zip");
    rw_mlexport::zipout::write(&zip, &[("run/wrfout/wrfout_d01_2026-09-29_00_00_00".into(), path.clone())]).unwrap();
    let ids = ["temperature", "total_precipitation", "total_precipitation_6hr"];
    let plain = dir.join("plain");
    assert_eq!(run(request(vec![path], &plain, rows(&ids), vec![500])).frames, 3);
    for (name, source) in [("gz", gz), ("zip", zip)] {
        let out = dir.join(name);
        assert_eq!(run(request(vec![source], &out, rows(&ids), vec![500])).frames, 3);
        let a = rw_mlexport::zipout::collect(&plain.join("d01.zarr"), "s").unwrap();
        let b = rw_mlexport::zipout::collect(&out.join("d01.zarr"), "s").unwrap();
        for ((key, pa), (_, pb)) in a.iter().zip(&b) {
            assert_eq!(std::fs::read(pa).unwrap(), std::fs::read(pb).unwrap(), "{name} {key}");
        }
    }
    std::fs::remove_dir_all(dir).unwrap();
}

/// A ZIP made on Linux names its history members the way WRF does,
/// `..._HH:MM:SS`.  Unpacking one used that member name for the scratch
/// copy, and a colon is not a legal Windows file name (os error 123), so a
/// Windows export of such a ZIP failed before reading anything.
#[test]
fn zip_members_with_the_wrf_colon_spelling_unpack_on_every_platform() {
    let dir = scratch("colon-member");
    let path = dir.join("wrfout_d01_2026-09-29_00_00_00");
    write_wrfout(&path, &D01, &[0, 1, 6], true);
    let zip = dir.join("history.zip");
    rw_mlexport::zipout::write(&zip, &[("run/wrfout/wrfout_d01_2026-09-29_00:00:00".into(), path.clone())]).unwrap();
    let ids = ["temperature", "total_precipitation"];
    let plain = dir.join("plain");
    assert_eq!(run(request(vec![path], &plain, rows(&ids), vec![500])).frames, 3);
    let out = dir.join("zip");
    assert_eq!(run(request(vec![zip], &out, rows(&ids), vec![500])).frames, 3);
    let a = rw_mlexport::zipout::collect(&plain.join("d01.zarr"), "s").unwrap();
    let b = rw_mlexport::zipout::collect(&out.join("d01.zarr"), "s").unwrap();
    assert_eq!(a.len(), b.len());
    for ((key, pa), (_, pb)) in a.iter().zip(&b) {
        assert_eq!(std::fs::read(pa).unwrap(), std::fs::read(pb).unwrap(), "{key}");
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn column_vapour_does_not_depend_on_selected_vertical_levels() {
    let dir = scratch("tcwv-lid");
    let path = dir.join("wrfout_d01_2026-09-29_00_00_00");
    write_wrfout(&path, &D01, &[0], true);
    let mut expected = None;
    for (name, model) in [("pressure", false), ("model", true), ("surface-only", false)] {
        let ids: &[&str] = if name == "surface-only" { &["total_column_water_vapour"] } else { &["temperature", "total_column_water_vapour"] };
        let out = dir.join(name);
        let mut r = request(vec![path.clone()], &out, rows(ids), vec![500]);
        if model { r.levels.kind = LevelKind::Model; }
        run(r);
        let data = std::fs::read(out.join("d01.zarr/total_column_water_vapour/0.0.0")).unwrap();
        if let Some(first) = &expected { assert_eq!(&data, first, "{name}"); } else { expected = Some(data); }
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn table_metadata_aliases_preserve_provenance_and_refuse_mixed_runs() {
    let dir = scratch("metadata-aliases");
    let mut paths = Vec::new();
    let mut digests = Vec::new();
    let names: Value = serde_json::from_str(&std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../../gpuwm/data/ml_export/names.json")
    ).unwrap()).unwrap();
    for prefix in ["GPUWM", "WOOF"] {
        let path = dir.join(format!("{prefix}-wrfout_d01_2026-09-29_00_00_00"));
        write_wrfout_metadata(&path, &D01, &[0], true, false, 0.0, &|_, _| 290.0, prefix, false);
        let out = dir.join(prefix);
        let mut r = request(vec![path.clone()], &out, rows(&["temperature"]), vec![500]);
        r.provenance.history_attributes = serde_json::from_value(names["history_attributes"].clone()).unwrap();
        r.provenance.history_engines = serde_json::from_value(names["history_engines"].clone()).unwrap();
        r.provenance.history_engine_titles = serde_json::from_value(names["history_engine_titles"].clone()).unwrap();
        r.provenance.engine = if prefix == "WOOF" { "gpuwm".into() } else { "woof".into() };
        run(r);
        let attrs = json(&out.join("d01.zarr/.zattrs"));
        assert_eq!(attrs["source"], format!("{} 9.9.9-test", prefix.to_lowercase()));
        assert!(attrs["exporter"].as_str().unwrap().starts_with(if prefix == "WOOF" { "gpuwm ml-export" } else { "woof ml-export" }));
        assert_eq!(attrs["initial_condition_source"], "ERA5");
        assert_eq!(attrs["initial_condition_cycle"], "2026-09-29T00:00:00Z");
        digests.push(attrs["model_config_sha256"].clone());
        paths.push(path);
    }
    assert_eq!(digests[0], digests[1]);
    // Earlier global tapes retain the original version-attribute prefix
    // while their title names the actual source package.
    let legacy = dir.join("legacy-wrfout_d01_2026-09-29_00_00_00");
    let mut bytes = std::fs::read(&paths[0]).unwrap();
    let at = bytes.windows(19).position(|b| b == b"GPUWM model history").unwrap();
    bytes[at..at+19].copy_from_slice(b"WOOF model history ");
    std::fs::write(&legacy, bytes).unwrap();
    let legacy_out = dir.join("legacy-export");
    let mut r = request(vec![legacy], &legacy_out, rows(&["temperature"]), vec![500]);
    r.provenance.history_attributes = serde_json::from_value(names["history_attributes"].clone()).unwrap();
    r.provenance.history_engines = serde_json::from_value(names["history_engines"].clone()).unwrap();
    r.provenance.history_engine_titles = serde_json::from_value(names["history_engine_titles"].clone()).unwrap();
    r.provenance.engine = "gpuwm".into();
    run(r);
    assert_eq!(json(&legacy_out.join("d01.zarr/.zattrs"))["source"], "woof 9.9.9-test");
    let first = dir.join("wrfout_d01_2026-09-29_00_00_00");
    let second = dir.join("wrfout_d01_2026-09-29_01_00_00");
    write_wrfout(&first, &D01, &[0], true);
    write_wrfout(&second, &D01, &[1], true);
    let mut bytes = std::fs::read(&second).unwrap();
    let at = bytes.windows(10).position(|b| b == b"9.9.9-test").unwrap();
    bytes[at..at + 10].copy_from_slice(b"9.9.8-test");
    std::fs::write(&second, bytes).unwrap();
    let mut sink = |_: Value| {};
    let err = execute(request(vec![first, second], &dir.join("mixed"), rows(&["temperature"]), vec![500]), &mut sink).unwrap_err();
    assert!(err.is_refusal() && err.message().contains("provenance"));
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn missing_moisture_and_accumulation_cells_stay_missing() {
    let dir = scratch("missing-cell");
    let path = dir.join("wrfout_d01_2026-09-29_00_00_00");
    write_wrfout_metadata(&path, &D01, &[0, 1], true, false, 0.0, &|_, _| 290.0, "GPUWM", true);
    let out = dir.join("out");
    run(request(vec![path], &out, rows(&["specific_humidity", "relative_humidity", "total_column_water_vapour", "total_precipitation"]), vec![500]));
    for (name, key) in [("specific_humidity", "0.0.0.0"), ("relative_humidity", "0.0.0.0"), ("total_column_water_vapour", "0.0.0"), ("total_precipitation", "1.0.0")] {
        let data = read_f32(&out.join("d01.zarr"), name, key);
        assert!(data[0].is_nan(), "{name} filled missing source as {}", data[0]);
        assert!(data[1].is_finite(), "{name} lost a valid neighbour");
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn surface_only_history_can_export_and_record_unavailable_volumes() {
    let dir = scratch("surface-only");
    let path = dir.join("wrfout_d01_2026-09-29_00_00_00");
    let mut schema = Schema::new(NcFormat::Offset64);
    let time = schema.def_dim("Time", 0, true).unwrap();
    let date = schema.def_dim("DateStrLen", 19, false).unwrap();
    let y = schema.def_dim("south_north", 2, false).unwrap();
    let x = schema.def_dim("west_east", 3, false).unwrap();
    schema.put_global_attr("MAP_PROJ", AttrValue::Ints(vec![6])).unwrap();
    schema.put_global_attr("GRID_ID", AttrValue::Ints(vec![1])).unwrap();
    let times = schema.def_var("Times", NcType::Char, &[time, date]).unwrap();
    let lat = schema.def_var("XLAT", NcType::Float, &[time, y, x]).unwrap();
    let lon = schema.def_var("XLONG", NcType::Float, &[time, y, x]).unwrap();
    let t2 = schema.def_var("T2", NcType::Float, &[time, y, x]).unwrap();
    let mut writer = NcWriter::create(&path, schema).unwrap();
    writer.write_record(0, times, VarData::Char(label(0).as_bytes())).unwrap();
    writer.write_record(0, lat, VarData::F32(&[10.,10.,10.,11.,11.,11.])).unwrap();
    writer.write_record(0, lon, VarData::F32(&[20.,21.,22.,20.,21.,22.])).unwrap();
    writer.write_record(0, t2, VarData::F32(&[280.;6])).unwrap();
    writer.finish().unwrap();
    let exact = dir.join("exact");
    run(request(vec![path.clone()], &exact, rows(&["2m_temperature"]), vec![500]));
    assert!(exact.join("d01.zarr/2m_temperature/.zarray").is_file());
    let skip = dir.join("skip");
    let mut r = request(vec![path], &skip, rows(&["temperature", "2m_temperature"]), vec![500]);
    r.skip_unavailable = true;
    run(r);
    assert_eq!(json(&skip.join("d01.zarr/.zattrs"))["variables_omitted"].as_array().unwrap().len(), 1);
    assert!(!skip.join("d01.zarr/temperature").exists());
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn a_file_that_is_not_a_history_file_is_refused_by_name() {
    let dir = scratch("refuse");
    let bad = dir.join("wrfout_d01_2026-09-29_00_00_00");
    std::fs::write(&bad, b"this is not NetCDF").unwrap();
    let mut sink = |_: Value| {};
    let err = execute(request(vec![bad], &dir.join("out"), rows(&["temperature"]), vec![500]), &mut sink).unwrap_err();
    assert!(err.is_refusal(), "{err}");
    assert!(err.message().contains("cannot be read as a WRF history file"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_missing_field_is_refused_or_skipped_and_recorded() {
    let dir = scratch("missing");
    let path = dir.join("wrfout_d01_2026-09-29_00_00_00");
    write_wrfout(&path, &D01, &[0], false);
    let mut sink = |_: Value| {};
    let err = execute(request(vec![path.clone()], &dir.join("a"), rows(&["temperature", "specific_humidity"]), vec![500]), &mut sink).unwrap_err();
    assert!(err.is_refusal());
    assert!(err.message().contains("QVAPOR"), "{err}");
    let mut r = request(vec![path], &dir.join("b"), rows(&["temperature", "specific_humidity"]), vec![500]);
    r.skip_unavailable = true;
    run(r);
    let attrs = json(&dir.join("b/d01.zarr/.zattrs"));
    assert!(attrs["variables_omitted"][0].as_str().unwrap().contains("QVAPOR"));
    assert!(dir.join("b/d01.zarr/temperature/.zarray").is_file());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_earlier_export_is_not_overwritten_without_asking() {
    let dir = scratch("overwrite");
    let files = inputs(&dir);
    let out = dir.join("x");
    run(request(files.clone(), &out, rows(&["temperature"]), vec![500]));
    let mut sink = |_: Value| {};
    let err = execute(request(files.clone(), &out, rows(&["temperature"]), vec![500]), &mut sink).unwrap_err();
    assert!(err.is_refusal() && err.message().contains("--overwrite"));
    let mut r = request(files, &out, rows(&["temperature"]), vec![500]);
    r.overwrite = true;
    run(r);
    // Someone else's file in the folder is never deleted.
    std::fs::write(out.join("notes.txt"), b"mine").unwrap();
    let mut r = request(vec![], &out, rows(&["temperature"]), vec![500]);
    r.inputs = vec![dir.join("wrfout_d01_2026-09-29_00_00_00")];
    r.overwrite = true;
    let err = execute(r, &mut sink).unwrap_err();
    assert!(err.message().contains("notes.txt"));
    assert!(out.join("notes.txt").is_file());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_latlon_regrid_reproduces_uniform_fields_and_lands_on_the_lattice() {
    let dir = scratch("regrid");
    let files = inputs(&dir);
    let out = dir.join("ll");
    let mut r = request(files, &out, rows(&["2m_temperature", "total_precipitation", "land_sea_mask", "temperature"]), vec![500]);
    r.grid = GridRequest { kind: GridKind::Latlon, deg: Some(0.25), method: RegridMethod::AreaMean };
    r.domains = Some(vec!["d01".into()]);
    run(r);
    let store = out.join("d01.zarr");
    let lat = blosc::decompress(&std::fs::read(store.join("latitude/0")).unwrap()).unwrap();
    let lat: Vec<f64> = lat.chunks_exact(8).map(|b| f64::from_le_bytes(b.try_into().unwrap())).collect();
    assert!(lat.len() >= 2);
    for value in &lat {
        assert!((value / 0.25 - (value / 0.25).round()).abs() < 1e-9, "{value} is not on the 0.25 lattice");
    }
    assert!(lat.windows(2).all(|w| w[1] > w[0]));
    let t2 = read_f32(&store, "2m_temperature", "0.0.0");
    assert!(t2.iter().all(|v| (v - 290.0).abs() < 1e-3));
    // A uniform accumulation keeps its value through the area mean.
    let tp = read_f32(&store, "total_precipitation", "1.0.0");
    assert!(tp.iter().all(|v| (v - 0.0025).abs() < 1e-7));
    let attrs = json(&store.join(".zattrs"));
    assert!(attrs["horizontal_grid"].as_str().unwrap().contains("0.25"));
    let _ = std::fs::remove_dir_all(&dir);
}

/// A smooth, non-uniform 2 m temperature defined in latitude and longitude:
/// a meridional gradient and a 6 degree wave in both directions.
fn t2_wave(lat: f64, lon: f64) -> f64 {
    let x = (lon + 98.0) * std::f64::consts::TAU / 6.0;
    let y = (lat - 38.0) * std::f64::consts::TAU / 6.0;
    280.0 + 0.8 * (lat - 38.0) + 5.0 * x.sin() * y.cos()
}

/// The true area mean of `t2_wave` over one lattice cell, weighted by
/// cos(latitude), by a 64 x 64 midpoint rule: the answer the regrid should
/// hold, computed without the exporter.
fn t2_wave_cell_mean(lat: f64, lon: f64, deg: f64) -> f64 {
    let n = 64;
    let (mut sum, mut weight) = (0.0, 0.0);
    for a in 0..n {
        let slat = lat - deg / 2.0 + (a as f64 + 0.5) * deg / n as f64;
        let w = slat.to_radians().cos();
        for b in 0..n {
            let slon = lon - deg / 2.0 + (b as f64 + 0.5) * deg / n as f64;
            sum += w * t2_wave(slat, if slon > 180.0 { slon - 360.0 } else { slon });
            weight += w;
        }
    }
    sum / weight
}

#[test]
fn an_area_mean_regrid_holds_each_cells_true_area_mean() {
    // Conservation where it applies: a field that varies inside a 0.25 degree
    // cell, on a 12 km grid.  Each regridded cell must hold that cell's true
    // area mean, and the box's cos-latitude integral must equal the analytic
    // one; a sub-grid placed off the cell centre or weighted wrongly moves the
    // mean by the gradient times the offset (0.8 K per degree here).  The
    // point-sampling bilinear regrid of the same field is measured beside it
    // to show the area mean is what closes the gap.
    const WIDE: Domain = Domain { id: 1, nx: 60, ny: 50, nz: 4, dx: 12_000.0 };
    let dir = scratch("conserve");
    let path = dir.join("wrfout_d01_2026-09-29_00_00_00");
    write_wrfout_full(&path, &WIDE, &[0], true, false, 0.0, &t2_wave);
    let deg = 0.25;
    let mut worst = BTreeMap::new();
    for method in [RegridMethod::AreaMean, RegridMethod::Bilinear] {
        let out = dir.join(format!("{method:?}"));
        let mut r = request(vec![path.clone()], &out, rows(&["2m_temperature"]), vec![500]);
        r.grid = GridRequest { kind: GridKind::Latlon, deg: Some(deg), method };
        run(r);
        let store = out.join("d01.zarr");
        let read_f64 = |name: &str| -> Vec<f64> {
            blosc::decompress(&std::fs::read(store.join(name).join("0")).unwrap())
                .unwrap()
                .chunks_exact(8)
                .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
                .collect()
        };
        let (lat, lon) = (read_f64("latitude"), read_f64("longitude"));
        assert!(lat.len() >= 15 && lon.len() >= 25, "the box is {} x {}", lat.len(), lon.len());
        let t2 = read_f32(&store, "2m_temperature", "0.0.0");
        assert_eq!(t2.len(), lat.len() * lon.len());
        let (mut max_err, mut got, mut want, mut weight) = (0.0f64, 0.0, 0.0, 0.0);
        for (r, &la) in lat.iter().enumerate() {
            for (c, &lo) in lon.iter().enumerate() {
                let truth = t2_wave_cell_mean(la, lo, deg);
                let value = t2[r * lon.len() + c] as f64;
                assert!(value.is_finite(), "cell ({la}, {lo}) is empty");
                max_err = max_err.max((value - truth).abs());
                let w = la.to_radians().cos();
                got += w * value;
                want += w * truth;
                weight += w;
            }
        }
        let box_err = (got - want) / weight;
        worst.insert(format!("{method:?}"), (max_err, box_err));
    }
    let (area_max, area_box) = worst["AreaMean"];
    let (bilinear_max, _) = worst["Bilinear"];
    // Measured (2026-10-01, 21 x 31 cells): area mean 0.0144 K at the worst
    // cell and -8.4e-6 K over the box; bilinear point samples 0.0208 K at the
    // worst cell.  What is left is the bilinear interpolation error of a 6
    // degree wave on a 12 km grid.  A sub-grid half a sample off the cell
    // centre (0.025 degree) moves a cell by up to 0.13 K on this wave's
    // 5.2 K per degree slopes and the box by 0.02 K on its 0.8 K per degree
    // gradient.
    assert!(area_max < 0.02, "area mean: worst cell {area_max} K from the cell's true mean");
    assert!(area_box.abs() < 2e-4, "area mean: box mean {area_box} K from the analytic integral");
    assert!(area_max < bilinear_max, "area mean {area_max} K is no closer to the cell means than point samples {bilinear_max} K");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn model_levels_carry_pressure_and_eta() {
    let dir = scratch("model");
    let files = inputs(&dir);
    let out = dir.join("m");
    let mut r = request(files, &out, rows(&["temperature", "pressure"]), vec![]);
    r.levels = LevelSpec { set: "model".into(), kind: LevelKind::Model, hpa: vec![], model_levels: vec![1, 2, 12] };
    r.domains = Some(vec!["d01".into()]);
    run(r);
    let store = out.join("d01.zarr");
    assert_eq!(json(&store.join("pressure/.zarray"))["shape"], serde_json::json!([3, 3, 18, 24]));
    let p = read_f32(&store, "pressure", "0.0.0.0");
    let cells = 18 * 24;
    let expected = mass_pressure(surface_pressure(0), 11, 12);
    assert!((f64::from(p[2 * cells]) - expected).abs() < 0.05);
    assert!(store.join("eta/.zarray").is_file());
    assert!(!store.join("below_ground").exists());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_eta_ladder_keeps_the_models_own_mass_level_geopotential() {
    // A WRF eta core's mass level is the middle of its layer in eta, and the
    // model's own geopotential there is the mean of the layer's two faces:
    // its hydrostatic relation is linear in dry pressure across a layer and
    // the mass level sits at the dry-pressure middle.  On model levels the
    // export carries exactly that, not a value re-placed in ln p.
    let dir = scratch("eta");
    let path = dir.join("wrfout_d01_2026-09-29_00_00_00");
    write_wrfout_ladder(&path, &D01, &[0], true, true);
    let out = dir.join("e");
    let mut r = request(vec![path], &out, rows(&["geopotential", "pressure"]), vec![]);
    r.levels = LevelSpec { set: "model".into(), kind: LevelKind::Model, hpa: vec![], model_levels: vec![] };
    run(r);
    let store = out.join("d01.zarr");
    let z = read_f32(&store, "geopotential", "0.0.0.0");
    let p = read_f32(&store, "pressure", "0.0.0.0");
    let (nx, ny, nz) = (D01.nx, D01.ny, D01.nz);
    let cells = nx * ny;
    for k in [0usize, 5, nz - 1] {
        for j in [0usize, ny - 1] {
            let ps = surface_pressure(j);
            let i = 3;
            let faces = 0.5 * (phi(ps, terrain(i), face_pressure(ps, k, nz)) + phi(ps, terrain(i), face_pressure(ps, k + 1, nz)));
            assert!((f64::from(z[k * cells + j * nx + i]) - faces).abs() < 0.1, "level {k}");
            assert!((f64::from(p[k * cells + j * nx + i]) - eta_mass_pressure(ps, k, nz)).abs() < 0.05);
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_moving_nest_keeps_coordinates_per_time_and_refuses_a_fixed_box() {
    let dir = scratch("moving");
    let first = dir.join("wrfout_d02_2026-09-29_00_00_00");
    let second = dir.join("wrfout_d02_2026-09-29_01_00_00");
    write_wrfout_full(&first, &D02, &[0], true, false, 0.0, &|_, _| 290.0);
    write_wrfout_full(&second, &D02, &[1], true, false, 3.0, &|_, _| 290.0);
    let out = dir.join("n");
    run(request(vec![first.clone(), second.clone()], &out, rows(&["temperature"]), vec![500]));
    let store = out.join("d02.zarr");
    let lat = json(&store.join("latitude/.zarray"));
    assert_eq!(lat["shape"], serde_json::json!([2, 14, 16]));
    assert_eq!(json(&store.join("latitude/.zattrs"))["_ARRAY_DIMENSIONS"], serde_json::json!(["time", "y", "x"]));
    let lon0 = read_f32(&store, "longitude", "0.0.0");
    let lon1 = read_f32(&store, "longitude", "1.0.0");
    assert!(lon1[0] > lon0[0], "the second time's grid sits east of the first");
    let mut r = request(vec![first, second], &dir.join("ll"), rows(&["temperature"]), vec![500]);
    r.grid = GridRequest { kind: GridKind::Latlon, deg: Some(0.1), method: RegridMethod::Bilinear };
    let mut sink = |_: Value| {};
    let err = execute(r, &mut sink).unwrap_err();
    assert!(err.is_refusal() && err.message().contains("moving nest"), "{err}");
    assert!(!dir.join("ll").exists(), "a refused export leaves no folder it made");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn every_table_row_names_an_operator_the_binary_has() {
    for row in table_rows() {
        rw_mlexport::ops::Op::parse(&row.op).unwrap_or_else(|e| panic!("{}: {e}", row.id));
        for scheme in ["wb2", "era5"] {
            assert!(row.names.contains_key(scheme), "{} lacks a {scheme} name", row.id);
        }
    }
}

#[test]
fn on_an_eta_ladder_pressure_level_geopotential_is_read_between_interfaces() {
    // A WRF eta core's mass level sits at the arithmetic mean of its faces'
    // pressures, and the mean of two face geopotentials belongs to their
    // GEOMETRIC mean: the layer mean paired with the mass-level pressure is
    // too high.  The file states its eta levels, so geopotential inside the
    // column is read between the faces, where the analytic column is exact.
    let dir = scratch("eta-pressure");
    let path = dir.join("wrfout_d01_2026-09-29_00_00_00");
    write_wrfout_ladder(&path, &D01, &[0], true, true);
    let out = dir.join("p");
    let kept = [850u32, 700, 500, 300];
    run(request(vec![path], &out, rows(&["geopotential", "temperature"]), kept.to_vec()));
    let store = out.join("d01.zarr");
    let z = read_f32(&store, "geopotential", "0.0.0.0");
    let (nx, ny, nz) = (D01.nx, D01.ny, D01.nz);
    let cells = nx * ny;
    let mut checked = 0;
    for (l, &hpa) in kept.iter().enumerate() {
        let target = f64::from(hpa) * 100.0;
        for j in 0..ny {
            let ps = surface_pressure(j);
            // Under the lowest mass level the ECMWF rule holds, as for
            // every field; this checks the column above it.
            if target > eta_mass_pressure(ps, 0, nz) {
                continue;
            }
            for i in [0usize, 7, nx - 1] {
                let truth = phi(ps, terrain(i), target);
                let got = f64::from(z[l * cells + j * nx + i]);
                assert!((got - truth).abs() < 0.2, "z at {hpa} hPa, row {j}: {got} vs {truth}");
                checked += 1;
            }
        }
    }
    assert!(checked > 100);

    // The pairing this replaces, at 500 hPa in the first row: the layer
    // means read at the mass-level pressures, in ln p.
    let ps = surface_pressure(0);
    let layer = |k: usize| 0.5 * (phi(ps, 0.0, face_pressure(ps, k, nz)) + phi(ps, 0.0, face_pressure(ps, k + 1, nz)));
    let target = 50_000.0f64;
    let k = (0..nz - 1)
        .find(|&k| eta_mass_pressure(ps, k, nz) >= target && eta_mass_pressure(ps, k + 1, nz) < target)
        .unwrap();
    let w = (target.ln() - eta_mass_pressure(ps, k, nz).ln())
        / (eta_mass_pressure(ps, k + 1, nz).ln() - eta_mass_pressure(ps, k, nz).ln());
    let paired = layer(k) + w * (layer(k + 1) - layer(k));
    assert!(paired - phi(ps, 0.0, target) > 20.0, "the layer-mean pairing is {} m2 s-2 high", paired - phi(ps, 0.0, target));

    // The metadata says which arithmetic each array holds.
    let geopotential = json(&store.join("geopotential/.zattrs"));
    assert!(geopotential["vertical_interpolation"].as_str().unwrap().contains("layer interfaces (PH + PHB)"));
    let temperature = json(&store.join("temperature/.zattrs"));
    assert!(temperature["vertical_interpolation"].as_str().unwrap().contains("mass levels"));
    let group = json(&store.join(".zattrs"));
    assert!(group["geopotential_vertical_interpolation"].is_string());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_file_without_eta_levels_keeps_the_mass_level_geopotential_and_says_so() {
    // No ZNW (the default ladder here, a height-coordinate frame elsewhere):
    // the mass level is the middle of its layer in ln p, the layer mean
    // already belongs to the mass-level pressure, and nothing changes.
    let dir = scratch("no-eta");
    let path = dir.join("wrfout_d01_2026-09-29_00_00_00");
    write_wrfout(&path, &D01, &[0], true);
    let out = dir.join("q");
    run(request(vec![path], &out, rows(&["geopotential"]), vec![500]));
    let store = out.join("d01.zarr");
    let geopotential = json(&store.join("geopotential/.zattrs"));
    assert!(geopotential["vertical_interpolation"].as_str().unwrap().contains("mass levels"));
    assert!(json(&store.join(".zattrs")).get("geopotential_vertical_interpolation").is_none());
    let _ = std::fs::remove_dir_all(&dir);
}
