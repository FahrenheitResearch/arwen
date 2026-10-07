//! `rw_nexrad grid-composite` on synthetic fixtures: a WPS Lambert model
//! grid with terrain, a source lattice offset from it by whole cells, an
//! observed 3D window with a known column shape, and frames with known
//! member values.  Everything is written to disk and read back through the
//! same entry point the binary uses.

use std::path::{Path, PathBuf};

use chrono::{DateTime, Duration, TimeZone, Utc};
use rw_nexrad::grid_composite::{
    self as gc, build_library, composite_bin, height_bin, reduce_columns, ColumnClass,
    CompositeRequest, Frames, Lattice, LambertLattice, ProfileLibrary,
};
use rw_nexrad::grid_ref::{ModelGrid, GRID_SCHEMA, NO_COVERAGE, OBSERVED_NO_ECHO, REF_SCHEMA};
use rw_nexrad::s3::hex_sha256;
use serde_json::json;
use static_fields::projection::{GridSpec, ProjectionKind};

const NX: usize = 40;
const NY: usize = 36;
const NZ: usize = 20;
const DX: f64 = 3000.0;
/// Source lattice offset from the model grid, whole cells.
const KI: usize = 3;
const KJ: usize = 2;
const SNX: usize = NX + 6;
const SNY: usize = NY + 4;
const IDENTITY: &str = "4f7c0de1a5e5ed00000000000000000000000000000000000000000000000001";

fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("grid_composite").join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn spec() -> GridSpec {
    GridSpec {
        kind: ProjectionKind::Lambert,
        ref_lat: 35.3,
        ref_lon: -97.3,
        truelat1: 30.0,
        truelat2: 60.0,
        stand_lon: -97.3,
        dx: DX,
        dy: DX,
        e_we: NX as i64 + 1,
        e_sn: NY as i64 + 1,
        known_x: (NX as f64 + 1.0) / 2.0,
        known_y: (NY as f64 + 1.0) / 2.0,
        moad_cen_lat: 35.3,
        moad_cen_lon: -97.3,
        lat_deg: Vec::new(),
        lon0_deg: 0.0,
        dlon_deg: 0.0,
    }
}

/// Terrain from 300 to ~700 m; 600 m layers above it.
fn z_w() -> Vec<f64> {
    let mut z = vec![0.0; (NZ + 1) * NY * NX];
    for k in 0..=NZ {
        for j in 0..NY {
            for i in 0..NX {
                let surface = 300.0 + 10.0 * i as f64;
                z[(k * NY + j) * NX + i] = surface + 600.0 * k as f64;
            }
        }
    }
    z
}

/// Level centre heights above ground: 300, 900, 1500, ...
fn agl(k: usize) -> f64 {
    300.0 + 600.0 * k as f64
}

fn write_grid(dir: &Path, identity: &str) -> PathBuf {
    let z = z_w();
    let bytes: Vec<u8> = z.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(dir.join("grid.z_w.f64"), &bytes).unwrap();
    let descriptor = json!({
        "schema": GRID_SCHEMA,
        "name": "synthetic",
        "identity_sha256": identity,
        "nx": NX, "ny": NY, "nz": NZ,
        "projection": serde_json::to_value(spec()).unwrap(),
        "z_w": {"file": "grid.z_w.f64", "dtype": "<f8", "shape": [NZ + 1, NY, NX],
                "sha256": hex_sha256(&bytes)},
    });
    let path = dir.join("grid.json");
    std::fs::write(&path, serde_json::to_string_pretty(&descriptor).unwrap()).unwrap();
    path
}

fn source_lattice(radius: f64) -> LambertLattice {
    let s = spec();
    LambertLattice::new(
        s.truelat1,
        s.truelat2,
        s.stand_lon,
        s.ref_lat,
        s.ref_lon,
        radius,
        (SNX, SNY),
        (DX, DX),
        ((1.0 - s.known_x) * DX - KI as f64 * DX, (1.0 - s.known_y) * DX - KJ as f64 * DX),
    )
    .unwrap()
}

fn lattice_json(lattice: &LambertLattice, shift_deg: f64) -> serde_json::Value {
    let s = spec();
    let mut corners = Lattice::Lambert(lattice.clone()).corners_json();
    for pair in corners.as_object_mut().unwrap().values_mut() {
        pair[0] = json!(pair[0].as_f64().unwrap() + shift_deg);
    }
    json!({
        "kind": "lambert", "truelat1": s.truelat1, "truelat2": s.truelat2,
        "stand_lon": s.stand_lon, "ref_lat": s.ref_lat, "ref_lon": s.ref_lon,
        "earth_radius_m": lattice.radius, "nx": SNX, "ny": SNY, "dx_m": DX, "dy_m": DX,
        "x0_m": lattice.x0, "y0_m": lattice.y0, "row_order": "south_to_north",
        "corners_latlon": corners,
    })
}

fn start() -> DateTime<Utc> {
    Utc.with_ymd_and_hms(2026, 10, 1, 18, 0, 0).unwrap()
}

/// Member values per model column, by region (columns are model (j, i)).
fn member_value(m: usize, j: usize, i: usize) -> f32 {
    match (j / 9, i / 10) {
        // echo: three members around a storm, shifted strengths
        (0, _) => 30.0 + (i % 10) as f32 * 2.0 + m as f32 * 1.5,
        // consensus clear
        (1, 0) | (1, 1) => [-10.0, 0.0, 4.9][m],
        // in between: no member is echo, not all are clear
        (1, 2) | (1, 3) => [8.0, 10.0, 12.0][m],
        // NaN in more than half the members
        (2, _) => if m == 0 { 2.0 } else { f32::NAN },
        // one NaN, the rest clear: not every member finite
        _ => if m == 2 { f32::NAN } else { 1.0 },
    }
}

/// Frames on the offset lattice for `stamps`, with three members.
fn write_frames(dir: &Path, stamps: &[DateTime<Utc>], causal: bool, kind: &str, corner_shift: f64,
                bad_shape: bool) -> PathBuf {
    let root = dir.join("frames");
    std::fs::create_dir_all(&root).unwrap();
    let lattice = source_lattice(6_370_000.0);
    let members = 3usize;
    let mut frames = Vec::new();
    for stamp in stamps {
        let mut data = vec![f32::NAN; members * SNY * SNX];
        for m in 0..members {
            for j in 0..NY {
                for i in 0..NX {
                    data[(m * SNY + j + KJ) * SNX + i + KI] = member_value(m, j, i);
                }
            }
        }
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let name = stamp.format("%Y%m%dT%H%MZ").to_string();
        std::fs::create_dir_all(root.join(&name)).unwrap();
        std::fs::write(root.join(&name).join("refc.f32"), &bytes).unwrap();
        let shape = if bad_shape { vec![members, SNY - 1, SNX] } else { vec![members, SNY, SNX] };
        frames.push(json!({
            "valid": stamp.format("%Y-%m-%dT%H:%M:%SZ").to_string(),
            "lead_minutes": (*stamp - start()).num_minutes(),
            "file": format!("{name}/refc.f32"), "shape": shape, "dtype": "float32-le",
            "sha256": hex_sha256(&bytes),
        }));
    }
    let receipt = json!({
        "schema": gc::FRAMES_SCHEMA, "status": "READY",
        "source": {"kind": kind, "id": "synthetic-3member", "package": "test", "revision": "0"},
        "issue_time": "2026-10-01T18:00:00Z", "latest_input_time": "2026-10-01T18:00:00Z",
        "causal": causal, "variable": "refc", "units": "dBZ", "members": members,
        "lattice": lattice_json(&lattice, corner_shift),
        "frames": frames,
    });
    std::fs::write(root.join("nowcast.json"), serde_json::to_string_pretty(&receipt).unwrap()).unwrap();
    root
}

/// The known column shape of the library window: echo below 7.5 km with a
/// peak at level 4, observed clear above.
fn shape_offset(k: usize) -> Option<f32> {
    (k <= 12).then(|| -0.5 * (k as f32 - 4.0).abs())
}

/// An observed 3D window: `strong` columns with composite 42 dBZ and the
/// known shape, the rest observed clear.
fn write_window(dir: &Path, identity: &str, strong: usize) -> PathBuf {
    let mut field = vec![OBSERVED_NO_ECHO; NZ * NY * NX];
    for c in 0..strong.min(NY * NX) {
        for k in 0..NZ {
            field[k * NY * NX + c] = match shape_offset(k) {
                Some(offset) => 42.0 + offset,
                None => OBSERVED_NO_ECHO,
            };
        }
    }
    // A strip with no coverage at all, which the library must ignore.
    for k in 0..NZ {
        field[k * NY * NX + NY * NX - 1] = NO_COVERAGE;
    }
    let bytes: Vec<u8> = field.iter().flat_map(|v| v.to_le_bytes()).collect();
    let wdir = dir.join("levelii-t0").join("20261001T1800Z");
    std::fs::create_dir_all(&wdir).unwrap();
    std::fs::write(wdir.join("ref.f32"), &bytes).unwrap();
    let receipt = json!({
        "schema": REF_SCHEMA, "status": "READY", "shape": [NZ, NY, NX],
        "data": {"file": "ref.f32", "bytes": bytes.len(), "sha256": hex_sha256(&bytes)},
        "grid": {"identity_sha256": identity, "nx": NX, "ny": NY, "nz": NZ},
        "window": {"end": "2026-10-01T18:00:00Z"},
    });
    std::fs::write(wdir.join("ref.json"), serde_json::to_string_pretty(&receipt).unwrap()).unwrap();
    wdir.join("ref.f32")
}

fn request(dir: &Path, grid: PathBuf, frames: PathBuf, window: Option<PathBuf>, windows: usize,
           threads: usize, out: &str) -> CompositeRequest {
    CompositeRequest {
        grid,
        frames,
        profile_from: window,
        profile_table: None,
        start: start(),
        window_minutes: 10,
        windows,
        out: dir.join(out),
        echo_dbz: 15.0,
        clear_dbz: 5.0,
        oracle: false,
        threads,
    }
}

fn stamps(n: usize) -> Vec<DateTime<Utc>> {
    (1..=n).map(|k| start() + Duration::minutes(10 * k as i64)).collect()
}

fn read_f32(path: &Path) -> Vec<f32> {
    std::fs::read(path)
        .unwrap()
        .chunks_exact(4)
        .map(|w| f32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        .collect()
}

fn read_json(path: &Path) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn error_of(request: &CompositeRequest) -> String {
    gc::grid_composite(request).expect_err("must refuse").to_string()
}

// ---------------------------------------------------------------------------

#[test]
fn a_lattice_offset_by_whole_cells_maps_exactly() {
    let dir = scratch("map");
    let grid = ModelGrid::load(&write_grid(&dir, IDENTITY)).unwrap();
    let lattice = Lattice::Lambert(source_lattice(6_370_000.0));
    let mapping = gc::map_columns(&grid, &lattice);
    for j in 0..NY {
        for i in 0..NX {
            assert_eq!(
                mapping[j * NX + i],
                Some(((j + KJ) * SNX + i + KI) as u32),
                "model (j={j}, i={i})"
            );
        }
    }
    // The same lattice numbers on HRRR's 6,371,229 m sphere are a different
    // lattice: away from the reference point every column sits elsewhere,
    // by the sphere ratio times its distance (sub-cell on this small grid,
    // about 260 m at the edge of HRRR's).
    let other = Lattice::Lambert(source_lattice(6_371_229.0));
    let (lat, lon) = grid.cell_latlon(0);
    let (fi, fj) = Lattice::Lambert(source_lattice(6_370_000.0)).fractional_index(lat, lon);
    let (gi, gj) = other.fractional_index(lat, lon);
    assert!((fi - KI as f64).abs() < 1e-6 && (fj - KJ as f64).abs() < 1e-6, "{fi} {fj}");
    let shift_m = ((gi - fi).hypot(gj - fj)) * DX;
    let distance_m = (((NX as f64 - 1.0) / 2.0).hypot((NY as f64 - 1.0) / 2.0)) * DX;
    let expected = distance_m * (6_371_229.0 / 6_370_000.0 - 1.0);
    assert!((shift_m - expected).abs() < 0.05 * expected, "{shift_m} vs {expected}");
}

#[test]
fn the_contract_lattice_conventions_are_read_as_written() {
    let s = spec();
    let lattice = source_lattice(6_371_229.0);
    let mut block = lattice_json(&lattice, 0.0);
    assert!(Lattice::from_json(&block).is_ok());
    // The corners are keyed, not ordered.
    assert_eq!(
        block["corners_latlon"].as_object().unwrap().keys().cloned().collect::<Vec<_>>(),
        ["j0_i0", "j0_in", "jn_i0", "jn_in"]
    );
    // A north-first copy of the same lattice: signed dy, row 0 at the top.
    let north = LambertLattice::new(
        s.truelat1, s.truelat2, s.stand_lon, s.ref_lat, s.ref_lon, 6_371_229.0,
        (SNX, SNY), (DX, -DX), (lattice.x0, lattice.y0 + (SNY as f64 - 1.0) * DX),
    )
    .unwrap();
    let mut flipped = lattice_json(&north, 0.0);
    flipped["dy_m"] = json!(-DX);
    flipped["row_order"] = json!("north_to_south");
    flipped["corners_latlon"] = Lattice::Lambert(north.clone()).corners_json();
    let flipped_lattice = Lattice::from_json(&flipped).unwrap();
    let south_lattice = Lattice::from_json(&block).unwrap();
    // Points off the half-cell lines, where both lattices round the same way.
    for (lat, lon) in [(35.31, -97.27), (35.52, -97.13), (35.13, -97.58)] {
        let a = south_lattice.nearest(lat, lon).unwrap();
        let b = flipped_lattice.nearest(lat, lon).unwrap();
        assert_eq!((a / SNX, a % SNX), (SNY - 1 - b / SNX, b % SNX));
    }
    // A row order that disagrees with the sign of dy is refused.
    flipped["row_order"] = json!("south_to_north");
    assert!(Lattice::from_json(&flipped).unwrap_err().contains("row_order"));
    // A reference meridian other than stand_lon is ambiguous: refused.
    block["ref_lon"] = json!(-96.0);
    assert!(Lattice::from_json(&block).unwrap_err().contains("stand_lon"));
    // Positional corners are not the contract.
    let mut listed = lattice_json(&lattice, 0.0);
    listed["corners_latlon"] = json!([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]);
    assert!(Lattice::from_json(&listed).unwrap_err().contains("j0_i0"));
}

#[test]
fn corner_causal_and_oracle_rules() {
    let dir = scratch("refusals");
    let grid = write_grid(&dir, IDENTITY);
    let window = write_window(&dir, IDENTITY, 800);

    // Corners shifted by 0.02 degrees (about 2 km) do not reproduce.
    let frames = write_frames(&dir, &stamps(2), true, "nowcast", 0.02, false);
    let err = Frames::load(&frames).unwrap_err();
    assert!(err.contains("corner"), "{err}");
    assert!(error_of(&request(&dir, grid.clone(), frames, Some(window.clone()), 2, 2, "o1")).contains("corner"));

    // Not causal and no --oracle: refused.
    let dir2 = scratch("refusals-causal");
    let grid2 = write_grid(&dir2, IDENTITY);
    let window2 = write_window(&dir2, IDENTITY, 800);
    let frames = write_frames(&dir2, &stamps(2), false, "nowcast", 0.0, false);
    let mut r = request(&dir2, grid2.clone(), frames.clone(), Some(window2.clone()), 2, 2, "o2");
    assert!(error_of(&r).contains("not causal"));
    // With --oracle it runs and every window says oracle.
    r.oracle = true;
    gc::grid_composite(&r).unwrap();
    for stamp in stamps(2) {
        let receipt = read_json(&r.out.join(gc::window_stamp(stamp)).join("ref.json"));
        assert_eq!(receipt["source"]["lead_class"], "oracle");
    }
    // A causal nowcast is a forecast.
    let dir3 = scratch("refusals-forecast");
    let grid3 = write_grid(&dir3, IDENTITY);
    let window3 = write_window(&dir3, IDENTITY, 800);
    let frames = write_frames(&dir3, &stamps(2), true, "nowcast", 0.0, false);
    let r = request(&dir3, grid3, frames, Some(window3), 2, 2, "o3");
    gc::grid_composite(&r).unwrap();
    let receipt = read_json(&r.out.join(gc::window_stamp(stamps(1)[0])).join("ref.json"));
    assert_eq!(receipt["source"]["lead_class"], "forecast");
    // Observed frames valid after the start are oracle even when the
    // receipt says causal: refused without --oracle.
    let dir4 = scratch("refusals-observed");
    let grid4 = write_grid(&dir4, IDENTITY);
    let window4 = write_window(&dir4, IDENTITY, 800);
    let frames = write_frames(&dir4, &stamps(2), true, "observed", 0.0, false);
    let r = request(&dir4, grid4, frames, Some(window4), 2, 2, "o4");
    assert!(error_of(&r).contains("--oracle was not given"));
}

#[test]
fn a_known_library_profile_is_recovered_from_a_synthetic_window() {
    let dir = scratch("library");
    let grid = ModelGrid::load(&write_grid(&dir, IDENTITY)).unwrap();
    let window = read_f32(&write_window(&dir, IDENTITY, 800));
    let bins = gc::level_height_bins(&grid);
    let library = build_library(&window, NZ, NX * NY, &bins, json!({}));
    assert_eq!(library.columns.binned, 800);
    assert_eq!(library.columns.at_or_above_30_dbz, 800);
    let b = composite_bin(42.0);
    assert_eq!(b, 4);
    for k in 0..NZ {
        let Some(h) = height_bin(agl(k)) else { continue };
        let cell = &library.cells[b][h];
        match shape_offset(k) {
            Some(offset) => {
                assert_eq!(cell.offset_db, Some(offset as f64), "level {k}");
                assert_eq!(cell.echo_fraction, 1.0, "level {k}");
                assert_eq!(cell.echo, 800);
            }
            None => {
                assert_eq!(cell.offset_db, None, "level {k}");
                assert_eq!(cell.echo_fraction, 0.0);
                assert_eq!(cell.clear, 800);
            }
        }
        // Thin bins take bin 4's shape at the same height.
        assert_eq!(library.cells[0][h].filled_from, Some(4));
        assert_eq!(library.cells[9][h].offset_db, cell.offset_db);
    }
}

#[test]
fn columns_classify_and_echo_columns_expand_to_their_composite() {
    let dir = scratch("expand");
    let grid = write_grid(&dir, IDENTITY);
    let window = write_window(&dir, IDENTITY, 800);
    let frames = write_frames(&dir, &stamps(3), true, "nowcast", 0.0, false);
    let r = request(&dir, grid.clone(), frames.clone(), Some(window), 3, 2, "out");
    let record: serde_json::Value = serde_json::from_str(&gc::grid_composite(&r).unwrap()).unwrap();
    assert_eq!(record["status"], "READY");
    assert_eq!(record["windows"].as_array().unwrap().len(), 3);

    // The reduced values, recomputed from the frames, to check the field.
    let loaded = Frames::load(&frames).unwrap();
    let members = loaded.read(&loaded.frames[0].1).unwrap();
    let model = ModelGrid::load(&grid).unwrap();
    let mapping = gc::map_columns(&model, &loaded.lattice);
    let classes = reduce_columns(&members, 3, &mapping, 15.0, 5.0);

    let field = read_f32(&r.out.join(gc::window_stamp(stamps(1)[0])).join("ref.f32"));
    let at = |k: usize, j: usize, i: usize| field[(k * NY + j) * NX + i];
    let (mut echo, mut clear, mut none) = (0, 0, 0);
    for j in 0..NY {
        for i in 0..NX {
            let column: Vec<f32> = (0..NZ).map(|k| at(k, j, i)).collect();
            match classes[j * NX + i] {
                ColumnClass::Echo(c) => {
                    echo += 1;
                    assert!(j / 9 == 0, "echo outside the storm region at ({j}, {i})");
                    let max = column.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    assert_eq!(max, c, "column ({j}, {i}) maximum is not its composite");
                    // The shape is the library's: echo at levels 0..=12,
                    // left to the model above.
                    for (k, v) in column.iter().enumerate() {
                        match shape_offset(k) {
                            Some(offset) => assert_eq!(*v, c + offset, "({j}, {i}) level {k}"),
                            None => assert_eq!(*v, NO_COVERAGE, "({j}, {i}) level {k}"),
                        }
                    }
                }
                ColumnClass::Clear => {
                    clear += 1;
                    assert!(column.iter().all(|v| *v == OBSERVED_NO_ECHO), "({j}, {i})");
                    assert_eq!((j / 9, i / 10 < 2), (1, true));
                }
                ColumnClass::NoCoverage => {
                    none += 1;
                    assert!(column.iter().all(|v| *v == NO_COVERAGE), "({j}, {i})");
                }
            }
        }
    }
    assert_eq!(echo, 9 * NX);
    assert_eq!(clear, 9 * 20);
    assert_eq!(none, NX * NY - echo - clear);
    // In between (8, 10, 12), NaN majority, and one NaN with clear members
    // are all no coverage.
    assert!(matches!(classes[9 * NX + 25], ColumnClass::NoCoverage));
    assert!(matches!(classes[20 * NX + 5], ColumnClass::NoCoverage));
    assert!(matches!(classes[30 * NX + 5], ColumnClass::NoCoverage));
}

#[test]
fn the_probability_matched_mean_keeps_the_member_value_distribution() {
    // Three members of one storm placed at three different columns: the
    // mean smears it, the PMM keeps the pooled amplitudes.
    let cells = 30;
    let mapping: Vec<Option<u32>> = (0..cells as u32).map(Some).collect();
    let mut members = vec![0.0f32; 3 * cells];
    for m in 0..3 {
        for c in 0..cells {
            let d = (c as i32 - (10 + 3 * m as i32)).abs() as f32;
            members[m * cells + c] = (55.0 - 6.0 * d).max(0.0);
        }
    }
    let classes = reduce_columns(&members, 3, &mapping, -1.0, -50.0);
    let mut reduced: Vec<f32> = classes
        .iter()
        .map(|c| match c {
            ColumnClass::Echo(v) => *v,
            other => panic!("{other:?}"),
        })
        .collect();
    let mut pool = members.clone();
    pool.sort_by(|a, b| b.total_cmp(a));
    // Every value the PMM writes is a member value, sampled at the middle
    // of each rank's share of the pool.
    reduced.sort_by(|a, b| b.total_cmp(a));
    let expected: Vec<f32> = (0..cells).map(|r| pool[3 * r + 1]).collect();
    assert_eq!(reduced, expected);
    // The peak survives; a plain mean would have cut it.
    let mean_peak = (0..cells)
        .map(|c| (0..3).map(|m| members[m * cells + c]).sum::<f32>() / 3.0)
        .fold(f32::NEG_INFINITY, f32::max);
    assert!(reduced[0] > mean_peak, "{} vs {mean_peak}", reduced[0]);
    assert_eq!(reduced[0], 55.0);
}

#[test]
fn receipts_match_their_bytes_and_the_descriptor() {
    let dir = scratch("receipts");
    let grid = write_grid(&dir, IDENTITY);
    let window = write_window(&dir, IDENTITY, 800);
    let frames = write_frames(&dir, &stamps(2), true, "nowcast", 0.0, false);
    let r = request(&dir, grid, frames, Some(window), 2, 2, "out");
    gc::grid_composite(&r).unwrap();
    let library_bytes = std::fs::read(r.out.join("profile-library.json")).unwrap();
    for stamp in stamps(2) {
        let wdir = r.out.join(gc::window_stamp(stamp));
        let bytes = std::fs::read(wdir.join("ref.f32")).unwrap();
        let receipt = read_json(&wdir.join("ref.json"));
        assert_eq!(receipt["schema"], REF_SCHEMA);
        assert_eq!(receipt["status"], "READY");
        assert_eq!(receipt["data"]["sha256"], hex_sha256(&bytes));
        assert_eq!(receipt["data"]["bytes"], bytes.len());
        assert_eq!(receipt["data"]["file"], "ref.f32");
        assert_eq!(receipt["grid"]["identity_sha256"], IDENTITY);
        assert_eq!(receipt["shape"], json!([NZ, NY, NX]));
        assert_eq!(receipt["window"]["end"], stamp.format("%Y-%m-%dT%H:%M:%SZ").to_string());
        assert_eq!(receipt["source"]["library_sha256"], hex_sha256(&library_bytes));
        let counts = &receipt["counts"];
        assert_eq!(
            counts["echo_cells"].as_u64().unwrap()
                + counts["clear_cells"].as_u64().unwrap()
                + counts["no_coverage_cells"].as_u64().unwrap(),
            (NZ * NY * NX) as u64
        );
    }
    let library = ProfileLibrary::load(&r.out.join("profile-library.json")).unwrap();
    assert_eq!(library.source["grid_identity_sha256"], IDENTITY);
}

#[test]
fn output_is_byte_identical_at_one_and_four_threads() {
    let dir = scratch("threads");
    let grid = write_grid(&dir, IDENTITY);
    let window = write_window(&dir, IDENTITY, 800);
    let frames = write_frames(&dir, &stamps(3), true, "nowcast", 0.0, false);
    let one = request(&dir, grid.clone(), frames.clone(), Some(window.clone()), 3, 1, "one");
    let four = request(&dir, grid, frames, Some(window), 3, 4, "four");
    gc::grid_composite(&one).unwrap();
    gc::grid_composite(&four).unwrap();
    for stamp in stamps(3) {
        let name = gc::window_stamp(stamp);
        assert_eq!(
            std::fs::read(one.out.join(&name).join("ref.f32")).unwrap(),
            std::fs::read(four.out.join(&name).join("ref.f32")).unwrap(),
            "{name}"
        );
    }
    assert_eq!(
        std::fs::read(one.out.join("profile-library.json")).unwrap(),
        std::fs::read(four.out.join("profile-library.json")).unwrap()
    );
}

#[test]
fn a_missing_frame_a_thin_library_and_a_foreign_window_are_refused() {
    let dir = scratch("missing");
    let grid = write_grid(&dir, IDENTITY);
    let window = write_window(&dir, IDENTITY, 800);
    let frames = write_frames(&dir, &stamps(2), true, "nowcast", 0.0, false);
    // Three windows, two frames.
    let r = request(&dir, grid.clone(), frames.clone(), Some(window), 3, 2, "out");
    let err = error_of(&r);
    assert!(err.contains("no frame is valid at window end 2026-10-01T18:30:00Z"), "{err}");
    assert!(!r.out.join(gc::window_stamp(stamps(1)[0])).exists(), "nothing written before the refusal");

    // A library window with 100 strong columns is too thin...
    let dir2 = scratch("thin");
    let grid2 = write_grid(&dir2, IDENTITY);
    let thin = write_window(&dir2, IDENTITY, 100);
    let frames2 = write_frames(&dir2, &stamps(2), true, "nowcast", 0.0, false);
    let r = request(&dir2, grid2.clone(), frames2.clone(), Some(thin), 2, 2, "out");
    assert!(error_of(&r).contains("fewer than 500"));
    // ...unless a library built elsewhere is given.
    let dir3 = scratch("table");
    let grid3 = write_grid(&dir3, IDENTITY);
    let full = write_window(&dir3, IDENTITY, 800);
    let frames3 = write_frames(&dir3, &stamps(2), true, "nowcast", 0.0, false);
    let first = request(&dir3, grid3.clone(), frames3.clone(), Some(full), 2, 2, "first");
    gc::grid_composite(&first).unwrap();
    let mut second = request(&dir2, grid2, frames2, None, 2, 2, "second");
    second.profile_table = Some(first.out.join("profile-library.json"));
    gc::grid_composite(&second).unwrap();

    // A window gridded onto another grid identity.
    let dir4 = scratch("foreign");
    let grid4 = write_grid(&dir4, IDENTITY);
    let foreign = write_window(&dir4, "another-grid", 800);
    let frames4 = write_frames(&dir4, &stamps(2), true, "nowcast", 0.0, false);
    let r = request(&dir4, grid4, frames4, Some(foreign), 2, 2, "out");
    assert!(error_of(&r).contains("gridded onto"));

    // Members whose shape disagrees with the lattice.
    let dir5 = scratch("shape");
    let frames5 = write_frames(&dir5, &stamps(2), true, "nowcast", 0.0, true);
    assert!(Frames::load(&frames5).unwrap_err().contains("shapes disagree"));
}

#[test]
fn the_cli_parses_and_refuses() {
    let args = |list: &[&str]| list.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    let r = gc::parse_request(&args(&[
        "--grid", "g.json", "--frames", "f", "--profile-from", "w/ref.f32", "--start",
        "2026-10-01T18:00Z", "--window-minutes", "10", "--windows", "12", "--out", "o",
        "--oracle", "--threads", "3",
    ]))
    .unwrap();
    assert_eq!(r.start, start());
    assert_eq!((r.window_minutes, r.windows, r.threads), (10, 12, 3));
    assert!(r.oracle);
    assert_eq!((r.echo_dbz, r.clear_dbz), (15.0, 5.0));
    assert!(gc::parse_request(&args(&["--grid", "g", "--frames", "f", "--start", "x"])).is_err());
    assert!(gc::parse_request(&args(&[
        "--grid", "g", "--frames", "f", "--start", "2026-10-01T18:00Z", "--window-minutes", "0",
        "--windows", "1", "--out", "o"
    ]))
    .is_err());
}

#[test]
fn a_library_window_after_the_start_is_an_oracle() {
    // The library window ends 10 minutes after the start: its column shapes
    // carry observations a forecast issued at the start could not have had.
    let dir = scratch("library-future");
    let grid = write_grid(&dir, IDENTITY);
    let window = write_window(&dir, IDENTITY, 800);
    let receipt_path = window.with_extension("json");
    let mut receipt = read_json(&receipt_path);
    receipt["window"]["end"] = json!("2026-10-01T18:10:00Z");
    std::fs::write(&receipt_path, serde_json::to_string_pretty(&receipt).unwrap()).unwrap();
    let frames = write_frames(&dir, &stamps(2), true, "nowcast", 0.0, false);
    let mut r = request(&dir, grid.clone(), frames.clone(), Some(window.clone()), 2, 2, "out");
    let err = error_of(&r);
    assert!(err.contains("after the start"), "{err}");
    assert!(!r.out.join(gc::window_stamp(stamps(1)[0])).exists());
    // With --oracle it runs, and every window says oracle.
    r.oracle = true;
    gc::grid_composite(&r).unwrap();
    let written = read_json(&r.out.join(gc::window_stamp(stamps(1)[0])).join("ref.json"));
    assert_eq!(written["source"]["lead_class"], "oracle");
    // That library, reused as a --profile-table, carries its window end and
    // is refused for a forecast the same way.
    let mut reuse = request(&dir, grid.clone(), frames.clone(), None, 2, 2, "reuse");
    reuse.profile_table = Some(r.out.join("profile-library.json"));
    assert!(error_of(&reuse).contains("after the start"));

    // A library window that states no end cannot be shown causal.
    receipt.as_object_mut().unwrap().remove("window");
    std::fs::write(&receipt_path, serde_json::to_string_pretty(&receipt).unwrap()).unwrap();
    let r = request(&dir, grid, frames, Some(window), 2, 2, "no-end");
    assert!(error_of(&r).contains("states no window end"));
}
