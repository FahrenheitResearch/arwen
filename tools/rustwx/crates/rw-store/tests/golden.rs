//! Golden-fixture tests that pin the rw-store v1 on-disk format FOREVER.
//!
//! The committed fixture files in `tests/golden/v1/` are the ground truth.
//! Any change to what the writer emits — byte layout, encoding, JSON schema —
//! will make `golden_v1_bytes_are_stable` fail. Any change to how the reader
//! decodes those bytes will make `golden_v1_reader_values_match_expected` fail.
//!
//! Fixture definition (all literal — never derived from the current writer):
//!   model="golden", run="20260101_00z", forecast_hour=0,
//!   writer_build="golden-v1", written_unix=1_770_000_000
//!   Grid: ny=300, nx=20  (2 tile-Y rows, 1 tile-X col; 19×2 col-chunks)
//!   lat[gy*nx+gx] = 30.0 + 0.01*(gy as f32)
//!   lon[gy*nx+gx] = -100.0 + 0.05*(gx as f32)
//!
//! Variables:
//!   t2m      surface2d "K"  selector {"var":"TMP","level":"2 m above ground"}
//!            v = 280.0 + sin(0.1*gx)*5.0 + 0.02*gy
//!   mask_demo surface2d "1" selector {"var":"MASK"}
//!            v = same formula, but NaN for all gy >= 256 and NaN for gx==3 && gy<10
//!   const_demo surface2d "Pa" selector {"var":"CONST"}
//!            v = 101325.0 everywhere (CONSTANT tiles)
//!   temp_iso  pressure3d "K" levels [850, 700, 500]
//!            selector_template {"var":"TMP","level":"{level} mb"}
//!            v = 270.0 - level_idx*10.0 + cos(0.05*gx) + 0.01*gy
//!            NaN at full column (gx==5, gy==5) across all 3 levels

use std::fs;
use std::path::{Path, PathBuf};

use rustwx_core::{GridShape, LatLonGrid};
use rw_store::grid::GridFile;
use rw_store::ingest::{HourIngestWriter, PressureVolumeInput};
use rw_store::reader::HourReader;
use rw_store::validate::{ValidateDepth, validate_hour_file, validate_run_dir};

// ---------------------------------------------------------------------------
// Fixture constants — the single source of truth for all three tests
// ---------------------------------------------------------------------------

const MODEL: &str = "golden";
const RUN: &str = "20260101_00z";
const FORECAST_HOUR: u16 = 0;
const WRITER_BUILD: &str = "golden-v1";
const WRITTEN_UNIX: u64 = 1_770_000_000;

const NX: usize = 20;
const NY: usize = 300;

/// Build the canonical lat/lon grid for the golden fixture.
fn golden_grid() -> LatLonGrid {
    let mut lat = Vec::with_capacity(NX * NY);
    let mut lon = Vec::with_capacity(NX * NY);
    for gy in 0..NY {
        for gx in 0..NX {
            lat.push(30.0_f32 + 0.01 * gy as f32);
            lon.push(-100.0_f32 + 0.05 * gx as f32);
        }
    }
    LatLonGrid::new(GridShape::new(NX, NY).unwrap(), lat, lon).unwrap()
}

/// t2m values: 280.0 + sin(0.1*gx)*5.0 + 0.02*gy
fn t2m_values() -> Vec<f32> {
    (0..NY)
        .flat_map(|gy| {
            (0..NX).map(move |gx| 280.0_f32 + (0.1_f32 * gx as f32).sin() * 5.0 + 0.02 * gy as f32)
        })
        .collect()
}

/// mask_demo values: same formula as t2m, but NaN for gy>=256 and for gx==3&&gy<10
fn mask_demo_values() -> Vec<f32> {
    let mut values = t2m_values();
    for gy in 0..NY {
        for gx in 0..NX {
            let idx = gy * NX + gx;
            if gy >= 256 || (gx == 3 && gy < 10) {
                values[idx] = f32::NAN;
            }
        }
    }
    values
}

/// const_demo values: all exactly 101325.0
fn const_demo_values() -> Vec<f32> {
    vec![101325.0_f32; NX * NY]
}

/// Levels for temp_iso in descending order (as the writer sorts them).
const TEMP_ISO_LEVELS: [u16; 3] = [850, 700, 500];

/// temp_iso value for a specific grid point and level index (0=850, 1=700, 2=500
/// after descending sort: 850>700>500 so level_idx 0 => 850 hPa).
fn temp_iso_value(gx: usize, gy: usize, level_idx: usize) -> f32 {
    270.0_f32 - level_idx as f32 * 10.0 + (0.05_f32 * gx as f32).cos() + 0.01 * gy as f32
}

/// One level plane for temp_iso at the given sorted level_idx.
/// NaN the full column at (gx==5, gy==5) across all levels.
fn temp_iso_plane(level_idx: usize) -> Vec<f32> {
    (0..NY)
        .flat_map(|gy| {
            (0..NX).map(move |gx| {
                if gx == 5 && gy == 5 {
                    f32::NAN
                } else {
                    temp_iso_value(gx, gy, level_idx)
                }
            })
        })
        .collect()
}

/// Write the golden fixture into `store_root` and return the run dir path.
/// This is the single authoritative builder used by both regen and the writer-pin test.
fn build_golden_store(store_root: &Path) -> PathBuf {
    let grid = golden_grid();

    let t2m = t2m_values();
    let mask = mask_demo_values();
    let const_ = const_demo_values();

    // Build temp_iso planes (level_idx 0 => 850 hPa (highest pressure, first in
    // descending sort), 1 => 700 hPa, 2 => 500 hPa).
    let plane0 = temp_iso_plane(0); // 850 hPa
    let plane1 = temp_iso_plane(1); // 700 hPa
    let plane2 = temp_iso_plane(2); // 500 hPa

    let volume = PressureVolumeInput {
        name: "temp_iso",
        units: "K",
        selector_template: serde_json::json!({ "var": "TMP", "level": "{level} mb" }),
        levels: vec![
            (TEMP_ISO_LEVELS[0], plane0.as_slice()),
            (TEMP_ISO_LEVELS[1], plane1.as_slice()),
            (TEMP_ISO_LEVELS[2], plane2.as_slice()),
        ],
    };

    let mut writer = HourIngestWriter::begin(
        store_root,
        MODEL,
        RUN,
        FORECAST_HOUR,
        &grid,
        None,
        WRITER_BUILD,
    )
    .expect("HourIngestWriter::begin");

    writer
        .add_field_2d(
            "t2m",
            "K",
            serde_json::json!({ "var": "TMP", "level": "2 m above ground" }),
            &t2m,
        )
        .expect("add t2m");

    writer
        .add_field_2d(
            "mask_demo",
            "1",
            serde_json::json!({ "var": "MASK" }),
            &mask,
        )
        .expect("add mask_demo");

    writer
        .add_field_2d(
            "const_demo",
            "Pa",
            serde_json::json!({ "var": "CONST" }),
            &const_,
        )
        .expect("add const_demo");

    writer
        .add_volume(
            "temp_iso",
            "K",
            serde_json::json!({ "var": "TMP", "level": "{level} mb" }),
            &volume.levels,
        )
        .expect("add temp_iso volume");

    writer.finish(WRITTEN_UNIX).expect("finish");

    store_root.join(MODEL).join(RUN)
}

/// Path to the committed golden fixture directory.
fn committed_dir() -> PathBuf {
    PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden/v1"))
}

/// Unique temp dir for a test run.
fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("rw-store-golden-{}-{}", std::process::id(), name));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

// ---------------------------------------------------------------------------
// Test 1: Writer pin — byte equality between rebuilt and committed fixtures
// ---------------------------------------------------------------------------

#[test]
fn golden_v1_bytes_are_stable() {
    let tmp = temp_dir("writer-pin");
    let store_root = tmp.join("store");
    let rebuilt_run_dir = build_golden_store(&store_root);

    let committed = committed_dir();

    // f000.rws: byte-exact
    let rebuilt_rws = fs::read(rebuilt_run_dir.join("f000.rws")).expect("rebuilt f000.rws");
    let committed_rws = fs::read(committed.join("f000.rws")).expect("committed f000.rws");
    assert_eq!(
        rebuilt_rws.len(),
        committed_rws.len(),
        "f000.rws length changed: committed={} bytes, rebuilt={} bytes — \
         FORMAT CHANGE requiring a version bump discussion",
        committed_rws.len(),
        rebuilt_rws.len()
    );
    if rebuilt_rws != committed_rws {
        let offset = rebuilt_rws
            .iter()
            .zip(committed_rws.iter())
            .position(|(a, b)| a != b)
            .unwrap_or(0);
        panic!(
            "f000.rws first divergence at byte offset {offset} — \
             FORMAT CHANGE requiring a version bump discussion"
        );
    }

    // grid.rwg: byte-exact
    let rebuilt_rwg = fs::read(rebuilt_run_dir.join("grid.rwg")).expect("rebuilt grid.rwg");
    let committed_rwg = fs::read(committed.join("grid.rwg")).expect("committed grid.rwg");
    assert_eq!(
        rebuilt_rwg.len(),
        committed_rwg.len(),
        "grid.rwg length changed: committed={} bytes, rebuilt={} bytes — \
         FORMAT CHANGE requiring a version bump discussion",
        committed_rwg.len(),
        rebuilt_rwg.len()
    );
    if rebuilt_rwg != committed_rwg {
        let offset = rebuilt_rwg
            .iter()
            .zip(committed_rwg.iter())
            .position(|(a, b)| a != b)
            .unwrap_or(0);
        panic!(
            "grid.rwg first divergence at byte offset {offset} — \
             FORMAT CHANGE requiring a version bump discussion"
        );
    }

    // run.json: parsed-value equality, excluding encode_ms (a wall-clock timing
    // that varies per run and is intentionally not pinned).
    let rebuilt_json: serde_json::Value = serde_json::from_slice(
        &fs::read(rebuilt_run_dir.join("run.json")).expect("rebuilt run.json"),
    )
    .expect("parse rebuilt run.json");
    let committed_json: serde_json::Value =
        serde_json::from_slice(&fs::read(committed.join("run.json")).expect("committed run.json"))
            .expect("parse committed run.json");

    fn strip_encode_ms(v: &mut serde_json::Value) {
        if let Some(hours) = v.get_mut("hours") {
            if let Some(obj) = hours.as_object_mut() {
                for hour_entry in obj.values_mut() {
                    if let Some(entry_obj) = hour_entry.as_object_mut() {
                        entry_obj.remove("encode_ms");
                    }
                }
            }
        }
    }
    let mut rebuilt = rebuilt_json.clone();
    let mut committed_stripped = committed_json.clone();
    strip_encode_ms(&mut rebuilt);
    strip_encode_ms(&mut committed_stripped);
    assert_eq!(
        rebuilt, committed_stripped,
        "run.json value changed — this is a FORMAT CHANGE"
    );

    let _ = fs::remove_dir_all(&tmp);
}

// ---------------------------------------------------------------------------
// Test 2: Reader pin — committed fixtures read back exactly as expected
// ---------------------------------------------------------------------------

#[test]
fn golden_v1_reader_values_match_expected() {
    let committed = committed_dir();
    let hour_path = committed.join("f000.rws");
    let grid_path = committed.join("grid.rwg");
    let expected_path = committed.join("expected.json");

    let reader = HourReader::open(&hour_path).expect("HourReader::open committed f000.rws");
    let grid_file = GridFile::open(&grid_path).expect("GridFile::open committed grid.rwg");
    let expected: serde_json::Value =
        serde_json::from_slice(&fs::read(&expected_path).expect("read expected.json"))
            .expect("parse expected.json");

    // ---- grid_hash ----
    let expected_grid_hash = expected["grid_hash"].as_str().expect("expected.grid_hash");
    assert_eq!(grid_file.hash, expected_grid_hash, "grid_hash mismatch");
    assert_eq!(
        reader.meta().grid_hash,
        expected_grid_hash,
        "hour file grid_hash mismatch"
    );

    // ---- t2m spot checks (2D reads are lossless, so bit-exact) ----
    let t2m_full = reader.read_full_2d("t2m").expect("read t2m full_2d");
    let expected_t2m = &expected["t2m_spot"];

    // flat_idx 4105 => gx=5, gy=205;  flat_idx 5999 => gx=19, gy=299
    let spot_indices = [0usize, 4105usize, 5999usize];
    for (i, &flat_idx) in spot_indices.iter().enumerate() {
        let got = t2m_full[flat_idx];
        let expected_val = expected_t2m[i].as_f64().expect("t2m spot value") as f32;
        assert_eq!(
            got.to_bits(),
            expected_val.to_bits(),
            "t2m spot check at flat index {flat_idx}: got {got}, expected {expected_val}"
        );
    }

    // ---- mask_demo NaN positions ----
    let mask_full = reader
        .read_full_2d("mask_demo")
        .expect("read mask_demo full_2d");
    // Flat index 3: gx=3, gy=0 (which satisfies gx==3 && gy<10) -> NaN
    assert!(
        mask_full[3].is_nan(),
        "mask_demo[3] (gx=3,gy=0) must be NaN"
    );
    // Flat index 256*20 = 5120: gy=256 -> all NaN
    assert!(
        mask_full[256 * NX].is_nan(),
        "mask_demo[256*NX] (gy=256) must be NaN"
    );

    // ---- const_demo: CONSTANT-tile reader path — all values must be exactly 101325.0 ----
    let const_full = reader
        .read_full_2d("const_demo")
        .expect("read const_demo full_2d");
    assert_eq!(
        const_full.len(),
        NX * NY,
        "const_demo read_full_2d length mismatch"
    );
    assert!(
        const_full.iter().all(|&v| v == 101325.0_f32),
        "const_demo read_full_2d: all values must be 101325.0 (CONSTANT-tile reader path)"
    );

    // ---- temp_iso profile at (fx=5.5, fy=10.5) ----
    let profile = reader
        .read_profile_3d("temp_iso", 5.5, 10.5)
        .expect("read temp_iso profile");
    assert_eq!(profile.len(), 3, "temp_iso profile must have 3 levels");
    let expected_profile = &expected["temp_iso_profile_5p5_10p5"];
    for (k, (got, expected_entry)) in profile
        .iter()
        .zip(expected_profile.as_array().unwrap().iter())
        .enumerate()
    {
        let expected_val = expected_entry.as_f64().expect("profile value") as f32;
        let rel_err = (got - expected_val).abs() / expected_val.abs().max(1e-6);
        assert!(
            rel_err < 1e-3,
            "temp_iso profile[{k}]: got {got}, expected {expected_val}, rel_err {rel_err}"
        );
    }

    // ---- t2m window checksum ----
    let window = reader
        .read_window_2d("t2m", 2, 250, 10, 270)
        .expect("read t2m window");
    let checksum: f64 = window
        .values
        .iter()
        .filter(|v| v.is_finite())
        .map(|&v| v as f64)
        .sum();
    let checksum_str = format!("{:.6}", checksum);
    let expected_checksum = expected["t2m_window_checksum"]
        .as_str()
        .expect("window checksum");
    assert_eq!(
        checksum_str, expected_checksum,
        "t2m window checksum mismatch"
    );

    // ---- validate both hour file and run dir (Deep) ----
    let hour_report =
        validate_hour_file(&hour_path, ValidateDepth::Deep).expect("validate_hour_file I/O");
    assert!(
        hour_report.is_ok(),
        "validate_hour_file(Deep) must pass on committed fixture; errors: {:?}",
        hour_report.errors
    );

    let run_report =
        validate_run_dir(&committed, ValidateDepth::Deep).expect("validate_run_dir I/O");
    assert!(
        run_report.is_ok(),
        "validate_run_dir(Deep) must pass on committed fixture; errors: {:?}",
        run_report.errors
    );
}

// ---------------------------------------------------------------------------
// Test 3: Regen (ignored) — writes the four fixture files
// ---------------------------------------------------------------------------

#[test]
#[ignore = "regen overwrites committed fixtures — only run deliberately"]
fn regen_golden_v1() {
    eprintln!();
    eprintln!("==========================================================================");
    eprintln!("WARNING: regen_golden_v1 is about to OVERWRITE committed fixture files.");
    eprintln!("Committing these changes constitutes a FORMAT CHANGE and REQUIRES a");
    eprintln!("version bump discussion before merging.");
    eprintln!("==========================================================================");
    eprintln!();

    // First: verify the freshly-built store is internally consistent —
    // a regen must not pin garbage.
    let tmp = temp_dir("regen-verify");
    let store_root = tmp.join("store");
    let built_run_dir = build_golden_store(&store_root);

    let reader = HourReader::open(&built_run_dir.join("f000.rws")).expect("open rebuilt f000.rws");
    let grid_file = GridFile::open(&built_run_dir.join("grid.rwg")).expect("open rebuilt grid.rwg");

    // Verify t2m reads back bit-exact vs formula.
    let t2m_full = reader.read_full_2d("t2m").expect("read t2m");
    let t2m_expected = t2m_values();
    for (i, (got, want)) in t2m_full.iter().zip(t2m_expected.iter()).enumerate() {
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "t2m regen self-check: index {i} mismatch (got {got}, want {want})"
        );
    }

    // Verify mask_demo reads back bit-exact vs formula.
    let mask_full = reader.read_full_2d("mask_demo").expect("read mask_demo");
    let mask_expected = mask_demo_values();
    for (i, (got, want)) in mask_full.iter().zip(mask_expected.iter()).enumerate() {
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "mask_demo regen self-check: index {i} mismatch (got {got}, want {want})"
        );
    }

    // Verify temp_iso profile reads within quantization tolerance.
    let quant_bound = {
        let vmin = temp_iso_value(0, 0, 2); // level_idx=2 (500 hPa), gx=0, gy=0
        let vmax = temp_iso_value(NX - 1, NY - 1, 0); // level_idx=0 (850 hPa)
        (vmax - vmin) / (2.0 * 32767.0) + 1e-5
    };
    // Sample a few column reads
    for &(gx, gy) in &[(0usize, 0usize), (10, 150), (19, 299)] {
        let column = reader
            .read_column_3d("temp_iso", gx, gy)
            .expect("read column");
        assert_eq!(column.len(), 3, "column length must be 3 levels");
        for (k, &val) in column.iter().enumerate() {
            let want = temp_iso_value(gx, gy, k);
            let err = (val - want).abs();
            assert!(
                err <= quant_bound,
                "temp_iso regen self-check column ({gx},{gy}) level_idx {k}: \
                 got {val}, want {want}, err {err}, bound {quant_bound}"
            );
        }
    }

    // ---- Compute expected.json values ----

    // grid_hash
    let grid_hash = grid_file.hash.clone();

    // t2m spot values at flat indices 0, 4105 (gx=5,gy=205), 5999 (gx=19,gy=299)
    let t2m_spots: Vec<f32> = [0usize, 4105, 5999].iter().map(|&i| t2m_full[i]).collect();

    // temp_iso profile at (fx=5.5, fy=10.5)
    let profile = reader
        .read_profile_3d("temp_iso", 5.5, 10.5)
        .expect("read profile for regen");

    // t2m window checksum
    let window = reader
        .read_window_2d("t2m", 2, 250, 10, 270)
        .expect("read window for regen");
    let checksum: f64 = window
        .values
        .iter()
        .filter(|v| v.is_finite())
        .map(|&v| v as f64)
        .sum();
    let checksum_str = format!("{:.6}", checksum);

    let expected_json = serde_json::json!({
        "grid_hash": grid_hash,
        "t2m_spot": t2m_spots.iter().map(|&v| v as f64).collect::<Vec<f64>>(),
        "t2m_window_checksum": checksum_str,
        "temp_iso_profile_5p5_10p5": profile.iter().map(|&v| v as f64).collect::<Vec<f64>>(),
    });

    // ---- Write the fixture files ----
    let fixture_dir = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden/v1"));
    fs::create_dir_all(&fixture_dir).expect("create fixture dir");

    // Copy f000.rws
    fs::copy(built_run_dir.join("f000.rws"), fixture_dir.join("f000.rws")).expect("copy f000.rws");

    // Copy grid.rwg
    fs::copy(built_run_dir.join("grid.rwg"), fixture_dir.join("grid.rwg")).expect("copy grid.rwg");

    // Copy run.json
    fs::copy(built_run_dir.join("run.json"), fixture_dir.join("run.json")).expect("copy run.json");

    // Write expected.json
    let expected_bytes =
        serde_json::to_vec_pretty(&expected_json).expect("serialize expected.json");
    fs::write(fixture_dir.join("expected.json"), &expected_bytes).expect("write expected.json");

    eprintln!("Fixture files written to: {}", fixture_dir.display());
    eprintln!(
        "  f000.rws  ({} bytes)",
        fs::metadata(fixture_dir.join("f000.rws")).unwrap().len()
    );
    eprintln!(
        "  grid.rwg  ({} bytes)",
        fs::metadata(fixture_dir.join("grid.rwg")).unwrap().len()
    );
    eprintln!(
        "  run.json  ({} bytes)",
        fs::metadata(fixture_dir.join("run.json")).unwrap().len()
    );
    eprintln!(
        "  expected.json  ({} bytes)",
        fs::metadata(fixture_dir.join("expected.json"))
            .unwrap()
            .len()
    );

    let _ = fs::remove_dir_all(&tmp);
}
