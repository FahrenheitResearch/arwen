//! Task 6 integration tests: 3D pressure volumes — column-chunked writes,
//! single-column reads, and bilinear profile reads.
//!
//! Synthetic analytic volume on a 100 x 80 grid: 7 x-chunks (last 4 cols
//! wide) x 5 y-chunks (exact) = 35 column chunks. Levels are
//! [1000, 850, 700, 500, 300, 200, 100] (L = 7), and
//! `v(x, y, k) = 0.1*x + 0.2*y - 1.5*k`.

use std::fs;
use std::path::{Path, PathBuf};

use rw_store::error::RwStoreError;
use rw_store::format::{
    CODEC_3D, COL_X, COL_Y, FLAG_CONSTANT, FLAG_EMPTY, INDEX_RECORD_LEN, KIND_COLUMN3D,
};
use rw_store::header::RwsHeader;
use rw_store::index::ChunkRecord;
use rw_store::reader::HourReader;
use rw_store::writer::HourWriter;

const NX: usize = 100; // 7 x-chunks of 16: six full + one 4 wide
const NY: usize = 80; // 5 y-chunks of 16, exact
const LEVELS: [u16; 7] = [1000, 850, 700, 500, 300, 200, 100];
const L: usize = LEVELS.len();
const CHUNKS_X: usize = 7;
const CHUNKS_Y: usize = 5;
const CHUNK_COUNT: usize = CHUNKS_X * CHUNKS_Y; // 35

/// Conservative quantization error bound: whole-volume value range over
/// 2 * Q_MAX (every per-chunk scale is <= this), plus float-noise epsilon.
/// v ranges from v(0,0,6) = -9.0 to v(99,79,0) = 25.7 -> range 34.7.
fn quant_bound() -> f32 {
    let vmin = analytic(0, 0, L - 1);
    let vmax = analytic(NX - 1, NY - 1, 0);
    (vmax - vmin) / (2.0 * 32767.0) + 1e-5
}

fn analytic(x: usize, y: usize, k: usize) -> f32 {
    0.1 * x as f32 + 0.2 * y as f32 - 1.5 * k as f32
}

/// Build the analytic volume as L row-major full-grid planes.
fn analytic_planes() -> Vec<Vec<f32>> {
    (0..L)
        .map(|k| {
            (0..NY)
                .flat_map(|y| (0..NX).map(move |x| analytic(x, y, k)))
                .collect()
        })
        .collect()
}

fn plane_refs(planes: &[Vec<f32>]) -> Vec<&[f32]> {
    planes.iter().map(|p| p.as_slice()).collect()
}

fn test_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "rw-store-pressure3d-{}-{}",
        std::process::id(),
        name
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn new_writer() -> HourWriter {
    HourWriter::new(
        "hrrr",
        "20260609_12z",
        6,
        NX,
        NY,
        "gridhash-test",
        "test-build",
    )
}

fn write_volume(path: &Path, planes: &[Vec<f32>]) {
    let mut writer = new_writer();
    writer
        .add_pressure3d(
            "temperature",
            "K",
            serde_json::json!({"grib_short_name": "TMP", "level_type": "isobaric"}),
            &LEVELS,
            &plane_refs(planes),
        )
        .unwrap();
    writer.finish(path).unwrap();
}

/// Parse the on-disk chunk index into records.
fn parse_records(bytes: &[u8]) -> (RwsHeader, Vec<ChunkRecord>) {
    let header = RwsHeader::parse(bytes).unwrap();
    let records = (0..header.index_count as usize)
        .map(|i| {
            let start = header.index_offset as usize + i * INDEX_RECORD_LEN;
            ChunkRecord::unpack(&bytes[start..start + INDEX_RECORD_LEN]).unwrap()
        })
        .collect();
    (header, records)
}

#[test]
fn volume_round_trips_within_quantization_bound() {
    let dir = test_dir("round-trip");
    let path = dir.join("hour.rws");
    let planes = analytic_planes();
    write_volume(&path, &planes);

    let reader = HourReader::open(&path).unwrap();
    let var = reader.variable("temperature").expect("variable present");
    assert_eq!(var.kind, "pressure3d");
    assert_eq!(var.codec, CODEC_3D);
    assert_eq!(var.levels_hpa, LEVELS.to_vec());

    let bound = quant_bound();
    // Grid corners, center, columns inside the 4-wide x-edge chunk (cx = 6),
    // and columns straddling interior chunk boundaries.
    let sample: &[(usize, usize)] = &[
        (0, 0),
        (NX - 1, 0),
        (0, NY - 1),
        (NX - 1, NY - 1),
        (50, 40),
        (96, 7),  // first column of edge chunk cx = 6
        (99, 79), // last column of edge chunk
        (15, 15), // last column of chunk (0,0)
        (16, 16), // first column of chunk (1,1)
    ];
    for &(ix, iy) in sample {
        let column = reader.read_column_3d("temperature", ix, iy).unwrap();
        assert_eq!(column.len(), L, "column ({ix},{iy}) must have L levels");
        for (k, value) in column.iter().enumerate() {
            let expected = analytic(ix, iy, k);
            assert!(
                (value - expected).abs() <= bound,
                "column ({ix},{iy}) level {k}: got {value}, expected {expected}, bound {bound}"
            );
        }
    }

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn column_read_matches_manual_extraction() {
    let dir = test_dir("mixed-kind");
    let path = dir.join("hour.rws");
    let planes = analytic_planes();
    let surface: Vec<f32> = (0..NY)
        .flat_map(|y| (0..NX).map(move |x| 2.0 * x as f32 + 3.0 * y as f32))
        .collect();

    let mut writer = new_writer();
    writer
        .add_surface2d(
            "temp_2m",
            "K",
            serde_json::json!({"grib_short_name": "TMP"}),
            &surface,
        )
        .unwrap();
    writer
        .add_pressure3d(
            "temperature",
            "K",
            serde_json::Value::Null,
            &LEVELS,
            &plane_refs(&planes),
        )
        .unwrap();
    writer.finish(&path).unwrap();

    let reader = HourReader::open(&path).unwrap();
    let bound = quant_bound();
    // 3D columns must match plucking the same (ix, iy) from the input planes.
    for &(ix, iy) in &[(5usize, 5usize), (16, 0), (97, 77), (0, 79)] {
        let column = reader.read_column_3d("temperature", ix, iy).unwrap();
        assert_eq!(column.len(), L);
        for k in 0..L {
            let expected = planes[k][iy * NX + ix];
            assert!(
                (column[k] - expected).abs() <= bound,
                "column ({ix},{iy}) level {k}: got {}, plucked {expected}",
                column[k]
            );
        }
    }

    // 2D reads still work in the mixed-kind file, bit-exact.
    let full = reader.read_full_2d("temp_2m").unwrap();
    assert_eq!(full.len(), NX * NY);
    for (i, (a, e)) in full.iter().zip(surface.iter()).enumerate() {
        assert_eq!(a.to_bits(), e.to_bits(), "2D mismatch at {i}");
    }
    let window = reader.read_window_2d("temp_2m", 10, 10, 20, 20).unwrap();
    assert_eq!((window.nx, window.ny), (10, 10));
    assert_eq!(window.values[0].to_bits(), surface[10 * NX + 10].to_bits());

    // Raw layout: 1 tile record (grid < 256 in both axes) + 35 column chunks.
    let bytes = fs::read(&path).unwrap();
    let (header, records) = parse_records(&bytes);
    assert_eq!(header.index_count as usize, 1 + CHUNK_COUNT);
    assert_eq!(
        records.iter().filter(|r| r.kind == KIND_COLUMN3D).count(),
        CHUNK_COUNT,
        "3D var must have {CHUNKS_X} x {CHUNKS_Y} = {CHUNK_COUNT} column chunks"
    );

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn profile_bilinear_matches_analytic() {
    let dir = test_dir("profile");
    let path = dir.join("hour.rws");
    let planes = analytic_planes();
    write_volume(&path, &planes);
    let reader = HourReader::open(&path).unwrap();
    let bound = quant_bound();

    // Bilinear interpolation of a field linear in x and y is exact, so the
    // profile must match the analytic value within quantization error.
    for &(fx, fy) in &[(10.5f64, 20.25f64), (0.0, 0.0), (99.0, 79.0)] {
        let profile = reader.read_profile_3d("temperature", fx, fy).unwrap();
        assert_eq!(profile.len(), L, "profile ({fx},{fy}) must have L levels");
        for (k, value) in profile.iter().enumerate() {
            let expected = 0.1 * fx as f32 + 0.2 * fy as f32 - 1.5 * k as f32;
            assert!(
                (value - expected).abs() <= bound,
                "profile ({fx},{fy}) level {k}: got {value}, expected {expected}, bound {bound}"
            );
        }
    }

    // Out-of-range coordinates clamp to the grid edge.
    let clamped = reader.read_profile_3d("temperature", -3.0, 200.0).unwrap();
    let edge = reader.read_profile_3d("temperature", 0.0, 79.0).unwrap();
    for k in 0..L {
        assert_eq!(
            clamped[k].to_bits(),
            edge[k].to_bits(),
            "clamped profile must equal edge profile at level {k}"
        );
    }

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn profile_handles_nan_corners() {
    let dir = test_dir("nan-corners");
    let path = dir.join("hour.rws");
    let mut planes = analytic_planes();
    // Poison one corner column entirely: (10, 20) at all levels.
    for plane in &mut planes {
        plane[20 * NX + 10] = f32::NAN;
    }
    // Poison all four columns around (40.5, 40.5).
    for plane in &mut planes {
        for &(x, y) in &[(40usize, 40usize), (41, 40), (40, 41), (41, 41)] {
            plane[y * NX + x] = f32::NAN;
        }
    }
    write_volume(&path, &planes);
    let reader = HourReader::open(&path).unwrap();
    let bound = quant_bound();

    // (10.5, 20.5): corner (10,20) is NaN, the other three corners are
    // finite with equal weights (0.25 each), so the renormalized result is
    // the plain mean of the three finite corners.
    let profile = reader.read_profile_3d("temperature", 10.5, 20.5).unwrap();
    assert_eq!(profile.len(), L);
    for (k, value) in profile.iter().enumerate() {
        let expected = (analytic(11, 20, k) + analytic(10, 21, k) + analytic(11, 21, k)) / 3.0;
        assert!(value.is_finite(), "level {k} must be finite, got {value}");
        assert!(
            (value - expected).abs() <= bound,
            "level {k}: got {value}, expected 3-corner mean {expected}, bound {bound}"
        );
        // Still close to the analytic point value (renormalization shifts it
        // by 0.05 here).
        let analytic_point = 0.1f32 * 10.5 + 0.2 * 20.5 - 1.5 * k as f32;
        assert!(
            (value - analytic_point).abs() <= 0.1 + bound,
            "level {k}: {value} not close to analytic {analytic_point}"
        );
    }

    // All four corners poisoned -> all-NaN profile.
    let poisoned = reader.read_profile_3d("temperature", 40.5, 40.5).unwrap();
    assert_eq!(poisoned.len(), L);
    assert!(
        poisoned.iter().all(|v| v.is_nan()),
        "profile among 4 poisoned columns must be all NaN, got {poisoned:?}"
    );

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn validation_errors() {
    let planes = analytic_planes();
    let refs = plane_refs(&planes);

    let format_err = |result: Result<u16, RwStoreError>, context: &str| {
        let err = result.unwrap_err();
        assert!(
            matches!(err, RwStoreError::Format(_)),
            "{context}: expected Format error, got {err:?}"
        );
    };

    // Empty levels.
    let mut writer = new_writer();
    format_err(
        writer.add_pressure3d("t", "K", serde_json::Value::Null, &[], &[]),
        "empty levels",
    );

    // Ascending levels.
    let mut writer = new_writer();
    let ascending: Vec<u16> = {
        let mut l = LEVELS.to_vec();
        l.reverse();
        l
    };
    format_err(
        writer.add_pressure3d("t", "K", serde_json::Value::Null, &ascending, &refs),
        "ascending levels",
    );

    // Non-strictly-descending (duplicate) levels.
    let mut writer = new_writer();
    let duplicated: Vec<u16> = vec![1000, 850, 850, 500, 300, 200, 100];
    format_err(
        writer.add_pressure3d("t", "K", serde_json::Value::Null, &duplicated, &refs),
        "duplicate levels",
    );

    // Plane count mismatch.
    let mut writer = new_writer();
    format_err(
        writer.add_pressure3d("t", "K", serde_json::Value::Null, &LEVELS, &refs[..L - 1]),
        "plane count mismatch",
    );

    // Plane length mismatch.
    let mut writer = new_writer();
    let short = vec![0.0f32; NX * NY - 1];
    let mut bad_refs = refs.clone();
    bad_refs[3] = &short;
    format_err(
        writer.add_pressure3d("t", "K", serde_json::Value::Null, &LEVELS, &bad_refs),
        "plane length mismatch",
    );

    // Duplicate name across kinds: 2D var then 3D var with the same name.
    let mut writer = new_writer();
    let surface = vec![1.0f32; NX * NY];
    writer
        .add_surface2d("temp", "K", serde_json::Value::Null, &surface)
        .unwrap();
    format_err(
        writer.add_pressure3d("temp", "K", serde_json::Value::Null, &LEVELS, &refs),
        "duplicate name vs 2D var",
    );
    // And the reverse: 3D var then 2D var with the same name.
    let mut writer = new_writer();
    writer
        .add_pressure3d("temp", "K", serde_json::Value::Null, &LEVELS, &refs)
        .unwrap();
    format_err(
        writer.add_surface2d("temp", "K", serde_json::Value::Null, &surface),
        "duplicate name vs 3D var",
    );

    // Reader-side errors need a real file with both kinds.
    let dir = test_dir("validation");
    let path = dir.join("hour.rws");
    let mut writer = new_writer();
    writer
        .add_surface2d("temp_2m", "K", serde_json::Value::Null, &surface)
        .unwrap();
    writer
        .add_pressure3d("temperature", "K", serde_json::Value::Null, &LEVELS, &refs)
        .unwrap();
    writer.finish(&path).unwrap();
    let reader = HourReader::open(&path).unwrap();

    // read_column_3d on a 2D variable -> Format.
    let err = reader.read_column_3d("temp_2m", 0, 0).unwrap_err();
    assert!(
        matches!(err, RwStoreError::Format(_)),
        "read_column_3d on 2D var: expected Format, got {err:?}"
    );
    // read_profile_3d on a 2D variable -> Format.
    let err = reader.read_profile_3d("temp_2m", 0.0, 0.0).unwrap_err();
    assert!(
        matches!(err, RwStoreError::Format(_)),
        "read_profile_3d on 2D var: expected Format, got {err:?}"
    );
    // Unknown variable -> UnknownVariable.
    let err = reader.read_column_3d("no_such_var", 0, 0).unwrap_err();
    assert!(
        matches!(&err, RwStoreError::UnknownVariable(name) if name == "no_such_var"),
        "expected UnknownVariable, got {err:?}"
    );
    // Out-of-bounds ix / iy -> Format.
    for &(ix, iy) in &[(NX, 0usize), (0, NY), (usize::MAX, 0)] {
        let err = reader.read_column_3d("temperature", ix, iy).unwrap_err();
        assert!(
            matches!(err, RwStoreError::Format(_)),
            "read_column_3d({ix},{iy}): expected Format, got {err:?}"
        );
    }

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn empty_and_constant_3d_chunks() {
    let dir = test_dir("empty-constant");
    let path = dir.join("hour.rws");
    let mut planes = analytic_planes();
    // Chunk (0,0): footprint rows 0..16 x cols 0..16 all-NaN at every level.
    // Chunk (1,1): rows 16..32 x cols 16..32 constant 5.0 at every level.
    for plane in &mut planes {
        for y in 0..COL_Y {
            for x in 0..COL_X {
                plane[y * NX + x] = f32::NAN;
            }
        }
        for y in COL_Y..2 * COL_Y {
            for x in COL_X..2 * COL_X {
                plane[y * NX + x] = 5.0;
            }
        }
    }
    write_volume(&path, &planes);

    // Raw index inspection: EMPTY and CONSTANT chunks carry no payload.
    let bytes = fs::read(&path).unwrap();
    let (header, records) = parse_records(&bytes);
    assert_eq!(header.index_count as usize, CHUNK_COUNT, "35 column chunks");
    let empty = records
        .iter()
        .find(|r| r.kind == KIND_COLUMN3D && r.tile_y == 0 && r.tile_x == 0)
        .expect("record for chunk (0,0)");
    assert_ne!(empty.flags & FLAG_EMPTY, 0, "all-NaN chunk must be EMPTY");
    assert_eq!(empty.len, 0, "EMPTY chunk must have len 0");
    let constant = records
        .iter()
        .find(|r| r.kind == KIND_COLUMN3D && r.tile_y == 1 && r.tile_x == 1)
        .expect("record for chunk (1,1)");
    assert_ne!(
        constant.flags & FLAG_CONSTANT,
        0,
        "uniform chunk must be CONSTANT"
    );
    assert_eq!(constant.len, 0, "CONSTANT chunk must have len 0");
    assert_eq!(constant.center, 5.0);

    let reader = HourReader::open(&path).unwrap();
    // Column inside the EMPTY chunk -> NaN at every level.
    let nan_column = reader.read_column_3d("temperature", 7, 9).unwrap();
    assert_eq!(nan_column.len(), L);
    assert!(
        nan_column.iter().all(|v| v.is_nan()),
        "EMPTY-chunk column must be all NaN, got {nan_column:?}"
    );
    // Column inside the CONSTANT chunk -> 5.0 at every level.
    let const_column = reader.read_column_3d("temperature", 20, 20).unwrap();
    assert_eq!(const_column.len(), L);
    assert!(
        const_column.iter().all(|v| v.to_bits() == 5.0f32.to_bits()),
        "CONSTANT-chunk column must be all 5.0, got {const_column:?}"
    );
    // Sanity: a column outside both special chunks still reads analytic.
    let bound = quant_bound();
    let normal = reader.read_column_3d("temperature", 50, 50).unwrap();
    for (k, value) in normal.iter().enumerate() {
        let expected = analytic(50, 50, k);
        assert!(
            (value - expected).abs() <= bound,
            "level {k}: got {value}, expected {expected}"
        );
    }

    let _ = fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------
// Task 4: read_full_3d
// ---------------------------------------------------------------------------

/// Build a file with:
///   - "temperature" (pressure3d, 3 levels [500, 300, 200])
///   - "temp_2m" (surface2d) — for the wrong-kind error test
///   - chunk (0,0) is EMPTY (all NaN at every level)
///   - chunk (1,1) is CONSTANT 5.0 at every level
///   - all other cells use the analytic formula (index k over the 3 levels)
///
/// Grid is the same NX x NY (100 x 80) as the rest of this file.
fn write_volume_for_full3d(path: &Path) {
    const LEVELS_3: [u16; 3] = [500, 300, 200];
    let l3: usize = LEVELS_3.len();

    // Build planes using the same analytic formula (k = 0..l3).
    let mut planes: Vec<Vec<f32>> = (0..l3)
        .map(|k| {
            (0..NY)
                .flat_map(|y| (0..NX).map(move |x| analytic(x, y, k)))
                .collect()
        })
        .collect();

    // Chunk (0,0): footprint rows 0..16 x cols 0..16 -> all NaN (EMPTY).
    for plane in &mut planes {
        for y in 0..COL_Y {
            for x in 0..COL_X {
                plane[y * NX + x] = f32::NAN;
            }
        }
    }
    // Chunk (1,1): rows 16..32 x cols 16..32 -> constant 5.0 (CONSTANT).
    for plane in &mut planes {
        for y in COL_Y..2 * COL_Y {
            for x in COL_X..2 * COL_X {
                plane[y * NX + x] = 5.0;
            }
        }
    }

    let surface: Vec<f32> = (0..NY)
        .flat_map(|y| (0..NX).map(move |x| 2.0 * x as f32 + 3.0 * y as f32))
        .collect();
    let plane_refs: Vec<&[f32]> = planes.iter().map(|p| p.as_slice()).collect();

    let mut writer = new_writer();
    writer
        .add_surface2d("temp_2m", "K", serde_json::Value::Null, &surface)
        .unwrap();
    writer
        .add_pressure3d(
            "temperature",
            "K",
            serde_json::Value::Null,
            &LEVELS_3,
            &plane_refs,
        )
        .unwrap();
    writer.finish(path).unwrap();
}

#[test]
fn read_full_3d_matches_column_reads() {
    let dir = test_dir("full3d");
    let path = dir.join("hour.rws");
    write_volume_for_full3d(&path);

    let reader = HourReader::open(&path).unwrap();
    let nl = 3usize; // 3 levels in this file

    let full = reader.read_full_3d("temperature").unwrap();
    assert_eq!(
        full.len(),
        nl * NY * NX,
        "full3d length must be levels * ny * nx"
    );

    // Coarse lattice: every 7th x, every 5th y, plus all 4 corners.
    let mut sample_points: Vec<(usize, usize)> = Vec::new();
    let mut iy = 0usize;
    while iy < NY {
        let mut ix = 0usize;
        while ix < NX {
            sample_points.push((ix, iy));
            ix += 7;
        }
        iy += 5;
    }
    // Add all 4 corners explicitly (some may already be in the lattice).
    for &corner in &[(0, 0), (NX - 1, 0), (0, NY - 1), (NX - 1, NY - 1)] {
        sample_points.push(corner);
    }

    for (ix, iy) in sample_points {
        let col = reader.read_column_3d("temperature", ix, iy).unwrap();
        assert_eq!(col.len(), nl);
        for lvl in 0..nl {
            let full_val = full[lvl * NY * NX + iy * NX + ix];
            let col_val = col[lvl];
            // NaN-safe bit-exact comparison.
            assert_eq!(
                full_val.to_bits(),
                col_val.to_bits(),
                "mismatch at ix={ix} iy={iy} lvl={lvl}: \
                 full={full_val} col={col_val}"
            );
        }
    }

    // Unknown variable -> UnknownVariable error.
    let err = reader.read_full_3d("no_such_var").unwrap_err();
    assert!(
        matches!(&err, RwStoreError::UnknownVariable(name) if name == "no_such_var"),
        "expected UnknownVariable, got {err:?}"
    );

    // 2D variable name -> Format error (same class as read_column_3d).
    let err_2d = reader.read_full_3d("temp_2m").unwrap_err();
    assert!(
        matches!(err_2d, RwStoreError::Format(_)),
        "read_full_3d on 2D var: expected Format error, got {err_2d:?}"
    );

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn read_level_3d_matches_full_volume_plane() {
    let dir = test_dir("level3d");
    let path = dir.join("hour.rws");
    write_volume_for_full3d(&path);

    let reader = HourReader::open(&path).unwrap();
    let full = reader.read_full_3d("temperature").unwrap();
    let plane = reader.read_level_3d("temperature", 300).unwrap();
    assert_eq!(plane.len(), NY * NX);
    let expected = &full[NY * NX..2 * NY * NX];
    for (index, (actual, expected)) in plane.iter().zip(expected).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "300 hPa mismatch at flat grid index {index}"
        );
    }

    let missing = reader.read_level_3d("temperature", 850).unwrap_err();
    assert!(matches!(missing, RwStoreError::Meta(_)));
    let wrong_kind = reader.read_level_3d("temp_2m", 300).unwrap_err();
    assert!(matches!(wrong_kind, RwStoreError::Format(_)));

    let _ = fs::remove_dir_all(&dir);
}
