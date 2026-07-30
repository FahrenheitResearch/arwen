//! Task 8 integration tests: the public ingest seam — `SelectedField2D` in,
//! `.rws` + `grid.rwg` + `run.json` out, and `read_field_2d` reconstruction
//! back to a `SelectedField2D`.
//!
//! Synthetic 80 x 60 regular lat/lon grid (Geographic projection), two 2D
//! fields (one with a NaN region) and one 5-level analytic pressure volume
//! `v(x, y, p) = 0.1*x + 0.2*y + 0.01*p`.

use std::fs;
use std::path::{Path, PathBuf};

use rustwx_core::{
    CanonicalField, FieldSelector, GridProjection, GridShape, LatLonGrid, SelectedField2D,
};
use rw_store::RwsExactTime;
use rw_store::error::RwStoreError;
use rw_store::grid::GridFile;
use rw_store::ingest::{
    DerivedFieldInput, HourIngestWriter, PressureVolumeInput, derived_selector,
    derived_selector_slug, read_field_2d, read_grid_2d, write_hour_from_fields,
    write_hour_from_fields_exact, write_hour_from_fields_with_derived,
    write_hour_from_grid_with_derived,
};
use rw_store::lock::RunLock;
use rw_store::reader::HourReader;
use rw_store::run::RwsRunManifest;

const NX: usize = 80;
const NY: usize = 60;
const MODEL: &str = "hrrr";
const RUN: &str = "20260609_12z";
const BUILD: &str = "test-build";
const WRITTEN_UNIX: u64 = 1_780_000_000;
/// Volume levels in canonical (descending) order.
const VLEVELS: [u16; 5] = [1000, 850, 700, 500, 300];

fn test_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("rw-store-ingest-{}-{}", std::process::id(), name));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// Regular 80 x 60 lat/lon grid: lat 35..40.9 by 0.1, lon -100..-92.1 by 0.1.
fn regular_grid() -> LatLonGrid {
    grid_with_dims(NX, NY)
}

fn grid_with_dims(nx: usize, ny: usize) -> LatLonGrid {
    let mut lat = Vec::with_capacity(nx * ny);
    let mut lon = Vec::with_capacity(nx * ny);
    for y in 0..ny {
        for x in 0..nx {
            lat.push((35.0 + 0.1 * y as f64) as f32);
            lon.push((-100.0 + 0.1 * x as f64) as f32);
        }
    }
    LatLonGrid::new(GridShape::new(nx, ny).unwrap(), lat, lon).unwrap()
}

fn temp_selector() -> FieldSelector {
    FieldSelector::height_agl(CanonicalField::Temperature, 2)
}

fn dewpoint_selector() -> FieldSelector {
    FieldSelector::height_agl(CanonicalField::Dewpoint, 2)
}

/// 2m temperature: smooth analytic field, finite everywhere.
fn temp_field() -> SelectedField2D {
    let values: Vec<f32> = (0..NY)
        .flat_map(|y| (0..NX).map(move |x| 280.0 + 0.05 * x as f32 - 0.02 * y as f32))
        .collect();
    SelectedField2D::new(temp_selector(), "K", regular_grid(), values)
        .unwrap()
        .with_projection(GridProjection::Geographic)
}

/// 2m dewpoint: analytic with a NaN region (rows 5..15, cols 10..30).
fn dewpoint_field() -> SelectedField2D {
    let mut values: Vec<f32> = (0..NY)
        .flat_map(|y| (0..NX).map(move |x| 270.0 + 0.03 * x as f32 + 0.01 * y as f32))
        .collect();
    for y in 5..15 {
        for x in 10..30 {
            values[y * NX + x] = f32::NAN;
        }
    }
    SelectedField2D::new(dewpoint_selector(), "K", regular_grid(), values)
        .unwrap()
        .with_projection(GridProjection::Geographic)
}

/// Analytic volume value as a function of the level itself, so level order
/// never matters when building planes.
fn vol_value(x: usize, y: usize, level_hpa: u16) -> f32 {
    0.1 * x as f32 + 0.2 * y as f32 + 0.01 * level_hpa as f32
}

fn vol_plane(level_hpa: u16) -> Vec<f32> {
    (0..NY)
        .flat_map(|y| (0..NX).map(move |x| vol_value(x, y, level_hpa)))
        .collect()
}

/// Conservative quantization error bound: whole-volume value range over
/// 2 * Q_MAX (every per-chunk scale is <= this), plus float-noise epsilon.
fn quant_bound() -> f32 {
    let vmin = vol_value(0, 0, *VLEVELS.iter().min().unwrap());
    let vmax = vol_value(NX - 1, NY - 1, *VLEVELS.iter().max().unwrap());
    (vmax - vmin) / (2.0 * 32767.0) + 1e-5
}

fn volume_input<'a>(levels: &[u16], planes: &'a [Vec<f32>]) -> PressureVolumeInput<'a> {
    PressureVolumeInput {
        name: "temperature",
        units: "K",
        selector_template: serde_json::json!({
            "field": "temperature",
            "vertical": "isobaric"
        }),
        levels: levels
            .iter()
            .zip(planes.iter())
            .map(|(&level, plane)| (level, plane.as_slice()))
            .collect(),
    }
}

fn run_dir(store_root: &Path) -> PathBuf {
    store_root.join(MODEL).join(RUN)
}

fn load_manifest(store_root: &Path) -> RwsRunManifest {
    let bytes = fs::read(run_dir(store_root).join("run.json")).unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

/// NaN-safe bit-exact slice comparison.
fn assert_bits_eq(actual: &[f32], expected: &[f32], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            e.to_bits(),
            "{context}: value mismatch at index {i} (actual {a}, expected {e})"
        );
    }
}

#[test]
fn hour_write_and_read_back_round_trips() {
    let dir = test_dir("round-trip");
    let store_root = dir.join("store");
    let temp = temp_field();
    let dewpoint = dewpoint_field();
    let planes: Vec<Vec<f32>> = VLEVELS.iter().map(|&level| vol_plane(level)).collect();
    let volume = volume_input(&VLEVELS, &planes);

    let written = write_hour_from_fields(
        &store_root,
        MODEL,
        RUN,
        6,
        &[("temp_2m", &temp), ("dewpoint_2m", &dewpoint)],
        &[volume],
        BUILD,
        WRITTEN_UNIX,
    )
    .unwrap();

    // WrittenHour reports every variable, in input order.
    assert_eq!(
        written.vars,
        vec![
            "temp_2m".to_string(),
            "dewpoint_2m".to_string(),
            "temperature".to_string()
        ]
    );
    assert!(written.bytes > 0, "bytes must be the final file size");

    // All three artifacts exist where the layout says they live.
    let run_dir = run_dir(&store_root);
    let hour_path = run_dir.join("f006.rws");
    assert_eq!(
        written.path, hour_path,
        "WrittenHour.path must be the hour file"
    );
    assert!(hour_path.exists(), "f006.rws must exist");
    assert!(run_dir.join("grid.rwg").exists(), "grid.rwg must exist");
    assert!(run_dir.join("run.json").exists(), "run.json must exist");
    assert_eq!(
        written.bytes,
        fs::metadata(&hour_path).unwrap().len(),
        "bytes must match the hour file on disk"
    );

    // run.json registered the hour with the caller's clock and var names.
    let manifest = load_manifest(&store_root);
    assert_eq!(manifest.hours.len(), 1);
    let entry = &manifest.hours[&6];
    assert_eq!(entry.file, "f006.rws");
    assert_eq!(entry.written_unix, WRITTEN_UNIX);
    assert_eq!(entry.encode_ms, written.encode_ms);
    assert_eq!(entry.variables, written.vars);

    // Read back both 2D fields through the reconstruction helper.
    let reader = HourReader::open(&hour_path).unwrap();
    let grid_file = GridFile::open(&run_dir.join("grid.rwg")).unwrap();
    assert_eq!(
        reader.meta().grid_hash,
        grid_file.hash,
        "hour meta must reference the grid file's hash"
    );

    for (name, original) in [("temp_2m", &temp), ("dewpoint_2m", &dewpoint)] {
        let round_tripped = read_field_2d(&reader, &grid_file, name).unwrap();
        assert_eq!(
            round_tripped.selector, original.selector,
            "{name}: selector must round-trip"
        );
        assert_eq!(round_tripped.units, original.units, "{name}: units");
        assert_eq!(
            round_tripped.grid.shape, original.grid.shape,
            "{name}: grid shape"
        );
        assert_bits_eq(
            &round_tripped.grid.lat_deg,
            &original.grid.lat_deg,
            &format!("{name}: grid lat"),
        );
        assert_bits_eq(
            &round_tripped.grid.lon_deg,
            &original.grid.lon_deg,
            &format!("{name}: grid lon"),
        );
        assert_bits_eq(
            &round_tripped.values,
            &original.values,
            &format!("{name}: values"),
        );
        assert_eq!(
            round_tripped.projection,
            Some(GridProjection::Geographic),
            "{name}: projection must come back from the grid file"
        );
    }

    // Volume columns read back within the quantization bound.
    let bound = quant_bound();
    for &(ix, iy) in &[(0usize, 0usize), (NX - 1, NY - 1), (40, 30), (17, 16)] {
        let column = reader.read_column_3d("temperature", ix, iy).unwrap();
        assert_eq!(column.len(), VLEVELS.len());
        for (k, value) in column.iter().enumerate() {
            let expected = vol_value(ix, iy, VLEVELS[k]);
            assert!(
                (value - expected).abs() <= bound,
                "column ({ix},{iy}) level {} hPa: got {value}, expected {expected}, bound {bound}",
                VLEVELS[k]
            );
        }
    }

    // read_field_2d error paths: unknown variable and non-2D variable.
    let err = read_field_2d(&reader, &grid_file, "no_such_var").unwrap_err();
    assert!(
        matches!(&err, RwStoreError::UnknownVariable(name) if name == "no_such_var"),
        "expected UnknownVariable, got {err:?}"
    );
    let err = read_field_2d(&reader, &grid_file, "temperature").unwrap_err();
    assert!(
        matches!(err, RwStoreError::Format(_)),
        "read_field_2d on a 3D variable: expected Format error, got {err:?}"
    );

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn second_hour_reuses_grid_file() {
    let dir = test_dir("grid-reuse");
    let store_root = dir.join("store");
    let temp = temp_field();

    write_hour_from_fields(
        &store_root,
        MODEL,
        RUN,
        6,
        &[("temp_2m", &temp)],
        &[],
        BUILD,
        WRITTEN_UNIX,
    )
    .unwrap();
    let grid_path = run_dir(&store_root).join("grid.rwg");
    let grid_bytes_first = fs::read(&grid_path).unwrap();

    write_hour_from_fields(
        &store_root,
        MODEL,
        RUN,
        12,
        &[("temp_2m", &temp)],
        &[],
        BUILD,
        WRITTEN_UNIX + 600,
    )
    .unwrap();
    let grid_bytes_second = fs::read(&grid_path).unwrap();
    assert_eq!(
        grid_bytes_first, grid_bytes_second,
        "grid.rwg must be written once and reused byte-identically"
    );

    let manifest = load_manifest(&store_root);
    assert_eq!(manifest.hours.len(), 2, "run.json must register both hours");
    assert_eq!(manifest.hours[&6].file, "f006.rws");
    assert_eq!(manifest.hours[&12].file, "f012.rws");

    // Both hour files reference the same grid hash, which is the grid file's.
    let grid_file = GridFile::open(&grid_path).unwrap();
    let meta_6 = HourReader::open(&run_dir(&store_root).join("f006.rws"))
        .unwrap()
        .meta()
        .clone();
    let meta_12 = HourReader::open(&run_dir(&store_root).join("f012.rws"))
        .unwrap()
        .meta()
        .clone();
    assert_eq!(meta_6.grid_hash, grid_file.hash);
    assert_eq!(meta_12.grid_hash, grid_file.hash);
    assert_eq!(manifest.grid_hash, grid_file.hash);

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn mismatched_grid_rejected() {
    let dir = test_dir("grid-mismatch");
    let store_root = dir.join("store");
    let temp = temp_field();

    write_hour_from_fields(
        &store_root,
        MODEL,
        RUN,
        6,
        &[("temp_2m", &temp)],
        &[],
        BUILD,
        WRITTEN_UNIX,
    )
    .unwrap();

    // Second hour arrives on a 70 x 60 grid -> rejected, naming both dims.
    let narrow_grid = grid_with_dims(70, 60);
    let narrow_values = vec![1.5f32; 70 * 60];
    let narrow = SelectedField2D::new(temp_selector(), "K", narrow_grid, narrow_values).unwrap();
    let err = write_hour_from_fields(
        &store_root,
        MODEL,
        RUN,
        12,
        &[("temp_2m", &narrow)],
        &[],
        BUILD,
        WRITTEN_UNIX + 600,
    )
    .unwrap_err();
    assert!(
        matches!(err, RwStoreError::Meta(_)),
        "expected Meta error for grid dims mismatch, got {err:?}"
    );
    let message = err.to_string();
    assert!(
        message.contains("70x60") && message.contains("80x60"),
        "error must name both grids' dims, got: {message}"
    );

    // Same dims but different coordinates is also a mismatch (bit-compare).
    let mut shifted_grid = regular_grid();
    shifted_grid.lon_deg[123] += 0.25;
    let shifted_values = vec![2.5f32; NX * NY];
    let shifted = SelectedField2D::new(temp_selector(), "K", shifted_grid, shifted_values).unwrap();
    let err = write_hour_from_fields(
        &store_root,
        MODEL,
        RUN,
        12,
        &[("temp_2m", &shifted)],
        &[],
        BUILD,
        WRITTEN_UNIX + 600,
    )
    .unwrap_err();
    assert!(
        matches!(err, RwStoreError::Meta(_)),
        "expected Meta error for grid coordinate mismatch, got {err:?}"
    );

    // The failed writes left no trace: one hour registered, no f012.rws.
    let manifest = load_manifest(&store_root);
    assert_eq!(
        manifest.hours.len(),
        1,
        "run.json must still have only hour 6"
    );
    assert!(manifest.hours.contains_key(&6));
    assert!(
        !run_dir(&store_root).join("f012.rws").exists(),
        "rejected hour must not leave an hour file behind"
    );

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn second_field_on_different_coords_rejected() {
    let dir = test_dir("second-field-coords");
    let store_root = dir.join("store");
    let temp = temp_field();

    // Same 80 x 60 dims, but the second field's lon coordinates are shifted:
    // without the per-field bit-compare this would store silently under the
    // first field's coordinates.
    let mut shifted_grid = regular_grid();
    for lon in &mut shifted_grid.lon_deg {
        *lon += 0.05;
    }
    let dewpoint = SelectedField2D::new(
        dewpoint_selector(),
        "K",
        shifted_grid,
        dewpoint_field().values.clone(),
    )
    .unwrap();

    let err = write_hour_from_fields(
        &store_root,
        MODEL,
        RUN,
        6,
        &[("temp_2m", &temp), ("dewpoint_2m", &dewpoint)],
        &[],
        BUILD,
        WRITTEN_UNIX,
    )
    .unwrap_err();
    assert!(
        matches!(err, RwStoreError::Format(_)),
        "expected Format error for second-field coordinate mismatch, got {err:?}"
    );
    let message = err.to_string();
    assert!(
        message.contains("dewpoint_2m"),
        "error must name the offending field, got: {message}"
    );
    assert!(
        !store_root.join(MODEL).exists(),
        "rejected write must not create store directories"
    );

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn volume_levels_sorted_and_deduped() {
    let dir = test_dir("levels");
    let store_root = dir.join("store");
    let temp = temp_field();

    // Levels arrive shuffled; planes are built per level so pairing must
    // survive the internal descending sort.
    let shuffled: [u16; 5] = [500, 1000, 300, 850, 700];
    let planes: Vec<Vec<f32>> = shuffled.iter().map(|&level| vol_plane(level)).collect();
    let volume = volume_input(&shuffled, &planes);

    let written = write_hour_from_fields(
        &store_root,
        MODEL,
        RUN,
        6,
        &[("temp_2m", &temp)],
        &[volume],
        BUILD,
        WRITTEN_UNIX,
    )
    .unwrap();

    let reader = HourReader::open(&written.path).unwrap();
    let var = reader
        .variable("temperature")
        .expect("volume variable present");
    assert_eq!(
        var.levels_hpa,
        VLEVELS.to_vec(),
        "stored levels must be sorted descending"
    );

    // The planes followed their levels through the sort: each stored level's
    // values match the analytic function of that level.
    let bound = quant_bound();
    let (ix, iy) = (12usize, 34usize);
    let column = reader.read_column_3d("temperature", ix, iy).unwrap();
    for (k, value) in column.iter().enumerate() {
        let expected = vol_value(ix, iy, VLEVELS[k]);
        assert!(
            (value - expected).abs() <= bound,
            "level {} hPa: got {value}, expected {expected} (plane/level pairing broken?)",
            VLEVELS[k]
        );
    }

    // A duplicate level is rejected.
    let dup_levels: [u16; 5] = [1000, 850, 850, 500, 300];
    let dup_planes: Vec<Vec<f32>> = dup_levels.iter().map(|&level| vol_plane(level)).collect();
    let dup_volume = volume_input(&dup_levels, &dup_planes);
    let err = write_hour_from_fields(
        &store_root,
        MODEL,
        RUN,
        9,
        &[("temp_2m", &temp)],
        &[dup_volume],
        BUILD,
        WRITTEN_UNIX,
    )
    .unwrap_err();
    assert!(
        matches!(err, RwStoreError::Format(_)),
        "expected Format error for duplicate level, got {err:?}"
    );
    let message = err.to_string();
    assert!(
        message.contains("duplicate") && message.contains("850"),
        "error must name the duplicate level, got: {message}"
    );
    assert!(
        !run_dir(&store_root).join("f009.rws").exists(),
        "rejected hour must not leave an hour file behind"
    );

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn no_2d_fields_errors() {
    let dir = test_dir("no-2d");
    let store_root = dir.join("store");
    let planes: Vec<Vec<f32>> = VLEVELS.iter().map(|&level| vol_plane(level)).collect();
    let volume = volume_input(&VLEVELS, &planes);

    let err = write_hour_from_fields(
        &store_root,
        MODEL,
        RUN,
        6,
        &[],
        &[volume],
        BUILD,
        WRITTEN_UNIX,
    )
    .unwrap_err();
    assert!(
        matches!(err, RwStoreError::Format(_)),
        "expected Format error for volumes-only input, got {err:?}"
    );
    let message = err.to_string();
    assert!(
        message.contains("2D field") && message.contains("grid"),
        "error must state a 2D field is required to carry the grid, got: {message}"
    );
    assert!(
        !store_root.join(MODEL).exists(),
        "rejected write must not create store directories"
    );

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn derived_vars_round_trip_through_read_grid_2d() {
    let dir = test_dir("derived");
    let store_root = dir.join("store");
    let temp = temp_field();

    // Two derived grids, one with a NaN region, written behind the extracted
    // field; their selector is the {"derived": "<slug>"} marker, not a
    // FieldSelector.
    let sbcape: Vec<f32> = (0..NY)
        .flat_map(|y| (0..NX).map(move |x| 10.0 * x as f32 + 2.0 * y as f32))
        .collect();
    let mut srh: Vec<f32> = (0..NY)
        .flat_map(|y| (0..NX).map(move |x| -50.0 + x as f32 - 0.5 * y as f32))
        .collect();
    for value in srh.iter_mut().take(40) {
        *value = f32::NAN;
    }
    let derived = [
        DerivedFieldInput {
            name: "sbcape",
            units: "J/kg",
            values: &sbcape,
        },
        DerivedFieldInput {
            name: "srh_0_3km",
            units: "m^2/s^2",
            values: &srh,
        },
    ];

    let written = write_hour_from_fields_with_derived(
        &store_root,
        MODEL,
        RUN,
        6,
        &[("temp_2m", &temp)],
        &derived,
        &[],
        BUILD,
        WRITTEN_UNIX,
    )
    .unwrap();
    assert_eq!(
        written.vars,
        vec![
            "temp_2m".to_string(),
            "sbcape".to_string(),
            "srh_0_3km".to_string()
        ],
        "derived vars ride behind the extracted 2D fields"
    );

    let reader = HourReader::open(&written.path).unwrap();
    let grid_file = GridFile::open(&run_dir(&store_root).join("grid.rwg")).unwrap();

    // read_grid_2d: bit-exact values, units, raw marker selector, grid.
    for (name, units, original) in [("sbcape", "J/kg", &sbcape), ("srh_0_3km", "m^2/s^2", &srh)] {
        let stored = read_grid_2d(&reader, &grid_file, name).unwrap();
        assert_eq!(stored.units, units, "{name}: units");
        assert_eq!(
            stored.selector,
            derived_selector(name),
            "{name}: selector must be the derived marker"
        );
        assert_eq!(
            derived_selector_slug(&stored.selector),
            Some(name),
            "{name}: marker slug must round-trip"
        );
        assert_eq!(stored.grid.shape, temp.grid.shape, "{name}: grid shape");
        assert_bits_eq(&stored.values, original, &format!("{name}: values"));
        assert_eq!(
            stored.projection,
            Some(GridProjection::Geographic),
            "{name}: projection comes from the grid file"
        );
    }

    // read_grid_2d also serves extracted fields, selector left raw.
    let stored_temp = read_grid_2d(&reader, &grid_file, "temp_2m").unwrap();
    assert_eq!(
        stored_temp.selector,
        serde_json::to_value(temp.selector).unwrap(),
        "extracted field selector stays raw but intact"
    );
    assert!(
        derived_selector_slug(&stored_temp.selector).is_none(),
        "an extracted field must not look like a derived marker"
    );

    // read_field_2d on a derived var fails cleanly, pointing at read_grid_2d.
    let err = read_field_2d(&reader, &grid_file, "sbcape").unwrap_err();
    assert!(
        matches!(err, RwStoreError::Meta(_)),
        "expected Meta error for the derived marker selector, got {err:?}"
    );
    let message = err.to_string();
    assert!(
        message.contains("read_grid_2d"),
        "error must point the caller at read_grid_2d, got: {message}"
    );

    // A wrong-sized derived plane is rejected before anything hits disk.
    let err = write_hour_from_fields_with_derived(
        &store_root,
        MODEL,
        RUN,
        9,
        &[("temp_2m", &temp)],
        &[DerivedFieldInput {
            name: "sbcape",
            units: "J/kg",
            values: &sbcape[..NX * NY - 1],
        }],
        &[],
        BUILD,
        WRITTEN_UNIX,
    )
    .unwrap_err();
    assert!(
        matches!(err, RwStoreError::Format(_)),
        "expected Format error for a wrong-sized derived plane, got {err:?}"
    );
    assert!(
        !run_dir(&store_root).join("f009.rws").exists(),
        "rejected hour must not leave an hour file behind"
    );

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn exact_time_hours_use_ordinal_slots_and_preserve_second_resolution() {
    let dir = test_dir("exact-time-round-trip");
    let store_root = dir.join("store");
    let temp = temp_field();
    let origin = 1_700_000_000i64;

    for (slot, lead_seconds) in [(0u16, 0u64), (1, 1_800)] {
        write_hour_from_fields_exact(
            &store_root,
            MODEL,
            RUN,
            slot,
            RwsExactTime {
                lead_seconds,
                valid_unix: origin + lead_seconds as i64,
            },
            &[("temp_2m", &temp)],
            &[],
            BUILD,
            WRITTEN_UNIX,
        )
        .unwrap();
    }

    let manifest = load_manifest(&store_root);
    assert!(manifest.is_exact_time_axis());
    assert_eq!(manifest.hours[&0].file, "f000.rws");
    assert_eq!(manifest.hours[&1].file, "f001.rws");
    assert_eq!(
        manifest.exact_times().collect::<Vec<_>>(),
        vec![
            (
                0,
                RwsExactTime {
                    lead_seconds: 0,
                    valid_unix: origin,
                },
            ),
            (
                1,
                RwsExactTime {
                    lead_seconds: 1_800,
                    valid_unix: origin + 1_800,
                },
            ),
        ]
    );
    let hour = HourReader::open(&run_dir(&store_root).join("f001.rws")).unwrap();
    assert_eq!(hour.meta().forecast_hour, 1);
    assert_eq!(
        hour.meta().exact_time(),
        Some(RwsExactTime {
            lead_seconds: 1_800,
            valid_unix: origin + 1_800,
        })
    );
    manifest.validate_hour_meta(1, hour.meta()).unwrap();
    let report = rw_store::validate::validate_run_dir(
        &run_dir(&store_root),
        rw_store::validate::ValidateDepth::Structural,
    )
    .unwrap();
    assert!(
        report.is_ok(),
        "exact v2 run validation: {:?}",
        report.errors
    );

    let remap_error = write_hour_from_fields_exact(
        &store_root,
        MODEL,
        RUN,
        1,
        RwsExactTime {
            lead_seconds: 1_860,
            valid_unix: origin + 1_860,
        },
        &[("temp_2m", &temp)],
        &[],
        BUILD,
        WRITTEN_UNIX,
    )
    .unwrap_err();
    assert!(
        remap_error.to_string().contains("refusing to remap"),
        "{remap_error}"
    );
    let unchanged = HourReader::open(&run_dir(&store_root).join("f001.rws")).unwrap();
    assert_eq!(unchanged.meta().exact_time().unwrap().lead_seconds, 1_800);

    let error = write_hour_from_fields_exact(
        &store_root,
        MODEL,
        RUN,
        2,
        RwsExactTime {
            lead_seconds: 900,
            valid_unix: origin + 900,
        },
        &[("temp_2m", &temp)],
        &[],
        BUILD,
        WRITTEN_UNIX,
    )
    .unwrap_err();
    assert!(error.to_string().contains("increase strictly"), "{error}");
    assert!(!run_dir(&store_root).join("f002.rws").exists());

    let error = write_hour_from_fields(
        &store_root,
        MODEL,
        RUN,
        3,
        &[("temp_2m", &temp)],
        &[],
        BUILD,
        WRITTEN_UNIX,
    )
    .unwrap_err();
    assert!(error.to_string().contains("cannot be opened"), "{error}");
    assert!(!run_dir(&store_root).join("f003.rws").exists());

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn grid_seam_writes_a_derived_only_selection() {
    let dir = test_dir("derived-only");
    let store_root = dir.join("store");
    let grid = regular_grid();
    let values = vec![123.0f32; NX * NY];
    let derived = [DerivedFieldInput {
        name: "sbcape",
        units: "J/kg",
        values: &values,
    }];

    let written = write_hour_from_grid_with_derived(
        &store_root,
        MODEL,
        RUN,
        0,
        &grid,
        Some(&GridProjection::Geographic),
        &[],
        &derived,
        &[],
        BUILD,
        WRITTEN_UNIX,
    )
    .unwrap();
    assert_eq!(written.vars, vec!["sbcape".to_string()]);
    let reader = HourReader::open(&written.path).unwrap();
    let stored = reader.read_full_2d("sbcape").unwrap();
    assert_bits_eq(&stored, &values, "derived-only field");

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn grid_seam_rejects_publicly_constructible_zero_and_overflow_shapes() {
    let dir = test_dir("malformed-grid-shape");
    let store_root = dir.join("store");
    let values = Vec::<f32>::new();
    let derived = [DerivedFieldInput {
        name: "diagnostic",
        units: "1",
        values: &values,
    }];
    for shape in [
        GridShape { nx: 0, ny: 1 },
        GridShape {
            nx: usize::MAX,
            ny: 2,
        },
    ] {
        let grid = LatLonGrid {
            shape,
            lat_deg: Vec::new(),
            lon_deg: Vec::new(),
        };
        let error = write_hour_from_grid_with_derived(
            &store_root,
            MODEL,
            RUN,
            0,
            &grid,
            None,
            &[],
            &derived,
            &[],
            BUILD,
            WRITTEN_UNIX,
        )
        .unwrap_err();
        assert!(matches!(error, RwStoreError::Format(_)));
    }
    assert!(!store_root.exists(), "invalid grids must not touch disk");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn read_field_2d_rejects_mismatched_grid_file() {
    let dir = test_dir("wrong-grid-file");
    let store_a = dir.join("store-a");
    let store_b = dir.join("store-b");
    let temp = temp_field();

    write_hour_from_fields(
        &store_a,
        MODEL,
        RUN,
        6,
        &[("temp_2m", &temp)],
        &[],
        BUILD,
        WRITTEN_UNIX,
    )
    .unwrap();

    // A second store with shifted coordinates -> different grid hash.
    let mut shifted_grid = regular_grid();
    for lon in &mut shifted_grid.lon_deg {
        *lon += 5.0;
    }
    let shifted =
        SelectedField2D::new(temp_selector(), "K", shifted_grid, temp.values.clone()).unwrap();
    write_hour_from_fields(
        &store_b,
        MODEL,
        RUN,
        6,
        &[("temp_2m", &shifted)],
        &[],
        BUILD,
        WRITTEN_UNIX,
    )
    .unwrap();

    let reader = HourReader::open(&run_dir(&store_a).join("f006.rws")).unwrap();
    let wrong_grid = GridFile::open(&run_dir(&store_b).join("grid.rwg")).unwrap();
    let err = read_field_2d(&reader, &wrong_grid, "temp_2m").unwrap_err();
    assert!(
        matches!(err, RwStoreError::Grid(_)),
        "expected Grid error for grid-hash mismatch, got {err:?}"
    );

    let _ = fs::remove_dir_all(&dir);
}

/// A live `HourIngestWriter` (post-`begin`, pre-`finish`) holds the run-dir
/// advisory lock, so a competing writer's `RunLock::try_acquire` sees it as
/// held; once the writer `finish`es and its guard drops, the lock is free.
/// (fs4 locks are per-handle, so a second handle in this same process probes
/// the lock exactly as a second process would.)
#[test]
fn ingest_writer_holds_run_lock_until_finish() {
    let dir = test_dir("ingest-lock");
    let store_root = dir.join("store");
    let temp = temp_field();
    let run_dir = run_dir(&store_root);

    // Before any writer: the run dir does not exist, so a probe is an I/O
    // error (begin creates the dir), not a held lock — sanity that we are not
    // accidentally locking nothing.
    assert!(RunLock::try_acquire(&run_dir).is_err());

    let mut writer = HourIngestWriter::begin(
        &store_root,
        MODEL,
        RUN,
        6,
        &temp.grid,
        temp.projection.as_ref(),
        BUILD,
    )
    .unwrap();
    writer
        .add_field_2d(
            "temp_2m",
            "K",
            serde_json::json!({"v": "TMP"}),
            &temp.values,
        )
        .unwrap();

    // While the writer is live the lock is held: a competing acquire backs off.
    assert!(
        RunLock::try_acquire(&run_dir).unwrap().is_none(),
        "the live ingest writer must hold the run-dir lock"
    );

    // Finishing drops the writer (and its lock guard) — now the lock is free.
    writer.finish(WRITTEN_UNIX).unwrap();
    assert!(
        RunLock::try_acquire(&run_dir).unwrap().is_some(),
        "after finish the run-dir lock must be released"
    );

    let _ = fs::remove_dir_all(&dir);
}
