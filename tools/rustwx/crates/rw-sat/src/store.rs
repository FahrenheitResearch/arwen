//! Decoded GOES ABI field -> rw-store frame files.
//!
//! Store convention (documented contract for every rw-sat consumer):
//!
//! ```text
//! <store_root>/<sat>/<sector>_c<band>_<YYYYMMDD>[_<k>]/
//!     grid.rwg     per-pixel lat/lon written once per run dir (hash-shared:
//!                  identical fixed grids produce byte-identical files)
//!     run.json     rw-store run manifest, hours keyed by HHMM
//!     tHHMM.rws    one frame per scan, regular rw-store hour file
//! ```
//!
//! - `model` (in rw-store terms) is the satellite slug (`g19`, `g18`).
//! - `run` is `<sector>_c<band>_<YYYYMMDD>` in UTC of the SCAN START time
//!   (from the filename `s` timestamp — never the local clock). When a
//!   mesoscale sector is repositioned mid-day its fixed grid changes, so a
//!   new run dir `..._2`, `..._3` is opened; CONUS/full-disk grids are
//!   stable and stay in one dir per day.
//! - the u16 `forecast_hour` slot carries `HHMM` of the scan start
//!   (`1851` = 18:51 UTC) and the frame file is `t{HHMM:04}.rws`.
//! - each frame holds one 2D variable `cmi_c<band>` (CMI: reflectance
//!   factor 0..1 for C01-06, brightness temperature Kelvin for C07-16) with
//!   a self-describing `{"goes": {...}}` selector carrying satellite,
//!   product, band, scan times, and the geostationary projection
//!   parameters. The `.rwg` projection slot is `None` — `GridProjection`
//!   has no geostationary variant; the per-pixel lat/lon mesh is the
//!   geometry of record.

use std::error::Error;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use chrono::{DateTime, NaiveDate, NaiveTime, TimeZone, Timelike, Utc};

use rustwx_core::{GridShape, LatLonGrid};
use rw_store::grid::{GridFile, write_grid};
use rw_store::lock::RunLock;
use rw_store::reader::HourReader;
use rw_store::run::{RwsHourEntry, RwsRunManifest};
use rw_store::writer::HourWriter;

use crate::abi::{AbiFixedGrid, AbiSector, GoesAbiField};
use crate::geostationary::SweepAngleAxis;

/// How long [`write_band_frame`] waits for the run-dir advisory lock before
/// failing with `RwStoreError::Locked`. Matches the ingest writer's 60s; the
/// normal contention is the rolling-window prune holding the lock briefly.
const FRAME_LOCK_TIMEOUT: Duration = Duration::from_secs(60);

/// What one frame write produced.
#[derive(Debug, Clone)]
pub struct WrittenFrame {
    /// Satellite slug = rw-store model (`g19`).
    pub model: String,
    /// Run dir name (`conus_c13_20260610`).
    pub run: String,
    /// Scan start as HHMM (the rw-store forecast_hour slot).
    pub hhmm: u16,
    pub scan_time_utc: DateTime<Utc>,
    pub path: PathBuf,
    pub bytes: u64,
    pub encode_ms: u64,
    pub grid_hash: String,
    /// Whether this write opened a new run dir (first frame of the day, or
    /// a mesoscale sector move).
    pub created_run: bool,
    pub variable: String,
}

/// A frame read back from the store.
#[derive(Debug, Clone)]
pub struct StoredFrame {
    pub values: Vec<f32>,
    pub nx: usize,
    pub ny: usize,
    pub units: String,
    pub variable: String,
    pub selector: serde_json::Value,
    /// Row-orientation hint from the grid (GOES grids store north first).
    pub lat_descending: Option<bool>,
}

/// `t{HHMM:04}.rws`
pub fn frame_file_name(hhmm: u16) -> String {
    format!("t{hhmm:04}.rws")
}

/// Variable name for a band frame (`cmi_c13`).
pub fn band_variable_name(band: u8) -> String {
    format!("cmi_c{band:02}")
}

/// Sector slug used in run names.
pub fn sector_slug(sector: &AbiSector) -> String {
    match sector {
        AbiSector::Conus => "conus".to_string(),
        AbiSector::FullDisk => "fulldisk".to_string(),
        AbiSector::Mesoscale1 => "meso1".to_string(),
        AbiSector::Mesoscale2 => "meso2".to_string(),
        AbiSector::Mesoscale => "meso".to_string(),
        AbiSector::Unknown(value) => value
            .chars()
            .map(|ch| {
                if ch.is_ascii_alphanumeric() {
                    ch.to_ascii_lowercase()
                } else {
                    '_'
                }
            })
            .collect(),
    }
}

/// Parse the `YYYYMMDD` day token out of a run name
/// (`conus_c13_20260610` or `..._20260610_2`).
pub fn run_day(run_name: &str) -> Option<NaiveDate> {
    run_name
        .split('_')
        .filter(|token| token.len() == 8 && token.bytes().all(|b| b.is_ascii_digit()))
        .find_map(|token| NaiveDate::parse_from_str(token, "%Y%m%d").ok())
}

/// Observation time of a stored frame: run-day + HHMM.
pub fn frame_time(run_name: &str, hhmm: u16) -> Option<DateTime<Utc>> {
    let day = run_day(run_name)?;
    let time = NaiveTime::from_hms_opt(u32::from(hhmm / 100), u32::from(hhmm % 100), 0)?;
    Some(Utc.from_utc_datetime(&day.and_time(time)))
}

/// Stride-`step` decimation of a field (values and both scan-angle axes),
/// for keeping hi-res bands (CONUS C02 is 10000x6000) at a sane store size.
/// Exact subsampling — no smoothing — so the science values pass through
/// untouched. `step <= 1` returns the field unchanged.
pub fn downsample_field(field: GoesAbiField, step: usize) -> GoesAbiField {
    if step <= 1 {
        return field;
    }
    let grid = &field.scene.fixed_grid;
    let (nx, ny) = (grid.nx, grid.ny);
    let xs: Vec<usize> = (0..nx).step_by(step).collect();
    let ys: Vec<usize> = (0..ny).step_by(step).collect();
    let mut values = Vec::with_capacity(xs.len() * ys.len());
    for &y in &ys {
        for &x in &xs {
            values.push(field.values[y * nx + x]);
        }
    }
    let x_scan_rad: Vec<f64> = xs.iter().map(|&x| grid.x_scan_rad[x]).collect();
    let y_scan_rad: Vec<f64> = ys.iter().map(|&y| grid.y_scan_rad[y]).collect();
    let mut scene = field.scene;
    scene.fixed_grid = AbiFixedGrid {
        nx: xs.len(),
        ny: ys.len(),
        x_scan_rad,
        y_scan_rad,
    };
    GoesAbiField {
        scene,
        variable_name: field.variable_name,
        units: field.units,
        values,
    }
}

/// Write one decoded band field as a store frame. `written_unix` is
/// supplied by the caller (the library never reads the wall clock), matching
/// the rw-store convention.
pub fn write_band_frame(
    store_root: &Path,
    field: &GoesAbiField,
    written_unix: u64,
) -> Result<WrittenFrame, Box<dyn Error>> {
    let scene = &field.scene;
    let band = scene
        .channel
        .ok_or_else(|| boxed_error("GOES ABI field has no band (multiband product?)"))?;
    let model = scene.satellite.as_str().to_ascii_lowercase();
    let sector = sector_slug(&scene.sector);
    let day = scene.start_time_utc.format("%Y%m%d").to_string();
    let hhmm = (scene.start_time_utc.hour() * 100 + scene.start_time_utc.minute()) as u16;
    let run_base = format!("{sector}_c{band:02}_{day}");

    let (nx, ny) = (scene.fixed_grid.nx, scene.fixed_grid.ny);
    if field.values.len() != nx.saturating_mul(ny) {
        return Err(boxed_error(format!(
            "field length {} does not match grid {nx}x{ny}",
            field.values.len()
        )));
    }
    let (lat, lon) = scene.lat_lon_mesh();
    let grid = LatLonGrid::new(GridShape::new(nx, ny)?, lat, lon)?;

    let model_dir = store_root.join(&model);
    let resolved = resolve_run_dir(&model_dir, &run_base, &grid)?;
    let run_dir = model_dir.join(&resolved.run_name);
    fs::create_dir_all(&run_dir)?;

    // Single-writer-per-run-dir (FORMAT.md §7). This frame writer does NOT go
    // through `HourIngestWriter` (it drives the lower-level `HourWriter` plus
    // its own run.json update), so it must take the same advisory lock around
    // its critical section: grid.rwg, the t*.rws frame, and the manifest. The
    // real incident this guards against was two processes (bowecho +
    // rusty-weather) writing one sat store's rolling window. 60s mirrors the
    // ingest writer; the normal contention is the window prune holding the
    // lock for a moment. Held until `_lock` drops at function return.
    let _lock = RunLock::acquire(&run_dir, FRAME_LOCK_TIMEOUT)?;

    let grid_path = run_dir.join("grid.rwg");
    let grid_hash = match resolved.existing_grid_hash {
        Some(hash) => hash,
        None => write_grid(&grid_path, &grid, None)?,
    };

    let started = Instant::now();
    let variable = band_variable_name(band);
    let units = field.units.clone().unwrap_or_default();
    let selector = goes_selector(field, band);
    let mut writer = HourWriter::new(
        &model,
        &resolved.run_name,
        hhmm,
        nx,
        ny,
        &grid_hash,
        concat!("rw-sat ", env!("CARGO_PKG_VERSION")),
    );
    writer.add_surface2d(&variable, &units, selector, &field.values)?;
    let file_name = frame_file_name(hhmm);
    let frame_path = run_dir.join(&file_name);
    writer.finish(&frame_path)?;
    let encode_ms = started.elapsed().as_millis() as u64;
    let bytes = fs::metadata(&frame_path)?.len();

    // Register the frame in run.json last, so a failed write never appears
    // in the manifest.
    let manifest_path = run_dir.join("run.json");
    let writer_info = rw_store::format::RwsWriterInfo {
        name: "rw-sat".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        build: concat!("rw-sat ", env!("CARGO_PKG_VERSION")).to_string(),
    };
    let mut manifest = RwsRunManifest::load_or_new(
        &manifest_path,
        &model,
        &resolved.run_name,
        &grid_hash,
        nx,
        ny,
        writer_info,
    )?;
    manifest.register_hour(
        hhmm,
        RwsHourEntry {
            file: file_name,
            // Satellite frames are keyed by scan hhmm, not by a forecast
            // lead, so they are v1 hour entries: rw-store's exact-time v2
            // pair stays absent (serde skips `None`), which is byte-for-byte
            // the manifest this writer produced before rw-store added the
            // fields.
            lead_seconds: None,
            valid_unix: None,
            written_unix,
            encode_ms,
            variables: vec![variable.clone()],
        },
    );
    manifest.save(&manifest_path)?;

    Ok(WrittenFrame {
        model,
        run: resolved.run_name,
        hhmm,
        scan_time_utc: scene.start_time_utc,
        path: frame_path,
        bytes,
        encode_ms,
        grid_hash,
        created_run: resolved.created,
        variable,
    })
}

/// Read one frame back: the first 2D variable of the hour file, with the
/// run grid validated against the hour's `grid_hash` (rw-store's check).
pub fn read_frame(
    store_root: &Path,
    model: &str,
    run: &str,
    hhmm: u16,
) -> Result<StoredFrame, Box<dyn Error>> {
    let run_dir = store_root.join(model).join(run);
    let reader = HourReader::open(&run_dir.join(frame_file_name(hhmm)))?;
    let grid = GridFile::open(&run_dir.join("grid.rwg"))?;
    let variable = reader
        .meta()
        .variables
        .iter()
        .find(|var| var.kind == "surface2d")
        .map(|var| var.name.clone())
        .ok_or_else(|| boxed_error("frame holds no 2D variable"))?;
    let stored = rw_store::read_grid_2d(&reader, &grid, &variable)?;
    Ok(StoredFrame {
        values: stored.values,
        nx: grid.nx,
        ny: grid.ny,
        units: stored.units,
        variable,
        selector: stored.selector,
        lat_descending: grid.lat_descending(),
    })
}

struct ResolvedRun {
    run_name: String,
    existing_grid_hash: Option<String>,
    created: bool,
}

/// Find the run dir for `run_base` whose stored grid is bit-identical to
/// `grid`, or pick the next free suffixed name. This is what keeps moving
/// mesoscale sectors honest: a sector move changes the fixed grid, which
/// opens a fresh run dir instead of corrupting the existing one.
fn resolve_run_dir(
    model_dir: &Path,
    run_base: &str,
    grid: &LatLonGrid,
) -> Result<ResolvedRun, Box<dyn Error>> {
    let mut candidates: Vec<String> = Vec::new();
    if model_dir.is_dir() {
        for entry in fs::read_dir(model_dir)? {
            let entry = entry?;
            if !entry.file_type()?.is_dir() {
                continue;
            }
            let name = entry.file_name().to_string_lossy().to_string();
            if name == run_base || name.starts_with(&format!("{run_base}_")) {
                candidates.push(name);
            }
        }
    }
    candidates.sort();

    for name in &candidates {
        let grid_path = model_dir.join(name).join("grid.rwg");
        if !grid_path.is_file() {
            continue;
        }
        let existing = GridFile::open(&grid_path)?;
        if existing.nx == grid.shape.nx
            && existing.ny == grid.shape.ny
            && coords_bit_identical(&existing.lat, &grid.lat_deg)
            && coords_bit_identical(&existing.lon, &grid.lon_deg)
        {
            return Ok(ResolvedRun {
                run_name: name.clone(),
                existing_grid_hash: Some(existing.hash),
                created: false,
            });
        }
    }

    // No matching grid: first free name (base, then base_2, base_3, ...).
    let mut suffix = 1usize;
    loop {
        let name = if suffix == 1 {
            run_base.to_string()
        } else {
            format!("{run_base}_{suffix}")
        };
        if !candidates.contains(&name) {
            return Ok(ResolvedRun {
                run_name: name,
                existing_grid_hash: None,
                created: true,
            });
        }
        suffix += 1;
    }
}

fn coords_bit_identical(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b.iter())
            .all(|(x, y)| x.to_bits() == y.to_bits())
}

fn goes_selector(field: &GoesAbiField, band: u8) -> serde_json::Value {
    let scene = &field.scene;
    serde_json::json!({
        "goes": {
            "satellite": scene.satellite.as_str(),
            "product": scene.product,
            "band": band,
            "source_variable": field.variable_name,
            "scan_start_utc": scene.start_time_utc.to_rfc3339(),
            "scan_end_utc": scene.end_time_utc.to_rfc3339(),
            "projection": {
                "perspective_point_height_m": scene.projection.perspective_point_height_m,
                "semi_major_axis_m": scene.projection.semi_major_axis_m,
                "semi_minor_axis_m": scene.projection.semi_minor_axis_m,
                "longitude_of_projection_origin_deg":
                    scene.projection.longitude_of_projection_origin_deg,
                "sweep_angle_axis": match scene.projection.sweep_angle_axis {
                    SweepAngleAxis::X => "x",
                    SweepAngleAxis::Y => "y",
                },
            },
        }
    })
}

fn boxed_error(message: impl Into<String>) -> Box<dyn Error> {
    Box::new(io::Error::new(io::ErrorKind::InvalidData, message.into()))
}

#[cfg(test)]
pub(crate) mod test_support {
    use chrono::TimeZone;

    use super::*;
    use crate::abi::{AbiFixedGrid, GoesAbiScene, GoesImagerProjection};
    use crate::goes::GoesSatellite;

    /// Small synthetic CONUS-ish scene near the sub-satellite point so every
    /// pixel projects to a finite lat/lon.
    pub fn synthetic_field(
        nx: usize,
        ny: usize,
        start: DateTime<Utc>,
        band: u8,
        x_offset_rad: f64,
    ) -> GoesAbiField {
        let x_scan_rad: Vec<f64> = (0..nx)
            .map(|i| x_offset_rad + -0.02 + 0.04 * i as f64 / (nx.max(2) - 1) as f64)
            .collect();
        // GOES y axes descend (north first); mirror that.
        let y_scan_rad: Vec<f64> = (0..ny)
            .map(|j| 0.05 - 0.03 * j as f64 / (ny.max(2) - 1) as f64)
            .collect();
        let scene = GoesAbiScene {
            path: PathBuf::from("synthetic.nc"),
            product: "ABI-L2-CMIPC".to_string(),
            sector: AbiSector::Conus,
            channel: Some(band),
            satellite: GoesSatellite::G19,
            start_time_utc: start,
            end_time_utc: start + chrono::Duration::seconds(150),
            projection: GoesImagerProjection {
                perspective_point_height_m: 35_786_023.0,
                semi_major_axis_m: 6_378_137.0,
                semi_minor_axis_m: 6_356_752.314_14,
                longitude_of_projection_origin_deg: -75.0,
                sweep_angle_axis: SweepAngleAxis::X,
            },
            fixed_grid: AbiFixedGrid {
                nx,
                ny,
                x_scan_rad,
                y_scan_rad,
            },
        };
        let values: Vec<f32> = (0..nx * ny).map(|i| 200.0 + (i % 97) as f32).collect();
        GoesAbiField {
            scene,
            variable_name: "CMI".to_string(),
            units: Some("K".to_string()),
            values,
        }
    }

    pub fn scan_start(hour: u32, minute: u32) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 6, 10, hour, minute, 18).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::{scan_start, synthetic_field};
    use super::*;
    use std::fs;

    fn test_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("rw-sat-store-{}-{}", std::process::id(), name));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn frame_round_trips_through_the_store() {
        let dir = test_dir("round-trip");
        let field = synthetic_field(12, 10, scan_start(18, 51), 13, 0.0);
        let written = write_band_frame(&dir, &field, 1_770_000_000).unwrap();

        assert_eq!(written.model, "g19");
        assert_eq!(written.run, "conus_c13_20260610");
        assert_eq!(written.hhmm, 1851);
        assert!(written.created_run);
        assert_eq!(written.variable, "cmi_c13");
        assert!(written.path.ends_with("t1851.rws"));
        assert!(written.path.is_file());
        assert!(
            dir.join("g19/conus_c13_20260610/grid.rwg").is_file(),
            "grid.rwg written once per run dir"
        );
        assert!(dir.join("g19/conus_c13_20260610/run.json").is_file());

        let frame = read_frame(&dir, "g19", "conus_c13_20260610", 1851).unwrap();
        assert_eq!((frame.nx, frame.ny), (12, 10));
        assert_eq!(frame.units, "K");
        assert_eq!(frame.variable, "cmi_c13");
        assert_eq!(frame.values, field.values);
        assert_eq!(
            frame.lat_descending,
            Some(true),
            "GOES grids store north first"
        );
        assert_eq!(
            frame.selector["goes"]["band"], 13,
            "selector is self-describing"
        );
        assert_eq!(frame.selector["goes"]["satellite"], "G19");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn same_grid_frames_share_one_run_dir() {
        let dir = test_dir("shared-run");
        let first = synthetic_field(12, 10, scan_start(18, 51), 13, 0.0);
        let second = synthetic_field(12, 10, scan_start(18, 56), 13, 0.0);
        let one = write_band_frame(&dir, &first, 1).unwrap();
        let two = write_band_frame(&dir, &second, 2).unwrap();
        assert_eq!(one.run, two.run);
        assert!(!two.created_run);
        assert_eq!(one.grid_hash, two.grid_hash);

        let manifest: RwsRunManifest = serde_json::from_slice(
            &fs::read(dir.join("g19").join(&one.run).join("run.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(manifest.hours.len(), 2);
        assert_eq!(manifest.hours[&1851].file, "t1851.rws");
        assert_eq!(manifest.hours[&1856].file, "t1856.rws");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn moved_grid_opens_a_new_run_dir() {
        let dir = test_dir("moved-grid");
        let before = synthetic_field(12, 10, scan_start(18, 50), 13, 0.0);
        // Sector repositioned: same shape, shifted scan angles.
        let after = synthetic_field(12, 10, scan_start(18, 51), 13, 0.004);
        let one = write_band_frame(&dir, &before, 1).unwrap();
        let two = write_band_frame(&dir, &after, 2).unwrap();
        assert_eq!(one.run, "conus_c13_20260610");
        assert_eq!(two.run, "conus_c13_20260610_2");
        assert!(two.created_run);
        assert_ne!(one.grid_hash, two.grid_hash);

        // A third frame back on the FIRST grid reuses the first run dir.
        let back = synthetic_field(12, 10, scan_start(18, 52), 13, 0.0);
        let three = write_band_frame(&dir, &back, 3).unwrap();
        assert_eq!(three.run, "conus_c13_20260610");
        assert!(!three.created_run);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn downsample_strides_values_and_axes() {
        let field = synthetic_field(9, 7, scan_start(12, 0), 2, 0.0);
        let full = field.clone();
        let down = downsample_field(field, 3);
        assert_eq!(down.scene.fixed_grid.nx, 3);
        assert_eq!(down.scene.fixed_grid.ny, 3);
        assert_eq!(down.values.len(), 9);
        // Sample (x=3, y=6) in the source lands at (1, 2) downsampled.
        assert_eq!(down.values[2 * 3 + 1], full.values[6 * 9 + 3]);
        assert_eq!(
            down.scene.fixed_grid.x_scan_rad[1],
            full.scene.fixed_grid.x_scan_rad[3]
        );
        assert_eq!(
            down.scene.fixed_grid.y_scan_rad[2],
            full.scene.fixed_grid.y_scan_rad[6]
        );
        // step <= 1 is the identity.
        let same = downsample_field(full.clone(), 1);
        assert_eq!(same.values, full.values);
    }

    #[test]
    fn run_name_helpers_parse_day_and_time() {
        assert_eq!(
            run_day("conus_c13_20260610"),
            NaiveDate::from_ymd_opt(2026, 6, 10)
        );
        assert_eq!(
            run_day("meso1_c02_20260610_2"),
            NaiveDate::from_ymd_opt(2026, 6, 10)
        );
        assert_eq!(run_day("bogus"), None);
        let time = frame_time("conus_c13_20260610", 1851).unwrap();
        assert_eq!(time.to_rfc3339(), "2026-06-10T18:51:00+00:00");
    }
}
