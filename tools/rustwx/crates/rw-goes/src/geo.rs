//! `rw_goes geo-fetch` and `rw_goes geo-bt`: the geostationary imagers
//! beyond GOES ABI (Himawari AHI, GK-2A AMI) on the same pack the ABI `bt`
//! subcommand writes, and one fetcher for every imager in the source table
//! (`rw_sat::geo_imager`'s `geo_imagers.json`: Himawari-8/9, GK-2A, GOES
//! 16/18/19 full disk and CONUS).
//!
//! * **geo-fetch** resolves a nominal scan from the table (`--time T`: the
//!   newest complete scan at or before T within `--tolerance-minutes`;
//!   `--live`: the newest complete scan now, looking back
//!   `--lookback-minutes`), downloads every band file (and the companion
//!   clear-sky mask where the table names one) into the content cache, and
//!   prints a `gpuwm-obs.geo-fetch.v1` record with the local paths.
//! * **geo-bt** decodes one band of one scan (`--rad` once per file: an AMI
//!   NetCDF, or the HSD segments of an AHI band, `.DAT` or `.DAT.bz2`),
//!   crops it to `--window` or the latitude/longitude box `--bbox`, and
//!   writes a `gpuwm-obs.goes-bt.v1` pack: planes bt, rad, lat, lon and
//!   rad_dqf, the producer's calibration recorded verbatim, and the
//!   measurement time of every row.

use std::collections::BTreeMap;
use std::error::Error;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Duration, Utc};
use serde::Serialize;

use rw_sat::ami::{read_ami_l1b, read_ami_navigation};
use rw_sat::geo_imager::{
    GeoImagerScan, GeoSource, find_source, fetch_scan, list_scan, scan_at_or_before, window_for_bbox,
};
use rw_sat::himawari::read_ahi_segments;
use rw_sat::s3::build_agent;

use crate::pack::{DqfRow, PayloadBuilder, ProjectionEntry, SourceEntry, boxed_error, hex_sha256, write_container};
use crate::radiance::{BT_BUILD_SCHEMA, BT_SCHEMA, BtCounts, BtMeta, ProvenanceRow};

pub const GEO_FETCH_SCHEMA: &str = "gpuwm-obs.geo-fetch.v1";

/// How a pixel's time is placed for the AHI/AMI packs, verbatim in each pack.
pub const GEO_ROW_TIME_MODEL: &str = "measurement time of a row = row_time_unix_s[row]: AHI from each HSD segment's own \
     observation start/end (linear inside the segment's lines), AMI from the file's observation_start_time/end_time \
     (linear over the full-disk lines); both scan north to south";

#[derive(Debug, Default, Clone)]
pub struct GeoOptions {
    pub satellite: Option<String>,
    pub sector: Option<String>,
    pub bands: Vec<u8>,
    pub time: Option<String>,
    pub live: bool,
    pub tolerance_minutes: Option<i64>,
    pub lookback_minutes: Option<i64>,
    pub cache: Option<PathBuf>,
    pub out: Option<PathBuf>,
    pub rad: Vec<PathBuf>,
    pub window: Option<[usize; 4]>,
    pub bbox: Option<[f64; 4]>,
    pub received_utc: Option<String>,
}

pub fn parse_bbox(raw: &str) -> Result<[f64; 4], Box<dyn Error>> {
    let parts: Vec<f64> = raw
        .split(',')
        .map(|p| p.trim().parse::<f64>())
        .collect::<Result<_, _>>()
        .map_err(|_| boxed_error(format!("--bbox expects SOUTH,NORTH,WEST,EAST degrees, got {raw:?}")))?;
    if parts.len() != 4 || parts[0] >= parts[1] || parts[0] < -90.0 || parts[1] > 90.0 {
        return Err(boxed_error(format!("--bbox expects SOUTH,NORTH,WEST,EAST degrees, got {raw:?}")));
    }
    Ok([parts[0], parts[1], parts[2], parts[3]])
}

// ---------------------------------------------------------------------------
// geo-fetch
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct FetchedFile {
    key: String,
    path: String,
    bytes: u64,
    last_modified: String,
    cache_hit: bool,
}

#[derive(Serialize)]
struct GeoFetchRecord {
    schema: &'static str,
    status: &'static str,
    satellite: String,
    instrument: String,
    sector: String,
    bucket: String,
    scan_nominal: String,
    mode: &'static str,
    requested: String,
    bands: BTreeMap<String, Vec<FetchedFile>>,
    companions: BTreeMap<String, Vec<FetchedFile>>,
    received_utc: String,
    scans_examined: usize,
}

pub fn cmd_geo_fetch(o: &GeoOptions) -> Result<String, Box<dyn Error>> {
    let satellite = o.satellite.as_deref().ok_or_else(|| boxed_error("--satellite is required (H09, GK2A, G18 ...)"))?;
    let sector = o.sector.as_deref().unwrap_or("F");
    let source: GeoSource = find_source(satellite, sector)?;
    if o.bands.is_empty() {
        return Err(boxed_error("--bands is required (for example 8,9,10,13)"));
    }
    let (target, mode, span) = match (&o.time, o.live) {
        (Some(_), true) => return Err(boxed_error("--time and --live are exclusive")),
        (Some(t), false) => (crate::parse_time(t)?, "archive", o.tolerance_minutes.unwrap_or(30)),
        (None, true) => (Utc::now(), "live", o.lookback_minutes.unwrap_or(60)),
        (None, false) => return Err(boxed_error("give --time T (archive) or --live")),
    };
    let agent = build_agent();
    let cache = o.cache.clone().unwrap_or_else(|| PathBuf::from(".rw-goes-cache"));
    // Newest complete scan at or before the target, stepping back one cadence at a time.
    let mut scan = scan_at_or_before(&source, target);
    let stop = target - Duration::minutes(span);
    let mut examined = 0usize;
    let listing = loop {
        if scan < stop {
            return Err(boxed_error(format!(
                "no complete {} {} scan with bands {:?} between {} and {} (examined {examined})",
                source.satellite,
                source.sector,
                o.bands,
                stop.format("%Y-%m-%dT%H:%M:%SZ"),
                target.format("%Y-%m-%dT%H:%M:%SZ")
            )));
        }
        examined += 1;
        let l = list_scan(&agent, &source, scan, &o.bands)?;
        if l.complete {
            break l;
        }
        scan -= Duration::minutes(source.cadence_minutes.max(1));
    };
    let fetched = fetch_scan(&agent, &source, &listing, &cache)?;
    let received: DateTime<Utc> = Utc::now();
    let mut bands: BTreeMap<String, Vec<FetchedFile>> = BTreeMap::new();
    let mut companions: BTreeMap<String, Vec<FetchedFile>> = BTreeMap::new();
    for (tag, d) in fetched {
        let row = FetchedFile {
            key: d.object.key.clone(),
            path: d.path.to_string_lossy().to_string(),
            bytes: d.object.size_bytes,
            last_modified: d.object.last_modified.clone(),
            cache_hit: d.cache_hit,
        };
        if let Some(b) = tag.strip_prefix("band:") {
            bands.entry(b.to_string()).or_default().push(row);
        } else if let Some(c) = tag.strip_prefix("companion:") {
            companions.entry(c.to_string()).or_default().push(row);
        }
    }
    for v in bands.values_mut() {
        v.sort_by(|a, b| a.key.cmp(&b.key));
    }
    let record = GeoFetchRecord {
        schema: GEO_FETCH_SCHEMA,
        status: "READY",
        satellite: source.satellite.clone(),
        instrument: source.instrument.clone(),
        sector: source.sector.clone(),
        bucket: source.bucket.clone(),
        scan_nominal: listing.scan_nominal.clone(),
        mode,
        requested: target.format("%Y-%m-%dT%H:%M:%SZ").to_string(),
        bands,
        companions,
        received_utc: received.format("%Y-%m-%dT%H:%M:%SZ").to_string(),
        scans_examined: examined,
    };
    Ok(format!("{}\n", serde_json::to_string_pretty(&record)?))
}

// ---------------------------------------------------------------------------
// geo-bt
// ---------------------------------------------------------------------------

fn is_ami(path: &Path) -> bool {
    path.file_name()
        .map(|n| n.to_string_lossy().to_ascii_lowercase())
        .is_some_and(|n| n.starts_with("gk2a_ami_") && n.ends_with(".nc"))
}

fn decode(o: &GeoOptions) -> Result<GeoImagerScan, Box<dyn Error>> {
    if o.rad.is_empty() {
        return Err(boxed_error("--rad is required (an AMI L1B file, or the HSD segments of one AHI band)"));
    }
    if o.rad.iter().all(|p| is_ami(p)) {
        if o.rad.len() != 1 {
            return Err(boxed_error("an AMI band is one file: give --rad once"));
        }
        let path = &o.rad[0];
        let window = match (o.window, o.bbox) {
            (Some(w), _) => Some(w),
            (None, Some(b)) => {
                let nav = read_ami_navigation(path)?;
                Some(window_for_bbox(&nav.projection, &nav.grid, b).ok_or_else(|| {
                    boxed_error(format!("--bbox {b:?} is not in view of {}", path.display()))
                })?)
            }
            (None, None) => None,
        };
        // The channel's centroid from the source table (see read_ami_l1b for the measured reason).
        let nav_file = rw_sat::netcdf::open_goes_netcdf_lossy(path)?;
        let satellite = nav_file
            .attribute("satellite_name")
            .and_then(|a| a.as_string().map(|s| s.trim().to_ascii_uppercase().replace('-', "")))
            .ok_or_else(|| boxed_error(format!("{}: no satellite_name", path.display())))?;
        let channel = nav_file
            .variable("image_pixel_values")
            .and_then(|v| v.attribute("channel_name").and_then(|a| a.as_string().map(|s| s.trim().to_ascii_uppercase())))
            .ok_or_else(|| boxed_error(format!("{}: no channel_name", path.display())))?;
        let band = rw_sat::ami::AMI_CHANNELS
            .iter()
            .find(|(n, _)| *n == channel)
            .map(|(_, b)| *b)
            .ok_or_else(|| boxed_error(format!("unknown AMI channel {channel}")))?;
        let centroid = find_source(&satellite, "F")?
            .band_centroid_wavenumber_cm1
            .get(&band.to_string())
            .copied();
        return read_ami_l1b(path, window, centroid);
    }
    let scan = read_ahi_segments(&o.rad)?;
    let window = match (o.window, o.bbox) {
        (Some(w), _) => Some(w),
        (None, Some(b)) => Some(scan.window_for_bbox(b).ok_or_else(|| {
            boxed_error(format!("--bbox {b:?} is not in view of the given segments"))
        })?),
        (None, None) => None,
    };
    match window {
        Some(w) => scan.crop(w),
        None => Ok(scan),
    }
}

fn source_entries(scan: &GeoImagerScan) -> Result<Vec<SourceEntry>, Box<dyn Error>> {
    let mut out = Vec::new();
    for path in &scan.sources {
        let bytes = std::fs::read(path).map_err(|e| boxed_error(format!("{}: {e}", path.display())))?;
        out.push(SourceEntry {
            product: format!("RAD{:02}", scan.band),
            filename: path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default(),
            bytes: bytes.len(),
            sha256: hex_sha256(&bytes),
            dqf_rule: "enumerated".to_string(),
            condemn_mask: None,
            dqf: DqfRow::default(),
            dqf_plane: "rad_dqf".to_string(),
        });
    }
    Ok(out)
}

pub fn cmd_geo_bt(o: &GeoOptions) -> Result<String, Box<dyn Error>> {
    let out = o.out.as_deref().ok_or_else(|| boxed_error("--out is required (the pack destination)"))?;
    if out.is_dir() {
        return Err(boxed_error(format!("--out {} is a directory; give the pack file path", out.display())));
    }
    let scan = decode(o)?;
    if let Some(b) = o.bands.first()
        && *b != scan.band
    {
        return Err(boxed_error(format!("--bands {b} but the file is band {}", scan.band)));
    }
    let (nx, ny) = (scan.fixed_grid.nx, scan.fixed_grid.ny);
    let shape = vec![ny, nx];
    let (lat, lon) = scan.lat_lon_mesh();
    let mut counts = BtCounts { total: nx * ny, ..BtCounts::default() };
    for ((&r, &q), &t) in scan.radiance.iter().zip(&scan.dqf).zip(&scan.bt) {
        if q != 0.0 {
            counts.dqf_bad += 1;
        }
        if !r.is_finite() {
            counts.rad_missing += 1;
        } else if r <= 0.0 {
            counts.rad_nonpositive += 1;
        }
        if t.is_finite() {
            counts.finite += 1;
        }
    }
    let mut sources = source_entries(&scan)?;
    let dqf_row = DqfRow {
        total: counts.total,
        primary_missing: counts.rad_missing,
        dqf_missing: 0,
        dqf_bad: counts.dqf_bad,
        masked: counts.total - counts.finite - counts.rad_missing - counts.rad_nonpositive,
        finite: counts.finite,
    };
    // The window's counts sit on the first source; the others carry their identity only.
    if let Some(first) = sources.first_mut() {
        first.dqf = dqf_row;
    }
    let mut builder = PayloadBuilder::new();
    let mut planes = BTreeMap::new();
    let mut order = Vec::new();
    for (name, values) in [
        ("bt", &scan.bt),
        ("rad", &scan.radiance),
        ("lat", &lat),
        ("lon", &lon),
        ("rad_dqf", &scan.dqf),
    ] {
        let key = builder.push_f32(values, shape.clone());
        planes.insert(name.to_string(), key);
        order.push(name.to_string());
    }
    let (payload, arrays) = builder.finish();
    let iso = |t: DateTime<Utc>| crate::iso8601(t);
    let dataset_name = scan
        .sources
        .first()
        .and_then(|p| p.file_name())
        .map(|n| n.to_string_lossy().to_string());
    let meta = BtMeta {
        schema: BT_SCHEMA.to_string(),
        status: "READY".to_string(),
        satellite: scan.satellite.clone(),
        sector: scan.sector.clone(),
        band: scan.band,
        scan_start: iso(scan.start_time_utc),
        scan_end: iso(scan.end_time_utc),
        provenance: ProvenanceRow {
            dataset_name,
            time_coverage_start: Some(iso(scan.start_time_utc)),
            time_coverage_end: Some(iso(scan.end_time_utc)),
            platform_id: Some(scan.satellite.clone()),
            processing_level: Some("L1b".to_string()),
            received_utc: o.received_utc.clone(),
            row_time_model: GEO_ROW_TIME_MODEL.to_string(),
            ..ProvenanceRow::default()
        },
        sources,
        window: scan.window,
        projection: ProjectionEntry {
            perspective_point_height_m: scan.projection.perspective_point_height_m,
            semi_major_axis_m: scan.projection.semi_major_axis_m,
            semi_minor_axis_m: scan.projection.semi_minor_axis_m,
            longitude_of_projection_origin_deg: scan.projection.longitude_of_projection_origin_deg,
            sweep_angle_axis: scan.projection.sweep_angle_axis.as_str().to_string(),
        },
        nx,
        ny,
        x_scan_rad: scan.fixed_grid.x_scan_rad.clone(),
        y_scan_rad: scan.fixed_grid.y_scan_rad.clone(),
        planes,
        plane_order: order,
        arrays,
        payload_bytes: payload.len(),
        content_sha256: hex_sha256(&payload),
        planck: None,
        brightness_temperature_formula: scan.calibration["formula"].as_str().unwrap_or_default().to_string(),
        dqf_policy: "per-pixel quality 0 good, 1 conditionally usable, 2 outside the scan, 3 error; only 0 keeps a \
                     brightness temperature"
            .to_string(),
        counts,
        clear_sky_mask: None,
        cross_check: None,
        instrument: Some(scan.instrument.clone()),
        channel_name: Some(scan.channel_name.clone()),
        central_wavelength_um: Some(scan.central_wavelength_um),
        calibration: Some(scan.calibration.clone()),
        row_time_unix_s: Some(scan.row_time_unix_s.clone()),
        radiance_units: Some(scan.radiance_units.clone()),
    };
    let pack_bytes = write_container(out, &meta, &payload)?;
    #[derive(Serialize)]
    struct Record<'a> {
        schema: &'static str,
        status: &'static str,
        path: String,
        pack_schema: &'a str,
        bytes: usize,
        content_sha256: &'a str,
        instrument: &'a str,
        satellite: &'a str,
        sector: &'a str,
        band: u8,
        channel_name: &'a str,
        scan_start: &'a str,
        scan_end: &'a str,
        nx: usize,
        ny: usize,
        window: Option<[usize; 4]>,
        counts: &'a BtCounts,
        bt_min: f32,
        bt_max: f32,
        files: usize,
    }
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for &t in &scan.bt {
        if t.is_finite() {
            lo = lo.min(t);
            hi = hi.max(t);
        }
    }
    let record = Record {
        schema: BT_BUILD_SCHEMA,
        status: "READY",
        path: out.to_string_lossy().to_string(),
        pack_schema: &meta.schema,
        bytes: pack_bytes,
        content_sha256: &meta.content_sha256,
        instrument: &scan.instrument,
        satellite: &meta.satellite,
        sector: &meta.sector,
        band: meta.band,
        channel_name: &scan.channel_name,
        scan_start: &meta.scan_start,
        scan_end: &meta.scan_end,
        nx,
        ny,
        window: meta.window,
        counts: &meta.counts,
        bt_min: lo,
        bt_max: hi,
        files: scan.sources.len(),
    };
    Ok(format!("{}\n", serde_json::to_string_pretty(&record)?))
}
