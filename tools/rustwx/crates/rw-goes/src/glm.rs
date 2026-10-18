//! `rw_goes glm` -- GOES GLM flash extent density (FED) on a local grid for
//! one valid time, in the GLM-pixel definition of Kong et al. (2020, 2022,
//! 2025):
//!
//! 1. every 20 s GLM L2 LCFA granule whose start lies in `[W - A/2, W + A/2)`
//!    is read (`A` = `--accumulation-minutes`, default 5), and fewer than
//!    `round(3 A)` granules is a refusal naming the count: a gap in the
//!    20 s stream lowers every rate by the same factor and would read as
//!    weak storms;
//! 2. flashes whose `flash_quality_flag` is not 0 are dropped with every
//!    event of theirs (`--no-quality` keeps them);
//! 3. on a local equidistant grid of `--pixel-km` cells (default 10, the
//!    ~10 km GLM pixel of the papers above) centred on `--center`, a cell's
//!    FED is the number of DISTINCT flashes with at least one event in it,
//!    divided by `A`: flashes per minute per pixel;
//! 4. every lit cell inside `--radius-km` is one row at the cell centre.
//!
//! The event decode follows `rusty-weather`'s `rw-glm` granule decoder
//! (FahrenheitResearch/rusty-weather, MIT, `crates/rw-glm/src/granule.rs`:
//! the same netcrust lossy open, raw reads and scale/offset), extended from
//! flashes to events because flash EXTENT density counts every cell a
//! flash's events touch, not the flash centroid.  Event latitude and
//! longitude are unpacked in single precision (`raw * scale + offset`, two
//! roundings, unsigned when `_Unsigned` says so), which is the arithmetic
//! netCDF4-python applies, so a Python reader of the same granules puts a
//! cell boundary on the same side.
//!
//! ```text
//! rw_goes glm --time 2026-10-16T08:00:00Z --center 32.573,-97.303 \
//!             --radius-km 420 --out fed.csv [--satellite auto|G16|G18|G19] \
//!             [--accumulation-minutes 5] [--pixel-km 10] [--cache DIR] \
//!             [--bucket NAME] [--files F,F,...] [--no-quality]
//! ```
//!
//! Output: `fed.csv` (one row per lit cell: `latitude_deg, longitude_deg,
//! fed_per_min_per_pixel, flash_count`, in grid row-major order) and the
//! `gpuwm-obs.glm-fed.v2` record `fed.json` beside it (granules with their
//! SHA-256, bucket LastModified and cache state; event, flash and cell
//! counters), also printed on stdout.

use super::*;

use std::collections::{HashMap, HashSet};

pub const GLM_FED_SCHEMA: &str = "gpuwm-obs.glm-fed.v2";
const PRODUCT: &str = "GLM-L2-LCFA";
const EARTH_R_KM: f64 = 6371.0;
/// GOES-19 became GOES-East on this day; `--satellite auto` takes GOES-16
/// before it.
const GOES19_EAST: (i32, u32, u32) = (2025, 4, 7);

#[derive(Debug)]
struct GlmOptions {
    time: Option<DateTime<Utc>>,
    center: Option<(f64, f64)>,
    radius_km: Option<f64>,
    out: Option<PathBuf>,
    satellite: String,
    bucket: Option<String>,
    cache: Option<PathBuf>,
    files: Vec<PathBuf>,
    minutes: f64,
    pixel_km: f64,
    quality: bool,
}

impl Default for GlmOptions {
    fn default() -> Self {
        GlmOptions {
            time: None,
            center: None,
            radius_km: None,
            out: None,
            satellite: "auto".to_string(),
            bucket: None,
            cache: None,
            files: Vec::new(),
            minutes: 5.0,
            pixel_km: 10.0,
            quality: true,
        }
    }
}

fn positive(name: &str, text: &str) -> Result<f64, Box<dyn Error>> {
    let v: f64 = text
        .trim()
        .parse()
        .map_err(|_| boxed_error(format!("{name} {text:?} is not a number")))?;
    if !(v.is_finite() && v > 0.0) {
        return Err(boxed_error(format!("{name} must be a positive number, got {text}")));
    }
    Ok(v)
}

impl GlmOptions {
    fn parse(args: &[String]) -> Result<Self, Box<dyn Error>> {
        let mut o = GlmOptions::default();
        let mut i = 0;
        while i < args.len() {
            let flag = args[i].as_str();
            let value = || -> Result<String, Box<dyn Error>> {
                args.get(i + 1)
                    .cloned()
                    .ok_or_else(|| boxed_error(format!("{flag} needs a value")))
            };
            match flag {
                "--time" => o.time = Some(parse_time(&value()?)?),
                "--center" => {
                    let text = value()?;
                    let parts: Vec<f64> = text
                        .split(',')
                        .map(|p| p.trim().parse::<f64>())
                        .collect::<Result<_, _>>()
                        .map_err(|_| boxed_error(format!("--center {text:?} is not LAT,LON")))?;
                    if parts.len() != 2 || parts[0].abs() > 90.0 || parts[1].abs() > 360.0 {
                        return Err(boxed_error(format!("--center {text:?} must be LAT,LON in degrees")));
                    }
                    o.center = Some((parts[0], parts[1]));
                }
                "--radius-km" => o.radius_km = Some(positive(flag, &value()?)?),
                "--out" => o.out = Some(PathBuf::from(value()?)),
                "--satellite" => o.satellite = value()?,
                "--bucket" => o.bucket = Some(value()?),
                "--cache" => o.cache = Some(PathBuf::from(value()?)),
                "--files" => {
                    for part in value()?.split(',').filter(|p| !p.is_empty()) {
                        o.files.push(PathBuf::from(part));
                    }
                }
                "--accumulation-minutes" => o.minutes = positive(flag, &value()?)?,
                "--pixel-km" => o.pixel_km = positive(flag, &value()?)?,
                "--no-quality" => {
                    o.quality = false;
                    i += 1;
                    continue;
                }
                other => return Err(boxed_error(format!("glm: unknown option {other:?}\n\n{USAGE}"))),
            }
            i += 2;
        }
        Ok(o)
    }
}

/// `G19` / `G16` for `auto` by date, else the token as given.
fn satellite_for(token: &str, when: DateTime<Utc>) -> String {
    if token.eq_ignore_ascii_case("auto") {
        let (y, m, d) = GOES19_EAST;
        let east = Utc.with_ymd_and_hms(y, m, d, 0, 0, 0).single().expect("valid date");
        if when >= east { "G19".to_string() } else { "G16".to_string() }
    } else {
        token.to_ascii_uppercase()
    }
}

/// The start instant in a GLM granule name (`_sYYYYJJJHHMMSSt_`).
pub(crate) fn granule_start(name: &str) -> Option<DateTime<Utc>> {
    let at = name.find("_s")? + 2;
    let digits = name.get(at..at + 14)?;
    if !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let n = |a: usize, b: usize| digits[a..b].parse::<i64>().ok();
    let (year, doy, hour, minute, second, tenth) = (n(0, 4)?, n(4, 7)?, n(7, 9)?, n(9, 11)?, n(11, 13)?, n(13, 14)?);
    let jan1 = Utc.with_ymd_and_hms(year as i32, 1, 1, 0, 0, 0).single()?;
    Some(
        jan1 + chrono::Duration::days(doy - 1)
            + chrono::Duration::hours(hour)
            + chrono::Duration::minutes(minute)
            + chrono::Duration::seconds(second)
            + chrono::Duration::milliseconds(tenth * 100),
    )
}

/// One granule's events: (lat, lon, flash key) per kept event, and the
/// number of distinct flashes dropped for quality.
struct GranuleEvents {
    lat: Vec<f64>,
    lon: Vec<f64>,
    flash: Vec<i64>,
    dropped: usize,
}

fn read_events(path: &Path, index: usize, quality: bool) -> Result<GranuleEvents, Box<dyn Error>> {
    let file = rw_sat::netcdf::open_goes_netcdf_lossy(path)
        .map_err(|e| boxed_error(format!("{}: not a readable NetCDF: {e}", path.display())))?;
    // The reader's type is rw-sat's netcrust handle, reached through
    // closures so this crate needs no second dependency on netcrust.
    let raw_values = |name: &str| -> Result<Vec<f64>, Box<dyn Error>> {
        let variable = file.variable(name).ok_or_else(|| {
            boxed_error(format!("{}: missing variable `{name}` (not a GLM L2 LCFA granule?)", path.display()))
        })?;
        variable
            .values_f64()
            .map_err(|e| boxed_error(format!("{}: reading `{name}`: {e}", path.display())))
    };
    // A packed coordinate unpacked the way netCDF4-python unpacks it:
    // unsigned when `_Unsigned` is true, then
    // `f32(raw) * scale + offset` in single precision (two roundings).
    let unpacked_f32 = |name: &str| -> Result<Vec<f64>, Box<dyn Error>> {
        let variable = file
            .variable(name)
            .ok_or_else(|| boxed_error(format!("{}: missing variable `{name}`", path.display())))?;
        let scale = variable.attribute("scale_factor").and_then(|a| a.as_f64());
        let offset = variable.attribute("add_offset").and_then(|a| a.as_f64());
        let unsigned = variable
            .attribute("_Unsigned")
            .and_then(|a| a.as_string().map(|s| s.trim().eq_ignore_ascii_case("true")))
            .unwrap_or(false);
        let wrap = |r: f64| if unsigned && r < 0.0 { r + 65536.0 } else { r };
        let fill = variable.attribute("_FillValue").and_then(|a| a.as_f64()).map(wrap);
        let raw = variable
            .values_f64()
            .map_err(|e| boxed_error(format!("{}: reading `{name}`: {e}", path.display())))?;
        Ok(raw
            .into_iter()
            .map(|r| {
                let r = wrap(r);
                if fill.is_some_and(|f| f == r) {
                    return f64::NAN;
                }
                match (scale, offset) {
                    (Some(s), Some(o)) if s != 1.0 || o != 0.0 => ((r as f32) * (s as f32) + (o as f32)) as f64,
                    (Some(s), None) if s != 1.0 => ((r as f32) * (s as f32)) as f64,
                    _ => r,
                }
            })
            .collect())
    };
    let n_events = file
        .dimension("number_of_events")
        .ok_or_else(|| boxed_error(format!("{}: no `number_of_events` dimension (not a GLM L2 LCFA granule?)", path.display())))?
        .len();
    let empty = GranuleEvents { lat: Vec::new(), lon: Vec::new(), flash: Vec::new(), dropped: 0 };
    if n_events == 0 {
        return Ok(empty);
    }
    let lat = unpacked_f32("event_lat")?;
    let lon = unpacked_f32("event_lon")?;
    let parent_group = raw_values("event_parent_group_id")?;
    let group_id = raw_values("group_id")?;
    let group_flash = raw_values("group_parent_flash_id")?;
    if lat.len() != n_events || lon.len() != n_events || parent_group.len() != n_events || group_id.len() != group_flash.len() {
        return Err(boxed_error(format!("{}: event or group arrays disagree in length", path.display())));
    }
    let group_to_flash: HashMap<i64, i64> = group_id
        .iter()
        .zip(&group_flash)
        .map(|(g, f)| (*g as i64, *f as i64))
        .collect();
    let mut bad: HashSet<i64> = HashSet::new();
    if quality && file.variable("flash_quality_flag").is_some() {
        let flash_id = raw_values("flash_id")?;
        let flags = raw_values("flash_quality_flag")?;
        for (id, flag) in flash_id.iter().zip(&flags) {
            if *flag != 0.0 {
                bad.insert(*id as i64);
            }
        }
    }
    let mut out = GranuleEvents { dropped: bad.len(), ..empty };
    for k in 0..n_events {
        let Some(&flash) = group_to_flash.get(&(parent_group[k] as i64)) else { continue };
        if bad.contains(&flash) || !lat[k].is_finite() || !lon[k].is_finite() {
            continue;
        }
        out.lat.push(lat[k]);
        out.lon.push(lon[k]);
        // Flash ids restart per granule; the granule index keeps them apart.
        out.flash.push(flash + ((index as i64) << 32));
    }
    Ok(out)
}

/// The local equidistant grid the FED is counted on.
struct LocalGrid {
    lat0: f64,
    lon0: f64,
    radius: f64,
    cell: f64,
    c0: f64,
    n: usize,
}

impl LocalGrid {
    fn new(lat0: f64, lon0: f64, radius: f64, cell: f64) -> Self {
        let n = (2.0 * radius / cell).ceil() as usize;
        LocalGrid { lat0, lon0, radius, cell, c0: lat0.to_radians().cos(), n }
    }
    fn xy(&self, lat: f64, lon: f64) -> (f64, f64) {
        let dlon = (lon - self.lon0 + 180.0).rem_euclid(360.0) - 180.0;
        (dlon.to_radians() * EARTH_R_KM * self.c0, (lat - self.lat0).to_radians() * EARTH_R_KM)
    }
    fn latlon(&self, x: f64, y: f64) -> (f64, f64) {
        (self.lat0 + (y / EARTH_R_KM).to_degrees(), self.lon0 + (x / (EARTH_R_KM * self.c0)).to_degrees())
    }
    fn centre(&self, i: usize) -> f64 {
        (i as f64 + 0.5) * self.cell - self.radius
    }
}

/// numpy's default (linear) percentile of sorted values.
fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = q / 100.0 * (sorted.len() - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    sorted[lo] + (sorted[hi] - sorted[lo]) * (rank - lo as f64)
}

struct Granule {
    path: PathBuf,
    name: String,
    last_modified: Option<String>,
    cache_hit: Option<bool>,
}

pub fn cmd_glm(args: &[String]) -> Result<String, Box<dyn Error>> {
    let o = GlmOptions::parse(args)?;
    let when = o.time.ok_or_else(|| boxed_error("glm needs --time (the valid time W)"))?;
    let (lat0, lon0) = o.center.ok_or_else(|| boxed_error("glm needs --center LAT,LON"))?;
    let radius = o.radius_km.ok_or_else(|| boxed_error("glm needs --radius-km"))?;
    let out = o.out.clone().ok_or_else(|| boxed_error("glm needs --out FILE.csv"))?;
    let half = chrono::Duration::milliseconds((o.minutes * 30_000.0).round() as i64);
    let (lo, hi) = (when - half, when + half);
    let satellite = satellite_for(&o.satellite, when);
    let mut bucket = o.bucket.clone().unwrap_or_default();

    // The granules: given, or listed and fetched from the open-data bucket.
    let mut granules: Vec<Granule> = Vec::new();
    if !o.files.is_empty() {
        for path in &o.files {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or_default().to_string();
            granules.push(Granule { path: path.clone(), name, last_modified: None, cache_hit: None });
        }
    } else {
        if bucket.is_empty() {
            bucket = rw_sat::s3::bucket_for_satellite(&satellite)?;
        }
        let cache = o.cache.clone().unwrap_or_else(|| PathBuf::from(DEFAULT_CACHE_DIR));
        let agent = rw_sat::s3::build_agent();
        let mut hour = lo.with_minute(0).and_then(|t| t.with_second(0)).and_then(|t| t.with_nanosecond(0)).unwrap_or(lo);
        let mut listed = Vec::new();
        while hour < hi {
            let prefix = rw_sat::s3::goes_hour_prefix(PRODUCT, hour);
            listed.extend(rw_sat::s3::list_s3_objects(&agent, &bucket, &prefix, None)?);
            hour += chrono::Duration::hours(1);
        }
        for object in listed {
            let name = rw_sat::s3::object_filename(&object.key).to_string();
            if !granule_start(&name).is_some_and(|s| lo <= s && s < hi) {
                continue;
            }
            let got = rw_sat::s3::download_object(&agent, &bucket, &cache, &object, true)?;
            granules.push(Granule {
                path: got.path,
                name,
                last_modified: Some(object.last_modified.clone()),
                cache_hit: Some(got.cache_hit),
            });
        }
    }
    granules.retain(|g| granule_start(&g.name).is_some_and(|s| lo <= s && s < hi));
    granules.sort_by(|a, b| a.path.cmp(&b.path));
    let expected = (o.minutes * 3.0).round_ties_even() as usize;
    if granules.len() < expected {
        return Err(boxed_error(format!(
            "{}: {} GLM granules start in [{}, {}) on {satellite}, expected {expected} (one per 20 s): \
             a gap in the stream lowers every rate by the same factor and would read as weak storms",
            iso8601(when), granules.len(), iso8601(lo), iso8601(hi)
        )));
    }

    // Events of every granule, flash keys kept apart per granule.
    let (mut lat, mut lon, mut flash, mut dropped) = (Vec::new(), Vec::new(), Vec::new(), 0usize);
    let mut sources = Vec::new();
    for (index, g) in granules.iter().enumerate() {
        let events = read_events(&g.path, index, o.quality)?;
        let bytes = std::fs::read(&g.path)?;
        sources.push(serde_json::json!({
            "name": g.name,
            "path": g.path.display().to_string(),
            "bytes": bytes.len(),
            "sha256": hex_sha256(&bytes),
            "last_modified": g.last_modified,
            "cache_hit": g.cache_hit,
            "events_kept": events.lat.len(),
        }));
        dropped += events.dropped;
        lat.extend(events.lat);
        lon.extend(events.lon);
        flash.extend(events.flash);
    }

    // Distinct (cell, flash) pairs per cell.
    let grid = LocalGrid::new(lat0, lon0, radius, o.pixel_km);
    let n = grid.n;
    let mut pairs: HashSet<(usize, i64)> = HashSet::new();
    for k in 0..lat.len() {
        let (x, y) = grid.xy(lat[k], lon[k]);
        let ix = ((x + grid.radius) / grid.cell).floor();
        let iy = ((y + grid.radius) / grid.cell).floor();
        if ix < 0.0 || iy < 0.0 || ix >= n as f64 || iy >= n as f64 {
            continue;
        }
        pairs.insert((iy as usize * n + ix as usize, flash[k]));
    }
    let mut count = vec![0i64; n * n];
    for (cell, _) in &pairs {
        count[*cell] += 1;
    }
    let lit_any: Vec<bool> = count.iter().map(|c| *c > 0).collect();
    let inside: Vec<bool> = (0..n * n)
        .map(|c| grid.centre(c % n).hypot(grid.centre(c / n)) <= radius)
        .collect();
    let mut lit_rows = Vec::new();
    for c in 0..n * n {
        let (iy, ix) = (c / n, c % n);
        let (cx, cy) = (grid.centre(ix), grid.centre(iy));
        if lit_any[c] && inside[c] {
            lit_rows.push((cx, cy, count[c] as f64 / o.minutes, count[c]));
        }
    }

    let mut text = String::from("latitude_deg,longitude_deg,fed_per_min_per_pixel,flash_count\n");
    for (cx, cy, rate, flashes) in &lit_rows {
        let (la, lo_) = grid.latlon(*cx, *cy);
        text.push_str(&format!("{la},{lo_},{rate},{flashes}\n"));
    }
    if let Some(parent) = out.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    std::fs::write(&out, &text)?;

    let mut lit_values: Vec<f64> = lit_rows.iter().map(|r| r.2).collect();
    lit_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let distinct_flashes: HashSet<i64> = flash.iter().copied().collect();
    let record = serde_json::json!({
        "schema": GLM_FED_SCHEMA,
        "status": "READY",
        "definition": "GLM FED, GLM-pixel definition (Kong et al. 2020, 2022, 2025)",
        "processing": format!(
            "GLM L2 LCFA (flash_quality_flag 0{}) -> flash extent density on a {} km local grid: distinct \
             flashes with an event in the cell / {} min centred on the valid time; lit cells only",
            if o.quality { "" } else { ": not applied, --no-quality" }, o.pixel_km, o.minutes),
        "valid_time": iso8601(when),
        "window": [iso8601(lo), iso8601(hi)],
        "satellite": satellite,
        "bucket": if bucket.is_empty() { None } else { Some(bucket.clone()) },
        "center_deg": [lat0, lon0],
        "radius_km": radius,
        "pixel_km": o.pixel_km,
        "grid_cells_per_side": n,
        "accumulation_minutes": o.minutes,
        "units": "flashes min-1 pixel-1",
        "granules": granules.len(),
        "granules_expected": expected,
        "sources": sources,
        "events": lat.len(),
        "flashes": distinct_flashes.len(),
        "flashes_dropped_quality": dropped,
        "lit_cells": lit_rows.len(),
        "fed_max_per_min": lit_values.last().copied().unwrap_or(0.0),
        "lit_mean": if lit_values.is_empty() { 0.0 } else { lit_values.iter().sum::<f64>() / lit_values.len() as f64 },
        "lit_p90": percentile(&lit_values, 90.0),
        "table": {"path": out.display().to_string(), "rows": lit_rows.len(),
                  "sha256": hex_sha256(text.as_bytes()), "bytes": text.len()},
        "written_utc": iso8601(Utc::now()),
    });
    let rendered = serde_json::to_string_pretty(&record)? + "\n";
    std::fs::write(out.with_extension("json"), &rendered)?;
    Ok(rendered)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn granule_start_reads_the_s_token() {
        let t = granule_start("OR_GLM-L2-LCFA_G19_s20262890759200_e20262890759400_c20262890759418.nc").unwrap();
        assert_eq!(t, Utc.with_ymd_and_hms(2026, 10, 16, 7, 59, 20).unwrap());
        assert!(granule_start("not-a-granule.nc").is_none());
    }

    #[test]
    fn auto_satellite_follows_the_goes_east_handover() {
        assert_eq!(satellite_for("auto", Utc.with_ymd_and_hms(2025, 4, 6, 23, 0, 0).unwrap()), "G16");
        assert_eq!(satellite_for("auto", Utc.with_ymd_and_hms(2026, 10, 16, 8, 0, 0).unwrap()), "G19");
        assert_eq!(satellite_for("g18", Utc::now()), "G18");
    }

    #[test]
    fn the_local_grid_is_centred_and_equidistant() {
        let g = LocalGrid::new(32.573, -97.303, 420.0, 10.0);
        assert_eq!(g.n, 84);
        assert_eq!(g.centre(0), -415.0);
        let (x, y) = g.xy(32.573, -97.303);
        assert_eq!((x, y), (0.0, 0.0));
        // Longitude wraps the short way round.
        let (x, _) = g.xy(32.573, 262.697);
        assert!(x.abs() < 1e-9);
        let (la, lo) = g.latlon(0.0, 0.0);
        assert_eq!((la, lo), (32.573, -97.303));
    }

    #[test]
    fn percentile_is_numpys_linear_rule() {
        assert_eq!(percentile(&[1.0, 2.0, 3.0, 4.0], 90.0), 3.7);
        assert_eq!(percentile(&[], 90.0), 0.0);
    }

    #[test]
    fn options_parse_and_refuse() {
        let a = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let o = GlmOptions::parse(&a(&["--time", "2026-10-16T08:00:00Z", "--center", "32.5,-97.3", "--radius-km", "420",
                                        "--out", "f.csv", "--no-quality", "--pixel-km", "8"])).unwrap();
        assert!(!o.quality);
        assert_eq!(o.pixel_km, 8.0);
        assert_eq!(o.minutes, 5.0);
        assert!(GlmOptions::parse(&a(&["--center", "95,0"])).is_err());
        assert!(GlmOptions::parse(&a(&["--pixel-km", "-1"])).is_err());
        // The observation-preparation options (zero-FED cells, errors) are
        // not this reader's: they are refused as unknown, not ignored.
        assert!(GlmOptions::parse(&a(&["--zero-every", "2"])).is_err());
        assert!(GlmOptions::parse(&a(&["--error", "0.5"])).is_err());
    }
}
