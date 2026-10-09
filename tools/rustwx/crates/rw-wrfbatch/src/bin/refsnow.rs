//! `rw_refsnow`: 10:1 snowfall over a forecast window, WOOF beside the raw
//! global models that drive it, on WOOF's own grid and one colour table.
//!
//! ```text
//! rw_refsnow --store-root DIR --out PNG --title TEXT --subtitle TEXT
//!            --start-lead H --end-lead H
//!            [--woof LABEL=WRFOUT]... [--gfs LABEL=DIR]... [--ecmwf LABEL=DIR]...
//!            [--columns N] [--width N] [--height N]
//! rw_refsnow --inventory FILE.grib2
//! ```
//!
//! * `--woof`: a WOOF frame at the window's END, imported with the importer's
//!   window selected by --start-lead and --end-lead, drawn
//!   from `wrf_snow_10to1_window`.  The first `--woof` frame is the grid every
//!   panel is drawn on.
//! * `--gfs`: a directory of GFS 0.25 deg GRIB2 files holding, for every 6 h
//!   step ending in the window, the 6 h APCP bucket and the 6 h AVERAGE of
//!   the categorical snow flag (`gfs.tCCz.pgrb2.0p25.fNNN` subsets).  GFS
//!   10:1 snowfall = 10 x sum(APCP_6h x CSNOW_6h_avg): the frozen share of
//!   each bucket's precipitation, as the GFS itself typed it.
//! * `--ecmwf`: a directory of ECMWF open-data GRIB2 files (`*-<lead>h-*`)
//!   holding the accumulated snowfall water equivalent `sf` at the window's
//!   start and end leads.  ECMWF 10:1 = 10 x (sf_end - sf_start).
//!
//! Breakage it prevents: a WOOF snowfall forecast could not be laid beside
//! the raw GFS and ECMWF snowfall the forecaster's colleagues are reading, so
//! nobody could see where the high-resolution run adds snow (terrain) or
//! takes it away.  Reference fields are sampled at the nearest reference
//! point (`compare::match_grids`); nothing is interpolated or smoothed.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use grib_core::grib2::{Grib2File, Grib2Message, flip_rows, grid_latlon, unpack_message};
use rustwx_render::{LegendControls, RenderDensity};
use rustwx_core::{CanonicalField, FieldSelector};
use rusty_weather::render_all::StoreFieldSource;
use rw_wrfbatch::compare::{SheetHeader, compose_sheet_rows, match_grids, sample};
use rw_wrfbatch::panel::{PanelRequest, render_panel};
use rw_wrfbatch::wrf_process::{WrfProcessMessage, WrfProcessOptions, spawn_process_paths};

/// Embed the build revision for release artifact verification.
pub static GPUWM_BRIDGE_SOURCE_REV_STAMP: &str =
    concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));

enum Source {
    Woof(PathBuf),
    Gfs(PathBuf),
    Ecmwf(PathBuf),
}

struct Args {
    store_root: PathBuf,
    out: PathBuf,
    title: String,
    subtitle: String,
    start_lead: u32,
    end_lead: u32,
    panels: Vec<(String, Source)>,
    columns: usize,
    width: u32,
    height: u32,
}

fn usage() -> &'static str {
    "usage: rw_refsnow --store-root DIR --out PNG --title TEXT --subtitle TEXT --start-lead H --end-lead H \
[--woof LABEL=WRFOUT]... [--gfs LABEL=DIR]... [--ecmwf LABEL=DIR]... [--columns N] [--width N] [--height N]\n\
       rw_refsnow --inventory FILE.grib2"
}

fn parse_args() -> Result<Option<Args>, String> {
    let mut raw = std::env::args().skip(1);
    let mut store_root = None;
    let mut out = None;
    let mut title = String::from("10:1 snowfall");
    let mut subtitle = String::new();
    let mut start_lead = None;
    let mut end_lead = None;
    let mut panels = Vec::new();
    let mut columns = 2usize;
    let mut width = 900u32;
    let mut height = 700u32;
    let labelled = |text: String| -> Result<(String, PathBuf), String> {
        let (label, path) = text
            .split_once('=')
            .ok_or_else(|| format!("expected LABEL=PATH, got {text:?}"))?;
        Ok((label.to_string(), PathBuf::from(path)))
    };
    while let Some(flag) = raw.next() {
        let mut value = || raw.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--help" | "-h" => {
                println!("{}", usage());
                return Ok(None);
            }
            "--inventory" => {
                inventory(Path::new(&value()?))?;
                return Ok(None);
            }
            "--store-root" => store_root = Some(PathBuf::from(value()?)),
            "--out" => out = Some(PathBuf::from(value()?)),
            "--title" => title = value()?,
            "--subtitle" => subtitle = value()?,
            "--start-lead" => start_lead = Some(value()?.parse().map_err(|e| format!("--start-lead: {e}"))?),
            "--end-lead" => end_lead = Some(value()?.parse().map_err(|e| format!("--end-lead: {e}"))?),
            "--columns" => columns = value()?.parse().map_err(|e| format!("--columns: {e}"))?,
            "--width" => width = value()?.parse().map_err(|e| format!("--width: {e}"))?,
            "--height" => height = value()?.parse().map_err(|e| format!("--height: {e}"))?,
            "--woof" => {
                let (label, path) = labelled(value()?)?;
                panels.push((label, Source::Woof(path)));
            }
            "--gfs" => {
                let (label, path) = labelled(value()?)?;
                panels.push((label, Source::Gfs(path)));
            }
            "--ecmwf" => {
                let (label, path) = labelled(value()?)?;
                panels.push((label, Source::Ecmwf(path)));
            }
            other => return Err(format!("unknown argument {other:?}\n{}", usage())),
        }
    }
    let start_lead: u32 = start_lead.ok_or("--start-lead is required")?;
    let end_lead: u32 = end_lead.ok_or("--end-lead is required")?;
    if end_lead <= start_lead {
        return Err("--end-lead must be after --start-lead".into());
    }
    if !panels.iter().any(|(_, s)| matches!(s, Source::Woof(_))) {
        return Err("at least one --woof frame is required: it is the grid every panel is drawn on".into());
    }
    Ok(Some(Args {
        store_root: store_root.ok_or("--store-root is required")?,
        out: out.ok_or("--out is required")?,
        title,
        subtitle,
        start_lead,
        end_lead,
        panels,
        columns: columns.max(1),
        width,
        height,
    }))
}

fn read_grib(path: &Path) -> Result<Grib2File, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    Grib2File::from_bytes(&bytes).map_err(|e| format!("{}: {e}", path.display()))
}

fn lead_hours(message: &Grib2Message) -> Option<u32> {
    // Forecast time in hours (time-range unit 1) or minutes (unit 0).
    match message.product.time_range_unit {
        1 => Some(message.product.forecast_time),
        0 if message.product.forecast_time % 60 == 0 => Some(message.product.forecast_time / 60),
        _ => None,
    }
}

fn inventory(path: &Path) -> Result<(), String> {
    let grib = read_grib(path)?;
    for (index, m) in grib.messages.iter().enumerate() {
        let p = &m.product;
        println!(
            "{index}\tdisc={}\tcat={}\tnum={}\tpdt={}\tlevel={}:{}\tlead={:?}\tstat={:?}\trange_h={:?}\tdrt={}\tgrid={}x{}",
            m.discipline,
            p.parameter_category,
            p.parameter_number,
            p.template,
            p.level_type,
            p.level_value,
            lead_hours(m),
            p.statistical_process_type,
            p.statistical_time_range_hours(),
            m.data_rep.template,
            m.grid.nx,
            m.grid.ny
        );
    }
    Ok(())
}

/// One decoded reference plane, rows west-to-east, south or north first as
/// the coordinates say, longitudes in -180..180.
struct Plane {
    lat: Vec<f32>,
    lon: Vec<f32>,
    ny: usize,
    nx: usize,
    values: Vec<f32>,
}

fn decode(message: &Grib2Message) -> Result<Plane, String> {
    let (nx, ny) = (message.grid.nx as usize, message.grid.ny as usize);
    if nx == 0 || ny == 0 || message.grid.is_reduced || message.grid.scan_mode & 0x20 != 0 {
        return Err("reference grid is not a rectangular row-major scan".into());
    }
    let mut values = unpack_message(message).map_err(|e| e.to_string())?;
    let (mut lat, mut lon) = grid_latlon(&message.grid).map_err(|e| e.to_string())?;
    if values.len() != nx * ny || lat.len() != nx * ny || lon.len() != nx * ny {
        return Err(format!("reference values and coordinates disagree with {ny}x{nx}"));
    }
    if message.grid.scan_mode & 0x10 != 0 {
        for row in (1..ny).step_by(2) {
            values[row * nx..(row + 1) * nx].reverse();
        }
    }
    if message.grid.scan_mode & 0x40 != 0 {
        flip_rows(&mut values, nx, ny);
        flip_rows(&mut lat, nx, ny);
        flip_rows(&mut lon, nx, ny);
    }
    for value in &mut lon {
        *value = (*value + 180.0).rem_euclid(360.0) - 180.0;
    }
    for row in 0..ny {
        let range = row * nx..(row + 1) * nx;
        if let Some(index) = lon[range.clone()].windows(2).position(|pair| pair[1] - pair[0] < -180.0) {
            lat[range.clone()].rotate_left(index + 1);
            lon[range.clone()].rotate_left(index + 1);
            values[range].rotate_left(index + 1);
        }
    }
    Ok(Plane {
        lat: lat.iter().map(|v| *v as f32).collect(),
        lon: lon.iter().map(|v| *v as f32).collect(),
        ny,
        nx,
        values: values.iter().map(|v| if v.is_finite() && v.abs() < 1.0e20 { *v as f32 } else { f32::NAN }).collect(),
    })
}

fn same_grid(a: &Plane, b: &Plane) -> bool {
    a.nx == b.nx && a.ny == b.ny && a.values.len() == b.values.len()
        && a.lat == b.lat && a.lon == b.lon
}

fn gfs_frozen(rain: &Plane, snow: &Plane) -> Result<Vec<f32>, String> {
    if !same_grid(rain, snow) { return Err("APCP and CSNOW have different grids".into()); }
    Ok(rain.values.iter().zip(&snow.values).map(|(&p,&s)| {
        if p.is_finite() && p >= 0. && s.is_finite() && (0. ..=1.).contains(&s) {
            10. * p * s
        } else { f32::NAN }
    }).collect())
}

fn accumulation_end(message: &Grib2Message) -> Option<u32> {
    let start = lead_hours(message)?;
    let length = message.product.statistical_time_range_hours()?;
    let end = start.checked_add(u32::from(length))?;
    if message.product.end_of_interval != Some(message.reference_time + chrono::Duration::hours(end as i64)) {
        return None;
    }
    Some(end)
}

fn ecmwf_difference(end: Plane, end_scale: f32, start: Option<(Plane,f32)>) -> Result<Plane,String> {
    let values = if let Some((start,start_scale)) = start {
        if !same_grid(&end,&start) { return Err("ECMWF start and end have different grids".into()); }
        end.values.iter().zip(&start.values).map(|(&e,&s)| {
            let (amount,scale) = if end_scale == start_scale {
                // Preserve the source patch's f32 order on ordinary,
                // equal-unit accumulators; convert first only for mixed units.
                (e-s,end_scale)
            } else { (e*end_scale - s*start_scale,1.) };
            if e.is_finite() && s.is_finite() && e >= 0. && s >= 0. && amount >= 0. {
                10.*amount*scale
            } else { f32::NAN }
        }).collect()
    } else {
        end.values.iter().map(|&e| if e.is_finite() && e >= 0. {10.*e*end_scale} else {f32::NAN}).collect()
    };
    Ok(Plane { values, ..end })
}

fn grib_files(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| format!("read {}: {e}", dir.display()))?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|p| p.is_file())
        .collect();
    files.sort();
    Ok(files)
}

/// GFS 10:1 snowfall (mm of snow) over (start, end]: 10 x sum of the 6 h
/// APCP buckets times the matching 6 h average categorical snow flag.
fn gfs_window(dir: &Path, start: u32, end: u32, cycle: chrono::NaiveDateTime) -> Result<(Plane, String), String> {
    if start % 6 != 0 || end % 6 != 0 {
        return Err(format!("GFS buckets are 6 h; window {start}-{end} h is not on them"));
    }
    let mut total: Option<Plane> = None;
    let mut buckets = 0;
    let mut wanted: Vec<u32> = ((start + 6)..=end).step_by(6).collect();
    for file in grib_files(dir)? {
        let grib = match read_grib(&file) {
            Ok(g) => g,
            Err(_) => continue,
        };
        let six = |m: &&Grib2Message, cat: u8, num: u8, stat: u8| {
            m.discipline == 0
                && m.product.parameter_category == cat
                && m.product.parameter_number == num
                && m.product.level_type == 1
                && m.product.statistical_process_type == Some(stat)
                && m.product.statistical_time_range_hours() == Some(6)
        };
        let apcp = grib.messages.iter().find(|m| six(m, 1, 8, 1));
        let csnow = grib.messages.iter().find(|m| six(m, 1, 195, 0));
        let (Some(apcp), Some(csnow)) = (apcp, csnow) else { continue };
        // The bucket's END lead: the PDT forecast time is its START.
        let Some(bucket_start) = lead_hours(apcp) else { continue };
        let bucket_end = bucket_start + 6;
        let Some(position) = wanted.iter().position(|lead| *lead == bucket_end) else { continue };
        if apcp.reference_time != cycle {
            return Err(format!("{}: reference cycle differs from the WOOF run; snowfall windows would describe different forecasts",file.display()));
        }
        if lead_hours(csnow) != Some(bucket_start)
            || accumulation_end(apcp) != Some(bucket_end) || accumulation_end(csnow) != Some(bucket_end)
            || apcp.reference_time != csnow.reference_time {
            return Err(format!("{}: APCP and CSNOW buckets disagree", file.display()));
        }
        let rain = decode(apcp)?;
        let snow = decode(csnow)?;
        let frozen = gfs_frozen(&rain, &snow)?;
        match total.as_mut() {
            None => total = Some(Plane { values: frozen, ..rain }),
            Some(sum) => {
                if !same_grid(sum, &rain) {
                    return Err("GFS files on two grids".into());
                }
                for (a, b) in sum.values.iter_mut().zip(frozen.iter()) {
                    *a += *b;
                }
            }
        }
        wanted.remove(position);
        buckets += 1;
    }
    if !wanted.is_empty() {
        return Err(format!("GFS buckets missing for leads {wanted:?} h in {}", dir.display()));
    }
    let total = total.ok_or("no GFS bucket in the window")?;
    Ok((total, format!("GFS 0.25 deg, 10 x sum of {buckets} 6 h APCP x 6 h avg CSNOW")))
}

/// ECMWF 10:1 snowfall (mm of snow): 10 x the accumulated snowfall water
/// equivalent `sf` difference between the window's end and start leads.
/// `sf` is matched by its GRIB2 identity in the ECMWF open-data files
/// (discipline 0, category 1, integrated 53 or ECMWF local 198).
fn ecmwf_window(dir: &Path, start: u32, end: u32, cycle: chrono::NaiveDateTime) -> Result<(Plane, String), String> {
    let at = |lead: u32| -> Result<Option<(Plane, f32)>, String> {
        if lead == 0 {
            return Ok(None);
        }
        for file in grib_files(dir)? {
            let grib = match read_grib(&file) { Ok(g) => g, Err(_) => continue };
            let sf = grib.messages.iter().find(|m| {
                m.discipline == 0
                    && m.product.parameter_category == 1
                    && matches!(m.product.parameter_number, 53 | 198)
                    && m.product.level_type == 1
                    && lead_hours(m) == Some(0)
                    && accumulation_end(m) == Some(lead)
                    && m.product.statistical_process_type == Some(1)
            });
            let Some(sf) = sf else { continue };
            if sf.reference_time != cycle {
                return Err(format!("{}: reference cycle differs from the WOOF run; snowfall windows would describe different forecasts",file.display()));
            }
            // ECMWF local sf (198) is m water equivalent. Integrated
            // 0/1/53 (statistic 1) is kg m-2, hence mm water. WMO 29
            // is snow depth and must not acquire a second 10:1 conversion.
            let plane = decode(sf)?;
            let to_mm = if sf.product.parameter_number == 53 { 1.0 } else { 1000.0 };
            return Ok(Some((plane, to_mm)));
        }
        Err(format!("no ECMWF sf message for lead {lead} h in {}", dir.display()))
    };
    let (end_plane, end_scale) = at(end)?.ok_or("ECMWF window end is lead 0")?;
    let start_plane = at(start)?;
    let result = ecmwf_difference(end_plane, end_scale, start_plane)?;
    Ok((result, format!("ECMWF IFS open data 0.25 deg, 10 x sf({end} h) - sf({start} h)")))
}


fn import_woof(frame: &Path, store_root: &Path, start: u32, end: u32) -> Result<StoreFieldSource, String> {
    let file = wrf_core::WrfFile::open(frame).map_err(|e| e.to_string())?;
    let origin = file.global_attr_str("SIMULATION_START_DATE")
        .or_else(|_| file.global_attr_str("START_DATE")).map_err(|e| e.to_string())?;
    let origin = chrono::NaiveDateTime::parse_from_str(&origin, "%Y-%m-%d_%H:%M:%S")
        .map_err(|e| format!("invalid WOOF run origin: {e}"))?;
    let since = origin + chrono::Duration::hours(start as i64);
    let until = origin + chrono::Duration::hours(end as i64);
    if file.times().map_err(|e| e.to_string())?.last() != Some(&until.format("%Y-%m-%d_%H:%M:%S").to_string()) {
        return Err(format!("{}: final valid time differs from --end-lead {end}; reference windows would be mismatched", frame.display()));
    }
    std::fs::create_dir_all(store_root).map_err(|e| format!("create {}: {e}", store_root.display()))?;
    let task = spawn_process_paths(vec![frame.to_path_buf()], store_root.to_path_buf(), WrfProcessOptions {
        snow_since: Some(since.format("%Y-%m-%d_%H:%M:%S").to_string()),
        ..Default::default()
    });
    let import = loop {
        match task.rx.recv().map_err(|e| format!("importer exited: {e}"))? {
            WrfProcessMessage::Progress(_) => {}
            WrfProcessMessage::Done(result) => break result?,
        }
    };
    let manifest_path = store_root.join(&import.model).join(&import.run).join("run.json");
    let manifest: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&manifest_path).map_err(|e| format!("read {}: {e}", manifest_path.display()))?,
    )
    .map_err(|e| format!("parse {}: {e}", manifest_path.display()))?;
    let slot = manifest
        .get("hours")
        .and_then(serde_json::Value::as_object)
        .and_then(|hours| hours.keys().filter_map(|k| k.parse::<u16>().ok()).max())
        .ok_or_else(|| format!("{} has no hours", manifest_path.display()))?;
    StoreFieldSource::open(store_root, &import.model, &import.run, slot).map_err(|e| format!("open store: {e}"))
}

fn run(args: Args) -> Result<(), String> {
    // The grid every panel is drawn on, and WOOF's own window planes.
    let mut grid: Option<(Vec<f32>, Vec<f32>, usize, usize, Option<rustwx_core::GridProjection>)> = None;
    let mut style = None;
    let mut cycle = None;
    let mut planes: Vec<(String, Vec<f32>, String)> = Vec::new();
    for (index, (label, source)) in args.panels.iter().enumerate() {
        if let Source::Woof(frame) = source {
            let file = wrf_core::WrfFile::open(frame).map_err(|e| e.to_string())?;
            let origin = file.global_attr_str("SIMULATION_START_DATE")
                .or_else(|_| file.global_attr_str("START_DATE")).map_err(|e| e.to_string())?;
            let origin = chrono::NaiveDateTime::parse_from_str(&origin,"%Y-%m-%d_%H:%M:%S").map_err(|e| e.to_string())?;
            if cycle.is_some_and(|other| other != origin) {
                return Err("WOOF panels have different run origins; their lead windows would differ".into());
            }
            cycle = Some(origin);
            let store = import_woof(frame, &args.store_root.join(format!("woof_{index:02}")), args.start_lead, args.end_lead)?;
            let selector = FieldSelector::surface(CanonicalField::Snowfall10to1);
            let variable = store.resolve(&selector).ok_or_else(|| format!(
                "{}: no 10:1 snowfall window; the exact --start-lead baseline and frozen precipitation fields are required", frame.display()))?;
            let field = store.fetch(&selector).map_err(|e| {
                format!("{}: no snowfall window ({e})", frame.display())
            })?;
            if grid.is_none() {
                let meta = store
                    .surface_variable(variable)
                    .ok_or("10:1 snowfall window vanished from the store")?
                    .clone();
                style = rustwx_products::viewer::operational_style_for_store_variable(
                    variable,
                    &meta.selector,
                    &meta.units,
                    rustwx_core::ModelId::WrfGdex,
                ).or_else(|| rustwx_products::viewer::curated_style_for_store_variable(
                    variable, &meta.selector, &meta.units, rustwx_core::ModelId::WrfGdex));
                grid = Some((
                    field.grid.lat_deg.clone(),
                    field.grid.lon_deg.clone(),
                    field.grid.shape.ny,
                    field.grid.shape.nx,
                    store.projection().cloned(),
                ));
            }
            let (lat, lon, ny, nx, _) = grid.as_ref().expect("grid set");
            if field.values.len() != lat.len() || field.grid.shape.ny != *ny || field.grid.shape.nx != *nx
                || field.grid.lat_deg != *lat || field.grid.lon_deg != *lon {
                return Err(format!("{label}: WOOF frame on a different grid from the first --woof frame"));
            }
            planes.push((label.clone(), field.values.clone(), "WOOF, 10 x (SNOWNC + GRAUPELNC) over the window".into()));
        }
    }
    let (lat, lon, ny, nx, projection) = grid.ok_or("no WOOF grid")?;
    let style = style.ok_or("no snowfall style resolved for wrf_snow_10to1_window")?;
    // Reference panels, sampled at the nearest reference point.
    let mut ordered: Vec<(String, Vec<f32>, String)> = Vec::new();
    let mut woof = planes.into_iter();
    for (label, source) in &args.panels {
        let plane = match source {
            Source::Woof(_) => {
                ordered.push(woof.next().expect("one WOOF plane per --woof"));
                continue;
            }
            Source::Gfs(dir) => gfs_window(dir, args.start_lead, args.end_lead, cycle.expect("WOOF origin")),
            Source::Ecmwf(dir) => ecmwf_window(dir, args.start_lead, args.end_lead, cycle.expect("WOOF origin")),
        };
        let (reference, note) = plane.map_err(|e| format!("{label}: {e}"))?;
        let matched = match_grids(&lat, &lon, ny, nx, &reference.lat, &reference.lon, reference.ny, reference.nx)
            .map_err(|e| format!("{label}: {e}"))?;
        let values = sample(&reference.values, &matched);
        println!(
            "MATCH {label} missing={} max_distance_km={:0.1} spacing_km={:0.1}",
            matched.missing, matched.max_distance_km, matched.source_spacing_km
        );
        ordered.push((label.clone(), values, format!("{note}, nearest point")));
    }
    // Draw each panel on one table, then the sheet.
    let scratch = args.store_root.join("panels");
    std::fs::create_dir_all(&scratch).map_err(|e| format!("create {}: {e}", scratch.display()))?;
    let mut images = Vec::new();
    for (index, (label, values, note)) in ordered.iter().enumerate() {
        let values: Vec<f32> = values.iter().map(|v| style.convert.apply(*v)).collect();
        let path = render_panel(PanelRequest {
            lat_deg: &lat,
            lon_deg: &lon,
            projection: projection.as_ref(),
            ny,
            nx,
            values,
            product_slug: format!("refsnow_{index}"),
            title: label.clone(),
            display_units: style.display_units.clone(),
            scale: style.scale.clone(),
            cbar_tick_step: style.cbar_tick_step,
            legend: LegendControls { density: Default::default(), mode: style.legend_mode },
            render_density: RenderDensity::default(),
            subtitle_left: note.clone(),
            subtitle_center: None,
            subtitle_right: String::new(),
            width: args.width,
            height: args.height,
            contours: Vec::new(),
            colorbar: true,
            overlays: None,
            annotations: None,
            out_path: scratch.join(format!("panel_{index}.png")),
        })?;
        images.push(
            image::open(&path)
                .map_err(|e| format!("read back {}: {e}", path.display()))?
                .to_rgba8(),
        );
    }
    let rows: Vec<Vec<rustwx_render::RgbaImage>> = images.chunks(args.columns).map(|c| c.to_vec()).collect();
    // A short last row is padded with a blank panel of the same size.
    let rows: Vec<Vec<rustwx_render::RgbaImage>> = rows
        .into_iter()
        .map(|mut row| {
            while row.len() < args.columns && images.len() > args.columns {
                let (w, h) = (images[0].width(), images[0].height());
                row.push(rustwx_render::RgbaImage::from_pixel(w, h, *images[0].get_pixel(0, 0)));
            }
            row
        })
        .collect();
    let sheet = compose_sheet_rows(
        &rows,
        &SheetHeader { title: format!("{} ({})", args.title, style.display_units), subtitle: args.subtitle.clone() },
    )?;
    if let Some(parent) = args.out.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("create {}: {e}", parent.display()))?;
    }
    sheet.save(&args.out).map_err(|e| format!("write {}: {e}", args.out.display()))?;
    println!("RENDERED {}", args.out.display());
    Ok(())
}

fn main() -> ExitCode {
    std::hint::black_box(GPUWM_BRIDGE_SOURCE_REV_STAMP);
    match parse_args() {
        Ok(None) => ExitCode::SUCCESS,
        Ok(Some(args)) => match run(args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(message) => {
                eprintln!("rw_refsnow: {message}");
                ExitCode::FAILURE
            }
        },
        Err(message) => {
            eprintln!("rw_refsnow: {message}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn plane(values: &[f32]) -> Plane {
        Plane { lat: vec![47.; values.len()], lon: (0..values.len()).map(|i| -112.+i as f32).collect(),
            ny: 1, nx: values.len(), values: values.to_vec() }
    }
    #[test]
    fn frozen_precipitation_uses_its_fraction_and_keeps_missing_cells() {
        let result = gfs_frozen(&plane(&[2.,4.,f32::NAN,-1.]),&plane(&[0.5,1.,1.,1.])).unwrap();
        assert_eq!(&result[..2],&[10.,40.]);
        assert!(result[2].is_nan() && result[3].is_nan());
        let mut shifted=plane(&[0.5,1.,1.,1.]); shifted.lon[0]+=1.;
        assert!(gfs_frozen(&plane(&[2.,4.,2.,1.]),&shifted).is_err());
    }
    #[test]
    fn accumulated_water_is_converted_before_differencing_and_resets_stay_missing() {
        let result=ecmwf_difference(plane(&[0.004,0.001,f32::NAN]),1000.,Some((plane(&[2.,2.,2.]),1.))).unwrap();
        assert_eq!(result.values[0],20.);
        assert!(result.values[1].is_nan() && result.values[2].is_nan());
        let zero=ecmwf_difference(plane(&[0.004]),1000.,None).unwrap();
        // The source patch's f32 evaluation gives 40 plus one ULP here.
        assert_eq!(zero.values,vec![f32::from_bits(0x42200001)]);
        let mut shifted=plane(&[1.]); shifted.lat[0]+=1.;
        assert!(ecmwf_difference(plane(&[2.]),1.,Some((shifted,1.))).is_err());
    }
    #[test]
    fn interval_end_must_agree_with_the_encoded_duration() {
        use grib_core::grib2::{DataRepresentation, GridDefinition, ProductDefinition};
        let cycle=chrono::NaiveDateTime::parse_from_str("2026-01-01_00:00:00","%Y-%m-%d_%H:%M:%S").unwrap();
        let mut message=Grib2Message { discipline:0, identification:Default::default(), reference_time:cycle,
            grid:GridDefinition::default(), product:ProductDefinition {
                forecast_time:0,time_range_unit:1,statistical_time_range_unit:Some(1),time_range_length:Some(6),
                end_of_interval:Some(cycle+chrono::Duration::hours(6)),..Default::default() },
            data_rep:DataRepresentation::default(),bitmap:None,raw_data:vec![] };
        assert_eq!(accumulation_end(&message),Some(6));
        message.product.end_of_interval=Some(cycle+chrono::Duration::hours(12));
        assert_eq!(accumulation_end(&message),None);
        message.product.time_range_unit=0;message.product.forecast_time=1;
        assert_eq!(lead_hours(&message),None);
    }
}
