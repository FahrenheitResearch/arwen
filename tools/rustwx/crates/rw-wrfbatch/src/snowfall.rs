//! Snowfall depths in mm of snow, integrated over the available WRF history.
//! The ratio and f32 interval arithmetic match the supplied snowfall patch.
use std::path::{Path, PathBuf};

use crate::wrf_process::{
    WrfHourFields, WrfProcessOptions, WrfProductGroup, compute_var, push_canonical_values,
};
use rustwx_core::{CanonicalField, FieldSelector, GridProjection, LatLonGrid};
use wrf_core::WrfFile;

/// Catalog metadata shared by the import and its field/selector preview.
pub const PRODUCTS: &[(&str, CanonicalField)] = &[
    ("snowfall_window", CanonicalField::ModelSnowfall),
    ("snow_10to1_window", CanonicalField::Snowfall10to1),
    ("snow_kuchera_window", CanonicalField::SnowfallKuchera),
];

/// Owned by one import, never shared between jobs or source snapshots.
/// Retains only the latest cumulative plane and liquid counter.
#[derive(Default)]
pub(crate) struct WindowCache {
    kuchera: Option<WindowState>,
    fixed: Option<WindowState>,
}

struct WindowState {
    directory: PathBuf,
    prefix: String,
    reference: Option<i64>,
    start: i64,
    until: i64,
    previous: Vec<f32>,
    accumulated: Vec<f32>,
}

/// Supplied formula: 12 + 2*(271.16 - Tmax) above 271.16 K,
/// else 12 + (271.16 - Tmax), floored at zero, as in the source patch.
pub fn kuchera_slr(tmax_k: f32) -> f32 {
    if !tmax_k.is_finite() {
        return f32::NAN;
    }
    let d = 271.16 - tmax_k;
    (if tmax_k > 271.16 {
        12.0 + 2.0 * d
    } else {
        12.0 + d
    })
    .max(0.0)
}

/// Native mass-level Tmax from the lowest level through 500 hPa, inclusive.
/// Unknown pressure or temperature below 500 hPa invalidates that column.
pub fn column_max(
    temperature: &[f64],
    pressure: &[f64],
    nz: usize,
    cells: usize,
) -> Result<Vec<f32>, String> {
    if nz == 0
        || cells == 0
        || nz.checked_mul(cells) != Some(temperature.len())
        || pressure.len() != temperature.len()
    {
        return Err("temperature/pressure must have the same nonempty [nz,ny,nx] shape".into());
    }
    Ok((0..cells)
        .map(|i| {
            let mut best = f64::NEG_INFINITY;
            for k in 0..nz {
                let p = pressure[k * cells + i];
                if !p.is_finite() || p <= 0.0 || p > 1.0e30 {
                    return f32::NAN;
                }
                if p < 50000.0 {
                    break;
                }
                let t = temperature[k * cells + i];
                if !t.is_finite() || t <= 0.0 || t > 1.0e30 {
                    return f32::NAN;
                }
                best = best.max(t);
            }
            if best.is_finite() {
                best as f32
            } else {
                f32::NAN
            }
        })
        .collect())
}

pub fn column_tmax_500(file: &WrfFile, timeidx: usize, cells: usize) -> Result<Vec<f32>, String> {
    let t = compute_var(file, "temp", timeidx, Some("K"))?;
    let p = compute_var(file, "pres", timeidx, Some("Pa"))?;
    if t.shape != [file.nz, file.ny, file.nx] || p.shape != t.shape || cells != file.ny * file.nx {
        return Err("temperature/pressure dimension order differs from [nz,ny,nx]".into());
    }
    column_max(&t.data, &p.data, file.nz, cells)
}

fn plane(file: &WrfFile, name: &str, ti: usize) -> Result<Vec<f32>, String> {
    let v = compute_var(file, name, ti, None)?;
    if v.shape != [file.ny, file.nx] || v.data.len() != file.ny * file.nx {
        return Err(format!("{name}: expected [ny,nx], got {:?}", v.shape));
    }
    Ok(v.data
        .iter()
        .map(|&v| {
            if v.is_finite() && (0.0..1.0e30).contains(&v) {
                v as f32
            } else {
                f32::NAN
            }
        })
        .collect())
}

fn frozen(file: &WrfFile, ti: usize) -> Result<Vec<f32>, String> {
    let mut snow = plane(file, "SNOWNC", ti)?;
    // Absence is allowed. A present but unreadable graupel accumulator is
    // an error, to prevent a silently understated snowfall total.
    if file.has_var("GRAUPELNC") {
        let graupel = plane(file, "GRAUPELNC", ti)?;
        for (s, g) in snow.iter_mut().zip(graupel) {
            *s += g;
        }
    }
    Ok(snow)
}

/// End-of-interval weighting, in source-patch order and precision.
/// A reset or missing accumulator poisons that cell instead of inventing
/// a zero snowfall interval after a restart.
pub fn accumulate(
    acc: &mut [f32],
    previous: &[f32],
    current: &[f32],
    tmax: &[f32],
) -> Result<(), String> {
    if [previous.len(), current.len(), tmax.len()]
        .iter()
        .any(|&n| n != acc.len())
    {
        return Err("snowfall interval planes have different lengths".into());
    }
    for i in 0..acc.len() {
        let d = current[i] - previous[i];
        let ratio = kuchera_slr(tmax[i]);
        if d.is_finite() && d >= 0.0 && ratio.is_finite() && acc[i].is_finite() {
            acc[i] += d.max(0.0) * ratio;
        } else {
            acc[i] = f32::NAN;
        }
    }
    Ok(())
}

/// Neighboring files of the same WRF domain. Other domains and scratch
/// files cannot enter a snowfall integration or its cache identity.
pub(crate) fn history_paths(path: &Path) -> Result<Vec<PathBuf>, String> {
    let mut paths = vec![path.to_path_buf()];
    let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
    if name.starts_with("wrfout_d") && name.as_bytes().get(10) == Some(&b'_') {
        let prefix = &name[..11];
        for entry in
            std::fs::read_dir(path.parent().unwrap_or(Path::new("."))).map_err(|e| e.to_string())?
        {
            let entry = entry.map_err(|e| e.to_string())?;
            let n = entry.file_name();
            let n = n.to_string_lossy();
            if n.starts_with(prefix) && n.len() == prefix.len() + 19 {
                paths.push(entry.path());
            }
        }
    }
    paths.sort();
    paths.dedup();
    Ok(paths)
}

// A future frame may still be being written. Its filename gives its
// earliest time under the ordinary WRF history naming contract.
fn starts_after(path: &Path, until: i64) -> bool {
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
        return false;
    };
    let Some(stamp) = name.get(11..) else {
        return false;
    };
    chrono::NaiveDateTime::parse_from_str(&stamp.replace(':', "_"), "%Y-%m-%d_%H_%M_%S")
        .ok()
        .is_some_and(|t| t.and_utc().timestamp() > until)
}

/// Integrate all records in (since,current], including records in a
/// multi-time file. The exact baseline must exist, so a truncated history
/// cannot be advertised as a total since initialization.
pub fn window(
    file: &WrfFile,
    path: &Path,
    timeidx: usize,
    since: Option<&str>,
    kuchera: bool,
) -> Result<Vec<f32>, String> {
    window_cached(file, path, timeidx, since, kuchera, &mut None)
}

fn window_cached(
    file: &WrfFile,
    path: &Path,
    timeidx: usize,
    since: Option<&str>,
    kuchera: bool,
    cache: &mut Option<WindowState>,
) -> Result<Vec<f32>, String> {
    let axis = crate::local_import::wrf_source_times(file, path)?;
    let now = axis
        .records
        .get(timeidx)
        .ok_or("current record missing")?
        .valid_unix;
    let start = match since {
        Some(s) => chrono::NaiveDateTime::parse_from_str(&s.replace(':', "_"), "%Y-%m-%d_%H_%M_%S")
            .map_err(|e| format!("snowfall baseline must be YYYY-MM-DD_HH:MM:SS: {e}"))?
            .and_utc()
            .timestamp(),
        None => axis
            .reference_unix
            .ok_or("snowfall requires a run initialization time")?,
    };
    if start > now {
        return Err("snowfall baseline is later than the current frame".into());
    }
    let mut records = Vec::new();
    for p in history_paths(path)? {
        if p != path && starts_after(&p, now) {
            continue;
        }
        let f = WrfFile::open(&p).map_err(|e| format!("{}: {e}", p.display()))?;
        let a = crate::local_import::wrf_source_times(&f, &p)?;
        for r in a.records {
            if r.valid_unix >= start && r.valid_unix <= now {
                if a.reference_unix != axis.reference_unix
                    || (f.nz, f.ny, f.nx) != (file.nz, file.ny, file.nx)
                    || f.xlat(r.time_index).map_err(|e| e.to_string())?
                        != file.xlat(timeidx).map_err(|e| e.to_string())?
                    || f.xlong(r.time_index).map_err(|e| e.to_string())?
                        != file.xlong(timeidx).map_err(|e| e.to_string())?
                {
                    return Err(
                        "snowfall history run origin or grid differs from current frame".into(),
                    );
                }
                records.push((r.valid_unix, p.clone(), r.time_index));
            }
        }
    }
    records.sort();
    if records.first().map(|r| r.0) != Some(start) {
        return Err(
            "exact snowfall baseline frame missing; refusing an understated window total".into(),
        );
    }
    if records.windows(2).any(|r| r[0].0 == r[1].0) {
        return Err("duplicate snowfall valid time; refusing double counting".into());
    }
    let cells = file.ny * file.nx;
    let directory = path.parent().unwrap_or(Path::new(".")).to_path_buf();
    let name = path.file_name().and_then(|v| v.to_str()).unwrap_or("");
    let prefix = if name.starts_with("wrfout_d") && name.as_bytes().get(10) == Some(&b'_') {
        &name[..11]
    } else {
        name
    };
    let reusable = cache.as_ref().filter(|state| {
        state.directory == directory
            && state.prefix == prefix
            && state.reference == axis.reference_unix
            && state.start == start
            && state.until <= now
            && records.iter().any(|r| r.0 == state.until)
            && state.previous.len() == cells
    });
    let (mut prev, mut acc, until) = if let Some(state) = reusable {
        (
            state.previous.clone(),
            state.accumulated.clone(),
            state.until,
        )
    } else {
        let base = WrfFile::open(&records[0].1).map_err(|e| e.to_string())?;
        (frozen(&base, records[0].2)?, vec![0.0; cells], start)
    };
    // A missing baseline value stays unknown even for a zero-length window.
    for (a, p) in acc.iter_mut().zip(&prev) {
        if !p.is_finite() {
            *a = f32::NAN;
        }
    }
    for (_, p, ti) in records.iter().filter(|r| r.0 > until) {
        let f = WrfFile::open(p).map_err(|e| e.to_string())?;
        let current = frozen(&f, *ti)?;
        // Keep 10:1 exact rather than obtaining it through rounded K.
        if kuchera {
            accumulate(&mut acc, &prev, &current, &column_tmax_500(&f, *ti, cells)?)?;
        } else {
            for i in 0..cells {
                let d = current[i] - prev[i];
                if d.is_finite() && d >= 0.0 {
                    acc[i] += d * 10.0;
                } else {
                    acc[i] = f32::NAN;
                }
            }
        }
        prev = current;
    }
    *cache = Some(WindowState {
        directory,
        prefix: prefix.into(),
        reference: axis.reference_unix,
        start,
        until: now,
        previous: prev,
        accumulated: acc.clone(),
    });
    Ok(acc)
}

pub(crate) fn push(
    fields: &mut WrfHourFields,
    file: &WrfFile,
    path: &Path,
    ti: usize,
    options: &WrfProcessOptions,
    grid: &LatLonGrid,
    projection: Option<GridProjection>,
    cache: &mut WindowCache,
) {
    for &(slug, field) in PRODUCTS {
        if !options.should_process(
            &FieldSelector::surface(field).key(),
            Some(slug),
            WrfProductGroup::Diagnostic,
        ) {
            continue;
        }
        let result = match field {
            CanonicalField::ModelSnowfall => {
                model_window(file, path, ti, options.snow_since.as_deref())
            }
            _ => window_cached(
                file,
                path,
                ti,
                options.snow_since.as_deref(),
                field == CanonicalField::SnowfallKuchera,
                if field == CanonicalField::SnowfallKuchera {
                    &mut cache.kuchera
                } else {
                    &mut cache.fixed
                },
            ),
        };
        match result {
            Ok(v) => {
                let missing = v.iter().filter(|v| !v.is_finite()).count();
                push_canonical_values(
                    fields,
                    grid,
                    projection.clone(),
                    slug,
                    FieldSelector::surface(field),
                    "mm",
                    v,
                );
                fields.notes.push(format!(
                    "{slug}: mm of snow; interval-end ratio; baseline {}",
                    options
                        .snow_since
                        .as_deref()
                        .unwrap_or("model initialization")
                ));
                if missing > 0 {
                    fields.notes.push(format!("{slug}: {missing} cells remain missing; unknown column temperatures, non-finite accumulators and counter resets are not zero snowfall"));
                }
            }
            Err(e) => fields.notes.push(format!("{slug} unavailable: {e}")),
        }
    }
}

fn model_window(
    file: &WrfFile,
    path: &Path,
    ti: usize,
    since: Option<&str>,
) -> Result<Vec<f32>, String> {
    let current = plane(file, "SNOWFALLAC", ti)?;
    // Explicit model snowfall is already a depth in mm, not snow water.
    if since.is_none() {
        return Ok(current);
    }
    let wanted = chrono::NaiveDateTime::parse_from_str(
        &since.unwrap().replace(':', "_"),
        "%Y-%m-%d_%H_%M_%S",
    )
    .map_err(|e| format!("invalid snowfall baseline: {e}"))?
    .and_utc()
    .timestamp();
    let axis = crate::local_import::wrf_source_times(file, path)?;
    let now = axis.records[ti].valid_unix;
    if wanted > now {
        return Err("snowfall baseline is later than current frame".into());
    }
    let mut base = None;
    for p in history_paths(path)? {
        if p != path && starts_after(&p, wanted) {
            continue;
        }
        let f = WrfFile::open(&p).map_err(|e| e.to_string())?;
        let a = crate::local_import::wrf_source_times(&f, &p)?;
        for r in a.records {
            if r.valid_unix == wanted {
                if base.is_some() {
                    return Err("duplicate explicit snowfall baseline".into());
                }
                if a.reference_unix != axis.reference_unix
                    || (f.ny, f.nx) != (file.ny, file.nx)
                    || f.xlat(r.time_index).map_err(|e| e.to_string())?
                        != file.xlat(ti).map_err(|e| e.to_string())?
                    || f.xlong(r.time_index).map_err(|e| e.to_string())?
                        != file.xlong(ti).map_err(|e| e.to_string())?
                {
                    return Err("explicit snowfall baseline grid differs".into());
                }
                base = Some(plane(&f, "SNOWFALLAC", r.time_index)?);
            }
        }
    }
    let base = base.ok_or("exact explicit snowfall baseline missing")?;
    Ok(current
        .iter()
        .zip(base)
        .map(|(&c, b)| {
            if c.is_finite() && b.is_finite() && c >= b {
                c - b
            } else {
                f32::NAN
            }
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hand_computed_columns_and_ratio_branches() {
        let p = [90000., 80000., 50000., 49999.];
        let t = [265.16, 270.16, 269.16, 310.];
        let m = column_max(&t, &p, 4, 1).unwrap()[0];
        assert!((kuchera_slr(m) - 13.).abs() < 0.0001);
        for (t, r) in [
            (271.16, 12.),
            (272.16, 10.),
            (273.16, 8.),
            (269.16, 14.),
            (277.16, 0.),
            (280., 0.),
        ] {
            assert!((kuchera_slr(t) - r).abs() < 0.0001);
        }
        assert!(kuchera_slr(f32::NAN).is_nan());
        assert!(kuchera_slr(f32::INFINITY).is_nan());
    }
    #[test]
    fn included_500hpa_and_invalid_columns() {
        assert_eq!(
            column_max(&[260., 275., 310.], &[90000., 50000., 49000.], 3, 1).unwrap(),
            [275.]
        );
        assert!(column_max(&[260., f64::NAN], &[90000., 70000.], 2, 1).unwrap()[0].is_nan());
        assert!(column_max(&[260.], &[49000.], 1, 1).unwrap()[0].is_nan());
        assert!(column_max(&[260.], &[f64::NAN], 1, 1).unwrap()[0].is_nan());
        assert!(column_max(&[260.], &[90000.], 2, 1).is_err());
    }
    #[test]
    fn intervals_use_end_temperature_and_propagate_missing_or_resets() {
        let mut v = [0., 0., 0.];
        accumulate(&mut v, &[0., 2., f32::NAN], &[2., 1., 3.], &[272.16; 3]).unwrap();
        assert!((v[0] - 20.).abs() < 0.0002);
        assert!(v[1].is_nan());
        assert!(v[2].is_nan());
        accumulate(&mut v, &[2., 1., 3.], &[3., 2., 4.], &[269.16; 3]).unwrap();
        assert!((v[0] - 34.).abs() < 0.0003);
        assert!(accumulate(&mut v, &[], &[], &[]).is_err());
    }
}
