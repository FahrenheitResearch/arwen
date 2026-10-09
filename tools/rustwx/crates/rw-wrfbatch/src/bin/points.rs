//! Geographic point extraction using the same Rust column diagnostics as
//! the renderer. No model, case, or site is built into this tool.
use rw_wrfbatch::snowfall::{column_tmax_500, kuchera_slr};
use rw_wrfbatch::wrf_process::compute_var;
use std::io::Write;
use std::path::PathBuf;
use wrf_core::WrfFile;

pub static SOURCE_REV_STAMP: &str =
    concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));

const FIELDS: &[&str] = &[
    "T2",
    "U10",
    "V10",
    "RAINNC",
    "RAINC",
    "SNOWNC",
    "GRAUPELNC",
    "SNOWFALLAC",
    "SNOWH",
    "SNOW",
    "HGT",
];
const DERIVED: &[&str] = &["TMAXCOL", "KUCH_SLR", "SNOWLVL_FT", "GUST10"];

fn csv(s: &str) -> String {
    format!("\"{}\"", s.replace('"', "\"\""))
}

fn distance(lat: f64, lon: f64, a: f64, b: f64) -> f64 {
    let dy = (lat - a).to_radians();
    let dx = (lon - b).to_radians();
    let h = (dy / 2.).sin().powi(2)
        + lat.to_radians().cos() * a.to_radians().cos() * (dx / 2.).sin().powi(2);
    2. * 6371. * h.clamp(0., 1.).sqrt().asin()
}

fn derived(file: &WrfFile, ti: usize, cells: usize) -> Vec<Option<Vec<f64>>> {
    let t = column_tmax_500(file, ti, cells).ok();
    let slr = t
        .as_ref()
        .map(|v| v.iter().map(|&t| kuchera_slr(t) as f64).collect());
    let t = t.map(|v| v.into_iter().map(f64::from).collect());
    let level = match (
        compute_var(file, "wet_bulb_0", ti, Some("m")),
        compute_var(file, "terrain", ti, Some("m")),
    ) {
        (Ok(w), Ok(h)) if w.data.len() == cells && h.data.len() == cells => Some(
            w.data
                .iter()
                .zip(h.data)
                .map(|(&w, h)| {
                    if w.is_finite() && h.is_finite() {
                        (w.max(0.) + h) * 3.28084
                    } else {
                        f64::NAN
                    }
                })
                .collect(),
        ),
        _ => None,
    };
    let get = |name| compute_var(file, name, ti, None).ok();
    let gust = match (get("height_agl"), get("wspd"), get("PBLH"), get("wspd10")) {
        (Some(h), Some(w), Some(p), Some(s))
            if h.shape == [file.nz, file.ny, file.nx]
                && w.shape == h.shape
                && h.data.len() == file.nz * cells
                && w.data.len() == h.data.len()
                && p.data.len() == cells
                && s.data.len() == cells =>
        {
            Some(
                (0..cells)
                    .map(|i| {
                        let (z, s) = (p.data[i], s.data[i]);
                        if !z.is_finite() || !s.is_finite() {
                            return f64::NAN;
                        }
                        let mut k = 0;
                        while k + 1 < file.nz && h.data[k * cells + i] < z {
                            k += 1;
                        }
                        let top = w.data[k * cells + i];
                        if !top.is_finite() {
                            return f64::NAN;
                        }
                        (s + (top - s) * (1. - (z.max(0.) / 2000.).min(0.5))).max(s)
                    })
                    .collect(),
            )
        }
        _ => None,
    };
    vec![t, slr, level, gust]
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let mut points = Vec::new();
    let mut files = Vec::new();
    let mut out = None;
    let mut label = String::new();
    let mut timeidx = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" => {
                println!(
                    "rw_points --point ID,LAT,LON [--point ...] --out FILE.csv [--label TEXT] [--timeidx N] wrfout...\nAll time records by default. Raw columns keep source units; TMAXCOL K, KUCH_SLR ratio, SNOWLVL_FT ft MSL, GUST10 m/s. Missing values are blank; distance_km records nearest-grid distance. U10/V10 retain native grid rotation."
                );
                return Ok(());
            }
            "--point" => {
                let value = args.next().ok_or("--point needs ID,LAT,LON")?;
                let p: Vec<_> = value.rsplitn(3, ',').collect();
                if p.len() != 3 {
                    return Err("--point needs ID,LAT,LON".into());
                }
                let lat = p[1].parse::<f64>().map_err(|_| "invalid point latitude")?;
                let lon = p[0].parse::<f64>().map_err(|_| "invalid point longitude")?;
                if p[2].is_empty()
                    || !lat.is_finite()
                    || lat.abs() > 90.
                    || !lon.is_finite()
                    || lon.abs() > 360.
                {
                    return Err("point id/coordinates invalid".into());
                }
                points.push((p[2].to_string(), lat, lon));
            }
            "--out" => out = Some(args.next().ok_or("--out needs a path")?),
            "--label" => label = args.next().ok_or("--label needs text")?,
            "--timeidx" => {
                timeidx = Some(
                    args.next()
                        .ok_or("--timeidx needs N")?
                        .parse::<usize>()
                        .map_err(|_| "invalid time index")?,
                )
            }
            _ if arg.starts_with('-') => return Err(format!("unknown option {arg}")),
            _ => files.push(PathBuf::from(arg)),
        }
    }
    if points.is_empty() || files.is_empty() {
        return Err("at least one --point and input file are required".into());
    }
    let out = out.ok_or("--out is required")?;
    let mut rows = Vec::new();
    for path in files {
        let file = WrfFile::open(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let times = file.times().map_err(|e| e.to_string())?;
        if times.len() != file.nt {
            return Err("Times record count differs from the file time dimension".into());
        }
        let indices = match timeidx {
            Some(n) if n < file.nt => vec![n],
            Some(_) => return Err("time index exceeds records".into()),
            None => (0..file.nt).collect(),
        };
        for ti in indices {
            let lat = file.xlat(ti).map_err(|e| e.to_string())?;
            let lon = file.xlong(ti).map_err(|e| e.to_string())?;
            let cells = file.ny.checked_mul(file.nx).ok_or("grid size overflow")?;
            if lat.len() != cells || lon.len() != cells {
                return Err("coordinate shape differs from grid".into());
            }
            let mut planes: Vec<Option<Vec<f64>>> = FIELDS
                .iter()
                .map(|&n| {
                    compute_var(&file, n, ti, None)
                        .ok()
                        .filter(|v| v.shape == [file.ny, file.nx] && v.data.len() == cells)
                        .map(|v| v.data)
                })
                .collect();
            planes.extend(derived(&file, ti, cells));
            for (id, a, b) in &points {
                let (k, d) = (0..cells)
                    .filter(|&k| {
                        lat[k].is_finite()
                            && lon[k].is_finite()
                            && lat[k].abs() <= 90.
                            && lon[k].abs() <= 360.
                    })
                    .map(|k| (k, distance(lat[k], lon[k], *a, *b)))
                    .min_by(|x, y| x.1.total_cmp(&y.1))
                    .ok_or("no finite geographic grid cells")?;
                let mut row = format!(
                    "{},{},{},{},{ti},{},{},{:.6},{:.6},{:.6}",
                    csv(&label),
                    csv(&path.display().to_string()),
                    csv(&times[ti]),
                    csv(id),
                    k / file.nx,
                    k % file.nx,
                    lat[k],
                    lon[k],
                    d
                );
                for plane in &planes {
                    row.push(',');
                    if let Some(v) = plane
                        .as_ref()
                        .map(|v| v[k])
                        .filter(|v| v.is_finite() && v.abs() < 1.0e30)
                    {
                        row.push_str(&format!("{v:.8}"));
                    }
                }
                rows.push(row);
            }
            file.clear_cache();
        }
    }
    // Validate every input before creating the output, preventing a failed
    // later frame from leaving a partial CSV that looks complete.
    let mut w =
        std::io::BufWriter::new(std::fs::File::create(&out).map_err(|e| format!("{out}: {e}"))?);
    write!(w, "label,file,valid,id,timeidx,j,i,lat,lon,distance_km").map_err(|e| e.to_string())?;
    for n in FIELDS.iter().chain(DERIVED) {
        write!(w, ",{n}").map_err(|e| e.to_string())?;
    }
    writeln!(w).map_err(|e| e.to_string())?;
    for row in rows {
        writeln!(w, "{row}").map_err(|e| e.to_string())?;
    }
    w.flush().map_err(|e| e.to_string())
}

fn main() {
    std::hint::black_box(SOURCE_REV_STAMP);
    if let Err(e) = run() {
        eprintln!("rw_points: {e}");
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn nearest_distance_wraps_longitude_and_csv_escapes() {
        assert!(distance(0., 179.9, 0., -179.9) < 23.);
        assert_eq!(csv("a,\"b\""), "\"a,\"\"b\"\"\"");
    }
}
