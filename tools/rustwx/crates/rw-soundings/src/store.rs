//! rw-store access: catalogue (cycles / members / leads), grid location, and
//! raw sounding columns (point = bilinear, box = area mean over a common
//! finite mask per level, the bowecho `box_sounding.rs` method).
//!
//! Two store conventions are read:
//! * `native` (native profile writer): `pressure_native` hPa, `height_native` m,
//!   `temperature_native` K, `dewpoint_native` K, `u_native`/`v_native` m/s
//!   (earth-relative), `omega_native` Pa/s on model levels (level code
//!   1000 - k), plus the 2-D surface anchors.
//! * `iso` (Drew's rw-ui / rusty-weather ingest): `temperature_iso`,
//!   `dewpoint_iso` | `rh_iso`, `u_iso`, `v_iso`, `height_iso` on isobaric
//!   levels, plus the same surface anchors (or their `approx_*` names).
//!
//! The public batch adapter supplies a plain-run inventory with physical valid times.

use std::collections::{BTreeMap, HashMap};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use lru::LruCache;
use rw_store::grid::{GridFile, GridLocator};
use rw_store::reader::HourReader;

pub type Res<T> = Result<T, String>;

#[derive(Clone, Debug, serde::Serialize)]
pub struct LeadEntry {
    pub lead_min: u32,
    #[serde(skip)]
    pub path: PathBuf,
}

/// cycle -> member -> lead_min -> file
pub type Catalog = BTreeMap<String, BTreeMap<String, BTreeMap<u32, PathBuf>>>;

pub struct GridInfo {
    pub grid: GridFile,
    pub locator: GridLocator,
    /// west, south, east, north
    pub bbox: [f64; 4],
}

pub struct Store {
    pub root: PathBuf,
    pub model: String,
    catalog: Mutex<Option<(Instant, Arc<Catalog>)>>,
    readers: Mutex<LruCache<PathBuf, Arc<HourReader>>>,
    grids: Mutex<HashMap<PathBuf, Arc<GridInfo>>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Convention {
    Native,
    Iso,
}

/// Surface anchor (bridge units: hPa, m, C, m/s).
#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct Sfc {
    pub psfc_hpa: f64,
    pub orog_m: f64,
    pub t2_c: f64,
    pub td2_c: f64,
    pub u10: f64,
    pub v10: f64,
}

/// One raw column, lowest level first, bridge units (hPa, m MSL, C, m/s, Pa/s).
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct RawColumn {
    pub p: Vec<f64>,
    pub z: Vec<f64>,
    pub t: Vec<f64>,
    pub td: Vec<f64>,
    pub u: Vec<f64>,
    pub v: Vec<f64>,
    pub omega: Vec<f64>,
    pub sfc: Sfc,
}

impl RawColumn {
    pub fn levels(&self) -> usize {
        self.p.len()
    }
}

/// Surface anchors, every accepted name (data-lane store first, then the
/// native profile writer / rw-ui names, then rw-ui's approx_* fallbacks).
const SFC_NAMES: [&[&str]; 6] = [
    &["psfc", "surface_pressure", "approx_surface_pressure"],
    &["hgt_sfc", "orography"],
    &["t2m", "temperature_2m", "approx_temperature_2m"],
    &["d2m", "dewpoint_2m", "approx_dewpoint_2m"],
    &["u10", "u_10m", "approx_u_10m"],
    &["v10", "v_10m", "approx_v_10m"],
];

/// Native-level volumes: p, z, T, Td, u, v, omega.
const NATIVE_NAMES: [&[&str]; 7] = [
    &["pres_native", "pressure_native"],
    &["hght_native", "height_native"],
    &["tmp_native", "temperature_native"],
    &["dpt_native", "dewpoint_native"],
    &["u_native"],
    &["v_native"],
    &["omega_native"],
];

fn native_name(reader: &HourReader, i: usize) -> Option<(&'static str, String)> {
    NATIVE_NAMES[i].iter().find_map(|n| {
        reader
            .variable(n)
            .filter(|v| v.kind == "pressure3d")
            .map(|v| (*n, v.units.clone()))
    })
}

/// Native value -> bridge units (hPa, m, C, m/s, Pa/s) by stored units.
fn native_convert(i: usize, v: f64, units: &str) -> f64 {
    match (i, units) {
        (0, "Pa") => v / 100.0,
        (2 | 3, "K") => v - 273.15,
        _ => v,
    }
}

fn sfc_convert(i: usize, v: f64, units: &str) -> f64 {
    match (i, units) {
        (0, "Pa") => v / 100.0,
        (2 | 3, "K") => v - 273.15,
        _ => v,
    }
}

impl Store {
    pub fn label(&self, _cycle: &str) -> String {
        self.model.clone()
    }

    pub fn new(root: PathBuf, model: String) -> Self {
        Store {
            root,
            model,
            catalog: Mutex::new(None),
            readers: Mutex::new(LruCache::new(NonZeroUsize::new(512).unwrap())),
            grids: Mutex::new(HashMap::new()),
        }
    }

    /// Rescanned at most every 5 s (the data lane appends cycles live).
    pub fn catalog(&self) -> Arc<Catalog> {
        let mut g = self.catalog.lock().unwrap();
        if let Some((t, c)) = g.as_ref()
            && (self.model == "forecast" || t.elapsed() < Duration::from_secs(5))
        {
            return c.clone();
        }
        let c = Arc::new(self.scan());
        *g = Some((Instant::now(), c.clone()));
        c
    }

    fn scan(&self) -> Catalog {
        Catalog::new()
    }

    pub fn with_catalog(root: PathBuf, catalog: Catalog) -> Self {
        let store = Self::new(root, "forecast".into());
        *store.catalog.lock().unwrap() = Some((Instant::now(), Arc::new(catalog)));
        store
    }

    pub fn reader(&self, path: &Path) -> Res<Arc<HourReader>> {
        if let Some(r) = self.readers.lock().unwrap().get(path) {
            return Ok(r.clone());
        }
        let r =
            Arc::new(HourReader::open(path).map_err(|e| format!("open {}: {e}", path.display()))?);
        self.readers
            .lock()
            .unwrap()
            .put(path.to_path_buf(), r.clone());
        Ok(r)
    }

    pub fn grid_for(&self, hour_path: &Path) -> Res<Arc<GridInfo>> {
        let gpath = hour_path
            .parent()
            .ok_or("hour path has no run dir")?
            .join("grid.rwg");
        if let Some(g) = self.grids.lock().unwrap().get(&gpath) {
            return Ok(g.clone());
        }
        let grid = GridFile::open(&gpath).map_err(|e| format!("open {}: {e}", gpath.display()))?;
        let locator = GridLocator::build(&grid);
        let mut bbox = [
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ];
        for (la, lo) in grid.lat.iter().zip(&grid.lon) {
            if la.is_finite() && lo.is_finite() {
                bbox[0] = bbox[0].min(*lo as f64);
                bbox[1] = bbox[1].min(*la as f64);
                bbox[2] = bbox[2].max(*lo as f64);
                bbox[3] = bbox[3].max(*la as f64);
            }
        }
        let info = Arc::new(GridInfo {
            grid,
            locator,
            bbox,
        });
        self.grids.lock().unwrap().insert(gpath, info.clone());
        Ok(info)
    }

    pub fn convention(reader: &HourReader) -> Res<Convention> {
        if native_name(reader, 0).is_some() {
            Ok(Convention::Native)
        } else if reader.variable("temperature_iso").is_some() {
            Ok(Convention::Iso)
        } else {
            Err("hour has neither native (pressure_native) nor isobaric (temperature_iso) sounding volumes".into())
        }
    }

    fn sfc_name(reader: &HourReader, i: usize) -> Option<(&'static str, String)> {
        for n in SFC_NAMES[i].iter().copied() {
            if let Some(v) = reader.variable(n)
                && v.kind == "surface2d"
            {
                return Some((n, v.units.clone()));
            }
        }
        None
    }

    /// Bilinear point column at fractional grid coords.
    pub fn point_column(reader: &HourReader, fx: f64, fy: f64) -> Res<RawColumn> {
        let conv = Self::convention(reader)?;
        let (nx, ny) = (reader.meta().nx, reader.meta().ny);
        let e = |n: &str, err: rw_store::RwStoreError| format!("read {n}: {err}");
        let prof = |n: &str| -> Res<Vec<f64>> {
            Ok(reader
                .read_profile_3d(n, fx, fy)
                .map_err(|x| e(n, x))?
                .into_iter()
                .map(f64::from)
                .collect())
        };
        // bilinear 2-D
        let x0 = (fx.floor().max(0.0) as usize).min(nx - 1);
        let y0 = (fy.floor().max(0.0) as usize).min(ny - 1);
        let x1 = (x0 + 1).min(nx - 1);
        let y1 = (y0 + 1).min(ny - 1);
        let (ax, ay) = (
            (fx - x0 as f64).clamp(0.0, 1.0),
            (fy - y0 as f64).clamp(0.0, 1.0),
        );
        let pt2 = |n: &str| -> Res<f64> {
            let g = |x, y| {
                reader
                    .read_window_2d(n, x, y, x + 1, y + 1)
                    .map(|w| w.values[0])
                    .map(f64::from)
                    .map_err(|x| e(n, x))
            };
            let (a, b, c, d) = (g(x0, y0)?, g(x1, y0)?, g(x0, y1)?, g(x1, y1)?);
            Ok((a * (1.0 - ax) + b * ax) * (1.0 - ay) + (c * (1.0 - ax) + d * ax) * ay)
        };
        let mut sfc = [f64::NAN; 6];
        for (i, s) in sfc.iter_mut().enumerate() {
            if let Some((n, units)) = Self::sfc_name(reader, i) {
                *s = sfc_convert(i, pt2(n)?, &units);
            }
        }
        let sfc = Sfc {
            psfc_hpa: sfc[0],
            orog_m: sfc[1],
            t2_c: sfc[2],
            td2_c: sfc[3],
            u10: sfc[4],
            v10: sfc[5],
        };
        match conv {
            Convention::Native => {
                let get = |i: usize| -> Res<Vec<f64>> {
                    let (n, u) = native_name(reader, i)
                        .ok_or_else(|| format!("missing native volume {:?}", NATIVE_NAMES[i]))?;
                    Ok(prof(n)?
                        .into_iter()
                        .map(|x| native_convert(i, x, &u))
                        .collect())
                };
                let p = get(0)?;
                let n = p.len();
                let omega = if native_name(reader, 6).is_some() {
                    get(6)?
                } else {
                    vec![f64::NAN; n]
                };
                Ok(RawColumn {
                    p,
                    z: get(1)?,
                    t: get(2)?,
                    td: get(3)?,
                    u: get(4)?,
                    v: get(5)?,
                    omega,
                    sfc,
                })
            }
            Convention::Iso => {
                let levels = reader
                    .variable("temperature_iso")
                    .unwrap()
                    .levels_hpa
                    .clone();
                let units = |n: &str| {
                    reader
                        .variable(n)
                        .map(|v| v.units.clone())
                        .unwrap_or_default()
                };
                let conv_t = |n: &str, v: Vec<f64>| {
                    if units(n) == "K" {
                        v.into_iter().map(|x| x - 273.15).collect()
                    } else {
                        v
                    }
                };
                let t = conv_t("temperature_iso", prof("temperature_iso")?);
                let td = if reader.variable("dewpoint_iso").is_some() {
                    conv_t("dewpoint_iso", prof("dewpoint_iso")?)
                } else {
                    let rh = prof("rh_iso")?;
                    t.iter()
                        .zip(&rh)
                        .map(|(t, rh)| dewpoint_c_from_rh(*t, *rh))
                        .collect()
                };
                Ok(RawColumn {
                    p: levels.iter().map(|l| *l as f64).collect(),
                    z: prof("height_iso")?,
                    t,
                    td,
                    u: prof("u_iso")?,
                    v: prof("v_iso")?,
                    omega: vec![f64::NAN; levels.len()],
                    sfc,
                })
            }
        }
    }

    /// Area-mean column over the selected cells (`(ix, iy)`): every primitive
    /// is averaged over one common finite-cell mask per level, then handed to
    /// the ordinary sounding path (bowecho `box_sounding.rs` method).
    pub fn box_column(reader: &HourReader, cells: &[(usize, usize)]) -> Res<RawColumn> {
        if cells.is_empty() {
            return Err("box contains no model grid cells".into());
        }
        let conv = Self::convention(reader)?;
        let names: Vec<&str> = match conv {
            Convention::Native => {
                let mut v = vec![];
                for i in 0..7 {
                    match native_name(reader, i) {
                        Some((n, _)) => v.push(n),
                        None if i == 6 => {}
                        None => return Err(format!("missing native volume {:?}", NATIVE_NAMES[i])),
                    }
                }
                v
            }
            Convention::Iso => {
                let moist = if reader.variable("dewpoint_iso").is_some() {
                    "dewpoint_iso"
                } else {
                    "rh_iso"
                };
                vec!["height_iso", "temperature_iso", moist, "u_iso", "v_iso"]
            }
        };
        let mut mats: Vec<Vec<Vec<f32>>> = Vec::new();
        for name in &names {
            let nl = reader.variable(name).unwrap().levels_hpa.len();
            let mut m = vec![vec![f32::NAN; cells.len()]; nl];
            for (ci, (ix, iy)) in cells.iter().enumerate() {
                let values = reader
                    .read_column_3d(name, *ix, *iy)
                    .map_err(|e| e.to_string())?;
                for (k, value) in values.into_iter().enumerate() {
                    m[k][ci] = value;
                }
            }
            mats.push(m);
        }
        let nl = mats.first().map(|m| m.len()).unwrap_or(0);
        let mut means = vec![vec![f64::NAN; nl]; names.len()];
        for k in 0..nl {
            let mut sums = vec![0.0f64; names.len()];
            let mut count = 0usize;
            for ci in 0..cells.len() {
                if mats.iter().all(|m| m[k][ci].is_finite()) {
                    count += 1;
                    for (vi, m) in mats.iter().enumerate() {
                        sums[vi] += m[k][ci] as f64;
                    }
                }
            }
            if count > 0 {
                for vi in 0..names.len() {
                    means[vi][k] = sums[vi] / count as f64;
                }
            }
        }
        // surface: mean over cells with every anchor finite
        let mut sfc_vals = vec![vec![f64::NAN; cells.len()]; 6];
        for (i, row) in sfc_vals.iter_mut().enumerate() {
            if let Some((n, units)) = Self::sfc_name(reader, i) {
                for (ci, (ix, iy)) in cells.iter().enumerate() {
                    row[ci] = sfc_convert(
                        i,
                        reader
                            .read_window_2d(n, *ix, *iy, ix + 1, iy + 1)
                            .map(|w| w.values[0])
                            .map_err(|e| format!("read {n}: {e}"))? as f64,
                        &units,
                    );
                }
            }
        }
        let mut s = [0.0f64; 6];
        let mut c = 0usize;
        for ci in 0..cells.len() {
            if (0..6).all(|i| sfc_vals[i][ci].is_finite()) {
                c += 1;
                for i in 0..6 {
                    s[i] += sfc_vals[i][ci];
                }
            }
        }
        let s: Vec<f64> = s
            .iter()
            .map(|x| if c > 0 { x / c as f64 } else { f64::NAN })
            .collect();
        let sfc = Sfc {
            psfc_hpa: s[0],
            orog_m: s[1],
            t2_c: s[2],
            td2_c: s[3],
            u10: s[4],
            v10: s[5],
        };
        let k2c = |v: &Vec<f64>| v.iter().map(|x| x - 273.15).collect::<Vec<_>>();
        match conv {
            Convention::Native => {
                let cv = |i: usize| -> Vec<f64> {
                    let u = native_name(reader, i).map(|x| x.1).unwrap_or_default();
                    means[i].iter().map(|x| native_convert(i, *x, &u)).collect()
                };
                Ok(RawColumn {
                    p: cv(0),
                    z: cv(1),
                    t: cv(2),
                    td: cv(3),
                    u: cv(4),
                    v: cv(5),
                    omega: if names.len() > 6 {
                        cv(6)
                    } else {
                        vec![f64::NAN; nl]
                    },
                    sfc,
                })
            }
            Convention::Iso => {
                let levels = reader.variable("height_iso").unwrap().levels_hpa.clone();
                let units = |n: &str| {
                    reader
                        .variable(n)
                        .map(|v| v.units.clone())
                        .unwrap_or_default()
                };
                let tc = |n: &str, v: &Vec<f64>| if units(n) == "K" { k2c(v) } else { v.clone() };
                let t = tc("temperature_iso", &means[1]);
                let td = if names[2] == "dewpoint_iso" {
                    tc("dewpoint_iso", &means[2])
                } else {
                    t.iter()
                        .zip(&means[2])
                        .map(|(t, rh)| dewpoint_c_from_rh(*t, *rh))
                        .collect()
                };
                Ok(RawColumn {
                    p: levels.iter().take(nl).map(|l| *l as f64).collect(),
                    z: means[0].clone(),
                    t,
                    td,
                    u: means[3].clone(),
                    v: means[4].clone(),
                    omega: vec![f64::NAN; nl],
                    sfc,
                })
            }
        }
    }
}

/// Same Bolton-style inversion rw-ui uses for RH-only stores.
pub fn dewpoint_c_from_rh(temperature_c: f64, rh_percent: f64) -> f64 {
    let rh = rh_percent.clamp(1.0, 100.0) / 100.0;
    let (a, b) = (17.67, 243.5);
    let gamma = (a * temperature_c) / (b + temperature_c) + rh.ln();
    (b * gamma) / (a - gamma)
}

/// Cells of the grid inside a lon/lat polygon (ray casting). A bbox is a
/// 4-corner polygon.
pub fn select_cells(grid: &GridFile, poly: &[(f64, f64)]) -> Vec<(usize, usize)> {
    let (mut w, mut s, mut e, mut n) = (
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    );
    for (x, y) in poly {
        w = w.min(*x);
        e = e.max(*x);
        s = s.min(*y);
        n = n.max(*y);
    }
    let inside = |x: f64, y: f64| {
        if x < w || x > e || y < s || y > n {
            return false;
        }
        let mut c = false;
        let m = poly.len();
        let mut j = m - 1;
        for i in 0..m {
            let (xi, yi) = poly[i];
            let (xj, yj) = poly[j];
            if ((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi) {
                c = !c;
            }
            j = i;
        }
        c
    };
    let mut out = Vec::new();
    for iy in 0..grid.ny {
        for ix in 0..grid.nx {
            let i = iy * grid.nx + ix;
            let (la, lo) = (grid.lat[i] as f64, grid.lon[i] as f64);
            if la.is_finite() && lo.is_finite() && inside(lo, la) {
                out.push((ix, iy));
            }
        }
    }
    out
}

/// Mean of several member columns, level by level, over one common finite
/// mask per level (the same rule as the area mean).
pub fn mean_columns(cols: &[RawColumn]) -> Res<RawColumn> {
    let first = cols.first().ok_or("no member columns")?;
    let nl = cols.iter().map(|c| c.levels()).min().unwrap_or(0);
    let mut out = RawColumn {
        omega: vec![f64::NAN; nl],
        ..Default::default()
    };
    for k in 0..nl {
        let ok: Vec<&RawColumn> = cols
            .iter()
            .filter(|c| {
                [c.p[k], c.z[k], c.t[k], c.td[k], c.u[k], c.v[k]]
                    .iter()
                    .all(|x| x.is_finite())
            })
            .collect();
        let m = |f: &dyn Fn(&RawColumn) -> f64| {
            if ok.is_empty() {
                f64::NAN
            } else {
                ok.iter().map(|c| f(c)).sum::<f64>() / ok.len() as f64
            }
        };
        out.p.push(m(&|c| c.p[k]));
        out.z.push(m(&|c| c.z[k]));
        out.t.push(m(&|c| c.t[k]));
        out.td.push(m(&|c| c.td[k]));
        out.u.push(m(&|c| c.u[k]));
        out.v.push(m(&|c| c.v[k]));
        let om: Vec<f64> = ok
            .iter()
            .map(|c| c.omega[k])
            .filter(|x| x.is_finite())
            .collect();
        out.omega[k] = if om.is_empty() {
            f64::NAN
        } else {
            om.iter().sum::<f64>() / om.len() as f64
        };
    }
    let sf: Vec<&Sfc> = cols
        .iter()
        .map(|c| &c.sfc)
        .filter(|s| {
            [s.psfc_hpa, s.orog_m, s.t2_c, s.td2_c, s.u10, s.v10]
                .iter()
                .all(|x| x.is_finite())
        })
        .collect();
    let ms = |f: &dyn Fn(&Sfc) -> f64| {
        if sf.is_empty() {
            f64::NAN
        } else {
            sf.iter().map(|s| f(s)).sum::<f64>() / sf.len() as f64
        }
    };
    out.sfc = Sfc {
        psfc_hpa: ms(&|s| s.psfc_hpa),
        orog_m: ms(&|s| s.orog_m),
        t2_c: ms(&|s| s.t2_c),
        td2_c: ms(&|s| s.td2_c),
        u10: ms(&|s| s.u10),
        v10: ms(&|s| s.v10),
    };
    let _ = first;
    Ok(out)
}

/// Linear-interpolated percentile of finite values (q in 0..=100).
pub fn percentile(vals: &mut Vec<f64>, q: f64) -> f64 {
    vals.retain(|x| x.is_finite());
    if vals.is_empty() {
        return f64::NAN;
    }
    vals.sort_by(|a, b| a.total_cmp(b));
    let pos = (q / 100.0).clamp(0.0, 1.0) * (vals.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    vals[lo] + (vals[hi] - vals[lo]) * (pos - lo as f64)
}

/// Member envelope per level: mean pressure, and lo/median/hi percentiles of
/// T and Td across members.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct Envelope {
    pub q_lo: f64,
    pub q_hi: f64,
    pub p: Vec<f64>,
    pub z: Vec<f64>,
    pub t_lo: Vec<f64>,
    pub t_med: Vec<f64>,
    pub t_hi: Vec<f64>,
    pub td_lo: Vec<f64>,
    pub td_med: Vec<f64>,
    pub td_hi: Vec<f64>,
    pub wspd_lo_kt: Vec<f64>,
    pub wspd_hi_kt: Vec<f64>,
}

pub fn envelope(cols: &[RawColumn], q_lo: f64, q_hi: f64) -> Envelope {
    let nl = cols.iter().map(|c| c.levels()).min().unwrap_or(0);
    let mut e = Envelope {
        q_lo,
        q_hi,
        ..Default::default()
    };
    for k in 0..nl {
        let mut p: Vec<f64> = cols
            .iter()
            .map(|c| c.p[k])
            .filter(|x| x.is_finite())
            .collect();
        let pm = if p.is_empty() {
            f64::NAN
        } else {
            p.iter().sum::<f64>() / p.len() as f64
        };
        p.clear();
        let mut z: Vec<f64> = cols
            .iter()
            .map(|c| c.z[k])
            .filter(|x| x.is_finite())
            .collect();
        let zm = if z.is_empty() {
            f64::NAN
        } else {
            z.iter().sum::<f64>() / z.len() as f64
        };
        z.clear();
        let get = |f: &dyn Fn(&RawColumn) -> f64| cols.iter().map(f).collect::<Vec<f64>>();
        let t = get(&|c| c.t[k]);
        let td = get(&|c| c.td[k]);
        let ws = get(&|c| c.u[k].hypot(c.v[k]) * 1.943_844_5);
        e.p.push(pm);
        e.z.push(zm);
        e.t_lo.push(percentile(&mut t.clone(), q_lo));
        e.t_med.push(percentile(&mut t.clone(), 50.0));
        e.t_hi.push(percentile(&mut t.clone(), q_hi));
        e.td_lo.push(percentile(&mut td.clone(), q_lo));
        e.td_med.push(percentile(&mut td.clone(), 50.0));
        e.td_hi.push(percentile(&mut td.clone(), q_hi));
        e.wspd_lo_kt.push(percentile(&mut ws.clone(), q_lo));
        e.wspd_hi_kt.push(percentile(&mut ws.clone(), q_hi));
    }
    e
}

/// p10/p25/p50/p75/p90 per level across members, for T, Td and wind speed.
pub fn percentiles5(cols: &[RawColumn]) -> serde_json::Value {
    let nl = cols.iter().map(|c| c.levels()).min().unwrap_or(0);
    let qs = [10.0, 25.0, 50.0, 75.0, 90.0];
    let names = ["p10", "p25", "p50", "p75", "p90"];
    let mut p = vec![];
    let mut z = vec![];
    let mut out: Vec<Vec<Vec<f64>>> = vec![vec![vec![]; 5]; 3];
    let r = |x: f64| {
        if x.is_finite() {
            serde_json::json!((x * 100.0).round() / 100.0)
        } else {
            serde_json::Value::Null
        }
    };
    for k in 0..nl {
        let mean = |f: &dyn Fn(&RawColumn) -> f64| {
            let v: Vec<f64> = cols.iter().map(f).filter(|x| x.is_finite()).collect();
            if v.is_empty() {
                f64::NAN
            } else {
                v.iter().sum::<f64>() / v.len() as f64
            }
        };
        p.push(r(mean(&|c| c.p[k])));
        z.push(r(mean(&|c| c.z[k])));
        let fields: [Vec<f64>; 3] = [
            cols.iter().map(|c| c.t[k]).collect(),
            cols.iter().map(|c| c.td[k]).collect(),
            cols.iter()
                .map(|c| c.u[k].hypot(c.v[k]) * 1.943_844_5)
                .collect(),
        ];
        for (fi, f) in fields.iter().enumerate() {
            for (qi, q) in qs.iter().enumerate() {
                out[fi][qi].push(percentile(&mut f.clone(), *q));
            }
        }
    }
    let block = |fi: usize| {
        let mut m = serde_json::Map::new();
        for (qi, n) in names.iter().enumerate() {
            m.insert(
                n.to_string(),
                serde_json::Value::Array(out[fi][qi].iter().map(|x| r(*x)).collect()),
            );
        }
        serde_json::Value::Object(m)
    };
    serde_json::json!({"p": p, "z": z, "t": block(0), "td": block(1), "wspd": block(2)})
}

#[cfg(test)]
mod engine_store_tests {
    use super::*;
    #[test]
    fn engine_isobaric_store_profile_units_and_surface_anchors() {
        let path =
            std::env::temp_dir().join(format!("gpuwm-sounding-store-{}.rws", std::process::id()));
        let mut w = rw_store::writer::HourWriter::new(
            "forecast",
            "20260101000000",
            0,
            1,
            1,
            "test",
            "test",
        );
        for (name, unit, values) in [
            ("temperature_iso", "K", [290., 270., 230.]),
            ("dewpoint_iso", "K", [285., 265., 220.]),
            ("height_iso", "m", [800., 3000., 9000.]),
            ("u_iso", "m/s", [5., 15., 25.]),
            ("v_iso", "m/s", [3., 6., 10.]),
        ] {
            let planes: Vec<Vec<f32>> = values.into_iter().map(|v| vec![v]).collect();
            w.add_pressure3d(
                name,
                unit,
                serde_json::json!({}),
                &[900, 700, 300],
                &planes.iter().map(|v| v.as_slice()).collect::<Vec<_>>(),
            )
            .unwrap();
        }
        for (name, unit, value) in [
            ("surface_pressure", "Pa", 98000.),
            ("orography", "m", 200.),
            ("temperature_2m", "K", 295.),
            ("dewpoint_2m", "K", 288.),
            ("u_10m", "m/s", 4.),
            ("v_10m", "m/s", 2.),
        ] {
            w.add_surface2d(name, unit, serde_json::json!({}), &[value])
                .unwrap();
        }
        w.finish(&path).unwrap();
        let r = HourReader::open(&path).unwrap();
        let c = Store::point_column(&r, 0., 0.).unwrap();
        assert_eq!(c.p, vec![900., 700., 300.]);
        assert!((c.t[0] - 16.85).abs() < 0.001);
        assert_eq!(c.sfc.psfc_hpa, 980.);
        assert_eq!(c.sfc.u10, 4.);
        assert!((c.sfc.td2_c - 14.85).abs() < 0.001);
        let analyzed = crate::analysis::to_column(&c, Default::default()).unwrap();
        assert_eq!(analyzed.pressure_hpa[0], 980.);
        std::fs::remove_file(path).unwrap();
    }
}
