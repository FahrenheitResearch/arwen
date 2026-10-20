//! Ordinary WRF/rw-store inputs; no cycling-layout or assimilation assumptions.
use crate::store::{RawColumn, Res, Sfc, Store};
use rw_store::{
    grid::{GridFile, GridLocator},
    reader::HourReader,
};
use serde::Deserialize;
use std::path::PathBuf;
use wrf_core::WrfFile;

#[derive(Clone, Deserialize)]
pub struct Source {
    pub path: PathBuf,
    pub domain: String,
    pub member: String,
}
#[derive(Clone, Deserialize)]
pub struct Point {
    pub label: String,
    pub lat: f64,
    pub lon: f64,
}
pub struct Frame {
    pub domain: String,
    pub member: String,
    pub valid: i64,
    pub columns: Vec<RawColumn>,
}
fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}
fn timestamp(s: &str) -> Res<i64> {
    let s = s.replace('_', "T");
    chrono::NaiveDateTime::parse_from_str(s.trim_end_matches('Z'), "%Y-%m-%dT%H:%M:%S")
        .map(|t| t.and_utc().timestamp())
        .map_err(err)
}
fn locate(g: &GridFile, point: &Point) -> Res<(f64, f64)> {
    GridLocator::build(g)
        .locate(point.lat, point.lon)
        .filter(|(x, y)| {
            *x >= 0.0 && *y >= 0.0 && *x <= (g.nx - 1) as f64 && *y <= (g.ny - 1) as f64
        })
        .ok_or_else(|| {
            format!(
                "{} ({},{}) is outside the model grid",
                point.label, point.lat, point.lon
            )
        })
}
/// Bolton vapour-pressure inversion: q is water-vapour mixing ratio (kg/kg).
fn dewpoint(p_pa: f64, q: f64) -> f64 {
    let e = (p_pa * q.max(1e-12) / (0.622 + q.max(1e-12)) / 100.0).max(1e-12);
    let l = (e / 6.112).ln();
    243.5 * l / (17.67 - l)
}
pub fn read(source: &Source, points: &[Point]) -> Res<Vec<Frame>> {
    if source.path.extension().is_some_and(|x| x == "rws") {
        let r = HourReader::open(&source.path).map_err(err)?;
        let valid = r.meta().exact_time().map(|t| t.valid_unix).or_else(|| {
            // Older whole-hour stores encode their UTC origin in the run key.
            let run = &r.meta().run;
            let stamp = run.strip_prefix("local_").and_then(|s| s.split('_').next()).unwrap_or(run);
            let origin = ["%Y%m%d%H%M%S", "%Y%m%d%H%M", "%Y%m%d%H"].iter().find_map(|fmt| {
                chrono::NaiveDateTime::parse_from_str(stamp, fmt).ok().map(|t|t.and_utc().timestamp())
            });
            origin.map(|t| t + i64::from(r.meta().forecast_hour)*3600)
        }).ok_or("rw-store input needs exact time metadata or a UTC run origin; cannot infer valid time from a storage slot")?;
        let grid = GridFile::open(&source.path.parent().unwrap().join("grid.rwg")).map_err(err)?;
        let columns = points
            .iter()
            .map(|p| {
                let (x, y) = locate(&grid, p)?;
                Store::point_column(&r, x, y)
            })
            .collect::<Res<_>>()?;
        return Ok(vec![Frame {
            domain: source.domain.clone(),
            member: source.member.clone(),
            valid,
            columns,
        }]);
    }
    let f = WrfFile::open(&source.path).map_err(err)?;
    let mut frames = vec![];
    for (ti, time) in f.times().map_err(err)?.iter().enumerate() {
        f.prepare_cache_for_time(ti);
        let g = GridFile {
            nx: f.nx,
            ny: f.ny,
            lat: f.xlat(ti).map_err(err)?.iter().map(|x| *x as f32).collect(),
            lon: f
                .xlong(ti)
                .map_err(err)?
                .iter()
                .map(|x| *x as f32)
                .collect(),
            projection: None,
            hash: String::new(),
        };
        let xy = points
            .iter()
            .map(|p| locate(&g, p))
            .collect::<Res<Vec<_>>>()?;
        let p = f.full_pressure(ti).map_err(err)?;
        let z = f.height_msl(ti).map_err(err)?;
        let t = f.temperature_c(ti).map_err(err)?;
        let q = f.qvapor(ti).map_err(err)?;
        let w = f.w_destag(ti).map_err(err)?;
        // Hydrostatic conversion of WRF vertical velocity to pressure velocity (Pa/s).
        let omega: Vec<f64> = p
            .iter()
            .zip(t.iter())
            .zip(q.iter())
            .zip(w.iter())
            .map(|(((p, t), q), w)| -9.80665 * p * w / (287.05 * (t + 273.15) * (1.0 + 0.61 * q)))
            .collect();
        let u = f.u_destag(ti).map_err(err)?;
        let v = f.v_destag(ti).map_err(err)?;
        let ps = f.psfc(ti).map_err(err)?;
        let terrain = f.terrain(ti).map_err(err)?;
        let t2 = f.t2(ti).map_err(err)?;
        let q2 = f.q2(ti).map_err(err)?;
        let u10 = f.u10(ti).map_err(err)?;
        let v10 = f.v10(ti).map_err(err)?;
        // Rotation must precede horizontal interpolation. WRF stores grid-relative winds.
        let ca = f.cosalpha(ti).map_err(err)?;
        let sa = f.sinalpha(ti).map_err(err)?;
        let n = f.nxy();
        let earth = |us: &[f64], vs: &[f64]| -> (Vec<f64>, Vec<f64>) {
            us.iter()
                .zip(vs)
                .enumerate()
                .map(|(i, (u, v))| (u * ca[i % n] - v * sa[i % n], v * ca[i % n] + u * sa[i % n]))
                .unzip()
        };
        let (u, v) = earth(&u, &v);
        let (u10, v10) = earth(&u10, &v10);
        let td: Vec<f64> = p
            .iter()
            .zip(q.iter())
            .map(|(p, q)| dewpoint(*p, *q))
            .collect();
        let columns = xy
            .iter()
            .map(|(x, y)| {
                let sample = |a: &[f64], k: usize| {
                    let ix = x.floor() as usize;
                    let iy = y.floor() as usize;
                    let jx = (ix + 1).min(f.nx - 1);
                    let jy = (iy + 1).min(f.ny - 1);
                    let ax = x - ix as f64;
                    let ay = y - iy as f64;
                    let g = |i, j| a[k * n + j * f.nx + i];
                    (g(ix, iy) * (1.0 - ax) + g(jx, iy) * ax) * (1.0 - ay)
                        + (g(ix, jy) * (1.0 - ax) + g(jx, jy) * ax) * ay
                };
                let mut c = RawColumn {
                    sfc: Sfc {
                        psfc_hpa: sample(&ps, 0) / 100.0,
                        orog_m: sample(&terrain, 0),
                        t2_c: sample(&t2, 0) - 273.15,
                        td2_c: dewpoint(sample(&ps, 0), sample(&q2, 0)),
                        u10: sample(&u10, 0),
                        v10: sample(&v10, 0),
                    },
                    ..Default::default()
                };
                // Derive Td on the grid before bilinear sampling, like the site store.
                for k in 0..f.nz {
                    c.p.push(sample(&p, k) / 100.0);
                    c.z.push(sample(&z, k));
                    c.t.push(sample(&t, k));
                    c.td.push(sample(&td, k));
                    c.u.push(sample(&u, k));
                    c.v.push(sample(&v, k));
                    c.omega.push(sample(&omega, k));
                }
                c
            })
            .collect();
        frames.push(Frame {
            domain: source.domain.clone(),
            member: source.member.clone(),
            valid: timestamp(time)?,
            columns,
        });
        f.clear_cache();
    }
    Ok(frames)
}
/// Common log-pressure ladder for ensemble statistics. Never average unrelated model levels.
pub fn common_columns(cols: &[RawColumn]) -> Vec<RawColumn> {
    let levels: Vec<f64> = (2..=40).rev().map(|n| n as f64 * 25.0).collect();
    cols.iter()
        .map(|c| {
            let mut o = RawColumn {
                sfc: c.sfc,
                ..Default::default()
            };
            for p in &levels {
                let pair = c.p.windows(2).position(|w| w[0] >= *p && w[1] <= *p);
                let interp = |a: &[f64]| {
                    pair.map(|k| {
                        let w = (p.ln() - c.p[k].ln()) / (c.p[k + 1].ln() - c.p[k].ln());
                        a[k] + w * (a[k + 1] - a[k])
                    })
                    .unwrap_or(f64::NAN)
                };
                o.p.push(*p);
                o.z.push(interp(&c.z));
                o.t.push(interp(&c.t));
                o.td.push(interp(&c.td));
                o.u.push(interp(&c.u));
                o.v.push(interp(&c.v));
                o.omega.push(interp(&c.omega));
            }
            o
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn synthetic_profile_surface_units_common_pressure_and_spread() {
        let base = RawColumn {
            p: vec![950., 800., 600., 400., 200., 75.],
            z: vec![500., 1800., 4000., 7000., 11500., 18000.],
            t: vec![23., 12., -2., -20., -52., -65.],
            td: vec![19., 8., -9., -30., -60., -75.],
            u: vec![3., 8., 15., 22., 30., 40.],
            v: vec![5., 7., 10., 15., 20., 25.],
            omega: vec![f64::NAN; 6],
            sfc: Sfc {
                psfc_hpa: 980.,
                orog_m: 250.,
                t2_c: 26.,
                td2_c: 21.,
                u10: 2.,
                v10: 4.,
            },
        };
        let mut warm = base.clone();
        for t in &mut warm.t {
            *t += 2.;
        }
        let cols = common_columns(&[base.clone(), warm]);
        let k = cols[0].p.iter().position(|p| *p == 800.).unwrap();
        assert_eq!(cols[0].t[k], 12.);
        let mean = crate::store::mean_columns(&cols).unwrap();
        assert_eq!(mean.t[k], 13.);
        let mut samples = vec![cols[0].t[k], cols[1].t[k]];
        assert!((crate::store::percentile(&mut samples, 10.) - 12.2).abs() < 1e-9);
        assert!((crate::store::percentile(&mut samples, 90.) - 13.8).abs() < 1e-9);
        let column = crate::analysis::to_column(&base, Default::default()).unwrap();
        assert_eq!(column.pressure_hpa[0], 980.);
        assert_eq!(column.temperature_c[0], 26.);
        let a = crate::analysis::analyze(column, &Default::default()).unwrap();
        assert!(!a.prof.inner.u.is_empty());
        assert!(dewpoint(100000., 0.014) > 18.);
    }
}
