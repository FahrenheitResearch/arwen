//! Monthly precipitation-rate normals to a supplied regular grid. Rust owns
//! NetCDF decoding, unit checks, calendar weighting and bilinear interpolation.
use crate::{Result, hash, publish, regular_axis, text};
use netcdf_writer::{AttrValue, NcFormat, NcType, NcWriter, Schema, VarData};
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::json;
use std::{
    fs,
    path::{Path, PathBuf},
    time::Instant,
};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    source: PathBuf,
    template: PathBuf,
    variable: String,
    latitude: String,
    longitude: String,
    days_per_month: [f64; 12],
    provenance: String,
    license: String,
    output: PathBuf,
    receipt: PathBuf,
}
fn bracket(axis: &[f64], x: f64) -> (usize, usize, f64) {
    let f = ((x - axis[0]) / (axis[1] - axis[0])).clamp(0., (axis.len() - 1) as f64);
    let a = (f.floor() as usize).min(axis.len() - 2);
    (a, a + 1, f - a as f64)
}
pub fn run(path: &Path) -> Result<()> {
    let start = Instant::now();
    let r: Request = serde_json::from_slice(&fs::read(path)?)?;
    if r.output.exists() || r.receipt.exists() {
        return Err("preserving existing climatology artifacts".into());
    }
    if r.provenance.is_empty()
        || r.license.is_empty()
        || r.days_per_month
            .iter()
            .any(|v| !v.is_finite() || *v < 28. || *v > 31.)
    {
        return Err("climatology requires provenance, license and twelve month lengths".into());
    }
    let src = netcrust::open(&r.source)?;
    let sy = src.read_f64(&r.latitude)?;
    let sx = src.read_f64(&r.longitude)?;
    if !regular_axis(&sy)
        || !regular_axis(&sx)
        || sx[1] <= sx[0]
        || ((sx[1] - sx[0]) * sx.len() as f64 - 360.).abs() > 1e-5
    {
        return Err(
            "monthly climatology requires regular axes and a global eastward longitude axis".into(),
        );
    }
    if text(&src, &r.variable, "units")? != "mm/day" {
        return Err("monthly climatology requires precipitation rate units mm/day".into());
    }
    if src
        .variable(&r.variable)
        .ok_or("missing monthly precipitation")?
        .shape()
        != vec![12, sy.len(), sx.len()]
    {
        return Err("climatology requires [12 months,latitude,longitude]".into());
    }
    let values = src.read_f64(&r.variable)?;
    let template = netcrust::open(&r.template)?;
    let lat = template.read_f64("latitude")?;
    let lon = template.read_f64("longitude")?;
    if !regular_axis(&lat) || !regular_axis(&lon) {
        return Err("template axes must be regular".into());
    }
    let annual: Result<Vec<f64>> = (0..lat.len() * lon.len())
        .into_par_iter()
        .map(|i| {
            let (y0, y1, wy) = bracket(&sy, lat[i / lon.len()]);
            let f = (lon[i % lon.len()] - sx[0]).rem_euclid(360.) / (sx[1] - sx[0]);
            let x0 = f.floor() as usize % sx.len();
            let x1 = (x0 + 1) % sx.len();
            let wx = f - f.floor();
            let mut total = 0.;
            for m in 0..12 {
                let offset = m * sy.len() * sx.len();
                let a = [
                    values[offset + y0 * sx.len() + x0],
                    values[offset + y0 * sx.len() + x1],
                    values[offset + y1 * sx.len() + x0],
                    values[offset + y1 * sx.len() + x1],
                ];
                if a.iter().any(|v| !v.is_finite() || *v < 0. || *v > 1000.) {
                    return Err(format!("missing/unphysical monthly normal near cell {i}").into());
                }
                total += ((a[0] * (1. - wx) + a[1] * wx) * (1. - wy)
                    + (a[2] * (1. - wx) + a[3] * wx) * wy)
                    * r.days_per_month[m];
            }
            Ok(total)
        })
        .collect();
    let annual = annual?;
    if let Some(p) = r.output.parent() {
        fs::create_dir_all(p)?;
    }
    let mut s = Schema::new(NcFormat::Offset64);
    let yd = s.def_dim("latitude", lat.len(), false)?;
    let xd = s.def_dim("longitude", lon.len(), false)?;
    let yv = s.def_var("latitude", NcType::Double, &[yd])?;
    let xv = s.def_var("longitude", NcType::Double, &[xd])?;
    let v = s.def_var("annual_rain_mm", NcType::Double, &[yd, xd])?;
    s.put_var_attr(v, "units", AttrValue::Text("mm".into()))?;
    s.put_global_attr("source", AttrValue::Text(r.provenance.clone()))?;
    let mut w = NcWriter::create(&r.output, s)?;
    w.write_var(yv, VarData::F64(&lat))?;
    w.write_var(xv, VarData::F64(&lon))?;
    w.write_var(v, VarData::F64(&annual))?;
    w.finish()?;
    publish(
        &r.receipt,
        &serde_json::to_vec_pretty(
            &json!({"schema":"arwen.fire-climatology.v1","source":r.source,"source_sha256":hash(&r.source)?,"template":r.template,"template_sha256":hash(&r.template)?,"provenance":r.provenance,"license":r.license,"days_per_month":r.days_per_month,"interpolation":"bilinear, cyclic longitude","output":r.output,"output_sha256":hash(&r.output)?,"annual_rain_mm_range":[annual.iter().copied().fold(f64::INFINITY,f64::min),annual.iter().copied().fold(f64::NEG_INFINITY,f64::max)],"wall_seconds":start.elapsed().as_secs_f64(),"parallelism":rayon::current_num_threads()}),
        )?,
    )?;
    Ok(())
}
