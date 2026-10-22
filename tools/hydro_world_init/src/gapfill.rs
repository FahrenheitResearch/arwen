//! Holes in a fetched source series, and filling them from the row's next source.
//!
//! Breakage this exists for (2.9.4 WOOF readiness, 2026-10-22): Google's ARCO
//! ERA5 Zarr store declared ERA5T through 2026-10-16 but served whole UTC days
//! of NaN inside that range (2026-10-10 and 2026-10-14, every variable), while
//! the same bucket's raw ERA5T day files held those days complete. A spin-up
//! over such a day either gave up ("missing over a catchment", hydro off) or,
//! when carrying, ran short (1,992 of 2,160 h).
//!
//!   forcing-gaps  every (file, record, variable) with a non-finite value in
//!                 the fetched crop (or a set missing__ flag), files and
//!                 records in parallel
//!   forcing-fill  a patched copy of each holed file: the named records of the
//!                 named variables replaced by the fill source's values on the
//!                 same crop (a day file on the source's global grid), every
//!                 other byte kept; the receipt says what is still missing
use crate::ncread;
use crate::time;
use crate::Result;
use netcdf_writer::{NcPatcher, NcType, VarData};
use rayon::prelude::*;
use serde_json::{json, Value};
use std::collections::BTreeMap;

fn strings(v: &Value, k: &str) -> Result<Vec<String>> {
    v[k].as_array()
        .ok_or_else(|| format!("request lacks array field {k}"))?
        .iter()
        .map(|x| x.as_str().map(str::to_string).ok_or_else(|| format!("{k} holds a non-string").into()))
        .collect()
}

/// Variables of `vars` that `src` holds at every record (a request may list
/// state variables a forcing file does not carry).
fn present(src: &ncread::Source, vars: &[String]) -> Vec<String> {
    vars.iter().filter(|v| src.file.variable(v).is_some()).cloned().collect()
}

pub fn cmd_forcing_gaps(r: &Value) -> Result<Value> {
    let files = strings(r, "files")?;
    let vars = strings(r, "variables")?;
    let per_file: Vec<Vec<Value>> = files
        .par_iter()
        .map(|path| -> Result<Vec<Value>> {
            let src = ncread::open(path)?;
            let names = present(&src, &vars);
            let jobs: Vec<(usize, &String)> = (0..src.times.len()).flat_map(|k| names.iter().map(move |v| (k, v))).collect();
            let holed: Vec<(usize, String)> = jobs
                .par_iter()
                .map(|&(k, v)| -> Result<Option<(usize, String)>> {
                    let field = ncread::field(&src, v, k)?;
                    Ok(field.iter().any(|x| !x.is_finite()).then(|| (k, v.clone())))
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .flatten()
                .collect();
            let mut by_record: BTreeMap<usize, Vec<String>> = BTreeMap::new();
            for (k, v) in holed {
                by_record.entry(k).or_default().push(v);
            }
            Ok(by_record
                .into_iter()
                .map(|(k, v)| json!({"file": path, "time": time::iso(src.times[k]), "variables": v}))
                .collect())
        })
        .collect::<Result<_>>()?;
    let gaps: Vec<Value> = per_file.into_iter().flatten().collect();
    Ok(json!({"schema": "arwen.hydro-world.forcing-gaps.v1", "files": files.len(), "gaps": gaps}))
}

/// The one (time, latitude, longitude) data variable of a source day file.
fn day_variable(src: &ncread::Source) -> Result<String> {
    let found: Vec<String> = src
        .file
        .variables()?
        .into_iter()
        .filter(|v| v.ndim() == 3)
        .map(|v| v.name().to_string())
        .collect();
    match &found[..] {
        [one] => Ok(one.clone()),
        _ => Err(format!("{}: a fill day file holds one (time, latitude, longitude) variable, found {found:?}", src.path).into()),
    }
}

/// Index of each wanted coordinate value in a source axis (longitudes compared modulo 360).
fn axis_index(source: &[f64], wanted: &[f64], longitude: bool, what: &str) -> Result<Vec<usize>> {
    let norm = |x: f64| if longitude { x.rem_euclid(360.0) } else { x };
    wanted
        .iter()
        .map(|w| {
            source
                .iter()
                .position(|s| {
                    let d = (norm(*s) - norm(*w)).abs();
                    d < 1e-6 || (longitude && (d - 360.0).abs() < 1e-6)
                })
                .ok_or_else(|| format!("{what} {w} of the crop is not on the fill source's grid").into())
        })
        .collect()
}

pub fn cmd_forcing_fill(r: &Value) -> Result<Value> {
    let fills = r["fills"].as_array().ok_or("request lacks array field fills")?;
    let receipts: Vec<Value> = fills
        .par_iter()
        .map(|fill| -> Result<Value> {
            let file = fill["file"].as_str().ok_or("fill lacks file")?;
            let output = fill["output"].as_str().ok_or("fill lacks output")?;
            let src = ncread::open(file)?;
            let lat = src.file.read_f64("latitude").map_err(|e| format!("{file}: latitude: {e}"))?;
            let lon = src.file.read_f64("longitude").map_err(|e| format!("{file}: longitude: {e}"))?;
            let mut patch = NcPatcher::open(file, output).map_err(|e| format!("{output}: {e}"))?;
            let types: BTreeMap<String, NcType> = patch.variables().into_iter().map(|v| (v.name, v.ty)).collect();
            let mut patched = Vec::new();
            let mut still = Vec::new();
            let mut opened: BTreeMap<String, (ncread::Source, String, Vec<usize>, Vec<usize>, usize)> = BTreeMap::new();
            for rec in fill["records"].as_array().ok_or("fill lacks records")? {
                let stamp = rec["time"].as_str().ok_or("record lacks time")?;
                let k = ncread::record_index(&src, time::parse(stamp)?)?;
                for source in rec["sources"].as_array().ok_or("record lacks sources")? {
                    let var = source["variable"].as_str().ok_or("source lacks variable")?;
                    let path = source["path"].as_str().ok_or("source lacks path")?;
                    if !opened.contains_key(path) {
                        let day = ncread::open(path)?;
                        let name = day_variable(&day)?;
                        let dlat = day.file.read_f64("latitude").map_err(|e| format!("{path}: latitude: {e}"))?;
                        let dlon = day.file.read_f64("longitude").map_err(|e| format!("{path}: longitude: {e}"))?;
                        let yi = axis_index(&dlat, &lat, false, "latitude")?;
                        let xi = axis_index(&dlon, &lon, true, "longitude")?;
                        let nx = dlon.len();
                        opened.insert(path.to_string(), (day, name, yi, xi, nx));
                    }
                    let (day, name, yi, xi, nx) = &opened[path];
                    let dk = ncread::record_index(day, time::parse(stamp)?)?;
                    let whole = ncread::field(day, name, dk)?;
                    let whole = &whole;
                    let crop: Vec<f64> = yi.iter().flat_map(|j| xi.iter().map(move |i| whole[j * nx + i])).collect();
                    let missing = crop.iter().filter(|x| !x.is_finite()).count();
                    if missing > 0 {
                        still.push(json!({"time": stamp, "variable": var, "non_finite": missing, "source": path}));
                        continue;
                    }
                    match types.get(var) {
                        Some(NcType::Double) => patch.put(var, Some(k as u64), VarData::F64(&crop))?,
                        Some(NcType::Float) => {
                            let narrow: Vec<f32> = crop.iter().map(|x| *x as f32).collect();
                            patch.put(var, Some(k as u64), VarData::F32(&narrow))?
                        }
                        t => return Err(format!("{file}: {var} is {t:?}; a fill writes a float or double field").into()),
                    }
                    let mask = format!("missing__{var}");
                    if types.get(&mask) == Some(&NcType::Byte) {
                        patch.put(&mask, Some(k as u64), VarData::I8(&vec![0i8; crop.len()]))?;
                    }
                    patched.push(json!({"time": stamp, "variable": var, "source": path}));
                }
            }
            patch.finish().map_err(|e| format!("{output}: {e}"))?;
            Ok(json!({"file": file, "output": output, "patched": patched, "still_missing": still}))
        })
        .collect::<Result<_>>()?;
    Ok(json!({"schema": "arwen.hydro-world.forcing-fill.v1", "files": receipts}))
}

#[cfg(test)]
mod tests {
    use super::*;
    use netcdf_writer::{AttrValue, NcFormat, NcWriter, Schema};

    const T0: i64 = 1_791_590_400; // 2026-10-10T00:00:00Z

    /// A crop the way rw_zarr writes it: double field plus missing__ flags,
    /// two records, the second all NaN (the store's hole).
    fn crop(path: &std::path::Path) {
        let mut s = Schema::new(NcFormat::Offset64);
        let td = s.def_dim("time", 0, true).unwrap();
        let yd = s.def_dim("latitude", 2, false).unwrap();
        let xd = s.def_dim("longitude", 3, false).unwrap();
        let t = s.def_var("time", NcType::Double, &[td]).unwrap();
        s.put_var_attr(t, "units", AttrValue::Text("seconds since 1970-01-01 00:00:00".into())).unwrap();
        let lat = s.def_var("latitude", NcType::Double, &[yd]).unwrap();
        let lon = s.def_var("longitude", NcType::Double, &[xd]).unwrap();
        let v = s.def_var("total_precipitation", NcType::Double, &[td, yd, xd]).unwrap();
        let m = s.def_var("missing__total_precipitation", NcType::Byte, &[td, yd, xd]).unwrap();
        let mut w = NcWriter::create(path, s).unwrap();
        w.write_var(lat, VarData::F64(&[23.0, 22.75])).unwrap();
        // A crop across the antimeridian side the store writes as negative longitudes.
        w.write_var(lon, VarData::F64(&[-0.25, 0.0, 0.25])).unwrap();
        for (r, field) in [[1.0; 6], [f64::NAN; 6]].iter().enumerate() {
            w.write_record(r as u64, t, VarData::F64(&[(T0 + 3600 * r as i64) as f64])).unwrap();
            w.write_record(r as u64, v, VarData::F64(field)).unwrap();
            w.write_record(r as u64, m, VarData::I8(&[0; 6])).unwrap();
        }
        w.finish().unwrap();
    }

    /// The next source's day file: one float variable on a wider global-style grid.
    fn day(path: &std::path::Path) {
        let mut s = Schema::new(NcFormat::Offset64);
        let td = s.def_dim("valid_time", 0, true).unwrap();
        let yd = s.def_dim("latitude", 3, false).unwrap();
        let xd = s.def_dim("longitude", 4, false).unwrap();
        let t = s.def_var("valid_time", NcType::Double, &[td]).unwrap();
        s.put_var_attr(t, "units", AttrValue::Text("seconds since 1970-01-01".into())).unwrap();
        let lat = s.def_var("latitude", NcType::Double, &[yd]).unwrap();
        let lon = s.def_var("longitude", NcType::Double, &[xd]).unwrap();
        let v = s.def_var("tp", NcType::Float, &[td, yd, xd]).unwrap();
        let mut w = NcWriter::create(path, s).unwrap();
        w.write_var(lat, VarData::F64(&[23.25, 23.0, 22.75])).unwrap();
        w.write_var(lon, VarData::F64(&[0.0, 0.25, 0.5, 359.75])).unwrap();
        for r in 0..2u64 {
            w.write_record(r, t, VarData::F64(&[(T0 + 3600 * r as i64) as f64])).unwrap();
            let field: Vec<f32> = (0..12).map(|i| (100 * r + i) as f32).collect();
            w.write_record(r, v, VarData::F32(&field)).unwrap();
        }
        w.finish().unwrap();
    }

    #[test]
    fn a_hole_is_found_and_filled_from_the_next_source_on_the_same_crop() {
        let dir = std::env::temp_dir().join(format!("gapfill-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (file, raw, out) = (dir.join("day.nc"), dir.join("raw.nc"), dir.join("filled.nc"));
        let _ = std::fs::remove_file(&out);
        crop(&file);
        day(&raw);
        let f = file.to_str().unwrap();
        let gaps = cmd_forcing_gaps(&json!({"files": [f], "variables": ["total_precipitation", "snowfall"]})).unwrap();
        assert_eq!(gaps["gaps"], json!([{"file": f, "time": "2026-10-10T01:00:00Z", "variables": ["total_precipitation"]}]));
        let filled = cmd_forcing_fill(&json!({"fills": [{"file": f, "output": out.to_str().unwrap(), "records": [
            {"time": "2026-10-10T01:00:00Z", "sources": [{"variable": "total_precipitation", "path": raw.to_str().unwrap()}]}]}]})).unwrap();
        assert_eq!(filled["files"][0]["still_missing"], json!([]));
        let src = ncread::open(out.to_str().unwrap()).unwrap();
        // The complete record keeps its bytes; the hole takes the day file's values
        // at latitudes 23.0, 22.75 and longitudes 359.75, 0.0, 0.25 (record 1: 100 + index).
        assert_eq!(ncread::field(&src, "total_precipitation", 0).unwrap(), vec![1.0; 6]);
        assert_eq!(ncread::field(&src, "total_precipitation", 1).unwrap(), vec![107.0, 104.0, 105.0, 111.0, 108.0, 109.0]);
        let again = cmd_forcing_gaps(&json!({"files": [out.to_str().unwrap()], "variables": ["total_precipitation"]})).unwrap();
        assert_eq!(again["gaps"], json!([]));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
