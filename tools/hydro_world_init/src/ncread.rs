//! Reading the source NetCDF files (rw_zarr's CF output or any CF file with
//! the row's variable names): coordinates, record times and 2-D fields.
use crate::grid::GridSpec;
use crate::table::Row;
use crate::Result;

pub struct Source {
    pub file: netcrust::File,
    pub path: String,
    /// Valid times, seconds since 1970-01-01T00:00:00Z.
    pub times: Vec<i64>,
}

pub fn open(path: &str) -> Result<Source> {
    let file = netcrust::File::open(path).map_err(|e| format!("{path}: {e}"))?;
    // CF time coordinate: "time" (rw_zarr, most CF files) or "valid_time"
    // (the Copernicus CDS netCDF4 conversion).
    let tname = ["time", "valid_time"]
        .into_iter()
        .find(|n| file.variable(n).is_some())
        .ok_or_else(|| format!("{path}: no time or valid_time coordinate"))?;
    let raw = file.read_f64(tname).map_err(|e| format!("{path}: {tname}: {e}"))?;
    let units = file
        .variable(tname)
        .and_then(|v| v.attribute("units").and_then(|a| a.as_string()).map(str::to_string))
        .unwrap_or_default();
    let scale = if units.starts_with("seconds since 1970-01-01") {
        1.0
    } else if units.starts_with("minutes since 1970-01-01") {
        60.0
    } else if units.starts_with("hours since 1970-01-01") {
        3600.0
    } else {
        return Err(format!("{path}: unsupported time units {units:?}").into());
    };
    let times = raw.iter().map(|t| (t * scale).round() as i64).collect();
    Ok(Source { file, path: path.to_string(), times })
}

pub fn grid(src: &Source, row: &Row) -> Result<GridSpec> {
    match row.grid.kind.as_str() {
        "regular_latlon" => {
            let lat = src.file.read_f64(row.grid.latitude.as_deref().unwrap_or("latitude")).map_err(|e| format!("{}: latitude: {e}", src.path))?;
            let lon = src.file.read_f64(row.grid.longitude.as_deref().unwrap_or("longitude")).map_err(|e| format!("{}: longitude: {e}", src.path))?;
            Ok(GridSpec { kind: "regular_latlon".into(), y: lat, x: lon, crs: "EPSG:4326".into() })
        }
        "projected" => {
            let xn = row.grid.x.as_deref().unwrap_or("x");
            let yn = row.grid.y.as_deref().unwrap_or("y");
            let x = src.file.read_f64(xn).map_err(|e| format!("{}: {xn}: {e}", src.path))?;
            let y = src.file.read_f64(yn).map_err(|e| format!("{}: {yn}: {e}", src.path))?;
            let crs = row.grid.crs.clone().ok_or("a projected grid row names its crs")?;
            Ok(GridSpec { kind: "projected".into(), y, x, crs })
        }
        k => Err(format!("grid kind {k}: no reader for this row's files yet").into()),
    }
}

pub fn record_index(src: &Source, epoch_s: i64) -> Result<usize> {
    src.times
        .iter()
        .position(|t| *t == epoch_s)
        .ok_or_else(|| format!("{}: no record at {}", src.path, crate::time::iso(epoch_s)).into())
}

/// One 2-D field (ny*nx, storage order) at a record; missing values are NaN.
///
/// `NAME[k]` selects layer k of a (y, layer, x) record (the NWM land layout,
/// SOIL_M(time, y, soil_layers_stag, x)). CF packing is undone where the
/// variable declares it: _FillValue / missing_value become NaN, then
/// scale_factor and add_offset apply.
pub fn field(src: &Source, name: &str, record: usize) -> Result<Vec<f64>> {
    let (var, layer) = match name.strip_suffix(']').and_then(|n| n.split_once('[')) {
        Some((v, k)) => (v, Some(k.parse::<usize>().map_err(|_| format!("{name}: layer index"))?)),
        None => (name, None),
    };
    let a = src
        .file
        .read_array_f64_record_or_all(var, record as u64)
        .map_err(|e| format!("{}: {var}: {e}", src.path))?;
    let shape: Vec<usize> = a.shape().iter().copied().skip_while(|d| *d == 1).collect();
    let mut v = a.into_values();
    if let Some(k) = layer {
        let [ny, nl, nx] = shape[..] else {
            return Err(format!("{}: {var}: layer selection needs a (y, layer, x) record, got {shape:?}", src.path).into());
        };
        if k >= nl {
            return Err(format!("{}: {var}: layer {k} of {nl}", src.path).into());
        }
        let mut out = Vec::with_capacity(ny * nx);
        for j in 0..ny {
            out.extend_from_slice(&v[(j * nl + k) * nx..(j * nl + k + 1) * nx]);
        }
        v = out;
    }
    if let Some(meta) = src.file.variable(var) {
        let get = |n: &str| meta.attribute(n).and_then(|a| a.as_f64());
        let fills: Vec<f64> = ["_FillValue", "missing_value"].iter().filter_map(|n| get(n)).collect();
        let (scale, offset) = (get("scale_factor").unwrap_or(1.0), get("add_offset").unwrap_or(0.0));
        if !fills.is_empty() || scale != 1.0 || offset != 0.0 {
            for x in v.iter_mut() {
                *x = if fills.iter().any(|f| *f == *x) { f64::NAN } else { *x * scale + offset };
            }
        }
    }
    if src.file.variable(&format!("missing__{name}")).is_some() {
        let m = src
            .file
            .read_array_f64_record_or_all(&format!("missing__{name}"), record as u64)
            .map_err(|e| format!("{}: missing__{name}: {e}", src.path))?
            .into_values();
        for (x, flag) in v.iter_mut().zip(m) {
            if flag != 0.0 {
                *x = f64::NAN;
            }
        }
    }
    Ok(v)
}
