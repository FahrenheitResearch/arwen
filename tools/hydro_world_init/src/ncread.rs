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
pub fn field(src: &Source, name: &str, record: usize) -> Result<Vec<f64>> {
    let a = src
        .file
        .read_array_f64_record_or_all(name, record as u64)
        .map_err(|e| format!("{}: {name}: {e}", src.path))?;
    let mut v = a.into_values();
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
