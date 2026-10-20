//! Earth-relative storm diagnostics on named geometric heights AGL.
//! Horizontal curl is evaluated on the sampled AGL surface, with its terrain
//! slope correction, on the sphere. All weather-array processing stays in Rust.
use netcdf_writer::{AttrValue, NcFormat, NcType, NcWriter, Schema, VarData};
use serde_json::{json, Value};
use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex,
    },
};
const R: f64 = 6_371_000.;
fn read(f: &netcrust::File, n: &str) -> Result<(Vec<f64>, Vec<usize>), String> {
    super::compose_source::values(f, n)
}
fn shaped(f: &netcrust::File, name: &str, expected: &[usize]) -> Result<Vec<f64>, String> {
    let (data, shape) = read(f, name)?;
    if shape != expected {
        return Err(format!("{name}: shape {shape:?}, expected {expected:?}"));
    }
    Ok(data)
}
fn interpolate(z: &[f64], v: &[f64], n: usize, c: usize, h: f64) -> Result<f64, String> {
    if h < z[c] || h > z[(z.len() / n - 1) * n + c] {
        return Err(format!("height {h} outside column {c}"));
    }
    for k in 0..z.len() / n - 1 {
        let a = z[k * n + c];
        let b = z[(k + 1) * n + c];
        if b <= a {
            return Err("nonmonotonic geometric height".into());
        }
        if h >= a && h <= b {
            return Ok(v[k * n + c] + (v[(k + 1) * n + c] - v[k * n + c]) * (h - a) / (b - a));
        }
    }
    Err("height interpolation failed".into())
}
fn diff(a: &[f64], ny: usize, nx: usize, j: usize, i: usize, di: bool) -> f64 {
    let (l, r, d) = if di {
        (
            j * nx + i.saturating_sub(1),
            j * nx + (i + 1).min(nx - 1),
            ((i + 1).min(nx - 1) - i.saturating_sub(1)) as f64,
        )
    } else {
        (
            j.saturating_sub(1) * nx + i,
            (j + 1).min(ny - 1) * nx + i,
            ((j + 1).min(ny - 1) - j.saturating_sub(1)) as f64,
        )
    };
    (a[r] - a[l]) / d
}
fn curl(
    u: &[f64],
    v: &[f64],
    lat: &[f64],
    lon: &[f64],
    ter: &[f64],
    duz: &[f64],
    dvz: &[f64],
    ny: usize,
    nx: usize,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let mut z = vec![0.; ny * nx];
    let mut area = z.clone();
    for j in 0..ny {
        for i in 0..nx {
            let c = j * nx + i;
            let phi = lat[c].to_radians();
            let xi = R * phi.cos() * diff(lon, ny, nx, j, i, true).to_radians();
            let xj = R * phi.cos() * diff(lon, ny, nx, j, i, false).to_radians();
            let yi = R * diff(lat, ny, nx, j, i, true).to_radians();
            let yj = R * diff(lat, ny, nx, j, i, false).to_radians();
            let det = xi * yj - xj * yi;
            if !det.is_finite() || det.abs() < 1. {
                return Err("singular geographic grid or dateline crossing".into());
            }
            let east = |a: &[f64]| {
                (diff(a, ny, nx, j, i, true) * yj - diff(a, ny, nx, j, i, false) * yi) / det
            };
            let north = |a: &[f64]| {
                (diff(a, ny, nx, j, i, false) * xi - diff(a, ny, nx, j, i, true) * xj) / det
            };
            z[c] = east(v) - north(u) + u[c] * phi.tan() / R - dvz[c] * east(ter)
                + duz[c] * north(ter);
            area[c] = det.abs();
        }
    }
    Ok((z, area))
}
fn frame(path: &Path, out: &Path, heights: &[f64], gravity: f64) -> Result<Value, String> {
    let (f, _) = super::open(path)?;
    let (lat, shape) = read(&f, "XLAT")?;
    if shape.len() != 3 || shape[0] != 1 {
        return Err("XLAT must be one Time/y/x frame".into());
    }
    let (ny, nx) = (shape[1], shape[2]);
    if ny < 5 || nx < 5 {
        return Err("diagnostics need at least five cells per horizontal axis for the two-cell statistics rim".into());
    }
    let n = ny * nx;
    let lon = shaped(&f, "XLONG", &[1, ny, nx])?;
    let ter = shaped(&f, "HGT", &[1, ny, nx])?;
    let (ph, sh) = read(&f, "PH")?;
    if sh.len() != 4 || sh[0] != 1 || sh[1] < 3 || sh[2..] != [ny, nx] {
        return Err(
            "PH must contain one complete Time/interface/y/x volume with at least two layers"
                .into(),
        );
    }
    let phb = shaped(&f, "PHB", &sh)?;
    let w = shaped(&f, "W", &sh)?;
    let nz = sh[1] - 1;
    let zi: Vec<f64> = ph
        .iter()
        .zip(phb)
        .enumerate()
        .map(|(k, (a, b))| (a + b) / gravity - ter[k % n])
        .collect();
    let zm: Vec<f64> = (0..nz * n).map(|k| (zi[k] + zi[k + n]) * 0.5).collect();
    let (u, v) = if f.variable("RADAR_U_EARTH").is_some() {
        (
            shaped(&f, "RADAR_U_EARTH", &[1, nz, ny, nx])?,
            shaped(&f, "RADAR_V_EARTH", &[1, nz, ny, nx])?,
        )
    } else {
        let a = shaped(&f, "U", &[1, nz, ny, nx + 1])?;
        let b = shaped(&f, "V", &[1, nz, ny + 1, nx])?;
        let co = shaped(&f, "COSALPHA", &[1, ny, nx])?;
        let si = shaped(&f, "SINALPHA", &[1, ny, nx])?;
        let mut u = vec![0.; nz * n];
        let mut v = u.clone();
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let c = j * nx + i;
                    let x =
                        (a[(k * ny + j) * (nx + 1) + i] + a[(k * ny + j) * (nx + 1) + i + 1]) * 0.5;
                    let y =
                        (b[(k * (ny + 1) + j) * nx + i] + b[(k * (ny + 1) + j + 1) * nx + i]) * 0.5;
                    u[k * n + c] = x * co[c] - y * si[c];
                    v[k * n + c] = y * co[c] + x * si[c];
                }
            }
        }
        (u, v)
    };
    if u.len() != nz * n || v.len() != nz * n {
        return Err("inconsistent mass wind shape".into());
    }
    let mut planes: Vec<(String, String, Vec<f64>)> = vec![
        ("XLAT".into(), "degree_north".into(), lat.clone()),
        ("XLONG".into(), "degree_east".into(), lon.clone()),
        ("HGT".into(), "m".into(), ter.clone()),
    ];
    let mut stats = vec![];
    for &h in heights {
        let mut ww = vec![0.; n];
        let mut uu = ww.clone();
        let mut vv = ww.clone();
        let mut duz = ww.clone();
        let mut dvz = ww.clone();
        for c in 0..n {
            ww[c] = interpolate(&zi, &w, n, c, h)?;
            uu[c] = interpolate(&zm, &u, n, c, h)?;
            vv[c] = interpolate(&zm, &v, n, c, h)?;
            let lo = (h - 10.).max(zm[c]);
            let hi = (h + 10.).min(zm[(nz - 1) * n + c]);
            duz[c] =
                (interpolate(&zm, &u, n, c, hi)? - interpolate(&zm, &u, n, c, lo)?) / (hi - lo);
            dvz[c] =
                (interpolate(&zm, &v, n, c, hi)? - interpolate(&zm, &v, n, c, lo)?) / (hi - lo);
        }
        let (zeta, area) = curl(&uu, &vv, &lat, &lon, &ter, &duz, &dvz, ny, nx)?;
        let cells: Vec<usize> = (2..ny - 2)
            .flat_map(|j| (2..nx - 2).map(move |i| j * nx + i))
            .collect();
        stats.push(json!({"height_agl_m":h,"max_w_m_s":cells.iter().map(|&c|ww[c]).fold(f64::NEG_INFINITY,f64::max),"max_abs_vertical_vorticity_s_1":cells.iter().map(|&c|zeta[c].abs()).fold(0.,f64::max),"updraft_area_w_ge_5_m_s_km2":cells.iter().filter(|&&c|ww[c]>=5.).map(|&c|area[c]/1e6).sum::<f64>()}));
        planes.push((format!("STORM_W_{h:.0}M"), "m s-1".into(), ww));
        planes.push((format!("STORM_ZETA_{h:.0}M"), "s-1".into(), zeta));
    }
    if f.variable("REFL_10CM").is_some() {
        let (refl, shape) = read(&f, "REFL_10CM")?;
        if shape != [1, ny, nx] && shape != [1, nz, ny, nx] {
            return Err("REFL_10CM must be one complete surface or mass-level volume".into());
        }
        let levels = if shape.len() == 4 { shape[1] } else { 1 };
        let max = (0..n)
            .map(|c| {
                (0..levels)
                    .map(|k| refl[k * n + c])
                    .fold(f64::NEG_INFINITY, f64::max)
            })
            .collect();
        planes.push(("STORM_REFL_MAX".into(), "dBZ".into(), max));
    }
    let mut schema = Schema::new(NcFormat::Classic);
    let t = schema
        .def_dim("Time", 1, false)
        .map_err(|e| e.to_string())?;
    let y = schema.def_dim("south_north", ny, false).unwrap();
    let x = schema.def_dim("west_east", nx, false).unwrap();
    schema.def_dim("bottom_top", nz, false).unwrap();
    let string = schema.def_dim("DateStrLen", 19, false).unwrap();
    for name in [
        "GRID_ID",
        "MAP_PROJ",
        "DX",
        "DY",
        "CEN_LAT",
        "CEN_LON",
        "TRUELAT1",
        "TRUELAT2",
        "STAND_LON",
        "POLE_LAT",
        "POLE_LON",
    ] {
        if let Some(a) = f.attribute(name) {
            if let Some(v) = a.as_f64() {
                schema
                    .put_global_attr(
                        name,
                        if ["MAP_PROJ", "GRID_ID"].contains(&name) {
                            AttrValue::Ints(vec![v as i32])
                        } else {
                            AttrValue::Floats(vec![v as f32])
                        },
                    )
                    .unwrap();
            }
        }
    }
    for name in ["START_DATE", "SIMULATION_START_DATE", "TITLE"] {
        if let Some(v) = f
            .attribute(name)
            .and_then(|a| a.as_string().map(str::to_owned))
        {
            schema
                .put_global_attr(name, AttrValue::Text(v.into()))
                .unwrap();
        }
    }
    schema.put_global_attr("STORM_DIAGNOSTIC_METHOD",AttrValue::Text("Earth-relative spherical vertical curl at constant geometric height; AGL interpolation with terrain-slope correction; stats exclude two-cell rim".into())).unwrap();
    let time_id = schema.def_var("Times", NcType::Char, &[t, string]).unwrap();
    let mut vars = vec![];
    for (name, units, _) in &planes {
        let id = schema.def_var(name, NcType::Float, &[t, y, x]).unwrap();
        schema
            .put_var_attr(id, "units", AttrValue::Text(units.clone()))
            .unwrap();
        schema
            .put_var_attr(id, "description", AttrValue::Text(name.replace('_', " ")))
            .unwrap();
        vars.push(id);
    }
    let times = f.read_strings("Times").map_err(|e| e.to_string())?;
    let filename = path.file_name().unwrap();
    let target = out.join(filename);
    let mut writer = NcWriter::create(&target, schema).map_err(|e| e.to_string())?;
    let chars = times.join("").into_bytes();
    if chars.len() != 19 {
        return Err("expected one 19-character WRF valid time".into());
    }
    writer
        .write_var(time_id, VarData::Char(&chars))
        .map_err(|e| e.to_string())?;
    for (id, (_, _, data)) in vars.into_iter().zip(planes) {
        let data: Vec<f32> = data.iter().map(|&v| v as f32).collect();
        writer
            .write_var(id, VarData::F32(&data))
            .map_err(|e| e.to_string())?;
    }
    writer.finish().map_err(|e| e.to_string())?;
    Ok(
        json!({"source":path,"output":target,"valid_time":String::from_utf8_lossy(&chars),"gravity_m_s2":gravity,"statistics":stats}),
    )
}
pub(super) fn run(
    out: &Path,
    heights: &[f64],
    gravity: f64,
    jobs: usize,
    files: &[PathBuf],
) -> Result<(), String> {
    if out.exists() {
        return Err("diagnostic output already exists".into());
    }
    if heights.is_empty()
        || heights.iter().any(|h| !h.is_finite() || *h <= 10.)
        || !gravity.is_finite()
        || gravity <= 0.
        || files.is_empty()
    {
        return Err("invalid heights, gravity or files".into());
    }
    let mut labels = BTreeSet::new();
    if heights.iter().any(|h| !labels.insert(format!("{h:.0}"))) {
        return Err("heights must have distinct whole-metre product names".into());
    }
    let mut names = BTreeSet::new();
    if files.iter().any(|p| !names.insert(p.file_name())) {
        return Err("input basenames must be distinct; diagnose separate runs in separate output directories".into());
    }
    fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let next = AtomicUsize::new(0);
    let results = Mutex::new(vec![]);
    let jobs = jobs.max(1).min(files.len());
    std::thread::scope(|s| {
        for _ in 0..jobs {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= files.len() {
                    break;
                }
                let result = frame(&files[i], out, heights, gravity);
                results.lock().unwrap().push((i, result));
            });
        }
    });
    let mut results = results.into_inner().unwrap();
    results.sort_by_key(|r| r.0);
    let receipts: Vec<_> = results
        .into_iter()
        .map(|(_, r)| r)
        .collect::<Result<_, _>>()?;
    fs::write(out.join("series.json"),serde_json::to_vec_pretty(&json!({"schema":"gpuwm-storm-diagnostics-v1","frame_threads":jobs,"vorticity":"Earth-relative spherical vertical curl at constant geometric height; interpolated AGL surface corrected for terrain slopes","statistics_rim_cells":2,"frames":receipts})).unwrap()).map_err(|e|e.to_string())?;
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn diagnostic_names_cannot_overwrite_a_frame_or_height() {
        let out = std::env::temp_dir().join(format!("storm-name-refusal-{}", std::process::id()));
        let paths = vec![
            PathBuf::from("run-a/wrfout_d01_same"),
            PathBuf::from("run-b/wrfout_d01_same"),
        ];
        assert!(run(&out, &[500.], 9.81, 2, &paths)
            .unwrap_err()
            .contains("basenames"));
        assert!(run(&out, &[500., 500.4], 9.81, 2, &paths[..1])
            .unwrap_err()
            .contains("product names"));
        assert!(!out.exists());
    }
    #[test]
    fn earth_curl_solid_rotation_and_vertical_interpolation() {
        let (ny, nx) = (7, 7);
        let lat: Vec<_> = (0..ny)
            .flat_map(|j| (0..nx).map(move |_| j as f64 * 0.001))
            .collect();
        let lon: Vec<_> = (0..ny)
            .flat_map(|_| (0..nx).map(|i| i as f64 * 0.001))
            .collect();
        let u: Vec<_> = lat.iter().map(|l| -l.to_radians() * R * 0.001).collect();
        let v: Vec<_> = lon.iter().map(|l| l.to_radians() * R * 0.001).collect();
        let zeros = vec![0.; nx * ny];
        let (z, _) = curl(&u, &v, &lat, &lon, &zeros, &zeros, &zeros, ny, nx).unwrap();
        assert!((z[3 * nx + 3] - 0.002).abs() < 1e-8);
        assert_eq!(
            interpolate(&[0., 100., 200.], &[0., 10., -10.], 1, 0, 150.).unwrap(),
            0.
        );
        assert!(interpolate(&[0., 100.], &[0., 1.], 1, 0, 101.).is_err());
    }
}
