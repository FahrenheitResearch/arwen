//! Native bathymetry ingest, model-independent forcing contract and surge products.
//! NetCDF parsing/writing is seeded from this tree's netcrust/netcdf-writer.
use netcdf_writer::{AttrValue, NcFormat, NcType, NcWriter, Schema, VarData};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    error::Error,
    fs,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;
#[derive(Clone, Deserialize, Serialize)]
struct Grid {
    nx: usize,
    ny: usize,
    west: f64,
    east: f64,
    south: f64,
    north: f64,
}
impl Grid {
    fn validate(&self) -> Result<()> {
        if self.nx < 3
            || self.ny < 3
            || self.nx.checked_mul(self.ny).is_none_or(|n| n > 100_000_000)
        {
            return Err("grid dimensions must each be >=3 and total cells <=100 million".into());
        }
        if ![self.west, self.east, self.south, self.north]
            .iter()
            .all(|x| x.is_finite())
            || self.east <= self.west
            || self.east - self.west >= 180.
            || self.north <= self.south
            || self.south < -85.
            || self.north > 85.
        {
            return Err("invalid regular geographic grid bounds".into());
        }
        Ok(())
    }
    fn dx(&self) -> f64 {
        (self.east - self.west) / (self.nx - 1) as f64
    }
    fn dy(&self) -> f64 {
        (self.north - self.south) / (self.ny - 1) as f64
    }
    fn lon(&self, i: usize) -> f64 {
        self.west + (i % self.nx) as f64 * self.dx()
    }
    fn lat(&self, i: usize) -> f64 {
        self.south + (i / self.nx) as f64 * self.dy()
    }
}
#[derive(Deserialize)]
struct Bathy {
    path: PathBuf,
    elevation: String,
    latitude: String,
    longitude: String,
    vertical_datum: String,
    source: String,
    license: String,
}
#[derive(Deserialize)]
struct Forcing {
    path: PathBuf,
    latitude: String,
    longitude: String,
    time: String,
    u10: String,
    v10: String,
    pressure: String,
    source: String,
    license: String,
}
#[derive(Clone, Deserialize, Serialize)]
struct Gauge {
    station: String,
    latitude: f64,
    longitude: f64,
    observations: PathBuf,
    predictions: PathBuf,
}
#[derive(Deserialize)]
struct Config {
    grid: Grid,
    bathymetry: Bathy,
    forcing: Forcing,
    duration_seconds: f64,
    #[serde(default = "default_output")]
    output_interval_seconds: f64,
    #[serde(default = "default_max_dt")]
    max_dt_seconds: f64,
    #[serde(default = "default_dry")]
    dry_depth_m: f32,
    #[serde(default = "default_manning")]
    manning_n: f32,
    #[serde(default = "default_cfl")]
    cfl: f32,
    #[serde(default = "default_boundary")]
    open_boundary_mask: i32,
    #[serde(default = "default_pressure")]
    reference_pressure_pa: f32,
    #[serde(default)]
    initial_water_level_m: f32,
    #[serde(default)]
    tide_levels_m: Vec<f64>,
    #[serde(default)]
    gauges: Vec<Gauge>,
}
fn default_output() -> f64 {
    21600.
}
fn default_max_dt() -> f64 {
    60.
}
fn default_dry() -> f32 {
    0.001
}
fn default_manning() -> f32 {
    0.025
}
fn default_cfl() -> f32 {
    0.4
}
fn default_boundary() -> i32 {
    15
}
fn default_pressure() -> f32 {
    101325.
}
fn data(file: &netcrust::File, name: &str) -> Result<Vec<f64>> {
    Ok(file.read_array_f64(name)?.into_values())
}
fn unit(file: &netcrust::File, name: &str, allowed: &[&str]) -> Result<()> {
    let var = file
        .variable(name)
        .ok_or_else(|| format!("missing variable {name}"))?;
    let u = var
        .attribute("units")
        .and_then(|a| a.as_string().map(str::to_string))
        .unwrap_or_default();
    if !allowed.contains(&u.as_str()) {
        return Err(format!("{name}: units {u:?} do not match {allowed:?}").into());
    }
    Ok(())
}
fn axis(a: &[f64]) -> Result<()> {
    if a.len() < 2
        || a.iter().any(|x| !x.is_finite())
        || a.windows(2).any(|w| (w[1] - w[0]) * (a[1] - a[0]) <= 0.)
    {
        return Err("coordinate axis must be finite and strictly monotone".into());
    }
    Ok(())
}
fn bracket(a: &[f64], x: f64) -> Result<(usize, f64)> {
    let ascending = a[1] > a[0];
    let lo = a[0].min(a[a.len() - 1]);
    let hi = a[0].max(a[a.len() - 1]);
    if x < lo - 1e-7 || x > hi + 1e-7 {
        return Err(format!("grid coordinate {x} is outside source [{lo},{hi}]").into());
    }
    let i = a
        .partition_point(|v| if ascending { *v <= x } else { *v >= x })
        .saturating_sub(1)
        .min(a.len() - 2);
    Ok((i, ((x - a[i]) / (a[i + 1] - a[i])).clamp(0., 1.)))
}
fn bilinear(a: &[f64], nx: usize, i: usize, j: usize, x: f64, y: f64) -> f32 {
    ((1. - y) * ((1. - x) * a[j * nx + i] + x * a[j * nx + i + 1])
        + y * ((1. - x) * a[(j + 1) * nx + i] + x * a[(j + 1) * nx + i + 1])) as f32
}
fn write_f32(path: impl AsRef<Path>, values: &[f32]) -> Result<()> {
    let mut w = BufWriter::new(fs::File::create(path)?);
    for &x in values {
        w.write_all(&x.to_le_bytes())?;
    }
    w.flush()?;
    Ok(())
}
fn write_f64(path: impl AsRef<Path>, values: &[f64]) -> Result<()> {
    let mut w = BufWriter::new(fs::File::create(path)?);
    for &x in values {
        w.write_all(&x.to_le_bytes())?;
    }
    w.flush()?;
    Ok(())
}
fn read_f32(path: impl AsRef<Path>, n: usize) -> Result<Vec<f32>> {
    let bytes = fs::read(path)?;
    if bytes.len() != 4 * n {
        return Err(format!("transfer buffer length {}, expected {}", bytes.len(), 4 * n).into());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}
fn hash(path: &Path) -> Result<String> {
    let bytes = fs::read(path)?;
    Ok(format!("{:x}", Sha256::digest(&bytes)))
}
fn save(path: impl AsRef<Path>, value: &Value) -> Result<()> {
    fs::write(path, serde_json::to_vec_pretty(value)?)?;
    Ok(())
}
fn prepare(config: &Path, out: &Path) -> Result<()> {
    let start = std::time::Instant::now();
    let mut cfg: Config = serde_json::from_slice(&fs::read(config)?)?;
    cfg.grid.validate()?;
    let base = config.parent().unwrap_or(Path::new("."));
    for path in [&mut cfg.bathymetry.path, &mut cfg.forcing.path] {
        if path.is_relative() {
            *path = base.join(&*path);
        }
    }
    for gauge in &mut cfg.gauges {
        for path in [&mut gauge.observations, &mut gauge.predictions] {
            if path.is_relative() {
                *path = base.join(&*path);
            }
        }
    }
    if !cfg.duration_seconds.is_finite()
        || cfg.duration_seconds <= 0.
        || !cfg.output_interval_seconds.is_finite()
        || cfg.output_interval_seconds <= 0.
        || !cfg.max_dt_seconds.is_finite()
        || cfg.max_dt_seconds <= 0.
        || !cfg.dry_depth_m.is_finite()
        || cfg.dry_depth_m <= 0.
        || !cfg.manning_n.is_finite()
        || cfg.manning_n < 0.
        || !cfg.cfl.is_finite()
        || !(0.01..=0.45).contains(&cfg.cfl)
        || !cfg.initial_water_level_m.is_finite()
        || !cfg.reference_pressure_pa.is_finite()
        || cfg.open_boundary_mask < 0
        || cfg.open_boundary_mask > 15
    {
        return Err("invalid integration controls: positive times/depth, Manning >=0, 0.01<=CFL<=0.45, mask in 0..15 required".into());
    }
    if cfg.bathymetry.source.is_empty()
        || cfg.bathymetry.license.is_empty()
        || cfg.bathymetry.vertical_datum.is_empty()
        || cfg.forcing.source.is_empty()
        || cfg.forcing.license.is_empty()
    {
        return Err(
            "bathymetry and forcing require source, license and vertical datum provenance".into(),
        );
    }
    fs::create_dir_all(out)?;
    let bf = netcrust::open(&cfg.bathymetry.path)?;
    unit(&bf, &cfg.bathymetry.elevation, &["m", "meters", "metres"])?;
    let bx = data(&bf, &cfg.bathymetry.longitude)?;
    let by = data(&bf, &cfg.bathymetry.latitude)?;
    axis(&bx)?;
    axis(&by)?;
    let ba = bf.read_array_f64(&cfg.bathymetry.elevation)?;
    if ba.shape() != [by.len(), bx.len()]
        || ba
            .values()
            .iter()
            .any(|x| !x.is_finite() || x.abs() > 12000.)
    {
        return Err(
            "elevation must be complete finite [latitude,longitude] in metres, positive upward"
                .into(),
        );
    }
    let bed: Vec<f32> = (0..cfg.grid.nx * cfg.grid.ny)
        .into_par_iter()
        .map(|i| -> Result<f32> {
            let (x, wx) = bracket(&bx, cfg.grid.lon(i))?;
            let (y, wy) = bracket(&by, cfg.grid.lat(i))?;
            Ok(bilinear(ba.values(), bx.len(), x, y, wx, wy))
        })
        .collect::<Result<_>>()?;
    write_f32(out.join("bed.f32"), &bed)?;
    let ff = netcrust::open(&cfg.forcing.path)?;
    for n in [&cfg.forcing.u10, &cfg.forcing.v10] {
        unit(&ff, n, &["m s-1", "m/s", "m s**-1"])?;
    }
    unit(&ff, &cfg.forcing.pressure, &["Pa", "pascal", "pascals"])?;
    let mut fx = data(&ff, &cfg.forcing.longitude)?;
    let fy = data(&ff, &cfg.forcing.latitude)?;
    axis(&fy)?;
    for x in &mut fx {
        while *x - cfg.grid.west > 180. {
            *x -= 360.;
        }
        while *x - cfg.grid.west < -180. {
            *x += 360.;
        }
    }
    axis(&fx)?;
    for v in [cfg.grid.west, cfg.grid.east] {
        bracket(&fx, v)?;
    }
    for v in [cfg.grid.south, cfg.grid.north] {
        bracket(&fy, v)?;
    }
    let ft = data(&ff, &cfg.forcing.time)?;
    axis(&ft)?;
    let tu = ff
        .variable(&cfg.forcing.time)
        .and_then(|v| {
            v.attribute("units")
                .and_then(|a| a.as_string().map(str::to_string))
        })
        .unwrap_or_default();
    if !tu.starts_with("seconds since 1970-01-01") {
        return Err(format!("forcing time must be epoch seconds, received {tu:?}").into());
    }
    if ft[ft.len() - 1] - ft[0] + 1e-6 < cfg.duration_seconds || ft[1] < ft[0] {
        return Err("forcing does not cover the full run, including its last boundary".into());
    }
    let relative: Vec<f64> = ft.iter().map(|v| v - ft[0]).collect();
    write_f64(out.join("times.f64"), &relative)?;
    let mut winds = Vec::new();
    for name in [&cfg.forcing.u10, &cfg.forcing.v10, &cfg.forcing.pressure] {
        let a = ff.read_array_f64(name)?;
        if a.shape() != [ft.len(), fy.len(), fx.len()] {
            return Err(format!("forcing {name} must have [time,latitude,longitude] shape").into());
        }
        if a.values().iter().any(|x| {
            !x.is_finite()
                || if name == &cfg.forcing.pressure {
                    *x < 70000. || *x > 110000.
                } else {
                    x.abs() > 150.
                }
        }) {
            return Err(format!("forcing {name} contains missing or invalid SI values").into());
        }
        winds.push(a.into_values());
    }
    let plane = fx.len() * fy.len();
    let mut force = Vec::with_capacity(3 * ft.len() * plane);
    // Normalize axes once. Regridding to solver cells and time interpolation stay on CUDA.
    for t in 0..ft.len() {
        for values in &winds {
            for j in 0..fy.len() {
                for i in 0..fx.len() {
                    let jj = if fy[1] > fy[0] { j } else { fy.len() - 1 - j };
                    let ii = if fx[1] > fx[0] { i } else { fx.len() - 1 - i };
                    force.push(values[t * plane + jj * fx.len() + ii] as f32);
                }
            }
        }
    }
    if fx
        .windows(2)
        .any(|w| ((w[1] - w[0]).abs() - (fx[1] - fx[0]).abs()).abs() > 1e-5)
        || fy
            .windows(2)
            .any(|w| ((w[1] - w[0]).abs() - (fy[1] - fy[0]).abs()).abs() > 1e-5)
    {
        return Err("CUDA forcing grid requires regular coordinate spacing".into());
    }
    write_f32(out.join("forcing.f32"), &force)?;
    let tides = if cfg.tide_levels_m.is_empty() {
        vec![0.; ft.len()]
    } else {
        cfg.tide_levels_m.clone()
    };
    if tides.len() != ft.len() || tides.iter().any(|x| !x.is_finite()) {
        return Err("tide_levels_m must match forcing times with finite metres".into());
    }
    write_f64(out.join("tides.f64"), &tides)?;
    let selected:Vec<Value>=cfg.gauges.par_iter().map(|g|->Result<Value>{
        if !g.latitude.is_finite() || !g.longitude.is_finite() || g.station.is_empty() || !g.station.bytes().all(|c|c.is_ascii_alphanumeric()||c==b'-'||c==b'_'){return Err("gauges need finite coordinates and a plain station identifier".into());}
        let mut best=None;
        for j in 0..cfg.grid.ny{let lat=cfg.grid.south+j as f64*cfg.grid.dy();if (lat-g.latitude).abs()>0.05{continue;}
        for i in 0..cfg.grid.nx{let k=j*cfg.grid.nx+i;if bed[k]>=cfg.initial_water_level_m-cfg.dry_depth_m{continue;}
        let dx=(cfg.grid.lon(k)-g.longitude)*g.latitude.to_radians().cos()*111195.;let dy=(lat-g.latitude)*111195.;let dist=dx.hypot(dy);
        if dist<=5000. && best.is_none_or(|(_,d)|dist<d){best=Some((k,dist));}}}
        let(k,d)=best.ok_or_else(||format!("gauge {} has no initially wet cell within 5 km",g.station))?;
        Ok(json!({"gauge":g,"cell":k,"distance_m":d,"cell_longitude":cfg.grid.lon(k),"cell_latitude":cfg.grid.lat(k),"bed_m":bed[k]}))
    }).collect::<Result<_>>()?;
    let mut cells = BufWriter::new(fs::File::create(out.join("gauges.i32"))?);
    for g in &selected {
        cells.write_all(&(g["cell"].as_u64().unwrap() as i32).to_le_bytes())?;
    }
    // An empty transfer buffer must still be mappable by the Python CUDA driver.
    if selected.is_empty() {
        cells.write_all(&0_i32.to_le_bytes())?;
    }
    cells.flush()?;
    let p = json!({"nx":cfg.grid.nx,"ny":cfg.grid.ny,"fx":fx.len(),"fy":fy.len(),"nt":ft.len(),"ng":selected.len(),
      "boundary_mask":cfg.open_boundary_mask,"west":cfg.grid.west,"south":cfg.grid.south,"dlon":cfg.grid.dx(),"dlat":cfg.grid.dy(),
      "fwest":fx[0].min(*fx.last().unwrap()),"fsouth":fy[0].min(*fy.last().unwrap()),"fdlon":(fx[1]-fx[0]).abs(),"fdlat":(fy[1]-fy[0]).abs(),
      "duration":cfg.duration_seconds,"max_dt":cfg.max_dt_seconds,"output_interval":cfg.output_interval_seconds,"dry":cfg.dry_depth_m,
      "manning":cfg.manning_n,"cfl":cfg.cfl,"reference_pressure":cfg.reference_pressure_pa,"initial_level":cfg.initial_water_level_m});
    let result = json!({"schema":"gpuwm.surge-input.v1","parameters":p,"grid":cfg.grid,"epoch_start":ft[0],
      "bed":"bed.f32","forcing":"forcing.f32","times":"times.f64","tides":"tides.f64","gauge_cells":"gauges.i32","gauges":selected,
      "bathymetry":{"source":cfg.bathymetry.source,"license":cfg.bathymetry.license,"vertical_datum":cfg.bathymetry.vertical_datum,"sha256":hash(&cfg.bathymetry.path)?},
      "atmosphere":{"source":cfg.forcing.source,"license":cfg.forcing.license,"sha256":hash(&cfg.forcing.path)?},
      "tides_enabled":!cfg.tide_levels_m.is_empty(),"wall_seconds":start.elapsed().as_secs_f64(),"workers":rayon::current_num_threads()});
    save(out.join("prepared.json"), &result)?;
    println!("{result}");
    Ok(())
}
fn gauge_series(path: &Path, key: &str) -> Result<BTreeMap<i64, f64>> {
    let v: Value = serde_json::from_slice(&fs::read(path)?)?;
    let rows = v[key].as_array().ok_or_else(|| {
        format!(
            "{}: missing NOAA {key}, API error={}",
            path.display(),
            v["error"]
        )
    })?;
    let mut out = BTreeMap::new();
    for row in rows {
        if key == "data" && row["q"].as_str().is_some_and(|q| q != "v") {
            continue;
        }
        let Some(t) = row["t"].as_str() else { continue };
        let Some(value) = row["v"]
            .as_str()
            .and_then(|x| x.parse::<f64>().ok())
            .filter(|x| x.is_finite())
        else {
            continue;
        };
        let time = chrono::NaiveDateTime::parse_from_str(t, "%Y-%m-%d %H:%M")?
            .and_utc()
            .timestamp();
        out.insert(time, value);
    }
    Ok(out)
}
fn finish(prepared: &Path, raw: &Path, out: &Path) -> Result<()> {
    let meta: Value = serde_json::from_slice(&fs::read(prepared)?)?;
    let grid: Grid = serde_json::from_value(meta["grid"].clone())?;
    let epoch = meta["epoch_start"].as_f64().unwrap();
    let n = grid.nx * grid.ny;
    let frames: Vec<f64> = serde_json::from_slice(&fs::read(raw.join("frames.json"))?)?;
    let bed = read_f32(prepared.parent().unwrap().join("bed.f32"), n)?;
    let mut schema = Schema::new(NcFormat::Offset64);
    let td = schema.def_dim("Time", 0, true)?;
    let yd = schema.def_dim("south_north", grid.ny, false)?;
    let xd = schema.def_dim("west_east", grid.nx, false)?;
    let sd = schema.def_dim("DateStrLen", 19, false)?;
    schema.def_dim("bottom_top", 1, false)?;
    schema.def_dim("bottom_top_stag", 2, false)?;
    schema.put_global_attr(
        "TITLE",
        AttrValue::Text("Depth averaged nonlinear shallow water storm surge".into()),
    )?;
    schema.put_global_attr("GPUWM_MODEL_LABEL", AttrValue::Text("SWE surge".into()))?;
    schema.put_global_attr(
        "START_DATE",
        AttrValue::Text(
            chrono::DateTime::from_timestamp(epoch as i64, 0)
                .ok_or("invalid epoch")?
                .format("%Y-%m-%d_%H:%M:%S")
                .to_string(),
        ),
    )?;
    schema.put_global_attr("GRID_ID", AttrValue::Ints(vec![1]))?;
    schema.put_global_attr("MAP_PROJ", AttrValue::Ints(vec![6]))?;
    schema.put_global_attr(
        "DX",
        AttrValue::Floats(vec![
            (grid.dx().to_radians()
                * 6371000.
                * ((grid.south + grid.north) / 2.).to_radians().cos()) as f32,
        ]),
    )?;
    schema.put_global_attr(
        "DY",
        AttrValue::Floats(vec![(grid.dy().to_radians() * 6371000.) as f32]),
    )?;
    schema.put_global_attr(
        "surge_vertical_datum",
        AttrValue::Text(
            meta["bathymetry"]["vertical_datum"]
                .as_str()
                .unwrap()
                .into(),
        ),
    )?;
    schema.put_global_attr("surge_definition",AttrValue::Text("water level minus imposed astronomical boundary tide minus initial level; no wave setup".into()))?;
    let time_id = schema.def_var("Times", NcType::Char, &[td, sd])?;
    // The renderer's surface-only native reader derives its grid from a
    // time-indexed XLAT plane; dimensions alone do not establish that ABI.
    let lat_id = schema.def_var("XLAT", NcType::Float, &[td, yd, xd])?;
    let lon_id = schema.def_var("XLONG", NcType::Float, &[td, yd, xd])?;
    let bed_id = schema.def_var("HGT", NcType::Float, &[yd, xd])?;
    schema.put_var_attr(bed_id, "units", AttrValue::Text("m".into()))?;
    schema.put_var_attr(
        bed_id,
        "description",
        AttrValue::Text("Bathymetry and terrain elevation positive upward".into()),
    )?;
    schema.put_var_attr(lat_id, "units", AttrValue::Text("degrees_north".into()))?;
    schema.put_var_attr(lon_id, "units", AttrValue::Text("degrees_east".into()))?;
    let definitions = [
        (
            "water_level",
            true,
            "Water level above the declared bathymetry vertical datum",
        ),
        (
            "inundation_depth",
            true,
            "Water depth on initially dry terrain",
        ),
        ("max_surge", true, "Maximum non-tidal water level anomaly"),
        (
            "max_inundation_depth",
            true,
            "Maximum water depth on initially dry terrain",
        ),
    ];
    let mut ids = Vec::new();
    for (name, record, description) in definitions {
        let dims = if record {
            vec![td, yd, xd]
        } else {
            vec![yd, xd]
        };
        let id = schema.def_var(name, NcType::Float, &dims)?;
        schema.put_var_attr(id, "units", AttrValue::Text("m".into()))?;
        schema.put_var_attr(id, "description", AttrValue::Text(description.into()))?;
        ids.push(id);
    }
    let mut writer = NcWriter::create(out.join("surge.nc"), schema)?;
    let lats: Vec<f32> = (0..n).into_par_iter().map(|i| grid.lat(i) as f32).collect();
    let lons: Vec<f32> = (0..n).into_par_iter().map(|i| grid.lon(i) as f32).collect();
    writer.write_var(bed_id, VarData::F32(&bed))?;
    let max = read_f32(raw.join("max-surge.f32"), n)?;
    let max_inund = read_f32(raw.join("max-inundation.f32"), n)?;
    let missing = vec![f32::NAN; n];
    for (rec, &time) in frames.iter().enumerate() {
        writer.write_record(rec as u64, lat_id, VarData::F32(&lats))?;
        writer.write_record(rec as u64, lon_id, VarData::F32(&lons))?;
        let stamp = chrono::DateTime::from_timestamp((epoch + time).round() as i64, 0)
            .ok_or("invalid frame time")?
            .format("%Y-%m-%d_%H:%M:%S")
            .to_string();
        writer.write_record(rec as u64, time_id, VarData::Char(stamp.as_bytes()))?;
        let eta = read_f32(raw.join(format!("water-level-{rec:04}.f32")), n)?;
        let inund: Vec<f32> = eta
            .par_iter()
            .zip(&bed)
            .map(|(&e, &b)| {
                if b >= meta["parameters"]["initial_level"].as_f64().unwrap() as f32 {
                    (e - b).max(0.)
                } else {
                    f32::NAN
                }
            })
            .collect();
        writer.write_record(rec as u64, ids[0], VarData::F32(&eta))?;
        writer.write_record(rec as u64, ids[1], VarData::F32(&inund))?;
        writer.write_record(
            rec as u64,
            ids[2],
            VarData::F32(if rec + 1 == frames.len() {
                &max
            } else {
                &missing
            }),
        )?;
        writer.write_record(
            rec as u64,
            ids[3],
            VarData::F32(if rec + 1 == frames.len() {
                &max_inund
            } else {
                &missing
            }),
        )?;
    }
    writer.finish()?;
    let gauges = meta["gauges"].as_array().unwrap();
    let lines = fs::read_to_string(raw.join("gauges.csv"))?;
    let models: Vec<(f64, Vec<f64>)> = lines
        .lines()
        .skip(1)
        .map(|l| -> Result<_> {
            let mut p = l.split(',');
            let time = p.next().ok_or("gauge row")?.parse()?;
            let v = p
                .map(str::parse)
                .collect::<std::result::Result<Vec<f64>, _>>()?;
            Ok((time, v))
        })
        .collect::<Result<_>>()?;
    let verification:Vec<Value>=gauges.par_iter().enumerate().map(|(i,entry)|->Result<Value>{
        let g:Gauge=serde_json::from_value(entry["gauge"].clone())?;let obs=gauge_series(&g.observations,"data")?;let pred=gauge_series(&g.predictions,"predictions")?;
        let mut matches=Vec::new();for (time,values) in &models {
            let stamp=(epoch+time).round() as i64; if let (Some(o),Some(t))=(obs.get(&stamp),pred.get(&stamp)) {if !values[i].is_finite(){continue;}
                matches.push((stamp,values[i],o-t,*o,*t));}}
        if matches.is_empty(){return Err(format!("gauge {} has no verified observation/prediction/model common times",g.station).into());}
        let count=matches.len() as f64;let bias=matches.iter().map(|(_,m,o,_,_)|m-o).sum::<f64>()/count;
        let rmse=(matches.iter().map(|(_,m,o,_,_)|(m-o).powi(2)).sum::<f64>()/count).sqrt();
        let model_peak=matches.iter().max_by(|a,b|a.1.total_cmp(&b.1)).unwrap();let obs_peak=matches.iter().max_by(|a,b|a.2.total_cmp(&b.2)).unwrap();
        let mut csv=BufWriter::new(fs::File::create(out.join(format!("gauge-{}.csv",g.station)))?);
        writeln!(csv,"epoch_seconds,model_surge_m,observed_residual_m,observed_water_level_m,predicted_tide_m")?;
        for (t,m,o,w,p)in &matches{writeln!(csv,"{t},{m},{o},{w},{p}")?;}
        Ok(json!({"station":g.station,"matched":matches.len(),"bias_m":bias,"rmse_m":rmse,"model_peak_m":model_peak.1,
            "observed_peak_m":obs_peak.2,"peak_time_error_seconds":model_peak.0-obs_peak.0,"selected_cell":entry,
            "comparison":"model non-tidal anomaly versus NOAA verified water level minus NOAA tide prediction; common times only",
            "observations_sha256":hash(&g.observations)?,"predictions_sha256":hash(&g.predictions)?}))
    }).collect::<Result<_>>()?;
    save(
        out.join("verification.json"),
        &json!({"schema":"gpuwm.surge-gauge-verification.v1","gauges":verification}),
    )?;
    fs::copy(
        raw.join("native-receipt.json"),
        out.join("native-receipt.json"),
    )?;
    println!(
        "{}",
        json!({"netcdf":out.join("surge.nc"),"frames":frames.len(),"gauges":gauges.len(),"workers":rayon::current_num_threads()})
    );
    Ok(())
}
fn fixture(kind: &str, out: &Path) -> Result<()> {
    if !["lake", "wind", "pressure", "tide", "deep-pressure"].contains(&kind) {
        return Err("unknown physical fixture".into());
    }
    fs::create_dir_all(out)?;
    let nx = 384;
    let ny = 256;
    let n = nx * ny;
    let nt = 13;
    let duration = 43200.;
    let mut schema = Schema::new(NcFormat::Offset64);
    let td = schema.def_dim("time", nt, false)?;
    let yd = schema.def_dim("latitude", ny, false)?;
    let xd = schema.def_dim("longitude", nx, false)?;
    let tv = schema.def_var("time", NcType::Double, &[td])?;
    schema.put_var_attr(
        tv,
        "units",
        AttrValue::Text("seconds since 1970-01-01 00:00:00".into()),
    )?;
    let yv = schema.def_var("latitude", NcType::Double, &[yd])?;
    let xv = schema.def_var("longitude", NcType::Double, &[xd])?;
    let bv = schema.def_var("bed", NcType::Float, &[yd, xd])?;
    schema.put_var_attr(bv, "units", AttrValue::Text("m".into()))?;
    let mut fields = Vec::new();
    for (name, units) in [("u", "m s-1"), ("v", "m s-1"), ("p", "Pa")] {
        let id = schema.def_var(name, NcType::Float, &[td, yd, xd])?;
        schema.put_var_attr(id, "units", AttrValue::Text(units.into()))?;
        fields.push(id);
    }
    let mut w = NcWriter::create(out.join("inputs.nc"), schema)?;
    let times: Vec<f64> = (0..nt).map(|i| 1664236800. + i as f64 * 3600.).collect();
    let x: Vec<f64> = (0..nx).map(|i| i as f64 * 0.6 / (nx - 1) as f64).collect();
    let y: Vec<f64> = (0..ny)
        .map(|i| -0.2 + i as f64 * 0.4 / (ny - 1) as f64)
        .collect();
    let bed: Vec<f32> = (0..n)
        .into_par_iter()
        .map(|k| {
            if kind == "deep-pressure" {
                -2000.
            } else if kind == "pressure" || kind == "tide" {
                -20.
            } else {
                -4. + 5. * (k % nx) as f32 / (nx - 1) as f32 + 0.15 * ((k / nx) as f32 * 0.05).sin()
            }
        })
        .collect();
    w.write_var(tv, VarData::F64(&times))?;
    w.write_var(xv, VarData::F64(&x))?;
    w.write_var(yv, VarData::F64(&y))?;
    w.write_var(bv, VarData::F32(&bed))?;
    for (c, id) in fields.into_iter().enumerate() {
        let values: Vec<f32> = (0..nt * n)
            .into_par_iter()
            .map(|k| match c {
                0 => {
                    if kind == "wind" {
                        20.
                    } else {
                        0.
                    }
                }
                1 => 0.,
                _ => {
                    if kind == "pressure" || kind == "deep-pressure" {
                        101325. - 1000. * ((k / n) as f32 / 6.).min(1.)
                    } else {
                        101325.
                    }
                }
            })
            .collect();
        w.write_var(id, VarData::F32(&values))?;
    }
    w.finish()?;
    let tides: Vec<f64> = if kind == "tide" {
        (0..nt)
            .map(|i| 0.25 * (std::f64::consts::TAU * i as f64 / 12.).sin())
            .collect()
    } else {
        Vec::new()
    };
    save(
        out.join("case.json"),
        &json!({"grid":{"nx":nx,"ny":ny,"west":0.,"east":0.6,"south":-0.2,"north":0.2},
      "bathymetry":{"path":"inputs.nc","elevation":"bed","latitude":"latitude","longitude":"longitude","vertical_datum":"analytic zero","source":"original analytic physical regression","license":"Apache-2.0"},
      "forcing":{"path":"inputs.nc","latitude":"latitude","longitude":"longitude","time":"time","u10":"u","v10":"v","pressure":"p","source":"original analytic physical regression","license":"Apache-2.0"},
      "duration_seconds":duration,"output_interval_seconds":3600.,"open_boundary_mask":if kind=="lake"||kind=="wind"{0}else{15},"tide_levels_m":tides}),
    )?;
    Ok(())
}
/// The commit this binary was built from; the release pin step reads it from the bytes.
const SOURCE_REV_STAMP: &str = concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));

fn main() {
    std::hint::black_box(SOURCE_REV_STAMP);
    let a: Vec<_> = std::env::args().collect();
    let result=match a.get(1).map(String::as_str){
    Some("--abi")=>{println!("gpuwm-rw-bathy-v1 prepare finish regular-grid SI NOAA-residual");Ok(())},
    Some("fixture") if a.len()==4=>fixture(&a[2],Path::new(&a[3])),
    Some("prepare") if a.len()==4=>prepare(Path::new(&a[2]),Path::new(&a[3])),
    Some("finish") if a.len()==5=>finish(Path::new(&a[2]),Path::new(&a[3]),Path::new(&a[4])),
    _=>Err("usage: rw_bathy prepare CONFIG.json OUTPUT_DIR | finish PREPARED.json RAW_DIR OUTPUT_DIR | --abi".into())};
    if let Err(e) = result {
        eprintln!("rw_bathy: {e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn axes_cover_edges_in_both_directions() {
        for a in [vec![1., 2., 3.], vec![3., 2., 1.]] {
            axis(&a).unwrap();
            for x in [1., 1.5, 2., 2.5, 3.] {
                let (i, w) = bracket(&a, x).unwrap();
                assert!((a[i] * (1. - w) + a[i + 1] * w - x).abs() < 1e-12);
            }
            assert!(bracket(&a, 0.9).is_err());
        }
        assert!(axis(&[1., 1.]).is_err());
        assert!(axis(&[1., f64::NAN]).is_err());
    }
    #[test]
    fn bilinear_preserves_linear_bathymetry() {
        assert_eq!(bilinear(&[-4., -2., 0., 2.], 2, 0, 0, 0.5, 0.25), -2.);
    }
    #[test]
    fn grid_rejects_invalid_geographic_and_allocation_contracts() {
        let g = Grid {
            nx: 800,
            ny: 780,
            west: -87.,
            east: -79.,
            south: 23.,
            north: 30.,
        };
        g.validate().unwrap();
        let mut bad = g.clone();
        bad.west = f64::NAN;
        assert!(bad.validate().is_err());
        bad = g.clone();
        bad.ny = 1;
        assert!(bad.validate().is_err());
        bad = g;
        bad.north = 90.;
        assert!(bad.validate().is_err());
    }
}
