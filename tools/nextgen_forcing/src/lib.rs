// SPDX-License-Identifier: Apache-2.0
//! Setup-only Rust field decoding, units and catchment-area remapping.
//! Source differences are JSON rows. A single Rayon path serves every source.
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, ffi::{c_char, CStr}, fs, io::Write, path::{Path, PathBuf}};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
pub const FIELDS: [&str; 8] = ["T2D", "Q2D", "U2D", "V2D", "PSFC", "LWDOWN", "SWDOWN", "RAINRATE"];

/// Additive ABI1 provenance entry; no numerical or packet contract changes.
pub const SOURCE_REV_STAMP: &str =
    concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));

#[no_mangle]
pub extern "C" fn arwen_forcing_source_revision() -> *const c_char {
    concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"), "\0")
        .as_ptr().cast()
}

#[derive(Clone, Deserialize)]
pub struct Field {
    pub variable: String,
    #[serde(default = "one")] pub scale: f64,
    #[serde(default)] pub offset: f64,
    /// instantaneous rate, interval total, cumulative since an explicit baseline,
    /// or daily_running_total: a total that restarts once per UTC day (ERA5-Land
    /// style), de-accumulated against the previous sample of the same day.
    #[serde(default = "rate")] pub temporal: String,
    /// daily_running_total only: UTC second of day at which the running total
    /// restarts (0 = the first interval of each day starts at 00 UTC).
    #[serde(default)] pub accumulation_reset_seconds: f64,
    /// Optional floor applied after the unit transform. A de-accumulated total
    /// below it (source packing noise) is raised to it; without it, a negative
    /// de-accumulated total is an error.
    #[serde(default)] pub clamp_min: Option<f64>,
}
fn one() -> f64 { 1.0 }
fn rate() -> String { "rate".into() }
fn accumulated(field: &Field) -> bool { matches!(field.temporal.as_str(), "cumulative" | "daily_running_total") }

#[derive(Clone, Copy, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Humidity {
    #[default]
    SpecificHumidity,
    MixingRatio,
    /// Q2D carries 2 m dewpoint temperature in K (after scale/offset). Specific
    /// humidity is computed per source cell, before area remapping, from the
    /// saturation vapour pressure over water at the dewpoint and the PSFC field
    /// of the same cell (ECMWF IFS Cy41r2 Part IV, eqs. 7.4-7.5: Tetens with
    /// a1=611.21 Pa, a3=17.502, a4=32.19 K, T0=273.16 K; q = eps e / (p - (1-eps) e),
    /// eps = Rdry/Rvap = 0.621981), the conversion ECMWF documents for ERA5.
    DewpointKelvin,
}

/// One time chunk of gridded input: one file, or several files that together
/// hold the canonical fields on the same clock (e.g. a source that splits
/// instantaneous and accumulated variables into separate files).
#[derive(Clone, Deserialize)]
#[serde(untagged)]
pub enum InputGroup {
    One(PathBuf),
    Many(Vec<PathBuf>),
}
impl InputGroup {
    pub fn paths(&self) -> Vec<&PathBuf> {
        match self { InputGroup::One(p) => vec![p], InputGroup::Many(v) => v.iter().collect() }
    }
}
impl From<PathBuf> for InputGroup { fn from(p: PathBuf) -> Self { InputGroup::One(p) } }

#[derive(Default, Deserialize)]
#[serde(tag = "frame", rename_all = "snake_case")]
pub enum Wind {
    #[default]
    EarthRelative,
    /// Angle in radians, positive counterclockwise from earth east to grid east.
    GridRelative { angle: Field },
}

#[derive(Deserialize)]
pub struct Profile {
    pub format: String,
    pub fields: BTreeMap<String, Field>,
    /// Q2D quantity after its declared scale/offset, before area remapping.
    #[serde(default)] pub humidity: Humidity,
    /// Resolve the source vector basis before catchment area averaging.
    #[serde(default)] pub wind: Wind,
    /// CF coordinate carrying interval starts; scalar or one-dimensional.
    #[serde(default = "time_variable")] pub time_variable: String,
    /// Explicitly align an interval-end coordinate, e.g. -3600 for hourly totals.
    #[serde(default)] pub time_offset_seconds: f64,
}
fn time_variable() -> String { "time".into() }

#[derive(Deserialize)]
pub struct Weights {
    pub catchment_ids: Vec<String>,
    pub grid_shape: Vec<usize>,
    pub offsets: Vec<u64>,
    pub indices: Vec<u32>,
    pub weights: Vec<f64>,
}
impl Weights {
    pub fn validate(&self) -> Result<usize> {
        let n = self.catchment_ids.len();
        let cells = self.grid_shape.iter().try_fold(1usize, |a, &b| a.checked_mul(b)).ok_or("grid size overflow")?;
        if n == 0 || cells == 0 || self.offsets.len() != n + 1 || self.offsets[0] != 0 ||
           self.offsets[n] as usize != self.indices.len() || self.indices.len() != self.weights.len() {
            return Err("invalid area weight CSR dimensions".into());
        }
        let unique: std::collections::HashSet<_> = self.catchment_ids.iter().collect();
        if unique.len() != n { return Err("duplicate catchment ID".into()); }
        for c in 0..n {
            let a=self.offsets[c] as usize; let b=self.offsets[c+1] as usize;
            if a >= b || b > self.indices.len() { return Err("empty or unordered catchment CSR row".into()); }
            let mut sum=0.0;
            for j in a..b {
                if self.indices[j] as usize >= cells || !self.weights[j].is_finite() || self.weights[j] <= 0.0 {
                    return Err("invalid source cell or area fraction".into());
                }
                sum += self.weights[j];
            }
            if (sum-1.0).abs() > 1e-12 { return Err("catchment area fractions must sum to one".into()); }
        }
        Ok(cells)
    }
}

#[derive(Deserialize)]
pub struct Request {
    pub source_profile: PathBuf,
    pub source_id: Option<String>,
    pub profile_row: Option<String>,
    /// Ordered source grid files (or file groups sharing one clock), or one CSV
    /// per catchment for catchment_csv.
    pub inputs: Vec<InputGroup>,
    pub output: PathBuf,
    pub time_origin: String,
    pub interval_seconds: f64,
    pub catchment_ids: Vec<String>,
    pub area_weights: Option<PathBuf>,
    /// One source-grid accumulation per cell before the first frame, in native units.
    pub rain_baseline: Option<PathBuf>,
}

#[derive(Serialize)]
pub struct Receipt {
    pub schema: &'static str,
    pub shape: [usize;3],
    pub fields: [&'static str;8],
    pub units: [&'static str;8],
    pub time_origin: String,
    pub interval_seconds: f64,
    pub catchment_ids: Vec<String>,
    pub files: BTreeMap<String,String>,
    pub data_sha256: BTreeMap<String,String>,
    pub source_profile_sha256: String,
    pub source_id: Option<String>,
    pub profile_row: Option<String>,
    pub input_sha256: BTreeMap<String,String>,
    pub area_weights_sha256: Option<String>,
    pub parallelism: usize,
    pub preparation_seconds: f64,
}

pub fn sha256(bytes: &[u8]) -> String { format!("{:x}",Sha256::digest(bytes)) }
fn field_order(profile: &Profile) -> Result<Vec<&Field>> {
    FIELDS.iter().map(|key| profile.fields.get(*key).ok_or_else(||format!("missing canonical field {key}").into())).collect()
}
fn transform(values: &mut [f64], field: &Field, dt: f64, previous: Option<&[f64]>) -> Result<()> {
    if !field.scale.is_finite() || !field.offset.is_finite() { return Err("nonfinite unit transform".into()); }
    for (i,x) in values.iter_mut().enumerate() {
        if !x.is_finite() { return Err("missing/nonfinite forcing cell".into()); }
        let native=*x;
        *x=match field.temporal.as_str() {
            "rate" => native*field.scale+field.offset,
            "interval_total" if field.offset==0.0 => native*field.scale/dt,
            "cumulative" if field.offset==0.0 => {
                let prior=previous.ok_or("cumulative forcing requires explicit first baseline")?[i];
                if !prior.is_finite() || native < prior { return Err("negative/reset cumulative forcing requires a new explicit baseline".into()); }
                (native-prior)*field.scale/dt
            },
            // The caller passes no previous sample exactly at the daily reset.
            "daily_running_total" if field.offset==0.0 => {
                let prior=previous.map(|p|p[i]).unwrap_or(0.0);
                if !prior.is_finite() { return Err("nonfinite previous running total".into()); }
                let value=(native-prior)*field.scale/dt;
                if value<0.0 && field.clamp_min.is_none() {
                    return Err("daily running total decreased within its day; declare clamp_min for source packing noise".into());
                }
                value
            },
            _ => return Err("unknown temporal mode or nonzero total offset".into()),
        };
        if let Some(floor)=field.clamp_min { if *x<floor { *x=floor; } }
        if !x.is_finite() { return Err("nonfinite transformed forcing".into()); }
    }
    Ok(())
}

fn specific_humidity(value: f64, humidity: Humidity) -> Result<f64> {
    match humidity {
        Humidity::SpecificHumidity => Ok(value),
        Humidity::MixingRatio => {
            // r = water/dry-air mass; q = water/(water+dry-air) mass.
            // Convert source cells before CSR: this mapping is nonlinear.
            if value < 0.0 { return Err("negative humidity mixing ratio".into()); }
            Ok(value / (1.0 + value))
        },
        Humidity::DewpointKelvin => Err("dewpoint humidity needs the cell's surface pressure".into()),
    }
}

/// Specific humidity (kg/kg) from dewpoint (K) and surface pressure (Pa); see
/// Humidity::DewpointKelvin for the cited constants.
fn specific_humidity_from_dewpoint(td: f64, p: f64) -> Result<f64> {
    if !(td > 150.0 && td < 350.0) || !(p > 1.0e4 && p < 1.2e5) {
        return Err("dewpoint/surface pressure outside physical range".into());
    }
    const EPS: f64 = 0.621981;
    let e = 611.21 * (17.502 * (td - 273.16) / (td - 32.19)).exp();
    if e >= p { return Err("vapour pressure at or above surface pressure".into()); }
    Ok(EPS * e / (p - (1.0 - EPS) * e))
}

fn rotate_wind(u: f64, v: f64, angle: f64) -> (f64, f64) {
    // Components of u*grid_east + v*grid_north in the earth east/north basis.
    let (sin, cos) = angle.sin_cos();
    (u*cos-v*sin, u*sin+v*cos)
}

/// Fixed-duration CF units only: no calendar approximation for weather clocks.
/// CF Conventions 1.12 section 4.4, public domain (CC0).
fn grid_time_delta(value: f64, units: &str, calendar: &str,
                   origin: chrono::NaiveDateTime, offset: f64) -> Result<f64> {
    if !matches!(calendar, "standard" | "gregorian" | "proleptic_gregorian") {
        return Err("forcing time requires a Gregorian UTC calendar".into());
    }
    let (unit, epoch)=units.split_once(" since ").ok_or("forcing time units require '<unit> since <epoch>'")?;
    let scale=match unit.trim() {
        "s" | "second" | "seconds" => 1.0,
        "min" | "minute" | "minutes" => 60.0,
        "h" | "hr" | "hour" | "hours" => 3600.0,
        "d" | "day" | "days" => 86400.0,
        _ => return Err("forcing time requires seconds, minutes, hours or days".into()),
    };
    if !value.is_finite() || !offset.is_finite() { return Err("nonfinite forcing time coordinate/offset".into()); }
    let epoch=parse_time(epoch.trim())?;
    let epoch_delta=(epoch-origin).num_nanoseconds().ok_or("forcing time epoch exceeds supported UTC range")? as f64/1e9;
    Ok(epoch_delta+value*scale+offset)
}

fn validate_grid_times(file: &netcdf_reader::NcFile, profile: &Profile,
                       origin: chrono::NaiveDateTime, first: usize,
                       cadence: f64) -> Result<(Option<String>,usize)> {
    let variable=file.variable(&profile.time_variable)?;
    let units=variable.attribute("units").and_then(|a|a.value.as_string())
        .ok_or("forcing time coordinate lacks CF units")?;
    let calendar=match variable.attribute("calendar") {
        Some(a) => a.value.as_string().ok_or("forcing calendar must be text")?,
        None => "standard".into(),
    };
    let data=file.read_variable_unpacked_masked(&profile.time_variable)?;
    if data.is_empty() || variable.dimensions().len()>1 {
        return Err("forcing time coordinate must be nonempty and scalar or one-dimensional".into());
    }
    for (i,&value) in data.iter().enumerate() {
        let elapsed=grid_time_delta(value,&units,&calendar,origin,profile.time_offset_seconds)?;
        let expected=(first+i) as f64*cadence;
        if elapsed!=expected {
            return Err(format!("forcing grid timestamp differs from realization clock: coordinate {} sample {i} elapsed {elapsed}s, expected {expected}s",
                               profile.time_variable).into());
        }
    }
    Ok((variable.dimensions().first().map(|d|d.name.clone()),data.len()))
}

/// Normalize only the time axis. Spatial order remains the weight grid's order.
fn read_grid_column(file: &netcdf_reader::NcFile, field: &Field,
                    time_dimension: Option<&str>, nt: usize, weights: &Weights,
                    cells: usize) -> Result<Vec<f64>> {
    let variable=file.variable(&field.variable)?;
    let data=file.read_variable_unpacked_masked(&field.variable)?;
    let axis=variable.dimensions().iter().position(|d|Some(d.name.as_str())==time_dimension);
    if axis.is_none() && nt!=1 {
        return Err(format!("{} lacks time dimension for {nt} forcing samples",field.variable).into());
    }
    if axis.is_some_and(|a|data.shape()[a]!=nt) || data.len()!=nt.checked_mul(cells).ok_or("forcing grid size overflow")? {
        return Err(format!("{} grid shape differs from time/area weights",field.variable).into());
    }
    let spatial:Vec<usize>=data.shape().iter().enumerate()
        .filter_map(|(a,&s)|if Some(a)!=axis && s!=1 {Some(s)}else{None}).collect();
    let expected:Vec<usize>=weights.grid_shape.iter().copied().filter(|&s|s!=1).collect();
    if spatial!=expected { return Err("forcing spatial axes differ from area weight grid".into()); }
    let values:Vec<f64>=data.iter().copied().collect();
    let stride=axis.map(|a|data.shape()[a+1..].iter().product()).unwrap_or(cells);
    if nt==1 || stride==cells { return Ok(values); }
    let mut ordered=vec![0.0;values.len()];
    ordered.par_chunks_mut(cells).enumerate().for_each(|(t,row)| {
        for (c,x) in row.iter_mut().enumerate() {
            *x=values[(c/stride)*nt*stride+t*stride+c%stride];
        }
    });
    Ok(ordered)
}

pub fn prepare(request: Request) -> Result<Receipt> {
    let started=std::time::Instant::now();
    if !request.interval_seconds.is_finite() || request.interval_seconds<=0.0 || request.time_origin.is_empty() || request.inputs.is_empty() {
        return Err("forcing requires inputs, positive cadence and explicit UTC origin".into());
    }
    let profile_bytes=fs::read(&request.source_profile)?;
    let table:serde_json::Value=serde_json::from_slice(&profile_bytes)?;
    let profile: Profile=if let Some(row)=&request.profile_row {
        serde_json::from_value(table.get("profiles").and_then(|p|p.get(row)).ok_or("missing source profile row")?.clone())?
    } else {serde_json::from_value(table)?};
    let fields=field_order(&profile)?;
    if let Wind::GridRelative { angle }=&profile.wind {
        if angle.temporal!="rate" { return Err("wind angle must be instantaneous, with temporal=rate".into()); }
    }
    let n=request.catchment_ids.len();
    if n==0 { return Err("catchment ordering must be explicit".into()); }
    if request.catchment_ids.iter().collect::<std::collections::HashSet<_>>().len()!=n {return Err("duplicate catchment ID".into());}
    let weight_bytes=request.area_weights.as_ref().map(fs::read).transpose()?;
    let weights: Option<Weights>=weight_bytes.as_ref().map(|b|serde_json::from_slice(b)).transpose()?;
    let mut input_sha256=BTreeMap::new();
    for group in &request.inputs { for path in group.paths() { input_sha256.insert(path.display().to_string(),sha256(&fs::read(path)?)); } }
    if let Some(path)=&request.rain_baseline {input_sha256.insert(path.display().to_string(),sha256(&fs::read(path)?));}
    let (nt,slab)=match profile.format.as_str() {
        "catchment_csv" => {
            let files: Vec<&PathBuf>=request.inputs.iter().map(|g| match g {
                InputGroup::One(p) => Ok(p),
                InputGroup::Many(_) => Err("catchment CSV takes one file per catchment, not file groups"),
            }).collect::<std::result::Result<_,_>>()?;
            if files.len()!=n || weights.is_some() { return Err("catchment CSV requires exactly one ordered file per catchment and no regrid".into()); }
            if matches!(profile.humidity, Humidity::DewpointKelvin) { return Err("dewpoint humidity is supported for gridded inputs only".into()); }
            let series: Vec<(Vec<String>,Vec<f64>)>=files.par_iter().map(|path| -> Result<_> {
                let mut reader=csv::Reader::from_path(path)?;
                let headers=reader.headers()?.clone();
                let columns: Vec<usize>=fields.iter().map(|f|headers.iter().position(|h|h==f.variable).ok_or_else(||format!("missing CSV field {}",f.variable))).collect::<std::result::Result<_,_>>()?;
                let time= headers.iter().position(|h|h=="time").ok_or("CSV requires time column")?;
                let angle_column=match &profile.wind {
                    Wind::EarthRelative => None,
                    Wind::GridRelative { angle } => Some(headers.iter().position(|h|h==angle.variable)
                        .ok_or_else(||format!("missing CSV wind angle {}",angle.variable))?),
                };
                let mut times=Vec::new(); let mut data=Vec::new();
                for record in reader.records() {
                    let record=record?; times.push(record[time].to_string());
                    let mut row=[0.0;8];
                    for (f,(field,&column)) in fields.iter().zip(&columns).enumerate() {
                        if accumulated(field) { return Err("catchment CSV cumulative fields require preconversion with explicit baseline".into()); }
                        let mut x=[record[column].parse::<f64>()?];
                        transform(&mut x,field,request.interval_seconds,None)?;
                        if f==1 { x[0]=specific_humidity(x[0],profile.humidity)?; }
                        row[f]=x[0];
                    }
                    if let (Wind::GridRelative { angle }, Some(column))=(&profile.wind,angle_column) {
                        let mut theta=[record[column].parse::<f64>()?];
                        transform(&mut theta,angle,request.interval_seconds,None)?;
                        (row[2],row[3])=rotate_wind(row[2],row[3],theta[0]);
                    }
                    data.extend(row);
                }
                Ok((times,data))
            }).collect::<Result<_>>()?;
            let nt=series[0].0.len();
            if nt==0 || series.iter().any(|s|s.0!=series[0].0) { return Err("forcing catchment times differ or are empty".into()); }
            let origin=parse_time(&request.time_origin)?;
            for (i,date) in series[0].0.iter().enumerate() {
                let elapsed=(parse_time(date)?-origin).num_milliseconds() as f64/1000.0;
                if elapsed != i as f64*request.interval_seconds {
                    return Err("CSV origin/cadence differs from realization clock".into());
                }
            }
            let mut out=vec![0.0;nt*8*n];
            out.par_chunks_mut(8*n).enumerate().for_each(|(t,row)| {
                for c in 0..n { for f in 0..8 { row[f*n+c]=series[c].1[t*8+f]; } }
            });
            fs::create_dir_all(&request.output)?;
            fs::write(request.output.join("source-times.json"),serde_json::to_vec(&series[0].0)?)?;
            (nt,out)
        },
        "grid_netcdf" => {
            let weights=weights.as_ref().ok_or("gridded forcing requires area weights")?;
            let cells=weights.validate()?;
            if weights.catchment_ids!=request.catchment_ids { return Err("weights and realization catchment order differ".into()); }
            // Per-field previous raw sample for accumulated fields; only rain can
            // carry an explicit baseline across the first frame.
            let mut previous: Vec<Option<Vec<f64>>>=vec![None;8];
            previous[7]=request.rain_baseline.as_ref().map(|p|read_f64(p)).transpose()?;
            if previous[7].as_ref().is_some_and(|v|v.len()!=cells) { return Err("rain baseline does not match grid".into()); }
            // Cells no catchment reads (e.g. sea cells a land-only source leaves
            // missing) are set to zero before any transform; they never reach a
            // catchment value, so a missing value there is not an input error.
            let mut used=vec![false;cells];
            for &i in &weights.indices { used[i as usize]=true; }
            let used_cells: Vec<usize>=(0..cells).filter(|&i|used[i]).collect();
            let mut out=Vec::new();
            let mut total_frames=0;
            let origin=parse_time(&request.time_origin)?;
            for group in &request.inputs {
                let paths=group.paths();
                let files: Vec<netcdf_reader::NcFile>=paths.iter().map(|p|netcdf_reader::NcFile::open(p)).collect::<std::result::Result<_,_>>()?;
                let mut clock=None;
                for (file,path) in files.iter().zip(&paths) {
                    let this=validate_grid_times(file,&profile,origin,total_frames,request.interval_seconds)
                        .map_err(|e|format!("{}: {e}",path.display()))?;
                    if clock.as_ref().is_some_and(|c|*c!=this) { return Err("files of one input group differ in time axis".into()); }
                    clock=Some(this);
                }
                let (time_dimension,nt)=clock.ok_or("empty input group")?;
                let holder=|name: &str| -> Result<&netcdf_reader::NcFile> {
                    files.iter().find(|f|f.variable(name).is_ok())
                        .ok_or_else(||format!("no file of the input group holds {name}").into())
                };
                let mut columns:Vec<Vec<f64>>=fields.par_iter().map(|f| -> Result<Vec<f64>> {
                    let mut v=read_grid_column(holder(&f.variable)?,f,time_dimension.as_deref(),nt,weights,cells)?;
                    v.par_chunks_mut(cells).for_each(|frame| for (x,&u) in frame.iter_mut().zip(&used) { if !u { *x=0.0; } });
                    Ok(v)
                }).collect::<Result<_>>()?;
                let angles=match &profile.wind {
                    Wind::EarthRelative => None,
                    Wind::GridRelative { angle } => {
                        let file=holder(&angle.variable)?;
                        let variable=file.variable(&angle.variable)?;
                        let has_time=variable.dimensions().iter().any(|d|Some(d.name.as_str())==time_dimension.as_deref());
                        let mut values=read_grid_column(file,angle,if has_time {time_dimension.as_deref()} else {None},
                            if has_time {nt} else {1},weights,cells)?;
                        values.par_iter_mut().try_for_each(|value|
                            transform(std::slice::from_mut(value),angle,request.interval_seconds,None))?;
                        Some(values)
                    },
                };
                for t in 0..nt {
                    let a=t*cells; let b=a+cells;
                    // Interval start of this frame; a daily running total restarts
                    // when the interval starts at its declared UTC reset second.
                    let start=origin+chrono::Duration::milliseconds(((total_frames+t) as f64*request.interval_seconds*1000.0).round() as i64);
                    let second_of_day=chrono::Timelike::num_seconds_from_midnight(&start.time()) as f64;
                    let raw: Vec<Option<Vec<f64>>>=columns.iter().zip(&fields)
                        .map(|(c,f)|accumulated(f).then(||c[a..b].to_vec())).collect();
                    for (f,column) in columns.iter_mut().enumerate() {
                        let prior=match fields[f].temporal.as_str() {
                            "daily_running_total" if second_of_day==fields[f].accumulation_reset_seconds => None,
                            "daily_running_total" => Some(previous[f].as_deref().ok_or(
                                "daily running total needs its previous sample: start the series at the daily reset")?),
                            "cumulative" => previous[f].as_deref(),
                            _ => None,
                        };
                        transform(&mut column[a..b],fields[f],request.interval_seconds,prior)?;
                    }
                    match profile.humidity {
                        Humidity::SpecificHumidity => {},
                        Humidity::MixingRatio => columns[1][a..b].par_iter_mut().try_for_each(|q| -> Result<()> {
                            *q=specific_humidity(*q,profile.humidity)?; Ok(())
                        })?,
                        Humidity::DewpointKelvin => {
                            let (left,right)=columns.split_at_mut(4);
                            let (q,p)=(&mut left[1][a..b],&right[0][a..b]);
                            let values: Vec<(usize,f64)>=used_cells.par_iter()
                                .map(|&i|specific_humidity_from_dewpoint(q[i],p[i]).map(|v|(i,v))).collect::<Result<_>>()?;
                            for (i,v) in values { q[i]=v; }
                        },
                    }
                    if let Some(angles)=&angles {
                        let offset=if angles.len()==cells {0} else {a};
                        let (left,right)=columns.split_at_mut(3);
                        left[2][a..b].par_iter_mut().zip(right[0][a..b].par_iter_mut())
                            .zip(angles[offset..offset+cells].par_iter()).for_each(|((u,v),&theta)| {
                                (*u,*v)=rotate_wind(*u,*v,theta);
                            });
                    }
                    for (f,r) in raw.into_iter().enumerate() { if r.is_some() { previous[f]=r; } }
                    let mut row=vec![0.0;8*n];
                    row.par_iter_mut().enumerate().for_each(|(k,value)| {
                        let f=k/n; let c=k%n;
                        for j in weights.offsets[c] as usize..weights.offsets[c+1] as usize {
                            *value += columns[f][a+weights.indices[j] as usize]*weights.weights[j];
                        }
                    });
                    out.extend(row);
                }
                total_frames+=nt;
            }
            (total_frames,out)
        },
        _ => return Err("unsupported forcing format: use declared catchment_csv or grid_netcdf".into()),
    };
    if slab.iter().enumerate().any(|(i,v)|!v.is_finite() || !(*v as f32).is_finite() || (i/n%8==7 && *v<0.0)) { return Err("forcing overflows resident precision or has negative rain".into()); }
    fs::create_dir_all(&request.output)?;
    // Encode both precisions on all cores and write each file in one call; the
    // earlier per-value write_all made one syscall per word (single core, ~45 s
    // for a 744 h x 18,917 catchment packet). Bytes are unchanged.
    let f64bytes: Vec<u8>=slab.par_iter().flat_map_iter(|v|v.to_le_bytes()).collect();
    let f32bytes: Vec<u8>=slab.par_iter().flat_map_iter(|v|(*v as f32).to_le_bytes()).collect();
    let (h64,h32)=rayon::join(
        || -> Result<String> { fs::write(request.output.join("forcing.f64"),&f64bytes)?; Ok(sha256(&f64bytes)) },
        || -> Result<String> { fs::write(request.output.join("forcing.f32"),&f32bytes)?; Ok(sha256(&f32bytes)) });
    let files=[("forcing.f64".into(),"<f8".into()),("forcing.f32".into(),"<f4".into())].into();
    let mut data_sha256=BTreeMap::new();
    data_sha256.insert("forcing.f64".into(),h64?);
    data_sha256.insert("forcing.f32".into(),h32?);
    let receipt=Receipt {schema:"arwen.nextgen-forcing.v1",shape:[nt,8,n],fields:FIELDS,
        units:["K","kg/kg","m/s","m/s","Pa","W/m2","W/m2","mm/s"],
        time_origin:request.time_origin, interval_seconds:request.interval_seconds,
        catchment_ids:request.catchment_ids,files,data_sha256,source_profile_sha256:sha256(&profile_bytes),
        profile_row:request.profile_row,source_id:request.source_id,input_sha256,area_weights_sha256:weight_bytes.as_ref().map(|b|sha256(b)),
        parallelism:rayon::current_num_threads(),preparation_seconds:started.elapsed().as_secs_f64()};
    fs::write(request.output.join("manifest.json"),serde_json::to_vec_pretty(&receipt)?)?;
    Ok(receipt)
}

fn parse_time(text:&str)->Result<chrono::NaiveDateTime> {
    let text=text.trim().strip_suffix(" UTC").unwrap_or(text.trim());
    if let Ok(time)=chrono::DateTime::parse_from_rfc3339(text) {return Ok(time.naive_utc());}
    for format in ["%Y-%m-%d %H:%M:%S%.f %:z", "%Y-%m-%d %H:%M:%S%.f%:z"] {
        if let Ok(time)=chrono::DateTime::parse_from_str(text,format) {return Ok(time.naive_utc());}
    }
    for format in ["%Y-%m-%d %H:%M:%S%.f", "%Y-%m-%dT%H:%M:%S%.f"] {
        if let Ok(time)=chrono::NaiveDateTime::parse_from_str(text,format) {return Ok(time);}
    }
    Ok(chrono::NaiveDate::parse_from_str(text,"%Y-%m-%d")?.and_hms_opt(0,0,0).unwrap())
}

#[no_mangle]
pub extern "C" fn arwen_forcing_abi_version()->u32 {1}

/// Setup-only metadata request; numerical decoding/regridding runs in Rust.
/// Safety: path points to a NUL-terminated UTF-8 request filename.
#[no_mangle]
pub unsafe extern "C" fn arwen_forcing_prepare(path:*const c_char)->i32 {
    if path.is_null() {return 1;}
    let Ok(path)=CStr::from_ptr(path).to_str() else {return 1;};
    let result=(||->Result<()> {
        let request:Request=serde_json::from_slice(&fs::read(path)?)?;
        prepare(request)?; Ok(())
    })();
    match result {Ok(())=>0,Err(e)=>{eprintln!("forcing preparation: {e}");1}}
}

/// Read a prepared typed packet after checking its content hash, before upload.
/// Safety: paths are NUL terminated; output addresses len writable bytes.
#[no_mangle]
pub unsafe extern "C" fn arwen_forcing_read_packet(path:*const c_char, expected_sha:*const c_char, output:*mut u8, len:usize)->i32 {
    if path.is_null() || expected_sha.is_null() || output.is_null() {return 1;}
    let (Ok(path),Ok(hash))=(CStr::from_ptr(path).to_str(),CStr::from_ptr(expected_sha).to_str()) else {return 1;};
    let Ok(bytes)=fs::read(path) else {return 1;};
    if bytes.len()!=len || sha256(&bytes)!=hash {return 1;}
    std::ptr::copy_nonoverlapping(bytes.as_ptr(),output,len); 0
}

pub fn read_f64(path: &Path) -> Result<Vec<f64>> {
    let data=fs::read(path)?;
    if data.len()%8!=0 {return Err("truncated float64 packet".into());}
    Ok(data.chunks_exact(8).map(|b|f64::from_le_bytes(b.try_into().unwrap())).collect())
}

/// Setup/replay I/O seam; Python only supplies paths and allocated host storage.
/// Returns 0 on success, 1 on invalid arguments or I/O; never unwinds across C.
/// Safety: path is NUL-terminated; output addresses len writable doubles.
#[no_mangle]
pub unsafe extern "C" fn arwen_forcing_read_f64(path:*const c_char, output:*mut f64, len:usize)->i32 {
    if path.is_null() || output.is_null() {return 1;}
    let Ok(path)=CStr::from_ptr(path).to_str() else {return 1;};
    let Ok(data)=read_f64(Path::new(path)) else {return 1;};
    if data.len()!=len {return 1;}
    std::ptr::copy_nonoverlapping(data.as_ptr(),output,len); 0
}
/// Safety: path is NUL-terminated; input addresses len readable doubles.
#[no_mangle]
pub unsafe extern "C" fn arwen_forcing_write_f64(path:*const c_char, input:*const f64, len:usize)->i32 {
    if path.is_null() || input.is_null() {return 1;}
    let Ok(path)=CStr::from_ptr(path).to_str() else {return 1;};
    let Ok(mut file)=fs::File::create(path) else {return 1;};
    for x in std::slice::from_raw_parts(input,len) {if file.write_all(&x.to_le_bytes()).is_err(){return 1;}}
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn csv_quarter_turn_is_earth_north() {
        let root=std::env::temp_dir().join(format!("nextgen-csv-wind-{}",std::process::id()));
        fs::create_dir(&root).unwrap();
        let table=root.join("profile.json");
        let csv=root.join("input.csv");
        let fields: BTreeMap<_,_>=FIELDS.iter().map(|&name|(name,serde_json::json!({"variable":name}))).collect();
        fs::write(&table,serde_json::to_vec(&serde_json::json!({"format":"catchment_csv","fields":fields,
            "wind":{"frame":"grid_relative","angle":{"variable":"bearing","scale":std::f64::consts::PI/180.0}}})).unwrap()).unwrap();
        fs::write(&csv,"time,T2D,Q2D,U2D,V2D,PSFC,LWDOWN,SWDOWN,RAINRATE,bearing\n2015-12-01T00:00:00Z,300,0.01,1,0,100000,350,200,0.001,90\n").unwrap();
        let output=root.join("packet");
        let request=Request {source_profile:table,source_id:None,profile_row:None,inputs:vec![csv.into()],output:output.clone(),
            time_origin:"2015-12-01T00:00:00Z".into(),interval_seconds:3600.0,
            catchment_ids:vec!["catchment".into()],area_weights:None,rain_baseline:None};
        let result=prepare(request).unwrap();
        assert_eq!(result.shape,[1,8,1]);
        let data=read_f64(&output.join("forcing.f64")).unwrap();
        assert!(data[2].abs()<1e-15,"quarter-turn must remove earth-east wind");
        assert_eq!(data[3],1.0,"grid east rotated counterclockwise becomes earth north");
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn totals_and_cumulative_rain_have_distinct_first_epochs() {
        let mut rain=[12.0,24.0];
        let mut field=Field{variable:"rain".into(),scale:1.0,offset:0.0,temporal:"interval_total".into(),
            accumulation_reset_seconds:0.0,clamp_min:None};
        transform(&mut rain,&field,2.0,None).unwrap();
        assert_eq!(rain,[6.0,12.0]);
        field.temporal="cumulative".into();
        assert!(transform(&mut [12.0],&field,2.0,None).is_err());
        assert!(transform(&mut [9.0],&field,2.0,Some(&[10.0])).is_err());
        let mut next=[12.0,24.0];
        transform(&mut next,&field,2.0,Some(&[10.0,20.0])).unwrap();
        assert_eq!(next,[1.0,2.0]);
    }
    #[test]
    fn daily_running_total_restarts_and_clamps() {
        let mut field=Field{variable:"tp".into(),scale:1000.0,offset:0.0,temporal:"daily_running_total".into(),
            accumulation_reset_seconds:0.0,clamp_min:None};
        // First interval of the day: the running total itself (m -> mm/s over 3600 s).
        let mut first=[0.0036];
        transform(&mut first,&field,3600.0,None).unwrap();
        assert!((first[0]-0.001).abs()<1e-15);
        // Later interval: difference to the previous sample of the same day.
        let mut later=[0.0108];
        transform(&mut later,&field,3600.0,Some(&[0.0036])).unwrap();
        assert!((later[0]-0.002).abs()<1e-15);
        // Packing noise below the previous sample errors unless a floor is declared.
        assert!(transform(&mut [0.0035],&field,3600.0,Some(&[0.0036])).is_err());
        field.clamp_min=Some(0.0);
        let mut noisy=[0.0035];
        transform(&mut noisy,&field,3600.0,Some(&[0.0036])).unwrap();
        assert_eq!(noisy[0],0.0);
    }
    #[test]
    fn dewpoint_humidity_matches_saturation_at_air_temperature() {
        // Td = 273.16 K at 1000 hPa: e = 611.21 Pa, q = eps e / (p - (1-eps) e).
        let q=specific_humidity_from_dewpoint(273.16,100000.0).unwrap();
        let e=611.21; let eps=0.621981;
        assert!((q-eps*e/(100000.0-(1.0-eps)*e)).abs()<1e-15);
        // Higher dewpoint, more water; out-of-range input refuses.
        assert!(specific_humidity_from_dewpoint(290.0,100000.0).unwrap()>q);
        assert!(specific_humidity_from_dewpoint(f64::NAN,100000.0).is_err());
        assert!(specific_humidity_from_dewpoint(280.0,0.0).is_err());
    }
    #[test]
    fn input_groups_accept_one_file_or_several() {
        let one: InputGroup=serde_json::from_str("\"/a.nc\"").unwrap();
        let many: InputGroup=serde_json::from_str("[\"/a.nc\",\"/b.nc\"]").unwrap();
        assert_eq!(one.paths().len(),1);
        assert_eq!(many.paths().len(),2);
    }
    #[test]
    fn csr_rejects_wrong_grid_and_non_normalized_area() {
        let mut w=Weights{catchment_ids:vec!["cat-1".into()],grid_shape:vec![2,2],offsets:vec![0,2],indices:vec![0,4],weights:vec![0.25,0.75]};
        assert!(w.validate().is_err());
        w.indices[1]=3;
        assert_eq!(w.validate().unwrap(),4);
        w.weights[1]=0.5;
        assert!(w.validate().is_err());
    }
    #[test]
    fn utc_offsets_map_to_same_instant() {
        assert_eq!(parse_time("2015-12-01T00:00:00Z").unwrap(),parse_time("2015-11-30T19:00:00-05:00").unwrap());
        assert!(parse_time("2015-12-01 00:00:00 garbage").is_err());
    }
    #[test]
    fn gridded_clock_preserves_units_offsets_and_calendar() {
        let origin=parse_time("2015-12-01T00:00:00Z").unwrap();
        assert_eq!(grid_time_delta(2.0,"hours since 2015-12-01 00:00:00","standard",origin,-3600.0).unwrap(),3600.0);
        assert_eq!(grid_time_delta(60.0,"minutes since 2015-11-30 19:00:00 -05:00","gregorian",origin,0.0).unwrap(),3600.0);
        assert_eq!(grid_time_delta(0.5,"seconds since 2015-12-01 00:00:00 UTC","proleptic_gregorian",origin,0.0).unwrap(),0.5);
        assert!(grid_time_delta(1.0,"months since 2015-12-01","standard",origin,0.0).is_err());
        assert!(grid_time_delta(1.0,"days since 2015-12-01","360_day",origin,0.0).is_err());
        assert!(grid_time_delta(f64::NAN,"hours since 2015-12-01","standard",origin,0.0).is_err());
    }
}
