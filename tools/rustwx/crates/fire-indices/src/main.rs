mod climatology;
mod compare;
mod nfdrs_verify;
mod station;
// Streaming hourly regular-NetCDF front door. Python only launches this binary.
use chrono::{DateTime, Datelike, Timelike, Utc};
use fire_indices::{FwiState, KbdiState, Weather, nfdrs};
mod nelson_cuda;
use netcdf_writer::{AttrValue, NcFormat, NcType, NcWriter, Schema, VarData};
use netcrust::{NcSliceInfo, NcSliceInfoElem};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
    time::Instant,
};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
const PRODUCTS: [&str; 17] = [
    "fwi_ffmc",
    "fwi_dmc",
    "fwi_dc",
    "fwi_isi",
    "fwi_bui",
    "fwi_fwi",
    "kbdi",
    "nfdrs_mc1",
    "nfdrs_mc10",
    "nfdrs_mc100",
    "nfdrs_mc1000",
    "nfdrs_live_herb",
    "nfdrs_live_wood",
    "nfdrs_gsi",
    "nfdrs_erc",
    "nfdrs_bi",
    "nfdrs_sc",
];
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    inputs: Vec<PathBuf>,
    #[serde(default)]
    nfdrs: nfdrs::Config,
    #[serde(default = "auto_backend")]
    compute_backend: String,
    state_in: Option<PathBuf>,
    state_out: PathBuf,
    output: PathBuf,
    receipt: PathBuf,
    member: String,
    utc_offset_hours: i32,
    annual_rain_mm: Option<f64>,
    annual_rain_grid: Option<PathBuf>,
    annual_rain_source: String,
    initial_kbdi: Option<f64>,
    initial_fwi: Option<FwiState>,
    fork_member: Option<String>,
}
fn auto_backend() -> String {
    "auto".into()
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Cell {
    fwi: FwiState,
    nfdrs: nfdrs::State,
    kbdi: KbdiState,
    rain: f64,
    tmax: f64,
    hours: u32,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct State {
    schema: String,
    nfdrs: nfdrs::Config,
    member: String,
    utc_offset_hours: i32,
    annual_rain_mm: Vec<f64>,
    annual_rain_source: String,
    latitude: Vec<f64>,
    longitude: Vec<f64>,
    last_hour: i64,
    completed_days: u32,
    cells: Vec<Cell>,
    lineage: Vec<String>,
}
fn hash(p: &Path) -> Result<String> {
    use std::io::Read;
    let mut f = fs::File::open(p)?;
    let mut h = Sha256::new();
    let mut b = [0u8; 65536];
    loop {
        let n = f.read(&mut b)?;
        if n == 0 {
            break;
        }
        h.update(&b[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}
fn text(n: &netcrust::File, name: &str, key: &str) -> Result<String> {
    Ok(n.variable(name)
        .ok_or(format!("missing {name}"))?
        .attribute(key)
        .and_then(|v| v.as_string().map(str::to_string))
        .ok_or(format!("missing {name}:{key}"))?)
}
fn plane(
    n: &netcrust::File,
    name: &str,
    t: usize,
    ny: usize,
    nx: usize,
    units: &str,
) -> Result<Vec<f64>> {
    if text(n, name, "units")? != units {
        return Err(format!("{name} requires units {units}").into());
    }
    let v = n.variable(name).ok_or(format!("missing {name}"))?;
    if v.shape() != vec![n.read_f64("time")?.len(), ny, nx] {
        return Err(format!("{name}: expected [time,latitude,longitude]").into());
    }
    let slice = NcSliceInfo {
        selections: vec![
            NcSliceInfoElem::Index(t as u64),
            NcSliceInfoElem::Slice {
                start: 0,
                end: ny as u64,
                step: 1,
            },
            NcSliceInfoElem::Slice {
                start: 0,
                end: nx as u64,
                step: 1,
            },
        ],
    };
    Ok(n.read_array_f64_slice(name, &slice)?.into_values())
}
fn publish(path: &Path, data: &[u8]) -> Result<()> {
    if path.exists() {
        return Err(format!("preserving existing artifact {}", path.display()).into());
    }
    if let Some(p) = path.parent() {
        fs::create_dir_all(p)?;
    }
    let mut f = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)?;
    use std::io::Write;
    f.write_all(data)?;
    f.sync_all()?;
    Ok(())
}
fn writer(
    path: &Path,
    lat: &[f64],
    lon: &[f64],
    member: &str,
    nfdrs: &nfdrs::Config,
) -> Result<(NcWriter, usize, Vec<usize>)> {
    let mut s = Schema::new(NcFormat::Offset64);
    let td = s.def_dim("time", 0, true)?;
    let yd = s.def_dim("latitude", lat.len(), false)?;
    let xd = s.def_dim("longitude", lon.len(), false)?;
    s.put_global_attr("Conventions", AttrValue::Text("CF-1.8".into()))?;
    s.put_global_attr(
        "source",
        AttrValue::Text(format!("Stateful FWI / NFDRS / KBDI; member {member}")),
    )?;
    s.put_global_attr(
        "nfdrs_configuration",
        AttrValue::Text(serde_json::to_string(nfdrs)?),
    )?;
    let tv = s.def_var("time", NcType::Double, &[td])?;
    s.put_var_attr(
        tv,
        "units",
        AttrValue::Text("seconds since 1970-01-01 00:00:00".into()),
    )?;
    let yv = s.def_var("latitude", NcType::Double, &[yd])?;
    s.put_var_attr(yv, "units", AttrValue::Text("degrees_north".into()))?;
    let xv = s.def_var("longitude", NcType::Double, &[xd])?;
    s.put_var_attr(xv, "units", AttrValue::Text("degrees_east".into()))?;
    let mut vv = vec![];
    for p in PRODUCTS {
        let v = s.def_var(p, NcType::Double, &[td, yd, xd])?;
        s.put_var_attr(
            v,
            "units",
            AttrValue::Text(
                if p.starts_with("nfdrs_mc") || p.starts_with("nfdrs_live") {
                    "%"
                } else {
                    "1"
                }
                .into(),
            ),
        )?;
        vv.push(v);
    }
    let mut w = NcWriter::create(path, s)?;
    w.write_var(yv, VarData::F64(lat))?;
    w.write_var(xv, VarData::F64(lon))?;
    Ok((w, tv, vv))
}
struct PendingOutput(PathBuf);
impl Drop for PendingOutput {
    fn drop(&mut self) {
        if let Ok(m) = fs::metadata(&self.0) {
            if fs::remove_file(&self.0).is_ok() {
                eprintln!(
                    "removed failed run temporary {} ({} bytes)",
                    self.0.display(),
                    m.len()
                );
            }
        }
    }
}
fn replay(path: &Path, r: &Request) -> Result<bool> {
    if !r.receipt.exists() {
        return Ok(false);
    }
    let doc: serde_json::Value = serde_json::from_slice(&fs::read(&r.receipt)?)?;
    let matches = doc["schema"] == "arwen.fire-danger-run.v1"
        && doc["request_sha256"] == hash(path)?
        && doc["output_sha256"] == hash(&r.output)?
        && doc["state_out_sha256"] == hash(&r.state_out)?;
    if !matches {
        return Err(
            "existing run receipt does not match this request/artifact; preserving it".into(),
        );
    }
    if let Some(p) = &r.state_in {
        if doc["state_in_sha256"] != hash(p)? {
            return Err("replay input state changed".into());
        }
    }
    let sources = doc["inputs"].as_array().ok_or("invalid prior receipt")?;
    if sources.len() != r.inputs.len() {
        return Err("replay input count changed".into());
    }
    for (p, source) in r.inputs.iter().zip(sources) {
        if source["sha256"] != hash(p)? {
            return Err("replay forcing changed".into());
        }
    }
    if let Some(p) = &r.annual_rain_grid {
        if doc["annual_rain_grid_sha256"] != hash(p)? {
            return Err("replay rainfall climatology changed".into());
        }
    }
    println!("{}", serde_json::to_string(&doc)?);
    Ok(true)
}
fn regular_axis(a: &[f64]) -> bool {
    a.len() >= 2
        && a.iter().all(|v| v.is_finite())
        && (a[1] - a[0]).abs() > 1e-9
        && a.windows(2)
            .all(|p| ((p[1] - p[0]) - (a[1] - a[0])).abs() < 1e-7)
}
fn run(path: &Path) -> Result<()> {
    let start = Instant::now();
    let r: Request = serde_json::from_slice(&fs::read(path)?)?;
    r.nfdrs.validate()?;
    if !matches!(r.compute_backend.as_str(), "auto" | "cpu" | "cuda") {
        return Err("compute_backend must be auto, cpu or cuda".into());
    }
    if r.inputs.is_empty()
        || r.member.is_empty()
        || !(-12..=14).contains(&r.utc_offset_hours)
        || r.annual_rain_source.trim().is_empty()
    {
        return Err(
            "inputs, member, standard-time offset and sourced mean annual rainfall are required"
                .into(),
        );
    }
    if replay(path, &r)? {
        return Ok(());
    }
    if r.output == r.state_out || r.output == r.receipt || r.state_out == r.receipt {
        return Err("output, state and receipt must be distinct artifacts".into());
    }
    for p in [&r.output, &r.state_out, &r.receipt] {
        if p.exists() {
            return Err(format!("preserving existing {}", p.display()).into());
        }
    }
    let first = netcrust::open(&r.inputs[0])?;
    let lat = first.read_f64("latitude")?;
    let lon = first.read_f64("longitude")?;
    if !regular_axis(&lat)
        || !regular_axis(&lon)
        || lat.iter().any(|v| !v.is_finite() || v.abs() > 90.)
        || lon
            .iter()
            .any(|v| !v.is_finite() || *v < -180. || *v > 180.)
    {
        return Err("invalid geographic axes; longitude must use -180..180".into());
    }
    let nx = lon.len();
    let ny = lat.len();
    let count = nx * ny;
    let first_time = *first.read_f64("time")?.first().ok_or("empty forcing")? as i64;
    let annual = match (r.annual_rain_mm, &r.annual_rain_grid) {
        (Some(v), None) if v.is_finite() && v > 0. => vec![v; count],
        (None, Some(p)) => {
            let n = netcrust::open(p)?;
            if n.read_f64("latitude")? != lat
                || n.read_f64("longitude")? != lon
                || text(&n, "annual_rain_mm", "units")? != "mm"
            {
                return Err("annual rainfall grid/units mismatch".into());
            }
            let v = n.read_f64("annual_rain_mm")?;
            if v.len() != count || v.iter().any(|x| !x.is_finite() || *x <= 0.) {
                return Err(
                    "annual rainfall grid requires positive finite mm at every cell".into(),
                );
            }
            v
        }
        _ => return Err("provide exactly one of annual_rain_mm or annual_rain_grid".into()),
    };
    let parent = r.state_in.as_ref().map(|p| hash(p)).transpose()?;
    let mut st: State = if let Some(p) = &r.state_in {
        serde_json::from_slice(&fs::read(p)?)?
    } else {
        if r.fork_member.is_some() {
            return Err("fork_member requires state_in".into());
        }
        let initial = r.initial_fwi.unwrap_or_default();
        initial.validate()?;
        let deficit = r
            .initial_kbdi
            .ok_or("initial_kbdi must be explicit for a new state")?;
        if !deficit.is_finite() || !(0. ..=800.).contains(&deficit) {
            return Err("initial KBDI must be 0..800".into());
        }
        let t = DateTime::from_timestamp(first_time - 3600, 0).ok_or("invalid first hour")?
            + chrono::Duration::hours(r.utc_offset_hours as i64);
        if t.hour() != 12 || t.minute() != 0 || t.second() != 0 {
            return Err(
                "new state must begin with the first hourly interval after local standard noon"
                    .into(),
            );
        }
        State {
            schema: "arwen.fire-danger-state.v2".into(),
            nfdrs: r.nfdrs.clone(),
            member: r.member.clone(),
            utc_offset_hours: r.utc_offset_hours,
            annual_rain_mm: annual.clone(),
            annual_rain_source: r.annual_rain_source.clone(),
            latitude: lat.clone(),
            longitude: lon.clone(),
            last_hour: first_time - 3600,
            completed_days: 0,
            cells: vec![
                Cell {
                    fwi: initial,
                    nfdrs: nfdrs::State::default(),
                    kbdi: KbdiState {
                        deficit,
                        wet_event_mm: 0.
                    },
                    rain: 0.,
                    tmax: -90.,
                    hours: 0
                };
                count
            ],
            lineage: vec![],
        }
    };
    if st.schema != "arwen.fire-danger-state.v2"
        || st.nfdrs != r.nfdrs
        || st.latitude != lat
        || st.longitude != lon
        || st.cells.len() != count
        || st.utc_offset_hours != r.utc_offset_hours
        || st.annual_rain_mm != annual
        || st.annual_rain_source != r.annual_rain_source
    {
        return Err("state grid/configuration mismatch".into());
    }
    if let Some(p) = r.fork_member {
        if p != st.member || r.member == st.member {
            return Err("fork_member must name the old member and member must be new".into());
        }
        st.member = r.member.clone();
    } else if st.member != r.member {
        return Err("member mismatch; deliberate branching requires fork_member".into());
    }
    if let Some(h) = &parent {
        st.lineage.push(h.clone());
    }
    for c in &st.cells {
        c.nfdrs.validate(&r.nfdrs)?;
        c.fwi.validate()?;
        c.kbdi.validate()?;
        if !c.rain.is_finite() || c.rain < 0. || !c.tmax.is_finite() || c.hours > 23 {
            return Err("invalid pending hourly state".into());
        }
    }
    let local_last = DateTime::from_timestamp(st.last_hour, 0).ok_or("invalid state time")?
        + chrono::Duration::hours(st.utc_offset_hours as i64);
    let pending = (local_last.hour() + 24 - 12) % 24;
    if local_last.minute() != 0
        || local_last.second() != 0
        || st.cells.iter().any(|c| c.hours != pending)
    {
        return Err("pending state hours do not match local clock".into());
    }
    let mut cuda = nelson_cuda::Backend::open(&r.compute_backend)?;
    let days_before = st.completed_days;
    let temporary = r.output.with_file_name(format!(
        "{}.partial",
        r.output
            .file_name()
            .ok_or("output requires a filename")?
            .to_string_lossy()
    ));
    if temporary.exists() {
        return Err("preserving incomplete prior output; choose a new request output".into());
    }
    if let Some(p) = r.output.parent() {
        fs::create_dir_all(p)?;
    }
    let _cleanup = PendingOutput(temporary.clone());
    let (mut out, tv, vv) = writer(&temporary, &lat, &lon, &st.member, &r.nfdrs)?;
    let mut record = 0;
    let mut hours = 0;
    let mut io_wall = 0.;
    let mut compute_wall = 0.;
    let mut sources = vec![];
    let mut summaries = vec![];
    let mut solar_noise_values = 0usize;
    let mut minimum_solar = 0.0_f64;
    for input in &r.inputs {
        let io = Instant::now();
        let n = netcrust::open(input)?;
        if n.read_f64("latitude")? != lat || n.read_f64("longitude")? != lon {
            return Err("forcing grid changed during chain".into());
        }
        if text(&n, "time", "units")? != "seconds since 1970-01-01 00:00:00" {
            return Err("time axis requires seconds since 1970-01-01 00:00:00".into());
        }
        sources.push(json!({"path":input,"sha256":hash(input)?}));
        io_wall += io.elapsed().as_secs_f64();
        for (ti, stamp) in n.read_f64("time")?.into_iter().enumerate() {
            if !stamp.is_finite() || stamp.fract() != 0. || stamp as i64 != st.last_hour + 3600 {
                return Err(format!(
                    "hourly forcing gap/overlap: expected {}, got {stamp}",
                    st.last_hour + 3600
                )
                .into());
            }
            let io = Instant::now();
            let fields: Vec<Vec<f64>> = [
                ("T2", "K"),
                ("D2", "K"),
                ("U10", "m s-1"),
                ("V10", "m s-1"),
                ("TP", "m"),
                ("SSRD", "J m-2"),
            ]
            .par_iter()
            .map(|(name, units)| plane(&n, name, ti, ny, nx, units))
            .collect::<Result<_>>()?;
            let [temp, dew, u, v, rain, solar]: [Vec<f64>; 6] = fields
                .try_into()
                .map_err(|_| "invalid forcing plane count")?;
            io_wall += io.elapsed().as_secs_f64();
            let time = DateTime::<Utc>::from_timestamp(stamp as i64, 0).ok_or("invalid hour")?;
            let local = time + chrono::Duration::hours(st.utc_offset_hours as i64);
            let noon = local.hour() == 12;
            let wall = Instant::now();
            // Validate all forcing before launching the device; no mutation on
            // malformed planes. Radiation is hourly accumulated energy.
            if let Some(i) = (0..count).find(|&i| {
                ![temp[i], dew[i], u[i], v[i], rain[i], solar[i]]
                    .iter()
                    .all(|x| x.is_finite())
                    || !(-90. ..=65.).contains(&(temp[i] - 273.15))
                    || !(-100. ..=65.).contains(&(dew[i] - 273.15))
                    || !(-1e-8..=1.).contains(&rain[i])
                    || u[i].hypot(v[i]) > 150.
                    // ERA5 packed nighttime radiation includes -0.25 J/m2.
                    // Accept at most half a joule of sub-zero packing noise;
                    // both native backends clamp it to zero, with counts below.
                    || !(-0.5..=7_200_000.).contains(&solar[i])
            }) {
                return Err(format!("invalid hourly forcing before Nelson transport: file={} time={time} cell={i} T2={} D2={} U10={} V10={} TP={} SSRD={}",input.display(),temp[i],dew[i],u[i],v[i],rain[i],solar[i]).into());
            }
            for &energy in &solar {
                if energy < 0. {
                    solar_noise_values += 1;
                    minimum_solar = minimum_solar.min(energy);
                }
            }
            cuda.hour(&mut st.cells, &temp, &dew, &solar, &rain)?;
            let rows: std::result::Result<Vec<Option<[f64; 17]>>, String> = st
                .cells
                .par_iter_mut()
                .enumerate()
                .map(|(i, c)| {
                    if ![temp[i], dew[i], u[i], v[i], rain[i]]
                        .iter()
                        .all(|v| v.is_finite())
                        || !(-1e-8..=1.).contains(&rain[i])
                        || u[i].hypot(v[i]) > 150.
                    {
                        return Err(format!("invalid hourly weather at cell {i}"));
                    }
                    let t = temp[i] - 273.15;
                    let d = dew[i] - 273.15;
                    if !(-90. ..=65.).contains(&t) || !(-100. ..=65.).contains(&d) {
                        return Err(format!("invalid temperature at cell {i}"));
                    }
                    let rh = (100.
                        * ((17.625 * d / (243.04 + d)) - (17.625 * t / (243.04 + t))).exp())
                    .clamp(0., 100.);
                    if !cuda.active() {
                        c.nfdrs.hour(
                            t,
                            rh / 100.,
                            solar[i].max(0.) / 3600.,
                            rain[i].max(0.) * 1000.,
                        );
                    }
                    c.rain += rain[i].max(0.) * 1000.;
                    c.tmax = c.tmax.max(t);
                    c.hours += 1;
                    if !noon {
                        return Ok(None);
                    }
                    if c.hours != 24 {
                        return Err(format!(
                            "noon cell {i} has {} hours rather than 24",
                            c.hours
                        ));
                    }
                    let f = c.fwi.step(Weather {
                        temperature: t,
                        humidity: rh,
                        wind: u[i].hypot(v[i]) * 3.6,
                        rain: c.rain,
                        latitude: lat[i / nx],
                        month: local.month(),
                    })?;
                    let k = c.kbdi.step(c.tmax, c.rain, annual[i])?;
                    let nf = c.nfdrs.day(
                        &r.nfdrs,
                        lat[i / nx],
                        local.ordinal(),
                        local.year(),
                        u[i].hypot(v[i]) * 2.2369362921,
                        k,
                        c.rain,
                    );
                    c.rain = 0.;
                    c.tmax = -90.;
                    c.hours = 0;
                    let mut values = [0.; 17];
                    values[..7].copy_from_slice(&[f.ffmc, f.dmc, f.dc, f.isi, f.bui, f.fwi, k]);
                    values[7..].copy_from_slice(&nf);
                    if values.iter().any(|v| !v.is_finite()) {
                        return Err(format!("nonfinite fire index at cell {i}"));
                    }
                    Ok(Some(values))
                })
                .collect();
            let rows = rows?;
            compute_wall += wall.elapsed().as_secs_f64();
            if noon {
                let io = Instant::now();
                out.write_record(record, tv, VarData::F64(&[stamp]))?;
                let mut summary =
                    json!({"valid":time.to_rfc3339(),"local_date":local.date_naive().to_string()});
                for (p, &id) in vv.iter().enumerate() {
                    let values: Vec<f64> = rows.iter().map(|x| x.unwrap()[p]).collect();
                    out.write_record(record, id, VarData::F64(&values))?;
                    summary[PRODUCTS[p]] = json!({"min":values.iter().copied().fold(f64::INFINITY,f64::min),"max":values.iter().copied().fold(f64::NEG_INFINITY,f64::max),"mean":values.iter().sum::<f64>()/count as f64});
                }
                summaries.push(summary);
                io_wall += io.elapsed().as_secs_f64();
                record += 1;
                st.completed_days += 1;
            }
            st.last_hour = stamp as i64;
            hours += 1;
        }
    }
    out.finish()?;
    if record == 0 {
        return Err("request completed no daily indices; state/output were not published".into());
    }
    fs::rename(&temporary, &r.output)?;
    publish(&r.state_out, &serde_json::to_vec(&st)?)?;
    let receipt = json!({"schema":"arwen.fire-danger-run.v1","request":path,"request_sha256":hash(path)?,"member":st.member,"state_in":r.state_in,"state_in_sha256":parent,"state_out":r.state_out,"state_out_sha256":hash(&r.state_out)?,"output":r.output,"output_sha256":hash(&r.output)?,"inputs":sources,"products":PRODUCTS,"grid":[ny,nx],"hours":hours,"days_this_run":record,"days_before":days_before,"completed_days":st.completed_days,"utc_offset_hours":st.utc_offset_hours,"annual_rain_mm_range":[annual.iter().copied().fold(f64::INFINITY,f64::min),annual.iter().copied().fold(f64::NEG_INFINITY,f64::max)],"annual_rain_grid":r.annual_rain_grid,"annual_rain_grid_sha256":r.annual_rain_grid.as_ref().map(|p|hash(p)).transpose()?,"annual_rain_source":st.annual_rain_source,"wall_seconds":start.elapsed().as_secs_f64(),"stage_wall_seconds":{"io":io_wall,"compute":compute_wall},"parallelism":{"compute":rayon::current_num_threads(),"forcing_read":rayon::current_num_threads().min(6),"output_write":1},"peak_vram_mib_allocations":cuda.allocated_mib(count),"backend":cuda.name(),"nfdrs_config":r.nfdrs,"nelson_numerics":{"radial_nodes":21,"substeps_per_hour":60,"scheme":"implicit cylindrical finite volume; positive latent conductance from Nelson eq8; 2016 adsorption/max-local-moisture parameters","radius_m":nfdrs::RADII},"forcing_roundoff":{"solar_negative_values_clamped":solar_noise_values,"minimum_solar_j_m2":minimum_solar,"accepted_floor_j_m2":-0.5},"binary_sha256":hash(&std::env::current_exe()?)?,"source_revision":option_env!("GPUWM_BRIDGE_SOURCE_REV").unwrap_or("unstamped"),"daily":summaries,"limitations":["Nelson independently discretized; not binary parity with the third-party GPL implementation","new states use declared FWI startup codes; multi-week spin-up does not prove DC equilibrium","KBDI follows noon-to-noon maximum temperature and rain"]});
    publish(&r.receipt, &serde_json::to_vec_pretty(&receipt)?)?;
    println!("{}", serde_json::to_string(&receipt)?);
    Ok(())
}
fn main() {
    let a: Vec<_> = std::env::args().collect();
    let result = match a.as_slice() {
        [_, flag] if flag == "--abi" => {
            println!("GPUWM_FIRE_INDICES_V1");
            Ok(())
        }
        [_, flag] if flag == "--version" => {
            println!(concat!(
                "rw_fire_indices 0.1.0 GPUWM_BRIDGE_SOURCE_REV=",
                env!("GPUWM_BRIDGE_SOURCE_REV")
            ));
            Ok(())
        }
        [_, command, path] if command == "compare" => compare::run(Path::new(path)),
        [_, command, path] if command == "verify-nfdrs" => nfdrs_verify::run(Path::new(path)),
        [_, command, path] if command == "verify" => station::run(Path::new(path)),
        [_, command, path] if command == "climatology" => climatology::run(Path::new(path)),
        [_, path] => run(Path::new(path)),
        _ => {
            eprintln!("usage: rw_fire_indices [verify|climatology] REQUEST.json");
            std::process::exit(2)
        }
    };
    if let Err(e) = result {
        eprintln!("fire indices: {e}");
        std::process::exit(1)
    }
}
