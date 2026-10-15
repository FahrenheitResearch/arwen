//! Stored-word MPAS stream comparison. No CF unpacking or precision conversion.
//! This is an instrument, not a port of a model routine.
//! `--request JSON --out JSON`; Python owns decks, execution and ratchet policy.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};
use std::path::{Path, PathBuf};
use std::io::Read;

type Result<T> = std::result::Result<T, String>;

// netCDF's unconverted typed storage API preserves float payloads and signed
// zero. nc_get_vara returns stored values; it never applies CF scale/offset.
// The library is loaded at run time (see netcdf_c); these keep its C names.
unsafe fn nc_open(path: *const c_char, mode: c_int, id: *mut c_int) -> c_int {
    unsafe { (crate::netcdf_c::api().nc_open)(path, mode, id) }
}
unsafe fn nc_close(id: c_int) -> c_int { unsafe { (crate::netcdf_c::api().nc_close)(id) } }
unsafe fn nc_strerror(code: c_int) -> *const c_char {
    unsafe { (crate::netcdf_c::api().nc_strerror)(code) }
}
unsafe fn nc_inq_nvars(id: c_int, n: *mut c_int) -> c_int {
    unsafe { (crate::netcdf_c::api().nc_inq_nvars)(id, n) }
}
unsafe fn nc_inq_format(id: c_int, format: *mut c_int) -> c_int {
    unsafe { (crate::netcdf_c::api().nc_inq_format)(id, format) }
}
unsafe fn nc_inq_grps(id: c_int, n: *mut c_int, ids: *mut c_int) -> c_int {
    unsafe { (crate::netcdf_c::api().nc_inq_grps)(id, n, ids) }
}
unsafe fn nc_inq_var(id: c_int, var: c_int, name: *mut c_char, kind: *mut c_int,
                     ndims: *mut c_int, dims: *mut c_int, atts: *mut c_int) -> c_int {
    unsafe { (crate::netcdf_c::api().nc_inq_var)(id, var, name, kind, ndims, dims, atts) }
}
unsafe fn nc_inq_dim(id: c_int, dim: c_int, name: *mut c_char, len: *mut usize) -> c_int {
    unsafe { (crate::netcdf_c::api().nc_inq_dim)(id, dim, name, len) }
}
unsafe fn nc_get_vara(id: c_int, var: c_int, start: *const usize, count: *const usize,
                      data: *mut c_void) -> c_int {
    unsafe { (crate::netcdf_c::api().nc_get_vara)(id, var, start, count, data) }
}

fn nc(code: c_int) -> Result<()> {
    if code == 0 { Ok(()) } else {
        Err(unsafe { CStr::from_ptr(nc_strerror(code)) }.to_string_lossy().into_owned())
    }
}

struct File(c_int);
impl Drop for File { fn drop(&mut self) { unsafe { nc_close(self.0); } } }

#[derive(Clone, Debug, PartialEq)]
struct Variable { id: c_int, name: String, kind: c_int, dims: Vec<(String, usize)> }
impl Variable {
    fn time_axis(&self) -> Option<usize> { self.dims.iter().position(|d| d.0 == "Time") }
    fn width(&self) -> Result<usize> {
        match self.kind { 1|2|7 => Ok(1), 3|8 => Ok(2), 4|5|9 => Ok(4), 6|10|11 => Ok(8),
            _ => Err(format!("unsupported storage type {} for {}", self.kind, self.name)) }
    }
}

impl File {
    fn open(path: &Path) -> Result<Self> {
        crate::netcdf_c::load()?;
        let name = CString::new(path.as_os_str().as_encoded_bytes()).map_err(|e| e.to_string())?;
        let mut id = 0;
        nc(unsafe { nc_open(name.as_ptr(), 0, &mut id) })
            .map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(Self(id))
    }
    fn variables(&self) -> Result<BTreeMap<String, Variable>> {
        let mut format = 0;
        nc(unsafe { nc_inq_format(self.0, &mut format) })?;
        if format == 3 || format == 4 {
            let mut groups = 0;
            nc(unsafe { nc_inq_grps(self.0, &mut groups, std::ptr::null_mut()) })?;
            if groups != 0 { return Err("grouped netCDF streams unsupported; refusing an incomplete field inventory".into()); }
        }
        let mut n = 0;
        nc(unsafe { nc_inq_nvars(self.0, &mut n) })?;
        let mut out = BTreeMap::new();
        for id in 0..n {
            let mut name = [0 as c_char; 257];
            let (mut kind, mut rank, mut atts) = (0, 0, 0);
            nc(unsafe { nc_inq_var(self.0, id, name.as_mut_ptr(), &mut kind, &mut rank,
                                  std::ptr::null_mut(), &mut atts) })?;
            let mut dimids = vec![0; rank as usize];
            nc(unsafe { nc_inq_var(self.0, id, name.as_mut_ptr(), &mut kind, &mut rank,
                                  dimids.as_mut_ptr(), &mut atts) })?;
            let mut dims = Vec::new();
            for dim in dimids {
                let mut dname = [0 as c_char; 257];
                let mut len = 0;
                nc(unsafe { nc_inq_dim(self.0, dim, dname.as_mut_ptr(), &mut len) })?;
                dims.push((unsafe { CStr::from_ptr(dname.as_ptr()) }.to_string_lossy().into_owned(), len));
            }
            let name = unsafe { CStr::from_ptr(name.as_ptr()) }.to_string_lossy().into_owned();
            out.insert(name.clone(), Variable { id, name, kind, dims });
        }
        Ok(out)
    }
    fn read(&self, var: &Variable, record: usize) -> Result<Vec<u8>> {
        let mut start = vec![0; var.dims.len()];
        let mut count: Vec<usize> = var.dims.iter().map(|d| d.1).collect();
        if let Some(axis) = var.time_axis() { start[axis] = record; count[axis] = 1; }
        let elements = count.iter().try_fold(1usize, |a, b| a.checked_mul(*b))
            .ok_or("variable shape overflow")?;
        let bytes = elements.checked_mul(var.width()?).ok_or("variable size overflow")?;
        // A u64 allocation guarantees sufficient alignment for every primitive
        // netCDF type. Never hand the C library an under-aligned Vec<u8>.
        let mut storage = vec![0u64; bytes.div_ceil(8).max(1)];
        nc(unsafe { nc_get_vara(self.0, var.id, start.as_ptr(), count.as_ptr(),
                               storage.as_mut_ptr().cast()) })?;
        Ok(unsafe { std::slice::from_raw_parts(storage.as_ptr().cast::<u8>(), bytes) }.to_vec())
    }
    fn numbers(&self, vars: &BTreeMap<String, Variable>, field: &str, record: usize) -> Result<Vec<f64>> {
        let var = vars.get(field).ok_or_else(|| format!("diagnostic field missing: {field}"))?;
        let bytes = self.read(var, record)?;
        let values = match var.kind {
            5 => bytes.chunks_exact(4).map(|b| f32::from_ne_bytes(b.try_into().unwrap()) as f64).collect(),
            6 => bytes.chunks_exact(8).map(|b| f64::from_ne_bytes(b.try_into().unwrap())).collect(),
            _ => return Err(format!("diagnostic {field} is not floating point")),
        };
        Ok(values)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    schema: String,
    samples: Vec<Sample>,
    #[serde(default)] diagnostics: Option<Diagnostics>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Sample {
    stream: String,
    steps: Vec<u64>,
    times: Vec<String>,
    strict: PathBuf,
    hex: PathBuf,
    #[serde(default)] optimized: Option<PathBuf>,
    #[serde(default)] clock_witness: Option<StreamClock>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StreamClock { filename_template: String }

fn record_count(vars: &BTreeMap<String, Variable>) -> Result<usize> {
    if vars.is_empty() { return Err("native stream contains no variables".into()); }
    let records = vars.values().filter_map(|v| v.time_axis().map(|i| v.dims[i].1)).max().unwrap_or(1);
    if records == 0 { return Err("native stream has no output records".into()); }
    for var in vars.values() {
        var.width()?;
        if var.time_axis().is_some_and(|i| var.dims[i].1 != records) {
            return Err("native fields disagree on output record count".into());
        }
    }
    Ok(records)
}

fn validate_clock(file: &File, vars: &BTreeMap<String, Variable>, path: &Path,
                  sample: &Sample, record: usize) -> Result<()> {
    if let Some(xtime) = vars.get("xtime") {
        if xtime.kind != 2 || xtime.time_axis().is_none() { return Err("xtime must be character records over Time".into()); }
        let stamp = String::from_utf8(file.read(xtime, record)?).map_err(|e|e.to_string())?;
        if stamp.trim_end_matches(['\0', ' ']) != sample.times[record] {
            return Err(format!("{}: xtime differs from deck at step {}", sample.stream, sample.steps[record]));
        }
    } else {
        let witness = sample.clock_witness.as_ref().ok_or("native stream lacks xtime and a sealed stream-filename clock witness")?;
        let stamp = sample.times[0].as_bytes();
        if stamp.len()!=19 || !stamp.is_ascii() { return Err("invalid stream filename timestamp".into()); }
        let mut filename = witness.filename_template.clone();
        for (token, range) in [("$Y",0..4),("$M",5..7),("$D",8..10),("$h",11..13),("$m",14..16),("$s",17..19)] {
            if !filename.contains(token) { return Err("stream clock template must contain the complete timestamp".into()); }
            filename = filename.replace(token, std::str::from_utf8(&stamp[range]).map_err(|e|e.to_string())?);
        }
        if !path.ends_with(&filename) { return Err("stream filename differs from sealed clock witness".into()); }
        if record_count(vars)? != sample.times.len() { return Err("stream clock record count differs".into()); }
    }
    Ok(())
}

// The deck binds native names, never case-specific aliases in the comparator.
// These fields must share the sample stream; absent diagnostics refuse spread.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Diagnostics {
    stream: String,
    area: PathBuf,
    area_field: String,
    pressure: String,
    height: String,
    eastward_wind: String,
    northward_wind: String,
    t2m: String,
    q2: String,
    precipitation: Vec<String>,
    prognostics: Vec<String>,
    water: Water,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Water {
    // Domain-storage and cumulative source/sink *per-cell* kg m^-2 fields.
    // Deck producers must emit native integrals (including land/snow/canopy
    // storage where active). Python may not synthesize or decode these arrays.
    storage: Vec<String>,
    cumulative_sources: Vec<String>,
    cumulative_sinks: Vec<String>,
}

#[derive(Serialize, Debug)]
struct Row {
    stream: String, field: String, step: u64, time: String, status: String,
    compared_words: u64, differing_words: u64, max_ulp: Option<u64>,
    first_differing_index: Option<usize>,
}

fn ordered32(bits: u32) -> u32 { if bits >> 31 != 0 { !bits } else { bits | 0x8000_0000 } }
fn ordered64(bits: u64) -> u64 { if bits >> 63 != 0 { !bits } else { bits | 0x8000_0000_0000_0000 } }

fn compare_bytes(a: &[u8], b: &[u8], kind: c_int) -> (u64, Option<u64>, Option<usize>) {
    // Count every 32-bit storage word, including both halves of doubles and
    // packed integer/text bytes. A trailing partial word also participates.
    let unit = 4;
    let mut words = 0;
    let mut first = None;
    for (i, (x, y)) in a.chunks(unit).zip(b.chunks(unit)).enumerate() {
        if x != y { words += 1; first.get_or_insert(i); }
    }
    let ulp = match kind {
        5 => Some(a.chunks_exact(4).zip(b.chunks_exact(4)).map(|(x,y)| {
            let x = u32::from_ne_bytes(x.try_into().unwrap());
            let y = u32::from_ne_bytes(y.try_into().unwrap());
            ordered32(x).abs_diff(ordered32(y)) as u64
        }).max().unwrap_or(0)),
        6 => Some(a.chunks_exact(8).zip(b.chunks_exact(8)).map(|(x,y)| {
            let x = u64::from_ne_bytes(x.try_into().unwrap());
            let y = u64::from_ne_bytes(y.try_into().unwrap());
            ordered64(x).abs_diff(ordered64(y))
        }).max().unwrap_or(0)),
        _ => None,
    };
    (words, ulp, first)
}

fn dimensions(var: &Variable) -> Vec<(String, usize)> {
    var.dims.iter().filter(|d| d.0 != "Time").cloned().collect()
}

fn compare_sample(sample: &Sample, rows: &mut Vec<Row>) -> Result<()> {
    let strict = File::open(&sample.strict)?;
    let sv = strict.variables()?;
    if sv.is_empty() { return Err("native stream contains no variables".into()); }
    // A killed writer can leave a corrupt container. The native field/time
    // inventory still determines every missing word; candidate decode failure
    // is a counted regression, never an empty comparison or ignored stream.
    let candidate = if sample.hex.is_file() {
        Some(File::open(&sample.hex).and_then(|file| {
            let vars = file.variables()?; Ok((file, vars))
        }))
    } else { None };
    let (hex, hv, hex_error) = match candidate {
        Some(Ok((file, vars))) => (Some(file), vars, None),
        Some(Err(error)) => (None, BTreeMap::new(), Some(error)),
        None => (None, BTreeMap::new(), None),
    };
    let records = record_count(&sv)?;
    if records == 0 || records != sample.steps.len() || records != sample.times.len() {
        return Err(format!("{}: native records do not match the deck's steps/times", sample.stream));
    }
    // An unlisted candidate time or field is a schema regression too.
    for v in hv.values() {
        if !sv.contains_key(&v.name) || v.time_axis().is_some_and(|i| v.dims[i].1 > records) {
            rows.push(Row { stream: sample.stream.clone(), field: v.name.clone(), step: sample.steps[0],
                time: sample.times[0].clone(), status: "extra field or output time in hex".into(),
                compared_words: 0, differing_words: 1, max_ulp: None, first_differing_index: None });
        }
    }
    for (r, step) in sample.steps.iter().enumerate() {
        // Validate valid times against actual native tape, independently of
        // hex. No manifest can relabel a cached reference's samples.
        validate_clock(&strict, &sv, &sample.strict, sample, r)?;
        for var in sv.values() {
            if var.time_axis().is_none() && r != 0 { continue; }
            let a = strict.read(var, r)?;
            let mut row = Row { stream: sample.stream.clone(), field: var.name.clone(), step: *step,
                time: sample.times[r].clone(), status: "compared".into(),
                compared_words: a.len().div_ceil(4) as u64,
                differing_words: 0, max_ulp: None, first_differing_index: None };
            match (hex.as_ref(), hv.get(&var.name)) {
                (Some(hex), Some(other)) if var.kind == other.kind && dimensions(var) == dimensions(other) => {
                    if other.time_axis().is_some_and(|i| r >= other.dims[i].1) || var.time_axis() != other.time_axis() {
                        row.status = "output time missing in hex".into();
                        row.differing_words = row.compared_words.max(1);
                    } else {
                        match hex.read(other, r) {
                            Ok(b) if a.len() == b.len() => {
                                (row.differing_words, row.max_ulp, row.first_differing_index) = compare_bytes(&a, &b, var.kind);
                            },
                            result => {
                                row.status = match result {
                                    Err(error) => format!("field unreadable in hex: {error}"),
                                    _ => "stored shape changed in hex during read".into(),
                                };
                                row.differing_words = row.compared_words.max(1);
                            },
                        }
                    }
                },
                (Some(_), Some(_)) => { row.status = "type or dimensions differ in hex".into(); row.differing_words = row.compared_words.max(1); },
                _ => {
                    row.status = hex_error.as_ref().map(|error| format!("stream unreadable in hex: {error}"))
                        .unwrap_or_else(|| "field missing in hex".into());
                    row.differing_words = row.compared_words.max(1);
                },
            }
            rows.push(row);
        }
    }
    Ok(())
}

// Shared log-pressure interpolation, downward unique crossing, common mask
// for all three arms. Like forecast_divergence, no extrapolation is allowed.
fn pressure_surface(p: &[f64], v: &[f64], levels: usize, cells: usize, target: f64) -> Result<Vec<f64>> {
    if levels < 2 || p.len() != levels*cells || v.len() != p.len() { return Err("invalid pressure-surface dimensions".into()); }
    let mut out = vec![f64::NAN; cells];
    for c in 0..cells {
        let mut crossing = None;
        let mut count = 0;
        let mut invalid = false;
        for k in 0..levels {
            let i = c*levels+k;
            if !p[i].is_finite() || p[i] <= 0.0 || !v[i].is_finite() { invalid = true; }
            if k+1 < levels {
                if p[i] >= target && p[i+1] < target { crossing = Some(i); count += 1; }
                if p[i] < target && p[i+1] >= target { invalid = true; }
            }
        }
        let top = c*levels+levels-1;
        if !invalid && p[top] == target && count == 0 { out[c] = v[top]; }
        else if !invalid && count == 1 {
            let i = crossing.unwrap();
            let w = (target.ln()-p[i].ln())/(p[i+1].ln()-p[i].ln());
            out[c] = v[i]*(1.0-w)+v[i+1]*w;
        }
    }
    Ok(out)
}

fn rms_three(arms: &[Vec<f64>;3], area: &[f64]) -> Result<(f64,f64,usize)> {
    if arms.iter().any(|v| v.len() != area.len()) { return Err("RMS shape differs from cell area".into()); }
    let (mut hs,mut so,mut weights,mut valid) = (0.0,0.0,0.0,0);
    for i in 0..area.len() {
        // The native pair defines the measurement mask. A candidate cannot
        // hide its own NaNs or lost pressure surface by shrinking that mask.
        if arms[0][i].is_finite() && arms[2][i].is_finite() {
            if !arms[1][i].is_finite() { return Err("hex diagnostic nonfinite inside native comparison mask".into()); }
            hs += area[i]*(arms[1][i]-arms[0][i]).powi(2);
            so += area[i]*(arms[2][i]-arms[0][i]).powi(2);
            weights += area[i]; valid += 1;
        }
    }
    if valid == 0 || weights <= 0.0 { return Err("diagnostic common mask is empty".into()); }
    Ok(((hs/weights).sqrt(),(so/weights).sqrt(),valid))
}

fn l2_three(arms: &[Vec<f64>;3]) -> Result<(f64,f64)> {
    if arms[0].is_empty() || arms.iter().any(|v| v.len()!=arms[0].len() || v.iter().any(|x| !x.is_finite())) {
        return Err("prognostic missing, nonfinite or shape differs".into());
    }
    let denom: f64 = arms[0].iter().map(|x| x*x).sum();
    let hs: f64 = arms[0].iter().zip(&arms[1]).map(|(a,b)| (a-b).powi(2)).sum();
    let so: f64 = arms[0].iter().zip(&arms[2]).map(|(a,b)| (a-b).powi(2)).sum();
    let relative = |v: f64| if denom == 0.0 { if v == 0.0 { 0.0 } else { f64::INFINITY } } else { (v/denom).sqrt() };
    Ok((relative(hs),relative(so)))
}

fn metric(name: &str, actual: f64, spread: f64, cells: usize) -> Value {
    json!({"metric":name,"hex_vs_strict":actual,"strict_vs_optimized":spread,
           "valid_cells":cells,"passed":actual.is_finite() && spread.is_finite() && actual <= spread})
}

fn cell_levels(file: &File, vars: &BTreeMap<String,Variable>, field: &str, record: usize, cells: usize) -> Result<(Vec<f64>,usize)> {
    let var = vars.get(field).ok_or_else(|| format!("diagnostic field missing: {field}"))?;
    let dims = dimensions(var);
    let values = file.numbers(vars,field,record)?;
    if dims.len()!=2 { return Err(format!("{field}: expected cell/level dimensions")); }
    let cell = dims.iter().position(|d| d.0=="nCells").ok_or("no nCells dimension")?;
    if dims[cell].1 != cells { return Err("cell count disagrees with area".into()); }
    let levels = dims[1-cell].1;
    if cell==0 { Ok((values,levels)) } else {
        let mut out = vec![0.0;values.len()];
        for c in 0..cells { for k in 0..levels { out[c*levels+k]=values[k*cells+c]; } }
        Ok((out,levels))
    }
}

fn sum_fields(file: &File, vars: &BTreeMap<String,Variable>, fields: &[String], r: usize, n: usize) -> Result<Vec<f64>> {
    if fields.is_empty() { return Err("required diagnostic field list is empty".into()); }
    let mut total = vec![0.0;n];
    for name in fields {
        let v = file.numbers(vars,name,r)?;
        if v.len()!=n || v.iter().any(|x| !x.is_finite()) { return Err(format!("{name}: invalid cell diagnostic")); }
        for (a,b) in total.iter_mut().zip(v) { *a += b; }
    }
    Ok(total)
}

fn spread(request: &Request) -> Result<Vec<Value>> {
    let d = request.diagnostics.as_ref().ok_or("spread diagnostics are not bound by the deck")?;
    if d.prognostics.is_empty() { return Err("prognostic inventory is empty".into()); }
    let area_file = File::open(&d.area)?;
    let area = area_file.numbers(&area_file.variables()?,&d.area_field,0)?;
    if area.is_empty() || area.iter().any(|x| !x.is_finite() || *x<=0.0) { return Err("invalid cell area".into()); }
    let mut rows = Vec::new();
    let mut initial_water: Option<[f64;3]> = None;
    let samples: Vec<_> = request.samples.iter().filter(|s| s.stream==d.stream).collect();
    if samples.is_empty() { return Err("diagnostic stream is absent".into()); }
    for sample in samples {
        let files = [File::open(&sample.strict)?, File::open(&sample.hex)?,
                     File::open(sample.optimized.as_ref().ok_or("optimized native reference absent")?)?];
        let vars = [files[0].variables()?,files[1].variables()?,files[2].variables()?];
        for (r,step) in sample.steps.iter().enumerate() {
            for i in 0..3 {
                let path = match i { 0 => &sample.strict, 1 => &sample.hex,
                                     _ => sample.optimized.as_ref().unwrap() };
                validate_clock(&files[i], &vars[i], path, sample, r)?;
            }
            let mut metrics = Vec::new();
            for name in [&d.t2m,&d.q2] {
                let a = [files[0].numbers(&vars[0],name,r)?, files[1].numbers(&vars[1],name,r)?,files[2].numbers(&vars[2],name,r)?];
                if a.iter().any(|v| v.iter().any(|x| !x.is_finite())) { return Err(format!("nonfinite {name}")); }
                let (hs,so,n)=rms_three(&a,&area)?; metrics.push(metric(name,hs,so,n));
            }
            let a = [sum_fields(&files[0],&vars[0],&d.precipitation,r,area.len())?,sum_fields(&files[1],&vars[1],&d.precipitation,r,area.len())?,sum_fields(&files[2],&vars[2],&d.precipitation,r,area.len())?];
            let (hs,so,n)=rms_three(&a,&area)?; metrics.push(metric("accumulated_precipitation",hs,so,n));
            let mut z: [Vec<f64>;3] = Default::default();
            let mut u: [Vec<f64>;3] = Default::default();
            let mut v: [Vec<f64>;3] = Default::default();
            for i in 0..3 {
                let (p,lev)=cell_levels(&files[i],&vars[i],&d.pressure,r,area.len())?;
                let (h,hl)=cell_levels(&files[i],&vars[i],&d.height,r,area.len())?;
                let (east,el)=cell_levels(&files[i],&vars[i],&d.eastward_wind,r,area.len())?;
                let (north,nl)=cell_levels(&files[i],&vars[i],&d.northward_wind,r,area.len())?;
                if [hl,el,nl].iter().any(|l| *l!=lev) { return Err("pressure/height/wind levels disagree".into()); }
                z[i]=pressure_surface(&p,&h,lev,area.len(),50000.0)?;
                u[i]=pressure_surface(&p,&east,lev,area.len(),25000.0)?;
                v[i]=pressure_surface(&p,&north,lev,area.len(),25000.0)?;
            }
            let (hs,so,n)=rms_three(&z,&area)?; metrics.push(metric("z500",hs,so,n));
            // Both native components define one vector mask. Invalid hex
            // winds at a valid native surface refuse before any masking.
            for c in 0..area.len() {
                let native_valid = [0,2].iter().all(|i| u[*i][c].is_finite() && v[*i][c].is_finite());
                if native_valid && !(u[1][c].is_finite() && v[1][c].is_finite()) {
                    return Err("hex wind250 nonfinite inside native vector mask".into());
                }
                if !native_valid {
                for i in 0..3 { u[i][c]=f64::NAN; v[i][c]=f64::NAN; }
            } }
            let (uh,us,n)=rms_three(&u,&area)?; let (vh,vs,_)=rms_three(&v,&area)?;
            metrics.push(metric("wind250",uh.hypot(vh),us.hypot(vs),n));
            for name in &d.prognostics {
                let a = [files[0].numbers(&vars[0],name,r)?,files[1].numbers(&vars[1],name,r)?,files[2].numbers(&vars[2],name,r)?];
                let (hs,so)=l2_three(&a)?; metrics.push(metric(&format!("relative_l2:{name}"),hs,so,a[0].len()));
            }
            let mut water = [0.0;3];
            for i in 0..3 {
                let storage=sum_fields(&files[i],&vars[i],&d.water.storage,r,area.len())?;
                let sources=sum_fields(&files[i],&vars[i],&d.water.cumulative_sources,r,area.len())?;
                let sinks=sum_fields(&files[i],&vars[i],&d.water.cumulative_sinks,r,area.len())?;
                water[i]=area.iter().enumerate().map(|(c,a)| a*(storage[c]-sources[c]+sinks[c])).sum();
                if !water[i].is_finite() { return Err("water integral is nonfinite".into()); }
            }
            if initial_water.is_none() {
                if *step!=0 { return Err("water budget requires time-zero sample".into()); }
                initial_water=Some(water);
            }
            let initial=initial_water.unwrap();
            let closure=[(water[0]-initial[0]).abs(),(water[1]-initial[1]).abs(),(water[2]-initial[2]).abs()];
            let water_pass=closure[1] <= closure[0].max(closure[2]);
            rows.push(json!({"stream":sample.stream,"step":step,"time":sample.times[r],"metrics":metrics,
                "water_budget":{"absolute_closure_kg":closure,"passed":water_pass},
                "passed":water_pass && metrics.iter().all(|m| m["passed"]==true)}));
        }
    }
    Ok(rows)
}

fn run(request: &Request) -> Result<Value> {
    if request.schema!="mpas-stream-compare-v1" || request.samples.is_empty() { return Err("empty or unsupported comparison request".into()); }
    let mut rows = Vec::new();
    for sample in &request.samples { compare_sample(sample,&mut rows)?; }
    let (spread,spread_error)=match spread(request) { Ok(v)=>(v,None),Err(e)=>(Vec::new(),Some(e)) };
    Ok(json!({"schema":"mpas-stream-compare-v1","fields":rows,"spread_samples":spread,
              "spread_error":spread_error,"reader":"netCDF stored-type nc_get_vara; CF unpacking disabled"}))
}

fn inventory(root: &Path) -> Result<Vec<PathBuf>> {
    // Inspect container signatures in Rust; MPAS stream names need not end
    // in .nc. Text logs/decks are not field streams. Never follow directory
    // symlinks recursively, which could escape a run or form a cycle.
    let mut paths = Vec::new();
    if !root.exists() { return Ok(paths); }
    for entry in std::fs::read_dir(root).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        let kind = entry.file_type().map_err(|e| e.to_string())?;
        if kind.is_dir() { paths.extend(inventory(&path)?); }
        else if kind.is_file() || kind.is_symlink() && path.is_file() {
            let mut signature = [0u8; 8];
            let n = std::fs::File::open(&path).map_err(|e| e.to_string())?
                .read(&mut signature).map_err(|e| e.to_string())?;
            let classic = n >= 4 && &signature[..3] == b"CDF" && [1,2,5].contains(&signature[3]);
            let hdf = n == 8 && signature == *b"\x89HDF\r\n\x1a\n";
            if classic || hdf {
                paths.push(path);
            }
        } else if kind.is_symlink() && path.is_dir() {
            return Err(format!("output directory symlink cannot certify a complete inventory: {}", path.display()));
        }
    }
    paths.sort();
    Ok(paths)
}

fn describe(path: &Path) -> Result<Value> {
    let file = File::open(path)?;
    let vars = file.variables()?;
    let records = record_count(&vars)?;
    let Some(xtime) = vars.get("xtime") else {
        return Ok(json!({"times":null,"records":records,"time_source":"STREAM-DECLARATION-REQUIRED",
                         "fields":vars.keys().collect::<Vec<_>>()}));
    };
    if xtime.kind != 2 { return Err("xtime must use character storage".into()); }
    let axis = xtime.time_axis().ok_or("xtime has no Time dimension")?;
    let records = xtime.dims[axis].1;
    if records == 0 { return Err("native stream has no output records".into()); }
    for var in vars.values() {
        var.width()?;
        if var.time_axis().is_some_and(|i| var.dims[i].1 != records) {
            return Err("native fields disagree on output record count".into());
        }
    }
    let times: Vec<String> = (0..records).map(|r| {
        String::from_utf8(file.read(xtime, r)?).map_err(|e| e.to_string())
            .map(|s| s.trim_end_matches(['\0', ' ']).to_owned())
    }).collect::<Result<_>>()?;
    Ok(json!({"times": times, "records":records,"time_source":"xtime", "fields": vars.keys().collect::<Vec<_>>()}))
}

/// `rw_mpas_static verify ...`, or this executable invoked as `rw_mpas_verify`.
pub fn main_with(rest: Vec<std::ffi::OsString>) -> std::process::ExitCode {
    let Ok(rest) = crate::utf8_arguments("rw_mpas_verify", rest) else { return std::process::ExitCode::from(2) };
    // Index 0 stays the program, so the vector reads as it always did.
    let args: Vec<String> = std::iter::once("rw_mpas_verify".to_string()).chain(rest).collect();
    if args.len()==2 && args[1]=="--abi" { println!("gpuwm-rw-mpas-verify-v1"); return std::process::ExitCode::SUCCESS; }
    let outcome = (|| -> Result<()> {
        if args.len()==3 && args[1]=="--inventory-root" {
            println!("{}", serde_json::to_string(&inventory(Path::new(&args[2]))?).map_err(|e|e.to_string())?);
            return Ok(());
        }
        if args.len()==3 && args[1]=="--describe" {
            println!("{}", describe(Path::new(&args[2]))?);
            return Ok(());
        }
        if args.len()!=5 || args[1]!="--request" || args[3]!="--out" { return Err("usage: rw_mpas_verify --request JSON --out JSON".into()); }
        let request: Request=serde_json::from_slice(&std::fs::read(&args[2]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
        let receipt=run(&request)?;
        std::fs::write(&args[4],serde_json::to_vec_pretty(&receipt).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
        Ok(())
    })();
    if let Err(e)=outcome { eprintln!("rw_mpas_verify: {e}"); return std::process::ExitCode::from(2); }
    std::process::ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn words_and_ulp_preserve_zero_nan_and_double_halves() {
        assert_eq!(compare_bytes(&1f32.to_ne_bytes(),&f32::from_bits(1f32.to_bits()+1).to_ne_bytes(),5),(1,Some(1),Some(0)));
        assert_eq!(compare_bytes(&(-0f32).to_ne_bytes(),&0f32.to_ne_bytes(),5).0,1);
        assert_eq!(compare_bytes(&f32::from_bits(0x7fc00001).to_ne_bytes(),&f32::from_bits(0x7fc00002).to_ne_bytes(),5).0,1);
        let a=1f64.to_ne_bytes(); let b=f64::from_bits(1f64.to_bits()+1).to_ne_bytes();
        assert_eq!(compare_bytes(&a,&b,6).0,1);
        assert_eq!(compare_bytes(&[0,0,0,0,0,0,0,0],&[1,0,0,0,1,0,0,0],6).0,2);
        assert_eq!(compare_bytes(&[0,0,0,0,0],&[1,1,1,1,1],2).0,2);
        assert_eq!(compare_bytes(&[0,0,0,0],&[1,1,1,1],3).0,1);
    }
    #[test] fn surfaces_do_not_extrapolate_or_accept_ambiguous_crossings() {
        let a=pressure_surface(&[100000.,50000.,20000.],&[0.,5.,10.],3,1,50000.).unwrap();
        assert_eq!(a,vec![5.]);
        assert!(pressure_surface(&[10000.,5000.],&[1.,2.],2,1,25000.).unwrap()[0].is_nan());
        assert!(pressure_surface(&[100000.,20000.,80000.,10000.],&[1.,2.,3.,4.],4,1,50000.).unwrap()[0].is_nan());
    }
    #[test] fn spread_uses_common_mask_and_no_added_tolerance() {
        assert_eq!(rms_three(&[vec![0.,f64::NAN],vec![1.,9.],vec![2.,0.]],&[1.,3.]).unwrap(),(1.,2.,1));
        assert!(rms_three(&[vec![0.,0.],vec![0.,f64::NAN],vec![0.,0.]],&[1.,3.]).is_err());
        assert!(!metric("x",f64::from_bits(1f64.to_bits()+1),1.,1)["passed"].as_bool().unwrap());
        assert!(l2_three(&[vec![0.],vec![1.],vec![0.]]).unwrap().0.is_infinite());
    }
}
