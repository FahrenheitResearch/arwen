//! woof-globe/1: NetCDF reading, field derivation and packing stay in Rust.
use std::{fs, io::{self, Read}, path::{Path, PathBuf}};
use base64::{Engine, engine::general_purpose::STANDARD};
use chrono::{DateTime, NaiveDateTime, Utc};
use serde::{Deserialize, Serialize, ser::{SerializeMap, Serializer}};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

type Result<T> = std::result::Result<T, String>;
const FORMAT: &str = "woof-globe/1";
pub static GPUWM_BRIDGE_SOURCE_REV_STAMP: &str =
    concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));

#[derive(Deserialize)]
struct Request {
    action: String,
    #[serde(default)] tape: PathBuf,
    #[serde(default)] out: PathBuf,
    #[serde(default)] id: String,
    #[serde(default)] init: String,
    previous: Option<PathBuf>,
    #[serde(default)] path: PathBuf,
    key: Option<String>,
    dest: Option<PathBuf>,
    truncation: Option<usize>,
}

struct Tape { file: netcrust::File, path: PathBuf }
struct Array { shape: Vec<usize>, values: Vec<f64> }
impl Tape {
    fn open(path: &Path) -> Result<Self> {
        Ok(Self { file: netcrust::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?, path: path.into() })
    }
    fn has(&self, name: &str) -> bool { self.file.variable(name).is_some() }
    fn get(&self, name: &str) -> Result<Array> {
        let variable = self.file.variable(name).ok_or_else(|| format!("{} has no {name}, which the globe pack needs", self.path.display()))?;
        let array = self.file.read_array_f64(name).map_err(|e| format!("{name}: {e}"))?;
        let mut shape = array.shape().to_vec();
        let mut values = array.into_values();
        if variable.dimensions().first().is_some_and(|d| d.name().eq_ignore_ascii_case("time")) {
            shape.remove(0);
            values.truncate(shape.iter().product());
        }
        // netCDF4's set_auto_mask(False) keeps sentinels but still unpacks CF.
        let scale = variable.attribute("scale_factor").and_then(|a| a.as_f64()).unwrap_or(1.0);
        let add = variable.attribute("add_offset").and_then(|a| a.as_f64()).unwrap_or(0.0);
        if scale != 1.0 || add != 0.0 { for v in &mut values { *v = *v * scale + add; } }
        Ok(Array { shape, values })
    }
    fn valid(&self) -> Result<DateTime<Utc>> {
        let name = self.path.file_name().unwrap_or_default().to_string_lossy();
        if let Some(t) = parse_tape_time(&name) { return Ok(t); }
        if let Ok(strings) = self.file.read_strings("Times") {
            if let Some(t) = parse_tape_time(&strings.concat()) { return Ok(t); }
        }
        Err(format!("{}: no valid time in its name or its Times", self.path.display()))
    }
    fn rain(&self, size: usize) -> Result<Vec<f64>> {
        let mut a = self.plane("RAINNC", size)?;
        if self.has("RAINC") { for (a, b) in a.iter_mut().zip(self.plane("RAINC", size)?) { *a += b; } }
        Ok(a)
    }
    fn plane(&self, name: &str, size: usize) -> Result<Vec<f64>> {
        let a = self.get(name)?;
        if a.shape.len() != 2 || a.values.len() != size { return Err(format!("{name}: expected a {size}-cell surface plane, got {:?}", a.shape)); }
        Ok(a.values)
    }
}

fn parse_tape_time(text: &str) -> Option<DateTime<Utc>> {
    for (i, _) in text.char_indices() {
        let Some(s) = text.get(i..i+19) else { continue; };
        for fmt in ["%Y-%m-%d_%H:%M:%S", "%Y-%m-%d_%H_%M_%S"] {
            if let Ok(t) = NaiveDateTime::parse_from_str(s, fmt) { return Some(t.and_utc()); }
        }
    }
    None
}
fn iso(t: DateTime<Utc>) -> String { t.format("%Y-%m-%dT%H:%M:%SZ").to_string() }
fn rounded(v: f64, digits: i32) -> f64 { let n = 10_f64.powi(digits); (v*n).round_ties_even()/n }
fn maximum(a: f64, b: f64) -> f64 { if a.is_nan() || b.is_nan() { f64::NAN } else { a.max(b) } }

#[derive(Clone, PartialEq, Serialize, Default)]
struct GridMeta { nlon: usize, nlat: usize, lon0: f64, lat0: f64, dlon: f64, dlat: f64 }
struct Grid { ny: usize, nx: usize, flip: bool, meta: GridMeta }
#[derive(Serialize)]
struct EntryMeta {
    offset: usize, dtype: &'static str, enc: &'static str,
    #[serde(skip_serializing_if="Option::is_none")] scale: Option<f64>,
    #[serde(skip_serializing_if="Option::is_none")] add: Option<f64>,
    #[serde(skip_serializing_if="Option::is_none")] max: Option<f64>,
    units: &'static str, name: &'static str, shape: [usize;2],
}
struct Layout(Vec<(&'static str,EntryMeta)>);
impl Serialize for Layout {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok,S::Error> {
        let mut map=serializer.serialize_map(Some(self.0.len()))?;
        for (key,value) in &self.0 { map.serialize_entry(key,value)?; }
        map.end()
    }
}
#[derive(Serialize)]
struct StaticMeta<'a> {
    format: &'static str, id: &'a str,
    #[serde(flatten)] grid: &'a GridMeta,
    layout: &'a Layout,
    #[serde(rename="hgtMax")] hgt_max: f64,
}
#[derive(Serialize)]
struct FrameMeta<'a> {
    format: &'static str, id: &'a str, lead: i64, valid: String,
    #[serde(flatten)] grid: &'a GridMeta,
    layout: &'a Layout,
}
#[derive(Serialize,Deserialize)]
struct RainPrevious { schema:String, valid:String, grid_json:String, flip:bool, key:String }
fn previous_rain(path: &Path, g: &Grid, valid: DateTime<Utc>) -> Result<(Vec<f64>,f64)> {
    if path.extension().is_some_and(|s|s=="f64") {
        let meta: RainPrevious=serde_json::from_slice(&fs::read(path.with_extension("json")).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
        if meta.schema!="woof-globe-acc/v1" || meta.grid_json!=ascii_json(&g.meta) || meta.flip!=g.flip {
            return Err("previous accumulation grid differs; prevents incorrect precipitation differences".into());
        }
        if hash(path)?!=meta.key { return Err("previous accumulation hash differs; prevents changed precipitation differences".into()); }
        let bytes=fs::read(path).map_err(|e|e.to_string())?;
        if bytes.len()!=g.ny*g.nx*8 { return Err("previous accumulation byte count differs from the surface grid".into()); }
        let values=bytes.chunks_exact(8).map(|b|f64::from_le_bytes(b.try_into().unwrap())).collect();
        let previous=DateTime::parse_from_rfc3339(&meta.valid).map_err(|e|e.to_string())?.with_timezone(&Utc);
        return Ok((values,(valid-previous).num_seconds() as f64/3600.0));
    }
    let previous=Tape::open(path)?;let pg=grid(&previous)?;
    if pg.meta!=g.meta || pg.flip!=g.flip { return Err("previous tape grid differs; prevents incorrect precipitation differences".into()); }
    Ok((previous.rain(g.ny*g.nx)?,(valid-previous.valid()?).num_seconds() as f64/3600.0))
}
fn retain_rain(out: &Path, key: &str, g: &Grid, valid: DateTime<Utc>, acc: &[f64]) -> Result<PathBuf> {
    let path=out.join(".acc").join(format!("{key}.f64"));
    let bytes:Vec<u8>=acc.iter().flat_map(|v|v.to_le_bytes()).collect();
    atomic_write(&path,&bytes)?;
    let meta=RainPrevious {schema:"woof-globe-acc/v1".into(),valid:iso(valid),grid_json:ascii_json(&g.meta),flip:g.flip,key:hash(&path)?};
    atomic_write(&path.with_extension("json"),serde_json::to_string(&meta).map_err(|e|e.to_string())?.as_bytes())?;
    Ok(path)
}
fn grid(tape: &Tape) -> Result<Grid> {
    let lat = tape.get("XLAT")?;
    let lon = tape.get("XLONG")?;
    let lats = match lat.shape.as_slice() {
        [ny, nx] => (0..*ny).map(|j| lat.values[j*nx]).collect::<Vec<_>>(),
        [_] => lat.values,
        _ => return Err("XLAT must be a vector or a plane".into()),
    };
    let lons = match lon.shape.as_slice() {
        [_, nx] => lon.values[..*nx].to_vec(),
        [_] => lon.values,
        _ => return Err("XLONG must be a vector or a plane".into()),
    };
    let (ny, nx) = (lats.len(), lons.len());
    if ny < 2 || nx < 2 || !lats.iter().chain(&lons).all(|v| v.is_finite()) { return Err("tape grid needs finite coordinates and at least two rows and columns".into()); }
    let mut diffs: Vec<f64> = lons.windows(2).map(|w| (w[1]-w[0]+180.0).rem_euclid(360.0)-180.0).collect();
    diffs.sort_by(f64::total_cmp);
    let mid = diffs.len()/2;
    let dlon = if diffs.len()%2 == 0 { (diffs[mid-1]+diffs[mid])/2.0 } else { diffs[mid] };
    if (nx as f64 * dlon - 360.0).abs() > 0.51*dlon.abs() { return Err(format!("the tape grid is not periodic in longitude ({nx} x {dlon} degrees)")); }
    let flip = lats[0] > lats[ny-1];
    let lat0 = if flip { lats[ny-1] } else { lats[0] };
    let dlat = (lats[ny-1]-lats[0]).abs() / (ny-1) as f64;
    let meta = GridMeta { nlon:nx, nlat:ny,
        lon0:rounded((lons[0]+180.0).rem_euclid(360.0)-180.0,6), lat0:rounded(lat0,6),
        dlon:rounded(360.0/nx as f64,9), dlat:rounded(dlat,9) };
    Ok(Grid { ny, nx, flip, meta })
}

struct Field { key: &'static str, enc: &'static str, scale: f64, add: f64, max: f64, units: &'static str, name: &'static str }
const FIELDS: &[Field] = &[
    Field { key:"u10", enc:"rowdiff", scale:0.2, add:0.0, max:0.0, units:"m/s", name:"10 m wind, eastward" },
    Field { key:"v10", enc:"rowdiff", scale:0.2, add:0.0, max:0.0, units:"m/s", name:"10 m wind, northward" },
    Field { key:"t2", enc:"rowdiff", scale:0.05, add:0.0, max:0.0, units:"degC", name:"2 m temperature" },
    Field { key:"mslp", enc:"rowdiff", scale:0.1, add:1000.0, max:0.0, units:"hPa", name:"Sea-level pressure" },
    Field { key:"precip", enc:"rowdiff", scale:0.02, add:0.0, max:0.0, units:"mm/h", name:"Precipitation rate" },
    Field { key:"cloud", enc:"sqrt8", scale:0.0, add:0.0, max:2.0, units:"kg/m2", name:"Cloud condensate path (liquid, ice and snow)" },
    Field { key:"u250", enc:"rowdiff", scale:0.2, add:0.0, max:0.0, units:"m/s", name:"250 hPa wind, eastward" },
    Field { key:"v250", enc:"rowdiff", scale:0.2, add:0.0, max:0.0, units:"m/s", name:"250 hPa wind, northward" },
];
const STATIC: &[Field] = &[
    Field { key:"hgt", enc:"rowdiff", scale:1.0, add:0.0, max:0.0, units:"m", name:"Terrain height" },
    Field { key:"land", enc:"raw", scale:0.0, add:0.0, max:0.0, units:"", name:"Land (1) or water (0)" },
];

fn assemble(g: &Grid, arrays: &[(Field, Vec<f64>)]) -> (Vec<u8>, Layout) {
    let mut raw = Vec::new(); let mut layout = Layout(Vec::new());
    for (field, values) in arrays {
        raw.resize((raw.len()+7)/8*8, 0);
        layout.0.push((field.key,EntryMeta { offset:raw.len(),
            dtype:if field.enc=="rowdiff" {"int16"} else {"uint8"}, enc:field.enc,
            scale:if field.enc=="rowdiff" {Some(field.scale)} else {None},
            add:if field.enc=="rowdiff" {Some(field.add)} else {None},
            max:if field.enc=="sqrt8" {Some(field.max)} else {None},
            units:field.units,name:field.name,shape:[g.ny,g.nx] }));
        for row in 0..g.ny {
            let src = if g.flip { g.ny-1-row } else { row };
            let mut previous = 0_i16;
            for column in 0..g.nx {
                let mut v = values[src*g.nx+column];
                if v.is_nan() { v = 0.0; }
                if field.enc == "rowdiff" {
                    let q = ((v-field.add)/field.scale).round_ties_even().clamp(-32768.0, 32767.0) as i16;
                    raw.extend_from_slice(&q.wrapping_sub(previous).to_le_bytes()); previous = q;
                } else if field.enc == "sqrt8" {
                    raw.push((255.0*(v.clamp(0.0, field.max)/field.max).sqrt()).round_ties_even() as u8);
                } else { raw.push(v.round_ties_even().clamp(0.0,255.0) as u8); }
            }
        }
    }
    (raw, layout)
}
impl Clone for Field { fn clone(&self) -> Self { Self { ..*self } } }

// scipy.ndimage.gaussian_filter(sigma=3, truncate=4, nearest/wrap).
// Its symmetric correlation accumulates the centre, then symmetric pairs
// from the outermost pair inward. No fused multiply-add is used.
fn smooth(input: &[f64], ny: usize, nx: usize) -> Vec<f64> {
    let radius = 12;
    let mut weights: Vec<f64> = (-radius..=radius).map(|x| (-0.5/(3.0*3.0)*(x*x) as f64).exp()).collect();
    let total: f64 = weights.iter().sum(); for w in &mut weights { *w /= total; }
    let mut a = vec![0.0; input.len()]; let mut b = a.clone();
    for y in 0..ny { for x in 0..nx {
        let mut sum = input[y*nx+x]*weights[radius as usize];
        for d in (1..=radius).rev() {
            let lo = (y as i32-d).max(0) as usize; let hi = (y as i32+d).min(ny as i32-1) as usize;
            sum += (input[lo*nx+x]+input[hi*nx+x])*weights[(radius-d) as usize];
        } a[y*nx+x] = sum;
    } }
    for y in 0..ny { for x in 0..nx {
        let mut sum = a[y*nx+x]*weights[radius as usize];
        for d in (1..=radius).rev() {
            let lo = (x as i32-d).rem_euclid(nx as i32) as usize; let hi = (x as i32+d).rem_euclid(nx as i32) as usize;
            sum += (a[y*nx+lo]+a[y*nx+hi])*weights[(radius-d) as usize];
        } b[y*nx+x] = sum;
    } } b
}

fn pressure(tape: &Tape, size: usize) -> Result<(usize, Vec<f64>)> {
    let p = tape.get("P")?; let pb = tape.get("PB")?;
    if p.shape.len() != 3 || p.shape != pb.shape || p.shape[0] < 2 || p.values.len() != p.shape[0]*size { return Err("P/PB must have matching volume shapes with at least two levels".into()); }
    Ok((p.shape[0], p.values.iter().zip(pb.values).map(|(a,b)| a+b).collect()))
}
fn cloud(tape: &Tape, psfc: &[f64], nz: usize, p: &[f64]) -> Result<Vec<f64>> {
    let size = psfc.len(); let mut q = tape.get("QCLOUD")?.values;
    if q.len() != nz*size { return Err("QCLOUD volume does not match pressure".into()); }
    for name in ["QICE", "QSNOW"] {
        if tape.has(name) { let extra = tape.get(name)?.values;
            if extra.len() != q.len() { return Err(format!("{name} volume does not match pressure")); }
            for (a,b) in q.iter_mut().zip(extra) { *a += b; }
        }
    }
    let mut result = vec![0.0; size];
    for i in 0..size {
        let mut lower = psfc[i];
        for k in 0..nz {
            let upper = if k+1 < nz { 0.5*(p[k*size+i]+p[(k+1)*size+i]) }
                else { maximum(p[k*size+i]-0.5*(p[(k-1)*size+i]-p[k*size+i]),0.0) };
            result[i] += maximum(q[k*size+i],0.0)*maximum(lower-upper,0.0); lower = upper;
        } result[i] /= 9.80665;
    } Ok(result)
}
fn wind(tape: &Tape, g: &Grid, nz: usize, p: &[f64]) -> Result<Option<(Vec<f64>,Vec<f64>)>> {
    if !tape.has("U") || !tape.has("V") { return Ok(None); }
    let size = g.ny*g.nx;
    let mut winds = Vec::new();
    for (name, stag_y, stag_x) in [("U",0,1),("V",1,0)] {
        let a = tape.get(name)?;
        if a.shape != [nz,g.ny,g.nx] && a.shape != [nz,g.ny+stag_y,g.nx+stag_x] { return Err(format!("{name}: wind shape does not match pressure")); }
        let mut dest = vec![0.0; nz*size];
        for k in 0..nz { for y in 0..g.ny { for x in 0..g.nx {
            let offset = (k*a.shape[1]+y)*a.shape[2]+x;
            dest[k*size+y*g.nx+x] = if a.shape == [nz,g.ny,g.nx] { a.values[offset] }
                else { 0.5*(a.values[offset]+a.values[offset+stag_y*a.shape[2]+stag_x]) };
        } } }
        let mut result = vec![0.0; size];
        for i in 0..size {
            let count = (0..nz).filter(|k| p[k*size+i] > 25000.0).count();
            let k = count.saturating_sub(1).min(nz-2);
            let l0 = maximum(p[k*size+i],1.0).ln(); let l1 = maximum(p[(k+1)*size+i],1.0).ln();
            let den = if (l1-l0).abs()<1e-9 { 1e-9 } else { l1-l0 };
            let w = ((25000.0_f64.ln()-l0)/den).clamp(0.0,1.0);
            result[i] = dest[k*size+i]+(dest[(k+1)*size+i]-dest[k*size+i])*w;
        } winds.push(result);
    } Ok(Some((winds.remove(0),winds.remove(0))))
}

fn ascii_json<T: Serialize + ?Sized>(value: &T) -> String {
    let text = serde_json::to_string(value).unwrap(); let mut out = String::new();
    for c in text.chars() {
        if c.is_ascii() { out.push(c); } else {
            let mut b = [0_u16;2]; for v in c.encode_utf16(&mut b) { out.push_str(&format!("\\u{v:04x}")); }
        }
    } out
}
fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    fs::create_dir_all(path.parent().unwrap()).map_err(|e| e.to_string())?;
    let tmp = path.with_file_name(format!("{}.tmp", path.file_name().unwrap().to_string_lossy()));
    let result = (|| { fs::write(&tmp, bytes).map_err(|e| e.to_string())?; fs::rename(&tmp,path).map_err(|e| e.to_string()) })();
    if result.is_err() { let _ = fs::remove_file(tmp); } result
}
fn write_js(path: &Path, registry: &str, key: &str, meta: impl Serialize, raw: &[u8]) -> Result<usize> {
    let z = STANDARD.encode(zlib(raw)?);
    let text = format!("(window.{registry}=window.{registry}||{{}})[{}]={{meta:{},z:\"{z}\"}};\n", ascii_json(&json!(key)),ascii_json(&meta));
    atomic_write(path,text.as_bytes())?; Ok(text.len())
}

fn zlib(raw: &[u8]) -> Result<Vec<u8>> {
    use std::ffi::{c_int,c_ulong};
    type Bound = unsafe extern "C" fn(c_ulong) -> c_ulong;
    type Compress = unsafe extern "C" fn(*mut u8,*mut c_ulong,*const u8,c_ulong,c_int) -> c_int;
    let names = if cfg!(windows) { & ["zlib1.dll","zlib.dll"][..] }
        else if cfg!(target_os="macos") { &["libz.1.dylib"][..] } else { &["libz.so.1"][..] };
    let explicit = std::env::var("GPUWM_GLOBE_ZLIB").ok();
    let candidates: Vec<&str> = explicit.as_deref().map(|v|vec![v]).unwrap_or_else(||names.to_vec());
    // Library ownership lasts through both calls. The signatures are zlib's
    // public ABI; c_ulong follows the host C ABI on Linux and Windows.
    let library = candidates.iter().find_map(|name|unsafe {libloading::Library::new(*name).ok()})
        .ok_or("zlib is absent; prevents changing compressed globe bytes with another compressor")?;
    unsafe {
        let bound: libloading::Symbol<Bound> = library.get(b"compressBound\0").map_err(|e|e.to_string())?;
        let compress: libloading::Symbol<Compress> = library.get(b"compress2\0").map_err(|e|e.to_string())?;
        let n: c_ulong = raw.len().try_into().map_err(|_|"globe payload exceeds zlib's source length")?;
        let mut length = bound(n);
        let capacity: usize = length.try_into().map_err(|_|"zlib bound exceeds address space")?;
        let mut out = vec![0_u8;capacity];
        let status = compress(out.as_mut_ptr(),&mut length,raw.as_ptr(),n,9);
        if status != 0 { return Err(format!("zlib compression failed ({status}); no globe file published")); }
        if length as usize > capacity { return Err("zlib returned more bytes than its output capacity".into()); }
        out.truncate(length as usize);
        Ok(out)
    }
}

fn pack(req: &Request) -> Result<Value> {
    let tape = Tape::open(&req.tape)?; let g = grid(&tape)?; let size = g.ny*g.nx;
    let hgt = tape.plane("HGT",size)?;
    if req.action == "static" {
        let max = hgt.iter().copied().fold(f64::NEG_INFINITY,f64::max);
        let arrays = vec![(STATIC[0].clone(),hgt),(STATIC[1].clone(),tape.plane("LANDMASK",size)?)];
        let (raw,layout) = assemble(&g,&arrays);
        let meta=StaticMeta {format:FORMAT,id:&req.id,grid:&g.meta,layout:&layout,hgt_max:rounded(max,1)};
        let bytes = write_js(&req.out.join("static.js"),"WG_STATIC",&req.id,meta,&raw)?;
        return Ok(json!({"bytes":bytes}));
    }
    let valid = tape.valid()?;
    let init = DateTime::parse_from_rfc3339(&req.init).map_err(|e| format!("init: {e}"))?.with_timezone(&Utc);
    let lead = ((valid-init).num_seconds() as f64/3600.0).round_ties_even() as i64;
    let key = format!("f{lead:03}");
    let t2 = tape.plane("T2",size)?; let psfc = tape.plane("PSFC",size)?;
    let slp = (0..size).map(|i| psfc[i]*(1.0-0.0065*hgt[i]/(t2[i]+0.0065*hgt[i])).powf(-5.257)/100.0).collect::<Vec<_>>();
    let acc = tape.rain(size)?;
    let mut rate = vec![0.0;size];
    if let Some(path) = &req.previous {
        let (previous,hours)=previous_rain(path,&g,valid)?;
        if hours > 0.0 { for ((r,a),b) in rate.iter_mut().zip(&acc).zip(previous) { *r = maximum(a-b,0.0)/hours; } }
    }
    let (nz,p) = pressure(&tape,size)?;
    let mut arrays = vec![
        (FIELDS[0].clone(),tape.plane("U10",size)?), (FIELDS[1].clone(),tape.plane("V10",size)?),
        (FIELDS[2].clone(),t2.iter().map(|v| v-273.15).collect()), (FIELDS[3].clone(),smooth(&slp,g.ny,g.nx)),
        (FIELDS[4].clone(),rate), (FIELDS[5].clone(),cloud(&tape,&psfc,nz,&p)?),
    ];
    if let Some((u,v)) = wind(&tape,&g,nz,&p)? { arrays.push((FIELDS[6].clone(),u)); arrays.push((FIELDS[7].clone(),v)); }
    let fields: Vec<&str> = arrays.iter().map(|(f,_)| f.key).collect();
    let (raw,layout) = assemble(&g,&arrays);
    let meta=FrameMeta {format:FORMAT,id:&req.id,lead,valid:iso(valid),grid:&g.meta,layout:&layout};
    let bytes = write_js(&req.out.join(format!("{key}.js")),"WG_FRAME",&format!("{}/{key}",req.id),meta,&raw)?;
    let retained=retain_rain(&req.out,&key,&g,valid,&acc)?;
    let model_dx_km = tape.file.attributes().ok().and_then(|attrs| attrs.iter().find(|a| a.name()=="DX").and_then(|a| a.as_f64()))
        .map(|dx| dx/1000.0).unwrap_or(2.0*std::f64::consts::PI*6371.22/g.nx as f64);
    Ok(json!({"key":key,"lead":lead,"valid":iso(valid),"bytes":bytes,"acc":retained,"fields":fields,"model_dx_km":model_dx_km}))
}

fn hash(path: &Path) -> Result<String> {
    let mut f = fs::File::open(path).map_err(|e| e.to_string())?; let mut sha = Sha256::new(); let mut buf = vec![0;1024*1024];
    loop { let n = f.read(&mut buf).map_err(|e| e.to_string())?; if n == 0 { break; } sha.update(&buf[..n]); }
    Ok(format!("sha256:{:x}",sha.finalize()))
}
fn verify_statics(req: &Request) -> Result<Value> {
    let sidecar = req.path.with_extension("json");
    let doc: Value = serde_json::from_slice(&fs::read(&sidecar).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    if doc["schema"] != "gpuwm.arwen-global-statics/v1" || doc["truncation"].as_u64() != req.truncation.map(|n|n as u64) {
        return Err("native statics provenance schema or truncation differs; prevents using another surface grid".into());
    }
    let key = hash(&req.path)?;
    if doc["npz_sha256"].as_str() != Some(&key[7..]) { return Err("native statics payload differs from its provenance hash; prevents changed surface fields".into()); }
    Ok(json!({"key":key,"bytes":fs::metadata(&req.path).map_err(|e|e.to_string())?.len()}))
}
fn run(req: Request) -> Result<Value> {
    match req.action.as_str() {
        "metadata" => Ok(json!({"valid":iso(Tape::open(&req.tape)?.valid()?)})),
        "static"|"frame" => pack(&req),
        "verify-statics" => verify_statics(&req),
        "verify"|"install" => {
            let key = hash(&req.path)?;
            if req.key.as_ref().is_some_and(|k| *k != key) { return Err(format!("statics hash differs: expected {}, got {key}; prevents loading changed surface fields",req.key.unwrap())); }
            if req.action == "install" {
                let dest = req.dest.ok_or("statics install destination absent")?;
                fs::create_dir_all(dest.parent().unwrap()).map_err(|e| e.to_string())?;
                let tmp = dest.with_file_name(format!("{}.tmp",dest.file_name().unwrap().to_string_lossy()));
                let result = (|| { fs::copy(&req.path,&tmp).map_err(|e| e.to_string())?;
                    if hash(&tmp)? != key { return Err("statics changed during copy; prevents publishing unverified surface fields".into()); }
                    fs::rename(&tmp,&dest).map_err(|e| e.to_string()) })();
                if result.is_err() { let _ = fs::remove_file(&tmp); } result?;
            }
            Ok(json!({"key":key,"bytes":fs::metadata(req.path).map_err(|e|e.to_string())?.len()}))
        },
        _ => Err("unknown globe-pack action".into()),
    }
}
fn main() {
    if std::env::args().any(|a| a=="--abi" || a=="--help") {
        println!("gpuwm-rw-globe-pack-v1 stdin-json actions=static,frame,metadata,verify,verify-statics,install woof-globe/1 {GPUWM_BRIDGE_SOURCE_REV_STAMP}"); return;
    }
    let mut text=String::new();
    let result=io::stdin().read_to_string(&mut text).map_err(|e|e.to_string())
        .and_then(|_| serde_json::from_str(&text).map_err(|e|e.to_string())).and_then(run);
    match result { Ok(v)=>println!("{v}"), Err(e)=>{eprintln!("rw_globe_pack: {e}");std::process::exit(1);} }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn row_wrap_ties_and_padding() {
        let g=Grid { ny:2,nx:3,flip:true,meta:GridMeta::default() };
        let a=vec![(STATIC[1].clone(),vec![0.5,1.5,255.5,0.0,1.0,2.0]),(STATIC[0].clone(),vec![-32768.0,32767.0,0.0,0.5,1.5,2.5])];
        let (raw,layout)=assemble(&g,&a);
        assert_eq!(&raw[..8],&[0,1,2,0,2,255,0,0]);
        assert_eq!(serde_json::to_value(&layout).unwrap()["hgt"]["offset"],8);
        let words:Vec<i16>=raw[8..].chunks_exact(2).map(|b|i16::from_le_bytes([b[0],b[1]])).collect();
        assert_eq!(words,vec![0,2,0,-32768,-1,-32767]);
    }
    #[test] fn gaussian_constant_and_wrap() {
        let a=smooth(&vec![5.0;36],3,12); assert!(a.iter().all(|v|(v-5.0).abs()<1e-14));
        let mut input=vec![0.0;72];input[0]=1.0;let a=smooth(&input,3,24);
        assert_eq!(a[1],a[23]); assert!(a.iter().all(|v|*v>=0.0));
    }
    #[test] fn time_and_unicode() {
        assert_eq!(iso(parse_tape_time("wrfout_d01_2026-10-07_12:00:00").unwrap()),"2026-10-07T12:00:00Z");
        assert_eq!(ascii_json(&json!("é😀")),"\"\\u00e9\\ud83d\\ude00\"");
    }
}
