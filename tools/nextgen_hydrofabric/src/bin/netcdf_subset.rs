//! Offline, lossless ID-row transport. No model parameter conversion or defaults.
//! libnetcdf is linked only by this optional acquisition binary.
use nextgen_hydrofabric::Result;
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Instant;
#[link(name = "netcdf")]
extern "C" {
    fn nc_open(path: *const c_char, mode: c_int, id: *mut c_int) -> c_int;
    fn nc_close(id: c_int) -> c_int;
    fn nc_inq(id: c_int, nd: *mut c_int, nv: *mut c_int, na: *mut c_int, un: *mut c_int) -> c_int;
    fn nc_inq_var(
        id: c_int,
        v: c_int,
        name: *mut c_char,
        t: *mut c_int,
        rank: *mut c_int,
        dims: *mut c_int,
        attrs: *mut c_int,
    ) -> c_int;
    fn nc_inq_dim(id: c_int, d: c_int, name: *mut c_char, n: *mut usize) -> c_int;
    fn nc_inq_attname(id: c_int, v: c_int, a: c_int, name: *mut c_char) -> c_int;
    fn nc_inq_att(id: c_int, v: c_int, name: *const c_char, t: *mut c_int, n: *mut usize) -> c_int;
    fn nc_get_att(id: c_int, v: c_int, name: *const c_char, out: *mut c_void) -> c_int;
    fn nc_get_var(id: c_int, v: c_int, out: *mut c_void) -> c_int;
    fn nc_get_var_longlong(id: c_int, v: c_int, out: *mut i64) -> c_int;
    fn nc_strerror(code: c_int) -> *const c_char;
    fn nc_inq_libvers() -> *const c_char;
}
fn check(rc: c_int) -> Result<()> {
    if rc != 0 {
        return Err(unsafe { CStr::from_ptr(nc_strerror(rc)) }
            .to_string_lossy()
            .into_owned()
            .into());
    }
    Ok(())
}
struct Nc(c_int);
impl Drop for Nc {
    fn drop(&mut self) {
        unsafe {
            nc_close(self.0);
        }
    }
}
fn hash(path: &Path) -> Result<String> {
    let mut f = std::fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut b = vec![0; 1 << 20];
    loop {
        let n = f.read(&mut b)?;
        if n == 0 {
            break;
        }
        h.update(&b[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}
fn sha(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}
fn primitive(t: c_int) -> Result<(&'static str, usize)> {
    Ok(match t {
        1 => ("|i1", 1),
        2 => ("|S1", 1),
        3 => ("<i2", 2),
        4 => ("<i4", 4),
        5 => ("<f4", 4),
        6 => ("<f8", 8),
        7 => ("|u1", 1),
        8 => ("<u2", 2),
        9 => ("<u4", 4),
        10 => ("<i8", 8),
        11 => ("<u8", 8),
        _ => return Err(format!("unsupported NetCDF type {t}; no lossy conversion").into()),
    })
}
fn ctext(buf: &[c_char]) -> String {
    unsafe { CStr::from_ptr(buf.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}
fn aligned(n: usize) -> Vec<u64> {
    vec![0; n.div_ceil(8)]
}
fn bytes(a: &[u64], n: usize) -> &[u8] {
    unsafe { std::slice::from_raw_parts(a.as_ptr().cast(), n) }
}
fn attributes(nc: c_int, v: c_int, n: c_int) -> Result<Value> {
    let mut m = BTreeMap::new();
    for a in 0..n {
        let mut name = [0; 257];
        unsafe {
            check(nc_inq_attname(nc, v, a, name.as_mut_ptr()))?;
        }
        let key = ctext(&name);
        let (mut t, mut count) = (0, 0);
        unsafe {
            check(nc_inq_att(nc, v, name.as_ptr(), &mut t, &mut count))?;
        }
        let (dtype, width) = primitive(t)?;
        let len = count.checked_mul(width).ok_or("attribute overflow")?;
        let mut b = aligned(len);
        unsafe {
            check(nc_get_att(nc, v, name.as_ptr(), b.as_mut_ptr().cast()))?;
        }
        let b = bytes(&b, len);
        let mut entry = json!({"dtype":dtype,"length":count,"little_endian_hex":b.iter().map(|x|format!("{x:02x}")).collect::<String>()});
        if t == 2 {
            if let Ok(text) = std::str::from_utf8(b) {
                entry["text"] = json!(text);
            }
        }
        m.insert(key, entry);
    }
    Ok(serde_json::to_value(m)?)
}
#[derive(Deserialize)]
struct Job {
    source: PathBuf,
    sha256: String,
    source_url: String,
    source_version: String,
    source_terms: String,
    selection_manifest: PathBuf,
    selection_sha256: String,
    ids_key: String,
    id_variable: String,
    output: PathBuf,
}
struct Variable {
    id: c_int,
    name: String,
    xtype: c_int,
    shape: Vec<usize>,
    dims: Vec<c_int>,
    attrs: Value,
}
fn run(job: Job) -> Result<Value> {
    if !cfg!(target_endian = "little") {
        return Err("native raw transport requires little-endian host".into());
    }
    let start = Instant::now();
    if job.source_version.trim().is_empty() || job.source_terms.trim().is_empty() {
        return Err("explicit source version and terms required".into());
    }
    if job.output.exists() {
        return Err("output already exists; preserve prior data".into());
    }
    let hstart = Instant::now();
    if hash(&job.source)? != job.sha256 || hash(&job.selection_manifest)? != job.selection_sha256 {
        return Err("source or selection SHA256 mismatch".into());
    }
    let hash_ms = hstart.elapsed().as_secs_f64() * 1000.;
    let selection: Value = serde_json::from_slice(&std::fs::read(&job.selection_manifest)?)?;
    let ids = selection[&job.ids_key]
        .as_array()
        .ok_or("selection IDs must be array")?
        .iter()
        .map(|v| v.as_i64().ok_or("selection ID must be signed i64"))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let wanted: BTreeSet<_> = ids.iter().copied().collect();
    if ids.is_empty() || wanted.len() != ids.len() {
        return Err("empty or duplicate selection IDs".into());
    }
    let io = Instant::now();
    let path = CString::new(job.source.to_str().ok_or("source path UTF8")?)?;
    let mut handle = 0;
    unsafe {
        check(nc_open(path.as_ptr(), 0, &mut handle))?;
    }
    let nc = Nc(handle);
    let (mut nd, mut nv, mut na, mut un) = (0, 0, 0, 0);
    unsafe {
        check(nc_inq(nc.0, &mut nd, &mut nv, &mut na, &mut un))?;
    }
    let mut dims = Vec::new();
    for d in 0..nd {
        let mut name = [0; 257];
        let mut n = 0;
        unsafe {
            check(nc_inq_dim(nc.0, d, name.as_mut_ptr(), &mut n))?;
        }
        dims.push((ctext(&name), n));
    }
    let global = attributes(nc.0, -1, na)?;
    let mut vars = Vec::new();
    for v in 0..nv {
        let mut name = [0; 257];
        let (mut t, mut rank, mut count) = (0, 0, 0);
        unsafe {
            check(nc_inq_var(
                nc.0,
                v,
                name.as_mut_ptr(),
                &mut t,
                &mut rank,
                std::ptr::null_mut(),
                &mut count,
            ))?;
        }
        if !(0..=1024).contains(&rank) {
            return Err("invalid rank".into());
        }
        let mut vd = vec![0; rank as usize];
        unsafe {
            check(nc_inq_var(
                nc.0,
                v,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                vd.as_mut_ptr(),
                std::ptr::null_mut(),
            ))?;
        }
        let shape = vd.iter().map(|d| dims[*d as usize].1).collect();
        vars.push(Variable {
            id: v,
            name: ctext(&name),
            xtype: t,
            shape,
            dims: vd,
            attrs: attributes(nc.0, v, count)?,
        });
    }
    let idvar = vars
        .iter()
        .find(|v| v.name == job.id_variable)
        .ok_or("ID variable missing")?;
    if idvar.shape.len() != 1 || !matches!(idvar.xtype, 3 | 4 | 10) {
        return Err("ID variable must be rank1 signed integer".into());
    }
    let dim = idvar.dims[0];
    let mut source_ids = vec![0i64; idvar.shape[0]];
    unsafe {
        check(nc_get_var_longlong(nc.0, idvar.id, source_ids.as_mut_ptr()))?;
    }
    let metadata_and_ids_ms = io.elapsed().as_secs_f64() * 1000.;
    let scan = Instant::now();
    let found = source_ids
        .par_iter()
        .enumerate()
        .filter_map(|(row, id)| wanted.contains(id).then_some((*id, row)))
        .collect::<Vec<_>>();
    let map: BTreeMap<_, _> = found.iter().copied().collect();
    if map.len() != found.len() {
        return Err("duplicate selected IDs in source".into());
    }
    let missing = ids
        .iter()
        .filter(|id| !map.contains_key(id))
        .copied()
        .collect::<Vec<_>>();
    let selected = ids
        .iter()
        .filter_map(|id| map.get(id).copied())
        .collect::<Vec<_>>();
    let matched = ids
        .iter()
        .filter(|id| map.contains_key(id))
        .copied()
        .collect::<Vec<_>>();
    if selected.is_empty() {
        return Err("no selected source IDs; no empty substitute emitted".into());
    }
    let scan_ms = scan.elapsed().as_secs_f64() * 1000.;
    // All validation precedes materialization. A distinct output preserves every
    // prior successful preparation; each field is checked before allocation.
    for v in &vars {
        primitive(v.xtype)?;
        if v.shape.contains(&0) {
            return Err("empty source dimension; no empty physical substitute emitted".into());
        }
        v.shape
            .iter()
            .try_fold(1usize, |a, b| a.checked_mul(*b).ok_or("shape overflow"))?;
    }
    std::fs::create_dir_all(&job.output)?;
    let mut files = BTreeMap::new();
    let mut skipped = Vec::new();
    let (mut read_ms, mut pack_ms, mut write_ms) = (0., 0., 0.);
    for v in vars {
        let axes = v
            .dims
            .iter()
            .enumerate()
            .filter_map(|(i, d)| (*d == dim).then_some(i))
            .collect::<Vec<_>>();
        if axes.len() != 1 {
            skipped.push(json!({"name":v.name,"source_shape":v.shape,"attributes":v.attrs,"reason":"not exactly one selected-ID dimension; not broadcast or guessed"}));
            continue;
        }
        let axis = axes[0];
        let (dtype, width) = primitive(v.xtype)?;
        let size = v
            .shape
            .iter()
            .try_fold(width, |a, b| a.checked_mul(*b).ok_or("payload overflow"))?;
        let before: usize = v.shape[..axis].iter().product();
        let after: usize = v.shape[axis + 1..].iter().product();
        let rowbytes = after.checked_mul(width).ok_or("row overflow")?;
        let outsize = before
            .checked_mul(selected.len())
            .and_then(|n| n.checked_mul(rowbytes))
            .ok_or("subset overflow")?;
        let t = Instant::now();
        let mut raw = aligned(size);
        unsafe {
            check(nc_get_var(nc.0, v.id, raw.as_mut_ptr().cast()))?;
        }
        read_ms += t.elapsed().as_secs_f64() * 1000.;
        let raw = bytes(&raw, size);
        let t = Instant::now();
        let mut subset = vec![0u8; outsize];
        subset
            .par_chunks_mut(rowbytes)
            .enumerate()
            .for_each(|(chunk, out)| {
                let outer = chunk / selected.len();
                let row = selected[chunk % selected.len()];
                let offset = (outer * v.shape[axis] + row) * rowbytes;
                out.copy_from_slice(&raw[offset..offset + rowbytes]);
            });
        pack_ms += t.elapsed().as_secs_f64() * 1000.;
        let t = Instant::now();
        let filename = format!("field-{:04}.bin", v.id);
        std::fs::write(job.output.join(&filename), &subset)?;
        let mut shape = v.shape.clone();
        shape[axis] = selected.len();
        files.insert(v.name,json!({"path":filename,"dtype":dtype,"netcdf_type":v.xtype,"shape":shape,"source_shape":v.shape,"row_axis":axis,"bytes":subset.len(),"sha256":sha(&subset),"attributes":v.attrs}));
        write_ms += t.elapsed().as_secs_f64() * 1000.;
    }
    let result = json!({"schema":"arwen.source-netcdf-subset.v1","scope":"lossless stored primitive values and attributes by explicit source IDs; no packing/units conversion, model initialization, routing policy, CFE/GW mapping or HYFeatures crosswalk","source":{"path":job.source,"sha256":job.sha256,"url":job.source_url,"version":job.source_version,"terms":job.source_terms,"global_attributes":global,"dimensions":dims},"selection":{"path":job.selection_manifest,"sha256":job.selection_sha256,"key":job.ids_key,"id_variable":job.id_variable,"requested":ids.len(),"matched":matched.len(),"missing_ids":missing,"ids":matched,"source_rows":selected},"files":files,"skipped_variables":skipped,"library":unsafe{CStr::from_ptr(nc_inq_libvers())}.to_string_lossy(),"source_revision":nextgen_hydrofabric::SOURCE_REVISION,"timings_ms":{"input_hash":hash_ms,"metadata_and_ids":metadata_and_ids_ms,"id_scan":scan_ms,"serialized_netcdf_read":read_ms,"parallel_raw_row_pack":pack_ms,"write_and_hash":write_ms,"total":start.elapsed().as_secs_f64()*1000.},"parallelism":{"rayon_threads":rayon::current_num_threads(),"id_scan_and_row_pack":"Rayon parallel rows; deterministic requested-ID output order","netcdf":"serialized C API; no unsafe concurrent HDF5 handles","gpu":"none; offline source preparation only"}});
    std::fs::write(
        job.output.join("manifest.json"),
        serde_json::to_vec_pretty(&result)?,
    )?;
    Ok(result)
}
fn main() {
    let result = (|| -> Result<Value> {
        let path = std::env::args()
            .nth(1)
            .ok_or("rw_hydrofabric_netcdf_subset JOB.json")?;
        run(serde_json::from_slice(&std::fs::read(path)?)?)
    })();
    match result {
        Ok(v) => println!("{}", serde_json::to_string_pretty(&v).unwrap()),
        Err(e) => {
            eprintln!("rw_hydrofabric_netcdf_subset: {e}");
            std::process::exit(1)
        }
    }
}
