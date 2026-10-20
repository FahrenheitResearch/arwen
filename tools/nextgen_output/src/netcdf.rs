//! Rust-owned NetCDF serialization, using only the permissively licensed C
//! container library. No netCDF-Fortran or hydrologic native runtime is linked.
use crate::Result;
use serde::{Deserialize, Serialize};
use std::{
    ffi::{CStr, CString, c_char, c_int, c_void},
    fs,
    path::Path,
    sync::{
        Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::Instant,
};

// netCDF-C is not required to be thread safe. Marshalling and CSV formatting
// use Rayon; the container library is serialized at the output boundary.
static LOCK: Mutex<()> = Mutex::new(());
static SERIAL: AtomicU64 = AtomicU64::new(0);

unsafe extern "C" {
    fn nc_strerror(code: c_int) -> *const c_char;
    fn nc_inq_libvers() -> *const c_char;
    fn nc_create(path: *const c_char, mode: c_int, id: *mut c_int) -> c_int;
    fn nc_open(path: *const c_char, mode: c_int, id: *mut c_int) -> c_int;
    fn nc_close(id: c_int) -> c_int;
    fn nc_def_dim(id: c_int, name: *const c_char, len: usize, dim: *mut c_int) -> c_int;
    fn nc_def_var(
        id: c_int,
        name: *const c_char,
        ty: c_int,
        n: c_int,
        dims: *const c_int,
        var: *mut c_int,
    ) -> c_int;
    fn nc_enddef(id: c_int) -> c_int;
    fn nc_put_att(
        id: c_int,
        var: c_int,
        name: *const c_char,
        ty: c_int,
        len: usize,
        data: *const c_void,
    ) -> c_int;
    fn nc_put_att_text(
        id: c_int,
        var: c_int,
        name: *const c_char,
        len: usize,
        data: *const c_char,
    ) -> c_int;
    fn nc_put_vara(
        id: c_int,
        var: c_int,
        start: *const usize,
        count: *const usize,
        data: *const c_void,
    ) -> c_int;
    fn nc_put_vara_string(
        id: c_int,
        var: c_int,
        start: *const usize,
        count: *const usize,
        data: *const *const c_char,
    ) -> c_int;
    fn nc_get_var(id: c_int, var: c_int, data: *mut c_void) -> c_int;
    fn nc_get_var_string(id: c_int, var: c_int, data: *mut *mut c_char) -> c_int;
    fn nc_free_string(n: usize, data: *mut *mut c_char) -> c_int;
    fn nc_inq(
        id: c_int,
        nd: *mut c_int,
        nv: *mut c_int,
        na: *mut c_int,
        unlim: *mut c_int,
    ) -> c_int;
    fn nc_inq_format(id: c_int, format: *mut c_int) -> c_int;
    fn nc_inq_unlimdims(id: c_int, count: *mut c_int, dims: *mut c_int) -> c_int;
    fn nc_inq_dim(id: c_int, dim: c_int, name: *mut c_char, len: *mut usize) -> c_int;
    fn nc_inq_var(
        id: c_int,
        var: c_int,
        name: *mut c_char,
        ty: *mut c_int,
        nd: *mut c_int,
        dims: *mut c_int,
        na: *mut c_int,
    ) -> c_int;
    fn nc_inq_attname(id: c_int, var: c_int, n: c_int, name: *mut c_char) -> c_int;
    fn nc_inq_att(
        id: c_int,
        var: c_int,
        name: *const c_char,
        ty: *mut c_int,
        len: *mut usize,
    ) -> c_int;
    fn nc_get_att(id: c_int, var: c_int, name: *const c_char, data: *mut c_void) -> c_int;
    fn nc_get_att_text(id: c_int, var: c_int, name: *const c_char, data: *mut c_char) -> c_int;
}

fn check(code: c_int) -> Result<()> {
    if code == 0 {
        return Ok(());
    }
    Err(unsafe { CStr::from_ptr(nc_strerror(code)) }
        .to_string_lossy()
        .into_owned()
        .into())
}
struct Handle(c_int);
impl Drop for Handle {
    fn drop(&mut self) {
        if self.0 >= 0 {
            unsafe {
                nc_close(self.0);
            }
        }
    }
}
impl Handle {
    fn close(mut self) -> Result<()> {
        let id = self.0;
        self.0 = -1;
        check(unsafe { nc_close(id) })
    }
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
#[serde(deny_unknown_fields)]
pub struct Dimension {
    pub name: String,
    pub len: usize,
    #[serde(default)]
    pub unlimited: bool,
}
#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
#[serde(deny_unknown_fields)]
pub struct Attribute {
    pub name: String,
    pub dtype: String,
    #[serde(default)]
    pub text: String,
    /// Native little-endian bytes, including the original float bit patterns.
    #[serde(default)]
    pub bytes: Vec<u8>,
}
impl Attribute {
    pub fn text(name: &str, value: &str) -> Self {
        Self {
            name: name.into(),
            dtype: "char".into(),
            text: value.into(),
            bytes: vec![],
        }
    }
    pub fn f64(name: &str, value: f64) -> Self {
        Self {
            name: name.into(),
            dtype: "f64le".into(),
            text: String::new(),
            bytes: value.to_le_bytes().to_vec(),
        }
    }
    pub fn f32(name: &str, value: f32) -> Self {
        Self {
            name: name.into(),
            dtype: "f32le".into(),
            text: String::new(),
            bytes: value.to_le_bytes().to_vec(),
        }
    }
}
#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
#[serde(deny_unknown_fields)]
pub struct Variable {
    pub name: String,
    pub dtype: String,
    pub dimensions: Vec<String>,
    #[serde(default)]
    pub attributes: Vec<Attribute>,
    #[serde(default)]
    pub offset: usize,
    #[serde(default)]
    pub strings: Vec<String>,
}
#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
#[serde(deny_unknown_fields)]
pub struct Dataset {
    pub format: String,
    pub dimensions: Vec<Dimension>,
    pub variables: Vec<Variable>,
    #[serde(default)]
    pub attributes: Vec<Attribute>,
}

pub fn datatype(dtype: &str) -> Result<(c_int, usize)> {
    Ok(match dtype {
        "i8" => (1, 1),
        "char" => (2, 1),
        "i16le" => (3, 2),
        "i32le" => (4, 4),
        "f32le" => (5, 4),
        "f64le" => (6, 8),
        "u8" => (7, 1),
        "u16le" => (8, 2),
        "u32le" => (9, 4),
        "i64le" => (10, 8),
        "u64le" => (11, 8),
        "string" => (12, 0),
        _ => return Err(format!("unsupported NetCDF type {dtype}").into()),
    })
}
fn typename(code: c_int) -> Result<String> {
    for name in [
        "i8", "char", "i16le", "i32le", "f32le", "f64le", "u8", "u16le", "u32le", "i64le", "u64le",
        "string",
    ] {
        if datatype(name)?.0 == code {
            return Ok(name.into());
        }
    }
    Err("compound/vlen/enum data are not primitive resident arrays".into())
}
fn name(value: &str) -> Result<CString> {
    Ok(CString::new(value)?)
}
fn attr(id: c_int, var: c_int, a: &Attribute) -> Result<()> {
    let key = name(&a.name)?;
    if a.dtype == "char" {
        check(unsafe {
            nc_put_att_text(id, var, key.as_ptr(), a.text.len(), a.text.as_ptr().cast())
        })
    } else {
        let (ty, width) = datatype(&a.dtype)?;
        if width == 0 || !a.bytes.len().is_multiple_of(width) {
            return Err("invalid attribute byte count".into());
        }
        // Align the buffer for the C container library's primitive loads.
        let aligned = aligned(&a.bytes);
        check(unsafe {
            nc_put_att(
                id,
                var,
                key.as_ptr(),
                ty,
                a.bytes.len() / width,
                aligned.as_ptr().cast(),
            )
        })
    }
}
fn aligned(bytes: &[u8]) -> Vec<u64> {
    let mut out = vec![0u64; bytes.len().div_ceil(8)];
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.as_mut_ptr().cast(), bytes.len());
    }
    out
}
impl Dataset {
    pub fn elements(&self, var: &Variable) -> Result<usize> {
        var.dimensions.iter().try_fold(1usize, |n, key| {
            let d = self
                .dimensions
                .iter()
                .find(|d| &d.name == key)
                .ok_or("undefined dimension")?;
            n.checked_mul(d.len)
                .ok_or_else(|| "dimension size overflow".into())
        })
    }
    pub fn validate(&self, payload: &[u8]) -> Result<()> {
        if cfg!(target_endian = "big") {
            return Err("little-endian snapshot transport required".into());
        }
        if !["netcdf4", "cdf5"].contains(&self.format.as_str()) {
            return Err("unsupported container".into());
        }
        let mut dims = std::collections::HashSet::new();
        for dim in &self.dimensions {
            if dim.len == 0 || !dims.insert(&dim.name) {
                return Err("empty or duplicate dimension".into());
            }
            name(&dim.name)?;
        }
        let mut vars = std::collections::HashSet::new();
        let mut ranges = Vec::new();
        for var in &self.variables {
            if !vars.insert(&var.name) {
                return Err("duplicate variable".into());
            }
            name(&var.name)?;
            let (_, width) = datatype(&var.dtype)?;
            let count = self.elements(var)?;
            if width == 0 {
                if var.strings.len() != count {
                    return Err("string variable count differs".into());
                }
                for s in &var.strings {
                    name(s)?;
                }
            } else {
                let size = count.checked_mul(width).ok_or("array size overflow")?;
                let end = var
                    .offset
                    .checked_add(size)
                    .ok_or("array offset overflow")?;
                if end > payload.len() {
                    return Err(format!("incomplete variable {}", var.name).into());
                }
                ranges.push((var.offset, end));
            }
        }
        ranges.sort_unstable();
        let mut end = 0;
        for (a, b) in ranges {
            if a != end {
                return Err("snapshot has gaps or overlapping fields".into());
            }
            end = b;
        }
        if end != payload.len() {
            return Err("snapshot has unclaimed bytes".into());
        }
        Ok(())
    }
}

pub fn write(path: &Path, schema: &Dataset, payload: &[u8]) -> Result<serde_json::Value> {
    schema.validate(payload)?;
    let _guard = LOCK.lock().map_err(|_| "NetCDF lock poisoned")?;
    let started = Instant::now();
    let parent = path.parent().unwrap_or(Path::new("."));
    fs::create_dir_all(parent)?;
    let serial = SERIAL.fetch_add(1, Ordering::Relaxed);
    let partial = path.with_file_name(format!(
        ".{}.partial-{}-{serial}",
        path.file_name()
            .ok_or("missing file name")?
            .to_string_lossy(),
        std::process::id()
    ));
    let mut id = -1;
    let flags = if schema.format == "netcdf4" {
        0x1000
    } else {
        0x0020
    };
    check(unsafe {
        nc_create(
            name(partial.to_str().ok_or("non-UTF8 path")?)?.as_ptr(),
            flags | 4,
            &mut id,
        )
    })?;
    let handle = Handle(id);
    let result = (|| -> Result<(f64, f64, f64)> {
        let mut dimids = std::collections::HashMap::new();
        for dim in &schema.dimensions {
            let mut d = -1;
            check(unsafe {
                nc_def_dim(
                    id,
                    name(&dim.name)?.as_ptr(),
                    if dim.unlimited { 0 } else { dim.len },
                    &mut d,
                )
            })?;
            dimids.insert(&dim.name, d);
        }
        let mut variables = Vec::with_capacity(schema.variables.len());
        for var in &schema.variables {
            let dims = var.dimensions.iter().map(|d| dimids[d]).collect::<Vec<_>>();
            let (ty, _) = datatype(&var.dtype)?;
            let mut v = -1;
            check(unsafe {
                nc_def_var(
                    id,
                    name(&var.name)?.as_ptr(),
                    ty,
                    dims.len().try_into()?,
                    dims.as_ptr(),
                    &mut v,
                )
            })?;
            // Define every attribute before data, retaining the caller's exact
            // order even when _FillValue follows another attribute. Defining
            // the entire header once also avoids repeated CDF5 data relocation.
            for a in &var.attributes {
                attr(id, v, a)?;
            }
            variables.push(v);
        }
        for a in &schema.attributes {
            attr(id, -1, a)?;
        }
        check(unsafe { nc_enddef(id) })?;
        let definition_ms = started.elapsed().as_secs_f64() * 1000.;
        let data_started = Instant::now();
        for (var, v) in schema.variables.iter().zip(variables) {
            let (_, width) = datatype(&var.dtype)?;
            let starts = vec![0usize; var.dimensions.len()];
            let counts = var
                .dimensions
                .iter()
                .map(|key| {
                    schema
                        .dimensions
                        .iter()
                        .find(|d| &d.name == key)
                        .unwrap()
                        .len
                })
                .collect::<Vec<_>>();
            if width == 0 {
                let strings = var
                    .strings
                    .iter()
                    .map(|s| name(s))
                    .collect::<Result<Vec<_>>>()?;
                let ptrs = strings.iter().map(|s| s.as_ptr()).collect::<Vec<_>>();
                check(unsafe {
                    nc_put_vara_string(id, v, starts.as_ptr(), counts.as_ptr(), ptrs.as_ptr())
                })?;
            } else {
                let size = schema.elements(var)? * width;
                let data = aligned(&payload[var.offset..var.offset + size]);
                check(unsafe {
                    nc_put_vara(
                        id,
                        v,
                        starts.as_ptr(),
                        counts.as_ptr(),
                        data.as_ptr().cast(),
                    )
                })?;
            }
        }
        let payload_write_ms = data_started.elapsed().as_secs_f64() * 1000.;
        let publish_started = Instant::now();
        handle.close()?;
        fs::File::open(&partial)?.sync_all()?;
        fs::hard_link(&partial, path)?;
        fs::File::open(parent)?.sync_all()?;
        Ok((
            definition_ms,
            payload_write_ms,
            publish_started.elapsed().as_secs_f64() * 1000.,
        ))
    })();
    let _ = fs::remove_file(&partial);
    let (definition_ms, payload_write_ms, close_fsync_publish_ms) = result?;
    Ok(
        serde_json::json!({"operation":"netcdf-write","variables":schema.variables.len(),"payload_bytes":payload.len(),"container_and_fsync_ms":started.elapsed().as_secs_f64()*1000.,
            "header_definition_ms":definition_ms,"payload_write_ms":payload_write_ms,"close_fsync_publish_ms":close_fsync_publish_ms,
            "container_format":schema.format,"container_backend":"netCDF-C (HDF5 C backend for NetCDF-4)",
            "netcdf_c_version":unsafe { CStr::from_ptr(nc_inq_libvers()) }.to_string_lossy(),
            "parallelism":"Rust boundary marshalling; serialized netCDF-C container calls; no Fortran"}),
    )
}

fn attrs(id: c_int, var: c_int, count: c_int) -> Result<Vec<Attribute>> {
    (0..count)
        .map(|n| {
            let mut key = vec![0i8; 257];
            check(unsafe { nc_inq_attname(id, var, n, key.as_mut_ptr()) })?;
            let mut ty = 0;
            let mut len = 0;
            check(unsafe { nc_inq_att(id, var, key.as_ptr(), &mut ty, &mut len) })?;
            let dtype = typename(ty)?;
            let mut a = Attribute {
                name: unsafe { CStr::from_ptr(key.as_ptr()) }.to_str()?.into(),
                dtype: dtype.clone(),
                text: String::new(),
                bytes: vec![],
            };
            if dtype == "char" {
                let mut text = vec![0u8; len];
                check(unsafe { nc_get_att_text(id, var, key.as_ptr(), text.as_mut_ptr().cast()) })?;
                a.text = String::from_utf8(text)?;
            } else {
                let width = datatype(&dtype)?.1;
                if width == 0 {
                    return Err("string attributes not yet captured".into());
                }
                let size = len.checked_mul(width).ok_or("attribute overflow")?;
                let mut data = vec![0u64; size.div_ceil(8)];
                check(unsafe { nc_get_att(id, var, key.as_ptr(), data.as_mut_ptr().cast()) })?;
                a.bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), size) }
                    .to_vec();
            }
            Ok(a)
        })
        .collect()
}

/// Return a complete typed snapshot. No widening through f64 is performed.
pub fn read(path: &Path) -> Result<(Dataset, Vec<u8>)> {
    let _guard = LOCK.lock().map_err(|_| "NetCDF lock poisoned")?;
    let mut id = -1;
    check(unsafe {
        nc_open(
            name(path.to_str().ok_or("non-UTF8 path")?)?.as_ptr(),
            0,
            &mut id,
        )
    })?;
    let handle = Handle(id);
    let (mut nd, mut nv, mut na, mut unlim, mut format) = (0, 0, 0, 0, 0);
    check(unsafe { nc_inq(id, &mut nd, &mut nv, &mut na, &mut unlim) })?;
    let mut unlimited_count = 0;
    check(unsafe { nc_inq_unlimdims(id, &mut unlimited_count, std::ptr::null_mut()) })?;
    let mut unlimited = vec![0; unlimited_count as usize];
    check(unsafe { nc_inq_unlimdims(id, &mut unlimited_count, unlimited.as_mut_ptr()) })?;
    check(unsafe { nc_inq_format(id, &mut format) })?;
    let mut schema = Dataset {
        format: if format == 3 {
            "netcdf4".into()
        } else if format == 5 {
            "cdf5".into()
        } else {
            return Err("expected NetCDF4 or CDF5".into());
        },
        dimensions: vec![],
        variables: vec![],
        attributes: attrs(id, -1, na)?,
    };
    for d in 0..nd {
        let mut key = vec![0i8; 257];
        let mut len = 0;
        check(unsafe { nc_inq_dim(id, d, key.as_mut_ptr(), &mut len) })?;
        schema.dimensions.push(Dimension {
            name: unsafe { CStr::from_ptr(key.as_ptr()) }.to_str()?.into(),
            len,
            unlimited: unlimited.contains(&d),
        });
    }
    let mut payload = Vec::new();
    for v in 0..nv {
        let mut key = vec![0i8; 257];
        let (mut ty, mut n, mut a) = (0, 0, 0);
        // netCDF's maximum number of dimensions per variable is NC_MAX_VAR_DIMS=1024.
        let mut dims = vec![0; 1024];
        check(unsafe {
            nc_inq_var(
                id,
                v,
                key.as_mut_ptr(),
                &mut ty,
                &mut n,
                dims.as_mut_ptr(),
                &mut a,
            )
        })?;
        let dtype = typename(ty)?;
        let mut var = Variable {
            name: unsafe { CStr::from_ptr(key.as_ptr()) }.to_str()?.into(),
            dtype: dtype.clone(),
            dimensions: dims[..n as usize]
                .iter()
                .map(|d| schema.dimensions[*d as usize].name.clone())
                .collect(),
            attributes: attrs(id, v, a)?,
            offset: payload.len(),
            strings: vec![],
        };
        let count = schema.elements(&var)?;
        let width = datatype(&dtype)?.1;
        if width == 0 {
            let mut strings = vec![std::ptr::null_mut(); count];
            check(unsafe { nc_get_var_string(id, v, strings.as_mut_ptr()) })?;
            let copied = strings
                .iter()
                .map(|p| unsafe { CStr::from_ptr(*p) }.to_str().map(String::from))
                .collect::<std::result::Result<Vec<_>, _>>();
            check(unsafe { nc_free_string(count, strings.as_mut_ptr()) })?;
            var.strings = copied?;
        } else {
            let size = count.checked_mul(width).ok_or("variable size overflow")?;
            let mut data = vec![0u64; size.div_ceil(8)];
            check(unsafe { nc_get_var(id, v, data.as_mut_ptr().cast()) })?;
            payload.extend_from_slice(unsafe {
                std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), size)
            });
        }
        schema.variables.push(var);
    }
    handle.close()?;
    schema.validate(&payload)?;
    Ok((schema, payload))
}
