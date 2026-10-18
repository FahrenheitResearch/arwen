//! Optional, lossless storage for the existing render-frame schema.
//!
//! The unchanged full global frame is over3.7GB in classic encoding. This
//! writer changes only its container: shuffle and deflate never quantize,
//! resample, omit or recompute a value. Standard builds still use the existing
//! classic writer. A feature-enabled build also stays classic unless the
//! explicit GPUWM_RENDER_NETCDF4_DEFLATE=1 storage option is selected.
//! Render frames are finite snapshots: their Time dimension stores the exact
//! declared record count, avoiding an unresolved unlimited HDF5 dimension
//! in the pinned typed reader. All stored dimensions and words are preserved.
use crate::{MpasError, MpasResult};
use rw_store::netcdf_classic::{
    NcAttr, NcAttrValue, NcClassicWriter, NcData, NcDim, NcFormat, NcType, NcVarDef,
};
use std::{
    collections::BTreeMap,
    ffi::{CStr, CString, c_char, c_int, c_void},
    path::Path,
};

// The whole module is compiled only with the `render-netcdf4` feature
// (src/convert.rs); the link follows it, so a default build needs no libnetcdf.
#[cfg_attr(feature = "render-netcdf4", link(name = "netcdf"))]
unsafe extern "C" {
    fn nc_create(path: *const c_char, mode: c_int, id: *mut c_int) -> c_int;
    fn nc_close(id: c_int) -> c_int;
    fn nc_def_dim(id: c_int, name: *const c_char, len: usize, dim: *mut c_int) -> c_int;
    fn nc_def_var(
        id: c_int,
        name: *const c_char,
        ty: c_int,
        nd: c_int,
        dims: *const c_int,
        var: *mut c_int,
    ) -> c_int;
    fn nc_def_var_deflate(
        id: c_int,
        var: c_int,
        shuffle: c_int,
        deflate: c_int,
        level: c_int,
    ) -> c_int;
    fn nc_put_att(
        id: c_int,
        var: c_int,
        name: *const c_char,
        ty: c_int,
        len: usize,
        values: *const c_void,
    ) -> c_int;
    fn nc_set_fill(id: c_int, mode: c_int, old: *mut c_int) -> c_int;
    fn nc_enddef(id: c_int) -> c_int;
    fn nc_put_vara(
        id: c_int,
        var: c_int,
        start: *const usize,
        count: *const usize,
        values: *const c_void,
    ) -> c_int;
    fn nc_strerror(code: c_int) -> *const c_char;
}
fn check(code: c_int) -> MpasResult<()> {
    if code == 0 {
        return Ok(());
    }
    // SAFETY: netCDF returns a permanent NUL-terminated error string.
    let text = unsafe { CStr::from_ptr(nc_strerror(code)) }.to_string_lossy();
    Err(MpasError::NetCdf(format!(
        "lossless render writer: {text} ({code})"
    )))
}
fn name(value: &str) -> MpasResult<CString> {
    CString::new(value).map_err(|_| MpasError::Refusal("netCDF schema name contains NUL".into()))
}
fn tag(ty: NcType) -> c_int {
    match ty {
        NcType::Char => 2,
        NcType::Short => 3,
        NcType::Int => 4,
        NcType::Float => 5,
        NcType::Double => 6,
    }
}
fn data_view(data: NcData<'_>) -> (c_int, usize, *const c_void) {
    match data {
        NcData::Chars(v) => (2, v.len(), v.as_ptr().cast()),
        NcData::Shorts(v) => (3, v.len(), v.as_ptr().cast()),
        NcData::Ints(v) => (4, v.len(), v.as_ptr().cast()),
        NcData::Floats(v) => (5, v.len(), v.as_ptr().cast()),
        NcData::Doubles(v) => (6, v.len(), v.as_ptr().cast()),
    }
}
fn attributes(id: c_int, var: c_int, attrs: &[NcAttr]) -> MpasResult<()> {
    for attr in attrs {
        let key = name(&attr.name)?;
        let (ty, len, ptr) = match &attr.value {
            NcAttrValue::Text(v) => (2, v.len(), v.as_ptr().cast()),
            NcAttrValue::Ints(v) => (4, v.len(), v.as_ptr().cast()),
            NcAttrValue::Floats(v) => (5, v.len(), v.as_ptr().cast()),
            NcAttrValue::Doubles(v) => (6, v.len(), v.as_ptr().cast()),
        };
        // SAFETY: matching native type, live value buffer and checked C name.
        check(unsafe { nc_put_att(id, var, key.as_ptr(), ty, len, ptr) })?;
    }
    Ok(())
}
struct Variable {
    id: c_int,
    ty: c_int,
    /// A fixed-size variable (a CF coordinate axis): one slab, no record.
    fixed: bool,
    count: Vec<usize>,
    written: Vec<bool>,
    expected: Vec<Option<[u8; 32]>>,
}
static NETCDF_WRITER_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
pub(super) struct Deflated {
    id: c_int,
    vars: BTreeMap<String, Variable>,
    records: u64,
    path: std::path::PathBuf,
    _library_lock: std::sync::MutexGuard<'static, ()>,
}
impl Drop for Deflated {
    fn drop(&mut self) {
        if self.id >= 0 {
            unsafe {
                nc_close(self.id);
            }
        }
    }
}
pub(super) enum RenderFrameWriter {
    Classic(NcClassicWriter),
    Deflated(Deflated),
}
impl RenderFrameWriter {
    pub fn create(
        path: &Path,
        format: NcFormat,
        dims: Vec<NcDim>,
        attrs: Vec<NcAttr>,
        vars: Vec<NcVarDef>,
        records: u64,
    ) -> MpasResult<Self> {
        let setting = std::env::var("GPUWM_RENDER_NETCDF4_DEFLATE").unwrap_or_default();
        match setting.as_str() {
            "" | "0" => {
                return Ok(Self::Classic(NcClassicWriter::create(
                    path, format, dims, attrs, vars, records,
                )?));
            }
            "1" => {}
            _ => {
                return Err(MpasError::Refusal(
                    "GPUWM_RENDER_NETCDF4_DEFLATE must be0 or1".into(),
                ));
            }
        }
        Self::create_deflated(path, dims, attrs, vars, records)
    }
    fn create_deflated(
        path: &Path,
        dims: Vec<NcDim>,
        attrs: Vec<NcAttr>,
        vars: Vec<NcVarDef>,
        records: u64,
    ) -> MpasResult<Self> {
        #[cfg(unix)]
        let path_bytes = {
            use std::os::unix::ffi::OsStrExt;
            path.as_os_str().as_bytes().to_vec()
        };
        #[cfg(not(unix))]
        let path_bytes = path
            .to_str()
            .ok_or_else(|| MpasError::Refusal("netCDF path is not UTF-8".into()))?
            .as_bytes()
            .to_vec();
        let filename = CString::new(path_bytes)
            .map_err(|_| MpasError::Refusal("netCDF path contains NUL".into()))?;
        let library_lock = NETCDF_WRITER_LOCK
            .lock()
            .map_err(|_| MpasError::Refusal("netCDF writer lock poisoned after a panic".into()))?;
        let record_count = usize::try_from(records)
            .map_err(|_| MpasError::Refusal("render record count exceeds address space".into()))?;
        if record_count == 0 {
            return Err(MpasError::Refusal(
                "a finite render frame needs at least one record".into(),
            ));
        }
        let mut id = -1;
        // NC_NETCDF4 (0x1000), classic typed schema, lossless deflate only.
        check(unsafe { nc_create(filename.as_ptr(), 0x1000, &mut id) })?;
        let mut out = Deflated {
            id,
            vars: BTreeMap::new(),
            records,
            path: path.to_owned(),
            _library_lock: library_lock,
        };
        let mut dimids = Vec::new();
        if dims.iter().filter(|d| d.unlimited).count() > 1 {
            return Err(MpasError::Refusal(
                "render schema has multiple record dimensions".into(),
            ));
        }
        for d in &dims {
            if !d.unlimited && d.len == 0 {
                return Err(MpasError::Refusal("render fixed dimension is empty".into()));
            }
            let key = name(&d.name)?;
            let mut dim = -1;
            check(unsafe {
                nc_def_dim(
                    id,
                    key.as_ptr(),
                    if d.unlimited { record_count } else { d.len },
                    &mut dim,
                )
            })?;
            dimids.push(dim);
        }
        attributes(id, -1, &attrs)?;
        for v in vars {
            let key = name(&v.name)?;
            let mut shape = Vec::new();
            let mut ids = Vec::new();
            for (axis, &slot) in v.dimids.iter().enumerate() {
                let d = dims.get(slot).ok_or_else(|| {
                    MpasError::Refusal("render variable has invalid dimension index".into())
                })?;
                if d.unlimited && axis != 0 {
                    return Err(MpasError::Refusal(
                        "render record dimension is not first".into(),
                    ));
                }
                shape.push(if d.unlimited { 1 } else { d.len });
                ids.push(dimids[slot]);
            }
            let fixed = !v.dimids.first().is_some_and(|&slot| dims[slot].unlimited);
            if out.vars.contains_key(&v.name) {
                return Err(MpasError::Refusal("duplicate render variable".into()));
            }
            let mut vid = -1;
            check(unsafe {
                nc_def_var(
                    id,
                    key.as_ptr(),
                    tag(v.ty),
                    c_int::try_from(ids.len())
                        .map_err(|_| MpasError::Refusal("render rank exceeds netCDF API".into()))?,
                    ids.as_ptr(),
                    &mut vid,
                )
            })?;
            check(unsafe { nc_def_var_deflate(id, vid, 1, 1, 1) })?;
            attributes(id, vid, &v.attrs)?;
            out.vars.insert(
                v.name,
                Variable {
                    id: vid,
                    ty: tag(v.ty),
                    fixed,
                    count: shape,
                    written: vec![false; if fixed { 1 } else { record_count }],
                    expected: vec![None; if fixed { 1 } else { record_count }],
                },
            );
        }
        let mut old = 0;
        check(unsafe { nc_set_fill(id, 0x100, &mut old) })?; // NC_NOFILL; finish verifies all slabs.
        check(unsafe { nc_enddef(id) })?;
        Ok(Self::Deflated(out))
    }
    /// A fixed-size variable's whole array (a CF coordinate axis).
    pub fn put(&mut self, key: &str, values: NcData<'_>) -> MpasResult<()> {
        match self {
            Self::Classic(w) => {
                w.put(key, values)?;
                Ok(())
            }
            Self::Deflated(w) => {
                if !w.vars.get(key).is_some_and(|v| v.fixed) {
                    return Err(MpasError::Refusal(format!(
                        "render variable {key} is not a fixed-size variable"
                    )));
                }
                self.put_record(key, 0, values)
            }
        }
    }
    pub fn put_record(&mut self, key: &str, record: u64, values: NcData<'_>) -> MpasResult<()> {
        match self {
            Self::Classic(w) => {
                w.put_record(key, record, values)?;
                Ok(())
            }
            Self::Deflated(w) => {
                let fixed = w.vars.get(key).is_some_and(|v| v.fixed);
                if record >= if fixed { 1 } else { w.records } {
                    return Err(MpasError::Refusal(
                        "render record index outside declared extent".into(),
                    ));
                }
                let v = w
                    .vars
                    .get_mut(key)
                    .ok_or_else(|| MpasError::Refusal(format!("render variable absent: {key}")))?;
                let (ty, len, ptr) = data_view(values);
                if ty != v.ty || len != v.count.iter().product::<usize>() {
                    return Err(MpasError::Refusal(format!(
                        "render slab type/extent mismatch: {key}"
                    )));
                }
                if v.written[record as usize] {
                    return Err(MpasError::Refusal(format!(
                        "render slab already written: {key}"
                    )));
                }
                let mut start = vec![0; v.count.len()];
                start[0] = record as usize;
                // SAFETY: schema type and complete slab extent checked above;
                // netCDF copies the buffer synchronously without conversion.
                check(unsafe { nc_put_vara(w.id, v.id, start.as_ptr(), v.count.as_ptr(), ptr) })?;
                let width = match ty {
                    2 => 1,
                    3 => 2,
                    4 | 5 => 4,
                    6 => 8,
                    _ => unreachable!(),
                };
                v.expected[record as usize] = Some(word_digest(
                    ptr,
                    len.checked_mul(width)
                        .ok_or_else(|| MpasError::Refusal("render byte extent overflow".into()))?,
                ));
                v.written[record as usize] = true;
                Ok(())
            }
        }
    }
    pub fn put_record_text(&mut self, key: &str, record: u64, value: &str) -> MpasResult<()> {
        match self {
            Self::Classic(w) => {
                w.put_record_text(key, record, value)?;
                Ok(())
            }
            Self::Deflated(w) => {
                let width = w
                    .vars
                    .get(key)
                    .ok_or_else(|| MpasError::Refusal(format!("render variable absent: {key}")))?
                    .count
                    .iter()
                    .product::<usize>();
                if value.len() > width {
                    return Err(MpasError::Refusal(
                        "render text exceeds declared slab width".into(),
                    ));
                }
                let mut data = vec![0u8; width];
                data[..value.len()].copy_from_slice(value.as_bytes());
                self.put_record(key, record, NcData::Chars(&data))
            }
        }
    }
    pub fn finish(self) -> MpasResult<()> {
        match self {
            Self::Classic(w) => {
                w.finish()?;
                Ok(())
            }
            Self::Deflated(mut w) => {
                let missing: Vec<_> = w
                    .vars
                    .iter()
                    .filter(|(_, v)| v.written.contains(&false))
                    .map(|(n, _)| n.as_str())
                    .collect();
                if !missing.is_empty() {
                    return Err(MpasError::Refusal(format!(
                        "unwritten render slabs: {missing:?}"
                    )));
                }
                let result = unsafe { nc_close(w.id) };
                w.id = -1;
                check(result)?;
                verify_stored_words(&w)
            }
        }
    }
}

// The pointers come exclusively from NcData's five primitive slices above or
// the same primitive typed arrays below. No padding or uninitialized byte is
// present; the complete borrowed slice stays live throughout synchronous SHA.
fn word_digest(pointer: *const c_void, bytes: usize) -> [u8; 32] {
    use sha2::Digest;
    let data = unsafe { std::slice::from_raw_parts(pointer.cast::<u8>(), bytes) };
    sha2::Sha256::digest(data).into()
}
fn verify_stored_words(writer: &Deflated) -> MpasResult<()> {
    let file = netcrust::File::open(&writer.path)?;
    let mut rows = Vec::new();
    for (key, v) in &writer.vars {
        let slab = v.count.iter().product::<usize>();
        let records = if v.fixed { 1 } else { writer.records as usize };
        let mut actual = Vec::new();
        macro_rules! numeric {
            ($ty:ty) => {{
                let array = file.read_array::<$ty>(key)?;
                let data = array.as_slice().ok_or_else(|| {
                    MpasError::Refusal("render typed decode is not contiguous".into())
                })?;
                if data.len() != slab * records {
                    return Err(MpasError::Refusal(format!(
                        "render decoded extent changed: {key}"
                    )));
                }
                for part in data.chunks(slab) {
                    actual.push(word_digest(
                        part.as_ptr().cast(),
                        part.len() * std::mem::size_of::<$ty>(),
                    ));
                }
            }};
        }
        match v.ty {
            2 => {
                // The pinned HDF5 reader exposes NC_CHAR as one S1 string per
                // stored character. Its renderer already groups Times by the
                // declared DateStrLen. Rebuild only those exact stored bytes.
                let chars = file.read_strings(key)?;
                if chars.len() != slab * records
                    || chars.iter().any(|s| s.len() > 1)
                {
                    return Err(MpasError::Refusal(format!(
                        "render character decode extent changed: {key}"
                    )));
                }
                let data: Vec<u8> = chars
                    .iter()
                    .map(|s| s.as_bytes().first().copied().unwrap_or(0))
                    .collect();
                for part in data.chunks(slab) {
                    actual.push(word_digest(part.as_ptr().cast(), part.len()));
                }
            }
            3 => numeric!(i16),
            4 => numeric!(i32),
            5 => numeric!(f32),
            6 => numeric!(f64),
            _ => unreachable!(),
        }
        let expected: Vec<_> = v
            .expected
            .iter()
            .map(|d| d.expect("finish checked every slab"))
            .collect();
        if expected != actual {
            return Err(MpasError::Refusal(format!(
                "lossless render storage changed words: {key}"
            )));
        }
        rows.push(serde_json::json!({"field":key,"type":v.ty,"values":slab*records,"record_shape":v.count,"sha256_per_record":actual.iter().map(|d|crate::weights::hex(d)).collect::<Vec<_>>(),"differing_words":0}));
    }
    let target = writer.path.with_file_name(format!(
        "{}.lossless-words.json",
        writer.path.file_name().unwrap().to_string_lossy()
    ));
    let record = serde_json::json!({"kind":"actual Rust lossless render storage word roundtrip","passed":true,"field_count":rows.len(),"source_schema_values_sha256_equal_stored_decode":true,"fields":rows,"container":"netCDF4 shuffle/deflate1; no quantization","time_extent":"finite exact declared record count","file_sha256":crate::sha256_file(&writer.path)?});
    std::fs::write(
        target,
        serde_json::to_vec_pretty(&record).map_err(|e| MpasError::Refusal(e.to_string()))?,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn schema() -> (Vec<NcDim>, Vec<NcAttr>, Vec<NcVarDef>) {
        let dims = vec![NcDim::record("Time"), NcDim::fixed("x", 8)];
        let attrs = vec![
            NcAttr::text("TITLE", "lossless ☁ test"),
            NcAttr::int("n", 8),
            NcAttr::floats("f", vec![0.0, -0.0, 1.25]),
            NcAttr::doubles("d", vec![1.0f64 / 3.0]),
        ];
        let vars = [
            ("C", NcType::Char),
            ("S", NcType::Short),
            ("I", NcType::Int),
            ("F", NcType::Float),
            ("D", NcType::Double),
        ]
        .into_iter()
        .map(|(n, t)| {
            NcVarDef::new(n, t, vec![0, 1]).with_attrs(vec![NcAttr::text("units", "native")])
        })
        .collect();
        (dims, attrs, vars)
    }
    fn values(w: &mut RenderFrameWriter) {
        for r in 0..2 {
            w.put_record_text("C", r, "stamp").unwrap();
            w.put_record(
                "S",
                r,
                NcData::Shorts(&[i16::MIN, -1, 0, 1, 2, 3, 4, i16::MAX]),
            )
            .unwrap();
            w.put_record(
                "I",
                r,
                NcData::Ints(&[i32::MIN, -1, 0, 1, 2, 3, 4, i32::MAX]),
            )
            .unwrap();
            let floats = [
                0x00000000, 0x80000000, 0x00000001, 0x807fffff, 0x3f800000, 0x7f800000, 0xff800000,
                0x7fc12345,
            ]
            .map(f32::from_bits);
            w.put_record("F", r, NcData::Floats(&floats)).unwrap();
            let doubles = [
                0,
                0x8000000000000000,
                1,
                0x800fffffffffffff,
                0x3ff0000000000000,
                0x7ff0000000000000,
                0xfff0000000000000,
                0x7ff812345678abcd,
            ]
            .map(f64::from_bits);
            w.put_record("D", r, NcData::Doubles(&doubles)).unwrap();
        }
    }
    #[test]
    fn render_deflate_round_trip_preserves_special_words_schema_and_reader_compatibility() {
        let dir =
            std::path::PathBuf::from(std::env::var_os("TMPDIR").expect("own TMPDIR required"));
        let classic = dir.join(format!("lossless-classic-{}.nc", std::process::id()));
        let compressed = dir.join(format!("lossless-deflated-{}.nc", std::process::id()));
        let (d, a, v) = schema();
        let mut w = RenderFrameWriter::Classic(
            NcClassicWriter::create(
                &classic,
                NcFormat::Offset64,
                d.clone(),
                a.clone(),
                v.clone(),
                2,
            )
            .unwrap(),
        );
        values(&mut w);
        w.finish().unwrap();
        let mut w = RenderFrameWriter::create_deflated(&compressed, d, a, v, 2).unwrap();
        values(&mut w);
        w.finish().unwrap();
        let c = netcrust::File::open(&classic).unwrap();
        let z = netcrust::File::open(&compressed).unwrap();
        assert_eq!(
            c.dimension("Time").unwrap().len(),
            z.dimension("Time").unwrap().len()
        );
        assert_eq!(
            c.read_array::<f32>("F")
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            z.read_array::<f32>("F")
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            c.read_array::<f64>("D")
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            z.read_array::<f64>("D")
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            c.read_array::<i32>("I").unwrap(),
            z.read_array::<i32>("I").unwrap()
        );
        assert_eq!(
            c.read_array::<i16>("S").unwrap(),
            z.read_array::<i16>("S").unwrap()
        );
        let chars = z.read_strings("C").unwrap();
        let grouped: Vec<_> = chars.chunks(8).map(|r| r.concat()).collect();
        assert_eq!(c.read_strings("C").unwrap(), grouped);
        for key in ["TITLE", "n", "f", "d"] {
            assert_eq!(
                format!("{:?}", c.attribute(key)),
                format!("{:?}", z.attribute(key))
            );
        }
        for key in ["C", "S", "I", "F", "D"] {
            assert_eq!(
                format!("{:?}", c.variable(key).unwrap().attribute("units")),
                format!("{:?}", z.variable(key).unwrap().attribute("units"))
            );
        }
        drop(c);
        drop(z);
        std::fs::remove_file(classic).unwrap();
        std::fs::remove_file(compressed).unwrap();
    }
    #[test]
    fn render_deflate_rejects_incomplete_mistyped_duplicate_and_overrun_slabs() {
        let dir =
            std::path::PathBuf::from(std::env::var_os("TMPDIR").expect("own TMPDIR required"));
        let path = dir.join(format!("lossless-negative-{}.nc", std::process::id()));
        let (d, a, v) = schema();
        let mut w = RenderFrameWriter::create_deflated(&path, d, a, v, 2).unwrap();
        assert!(w.put_record("F", 0, NcData::Ints(&[0; 8])).is_err());
        assert!(w.put_record("F", 0, NcData::Floats(&[0.0; 7])).is_err());
        assert!(w.put_record("F", 2, NcData::Floats(&[0.0; 8])).is_err());
        assert!(
            w.put_record("absent", 0, NcData::Floats(&[0.0; 8]))
                .is_err()
        );
        assert!(w.put_record_text("C", 0, "123456789").is_err());
        w.put_record("F", 0, NcData::Floats(&[0.0; 8])).unwrap();
        assert!(w.put_record("F", 0, NcData::Floats(&[0.0; 8])).is_err());
        assert!(w.finish().is_err());
        std::fs::remove_file(path).unwrap();
    }
}
