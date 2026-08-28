//! Grading a produced lbc file against a native one.
//!
//! The pin's shape, stated once: **byte-exact where integer or coordinate,
//! ULP-characterized where float**.  Every variable gets a measured row —
//! element count, differing elements, max ULP distance, max absolute
//! difference and where it happened — never a wave-off.  The header is
//! compared field-by-field so the *expected* identity deltas (`file_id`,
//! `history`, the trailing `gpuwm_provenance`) are named by measurement
//! rather than assumed, and anything else differing is loud.
//!
//! ## Why this walks the bytes itself
//! The comparison reads both files through its own CDF-5 header walk and
//! compares each variable's **raw slab bytes** straight off the disk, rather
//! than through a reading library's numeric promotion.  Byte-exactness is the
//! claim being graded, so the instrument must observe bytes; a promotion to
//! f64 and back would be one more place for the instrument itself to lie.

use std::path::Path;

use crate::error::{MpasError, MpasResult};

/// One variable's measured agreement.
#[derive(Debug, Default, serde::Serialize)]
pub struct VarDiff {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub elements: usize,
    /// Elements whose stored bit patterns differ (0.0 vs -0.0 counts here).
    pub bits_differing: usize,
    pub byte_exact: bool,
    pub max_ulp: u64,
    pub max_abs: f64,
    /// Largest |a-b| / max(|a|,|b|) over differing elements.
    pub max_rel: f64,
    /// Where the max-ULP element sits, decoded through the variable's own
    /// dimensions (e.g. `nCells=123 nVertLevels=4`).
    pub max_at: Option<String>,
    pub ours_sample_at_max: Option<f64>,
    pub native_sample_at_max: Option<f64>,
}

/// Header agreement, by attribute name.
#[derive(Debug, Default, serde::Serialize)]
pub struct HeaderDiff {
    pub missing_in_ours: Vec<String>,
    pub extra_in_ours: Vec<String>,
    pub value_differs: Vec<String>,
    pub equal: usize,
}

#[derive(Debug, Default, serde::Serialize)]
pub struct CompareReport {
    pub ours: String,
    pub native: String,
    pub dims_equal: bool,
    pub dim_notes: Vec<String>,
    pub variables: Vec<VarDiff>,
    pub header: HeaderDiff,
    /// True when every variable row is byte-exact.
    pub payload_byte_exact: bool,
}

/// Monotone key for ULP distance; +0.0 and -0.0 both map to 0.
fn ulp_key(f: f32) -> i64 {
    let b = f.to_bits();
    if b & 0x8000_0000 != 0 {
        -((b & 0x7fff_ffff) as i64)
    } else {
        b as i64
    }
}

// ---------------------------------------------------------------------------
// A minimal CDF-5 header walk.  Field widths follow the pnetcdf CDF-5
// specification: every NON_NEG (counts, name lengths, dim lengths, dimids,
// vsize) and every OFFSET (begin) is a big-endian 64-bit integer; type tags
// stay 32-bit.  This is the same convention `rw_store::netcdf_classic`
// writes.
// ---------------------------------------------------------------------------

const NC_DIMENSION: u32 = 0x0A;
const NC_VARIABLE: u32 = 0x0B;
const NC_ATTRIBUTE: u32 = 0x0C;

#[derive(Debug)]
struct RawAttr {
    name: String,
    nc_type: u32,
    value: Vec<u8>,
}

#[derive(Debug)]
struct RawVar {
    name: String,
    dimids: Vec<usize>,
    #[allow(dead_code)]
    attrs: Vec<RawAttr>,
    nc_type: u32,
    vsize: u64,
    begin: u64,
}

#[derive(Debug)]
struct RawFile {
    bytes: Vec<u8>,
    numrecs: u64,
    dims: Vec<(String, u64)>,
    record_dim: Option<usize>,
    gattrs: Vec<RawAttr>,
    vars: Vec<RawVar>,
    record_stride: u64,
}

struct Walk<'a> {
    b: &'a [u8],
    at: usize,
    path: &'a Path,
}

impl<'a> Walk<'a> {
    fn take(&mut self, n: usize, what: &str) -> MpasResult<&'a [u8]> {
        if self.at + n > self.b.len() {
            return Err(MpasError::Refusal(format!(
                "{} ends inside its own header while reading {what}",
                self.path.display()
            )));
        }
        let out = &self.b[self.at..self.at + n];
        self.at += n;
        Ok(out)
    }
    fn u32(&mut self, what: &str) -> MpasResult<u32> {
        let r = self.take(4, what)?;
        Ok(u32::from_be_bytes([r[0], r[1], r[2], r[3]]))
    }
    fn u64(&mut self, what: &str) -> MpasResult<u64> {
        let r = self.take(8, what)?;
        Ok(u64::from_be_bytes([
            r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7],
        ]))
    }
    fn name(&mut self) -> MpasResult<String> {
        let len = self.u64("a name length")? as usize;
        let raw = self.take(len, "a name")?.to_vec();
        let pad = (4 - (len % 4)) % 4;
        self.take(pad, "name padding")?;
        String::from_utf8(raw).map_err(|_| {
            MpasError::Refusal(format!("{} carries a non-UTF-8 name", self.path.display()))
        })
    }
    fn attr_list(&mut self) -> MpasResult<Vec<RawAttr>> {
        let tag = self.u32("an attribute list tag")?;
        let nelems = self.u64("an attribute count")?;
        if tag != NC_ATTRIBUTE {
            if tag == 0 && nelems == 0 {
                return Ok(Vec::new());
            }
            return Err(MpasError::Refusal(format!(
                "{} has attribute-list tag {tag:#x}",
                self.path.display()
            )));
        }
        let mut out = Vec::with_capacity(nelems as usize);
        for _ in 0..nelems {
            let name = self.name()?;
            let nc_type = self.u32("an attribute type")?;
            let n = self.u64("an attribute element count")?;
            let elem = type_size(nc_type).ok_or_else(|| {
                MpasError::Refusal(format!(
                    "{} attribute {name} has unknown type {nc_type}",
                    self.path.display()
                ))
            })?;
            let bytes = (n * elem) as usize;
            let value = self.take(bytes, "attribute values")?.to_vec();
            let pad = (4 - (bytes % 4)) % 4;
            self.take(pad, "attribute padding")?;
            out.push(RawAttr {
                name,
                nc_type,
                value,
            });
        }
        Ok(out)
    }
}

fn type_size(nc_type: u32) -> Option<u64> {
    Some(match nc_type {
        1 | 2 | 7 => 1,       // byte, char, ubyte
        3 | 8 => 2,           // short, ushort
        4 | 5 | 9 => 4,       // int, float, uint
        6 | 10 | 11 => 8,     // double, int64, uint64
        _ => return None,
    })
}

fn type_name(nc_type: u32) -> &'static str {
    match nc_type {
        1 => "byte",
        2 => "char",
        3 => "short",
        4 => "int",
        5 => "float",
        6 => "double",
        7 => "ubyte",
        8 => "ushort",
        9 => "uint",
        10 => "int64",
        11 => "uint64",
        _ => "unknown",
    }
}

fn parse_cdf5(path: &Path) -> MpasResult<RawFile> {
    let bytes = std::fs::read(path)?;
    parse_cdf5_bytes(path, bytes)
}

fn parse_cdf5_bytes(path: &Path, bytes: Vec<u8>) -> MpasResult<RawFile> {
    if bytes.len() < 8 || &bytes[0..3] != b"CDF" {
        return Err(MpasError::Refusal(format!(
            "{} is not a classic netCDF file",
            path.display()
        )));
    }
    if bytes[3] != 5 {
        return Err(MpasError::Refusal(format!(
            "{} is CDF version {}; this comparison walks CDF-5 (the format the native lbc \
             stream and this producer both write)",
            path.display(),
            bytes[3]
        )));
    }
    let mut w = Walk {
        b: &bytes,
        at: 4,
        path,
    };
    let numrecs = w.u64("numrecs")?;
    let tag = w.u32("the dimension list tag")?;
    let ndims = w.u64("the dimension count")?;
    let mut dims = Vec::new();
    let mut record_dim = None;
    if tag == NC_DIMENSION {
        for i in 0..ndims as usize {
            let name = w.name()?;
            let len = w.u64("a dimension length")?;
            if len == 0 {
                record_dim = Some(i);
            }
            dims.push((name, len));
        }
    }
    let gattrs = w.attr_list()?;
    let tag = w.u32("the variable list tag")?;
    let nvars = w.u64("the variable count")?;
    let mut vars = Vec::new();
    if tag == NC_VARIABLE {
        for _ in 0..nvars {
            let name = w.name()?;
            let nd = w.u64("a dimension count")? as usize;
            let mut dimids = Vec::with_capacity(nd);
            for _ in 0..nd {
                dimids.push(w.u64("a dimension id")? as usize);
            }
            let attrs = w.attr_list()?;
            let nc_type = w.u32("a variable type")?;
            let vsize = w.u64("a vsize")?;
            let begin = w.u64("a begin offset")?;
            vars.push(RawVar {
                name,
                dimids,
                attrs,
                nc_type,
                vsize,
                begin,
            });
        }
    }
    let record_stride: u64 = vars
        .iter()
        .filter(|v| record_dim.is_some() && v.dimids.first() == record_dim.as_ref())
        .map(|v| v.vsize)
        .sum();
    Ok(RawFile {
        bytes,
        numrecs,
        dims,
        record_dim,
        gattrs,
        vars,
        record_stride,
    })
}

impl RawFile {
    fn is_record(&self, v: &RawVar) -> bool {
        self.record_dim.is_some() && v.dimids.first() == self.record_dim.as_ref()
    }

    /// The variable's data bytes in element order, records concatenated,
    /// per-slab padding excluded.
    fn slab(&self, v: &RawVar, path: &Path) -> MpasResult<Vec<u8>> {
        let elem = type_size(v.nc_type).ok_or_else(|| {
            MpasError::Refusal(format!("variable {} has unknown type {}", v.name, v.nc_type))
        })?;
        let mut per_slab: u64 = elem;
        for (i, &d) in v.dimids.iter().enumerate() {
            if self.is_record(v) && i == 0 {
                continue;
            }
            per_slab *= self.dims[d].1;
        }
        let records = if self.is_record(v) { self.numrecs } else { 1 };
        let stride = if self.is_record(v) {
            self.record_stride
        } else {
            0
        };
        let mut out = Vec::with_capacity((per_slab * records) as usize);
        for r in 0..records {
            let start = (v.begin + r * stride) as usize;
            let end = start + per_slab as usize;
            if end > self.bytes.len() {
                return Err(MpasError::Refusal(format!(
                    "variable {} record {r} runs past the end of {}",
                    v.name,
                    path.display()
                )));
            }
            out.extend_from_slice(&self.bytes[start..end]);
        }
        Ok(out)
    }

    fn shape(&self, v: &RawVar) -> Vec<u64> {
        v.dimids
            .iter()
            .enumerate()
            .map(|(i, &d)| {
                if self.is_record(v) && i == 0 {
                    self.numrecs
                } else {
                    self.dims[d].1
                }
            })
            .collect()
    }
}

fn locate(shape: &[u64], dim_names: &[String], flat: usize) -> String {
    let mut rem = flat as u64;
    let mut strides = vec![1u64; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    let mut parts = Vec::new();
    for (i, &s) in strides.iter().enumerate() {
        let idx = rem / s;
        rem %= s;
        parts.push(format!(
            "{}={}",
            dim_names.get(i).map(String::as_str).unwrap_or("?"),
            idx
        ));
    }
    parts.join(" ")
}

fn attr_render(a: &RawAttr) -> String {
    if a.nc_type == 2 {
        format!("{:?}", String::from_utf8_lossy(&a.value))
    } else {
        format!("type{} {:02x?}", a.nc_type, a.value)
    }
}

/// Read a character variable's first record as text.
///
/// The promoting reader this crate uses elsewhere handles every numeric type
/// and no character one, so a stream's `xtime` — the only thing in these files
/// that is text rather than numbers — has to come from the header walk above.
/// It reads the header from the front of the file and then only the bytes the
/// variable actually occupies, rather than the whole file, because the caller
/// is about to read that file again for its fields.
///
/// `Ok(None)` means the file is a well-formed classic file that carries no
/// such character variable.  An `Err` means it is not a classic file at all.
pub(crate) fn read_char_variable(path: &Path, name: &str) -> MpasResult<Option<String>> {
    // Big enough for any MPAS header seen so far; a walk that runs off the end
    // falls back to the whole file rather than guessing.
    const HEADER_PROBE_BYTES: usize = 4 * 1024 * 1024;
    let raw = match parse_cdf5_prefix(path, HEADER_PROBE_BYTES) {
        Ok(raw) => raw,
        Err(_) => parse_cdf5(path)?,
    };
    let Some(var) = raw.vars.iter().find(|v| v.name == name) else {
        return Ok(None);
    };
    if var.nc_type != 2 {
        return Ok(None);
    }
    // One record's worth: every dimension but the record dimension.
    let mut count: u64 = 1;
    for &d in &var.dimids {
        if Some(d) != raw.record_dim {
            count *= raw.dims[d].1;
        }
    }
    if count == 0 {
        return Ok(None);
    }
    let mut buffer = vec![0u8; count as usize];
    use std::io::{Read, Seek, SeekFrom};
    let mut file = std::fs::File::open(path)?;
    file.seek(SeekFrom::Start(var.begin))?;
    file.read_exact(&mut buffer)?;
    Ok(Some(String::from_utf8_lossy(&buffer).to_string()))
}

/// [`parse_cdf5`] over the first `limit` bytes of the file, for callers that
/// want the header and one small variable rather than the whole thing.
fn parse_cdf5_prefix(path: &Path, limit: usize) -> MpasResult<RawFile> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut bytes = Vec::with_capacity(limit);
    file.by_ref()
        .take(limit as u64)
        .read_to_end(&mut bytes)?;
    parse_cdf5_bytes(path, bytes)
}

/// Compare two lbc files variable-by-variable and header field-by-field.
pub fn compare_lbc(ours_path: &Path, native_path: &Path) -> MpasResult<CompareReport> {
    let ours = parse_cdf5(ours_path)?;
    let native = parse_cdf5(native_path)?;

    let mut report = CompareReport {
        ours: ours_path.display().to_string(),
        native: native_path.display().to_string(),
        dims_equal: true,
        ..Default::default()
    };

    if ours.numrecs != native.numrecs {
        report.dims_equal = false;
        report.dim_notes.push(format!(
            "ours holds {} record(s), native {}",
            ours.numrecs, native.numrecs
        ));
    }
    if ours.dims.len() != native.dims.len() {
        report.dims_equal = false;
        report.dim_notes.push(format!(
            "ours declares {} dimension(s), native {}",
            ours.dims.len(),
            native.dims.len()
        ));
    }
    for (a, b) in ours.dims.iter().zip(native.dims.iter()) {
        if a != b {
            report.dims_equal = false;
            report.dim_notes.push(format!(
                "dimension order differs: ours {}={} vs native {}={}",
                a.0, a.1, b.0, b.1
            ));
        }
    }

    let mut all_byte_exact = true;
    for nv in &native.vars {
        let Some(ov) = ours.vars.iter().find(|v| v.name == nv.name) else {
            return Err(MpasError::Refusal(format!(
                "ours carries no variable {}; the comparison is meaningless on different \
                 containers",
                nv.name
            )));
        };
        let n_shape = native.shape(nv);
        let o_shape = ours.shape(ov);
        if n_shape != o_shape || nv.nc_type != ov.nc_type {
            return Err(MpasError::Refusal(format!(
                "{} is {} {:?} in ours and {} {:?} in native",
                nv.name,
                type_name(ov.nc_type),
                o_shape,
                type_name(nv.nc_type),
                n_shape
            )));
        }
        let a = ours.slab(ov, ours_path)?;
        let b = native.slab(nv, native_path)?;
        let dim_names: Vec<String> = nv
            .dimids
            .iter()
            .map(|&d| native.dims[d].0.clone())
            .collect();

        let elem = type_size(nv.nc_type).unwrap() as usize;
        let elements = a.len() / elem;
        let mut row = VarDiff {
            name: nv.name.clone(),
            dtype: type_name(nv.nc_type).to_string(),
            shape: n_shape.clone(),
            elements,
            ..Default::default()
        };

        let mut max_ulp: u64 = 0;
        let mut max_abs: f64 = 0.0;
        let mut max_rel: f64 = 0.0;
        let mut max_idx: Option<usize> = None;
        let mut bits_differing = 0usize;
        for i in 0..elements {
            let ab = &a[i * elem..(i + 1) * elem];
            let bb = &b[i * elem..(i + 1) * elem];
            if ab == bb {
                continue;
            }
            bits_differing += 1;
            if nv.nc_type == 5 {
                let fa = f32::from_be_bytes([ab[0], ab[1], ab[2], ab[3]]);
                let fb = f32::from_be_bytes([bb[0], bb[1], bb[2], bb[3]]);
                let ulp = (ulp_key(fa) - ulp_key(fb)).unsigned_abs();
                let abs = (fa as f64 - fb as f64).abs();
                let denom = (fa as f64).abs().max((fb as f64).abs());
                if denom > 0.0 {
                    max_rel = max_rel.max(abs / denom);
                }
                if ulp > max_ulp {
                    max_ulp = ulp;
                    max_idx = Some(i);
                    row.ours_sample_at_max = Some(fa as f64);
                    row.native_sample_at_max = Some(fb as f64);
                }
                max_abs = max_abs.max(abs);
            } else if max_idx.is_none() {
                max_idx = Some(i);
            }
        }
        row.bits_differing = bits_differing;
        row.byte_exact = bits_differing == 0;
        row.max_ulp = max_ulp;
        row.max_abs = max_abs;
        row.max_rel = max_rel;
        if let Some(i) = max_idx {
            row.max_at = Some(locate(&n_shape, &dim_names, i));
        }
        all_byte_exact &= row.byte_exact;
        report.variables.push(row);
    }
    report.payload_byte_exact = all_byte_exact;

    // Global attributes, by name.
    let render =
        |attrs: &[RawAttr]| -> std::collections::BTreeMap<String, String> {
            attrs
                .iter()
                .map(|a| (a.name.clone(), attr_render(a)))
                .collect()
        };
    let oa_by = render(&ours.gattrs);
    let na_by = render(&native.gattrs);
    for (name, nval) in &na_by {
        match oa_by.get(name) {
            None => report.header.missing_in_ours.push(name.clone()),
            Some(oval) if oval != nval => report.header.value_differs.push(name.clone()),
            Some(_) => report.header.equal += 1,
        }
    }
    for name in oa_by.keys() {
        if !na_by.contains_key(name) {
            report.header.extra_in_ours.push(name.clone());
        }
    }

    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ulp_distance_is_zero_across_signed_zero_and_one_across_neighbours() {
        assert_eq!(ulp_key(0.0) - ulp_key(-0.0), 0);
        let one = 1.0f32;
        let next = f32::from_bits(one.to_bits() + 1);
        assert_eq!((ulp_key(next) - ulp_key(one)).unsigned_abs(), 1);
        // Distance is well-defined across the sign boundary.
        let tiny_neg = f32::from_bits(0x8000_0001);
        let tiny_pos = f32::from_bits(0x0000_0001);
        assert_eq!((ulp_key(tiny_pos) - ulp_key(tiny_neg)).unsigned_abs(), 2);
    }

    #[test]
    fn locate_decodes_a_flat_index_through_the_dims() {
        let shape = [1u64, 10, 5];
        let names = vec![
            "Time".to_string(),
            "nCells".to_string(),
            "nVertLevels".to_string(),
        ];
        assert_eq!(locate(&shape, &names, 0), "Time=0 nCells=0 nVertLevels=0");
        assert_eq!(
            locate(&shape, &names, 5 * 3 + 2),
            "Time=0 nCells=3 nVertLevels=2"
        );
    }

    #[test]
    fn every_classic_type_width_is_declared_or_refused() {
        assert_eq!(type_size(2), Some(1));
        assert_eq!(type_size(5), Some(4));
        assert_eq!(type_size(6), Some(8));
        assert_eq!(type_size(12), None, "compound types are not classic");
    }
}
