//! Dependency-free NetCDF "classic 64-bit offset" (CDF-2) writer.
//!
//! rusty-weather stores weather data in a custom per-hour format. This module
//! emits a minimal but standards-correct CDF-2 container so an exported hour
//! can be opened by every scientific stack (xarray/scipy, MetPy, Panoply,
//! `ncdump`). NetCDF4 would mean writing HDF5; the classic CDF-2 format is a
//! simple big-endian byte container, so we implement it directly against the
//! Unidata "classic format" specification — no new crate dependencies.
//!
//! Scope deliberately narrow: fixed-size dimensions only (NO record/unlimited
//! dimension — `numrecs` is always 0), variables are always `NC_FLOAT`, and
//! attributes are either text (`NC_CHAR`) or `f32` (`NC_FLOAT`). That covers
//! everything the hour exporter (Task 6) needs.
//!
//! ## Byte layout emitted (everything BIG-ENDIAN)
//! ```text
//! header   := magic numrecs dim_list gatt_list var_list
//! magic    := 'C' 'D' 'F' 0x02
//! numrecs  := u32 = 0
//! dim_list := ABSENT | NC_DIMENSION(0x0A) nelems [dim ...]
//!   dim    := name len(u32 > 0)
//! att_list := ABSENT | NC_ATTRIBUTE(0x0C) nelems [attr ...]
//!   attr   := name nc_type(u32) nelems(u32) [values, zero-padded to 4]
//! var_list := ABSENT | NC_VARIABLE(0x0B) nelems [var ...]
//!   var    := name ndims(u32) [dimid u32 × ndims] vatt_list
//!             nc_type(u32) vsize(u32) begin(u64)   // u64 begin is the CDF-2 part
//! ABSENT   := 0x0000_0000 0x0000_0000
//! name     := nelems(u32 byte length, NOT counting padding) bytes pad-to-4
//! data     := each var's values contiguous at `begin`, row-major, BE, pad-to-4
//! ```
//!
//! `vsize` = product(dim lens) × 4, rounded up to a multiple of 4 (already
//! aligned for f32). If the true value would exceed `u32::MAX - 3` it is
//! clamped to `u32::MAX` — the spec documents `vsize` as a buggy field that
//! readers recompute from the dimensions.
//!
//! `begin` offsets are computed with a two-pass serialization: the header is
//! serialized once with placeholder begins to measure its exact length, then
//! `begin_i = header_len + Σ_{j<i} vsize64_j` (using the true u64 byte size for
//! layout, not the clamped u32 `vsize`), then re-serialized with the real
//! offsets. The begin fields are fixed width, so the header length is identical
//! across passes — the writer asserts this.
//!
//! ## Crash-safety
//! [`Nc3Writer::create`] writes straight to the destination path through a
//! [`BufWriter`], and [`Nc3Writer::finish`] flushes + `fsync`s before
//! returning. We deliberately do NOT use the temp-file + atomic-rename
//! discipline of [`crate::atomic`]: that module materializes its payload up
//! front (or drives a closure that owns the whole write), whereas this writer
//! streams an arbitrarily large data section (hundreds of MB) variable by
//! variable while the caller pulls each volume from the store, so holding it
//! all to swap at the end is the wrong shape. The export tool writes to its
//! chosen destination directly; on failure the partial `.nc` is the caller's
//! to discard. This matches typical NetCDF tooling, which also writes in place.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::error::{RwResult, RwStoreError};

/// NetCDF external type tag: 8-bit characters (text attributes).
pub const NC_CHAR: u32 = 2;
/// NetCDF external type tag: IEEE single-precision float (all data vars).
pub const NC_FLOAT: u32 = 5;

// Component tags from the classic format grammar.
const NC_DIMENSION: u32 = 0x0000_000A;
const NC_VARIABLE: u32 = 0x0000_000B;
const NC_ATTRIBUTE: u32 = 0x0000_000C;
const ABSENT_TAG: u32 = 0x0000_0000;

/// An attribute value: either UTF-8 text (`NC_CHAR`) or a vector of `f32`
/// (`NC_FLOAT`). These are the only two attribute kinds the exporter needs.
#[derive(Debug, Clone, PartialEq)]
pub enum Nc3AttrValue {
    Text(String),
    Floats(Vec<f32>),
}

/// A single named attribute (global, or attached to a variable).
#[derive(Debug, Clone, PartialEq)]
pub struct Nc3Attr {
    pub name: String,
    pub value: Nc3AttrValue,
}

impl Nc3Attr {
    /// Convenience constructor for a text attribute.
    pub fn text(name: impl Into<String>, value: impl Into<String>) -> Self {
        Nc3Attr {
            name: name.into(),
            value: Nc3AttrValue::Text(value.into()),
        }
    }

    /// Convenience constructor for an f32 attribute.
    pub fn floats(name: impl Into<String>, value: Vec<f32>) -> Self {
        Nc3Attr {
            name: name.into(),
            value: Nc3AttrValue::Floats(value),
        }
    }
}

/// A fixed-size dimension. Its index in the `dims` vector passed to
/// [`Nc3Writer::create`] is its dimid.
#[derive(Debug, Clone)]
pub struct Nc3Dim {
    pub name: String,
    pub len: usize,
}

/// A variable definition. All variables are `NC_FLOAT`. `dimids` indexes into
/// the `dims` vector (row-major: the last dimid varies fastest in the data).
#[derive(Debug, Clone)]
pub struct Nc3VarDef {
    pub name: String,
    pub dimids: Vec<usize>,
    pub attrs: Vec<Nc3Attr>,
}

/// Streaming CDF-2 writer. Created with the full schema (dims, global attrs,
/// var defs); the caller then pushes each variable's data once, in definition
/// order, via [`write_var`](Nc3Writer::write_var), and ends with
/// [`finish`](Nc3Writer::finish).
#[derive(Debug)]
pub struct Nc3Writer {
    out: BufWriter<File>,
    /// Per-var element count (product of its dim lens); index = var order.
    var_elems: Vec<u64>,
    /// Per-var name, for contextual error messages.
    var_names: Vec<String>,
    /// How many vars have been written so far.
    written: usize,
}

/// Round `n` up to the next multiple of 4.
#[inline]
fn pad4(n: u64) -> u64 {
    (n + 3) & !3
}

impl Nc3Writer {
    pub fn create(
        path: &Path,
        dims: Vec<Nc3Dim>,
        gattrs: Vec<Nc3Attr>,
        vars: Vec<Nc3VarDef>,
    ) -> RwResult<Self> {
        validate_defs(&dims, &gattrs, &vars)?;

        // Per-var element counts (checked product of dim lens).
        let mut var_elems: Vec<u64> = Vec::with_capacity(vars.len());
        for var in &vars {
            let mut elems: u64 = 1;
            for &dimid in &var.dimids {
                let len = dims[dimid].len as u64;
                elems = elems.checked_mul(len).ok_or_else(|| {
                    RwStoreError::Format(format!(
                        "netcdf3: variable '{}' element count overflows u64",
                        var.name
                    ))
                })?;
            }
            var_elems.push(elems);
        }

        // Two-pass header serialization. Pass 1 measures the header length with
        // placeholder begins; pass 2 emits the real begins. Begin fields are
        // fixed-width (u64), so both passes produce identical lengths.
        let placeholder_begins = vec![0u64; vars.len()];
        let header_pass1 = serialize_header(&dims, &gattrs, &vars, &var_elems, &placeholder_begins);
        let header_len = header_pass1.len() as u64;

        let mut begins: Vec<u64> = Vec::with_capacity(vars.len());
        let mut cursor = header_len;
        for &elems in &var_elems {
            begins.push(cursor);
            // True byte size for layout: elems × 4, padded to 4 (f32 already
            // aligned). Use checked math — a var can be ~640 MB legitimately.
            let bytes = elems.checked_mul(4).ok_or_else(|| {
                RwStoreError::Format("netcdf3: variable byte size overflows u64".to_string())
            })?;
            let padded = pad4(bytes);
            cursor = cursor.checked_add(padded).ok_or_else(|| {
                RwStoreError::Format("netcdf3: data section size overflows u64".to_string())
            })?;
        }

        let header_pass2 = serialize_header(&dims, &gattrs, &vars, &var_elems, &begins);
        debug_assert_eq!(
            header_pass2.len() as u64,
            header_len,
            "netcdf3: header length changed between passes"
        );

        let file = File::create(path)?;
        let mut out = BufWriter::with_capacity(1 << 20, file);
        out.write_all(&header_pass2)?;

        let var_names = vars.iter().map(|v| v.name.clone()).collect();
        Ok(Nc3Writer {
            out,
            var_elems,
            var_names,
            written: 0,
        })
    }

    pub fn write_var(&mut self, values: &[f32]) -> RwResult<()> {
        if self.written >= self.var_elems.len() {
            return Err(RwStoreError::Format(format!(
                "netcdf3: write_var called {} time(s) but only {} variable(s) were defined",
                self.written + 1,
                self.var_elems.len()
            )));
        }
        let expected = self.var_elems[self.written];
        if values.len() as u64 != expected {
            return Err(RwStoreError::Format(format!(
                "netcdf3: variable '{}' expects {} value(s) but got {}",
                self.var_names[self.written],
                expected,
                values.len()
            )));
        }

        // Big-endian f32, contiguous. f32 data is already 4-aligned so no
        // trailing pad bytes are needed.
        let mut buf = Vec::with_capacity(values.len() * 4);
        for &v in values {
            buf.extend_from_slice(&v.to_be_bytes());
        }
        self.out.write_all(&buf)?;
        self.written += 1;
        Ok(())
    }

    pub fn finish(mut self) -> RwResult<()> {
        if self.written != self.var_elems.len() {
            return Err(RwStoreError::Format(format!(
                "netcdf3: finish called after writing {} of {} variable(s)",
                self.written,
                self.var_elems.len()
            )));
        }
        self.out.flush()?;
        let file = self
            .out
            .into_inner()
            .map_err(|err| RwStoreError::Io(err.into_error()))?;
        file.sync_all()?;
        Ok(())
    }
}

/// First char must be a letter or underscore; the rest are restricted to the
/// NC-safe subset `[A-Za-z0-9_+.@-]`. Rejects (never renames) on violation.
///
/// `pub(crate)` so the export glue (`export.rs`) checks variable names against
/// the exact same rule the writer enforces, with no second copy to drift.
pub(crate) fn name_is_valid(name: &str) -> bool {
    let mut chars = name.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '+' | '.' | '@' | '-'))
}

fn check_name(kind: &str, name: &str) -> RwResult<()> {
    if name.is_empty() {
        return Err(RwStoreError::Format(format!(
            "netcdf3: {kind} name is empty"
        )));
    }
    if !name_is_valid(name) {
        return Err(RwStoreError::Format(format!(
            "netcdf3: {kind} name '{name}' is not NC-safe (must start [A-Za-z_], rest [A-Za-z0-9_+.@-])"
        )));
    }
    Ok(())
}

fn check_attrs(scope: &str, attrs: &[Nc3Attr]) -> RwResult<()> {
    let mut seen: Vec<&str> = Vec::with_capacity(attrs.len());
    for attr in attrs {
        check_name("attribute", &attr.name)?;
        if seen.contains(&attr.name.as_str()) {
            return Err(RwStoreError::Format(format!(
                "netcdf3: duplicate attribute name '{}' in {scope}",
                attr.name
            )));
        }
        seen.push(&attr.name);
    }
    Ok(())
}

fn validate_defs(dims: &[Nc3Dim], gattrs: &[Nc3Attr], vars: &[Nc3VarDef]) -> RwResult<()> {
    // Dimensions: NC-safe names, len > 0 (no record dim), unique names.
    let mut dim_names: Vec<&str> = Vec::with_capacity(dims.len());
    for dim in dims {
        check_name("dimension", &dim.name)?;
        if dim.len == 0 {
            return Err(RwStoreError::Format(format!(
                "netcdf3: dimension '{}' has length 0 (record/unlimited dims unsupported)",
                dim.name
            )));
        }
        if dim_names.contains(&dim.name.as_str()) {
            return Err(RwStoreError::Format(format!(
                "netcdf3: duplicate dimension name '{}'",
                dim.name
            )));
        }
        dim_names.push(&dim.name);
    }

    check_attrs("global attributes", gattrs)?;

    // Variables: NC-safe names, dimids in range, unique names, valid attrs.
    let mut var_names: Vec<&str> = Vec::with_capacity(vars.len());
    for var in vars {
        check_name("variable", &var.name)?;
        if var_names.contains(&var.name.as_str()) {
            return Err(RwStoreError::Format(format!(
                "netcdf3: duplicate variable name '{}'",
                var.name
            )));
        }
        var_names.push(&var.name);
        for &dimid in &var.dimids {
            if dimid >= dims.len() {
                return Err(RwStoreError::Format(format!(
                    "netcdf3: variable '{}' references dimid {} but only {} dimension(s) exist",
                    var.name,
                    dimid,
                    dims.len()
                )));
            }
        }
        check_attrs(&format!("variable '{}'", var.name), &var.attrs)?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Header serialization (big-endian byte emission against the CDF-2 grammar).
// ---------------------------------------------------------------------------

fn put_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_be_bytes());
}

fn put_u64(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_be_bytes());
}

/// Emit `name := nelems(u32) bytes pad-to-4`. nelems is the byte length and
/// does NOT include the padding.
fn put_name(buf: &mut Vec<u8>, name: &str) {
    let bytes = name.as_bytes();
    put_u32(buf, bytes.len() as u32);
    buf.extend_from_slice(bytes);
    while buf.len() % 4 != 0 {
        buf.push(0);
    }
}

/// Emit one attribute: name nc_type(u32) nelems(u32) values pad-to-4.
fn put_attr(buf: &mut Vec<u8>, attr: &Nc3Attr) {
    put_name(buf, &attr.name);
    match &attr.value {
        Nc3AttrValue::Text(s) => {
            let bytes = s.as_bytes();
            put_u32(buf, NC_CHAR);
            // For NC_CHAR, nelems = byte count of the string (no NUL).
            put_u32(buf, bytes.len() as u32);
            buf.extend_from_slice(bytes);
            while buf.len() % 4 != 0 {
                buf.push(0);
            }
        }
        Nc3AttrValue::Floats(vals) => {
            put_u32(buf, NC_FLOAT);
            // For NC_FLOAT, nelems = number of values (not bytes).
            put_u32(buf, vals.len() as u32);
            for &v in vals {
                buf.extend_from_slice(&v.to_be_bytes());
            }
            // f32 values are 4-aligned; pad anyway for grammar uniformity.
            while buf.len() % 4 != 0 {
                buf.push(0);
            }
        }
    }
}

/// Emit an att_list (global or per-var). Empty ⇒ ABSENT (0x0 tag, 0x0 nelems).
fn put_att_list(buf: &mut Vec<u8>, attrs: &[Nc3Attr]) {
    if attrs.is_empty() {
        put_u32(buf, ABSENT_TAG);
        put_u32(buf, 0);
        return;
    }
    put_u32(buf, NC_ATTRIBUTE);
    put_u32(buf, attrs.len() as u32);
    for attr in attrs {
        put_attr(buf, attr);
    }
}

/// Clamp the true padded byte size into the u32 `vsize` field per the spec.
fn vsize_field(elems: u64) -> u32 {
    // bytes = elems * 4, padded to 4 (already aligned for f32).
    let bytes = elems.saturating_mul(4);
    // Use saturating arithmetic instead of pad4() to avoid a debug-mode panic
    // (or release-mode wraparound to 0) when bytes ≈ u64::MAX.
    let padded = bytes.saturating_add(3) & !3; // was: pad4(bytes)
    if padded > (u32::MAX as u64) - 3 {
        u32::MAX
    } else {
        padded as u32
    }
}

/// Serialize the full header given per-var element counts and begin offsets.
fn serialize_header(
    dims: &[Nc3Dim],
    gattrs: &[Nc3Attr],
    vars: &[Nc3VarDef],
    var_elems: &[u64],
    begins: &[u64],
) -> Vec<u8> {
    let mut buf = Vec::with_capacity(256);

    // magic
    buf.extend_from_slice(b"CDF");
    buf.push(2);
    // numrecs (always 0 — no record dimension)
    put_u32(&mut buf, 0);

    // dim_list
    if dims.is_empty() {
        put_u32(&mut buf, ABSENT_TAG);
        put_u32(&mut buf, 0);
    } else {
        put_u32(&mut buf, NC_DIMENSION);
        put_u32(&mut buf, dims.len() as u32);
        for dim in dims {
            put_name(&mut buf, &dim.name);
            put_u32(&mut buf, dim.len as u32);
        }
    }

    // gatt_list
    put_att_list(&mut buf, gattrs);

    // var_list
    if vars.is_empty() {
        put_u32(&mut buf, ABSENT_TAG);
        put_u32(&mut buf, 0);
    } else {
        put_u32(&mut buf, NC_VARIABLE);
        put_u32(&mut buf, vars.len() as u32);
        for (i, var) in vars.iter().enumerate() {
            put_name(&mut buf, &var.name);
            put_u32(&mut buf, var.dimids.len() as u32);
            for &dimid in &var.dimids {
                put_u32(&mut buf, dimid as u32);
            }
            put_att_list(&mut buf, &var.attrs);
            put_u32(&mut buf, NC_FLOAT);
            put_u32(&mut buf, vsize_field(var_elems[i]));
            put_u64(&mut buf, begins[i]);
        }
    }

    buf
}

#[cfg(test)]
mod tests {
    use super::test_parser::*;
    use super::*;

    fn tmp_path(name: &str) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("rw-store-nc3-{}-{}", std::process::id(), name));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(format!("{name}.nc"))
    }

    fn dim(name: &str, len: usize) -> Nc3Dim {
        Nc3Dim {
            name: name.to_string(),
            len,
        }
    }

    fn var(name: &str, dimids: Vec<usize>, attrs: Vec<Nc3Attr>) -> Nc3VarDef {
        Nc3VarDef {
            name: name.to_string(),
            dimids,
            attrs,
        }
    }

    #[test]
    fn cdf2_minimal_file_parses() {
        let path = tmp_path("minimal");
        let dims = vec![dim("y", 3), dim("x", 2)];
        let gattrs = vec![Nc3Attr::text("Conventions", "CF-1.6")];
        let mk_var = |n: &str, units: &str| {
            var(
                n,
                vec![0, 1],
                vec![
                    Nc3Attr::text("units", units),
                    Nc3Attr::floats("_FillValue", vec![f32::NAN]),
                ],
            )
        };
        let vars = vec![
            mk_var("lat", "degrees_north"),
            mk_var("lon", "degrees_east"),
            mk_var("t2m", "K"),
        ];

        let mut w = Nc3Writer::create(&path, dims, gattrs, vars).unwrap();
        // 6 values per var. t2m has a NaN in the middle.
        let lat = [10.0f32, 10.0, 20.0, 20.0, 30.0, 30.0];
        let lon = [-100.0f32, -99.0, -100.0, -99.0, -100.0, -99.0];
        let t2m = [280.0f32, 281.5, f32::NAN, 282.0, 283.25, 284.0];
        w.write_var(&lat).unwrap();
        w.write_var(&lon).unwrap();
        w.write_var(&t2m).unwrap();
        w.finish().unwrap();

        let bytes = std::fs::read(&path).unwrap();
        let parsed = ParsedNc::parse(&bytes).expect("parse own output");

        // Magic + numrecs.
        assert_eq!(&parsed.magic, b"CDF\x02");
        assert_eq!(parsed.numrecs, 0);

        // Dims.
        assert_eq!(parsed.dims.len(), 2);
        assert_eq!(parsed.dims[0], ("y".to_string(), 3));
        assert_eq!(parsed.dims[1], ("x".to_string(), 2));

        // Global attrs.
        assert_eq!(parsed.gattrs.len(), 1);
        let conv = parsed.gattr("Conventions").unwrap();
        assert_eq!(conv, &ParsedAttrValue::Text("CF-1.6".to_string()));

        // Vars: names, dims, type, attrs.
        assert_eq!(parsed.vars.len(), 3);
        for (i, (name, units)) in [
            ("lat", "degrees_north"),
            ("lon", "degrees_east"),
            ("t2m", "K"),
        ]
        .iter()
        .enumerate()
        {
            let v = &parsed.vars[i];
            assert_eq!(&v.name, name);
            assert_eq!(v.dimids, vec![0, 1]);
            assert_eq!(v.nc_type, NC_FLOAT);
            assert_eq!(
                v.attr("units").unwrap(),
                &ParsedAttrValue::Text(units.to_string())
            );
            // _FillValue is a single NaN float.
            match v.attr("_FillValue").unwrap() {
                ParsedAttrValue::Floats(fs) => {
                    assert_eq!(fs.len(), 1);
                    assert!(fs[0].is_nan());
                }
                other => panic!("expected Floats _FillValue, got {other:?}"),
            }
            // begin ≥ header length.
            assert!(
                v.begin >= parsed.header_len,
                "var {name} begin {} < header_len {}",
                v.begin,
                parsed.header_len
            );
            // vsize = 6 floats × 4 = 24, already 4-aligned.
            assert_eq!(v.vsize, 24);
        }

        // Begins ascending and contiguous (begin_{i+1} == begin_i + vsize_i).
        for i in 0..parsed.vars.len() - 1 {
            assert!(parsed.vars[i + 1].begin > parsed.vars[i].begin);
            assert_eq!(
                parsed.vars[i + 1].begin,
                parsed.vars[i].begin + parsed.vars[i].vsize as u64
            );
        }

        // Exact BE f32 round-trip incl. NaN. The parser reads f32 from the
        // data section at each var's begin.
        let lat_data = parsed.read_var_floats(&bytes, 0, 6);
        let lon_data = parsed.read_var_floats(&bytes, 1, 6);
        let t2m_data = parsed.read_var_floats(&bytes, 2, 6);
        assert_eq!(lat_data, lat);
        assert_eq!(lon_data, lon);
        // NaN compares unequal — check bit pattern preserved (we preserve the
        // exact bit pattern: NaN here is the canonical f32::NAN).
        for (got, want) in t2m_data.iter().zip(t2m.iter()) {
            if want.is_nan() {
                assert!(got.is_nan());
                assert_eq!(got.to_bits(), want.to_bits(), "NaN bit pattern preserved");
            } else {
                assert_eq!(got, want);
            }
        }
    }

    #[test]
    fn cdf2_name_padding_exact() {
        // A 5-char name "lat_q" → 4 (length) + 5 bytes + 3 pad = 12 bytes,
        // padded out to 8 content bytes. Assert pad bytes zero at the exact
        // computed offsets in the raw byte stream.
        let path = tmp_path("name_pad");
        let dims = vec![dim("y", 1), dim("x", 1)];
        let vars = vec![var("lat_q", vec![0, 1], vec![])];
        let mut w = Nc3Writer::create(&path, dims, vec![], vars).unwrap();
        w.write_var(&[42.0f32]).unwrap();
        w.finish().unwrap();

        let bytes = std::fs::read(&path).unwrap();

        // Locate "lat_q" in the var_list. magic(4) numrecs(4) dim_list ...
        // Rather than re-derive, search for the name length+bytes pattern: a
        // u32 BE 5 followed by b"lat_q".
        let needle_len = 5u32.to_be_bytes();
        let mut found = None;
        for i in 0..bytes.len().saturating_sub(9) {
            if bytes[i..i + 4] == needle_len && &bytes[i + 4..i + 9] == b"lat_q" {
                found = Some(i);
                break;
            }
        }
        let off = found.expect("found lat_q name");
        // Name bytes occupy off+4 .. off+9 ("lat_q"); padding to 4-byte
        // boundary means content padded length = ceil(5/4)*4 = 8 → pad bytes
        // are off+9, off+10, off+11.
        assert_eq!(bytes[off + 9], 0, "pad byte 0 must be zero");
        assert_eq!(bytes[off + 10], 0, "pad byte 1 must be zero");
        assert_eq!(bytes[off + 11], 0, "pad byte 2 must be zero");
        // The next field (ndims = 2) begins at off+12.
        assert_eq!(&bytes[off + 12..off + 16], &2u32.to_be_bytes());
    }

    #[test]
    fn cdf2_misuse_errors() {
        // Wrong value count.
        {
            let path = tmp_path("misuse_count");
            let dims = vec![dim("y", 3), dim("x", 2)];
            let vars = vec![var("a", vec![0, 1], vec![])];
            let mut w = Nc3Writer::create(&path, dims, vec![], vars).unwrap();
            let err = w.write_var(&[1.0, 2.0, 3.0]).unwrap_err();
            let msg = format!("{err}");
            assert!(msg.contains("expects 6 value"), "got: {msg}");
            assert!(matches!(err, RwStoreError::Format(_)));
        }
        // write_var more times than vars.
        {
            let path = tmp_path("misuse_toomany");
            let dims = vec![dim("y", 1), dim("x", 1)];
            let vars = vec![var("a", vec![0, 1], vec![])];
            let mut w = Nc3Writer::create(&path, dims, vec![], vars).unwrap();
            w.write_var(&[1.0]).unwrap();
            let err = w.write_var(&[2.0]).unwrap_err();
            let msg = format!("{err}");
            assert!(msg.contains("only 1 variable"), "got: {msg}");
            assert!(matches!(err, RwStoreError::Format(_)));
        }
        // finish before all vars written.
        {
            let path = tmp_path("misuse_finish");
            let dims = vec![dim("y", 1), dim("x", 1)];
            let vars = vec![var("a", vec![0, 1], vec![]), var("b", vec![0, 1], vec![])];
            let mut w = Nc3Writer::create(&path, dims, vec![], vars).unwrap();
            w.write_var(&[1.0]).unwrap();
            let err = w.finish().unwrap_err();
            let msg = format!("{err}");
            assert!(msg.contains("writing 1 of 2"), "got: {msg}");
            assert!(matches!(err, RwStoreError::Format(_)));
        }
        // duplicate var names.
        {
            let path = tmp_path("misuse_dupvar");
            let dims = vec![dim("y", 1), dim("x", 1)];
            let vars = vec![var("a", vec![0, 1], vec![]), var("a", vec![0, 1], vec![])];
            let err = Nc3Writer::create(&path, dims, vec![], vars).unwrap_err();
            let msg = format!("{err}");
            assert!(msg.contains("duplicate variable name 'a'"), "got: {msg}");
            assert!(matches!(err, RwStoreError::Format(_)));
        }
        // dimid out of range.
        {
            let path = tmp_path("misuse_dimid");
            let dims = vec![dim("y", 1)];
            let vars = vec![var("a", vec![0, 5], vec![])];
            let err = Nc3Writer::create(&path, dims, vec![], vars).unwrap_err();
            let msg = format!("{err}");
            assert!(msg.contains("references dimid 5"), "got: {msg}");
            assert!(matches!(err, RwStoreError::Format(_)));
        }
        // bad name: leading digit.
        {
            let path = tmp_path("misuse_digit");
            let dims = vec![dim("y", 1), dim("x", 1)];
            let vars = vec![var("2bad", vec![0, 1], vec![])];
            let err = Nc3Writer::create(&path, dims, vec![], vars).unwrap_err();
            let msg = format!("{err}");
            assert!(msg.contains("not NC-safe"), "got: {msg}");
            assert!(matches!(err, RwStoreError::Format(_)));
        }
        // bad name: embedded space.
        {
            let path = tmp_path("misuse_space");
            let dims = vec![dim("y", 1), dim("x", 1)];
            let vars = vec![var("bad name", vec![0, 1], vec![])];
            let err = Nc3Writer::create(&path, dims, vec![], vars).unwrap_err();
            let msg = format!("{err}");
            assert!(msg.contains("not NC-safe"), "got: {msg}");
            assert!(matches!(err, RwStoreError::Format(_)));
        }
        // zero-length dim.
        {
            let path = tmp_path("misuse_zerodim");
            let dims = vec![dim("y", 0), dim("x", 2)];
            let vars = vec![var("a", vec![0, 1], vec![])];
            let err = Nc3Writer::create(&path, dims, vec![], vars).unwrap_err();
            let msg = format!("{err}");
            assert!(msg.contains("length 0"), "got: {msg}");
            assert!(matches!(err, RwStoreError::Format(_)));
        }
    }

    #[test]
    fn cdf2_3d_var_layout() {
        // dims (level=2, y=3, x=2), var temp(level,y,x). flat index
        // lvl*6 + y*2 + x must land at begin + idx*4.
        let path = tmp_path("layout3d");
        let dims = vec![dim("level", 2), dim("y", 3), dim("x", 2)];
        let vars = vec![var("temp", vec![0, 1, 2], vec![])];
        let mut w = Nc3Writer::create(&path, dims, vec![], vars).unwrap();

        // Fill with a recognizable per-cell value = idx as f32 so we can read
        // it back at the computed offset.
        let mut data = vec![0.0f32; 12];
        for (idx, slot) in data.iter_mut().enumerate() {
            *slot = (idx as f32) * 1.5 + 0.25;
        }
        w.write_var(&data).unwrap();
        w.finish().unwrap();

        let bytes = std::fs::read(&path).unwrap();
        let parsed = ParsedNc::parse(&bytes).unwrap();
        let begin = parsed.vars[0].begin;
        assert_eq!(parsed.vars[0].dimids, vec![0, 1, 2]);
        // vsize = 12 × 4 = 48.
        assert_eq!(parsed.vars[0].vsize, 48);

        for lvl in 0..2 {
            for y in 0..3 {
                for x in 0..2 {
                    let idx = lvl * 6 + y * 2 + x;
                    let byte_off = (begin as usize) + idx * 4;
                    let got = f32::from_be_bytes([
                        bytes[byte_off],
                        bytes[byte_off + 1],
                        bytes[byte_off + 2],
                        bytes[byte_off + 3],
                    ]);
                    assert_eq!(got, data[idx], "idx {idx} at byte {byte_off}");
                }
            }
        }
    }

    #[test]
    fn cdf2_empty_lists() {
        // No global attrs, and a var with no attrs ⇒ ABSENT encodings.
        let path = tmp_path("empty_lists");
        let dims = vec![dim("y", 1), dim("x", 1)];
        let vars = vec![var("a", vec![0, 1], vec![])];
        let mut w = Nc3Writer::create(&path, dims, vec![], vars).unwrap();
        w.write_var(&[7.0]).unwrap();
        w.finish().unwrap();

        let bytes = std::fs::read(&path).unwrap();

        // gatt_list sits right after the dim_list. Compute its offset by hand:
        // magic(4) numrecs(4) = 8.
        // dim_list: NC_DIMENSION(4) nelems=2(4) then 2 dims.
        //   dim "y": name nelems(4) "y"(1) pad(3) len(4) = 12
        //   dim "x": same = 12
        // → dim_list = 8 + 24 = 32 bytes. gatt_list begins at offset 8+32=40.
        let gatt_off = 40;
        assert_eq!(
            &bytes[gatt_off..gatt_off + 4],
            &ABSENT_TAG.to_be_bytes(),
            "global att_list ABSENT tag"
        );
        assert_eq!(
            &bytes[gatt_off + 4..gatt_off + 8],
            &0u32.to_be_bytes(),
            "global att_list ABSENT nelems"
        );

        // The var's vatt_list is ABSENT too. Find it relative to the var name.
        // var_list begins at gatt_off+8 = 48: NC_VARIABLE(4) nelems=1(4),
        // then var "a": name nelems(4) "a"(1) pad(3) = 8, ndims(4)=1,
        // dimids[2]×4 = 8 → vatt_list starts at 48 + 8 + 8 + 4 + 8 = 76.
        let vatt_off = 48 + 8 + 8 + 4 + 8;
        assert_eq!(
            &bytes[vatt_off..vatt_off + 4],
            &ABSENT_TAG.to_be_bytes(),
            "var att_list ABSENT tag"
        );
        assert_eq!(
            &bytes[vatt_off + 4..vatt_off + 8],
            &0u32.to_be_bytes(),
            "var att_list ABSENT nelems"
        );

        // Cross-check against the parser's view.
        let parsed = ParsedNc::parse(&bytes).unwrap();
        assert!(parsed.gattrs.is_empty());
        assert!(parsed.vars[0].attrs.is_empty());
    }

    #[test]
    fn vsize_field_clamps_at_u64_max() {
        // bytes = u64::MAX * 4 saturates to u64::MAX; saturating_add(3) stays
        // u64::MAX; the guard `> u32::MAX - 3` triggers ⇒ returns u32::MAX.
        assert_eq!(super::vsize_field(u64::MAX), u32::MAX);
        // A normal small case should still round-trip: 6 elems * 4 = 24, already
        // aligned, fits in u32.
        assert_eq!(super::vsize_field(6), 24);
    }

    #[test]
    fn cdf2_header_two_pass_stable() {
        // Determinism: the same defs produce byte-identical output.
        let build = || {
            let dims = vec![dim("level", 3), dim("y", 4), dim("x", 5)];
            let gattrs = vec![
                Nc3Attr::text("Conventions", "CF-1.6"),
                Nc3Attr::floats("forecast_hour", vec![6.0]),
            ];
            let vars = vec![
                var(
                    "lat",
                    vec![1, 2],
                    vec![Nc3Attr::text("units", "degrees_north")],
                ),
                var(
                    "temp",
                    vec![0, 1, 2],
                    vec![
                        Nc3Attr::text("units", "K"),
                        Nc3Attr::floats("_FillValue", vec![f32::NAN]),
                    ],
                ),
            ];
            (dims, gattrs, vars)
        };

        let path_a = tmp_path("stable_a");
        let (d, g, v) = build();
        let mut wa = Nc3Writer::create(&path_a, d, g, v).unwrap();
        wa.write_var(&[1.0f32; 20]).unwrap();
        wa.write_var(&[2.0f32; 60]).unwrap();
        wa.finish().unwrap();

        let path_b = tmp_path("stable_b");
        let (d, g, v) = build();
        let mut wb = Nc3Writer::create(&path_b, d, g, v).unwrap();
        wb.write_var(&[1.0f32; 20]).unwrap();
        wb.write_var(&[2.0f32; 60]).unwrap();
        wb.finish().unwrap();

        let bytes_a = std::fs::read(&path_a).unwrap();
        let bytes_b = std::fs::read(&path_b).unwrap();
        assert_eq!(bytes_a, bytes_b, "identical defs ⇒ identical bytes");

        // Header length is stable / measurable via the parser.
        let parsed = ParsedNc::parse(&bytes_a).unwrap();
        assert!(parsed.header_len > 0);
        assert_eq!(parsed.vars[0].begin, parsed.header_len);
    }
}

/// An independent, test-only mini-parser for CDF-2 files. It reads the raw
/// byte stream with its own big-endian readers and the spec grammar — it does
/// NOT call any writer internals or share serialization helpers. Task 6's
/// tests reuse it, hence `pub(crate)`.
#[cfg(test)]
pub(crate) mod test_parser {
    /// A parsed attribute value.
    #[derive(Debug, Clone, PartialEq)]
    pub enum ParsedAttrValue {
        Text(String),
        Floats(Vec<f32>),
    }

    /// A parsed attribute (name + value).
    #[derive(Debug, Clone)]
    pub struct ParsedAttr {
        pub name: String,
        pub value: ParsedAttrValue,
    }

    /// A parsed variable.
    #[derive(Debug, Clone)]
    pub struct ParsedVar {
        pub name: String,
        pub dimids: Vec<usize>,
        pub attrs: Vec<ParsedAttr>,
        pub nc_type: u32,
        pub vsize: u32,
        pub begin: u64,
    }

    impl ParsedVar {
        pub fn attr(&self, name: &str) -> Option<&ParsedAttrValue> {
            self.attrs.iter().find(|a| a.name == name).map(|a| &a.value)
        }
    }

    /// A fully parsed CDF-2 header.
    #[derive(Debug, Clone)]
    pub struct ParsedNc {
        pub magic: [u8; 4],
        pub numrecs: u32,
        pub dims: Vec<(String, u32)>,
        pub gattrs: Vec<ParsedAttr>,
        pub vars: Vec<ParsedVar>,
        /// Byte length of the header (== where the cursor sat after var_list).
        pub header_len: u64,
    }

    // --- independent big-endian readers operating on a moving cursor ---

    struct Cur<'a> {
        b: &'a [u8],
        p: usize,
    }

    impl<'a> Cur<'a> {
        fn u32(&mut self) -> u32 {
            let v = u32::from_be_bytes([
                self.b[self.p],
                self.b[self.p + 1],
                self.b[self.p + 2],
                self.b[self.p + 3],
            ]);
            self.p += 4;
            v
        }

        fn u64(&mut self) -> u64 {
            let mut a = [0u8; 8];
            a.copy_from_slice(&self.b[self.p..self.p + 8]);
            self.p += 8;
            u64::from_be_bytes(a)
        }

        fn f32(&mut self) -> f32 {
            let v = f32::from_be_bytes([
                self.b[self.p],
                self.b[self.p + 1],
                self.b[self.p + 2],
                self.b[self.p + 3],
            ]);
            self.p += 4;
            v
        }

        /// name := nelems(u32) bytes pad-to-4.
        fn name(&mut self) -> String {
            let n = self.u32() as usize;
            let s = String::from_utf8(self.b[self.p..self.p + n].to_vec()).unwrap();
            self.p += n;
            self.align4();
            s
        }

        fn align4(&mut self) {
            while self.p % 4 != 0 {
                self.p += 1;
            }
        }
    }

    impl ParsedNc {
        pub fn parse(bytes: &[u8]) -> Result<ParsedNc, String> {
            const NC_DIMENSION: u32 = 0x0000_000A;
            const NC_VARIABLE: u32 = 0x0000_000B;
            const NC_CHAR: u32 = 2;
            const NC_FLOAT: u32 = 5;

            let mut c = Cur { b: bytes, p: 0 };
            let magic = [bytes[0], bytes[1], bytes[2], bytes[3]];
            if &magic[0..3] != b"CDF" {
                return Err(format!("bad magic {magic:?}"));
            }
            if magic[3] != 2 {
                return Err(format!("not CDF-2 (version byte {})", magic[3]));
            }
            c.p = 4;
            let numrecs = c.u32();

            // dim_list
            let mut dims = Vec::new();
            let tag = c.u32();
            let n = c.u32();
            if tag == 0 {
                // ABSENT: tag 0, nelems also 0 (already consumed).
            } else if tag == NC_DIMENSION {
                for _ in 0..n {
                    let name = c.name();
                    let len = c.u32();
                    dims.push((name, len));
                }
            } else {
                return Err(format!("unexpected dim_list tag {tag:#x}"));
            }

            // helper to parse an att_list
            fn parse_atts(c: &mut Cur, nc_char: u32, nc_float: u32) -> Vec<ParsedAttr> {
                const NC_ATTRIBUTE: u32 = 0x0000_000C;
                let tag = c.u32();
                let n = c.u32();
                let mut out = Vec::new();
                if tag == 0 {
                    return out; // ABSENT
                }
                assert_eq!(tag, NC_ATTRIBUTE, "att_list tag");
                for _ in 0..n {
                    let name = c.name();
                    let ty = c.u32();
                    let nelems = c.u32() as usize;
                    let value = if ty == nc_char {
                        let s = String::from_utf8(c.b[c.p..c.p + nelems].to_vec()).unwrap();
                        c.p += nelems;
                        c.align4();
                        ParsedAttrValue::Text(s)
                    } else if ty == nc_float {
                        let mut vals = Vec::with_capacity(nelems);
                        for _ in 0..nelems {
                            vals.push(c.f32());
                        }
                        c.align4();
                        ParsedAttrValue::Floats(vals)
                    } else {
                        panic!("unsupported attr type {ty}");
                    };
                    out.push(ParsedAttr { name, value });
                }
                out
            }

            let gattrs = parse_atts(&mut c, NC_CHAR, NC_FLOAT);

            // var_list
            let mut vars = Vec::new();
            let tag = c.u32();
            let n = c.u32();
            if tag == 0 {
                // ABSENT
            } else if tag == NC_VARIABLE {
                for _ in 0..n {
                    let name = c.name();
                    let ndims = c.u32() as usize;
                    let mut dimids = Vec::with_capacity(ndims);
                    for _ in 0..ndims {
                        dimids.push(c.u32() as usize);
                    }
                    let attrs = parse_atts(&mut c, NC_CHAR, NC_FLOAT);
                    let nc_type = c.u32();
                    let vsize = c.u32();
                    let begin = c.u64();
                    vars.push(ParsedVar {
                        name,
                        dimids,
                        attrs,
                        nc_type,
                        vsize,
                        begin,
                    });
                }
            } else {
                return Err(format!("unexpected var_list tag {tag:#x}"));
            }

            let header_len = c.p as u64;
            Ok(ParsedNc {
                magic,
                numrecs,
                dims,
                gattrs,
                vars,
                header_len,
            })
        }

        pub fn gattr(&self, name: &str) -> Option<&ParsedAttrValue> {
            self.gattrs
                .iter()
                .find(|a| a.name == name)
                .map(|a| &a.value)
        }

        /// Read `count` big-endian f32 values from var `i`'s data section.
        pub fn read_var_floats(&self, bytes: &[u8], i: usize, count: usize) -> Vec<f32> {
            let begin = self.vars[i].begin as usize;
            let mut out = Vec::with_capacity(count);
            for k in 0..count {
                let off = begin + k * 4;
                out.push(f32::from_be_bytes([
                    bytes[off],
                    bytes[off + 1],
                    bytes[off + 2],
                    bytes[off + 3],
                ]));
            }
            out
        }
    }
}
