//! Rewrite chosen variables of an existing classic file and keep every
//! other byte of it.
//!
//! # Why this exists
//!
//! A CPU analysis that edits a WRF file in place (GSI reads `wrf_inout`,
//! rewrites the fields it analyses and leaves every other byte alone)
//! needs the model side of the exchange to hand that file back the same
//! way: the header bytes, the data-section alignment the producing
//! library chose, and every variable the model does not carry must come
//! back untouched. Writing a fresh file from a schema reproduces the
//! values but not necessarily the layout, because netCDF-C, PnetCDF and
//! this crate each place the data section by their own alignment rule.
//! Patching a copy of the incoming file keeps the layout by construction,
//! and a variable written back with the values it was read with leaves
//! the file byte-identical.
//!
//! # The rules it enforces
//!
//! * The header never changes. A patch cannot add, remove, rename or
//!   retype a variable, add a record, or touch an attribute; those would
//!   move every offset after them. Such a change is a different file and
//!   belongs to [`crate::NcWriter`].
//! * A payload must be the variable's own type, or `f64` values that
//!   narrow to the variable's type EXACTLY (the shape a decoder that
//!   widens everything to `f64` hands back). An `f64` value that is not
//!   representable in the stored type is refused by name, because a
//!   rounded write would put a value in the analysis that no one chose.
//! * The copy is written beside the target under a partial name and
//!   renamed into place only at [`NcPatcher::finish`]; a patch that dies
//!   leaves no file at the target path that looks finished.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::error::{NcWriteError, Result};
use crate::scan::{parse_header, ScanHeader};
use crate::types::{NcType, VarData};

/// Elements converted per write while streaming a payload to disk.
const CHUNK_ELEMS: usize = 1 << 18;

/// One variable of the template, as its header declares it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PatchVar {
    pub name: String,
    pub ty: NcType,
    /// Whether the variable uses the record dimension.
    pub is_record: bool,
    /// Elements in the whole variable (fixed) or in one record slab.
    pub elems: u64,
}

/// A copy of a classic file, opened for rewriting variable data in place.
pub struct NcPatcher {
    file: File,
    partial: PathBuf,
    target: PathBuf,
    header: ScanHeader,
    index: HashMap<String, usize>,
    written: Vec<bool>,
    finished: bool,
}

impl NcPatcher {
    /// Copy `template` to a partial file beside `target` and parse its
    /// header. `target` must not exist yet and must not be `template`.
    pub fn open(template: impl AsRef<Path>, target: impl AsRef<Path>) -> Result<NcPatcher> {
        let template = template.as_ref();
        let target = target.as_ref().to_path_buf();
        if target.exists() {
            return Err(NcWriteError::Usage(format!(
                "{} already exists; a patched copy is never written over an \
                 existing file, so remove it or choose another path",
                target.display()
            )));
        }
        let name = target.file_name().ok_or_else(|| {
            NcWriteError::Usage(format!("{} names no file", target.display()))
        })?;
        let mut partial_name = std::ffi::OsString::from(".");
        partial_name.push(name);
        partial_name.push(format!(".partial-{}", std::process::id()));
        let partial = target.with_file_name(partial_name);

        std::fs::copy(template, &partial).map_err(|error| {
            NcWriteError::Usage(format!(
                "cannot copy {} to {}: {error}",
                template.display(),
                partial.display()
            ))
        })?;
        let opened = (|| -> Result<(File, ScanHeader)> {
            let file = OpenOptions::new().read(true).write(true).open(&partial)?;
            let file_len = file.metadata()?.len();
            let mut reader = BufReader::with_capacity(1 << 16, file.try_clone()?);
            let header = parse_header(&mut reader, template)?;
            for var in &header.vars {
                let last = if var.is_record {
                    header.numrecs.saturating_sub(1)
                } else {
                    0
                };
                let end = var.begin + last * header.recsize + var.slab_bytes;
                if (!var.is_record || header.numrecs > 0) && end > file_len {
                    return Err(NcWriteError::Usage(format!(
                        "{}: variable '{}' needs bytes up to {end} but the file is \
                         {file_len} bytes long; a truncated template cannot be \
                         patched into a whole file",
                        template.display(),
                        var.name
                    )));
                }
            }
            Ok((file, header))
        })();
        let (file, header) = match opened {
            Ok(pair) => pair,
            Err(error) => {
                let _ = std::fs::remove_file(&partial);
                return Err(error);
            }
        };
        let index = header
            .vars
            .iter()
            .enumerate()
            .map(|(position, var)| (var.name.clone(), position))
            .collect();
        let written = vec![false; header.vars.len()];
        Ok(NcPatcher {
            file,
            partial,
            target,
            header,
            index,
            written,
            finished: false,
        })
    }

    /// Variable names in definition order.
    pub fn names(&self) -> Vec<String> {
        self.header.vars.iter().map(|var| var.name.clone()).collect()
    }

    /// Every variable's stored type and slab size, in definition order.
    /// A caller chooses its payload type from this, so it never has to
    /// guess what the template stores.
    pub fn variables(&self) -> Vec<PatchVar> {
        self.header
            .vars
            .iter()
            .map(|var| PatchVar {
                name: var.name.clone(),
                ty: var.ty,
                is_record: var.is_record,
                elems: var.elems,
            })
            .collect()
    }

    /// Records the template holds. A patch cannot change this.
    pub fn num_records(&self) -> u64 {
        self.header.numrecs
    }

    /// Names of the variables a `put` has rewritten so far.
    pub fn written(&self) -> Vec<String> {
        self.header
            .vars
            .iter()
            .zip(&self.written)
            .filter(|(_, done)| **done)
            .map(|(var, _)| var.name.clone())
            .collect()
    }

    /// Rewrite variable `name`. `record` is `Some(r)` for a record
    /// variable's slab `r` and `None` for a fixed variable.
    pub fn put(&mut self, name: &str, record: Option<u64>, data: VarData<'_>) -> Result<()> {
        let position = *self.index.get(name).ok_or_else(|| {
            NcWriteError::Usage(format!(
                "'{name}' is not a variable of the template; a patch rewrites \
                 existing variables and cannot add one"
            ))
        })?;
        let var = &self.header.vars[position];
        let offset = match (var.is_record, record) {
            (true, Some(rec)) if rec < self.header.numrecs => {
                var.begin + rec * self.header.recsize
            }
            (true, Some(rec)) => {
                return Err(NcWriteError::Usage(format!(
                    "'{name}' record {rec} does not exist (the template holds {}); \
                     a patch cannot add records, that changes the file length",
                    self.header.numrecs
                )))
            }
            (true, None) => {
                return Err(NcWriteError::Usage(format!(
                    "'{name}' is a record variable; say which record to rewrite"
                )))
            }
            (false, Some(rec)) => {
                return Err(NcWriteError::Usage(format!(
                    "'{name}' is a fixed variable, so record {rec} names nothing"
                )))
            }
            (false, None) => var.begin,
        };
        if data.len() as u64 != var.elems {
            return Err(NcWriteError::Usage(format!(
                "'{name}' holds {} element(s) per slab; the payload has {}",
                var.elems,
                data.len()
            )));
        }
        let ty = var.ty;
        // Validate the whole payload before the first byte moves, so a
        // refusal never leaves the variable half rewritten.
        check_exact(name, ty, &data)?;
        self.file.seek(SeekFrom::Start(offset))?;
        let file = &mut self.file;
        stream_be(ty, &data, &mut |bytes: &[u8]| file.write_all(bytes))?;
        self.written[position] = true;
        Ok(())
    }

    /// Flush, fsync and rename the partial copy onto the target.
    pub fn finish(mut self) -> Result<PathBuf> {
        self.file.flush()?;
        self.file.sync_all()?;
        if self.target.exists() {
            return Err(NcWriteError::Usage(format!(
                "{} appeared while the patch was being written; it is left \
                 alone and the patched copy stays at {}",
                self.target.display(),
                self.partial.display()
            )));
        }
        std::fs::rename(&self.partial, &self.target)?;
        self.finished = true;
        Ok(self.target.clone())
    }
}

impl Drop for NcPatcher {
    fn drop(&mut self) {
        if !self.finished {
            let _ = std::fs::remove_file(&self.partial);
        }
    }
}

/// Refuse a payload that is neither the variable's own type nor `f64`
/// values that narrow to it exactly.
fn check_exact(name: &str, ty: NcType, data: &VarData<'_>) -> Result<()> {
    if data.nc_type() == ty {
        return Ok(());
    }
    let VarData::F64(values) = data else {
        return Err(NcWriteError::Usage(format!(
            "'{name}' is stored as {}; a {} payload is neither that type nor \
             f64 values that narrow to it exactly",
            ty.name(),
            data.nc_type().name()
        )));
    };
    let inexact = |index: usize, value: f64| {
        NcWriteError::Usage(format!(
            "'{name}' is stored as {}; payload element {index} = {value:?} is not \
             exactly representable in that type, and a patch never rounds a value \
             into an analysis file",
            ty.name()
        ))
    };
    macro_rules! check_int {
        ($t:ty) => {{
            for (index, &value) in values.iter().enumerate() {
                if !(value.is_finite()
                    && value.fract() == 0.0
                    && value >= <$t>::MIN as f64
                    && value <= <$t>::MAX as f64
                    && (value as $t) as f64 == value)
                {
                    return Err(inexact(index, value));
                }
            }
        }};
    }
    match ty {
        NcType::Float => {
            for (index, &value) in values.iter().enumerate() {
                // to_bits keeps -0.0 distinct and refuses NaN: an f64 NaN
                // does not say which f32 NaN bit pattern was stored.
                if ((value as f32) as f64).to_bits() != value.to_bits() {
                    return Err(inexact(index, value));
                }
            }
        }
        NcType::Byte => check_int!(i8),
        NcType::UByte => check_int!(u8),
        NcType::Short => check_int!(i16),
        NcType::UShort => check_int!(u16),
        NcType::Int => check_int!(i32),
        NcType::UInt => check_int!(u32),
        NcType::Int64 => check_int!(i64),
        NcType::UInt64 => check_int!(u64),
        NcType::Char | NcType::Double => {
            return Err(NcWriteError::Usage(format!(
                "'{name}' is stored as {}; character data must arrive as bytes",
                ty.name()
            )))
        }
    }
    Ok(())
}

/// Hand `sink` the payload as the variable's big-endian bytes, a chunk at
/// a time. The payload has already passed [`check_exact`].
fn stream_be(
    ty: NcType,
    data: &VarData<'_>,
    sink: &mut dyn FnMut(&[u8]) -> std::io::Result<()>,
) -> Result<()> {
    let mut buf: Vec<u8> = Vec::with_capacity(CHUNK_ELEMS * 8);
    macro_rules! own {
        ($vals:expr) => {{
            for chunk in $vals.chunks(CHUNK_ELEMS) {
                buf.clear();
                for value in chunk {
                    buf.extend_from_slice(&value.to_be_bytes());
                }
                sink(&buf)?;
            }
        }};
    }
    macro_rules! narrow {
        ($vals:expr, $t:ty) => {{
            for chunk in $vals.chunks(CHUNK_ELEMS) {
                buf.clear();
                for &value in chunk {
                    buf.extend_from_slice(&(value as $t).to_be_bytes());
                }
                sink(&buf)?;
            }
        }};
    }
    match (data, ty) {
        (VarData::Char(v), _) | (VarData::U8(v), _) => sink(v)?,
        (VarData::I8(v), _) => own!(v),
        (VarData::I16(v), _) => own!(v),
        (VarData::U16(v), _) => own!(v),
        (VarData::I32(v), _) => own!(v),
        (VarData::U32(v), _) => own!(v),
        (VarData::F32(v), _) => own!(v),
        (VarData::I64(v), _) => own!(v),
        (VarData::U64(v), _) => own!(v),
        (VarData::F64(v), NcType::Double) => own!(v),
        (VarData::F64(v), NcType::Float) => narrow!(v, f32),
        (VarData::F64(v), NcType::Byte) => narrow!(v, i8),
        (VarData::F64(v), NcType::UByte) => narrow!(v, u8),
        (VarData::F64(v), NcType::Short) => narrow!(v, i16),
        (VarData::F64(v), NcType::UShort) => narrow!(v, u16),
        (VarData::F64(v), NcType::Int) => narrow!(v, i32),
        (VarData::F64(v), NcType::UInt) => narrow!(v, u32),
        (VarData::F64(v), NcType::Int64) => narrow!(v, i64),
        (VarData::F64(v), NcType::UInt64) => narrow!(v, u64),
        (VarData::F64(_), NcType::Char) => unreachable!("check_exact refuses f64 into NC_CHAR"),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::Schema;
    use crate::types::{AttrValue, NcFormat};
    use crate::writer::NcWriter;

    fn temp_dir(stem: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "gpuwm-ncpatch-{stem}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    const T2: [f32; 12] = [
        280.5, 281.25, 282.0, 283.125, 284.0, 285.5, 286.0, 287.75, 288.0, 289.0, 290.5, -0.0,
    ];

    /// A wrf_inout-shaped template: char Times, float and int record
    /// variables, a fixed double, one variable attribute.
    fn write_template(path: &Path, format: NcFormat) {
        let mut schema = Schema::new(format);
        let time = schema.def_dim("Time", 0, true).unwrap();
        let strlen = schema.def_dim("DateStrLen", 19, false).unwrap();
        let south_north = schema.def_dim("south_north", 3, false).unwrap();
        let west_east = schema.def_dim("west_east", 4, false).unwrap();
        schema
            .put_global_attr("TITLE", AttrValue::Text(" OUTPUT FROM WRF V3.9 MODEL".into()))
            .unwrap();
        let times = schema.def_var("Times", NcType::Char, &[time, strlen]).unwrap();
        let t2 = schema
            .def_var("T2", NcType::Float, &[time, south_north, west_east])
            .unwrap();
        schema.put_var_attr(t2, "units", AttrValue::Text("K".into())).unwrap();
        let isltyp = schema
            .def_var("ISLTYP", NcType::Int, &[time, south_north, west_east])
            .unwrap();
        let znu = schema.def_var("ZNU", NcType::Double, &[south_north]).unwrap();
        let mut writer = NcWriter::create(path, schema).unwrap();
        writer
            .write_record(0, times, VarData::Char(b"2026-10-03_12:00:00"))
            .unwrap();
        writer.write_record(0, t2, VarData::F32(&T2)).unwrap();
        writer
            .write_record(0, isltyp, VarData::I32(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16]))
            .unwrap();
        writer
            .write_var(znu, VarData::F64(&[0.9959, 0.9875, 0.9713]))
            .unwrap();
        writer.finish().unwrap();
    }

    #[test]
    fn an_untouched_patch_is_byte_identical() {
        for format in [NcFormat::Classic, NcFormat::Offset64, NcFormat::Cdf5] {
            let dir = temp_dir("untouched");
            let template = dir.join("wrf_inout");
            write_template(&template, format);
            let out = dir.join("wrf_inout.back");
            NcPatcher::open(&template, &out).unwrap().finish().unwrap();
            assert_eq!(std::fs::read(&template).unwrap(), std::fs::read(&out).unwrap());
            std::fs::remove_dir_all(&dir).ok();
        }
    }

    #[test]
    fn the_template_is_described_as_its_header_declares_it() {
        let dir = temp_dir("describe");
        let template = dir.join("wrf_inout");
        write_template(&template, NcFormat::Offset64);
        let out = dir.join("wrf_inout.back");
        let patch = NcPatcher::open(&template, &out).unwrap();
        assert_eq!(patch.num_records(), 1);
        let described: Vec<(String, NcType, bool, u64)> = patch
            .variables()
            .into_iter()
            .map(|var| (var.name, var.ty, var.is_record, var.elems))
            .collect();
        assert_eq!(
            described,
            vec![
                ("Times".to_string(), NcType::Char, true, 19),
                ("T2".to_string(), NcType::Float, true, 12),
                ("ISLTYP".to_string(), NcType::Int, true, 12),
                ("ZNU".to_string(), NcType::Double, false, 3),
            ]
        );
        drop(patch);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn writing_back_the_read_values_is_byte_identical_even_through_f64() {
        let dir = temp_dir("same");
        let template = dir.join("wrf_inout");
        write_template(&template, NcFormat::Offset64);
        let out = dir.join("wrf_inout.back");
        let mut patch = NcPatcher::open(&template, &out).unwrap();
        let widened: Vec<f64> = T2.iter().map(|&v| v as f64).collect();
        patch.put("T2", Some(0), VarData::F64(&widened)).unwrap();
        let ints: Vec<f64> = (1..=11).map(f64::from).chain([16.0]).collect();
        patch.put("ISLTYP", Some(0), VarData::F64(&ints)).unwrap();
        patch
            .put("ZNU", None, VarData::F64(&[0.9959, 0.9875, 0.9713]))
            .unwrap();
        patch
            .put("Times", Some(0), VarData::Char(b"2026-10-03_12:00:00"))
            .unwrap();
        assert_eq!(patch.written(), vec!["Times", "T2", "ISLTYP", "ZNU"]);
        patch.finish().unwrap();
        assert_eq!(std::fs::read(&template).unwrap(), std::fs::read(&out).unwrap());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_changed_variable_moves_only_its_own_bytes() {
        let dir = temp_dir("changed");
        let template = dir.join("wrf_inout");
        write_template(&template, NcFormat::Offset64);
        let out = dir.join("wrf_inout.anl");
        let mut patch = NcPatcher::open(&template, &out).unwrap();
        let mut analysed = T2;
        analysed[5] += 1.5;
        patch.put("T2", Some(0), VarData::F32(&analysed)).unwrap();
        patch.finish().unwrap();
        let before = std::fs::read(&template).unwrap();
        let after = std::fs::read(&out).unwrap();
        assert_eq!(before.len(), after.len());
        let differing: Vec<usize> = (0..before.len()).filter(|&i| before[i] != after[i]).collect();
        assert!(!differing.is_empty() && differing.len() <= 4, "{differing:?}");
        let first = differing[0];
        assert!(differing.iter().all(|&i| i < first + 4));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_value_that_does_not_narrow_exactly_is_refused_by_name() {
        let dir = temp_dir("inexact");
        let template = dir.join("wrf_inout");
        write_template(&template, NcFormat::Offset64);
        let out = dir.join("wrf_inout.back");
        let mut patch = NcPatcher::open(&template, &out).unwrap();
        let mut widened: Vec<f64> = T2.iter().map(|&v| v as f64).collect();
        widened[3] = 0.1;
        let error = patch.put("T2", Some(0), VarData::F64(&widened)).unwrap_err().to_string();
        assert!(error.contains("'T2'") && error.contains("element 3"), "{error}");
        let error = patch
            .put("ISLTYP", Some(0), VarData::F64(&[1.5; 12]))
            .unwrap_err()
            .to_string();
        assert!(error.contains("'ISLTYP'"), "{error}");
        drop(patch);
        assert!(!out.exists());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn header_changes_are_refused() {
        let dir = temp_dir("refuse");
        let template = dir.join("wrf_inout");
        write_template(&template, NcFormat::Offset64);
        let out = dir.join("wrf_inout.back");
        let mut patch = NcPatcher::open(&template, &out).unwrap();
        let unknown = patch.put("QNEW", Some(0), VarData::F32(&[0.0; 12])).unwrap_err();
        assert!(unknown.to_string().contains("cannot add one"));
        let record = patch.put("T2", Some(1), VarData::F32(&T2)).unwrap_err();
        assert!(record.to_string().contains("cannot add records"));
        let fixed = patch.put("ZNU", Some(0), VarData::F64(&[0.0; 3])).unwrap_err();
        assert!(fixed.to_string().contains("fixed variable"));
        let count = patch.put("T2", Some(0), VarData::F32(&T2[..11])).unwrap_err();
        assert!(count.to_string().contains("12 element"));
        let retype = patch.put("T2", Some(0), VarData::I32(&[0; 12])).unwrap_err();
        assert!(retype.to_string().contains("stored as NC_FLOAT"), "{retype}");
        drop(patch);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn an_existing_target_and_a_truncated_template_are_refused() {
        let dir = temp_dir("guards");
        let template = dir.join("wrf_inout");
        write_template(&template, NcFormat::Offset64);
        let exists = NcPatcher::open(&template, &template).err().unwrap().to_string();
        assert!(exists.contains("already exists"), "{exists}");
        let short = dir.join("short");
        let bytes = std::fs::read(&template).unwrap();
        std::fs::write(&short, &bytes[..bytes.len() - 8]).unwrap();
        let out = dir.join("short.back");
        let truncated = NcPatcher::open(&short, &out).err().unwrap().to_string();
        assert!(truncated.contains("truncated"), "{truncated}");
        assert!(std::fs::read_dir(&dir).unwrap().all(|entry| {
            !entry.unwrap().file_name().to_string_lossy().contains("partial")
        }));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A template whose data section starts on a 512-byte boundary, the
    /// way PnetCDF or `nc__enddef` with a large `v_align` lays a file out.
    /// Everything this crate writes itself starts right after the header,
    /// so the begin offsets are rewritten by hand: the patch must follow
    /// the header's offsets, not recompute them.
    #[test]
    fn a_foreign_alignment_is_kept() {
        let dir = temp_dir("aligned");
        let plain = dir.join("plain");
        write_template(&plain, NcFormat::Offset64);
        let original = std::fs::read(&plain).unwrap();
        let mut reader = BufReader::new(File::open(&plain).unwrap());
        let header = parse_header(&mut reader, &plain).unwrap();
        let data_start = header.vars.iter().map(|v| v.begin).min().unwrap() as usize;
        let shift = 512 - data_start;
        let mut aligned = original[..data_start].to_vec();
        // CDF-2 begin fields are 8-byte big-endian words. Find every one on
        // the untouched header first, then rewrite them.
        let fields: Vec<(usize, u64)> = header
            .vars
            .iter()
            .map(|var| {
                let needle = var.begin.to_be_bytes();
                let at = aligned
                    .windows(8)
                    .rposition(|w| w == needle)
                    .expect("begin field present in header");
                (at, var.begin + shift as u64)
            })
            .collect();
        for (at, begin) in fields {
            aligned[at..at + 8].copy_from_slice(&begin.to_be_bytes());
        }
        aligned.resize(512, 0);
        aligned.extend_from_slice(&original[data_start..]);
        let template = dir.join("wrf_inout");
        std::fs::write(&template, &aligned).unwrap();

        let out = dir.join("wrf_inout.back");
        let mut patch = NcPatcher::open(&template, &out).unwrap();
        patch.put("T2", Some(0), VarData::F32(&T2)).unwrap();
        patch.finish().unwrap();
        assert_eq!(std::fs::read(&out).unwrap(), aligned);
        std::fs::remove_dir_all(&dir).ok();
    }
}
