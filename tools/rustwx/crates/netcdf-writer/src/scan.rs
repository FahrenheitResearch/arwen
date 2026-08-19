//! Read the file back and say which floating-point variables hold a
//! non-finite value.
//!
//! # Why a reader lives in a writer crate
//!
//! This is not a general NetCDF reader -- that is `netcrust`'s job, and
//! `rw_netcdf` is the front door for it. This parses only the header
//! grammar [`crate::header`] emits, and it exists for one caller:
//! `gpuwm.wrf_direct`, whose export of `wrfinput`/`wrfbdy` has always
//! ended with a finiteness sweep over every float variable of the file it
//! just wrote. That sweep is the gate that stops a NaN reaching stock
//! WRF, where it surfaces hours later as a blown-up integration with no
//! trace of where it came from.
//!
//! Doing it through `rw_netcdf` instead would be one process launch and
//! one f64 temp file PER VARIABLE -- 168 of each for a wrfinput, whose
//! 3-D fields are tens of megabytes -- on the preparation path, which is
//! the measured critical path to a first plot. Doing it here is one
//! sequential pass over bytes the writer already knows the geometry of.
//!
//! # What it is NOT
//!
//! It does not decode values for a caller, it does not interpret
//! attributes, and it does not touch the NetCDF-4/HDF5 container. It
//! answers exactly one question about a classic file: which float
//! variables contain a value that is not finite. A file too short for its
//! own header geometry is a refusal naming the truncation, because a
//! read-back that quietly stops early would report a corrupt file clean.

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use crate::error::{NcWriteError, Result};
use crate::header::pad4;
use crate::types::{NcFormat, NcType};

const NC_DIMENSION: u32 = 0x0000_000A;
const NC_VARIABLE: u32 = 0x0000_000B;
const NC_ATTRIBUTE: u32 = 0x0000_000C;

/// Bytes pulled per read while sweeping a variable. Bounds the scan's
/// peak memory no matter how wide the domain is; a wrfinput 3-D slab is
/// tens of megabytes and there are dozens of them.
const CHUNK_BYTES: usize = 1 << 20;

/// One variable's geometry, as the header declares it.
#[derive(Debug)]
struct ScanVar {
    name: String,
    ty: NcType,
    /// Elements in the whole variable (fixed) or in one record slab.
    elems: u64,
    slab_bytes: u64,
    /// Absolute offset (fixed) or offset of the slab inside record 0.
    begin: u64,
    is_record: bool,
}

/// What a classic header says about the file's data section.
#[derive(Debug)]
struct ScanHeader {
    numrecs: u64,
    recsize: u64,
    vars: Vec<ScanVar>,
}

/// Which float/double variables of the classic file at `path` hold a
/// non-finite value, in definition order.
///
/// An empty vector is a clean file. Every other outcome -- a container
/// this crate did not write, a truncated data section, an unreadable file
/// -- is an error rather than an empty answer, because "no non-finite
/// values found" and "could not look" must never read the same.
pub fn scan_nonfinite(path: impl AsRef<Path>) -> Result<Vec<String>> {
    let path = path.as_ref();
    let file = File::open(path)?;
    let file_len = file.metadata()?.len();
    let mut reader = BufReader::with_capacity(1 << 16, file);
    let header = parse_header(&mut reader, path)?;

    let mut offending: Vec<String> = Vec::new();
    let mut buffer = vec![0u8; CHUNK_BYTES];
    for var in &header.vars {
        let width = match var.ty {
            NcType::Float | NcType::Double => var.ty.size() as u64,
            _ => continue,
        };
        if var.elems == 0 {
            continue;
        }
        let slabs: Vec<u64> = if var.is_record {
            (0..header.numrecs)
                .map(|rec| var.begin + rec * header.recsize)
                .collect()
        } else {
            vec![var.begin]
        };
        let mut dirty = false;
        for offset in slabs {
            let end = offset + var.slab_bytes;
            if end > file_len {
                return Err(NcWriteError::Usage(format!(
                    "{}: variable '{}' needs bytes up to {end} but the file is \
                     {file_len} bytes long; it is truncated, so the read-back \
                     cannot say whether its values are finite",
                    path.display(),
                    var.name
                )));
            }
            if scan_slab(&mut reader, offset, var.slab_bytes, width, &mut buffer)? {
                dirty = true;
                break;
            }
        }
        if dirty {
            offending.push(var.name.clone());
        }
    }
    Ok(offending)
}

/// True as soon as one non-finite value is seen in the slab at `offset`.
fn scan_slab(
    reader: &mut BufReader<File>,
    offset: u64,
    slab_bytes: u64,
    width: u64,
    buffer: &mut [u8],
) -> Result<bool> {
    reader.seek(SeekFrom::Start(offset))?;
    let mut remaining = slab_bytes;
    while remaining > 0 {
        // Read whole elements per pass, so a value never straddles two
        // buffers and the check below needs no carry state.
        let want = remaining.min((buffer.len() as u64 / width) * width) as usize;
        let slice = &mut buffer[..want];
        reader.read_exact(slice)?;
        let dirty = if width == 4 {
            slice
                .chunks_exact(4)
                .any(|word| !f32::from_be_bytes([word[0], word[1], word[2], word[3]]).is_finite())
        } else {
            slice.chunks_exact(8).any(|word| {
                !f64::from_be_bytes([
                    word[0], word[1], word[2], word[3], word[4], word[5], word[6], word[7],
                ])
                .is_finite()
            })
        };
        if dirty {
            return Ok(true);
        }
        remaining -= want as u64;
    }
    Ok(false)
}

// ---------------------------------------------------------------- header

struct HeaderReader<'a> {
    reader: &'a mut BufReader<File>,
    format: NcFormat,
    label: String,
}

impl HeaderReader<'_> {
    fn u32(&mut self) -> Result<u32> {
        let mut word = [0u8; 4];
        self.reader.read_exact(&mut word)?;
        Ok(u32::from_be_bytes(word))
    }

    /// A `NON_NEG`: 4 bytes, or 8 in CDF-5.
    fn count(&mut self) -> Result<u64> {
        if self.format.wide_counts() {
            let mut word = [0u8; 8];
            self.reader.read_exact(&mut word)?;
            Ok(u64::from_be_bytes(word))
        } else {
            Ok(self.u32()? as u64)
        }
    }

    /// An `OFFSET`: 4 bytes in CDF-1, 8 in CDF-2/CDF-5.
    fn offset(&mut self) -> Result<u64> {
        if self.format.wide_offsets() {
            let mut word = [0u8; 8];
            self.reader.read_exact(&mut word)?;
            Ok(u64::from_be_bytes(word))
        } else {
            Ok(self.u32()? as u64)
        }
    }

    fn skip(&mut self, bytes: u64) -> Result<()> {
        if bytes > 0 {
            self.reader.seek(SeekFrom::Current(bytes as i64))?;
        }
        Ok(())
    }

    fn name(&mut self) -> Result<String> {
        let len = self.count()?;
        let mut raw = vec![0u8; len as usize];
        self.reader.read_exact(&mut raw)?;
        self.skip(pad4(len) - len)?;
        String::from_utf8(raw).map_err(|error| {
            NcWriteError::Usage(format!("{}: a name is not UTF-8: {error}", self.label))
        })
    }

    /// Walk an attribute list without interpreting it.
    fn skip_att_list(&mut self) -> Result<()> {
        let tag = self.u32()?;
        let count = self.count()?;
        if tag == 0 {
            if count != 0 {
                return Err(NcWriteError::Usage(format!(
                    "{}: an ABSENT attribute list declares {count} entries",
                    self.label
                )));
            }
            return Ok(());
        }
        if tag != NC_ATTRIBUTE {
            return Err(NcWriteError::Usage(format!(
                "{}: expected an attribute list tag, read {tag:#x}",
                self.label
            )));
        }
        for _ in 0..count {
            let _name = self.name()?;
            let code = self.u32()?;
            let nelems = self.count()?;
            let ty = NcType::from_code(code).ok_or_else(|| {
                NcWriteError::Usage(format!(
                    "{}: attribute type code {code} is not a classic type",
                    self.label
                ))
            })?;
            let bytes = nelems * ty.size() as u64;
            self.skip(pad4(bytes))?;
        }
        Ok(())
    }
}

fn parse_header(reader: &mut BufReader<File>, path: &Path) -> Result<ScanHeader> {
    let label = path.display().to_string();
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic).map_err(|error| {
        NcWriteError::Usage(format!("{label}: cannot read the 4-byte magic: {error}"))
    })?;
    if &magic[..3] != b"CDF" {
        return Err(NcWriteError::Usage(format!(
            "{label}: magic is {:?}, not a classic 'CDF' container -- the \
             read-back reads what this crate writes, so an HDF5/NetCDF-4 file \
             belongs to rw_netcdf instead",
            &magic[..]
        )));
    }
    let format = match magic[3] {
        1 => NcFormat::Classic,
        2 => NcFormat::Offset64,
        5 => NcFormat::Cdf5,
        other => {
            return Err(NcWriteError::Usage(format!(
                "{label}: container version byte {other} is none of CDF-1/2/5"
            )))
        }
    };
    let mut head = HeaderReader {
        reader,
        format,
        label,
    };

    let numrecs = head.count()?;

    // dim_list
    let mut dim_lens: Vec<u64> = Vec::new();
    let mut record_dimid: Option<usize> = None;
    let tag = head.u32()?;
    let ndims = head.count()?;
    if tag == NC_DIMENSION {
        for index in 0..ndims {
            let _name = head.name()?;
            let len = head.count()?;
            if len == 0 {
                record_dimid = Some(index as usize);
            }
            dim_lens.push(len);
        }
    } else if tag != 0 || ndims != 0 {
        return Err(NcWriteError::Usage(format!(
            "{}: expected a dimension list, read tag {tag:#x} count {ndims}",
            head.label
        )));
    }

    head.skip_att_list()?;

    // var_list
    let mut vars: Vec<ScanVar> = Vec::new();
    let tag = head.u32()?;
    let nvars = head.count()?;
    if tag == NC_VARIABLE {
        for _ in 0..nvars {
            let name = head.name()?;
            let ndims_var = head.count()?;
            let mut dimids: Vec<usize> = Vec::with_capacity(ndims_var as usize);
            for _ in 0..ndims_var {
                dimids.push(head.count()? as usize);
            }
            head.skip_att_list()?;
            let code = head.u32()?;
            let ty = NcType::from_code(code).ok_or_else(|| {
                NcWriteError::Usage(format!(
                    "{}: variable '{name}' has type code {code}, not a classic type",
                    head.label
                ))
            })?;
            let _vsize = head.count()?;
            let begin = head.offset()?;

            let is_record = matches!((dimids.first(), record_dimid), (Some(&d), Some(r)) if d == r);
            let counted = if is_record { &dimids[1..] } else { &dimids[..] };
            let mut elems: u64 = 1;
            for &dimid in counted {
                let len = *dim_lens.get(dimid).ok_or_else(|| {
                    NcWriteError::Usage(format!(
                        "{}: variable '{name}' names dimension {dimid}, which the \
                         header does not define",
                        head.label
                    ))
                })?;
                if Some(dimid) == record_dimid {
                    return Err(NcWriteError::Usage(format!(
                        "{}: variable '{name}' uses the record dimension somewhere \
                         other than first, which the classic format does not allow",
                        head.label
                    )));
                }
                elems = elems.checked_mul(len).ok_or_else(|| {
                    NcWriteError::Capacity(format!(
                        "{}: variable '{name}' element count overflows 64 bits",
                        head.label
                    ))
                })?;
            }
            let slab_bytes = elems * ty.size() as u64;
            vars.push(ScanVar {
                name,
                ty,
                elems,
                slab_bytes,
                begin,
                is_record,
            });
        }
    } else if tag != 0 || nvars != 0 {
        return Err(NcWriteError::Usage(format!(
            "{}: expected a variable list, read tag {tag:#x} count {nvars}",
            head.label
        )));
    }

    // The record stride, by the same rule the layout writes it with: one
    // record variable packs unpadded, more than one pads each slab to 4.
    let record_count = vars.iter().filter(|var| var.is_record).count();
    let recsize = vars
        .iter()
        .filter(|var| var.is_record)
        .map(|var| {
            if record_count == 1 {
                var.slab_bytes
            } else {
                pad4(var.slab_bytes)
            }
        })
        .sum();

    Ok(ScanHeader {
        numrecs,
        recsize,
        vars,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::Schema;
    use crate::types::{AttrValue, VarData};
    use crate::writer::NcWriter;

    fn temp_path(stem: &str) -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "gpuwm-ncwrite-scan-{stem}-{}-{:?}.nc",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        path
    }

    /// A wrfinput-shaped file: a Times char record variable, a float
    /// record variable, an int record variable and a fixed double.
    fn write_sample(path: &Path, format: NcFormat, poison: Option<f32>) {
        let mut schema = Schema::new(format);
        let time = schema.def_dim("Time", 0, true).unwrap();
        let strlen = schema.def_dim("DateStrLen", 19, false).unwrap();
        let south_north = schema.def_dim("south_north", 3, false).unwrap();
        let west_east = schema.def_dim("west_east", 4, false).unwrap();
        schema
            .put_global_attr("TITLE", AttrValue::Text(" OUTPUT FROM GPUWM".into()))
            .unwrap();
        let times = schema.def_var("Times", NcType::Char, &[time, strlen]).unwrap();
        let t2 = schema
            .def_var("T2", NcType::Float, &[time, south_north, west_east])
            .unwrap();
        schema.put_var_attr(t2, "units", AttrValue::Text("K".into())).unwrap();
        let flag = schema.def_var("ITIMESTEP", NcType::Int, &[time]).unwrap();
        let znu = schema.def_var("ZNU", NcType::Double, &[south_north]).unwrap();

        let mut writer = NcWriter::create(path, schema).unwrap();
        let mut field: Vec<f32> = (0..12).map(|value| value as f32).collect();
        if let Some(bad) = poison {
            field[7] = bad;
        }
        writer
            .write_record(0, times, VarData::Char(b"2026-08-18_00:00:00"))
            .unwrap();
        writer.write_record(0, t2, VarData::F32(&field)).unwrap();
        writer.write_record(0, flag, VarData::I32(&[3])).unwrap();
        writer
            .write_var(znu, VarData::F64(&[0.9959, 0.9875, 0.9713]))
            .unwrap();
        writer.finish().unwrap();
    }

    #[test]
    fn a_clean_file_names_nothing() {
        let path = temp_path("clean");
        write_sample(&path, NcFormat::Offset64, None);
        assert_eq!(scan_nonfinite(&path).unwrap(), Vec::<String>::new());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn a_nan_in_a_record_variable_is_named() {
        let path = temp_path("nan");
        write_sample(&path, NcFormat::Offset64, Some(f32::NAN));
        assert_eq!(scan_nonfinite(&path).unwrap(), vec!["T2".to_string()]);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn an_infinity_is_non_finite_too() {
        let path = temp_path("inf");
        write_sample(&path, NcFormat::Offset64, Some(f32::NEG_INFINITY));
        assert_eq!(scan_nonfinite(&path).unwrap(), vec!["T2".to_string()]);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn every_container_version_parses() {
        for format in [NcFormat::Classic, NcFormat::Offset64, NcFormat::Cdf5] {
            let path = temp_path("format");
            write_sample(&path, format, Some(f32::NAN));
            assert_eq!(
                scan_nonfinite(&path).unwrap(),
                vec!["T2".to_string()],
                "{:?} did not parse",
                format
            );
            std::fs::remove_file(&path).ok();
        }
    }

    #[test]
    fn a_fixed_double_variable_is_swept_as_well() {
        let path = temp_path("fixed");
        let mut schema = Schema::new(NcFormat::Offset64);
        let n = schema.def_dim("n", 3, false).unwrap();
        let znu = schema.def_var("ZNU", NcType::Double, &[n]).unwrap();
        let mut writer = NcWriter::create(&path, schema).unwrap();
        writer
            .write_var(znu, VarData::F64(&[1.0, f64::INFINITY, 3.0]))
            .unwrap();
        writer.finish().unwrap();
        assert_eq!(scan_nonfinite(&path).unwrap(), vec!["ZNU".to_string()]);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn every_record_is_swept_not_only_the_first() {
        let path = temp_path("records");
        let mut schema = Schema::new(NcFormat::Offset64);
        let time = schema.def_dim("Time", 0, true).unwrap();
        let n = schema.def_dim("n", 2, false).unwrap();
        let u = schema.def_var("U_BXS", NcType::Float, &[time, n]).unwrap();
        let v = schema.def_var("V_BXS", NcType::Float, &[time, n]).unwrap();
        let mut writer = NcWriter::create(&path, schema).unwrap();
        writer.write_record(0, u, VarData::F32(&[1.0, 2.0])).unwrap();
        writer.write_record(0, v, VarData::F32(&[3.0, 4.0])).unwrap();
        writer.write_record(1, u, VarData::F32(&[5.0, 6.0])).unwrap();
        // The defect lives in the SECOND boundary record only.
        writer
            .write_record(1, v, VarData::F32(&[7.0, f32::NAN]))
            .unwrap();
        writer.finish().unwrap();
        assert_eq!(scan_nonfinite(&path).unwrap(), vec!["V_BXS".to_string()]);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn a_truncated_file_refuses_rather_than_reporting_clean() {
        let path = temp_path("truncated");
        write_sample(&path, NcFormat::Offset64, None);
        let full = std::fs::read(&path).unwrap();
        std::fs::write(&path, &full[..full.len() - 8]).unwrap();
        let error = scan_nonfinite(&path).unwrap_err().to_string();
        assert!(error.contains("truncated"), "got: {error}");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn an_hdf5_file_is_refused_by_name() {
        let path = temp_path("hdf5");
        // The HDF5 signature, which is what a NETCDF4 wrfinput starts with.
        std::fs::write(&path, b"\x89HDF\r\n\x1a\n................").unwrap();
        let error = scan_nonfinite(&path).unwrap_err().to_string();
        assert!(error.contains("rw_netcdf"), "got: {error}");
        std::fs::remove_file(&path).ok();
    }
}
