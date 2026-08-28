//! WPS intermediate (IFV=5) **reader** -- the inverse of the writer in
//! `src/bin/met_intermediate.rs`, in the same package for the same reason
//! the writer is here: this crate already owns the byte layout, and one
//! layout described twice is one layout that can disagree with itself.
//!
//! Nothing in this module knows what a field MEANS.  It returns every
//! record's header, projection and data exactly as written, so the same
//! reader serves ungrib output, `met_intermediate` output, and the static
//! datasets WPS routes through metgrid's `constants_name` (the Thompson
//! aerosol climatology is one of those, not a special case).  Field
//! selection, stacking and units are the caller's table work.
//!
//! Layout (`WPS/ungrib/src/read_met_module.F90`, big-endian Fortran
//! sequential records, five physical records per field):
//!   1. `version` (i32) -- 5, and only 5 is read here
//!   2. header: hdate[24] xfcst(f32) map_source[32] field[9] units[25]
//!      desc[46] xlvl(f32) nx(i32) ny(i32) iproj(i32)
//!   3. projection: tag[8] + projection-dependent reals
//!   4. `is_wind_earth_rel` (i32)
//!   5. data: nx*ny f32, x fastest

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

/// The version this reader admits.  A different IFV is a refusal, not a
/// guess: the header field offsets moved between versions, so reading a
/// version-4 file with these offsets returns a plausible-looking wrong
/// grid rather than an error.
pub const WPS_FORMAT_VERSION: i32 = 5;

/// One field record's metadata, in the file's own words.
#[derive(Debug, Clone)]
pub struct RecordMeta {
    pub field: String,
    pub xlvl: f32,
    pub nx: usize,
    pub ny: usize,
    pub iproj: i32,
    /// SWCORNER latitude/longitude and the grid increments, as declared by
    /// the file.  For `iproj != 0` these carry the projection's own first
    /// four reals and the caller must interpret them; for `iproj = 0`
    /// (cylindrical equidistant, what the aerosol climatology declares)
    /// they are exactly startlat, startlon, deltalat, deltalon in degrees.
    pub startlat: f32,
    pub startlon: f32,
    pub deltalat: f32,
    pub deltalon: f32,
}

#[derive(Debug)]
pub enum ReadError {
    Io(std::io::Error),
    Truncated,
    RecordMarkerMismatch,
    UnsupportedVersion(i32),
    ShortHeader,
    ShortProjection,
    DataLengthMismatch,
    NonAsciiField,
}

impl std::fmt::Display for ReadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReadError::Io(err) => write!(f, "io error reading WPS intermediate file: {err}"),
            ReadError::Truncated => write!(f, "WPS intermediate file ends mid-record"),
            ReadError::RecordMarkerMismatch => write!(
                f,
                "Fortran sequential record trailer disagrees with its header length"
            ),
            ReadError::UnsupportedVersion(v) => write!(
                f,
                "unsupported WPS intermediate version {v}; this reader ports the IFV=5 layout"
            ),
            ReadError::ShortHeader => write!(f, "WPS intermediate header record is too short"),
            ReadError::ShortProjection => {
                write!(f, "WPS intermediate projection record is too short")
            }
            ReadError::DataLengthMismatch => {
                write!(f, "WPS intermediate data record does not match nx*ny")
            }
            ReadError::NonAsciiField => write!(f, "WPS intermediate field name is not ASCII"),
        }
    }
}

impl From<std::io::Error> for ReadError {
    fn from(err: std::io::Error) -> Self {
        ReadError::Io(err)
    }
}

fn be_i32(bytes: &[u8]) -> i32 {
    i32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

fn be_f32(bytes: &[u8]) -> f32 {
    f32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

/// Read one Fortran sequential record's payload, or `None` at clean EOF.
fn read_record<R: Read>(reader: &mut R) -> Result<Option<Vec<u8>>, ReadError> {
    let mut head = [0u8; 4];
    let mut filled = 0usize;
    while filled < 4 {
        match reader.read(&mut head[filled..])? {
            0 => break,
            n => filled += n,
        }
    }
    if filled == 0 {
        return Ok(None);
    }
    if filled != 4 {
        return Err(ReadError::Truncated);
    }
    let length = be_i32(&head);
    if length < 0 {
        return Err(ReadError::RecordMarkerMismatch);
    }
    let mut payload = vec![0u8; length as usize];
    reader
        .read_exact(&mut payload)
        .map_err(|_| ReadError::Truncated)?;
    let mut tail = [0u8; 4];
    reader
        .read_exact(&mut tail)
        .map_err(|_| ReadError::Truncated)?;
    if be_i32(&tail) != length {
        return Err(ReadError::RecordMarkerMismatch);
    }
    Ok(Some(payload))
}

/// Skip one record without materialising it, returning its payload length.
fn skip_record<R: Read + Seek>(reader: &mut R) -> Result<Option<usize>, ReadError> {
    let mut head = [0u8; 4];
    let mut filled = 0usize;
    while filled < 4 {
        match reader.read(&mut head[filled..])? {
            0 => break,
            n => filled += n,
        }
    }
    if filled == 0 {
        return Ok(None);
    }
    if filled != 4 {
        return Err(ReadError::Truncated);
    }
    let length = be_i32(&head);
    if length < 0 {
        return Err(ReadError::RecordMarkerMismatch);
    }
    reader.seek(SeekFrom::Current(i64::from(length)))?;
    let mut tail = [0u8; 4];
    reader
        .read_exact(&mut tail)
        .map_err(|_| ReadError::Truncated)?;
    if be_i32(&tail) != length {
        return Err(ReadError::RecordMarkerMismatch);
    }
    Ok(Some(length as usize))
}

fn parse_header(header: &[u8]) -> Result<(String, f32, usize, usize, i32), ReadError> {
    // 24 + 4 + 32 = 60 is the field-name offset; 140 begins xlvl.
    if header.len() < 156 {
        return Err(ReadError::ShortHeader);
    }
    let field = std::str::from_utf8(&header[60..69])
        .map_err(|_| ReadError::NonAsciiField)?
        .trim()
        .to_string();
    let xlvl = be_f32(&header[140..144]);
    let nx = be_i32(&header[144..148]);
    let ny = be_i32(&header[148..152]);
    let iproj = be_i32(&header[152..156]);
    if nx <= 0 || ny <= 0 {
        return Err(ReadError::ShortHeader);
    }
    Ok((field, xlvl, nx as usize, ny as usize, iproj))
}

fn parse_projection(proj: &[u8]) -> Result<(f32, f32, f32, f32), ReadError> {
    // tag[8] then the projection reals; for iproj=0 the writer emits
    // startlat, startlon, deltalat, deltalon, earth_radius_km.
    if proj.len() < 24 {
        return Err(ReadError::ShortProjection);
    }
    Ok((
        be_f32(&proj[8..12]),
        be_f32(&proj[12..16]),
        be_f32(&proj[16..20]),
        be_f32(&proj[20..24]),
    ))
}

/// Header-only pass: how many field records, and how many total points.
///
/// Data payloads are seeked over, so this costs headers only -- the whole
/// point of having it separate from [`read_all`] is that a caller can size
/// its buffers without paying for the 225 MB the aerosol climatology puts
/// in data records.
pub fn inventory(path: &Path) -> Result<(usize, usize), ReadError> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut records = 0usize;
    let mut points = 0usize;
    loop {
        let version_payload = match read_record(&mut reader)? {
            None => break,
            Some(payload) => payload,
        };
        if version_payload.len() != 4 {
            return Err(ReadError::ShortHeader);
        }
        let version = be_i32(&version_payload);
        if version != WPS_FORMAT_VERSION {
            return Err(ReadError::UnsupportedVersion(version));
        }
        let header = read_record(&mut reader)?.ok_or(ReadError::Truncated)?;
        let (_field, _xlvl, nx, ny, _iproj) = parse_header(&header)?;
        skip_record(&mut reader)?.ok_or(ReadError::Truncated)?;
        skip_record(&mut reader)?.ok_or(ReadError::Truncated)?;
        let data_len = skip_record(&mut reader)?.ok_or(ReadError::Truncated)?;
        if data_len != 4 * nx * ny {
            return Err(ReadError::DataLengthMismatch);
        }
        records += 1;
        points += nx * ny;
    }
    Ok((records, points))
}

/// Full pass: every record's metadata plus its data, concatenated in file
/// order (`data` holds record 0's `ny*nx` values, then record 1's, ...).
pub fn read_all(path: &Path) -> Result<(Vec<RecordMeta>, Vec<f32>), ReadError> {
    let (records, points) = inventory(path)?;
    let mut metas = Vec::with_capacity(records);
    let mut data = Vec::with_capacity(points);
    let mut reader = BufReader::new(File::open(path)?);
    loop {
        let version_payload = match read_record(&mut reader)? {
            None => break,
            Some(payload) => payload,
        };
        if version_payload.len() != 4 {
            return Err(ReadError::ShortHeader);
        }
        let version = be_i32(&version_payload);
        if version != WPS_FORMAT_VERSION {
            return Err(ReadError::UnsupportedVersion(version));
        }
        let header = read_record(&mut reader)?.ok_or(ReadError::Truncated)?;
        let (field, xlvl, nx, ny, iproj) = parse_header(&header)?;
        let proj = read_record(&mut reader)?.ok_or(ReadError::Truncated)?;
        let (startlat, startlon, deltalat, deltalon) = parse_projection(&proj)?;
        read_record(&mut reader)?.ok_or(ReadError::Truncated)?;
        let payload = read_record(&mut reader)?.ok_or(ReadError::Truncated)?;
        if payload.len() != 4 * nx * ny {
            return Err(ReadError::DataLengthMismatch);
        }
        // Row-major, x fastest -- the writer's own order, so no transpose.
        data.extend(payload.chunks_exact(4).map(be_f32));
        metas.push(RecordMeta {
            field,
            xlvl,
            nx,
            ny,
            iproj,
            startlat,
            startlon,
            deltalat,
            deltalon,
        });
    }
    Ok((metas, data))
}
