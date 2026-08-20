//! Reading the WPS "intermediate" format, version 5.
//!
//! This is the counterpart of the writer in `tools/grib1_bridge`'s
//! `met_intermediate` binary: the same Fortran sequential-unformatted
//! container, read rather than written, so an init can be built straight from
//! a file this project produced without a Fortran program in the chain.
//!
//! ## Container
//! Every record is a big-endian byte count, the payload, then the same count
//! again.  One field is five records:
//!
//! ```text
//! 1  version                       i32   (5 for WPS; nothing else is read)
//! 2  hdate(24) xfcst(f32) map_source(32) field(9) units(25) desc(46)
//!    xlvl(f32) nx(i32) ny(i32) iproj(i32)
//! 3  projection block, shape depending on iproj
//! 4  is_wind_grid_rel              i32
//! 5  slab                          nx*ny f32, x fastest (Fortran order)
//! ```
//!
//! ## Deliberate narrowness
//! Only `iproj == 0` (cylindrical equidistant, the projection every global
//! first-guess this lane consumes is on) is accepted.  A Lambert, Mercator,
//! polar-stereographic or Gaussian slab is **refused by name** rather than
//! read with a projection this module cannot invert: `mpas_init_atm_llxy.F`
//! carries six more projections and each is its own `latlon_to_ij`, so
//! accepting the bytes and guessing the geometry would put a structurally
//! perfect file with fields sampled at the wrong points into the chain.  The
//! refusal names the projection code and the field that carried it.

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use crate::error::{MpasError, MpasResult};

/// The only intermediate-format version this reader accepts.
pub const WPS_FORMAT_VERSION: i32 = 5;

/// Cylindrical equidistant, the one projection code this module inverts.
pub const IPROJ_LATLON: i32 = 0;

/// One field at one level, exactly as the file carries it.
#[derive(Debug, Clone)]
pub struct MetSlab {
    /// `YYYY-MM-DD_HH:MM:SS`, trimmed.
    pub hdate: String,
    /// The producing centre's own 32-character label.
    pub map_source: String,
    /// `TT`, `UU`, `SOILT001`, ... trimmed.
    pub field: String,
    pub units: String,
    pub desc: String,
    /// The level tag.  `200100.0` marks the surface level; pressure levels
    /// carry their pressure in Pa.
    pub xlvl: f32,
    pub nx: usize,
    pub ny: usize,
    pub start_lat: f32,
    pub start_lon: f32,
    pub delta_lat: f32,
    pub delta_lon: f32,
    pub earth_radius_km: f32,
    /// `nx * ny` values, x fastest.
    pub values: Vec<f32>,
}

impl MetSlab {
    /// Value at the one-based `(i, j)` the Fortran indexes with.
    #[inline]
    pub fn at(&self, i: usize, j: usize) -> f32 {
        self.values[(j - 1) * self.nx + (i - 1)]
    }
}

fn read_record<R: Read>(r: &mut R) -> MpasResult<Option<Vec<u8>>> {
    let mut head = [0u8; 4];
    match r.read_exact(&mut head) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e.into()),
    }
    let len = u32::from_be_bytes(head) as usize;
    let mut payload = vec![0u8; len];
    r.read_exact(&mut payload)?;
    let mut tail = [0u8; 4];
    r.read_exact(&mut tail)?;
    let trailer = u32::from_be_bytes(tail) as usize;
    if trailer != len {
        return Err(MpasError::Refusal(format!(
            "intermediate record is framed {len} bytes at its head and {trailer} at its tail; \
             the file is not a Fortran sequential-unformatted stream this reader can trust"
        )));
    }
    Ok(Some(payload))
}

struct Cursor<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Cursor { bytes, at: 0 }
    }

    fn take(&mut self, n: usize, what: &str) -> MpasResult<&'a [u8]> {
        if self.at + n > self.bytes.len() {
            return Err(MpasError::Refusal(format!(
                "intermediate record ran out reading {what}: wanted {n} byte(s) at offset {}, \
                 record is {} byte(s)",
                self.at,
                self.bytes.len()
            )));
        }
        let out = &self.bytes[self.at..self.at + n];
        self.at += n;
        Ok(out)
    }

    fn text(&mut self, n: usize, what: &str) -> MpasResult<String> {
        let raw = self.take(n, what)?;
        Ok(String::from_utf8_lossy(raw).trim().to_string())
    }

    fn f32(&mut self, what: &str) -> MpasResult<f32> {
        let raw = self.take(4, what)?;
        Ok(f32::from_be_bytes([raw[0], raw[1], raw[2], raw[3]]))
    }

    fn i32(&mut self, what: &str) -> MpasResult<i32> {
        let raw = self.take(4, what)?;
        Ok(i32::from_be_bytes([raw[0], raw[1], raw[2], raw[3]]))
    }
}

/// Read every field in one intermediate file, in file order.
///
/// File order is load-bearing downstream: the level table the case-7 code
/// builds is "first sighting wins", so re-ordering the slabs re-orders the
/// first-guess levels and moves every vertically interpolated number.
pub fn read_met_file(path: &Path) -> MpasResult<Vec<MetSlab>> {
    let file = File::open(path)?;
    let mut reader = BufReader::with_capacity(8 << 20, file);
    let mut out: Vec<MetSlab> = Vec::new();

    loop {
        let Some(version_record) = read_record(&mut reader)? else {
            break;
        };
        let mut c = Cursor::new(&version_record);
        let version = c.i32("format version")?;
        if version != WPS_FORMAT_VERSION {
            return Err(MpasError::Refusal(format!(
                "{} carries intermediate format version {version}; this reader implements \
                 version {WPS_FORMAT_VERSION} (WPS) only, and versions 3 and 4 lay their \
                 header out differently",
                path.display()
            )));
        }

        let header = read_record(&mut reader)?.ok_or_else(|| {
            MpasError::Refusal(format!(
                "{} ends after a version record, with no field header behind it",
                path.display()
            ))
        })?;
        let mut c = Cursor::new(&header);
        let hdate = c.text(24, "hdate")?;
        let _xfcst = c.f32("xfcst")?;
        let map_source = c.text(32, "map_source")?;
        let mut field = c.text(9, "field")?;
        let units = c.text(25, "units")?;
        let desc = c.text(46, "desc")?;
        let xlvl = c.f32("xlvl")?;
        let nx = c.i32("nx")?;
        let ny = c.i32("ny")?;
        let iproj = c.i32("iproj")?;

        // read_met's own aliasing, reproduced: some producers spell the
        // geopotential height field HGT and MPAS only knows it as GHT.
        if field == "HGT" {
            field = "GHT".to_string();
        }

        if nx <= 0 || ny <= 0 {
            return Err(MpasError::Refusal(format!(
                "field {field} declares a {nx} x {ny} slab"
            )));
        }
        let nx = nx as usize;
        let ny = ny as usize;

        if iproj != IPROJ_LATLON {
            return Err(MpasError::Refusal(format!(
                "field {field} is on projection code {iproj}; this reader inverts only code \
                 {IPROJ_LATLON} (cylindrical equidistant).  Codes 1 (Mercator), 3 (Lambert), \
                 4 (Gaussian) and 5 (polar stereographic) each need their own latlon_to_ij \
                 out of mpas_init_atm_llxy.F, and sampling a slab through the wrong one \
                 produces a structurally perfect file whose fields were read at the wrong \
                 points"
            )));
        }

        let proj = read_record(&mut reader)?.ok_or_else(|| {
            MpasError::Refusal(format!("field {field} has no projection record"))
        })?;
        let mut c = Cursor::new(&proj);
        let _startloc = c.text(8, "startloc")?;
        let start_lat = c.f32("startlat")?;
        let start_lon = c.f32("startlon")?;
        let delta_lat = c.f32("deltalat")?;
        let delta_lon = c.f32("deltalon")?;
        let earth_radius_km = c.f32("earth_radius")?;

        let _wind_rel = read_record(&mut reader)?.ok_or_else(|| {
            MpasError::Refusal(format!("field {field} has no is_wind_grid_rel record"))
        })?;

        let slab = read_record(&mut reader)?
            .ok_or_else(|| MpasError::Refusal(format!("field {field} has no data record")))?;
        let want = nx * ny * 4;
        if slab.len() != want {
            return Err(MpasError::Refusal(format!(
                "field {field} declares {nx} x {ny} but its data record is {} byte(s), not {want}",
                slab.len()
            )));
        }
        let mut values = Vec::with_capacity(nx * ny);
        for chunk in slab.chunks_exact(4) {
            values.push(f32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }

        out.push(MetSlab {
            hdate,
            map_source,
            field,
            units,
            desc,
            xlvl,
            nx,
            ny,
            start_lat,
            start_lon,
            delta_lat,
            delta_lon,
            earth_radius_km,
            values,
        });
    }

    if out.is_empty() {
        return Err(MpasError::Refusal(format!(
            "{} holds no intermediate records",
            path.display()
        )));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn record(payload: &[u8]) -> Vec<u8> {
        let n = payload.len() as u32;
        let mut out = n.to_be_bytes().to_vec();
        out.extend_from_slice(payload);
        out.extend_from_slice(&n.to_be_bytes());
        out
    }

    fn fixed(text: &str, width: usize) -> Vec<u8> {
        let mut out = text.as_bytes().to_vec();
        out.truncate(width);
        out.resize(width, b' ');
        out
    }

    fn one_field(name: &str, iproj: i32, nx: usize, ny: usize, values: &[f32]) -> Vec<u8> {
        let mut out = record(&WPS_FORMAT_VERSION.to_be_bytes());
        let mut header = Vec::new();
        header.extend_from_slice(&fixed("2025-03-14_12:00:00", 24));
        header.extend_from_slice(&0f32.to_be_bytes());
        header.extend_from_slice(&fixed("ECMWF", 32));
        header.extend_from_slice(&fixed(name, 9));
        header.extend_from_slice(&fixed("K", 25));
        header.extend_from_slice(&fixed("Temperature", 46));
        header.extend_from_slice(&200100f32.to_be_bytes());
        header.extend_from_slice(&(nx as i32).to_be_bytes());
        header.extend_from_slice(&(ny as i32).to_be_bytes());
        header.extend_from_slice(&iproj.to_be_bytes());
        out.extend_from_slice(&record(&header));

        let mut proj = Vec::new();
        proj.extend_from_slice(&fixed("SWCORNER", 8));
        proj.extend_from_slice(&(-90f32).to_be_bytes());
        proj.extend_from_slice(&0f32.to_be_bytes());
        proj.extend_from_slice(&0.25f32.to_be_bytes());
        proj.extend_from_slice(&0.25f32.to_be_bytes());
        proj.extend_from_slice(&6371.229f32.to_be_bytes());
        out.extend_from_slice(&record(&proj));
        out.extend_from_slice(&record(&0i32.to_be_bytes()));

        let mut data = Vec::new();
        for v in values {
            data.extend_from_slice(&v.to_be_bytes());
        }
        out.extend_from_slice(&record(&data));
        out
    }

    fn write_temp(name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(name);
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(bytes).unwrap();
        path
    }

    #[test]
    fn a_latlon_field_round_trips_out_of_the_container() {
        let values = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let path = write_temp(
            "rwmpas_metfile_latlon.bin",
            &one_field("TT", IPROJ_LATLON, 3, 2, &values),
        );
        let slabs = read_met_file(&path).unwrap();
        assert_eq!(slabs.len(), 1);
        let s = &slabs[0];
        assert_eq!(s.field, "TT");
        assert_eq!(s.map_source, "ECMWF");
        assert_eq!((s.nx, s.ny), (3, 2));
        // Fortran order: x fastest.  (2,1) is the second value.
        assert_eq!(s.at(2, 1), 2.0);
        assert_eq!(s.at(1, 2), 4.0);
        assert_eq!(s.xlvl, 200100.0);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn hgt_is_read_under_the_name_the_case_code_uses() {
        let path = write_temp(
            "rwmpas_metfile_hgt.bin",
            &one_field("HGT", IPROJ_LATLON, 2, 1, &[10.0, 20.0]),
        );
        let slabs = read_met_file(&path).unwrap();
        assert_eq!(slabs[0].field, "GHT");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_projection_this_reader_cannot_invert_is_refused_by_code() {
        // Lambert conformal.  Reading its slab through the cylindrical
        // equidistant inverse would sample every cell at the wrong point and
        // say nothing about it.
        let path = write_temp(
            "rwmpas_metfile_lambert.bin",
            &one_field("TT", 3, 2, 1, &[1.0, 2.0]),
        );
        let err = read_met_file(&path).unwrap_err().to_string();
        assert!(err.contains("projection code 3"), "{err}");
        assert!(err.contains("read at the wrong"), "{err}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_mismatched_record_trailer_is_refused() {
        let mut bytes = one_field("TT", IPROJ_LATLON, 2, 1, &[1.0, 2.0]);
        let n = bytes.len();
        bytes[n - 1] ^= 0xff;
        let path = write_temp("rwmpas_metfile_torn.bin", &bytes);
        let err = read_met_file(&path).unwrap_err().to_string();
        assert!(err.contains("at its tail"), "{err}");
        let _ = std::fs::remove_file(&path);
    }
}
