//! ESRI shapefile (.shp/.shx) and dBASE (.dbf) reader for vector inputs
//! (HydroLAKES points, HydroRIVERS lines, HydroBASINS polygons). Records are
//! decoded in parallel through the .shx offsets; attribute columns are pulled
//! by name, so a dataset is a column mapping in the source table, not code.
use crate::Result;
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone)]
pub enum Shape {
    Null,
    Point(f64, f64),
    /// Parts of a polyline or polygon rings, each a vertex list (x = lon, y = lat).
    Parts(Vec<Vec<(f64, f64)>>),
}

pub struct Dbf {
    bytes: Vec<u8>,
    pub n: usize,
    header_len: usize,
    rec_len: usize,
    fields: HashMap<String, (usize, usize, u8)>,
}

impl Dbf {
    pub fn open(path: &Path) -> Result<Dbf> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        if bytes.len() < 32 {
            return Err(format!("{}: dbf truncated", path.display()).into());
        }
        let n = u32::from_le_bytes(bytes[4..8].try_into()?) as usize;
        let header_len = u16::from_le_bytes(bytes[8..10].try_into()?) as usize;
        let rec_len = u16::from_le_bytes(bytes[10..12].try_into()?) as usize;
        let mut fields = HashMap::new();
        let mut o = 32;
        let mut pos = 1; // deletion flag
        while o + 32 <= header_len && bytes[o] != 0x0D {
            let name = String::from_utf8_lossy(&bytes[o..o + 11]).trim_end_matches('\0').trim().to_string();
            let ty = bytes[o + 11];
            let len = bytes[o + 16] as usize;
            fields.insert(name, (pos, len, ty));
            pos += len;
            o += 32;
        }
        if bytes.len() < header_len + n * rec_len {
            return Err(format!("{}: dbf shorter than its header says", path.display()).into());
        }
        Ok(Dbf { bytes, n, header_len, rec_len, fields })
    }
    fn raw(&self, field: &str) -> Result<(usize, usize)> {
        // NHDPlus editions vary DBF field-name case between VPUs. dBASE
        // identifiers are case-insensitive; the semantic mapping stays in the table.
        let &(p, l, _) = self.fields.get(field)
            .or_else(|| self.fields.iter().find(|(name,_)| name.eq_ignore_ascii_case(field)).map(|(_,v)|v))
            .ok_or_else(|| format!("dbf column {field} missing"))?;
        Ok((p, l))
    }
    /// Numeric column as f64 for every record (blank -> NaN), in parallel.
    pub fn f64s(&self, field: &str) -> Result<Vec<f64>> {
        let (p, l) = self.raw(field)?;
        Ok((0..self.n)
            .into_par_iter()
            .map(|i| {
                let o = self.header_len + i * self.rec_len + p;
                let s = std::str::from_utf8(&self.bytes[o..o + l]).unwrap_or("").trim();
                if s.is_empty() { f64::NAN } else { s.parse::<f64>().unwrap_or(f64::NAN) }
            })
            .collect())
    }
    /// Integer column as i64 (blank -> i64::MIN), in parallel.
    pub fn i64s(&self, field: &str) -> Result<Vec<i64>> {
        let (p, l) = self.raw(field)?;
        Ok((0..self.n)
            .into_par_iter()
            .map(|i| {
                let o = self.header_len + i * self.rec_len + p;
                let s = std::str::from_utf8(&self.bytes[o..o + l]).unwrap_or("").trim();
                s.parse::<i64>().or_else(|_| s.parse::<f64>().map(|x| x as i64)).unwrap_or(i64::MIN)
            })
            .collect())
    }
    pub fn strings(&self, field: &str) -> Result<Vec<String>> {
        let (p, l) = self.raw(field)?;
        Ok((0..self.n)
            .into_par_iter()
            .map(|i| {
                let o = self.header_len + i * self.rec_len + p;
                String::from_utf8_lossy(&self.bytes[o..o + l]).trim().to_string()
            })
            .collect())
    }
}

/// Read every shape of a .shp using its .shx index (parallel decode).
pub fn read_shapes(shp: &Path) -> Result<Vec<Shape>> {
    let b = std::fs::read(shp).map_err(|e| format!("{}: {e}", shp.display()))?;
    let shx = std::fs::read(shp.with_extension("shx")).map_err(|e| format!("{}.shx: {e}", shp.display()))?;
    if shx.len() < 100 || b.len() < 100 {
        return Err("shapefile truncated".into());
    }
    let n = (shx.len() - 100) / 8;
    (0..n)
        .into_par_iter()
        .map(|i| -> Result<Shape> {
            let o = u32::from_be_bytes(shx[100 + 8 * i..104 + 8 * i].try_into()?) as usize * 2;
            let len = u32::from_be_bytes(shx[104 + 8 * i..108 + 8 * i].try_into()?) as usize * 2;
            let c = b.get(o + 8..o + 8 + len).ok_or("shp record out of range")?;
            let ty = i32::from_le_bytes(c[0..4].try_into()?);
            let f = |k: usize| -> Result<f64> { Ok(f64::from_le_bytes(c.get(k..k + 8).ok_or("shp truncated")?.try_into()?)) };
            Ok(match ty {
                0 => Shape::Null,
                1 | 11 | 21 => Shape::Point(f(4)?, f(12)?),
                3 | 5 | 13 | 15 | 23 | 25 => {
                    let np = i32::from_le_bytes(c[36..40].try_into()?) as usize;
                    let npt = i32::from_le_bytes(c[40..44].try_into()?) as usize;
                    let parts = (0..np).map(|k| Ok(i32::from_le_bytes(c[44 + 4 * k..48 + 4 * k].try_into()?) as usize)).collect::<Result<Vec<_>>>()?;
                    let base = 44 + 4 * np;
                    let mut out = Vec::with_capacity(np);
                    for k in 0..np {
                        let (s, e) = (parts[k], if k + 1 < np { parts[k + 1] } else { npt });
                        out.push((s..e).map(|j| Ok((f(base + 16 * j)?, f(base + 16 * j + 8)?))).collect::<Result<Vec<_>>>()?);
                    }
                    Shape::Parts(out)
                }
                t => return Err(format!("shape type {t} is not supported").into()),
            })
        })
        .collect()
}

#[cfg(test)]
mod native_tests {
    use super::*;
    #[test]
    fn native_vpu_column_case_is_not_a_different_field() {
        let d=Dbf { bytes:b" 42".to_vec(),n:1,header_len:0,rec_len:3,
            fields:HashMap::from([("ComID".into(),(1,2,b'N'))]) };
        assert_eq!(d.i64s("COMID").unwrap(),vec![42]);
        assert!(d.i64s("FEATUREID").is_err());
    }
}
