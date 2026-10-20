//! Fixed-width dBASE III tables, decoded in Rust. Character values stay strings
//! so gauge IDs retain leading zeroes. No inferred column or missing-value repair.
use crate::Result;
use rayon::prelude::*;
use serde::Serialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::Path;
#[derive(Serialize)]
pub struct Field {
    pub name: String,
    pub kind: char,
    pub width: usize,
    pub decimals: u8,
    offset: usize,
}
pub struct Table {
    bytes: Vec<u8>,
    pub fields: Vec<Field>,
    pub count: usize,
    header: usize,
    stride: usize,
    pub sha256: String,
}
impl Table {
    pub fn open(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        if bytes.len() < 33 || bytes[0] != 3 {
            return Err("expected dBASE III no-memo table".into());
        }
        let count = u32::from_le_bytes(bytes[4..8].try_into()?) as usize;
        let header = u16::from_le_bytes(bytes[8..10].try_into()?) as usize;
        let stride = u16::from_le_bytes(bytes[10..12].try_into()?) as usize;
        if header < 33
            || (header - 33) % 32 != 0
            || header > bytes.len()
            || bytes[header - 1] != 13
            || stride == 0
        {
            return Err("invalid DBF header/field boundary".into());
        }
        let end = header
            .checked_add(count.checked_mul(stride).ok_or("DBF size overflow")?)
            .ok_or("DBF size overflow")?;
        if end > bytes.len()
            || (bytes.len() != end && !(bytes.len() == end + 1 && bytes[end] == 26))
        {
            return Err("DBF record count/length does not match file".into());
        }
        let mut offset = 1usize;
        let mut fields = Vec::new();
        for raw in bytes[32..header - 1].chunks_exact(32) {
            let name = std::str::from_utf8(&raw[..11])?
                .trim_end_matches('\0')
                .to_owned();
            if fields.iter().any(|f: &Field| f.name == name) {
                return Err("duplicate DBF field".into());
            }
            let width = raw[16] as usize;
            fields.push(Field {
                name,
                kind: raw[11] as char,
                width,
                decimals: raw[17],
                offset,
            });
            offset += width;
        }
        if offset != stride {
            return Err("DBF fields do not cover record".into());
        }
        let sha256 = format!("{:x}", Sha256::digest(&bytes));
        Ok(Self {
            bytes,
            fields,
            count,
            header,
            stride,
            sha256,
        })
    }
    pub fn column(&self, name: &str) -> Result<usize> {
        self.fields
            .iter()
            .position(|f| f.name == name)
            .ok_or_else(|| format!("DBF column absent: {name}").into())
    }
    fn row(&self, index: usize, columns: &[usize]) -> Result<Option<Vec<String>>> {
        let row =
            &self.bytes[self.header + index * self.stride..self.header + (index + 1) * self.stride];
        if row[0] == b'*' {
            return Ok(None);
        }
        if row[0] != b' ' {
            return Err("invalid DBF deletion marker".into());
        }
        Ok(Some(
            columns
                .iter()
                .map(|&c| {
                    let f = &self.fields[c];
                    if !matches!(f.kind, 'C' | 'N' | 'F' | 'L' | 'D') {
                        return Err(format!("unsupported DBF column type {}", f.kind).into());
                    }
                    Ok(std::str::from_utf8(&row[f.offset..f.offset + f.width])?
                        .trim_matches(' ')
                        .to_owned())
                })
                .collect::<Result<Vec<_>>>()?,
        ))
    }
    /// Preserve physical record numbers, including deleted rows, for SHP/DBF joins.
    pub fn indexed_rows(&self, names: &[&str]) -> Result<Vec<Option<Vec<String>>>> {
        let cols = names
            .iter()
            .map(|n| self.column(n))
            .collect::<Result<Vec<_>>>()?;
        (0..self.count)
            .into_par_iter()
            .map(|i| self.row(i, &cols))
            .collect()
    }
    /// Copy selected records verbatim; the dBASE schema and numeric text survive.
    pub fn subset_bytes(&self, indices: &[usize]) -> Result<Vec<u8>> {
        let count = u32::try_from(indices.len())?;
        let mut out = self.bytes[..self.header].to_vec();
        out[4..8].copy_from_slice(&count.to_le_bytes());
        for &i in indices {
            if i >= self.count {
                return Err("DBF subset index out of range".into());
            }
            out.extend_from_slice(
                &self.bytes[self.header + i * self.stride..self.header + (i + 1) * self.stride],
            );
        }
        out.push(26);
        Ok(out)
    }
    pub fn rows(&self, names: &[&str]) -> Result<Vec<Vec<String>>> {
        let cols = names
            .iter()
            .map(|n| self.column(n))
            .collect::<Result<Vec<_>>>()?;
        let rows = (0..self.count)
            .into_par_iter()
            .map(|i| self.row(i, &cols))
            .collect::<Result<Vec<_>>>()?;
        Ok(rows.into_iter().flatten().collect())
    }
    pub fn inspect(&self) -> Result<Value> {
        let columns = (0..self.fields.len()).collect::<Vec<_>>();
        let sample = (0..self.count.min(3))
            .map(|i| self.row(i, &columns))
            .collect::<Result<Vec<_>>>()?;
        Ok(json!({"sha256":self.sha256,"records":self.count,"fields":self.fields,"sample":sample}))
    }
}
pub fn query(path: &Path) -> Result<Value> {
    #[derive(serde::Deserialize)]
    struct Query {
        path: std::path::PathBuf,
        sha256: String,
        columns: Vec<String>,
        where_column: String,
        equals: Vec<String>,
    }
    let q: Query = serde_json::from_slice(&std::fs::read(path)?)?;
    let t = Table::open(&q.path)?;
    if t.sha256 != q.sha256 {
        return Err("DBF query source digest mismatch".into());
    }
    let mut cols = vec![q.where_column.as_str()];
    cols.extend(q.columns.iter().map(String::as_str));
    let values = q.equals.iter().collect::<std::collections::BTreeSet<_>>();
    let rows = t
        .rows(&cols)?
        .into_par_iter()
        .filter(|r| values.contains(&r[0]))
        .map(|r| r[1..].to_vec())
        .collect::<Vec<_>>();
    Ok(json!({"path":q.path,"sha256":q.sha256,"columns":q.columns,"rows":rows}))
}
