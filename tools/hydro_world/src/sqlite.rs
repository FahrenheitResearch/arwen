//! Minimal SQLite transport (read and write), using the bundled static SQLite
//! shared with tools/nextgen_hydrofabric. Startup/prep work only.
use crate::Result;
use std::ffi::{c_char, c_int, c_void, CStr, CString};

// SQLite is bundled with hydro-proj.
extern crate hydro_proj;
extern "C" {
    fn sqlite3_open_v2(p: *const c_char, db: *mut *mut c_void, flags: c_int, vfs: *const c_char) -> c_int;
    fn sqlite3_close_v2(db: *mut c_void) -> c_int;
    fn sqlite3_exec(db: *mut c_void, sql: *const c_char, cb: *const c_void, arg: *mut c_void, err: *mut *mut c_char) -> c_int;
    fn sqlite3_free(p: *mut c_void);
    fn sqlite3_prepare_v2(db: *mut c_void, sql: *const c_char, n: c_int, s: *mut *mut c_void, tail: *mut *const c_char) -> c_int;
    fn sqlite3_step(s: *mut c_void) -> c_int;
    fn sqlite3_reset(s: *mut c_void) -> c_int;
    fn sqlite3_clear_bindings(s: *mut c_void) -> c_int;
    fn sqlite3_finalize(s: *mut c_void) -> c_int;
    fn sqlite3_errmsg(db: *mut c_void) -> *const c_char;
    fn sqlite3_column_count(s: *mut c_void) -> c_int;
    fn sqlite3_column_name(s: *mut c_void, i: c_int) -> *const c_char;
    fn sqlite3_column_type(s: *mut c_void, i: c_int) -> c_int;
    fn sqlite3_column_int64(s: *mut c_void, i: c_int) -> i64;
    fn sqlite3_column_double(s: *mut c_void, i: c_int) -> f64;
    fn sqlite3_column_text(s: *mut c_void, i: c_int) -> *const u8;
    fn sqlite3_column_blob(s: *mut c_void, i: c_int) -> *const c_void;
    fn sqlite3_column_bytes(s: *mut c_void, i: c_int) -> c_int;
    fn sqlite3_bind_int64(s: *mut c_void, i: c_int, v: i64) -> c_int;
    fn sqlite3_bind_double(s: *mut c_void, i: c_int, v: f64) -> c_int;
    fn sqlite3_bind_text(s: *mut c_void, i: c_int, v: *const c_char, n: c_int, d: isize) -> c_int;
    fn sqlite3_bind_blob(s: *mut c_void, i: c_int, v: *const c_void, n: c_int, d: isize) -> c_int;
    fn sqlite3_bind_null(s: *mut c_void, i: c_int) -> c_int;
}

fn ensure_sqlite() -> Result<()> { Ok(()) }

const SQLITE_TRANSIENT: isize = -1;

/// One SQLite value with its storage class preserved.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Null,
    Integer(i64),
    Real(f64),
    Text(String),
    Blob(Vec<u8>),
}

pub struct Db(*mut c_void);
pub struct Rows {
    #[allow(dead_code)]
    pub columns: Vec<String>,
    pub rows: Vec<Vec<Value>>,
}

struct Stmt(*mut c_void);
impl Drop for Stmt {
    fn drop(&mut self) {
        unsafe {
            sqlite3_finalize(self.0);
        }
    }
}

impl Db {
    pub fn open_read(path: &str) -> Result<Self> {
        Self::open(path, 0x1 | 0x8000)
    }
    pub fn create(path: &str) -> Result<Self> {
        if std::path::Path::new(path).exists() {
            return Err(format!("refusing to overwrite existing {path}").into());
        }
        Self::open(path, 0x2 | 0x4 | 0x8000)
    }
    fn open(path: &str, flags: c_int) -> Result<Self> {
        ensure_sqlite()?;
        unsafe {
            let c = CString::new(path)?;
            let mut p = std::ptr::null_mut();
            let code = sqlite3_open_v2(c.as_ptr(), &mut p, flags, std::ptr::null());
            let db = Self(p);
            if code != 0 {
                return Err(if p.is_null() { "SQLite open failed".to_string() } else { db.error() }.into());
            }
            Ok(db)
        }
    }
    fn error(&self) -> String {
        unsafe { CStr::from_ptr(sqlite3_errmsg(self.0)).to_string_lossy().into_owned() }
    }
    pub fn exec(&self, sql: &str) -> Result<()> {
        unsafe {
            let c = CString::new(sql)?;
            let mut err = std::ptr::null_mut();
            if sqlite3_exec(self.0, c.as_ptr(), std::ptr::null(), std::ptr::null_mut(), &mut err) != 0 {
                let msg = if err.is_null() {
                    self.error()
                } else {
                    let m = CStr::from_ptr(err).to_string_lossy().into_owned();
                    sqlite3_free(err.cast());
                    m
                };
                return Err(format!("{msg} in: {sql}").into());
            }
            Ok(())
        }
    }
    fn prepare(&self, sql: &str) -> Result<Stmt> {
        unsafe {
            let c = CString::new(sql)?;
            let mut p = std::ptr::null_mut();
            if sqlite3_prepare_v2(self.0, c.as_ptr(), -1, &mut p, std::ptr::null_mut()) != 0 {
                return Err(format!("{} in: {sql}", self.error()).into());
            }
            Ok(Stmt(p))
        }
    }
    pub fn query(&self, sql: &str) -> Result<Rows> {
        let mut rows = Vec::new();
        let columns = self.query_chunks(sql, 4096, |part| {
            rows.extend(part);
            Ok(())
        })?;
        Ok(Rows { columns, rows })
    }
    /// Scan once, retaining only a bounded raw-value block beside the typed
    /// output. A continental geometry table must not be duplicated in memory.
    pub fn query_chunks(
        &self,
        sql: &str,
        chunk_size: usize,
        mut consume: impl FnMut(Vec<Vec<Value>>) -> Result<()>,
    ) -> Result<Vec<String>> {
        if chunk_size == 0 {
            return Err("SQLite scan block must be nonzero".into());
        }
        unsafe {
            let s = self.prepare(sql)?;
            let p = s.0;
            let columns = (0..sqlite3_column_count(p))
                .map(|i| {
                    CStr::from_ptr(sqlite3_column_name(p, i))
                        .to_string_lossy()
                        .into_owned()
                })
                .collect::<Vec<_>>();
            let mut rows = Vec::with_capacity(chunk_size);
            loop {
                match sqlite3_step(p) {
                    101 => break,
                    100 => {}
                    _ => return Err(self.error().into()),
                }
                let mut row = Vec::with_capacity(columns.len());
                for i in 0..columns.len() as c_int {
                    let n = sqlite3_column_bytes(p, i).max(0) as usize;
                    row.push(match sqlite3_column_type(p, i) {
                        1 => Value::Integer(sqlite3_column_int64(p, i)),
                        2 => Value::Real(sqlite3_column_double(p, i)),
                        3 => Value::Text(String::from_utf8(
                            std::slice::from_raw_parts(sqlite3_column_text(p, i), n).to_vec(),
                        )?),
                        4 => Value::Blob(if n == 0 {
                            vec![]
                        } else {
                            std::slice::from_raw_parts(sqlite3_column_blob(p, i).cast::<u8>(), n)
                                .to_vec()
                        }),
                        _ => Value::Null,
                    });
                }
                rows.push(row);
                if rows.len() == chunk_size {
                    consume(std::mem::replace(&mut rows, Vec::with_capacity(chunk_size)))?;
                }
            }
            if !rows.is_empty() {
                consume(rows)?;
            }
            Ok(columns)
        }
    }
    /// Insert every row with one prepared statement (caller owns the transaction).
    pub fn insert_all(&self, sql: &str, rows: &[Vec<Value>]) -> Result<()> {
        unsafe {
            let s = self.prepare(sql)?;
            let p = s.0;
            for row in rows {
                for (i, v) in row.iter().enumerate() {
                    let i = i as c_int + 1;
                    let rc = match v {
                        Value::Null => sqlite3_bind_null(p, i),
                        Value::Integer(x) => sqlite3_bind_int64(p, i, *x),
                        Value::Real(x) => sqlite3_bind_double(p, i, *x),
                        Value::Text(t) => sqlite3_bind_text(p, i, t.as_ptr().cast(), c_int::try_from(t.len())?, SQLITE_TRANSIENT),
                        Value::Blob(b) => sqlite3_bind_blob(p, i, b.as_ptr().cast(), c_int::try_from(b.len())?, SQLITE_TRANSIENT),
                    };
                    if rc != 0 {
                        return Err(self.error().into());
                    }
                }
                if sqlite3_step(p) != 101 {
                    return Err(format!("{} in: {sql}", self.error()).into());
                }
                sqlite3_reset(p);
                sqlite3_clear_bindings(p);
            }
            Ok(())
        }
    }
}
impl Drop for Db {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe {
                sqlite3_close_v2(self.0);
            }
        }
    }
}
pub fn ident(s: &str) -> String {
    format!("\"{}\"", s.replace('"', "\"\""))
}
pub fn literal(s: &str) -> String {
    format!("'{}'", s.replace('\'', "''"))
}

#[cfg(test)]
mod bounded_scan_tests {
    use super::*;
    #[test]
    fn continental_scan_preserves_values_and_order_across_blocks() {
        let path = std::env::temp_dir().join(format!("hydro-scan-{}.sqlite", std::process::id()));
        let db = Db::create(path.to_str().unwrap()).unwrap();
        db.exec(
            "CREATE TABLE data (id INTEGER, real REAL, text TEXT, blob BLOB, absent TEXT); BEGIN",
        )
        .unwrap();
        let rows: Vec<_> = (0..8193)
            .map(|i| {
                vec![
                    Value::Integer(i),
                    Value::Real(i as f64 / 7.0),
                    Value::Text(format!("row-{i}-é")),
                    Value::Blob((i as u64).to_le_bytes().to_vec()),
                    Value::Null,
                ]
            })
            .collect();
        db.insert_all("INSERT INTO data VALUES (?,?,?,?,?)", &rows)
            .unwrap();
        db.exec("COMMIT").unwrap();
        let mut seen = Vec::new();
        let mut lengths = Vec::new();
        db.query_chunks("SELECT * FROM data", 4096, |part| {
            lengths.push(part.len());
            seen.extend(part);
            Ok(())
        })
        .unwrap();
        assert_eq!(lengths, [4096, 4096, 1]);
        assert_eq!(seen, rows);
        assert_eq!(db.query("SELECT * FROM data").unwrap().rows, rows);
        assert!(db
            .query_chunks("SELECT * FROM data", 0, |_| Ok(()))
            .is_err());
        drop(db);
        std::fs::remove_file(path).unwrap();
    }
}
