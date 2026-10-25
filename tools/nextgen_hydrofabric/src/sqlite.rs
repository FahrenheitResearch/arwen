//! Read-only SQLite transport, with one connection per worker.
use crate::Result;
use serde_json::{Map, Value};
use std::ffi::{c_char, c_int, c_void, CStr, CString};
// SQLite is bundled with hydro-proj.
extern crate hydro_proj;
extern "C" {
    fn sqlite3_open_v2(
        p: *const c_char,
        db: *mut *mut c_void,
        flags: c_int,
        vfs: *const c_char,
    ) -> c_int;
    fn sqlite3_close_v2(db: *mut c_void) -> c_int;
    fn sqlite3_prepare_v2(
        db: *mut c_void,
        sql: *const c_char,
        n: c_int,
        s: *mut *mut c_void,
        tail: *mut *const c_char,
    ) -> c_int;
    fn sqlite3_step(s: *mut c_void) -> c_int;
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
}

fn ensure_sqlite() -> Result<()> { Ok(()) }

pub struct Db(*mut c_void);
pub struct Row {
    pub properties: Map<String, Value>,
    pub geometry: Option<Vec<u8>>,
}
impl Db {
    pub fn open(path: &str) -> Result<Self> {
        ensure_sqlite()?;
        unsafe {
            let path = CString::new(path)?;
            let mut p = std::ptr::null_mut();
            let code = sqlite3_open_v2(path.as_ptr(), &mut p, 1 | 0x8000, std::ptr::null());
            let db = Self(p);
            if code != 0 {
                return Err(if p.is_null() {
                    "SQLite open failed".into()
                } else {
                    db.error()
                }
                .into());
            }
            Ok(db)
        }
    }
    fn error(&self) -> String {
        unsafe {
            CStr::from_ptr(sqlite3_errmsg(self.0))
                .to_string_lossy()
                .into_owned()
        }
    }
    pub fn query(&self, sql: &str, geom: Option<&str>) -> Result<Vec<Row>> {
        unsafe {
            let sql = CString::new(sql)?;
            let mut p = std::ptr::null_mut();
            if sqlite3_prepare_v2(self.0, sql.as_ptr(), -1, &mut p, std::ptr::null_mut()) != 0 {
                return Err(self.error().into());
            }
            struct Stmt(*mut c_void);
            impl Drop for Stmt {
                fn drop(&mut self) {
                    unsafe {
                        sqlite3_finalize(self.0);
                    }
                }
            }
            let _stmt = Stmt(p);
            let names = (0..sqlite3_column_count(p))
                .map(|i| {
                    CStr::from_ptr(sqlite3_column_name(p, i))
                        .to_string_lossy()
                        .into_owned()
                })
                .collect::<Vec<_>>();
            let mut rows = Vec::new();
            loop {
                let code = sqlite3_step(p);
                if code == 101 {
                    break;
                }
                if code != 100 {
                    return Err(self.error().into());
                }
                let mut row = Row {
                    properties: Map::new(),
                    geometry: None,
                };
                for (i, name) in names.iter().enumerate() {
                    let i = i as c_int;
                    let v = match sqlite3_column_type(p, i) {
                        1 => Value::from(sqlite3_column_int64(p, i)),
                        2 => Value::from(sqlite3_column_double(p, i)),
                        3 => {
                            let n = sqlite3_column_bytes(p, i) as usize;
                            Value::String(String::from_utf8(
                                std::slice::from_raw_parts(sqlite3_column_text(p, i), n).to_vec(),
                            )?)
                        }
                        4 => {
                            if geom == Some(name.as_str()) {
                                let n = sqlite3_column_bytes(p, i) as usize;
                                row.geometry = Some(if n == 0 {
                                    vec![]
                                } else {
                                    std::slice::from_raw_parts(
                                        sqlite3_column_blob(p, i).cast::<u8>(),
                                        n,
                                    )
                                    .to_vec()
                                })
                            }
                            Value::Null
                        }
                        _ => Value::Null,
                    };
                    if geom != Some(name.as_str()) {
                        row.properties.insert(name.clone(), v);
                    }
                }
                rows.push(row);
            }
            Ok(rows)
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
