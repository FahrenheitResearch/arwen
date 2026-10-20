//! Minimal SQLite transport (read and write), linking the system library the
//! same way tools/nextgen_hydrofabric does. Startup/prep work only.
use crate::Result;
use std::ffi::{c_char, c_int, c_void, CStr, CString};

// Static link off Linux and Windows (unchanged); on Linux the library is opened at run time so the
// binary carries no libsqlite3.so.0 DT_NEEDED (manylinux_2_28 policy), and on Windows so the build
// needs no sqlite3.lib and the binary no sqlite3.dll import. Call sites are unchanged: each
// `sqlite3_*` below is a thin wrapper over the table `dyn_sqlite::api()` resolves (dlsym /
// GetProcAddress).
#[cfg(not(any(target_os = "linux", windows)))]
#[link(name = "sqlite3")]
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
#[cfg(any(target_os = "linux", windows))]
mod dyn_sqlite {
    use std::ffi::{c_char, c_int, c_void};
    use std::sync::OnceLock;

    // The platform loader: dlopen/dlsym on Linux, LoadLibraryA/GetProcAddress (kernel32) on Windows.
    #[cfg(target_os = "linux")]
    mod sys {
        use std::ffi::{c_char, c_int, c_void, CStr};
        #[link(name = "dl")]
        extern "C" {
            fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
            fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
            fn dlerror() -> *const c_char;
        }
        const RTLD_NOW: c_int = 2;
        pub unsafe fn open(name: &[u8]) -> *mut c_void {
            dlopen(name.as_ptr().cast(), RTLD_NOW)
        }
        pub unsafe fn sym(handle: *mut c_void, name: *const c_char) -> *mut c_void {
            dlsym(handle, name)
        }
        pub unsafe fn last_error() -> String {
            let p = dlerror();
            if p.is_null() { String::new() } else { CStr::from_ptr(p).to_string_lossy().into_owned() }
        }
    }
    #[cfg(windows)]
    mod sys {
        use std::ffi::{c_char, c_void};
        #[link(name = "kernel32")]
        extern "system" {
            fn LoadLibraryA(name: *const c_char) -> *mut c_void;
            fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
            fn GetLastError() -> u32;
        }
        pub unsafe fn open(name: &[u8]) -> *mut c_void {
            LoadLibraryA(name.as_ptr().cast())
        }
        pub unsafe fn sym(handle: *mut c_void, name: *const c_char) -> *mut c_void {
            GetProcAddress(handle, name)
        }
        pub unsafe fn last_error() -> String {
            match GetLastError() {
                126 => "Windows error 126: the module or one of its dependent DLLs was not found".into(),
                127 => "Windows error 127: the procedure was not found".into(),
                193 => "Windows error 193: the DLL is not a valid 64-bit Windows library".into(),
                code => format!("Windows error {code}"),
            }
        }
    }
    #[cfg(target_os = "linux")]
    const SONAMES: [&[u8]; 2] = [b"libsqlite3.so.0\0", b"libsqlite3.so\0"];
    #[cfg(windows)]
    const SONAMES: [&[u8]; 2] = [b"sqlite3.dll\0", b"winsqlite3.dll\0"];

    #[allow(non_snake_case)]
    pub struct Api {
        pub sqlite3_open_v2: unsafe extern "C" fn(*const c_char, *mut *mut c_void, c_int, *const c_char) -> c_int,
        pub sqlite3_close_v2: unsafe extern "C" fn(*mut c_void) -> c_int,
        pub sqlite3_exec: unsafe extern "C" fn(*mut c_void, *const c_char, *const c_void, *mut c_void, *mut *mut c_char) -> c_int,
        pub sqlite3_free: unsafe extern "C" fn(*mut c_void),
        pub sqlite3_prepare_v2: unsafe extern "C" fn(*mut c_void, *const c_char, c_int, *mut *mut c_void, *mut *const c_char) -> c_int,
        pub sqlite3_step: unsafe extern "C" fn(*mut c_void) -> c_int,
        pub sqlite3_reset: unsafe extern "C" fn(*mut c_void) -> c_int,
        pub sqlite3_clear_bindings: unsafe extern "C" fn(*mut c_void) -> c_int,
        pub sqlite3_finalize: unsafe extern "C" fn(*mut c_void) -> c_int,
        pub sqlite3_errmsg: unsafe extern "C" fn(*mut c_void) -> *const c_char,
        pub sqlite3_column_count: unsafe extern "C" fn(*mut c_void) -> c_int,
        pub sqlite3_column_name: unsafe extern "C" fn(*mut c_void, c_int) -> *const c_char,
        pub sqlite3_column_type: unsafe extern "C" fn(*mut c_void, c_int) -> c_int,
        pub sqlite3_column_int64: unsafe extern "C" fn(*mut c_void, c_int) -> i64,
        pub sqlite3_column_double: unsafe extern "C" fn(*mut c_void, c_int) -> f64,
        pub sqlite3_column_text: unsafe extern "C" fn(*mut c_void, c_int) -> *const u8,
        pub sqlite3_column_blob: unsafe extern "C" fn(*mut c_void, c_int) -> *const c_void,
        pub sqlite3_column_bytes: unsafe extern "C" fn(*mut c_void, c_int) -> c_int,
        pub sqlite3_bind_int64: unsafe extern "C" fn(*mut c_void, c_int, i64) -> c_int,
        pub sqlite3_bind_double: unsafe extern "C" fn(*mut c_void, c_int, f64) -> c_int,
        pub sqlite3_bind_text: unsafe extern "C" fn(*mut c_void, c_int, *const c_char, c_int, isize) -> c_int,
        pub sqlite3_bind_blob: unsafe extern "C" fn(*mut c_void, c_int, *const c_void, c_int, isize) -> c_int,
        pub sqlite3_bind_null: unsafe extern "C" fn(*mut c_void, c_int) -> c_int,
    }

    static API: OnceLock<Result<Api, String>> = OnceLock::new();

    unsafe fn message() -> String {
        sys::last_error()
    }

    unsafe fn load() -> Result<Api, String> {
        let mut handle = std::ptr::null_mut();
        for soname in SONAMES {
            handle = sys::open(soname);
            if !handle.is_null() {
                break;
            }
        }
        if handle.is_null() {
            #[cfg(target_os = "linux")]
            return Err(format!(
                "the system SQLite 3 library (libsqlite3.so.0) is not installed; reading or writing a GeoPackage hydrofabric needs it (install the sqlite3 / libsqlite3 package of your distribution): {}",
                message()
            ));
            #[cfg(windows)]
            return Err(format!(
                "the SQLite 3 library (sqlite3.dll or winsqlite3.dll) was not found; reading or writing a GeoPackage hydrofabric needs it (put sqlite3.dll from sqlite.org, OSGeo4W or conda-forge on PATH or next to this program): {}",
                message()
            ));
        }
        macro_rules! sym {
            ($name:literal) => {{
                let p = sys::sym(handle, concat!($name, "\0").as_ptr().cast());
                if p.is_null() {
                    #[cfg(target_os = "linux")]
                    return Err(format!("libsqlite3.so.0 has no symbol {}: {}", $name, message()));
                    #[cfg(windows)]
                    return Err(format!("the SQLite 3 DLL has no symbol {}: {}", $name, message()));
                }
                std::mem::transmute(p)
            }};
        }
        Ok(Api {
            sqlite3_open_v2: sym!("sqlite3_open_v2"),
            sqlite3_close_v2: sym!("sqlite3_close_v2"),
            sqlite3_exec: sym!("sqlite3_exec"),
            sqlite3_free: sym!("sqlite3_free"),
            sqlite3_prepare_v2: sym!("sqlite3_prepare_v2"),
            sqlite3_step: sym!("sqlite3_step"),
            sqlite3_reset: sym!("sqlite3_reset"),
            sqlite3_clear_bindings: sym!("sqlite3_clear_bindings"),
            sqlite3_finalize: sym!("sqlite3_finalize"),
            sqlite3_errmsg: sym!("sqlite3_errmsg"),
            sqlite3_column_count: sym!("sqlite3_column_count"),
            sqlite3_column_name: sym!("sqlite3_column_name"),
            sqlite3_column_type: sym!("sqlite3_column_type"),
            sqlite3_column_int64: sym!("sqlite3_column_int64"),
            sqlite3_column_double: sym!("sqlite3_column_double"),
            sqlite3_column_text: sym!("sqlite3_column_text"),
            sqlite3_column_blob: sym!("sqlite3_column_blob"),
            sqlite3_column_bytes: sym!("sqlite3_column_bytes"),
            sqlite3_bind_int64: sym!("sqlite3_bind_int64"),
            sqlite3_bind_double: sym!("sqlite3_bind_double"),
            sqlite3_bind_text: sym!("sqlite3_bind_text"),
            sqlite3_bind_blob: sym!("sqlite3_bind_blob"),
            sqlite3_bind_null: sym!("sqlite3_bind_null"),
        })
    }

    /// The function table, loaded once; the error names what is missing.
    pub fn api() -> Result<&'static Api, String> {
        API.get_or_init(|| unsafe { load() }).as_ref().map_err(Clone::clone)
    }

    /// After a successful `api()`; only reached through an open connection.
    pub fn table() -> &'static Api {
        match api() {
            Ok(t) => t,
            Err(e) => panic!("SQLite was used before it was loaded: {e}"),
        }
    }
}

#[cfg(any(target_os = "linux", windows))]
#[inline]
unsafe fn sqlite3_open_v2(p: *const c_char, db: *mut *mut c_void, flags: c_int, vfs: *const c_char) -> c_int {
    (dyn_sqlite::table().sqlite3_open_v2)(p, db, flags, vfs)
}
#[cfg(any(target_os = "linux", windows))]
#[inline]
unsafe fn sqlite3_close_v2(db: *mut c_void) -> c_int {
    (dyn_sqlite::table().sqlite3_close_v2)(db)
}
#[cfg(any(target_os = "linux", windows))]
#[inline]
unsafe fn sqlite3_exec(db: *mut c_void, sql: *const c_char, cb: *const c_void, arg: *mut c_void, err: *mut *mut c_char) -> c_int {
    (dyn_sqlite::table().sqlite3_exec)(db, sql, cb, arg, err)
}
#[cfg(any(target_os = "linux", windows))]
#[inline]
unsafe fn sqlite3_free(p: *mut c_void) {
    (dyn_sqlite::table().sqlite3_free)(p)
}
#[cfg(any(target_os = "linux", windows))]
#[inline]
unsafe fn sqlite3_prepare_v2(db: *mut c_void, sql: *const c_char, n: c_int, s: *mut *mut c_void, tail: *mut *const c_char) -> c_int {
    (dyn_sqlite::table().sqlite3_prepare_v2)(db, sql, n, s, tail)
}
#[cfg(any(target_os = "linux", windows))]
#[inline]
unsafe fn sqlite3_step(s: *mut c_void) -> c_int {
    (dyn_sqlite::table().sqlite3_step)(s)
}
#[cfg(any(target_os = "linux", windows))]
#[inline]
unsafe fn sqlite3_reset(s: *mut c_void) -> c_int {
    (dyn_sqlite::table().sqlite3_reset)(s)
}
#[cfg(any(target_os = "linux", windows))]
#[inline]
unsafe fn sqlite3_clear_bindings(s: *mut c_void) -> c_int {
    (dyn_sqlite::table().sqlite3_clear_bindings)(s)
}
#[cfg(any(target_os = "linux", windows))]
#[inline]
unsafe fn sqlite3_finalize(s: *mut c_void) -> c_int {
    (dyn_sqlite::table().sqlite3_finalize)(s)
}
#[cfg(any(target_os = "linux", windows))]
#[inline]
unsafe fn sqlite3_errmsg(db: *mut c_void) -> *const c_char {
    (dyn_sqlite::table().sqlite3_errmsg)(db)
}
#[cfg(any(target_os = "linux", windows))]
#[inline]
unsafe fn sqlite3_column_count(s: *mut c_void) -> c_int {
    (dyn_sqlite::table().sqlite3_column_count)(s)
}
#[cfg(any(target_os = "linux", windows))]
#[inline]
unsafe fn sqlite3_column_name(s: *mut c_void, i: c_int) -> *const c_char {
    (dyn_sqlite::table().sqlite3_column_name)(s, i)
}
#[cfg(any(target_os = "linux", windows))]
#[inline]
unsafe fn sqlite3_column_type(s: *mut c_void, i: c_int) -> c_int {
    (dyn_sqlite::table().sqlite3_column_type)(s, i)
}
#[cfg(any(target_os = "linux", windows))]
#[inline]
unsafe fn sqlite3_column_int64(s: *mut c_void, i: c_int) -> i64 {
    (dyn_sqlite::table().sqlite3_column_int64)(s, i)
}
#[cfg(any(target_os = "linux", windows))]
#[inline]
unsafe fn sqlite3_column_double(s: *mut c_void, i: c_int) -> f64 {
    (dyn_sqlite::table().sqlite3_column_double)(s, i)
}
#[cfg(any(target_os = "linux", windows))]
#[inline]
unsafe fn sqlite3_column_text(s: *mut c_void, i: c_int) -> *const u8 {
    (dyn_sqlite::table().sqlite3_column_text)(s, i)
}
#[cfg(any(target_os = "linux", windows))]
#[inline]
unsafe fn sqlite3_column_blob(s: *mut c_void, i: c_int) -> *const c_void {
    (dyn_sqlite::table().sqlite3_column_blob)(s, i)
}
#[cfg(any(target_os = "linux", windows))]
#[inline]
unsafe fn sqlite3_column_bytes(s: *mut c_void, i: c_int) -> c_int {
    (dyn_sqlite::table().sqlite3_column_bytes)(s, i)
}
#[cfg(any(target_os = "linux", windows))]
#[inline]
unsafe fn sqlite3_bind_int64(s: *mut c_void, i: c_int, v: i64) -> c_int {
    (dyn_sqlite::table().sqlite3_bind_int64)(s, i, v)
}
#[cfg(any(target_os = "linux", windows))]
#[inline]
unsafe fn sqlite3_bind_double(s: *mut c_void, i: c_int, v: f64) -> c_int {
    (dyn_sqlite::table().sqlite3_bind_double)(s, i, v)
}
#[cfg(any(target_os = "linux", windows))]
#[inline]
unsafe fn sqlite3_bind_text(s: *mut c_void, i: c_int, v: *const c_char, n: c_int, d: isize) -> c_int {
    (dyn_sqlite::table().sqlite3_bind_text)(s, i, v, n, d)
}
#[cfg(any(target_os = "linux", windows))]
#[inline]
unsafe fn sqlite3_bind_blob(s: *mut c_void, i: c_int, v: *const c_void, n: c_int, d: isize) -> c_int {
    (dyn_sqlite::table().sqlite3_bind_blob)(s, i, v, n, d)
}
#[cfg(any(target_os = "linux", windows))]
#[inline]
unsafe fn sqlite3_bind_null(s: *mut c_void, i: c_int) -> c_int {
    (dyn_sqlite::table().sqlite3_bind_null)(s, i)
}

/// Load the SQLite library before the first connection; a clear refusal when it is absent.
fn ensure_sqlite() -> Result<()> {
    #[cfg(any(target_os = "linux", windows))]
    dyn_sqlite::api().map_err(|e| crate::Error::from(e))?;
    Ok(())
}
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
        unsafe {
            let s = self.prepare(sql)?;
            let p = s.0;
            let columns = (0..sqlite3_column_count(p))
                .map(|i| CStr::from_ptr(sqlite3_column_name(p, i)).to_string_lossy().into_owned())
                .collect::<Vec<_>>();
            let mut rows = Vec::new();
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
                        3 => Value::Text(String::from_utf8(std::slice::from_raw_parts(sqlite3_column_text(p, i), n).to_vec())?),
                        4 => Value::Blob(if n == 0 { vec![] } else { std::slice::from_raw_parts(sqlite3_column_blob(p, i).cast::<u8>(), n).to_vec() }),
                        _ => Value::Null,
                    });
                }
                rows.push(row);
            }
            Ok(Rows { columns, rows })
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
