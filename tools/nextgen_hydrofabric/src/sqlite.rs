//! Read-only SQLite transport, with one connection per worker.
use crate::Result;
use serde_json::{Map, Value};
use std::ffi::{c_char, c_int, c_void, CStr, CString};
// Static link off Linux and Windows (unchanged); on Linux the library is opened at run time so the
// binary carries no libsqlite3.so.0 DT_NEEDED (manylinux_2_28 policy), and on Windows so the build
// needs no sqlite3.lib and the binary no sqlite3.dll import. Call sites are unchanged: each
// `sqlite3_*` below is a thin wrapper over the table `dyn_sqlite::api()` resolves (dlsym /
// GetProcAddress).
#[cfg(not(any(target_os = "linux", windows)))]
#[link(name = "sqlite3")]
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
        pub sqlite3_prepare_v2: unsafe extern "C" fn(*mut c_void, *const c_char, c_int, *mut *mut c_void, *mut *const c_char) -> c_int,
        pub sqlite3_step: unsafe extern "C" fn(*mut c_void) -> c_int,
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
            sqlite3_prepare_v2: sym!("sqlite3_prepare_v2"),
            sqlite3_step: sym!("sqlite3_step"),
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

/// Load the SQLite library before the first connection; a clear refusal when it is absent.
fn ensure_sqlite() -> Result<()> {
    #[cfg(any(target_os = "linux", windows))]
    dyn_sqlite::api().map_err(|e| crate::Error::from(e))?;
    Ok(())
}
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
