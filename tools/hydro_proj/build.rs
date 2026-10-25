use std::{env, fs, path::{Path, PathBuf}};
use sha2::{Digest, Sha256};

fn unpack(name: &str, digest: &str, out: &Path) {
    let path = Path::new("vendor").join(name);
    println!("cargo:rerun-if-changed={}", path.display());
    let bytes = fs::read(path).unwrap();
    assert_eq!(format!("{:x}", Sha256::digest(&bytes)), digest, "vendored source checksum");
    fs::create_dir_all(out).unwrap();
    tar::Archive::new(flate2::read::GzDecoder::new(bytes.as_slice())).unpack(out).unwrap();
}
fn main() {
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    unpack("proj-9.7.1.tar.gz", "6c097dc803c561929cdfcc46e4bf9945ea977611fb31493ad14e88edaeae260f", &out);
    unpack("sqlite-3.46.1.tar.gz", "4b1a04df3638bcf51960eeebca6305c60cbeab9846306b0b9c9a296e9a1b5b5e", &out.join("sqlite-3.46.1"));
    fs::write(out.join("sqlite-3.46.1/CMakeLists.txt"), r#"
cmake_minimum_required(VERSION 3.22)
project(hydro_sqlite C)
add_library(sqlite3 STATIC sqlite3.c)
set_target_properties(sqlite3 PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(sqlite3 PRIVATE SQLITE_THREADSAFE=1 SQLITE_OMIT_LOAD_EXTENSION SQLITE_SECURE_DELETE)
add_executable(sqlite3_shell shell.c)
target_compile_definitions(sqlite3_shell PRIVATE SQLITE_OMIT_LOAD_EXTENSION)
target_link_libraries(sqlite3_shell PRIVATE sqlite3)
if(UNIX)
  target_link_libraries(sqlite3_shell PRIVATE m pthread ${CMAKE_DL_LIBS})
endif()
install(TARGETS sqlite3 sqlite3_shell ARCHIVE DESTINATION lib RUNTIME DESTINATION bin)
install(FILES sqlite3.h DESTINATION include)
"#).unwrap();
    let windows = env::var("CARGO_CFG_TARGET_OS").unwrap() == "windows";
    let mut sqlite_config = cmake::Config::new(out.join("sqlite-3.46.1"));
    sqlite_config.out_dir(out.join("sqlite-build-3.46.1")).profile("Release");
    if windows {
        sqlite_config.static_crt(true).define("CMAKE_MSVC_RUNTIME_LIBRARY", "MultiThreaded");
    }
    let sqlite = sqlite_config.build();
    let mut proj = cmake::Config::new(out.join("proj-9.7.1"));
    if windows {
        proj.static_crt(true).define("CMAKE_MSVC_RUNTIME_LIBRARY", "MultiThreaded");
    }
    proj.out_dir(out.join("proj-build")).profile("Release")
        .define("BUILD_SHARED_LIBS", "OFF")
        // GNUInstallDirs puts the archive in lib64 on the manylinux_2_28 baseline (a RHEL family); the link search
        // below reads lib. Breakage it prevents: the release build of every hydro bridge failed "could not find
        // native static library `proj`" there, while an Ubuntu build (lib) linked.
        .define("CMAKE_INSTALL_LIBDIR", "lib")
        .define("BUILD_TESTING", "OFF")
        .define("BUILD_APPS", "OFF")
        .define("ENABLE_CURL", "OFF")
        .define("ENABLE_TIFF", "OFF")
        .define("EMBED_RESOURCE_FILES", "ON")
        .define("USE_ONLY_EMBEDDED_RESOURCE_FILES", "ON")
        .define("EMBED_PROJ_DATA_PATH", "OFF")
        .define("CMAKE_POSITION_INDEPENDENT_CODE", "ON")
        .define("PROJ_OBJECT_LIBRARIES_POSITION_INDEPENDENT_CODE", "ON")
        .define("SQLite3_INCLUDE_DIR", sqlite.join("include"))
        .define("SQLite3_LIBRARY", sqlite.join(if windows {"lib/sqlite3.lib"} else {"lib/libsqlite3.a"}))
        .define("EXE_SQLITE3", sqlite.join(if windows {"bin/sqlite3_shell.exe"} else {"bin/sqlite3_shell"}));
    let built = proj.build();
    println!("cargo:rustc-link-search=native={}", built.join("lib").display());
    println!("cargo:rustc-link-search=native={}", sqlite.join("lib").display());
    println!("cargo:rustc-link-lib=static=proj");
    println!("cargo:rustc-link-lib=static=sqlite3");
    if windows {
        println!("cargo:rustc-link-lib=shell32");
        println!("cargo:rustc-link-lib=ole32");
    } else {
        println!("cargo:rustc-link-lib={}", if env::var("CARGO_CFG_TARGET_OS").unwrap() == "macos" {"c++"} else {"stdc++"});
        println!("cargo:rustc-link-lib=m");
        println!("cargo:rustc-link-lib=pthread");
        println!("cargo:rustc-link-lib=dl");
    }
}
